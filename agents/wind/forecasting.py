import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .config import (
    CONTEXT_LENGTH, DATETIME_COL, FEATURES, FUTURE_COVARIATES,
    MODEL_NAME, PREDICTION_QUANTILE, QUANTILE_LEVELS, TARGET,
)


def _prepare_dataframe(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = [DATETIME_COL, TARGET] + [c for c in [
        "WS50M", "WS10M", "WD50M", "WD10M", "T2M", "PS", "RH2M", "PRECTOTCORR"
    ] if c not in df.columns]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df[DATETIME_COL] = pd.to_datetime(df[DATETIME_COL], errors="coerce")
    df = df.dropna(subset=[DATETIME_COL]).sort_values(DATETIME_COL).reset_index(drop=True)

    df["hour"] = df[DATETIME_COL].dt.hour
    df["day_of_year"] = df[DATETIME_COL].dt.dayofyear
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["day_of_year_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["day_of_year_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)

    df["wind_lag_1"] = df[TARGET].shift(1)
    df["wind_lag_24"] = df[TARGET].shift(24)
    df["wind_lag_168"] = df[TARGET].shift(168)
    previous_wind = df[TARGET].shift(1)
    df["wind_rolling_mean_24"] = previous_wind.rolling(24).mean()
    df["wind_rolling_mean_168"] = previous_wind.rolling(168).mean()

    df["WS50M_cube"] = df["WS50M"] ** 3
    df["WD50M_sin"] = np.sin(np.deg2rad(df["WD50M"]))
    df["WD50M_cos"] = np.cos(np.deg2rad(df["WD50M"]))

    missing_features = [c for c in FEATURES if c not in df.columns]
    if missing_features:
        raise ValueError(f"Missing wind features: {missing_features}")

    for col in FEATURES + [TARGET]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _prediction_column(forecast: pd.DataFrame) -> str:
    for col in forecast.columns:
        if str(col) in {"0.5", "0.5_quantile", "prediction"}:
            return col
    numeric = forecast.select_dtypes(include=np.number).columns.tolist()
    if not numeric:
        raise ValueError(f"Could not identify prediction column. Returned columns: {forecast.columns.tolist()}")
    return numeric[-1]


def forecast_once(dataset_path: str, horizon_hours: int, forecast_origin: str | None = None,
                  model_name: str = MODEL_NAME) -> dict:
    start = time.time()
    df = _prepare_dataframe(dataset_path)

    if forecast_origin:
        origin_time = pd.Timestamp(forecast_origin)
        matches = df.index[df[DATETIME_COL] == origin_time].tolist()
        if not matches:
            raise ValueError(f"forecast_origin {forecast_origin} was not found in the dataset.")
        origin = matches[0] + 1
        origin_label = origin_time.isoformat()
    else:
        origin = len(df)
        origin_label = str(df[DATETIME_COL].iloc[-1])

    if origin < CONTEXT_LENGTH:
        raise ValueError(f"Need at least {CONTEXT_LENGTH} historical rows before forecast origin; only {origin} are available.")
    if origin + horizon_hours > len(df):
        available = len(df) - origin
        raise ValueError(
            f"Need {horizon_hours} future weather rows after {origin_label}; only {available} are available. "
            "Provide a future weather-forecast dataset for live forecasting."
        )

    context = df.iloc[origin - CONTEXT_LENGTH:origin].copy()
    future = df.iloc[origin:origin + horizon_hours].copy()

    historical = context[[DATETIME_COL, TARGET] + FEATURES].copy()
    historical[DATETIME_COL] = pd.to_datetime(historical[DATETIME_COL])
    historical["series_id"] = "Gujarat_Wind"
    historical = historical[["series_id", DATETIME_COL, TARGET] + FEATURES]

    future_covariates = future[[DATETIME_COL] + FUTURE_COVARIATES].copy()
    future_covariates[DATETIME_COL] = pd.to_datetime(future_covariates[DATETIME_COL])
    future_covariates["series_id"] = "Gujarat_Wind"
    future_covariates = future_covariates[["series_id", DATETIME_COL] + FUTURE_COVARIATES]

    import torch  # lazy: only the legacy Chronos-2 engine needs torch/chronos
    from chronos import Chronos2Pipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = Chronos2Pipeline.from_pretrained(model_name, device_map=device)
    forecast = pipeline.predict_df(
        historical,
        future_df=future_covariates,
        target=TARGET,
        prediction_length=horizon_hours,
        quantile_levels=QUANTILE_LEVELS,
        id_column="series_id",
        timestamp_column=DATETIME_COL,
    )
    if not isinstance(forecast, pd.DataFrame):
        raise ValueError(f"Chronos-2 returned unexpected type: {type(forecast)}")

    predicted = forecast[_prediction_column(forecast)].to_numpy(dtype=float).reshape(-1)
    if len(predicted) != horizon_hours:
        raise ValueError(f"Chronos-2 returned {len(predicted)} predictions; expected {horizon_hours}.")

    actual = future[TARGET].to_numpy(dtype=float)
    metrics = {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)) if len(actual) >= 2 else None,
    }
    predictions = [
        {"datetime": str(ts), "predicted": float(pred), "actual": float(act)}
        for ts, pred, act in zip(future[DATETIME_COL], predicted, actual)
    ]
    return {
        "model": model_name,
        "device": device,
        "target": TARGET,
        "forecast_origin": origin_label,
        "horizon_hours": horizon_hours,
        "context_length": CONTEXT_LENGTH,
        "prediction_quantile": PREDICTION_QUANTILE,
        "future_covariates": FUTURE_COVARIATES,
        "metrics": metrics,
        "predictions": predictions,
        "execution_time_seconds": round(time.time() - start, 3),
    }