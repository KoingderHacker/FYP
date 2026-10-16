import os
import numpy as np
import pandas as pd
from .config import MODEL_NAME, DATETIME_COL, TARGET, CONTEXT_LENGTH, HORIZONS, QUANTILE_LEVELS, FEATURES

HOLIDAY_DATES = set([
"2024-01-26","2024-03-08","2024-03-25","2024-04-09","2024-04-11","2024-04-17","2024-05-23","2024-06-17","2024-07-17","2024-08-15","2024-08-26","2024-09-07","2024-09-16","2024-10-02","2024-10-12","2024-10-31","2024-11-01","2024-11-15","2024-12-25",
"2025-01-26","2025-02-26","2025-03-14","2025-03-31","2025-04-06","2025-04-10","2025-04-14","2025-05-12","2025-06-07","2025-08-15","2025-08-16","2025-08-27","2025-09-05","2025-10-02","2025-10-20","2025-10-21","2025-11-05","2025-12-25",
"2026-01-26","2026-02-15","2026-03-04","2026-03-21","2026-03-27","2026-04-03","2026-04-14","2026-05-27","2026-06-26","2026-08-15","2026-08-26","2026-09-04","2026-10-02","2026-10-20","2026-11-08","2026-11-09","2026-11-10","2026-11-11","2026-12-25"])
FESTIVAL_DATES = set([
"2024-03-08","2024-03-25","2024-08-26","2024-09-07","2024-10-12","2024-10-31","2024-11-01",
"2025-03-14","2025-03-31","2025-08-16","2025-08-27","2025-10-20","2025-10-21","2025-11-05",
"2026-02-15","2026-03-04","2026-03-21","2026-08-26","2026-10-20","2026-11-08","2026-11-09","2026-11-10","2026-11-11"])

def create_calendar_features(df):
    df = df.copy()
    df[DATETIME_COL] = pd.to_datetime(df[DATETIME_COL], errors="coerce")
    if df[DATETIME_COL].isna().any(): raise ValueError("Invalid datetime values found.")
    df["hour"] = df[DATETIME_COL].dt.hour
    df["day_of_week"] = df[DATETIME_COL].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["month"] = df[DATETIME_COL].dt.month
    df["day_of_year"] = df[DATETIME_COL].dt.dayofyear
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["day_of_year_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
    df["day_of_year_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)
    dates = df[DATETIME_COL].dt.strftime("%Y-%m-%d")
    df["is_holiday"] = dates.isin(HOLIDAY_DATES).astype(int)
    df["is_festival"] = dates.isin(FESTIVAL_DATES).astype(int)
    return df

def validate_dataset(df):
    required = [DATETIME_COL, TARGET]
    missing = [c for c in required if c not in df.columns]
    if missing: raise ValueError(f"Missing required columns: {missing}")
    df = df.copy()
    df[DATETIME_COL] = pd.to_datetime(df[DATETIME_COL], errors="coerce")
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=[DATETIME_COL, TARGET])
    df = df.sort_values(DATETIME_COL).drop_duplicates(subset=[DATETIME_COL], keep="last").reset_index(drop=True)
    if df.empty: raise ValueError("Demand dataset contains no valid rows.")
    if (df[TARGET] < 0).any(): raise ValueError("Negative demand values found.")
    return df

def load_model():
    import torch  # lazy: only the legacy Chronos-2 engine needs torch/chronos
    from chronos import Chronos2Pipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = Chronos2Pipeline.from_pretrained(MODEL_NAME, device_map=device, torch_dtype=torch.float32 if device == "cpu" else torch.float16)
    return pipeline, device

def build_future_dataframe(last_timestamp, horizon):
    timestamps = pd.date_range(start=last_timestamp + pd.Timedelta(hours=1), periods=horizon, freq="h")
    return create_calendar_features(pd.DataFrame({DATETIME_COL: timestamps}))

def forecast_demand(horizon="1d", pipeline=None, data=None, dataset_path=None, forecast_origin=None):
    if horizon not in HORIZONS: raise ValueError(f"Invalid horizon '{horizon}'. Choose from {list(HORIZONS.keys())}.")
    prediction_length = HORIZONS[horizon]
    if data is None:
        if not dataset_path: raise ValueError("Demand dataset path is required.")
        if not os.path.exists(dataset_path): raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        data = pd.read_csv(dataset_path)
    data = create_calendar_features(validate_dataset(data))
    if forecast_origin is not None:
        origin = pd.to_datetime(forecast_origin, errors="coerce")
        if pd.isna(origin): raise ValueError(f"Invalid forecast_origin: {forecast_origin}")
        data = data[data[DATETIME_COL] <= origin].copy()
        if data.empty: raise ValueError(f"No historical data available at or before {origin}.")
    if len(data) < CONTEXT_LENGTH: raise ValueError(f"Need at least {CONTEXT_LENGTH} historical hours, but only {len(data)} are available.")
    context = data.tail(CONTEXT_LENGTH).copy()
    historical_df = context[[DATETIME_COL, TARGET] + FEATURES].copy()
    historical_df["item_id"] = "gujarat_demand"
    historical_df = historical_df[["item_id", DATETIME_COL, TARGET] + FEATURES]
    future_df = build_future_dataframe(context[DATETIME_COL].iloc[-1], prediction_length)
    future_df["item_id"] = "gujarat_demand"
    future_df = future_df[["item_id", DATETIME_COL] + FEATURES]
    if pipeline is None: pipeline, device = load_model()
    else:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    import torch
    with torch.inference_mode():
        predictions = pipeline.predict_df(historical_df, future_df=future_df, id_column="item_id", timestamp_column=DATETIME_COL, target=TARGET, prediction_length=prediction_length, quantile_levels=QUANTILE_LEVELS, context_length=CONTEXT_LENGTH, freq="h")
    forecast_df = predictions.copy()
    if "0.5" not in forecast_df.columns: raise RuntimeError("Chronos-2 output does not contain the expected 0.5 quantile.")
    forecast_df = forecast_df.rename(columns={"0.1":"lower_10", "0.5":"forecast", "0.9":"upper_90"})
    keep = [DATETIME_COL, "forecast", "lower_10", "upper_90"]
    forecast_df = forecast_df[[c for c in keep if c in forecast_df.columns]].copy()
    forecast_df[DATETIME_COL] = pd.to_datetime(forecast_df[DATETIME_COL])
    if forecast_df.empty: raise RuntimeError("Chronos-2 returned an empty forecast.")
    if forecast_df["forecast"].isna().any(): raise RuntimeError("Forecast contains NaN values.")
    for col in ["forecast", "lower_10", "upper_90"]:
        if col in forecast_df.columns: forecast_df[col] = forecast_df[col].clip(lower=0)
    result = {
        "model": MODEL_NAME, "device": device, "target": TARGET,
        "forecast_origin": pd.to_datetime(forecast_origin).isoformat() if forecast_origin is not None else None,
        "horizon": horizon, "horizon_hours": prediction_length, "context_hours": CONTEXT_LENGTH,
        "unit": "MU", "prediction_quantile": 0.5, "feature_set": FEATURES,
        "predictions": [{"datetime": r[DATETIME_COL].strftime("%Y-%m-%d %H:%M:%S"), "forecast": round(float(r["forecast"]),6), "lower_10": round(float(r["lower_10"]),6), "upper_90": round(float(r["upper_90"]),6)} for _, r in forecast_df.iterrows()],
        "status": "success"
    }
    return result