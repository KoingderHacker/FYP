"""Train, store and run the weather-only solar / wind / demand models.

Design
------
* One model per target (solar, wind, demand). The model is independent of the
  forecast horizon: it maps the weather and calendar of a *target hour* to
  generation / demand for that hour. A 1-hour, 24-hour, 7-day or 15-day
  forecast is just "run the same model over the weather forecast for those
  hours", so no recent generation measurement is ever needed.
* Each target has three gradient-boosted regressors: mean (point forecast),
  q10 and q90 (uncertainty band).
* Installed capacity / load grows over time. A multiplicative linear *level*
  (fitted on training data with month dummies so seasons do not leak into the
  slope) is divided out before training and multiplied back at prediction.
  The level is frozen MAX_EXTRAPOLATION_DAYS after the end of the training
  data so that a stale model never extrapolates growth indefinitely.
* Honest evaluation: rows with sentinel irradiance (-999) or imputed targets
  (generation_source != "true_CEA") are excluded, selection uses a validation
  block, and the test block is scored once.
"""

from __future__ import annotations

import json
import logging
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .features import (
    DATETIME_COL,
    build_features,
    mask_sentinels,
    naive_local_datetime,
    required_weather_columns,
    trend_days,
)

logger = logging.getLogger(__name__)

MODEL_DIR = Path("models/weather_only")
REPORT_PATH = Path("reports/weather_only/metrics.json")
DEFAULT_WEATHER_CSV = "data/wind/raw/gujarat_hourly_weather_generation_final.csv"
DEFAULT_DEMAND_CSV = "data/demand/raw/gujarat_hourly_demand_estimated.csv"

TARGET_COLUMNS = {
    "solar": "Gujarat_Solar_MU_hourly_est",
    "wind": "Gujarat_Wind_MU_hourly_est",
    "demand": "Energy_Met_MU_hourly_est",
}

# Do not let the growth level extrapolate further than this past the end of
# the training data (days).
MAX_EXTRAPOLATION_DAYS = 120

HYPERPARAMETER_GRID = [
    {"max_iter": 300, "max_leaf_nodes": 31, "min_samples_leaf": 20},
    {"max_iter": 300, "max_leaf_nodes": 15, "min_samples_leaf": 40},
    {"max_iter": 600, "max_leaf_nodes": 15, "min_samples_leaf": 40},
]
LEARNING_RATE = 0.05
SEED = 0


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
def load_training_frame(
    target: str,
    weather_csv: str = DEFAULT_WEATHER_CSV,
    demand_csv: str = DEFAULT_DEMAND_CSV,
) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    """Return (weather rows, target series, data-quality summary).

    Invalid rows are removed, never imputed.
    """
    if target not in TARGET_COLUMNS:
        raise ValueError(f"Unknown target '{target}'.")
    weather = pd.read_csv(weather_csv)
    weather[DATETIME_COL] = pd.to_datetime(weather[DATETIME_COL])
    weather = mask_sentinels(weather)
    total = len(weather)
    column = TARGET_COLUMNS[target]

    if target == "demand":
        demand = pd.read_csv(demand_csv, usecols=[DATETIME_COL, column])
        demand[DATETIME_COL] = pd.to_datetime(demand[DATETIME_COL])
        frame = weather.drop(columns=[c for c in [column] if c in weather.columns])
        frame = frame.merge(demand, on=DATETIME_COL, how="inner")
    else:
        frame = weather

    keep = frame[column].notna()
    removed = {"missing_target": int((~keep).sum())}
    if target in {"solar", "wind"} and "generation_source" in frame.columns:
        imputed = frame["generation_source"] != "true_CEA"
        removed["imputed_target"] = int((keep & imputed).sum())
        keep &= ~imputed
    needed = required_weather_columns(target)
    bad_weather = frame[needed].isna().any(axis=1)
    removed["invalid_weather"] = int((keep & bad_weather).sum())
    keep &= ~bad_weather

    frame = frame[keep].sort_values(DATETIME_COL).reset_index(drop=True)
    summary = {
        "rows_total": total,
        "rows_used": int(len(frame)),
        "rows_removed": removed,
        "data_start": frame[DATETIME_COL].iloc[0].isoformat(),
        "data_end": frame[DATETIME_COL].iloc[-1].isoformat(),
    }
    return frame, frame[column].astype(float), summary


# --------------------------------------------------------------------------
# Growth level
# --------------------------------------------------------------------------
def _month_dummies(datetimes: pd.Series) -> np.ndarray:
    months = naive_local_datetime(pd.Series(datetimes)).dt.month.to_numpy()
    return (months[:, None] == np.arange(1, 13)[None, :]).astype(float)


def fit_level(datetimes: pd.Series, y: np.ndarray) -> dict[str, float]:
    """Fit y ~ slope * years + month effects; return a normalised level."""
    t = trend_days(datetimes)
    design = np.column_stack([t / 365.0, _month_dummies(datetimes)])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    base = float(np.mean(y))
    return {
        "slope_per_year": float(coef[0]),
        "base": base,
        "t_center": float(np.mean(t)),
        "t_end": float(np.max(t)),
    }


def level_at(datetimes: pd.Series, params: dict[str, float]) -> np.ndarray:
    t = trend_days(datetimes)
    t = np.minimum(t, params["t_end"] + MAX_EXTRAPOLATION_DAYS)
    level = 1.0 + params["slope_per_year"] * (t - params["t_center"]) / 365.0 / params["base"]
    return np.clip(level, 0.2, None)


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
def _regressor(params: dict[str, Any], **extra: Any) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        learning_rate=LEARNING_RATE, random_state=SEED, **params, **extra
    )


def _metrics(actual: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(actual, pred)),
        "rmse": float(np.sqrt(mean_squared_error(actual, pred))),
        "r2": float(r2_score(actual, pred)),
    }


def _postprocess(target: str, point: np.ndarray, low: np.ndarray, high: np.ndarray,
                 weather: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    point = np.clip(point, 0.0, None)
    low = np.clip(np.minimum(low, point), 0.0, None)
    high = np.maximum(high, point)
    if target == "solar":
        night = (weather["CLRSKY_SFC_SW_DWN"].to_numpy() <= 1.0) | (
            weather["ALLSKY_SFC_SW_DWN"].to_numpy() <= 0.0
        )
        point = np.where(night, 0.0, point)
        low = np.where(night, 0.0, low)
        high = np.where(night, 0.0, high)
    return point, low, high


def _fit_models(X: pd.DataFrame, y_norm: np.ndarray, params: dict[str, Any]) -> dict[str, Any]:
    return {
        "point": _regressor(params).fit(X, y_norm),
        "q10": _regressor(params, loss="quantile", quantile=0.1).fit(X, y_norm),
        "q90": _regressor(params, loss="quantile", quantile=0.9).fit(X, y_norm),
    }


def _predict_raw(models: dict[str, Any], X: pd.DataFrame, level: np.ndarray,
                 target: str, weather: pd.DataFrame, delta: float = 0.0):
    """Point + band. ``delta`` is the conformal widening (MU) added to q10/q90."""
    point = models["point"].predict(X) * level
    low = models["q10"].predict(X) * level - delta
    high = models["q90"].predict(X) * level + delta
    return _postprocess(target, point, low, high, weather)


def _is_active(target: str, weather: pd.DataFrame) -> np.ndarray:
    """Rows where the forecast is not forced to 0 (solar: daylight hours)."""
    if target == "solar":
        return (weather["CLRSKY_SFC_SW_DWN"].to_numpy() > 1.0) & (
            weather["ALLSKY_SFC_SW_DWN"].to_numpy() > 0.0
        )
    return np.ones(len(weather), dtype=bool)


def conformal_delta(y: np.ndarray, low: np.ndarray, high: np.ndarray,
                    active: np.ndarray, coverage: float = 0.8) -> float:
    """Conformalised-quantile-regression widening (Romano et al., 2019).

    Raw quantile regressors are too narrow on unseen periods. The widening is
    the (coverage)-quantile of how far held-out values fall outside [q10, q90].
    """
    score = np.maximum(low - y, y - high)[active]
    if len(score) == 0:
        return 0.0
    k = min(1.0, coverage * (1.0 + 1.0 / len(score)))
    return float(max(0.0, np.quantile(score, k)))


def train_target(
    target: str,
    weather_csv: str = DEFAULT_WEATHER_CSV,
    demand_csv: str = DEFAULT_DEMAND_CSV,
    model_dir: Path | str = MODEL_DIR,
) -> dict[str, Any]:
    """Train + evaluate one target and write ``<model_dir>/<target>.joblib``."""
    started = time.time()
    frame, y_series, quality = load_training_frame(target, weather_csv, demand_csv)
    y = y_series.to_numpy()
    n = len(frame)
    if n < 2000:
        raise ValueError(f"Only {n} valid training rows for '{target}'; need at least 2000.")
    a, b = int(n * 0.70), int(n * 0.85)
    X = build_features(target, frame)
    dt = frame[DATETIME_COL]

    # --- honest evaluation: level + models fitted on the training block only
    level_params = fit_level(dt.iloc[:a], y[:a])
    level = level_at(dt, level_params)
    best, best_val = None, np.inf
    grid_log = []
    for params in HYPERPARAMETER_GRID:
        model = _regressor(params).fit(X.iloc[:a], y[:a] / level[:a])
        val_pred = model.predict(X.iloc[a:b]) * level[a:b]
        val_pred, _, _ = _postprocess(target, val_pred, val_pred, val_pred, frame.iloc[a:b])
        val_mae = float(mean_absolute_error(y[a:b], val_pred))
        grid_log.append({**params, "val_mae": val_mae})
        if val_mae < best_val:
            best, best_val = params, val_mae
    assert best is not None

    models = _fit_models(X.iloc[:a], y[:a] / level[:a], best)
    # Calibrate the q10-q90 band on the validation block (never on test).
    raw_low, raw_high = (models[k].predict(X.iloc[a:b]) * level[a:b] for k in ("q10", "q90"))
    delta = conformal_delta(y[a:b], raw_low, raw_high, _is_active(target, frame.iloc[a:b]))
    scores: dict[str, Any] = {"band_widening_mu": delta}
    for name, sl in (("validation", slice(a, b)), ("test", slice(b, n))):
        point, low, high = _predict_raw(models, X.iloc[sl], level[sl], target,
                                        frame.iloc[sl], delta)
        scores[name] = {
            **_metrics(y[sl], point),
            "rows": int(len(point)),
            "start": dt.iloc[sl.start].isoformat(),
            "end": dt.iloc[sl.stop - 1].isoformat(),
            "q10_q90_coverage": float(np.mean(((y[sl] >= low) & (y[sl] <= high))[
                _is_active(target, frame.iloc[sl])])),
        }
    # Reference: month x hour climatology (scaled by the same level).
    clim_key = dt.dt.month * 100 + dt.dt.hour
    clim = pd.Series(y[:a] / level[:a]).groupby(clim_key.iloc[:a].to_numpy()).mean()
    clim_pred = clim.reindex(clim_key.iloc[b:]).fillna(clim.mean()).to_numpy() * level[b:]
    scores["test_climatology_baseline"] = _metrics(y[b:], clim_pred)

    # --- deployment models: refit on every valid row with the chosen settings
    level_params_final = fit_level(dt, y)
    level_final = level_at(dt, level_params_final)
    final_models = _fit_models(X, y / level_final, best)

    artifact = {
        "target": target,
        "feature_names": list(X.columns),
        "models": final_models,
        "band_delta": delta,
        "level": level_params_final,
        "hyperparameters": best,
        "trained_through": dt.iloc[-1].isoformat(),
        "trained_at_unix": time.time(),
        "data_quality": quality,
        "scores_observed_weather_backtest": scores,
        "hyperparameter_search": grid_log,
        "annual_growth_pct": 100.0 * level_params["slope_per_year"] / level_params["base"],
    }
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_dir / f"{target}.joblib")
    logger.info("Trained %s in %.1fs", target, time.time() - started)
    return {k: v for k, v in artifact.items() if k != "models"}


def train_all(**kwargs: Any) -> dict[str, Any]:
    summary = {t: train_target(t, **kwargs) for t in TARGET_COLUMNS}
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary


# --------------------------------------------------------------------------
# Loading + prediction
# --------------------------------------------------------------------------
@lru_cache(maxsize=8)
def _load_artifact(target: str, model_dir: str) -> dict[str, Any]:
    path = Path(model_dir) / f"{target}.joblib"
    if path.exists():
        try:
            return joblib.load(path)
        except Exception as exc:  # e.g. scikit-learn version change
            logger.warning("Could not load %s (%s); retraining.", path, exc)
    logger.warning("Weather-only model for '%s' not found; training it now.", target)
    train_target(target, model_dir=model_dir)
    return joblib.load(path)


def clear_model_cache() -> None:
    _load_artifact.cache_clear()


def model_info(target: str, model_dir: Path | str = MODEL_DIR) -> dict[str, Any]:
    art = _load_artifact(target, str(model_dir))
    return {
        "trained_through": art["trained_through"],
        "annual_growth_pct": art["annual_growth_pct"],
        "hyperparameters": art["hyperparameters"],
        "test_scores_observed_weather": art["scores_observed_weather_backtest"]["test"],
    }


def predict(target: str, weather: pd.DataFrame,
            model_dir: Path | str = MODEL_DIR) -> pd.DataFrame:
    """Predict from weather rows (``datetime`` + weather columns).

    Returns a frame with ``datetime`` (naive Asia/Kolkata), ``forecast``,
    ``lower_10`` and ``upper_90`` in MU, plus ``trend_extrapolated_days``.
    """
    art = _load_artifact(target, str(model_dir))
    needed = required_weather_columns(target)
    missing = [c for c in [DATETIME_COL, *needed] if c not in weather.columns]
    if missing:
        raise ValueError(f"Weather rows are missing required columns: {missing}")
    work = mask_sentinels(weather).reset_index(drop=True)
    work[DATETIME_COL] = naive_local_datetime(work[DATETIME_COL])
    invalid = work[needed].isna().any(axis=1)
    if invalid.any():
        first = work.loc[invalid, DATETIME_COL].iloc[0]
        raise ValueError(
            f"{int(invalid.sum())} weather rows have missing/sentinel values in "
            f"{needed} (first at {first}); refusing to forecast from invalid inputs."
        )
    if not np.isfinite(work[needed].to_numpy(dtype=float)).all():
        raise ValueError("Weather rows contain non-finite values.")

    X = build_features(target, work)[art["feature_names"]]
    level = level_at(work[DATETIME_COL], art["level"])
    point, low, high = _predict_raw(art["models"], X, level, target, work,
                                    art.get("band_delta", 0.0))
    over = np.maximum(trend_days(work[DATETIME_COL]) - art["level"]["t_end"], 0.0)
    return pd.DataFrame({
        DATETIME_COL: work[DATETIME_COL],
        "forecast": point,
        "lower_10": low,
        "upper_90": high,
        "trend_extrapolated_days": over,
    })
