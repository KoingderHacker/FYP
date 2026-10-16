"""Agent-facing helpers for the weather-only forecasters.

Two input modes, same model:

* ``live_weather_forecast``  - ``data["future_weather"]`` holds the hourly
  weather forecast (Open-Meteo, from the Weather Agent). No CSV is read, so it
  works on any date regardless of how stale the historical files are.
* Historical evaluation is handled by ``common.weather_only.backtest``. It
  fits models using only pre-origin data instead of loading deployment models.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .features import DATETIME_COL, naive_local_datetime, required_weather_columns
from .models import (
    MODEL_DIR,
    TARGET_COLUMNS,
    model_info,
    predict,
)

LIVE = "live_weather_forecast"
KOLKATA = "Asia/Kolkata"
MODEL_LABEL = "weather_only_gradient_boosting"


def use_weather_only(config: dict[str, Any], data: dict[str, Any]) -> bool:
    """Engine routing: explicit ``config.engine`` wins; otherwise live weather
    in the request selects the weather-only engine and everything else keeps
    the legacy Chronos-2 path."""
    engine = (config or {}).get("engine")
    if engine == "chronos2":
        return False
    if engine == "weather_only":
        return True
    return (data or {}).get("future_weather") is not None


def _expected_index(first: pd.Timestamp, hours: int) -> pd.DatetimeIndex:
    return pd.date_range(start=first, periods=hours, freq="h")


def _live_weather(data: dict[str, Any], hours: int, forecast_origin: str | None) -> pd.DataFrame:
    if not forecast_origin:
        raise ValueError("Live forecasting requires data.forecast_origin.")
    origin = pd.Timestamp(forecast_origin)
    if origin.tzinfo is None:
        raise ValueError("Live forecast_origin must be timezone-aware (e.g. +05:30).")
    origin = origin.tz_convert(KOLKATA).tz_localize(None)

    weather = pd.DataFrame.from_records(data["future_weather"])
    if weather.empty:
        raise ValueError("data.future_weather is empty.")
    if DATETIME_COL not in weather.columns:
        raise ValueError("data.future_weather rows need a 'datetime' field.")
    weather[DATETIME_COL] = naive_local_datetime(weather[DATETIME_COL])
    weather = weather.sort_values(DATETIME_COL).reset_index(drop=True)

    expected = _expected_index(origin.floor("h") + pd.Timedelta(1, unit="h"), hours)
    weather = weather[weather[DATETIME_COL] > origin].head(hours).reset_index(drop=True)
    if len(weather) != hours or not pd.DatetimeIndex(weather[DATETIME_COL]).equals(expected):
        raise ValueError(
            "Weather forecast must cover each hour immediately after the forecast "
            f"origin for the requested {hours}-hour horizon."
        )
    return weather


def forecast_target(
    target: str,
    data: dict[str, Any],
    horizon_hours: int,
    *,
    model_dir: str | None = None,
) -> dict[str, Any]:
    """Run one weather-only forecast; returns arrays + provenance."""
    if target not in TARGET_COLUMNS:
        raise ValueError(f"Unknown target '{target}'.")
    horizon_hours = int(horizon_hours)
    if horizon_hours < 1:
        raise ValueError("horizon_hours must be >= 1.")
    origin = data.get("forecast_origin")

    if data.get("future_weather") is None:
        raise ValueError(
            "Historical evaluation cannot use a deployment model artifact. "
            "Use common.weather_only.backtest or scripts/backtest_weather_only.py "
            "to train an origin-specific model."
        )
    mode = LIVE
    weather = _live_weather(data, horizon_hours, origin)

    needed = required_weather_columns(target)
    missing = [c for c in needed if c not in weather.columns]
    if missing:
        raise ValueError(f"Weather input is missing required columns: {missing}")

    model_dir = model_dir or str(MODEL_DIR)
    out = predict(target, weather, model_dir=model_dir)
    info = model_info(target, model_dir)
    return {
        "model": MODEL_LABEL,
        "engine": "weather_only",
        "target": TARGET_COLUMNS[target],
        "input_mode": mode,
        "forecast_origin": origin,
        "horizon_hours": horizon_hours,
        "datetime": out[DATETIME_COL],
        "forecast": out["forecast"].to_numpy(),
        "lower_10": out["lower_10"].to_numpy(),
        "upper_90": out["upper_90"].to_numpy(),
        "actual": None,
        "metrics": None,
        "model_trained_through": info["trained_through"],
        "historical_test_scores": info["test_scores_observed_weather"],
        "trend_extrapolated_days": float(out["trend_extrapolated_days"].max()),
        "unit": "MU",
        "notes": (
            "Weather-only forecast: no recent generation/demand measurements are used. "
            "Accuracy depends on the quality of the supplied weather forecast."
        ),
    }


def format_timestamp(ts: pd.Timestamp, mode: str, style: str = "iso") -> str:
    """Format live forecast timestamps as timezone-aware Asia/Kolkata values."""
    ts = pd.Timestamp(ts)
    if mode == LIVE:
        return ts.tz_localize(KOLKATA).isoformat()
    return ts.isoformat() if style == "iso" else ts.strftime("%Y-%m-%d %H:%M:%S")


def _opt(value: float | None) -> float | None:
    return None if value is None or not np.isfinite(value) else float(value)


def shared_meta(r: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": r["model"],
        "engine": r["engine"],
        "target": r["target"],
        "input_mode": r["input_mode"],
        "forecast_origin": r["forecast_origin"],
        "horizon_hours": r["horizon_hours"],
        "unit": r["unit"],
        "metrics": r["metrics"],
        "model_trained_through": r["model_trained_through"],
        "historical_test_scores": r["historical_test_scores"],
        "trend_extrapolated_days": r["trend_extrapolated_days"],
        "notes": r["notes"],
    }


def solar_payload(r: dict[str, Any], horizon_name: str) -> dict[str, Any]:
    mode = r["input_mode"]
    records = [
        {
            "timestamp": format_timestamp(ts, mode),
            "forecast_mu": float(f),
            "q10": float(lo),
            "q90": float(hi),
        }
        for ts, f, lo, hi in zip(r["datetime"], r["forecast"], r["lower_10"], r["upper_90"])
    ]
    return {**shared_meta(r), "horizon": horizon_name, "point_quantile": "mean",
            "context_length": 0, "forecast": records}


def wind_payload(r: dict[str, Any]) -> dict[str, Any]:
    mode = r["input_mode"]
    actual = r["actual"] if r["actual"] is not None else [None] * len(r["forecast"])
    preds = [
        {
            "datetime": format_timestamp(ts, mode, "space"),
            "predicted": float(f),
            "actual": _opt(a),
            "lower_10": float(lo),
            "upper_90": float(hi),
        }
        for ts, f, a, lo, hi in zip(r["datetime"], r["forecast"], actual, r["lower_10"], r["upper_90"])
    ]
    return {**shared_meta(r), "predictions": preds}


def demand_payload(r: dict[str, Any]) -> dict[str, Any]:
    mode = r["input_mode"]
    preds = [
        {
            "datetime": format_timestamp(ts, mode, "space"),
            "forecast": round(float(f), 6),
            "lower_10": round(float(lo), 6),
            "upper_90": round(float(hi), 6),
        }
        for ts, f, lo, hi in zip(r["datetime"], r["forecast"], r["lower_10"], r["upper_90"])
    ]
    return {**shared_meta(r), "status": "success", "predictions": preds}
