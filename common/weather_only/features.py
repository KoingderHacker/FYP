"""Weather + calendar feature engineering for the weather-only models.

Every feature here can be computed from (a) a timestamp and (b) the weather
values for that same timestamp. Nothing depends on recent generation or demand
measurements, so the same code runs for training, backtests and live
forecasts driven by a weather forecast.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .calendar_data import FESTIVAL_DATES, HOLIDAY_DATES

DATETIME_COL = "datetime"

WEATHER_COLUMNS = [
    "ALLSKY_SFC_SW_DWN",
    "CLRSKY_SFC_SW_DWN",
    "T2M",
    "RH2M",
    "WS10M",
    "WS50M",
    "PS",
    "PRECTOTCORR",
    "WD10M",
    "WD50M",
]

# NASA POWER writes -999 for "no data". The raw CSV contains a block of them
# (2026-05-31 .. 2026-07-31) in the two irradiance columns.
SENTINEL_VALUE = -999.0

# Reference date for the slow-trend feature (first row of the training data).
TREND_ORIGIN = pd.Timestamp("2024-07-01")


def trend_days(datetimes: pd.Series) -> np.ndarray:
    """Days since TREND_ORIGIN (naive local time); drives the growth level."""
    dt = naive_local_datetime(pd.Series(datetimes))
    return ((dt - TREND_ORIGIN).dt.total_seconds() / 86400.0).to_numpy()


def mask_sentinels(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with -999 sentinels in weather columns replaced by NaN."""
    out = df.copy()
    for col in WEATHER_COLUMNS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
            out.loc[out[col] <= SENTINEL_VALUE + 1e-6, col] = np.nan
    return out


def naive_local_datetime(values: pd.Series) -> pd.Series:
    """Parse timestamps and return naive Asia/Kolkata wall-clock time."""
    ts = pd.to_datetime(values, errors="coerce")
    if ts.isna().any():
        raise ValueError("datetime column contains invalid timestamps.")
    if ts.dt.tz is not None:
        ts = ts.dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    return ts


def _calendar(df: pd.DataFrame) -> pd.DataFrame:
    dt = df[DATETIME_COL]
    feats = pd.DataFrame(index=df.index)
    feats["hour"] = dt.dt.hour
    feats["hour_sin"] = np.sin(2 * np.pi * dt.dt.hour / 24.0)
    feats["hour_cos"] = np.cos(2 * np.pi * dt.dt.hour / 24.0)
    doy = dt.dt.dayofyear - 1
    feats["day_of_year_sin"] = np.sin(2 * np.pi * doy / 365.25)
    feats["day_of_year_cos"] = np.cos(2 * np.pi * doy / 365.25)
    feats["month"] = dt.dt.month
    feats["day_of_week"] = dt.dt.dayofweek
    feats["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
    dates = dt.dt.strftime("%Y-%m-%d")
    feats["is_holiday"] = dates.isin(HOLIDAY_DATES).astype(int)
    feats["is_festival"] = dates.isin(FESTIVAL_DATES).astype(int)
    return feats


def _solar(df: pd.DataFrame, cal: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)
    allsky = df["ALLSKY_SFC_SW_DWN"]
    clear = df["CLRSKY_SFC_SW_DWN"]
    for c in ["ALLSKY_SFC_SW_DWN", "CLRSKY_SFC_SW_DWN", "T2M", "RH2M",
              "WS10M", "PS", "PRECTOTCORR"]:
        f[c] = df[c]
    kt = (allsky / clear.where(clear > 1.0)).clip(0, 1.5)
    f["clear_sky_index"] = kt.fillna(0.0)
    f["is_daylight"] = (clear > 1.0).astype(int)
    for c in ["hour_sin", "hour_cos", "day_of_year_sin", "day_of_year_cos",
              "hour", "month"]:
        f[c] = cal[c]
    return f


def _wind(df: pd.DataFrame, cal: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)
    for c in ["WS50M", "WS10M", "T2M", "PS", "RH2M", "PRECTOTCORR"]:
        f[c] = df[c]
    f["WS50M_cube"] = df["WS50M"] ** 3
    f["WS10M_cube"] = df["WS10M"] ** 3
    f["wind_shear_ratio"] = df["WS50M"] / df["WS10M"].clip(lower=0.1)
    rho = df["PS"] * 1000.0 / (287.05 * (df["T2M"] + 273.15))  # PS is in kPa
    f["air_density"] = rho
    f["wind_power_density"] = 0.5 * rho * df["WS50M"] ** 3
    f["WD50M_sin"] = np.sin(np.deg2rad(df["WD50M"]))
    f["WD50M_cos"] = np.cos(np.deg2rad(df["WD50M"]))
    f["WD10M_sin"] = np.sin(np.deg2rad(df["WD10M"]))
    f["WD10M_cos"] = np.cos(np.deg2rad(df["WD10M"]))
    for c in ["hour_sin", "hour_cos", "day_of_year_sin", "day_of_year_cos",
              "hour", "month"]:
        f[c] = cal[c]
    return f


def _demand(df: pd.DataFrame, cal: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)
    for c in ["T2M", "RH2M", "WS10M", "PRECTOTCORR"]:
        f[c] = df[c]
    # Simple apparent-heat proxy: hot *and* humid hours drive cooling load.
    f["heat_proxy"] = df["T2M"] * (1.0 + df["RH2M"] / 100.0)
    for c in ["hour", "hour_sin", "hour_cos", "day_of_year_sin",
              "day_of_year_cos", "month", "day_of_week", "is_weekend",
              "is_holiday", "is_festival"]:
        f[c] = cal[c]
    return f


_BUILDERS = {"solar": _solar, "wind": _wind, "demand": _demand}

def build_features(target: str, df: pd.DataFrame) -> pd.DataFrame:
    """Build the model input matrix for ``target`` from weather rows.

    ``df`` must contain ``datetime`` plus the weather columns used by that
    target. Raises ``ValueError`` when a required value is missing/invalid.
    """
    if target not in _BUILDERS:
        raise ValueError(f"Unknown target '{target}'. Use solar, wind or demand.")
    if DATETIME_COL not in df.columns:
        raise ValueError("Weather rows must contain a 'datetime' column.")

    work = mask_sentinels(df).reset_index(drop=True)
    work[DATETIME_COL] = naive_local_datetime(work[DATETIME_COL])
    cal = _calendar(work)
    return _BUILDERS[target](work, cal)


def required_weather_columns(target: str) -> list[str]:
    return {
        "solar": ["ALLSKY_SFC_SW_DWN", "CLRSKY_SFC_SW_DWN", "T2M", "RH2M",
                  "WS10M", "PS", "PRECTOTCORR"],
        "wind": ["WS50M", "WS10M", "T2M", "PS", "RH2M", "PRECTOTCORR",
                 "WD50M", "WD10M"],
        "demand": ["T2M", "RH2M", "WS10M", "PRECTOTCORR"],
    }[target]
