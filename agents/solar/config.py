"""Production configuration for the Chronos-2 Solar Agent.

These values reproduce the documented Chronos-2 solar experiment configuration.
"""

MODEL_NAME = "amazon/chronos-2"
TARGET_COLUMN = "Gujarat_Solar_MU_hourly_est"
DATETIME_COLUMN = "datetime"
ID_COLUMN = "id"

CONTEXT_LENGTH = 720

HORIZONS = {
    "1_hour": 1,
    "1_day": 24,
    "1_week": 168,
    "15_days": 360,
}

QUANTILE_LEVELS = [0.1, 0.5, 0.9]
POINT_QUANTILE = 0.5
DEVICE = "cpu"

FEATURE_COLUMNS = [
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
    "hour_sin",
    "hour_cos",
    "day_of_year_sin",
    "day_of_year_cos",
    "solar_lag_1",
    "solar_lag_24",
    "solar_lag_168",
    "solar_rolling_mean_24",
    "solar_rolling_mean_168",
]

FUTURE_COVARIATES = [
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
    "hour_sin",
    "hour_cos",
    "day_of_year_sin",
    "day_of_year_cos",
]

HISTORICAL_ONLY_FEATURES = [
    "solar_lag_1",
    "solar_lag_24",
    "solar_lag_168",
    "solar_rolling_mean_24",
    "solar_rolling_mean_168",
]
