MODEL_NAME = "amazon/chronos-2"
TARGET = "Gujarat_Wind_MU_hourly_est"
DATETIME_COL = "datetime"
CONTEXT_LENGTH = 720
QUANTILE_LEVELS = [0.1, 0.5, 0.9]
PREDICTION_QUANTILE = 0.5

HORIZONS = {
    "1_hour": 1,
    "1_day": 24,
    "1_week": 168,
    "15_days": 360,
}

FEATURES = [
    "WS50M", "WS10M", "WD50M", "WD10M", "T2M", "PS", "RH2M", "PRECTOTCORR",
    "hour_sin", "hour_cos", "day_of_year_sin", "day_of_year_cos",
    "wind_lag_1", "wind_lag_24", "wind_lag_168",
    "wind_rolling_mean_24", "wind_rolling_mean_168",
    "WS50M_cube", "WD50M_sin", "WD50M_cos",
]

FUTURE_COVARIATES = [
    "WS50M", "WS10M", "WD50M", "WD10M", "T2M", "PS", "RH2M", "PRECTOTCORR",
    "hour_sin", "hour_cos", "day_of_year_sin", "day_of_year_cos",
]

TARGET_DERIVED_FEATURES = [
    "wind_lag_1", "wind_lag_24", "wind_lag_168",
    "wind_rolling_mean_24", "wind_rolling_mean_168",
]

HISTORICAL_ONLY_ENGINEERED = ["WS50M_cube", "WD50M_sin", "WD50M_cos"]
