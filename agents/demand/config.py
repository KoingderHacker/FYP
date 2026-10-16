MODEL_NAME = "amazon/chronos-2"
DATETIME_COL = "datetime"
TARGET = "Energy_Met_MU_hourly_est"
CONTEXT_LENGTH = 1440
HORIZONS = {"1h": 1, "1d": 24, "1w": 168, "15d": 360}
QUANTILE_LEVELS = [0.1, 0.5, 0.9]
FEATURES = ["hour", "day_of_week", "is_weekend", "month", "hour_sin", "hour_cos", "day_of_year_sin", "day_of_year_cos", "is_holiday", "is_festival"]
DEFAULT_REPORT_PATH = "reports/demand/demand_report.json"
