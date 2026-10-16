DOMAIN_RULES = {
    "RH2M": {"min": 0, "max": 100},
    "WS10M": {"min": 0},
    "WS50M": {"min": 0},
    "PS": {"min": 0},
    "WD10M": {"min": 0, "max": 360},
    "WD50M": {"min": 0, "max": 360},
    "Gujarat_Solar_MU_hourly_est": {"min": 0},
    "Gujarat_Wind_MU_hourly_est": {"min": 0},
    "Gujarat_Solar_MU_daily_total": {"min": 0},
    "Gujarat_Wind_MU_daily_total": {"min": 0},
    "PRECTOTCORR": {"min": 0},
}

# Regional Gujarat default, not a specific solar-plant location. Override these
# values when the plant coordinates are known.
OPEN_METEO_BASE_URL = "https://api.open-meteo.com/v1/forecast"
WEATHER_LATITUDE = 22.3
WEATHER_LONGITUDE = 71.5
WEATHER_TIMEZONE = "Asia/Kolkata"
WEATHER_REQUEST_TIMEOUT = 15

SENTINEL_VALUES = {
    "ALLSKY_SFC_SW_DWN": [-999],
    "CLRSKY_SFC_SW_DWN": [-999],
}

# Long unresolved gaps are deliberately not auto-imputed.
MAX_INTERPOLATION_GAP = 6
