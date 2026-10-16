"""Live hourly weather forecasts from Open-Meteo."""

from __future__ import annotations

import json
import logging
import math
import re
from datetime import datetime
from numbers import Integral
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .config import (
    OPEN_METEO_BASE_URL,
    WEATHER_LATITUDE,
    WEATHER_LONGITUDE,
    WEATHER_REQUEST_TIMEOUT,
    WEATHER_TIMEZONE,
)

logger = logging.getLogger(__name__)

SUPPORTED_HORIZONS = {1, 24, 168, 360}
TARGET_WIND_HEIGHT = 50

FEATURE_MAPPING = {
    "shortwave_radiation": "ALLSKY_SFC_SW_DWN",
    "shortwave_radiation_clear_sky": "CLRSKY_SFC_SW_DWN",
    "temperature_2m": "T2M",
    "relative_humidity_2m": "RH2M",
    "wind_speed_10m": "WS10M",
    "wind_direction_10m": "WD10M",
    "surface_pressure": "PS",
    "precipitation": "PRECTOTCORR",
}

REQUIRED_COLUMNS = [
    "datetime",
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

# Request the API's available standard height variables in one response. Which
# heights are usable is determined from the returned data, not this request list.
WIND_HEIGHTS_TO_REQUEST = tuple(range(10, 181, 10))


class OpenMeteoForecastError(RuntimeError):
    """Raised when Open-Meteo cannot provide a complete usable forecast."""


def _validate_coordinates(latitude: float, longitude: float) -> tuple[float, float]:
    try:
        latitude = float(latitude)
        longitude = float(longitude)
    except (TypeError, ValueError) as exc:
        raise ValueError("Open-Meteo latitude and longitude must be numeric.") from exc

    if (
        not math.isfinite(latitude)
        or not math.isfinite(longitude)
        or not -90 <= latitude <= 90
        or not -180 <= longitude <= 180
    ):
        raise ValueError("Open-Meteo latitude or longitude is outside valid bounds.")
    return latitude, longitude


def _to_local_hour(timestamp: datetime | pd.Timestamp | str) -> pd.Timestamp:
    try:
        result = pd.Timestamp(timestamp)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid forecast start time: {timestamp!r}") from exc

    if result.tzinfo is None:
        raise ValueError("Forecast start time must be timezone-aware.")

    result = result.tz_convert(ZoneInfo(WEATHER_TIMEZONE))
    if result.minute or result.second or result.microsecond or result.nanosecond:
        raise ValueError("Forecast start time must be aligned to an hourly timestamp.")
    return result


def _build_request_params(
    start: pd.Timestamp,
    horizon_hours: int,
    latitude: float,
    longitude: float,
) -> dict[str, Any]:
    forecast_end = start + pd.Timedelta(hours=horizon_hours - 1)
    requested_days = (forecast_end.date() - start.date()).days + 2
    forecast_days = min(requested_days, 16)

    hourly_variables = [
        *FEATURE_MAPPING.keys(),
        *(
            variable
            for height in WIND_HEIGHTS_TO_REQUEST
            for variable in (
                f"wind_speed_{height}m",
                f"wind_direction_{height}m",
            )
        ),
    ]
    return {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": ",".join(hourly_variables),
        "timezone": WEATHER_TIMEZONE,
        "forecast_days": forecast_days,
        "forecast_hours": min(horizon_hours + 24, 384),
    }


def _request_forecast(params: dict[str, Any]) -> dict[str, Any]:
    url = f"{OPEN_METEO_BASE_URL}?{urlencode(params)}"
    try:
        with urlopen(url, timeout=WEATHER_REQUEST_TIMEOUT) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise OpenMeteoForecastError(
            f"Open-Meteo forecast request failed: HTTP {exc.code} {exc.reason}"
        ) from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise OpenMeteoForecastError(
            f"Open-Meteo forecast request failed: {exc}"
        ) from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OpenMeteoForecastError(
            f"Open-Meteo returned invalid JSON: {exc}"
        ) from exc

    if not isinstance(payload, dict):
        raise OpenMeteoForecastError("Open-Meteo returned an invalid response object.")
    if payload.get("error"):
        raise OpenMeteoForecastError(
            f"Open-Meteo forecast request failed: {payload.get('reason', 'unknown API error')}"
        )
    return payload


def _parse_hourly_response(
    payload: dict[str, Any],
    start: pd.Timestamp,
    horizon_hours: int,
) -> pd.DataFrame:
    hourly = payload.get("hourly")
    if not isinstance(hourly, dict):
        raise OpenMeteoForecastError("Open-Meteo response is missing hourly data.")

    times = hourly.get("time")
    if not isinstance(times, list) or not times:
        raise OpenMeteoForecastError(
            "Open-Meteo response hourly data is missing datetime values."
        )

    try:
        parsed_times = pd.DatetimeIndex(
            pd.to_datetime(times, format="ISO8601", errors="raise")
        )
    except (TypeError, ValueError) as exc:
        raise OpenMeteoForecastError(
            f"Open-Meteo returned invalid hourly timestamps: {exc}"
        ) from exc

    if parsed_times.tz is None:
        parsed_times = parsed_times.tz_localize(ZoneInfo(WEATHER_TIMEZONE))
    else:
        parsed_times = parsed_times.tz_convert(ZoneInfo(WEATHER_TIMEZONE))

    if parsed_times.has_duplicates:
        raise OpenMeteoForecastError("Open-Meteo returned duplicate timestamps.")
    if not parsed_times.is_monotonic_increasing:
        raise OpenMeteoForecastError(
            "Open-Meteo timestamps are not sorted in ascending order."
        )
    if len(parsed_times) > 1 and not (
        (parsed_times[1:] - parsed_times[:-1]) == pd.Timedelta(hours=1)
    ).all():
        raise OpenMeteoForecastError(
            "Open-Meteo timestamps contain a missing or non-hourly interval."
        )

    data: dict[str, Any] = {"datetime": parsed_times}
    for variable in FEATURE_MAPPING:
        values = hourly.get(variable)
        if variable == "shortwave_radiation_clear_sky":
            if isinstance(values, list) and len(values) == len(parsed_times):
                data[variable] = values
            continue
        if not isinstance(values, list):
            raise OpenMeteoForecastError(
                f"Open-Meteo response is missing required variable '{variable}'."
            )
        if len(values) != len(parsed_times):
            raise OpenMeteoForecastError(
                f"Open-Meteo variable '{variable}' does not match the timestamp count."
            )
        data[variable] = values

    for key, values in hourly.items():
        if not isinstance(key, str) or not re.fullmatch(
            r"wind_(?:speed|direction)_\d+m", key
        ):
            continue
        if isinstance(values, list) and len(values) == len(parsed_times):
            data[key] = values

    complete = pd.DataFrame(data).set_index("datetime")
    expected = pd.date_range(start=start, periods=horizon_hours, freq="h")
    missing_times = expected.difference(complete.index)
    if not missing_times.empty:
        raise OpenMeteoForecastError(
            "Open-Meteo returned an incomplete forecast horizon: "
            f"{len(missing_times)} required hourly records are missing."
        )
    return complete.loc[expected].copy()


def _numeric_values(
    frame: pd.DataFrame,
    variable: str,
    *,
    required: bool = True,
) -> pd.Series | None:
    if variable not in frame:
        if required:
            raise OpenMeteoForecastError(
                f"Open-Meteo response is missing required variable '{variable}'."
            )
        return None

    values = pd.to_numeric(frame[variable], errors="coerce")
    if not np.isfinite(values.to_numpy(dtype=float, na_value=np.nan)).all():
        if required:
            raise OpenMeteoForecastError(
                f"Open-Meteo variable '{variable}' contains null, non-numeric, "
                "or non-finite forecast values."
            )
        return None
    return values.astype(float)


def _haurwitz_clear_sky_irradiance(
    timestamps: pd.Series,
    latitude: float,
    longitude: float,
) -> pd.Series:
    """Estimate clear-sky horizontal irradiance with the Haurwitz model."""
    local_times = pd.DatetimeIndex(timestamps)
    utc_times = local_times.tz_convert("UTC")
    day = utc_times.dayofyear.to_numpy(dtype=float)
    utc_hour = (
        utc_times.hour.to_numpy(dtype=float)
        + utc_times.minute.to_numpy(dtype=float) / 60.0
        + utc_times.second.to_numpy(dtype=float) / 3600.0
    )
    gamma = 2.0 * np.pi / 365.0 * (day - 1.0 + (utc_hour - 12.0) / 24.0)

    equation_of_time = 229.18 * (
        0.000075
        + 0.001868 * np.cos(gamma)
        - 0.032077 * np.sin(gamma)
        - 0.014615 * np.cos(2.0 * gamma)
        - 0.040849 * np.sin(2.0 * gamma)
    )
    declination = (
        0.006918
        - 0.399912 * np.cos(gamma)
        + 0.070257 * np.sin(gamma)
        - 0.006758 * np.cos(2.0 * gamma)
        + 0.000907 * np.sin(2.0 * gamma)
        - 0.002697 * np.cos(3.0 * gamma)
        + 0.001480 * np.sin(3.0 * gamma)
    )
    solar_minutes = utc_hour * 60.0 + equation_of_time + 4.0 * longitude
    hour_angle = np.deg2rad(solar_minutes / 4.0 - 180.0)
    latitude_radians = np.deg2rad(latitude)
    cosine_zenith = (
        np.sin(latitude_radians) * np.sin(declination)
        + np.cos(latitude_radians) * np.cos(declination) * np.cos(hour_angle)
    )
    cosine_zenith = np.clip(cosine_zenith, 0.0, 1.0)
    irradiance = 1098.0 * cosine_zenith * np.exp(
        -0.059 / np.maximum(cosine_zenith, np.finfo(float).tiny)
    )
    return pd.Series(irradiance, index=timestamps.index, dtype=float)


def _resolve_wind_height(frame: pd.DataFrame) -> tuple[int, pd.Series, pd.Series]:
    speed_variables: dict[int, str] = {}
    direction_variables: dict[int, str] = {}
    for column in frame.columns:
        match = re.fullmatch(r"wind_(speed|direction)_(\d+)m", str(column))
        if match:
            target = speed_variables if match.group(1) == "speed" else direction_variables
            target[int(match.group(2))] = str(column)

    valid_pairs = []
    for height in speed_variables.keys() & direction_variables.keys():
        # The 10 m pair remains a separate required feature and is not a
        # substitute for the model's 50 m input.
        if height == 10 and TARGET_WIND_HEIGHT != 10:
            continue
        speed = _numeric_values(frame, speed_variables[height], required=False)
        direction = _numeric_values(frame, direction_variables[height], required=False)
        if speed is not None and direction is not None:
            valid_pairs.append((height, speed, direction))

    if not valid_pairs:
        raise OpenMeteoForecastError(
            "No valid Open-Meteo wind speed/direction height pair is available "
            "for the requested forecast horizon."
        )

    height, speed, direction = min(
        valid_pairs,
        key=lambda candidate: (
            abs(candidate[0] - TARGET_WIND_HEIGHT),
            candidate[0],
        ),
    )
    return height, speed, direction


def get_weather_forecast(
    start_time: datetime | pd.Timestamp | str,
    horizon_hours: int,
    *,
    latitude: float | None = None,
    longitude: float | None = None,
) -> pd.DataFrame:
    """Fetch and normalize Open-Meteo hourly weather for the requested horizon.

    ``start_time`` is the first requested forecast timestamp and must be an
    aware, hourly-aligned datetime. Returned ``datetime`` values are aware in
    ``Asia/Kolkata``. Wind height metadata is available in ``DataFrame.attrs``.
    """
    if (
        isinstance(horizon_hours, bool)
        or not isinstance(horizon_hours, Integral)
        or horizon_hours not in SUPPORTED_HORIZONS
    ):
        raise ValueError(
            "Unsupported weather forecast horizon. Use 1, 24, 168, or 360 hours."
        )
    horizon_hours = int(horizon_hours)

    start = _to_local_hour(start_time)
    latitude, longitude = _validate_coordinates(
        WEATHER_LATITUDE if latitude is None else latitude,
        WEATHER_LONGITUDE if longitude is None else longitude,
    )
    params = _build_request_params(start, horizon_hours, latitude, longitude)

    logger.info("[WeatherAgent] Provider: Open-Meteo")
    logger.info("[WeatherAgent] Forecast start: %s", start.isoformat())
    logger.info("[WeatherAgent] Horizon: %s hours", horizon_hours)

    payload = _request_forecast(params)
    hourly = _parse_hourly_response(payload, start, horizon_hours)

    normalized = pd.DataFrame(index=hourly.index)
    resolved_height, resolved_speed, resolved_direction = _resolve_wind_height(hourly)
    for source, target in FEATURE_MAPPING.items():
        if source == "shortwave_radiation_clear_sky":
            continue
        normalized[target] = _numeric_values(hourly, source)

    try:
        clear_sky = _numeric_values(
            hourly, "shortwave_radiation_clear_sky", required=False
        )
    except (TypeError, ValueError):
        clear_sky = None
    if clear_sky is None:
        logger.warning(
            "[WeatherAgent] Open-Meteo clear-sky radiation is unavailable; "
            "using the Haurwitz clear-sky irradiance model."
        )
        normalized["CLRSKY_SFC_SW_DWN"] = _haurwitz_clear_sky_irradiance(
            pd.Series(hourly.index, index=hourly.index),
            latitude,
            longitude,
        )
        clear_sky_source = "haurwitz_clear_sky_model"
    else:
        normalized["CLRSKY_SFC_SW_DWN"] = clear_sky
        clear_sky_source = "open-meteo"

    normalized["WS50M"] = resolved_speed
    normalized["WD50M"] = resolved_direction
    normalized = normalized.reset_index(names="datetime")
    normalized = normalized[REQUIRED_COLUMNS]

    if len(normalized) != horizon_hours:
        raise OpenMeteoForecastError(
            f"Open-Meteo returned {len(normalized)} records; "
            f"{horizon_hours} were requested."
        )
    if normalized["datetime"].duplicated().any():
        raise OpenMeteoForecastError("Normalized forecast contains duplicate timestamps.")
    if not normalized["datetime"].is_monotonic_increasing:
        raise OpenMeteoForecastError("Normalized forecast timestamps are not sorted.")
    if len(normalized) > 1 and not (
        normalized["datetime"].diff().dropna() == pd.Timedelta(hours=1)
    ).all():
        raise OpenMeteoForecastError("Normalized forecast contains an hourly gap.")
    numeric = normalized.drop(columns="datetime").to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise OpenMeteoForecastError(
            "Normalized forecast contains null or non-finite weather values."
        )

    normalized.attrs.update(
        {
            "weather_source": "open-meteo",
            "latitude": latitude,
            "longitude": longitude,
            "requested_wind_height": TARGET_WIND_HEIGHT,
            "resolved_wind_height": resolved_height,
            "wind_height_fallback": resolved_height != TARGET_WIND_HEIGHT,
            "clear_sky_radiation_source": clear_sky_source,
            "forecast_start": normalized["datetime"].iloc[0].isoformat(),
            "forecast_end": normalized["datetime"].iloc[-1].isoformat(),
            "horizon_hours": horizon_hours,
        }
    )
    logger.info("[WeatherAgent] Retrieved records: %s", len(normalized))
    logger.info("[WeatherAgent] Wind height resolved: %sm", resolved_height)
    logger.info("[WeatherAgent] Validation: PASSED")
    return normalized


get_live_forecast = get_weather_forecast
