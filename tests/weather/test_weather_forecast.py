import json
from datetime import datetime
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd
import pytest

from agents.weather import forecast
from agents.weather.agent import run_agent


START = pd.Timestamp("2026-10-06T01:00:00+05:30")


class MockResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


def make_payload(horizon, heights=(10, 50, 80, 100), *, start=START):
    times = pd.date_range(start=start, periods=horizon, freq="h")
    hourly = {
        "time": [timestamp.strftime("%Y-%m-%dT%H:%M") for timestamp in times],
        "shortwave_radiation": [500.0] * horizon,
        "shortwave_radiation_clear_sky": [600.0] * horizon,
        "temperature_2m": [28.0] * horizon,
        "relative_humidity_2m": [70.0] * horizon,
        "surface_pressure": [1005.0] * horizon,
        "precipitation": [0.0] * horizon,
    }
    for height in heights:
        hourly[f"wind_speed_{height}m"] = [height / 10.0] * horizon
        hourly[f"wind_direction_{height}m"] = [height * 2.0] * horizon
    return {"hourly": hourly}


def mock_api(monkeypatch, payload):
    captured = {}

    def fake_urlopen(url, timeout):
        captured["url"] = url
        captured["timeout"] = timeout
        return MockResponse(payload)

    monkeypatch.setattr(forecast, "urlopen", fake_urlopen)
    return captured


@pytest.mark.parametrize("horizon", [1, 24, 168, 360])
def test_supported_horizons_return_exact_hourly_records(monkeypatch, horizon):
    payload = make_payload(horizon)
    captured = mock_api(monkeypatch, payload)

    result = forecast.get_weather_forecast(START, horizon)

    assert len(result) == horizon
    assert result["datetime"].iloc[0] == START
    assert str(result["datetime"].dt.tz) == "Asia/Kolkata"
    assert result["datetime"].diff().dropna().eq(pd.Timedelta(hours=1)).all()
    params = parse_qs(urlparse(captured["url"]).query)
    assert params["timezone"] == ["Asia/Kolkata"]
    expected_days = {1: 2, 24: 3, 168: 9, 360: 16}[horizon]
    assert int(params["forecast_days"][0]) == expected_days
    assert int(params["forecast_hours"][0]) == min(horizon + 24, 384)
    assert captured["timeout"] == forecast.WEATHER_REQUEST_TIMEOUT
    assert "wind_speed_50m" in params["hourly"][0]


def test_request_uses_configurable_coordinates_and_mapping(monkeypatch):
    captured = mock_api(monkeypatch, make_payload(24))
    result = forecast.get_weather_forecast(
        START, 24, latitude=22.1, longitude=71.2
    )

    params = parse_qs(urlparse(captured["url"]).query)
    assert params["latitude"] == ["22.1"]
    assert params["longitude"] == ["71.2"]
    first = result.iloc[0]
    assert first["ALLSKY_SFC_SW_DWN"] == 500.0
    assert first["CLRSKY_SFC_SW_DWN"] == 600.0
    assert first["T2M"] == 28.0
    assert first["RH2M"] == 70.0
    assert first["WS10M"] == 1.0
    assert first["WD10M"] == 20.0
    assert first["WS50M"] == 5.0
    assert first["WD50M"] == 100.0
    assert first["PS"] == 1005.0
    assert first["PRECTOTCORR"] == 0.0


@pytest.mark.parametrize(
    ("heights", "expected"),
    [
        ((10, 50, 80), 50),
        ((10, 40, 80), 40),
        ((10, 80), 80),
        ((10, 80, 100), 80),
        ((10, 20, 80), 20),
        ((10, 40, 80), 40),
        ((10, 100), 100),
    ],
)
def test_wind_height_selection_uses_nearest_valid_common_pair(
    monkeypatch, heights, expected
):
    payload = make_payload(24, heights)
    mock_api(monkeypatch, payload)

    result = forecast.get_weather_forecast(START, 24)

    assert result.attrs["requested_wind_height"] == 50
    assert result.attrs["resolved_wind_height"] == expected
    assert result.attrs["wind_height_fallback"] is (expected != 50)
    assert result["WS50M"].eq(expected / 10.0).all()
    assert result["WD50M"].eq(expected * 2.0).all()


def test_height_pair_with_missing_direction_is_rejected(monkeypatch):
    payload = make_payload(24, (10, 80, 100))
    del payload["hourly"]["wind_direction_80m"]
    mock_api(monkeypatch, payload)

    result = forecast.get_weather_forecast(START, 24)

    assert result.attrs["resolved_wind_height"] == 100
    assert result["WS50M"].eq(10.0).all()
    assert result["WD50M"].eq(200.0).all()


def test_height_pair_with_incomplete_values_is_rejected(monkeypatch):
    payload = make_payload(24, (10, 80, 100))
    payload["hourly"]["wind_direction_80m"][4] = None
    mock_api(monkeypatch, payload)

    result = forecast.get_weather_forecast(START, 24)

    assert result.attrs["resolved_wind_height"] == 100
    assert result["WS50M"].eq(10.0).all()
    assert result["WD50M"].eq(200.0).all()


def test_speed_and_direction_are_resolved_from_the_same_height(monkeypatch):
    payload = make_payload(24, (10, 50, 80))
    payload["hourly"]["wind_direction_50m"] = [None] * 24
    mock_api(monkeypatch, payload)

    result = forecast.get_weather_forecast(START, 24)

    assert result.attrs["resolved_wind_height"] == 80
    assert result["WS50M"].eq(8.0).all()
    assert result["WD50M"].eq(160.0).all()


def test_no_valid_common_wind_pair_fails(monkeypatch):
    payload = make_payload(24, (10, 50, 80))
    for height in (10, 50, 80):
        payload["hourly"][f"wind_direction_{height}m"] = [None] * 24
    mock_api(monkeypatch, payload)

    with pytest.raises(forecast.OpenMeteoForecastError, match="No valid Open-Meteo"):
        forecast.get_weather_forecast(START, 24)


def test_10m_wind_is_not_used_as_a_50m_fallback(monkeypatch):
    mock_api(monkeypatch, make_payload(24, (10,)))

    with pytest.raises(forecast.OpenMeteoForecastError, match="No valid Open-Meteo"):
        forecast.get_weather_forecast(START, 24)


@pytest.mark.parametrize(
    ("mutate", "error"),
    [
        (lambda hourly: hourly["time"].__setitem__(1, hourly["time"][0]), "duplicate"),
        (lambda hourly: hourly["time"].__setitem__(1, "2026-10-06T01:30"), "hourly"),
        (lambda hourly: hourly["temperature_2m"].__setitem__(0, np.nan), "non-finite"),
        (lambda hourly: hourly["temperature_2m"].__setitem__(0, np.inf), "non-finite"),
        (lambda hourly: hourly.pop("temperature_2m"), "missing required variable"),
        (
            lambda hourly: [
                hourly.__setitem__(key, values[1:])
                for key, values in list(hourly.items())
            ],
            "incomplete forecast horizon",
        ),
        (
            lambda hourly: hourly["temperature_2m"].__setitem__(0, "not-a-number"),
            "non-finite",
        ),
        (
            lambda hourly: hourly["time"].__setitem__(0, "not-a-timestamp"),
            "invalid hourly timestamps",
        ),
    ],
)
def test_invalid_forecast_data_is_rejected(monkeypatch, mutate, error):
    payload = make_payload(24)
    mutate(payload["hourly"])
    mock_api(monkeypatch, payload)

    with pytest.raises(forecast.OpenMeteoForecastError, match=error):
        forecast.get_weather_forecast(START, 24)


def test_unsorted_timestamps_are_rejected(monkeypatch):
    payload = make_payload(24)
    payload["hourly"]["time"][0], payload["hourly"]["time"][1] = (
        payload["hourly"]["time"][1],
        payload["hourly"]["time"][0],
    )
    mock_api(monkeypatch, payload)

    with pytest.raises(forecast.OpenMeteoForecastError, match="not sorted"):
        forecast.get_weather_forecast(START, 24)


def test_non_hourly_response_is_rejected(monkeypatch):
    payload = make_payload(24)
    payload["hourly"]["time"][1] = "2026-10-06T01:30"
    mock_api(monkeypatch, payload)

    with pytest.raises(forecast.OpenMeteoForecastError, match="non-hourly"):
        forecast.get_weather_forecast(START, 24)


def test_missing_hourly_section_is_rejected(monkeypatch):
    mock_api(monkeypatch, {})

    with pytest.raises(forecast.OpenMeteoForecastError, match="missing hourly"):
        forecast.get_weather_forecast(START, 24)


@pytest.mark.parametrize("status", [400, 500])
def test_timeout_and_http_failures_are_reported(monkeypatch, status):
    monkeypatch.setattr(
        forecast, "urlopen", lambda *_args, **_kwargs: (_ for _ in ()).throw(TimeoutError("timed out"))
    )
    with pytest.raises(forecast.OpenMeteoForecastError, match="timed out"):
        forecast.get_weather_forecast(START, 24)

    reason = "Bad Request" if status == 400 else "Server Error"
    monkeypatch.setattr(
        forecast,
        "urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            HTTPError(
                "https://api.open-meteo.com",
                status,
                reason,
                {},
                None,
            )
        ),
    )
    with pytest.raises(forecast.OpenMeteoForecastError, match=f"HTTP {status}"):
        forecast.get_weather_forecast(START, 24)


def test_malformed_json_is_reported(monkeypatch):
    class BadJsonResponse(MockResponse):
        def read(self):
            return b"{"

    monkeypatch.setattr(
        forecast,
        "urlopen",
        lambda *_args, **_kwargs: BadJsonResponse({}),
    )
    with pytest.raises(forecast.OpenMeteoForecastError, match="invalid JSON"):
        forecast.get_weather_forecast(START, 24)


def test_null_open_meteo_clear_sky_uses_explicit_physical_model(monkeypatch):
    start = pd.Timestamp("2026-10-06T12:00:00+05:30")
    payload = make_payload(24, start=start)
    payload["hourly"]["shortwave_radiation_clear_sky"] = [None] * 24
    mock_api(monkeypatch, payload)

    result = forecast.get_weather_forecast(start, 24)

    assert result.attrs["clear_sky_radiation_source"] == "haurwitz_clear_sky_model"
    assert result["CLRSKY_SFC_SW_DWN"].notna().all()
    assert np.isfinite(result["CLRSKY_SFC_SW_DWN"]).all()
    assert (result["CLRSKY_SFC_SW_DWN"] > 0).any()
    assert result.loc[
        result["datetime"].dt.hour.between(0, 5),
        "CLRSKY_SFC_SW_DWN",
    ].eq(0).all()


def test_missing_open_meteo_clear_sky_uses_explicit_physical_model(monkeypatch):
    start = pd.Timestamp("2026-10-06T12:00:00+05:30")
    payload = make_payload(24, start=start)
    payload["hourly"].pop("shortwave_radiation_clear_sky")
    mock_api(monkeypatch, payload)

    result = forecast.get_weather_forecast(start, 24)

    assert result.attrs["clear_sky_radiation_source"] == "haurwitz_clear_sky_model"
    assert np.isfinite(result["CLRSKY_SFC_SW_DWN"]).all()


def test_valid_open_meteo_clear_sky_is_preserved(monkeypatch):
    payload = make_payload(24)
    mock_api(monkeypatch, payload)

    result = forecast.get_weather_forecast(START, 24)

    assert result.attrs["clear_sky_radiation_source"] == "open-meteo"
    assert result["CLRSKY_SFC_SW_DWN"].eq(600.0).all()


@pytest.mark.parametrize(
    ("start", "horizon", "latitude", "longitude", "error"),
    [
        ("2026-10-06T01:00:00", 24, None, None, "timezone-aware"),
        (START, 12, None, None, "horizon"),
        (START + pd.Timedelta(minutes=15), 24, None, None, "hourly timestamp"),
        (START, 24, 91, 71, "outside valid bounds"),
        (START, 24, 22, float("inf"), "outside valid bounds"),
    ],
)
def test_request_arguments_are_validated(start, horizon, latitude, longitude, error):
    with pytest.raises(ValueError, match=error):
        forecast.get_weather_forecast(
            start, horizon, latitude=latitude, longitude=longitude
        )


def test_forecast_mode_keeps_standard_weather_agent_output(monkeypatch):
    mock_api(monkeypatch, make_payload(24))

    result = run_agent({
        "request_id": "weather_live_test",
        "timestamp": START.isoformat(),
        "location": "Gujarat",
        "horizon": {"name": "1_day", "hours": 24},
        "data": {"forecast_start": START.isoformat()},
        "config": {"mode": "forecast"},
    })

    assert result["status"] == "success"
    assert result["agent"] == "weather"
    assert result["metadata"]["resolved_wind_height"] == 50
    assert len(result["result"]["records"]) == 24
    assert result["result"]["records"][0]["datetime"] == START.isoformat()


def test_live_origin_is_timezone_aware_and_next_hour():
    from backend.main import _live_forecast_times

    issued, start = _live_forecast_times(datetime.fromisoformat("2026-10-06T00:45:00+05:30"))

    assert issued.isoformat() == "2026-10-06T00:45:00+05:30"
    assert start.isoformat() == "2026-10-06T01:00:00+05:30"
