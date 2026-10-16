import pytest
from fastapi import HTTPException

from backend import main
from backend.schemas import FYPRequest, Horizon


def test_live_route_calls_weather_then_existing_solar_agent(monkeypatch):
    calls = {}
    weather_records = [{"datetime": "2026-10-06T01:00:00+05:30", "T2M": 28.0}]

    def weather_agent(request):
        calls["weather"] = request
        return {
            "status": "success",
            "result": {"records": weather_records},
            "metadata": {"resolved_wind_height": 80},
        }

    def solar_agent(request):
        calls["solar"] = request
        return {"status": "success", "result": {"forecast": []}}

    monkeypatch.setattr(main, "run_weather_agent", weather_agent)
    monkeypatch.setattr(main, "run_solar_agent", solar_agent)

    result = main.run_live_solar_forecast(
        FYPRequest(
            request_id="live-test",
            location="Gujarat",
            horizon=Horizon(name="1d", hours=24),
        )
    )

    assert result["status"] == "success"
    assert result["forecast_mode"] == "live"
    assert calls["weather"]["config"]["mode"] == "forecast"
    assert calls["weather"]["horizon"]["hours"] == 24
    assert calls["solar"]["data"]["future_weather"] == weather_records
    assert calls["solar"]["data"]["forecast_origin"] == result["forecast_origin"]
    assert calls["solar"]["config"]["save_output"] is False
    assert calls["solar"]["horizon"]["name"] == "1_day"


def test_live_route_surfaces_weather_errors_without_fallback(monkeypatch):
    monkeypatch.setattr(
        main,
        "run_weather_agent",
        lambda _request: {
            "status": "error",
            "error": {"message": "No valid Open-Meteo wind speed/direction pair."},
        },
    )
    monkeypatch.setattr(
        main,
        "run_solar_agent",
        lambda _request: pytest.fail("Solar must not run after weather errors."),
    )

    with pytest.raises(HTTPException) as exc_info:
        main.run_live_solar_forecast(
            FYPRequest(
                request_id="live-failure",
                horizon=Horizon(name="1d", hours=24),
            )
        )

    assert exc_info.value.status_code == 502
    assert "No valid Open-Meteo" in exc_info.value.detail


def test_live_route_rejects_horizon_mismatch_before_api(monkeypatch):
    monkeypatch.setattr(
        main,
        "run_weather_agent",
        lambda _request: pytest.fail("Invalid horizon must be rejected first."),
    )

    with pytest.raises(HTTPException) as exc_info:
        main.run_live_solar_forecast(
            FYPRequest(
                request_id="bad-horizon",
                horizon=Horizon(name="1d", hours=360),
            )
        )

    assert exc_info.value.status_code == 422


def test_workflow_provenance_distinguishes_live_forecast_from_replay():
    from datetime import datetime
    from zoneinfo import ZoneInfo

    issued_at = datetime(2026, 10, 11, 20, 30, tzinfo=ZoneInfo("Asia/Kolkata"))
    result = {
        "timestamps": ["2026-10-11T21:00:00+05:30", "2026-10-11T22:00:00+05:30"],
        "weather_result": {
            "result": {
                "weather_source": "open-meteo",
                "forecast_start": "2026-10-11T21:00:00+05:30",
                "forecast_end": "2026-10-11T22:00:00+05:30",
                "records": [{}, {}],
            },
            "metadata": {"latitude": 22.3, "longitude": 71.5},
        },
    }
    live = main._workflow_provenance(
        result,
        mode="live",
        issued_at=issued_at,
        forecast_origin=issued_at.isoformat(),
        horizon_name="1h",
        horizon_hours=1,
        sources={"historical_weather_dataset_reference": "history.csv"},
    )
    replay = main._workflow_provenance(
        {"timestamps": ["2026-05-11T00:00:00"]},
        mode="replay",
        issued_at=issued_at,
        forecast_origin="2026-05-10T23:00:00",
        horizon_name="1h",
        horizon_hours=1,
        sources={"historical_weather_dataset_reference": "history.csv"},
    )

    assert live["issued_at"] == issued_at.isoformat()
    assert live["forecast_window"]["start"] == "2026-10-11T21:00:00+05:30"
    assert live["weather"]["provider"] == "open-meteo"
    assert live["weather"]["records"] == 2
    assert live["evaluation"]["kind"] == "not_yet_available_for_unobserved_live_forecast"
    assert replay["forecast_origin"] == "2026-05-10T23:00:00"
    assert replay["evaluation"]["kind"] == "historical_replay_on_observed_weather"


def test_replay_route_uses_origin_models_and_keeps_dashboard_payload(monkeypatch):
    calls = []
    optimization_request = {}
    origin = "2026-05-10T23:00:00"

    def fake_evaluation(_frame, target, requested_origin, horizon):
        calls.append((target, requested_origin, horizon))
        timestamp = "2026-05-11T00:00:00"
        return {
            "training_cutoff": "2026-05-10T00:00:00",
            "training": {"fit_rows": 10, "model_selection_rows": 2, "calibration_rows": 2},
            "model": {"configuration": {"max_iter": 5}},
            "weather_evaluation": "observed_weather_conditional",
            "weather_basis": "Historical observed weather; not forecast vintages.",
            "metrics": {"mae": 0.1, "rmse": 0.1, "r2": None, "wape_pct": 1.0},
            "predictions": [{
                "timestamp": timestamp,
                "forecast": 1.0,
                "actual": 1.1,
                "lower_10": 0.5,
                "upper_90": 1.5,
            }],
        }

    monkeypatch.setattr(main, "load_backtest_frame", lambda *_args: object())
    monkeypatch.setattr(main, "evaluate_origin", fake_evaluation)
    monkeypatch.setattr(main, "build_graph", lambda: pytest.fail("Replay must not load deployment graph."))
    monkeypatch.setattr(
        main,
        "run_optimization_agent",
        lambda request: (
            optimization_request.update(request)
            or {"status": "success", "result": {"summary_metrics": {}, "hourly_schedule": []}}
        ),
    )

    response = main.run_workflow(FYPRequest(
        request_id="replay-test",
        horizon=Horizon(name="1h", hours=1),
        mode="replay",
        forecast_origin=origin,
    ))

    assert response["status"] == "completed"
    assert response["solar_result"]["result"]["training"]["fit_rows"] == 10
    assert response["provenance"]["evaluation"]["kind"] == "historical_replay_on_observed_weather"
    assert "measured" not in response["provenance"]["dataset_usage"]
    assert optimization_request["config"]["save_output"] is False
    assert calls == [
        ("solar", origin, "1h"),
        ("wind", origin, "1h"),
        ("demand", origin, "1h"),
    ]


def test_replay_route_returns_validation_error_for_unavailable_horizon(monkeypatch):
    monkeypatch.setattr(main, "load_backtest_frame", lambda *_args: object())
    monkeypatch.setattr(
        main,
        "evaluate_origin",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ValueError("Insufficient subsequent timestamps.")
        ),
    )

    with pytest.raises(HTTPException) as exc_info:
        main.run_workflow(FYPRequest(
            request_id="replay-failure",
            horizon=Horizon(name="15d", hours=360),
            mode="replay",
            forecast_origin="2026-07-30T23:00:00",
        ))
    assert exc_info.value.status_code == 422
    assert "Insufficient subsequent timestamps" in str(exc_info.value.detail)


def test_replay_timezone_origin_is_normalized_consistently(monkeypatch):
    origins = []

    def fake_evaluation(_frame, _target, requested_origin, _horizon):
        origins.append(requested_origin)
        timestamp = "2026-05-11T06:00:00"
        return {
            "training_cutoff": "2026-05-10T00:00:00",
            "training": {"fit_rows": 10, "model_selection_rows": 2, "calibration_rows": 2},
            "model": {"configuration": {"max_iter": 5}},
            "weather_evaluation": "observed_weather_conditional",
            "weather_basis": "Historical observed weather; not forecast vintages.",
            "metrics": {"mae": 0.1, "rmse": 0.1, "r2": None, "wape_pct": 1.0},
            "predictions": [{
                "timestamp": timestamp,
                "forecast": 1.0,
                "actual": 1.1,
                "lower_10": 0.5,
                "upper_90": 1.5,
            }],
        }

    monkeypatch.setattr(main, "load_backtest_frame", lambda *_args: object())
    monkeypatch.setattr(main, "evaluate_origin", fake_evaluation)
    monkeypatch.setattr(
        main,
        "run_optimization_agent",
        lambda _request: {"status": "success", "result": {}},
    )

    result = main.run_workflow(FYPRequest(
        request_id="tz-replay",
        horizon=Horizon(name="1h", hours=1),
        mode="replay",
        forecast_origin="2026-05-11T05:30:00+05:30",
    ))

    assert set(origins) == {"2026-05-11T05:30:00"}
    assert result["forecast_origin"] == result["provenance"]["forecast_origin"]
