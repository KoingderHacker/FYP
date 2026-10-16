"""Tests for the weather-only forecasting engine and the live workflow."""
import numpy as np
import pandas as pd
import pytest

from common.weather_only import features, models, service

WEATHER_CSV = "data/wind/raw/gujarat_hourly_weather_generation_final.csv"


def _synthetic_forecast(start="2026-10-12 00:00", hours=24, source_start="2025-10-12 00:00"):
    """Fake 'live forecast': last October's observed weather relabelled to the
    requested dates (stands in for Open-Meteo, which needs network access)."""
    raw = pd.read_csv(WEATHER_CSV, parse_dates=["datetime"]).set_index("datetime")
    idx = pd.date_range(source_start, periods=hours, freq="h")
    rows = raw.loc[idx, features.WEATHER_COLUMNS].reset_index(drop=True)
    rows.insert(0, "datetime",
                pd.date_range(start, periods=hours, freq="h", tz="Asia/Kolkata").map(lambda t: t.isoformat()))
    return rows.to_dict(orient="records")


# ---------------------------------------------------------------- features
def test_sentinels_are_masked_not_used():
    df = pd.read_csv(WEATHER_CSV).tail(48)
    assert (df["ALLSKY_SFC_SW_DWN"] == -999).all()  # the corrupted block
    masked = features.mask_sentinels(df)
    assert masked["ALLSKY_SFC_SW_DWN"].isna().all()


def test_training_data_excludes_corrupt_and_imputed_rows():
    frame, y, summary = models.load_training_frame("solar")
    assert (frame["ALLSKY_SFC_SW_DWN"] > -900).all()
    assert (frame["generation_source"] == "true_CEA").all()
    assert summary["rows_removed"]["invalid_weather"] > 0
    assert y.notna().all()


def test_features_never_reference_target_history():
    df = pd.read_csv(WEATHER_CSV).head(100)
    for target in ("solar", "wind", "demand"):
        cols = features.build_features(target, df).columns
        assert not any("lag" in c or "rolling" in c or "MU" in c for c in cols)


# ---------------------------------------------------------------- models
def test_models_report_sane_backtest_scores(use_trained):
    for target, floor in (("solar", 0.85), ("wind", 0.3), ("demand", 0.3)):
        test = models.model_info(target, use_trained)["test_scores_observed_weather"]
        assert test["r2"] > floor, (target, test)


def test_predict_refuses_sentinel_weather(use_trained):
    bad = pd.read_csv(WEATHER_CSV).tail(24)  # all -999 irradiance
    with pytest.raises(ValueError, match="missing/sentinel"):
        models.predict("solar", bad, model_dir=use_trained)


def test_live_forecast_shape_ordering_and_night(use_trained):
    data = {"forecast_origin": "2026-10-11T23:30:00+05:30",
            "future_weather": _synthetic_forecast(hours=24)}
    r = service.forecast_target("solar", data, 24)
    assert r["input_mode"] == service.LIVE
    assert len(r["forecast"]) == 24
    assert (r["lower_10"] <= r["forecast"]).all() and (r["forecast"] <= r["upper_90"]).all()
    assert (r["forecast"] >= 0).all()
    hours = pd.DatetimeIndex(r["datetime"]).hour
    assert r["forecast"][(hours <= 4) | (hours >= 20)].max() == 0.0   # night
    assert r["forecast"][(hours >= 10) & (hours <= 14)].max() > 3.0   # midday


def test_live_forecast_rejects_wrong_window(use_trained):
    data = {"forecast_origin": "2026-10-11T23:30:00+05:30",
            "future_weather": _synthetic_forecast(start="2026-10-13 00:00", hours=24)}
    with pytest.raises(ValueError, match="immediately after the forecast origin"):
        service.forecast_target("wind", data, 24)


def test_deployment_forecaster_refuses_historical_evaluation(use_trained):
    with pytest.raises(ValueError, match="origin-specific model"):
        service.forecast_target(
            "solar",
            {"forecast_origin": "2026-05-10T23:00:00", "source": WEATHER_CSV},
            24,
        )


# ---------------------------------------------------------------- workflow
@pytest.mark.parametrize("hours,name", [(1, "1h"), (24, "1d"), (168, "1w"), (360, "15d")])
def test_full_graph_live_mode_runs_without_fresh_history(use_trained, monkeypatch, hours, name):
    from orchestration import nodes
    from orchestration.graph import build_graph

    records = _synthetic_forecast(hours=hours)
    calls = {}

    def fake_weather(request):
        calls["req"] = request
        return {"status": "success", "result": {"records": records}, "metadata": {}, "error": None}

    monkeypatch.setattr(nodes, "run_weather_agent", fake_weather)
    state = {
        "request_id": "t", "location": "Gujarat", "mode": "live",
        "save_agent_reports": False,
        "timestamp": "2026-10-11T23:30:00+05:30",
        "forecast_origin": "2026-10-11T23:30:00+05:30",
        "horizon_name": name, "horizon_hours": hours,
        "solar_source": WEATHER_CSV, "wind_source": WEATHER_CSV,
        "demand_source": "data/demand/raw/gujarat_hourly_demand_estimated.csv",
        "status": "starting", "errors": [],
    }
    out = build_graph().invoke(state)
    assert out["status"] == "completed", out.get("errors")
    assert len(out["solar_forecast"]) == len(out["wind_forecast"]) == len(out["demand_forecast"]) == hours
    assert out["timestamps"][0].startswith("2026-10-12T00:00:00")
    assert calls["req"]["data"]["forecast_start"].startswith("2026-10-12T00:00:00")


def test_graph_stops_cleanly_when_weather_fails(use_trained, monkeypatch):
    from orchestration import nodes
    from orchestration.graph import build_graph

    monkeypatch.setattr(nodes, "run_weather_agent", lambda r: {
        "status": "error", "result": None, "metadata": {},
        "error": {"code": "URLError", "message": "offline"}})
    out = build_graph().invoke({
        "request_id": "t", "location": "Gujarat", "mode": "live",
        "timestamp": "2026-10-11T23:30:00+05:30",
        "forecast_origin": "2026-10-11T23:30:00+05:30",
        "horizon_name": "1d", "horizon_hours": 24,
        "solar_source": WEATHER_CSV, "wind_source": WEATHER_CSV,
        "demand_source": "data/demand/raw/gujarat_hourly_demand_estimated.csv",
        "status": "starting", "errors": []})
    assert out["status"] == "error" and out["errors"][0]["agent"] == "weather"
    assert "solar_forecast" not in out


def test_graph_refuses_unsafe_historical_replay(use_trained):
    from orchestration.graph import build_graph

    out = build_graph().invoke({
        "request_id": "t", "location": "Gujarat", "mode": "replay",
        "timestamp": "2026-10-03T12:00:00", "forecast_origin": "2026-05-10T23:00:00",
        "horizon_name": "1d", "horizon_hours": 24,
        "solar_source": WEATHER_CSV, "wind_source": WEATHER_CSV,
        "demand_source": "data/demand/raw/gujarat_hourly_demand_estimated.csv",
        "status": "starting", "errors": []})
    assert out["status"] == "error"
    assert "deployment artifacts" in out["errors"][0]["message"]
