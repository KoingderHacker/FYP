from agents.weather import run_agent


def test_weather_agent_manual_run(tmp_path):
    out = tmp_path / "weather.csv"
    report = tmp_path / "report.json"
    result = run_agent({
        "request_id": "test_001",
        "timestamp": "2026-10-01T14:00:00",
        "location": "Gujarat",
        "data": {"source": "data/weather/raw/dataset.csv"},
        "config": {"run_llm": False, "output_path": str(out), "report_path": str(report)},
    })
    assert result["status"] == "success"
    assert result["agent"] == "weather"
    assert out.exists()
    # The source has a long -999 trailing block; it must remain unresolved, not fabricated.
    decisions = result["result"]["decisions"]
    assert any(d["action"] == "investigate" for d in decisions)
