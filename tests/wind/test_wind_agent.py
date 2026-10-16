from agents.wind import run_agent


def test_missing_dataset_returns_standard_error():
    result = run_agent({
        "request_id": "wind_test_error",
        "timestamp": "2026-09-22T20:00:00",
        "location": "Gujarat",
        "horizon": {"name": "1_day", "hours": 24},
        "data": {},
        "config": {"save_output": False},
    })
    assert result["status"] == "error"
    assert result["agent"] == "wind"
    assert result["error"]["code"] == "ValueError"
    assert set(result) == {
        "status", "agent", "request_id", "result", "metadata", "error"
    }
