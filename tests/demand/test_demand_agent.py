from agents.demand import run_agent


def test_missing_dataset_returns_standard_error():
    result = run_agent({
        "request_id": "demand_test_error",
        "timestamp": "2026-10-02T15:00:00",
        "location": "Gujarat",
        "horizon": {"name": "1d", "hours": 24},
        "data": {},
        "config": {"save_output": False},
    })
    assert result["status"] == "error"
    assert result["agent"] == "demand"
    assert result["error"]["code"] == "ValueError"
    assert set(result) == {
        "status", "agent", "request_id", "result", "metadata", "error"
    }
