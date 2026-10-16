from agents.optimization.agent import run_agent


def test_optimization_contract():
    payload = {
        "request_id": "test",
        "timestamp": "2026-10-03T12:00:00",
        "location": "Gujarat",
        "horizon": {"name": "1d", "hours": 3},
        "data": {
            "timestamps": ["2026-08-01 00:00:00","2026-08-01 01:00:00","2026-08-01 02:00:00"],
            "solar_forecast": [2, 3, 4],
            "wind_forecast": [2, 2, 2],
            "demand_forecast": [10, 10, 10],
        },
        "config": {"save_output": False},
    }
    result = run_agent(payload)
    assert result["status"] == "success"
    assert result["agent"] == "optimization"
    assert len(result["result"]["hourly_schedule"]) == 3
