"""Basic Solar Agent contract test.

This test validates the agent interface without loading Chronos-2.
For an actual forecast run, use the manual test described in the README.
"""

from agents.solar.agent import run_agent


def test_solar_agent_missing_source_returns_standard_error():
    result = run_agent({
        "request_id": "solar_contract_test",
        "timestamp": "2026-10-02T12:00:00",
        "location": "Gujarat",
        "horizon": {"name": "1_day", "hours": 24},
        "data": {},
        "config": {},
    })

    assert result["status"] == "error"
    assert result["agent"] == "solar"
    assert result["request_id"] == "solar_contract_test"
    assert set(result) == {
        "status", "agent", "request_id", "result", "metadata", "error"
    }
