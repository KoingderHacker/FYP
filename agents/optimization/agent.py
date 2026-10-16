from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from common.schemas import AgentInput, AgentOutput, AgentError

from .config import (
    BATTERY_CAPACITY_MU,
    MAX_CHARGE_RATE_MU,
    MAX_DISCHARGE_RATE_MU,
    INITIAL_SOC_PCT,
    OBJECTIVE,
)
from .optimizer import GridOptimizer
from .reporting import save_report


def _extract_forecast(data: dict, key: str):
    value = data.get(key)
    if value is None:
        return None

    # Supports:
    # 1) direct list
    # 2) {"predictions": [...]}
    # 3) {"forecast": [...]}
    if isinstance(value, dict):
        value = value.get("predictions", value.get("forecast"))
    return value


def run_agent(input_data: dict | AgentInput) -> dict:
    request = AgentInput.model_validate(input_data)

    try:
        data = request.data or {}
        config = request.config or {}
        horizon_name = request.horizon.name
        horizon_hours = request.horizon.hours

        solar = _extract_forecast(data, "solar_forecast")
        wind = _extract_forecast(data, "wind_forecast")
        demand = _extract_forecast(data, "demand_forecast")
        timestamps = data.get("timestamps")

        if solar is None or wind is None or demand is None or timestamps is None:
            raise ValueError(
                "Optimization Agent requires solar_forecast, wind_forecast, "
                "demand_forecast and timestamps in data. "
                "Forecast agents are intentionally kept separate so this agent "
                "can be used directly by LangGraph later."
            )

        if len(timestamps) != horizon_hours:
            raise ValueError(
                f"Expected {horizon_hours} timestamps for horizon {horizon_name}, "
                f"received {len(timestamps)}."
            )

        optimizer = GridOptimizer(
            battery_capacity_mu=float(
                config.get("battery_capacity_mu", BATTERY_CAPACITY_MU)
            ),
            max_charge_rate_mu=float(
                config.get("max_charge_rate_mu", MAX_CHARGE_RATE_MU)
            ),
            max_discharge_rate_mu=float(
                config.get("max_discharge_rate_mu", MAX_DISCHARGE_RATE_MU)
            ),
            initial_soc_pct=float(
                config.get("initial_soc_pct", INITIAL_SOC_PCT)
            ),
        )

        result = optimizer.optimize(
            solar_predictions=solar,
            wind_predictions=wind,
            demand_predictions=demand,
            timestamps=timestamps,
            horizon_name=horizon_name,
        )

        if config.get("save_output", True):
            report_path = config.get(
                "report_path", "reports/optimization/optimization_report.json"
            )
            result["report_path"] = save_report(result, report_path)

        return AgentOutput(
            status="success",
            agent="optimization",
            request_id=request.request_id,
            result=result,
            metadata={
                "location": request.location,
                "horizon": horizon_name,
                "optimization_objective": OBJECTIVE,
                "execution_timestamp": datetime.now(timezone.utc).isoformat(),
            },
            error=None,
        ).model_dump()

    except Exception as exc:
        return AgentOutput(
            status="error",
            agent="optimization",
            request_id=request.request_id,
            result=None,
            metadata={"location": request.location},
            error=AgentError(
                code=type(exc).__name__,
                message=str(exc),
            ),
        ).model_dump()
