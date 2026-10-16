from datetime import datetime, timezone

from common.horizons import validate_horizon
from common.schemas import AgentInput, AgentOutput, AgentError
from common.weather_only.service import demand_payload, forecast_target, use_weather_only
from .config import HORIZONS, DEFAULT_REPORT_PATH
from .forecasting import forecast_demand
from .reporting import save_report


def run_agent(input_data: dict | AgentInput) -> dict:
    request = AgentInput.model_validate(input_data)
    try:
        dataset_path = request.data.get("source") or request.data.get("dataset_path")
        if not dataset_path: raise ValueError("Demand agent requires data.source or data.dataset_path pointing to a CSV dataset.")
        name, hours = request.horizon.name, request.horizon.hours
        canonical_name, expected_hours = validate_horizon(name, hours)
        request.horizon.name = canonical_name
        request.horizon.hours = expected_hours
        name, hours = request.horizon.name, request.horizon.hours
        if name in HORIZONS and HORIZONS[name] != hours: raise ValueError(f"Horizon mismatch: {name} expects {HORIZONS[name]} hours, got {hours}.")
        if use_weather_only(request.config, request.data):
            # Weather + calendar engine: forecasts the hours after "now" rather
            # than the hours after the last (stale) row of the CSV.
            data = {**request.data, "demand_source": dataset_path}
            result = demand_payload(forecast_target("demand", data, hours))
            if request.config.get("save_output", True): result["report_path"] = save_report(result, request.config.get("report_path", DEFAULT_REPORT_PATH))
            return AgentOutput(status="success", agent="demand", request_id=request.request_id, result=result, metadata={"location":request.location,"source":dataset_path,"input_mode":result["input_mode"],"execution_timestamp":datetime.now(timezone.utc).isoformat()}, error=None).model_dump()
        result = forecast_demand(horizon=name, dataset_path=dataset_path, forecast_origin=request.data.get("forecast_origin"))
        if request.config.get("save_output", True): result["report_path"] = save_report(result, request.config.get("report_path", DEFAULT_REPORT_PATH))
        return AgentOutput(status="success", agent="demand", request_id=request.request_id, result=result, metadata={"location":request.location,"source":dataset_path,"execution_timestamp":datetime.now(timezone.utc).isoformat()}, error=None).model_dump()
    except Exception as exc:
        return AgentOutput(status="error", agent="demand", request_id=request.request_id, result=None, metadata={"location":request.location}, error=AgentError(code=type(exc).__name__, message=str(exc))).model_dump()
