from __future__ import annotations

import logging
import time
from pathlib import Path

from common.schemas import AgentInput, AgentOutput, AgentError
from .diagnostics import (
    load_dataset, profile_dataset, validate_datetime, analyze_duplicates,
    analyze_missing_values, detect_sentinel_values, detect_domain_violations,
    analyze_statistics, detect_outliers,
)
from .decision import generate_decisions
from .cleaning import apply_cleaning
from .reporting import quality_impact, save_report
from .forecast import get_weather_forecast

AGENT_NAME = "weather"
DEFAULT_INPUT = "data/weather/raw/dataset.csv"
DEFAULT_OUTPUT = "data/weather/processed/weather_cleaned_dataset.csv"
DEFAULT_REPORT = "reports/weather/weather_report.json"


def run_agent(input_data: dict | AgentInput) -> dict:
    started = time.perf_counter()
    request = input_data if isinstance(input_data, AgentInput) else AgentInput.model_validate(input_data)

    if request.config.get("mode", "clean") == "forecast":
        return _run_forecast_mode(request, started)

    source = request.data.get("source", DEFAULT_INPUT)
    output_path = request.config.get("output_path", DEFAULT_OUTPUT)
    report_path = request.config.get("report_path", DEFAULT_REPORT)
    run_llm = bool(request.config.get("run_llm", False))

    try:
        df = load_dataset(source)
        profile = profile_dataset(df)
        datetime_result = validate_datetime(df)
        missing = analyze_missing_values(df)
        duplicates = analyze_duplicates(df)
        statistics = analyze_statistics(df)
        sentinel = detect_sentinel_values(df)
        domain = detect_domain_violations(df)
        outliers = detect_outliers(df)

        diagnostics = {
            "profile": profile,
            "datetime": datetime_result,
            "missing": missing,
            "duplicates": duplicates,
            "statistics": statistics,
            "sentinel": sentinel,
            "domain": domain,
            "outliers": outliers,
        }

        deterministic_decisions = generate_decisions(df, sentinel, domain, outliers)
        llm_analysis = None
        if run_llm:
            from .llm_decision import run_llm_decision
            llm_analysis = run_llm_decision(diagnostics)

        cleaned, execution_log = apply_cleaning(df, deterministic_decisions)
        final_missing = int(cleaned.isna().sum().sum())
        final_domain = detect_domain_violations(cleaned)
        final_validation = {
            "rows": int(len(cleaned)),
            "columns": int(len(cleaned.columns)),
            "missing_values": final_missing,
            "duplicate_rows": int(cleaned.duplicated().sum()),
            "domain_violations": final_domain,
            "ready_for_forecasting": final_missing == 0 and not final_domain,
        }

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cleaned.to_csv(output_path, index=False)

        result = {
            "cleaned_dataset_path": output_path,
            "diagnostics": diagnostics,
            "decisions": deterministic_decisions,
            "llm_analysis": llm_analysis,
            "execution_log": execution_log,
            "final_validation": final_validation,
            "quality_impact": quality_impact(df, cleaned, execution_log),
        }
        metadata = {
            "location": request.location,
            "source": source,
            "execution_time_seconds": round(time.perf_counter() - started, 3),
            "llm_enabled": run_llm,
        }
        payload = AgentOutput(
            status="success",
            agent=AGENT_NAME,
            request_id=request.request_id,
            result=result,
            metadata=metadata,
            error=None,
        ).model_dump()
        save_report(report_path, payload)
        return payload
    except Exception as exc:
        return AgentOutput(
            status="error",
            agent=AGENT_NAME,
            request_id=request.request_id,
            result=None,
            metadata={"execution_time_seconds": round(time.perf_counter() - started, 3)},
            error=AgentError(code=type(exc).__name__, message=str(exc)),
        ).model_dump()


def _run_forecast_mode(request: AgentInput, started: float) -> dict:
    logger = logging.getLogger(__name__)
    logger.info("[WeatherAgent] Mode: forecast")
    try:
        forecast_start = request.data.get("forecast_start")
        if forecast_start is None:
            raise ValueError("Forecast mode requires data.forecast_start.")
        forecast = get_weather_forecast(
            forecast_start,
            request.horizon.hours,
            latitude=request.config.get("latitude"),
            longitude=request.config.get("longitude"),
        )
        forecast_info = forecast.attrs
        records = [
            {
                key: value.isoformat() if key == "datetime" else float(value)
                for key, value in row.items()
            }
            for row in forecast.to_dict(orient="records")
        ]
        return AgentOutput(
            status="success",
            agent=AGENT_NAME,
            request_id=request.request_id,
            result={
                "weather_source": forecast_info["weather_source"],
                "forecast_start": forecast_info["forecast_start"],
                "forecast_end": forecast_info["forecast_end"],
                "horizon_hours": forecast_info["horizon_hours"],
                "records": records,
            },
            metadata={
                "location": request.location,
                "latitude": forecast_info["latitude"],
                "longitude": forecast_info["longitude"],
                "requested_wind_height": forecast_info["requested_wind_height"],
                "resolved_wind_height": forecast_info["resolved_wind_height"],
                "wind_height_fallback": forecast_info["wind_height_fallback"],
                "clear_sky_radiation_source": forecast_info[
                    "clear_sky_radiation_source"
                ],
                "execution_time_seconds": round(time.perf_counter() - started, 3),
            },
            error=None,
        ).model_dump()
    except Exception as exc:
        logger.error("[WeatherAgent] Forecast failed: %s", exc)
        return AgentOutput(
            status="error",
            agent=AGENT_NAME,
            request_id=request.request_id,
            result=None,
            metadata={
                "location": request.location,
                "execution_time_seconds": round(time.perf_counter() - started, 3),
            },
            error=AgentError(code=type(exc).__name__, message=str(exc)),
        ).model_dump()
