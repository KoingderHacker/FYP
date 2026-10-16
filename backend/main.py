from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from agents.solar.agent import run_agent as run_solar_agent
from agents.weather.agent import run_agent as run_weather_agent
from agents.optimization.agent import run_agent as run_optimization_agent
from backend.schemas import FYPRequest
from common.horizons import validate_horizon
from common.weather_only.backtest import (
    evaluate_origin,
    load_backtest_frame,
)
from orchestration.graph import build_graph

app = FastAPI(
    title="FYP Renewable Energy Multi-Agent API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "status": "success",
        "message": "FYP Multi-Agent API is running",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
    }


def _live_forecast_times(
    now: datetime | None = None,
) -> tuple[datetime, datetime]:
    timezone = ZoneInfo("Asia/Kolkata")
    issued_at = now or datetime.now(timezone)
    if issued_at.tzinfo is None:
        raise ValueError("Live forecast issuance time must be timezone-aware.")
    issued_at = issued_at.astimezone(timezone)
    forecast_start = issued_at.replace(
        minute=0, second=0, microsecond=0
    ) + timedelta(hours=1)
    return issued_at, forecast_start


def _validate_horizon_or_422(horizon_name: str, hours: int) -> tuple[str, int]:
    try:
        canonical_name, expected_hours = validate_horizon(horizon_name, hours)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return canonical_name, expected_hours


def _normalize_replay_origin(value: str) -> str:
    try:
        origin = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"Invalid forecast_origin: {value}") from exc
    if origin.tzinfo is not None:
        origin = origin.astimezone(ZoneInfo("Asia/Kolkata")).replace(tzinfo=None)
    return origin.isoformat()


def _workflow_provenance(
    result: dict,
    *,
    mode: str,
    issued_at: datetime,
    forecast_origin: str,
    horizon_name: str,
    horizon_hours: int,
    sources: dict[str, str],
) -> dict:
    timestamps = result.get("timestamps") or []
    weather = result.get("weather_result") or {}
    weather_result = weather.get("result") or {}
    weather_metadata = weather.get("metadata") or {}
    solar_metadata = (result.get("solar_result") or {}).get("result") or {}
    wind_metadata = (result.get("wind_result") or {}).get("result") or {}
    demand_metadata = (result.get("demand_result") or {}).get("result") or {}
    models = {
        "solar": {
            "name": solar_metadata.get("model"),
            "trained_through": solar_metadata.get("model_trained_through"),
            "historical_test_scores": solar_metadata.get("historical_test_scores"),
        },
        "wind": {
            "name": wind_metadata.get("model"),
            "trained_through": wind_metadata.get("model_trained_through"),
            "historical_test_scores": wind_metadata.get("historical_test_scores"),
        },
        "demand": {
            "name": demand_metadata.get("model"),
            "trained_through": demand_metadata.get("model_trained_through"),
            "historical_test_scores": demand_metadata.get("historical_test_scores"),
        },
    }
    return {
        "mode": mode,
        "issued_at": issued_at.isoformat(),
        "timezone": "Asia/Kolkata",
        "forecast_origin": forecast_origin,
        "horizon": {"name": horizon_name, "hours": horizon_hours},
        "forecast_window": {
            "start": timestamps[0] if timestamps else None,
            "end": timestamps[-1] if timestamps else None,
            "points": len(timestamps),
        },
        "weather": {
            "provider": weather_result.get("weather_source")
            or ("Open-Meteo" if mode == "live" else "Historical weather CSV"),
            "start": weather_result.get("forecast_start"),
            "end": weather_result.get("forecast_end"),
            "records": (
                len(weather_result["records"])
                if isinstance(weather_result.get("records"), list)
                else None
            ),
            "latitude": weather_metadata.get("latitude"),
            "longitude": weather_metadata.get("longitude"),
            "resolved_wind_height_m": weather_metadata.get("resolved_wind_height"),
            "clear_sky_radiation_source": weather_metadata.get(
                "clear_sky_radiation_source"
            ),
        },
        "sources": sources,
        "dataset_usage": (
            "The configured CSV paths are supplied as agent references, but live "
            "prediction context comes from the Open-Meteo future_weather records; "
            "the weather-only live forecasters do not read historical target rows."
            if mode == "live"
            else "Origin-aware backtest trained without deployment artifacts. Historical "
            "observed weather and included dataset target values are used; source "
            "authenticity is not established by this evaluation."
        ),
        "input_basis": (
            [
                "Open-Meteo hourly forecast for the requested future window",
                "Solar: irradiance, temperature, humidity, wind, pressure and precipitation features",
                "Wind: wind speed/direction, temperature, pressure, humidity and precipitation features",
                "Demand: temperature, humidity, wind, precipitation and calendar features",
                "Historical generation/demand observations are not used as live forecast context",
            ]
            if mode == "live"
            else [
                "Historical weather observations at the replay timestamps",
                "Origin-specific models are trained using only rows available by the forecast origin",
                "Predictions are compared with subsequent dataset target values",
                "Replay uses observed weather; scores do not measure weather-provider forecast error",
            ]
        ),
        "models": models,
        "evaluation": {
            "kind": (
                "historical_replay_on_observed_weather"
                if mode == "replay"
                else "not_yet_available_for_unobserved_live_forecast"
            ),
            "note": (
                "Origin-aware replay metrics compare predictions with subsequent "
                "dataset target values using observed weather. They are not an "
                "operational weather-forecast accuracy guarantee."
                if mode == "replay"
                else "Accuracy cannot be measured until actual generation and demand "
                "for the forecast period have been observed."
            ),
        },
    }


@app.post("/api/live/solar")
def run_live_solar_forecast(request: FYPRequest):
    """Run Open-Meteo weather acquisition followed by Chronos-2 solar inference."""
    request_horizon_name, expected_hours = _validate_horizon_or_422(
        request.horizon.name, request.horizon.hours
    )
    issued_at, forecast_start = _live_forecast_times()
    horizon = {"name": request_horizon_name, "hours": expected_hours}
    solar_horizon = {"name": "1_day" if request_horizon_name == "1d" else request_horizon_name, "hours": expected_hours}
    if request_horizon_name == "1h":
        solar_horizon["name"] = "1_hour"
    elif request_horizon_name == "1w":
        solar_horizon["name"] = "1_week"
    elif request_horizon_name == "15d":
        solar_horizon["name"] = "15_days"

    weather = run_weather_agent(
        {
            "request_id": f"{request.request_id}_weather",
            "timestamp": issued_at.isoformat(),
            "location": request.location,
            "horizon": {"name": request_horizon_name, "hours": expected_hours},
            "data": {"forecast_start": forecast_start.isoformat()},
            "config": {"mode": "forecast"},
        }
    )
    if weather["status"] != "success":
        error = weather.get("error") or {}
        raise HTTPException(
            status_code=502,
            detail=f"Live weather forecast failed: {error.get('message', 'unknown error')}",
        )

    solar = run_solar_agent(
        {
            "request_id": f"{request.request_id}_solar",
            "timestamp": issued_at.isoformat(),
            "location": request.location,
            "horizon": solar_horizon,
            "data": {
                "source": "data/wind/raw/gujarat_hourly_weather_generation_final.csv",
                "forecast_origin": issued_at.isoformat(),
                "future_weather": weather["result"]["records"],
            },
            "config": {"save_output": False},
        }
    )
    if solar["status"] != "success":
        error = solar.get("error") or {}
        raise HTTPException(
            status_code=500,
            detail=f"Solar forecast failed: {error.get('message', 'unknown error')}",
        )

    return {
        "status": "success",
        "forecast_mode": "live",
        "forecast_origin": issued_at.isoformat(),
        "requested_horizon": {"name": request_horizon_name, "hours": expected_hours},
        "weather_result": weather,
        "solar_result": solar,
    }


@app.post("/api/run")
def run_workflow(request: FYPRequest):
    canonical_horizon_name, canonical_horizon_hours = _validate_horizon_or_422(
        request.horizon.name, request.horizon.hours
    )
    issued_at = datetime.now(ZoneInfo("Asia/Kolkata"))

    state = {
        "request_id": request.request_id,
        "timestamp": issued_at.isoformat(),
        "location": request.location,
        "horizon_name": canonical_horizon_name,
        "horizon_hours": canonical_horizon_hours,

        "solar_source": "data/wind/raw/gujarat_hourly_weather_generation_final.csv",
        "wind_source": "data/wind/raw/gujarat_hourly_weather_generation_final.csv",
        "demand_source": "data/demand/raw/gujarat_hourly_demand_estimated.csv",
        "forecast_origin": request.forecast_origin or "2026-05-10T23:00:00",
        "mode": "replay",
        "status": "starting",
        "errors": [],
    }

    if request.mode == "live":
        issued_at, _ = _live_forecast_times(issued_at)
        state["timestamp"] = issued_at.isoformat()
        state["forecast_origin"] = issued_at.isoformat()
        state["mode"] = "live"
        if request.forecast_origin is not None:
            raise HTTPException(
                status_code=422,
                detail="forecast_origin is accepted only when mode='replay'.",
            )
    elif request.mode != "replay":
        raise HTTPException(
            status_code=422,
            detail="mode must be 'live' or 'replay'.",
        )
    else:
        state["forecast_origin"] = _normalize_replay_origin(state["forecast_origin"])
        try:
            result = _run_origin_aware_replay(
                request=request,
                issued_at=issued_at,
                horizon_name=canonical_horizon_name,
                horizon_hours=canonical_horizon_hours,
                state=state,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if result.get("status") == "error":
            raise HTTPException(
                status_code=422,
                detail={
                    "message": "Historical backtest could not evaluate the requested forecast.",
                    "errors": result.get("errors", []),
                },
            )
        result["provenance"] = _workflow_provenance(
            result,
            mode="replay",
            issued_at=issued_at,
            forecast_origin=state["forecast_origin"],
            horizon_name=canonical_horizon_name,
            horizon_hours=canonical_horizon_hours,
            sources={
                "historical_weather_dataset_reference": state["solar_source"],
                "historical_demand_dataset_reference": state["demand_source"],
                "weather": state["solar_source"],
            },
        )
        return result

    graph = build_graph()
    result = graph.invoke(state)
    result["provenance"] = _workflow_provenance(
        result,
        mode=request.mode,
        issued_at=issued_at,
        forecast_origin=state["forecast_origin"],
        horizon_name=canonical_horizon_name,
        horizon_hours=canonical_horizon_hours,
        sources={
            "historical_weather_dataset_reference": state["solar_source"],
            "historical_demand_dataset_reference": state["demand_source"],
            "weather": (
                "Open-Meteo hourly forecast"
                if request.mode == "live"
                else state["solar_source"]
            ),
        },
    )
    if result.get("status") == "error" and result.get("errors"):
        result["partial_results"] = {
            "solar_result": result.get("solar_result"),
            "wind_result": result.get("wind_result"),
            "demand_result": result.get("demand_result"),
            "forecast_status": "partial",
        }
    return result


def _run_origin_aware_replay(
    *,
    request: FYPRequest,
    issued_at: datetime,
    horizon_name: str,
    horizon_hours: int,
    state: dict,
) -> dict:
    """Run historical target forecasts with freshly fitted pre-origin models."""
    frame = load_backtest_frame(state["solar_source"], state["demand_source"])
    origin = state["forecast_origin"]
    evaluations = {}
    errors = []
    for target in ("solar", "wind", "demand"):
        try:
            evaluations[target] = evaluate_origin(frame, target, origin, horizon_name)
        except ValueError as exc:
            errors.append({"agent": target, "message": str(exc)})
    if errors:
        return {
            "status": "error",
            "request_id": request.request_id,
            "mode": "replay",
            "forecast_origin": origin,
            "horizon_name": horizon_name,
            "horizon_hours": horizon_hours,
            "errors": errors,
            "partial_results": {
                "solar_result": None,
                "wind_result": None,
                "demand_result": None,
                "forecast_status": "partial",
            },
        }

    result: dict = {
        "status": "ready_for_optimization",
        "request_id": request.request_id,
        "mode": "replay",
        "forecast_origin": origin,
        "solar_source": state["solar_source"],
        "wind_source": state["wind_source"],
        "demand_source": state["demand_source"],
        "backtest_data_sources": getattr(frame, "attrs", {}).get("source_files", []),
        "timestamp": issued_at.isoformat(),
        "location": request.location,
        "horizon_name": horizon_name,
        "horizon_hours": horizon_hours,
        "errors": [],
    }
    timestamps: list[str] | None = None
    forecasts: dict[str, list[float]] = {}

    for target, evaluation in evaluations.items():
        rows = evaluation["predictions"]
        target_timestamps = [row["timestamp"] for row in rows]
        if timestamps is None:
            timestamps = target_timestamps
        elif timestamps != target_timestamps:
            raise ValueError("Origin-aware target forecasts are not timestamp-aligned.")
        forecasts[target] = [row["forecast"] for row in rows]
        metadata = {
            "model": "origin_aware_weather_only_gradient_boosting",
            "input_mode": evaluation["weather_evaluation"],
            "model_trained_through": evaluation["training_cutoff"],
            "historical_test_scores": None,
            "origin_backtest_metrics": evaluation["metrics"],
            "training": evaluation["training"],
            "model_configuration": evaluation["model"],
            "evaluation_method": evaluation["weather_evaluation"],
            "notes": evaluation["weather_basis"],
        }
        common = {
            "status": "success",
            "agent": target,
            "request_id": f"{request.request_id}_{target}",
            "metadata": metadata,
            "error": None,
        }
        if target == "solar":
            target_result = {
                **metadata,
                "metrics": evaluation["metrics"],
                "forecast": [
                    {
                        "timestamp": row["timestamp"],
                        "forecast_mu": row["forecast"],
                        "q10": row["lower_10"],
                        "q90": row["upper_90"],
                        "actual": row["actual"],
                    }
                    for row in rows
                ],
            }
            result["solar_result"] = {**common, "result": target_result}
        elif target == "wind":
            target_result = {
                **metadata,
                "metrics": evaluation["metrics"],
                "predictions": [
                    {
                        "datetime": row["timestamp"],
                        "predicted": row["forecast"],
                        "actual": row["actual"],
                        "lower_10": row["lower_10"],
                        "upper_90": row["upper_90"],
                    }
                    for row in rows
                ],
            }
            result["wind_result"] = {**common, "result": target_result}
        else:
            target_result = {
                **metadata,
                "metrics": evaluation["metrics"],
                "predictions": [
                    {
                        "datetime": row["timestamp"],
                        "forecast": row["forecast"],
                        "actual": row["actual"],
                        "lower_10": row["lower_10"],
                        "upper_90": row["upper_90"],
                    }
                    for row in rows
                ],
            }
            result["demand_result"] = {**common, "result": target_result}

    assert timestamps is not None
    result.update({
        "timestamps": timestamps,
        "solar_forecast": forecasts["solar"],
        "wind_forecast": forecasts["wind"],
        "demand_forecast": forecasts["demand"],
    })
    optimization = run_optimization_agent({
        "request_id": f"{request.request_id}_optimization",
        "timestamp": issued_at.isoformat(),
        "location": request.location,
        "horizon": {"name": horizon_name, "hours": horizon_hours},
        "data": {
            "solar_forecast": forecasts["solar"],
            "wind_forecast": forecasts["wind"],
            "demand_forecast": forecasts["demand"],
            "timestamps": timestamps,
        },
        "config": {"save_output": False},
    })
    result["optimization_result"] = optimization
    result["status"] = "completed" if optimization.get("status") == "success" else "error"
    if result["status"] == "error":
        result["errors"].append({
            "agent": "optimization",
            "message": str(optimization.get("error")),
        })
    return result