from typing import Any, TypedDict


class FYPState(TypedDict, total=False):
    request_id: str
    timestamp: str
    location: str

    horizon_name: str
    horizon_hours: int

    # Replay/live configuration
    forecast_origin: str
    # "live": Open-Meteo weather drives the graph. Historical replay is
    # evaluated with origin-specific models outside the deployment graph.
    mode: str
    save_agent_reports: bool
    future_weather: list[dict[str, Any]]
    weather_result: dict[str, Any]

    # Dataset sources
    solar_source: str
    wind_source: str
    demand_source: str

    # Agent responses
    solar_result: dict[str, Any]
    wind_result: dict[str, Any]
    demand_result: dict[str, Any]
    optimization_result: dict[str, Any]

    # Normalized forecasts
    solar_forecast: list[float]
    wind_forecast: list[float]
    demand_forecast: list[float]
    timestamps: list[str]

    # Workflow
    status: str
    errors: list[dict[str, Any]]