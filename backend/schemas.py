from pydantic import BaseModel


class Horizon(BaseModel):
    name: str = "1d"
    hours: int = 24


class FYPRequest(BaseModel):
    request_id: str
    location: str = "Gujarat"
    horizon: Horizon
    # "live": forecast the hours after *now* from the Open-Meteo weather
    # forecast (no stale-history dependency). "replay": origin-aware historical
    # evaluation using only data that predates the supplied forecast origin.
    mode: str = "live"
    forecast_origin: str | None = None