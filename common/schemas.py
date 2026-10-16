from typing import Any, Literal
from pydantic import BaseModel, Field


class Horizon(BaseModel):
    name: str = "weather_quality"
    hours: int = 0


class AgentInput(BaseModel):
    request_id: str
    timestamp: str
    location: str = "Gujarat"
    horizon: Horizon = Field(default_factory=Horizon)
    data: dict[str, Any] = Field(default_factory=dict)
    config: dict[str, Any] = Field(default_factory=dict)


class AgentError(BaseModel):
    code: str
    message: str


class AgentOutput(BaseModel):
    status: Literal["success", "error"]
    agent: str
    request_id: str
    result: dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    error: AgentError | None = None
