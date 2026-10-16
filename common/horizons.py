"""Canonical forecast horizon definitions used across the forecasting system."""

CANONICAL_HORIZONS = {
    "1h": 1,
    "1d": 24,
    "1w": 168,
    "15d": 360,
}

SOLAR_HORIZON_ALIASES = {
    "1h": "1_hour",
    "1d": "1_day",
    "1w": "1_week",
    "15d": "15_days",
}

WIND_HORIZON_ALIASES = {
    "1h": "1_hour",
    "1d": "1_day",
    "1w": "1_week",
    "15d": "15_days",
}


def validate_horizon_name(name: str | None) -> str:
    if name is None:
        raise ValueError("Forecast horizon name is required.")
    if name in CANONICAL_HORIZONS:
        return name
    for canonical, alias in SOLAR_HORIZON_ALIASES.items():
        if name == alias:
            return canonical
    for canonical, alias in WIND_HORIZON_ALIASES.items():
        if name == alias:
            return canonical
    raise ValueError(
        f"Unsupported forecast horizon '{name}'. Supported values: "
        f"{list(CANONICAL_HORIZONS)}."
    )


def validate_horizon(name: str | None, hours: int | None = None) -> tuple[str, int]:
    canonical = validate_horizon_name(name)
    expected = CANONICAL_HORIZONS[canonical]
    if hours is not None and int(hours) != expected:
        raise ValueError(
            f"Horizon '{canonical}' requires {expected} hours, received {hours}."
        )
    return canonical, expected


def to_solar_horizon(name: str) -> str:
    return SOLAR_HORIZON_ALIASES[validate_horizon_name(name)]


def to_wind_horizon(name: str) -> str:
    return WIND_HORIZON_ALIASES[validate_horizon_name(name)]
