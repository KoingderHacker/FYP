"""Solar forecast reporting helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def save_forecast_report(result: dict[str, Any], path: str) -> str:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return str(output)
