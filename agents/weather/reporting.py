from __future__ import annotations

import json
from pathlib import Path


def quality_impact(original, cleaned, execution_log):
    return {
        "original_rows": int(len(original)),
        "cleaned_rows": int(len(cleaned)),
        "original_missing": int(original.isna().sum().sum()),
        "cleaned_missing": int(cleaned.isna().sum().sum()),
        "original_duplicates": int(original.duplicated().sum()),
        "cleaned_duplicates": int(cleaned.duplicated().sum()),
        "executed_actions": sum(x["status"] == "executed" for x in execution_log),
        "flagged_actions": sum(x["status"] == "flagged" for x in execution_log),
        "retained_actions": sum(x["status"] == "retained" for x in execution_log),
    }


def save_report(path: str, payload: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
