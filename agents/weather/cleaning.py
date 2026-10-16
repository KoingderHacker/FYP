from __future__ import annotations

import numpy as np
import pandas as pd


def _replace_sentinels(df: pd.DataFrame, decisions: list[dict]) -> tuple[pd.DataFrame, list[dict]]:
    out = df.copy()
    log = []
    for decision in decisions:
        if decision["issue_type"] != "sentinel_value":
            continue
        column = decision["column"]
        if column not in out.columns or decision["action"] != "replace_and_interpolate":
            continue
        value = decision["detected_value"]
        mask = out[column].eq(value)
        affected = int(mask.sum())
        out.loc[mask, column] = np.nan
        # Only interpolate bounded/short gaps; never extrapolate boundaries.
        before_missing = int(out[column].isna().sum())
        out[column] = out[column].interpolate(method="linear", limit=6, limit_area="inside")
        # Safety: a long trailing/leading gap should not be filled by extrapolation.
        # The decision engine only allows short runs, so this is bounded by design.
        after_missing = int(out[column].isna().sum())
        log.append({
            "column": column,
            "issue_type": "sentinel_value",
            "action": "replace_and_interpolate",
            "affected_rows": affected,
            "missing_before_interpolation": before_missing,
            "remaining_missing": after_missing,
            "status": "executed",
        })
    return out, log


def apply_cleaning(df: pd.DataFrame, decisions: list[dict]) -> tuple[pd.DataFrame, list[dict]]:
    out = df.copy()
    log = []
    if "datetime" in out.columns:
        out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce")
        out = out.drop_duplicates().sort_values("datetime").reset_index(drop=True)

    out, sentinel_log = _replace_sentinels(out, decisions)
    log.extend(sentinel_log)

    for decision in decisions:
        if decision["issue_type"] == "domain_violation":
            log.append({
                "column": decision["column"],
                "issue_type": decision["issue_type"],
                "action": "investigate",
                "affected_rows": decision["issue_count"],
                "status": "flagged",
            })
        elif decision["issue_type"] == "statistical_outlier":
            log.append({
                "column": decision["column"],
                "issue_type": decision["issue_type"],
                "action": "retain",
                "affected_rows": decision["issue_count"],
                "status": "retained",
            })
        elif decision["issue_type"] == "sentinel_value" and decision["action"] == "investigate":
            log.append({
                "column": decision["column"],
                "issue_type": decision["issue_type"],
                "action": "investigate",
                "affected_rows": decision["issue_count"],
                "status": "flagged",
            })
    return out, log
