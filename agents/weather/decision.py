from __future__ import annotations

from .config import MAX_INTERPOLATION_GAP


def _sentinel_gap_hours(df, column: str, value: float) -> dict:
    mask = df[column].eq(value)
    if not mask.any():
        return {"count": 0, "max_consecutive_hours": 0}
    groups = (mask != mask.shift()).cumsum()
    lengths = mask.groupby(groups).sum()
    max_run = int(lengths.max()) if not lengths.empty else 0
    return {"count": int(mask.sum()), "max_consecutive_hours": max_run}


def generate_decisions(df, sentinel_results, domain_results, outlier_results):
    decisions = []
    for column, issues in sentinel_results.items():
        for issue in issues:
            gap = _sentinel_gap_hours(df, column, issue["value"])
            if gap["max_consecutive_hours"] <= MAX_INTERPOLATION_GAP:
                action = "replace_and_interpolate"
                reason = "Short sentinel gaps can be safely interpolated between observed values."
            else:
                action = "investigate"
                reason = (
                    f"Long sentinel gap ({gap['max_consecutive_hours']} consecutive hours) "
                    "is not automatically imputed because it lacks sufficient observations."
                )
            decisions.append({
                "column": column,
                "issue_type": "sentinel_value",
                "issue_count": issue["count"],
                "detected_value": issue["value"],
                "max_consecutive_hours": gap["max_consecutive_hours"],
                "action": action,
                "reason": reason,
                "confidence": "HIGH",
            })

    for column, info in domain_results.items():
        decisions.append({
            "column": column,
            "issue_type": "domain_violation",
            "issue_count": info["count"],
            "action": "investigate",
            "reason": "Values violate configured physical/domain constraints.",
            "confidence": "HIGH",
        })

    for column, info in outlier_results.items():
        decisions.append({
            "column": column,
            "issue_type": "statistical_outlier",
            "issue_count": info["count"],
            "action": "retain",
            "reason": "Statistical outliers may represent legitimate weather events.",
            "confidence": "MEDIUM",
        })
    return decisions
