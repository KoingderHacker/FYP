from __future__ import annotations

import pandas as pd

from .config import DOMAIN_RULES, SENTINEL_VALUES


def load_dataset(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def profile_dataset(df: pd.DataFrame) -> dict:
    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "columns_list": list(df.columns),
        "data_types": {c: str(t) for c, t in df.dtypes.items()},
        "missing_values": {
            c: int(v) for c, v in df.isna().sum().items() if v > 0
        },
        "duplicate_rows": int(df.duplicated().sum()),
        "numeric_columns": list(df.select_dtypes(include="number").columns),
        "categorical_columns": list(df.select_dtypes(exclude="number").columns),
    }


def validate_datetime(df: pd.DataFrame, column: str = "datetime") -> dict:
    if column not in df.columns:
        return {"valid": False, "error": f"{column} column not found"}

    s = pd.to_datetime(df[column], errors="coerce")
    invalid = int(s.isna().sum())
    duplicate = int(s.duplicated().sum())
    sorted_ok = bool(s.is_monotonic_increasing)
    ordered = s.dropna().sort_values()
    diffs = ordered.diff().dropna()
    gap_count = int((diffs != pd.Timedelta(hours=1)).sum())
    max_gap_hours = 0
    if not diffs.empty:
        max_gap_hours = int(diffs.max() / pd.Timedelta(hours=1))

    valid = invalid == 0 and duplicate == 0 and gap_count == 0
    return {
        "valid": valid,
        "invalid_dates": invalid,
        "duplicate_timestamps": duplicate,
        "is_sorted": sorted_ok,
        "non_hourly_intervals": gap_count,
        "max_interval_hours": max_gap_hours,
        "start": str(s.min()),
        "end": str(s.max()),
    }


def analyze_duplicates(df: pd.DataFrame) -> dict:
    count = int(df.duplicated().sum())
    return {"duplicate_rows": count, "has_duplicates": count > 0}


def analyze_missing_values(df: pd.DataFrame) -> dict:
    return {
        c: {"count": int(v), "percentage": round(v / len(df) * 100, 2)}
        for c, v in df.isna().sum().items()
        if v > 0
    }


def detect_sentinel_values(df: pd.DataFrame) -> dict:
    results = {}
    for column, values in SENTINEL_VALUES.items():
        if column not in df.columns:
            continue
        for value in values:
            count = int(df[column].eq(value).sum())
            if count:
                results.setdefault(column, []).append({"value": value, "count": count})
    return results


def detect_domain_violations(df: pd.DataFrame) -> dict:
    results = {}
    for column, rules in DOMAIN_RULES.items():
        if column not in df.columns:
            continue
        s = pd.to_numeric(df[column], errors="coerce")
        mask = pd.Series(False, index=df.index)
        if rules.get("min") is not None:
            mask |= s < rules["min"]
        if rules.get("max") is not None:
            mask |= s > rules["max"]
        count = int(mask.sum())
        if count:
            results[column] = {
                "count": count,
                "min_allowed": rules.get("min"),
                "max_allowed": rules.get("max"),
            }
    return results


def _analysis_copy(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for column, values in SENTINEL_VALUES.items():
        if column in out.columns:
            out[column] = out[column].replace(values, pd.NA)
    return out


def analyze_statistics(df: pd.DataFrame) -> dict:
    clean = _analysis_copy(df)
    results = {}
    for column in clean.select_dtypes(include="number").columns:
        s = clean[column].dropna()
        if s.empty:
            continue
        results[column] = {
            "min": float(s.min()),
            "max": float(s.max()),
            "mean": float(s.mean()),
            "median": float(s.median()),
            "std": float(s.std()),
            "q1": float(s.quantile(0.25)),
            "q3": float(s.quantile(0.75)),
            "zero_count": int(s.eq(0).sum()),
        }
    return results


def detect_outliers(df: pd.DataFrame) -> dict:
    clean = _analysis_copy(df)
    results = {}
    for column in clean.select_dtypes(include="number").columns:
        s = clean[column].dropna()
        if s.empty:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        count = int(((s < lower) | (s > upper)).sum())
        if count:
            results[column] = {
                "count": count,
                "percentage": count / len(s) * 100,
                "lower_bound": float(lower),
                "upper_bound": float(upper),
            }
    return results
