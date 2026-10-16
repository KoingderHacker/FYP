"""Origin-aware historical evaluation, isolated from deployment artifacts."""

from __future__ import annotations

import json
import hashlib
import platform
from itertools import product
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from common.horizons import CANONICAL_HORIZONS, validate_horizon

from . import models
from .features import (
    DATETIME_COL,
    build_features,
    mask_sentinels,
    naive_local_datetime,
    required_weather_columns,
)

DEFAULT_ORIGINS = ("2026-04-30T23:00:00",)
DEFAULT_TARGETS = ("solar", "wind", "demand")
DEFAULT_HORIZONS = tuple(CANONICAL_HORIZONS)
WEATHER_MODE = "observed_weather_conditional"
SUPPORTED_WEATHER_MODE = WEATHER_MODE


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _local_timestamp(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError(f"Invalid timestamp: {value!r}.")
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("Asia/Kolkata").tz_localize(None)
    return timestamp


def load_backtest_frame(
    weather_csv: str | Path = models.DEFAULT_WEATHER_CSV,
    demand_csv: str | Path = models.DEFAULT_DEMAND_CSV,
) -> pd.DataFrame:
    """Load source files once, preserving missing values and rejecting duplicates."""
    weather = pd.read_csv(weather_csv)
    demand = pd.read_csv(demand_csv, usecols=[DATETIME_COL, models.TARGET_COLUMNS["demand"]])

    for name, frame in (("weather", weather), ("demand", demand)):
        if DATETIME_COL not in frame:
            raise ValueError(f"{name} dataset is missing '{DATETIME_COL}'.")
        parsed = pd.to_datetime(frame[DATETIME_COL], errors="coerce")
        if parsed.isna().any():
            raise ValueError(f"{name} dataset contains invalid timestamps.")
        frame[DATETIME_COL] = naive_local_datetime(parsed)
        if frame[DATETIME_COL].duplicated().any():
            duplicate = frame.loc[frame[DATETIME_COL].duplicated(), DATETIME_COL].iloc[0]
            raise ValueError(f"{name} dataset contains duplicate timestamp {duplicate}.")

    weather = mask_sentinels(weather)
    demand_column = models.TARGET_COLUMNS["demand"]
    weather = weather.drop(columns=[demand_column], errors="ignore")
    frame = weather.merge(
        demand,
        on=DATETIME_COL,
        how="left",
        validate="one_to_one",
        sort=True,
    )
    frame = frame.sort_values(DATETIME_COL).reset_index(drop=True)
    frame.attrs["source_files"] = [
        {"path": str(Path(path).resolve()), "sha256": _sha256(path)}
        for path in (weather_csv, demand_csv)
    ]
    return frame


def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | None]:
    """Calculate deterministic metrics for finite, aligned target/prediction arrays."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if actual.ndim != 1 or predicted.ndim != 1 or len(actual) != len(predicted):
        raise ValueError("Actual and predicted values must be aligned one-dimensional arrays.")
    if len(actual) == 0:
        raise ValueError("At least one aligned actual/predicted point is required.")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Metrics require finite actual and predicted values.")
    total_actual = float(np.abs(actual).sum())
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)) if len(actual) >= 2 else None,
        "wape_pct": (
            float(np.abs(actual - predicted).sum() / total_actual * 100.0)
            if total_actual > 0
            else None
        ),
        "points_scored": int(len(actual)),
    }


def _normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if DATETIME_COL not in frame:
        raise ValueError(f"Backtest data is missing '{DATETIME_COL}'.")
    work = frame.copy()
    work[DATETIME_COL] = naive_local_datetime(work[DATETIME_COL])
    if work[DATETIME_COL].duplicated().any():
        duplicate = work.loc[work[DATETIME_COL].duplicated(), DATETIME_COL].iloc[0]
        raise ValueError(f"Backtest data contains duplicate timestamp {duplicate}.")
    work = mask_sentinels(work)
    return work.sort_values(DATETIME_COL).reset_index(drop=True)


def _valid_training_rows(frame: pd.DataFrame, target: str) -> tuple[pd.DataFrame, dict[str, int]]:
    column = models.TARGET_COLUMNS[target]
    if column not in frame:
        raise ValueError(f"Backtest data is missing target column '{column}'.")
    required = required_weather_columns(target)
    missing = [name for name in required if name not in frame]
    if missing:
        raise ValueError(f"Backtest data is missing required weather columns: {missing}.")

    y = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    valid_target = np.isfinite(y) & (y >= 0)
    if target in {"solar", "wind"} and "generation_source" not in frame:
        raise ValueError("Backtest data is missing 'generation_source' for generation targets.")
    if target in {"solar", "wind"}:
        valid_target &= frame["generation_source"].eq("true_CEA").to_numpy()
    weather = frame[required].apply(pd.to_numeric, errors="coerce")
    weather_valid = np.isfinite(weather.to_numpy(dtype=float)).all(axis=1)
    valid = valid_target & weather_valid
    return frame.loc[valid].copy().reset_index(drop=True), {
        "invalid_target_or_source_rows": int((~valid_target).sum()),
        "invalid_weather_rows": int((valid_target & ~weather_valid).sum()),
    }


def _future_test_rows(
    frame: pd.DataFrame,
    target: str,
    origin: pd.Timestamp,
    hours: int,
) -> pd.DataFrame:
    start = origin.floor("h") + pd.Timedelta(1, unit="h")
    expected = pd.date_range(start=start, periods=hours, freq="h")
    indexed = frame.set_index(DATETIME_COL, drop=False)
    missing = expected.difference(indexed.index)
    if len(missing):
        raise ValueError(
            f"Insufficient subsequent timestamps for {hours}-hour horizon after "
            f"{origin}; first missing timestamp is {missing[0]}."
        )
    rows = indexed.loc[expected].reset_index(drop=True)
    column = models.TARGET_COLUMNS[target]
    if column not in rows:
        raise ValueError(f"Backtest data is missing target column '{column}'.")
    actual = pd.to_numeric(rows[column], errors="coerce").to_numpy(dtype=float)
    invalid_actual = ~np.isfinite(actual) | (actual < 0)
    if target in {"solar", "wind"} and "generation_source" not in rows:
        raise ValueError("Backtest data is missing 'generation_source' for generation targets.")
    if target in {"solar", "wind"}:
        invalid_actual |= ~rows["generation_source"].eq("true_CEA").to_numpy()
    if invalid_actual.any():
        bad_timestamp = rows.loc[invalid_actual, DATETIME_COL].iloc[0]
        raise ValueError(
            f"Invalid or unavailable actual target for {target} at {bad_timestamp}."
        )
    required = required_weather_columns(target)
    weather = rows[required].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(weather).all():
        bad_row = int(np.where(~np.isfinite(weather).all(axis=1))[0][0])
        raise ValueError(
            f"Invalid observed weather for {target} at "
            f"{rows.loc[bad_row, DATETIME_COL]}."
        )
    return rows


def _split_indices(count: int) -> tuple[slice, slice, slice]:
    train_end = int(count * 0.70)
    validation_end = int(count * 0.85)
    if train_end < 1 or validation_end <= train_end or validation_end >= count:
        raise ValueError(
            "Insufficient pre-origin history to create non-empty training, "
            "validation, and calibration blocks."
        )
    return slice(0, train_end), slice(train_end, validation_end), slice(validation_end, count)


def _features(target: str, rows: pd.DataFrame) -> pd.DataFrame:
    result = build_features(target, rows)
    if not np.isfinite(result.to_numpy(dtype=float)).all():
        raise ValueError(f"Feature construction produced non-finite values for {target}.")
    return result


def evaluate_origin(
    frame: pd.DataFrame,
    target: str,
    forecast_origin: str | pd.Timestamp,
    horizon: str | int,
    *,
    min_history_rows: int = 2000,
    hyperparameter_grid: Iterable[dict[str, Any]] | None = None,
    weather_mode: str = WEATHER_MODE,
) -> dict[str, Any]:
    """Train at one origin, calibrate before it, and score only later observations.

    Historical rows provide observed weather, not archived weather forecasts.
    No deployment artifact is loaded, mutated, or written.
    """
    if target not in models.TARGET_COLUMNS:
        raise ValueError(f"Unknown target '{target}'.")
    if weather_mode != SUPPORTED_WEATHER_MODE:
        raise ValueError(
            "Operational weather-forecast backtesting is unsupported without "
            "archived, origin-available weather forecast vintages."
        )
    if min_history_rows < 3:
        raise ValueError("min_history_rows must be at least 3.")
    if isinstance(horizon, str):
        horizon_name, hours = validate_horizon(horizon)
    else:
        hours = int(horizon)
        matching = [name for name, value in CANONICAL_HORIZONS.items() if value == hours]
        if not matching:
            raise ValueError(f"Unsupported horizon length {hours}.")
        horizon_name = matching[0]

    origin = _local_timestamp(forecast_origin)
    work = _normalize_frame(frame)
    training_source = work.loc[work[DATETIME_COL] <= origin].copy()
    valid_training, exclusions = _valid_training_rows(training_source, target)
    if len(valid_training) < min_history_rows:
        raise ValueError(
            f"Insufficient training history for {target} at {origin}: "
            f"{len(valid_training)} valid rows; need {min_history_rows}."
        )
    train_slice, validation_slice, calibration_slice = _split_indices(len(valid_training))

    training_rows = valid_training.iloc[train_slice].reset_index(drop=True)
    validation_rows = valid_training.iloc[validation_slice].reset_index(drop=True)
    calibration_rows = valid_training.iloc[calibration_slice].reset_index(drop=True)
    fitting_rows = valid_training.iloc[:validation_slice.stop].reset_index(drop=True)

    column = models.TARGET_COLUMNS[target]
    y_train = training_rows[column].to_numpy(dtype=float)
    y_validation = validation_rows[column].to_numpy(dtype=float)
    if float(np.mean(y_train)) <= 0:
        raise ValueError(f"Training targets for {target} must have a positive mean.")
    X_train = _features(target, training_rows)
    X_validation = _features(target, validation_rows)
    initial_level_params = models.fit_level(training_rows[DATETIME_COL], y_train)
    initial_train_level = models.level_at(training_rows[DATETIME_COL], initial_level_params)
    validation_level = models.level_at(validation_rows[DATETIME_COL], initial_level_params)

    candidates = list(
        models.HYPERPARAMETER_GRID if hyperparameter_grid is None else hyperparameter_grid
    )
    if not candidates:
        raise ValueError("At least one hyperparameter candidate is required.")
    best_params: dict[str, Any] | None = None
    best_validation_mae = float("inf")
    candidate_scores = []
    search_train_matrix = np.vstack((X_train.to_numpy(), X_validation.to_numpy()))
    if not np.isfinite(search_train_matrix).all():
        raise ValueError(f"Model-selection features contain non-finite values for {target}.")
    for params in candidates:
        candidate = models._regressor(params).fit(X_train, y_train / initial_train_level)
        prediction = candidate.predict(X_validation) * validation_level
        prediction, _, _ = models._postprocess(
            target, prediction, prediction, prediction, validation_rows
        )
        score = float(mean_absolute_error(y_validation, prediction))
        candidate_scores.append({"hyperparameters": dict(params), "validation_mae": score})
        if score < best_validation_mae:
            best_params = dict(params)
            best_validation_mae = score
    if best_params is None:
        raise ValueError("Model selection did not produce a valid configuration.")

    y_fit = fitting_rows[column].to_numpy(dtype=float)
    if float(np.mean(y_fit)) <= 0:
        raise ValueError(f"Fitting targets for {target} must have a positive mean.")
    X_fit = _features(target, fitting_rows)
    level_params = models.fit_level(fitting_rows[DATETIME_COL], y_fit)
    fit_level = models.level_at(fitting_rows[DATETIME_COL], level_params)
    fitted_models = models._fit_models(X_fit, y_fit / fit_level, best_params)

    calibration_y = calibration_rows[column].to_numpy(dtype=float)
    calibration_X = _features(target, calibration_rows)
    calibration_level = models.level_at(calibration_rows[DATETIME_COL], level_params)
    calibration_prediction_matrix = np.vstack((X_fit.to_numpy(), calibration_X.to_numpy()))
    if not np.isfinite(calibration_prediction_matrix).all():
        raise ValueError(f"Calibration features contain non-finite values for {target}.")
    raw_low = fitted_models["q10"].predict(calibration_X) * calibration_level
    raw_high = fitted_models["q90"].predict(calibration_X) * calibration_level
    delta = models.conformal_delta(
        calibration_y,
        raw_low,
        raw_high,
        models._is_active(target, calibration_rows),
    )

    test_rows = _future_test_rows(work, target, origin, hours)
    X_test = _features(target, test_rows)
    test_level = models.level_at(test_rows[DATETIME_COL], level_params)
    point, lower, upper = models._predict_raw(
        fitted_models,
        X_test,
        test_level,
        target,
        test_rows,
        delta,
    )
    actual = test_rows[column].to_numpy(dtype=float)
    aligned_rows = int(len(actual))
    score = calculate_metrics(actual, point)
    active = models._is_active(target, test_rows)
    interval_points = int(active.sum())
    score["q10_q90_coverage"] = (
        float(np.mean((actual[active] >= lower[active]) & (actual[active] <= upper[active])))
        if interval_points
        else None
    )
    score["interval_points_scored"] = interval_points

    fit_cutoff = fitting_rows[DATETIME_COL].max()
    if fit_cutoff > origin:
        raise AssertionError("Origin-aware model fitting crossed the forecast origin.")
    raw_prior = work.loc[work[DATETIME_COL] <= origin, DATETIME_COL].sort_values()
    if len(raw_prior) > 1:
        expected_prior = pd.date_range(raw_prior.iloc[0], raw_prior.iloc[-1], freq="h")
        prior_gaps = len(expected_prior.difference(pd.DatetimeIndex(raw_prior)))
    else:
        prior_gaps = 0

    predictions = [
        {
            "timestamp": timestamp.isoformat(),
            "forecast": float(predicted),
            "actual": float(observed),
            "lower_10": float(low),
            "upper_90": float(high),
        }
        for timestamp, predicted, observed, low, high in zip(
            test_rows[DATETIME_COL], point, actual, lower, upper
        )
    ]
    return {
        "status": "success",
        "target": target,
        "target_column": column,
        "weather_evaluation": WEATHER_MODE,
        "weather_basis": (
            "Historical observed weather at target timestamps; not weather forecasts "
            "that were available at the forecast origin."
        ),
        "forecast_origin": origin.isoformat(),
        "horizon": {"name": horizon_name, "hours": hours},
        "forecast_window": {
            "start": test_rows[DATETIME_COL].iloc[0].isoformat(),
            "end": test_rows[DATETIME_COL].iloc[-1].isoformat(),
            "points": aligned_rows,
        },
        "training_cutoff": fit_cutoff.isoformat(),
        "latest_eligible_observation": valid_training[DATETIME_COL].max().isoformat(),
        "availability_basis": (
            "Rows are bounded by their observation timestamp; the source files do not "
            "provide publication/availability timestamps or historical data vintages."
        ),
        "training": {
            "valid_pre_origin_rows": int(len(valid_training)),
            "fit_rows": int(len(fitting_rows)),
            "model_selection_rows": int(len(validation_rows)),
            "calibration_rows": int(len(calibration_rows)),
            "validation_start": validation_rows[DATETIME_COL].iloc[0].isoformat(),
            "validation_end": validation_rows[DATETIME_COL].iloc[-1].isoformat(),
            "calibration_start": calibration_rows[DATETIME_COL].iloc[0].isoformat(),
            "calibration_end": calibration_rows[DATETIME_COL].iloc[-1].isoformat(),
            "pre_origin_missing_hour_count": int(prior_gaps),
            "excluded_rows": exclusions,
        },
        "model": {
            "class": "sklearn.ensemble.HistGradientBoostingRegressor",
            "configuration": best_params,
            "learning_rate": models.LEARNING_RATE,
            "random_state": models.SEED,
            "validation_mae": best_validation_mae,
            "candidate_scores": candidate_scores,
            "interval_calibration": "q10/q90 widening fitted only on pre-origin calibration block",
            "band_widening_mu": float(delta),
        },
        "metrics": score,
        "predictions": predictions,
    }


def run_backtest_suite(
    frame: pd.DataFrame,
    *,
    origins: Iterable[str],
    horizons: Iterable[str],
    targets: Iterable[str] = DEFAULT_TARGETS,
    min_history_rows: int = 2000,
    hyperparameter_grid: Iterable[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Evaluate a modest explicit set of origins, horizons, and targets."""
    origins = tuple(origins)
    horizons = tuple(horizons)
    targets = tuple(targets)
    if not origins or not horizons or not targets:
        raise ValueError("At least one origin, horizon, and target must be specified.")
    cases = []
    for target, origin, horizon in product(targets, origins, horizons):
        try:
            cases.append(
                evaluate_origin(
                    frame,
                    target,
                    origin,
                    horizon,
                    min_history_rows=min_history_rows,
                    hyperparameter_grid=hyperparameter_grid,
                )
            )
        except ValueError as exc:
            cases.append({
                "status": "failed",
                "target": target,
                "forecast_origin": str(origin),
                "horizon": horizon,
                "error": str(exc),
                "weather_evaluation": WEATHER_MODE,
            })
    return {
        "evaluation_method": WEATHER_MODE,
        "method_note": (
            "Observed-weather conditional evaluation. It is not an operational "
            "weather-forecast evaluation because historical forecast vintages are "
            "not supplied. Training eligibility is bounded by observation timestamp; "
            "historical publication-time availability is not verifiable from these files."
        ),
        "data_sources": frame.attrs.get("source_files", []),
        "runtime": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "cases": cases,
    }


def write_backtest_outputs(result: dict[str, Any], output_dir: str | Path) -> tuple[Path, ...]:
    """Write a new run directory only; never overwrite existing outputs."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=False)
    successful = [case for case in result["cases"] if case["status"] == "success"]
    summary_rows = [
        {
            "target": case["target"],
            "forecast_origin": case["forecast_origin"],
            "training_cutoff": case["training_cutoff"],
            "horizon": case["horizon"]["name"],
            "horizon_hours": case["horizon"]["hours"],
            **case["metrics"],
        }
        if case["status"] == "success"
        else {
            "target": case["target"],
            "forecast_origin": case["forecast_origin"],
            "horizon": case["horizon"],
            "status": "failed",
            "error": case["error"],
        }
        for case in result["cases"]
    ]
    predictions = [
        {
            "target": case["target"],
            "forecast_origin": case["forecast_origin"],
            "horizon": case["horizon"]["name"],
            **row,
        }
        for case in successful
        for row in case["predictions"]
    ]
    json_path = directory / "results.json"
    summary_path = directory / "summary.csv"
    predictions_path = directory / "predictions.csv"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    pd.DataFrame(predictions).to_csv(predictions_path, index=False)
    return json_path, summary_path, predictions_path
