from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import pytest

from common.weather_only import backtest, models
from common.weather_only.features import DATETIME_COL


SMALL_GRID = [{"max_iter": 5, "max_leaf_nodes": 7, "min_samples_leaf": 2}]


@pytest.fixture
def history_frame():
    timestamps = pd.date_range("2025-01-01 00:00", periods=90 + 360, freq="h")
    hour = timestamps.hour.to_numpy()
    weather = {
        "ALLSKY_SFC_SW_DWN": np.maximum(0, 500 * np.sin((hour - 6) * np.pi / 12)),
        "CLRSKY_SFC_SW_DWN": np.maximum(0, 600 * np.sin((hour - 6) * np.pi / 12)),
        "T2M": 25 + 4 * np.sin(hour * np.pi / 12),
        "RH2M": 55 + 10 * np.cos(hour * np.pi / 12),
        "WS10M": 4 + 0.3 * np.sin(hour * np.pi / 12),
        "WS50M": 7 + 0.5 * np.sin(hour * np.pi / 12),
        "PS": np.full(len(timestamps), 101.0),
        "PRECTOTCORR": np.full(len(timestamps), 0.1),
        "WD10M": (hour * 12).astype(float),
        "WD50M": (hour * 12 + 10).astype(float),
    }
    frame = pd.DataFrame({DATETIME_COL: timestamps, **weather})
    frame["generation_source"] = "true_CEA"
    daylight = weather["ALLSKY_SFC_SW_DWN"] / 100.0
    frame[models.TARGET_COLUMNS["solar"]] = daylight * 2.0
    frame[models.TARGET_COLUMNS["wind"]] = 3.0 + np.asarray(weather["WS50M"]) * 0.2
    frame[models.TARGET_COLUMNS["demand"]] = 8.0 + np.asarray(weather["T2M"]) * 0.1
    return frame


def evaluate(frame, target="wind", horizon="1h", **kwargs):
    origin = pd.Timestamp("2025-01-04 17:00:00")
    options = {"min_history_rows": 60, "hyperparameter_grid": SMALL_GRID}
    options.update(kwargs)
    return backtest.evaluate_origin(
        frame,
        target,
        origin,
        horizon,
        **options,
    )


def test_training_validation_and_calibration_cutoffs_precede_origin(history_frame):
    result = evaluate(history_frame)
    origin = pd.Timestamp(result["forecast_origin"])
    assert pd.Timestamp(result["training_cutoff"]) <= origin
    assert pd.Timestamp(result["training"]["validation_end"]) <= origin
    assert pd.Timestamp(result["training"]["calibration_end"]) <= origin
    assert pd.Timestamp(result["latest_eligible_observation"]) <= origin
    assert "availability timestamps" in result["availability_basis"]
    assert result["training"]["model_selection_rows"] > 0
    assert result["training"]["calibration_rows"] > 0


@pytest.mark.parametrize(("horizon", "hours"), [("1h", 1), ("1d", 24), ("1w", 168), ("15d", 360)])
def test_all_canonical_horizons_have_exact_aligned_rows(history_frame, horizon, hours):
    result = evaluate(history_frame, horizon=horizon)
    assert result["status"] == "success"
    assert result["horizon"] == {"name": horizon, "hours": hours}
    timestamps = [pd.Timestamp(row["timestamp"]) for row in result["predictions"]]
    expected_start = pd.Timestamp(result["forecast_origin"]) + pd.Timedelta(1, unit="h")
    assert timestamps == list(pd.date_range(expected_start, periods=hours, freq="h"))
    assert result["metrics"]["points_scored"] == hours


def test_insufficient_pre_origin_history_is_explicit(history_frame):
    with pytest.raises(ValueError, match="Insufficient training history"):
        evaluate(history_frame, min_history_rows=100)


def test_missing_future_timestamp_is_not_filled(history_frame):
    missing_timestamp = history_frame[DATETIME_COL].iloc[90 + 4]
    frame = history_frame.loc[history_frame[DATETIME_COL] != missing_timestamp]
    with pytest.raises(ValueError, match="Insufficient subsequent timestamps"):
        evaluate(frame, horizon="1d")


def test_missing_pre_origin_hours_are_reported_without_fabricating_rows(history_frame):
    missing_timestamp = history_frame[DATETIME_COL].iloc[20]
    frame = history_frame.loc[history_frame[DATETIME_COL] != missing_timestamp]
    result = evaluate(frame)
    assert result["training"]["pre_origin_missing_hour_count"] == 1
    assert result["training"]["valid_pre_origin_rows"] == 89


def test_invalid_future_target_is_rejected(history_frame):
    frame = history_frame.copy()
    frame.loc[90, models.TARGET_COLUMNS["wind"]] = -1
    with pytest.raises(ValueError, match="Invalid or unavailable actual target"):
        evaluate(frame, horizon="1d")


def test_imputed_generation_targets_are_excluded_before_origin(history_frame):
    frame = history_frame.copy()
    frame.loc[10, "generation_source"] = "month_median_fill"
    result = evaluate(frame, target="solar")
    assert result["training"]["excluded_rows"]["invalid_target_or_source_rows"] == 1
    assert result["training"]["valid_pre_origin_rows"] == 89


def test_imputed_future_generation_target_is_rejected(history_frame):
    frame = history_frame.copy()
    frame.loc[90, "generation_source"] = "month_median_fill"
    with pytest.raises(ValueError, match="Invalid or unavailable actual target"):
        evaluate(frame, target="solar", horizon="1d")


def test_invalid_pre_origin_weather_is_excluded_from_training(history_frame):
    frame = history_frame.copy()
    frame.loc[10, "WS50M"] = -999
    result = evaluate(frame)
    assert result["training"]["excluded_rows"]["invalid_weather_rows"] == 1
    assert result["training"]["valid_pre_origin_rows"] == 89


def test_future_prediction_and_actual_are_joined_by_timestamp(history_frame):
    result = evaluate(history_frame, target="demand", horizon="1d")
    source = history_frame.set_index(DATETIME_COL)
    for row in result["predictions"]:
        assert row["actual"] == source.loc[pd.Timestamp(row["timestamp"]), models.TARGET_COLUMNS["demand"]]


def test_backtest_does_not_touch_deployment_artifact_files(history_frame, tmp_path, monkeypatch):
    artifact_dir = tmp_path / "weather_only"
    artifact_dir.mkdir()
    artifact = artifact_dir / "wind.joblib"
    artifact.write_bytes(b"deployment artifact sentinel")
    digest_before = hashlib.sha256(artifact.read_bytes()).hexdigest()
    mtime_before = artifact.stat().st_mtime_ns
    monkeypatch.setattr(models, "MODEL_DIR", artifact_dir)

    evaluate(history_frame)

    assert hashlib.sha256(artifact.read_bytes()).hexdigest() == digest_before
    assert artifact.stat().st_mtime_ns == mtime_before
    assert list(artifact_dir.iterdir()) == [artifact]


def test_metric_calculation_is_reproducible_and_aligned():
    actual = np.array([2.0, 4.0, 8.0])
    predicted = np.array([1.0, 5.0, 7.0])
    first = backtest.calculate_metrics(actual, predicted)
    second = backtest.calculate_metrics(actual.copy(), predicted.copy())
    assert first == second
    assert first["mae"] == pytest.approx(1.0)
    assert first["rmse"] == pytest.approx(1.0)
    assert first["wape_pct"] == pytest.approx(3 / 14 * 100)
    with pytest.raises(ValueError, match="aligned"):
        backtest.calculate_metrics(actual, predicted[:2])


def test_observed_weather_evaluation_is_not_operational():
    with pytest.raises(ValueError, match="archived.*weather forecast vintages"):
        backtest.evaluate_origin(
            pd.DataFrame(),
            "wind",
            "2025-01-01T00:00:00",
            "1h",
            weather_mode="operational_weather_forecast",
        )


def test_suite_records_unavailable_horizon_as_failed_case(history_frame):
    result = backtest.run_backtest_suite(
        history_frame.iloc[:200],
        origins=[history_frame[DATETIME_COL].iloc[89].isoformat()],
        horizons=["15d"],
        targets=["wind"],
        min_history_rows=60,
        hyperparameter_grid=SMALL_GRID,
    )
    assert result["cases"][0]["status"] == "failed"
    assert "Insufficient subsequent timestamps" in result["cases"][0]["error"]


def test_output_writer_creates_new_files_and_refuses_overwrite(history_frame, tmp_path):
    result = backtest.run_backtest_suite(
        history_frame,
        origins=[history_frame[DATETIME_COL].iloc[89].isoformat()],
        horizons=["1h"],
        targets=["wind"],
        min_history_rows=60,
        hyperparameter_grid=SMALL_GRID,
    )
    output = tmp_path / "run"
    paths = backtest.write_backtest_outputs(result, output)
    assert {path.name for path in paths} == {"results.json", "summary.csv", "predictions.csv"}
    with pytest.raises(FileExistsError):
        backtest.write_backtest_outputs(result, output)


def test_loader_rejects_duplicate_source_timestamps(tmp_path, history_frame):
    weather_path = tmp_path / "weather.csv"
    demand_path = tmp_path / "demand.csv"
    pd.concat([history_frame.iloc[:2], history_frame.iloc[[1]]]).to_csv(weather_path, index=False)
    history_frame[[DATETIME_COL, models.TARGET_COLUMNS["demand"]]].head(2).to_csv(demand_path, index=False)
    with pytest.raises(ValueError, match="duplicate timestamp"):
        backtest.load_backtest_frame(weather_path, demand_path)
