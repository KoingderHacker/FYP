"""Run explicit origin-aware observed-weather backtests without deployment artifacts."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.weather_only.backtest import (  # noqa: E402
    DEFAULT_HORIZONS,
    DEFAULT_ORIGINS,
    DEFAULT_TARGETS,
    load_backtest_frame,
    run_backtest_suite,
    write_backtest_outputs,
)


def _csv_values(value: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("Provide at least one comma-separated value.")
    return values


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Train origin-specific models and evaluate them against later target "
            "values using observed historical weather."
        )
    )
    parser.add_argument(
        "--origins",
        type=_csv_values,
        default=DEFAULT_ORIGINS,
        help="Comma-separated local timestamps; defaults to one origin.",
    )
    parser.add_argument(
        "--horizons",
        type=_csv_values,
        default=DEFAULT_HORIZONS,
        help="Comma-separated canonical horizons: 1h,1d,1w,15d.",
    )
    parser.add_argument(
        "--targets",
        type=_csv_values,
        default=DEFAULT_TARGETS,
        help="Comma-separated targets: solar,wind,demand.",
    )
    parser.add_argument("--min-history-rows", type=int, default=2000)
    parser.add_argument("--weather-csv", default="data/wind/raw/gujarat_hourly_weather_generation_final.csv")
    parser.add_argument("--demand-csv", default="data/demand/raw/gujarat_hourly_demand_estimated.csv")
    parser.add_argument(
        "--output-dir",
        help="New output directory. It must not already exist.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or (
        Path("reports/backtests")
        / f"run_{datetime.now().astimezone().strftime('%Y%m%d_%H%M%S')}"
    )
    try:
        frame = load_backtest_frame(args.weather_csv, args.demand_csv)
        result = run_backtest_suite(
            frame,
            origins=args.origins,
            horizons=args.horizons,
            targets=args.targets,
            min_history_rows=args.min_history_rows,
        )
        paths = write_backtest_outputs(result, output_dir)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"Backtest failed: {exc}\n")

    failed = [case for case in result["cases"] if case["status"] != "success"]
    for case in result["cases"]:
        if case["status"] == "success":
            metrics = case["metrics"]
            print(
                f"{case['target']:6s} origin={case['forecast_origin']} "
                f"horizon={case['horizon']['name']:3s} "
                f"MAE={metrics['mae']:.4f} RMSE={metrics['rmse']:.4f} "
                f"WAPE={metrics['wape_pct']}"
            )
        else:
            print(
                f"FAILED target={case['target']} origin={case['forecast_origin']} "
                f"horizon={case['horizon']}: {case['error']}"
            )
    print(f"Evaluation basis: {result['evaluation_method']}")
    for path in paths:
        print(f"Saved: {path}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
