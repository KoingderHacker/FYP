"""Train and evaluate the weather-only solar / wind / demand models.

    python scripts/train_weather_only.py

Writes models/weather_only/*.joblib and reports/weather_only/metrics.json.
Re-run it whenever the CSVs are extended with new observations.
"""
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.weather_only.models import train_all  # noqa: E402


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    summary = train_all()
    print("\nWeather-only models (observed-weather backtest, chronological split)")
    print("-" * 72)
    for target, info in summary.items():
        test = info["scores_observed_weather_backtest"]["test"]
        base = info["scores_observed_weather_backtest"]["test_climatology_baseline"]
        q = info["data_quality"]
        print(f"{target:7s} rows={q['rows_used']:>6}  test R2={test['r2']:.3f}  "
              f"MAE={test['mae']:.3f} MU  RMSE={test['rmse']:.3f}  "
              f"(climatology R2={base['r2']:.3f})  "
              f"coverage q10-q90={test['q10_q90_coverage']:.2f}  "
              f"growth={info['annual_growth_pct']:.1f}%/yr  "
              f"trained through {info['trained_through']}")
    print("\nSaved: models/weather_only/*.joblib, reports/weather_only/metrics.json")


if __name__ == "__main__":
    main()
