# Standardized Demand Agent

Chronos-2 Gujarat demand forecasting wrapped behind the common FYP `run_agent()` interface.

- 1440-hour / 60-day context
- `Energy_Met_MU_hourly_est` target
- calendar + holiday/festival features
- 1h / 1d / 1w / 15d horizons
- P10 / P50 / P90, P50 as point forecast
- zero-shot Chronos-2, no retraining
- optional `forecast_origin` for historical replay
