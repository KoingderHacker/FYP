# Wind Agent — Chronos-2

Standardized Chronos-2 wind-generation forecasting agent.

## Final documented configuration

- Model: amazon/chronos-2
- Target: Gujarat_Wind_MU_hourly_est
- Context: 720 hours
- Horizons: 1, 24, 168, 360 hours ("1_hour", "1_day", "1_week", "15_days")
- Quantiles: 0.1, 0.5, 0.9
- Point forecast: q0.5
- Device: CPU/CUDA
- Historical features: 20
- Future covariates: 12 genuinely known/forecast weather covariates
- Target-derived features: historical only (wind_lag_1, wind_lag_24, wind_lag_168, wind_rolling_mean_24, wind_rolling_mean_168)
- No future wind target or target-derived features are supplied to Chronos-2.

## Standard Interface

```python
from agents.wind import run_agent

result = run_agent(input_data)
```

## Important data requirement

Historical replay/backtesting requires a historical `forecast_origin`. Live forecasting requires enough future weather/covariate rows after the latest historical target.

## Manual Test

```powershell
python -c "from agents.wind import run_agent; import json; x=json.load(open('sample_inputs/wind_manual.json')); r=run_agent(x); print(r['status'])"
```
