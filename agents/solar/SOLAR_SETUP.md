# Solar Agent — Chronos-2

This is the production-agent wrapper around the already evaluated Chronos-2
solar forecasting configuration.

## Final documented configuration

- Model: amazon/chronos-2
- Target: Gujarat_Solar_MU_hourly_est
- Context: 720 hours
- Horizons: 1, 24, 168, 360 hours
- Quantiles: 0.1, 0.5, 0.9
- Point forecast: q0.5
- Device: CPU
- Historical features: 19
- Future covariates: 14
- Target-derived features: historical only
- No future solar target or target-derived features are supplied.

## Important data requirement

A forecast requires:

1. 720 historical rows ending at the forecast origin.
2. The next N rows containing the 14 future weather covariates.

For a historical replay/test, the same CSV can be used and `forecast_origin`
should point to a timestamp before the end of the dataset.

For real deployment, `forecast_origin` should be the latest observed timestamp
and the rows after it must come from a weather forecast source. The future
solar target must not be supplied.

## Manual test

From the FYP-Lang root:

```powershell
python -c "from agents.solar.agent import run_agent; import json; x=json.load(open('sample_inputs/solar_manual.json')); r=run_agent(x); print(r['status']); print(r['error']); print(r['result'])"
```

The first run will load Chronos-2 and can take roughly the documented model
load time (~10.65 seconds) before inference.

## Expected contract

Success:

```text
status = success
agent = solar
request_id = solar_test_001
```

The forecast is in:

```text
result.forecast
```

with timestamp, q10, forecast_mu (q50), and q90 when the installed Chronos-2
version returns all three quantiles.
