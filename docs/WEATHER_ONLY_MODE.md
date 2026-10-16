# Weather-only forecasting mode

## Why this exists

The original solar / wind / demand agents (Chronos-2) were fed the last
720-1440 hours of *measured* generation or demand and forecast the hours right
after the last row of the CSV. The CSVs end on 2026-07-31 and are not updated
daily, so a forecast for "tomorrow" was impossible: the context was months old
and the future timestamps did not line up. In addition, the raw CSV has
`-999` (NASA "no data") in both irradiance columns from 2026-05-31 to
2026-07-31, so the whole 720-hour solar context window was invalid, and 1,368
target rows are imputed month medians (`generation_source != "true_CEA"`).

## How it works now

```
Open-Meteo forecast  ->  weather agent  ->  solar / wind / demand agents  ->  optimizer
   (fresh every run)                         (weather + calendar only)
```

* One gradient-boosted model per target maps *the weather and calendar of a
  target hour* to generation/demand for that hour. No lags, no rolling means,
  no recent measurements. 1 h, 24 h, 7 d and 15 d forecasts are the same model
  run over a longer weather forecast.
* In **live mode**, the historical CSV paths are agent references only; the
  weather-only forecasters use the supplied Open-Meteo `future_weather` records
  as forecast context and do not read historical target rows. Model artifacts
  were trained from historical CSV data. In dashboard **replay mode**, models
  are now fit specifically for the requested historical origin; no deployment
  artifact trained after that origin is loaded. Replay uses observed weather
  and included dataset target values, whose upstream authenticity is not
  established by this repository.
* Bad rows are removed, never imputed: sentinel irradiance, imputed targets.
* A multiplicative linear *growth level* (installed capacity / load growth,
  fitted with month dummies) is divided out for training and applied at
  prediction. It is frozen 120 days after the last training row, and every
  forecast reports `trend_extrapolated_days` so staleness is visible.
* Uncertainty band: q10/q90 gradient-boosting models, widened by conformal
  calibration on the validation block (never the test block).

## Running it

```powershell
pip install -r requirements.txt
python scripts/train_weather_only.py          # one-off, ~30 s (auto-runs on first use if skipped)
python main.py                                # LIVE forecast for the next 24 h (default)
python main.py --horizon 1w                   # live, 7 days (also 1h, 15d)
python main.py --replay                       # dashboard historical replay, origin-aware
python scripts/backtest_weather_only.py       # explicit default origin/horizon suite
python scripts/backtest_weather_only.py --origins 2026-03-31T23:00:00 --horizons 1d,1w
python -m uvicorn backend.main:app --reload   # POST /api/run -> live by default
python -m pytest tests -q
```

API: `POST /api/run` with `{"request_id": "...", "horizon": {"name": "1d", "hours": 24}}`
is live by default; add `"mode": "replay"` for an origin-aware backtest. Replay
accepts an optional `"forecast_origin"` ISO timestamp; if omitted, it uses the
existing demonstration origin `2026-05-10T23:00:00`. An origin supplied with
`mode: "live"` is rejected. The response keeps the
existing agent payload shape, reports origin-specific training/model-selection/
calibration cutoffs, and includes observed-weather metrics. `POST /api/live/solar`
keeps working and now uses the weather-only solar model. The live response
continues through the LangGraph pipeline.

## Origin-aware backtesting

Use `python scripts/backtest_weather_only.py --help` to configure the explicit
CPU-sequential experiment. Defaults are one origin, all three targets, and all
four canonical horizons; this is a modest 12-case suite, not hundreds of
automatic retraining jobs. Each case:

1. Filters training data to valid observations at or before the requested
   origin, excluding non-`true_CEA` solar/wind rows and invalid required weather.
2. Splits the pre-origin history chronologically into 70% fitting, 15% model
   selection, and 15% interval calibration data. Hyperparameters are selected
   on validation MAE; the selected models are fit on fitting + validation rows;
   q10/q90 widening uses only the later, still-pre-origin calibration block.
3. Predicts the exact contiguous hourly horizon after the origin and scores
   only against finite, nonnegative, source-eligible target values aligned by
   timestamp. Missing future timestamps, invalid future targets/weather, and
   insufficient history are recorded as failed cases rather than filled.

The training cutoff in each result is the last timestamp used to fit the final
point/quantile models. Validation, calibration, and all training observation
timestamps must precede or equal the origin. The source CSVs do not include
publication/availability timestamps or historical data vintages, so whether a
record with an earlier observation time was actually available at that origin
cannot be verified. The JSON includes model configuration,
split boundaries, data exclusions, predictions, and metrics. CSV outputs are
`summary.csv` and `predictions.csv`. Each run is written to a new directory
under `reports/backtests/`; an existing output directory is never overwritten.

This service uses **observed historical weather** at each target timestamp.
It is a conditional weather-to-energy evaluation, not an operational
weather-to-energy forecast evaluation. Operational evaluation requires archived
weather forecast vintages demonstrably available at each historical origin.
Backtesting fits fresh in-memory models and neither reads nor writes
`models/weather_only/*.joblib` or `reports/weather_only/metrics.json`; normal
live inference and its deployment artifacts remain unchanged.

## Interpreting dashboard evaluation

* Live forecasts display the current issue time in `Asia/Kolkata`, the exact
  forecast origin, the requested future window, and the weather provider/model
  basis. A live-period accuracy score is deliberately not shown: the actual
  generation and demand for future hours are not yet known.
* The dashboard's held-out test MAE, RMSE and R² are historical model scores
  from the models' reserved chronological test split using **observed weather**.
  They assess the weather-to-output mapping, not the accuracy of an Open-Meteo
  forecast or the current live forecast.
* Historical replay reports WAPE and related metrics against the included
  dataset target values over the selected replay window. This is not evidence
  that targets are verified meter measurements. Replay uses observed weather
  (an optimistic input compared with live weather).
* A live-forecast performance claim requires archiving each forecast's weather
  inputs and later scoring the resulting forecast against measured outcomes.
  The dashboard does not fabricate a percentage when that evidence is absent.

Engine selection per agent: `config.engine = "weather_only"` (used by the graph),
`"chronos2"` forces the legacy history-based model, and if unset the engine is
weather-only exactly when `data.future_weather` is supplied.

## Measured accuracy (observed-weather backtest, chronological 70/15/15)

Produced by `scripts/train_weather_only.py`; see `reports/weather_only/metrics.json`.
These use *recorded* weather as input, so they measure the weather->output
mapping with perfect weather, not live accuracy.

| Target | Test R2 | Test MAE (MU) | Climatology baseline R2 | q10-q90 coverage (nominal 80 %) |
|---|---:|---:|---:|---:|
| Solar  | 0.957 | 0.81 | 0.935 | 0.86 |
| Wind   | 0.606 | 1.40 | -0.121 | 0.65 |
| Demand | 0.574 | 1.95 | 0.490 | 0.35 |

## Known limits (read before presenting results)

1. **Live accuracy is lower than the table.** Real forecasts carry weather-forecast
   error. Measure it by archiving Open-Meteo forecasts (or using its historical
   forecast archive) and scoring them against later generation.
2. **Source mismatch.** Training weather is NASA POWER (gridded, 110 sites);
   live weather is Open-Meteo at one point (22.3 N, 71.5 E). Wind speed height may
   also fall back from 50 m (see the weather agent's metadata). Expect some bias.
3. **Growth extrapolation.** Solar grew about 52 %/yr in the data. Forecasts months
   after the last training row rely on that trend; keep appending data and retraining.
4. **Demand and wind generalise worst** (the test window contains an unusually warm
   June and the monsoon). Demand is largely a daily-total signal that hourly weather
   cannot fully explain; its band under-covers on shifted periods.
5. **Solar training stops at 2026-05-30** because June-July irradiance is missing.
   Re-download NASA POWER for 2026-05-31 onward (processing lag) and retrain.
6. Holiday/festival flags only exist up to 2026-12-25 (`calendar_data.py`).
7. `-999` handling for the legacy Chronos path was not changed; it is only safe to
   use `engine = "chronos2"` on date ranges with valid irradiance.
