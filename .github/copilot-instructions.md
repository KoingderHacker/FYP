# Copilot instructions for this repository

## Project overview
This repo is a renewable-energy forecasting and optimization system. It is organized as a multi-agent pipeline, not a monolithic app:

- `agents/` contains the forecasting and data-cleaning agents: `weather`, `solar`, `wind`, `demand`, and `optimization`.
- `orchestration/graph.py` defines the LangGraph workflow. The three forecasting agents run in parallel, then `prepare_forecasts` normalizes their outputs before `optimization` runs.
- `common/schemas.py` defines the shared request/response contract (`AgentInput`, `AgentOutput`, `Horizon`).
- `backend/main.py` exposes the FastAPI API used by the frontend dashboard.
- `frontend/` is a Vite + React app that calls the backend and visualizes the optimization results.
- Raw datasets live under `data/` and generated reports under `reports/`.

When tracing the system, read these files in order: `orchestration/graph.py`, `orchestration/nodes.py`, `common/schemas.py`, `backend/main.py`, and `frontend/src/App.jsx`.

## Build, test, and lint commands

### Python environment / workflow
```powershell
cd D:\SCET\FYP
python -m pip install -r requirements.txt
python main.py
```

The project uses pytest-style tests. `pytest` is not currently pinned in `requirements.txt`, so install it in the active environment if needed:

```powershell
python -m pip install pytest
```

Run one test file:
```powershell
python -m pytest tests/weather/test_weather_agent.py -q
```

Run the full Python test suite:
```powershell
python -m pytest tests -q
```

### Backend API
```powershell
cd D:\SCET\FYP
python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

### Frontend app
```powershell
cd D:\SCET\FYP\frontend
npm install
npm run dev
npm run build
npm run lint
```

## High-level architecture
The repo is best understood as a data pipeline plus a dashboard:

1. Inputs are passed as a request object with a `location`, `horizon`, and dataset source paths.
2. The LangGraph workflow runs forecasting agents for solar, wind, and demand.
3. `prepare_forecasts` collapses each agent result into plain numeric arrays and timestamps that the optimizer can consume.
4. The optimization node produces a schedule and summary metrics, which the backend returns to the frontend.
5. The React app renders charts and KPI cards from those API results.

Key project-specific file relationships:

- `orchestration/graph.py` decides execution order and parallel fan-out.
- `orchestration/nodes.py` is the integration layer. It maps workflow state into per-agent requests and normalizes outputs.
- `agents/*/agent.py` is the per-agent entry point. Each one accepts a dict or `AgentInput` and returns a `AgentOutput` payload.
- `common/schemas.py` is the contract used by both the orchestration layer and the agents.
- `main.py` at the repo root is the historical/manual workflow runner; `backend/main.py` is the HTTP API entry used by the frontend.

## Key conventions and repo-specific patterns

- Shared payload contract: agent responses must include `status`, `agent`, `request_id`, `result`, and optional `error` metadata matching `common/schemas.py`.
- The orchestration state keys are fixed: `solar_result`, `wind_result`, `demand_result`, `solar_forecast`, `wind_forecast`, `demand_forecast`, `timestamps`, and `optimization_result`.
- `prepare_forecasts` expects all three forecast arrays to be the same length before optimization. If lengths differ, the workflow returns an orchestration error instead of silently proceeding.
- Forecast timestamps are often derived from the `forecast_origin` if an agent does not return them explicitly; do not break that assumption without updating the optimizer contract.
- Data sources are usually specified in `request.data["source"]` or by explicit keys like `solar_source`, `wind_source`, and `demand_source` in the graph state.
- Weather data cleaning rules are intentionally strict: the dataset includes a long trailing `-999` irradiance block and the system does not fabricate values for it; long gaps are flagged for investigation rather than interpolated.
- The frontend assumes optimizer payloads may be nested under `optimization_result.result.summary_metrics` or similar shapes; keep compatibility with those response conventions when changing the response schema.

## Safe change guidance
Keep changes aligned with the existing LangGraph + agent contract model. If you change an agent response shape, update the corresponding transformation in `orchestration/nodes.py` and the frontend consumption logic in `frontend/src/App.jsx`.
