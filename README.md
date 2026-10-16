
# FYP — Renewable Energy Multi-Agent Forecasting System

A Python-based renewable energy forecasting project for Gujarat, combining weather processing, solar and wind forecasting, demand estimation, and optimization through a multi-agent workflow. The project includes a FastAPI backend, a React frontend, and a command-line workflow powered by LangGraph.

## Features

- Weather data processing, diagnostics, and cleaning.
- Solar, wind, and electricity-demand forecasting agents.
- Forecast horizons of 1 hour, 1 day, 1 week, and 15 days.
- FastAPI backend for the forecasting application.
- React and Vite frontend for interacting with the application.
- LangGraph-based workflow orchestration.
- Historical replay and backtesting utilities.
- Optional Gemini-based decision analysis where configured.
- Automated tests for agents, backend workflows, and weather-only functionality.

## Project Structure

```text
FYP/
├── agents/                 # Weather, solar, wind, demand, optimization
├── backend/                # FastAPI application and request schemas
├── common/                 # Shared schemas, horizons, forecasting services
├── data/                   # Input datasets
├── docs/                   # Project documentation and audits
├── frontend/               # React + Vite application
├── models/                 # Saved forecasting models
├── orchestration/           # LangGraph workflow
├── sample_inputs/           # Example agent inputs
├── scripts/                 # Training and backtesting scripts
├── tests/                   # Automated tests
├── main.py                  # Command-line workflow
├── requirements.txt         # Python dependencies
└── .env.example             # Environment variable template
```

## Prerequisites

Install the following software:

- Python 3.10 or a compatible version supported by the project dependencies.
- Node.js and npm versions compatible with the frontend dependencies.
- Git.

Check your installed versions:

```powershell
python --version
node --version
npm --version
```

## Installation

Run commands from the project root directory.

### 1. Set up the Python environment

Create a virtual environment:

```powershell
python -m venv .venv
```

Activate it in PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, consult the Python documentation or use a terminal configured to activate the environment without changing system-wide security settings.

Install Python dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Configure environment variables

Copy the example environment file:

```powershell
Copy-Item .env.example .env
```

The example configuration contains:

```dotenv
GEMINI_API_KEY=Your_Api_Key
GEMINI_MODEL=gemini-2.5-flash
```

Replace the placeholder with your own Gemini API key if Gemini functionality is required.

Keep `.env` private. Never commit real API keys, passwords, tokens, or other credentials.

### 3. Install frontend dependencies

Open a terminal at the project root:

```powershell
cd frontend
npm install
```

Return to the project root when needed:

```powershell
cd ..
```

## Running the Application

The web application has two components: a FastAPI backend and a React frontend. Start them in separate terminals.

### Terminal 1 — Start the backend

From the project root, activate the Python virtual environment and run:

```powershell
python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

The backend will be available at:

- API root: http://127.0.0.1:8000/
- Health check: http://127.0.0.1:8000/health
- Interactive API documentation: http://127.0.0.1:8000/docs

The `/health` endpoint should return a healthy status when the application is running.

### Terminal 2 — Start the frontend

From the project root:

```powershell
cd frontend
npm run dev
```

Open the local URL printed by Vite, normally:

http://localhost:5173

The Vite development server proxies requests beginning with `/api` to the backend at `http://127.0.0.1:8000`.

Keep both terminals running while using the web application.

### Build the frontend

To create a production frontend build:

```powershell
cd frontend
npm run build
```

The generated frontend files are placed in `frontend/dist/`.

The build command does not automatically deploy the backend or configure production hosting.

## Running the Command-Line Workflow

The root `main.py` runs the multi-agent forecasting workflow independently of the frontend.

From the project root, with the Python environment activated:

```powershell
python main.py
```

Select a forecast horizon:

```powershell
python main.py --horizon 1h
python main.py --horizon 1d
python main.py --horizon 1w
python main.py --horizon 15d
```

The supported horizons are:

| Argument | Forecast horizon |
|---|---|
| `1h` | 1 hour |
| `1d` | 1 day |
| `1w` | 1 week |
| `15d` | 15 days |

### Historical replay and backtesting

The CLI also supports an observed-weather replay evaluation:

```powershell
python main.py --replay --horizon 1d --origin "2026-05-10T23:00:00"
```

The origin must be compatible with the available historical data and the evaluation requirements.

**Important:** Observed-weather backtesting is not equivalent to operational forecast accuracy using weather forecasts available at the time of prediction. Interpret evaluation results accordingly.

## Testing

Run the automated test suite from the project root:

```powershell
python -m pytest
```

Run the weather agent tests:

```powershell
python -m pytest tests/weather -q
```

Run the weather-only tests:

```powershell
python -m pytest tests/weather_only -q
```

Run frontend linting:

```powershell
cd frontend
npm run lint
```

## Configuration and Data

- Python dependencies are declared in `requirements.txt`.
- Frontend dependencies and scripts are declared in `frontend/package.json`.
- Environment variables are documented in `.env.example`.
- Input datasets are stored under `data/`.
- Saved models are stored under `models/`.
- Example agent inputs are stored under `sample_inputs/`.

Ensure required datasets and model artifacts are available before running workflows that depend on them. Do not assume that every dataset or trained model is automatically downloaded during installation.

Check dataset provenance, licensing, privacy restrictions, and file sizes before publishing data or model artifacts to a public repository.

## Troubleshooting

### Backend does not start

- Confirm that the Python virtual environment is activated.
- Install dependencies from `requirements.txt`.
- Check the terminal output for missing modules or configuration errors.
- Ensure port `8000` is available.

### Frontend cannot communicate with the backend

- Confirm that the backend is running at `http://127.0.0.1:8000`.
- Confirm that the frontend was started using `npm run dev`.
- Inspect the browser developer console and Network tab.
- Verify that the frontend requests use the configured `/api` proxy where appropriate.

### Missing datasets or model files

Check the paths expected by the relevant agent or workflow. Some features require local datasets or model artifacts that may not be included in a fresh checkout.

## Security

- Never commit `.env` or real credentials.
- Review datasets and saved model artifacts before publishing.
- Do not expose development server settings directly to the public internet.
- Configure production CORS, authentication, secrets, and deployment settings before production use.

## License

No license has been specified. Unless a license is added, do not assume that others have permission to redistribute or reuse this project.
