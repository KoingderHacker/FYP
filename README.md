# Weather Agent - Standard FYP Version

This is the first standardized agent in the FYP architecture. It exposes one public interface:

```python
from agents.weather import run_agent
result = run_agent(input_data)
```

The agent performs deterministic diagnostics and cleaning. Gemini is optional and provides a structured decision analysis; it does not directly modify the dataset.

## Run manually

```powershell
pip install -r requirements.txt
python main.py
```

## Run with Gemini

Copy `.env.example` to `.env`, add `GEMINI_API_KEY`, then:

```powershell
python main.py --llm
```

## Test

```powershell
python -m pytest tests/weather -q
```

## Important cleaning rule

The source dataset contains a long trailing `-999` irradiance block (1488 hourly rows / 62 days). The agent does not extrapolate or fabricate those values. Long gaps are flagged for investigation. Only short sentinel gaps are eligible for interpolation.

## LangGraph

LangGraph is intentionally not used inside this agent. Later, `run_agent()` will be wrapped by a LangGraph node, keeping model/cleaning logic independent from orchestration.
