import { useState } from "react";
import axios from "axios";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  AreaChart,
  Area,
} from "recharts";
import "./App.css";

const FORECAST_HORIZONS = {
  "1h": 1,
  "1d": 24,
  "1w": 168,
  "15d": 360,
};

const HORIZON_OPTIONS = [
  { value: "1h", label: "1 Hour" },
  { value: "1d", label: "24 Hours" },
  { value: "1w", label: "7 Days" },
  { value: "15d", label: "15 Days" },
];

const formatTimestamp = (value) =>
  value ? String(value).replace("T", " ").replace(/\+00:00$/, " UTC") : "Not reported";

const metricValue = (value, digits = 3) =>
  typeof value === "number" && Number.isFinite(value) ? value.toFixed(digits) : "—";

const chartTimestamp = (value) =>
  value ? String(value).slice(5, 16).replace("T", " ") : "";

function App() {
  const [loading, setLoading] = useState(false);
  const [mode, setMode] = useState("live");
  const [selectedHorizon, setSelectedHorizon] = useState("1d");
  const [data, setData] = useState(null);
  const [error, setError] = useState("");

  const runForecast = async () => {
    setLoading(true);
    setError("");
    setData(null);

    try {
      const horizon = FORECAST_HORIZONS[selectedHorizon];
      const response = await axios.post(
        "/api/run",
        {
          request_id: `dashboard_${Date.now()}`,
          mode,
          location: "Gujarat",
          horizon: {
            name: selectedHorizon,
            hours: horizon,
          },
        }
      );
      setData(response.data);
    } catch (err) {
      console.error(err);
      const detail = err.response?.data?.detail || err.message;
      setError(
        typeof detail === "string"
          ? detail
          : "Failed to run forecasting workflow. Make sure the backend is running at http://127.0.0.1:8000"
      );
    } finally {
      setLoading(false);
    }
  };

  // ── Fix: The optimizer puts summary data under "summary_metrics", not "summary" ──
  const optimization =
    data?.optimization_result?.result?.summary_metrics ||
    data?.optimization_result?.summary_metrics ||
    data?.optimization_result?.result?.summary ||
    data?.optimization_result?.summary;

  const solarForecast =
    data?.solar_result?.result?.forecast ||
    data?.solar_result?.forecast ||
    [];

  const windForecast =
    data?.wind_result?.result?.predictions ||
    data?.wind_result?.predictions ||
    [];

  const demandForecast =
    data?.demand_result?.result?.predictions ||
    data?.demand_result?.predictions ||
    [];

  // Hourly schedule from optimizer (has combined solar+wind+demand per hour)
  const hourlySchedule =
    data?.optimization_result?.result?.hourly_schedule ||
    data?.optimization_result?.hourly_schedule ||
    [];
  const provenance = data?.provenance || {};
  const outputHorizon = provenance.horizon?.name || selectedHorizon;

  /*
   * Combine the three agent outputs into one chart dataset.
   * Prefer the optimizer's hourly_schedule when available (already aligned).
   */
  const chartData =
    hourlySchedule.length > 0
      ? hourlySchedule.map((row) => ({
          hour: chartTimestamp(row.timestamp),
          solar: Number(row.solar_mu ?? 0),
          wind: Number(row.wind_mu ?? 0),
          demand: Number(row.demand_mu ?? 0),
          gridImport: Number(row.final_grid_import_mu ?? 0),
          renewable: Number(row.total_renewable_mu ?? 0),
        }))
      : Array.from(
          { length: Math.max(solarForecast.length, windForecast.length, demandForecast.length) },
          (_, index) => {
          const solar = solarForecast[index];
          const wind = windForecast[index];
          const demand = demandForecast[index];
          const timestamp = solar?.timestamp || wind?.datetime || demand?.datetime;
          return {
            hour: chartTimestamp(timestamp) || `H${index + 1}`,
            solar: Number(solar?.forecast_mu ?? 0),
            wind: Number(wind?.predicted ?? 0),
            demand: Number(demand?.forecast ?? 0),
            gridImport: 0,
            renewable: 0,
          };
        }
      );

  const selectedHorizonLabel =
    HORIZON_OPTIONS.find((option) => option.value === outputHorizon)?.label || "1 Day";
  const hasResults = !!optimization;
  const accuracyRows = [
    ["Solar", data?.solar_result?.result?.metrics, provenance.models?.solar?.historical_test_scores],
    ["Wind", data?.wind_result?.result?.metrics, provenance.models?.wind?.historical_test_scores],
    ["Demand", data?.demand_result?.result?.metrics, provenance.models?.demand?.historical_test_scores],
  ];

  return (
    <div className="app">

      {/* HEADER */}
      <header className="header">
        <div className="header-text">
          <div className="header-logo">⚡</div>
          <div>
            <h1>Renewable Energy AI Dashboard</h1>
            <p className="header-subtitle">
              Multi-Agent Forecasting &amp; Smart Grid Management · Gujarat, India
            </p>
          </div>
        </div>

        <div className="header-actions">
          <div className="control-group">
            <label htmlFor="forecast-mode">Mode</label>
            <select
              id="forecast-mode"
              value={mode}
              onChange={(event) => setMode(event.target.value)}
            >
              <option value="replay">Replay</option>
              <option value="live">Live</option>
            </select>
          </div>

          <div className="control-group">
            <label htmlFor="forecast-horizon">Horizon</label>
            <select
              id="forecast-horizon"
              value={selectedHorizon}
              onChange={(event) => setSelectedHorizon(event.target.value)}
            >
              {HORIZON_OPTIONS.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </div>

          <button
            className="run-btn"
            onClick={runForecast}
            disabled={loading}
          >
            {loading ? (
              <>
                <span className="spinner" /> Running…
              </>
            ) : (
              <>{mode === "live" ? "Run Full Live Forecast" : "Run Historical Replay"}</>
            )}
          </button>
          <p className="mode-help">
            {mode === "live"
              ? "Live uses the current IST time and upcoming Open-Meteo weather; actual accuracy is unavailable until outcomes are observed."
              : "Replay uses a fixed historical period and observed weather; it is not a live forecast."}
          </p>
        </div>
      </header>

      {/* SUCCESS BANNER */}
      {data?.status === "completed" && !loading && (
        <div className="banner success-banner">
          ✓ Full workflow completed &mdash; Solar · Wind · Demand · Optimization
        </div>
      )}

      {/* ERROR */}
      {error && (
        <div className="banner error-banner">
          ⚠ {error}
        </div>
      )}
      {data?.status === "error" && (
        <div className="banner error-banner">
          Workflow did not complete.{" "}
          {(data.errors || []).map((item) => `[${item.agent}] ${item.message}`).join(" · ")}
        </div>
      )}
      {/* INITIAL STATE */}
      {!data && !loading && !error && (
        <div className="welcome">
          <div className="welcome-icon">🌿</div>
          <h2>Ready to run the AI workflow</h2>
          <p>
            Select Live or Historical Replay and a horizon, then run the complete
            Solar, Wind, Demand and Optimization workflow. Live runs use current
            Asia/Kolkata time as their forecast reference.
          </p>
          <div className="agent-pills">
            <span className="pill solar-pill">☀️ Solar Agent</span>
            <span className="pill wind-pill">🌬️ Wind Agent</span>
            <span className="pill demand-pill">⚡ Demand Agent</span>
            <span className="pill opt-pill">🔧 Optimizer</span>
          </div>
        </div>
      )}

      {/* LOADING */}
      {loading && (
        <div className="loading-card">
          <div className="loading-spinner" />
          <div>
            <p className="loading-title">
              Running {mode === "live" ? "live forecast" : "historical replay"}…
            </p>
            <p className="loading-sub">Weather → Solar · Wind · Demand → Optimization</p>
          </div>
        </div>
      )}

      {data && (
        <>
          <section className="panel provenance-panel">
            <div className="panel-header">
              <div>
                <h2>Forecast Provenance</h2>
                <p className="subtitle">
                  Traceable issue time, reference time, data window and model basis.
                </p>
              </div>
              <span className={`mode-badge ${provenance.mode === "live" ? "live" : "replay"}`}>
                {provenance.mode === "live" ? "LIVE FORECAST" : "HISTORICAL REPLAY"}
              </span>
            </div>
            <div className="provenance-grid">
              <div className="provenance-item">
                <span className="stat-label">Run issued at ({provenance.timezone || "local time"})</span>
                <b>{formatTimestamp(provenance.issued_at)}</b>
              </div>
              <div className="provenance-item">
                <span className="stat-label">Forecast reference / origin</span>
                <b>{formatTimestamp(provenance.forecast_origin)}</b>
              </div>
              <div className="provenance-item">
                <span className="stat-label">Forecast window</span>
                <b>
                  {formatTimestamp(provenance.forecast_window?.start)} —{" "}
                  {formatTimestamp(provenance.forecast_window?.end)}
                </b>
              </div>
              <div className="provenance-item">
                <span className="stat-label">Requested horizon / returned points</span>
                <b>
                  {provenance.horizon?.name || outputHorizon} ·{" "}
                  {provenance.horizon?.hours ?? "—"} hours ·{" "}
                  {provenance.forecast_window?.points ?? "—"} points
                </b>
              </div>
              <div className="provenance-item">
                <span className="stat-label">Weather reference</span>
                <b>{provenance.weather?.provider || "Not reported"}</b>
                <small>
                  {provenance.weather?.latitude != null && provenance.weather?.longitude != null
                    ? `${provenance.weather.latitude}, ${provenance.weather.longitude} · `
                    : ""}
                  {typeof provenance.weather?.records === "number"
                    ? `${provenance.weather.records} hourly weather records`
                    : "Historical weather-row count not separately reported"}
                </small>
                {provenance.mode === "live" && (
                  <small>
                    Wind at {provenance.weather?.resolved_wind_height_m ?? "not reported"} m ·
                    clear-sky radiation:{" "}
                    {provenance.weather?.clear_sky_radiation_source || "not reported"}
                  </small>
                )}
              </div>
              <div className="provenance-item">
                <span className="stat-label">Model identifiers / trained through</span>
                {Object.entries(provenance.models || {}).map(([name, model]) => (
                  <small key={name}>
                    {name}: {model.name || "not reported"} · trained through{" "}
                    {model.trained_through || "not reported"}
                  </small>
                ))}
              </div>
            </div>
            <div className="basis-block">
              <h3>What drives these predictions</h3>
              <ul>
                {(provenance.input_basis || []).map((basis) => <li key={basis}>{basis}</li>)}
              </ul>
              <p className="source-note">
                {provenance.dataset_usage} Dataset references:{" "}
                {Object.entries(provenance.sources || {})
                  .map(([name, source]) => `${name}: ${source}`)
                  .join(" · ") || "Not reported"}
              </p>
            </div>
          </section>

          <section className="panel accuracy-panel">
            <h2>Forecast Evaluation</h2>
            <p className="subtitle">
              {provenance.evaluation?.note ||
                "Evaluation metrics are shown only when predictions can be compared with measured outcomes."}
            </p>
            {provenance.mode === "live" && (
              <div className="accuracy-notice">
                Live-period accuracy is not available yet: the future actual values do not exist.
                The historical held-out scores below are provided as evidence about the trained
                models, not as measured accuracy for this live forecast.
              </div>
            )}
            <div className="accuracy-notice">
              Historical test scores come from each model’s reserved test split using observed
              weather. They measure the weather-to-output model on that test period, not the
              error in future weather forecasts. Selected-period scores are available only in
              replay mode.
            </div>
            <div className="table-scroll">
              <table className="accuracy-table">
                <thead>
                  <tr>
                    <th>Series</th>
                    <th>Replay 100 − WAPE (%)</th>
                    <th>Replay WAPE (%)</th>
                    <th>Replay MAE (MU)</th>
                    <th>Replay RMSE (MU)</th>
                    <th>Replay R²</th>
                    <th>Replay points</th>
                    <th>Test MAE (MU)</th>
                    <th>Test RMSE (MU)</th>
                    <th>Test R²</th>
                    <th>Test period</th>
                  </tr>
                </thead>
                <tbody>
                  {accuracyRows.map(([series, metrics, testScores]) => (
                    <tr key={series}>
                      <th>{series}</th>
                      <td>{metricValue(metrics?.wape_accuracy_pct, 1)}</td>
                      <td>{metricValue(metrics?.wape_pct, 1)}</td>
                      <td>{metricValue(metrics?.mae)}</td>
                      <td>{metricValue(metrics?.rmse)}</td>
                      <td>{metricValue(metrics?.r2)}</td>
                      <td>{metrics?.points_scored ?? "—"}</td>
                      <td>{metricValue(testScores?.mae)}</td>
                      <td>{metricValue(testScores?.rmse)}</td>
                      <td>{metricValue(testScores?.r2)}</td>
                      <td>
                        {testScores?.start && testScores?.end
                          ? `${testScores.start} — ${testScores.end} (${testScores.rows ?? "?"} rows)`
                          : "—"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="metric-footnote">
              100 − WAPE is a scale-relative score, not a probability or guaranteed accuracy.
              It is calculated only for the selected historical replay by comparing predictions
              to measured target values. “—” means the corresponding actual values or saved
              test metrics are unavailable.
            </p>
          </section>
        </>
      )}

      {/* RESULTS */}
      {hasResults && (
        <>
          {/* SUMMARY KPI CARDS */}
          <section className="cards">

            <div className="card solar-card">
              <div className="card-icon">☀️</div>
              <span className="card-label">Solar Generation</span>
              <strong className="card-value">
                {optimization.total_solar_generation_mu?.toFixed(2)}
              </strong>
              <span className="card-unit">MU</span>
            </div>

            <div className="card wind-card">
              <div className="card-icon">🌬️</div>
              <span className="card-label">Wind Generation</span>
              <strong className="card-value">
                {optimization.total_wind_generation_mu?.toFixed(2)}
              </strong>
              <span className="card-unit">MU</span>
            </div>

            <div className="card demand-card">
              <div className="card-icon">⚡</div>
              <span className="card-label">Electricity Demand</span>
              <strong className="card-value">
                {optimization.total_electricity_demand_mu?.toFixed(2)}
              </strong>
              <span className="card-unit">MU</span>
            </div>

            <div className="card renew-card">
              <div className="card-icon">♻️</div>
              <span className="card-label">Renewable Penetration</span>
              <strong className="card-value">
                {optimization.renewable_penetration_pct?.toFixed(1)}
              </strong>
              <span className="card-unit">%</span>
            </div>

          </section>

          {/* 24-HOUR FORECAST CHART */}
          <section className="panel">
            <div className="panel-header">
              <div>
                <h2>{selectedHorizonLabel} Energy Forecast</h2>
                <p className="subtitle">Solar, Wind and Electricity Demand (MU/hour)</p>
              </div>
            </div>

            <div className="chart-container">
              <ResponsiveContainer width="100%" height={380}>
                <AreaChart data={chartData} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
                  <defs>
                    <linearGradient id="solarGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#f59e0b" stopOpacity={0.25} />
                      <stop offset="95%" stopColor="#f59e0b" stopOpacity={0} />
                    </linearGradient>
                    <linearGradient id="windGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#2563eb" stopOpacity={0.2} />
                      <stop offset="95%" stopColor="#2563eb" stopOpacity={0} />
                    </linearGradient>
                    <linearGradient id="demandGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#dc2626" stopOpacity={0.15} />
                      <stop offset="95%" stopColor="#dc2626" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                  <XAxis dataKey="hour" tick={{ fontSize: 12 }} />
                  <YAxis tick={{ fontSize: 12 }} unit=" MU" />
                  <Tooltip
                    contentStyle={{ borderRadius: "8px", border: "1px solid #e5e7eb" }}
                    formatter={(value, name) => [`${Number(value).toFixed(3)} MU`, name]}
                  />
                  <Legend />
                  <Area
                    type="monotone"
                    dataKey="solar"
                    name="Solar"
                    stroke="#f59e0b"
                    strokeWidth={2.5}
                    fill="url(#solarGrad)"
                    dot={false}
                  />
                  <Area
                    type="monotone"
                    dataKey="wind"
                    name="Wind"
                    stroke="#2563eb"
                    strokeWidth={2.5}
                    fill="url(#windGrad)"
                    dot={false}
                  />
                  <Area
                    type="monotone"
                    dataKey="demand"
                    name="Demand"
                    stroke="#dc2626"
                    strokeWidth={2.5}
                    fill="url(#demandGrad)"
                    dot={false}
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </section>

          {/* GRID IMPORT VS RENEWABLE CHART */}
          {hourlySchedule.length > 0 && (
            <section className="panel">
              <div className="panel-header">
                <div>
                  <h2>Grid Import vs Renewable Coverage</h2>
                  <p className="subtitle">Hourly grid import and total renewable generation (MU/hour)</p>
                </div>
              </div>

              <div className="chart-container">
                <ResponsiveContainer width="100%" height={280}>
                  <LineChart data={chartData} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                    <XAxis dataKey="hour" tick={{ fontSize: 12 }} />
                    <YAxis tick={{ fontSize: 12 }} unit=" MU" />
                    <Tooltip
                      contentStyle={{ borderRadius: "8px", border: "1px solid #e5e7eb" }}
                      formatter={(value, name) => [`${Number(value).toFixed(3)} MU`, name]}
                    />
                    <Legend />
                    <Line
                      type="monotone"
                      dataKey="renewable"
                      name="Renewable"
                      stroke="#16a34a"
                      strokeWidth={2.5}
                      dot={false}
                    />
                    <Line
                      type="monotone"
                      dataKey="gridImport"
                      name="Grid Import"
                      stroke="#7c3aed"
                      strokeWidth={2.5}
                      dot={false}
                      strokeDasharray="5 3"
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </section>
          )}


          {/* OPTIMIZATION SUMMARY */}
          <section className="panel">
            <h2>Optimization Summary</h2>
            <p className="subtitle">Battery dispatch &amp; grid stability metrics</p>

            <div className="stats">

              <div className="stat-item">
                <span className="stat-label">Grid Import Required</span>
                <b className="stat-value">
                  {optimization.total_grid_import_required_mu?.toFixed(3)}
                  <span className="stat-unit"> MU</span>
                </b>
              </div>

              <div className="stat-item">
                <span className="stat-label">Battery Charged</span>
                <b className="stat-value">
                  {optimization.total_battery_charged_mu?.toFixed(3)}
                  <span className="stat-unit"> MU</span>
                </b>
              </div>

              <div className="stat-item">
                <span className="stat-label">Battery Discharged</span>
                <b className="stat-value">
                  {optimization.total_battery_discharged_mu?.toFixed(3)}
                  <span className="stat-unit"> MU</span>
                </b>
              </div>

              <div className="stat-item">
                <span className="stat-label">Grid Stability Index</span>
                <b className="stat-value stability">
                  {optimization.grid_stability_index?.toFixed(1)}
                  <span className="stat-unit"> / 100</span>
                </b>
              </div>

              <div className="stat-item">
                <span className="stat-label">Total Renewable</span>
                <b className="stat-value green">
                  {optimization.total_renewable_generation_mu?.toFixed(3)}
                  <span className="stat-unit"> MU</span>
                </b>
              </div>

              <div className="stat-item">
                <span className="stat-label">Curtailed Renewable</span>
                <b className="stat-value">
                  {optimization.total_curtailed_renewable_mu?.toFixed(3)}
                  <span className="stat-unit"> MU</span>
                </b>
              </div>

              <div className="stat-item">
                <span className="stat-label">Effective RE Coverage</span>
                <b className="stat-value green">
                  {optimization.effective_renewable_coverage_pct?.toFixed(1)}
                  <span className="stat-unit"> %</span>
                </b>
              </div>

              <div className="stat-item">
                <span className="stat-label">Total Demand</span>
                <b className="stat-value">
                  {optimization.total_electricity_demand_mu?.toFixed(3)}
                  <span className="stat-unit"> MU</span>
                </b>
              </div>

            </div>
          </section>
        </>
      )}

    </div>
  );
}

export default App;