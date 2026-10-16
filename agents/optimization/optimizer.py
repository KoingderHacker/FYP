from __future__ import annotations

from typing import Any, Sequence
import numpy as np


def _as_float_array(values: Sequence[Any], name: str) -> np.ndarray:
    if values is None:
        raise ValueError(f"{name} is required.")
    # Accept both [1,2,3] and [{"forecast": 1}, ...]
    normalized = []
    for value in values:
        if isinstance(value, dict):
            if "forecast" in value:
                value = value["forecast"]
            elif "value" in value:
                value = value["value"]
            else:
                raise ValueError(f"{name} contains an object without 'forecast' or 'value'.")
        normalized.append(float(value))
    arr = np.asarray(normalized, dtype=float)
    if arr.size == 0:
        raise ValueError(f"{name} cannot be empty.")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN or infinite values.")
    if (arr < 0).any():
        raise ValueError(f"{name} contains negative values.")
    return arr


class GridOptimizer:
    def __init__(
        self,
        battery_capacity_mu: float = 1.0,
        max_charge_rate_mu: float = 0.2,
        max_discharge_rate_mu: float = 0.2,
        initial_soc_pct: float = 50.0,
    ):
        if battery_capacity_mu <= 0:
            raise ValueError("battery_capacity_mu must be > 0.")
        if max_charge_rate_mu < 0 or max_discharge_rate_mu < 0:
            raise ValueError("Battery charge/discharge rates cannot be negative.")
        if not 0 <= initial_soc_pct <= 100:
            raise ValueError("initial_soc_pct must be between 0 and 100.")

        self.battery_capacity_mu = float(battery_capacity_mu)
        self.max_charge_rate_mu = float(max_charge_rate_mu)
        self.max_discharge_rate_mu = float(max_discharge_rate_mu)
        self.initial_soc_pct = float(initial_soc_pct)

    def optimize(
        self,
        solar_predictions,
        wind_predictions,
        demand_predictions,
        timestamps,
        horizon_name: str = "1d",
    ) -> dict[str, Any]:
        solar = _as_float_array(solar_predictions, "solar_predictions")
        wind = _as_float_array(wind_predictions, "wind_predictions")
        demand = _as_float_array(demand_predictions, "demand_predictions")

        if timestamps is None or len(timestamps) == 0:
            raise ValueError("timestamps cannot be empty.")

        n = len(timestamps)
        if not (len(solar) == len(wind) == len(demand) == n):
            raise ValueError(
                f"Forecast length mismatch: solar={len(solar)}, wind={len(wind)}, "
                f"demand={len(demand)}, timestamps={n}."
            )

        renewable = solar + wind
        net_load = demand - renewable

        soc = self.initial_soc_pct / 100.0 * self.battery_capacity_mu
        soc_series, dispatch_series, grid_series, curtailed_series = [], [], [], []

        for i in range(n):
            deficit = float(demand[i] - renewable[i])

            if deficit > 0:
                discharge = min(soc, self.max_discharge_rate_mu, deficit)
                soc -= discharge
                dispatch = discharge
                grid_import = deficit - discharge
                curtailed = 0.0
            else:
                surplus = -deficit
                headroom = self.battery_capacity_mu - soc
                charge = min(headroom, self.max_charge_rate_mu, surplus)
                soc += charge
                dispatch = -charge
                grid_import = 0.0
                curtailed = surplus - charge

            soc_series.append(round(float(soc), 6))
            dispatch_series.append(round(float(dispatch), 6))
            grid_series.append(round(float(grid_import), 6))
            curtailed_series.append(round(float(curtailed), 6))

        total_solar = float(solar.sum())
        total_wind = float(wind.sum())
        total_renewable = float(renewable.sum())
        total_demand = float(demand.sum())
        total_grid = float(sum(grid_series))
        total_curtailed = float(sum(curtailed_series))
        total_discharged = float(sum(x for x in dispatch_series if x > 0))
        total_charged = float(sum(-x for x in dispatch_series if x < 0))

        renewable_penetration = (
            total_renewable / total_demand * 100.0 if total_demand > 0 else 0.0
        )
        effective_coverage = (
            max(0.0, total_renewable - total_curtailed) / total_demand * 100.0
            if total_demand > 0 else 0.0
        )

        peak_deficit = float(np.max(np.maximum(demand - renewable, 0.0)))
        peak_demand = max(float(np.max(demand)), 1e-12)
        peak_deficit_ratio = peak_deficit / peak_demand
        ramp = np.abs(np.diff(net_load)) if n > 1 else np.array([0.0])
        avg_ramp = float(np.mean(ramp))
        stability = max(
            0.0,
            min(
                100.0,
                round(
                    100.0
                    - (peak_deficit_ratio * 40.0)
                    - ((avg_ramp / peak_demand) * 60.0),
                    2,
                ),
            ),
        )

        schedule = []
        for i in range(n):
            schedule.append({
                "timestamp": str(timestamps[i]),
                "solar_mu": round(float(solar[i]), 6),
                "wind_mu": round(float(wind[i]), 6),
                "total_renewable_mu": round(float(renewable[i]), 6),
                "demand_mu": round(float(demand[i]), 6),
                "net_load_mu": round(float(net_load[i]), 6),
                "battery_soc_mu": soc_series[i],
                "battery_dispatch_mu": dispatch_series[i],
                "final_grid_import_mu": grid_series[i],
                "curtailed_surplus_mu": curtailed_series[i],
            })

        return {
            "horizon": horizon_name,
            "horizon_hours": n,
            "summary_metrics": {
                "total_solar_generation_mu": round(total_solar, 4),
                "total_wind_generation_mu": round(total_wind, 4),
                "total_renewable_generation_mu": round(total_renewable, 4),
                "total_electricity_demand_mu": round(total_demand, 4),
                "total_grid_import_required_mu": round(total_grid, 4),
                "total_curtailed_renewable_mu": round(total_curtailed, 4),
                "total_battery_charged_mu": round(total_charged, 4),
                "total_battery_discharged_mu": round(total_discharged, 4),
                "renewable_penetration_pct": round(renewable_penetration, 2),
                "effective_renewable_coverage_pct": round(effective_coverage, 2),
                "grid_stability_index": stability,
            },
            "battery_config": {
                "capacity_mu": self.battery_capacity_mu,
                "max_charge_rate_mu": self.max_charge_rate_mu,
                "max_discharge_rate_mu": self.max_discharge_rate_mu,
                "initial_soc_pct": self.initial_soc_pct,
            },
            "hourly_schedule": schedule,
        }
