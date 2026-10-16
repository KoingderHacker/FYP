import argparse
from datetime import datetime
from zoneinfo import ZoneInfo

from common.weather_only.backtest import evaluate_origin, load_backtest_frame
from orchestration.graph import build_graph

HORIZON_HOURS = {"1h": 1, "1d": 24, "1w": 168, "15d": 360}


def main():
    parser = argparse.ArgumentParser(description="FYP multi-agent workflow")
    parser.add_argument(
        "--replay",
        action="store_true",
        help="Run origin-aware observed-weather evaluation instead of live forecasts.",
    )
    parser.add_argument(
        "--horizon",
        choices=["1h", "1d", "1w", "15d"],
        default="1d",
        help="Forecast horizon (default: 1d).",
    )
    parser.add_argument(
        "--origin",
        default="2026-05-10T23:00:00",
        help="Historical local forecast origin used with --replay.",
    )
    args = parser.parse_args()

    if args.replay:
        try:
            frame = load_backtest_frame(
                "data/wind/raw/gujarat_hourly_weather_generation_final.csv",
                "data/demand/raw/gujarat_hourly_demand_estimated.csv",
            )
            print("Origin-aware observed-weather backtest")
            print(f"Origin: {args.origin} · Horizon: {args.horizon}")
            for target in ("solar", "wind", "demand"):
                evaluation = evaluate_origin(frame, target, args.origin, args.horizon)
                metrics = evaluation["metrics"]
                print(
                    f"{target:7s} cutoff={evaluation['training_cutoff']} "
                    f"MAE={metrics['mae']:.4f} RMSE={metrics['rmse']:.4f} "
                    f"R2={metrics['r2']} WAPE={metrics['wape_pct']}"
                )
            print(
                "Observed weather is used; these scores are not operational "
                "weather-forecast accuracy."
            )
            return
        except ValueError as exc:
            parser.error(str(exc))

    graph = build_graph()

    initial_state = {
        "request_id": "fyp_replay_001",
        "timestamp": "2026-10-03T12:00:00",
        "location": "Gujarat",

        "horizon_name": "1d",
        "horizon_hours": 24,

        "forecast_origin": "2026-05-10T23:00:00",
        "mode": "replay",

        "solar_source": (
            "data/wind/raw/"
            "gujarat_hourly_weather_generation_final.csv"
        ),

        "wind_source": (
            "data/wind/raw/"
            "gujarat_hourly_weather_generation_final.csv"
        ),

        "demand_source": (
            "data/demand/raw/"
            "gujarat_hourly_demand_estimated.csv"
        ),

        "status": "starting",
        "errors": [],
    }

    initial_state["horizon_name"] = args.horizon
    initial_state["horizon_hours"] = HORIZON_HOURS[args.horizon]

    if not args.replay:
        # Live mode (default): forecast the hours after "now" from the live
        # weather forecast. The CSVs are not used as model context, so it
        # does not matter that they end on 2026-07-31.
        issued_at = datetime.now(ZoneInfo("Asia/Kolkata"))
        initial_state["mode"] = "live"
        initial_state["timestamp"] = issued_at.isoformat()
        initial_state["forecast_origin"] = issued_at.isoformat()

    print("\n================================")
    print("FYP MULTI-AGENT LANGGRAPH")
    print("================================")

    result = graph.invoke(initial_state)

    

    print("\nWorkflow status:")
    print(result.get("status"))

    if result.get("status") == "completed":

        print("\nAgents executed:")
        print("  Solar        : SUCCESS")
        print("  Wind         : SUCCESS")
        print("  Demand       : SUCCESS")
        print("  Optimization : SUCCESS")

        optimization = result["optimization_result"]

        metrics = (
            optimization
            .get("result", {})
            .get("summary_metrics", {})
        )

        print("\nOptimization Summary")
        print("--------------------")

        for key, value in metrics.items():
            print(f"{key}: {value}")

    else:

        print("\nWorkflow failed.")

        for error in result.get("errors", []):
            print(
                f"[{error['agent']}] "
                f"{error['message']}"
            )


if __name__ == "__main__":
    main()