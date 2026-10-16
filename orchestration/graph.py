from langgraph.graph import StateGraph, START, END

from .state import FYPState
from .nodes import (
    weather_node,
    solar_node,
    wind_node,
    demand_node,
    prepare_forecasts,
    optimization_node,
)


def build_graph():

    graph = StateGraph(FYPState)

    # Nodes
    graph.add_node("weather", weather_node)
    graph.add_node("solar", solar_node)
    graph.add_node("wind", wind_node)
    graph.add_node("demand", demand_node)
    graph.add_node("prepare_forecasts", prepare_forecasts)
    graph.add_node("optimization", optimization_node)

    # Weather first; a failed fetch or unsupported graph replay ends the run
    # explicitly instead of running stale deployment models.
    graph.add_edge(START, "weather")
    graph.add_conditional_edges(
        "weather",
        lambda state: (
            ["solar", "wind", "demand"]
            if state.get("status") != "error"
            else END
        ),
        ["solar", "wind", "demand", END],
    )

    # All three must finish before preparation
    graph.add_edge("solar", "prepare_forecasts")
    graph.add_edge("wind", "prepare_forecasts")
    graph.add_edge("demand", "prepare_forecasts")

    # Prepare → Optimization
    graph.add_conditional_edges(
        "prepare_forecasts",
        lambda state: (
            "optimization"
            if state.get("status") == "ready_for_optimization"
            else END
        ),
        {
            "optimization": "optimization",
            END: END,
        },
    )

    graph.add_edge("optimization", END)

    return graph.compile()