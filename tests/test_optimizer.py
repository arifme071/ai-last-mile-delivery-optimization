"""
Unit tests for the optimization and ML components.

Run with: pytest tests/test_optimizer.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.demand_prediction import (
    forecast_next_days,
    generate_demand_history,
    train_demand_model,
)
from models.travel_time_model import generate_trip_history, train_travel_time_model
from optimization.ortools_solver import haversine_km, load_instance, solve_cvrptw
from optimization.vrp_model import load_instance as load_gurobi_instance
from optimization.vrp_model import solve_demo
from optimization.multi_solver_vrp import solve_cvrptw_multi


@pytest.fixture(scope="module")
def instance():
    return load_instance()


def test_ortools_solution_visits_every_customer(instance):
    customers, trucks, depot = instance
    result = solve_cvrptw(customers, trucks, depot, time_limit_sec=10)

    assert result is not None

    visited = set()
    for route in result["routes"].values():
        visited.update(node for node in route if node != "depot")

    assert visited == set(customers["customer_id"])


def test_ortools_respects_truck_capacity(instance):
    customers, trucks, depot = instance
    result = solve_cvrptw(customers, trucks, depot, time_limit_sec=10)

    demand = customers.set_index("customer_id")["package_weight_kg"].to_dict()
    capacity = trucks.set_index("truck_id")["capacity_kg"].to_dict()

    for truck_id, route in result["routes"].items():
        load = sum(demand[node] for node in route if node != "depot")
        assert load <= capacity[truck_id], f"{truck_id} overloaded: {load} kg"


def test_gurobi_small_instance_is_feasible():
    result = solve_demo(max_customers=6, time_limit_sec=30)

    assert result is not None
    assert result["objective"] > 0
    assert result["trucks_used"] >= 1


def test_gurobi_solution_visits_every_customer():
    customers, trucks, depot = load_gurobi_instance(max_customers=6)
    result = solve_demo(max_customers=6, time_limit_sec=30)

    visited = set()
    for route in result["routes"].values():
        visited.update(node for node in route if node != "depot")

    assert visited == set(customers["customer_id"])


def test_demand_forecast_is_reasonable():
    history = generate_demand_history(num_days=400)
    model, metrics = train_demand_model(history)

    # A reasonably fit model on synthetic, well-behaved data should
    # comfortably beat naive error levels.
    assert metrics["mape"] < 0.15

    forecast = forecast_next_days(model, history, num_days=7)
    assert len(forecast) == 7
    assert (forecast["package_volume"] > 0).all()


def test_travel_time_model_beats_naive_baseline():
    history = generate_trip_history(num_trips=2000)
    model, metrics = train_travel_time_model(history)

    # RandomForest should explain the bulk of variance in a fully
    # synthetic, low-noise dataset like this one.
    assert metrics["r2"] > 0.85


def test_multi_solver_backends_agree_on_optimal_cost():
    customers, trucks, depot = load_gurobi_instance(max_customers=6)

    objectives = {}
    for solver_name in ["CBC", "SCIP", "GUROBI"]:
        result = solve_cvrptw_multi(customers, trucks, depot, solver_name=solver_name, time_limit_sec=30)
        if not result.get("error") and result.get("objective") is not None:
            objectives[solver_name] = round(result["objective"], 1)

    # At least two solvers should be available in any environment, and
    # any solvers that did run should agree on the same optimal cost —
    # confirming the formulation itself is solver-independent.
    assert len(objectives) >= 2
    assert len(set(objectives.values())) == 1


def test_ortools_route_distance_matches_manual_haversine_sum():
    """
    Regression test for a real bug: the original route-distance extraction
    in ortools_solver.py skipped the final leg back to the depot on every
    route (an `if not routing.IsEnd(next_index)` guard silently dropped
    the last arc), undercounting total distance/cost on every solve. This
    test independently recomputes one route's distance leg-by-leg with
    haversine and checks it matches what solve_cvrptw reports — so this
    class of bug can't silently reappear.
    """
    customers, trucks, depot = load_instance()
    small_customers = customers.head(6)
    result = solve_cvrptw(small_customers, trucks, depot, time_limit_sec=20)

    assert result is not None

    coords = {"depot": (depot["latitude"], depot["longitude"])}
    for _, row in small_customers.iterrows():
        coords[row["customer_id"]] = (row["latitude"], row["longitude"])

    manual_total_km = 0.0
    for route in result["routes"].values():
        for a, b in zip(route[:-1], route[1:]):
            manual_total_km += haversine_km(
                coords[a][0], coords[a][1], coords[b][0], coords[b][1]
            )

    assert abs(manual_total_km - result["total_distance_km"]) < 0.01
