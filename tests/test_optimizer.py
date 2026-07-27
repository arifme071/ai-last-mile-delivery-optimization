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
from optimization.ortools_solver import load_instance, solve_cvrptw
from optimization.vrp_model import load_instance as load_gurobi_instance
from optimization.vrp_model import solve_demo


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
