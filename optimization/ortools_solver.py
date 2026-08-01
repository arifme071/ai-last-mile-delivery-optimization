"""
Full-scale Capacitated VRP with Time Windows (CVRPTW), solved with
Google OR-Tools' constraint-programming routing engine.

Why a second solver alongside the Gurobi MIP in vrp_model.py:
Gurobi's free size-limited license caps a model at ~2000
variables/constraints. A full three-index CVRPTW formulation for this
project's realistic scenario (30 customers x 4 trucks) needs roughly
30 * 30 * 4 ~= 3,600 arc variables, which exceeds that limit. OR-Tools'
routing library has no such cap and is purpose-built for VRPs at this
scale — it uses metaheuristics (guided local search) rather than
branch-and-bound, so it trades a global-optimality guarantee for the
ability to solve much larger instances in seconds. In practice this is
exactly the same trade-off UPS's ORION system makes: exact solvers for
small/benchmark instances, fast heuristic solvers for daily production
routing at scale.

Run directly for a demo solve:
    python optimization/ortools_solver.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

AVERAGE_SPEED_KMH = 35.0
SOLVE_TIME_LIMIT_SEC = 15


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlambda / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def load_instance():
    customers = pd.read_csv(DATA_DIR / "customers.csv")
    trucks = pd.read_csv(DATA_DIR / "trucks.csv")
    depot = pd.read_csv(DATA_DIR / "depot.csv").iloc[0]
    return customers, trucks, depot


def build_matrices(customers: pd.DataFrame, depot: pd.Series):
    """Node 0 is the depot; nodes 1..n are customers, in dataframe order."""
    lats = [depot["latitude"]] + customers["latitude"].tolist()
    lons = [depot["longitude"]] + customers["longitude"].tolist()
    n = len(lats)

    distance_km = np.zeros((n, n))
    travel_time_min = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d = haversine_km(lats[i], lons[i], lats[j], lons[j])
            distance_km[i, j] = d
            travel_time_min[i, j] = d / AVERAGE_SPEED_KMH * 60.0

    return distance_km, travel_time_min


def solve_cvrptw(
    customers: pd.DataFrame,
    trucks: pd.DataFrame,
    depot: pd.Series,
    time_limit_sec: int = SOLVE_TIME_LIMIT_SEC,
):
    """
    Solve the full CVRPTW instance with OR-Tools. Returns a dict with
    routes (list of node indices per truck), objective cost, and
    per-truck load/distance/time summaries.
    """
    n_customers = len(customers)
    n_trucks = len(trucks)
    distance_km, travel_time_min = build_matrices(customers, depot)

    demand = [0] + customers["package_weight_kg"].tolist()
    capacities = trucks["capacity_kg"].tolist()
    cost_per_km = trucks["cost_per_km_usd"].tolist()
    fixed_cost = trucks["fixed_cost_usd"].tolist()

    service_time = [0] + customers["service_time_min"].tolist()
    tw_start = [int(depot["opening_time_min"])] + customers["time_window_start_min"].tolist()
    tw_end = [int(depot["closing_time_min"])] + customers["time_window_end_min"].tolist()

    manager = pywrapcp.RoutingIndexManager(n_customers + 1, n_trucks, 0)
    routing = pywrapcp.RoutingModel(manager)

    # --- Distance-based arc cost, scaled by each truck's own $/km rate ---
    def make_distance_callback(truck_idx):
        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return int(distance_km[from_node, to_node] * cost_per_km[truck_idx] * 100)
        return distance_callback

    transit_callback_indices = []
    for truck_idx in range(n_trucks):
        callback_index = routing.RegisterTransitCallback(make_distance_callback(truck_idx))
        transit_callback_indices.append(callback_index)
        routing.SetArcCostEvaluatorOfVehicle(callback_index, truck_idx)

    # Fixed dispatch cost per truck used
    for truck_idx in range(n_trucks):
        routing.SetFixedCostOfVehicle(int(fixed_cost[truck_idx] * 100), truck_idx)

    # --- Capacity dimension ---
    def demand_callback(from_index):
        node = manager.IndexToNode(from_index)
        return demand[node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,
        capacities,
        True,
        "Capacity",
    )

    # --- Time-window dimension (uses a representative travel-time callback) ---
    def time_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(travel_time_min[from_node, to_node] + service_time[from_node])

    time_callback_index = routing.RegisterTransitCallback(time_callback)
    routing.AddDimension(
        time_callback_index,
        120,  # allow up to 2 hours of slack/waiting at a stop
        int(depot["closing_time_min"]),
        False,
        "Time",
    )
    time_dimension = routing.GetDimensionOrDie("Time")

    for node in range(1, n_customers + 1):
        index = manager.NodeToIndex(node)
        time_dimension.CumulVar(index).SetRange(int(tw_start[node]), int(tw_end[node]))

    for truck_idx in range(n_trucks):
        index = routing.Start(truck_idx)
        time_dimension.CumulVar(index).SetRange(
            int(depot["opening_time_min"]), int(depot["closing_time_min"])
        )

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_parameters.time_limit.FromSeconds(time_limit_sec)

    solution = routing.SolveWithParameters(search_parameters)

    if solution is None:
        return None

    node_names = ["depot"] + customers["customer_id"].tolist()
    routes = {}
    total_distance_km = 0.0
    total_cost_usd = 0.0
    trucks_used = 0

    for truck_idx in range(n_trucks):
        index = routing.Start(truck_idx)
        route_nodes = []
        route_distance_km = 0.0
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            route_nodes.append(node_names[node])
            next_index = solution.Value(routing.NextVar(index))
            next_node = manager.IndexToNode(next_index)
            route_distance_km += distance_km[node, next_node]
            index = next_index
        route_nodes.append("depot")

        if len(route_nodes) > 2:  # more than depot -> depot
            trucks_used += 1
            truck_id = trucks.iloc[truck_idx]["truck_id"]
            routes[truck_id] = route_nodes
            total_distance_km += route_distance_km
            total_cost_usd += (
                route_distance_km * cost_per_km[truck_idx] + fixed_cost[truck_idx]
            )

    return {
        "routes": routes,
        "total_distance_km": total_distance_km,
        "total_cost_usd": total_cost_usd,
        "trucks_used": trucks_used,
        "objective_raw": solution.ObjectiveValue(),
    }


if __name__ == "__main__":
    customers, trucks, depot = load_instance()
    result = solve_cvrptw(customers, trucks, depot)

    if result is None:
        print("No feasible solution found.")
    else:
        print(
            f"\nOR-Tools CVRPTW solve — {len(customers)} customers, "
            f"{len(trucks)} trucks available"
        )
        print(f"Trucks used: {result['trucks_used']}")
        print(f"Total distance: {result['total_distance_km']:.1f} km")
        print(f"Total cost: ${result['total_cost_usd']:,.2f}")
        for truck_id, route in result["routes"].items():
            print(f"  {truck_id}: {' -> '.join(route)}")
