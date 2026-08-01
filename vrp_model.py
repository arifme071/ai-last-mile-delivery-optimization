"""
Exact Capacitated Vehicle Routing Problem with Time Windows (CVRPTW),
solved with Gurobi.

This is the "prove it's optimal" solver in the project: it formulates the
full mixed-integer program (arc variables, capacity, time windows) and
solves it to global optimality for a small/medium instance. Gurobi's
free size-limited license caps models at ~2000 variables/constraints, so
by default this model runs on a subset of customers (max_customers). For
the full 100-stop / 15-truck scenario used in the Streamlit dashboard, see
optimization/ortools_solver.py, which uses Google OR-Tools' routing
engine — no license limits, and built for exactly this kind of
production-scale routing problem.

Run directly for a demo solve:
    python optimization/vrp_model.py
"""

from __future__ import annotations

import time
from pathlib import Path

import gurobipy as gp
import numpy as np
import pandas as pd
from gurobipy import GRB

from optimization.constraints import (
    add_capacity_constraints,
    add_degree_constraints,
    add_depot_dispatch_constraints,
    add_flow_conservation,
    add_time_window_constraints,
)
from optimization.objective import build_total_cost_objective

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

AVERAGE_SPEED_KMH = 35.0  # typical urban last-mile delivery speed


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    """Great-circle distance between two lat/lon points, in kilometers."""
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlambda / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def build_distance_and_time_matrices(
    nodes: pd.DataFrame,
) -> tuple[dict, dict]:
    """
    nodes must have columns: node_id, latitude, longitude.
    Returns (distance_km, travel_time_min) dicts keyed by (i, j) node_id pairs.
    """
    distance_km: dict[tuple[str, str], float] = {}
    travel_time_min: dict[tuple[str, str], float] = {}

    for _, row_i in nodes.iterrows():
        for _, row_j in nodes.iterrows():
            if row_i["node_id"] == row_j["node_id"]:
                continue
            d = haversine_km(
                row_i["latitude"], row_i["longitude"],
                row_j["latitude"], row_j["longitude"],
            )
            distance_km[row_i["node_id"], row_j["node_id"]] = d
            travel_time_min[row_i["node_id"], row_j["node_id"]] = (
                d / AVERAGE_SPEED_KMH * 60.0
            )

    return distance_km, travel_time_min


def load_instance(max_customers: int | None = None):
    """
    Load customers/trucks/depot CSVs. max_customers truncates the
    instance so it fits Gurobi's size-limited license (used for the
    exact-solve demo; the full instance runs through OR-Tools instead).
    """
    customers = pd.read_csv(DATA_DIR / "customers.csv")
    trucks = pd.read_csv(DATA_DIR / "trucks.csv")
    depot = pd.read_csv(DATA_DIR / "depot.csv").iloc[0]

    if max_customers is not None:
        customers = customers.head(max_customers).reset_index(drop=True)

    return customers, trucks, depot


def build_cvrptw_model(
    customers: pd.DataFrame,
    trucks: pd.DataFrame,
    depot: pd.Series,
):
    """Assemble the full Gurobi CVRPTW model. Returns (model, x, y, s, meta)."""

    customer_ids = customers["customer_id"].tolist()
    truck_ids = trucks["truck_id"].tolist()
    all_nodes = ["depot"] + customer_ids

    nodes_df = pd.concat(
        [
            pd.DataFrame(
                {
                    "node_id": ["depot"],
                    "latitude": [depot["latitude"]],
                    "longitude": [depot["longitude"]],
                }
            ),
            customers[["customer_id", "latitude", "longitude"]].rename(
                columns={"customer_id": "node_id"}
            ),
        ],
        ignore_index=True,
    )

    distance_km, travel_time_min = build_distance_and_time_matrices(nodes_df)

    demand = customers.set_index("customer_id")["package_weight_kg"].to_dict()
    capacity = trucks.set_index("truck_id")["capacity_kg"].to_dict()
    fixed_cost = trucks.set_index("truck_id")["fixed_cost_usd"].to_dict()
    cost_per_km = trucks.set_index("truck_id")["cost_per_km_usd"].to_dict()

    service_time = customers.set_index("customer_id")["service_time_min"].to_dict()
    tw_start = customers.set_index("customer_id")["time_window_start_min"].to_dict()
    tw_end = customers.set_index("customer_id")["time_window_end_min"].to_dict()

    priority_bonus = {"Express": 2.0, "Priority": 1.0, "Standard": 0.0}
    priority_weight = {
        row["customer_id"]: priority_bonus.get(row["priority"], 0.0)
        for _, row in customers.iterrows()
    }

    model = gp.Model("cvrptw")
    model.setParam("OutputFlag", 1)

    x = model.addVars(
        [
            (i, j, k)
            for i in all_nodes
            for j in all_nodes
            for k in truck_ids
            if i != j
        ],
        vtype=GRB.BINARY,
        name="x",
    )
    y = model.addVars(truck_ids, vtype=GRB.BINARY, name="y")
    s = model.addVars(
        all_nodes,
        lb=0,
        ub=depot["closing_time_min"],
        vtype=GRB.CONTINUOUS,
        name="s",
    )

    add_degree_constraints(model, x, customer_ids, truck_ids)
    add_flow_conservation(model, x, customer_ids, truck_ids)
    add_depot_dispatch_constraints(model, x, y, customer_ids, truck_ids)
    add_capacity_constraints(model, x, customer_ids, truck_ids, demand, capacity)
    add_time_window_constraints(
        model, x, s, customer_ids, truck_ids,
        travel_time_min, service_time, tw_start, tw_end,
        depot_open=depot["opening_time_min"],
        depot_close=depot["closing_time_min"],
        big_m=depot["closing_time_min"] - depot["opening_time_min"],
    )

    objective = build_total_cost_objective(
        x, y, distance_km, cost_per_km, fixed_cost,
        priority_weight=priority_weight, service_start=s,
    )
    model.setObjective(objective, GRB.MINIMIZE)

    meta = {
        "customer_ids": customer_ids,
        "truck_ids": truck_ids,
        "distance_km": distance_km,
        "travel_time_min": travel_time_min,
    }
    return model, x, y, s, meta


def extract_routes(model, x, meta) -> dict[str, list[str]]:
    """Reconstruct each truck's ordered stop sequence from the solved x variables."""
    truck_ids = meta["truck_ids"]
    routes: dict[str, list[str]] = {}

    for k in truck_ids:
        arcs = {
            i: j
            for (i, j, kk) in x.keys()
            if kk == k and x[i, j, kk].X > 0.5
        }
        if "depot" not in arcs:
            continue

        route = ["depot"]
        current = arcs["depot"]
        while current != "depot":
            route.append(current)
            current = arcs[current]
        route.append("depot")
        routes[k] = route

    return routes


def solve_demo(max_customers: int = 12, time_limit_sec: int = 60):
    """
    Solve a small CVRPTW instance to (near-)optimality with Gurobi and
    print the resulting routes and cost breakdown.
    """
    customers, trucks, depot = load_instance(max_customers=max_customers)
    model, x, y, s, meta = build_cvrptw_model(customers, trucks, depot)
    model.setParam("TimeLimit", time_limit_sec)
    model.setParam("MIPGap", 0.02)

    start = time.time()
    model.optimize()
    elapsed = time.time() - start

    if model.SolCount == 0:
        print("No feasible solution found within the time limit.")
        return None

    routes = extract_routes(model, x, meta)
    trucks_used = sum(1 for k in meta["truck_ids"] if y[k].X > 0.5)

    print(f"\nSolved in {elapsed:.1f}s | status: {model.Status} | "
          f"objective: ${model.ObjVal:,.2f} | trucks used: {trucks_used}")
    for truck_id, route in routes.items():
        print(f"  {truck_id}: {' -> '.join(route)}")

    return {
        "model": model,
        "routes": routes,
        "objective": model.ObjVal,
        "trucks_used": trucks_used,
        "runtime_sec": elapsed,
    }


if __name__ == "__main__":
    solve_demo()
