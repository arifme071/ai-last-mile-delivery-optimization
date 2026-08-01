"""
Solver-agnostic exact CVRPTW model, built with PuLP.

This is a companion to vrp_model.py (which uses gurobipy directly).
That version is Gurobi-only; this version uses PuLP as a common modeling
layer so the exact same formulation can run against multiple solver
backends — Gurobi, IBM CPLEX, SCIP, or the free bundled CBC solver —
just by swapping which solver PuLP calls underneath.

Why this exists: being able to point one formulation at four different
solvers and compare their behavior (free vs. commercial, size limits,
solve speed) is a genuinely useful thing to be able to demonstrate,
and it's a realistic reflection of how OR teams evaluate solver
options in practice rather than committing to one vendor by default.

Supported solver keys: "CBC" (always available, no license, bundled
with PuLP), "GUROBI", "CPLEX", "SCIP". CBC is the safe default for a
hosted/deployed app since it never depends on an external license.

Uses the same implicit-subtour-elimination-via-time-windows technique
as vrp_model.py — see that module's docstring for the reasoning.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import pulp

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

AVERAGE_SPEED_KMH = 35.0

SOLVER_FACTORIES = {
    "CBC": lambda time_limit: pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit),
    "GUROBI": lambda time_limit: pulp.GUROBI(msg=False, timeLimit=time_limit),
    "CPLEX": lambda time_limit: pulp.CPLEX_PY(msg=False, timeLimit=time_limit),
    "SCIP": lambda time_limit: pulp.SCIP_PY(msg=False, timeLimit=time_limit),
}

SOLVER_NOTES = {
    "CBC": "Free, open-source, bundled with PuLP — no license, no size limit beyond hardware/time.",
    "GUROBI": "Commercial; free size-limited license caps ~2,000 variables/constraints.",
    "CPLEX": "Commercial (IBM); free Community Edition caps ~1,000 variables/constraints.",
    "SCIP": "Free, open-source, academic-grade solver — no license, no size limit beyond hardware/time.",
}


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlambda / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def load_instance(max_customers: int | None = None):
    customers = pd.read_csv(DATA_DIR / "customers.csv")
    trucks = pd.read_csv(DATA_DIR / "trucks.csv")
    depot = pd.read_csv(DATA_DIR / "depot.csv").iloc[0]
    if max_customers is not None:
        customers = customers.head(max_customers).reset_index(drop=True)
    return customers, trucks, depot


def solve_cvrptw_multi(
    customers: pd.DataFrame,
    trucks: pd.DataFrame,
    depot: pd.Series,
    solver_name: str = "CBC",
    time_limit_sec: int = 30,
):
    """
    Build and solve the exact CVRPTW MIP with the chosen solver backend.
    Returns a dict with routes, objective, trucks_used, solve time, and
    solver status — or None if the chosen solver isn't available in
    this environment (e.g., Gurobi/CPLEX not installed/licensed).
    """
    if solver_name not in SOLVER_FACTORIES:
        raise ValueError(f"Unknown solver '{solver_name}'. Choose from {list(SOLVER_FACTORIES)}.")

    customer_ids = customers["customer_id"].tolist()
    truck_ids = trucks["truck_id"].tolist()
    all_nodes = ["depot"] + customer_ids

    nodes_df = pd.concat(
        [
            pd.DataFrame({"node_id": ["depot"], "latitude": [depot["latitude"]], "longitude": [depot["longitude"]]}),
            customers[["customer_id", "latitude", "longitude"]].rename(columns={"customer_id": "node_id"}),
        ],
        ignore_index=True,
    )

    distance_km, travel_time_min = {}, {}
    for _, ri in nodes_df.iterrows():
        for _, rj in nodes_df.iterrows():
            if ri["node_id"] == rj["node_id"]:
                continue
            d = haversine_km(ri["latitude"], ri["longitude"], rj["latitude"], rj["longitude"])
            distance_km[ri["node_id"], rj["node_id"]] = d
            travel_time_min[ri["node_id"], rj["node_id"]] = d / AVERAGE_SPEED_KMH * 60.0

    demand = customers.set_index("customer_id")["package_weight_kg"].to_dict()
    capacity = trucks.set_index("truck_id")["capacity_kg"].to_dict()
    fixed_cost = trucks.set_index("truck_id")["fixed_cost_usd"].to_dict()
    cost_per_km = trucks.set_index("truck_id")["cost_per_km_usd"].to_dict()
    service_time = customers.set_index("customer_id")["service_time_min"].to_dict()
    tw_start = customers.set_index("customer_id")["time_window_start_min"].to_dict()
    tw_end = customers.set_index("customer_id")["time_window_end_min"].to_dict()

    prob = pulp.LpProblem("cvrptw", pulp.LpMinimize)

    x = {
        (i, j, k): pulp.LpVariable(f"x_{i}_{j}_{k}", cat="Binary")
        for i in all_nodes for j in all_nodes for k in truck_ids if i != j
    }
    y = {k: pulp.LpVariable(f"y_{k}", cat="Binary") for k in truck_ids}
    s = {i: pulp.LpVariable(f"s_{i}", lowBound=0, upBound=depot["closing_time_min"]) for i in all_nodes}

    # Objective: variable distance cost (per truck rate) + fixed dispatch cost
    prob += pulp.lpSum(
        distance_km[i, j] * cost_per_km[k] * x[i, j, k]
        for (i, j, k) in x
    ) + pulp.lpSum(fixed_cost[k] * y[k] for k in truck_ids)

    # Each customer visited exactly once (in and out)
    for c in customer_ids:
        prob += pulp.lpSum(x[i, c, k] for i in all_nodes for k in truck_ids if i != c) == 1
        prob += pulp.lpSum(x[c, j, k] for j in all_nodes for k in truck_ids if j != c) == 1

    # Flow conservation per truck
    for k in truck_ids:
        for node in all_nodes:
            inflow = pulp.lpSum(x[i, node, k] for i in all_nodes if i != node)
            outflow = pulp.lpSum(x[node, j, k] for j in all_nodes if j != node)
            prob += inflow == outflow

    # Depot dispatch linked to y[k]
    for k in truck_ids:
        prob += pulp.lpSum(x["depot", j, k] for j in customer_ids) == y[k]
        prob += pulp.lpSum(x[i, "depot", k] for i in customer_ids) == y[k]

    # Capacity
    for k in truck_ids:
        prob += pulp.lpSum(
            demand[c] * pulp.lpSum(x[i, c, k] for i in all_nodes if i != c)
            for c in customer_ids
        ) <= capacity[k]

    # Time windows + implicit subtour elimination (see module docstring)
    prob += s["depot"] >= depot["opening_time_min"]
    prob += s["depot"] <= depot["closing_time_min"]
    big_m = depot["closing_time_min"] - depot["opening_time_min"]
    for c in customer_ids:
        prob += s[c] >= tw_start[c]
        prob += s[c] <= tw_end[c]
    for i in all_nodes:
        i_service = 0.0 if i == "depot" else service_time[i]
        for j in customer_ids:
            if i == j:
                continue
            for k in truck_ids:
                prob += s[j] >= s[i] + i_service + travel_time_min[i, j] - big_m * (1 - x[i, j, k])

    solver = SOLVER_FACTORIES[solver_name](time_limit_sec)

    start = time.time()
    try:
        prob.solve(solver)
    except Exception as exc:
        return {"error": str(exc), "solver": solver_name}
    elapsed = time.time() - start

    status = pulp.LpStatus[prob.status]
    if status not in ("Optimal", "Not Solved") or prob.objective.value() is None:
        return {"status": status, "solver": solver_name, "routes": {}, "objective": None, "runtime_sec": elapsed}

    routes = {}
    for k in truck_ids:
        arcs = {i: j for (i, j, kk) in x if kk == k and x[i, j, kk].value() and x[i, j, kk].value() > 0.5}
        if "depot" not in arcs:
            continue
        route, current = ["depot"], arcs["depot"]
        while current != "depot":
            route.append(current)
            current = arcs[current]
        route.append("depot")
        routes[k] = route

    return {
        "status": status,
        "solver": solver_name,
        "routes": routes,
        "objective": prob.objective.value(),
        "trucks_used": sum(1 for k in truck_ids if y[k].value() and y[k].value() > 0.5),
        "runtime_sec": elapsed,
    }


if __name__ == "__main__":
    customers, trucks, depot = load_instance(max_customers=8)
    for solver_name in ["CBC", "SCIP", "CPLEX", "GUROBI"]:
        print(f"\n--- {solver_name} ---")
        result = solve_cvrptw_multi(customers, trucks, depot, solver_name=solver_name, time_limit_sec=20)
        if result.get("error"):
            print(f"  Not available: {result['error']}")
        else:
            print(f"  status={result['status']}, objective=${result['objective']:.2f}, "
                  f"trucks_used={result['trucks_used']}, time={result['runtime_sec']:.2f}s")
