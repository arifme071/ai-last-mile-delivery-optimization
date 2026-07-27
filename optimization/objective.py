"""
Objective function components for the Capacitated Vehicle Routing Problem
with Time Windows (CVRPTW).

Kept as small, composable functions so the same building blocks can be
reused by both the Gurobi MIP model (vrp_model.py) and any future solver
backend (e.g., OR-Tools) without duplicating cost logic.
"""

from __future__ import annotations

import gurobipy as gp


def fixed_truck_cost_expr(
    y: gp.tupledict,
    fixed_cost: dict[str, float],
) -> gp.LinExpr:
    """Fixed cost incurred for every truck that leaves the depot (y[t] = 1)."""
    return gp.quicksum(fixed_cost[truck_id] * y[truck_id] for truck_id in y.keys())


def build_total_cost_objective(
    x: gp.tupledict,
    y: gp.tupledict,
    distance_km: dict[tuple[str, str], float],
    cost_per_km: dict[str, float],
    fixed_cost: dict[str, float],
    priority_weight: dict[str, float] | None = None,
    service_start: gp.tupledict | None = None,
) -> gp.LinExpr:
    """
    Total cost = variable distance cost (per truck's own $/km rate)
               + fixed dispatch cost per truck used
               - (optional) small priority tie-break term that nudges
                 higher-priority customers earlier in the schedule
                 without overriding genuine cost savings.

    x is indexed [i, j, truck_id]; kept per-truck so a heterogeneous
    fleet (different $/km rates) can share one model.
    """

    variable_cost = gp.quicksum(
        distance_km[i, j] * cost_per_km[truck_id] * x[i, j, truck_id]
        for (i, j, truck_id) in x.keys()
        if i != j
    )

    dispatch_cost = fixed_truck_cost_expr(y, fixed_cost)

    total = variable_cost + dispatch_cost

    if priority_weight is not None and service_start is not None:
        total += gp.quicksum(
            priority_weight.get(customer_id, 0.0) * service_start[customer_id]
            for customer_id in service_start.keys()
        )

    return total
