"""
Constraint builders for the Capacitated Vehicle Routing Problem with Time
Windows (CVRPTW), solved exactly with Gurobi.

Formulation notes
------------------
Nodes: node 0 is the depot; nodes 1..n are customers.
x[i, j, k] = 1 if truck k travels directly from node i to node j.
y[k]       = 1 if truck k is dispatched from the depot at all.
s[i]       = continuous service-start time at node i (minutes since 00:00).

Subtour elimination is handled implicitly by the time-window propagation
constraint below (a well-known property of VRPTW formulations): if
x[i, j, k] = 1 then s[j] >= s[i] + service_time[i] + travel_time[i, j].
Because service and travel times are strictly positive, no cycle that
excludes the depot can ever satisfy this chain of inequalities, so no
separate MTZ or lazy subtour-elimination constraints are required.
"""

from __future__ import annotations

import gurobipy as gp
from gurobipy import GRB


def add_degree_constraints(
    model: gp.Model,
    x: gp.tupledict,
    customer_ids: list[str],
    truck_ids: list[str],
) -> None:
    """Every customer is visited exactly once, by exactly one truck."""
    for customer_id in customer_ids:
        model.addConstr(
            gp.quicksum(
                x[i, customer_id, k]
                for i in ["depot"] + customer_ids
                for k in truck_ids
                if i != customer_id
            )
            == 1,
            name=f"visit_in_{customer_id}",
        )
        model.addConstr(
            gp.quicksum(
                x[customer_id, j, k]
                for j in ["depot"] + customer_ids
                for k in truck_ids
                if j != customer_id
            )
            == 1,
            name=f"visit_out_{customer_id}",
        )


def add_flow_conservation(
    model: gp.Model,
    x: gp.tupledict,
    customer_ids: list[str],
    truck_ids: list[str],
) -> None:
    """For every truck, inflow into a node equals outflow from that node."""
    all_nodes = ["depot"] + customer_ids
    for k in truck_ids:
        for node in all_nodes:
            inflow = gp.quicksum(
                x[i, node, k] for i in all_nodes if i != node
            )
            outflow = gp.quicksum(
                x[node, j, k] for j in all_nodes if j != node
            )
            model.addConstr(inflow == outflow, name=f"flow_{node}_{k}")


def add_depot_dispatch_constraints(
    model: gp.Model,
    x: gp.tupledict,
    y: gp.tupledict,
    customer_ids: list[str],
    truck_ids: list[str],
) -> None:
    """
    A truck leaves the depot at most once, and only if it is marked used
    (y[k] = 1). Linking y[k] to actual depot departures keeps the fixed
    dispatch cost honest in the objective.
    """
    for k in truck_ids:
        model.addConstr(
            gp.quicksum(x["depot", j, k] for j in customer_ids) == y[k],
            name=f"depot_out_{k}",
        )
        model.addConstr(
            gp.quicksum(x[i, "depot", k] for i in customer_ids) == y[k],
            name=f"depot_in_{k}",
        )


def add_capacity_constraints(
    model: gp.Model,
    x: gp.tupledict,
    customer_ids: list[str],
    truck_ids: list[str],
    demand: dict[str, float],
    capacity: dict[str, float],
) -> None:
    """Total package weight served by a truck cannot exceed its capacity."""
    all_nodes = ["depot"] + customer_ids
    for k in truck_ids:
        served_weight = gp.quicksum(
            demand[customer_id]
            * gp.quicksum(
                x[i, customer_id, k] for i in all_nodes if i != customer_id
            )
            for customer_id in customer_ids
        )
        model.addConstr(served_weight <= capacity[k], name=f"capacity_{k}")


def add_time_window_constraints(
    model: gp.Model,
    x: gp.tupledict,
    s: gp.tupledict,
    customer_ids: list[str],
    truck_ids: list[str],
    travel_time_min: dict[tuple[str, str], float],
    service_time: dict[str, float],
    time_window_start: dict[str, float],
    time_window_end: dict[str, float],
    depot_open: float,
    depot_close: float,
    big_m: float,
) -> None:
    """
    Time-window feasibility and the implicit subtour-elimination chain
    described in the module docstring.
    """
    all_nodes = ["depot"] + customer_ids

    # Depot service window
    model.addConstr(s["depot"] >= depot_open, name="depot_open")
    model.addConstr(s["depot"] <= depot_close, name="depot_close")

    for customer_id in customer_ids:
        model.addConstr(
            s[customer_id] >= time_window_start[customer_id],
            name=f"tw_start_{customer_id}",
        )
        model.addConstr(
            s[customer_id] <= time_window_end[customer_id],
            name=f"tw_end_{customer_id}",
        )

    for i in all_nodes:
        i_service = 0.0 if i == "depot" else service_time[i]
        for j in customer_ids:
            if i == j:
                continue
            for k in truck_ids:
                model.addConstr(
                    s[j]
                    >= s[i]
                    + i_service
                    + travel_time_min[i, j]
                    - big_m * (1 - x[i, j, k]),
                    name=f"tw_link_{i}_{j}_{k}",
                )
