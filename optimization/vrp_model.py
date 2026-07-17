from pathlib import Path

import gurobipy as gp
import pandas as pd
from gurobipy import GRB


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load customer and truck datasets."""
    customers = pd.read_csv(DATA_DIR / "customers.csv")
    trucks = pd.read_csv(DATA_DIR / "trucks.csv")
    return customers, trucks


def build_assignment_model(
    customers: pd.DataFrame,
    trucks: pd.DataFrame,
) -> tuple[gp.Model, gp.tupledict]:
    """
    Assign every customer to exactly one truck while respecting capacity.

    Objective:
        Minimize the number of trucks used.
    """

    model = gp.Model("customer_truck_assignment")

    customer_ids = customers["customer_id"].tolist()
    truck_ids = trucks["truck_id"].tolist()

    demand = customers.set_index("customer_id")["package_weight_kg"].to_dict()
    capacity = trucks.set_index("truck_id")["capacity_kg"].to_dict()

    # x[c, t] = 1 if customer c is assigned to truck t
    x = model.addVars(
        customer_ids,
        truck_ids,
        vtype=GRB.BINARY,
        name="assign",
    )

    # y[t] = 1 if truck t is used
    y = model.addVars(
        truck_ids,
        vtype=GRB.BINARY,
        name="truck_used",
    )

    # Every customer must be assigned to exactly one truck
    model.addConstrs(
        (
            gp.quicksum(x[customer_id, truck_id] for truck_id in truck_ids) == 1
            for customer_id in customer_ids
        ),
        name="assign_once",
    )

    # Truck capacity constraints
    model.addConstrs(
        (
            gp.quicksum(
                demand[customer_id] * x[customer_id, truck_id]
                for customer_id in customer_ids
            )
            <= capacity[truck_id] * y[truck_id]
            for truck_id in truck_ids
        ),
        name="truck_capacity",
    )

    # Minimize the number of trucks used
    model.setObjective(
        gp.quicksum(y[truck_id] for truck_id in truck_ids),
        GRB.MINIMIZE,
    )

    return model, x


def solve_assignment() -> None:
    """Solve and print customer-to-truck assignments."""
    customers, trucks = load_data()
    model, x = build_assignment_model(customers, trucks)

    model.optimize()

    if model.status != GRB.OPTIMAL:
        print("No optimal solution was found.")
        return

    print("\nOptimal customer-to-truck assignment")
    print("-" * 45)

    for truck_id in trucks["truck_id"]:
        assigned_customers = [
            customer_id
            for customer_id in customers["customer_id"]
            if x[customer_id, truck_id].X > 0.5
        ]

        if not assigned_customers:
            continue

        truck_load = customers.loc[
            customers["customer_id"].isin(assigned_customers),
            "package_weight_kg",
        ].sum()

        truck_capacity = trucks.loc[
            trucks["truck_id"] == truck_id,
            "capacity_kg",
        ].iloc[0]

        print(f"\n{truck_id}")
        print(f"Customers assigned: {len(assigned_customers)}")
        print(f"Load: {truck_load} / {truck_capacity} kg")
        print(", ".join(assigned_customers))

    print(f"\nTrucks used: {model.ObjVal:.0f}")


if __name__ == "__main__":
    solve_assignment()