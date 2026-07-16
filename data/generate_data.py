from pathlib import Path

import numpy as np
import pandas as pd


RANDOM_SEED = 42

NUM_CUSTOMERS = 30
NUM_TRUCKS = 4

# Approximate downtown Atlanta coordinates
DEPOT_LATITUDE = 33.7490
DEPOT_LONGITUDE = -84.3880


def generate_customers(num_customers: int = NUM_CUSTOMERS) -> pd.DataFrame:
    """Generate synthetic last-mile delivery customers."""

    rng = np.random.default_rng(RANDOM_SEED)

    customer_ids = [
        f"ATL-{customer_number:04d}"
        for customer_number in range(1, num_customers + 1)
    ]

    latitudes = DEPOT_LATITUDE + rng.normal(
        loc=0,
        scale=0.07,
        size=num_customers,
    )

    longitudes = DEPOT_LONGITUDE + rng.normal(
        loc=0,
        scale=0.07,
        size=num_customers,
    )

    package_weights = rng.integers(
        low=2,
        high=26,
        size=num_customers,
    )

    service_times = rng.integers(
        low=5,
        high=16,
        size=num_customers,
    )

    time_window_starts = rng.integers(
        low=8 * 60,
        high=15 * 60,
        size=num_customers,
    )

    time_window_lengths = rng.choice(
        [120, 180, 240],
        size=num_customers,
    )

    time_window_ends = np.minimum(
        time_window_starts + time_window_lengths,
        18 * 60,
    )

    priorities = rng.choice(
        ["Standard", "Priority", "Express"],
        size=num_customers,
        p=[0.60, 0.25, 0.15],
    )

    return pd.DataFrame(
        {
            "customer_id": customer_ids,
            "latitude": latitudes.round(6),
            "longitude": longitudes.round(6),
            "package_weight_kg": package_weights,
            "service_time_min": service_times,
            "time_window_start_min": time_window_starts,
            "time_window_end_min": time_window_ends,
            "priority": priorities,
        }
    )


def generate_trucks(num_trucks: int = NUM_TRUCKS) -> pd.DataFrame:
    """Generate a synthetic delivery fleet."""

    return pd.DataFrame(
        {
            "truck_id": [
                f"Fleet-{truck_number:02d}"
                for truck_number in range(1, num_trucks + 1)
            ],
            "capacity_kg": [150] * num_trucks,
            "fixed_cost_usd": [65.00] * num_trucks,
            "cost_per_km_usd": [0.85] * num_trucks,
            "max_shift_min": [600] * num_trucks,
        }
    )


def generate_depot() -> pd.DataFrame:
    """Generate the Atlanta distribution center record."""

    return pd.DataFrame(
        {
            "depot_id": ["ATL-DC-01"],
            "name": ["Atlanta Distribution Center"],
            "latitude": [DEPOT_LATITUDE],
            "longitude": [DEPOT_LONGITUDE],
            "opening_time_min": [7 * 60],
            "closing_time_min": [19 * 60],
        }
    )


def save_datasets() -> None:
    """Save all generated datasets as CSV files."""

    output_directory = Path(__file__).resolve().parent

    customers = generate_customers()
    trucks = generate_trucks()
    depot = generate_depot()

    customers.to_csv(
        output_directory / "customers.csv",
        index=False,
    )

    trucks.to_csv(
        output_directory / "trucks.csv",
        index=False,
    )

    depot.to_csv(
        output_directory / "depot.csv",
        index=False,
    )

    total_capacity = trucks["capacity_kg"].sum()
    total_demand = customers["package_weight_kg"].sum()

    print("Synthetic logistics datasets created successfully.")
    print(f"Customers: {len(customers)}")
    print(f"Trucks: {len(trucks)}")
    print(f"Total package demand: {total_demand} kg")
    print(f"Total fleet capacity: {total_capacity} kg")

    if total_demand > total_capacity:
        print("Warning: Total demand exceeds fleet capacity.")


if __name__ == "__main__":
    save_datasets()