"""
ML-based travel-time prediction.

The optimization models estimate travel time as distance / average
speed, which ignores real-world effects like time-of-day congestion.
This module trains a model that predicts actual travel time from
distance, time of day, and day of week, learned from synthetic
historical trip data with a realistic congestion pattern (AM/PM rush
hour slowdowns). It's meant as a drop-in replacement for the flat-speed
assumption used in optimization/vrp_model.py and ortools_solver.py —
swap the constant AVERAGE_SPEED_KMH conversion for a call to
predict_travel_time_min() once a route's approximate departure time is
known.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42
BASE_SPEED_KMH = 35.0


def _congestion_factor(hour_of_day: np.ndarray) -> np.ndarray:
    """
    Multiplicative slowdown factor by hour: >1 during rush hours,
    close to 1 off-peak. Modeled as two Gaussian bumps around 8am/5pm.
    """
    morning_peak = np.exp(-((hour_of_day - 8.0) ** 2) / (2 * 1.2 ** 2))
    evening_peak = np.exp(-((hour_of_day - 17.0) ** 2) / (2 * 1.5 ** 2))
    return 1.0 + 0.55 * morning_peak + 0.65 * evening_peak


def generate_trip_history(num_trips: int = 6000) -> pd.DataFrame:
    """Synthetic historical trip log: distance, time of day, day of week -> observed travel time."""
    rng = np.random.default_rng(RANDOM_SEED)

    distance_km = rng.uniform(0.5, 25.0, size=num_trips)
    hour_of_day = rng.uniform(6.0, 20.0, size=num_trips)
    day_of_week = rng.integers(0, 7, size=num_trips)

    weekend_relief = np.where(day_of_week >= 5, 0.85, 1.0)
    congestion = _congestion_factor(hour_of_day) * weekend_relief

    base_time_min = distance_km / BASE_SPEED_KMH * 60.0
    noise = rng.normal(loc=1.0, scale=0.08, size=num_trips)
    observed_time_min = base_time_min * congestion * noise

    return pd.DataFrame(
        {
            "distance_km": distance_km,
            "hour_of_day": hour_of_day,
            "day_of_week": day_of_week,
            "travel_time_min": observed_time_min,
        }
    )


FEATURE_COLUMNS = ["distance_km", "hour_of_day", "day_of_week"]


def train_travel_time_model(df: pd.DataFrame | None = None):
    """Train a RandomForest regressor to predict travel time from trip context."""
    if df is None:
        df = generate_trip_history()

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=RANDOM_SEED
    )

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    model.fit(train_df[FEATURE_COLUMNS], train_df["travel_time_min"])

    predictions = model.predict(test_df[FEATURE_COLUMNS])
    mae = mean_absolute_error(test_df["travel_time_min"], predictions)
    r2 = r2_score(test_df["travel_time_min"], predictions)

    return model, {"mae": mae, "r2": r2}


def predict_travel_time_min(
    model, distance_km: float, hour_of_day: float, day_of_week: int
) -> float:
    """Convenience wrapper for a single (or batched) travel-time prediction."""
    row = pd.DataFrame(
        [{"distance_km": distance_km, "hour_of_day": hour_of_day, "day_of_week": day_of_week}]
    )
    return float(model.predict(row[FEATURE_COLUMNS])[0])


if __name__ == "__main__":
    history = generate_trip_history()
    model, metrics = train_travel_time_model(history)

    print(f"Test MAE: {metrics['mae']:.2f} minutes")
    print(f"Test R^2: {metrics['r2']:.3f}")

    example = predict_travel_time_min(model, distance_km=8.0, hour_of_day=8.0, day_of_week=1)
    naive = 8.0 / BASE_SPEED_KMH * 60.0
    print(
        f"\n8km trip at 8am on a Tuesday — "
        f"naive estimate: {naive:.1f} min, ML-adjusted: {example:.1f} min"
    )
