"""
Daily package-volume demand forecasting for the Atlanta last-mile
delivery network.

Generates a synthetic multi-year daily order history with weekly
seasonality (weekday vs. weekend), an annual peak-season bump (Nov-Dec),
and a mild year-over-year growth trend, then trains a gradient-boosted
model (XGBoost) to forecast the next N days of demand. This is the
forecasting/predictive-analytics component of the platform: it feeds
the number-of-trucks-needed scenario in app.py, since tomorrow's fleet
size should follow tomorrow's predicted demand, not just today's.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from xgboost import XGBRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

RANDOM_SEED = 42
HISTORY_DAYS = 3 * 365  # three years of synthetic daily history


def generate_demand_history(num_days: int = HISTORY_DAYS) -> pd.DataFrame:
    """
    Synthetic daily package-volume series: base level + weekly seasonality
    + holiday-peak bump + linear growth trend + noise.
    """
    rng = np.random.default_rng(RANDOM_SEED)

    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=num_days)
    day_index = np.arange(num_days)

    base_level = 480.0
    weekday_effect = np.array([1.05, 1.10, 1.08, 1.12, 1.20, 0.75, 0.55])  # Mon..Sun
    weekly_multiplier = weekday_effect[dates.dayofweek]

    # Peak season bump: mid-November through December
    is_peak_season = ((dates.month == 11) & (dates.day >= 15)) | (dates.month == 12)
    peak_multiplier = np.where(is_peak_season, 1.55, 1.0)

    # Mild long-run growth trend (~6% per year)
    growth_multiplier = 1 + 0.06 * (day_index / 365.0)

    noise = rng.normal(loc=1.0, scale=0.06, size=num_days)

    volume = (
        base_level
        * weekly_multiplier
        * peak_multiplier
        * growth_multiplier
        * noise
    )

    return pd.DataFrame(
        {
            "date": dates,
            "day_of_week": dates.dayofweek,
            "month": dates.month,
            "is_peak_season": is_peak_season.astype(int),
            "day_index": day_index,
            "package_volume": volume.round(0).astype(int),
        }
    )


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Lag/rolling features a gradient-boosted model can use without a full time-series model."""
    features = df.copy()
    features["lag_1"] = features["package_volume"].shift(1)
    features["lag_7"] = features["package_volume"].shift(7)
    features["rolling_mean_7"] = features["package_volume"].shift(1).rolling(7).mean()
    features["rolling_mean_28"] = features["package_volume"].shift(1).rolling(28).mean()
    return features.dropna().reset_index(drop=True)


FEATURE_COLUMNS = [
    "day_of_week", "month", "is_peak_season", "day_index",
    "lag_1", "lag_7", "rolling_mean_7", "rolling_mean_28",
]

DEMAND_MODEL_FACTORIES = {
    "XGBoost": lambda: XGBRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, random_state=RANDOM_SEED,
    ),
    "Random Forest": lambda: RandomForestRegressor(
        n_estimators=300, max_depth=8, random_state=RANDOM_SEED, n_jobs=-1,
    ),
    "Linear Regression": lambda: LinearRegression(),
}

DEMAND_MODEL_NOTES = {
    "XGBoost": "Gradient-boosted trees — usually the strongest fit on this kind of seasonal, non-linear demand data.",
    "Random Forest": "Ensemble of decision trees — robust, less prone to overfitting than a single boosted model, slightly less precise on smooth trends.",
    "Linear Regression": "Simple linear baseline — fast and interpretable, but can't capture the non-linear peak-season effect well. Useful as a sanity-check floor.",
}


def train_demand_model(df: pd.DataFrame | None = None, model_type: str = "XGBoost"):
    """
    Train a demand forecasting model on a time-ordered train/test split.
    model_type selects the algorithm: "XGBoost" (default), "Random Forest",
    or "Linear Regression" — useful for comparing a gradient-boosted model
    against simpler baselines on the same seasonal, non-linear demand series.
    """
    if df is None:
        df = generate_demand_history()
    if model_type not in DEMAND_MODEL_FACTORIES:
        raise ValueError(f"Unknown model_type '{model_type}'. Choose from {list(DEMAND_MODEL_FACTORIES)}.")

    features = build_features(df)
    split_idx = int(len(features) * 0.85)
    train, test = features.iloc[:split_idx], features.iloc[split_idx:]

    model = DEMAND_MODEL_FACTORIES[model_type]()
    model.fit(train[FEATURE_COLUMNS], train["package_volume"])

    predictions = model.predict(test[FEATURE_COLUMNS])
    mae = mean_absolute_error(test["package_volume"], predictions)
    mape = mean_absolute_percentage_error(test["package_volume"], predictions)

    return model, {"mae": mae, "mape": mape, "test": test, "predictions": predictions, "model_type": model_type}


def forecast_next_days(model, history: pd.DataFrame, num_days: int = 14) -> pd.DataFrame:
    """
    Roll the model forward day by day (each day's forecast becomes an
    input lag feature for the next day), producing a short-term demand
    forecast for fleet-sizing decisions.
    """
    working_history = history.copy()
    forecasts = []

    last_date = working_history["date"].max()
    last_day_index = working_history["day_index"].max()

    for step in range(1, num_days + 1):
        next_date = last_date + pd.Timedelta(days=step)
        next_row = {
            "date": next_date,
            "day_of_week": next_date.dayofweek,
            "month": next_date.month,
            "is_peak_season": int(
                ((next_date.month == 11) and (next_date.day >= 15))
                or (next_date.month == 12)
            ),
            "day_index": last_day_index + step,
        }

        feature_row = pd.DataFrame([next_row])
        feature_row["lag_1"] = working_history["package_volume"].iloc[-1]
        feature_row["lag_7"] = working_history["package_volume"].iloc[-7]
        feature_row["rolling_mean_7"] = working_history["package_volume"].iloc[-7:].mean()
        feature_row["rolling_mean_28"] = working_history["package_volume"].iloc[-28:].mean()

        predicted_volume = float(model.predict(feature_row[FEATURE_COLUMNS])[0])
        next_row["package_volume"] = round(predicted_volume)
        forecasts.append(next_row)

        working_history = pd.concat(
            [working_history, pd.DataFrame([next_row])], ignore_index=True
        )

    return pd.DataFrame(forecasts)[["date", "package_volume", "is_peak_season"]]


if __name__ == "__main__":
    history = generate_demand_history()
    model, metrics = train_demand_model(history)

    print(f"Backtest MAE:  {metrics['mae']:.1f} packages/day")
    print(f"Backtest MAPE: {metrics['mape'] * 100:.2f}%")

    forecast = forecast_next_days(model, history, num_days=14)
    print("\n14-day forward demand forecast:")
    print(forecast.to_string(index=False))
