# 🚚 AI-Powered Last-Mile Delivery Optimization Platform

An end-to-end logistics optimization platform combining **mathematical
optimization (Gurobi + OR-Tools)**, **machine learning**, and
**interactive analytics** to plan cost-minimizing, time-window-feasible
delivery routes for a last-mile fleet — built around a synthetic
Atlanta delivery network (30 customers, 4 trucks, 1 depot).

## 🎯 Objectives

- Optimize vehicle routing under capacity and time-window constraints
- Reduce delivery distance and total operating cost
- Improve fleet utilization and support scenario/sensitivity analysis
- Forecast demand and predict travel time to support planning decisions
- Provide an interactive dashboard for exploring trade-offs

## ✅ Implemented Features

- **Exact CVRPTW solve (Gurobi)** — full mixed-integer formulation
  (arc variables, capacity constraints, time windows) solved to proven
  near-optimality on a small instance
- **Large-scale CVRPTW solve (OR-Tools)** — the full 30-stop / 4-truck
  instance solved in seconds via constraint programming + guided local
  search
- **Vehicle routing with capacity constraints and delivery time windows**
- **Demand forecasting (XGBoost)** — 14-day-ahead package volume
  forecast, backtested MAPE ~5%
- **Travel-time prediction (Random Forest)** — replaces the flat-speed
  assumption with a model that captures rush-hour congestion, R² ~0.97
- **Interactive route maps (Folium)** — colored per-truck routes with
  stop-level popups (priority, package weight, time window)
- **Scenario / sensitivity analysis** — re-solves across a range of
  fleet sizes to chart the cost-vs-fleet-size trade-off
- **KPI dashboard (Streamlit + Plotly)** — trucks used, total distance,
  total cost, solve time
- **Unit tests (pytest)** — capacity feasibility and full-coverage
  checks for both solvers, plus sanity checks on the ML models

## 🚀 Possible Next Steps

- Hugging Face Spaces deployment
- Real historical order data in place of the synthetic generator
- Multi-depot routing
- Live traffic API integration for the travel-time model

## 🛠 Tech Stack

**Optimization:** Gurobi Optimizer, Google OR-Tools
**ML:** scikit-learn, XGBoost
**Data:** Python, Pandas, NumPy
**App & Visualization:** Streamlit, Plotly, Folium
**Testing:** pytest

## 📁 Project Structure

```
ai-last-mile-delivery-optimization/
├── app.py                          # Streamlit dashboard (entry point)
├── data/
│   ├── generate_data.py            # Synthetic customers/trucks/depot generator
│   ├── customers.csv
│   ├── trucks.csv
│   └── depot.csv
├── optimization/
│   ├── vrp_model.py                # Exact CVRPTW MIP (Gurobi)
│   ├── ortools_solver.py           # Full-scale CVRPTW (OR-Tools)
│   ├── constraints.py              # Degree/flow/capacity/time-window constraint builders
│   ├── objective.py                # Cost objective builders
│   └── assignment_model.py         # Simpler customer-to-truck bin-packing baseline (Gurobi)
├── models/
│   ├── demand_prediction.py        # XGBoost daily demand forecast
│   └── travel_time_model.py        # RandomForest travel-time prediction
├── visualization/
│   └── route_map.py                # Folium route map builder
├── tests/
│   └── test_optimizer.py           # pytest suite
├── notebooks/
│   └── prototype.ipynb             # exploratory notebook
├── test_gurobi.py                  # Gurobi install/license smoke test
└── requirements.txt
```

## ▶️ Running the project

```bash
pip install -r requirements.txt

# Regenerate the synthetic instance (optional — CSVs are already committed)
python data/generate_data.py

# Exact small-instance solve (Gurobi)
python optimization/vrp_model.py

# Full-scale solve (OR-Tools)
python optimization/ortools_solver.py

# Interactive dashboard
streamlit run app.py

# Tests
pytest tests/test_optimizer.py -v
```

## Why two solvers?

Gurobi's free size-limited license caps a model at ~2,000 variables.
A full three-index CVRPTW formulation for 30 customers × 4 trucks needs
roughly 3,600 arc variables — over that limit. So this project uses
**Gurobi to formulate and solve the exact MIP on a small instance** (to
verify correctness and see how far a true global optimum is from a
heuristic solution), and **OR-Tools' routing engine to solve the full
production-scale instance** with guided local search. That's the same
exact-vs-heuristic trade-off production routing systems (e.g., UPS's
ORION) make at scale: exact solvers validate quality on tractable
benchmarks; fast metaheuristics run the real daily routing.

## 📌 Status

Core optimization, forecasting, and dashboard functionality complete.
See "Possible Next Steps" for planned extensions.
