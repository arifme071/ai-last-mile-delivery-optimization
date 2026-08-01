# 🚚 AI-Powered Last-Mile Delivery Optimization Platform

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://ai-last-mile-delivery-optimization.streamlit.app/)

An end-to-end logistics optimization platform combining **mathematical
optimization (Gurobi, IBM CPLEX, SCIP, CBC via PuLP + Google OR-Tools)**,
**machine learning**, and **interactive analytics** to plan
cost-minimizing, time-window-feasible delivery routes for a last-mile
fleet — built around a synthetic Atlanta delivery network (100 customers,
15 trucks, 1 depot).

## 🎯 Objectives

- Optimize vehicle routing under capacity and time-window constraints
- Reduce delivery distance and total operating cost
- Improve fleet utilization and support scenario/sensitivity analysis
- Forecast demand and predict travel time to support planning decisions
- Compare exact solver backends (commercial vs. open-source) on the
  same formulation
- Provide an interactive dashboard for exploring trade-offs

## ✅ Implemented Features

- **Exact CVRPTW solve (Gurobi)** — full mixed-integer formulation
  (arc variables, capacity constraints, time windows) solved to proven
  near-optimality on a small instance
- **Multi-solver comparison (Gurobi / IBM CPLEX / SCIP / CBC)** — the
  same CVRPTW formulation built once in PuLP and run against four
  solver backends — two commercial (Gurobi, CPLEX), two free/open-source
  (SCIP, CBC) — to compare license limits and solve speed on identical
  problems. All four converge to the same optimal cost; solve time is
  where they differ (CPLEX/Gurobi < 0.2s, SCIP < 1s, CBC ~11s on the
  benchmark instance)
- **Large-scale CVRPTW solve (OR-Tools)** — the full 100-stop / 6-truck
  instance solved in seconds via constraint programming + guided local
  search
- **Vehicle routing with capacity constraints and delivery time windows**
- **Demand forecasting with model comparison** — XGBoost, Random
  Forest, and Linear Regression, selectable in the dashboard; 14-day-
  ahead package volume forecast, backtested MAPE ~5% (XGBoost/RF)
- **Travel-time prediction with model comparison** — Random Forest,
  Gradient Boosting, and Linear Regression, selectable in the
  dashboard; replaces the flat-speed assumption with a model that
  captures rush-hour congestion (R² ~0.97 for tree-based models vs.
  ~0.88 for the linear baseline)
- **Interactive route maps (Folium)** — colored per-truck routes with
  stop-level popups (priority, package weight, time window)
- **Scenario / sensitivity analysis** — re-solves across a range of
  fleet sizes to chart the cost-vs-fleet-size trade-off; surfaces
  infeasible fleet sizes explicitly (with the reason) rather than
  silently dropping them
- **KPI dashboard (Streamlit + Plotly)** — trucks used, total distance,
  total cost, solve time
- **Unit tests (pytest)** — capacity feasibility and full-coverage
  checks for both routing solvers, cross-solver optimal-cost agreement,
  plus sanity checks on all ML models

## 🚀 Possible Next Steps

- Stochastic demand / chance-constrained formulation
- Split-delivery VRP extension
- Robust optimization against travel-time uncertainty
- Real historical order data in place of the synthetic generator
- Multi-depot routing
- Live traffic API integration for the travel-time model

## 🛠 Tech Stack

**Optimization:** Gurobi Optimizer, IBM CPLEX, SCIP, PuLP, Google OR-Tools
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
│   ├── trucks.csv                  # 6-truck fleet
│   └── depot.csv
├── optimization/
│   ├── vrp_model.py                # Exact CVRPTW MIP (Gurobi)
│   ├── multi_solver_vrp.py         # Same CVRPTW MIP via PuLP — Gurobi/CPLEX/SCIP/CBC
│   ├── ortools_solver.py           # Full-scale CVRPTW (OR-Tools)
│   ├── constraints.py              # Degree/flow/capacity/time-window constraint builders
│   ├── objective.py                # Cost objective builders
│   └── assignment_model.py         # Simpler customer-to-truck bin-packing baseline (Gurobi)
├── models/
│   ├── demand_prediction.py        # XGBoost / Random Forest / Linear demand forecast
│   └── travel_time_model.py        # Random Forest / Gradient Boosting / Linear travel-time model
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

# Exact small-instance solve (Gurobi only)
python optimization/vrp_model.py

# Exact small-instance solve, compare all 4 solver backends
python optimization/multi_solver_vrp.py

# Full-scale solve (OR-Tools)
python optimization/ortools_solver.py

# Demand forecast model comparison
python models/demand_prediction.py

# Travel-time model comparison
python models/travel_time_model.py

# Interactive dashboard
streamlit run app.py

# Tests
pytest tests/test_optimizer.py -v
```

## Why more than one solver?

Gurobi's and CPLEX's free size-limited licenses cap a model at roughly
1,000–2,000 variables. A full three-index CVRPTW formulation for 100
customers × 15 trucks needs roughly 151,500 arc variables — vastly over
both limits. So this project uses two complementary strategies:

1. **Exact solve on a small instance, across four backends.**
   `multi_solver_vrp.py` builds one formulation in PuLP and solves it
   with Gurobi, CPLEX, SCIP, and CBC. All four agree on the same
   optimal cost — confirming the formulation itself is correct,
   independent of solver — while differing meaningfully in speed
   (commercial solvers fastest, free CBC solver slowest but still
   correct).
2. **Heuristic solve on the full-scale instance.** `ortools_solver.py`
   solves the full 100-stop / 15-truck problem in seconds using
   constraint programming + guided local search, trading a formal
   optimality guarantee for the ability to solve at production scale.

That's the same exact-vs-heuristic trade-off production routing
systems (e.g., UPS's ORION) make: exact solvers validate quality on
tractable benchmarks; fast metaheuristics run the real daily routing.

## 📌 Status

Core optimization (single- and multi-solver), forecasting with model
comparison, and dashboard functionality complete. See "Possible Next
Steps" for planned extensions.
