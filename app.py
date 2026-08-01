"""
Streamlit dashboard for the AI-Powered Last-Mile Delivery Optimization
Platform.

Ties together:
  - optimization/ortools_solver.py  -> full-scale (30-stop) route solve
  - optimization/vrp_model.py       -> exact Gurobi solve on a small
                                        subset, for optimality benchmarking
  - models/demand_prediction.py     -> next-14-day package volume forecast
  - models/travel_time_model.py     -> ML travel-time adjustment vs. the
                                        flat-speed assumption
  - visualization/route_map.py      -> interactive Folium route map

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import time

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_folium import st_folium

from models.demand_prediction import (
    DEMAND_MODEL_FACTORIES,
    DEMAND_MODEL_NOTES,
    forecast_next_days,
    generate_demand_history,
    train_demand_model,
)
from models.travel_time_model import (
    TRAVEL_TIME_MODEL_FACTORIES,
    TRAVEL_TIME_MODEL_NOTES,
    predict_travel_time_min,
    train_travel_time_model,
)
from optimization.ortools_solver import load_instance, solve_cvrptw
from optimization.vrp_model import load_instance as load_gurobi_instance
from optimization.vrp_model import solve_demo as solve_gurobi_demo
from optimization.multi_solver_vrp import (
    SOLVER_FACTORIES,
    SOLVER_NOTES,
    haversine_km,
    solve_cvrptw_multi,
)
from visualization.route_map import build_route_map

st.set_page_config(
    page_title="Last-Mile Delivery Optimization",
    page_icon="🚚",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def _cached_load_instance():
    return load_instance()


@st.cache_data(show_spinner=False)
def _cached_demand_forecast(model_type: str = "XGBoost"):
    history = generate_demand_history()
    model, metrics = train_demand_model(history, model_type=model_type)
    forecast = forecast_next_days(model, history, num_days=14)
    return history, forecast, metrics


@st.cache_data(show_spinner=False)
def _cached_travel_time_metrics(model_type: str = "Random Forest"):
    _, metrics = train_travel_time_model(model_type=model_type)
    return metrics


st.title("🚚 AI-Powered Last-Mile Delivery Optimization Platform")
st.caption(
    "Vehicle routing (Gurobi + OR-Tools), demand forecasting, and travel-time "
    "prediction for an Atlanta last-mile delivery network."
)

customers, trucks, depot = _cached_load_instance()

tab_routing, tab_forecast, tab_scenario, tab_solvers, tab_about = st.tabs(
    ["📍 Route Optimization", "📈 Demand Forecast", "🔧 Scenario Analysis", "⚖️ Solver Comparison", "ℹ️ About"]
)

# ----------------------------------------------------------------------
# TAB 1 — Route optimization (full-scale OR-Tools solve + map + KPIs)
# ----------------------------------------------------------------------
with tab_routing:
    col_controls, col_map = st.columns([1, 2])

    with col_controls:
        st.subheader("Fleet & Solve Settings")

        route_solver_choice = st.selectbox(
            "Optimizer", options=["OR-Tools"] + list(SOLVER_FACTORIES.keys()), index=0,
            help="OR-Tools solves the full network fast (heuristic). The exact solvers "
                 "(Gurobi/CPLEX/SCIP/CBC) prove optimality but only on a size-limited "
                 "subset of customers — Gurobi and CPLEX in particular will hard-fail "
                 "past their free license limits.",
        )

        if route_solver_choice == "OR-Tools":
            num_trucks_available = st.slider(
                "Trucks available", min_value=2, max_value=len(trucks), value=len(trucks)
            )
            solve_time_limit = st.slider(
                "OR-Tools solve time limit (sec)", min_value=5, max_value=30, value=15
            )
            n_customers_route = len(customers)
        else:
            st.caption(
                f"{route_solver_choice} is an exact solver — restricted to a smaller "
                f"customer subset here to stay within license limits and keep solve "
                f"time reasonable. {SOLVER_NOTES.get(route_solver_choice, '')}"
            )
            n_customers_route = st.slider(
                "Customers to include", min_value=4, max_value=10, value=6,
            )
            num_trucks_available = len(trucks)
            solve_time_limit = st.slider(
                f"{route_solver_choice} solve time limit (sec)", min_value=5, max_value=60, value=20
            )

        run_solve = st.button("Solve routes", type="primary")

    if run_solve:
        active_customers = customers.head(n_customers_route)
        active_trucks = trucks.head(num_trucks_available)

        if route_solver_choice == "OR-Tools":
            with st.spinner("Solving vehicle routing problem with OR-Tools..."):
                start = time.time()
                result = solve_cvrptw(active_customers, active_trucks, depot, time_limit_sec=solve_time_limit)
                elapsed = time.time() - start
        else:
            with st.spinner(f"Solving exactly with {route_solver_choice}..."):
                start = time.time()
                raw_result = solve_cvrptw_multi(
                    active_customers, active_trucks, depot,
                    solver_name=route_solver_choice, time_limit_sec=solve_time_limit,
                )
                elapsed = time.time() - start

            if raw_result.get("error") or not raw_result.get("routes"):
                result = None
                st.error(
                    f"{route_solver_choice} could not solve this instance: "
                    f"{raw_result.get('error') or raw_result.get('status', 'no feasible solution found')}. "
                    f"Try fewer customers, a longer time limit, or OR-Tools instead."
                )
            else:
                coords = {"depot": (depot["latitude"], depot["longitude"])}
                for _, row in active_customers.iterrows():
                    coords[row["customer_id"]] = (row["latitude"], row["longitude"])

                total_distance_km = 0.0
                for route in raw_result["routes"].values():
                    for a, b in zip(route[:-1], route[1:]):
                        total_distance_km += haversine_km(
                            coords[a][0], coords[a][1], coords[b][0], coords[b][1]
                        )

                result = {
                    "routes": raw_result["routes"],
                    "trucks_used": raw_result["trucks_used"],
                    "total_distance_km": total_distance_km,
                    "total_cost_usd": raw_result["objective"],
                }

        st.session_state["last_result"] = result
        st.session_state["last_elapsed"] = elapsed
        st.session_state["last_customers_used"] = active_customers
        st.session_state["last_solver_used"] = route_solver_choice

    result = st.session_state.get("last_result")

    if result is not None:
        elapsed = st.session_state.get("last_elapsed", 0.0)
        solver_used = st.session_state.get("last_solver_used", "OR-Tools")
        customers_for_map = st.session_state.get("last_customers_used", customers)

        kpi_cols = st.columns(4)
        kpi_cols[0].metric("Trucks used", result["trucks_used"])
        kpi_cols[1].metric("Total distance", f"{result['total_distance_km']:.1f} km")
        kpi_cols[2].metric("Total cost", f"${result['total_cost_usd']:,.2f}")
        kpi_cols[3].metric("Solve time", f"{elapsed:.1f}s")

        if solver_used != "OR-Tools":
            st.caption(f"Solved exactly with {solver_used} on {len(customers_for_map)} customers.")

        with col_map:
            st.subheader("Route Map")
            route_map = build_route_map(result["routes"], customers_for_map, depot)
            st_folium(route_map, width=None, height=500)

        st.subheader("Route Detail")
        for truck_id, route in result["routes"].items():
            st.write(f"**{truck_id}**: {' → '.join(route)}")
    else:
        st.info("Set your fleet size and click **Solve routes** to generate an optimized plan.")

# ----------------------------------------------------------------------
# TAB 2 — Demand forecasting
# ----------------------------------------------------------------------
with tab_forecast:
    st.subheader("14-Day Package Volume Forecast")

    demand_model_choice = st.selectbox(
        "Forecasting model", options=list(DEMAND_MODEL_FACTORIES.keys()), index=0,
        help="Compare a gradient-boosted model against simpler baselines on the same seasonal demand data.",
    )
    st.caption(DEMAND_MODEL_NOTES[demand_model_choice])

    history, forecast, metrics = _cached_demand_forecast(model_type=demand_model_choice)

    forecast_cols = st.columns(3)
    forecast_cols[0].metric("Backtest MAE", f"{metrics['mae']:.0f} pkgs/day")
    forecast_cols[1].metric("Backtest MAPE", f"{metrics['mape'] * 100:.1f}%")
    forecast_cols[2].metric(
        "Next-day forecast", f"{forecast.iloc[0]['package_volume']:.0f} pkgs"
    )

    recent_history = history.tail(60)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=recent_history["date"], y=recent_history["package_volume"],
            name="Historical", line=dict(color="#1F77B4"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast["date"], y=forecast["package_volume"],
            name="Forecast", line=dict(color="#C41E3A", dash="dash"),
        )
    )
    fig.update_layout(
        xaxis_title="Date", yaxis_title="Package volume",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Compare all forecasting models on this data"):
        compare_rows = []
        for name in DEMAND_MODEL_FACTORIES:
            _, _, m = _cached_demand_forecast(model_type=name)
            compare_rows.append(
                {"Model": name, "Backtest MAE (pkgs/day)": round(m["mae"], 1),
                 "Backtest MAPE (%)": round(m["mape"] * 100, 2)}
            )
        st.dataframe(pd.DataFrame(compare_rows), use_container_width=True, hide_index=True)
        st.caption(
            "Linear Regression sets a useful floor — if a more complex model can't beat it "
            "by much, that's a signal the extra complexity isn't earning its keep."
        )

    with st.expander("What this forecast is used for"):
        st.write(
            "Tomorrow's fleet size should follow tomorrow's predicted demand, not "
            "today's — this forecast is what would drive the truck-count slider in "
            "the Scenario Analysis tab in a production deployment, rather than "
            "sizing the fleet off yesterday's volume."
        )

# ----------------------------------------------------------------------
# TAB 3 — Scenario / sensitivity analysis
# ----------------------------------------------------------------------
with tab_scenario:
    st.subheader("Fleet-Size Sensitivity Analysis")
    st.caption(
        "Re-solves the routing problem across a range of fleet sizes to show "
        "the cost/trucks trade-off — the kind of scenario and sensitivity "
        "testing used to evaluate business alternatives before committing "
        "budget to additional vehicles."
    )

    max_trucks_to_test = st.slider("Test fleet sizes up to", 2, len(trucks), len(trucks))
    run_scenario = st.button("Run scenario sweep")

    if run_scenario:
        scenario_rows = []
        progress = st.progress(0.0)
        for i, n_trucks in enumerate(range(1, max_trucks_to_test + 1)):
            active_trucks = trucks.head(n_trucks)
            scenario_result = solve_cvrptw(customers, active_trucks, depot, time_limit_sec=8)
            if scenario_result is not None:
                scenario_rows.append(
                    {
                        "Trucks available": n_trucks,
                        "Feasible": "Yes",
                        "Trucks used": scenario_result["trucks_used"],
                        "Total distance (km)": round(scenario_result["total_distance_km"], 1),
                        "Total cost ($)": round(scenario_result["total_cost_usd"], 2),
                    }
                )
            else:
                scenario_rows.append(
                    {
                        "Trucks available": n_trucks,
                        "Feasible": "No — insufficient capacity/time-window coverage",
                        "Trucks used": None,
                        "Total distance (km)": None,
                        "Total cost ($)": None,
                    }
                )
            progress.progress((i + 1) / max_trucks_to_test)

        scenario_df = pd.DataFrame(scenario_rows)
        st.session_state["scenario_df"] = scenario_df

    scenario_df = st.session_state.get("scenario_df")
    if scenario_df is not None:
        st.dataframe(scenario_df, use_container_width=True, hide_index=True)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=scenario_df["Trucks available"], y=scenario_df["Total cost ($)"],
                name="Total cost", mode="lines+markers",
            )
        )
        fig.update_layout(
            xaxis_title="Trucks available", yaxis_title="Total cost ($)",
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

        infeasible_count = (scenario_df["Feasible"] == "No — insufficient capacity/time-window coverage").sum()
        if infeasible_count > 0:
            st.caption(
                f"{infeasible_count} fleet size(s) below the minimum are infeasible — not enough combined "
                f"capacity and/or time-window coverage to serve all customers. Cost plateaus once enough "
                f"trucks are available to cover the network; additional trucks beyond that add fixed "
                f"dispatch cost without further route savings."
            )
    else:
        st.info("Click **Run scenario sweep** to compare cost across fleet sizes.")

    st.divider()
    st.subheader("Travel-Time Model: ML vs. Flat-Speed Assumption")

    tt_model_choice = st.selectbox(
        "Travel-time model", options=list(TRAVEL_TIME_MODEL_FACTORIES.keys()), index=0,
        help="Compare how well each model captures the non-linear rush-hour congestion pattern.",
    )
    st.caption(TRAVEL_TIME_MODEL_NOTES[tt_model_choice])

    tt_metrics = _cached_travel_time_metrics(model_type=tt_model_choice)
    st.write(
        f"The optimization models above assume a flat {35} km/h average speed. "
        f"The **{tt_model_choice}** model, trained on synthetic historical trips "
        f"(distance, hour of day, day of week), predicts actual travel time "
        f"with a test MAE of **{tt_metrics['mae']:.1f} minutes** "
        f"(R² = {tt_metrics['r2']:.2f}), capturing rush-hour slowdowns the "
        f"flat-speed model misses."
    )

    with st.expander("Compare all travel-time models on this data"):
        tt_compare_rows = []
        for name in TRAVEL_TIME_MODEL_FACTORIES:
            m = _cached_travel_time_metrics(model_type=name)
            tt_compare_rows.append(
                {"Model": name, "Test MAE (min)": round(m["mae"], 2), "Test R²": round(m["r2"], 3)}
            )
        st.dataframe(pd.DataFrame(tt_compare_rows), use_container_width=True, hide_index=True)
        st.caption(
            "Linear Regression underperforms here because rush-hour congestion is a "
            "non-linear bump, not a straight-line trend with hour of day — a good "
            "illustration of when a simple model structurally can't fit the pattern, "
            "no matter how much data it sees."
        )

# ----------------------------------------------------------------------
# TAB 4 — Solver Comparison (Gurobi vs CPLEX vs SCIP vs CBC)
# ----------------------------------------------------------------------
with tab_solvers:
    st.subheader("Exact vs. Heuristic Solver Comparison")
    st.caption(
        "Runs the exact same CVRPTW formulation through five different solvers — "
        "Gurobi, IBM CPLEX, SCIP, and CBC (all exact, via a common PuLP model), plus "
        "Google OR-Tools (heuristic, guided local search) — on the identical small "
        "problem. Useful for comparing not just commercial vs. open-source solvers, "
        "but exact optimality guarantees vs. fast heuristic search."
    )

    col_a, col_b = st.columns([1, 1])
    with col_a:
        n_customers_solver = st.slider(
            "Customers in this small instance", min_value=4, max_value=10, value=6,
            help="Kept small so this stays within Gurobi's/CPLEX's free license limits.",
        )
    with col_b:
        solver_time_limit = st.slider("Per-solver time limit (sec)", 5, 60, 20)

    all_solver_options = list(SOLVER_FACTORIES.keys()) + ["OR-Tools"]
    selected_solvers = st.multiselect(
        "Solvers to compare",
        options=all_solver_options,
        default=all_solver_options,
    )

    with st.expander("What each solver is"):
        for name, note in SOLVER_NOTES.items():
            st.write(f"**{name}** — {note}")
        st.write(
            "**OR-Tools** — Free, open-source (Google); heuristic constraint-programming "
            "solver, not an exact MIP solver. No optimality guarantee, but built to solve "
            "much larger problems fast — this is what powers the full-scale Route "
            "Optimization and Scenario Analysis tabs elsewhere in this app."
        )

    run_solver_comparison = st.button("Run solver comparison", type="primary")

    if run_solver_comparison:
        small_customers = customers.head(n_customers_solver)
        comparison_rows = []
        progress = st.progress(0.0)

        for i, solver_name in enumerate(selected_solvers):
            if solver_name == "OR-Tools":
                ortools_result = solve_cvrptw(
                    small_customers, trucks, depot, time_limit_sec=solver_time_limit,
                )
                if ortools_result is None:
                    comparison_rows.append(
                        {"Solver": "OR-Tools", "Status": "Infeasible", "Objective ($)": None,
                         "Trucks used": None, "Solve time (s)": None}
                    )
                else:
                    comparison_rows.append(
                        {
                            "Solver": "OR-Tools",
                            "Status": "Feasible (heuristic)",
                            "Objective ($)": round(ortools_result["total_cost_usd"], 2),
                            "Trucks used": ortools_result["trucks_used"],
                            "Solve time (s)": None,  # OR-Tools always runs its full time budget internally
                        }
                    )
                progress.progress((i + 1) / max(len(selected_solvers), 1))
                continue

            result = solve_cvrptw_multi(
                small_customers, trucks, depot,
                solver_name=solver_name, time_limit_sec=solver_time_limit,
            )
            if result.get("error"):
                comparison_rows.append(
                    {"Solver": solver_name, "Status": "Not available", "Objective ($)": None,
                     "Trucks used": None, "Solve time (s)": None}
                )
            else:
                comparison_rows.append(
                    {
                        "Solver": solver_name,
                        "Status": result["status"],
                        "Objective ($)": round(result["objective"], 2) if result["objective"] is not None else None,
                        "Trucks used": result.get("trucks_used"),
                        "Solve time (s)": round(result["runtime_sec"], 2),
                    }
                )
            progress.progress((i + 1) / max(len(selected_solvers), 1))

        comparison_df = pd.DataFrame(comparison_rows)
        st.session_state["solver_comparison_df"] = comparison_df

    comparison_df = st.session_state.get("solver_comparison_df")
    if comparison_df is not None:
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)

        exact_df = comparison_df[comparison_df["Solver"] != "OR-Tools"].dropna(subset=["Objective ($)"])
        if len(exact_df) > 1 and exact_df["Objective ($)"].nunique() == 1:
            st.success(
                f"All exact solvers that ran agree on the same optimal cost — "
                f"${exact_df['Objective ($)'].iloc[0]:.2f} — confirming the formulation "
                f"is solver-independent. Solve time is where they actually differ."
            )

        ortools_row = comparison_df[comparison_df["Solver"] == "OR-Tools"]
        if not ortools_row.empty and not exact_df.empty and ortools_row["Objective ($)"].iloc[0] is not None:
            ortools_cost = ortools_row["Objective ($)"].iloc[0]
            optimal_cost = exact_df["Objective ($)"].iloc[0]
            gap_pct = abs(ortools_cost - optimal_cost) / optimal_cost * 100
            st.info(
                f"OR-Tools (heuristic) found a solution at ${ortools_cost:.2f}, "
                f"a {gap_pct:.1f}% gap from the proven optimal cost of ${optimal_cost:.2f} — "
                f"showing how close a fast heuristic gets without a formal optimality guarantee."
            )

        chart_df = comparison_df.dropna(subset=["Solve time (s)"])
        if not chart_df.empty:
            fig = go.Figure()
            fig.add_trace(
                go.Bar(x=chart_df["Solver"], y=chart_df["Solve time (s)"], marker_color="#C41E3A")
            )
            fig.update_layout(
                xaxis_title="Solver", yaxis_title="Solve time (seconds)",
                margin=dict(l=10, r=10, t=10, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)
            if "OR-Tools" in comparison_df["Solver"].values:
                st.caption(
                    "OR-Tools isn't shown on the timing chart — it always runs its full "
                    "search-time budget internally rather than stopping early like an "
                    "exact solver does once it proves optimality, so its wall-clock time "
                    "isn't directly comparable to the exact solvers' solve times."
                )
    else:
        st.info("Click **Run solver comparison** to benchmark Gurobi, CPLEX, SCIP, CBC, and OR-Tools on the same problem.")

# ----------------------------------------------------------------------
# TAB 5 — About
# ----------------------------------------------------------------------
with tab_about:
    st.subheader("About this project")
    st.markdown(
        """
This platform demonstrates an end-to-end operations research + ML workflow
for last-mile delivery, built around a synthetic Atlanta delivery network
(30 customers, 6 trucks, a single depot):

- **Exact optimization (Gurobi):** `optimization/vrp_model.py` formulates the
  full Capacitated VRP with Time Windows (CVRPTW) as a mixed-integer program
  and solves it to proven (near-)optimality on a small instance.
- **Multi-solver comparison (Gurobi / CPLEX / SCIP / CBC):**
  `optimization/multi_solver_vrp.py` builds the same formulation in PuLP so
  it can run against four different solver backends — two commercial
  (Gurobi, IBM CPLEX) and two free/open-source (SCIP, CBC) — to compare
  license limits and solve speed on identical problems.
- **Large-scale optimization (OR-Tools):** `optimization/ortools_solver.py`
  solves the full 30-stop / 6-truck instance in seconds using constraint
  programming + guided local search — the same exact-vs-heuristic trade-off
  that underlies production routing systems like UPS's ORION.
- **Demand forecasting (XGBoost):** `models/demand_prediction.py` forecasts
  daily package volume 14 days out, to inform fleet-sizing decisions ahead
  of time rather than reactively.
- **Travel-time prediction (Random Forest):** `models/travel_time_model.py`
  replaces the flat-speed assumption with a model that captures rush-hour
  congestion effects.
- **Scenario & sensitivity analysis:** the Scenario Analysis tab re-solves
  the routing problem across a range of fleet sizes to show the cost/fleet
  trade-off curve.

See the README for setup instructions and a fuller writeup of each module.
        """
    )
