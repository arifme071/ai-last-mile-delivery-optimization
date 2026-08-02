"""
PDF report generation for the last-mile delivery optimization dashboard.

Builds downloadable PDF reports from whatever the user has actually run
in the app — either a single tab's results (individual report) or a
combined report across every tab that has results available. Uses
fpdf2, a pure-Python PDF library with no external system dependency
(unlike wkhtmltopdf), which keeps this reliable on Streamlit Cloud.

All builder functions take plain Python data (dicts/DataFrames already
computed by app.py) rather than Streamlit objects, so this module has
no dependency on Streamlit and can be tested/run standalone.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
from fpdf import FPDF

BRAND_COLOR = (196, 30, 58)  # matches the app's red accent
TEXT_COLOR = (30, 30, 30)
MUTED_COLOR = (100, 100, 100)


class Report(FPDF):
    """FPDF subclass with a consistent header/footer for every page."""

    def header(self):
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(*BRAND_COLOR)
        self.cell(0, 10, "AI-Powered Last-Mile Delivery Optimization Platform", ln=True)
        self.set_font("Helvetica", "", 9)
        self.set_text_color(*MUTED_COLOR)
        self.cell(0, 6, f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
        self.set_draw_color(*BRAND_COLOR)
        self.line(10, self.get_y() + 2, 200, self.get_y() + 2)
        self.ln(8)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(*MUTED_COLOR)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")


def _section_title(pdf: Report, title: str):
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(*TEXT_COLOR)
    pdf.cell(0, 9, title, ln=True)
    pdf.ln(1)


def _kpi_row(pdf: Report, kpis: dict[str, str]):
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(*TEXT_COLOR)
    col_width = 190 / max(len(kpis), 1)
    for label, value in kpis.items():
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(*MUTED_COLOR)
        x, y = pdf.get_x(), pdf.get_y()
        pdf.multi_cell(col_width, 5, label, align="L")
        pdf.set_xy(x, y + 5)
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(*TEXT_COLOR)
        pdf.set_xy(x, y + 5)
        pdf.cell(col_width, 7, value, align="L")
        pdf.set_xy(x + col_width, y)
    pdf.ln(14)


def _dataframe_table(pdf: Report, df: pd.DataFrame, max_rows: int = 25):
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_fill_color(240, 240, 240)
    pdf.set_text_color(*TEXT_COLOR)
    col_width = 190 / len(df.columns)
    for col in df.columns:
        pdf.cell(col_width, 7, str(col)[:22], border=1, fill=True)
    pdf.ln()

    pdf.set_font("Helvetica", "", 8)
    for _, row in df.head(max_rows).iterrows():
        for val in row:
            pdf.cell(col_width, 6, str(val)[:22], border=1)
        pdf.ln()

    if len(df) > max_rows:
        pdf.set_font("Helvetica", "I", 8)
        pdf.set_text_color(*MUTED_COLOR)
        pdf.cell(0, 6, f"... {len(df) - max_rows} more row(s) not shown", ln=True)
    pdf.ln(4)


def _text_block(pdf: Report, text: str):
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(*TEXT_COLOR)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(0, 5, text)
    pdf.ln(3)


# ----------------------------------------------------------------------
# Section builders — one per tab. Each takes the already-computed
# result data (as stored in st.session_state by app.py) and appends
# a section to the given Report object.
# ----------------------------------------------------------------------

def add_route_optimization_section(pdf: Report, result: dict, solver_used: str, elapsed: float, num_customers: int):
    _section_title(pdf, "Route Optimization")
    _text_block(pdf, f"Solver: {solver_used}  |  Customers included: {num_customers}")
    _kpi_row(pdf, {
        "Trucks used": str(result["trucks_used"]),
        "Total distance": f"{result['total_distance_km']:.1f} km",
        "Total cost": f"${result['total_cost_usd']:,.2f}",
        "Solve time": f"{elapsed:.1f}s",
    })
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 7, "Route Detail", ln=True)
    pdf.set_font("Helvetica", "", 8)
    for truck_id, route in result["routes"].items():
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(0, 5, f"{truck_id}: {' -> '.join(route)}")
    pdf.ln(4)


def add_demand_forecast_section(pdf: Report, model_type: str, metrics: dict, forecast_df: pd.DataFrame):
    _section_title(pdf, "Demand Forecast")
    _text_block(pdf, f"Model: {model_type}")
    _kpi_row(pdf, {
        "Backtest MAE": f"{metrics['mae']:.0f} pkgs/day",
        "Backtest MAPE": f"{metrics['mape'] * 100:.1f}%",
        "Next-day forecast": f"{forecast_df.iloc[0]['package_volume']:.0f} pkgs",
    })
    table_df = forecast_df.copy()
    table_df["date"] = table_df["date"].dt.strftime("%Y-%m-%d")
    _dataframe_table(pdf, table_df[["date", "package_volume"]])


def add_scenario_analysis_section(pdf: Report, scenario_df: pd.DataFrame):
    _section_title(pdf, "Scenario / Sensitivity Analysis")
    _text_block(pdf, "Fleet-size sweep - cost and feasibility across a range of truck counts.")
    _dataframe_table(pdf, scenario_df)


def add_solver_comparison_section(pdf: Report, comparison_df: pd.DataFrame):
    _section_title(pdf, "Solver Comparison")
    _text_block(pdf, "Same CVRPTW formulation solved via multiple backends on an identical instance.")
    _dataframe_table(pdf, comparison_df)


def add_travel_time_section(pdf: Report, model_type: str, metrics: dict):
    _section_title(pdf, "Travel-Time Model")
    _kpi_row(pdf, {
        "Model": model_type,
        "Test MAE": f"{metrics['mae']:.1f} min",
        "Test R^2": f"{metrics['r2']:.3f}",
    })


# ----------------------------------------------------------------------
# Top-level report builders
# ----------------------------------------------------------------------

def build_report(section_callbacks: list) -> bytes:
    """
    section_callbacks: list of zero-arg callables, each of which calls
    one of the add_*_section() functions above on a shared Report
    instance. Returns the finished PDF as bytes, ready for
    st.download_button.
    """
    pdf = Report()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    if not section_callbacks:
        _text_block(pdf, "No results available yet - run a solve or forecast first.")

    for callback in section_callbacks:
        callback(pdf)

    return bytes(pdf.output())
