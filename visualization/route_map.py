"""
Interactive route map rendering with Folium.

Given a routes dict (truck_id -> ordered list of node_ids, as produced
by optimization/ortools_solver.py or optimization/vrp_model.py) plus the
customer/depot coordinate tables, renders a colored Folium map: one
color per truck route, the depot marked distinctly, and popups showing
package weight / time window / priority per stop.
"""

from __future__ import annotations

import folium
import pandas as pd

ROUTE_COLORS = [
    "#C41E3A",  # UPS-adjacent red-brown
    "#1F77B4",
    "#2CA02C",
    "#9467BD",
    "#FF7F0E",
    "#17BECF",
    "#8C564B",
    "#E377C2",
]


def build_route_map(
    routes: dict[str, list[str]],
    customers: pd.DataFrame,
    depot: pd.Series,
) -> folium.Map:
    """Return a Folium map object with one polyline + marker set per truck route."""
    coords_lookup = customers.set_index("customer_id")[["latitude", "longitude"]].to_dict("index")
    coords_lookup["depot"] = {"latitude": depot["latitude"], "longitude": depot["longitude"]}

    priority_lookup = customers.set_index("customer_id")["priority"].to_dict()
    weight_lookup = customers.set_index("customer_id")["package_weight_kg"].to_dict()
    tw_start_lookup = customers.set_index("customer_id")["time_window_start_min"].to_dict()
    tw_end_lookup = customers.set_index("customer_id")["time_window_end_min"].to_dict()

    fmap = folium.Map(
        location=[depot["latitude"], depot["longitude"]],
        zoom_start=11,
        tiles="cartodbpositron",
    )

    folium.Marker(
        location=[depot["latitude"], depot["longitude"]],
        popup="Atlanta Distribution Center (Depot)",
        icon=folium.Icon(color="black", icon="warehouse", prefix="fa"),
    ).add_to(fmap)

    for route_index, (truck_id, route_nodes) in enumerate(routes.items()):
        color = ROUTE_COLORS[route_index % len(ROUTE_COLORS)]

        path_coords = [
            [coords_lookup[node]["latitude"], coords_lookup[node]["longitude"]]
            for node in route_nodes
        ]
        folium.PolyLine(
            path_coords,
            color=color,
            weight=4,
            opacity=0.8,
            tooltip=f"{truck_id} route",
        ).add_to(fmap)

        for stop_number, node in enumerate(route_nodes):
            if node == "depot":
                continue

            start_min = tw_start_lookup[node]
            end_min = tw_end_lookup[node]
            popup_html = (
                f"<b>{node}</b> ({truck_id}, stop #{stop_number})<br>"
                f"Priority: {priority_lookup[node]}<br>"
                f"Package: {weight_lookup[node]} kg<br>"
                f"Time window: {start_min // 60:02d}:{start_min % 60:02d}"
                f"-{end_min // 60:02d}:{end_min % 60:02d}"
            )
            folium.CircleMarker(
                location=[coords_lookup[node]["latitude"], coords_lookup[node]["longitude"]],
                radius=6,
                color=color,
                fill=True,
                fill_opacity=0.9,
                popup=folium.Popup(popup_html, max_width=250),
            ).add_to(fmap)

    return fmap


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from optimization.ortools_solver import load_instance, solve_cvrptw

    customers, trucks, depot = load_instance()
    result = solve_cvrptw(customers, trucks, depot)

    output_map = build_route_map(result["routes"], customers, depot)
    output_path = Path(__file__).resolve().parents[1] / "route_map_demo.html"
    output_map.save(str(output_path))
    print(f"Saved demo route map to {output_path}")
