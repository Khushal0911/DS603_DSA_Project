"""Ahmedabad route planner - Streamlit frontend.  Run:  streamlit run app.py"""
import numpy as np
import streamlit as st
import folium
from streamlit_folium import st_folium

import travel_model as tm
from osm_loader import CENTER, demo_city, load_ahmedabad, to_km, to_latlon
from planner import plan, route_graph_for_locations, snap_to_road

st.set_page_config(page_title="Ahmedabad Route Planner", layout="wide")

PRESETS = {
    "Kalupur Railway Station": (23.0268, 72.6013),
    "Lal Darwaja": (23.0240, 72.5812),
    # Centre of the Law Garden park (the previous longitude pointed west of
    # the landmark, near Panchvati Cross Roads).
    "Law Garden": (23.0264, 72.5608),
    "Sabarmati Ashram": (23.0607, 72.5800),
    "Gujarat University": (23.0343, 72.5461),
    "Vastrapur Lake": (23.0372, 72.5276),
    "Maninagar Station": (22.9965, 72.6040),
    "SVPI Airport": (23.0772, 72.6347),
}


@st.cache_resource(show_spinner="Loading road network and training the travel-time model...")
def get_city(source):
    note = None
    if source.startswith("Real"):
        try:
            g = load_ahmedabad()
        except Exception as ex:
            note = f"Could not load OpenStreetMap data ({type(ex).__name__}); using the offline demo network."
            g = demo_city()
    else:
        g = demo_city()
    tm.CENTER_SCALE = 3.5
    model, metrics = tm.train_model(g)
    segments = {}
    for edge in g.edges:
        key = (min(edge.u, edge.v), max(edge.u, edge.v))
        segments.setdefault(key, (g.pos[key[0]], g.pos[key[1]]))
    segment_nodes = np.asarray(list(segments), dtype=int)
    segment_xy = np.asarray(list(segments.values()), dtype=float)
    return g, model, metrics, segment_nodes, segment_xy, note


@st.cache_data(show_spinner=False)
def get_table(source, hour):
    g, model, *_ = get_city(source)
    return tm.predict_table(model, g, hour)


ss = st.session_state

st.title("Ahmedabad route planner: shortest distance vs predicted fastest")
st.caption("Choose a landmark or click the map. The selected pin stays at the chosen location; routing uses a separate nearest-road point.")
with st.sidebar:
    st.header("Route settings")
    source = st.radio("Road network", ["Real Ahmedabad (OpenStreetMap)", "Offline demo network"], key="source")
    hour = st.slider("Time of day (hour)", 0.0, 23.5, 8.5, 0.5)
    algo = st.radio("Algorithm", ["A*", "Dijkstra", "Compare both"], horizontal=True)
    mode = st.radio("Next map click sets", ["Start", "Destination"], horizontal=True)
    a = st.selectbox("Search start landmark", ["-"] + list(PRESETS), key="start_preset")
    b = st.selectbox("Search destination landmark", ["-"] + list(PRESETS), key="end_preset")
g, model, metrics, segment_nodes, segment_xy, note = get_city(source)
if note:
    st.warning(note)
st.caption(f"{len(g.pos):,} intersections, {len(g.edges):,} road segments | travel-time model "
           f"MAE {metrics['MAE_min']:.3f} min/segment, R2 {metrics['R2']:.3f} (traffic data is synthetic)")

for key, preset in (("start", a), ("end", b)):
    if preset != "-" and ss.get(f"preset_{key}") != preset:
        ss[key] = PRESETS[preset]
        ss[f"preset_{key}"] = preset

m = folium.Map(location=list(CENTER), zoom_start=12, tiles="OpenStreetMap")
result = None
comparison = None
route_graph = g
snap_error = False
route_attempted = False
if ss.get("start") and ss.get("end"):
    start_xy = to_km(*ss["start"])
    end_xy = to_km(*ss["end"])
    start_snap = snap_to_road(segment_nodes, segment_xy, *start_xy)
    end_snap = snap_to_road(segment_nodes, segment_xy, *end_xy)
    for location, snapped in ((ss["start"], start_snap), (ss["end"], end_snap)):
        snapped_latlon = to_latlon(snapped["x"], snapped["y"])
        folium.PolyLine([location, snapped_latlon], color="#727272", weight=2, dash_array="3 5",
                        tooltip=f"Road snap: {snapped['distance_km'] * 1000:.0f} m").add_to(m)
        folium.CircleMarker(snapped_latlon, radius=4, color="#555555", fill=True,
                            tooltip="Snapped road position").add_to(m)
    max_snap_distance_km = 0.75
    if max(start_snap["distance_km"], end_snap["distance_km"]) > max_snap_distance_km:
        snap_error = True
        st.error("A selected place is more than 750 m from the loaded road network. Choose a closer location or load the real Ahmedabad network.")
    else:
        route_attempted = True
        route_graph, route_lookup, s_node, e_node = route_graph_for_locations(g, start_snap, end_snap)
        table = get_table(source, hour)
        selected_algo = "A*" if algo == "Compare both" else algo
        result = plan(route_graph, route_lookup, table, s_node, e_node, hour, selected_algo)
        if algo == "Compare both":
            comparison = {
                "A*": result,
                "Dijkstra": plan(route_graph, route_lookup, table, s_node, e_node, hour, "Dijkstra"),
            }

    pts = lambda path: [to_latlon(*route_graph.pos[n]) for n in path]
    if result:
        route_algorithm = "A*" if algo == "Compare both" else algo
        folium.PolyLine(pts(result["dist_path"]), color="#1f5fd1", weight=5, dash_array="8", opacity=0.9,
                        tooltip=f"Shortest distance ({route_algorithm})").add_to(m)
        folium.PolyLine(pts(result["time_path"]), color="#d6322e", weight=5, opacity=0.9,
                        tooltip=f"Predicted fastest ({route_algorithm})").add_to(m)
        if comparison:
            dij = comparison["Dijkstra"]
            if dij:
                if dij["dist_path"] != result["dist_path"]:
                    folium.PolyLine(pts(dij["dist_path"]), color="#368b45", weight=3, dash_array="4 7",
                                    tooltip="Shortest distance (Dijkstra)").add_to(m)
                if dij["time_path"] != result["time_path"]:
                    folium.PolyLine(pts(dij["time_path"]), color="#8b4db5", weight=3, dash_array="4 7",
                                    tooltip="Predicted fastest (Dijkstra)").add_to(m)
if ss.get("start"):
    folium.Marker(ss["start"], tooltip="Start", icon=folium.Icon(color="green")).add_to(m)
if ss.get("end"):
    folium.Marker(ss["end"], tooltip="Destination", icon=folium.Icon(color="red")).add_to(m)

left, right = st.columns([3, 2])
with left:
    out = st_folium(m, height=560, use_container_width=True, returned_objects=["last_clicked"])
    click = out.get("last_clicked") if out else None
    if click and ss.get("last_click") != click:
        ss["last_click"] = click
        ss["start" if mode == "Start" else "end"] = (click["lat"], click["lng"])
        st.rerun()
with right:
    st.subheader(f"Results at {int(hour):02d}:{int(hour % 1 * 60):02d}")
    if not result:
        if snap_error:
            st.info("Move the selected location closer to a road in the loaded network.")
        elif route_attempted:
            st.info("No route could be found between these locations on the loaded road network.")
        else:
            st.info("Choose a start and destination landmark, or click the map to set both endpoints.")
    else:
        d, t = result["dist"], result["time"]
        saved = 100 * (d["min"] - t["min"]) / d["min"] if d["min"] else 0
        st.markdown("Blue dashed = shortest distance. Red = predicted fastest. Grey dotted = location-to-road snap.")
        st.dataframe({"": ["Distance (km)", "Travel time (min)"],
                      "Shortest distance": [f"{d['km']:.2f}", f"{d['min']:.1f}"],
                      "Predicted fastest": [f"{t['km']:.2f}", f"{t['min']:.1f}"]}, hide_index=True)
        if result["same"]:
            st.success("Both cost functions chose the same route at this hour.")
        else:
            st.metric("Time saved by the learned cost", f"{saved:.1f}%", f"{d['min'] - t['min']:.1f} min")
        if comparison:
            rows = []
            for name, outcome in comparison.items():
                rows.append({
                    "Algorithm": name,
                    "Shortest route (km)": f"{outcome['dist']['km']:.2f}",
                    "Fastest route (min)": f"{outcome['time']['min']:.1f}",
                    "Nodes expanded (fastest)": outcome["expanded_selected"],
                    "Distance/time paths match": outcome["same"],
                })
            st.subheader("A* vs Dijkstra")
            st.dataframe(rows, hide_index=True)
            st.caption("Dijkstra and A* return equal-cost routes; their paths can differ when multiple routes tie.")
        else:
            st.caption(f"Nodes expanded ({algo}): {result['expanded_selected']:,}. "
                       "Times use the hidden traffic model.")
        st.caption(
            f"Start snapped {start_snap['distance_km'] * 1000:.0f} m to a road; "
            f"destination snapped {end_snap['distance_km'] * 1000:.0f} m."
        )
