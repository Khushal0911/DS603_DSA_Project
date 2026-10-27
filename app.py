"""Ahmedabad route planner - Streamlit frontend.  Run:  streamlit run app.py"""
import numpy as np
import streamlit as st
import folium
from streamlit_folium import st_folium

import travel_model as tm
from osm_loader import CENTER, demo_city, load_ahmedabad, to_km, to_latlon
from planner import plan

st.set_page_config(page_title="Ahmedabad Route Planner", layout="wide")

PRESETS = {  # approximate coordinates - or just click the map
    "Kalupur Railway Station": (23.0268, 72.6013),
    "Lal Darwaja": (23.0240, 72.5812),
    "Law Garden": (23.0258, 72.5547),
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
    xy = np.array([g.pos[i] for i in range(len(g.pos))])
    lookup = {(e.u, e.v): e for e in g.edges}
    return g, model, metrics, xy, lookup, note


@st.cache_data(show_spinner=False)
def get_table(source, hour):
    g, model, *_ = get_city(source)
    return tm.predict_table(model, g, hour)


def snap(xy, lat, lon):
    x, y = to_km(lat, lon)
    return int(np.argmin((xy[:, 0] - x) ** 2 + (xy[:, 1] - y) ** 2))


st.title("Ahmedabad route planner: shortest distance vs predicted fastest")
with st.sidebar:
    source = st.radio("Road network", ["Real Ahmedabad (OpenStreetMap)", "Offline demo network"], key="source")
    hour = st.slider("Time of day (hour)", 0.0, 23.5, 8.5, 0.5)
    algo = st.radio("Algorithm", ["A*", "Dijkstra"], horizontal=True)
    mode = st.radio("Next map click sets", ["Start", "Destination"], horizontal=True)
    a = st.selectbox("Start (preset)", ["-"] + list(PRESETS))
    b = st.selectbox("Destination (preset)", ["-"] + list(PRESETS))

g, model, metrics, xy, lookup, note = get_city(source)
if note:
    st.warning(note)
st.caption(f"{len(g.pos):,} intersections, {len(g.edges):,} road segments | travel-time model "
           f"MAE {metrics['MAE_min']:.3f} min/segment, R2 {metrics['R2']:.3f} (traffic data is synthetic)")

ss = st.session_state
for key, preset in (("start", a), ("end", b)):
    if preset != "-" and ss.get(f"preset_{key}") != preset:
        ss[key] = PRESETS[preset]
        ss[f"preset_{key}"] = preset

m = folium.Map(location=list(CENTER), zoom_start=12, tiles="OpenStreetMap")
result = None
if ss.get("start") and ss.get("end"):
    s_node, e_node = snap(xy, *ss["start"]), snap(xy, *ss["end"])
    if s_node != e_node:
        result = plan(g, lookup, get_table(source, hour), s_node, e_node, hour, algo)
    pts = lambda path: [to_latlon(*g.pos[n]) for n in path]
    if result:
        folium.PolyLine(pts(result["dist_path"]), color="#1f5fd1", weight=5, dash_array="8", opacity=0.9,
                        tooltip="Shortest distance").add_to(m)
        folium.PolyLine(pts(result["time_path"]), color="#d6322e", weight=5, opacity=0.9,
                        tooltip="Predicted fastest").add_to(m)
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
        st.info("Click the map (or pick presets) to set a start and a destination.")
    else:
        d, t = result["dist"], result["time"]
        saved = 100 * (d["min"] - t["min"]) / d["min"] if d["min"] else 0
        st.markdown("Blue dashed = shortest distance. Red = predicted fastest.")
        st.dataframe({"": ["Distance (km)", "Travel time (min)"],
                      "Shortest distance": [f"{d['km']:.2f}", f"{d['min']:.1f}"],
                      "Predicted fastest": [f"{t['km']:.2f}", f"{t['min']:.1f}"]}, hide_index=True)
        if result["same"]:
            st.success("Both cost functions chose the same route at this hour.")
        else:
            st.metric("Time saved by the learned cost", f"{saved:.1f}%", f"{d['min'] - t['min']:.1f} min")
        st.caption(f"Nodes expanded ({algo}): {result['expanded_selected']:,} | "
                   f"Dijkstra: {result['expanded_dijkstra']:,}. Times are scored on the hidden true traffic model.")
