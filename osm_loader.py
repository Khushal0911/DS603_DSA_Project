"""Load the real Ahmedabad road network from OpenStreetMap and convert it to our own Graph class."""
import ast
import math
import os
from graph import Graph, generate_city

CENTER = (23.0300, 72.5800)   # (lat, lon) - middle of the study area
KM_PER_DEG = 111.32

HIGHWAY_CLASS = {
    "motorway": "highway", "motorway_link": "highway", "trunk": "highway", "trunk_link": "highway",
    "primary": "highway", "primary_link": "highway",
    "secondary": "arterial", "secondary_link": "arterial", "tertiary": "arterial", "tertiary_link": "arterial",
}


def to_km(lat, lon, c=CENTER):
    return (lon - c[1]) * KM_PER_DEG * math.cos(math.radians(c[0])), (lat - c[0]) * KM_PER_DEG


def to_latlon(x, y, c=CENTER):
    return c[0] + y / KM_PER_DEG, c[1] + x / (KM_PER_DEG * math.cos(math.radians(c[0])))


def classify(tag):
    if isinstance(tag, str) and tag.startswith("["):
        try:
            tag = ast.literal_eval(tag)
        except Exception:
            pass
    if isinstance(tag, (list, tuple)):
        tag = tag[0]
    return HIGHWAY_CLASS.get(tag, "local")


def from_networkx(G, center=CENTER):
    """G = osmnx MultiDiGraph: nodes have x=lon, y=lat; edges have length (metres) and a highway tag."""
    g = Graph()
    g.center = (0.0, 0.0)
    ids = {}
    for i, (n, d) in enumerate(G.nodes(data=True)):
        ids[n] = i
        x, y = to_km(float(d["y"]), float(d["x"]), center)
        g.add_node(i, x, y)
    best = {}   # collapse parallel roads: keep the shortest per (u, v) so each pair has exactly one edge
    for u, v, d in G.edges(data=True):
        a, b = ids[u], ids[v]
        if a == b:
            continue
        length = max(float(d.get("length", 0)) / 1000.0, g.euclid(a, b))   # never shorter than straight line
        if (a, b) not in best or length < best[(a, b)][0]:
            best[(a, b)] = (length, classify(d.get("highway", "residential")))
    for (a, b), (length, road) in best.items():
        g.add_directed(a, b, length, road)
    return g


def load_ahmedabad(dist=8000, cache="ahmedabad_drive.graphml"):
    """First run downloads from OpenStreetMap (1-3 min) and caches to disk; later runs are instant."""
    import osmnx as ox
    if os.path.exists(cache):
        G = ox.load_graphml(cache)
    else:
        G = ox.graph_from_point(CENTER, dist=dist, network_type="drive")
        ox.save_graphml(G, cache)
    return from_networkx(G)


def demo_city():
    """Offline fallback: the toy grid city, re-centred on Ahmedabad's coordinates."""
    g = generate_city()
    cx, cy = g.center
    g.pos = {n: (x - cx, y - cy) for n, (x, y) in g.pos.items()}
    g.center = (0.0, 0.0)
    return g
