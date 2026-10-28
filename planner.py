"""Glue between the UI and the algorithms: compute both routes and their stats."""
import numpy as np

from algorithms import astar, dijkstra
from graph import Graph
from travel_model import true_time, V_MAX


def snap_to_road(segment_nodes, segment_xy, x, y):
    point = np.array([x, y])
    starts = segment_xy[:, 0, :]
    vectors = segment_xy[:, 1, :] - starts
    fractions = np.clip(
        np.sum((point - starts) * vectors, axis=1)
        / np.maximum(np.sum(vectors * vectors, axis=1), 1e-12),
        0.0,
        1.0,
    )
    projections = starts + fractions[:, None] * vectors
    distances = np.linalg.norm(projections - point, axis=1)
    closest = int(np.argmin(distances))
    return {
        "segment": tuple(int(node) for node in segment_nodes[closest]),
        "fraction": float(fractions[closest]),
        "x": float(projections[closest, 0]),
        "y": float(projections[closest, 1]),
        "distance_km": float(distances[closest]),
    }


def path_stats(path, tab, hour, lookup, edges=None):
    km = truth = 0.0
    for u, v in zip(path, path[1:]):
        e = lookup[(u, v)]
        km += e.length
        base = edges[e.base_idx] if edges is not None and e.base_idx is not None else e
        truth += true_time(base, hour) * e.fraction
    return {"km": km, "min": truth}


def route_graph_for_locations(g, start, end):
    """Split the selected directed road segments so routes begin/end at snapped points."""
    route_graph = Graph()
    route_graph.center = g.center
    for node, (x, y) in g.pos.items():
        route_graph.add_node(node, x, y)
    for edge in g.edges:
        route_graph.add_directed(
            edge.u, edge.v, edge.length, edge.road,
            base_idx=edge.idx,
        )

    start_node = max(g.pos) + 1
    end_node = start_node + 1
    route_graph.add_node(start_node, start["x"], start["y"])
    route_graph.add_node(end_node, end["x"], end["y"])

    for edge in g.edges:
        key = (min(edge.u, edge.v), max(edge.u, edge.v))
        along = start["fraction"] if edge.u == key[0] else 1.0 - start["fraction"]
        if key == start["segment"]:
            fraction = 1.0 - along
            route_graph.add_directed(
                start_node, edge.v, edge.length * fraction, edge.road,
                base_idx=edge.idx, fraction=fraction,
            )

        along_end = end["fraction"] if edge.u == key[0] else 1.0 - end["fraction"]
        if key == end["segment"]:
            route_graph.add_directed(
                edge.u, end_node, edge.length * along_end, edge.road,
                base_idx=edge.idx, fraction=along_end,
            )

        if key == start["segment"] == end["segment"] and along <= along_end:
            fraction = along_end - along
            route_graph.add_directed(
                start_node, end_node, edge.length * fraction, edge.road,
                base_idx=edge.idx, fraction=fraction,
            )

    lookup = {(edge.u, edge.v): edge for edge in route_graph.edges}
    return route_graph, lookup, start_node, end_node


def plan(g, lookup, table, src, dst, hour, algo="A*"):
    tab = table.tolist()
    dist_cost = lambda e: e.length
    time_cost = lambda e: tab[e.idx]
    if any(edge.base_idx is not None for edge in g.edges):
        time_cost = lambda e: tab[e.base_idx if e.base_idx is not None else e.idx] * e.fraction
    h_dist = lambda n, goal: g.euclid(n, goal)
    h_time = lambda n, goal: g.euclid(n, goal) / V_MAX * 60 * 0.85   # optimistic -> admissible
    if algo == "A*":
        d = astar(g, src, dst, dist_cost, h_dist)
        t = astar(g, src, dst, time_cost, h_time)
    else:
        d = dijkstra(g, src, dst, dist_cost)
        t = dijkstra(g, src, dst, time_cost)
    if not d.path or not t.path:
        return None
    t_dij = t if algo != "A*" else dijkstra(g, src, dst, time_cost)
    return {
        "dist_path": d.path, "time_path": t.path,
        "dist": path_stats(d.path, tab, hour, lookup, g.edges),
        "time": path_stats(t.path, tab, hour, lookup, g.edges),
        "expanded_selected": t.expanded, "expanded_dijkstra": t_dij.expanded,
        "same": d.path == t.path,
    }
