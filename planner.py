"""Glue between the UI and the algorithms: compute both routes and their stats."""
from algorithms import astar, dijkstra
from travel_model import true_time, V_MAX


def path_stats(path, tab, hour, lookup):
    km = truth = 0.0
    for u, v in zip(path, path[1:]):
        e = lookup[(u, v)]
        km += e.length
        truth += true_time(e, hour)
    return {"km": km, "min": truth}


def plan(g, lookup, table, src, dst, hour, algo="A*"):
    tab = table.tolist()
    dist_cost = lambda e: e.length
    time_cost = lambda e: tab[e.idx]
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
        "dist": path_stats(d.path, tab, hour, lookup), "time": path_stats(t.path, tab, hour, lookup),
        "expanded_selected": t.expanded, "expanded_dijkstra": t_dij.expanded,
        "same": d.path == t.path,
    }
