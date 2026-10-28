"""Correctness checks: heap order, Dijkstra/A* vs an independent Bellman-Ford reference."""
import random
import numpy as np

from heap import MinHeap
from graph import Graph, generate_city
from algorithms import dijkstra, astar
from planner import plan, route_graph_for_locations, snap_to_road


def bellman_ford(g, src, cost):
    d = {n: float("inf") for n in g.pos}
    d[src] = 0.0
    for _ in range(len(g.pos) - 1):
        changed = False
        for e in g.edges:
            if d[e.u] + cost(e) < d[e.v]:
                d[e.v] = d[e.u] + cost(e)
                changed = True
        if not changed:
            break
    return d


def test_heap():
    rng = random.Random(1)
    vals = [rng.random() for _ in range(2000)]
    h = MinHeap()
    for v in vals:
        h.push(v, v)
    out = [h.pop()[0] for _ in range(len(vals))]
    assert out == sorted(vals)
    print("heap: OK")


def test_paths():
    g = generate_city()
    rng = random.Random(3)
    tables = {"distance": lambda e: e.length,
              "random-cost": None}
    rand_cost = [rng.uniform(0.1, 5.0) for _ in g.edges]
    tables["random-cost"] = lambda e: rand_cost[e.idx]
    nodes = list(g.pos)
    for name, cost in tables.items():
        ref_cache = {}
        for _ in range(25):
            s, t = rng.sample(nodes, 2)
            ref_cache.setdefault(s, bellman_ford(g, s, cost))
            ref = ref_cache[s][t]
            d = dijkstra(g, s, t, cost)
            assert abs(d.cost - ref) < 1e-9, (name, s, t, d.cost, ref)
            if name == "distance":
                a = astar(g, s, t, cost, lambda n, goal: g.euclid(n, goal))
                assert abs(a.cost - ref) < 1e-9
                assert a.expanded <= d.expanded
        print(f"dijkstra{' + astar' if name == 'distance' else ''} [{name}]: OK (25 random pairs)")


def test_precise_location_routes():
    snap = snap_to_road(
        np.array([[4, 9]]),
        np.array([[[0.0, 0.0], [1.0, 0.0]]]),
        0.5,
        0.2,
    )
    assert snap["segment"] == (4, 9)
    assert abs(snap["fraction"] - 0.5) < 1e-9
    assert abs(snap["distance_km"] - 0.2) < 1e-9

    one_way = Graph()
    one_way.add_node(0, 0.0, 0.0)
    one_way.add_node(1, 1.0, 0.0)
    one_way.add_directed(0, 1, 1.0, "local")
    forward_start = {"segment": (0, 1), "fraction": 0.25, "x": 0.25, "y": 0.0}
    forward_end = {"segment": (0, 1), "fraction": 0.75, "x": 0.75, "y": 0.0}
    graph, lookup, src, dst = route_graph_for_locations(one_way, forward_start, forward_end)
    forward = plan(graph, lookup, np.ones(1), src, dst, 12.0, "Dijkstra")
    assert forward is not None and abs(forward["dist"]["km"] - 0.5) < 1e-9
    graph, lookup, src, dst = route_graph_for_locations(one_way, forward_end, forward_start)
    assert plan(graph, lookup, np.ones(1), src, dst, 12.0, "Dijkstra") is None

    g = generate_city()
    edge = g.edges[0]
    key = (min(edge.u, edge.v), max(edge.u, edge.v))
    start_fraction = 0.25 if edge.u == key[0] else 0.75
    end_fraction = 0.75 if edge.u == key[0] else 0.25

    def location(fraction):
        x1, y1 = g.pos[key[0]]
        x2, y2 = g.pos[key[1]]
        return {
            "segment": key,
            "fraction": fraction,
            "x": x1 + (x2 - x1) * fraction,
            "y": y1 + (y2 - y1) * fraction,
            "distance_km": 0.0,
        }

    route_graph, lookup, src, dst = route_graph_for_locations(
        g, location(start_fraction), location(end_fraction)
    )
    result = plan(route_graph, lookup, np.ones(len(g.edges)), src, dst, 8.5, "A*")
    dijkstra_result = plan(route_graph, lookup, np.ones(len(g.edges)), src, dst, 8.5, "Dijkstra")
    expected_km = edge.length * 0.5
    assert result is not None
    assert dijkstra_result is not None
    assert abs(result["dist"]["km"] - expected_km) < 1e-9
    assert abs(dijkstra_result["dist"]["km"] - expected_km) < 1e-9
    assert abs(result["time"]["min"] - dijkstra_result["time"]["min"]) < 1e-9
    assert result["dist_path"][0] == src and result["dist_path"][-1] == dst
    print("precise road-segment endpoints: OK")


if __name__ == "__main__":
    test_heap()
    test_paths()
    test_precise_location_routes()
    print("ALL TESTS PASSED")
