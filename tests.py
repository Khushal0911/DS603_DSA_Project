"""Correctness checks: heap order, Dijkstra/A* vs an independent Bellman-Ford reference."""
import random
from heap import MinHeap
from graph import generate_city
from algorithms import dijkstra, astar


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


if __name__ == "__main__":
    test_heap()
    test_paths()
    print("ALL TESTS PASSED")
