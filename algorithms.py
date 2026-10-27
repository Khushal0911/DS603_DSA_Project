"""Dijkstra and A* from scratch on top of our own MinHeap. Cost is a pluggable function."""
from dataclasses import dataclass
from heap import MinHeap


@dataclass
class Result:
    path: list
    cost: float
    expanded: int   # nodes popped & settled -> measure of search effort


def _rebuild(prev, src, dst):
    path, n = [dst], dst
    while n != src:
        n = prev[n]
        path.append(n)
    return path[::-1]


def astar(graph, src, dst, cost, h=None):
    """cost(edge) -> float. h(node, dst) -> float (must never overestimate). h=None => Dijkstra."""
    h = h or (lambda n, goal: 0.0)
    dist = {src: 0.0}
    prev = {}
    closed = set()
    pq = MinHeap()
    pq.push(h(src, dst), src)
    expanded = 0
    while len(pq):
        _, u = pq.pop()
        if u in closed:            # stale entry (lazy deletion)
            continue
        closed.add(u)
        expanded += 1
        if u == dst:
            return Result(_rebuild(prev, src, dst), dist[u], expanded)
        for e in graph.adj[u]:
            nd = dist[u] + cost(e)
            if nd < dist.get(e.v, float("inf")):
                dist[e.v] = nd
                prev[e.v] = u
                pq.push(nd + h(e.v, dst), e.v)
    return Result([], float("inf"), expanded)


def dijkstra(graph, src, dst, cost):
    return astar(graph, src, dst, cost, h=None)
