"""Synthetic city road network: a jittered grid with highways, arterials and local streets."""
import math
import random
from collections import deque
from dataclasses import dataclass


@dataclass
class Edge:
    idx: int
    u: int
    v: int
    length: float       # km
    road: str           # highway / arterial / local
    center_dist: float  # km from edge midpoint to city centre
    base_idx: int | None = None
    fraction: float = 1.0


class Graph:
    def __init__(self):
        self.pos = {}      # node -> (x, y) in km
        self.adj = {}      # node -> list[Edge]
        self.edges = []    # all directed edges, edge.idx == position in this list
        self.center = (0.0, 0.0)

    def add_node(self, n, x, y):
        self.pos[n] = (x, y)
        self.adj[n] = []

    def add_directed(self, u, v, length, road, base_idx=None, fraction=1.0):
        """One directed edge (used for one-way streets from OpenStreetMap)."""
        mx = (self.pos[u][0] + self.pos[v][0]) / 2
        my = (self.pos[u][1] + self.pos[v][1]) / 2
        cd = math.hypot(mx - self.center[0], my - self.center[1])
        e = Edge(len(self.edges), u, v, length, road, cd, base_idx, fraction)
        self.edges.append(e)
        self.adj[u].append(e)

    def add_road(self, u, v, length, road):
        """Two-way road = two directed edges."""
        self.add_directed(u, v, length, road)
        self.add_directed(v, u, length, road)

    def euclid(self, a, b):
        (x1, y1), (x2, y2) = self.pos[a], self.pos[b]
        return math.hypot(x1 - x2, y1 - y2)

    def is_connected(self, banned=frozenset()):
        start = next(iter(self.pos))
        seen, q = {start}, deque([start])
        while q:
            u = q.popleft()
            for e in self.adj[u]:
                if e.v not in seen and (min(e.u, e.v), max(e.u, e.v)) not in banned:
                    seen.add(e.v)
                    q.append(e.v)
        return len(seen) == len(self.pos)


def generate_city(n=16, spacing=0.6, seed=7, drop_local=0.12):
    rng = random.Random(seed)
    g = Graph()
    size = (n - 1) * spacing
    g.center = (size / 2, size / 2)
    for r in range(n):
        for c in range(n):
            j = spacing * 0.12
            g.add_node(r * n + c, c * spacing + rng.uniform(-j, j), r * spacing + rng.uniform(-j, j))

    def road_type(line):
        return "highway" if line % 5 == 0 else "arterial" if line % 2 == 0 else "local"

    roads = []
    for r in range(n):
        for c in range(n):
            if c + 1 < n:
                roads.append((r * n + c, r * n + c + 1, road_type(r)))   # horizontal, line = row
            if r + 1 < n:
                roads.append((r * n + c, (r + 1) * n + c, road_type(c)))  # vertical, line = column

    # drop some local streets (keeping the network connected) for a more realistic map
    tmp = Graph(); tmp.pos = g.pos; tmp.center = g.center
    tmp.adj = {k: [] for k in g.pos}
    banned = set()
    for u, v, t in roads:
        tmp.add_road(u, v, 1.0, t)
    locals_ = [(u, v) for u, v, t in roads if t == "local"]
    rng.shuffle(locals_)
    for u, v in locals_[: int(len(locals_) * drop_local)]:
        banned.add((min(u, v), max(u, v)))
        if not tmp.is_connected(banned):
            banned.discard((min(u, v), max(u, v)))

    for u, v, t in roads:
        if (min(u, v), max(u, v)) in banned:
            continue
        length = g.euclid(u, v) * rng.uniform(1.0, 1.08)  # road >= straight line -> admissible heuristic
        g.add_road(u, v, length, t)
    return g
