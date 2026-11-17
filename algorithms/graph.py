"""Convert an OSMnx road graph into our own adjacency-list graph."""
from dataclasses import dataclass
from math import radians, sin, cos, sqrt, atan2


@dataclass
class RoadEdge:
    target: int
    length_m: float
    road_type: str
    coordinates: list  # Leaflet order: [latitude, longitude]
    edge_id: tuple | None = None  # OSM (source, target, key), including parallel roads


class RoadGraph:
    def __init__(self):
        self.adjacency = {}
        self.coordinates = {}
        self.revision = 0
        self.distance_lower_bound = 1.0
        self.max_step_m = 0.0
        self._nearest_index = None
        self._has_edges = False

    def add_node(self, node_id, lat, lon):
        self.adjacency.setdefault(node_id, [])
        self.coordinates[node_id] = (float(lat), float(lon))
        self.revision += 1
        self._nearest_index = None
        # Coordinate edits can invalidate the previously certified bounds.
        if self._has_edges:
            self.distance_lower_bound = 0.0
            self.max_step_m = float('inf')

    def add_edge(self, source, edge):
        if edge.edge_id is None:
            edge.edge_id = (source, edge.target, len(self.adjacency.get(source, ())))
        self.adjacency.setdefault(source, []).append(edge)
        self._has_edges = True
        self.revision += 1
        step = haversine_m(self.coordinates[source], self.coordinates[edge.target])
        self.max_step_m = max(self.max_step_m, step)
        if step:
            self.distance_lower_bound = min(self.distance_lower_bound, edge.length_m / step)

    def neighbors(self, node_id):
        return self.adjacency.get(node_id, [])


def haversine_m(point_a, point_b):
    lat1, lon1 = point_a
    lat2, lon2 = point_b
    radius = 6371000
    p1, p2 = radians(lat1), radians(lat2)
    dlat, dlon = radians(lat2 - lat1), radians(lon2 - lon1)
    value = sin(dlat / 2) ** 2 + cos(p1) * cos(p2) * sin(dlon / 2) ** 2
    return 2 * radius * atan2(sqrt(value), sqrt(1 - value))
