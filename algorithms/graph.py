"""Convert an OSMnx road graph into our own adjacency-list graph."""
from dataclasses import dataclass
from math import radians, sin, cos, sqrt, atan2


@dataclass
class RoadEdge:
    target: int
    length_m: float
    road_type: str
    coordinates: list  # Leaflet order: [latitude, longitude]


class RoadGraph:
    def __init__(self):
        self.adjacency = {}
        self.coordinates = {}

    def add_node(self, node_id, lat, lon):
        self.adjacency.setdefault(node_id, [])
        self.coordinates[node_id] = (float(lat), float(lon))

    def add_edge(self, source, edge):
        self.adjacency.setdefault(source, []).append(edge)

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
