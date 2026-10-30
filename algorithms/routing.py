"""Dijkstra and A* implemented from scratch over an adjacency list."""
from algorithms.priority_queue import PriorityQueue
from algorithms.graph import haversine_m


def _edge_cost(edge, mode, predicted_time):
    if mode == "time":
        return predicted_time(edge)
    return edge.length_m


def _reconstruct_path(previous, start, goal):
    if goal != start and goal not in previous:
        return []
    path = [goal]
    while path[-1] != start:
        path.append(previous[path[-1]])
    path.reverse()
    return path


def dijkstra(graph, start, goal, mode="distance", predicted_time=None):
    queue = PriorityQueue()
    queue.push(0, start)
    best = {start: 0.0}
    previous = {}
    visited = set()

    while not queue.is_empty():
        current_cost, current = queue.pop()
        if current in visited:
            continue
        visited.add(current)
        if current == goal:
            break

        for edge in graph.neighbors(current):
            if edge.target in visited:
                continue
            weight = _edge_cost(edge, mode, predicted_time)
            candidate = current_cost + weight
            if candidate < best.get(edge.target, float("inf")):
                best[edge.target] = candidate
                previous[edge.target] = current
                queue.push(candidate, edge.target)

    return _reconstruct_path(previous, start, goal), best.get(goal)


def astar(graph, start, goal, mode="distance", predicted_time=None):
    goal_point = graph.coordinates[goal]

    def heuristic(node):
        straight_distance = haversine_m(graph.coordinates[node], goal_point)
        if mode == "time":
            # Optimistic upper speed keeps this a lower-bound time estimate.
            return straight_distance / (130000 / 3600)
        return straight_distance

    queue = PriorityQueue()
    queue.push(heuristic(start), start)
    best = {start: 0.0}
    previous = {}
    visited = set()

    while not queue.is_empty():
        _, current = queue.pop()
        if current in visited:
            continue
        visited.add(current)
        if current == goal:
            break

        for edge in graph.neighbors(current):
            if edge.target in visited:
                continue
            candidate = best[current] + _edge_cost(edge, mode, predicted_time)
            if candidate < best.get(edge.target, float("inf")):
                best[edge.target] = candidate
                previous[edge.target] = current
                queue.push(candidate + heuristic(edge.target), edge.target)

    return _reconstruct_path(previous, start, goal), best.get(goal)
