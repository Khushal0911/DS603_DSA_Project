"""Dijkstra and A* implemented from scratch over an adjacency list."""
from algorithms.priority_queue import PriorityQueue
from algorithms.graph import haversine_m


def _edge_cost(edge, mode, predicted_time):
    if mode == 'time':
        return predicted_time(edge)
    return edge.length_m


def _reconstruct_path(previous, start, goal, previous_edges=None, path_edges=None):
    if path_edges is not None:
        path_edges.clear()
    if goal != start and goal not in previous:
        return []
    path = [goal]
    while path[-1] != start:
        if path_edges is not None:
            path_edges.append(previous_edges[path[-1]])
        path.append(previous[path[-1]])
    path.reverse()
    if path_edges is not None:
        path_edges.reverse()
    return path


def dijkstra(graph, start, goal, mode='distance', predicted_time=None, *, path_edges=None):
    queue = PriorityQueue()
    queue.push(0, start)
    best = {start: 0.0}
    previous = {}
    previous_edges = {}
    visited = set()
    adjacency = graph.adjacency
    infinity = float('inf')

    while not queue.is_empty():
        current_cost, current = queue.pop()
        if current_cost != best.get(current) or current in visited:
            continue
        visited.add(current)
        if current == goal:
            break

        for edge in adjacency.get(current, ()):
            target = edge.target
            if target in visited:
                continue
            # Nonnegative weights cannot improve an already cheaper path.
            if current_cost >= best.get(target, infinity):
                continue
            candidate = current_cost + _edge_cost(edge, mode, predicted_time)
            if candidate < best.get(target, infinity):
                best[target] = candidate
                previous[target] = current
                previous_edges[target] = edge
                queue.push(candidate, target)

    return _reconstruct_path(previous, start, goal, previous_edges, path_edges), best.get(goal)


def astar(graph, start, goal, mode='distance', predicted_time=None, *, path_edges=None):
    goal_point = graph.coordinates[goal]
    if mode == 'time':
        # Every prediction >= minimum_seconds, every geographic step <= max_step_m.
        # Unlike an assumed maximum speed, this remains a lower bound for RF outputs.
        scale = (getattr(predicted_time, 'minimum_seconds', 0.0) / graph.max_step_m
                 if graph.max_step_m else 0.0)
    else:
        scale = graph.distance_lower_bound
    scale *= 1 - 1e-12  # Keep the certified lower bound conservative under rounding.
    heuristics = {}

    def heuristic(node):
        if node not in heuristics:
            heuristics[node] = haversine_m(graph.coordinates[node], goal_point) * scale
        return heuristics[node]

    queue = PriorityQueue()
    queue.push(heuristic(start), (start, 0.0))
    best = {start: 0.0}
    previous = {}
    previous_edges = {}
    visited = set()
    adjacency = graph.adjacency
    infinity = float('inf')

    while not queue.is_empty():
        _, (current, current_cost) = queue.pop()
        if current_cost != best.get(current) or current in visited:
            continue
        visited.add(current)
        if current == goal:
            break

        for edge in adjacency.get(current, ()):
            target = edge.target
            if current_cost >= best.get(target, infinity):
                continue
            candidate = current_cost + _edge_cost(edge, mode, predicted_time)
            if candidate < best.get(target, infinity):
                best[target] = candidate
                previous[target] = current
                previous_edges[target] = edge
                # Reopen improvements, including with inconsistent heuristics.
                visited.discard(target)
                queue.push(candidate + heuristic(target), (target, candidate))

    return _reconstruct_path(previous, start, goal, previous_edges, path_edges), best.get(goal)
