from algorithms.priority_queue import MinPriorityQueue

def dijkstra(graph, source, destination, cost_function):
    """Return (path, total_cost, stats), or None if unreachable."""
    distances = {node: float("inf") for node in graph}
    previous = {}
    distances[source] = 0.0
    queue = MinPriorityQueue()
    queue.push(source, 0.0)
    visited = set()
    expanded = 0
    pushes = 1

    while not queue.is_empty():
        current, current_cost = queue.pop()
        if current in visited:
            continue
        visited.add(current)
        expanded += 1
        if current == destination:
            break
        for neighbor, edge in graph[current].items():
            new_cost = current_cost + cost_function(edge)
            if new_cost < distances[neighbor]:
                distances[neighbor] = new_cost
                previous[neighbor] = current
                queue.push(neighbor, new_cost)
                pushes += 1

    if distances[destination] == float("inf"):
        return None
    path = []
    node = destination
    while node != source:
        path.append(node)
        node = previous[node]
    path.append(source)
    path.reverse()
    return path, distances[destination], {
        "nodes_explored": expanded,
        "queue_pushes": pushes,
        "visited_nodes": len(visited)
    }
