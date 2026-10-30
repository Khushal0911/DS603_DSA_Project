from algorithms.priority_queue import MinPriorityQueue

def astar(graph, source, destination, cost_function, heuristic):
    """A* search. Optimality requires an admissible heuristic."""
    frontier = MinPriorityQueue()
    frontier.push(source, heuristic(source, destination))
    g_score = {node: float("inf") for node in graph}
    g_score[source] = 0.0
    previous = {}
    closed = set()
    expanded = 0
    pushes = 1

    while not frontier.is_empty():
        current, _ = frontier.pop()
        if current in closed:
            continue
        if current == destination:
            break
        closed.add(current)
        expanded += 1
        for neighbor, edge in graph[current].items():
            tentative = g_score[current] + cost_function(edge)
            if tentative < g_score[neighbor]:
                g_score[neighbor] = tentative
                previous[neighbor] = current
                frontier.push(neighbor, tentative + heuristic(neighbor, destination))
                pushes += 1

    if g_score[destination] == float("inf"):
        return None
    path = []
    node = destination
    while node != source:
        path.append(node)
        node = previous[node]
    path.append(source)
    path.reverse()
    return path, g_score[destination], {
        "nodes_explored": expanded,
        "queue_pushes": pushes,
        "visited_nodes": len(closed) + 1
    }
