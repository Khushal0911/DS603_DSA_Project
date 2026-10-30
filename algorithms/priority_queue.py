"""Small min-priority queue wrapper used by Dijkstra and A*."""
import heapq


class PriorityQueue:
    def __init__(self):
        self._items = []
        self._counter = 0

    def push(self, priority, item):
        heapq.heappush(self._items, (priority, self._counter, item))
        self._counter += 1

    def pop(self):
        priority, _, item = heapq.heappop(self._items)
        return priority, item

    def is_empty(self):
        return not self._items
