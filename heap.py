"""Binary min-heap written from scratch (no heapq). Ties broken by insertion order."""


class MinHeap:
    def __init__(self):
        self._a = []
        self._count = 0

    def __len__(self):
        return len(self._a)

    def push(self, priority, item):
        self._count += 1
        self._a.append((priority, self._count, item))
        self._sift_up(len(self._a) - 1)

    def pop(self):
        a = self._a
        top = a[0]
        last = a.pop()
        if a:
            a[0] = last
            self._sift_down(0)
        return top[0], top[2]

    def _sift_up(self, i):
        a = self._a
        while i > 0:
            p = (i - 1) // 2
            if (a[i][0], a[i][1]) < (a[p][0], a[p][1]):
                a[i], a[p] = a[p], a[i]
                i = p
            else:
                break

    def _sift_down(self, i):
        a, n = self._a, len(self._a)
        while True:
            l, r, s = 2 * i + 1, 2 * i + 2, i
            if l < n and (a[l][0], a[l][1]) < (a[s][0], a[s][1]):
                s = l
            if r < n and (a[r][0], a[r][1]) < (a[s][0], a[s][1]):
                s = r
            if s == i:
                break
            a[i], a[s] = a[s], a[i]
            i = s
