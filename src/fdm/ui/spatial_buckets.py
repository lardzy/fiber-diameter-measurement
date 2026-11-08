"""Broad-phase spatial queries in a single coordinate space."""

import math


class SpatialBucketIndex:
    """Bounded rectangular buckets for repeated viewport and overlap queries."""
    def __init__(self, cell_size=64.):
        self.cells = {}
        self.cell_size = cell_size
        self.broad = set()

    def _cells(self, left, top, right, bottom):
        x1, x2 = math.floor(left / self.cell_size), math.floor(right / self.cell_size)
        y1, y2 = math.floor(top / self.cell_size), math.floor(bottom / self.cell_size)
        if (x2 - x1 + 1) * (y2 - y1 + 1) > 4096:
            return None
        return ((x, y) for x in range(x1, x2 + 1) for y in range(y1, y2 + 1))

    def add(self, identity, bounds):
        cells = self._cells(*bounds)
        if cells is None:
            self.broad.add(identity)
        else:
            for cell in cells:
                self.cells.setdefault(cell, []).append(identity)

    def near(self, rect):
        cells = self._cells(rect.left(), rect.top(), rect.right(), rect.bottom())
        if cells is None:
            return set(value for values in self.cells.values() for value in values) | self.broad
        return self.broad | {value for cell in cells for value in self.cells.get(cell, ())}
