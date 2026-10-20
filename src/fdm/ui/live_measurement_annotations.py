"""Deterministic, viewport-relative label placement for measurement overlays.

All geometry is in Qt logical screen pixels. Labels never get permission to
cover a diameter simply because the image has been zoomed or the view is busy.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from PySide6.QtCore import QPointF, QRectF


@dataclass(frozen=True)
class Annotation:
    key: str
    start: tuple[float, float]
    end: tuple[float, float]
    width: float
    height: float
    priority: int = 0


@dataclass(frozen=True)
class LabelPlacement:
    rect: QRectF
    anchor: QPointF
    leader_end: QPointF


def segment_distance(point, start, end):
    """Euclidean point/segment distance, including zero-length segments."""
    dx, dy = end[0] - start[0], end[1] - start[1]
    denominator = dx * dx + dy * dy
    t = max(0., min(1., ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy)
                   / denominator)) if denominator else 0.
    return math.hypot(point[0] - start[0] - t * dx, point[1] - start[1] - t * dy)


def segment_intersects_rect(start, end, rect):
    """Slab clipping treats the protected border as occupied, too."""
    lower, upper = 0., 1.
    for origin, delta, minimum, maximum in (
        (start[0], end[0] - start[0], rect.left(), rect.right()),
        (start[1], end[1] - start[1], rect.top(), rect.bottom()),
    ):
        if abs(delta) < 1e-12:
            if origin < minimum or origin > maximum:
                return False
            continue
        a, b = (minimum - origin) / delta, (maximum - origin) / delta
        lower, upper = max(lower, min(a, b)), min(upper, max(a, b))
        if lower > upper:
            return False
    return True


class _SpatialIndex:
    """Small screen-space buckets keep dense label placement off the hot path."""
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


def layout_annotations(annotations, viewport, *, protected_segments=(), gap=15.,
                       previous_annotations=(), previous_placements=None,
                       previous_protected_segments=()):
    """Place high-priority labels first and omit labels that cannot fit safely.

    The 15 DIP protection is the 12 DIP clearance plus a 3 DIP endpoint radius.
    Candidate positions and priority ties are deterministic; frame arrival does
    not enter this function's input, so streaming cannot jitter labels.
    """
    viewport = QRectF(viewport).adjusted(4., 4., -4., -4.)
    occupied = []
    results = {}
    current = {item.key: item for item in annotations}
    previous = {item.key: item for item in previous_annotations}
    def geometry(item):
        return item.start, item.end, item.width, item.height
    # Additions can only take space away. Keep old labels stable, but validate
    # them against EVERY current line and each already placed priority label.
    # Removal, changed geometry, or a finished draft frees space: allow all
    # previously omitted labels to be considered again.
    incremental = previous_placements is not None and all(
        key in current and geometry(item) == geometry(current[key])
        for key, item in previous.items()) and not (
            previous_protected_segments and not protected_segments)
    if not incremental:
        previous, previous_placements = {}, {}
    segments = [(item.start, item.end) for item in annotations] + list(protected_segments)
    line_index, label_index = _SpatialIndex(), _SpatialIndex()
    for index, (start, end) in enumerate(segments):
        line_index.add(index, (min(start[0], end[0]), min(start[1], end[1]),
                               max(start[0], end[0]), max(start[1], end[1])))

    def safe(rect):
        protected = rect.adjusted(-gap, -gap, gap, gap)
        if not viewport.contains(rect) or any(segment_intersects_rect(*segments[index], protected)
                                             for index in line_index.near(protected)):
            return False
        expanded = rect.adjusted(-4., -3., 4., 3.)
        return not any(expanded.intersects(occupied[index]) for index in label_index.near(expanded))

    def place(item, rect, mid):
        anchor = QPointF(*mid)
        leader = QPointF(min(max(mid[0], rect.left()), rect.right()),
                         min(max(mid[1], rect.top()), rect.bottom()))
        results[item.key] = LabelPlacement(rect, anchor, leader)
        label_index.add(len(occupied), (rect.left(), rect.top(), rect.right(), rect.bottom()))
        occupied.append(rect)

    for item in sorted(annotations, key=lambda item: -item.priority):
        if item.width > viewport.width() or item.height > viewport.height():
            continue
        mid = ((item.start[0] + item.end[0]) / 2., (item.start[1] + item.end[1]) / 2.)
        dx, dy = item.end[0] - item.start[0], item.end[1] - item.start[1]
        length = math.hypot(dx, dy)
        normal = (-dy / length, dx / length) if length else (0., -1.)
        # A line entirely outside the visible viewport should not gain an
        # orphan label along the edge when its coordinates are clamped.
        if not segment_intersects_rect(item.start, item.end, viewport):
            continue
        if item.key in previous:
            old = previous_placements.get(item.key)
            if old is not None and safe(old.rect):
                place(item, old.rect, mid)
                continue
            if old is None and item.priority <= 0:
                # A secondary label already omitted need not try all 42
                # positions on every accepted measurement. A previously visible
                # label displaced by a new line can seek another safe position;
                # hover also promotes omitted labels and searches again.
                continue
        support = abs(normal[0]) * item.width / 2. + abs(normal[1]) * item.height / 2.
        positions = []
        for extra in (0., 16., 36., 64., 96.):
            distance = support + gap + 3. + extra
            for sign in (-1., 1.):
                positions.append((mid[0] + sign * normal[0] * distance,
                                  mid[1] + sign * normal[1] * distance))
        # Near view edges or other diameters, use all cardinal/diagonal sides
        # before omitting a label. The segment itself is always retained.
        for radius in (32., 56., 88., 128.):
            for angle in range(0, 360, 45):
                radians = math.radians(angle)
                positions.append((mid[0] + math.cos(radians) * (radius + item.width / 2.),
                                  mid[1] + math.sin(radians) * (radius + item.height / 2.)))
        for center_x, center_y in positions:
            rect = QRectF(min(max(center_x - item.width / 2., viewport.left()),
                             viewport.right() - item.width),
                          min(max(center_y - item.height / 2., viewport.top()),
                             viewport.bottom() - item.height), item.width, item.height)
            if not safe(rect):
                continue
            place(item, rect, mid)
            break
    return results
