"""Common geometry for linked labels, independent of canvas or measurement kind."""
from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Sequence

from PySide6.QtCore import QPointF, QRectF

from fdm.geometry import Point


@dataclass(frozen=True, slots=True)
class LinkedLabelLayout:
    anchor: QPointF
    text_rect: QRectF
    leader: tuple[QPointF, ...]
    bounds: QRectF
    witness_lines: tuple[tuple[QPointF, QPointF], ...] = ()
    dimension_lines: tuple[tuple[QPointF, QPointF], ...] = ()
    ticks: tuple[tuple[QPointF, QPointF], ...] = ()
    arrowheads: tuple[tuple[QPointF, ...], ...] = ()
    source_points: tuple[QPointF, ...] = ()


def resolve_linked_label(
    anchor: QPointF, center: QPointF, width: float, height: float,
    *, show_leader: bool = True, landing_length: float = 10.0,
) -> LinkedLabelLayout:
    """Return exactly the geometry painted, hit-tested and exported.

    The horizontal landing joins the near side of the label.  A hidden or
    internally anchored leader does not manufacture an extra hit target.
    """
    rect = QRectF(center.x() - width / 2, center.y() - height / 2, width, height)
    leader: tuple[QPointF, ...] = ()
    bounds = QRectF(rect)
    if show_leader and not rect.contains(anchor):
        left = anchor.x() <= center.x()
        end = QPointF(rect.left() if left else rect.right(), center.y())
        elbow = QPointF(end.x() + (-landing_length if left else landing_length), end.y())
        leader = (QPointF(anchor), elbow, end)
        for point in leader:
            bounds = bounds.united(QRectF(point.x() - 1, point.y() - 1, 2, 2))
    return LinkedLabelLayout(QPointF(anchor), rect, leader, bounds)


def _line_outside_rect(start: QPointF, end: QPointF, rect: QRectF):
    """Keep only the portions of a dimension line outside its text box."""
    delta = end - start
    first, last = 0.0, 1.0
    for origin, direction, low, high in (
        (start.x(), delta.x(), rect.left(), rect.right()),
        (start.y(), delta.y(), rect.top(), rect.bottom()),
    ):
        if abs(direction) < 1e-12:
            if origin < low or origin > high:
                return ((start, end),)
            continue
        entering, leaving = sorted(((low - origin) / direction, (high - origin) / direction))
        first, last = max(first, entering), min(last, leaving)
        if first > last:
            return ((start, end),)
    lines = []
    if first > 1e-10:
        lines.append((start, start + delta * first))
    if last < 1.0 - 1e-10:
        lines.append((start + delta * last, end))
    return tuple(lines)


def resolve_dual_dimension(
    start: QPointF, end: QPointF, center: QPointF, width: float, height: float,
    *, show_leader: bool = True, tick_length: float = 8.0,
    landing_length: float = 10.0, arrow_length: float = 7.0,
    endpoint_gap: float = 2.0, witness_overrun: float = 3.0,
) -> LinkedLabelLayout:
    """Resolve two endpoint witnesses and one parallel offset dimension.

    Geometry uses the output plane, never calibrated lengths. Each witness
    remains attached to its own endpoint; moving the text cannot merge them
    into a common leader target or alter the measured interval.
    """
    anchor = (start + end) * .5
    rect = QRectF(center.x() - width / 2, center.y() - height / 2, width, height)
    if not show_leader:
        return LinkedLabelLayout(anchor, rect, (), QRectF(rect))
    vector = end - start
    length = math.hypot(vector.x(), vector.y())
    if length < 1e-9:
        return resolve_linked_label(anchor, center, width, height, landing_length=landing_length)
    tangent = vector / length
    normal = QPointF(-tangent.y(), tangent.x())
    delta = center - anchor
    offset = delta.x() * normal.x() + delta.y() * normal.y()
    first, last = start + normal * offset, end + normal * offset
    text_box = rect.adjusted(-3, -3, 3, 3)
    # A short dimension must never vanish behind its text. Preserve the
    # manually placed text and move only its rail beyond the text box.
    shifted = text_box.contains(first) or text_box.contains(last)
    side = 1.0 if offset >= 0 else -1.0
    if shifted:
        normal_extent = abs(normal.x()) * width / 2 + abs(normal.y()) * height / 2
        offset += side * (normal_extent + 8.0)
    minimum_offset = max(0.0, endpoint_gap) + max(0.0, witness_overrun) + 1.0
    if abs(offset) < minimum_offset:
        offset = side * minimum_offset
        shifted = True
    first, last = start + normal * offset, end + normal * offset
    direction = normal * (1.0 if offset >= 0 else -1.0)
    witness = (
        (start + direction * endpoint_gap, first + direction * witness_overrun),
        (end + direction * endpoint_gap, last + direction * witness_overrun),
    )
    arrow_size = max(1.0, float(arrow_length))
    outside = length < 2 * arrow_size + 4
    rail_start = first - tangent * (arrow_size + 4) if outside else first
    rail_end = last + tangent * (arrow_size + 4) if outside else last
    dimensions = _line_outside_rect(rail_start, rail_end, text_box)
    arrow_direction = -1.0 if outside else 1.0
    half_arrow = normal * (arrow_size * .38)
    first_base = first + tangent * (arrow_direction * arrow_size)
    last_base = last - tangent * (arrow_direction * arrow_size)
    arrows = ((first, first_base + half_arrow, first_base - half_arrow),
              (last, last_base + half_arrow, last_base - half_arrow))
    # Arrowheads carry the endpoint information; the retained ticks field is
    # empty for this CAD style, and stays available to shared callers.
    ticks = ()
    projected_center = (center.x() - first.x()) * tangent.x() + (center.y() - first.y()) * tangent.y()
    leader = ()
    if shifted or projected_center < 0 or projected_center > length:
        near = first + tangent * min(length, max(0.0, projected_center))
        if shifted:
            # Attach to the nearest box edge, rather than walking around a
            # wide text box to its left/right midpoint for a short dimension.
            edge = QPointF(min(rect.right(), max(rect.left(), near.x())),
                           min(rect.bottom(), max(rect.top(), near.y())))
            if edge.x() == rect.left():
                elbow = edge - QPointF(landing_length, 0)
            elif edge.x() == rect.right():
                elbow = edge + QPointF(landing_length, 0)
            else:
                along_edge = min(landing_length, max(0.0, rect.right() - edge.x()))
                elbow = edge + QPointF(along_edge, 0)
            leader = (near, elbow, edge)
        else:
            leader = resolve_linked_label(near, center, width, height, landing_length=landing_length).leader
    bounds = QRectF(rect)
    padding = max(2.0, arrow_size * 2.0 / 7.0)
    for segment in (*witness, *dimensions, *ticks):
        for point in segment:
            bounds = bounds.united(QRectF(point.x() - padding, point.y() - padding, padding * 2, padding * 2))
    for point in leader:
        bounds = bounds.united(QRectF(point.x() - padding, point.y() - padding, padding * 2, padding * 2))
    for point in (start, end, *(point for polygon in arrows for point in polygon)):
        bounds = bounds.united(QRectF(point.x() - padding, point.y() - padding, padding * 2, padding * 2))
    return LinkedLabelLayout(anchor, rect, leader, bounds, witness, dimensions, ticks, arrows, (start, end))


def choose_automatic_offset(
    anchor: Point, width: float, height: float,
    *, occupied: Sequence[QRectF] = (), geometry: Sequence[QRectF] = (),
    image_rect: QRectF | None = None, gap: float = 18.0,
) -> Point:
    """Choose a deterministic nearby position with label/geometry avoidance.

    All inputs and the returned offset use the same image-coordinate plane.
    The caller caches the chosen offset so view transforms cannot move a label.
    """
    candidates: list[tuple[float, Point]] = []
    sx, sy = width / 2 + gap, height / 2 + gap
    for ring in range(1, 5):
        for dx, dy in ((1, -1), (-1, -1), (1, 1), (-1, 1), (0, -1), (0, 1), (1, 0), (-1, 0)):
            offset = Point(dx * sx * ring, dy * sy * ring)
            rect = QRectF(anchor.x + offset.x - width / 2, anchor.y + offset.y - height / 2, width, height)
            score = math.hypot(offset.x, offset.y)
            for obstacle in occupied:
                overlap = rect.adjusted(-gap / 3, -gap / 3, gap / 3, gap / 3).intersected(obstacle)
                score += max(0.0, overlap.width()) * max(0.0, overlap.height()) * 20
            for obstacle in geometry:
                overlap = rect.intersected(obstacle)
                score += max(0.0, overlap.width()) * max(0.0, overlap.height()) * 2
            if image_rect is not None and not image_rect.isEmpty():
                inside = rect.intersected(image_rect)
                score += max(0.0, width * height - max(0.0, inside.width()) * max(0.0, inside.height())) * 25
            candidates.append((score, offset))
    return min(candidates, key=lambda item: item[0])[1]
