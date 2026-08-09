from __future__ import annotations

from collections.abc import Callable, Iterable

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPen, QPolygonF

from fdm.construction_document import clip_line_to_rect
from fdm.construction_geometry import (
    CircleCenterDiameterDefinition,
    CircleCenterRadiusDefinition,
    CircleThreePointDefinition,
    CircleTwoPointDefinition,
    ConstructionEntity,
    FreePointDefinition,
    FrozenFeatureSnapshot,
    LineDefinition,
    LineExtent,
    LiveFeatureRef,
    OffsetParallelDefinition,
    ParallelArrayDefinition,
    ParallelLineSequence,
    ParallelThroughPointDefinition,
    PerpendicularDefinition,
    ResolvedCircle,
    ResolvedConstruction,
    ResolvedGeometry,
    ResolvedLine,
    ResolvedLineArray,
    ResolvedPoint,
)
from fdm.geometry import Point
from fdm.services.object_snap_service import SnapCandidate, SnapKind


PointMapper = Callable[[Point], QPointF]


def draw_construction_entities(
    painter: QPainter,
    entries: Iterable[tuple[ConstructionEntity, ResolvedConstruction]],
    image_to_widget: PointMapper,
    *,
    visible_image_rect: QRectF,
    selected_id: str | None = None,
    hovered_id: str | None = None,
    show_handles: bool = False,
) -> None:
    """Draw the persistent construction layer without interaction artifacts."""

    clip = (
        float(visible_image_rect.left()),
        float(visible_image_rect.top()),
        float(visible_image_rect.right()),
        float(visible_image_rect.bottom()),
    )
    painter.save()
    try:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        for entity, resolved in entries:
            if not entity.visible:
                continue
            selected = entity.id == selected_id
            hovered = entity.id == hovered_id
            if not resolved.valid or resolved.geometry is None:
                continue
            color = QColor(entity.style.stroke_color)
            color.setAlphaF(max(0.0, min(1.0, entity.style.opacity)))
            pen = QPen(
                color,
                max(0.5, float(entity.style.stroke_width) + (1.0 if selected else 0.0)),
                Qt.PenStyle.DashLine if entity.style.dashed else Qt.PenStyle.SolidLine,
            )
            pen.setCosmetic(True)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            _draw_geometry(
                painter,
                resolved.geometry,
                image_to_widget,
                clip,
            )
            if selected or hovered:
                _draw_emphasis(
                    painter,
                    entity,
                    resolved.geometry,
                    image_to_widget,
                    selected=selected,
                    show_handles=show_handles,
                    locked=entity.locked,
                )
    finally:
        painter.restore()


def _draw_geometry(
    painter: QPainter,
    geometry: ResolvedGeometry,
    image_to_widget: PointMapper,
    clip: tuple[float, float, float, float],
) -> None:
    if isinstance(geometry, ResolvedPoint):
        center = image_to_widget(geometry.point)
        painter.drawLine(center + QPointF(-6.0, 0.0), center + QPointF(6.0, 0.0))
        painter.drawLine(center + QPointF(0.0, -6.0), center + QPointF(0.0, 6.0))
        painter.drawEllipse(center, 2.25, 2.25)
        return
    if isinstance(geometry, ResolvedCircle):
        center = image_to_widget(geometry.center)
        radius_point = image_to_widget(
            Point(geometry.center.x + geometry.radius, geometry.center.y)
        )
        radius = abs(radius_point.x() - center.x())
        if radius > 0.0:
            painter.drawEllipse(center, radius, radius)
        return
    if isinstance(geometry, ResolvedLineArray):
        lines = geometry.lines
        visible_lines = (
            (
                line
                for _index, line in lines.indexed_intersecting_rect(
                    clip,
                    padding=2.0,
                )
            )
            if isinstance(lines, ParallelLineSequence)
            else lines
        )
        for line in visible_lines:
            _draw_geometry(painter, line, image_to_widget, clip)
        return
    segment = (
        (geometry.start, geometry.end)
        if geometry.extent is LineExtent.SEGMENT
        else clip_line_to_rect(geometry, clip)
    )
    if segment is not None:
        painter.drawLine(image_to_widget(segment[0]), image_to_widget(segment[1]))


def _draw_emphasis(
    painter: QPainter,
    entity: ConstructionEntity,
    geometry: ResolvedGeometry,
    image_to_widget: PointMapper,
    *,
    selected: bool,
    show_handles: bool,
    locked: bool,
) -> None:
    points = (
        _control_points(entity, geometry)
        if show_handles and not locked
        else ()
    )
    if not points:
        return
    painter.save()
    try:
        fill = QColor("#58C4C7") if selected else QColor("#F4D35E")
        painter.setPen(QPen(QColor("#0B0B0B"), 1.2))
        painter.setBrush(fill)
        for point in points:
            painter.drawRect(QRectF(image_to_widget(point) - QPointF(4.5, 4.5), QPointF(9.0, 9.0)))
    finally:
        painter.restore()


def _control_points(
    entity: ConstructionEntity,
    geometry: ResolvedGeometry,
) -> tuple[Point, ...]:
    definition = entity.definition
    if isinstance(definition, FreePointDefinition):
        return (definition.point,)
    if isinstance(definition, LineDefinition):
        if definition.axis_constraint is not None:
            return (definition.start,)
        return (definition.start, definition.end)
    if isinstance(definition, CircleCenterRadiusDefinition):
        return (
            definition.center,
            Point(
                definition.center.x + definition.radius,
                definition.center.y,
            ),
        )
    if isinstance(definition, CircleCenterDiameterDefinition):
        return (
            definition.center,
            Point(
                definition.center.x + definition.diameter / 2.0,
                definition.center.y,
            ),
        )
    if isinstance(definition, CircleTwoPointDefinition):
        return (definition.first, definition.second)
    if isinstance(definition, CircleThreePointDefinition):
        return (definition.first, definition.second, definition.third)
    if isinstance(
        definition,
        (ParallelThroughPointDefinition, PerpendicularDefinition),
    ):
        if isinstance(definition.point_source, LiveFeatureRef):
            return ()
        if (
            isinstance(definition.point_source, FrozenFeatureSnapshot)
            and isinstance(definition.point_source.geometry, ResolvedPoint)
        ):
            return (definition.point_source.geometry.point,)
        return (definition.point,)
    if isinstance(definition, (OffsetParallelDefinition, ParallelArrayDefinition)):
        line = (
            geometry.lines[0]
            if isinstance(geometry, ResolvedLineArray) and geometry.lines
            else geometry
        )
        if isinstance(line, ResolvedLine):
            return (
                Point(
                    (line.start.x + line.end.x) / 2.0,
                    (line.start.y + line.end.y) / 2.0,
                ),
            )
    return ()


def draw_snap_candidate(
    painter: QPainter,
    candidate: SnapCandidate | None,
    image_to_widget: PointMapper,
) -> None:
    if candidate is None:
        return
    center = image_to_widget(candidate.point_px)
    color = QColor("#F4D35E")
    painter.save()
    try:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(QPen(QColor("#111111"), 3.8))
        _draw_snap_symbol(painter, center, candidate.kind)
        painter.setPen(QPen(color, 2.0))
        _draw_snap_symbol(painter, center, candidate.kind)
        label_rect = QRectF(center.x() + 10.0, center.y() - 24.0, 92.0, 22.0)
        painter.setPen(Qt.PenStyle.NoPen)
        background = QColor("#15191F")
        background.setAlpha(220)
        painter.setBrush(background)
        painter.drawRoundedRect(label_rect, 4.0, 4.0)
        painter.setPen(QColor("#FFFFFF"))
        painter.drawText(label_rect.adjusted(6.0, 0.0, -4.0, 0.0), Qt.AlignmentFlag.AlignVCenter, candidate.label)
    finally:
        painter.restore()


def _draw_snap_symbol(painter: QPainter, center: QPointF, kind: SnapKind) -> None:
    radius = 5.0
    if kind in {SnapKind.POINT, SnapKind.ENDPOINT}:
        painter.drawRect(QRectF(center - QPointF(radius, radius), QPointF(radius * 2.0, radius * 2.0)))
    elif kind is SnapKind.MIDPOINT:
        painter.drawPolygon(
            QPolygonF(
                [
                    center + QPointF(0.0, -radius - 1.0),
                    center + QPointF(radius + 1.0, radius),
                    center + QPointF(-radius - 1.0, radius),
                ]
            )
        )
    elif kind is SnapKind.CENTER:
        painter.drawEllipse(center, radius, radius)
        painter.drawLine(center + QPointF(-radius - 3.0, 0.0), center + QPointF(radius + 3.0, 0.0))
        painter.drawLine(center + QPointF(0.0, -radius - 3.0), center + QPointF(0.0, radius + 3.0))
    elif kind is SnapKind.INTERSECTION:
        painter.drawLine(center + QPointF(-radius, -radius), center + QPointF(radius, radius))
        painter.drawLine(center + QPointF(-radius, radius), center + QPointF(radius, -radius))
    elif kind is SnapKind.QUADRANT:
        painter.drawPolygon(
            QPolygonF(
                [
                    center + QPointF(0.0, -radius - 1.0),
                    center + QPointF(radius + 1.0, 0.0),
                    center + QPointF(0.0, radius + 1.0),
                    center + QPointF(-radius - 1.0, 0.0),
                ]
            )
        )
    elif kind is SnapKind.PERPENDICULAR:
        # CAD-style right-angle glyph: the corner itself is the acquired foot.
        painter.drawPolyline(
            QPolygonF(
                [
                    center + QPointF(-radius - 1.0, radius + 1.0),
                    center + QPointF(-radius - 1.0, -radius + 1.0),
                    center + QPointF(radius - 1.0, -radius + 1.0),
                ]
            )
        )
        painter.drawLine(
            center + QPointF(-radius - 1.0, 1.0),
            center + QPointF(-1.0, 1.0),
        )
        painter.drawLine(
            center + QPointF(-1.0, 1.0),
            center + QPointF(-1.0, -radius + 1.0),
        )
    elif kind is SnapKind.TANGENT:
        # A small circle touched by a short line stays distinguishable from
        # CENTER/QUADRANT at the same logical-screen size.
        painter.drawEllipse(center + QPointF(0.0, 1.0), radius - 1.0, radius - 1.0)
        painter.drawLine(
            center + QPointF(-radius - 2.0, -radius + 1.0),
            center + QPointF(radius + 2.0, -radius + 1.0),
        )
    else:
        painter.drawEllipse(center, radius, radius)


__all__ = ["draw_construction_entities", "draw_snap_candidate"]
