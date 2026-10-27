"""Shared canvas/export presentation for professional metrology.

Only derived overlays are painted here. Neither labels nor viewing transforms
are ever inputs to geometric evaluation or edge extraction.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QFontMetricsF, QPainter, QPen, QPolygonF

from fdm.geometry import Point
from fdm.measurement_annotation import MeasurementAnnotationLayout
from fdm.metrology import MeasurementPrimitive
from fdm.ui.measurement_annotation_layout import resolve_dual_dimension, resolve_linked_label
from fdm.ui.rendering import draw_measurement_annotation_lines


def result_caption(definition, result, index: int) -> str:
    prefix = f"P{index} · " if definition.annotation_layout is None or definition.annotation_layout.show_identifier else ""
    if not result.valid:
        return prefix + "无效：" + result.reason
    key = definition.annotation_layout.metric_key if definition.annotation_layout else None
    metric = next((m for m in result.metrics if m.key == key), result.primary)
    if metric is None:
        return prefix + "无有效指标"
    symbol = "⌀ " if metric.key == "diameter" else "R " if metric.key == "radius" else ""
    return f"{prefix}{metric.label} {symbol}{metric.value:.4g} {metric.unit}"


def presentation_result(document, definition, result):
    """Select a dimension's visual witness without recomputing any metrics."""
    key = definition.annotation_layout.metric_key if definition.annotation_layout else None
    if not result.valid or key is None:
        return result
    if key in {"dx", "dy"} and definition.kind == "distance":
        line = next((p for p in result.geometry if p.role == "measurement" and len(p.points_px) == 2), None)
        if line is not None:
            a, b = line.points_px
            end = Point(b.x, a.y) if key == "dx" else Point(a.x, b.y)
            return replace(result, geometry=(MeasurementPrimitive("line", (a, end)), MeasurementPrimitive("line", (end, b), "witness")),
                           anchor_px=Point((a.x+end.x)/2, (a.y+end.y)/2))
    if key in {"radius", "diameter", "center_x", "center_y"} and definition.kind in {"circle", "radius", "diameter", "fit_circle", "arc"}:
        metrics = {m.key: m.value for m in result.metrics}
        scale = (document.calibration.pixel_size_x, document.calibration.pixel_size_y) if document.calibration else (1, 1)
        if "center_x" in metrics:
            center = Point(metrics["center_x"] / scale[0], metrics["center_y"] / scale[1])
            right = Point(center.x + metrics["radius"] / scale[0], center.y)
        else:
            witness = next((p for p in result.geometry if p.role == "witness" and len(p.points_px) == 3), None)
            if witness is None:
                return result
            right, center, _ = witness.points_px
        source = tuple(p for p in result.geometry if p.kind == "polyline")
        if key in {"center_x", "center_y"}:
            return replace(result, geometry=source + (MeasurementPrimitive("point", (center,)),), anchor_px=center)
        start = center if key == "radius" else Point(2*center.x-right.x, 2*center.y-right.y)
        return replace(result, geometry=source + (MeasurementPrimitive("line", (start, right)),), anchor_px=right)
    return result


def dimension_endpoints(result):
    lines = [p.points_px for p in result.geometry if p.role == "measurement" and p.kind == "line" and len(p.points_px) == 2]
    return lines[0] if len(lines) == 1 else None


def draw_professional_measurements(painter: QPainter, document, transform,
                                  *, results: Mapping | None = None,
                                  focus_index=None, layouts=None, selected_id=None,
                                  hovered_id=None, show_labels=True, selected_labels_only=False,
                                  exclude_label_ids=(), only_measurement_id=None, draw_geometry=True):
    """Return painted label bounds for identical hit testing and clipping."""
    if results is None:
        from fdm.services.metrology_service import evaluate_document
        results = evaluate_document(document)
    layouts = layouts or {}
    hits = {}
    painter.save()
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    metrics = QFontMetricsF(painter.font())
    ordered = list(enumerate(document.professional_measurements, 1))
    # Keep an actively edited dimension above the other overlays in both the
    # direct path and the transient drag layer, without changing its identifier.
    if selected_id is not None:
        ordered.sort(key=lambda item: item[1].id == selected_id)
    for index, definition in ordered:
        if only_measurement_id is not None and definition.id != only_measurement_id:
            continue
        if definition.focus_index is not None and definition.focus_index != focus_index:
            continue
        result = results.get(definition.id)
        if result is None:
            continue
        result = presentation_result(document, definition, result)
        active = definition.id in {selected_id, hovered_id}
        color = QColor("#ffbf47" if active else "#00b7c7") if result.valid else QColor("#ef805e")
        for primitive in result.geometry if draw_geometry else ():
            points = [transform(p) for p in primitive.points_px]
            pen = QPen(color, 2.2 if active else 1.4)
            pen.setCosmetic(True)
            if primitive.role == "source":
                pen.setStyle(Qt.PenStyle.DotLine)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            if len(points) >= 2:
                painter.drawPolyline(QPolygonF(points))
                if primitive.role == "measurement" and primitive.kind == "line":
                    # Endpoint ticks identify the actual measured span separately
                    # from the dashed label association.
                    import math
                    dx, dy = points[-1].x() - points[0].x(), points[-1].y() - points[0].y()
                    length = math.hypot(dx, dy)
                    if length > 1e-8:
                        for p in (points[0], points[-1]):
                            n = QPointF(-dy / length * 4, dx / length * 4)
                            painter.drawLine(p - n, p + n)
            elif points:
                painter.drawEllipse(points[0], 3, 3)
        if not show_labels or definition.id in exclude_label_ids or (selected_labels_only and definition.id != selected_id):
            continue
        layout = layouts.get(definition.id) or definition.annotation_layout or MeasurementAnnotationLayout()
        anchor = result.anchor_px
        offset = layout.offset_px or Point(65, -38)
        text = result_caption(definition, result, index)
        center = transform(Point(anchor.x + offset.x, anchor.y + offset.y))
        ends = dimension_endpoints(result)
        if layout.leader_style == "dual" and ends is not None:
            label = resolve_dual_dimension(transform(ends[0]), transform(ends[1]), center,
                metrics.horizontalAdvance(text) + 14, metrics.height() + 8, show_leader=layout.show_leader)
        else:
            label = resolve_linked_label(transform(anchor), center,
                metrics.horizontalAdvance(text) + 14, metrics.height() + 8, show_leader=layout.show_leader)
        draw_measurement_annotation_lines(painter, label, color, width=1.2 if active else 1.0)
        painter.setPen(QPen(color, 1.5 if active else .7))
        painter.setBrush(QColor(25, 31, 39, 225))
        painter.drawRoundedRect(label.text_rect, 3, 3)
        painter.setPen(QColor("#ffffff"))
        painter.drawText(label.text_rect, Qt.AlignmentFlag.AlignCenter, text)
        hits[definition.id] = label
    painter.restore()
    return hits
