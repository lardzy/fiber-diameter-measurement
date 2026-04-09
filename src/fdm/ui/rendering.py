from __future__ import annotations

from dataclasses import dataclass
import math

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QFontMetricsF, QPainter, QPen

from fdm.geometry import Line, Point, direction, normal, point_to_segment_distance
from fdm.models import ImageDocument, Measurement, OverlayAnnotation, OverlayAnnotationKind, TextAnnotation, format_measurement_label_value
from fdm.settings import AppSettings, MeasurementEndpointStyle, ScaleOverlayPlacementMode, ScaleOverlayStyle


@dataclass(slots=True)
class OverlayMetrics:
    line_width: float
    endpoint_radius: float
    scale_bg_width: float
    scale_fg_width: float
    font_px: float


def overlay_metrics(width: int, height: int, render_mode: str) -> OverlayMetrics:
    long_edge = float(max(width, height))
    if render_mode == "full_resolution":
        return OverlayMetrics(
            line_width=2.0,
            endpoint_radius=3.6,
            scale_bg_width=5.0,
            scale_fg_width=2.5,
            font_px=18.0,
        )
    line_width = max(2.0, min(6.0, long_edge * 0.003))
    endpoint_radius = max(4.0, line_width * 1.6)
    return OverlayMetrics(
        line_width=line_width,
        endpoint_radius=endpoint_radius,
        scale_bg_width=max(6.0, line_width * 2.2),
        scale_fg_width=max(3.0, line_width * 1.1),
        font_px=max(12.0, long_edge * 0.022),
    )


def measurement_color(document: ImageDocument, measurement: Measurement, settings: AppSettings) -> QColor:
    group = document.get_group(measurement.fiber_group_id)
    return QColor(group.color if group else settings.default_measurement_color)


def measurement_display_text(measurement: Measurement, document: ImageDocument) -> str:
    return measurement_display_text_with_settings(measurement, document, None)


def measurement_display_text_with_settings(
    measurement: Measurement,
    document: ImageDocument,
    settings: AppSettings | None,
) -> str:
    value = measurement.display_value()
    unit = measurement.display_unit(document.calibration)
    decimals = settings.measurement_label_decimals if settings is not None else 4
    return format_measurement_label_value(value, unit, decimals)


def measurement_label_font(settings: AppSettings) -> QFont:
    font = QFont()
    font.setFamily(settings.measurement_label_font_family)
    font.setPixelSize(int(max(8, settings.measurement_label_font_size)))
    font.setBold(True)
    return font


def overlay_text_font(settings: AppSettings) -> QFont:
    font = QFont()
    font.setFamily(settings.text_font_family)
    font.setPixelSize(int(max(8, settings.text_font_size)))
    return font


def overlay_annotation_line_width(
    settings: AppSettings,
    *,
    suggested_line_width: float,
    render_mode: str,
) -> float:
    base_width = float(max(0.5, settings.overlay_line_width))
    if render_mode == "full_resolution":
        return min(max(base_width, 1.0), 12.0)
    lower_bound = max(1.2, suggested_line_width * 0.75)
    upper_bound = max(lower_bound, suggested_line_width * 1.8)
    return min(max(base_width, lower_bound), upper_bound)


def scale_overlay_font(settings: AppSettings, *, suggested_font_px: float, render_mode: str) -> tuple[QFont, float]:
    font = QFont()
    font.setFamily(settings.scale_overlay_font_family)
    base_font_px = float(max(8, settings.scale_overlay_font_size))
    if render_mode == "full_resolution":
        resolved_px = min(max(base_font_px, 12.0), 28.0)
    else:
        lower_bound = max(10.0, suggested_font_px * 0.75)
        upper_bound = max(lower_bound, suggested_font_px * 1.6)
        resolved_px = min(max(base_font_px, lower_bound), upper_bound)
    font.setPixelSize(int(round(resolved_px)))
    font.setBold(True)
    return font, resolved_px


def _text_layout(font: QFont, content: str) -> tuple[QFontMetricsF, list[str], float, float]:
    metrics = QFontMetricsF(font)
    lines = content.splitlines() or [""]
    width = max(metrics.horizontalAdvance(line or " ") for line in lines)
    height = max(1.0, metrics.lineSpacing() * len(lines))
    return metrics, lines, width, height


def overlay_annotation_bounds(annotation: OverlayAnnotation) -> tuple[float, float, float, float]:
    if annotation.normalized_kind() == OverlayAnnotationKind.TEXT:
        return (
            annotation.anchor_px.x,
            annotation.anchor_px.y,
            annotation.anchor_px.x,
            annotation.anchor_px.y,
        )
    min_x = min(annotation.start_px.x, annotation.end_px.x)
    min_y = min(annotation.start_px.y, annotation.end_px.y)
    max_x = max(annotation.start_px.x, annotation.end_px.x)
    max_y = max(annotation.start_px.y, annotation.end_px.y)
    return min_x, min_y, max_x, max_y


def overlay_annotation_handle_points(annotation: OverlayAnnotation) -> list[tuple[str, Point]]:
    kind = annotation.normalized_kind()
    if kind in {OverlayAnnotationKind.LINE, OverlayAnnotationKind.ARROW}:
        return [("start", annotation.start_px), ("end", annotation.end_px)]
    if kind in {OverlayAnnotationKind.RECT, OverlayAnnotationKind.CIRCLE}:
        min_x, min_y, max_x, max_y = overlay_annotation_bounds(annotation)
        return [
            ("top_left", Point(min_x, min_y)),
            ("top_right", Point(max_x, min_y)),
            ("bottom_left", Point(min_x, max_y)),
            ("bottom_right", Point(max_x, max_y)),
        ]
    return []


def overlay_annotation_rect(annotation: OverlayAnnotation, settings: AppSettings, image_to_output) -> QRectF:
    if annotation.normalized_kind() != OverlayAnnotationKind.TEXT:
        start = image_to_output(annotation.start_px)
        end = image_to_output(annotation.end_px)
        left = min(start.x(), end.x())
        top = min(start.y(), end.y())
        width = max(1.0, abs(end.x() - start.x()))
        height = max(1.0, abs(end.y() - start.y()))
        return QRectF(left, top, width, height)
    font = overlay_text_font(settings)
    _metrics, _lines, width, height = _text_layout(font, annotation.content)
    top_left = image_to_output(annotation.anchor_px)
    padding_x = 6.0
    padding_y = 4.0
    return QRectF(
        top_left.x() - padding_x,
        top_left.y() - padding_y,
        width + (padding_x * 2.0),
        height + (padding_y * 2.0),
    )


def annotation_rect(annotation: TextAnnotation | OverlayAnnotation, settings: AppSettings, image_to_output) -> QRectF:
    if isinstance(annotation, TextAnnotation):
        return overlay_annotation_rect(annotation.to_overlay(), settings, image_to_output)
    return overlay_annotation_rect(annotation, settings, image_to_output)


def draw_overlay_annotations(
    painter: QPainter,
    document: ImageDocument,
    image_to_output,
    settings: AppSettings,
    *,
    selected_overlay_id: str | None = None,
    show_handles: bool = False,
    render_mode: str = "screen_scale_full_image",
) -> None:
    font = overlay_text_font(settings)
    text_metrics, _lines_template, _width, _height = _text_layout(font, " ")
    line_spacing = text_metrics.lineSpacing()
    line_color = QColor(settings.overlay_line_color)
    outline_color = _overlay_outline_color(line_color)
    text_color = QColor(settings.text_color)
    text_outline = QColor("#101820")
    resolved_line_width = overlay_annotation_line_width(
        settings,
        suggested_line_width=2.2,
        render_mode=render_mode,
    )
    annotations = list(getattr(document, "overlay_annotations", []))
    for annotation in annotations:
        kind = annotation.normalized_kind()
        if kind == OverlayAnnotationKind.TEXT:
            painter.setFont(font)
            rect = overlay_annotation_rect(annotation, settings, image_to_output)
            top_left = rect.topLeft() + QPointF(6.0, 4.0)
            painter.setPen(QPen(text_outline, 3))
            y = top_left.y() + text_metrics.ascent()
            for line in annotation.content.splitlines() or [""]:
                painter.drawText(QPointF(top_left.x(), y), line)
                y += line_spacing
            painter.setPen(QPen(text_color, 1))
            y = top_left.y() + text_metrics.ascent()
            for line in annotation.content.splitlines() or [""]:
                painter.drawText(QPointF(top_left.x(), y), line)
                y += line_spacing
            if annotation.id == selected_overlay_id:
                painter.setBrush(QColor(0, 0, 0, 0))
                painter.setPen(QPen(QColor("#F4D35E"), 1.8, Qt.PenStyle.DashLine))
                painter.drawRoundedRect(rect, 6.0, 6.0)
            continue
        _draw_shape_overlay_annotation(
            painter,
            annotation,
            image_to_output,
            color=line_color,
            outline_color=outline_color,
            line_width=resolved_line_width * (1.12 if annotation.id == selected_overlay_id else 1.0),
            selected=annotation.id == selected_overlay_id,
            show_handles=show_handles and annotation.id == selected_overlay_id,
        )


def draw_text_annotations(
    painter: QPainter,
    document: ImageDocument,
    image_to_output,
    settings: AppSettings,
    *,
    selected_text_id: str | None = None,
) -> None:
    draw_overlay_annotations(
        painter,
        document,
        image_to_output,
        settings,
        selected_overlay_id=selected_text_id,
    )


def _draw_shape_overlay_annotation(
    painter: QPainter,
    annotation: OverlayAnnotation,
    image_to_output,
    *,
    color: QColor,
    outline_color: QColor,
    line_width: float,
    selected: bool,
    show_handles: bool,
) -> None:
    kind = annotation.normalized_kind()
    start_point = image_to_output(annotation.start_px)
    end_point = image_to_output(annotation.end_px)
    rect = QRectF(
        min(start_point.x(), end_point.x()),
        min(start_point.y(), end_point.y()),
        max(1.0, abs(end_point.x() - start_point.x())),
        max(1.0, abs(end_point.y() - start_point.y())),
    )
    outline_width = max(line_width * 1.8, line_width + 1.1)
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.setPen(
        QPen(
            outline_color,
            outline_width,
            Qt.PenStyle.SolidLine,
            Qt.PenCapStyle.RoundCap,
            Qt.PenJoinStyle.RoundJoin,
        )
    )
    _draw_overlay_shape_geometry(painter, kind, start_point, end_point, rect)
    painter.setPen(
        QPen(
            color,
            line_width,
            Qt.PenStyle.SolidLine,
            Qt.PenCapStyle.RoundCap,
            Qt.PenJoinStyle.RoundJoin,
        )
    )
    _draw_overlay_shape_geometry(painter, kind, start_point, end_point, rect)
    if selected and kind in {OverlayAnnotationKind.RECT, OverlayAnnotationKind.CIRCLE}:
        painter.setPen(QPen(QColor("#F4D35E"), 1.5, Qt.PenStyle.DashLine))
        painter.drawRect(rect.adjusted(-3.0, -3.0, 3.0, 3.0))
    if show_handles:
        for _handle_name, handle_point in overlay_annotation_handle_points(annotation):
            _draw_overlay_handle(painter, image_to_output(handle_point))


def _draw_overlay_shape_geometry(
    painter: QPainter,
    kind: str,
    start_point: QPointF,
    end_point: QPointF,
    rect: QRectF,
) -> None:
    if kind == OverlayAnnotationKind.RECT:
        painter.drawRect(rect)
        return
    if kind == OverlayAnnotationKind.CIRCLE:
        painter.drawEllipse(rect)
        return
    painter.drawLine(start_point, end_point)
    if kind == OverlayAnnotationKind.ARROW:
        _draw_overlay_arrow_head(painter, start_point, end_point)


def _draw_overlay_arrow_head(painter: QPainter, start_point: QPointF, end_point: QPointF) -> None:
    dx = end_point.x() - start_point.x()
    dy = end_point.y() - start_point.y()
    axis = _normalize(dx, dy)
    side = _normal(axis)
    pen_width = max(1.0, painter.pen().widthF())
    arrow_length = max(10.0, pen_width * 4.8)
    arrow_half_width = max(5.0, pen_width * 2.8)
    tail = QPointF(end_point.x() - axis[0] * arrow_length, end_point.y() - axis[1] * arrow_length)
    left = QPointF(tail.x() + side[0] * arrow_half_width, tail.y() + side[1] * arrow_half_width)
    right = QPointF(tail.x() - side[0] * arrow_half_width, tail.y() - side[1] * arrow_half_width)
    painter.drawLine(end_point, left)
    painter.drawLine(end_point, right)


def _draw_overlay_handle(painter: QPainter, point: QPointF) -> None:
    painter.setBrush(QColor("#FFFFFF"))
    painter.setPen(QPen(QColor("#0B0B0B"), 1.3))
    painter.drawEllipse(point, 4.2, 4.2)


def draw_measurements(
    painter: QPainter,
    document: ImageDocument,
    image_to_output,
    settings: AppSettings,
    *,
    line_width: float,
    endpoint_radius: float,
    selected_measurement_id: str | None = None,
    show_area_fill: bool = True,
    show_area_handles: bool = False,
) -> None:
    for measurement in document.measurements:
        if measurement.measurement_kind == "area":
            draw_area_measurement(
                painter,
                document,
                measurement,
                image_to_output,
                settings,
                line_width=line_width,
                endpoint_radius=endpoint_radius,
                selected=measurement.id == selected_measurement_id,
                show_fill=show_area_fill,
                show_handles=show_area_handles and measurement.id == selected_measurement_id,
            )
            continue
        line = measurement.effective_line()
        start_point = image_to_output(line.start)
        end_point = image_to_output(line.end)
        color = measurement_color(document, measurement, settings)
        selected = measurement.id == selected_measurement_id
        actual_width = line_width * (1.7 if selected else 1.0)
        outline_width = max(actual_width * 1.7, actual_width + 1.0)
        painter.setPen(
            QPen(
                QColor("#0B0B0B"),
                outline_width,
                Qt.PenStyle.SolidLine,
                Qt.PenCapStyle.RoundCap,
                Qt.PenJoinStyle.RoundJoin,
            )
        )
        painter.drawLine(start_point, end_point)
        painter.setPen(
            QPen(
                color,
                actual_width,
                Qt.PenStyle.SolidLine,
                Qt.PenCapStyle.RoundCap,
                Qt.PenJoinStyle.RoundJoin,
            )
        )
        painter.drawLine(start_point, end_point)
        draw_endpoint_style(
            painter,
            QPointF(start_point),
            QPointF(end_point),
            color,
            settings.measurement_endpoint_style,
            line_width=actual_width,
            endpoint_radius=endpoint_radius * (1.15 if selected else 1.0),
        )
        if settings.show_measurement_labels:
            draw_measurement_label(painter, measurement, document, settings, start_point, end_point)


def draw_area_measurement(
    painter: QPainter,
    document: ImageDocument,
    measurement: Measurement,
    image_to_output,
    settings: AppSettings,
    *,
    line_width: float,
    endpoint_radius: float,
    selected: bool,
    show_fill: bool,
    show_handles: bool,
) -> None:
    if len(measurement.polygon_px) < 3:
        return
    color = measurement_color(document, measurement, settings)
    polygon = [image_to_output(point) for point in measurement.polygon_px]
    outline_width = max(line_width * (1.65 if selected else 1.0), 1.8)
    if show_fill:
        fill = QColor(color)
        fill.setAlpha(80 if not selected else 110)
        painter.setBrush(fill)
    else:
        painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.setPen(QPen(QColor("#0B0B0B"), max(outline_width * 1.9, outline_width + 1.0), Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
    painter.drawPolygon(polygon)
    painter.setPen(QPen(color, outline_width, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
    painter.drawPolygon(polygon)
    if not show_handles:
        return
    for point in polygon:
        _draw_circle_endpoint(painter, point, color, endpoint_radius * 0.95)
    center_point = image_to_output(measurement.polygon_center())
    painter.setBrush(QColor("#FFFFFF"))
    painter.setPen(QPen(QColor("#0B0B0B"), 1.6))
    painter.drawEllipse(center_point, endpoint_radius * 0.9, endpoint_radius * 0.9)
    painter.setPen(QPen(QColor("#0B0B0B"), 1.3))
    painter.drawLine(
        QPointF(center_point.x() - endpoint_radius * 0.45, center_point.y()),
        QPointF(center_point.x() + endpoint_radius * 0.45, center_point.y()),
    )
    painter.drawLine(
        QPointF(center_point.x(), center_point.y() - endpoint_radius * 0.45),
        QPointF(center_point.x(), center_point.y() + endpoint_radius * 0.45),
    )


def draw_measurement_label(
    painter: QPainter,
    measurement: Measurement,
    document: ImageDocument,
    settings: AppSettings,
    start_point: QPointF,
    end_point: QPointF,
) -> None:
    font = measurement_label_font(settings)
    painter.setFont(font)
    metrics = QFontMetricsF(font)
    text = measurement_display_text_with_settings(measurement, document, settings)
    text_width = metrics.horizontalAdvance(text)
    text_height = metrics.height()
    axis = direction(measurement.effective_line())
    normal_axis = normal(axis)
    offset = max(12.0, text_height * 0.75)
    center = QPointF(
        (start_point.x() + end_point.x()) / 2.0 + (normal_axis[0] * offset),
        (start_point.y() + end_point.y()) / 2.0 + (normal_axis[1] * offset),
    )
    rect = QRectF(
        center.x() - (text_width / 2.0) - 6.0,
        center.y() - (text_height / 2.0) - 3.0,
        text_width + 12.0,
        text_height + 6.0,
    )
    if settings.measurement_label_parallel_to_line:
        angle = math.degrees(math.atan2(end_point.y() - start_point.y(), end_point.x() - start_point.x()))
        if angle > 90.0:
            angle -= 180.0
        elif angle < -90.0:
            angle += 180.0
        painter.save()
        painter.translate(center)
        painter.rotate(angle)
        parallel_rect = QRectF(
            -(text_width / 2.0) - 6.0,
            -(text_height / 2.0) - 3.0,
            text_width + 12.0,
            text_height + 6.0,
        )
        if settings.measurement_label_background_enabled:
            painter.fillRect(parallel_rect, QColor(16, 24, 32, 168))
        painter.setPen(QPen(QColor("#101820"), 3))
        painter.drawText(parallel_rect, Qt.AlignmentFlag.AlignCenter, text)
        painter.setPen(QPen(QColor(settings.measurement_label_color), 1))
        painter.drawText(parallel_rect, Qt.AlignmentFlag.AlignCenter, text)
        painter.restore()
        return
    if settings.measurement_label_background_enabled:
        painter.fillRect(rect, QColor(16, 24, 32, 168))
    painter.setPen(QPen(QColor("#101820"), 3))
    painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)
    painter.setPen(QPen(QColor(settings.measurement_label_color), 1))
    painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)


def draw_endpoint_style(
    painter: QPainter,
    start_point: QPointF,
    end_point: QPointF,
    color: QColor,
    endpoint_style: str,
    *,
    line_width: float,
    endpoint_radius: float,
) -> None:
    if endpoint_style == MeasurementEndpointStyle.NONE:
        return
    axis = _normalize(end_point.x() - start_point.x(), end_point.y() - start_point.y())
    if endpoint_style == MeasurementEndpointStyle.CIRCLE:
        _draw_circle_endpoint(painter, start_point, color, endpoint_radius)
        _draw_circle_endpoint(painter, end_point, color, endpoint_radius)
        return
    if endpoint_style == MeasurementEndpointStyle.BAR:
        _draw_bar_endpoint(painter, start_point, axis, color, line_width)
        _draw_bar_endpoint(painter, end_point, axis, color, line_width)
        return
    if endpoint_style == MeasurementEndpointStyle.ARROW_INSIDE:
        _draw_arrow_endpoint(painter, start_point, axis, color, inward=True, is_start=True, line_width=line_width)
        _draw_arrow_endpoint(painter, end_point, axis, color, inward=True, is_start=False, line_width=line_width)
        return
    if endpoint_style == MeasurementEndpointStyle.ARROW_OUTSIDE:
        _draw_arrow_endpoint(painter, start_point, axis, color, inward=False, is_start=True, line_width=line_width)
        _draw_arrow_endpoint(painter, end_point, axis, color, inward=False, is_start=False, line_width=line_width)


def draw_preview_scale_anchor(
    painter: QPainter,
    position: QPointF,
    *,
    bar_px: float = 110.0,
    text: str = "比例尺位置",
) -> None:
    painter.setPen(QPen(QColor("#101820"), 5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
    painter.drawLine(position, QPointF(position.x() + bar_px, position.y()))
    painter.setPen(QPen(QColor("#FFFFFF"), 2.5, Qt.PenStyle.DashLine, Qt.PenCapStyle.RoundCap))
    painter.drawLine(position, QPointF(position.x() + bar_px, position.y()))
    font = QFont()
    font.setPixelSize(14)
    font.setBold(True)
    painter.setFont(font)
    painter.setPen(QPen(QColor("#101820"), 3))
    painter.drawText(QPointF(position.x(), position.y() - 12.0), text)
    painter.setPen(QPen(QColor("#F7F4EA"), 1))
    painter.drawText(QPointF(position.x(), position.y() - 12.0), text)


def _overlay_outline_color(color: QColor) -> QColor:
    if color.lightnessF() > 0.58:
        return QColor("#101820")
    return QColor("#F7F4EA")


def _draw_scale_ticks(
    painter: QPainter,
    start_point: QPointF,
    end_point: QPointF,
    *,
    foreground_color: QColor,
    fg_width: float,
    tick_length: float,
) -> None:
    for anchor in (start_point, end_point):
        tick_start = QPointF(anchor.x(), anchor.y() - tick_length)
        tick_end = QPointF(anchor.x(), anchor.y() + tick_length)
        painter.setPen(QPen(foreground_color, fg_width, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawLine(tick_start, tick_end)


def draw_scale_overlay(
    painter: QPainter,
    document: ImageDocument,
    settings: AppSettings,
    *,
    image_width: int,
    image_height: int,
    image_to_output_scale: float,
    scale_bg_width: float,
    scale_fg_width: float,
    font_px: float,
    render_mode: str,
) -> None:
    scale_value = resolve_scale_overlay_value(
        document,
        settings,
        image_to_output_scale=image_to_output_scale,
    )
    if scale_value is None:
        return
    value, unit, bar_px = scale_value
    font, resolved_font_px = scale_overlay_font(settings, suggested_font_px=font_px, render_mode=render_mode)
    line_color = QColor(settings.scale_overlay_color)
    text_color = QColor(settings.scale_overlay_text_color)
    text_outline = _overlay_outline_color(text_color)
    start_point, draw_below = _scale_overlay_start(
        document,
        settings,
        image_width=image_width,
        image_height=image_height,
        bar_px=bar_px,
        font_px=resolved_font_px,
        image_to_output_scale=image_to_output_scale,
    )
    end_point = QPointF(start_point.x() + bar_px, start_point.y())
    bg_width = scale_bg_width
    fg_width = scale_fg_width
    cap_style = Qt.PenCapStyle.RoundCap
    if settings.scale_overlay_style == ScaleOverlayStyle.BAR:
        bg_width = max(bg_width * 2.2, fg_width * 2.0)
        fg_width = max(fg_width * 1.9, scale_fg_width + 1.5)
        cap_style = Qt.PenCapStyle.SquareCap
    painter.setPen(QPen(line_color, fg_width, Qt.PenStyle.SolidLine, cap_style))
    painter.drawLine(start_point, end_point)
    if settings.scale_overlay_style == ScaleOverlayStyle.TICKS:
        tick_length = max(resolved_font_px * 0.34, fg_width * 2.4, 6.0)
        _draw_scale_ticks(
            painter,
            start_point,
            end_point,
            foreground_color=line_color,
            fg_width=max(1.0, fg_width * 0.8),
            tick_length=tick_length,
        )
    painter.setFont(font)
    metrics = QFontMetricsF(font)
    text = f"{value:g} {unit}"
    text_top = start_point.y() + max(resolved_font_px * 0.45, 10.0) if draw_below else start_point.y() - metrics.height() - max(6.0, resolved_font_px * 0.18)
    text_rect = QRectF(start_point.x(), text_top, bar_px, metrics.height())
    painter.setPen(QPen(text_outline, 3))
    painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, text)
    painter.setPen(QPen(text_color, 1))
    painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, text)


def resolve_scale_overlay_value(
    document: ImageDocument,
    settings: AppSettings,
    *,
    image_to_output_scale: float,
) -> tuple[float, str, float] | None:
    value = float(settings.scale_overlay_length_value)
    if value <= 0:
        return None
    calibration = document.calibration
    if calibration is None:
        return value, "px", value * image_to_output_scale
    return value, calibration.unit, calibration.unit_to_px(value) * image_to_output_scale


def _scale_overlay_start(
    document: ImageDocument,
    settings: AppSettings,
    *,
    image_width: int,
    image_height: int,
    bar_px: float,
    font_px: float,
    image_to_output_scale: float,
) -> tuple[QPointF, bool]:
    margin = max(24.0, min(image_width, image_height) * 0.04)
    placement = settings.scale_overlay_placement_mode
    if placement == ScaleOverlayPlacementMode.MANUAL and document.scale_overlay_anchor is not None:
        point = QPointF(
            document.scale_overlay_anchor.x * image_to_output_scale,
            document.scale_overlay_anchor.y * image_to_output_scale,
        )
        clamped = QPointF(
            min(max(margin, point.x()), max(margin, image_width - margin - bar_px)),
            min(max(margin + font_px, point.y()), max(margin + font_px, image_height - margin)),
        )
        return clamped, clamped.y() <= (image_height * 0.22)
    if placement == ScaleOverlayPlacementMode.TOP_LEFT:
        return QPointF(margin, margin + font_px + 6.0), True
    if placement == ScaleOverlayPlacementMode.TOP_RIGHT:
        return QPointF(max(margin, image_width - margin - bar_px), margin + font_px + 6.0), True
    if placement == ScaleOverlayPlacementMode.BOTTOM_RIGHT:
        return QPointF(max(margin, image_width - margin - bar_px), image_height - margin), False
    return QPointF(margin, image_height - margin), False


def _draw_circle_endpoint(painter: QPainter, point: QPointF, color: QColor, radius: float) -> None:
    painter.setBrush(QColor("#0B0B0B"))
    painter.setPen(Qt.PenStyle.NoPen)
    painter.drawEllipse(point, radius * 1.15, radius * 1.15)
    painter.setBrush(color)
    painter.drawEllipse(point, radius * 0.72, radius * 0.72)


def _draw_bar_endpoint(painter: QPainter, point: QPointF, axis: tuple[float, float], color: QColor, line_width: float) -> None:
    tangent = _normal(axis)
    length = max(6.0, line_width * 3.4)
    start = QPointF(point.x() - tangent[0] * length, point.y() - tangent[1] * length)
    end = QPointF(point.x() + tangent[0] * length, point.y() + tangent[1] * length)
    painter.setPen(QPen(QColor("#0B0B0B"), line_width * 1.8, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
    painter.drawLine(start, end)
    painter.setPen(QPen(color, max(1.0, line_width * 0.9), Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
    painter.drawLine(start, end)


def _draw_arrow_endpoint(
    painter: QPainter,
    point: QPointF,
    axis: tuple[float, float],
    color: QColor,
    *,
    inward: bool,
    is_start: bool,
    line_width: float,
) -> None:
    direction_sign = 1.0 if (inward == is_start) else -1.0
    tip_dir = (-axis[0] * direction_sign, -axis[1] * direction_sign)
    side = _normal(tip_dir)
    arrow_length = max(8.0, line_width * 4.0)
    half_width = max(4.0, line_width * 2.4)
    tip = point
    tail = QPointF(point.x() - tip_dir[0] * arrow_length, point.y() - tip_dir[1] * arrow_length)
    left = QPointF(tail.x() + side[0] * half_width, tail.y() + side[1] * half_width)
    right = QPointF(tail.x() - side[0] * half_width, tail.y() - side[1] * half_width)
    painter.setPen(QPen(QColor("#0B0B0B"), line_width * 1.8, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
    painter.drawLine(tip, left)
    painter.drawLine(tip, right)
    painter.setPen(QPen(color, max(1.0, line_width * 0.9), Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
    painter.drawLine(tip, left)
    painter.drawLine(tip, right)


def _normalize(x: float, y: float) -> tuple[float, float]:
    length = math.hypot(x, y)
    if length <= 1e-9:
        return 1.0, 0.0
    return x / length, y / length


def _normal(axis: tuple[float, float]) -> tuple[float, float]:
    return -axis[1], axis[0]
