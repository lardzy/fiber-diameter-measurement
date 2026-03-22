from __future__ import annotations

from dataclasses import dataclass
import math

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QFontMetricsF, QPainter, QPen

from fdm.geometry import Line, Point, direction, normal
from fdm.models import ImageDocument, Measurement, TextAnnotation, format_measurement_label_value
from fdm.settings import AppSettings, MeasurementEndpointStyle, ScaleOverlayPlacementMode


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
    value = measurement.diameter_unit
    if value is None:
        value = measurement.diameter_px or 0.0
    unit = document.calibration.unit if document.calibration else "px"
    decimals = settings.measurement_label_decimals if settings is not None else 4
    return format_measurement_label_value(value, unit, decimals)


def measurement_label_font(settings: AppSettings) -> QFont:
    font = QFont()
    font.setFamily(settings.measurement_label_font_family)
    font.setPixelSize(int(max(8, settings.measurement_label_font_size)))
    font.setBold(True)
    return font


def text_annotation_font(settings: AppSettings) -> QFont:
    font = QFont()
    font.setFamily(settings.text_font_family)
    font.setPixelSize(int(max(8, settings.text_font_size)))
    return font


def _text_layout(font: QFont, content: str) -> tuple[QFontMetricsF, list[str], float, float]:
    metrics = QFontMetricsF(font)
    lines = content.splitlines() or [""]
    width = max(metrics.horizontalAdvance(line or " ") for line in lines)
    height = max(1.0, metrics.lineSpacing() * len(lines))
    return metrics, lines, width, height


def annotation_rect(annotation: TextAnnotation, settings: AppSettings, image_to_output) -> QRectF:
    font = text_annotation_font(settings)
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


def draw_text_annotations(
    painter: QPainter,
    document: ImageDocument,
    image_to_output,
    settings: AppSettings,
    *,
    selected_text_id: str | None = None,
) -> None:
    font = text_annotation_font(settings)
    painter.setFont(font)
    metrics, _lines_template, _width, _height = _text_layout(font, " ")
    line_spacing = metrics.lineSpacing()
    for annotation in document.text_annotations:
        rect = annotation_rect(annotation, settings, image_to_output)
        top_left = rect.topLeft() + QPointF(6.0, 4.0)
        painter.setPen(QPen(QColor("#101820"), 3))
        y = top_left.y() + metrics.ascent()
        for line in annotation.content.splitlines() or [""]:
            painter.drawText(QPointF(top_left.x(), y), line)
            y += line_spacing
        painter.setPen(QPen(QColor(settings.text_color), 1))
        y = top_left.y() + metrics.ascent()
        for line in annotation.content.splitlines() or [""]:
            painter.drawText(QPointF(top_left.x(), y), line)
            y += line_spacing
        if annotation.id == selected_text_id:
            painter.setBrush(QColor(0, 0, 0, 0))
            painter.setPen(QPen(QColor("#F4D35E"), 1.8, Qt.PenStyle.DashLine))
            painter.drawRoundedRect(rect, 6.0, 6.0)


def draw_measurements(
    painter: QPainter,
    document: ImageDocument,
    image_to_output,
    settings: AppSettings,
    *,
    line_width: float,
    endpoint_radius: float,
    selected_measurement_id: str | None = None,
) -> None:
    for measurement in document.measurements:
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
    target_output_px: float,
) -> None:
    scale_value = nice_scale_value(document, target_output_px=target_output_px, image_to_output_scale=image_to_output_scale)
    if scale_value is None:
        return
    value, bar_px = scale_value
    start_point, draw_below = _scale_overlay_start(
        document,
        settings,
        image_width=image_width,
        image_height=image_height,
        bar_px=bar_px,
        font_px=font_px,
        image_to_output_scale=image_to_output_scale,
    )
    painter.setPen(QPen(QColor("#111111"), scale_bg_width, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
    painter.drawLine(start_point, QPointF(start_point.x() + bar_px, start_point.y()))
    painter.setPen(QPen(QColor("#FFFFFF"), scale_fg_width, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
    painter.drawLine(start_point, QPointF(start_point.x() + bar_px, start_point.y()))
    font = QFont(painter.font())
    font.setPixelSize(int(round(font_px)))
    font.setBold(True)
    painter.setFont(font)
    text_y = start_point.y() + max(font_px * 1.1, 18.0) if draw_below else start_point.y() - max(12.0, font_px * 0.55)
    painter.setPen(QPen(QColor("#101820"), 3))
    painter.drawText(QPointF(start_point.x(), text_y), f"{value:g} {document.calibration.unit}")
    painter.setPen(QPen(QColor("#FFFFFF"), 1))
    painter.drawText(QPointF(start_point.x(), text_y), f"{value:g} {document.calibration.unit}")


def nice_scale_value(
    document: ImageDocument,
    *,
    target_output_px: float,
    image_to_output_scale: float,
) -> tuple[float, float] | None:
    calibration = document.calibration
    if calibration is None:
        return None
    scaled_pixels_per_unit = calibration.pixels_per_unit * max(image_to_output_scale, 1e-9)
    raw_value = target_output_px / scaled_pixels_per_unit
    if raw_value <= 0:
        return None
    exponent = math.floor(math.log10(raw_value))
    base = raw_value / (10 ** exponent)
    if base < 2:
        nice_base = 1
    elif base < 5:
        nice_base = 2
    else:
        nice_base = 5
    nice_value = nice_base * (10 ** exponent)
    return nice_value, calibration.unit_to_px(nice_value) * image_to_output_scale


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
