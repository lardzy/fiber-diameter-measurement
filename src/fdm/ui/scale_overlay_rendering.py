"""One immutable layout in source pixels, shared by canvas and image exports."""

from __future__ import annotations

import math
from functools import lru_cache

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QFontMetricsF, QPainter, QPainterPath

from fdm.scale_overlay import ScaleOverlayLayout, ScaleOverlaySpec, nice_length
from fdm.units import millimeters_per_unit, resolve_length_unit


def scale_font(spec: ScaleOverlaySpec, size: int) -> QFont:
    font = QFont(spec.font_family)
    font.setPixelSize(size)
    font.setBold(spec.bold)
    return font


@lru_cache(maxsize=256)
def layout_scale_overlay(
    spec: ScaleOverlaySpec,
    target: tuple[float, float, float, float],
    pixels_per_unit: float | None = None,
    calibration_unit: str | None = None,
) -> ScaleOverlayLayout:
    x, y, width, height = target
    if not all(math.isfinite(v) for v in target) or width <= 0 or height <= 0:
        raise ValueError("比例尺预览范围为空。")
    unit = spec.unit
    if pixels_per_unit is None:
        unit, pixels_per_display_unit = "px", 1.0
        if spec.length_mode == "custom" and spec.unit != "px":
            raise ValueError(
                "图片尚未标定，无法使用固定物理长度；请去标定或恢复自动像素长度。"
            )
    else:
        if unit == "px" and spec.length_mode == "auto":
            unit = calibration_unit or "um"
        origin_factor = millimeters_per_unit(calibration_unit or "")
        display_factor = millimeters_per_unit(unit)
        if spec.unit == "px" and spec.length_mode == "custom":
            raise ValueError(
                "已标定图片不能套用固定像素比例尺，请选择物理单位或恢复自动长度。"
            )
        if origin_factor is None or display_factor is None or pixels_per_unit <= 0:
            raise ValueError("比例尺标定单位无效。")
        pixels_per_display_unit = pixels_per_unit * display_factor / origin_factor
    value = (
        nice_length(width * 0.2 / pixels_per_display_unit)
        if spec.length_mode == "auto"
        else spec.length
    )
    bar_width = value * pixels_per_display_unit
    font_size = (
        max(12, round(min(width, height) * 0.015))
        if spec.font_mode == "auto"
        else max(1, round(spec.font_size))
    )
    font = scale_font(spec, font_size)
    metrics = QFontMetricsF(font)
    definition = resolve_length_unit(unit)
    label = f"{value:.8g} {definition.symbol if definition else unit}"
    stroke = spec.line_width * (2 if spec.style == "bar" else 1)
    has_ticks = spec.style in ("ticks", "ticks_up", "ticks_down", "divisions")
    tick_height = max(font_size * 0.4, stroke * 2) if has_ticks else 0.0
    above, below = scale_bar_vertical_extents(spec.style, stroke, tick_height)
    gap = max(3.0, font_size * 0.2)
    # Include overhang, antialiasing, and the entire label when clamping movement.
    text_width = (
        max(metrics.horizontalAdvance(label), metrics.boundingRect(label).width()) + 4
    )
    content_width = max(bar_width, text_width) + 2
    content_height = metrics.height() + gap + above + below + 2
    margin = min(width, height) * 0.02
    if content_width > width - 2 * margin or content_height > height - 2 * margin:
        raise ValueError(
            "比例尺和文字超出目标范围；请缩短长度、减小字号或恢复自动设置。"
        )
    positions = {
        "top_left": (0, 0),
        "top_right": (1, 0),
        "bottom_left": (0, 1),
        "bottom_right": (1, 1),
    }
    rx, ry = positions.get(spec.position, (spec.relative_x, spec.relative_y))
    left = x + margin + (width - 2 * margin - content_width) * rx
    top = y + margin + (height - 2 * margin - content_height) * ry
    center = left + content_width / 2
    if spec.text_position == "above":
        text_top = top + 1
        line_y = top + 1 + metrics.height() + gap + above
    else:
        line_y = top + 1 + above
        text_top = line_y + below + gap
    return ScaleOverlayLayout(
        spec,
        target,
        (left, top, content_width, content_height),
        (center - text_width / 2, text_top, text_width, metrics.height()),
        (center - bar_width / 2, line_y),
        (center + bar_width / 2, line_y),
        label,
        value,
        unit,
        font_size,
        stroke,
        tick_height,
    )


def scale_bar_vertical_extents(
    style: str, stroke: float, tick_height: float
) -> tuple[float, float]:
    """Distances above/below the horizontal baseline, including the fill."""
    if style in ("ticks_up", "divisions"):
        return max(stroke / 2, tick_height - stroke / 2), stroke / 2
    if style == "ticks_down":
        return stroke / 2, max(stroke / 2, tick_height - stroke / 2)
    return max(stroke, tick_height) / 2, max(stroke, tick_height) / 2


def scale_bar_path(layout: ScaleOverlayLayout) -> QPainterPath:
    """Filled, local-coordinate geometry with a calibrated outer-edge span.

    End ticks extend inward, so their pen width can never add length. Division
    marks use x0 + fraction * total_length, never independently rounded steps.
    One non-overlapping outline avoids seams where antialiased shapes meet.
    This is an explicit application convention, not an asserted ISO style.
    """
    origin_x, origin_y, _, _ = layout.target
    # Subtract the source origin before rasterization so a digital-slide global
    # coordinate does not change the renderer's subpixel edge rounding.
    start_x = round(layout.start[0] - origin_x, 8)
    end_x = round(layout.end[0] - origin_x, 8)
    baseline = round(layout.start[1] - origin_y, 8)
    width = end_x - start_x
    if width <= 0:
        # A length below the local-coordinate precision has no visible area.
        return QPainterPath()
    top, bottom = baseline - layout.stroke / 2, baseline + layout.stroke / 2
    # Store explicit edges, not rectangle origin + extent: repeatedly adding
    # fractional extents can create almost coincident edges during path boolean
    # operations. Qt's simplified()/united() can then remove the baseline.
    parts = [(start_x, end_x, top, bottom)]
    if not layout.tick_height:
        path = QPainterPath()
        path.addRect(QRectF(start_x, top, width, layout.stroke))
        return path

    above, below = scale_bar_vertical_extents(
        layout.spec.style, layout.stroke, layout.tick_height
    )
    # If a physical length is subpixel or shorter than the requested stroke,
    # intersect its end ticks with the actual span instead of expanding it.
    tick_width = min(layout.stroke, width / 2)
    for left, right in ((start_x, start_x + tick_width), (end_x - tick_width, end_x)):
        parts.append((left, right, baseline - above, baseline + below))
    if layout.spec.style == "divisions":
        for fraction in (0.25, 0.5, 0.75):
            x = start_x + width * fraction
            division_width = min(layout.stroke, width / 8)
            # The middle division is as tall as the endpoints; quarter marks
            # are shorter. All refer to the same zero-to-total physical span.
            height = layout.tick_height * (1.0 if fraction == 0.5 else 0.65)
            parts.append(
                (
                    x - division_width / 2,
                    x + division_width / 2,
                    baseline + below - height,
                    baseline + below,
                )
            )

    # Every part meets the continuous horizontal baseline. The shape therefore
    # has exactly one top and bottom at each x; construct that outline directly
    # instead of relying on floating-point polygon boolean operations.
    edges = sorted({edge for part in parts for edge in part[:2]})
    strips = []
    for left, right in zip(edges, edges[1:]):
        covering = [part for part in parts if part[0] < right and part[1] > left]
        strips.append(
            (
                left,
                right,
                min(part[2] for part in covering),
                max(part[3] for part in covering),
            )
        )
    path = QPainterPath()
    path.moveTo(strips[0][0], strips[0][2])
    for left, right, top, _ in strips:
        path.lineTo(left, top)
        path.lineTo(right, top)
    for left, right, _, bottom in reversed(strips):
        path.lineTo(right, bottom)
        path.lineTo(left, bottom)
    path.closeSubpath()
    return path


def paint_scale_overlay(painter: QPainter, layout: ScaleOverlayLayout) -> None:
    """Caller supplies the source-to-output transform. DPR is never geometry."""
    painter.save()
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
    origin_x, origin_y, width, height = layout.target
    # Work near the local origin: glyph subpixel rounding must not depend on a
    # large digital-slide coordinate. Eight decimals retain image-pixel accuracy.
    painter.translate(origin_x, origin_y)

    def local_point(point):
        return QPointF(round(point[0] - origin_x, 8), round(point[1] - origin_y, 8))

    painter.setClipRect(QRectF(0.0, 0.0, width, height), Qt.ClipOperation.IntersectClip)
    painter.fillPath(scale_bar_path(layout), QColor(layout.spec.color))
    painter.setFont(scale_font(layout.spec, layout.font_size))
    painter.setPen(QColor(layout.spec.text_color))
    text_origin = local_point(layout.text_rect)
    painter.drawText(
        QRectF(
            text_origin.x(), text_origin.y(), layout.text_rect[2], layout.text_rect[3]
        ),
        Qt.AlignmentFlag.AlignCenter,
        layout.label,
    )
    painter.restore()


def moved_spec(layout: ScaleOverlayLayout, left: float, top: float) -> ScaleOverlaySpec:
    from dataclasses import replace

    x, y, width, height = layout.target
    margin = min(width, height) * 0.02
    travel_x = width - 2 * margin - layout.bounds[2]
    travel_y = height - 2 * margin - layout.bounds[3]
    return replace(
        layout.spec,
        position="manual",
        relative_x=min(1.0, max(0.0, (left - x - margin) / max(1e-9, travel_x))),
        relative_y=min(1.0, max(0.0, (top - y - margin) / max(1e-9, travel_y))),
    )
