from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence, Set as AbstractSet
from dataclasses import dataclass
from functools import lru_cache
import math

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (
    QColor,
    QFont,
    QFontMetricsF,
    QImage,
    QPainter,
    QPainterPath,
    QPen,
    QPolygonF,
    QStaticText,
    QTransform,
)
from PySide6.QtWidgets import QWidget

from fdm.area_display import (
    AREA_GEOMETRY_RAW,
    AREA_GEOMETRY_SCREEN,
    AreaProxyBuildBudget,
    area_derived_geometry_service,
    area_geometry_raw,
)
from fdm.geometry import Line, Point, direction, normal, point_to_segment_distance
from fdm.models import (
    ImageDocument,
    Measurement,
    OverlayAnnotation,
    OverlayAnnotationKind,
    OverlayTextAnchorAlignment,
    OverlayTextSizeSpace,
    TextAnnotation,
    format_measurement_label_value,
)
from fdm.settings import AppSettings, MeasurementEndpointStyle, ScaleOverlayPlacementMode, ScaleOverlayStyle
from fdm.ui.area_handle_cache import area_handle_display_cache
from fdm.ui.canvas_overlay_cache import (
    AreaOverlayDrawCommand,
    AreaOverlayLabelCommand,
)
from fdm.ui.screen_label_sprite_cache import (
    ScreenLabelSprite,
    screen_label_sprite_cache,
)

_TEXT_LAYOUT_MAX_ENTRIES = 2048
_TEXT_LAYOUT_MAX_CHARACTERS = 128 * 1024
_OVERLAY_TEXT_MIN_OUTPUT_FONT_PX = 1
_OVERLAY_TEXT_MAX_OUTPUT_FONT_PX = 8192
_OVERLAY_TEXT_SELECTION_PADDING_X = 6.0
_OVERLAY_TEXT_SELECTION_PADDING_Y = 4.0
MEASUREMENT_DECORATION_PADDING_SCREEN_PX = 48.0


@dataclass(slots=True)
class _CachedTextLine:
    static_text: QStaticText
    width: float
    top: float
    baseline: float


@dataclass(slots=True)
class _CachedTextLayout:
    metrics: QFontMetricsF
    lines: tuple[_CachedTextLine, ...]
    width: float
    height: float
    character_count: int


@dataclass(slots=True)
class ResolvedOverlayTextLayout:
    """One authoritative layout shared by paint, culling and hit testing."""

    font: QFont
    layout: _CachedTextLayout
    anchor: QPointF
    text_rect: QRectF
    annotation_rect: QRectF
    image_to_output_scale: float


_TEXT_LAYOUT_CACHE: OrderedDict[tuple[object, ...], _CachedTextLayout] = OrderedDict()
_TEXT_LAYOUT_CACHE_CHARACTERS = 0


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
    if measurement.appearance is not None and measurement.appearance.stroke_color:
        return QColor(measurement.appearance.stroke_color)
    group = document.get_group(measurement.fiber_group_id)
    return QColor(group.color if group else settings.default_measurement_color)


def measurement_line_width(measurement: Measurement, suggested_line_width: float) -> float:
    """Scale a logical per-object width through the active screen/export metric."""
    if measurement.appearance is None or measurement.appearance.stroke_width is None:
        return suggested_line_width
    return max(0.25, suggested_line_width * (measurement.appearance.stroke_width / 2.0))


def measurement_marker_scale(measurement: Measurement) -> float:
    if measurement.appearance is None or measurement.appearance.marker_scale is None:
        return 1.0
    return measurement.appearance.marker_scale


def measurement_text_color(measurement: Measurement, fallback: str) -> QColor:
    if measurement.appearance is not None and measurement.appearance.text_color:
        return QColor(measurement.appearance.text_color)
    return QColor(fallback)


def measurement_display_text(measurement: Measurement, document: ImageDocument) -> str:
    return measurement_display_text_with_settings(measurement, document, None)


def measurement_display_text_with_settings(
    measurement: Measurement,
    document: ImageDocument,
    settings: AppSettings | None,
) -> str:
    value = measurement.display_value()
    unit = measurement.display_unit(document.calibration)
    style = _measurement_label_style(settings, measurement) if settings is not None else None
    decimals = int(
        getattr(
            style,
            "decimals",
            getattr(settings, "measurement_label_decimals", 4) if settings is not None else 4,
        )
    )
    return format_measurement_label_value(value, unit, decimals)


@lru_cache(maxsize=256)
def _cached_measurement_label_font(font_family: str, font_size: int) -> QFont:
    font = QFont()
    font.setFamily(font_family)
    font.setPixelSize(font_size)
    font.setBold(True)
    return font


def measurement_label_font(settings: AppSettings, measurement: Measurement | None = None) -> QFont:
    style = _measurement_label_style(settings, measurement)
    appearance = measurement.appearance if measurement is not None else None
    font_family = (
        appearance.font_family
        if appearance is not None and appearance.font_family
        else str(
            getattr(
                style,
                "font_family",
                getattr(settings, "measurement_label_font_family", "Segoe UI"),
            )
        )
    )
    font_size = int(
        max(
            8,
            appearance.font_size
            if appearance is not None and appearance.font_size is not None
            else getattr(
                style,
                "font_size",
                getattr(settings, "measurement_label_font_size", 16),
            ),
        )
    )
    return _cached_measurement_label_font(font_family, font_size)


def _measurement_label_style(settings: AppSettings | None, measurement: Measurement | None):
    if settings is None:
        return None
    attribute = (
        "area_measurement_label_style"
        if measurement is not None and measurement.measurement_kind == "area"
        else "length_measurement_label_style"
    )
    return getattr(settings, attribute, None)


def _measurement_label_enabled(settings: AppSettings, measurement: Measurement) -> bool:
    style = _measurement_label_style(settings, measurement)
    return bool(
        getattr(
            style,
            "enabled",
            getattr(settings, "show_measurement_labels", True),
        )
    )


def _measurement_label_color(settings: AppSettings, measurement: Measurement) -> str:
    style = _measurement_label_style(settings, measurement)
    return str(
        getattr(
            style,
            "color",
            getattr(settings, "measurement_label_color", "#FFFFFF"),
        )
    )


def _measurement_label_background_enabled(settings: AppSettings, measurement: Measurement) -> bool:
    style = _measurement_label_style(settings, measurement)
    return bool(
        getattr(
            style,
            "background_enabled",
            getattr(settings, "measurement_label_background_enabled", True),
        )
    )


def _measurement_label_parallel_to_line(settings: AppSettings, measurement: Measurement) -> bool:
    style = _measurement_label_style(settings, measurement)
    return bool(
        getattr(
            style,
            "parallel_to_line",
            getattr(settings, "measurement_label_parallel_to_line", False),
        )
    )


def area_rings_path(area_rings: list[list[Point]], image_to_output) -> QPainterPath:
    path = QPainterPath()
    path.setFillRule(Qt.FillRule.OddEvenFill)
    for ring in area_rings:
        if len(ring) < 3:
            continue
        polygon = QPolygonF([image_to_output(point) for point in ring])
        if polygon.size() >= 3:
            path.addPolygon(polygon)
            path.closeSubpath()
    return path


def _image_to_output_transform(image_to_output) -> QTransform:
    """Build the affine transform represented by the rendering callback."""

    samples = (
        image_to_output(Point(0.0, 0.0)),
        image_to_output(Point(1.0, 0.0)),
        image_to_output(Point(0.0, 1.0)),
    )
    origin, unit_x, unit_y = samples
    return QTransform(
        float(unit_x.x() - origin.x()),
        float(unit_x.y() - origin.y()),
        float(unit_y.x() - origin.x()),
        float(unit_y.y() - origin.y()),
        float(origin.x()),
        float(origin.y()),
    )


def _image_to_output_scale(transform: QTransform) -> float:
    scale_x = math.hypot(transform.m11(), transform.m12())
    scale_y = math.hypot(transform.m21(), transform.m22())
    return max(1e-9, math.sqrt(max(1e-18, scale_x * scale_y)))


def _area_geometry_and_output_path(
    measurement: Measurement,
    image_to_output,
    *,
    selected: bool,
    geometry_mode: str,
    proxy_build_budget: AreaProxyBuildBudget | None = None,
):
    transform = _image_to_output_transform(image_to_output)
    if geometry_mode == AREA_GEOMETRY_RAW:
        geometry = area_derived_geometry_service.raw_geometry(measurement)
    else:
        geometry = area_derived_geometry_service.screen_geometry(
            measurement,
            zoom=_image_to_output_scale(transform),
            selected=selected,
            build_budget=proxy_build_budget,
        )
    # Keep the cached path in image coordinates.  Mapping every element for
    # every tile made a 600k-vertex object proportional to the number of
    # visible tiles.  QPainter can replay the same immutable path under this
    # affine transform without touching persistent geometry.
    return geometry, transform, geometry.path


def _area_handle_points_for_display(
    rings: list[list[Point]],
    *,
    output_scale: float,
    spacing_px: float = 8.0,
) -> list[Point]:
    """Thin only painted handles; exact hit testing retains every raw vertex."""

    cell_size = max(1e-9, float(spacing_px) / max(float(output_scale), 1e-9))
    visible: list[Point] = []
    occupied: set[tuple[int, int]] = set()
    for ring in rings:
        for point in ring:
            key = (
                math.floor(point.x / cell_size),
                math.floor(point.y / cell_size),
            )
            if key in occupied:
                continue
            occupied.add(key)
            visible.append(point)
    return visible


def overlay_text_font(settings: AppSettings, annotation: OverlayAnnotation | None = None) -> QFont:
    font = QFont()
    appearance = annotation.appearance if annotation is not None else None
    font.setFamily(
        appearance.font_family
        if appearance is not None and appearance.font_family
        else settings.text_font_family
    )
    font.setPixelSize(
        int(
            max(
                8,
                appearance.font_size
                if appearance is not None and appearance.font_size is not None
                else settings.text_font_size,
            )
        )
    )
    return font


def overlay_annotation_line_width(
    settings: AppSettings,
    *,
    suggested_line_width: float,
    render_mode: str,
    annotation: OverlayAnnotation | None = None,
) -> float:
    appearance = annotation.appearance if annotation is not None else None
    base_width = float(
        max(
            0.5,
            appearance.stroke_width
            if appearance is not None and appearance.stroke_width is not None
            else settings.overlay_line_width,
        )
    )
    if render_mode == "full_resolution":
        return base_width
    # Object/global values are logical screen pixels.  Preserve their full
    # relative range while still following the active renderer's baseline
    # metric when a scaled preview supplies one.
    metric_scale = max(0.5, float(suggested_line_width) / 2.2)
    return max(0.5, base_width * metric_scale)


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


def _font_cache_key(font: QFont) -> tuple[object, ...]:
    return (
        font.family(),
        font.styleName(),
        font.pixelSize(),
        round(font.pointSizeF(), 3),
        font.weight(),
        font.italic(),
        font.underline(),
        font.strikeOut(),
        font.letterSpacingType().value,
        round(font.letterSpacing(), 3),
    )


def _cached_text_layout(font: QFont, content: str, *, render_mode: str = "default") -> _CachedTextLayout:
    global _TEXT_LAYOUT_CACHE_CHARACTERS

    normalized_content = str(content)
    key = (normalized_content, *_font_cache_key(font), str(render_mode))
    cached = _TEXT_LAYOUT_CACHE.get(key)
    if cached is not None:
        _TEXT_LAYOUT_CACHE.move_to_end(key)
        return cached

    metrics = QFontMetricsF(font)
    source_lines = normalized_content.splitlines() or [""]
    line_spacing = max(1.0, metrics.lineSpacing())
    cached_lines: list[_CachedTextLine] = []
    width = 0.0
    for index, line in enumerate(source_lines):
        line_width = metrics.horizontalAdvance(line or " ")
        static_text = QStaticText(line)
        try:
            static_text.setPerformanceHint(QStaticText.PerformanceHint.AggressiveCaching)
        except AttributeError:
            pass
        static_text.prepare(QTransform(), font)
        top = index * line_spacing
        cached_lines.append(
            _CachedTextLine(
                static_text=static_text,
                width=line_width,
                top=top,
                baseline=top + metrics.ascent(),
            )
        )
        width = max(width, line_width)
    height = max(1.0, ((len(source_lines) - 1) * line_spacing) + metrics.height())
    layout = _CachedTextLayout(
        metrics=metrics,
        lines=tuple(cached_lines),
        width=width,
        height=height,
        character_count=len(normalized_content),
    )

    if layout.character_count > _TEXT_LAYOUT_MAX_CHARACTERS:
        return layout
    while _TEXT_LAYOUT_CACHE and (
        len(_TEXT_LAYOUT_CACHE) >= _TEXT_LAYOUT_MAX_ENTRIES
        or _TEXT_LAYOUT_CACHE_CHARACTERS + layout.character_count > _TEXT_LAYOUT_MAX_CHARACTERS
    ):
        _old_key, old = _TEXT_LAYOUT_CACHE.popitem(last=False)
        _TEXT_LAYOUT_CACHE_CHARACTERS -= old.character_count
    _TEXT_LAYOUT_CACHE[key] = layout
    _TEXT_LAYOUT_CACHE_CHARACTERS += layout.character_count
    return layout


def _overlay_text_output_anchor_and_scale(
    image_to_output,
    anchor_px: Point,
) -> tuple[QPointF, float]:
    """Resolve the local logical-output scale without involving device DPR."""

    transform = _image_to_output_transform(image_to_output)
    return (
        transform.map(QPointF(anchor_px.x, anchor_px.y)),
        _image_to_output_scale(transform),
    )


def _overlay_text_anchor_fractions(alignment: str) -> tuple[float, float]:
    return {
        OverlayTextAnchorAlignment.TOP_LEFT: (0.0, 0.0),
        OverlayTextAnchorAlignment.TOP_CENTER: (0.5, 0.0),
        OverlayTextAnchorAlignment.TOP_RIGHT: (1.0, 0.0),
        OverlayTextAnchorAlignment.CENTER_LEFT: (0.0, 0.5),
        OverlayTextAnchorAlignment.CENTER: (0.5, 0.5),
        OverlayTextAnchorAlignment.CENTER_RIGHT: (1.0, 0.5),
        OverlayTextAnchorAlignment.BOTTOM_LEFT: (0.0, 1.0),
        OverlayTextAnchorAlignment.BOTTOM_CENTER: (0.5, 1.0),
        OverlayTextAnchorAlignment.BOTTOM_RIGHT: (1.0, 1.0),
    }.get(alignment, (0.0, 0.0))


def resolve_overlay_text_layout(
    annotation: OverlayAnnotation,
    settings: AppSettings,
    image_to_output,
    *,
    render_mode: str = "overlay",
) -> ResolvedOverlayTextLayout:
    """Resolve font, anchor and complete multiline bounds exactly once.

    A missing layout specification is the legacy contract: the configured
    font size is already an output-pixel size and the image anchor is the text
    block's top-left corner. An explicit specification always owns its numeric
    font size; IMAGE_PX additionally scales that value through the same
    transform that maps its anchor.
    """

    anchor, output_scale = _overlay_text_output_anchor_and_scale(
        image_to_output,
        annotation.anchor_px,
    )
    font = overlay_text_font(settings, annotation)
    text_layout_spec = getattr(annotation, "text_layout", None)
    alignment = OverlayTextAnchorAlignment.TOP_LEFT
    if text_layout_spec is not None:
        alignment = text_layout_spec.anchor_alignment
        requested_font_px = float(text_layout_spec.image_font_size_px)
        if text_layout_spec.size_space == OverlayTextSizeSpace.IMAGE_PX:
            requested_font_px *= output_scale
        if not math.isfinite(requested_font_px):
            requested_font_px = float(_OVERLAY_TEXT_MIN_OUTPUT_FONT_PX)
        resolved_font_px = int(
            round(
                max(
                    float(_OVERLAY_TEXT_MIN_OUTPUT_FONT_PX),
                    min(
                        float(_OVERLAY_TEXT_MAX_OUTPUT_FONT_PX),
                        requested_font_px,
                    ),
                )
            )
        )
        font = QFont(font)
        font.setPixelSize(resolved_font_px)

    layout = _cached_text_layout(
        font,
        annotation.content,
        render_mode=f"overlay:{render_mode}",
    )
    horizontal_fraction, vertical_fraction = _overlay_text_anchor_fractions(
        alignment
    )
    text_rect = QRectF(
        anchor.x() - (layout.width * horizontal_fraction),
        anchor.y() - (layout.height * vertical_fraction),
        layout.width,
        layout.height,
    )
    selection_rect = text_rect.adjusted(
        -_OVERLAY_TEXT_SELECTION_PADDING_X,
        -_OVERLAY_TEXT_SELECTION_PADDING_Y,
        _OVERLAY_TEXT_SELECTION_PADDING_X,
        _OVERLAY_TEXT_SELECTION_PADDING_Y,
    )
    return ResolvedOverlayTextLayout(
        font=font,
        layout=layout,
        anchor=anchor,
        text_rect=text_rect,
        annotation_rect=selection_rect,
        image_to_output_scale=output_scale,
    )


def _text_layout(font: QFont, content: str) -> tuple[QFontMetricsF, list[str], float, float]:
    """Compatibility wrapper backed by the shared text-layout cache."""
    layout = _cached_text_layout(font, content)
    return layout.metrics, content.splitlines() or [""], layout.width, layout.height


def _painter_visible_rect(painter: QPainter) -> QRectF | None:
    has_clipping = getattr(painter, "hasClipping", None)
    if callable(has_clipping) and has_clipping():
        clipped = painter.clipBoundingRect()
        if clipped.isValid() and not clipped.isEmpty():
            return clipped
    viewport = getattr(painter, "viewport", None)
    return QRectF(viewport()) if callable(viewport) else None


def _is_visible_to_painter(painter: QPainter, rect: QRectF, *, padding: float = 4.0) -> bool:
    visible_rect = _painter_visible_rect(painter)
    return visible_rect is None or visible_rect.intersects(
        rect.adjusted(-padding, -padding, padding, padding)
    )


def _draw_cached_text(
    painter: QPainter,
    layout: _CachedTextLayout,
    top_left: QPointF,
    *,
    color: QColor,
    outline: QColor | None,
    horizontal_center: float | None = None,
) -> None:
    save = getattr(painter, "save", None)
    restore = getattr(painter, "restore", None)
    draw_static_text = getattr(painter, "drawStaticText", None)
    if callable(save):
        save()
    for line in layout.lines:
        x = top_left.x()
        if horizontal_center is not None:
            x = horizontal_center - (line.width / 2.0)
        anchor = QPointF(x, top_left.y() + line.top)
        if callable(draw_static_text):
            if outline is not None:
                painter.setPen(outline)
                for dx, dy in ((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)):
                    draw_static_text(QPointF(anchor.x() + dx, anchor.y() + dy), line.static_text)
            painter.setPen(color)
            draw_static_text(anchor, line.static_text)
            continue
        # Lightweight recording painters used by rendering tests may only
        # implement drawText. Keep their compatibility without bypassing the
        # static-text fast path used by real QPainter instances.
        baseline_anchor = QPointF(anchor.x(), top_left.y() + line.baseline)
        painter.setPen(color)
        painter.drawText(baseline_anchor, line.static_text.text())
    if callable(restore):
        restore()


def _painter_device_pixel_ratio(painter: QPainter) -> float:
    device_method = getattr(painter, "device", None)
    device = device_method() if callable(device_method) else None
    if device is None:
        return 1.0
    for attribute in ("devicePixelRatioF", "devicePixelRatio"):
        ratio_method = getattr(device, attribute, None)
        if callable(ratio_method):
            try:
                ratio = float(ratio_method())
            except (TypeError, ValueError):
                continue
            if math.isfinite(ratio) and ratio > 0.0:
                return ratio
    return 1.0


def _painter_targets_screen_widget(painter: QPainter) -> bool:
    device_method = getattr(painter, "device", None)
    device = device_method() if callable(device_method) else None
    return isinstance(device, QWidget)


def _screen_measurement_label_sprite(
    painter: QPainter,
    *,
    text: str,
    font: QFont,
    text_color: QColor,
    outline_color: QColor | None,
    background_color: QColor | None,
    arrangement_mode: str,
    use_sprite_cache: bool | None,
    sprite_device_pixel_ratio: float | None = None,
) -> ScreenLabelSprite | None:
    enabled = (
        _painter_targets_screen_widget(painter)
        if use_sprite_cache is None
        else bool(use_sprite_cache)
    )
    if not enabled or not callable(getattr(painter, "drawImage", None)):
        return None
    device_pixel_ratio = (
        float(sprite_device_pixel_ratio)
        if sprite_device_pixel_ratio is not None
        and math.isfinite(float(sprite_device_pixel_ratio))
        and float(sprite_device_pixel_ratio) > 0.0
        else _painter_device_pixel_ratio(painter)
    )
    return screen_label_sprite_cache.get_or_create(
        text=text,
        font=font,
        text_color=text_color,
        outline_color=outline_color,
        background_color=background_color,
        device_pixel_ratio=device_pixel_ratio,
        arrangement_mode=arrangement_mode,
    )


def _draw_screen_label_sprite(
    painter: QPainter,
    sprite: ScreenLabelSprite,
    rect: QRectF,
) -> None:
    # The sprite already carries its device-pixel ratio. Drawing at a point
    # preserves its native pixels; fitting it into a QRectF would resample the
    # ceil-rounded backing image on every frame.
    painter.drawImage(rect.topLeft(), sprite.image)


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


def overlay_annotation_rect(
    annotation: OverlayAnnotation,
    settings: AppSettings,
    image_to_output,
    *,
    render_mode: str = "overlay",
) -> QRectF:
    if annotation.normalized_kind() != OverlayAnnotationKind.TEXT:
        start = image_to_output(annotation.start_px)
        end = image_to_output(annotation.end_px)
        left = min(start.x(), end.x())
        top = min(start.y(), end.y())
        width = max(1.0, abs(end.x() - start.x()))
        height = max(1.0, abs(end.y() - start.y()))
        return QRectF(left, top, width, height)
    return resolve_overlay_text_layout(
        annotation,
        settings,
        image_to_output,
        render_mode=render_mode,
    ).annotation_rect


def annotation_rect(
    annotation: TextAnnotation | OverlayAnnotation,
    settings: AppSettings,
    image_to_output,
    *,
    render_mode: str = "overlay",
) -> QRectF:
    if isinstance(annotation, TextAnnotation):
        return overlay_annotation_rect(
            annotation.to_overlay(),
            settings,
            image_to_output,
            render_mode=render_mode,
        )
    return overlay_annotation_rect(
        annotation,
        settings,
        image_to_output,
        render_mode=render_mode,
    )


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
    annotations = list(getattr(document, "overlay_annotations", []))
    for annotation in annotations:
        kind = annotation.normalized_kind()
        if kind == OverlayAnnotationKind.TEXT:
            resolved_text = resolve_overlay_text_layout(
                annotation,
                settings,
                image_to_output,
                render_mode=render_mode,
            )
            text_color = QColor(
                annotation.appearance.text_color
                if annotation.appearance is not None and annotation.appearance.text_color
                else settings.text_color
            )
            painter.setFont(resolved_text.font)
            rect = resolved_text.annotation_rect
            if not _is_visible_to_painter(painter, rect, padding=4.0):
                continue
            _draw_cached_text(
                painter,
                resolved_text.layout,
                resolved_text.text_rect.topLeft(),
                color=text_color,
                outline=None,
            )
            if annotation.id == selected_overlay_id:
                painter.setBrush(QColor(0, 0, 0, 0))
                painter.setPen(QPen(QColor("#F4D35E"), 1.8, Qt.PenStyle.DashLine))
                painter.drawRoundedRect(rect, 6.0, 6.0)
            continue
        line_color = QColor(
            annotation.appearance.stroke_color
            if annotation.appearance is not None and annotation.appearance.stroke_color
            else settings.overlay_line_color
        )
        resolved_line_width = overlay_annotation_line_width(
            settings,
            suggested_line_width=2.2,
            render_mode=render_mode,
            annotation=annotation,
        )
        _draw_shape_overlay_annotation(
            painter,
            annotation,
            image_to_output,
            color=line_color,
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
    painter.setBrush(Qt.BrushStyle.NoBrush)
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
    pen = QPen(painter.pen())
    pen.setCapStyle(Qt.PenCapStyle.FlatCap)
    pen.setJoinStyle(Qt.PenJoinStyle.MiterJoin)
    painter.save()
    painter.setPen(pen)
    painter.drawLine(end_point, left)
    painter.drawLine(end_point, right)
    painter.restore()


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
    visible_rect: QRectF | None = None,
    area_geometry_mode: str = AREA_GEOMETRY_SCREEN,
    measurement_sequence: Sequence[Measurement] | None = None,
    count_numbers: Mapping[str, int] | None = None,
    excluded_measurement_ids: AbstractSet[str] | None = None,
    raw_area_measurement_ids: AbstractSet[str] | None = None,
    proxy_build_budget: AreaProxyBuildBudget | None = None,
    use_sprite_cache: bool | None = None,
    sprite_device_pixel_ratio: float | None = None,
    cull_by_geometry: bool = True,
) -> bool:
    if proxy_build_budget is None and area_geometry_mode == AREA_GEOMETRY_SCREEN:
        proxy_build_budget = AreaProxyBuildBudget(max_build_ms=12.0)
    count_measurements: list[tuple[Measurement, int]] = []
    selected_count_measurement: tuple[Measurement, int] | None = None
    visible_padding = measurement_geometry_cull_padding(
        image_to_output,
        endpoint_radius=endpoint_radius,
    )
    count_index = 0
    measurements = (
        document.measurements
        if measurement_sequence is None
        else measurement_sequence
    )
    excluded = excluded_measurement_ids or frozenset()
    raw_area_ids = raw_area_measurement_ids or frozenset()
    for measurement in measurements:
        selected = measurement.id == selected_measurement_id
        if measurement.measurement_kind == "count":
            count_index = (
                int(count_numbers[measurement.id])
                if count_numbers is not None and measurement.id in count_numbers
                else count_index + 1
            )
        if measurement.id in excluded:
            continue
        if (
            cull_by_geometry
            and visible_rect is not None
            and not selected
            and not _measurement_display_intersects_rect(
                measurement,
                document,
                settings,
                image_to_output,
                visible_rect,
                padding=visible_padding,
                count_number=count_index if measurement.measurement_kind == "count" else None,
                suggested_line_width=line_width,
                endpoint_radius=endpoint_radius,
            )
        ):
            continue
        if measurement.measurement_kind == "count":
            if selected:
                selected_count_measurement = (measurement, count_index)
            else:
                count_measurements.append((measurement, count_index))
            continue
        if measurement.measurement_kind == "area":
            draw_area_measurement(
                painter,
                document,
                measurement,
                image_to_output,
                settings,
                line_width=line_width,
                endpoint_radius=endpoint_radius,
                selected=selected,
                show_fill=show_area_fill,
                show_handles=show_area_handles and selected,
                geometry_mode=(
                    AREA_GEOMETRY_RAW
                    if measurement.id in raw_area_ids
                    else area_geometry_mode
                ),
                proxy_build_budget=proxy_build_budget,
                use_sprite_cache=use_sprite_cache,
                sprite_device_pixel_ratio=sprite_device_pixel_ratio,
            )
            continue
        if measurement.measurement_kind == "polyline":
            draw_polyline_measurement(
                painter,
                document,
                measurement,
                image_to_output,
                settings,
                line_width=line_width,
                endpoint_radius=endpoint_radius,
                selected=selected,
                use_sprite_cache=use_sprite_cache,
                sprite_device_pixel_ratio=sprite_device_pixel_ratio,
            )
            continue
        line = measurement.effective_line()
        start_point = image_to_output(line.start)
        end_point = image_to_output(line.end)
        color = measurement_color(document, measurement, settings)
        base_line_width = measurement_line_width(measurement, line_width)
        actual_width = base_line_width * (1.7 if selected else 1.0)
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
            endpoint_radius=(
                endpoint_radius
                * measurement_marker_scale(measurement)
                * (1.15 if selected else 1.0)
            ),
        )
        if _measurement_label_enabled(settings, measurement):
            if use_sprite_cache is None:
                draw_measurement_label(
                    painter,
                    measurement,
                    document,
                    settings,
                    start_point,
                    end_point,
                )
            else:
                label_kwargs = {"use_sprite_cache": use_sprite_cache}
                if sprite_device_pixel_ratio is not None:
                    label_kwargs["sprite_device_pixel_ratio"] = (
                        sprite_device_pixel_ratio
                    )
                draw_measurement_label(
                    painter,
                    measurement,
                    document,
                    settings,
                    start_point,
                    end_point,
                    **label_kwargs,
                )
    draw_count_measurements_batch(
        painter,
        document,
        count_measurements,
        image_to_output,
        settings,
        endpoint_radius=endpoint_radius,
        use_sprite_cache=use_sprite_cache,
        sprite_device_pixel_ratio=sprite_device_pixel_ratio,
    )
    if selected_count_measurement is not None:
        selected_measurement, selected_number = selected_count_measurement
        draw_count_measurement(
            painter,
            document,
            selected_measurement,
            image_to_output,
            settings,
            endpoint_radius=endpoint_radius,
            selected=True,
            count_number=selected_number,
            use_sprite_cache=use_sprite_cache,
            sprite_device_pixel_ratio=sprite_device_pixel_ratio,
        )
    return bool(proxy_build_budget is not None and proxy_build_budget.deferred)


def _measurement_intersects_rect(measurement: Measurement, rect: QRectF, *, padding: float) -> bool:
    bounds = _measurement_bounds(measurement)
    if bounds is None:
        return True
    left, top, right, bottom = bounds
    return (
        right >= rect.left() - padding
        and left <= rect.right() + padding
        and bottom >= rect.top() - padding
        and top <= rect.bottom() + padding
    )


def _measurement_display_intersects_rect(
    measurement: Measurement,
    document: ImageDocument,
    settings: AppSettings,
    image_to_output,
    rect: QRectF,
    *,
    padding: float,
    count_number: int | None,
    suggested_line_width: float = 2.0,
    endpoint_radius: float = 4.0,
) -> bool:
    """Cull against the complete object display envelope."""

    display_bounds = measurement_display_image_bounds(
        measurement,
        document,
        settings,
        image_to_output,
        suggested_line_width=suggested_line_width,
        endpoint_radius=endpoint_radius,
        count_number=count_number,
        minimum_image_padding=padding,
    )
    return display_bounds is not None and display_bounds.intersects(rect)


def measurement_display_intersects_rect(
    measurement: Measurement,
    document: ImageDocument,
    settings: AppSettings,
    image_to_output,
    rect: QRectF,
    *,
    padding: float,
    count_number: int | None = None,
    suggested_line_width: float = 2.0,
    endpoint_radius: float = 4.0,
) -> bool:
    """Public culling predicate shared by direct and worker snapshot paths."""

    return _measurement_display_intersects_rect(
        measurement,
        document,
        settings,
        image_to_output,
        rect,
        padding=padding,
        count_number=count_number,
        suggested_line_width=suggested_line_width,
        endpoint_radius=endpoint_radius,
    )


def measurement_geometry_cull_padding(
    image_to_output,
    *,
    endpoint_radius: float = 4.0,
) -> float:
    """Return conservative image-space padding for cosmetic measurement marks."""

    output_scale = _image_to_output_scale(
        _image_to_output_transform(image_to_output)
    )
    screen_padding = max(
        MEASUREMENT_DECORATION_PADDING_SCREEN_PX,
        float(endpoint_radius) * 8.0,
    )
    return screen_padding / output_scale


def measurement_label_image_bounds(
    measurement: Measurement,
    document: ImageDocument,
    settings: AppSettings,
    image_to_output,
    *,
    count_number: int | None = None,
    exact_area: bool = False,
) -> QRectF | None:
    """Return a conservative label envelope in persistent image coordinates."""

    return _measurement_label_image_bounds(
        measurement,
        document,
        settings,
        image_to_output,
        count_number=count_number,
        exact_area=exact_area,
    )


def _measurement_label_image_bounds(
    measurement: Measurement,
    document: ImageDocument,
    settings: AppSettings,
    image_to_output,
    *,
    count_number: int | None = None,
    exact_area: bool = False,
) -> QRectF | None:
    origin = image_to_output(Point(0.0, 0.0))
    unit_x = image_to_output(Point(1.0, 0.0))
    unit_y = image_to_output(Point(0.0, 1.0))
    output_scale = max(
        1e-9,
        math.hypot(unit_x.x() - origin.x(), unit_x.y() - origin.y()),
        math.hypot(unit_y.x() - origin.x(), unit_y.y() - origin.y()),
    )

    if measurement.measurement_kind == "count":
        if (
            not settings.show_count_numbers
            or measurement.point_px is None
            or count_number is None
        ):
            return None
        font = _count_number_font(settings, measurement)
        layout = _cached_text_layout(
            font,
            str(count_number),
            render_mode="count",
        )
        marker_radius = 4.0 * measurement_marker_scale(measurement)
        anchor_x = measurement.point_px.x + (marker_radius * 1.35 / output_scale)
        anchor_y = measurement.point_px.y - (marker_radius * 2.05 / output_scale)
        return QRectF(
            anchor_x - (6.0 / output_scale),
            anchor_y - (3.0 / output_scale),
            (layout.width + 12.0) / output_scale,
            (layout.height + 6.0) / output_scale,
        )

    if not _measurement_label_enabled(settings, measurement):
        return None
    font = measurement_label_font(settings, measurement)
    text = measurement_display_text_with_settings(measurement, document, settings)
    parallel = (
        measurement.measurement_kind == "line"
        and _measurement_label_parallel_to_line(settings, measurement)
    )
    render_mode = {
        "area": "measurement-area",
        "polyline": "measurement-polyline",
    }.get(
        measurement.measurement_kind,
        "measurement-length-parallel" if parallel else "measurement-length",
    )
    layout = _cached_text_layout(font, text, render_mode=render_mode)
    width = (layout.width + 12.0) / output_scale
    height = (layout.height + 6.0) / output_scale
    offset = max(12.0, layout.height * 0.75) / output_scale

    if measurement.measurement_kind == "line":
        line = measurement.effective_line()
        axis = direction(line)
        normal_axis = normal(axis)
        center_x = (line.start.x + line.end.x) / 2.0 + normal_axis[0] * offset
        center_y = (line.start.y + line.end.y) / 2.0 + normal_axis[1] * offset
        if parallel:
            radius = math.hypot(width, height) / 2.0
            return QRectF(
                center_x - radius,
                center_y - radius,
                radius * 2.0,
                radius * 2.0,
            )
        return QRectF(
            center_x - width / 2.0,
            center_y - height / 2.0,
            width,
            height,
        )

    if measurement.measurement_kind == "polyline" and measurement.polyline_px:
        first = measurement.polyline_px[0]
        last = measurement.polyline_px[-1]
        axis = _normalize(last.x - first.x, last.y - first.y)
        normal_axis = _normal(axis)
        center = measurement.geometry_center()
        return QRectF(
            center.x + normal_axis[0] * offset - width / 2.0,
            center.y + normal_axis[1] * offset - height / 2.0,
            width,
            height,
        )

    if measurement.measurement_kind == "area":
        if exact_area:
            center = area_derived_geometry_service.cached_centroid(measurement)
            if center is None:
                center = area_derived_geometry_service.centroid(measurement)
            label_center_y = center.y - (
                max(14.0, layout.height * 0.9) / output_scale
            )
            return QRectF(
                center.x - (width / 2.0),
                label_center_y - (height / 2.0),
                width,
                height,
            )
        bounds = area_derived_geometry_service.raw_bounds(measurement)
        if bounds is None:
            return None
        left, top, right, bottom = bounds
        # The exact odd-even centroid is guaranteed to lie inside this
        # conservative body envelope. The rendered center is shifted upward by
        # ``max(14, 0.9 * text_height)`` and the sprite extends another half
        # height above that center. Keep the same formula here; a fixed
        # ``height + 14`` margin clips 96/144 px object-level labels.
        label_offset = max(14.0, layout.height * 0.9) / output_scale
        return QRectF(
            left - width / 2.0,
            top - label_offset - (height / 2.0),
            max(0.0, right - left) + width,
            max(0.0, bottom - top) + height,
        )
    return None


def _measurement_bounds(measurement: Measurement) -> tuple[float, float, float, float] | None:
    if measurement.measurement_kind == "count" and measurement.point_px is not None:
        point = measurement.point_px
        return point.x, point.y, point.x, point.y
    if measurement.measurement_kind == "line" and measurement.line_px is not None:
        line = measurement.effective_line()
        return (
            min(line.start.x, line.end.x),
            min(line.start.y, line.end.y),
            max(line.start.x, line.end.x),
            max(line.start.y, line.end.y),
        )
    if measurement.measurement_kind == "polyline" and measurement.polyline_px:
        xs = [point.x for point in measurement.polyline_px]
        ys = [point.y for point in measurement.polyline_px]
        return min(xs), min(ys), max(xs), max(ys)
    if measurement.measurement_kind == "area":
        return area_derived_geometry_service.raw_bounds(measurement)
    return None


def _measurement_decoration_screen_padding(
    measurement: Measurement,
    settings: AppSettings,
    *,
    suggested_line_width: float,
    endpoint_radius: float,
    selected: bool,
) -> float:
    """Return a radial screen-space envelope for every rendered decoration."""

    marker_scale = measurement_marker_scale(measurement)
    base_width = measurement_line_width(measurement, suggested_line_width)
    selected_multiplier = (
        1.7
        if measurement.measurement_kind == "line"
        else 1.55
        if measurement.measurement_kind == "polyline"
        else 1.65
        if measurement.measurement_kind == "area"
        else 1.0
    )
    actual_width = base_width * (selected_multiplier if selected else 1.0)
    outline_width = (
        max(actual_width * 1.9, actual_width + 1.0)
        if measurement.measurement_kind == "area"
        else max(actual_width * 1.75, actual_width + 1.0)
        if measurement.measurement_kind == "polyline"
        else max(actual_width * 1.7, actual_width + 1.0)
    )
    padding = outline_width / 2.0

    if measurement.measurement_kind == "count":
        marker_radius = endpoint_radius * marker_scale
        return max(
            padding,
            marker_radius * (1.7 if selected else 1.35),
        )
    if measurement.measurement_kind != "line":
        if selected:
            padding = max(padding, endpoint_radius * marker_scale * 1.2)
        return padding

    resolved_endpoint_radius = (
        endpoint_radius
        * marker_scale
        * (1.15 if selected else 1.0)
    )
    endpoint_style = settings.measurement_endpoint_style
    if endpoint_style == MeasurementEndpointStyle.CIRCLE:
        endpoint_extent = resolved_endpoint_radius * 1.15
    elif endpoint_style == MeasurementEndpointStyle.BAR:
        endpoint_extent = (
            max(6.0, actual_width * 3.4)
            + (actual_width * 1.8 / 2.0)
        )
    elif endpoint_style in {
        MeasurementEndpointStyle.ARROW_INSIDE,
        MeasurementEndpointStyle.ARROW_OUTSIDE,
    }:
        arrow_length = max(8.0, actual_width * 4.0)
        arrow_half_width = max(4.0, actual_width * 2.4)
        endpoint_extent = (
            math.hypot(arrow_length, arrow_half_width)
            + (actual_width * 1.8 / 2.0)
        )
    else:
        endpoint_extent = 0.0
    return max(padding, endpoint_extent)


def measurement_display_image_bounds(
    measurement: Measurement,
    document: ImageDocument,
    settings: AppSettings,
    image_to_output,
    *,
    suggested_line_width: float = 2.0,
    endpoint_radius: float = 4.0,
    count_number: int | None = None,
    selected: bool = False,
    minimum_image_padding: float = 0.0,
    exact_area_label: bool = False,
) -> QRectF | None:
    """Return a conservative image-space envelope for every drawn pixel.

    The envelope includes RAW geometry, object-level stroke/marker overrides,
    BAR/ARROW endpoints and the formatted result label. It is shared by scene
    indexing, dirty-region updates and second-stage culling so those paths
    cannot disagree about what is visible.
    """

    bounds = _measurement_bounds(measurement)
    display_rect: QRectF | None = None
    if bounds is not None:
        left, top, right, bottom = bounds
        output_scale = _image_to_output_scale(
            _image_to_output_transform(image_to_output)
        )
        decoration_padding = (
            _measurement_decoration_screen_padding(
                measurement,
                settings,
                suggested_line_width=suggested_line_width,
                endpoint_radius=endpoint_radius,
                selected=selected,
            )
            / output_scale
        )
        decoration_padding = max(
            float(minimum_image_padding),
            decoration_padding,
        )
        display_rect = QRectF(
            left,
            top,
            max(1e-9, right - left),
            max(1e-9, bottom - top),
        ).adjusted(
            -decoration_padding,
            -decoration_padding,
            decoration_padding,
            decoration_padding,
        )

    label_rect = _measurement_label_image_bounds(
        measurement,
        document,
        settings,
        image_to_output,
        count_number=count_number,
        exact_area=exact_area_label,
    )
    if label_rect is not None and label_rect.isValid():
        display_rect = (
            QRectF(label_rect)
            if display_rect is None
            else display_rect.united(label_rect)
        )
    return display_rect


def draw_polyline_measurement(
    painter: QPainter,
    document: ImageDocument,
    measurement: Measurement,
    image_to_output,
    settings: AppSettings,
    *,
    line_width: float,
    endpoint_radius: float,
    selected: bool,
    use_sprite_cache: bool | None = None,
    sprite_device_pixel_ratio: float | None = None,
) -> None:
    if len(measurement.polyline_px) < 2:
        return
    path_points = [image_to_output(point) for point in measurement.polyline_px]
    color = measurement_color(document, measurement, settings)
    base_line_width = measurement_line_width(measurement, line_width)
    actual_width = base_line_width * (1.55 if selected else 1.0)
    outline_width = max(actual_width * 1.75, actual_width + 1.0)
    painter.setPen(
        QPen(
            QColor("#0B0B0B"),
            outline_width,
            Qt.PenStyle.SolidLine,
            Qt.PenCapStyle.RoundCap,
            Qt.PenJoinStyle.RoundJoin,
        )
    )
    painter.drawPolyline(QPolygonF(path_points))
    painter.setPen(
        QPen(
            color,
            actual_width,
            Qt.PenStyle.SolidLine,
            Qt.PenCapStyle.RoundCap,
            Qt.PenJoinStyle.RoundJoin,
        )
    )
    painter.drawPolyline(QPolygonF(path_points))
    if selected:
        painter.setBrush(QColor("#FFFFFF"))
        painter.setPen(QPen(QColor("#0B0B0B"), 1.2))
        resolved_endpoint_radius = endpoint_radius * measurement_marker_scale(measurement)
        for point in path_points:
            painter.drawEllipse(point, resolved_endpoint_radius * 0.72, resolved_endpoint_radius * 0.72)
    if _measurement_label_enabled(settings, measurement):
        if use_sprite_cache is None:
            draw_polyline_measurement_label(
                painter,
                measurement,
                document,
                settings,
                path_points,
                image_to_output,
            )
        else:
            label_kwargs = {"use_sprite_cache": use_sprite_cache}
            if sprite_device_pixel_ratio is not None:
                label_kwargs["sprite_device_pixel_ratio"] = (
                    sprite_device_pixel_ratio
                )
            draw_polyline_measurement_label(
                painter,
                measurement,
                document,
                settings,
                path_points,
                image_to_output,
                **label_kwargs,
            )


def draw_count_measurement(
    painter: QPainter,
    document: ImageDocument,
    measurement: Measurement,
    image_to_output,
    settings: AppSettings,
    *,
    endpoint_radius: float,
    selected: bool,
    count_number: int | None = None,
    use_sprite_cache: bool | None = None,
    sprite_device_pixel_ratio: float | None = None,
) -> None:
    if measurement.point_px is None:
        return
    point = image_to_output(measurement.point_px)
    color = measurement_color(document, measurement, settings)
    resolved_endpoint_radius = endpoint_radius * measurement_marker_scale(measurement)
    outline_radius = resolved_endpoint_radius * (1.7 if selected else 1.35)
    inner_radius = resolved_endpoint_radius * (1.1 if selected else 0.9)
    painter.setBrush(QColor("#0B0B0B"))
    painter.setPen(Qt.PenStyle.NoPen)
    painter.drawEllipse(point, outline_radius, outline_radius)
    if selected:
        painter.setBrush(QColor("#FFFFFF"))
        painter.drawEllipse(point, inner_radius * 1.02, inner_radius * 1.02)
        painter.setBrush(color)
        painter.drawEllipse(point, inner_radius * 0.72, inner_radius * 0.72)
        _draw_count_number_label(
            painter,
            point,
            settings,
            endpoint_radius=resolved_endpoint_radius,
            count_number=count_number,
            measurement=measurement,
            use_sprite_cache=use_sprite_cache,
            sprite_device_pixel_ratio=sprite_device_pixel_ratio,
        )
        return
    painter.setBrush(color)
    painter.drawEllipse(point, inner_radius, inner_radius)
    _draw_count_number_label(
        painter,
        point,
        settings,
        endpoint_radius=resolved_endpoint_radius,
        count_number=count_number,
        measurement=measurement,
        use_sprite_cache=use_sprite_cache,
        sprite_device_pixel_ratio=sprite_device_pixel_ratio,
    )


def draw_count_measurements_batch(
    painter: QPainter,
    document: ImageDocument,
    measurements: list[tuple[Measurement, int]],
    image_to_output,
    settings: AppSettings,
    *,
    endpoint_radius: float,
    use_sprite_cache: bool | None = None,
    sprite_device_pixel_ratio: float | None = None,
) -> None:
    if not measurements:
        return
    grouped_points: dict[tuple[str, float], list[QPointF]] = {}
    label_groups: dict[
        tuple[str, int, str, float],
        tuple[Measurement, list[tuple[QPointF, int]]],
    ] = {}
    for measurement, count_number in measurements:
        if measurement.point_px is None:
            continue
        point = image_to_output(measurement.point_px)
        color = measurement_color(document, measurement, settings).name()
        marker_scale = measurement_marker_scale(measurement)
        grouped_points.setdefault((color, marker_scale), []).append(point)
        if settings.show_count_numbers:
            label_font = _count_number_font(settings, measurement)
            label_color = measurement_text_color(measurement, settings.count_number_color).name()
            label_key = (
                label_font.family(),
                label_font.pointSize(),
                label_color,
                marker_scale,
            )
            label_group = label_groups.setdefault(label_key, (measurement, []))
            label_group[1].append((point, count_number))
    if not grouped_points:
        return
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor("#0B0B0B"))
    for (_color_name, marker_scale), points in grouped_points.items():
        outline_radius = endpoint_radius * marker_scale * 1.35
        for point in points:
            painter.drawEllipse(point, outline_radius, outline_radius)
    for (color_name, marker_scale), points in grouped_points.items():
        inner_radius = endpoint_radius * marker_scale * 0.9
        painter.setBrush(QColor(color_name))
        for point in points:
            painter.drawEllipse(point, inner_radius, inner_radius)
    for representative, label_points in label_groups.values():
        _draw_count_number_labels(
            painter,
            label_points,
            settings,
            endpoint_radius=endpoint_radius * measurement_marker_scale(representative),
            measurement=representative,
            use_sprite_cache=use_sprite_cache,
            sprite_device_pixel_ratio=sprite_device_pixel_ratio,
        )


def _count_label_static_text(text: str, font: QFont) -> QStaticText:
    return _cached_text_layout(font, text, render_mode="count").lines[0].static_text


def _count_number_font(settings: AppSettings, measurement: Measurement | None = None) -> QFont:
    appearance = measurement.appearance if measurement is not None else None
    font = QFont(
        appearance.font_family
        if appearance is not None and appearance.font_family
        else settings.count_number_font_family
    )
    font.setPointSize(
        max(
            8,
            int(
                appearance.font_size
                if appearance is not None and appearance.font_size is not None
                else settings.count_number_font_size
            ),
        )
    )
    return font


def _draw_count_number_labels(
    painter: QPainter,
    label_points: list[tuple[QPointF, int]],
    settings: AppSettings,
    *,
    endpoint_radius: float,
    measurement: Measurement | None = None,
    use_sprite_cache: bool | None = None,
    sprite_device_pixel_ratio: float | None = None,
) -> None:
    if not settings.show_count_numbers or not label_points:
        return
    font = _count_number_font(settings, measurement)
    painter.save()
    painter.setFont(font)
    color = (
        measurement_text_color(measurement, settings.count_number_color)
        if measurement
        else QColor(settings.count_number_color)
    )
    outline = _overlay_outline_color(color)
    offset_x = endpoint_radius * 1.35
    offset_y = -endpoint_radius * 2.05
    for point, count_number in label_points:
        text = str(count_number)
        anchor = QPointF(point.x() + offset_x, point.y() + offset_y)
        sprite = _screen_measurement_label_sprite(
            painter,
            text=text,
            font=font,
            text_color=color,
            outline_color=outline,
            background_color=None,
            arrangement_mode="count",
            use_sprite_cache=use_sprite_cache,
            sprite_device_pixel_ratio=sprite_device_pixel_ratio,
        )
        if sprite is not None:
            horizontal_padding = (
                sprite.logical_width - sprite.content_width
            ) / 2.0
            vertical_padding = (
                sprite.logical_height - sprite.content_height
            ) / 2.0
            painter.drawImage(
                QPointF(
                    anchor.x() - horizontal_padding,
                    anchor.y() - vertical_padding,
                ),
                sprite.image,
            )
            continue
        static = _count_label_static_text(text, font)
        painter.setPen(outline)
        for dx, dy in ((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)):
            painter.drawStaticText(QPointF(anchor.x() + dx, anchor.y() + dy), static)
        painter.setPen(color)
        painter.drawStaticText(anchor, static)
    painter.restore()


def _draw_count_number_label(
    painter: QPainter,
    point: QPointF,
    settings: AppSettings,
    *,
    endpoint_radius: float,
    count_number: int | None,
    measurement: Measurement | None = None,
    use_sprite_cache: bool | None = None,
    sprite_device_pixel_ratio: float | None = None,
) -> None:
    if count_number is None:
        return
    _draw_count_number_labels(
        painter,
        [(point, count_number)],
        settings,
        endpoint_radius=endpoint_radius,
        measurement=measurement,
        use_sprite_cache=use_sprite_cache,
        sprite_device_pixel_ratio=sprite_device_pixel_ratio,
    )


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
    geometry_mode: str = AREA_GEOMETRY_SCREEN,
    proxy_build_budget: AreaProxyBuildBudget | None = None,
    use_sprite_cache: bool | None = None,
    sprite_device_pixel_ratio: float | None = None,
    show_label: bool = True,
) -> None:
    geometry, output_transform, display_path = _area_geometry_and_output_path(
        measurement,
        image_to_output,
        selected=selected,
        geometry_mode=geometry_mode,
        proxy_build_budget=proxy_build_budget,
    )
    outline_points = geometry.outline_points
    fill_rings = geometry.fill_rings
    if len(outline_points) < 3:
        return
    color = measurement_color(document, measurement, settings)
    base_line_width = measurement_line_width(measurement, line_width)
    minimum_width = (
        0.5
        if measurement.appearance is not None and measurement.appearance.stroke_width is not None
        else 1.8
    )
    outline_width = max(base_line_width * (1.65 if selected else 1.0), minimum_width)
    transform_capable = all(
        hasattr(painter, attribute)
        for attribute in ("save", "restore", "setWorldTransform")
    )
    path_to_draw = (
        display_path
        if transform_capable
        else output_transform.map(display_path)
    )
    if transform_capable:
        painter.save()
    try:
        if transform_capable:
            painter.setWorldTransform(output_transform, combine=True)
        if show_fill:
            fill = QColor(color)
            fill.setAlpha(80 if not selected else 110)
            painter.setBrush(fill)
        else:
            painter.setBrush(Qt.BrushStyle.NoBrush)
        outline_pen = QPen(
            QColor("#0B0B0B"),
            max(outline_width * 1.9, outline_width + 1.0),
            Qt.PenStyle.SolidLine,
            Qt.PenCapStyle.RoundCap,
            Qt.PenJoinStyle.RoundJoin,
        )
        if transform_capable:
            outline_pen.setCosmetic(True)
        painter.setPen(outline_pen)
        # QPainter fills a path before stroking it.  Combining the fill and
        # outer outline therefore preserves the existing composition while
        # avoiding one full drawPath() traversal for every area object.
        painter.drawPath(path_to_draw)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        color_pen = QPen(
            color,
            outline_width,
            Qt.PenStyle.SolidLine,
            Qt.PenCapStyle.RoundCap,
            Qt.PenJoinStyle.RoundJoin,
        )
        if transform_capable:
            color_pen.setCosmetic(True)
        painter.setPen(color_pen)
        painter.drawPath(path_to_draw)
    finally:
        if transform_capable:
            painter.restore()
    if show_label and _measurement_label_enabled(settings, measurement):
        label_center = area_derived_geometry_service.cached_centroid(measurement)
        if label_center is None:
            # Labels are part of the exact cold-frame contract.  Proxy warming
            # may be deferred, but the RAW odd-even centroid is cached
            # independently and must be available before this frame is shown.
            label_center = area_derived_geometry_service.centroid(measurement)
        if use_sprite_cache is None:
            draw_area_measurement_label(
                painter,
                measurement,
                document,
                settings,
                image_to_output(label_center),
            )
        else:
            label_kwargs = {"use_sprite_cache": use_sprite_cache}
            if sprite_device_pixel_ratio is not None:
                label_kwargs["sprite_device_pixel_ratio"] = (
                    sprite_device_pixel_ratio
                )
            draw_area_measurement_label(
                painter,
                measurement,
                document,
                settings,
                image_to_output(label_center),
                **label_kwargs,
            )
    if not show_handles:
        return
    resolved_endpoint_radius = endpoint_radius * measurement_marker_scale(measurement)
    # Editing controls always expose the exact stored geometry, even when the
    # unselected screen outline uses a simplified proxy.
    raw_geometry = area_geometry_raw(measurement)
    handle_rings = raw_geometry.fill_rings or (
        [raw_geometry.outline_points]
        if len(raw_geometry.outline_points) >= 3
        else []
    )
    device = painter.device()
    device_pixel_ratio = (
        float(device.devicePixelRatioF())
        if device is not None and hasattr(device, "devicePixelRatioF")
        else 1.0
    )
    handle_coordinates = area_handle_display_cache.coordinates(
        measurement,
        handle_rings,
        output_scale=_image_to_output_scale(output_transform),
        device_pixel_ratio=device_pixel_ratio,
    )
    for x, y in handle_coordinates:
        _draw_circle_endpoint(
            painter,
            output_transform.map(QPointF(x, y)),
            color,
            resolved_endpoint_radius * 0.95,
        )
    center_point = image_to_output(measurement.polygon_center())
    painter.setBrush(QColor("#FFFFFF"))
    painter.setPen(QPen(QColor("#0B0B0B"), 1.6))
    painter.drawEllipse(center_point, resolved_endpoint_radius * 0.9, resolved_endpoint_radius * 0.9)
    painter.setPen(QPen(QColor("#0B0B0B"), 1.3))
    painter.drawLine(
        QPointF(center_point.x() - resolved_endpoint_radius * 0.45, center_point.y()),
        QPointF(center_point.x() + resolved_endpoint_radius * 0.45, center_point.y()),
    )
    painter.drawLine(
        QPointF(center_point.x(), center_point.y() - resolved_endpoint_radius * 0.45),
        QPointF(center_point.x(), center_point.y() + resolved_endpoint_radius * 0.45),
    )


def build_passive_area_overlay_command(
    document: ImageDocument,
    measurement: Measurement,
    settings: AppSettings,
    *,
    zoom: float,
    line_width: float,
    show_fill: bool,
    sprite_device_pixel_ratio: float,
) -> AreaOverlayDrawCommand | None:
    """Capture one unselected RAW area without recording its path to QPicture.

    Only Qt's implicitly-shared value objects leave the UI thread.  Persistent
    rings are neither copied nor mapped, and the worker never receives the
    mutable ``Measurement`` or ``ImageDocument`` objects.
    """

    if measurement.measurement_kind != "area":
        return None
    geometry = area_derived_geometry_service.raw_geometry(measurement)
    if len(geometry.outline_points) < 3:
        return None

    color = measurement_color(document, measurement, settings)
    base_line_width = measurement_line_width(measurement, line_width)
    minimum_width = (
        0.5
        if measurement.appearance is not None
        and measurement.appearance.stroke_width is not None
        else 1.8
    )
    stroke_width = max(base_line_width, minimum_width)
    fill_rgba: int | None = None
    if show_fill:
        fill = QColor(color)
        fill.setAlpha(80)
        fill_rgba = int(fill.rgba())

    label_command: AreaOverlayLabelCommand | None = None
    if _measurement_label_enabled(settings, measurement):
        label_center = area_derived_geometry_service.cached_centroid(measurement)
        font = measurement_label_font(settings, measurement)
        text = measurement_display_text_with_settings(
            measurement,
            document,
            settings,
        )
        text_color = measurement_text_color(
            measurement,
            _measurement_label_color(settings, measurement),
        )
        background_color = (
            QColor(16, 24, 32, 168)
            if _measurement_label_background_enabled(settings, measurement)
            else None
        )
        sprite = screen_label_sprite_cache.get_or_create(
            text=text,
            font=font,
            text_color=text_color,
            outline_color=None,
            background_color=background_color,
            device_pixel_ratio=sprite_device_pixel_ratio,
            arrangement_mode="measurement-area",
        )
        center_offset = QPointF(
            -(sprite.content_width / 2.0) - 6.0,
            -max(14.0, sprite.content_height * 0.9)
            - (sprite.content_height / 2.0)
            - 3.0,
        )
        top_left = (
            None
            if label_center is None
            else QPointF(
                (float(label_center.x) * float(zoom)) + center_offset.x(),
                (float(label_center.y) * float(zoom)) + center_offset.y(),
            )
        )
        label_command = AreaOverlayLabelCommand(
            image=QImage(sprite.image),
            top_left=top_left,
            center_offset=center_offset,
            centroid_key=(
                id(document),
                id(measurement),
                measurement.id,
                measurement.geometry_revision,
            ),
        )

    return AreaOverlayDrawCommand(
        path=QPainterPath(geometry.path),
        image_to_overlay=QTransform.fromScale(float(zoom), float(zoom)),
        fill_rgba=fill_rgba,
        outline_rgba=int(QColor("#0B0B0B").rgba()),
        outline_width=max(stroke_width * 1.9, stroke_width + 1.0),
        stroke_rgba=int(color.rgba()),
        stroke_width=stroke_width,
        label=label_command,
    )


def draw_area_measurement_label(
    painter: QPainter,
    measurement: Measurement,
    document: ImageDocument,
    settings: AppSettings,
    center: QPointF,
    *,
    use_sprite_cache: bool | None = None,
    sprite_device_pixel_ratio: float | None = None,
) -> None:
    font = measurement_label_font(settings, measurement)
    painter.setFont(font)
    text = measurement_display_text_with_settings(measurement, document, settings)
    text_color = measurement_text_color(measurement, _measurement_label_color(settings, measurement))
    background_color = (
        QColor(16, 24, 32, 168)
        if _measurement_label_background_enabled(settings, measurement)
        else None
    )
    sprite = _screen_measurement_label_sprite(
        painter,
        text=text,
        font=font,
        text_color=text_color,
        outline_color=None,
        background_color=background_color,
        arrangement_mode="measurement-area",
        use_sprite_cache=use_sprite_cache,
        sprite_device_pixel_ratio=sprite_device_pixel_ratio,
    )
    layout = (
        None
        if sprite is not None
        else _cached_text_layout(font, text, render_mode="measurement-area")
    )
    text_width = sprite.content_width if sprite is not None else layout.width
    text_height = sprite.content_height if sprite is not None else layout.height
    label_center = QPointF(center.x(), center.y() - max(14.0, text_height * 0.9))
    rect = QRectF(
        label_center.x() - (text_width / 2.0) - 6.0,
        label_center.y() - (text_height / 2.0) - 3.0,
        text_width + 12.0,
        text_height + 6.0,
    )
    if not _is_visible_to_painter(painter, rect, padding=4.0):
        return
    if sprite is not None:
        _draw_screen_label_sprite(painter, sprite, rect)
        return
    if background_color is not None:
        painter.fillRect(rect, background_color)
    _draw_cached_text(
        painter,
        layout,
        QPointF(rect.left() + 6.0, rect.top() + 3.0),
        color=text_color,
        outline=None,
        horizontal_center=rect.center().x(),
    )


def draw_measurement_label(
    painter: QPainter,
    measurement: Measurement,
    document: ImageDocument,
    settings: AppSettings,
    start_point: QPointF,
    end_point: QPointF,
    *,
    use_sprite_cache: bool | None = None,
    sprite_device_pixel_ratio: float | None = None,
) -> None:
    font = measurement_label_font(settings, measurement)
    painter.setFont(font)
    text = measurement_display_text_with_settings(measurement, document, settings)
    text_color = measurement_text_color(measurement, _measurement_label_color(settings, measurement))
    background_color = (
        QColor(16, 24, 32, 168)
        if _measurement_label_background_enabled(settings, measurement)
        else None
    )
    parallel_to_line = _measurement_label_parallel_to_line(settings, measurement)
    arrangement_mode = (
        "measurement-length-parallel"
        if parallel_to_line
        else "measurement-length"
    )
    sprite = _screen_measurement_label_sprite(
        painter,
        text=text,
        font=font,
        text_color=text_color,
        outline_color=None,
        background_color=background_color,
        arrangement_mode=arrangement_mode,
        use_sprite_cache=use_sprite_cache,
        sprite_device_pixel_ratio=sprite_device_pixel_ratio,
    )
    layout = (
        None
        if sprite is not None
        else _cached_text_layout(font, text, render_mode=arrangement_mode)
    )
    text_width = sprite.content_width if sprite is not None else layout.width
    text_height = sprite.content_height if sprite is not None else layout.height
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
    if parallel_to_line:
        angle = math.degrees(math.atan2(end_point.y() - start_point.y(), end_point.x() - start_point.x()))
        if angle > 90.0:
            angle -= 180.0
        elif angle < -90.0:
            angle += 180.0
        parallel_rect = QRectF(
            -(text_width / 2.0) - 6.0,
            -(text_height / 2.0) - 3.0,
            text_width + 12.0,
            text_height + 6.0,
        )
        radius = math.hypot(parallel_rect.width(), parallel_rect.height()) / 2.0
        visible_bounds = QRectF(center.x() - radius, center.y() - radius, radius * 2.0, radius * 2.0)
        if not _is_visible_to_painter(painter, visible_bounds, padding=4.0):
            return
        painter.save()
        painter.translate(center)
        painter.rotate(angle)
        if sprite is not None:
            _draw_screen_label_sprite(painter, sprite, parallel_rect)
        else:
            if background_color is not None:
                painter.fillRect(parallel_rect, background_color)
            _draw_cached_text(
                painter,
                layout,
                QPointF(parallel_rect.left() + 6.0, parallel_rect.top() + 3.0),
                color=text_color,
                outline=None,
                horizontal_center=0.0,
            )
        painter.restore()
        return
    if not _is_visible_to_painter(painter, rect, padding=4.0):
        return
    if sprite is not None:
        _draw_screen_label_sprite(painter, sprite, rect)
        return
    if background_color is not None:
        painter.fillRect(rect, background_color)
    _draw_cached_text(
        painter,
        layout,
        QPointF(rect.left() + 6.0, rect.top() + 3.0),
        color=text_color,
        outline=None,
        horizontal_center=rect.center().x(),
    )


def draw_polyline_measurement_label(
    painter: QPainter,
    measurement: Measurement,
    document: ImageDocument,
    settings: AppSettings,
    path_points: list[QPointF],
    image_to_output,
    *,
    use_sprite_cache: bool | None = None,
    sprite_device_pixel_ratio: float | None = None,
) -> None:
    if len(path_points) < 2:
        return
    font = measurement_label_font(settings, measurement)
    painter.setFont(font)
    text = measurement_display_text_with_settings(measurement, document, settings)
    text_color = measurement_text_color(measurement, _measurement_label_color(settings, measurement))
    background_color = (
        QColor(16, 24, 32, 168)
        if _measurement_label_background_enabled(settings, measurement)
        else None
    )
    sprite = _screen_measurement_label_sprite(
        painter,
        text=text,
        font=font,
        text_color=text_color,
        outline_color=None,
        background_color=background_color,
        arrangement_mode="measurement-polyline",
        use_sprite_cache=use_sprite_cache,
        sprite_device_pixel_ratio=sprite_device_pixel_ratio,
    )
    layout = (
        None
        if sprite is not None
        else _cached_text_layout(font, text, render_mode="measurement-polyline")
    )
    text_width = sprite.content_width if sprite is not None else layout.width
    text_height = sprite.content_height if sprite is not None else layout.height
    center_point = measurement.geometry_center()
    center = image_to_output(center_point)
    axis = _normalize(path_points[-1].x() - path_points[0].x(), path_points[-1].y() - path_points[0].y())
    normal_axis = _normal(axis)
    offset = max(12.0, text_height * 0.75)
    rect = QRectF(
        center.x() - (text_width / 2.0) - 6.0 + (normal_axis[0] * offset),
        center.y() - (text_height / 2.0) - 3.0 + (normal_axis[1] * offset),
        text_width + 12.0,
        text_height + 6.0,
    )
    if not _is_visible_to_painter(painter, rect, padding=4.0):
        return
    if sprite is not None:
        _draw_screen_label_sprite(painter, sprite, rect)
        return
    if background_color is not None:
        painter.fillRect(rect, background_color)
    _draw_cached_text(
        painter,
        layout,
        QPointF(rect.left() + 6.0, rect.top() + 3.0),
        color=text_color,
        outline=None,
        horizontal_center=rect.center().x(),
    )


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
    bar_px = max(0.0, end_point.x() - start_point.x())
    tick_width = min(max(1.0, fg_width), bar_px)
    if bar_px <= 0.0 or tick_width <= 0.0:
        return
    tick_height = max(1.0, tick_length * 2.0)
    tick_top = start_point.y() - tick_length
    left_tick = QRectF(start_point.x(), tick_top, tick_width, tick_height)
    right_tick = QRectF(end_point.x() - tick_width, tick_top, tick_width, tick_height)
    painter.fillRect(left_tick, foreground_color)
    painter.fillRect(right_tick, foreground_color)


def _draw_scale_segment(
    painter: QPainter,
    start_point: QPointF,
    *,
    bar_px: float,
    foreground_color: QColor,
    fg_width: float,
) -> None:
    if bar_px <= 0.0 or fg_width <= 0.0:
        return
    # Filled rectangles keep the visible bar inside the calibrated endpoints;
    # pen caps would otherwise extend the exported scale by part of the stroke.
    segment_rect = QRectF(start_point.x(), start_point.y() - (fg_width / 2.0), bar_px, fg_width)
    painter.fillRect(segment_rect, foreground_color)


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
    image_origin: Point | None = None,
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
        image_origin=image_origin,
    )
    end_point = QPointF(start_point.x() + bar_px, start_point.y())
    fg_width = scale_fg_width
    if settings.scale_overlay_style == ScaleOverlayStyle.BAR:
        fg_width = max(fg_width * 1.9, scale_fg_width + 1.5)
    _draw_scale_segment(
        painter,
        start_point,
        bar_px=bar_px,
        foreground_color=line_color,
        fg_width=fg_width,
    )
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
    text_padding_x = max(4.0, resolved_font_px * 0.25)
    text_width = max(bar_px, metrics.horizontalAdvance(text) + (text_padding_x * 2.0))
    text_center_x = start_point.x() + (bar_px / 2.0)
    text_rect = QRectF(text_center_x - (text_width / 2.0), text_top, text_width, metrics.height())
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
    image_origin: Point | None = None,
) -> tuple[QPointF, bool]:
    margin = max(24.0, min(image_width, image_height) * 0.04)
    placement = settings.scale_overlay_placement_mode
    if placement == ScaleOverlayPlacementMode.MANUAL and document.scale_overlay_anchor is not None:
        origin = image_origin or Point(0.0, 0.0)
        point = QPointF(
            (document.scale_overlay_anchor.x - origin.x) * image_to_output_scale,
            (document.scale_overlay_anchor.y - origin.y) * image_to_output_scale,
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
