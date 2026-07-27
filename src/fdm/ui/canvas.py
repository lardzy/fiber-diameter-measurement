from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
import math
import os
import time
from uuid import uuid4

import cv2
import numpy as np

from PySide6.QtCore import QEvent, QPointF, QRectF, QTimer, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QImage,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPalette,
    QPen,
    QPicture,
    QPolygonF,
    QTransform,
    QWheelEvent,
)
from PySide6.QtWidgets import QWidget

from fdm.area_display import (
    AREA_GEOMETRY_RAW,
    AREA_GEOMETRY_SCREEN,
    AreaProxyBuildBudget,
    area_derived_geometry_service,
    area_geometry_raw,
)
from fdm.geometry import (
    Line,
    Point,
    clamp,
    distance,
    nearest_endpoint,
    point_in_polygon,
    point_near_bounds,
    point_to_polyline_distance,
    point_to_polygon_edge_distance,
    polygon_translate,
)
from fdm.models import (
    ImageDocument,
    Measurement,
    OverlayAnnotation,
    OverlayAnnotationKind,
    OverlayTextSizeSpace,
)
from fdm.project_roi import (
    EllipseRoiGeometry,
    FreehandRoiGeometry,
    PolygonRoiGeometry,
    ProjectRoi,
    ProjectRoiKind,
    RectangleRoiGeometry,
    RoiBooleanExpression,
    RoiBooleanOperator,
    RoiPoint,
)
from fdm.services.prompt_segmentation import (
    finalize_magic_subtraction_mask,
    fill_magic_draft_internal_holes,
    magic_mask_area_px,
    magic_mask_to_geometry,
    magic_mask_to_polygon,
    normalize_magic_draft_mask,
)
from fdm.settings import (
    AppSettings,
    MagicSegmentToolMode,
    is_fiber_quick_tool_mode,
    is_magic_segment_tool_mode,
    is_magic_toolbar_tool_mode,
    is_reference_propagation_tool_mode,
)
from fdm.ui.canvas_tool_strategies import (
    ContinuousManualToolStrategy,
    CountToolStrategy,
    LineToolStrategy,
    clamp_point_to_image,
)
from fdm.ui.canvas_overlay_cache import (
    OVERLAY_TILE_LOGICAL_SIZE,
    OVERLAY_TILE_MAX_BYTES,
    OVERLAY_TILE_MAX_ENTRIES,
    CanvasOverlayRenderSnapshot,
    CanvasOverlayTileKey,
    canvas_overlay_tile_cache,
)
from fdm.ui.area_handle_cache import area_handle_display_cache
from fdm.ui.view_transform import (
    MAX_VIEW_ZOOM,
    MIN_VIEW_ZOOM,
    CanvasViewportSnapshot,
    CanvasZoomMode,
)
from fdm.ui.rendering import (
    area_rings_path,
    annotation_rect,
    build_passive_area_overlay_command,
    draw_area_measurement,
    draw_endpoint_style,
    draw_overlay_annotations,
    draw_measurements,
    draw_preview_scale_anchor,
    measurement_display_intersects_rect,
    measurement_display_image_bounds,
    measurement_geometry_cull_padding,
    overlay_annotation_bounds,
    overlay_annotation_handle_points,
)


class MagicSegmentOperationMode:
    ADD = "add"
    SUBTRACT = "subtract"


OVERLAY_CACHE_MIN_MEASUREMENTS = 64
OVERLAY_CACHE_MIN_AREA_VERTICES = 10_000


def _bounded_view_zoom(value: float) -> float:
    try:
        zoom = float(value)
    except (TypeError, ValueError):
        return 1.0
    if not math.isfinite(zoom):
        return 1.0
    return max(MIN_VIEW_ZOOM, min(MAX_VIEW_ZOOM, zoom))


def _normalized_capture_roi_kind(
    value: ProjectRoiKind | str,
) -> ProjectRoiKind | None:
    aliases = {
        "rectangle": ProjectRoiKind.RECTANGLE,
        "rect": ProjectRoiKind.RECTANGLE,
        "ellipse": ProjectRoiKind.ELLIPSE,
        "polygon": ProjectRoiKind.POLYGON,
        "freehand": ProjectRoiKind.FREEHAND,
        "free": ProjectRoiKind.FREEHAND,
    }
    if isinstance(value, ProjectRoiKind):
        normalized = value
    else:
        normalized = aliases.get(str(value).strip().lower())
    if normalized is ProjectRoiKind.COMPOSITE:
        return None
    return normalized


def _normalized_roi_capture_points(points: list[Point]) -> list[Point]:
    normalized: list[Point] = []
    for point in points:
        if normalized and distance(normalized[-1], point) <= 1e-6:
            continue
        normalized.append(Point(float(point.x), float(point.y)))
    if (
        len(normalized) >= 2
        and distance(normalized[0], normalized[-1]) <= 1e-6
    ):
        normalized.pop()
    if len({(point.x, point.y) for point in normalized}) < 3:
        return []
    return normalized


@dataclass(frozen=True, slots=True)
class CanvasSelectionRef:
    """The canvas' single, mutually-exclusive object selection."""

    kind: str
    object_id: str | None = None
    overlay_kind: str | None = None

    @classmethod
    def none(cls) -> "CanvasSelectionRef":
        return cls(kind="none")

    @classmethod
    def measurement(cls, measurement_id: str) -> "CanvasSelectionRef":
        return cls(kind="measurement", object_id=measurement_id)

    @classmethod
    def overlay(cls, overlay_id: str, overlay_kind: str) -> "CanvasSelectionRef":
        return cls(kind="overlay", object_id=overlay_id, overlay_kind=overlay_kind)


@dataclass(frozen=True, slots=True)
class RoiGeometryCommit:
    """One exact ROI geometry captured in original image-pixel coordinates."""

    request_id: str
    document_id: str
    kind: ProjectRoiKind
    geometry: (
        RectangleRoiGeometry
        | EllipseRoiGeometry
        | PolygonRoiGeometry
        | FreehandRoiGeometry
    )


@dataclass(slots=True)
class _RoiCaptureSession:
    request_id: str
    kind: ProjectRoiKind
    restore_tool_mode: str
    restore_overlay_kind: str
    points: list[Point] = field(default_factory=list)
    hover_point: Point | None = None
    drag_start: Point | None = None
    drag_end: Point | None = None
    dragging: bool = False


@dataclass(frozen=True, slots=True)
class CanvasPaintContext:
    """Immutable transform and clip information shared by one paint pass."""

    widget_rect: QRectF
    image_rect: QRectF
    image_to_widget_transform: QTransform
    widget_to_image_transform: QTransform
    zoom: float
    device_pixel_ratio: float


@dataclass(frozen=True, slots=True)
class CanvasDisplayBounds:
    """Screen-visible bounds used for conservative local invalidation."""

    image_rect: QRectF

    def expanded(self, image_padding: float) -> "CanvasDisplayBounds":
        padding = max(0.0, float(image_padding))
        return CanvasDisplayBounds(
            self.image_rect.adjusted(-padding, -padding, padding, padding)
        )


@dataclass(frozen=True, slots=True)
class CanvasVisualChange:
    """A local visual mutation; global changes explicitly request full paint."""

    object_ids: tuple[str, ...] = ()
    old_bounds: CanvasDisplayBounds | None = None
    new_bounds: CanvasDisplayBounds | None = None
    full_invalidation: bool = False


def canvas_workspace_background(palette: QPalette) -> QColor:
    """Return a neutral work-surface color with clear image separation."""

    if palette.color(QPalette.ColorRole.Window).lightnessF() < 0.5:
        return QColor("#101820")
    return QColor("#D6DEE7")


def canvas_workspace_foreground(palette: QPalette) -> QColor:
    if palette.color(QPalette.ColorRole.Window).lightnessF() < 0.5:
        return QColor("#F2F2F2")
    return QColor("#334155")


def canvas_image_border(palette: QPalette) -> QColor:
    if palette.color(QPalette.ColorRole.Window).lightnessF() < 0.5:
        return QColor("#425466")
    return QColor("#8796A5")


def _visual_color_signature(value: object) -> tuple[bool, int]:
    """Return the QColor result, not the spelling used to construct it."""

    color = QColor(str(value or ""))
    return color.isValid(), int(color.rgba())


def _measurement_label_style_signature(
    settings: AppSettings,
    attribute: str,
) -> tuple[object, ...]:
    style = getattr(settings, attribute, None)
    return (
        bool(
            getattr(
                style,
                "enabled",
                getattr(settings, "show_measurement_labels", True),
            )
        ),
        str(
            getattr(
                style,
                "font_family",
                getattr(settings, "measurement_label_font_family", ""),
            )
        ),
        max(
            8,
            int(
                getattr(
                    style,
                    "font_size",
                    getattr(settings, "measurement_label_font_size", 8),
                )
            ),
        ),
        _visual_color_signature(
            getattr(
                style,
                "color",
                getattr(settings, "measurement_label_color", ""),
            )
        ),
        int(
            getattr(
                style,
                "decimals",
                getattr(settings, "measurement_label_decimals", 0),
            )
        ),
        bool(
            getattr(
                style,
                "background_enabled",
                getattr(settings, "measurement_label_background_enabled", True),
            )
        ),
        bool(
            getattr(
                style,
                "parallel_to_line",
                getattr(settings, "measurement_label_parallel_to_line", False),
            )
        ),
    )


def _measurement_overlay_settings_signature(
    settings: AppSettings,
) -> tuple[object, ...]:
    """Visual inputs captured by passive measurement overlay tiles.

    Keep this separate from the complete ``AppSettings`` value: capture,
    export and window-layout preferences must not evict otherwise reusable
    canvas tiles.  Object-specific appearance and geometry are tracked by the
    document fingerprint path instead.
    """

    return (
        _measurement_label_style_signature(
            settings,
            "length_measurement_label_style",
        ),
        _measurement_label_style_signature(
            settings,
            "area_measurement_label_style",
        ),
        bool(settings.show_count_numbers),
        str(settings.count_number_font_family),
        max(8, int(settings.count_number_font_size)),
        _visual_color_signature(settings.count_number_color),
        str(settings.measurement_endpoint_style),
        _visual_color_signature(settings.default_measurement_color),
    )


def _canvas_visual_settings_signature(
    settings: AppSettings,
) -> tuple[object, ...]:
    """All global settings that can change pixels drawn by DocumentCanvas."""

    return (
        _measurement_overlay_settings_signature(settings),
        str(settings.text_font_family),
        max(8, int(settings.text_font_size)),
        _visual_color_signature(settings.text_color),
        _visual_color_signature(settings.overlay_line_color),
        max(0.5, float(settings.overlay_line_width)),
        bool(settings.magic_segment_fill_draft_holes_enabled),
    )


class AreaEditOperationMode:
    ADD = "add"
    SUBTRACT = "subtract"

    @classmethod
    def normalize(cls, value: str | None) -> str:
        normalized = str(value or "").strip()
        if normalized in {cls.ADD, cls.SUBTRACT}:
            return normalized
        return cls.ADD


class MagicSegmentSubtractInputMode:
    SMART = "smart"
    POLYGON = "polygon"
    FREEHAND = "freehand"

    @classmethod
    def normalize(cls, value: str | None) -> str:
        normalized = str(value or "").strip()
        if normalized in {cls.SMART, cls.POLYGON, cls.FREEHAND}:
            return normalized
        return cls.SMART


@dataclass(frozen=True, slots=True)
class MagicPromptVisual:
    prompt_type: str
    button_label: str
    prompt_label: str
    marker_color: str
    chip_background: tuple[int, int, int, int]
    chip_border: str
    chip_text: str


def magic_prompt_visual(prompt_type: str) -> MagicPromptVisual:
    if prompt_type == "negative":
        return MagicPromptVisual(
            prompt_type="negative",
            button_label="负采样",
            prompt_label="负采样点",
            marker_color="#EF4444",
            chip_background=(127, 29, 29, 220),
            chip_border="#F87171",
            chip_text="#FFFFFF",
        )
    return MagicPromptVisual(
        prompt_type="positive",
        button_label="正采样",
        prompt_label="正采样点",
        marker_color="#22C55E",
        chip_background=(6, 78, 59, 220),
        chip_border="#34D399",
        chip_text="#FFFFFF",
    )


@dataclass(frozen=True, slots=True)
class MeasurementIndexEntry:
    measurement: Measurement
    bounds: tuple[float, float, float, float]
    order: int
    count_number: int | None = None


class MeasurementSceneIndex:
    """A lightweight image-space index shared by painting and hit testing.

    Large area objects are deliberately kept in a separate bucket instead of
    being inserted into thousands of grid cells.  The final hit test remains
    exact and is performed by the caller against the original measurement
    geometry.
    """

    _MAX_ENTRY_CELLS = 256
    _MAX_QUERY_CELLS = 4096

    def __init__(
        self,
        measurements: list[Measurement],
        *,
        cell_size: float = 128.0,
        bounds_by_id: dict[
            str,
            tuple[float, float, float, float],
        ]
        | None = None,
    ) -> None:
        self._cell_size = max(32.0, float(cell_size))
        self._entries: dict[str, MeasurementIndexEntry] = {}
        self._cells: dict[tuple[int, int], list[str]] = {}
        self._oversized_ids: list[str] = []
        count_number = 0
        for order, measurement in enumerate(measurements):
            if measurement.measurement_kind == "count":
                count_number += 1
            bounds = (
                bounds_by_id.get(measurement.id)
                if bounds_by_id is not None
                else self._measurement_bounds(measurement)
            )
            if bounds is None:
                continue
            entry = MeasurementIndexEntry(
                measurement=measurement,
                bounds=bounds,
                order=order,
                count_number=(
                    count_number
                    if measurement.measurement_kind == "count"
                    else None
                ),
            )
            self._entries[measurement.id] = entry
            if self._cell_count_for_bounds(bounds) > self._MAX_ENTRY_CELLS:
                self._oversized_ids.append(measurement.id)
                continue
            for cell in self._iter_cells_for_bounds(bounds):
                self._cells.setdefault(cell, []).append(measurement.id)

    def query_point(self, point: Point, *, tolerance: float) -> list[Measurement]:
        query_bounds = (
            point.x - tolerance,
            point.y - tolerance,
            point.x + tolerance,
            point.y + tolerance,
        )
        entries = self._entries_for_bounds(query_bounds)
        entries = [
            entry
            for entry in entries
            if point_near_bounds(point, entry.bounds, tolerance)
        ]
        entries.sort(key=lambda entry: entry.order, reverse=True)
        return [entry.measurement for entry in entries]

    def query(self, point: Point, *, tolerance: float) -> list[Measurement]:
        """Compatibility wrapper for the original point-only index API."""

        return self.query_point(point, tolerance=tolerance)

    def query_rect(self, rect: QRectF) -> list[Measurement]:
        if not rect.isValid() and not rect.isNull():
            return []
        query_bounds = (
            float(rect.left()),
            float(rect.top()),
            float(rect.right()),
            float(rect.bottom()),
        )
        entries = self._entries_for_bounds(query_bounds)
        entries = [
            entry
            for entry in entries
            if self._bounds_intersect(entry.bounds, query_bounds)
        ]
        entries.sort(key=lambda entry: entry.order)
        return [entry.measurement for entry in entries]

    def count_number(self, measurement_id: str) -> int | None:
        entry = self._entries.get(measurement_id)
        return entry.count_number if entry is not None else None

    def document_order(self, measurement_id: str) -> int | None:
        entry = self._entries.get(measurement_id)
        return entry.order if entry is not None else None

    def _entries_for_bounds(
        self,
        bounds: tuple[float, float, float, float],
    ) -> list[MeasurementIndexEntry]:
        if self._cell_count_for_bounds(bounds) > self._MAX_QUERY_CELLS:
            return list(self._entries.values())
        candidate_ids: set[str] = set(self._oversized_ids)
        for cell in self._iter_cells_for_bounds(bounds):
            candidate_ids.update(self._cells.get(cell, ()))
        return [
            entry
            for measurement_id in candidate_ids
            if (entry := self._entries.get(measurement_id)) is not None
        ]

    def _cell_count_for_bounds(self, bounds: tuple[float, float, float, float]) -> int:
        min_col, min_row, max_col, max_row = self._cell_extents(bounds)
        return max(0, max_col - min_col + 1) * max(0, max_row - min_row + 1)

    def _cell_extents(
        self,
        bounds: tuple[float, float, float, float],
    ) -> tuple[int, int, int, int]:
        left, top, right, bottom = bounds
        min_col = int(left // self._cell_size)
        max_col = int(right // self._cell_size)
        min_row = int(top // self._cell_size)
        max_row = int(bottom // self._cell_size)
        return min_col, min_row, max_col, max_row

    def _iter_cells_for_bounds(self, bounds: tuple[float, float, float, float]):
        min_col, min_row, max_col, max_row = self._cell_extents(bounds)
        for col in range(min_col, max_col + 1):
            for row in range(min_row, max_row + 1):
                yield col, row

    @staticmethod
    def _bounds_intersect(
        first: tuple[float, float, float, float],
        second: tuple[float, float, float, float],
    ) -> bool:
        return not (
            first[2] < second[0]
            or first[0] > second[2]
            or first[3] < second[1]
            or first[1] > second[3]
        )

    @staticmethod
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


# Kept for downstream imports while the shared painting/hit-test index migrates
# to the more descriptive name.
MeasurementSpatialIndex = MeasurementSceneIndex


@dataclass(slots=True)
class PromptSegmentationSession:
    active_stage: str = MagicSegmentOperationMode.ADD
    subtract_input_mode: str = MagicSegmentSubtractInputMode.SMART
    primary_prompt_type: str = "positive"
    subtract_prompt_type: str = "positive"
    primary_positive_points: list[Point] = field(default_factory=list)
    primary_negative_points: list[Point] = field(default_factory=list)
    subtract_positive_points: list[Point] = field(default_factory=list)
    subtract_negative_points: list[Point] = field(default_factory=list)
    primary_polygon: list[Point] = field(default_factory=list)
    subtract_polygon: list[Point] = field(default_factory=list)
    confirmed_subtract_polygons: list[list[Point]] = field(default_factory=list)
    primary_rings: list[list[Point]] = field(default_factory=list)
    subtract_rings: list[list[Point]] = field(default_factory=list)
    confirmed_subtract_rings: list[list[list[Point]]] = field(default_factory=list)
    primary_mask: object | None = None
    subtract_mask: object | None = None
    confirmed_subtract_masks: list[object] = field(default_factory=list)
    primary_debug_payload: dict[str, object] = field(default_factory=dict)
    subtract_debug_payload: dict[str, object] = field(default_factory=dict)
    small_object_workspace_box: tuple[int, int, int, int] | None = None
    request_id: int = 0
    inflight_request_id: int = 0
    pending_stage: str = MagicSegmentOperationMode.ADD
    pending_recompute: bool = False
    busy: bool = False

    def prompt_type_for_stage(self, stage: str) -> str:
        if stage == MagicSegmentOperationMode.SUBTRACT:
            return self.subtract_prompt_type
        return self.primary_prompt_type

    def set_prompt_type_for_stage(self, stage: str, prompt_type: str) -> None:
        normalized = "negative" if prompt_type == "negative" else "positive"
        if stage == MagicSegmentOperationMode.SUBTRACT:
            self.subtract_prompt_type = normalized
        else:
            self.primary_prompt_type = normalized

    def positive_points_for_stage(self, stage: str) -> list[Point]:
        if stage == MagicSegmentOperationMode.SUBTRACT:
            return self.subtract_positive_points
        return self.primary_positive_points

    def negative_points_for_stage(self, stage: str) -> list[Point]:
        if stage == MagicSegmentOperationMode.SUBTRACT:
            return self.subtract_negative_points
        return self.primary_negative_points

    def mask_for_stage(self, stage: str):
        if stage == MagicSegmentOperationMode.SUBTRACT:
            return self.subtract_mask
        return self.primary_mask

    def polygon_for_stage(self, stage: str) -> list[Point]:
        if stage == MagicSegmentOperationMode.SUBTRACT:
            return self.subtract_polygon
        return self.primary_polygon

    def rings_for_stage(self, stage: str) -> list[list[Point]]:
        if stage == MagicSegmentOperationMode.SUBTRACT:
            return self.subtract_rings
        return self.primary_rings

    def set_mask_for_stage(self, stage: str, mask) -> None:
        if stage == MagicSegmentOperationMode.SUBTRACT:
            self.subtract_mask = mask
        else:
            self.primary_mask = mask

    def set_polygon_for_stage(self, stage: str, polygon: list[Point]) -> None:
        if stage == MagicSegmentOperationMode.SUBTRACT:
            self.subtract_polygon = polygon
        else:
            self.primary_polygon = polygon

    def set_rings_for_stage(self, stage: str, rings: list[list[Point]]) -> None:
        if stage == MagicSegmentOperationMode.SUBTRACT:
            self.subtract_rings = rings
        else:
            self.primary_rings = rings

    def debug_payload_for_stage(self, stage: str) -> dict[str, object]:
        if stage == MagicSegmentOperationMode.SUBTRACT:
            return self.subtract_debug_payload
        return self.primary_debug_payload

    def set_debug_payload_for_stage(self, stage: str, payload: dict[str, object]) -> None:
        if stage == MagicSegmentOperationMode.SUBTRACT:
            self.subtract_debug_payload = dict(payload)
        else:
            self.primary_debug_payload = dict(payload)

    def has_points(self) -> bool:
        return bool(
            self.primary_positive_points
            or self.primary_negative_points
            or self.subtract_positive_points
            or self.subtract_negative_points
        )

    def has_primary_preview(self) -> bool:
        return len(self.primary_polygon) >= 3 or (bool(self.primary_rings) and len(self.primary_rings[0]) >= 3)

    def has_any_preview(self) -> bool:
        return (
            self.has_primary_preview()
            or len(self.subtract_polygon) >= 3
            or (bool(self.subtract_rings) and len(self.subtract_rings[0]) >= 3)
            or bool(self.confirmed_subtract_polygons)
            or bool(self.confirmed_subtract_rings)
        )

    def confirmed_subtract_count(self) -> int:
        return len(self.confirmed_subtract_masks)


@dataclass(slots=True)
class ReferenceInstancePreviewCandidate:
    polygon_px: list[Point] = field(default_factory=list)
    area_rings_px: list[list[Point]] = field(default_factory=list)
    confidence: float = 0.0


@dataclass(slots=True)
class ReferenceInstanceSession:
    drag_start: Point | None = None
    drag_end: Point | None = None
    dragging: bool = False
    busy: bool = False
    request_id: int = 0
    reference_polygon: list[Point] = field(default_factory=list)
    reference_rings: list[list[Point]] = field(default_factory=list)
    preview_candidates: list[ReferenceInstancePreviewCandidate] = field(default_factory=list)

    def has_reference_geometry(self) -> bool:
        return len(self.reference_polygon) >= 3 or bool(self.reference_rings)

    def has_preview(self) -> bool:
        return bool(self.preview_candidates)

    def has_session(self) -> bool:
        return self.dragging or self.busy or self.has_reference_geometry() or self.has_preview()


@dataclass(slots=True)
class FiberQuickDiameterSession:
    prompt_type: str = "positive"
    positive_points: list[Point] = field(default_factory=list)
    negative_points: list[Point] = field(default_factory=list)
    preview_line: Line | None = None
    preview_mask: object | None = None
    preview_polygon: list[Point] = field(default_factory=list)
    preview_rings: list[list[Point]] = field(default_factory=list)
    confidence: float = 0.0
    request_id: int = 0
    inflight_request_id: int = 0
    pending_recompute: bool = False
    commit_pending: bool = False
    segmentation_busy: bool = False
    geometry_busy: bool = False
    debug_payload: dict[str, object] = field(default_factory=dict)

    def has_points(self) -> bool:
        return bool(self.positive_points or self.negative_points)

    def has_shape_preview(self) -> bool:
        return bool(self.preview_polygon or self.preview_rings)

    def has_preview(self) -> bool:
        return self.preview_line is not None

    def has_session(self) -> bool:
        return self.has_points() or self.has_shape_preview() or self.has_preview() or self.segmentation_busy or self.geometry_busy


class DocumentCanvas(QWidget):
    lineCommitted = Signal(str, str, object)
    objectSelectionChanged = Signal(str, object)
    measurementSelected = Signal(str, object)
    measurementEdited = Signal(str, str, object)
    pathSessionChanged = Signal(str)
    overlayCreateRequested = Signal(str, object)
    overlaySelected = Signal(str, object)
    overlayEdited = Signal(str, str, object)
    textPlacementRequested = Signal(str, object)
    textSelected = Signal(str, object)
    textMoved = Signal(str, str, object)
    scaleAnchorPicked = Signal(str, object)
    magicSegmentRequested = Signal(str, object)
    magicSegmentSessionChanged = Signal(str)
    areaEditRejected = Signal(str, str)
    viewZoomChanged = Signal(float)
    viewTransformChanged = Signal(object)
    roiGeometryCommitted = Signal(object)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._document: ImageDocument | None = None
        self._image: QImage | None = None
        self._tool_mode = "select"
        self._overlay_tool_kind = OverlayAnnotationKind.TEXT
        self._zoom = 1.0
        self._zoom_mode = CanvasZoomMode.CUSTOM
        self._pan = Point(20.0, 20.0)
        self._last_view_transform_snapshot: CanvasViewportSnapshot | None = None

        self._drawing_anchor_raw: Point | None = None
        self._drawing_line: Line | None = None
        self._line_commit_on_second_click = False
        self._line_tool_strategy = LineToolStrategy()
        self._continuous_manual_tool_strategy = ContinuousManualToolStrategy()
        self._count_tool_strategy = CountToolStrategy()

        self._drawing_polygon_points: list[Point] = []
        self._area_hover_point: Point | None = None
        self._drawing_freehand_active = False
        self._freehand_last_sample_at = 0.0
        self._area_edit_operation_mode = AreaEditOperationMode.ADD

        self._dragging_handle: tuple[str, str] | None = None
        self._drag_preview_line: Line | None = None

        self._dragging_area_handle: tuple[str, str, int | None, int | None] | None = None
        self._drag_area_preview_points: list[Point] | None = None
        self._drag_area_origin_points: list[Point] | None = None
        self._drag_area_preview_rings: list[list[Point]] | None = None
        self._drag_area_origin_rings: list[list[Point]] | None = None
        self._drag_area_press_point: Point | None = None
        self._drag_area_preview_offset: Point | None = None

        self._drawing_overlay_start: Point | None = None
        self._drawing_overlay_end: Point | None = None
        self._dragging_overlay_id: str | None = None
        self._dragging_overlay_handle: tuple[str, str] | None = None
        self._drag_overlay_press_point: Point | None = None
        self._drag_overlay_origin: OverlayAnnotation | None = None
        self._drag_overlay_preview: OverlayAnnotation | None = None

        self._panning = False
        self._pan_button: Qt.MouseButton | None = None
        self._last_mouse_pos = QPointF()
        self._pan_drag_unsnapped: Point | None = None
        self._pan_drag_device_phase: tuple[float, float] | None = None
        self._pan_drag_device_pixel_ratio: float | None = None
        self._space_pressed = False
        self._temporary_grab_active = False

        self._settings = AppSettings()
        self._overlay_settings_signature = _measurement_overlay_settings_signature(
            self._settings
        )
        self._canvas_visual_settings_signature = (
            _canvas_visual_settings_signature(self._settings)
        )
        self._scale_anchor_pick_active = False
        self._scale_anchor_preview_point: Point | None = None
        self._roi_capture: _RoiCaptureSession | None = None
        self._project_rois: tuple[ProjectRoi, ...] = ()
        self._project_roi_lookup: dict[str, ProjectRoi] = {}
        self._project_roi_paths: tuple[tuple[ProjectRoi, QPainterPath], ...] = ()
        self._show_area_fill = True
        self._magic_segment = PromptSegmentationSession()
        self._reference_instance = ReferenceInstanceSession()
        self._fiber_quick = FiberQuickDiameterSession()
        self._fiber_quick_request_serial = 0
        self._read_only = False
        self._fit_alignment = "center"
        self._measurement_hit_index: MeasurementSceneIndex | None = None
        self._measurement_hit_index_revision = -1
        self._measurement_display_index: MeasurementSceneIndex | None = None
        self._measurement_display_index_signature: tuple[object, ...] | None = (
            None
        )
        self._proxy_warm_scheduled = False
        self._proxy_warm_active_key: tuple[object, ...] | None = None
        self._proxy_warm_blocked_key: tuple[object, ...] | None = None
        self._proxy_warm_cache_generation = -1
        self._proxy_warm_cursor = 0
        self._proxy_warm_pending: tuple[
            tuple[object, ...],
            QRectF,
            object,
        ] | None = None
        self._proxy_warm_timer = QTimer(self)
        self._proxy_warm_timer.setSingleShot(True)
        self._proxy_warm_timer.timeout.connect(self._run_scheduled_proxy_warm)
        self._overlay_style_generation = 0
        self._overlay_tile_epochs: dict[tuple[float, float, int, int], int] = {}
        self._overlay_known_namespaces: set[tuple[float, float]] = set()
        self._overlay_namespace_order: list[tuple[float, float]] = []
        self._overlay_visible_keys: set[CanvasOverlayTileKey] = set()
        self._overlay_strict_visible_keys: set[CanvasOverlayTileKey] = set()
        self._overlay_tile_queue: list[CanvasOverlayTileKey] = []
        self._overlay_tile_queued: set[CanvasOverlayTileKey] = set()
        self._overlay_tile_active: CanvasOverlayTileKey | None = None
        self._overlay_tile_build_scheduled = False
        self._overlay_tile_failed: set[CanvasOverlayTileKey] = set()
        self._overlay_tile_request_serial = 0
        self._overlay_document_stamp: tuple[object, ...] | None = None
        self._overlay_group_signature: tuple[object, ...] | None = None
        self._overlay_calibration_signature: tuple[object, ...] | None = None
        self._overlay_measurement_order_signature: tuple[str, ...] | None = None
        self._overlay_max_object_font_size = 0.0
        self._overlay_area_vertex_count = 0
        self._overlay_measurement_state: dict[
            str,
            tuple[tuple[object, ...], tuple[float, float, float, float] | None],
        ] = {}
        self._overlay_annotation_state: dict[
            str,
            tuple[tuple[object, ...], tuple[float, float, float, float] | None],
        ] = {}
        self._overlay_selected_measurement_id: str | None = None
        canvas_overlay_tile_cache.tileReady.connect(self._on_overlay_tile_ready)
        canvas_overlay_tile_cache.tileFailed.connect(self._on_overlay_tile_failed)

    def changeEvent(self, event: QEvent) -> None:
        super().changeEvent(event)
        if event.type() == QEvent.Type.DevicePixelRatioChange:
            self._publish_view_transform()
            return
        if event.type() not in {
            QEvent.Type.PaletteChange,
            QEvent.Type.ApplicationPaletteChange,
        }:
            return
        # Theme changes can alter palette-derived outlines and backgrounds even
        # when persisted measurement settings stay unchanged.
        if not hasattr(self, "_overlay_style_generation"):
            return
        self._overlay_style_generation += 1
        self._invalidate_all_overlay_tiles()
        self.update()

    @property
    def document_id(self) -> str | None:
        return self._document.id if self._document else None

    def begin_roi_capture(
        self,
        kind: ProjectRoiKind | str,
        *,
        request_id: str | None = None,
    ) -> bool:
        """Begin a temporary ROI capture without changing the document model.

        The existing measurement/overlay draft is cancelled so pointer events
        cannot be consumed by two tools at once.  The selected measurement tool
        itself is only suspended and is restored after commit or cancellation.
        """

        if self._document is None or self._image is None or self._read_only:
            return False
        normalized_kind = _normalized_capture_roi_kind(kind)
        if normalized_kind is None:
            return False
        if self._roi_capture is not None:
            self._clear_roi_capture(restore_tool=True)
        self._cancel_area_drawing()
        self._cancel_line_drawing()
        self._cancel_overlay_interaction()
        if self.has_magic_segment_session():
            self.clear_magic_segment_session()
        if self._reference_instance.has_session():
            self.clear_reference_instance_session()
        if self.has_fiber_quick_session():
            self.clear_fiber_quick_session()
        self._scale_anchor_pick_active = False
        self._scale_anchor_preview_point = None
        capture_request_id = str(request_id or "").strip() or uuid4().hex
        self._roi_capture = _RoiCaptureSession(
            request_id=capture_request_id,
            kind=normalized_kind,
            restore_tool_mode=self._tool_mode,
            restore_overlay_kind=self._overlay_tool_kind,
        )
        self._temporary_grab_active = False
        self._update_cursor()
        self.update()
        return True

    def cancel_roi_capture(self) -> bool:
        """Cancel the current ROI draft and restore the suspended tool."""

        if self._roi_capture is None:
            return False
        self._clear_roi_capture(restore_tool=True)
        return True

    def set_project_rois(
        self,
        rois: Iterable[ProjectRoi],
        lookup: Mapping[str, ProjectRoi] | None = None,
    ) -> None:
        """Set project ROI display state for this canvas.

        Only visible ROI objects belonging to the mounted document are drawn.
        The lookup may include invisible operand ROI objects used by a composite
        expression.  All paths are built from authoritative image coordinates;
        no screen simplification is used.
        """

        normalized_rois = tuple(
            roi for roi in rois if isinstance(roi, ProjectRoi)
        )
        normalized_lookup: dict[str, ProjectRoi] = {
            roi.id: roi for roi in normalized_rois
        }
        if lookup is not None:
            for roi_id, roi in lookup.items():
                if isinstance(roi, ProjectRoi):
                    normalized_lookup[str(roi_id)] = roi
        signature = tuple(
            (
                roi.id,
                roi.document_id,
                roi.visible,
                roi.locked,
                roi.color,
                roi.revision,
                id(roi.geometry),
            )
            for roi in normalized_rois
        )
        previous_signature = tuple(
            (
                roi.id,
                roi.document_id,
                roi.visible,
                roi.locked,
                roi.color,
                roi.revision,
                id(roi.geometry),
            )
            for roi in self._project_rois
        )
        lookup_signature = tuple(
            sorted(
                (
                    roi_id,
                    roi.document_id,
                    roi.revision,
                    id(roi.geometry),
                )
                for roi_id, roi in normalized_lookup.items()
            )
        )
        previous_lookup_signature = tuple(
            sorted(
                (
                    roi_id,
                    roi.document_id,
                    roi.revision,
                    id(roi.geometry),
                )
                for roi_id, roi in self._project_roi_lookup.items()
            )
        )
        if (
            signature == previous_signature
            and lookup_signature == previous_lookup_signature
        ):
            return
        self._project_rois = normalized_rois
        self._project_roi_lookup = normalized_lookup
        self._rebuild_project_roi_paths()
        self.update()

    def set_document(self, document: ImageDocument, image: QImage) -> None:
        self._clear_roi_capture(restore_tool=True)
        self._end_canvas_pan()
        canvas_overlay_tile_cache.protect(id(self), ())
        previous_document = self._document
        self._document = document
        self._image = image
        self._cancel_area_drawing()
        self._cancel_line_drawing()
        self._magic_segment = PromptSegmentationSession()
        self._reference_instance = ReferenceInstanceSession()
        self._fiber_quick = FiberQuickDiameterSession()
        self._fiber_quick_request_serial = 0
        self._measurement_hit_index = None
        self._measurement_hit_index_revision = -1
        self._measurement_display_index = None
        self._measurement_display_index_signature = None
        self._overlay_max_object_font_size = self._max_object_font_size(
            document.measurements
        )
        self._reset_proxy_warming()
        if previous_document is not None and previous_document is not document:
            canvas_overlay_tile_cache.invalidate_document(id(previous_document))
            area_derived_geometry_service.discard_document(
                previous_document.measurements
            )
            area_handle_display_cache.discard_document(
                previous_document.measurements
            )
        self._reset_overlay_tracking(invalidate_document=False)
        self._zoom = _bounded_view_zoom(document.view_state.zoom or 1.0)
        self._zoom_mode = CanvasZoomMode.CUSTOM
        self._pan = Point(document.view_state.pan.x, document.view_state.pan.y)
        self._last_view_transform_snapshot = None
        self._rebuild_project_roi_paths()
        self._publish_view_transform()
        self.update()

    def set_image(self, image: QImage) -> None:
        self._image = image
        if self._document is not None:
            self._document.image_size = (image.width(), image.height())
        self._publish_view_transform()
        self.update()

    def clear_document(self) -> None:
        self._clear_roi_capture(restore_tool=True)
        self._end_canvas_pan()
        canvas_overlay_tile_cache.protect(id(self), ())
        previous_document = self._document
        document_id = self.document_id
        document_token = id(self._document) if self._document is not None else None
        self._document = None
        self._image = None
        self._last_view_transform_snapshot = None
        self._cancel_line_drawing()
        self._cancel_area_drawing()
        self._dragging_handle = None
        self._drag_preview_line = None
        self._dragging_area_handle = None
        self._drag_area_preview_points = None
        self._drag_area_origin_points = None
        self._drag_area_preview_rings = None
        self._drag_area_origin_rings = None
        self._drag_area_press_point = None
        self._drag_area_preview_offset = None
        self._drawing_overlay_start = None
        self._drawing_overlay_end = None
        self._dragging_overlay_id = None
        self._dragging_overlay_handle = None
        self._drag_overlay_press_point = None
        self._drag_overlay_origin = None
        self._drag_overlay_preview = None
        self._magic_segment = PromptSegmentationSession()
        self._reference_instance = ReferenceInstanceSession()
        self._fiber_quick = FiberQuickDiameterSession()
        self._fiber_quick_request_serial = 0
        self._measurement_hit_index = None
        self._measurement_hit_index_revision = -1
        self._measurement_display_index = None
        self._measurement_display_index_signature = None
        self._overlay_max_object_font_size = 0.0
        self._project_rois = ()
        self._project_roi_lookup = {}
        self._project_roi_paths = ()
        self._reset_proxy_warming()
        if document_token is not None:
            canvas_overlay_tile_cache.invalidate_document(document_token)
        if previous_document is not None:
            area_derived_geometry_service.discard_document(
                previous_document.measurements
            )
            area_handle_display_cache.discard_document(
                previous_document.measurements
            )
        self._reset_overlay_tracking(invalidate_document=False)
        if document_id is not None:
            self.pathSessionChanged.emit(document_id)
        self.update()

    def set_read_only(self, read_only: bool) -> None:
        self._read_only = read_only
        if read_only:
            self._clear_roi_capture(restore_tool=True)
            self._cancel_area_drawing()
            self._cancel_line_drawing()
            self._dragging_handle = None
            self._drag_preview_line = None
            self._drawing_overlay_start = None
            self._drawing_overlay_end = None
            self._dragging_overlay_id = None
            self._dragging_overlay_handle = None
            self._drag_overlay_press_point = None
            self._drag_overlay_origin = None
            self._drag_overlay_preview = None
            self._scale_anchor_pick_active = False
        self._update_cursor()
        self.update()

    def set_fit_alignment(self, alignment: str) -> None:
        self._fit_alignment = "top_left" if alignment == "top_left" else "center"
        self.update()

    def set_tool_mode(self, mode: str, *, overlay_kind: str | None = None) -> None:
        if self._roi_capture is not None:
            self._clear_roi_capture(restore_tool=False)
        next_overlay_kind = (
            overlay_kind
            if overlay_kind
            in {
                OverlayAnnotationKind.TEXT,
                OverlayAnnotationKind.RECT,
                OverlayAnnotationKind.CIRCLE,
                OverlayAnnotationKind.LINE,
                OverlayAnnotationKind.ARROW,
            }
            else self._overlay_tool_kind
        )
        if mode == self._tool_mode and next_overlay_kind == self._overlay_tool_kind:
            return
        if mode != self._tool_mode:
            self._cancel_area_drawing()
            self._cancel_line_drawing()
            if mode in {"polygon_area", "freehand_area"}:
                self._area_edit_operation_mode = AreaEditOperationMode.ADD
            if is_magic_segment_tool_mode(self._tool_mode) or not is_magic_segment_tool_mode(mode):
                self.clear_magic_segment_session()
            if is_reference_propagation_tool_mode(self._tool_mode) or not is_reference_propagation_tool_mode(mode):
                self.clear_reference_instance_session()
            if is_fiber_quick_tool_mode(self._tool_mode) or not is_fiber_quick_tool_mode(mode):
                self.clear_fiber_quick_session()
            self._cancel_overlay_interaction()
        self._tool_mode = mode
        self._overlay_tool_kind = next_overlay_kind
        self._update_cursor()
        self.update()

    def set_settings(self, settings: AppSettings) -> None:
        overlay_signature = _measurement_overlay_settings_signature(settings)
        overlay_changed = overlay_signature != self._overlay_settings_signature
        canvas_visual_signature = _canvas_visual_settings_signature(settings)
        canvas_visual_changed = (
            canvas_visual_signature != self._canvas_visual_settings_signature
        )
        self._settings = settings
        self._overlay_settings_signature = overlay_signature
        self._canvas_visual_settings_signature = canvas_visual_signature
        if overlay_changed:
            self._overlay_style_generation += 1
            self._invalidate_all_overlay_tiles()
        if canvas_visual_changed:
            self._measurement_display_index = None
            self._measurement_display_index_signature = None
            self.update()

    def notify_document_visual_changed(self) -> None:
        """Refresh only the visual envelope changed by a model mutation.

        MainWindow calls this after applying a document command.  Repeated UI
        synchronization must not turn a single object edit into an unbounded
        QWidget update; global changes such as calibration or group palette
        changes remain explicit full invalidations.
        """

        full_invalidation, changed_bounds = self._sync_overlay_visual_state()
        if full_invalidation:
            self.update()
            return
        if not changed_bounds:
            return
        dirty_rect = QRectF(changed_bounds[0].image_rect)
        for bounds in changed_bounds[1:]:
            dirty_rect = dirty_rect.united(bounds.image_rect)
        self._apply_visual_change(
            CanvasVisualChange(
                old_bounds=CanvasDisplayBounds(dirty_rect),
            )
        )

    def set_show_area_fill(self, visible: bool) -> None:
        if self._show_area_fill == visible:
            return
        self._show_area_fill = visible
        self._overlay_style_generation += 1
        self._invalidate_all_overlay_tiles()
        self.update()

    def current_magic_segment_prompt_type(self) -> str:
        return self._magic_segment.prompt_type_for_stage(self._magic_segment.active_stage)

    def current_magic_segment_operation_mode(self) -> str:
        return self._magic_segment.active_stage

    def magic_segment_primary_bounds(self) -> tuple[int, int, int, int] | None:
        if self._image is None:
            return None
        points: list[Point] = []
        if self._magic_segment.primary_rings:
            for ring in self._magic_segment.primary_rings:
                points.extend(ring)
        else:
            points.extend(self._magic_segment.primary_polygon)
        if len(points) < 3:
            return None
        min_x = max(0, int(math.floor(min(point.x for point in points))))
        min_y = max(0, int(math.floor(min(point.y for point in points))))
        max_x = min(self._image.width(), int(math.ceil(max(point.x for point in points))) + 1)
        max_y = min(self._image.height(), int(math.ceil(max(point.y for point in points))) + 1)
        if max_x <= min_x or max_y <= min_y:
            return None
        return min_x, min_y, max_x, max_y

    def magic_segment_small_object_workspace_box(self) -> tuple[int, int, int, int] | None:
        return self._magic_segment.small_object_workspace_box

    @staticmethod
    def point_in_box(point: Point, box: tuple[int, int, int, int]) -> bool:
        x0, y0, x1, y1 = box
        return x0 <= point.x < x1 and y0 <= point.y < y1

    def has_magic_segment_session(self) -> bool:
        return bool(
            self._magic_segment.has_points()
            or self._magic_segment.has_any_preview()
            or self._magic_segment.busy
        )

    def has_magic_segment_preview(self) -> bool:
        return self._magic_segment.has_primary_preview()

    def is_magic_segment_busy(self) -> bool:
        return self._magic_segment.busy

    def current_magic_subtract_input_mode(self) -> str:
        return MagicSegmentSubtractInputMode.normalize(self._magic_segment.subtract_input_mode)

    def set_magic_subtract_input_mode(self, mode: str) -> bool:
        normalized = MagicSegmentSubtractInputMode.normalize(mode)
        if normalized == self._magic_segment.subtract_input_mode:
            return True
        self._cancel_area_drawing()
        self._magic_segment.subtract_input_mode = normalized
        self._emit_magic_segment_session_changed()
        return True

    def has_magic_manual_subtract_draft(self) -> bool:
        return bool(
            is_magic_segment_tool_mode(self._tool_mode)
            and self._magic_segment.active_stage == MagicSegmentOperationMode.SUBTRACT
            and self.current_magic_subtract_input_mode()
            in {MagicSegmentSubtractInputMode.POLYGON, MagicSegmentSubtractInputMode.FREEHAND}
            and (self._drawing_freehand_active or self._drawing_polygon_points)
        )

    def complete_magic_manual_subtract_draft(self) -> bool:
        if not self.has_magic_manual_subtract_draft() or self._drawing_freehand_active:
            return False
        if len(self._drawing_polygon_points) < 3:
            return False
        return self._complete_magic_manual_subtract_polygon(list(self._drawing_polygon_points))

    def cancel_magic_subtract_draft(self) -> bool:
        if self.has_magic_manual_subtract_draft():
            self._cancel_area_drawing()
            self._emit_magic_segment_session_changed()
            return True
        if (
            not is_magic_segment_tool_mode(self._tool_mode)
            or self._magic_segment.active_stage != MagicSegmentOperationMode.SUBTRACT
        ):
            return False
        has_subtract_draft = bool(
            self._magic_segment.subtract_positive_points
            or self._magic_segment.subtract_negative_points
            or self._magic_segment.subtract_polygon
            or self._magic_segment.subtract_rings
            or self._magic_segment.subtract_mask is not None
            or self._magic_segment.subtract_debug_payload
            or self._magic_segment.small_object_workspace_box is not None
        )
        if not has_subtract_draft:
            return False
        self._clear_current_magic_subtract_draft()
        self._emit_magic_segment_session_changed()
        self.update()
        return True

    def has_reference_instance_session(self) -> bool:
        return self._reference_instance.has_session()

    def has_reference_instance_preview(self) -> bool:
        return self._reference_instance.has_preview()

    def is_reference_instance_busy(self) -> bool:
        return self._reference_instance.busy

    def current_fiber_quick_prompt_type(self) -> str:
        return self._fiber_quick.prompt_type

    def has_fiber_quick_session(self) -> bool:
        return self._fiber_quick.has_session()

    def has_fiber_quick_preview(self) -> bool:
        return self._fiber_quick.has_preview()

    def has_fiber_quick_shape_preview(self) -> bool:
        return self._fiber_quick.has_shape_preview()

    def is_fiber_quick_busy(self) -> bool:
        return self._fiber_quick.segmentation_busy or self._fiber_quick.geometry_busy

    def current_area_edit_operation_mode(self) -> str:
        return AreaEditOperationMode.normalize(self._area_edit_operation_mode)

    def set_area_edit_operation_mode(self, mode: str) -> bool:
        normalized = AreaEditOperationMode.normalize(mode)
        if normalized == self._area_edit_operation_mode:
            return False
        self._area_edit_operation_mode = normalized
        self.update()
        return True

    def has_selected_area_measurement(self) -> bool:
        return self._selected_area_measurement() is not None

    def has_area_edit_draft(self) -> bool:
        return self._tool_mode in {"polygon_area", "freehand_area"} and bool(
            self._drawing_polygon_points or self._drawing_freehand_active
        )

    def has_pending_path_drawing(self) -> bool:
        return bool(self._drawing_polygon_points or self._drawing_freehand_active or self._drawing_line is not None)

    def can_commit_pending_path(self) -> bool:
        if self._drawing_freehand_active:
            return False
        if self._tool_mode == "polygon_area":
            return len(self._drawing_polygon_points) >= 3
        if self._tool_mode == "continuous_manual":
            return self._continuous_manual_tool_strategy.can_commit(self._drawing_polygon_points)
        if self._tool_mode in {"manual", "snap"} and self._drawing_line is not None:
            return self._line_tool_strategy.can_commit(self._drawing_line)
        return False

    def commit_pending_path(self) -> bool:
        if not self.can_commit_pending_path():
            return False
        if self._tool_mode == "polygon_area":
            if self._area_subtract_mode_active():
                return self._complete_area_subtract_polygon(list(self._drawing_polygon_points))
            self._complete_area_measurement("polygon_area", list(self._drawing_polygon_points))
            return True
        if self._tool_mode == "continuous_manual":
            self._complete_continuous_measurement(list(self._drawing_polygon_points))
            return True
        if (
            self._tool_mode in {"manual", "snap"}
            and self._document is not None
            and self._drawing_line is not None
        ):
            line = self._line_tool_strategy.commit_payload(self._drawing_line)
            self._cancel_line_drawing()
            if line is None:
                return False
            self.lineCommitted.emit(self._document.id, self._tool_mode, line)
            self.update()
            return True
        return False

    def cancel_pending_path(self) -> bool:
        if not self.has_pending_path_drawing():
            return False
        if self._drawing_line is not None:
            self._cancel_line_drawing()
        else:
            self._cancel_area_drawing()
        return True

    def _begin_magic_segment_request(self, stage: str) -> dict[str, object] | None:
        positive_points = list(self._magic_segment.positive_points_for_stage(stage))
        if not positive_points:
            return None
        self._magic_segment.request_id += 1
        self._magic_segment.inflight_request_id = self._magic_segment.request_id
        self._magic_segment.pending_stage = stage
        self._magic_segment.pending_recompute = False
        self._magic_segment.busy = True
        return {
            "request_id": self._magic_segment.request_id,
            "positive_points": positive_points,
            "negative_points": list(self._magic_segment.negative_points_for_stage(stage)),
            "tool_mode": self._tool_mode,
            "active_stage": stage,
            "small_object_workspace_box": self._magic_segment.small_object_workspace_box
            if stage == MagicSegmentOperationMode.SUBTRACT
            else None,
        }

    def dequeue_pending_magic_segment_request(self, completed_request_id: int) -> dict[str, object] | None:
        if completed_request_id != self._magic_segment.inflight_request_id:
            return None
        self._magic_segment.inflight_request_id = 0
        if not self._magic_segment.pending_recompute:
            return None
        return self._begin_magic_segment_request(self._magic_segment.pending_stage)

    def _begin_fiber_quick_request(self) -> dict[str, object] | None:
        positive_points = list(self._fiber_quick.positive_points)
        if not positive_points:
            return None
        self._fiber_quick_request_serial += 1
        self._fiber_quick.request_id = self._fiber_quick_request_serial
        self._fiber_quick.inflight_request_id = self._fiber_quick.request_id
        self._fiber_quick.pending_recompute = False
        self._fiber_quick.commit_pending = False
        self._fiber_quick.segmentation_busy = True
        self._fiber_quick.geometry_busy = False
        self._fiber_quick.preview_line = None
        return {
            "request_id": self._fiber_quick.request_id,
            "positive_points": positive_points,
            "negative_points": list(self._fiber_quick.negative_points),
            "tool_mode": self._tool_mode,
        }

    def dequeue_pending_fiber_quick_request(self, completed_request_id: int) -> dict[str, object] | None:
        if completed_request_id != self._fiber_quick.inflight_request_id:
            return None
        self._fiber_quick.inflight_request_id = 0
        if not self._fiber_quick.pending_recompute:
            return None
        return self._begin_fiber_quick_request()

    def cycle_fiber_quick_prompt_type(self) -> str:
        self._fiber_quick.prompt_type = "negative" if self._fiber_quick.prompt_type == "positive" else "positive"
        self.update()
        self._emit_magic_segment_session_changed()
        return self._fiber_quick.prompt_type

    def set_magic_segment_prompt_type(self, prompt_type: str) -> None:
        self._magic_segment.set_prompt_type_for_stage(self._magic_segment.active_stage, prompt_type)
        self.update()
        self._emit_magic_segment_session_changed()

    def cycle_magic_segment_prompt_type(self) -> str:
        prompt_type = self.current_magic_segment_prompt_type()
        self._magic_segment.set_prompt_type_for_stage(
            self._magic_segment.active_stage,
            "negative" if prompt_type == "positive" else "positive",
        )
        self.update()
        self._emit_magic_segment_session_changed()
        return self.current_magic_segment_prompt_type()

    def cycle_magic_segment_operation_mode(self) -> str:
        if self._magic_segment.active_stage == MagicSegmentOperationMode.ADD:
            if not self._magic_segment.has_primary_preview():
                return self._magic_segment.active_stage
            self._magic_segment.active_stage = MagicSegmentOperationMode.SUBTRACT
        else:
            self._magic_segment.active_stage = MagicSegmentOperationMode.ADD
            self._magic_segment.small_object_workspace_box = None
        self.update()
        self._emit_magic_segment_session_changed()
        return self._magic_segment.active_stage

    def can_confirm_current_magic_subtract_shape(self) -> bool:
        return bool(
            self._magic_segment.active_stage == MagicSegmentOperationMode.SUBTRACT
            and not self._magic_segment.busy
            and self._magic_segment.subtract_mask is not None
            and (
                len(self._magic_segment.subtract_polygon) >= 3
                or (bool(self._magic_segment.subtract_rings) and len(self._magic_segment.subtract_rings[0]) >= 3)
            )
        )

    def confirmed_magic_subtract_shape_count(self) -> int:
        return self._magic_segment.confirmed_subtract_count()

    def reject_magic_segment_subtract_points_outside_primary_bounds(self, request_id: int) -> int:
        if request_id != self._magic_segment.request_id:
            return 0
        bounds = self.magic_segment_primary_bounds()
        removed = 0
        if bounds is not None:
            before = len(self._magic_segment.subtract_positive_points)
            self._magic_segment.subtract_positive_points = [
                point
                for point in self._magic_segment.subtract_positive_points
                if self.point_in_box(point, bounds)
            ]
            removed = before - len(self._magic_segment.subtract_positive_points)
        self._magic_segment.busy = False
        self._magic_segment.inflight_request_id = 0
        self._magic_segment.pending_recompute = False
        self._magic_segment.small_object_workspace_box = None
        self.update()
        self._emit_magic_segment_session_changed()
        return removed

    def confirm_current_magic_subtract_shape(self) -> dict[str, object]:
        result = {
            "confirmed": False,
            "count": self._magic_segment.confirmed_subtract_count(),
        }
        if not self.can_confirm_current_magic_subtract_shape():
            return result
        self._magic_segment.confirmed_subtract_masks.append(self._clone_magic_mask(self._magic_segment.subtract_mask))
        self._magic_segment.confirmed_subtract_polygons.append(self._clone_magic_polygon(self._magic_segment.subtract_polygon))
        self._magic_segment.confirmed_subtract_rings.append(self._clone_magic_rings(self._magic_segment.subtract_rings))
        self._clear_current_magic_subtract_draft()
        self._magic_segment.pending_stage = MagicSegmentOperationMode.SUBTRACT
        self.update()
        self._emit_magic_segment_session_changed()
        result["confirmed"] = True
        result["count"] = self._magic_segment.confirmed_subtract_count()
        return result

    def _clear_current_magic_subtract_draft(self) -> None:
        self._magic_segment.subtract_positive_points.clear()
        self._magic_segment.subtract_negative_points.clear()
        self._magic_segment.subtract_prompt_type = "positive"
        self._magic_segment.subtract_polygon = []
        self._magic_segment.subtract_rings = []
        self._magic_segment.subtract_mask = None
        self._magic_segment.subtract_debug_payload = {}
        self._magic_segment.small_object_workspace_box = None

    def apply_magic_segment_result(
        self,
        request_id: int,
        mask,
        polygon_points: list[Point] | None = None,
        area_rings_points: list[list[Point]] | None = None,
        debug_payload: dict[str, object] | None = None,
    ) -> dict[str, object] | None:
        if request_id != self._magic_segment.request_id:
            return None
        self._magic_segment.busy = False
        stage = self._magic_segment.pending_stage
        debug_payload = dict(debug_payload or {})
        if stage == MagicSegmentOperationMode.SUBTRACT:
            self._update_magic_small_object_workspace(debug_payload)
            if mask is None and debug_payload.get("small_object_reject_reason"):
                self._magic_segment.set_debug_payload_for_stage(stage, debug_payload)
                self.update()
                self._emit_magic_segment_session_changed()
                return {
                    "stage": stage,
                    "has_preview": len(self._magic_segment.polygon_for_stage(stage)) >= 3
                    or bool(self._magic_segment.rings_for_stage(stage)),
                    "rejected": True,
                }
        draft_mask = normalize_magic_draft_mask(mask)
        draft_polygon = self._normalize_magic_polygon(polygon_points)
        draft_rings = self._normalize_magic_rings(area_rings_points)
        if draft_mask is not None and (len(draft_polygon) < 3 or not draft_rings):
            selected_mask, selected_rings, selected_polygon, _stats = magic_mask_to_geometry(draft_mask)
            draft_mask = selected_mask
            if selected_rings:
                draft_rings = self._clone_magic_rings(selected_rings)
            if len(selected_polygon) >= 3:
                draft_polygon = self._clone_magic_polygon(selected_polygon)
        if (
            draft_mask is not None
            and self._tool_mode == MagicSegmentToolMode.STANDARD
            and self._settings.magic_segment_fill_draft_holes_enabled
        ):
            filled_mask = fill_magic_draft_internal_holes(draft_mask)
            selected_mask, selected_rings, selected_polygon, _stats = magic_mask_to_geometry(filled_mask)
            draft_mask = selected_mask
            draft_rings = self._clone_magic_rings(selected_rings)
            draft_polygon = self._clone_magic_polygon(selected_polygon)
        if len(draft_polygon) < 3 and draft_rings:
            draft_polygon = self._clone_magic_polygon(draft_rings[0])
        if len(draft_polygon) < 3 and not draft_rings:
            draft_polygon = []
            draft_mask = None
        self._magic_segment.set_polygon_for_stage(stage, self._clone_magic_polygon(draft_polygon))
        self._magic_segment.set_rings_for_stage(stage, self._clone_magic_rings(draft_rings))
        self._magic_segment.set_mask_for_stage(stage, self._clone_magic_mask(draft_mask))
        self._magic_segment.set_debug_payload_for_stage(stage, debug_payload)
        self.update()
        self._emit_magic_segment_session_changed()
        return {
            "stage": stage,
            "has_preview": len(self._magic_segment.polygon_for_stage(stage)) >= 3 or bool(self._magic_segment.rings_for_stage(stage)),
        }

    def fail_magic_segment_result(self, request_id: int) -> None:
        if request_id != self._magic_segment.request_id:
            return
        self._magic_segment.busy = False
        self.update()
        self._emit_magic_segment_session_changed()

    def clear_magic_segment_session(self) -> None:
        self._magic_segment = PromptSegmentationSession()
        self.update()
        self._emit_magic_segment_session_changed()

    def _update_magic_small_object_workspace(self, debug_payload: dict[str, object]) -> None:
        if not bool(debug_payload.get("small_object_enhancement_used")):
            self._magic_segment.small_object_workspace_box = None
            return
        box = debug_payload.get("small_object_workspace_box")
        if not isinstance(box, (tuple, list)) or len(box) != 4:
            self._magic_segment.small_object_workspace_box = None
            return
        try:
            x0, y0, x1, y1 = [int(round(float(value))) for value in box]
        except (TypeError, ValueError):
            self._magic_segment.small_object_workspace_box = None
            return
        if x1 <= x0 or y1 <= y0:
            self._magic_segment.small_object_workspace_box = None
            return
        self._magic_segment.small_object_workspace_box = (x0, y0, x1, y1)

    def apply_reference_instance_result(
        self,
        request_id: int,
        *,
        reference_polygon_points: list[Point] | None = None,
        reference_area_rings_points: list[list[Point]] | None = None,
        candidates: list[ReferenceInstancePreviewCandidate] | None = None,
    ) -> dict[str, object] | None:
        if request_id != self._reference_instance.request_id:
            return None
        self._reference_instance.busy = False
        self._reference_instance.dragging = False
        self._reference_instance.drag_start = None
        self._reference_instance.drag_end = None
        self._reference_instance.reference_polygon = self._normalize_magic_polygon(reference_polygon_points)
        self._reference_instance.reference_rings = self._normalize_magic_rings(reference_area_rings_points)
        if len(self._reference_instance.reference_polygon) < 3 and self._reference_instance.reference_rings:
            self._reference_instance.reference_polygon = self._clone_magic_polygon(self._reference_instance.reference_rings[0])
        normalized_candidates: list[ReferenceInstancePreviewCandidate] = []
        for candidate in candidates or []:
            polygon = self._normalize_magic_polygon(candidate.polygon_px)
            rings = self._normalize_magic_rings(candidate.area_rings_px)
            if len(polygon) < 3 and rings:
                polygon = self._clone_magic_polygon(rings[0])
            if len(polygon) < 3 and not rings:
                continue
            normalized_candidates.append(
                ReferenceInstancePreviewCandidate(
                    polygon_px=polygon,
                    area_rings_px=rings,
                    confidence=float(candidate.confidence),
                )
            )
        self._reference_instance.preview_candidates = normalized_candidates
        self.update()
        self._emit_magic_segment_session_changed()
        return {
            "candidate_count": len(normalized_candidates),
            "has_reference": self._reference_instance.has_reference_geometry(),
        }

    def fail_reference_instance_result(self, request_id: int) -> None:
        if request_id != self._reference_instance.request_id:
            return
        self._reference_instance.busy = False
        self._reference_instance.dragging = False
        self._reference_instance.drag_start = None
        self._reference_instance.drag_end = None
        self.update()
        self._emit_magic_segment_session_changed()

    def clear_reference_instance_session(self) -> None:
        self._reference_instance = ReferenceInstanceSession()
        self.update()
        self._emit_magic_segment_session_changed()

    def apply_fiber_quick_segmentation_result(
        self,
        request_id: int,
        *,
        mask=None,
        preview_polygon_points: list[Point] | None = None,
        preview_area_rings_points: list[list[Point]] | None = None,
        debug_payload: dict[str, object] | None = None,
    ) -> dict[str, object] | None:
        if request_id != self._fiber_quick.request_id:
            return None
        self._fiber_quick.segmentation_busy = False
        self._fiber_quick.preview_mask = normalize_magic_draft_mask(mask)
        self._fiber_quick.preview_polygon = self._normalize_magic_polygon(preview_polygon_points)
        self._fiber_quick.preview_rings = self._normalize_magic_rings(preview_area_rings_points)
        self._fiber_quick.debug_payload = dict(debug_payload or {})
        self.update()
        self._emit_magic_segment_session_changed()
        return {
            "has_shape_preview": bool(self._fiber_quick.preview_polygon or self._fiber_quick.preview_rings),
        }

    def set_fiber_quick_pending_roi(self, request_id: int, crop_box: tuple[int, int, int, int] | None) -> bool:
        if request_id != self._fiber_quick.request_id:
            return False
        self._fiber_quick.debug_payload = {
            **dict(self._fiber_quick.debug_payload),
            "segmentation_crop_box": crop_box,
            "segmentation_pending": True,
        }
        self.update()
        self._emit_magic_segment_session_changed()
        return True

    def begin_fiber_quick_geometry(self, request_id: int) -> bool:
        if request_id != self._fiber_quick.request_id:
            return False
        self._fiber_quick.geometry_busy = True
        self.update()
        self._emit_magic_segment_session_changed()
        return True

    def apply_fiber_quick_geometry_result(
        self,
        request_id: int,
        *,
        preview_line: Line | None = None,
        confidence: float = 0.0,
        debug_payload: dict[str, object] | None = None,
    ) -> dict[str, object] | None:
        if request_id != self._fiber_quick.request_id:
            return None
        self._fiber_quick.geometry_busy = False
        self._fiber_quick.preview_line = preview_line
        self._fiber_quick.confidence = float(confidence)
        if debug_payload:
            merged_payload = dict(self._fiber_quick.debug_payload)
            merged_payload.update(debug_payload)
            merged_payload.pop("segmentation_pending", None)
            self._fiber_quick.debug_payload = merged_payload
        self.update()
        self._emit_magic_segment_session_changed()
        return {
            "has_preview": self._fiber_quick.has_preview(),
        }

    def fail_fiber_quick_result(self, request_id: int, *, stage: str = "all") -> None:
        if request_id != self._fiber_quick.request_id:
            return
        if stage in {"segmentation", "all"}:
            self._fiber_quick.segmentation_busy = False
            self._fiber_quick.commit_pending = False
            if stage == "all":
                self._fiber_quick.preview_mask = None
            if stage == "segmentation":
                self._fiber_quick.debug_payload = dict(self._fiber_quick.debug_payload)
        if stage in {"geometry", "all"}:
            self._fiber_quick.geometry_busy = False
            self._fiber_quick.preview_line = None
            if stage == "geometry":
                self._fiber_quick.commit_pending = False
        self.update()
        self._emit_magic_segment_session_changed()

    def clear_fiber_quick_session(self) -> None:
        self._fiber_quick = FiberQuickDiameterSession()
        self.update()
        self._emit_magic_segment_session_changed()

    def commit_fiber_quick_preview(self) -> dict[str, object]:
        document_id = self._document.id if self._document is not None else None
        preview_line = self._fiber_quick.preview_line
        if preview_line is None and self._fiber_quick.has_shape_preview():
            snapshot = {
                "measurement_kind": "line",
                "mode": "fiber_quick",
                "mask": self._clone_magic_mask(self._fiber_quick.preview_mask),
                "polygon_px": self._clone_magic_polygon(self._fiber_quick.preview_polygon),
                "area_rings_px": self._clone_magic_rings(self._fiber_quick.preview_rings),
                "positive_points": list(self._fiber_quick.positive_points),
                "negative_points": list(self._fiber_quick.negative_points),
                "debug_payload": dict(self._fiber_quick.debug_payload),
            }
            self.clear_fiber_quick_session()
            return {
                "committed": False,
                "pending": True,
                "snapshot": snapshot,
            }
        payload = {
            "measurement_kind": "line",
            "line_px": preview_line,
            "confidence": float(self._fiber_quick.confidence),
            "status": "fiber_quick",
            "debug_payload": dict(self._fiber_quick.debug_payload),
        }
        committed = document_id is not None and preview_line is not None
        self.clear_fiber_quick_session()
        if committed:
            self.lineCommitted.emit(document_id, "fiber_quick", payload)
        return {
            "committed": committed,
        }

    def commit_reference_instance_preview(self) -> dict[str, object]:
        candidates = [
            {
                "polygon_px": self._clone_magic_polygon(candidate.polygon_px),
                "area_rings_px": self._clone_magic_rings(candidate.area_rings_px),
                "confidence": float(candidate.confidence),
            }
            for candidate in self._reference_instance.preview_candidates
            if len(candidate.polygon_px) >= 3 or candidate.area_rings_px
        ]
        committed = bool(candidates)
        self.clear_reference_instance_session()
        return {
            "committed": committed,
            "candidates": candidates,
        }

    def commit_magic_segment_preview(self) -> dict[str, object]:
        document_id = self._document.id if self._document is not None else None
        primary_polygon = self._clone_magic_polygon(self._magic_segment.primary_polygon)
        primary_mask = normalize_magic_draft_mask(self._magic_segment.primary_mask)
        result: dict[str, object] = {
            "committed": False,
            "hole_count": 0,
            "discarded_fragments": False,
            "result_empty": False,
        }
        if document_id is None or len(primary_polygon) < 3 or primary_mask is None:
            result["reason"] = "missing_primary"
            return result
        polygon_points = primary_polygon
        area_rings_points = self._clone_magic_rings(self._magic_segment.primary_rings)
        final_mask = primary_mask.copy()
        subtract_masks = [self._clone_magic_mask(mask) for mask in self._magic_segment.confirmed_subtract_masks]
        current_subtract_mask = normalize_magic_draft_mask(self._magic_segment.subtract_mask)
        if current_subtract_mask is not None:
            subtract_masks.append(self._clone_magic_mask(current_subtract_mask))
        if subtract_masks:
            final_mask, stats = finalize_magic_subtraction_mask(primary_mask, subtract_masks)
            result.update(stats)
            if final_mask is None:
                self.clear_magic_segment_session()
                return result
        selected_mask, selected_rings, selected_polygon, geometry_stats = magic_mask_to_geometry(
            final_mask,
            select_prompt_component=False,
        )
        if selected_mask is None:
            self.clear_magic_segment_session()
            result["result_empty"] = True
            return result
        if selected_rings:
            area_rings_points = self._clone_magic_rings(selected_rings)
        if len(selected_polygon) >= 3:
            polygon_points = self._clone_magic_polygon(selected_polygon)
        result["hole_count"] = int(geometry_stats.get("hole_count", 0) or 0)
        self.clear_magic_segment_session()
        if document_id is None or len(polygon_points) < 3:
            result["reason"] = "missing_polygon"
            return result
        self.lineCommitted.emit(
            document_id,
            "magic_segment",
            {
                "measurement_kind": "area",
                "polygon_px": polygon_points,
                "area_rings_px": area_rings_points,
                "exact_area_px": magic_mask_area_px(selected_mask),
            },
        )
        result["committed"] = True
        return result

    def _clone_magic_mask(self, mask):
        if mask is None:
            return None
        return mask.copy()

    def _clone_magic_polygon(self, polygon_points: list[Point]) -> list[Point]:
        return [Point(float(point.x), float(point.y)) for point in polygon_points]

    def _clone_magic_rings(self, area_rings: list[list[Point]]) -> list[list[Point]]:
        return [self._clone_magic_polygon(ring) for ring in area_rings]

    def _normalize_magic_polygon(self, polygon_points: list[Point] | None) -> list[Point]:
        if not polygon_points:
            return []
        return self._clone_magic_polygon(list(polygon_points))

    def _normalize_magic_rings(self, area_rings: list[list[Point]] | None) -> list[list[Point]]:
        if not area_rings:
            return []
        normalized: list[list[Point]] = []
        for ring in area_rings:
            cloned = self._normalize_magic_polygon(ring)
            if len(cloned) >= 3:
                normalized.append(cloned)
        return normalized

    def _magic_polygon_to_mask(self, polygon_points: list[Point]):
        if self._image is None or len(polygon_points) < 3:
            return None
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover - dependency is required by the app
            raise RuntimeError("numpy is required for the magic segmentation tool.") from exc
        mask = np.zeros((self._image.height(), self._image.width()), dtype=np.uint8)
        contour = np.array(
            [
                [
                    int(clamp(round(point.x), 0, self._image.width() - 1)),
                    int(clamp(round(point.y), 0, self._image.height() - 1)),
                ]
                for point in polygon_points
            ],
            dtype=np.int32,
        )
        if contour.shape[0] < 3:
            return None
        cv2.fillPoly(mask, [contour], 1)
        if not mask.any():
            return None
        return mask.astype(bool)

    def _selected_area_measurement(self) -> Measurement | None:
        if self._document is None:
            return None
        measurement = self._document.get_measurement(self._document.view_state.selected_measurement_id)
        if (
            measurement is None
            or measurement.measurement_kind != "area"
            or (len(measurement.polygon_px) < 3 and not measurement.area_rings_px)
        ):
            return None
        return measurement

    def _area_subtract_mode_active(self) -> bool:
        return (
            self._tool_mode in {"polygon_area", "freehand_area"}
            and self.current_area_edit_operation_mode() == AreaEditOperationMode.SUBTRACT
            and self._selected_area_measurement() is not None
        )

    def _magic_manual_subtract_mode_active(self, mode: str | None = None) -> bool:
        if not is_magic_segment_tool_mode(self._tool_mode):
            return False
        if self._magic_segment.active_stage != MagicSegmentOperationMode.SUBTRACT:
            return False
        active_mode = self.current_magic_subtract_input_mode()
        if mode is not None:
            return active_mode == mode
        return active_mode in {MagicSegmentSubtractInputMode.POLYGON, MagicSegmentSubtractInputMode.FREEHAND}

    def _complete_magic_manual_subtract_polygon(self, polygon_points: list[Point]) -> bool:
        if len(polygon_points) < 3:
            return False
        mask = self._magic_polygon_to_mask(polygon_points)
        if mask is None:
            self._cancel_area_drawing()
            self._emit_magic_segment_session_changed()
            return False
        selected_mask, selected_rings, selected_polygon, stats = magic_mask_to_geometry(mask, select_prompt_component=False)
        if selected_mask is None or (len(selected_polygon) < 3 and not selected_rings):
            self._cancel_area_drawing()
            self._emit_magic_segment_session_changed()
            return False
        self._magic_segment.subtract_positive_points.clear()
        self._magic_segment.subtract_negative_points.clear()
        self._magic_segment.subtract_prompt_type = "positive"
        self._magic_segment.subtract_mask = self._clone_magic_mask(selected_mask)
        self._magic_segment.subtract_rings = self._clone_magic_rings(selected_rings)
        self._magic_segment.subtract_polygon = self._clone_magic_polygon(
            selected_polygon if len(selected_polygon) >= 3 else selected_rings[0]
        )
        self._magic_segment.subtract_debug_payload = {
            "manual_subtract_input_mode": self.current_magic_subtract_input_mode(),
            "manual_subtract_point_count": len(polygon_points),
            "manual_subtract_stats": stats,
        }
        self._magic_segment.small_object_workspace_box = None
        self._magic_segment.pending_stage = MagicSegmentOperationMode.SUBTRACT
        self._cancel_area_drawing()
        self._emit_magic_segment_session_changed()
        return True

    def _area_measurement_to_mask(self, measurement: Measurement):
        if self._image is None:
            return None
        mask = np.zeros((self._image.height(), self._image.width()), dtype=np.uint8)

        def contour(points: list[Point]):
            return np.array(
                [
                    [
                        int(clamp(round(point.x), 0, self._image.width() - 1)),
                        int(clamp(round(point.y), 0, self._image.height() - 1)),
                    ]
                    for point in points
                ],
                dtype=np.int32,
            )

        if measurement.area_rings_px:
            outer = measurement.area_rings_px[0]
            if len(outer) >= 3:
                cv2.fillPoly(mask, [contour(outer)], 1)
            for hole in measurement.area_rings_px[1:]:
                if len(hole) >= 3:
                    cv2.fillPoly(mask, [contour(hole)], 0)
        elif len(measurement.polygon_px) >= 3:
            cv2.fillPoly(mask, [contour(measurement.polygon_px)], 1)
        if not mask.any():
            return None
        return mask.astype(bool)

    def _area_mask_component_count(self, mask) -> int:
        component_count, _labels = cv2.connectedComponents(np.asarray(mask, dtype=np.uint8), connectivity=8)
        return max(0, int(component_count) - 1)

    def _reject_area_subtract(self, reason: str) -> bool:
        document_id = self.document_id
        self._cancel_area_drawing()
        if document_id is not None:
            self.areaEditRejected.emit(document_id, reason)
        return False

    def _complete_area_subtract_polygon(self, polygon_points: list[Point]) -> bool:
        measurement = self._selected_area_measurement()
        if self._document is None or measurement is None or len(polygon_points) < 3:
            return False
        source_mask = self._area_measurement_to_mask(measurement)
        subtract_mask = self._magic_polygon_to_mask(polygon_points)
        if source_mask is None or subtract_mask is None:
            return self._reject_area_subtract("剔除区域无效，未修改")
        if not np.any(source_mask & subtract_mask):
            return self._reject_area_subtract("剔除区域未与当前面积相交")
        result_mask = source_mask & ~subtract_mask
        if not np.any(result_mask):
            return self._reject_area_subtract("剔除后无剩余面积，未修改")
        if self._area_mask_component_count(result_mask) > 1:
            return self._reject_area_subtract("当前版本不支持剔除后拆成多个独立区域")
        selected_mask, result_rings, result_polygon, _stats = magic_mask_to_geometry(
            result_mask,
            select_prompt_component=False,
        )
        if selected_mask is None or (len(result_polygon) < 3 and not result_rings):
            return self._reject_area_subtract("剔除结果无有效面积，未修改")
        if result_rings and len(result_polygon) < 3:
            result_polygon = list(result_rings[0])
        self._cancel_area_drawing()
        self.measurementEdited.emit(
            self._document.id,
            measurement.id,
            {
                "measurement_kind": "area",
                "mode": measurement.mode,
                "polygon_px": result_polygon,
                "area_rings_px": result_rings,
                "exact_area_px": magic_mask_area_px(selected_mask),
            },
        )
        return True

    def set_selected_measurement(self, measurement_id: str | None) -> None:
        if self._document is None:
            return
        previous_measurement_id = self._document.view_state.selected_measurement_id
        if measurement_id and self._document.get_measurement(measurement_id) is not None:
            self._set_object_selection(CanvasSelectionRef.measurement(measurement_id), notify=False)
        else:
            # Programmatic table synchronization clears only its own domain.
            previous = self._current_object_selection()
            self._document.select_measurement(None)
            current = self._current_object_selection()
            if current != previous:
                self._refresh_selection_visual(previous, current)
        if previous_measurement_id != measurement_id and self._tool_mode in {"polygon_area", "freehand_area"}:
            self._area_edit_operation_mode = AreaEditOperationMode.ADD

    def set_selected_overlay_annotation(self, overlay_id: str | None) -> None:
        if self._document is None:
            return
        annotation = self._document.get_overlay_annotation(overlay_id)
        if annotation is not None:
            self._set_object_selection(
                CanvasSelectionRef.overlay(annotation.id, annotation.normalized_kind()),
                notify=False,
            )
        else:
            previous = self._current_object_selection()
            self._document.select_overlay_annotation(None)
            current = self._current_object_selection()
            if current != previous:
                self._refresh_selection_visual(previous, current)

    def set_selected_text_annotation(self, text_id: str | None) -> None:
        if self._document is None:
            return
        annotation = self._document.get_text_annotation(text_id)
        if annotation is not None:
            self._set_object_selection(
                CanvasSelectionRef.overlay(annotation.id, annotation.normalized_kind()),
                notify=False,
            )
        else:
            previous = self._current_object_selection()
            self._document.select_text_annotation(None)
            current = self._current_object_selection()
            if current != previous:
                self._refresh_selection_visual(previous, current)

    def _current_object_selection(self) -> CanvasSelectionRef:
        if self._document is None:
            return CanvasSelectionRef.none()
        annotation = self._document.get_overlay_annotation(self._document.selected_overlay_id)
        if annotation is not None:
            return CanvasSelectionRef.overlay(annotation.id, annotation.normalized_kind())
        measurement = self._document.get_measurement(self._document.view_state.selected_measurement_id)
        if measurement is not None:
            return CanvasSelectionRef.measurement(measurement.id)
        return CanvasSelectionRef.none()

    def _set_object_selection(self, selection: CanvasSelectionRef, *, notify: bool = True) -> bool:
        """Apply one selection and notify only when the effective state changes."""

        if self._document is None:
            return False
        previous = self._current_object_selection()
        if selection.kind == "measurement" and selection.object_id:
            measurement = self._document.get_measurement(selection.object_id)
            if measurement is None:
                selection = CanvasSelectionRef.none()
            else:
                self._document.select_measurement(measurement.id)
        elif selection.kind == "overlay" and selection.object_id:
            annotation = self._document.get_overlay_annotation(selection.object_id)
            if annotation is None:
                selection = CanvasSelectionRef.none()
            else:
                selection = CanvasSelectionRef.overlay(annotation.id, annotation.normalized_kind())
                self._document.select_overlay_annotation(annotation.id)
        if selection.kind == "none":
            self._document.select_measurement(None)
            self._document.select_overlay_annotation(None)

        current = self._current_object_selection()
        if current == previous:
            return False
        self._refresh_selection_visual(previous, current)
        if notify:
            self.objectSelectionChanged.emit(self._document.id, current)
            # Compatibility signals remain for integrations outside MainWindow.
            self.measurementSelected.emit(
                self._document.id,
                current.object_id if current.kind == "measurement" else "",
            )
            self.overlaySelected.emit(
                self._document.id,
                current.object_id if current.kind == "overlay" else "",
            )
            self.textSelected.emit(
                self._document.id,
                current.object_id
                if current.kind == "overlay" and current.overlay_kind == OverlayAnnotationKind.TEXT
                else "",
            )
        return True

    def _refresh_selection_visual(
        self,
        previous: CanvasSelectionRef,
        current: CanvasSelectionRef,
    ) -> None:
        old_bounds = self._selection_display_bounds(previous)
        new_bounds = self._selection_display_bounds(current)
        self._apply_visual_change(
            CanvasVisualChange(
                object_ids=tuple(
                    object_id
                    for object_id in (previous.object_id, current.object_id)
                    if object_id
                ),
                old_bounds=old_bounds,
                new_bounds=new_bounds,
            )
        )

    def _selection_display_bounds(
        self,
        selection: CanvasSelectionRef,
    ) -> CanvasDisplayBounds | None:
        if self._document is None or selection.object_id is None:
            return None
        if selection.kind == "measurement":
            measurement = self._document.get_measurement(selection.object_id)
            if measurement is None:
                return None
            index = self._measurement_index()
            rect = measurement_display_image_bounds(
                measurement,
                self._document,
                self._settings,
                self.image_to_widget,
                suggested_line_width=2.0,
                endpoint_radius=4.0,
                count_number=(
                    index.count_number(measurement.id)
                    if index is not None
                    else None
                ),
                selected=True,
                exact_area_label=True,
            )
            if rect is None or not rect.isValid():
                return None
            # Preserve a small antialiasing/dirty-region safety band after the
            # shared function has accounted for dynamic strokes, endpoint
            # styles, handles and exact area-label placement.
            return CanvasDisplayBounds(rect).expanded(
                6.0 / max(self._zoom, 0.001)
            )
        if selection.kind == "overlay":
            annotation = self._document.get_overlay_annotation(selection.object_id)
            if annotation is None:
                return None
            if annotation.normalized_kind() == OverlayAnnotationKind.TEXT:
                widget_rect = annotation_rect(
                    annotation,
                    self._settings,
                    self.image_to_widget,
                ).adjusted(-8.0, -8.0, 8.0, 8.0)
                image_rect = self._paint_context().widget_to_image_transform.mapRect(
                    widget_rect
                )
                return CanvasDisplayBounds(image_rect)
            left, top, right, bottom = overlay_annotation_bounds(annotation)
            return CanvasDisplayBounds(
                QRectF(
                    left,
                    top,
                    max(1e-6, right - left),
                    max(1e-6, bottom - top),
                )
            ).expanded(16.0 / max(self._zoom, 0.001))
        return None

    def _apply_visual_change(self, change: CanvasVisualChange) -> None:
        if change.full_invalidation:
            self.update()
            return
        image_rects = [
            bounds.image_rect
            for bounds in (change.old_bounds, change.new_bounds)
            if bounds is not None and bounds.image_rect.isValid()
        ]
        if not image_rects:
            return
        dirty_image_rect = QRectF(image_rects[0])
        for image_rect in image_rects[1:]:
            dirty_image_rect = dirty_image_rect.united(image_rect)
        transform = self._paint_context().image_to_widget_transform
        dirty_widget_rect = (
            transform.mapRect(dirty_image_rect)
            .adjusted(-8.0, -8.0, 8.0, 8.0)
            .toAlignedRect()
            .intersected(self.rect())
        )
        if not dirty_widget_rect.isEmpty():
            self.update(dirty_widget_rect)

    def _preview_display_bounds(
        self,
        points: list[Point],
        *,
        padding_screen: float = 14.0,
    ) -> CanvasDisplayBounds | None:
        if not points:
            return None
        left = min(point.x for point in points)
        top = min(point.y for point in points)
        right = max(point.x for point in points)
        bottom = max(point.y for point in points)
        rect = QRectF(
            left,
            top,
            max(1e-6, right - left),
            max(1e-6, bottom - top),
        )
        return CanvasDisplayBounds(rect).expanded(
            padding_screen / max(self._zoom, 0.001)
        )

    def _update_preview_regions(
        self,
        old_points: list[Point],
        new_points: list[Point],
        *,
        padding_screen: float = 14.0,
    ) -> None:
        self._apply_visual_change(
            CanvasVisualChange(
                old_bounds=self._preview_display_bounds(
                    old_points,
                    padding_screen=padding_screen,
                ),
                new_bounds=self._preview_display_bounds(
                    new_points,
                    padding_screen=padding_screen,
                ),
            )
        )

    def _area_drag_display_bounds(
        self,
        measurement: Measurement,
        offset: Point,
    ) -> CanvasDisplayBounds | None:
        if self._document is None:
            return None
        rect = measurement_display_image_bounds(
            measurement,
            self._document,
            self._settings,
            self.image_to_widget,
            suggested_line_width=2.0,
            endpoint_radius=4.0,
            selected=True,
            exact_area_label=True,
        )
        if rect is None or not rect.isValid():
            return None
        return CanvasDisplayBounds(
            rect.translated(float(offset.x), float(offset.y))
        ).expanded(
            6.0 / max(self._zoom, 0.001)
        )

    def begin_scale_anchor_pick(self) -> None:
        self._scale_anchor_pick_active = True
        self._scale_anchor_preview_point = None
        self._update_cursor()
        self.focus_canvas()
        self.update()

    def end_scale_anchor_pick(self) -> None:
        self._scale_anchor_pick_active = False
        self._scale_anchor_preview_point = None
        self._update_cursor()
        self.update()

    def focus_canvas(self) -> None:
        self.setFocus(Qt.FocusReason.OtherFocusReason)

    def view_zoom(self) -> float:
        """Return the current logical image-to-widget scale."""

        return _bounded_view_zoom(self._zoom)

    def zoom_mode(self) -> CanvasZoomMode:
        return self._zoom_mode

    def viewport_snapshot(self) -> CanvasViewportSnapshot | None:
        if self._document is None or self._image is None:
            return None
        return CanvasViewportSnapshot(
            document_id=self._document.id,
            full_image_rect=QRectF(self._full_image_bounds()),
            mounted_image_rect=QRectF(self._paint_image_bounds()),
            visible_image_rect=QRectF(self._exact_visible_image_rect()),
            zoom=self.view_zoom(),
            mode=self._zoom_mode,
            device_pixel_ratio=max(1.0, float(self.devicePixelRatioF())),
            focus_index=self._viewport_focus_index(),
        )

    def set_view_zoom(self, zoom: float) -> None:
        """Set a custom logical zoom while preserving the view center."""

        if self._image is None:
            return
        center = QPointF(self.width() / 2.0, self.height() / 2.0)
        self._set_zoom_at_widget_position(
            _bounded_view_zoom(zoom),
            center,
            mode=CanvasZoomMode.CUSTOM,
        )

    def center_on_image_point(self, point: Point) -> None:
        """Keep the current zoom while bringing an image point to view center."""

        if self._image is None:
            return
        self._center_image_point_in_widget(point)
        if self._zoom_mode is CanvasZoomMode.FIT:
            self._zoom_mode = CanvasZoomMode.CUSTOM
        self._persist_view_state()
        self._publish_view_transform()
        self.update()

    def fit_to_view(self) -> None:
        if self._image is None:
            return
        image_width = max(1.0, float(self._image.width()))
        image_height = max(1.0, float(self._image.height()))
        viewport_width = max(1.0, float(self.width() - 40))
        viewport_height = max(1.0, float(self.height() - 40))
        zoom_x = viewport_width / image_width
        zoom_y = viewport_height / image_height
        self._zoom = _bounded_view_zoom(min(zoom_x, zoom_y))
        self._zoom_mode = CanvasZoomMode.FIT
        self._reset_proxy_warming()
        self._cancel_overlay_requests()
        target_width = image_width * self._zoom
        target_height = image_height * self._zoom
        if self._fit_alignment == "top_left":
            self._pan = Point(20.0, 20.0)
        else:
            self._pan = Point(
                (self.width() - target_width) / 2.0,
                (self.height() - target_height) / 2.0,
            )
        self._persist_view_state()
        self._publish_view_transform(zoom_changed=True)
        self.update()

    def actual_size(self) -> None:
        if self._image is None:
            return
        center = QPointF(self.width() / 2.0, self.height() / 2.0)
        center_image_point = self.widget_to_image(center)
        self._zoom = 1.0
        self._zoom_mode = CanvasZoomMode.ACTUAL
        self._reset_proxy_warming()
        self._cancel_overlay_requests()
        self._center_image_point_in_widget(center_image_point)
        self._persist_view_state()
        self._publish_view_transform(zoom_changed=True)
        self.update()

    def set_temporary_grab_pressed(self, pressed: bool) -> None:
        self._space_pressed = pressed
        if not pressed and not self._panning:
            self._temporary_grab_active = False
        elif pressed and not self._has_pointer_edit_operation():
            self._temporary_grab_active = True
        self._update_cursor()

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Space and not getattr(event, "isAutoRepeat", lambda: False)():
            self.set_temporary_grab_pressed(True)
            event.accept()
            return
        if self._roi_capture is not None:
            if event.key() == Qt.Key.Key_Escape:
                self.cancel_roi_capture()
                event.accept()
                return
            if (
                self._roi_capture.kind is ProjectRoiKind.POLYGON
                and event.modifiers() == Qt.KeyboardModifier.NoModifier
                and event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter)
            ):
                self._commit_roi_capture()
                event.accept()
                return
        if (
            event.modifiers() == Qt.KeyboardModifier.NoModifier
            and event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_F)
            and self.complete_magic_manual_subtract_draft()
        ):
            event.accept()
            return
        if (
            event.key() == Qt.Key.Key_Escape
            and is_magic_segment_tool_mode(self._tool_mode)
            and self.cancel_magic_subtract_draft()
        ):
            event.accept()
            return
        if (
            event.modifiers() == Qt.KeyboardModifier.NoModifier
            and event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_F)
            and self.commit_pending_path()
        ):
            event.accept()
            return
        if event.key() == Qt.Key.Key_Escape and is_magic_segment_tool_mode(self._tool_mode) and self.has_magic_segment_session():
            self.clear_magic_segment_session()
            event.accept()
            return
        if event.key() == Qt.Key.Key_Escape and is_fiber_quick_tool_mode(self._tool_mode) and self.has_fiber_quick_session():
            self.clear_fiber_quick_session()
            event.accept()
            return
        if event.key() == Qt.Key.Key_Escape and self._drawing_anchor_raw is not None:
            self._cancel_line_drawing()
            event.accept()
            return
        if event.key() == Qt.Key.Key_Escape and self.cancel_pending_path():
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Space and not getattr(event, "isAutoRepeat", lambda: False)():
            self.set_temporary_grab_pressed(False)
            event.accept()
            return
        super().keyReleaseEvent(event)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), canvas_workspace_background(self.palette()))
        if self._image is None or self._document is None:
            painter.setPen(canvas_workspace_foreground(self.palette()))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "打开图片后开始测量")
            return
        paint_context = self._paint_context(QRectF(event.rect()))

        target = QRectF(
            self._pan.x,
            self._pan.y,
            self._image.width() * self._zoom,
            self._image.height() * self._zoom,
        )
        painter.drawImage(target, self._image)
        painter.save()
        border_pen = QPen(canvas_image_border(self.palette()))
        border_pen.setWidthF(1.0)
        painter.setPen(border_pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(target)
        painter.restore()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self._draw_project_rois(painter, target)
        self._draw_annotations(painter, paint_context)
        self._draw_preview(painter)

    def resizeEvent(self, event) -> None:
        old_size = event.oldSize()
        old_center: Point | None = None
        if (
            self._image is not None
            and old_size.isValid()
            and old_size.width() > 0
            and old_size.height() > 0
        ):
            old_center = self.widget_to_image(
                QPointF(old_size.width() / 2.0, old_size.height() / 2.0)
            )
        super().resizeEvent(event)
        if self._image is None or self._document is None:
            return
        if self._zoom_mode is CanvasZoomMode.FIT:
            self.fit_to_view()
            return
        if old_center is not None:
            self._center_image_point_in_widget(old_center)
            self._persist_view_state()
        self._publish_view_transform()
        self.update()

    def hideEvent(self, event) -> None:
        """Stop producers owned by a canvas that is no longer visible."""

        self._end_canvas_pan()
        canvas_overlay_tile_cache.protect(id(self), ())
        self._reset_proxy_warming()
        self._cancel_overlay_requests()
        super().hideEvent(event)

    def wheelEvent(self, event: QWheelEvent) -> None:
        if self._image is None:
            return
        delta_y = event.angleDelta().y()
        delta_x = event.angleDelta().x()
        effective_delta = delta_y if delta_y != 0 else delta_x
        if effective_delta == 0:
            return
        cursor_position = event.position()
        zoom_factor = 1.15 if effective_delta > 0 else 1 / 1.15
        self._set_zoom_at_widget_position(
            _bounded_view_zoom(self._zoom * zoom_factor),
            cursor_position,
            mode=CanvasZoomMode.CUSTOM,
        )
        event.accept()

    def _begin_canvas_pan(self, button: Qt.MouseButton) -> None:
        """Start one physical-pixel-aligned pan session.

        Qt reports high-DPI pointer coordinates as logical floats. Accumulating
        those values directly can change ``pan * DPR``'s fractional phase on
        every event, which invalidates every exact overlay tile. Preserve the
        phase that was visible at press time and keep a separate unsnapped
        accumulator so sub-pixel mouse movement is never lost.
        """

        self._panning = True
        self._pan_button = button
        self._pan_drag_unsnapped = Point(self._pan.x, self._pan.y)
        dpr = max(1.0, float(self.devicePixelRatioF()))
        scaled_x = float(self._pan.x) * dpr
        scaled_y = float(self._pan.y) * dpr
        self._pan_drag_device_phase = (
            scaled_x - math.floor(scaled_x),
            scaled_y - math.floor(scaled_y),
        )
        self._pan_drag_device_pixel_ratio = dpr

    def _pan_at_stable_device_phase(self, unsnapped: Point) -> Point:
        phase = self._pan_drag_device_phase
        if phase is None:
            return Point(float(unsnapped.x), float(unsnapped.y))
        dpr = max(1.0, float(self.devicePixelRatioF()))
        if (
            self._pan_drag_device_pixel_ratio is None
            or not math.isclose(
                dpr,
                self._pan_drag_device_pixel_ratio,
                rel_tol=0.0,
                abs_tol=1e-8,
            )
        ):
            # Moving a window between monitors starts a new DPR namespace.
            # Preserve the last displayed phase once, then keep that new
            # namespace stable for the remainder of the drag.
            scaled_x = float(self._pan.x) * dpr
            scaled_y = float(self._pan.y) * dpr
            phase = (
                scaled_x - math.floor(scaled_x),
                scaled_y - math.floor(scaled_y),
            )
            self._pan_drag_device_phase = phase
            self._pan_drag_device_pixel_ratio = dpr
        snap_x = math.floor(
            (float(unsnapped.x) * dpr) - phase[0] + 0.5
        )
        snap_y = math.floor(
            (float(unsnapped.y) * dpr) - phase[1] + 0.5
        )
        return Point(
            (snap_x + phase[0]) / dpr,
            (snap_y + phase[1]) / dpr,
        )

    def _end_canvas_pan(self) -> None:
        self._panning = False
        self._pan_button = None
        self._pan_drag_unsnapped = None
        self._pan_drag_device_phase = None
        self._pan_drag_device_pixel_ratio = None

    def _overlay_motion_active(self) -> bool:
        return self._panning

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if self._image is None or self._document is None:
            return
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        self._last_mouse_pos = event.position()
        if event.button() in (Qt.MouseButton.MiddleButton, Qt.MouseButton.RightButton):
            self._begin_canvas_pan(event.button())
            self._update_cursor()
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return
        if self._temporary_grab_active:
            self._begin_canvas_pan(event.button())
            self._update_cursor()
            return
        if self._read_only:
            return

        image_point = self.widget_to_image(event.position())
        if self._roi_capture is not None:
            self._roi_capture_mouse_press(image_point)
            return

        if self._scale_anchor_pick_active:
            if self._point_in_image(image_point):
                self._scale_anchor_preview_point = self._clamp_to_image(image_point, pixel_center=False)
                self.scaleAnchorPicked.emit(self._document.id, self._scale_anchor_preview_point)
            return

        if is_magic_segment_tool_mode(self._tool_mode):
            if not self._point_in_image(image_point):
                return
            self._set_object_selection(CanvasSelectionRef.none())
            point = self._clamp_to_image(image_point, pixel_center=False)
            active_stage = self._magic_segment.active_stage
            if active_stage == MagicSegmentOperationMode.SUBTRACT and self._magic_manual_subtract_mode_active():
                if self._magic_segment.busy:
                    return
                if self.current_magic_subtract_input_mode() == MagicSegmentSubtractInputMode.POLYGON:
                    if self._can_close_polygon_with_point(point):
                        self._complete_magic_manual_subtract_polygon(list(self._drawing_polygon_points))
                        return
                    previous_points = list(self._drawing_polygon_points)
                    self._drawing_polygon_points.append(point)
                    self._area_hover_point = point
                    self._update_preview_regions(
                        previous_points,
                        list(self._drawing_polygon_points),
                    )
                    self._emit_magic_segment_session_changed(repaint=False)
                    return
                previous_points = list(self._drawing_polygon_points)
                self._drawing_polygon_points = [point]
                self._area_hover_point = point
                self._drawing_freehand_active = True
                self._freehand_last_sample_at = time.monotonic()
                self._update_preview_regions(
                    previous_points,
                    list(self._drawing_polygon_points),
                )
                self._emit_magic_segment_session_changed(repaint=False)
                return
            if self._magic_segment.prompt_type_for_stage(active_stage) == "negative":
                self._magic_segment.negative_points_for_stage(active_stage).append(point)
            else:
                self._magic_segment.positive_points_for_stage(active_stage).append(point)
            if self._magic_segment.busy:
                self._magic_segment.pending_stage = active_stage
                self._magic_segment.pending_recompute = True
            else:
                payload = self._begin_magic_segment_request(active_stage)
                if payload is not None:
                    self.magicSegmentRequested.emit(self._document.id, payload)
            self.update()
            self._emit_magic_segment_session_changed()
            return

        if is_fiber_quick_tool_mode(self._tool_mode):
            if not self._point_in_image(image_point):
                return
            self._set_object_selection(CanvasSelectionRef.none())
            point = self._clamp_to_image(image_point, pixel_center=False)
            if self._fiber_quick.prompt_type == "negative":
                self._fiber_quick.negative_points.append(point)
            else:
                self._fiber_quick.positive_points.append(point)
            if not self._fiber_quick.positive_points:
                self.update()
                self._emit_magic_segment_session_changed()
                return
            if self._fiber_quick.segmentation_busy:
                self._fiber_quick.pending_recompute = True
            else:
                self._fiber_quick.debug_payload = {}
                payload = self._begin_fiber_quick_request()
                if payload is not None:
                    self.magicSegmentRequested.emit(self._document.id, payload)
            self.update()
            self._emit_magic_segment_session_changed()
            return

        if is_reference_propagation_tool_mode(self._tool_mode):
            if not self._point_in_image(image_point):
                return
            if self._reference_instance.busy:
                return
            self._set_object_selection(CanvasSelectionRef.none())
            point = self._clamp_to_image(image_point, pixel_center=False)
            measurement_id = self._hit_test_area_measurement(point)
            if measurement_id is not None:
                self._reference_instance = ReferenceInstanceSession()
                self._reference_instance.request_id += 1
                self._reference_instance.busy = True
                self.magicSegmentRequested.emit(
                    self._document.id,
                    {
                        "request_id": self._reference_instance.request_id,
                        "tool_mode": self._tool_mode,
                        "reference_measurement_id": measurement_id,
                    },
                )
                self.update()
                self._emit_magic_segment_session_changed()
                return
            self._reference_instance = ReferenceInstanceSession()
            self._reference_instance.dragging = True
            self._reference_instance.drag_start = point
            self._reference_instance.drag_end = point
            self.update()
            self._emit_magic_segment_session_changed()
            return

        if self._tool_mode == "overlay":
            if self._point_in_image(image_point):
                if self._overlay_tool_kind == OverlayAnnotationKind.TEXT:
                    anchor = self._clamp_to_image(image_point, pixel_center=False)
                    self.overlayCreateRequested.emit(
                        self._document.id,
                        {
                            "kind": OverlayAnnotationKind.TEXT,
                            "anchor_px": anchor,
                        },
                    )
                    self.textPlacementRequested.emit(self._document.id, anchor)
                else:
                    anchor = self._clamp_to_image(image_point, pixel_center=False)
                    self._drawing_overlay_start = anchor
                    self._drawing_overlay_end = anchor
                    self._set_object_selection(CanvasSelectionRef.none())
                    self.update()
            return

        if self._tool_mode == "polygon_area":
            if not self._point_in_image(image_point):
                return
            subtract_active = self._area_subtract_mode_active()
            if not subtract_active:
                self._set_object_selection(CanvasSelectionRef.none())
            elif self._document.selected_overlay_id is not None:
                # Subtract keeps the selected area but still dismisses overlays.
                self._document.select_overlay_annotation(None)
            point = self._clamp_to_image(image_point, pixel_center=False)
            if self._can_close_polygon_with_point(point):
                if subtract_active:
                    self._complete_area_subtract_polygon(list(self._drawing_polygon_points))
                    return
                self._complete_area_measurement("polygon_area", list(self._drawing_polygon_points))
                return
            previous_points = list(self._drawing_polygon_points)
            self._drawing_polygon_points.append(point)
            self._area_hover_point = point
            self.pathSessionChanged.emit(self._document.id)
            self._update_preview_regions(
                previous_points,
                list(self._drawing_polygon_points),
            )
            return

        if self._tool_mode == "continuous_manual":
            if not self._point_in_image(image_point):
                return
            self._set_object_selection(CanvasSelectionRef.none())
            point = self._clamp_to_image(image_point, pixel_center=False)
            if not self._continuous_manual_tool_strategy.should_append_point(
                self._drawing_polygon_points,
                point,
            ):
                return
            previous_points = list(self._drawing_polygon_points)
            self._drawing_polygon_points.append(point)
            self._area_hover_point = point
            self.pathSessionChanged.emit(self._document.id)
            self._update_preview_regions(
                previous_points,
                list(self._drawing_polygon_points),
            )
            return

        if self._tool_mode == "freehand_area":
            if not self._point_in_image(image_point):
                return
            if not self._area_subtract_mode_active():
                self._set_object_selection(CanvasSelectionRef.none())
            elif self._document.selected_overlay_id is not None:
                self._document.select_overlay_annotation(None)
            point = self._clamp_to_image(image_point, pixel_center=False)
            previous_points = list(self._drawing_polygon_points)
            self._drawing_polygon_points = [point]
            self._area_hover_point = point
            self._drawing_freehand_active = True
            self._freehand_last_sample_at = time.monotonic()
            self.pathSessionChanged.emit(self._document.id)
            self._update_preview_regions(
                previous_points,
                list(self._drawing_polygon_points),
            )
            return

        if self._tool_mode == "count":
            if not self._point_in_image(image_point):
                return
            point = self._clamp_to_image(image_point, pixel_center=False)
            if self._document.selected_overlay_id is not None:
                self._set_object_selection(CanvasSelectionRef.none())
            self.lineCommitted.emit(
                self._document.id,
                "count",
                self._count_tool_strategy.commit_payload(point),
            )
            return

        if self._tool_mode == "calibration":
            if self._point_in_image(image_point):
                anchor = self._anchor_point_for_event(image_point, event.modifiers())
                self._begin_line_drawing(anchor)
                self._update_preview_regions(
                    [],
                    [anchor],
                    padding_screen=18.0,
                )
            return

        if self._tool_mode == "snap":
            if self._drawing_anchor_raw is not None and self._line_commit_on_second_click:
                self._commit_click_line(image_point, event.modifiers())
                return
            if self._point_in_image(image_point):
                anchor = self._anchor_point_for_event(image_point, event.modifiers())
                self._begin_line_drawing(anchor, commit_on_second_click=True)
                self._update_preview_regions(
                    [],
                    [anchor],
                    padding_screen=18.0,
                )
            return

        if self._tool_mode == "select":
            overlay_handle = self._hit_test_selected_overlay_handle(image_point)
            if overlay_handle is not None:
                annotation = self._document.get_overlay_annotation(overlay_handle[0])
                if annotation is not None:
                    self._set_object_selection(
                        CanvasSelectionRef.overlay(annotation.id, annotation.normalized_kind())
                    )
                    self._dragging_overlay_handle = overlay_handle
                    self._drag_overlay_press_point = image_point
                    self._drag_overlay_origin = annotation.clone()
                    self._drag_overlay_preview = annotation.clone()
                    self.update()
                    return

            overlay_hit = self._hit_test_overlay_annotation(event.position(), image_point)
            if overlay_hit is not None:
                annotation = self._document.get_overlay_annotation(overlay_hit)
                if annotation is not None:
                    self._set_object_selection(
                        CanvasSelectionRef.overlay(annotation.id, annotation.normalized_kind())
                    )
                    self._dragging_overlay_id = annotation.id
                    self._drag_overlay_press_point = image_point
                    self._drag_overlay_origin = annotation.clone()
                    self._drag_overlay_preview = annotation.clone()
                    self.update()
                    return

            area_handle = self._hit_test_selected_area_handle(image_point)
            if area_handle is not None:
                self._begin_area_drag(area_handle, image_point)
                self.update()
                return

        selected_handle = self._hit_test_selected_endpoint(image_point)
        if selected_handle is not None:
            self._dragging_handle = selected_handle
            self._drag_preview_line = self._measurement_line(selected_handle[0])
            self.update()
            return

        if self._tool_mode == "select":
            area_measurement_id = self._hit_test_area_measurement(image_point)
            if area_measurement_id is not None:
                self._set_object_selection(CanvasSelectionRef.measurement(area_measurement_id))
                return

            handle = self._hit_test_endpoint(image_point)
            if handle is not None:
                self._dragging_handle = handle
                self._drag_preview_line = self._measurement_line(handle[0])
                self._set_object_selection(CanvasSelectionRef.measurement(handle[0]))
                self.update()
                return

            measurement_id = self._hit_test_measurement(image_point)
            selection = (
                CanvasSelectionRef.measurement(measurement_id)
                if measurement_id is not None
                else CanvasSelectionRef.none()
            )
            self._set_object_selection(selection)
            return

        if self._point_in_image(image_point):
            anchor = self._anchor_point_for_event(image_point, event.modifiers())
            self._begin_line_drawing(anchor)
            self._update_preview_regions(
                [],
                [anchor],
                padding_screen=18.0,
            )

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._image is None or self._document is None:
            return
        if self._panning:
            delta = event.position() - self._last_mouse_pos
            unsnapped = self._pan_drag_unsnapped or self._pan
            unsnapped = Point(
                unsnapped.x + delta.x(),
                unsnapped.y + delta.y(),
            )
            self._pan_drag_unsnapped = unsnapped
            self._pan = self._pan_at_stable_device_phase(unsnapped)
            if self._zoom_mode is CanvasZoomMode.FIT:
                self._zoom_mode = CanvasZoomMode.CUSTOM
            self._last_mouse_pos = event.position()
            self._persist_view_state()
            self._publish_view_transform()
            self.update()
            return

        if self._scale_anchor_pick_active:
            self._scale_anchor_preview_point = self._clamp_to_image(self.widget_to_image(event.position()), pixel_center=False)
            self.update()
            return
        if self._read_only:
            return

        image_point = self.widget_to_image(event.position())
        if self._roi_capture is not None:
            self._roi_capture_mouse_move(image_point)
            return

        if is_reference_propagation_tool_mode(self._tool_mode) and self._reference_instance.dragging:
            self._reference_instance.drag_end = self._clamp_to_image(image_point, pixel_center=False)
            self.update()
            return

        if (
            (
                self._tool_mode in {"polygon_area", "continuous_manual"}
                or self._magic_manual_subtract_mode_active(MagicSegmentSubtractInputMode.POLYGON)
            )
            and self._drawing_polygon_points
        ):
            previous_hover = self._area_hover_point
            next_hover = self._clamp_to_image(
                image_point,
                pixel_center=False,
            )
            self._area_hover_point = next_hover
            fixed_points = [self._drawing_polygon_points[-1]]
            if len(self._drawing_polygon_points) >= 2:
                fixed_points.append(self._drawing_polygon_points[0])
            self._update_preview_regions(
                fixed_points
                + ([previous_hover] if previous_hover is not None else []),
                fixed_points + [next_hover],
            )
            return

        if self._drawing_freehand_active:
            previous_last = (
                self._drawing_polygon_points[-1]
                if self._drawing_polygon_points
                else None
            )
            previous_count = len(self._drawing_polygon_points)
            self._append_freehand_point(
                self._clamp_to_image(image_point, pixel_center=False)
            )
            self._area_hover_point = self._drawing_polygon_points[-1] if self._drawing_polygon_points else None
            if len(self._drawing_polygon_points) != previous_count:
                changed_points = (
                    [previous_last, self._drawing_polygon_points[-1]]
                    if previous_last is not None
                    else [self._drawing_polygon_points[-1]]
                )
                self._update_preview_regions(
                    (
                        [previous_last]
                        if previous_last is not None
                        else []
                    ),
                    changed_points,
                )
            return

        if self._drawing_anchor_raw is not None:
            previous_line = self._drawing_line
            start, end = self._apply_line_constraints(
                self._drawing_anchor_raw,
                image_point,
                event.modifiers(),
                snap_anchor=True,
            )
            self._drawing_line = Line(start=start, end=end)
            if self.document_id is not None:
                self.pathSessionChanged.emit(self.document_id)
            self._update_preview_regions(
                (
                    [previous_line.start, previous_line.end]
                    if previous_line is not None
                    else [self._drawing_anchor_raw]
                ),
                [start, end],
                padding_screen=18.0,
            )
            return

        if self._drawing_overlay_start is not None:
            self._drawing_overlay_end = self._constrain_overlay_candidate(
                self._drawing_overlay_start,
                image_point,
                event.modifiers(),
            )
            self.update()
            return

        if (
            self._drag_overlay_origin is not None
            and self._drag_overlay_press_point is not None
            and self._dragging_overlay_id is not None
        ):
            dx = image_point.x - self._drag_overlay_press_point.x
            dy = image_point.y - self._drag_overlay_press_point.y
            self._drag_overlay_preview = self._translate_overlay_annotation(self._drag_overlay_origin, dx, dy)
            self.update()
            return

        if (
            self._drag_overlay_origin is not None
            and self._dragging_overlay_handle is not None
        ):
            self._drag_overlay_preview = self._resize_overlay_annotation(
                self._drag_overlay_origin,
                self._dragging_overlay_handle[1],
                self._clamp_to_image(image_point, pixel_center=False),
                event.modifiers(),
            )
            self.update()
            return

        if (
            self._dragging_area_handle is not None
            and self._drag_area_press_point is not None
        ):
            _measurement_id, handle_kind, ring_index, point_index = self._dragging_area_handle
            measurement = self._document.get_measurement(_measurement_id)
            if measurement is None:
                return
            if handle_kind == "center":
                previous_offset = self._drag_area_preview_offset or Point(
                    0.0,
                    0.0,
                )
                dx = image_point.x - self._drag_area_press_point.x
                dy = image_point.y - self._drag_area_press_point.y
                # Whole-object movement keeps only a scalar delta.  Rebuilding
                # every RAW ring for each mouse event made dense magic-segment
                # areas proportional to their vertex count while dragging.
                next_offset = Point(dx, dy)
                self._drag_area_preview_offset = next_offset
                self._apply_visual_change(
                    CanvasVisualChange(
                        object_ids=(measurement.id,),
                        old_bounds=self._area_drag_display_bounds(
                            measurement,
                            previous_offset,
                        ),
                        new_bounds=self._area_drag_display_bounds(
                            measurement,
                            next_offset,
                        ),
                    )
                )
                return
            elif point_index is not None and self._drag_area_origin_points is not None:
                previous_points = list(
                    self._drag_area_preview_points
                    or self._drag_area_origin_points
                )
                preview = list(self._drag_area_origin_points)
                preview[point_index] = self._clamp_to_image(image_point, pixel_center=False)
                self._drag_area_preview_points = preview
                if self._drag_area_origin_rings is not None and ring_index is not None:
                    preview_rings = self._clone_magic_rings(self._drag_area_origin_rings)
                    if 0 <= ring_index < len(preview_rings) and 0 <= point_index < len(preview_rings[ring_index]):
                        preview_rings[ring_index][point_index] = self._clamp_to_image(image_point, pixel_center=False)
                    self._drag_area_preview_rings = preview_rings
                    if preview_rings:
                        self._drag_area_preview_points = list(preview_rings[0])
            else:
                return
            old_preview_bounds = self._preview_display_bounds(
                previous_points,
                padding_screen=18.0,
            )
            new_preview_bounds = self._preview_display_bounds(
                list(self._drag_area_preview_points or []),
                padding_screen=18.0,
            )
            committed_bounds = self._area_drag_display_bounds(
                measurement,
                Point(0.0, 0.0),
            )
            if committed_bounds is not None:
                old_preview_bounds = CanvasDisplayBounds(
                    committed_bounds.image_rect.united(
                        old_preview_bounds.image_rect
                        if old_preview_bounds is not None
                        else committed_bounds.image_rect
                    )
                )
                new_preview_bounds = CanvasDisplayBounds(
                    committed_bounds.image_rect.united(
                        new_preview_bounds.image_rect
                        if new_preview_bounds is not None
                        else committed_bounds.image_rect
                    )
                )
            self._apply_visual_change(
                CanvasVisualChange(
                    object_ids=(measurement.id,),
                    old_bounds=old_preview_bounds,
                    new_bounds=new_preview_bounds,
                )
            )
            return

        if self._dragging_handle is not None:
            previous_line = self._drag_preview_line
            measurement_id, endpoint_name = self._dragging_handle
            measurement = self._document.get_measurement(measurement_id)
            if measurement is None:
                return
            base_line = measurement.effective_line()
            fixed_point = base_line.end if endpoint_name == "start" else base_line.start
            fixed_point, moving_point = self._apply_line_constraints(
                fixed_point,
                image_point,
                event.modifiers(),
                snap_anchor=False,
            )
            if endpoint_name == "start":
                self._drag_preview_line = Line(start=moving_point, end=fixed_point)
            else:
                self._drag_preview_line = Line(start=fixed_point, end=moving_point)
            old_preview_bounds = self._preview_display_bounds(
                (
                    [previous_line.start, previous_line.end]
                    if previous_line is not None
                    else [base_line.start, base_line.end]
                ),
                padding_screen=18.0,
            )
            new_preview_bounds = self._preview_display_bounds(
                [
                    self._drag_preview_line.start,
                    self._drag_preview_line.end,
                ],
                padding_screen=18.0,
            )
            committed_bounds = self._selection_display_bounds(
                CanvasSelectionRef.measurement(measurement.id)
            )
            if committed_bounds is not None:
                old_preview_bounds = CanvasDisplayBounds(
                    committed_bounds.image_rect.united(
                        old_preview_bounds.image_rect
                        if old_preview_bounds is not None
                        else committed_bounds.image_rect
                    )
                )
                new_preview_bounds = CanvasDisplayBounds(
                    committed_bounds.image_rect.united(
                        new_preview_bounds.image_rect
                        if new_preview_bounds is not None
                        else committed_bounds.image_rect
                    )
                )
            self._apply_visual_change(
                CanvasVisualChange(
                    object_ids=(measurement.id,),
                    old_bounds=old_preview_bounds,
                    new_bounds=new_preview_bounds,
                )
            )

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._document is None:
            return
        if self._panning and self._pan_button == event.button():
            self._end_canvas_pan()
            if self._space_pressed and not self._has_pointer_edit_operation():
                self._temporary_grab_active = True
            elif not self._space_pressed:
                self._temporary_grab_active = False
            self._update_cursor()
            # Intermediate pan frames deliberately avoid warming the tile
            # cache. Once motion stops, request exactly one current-view frame.
            self.update()
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return
        if self._read_only:
            self._update_cursor()
            return
        if self._roi_capture is not None:
            self._roi_capture_mouse_release(
                self.widget_to_image(event.position())
            )
            return

        if is_reference_propagation_tool_mode(self._tool_mode) and self._reference_instance.dragging:
            start = self._reference_instance.drag_start
            end = self._reference_instance.drag_end
            self._reference_instance.dragging = False
            self._reference_instance.drag_start = None
            self._reference_instance.drag_end = None
            if start is not None and end is not None:
                width = abs(end.x - start.x)
                height = abs(end.y - start.y)
                if width >= 8.0 and height >= 8.0:
                    self._reference_instance.request_id += 1
                    self._reference_instance.busy = True
                    self.magicSegmentRequested.emit(
                        self._document.id,
                        {
                            "request_id": self._reference_instance.request_id,
                            "tool_mode": self._tool_mode,
                            "reference_box": {
                                "start": start,
                                "end": end,
                            },
                        },
                    )
                else:
                    self._reference_instance = ReferenceInstanceSession()
            self.update()
            self._emit_magic_segment_session_changed()
            return

        if self._drawing_freehand_active:
            polygon_points = list(self._drawing_polygon_points)
            if self._magic_manual_subtract_mode_active(MagicSegmentSubtractInputMode.FREEHAND):
                if not self._complete_magic_manual_subtract_polygon(polygon_points):
                    self._cancel_area_drawing()
                    self._emit_magic_segment_session_changed()
                return
            if self._area_subtract_mode_active():
                if not self._complete_area_subtract_polygon(polygon_points) and self.has_pending_path_drawing():
                    self._cancel_area_drawing()
                return
            self._cancel_area_drawing()
            if len(polygon_points) >= 3:
                self._complete_area_measurement("freehand_area", polygon_points)
            return

        if self._drawing_line is not None:
            if self._line_commit_on_second_click:
                self.update()
                return
            line = self._line_tool_strategy.commit_payload(self._drawing_line)
            self._cancel_line_drawing()
            if line is not None:
                self.lineCommitted.emit(self._document.id, self._tool_mode, line)
            if self._space_pressed:
                self._temporary_grab_active = True
                self._update_cursor()
            return

        if self._drawing_overlay_start is not None and self._drawing_overlay_end is not None:
            start_point = self._drawing_overlay_start
            end_point = self._drawing_overlay_end
            self._drawing_overlay_start = None
            self._drawing_overlay_end = None
            if self._overlay_geometry_visible(start_point, end_point):
                self.overlayCreateRequested.emit(
                    self._document.id,
                    {
                        "kind": self._overlay_tool_kind,
                        "start_px": start_point,
                        "end_px": end_point,
                    },
                )
            if self._space_pressed:
                self._temporary_grab_active = True
                self._update_cursor()
            self.update()
            return

        if self._dragging_overlay_id is not None and self._drag_overlay_preview is not None:
            overlay_id = self._dragging_overlay_id
            preview = self._drag_overlay_preview
            self._cancel_overlay_interaction()
            self.overlayEdited.emit(self._document.id, overlay_id, preview)
            if preview.normalized_kind() == OverlayAnnotationKind.TEXT:
                self.textMoved.emit(self._document.id, overlay_id, preview.anchor_px)
            if self._space_pressed:
                self._temporary_grab_active = True
                self._update_cursor()
            self.update()
            return

        if self._dragging_overlay_handle is not None and self._drag_overlay_preview is not None:
            overlay_id, _handle = self._dragging_overlay_handle
            preview = self._drag_overlay_preview
            self._cancel_overlay_interaction()
            self.overlayEdited.emit(self._document.id, overlay_id, preview)
            if preview.normalized_kind() == OverlayAnnotationKind.TEXT:
                self.textMoved.emit(self._document.id, overlay_id, preview.anchor_px)
            if self._space_pressed:
                self._temporary_grab_active = True
                self._update_cursor()
            self.update()
            return

        if self._dragging_area_handle is not None:
            measurement_id, handle_kind, _ring_index, _index = self._dragging_area_handle
            measurement = self._document.get_measurement(measurement_id)
            preview_polygon: list[Point] = []
            preview_rings: list[list[Point]] = []
            exact_area_px: float | None = None
            if handle_kind == "center" and measurement is not None:
                offset = self._drag_area_preview_offset or Point(0.0, 0.0)
                # The only O(vertices) work for a whole-object drag happens at
                # commit.  RAW coordinates remain untouched until this point.
                if measurement.area_rings_px:
                    preview_rings = [
                        polygon_translate(ring, offset.x, offset.y)
                        for ring in measurement.area_rings_px
                    ]
                    if preview_rings:
                        preview_polygon = list(preview_rings[0])
                else:
                    preview_polygon = polygon_translate(
                        measurement.polygon_px,
                        offset.x,
                        offset.y,
                    )
                # A pure translation does not alter mask/vector area, so its
                # exact scalar remains valid and keeps highest priority.
                exact_area_px = measurement.exact_area_px
            elif self._drag_area_preview_points is not None:
                preview_polygon = list(self._drag_area_preview_points)
                preview_rings = (
                    self._clone_magic_rings(self._drag_area_preview_rings)
                    if self._drag_area_preview_rings is not None
                    else []
                )
                if preview_rings:
                    preview_polygon = list(preview_rings[0])
            self._clear_area_drag_state()
            if measurement is None or len(preview_polygon) < 3:
                self.update()
                return
            self.measurementEdited.emit(
                self._document.id,
                measurement_id,
                {
                    "measurement_kind": "area",
                    "mode": "polygon_area",
                    "polygon_px": preview_polygon,
                    "area_rings_px": preview_rings,
                    "exact_area_px": exact_area_px,
                },
            )
            if self._space_pressed:
                self._temporary_grab_active = True
                self._update_cursor()
            self.update()
            return

        if self._dragging_handle is not None and self._drag_preview_line is not None:
            measurement_id, _ = self._dragging_handle
            preview = self._drag_preview_line
            self._dragging_handle = None
            self._drag_preview_line = None
            self.measurementEdited.emit(self._document.id, measurement_id, preview)
            if self._space_pressed:
                self._temporary_grab_active = True
                self._update_cursor()
            self.update()
            return

        self._dragging_handle = None
        self._drag_preview_line = None
        self._cancel_overlay_interaction()
        self._clear_area_drag_state()
        self._update_cursor()

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        if self._image is None or self._document is None:
            return
        if self._read_only:
            return
        if (
            self._roi_capture is not None
            and self._roi_capture.kind is ProjectRoiKind.POLYGON
            and event.button() == Qt.MouseButton.LeftButton
        ):
            point = self.widget_to_image(event.position())
            if self._point_in_image(point):
                point = self._clamp_roi_point(point)
                if (
                    not self._roi_capture.points
                    or distance(self._roi_capture.points[-1], point) > 1e-6
                ):
                    self._roi_capture.points.append(point)
            self._commit_roi_capture()
            event.accept()
            return
        if (
            event.button() == Qt.MouseButton.LeftButton
            and (
                self._tool_mode in {"polygon_area", "continuous_manual"}
                or self._magic_manual_subtract_mode_active(MagicSegmentSubtractInputMode.POLYGON)
            )
            and len(self._drawing_polygon_points) >= 1
        ):
            point = self._clamp_to_image(self.widget_to_image(event.position()), pixel_center=False)
            previous_point_count = len(self._drawing_polygon_points)
            if (
                self._tool_mode == "continuous_manual"
                and not self._magic_manual_subtract_mode_active(MagicSegmentSubtractInputMode.POLYGON)
            ):
                self._drawing_polygon_points = self._continuous_manual_tool_strategy.points_with_candidate(
                    self._drawing_polygon_points,
                    point,
                    include_threshold=False,
                )
            elif distance(point, self._drawing_polygon_points[-1]) > 1.0:
                self._drawing_polygon_points.append(point)
            if len(self._drawing_polygon_points) != previous_point_count:
                self.pathSessionChanged.emit(self._document.id)
                if self._magic_manual_subtract_mode_active(MagicSegmentSubtractInputMode.POLYGON):
                    self._emit_magic_segment_session_changed()
            if self._magic_manual_subtract_mode_active(MagicSegmentSubtractInputMode.POLYGON):
                if len(self._drawing_polygon_points) >= 3:
                    self._complete_magic_manual_subtract_polygon(list(self._drawing_polygon_points))
                    event.accept()
                    return
            if (
                self._tool_mode == "polygon_area"
                and self._area_subtract_mode_active()
                and len(self._drawing_polygon_points) >= 3
            ):
                self._complete_area_subtract_polygon(list(self._drawing_polygon_points))
                event.accept()
                return
            if self._tool_mode == "polygon_area" and len(self._drawing_polygon_points) >= 3:
                self._complete_area_measurement("polygon_area", list(self._drawing_polygon_points))
                event.accept()
                return
            if self._tool_mode == "continuous_manual" and self._continuous_manual_tool_strategy.can_commit(
                self._drawing_polygon_points,
            ):
                self._complete_continuous_measurement(list(self._drawing_polygon_points))
                event.accept()
                return
        super().mouseDoubleClickEvent(event)

    def widget_to_image(self, position: QPointF) -> Point:
        return Point(
            x=(position.x() - self._pan.x) / self._zoom,
            y=(position.y() - self._pan.y) / self._zoom,
        )

    def image_to_widget(self, point: Point) -> QPointF:
        return QPointF(
            self._pan.x + (point.x * self._zoom),
            self._pan.y + (point.y * self._zoom),
        )

    def _center_image_point_in_widget(self, point: Point) -> None:
        current = self.image_to_widget(point)
        self._pan = Point(
            self._pan.x + (self.width() / 2.0 - current.x()),
            self._pan.y + (self.height() / 2.0 - current.y()),
        )

    def _set_zoom_at_widget_position(
        self,
        zoom: float,
        position: QPointF,
        *,
        mode: CanvasZoomMode,
    ) -> None:
        if self._image is None:
            return
        image_before = self.widget_to_image(position)
        self._zoom = _bounded_view_zoom(zoom)
        self._zoom_mode = mode
        self._reset_proxy_warming()
        self._cancel_overlay_requests()
        current = self.image_to_widget(image_before)
        self._pan = Point(
            self._pan.x + position.x() - current.x(),
            self._pan.y + position.y() - current.y(),
        )
        self._persist_view_state()
        self._publish_view_transform(zoom_changed=True)
        self.update()

    def _point_in_image(self, point: Point) -> bool:
        if self._image is None:
            return False
        return 0 <= point.x < self._image.width() and 0 <= point.y < self._image.height()

    def _measurement_line(self, measurement_id: str) -> Line | None:
        if self._document is None:
            return None
        measurement = self._document.get_measurement(measurement_id)
        if measurement is None or measurement.measurement_kind != "line":
            return None
        return measurement.effective_line()

    def _overlay_widget_origin(self) -> QPointF:
        """Return the widget translation for global image coordinates.

        Ordinary images start at image coordinate ``(0, 0)``, so the
        translation is the canvas pan. Virtualized canvases can override this
        while measurements and overlay-cache tiles remain in global image
        coordinates.
        """

        return QPointF(float(self._pan.x), float(self._pan.y))

    def _paint_image_bounds(self) -> QRectF:
        """Return the global image-space bounds of the displayed raster."""

        if self._image is None:
            return QRectF()
        return QRectF(
            0.0,
            0.0,
            float(self._image.width()),
            float(self._image.height()),
        )

    def _full_image_bounds(self) -> QRectF:
        if self._document is not None:
            width, height = self._document.image_size
            return QRectF(0.0, 0.0, float(width), float(height))
        return QRectF(self._paint_image_bounds())

    def _viewport_focus_index(self) -> int | None:
        return None

    def _exact_visible_image_rect(self) -> QRectF:
        if self._image is None:
            return QRectF()
        transform = QTransform()
        overlay_origin = self._overlay_widget_origin()
        transform.translate(float(overlay_origin.x()), float(overlay_origin.y()))
        transform.scale(float(self._zoom), float(self._zoom))
        inverse, invertible = transform.inverted()
        if not invertible:
            return QRectF()
        visible = inverse.mapRect(QRectF(self.rect()))
        return visible.intersected(self._paint_image_bounds())

    def visible_source_pixel_rect(self) -> tuple[float, float, float, float] | None:
        """Return the exact visible field in authoritative image coordinates.

        This public, read-only snapshot is used by bounded pixel-processing
        previews.  It deliberately omits the paint-time padding used for
        labels, hit targets and overlay cache warming.
        """

        visible = self._exact_visible_image_rect()
        if visible.isEmpty() or not visible.isValid():
            return None
        return (
            float(visible.x()),
            float(visible.y()),
            float(visible.width()),
            float(visible.height()),
        )

    def _publish_view_transform(self, *, zoom_changed: bool = False) -> None:
        snapshot = self.viewport_snapshot()
        if zoom_changed:
            self.viewZoomChanged.emit(self.view_zoom())
        if snapshot is None or snapshot == self._last_view_transform_snapshot:
            return
        self._last_view_transform_snapshot = snapshot
        self.viewTransformChanged.emit(snapshot)

    def _paint_context(self, widget_rect: QRectF | None = None) -> CanvasPaintContext:
        overlay_origin = self._overlay_widget_origin()
        transform = QTransform()
        transform.translate(float(overlay_origin.x()), float(overlay_origin.y()))
        transform.scale(float(self._zoom), float(self._zoom))
        inverse, invertible = transform.inverted()
        if not invertible:
            inverse = QTransform()
        clipped_widget_rect = (
            QRectF(self.rect())
            if widget_rect is None
            else widget_rect.intersected(QRectF(self.rect()))
        )
        image_rect = inverse.mapRect(clipped_widget_rect)
        image_padding = max(16.0, 28.0 / max(self._zoom, 0.001))
        image_rect = image_rect.adjusted(
            -image_padding,
            -image_padding,
            image_padding,
            image_padding,
        )
        if self._image is not None:
            image_rect = image_rect.intersected(self._paint_image_bounds())
        return CanvasPaintContext(
            widget_rect=clipped_widget_rect,
            image_rect=image_rect,
            image_to_widget_transform=transform,
            widget_to_image_transform=inverse,
            zoom=float(self._zoom),
            device_pixel_ratio=max(1.0, float(self.devicePixelRatioF())),
        )

    def _visible_image_rect(self) -> QRectF:
        if self._image is None:
            return QRectF()
        return self._paint_context().image_rect

    def _measurement_index(self) -> MeasurementSceneIndex | None:
        if self._document is None:
            return None
        revision = (
            self._document.measurement_geometry_revision,
            self._document.state_stamp.session_state_id,
            len(self._document.measurements),
        )
        if self._measurement_hit_index is None or self._measurement_hit_index_revision != revision:
            self._measurement_hit_index = MeasurementSceneIndex(self._document.measurements)
            self._measurement_hit_index_revision = revision
        return self._measurement_hit_index

    def _measurement_candidates(self, image_point: Point, *, tolerance: float) -> list[Measurement]:
        index = self._measurement_index()
        if index is None:
            return []
        return index.query_point(image_point, tolerance=tolerance)

    def _measurement_display_scene_index(
        self,
        *,
        zoom: float,
    ) -> MeasurementSceneIndex | None:
        if self._document is None:
            return None
        stamp = self._document.state_stamp
        normalized_zoom = max(float(zoom), 0.001)
        signature = (
            id(self._document),
            stamp.session_state_id,
            stamp.calibration_state_id,
            self._document.measurement_geometry_revision,
            len(self._document.measurements),
            round(normalized_zoom, 8),
            self._canvas_visual_settings_signature,
        )
        if (
            self._measurement_display_index is not None
            and self._measurement_display_index_signature == signature
        ):
            return self._measurement_display_index

        raw_index = self._measurement_index()
        image_to_scaled_output = lambda point: QPointF(
            float(point.x) * normalized_zoom,
            float(point.y) * normalized_zoom,
        )
        bounds_by_id: dict[
            str,
            tuple[float, float, float, float],
        ] = {}
        for measurement in self._document.measurements:
            count_number = (
                raw_index.count_number(measurement.id)
                if raw_index is not None
                else None
            )
            bounds = measurement_display_image_bounds(
                measurement,
                self._document,
                self._settings,
                image_to_scaled_output,
                suggested_line_width=2.0,
                endpoint_radius=4.0,
                count_number=count_number,
            )
            if bounds is None or not bounds.isValid():
                continue
            bounds_by_id[measurement.id] = (
                bounds.left(),
                bounds.top(),
                bounds.right(),
                bounds.bottom(),
            )
        self._measurement_display_index = MeasurementSceneIndex(
            self._document.measurements,
            bounds_by_id=bounds_by_id,
        )
        self._measurement_display_index_signature = signature
        return self._measurement_display_index

    def _draw_annotations(
        self,
        painter: QPainter,
        paint_context: CanvasPaintContext | None = None,
    ) -> None:
        if self._document is None:
            return
        context = paint_context or self._paint_context()
        self._sync_overlay_visual_state()
        excluded_measurement_ids = self._actively_edited_measurement_ids()
        if self._overlay_cache_enabled():
            proxies_deferred = self._draw_measurement_overlay_tiles(painter, context)
        else:
            canvas_overlay_tile_cache.protect(id(self), ())
            selected_measurement = (
                self._document.get_measurement(
                    self._document.view_state.selected_measurement_id
                )
                if self._document.view_state.selected_measurement_id
                else None
            )
            area_uses_active_layer = (
                selected_measurement is not None
                and selected_measurement.measurement_kind == "area"
            )
            proxies_deferred = self._draw_measurements_direct(
                painter,
                image_rect=context.image_rect,
                image_to_output=self.image_to_widget,
                use_sprite_cache=True,
                excluded_measurement_ids=excluded_measurement_ids,
                render_selected_state=not area_uses_active_layer,
                raw_area_measurement_ids=(
                    frozenset((selected_measurement.id,))
                    if area_uses_active_layer
                    else frozenset()
                ),
            )
            if area_uses_active_layer:
                self._draw_selected_measurement_active_layer(painter, context)
        draw_overlay_annotations(
            painter,
            self._document,
            self.image_to_widget,
            self._settings,
            selected_overlay_id=self._document.selected_overlay_id,
            show_handles=self._tool_mode == "select",
            render_mode="screen_scale_full_image",
        )
        if proxies_deferred:
            self._schedule_proxy_warm(context.image_rect)
        else:
            self._proxy_warm_active_key = None

    def _measurement_render_inputs(
        self,
        image_rect: QRectF,
        *,
        zoom: float | None = None,
    ) -> tuple[list[Measurement], dict[str, int] | None]:
        effective_zoom = max(float(zoom or self._zoom), 0.001)
        index = self._measurement_display_scene_index(zoom=effective_zoom)
        measurements = (
            index.query_rect(image_rect)
            if index is not None
            else list(self._document.measurements)
        )
        selected_id = self._document.view_state.selected_measurement_id
        if (
            index is not None
            and selected_id
            and all(
                measurement.id != selected_id
                for measurement in measurements
            )
        ):
            selected_measurement = self._document.get_measurement(selected_id)
            if selected_measurement is not None:
                selected_bounds = measurement_display_image_bounds(
                    selected_measurement,
                    self._document,
                    self._settings,
                    lambda point: QPointF(
                        float(point.x) * effective_zoom,
                        float(point.y) * effective_zoom,
                    ),
                    suggested_line_width=2.0,
                    endpoint_radius=4.0,
                    count_number=index.count_number(selected_id),
                    selected=True,
                    exact_area_label=True,
                )
                if (
                    selected_bounds is not None
                    and selected_bounds.isValid()
                    and selected_bounds.right() >= image_rect.left()
                    and selected_bounds.left() <= image_rect.right()
                    and selected_bounds.bottom() >= image_rect.top()
                    and selected_bounds.top() <= image_rect.bottom()
                ):
                    measurements.append(selected_measurement)
                    measurements.sort(
                        key=lambda measurement: (
                            index.document_order(measurement.id)
                            if index.document_order(measurement.id) is not None
                            else len(self._document.measurements)
                        )
                    )
        count_numbers = (
            {
                measurement.id: number
                for measurement in measurements
                if (number := index.count_number(measurement.id)) is not None
            }
            if index is not None
            else None
        )
        return measurements, count_numbers

    @staticmethod
    def _max_object_font_size(
        measurements: list[Measurement],
    ) -> float:
        maximum = 0.0
        for measurement in measurements:
            appearance = measurement.appearance
            value = appearance.font_size if appearance is not None else None
            try:
                size = float(value) if value is not None else 0.0
            except (TypeError, ValueError):
                continue
            if math.isfinite(size):
                maximum = max(maximum, size)
        return maximum

    def _measurement_label_padding_screen(self) -> float:
        label_font_sizes = [
            float(
                getattr(
                    getattr(
                        self._settings,
                        "length_measurement_label_style",
                        None,
                    ),
                    "font_size",
                    14,
                )
            ),
            float(
                getattr(
                    getattr(
                        self._settings,
                        "area_measurement_label_style",
                        None,
                    ),
                    "font_size",
                    14,
                )
            ),
            float(getattr(self._settings, "count_number_font_size", 14)),
            float(self._overlay_max_object_font_size),
        ]
        maximum = max(
            (
                size
                for size in label_font_sizes
                if math.isfinite(size) and size > 0.0
            ),
            default=14.0,
        )
        # This is only the broad-phase query envelope. Exact text bounds still
        # perform the second-stage cull. A generous width factor is necessary
        # for object-level 96px+ fonts and long calibrated values that extend
        # into a neighbouring 512px tile.
        return max(64.0, (maximum * 8.0) + 32.0)

    def _draw_measurements_direct(
        self,
        painter: QPainter,
        *,
        image_rect: QRectF,
        image_to_output,
        use_sprite_cache: bool,
        excluded_measurement_ids: frozenset[str] = frozenset(),
        sprite_device_pixel_ratio: float | None = None,
        render_selected_state: bool = True,
        area_geometry_mode: str = AREA_GEOMETRY_SCREEN,
        raw_area_measurement_ids: frozenset[str] = frozenset(),
    ) -> bool:
        if self._document is None:
            return False
        measurements, count_numbers = self._measurement_render_inputs(image_rect)
        with area_derived_geometry_service.path_render_pass():
            return draw_measurements(
                painter,
                self._document,
                image_to_output,
                self._settings,
                line_width=2.0,
                endpoint_radius=4.0,
                selected_measurement_id=(
                    self._document.view_state.selected_measurement_id
                    if render_selected_state
                    else None
                ),
                show_area_fill=self._show_area_fill,
                show_area_handles=self._tool_mode == "select",
                visible_rect=image_rect,
                area_geometry_mode=area_geometry_mode,
                measurement_sequence=measurements,
                count_numbers=count_numbers,
                excluded_measurement_ids=excluded_measurement_ids,
                raw_area_measurement_ids=raw_area_measurement_ids,
                proxy_build_budget=AreaProxyBuildBudget(
                    max_builds=0,
                    max_build_ms=0.0,
                ),
                use_sprite_cache=use_sprite_cache,
                sprite_device_pixel_ratio=sprite_device_pixel_ratio,
                # ``_measurement_render_inputs()`` now queries an index built
                # from the complete display envelope (geometry, cosmetic
                # strokes, endpoint styles, markers and labels). Repeating the
                # same envelope calculation for every visible object costs a
                # measurable amount on the direct fallback path and cannot
                # reject any additional object.
                cull_by_geometry=False,
            )

    def _actively_edited_measurement_ids(self) -> frozenset[str]:
        """Objects whose committed body must not be painted under a preview."""

        if self._dragging_area_handle is not None:
            return frozenset((self._dragging_area_handle[0],))
        if self._dragging_handle is not None:
            return frozenset((self._dragging_handle[0],))
        return frozenset()

    def _overlay_cache_enabled(self) -> bool:
        if self._document is None:
            return False
        if os.environ.get("FDM_DISABLE_CANVAS_OVERLAY_CACHE", "").strip() == "1":
            return False
        force_enabled = (
            os.environ.get("FDM_ENABLE_CANVAS_OVERLAY_CACHE", "").strip() == "1"
        )
        if force_enabled:
            return True
        if os.environ.get("QT_QPA_PLATFORM", "").strip().lower() == "offscreen":
            return False
        # Object count alone badly underestimates standard-magic-wand output:
        # one object can retain more than a thousand exact ring vertices.
        # The cached passive pipeline therefore also turns on for geometrically
        # dense area documents, while small ordinary documents stay direct.
        return (
            len(self._document.measurements)
            >= OVERLAY_CACHE_MIN_MEASUREMENTS
            or self._overlay_area_vertex_count
            >= OVERLAY_CACHE_MIN_AREA_VERTICES
        )

    def _draw_measurement_overlay_tiles(
        self,
        painter: QPainter,
        context: CanvasPaintContext,
    ) -> bool:
        # A Qt paint event may cover only one dirty object or one completed
        # tile.  Use that narrow region solely for drawing, while request
        # admission/cancellation is governed by the complete viewport.  If
        # these sets are conflated, every tile-ready update shrinks the
        # "visible" set to one tile and cancels the next asynchronous build.
        paint_keys = self._visible_overlay_tile_keys(context)
        viewport_keys = self._visible_overlay_tile_keys(self._paint_context())
        working_keys = self._overlay_prefetch_tile_keys(viewport_keys)
        strict_visible = set(viewport_keys)
        newly_visible = strict_visible - self._overlay_strict_visible_keys
        self._overlay_tile_failed.difference_update(newly_visible)
        self._overlay_strict_visible_keys = strict_visible
        canvas_overlay_tile_cache.protect(id(self), viewport_keys)
        self._reconcile_overlay_visible_keys(working_keys)
        if not paint_keys:
            return False
        selected_area_ids: frozenset[str] = frozenset()
        if self._document is not None:
            selected_id = self._document.view_state.selected_measurement_id
            selected_measurement = (
                self._document.get_measurement(selected_id)
                if selected_id
                else None
            )
            if (
                selected_measurement is not None
                and selected_measurement.measurement_kind == "area"
            ):
                selected_area_ids = frozenset((selected_measurement.id,))
        cached: dict[
            CanvasOverlayTileKey,
            tuple[QImage | None, QPicture | None],
        ] = {}
        missing: list[CanvasOverlayTileKey] = []
        for key in paint_keys:
            payload = canvas_overlay_tile_cache.get_payload(key)
            if payload is None:
                missing.append(key)
            else:
                cached[key] = payload
        proxies_deferred = False
        if len(missing) == len(paint_keys):
            proxies_deferred = self._draw_measurements_direct(
                painter,
                image_rect=context.image_rect,
                image_to_output=self.image_to_widget,
                use_sprite_cache=True,
                render_selected_state=False,
                raw_area_measurement_ids=selected_area_ids,
            )
        else:
            for key in missing:
                target = self._overlay_tile_widget_rect(key)
                painter.save()
                painter.setClipRect(target)
                try:
                    proxies_deferred = (
                        self._draw_measurements_direct(
                            painter,
                            image_rect=self._overlay_tile_image_rect(key),
                            image_to_output=self.image_to_widget,
                            use_sprite_cache=True,
                            render_selected_state=False,
                            raw_area_measurement_ids=selected_area_ids,
                        )
                        or proxies_deferred
                    )
                finally:
                    painter.restore()
        for key, (image, picture) in cached.items():
            if image is not None and (
                self._overlay_motion_active()
                or picture is None
            ):
                # The raster already has this exact DPR and subpixel phase;
                # point placement avoids a second scaling/filtering pass.
                painter.drawImage(
                    self._overlay_tile_widget_rect(key).topLeft(),
                    image,
                )
                continue
            if picture is None:  # pragma: no cover - cache envelope guard
                continue
            target = self._overlay_tile_raw_widget_rect(key)
            painter.save()
            painter.setClipRect(target)
            origin = self._overlay_widget_origin()
            painter.translate(float(origin.x()), float(origin.y()))
            try:
                picture.play(painter)
            finally:
                painter.restore()
        self._redraw_selected_measurement_background(painter, context)
        self._draw_selected_measurement_active_layer(painter, context)
        # Queue the complete visible set plus a byte-bounded one-tile guard
        # ring. Physical-pixel-aligned dragging keeps this generation stable,
        # so a recently committed magic-wand object can finish warming while
        # the user pans and an edge tile is normally ready before it enters the
        # viewport. Cached and pending keys are filtered by the controller.
        self._enqueue_overlay_tiles(working_keys)
        return proxies_deferred

    def _redraw_selected_measurement_background(
        self,
        painter: QPainter,
        context: CanvasPaintContext,
    ) -> None:
        """Remove the selected object's passive body from cached tiles.

        Passive tiles deliberately remain independent of selection so a click
        never invalidates and synchronously replays a dense tile. Repaint the
        selected object's small exact display envelope from the source image,
        restore intersecting passive objects in document order, then let the
        active layer draw the selected RAW object. Drag previews therefore get
        a clean background without a stale committed-body ghost.
        """

        if self._document is None or self._image is None:
            return
        selected_id = self._document.view_state.selected_measurement_id
        if not selected_id:
            return
        measurement = self._document.get_measurement(selected_id)
        if measurement is None:
            return
        actively_edited = selected_id in self._actively_edited_measurement_ids()
        if measurement.measurement_kind == "area" and not actively_edited:
            # The passive tile already contains the ordinary area body and
            # label.  Selection is represented by an active emphasis layer, so
            # there is nothing to erase until an edit preview moves/replaces
            # the committed geometry.  This is the ImageJ-style fast path that
            # keeps a click independent of neighbouring high-vertex areas.
            return
        bounds = (
            self._area_drag_display_bounds(measurement, Point(0.0, 0.0))
            if measurement.measurement_kind == "area" and actively_edited
            else self._selection_display_bounds(
                CanvasSelectionRef.measurement(selected_id)
            )
        )
        if bounds is None:
            return
        image_rect = bounds.image_rect
        if image_rect.isEmpty():
            return
        widget_rect = (
            context.image_to_widget_transform.mapRect(image_rect)
            .adjusted(-3.0, -3.0, 3.0, 3.0)
            .intersected(context.widget_rect)
            .intersected(QRectF(self.rect()))
        )
        if widget_rect.isEmpty():
            return

        target = QRectF(
            self._pan.x,
            self._pan.y,
            self._image.width() * self._zoom,
            self._image.height() * self._zoom,
        )
        painter.save()
        try:
            painter.setClipRect(widget_rect)
            painter.fillRect(
                widget_rect,
                canvas_workspace_background(self.palette()),
            )
            painter.drawImage(target, self._image)
            border_pen = QPen(canvas_image_border(self.palette()))
            border_pen.setWidthF(1.0)
            painter.setPen(border_pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(target)
            self._draw_measurements_direct(
                painter,
                image_rect=image_rect,
                image_to_output=self.image_to_widget,
                use_sprite_cache=True,
                excluded_measurement_ids=frozenset((selected_id,)),
                render_selected_state=False,
                area_geometry_mode=AREA_GEOMETRY_RAW,
            )
        finally:
            painter.restore()

    def _draw_selected_measurement_active_layer(
        self,
        painter: QPainter,
        context: CanvasPaintContext,
    ) -> None:
        """Draw the selected object exactly, after every passive tile."""

        if self._document is None:
            return
        selected_id = self._document.view_state.selected_measurement_id
        if (
            not selected_id
            or selected_id in self._actively_edited_measurement_ids()
        ):
            return
        measurement = self._document.get_measurement(selected_id)
        if measurement is None:
            return
        selection_bounds = self._selection_display_bounds(
            CanvasSelectionRef.measurement(selected_id)
        )
        if selection_bounds is None:
            return
        if not context.image_to_widget_transform.mapRect(
            selection_bounds.image_rect
        ).intersects(context.widget_rect):
            return
        if measurement.measurement_kind == "area":
            with area_derived_geometry_service.path_render_pass():
                draw_area_measurement(
                    painter,
                    self._document,
                    measurement,
                    self.image_to_widget,
                    self._settings,
                    line_width=2.0,
                    endpoint_radius=4.0,
                    selected=True,
                    # The ordinary RAW fill and result label already belong to
                    # the passive layer. Selection adds only opaque outline
                    # emphasis and controls, so it cannot darken the label or
                    # reorder translucent overlaps.
                    show_fill=False,
                    show_handles=self._tool_mode == "select",
                    geometry_mode=AREA_GEOMETRY_RAW,
                    proxy_build_budget=AreaProxyBuildBudget(
                        max_builds=0,
                        max_build_ms=0.0,
                    ),
                    use_sprite_cache=True,
                    show_label=False,
                )
            return
        index = self._measurement_index()
        count_numbers = None
        if index is not None:
            number = index.count_number(measurement.id)
            if number is not None:
                count_numbers = {measurement.id: number}
        with area_derived_geometry_service.path_render_pass():
            draw_measurements(
                painter,
                self._document,
                self.image_to_widget,
                self._settings,
                line_width=2.0,
                endpoint_radius=4.0,
                selected_measurement_id=selected_id,
                show_area_fill=self._show_area_fill,
                show_area_handles=self._tool_mode == "select",
                visible_rect=context.image_rect,
                measurement_sequence=(measurement,),
                count_numbers=count_numbers,
                proxy_build_budget=AreaProxyBuildBudget(
                    max_builds=0,
                    max_build_ms=0.0,
                ),
                use_sprite_cache=True,
            )

    def _visible_overlay_tile_keys(
        self,
        context: CanvasPaintContext,
    ) -> list[CanvasOverlayTileKey]:
        if self._document is None:
            return []
        zoom = round(float(context.zoom), 8)
        dpr = round(float(context.device_pixel_ratio), 4)
        self._remember_overlay_namespace(zoom, dpr)
        overlay_origin = self._overlay_widget_origin()
        overlay_rect = context.widget_rect.translated(
            -float(overlay_origin.x()),
            -float(overlay_origin.y()),
        )
        tile_size = float(OVERLAY_TILE_LOGICAL_SIZE)
        min_x = math.floor(overlay_rect.left() / tile_size)
        max_x = math.floor(
            (overlay_rect.right() - 1e-9) / tile_size
        )
        min_y = math.floor(overlay_rect.top() / tile_size)
        max_y = math.floor(
            (overlay_rect.bottom() - 1e-9) / tile_size
        )
        keys: list[CanvasOverlayTileKey] = []
        for tile_y in range(min_y, max_y + 1):
            for tile_x in range(min_x, max_x + 1):
                keys.append(
                    self._overlay_tile_key(
                        tile_x,
                        tile_y,
                        zoom=zoom,
                        dpr=dpr,
                        overlay_origin=overlay_origin,
                    )
                )
        # Keep the compatibility side effect for callers requesting the whole
        # viewport, but never let a local dirty-region query shrink the
        # lifecycle set used for cancellation and late-result validation.
        if context.widget_rect == QRectF(self.rect()):
            self._overlay_visible_keys = set(keys)
        return keys

    def _overlay_tile_key(
        self,
        tile_x: int,
        tile_y: int,
        *,
        zoom: float,
        dpr: float,
        overlay_origin: QPointF | None = None,
    ) -> CanvasOverlayTileKey:
        if self._document is None:  # pragma: no cover - guarded by callers
            raise RuntimeError("overlay tile key requires an active document")
        origin = overlay_origin or self._overlay_widget_origin()
        tile_size = float(OVERLAY_TILE_LOGICAL_SIZE)
        epoch_key = (zoom, dpr, int(tile_x), int(tile_y))
        device_start_x = (
            float(origin.x()) + (int(tile_x) * tile_size)
        ) * dpr
        device_start_y = (
            float(origin.y()) + (int(tile_y) * tile_size)
        ) * dpr
        phase_x = device_start_x - math.floor(device_start_x)
        phase_y = device_start_y - math.floor(device_start_y)
        phase_x = 0.0 if phase_x > 1.0 - 1e-8 else round(phase_x, 8)
        phase_y = 0.0 if phase_y > 1.0 - 1e-8 else round(phase_y, 8)
        return CanvasOverlayTileKey(
            document_token=id(self._document),
            document_id=self._document.id,
            zoom=zoom,
            device_pixel_ratio=dpr,
            tile_x=int(tile_x),
            tile_y=int(tile_y),
            style_generation=self._overlay_style_generation,
            tile_epoch=self._overlay_tile_epochs.get(epoch_key, 0),
            show_area_fill=self._show_area_fill,
            device_phase_x=phase_x,
            device_phase_y=phase_y,
        )

    def _overlay_prefetch_tile_keys(
        self,
        visible_keys: list[CanvasOverlayTileKey],
    ) -> list[CanvasOverlayTileKey]:
        """Return visible tiles followed by a byte-bounded guard ring."""

        if not visible_keys or self._document is None:
            return list(visible_keys)
        anchor = visible_keys[0]
        dpr = max(1.0, float(anchor.device_pixel_ratio))
        physical_edge = max(
            1,
            int(math.ceil(float(OVERLAY_TILE_LOGICAL_SIZE) * dpr)),
        )
        estimated_raster_bytes = physical_edge * physical_edge * 4
        cache_byte_budget = int(
            getattr(
                canvas_overlay_tile_cache,
                "max_bytes",
                OVERLAY_TILE_MAX_BYTES,
            )
        )
        cache_entry_budget = int(
            getattr(
                canvas_overlay_tile_cache,
                "max_entries",
                OVERLAY_TILE_MAX_ENTRIES,
            )
        )
        prefetch_byte_budget = max(
            estimated_raster_bytes,
            cache_byte_budget // 2,
        )
        maximum_tiles = min(
            cache_entry_budget,
            max(
                len(visible_keys),
                prefetch_byte_budget // max(1, estimated_raster_bytes),
            ),
        )
        if maximum_tiles <= len(visible_keys):
            return list(visible_keys)

        min_x = min(key.tile_x for key in visible_keys)
        max_x = max(key.tile_x for key in visible_keys)
        min_y = min(key.tile_y for key in visible_keys)
        max_y = max(key.tile_y for key in visible_keys)
        visible_coordinates = {
            (key.tile_x, key.tile_y)
            for key in visible_keys
        }
        center_x = (min_x + max_x) / 2.0
        center_y = (min_y + max_y) / 2.0
        guard_coordinates = [
            (tile_x, tile_y)
            for tile_y in range(min_y - 1, max_y + 2)
            for tile_x in range(min_x - 1, max_x + 2)
            if (tile_x, tile_y) not in visible_coordinates
        ]
        # Prefer the cardinal neighbours nearest the viewport center when high
        # DPR limits how many guard tiles fit into half of the global budget.
        guard_coordinates.sort(
            key=lambda coordinate: (
                abs(coordinate[0] - center_x)
                + abs(coordinate[1] - center_y),
                coordinate[1],
                coordinate[0],
            )
        )
        remaining = maximum_tiles - len(visible_keys)
        overlay_origin = self._overlay_widget_origin()
        display_index = self._measurement_display_scene_index(
            zoom=float(anchor.zoom)
        )
        guard_keys: list[CanvasOverlayTileKey] = []
        for tile_x, tile_y in guard_coordinates:
            guard_key = self._overlay_tile_key(
                tile_x,
                tile_y,
                zoom=float(anchor.zoom),
                dpr=dpr,
                overlay_origin=overlay_origin,
            )
            tile_image_rect = self._overlay_tile_image_rect(guard_key)
            if not tile_image_rect.intersects(self._paint_image_bounds()):
                continue
            if (
                display_index is not None
                and not display_index.query_rect(tile_image_rect)
            ):
                continue
            guard_keys.append(guard_key)
            if len(guard_keys) >= remaining:
                break
        return [*visible_keys, *guard_keys]

    def _remember_overlay_namespace(self, zoom: float, dpr: float) -> None:
        namespace = (zoom, dpr)
        if namespace in self._overlay_known_namespaces:
            if namespace in self._overlay_namespace_order:
                self._overlay_namespace_order.remove(namespace)
            self._overlay_namespace_order.append(namespace)
            return
        self._overlay_known_namespaces.add(namespace)
        self._overlay_namespace_order.append(namespace)
        while len(self._overlay_namespace_order) > 8:
            stale = self._overlay_namespace_order.pop(0)
            self._overlay_known_namespaces.discard(stale)
            if self._document is not None:
                canvas_overlay_tile_cache.invalidate_namespace(
                    id(self._document),
                    stale[0],
                    stale[1],
                )
            self._overlay_tile_epochs = {
                key: epoch
                for key, epoch in self._overlay_tile_epochs.items()
                if key[:2] != stale
            }
            self._overlay_tile_failed = {
                key
                for key in self._overlay_tile_failed
                if (key.zoom, key.device_pixel_ratio) != stale
            }

    def _overlay_tile_widget_rect(self, key: CanvasOverlayTileKey) -> QRectF:
        size = float(OVERLAY_TILE_LOGICAL_SIZE)
        overlay_origin = self._overlay_widget_origin()
        dpr = max(float(key.device_pixel_ratio), 1e-9)
        raw_x = float(overlay_origin.x()) + (key.tile_x * size)
        raw_y = float(overlay_origin.y()) + (key.tile_y * size)
        return QRectF(
            math.floor(raw_x * dpr) / dpr,
            math.floor(raw_y * dpr) / dpr,
            size,
            size,
        )

    def _overlay_tile_raw_widget_rect(self, key: CanvasOverlayTileKey) -> QRectF:
        size = float(OVERLAY_TILE_LOGICAL_SIZE)
        overlay_origin = self._overlay_widget_origin()
        return QRectF(
            float(overlay_origin.x()) + (key.tile_x * size),
            float(overlay_origin.y()) + (key.tile_y * size),
            size,
            size,
        )

    @staticmethod
    def _overlay_tile_image_rect(key: CanvasOverlayTileKey) -> QRectF:
        size = float(OVERLAY_TILE_LOGICAL_SIZE)
        zoom = max(float(key.zoom), 1e-9)
        return QRectF(
            (key.tile_x * size) / zoom,
            (key.tile_y * size) / zoom,
            size / zoom,
            size / zoom,
        )

    def _enqueue_overlay_tiles(self, keys: list[CanvasOverlayTileKey]) -> None:
        if self._document is None:
            return
        self._reconcile_overlay_visible_keys(keys)
        reordered_queue: list[CanvasOverlayTileKey] = []
        for key in keys:
            if (
                key == self._overlay_tile_active
                or key in self._overlay_tile_failed
                or canvas_overlay_tile_cache.contains(key)
                or canvas_overlay_tile_cache.is_pending(key)
            ):
                continue
            reordered_queue.append(key)
        self._overlay_tile_queue = reordered_queue
        self._overlay_tile_queued = set(reordered_queue)
        if (
            self._overlay_tile_queue
            and self._overlay_tile_active is None
            and not self._overlay_tile_build_scheduled
        ):
            self._overlay_tile_build_scheduled = True
            QTimer.singleShot(0, self._start_next_overlay_tile)

    def _reconcile_overlay_visible_keys(
        self,
        keys: list[CanvasOverlayTileKey],
    ) -> None:
        current = set(keys)
        self._overlay_visible_keys = current
        if (
            self._overlay_tile_active is not None
            and self._overlay_tile_active not in current
        ):
            canvas_overlay_tile_cache.cancel(self._overlay_tile_active)
            self._overlay_tile_active = None
        self._overlay_tile_queue = [
            key for key in self._overlay_tile_queue if key in current
        ]
        self._overlay_tile_queued = set(self._overlay_tile_queue)
        # A failed tile is suppressed only while it remains in the current
        # viewport. Leaving and returning gives it one fresh bounded retry,
        # while this set can never grow with hours of navigation.
        self._overlay_tile_failed.intersection_update(current)

    def _start_next_overlay_tile(self) -> None:
        self._overlay_tile_build_scheduled = False
        if self._document is None or self._overlay_tile_active is not None:
            return
        while self._overlay_tile_queue:
            key = self._overlay_tile_queue.pop(0)
            self._overlay_tile_queued.discard(key)
            if not self._overlay_tile_key_is_current(key):
                continue
            if key in self._overlay_tile_failed or canvas_overlay_tile_cache.contains(key):
                continue
            if canvas_overlay_tile_cache.is_pending(key):
                continue
            snapshot = self._build_overlay_tile_snapshot(key)
            if snapshot is None:
                self._overlay_tile_failed.add(key)
                continue
            if canvas_overlay_tile_cache.request(snapshot):
                self._overlay_tile_active = key
                return
            # Admission can be refused by the global pending-byte budget.
            # Keep drawing this tile through the exact direct path and suppress
            # repeated snapshot construction for the same epoch.
            self._overlay_tile_failed.add(key)

    def _build_overlay_tile_snapshot(
        self,
        key: CanvasOverlayTileKey,
    ) -> CanvasOverlayRenderSnapshot | None:
        if self._document is None or not self._overlay_tile_key_is_current(key):
            return None
        image_rect = self._overlay_tile_image_rect(key)
        candidate_measurements, count_numbers = self._measurement_render_inputs(
            image_rect,
            zoom=key.zoom,
        )
        excluded: frozenset[str] = frozenset()
        image_to_overlay = lambda point: QPointF(
            float(point.x) * key.zoom,
            float(point.y) * key.zoom,
        )
        passive_candidates = [
            measurement
            for measurement in candidate_measurements
            if measurement.id not in excluded
            and measurement_display_intersects_rect(
                measurement,
                self._document,
                self._settings,
                image_to_overlay,
                image_rect,
                padding=measurement_geometry_cull_padding(
                    image_to_overlay,
                    endpoint_radius=4.0,
                ),
                suggested_line_width=2.0,
                endpoint_radius=4.0,
                count_number=(
                    count_numbers.get(measurement.id)
                    if count_numbers is not None
                    else None
                ),
            )
        ]
        if not passive_candidates:
            self._overlay_tile_request_serial += 1
            return CanvasOverlayRenderSnapshot(
                request_id=self._overlay_tile_request_serial,
                key=key,
                known_empty=True,
            )
        if (
            passive_candidates
            and all(
                measurement.measurement_kind == "area"
                for measurement in passive_candidates
            )
        ):
            commands = []
            with area_derived_geometry_service.path_render_pass():
                for measurement in passive_candidates:
                    command = build_passive_area_overlay_command(
                        self._document,
                        measurement,
                        self._settings,
                        zoom=key.zoom,
                        line_width=2.0,
                        show_fill=self._show_area_fill,
                        sprite_device_pixel_ratio=key.device_pixel_ratio,
                    )
                    if command is None:
                        commands.clear()
                        break
                    commands.append(command)
            if commands:
                self._overlay_tile_request_serial += 1
                return CanvasOverlayRenderSnapshot(
                    request_id=self._overlay_tile_request_serial,
                    key=key,
                    area_commands=tuple(commands),
                    bleed_device_pixels=2,
                    exact_composition=False,
                    # Preserve the current worker-side composition guard.  If a
                    # label overlaps its translucent area body, the worker can
                    # record these same detached commands to QPicture without
                    # ever serializing the large path on the UI thread.
                    # One isolated area cannot interact with another passive
                    # object. With two or more objects, however, result labels
                    # may overlap even when the area bodies do not. Always run
                    # the opaque-background equivalence probe for that case;
                    # a numeric object-count shortcut would make dense tiles
                    # faster at the cost of incorrect alpha composition.
                    adaptive_composition=len(passive_candidates) > 1,
                    composition_probe_rgba=0xFFFFFFFF,
                )

        # Mixed object kinds and defensive area-command failures retain the
        # UI-recorded QPicture input. Area-heavy pictures are still flattened
        # adaptively in the worker when the opaque-background probe proves the
        # raster visually equivalent.
        picture = QPicture()
        picture_painter = QPainter(picture)
        if not picture_painter.isActive():
            return None
        tile_size = float(OVERLAY_TILE_LOGICAL_SIZE)
        bleed_device_pixels = 2
        bleed = bleed_device_pixels / max(key.device_pixel_ratio, 1.0)
        overlay_rect = QRectF(
            key.tile_x * tile_size,
            key.tile_y * tile_size,
            tile_size,
            tile_size,
        )
        picture_painter.setClipRect(
            overlay_rect.adjusted(-bleed, -bleed, bleed, bleed)
        )
        picture_painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        picture_painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        try:
            self._draw_measurements_direct(
                picture_painter,
                image_rect=image_rect,
                image_to_output=image_to_overlay,
                # Record complete label sprites at the tile's target DPR.
                # The worker then replays one drawImage per label without
                # accessing QWidget state or resampling a DPR=1 sprite.
                use_sprite_cache=True,
                excluded_measurement_ids=frozenset(),
                sprite_device_pixel_ratio=key.device_pixel_ratio,
                render_selected_state=False,
                area_geometry_mode=AREA_GEOMETRY_RAW,
            )
        finally:
            picture_painter.end()
        self._overlay_tile_request_serial += 1
        contains_area = any(
            measurement.measurement_kind == "area"
            for measurement in passive_candidates
        )
        adaptive_picture_tile = (
            contains_area or len(passive_candidates) > 64
        )
        return CanvasOverlayRenderSnapshot(
            request_id=self._overlay_tile_request_serial,
            key=key,
            picture=QPicture(picture),
            bleed_device_pixels=bleed_device_pixels,
            # Dense line/polyline/count and mixed area tiles benefit materially
            # from one flattened image. The worker probes composition on an
            # opaque background and retains the exact command stream whenever
            # alpha rounding would become visible. Small non-area tiles remain
            # exact because replaying them is already cheap.
            exact_composition=not adaptive_picture_tile,
            adaptive_composition=adaptive_picture_tile,
            # White maximizes the observable error from flattening
            # semi-transparent antialias/background layers.  Tiles that fail
            # this probe retain exact commands instead of a raster.
            composition_probe_rgba=0xFFFFFFFF,
        )

    def _overlay_tile_key_is_current(self, key: CanvasOverlayTileKey) -> bool:
        if self._document is None:
            return False
        if key not in self._overlay_visible_keys:
            return False
        if (
            key.document_token != id(self._document)
            or key.document_id != self._document.id
            or key.style_generation != self._overlay_style_generation
            or key.show_area_fill != self._show_area_fill
        ):
            return False
        epoch_key = (
            key.zoom,
            key.device_pixel_ratio,
            key.tile_x,
            key.tile_y,
        )
        return key.tile_epoch == self._overlay_tile_epochs.get(epoch_key, 0)

    def _on_overlay_tile_ready(self, key: CanvasOverlayTileKey) -> None:
        if self._overlay_tile_active == key:
            self._overlay_tile_active = None
        self._overlay_tile_queued.discard(key)
        if self._overlay_tile_key_is_current(key):
            current_zoom = round(float(self._zoom), 8)
            current_dpr = round(max(1.0, float(self.devicePixelRatioF())), 4)
            if key.zoom == current_zoom and key.device_pixel_ratio == current_dpr:
                self.update(
                    self._overlay_tile_widget_rect(key)
                    .adjusted(-2.0, -2.0, 2.0, 2.0)
                    .toAlignedRect()
                    .intersected(self.rect())
                )
        self._start_next_overlay_tile()

    def _on_overlay_tile_failed(
        self,
        key: CanvasOverlayTileKey,
        _message: str,
    ) -> None:
        if self._overlay_tile_active == key:
            self._overlay_tile_active = None
        if self._overlay_tile_key_is_current(key):
            # Keep rendering exact direct vectors, but do not retry a backend
            # failure on every paint for the same tile epoch.
            self._overlay_tile_failed.add(key)
        self._start_next_overlay_tile()

    def _cancel_overlay_requests(self) -> None:
        if self._overlay_tile_active is not None:
            canvas_overlay_tile_cache.cancel(self._overlay_tile_active)
        self._overlay_tile_active = None
        self._overlay_tile_queue.clear()
        self._overlay_tile_queued.clear()
        self._overlay_tile_build_scheduled = False

    def _reset_overlay_tracking(self, *, invalidate_document: bool) -> None:
        if invalidate_document and self._document is not None:
            canvas_overlay_tile_cache.invalidate_document(id(self._document))
        self._cancel_overlay_requests()
        self._overlay_tile_epochs.clear()
        self._overlay_known_namespaces.clear()
        self._overlay_namespace_order.clear()
        self._overlay_visible_keys.clear()
        self._overlay_strict_visible_keys.clear()
        self._overlay_tile_failed.clear()
        self._overlay_document_stamp = None
        self._overlay_group_signature = None
        self._overlay_calibration_signature = None
        self._overlay_measurement_order_signature = None
        self._overlay_area_vertex_count = 0
        self._overlay_measurement_state.clear()
        self._overlay_annotation_state.clear()
        self._overlay_selected_measurement_id = (
            self._document.view_state.selected_measurement_id
            if self._document is not None
            else None
        )

    def _invalidate_all_overlay_tiles(self) -> None:
        if self._document is not None:
            canvas_overlay_tile_cache.invalidate_document(id(self._document))
        self._cancel_overlay_requests()
        self._overlay_tile_epochs.clear()
        self._overlay_tile_failed.clear()

    def _sync_overlay_visual_state(
        self,
    ) -> tuple[bool, tuple[CanvasDisplayBounds, ...]]:
        if self._document is None:
            return False, ()
        stamp = self._document.state_stamp
        group_signature = tuple(
            (group.id, group.label, group.color, group.number)
            for group in self._document.fiber_groups
        )
        document_stamp = (
            stamp.session_state_id,
            stamp.calibration_state_id,
            self._document.measurement_geometry_revision,
            len(self._document.measurements),
        )
        calibration_signature = self._document.calibration_signature()
        # Drawing a hot cached frame must not scan thousands of offscreen
        # objects merely to rediscover an unchanged order. Every production
        # mutation that can reorder measurements advances the document stamp;
        # only rebuild the exact order signature in that generation.
        order_signature = (
            tuple(measurement.id for measurement in self._document.measurements)
            if self._overlay_document_stamp != document_stamp
            else self._overlay_measurement_order_signature
        )
        order_changed = (
            self._overlay_measurement_order_signature is not None
            and self._overlay_measurement_order_signature != order_signature
        )
        pure_reorder = (
            order_changed
            and len(self._overlay_measurement_order_signature or ()) == len(order_signature)
            and set(self._overlay_measurement_order_signature or ()) == set(order_signature)
        )
        full_visual_change = (
            self._overlay_group_signature not in (None, group_signature)
            or self._overlay_calibration_signature
            not in (None, calibration_signature)
            or pure_reorder
        )
        if full_visual_change:
            self._invalidate_all_overlay_tiles()
            self._overlay_measurement_state.clear()
            self._overlay_annotation_state.clear()
            self._overlay_document_stamp = None
        dirty_display_bounds: list[CanvasDisplayBounds] = []
        if self._overlay_document_stamp != document_stamp:
            previous_max_object_font_size = (
                self._overlay_max_object_font_size
            )
            current: dict[
                str,
                tuple[tuple[object, ...], tuple[float, float, float, float] | None],
            ] = {}
            changed_bounds: list[tuple[float, float, float, float]] = []
            count_number = 0
            max_object_font_size = 0.0
            area_vertex_count = 0
            for measurement in self._document.measurements:
                if measurement.measurement_kind == "area":
                    rings = measurement.area_rings_px
                    area_vertex_count += sum(
                        len(ring)
                        for ring in (
                            rings
                            if rings
                            else (
                                [measurement.polygon_px]
                                if measurement.polygon_px
                                else []
                            )
                        )
                    )
                if measurement.measurement_kind == "count":
                    count_number += 1
                appearance = measurement.appearance
                if appearance is not None and appearance.font_size is not None:
                    try:
                        font_size = float(appearance.font_size)
                    except (TypeError, ValueError):
                        font_size = 0.0
                    if math.isfinite(font_size):
                        max_object_font_size = max(
                            max_object_font_size,
                            font_size,
                        )
                bounds = MeasurementSceneIndex._measurement_bounds(measurement)
                fingerprint = self._measurement_visual_fingerprint(
                    measurement,
                    count_number=(
                        count_number
                        if measurement.measurement_kind == "count"
                        else None
                    ),
                )
                current[measurement.id] = (fingerprint, bounds)
                previous = self._overlay_measurement_state.get(measurement.id)
                if previous is None or previous[0] != fingerprint:
                    if previous is not None and previous[1] is not None:
                        changed_bounds.append(previous[1])
                    if bounds is not None:
                        changed_bounds.append(bounds)
            for measurement_id, (_fingerprint, bounds) in self._overlay_measurement_state.items():
                if measurement_id not in current and bounds is not None:
                    changed_bounds.append(bounds)
            current_annotations: dict[
                str,
                tuple[
                    tuple[object, ...],
                    tuple[float, float, float, float] | None,
                ],
            ] = {}
            for annotation in self._document.overlay_annotations:
                annotation_bounds = self._overlay_annotation_display_bounds(
                    annotation
                )
                annotation_fingerprint = (
                    annotation.normalized_kind(),
                    annotation.content,
                    annotation.anchor_px.x,
                    annotation.anchor_px.y,
                    annotation.start_px.x,
                    annotation.start_px.y,
                    annotation.end_px.x,
                    annotation.end_px.y,
                    repr(annotation.appearance),
                    repr(annotation.text_layout),
                )
                current_annotations[annotation.id] = (
                    annotation_fingerprint,
                    annotation_bounds,
                )
                previous_annotation = self._overlay_annotation_state.get(
                    annotation.id
                )
                if (
                    previous_annotation is None
                    or previous_annotation[0] != annotation_fingerprint
                ):
                    if (
                        previous_annotation is not None
                        and previous_annotation[1] is not None
                    ):
                        changed_bounds.append(previous_annotation[1])
                    if annotation_bounds is not None:
                        changed_bounds.append(annotation_bounds)
            for annotation_id, (_fingerprint, bounds) in (
                self._overlay_annotation_state.items()
            ):
                if annotation_id not in current_annotations and bounds is not None:
                    changed_bounds.append(bounds)
            self._overlay_max_object_font_size = max(
                previous_max_object_font_size,
                max_object_font_size,
            )
            if self._overlay_measurement_state:
                if len(changed_bounds) > 96:
                    self._invalidate_all_overlay_tiles()
                    full_visual_change = True
                else:
                    self._invalidate_overlay_bounds(changed_bounds)
            if not full_visual_change:
                display_padding = (
                    self._measurement_label_padding_screen()
                    / max(self._zoom, 0.001)
                )
                dirty_display_bounds = [
                    CanvasDisplayBounds(
                        QRectF(
                            left,
                            top,
                            max(1e-6, right - left),
                            max(1e-6, bottom - top),
                        )
                    ).expanded(display_padding)
                    for left, top, right, bottom in changed_bounds
                ]
            self._overlay_measurement_state = current
            self._overlay_annotation_state = current_annotations
            self._overlay_max_object_font_size = max_object_font_size
            self._overlay_area_vertex_count = area_vertex_count
            self._overlay_document_stamp = document_stamp
        selected_id = self._document.view_state.selected_measurement_id
        if selected_id != self._overlay_selected_measurement_id:
            # Selection is an active-layer concern. The passive cache keeps
            # the ordinary object body and a small exact background patch
            # removes it before the selected RAW rendering is composed.
            self._overlay_selected_measurement_id = selected_id
        self._overlay_group_signature = group_signature
        self._overlay_calibration_signature = calibration_signature
        self._overlay_measurement_order_signature = order_signature
        return full_visual_change, tuple(dirty_display_bounds)

    def _overlay_annotation_display_bounds(
        self,
        annotation: OverlayAnnotation,
    ) -> tuple[float, float, float, float] | None:
        if annotation.normalized_kind() == OverlayAnnotationKind.TEXT:
            widget_rect = annotation_rect(
                annotation,
                self._settings,
                self.image_to_widget,
            ).adjusted(-12.0, -12.0, 12.0, 12.0)
            image_rect = self._paint_context().widget_to_image_transform.mapRect(
                widget_rect
            )
        else:
            left, top, right, bottom = overlay_annotation_bounds(annotation)
            image_rect = QRectF(
                left,
                top,
                max(1e-6, right - left),
                max(1e-6, bottom - top),
            ).adjusted(
                -24.0 / max(self._zoom, 0.001),
                -24.0 / max(self._zoom, 0.001),
                24.0 / max(self._zoom, 0.001),
                24.0 / max(self._zoom, 0.001),
            )
        if not image_rect.isValid():
            return None
        return (
            image_rect.left(),
            image_rect.top(),
            image_rect.right(),
            image_rect.bottom(),
        )

    @staticmethod
    def _measurement_visual_fingerprint(
        measurement: Measurement,
        *,
        count_number: int | None = None,
    ) -> tuple[object, ...]:
        appearance = measurement.appearance
        return (
            id(measurement),
            measurement.geometry_revision,
            measurement.measurement_kind,
            measurement.mode,
            measurement.fiber_group_id,
            measurement.diameter_px,
            measurement.diameter_unit,
            measurement.area_px,
            measurement.area_unit,
            measurement.exact_area_px,
            measurement.confidence,
            measurement.status,
            repr(appearance),
            count_number if measurement.measurement_kind == "count" else None,
        )

    def _invalidate_overlay_bounds(
        self,
        bounds_list: list[tuple[float, float, float, float]],
    ) -> None:
        if self._document is None or not bounds_list:
            return
        coordinates: set[tuple[float, float, int, int]] = set()
        tile_size = float(OVERLAY_TILE_LOGICAL_SIZE)
        for zoom, dpr in self._overlay_known_namespaces:
            label_padding = (
                self._measurement_label_padding_screen()
                / max(zoom, 0.001)
            )
            for left, top, right, bottom in bounds_list:
                min_x = math.floor(((left - label_padding) * zoom) / tile_size)
                max_x = math.floor(((right + label_padding) * zoom) / tile_size)
                min_y = math.floor(((top - label_padding) * zoom) / tile_size)
                max_y = math.floor(((bottom + label_padding) * zoom) / tile_size)
                for tile_y in range(min_y, max_y + 1):
                    for tile_x in range(min_x, max_x + 1):
                        epoch_key = (zoom, dpr, tile_x, tile_y)
                        self._overlay_tile_epochs[epoch_key] = (
                            self._overlay_tile_epochs.get(epoch_key, 0) + 1
                        )
                        coordinates.add((zoom, dpr, tile_x, tile_y))
        if not coordinates:
            return
        canvas_overlay_tile_cache.invalidate_coordinates(
            id(self._document),
            coordinates,
        )
        self._cancel_overlay_requests()
        self._overlay_tile_failed.clear()

    def _proxy_warm_key(self) -> tuple[object, ...] | None:
        if self._document is None:
            return None
        return (
            id(self._document),
            self._document.id,
            self._document.measurement_geometry_revision,
            round(float(self._zoom), 8),
            self._document.view_state.selected_measurement_id,
        )

    @staticmethod
    def _area_cache_generation() -> int:
        generation = getattr(
            area_derived_geometry_service,
            "path_cache_generation",
            0,
        )
        if callable(generation):
            generation = generation()
        try:
            return int(generation)
        except (TypeError, ValueError):
            return 0

    def _schedule_proxy_warm(self, image_rect: QRectF) -> None:
        key = self._proxy_warm_key()
        if key is None or key == self._proxy_warm_blocked_key:
            return
        cache_generation = self._area_cache_generation()
        if self._proxy_warm_active_key == key:
            if cache_generation <= self._proxy_warm_cache_generation:
                # The preceding warm paint admitted no new path. Repeating it
                # would create an idle paint loop when the byte budget is full.
                self._proxy_warm_blocked_key = key
                self._proxy_warm_active_key = None
                return
        else:
            self._proxy_warm_active_key = key
        self._proxy_warm_cache_generation = cache_generation
        if self._proxy_warm_scheduled:
            return
        self._proxy_warm_scheduled = True
        update_rect = self._image_rect_to_widget_update_rect(image_rect)
        image_rect_copy = QRectF(image_rect)
        self._proxy_warm_pending = (key, image_rect_copy, update_rect)
        self._proxy_warm_timer.start(1)

    def _run_scheduled_proxy_warm(self) -> None:
        pending = self._proxy_warm_pending
        self._proxy_warm_pending = None
        if pending is None:
            self._proxy_warm_scheduled = False
            return
        key, image_rect, update_rect = pending
        self._run_proxy_warm(key, image_rect, update_rect)

    def _run_proxy_warm(
        self,
        key: tuple[object, ...],
        image_rect: QRectF,
        update_rect,
    ) -> None:
        self._proxy_warm_scheduled = False
        if (
            self._document is None
            or self._proxy_warm_active_key != key
            or self._proxy_warm_key() != key
        ):
            return
        index = self._measurement_index()
        if index is None or update_rect.isEmpty():
            return
        candidates = [
            measurement
            for measurement in index.query_rect(image_rect)
            if measurement.measurement_kind == "area"
            and measurement.id != self._document.view_state.selected_measurement_id
        ]
        if not candidates:
            self._proxy_warm_active_key = None
            return
        start = self._proxy_warm_cursor % len(candidates)
        ordered = candidates[start:] + candidates[:start]
        warm_budget = AreaProxyBuildBudget(max_builds=2, max_build_ms=12.0)
        generation_before = self._area_cache_generation()
        visited = 0
        with area_derived_geometry_service.path_render_pass():
            # Pin all currently cached visible paths before admitting new
            # proxies. This makes an over-budget working set stable instead of
            # turning sequential paints into an LRU scan.
            pin_budget = AreaProxyBuildBudget(max_builds=0, max_build_ms=0.0)
            for measurement in ordered:
                area_derived_geometry_service.screen_geometry(
                    measurement,
                    zoom=self._zoom,
                    selected=False,
                    build_budget=pin_budget,
                )
            for measurement in ordered:
                visited += 1
                area_derived_geometry_service.screen_geometry(
                    measurement,
                    zoom=self._zoom,
                    selected=False,
                    build_budget=warm_budget,
                )
                if (
                    warm_budget.builds >= warm_budget.max_builds
                    or warm_budget.build_ms >= warm_budget.max_build_ms
                ):
                    break
        self._proxy_warm_cursor = (start + max(1, visited)) % len(candidates)
        generation_after = self._area_cache_generation()
        if generation_after <= generation_before:
            self._proxy_warm_blocked_key = key
            self._proxy_warm_active_key = None
            return
        self.update(update_rect)

    def _reset_proxy_warming(self) -> None:
        self._proxy_warm_timer.stop()
        self._proxy_warm_pending = None
        self._proxy_warm_scheduled = False
        self._proxy_warm_active_key = None
        self._proxy_warm_blocked_key = None
        self._proxy_warm_cache_generation = -1
        self._proxy_warm_cursor = 0

    def _image_rect_to_widget_update_rect(self, image_rect: QRectF):
        transform = self._paint_context().image_to_widget_transform
        widget_rect = transform.mapRect(image_rect).adjusted(-6.0, -6.0, 6.0, 6.0)
        return widget_rect.toAlignedRect().intersected(self.rect())

    def _draw_translated_area_drag_preview(self, painter: QPainter) -> bool:
        if (
            self._document is None
            or self._dragging_area_handle is None
            or self._dragging_area_handle[1] != "center"
            or self._drag_area_preview_offset is None
        ):
            return False
        measurement = self._document.get_measurement(self._dragging_area_handle[0])
        if measurement is None or measurement.measurement_kind != "area":
            return False
        raw_path = area_derived_geometry_service.raw_path(measurement)
        if raw_path.elementCount() <= 0:
            return False

        offset = self._drag_area_preview_offset
        zoom = max(float(self._zoom), 0.001)
        preview_fill = QColor(244, 211, 94, 56)
        preview_stroke = QColor("#F4D35E")

        painter.save()
        try:
            # Keep the exact RAW path in image coordinates and express the
            # whole drag as one painter transform.  Pen widths and handles are
            # cosmetic so their screen appearance remains unchanged by zoom.
            overlay_origin = self._overlay_widget_origin()
            painter.translate(
                float(overlay_origin.x()) + (offset.x * zoom),
                float(overlay_origin.y()) + (offset.y * zoom),
            )
            painter.scale(zoom, zoom)
            if self._show_area_fill:
                painter.setBrush(preview_fill)
            else:
                painter.setBrush(Qt.BrushStyle.NoBrush)
            outline_pen = QPen(
                QColor("#0B0B0B"),
                3.0,
                Qt.PenStyle.SolidLine,
                Qt.PenCapStyle.RoundCap,
                Qt.PenJoinStyle.RoundJoin,
            )
            outline_pen.setCosmetic(True)
            painter.setPen(outline_pen)
            painter.drawPath(raw_path)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            preview_pen = QPen(
                preview_stroke,
                1.8,
                Qt.PenStyle.DashLine,
                Qt.PenCapStyle.RoundCap,
                Qt.PenJoinStyle.RoundJoin,
            )
            preview_pen.setCosmetic(True)
            painter.setPen(preview_pen)
            painter.drawPath(raw_path)

            rings = measurement.area_rings_px or (
                [measurement.polygon_px] if measurement.polygon_px else []
            )
            painter.setBrush(preview_stroke)
            handle_pen = QPen(QColor("#0B0B0B"), 1.0)
            handle_pen.setCosmetic(True)
            painter.setPen(handle_pen)
            handle_radius = 4.5 / zoom
            device = painter.device()
            device_pixel_ratio = (
                float(device.devicePixelRatioF())
                if device is not None and hasattr(device, "devicePixelRatioF")
                else 1.0
            )
            for x, y in area_handle_display_cache.coordinates(
                measurement,
                rings,
                output_scale=zoom,
                device_pixel_ratio=device_pixel_ratio,
            ):
                painter.drawEllipse(
                    QPointF(x, y),
                    handle_radius,
                    handle_radius,
                )
        finally:
            painter.restore()
        return True

    def _draw_pending_path_preview(
        self,
        painter: QPainter,
        preview_points: list[Point],
        preview_rings: list[list[Point]],
        *,
        destructive_preview: bool,
    ) -> None:
        if not preview_points:
            return
        preview_fill = QColor(248, 113, 113, 52) if destructive_preview else QColor(244, 211, 94, 56)
        preview_stroke = QColor("#F87171") if destructive_preview else QColor("#F4D35E")
        if self._drag_area_preview_points is not None and preview_rings:
            fill_path = area_rings_path(preview_rings, self.image_to_widget)
            if self._show_area_fill:
                painter.setBrush(preview_fill)
                painter.setPen(Qt.PenStyle.NoPen)
                if fill_path.elementCount() > 0:
                    painter.drawPath(fill_path)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor("#0B0B0B"), 3, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
            for ring in preview_rings:
                if len(ring) >= 3:
                    painter.drawPolygon(QPolygonF([self.image_to_widget(point) for point in ring]))
            painter.setPen(QPen(preview_stroke, 1.8, Qt.PenStyle.DashLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
            for ring in preview_rings:
                if len(ring) >= 3:
                    painter.drawPolygon(QPolygonF([self.image_to_widget(point) for point in ring]))
            painter.setBrush(preview_stroke)
            painter.setPen(QPen(QColor("#0B0B0B"), 1))
            for ring in preview_rings:
                for point in ring:
                    painter.drawEllipse(self.image_to_widget(point), 4.5, 4.5)
        else:
            polygon = QPolygonF([self.image_to_widget(point) for point in preview_points])
            if self._show_area_fill and len(preview_points) >= 3:
                painter.setBrush(preview_fill)
            else:
                painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor("#0B0B0B"), 3, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
            if len(preview_points) >= 3 and (self._drag_area_preview_points is not None or self._drawing_freehand_active):
                painter.drawPolygon(polygon)
            else:
                painter.drawPolyline(polygon)
            painter.setPen(QPen(preview_stroke, 1.8, Qt.PenStyle.DashLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
            if len(preview_points) >= 3 and (self._drag_area_preview_points is not None or self._drawing_freehand_active):
                painter.drawPolygon(polygon)
            else:
                painter.drawPolyline(polygon)
            painter.setBrush(preview_stroke)
            painter.setPen(QPen(QColor("#0B0B0B"), 1))
            for point in preview_points:
                painter.drawEllipse(self.image_to_widget(point), 4.5, 4.5)
        if self._tool_mode in {"polygon_area", "continuous_manual"} and self._area_hover_point is not None and self._drawing_polygon_points:
            painter.setPen(QPen(preview_stroke, 1.2, Qt.PenStyle.DashLine))
            painter.drawLine(self.image_to_widget(self._drawing_polygon_points[-1]), self.image_to_widget(self._area_hover_point))
            if self._tool_mode == "polygon_area" and len(self._drawing_polygon_points) >= 2:
                painter.drawLine(self.image_to_widget(self._area_hover_point), self.image_to_widget(self._drawing_polygon_points[0]))
        elif (
            self._magic_manual_subtract_mode_active(MagicSegmentSubtractInputMode.POLYGON)
            and self._area_hover_point is not None
            and self._drawing_polygon_points
        ):
            painter.setPen(QPen(preview_stroke, 1.2, Qt.PenStyle.DashLine))
            painter.drawLine(self.image_to_widget(self._drawing_polygon_points[-1]), self.image_to_widget(self._area_hover_point))
            if len(self._drawing_polygon_points) >= 2:
                painter.drawLine(self.image_to_widget(self._area_hover_point), self.image_to_widget(self._drawing_polygon_points[0]))

    def _rebuild_project_roi_paths(self) -> None:
        document_id = self.document_id
        if document_id is None:
            self._project_roi_paths = ()
            return
        paths: list[tuple[ProjectRoi, QPainterPath]] = []
        for roi in self._project_rois:
            if not roi.visible or roi.document_id != document_id:
                continue
            try:
                path = self._project_roi_path(roi, stack=())
            except (KeyError, TypeError, ValueError):
                continue
            if not path.isEmpty():
                paths.append((roi, path))
        self._project_roi_paths = tuple(paths)

    def _project_roi_path(
        self,
        roi: ProjectRoi,
        *,
        stack: tuple[str, ...],
    ) -> QPainterPath:
        if roi.id in stack:
            raise ValueError("ROI 布尔表达式存在循环引用")
        geometry = roi.geometry
        path = QPainterPath()
        path.setFillRule(Qt.FillRule.OddEvenFill)
        if isinstance(geometry, RectangleRoiGeometry):
            path.addRect(
                QRectF(
                    geometry.x,
                    geometry.y,
                    geometry.width,
                    geometry.height,
                )
            )
            return path
        if isinstance(geometry, EllipseRoiGeometry):
            path.addEllipse(
                QRectF(
                    geometry.x,
                    geometry.y,
                    geometry.width,
                    geometry.height,
                )
            )
            return path
        if isinstance(geometry, (PolygonRoiGeometry, FreehandRoiGeometry)):
            for ring in geometry.rings:
                first = ring[0]
                path.moveTo(first.x, first.y)
                for point in ring[1:]:
                    path.lineTo(point.x, point.y)
                path.closeSubpath()
            return path
        if not isinstance(geometry, RoiBooleanExpression):
            return path

        operands: list[QPainterPath] = []
        for operand_id in geometry.operand_ids:
            operand = self._project_roi_lookup.get(operand_id)
            if operand is None:
                raise KeyError(operand_id)
            if operand.document_id != roi.document_id:
                raise ValueError("组合 ROI 不能引用其他文档")
            operands.append(
                self._project_roi_path(
                    operand,
                    stack=(*stack, roi.id),
                )
            )
        if not operands:
            return path
        result = QPainterPath(operands[0])
        for operand_path in operands[1:]:
            if geometry.operator is RoiBooleanOperator.UNION:
                result = result.united(operand_path)
            elif geometry.operator is RoiBooleanOperator.INTERSECTION:
                result = result.intersected(operand_path)
            elif geometry.operator is RoiBooleanOperator.DIFFERENCE:
                result = result.subtracted(operand_path)
            else:
                result = result.united(operand_path).subtracted(
                    result.intersected(operand_path)
                )
        result.setFillRule(Qt.FillRule.OddEvenFill)
        return result

    def _draw_project_rois(self, painter: QPainter, image_target: QRectF) -> None:
        if not self._project_roi_paths:
            return
        origin = self.image_to_widget(Point(0.0, 0.0))
        x_unit = self.image_to_widget(Point(1.0, 0.0))
        y_unit = self.image_to_widget(Point(0.0, 1.0))
        transform = QTransform(
            x_unit.x() - origin.x(),
            x_unit.y() - origin.y(),
            y_unit.x() - origin.x(),
            y_unit.y() - origin.y(),
            origin.x(),
            origin.y(),
        )
        painter.save()
        try:
            painter.setClipRect(image_target, Qt.ClipOperation.IntersectClip)
            painter.setTransform(transform, True)
            for roi, path in self._project_roi_paths:
                stroke = QColor(roi.color)
                fill = QColor(stroke)
                fill.setAlpha(36)
                pen = QPen(stroke, 1.6, Qt.PenStyle.DashLine)
                pen.setCosmetic(True)
                painter.setPen(pen)
                painter.setBrush(fill)
                painter.drawPath(path)
        finally:
            painter.restore()

    def _draw_roi_capture_preview(self, painter: QPainter) -> None:
        session = self._roi_capture
        if session is None:
            return
        stroke = QColor(self.palette().color(QPalette.ColorRole.Highlight))
        if not stroke.isValid():
            stroke = QColor("#2A9D8F")
        fill = QColor(stroke)
        fill.setAlpha(46)
        pen = QPen(stroke, 1.8, Qt.PenStyle.DashLine)
        pen.setCosmetic(True)
        painter.save()
        try:
            painter.setPen(pen)
            painter.setBrush(fill)
            if (
                session.kind
                in {ProjectRoiKind.RECTANGLE, ProjectRoiKind.ELLIPSE}
                and session.drag_start is not None
                and session.drag_end is not None
            ):
                rect = QRectF(
                    self.image_to_widget(session.drag_start),
                    self.image_to_widget(session.drag_end),
                ).normalized()
                if session.kind is ProjectRoiKind.ELLIPSE:
                    painter.drawEllipse(rect)
                else:
                    painter.drawRect(rect)
                return
            if not session.points:
                return
            polygon = QPolygonF(
                [self.image_to_widget(point) for point in session.points]
            )
            if len(session.points) >= 3:
                painter.drawPolygon(polygon, Qt.FillRule.OddEvenFill)
            else:
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawPolyline(polygon)
            if (
                session.kind is ProjectRoiKind.POLYGON
                and session.hover_point is not None
            ):
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawLine(
                    self.image_to_widget(session.points[-1]),
                    self.image_to_widget(session.hover_point),
                )
                if len(session.points) >= 2:
                    painter.drawLine(
                        self.image_to_widget(session.hover_point),
                        self.image_to_widget(session.points[0]),
                    )
        finally:
            painter.restore()

    def _draw_preview(self, painter: QPainter) -> None:
        preview_line = self._drag_preview_line or self._drawing_line
        if preview_line is not None:
            color = QColor("#FF7F50") if self._tool_mode == "calibration" else QColor("#F4D35E")
            painter.setPen(QPen(color, 2, Qt.PenStyle.DashLine))
            start_point = self.image_to_widget(preview_line.start)
            end_point = self.image_to_widget(preview_line.end)
            painter.drawLine(start_point, end_point)
            draw_endpoint_style(
                painter,
                QPointF(start_point),
                QPointF(end_point),
                color,
                self._settings.measurement_endpoint_style,
                line_width=2.0,
                endpoint_radius=6.0,
            )

        self._draw_translated_area_drag_preview(painter)
        preview_points = self._drag_area_preview_points or self._drawing_polygon_points
        preview_rings = self._drag_area_preview_rings or []
        destructive_area_preview = bool(
            preview_points
            and (
                self._magic_manual_subtract_mode_active()
                or self._area_subtract_mode_active()
            )
        )
        if preview_points and not destructive_area_preview:
            self._draw_pending_path_preview(
                painter,
                preview_points,
                preview_rings,
                destructive_preview=False,
            )

        preview_overlay = self._drag_overlay_preview
        if preview_overlay is not None:
            draw_overlay_annotations(
                painter,
                type("PreviewDoc", (), {"overlay_annotations": [preview_overlay], "selected_overlay_id": preview_overlay.id})(),
                self.image_to_widget,
                self._settings,
                selected_overlay_id=preview_overlay.id,
                show_handles=False,
                render_mode="screen_scale_full_image",
            )

        if self._drawing_overlay_start is not None and self._drawing_overlay_end is not None:
            preview_kind = self._overlay_tool_kind
            preview_annotation = OverlayAnnotation(
                id="preview_overlay",
                image_id=self._document.id if self._document is not None else "",
                kind=preview_kind,
                start_px=self._drawing_overlay_start,
                end_px=self._drawing_overlay_end,
            )
            draw_overlay_annotations(
                painter,
                type("PreviewDoc", (), {"overlay_annotations": [preview_annotation], "selected_overlay_id": preview_annotation.id})(),
                self.image_to_widget,
                self._settings,
                selected_overlay_id=preview_annotation.id,
                show_handles=False,
                render_mode="screen_scale_full_image",
            )

        if is_magic_segment_tool_mode(self._tool_mode):
            self._draw_magic_segment_preview(painter)
        elif is_fiber_quick_tool_mode(self._tool_mode):
            self._draw_fiber_quick_preview(painter)
        elif is_reference_propagation_tool_mode(self._tool_mode) or self._reference_instance.has_session():
            self._draw_reference_instance_preview(painter)

        if preview_points and destructive_area_preview:
            self._draw_pending_path_preview(
                painter,
                preview_points,
                preview_rings,
                destructive_preview=True,
            )

        if self._tool_mode == "calibration":
            self._draw_magic_prompt_status_label(
                painter,
                prompt_type=None,
                operation_text="标定 · 拖拽标尺线 · Shift 锁定水平/垂直 · Ctrl 吸附像素中心",
                busy=False,
            )

        if self._scale_anchor_pick_active:
            preview_point = self._scale_anchor_preview_point or Point(self._image.width() * 0.15, self._image.height() * 0.2)
            draw_preview_scale_anchor(painter, self.image_to_widget(preview_point))

        self._draw_roi_capture_preview(painter)

    def _hit_test_selected_endpoint(self, image_point: Point) -> tuple[str, str] | None:
        if self._document is None or self._document.view_state.selected_measurement_id is None:
            return None
        measurement = self._document.get_measurement(self._document.view_state.selected_measurement_id)
        if measurement is None or measurement.measurement_kind != "line":
            return None
        endpoint_name, endpoint_distance = nearest_endpoint(measurement.effective_line(), image_point)
        if endpoint_distance <= self._selected_endpoint_tolerance():
            return measurement.id, endpoint_name
        return None

    def _hit_test_endpoint(self, image_point: Point) -> tuple[str, str] | None:
        if self._document is None:
            return None
        tolerance = self._endpoint_tolerance()
        for measurement in self._measurement_candidates(image_point, tolerance=tolerance):
            if measurement.measurement_kind != "line":
                continue
            line = measurement.effective_line()
            bounds = (min(line.start.x, line.end.x), min(line.start.y, line.end.y),
                      max(line.start.x, line.end.x), max(line.start.y, line.end.y))
            if not point_near_bounds(image_point, bounds, tolerance):
                continue
            endpoint_name, endpoint_distance = nearest_endpoint(line, image_point)
            if endpoint_distance <= tolerance:
                return measurement.id, endpoint_name
        return None

    def _hit_test_selected_area_handle(self, image_point: Point) -> tuple[str, str, int | None, int | None] | None:
        if self._document is None or self._document.view_state.selected_measurement_id is None:
            return None
        measurement = self._document.get_measurement(self._document.view_state.selected_measurement_id)
        if (
            measurement is None
            or measurement.measurement_kind != "area"
            or (len(measurement.polygon_px) < 3 and not measurement.area_rings_px)
        ):
            return None
        nearest_vertex = area_derived_geometry_service.nearest_vertex(
            measurement,
            image_point,
            self._selected_endpoint_tolerance(),
        )
        if nearest_vertex is not None:
            return measurement.id, "vertex", nearest_vertex[0], nearest_vertex[1]
        center = measurement.polygon_center()
        if distance(center, image_point) <= max(3.0, 5.0 / max(self._zoom, 0.001)):
            return measurement.id, "center", None, None
        return None

    def _hit_test_area_measurement(self, image_point: Point) -> str | None:
        if self._document is None:
            return None
        tolerance = self._measurement_query_tolerance()
        for measurement in self._measurement_candidates(image_point, tolerance=tolerance):
            measurement_tolerance = self._measurement_hit_tolerance(measurement)
            raw_geometry = area_geometry_raw(measurement)
            outline_points = raw_geometry.outline_points
            fill_rings = raw_geometry.fill_rings
            bounds = raw_geometry.bounds
            if measurement.measurement_kind != "area" or (len(outline_points) < 3 and not fill_rings):
                continue
            if bounds is None:
                continue
            if not point_near_bounds(image_point, bounds, measurement_tolerance):
                continue
            if fill_rings:
                if area_derived_geometry_service.contains_raw(measurement, image_point):
                    return measurement.id
                if area_derived_geometry_service.near_edge(
                    measurement,
                    image_point,
                    measurement_tolerance,
                ):
                    return measurement.id
                continue
            if point_in_polygon(image_point, outline_points):
                return measurement.id
            if point_to_polygon_edge_distance(image_point, outline_points) <= measurement_tolerance:
                return measurement.id
        return None

    def _hit_test_measurement(self, image_point: Point) -> str | None:
        if self._document is None:
            return None
        tolerance = self._measurement_query_tolerance()
        for measurement in self._measurement_candidates(image_point, tolerance=tolerance):
            measurement_tolerance = self._measurement_hit_tolerance(measurement)
            if measurement.measurement_kind == "line":
                line = measurement.effective_line()
                bounds = (
                    min(line.start.x, line.end.x),
                    min(line.start.y, line.end.y),
                    max(line.start.x, line.end.x),
                    max(line.start.y, line.end.y),
                )
                if not point_near_bounds(image_point, bounds, measurement_tolerance):
                    continue
                if self._point_to_segment_distance(image_point, line) <= measurement_tolerance:
                    return measurement.id
                continue
            if measurement.measurement_kind == "polyline" and len(measurement.polyline_px) >= 2:
                xs = [point.x for point in measurement.polyline_px]
                ys = [point.y for point in measurement.polyline_px]
                bounds = (min(xs), min(ys), max(xs), max(ys))
                if not point_near_bounds(image_point, bounds, measurement_tolerance):
                    continue
                if point_to_polyline_distance(image_point, measurement.polyline_px) <= measurement_tolerance:
                    return measurement.id
                continue
            if measurement.measurement_kind == "count" and measurement.point_px is not None:
                if distance(image_point, measurement.point_px) <= measurement_tolerance:
                    return measurement.id
        return None

    def _measurement_hit_tolerance(self, measurement: Measurement) -> float:
        zoom = max(self._zoom, 0.001)
        base = max(5.0, 10.0 / zoom)
        appearance = measurement.appearance
        if measurement.measurement_kind == "count":
            marker_scale = (
                appearance.marker_scale
                if appearance is not None and appearance.marker_scale is not None
                else 1.0
            )
            return max(base, (5.5 * marker_scale) / zoom)
        stroke_width = (
            appearance.stroke_width
            if appearance is not None and appearance.stroke_width is not None
            else 2.0
        )
        return max(base, ((stroke_width * 0.6) + 3.0) / zoom)

    def _measurement_query_tolerance(self) -> float:
        """Conservative broad-phase radius independent of document size.

        Object appearance normalization caps marker scale at 4 and stroke
        width at 24 logical pixels.  The largest possible point radius
        (5.5 * 4) therefore dominates the stroke tolerance (24 * 0.6 + 3).
        Individual candidates still use their exact tolerance in the narrow
        phase; this value only expands the spatial-index query.
        """

        zoom = max(self._zoom, 0.001)
        return max(5.0, 22.0 / zoom)

    def _selected_endpoint_tolerance(self) -> float:
        return max(4.0, 9.0 / max(self._zoom, 0.001))

    def _endpoint_tolerance(self) -> float:
        return max(3.0, 6.0 / max(self._zoom, 0.001))

    def _polygon_close_tolerance(self) -> float:
        # Closing is a drawing gesture, so keep it tighter than selected-handle picking.
        return 5.0 / max(self._zoom, 0.001)

    def _hit_test_selected_overlay_handle(self, image_point: Point) -> tuple[str, str] | None:
        if self._document is None or self._document.selected_overlay_id is None:
            return None
        annotation = self._document.get_overlay_annotation(self._document.selected_overlay_id)
        if annotation is None or annotation.normalized_kind() == OverlayAnnotationKind.TEXT:
            return None
        tolerance = self._selected_endpoint_tolerance()
        for handle_name, handle_point in overlay_annotation_handle_points(annotation):
            if distance(handle_point, image_point) <= tolerance:
                return annotation.id, handle_name
        return None

    def _hit_test_overlay_annotation(self, widget_point: QPointF, image_point: Point) -> str | None:
        if self._document is None:
            return None
        tolerance = max(5.0, 10.0 / max(self._zoom, 0.001))
        for annotation in reversed(self._document.overlay_annotations):
            kind = annotation.normalized_kind()
            if kind == OverlayAnnotationKind.TEXT:
                rect = annotation_rect(annotation, self._settings, self.image_to_widget)
                if rect.contains(widget_point):
                    return annotation.id
                continue
            if self._overlay_shape_hit(annotation, image_point, tolerance):
                return annotation.id
        return None

    def _point_to_segment_distance(self, point: Point, line: Line) -> float:
        vx = line.end.x - line.start.x
        vy = line.end.y - line.start.y
        length_sq = (vx * vx) + (vy * vy)
        if length_sq == 0:
            return distance(point, line.start)
        projection = ((point.x - line.start.x) * vx + (point.y - line.start.y) * vy) / length_sq
        projection = max(0.0, min(1.0, projection))
        closest = Point(
            x=line.start.x + (projection * vx),
            y=line.start.y + (projection * vy),
        )
        return distance(point, closest)

    def _overlay_shape_hit(self, annotation: OverlayAnnotation, image_point: Point, tolerance: float) -> bool:
        kind = annotation.normalized_kind()
        if annotation.appearance is not None and annotation.appearance.stroke_width is not None:
            tolerance = max(
                tolerance,
                ((annotation.appearance.stroke_width * 0.5) + 3.0)
                / max(self._zoom, 0.001),
            )
        min_x, min_y, max_x, max_y = overlay_annotation_bounds(annotation)
        bounds = (min_x, min_y, max_x, max_y)
        if not point_near_bounds(image_point, bounds, tolerance):
            return False
        if kind == OverlayAnnotationKind.RECT:
            inside = min_x <= image_point.x <= max_x and min_y <= image_point.y <= max_y
            if inside:
                return True
            edges = [
                Line(Point(min_x, min_y), Point(max_x, min_y)),
                Line(Point(max_x, min_y), Point(max_x, max_y)),
                Line(Point(max_x, max_y), Point(min_x, max_y)),
                Line(Point(min_x, max_y), Point(min_x, min_y)),
            ]
            return any(self._point_to_segment_distance(image_point, edge) <= tolerance for edge in edges)
        if kind == OverlayAnnotationKind.CIRCLE:
            if max_x - min_x <= 1e-6 or max_y - min_y <= 1e-6:
                return False
            cx = (min_x + max_x) / 2.0
            cy = (min_y + max_y) / 2.0
            rx = max((max_x - min_x) / 2.0, 1e-6)
            ry = max((max_y - min_y) / 2.0, 1e-6)
            normalized = (((image_point.x - cx) / rx) ** 2) + (((image_point.y - cy) / ry) ** 2)
            edge_tolerance = max(tolerance / max(rx, ry), 0.02)
            return normalized <= 1.0 or abs(normalized - 1.0) <= edge_tolerance
        segment = Line(annotation.start_px, annotation.end_px)
        return self._point_to_segment_distance(image_point, segment) <= tolerance

    def _overlay_annotation_clamped(self, annotation: OverlayAnnotation) -> OverlayAnnotation:
        if annotation.normalized_kind() == OverlayAnnotationKind.TEXT:
            candidate = annotation.clone(
                anchor_px=self._clamp_to_image(
                    annotation.anchor_px,
                    pixel_center=False,
                )
            )
            image_size = self._image_size()
            if image_size is None:
                return candidate
            image_space_layout = (
                candidate.text_layout is not None
                and candidate.text_layout.size_space
                == OverlayTextSizeSpace.IMAGE_PX
            )
            screen_rect = annotation_rect(
                candidate,
                self._settings,
                self.image_to_widget,
                render_mode="screen_scale_full_image",
            )
            if not screen_rect.isValid() or screen_rect.isEmpty():
                return candidate
            image_corners = [
                self.widget_to_image(point)
                for point in (
                    screen_rect.topLeft(),
                    screen_rect.topRight(),
                    screen_rect.bottomLeft(),
                    screen_rect.bottomRight(),
                )
            ]
            if image_space_layout:
                full_resolution_rect = annotation_rect(
                    candidate,
                    self._settings,
                    lambda point: QPointF(point.x, point.y),
                    render_mode="full_resolution",
                )
                image_corners.extend(
                    Point(point.x(), point.y())
                    for point in (
                        full_resolution_rect.topLeft(),
                        full_resolution_rect.topRight(),
                        full_resolution_rect.bottomLeft(),
                        full_resolution_rect.bottomRight(),
                    )
                )
            left = min(point.x for point in image_corners)
            top = min(point.y for point in image_corners)
            right = max(point.x for point in image_corners)
            bottom = max(point.y for point in image_corners)
            width, height = image_size
            right_limit = max(0.0, float(width) - 1.0)
            bottom_limit = max(0.0, float(height) - 1.0)
            dx = 0.0
            dy = 0.0
            if right - left <= right_limit:
                if left < 0.0:
                    dx = -left
                elif right > right_limit:
                    dx = right_limit - right
            if bottom - top <= bottom_limit:
                if top < 0.0:
                    dy = -top
                elif bottom > bottom_limit:
                    dy = bottom_limit - bottom
            if dx or dy:
                candidate = candidate.translated(dx, dy)
                candidate = candidate.clone(
                    anchor_px=self._clamp_to_image(
                        candidate.anchor_px,
                        pixel_center=False,
                    )
                )
            return candidate
        return annotation.clone(
            start_px=self._clamp_to_image(annotation.start_px, pixel_center=False),
            end_px=self._clamp_to_image(annotation.end_px, pixel_center=False),
        )

    def constrain_overlay_annotation(
        self,
        annotation: OverlayAnnotation,
    ) -> OverlayAnnotation:
        """Keep a new or edited overlay within the source-image boundary."""

        return self._overlay_annotation_clamped(annotation)

    def _translate_overlay_annotation(self, annotation: OverlayAnnotation, dx: float, dy: float) -> OverlayAnnotation:
        if annotation.normalized_kind() == OverlayAnnotationKind.TEXT:
            return self._overlay_annotation_clamped(annotation.translated(dx, dy))
        if self._image is None:
            return annotation.translated(dx, dy)
        min_x, min_y, max_x, max_y = overlay_annotation_bounds(annotation)
        dx = clamp(dx, -min_x, (self._image.width() - 1.0) - max_x)
        dy = clamp(dy, -min_y, (self._image.height() - 1.0) - max_y)
        return annotation.translated(dx, dy)

    def _resize_overlay_annotation(
        self,
        annotation: OverlayAnnotation,
        handle_name: str,
        point: Point,
        modifiers: Qt.KeyboardModifiers = Qt.KeyboardModifier.NoModifier,
    ) -> OverlayAnnotation:
        point = self._clamp_to_image(point, pixel_center=False)
        kind = annotation.normalized_kind()
        if kind in {OverlayAnnotationKind.LINE, OverlayAnnotationKind.ARROW}:
            if handle_name == "start":
                return annotation.clone(start_px=point)
            return annotation.clone(end_px=point)
        if bool(modifiers & Qt.KeyboardModifier.ShiftModifier) and kind in {OverlayAnnotationKind.RECT, OverlayAnnotationKind.CIRCLE}:
            opposite = self._overlay_opposite_corner(annotation, handle_name)
            point = self._constrain_overlay_candidate(opposite, point, modifiers, kind=kind)
        min_x, min_y, max_x, max_y = overlay_annotation_bounds(annotation)
        if handle_name == "top_left":
            min_x, min_y = point.x, point.y
        elif handle_name == "top_right":
            max_x, min_y = point.x, point.y
        elif handle_name == "bottom_left":
            min_x, max_y = point.x, point.y
        else:
            max_x, max_y = point.x, point.y
        return annotation.clone(
            start_px=Point(min_x, min_y),
            end_px=Point(max_x, max_y),
        )

    def _overlay_geometry_visible(self, start_point: Point, end_point: Point) -> bool:
        if self._overlay_tool_kind in {OverlayAnnotationKind.LINE, OverlayAnnotationKind.ARROW}:
            return distance(start_point, end_point) >= 1.0
        return abs(end_point.x - start_point.x) >= 2.0 and abs(end_point.y - start_point.y) >= 2.0

    def _constrain_overlay_candidate(
        self,
        anchor: Point,
        candidate: Point,
        modifiers: Qt.KeyboardModifiers,
        *,
        kind: str | None = None,
    ) -> Point:
        candidate = self._clamp_to_image(candidate, pixel_center=False)
        if not bool(modifiers & Qt.KeyboardModifier.ShiftModifier):
            return candidate
        target_kind = kind or self._overlay_tool_kind
        if target_kind not in {OverlayAnnotationKind.RECT, OverlayAnnotationKind.CIRCLE}:
            return candidate
        dx = candidate.x - anchor.x
        dy = candidate.y - anchor.y
        sign_x = -1.0 if dx < 0 else 1.0
        sign_y = -1.0 if dy < 0 else 1.0
        available_x = self._overlay_axis_room(anchor.x, sign_x, axis="x")
        available_y = self._overlay_axis_room(anchor.y, sign_y, axis="y")
        size = min(max(abs(dx), abs(dy)), available_x, available_y)
        return Point(
            anchor.x + (sign_x * size),
            anchor.y + (sign_y * size),
        )

    def _overlay_axis_room(self, origin: float, sign: float, *, axis: str) -> float:
        if self._image is None:
            return float("inf")
        limit = (self._image.width() - 1.0) if axis == "x" else (self._image.height() - 1.0)
        if sign >= 0:
            return max(0.0, limit - origin)
        return max(0.0, origin)

    def _overlay_opposite_corner(self, annotation: OverlayAnnotation, handle_name: str) -> Point:
        min_x, min_y, max_x, max_y = overlay_annotation_bounds(annotation)
        if handle_name == "top_left":
            return Point(max_x, max_y)
        if handle_name == "top_right":
            return Point(min_x, max_y)
        if handle_name == "bottom_left":
            return Point(max_x, min_y)
        return Point(min_x, min_y)

    def _cancel_overlay_interaction(self) -> None:
        self._drawing_overlay_start = None
        self._drawing_overlay_end = None
        self._dragging_overlay_id = None
        self._dragging_overlay_handle = None
        self._drag_overlay_press_point = None
        self._drag_overlay_origin = None
        self._drag_overlay_preview = None

    def _image_size(self) -> tuple[int, int] | None:
        if self._image is None:
            return None
        return self._image.width(), self._image.height()

    def _apply_line_constraints(
        self,
        anchor: Point,
        candidate: Point,
        modifiers: Qt.KeyboardModifiers,
        *,
        snap_anchor: bool,
    ) -> tuple[Point, Point]:
        use_ctrl = bool(modifiers & Qt.KeyboardModifier.ControlModifier)
        use_shift = bool(modifiers & Qt.KeyboardModifier.ShiftModifier)
        line = self._line_tool_strategy.preview_line(
            anchor,
            candidate,
            image_size=self._image_size(),
            constrain_axis=use_shift,
            snap_to_pixel=use_ctrl,
            snap_anchor=snap_anchor,
        )
        return line.start, line.end

    def _anchor_point_for_event(self, image_point: Point, modifiers: Qt.KeyboardModifiers) -> Point:
        use_ctrl = bool(modifiers & Qt.KeyboardModifier.ControlModifier)
        return self._line_tool_strategy.anchor_for_event(
            image_point,
            image_size=self._image_size(),
            snap_to_pixel=use_ctrl,
        )

    def _begin_line_drawing(self, anchor: Point, *, commit_on_second_click: bool = False) -> None:
        state = self._line_tool_strategy.begin(anchor, commit_on_second_click=commit_on_second_click)
        self._drawing_anchor_raw = state.anchor_raw
        self._drawing_line = state.preview_line
        self._line_commit_on_second_click = state.commit_on_second_click
        if self.document_id is not None:
            self.pathSessionChanged.emit(self.document_id)

    def _cancel_line_drawing(self) -> None:
        previous_line = self._drawing_line
        had_line = previous_line is not None
        document_id = self.document_id
        state = self._line_tool_strategy.cancel()
        self._drawing_anchor_raw = state.anchor_raw
        self._drawing_line = state.preview_line
        self._line_commit_on_second_click = state.commit_on_second_click
        if had_line and document_id is not None:
            self.pathSessionChanged.emit(document_id)
        if previous_line is not None:
            self._update_preview_regions(
                [previous_line.start, previous_line.end],
                [],
                padding_screen=18.0,
            )

    def _commit_click_line(self, image_point: Point, modifiers: Qt.KeyboardModifiers) -> None:
        if self._document is None or self._drawing_anchor_raw is None:
            return
        start, end = self._apply_line_constraints(
            self._drawing_anchor_raw,
            image_point,
            modifiers,
            snap_anchor=True,
        )
        line = Line(start=start, end=end)
        self._cancel_line_drawing()
        payload = self._line_tool_strategy.commit_payload(line)
        if payload is not None:
            self.lineCommitted.emit(self._document.id, self._tool_mode, payload)
        if self._space_pressed:
            self._temporary_grab_active = True
            self._update_cursor()

    def _clamp_to_image(self, point: Point, *, pixel_center: bool) -> Point:
        return clamp_point_to_image(point, self._image_size(), pixel_center=pixel_center)

    def _clamp_roi_point(self, point: Point) -> Point:
        """Clamp ROI boundary coordinates without dropping the last pixel.

        Measurement control points are clamped to ``width - 1``/``height - 1``.
        ROI coordinates describe pixel boundaries, so the valid right and
        bottom edges are exactly ``width`` and ``height`` under the
        pixel-centre mask rule.
        """

        image_size = self._image_size()
        if image_size is None:
            return point
        width, height = image_size
        return Point(
            clamp(float(point.x), 0.0, float(width)),
            clamp(float(point.y), 0.0, float(height)),
        )

    def _persist_view_state(self) -> None:
        if self._document is None:
            return
        self._document.view_state.zoom = self._zoom
        self._document.view_state.pan = Point(self._pan.x, self._pan.y)

    def _has_pointer_edit_operation(self) -> bool:
        return (
            self._roi_capture is not None
            or self._drawing_anchor_raw is not None
            or bool(self._drawing_polygon_points)
            or self._drawing_freehand_active
            or self._dragging_handle is not None
            or self._dragging_area_handle is not None
            or self._drawing_overlay_start is not None
            or self._dragging_overlay_id is not None
            or self._dragging_overlay_handle is not None
            or self._scale_anchor_pick_active
            or self._reference_instance.dragging
        )

    def _roi_capture_mouse_press(self, image_point: Point) -> None:
        session = self._roi_capture
        if session is None or not self._point_in_image(image_point):
            return
        point = self._clamp_roi_point(image_point)
        if session.kind in {
            ProjectRoiKind.RECTANGLE,
            ProjectRoiKind.ELLIPSE,
        }:
            session.drag_start = point
            session.drag_end = point
            session.dragging = True
        elif session.kind is ProjectRoiKind.POLYGON:
            if (
                len(session.points) >= 3
                and distance(point, session.points[0])
                <= self._polygon_close_tolerance()
            ):
                self._commit_roi_capture()
                return
            if not session.points or distance(session.points[-1], point) > 1e-6:
                session.points.append(point)
            session.hover_point = point
        else:
            session.points = [point]
            session.hover_point = point
            session.dragging = True
        self.update()

    def _roi_capture_mouse_move(self, image_point: Point) -> None:
        session = self._roi_capture
        if session is None:
            return
        point = self._clamp_roi_point(image_point)
        if (
            session.kind
            in {ProjectRoiKind.RECTANGLE, ProjectRoiKind.ELLIPSE}
            and session.dragging
        ):
            session.drag_end = point
        elif session.kind is ProjectRoiKind.POLYGON and session.points:
            session.hover_point = point
        elif (
            session.kind is ProjectRoiKind.FREEHAND
            and session.dragging
            and session.points
        ):
            minimum_distance = max(0.25, 1.0 / max(self._zoom, 1e-6))
            if distance(session.points[-1], point) >= minimum_distance:
                session.points.append(point)
            session.hover_point = point
        self.update()

    def _roi_capture_mouse_release(self, image_point: Point) -> None:
        session = self._roi_capture
        if session is None:
            return
        point = self._clamp_roi_point(image_point)
        if session.kind in {
            ProjectRoiKind.RECTANGLE,
            ProjectRoiKind.ELLIPSE,
        }:
            if not session.dragging:
                return
            session.drag_end = point
            session.dragging = False
            if not self._commit_roi_capture():
                self._clear_roi_capture(restore_tool=True)
            return
        if session.kind is ProjectRoiKind.FREEHAND and session.dragging:
            if not session.points or distance(session.points[-1], point) > 1e-6:
                session.points.append(point)
            session.dragging = False
            if not self._commit_roi_capture():
                self._clear_roi_capture(restore_tool=True)

    def _commit_roi_capture(self) -> bool:
        session = self._roi_capture
        document_id = self.document_id
        if session is None or document_id is None:
            return False
        geometry = None
        if session.kind in {
            ProjectRoiKind.RECTANGLE,
            ProjectRoiKind.ELLIPSE,
        }:
            if session.drag_start is None or session.drag_end is None:
                return False
            left = min(session.drag_start.x, session.drag_end.x)
            top = min(session.drag_start.y, session.drag_end.y)
            width = abs(session.drag_end.x - session.drag_start.x)
            height = abs(session.drag_end.y - session.drag_start.y)
            if width <= 1e-6 or height <= 1e-6:
                return False
            if session.kind is ProjectRoiKind.RECTANGLE:
                geometry = RectangleRoiGeometry(left, top, width, height)
            else:
                geometry = EllipseRoiGeometry(left, top, width, height)
        else:
            points = _normalized_roi_capture_points(session.points)
            if len(points) < 3:
                return False
            rings = (
                tuple(RoiPoint(point.x, point.y) for point in points),
            )
            if session.kind is ProjectRoiKind.POLYGON:
                geometry = PolygonRoiGeometry(rings)
            else:
                geometry = FreehandRoiGeometry(rings)
        commit = RoiGeometryCommit(
            request_id=session.request_id,
            document_id=document_id,
            kind=session.kind,
            geometry=geometry,
        )
        self._clear_roi_capture(restore_tool=True)
        self.roiGeometryCommitted.emit(commit)
        return True

    def _clear_roi_capture(self, *, restore_tool: bool) -> None:
        session = self._roi_capture
        if session is None:
            return
        self._roi_capture = None
        if restore_tool:
            self._tool_mode = session.restore_tool_mode
            self._overlay_tool_kind = session.restore_overlay_kind
        self._update_cursor()
        self.update()

    def _draw_magic_segment_preview(self, painter: QPainter) -> None:
        if self._image is None:
            return
        for polygon_points, area_rings in zip(
            self._magic_segment.confirmed_subtract_polygons,
            self._magic_segment.confirmed_subtract_rings,
        ):
            self._draw_magic_area_preview(
                painter,
                polygon_points,
                area_rings,
                fill_color=QColor(248, 113, 113, 68),
                stroke_color=QColor("#F87171"),
            )
        self._draw_magic_area_preview(
            painter,
            self._magic_segment.primary_polygon,
            self._magic_segment.primary_rings,
            fill_color=QColor(52, 211, 153, 72),
            stroke_color=QColor("#34D399"),
        )
        self._draw_magic_area_preview(
            painter,
            self._magic_segment.subtract_polygon,
            self._magic_segment.subtract_rings,
            fill_color=QColor(248, 113, 113, 68),
            stroke_color=QColor("#F87171"),
        )

        active_stage = self._magic_segment.active_stage
        subtract_input_mode = self.current_magic_subtract_input_mode()
        show_prompt = active_stage != MagicSegmentOperationMode.SUBTRACT or subtract_input_mode == MagicSegmentSubtractInputMode.SMART
        if show_prompt:
            self._draw_magic_prompt_points(
                painter,
                self._magic_segment.positive_points_for_stage(active_stage),
                QColor(magic_prompt_visual("positive").marker_color),
                positive=True,
            )
            self._draw_magic_prompt_points(
                painter,
                self._magic_segment.negative_points_for_stage(active_stage),
                QColor(magic_prompt_visual("negative").marker_color),
                positive=False,
            )
        self._draw_roi_debug_box(
            painter,
            self._magic_segment.debug_payload_for_stage(active_stage).get("segmentation_crop_box"),
            stroke_color=QColor("#F4D35E"),
        )

        prompt_type = self._magic_segment.prompt_type_for_stage(active_stage) if show_prompt else None
        if active_stage == MagicSegmentOperationMode.SUBTRACT:
            subtract_label = {
                MagicSegmentSubtractInputMode.POLYGON: "多边形剔除",
                MagicSegmentSubtractInputMode.FREEHAND: "自由圈选剔除",
            }.get(subtract_input_mode, "智能剔除")
            operation_text = f"标准魔棒 · {subtract_label}"
            if self._magic_segment.confirmed_subtract_count() > 0:
                operation_text += f" · 已加入 {self._magic_segment.confirmed_subtract_count()} 块"
        else:
            operation_text = "标准魔棒 · 添加主体"
        self._draw_magic_prompt_status_label(
            painter,
            prompt_type=prompt_type,
            operation_text=operation_text,
            busy=self._magic_segment.busy,
        )

    def _draw_reference_instance_preview(self, painter: QPainter) -> None:
        if self._image is None:
            return
        if self._reference_instance.dragging and self._reference_instance.drag_start is not None and self._reference_instance.drag_end is not None:
            rect = QRectF(
                self.image_to_widget(self._reference_instance.drag_start),
                self.image_to_widget(self._reference_instance.drag_end),
            ).normalized()
            painter.setBrush(QColor(96, 165, 250, 40))
            painter.setPen(QPen(QColor("#60A5FA"), 1.8, Qt.PenStyle.DashLine))
            painter.drawRect(rect)
        self._draw_magic_area_preview(
            painter,
            self._reference_instance.reference_polygon,
            self._reference_instance.reference_rings,
            fill_color=QColor(96, 165, 250, 54),
            stroke_color=QColor("#60A5FA"),
        )
        for candidate in self._reference_instance.preview_candidates:
            self._draw_magic_area_preview(
                painter,
                candidate.polygon_px,
                candidate.area_rings_px,
                fill_color=QColor(52, 211, 153, 56),
                stroke_color=QColor("#34D399"),
            )
        label_text = "拖框或点已确认面积作为参考"
        if self._reference_instance.busy:
            label_text = "同类扩选推理中..."
        elif self._reference_instance.preview_candidates:
            label_text = f"已找到 {len(self._reference_instance.preview_candidates)} 个候选，按 Enter / F 加入当前类别"
        rect = QRectF(14.0, 14.0, 420.0, 32.0)
        painter.fillRect(rect, QColor(16, 24, 32, 188))
        painter.setPen(QPen(QColor("#FFFFFF"), 1))
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, label_text)

    def _draw_fiber_quick_preview(self, painter: QPainter) -> None:
        if self._image is None:
            return
        self._draw_magic_area_preview(
            painter,
            self._fiber_quick.preview_polygon,
            self._fiber_quick.preview_rings,
            fill_color=QColor(80, 180, 255, 52),
            stroke_color=QColor("#60A5FA"),
        )
        if self._fiber_quick.preview_line is not None:
            painter.setPen(QPen(QColor("#0B0B0B"), 3.2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            painter.drawLine(
                self.image_to_widget(self._fiber_quick.preview_line.start),
                self.image_to_widget(self._fiber_quick.preview_line.end),
            )
            painter.setPen(QPen(QColor("#F4D35E"), 1.8, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            painter.drawLine(
                self.image_to_widget(self._fiber_quick.preview_line.start),
                self.image_to_widget(self._fiber_quick.preview_line.end),
            )
        self._draw_magic_prompt_points(
            painter,
            self._fiber_quick.positive_points,
            QColor("#34D399"),
            positive=True,
        )
        self._draw_magic_prompt_points(
            painter,
            self._fiber_quick.negative_points,
            QColor("#F87171"),
            positive=False,
        )
        self._draw_roi_debug_box(
            painter,
            self._fiber_quick.debug_payload.get("segmentation_crop_box"),
            stroke_color=QColor("#F4D35E" if self._fiber_quick.debug_payload.get("segmentation_pending") else "#60A5FA"),
        )
        prompt_text = "当前提示：负采样点" if self._fiber_quick.prompt_type == "negative" else "当前提示：正采样点"
        label_text = prompt_text
        if self._fiber_quick.segmentation_busy:
            label_text += " / 分割中..."
        elif self._fiber_quick.geometry_busy:
            label_text += " / 测径中..."
        rect = QRectF(14.0, 14.0, 360.0, 32.0)
        painter.fillRect(rect, QColor(16, 24, 32, 188))
        painter.setPen(QPen(QColor("#FFFFFF"), 1))
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, label_text)

    def _draw_magic_prompt_status_label(
        self,
        painter: QPainter,
        *,
        prompt_type: str | None,
        operation_text: str,
        busy: bool,
    ) -> None:
        visual = magic_prompt_visual(prompt_type or "positive") if prompt_type is not None else None
        prefix_text = "当前提示：" if visual is not None else ""
        suffix_text = f"  {operation_text}" if visual is not None else operation_text
        if busy:
            suffix_text += " / 推理中..."

        metrics = painter.fontMetrics()
        label_height = 34.0
        chip_height = 22.0
        outer_padding = 10.0
        gap = 7.0
        chip_padding = 10.0
        prefix_width = float(metrics.horizontalAdvance(prefix_text))
        chip_width = float(metrics.horizontalAdvance(visual.prompt_label)) + chip_padding * 2 if visual is not None else 0.0
        suffix_width = float(metrics.horizontalAdvance(suffix_text))
        desired_width = outer_padding * 2 + suffix_width
        if visual is not None:
            desired_width += prefix_width + gap + chip_width + gap
        max_width = max(180.0, float(self.width()) - 28.0) if self.width() > 0 else desired_width
        label_width = min(max(desired_width, 300.0), max_width)
        rect = QRectF(14.0, 14.0, label_width, label_height)

        painter.save()
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(16, 24, 32, 205))
        painter.drawRoundedRect(rect, 7.0, 7.0)

        x = rect.left() + outer_padding
        if visual is not None:
            prefix_rect = QRectF(x, rect.top(), prefix_width, label_height)
            painter.setPen(QPen(QColor("#F7F4EA"), 1))
            painter.drawText(prefix_rect, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, prefix_text)
            x += prefix_width + gap

            chip_rect = QRectF(x, rect.top() + (label_height - chip_height) / 2, chip_width, chip_height)
            painter.setBrush(QColor(*visual.chip_background))
            painter.setPen(QPen(QColor(visual.chip_border), 1.2))
            painter.drawRoundedRect(chip_rect, 6.0, 6.0)
            painter.setPen(QPen(QColor(visual.chip_text), 1))
            painter.drawText(chip_rect, Qt.AlignmentFlag.AlignCenter, visual.prompt_label)
            x += chip_width + gap

        suffix_rect = QRectF(x, rect.top(), max(0.0, rect.right() - x - outer_padding), label_height)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor("#F7F4EA"), 1))
        painter.drawText(suffix_rect, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, suffix_text)
        painter.restore()

    def _draw_magic_area_preview(
        self,
        painter: QPainter,
        polygon_points: list[Point],
        area_rings: list[list[Point]],
        *,
        fill_color: QColor,
        stroke_color: QColor,
    ) -> None:
        outline_points = polygon_points if len(polygon_points) >= 3 else (area_rings[0] if area_rings else [])
        if len(outline_points) < 3 and not area_rings:
            return
        preview_rings = area_rings or [outline_points]
        path = area_rings_path(preview_rings, self.image_to_widget)
        if self._show_area_fill:
            painter.setBrush(fill_color)
            painter.setPen(Qt.PenStyle.NoPen)
            if path.elementCount() > 0:
                painter.drawPath(path)
            else:
                painter.drawPolygon(QPolygonF([self.image_to_widget(point) for point in outline_points]))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor("#0B0B0B"), 3.2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        for ring in preview_rings:
            if len(ring) < 3:
                continue
            painter.drawPolygon(QPolygonF([self.image_to_widget(point) for point in ring]))
        painter.setPen(QPen(stroke_color, 1.8, Qt.PenStyle.DashLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        for ring in preview_rings:
            if len(ring) < 3:
                continue
            painter.drawPolygon(QPolygonF([self.image_to_widget(point) for point in ring]))

    def _draw_magic_prompt_points(
        self,
        painter: QPainter,
        points: list[Point],
        color: QColor,
        *,
        positive: bool,
    ) -> None:
        for point in points:
            widget_point = self.image_to_widget(point)
            painter.setBrush(QColor("#0B0B0B"))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(widget_point, 5.6, 5.6)
            painter.setBrush(color)
            painter.drawEllipse(widget_point, 3.6, 3.6)
            painter.setPen(QPen(QColor("#FFFFFF"), 1.5))
            painter.drawLine(
                QPointF(widget_point.x() - 2.4, widget_point.y()),
                QPointF(widget_point.x() + 2.4, widget_point.y()),
            )
            if positive:
                painter.drawLine(
                    QPointF(widget_point.x(), widget_point.y() - 2.4),
                    QPointF(widget_point.x(), widget_point.y() + 2.4),
                )

    def _draw_roi_debug_box(self, painter: QPainter, crop_box, *, stroke_color: QColor) -> None:
        if self._image is None or not isinstance(crop_box, (tuple, list)) or len(crop_box) != 4:
            return
        try:
            x0, y0, x1, y1 = (float(crop_box[0]), float(crop_box[1]), float(crop_box[2]), float(crop_box[3]))
        except (TypeError, ValueError):
            return
        rect = QRectF(
            self.image_to_widget(Point(x0, y0)),
            self.image_to_widget(Point(x1, y1)),
        ).normalized()
        if rect.width() < 2.0 or rect.height() < 2.0:
            return
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor("#0B0B0B"), 3.0, Qt.PenStyle.SolidLine))
        painter.drawRect(rect)
        painter.setPen(QPen(stroke_color, 1.4, Qt.PenStyle.DashLine))
        painter.drawRect(rect)

    def _cancel_area_drawing(self) -> None:
        previous_points = list(self._drawing_polygon_points)
        if (
            self._area_hover_point is not None
            and (
                not previous_points
                or self._area_hover_point != previous_points[-1]
            )
        ):
            previous_points.append(self._area_hover_point)
        had_draft = bool(
            previous_points
            or self._drawing_freehand_active
            or self._dragging_area_handle is not None
        )
        document_id = self.document_id
        self._drawing_polygon_points = []
        self._area_hover_point = None
        self._drawing_freehand_active = False
        self._freehand_last_sample_at = 0.0
        self._clear_area_drag_state()
        if had_draft and document_id is not None:
            self.pathSessionChanged.emit(document_id)
        if previous_points:
            self._update_preview_regions(
                previous_points,
                [],
                padding_screen=18.0,
            )

    def _clear_area_drag_state(self) -> None:
        self._dragging_area_handle = None
        self._drag_area_preview_points = None
        self._drag_area_origin_points = None
        self._drag_area_preview_rings = None
        self._drag_area_origin_rings = None
        self._drag_area_press_point = None
        self._drag_area_preview_offset = None

    def _append_freehand_point(self, point: Point) -> None:
        if not self._drawing_polygon_points:
            self._drawing_polygon_points = [point]
            self._freehand_last_sample_at = time.monotonic()
            return
        now = time.monotonic()
        if now - self._freehand_last_sample_at < 0.075:
            return
        if distance(self._drawing_polygon_points[-1], point) < 2.0:
            return
        self._drawing_polygon_points.append(point)
        self._freehand_last_sample_at = now

    def _can_close_polygon_with_point(self, point: Point) -> bool:
        return (
            len(self._drawing_polygon_points) >= 3
            and distance(point, self._drawing_polygon_points[0]) <= self._polygon_close_tolerance()
        )

    def _complete_area_measurement(self, mode: str, polygon_points: list[Point]) -> None:
        document_id = self._document.id if self._document is not None else None
        self._cancel_area_drawing()
        if document_id is None or len(polygon_points) < 3:
            return
        self.lineCommitted.emit(
            document_id,
            mode,
            {
                "measurement_kind": "area",
                "polygon_px": polygon_points,
            },
        )

    def _complete_continuous_measurement(self, polyline_points: list[Point]) -> None:
        document_id = self._document.id if self._document is not None else None
        payload = self._continuous_manual_tool_strategy.commit_payload(polyline_points)
        self._cancel_area_drawing()
        if document_id is None or payload is None:
            return
        self.lineCommitted.emit(
            document_id,
            "continuous_manual",
            payload,
        )

    def _begin_area_drag(self, handle: tuple[str, str, int | None, int | None], image_point: Point) -> None:
        if self._document is None:
            return
        measurement = self._document.get_measurement(handle[0])
        if measurement is None or measurement.measurement_kind != "area":
            return
        self._set_object_selection(CanvasSelectionRef.measurement(measurement.id))
        self._dragging_area_handle = handle
        self._drag_area_press_point = image_point
        self._drag_area_preview_offset = (
            Point(0.0, 0.0)
            if handle[1] == "center"
            else None
        )
        if handle[1] == "center":
            # Whole-object preview reads immutable-for-the-gesture RAW geometry
            # through the derived path cache.  Do not clone dense rings here.
            self._drag_area_origin_rings = None
            self._drag_area_origin_points = None
            self._drag_area_preview_points = None
            self._drag_area_preview_rings = None
            with area_derived_geometry_service.path_render_pass():
                area_derived_geometry_service.raw_path(measurement)
            return
        self._drag_area_origin_rings = self._clone_magic_rings(measurement.area_rings_px) if measurement.area_rings_px else None
        if self._drag_area_origin_rings:
            self._drag_area_origin_points = list(self._drag_area_origin_rings[0])
            self._drag_area_preview_points = list(self._drag_area_origin_rings[0])
            self._drag_area_preview_rings = self._clone_magic_rings(self._drag_area_origin_rings)
        else:
            self._drag_area_origin_points = list(measurement.polygon_px)
            self._drag_area_preview_points = list(measurement.polygon_px)
            self._drag_area_preview_rings = None

    def _emit_magic_segment_session_changed(self, *, repaint: bool = True) -> None:
        if repaint:
            self.update()
        if self._document is not None:
            self.magicSegmentSessionChanged.emit(self._document.id)

    def _update_cursor(self) -> None:
        if self._panning:
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        elif self._roi_capture is not None:
            self.setCursor(Qt.CursorShape.CrossCursor)
        elif self._scale_anchor_pick_active:
            self.setCursor(Qt.CursorShape.CrossCursor)
        elif self._read_only:
            self.setCursor(Qt.CursorShape.OpenHandCursor if self._temporary_grab_active else Qt.CursorShape.ArrowCursor)
        elif self._temporary_grab_active:
            self.setCursor(Qt.CursorShape.OpenHandCursor)
        else:
            self.unsetCursor()
