from __future__ import annotations

from dataclasses import dataclass, field
import time

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QMouseEvent, QPainter, QPen, QPolygonF, QWheelEvent
from PySide6.QtWidgets import QWidget

from fdm.geometry import (
    Line,
    Point,
    clamp,
    distance,
    line_length,
    nearest_endpoint,
    point_in_polygon,
    point_near_bounds,
    point_to_polygon_edge_distance,
    polygon_bounds,
    polygon_translate,
    snap_to_pixel_center,
)
from fdm.models import ImageDocument, OverlayAnnotation, OverlayAnnotationKind
from fdm.services.prompt_segmentation import magic_mask_to_polygon, sanitize_magic_session_mask
from fdm.settings import AppSettings
from fdm.ui.rendering import (
    annotation_rect,
    draw_overlay_annotations,
    draw_measurements,
    draw_preview_scale_anchor,
    overlay_annotation_bounds,
    overlay_annotation_handle_points,
)


class MagicSegmentOperationMode:
    ADD = "add"
    SUBTRACT = "subtract"


@dataclass(slots=True)
class PromptSegmentationSession:
    prompt_type: str = "positive"
    operation_mode: str = MagicSegmentOperationMode.ADD
    positive_points: list[Point] = field(default_factory=list)
    negative_points: list[Point] = field(default_factory=list)
    preview_polygon: list[Point] = field(default_factory=list)
    preview_mask: object | None = None
    last_result_mask: object | None = None
    request_id: int = 0
    busy: bool = False

    def has_points(self) -> bool:
        return bool(self.positive_points or self.negative_points)

    def has_preview(self) -> bool:
        return len(self.preview_polygon) >= 3


class DocumentCanvas(QWidget):
    lineCommitted = Signal(str, str, object)
    measurementSelected = Signal(str, object)
    measurementEdited = Signal(str, str, object)
    overlayCreateRequested = Signal(str, object)
    overlaySelected = Signal(str, object)
    overlayEdited = Signal(str, str, object)
    textPlacementRequested = Signal(str, object)
    textSelected = Signal(str, object)
    textMoved = Signal(str, str, object)
    scaleAnchorPicked = Signal(str, object)
    magicSegmentRequested = Signal(str, object)
    magicSegmentSessionChanged = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._document: ImageDocument | None = None
        self._image: QImage | None = None
        self._tool_mode = "select"
        self._overlay_tool_kind = OverlayAnnotationKind.TEXT
        self._zoom = 1.0
        self._pan = Point(20.0, 20.0)

        self._drawing_anchor_raw: Point | None = None
        self._drawing_line: Line | None = None
        self._line_commit_on_second_click = False

        self._drawing_polygon_points: list[Point] = []
        self._area_hover_point: Point | None = None
        self._drawing_freehand_active = False
        self._freehand_last_sample_at = 0.0

        self._dragging_handle: tuple[str, str] | None = None
        self._drag_preview_line: Line | None = None

        self._dragging_area_handle: tuple[str, str, int | None] | None = None
        self._drag_area_preview_points: list[Point] | None = None
        self._drag_area_origin_points: list[Point] | None = None
        self._drag_area_press_point: Point | None = None

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
        self._space_pressed = False
        self._temporary_grab_active = False

        self._settings = AppSettings()
        self._scale_anchor_pick_active = False
        self._scale_anchor_preview_point: Point | None = None
        self._show_area_fill = True
        self._magic_segment = PromptSegmentationSession()
        self._read_only = False
        self._fit_alignment = "center"

    @property
    def document_id(self) -> str | None:
        return self._document.id if self._document else None

    def set_document(self, document: ImageDocument, image: QImage) -> None:
        self._document = document
        self._image = image
        self._magic_segment = PromptSegmentationSession()
        self._zoom = max(0.05, document.view_state.zoom or 1.0)
        self._pan = Point(document.view_state.pan.x, document.view_state.pan.y)
        if self._zoom == 1.0 and self._pan.x == 0.0 and self._pan.y == 0.0:
            self.fit_to_view()
        self.update()

    def set_image(self, image: QImage) -> None:
        self._image = image
        if self._document is not None:
            self._document.image_size = (image.width(), image.height())
        self.update()

    def clear_document(self) -> None:
        self._document = None
        self._image = None
        self._cancel_line_drawing()
        self._dragging_handle = None
        self._drag_preview_line = None
        self._dragging_area_handle = None
        self._drag_area_preview_points = None
        self._drag_area_origin_points = None
        self._drag_area_press_point = None
        self._drawing_overlay_start = None
        self._drawing_overlay_end = None
        self._dragging_overlay_id = None
        self._dragging_overlay_handle = None
        self._drag_overlay_press_point = None
        self._drag_overlay_origin = None
        self._drag_overlay_preview = None
        self._magic_segment = PromptSegmentationSession()
        self.update()

    def set_read_only(self, read_only: bool) -> None:
        self._read_only = read_only
        if read_only:
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
        if mode != self._tool_mode:
            self._cancel_area_drawing()
            self._cancel_line_drawing()
            if self._tool_mode == "magic_segment" or mode != "magic_segment":
                self.clear_magic_segment_session()
            self._cancel_overlay_interaction()
        self._tool_mode = mode
        if overlay_kind in {
            OverlayAnnotationKind.TEXT,
            OverlayAnnotationKind.RECT,
            OverlayAnnotationKind.CIRCLE,
            OverlayAnnotationKind.LINE,
            OverlayAnnotationKind.ARROW,
        }:
            self._overlay_tool_kind = overlay_kind
        self._update_cursor()
        self.update()

    def set_settings(self, settings: AppSettings) -> None:
        self._settings = settings
        self.update()

    def set_show_area_fill(self, visible: bool) -> None:
        self._show_area_fill = visible
        self.update()

    def current_magic_segment_prompt_type(self) -> str:
        return self._magic_segment.prompt_type

    def current_magic_segment_operation_mode(self) -> str:
        return self._magic_segment.operation_mode

    def has_magic_segment_session(self) -> bool:
        return bool(
            self._magic_segment.has_points()
            or self._magic_segment.has_preview()
            or self._magic_segment.preview_mask is not None
            or self._magic_segment.last_result_mask is not None
        )

    def has_magic_segment_preview(self) -> bool:
        return self._magic_segment.has_preview()

    def is_magic_segment_busy(self) -> bool:
        return self._magic_segment.busy

    def set_magic_segment_prompt_type(self, prompt_type: str) -> None:
        self._magic_segment.prompt_type = "negative" if prompt_type == "negative" else "positive"
        self._emit_magic_segment_session_changed()

    def cycle_magic_segment_prompt_type(self) -> str:
        self._magic_segment.prompt_type = "negative" if self._magic_segment.prompt_type == "positive" else "positive"
        self._emit_magic_segment_session_changed()
        return self._magic_segment.prompt_type

    def cycle_magic_segment_operation_mode(self) -> str:
        if self._magic_segment.last_result_mask is not None or self._magic_segment.has_points():
            self._commit_magic_segment_candidate()
        self._magic_segment.operation_mode = (
            MagicSegmentOperationMode.SUBTRACT
            if self._magic_segment.operation_mode == MagicSegmentOperationMode.ADD
            else MagicSegmentOperationMode.ADD
        )
        self._emit_magic_segment_session_changed()
        return self._magic_segment.operation_mode

    def apply_magic_segment_result(self, request_id: int, mask) -> dict[str, object] | None:
        if request_id != self._magic_segment.request_id:
            return None
        self._magic_segment.busy = False
        self._magic_segment.last_result_mask = self._clone_magic_mask(mask)
        stats = self._refresh_magic_segment_preview()
        self._emit_magic_segment_session_changed()
        return stats

    def fail_magic_segment_result(self, request_id: int) -> None:
        if request_id != self._magic_segment.request_id:
            return
        self._magic_segment.busy = False
        self._magic_segment.last_result_mask = None
        self._refresh_magic_segment_preview()
        self._emit_magic_segment_session_changed()

    def clear_magic_segment_session(self) -> None:
        self._magic_segment = PromptSegmentationSession()
        self._emit_magic_segment_session_changed()

    def commit_magic_segment_preview(self) -> bool:
        document_id = self._document.id if self._document is not None else None
        self._commit_magic_segment_candidate()
        polygon_points = list(self._magic_segment.preview_polygon)
        self.clear_magic_segment_session()
        if document_id is None or len(polygon_points) < 3:
            return False
        self.lineCommitted.emit(
            document_id,
            "magic_segment",
            {
                "measurement_kind": "area",
                "polygon_px": polygon_points,
            },
        )
        return True

    def _commit_magic_segment_candidate(self) -> None:
        effective_mask = self._effective_magic_segment_mask()
        sanitized_mask, _stats = sanitize_magic_session_mask(effective_mask)
        self._magic_segment.preview_mask = self._clone_magic_mask(sanitized_mask)
        self._magic_segment.last_result_mask = None
        self._magic_segment.positive_points.clear()
        self._magic_segment.negative_points.clear()
        self._refresh_magic_segment_preview()

    def _effective_magic_segment_mask(self):
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover - dependency is required by the app
            raise RuntimeError("numpy is required for the magic segmentation tool.") from exc
        base_mask = self._magic_segment.preview_mask
        candidate_mask = self._magic_segment.last_result_mask
        if base_mask is None and candidate_mask is None:
            return None
        if candidate_mask is None:
            return self._clone_magic_mask(base_mask)
        candidate = np.asarray(candidate_mask, dtype=bool)
        if base_mask is None:
            if self._magic_segment.operation_mode == MagicSegmentOperationMode.SUBTRACT:
                return np.zeros(candidate.shape, dtype=bool)
            return candidate.copy()
        base = np.asarray(base_mask, dtype=bool)
        if self._magic_segment.operation_mode == MagicSegmentOperationMode.SUBTRACT:
            return base & ~candidate
        return base | candidate

    def _refresh_magic_segment_preview(self) -> dict[str, object]:
        effective_mask = self._effective_magic_segment_mask()
        sanitized_mask, stats = sanitize_magic_session_mask(effective_mask)
        self._magic_segment.preview_polygon = magic_mask_to_polygon(sanitized_mask)
        return stats

    def _clone_magic_mask(self, mask):
        if mask is None:
            return None
        return mask.copy()

    def set_selected_measurement(self, measurement_id: str | None) -> None:
        if self._document is None:
            return
        self._document.select_measurement(measurement_id)
        self.update()

    def set_selected_overlay_annotation(self, overlay_id: str | None) -> None:
        if self._document is None:
            return
        self._document.select_overlay_annotation(overlay_id)
        self.update()

    def set_selected_text_annotation(self, text_id: str | None) -> None:
        if self._document is None:
            return
        self._document.select_text_annotation(text_id)
        self.update()

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

    def fit_to_view(self) -> None:
        if self._image is None:
            return
        viewport_width = max(100, self.width() - 40)
        viewport_height = max(100, self.height() - 40)
        zoom_x = viewport_width / self._image.width()
        zoom_y = viewport_height / self._image.height()
        self._zoom = max(0.05, min(zoom_x, zoom_y))
        target_width = self._image.width() * self._zoom
        target_height = self._image.height() * self._zoom
        if self._fit_alignment == "top_left":
            self._pan = Point(20.0, 20.0)
        else:
            self._pan = Point(
                (self.width() - target_width) / 2.0,
                (self.height() - target_height) / 2.0,
            )
        self._persist_view_state()
        self.update()

    def actual_size(self) -> None:
        self._zoom = 1.0
        self._pan = Point(20.0, 20.0)
        self._persist_view_state()
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
        if event.key() == Qt.Key.Key_Escape and self._tool_mode == "magic_segment" and self.has_magic_segment_session():
            self.clear_magic_segment_session()
            event.accept()
            return
        if event.key() == Qt.Key.Key_Escape and self._drawing_anchor_raw is not None:
            self._cancel_line_drawing()
            self.update()
            event.accept()
            return
        if event.key() == Qt.Key.Key_Escape and (self._drawing_polygon_points or self._drawing_freehand_active):
            self._cancel_area_drawing()
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
        painter.fillRect(self.rect(), QColor("#101820"))
        if self._image is None or self._document is None:
            painter.setPen(QColor("#F2F2F2"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "打开图片后开始测量")
            return

        target = QRectF(
            self._pan.x,
            self._pan.y,
            self._image.width() * self._zoom,
            self._image.height() * self._zoom,
        )
        painter.drawImage(target, self._image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self._draw_annotations(painter)
        self._draw_preview(painter)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._image is not None and self._document is not None and self._document.view_state.zoom == 1.0:
            self.fit_to_view()

    def wheelEvent(self, event: QWheelEvent) -> None:
        if self._image is None:
            return
        delta_y = event.angleDelta().y()
        delta_x = event.angleDelta().x()
        effective_delta = delta_y if delta_y != 0 else delta_x
        if effective_delta == 0:
            return
        cursor_position = event.position()
        image_before = self.widget_to_image(cursor_position)
        zoom_factor = 1.15 if effective_delta > 0 else 1 / 1.15
        self._zoom = max(0.05, min(40.0, self._zoom * zoom_factor))
        self._pan = Point(
            cursor_position.x() - (image_before.x * self._zoom),
            cursor_position.y() - (image_before.y * self._zoom),
        )
        self._persist_view_state()
        self.update()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if self._image is None or self._document is None:
            return
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        self._last_mouse_pos = event.position()
        if event.button() in (Qt.MouseButton.MiddleButton, Qt.MouseButton.RightButton):
            self._panning = True
            self._pan_button = event.button()
            self._update_cursor()
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return
        if self._temporary_grab_active:
            self._panning = True
            self._pan_button = event.button()
            self._update_cursor()
            return
        if self._read_only:
            return

        image_point = self.widget_to_image(event.position())

        if self._scale_anchor_pick_active:
            if self._point_in_image(image_point):
                self._scale_anchor_preview_point = self._clamp_to_image(image_point, pixel_center=False)
                self.scaleAnchorPicked.emit(self._document.id, self._scale_anchor_preview_point)
            return

        if self._tool_mode == "magic_segment":
            if not self._point_in_image(image_point):
                return
            self._document.select_measurement(None)
            self._document.select_overlay_annotation(None)
            self.measurementSelected.emit(self._document.id, "")
            self.overlaySelected.emit(self._document.id, "")
            self.textSelected.emit(self._document.id, "")
            point = self._clamp_to_image(image_point, pixel_center=False)
            if self._magic_segment.prompt_type == "negative":
                self._magic_segment.negative_points.append(point)
            else:
                self._magic_segment.positive_points.append(point)
            self._magic_segment.request_id += 1
            self._magic_segment.busy = True
            self.magicSegmentRequested.emit(
                self._document.id,
                {
                    "request_id": self._magic_segment.request_id,
                    "positive_points": list(self._magic_segment.positive_points),
                    "negative_points": list(self._magic_segment.negative_points),
                },
            )
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
                    self._document.select_measurement(None)
                    self._document.select_overlay_annotation(None)
                    self.measurementSelected.emit(self._document.id, "")
                    self.overlaySelected.emit(self._document.id, "")
                    self.textSelected.emit(self._document.id, "")
                    self.update()
            return

        if self._tool_mode == "polygon_area":
            if not self._point_in_image(image_point):
                return
            point = self._clamp_to_image(image_point, pixel_center=False)
            if self._can_close_polygon_with_point(point):
                self._complete_area_measurement("polygon_area", list(self._drawing_polygon_points))
                return
            self._drawing_polygon_points.append(point)
            self._area_hover_point = point
            self.update()
            return

        if self._tool_mode == "freehand_area":
            if not self._point_in_image(image_point):
                return
            point = self._clamp_to_image(image_point, pixel_center=False)
            self._drawing_polygon_points = [point]
            self._area_hover_point = point
            self._drawing_freehand_active = True
            self._freehand_last_sample_at = time.monotonic()
            self.update()
            return

        if self._tool_mode == "calibration":
            if self._point_in_image(image_point):
                anchor = self._anchor_point_for_event(image_point, event.modifiers())
                self._begin_line_drawing(anchor)
                self.update()
            return

        if self._tool_mode == "snap":
            if self._drawing_anchor_raw is not None and self._line_commit_on_second_click:
                self._commit_click_line(image_point, event.modifiers())
                return
            if self._point_in_image(image_point):
                anchor = self._anchor_point_for_event(image_point, event.modifiers())
                self._begin_line_drawing(anchor, commit_on_second_click=True)
                self.update()
            return

        if self._tool_mode == "select":
            overlay_handle = self._hit_test_selected_overlay_handle(image_point)
            if overlay_handle is not None:
                annotation = self._document.get_overlay_annotation(overlay_handle[0])
                if annotation is not None:
                    self._document.select_overlay_annotation(annotation.id)
                    self.overlaySelected.emit(self._document.id, annotation.id)
                    if annotation.normalized_kind() == OverlayAnnotationKind.TEXT:
                        self.textSelected.emit(self._document.id, annotation.id)
                    else:
                        self.textSelected.emit(self._document.id, "")
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
                    self._document.select_overlay_annotation(annotation.id)
                    self.overlaySelected.emit(self._document.id, annotation.id)
                    if annotation.normalized_kind() == OverlayAnnotationKind.TEXT:
                        self.textSelected.emit(self._document.id, annotation.id)
                    else:
                        self.textSelected.emit(self._document.id, "")
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
                self._document.select_measurement(area_measurement_id)
                self.measurementSelected.emit(self._document.id, area_measurement_id)
                self.overlaySelected.emit(self._document.id, "")
                self.textSelected.emit(self._document.id, "")
                self.update()
                return

            handle = self._hit_test_endpoint(image_point)
            if handle is not None:
                self._dragging_handle = handle
                self._drag_preview_line = self._measurement_line(handle[0])
                self._document.select_measurement(handle[0])
                self.measurementSelected.emit(self._document.id, handle[0])
                self.overlaySelected.emit(self._document.id, "")
                self.update()
                return

            measurement_id = self._hit_test_measurement(image_point)
            self._document.select_measurement(measurement_id)
            self.measurementSelected.emit(self._document.id, measurement_id or "")
            self.overlaySelected.emit(self._document.id, "")
            self.textSelected.emit(self._document.id, "")
            self.update()
            return

        if self._point_in_image(image_point):
            anchor = self._anchor_point_for_event(image_point, event.modifiers())
            self._begin_line_drawing(anchor)
            self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._image is None or self._document is None:
            return
        if self._panning:
            delta = event.position() - self._last_mouse_pos
            self._pan = Point(self._pan.x + delta.x(), self._pan.y + delta.y())
            self._last_mouse_pos = event.position()
            self._persist_view_state()
            self.update()
            return

        if self._scale_anchor_pick_active:
            self._scale_anchor_preview_point = self._clamp_to_image(self.widget_to_image(event.position()), pixel_center=False)
            self.update()
            return
        if self._read_only:
            return

        image_point = self.widget_to_image(event.position())

        if self._tool_mode == "polygon_area" and self._drawing_polygon_points:
            self._area_hover_point = self._clamp_to_image(image_point, pixel_center=False)
            self.update()
            return

        if self._drawing_freehand_active:
            self._append_freehand_point(self._clamp_to_image(image_point, pixel_center=False))
            self._area_hover_point = self._drawing_polygon_points[-1] if self._drawing_polygon_points else None
            self.update()
            return

        if self._drawing_anchor_raw is not None:
            start, end = self._apply_line_constraints(
                self._drawing_anchor_raw,
                image_point,
                event.modifiers(),
                snap_anchor=True,
            )
            self._drawing_line = Line(start=start, end=end)
            self.update()
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
            and self._drag_area_origin_points is not None
            and self._drag_area_press_point is not None
        ):
            _measurement_id, handle_kind, point_index = self._dragging_area_handle
            if handle_kind == "center":
                dx = image_point.x - self._drag_area_press_point.x
                dy = image_point.y - self._drag_area_press_point.y
                self._drag_area_preview_points = polygon_translate(self._drag_area_origin_points, dx, dy)
            elif point_index is not None:
                preview = list(self._drag_area_origin_points)
                preview[point_index] = self._clamp_to_image(image_point, pixel_center=False)
                self._drag_area_preview_points = preview
            self.update()
            return

        if self._dragging_handle is not None:
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
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._document is None:
            return
        if self._panning and self._pan_button == event.button():
            self._panning = False
            self._pan_button = None
            if self._space_pressed and not self._has_pointer_edit_operation():
                self._temporary_grab_active = True
            elif not self._space_pressed:
                self._temporary_grab_active = False
            self._update_cursor()
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return
        if self._read_only:
            self._update_cursor()
            return

        if self._drawing_freehand_active:
            polygon_points = list(self._drawing_polygon_points)
            self._cancel_area_drawing()
            if len(polygon_points) >= 3:
                self._complete_area_measurement("freehand_area", polygon_points)
            return

        if self._drawing_line is not None:
            if self._line_commit_on_second_click:
                self.update()
                return
            line = self._drawing_line
            self._cancel_line_drawing()
            if line_length(line) >= 1.0:
                self.lineCommitted.emit(self._document.id, self._tool_mode, line)
            if self._space_pressed:
                self._temporary_grab_active = True
                self._update_cursor()
            self.update()
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

        if self._dragging_area_handle is not None and self._drag_area_preview_points is not None:
            measurement_id, _handle_kind, _index = self._dragging_area_handle
            preview = list(self._drag_area_preview_points)
            self._dragging_area_handle = None
            self._drag_area_preview_points = None
            self._drag_area_origin_points = None
            self._drag_area_press_point = None
            self.measurementEdited.emit(
                self._document.id,
                measurement_id,
                {"measurement_kind": "area", "polygon_px": preview},
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
        self._dragging_area_handle = None
        self._drag_area_preview_points = None
        self._drag_area_origin_points = None
        self._drag_area_press_point = None
        self._update_cursor()

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        if self._image is None or self._document is None:
            return
        if self._read_only:
            return
        if event.button() == Qt.MouseButton.LeftButton and self._tool_mode == "polygon_area" and len(self._drawing_polygon_points) >= 2:
            point = self._clamp_to_image(self.widget_to_image(event.position()), pixel_center=False)
            if distance(point, self._drawing_polygon_points[-1]) > 1.0:
                self._drawing_polygon_points.append(point)
            if len(self._drawing_polygon_points) >= 3:
                self._complete_area_measurement("polygon_area", list(self._drawing_polygon_points))
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

    def _draw_annotations(self, painter: QPainter) -> None:
        if self._document is None:
            return
        draw_measurements(
            painter,
            self._document,
            self.image_to_widget,
            self._settings,
            line_width=2.0,
            endpoint_radius=4.0,
            selected_measurement_id=self._document.view_state.selected_measurement_id,
            show_area_fill=self._show_area_fill,
            show_area_handles=self._tool_mode == "select",
        )
        draw_overlay_annotations(
            painter,
            self._document,
            self.image_to_widget,
            self._settings,
            selected_overlay_id=self._document.selected_overlay_id,
            show_handles=self._tool_mode == "select",
            render_mode="screen_scale_full_image",
        )

    def _draw_preview(self, painter: QPainter) -> None:
        preview_line = self._drag_preview_line or self._drawing_line
        if preview_line is not None:
            color = QColor("#FF7F50") if self._tool_mode == "calibration" else QColor("#F4D35E")
            painter.setPen(QPen(color, 2, Qt.PenStyle.DashLine))
            painter.drawLine(self.image_to_widget(preview_line.start), self.image_to_widget(preview_line.end))
            painter.setBrush(color)
            painter.setPen(QPen(QColor("#0B0B0B"), 1))
            painter.drawEllipse(self.image_to_widget(preview_line.start), 6, 6)
            painter.drawEllipse(self.image_to_widget(preview_line.end), 6, 6)

        preview_points = self._drag_area_preview_points or self._drawing_polygon_points
        if preview_points:
            polygon = QPolygonF([self.image_to_widget(point) for point in preview_points])
            if self._show_area_fill and len(preview_points) >= 3:
                painter.setBrush(QColor(244, 211, 94, 56))
            else:
                painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor("#0B0B0B"), 3, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
            if len(preview_points) >= 3 and (self._drag_area_preview_points is not None or self._drawing_freehand_active):
                painter.drawPolygon(polygon)
            else:
                painter.drawPolyline(polygon)
            painter.setPen(QPen(QColor("#F4D35E"), 1.8, Qt.PenStyle.DashLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
            if len(preview_points) >= 3 and (self._drag_area_preview_points is not None or self._drawing_freehand_active):
                painter.drawPolygon(polygon)
            else:
                painter.drawPolyline(polygon)
            painter.setBrush(QColor("#F4D35E"))
            painter.setPen(QPen(QColor("#0B0B0B"), 1))
            for point in preview_points:
                painter.drawEllipse(self.image_to_widget(point), 4.5, 4.5)
            if self._tool_mode == "polygon_area" and self._area_hover_point is not None and self._drawing_polygon_points:
                painter.setPen(QPen(QColor("#F4D35E"), 1.2, Qt.PenStyle.DashLine))
                painter.drawLine(self.image_to_widget(self._drawing_polygon_points[-1]), self.image_to_widget(self._area_hover_point))
                if len(self._drawing_polygon_points) >= 2:
                    painter.drawLine(self.image_to_widget(self._area_hover_point), self.image_to_widget(self._drawing_polygon_points[0]))

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

        if self._tool_mode == "magic_segment":
            self._draw_magic_segment_preview(painter)

        if self._scale_anchor_pick_active:
            preview_point = self._scale_anchor_preview_point or Point(self._image.width() * 0.15, self._image.height() * 0.2)
            draw_preview_scale_anchor(painter, self.image_to_widget(preview_point))

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
        for measurement in reversed(self._document.measurements):
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

    def _hit_test_selected_area_handle(self, image_point: Point) -> tuple[str, str, int | None] | None:
        if self._document is None or self._document.view_state.selected_measurement_id is None:
            return None
        measurement = self._document.get_measurement(self._document.view_state.selected_measurement_id)
        if measurement is None or measurement.measurement_kind != "area" or len(measurement.polygon_px) < 3:
            return None
        nearest_vertex: tuple[int, float] | None = None
        for index, point in enumerate(measurement.polygon_px):
            vertex_distance = distance(point, image_point)
            if vertex_distance <= self._selected_endpoint_tolerance():
                if nearest_vertex is None or vertex_distance < nearest_vertex[1]:
                    nearest_vertex = (index, vertex_distance)
        if nearest_vertex is not None:
            return measurement.id, "vertex", nearest_vertex[0]
        center = measurement.polygon_center()
        if distance(center, image_point) <= max(3.0, 5.0 / max(self._zoom, 0.001)):
            return measurement.id, "center", None
        return None

    def _hit_test_area_measurement(self, image_point: Point) -> str | None:
        if self._document is None:
            return None
        tolerance = max(5.0, 10.0 / max(self._zoom, 0.001))
        for measurement in reversed(self._document.measurements):
            if measurement.measurement_kind != "area" or len(measurement.polygon_px) < 3:
                continue
            bounds = polygon_bounds(measurement.polygon_px)
            if not point_near_bounds(image_point, bounds, tolerance):
                continue
            if point_in_polygon(image_point, measurement.polygon_px):
                return measurement.id
            if point_to_polygon_edge_distance(image_point, measurement.polygon_px) <= tolerance:
                return measurement.id
        return None

    def _hit_test_measurement(self, image_point: Point) -> str | None:
        if self._document is None:
            return None
        tolerance = max(5.0, 10.0 / max(self._zoom, 0.001))
        for measurement in reversed(self._document.measurements):
            if measurement.measurement_kind != "line":
                continue
            line = measurement.effective_line()
            bounds = (min(line.start.x, line.end.x), min(line.start.y, line.end.y),
                      max(line.start.x, line.end.x), max(line.start.y, line.end.y))
            if not point_near_bounds(image_point, bounds, tolerance):
                continue
            if self._point_to_segment_distance(image_point, line) <= tolerance:
                return measurement.id
        return None

    def _selected_endpoint_tolerance(self) -> float:
        return max(4.0, 9.0 / max(self._zoom, 0.001))

    def _endpoint_tolerance(self) -> float:
        return max(3.0, 6.0 / max(self._zoom, 0.001))

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
            return annotation.clone(anchor_px=self._clamp_to_image(annotation.anchor_px, pixel_center=False))
        return annotation.clone(
            start_px=self._clamp_to_image(annotation.start_px, pixel_center=False),
            end_px=self._clamp_to_image(annotation.end_px, pixel_center=False),
        )

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
        fixed = snap_to_pixel_center(anchor) if use_ctrl and snap_anchor else anchor
        moving = candidate
        if use_shift:
            dx = moving.x - fixed.x
            dy = moving.y - fixed.y
            if abs(dx) >= abs(dy):
                moving = Point(moving.x, fixed.y)
            else:
                moving = Point(fixed.x, moving.y)
        if use_ctrl:
            moving = snap_to_pixel_center(moving)
        fixed = self._clamp_to_image(fixed, pixel_center=use_ctrl and snap_anchor)
        moving = self._clamp_to_image(moving, pixel_center=use_ctrl)
        return fixed, moving

    def _anchor_point_for_event(self, image_point: Point, modifiers: Qt.KeyboardModifiers) -> Point:
        use_ctrl = bool(modifiers & Qt.KeyboardModifier.ControlModifier)
        return self._clamp_to_image(
            snap_to_pixel_center(image_point) if use_ctrl else image_point,
            pixel_center=use_ctrl,
        )

    def _begin_line_drawing(self, anchor: Point, *, commit_on_second_click: bool = False) -> None:
        self._drawing_anchor_raw = anchor
        self._drawing_line = Line(start=anchor, end=anchor)
        self._line_commit_on_second_click = commit_on_second_click

    def _cancel_line_drawing(self) -> None:
        self._drawing_anchor_raw = None
        self._drawing_line = None
        self._line_commit_on_second_click = False

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
        if line_length(line) >= 1.0:
            self.lineCommitted.emit(self._document.id, self._tool_mode, line)
        if self._space_pressed:
            self._temporary_grab_active = True
            self._update_cursor()
        self.update()

    def _clamp_to_image(self, point: Point, *, pixel_center: bool) -> Point:
        if self._image is None:
            return point
        minimum = 0.5 if pixel_center else 0.0
        maximum_x = (self._image.width() - 0.5) if pixel_center else (self._image.width() - 1.0)
        maximum_y = (self._image.height() - 0.5) if pixel_center else (self._image.height() - 1.0)
        return Point(
            x=clamp(point.x, minimum, max(minimum, maximum_x)),
            y=clamp(point.y, minimum, max(minimum, maximum_y)),
        )

    def _persist_view_state(self) -> None:
        if self._document is None:
            return
        self._document.view_state.zoom = self._zoom
        self._document.view_state.pan = Point(self._pan.x, self._pan.y)

    def _has_pointer_edit_operation(self) -> bool:
        return (
            self._drawing_anchor_raw is not None
            or bool(self._drawing_polygon_points)
            or self._drawing_freehand_active
            or self._dragging_handle is not None
            or self._dragging_area_handle is not None
            or self._drawing_overlay_start is not None
            or self._dragging_overlay_id is not None
            or self._dragging_overlay_handle is not None
            or self._scale_anchor_pick_active
        )

    def _draw_magic_segment_preview(self, painter: QPainter) -> None:
        if self._image is None:
            return
        if len(self._magic_segment.preview_polygon) >= 3:
            polygon = QPolygonF([self.image_to_widget(point) for point in self._magic_segment.preview_polygon])
            if self._show_area_fill:
                painter.setBrush(QColor(52, 211, 153, 72))
            else:
                painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor("#0B0B0B"), 3.2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
            painter.drawPolygon(polygon)
            painter.setPen(QPen(QColor("#34D399"), 1.8, Qt.PenStyle.DashLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
            painter.drawPolygon(polygon)

        self._draw_magic_prompt_points(painter, self._magic_segment.positive_points, QColor("#34D399"), positive=True)
        self._draw_magic_prompt_points(painter, self._magic_segment.negative_points, QColor("#F87171"), positive=False)

        prompt_text = "当前提示：负采样点" if self._magic_segment.prompt_type == "negative" else "当前提示：正采样点"
        operation_text = "当前模式：剔除" if self._magic_segment.operation_mode == MagicSegmentOperationMode.SUBTRACT else "当前模式：添加"
        if self._magic_segment.busy:
            prompt_text += " / 推理中..."
        rect = QRectF(14.0, 14.0, 340.0, 32.0)
        painter.fillRect(rect, QColor(16, 24, 32, 188))
        painter.setPen(QPen(QColor("#FFFFFF"), 1))
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, f"{prompt_text}  {operation_text}")

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

    def _cancel_area_drawing(self) -> None:
        self._drawing_polygon_points = []
        self._area_hover_point = None
        self._drawing_freehand_active = False
        self._freehand_last_sample_at = 0.0
        self._dragging_area_handle = None
        self._drag_area_preview_points = None
        self._drag_area_origin_points = None
        self._drag_area_press_point = None
        self.update()

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
        return len(self._drawing_polygon_points) >= 3 and distance(point, self._drawing_polygon_points[0]) <= self._selected_endpoint_tolerance()

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

    def _begin_area_drag(self, handle: tuple[str, str, int | None], image_point: Point) -> None:
        if self._document is None:
            return
        measurement = self._document.get_measurement(handle[0])
        if measurement is None or measurement.measurement_kind != "area":
            return
        self._document.select_measurement(measurement.id)
        self.measurementSelected.emit(self._document.id, measurement.id)
        self._dragging_area_handle = handle
        self._drag_area_origin_points = list(measurement.polygon_px)
        self._drag_area_preview_points = list(measurement.polygon_px)
        self._drag_area_press_point = image_point

    def _emit_magic_segment_session_changed(self) -> None:
        self.update()
        if self._document is not None:
            self.magicSegmentSessionChanged.emit(self._document.id)

    def _update_cursor(self) -> None:
        if self._panning:
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        elif self._scale_anchor_pick_active:
            self.setCursor(Qt.CursorShape.CrossCursor)
        elif self._read_only:
            self.setCursor(Qt.CursorShape.OpenHandCursor if self._temporary_grab_active else Qt.CursorShape.ArrowCursor)
        elif self._temporary_grab_active:
            self.setCursor(Qt.CursorShape.OpenHandCursor)
        else:
            self.unsetCursor()
