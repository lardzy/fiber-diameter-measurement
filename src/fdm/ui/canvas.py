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
    point_to_polygon_edge_distance,
    polygon_translate,
    snap_to_pixel_center,
)
from fdm.models import ImageDocument
from fdm.settings import AppSettings
from fdm.ui.rendering import (
    annotation_rect,
    draw_measurements,
    draw_preview_scale_anchor,
    draw_text_annotations,
)


@dataclass(slots=True)
class PromptSegmentationSession:
    prompt_type: str = "positive"
    positive_points: list[Point] = field(default_factory=list)
    negative_points: list[Point] = field(default_factory=list)
    preview_polygon: list[Point] = field(default_factory=list)
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
        self._zoom = 1.0
        self._pan = Point(20.0, 20.0)

        self._drawing_anchor_raw: Point | None = None
        self._drawing_line: Line | None = None

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

        self._dragging_text_id: str | None = None
        self._drag_text_offset = Point(0.0, 0.0)
        self._drag_text_preview_anchor: Point | None = None

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

    def set_tool_mode(self, mode: str) -> None:
        if mode != self._tool_mode:
            self._cancel_area_drawing()
            if self._tool_mode == "magic_segment" or mode != "magic_segment":
                self.clear_magic_segment_session()
        self._tool_mode = mode
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

    def has_magic_segment_session(self) -> bool:
        return self._magic_segment.has_points() or self._magic_segment.has_preview()

    def has_magic_segment_preview(self) -> bool:
        return self._magic_segment.has_preview()

    def set_magic_segment_prompt_type(self, prompt_type: str) -> None:
        self._magic_segment.prompt_type = "negative" if prompt_type == "negative" else "positive"
        self._emit_magic_segment_session_changed()

    def cycle_magic_segment_prompt_type(self) -> str:
        self._magic_segment.prompt_type = "negative" if self._magic_segment.prompt_type == "positive" else "positive"
        self._emit_magic_segment_session_changed()
        return self._magic_segment.prompt_type

    def apply_magic_segment_result(self, request_id: int, polygon_points: list[Point]) -> None:
        if request_id != self._magic_segment.request_id:
            return
        self._magic_segment.busy = False
        self._magic_segment.preview_polygon = list(polygon_points)
        self._emit_magic_segment_session_changed()

    def fail_magic_segment_result(self, request_id: int) -> None:
        if request_id != self._magic_segment.request_id:
            return
        self._magic_segment.busy = False
        self._magic_segment.preview_polygon = []
        self._emit_magic_segment_session_changed()

    def clear_magic_segment_session(self) -> None:
        self._magic_segment = PromptSegmentationSession()
        self._emit_magic_segment_session_changed()

    def commit_magic_segment_preview(self) -> bool:
        document_id = self._document.id if self._document is not None else None
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

    def set_selected_measurement(self, measurement_id: str | None) -> None:
        if self._document is None:
            return
        self._document.select_measurement(measurement_id)
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
            self._document.select_text_annotation(None)
            self.measurementSelected.emit(self._document.id, "")
            self.textSelected.emit(self._document.id, "")
            point = self._clamp_to_image(image_point, pixel_center=False)
            if self._magic_segment.prompt_type == "negative":
                self._magic_segment.negative_points.append(point)
            else:
                self._magic_segment.positive_points.append(point)
            self._magic_segment.request_id += 1
            self._magic_segment.busy = True
            self._magic_segment.preview_polygon = []
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

        if self._tool_mode == "text":
            if self._point_in_image(image_point):
                self.textPlacementRequested.emit(
                    self._document.id,
                    self._clamp_to_image(image_point, pixel_center=False),
                )
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
                anchor = self._clamp_to_image(
                    snap_to_pixel_center(image_point) if event.modifiers() & Qt.KeyboardModifier.ControlModifier else image_point,
                    pixel_center=bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier),
                )
                self._drawing_anchor_raw = anchor
                self._drawing_line = Line(start=anchor, end=anchor)
                self.update()
            return

        if self._tool_mode == "select":
            text_hit = self._hit_test_text_annotation(event.position())
            if text_hit is not None:
                annotation = self._document.get_text_annotation(text_hit)
                if annotation is not None:
                    self._document.select_text_annotation(text_hit)
                    self.textSelected.emit(self._document.id, text_hit)
                    self._dragging_text_id = text_hit
                    self._drag_text_offset = Point(
                        image_point.x - annotation.anchor_px.x,
                        image_point.y - annotation.anchor_px.y,
                    )
                    self._drag_text_preview_anchor = annotation.anchor_px
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
                self.textSelected.emit(self._document.id, "")
                self.update()
                return

            handle = self._hit_test_endpoint(image_point)
            if handle is not None:
                self._dragging_handle = handle
                self._drag_preview_line = self._measurement_line(handle[0])
                self._document.select_measurement(handle[0])
                self.measurementSelected.emit(self._document.id, handle[0])
                self.update()
                return

            measurement_id = self._hit_test_measurement(image_point)
            self._document.select_measurement(measurement_id)
            self.measurementSelected.emit(self._document.id, measurement_id or "")
            if measurement_id is None:
                self.textSelected.emit(self._document.id, "")
            self.update()
            return

        if self._point_in_image(image_point):
            anchor = self._clamp_to_image(
                snap_to_pixel_center(image_point) if event.modifiers() & Qt.KeyboardModifier.ControlModifier else image_point,
                pixel_center=bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier),
            )
            self._drawing_anchor_raw = anchor
            self._drawing_line = Line(start=anchor, end=anchor)
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

        if self._dragging_text_id is not None:
            self._drag_text_preview_anchor = self._clamp_to_image(
                Point(
                    image_point.x - self._drag_text_offset.x,
                    image_point.y - self._drag_text_offset.y,
                ),
                pixel_center=False,
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

        if self._drawing_freehand_active:
            polygon_points = list(self._drawing_polygon_points)
            self._cancel_area_drawing()
            if len(polygon_points) >= 3:
                self._complete_area_measurement("freehand_area", polygon_points)
            return

        if self._drawing_line is not None:
            line = self._drawing_line
            self._drawing_anchor_raw = None
            self._drawing_line = None
            if line_length(line) >= 1.0:
                self.lineCommitted.emit(self._document.id, self._tool_mode, line)
            if self._space_pressed:
                self._temporary_grab_active = True
                self._update_cursor()
            self.update()
            return

        if self._dragging_text_id is not None and self._drag_text_preview_anchor is not None:
            text_id = self._dragging_text_id
            preview_anchor = self._drag_text_preview_anchor
            self._dragging_text_id = None
            self._drag_text_preview_anchor = None
            self.textMoved.emit(self._document.id, text_id, preview_anchor)
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
        self._dragging_area_handle = None
        self._drag_area_preview_points = None
        self._drag_area_origin_points = None
        self._drag_area_press_point = None
        self._update_cursor()

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        if self._image is None or self._document is None:
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
        draw_text_annotations(
            painter,
            self._document,
            self.image_to_widget,
            self._settings,
            selected_text_id=self._document.selected_text_id,
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

        if self._dragging_text_id is not None and self._drag_text_preview_anchor is not None:
            annotation = self._document.get_text_annotation(self._dragging_text_id) if self._document else None
            if annotation is not None:
                preview_annotation = type(annotation)(
                    id=annotation.id,
                    image_id=annotation.image_id,
                    content=annotation.content,
                    anchor_px=self._drag_text_preview_anchor,
                    created_at=annotation.created_at,
                )
                draw_text_annotations(
                    painter,
                    type("PreviewDoc", (), {"text_annotations": [preview_annotation], "selected_text_id": annotation.id})(),
                    self.image_to_widget,
                    self._settings,
                    selected_text_id=annotation.id,
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
        for measurement in reversed(self._document.measurements):
            if measurement.measurement_kind != "line":
                continue
            endpoint_name, endpoint_distance = nearest_endpoint(measurement.effective_line(), image_point)
            if endpoint_distance <= self._endpoint_tolerance():
                return measurement.id, endpoint_name
        return None

    def _hit_test_selected_area_handle(self, image_point: Point) -> tuple[str, str, int | None] | None:
        if self._document is None or self._document.view_state.selected_measurement_id is None:
            return None
        measurement = self._document.get_measurement(self._document.view_state.selected_measurement_id)
        if measurement is None or measurement.measurement_kind != "area" or len(measurement.polygon_px) < 3:
            return None
        for index, point in enumerate(measurement.polygon_px):
            if distance(point, image_point) <= self._selected_endpoint_tolerance():
                return measurement.id, "vertex", index
        center = measurement.polygon_center()
        if distance(center, image_point) <= max(5.0, 8.0 / max(self._zoom, 0.001)):
            return measurement.id, "center", None
        return None

    def _hit_test_area_measurement(self, image_point: Point) -> str | None:
        if self._document is None:
            return None
        tolerance = max(5.0, 10.0 / max(self._zoom, 0.001))
        for measurement in reversed(self._document.measurements):
            if measurement.measurement_kind != "area" or len(measurement.polygon_px) < 3:
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
            if self._point_to_segment_distance(image_point, measurement.effective_line()) <= tolerance:
                return measurement.id
        return None

    def _selected_endpoint_tolerance(self) -> float:
        return max(4.0, 9.0 / max(self._zoom, 0.001))

    def _endpoint_tolerance(self) -> float:
        return max(3.0, 6.0 / max(self._zoom, 0.001))

    def _hit_test_text_annotation(self, widget_point: QPointF) -> str | None:
        if self._document is None:
            return None
        for annotation in reversed(self._document.text_annotations):
            rect = annotation_rect(annotation, self._settings, self.image_to_widget)
            if rect.contains(widget_point):
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
            or self._dragging_text_id is not None
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
        if self._magic_segment.busy:
            prompt_text += " / 推理中..."
        rect = QRectF(14.0, 14.0, 240.0, 32.0)
        painter.fillRect(rect, QColor(16, 24, 32, 188))
        painter.setPen(QPen(QColor("#FFFFFF"), 1))
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, prompt_text)

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
        elif self._temporary_grab_active:
            self.setCursor(Qt.CursorShape.OpenHandCursor)
        else:
            self.unsetCursor()
