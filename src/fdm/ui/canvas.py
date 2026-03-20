from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QMouseEvent, QPainter, QPen, QWheelEvent
from PySide6.QtWidgets import QWidget

from fdm.geometry import Line, Point, distance, line_length, nearest_endpoint
from fdm.models import ImageDocument, Measurement


class DocumentCanvas(QWidget):
    lineCommitted = Signal(str, str, object)
    measurementSelected = Signal(str, str)
    measurementEdited = Signal(str, str, object)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._document: ImageDocument | None = None
        self._image: QImage | None = None
        self._tool_mode = "select"
        self._zoom = 1.0
        self._pan = Point(20.0, 20.0)
        self._drawing_line: Line | None = None
        self._dragging_handle: tuple[str, str] | None = None
        self._drag_preview_line: Line | None = None
        self._panning = False
        self._last_mouse_pos = QPointF()

    @property
    def document_id(self) -> str | None:
        return self._document.id if self._document else None

    def set_document(self, document: ImageDocument, image: QImage) -> None:
        self._document = document
        self._image = image
        self._zoom = max(0.05, document.view_state.zoom or 1.0)
        self._pan = Point(document.view_state.pan.x, document.view_state.pan.y)
        if self._zoom == 1.0 and self._pan.x == 0.0 and self._pan.y == 0.0:
            self.fit_to_view()
        self.update()

    def set_tool_mode(self, mode: str) -> None:
        self._tool_mode = mode
        self.update()

    def set_selected_measurement(self, measurement_id: str | None) -> None:
        if self._document is None:
            return
        self._document.view_state.selected_measurement_id = measurement_id
        self.update()

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

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#101820"))
        if self._image is None or self._document is None:
            painter.setPen(QColor("#f2f2f2"))
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
        self._draw_measurements(painter)
        self._draw_preview(painter)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._image is not None and self._document is not None and self._document.view_state.zoom == 1.0:
            self.fit_to_view()

    def wheelEvent(self, event: QWheelEvent) -> None:
        if self._image is None:
            return
        cursor_position = event.position()
        image_before = self.widget_to_image(cursor_position)
        zoom_factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
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
        self._last_mouse_pos = event.position()
        if event.button() in (Qt.MouseButton.MiddleButton, Qt.MouseButton.RightButton):
            self._panning = True
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return

        image_point = self.widget_to_image(event.position())
        if self._tool_mode == "select":
            handle = self._hit_test_endpoint(image_point)
            if handle is not None:
                self._dragging_handle = handle
                self._drag_preview_line = self._measurement_line(handle[0])
                self._document.view_state.selected_measurement_id = handle[0]
                self.measurementSelected.emit(self._document.id, handle[0])
                self.update()
                return
            measurement_id = self._hit_test_measurement(image_point)
            if measurement_id is not None:
                self._document.view_state.selected_measurement_id = measurement_id
                self.measurementSelected.emit(self._document.id, measurement_id)
                self.update()
            return

        if self._point_in_image(image_point):
            self._drawing_line = Line(start=image_point, end=image_point)
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

        image_point = self.widget_to_image(event.position())
        if self._drawing_line is not None:
            self._drawing_line = Line(start=self._drawing_line.start, end=image_point)
            self.update()
            return

        if self._dragging_handle is not None:
            measurement_id, endpoint_name = self._dragging_handle
            measurement = self._document.get_measurement(measurement_id)
            if measurement is None:
                return
            base_line = measurement.effective_line()
            if endpoint_name == "start":
                self._drag_preview_line = Line(start=image_point, end=base_line.end)
            else:
                self._drag_preview_line = Line(start=base_line.start, end=image_point)
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._document is None:
            return
        if event.button() in (Qt.MouseButton.MiddleButton, Qt.MouseButton.RightButton):
            self._panning = False
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return
        if self._drawing_line is not None:
            line = self._drawing_line
            self._drawing_line = None
            if line_length(line) >= 3.0:
                self.lineCommitted.emit(self._document.id, self._tool_mode, line)
            self.update()
            return
        if self._dragging_handle is not None and self._drag_preview_line is not None:
            measurement_id, _ = self._dragging_handle
            preview = self._drag_preview_line
            self._dragging_handle = None
            self._drag_preview_line = None
            self.measurementEdited.emit(self._document.id, measurement_id, preview)
            self.update()
            return
        self._dragging_handle = None
        self._drag_preview_line = None

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
        return measurement.effective_line() if measurement else None

    def _draw_measurements(self, painter: QPainter) -> None:
        if self._document is None:
            return
        for measurement in self._document.measurements:
            line = measurement.effective_line()
            selected = measurement.id == self._document.view_state.selected_measurement_id
            color = QColor(self._measurement_color(measurement))
            if measurement.mode == "snap":
                color = QColor("#1DD3B0") if measurement.status == "snapped" else QColor("#FFB000")
            pen = QPen(color, 3 if selected else 2)
            painter.setPen(pen)
            painter.drawLine(self.image_to_widget(line.start), self.image_to_widget(line.end))
            self._draw_endpoint(painter, line.start, color, selected)
            self._draw_endpoint(painter, line.end, color, selected)

    def _draw_preview(self, painter: QPainter) -> None:
        preview = self._drag_preview_line or self._drawing_line
        if preview is None:
            return
        color = QColor("#FF7F50") if self._tool_mode == "calibration" else QColor("#F4D35E")
        painter.setPen(QPen(color, 2, Qt.PenStyle.DashLine))
        painter.drawLine(self.image_to_widget(preview.start), self.image_to_widget(preview.end))

    def _draw_endpoint(self, painter: QPainter, point: Point, color: QColor, selected: bool) -> None:
        widget_point = self.image_to_widget(point)
        radius = 5 if selected else 4
        painter.setBrush(color)
        painter.setPen(QPen(QColor("#0B0B0B"), 1))
        painter.drawEllipse(widget_point, radius, radius)

    def _measurement_color(self, measurement: Measurement) -> str:
        if self._document is None:
            return "#E0FBFC"
        group = self._document.get_group(measurement.fiber_group_id)
        return group.color if group else "#E0FBFC"

    def _hit_test_endpoint(self, image_point: Point) -> tuple[str, str] | None:
        if self._document is None:
            return None
        tolerance = max(4.0, 10.0 / max(self._zoom, 0.001))
        for measurement in reversed(self._document.measurements):
            line = measurement.effective_line()
            endpoint_name, endpoint_distance = nearest_endpoint(line, image_point)
            if endpoint_distance <= tolerance:
                return measurement.id, endpoint_name
        return None

    def _hit_test_measurement(self, image_point: Point) -> str | None:
        if self._document is None:
            return None
        tolerance = max(4.0, 10.0 / max(self._zoom, 0.001))
        for measurement in reversed(self._document.measurements):
            line = measurement.effective_line()
            if self._point_to_segment_distance(image_point, line) <= tolerance:
                return measurement.id
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

    def _persist_view_state(self) -> None:
        if self._document is None:
            return
        self._document.view_state.zoom = self._zoom
        self._document.view_state.pan = Point(self._pan.x, self._pan.y)
