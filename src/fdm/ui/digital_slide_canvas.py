from __future__ import annotations

from threading import Thread
from time import perf_counter
from weakref import ref

from PySide6.QtCore import QPointF, QRect, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QImage, QKeyEvent, QWheelEvent
from shiboken6 import isValid as is_qobject_valid

from fdm.geometry import Point
from fdm.models import ImageDocument
from fdm.services.digital_slide_store import DigitalSlideManifest, DigitalSlideStore
from fdm.ui.canvas import DocumentCanvas


class DigitalSlideCanvas(DocumentCanvas):
    viewportChanged = Signal(int, int, int)
    navigationModeChanged = Signal(str)
    _bufferRendered = Signal(int, int, int, int, QImage)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._slide_store: DigitalSlideStore | None = None
        self._slide_manifest: DigitalSlideManifest | None = None
        self._viewport_origin = Point(0.0, 0.0)
        self._focus_index = 0
        self._initial_fit_pending = False
        self._initial_fit_done = False
        self._initial_fit_attempts = 0
        self._navigation_mode = "step"
        self._smooth_nav_keys: set[int] = set()
        self._smooth_nav_shift = False
        self._smooth_nav_last_at = 0.0
        self._smooth_nav_timer = QTimer(self)
        self._smooth_nav_timer.setInterval(16)
        self._smooth_nav_timer.timeout.connect(self._apply_smooth_navigation)
        self._viewport_buffer = QImage()
        self._viewport_buffer_origin = Point(0.0, 0.0)
        self._viewport_buffer_focus_index = -1
        self._viewport_buffer_request_id = 0
        self._viewport_buffer_thread: Thread | None = None
        self._viewport_buffer_pending = False
        self._viewport_last_publish_at = 0.0
        self._bufferRendered.connect(self._on_viewport_buffer_rendered)

    def set_slide_document(self, document: ImageDocument, store: DigitalSlideStore) -> None:
        self._slide_store = store
        self._slide_manifest = store.read_manifest()
        document.image_size = (self._slide_manifest.width, self._slide_manifest.height)
        slide_meta = dict(document.metadata.get("digital_slide", {})) if isinstance(document.metadata.get("digital_slide"), dict) else {}
        origin = slide_meta.get("viewport_origin")
        if isinstance(origin, (list, tuple)) and len(origin) >= 2:
            self._viewport_origin = Point(float(origin[0]), float(origin[1]))
        if "focus_index" in slide_meta:
            focus_index = int(slide_meta.get("focus_index", 0) or 0)
        else:
            focus_index = max(0, len(self._slide_manifest.focus_levels) // 2)
        self._focus_index = self._normalized_focus_index(focus_index)
        self._initial_fit_done = False
        self._invalidate_viewport_buffer()
        image = self._render_current_viewport()
        super().set_document(document, image)
        self._clamp_viewport()
        self._request_viewport_buffer()
        self.schedule_initial_fit()

    def set_image(self, image: QImage) -> None:
        self._image = image
        self.update()

    def focus_index(self) -> int:
        return self._focus_index

    def viewport_origin(self) -> Point:
        return Point(self._viewport_origin.x, self._viewport_origin.y)

    def navigation_mode(self) -> str:
        return self._navigation_mode

    def navigation_mode_label(self) -> str:
        return "平滑移动" if self._navigation_mode == "smooth" else "步进移动"

    def set_navigation_mode(self, mode: str) -> None:
        mode = "smooth" if mode == "smooth" else "step"
        if mode == self._navigation_mode:
            return
        self._navigation_mode = mode
        self._smooth_nav_keys.clear()
        self._smooth_nav_timer.stop()
        self._smooth_nav_last_at = 0.0
        self.navigationModeChanged.emit(mode)

    def toggle_navigation_mode(self) -> str:
        self.set_navigation_mode("smooth" if self._navigation_mode != "smooth" else "step")
        return self._navigation_mode

    def move_viewport_by(self, dx: float, dy: float, *, throttled: bool = False) -> None:
        self._viewport_origin = Point(self._viewport_origin.x + dx, self._viewport_origin.y + dy)
        self._clamp_viewport()
        if self._render_current_viewport_from_buffer():
            self._publish_viewport_state(throttled=throttled)
            self._request_viewport_buffer()
            return
        self._reload_viewport(throttled=throttled)

    def set_focus_index(self, focus_index: int) -> None:
        focus_index = self._normalized_focus_index(focus_index)
        if focus_index == self._focus_index:
            return
        self._focus_index = focus_index
        self._invalidate_viewport_buffer()
        self._reload_viewport()

    def widget_to_image(self, position: QPointF) -> Point:
        local = super().widget_to_image(position)
        return Point(local.x + self._viewport_origin.x, local.y + self._viewport_origin.y)

    def image_to_widget(self, point: Point) -> QPointF:
        return QPointF(
            self._pan.x + ((point.x - self._viewport_origin.x) * self._zoom),
            self._pan.y + ((point.y - self._viewport_origin.y) * self._zoom),
        )

    def fit_to_view(self) -> None:
        self._initial_fit_pending = False
        if self._image is None:
            return
        self._initial_fit_done = True
        super().fit_to_view()

    def actual_size(self) -> None:
        self._initial_fit_pending = False
        if self._image is None:
            return
        self._initial_fit_done = True
        super().actual_size()

    def schedule_initial_fit(self) -> None:
        if self._initial_fit_done or self._initial_fit_pending:
            return
        self._initial_fit_pending = True
        self._initial_fit_attempts = 0
        QTimer.singleShot(0, self._apply_initial_fit)

    def wheelEvent(self, event: QWheelEvent) -> None:
        if self._slide_manifest is None:
            return
        modifiers = getattr(event, "modifiers", lambda: Qt.KeyboardModifier.NoModifier)()
        if modifiers & Qt.KeyboardModifier.ControlModifier:
            self._zoom_current_viewport(event)
            return
        delta_y = event.angleDelta().y()
        delta_x = event.angleDelta().x()
        effective_delta = delta_y if delta_y != 0 else delta_x
        if effective_delta == 0:
            return
        wheel_step = max(1, int(getattr(self._settings, "digital_slide_focus_wheel_step", 1) or 1))
        self.set_focus_index(self._focus_index + (wheel_step if effective_delta > 0 else -wheel_step))
        event.accept()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.modifiers() == Qt.KeyboardModifier.NoModifier and event.key() == Qt.Key.Key_M:
            self.toggle_navigation_mode()
            event.accept()
            return
        if event.modifiers() == Qt.KeyboardModifier.NoModifier and event.key() in {
            Qt.Key.Key_Left,
            Qt.Key.Key_Right,
            Qt.Key.Key_Up,
            Qt.Key.Key_Down,
        }:
            if self._navigation_mode == "smooth":
                self._begin_smooth_navigation(event)
                event.accept()
                return
            step_x, step_y = self._navigation_step()
            if event.key() == Qt.Key.Key_Left:
                self.move_viewport_by(-step_x, 0)
            elif event.key() == Qt.Key.Key_Right:
                self.move_viewport_by(step_x, 0)
            elif event.key() == Qt.Key.Key_Up:
                self.move_viewport_by(0, -step_y)
            else:
                self.move_viewport_by(0, step_y)
            event.accept()
            return
        if event.modifiers() == Qt.KeyboardModifier.ShiftModifier and event.key() in {
            Qt.Key.Key_Left,
            Qt.Key.Key_Right,
            Qt.Key.Key_Up,
            Qt.Key.Key_Down,
        }:
            if self._navigation_mode == "smooth":
                self._begin_smooth_navigation(event)
                event.accept()
                return
            step_x = float(self._image.width() if self._image is not None else 0)
            step_y = float(self._image.height() if self._image is not None else 0)
            if event.key() == Qt.Key.Key_Left:
                self.move_viewport_by(-step_x, 0)
            elif event.key() == Qt.Key.Key_Right:
                self.move_viewport_by(step_x, 0)
            elif event.key() == Qt.Key.Key_Up:
                self.move_viewport_by(0, -step_y)
            else:
                self.move_viewport_by(0, step_y)
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent) -> None:
        if (
            self._navigation_mode == "smooth"
            and event.key() in {Qt.Key.Key_Left, Qt.Key.Key_Right, Qt.Key.Key_Up, Qt.Key.Key_Down}
        ):
            if not getattr(event, "isAutoRepeat", lambda: False)():
                self._smooth_nav_keys.discard(int(event.key()))
                if not self._smooth_nav_keys:
                    self._smooth_nav_timer.stop()
                    self._smooth_nav_last_at = 0.0
                    self._publish_viewport_state(throttled=False)
                    self._request_viewport_buffer()
            event.accept()
            return
        super().keyReleaseEvent(event)

    def _image_size(self) -> tuple[int, int] | None:
        if self._document is None:
            return None
        return self._document.image_size

    def _point_in_image(self, point: Point) -> bool:
        if self._document is None:
            return False
        width, height = self._document.image_size
        return 0 <= point.x < width and 0 <= point.y < height

    def _visible_image_rect(self) -> QRectF:
        if self._image is None:
            return QRectF()
        zoom = max(self._zoom, 0.001)
        left = (0.0 - self._pan.x) / zoom
        top = (0.0 - self._pan.y) / zoom
        right = (self.width() - self._pan.x) / zoom
        bottom = (self.height() - self._pan.y) / zoom
        padding = max(16.0, 28.0 / zoom)
        return QRectF(
            self._viewport_origin.x + max(0.0, min(left, right) - padding),
            self._viewport_origin.y + max(0.0, min(top, bottom) - padding),
            min(float(self._image.width()), abs(right - left) + (padding * 2.0)),
            min(float(self._image.height()), abs(bottom - top) + (padding * 2.0)),
        )

    def _persist_view_state(self) -> None:
        super()._persist_view_state()
        if self._document is None:
            return
        slide_meta = dict(self._document.metadata.get("digital_slide", {})) if isinstance(self._document.metadata.get("digital_slide"), dict) else {}
        slide_meta["viewport_origin"] = [int(round(self._viewport_origin.x)), int(round(self._viewport_origin.y))]
        slide_meta["focus_index"] = int(self._focus_index)
        self._document.metadata["digital_slide"] = slide_meta

    def _normalized_focus_index(self, focus_index: int) -> int:
        if self._slide_manifest is None:
            return 0
        return max(0, min(int(focus_index), max(0, len(self._slide_manifest.focus_levels) - 1)))

    def _navigation_step(self) -> tuple[float, float]:
        if self._image is None:
            return 0.0, 0.0
        return max(1.0, self._image.width() * 0.25), max(1.0, self._image.height() * 0.25)

    def _begin_smooth_navigation(self, event: QKeyEvent) -> None:
        if getattr(event, "isAutoRepeat", lambda: False)():
            return
        self._smooth_nav_keys.add(int(event.key()))
        self._smooth_nav_shift = event.modifiers() == Qt.KeyboardModifier.ShiftModifier
        self._smooth_nav_last_at = perf_counter()
        if not self._smooth_nav_timer.isActive():
            self._smooth_nav_timer.start()
        self._request_viewport_buffer()
        self._apply_smooth_navigation()

    def _apply_smooth_navigation(self) -> None:
        if self._image is None or not self._smooth_nav_keys:
            self._smooth_nav_timer.stop()
            self._smooth_nav_last_at = 0.0
            return
        now = perf_counter()
        previous = self._smooth_nav_last_at or now
        self._smooth_nav_last_at = now
        dt = max(0.001, min(0.12, now - previous))
        multiplier = 3.0 if self._smooth_nav_shift else 1.0
        speed = 0.75 * multiplier
        step_x = max(1.0, self._image.width() * speed * dt)
        step_y = max(1.0, self._image.height() * speed * dt)
        dx = 0.0
        dy = 0.0
        if int(Qt.Key.Key_Left) in self._smooth_nav_keys:
            dx -= step_x
        if int(Qt.Key.Key_Right) in self._smooth_nav_keys:
            dx += step_x
        if int(Qt.Key.Key_Up) in self._smooth_nav_keys:
            dy -= step_y
        if int(Qt.Key.Key_Down) in self._smooth_nav_keys:
            dy += step_y
        if dx or dy:
            self.move_viewport_by(dx, dy, throttled=True)

    def _apply_initial_fit(self) -> None:
        if not self._initial_fit_pending or self._image is None:
            return
        if (self.width() < 120 or self.height() < 120) and self._initial_fit_attempts < 6:
            self._initial_fit_attempts += 1
            QTimer.singleShot(50, self._apply_initial_fit)
            return
        self.fit_to_view()

    def _zoom_current_viewport(self, event: QWheelEvent) -> None:
        if self._image is None:
            return
        delta_y = event.angleDelta().y()
        delta_x = event.angleDelta().x()
        effective_delta = delta_y if delta_y != 0 else delta_x
        if effective_delta == 0:
            return
        self._initial_fit_pending = False
        self._initial_fit_done = True
        cursor_position = event.position()
        local_before = DocumentCanvas.widget_to_image(self, cursor_position)
        zoom_factor = 1.15 if effective_delta > 0 else 1 / 1.15
        self._zoom = max(0.05, min(40.0, self._zoom * zoom_factor))
        self._pan = Point(
            cursor_position.x() - (local_before.x * self._zoom),
            cursor_position.y() - (local_before.y * self._zoom),
        )
        self._persist_view_state()
        self.update()
        event.accept()

    def _clamp_viewport(self) -> None:
        if self._slide_manifest is None:
            return
        view_width = self._slide_manifest.viewport_width
        view_height = self._slide_manifest.viewport_height
        max_x = max(0, self._slide_manifest.width - view_width)
        max_y = max(0, self._slide_manifest.height - view_height)
        self._viewport_origin = Point(
            max(0.0, min(float(max_x), self._viewport_origin.x)),
            max(0.0, min(float(max_y), self._viewport_origin.y)),
        )

    def _render_current_viewport(self) -> QImage:
        if self._slide_store is None or self._slide_manifest is None:
            return QImage()
        self._clamp_viewport()
        metadata = self._slide_manifest.metadata if isinstance(self._slide_manifest.metadata, dict) else {}
        try:
            blend_width = int(metadata.get("blend_width", 0) or 0)
        except (TypeError, ValueError):
            blend_width = 0
        return self._slide_store.render_viewport(
            x=int(round(self._viewport_origin.x)),
            y=int(round(self._viewport_origin.y)),
            width=self._slide_manifest.viewport_width,
            height=self._slide_manifest.viewport_height,
            z_index=self._focus_index,
            blend_width=blend_width,
        )

    def _buffer_rect(self) -> QRect:
        if self._viewport_buffer.isNull():
            return QRect()
        return QRect(
            int(round(self._viewport_buffer_origin.x)),
            int(round(self._viewport_buffer_origin.y)),
            self._viewport_buffer.width(),
            self._viewport_buffer.height(),
        )

    def _current_viewport_rect(self) -> QRect:
        if self._slide_manifest is None:
            return QRect()
        return QRect(
            int(round(self._viewport_origin.x)),
            int(round(self._viewport_origin.y)),
            int(self._slide_manifest.viewport_width),
            int(self._slide_manifest.viewport_height),
        )

    def _invalidate_viewport_buffer(self) -> None:
        self._viewport_buffer = QImage()
        self._viewport_buffer_focus_index = -1
        self._viewport_buffer_request_id += 1
        self._viewport_buffer_pending = False

    def _render_current_viewport_from_buffer(self) -> bool:
        if self._slide_manifest is None or self._viewport_buffer.isNull():
            return False
        if self._viewport_buffer_focus_index != self._focus_index:
            return False
        viewport_rect = self._current_viewport_rect()
        buffer_rect = self._buffer_rect()
        if viewport_rect.isEmpty() or buffer_rect.isEmpty() or not buffer_rect.contains(viewport_rect):
            return False
        source_rect = QRect(
            viewport_rect.left() - buffer_rect.left(),
            viewport_rect.top() - buffer_rect.top(),
            viewport_rect.width(),
            viewport_rect.height(),
        )
        image = self._viewport_buffer.copy(source_rect)
        if image.isNull():
            return False
        self.set_image(image)
        return True

    def _buffer_margin(self) -> tuple[int, int]:
        if self._slide_manifest is None:
            return 0, 0
        return (
            max(1, int(round(self._slide_manifest.viewport_width * 0.5))),
            max(1, int(round(self._slide_manifest.viewport_height * 0.5))),
        )

    def _desired_buffer_rect(self) -> QRect:
        if self._slide_manifest is None:
            return QRect()
        viewport = self._current_viewport_rect()
        margin_x, margin_y = self._buffer_margin()
        width = min(int(self._slide_manifest.width), max(viewport.width(), viewport.width() + (margin_x * 2)))
        height = min(int(self._slide_manifest.height), max(viewport.height(), viewport.height() + (margin_y * 2)))
        left = int(round(viewport.center().x() - (width / 2)))
        top = int(round(viewport.center().y() - (height / 2)))
        left = max(0, min(left, max(0, int(self._slide_manifest.width) - width)))
        top = max(0, min(top, max(0, int(self._slide_manifest.height) - height)))
        return QRect(left, top, width, height)

    def _viewport_needs_buffer_refresh(self) -> bool:
        if self._slide_manifest is None:
            return False
        viewport = self._current_viewport_rect()
        buffer_rect = self._buffer_rect()
        if self._viewport_buffer.isNull() or self._viewport_buffer_focus_index != self._focus_index:
            return True
        if not buffer_rect.contains(viewport):
            return True
        margin_x, margin_y = self._buffer_margin()
        safe_rect = buffer_rect.adjusted(margin_x // 2, margin_y // 2, -(margin_x // 2), -(margin_y // 2))
        if safe_rect.isEmpty():
            return False
        return not safe_rect.contains(viewport)

    def _request_viewport_buffer(self) -> None:
        if self._slide_store is None or self._slide_manifest is None:
            return
        if not self.isVisible():
            return
        if self._viewport_buffer_thread is not None and self._viewport_buffer_thread.is_alive():
            self._viewport_buffer_pending = True
            return
        if not self._viewport_needs_buffer_refresh():
            return
        desired = self._desired_buffer_rect()
        if desired.isEmpty():
            return
        store_path = self._slide_store.path
        focus_index = int(self._focus_index)
        metadata = self._slide_manifest.metadata if isinstance(self._slide_manifest.metadata, dict) else {}
        try:
            blend_width = int(metadata.get("blend_width", 0) or 0)
        except (TypeError, ValueError):
            blend_width = 0
        self._viewport_buffer_request_id += 1
        request_id = self._viewport_buffer_request_id
        self._viewport_buffer_pending = False
        canvas_ref = ref(self)

        def render() -> None:
            store = DigitalSlideStore(store_path)
            try:
                image = store.render_viewport(
                    x=desired.left(),
                    y=desired.top(),
                    width=desired.width(),
                    height=desired.height(),
                    z_index=focus_index,
                    blend_width=blend_width,
                )
            finally:
                store.close()
            canvas = canvas_ref()
            if canvas is None or not is_qobject_valid(canvas):
                return
            canvas._bufferRendered.emit(request_id, desired.left(), desired.top(), focus_index, image)

        thread = Thread(target=render, name=f"fdm-slide-buffer-{store_path.name}", daemon=True)
        self._viewport_buffer_thread = thread
        thread.start()

    def _on_viewport_buffer_rendered(self, request_id: int, x: int, y: int, focus_index: int, image: QImage) -> None:
        self._viewport_buffer_thread = None
        if request_id != self._viewport_buffer_request_id or focus_index != self._focus_index or image.isNull():
            self._request_viewport_buffer()
            return
        self._viewport_buffer = image
        self._viewport_buffer_origin = Point(float(x), float(y))
        self._viewport_buffer_focus_index = int(focus_index)
        self._render_current_viewport_from_buffer()
        self.update()
        if self._viewport_buffer_pending or self._viewport_needs_buffer_refresh():
            self._request_viewport_buffer()

    def _publish_viewport_state(self, *, throttled: bool) -> None:
        now = perf_counter()
        if throttled and (now - self._viewport_last_publish_at) < 0.12:
            return
        self._viewport_last_publish_at = now
        self._persist_view_state()
        self.viewportChanged.emit(
            int(round(self._viewport_origin.x)),
            int(round(self._viewport_origin.y)),
            int(self._focus_index),
        )

    def _reload_viewport(self, *, throttled: bool = False) -> None:
        image = self._render_current_viewport()
        if not image.isNull():
            self.set_image(image)
        self._publish_viewport_state(throttled=throttled)
        self._request_viewport_buffer()
