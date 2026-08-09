from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
import sys
from time import monotonic, sleep
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication, QTabWidget, QWidget

from fdm.construction_geometry import ConstructionEntity, FreePointDefinition
from fdm.geometry import Line, Point
from fdm.models import ImageDocument, Measurement
from fdm.services.digital_slide_store import DigitalSlideManifest
from fdm.ui.canvas import DocumentCanvas
from fdm.ui.digital_slide_canvas import DigitalSlideCanvas


class _WheelEvent:
    def __init__(self, delta: int, position: QPointF) -> None:
        self._delta = delta
        self._position = position
        self.accepted = False

    def angleDelta(self) -> QPoint:
        return QPoint(0, self._delta)

    def position(self) -> QPointF:
        return self._position

    def accept(self) -> None:
        self.accepted = True


class _MouseEvent:
    def __init__(
        self,
        position: QPointF,
        *,
        button: Qt.MouseButton = Qt.MouseButton.LeftButton,
    ) -> None:
        self._position = QPointF(position)
        self._button = button
        self.accepted = False

    def position(self) -> QPointF:
        return QPointF(self._position)

    def button(self) -> Qt.MouseButton:
        return self._button

    @staticmethod
    def modifiers() -> Qt.KeyboardModifier:
        return Qt.KeyboardModifier.NoModifier

    def accept(self) -> None:
        self.accepted = True


class _KeyReleaseEvent:
    def __init__(self, key: Qt.Key) -> None:
        self._key = key
        self.accepted = False

    def key(self) -> Qt.Key:
        return self._key

    @staticmethod
    def isAutoRepeat() -> bool:
        return False

    def accept(self) -> None:
        self.accepted = True


class _AliveThread:
    @staticmethod
    def is_alive() -> bool:
        return True


class _FailingBufferStore:
    def __init__(self, _path: Path) -> None:
        pass

    @staticmethod
    def render_viewport(**_kwargs) -> QImage:
        raise OSError("permanent tile read failure")

    @staticmethod
    def close() -> None:
        return


class DigitalSlideOverlayLifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    @staticmethod
    def _canvas() -> DigitalSlideCanvas:
        document = ImageDocument(
            id="slide",
            path="/tmp/slide.fdmslide",
            image_size=(4096, 4096),
            document_kind="digital_slide",
        )
        canvas = DigitalSlideCanvas()
        canvas.resize(320, 240)
        image = QImage(320, 240, QImage.Format.Format_RGB32)
        image.fill(0)
        canvas.set_document(document, image)
        return canvas

    def test_digital_slide_zoom_cancels_previous_exact_generation(self) -> None:
        canvas = self._canvas()
        canvas._zoom = 1.0  # noqa: SLF001
        canvas._pan = Point(12.0, 18.0)  # noqa: SLF001
        event = _WheelEvent(120, QPointF(100.0, 80.0))
        try:
            with (
                patch.object(canvas, "_reset_proxy_warming") as reset_proxy,
                patch.object(canvas, "_cancel_overlay_requests") as cancel_tiles,
                patch.object(canvas, "update") as update,
            ):
                canvas._zoom_current_viewport(event)  # noqa: SLF001

            reset_proxy.assert_called_once_with()
            cancel_tiles.assert_called_once_with()
            update.assert_called_once_with()
            self.assertAlmostEqual(canvas._zoom, 1.15)  # noqa: SLF001
            self.assertTrue(event.accepted)
        finally:
            canvas.clear_document()
            canvas.close()

    def test_transient_navigation_does_not_enqueue_passive_tiles(self) -> None:
        canvas = self._canvas()
        keys = [object()]
        try:
            canvas._smooth_nav_keys.add(int(Qt.Key.Key_Right))  # noqa: SLF001
            with (
                patch.object(canvas, "_cancel_overlay_requests") as cancel,
                patch.object(
                    DocumentCanvas,
                    "_enqueue_overlay_tiles",
                ) as enqueue,
            ):
                canvas._enqueue_overlay_tiles(keys)  # noqa: SLF001
            cancel.assert_called_once_with()
            enqueue.assert_not_called()

            canvas._smooth_nav_keys.clear()  # noqa: SLF001
            canvas._viewport_buffer_thread = _AliveThread()  # type: ignore[assignment]  # noqa: SLF001
            with (
                patch.object(canvas, "_cancel_overlay_requests") as cancel,
                patch.object(
                    DocumentCanvas,
                    "_enqueue_overlay_tiles",
                ) as enqueue,
            ):
                canvas._enqueue_overlay_tiles(keys)  # noqa: SLF001
            cancel.assert_called_once_with()
            enqueue.assert_not_called()

            canvas._viewport_buffer_thread = None  # noqa: SLF001
            canvas._viewport_buffer_pending = False  # noqa: SLF001
            with (
                patch.object(canvas, "_cancel_overlay_requests") as cancel,
                patch.object(
                    DocumentCanvas,
                    "_enqueue_overlay_tiles",
                ) as enqueue,
            ):
                canvas._enqueue_overlay_tiles(keys)  # noqa: SLF001
            cancel.assert_not_called()
            enqueue.assert_called_once_with(keys)
        finally:
            canvas._viewport_buffer_thread = None  # noqa: SLF001
            canvas.clear_document()
            canvas.close()

    def test_tab_hide_cancels_navigation_and_buffer_generation_then_show_retries(
        self,
    ) -> None:
        canvas = self._canvas()
        tabs = QTabWidget()
        other = QWidget()
        try:
            tabs.addTab(canvas, "slide")
            tabs.addTab(other, "other")
            tabs.show()
            self.app.processEvents()
            canvas._smooth_nav_keys.add(int(Qt.Key.Key_Right))  # noqa: SLF001
            canvas._smooth_nav_timer.start()  # noqa: SLF001
            canvas._viewport_buffer_request_id = 7  # noqa: SLF001
            cancellation = canvas._viewport_buffer_cancel  # noqa: SLF001

            tabs.setCurrentIndex(1)
            self.app.processEvents()

            self.assertTrue(cancellation.is_set())
            self.assertEqual(canvas._viewport_buffer_request_id, 8)  # noqa: SLF001
            self.assertFalse(canvas._smooth_nav_keys)  # noqa: SLF001
            self.assertFalse(canvas._smooth_nav_timer.isActive())  # noqa: SLF001

            with patch.object(canvas, "_request_viewport_buffer") as request:
                tabs.setCurrentIndex(0)
                self.app.processEvents()
            request.assert_called_once_with()
        finally:
            tabs.close()
            canvas.clear_document()
            canvas.close()
            other.close()

    def test_smooth_navigation_release_requests_final_warmable_frame(self) -> None:
        canvas = self._canvas()
        canvas._navigation_mode = "smooth"  # noqa: SLF001
        canvas._smooth_nav_keys.add(int(Qt.Key.Key_Right))  # noqa: SLF001
        canvas._smooth_nav_timer.start()  # noqa: SLF001
        event = _KeyReleaseEvent(Qt.Key.Key_Right)
        try:
            with (
                patch.object(canvas, "_publish_viewport_state") as publish,
                patch.object(canvas, "_request_viewport_buffer") as request_buffer,
                patch.object(canvas, "update") as update,
            ):
                canvas.keyReleaseEvent(event)  # type: ignore[arg-type]

            self.assertFalse(canvas._smooth_nav_keys)  # noqa: SLF001
            self.assertFalse(canvas._smooth_nav_timer.isActive())  # noqa: SLF001
            publish.assert_called_once_with(throttled=False)
            request_buffer.assert_called_once_with()
            update.assert_called_once_with()
            self.assertTrue(event.accepted)
        finally:
            canvas._smooth_nav_timer.stop()  # noqa: SLF001
            canvas.clear_document()
            canvas.close()

    def test_viewport_move_cancels_stale_phase_and_keeps_global_mapping(self) -> None:
        canvas = self._canvas()
        canvas._slide_manifest = SimpleNamespace(  # type: ignore[assignment]  # noqa: SLF001
            width=4096,
            height=4096,
            viewport_width=320,
            viewport_height=240,
        )
        canvas._viewport_origin = Point(1000.0, 2000.0)  # noqa: SLF001
        canvas._zoom = 2.0  # noqa: SLF001
        canvas._pan = Point(20.0, 30.0)  # noqa: SLF001
        canvas._viewport_buffer_error_blocked = True  # noqa: SLF001
        measurement = Measurement(
            id="hovered-line",
            image_id="slide",
            fiber_group_id=None,
            mode="manual",
            line_px=Line(Point(1040.0, 2040.0), Point(1120.0, 2040.0)),
        )
        assert canvas._document is not None  # noqa: SLF001
        canvas._document.add_measurement(measurement)  # noqa: SLF001
        canvas.set_selected_measurement(measurement.id)
        canvas.set_tool_mode("select")
        canvas._set_hovered_line_endpoint(  # noqa: SLF001
            (measurement.id, "start")
        )
        self.assertEqual(
            canvas.cursor().shape(),
            Qt.CursorShape.SizeAllCursor,
        )
        try:
            with (
                patch.object(canvas, "_cancel_overlay_requests") as cancel,
                patch.object(
                    canvas,
                    "_render_current_viewport_from_buffer",
                    return_value=True,
                ),
                patch.object(canvas, "_publish_viewport_state"),
                patch.object(canvas, "_request_viewport_buffer"),
            ):
                canvas.move_viewport_by(15.0, -10.0)

            cancel.assert_called_once_with()
            self.assertFalse(canvas._viewport_buffer_error_blocked)  # noqa: SLF001
            self.assertEqual(canvas.viewport_origin(), Point(1015.0, 1990.0))
            self.assertIsNone(canvas._hovered_line_endpoint)  # noqa: SLF001
            self.assertEqual(
                canvas.cursor().shape(),
                Qt.CursorShape.ArrowCursor,
            )
            mapped_origin = canvas.image_to_widget(canvas.viewport_origin())
            self.assertAlmostEqual(mapped_origin.x(), 20.0)
            self.assertAlmostEqual(mapped_origin.y(), 30.0)
        finally:
            canvas.clear_document()
            canvas.close()

    def test_fit_padding_rejects_creation_and_drag_clamps_to_mounted_global_viewport(
        self,
    ) -> None:
        document = ImageDocument(
            id="slide-pointer-bounds",
            path="/tmp/slide-pointer-bounds.fdmslide",
            image_size=(4096, 4096),
            document_kind="digital_slide",
        )
        document.initialize_runtime_state()
        image = QImage(200, 100, QImage.Format.Format_RGB32)
        image.fill(0)
        canvas = DigitalSlideCanvas()
        canvas.resize(400, 400)
        canvas.set_document(document, image)
        canvas._viewport_origin = Point(1000.0, 2000.0)  # noqa: SLF001
        canvas.fit_to_view()
        created: list[ConstructionEntity] = []
        edited: list[ConstructionEntity] = []
        canvas.constructionCreateRequested.connect(
            lambda _document_id, entity: created.append(entity)
        )
        canvas.constructionEdited.connect(
            lambda _document_id, _entity_id, entity: edited.append(entity)
        )
        try:
            canvas.set_tool_mode("construction", construction_kind="point")
            padding_position = QPointF(200.0, 50.0)
            padding_global = canvas.widget_to_image(padding_position)
            self.assertLess(padding_global.y, canvas.viewport_origin().y)
            canvas.mousePressEvent(_MouseEvent(padding_position))
            canvas.mouseReleaseEvent(_MouseEvent(padding_position))
            self.assertEqual(created, [])

            target = Point(1050.0, 2050.0)
            target_position = canvas.image_to_widget(target)
            canvas.mousePressEvent(_MouseEvent(target_position))
            canvas.mouseReleaseEvent(_MouseEvent(target_position))
            self.assertEqual(len(created), 1)
            self.assertIsInstance(created[0].definition, FreePointDefinition)
            self.assertEqual(created[0].definition.point, target)

            entity = ConstructionEntity(
                id="drag-point",
                name="拖动点",
                definition=FreePointDefinition(target),
            )
            document.add_construction_entity(entity)
            canvas.set_tool_mode("select")
            canvas.set_selected_construction(entity.id)
            canvas.mousePressEvent(_MouseEvent(target_position))
            canvas.mouseMoveEvent(_MouseEvent(padding_position))
            canvas.mouseReleaseEvent(_MouseEvent(padding_position))

            self.assertEqual(len(edited), 1)
            self.assertIsInstance(edited[0].definition, FreePointDefinition)
            dragged = edited[0].definition.point
            self.assertGreaterEqual(dragged.x, 1000.0)
            self.assertLess(dragged.x, 1200.0)
            self.assertEqual(dragged.y, 2000.0)
        finally:
            canvas.clear_document()
            canvas.close()

    def test_permanent_buffer_error_is_latched_logged_and_not_retried(self) -> None:
        canvas = self._canvas()
        canvas._slide_store = SimpleNamespace(path=Path("/tmp/broken.fdmslide"))  # type: ignore[assignment]  # noqa: SLF001
        canvas._viewport_buffer_request_id = 7  # noqa: SLF001
        canvas._viewport_buffer_thread_request_id = 7  # noqa: SLF001
        canvas._viewport_buffer_pending = True  # noqa: SLF001
        failures: list[str] = []
        canvas.viewportBufferFailed.connect(failures.append)
        try:
            with (
                patch.object(canvas, "_request_viewport_buffer") as request,
                patch("fdm.ui.digital_slide_canvas.append_runtime_log") as log,
            ):
                canvas._on_viewport_buffer_rendered(  # noqa: SLF001
                    7,
                    100,
                    200,
                    0,
                    QImage(),
                    "error",
                    "OSError: permanent tile read failure",
                )

            request.assert_not_called()
            log.assert_called_once()
            self.assertEqual(failures, ["OSError: permanent tile read failure"])
            self.assertTrue(canvas._viewport_buffer_error_blocked)  # noqa: SLF001
            self.assertFalse(canvas._viewport_buffer_pending)  # noqa: SLF001
            self.assertIsNone(canvas._viewport_buffer_thread_request_id)  # noqa: SLF001

            with (
                patch.object(canvas, "isVisible", return_value=True),
                patch.object(canvas, "_viewport_needs_buffer_refresh") as needs_refresh,
            ):
                canvas._request_viewport_buffer()  # noqa: SLF001
            needs_refresh.assert_not_called()
        finally:
            canvas.clear_document()
            canvas.close()

    def test_cancelled_buffer_schedules_latest_request_without_error_latch(self) -> None:
        canvas = self._canvas()
        canvas._viewport_buffer_request_id = 4  # noqa: SLF001
        canvas._viewport_buffer_thread_request_id = 4  # noqa: SLF001
        try:
            with (
                patch.object(canvas, "_request_viewport_buffer") as request,
                patch("fdm.ui.digital_slide_canvas.append_runtime_log") as log,
            ):
                canvas._on_viewport_buffer_rendered(  # noqa: SLF001
                    4,
                    0,
                    0,
                    0,
                    QImage(),
                    "cancelled",
                    "",
                )

            request.assert_called_once_with()
            log.assert_not_called()
            self.assertFalse(canvas._viewport_buffer_error_blocked)  # noqa: SLF001
        finally:
            canvas.clear_document()
            canvas.close()

    def test_stale_error_does_not_cancel_or_replace_newer_worker(self) -> None:
        canvas = self._canvas()
        newer_thread = _AliveThread()
        canvas._viewport_buffer_request_id = 9  # noqa: SLF001
        canvas._viewport_buffer_thread_request_id = 9  # noqa: SLF001
        canvas._viewport_buffer_thread = newer_thread  # type: ignore[assignment]  # noqa: SLF001
        try:
            with (
                patch.object(canvas, "_request_viewport_buffer") as request,
                patch("fdm.ui.digital_slide_canvas.append_runtime_log") as log,
            ):
                canvas._on_viewport_buffer_rendered(  # noqa: SLF001
                    8,
                    0,
                    0,
                    0,
                    QImage(),
                    "error",
                    "old request failed",
                )

            request.assert_not_called()
            log.assert_not_called()
            self.assertIs(canvas._viewport_buffer_thread, newer_thread)  # noqa: SLF001
            self.assertFalse(canvas._viewport_buffer_error_blocked)  # noqa: SLF001
        finally:
            canvas._viewport_buffer_thread = None  # noqa: SLF001
            canvas._viewport_buffer_thread_request_id = None  # noqa: SLF001
            canvas.clear_document()
            canvas.close()

    def test_explicit_refresh_releases_permanent_error_latch(self) -> None:
        canvas = self._canvas()
        canvas._viewport_buffer_error_blocked = True  # noqa: SLF001
        try:
            with patch.object(canvas, "_request_viewport_buffer") as request:
                canvas.refresh_viewport_buffer()
            self.assertFalse(canvas._viewport_buffer_error_blocked)  # noqa: SLF001
            request.assert_called_once_with()
        finally:
            canvas.clear_document()
            canvas.close()

    def test_smooth_navigation_ticks_do_not_restart_after_permanent_error(self) -> None:
        canvas = self._canvas()
        canvas._slide_store = SimpleNamespace(path=Path("/tmp/broken.fdmslide"))  # type: ignore[assignment]  # noqa: SLF001
        canvas._slide_manifest = DigitalSlideManifest(  # noqa: SLF001
            version=1,
            width=4096,
            height=4096,
            viewport_width=320,
            viewport_height=240,
            focus_levels=[0],
        )
        canvas._smooth_nav_keys.add(int(Qt.Key.Key_Right))  # noqa: SLF001
        canvas._viewport_buffer_error_blocked = True  # noqa: SLF001
        try:
            with (
                patch.object(canvas, "_render_current_viewport_from_buffer", return_value=True),
                patch.object(canvas, "_publish_viewport_state"),
                patch.object(canvas, "isVisible", return_value=True),
            ):
                canvas.move_viewport_by(4.0, 0.0, throttled=True)

            self.assertTrue(canvas._viewport_buffer_error_blocked)  # noqa: SLF001
            self.assertIsNone(canvas._viewport_buffer_thread)  # noqa: SLF001
        finally:
            canvas._smooth_nav_keys.clear()  # noqa: SLF001
            canvas.clear_document()
            canvas.close()

    def test_worker_reports_real_exception_instead_of_cancelled_result(self) -> None:
        canvas = self._canvas()
        canvas._slide_store = SimpleNamespace(path=Path("/tmp/broken.fdmslide"))  # type: ignore[assignment]  # noqa: SLF001
        canvas._slide_manifest = DigitalSlideManifest(  # noqa: SLF001
            version=1,
            width=1024,
            height=768,
            viewport_width=320,
            viewport_height=240,
            focus_levels=[0],
        )
        results: list[tuple] = []
        canvas._bufferRendered.disconnect(canvas._on_viewport_buffer_rendered)  # noqa: SLF001
        canvas._bufferRendered.connect(lambda *args: results.append(args))  # noqa: SLF001
        try:
            with (
                patch.object(canvas, "isVisible", return_value=True),
                patch("fdm.ui.digital_slide_canvas.DigitalSlideStore", _FailingBufferStore),
            ):
                canvas._request_viewport_buffer()  # noqa: SLF001
                thread = canvas._viewport_buffer_thread  # noqa: SLF001
                self.assertIsNotNone(thread)
                thread.join(timeout=1.0)
                deadline = monotonic() + 1.0
                while not results and monotonic() < deadline:
                    self.app.processEvents()
                    sleep(0.005)

            self.assertEqual(len(results), 1)
            self.assertEqual(results[0][5], "error")
            self.assertIn("OSError: permanent tile read failure", results[0][6])
        finally:
            canvas._viewport_buffer_thread = None  # noqa: SLF001
            canvas._viewport_buffer_thread_request_id = None  # noqa: SLF001
            canvas.clear_document()
            canvas.close()


if __name__ == "__main__":
    unittest.main()
