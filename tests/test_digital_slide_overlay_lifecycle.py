from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
import sys
from tempfile import TemporaryDirectory
from threading import Event
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
from fdm.services.digital_slide_renderer import (
    DigitalSlideRenderFailure,
    DigitalSlideRenderFrame,
    DigitalSlideRenderRequest,
    DigitalSlideRenderer,
)
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


class _RecordingRenderer:
    def __init__(self) -> None:
        self.requests: list[DigitalSlideRenderRequest] = []
        self.closed = False

    def submit(self, request: DigitalSlideRenderRequest) -> None:
        self.requests.append(request)

    def close(self, *, timeout: float = 2.0) -> None:
        del timeout
        self.closed = True


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
        canvas._slide_manifest = DigitalSlideManifest(  # noqa: SLF001
            version=1,
            width=4096,
            height=4096,
            viewport_width=320,
            viewport_height=240,
            focus_levels=[0],
        )
        canvas._browse_center = Point(160.0, 120.0)  # noqa: SLF001
        canvas._update_native_viewport_origin()  # noqa: SLF001
        canvas._native_frame_key = canvas._native_request_key()  # noqa: SLF001
        canvas._sync_pan_from_browse_center()  # noqa: SLF001
        canvas._update_pixel_work_state()  # noqa: SLF001
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
            with (
                patch.object(
                    canvas,
                    "renderer_stats",
                    return_value=SimpleNamespace(pending_requests=1),
                ),
                patch.object(canvas, "_cancel_overlay_requests") as cancel,
                patch.object(
                    DocumentCanvas,
                    "_enqueue_overlay_tiles",
                ) as enqueue,
            ):
                canvas._enqueue_overlay_tiles(keys)  # noqa: SLF001
            cancel.assert_called_once_with()
            enqueue.assert_not_called()

            with (
                patch.object(canvas, "renderer_stats", return_value=None),
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

            tabs.setCurrentIndex(1)
            self.app.processEvents()

            self.assertFalse(canvas._smooth_nav_keys)  # noqa: SLF001
            self.assertFalse(canvas._smooth_nav_timer.isActive())  # noqa: SLF001

            with (
                patch.object(canvas, "_request_display_frame") as display_request,
                patch.object(canvas, "_request_native_frame") as native_request,
            ):
                tabs.setCurrentIndex(0)
                self.app.processEvents()
            display_request.assert_called_once_with()
            native_request.assert_called_once_with()
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
        canvas._browse_center = Point(1160.0, 2120.0)  # noqa: SLF001
        canvas._zoom = 2.0  # noqa: SLF001
        canvas._sync_pan_from_browse_center()  # noqa: SLF001
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
                patch.object(canvas, "_publish_viewport_state"),
                patch.object(canvas, "_request_display_frame"),
                patch.object(canvas, "_request_native_frame"),
            ):
                canvas.move_viewport_by(15.0, -10.0)

            cancel.assert_called_once_with()
            self.assertTrue(canvas._viewport_buffer_error_blocked)  # noqa: SLF001
            self.assertEqual(canvas.viewport_origin(), Point(1015.0, 1990.0))
            self.assertIsNone(canvas._hovered_line_endpoint)  # noqa: SLF001
            self.assertEqual(
                canvas.cursor().shape(),
                Qt.CursorShape.ArrowCursor,
            )
            mapped_center = canvas.image_to_widget(canvas.browse_view().center_px)
            self.assertAlmostEqual(mapped_center.x(), canvas._content_rect().center().x())  # noqa: SLF001
            self.assertAlmostEqual(mapped_center.y(), canvas._content_rect().center().y())  # noqa: SLF001
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
        canvas._slide_manifest = DigitalSlideManifest(  # noqa: SLF001
            version=1,
            width=4096,
            height=4096,
            viewport_width=200,
            viewport_height=100,
            focus_levels=[0],
        )
        canvas._viewport_origin = Point(1000.0, 2000.0)  # noqa: SLF001
        canvas._browse_center = Point(1100.0, 2050.0)  # noqa: SLF001
        canvas.fit_native_viewport()
        canvas._native_frame_key = canvas._native_request_key()  # noqa: SLF001
        canvas._update_pixel_work_state()  # noqa: SLF001
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

    def test_current_render_error_is_latched_logged_and_published(self) -> None:
        canvas = self._canvas()
        canvas._slide_store = SimpleNamespace(path=Path("/tmp/broken.fdmslide"))  # type: ignore[assignment]  # noqa: SLF001
        canvas._latest_native_request_id = 7  # noqa: SLF001
        failures: list[str] = []
        canvas.viewportBufferFailed.connect(failures.append)
        try:
            with patch("fdm.ui.digital_slide_canvas.append_runtime_log") as log:
                canvas._on_render_frame_failed(  # noqa: SLF001
                    DigitalSlideRenderFailure(
                        request_id=7,
                        purpose="native",
                        focus_index=0,
                        message="OSError: permanent tile read failure",
                    )
                )

            log.assert_called_once()
            self.assertEqual(failures, ["OSError: permanent tile read failure"])
            self.assertTrue(canvas._viewport_buffer_error_blocked)  # noqa: SLF001
            self.assertIn("读取失败", canvas.pixel_work_unavailable_reason())
        finally:
            canvas.clear_document()
            canvas.close()

    def test_stale_render_error_does_not_replace_current_generation(self) -> None:
        canvas = self._canvas()
        canvas._latest_display_request_id = 5  # noqa: SLF001
        failures: list[str] = []
        canvas.viewportBufferFailed.connect(failures.append)
        try:
            with patch("fdm.ui.digital_slide_canvas.append_runtime_log") as log:
                canvas._on_render_frame_failed(  # noqa: SLF001
                    DigitalSlideRenderFailure(
                        request_id=4,
                        purpose="display",
                        focus_index=0,
                        message="old request failed",
                    )
                )

            log.assert_not_called()
            self.assertEqual(failures, [])
            self.assertFalse(canvas._viewport_buffer_error_blocked)  # noqa: SLF001
        finally:
            canvas.clear_document()
            canvas.close()

    def test_stale_focus_error_is_ignored(self) -> None:
        canvas = self._canvas()
        canvas._latest_native_request_id = 9  # noqa: SLF001
        try:
            with patch("fdm.ui.digital_slide_canvas.append_runtime_log") as log:
                canvas._on_render_frame_failed(  # noqa: SLF001
                    DigitalSlideRenderFailure(
                        request_id=9,
                        purpose="native",
                        focus_index=1,
                        message="old focus failed",
                    )
                )

            log.assert_not_called()
            self.assertFalse(canvas._viewport_buffer_error_blocked)  # noqa: SLF001
        finally:
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

    def test_navigation_at_boundary_does_not_enqueue_or_repaint(self) -> None:
        canvas = self._canvas()
        canvas._browse_center = Point(160.0, 120.0)  # noqa: SLF001
        canvas._zoom = canvas._native_field_fit_zoom()  # noqa: SLF001
        try:
            with (
                patch.object(canvas, "_request_display_frame") as display_request,
                patch.object(canvas, "_request_native_frame") as native_request,
                patch.object(canvas, "_publish_viewport_state") as publish,
                patch.object(canvas, "update") as update,
            ):
                canvas.move_viewport_by(-40.0, -30.0, throttled=True)

            display_request.assert_not_called()
            native_request.assert_not_called()
            publish.assert_not_called()
            update.assert_not_called()
        finally:
            canvas.clear_document()
            canvas.close()

    def test_worker_reports_real_exception_instead_of_cancelled_result(self) -> None:
        manifest = DigitalSlideManifest(
            version=1,
            width=1024,
            height=768,
            viewport_width=320,
            viewport_height=240,
            focus_levels=[0],
        )
        failures: list[DigitalSlideRenderFailure] = []
        completed = Event()
        with TemporaryDirectory() as temporary_directory:
            source = Path(temporary_directory) / "missing.fdmslide"
            renderer = DigitalSlideRenderer(
                source,
                manifest,
                cache_root=Path(temporary_directory) / "cache",
                disk_cache_bytes=0,
                result_callback=lambda _frame: completed.set(),
                failure_callback=lambda failure: (failures.append(failure), completed.set()),
            )
            try:
                renderer.submit(
                    DigitalSlideRenderRequest(
                        request_id=1,
                        purpose="display",
                        source_rect=(0.0, 0.0, 320.0, 240.0),
                        output_size_px=(320, 240),
                        focus_index=0,
                        device_pixel_ratio=1.0,
                    )
                )
                self.assertTrue(completed.wait(1.0))
                completed.clear()
                renderer.submit(
                    DigitalSlideRenderRequest(
                        request_id=2,
                        purpose="native",
                        source_rect=(0.0, 0.0, 320.0, 240.0),
                        output_size_px=(320, 240),
                        focus_index=0,
                        device_pixel_ratio=1.0,
                        force_lod=0,
                    )
                )
                self.assertTrue(completed.wait(1.0))
                self.assertTrue(renderer.is_alive())
            finally:
                renderer.close()

        self.assertEqual(len(failures), 2)
        self.assertEqual(failures[0].request_id, 1)
        self.assertEqual(failures[1].request_id, 2)
        self.assertIn("filenotfounderror", failures[0].message.lower())

    def test_static_overview_uses_center_focus_without_changing_display_focus(self) -> None:
        canvas = self._canvas()
        canvas._slide_store = SimpleNamespace(path=Path("/tmp/focus.fdmslide"))  # type: ignore[assignment]  # noqa: SLF001
        canvas._slide_manifest = DigitalSlideManifest(  # noqa: SLF001
            version=1,
            width=1024,
            height=768,
            viewport_width=320,
            viewport_height=240,
            focus_levels=[-20, -10, 0, 10, 20],
        )
        canvas._dynamic_focus_overview_enabled = False  # noqa: SLF001
        canvas._focus_index = 4  # noqa: SLF001
        renderer = _RecordingRenderer()
        canvas._renderer = renderer  # type: ignore[assignment]  # noqa: SLF001
        canvas._overview_enabled = True  # noqa: SLF001

        try:
            with patch.object(canvas, "isVisible", return_value=True):
                canvas._request_viewport_buffer()  # noqa: SLF001
                canvas.request_overview()

            self.assertEqual(canvas._overview_target_focus_index(), 2)  # noqa: SLF001
            self.assertEqual(
                [(request.purpose, request.focus_index) for request in renderer.requests],
                [("display", 4), ("native", 4), ("overview", 2)],
            )
        finally:
            canvas._renderer = None  # noqa: SLF001
            canvas.clear_document()
            canvas.close()

    def test_stale_overview_never_replaces_current_focus_thumbnail(self) -> None:
        canvas = self._canvas()
        canvas._focus_index = 1  # noqa: SLF001
        canvas._latest_overview_request_id = 2  # noqa: SLF001
        emitted: list[QImage] = []
        canvas.overviewImageChanged.connect(emitted.append)
        stale = QImage(32, 16, QImage.Format.Format_RGB32)
        stale.fill(0xFFCC0000)
        current = QImage(32, 16, QImage.Format.Format_RGB32)
        current.fill(0xFF00CC00)
        try:
            canvas._on_render_frame_ready(  # noqa: SLF001
                DigitalSlideRenderFrame(
                    request_id=1,
                    purpose="overview",
                    source_rect=(0.0, 0.0, 4096.0, 4096.0),
                    output_size_px=(32, 16),
                    focus_index=0,
                    device_pixel_ratio=1.0,
                    lod=3,
                    image=stale,
                    elapsed_ms=1.0,
                    decoded_tiles=1,
                    cache_hits=0,
                )
            )
            self.assertTrue(canvas.overview_image().isNull())
            self.assertEqual(emitted, [])

            canvas._on_render_frame_ready(  # noqa: SLF001
                DigitalSlideRenderFrame(
                    request_id=2,
                    purpose="overview",
                    source_rect=(0.0, 0.0, 4096.0, 4096.0),
                    output_size_px=(32, 16),
                    focus_index=1,
                    device_pixel_ratio=1.0,
                    lod=3,
                    image=current,
                    elapsed_ms=1.0,
                    decoded_tiles=1,
                    cache_hits=0,
                )
            )
            self.assertFalse(canvas.overview_image().isNull())
            self.assertEqual(len(emitted), 1)
            self.assertGreater(
                canvas.overview_image().pixelColor(5, 5).green(),
                150,
            )

            canvas._focus_index = 2  # noqa: SLF001
            canvas._latest_overview_request_id = 3  # noqa: SLF001
            with patch("fdm.ui.digital_slide_canvas.append_runtime_log"):
                canvas._on_render_frame_failed(  # noqa: SLF001
                    DigitalSlideRenderFailure(
                        request_id=3,
                        purpose="overview",
                        focus_index=2,
                        message="no overview tiles",
                    )
                )
            self.assertTrue(canvas.overview_image().isNull())
            self.assertEqual(len(emitted), 2)
            self.assertTrue(emitted[-1].isNull())
        finally:
            canvas.clear_document()
            canvas.close()

    def test_disabled_overview_does_not_start_background_reader(self) -> None:
        canvas = self._canvas()
        canvas._slide_store = SimpleNamespace(  # type: ignore[assignment]  # noqa: SLF001
            path=Path("/tmp/not-read.fdmslide")
        )
        canvas._slide_manifest = DigitalSlideManifest(  # noqa: SLF001
            version=1,
            width=4096,
            height=4096,
            viewport_width=320,
            viewport_height=240,
            focus_levels=[0],
        )
        try:
            with (
                patch.object(canvas, "isVisible", return_value=True),
                patch.object(canvas, "_start_renderer") as start_renderer,
            ):
                canvas.request_overview()
            start_renderer.assert_not_called()
            self.assertIsNone(canvas._renderer)  # noqa: SLF001
        finally:
            canvas.clear_document()
            canvas.close()


if __name__ == "__main__":
    unittest.main()
