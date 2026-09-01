from __future__ import annotations

import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPicture
from PySide6.QtWidgets import QApplication, QTabWidget

from fdm.area_display import area_derived_geometry_service
from fdm.geometry import Point
from fdm.models import ImageDocument, Measurement
import fdm.ui.canvas as canvas_module
from fdm.ui.area_handle_cache import area_handle_display_cache
from fdm.ui.canvas_overlay_cache import (
    CanvasOverlayRenderSnapshot,
    CanvasOverlayTileCache,
    CanvasOverlayTileKey,
)
from fdm.ui.digital_slide_canvas import DigitalSlideCanvas
from fdm.ui.main_window import MainWindow
from fdm.ui.canvas import DocumentCanvas


class _DeferredThreadPool:
    def __init__(self) -> None:
        self.runnables: list[object] = []

    def start(self, runnable) -> None:
        self.runnables.append(runnable)


def _picture() -> QPicture:
    picture = QPicture()
    painter = QPainter(picture)
    painter.fillRect(QRectF(0.0, 0.0, 24.0, 24.0), QColor("#2A9D8F"))
    painter.end()
    return picture


def _area_document(document_id: str) -> tuple[ImageDocument, Measurement, QImage]:
    document = ImageDocument(
        id=document_id,
        path=f"/tmp/{document_id}.png",
        image_size=(160, 120),
    )
    ring = [
        Point(10.0, 10.0),
        Point(140.0, 10.0),
        Point(140.0, 100.0),
        Point(10.0, 100.0),
    ]
    measurement = Measurement(
        id=f"{document_id}-area",
        image_id=document.id,
        fiber_group_id=None,
        mode="polygon_area",
        measurement_kind="area",
        polygon_px=list(ring),
        area_rings_px=[list(ring)],
    )
    document.add_measurement(measurement)
    document.mark_session_saved()
    image = QImage(160, 120, QImage.Format.Format_RGB32)
    image.fill(Qt.GlobalColor.white)
    return document, measurement, image


def _queue_pending_tile(
    cache: CanvasOverlayTileCache,
    document: ImageDocument,
) -> CanvasOverlayTileKey:
    key = CanvasOverlayTileKey(
        document_token=id(document),
        document_id=document.id,
        zoom=1.0,
        device_pixel_ratio=1.0,
        tile_x=0,
        tile_y=0,
        style_generation=0,
        tile_epoch=0,
        show_area_fill=True,
    )
    accepted = cache.request(
        CanvasOverlayRenderSnapshot(
            request_id=1,
            key=key,
            picture=_picture(),
        )
    )
    if not accepted:
        raise AssertionError("test tile request was not accepted")
    return key


class CanvasLifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_remove_document_cancels_pending_tile_and_releases_all_canvas_caches(
        self,
    ) -> None:
        pool = _DeferredThreadPool()
        cache = CanvasOverlayTileCache(thread_pool=pool)
        window = MainWindow()
        with patch.object(canvas_module, "canvas_overlay_tile_cache", cache):
            try:
                document, measurement, image = _area_document("close-one")
                window._mount_document(document, image, tooltip=document.path)
                canvas = window._canvases[document.id]
                area_derived_geometry_service.raw_geometry(measurement)
                area_handle_display_cache.coordinates(
                    measurement,
                    measurement.area_rings_px,
                    output_scale=1.0,
                    device_pixel_ratio=1.0,
                )
                identity = (id(measurement), measurement.id)
                self.assertIn(id(measurement), area_handle_display_cache._owner_keys)
                _queue_pending_tile(cache, document)
                self.assertEqual(cache.stats().pending, 1)
                self.assertEqual(len(pool.runnables), 1)

                window._remove_document(document.id)

                self.assertIsNone(canvas.document_id)
                self.assertEqual(cache.stats().pending, 0)
                self.assertNotIn(id(measurement), area_handle_display_cache._owner_keys)
                for derived_cache in (
                    area_derived_geometry_service._bounds,
                    area_derived_geometry_service._moments,
                    area_derived_geometry_service._hole_areas,
                    area_derived_geometry_service._raw_paths,
                    area_derived_geometry_service._proxies,
                    area_derived_geometry_service._hit_indexes,
                ):
                    self.assertTrue(
                        all(key[:2] != identity for key in derived_cache)
                    )

                # The worker may finish after the tab has been removed.  Its
                # completion must be rejected instead of repopulating a tile
                # or touching the detached canvas.
                pool.runnables[0].run()
                cache._drain_completions()
                self.assertEqual(cache.stats().entries, 0)
                self.assertGreaterEqual(cache.stats().dropped, 1)
            finally:
                window._reset_workspace()
                window.close()
                cache.clear()

    def test_reset_workspace_cancels_pending_tiles_before_canvas_deletion(
        self,
    ) -> None:
        pool = _DeferredThreadPool()
        cache = CanvasOverlayTileCache(thread_pool=pool)
        window = MainWindow()
        with patch.object(canvas_module, "canvas_overlay_tile_cache", cache):
            try:
                first, _first_measurement, first_image = _area_document("reset-first")
                second, _second_measurement, second_image = _area_document("reset-second")
                window._mount_document(first, first_image, tooltip=first.path)
                window._mount_document(second, second_image, tooltip=second.path)
                canvases = tuple(window._canvases.values())
                _queue_pending_tile(cache, first)
                _queue_pending_tile(cache, second)
                self.assertEqual(cache.stats().pending, 2)

                # One worker has already produced an image, but its UI-thread
                # completion has not been drained yet.  Reset must invalidate
                # both this late result and the worker that has not started.
                pool.runnables[0].run()
                window._reset_workspace()

                self.assertEqual(window._canvases, {})
                self.assertEqual(cache.stats().pending, 0)
                self.assertTrue(all(canvas.document_id is None for canvas in canvases))
                for runnable in pool.runnables[1:]:
                    runnable.run()
                cache._drain_completions()
                self.assertEqual(cache.stats().entries, 0)
                self.assertGreaterEqual(cache.stats().dropped, 2)
            finally:
                window._reset_workspace()
                window.close()
                cache.clear()

    def test_switching_tabs_cancels_hidden_canvas_tile_queue_and_late_result(
        self,
    ) -> None:
        pool = _DeferredThreadPool()
        cache = CanvasOverlayTileCache(thread_pool=pool)
        tabs = QTabWidget()
        with patch.object(canvas_module, "canvas_overlay_tile_cache", cache):
            first = DocumentCanvas()
            second = DocumentCanvas()
            try:
                first_document, _measurement, _image = _area_document(
                    "hidden-first"
                )
                first_document.image_size = (2048, 1024)
                first_image = QImage(
                    2048,
                    1024,
                    QImage.Format.Format_RGB32,
                )
                first_image.fill(Qt.GlobalColor.white)
                second_document, _measurement, second_image = _area_document(
                    "visible-second"
                )
                first.set_document(first_document, first_image)
                second.set_document(second_document, second_image)
                tabs.addTab(first, "first")
                tabs.addTab(second, "second")
                tabs.resize(1200, 800)
                tabs.show()
                self.app.processEvents()
                first._zoom = 1.0  # noqa: SLF001
                first._pan = Point(0.0, 0.0)  # noqa: SLF001
                keys = first._visible_overlay_tile_keys(  # noqa: SLF001
                    first._paint_context()  # noqa: SLF001
                )
                self.assertGreater(len(keys), 1)

                first._enqueue_overlay_tiles(keys)  # noqa: SLF001
                self.app.processEvents()
                self.assertIsNotNone(first._overlay_tile_active)  # noqa: SLF001
                self.assertTrue(first._overlay_tile_queue)  # noqa: SLF001
                self.assertEqual(cache.stats().pending, 1)
                self.assertEqual(len(pool.runnables), 1)

                tabs.setCurrentIndex(1)
                self.app.processEvents()

                self.assertIsNone(first._overlay_tile_active)  # noqa: SLF001
                self.assertFalse(first._overlay_tile_queue)  # noqa: SLF001
                self.assertFalse(first._overlay_tile_queued)  # noqa: SLF001
                self.assertEqual(cache.stats().pending, 0)

                pool.runnables[0].run()
                cache._drain_completions()  # noqa: SLF001
                self.app.processEvents()
                self.assertEqual(len(pool.runnables), 1)
                self.assertEqual(cache.stats().entries, 0)
                self.assertGreaterEqual(cache.stats().dropped, 1)
            finally:
                first.clear_document()
                second.clear_document()
                tabs.close()
                first.close()
                second.close()
                cache.clear()

    def test_digital_slide_shutdown_closes_long_lived_renderer(
        self,
    ) -> None:
        canvas = DigitalSlideCanvas()

        class RecordingRenderer:
            def __init__(self) -> None:
                self.closed = False

            def close(self) -> None:
                self.closed = True

        renderer = RecordingRenderer()
        canvas._renderer = renderer  # type: ignore[assignment]  # noqa: SLF001

        canvas.shutdown()

        self.assertTrue(renderer.closed)
        self.assertIsNone(canvas._renderer)  # noqa: SLF001
        canvas.close()


if __name__ == "__main__":
    unittest.main()
