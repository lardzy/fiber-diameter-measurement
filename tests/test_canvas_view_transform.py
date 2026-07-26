from __future__ import annotations

import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication

from fdm.geometry import Point
from fdm.models import ImageDocument
from fdm.services.digital_slide_store import DigitalSlideManifest
from fdm.ui.canvas import DocumentCanvas
from fdm.ui.digital_slide_canvas import DigitalSlideCanvas
from fdm.ui.view_transform import CanvasViewportSnapshot, CanvasZoomMode


class _WheelEvent:
    def __init__(
        self,
        delta: int,
        position: QPointF,
        *,
        modifiers: Qt.KeyboardModifier = Qt.KeyboardModifier.NoModifier,
    ) -> None:
        self._delta = delta
        self._position = position
        self._modifiers = modifiers
        self.accepted = False

    def angleDelta(self) -> QPoint:
        return QPoint(0, self._delta)

    def position(self) -> QPointF:
        return self._position

    def modifiers(self) -> Qt.KeyboardModifier:
        return self._modifiers

    def accept(self) -> None:
        self.accepted = True


class CanvasViewTransformTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def _canvas(
        self,
        *,
        canvas_size: tuple[int, int] = (400, 300),
        image_size: tuple[int, int] = (1200, 800),
    ) -> tuple[DocumentCanvas, ImageDocument]:
        canvas = DocumentCanvas()
        canvas.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
        canvas.resize(*canvas_size)
        document = ImageDocument(
            id="ordinary",
            path="/tmp/ordinary.png",
            image_size=image_size,
        )
        image = QImage(
            image_size[0],
            image_size[1],
            QImage.Format.Format_RGB32,
        )
        image.fill(0)
        canvas.set_document(document, image)
        return canvas, document

    @staticmethod
    def _close(canvas: DocumentCanvas) -> None:
        canvas.clear_document()
        canvas.close()
        canvas.deleteLater()

    def test_fit_supports_zoom_below_five_percent(self) -> None:
        canvas, _document = self._canvas(
            canvas_size=(240, 180),
            image_size=(20_000, 10_000),
        )
        try:
            canvas.fit_to_view()

            self.assertEqual(canvas.zoom_mode(), CanvasZoomMode.FIT)
            self.assertLess(canvas.view_zoom(), 0.05)
            self.assertAlmostEqual(canvas.view_zoom(), 0.01)
        finally:
            self._close(canvas)

    def test_custom_wheel_keeps_a_low_zoom_continuous(self) -> None:
        canvas, _document = self._canvas()
        try:
            canvas.set_view_zoom(0.001)
            before = canvas.view_zoom()
            event = _WheelEvent(-120, QPointF(130.0, 90.0))

            canvas.wheelEvent(event)

            self.assertTrue(event.accepted)
            self.assertEqual(canvas.zoom_mode(), CanvasZoomMode.CUSTOM)
            self.assertLess(canvas.view_zoom(), before)
            self.assertAlmostEqual(canvas.view_zoom(), before / 1.15)
        finally:
            self._close(canvas)

    def test_actual_size_preserves_current_view_center(self) -> None:
        canvas, _document = self._canvas()
        try:
            canvas.set_view_zoom(2.5)
            canvas.center_on_image_point(Point(720.0, 510.0))
            widget_center = QPointF(canvas.width() / 2.0, canvas.height() / 2.0)
            before = canvas.widget_to_image(widget_center)

            canvas.actual_size()

            after = canvas.widget_to_image(widget_center)
            self.assertEqual(canvas.zoom_mode(), CanvasZoomMode.ACTUAL)
            self.assertEqual(canvas.view_zoom(), 1.0)
            self.assertAlmostEqual(after.x, before.x)
            self.assertAlmostEqual(after.y, before.y)
        finally:
            self._close(canvas)

    def test_resize_recomputes_fit_but_preserves_custom_center(self) -> None:
        canvas, _document = self._canvas(canvas_size=(400, 300))
        canvas.show()
        self.app.processEvents()
        try:
            canvas.fit_to_view()
            fit_before = canvas.view_zoom()

            canvas.resize(700, 500)
            self.app.processEvents()

            self.assertEqual(canvas.zoom_mode(), CanvasZoomMode.FIT)
            self.assertGreater(canvas.view_zoom(), fit_before)

            canvas.set_view_zoom(1.7)
            canvas.center_on_image_point(Point(640.0, 420.0))
            center_before = canvas.widget_to_image(
                QPointF(canvas.width() / 2.0, canvas.height() / 2.0)
            )
            canvas.resize(850, 620)
            self.app.processEvents()
            center_after = canvas.widget_to_image(
                QPointF(canvas.width() / 2.0, canvas.height() / 2.0)
            )

            self.assertEqual(canvas.zoom_mode(), CanvasZoomMode.CUSTOM)
            self.assertAlmostEqual(canvas.view_zoom(), 1.7)
            self.assertAlmostEqual(center_after.x, center_before.x)
            self.assertAlmostEqual(center_after.y, center_before.y)
        finally:
            self._close(canvas)

    def test_snapshot_signal_covers_zoom_center_and_pan(self) -> None:
        canvas, _document = self._canvas()
        snapshots: list[CanvasViewportSnapshot] = []
        canvas.viewTransformChanged.connect(snapshots.append)
        try:
            canvas.fit_to_view()
            canvas.center_on_image_point(Point(500.0, 300.0))
            canvas._begin_canvas_pan(Qt.MouseButton.MiddleButton)  # noqa: SLF001
            canvas._last_mouse_pos = QPointF(100.0, 100.0)  # noqa: SLF001
            from PySide6.QtGui import QMouseEvent
            from PySide6.QtCore import QEvent

            event = QMouseEvent(
                QEvent.Type.MouseMove,
                QPointF(115.0, 108.0),
                QPointF(115.0, 108.0),
                Qt.MouseButton.NoButton,
                Qt.MouseButton.MiddleButton,
                Qt.KeyboardModifier.NoModifier,
            )
            canvas.mouseMoveEvent(event)

            self.assertGreaterEqual(len(snapshots), 3)
            self.assertTrue(
                all(isinstance(snapshot, CanvasViewportSnapshot) for snapshot in snapshots)
            )
            latest = snapshots[-1]
            self.assertEqual(latest.document_id, "ordinary")
            self.assertEqual(latest.full_image_rect, latest.mounted_image_rect)
            self.assertEqual(latest.mode, CanvasZoomMode.CUSTOM)
            self.assertGreater(latest.visible_image_rect.width(), 0.0)
            self.assertIsNone(latest.focus_index)
        finally:
            self._close(canvas)

    def test_digital_slide_snapshot_uses_global_coordinates(self) -> None:
        canvas = DigitalSlideCanvas()
        canvas.resize(320, 240)
        document = ImageDocument(
            id="slide",
            path="/tmp/slide.fdmslide",
            image_size=(4096, 3072),
            document_kind="digital_slide",
        )
        image = QImage(640, 480, QImage.Format.Format_RGB32)
        image.fill(0)
        canvas.set_document(document, image)
        canvas._slide_manifest = DigitalSlideManifest(  # noqa: SLF001
            version=1,
            width=4096,
            height=3072,
            viewport_width=640,
            viewport_height=480,
            focus_levels=[-1, 0, 1],
        )
        canvas._viewport_origin = Point(1200.0, 900.0)  # noqa: SLF001
        canvas._focus_index = 2  # noqa: SLF001
        snapshots: list[CanvasViewportSnapshot] = []
        canvas.viewTransformChanged.connect(snapshots.append)
        try:
            canvas.actual_size()
            canvas._publish_viewport_state(throttled=False)  # noqa: SLF001

            snapshot = canvas.viewport_snapshot()
            assert snapshot is not None
            self.assertEqual(snapshot.full_image_rect.width(), 4096.0)
            self.assertEqual(snapshot.full_image_rect.height(), 3072.0)
            self.assertEqual(snapshot.mounted_image_rect.x(), 1200.0)
            self.assertEqual(snapshot.mounted_image_rect.y(), 900.0)
            self.assertEqual(snapshot.mounted_image_rect.width(), 640.0)
            self.assertEqual(snapshot.mounted_image_rect.height(), 480.0)
            self.assertGreaterEqual(snapshot.visible_image_rect.left(), 1200.0)
            self.assertGreaterEqual(snapshot.visible_image_rect.top(), 900.0)
            self.assertEqual(snapshot.focus_index, 2)
            self.assertTrue(snapshots)
        finally:
            canvas.clear_document()
            canvas.close()
            canvas.deleteLater()

    def test_digital_slide_overview_center_preserves_local_zoom_mode(self) -> None:
        canvas = DigitalSlideCanvas()
        canvas.resize(320, 240)
        document = ImageDocument(
            id="slide-center",
            path="/tmp/slide-center.fdmslide",
            image_size=(4096, 3072),
            document_kind="digital_slide",
        )
        image = QImage(640, 480, QImage.Format.Format_RGB32)
        image.fill(0)
        canvas.set_document(document, image)
        canvas._slide_manifest = DigitalSlideManifest(  # noqa: SLF001
            version=1,
            width=4096,
            height=3072,
            viewport_width=640,
            viewport_height=480,
            focus_levels=[0],
        )
        try:
            canvas.fit_to_view()
            zoom_before = canvas.view_zoom()
            with patch.object(
                canvas,
                "_reload_viewport",
                side_effect=lambda: canvas._publish_viewport_state(  # noqa: SLF001
                    throttled=False
                ),
            ) as reload_viewport:
                canvas.center_on_image_point(Point(2000.0, 1500.0))

            self.assertEqual(canvas.zoom_mode(), CanvasZoomMode.FIT)
            self.assertAlmostEqual(canvas.view_zoom(), zoom_before)
            reload_viewport.assert_called_once_with()
            mapped = canvas.image_to_widget(Point(2000.0, 1500.0))
            self.assertAlmostEqual(mapped.x(), canvas.width() / 2.0)
            self.assertAlmostEqual(mapped.y(), canvas.height() / 2.0)
        finally:
            canvas.clear_document()
            canvas.close()
            canvas.deleteLater()


if __name__ == "__main__":
    unittest.main()
