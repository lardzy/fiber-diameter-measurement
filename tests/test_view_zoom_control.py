from __future__ import annotations

import os
from pathlib import Path
import sys
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PySide6.QtCore import QPoint, QPointF, QRectF, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtWidgets import QApplication

from fdm.ui.view_transform import CanvasViewportSnapshot, CanvasZoomMode
from fdm.ui.view_zoom_control import ViewZoomStatusButton, _format_percentage


def _snapshot(zoom: float, mode: CanvasZoomMode) -> CanvasViewportSnapshot:
    return CanvasViewportSnapshot(
        document_id="doc",
        full_image_rect=QRectF(0, 0, 1000, 500),
        mounted_image_rect=QRectF(0, 0, 1000, 500),
        visible_image_rect=QRectF(100, 50, 500, 250),
        zoom=zoom,
        mode=mode,
        device_pixel_ratio=1.5,
    )


class ViewZoomStatusButtonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_formats_fit_actual_custom_and_tiny_zoom(self) -> None:
        button = ViewZoomStatusButton()
        button.set_viewport_snapshot(_snapshot(0.237, CanvasZoomMode.FIT))
        self.assertEqual(button.text(), "适合窗口 · 23.7%")
        button.set_viewport_snapshot(_snapshot(1.0, CanvasZoomMode.ACTUAL))
        self.assertEqual(button.text(), "原始像素 · 100%")
        button.set_viewport_snapshot(_snapshot(2.4, CanvasZoomMode.CUSTOM))
        self.assertEqual(button.text(), "视图缩放 · 240%")
        self.assertEqual(_format_percentage(0.0001), "0.01%")

    def test_digital_slide_uses_viewport_wording(self) -> None:
        button = ViewZoomStatusButton()
        button.set_viewport_snapshot(
            _snapshot(1.0, CanvasZoomMode.ACTUAL),
            digital_slide=True,
        )
        self.assertEqual(button.text(), "视场原始像素 · 100%")

    def test_wheel_is_consumed_without_emitting_zoom(self) -> None:
        button = ViewZoomStatusButton()
        emitted: list[float] = []
        button.zoomRequested.connect(emitted.append)
        event = QWheelEvent(
            QPointF(4, 4),
            QPointF(4, 4),
            QPoint(),
            QPoint(0, 120),
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
            Qt.ScrollPhase.ScrollUpdate,
            False,
        )
        button.wheelEvent(event)
        self.assertTrue(event.isAccepted())
        self.assertEqual(emitted, [])


if __name__ == "__main__":
    unittest.main()
