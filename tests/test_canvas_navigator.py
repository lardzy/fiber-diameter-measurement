from __future__ import annotations

import os
from pathlib import Path
import sys
import unittest


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PySide6.QtCore import QPoint, QPointF, QRectF, Qt
from PySide6.QtGui import QImage, QMouseEvent, QPalette, QWheelEvent
from PySide6.QtWidgets import QApplication, QWidget

from fdm.geometry import Point
from fdm.ui.canvas_navigator import CanvasNavigatorWidget, _navigator_colors
from fdm.ui.theme import build_dark_palette, build_light_palette
from fdm.ui.view_transform import CanvasViewportSnapshot, CanvasZoomMode


def _snapshot(
    *,
    full: QRectF = QRectF(100.0, 200.0, 1000.0, 500.0),
    mounted: QRectF = QRectF(100.0, 200.0, 1000.0, 500.0),
    visible: QRectF = QRectF(350.0, 300.0, 500.0, 250.0),
) -> CanvasViewportSnapshot:
    return CanvasViewportSnapshot(
        document_id="doc_1",
        full_image_rect=full,
        mounted_image_rect=mounted,
        visible_image_rect=visible,
        zoom=2.0,
        mode=CanvasZoomMode.CUSTOM,
        device_pixel_ratio=1.0,
    )


class CanvasNavigatorWidgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.parent = QWidget()
        self.parent.resize(800, 600)
        self.widget = CanvasNavigatorWidget(self.parent)
        self.parent.show()
        self.app.processEvents()

    def tearDown(self) -> None:
        self.parent.close()

    def test_maps_image_coordinates_through_non_zero_global_origin(self) -> None:
        self.widget.set_viewport_snapshot(_snapshot())
        content = self.widget.content_rect()

        self.assertEqual(
            self.widget.map_image_point_to_widget(Point(100.0, 200.0)),
            content.topLeft(),
        )
        self.assertEqual(
            self.widget.map_image_point_to_widget(Point(1100.0, 700.0)),
            content.bottomRight(),
        )
        center = self.widget.map_widget_point_to_image(content.center())
        self.assertIsNotNone(center)
        assert center is not None
        self.assertAlmostEqual(center.x, 600.0)
        self.assertAlmostEqual(center.y, 450.0)

    def test_visibility_tracks_whether_whole_image_is_visible(self) -> None:
        self.widget.set_viewport_snapshot(
            _snapshot(visible=QRectF(100.0, 200.0, 1000.0, 500.0))
        )
        self.assertTrue(self.widget.isHidden())

        self.widget.set_viewport_snapshot(_snapshot())
        self.assertFalse(self.widget.isHidden())
        self.assertEqual(self.widget.pos().x(), 800 - 176 - 12)

        self.parent.resize(1000, 600)
        self.app.processEvents()
        self.assertEqual(self.widget.pos().x(), 1000 - 176 - 12)

        self.widget.set_navigator_enabled(False)
        self.assertTrue(self.widget.isHidden())
        self.widget.set_navigator_enabled(True)
        self.assertFalse(self.widget.isHidden())

    def test_digital_slide_draws_mounted_and_visible_rectangles_in_global_space(self) -> None:
        self.widget.set_viewport_snapshot(
            _snapshot(
                mounted=QRectF(300.0, 250.0, 600.0, 300.0),
                visible=QRectF(450.0, 325.0, 300.0, 150.0),
            )
        )
        mounted = self.widget.map_image_rect_to_widget(
            QRectF(300.0, 250.0, 600.0, 300.0)
        )
        visible = self.widget.map_image_rect_to_widget(
            QRectF(450.0, 325.0, 300.0, 150.0)
        )
        self.assertTrue(mounted.contains(visible))
        self.assertLess(visible.width(), mounted.width())
        self.assertLess(visible.height(), mounted.height())

        target = QImage(
            self.widget.size(),
            QImage.Format.Format_ARGB32_Premultiplied,
        )
        target.fill(Qt.GlobalColor.transparent)
        self.widget.render(target)
        self.assertFalse(target.isNull())

    def test_click_and_drag_emit_clamped_global_image_centers(self) -> None:
        self.widget.set_viewport_snapshot(_snapshot())
        emitted: list[Point] = []
        self.widget.centerRequested.connect(emitted.append)
        content = self.widget.content_rect()

        press = QMouseEvent(
            QMouseEvent.Type.MouseButtonPress,
            content.center(),
            content.center(),
            content.center(),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        self.widget.mousePressEvent(press)
        move = QMouseEvent(
            QMouseEvent.Type.MouseMove,
            QPointF(content.right() + 30.0, content.bottom() + 30.0),
            QPointF(content.right() + 30.0, content.bottom() + 30.0),
            QPointF(content.right() + 30.0, content.bottom() + 30.0),
            Qt.MouseButton.NoButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        self.widget.mouseMoveEvent(move)
        release = QMouseEvent(
            QMouseEvent.Type.MouseButtonRelease,
            QPointF(content.right() + 30.0, content.bottom() + 30.0),
            QPointF(content.right() + 30.0, content.bottom() + 30.0),
            QPointF(content.right() + 30.0, content.bottom() + 30.0),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
        )
        self.widget.mouseReleaseEvent(release)

        self.assertGreaterEqual(len(emitted), 3)
        self.assertAlmostEqual(emitted[0].x, 600.0)
        self.assertAlmostEqual(emitted[0].y, 450.0)
        self.assertAlmostEqual(emitted[-1].x, 1100.0)
        self.assertAlmostEqual(emitted[-1].y, 700.0)

    def test_wheel_is_consumed_without_navigation_or_thumbnail_change(self) -> None:
        self.widget.set_viewport_snapshot(_snapshot())
        source = QImage(640, 320, QImage.Format.Format_RGB32)
        source.fill(Qt.GlobalColor.white)
        self.widget.set_source_image(source)
        before = self.widget.thumbnail_build_count
        emitted: list[Point] = []
        self.widget.centerRequested.connect(emitted.append)
        event = QWheelEvent(
            QPointF(20.0, 20.0),
            QPointF(20.0, 20.0),
            QPoint(),
            QPoint(0, 120),
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
            Qt.ScrollPhase.ScrollUpdate,
            False,
        )
        self.widget.wheelEvent(event)
        self.assertTrue(event.isAccepted())
        self.assertEqual(self.widget.thumbnail_build_count, before)
        self.assertEqual(emitted, [])

    def test_thumbnail_is_built_once_for_same_source_and_is_bounded(self) -> None:
        source = QImage(1200, 600, QImage.Format.Format_RGB32)
        source.fill(Qt.GlobalColor.red)
        self.widget.set_source_image(source)
        self.widget.set_source_image(source)
        self.assertEqual(self.widget.thumbnail_build_count, 1)
        self.assertLessEqual(self.widget._thumbnail.width(), 256)  # noqa: SLF001
        self.assertLessEqual(self.widget._thumbnail.height(), 256)  # noqa: SLF001

        source.setPixelColor(0, 0, Qt.GlobalColor.blue)
        self.widget.set_source_image(source)
        self.assertEqual(self.widget.thumbnail_build_count, 2)

    def test_dark_and_light_palettes_produce_readable_adaptive_colors(self) -> None:
        for palette in (build_dark_palette(), build_light_palette()):
            background, border, foreground, highlight = _navigator_colors(
                palette,
                hovered=False,
            )
            self.assertEqual(background.alpha(), 205)
            self.assertGreaterEqual(border.alpha(), 200)
            self.assertGreaterEqual(foreground.alpha(), 150)
            self.assertEqual(
                highlight.name(),
                palette.color(QPalette.ColorRole.Highlight).name(),
            )
            self.assertNotEqual(
                background.name(),
                foreground.name(),
            )


if __name__ == "__main__":
    unittest.main()
