from __future__ import annotations

import os
from pathlib import Path
from tempfile import TemporaryDirectory
import sys
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PySide6.QtCore import QPoint, QPointF, Qt
    from PySide6.QtGui import QImage, QColor
    from PySide6.QtWidgets import QApplication, QListView

    PYSIDE_AVAILABLE = True
except ModuleNotFoundError:
    PYSIDE_AVAILABLE = False

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm.geometry import Line, Point
from fdm.models import Calibration, ImageDocument, Measurement, new_id
from fdm.services.export_service import ExportImageRenderMode

if PYSIDE_AVAILABLE:
    from fdm.ui.canvas import DocumentCanvas
    from fdm.ui.main_window import MainWindow
else:
    DocumentCanvas = object  # type: ignore[assignment]
    MainWindow = object  # type: ignore[assignment]


class FakeWheelEvent:
    def __init__(self, position: QPointF, *, delta_x: int = 0, delta_y: int = 0) -> None:
        self._position = position
        self._delta = QPoint(delta_x, delta_y)

    def position(self) -> QPointF:
        return self._position

    def angleDelta(self) -> QPoint:
        return self._delta


class FakeMouseEvent:
    def __init__(self, position, *, button=None, modifiers=None) -> None:
        self._position = position
        self._button = button if button is not None else (Qt.MouseButton.LeftButton if PYSIDE_AVAILABLE else None)
        self._modifiers = modifiers if modifiers is not None else (Qt.KeyboardModifier.NoModifier if PYSIDE_AVAILABLE else None)

    def position(self) -> QPointF:
        return self._position

    def button(self):
        return self._button

    def modifiers(self):
        return self._modifiers


@unittest.skipUnless(PYSIDE_AVAILABLE, "requires PySide6")
class CanvasAndExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def _create_canvas_document(self) -> tuple[ImageDocument, QImage, DocumentCanvas]:
        image = QImage(200, 120, QImage.Format.Format_RGB32)
        image.fill(QColor("#FFFFFF"))
        document = ImageDocument(
            id=new_id("image"),
            path="/tmp/canvas_test.png",
            image_size=(image.width(), image.height()),
        )
        document.initialize_runtime_state()
        canvas = DocumentCanvas()
        canvas.resize(320, 240)
        canvas.set_document(document, image)
        return document, image, canvas

    def _create_main_window_fixture(self) -> tuple[MainWindow, ImageDocument]:
        window = MainWindow()
        image = QImage(260, 180, QImage.Format.Format_RGB32)
        image.fill(QColor("#FFFFFF"))
        document = ImageDocument(
            id=new_id("image"),
            path="/tmp/export_visibility.png",
            image_size=(image.width(), image.height()),
        )
        document.initialize_runtime_state()
        group = document.create_group(color="#1F7A8C", label="棉")
        document.set_active_group(group.id)
        document.calibration = Calibration(
            mode="preset",
            pixels_per_unit=5.0,
            unit="um",
            source_label="demo",
        )
        measurement = Measurement(
            id=new_id("meas"),
            image_id=document.id,
            fiber_group_id=group.id,
            mode="manual",
            line_px=Line(Point(20, 30), Point(200, 140)),
            confidence=1.0,
            status="manual",
        )
        document.add_measurement(measurement)
        canvas = DocumentCanvas()
        canvas.resize(340, 240)
        canvas.set_document(document, image)

        window._images[document.id] = image
        window._canvases[document.id] = canvas
        return window, document

    def _count_diff_pixels(self, left: QImage, right: QImage) -> int:
        self.assertEqual(left.size(), right.size())
        diff = 0
        for y in range(left.height()):
            for x in range(left.width()):
                if left.pixel(x, y) != right.pixel(x, y):
                    diff += 1
        return diff

    def test_wheel_event_uses_horizontal_delta_fallback_for_zoom_in(self) -> None:
        _, _, canvas = self._create_canvas_document()
        before_zoom = canvas._zoom

        canvas.wheelEvent(FakeWheelEvent(QPointF(120.0, 80.0), delta_x=120, delta_y=0))

        self.assertGreater(canvas._zoom, before_zoom)

    def test_calibration_mode_does_not_grab_measurement_handles(self) -> None:
        document, _, canvas = self._create_canvas_document()
        measurement = Measurement(
            id=new_id("meas"),
            image_id=document.id,
            fiber_group_id=None,
            mode="manual",
            line_px=Line(Point(20, 20), Point(80, 20)),
        )
        document.add_measurement(measurement)
        document.view_state.selected_measurement_id = measurement.id
        canvas.set_selected_measurement(measurement.id)
        canvas.set_tool_mode("calibration")

        endpoint_position = canvas.image_to_widget(Point(20, 20))
        canvas.mousePressEvent(FakeMouseEvent(endpoint_position))

        self.assertIsNone(canvas._dragging_handle)
        self.assertIsNotNone(canvas._drawing_line)

    def test_overlay_exports_render_visible_pixels_in_all_modes(self) -> None:
        window, document = self._create_main_window_fixture()

        try:
            with TemporaryDirectory() as tmp_dir:
                for render_mode in [
                    ExportImageRenderMode.FULL_RESOLUTION,
                    ExportImageRenderMode.SCREEN_SCALE_FULL_IMAGE,
                    ExportImageRenderMode.CURRENT_VIEWPORT,
                ]:
                    baseline_path = Path(tmp_dir) / f"baseline_{render_mode}.png"
                    measurement_path = Path(tmp_dir) / f"measurement_{render_mode}.png"
                    scale_path = Path(tmp_dir) / f"scale_{render_mode}.png"

                    window._render_overlay_image(
                        document,
                        baseline_path,
                        include_measurements=False,
                        include_scale=False,
                        render_mode=render_mode,
                    )
                    window._render_overlay_image(
                        document,
                        measurement_path,
                        include_measurements=True,
                        include_scale=False,
                        render_mode=render_mode,
                    )
                    window._render_overlay_image(
                        document,
                        scale_path,
                        include_measurements=False,
                        include_scale=True,
                        render_mode=render_mode,
                    )

                    baseline_image = QImage(str(baseline_path))
                    measurement_image = QImage(str(measurement_path))
                    scale_image = QImage(str(scale_path))

                    self.assertFalse(measurement_image.isNull())
                    self.assertFalse(scale_image.isNull())
                    self.assertGreater(self._count_diff_pixels(baseline_image, measurement_image), 0)
                    self.assertGreater(self._count_diff_pixels(baseline_image, scale_image), 0)
        finally:
            window.close()

    def test_group_list_keeps_color_icon_and_hides_uncategorized_after_first_group(self) -> None:
        window = MainWindow()
        try:
            document = ImageDocument(
                id=new_id("image"),
                path="/tmp/group_list.png",
                image_size=(300, 200),
            )
            document.initialize_runtime_state()

            window._populate_group_list(document)
            self.assertEqual(window.group_list.viewMode(), QListView.ViewMode.ListMode)
            self.assertEqual(window.group_list.count(), 1)
            self.assertFalse(window.group_list.item(0).icon().isNull())

            group = document.create_group(color="#1F7A8C", label="棉")
            document.set_active_group(group.id)
            window._populate_group_list(document)

            self.assertEqual(window.group_list.count(), 1)
            item = window.group_list.item(0)
            self.assertIn("棉", item.text())
            self.assertFalse(item.icon().isNull())
        finally:
            window.close()

    def test_full_resolution_metrics_scale_above_old_cap_for_large_images(self) -> None:
        window = MainWindow()
        try:
            metrics = window._overlay_metrics(12000, 8000, ExportImageRenderMode.FULL_RESOLUTION)
            self.assertLessEqual(metrics["line_width"], 2.5)
            self.assertLessEqual(metrics["font_px"], 20.0)
            self.assertGreater(metrics["line_width"], 1.0)
            self.assertGreater(metrics["endpoint_radius"], metrics["line_width"])
        finally:
            window.close()

    def test_full_resolution_export_remains_visible_after_preview_downscale(self) -> None:
        window = MainWindow()
        try:
            image = QImage(6000, 4000, QImage.Format.Format_RGB32)
            image.fill(QColor("#FFFFFF"))
            document = ImageDocument(
                id=new_id("image"),
                path="/tmp/fullres_preview.png",
                image_size=(image.width(), image.height()),
            )
            document.initialize_runtime_state()
            group = document.create_group(color="#E07A5F", label="棉")
            document.set_active_group(group.id)
            document.add_measurement(
                Measurement(
                    id=new_id("meas"),
                    image_id=document.id,
                    fiber_group_id=group.id,
                    mode="manual",
                    line_px=Line(Point(500, 800), Point(5200, 3200)),
                )
            )
            window._images[document.id] = image

            with TemporaryDirectory() as tmp_dir:
                baseline_path = Path(tmp_dir) / "fullres_baseline.png"
                measurement_path = Path(tmp_dir) / "fullres_measurement.png"
                window._render_overlay_image(
                    document,
                    baseline_path,
                    include_measurements=False,
                    include_scale=False,
                    render_mode=ExportImageRenderMode.FULL_RESOLUTION,
                )
                window._render_overlay_image(
                    document,
                    measurement_path,
                    include_measurements=True,
                    include_scale=False,
                    render_mode=ExportImageRenderMode.FULL_RESOLUTION,
                )

                baseline_preview = QImage(str(baseline_path)).scaledToWidth(1400)
                measurement_preview = QImage(str(measurement_path)).scaledToWidth(1400)
                self.assertGreater(self._count_diff_pixels(baseline_preview, measurement_preview), 500)
        finally:
            window.close()

    def test_full_resolution_export_works_for_non_paintable_source_format(self) -> None:
        if not hasattr(QImage.Format, "Format_CMYK8888"):
            self.skipTest("current Qt build does not expose CMYK8888")

        window = MainWindow()
        try:
            image = QImage(800, 600, getattr(QImage.Format, "Format_CMYK8888"))
            image.fill(QColor("#FFFFFF"))
            document = ImageDocument(
                id=new_id("image"),
                path="/tmp/fullres_cmyk.jpg",
                image_size=(image.width(), image.height()),
            )
            document.initialize_runtime_state()
            group = document.create_group(color="#E07A5F", label="棉")
            document.set_active_group(group.id)
            document.add_measurement(
                Measurement(
                    id=new_id("meas"),
                    image_id=document.id,
                    fiber_group_id=group.id,
                    mode="manual",
                    line_px=Line(Point(100, 100), Point(700, 500)),
                )
            )
            window._images[document.id] = image

            with TemporaryDirectory() as tmp_dir:
                baseline_path = Path(tmp_dir) / "cmyk_baseline.png"
                measurement_path = Path(tmp_dir) / "cmyk_measurement.png"
                window._render_overlay_image(
                    document,
                    baseline_path,
                    include_measurements=False,
                    include_scale=False,
                    render_mode=ExportImageRenderMode.FULL_RESOLUTION,
                )
                window._render_overlay_image(
                    document,
                    measurement_path,
                    include_measurements=True,
                    include_scale=False,
                    render_mode=ExportImageRenderMode.FULL_RESOLUTION,
                )

                baseline_image = QImage(str(baseline_path))
                measurement_image = QImage(str(measurement_path))
                self.assertFalse(measurement_image.isNull())
                self.assertGreater(self._count_diff_pixels(baseline_image, measurement_image), 0)
        finally:
            window.close()


if __name__ == "__main__":
    unittest.main()
