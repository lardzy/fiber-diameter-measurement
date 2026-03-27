from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    from PySide6.QtGui import QColor, QImage

    PYSIDE_AVAILABLE = True
except ModuleNotFoundError:
    PYSIDE_AVAILABLE = False

try:
    import numpy as np  # noqa: F401

    NUMPY_AVAILABLE = True
except ModuleNotFoundError:
    NUMPY_AVAILABLE = False

from fdm.geometry import Line, Point, direction
from fdm.raster import RasterImage, extract_rotated_roi
from fdm.services.snap_service import SnapService


def make_vertical_fiber_image(width: int = 120, height: int = 80, *, x0: int = 54, x1: int = 66) -> RasterImage:
    image = RasterImage.blank(width, height, fill=235)
    for y in range(height):
        for x in range(x0, x1):
            image.set(x, y, 25)
    return image


def make_bright_vertical_fiber_image(width: int = 120, height: int = 80, *, x0: int = 54, x1: int = 66) -> RasterImage:
    image = RasterImage.blank(width, height, fill=25)
    for y in range(height):
        for x in range(x0, x1):
            image.set(x, y, 235)
    return image


def make_low_contrast_fiber_image(width: int = 120, height: int = 80, *, x0: int = 54, x1: int = 66) -> RasterImage:
    image = RasterImage.blank(width, height, fill=128)
    for y in range(height):
        for x in range(x0, x1):
            image.set(x, y, 126)
    return image


def make_layered_fiber_image(width: int = 120, height: int = 80, *, x0: int = 48, x1: int = 72, cx0: int = 58, cx1: int = 62) -> RasterImage:
    image = RasterImage.blank(width, height, fill=235)
    for y in range(height):
        for x in range(x0, x1):
            image.set(x, y, 80)
        for x in range(cx0, cx1):
            image.set(x, y, 20)
    return image


@unittest.skipUnless(NUMPY_AVAILABLE, "requires numpy")
class RasterAndSnapTests(unittest.TestCase):
    def test_extract_rotated_roi_preserves_midpoint(self) -> None:
        image = make_vertical_fiber_image()
        line = Line(Point(20, 40), Point(100, 40))
        roi = extract_rotated_roi(image, line, padding=20, half_height=20)
        mapped_midpoint = roi.map_roi_to_image(roi.midpoint)
        self.assertAlmostEqual(mapped_midpoint.x, 60.0, places=1)
        self.assertAlmostEqual(mapped_midpoint.y, 40.0, places=1)
        self.assertEqual(roi.height, 40)
        self.assertGreaterEqual(roi.width, 120)

    def test_snap_service_finds_fiber_edges(self) -> None:
        image = make_vertical_fiber_image()
        line = Line(Point(30, 40), Point(90, 40))
        result = SnapService().snap_measurement(image, line)
        self.assertEqual(result.status, "snapped")
        self.assertIsNotNone(result.snapped_line)
        self.assertIsNotNone(result.diameter_px)
        self.assertAlmostEqual(result.diameter_px or 0.0, 12.0, delta=2.0)
        self.assertGreater(result.confidence, 0.1)

    def test_snap_service_rejects_too_short_line(self) -> None:
        image = make_vertical_fiber_image()
        line = Line(Point(10, 10), Point(11, 10))
        result = SnapService().snap_measurement(image, line)
        self.assertEqual(result.status, "line_too_short")
        self.assertIsNone(result.snapped_line)

    def test_snap_service_preserves_user_line_angle(self) -> None:
        image = make_vertical_fiber_image()
        line = Line(Point(32, 30), Point(88, 50))
        result = SnapService().snap_measurement(image, line)

        self.assertEqual(result.status, "snapped")
        self.assertIsNotNone(result.snapped_line)

        input_direction = direction(line)
        snapped_direction = direction(result.snapped_line)
        alignment = abs((input_direction[0] * snapped_direction[0]) + (input_direction[1] * snapped_direction[1]))

        self.assertGreater(alignment, 0.995)
        self.assertTrue(result.debug_payload.get("angle_preserved"))

    def test_snap_service_handles_bright_fiber_on_dark_background(self) -> None:
        image = make_bright_vertical_fiber_image()
        line = Line(Point(30, 40), Point(90, 40))

        result = SnapService().snap_measurement(image, line)

        self.assertEqual(result.status, "snapped")
        self.assertIsNotNone(result.snapped_line)
        self.assertAlmostEqual(result.diameter_px or 0.0, 12.0, delta=2.0)
        self.assertEqual(result.debug_payload.get("polarity"), "light_on_dark")

    def test_snap_service_reports_flat_profile_for_low_contrast_image(self) -> None:
        image = make_low_contrast_fiber_image()
        line = Line(Point(30, 40), Point(90, 40))

        result = SnapService().snap_measurement(image, line)

        self.assertIn(result.status, {"profile_too_flat", "edge_pair_not_found"})
        self.assertIsNone(result.snapped_line)

    def test_snap_service_prefers_outer_edges_over_internal_texture(self) -> None:
        image = make_layered_fiber_image()
        line = Line(Point(30, 40), Point(90, 40))

        result = SnapService().snap_measurement(image, line)

        self.assertEqual(result.status, "snapped")
        self.assertIsNotNone(result.snapped_line)
        self.assertAlmostEqual(result.snapped_line.start.x, 48.0, delta=2.0)
        self.assertAlmostEqual(result.snapped_line.end.x, 72.0, delta=2.0)
        self.assertGreater((result.diameter_px or 0.0), 16.0)

    def test_snap_service_uses_local_roi_for_grayscale_sampling(self) -> None:
        image = make_vertical_fiber_image(width=400, height=240, x0=190, x1=210)
        line = Line(Point(120, 120), Point(280, 120))
        service = SnapService()

        with patch.object(service, "_to_grayscale_array", wraps=service._to_grayscale_array) as grayscale_mock:
            result = service.snap_measurement(image, line)

        self.assertEqual(result.status, "snapped")
        _image_arg = grayscale_mock.call_args.args[0]
        roi = grayscale_mock.call_args.kwargs.get("roi")
        self.assertIsNotNone(roi)
        self.assertLess(roi[2], image.width)
        self.assertLess(roi[3], image.height)

    @unittest.skipUnless(PYSIDE_AVAILABLE, "requires PySide6")
    def test_snap_service_accepts_qimage_input(self) -> None:
        image = QImage(120, 80, QImage.Format.Format_RGB32)
        image.fill(QColor("#EBEBEB"))
        for y in range(image.height()):
            for x in range(54, 66):
                image.setPixelColor(x, y, QColor("#191919"))
        line = Line(Point(30, 40), Point(90, 40))

        result = SnapService().snap_measurement(image, line)

        self.assertEqual(result.status, "snapped")
        self.assertIsNotNone(result.snapped_line)
        self.assertAlmostEqual(result.diameter_px or 0.0, 12.0, delta=2.0)


if __name__ == "__main__":
    unittest.main()
