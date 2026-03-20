from __future__ import annotations

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm.geometry import Line, Point
from fdm.raster import RasterImage, extract_rotated_roi
from fdm.services.snap_service import SnapService


def make_vertical_fiber_image(width: int = 120, height: int = 80, *, x0: int = 54, x1: int = 66) -> RasterImage:
    image = RasterImage.blank(width, height, fill=235)
    for y in range(height):
        for x in range(x0, x1):
            image.set(x, y, 25)
    return image


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


if __name__ == "__main__":
    unittest.main()
