from __future__ import annotations

import math
from pathlib import Path
from time import perf_counter
import sys
import unittest

from PySide6.QtCore import Qt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import fdm.geometry as geometry
from fdm.geometry import Point, area_rings_area, area_rings_centroid, polygon_area, polygon_centroid
from fdm.models import Calibration, Measurement


class OddEvenGeometryTests(unittest.TestCase):
    def test_qt_odd_even_simplified_path_splits_bow_tie(self) -> None:
        bow_tie = [Point(0, 0), Point(10, 10), Point(0, 10), Point(10, 0)]

        path = geometry._simplified_odd_even_path([bow_tie])  # noqa: SLF001

        self.assertEqual(path.fillRule(), Qt.FillRule.OddEvenFill)
        self.assertEqual(len(path.toSubpathPolygons()), 2)
        self.assertAlmostEqual(polygon_area(bow_tie), 50.0)
        center = polygon_centroid(bow_tie)
        self.assertAlmostEqual(center.x, 5.0)
        self.assertAlmostEqual(center.y, 5.0)

    def test_concave_polygon_area_and_centroid(self) -> None:
        concave = [
            Point(0, 0),
            Point(6, 0),
            Point(6, 2),
            Point(2, 2),
            Point(2, 6),
            Point(0, 6),
        ]

        self.assertAlmostEqual(polygon_area(concave), 20.0)
        center = polygon_centroid(concave)
        self.assertAlmostEqual(center.x, 2.2)
        self.assertAlmostEqual(center.y, 2.2)

    def test_odd_even_hole_area_and_weighted_centroid_ignore_ring_orientation(self) -> None:
        outer = [Point(0, 0), Point(20, 0), Point(20, 20), Point(0, 20)]
        hole = [Point(2, 2), Point(6, 2), Point(6, 10), Point(2, 10)]
        expected_x = ((400.0 * 10.0) - (32.0 * 4.0)) / 368.0
        expected_y = ((400.0 * 10.0) - (32.0 * 6.0)) / 368.0

        for hole_ring in (hole, list(reversed(hole))):
            with self.subTest(reversed=hole_ring is not hole):
                rings = [outer, hole_ring]
                self.assertAlmostEqual(area_rings_area(rings), 368.0)
                center = area_rings_centroid(rings)
                self.assertAlmostEqual(center.x, expected_x)
                self.assertAlmostEqual(center.y, expected_y)

    def test_exact_mask_area_remains_authoritative(self) -> None:
        bow_tie = [Point(0, 0), Point(10, 10), Point(0, 10), Point(10, 0)]
        measurement = Measurement(
            id="area_exact_qt",
            image_id="image_qt",
            fiber_group_id=None,
            mode="magic_segment",
            measurement_kind="area",
            polygon_px=bow_tie,
            exact_area_px=123.0,
        )
        calibration = Calibration(
            mode="preset",
            pixels_per_unit=2.0,
            unit="um",
            source_label="demo",
        )

        measurement.recalculate(calibration)

        self.assertAlmostEqual(polygon_area(bow_tie), 50.0)
        self.assertEqual(measurement.area_px, 123.0)
        self.assertAlmostEqual(measurement.area_unit or 0.0, 30.75)

    def test_large_simple_polygon_avoids_quadratic_scanline_regression(self) -> None:
        point_count = 10_000
        radius = 100.0
        points = [
            Point(
                radius * math.cos((2.0 * math.pi * index) / point_count),
                radius * math.sin((2.0 * math.pi * index) / point_count),
            )
            for index in range(point_count)
        ]

        started = perf_counter()
        area = polygon_area(points)
        elapsed = perf_counter() - started

        self.assertAlmostEqual(area, math.pi * radius * radius, delta=2.0)
        self.assertLess(elapsed, 2.0)


if __name__ == "__main__":
    unittest.main()
