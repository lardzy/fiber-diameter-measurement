from __future__ import annotations

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

import cv2

from fdm.geometry import Point, distance
from fdm.services.fiber_quick_geometry import FiberQuickDiameterGeometryService, _draw_block_circle


@unittest.skipIf(np is None, "requires numpy")
class FiberQuickGeometryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = FiberQuickDiameterGeometryService()

    def _blank_mask(self, width: int = 160, height: int = 120):
        return np.zeros((height, width), dtype=bool)

    def test_returns_stable_line_for_rectangular_fiber_mask(self) -> None:
        mask = self._blank_mask()
        mask[35:85, 50:110] = True

        result = self.service.measure_from_mask(mask)

        self.assertIsNotNone(result.line_px)
        width_px = distance(result.line_px.start, result.line_px.end)
        self.assertGreater(width_px, 45.0)
        self.assertLess(width_px, 55.0)
        self.assertEqual(result.status, "fiber_quick")
        self.assertGreaterEqual(len(result.preview_polygon_px), 3)

    def test_returns_line_for_gently_curved_fiber_mask(self) -> None:
        mask = self._blank_mask(220, 180).astype(np.uint8)
        points = np.array(
            [
                [30, 120],
                [70, 90],
                [110, 70],
                [150, 80],
                [190, 110],
            ],
            dtype=np.int32,
        )
        cv2.polylines(mask, [points], False, 1, thickness=28)

        result = self.service.measure_from_mask(mask.astype(bool))

        self.assertIsNotNone(result.line_px)
        self.assertGreater(distance(result.line_px.start, result.line_px.end), 20.0)
        self.assertGreater(result.confidence, 0.2)

    def test_avoids_crossing_center_for_cross_mask(self) -> None:
        mask = self._blank_mask(220, 220).astype(np.uint8)
        cv2.rectangle(mask, (90, 20), (130, 200), 1, thickness=-1)
        cv2.rectangle(mask, (20, 90), (200, 130), 1, thickness=-1)

        result = self.service.measure_from_mask(mask.astype(bool))

        self.assertIsNotNone(result.line_px)
        midpoint_x = (result.line_px.start.x + result.line_px.end.x) / 2.0
        midpoint_y = (result.line_px.start.y + result.line_px.end.y) / 2.0
        self.assertTrue(abs(midpoint_x - 110.0) > 12.0 or abs(midpoint_y - 110.0) > 12.0)

    def test_raises_for_tiny_noisy_component(self) -> None:
        mask = self._blank_mask(40, 40)
        mask[18:21, 19:22] = True

        with self.assertRaises(RuntimeError):
            self.service.measure_from_mask(mask)

    def test_fails_when_mask_touches_image_edge_across_large_span(self) -> None:
        mask = self._blank_mask(220, 160).astype(np.uint8)
        points = np.array(
            [
                [0, 120],
                [40, 90],
                [100, 55],
                [170, 30],
                [219, 18],
                [219, 42],
                [168, 54],
                [95, 82],
                [36, 118],
                [0, 148],
            ],
            dtype=np.int32,
        )
        cv2.fillPoly(mask, [points], 1)

        with self.assertRaisesRegex(RuntimeError, "视野边缘不完整"):
            self.service.measure_from_mask(mask.astype(bool))

    def test_fails_fast_for_border_touching_mask(self) -> None:
        mask = self._blank_mask(220, 160)
        mask[40:120, 0:58] = True

        with self.assertRaisesRegex(RuntimeError, "视野边缘不完整"):
            self.service.measure_from_mask(mask)

    def test_fails_fast_for_overlarge_mask(self) -> None:
        mask = self._blank_mask(220, 160)
        mask[16:148, 20:200] = True

        with self.assertRaisesRegex(RuntimeError, "范围过大"):
            self.service.measure_from_mask(mask)

    def test_draw_block_circle_supports_bool_masks(self) -> None:
        mask = self._blank_mask(32, 32)
        mask[8:24, 8:24] = True

        _draw_block_circle(mask, point=Point(16.0, 16.0), radius=3, value=0)

        self.assertFalse(mask[16, 16])


if __name__ == "__main__":
    unittest.main()
