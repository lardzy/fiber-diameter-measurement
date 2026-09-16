from __future__ import annotations

from pathlib import Path
import json
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

import cv2

from fdm.geometry import Line, Point, distance
from fdm.services.fiber_quick_geometry import (
    FiberQuickDiameterGeometryService,
    FiberQuickGeometryError,
    _CandidateLine,
    _apply_line_extension,
    _compute_skeleton,
    _draw_block_circle,
    _measure_line,
    _pick_representative_candidate,
    prepare_fiber_quick_geometry_backend,
)
from fdm.ui.fiber_quick_geometry_worker import FiberQuickGeometryRequest, FiberQuickGeometryWorker


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
        self.assertAlmostEqual(distance(result.line_px.start, result.line_px.end), 40.0, delta=2.0)

    def test_compiled_skeleton_preserves_a_connected_long_fiber(self) -> None:
        mask = np.zeros((90, 500), dtype=bool)
        mask[25:65, 30:470] = True
        skeleton = _compute_skeleton(mask)
        self.assertEqual(cv2.connectedComponents(skeleton.astype(np.uint8), connectivity=8)[0], 2)
        self.assertGreater(int(skeleton.sum()), 380)

    def test_missing_compiled_backend_fails_instead_of_using_a_different_algorithm(self) -> None:
        prepare_fiber_quick_geometry_backend.cache_clear()
        try:
            with patch.dict(sys.modules, {"skimage.morphology": None}):
                with self.assertRaisesRegex(RuntimeError, "scikit-image"):
                    _compute_skeleton(np.ones((10, 10), dtype=bool))
        finally:
            prepare_fiber_quick_geometry_backend.cache_clear()

    def test_width_and_direction_across_angles_and_roi_resizing(self) -> None:
        for size in (512, 1024):
            ys, xs = np.mgrid[:size, :size]
            dx, dy = xs - size / 2, ys - size / 2
            for angle in (0, 15, 45, 75, 90, 135):
                theta = np.deg2rad(angle)
                along = dx * np.cos(theta) + dy * np.sin(theta)
                across = -dx * np.sin(theta) + dy * np.cos(theta)
                for width in (8, 24, 60):
                    with self.subTest(size=size, angle=angle, width=width):
                        mask = (np.abs(along) < size * 0.4) & (np.abs(across) < width / 2)
                        result = self.service.measure_from_mask(mask)
                        line = result.line_px
                        self.assertIsNotNone(line)
                        self.assertAlmostEqual(distance(line.start, line.end), width, delta=2.1)
                        axis = np.array([line.end.x - line.start.x, line.end.y - line.start.y])
                        axis /= np.linalg.norm(axis)
                        self.assertLess(abs(axis @ np.array([np.cos(theta), np.sin(theta)])), 0.08)

    def test_crossing_exclusion_scales_with_fiber_width_and_angle(self) -> None:
        ys, xs = np.mgrid[:512, :512]
        dx, dy = xs - 256, ys - 256
        for angle in (0, 5, 20, 25, 40, 45, 75, 85):
            theta = np.deg2rad(angle)
            along = dx * np.cos(theta) + dy * np.sin(theta)
            across = -dx * np.sin(theta) + dy * np.cos(theta)
            for width in (8, 16, 40, 80):
                with self.subTest(angle=angle, width=width):
                    mask = (
                        ((np.abs(along) < 175) & (np.abs(across) < width / 2))
                        | ((np.abs(across) < 175) & (np.abs(along) < width / 2))
                    )
                    result = self.service.measure_from_mask(mask)
                    line = result.line_px
                    self.assertAlmostEqual(distance(line.start, line.end), width, delta=2.1)
                    axis = np.array([line.end.x - line.start.x, line.end.y - line.start.y])
                    axis /= np.linalg.norm(axis)
                    self.assertLess(min(
                        abs(axis @ np.array([np.cos(theta), np.sin(theta)])),
                        abs(axis @ np.array([-np.sin(theta), np.cos(theta)])),
                    ), 0.08)

    def test_wide_boundary_search_reaches_both_edges(self) -> None:
        mask = np.zeros((100, 1400), dtype=bool)
        mask[30:70, 100:1300] = True
        line, hit_border = _measure_line(mask, Point(700, 50), (1, 0))
        self.assertFalse(hit_border)
        self.assertEqual(line, Line(Point(100, 50), Point(1299, 50)))

    def test_short_rotated_fibers_use_a_stable_direction(self) -> None:
        ys, xs = np.mgrid[:256, :256]
        for width, ratio in ((8, 1.5), (8, 2), (16, 1.2), (16, 1.5), (40, 1.2), (40, 2)):
            for angle in (0, 15, 30, 45, 75):
                with self.subTest(width=width, ratio=ratio, angle=angle):
                    theta = np.deg2rad(angle)
                    along = (xs - 128) * np.cos(theta) + (ys - 128) * np.sin(theta)
                    across = -(xs - 128) * np.sin(theta) + (ys - 128) * np.cos(theta)
                    mask = (abs(along) < width * ratio / 2) & (abs(across) < width / 2)
                    result = self.service.measure_from_mask(mask)
                    line = result.line_px
                    length = distance(line.start, line.end)
                    self.assertAlmostEqual(length, width, delta=2.1)
                    axis = np.array([line.end.x - line.start.x, line.end.y - line.start.y]) / length
                    self.assertLess(abs(axis @ np.array([np.cos(theta), np.sin(theta)])), 0.12)

    def test_complete_cross_sections_can_be_close_to_each_image_edge(self) -> None:
        for margin in (2, 4, 6, 10):
            mask = np.zeros((240, 320), dtype=bool)
            mask[margin:margin + 40, 50:270] = True
            for rotation in range(4):
                with self.subTest(margin=margin, rotation=rotation):
                    result = self.service.measure_from_mask(np.rot90(mask, rotation))
                    self.assertAlmostEqual(distance(result.line_px.start, result.line_px.end), 39, delta=1.5)
                    self.assertEqual(result.debug_payload["edge_trim_pixels"], 0)

    def test_incomplete_cross_sections_are_rejected_even_after_trimming(self) -> None:
        mask = np.zeros((200, 240), dtype=bool)
        mask[40:160, :25] = True
        for enabled in (True, False):
            for rotation in range(4):
                with self.subTest(trim=enabled, rotation=rotation):
                    with self.assertRaises(FiberQuickGeometryError) as error:
                        self.service.measure_from_mask(np.rot90(mask, rotation), edge_trim_enabled=enabled)
                    self.assertEqual(error.exception.code, "incomplete_boundary")
                    self.assertGreater(error.exception.debug_payload["rejection_counts"]["incomplete_boundary"], 0)
                    self.assertIn("两侧边界", str(error.exception))

    def test_short_crossing_branches_keep_width_and_direction(self) -> None:
        ys, xs = np.mgrid[:320, :320]
        for width, ratio, angle in ((16, 4, 0), (40, 3, 15), (40, 3, 75), (40, 4, 25)):
            with self.subTest(width=width, ratio=ratio, angle=angle):
                theta = np.deg2rad(angle)
                along = (xs - 160) * np.cos(theta) + (ys - 160) * np.sin(theta)
                across = -(xs - 160) * np.sin(theta) + (ys - 160) * np.cos(theta)
                mask = ((abs(along) < width * ratio / 2) & (abs(across) < width / 2)) | (
                    (abs(across) < width * ratio / 2) & (abs(along) < width / 2)
                )
                result = self.service.measure_from_mask(mask)
                line = result.line_px
                length = distance(line.start, line.end)
                self.assertAlmostEqual(length, width, delta=2.1)
                axis = np.array([line.end.x - line.start.x, line.end.y - line.start.y]) / length
                # Short branches have fewer directional pixels. A 10-degree
                # bound limits the angular width bias to roughly 1.5 percent.
                error = min(
                    abs(axis @ np.array([np.cos(theta), np.sin(theta)])),
                    abs(axis @ np.array([-np.sin(theta), np.cos(theta)])),
                )
                self.assertLess(error, np.sin(np.deg2rad(10)))

    def test_frame_spanning_fiber_can_have_a_complete_cross_section(self) -> None:
        mask = np.zeros((256, 320), dtype=bool)
        mask[116:140, :] = True
        result = self.service.measure_from_mask(mask)
        self.assertAlmostEqual(distance(result.line_px.start, result.line_px.end), 23, delta=1.5)

    def test_long_thin_fiber_keeps_enough_working_pixels(self) -> None:
        mask = np.zeros((160, 4096), dtype=bool)
        mask[77:83, 248:3848] = True
        result = self.service.measure_from_mask(mask)
        self.assertAlmostEqual(distance(result.line_px.start, result.line_px.end), 5, delta=1)
        self.assertGreater(result.debug_payload["roi_size"][0], 640)
        self.assertLessEqual(np.prod(result.debug_payload["roi_size"]), 640 * 640)

    def test_round_or_square_blobs_do_not_get_an_arbitrary_diameter(self) -> None:
        square = np.zeros((200, 240), dtype=np.uint8)
        square[80:120, 100:140] = 1
        circle = np.zeros_like(square)
        cv2.circle(circle, (120, 100), 25, 1, -1)
        for mask in (square, circle):
            with self.assertRaises(FiberQuickGeometryError) as error:
                self.service.measure_from_mask(mask)
            self.assertEqual(error.exception.code, "ambiguous_direction")

    def test_low_confidence_median_does_not_discard_reliable_candidates(self) -> None:
        candidates = [
            _CandidateLine(Line(Point(0, 0), Point(width, 0)), width, score)
            for width, score in ((10, 0.1), (9.5, 0.9), (10.5, 0.8))
        ]
        self.assertIs(_pick_representative_candidate(candidates), candidates[1])

    def test_failure_logs_reason_counts_and_keeps_the_user_message(self) -> None:
        worker = FiberQuickGeometryWorker()
        request = FiberQuickGeometryRequest("doc", 1, None, [], [], [], [])
        error = FiberQuickGeometryError("incomplete_boundary", "两侧边界不完整", {
            "rejection_counts": {"incomplete_boundary": 3},
        })
        failures = []
        worker.failed.connect(lambda *args: failures.append(args))
        with (
            patch.object(worker._service, "measure_from_mask", side_effect=error),
            patch("fdm.ui.fiber_quick_geometry_worker.append_runtime_log") as log,
        ):
            worker.register_request("doc", 1)
            worker.measure(request)
            self.assertEqual(failures, [("doc", 1, "两侧边界不完整")])
            payload = json.loads(log.call_args.args[1])
            self.assertEqual(payload["failure_code"], "incomplete_boundary")
            self.assertEqual(payload["rejection_counts"], {"incomplete_boundary": 3})
            self.assertEqual(payload["request_id"], 1)
            log.reset_mock()
            worker.register_request("doc", 2)
            worker.measure(request)
            log.assert_not_called()
            self.assertEqual(len(failures), 1)

    def test_boundary_search_reports_incomplete_image_edge(self) -> None:
        mask = np.ones((30, 50), dtype=bool)
        _, hit_border = _measure_line(mask, Point(25, 15), (1, 0))
        self.assertTrue(hit_border)

    def test_cancelled_geometry_stops_before_loading_the_kernel(self) -> None:
        with patch("fdm.services.fiber_quick_geometry.prepare_fiber_quick_geometry_backend") as prepare:
            with self.assertRaisesRegex(RuntimeError, "取消"):
                _compute_skeleton(np.ones((30, 30)), cancel_check=lambda: True)
        prepare.assert_not_called()

    def test_raises_for_tiny_noisy_component(self) -> None:
        mask = self._blank_mask(40, 40)
        mask[18:21, 19:22] = True

        with self.assertRaises(RuntimeError):
            self.service.measure_from_mask(mask)

    def test_edge_trim_can_salvage_border_touching_mask(self) -> None:
        mask = self._blank_mask(220, 160).astype(np.uint8)
        points = np.array(
            [
                [0, 110],
                [24, 96],
                [74, 66],
                [120, 50],
                [150, 38],
                [150, 62],
                [118, 74],
                [70, 92],
                [22, 122],
                [0, 138],
            ],
            dtype=np.int32,
        )
        cv2.fillPoly(mask, [points], 1)

        result = self.service.measure_from_mask(mask.astype(bool))

        self.assertIsNotNone(result.line_px)
        self.assertTrue(bool(result.debug_payload.get("edge_trim_enabled")))
        self.assertGreater(int(result.debug_payload.get("edge_trim_pixels", 0) or 0), 0)

    def test_border_touching_mask_can_still_fail_without_edge_trim(self) -> None:
        mask = self._blank_mask(220, 160)
        mask[40:120, 0:58] = True

        with self.assertRaisesRegex(RuntimeError, "未找到可靠直径线|视野边缘不完整"):
            self.service.measure_from_mask(mask, edge_trim_enabled=False)

    def test_fails_fast_for_overlarge_mask(self) -> None:
        mask = self._blank_mask(220, 160)
        mask[16:148, 20:200] = True

        with self.assertRaisesRegex(RuntimeError, "范围过大"):
            self.service.measure_from_mask(mask)

    def test_negative_line_extension_shrinks_final_line(self) -> None:
        mask = self._blank_mask()
        mask[35:85, 50:110] = True

        baseline = self.service.measure_from_mask(mask, line_extension_px=0.0)
        shrunk = self.service.measure_from_mask(mask, line_extension_px=-5.0)

        self.assertIsNotNone(baseline.line_px)
        self.assertIsNotNone(shrunk.line_px)
        self.assertAlmostEqual(distance(shrunk.line_px.start, shrunk.line_px.end), distance(baseline.line_px.start, baseline.line_px.end) - 10.0)
        self.assertAlmostEqual(float(shrunk.debug_payload.get("line_extension_px", 0.0) or 0.0), -5.0)

    def test_positive_line_extension_extends_endpoints_along_line_direction(self) -> None:
        line = Line(Point(30, 60), Point(89, 60))

        extended = _apply_line_extension(
            line,
            extension_px=4.0,
            cancel_check=None,
            deadline_at=None,
        )

        self.assertEqual(extended, Line(Point(26, 60), Point(93, 60)))

    def test_line_extension_corrects_real_measurements_in_original_pixels(self) -> None:
        for size, angle in ((256, 0), (256, 35), (1024, 25)):
            ys, xs = np.mgrid[:size, :size]
            theta = np.deg2rad(angle)
            along = (xs - size / 2) * np.cos(theta) + (ys - size / 2) * np.sin(theta)
            across = -(xs - size / 2) * np.sin(theta) + (ys - size / 2) * np.cos(theta)
            mask = (abs(along) < size * 0.4) & (abs(across) < 12)
            baseline = self.service.measure_from_mask(mask)
            line = baseline.line_px
            width = distance(line.start, line.end)
            for extension in (0.5, 3.5, -0.5, -3.5):
                with self.subTest(size=size, angle=angle, extension=extension):
                    corrected = self.service.measure_from_mask(mask, line_extension_px=extension)
                    result = corrected.line_px
                    self.assertAlmostEqual(distance(result.start, result.end), width + 2 * extension)
                    self.assertAlmostEqual(result.start.x + result.end.x, line.start.x + line.end.x)
                    self.assertAlmostEqual(result.start.y + result.end.y, line.start.y + line.end.y)
                    self.assertAlmostEqual(corrected.debug_payload["uncorrected_diameter_px"], width)
                    self.assertAlmostEqual(corrected.debug_payload["diameter_correction_px"], 2 * extension)

    def test_large_negative_extension_preserves_a_non_inverted_line(self) -> None:
        line = Line(Point(10.25, 12.75), Point(13.25, 16.75))
        result = _apply_line_extension(line, extension_px=-20, cancel_check=None, deadline_at=None)
        self.assertAlmostEqual(distance(result.start, result.end), 2.0)
        self.assertAlmostEqual(result.start.x + result.end.x, 23.5)
        self.assertAlmostEqual(result.start.y + result.end.y, 29.5)
        self.assertGreater(result.end.x, result.start.x)
        self.assertGreater(result.end.y, result.start.y)

    def test_draw_block_circle_supports_bool_masks(self) -> None:
        mask = self._blank_mask(32, 32)
        mask[8:24, 8:24] = True

        _draw_block_circle(mask, point=Point(16.0, 16.0), radius=3, value=0)

        self.assertFalse(mask[16, 16])

    def test_background_geometry_worker_does_not_stale_confirmed_jobs(self) -> None:
        worker = FiberQuickGeometryWorker(coalesce_latest=False)

        worker.register_request("doc_1", 1)
        worker.register_request("doc_1", 2)

        self.assertFalse(worker._is_request_stale("doc_1", 1))  # noqa: SLF001
        self.assertFalse(worker._is_request_stale("doc_1", 2))  # noqa: SLF001


if __name__ == "__main__":
    unittest.main()
