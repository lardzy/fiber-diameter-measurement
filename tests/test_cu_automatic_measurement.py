from __future__ import annotations

import unittest
from pathlib import Path
import hashlib
import json

import cv2
import numpy as np

from fdm.services.cu_automatic_measurement import analyze_frame, CuAutomaticCancelled


def horizontal(width: int = 8) -> np.ndarray:
    # Pixel-center rasterization matches the independently audited CU fixture.
    image = np.full((240, 320, 3), 220, dtype=np.uint8)
    image[120 - width // 2:120 + width // 2, 40:280] = 40
    return image


class CuAutomaticMeasurementTests(unittest.TestCase):
    def test_legacy_aa_clipping_and_contour_order_match_audited_stage(self):
        fixtures = Path(__file__).parent / "fixtures" / "cu_automatic"
        for name in ("help_qualitative", "help_semiauto"):
            with self.subTest(image=name):
                image = cv2.imread(str(fixtures / f"{name}.png"))
                contour = cv2.imread(str(fixtures / f"{name}_contour.png"), cv2.IMREAD_GRAYSCALE)
                self.assertIsNotNone(image)
                self.assertIsNotNone(contour)
                result = analyze_frame(image, _debug=True)
                np.testing.assert_array_equal(result["diagnostics"]["contour"], contour)

    def test_actual_buffer_accounting_allows_bounded_multiscale_work(self):
        # The former 104 bytes/pixel preflight rejected this ordinary frame
        # without running. Metered allocations must keep its real candidate.
        result = analyze_frame(horizontal(12), "multiscale", memory_budget=6 * 1024**2)
        self.assertEqual(result["status"], "ok")
        self.assertAlmostEqual(result["candidates"][0]["width_px"], 10.997208727965706, places=9)
        self.assertLess(result["diagnostics"]["native_peak_reserved_bytes"], 6 * 1024**2)

    def test_rotating_calipers_keep_original_strict_ratio_boundaries(self):
        fixtures = Path(__file__).parent / "fixtures" / "cu_automatic"
        manifest = json.loads((fixtures / "provenance.json").read_text())
        expected = next(case for case in manifest["cases"] if case["image"] == "stress_connected_01.png")
        result = analyze_frame(cv2.imread(str(fixtures / expected["image"])), "multiscale", _debug=True)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(len(result["candidates"]), 11)
        self.assertEqual(result["diagnostics"]["passes"], [
            {"count": item["count"], "valid": item["valid"]} for item in expected["pass_candidates"]
        ])
        # The third coarse candidate is later rejected, but its original L1
        # orientation must still be preserved before width measurement.
        self.assertEqual(result["diagnostics"]["pass_axes"][1], [True, False, True, True, True])
        for name, digest in expected["stage_sha256"].items():
            self.assertEqual(hashlib.sha256(result["diagnostics"][name].tobytes()).hexdigest(), digest, name)

    def test_real_traditional_golden_widths_and_order(self):
        expected = {8: 7.015789053996511, 12: 11.0, 20: 19.0, 32: 31.0, 48: 47.0}
        for width, golden in expected.items():
            with self.subTest(width=width):
                result = analyze_frame(horizontal(width), "traditional")
                self.assertEqual(result["status"], "ok")
                self.assertEqual(len(result["candidates"]), 1)
                self.assertAlmostEqual(result["candidates"][0]["width_px"], golden, places=10)
                self.assertEqual(len(result["candidates"][0]["path"]), len(result["candidates"][0]["samples"]))

    def test_real_multiscale_golden_and_original_rejection(self):
        for width, golden in [(8, 6.995971498103516), (12, 10.997208727965706)]:
            result = analyze_frame(horizontal(width), "multiscale")
            self.assertEqual(result["status"], "ok")
            self.assertAlmostEqual(result["candidates"][0]["width_px"], golden, places=9)
            self.assertTrue(result["candidates"][0]["segments"])
        result = analyze_frame(horizontal(20), "multiscale")
        self.assertEqual(result["status"], "empty")
        self.assertEqual(result["candidates"], [])

    def test_no_old_thirty_fiber_output_cap(self):
        image = np.full((645, 440, 3), 220, np.uint8)
        for row in range(35):
            y = 15 + 18 * row
            image[y - 4:y + 4, 45:395] = 40
        result = analyze_frame(image)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(len(result["candidates"]), 35)

    def test_strided_readonly_gray_and_bgra_inputs_keep_coordinates(self):
        base = horizontal(12)
        view = base[:, ::-1]
        before = base.copy()
        gray = view[:, :, 0]
        gray.setflags(write=False)
        a = analyze_frame(view)
        b = analyze_frame(gray)
        bgra = np.concatenate((view, np.full((*view.shape[:2], 1), 255, np.uint8)), axis=2)
        c = analyze_frame(bgra)
        self.assertEqual(a["candidates"], b["candidates"])
        self.assertEqual(a["candidates"], c["candidates"])
        np.testing.assert_array_equal(base, before)

    def test_budget_and_candidate_limit_do_not_return_partial_results(self):
        for kwargs in [{"memory_budget": 1024}, {"max_candidates": 2}]:
            result = analyze_frame(horizontal(), **kwargs)
            self.assertEqual(result["status"], "resource_limit")
            self.assertEqual(result["candidates"], [])

    def test_cancel_before_and_inside_native_then_reuse(self):
        with self.assertRaises(CuAutomaticCancelled):
            analyze_frame(horizontal(), checkpoint=lambda: True)
        count = 0

        class UserCancelled(Exception):
            pass

        def checkpoint():
            nonlocal count
            count += 1
            if count >= 8:
                raise UserCancelled("original exception")

        with self.assertRaisesRegex(UserCancelled, "original exception"):
            analyze_frame(horizontal(), "multiscale", checkpoint)
        self.assertEqual(analyze_frame(horizontal())["status"], "ok")

    def test_blank_and_invalid_inputs(self):
        for algorithm in ["traditional", "multiscale"]:
            result = analyze_frame(np.full((32, 32), 220, np.uint8), algorithm)
            self.assertEqual(result["status"], "empty")
        for bad in [np.zeros((16,), np.uint8), np.zeros((16, 16, 2), np.uint8), np.zeros((8, 8), np.uint8)]:
            with self.assertRaises(ValueError):
                analyze_frame(bad)
        with self.assertRaises(TypeError):
            analyze_frame(horizontal().astype(np.float32))
