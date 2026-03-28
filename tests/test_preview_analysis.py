from __future__ import annotations

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    import cv2
    import numpy as np
    from PySide6.QtGui import QImage

    from fdm.services.preview_analysis import (
        FocusStackAnalyzer,
        MapBuildAnalyzer,
        _focus_measure,
        bgr_array_to_qimage,
        qimage_to_bgr_array,
    )

    PREVIEW_ANALYSIS_READY = True
except ModuleNotFoundError:
    PREVIEW_ANALYSIS_READY = False
    cv2 = None
    np = None
    QImage = None
    FocusStackAnalyzer = None
    MapBuildAnalyzer = None
    _focus_measure = None
    bgr_array_to_qimage = None
    qimage_to_bgr_array = None


@unittest.skipUnless(PREVIEW_ANALYSIS_READY, "requires numpy, opencv-python and PySide6")
class PreviewAnalysisTests(unittest.TestCase):
    def _make_map_base(self) -> np.ndarray:
        base = np.full((220, 320, 3), 245, dtype=np.uint8)
        cv2.circle(base, (120, 110), 26, (30, 30, 30), -1, cv2.LINE_AA)
        cv2.rectangle(base, (170, 40), (250, 170), (70, 70, 70), -1)
        cv2.putText(base, "A1", (36, 76), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)
        return base

    def _shift_frame(self, image: np.ndarray, *, dx: int = 0, dy: int = 0, blur_sigma: float | None = None) -> QImage:
        shifted = np.roll(image, shift=dy, axis=0)
        shifted = np.roll(shifted, shift=dx, axis=1)
        if blur_sigma is not None:
            shifted = cv2.GaussianBlur(shifted, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)
        return bgr_array_to_qimage(shifted)

    def _make_focus_frames(self) -> tuple[QImage, QImage]:
        base = np.full((180, 260, 3), 255, dtype=np.uint8)
        cv2.putText(base, "FOCUS", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.line(base, (20, 140), (240, 35), (0, 0, 0), 3, cv2.LINE_AA)
        blurred = cv2.GaussianBlur(base, (0, 0), sigmaX=5.0, sigmaY=5.0)
        left_sharp = blurred.copy()
        left_sharp[:, :130] = base[:, :130]
        right_sharp = blurred.copy()
        right_sharp[:, 130:] = base[:, 130:]
        return bgr_array_to_qimage(left_sharp), bgr_array_to_qimage(right_sharp)

    def test_focus_stack_analyzer_combines_two_focus_planes(self) -> None:
        left_frame, right_frame = self._make_focus_frames()
        analyzer = FocusStackAnalyzer(device_id="microview:0", device_name="Microview #1")

        report_a = analyzer.add_frame(left_frame)
        report_b = analyzer.add_frame(right_frame)
        result = analyzer.finalize()

        self.assertFalse(report_a.preview_image.isNull())
        self.assertFalse(report_b.preview_image.isNull())
        self.assertFalse(result.image.isNull())
        self.assertEqual(result.accepted_frames, 2)

        fused_bgr = qimage_to_bgr_array(result.image)
        fused_score = float(_focus_measure(cv2.cvtColor(fused_bgr, cv2.COLOR_BGR2GRAY)).mean())
        left_score = float(_focus_measure(cv2.cvtColor(qimage_to_bgr_array(left_frame), cv2.COLOR_BGR2GRAY)).mean())
        right_score = float(_focus_measure(cv2.cvtColor(qimage_to_bgr_array(right_frame), cv2.COLOR_BGR2GRAY)).mean())
        self.assertGreaterEqual(fused_score, max(left_score, right_score) * 0.9)

    def test_focus_stack_finalize_can_apply_optional_post_sharpen(self) -> None:
        left_frame, right_frame = self._make_focus_frames()
        analyzer = FocusStackAnalyzer(device_id="qt_multimedia:test", device_name="USB Camera")
        analyzer.add_frame(left_frame)
        analyzer.add_frame(right_frame)

        plain = analyzer.finalize(post_sharpen=False)
        sharpened = analyzer.finalize(post_sharpen=True)

        plain_bgr = qimage_to_bgr_array(plain.image)
        sharpened_bgr = qimage_to_bgr_array(sharpened.image)
        mean_diff = float(np.mean(np.abs(sharpened_bgr.astype(np.int16) - plain_bgr.astype(np.int16))))

        self.assertFalse(sharpened.image.isNull())
        self.assertTrue(sharpened.metadata.get("post_sharpen"))
        self.assertFalse(plain.metadata.get("post_sharpen"))
        self.assertGreater(mean_diff, 0.05)

    def test_map_build_analyzer_creates_two_tile_mosaic(self) -> None:
        base = self._make_map_base()
        frame_a = bgr_array_to_qimage(base)
        frame_b = self._shift_frame(base, dx=-120)

        analyzer = MapBuildAnalyzer(device_id="microview:0", device_name="Microview #1")
        report_a = analyzer.add_frame(frame_a)
        analyzer.add_frame(frame_a)
        analyzer.add_frame(frame_a)
        analyzer.add_frame(frame_b)
        analyzer.add_frame(frame_b)
        report_b = analyzer.add_frame(frame_b)
        result = analyzer.finalize()

        self.assertTrue(report_a.preview_image.isNull() or report_a.motion_state in {"settling", "moving"})
        self.assertFalse(report_b.preview_image.isNull())
        self.assertFalse(result.image.isNull())
        self.assertEqual(result.tile_count, 2)
        self.assertGreater(result.image.width(), frame_a.width())

    def test_map_build_waits_for_three_stable_frames_before_sampling(self) -> None:
        base = self._make_map_base()
        frame = bgr_array_to_qimage(base)
        analyzer = MapBuildAnalyzer(device_id="microview:0", device_name="Microview #1")

        report_a = analyzer.add_frame(frame)
        report_b = analyzer.add_frame(frame)
        report_c = analyzer.add_frame(frame)

        self.assertEqual(report_a.motion_state, "settling")
        self.assertEqual(report_b.motion_state, "settling")
        self.assertEqual(report_c.motion_state, "stable")
        self.assertEqual(report_c.accepted_frames, 1)
        self.assertIn("已静止", report_c.message)

    def test_map_build_rejects_moving_blurred_frames_until_new_tile_is_stable(self) -> None:
        base = self._make_map_base()
        stable = bgr_array_to_qimage(base)
        moving = self._shift_frame(base, dx=-100, blur_sigma=4.0)
        shifted = self._shift_frame(base, dx=-120)
        analyzer = MapBuildAnalyzer(device_id="microview:0", device_name="Microview #1")

        analyzer.add_frame(stable)
        analyzer.add_frame(stable)
        analyzer.add_frame(stable)
        steady_report = analyzer.add_frame(stable)
        moving_report = analyzer.add_frame(moving)
        settling_a = analyzer.add_frame(shifted)
        settling_b = analyzer.add_frame(shifted)
        stable_report = analyzer.add_frame(shifted)
        result = analyzer.finalize()

        self.assertEqual(steady_report.accepted_frames, 2)
        self.assertEqual(moving_report.motion_state, "transition_pending")
        self.assertEqual(moving_report.accepted_frames, 2)
        self.assertIn("等待稳定", moving_report.message)
        self.assertEqual(settling_a.motion_state, "transition_pending")
        self.assertEqual(settling_b.motion_state, "transition_pending")
        self.assertEqual(stable_report.motion_state, "stable")
        self.assertEqual(stable_report.accepted_frames, 3)
        self.assertEqual(result.tile_count, 2)

    def test_map_build_focus_only_changes_are_allowed_inside_same_tile(self) -> None:
        base = self._make_map_base()
        frame = bgr_array_to_qimage(base)
        softened = self._shift_frame(base, blur_sigma=2.5)
        analyzer = MapBuildAnalyzer(device_id="microview:0", device_name="Microview #1")

        analyzer.add_frame(frame)
        analyzer.add_frame(frame)
        start_report = analyzer.add_frame(frame)
        focus_report = analyzer.add_frame(softened)
        result = analyzer.finalize()

        self.assertEqual(start_report.accepted_frames, 1)
        self.assertEqual(focus_report.motion_state, "stable")
        self.assertEqual(focus_report.tile_count, 1)
        self.assertGreaterEqual(focus_report.accepted_frames, 2)
        self.assertEqual(result.tile_count, 1)

    def test_map_build_returns_to_same_position_without_opening_new_tile(self) -> None:
        base = self._make_map_base()
        frame = bgr_array_to_qimage(base)
        wobble = self._shift_frame(base, dx=-20, blur_sigma=3.0)
        analyzer = MapBuildAnalyzer(device_id="microview:0", device_name="Microview #1")

        analyzer.add_frame(frame)
        analyzer.add_frame(frame)
        analyzer.add_frame(frame)
        analyzer.add_frame(frame)
        analyzer.add_frame(wobble)
        analyzer.add_frame(frame)
        analyzer.add_frame(frame)
        resume_report = analyzer.add_frame(frame)
        result = analyzer.finalize()

        self.assertEqual(resume_report.motion_state, "stable")
        self.assertEqual(result.tile_count, 1)

    def test_map_build_rejects_far_stable_position_when_overlap_is_too_small(self) -> None:
        base = self._make_map_base()
        frame = bgr_array_to_qimage(base)
        far = self._shift_frame(base, dx=-300)
        analyzer = MapBuildAnalyzer(device_id="microview:0", device_name="Microview #1")

        analyzer.add_frame(frame)
        analyzer.add_frame(frame)
        analyzer.add_frame(frame)
        analyzer.add_frame(frame)
        analyzer.add_frame(far)
        analyzer.add_frame(far)
        low_conf_report = analyzer.add_frame(far)
        result = analyzer.finalize()

        self.assertTrue(low_conf_report.low_confidence or "未创建新 tile" in low_conf_report.message)
        self.assertIn("未创建新 tile", low_conf_report.message)
        self.assertEqual(result.tile_count, 1)

    def test_map_build_allows_small_but_stable_translation_to_create_new_tile(self) -> None:
        base = self._make_map_base()
        frame = bgr_array_to_qimage(base)
        nearby = self._shift_frame(base, dx=-26)
        analyzer = MapBuildAnalyzer(device_id="microview:0", device_name="Microview #1")

        analyzer.add_frame(frame)
        analyzer.add_frame(frame)
        analyzer.add_frame(frame)
        analyzer.add_frame(frame)
        analyzer.add_frame(nearby)
        analyzer.add_frame(nearby)
        stable_report = analyzer.add_frame(nearby)
        result = analyzer.finalize()

        self.assertEqual(stable_report.motion_state, "stable")
        self.assertGreaterEqual(stable_report.accepted_frames, 3)
        self.assertEqual(result.tile_count, 2)
