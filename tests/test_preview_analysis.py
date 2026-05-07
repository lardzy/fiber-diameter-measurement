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
        FocusStackRenderConfig,
        _focus_measure,
        bgr_array_to_qimage,
        qimage_to_bgr_array,
    )
    from fdm.settings import FocusStackProfile

    PREVIEW_ANALYSIS_READY = True
except ModuleNotFoundError:
    PREVIEW_ANALYSIS_READY = False
    cv2 = None
    np = None
    QImage = None
    FocusStackAnalyzer = None
    MapBuildAnalyzer = None
    FocusStackRenderConfig = None
    FocusStackProfile = None
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

    def _make_map_scene(self) -> np.ndarray:
        scene = np.full((520, 860, 3), 238, dtype=np.uint8)
        for index, (x, y) in enumerate(
            (
                (70, 95),
                (160, 280),
                (250, 140),
                (360, 350),
                (470, 210),
                (590, 105),
                (700, 330),
                (795, 185),
            )
        ):
            color = (35 + index * 19, 55 + index * 13, 75 + index * 11)
            cv2.circle(scene, (x, y), 18 + index % 5, color, -1, cv2.LINE_AA)
            cv2.rectangle(scene, (x - 26, y + 34), (x + 48, y + 58), (30, 30, 30), 2, cv2.LINE_AA)
        cv2.putText(scene, "A1", (95, 190), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(scene, "B7", (410, 305), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (15, 15, 15), 3, cv2.LINE_AA)
        cv2.putText(scene, "C3", (660, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (20, 20, 20), 3, cv2.LINE_AA)
        cv2.line(scene, (50, 440), (810, 65), (80, 80, 80), 3, cv2.LINE_AA)
        cv2.line(scene, (130, 70), (780, 430), (120, 120, 120), 2, cv2.LINE_AA)
        return scene

    def _crop_map_frame(
        self,
        scene: np.ndarray,
        *,
        x: int,
        y: int = 130,
        width: int = 320,
        height: int = 220,
        blur_sigma: float | None = None,
    ) -> QImage:
        crop = scene[y : y + height, x : x + width].copy()
        if blur_sigma is not None:
            crop = cv2.GaussianBlur(crop, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)
        return bgr_array_to_qimage(crop)

    def _feed_stable_position(self, analyzer: MapBuildAnalyzer, frame: QImage, *, count: int = 4):
        report = None
        for _ in range(count):
            report = analyzer.add_frame(frame)
        self.assertIsNotNone(report)
        return report

    def _make_repetitive_scene(self) -> np.ndarray:
        scene = np.full((360, 760, 3), 235, dtype=np.uint8)
        for x in range(0, scene.shape[1], 32):
            shade = 60 if (x // 32) % 2 == 0 else 180
            cv2.rectangle(scene, (x, 0), (x + 15, scene.shape[0]), (shade, shade, shade), -1)
        for y in range(30, scene.shape[0], 70):
            cv2.line(scene, (0, y), (scene.shape[1], y), (100, 100, 100), 1, cv2.LINE_AA)
        return scene

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
        analyzer = FocusStackAnalyzer(
            device_id="microview:0",
            device_name="Microview #1",
            render_config=FocusStackRenderConfig(
                profile=FocusStackProfile.BALANCED,
                sharpen_strength=35,
            ),
        )

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

    def test_focus_stack_preview_matches_final_when_using_same_render_config(self) -> None:
        left_frame, right_frame = self._make_focus_frames()
        analyzer = FocusStackAnalyzer(
            device_id="qt_multimedia:test",
            device_name="USB Camera",
            render_config=FocusStackRenderConfig(
                profile=FocusStackProfile.BALANCED,
                sharpen_strength=35,
            ),
        )
        analyzer.add_frame(left_frame)
        preview_report = analyzer.add_frame(right_frame)
        result = analyzer.finalize()

        preview_bgr = qimage_to_bgr_array(preview_report.preview_image)
        final_bgr = qimage_to_bgr_array(result.image)
        mean_diff = float(np.mean(np.abs(final_bgr.astype(np.int16) - preview_bgr.astype(np.int16))))

        self.assertFalse(result.image.isNull())
        self.assertLess(mean_diff, 1.0)

    def test_focus_stack_profiles_follow_expected_sharpness_order(self) -> None:
        left_frame, right_frame = self._make_focus_frames()
        scores: dict[str, float] = {}
        for profile in (
            FocusStackProfile.SHARP,
            FocusStackProfile.BALANCED,
            FocusStackProfile.SOFT,
        ):
            analyzer = FocusStackAnalyzer(
                device_id=f"profile:{profile}",
                device_name="USB Camera",
                render_config=FocusStackRenderConfig(profile=profile, sharpen_strength=0),
            )
            analyzer.add_frame(left_frame)
            analyzer.add_frame(right_frame)
            fused = qimage_to_bgr_array(analyzer.finalize().image)
            scores[profile] = float(_focus_measure(cv2.cvtColor(fused, cv2.COLOR_BGR2GRAY)).mean())

        self.assertGreater(scores[FocusStackProfile.SHARP], scores[FocusStackProfile.BALANCED])
        self.assertGreater(scores[FocusStackProfile.BALANCED], scores[FocusStackProfile.SOFT])

    def test_focus_stack_sharpen_strength_affects_preview_and_final_output(self) -> None:
        left_frame, right_frame = self._make_focus_frames()
        analyzer = FocusStackAnalyzer(
            device_id="qt_multimedia:test",
            device_name="USB Camera",
            render_config=FocusStackRenderConfig(
                profile=FocusStackProfile.BALANCED,
                sharpen_strength=0,
            ),
        )
        analyzer.add_frame(left_frame)
        preview_plain = analyzer.add_frame(right_frame)
        plain = analyzer.finalize()

        analyzer.set_render_config(
            FocusStackRenderConfig(
                profile=FocusStackProfile.BALANCED,
                sharpen_strength=85,
            )
        )
        preview_sharp = analyzer.refresh_preview()
        sharp = analyzer.finalize()

        preview_plain_bgr = qimage_to_bgr_array(preview_plain.preview_image)
        preview_sharp_bgr = qimage_to_bgr_array(preview_sharp.preview_image)
        plain_bgr = qimage_to_bgr_array(plain.image)
        sharp_bgr = qimage_to_bgr_array(sharp.image)

        preview_diff = float(np.mean(np.abs(preview_sharp_bgr.astype(np.int16) - preview_plain_bgr.astype(np.int16))))
        final_diff = float(np.mean(np.abs(sharp_bgr.astype(np.int16) - plain_bgr.astype(np.int16))))

        self.assertGreater(preview_diff, 0.05)
        self.assertGreater(final_diff, 0.05)
        self.assertEqual(sharp.metadata.get("focus_stack_profile"), FocusStackProfile.BALANCED)
        self.assertEqual(sharp.metadata.get("sharpen_strength"), 85)

    def test_focus_stack_refresh_preview_after_config_change_keeps_accepted_frames(self) -> None:
        left_frame, right_frame = self._make_focus_frames()
        analyzer = FocusStackAnalyzer(device_id="microview:0", device_name="Microview #1")
        analyzer.add_frame(left_frame)
        analyzer.add_frame(right_frame)

        analyzer.set_render_config(
            FocusStackRenderConfig(
                profile=FocusStackProfile.SHARP,
                sharpen_strength=60,
            )
        )
        refreshed = analyzer.refresh_preview()

        self.assertEqual(refreshed.accepted_frames, 2)
        self.assertFalse(refreshed.preview_image.isNull())
        self.assertIn("预览参数已更新", refreshed.message)

    def test_map_build_analyzer_creates_reliable_mosaics_from_real_crops(self) -> None:
        scene = self._make_map_scene()
        for shift in (160, 208, 256):
            with self.subTest(shift=shift):
                frame_a = self._crop_map_frame(scene, x=80)
                frame_b = self._crop_map_frame(scene, x=80 + shift)
                analyzer = MapBuildAnalyzer(device_id="microview:0", device_name="Microview #1")

                report_a = self._feed_stable_position(analyzer, frame_a, count=3)
                report_b = self._feed_stable_position(analyzer, frame_b, count=4)
                result = analyzer.finalize()

                self.assertEqual(report_a.motion_state, "sampling")
                self.assertEqual(report_b.motion_state, "tile_committed")
                self.assertFalse(result.image.isNull())
                self.assertEqual(result.tile_count, 2)
                self.assertGreater(result.image.width(), frame_a.width())
                self.assertEqual(result.metadata.get("edge_count"), 1)
                self.assertIn("registration_thresholds", result.metadata)
                self.assertLess(abs(analyzer._tiles[1].x - shift), 8.0)  # noqa: SLF001
                self.assertLess(abs(analyzer._tiles[1].y), 6.0)  # noqa: SLF001

    def test_map_build_waits_for_three_stable_frames_before_sampling(self) -> None:
        base = self._make_map_base()
        frame = bgr_array_to_qimage(base)
        analyzer = MapBuildAnalyzer(device_id="microview:0", device_name="Microview #1")

        report_a = analyzer.add_frame(frame)
        report_b = analyzer.add_frame(frame)
        report_c = analyzer.add_frame(frame)

        self.assertEqual(report_a.motion_state, "settling")
        self.assertEqual(report_b.motion_state, "settling")
        self.assertEqual(report_c.motion_state, "sampling")
        self.assertEqual(report_c.accepted_frames, 1)
        self.assertIn("已静止", report_c.message)

    def test_map_build_rejects_moving_blurred_frames_until_new_tile_is_stable(self) -> None:
        scene = self._make_map_scene()
        stable = self._crop_map_frame(scene, x=80)
        moving = self._crop_map_frame(scene, x=245, blur_sigma=4.0)
        shifted = self._crop_map_frame(scene, x=288)
        analyzer = MapBuildAnalyzer(device_id="microview:0", device_name="Microview #1")

        steady_report = self._feed_stable_position(analyzer, stable, count=3)
        moving_report = analyzer.add_frame(moving)
        settling_a = analyzer.add_frame(shifted)
        settling_b = analyzer.add_frame(shifted)
        settling_c = analyzer.add_frame(shifted)
        stable_report = analyzer.add_frame(shifted)
        result = analyzer.finalize()

        self.assertEqual(steady_report.accepted_frames, 1)
        self.assertEqual(moving_report.motion_state, "moving")
        self.assertIn("等待静止", moving_report.message)
        self.assertIn(settling_a.motion_state, {"moving", "settling"})
        self.assertEqual(settling_b.motion_state, "settling")
        self.assertEqual(settling_c.motion_state, "settling")
        self.assertEqual(stable_report.motion_state, "tile_committed")
        self.assertEqual(result.tile_count, 2)

    def test_map_build_focus_only_changes_are_allowed_inside_same_tile(self) -> None:
        scene = self._make_map_scene()
        frame = self._crop_map_frame(scene, x=80)
        softened = self._crop_map_frame(scene, x=80, blur_sigma=2.5)
        analyzer = MapBuildAnalyzer(device_id="microview:0", device_name="Microview #1")

        analyzer.add_frame(frame)
        analyzer.add_frame(frame)
        start_report = analyzer.add_frame(frame)
        focus_report = analyzer.add_frame(softened)

        self.assertEqual(start_report.accepted_frames, 1)
        self.assertEqual(focus_report.motion_state, "sampling")
        self.assertEqual(focus_report.tile_count, 1)
        self.assertGreaterEqual(focus_report.accepted_frames, 2)
        with self.assertRaisesRegex(RuntimeError, "至少需要两个可靠 tile"):
            analyzer.finalize()

    def test_map_build_returns_to_same_position_without_opening_new_tile(self) -> None:
        scene = self._make_map_scene()
        frame = self._crop_map_frame(scene, x=80)
        wobble = self._crop_map_frame(scene, x=104, blur_sigma=3.0)
        analyzer = MapBuildAnalyzer(device_id="microview:0", device_name="Microview #1")

        self._feed_stable_position(analyzer, frame, count=3)
        analyzer.add_frame(wobble)
        analyzer.add_frame(frame)
        analyzer.add_frame(frame)
        analyzer.add_frame(frame)
        resume_report = analyzer.add_frame(frame)

        self.assertEqual(resume_report.motion_state, "sampling")
        self.assertEqual(resume_report.tile_count, 1)
        with self.assertRaisesRegex(RuntimeError, "至少需要两个可靠 tile"):
            analyzer.finalize()

    def test_map_build_rejects_far_stable_position_when_overlap_is_too_small(self) -> None:
        scene = self._make_map_scene()
        frame = self._crop_map_frame(scene, x=80)
        far = self._crop_map_frame(scene, x=380)
        analyzer = MapBuildAnalyzer(device_id="microview:0", device_name="Microview #1")

        self._feed_stable_position(analyzer, frame, count=3)
        low_conf_report = self._feed_stable_position(analyzer, far, count=4)

        self.assertEqual(low_conf_report.motion_state, "candidate_rejected")
        self.assertTrue(low_conf_report.low_confidence)
        self.assertIn("未创建新 tile", low_conf_report.message)
        with self.assertRaisesRegex(RuntimeError, "至少需要两个可靠 tile"):
            analyzer.finalize()

    def test_map_build_rejects_repetitive_texture_instead_of_guessing_tile_position(self) -> None:
        scene = self._make_repetitive_scene()
        frame = self._crop_map_frame(scene, x=48, y=70)
        shifted = self._crop_map_frame(scene, x=256, y=70)
        analyzer = MapBuildAnalyzer(device_id="microview:0", device_name="Microview #1")

        self._feed_stable_position(analyzer, frame, count=3)
        report = self._feed_stable_position(analyzer, shifted, count=4)

        self.assertEqual(report.motion_state, "candidate_rejected")
        self.assertTrue(report.low_confidence)
        with self.assertRaisesRegex(RuntimeError, "至少需要两个可靠 tile"):
            analyzer.finalize()
