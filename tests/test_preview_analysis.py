from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    import cv2
    import numpy as np
    from PySide6.QtGui import QImage

    from fdm.services.preview_analysis import (
        AnalysisResourceLimits,
        FocusAccumulator,
        FocusStackAnalyzer,
        MapBuildAnalyzer,
        FocusStackRenderConfig,
        MAP_BUILD_MAX_TILE_FRAMES,
        MAP_BUILD_STABLE_REQUIRED_FRAMES,
        _focus_measure,
        bgr_array_to_qimage,
        qimage_to_bgr_array,
    )
    import fdm.services.preview_analysis as preview_analysis
    from fdm.settings import FocusStackProfile

    PREVIEW_ANALYSIS_READY = True
except ModuleNotFoundError:
    PREVIEW_ANALYSIS_READY = False
    cv2 = None
    np = None
    QImage = None
    FocusStackAnalyzer = None
    FocusAccumulator = None
    AnalysisResourceLimits = None
    MapBuildAnalyzer = None
    FocusStackRenderConfig = None
    FocusStackProfile = None
    MAP_BUILD_MAX_TILE_FRAMES = 0
    MAP_BUILD_STABLE_REQUIRED_FRAMES = 0
    _focus_measure = None
    bgr_array_to_qimage = None
    qimage_to_bgr_array = None
    preview_analysis = None


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

    def test_focus_accumulator_stops_at_configured_frame_limit(self) -> None:
        left_frame, right_frame = self._make_focus_frames()
        accumulator = FocusAccumulator(
            limits=AnalysisResourceLimits(
                focus_max_frames=1,
                focus_max_retained_bytes=256 * 1024 * 1024,
            )
        )

        self.assertTrue(accumulator.add_qimage(left_frame))
        self.assertFalse(accumulator.add_qimage(right_frame))
        self.assertTrue(accumulator.limit_reached)
        self.assertEqual(accumulator.accepted_frames, 1)
        self.assertIn("1 张上限", accumulator.limit_reason)

    def test_focus_stack_report_exposes_resource_limit(self) -> None:
        left_frame, right_frame = self._make_focus_frames()
        analyzer = FocusStackAnalyzer(
            device_id="test",
            device_name="test",
            resource_limits=AnalysisResourceLimits(focus_max_frames=1),
        )

        analyzer.add_frame(left_frame)
        report = analyzer.add_frame(right_frame)

        self.assertTrue(report.limit_reached)
        self.assertGreater(report.retained_bytes, 0)
        self.assertIn("上限", report.message)

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

    def test_incremental_focus_stack_stays_within_legacy_render_tolerance(self) -> None:
        left_frame, right_frame = self._make_focus_frames()
        prepared = [
            preview_analysis._prepare_frame(left_frame),  # noqa: SLF001
            preview_analysis._prepare_frame(right_frame),  # noqa: SLF001
        ]
        accumulator = FocusAccumulator()
        for frame in prepared:
            self.assertTrue(accumulator.add_prepared_frame(frame))

        for profile in (
            FocusStackProfile.SHARP,
            FocusStackProfile.BALANCED,
            FocusStackProfile.SOFT,
        ):
            with self.subTest(profile=profile):
                config = FocusStackRenderConfig(profile=profile, sharpen_strength=0)
                legacy = preview_analysis._focus_stack_render(  # noqa: SLF001
                    [frame.bgr for frame in prepared],
                    [frame.focus_map for frame in prepared],
                    config,
                )
                incremental = qimage_to_bgr_array(accumulator.final_image(config))
                difference = np.abs(legacy.astype(np.int16) - incremental.astype(np.int16))

                self.assertLess(float(difference.mean()), 4.0)
                self.assertLessEqual(float(np.percentile(difference, 95)), 20.0)

    def test_incremental_focus_stack_retained_memory_does_not_grow_with_frame_count(self) -> None:
        rng = np.random.default_rng(20260710)
        base = rng.integers(0, 256, size=(180, 260, 3), dtype=np.uint8)
        blurred = cv2.GaussianBlur(base, (0, 0), sigmaX=4.0, sigmaY=4.0)
        frames: list[QImage] = []
        segment_width = base.shape[1] // 6
        for index in range(6):
            frame = blurred.copy()
            start = index * segment_width
            end = base.shape[1] if index == 5 else (index + 1) * segment_width
            frame[:, start:end] = base[:, start:end]
            frames.append(bgr_array_to_qimage(frame))

        accumulator = FocusAccumulator()
        retained_sizes: list[int] = []
        for frame in frames:
            self.assertTrue(accumulator.add_qimage(frame))
            retained_sizes.append(accumulator.retained_bytes)

        self.assertEqual(accumulator.accepted_frames, 6)
        self.assertEqual(len(set(retained_sizes)), 1)
        prepared = [preview_analysis._prepare_frame(frame) for frame in frames]  # noqa: SLF001
        soft_config = FocusStackRenderConfig(profile=FocusStackProfile.SOFT, sharpen_strength=0)
        legacy_soft = preview_analysis._focus_stack_render(  # noqa: SLF001
            [frame.bgr for frame in prepared],
            [frame.focus_map for frame in prepared],
            soft_config,
        )
        incremental_soft = qimage_to_bgr_array(accumulator.final_image(soft_config))
        soft_difference = np.abs(legacy_soft.astype(np.int16) - incremental_soft.astype(np.int16))
        self.assertLess(float(soft_difference.mean()), 5.0)
        self.assertLessEqual(float(np.percentile(soft_difference, 95)), 20.0)
        with patch.object(np, "stack", side_effect=AssertionError("增量渲染不应堆叠历史帧")):
            rendered = accumulator.final_image(
                FocusStackRenderConfig(profile=FocusStackProfile.BALANCED, sharpen_strength=0)
            )
        self.assertFalse(rendered.isNull())

    def test_incremental_focus_stack_matches_legacy_on_high_contrast_complementary_planes(self) -> None:
        rng = np.random.default_rng(20260711)
        base = rng.integers(0, 256, size=(220, 320, 3), dtype=np.uint8)
        base[:, ::8] = 255 - base[:, ::8]
        blurred = cv2.GaussianBlur(base, (0, 0), sigmaX=8.0, sigmaY=8.0)
        left = blurred.copy()
        left[:, :160] = base[:, :160]
        right = blurred.copy()
        right[:, 160:] = base[:, 160:]
        prepared = [
            preview_analysis._prepare_frame(bgr_array_to_qimage(left)),  # noqa: SLF001
            preview_analysis._prepare_frame(bgr_array_to_qimage(right)),  # noqa: SLF001
        ]
        accumulator = FocusAccumulator()
        for frame in prepared:
            self.assertTrue(accumulator.add_prepared_frame(frame))

        for profile in (FocusStackProfile.SHARP, FocusStackProfile.BALANCED, FocusStackProfile.SOFT):
            with self.subTest(profile=profile):
                config = FocusStackRenderConfig(profile=profile, sharpen_strength=0)
                legacy = preview_analysis._focus_stack_render(  # noqa: SLF001
                    [frame.bgr for frame in prepared],
                    [frame.focus_map for frame in prepared],
                    config,
                )
                incremental = qimage_to_bgr_array(accumulator.final_image(config))
                difference = np.abs(legacy.astype(np.int16) - incremental.astype(np.int16))
                self.assertLess(float(difference.mean()), 4.0)
                self.assertLessEqual(float(np.percentile(difference, 95)), 20.0)

    def test_map_promotes_focus_accumulator_limit_to_map_report(self) -> None:
        left, right = self._make_focus_frames()
        analyzer = MapBuildAnalyzer(
            device_id="limit",
            device_name="limit",
            resource_limits=AnalysisResourceLimits(focus_max_frames=1),
        )
        first = preview_analysis._prepare_frame(left)  # noqa: SLF001
        second = preview_analysis._prepare_frame(right)  # noqa: SLF001

        self.assertEqual(analyzer._accept_prepared_frame(first), "accepted")  # noqa: SLF001
        self.assertEqual(analyzer._accept_prepared_frame(second), "limit")  # noqa: SLF001
        report = analyzer._build_report()  # noqa: SLF001

        self.assertTrue(report.limit_reached)
        self.assertEqual(report.motion_state, "limit_reached")
        self.assertIn("1 张上限", report.limit_reason)

    def test_candidate_fusion_inherits_configured_resource_limits(self) -> None:
        left, _right = self._make_focus_frames()
        prepared = preview_analysis._prepare_frame(left)  # noqa: SLF001
        render_config = FocusStackRenderConfig(sharpen_strength=0)
        estimated_high_resolution_bytes = 300 * preview_analysis.MIB

        with patch.object(
            FocusAccumulator,
            "estimated_retained_bytes_for",
            return_value=estimated_high_resolution_bytes,
        ):
            default_result = preview_analysis._fuse_prepared_frames(  # noqa: SLF001
                [prepared],
                render_config,
            )
            configured_result = preview_analysis._fuse_prepared_frames(  # noqa: SLF001
                [prepared],
                render_config,
                limits=AnalysisResourceLimits(
                    focus_max_retained_bytes=512 * preview_analysis.MIB,
                ),
            )

        self.assertTrue(default_result.limit_reached)
        self.assertIn("256 MiB", default_result.limit_reason)
        self.assertFalse(configured_result.limit_reached)
        self.assertIsNotNone(configured_result.bgr)

    def test_candidate_fusion_budget_rejection_is_reported_as_map_limit(self) -> None:
        left, _right = self._make_focus_frames()
        prepared = preview_analysis._prepare_frame(left)  # noqa: SLF001
        candidate = preview_analysis._prepare_map_motion_frame(left)  # noqa: SLF001
        analyzer = MapBuildAnalyzer(
            device_id="limit",
            device_name="limit",
            resource_limits=AnalysisResourceLimits(
                focus_max_retained_bytes=384 * preview_analysis.MIB,
            ),
        )
        reference = preview_analysis._TileRecord(  # noqa: SLF001
            tile_id=-1,
            bgr=prepared.bgr,
            gray=prepared.gray,
            x=0.0,
            y=0.0,
        )
        fusion_limit = preview_analysis._FrameFusionResult(  # noqa: SLF001
            None,
            limit_reached=True,
            limit_reason="景深保留数据已达到 384 MiB 上限",
        )

        with patch.object(
            analyzer,
            "_current_tile_preview_record",
            return_value=reference,
        ), patch.object(
            analyzer,
            "_promote_motion_frame",
            return_value=prepared,
        ), patch.object(
            preview_analysis,
            "_fuse_prepared_frames",
            return_value=fusion_limit,
        ) as fuse_frames:
            analyzer._try_commit_candidate_tile(  # noqa: SLF001
                [candidate],
                coarse_dx=20.0,
                coarse_dy=0.0,
            )

        report = analyzer._build_report()  # noqa: SLF001
        self.assertTrue(report.limit_reached)
        self.assertEqual(report.motion_state, "limit_reached")
        self.assertIn("地图候选 tile", report.limit_reason)
        self.assertIn("384 MiB", report.message)
        self.assertNotIn("图像为空", report.message)
        self.assertEqual(analyzer._rejected_registration_frames, 0)  # noqa: SLF001
        self.assertEqual(
            fuse_frames.call_args.kwargs["limits"],
            analyzer._resource_limits,  # noqa: SLF001
        )

    def test_map_build_analyzer_creates_reliable_mosaics_from_real_crops(self) -> None:
        scene = self._make_map_scene()
        for shift in (160, 208, 256):
            with self.subTest(shift=shift):
                frame_a = self._crop_map_frame(scene, x=80)
                frame_b = self._crop_map_frame(scene, x=80 + shift)
                analyzer = MapBuildAnalyzer(device_id="microview:0", device_name="Microview #1")

                report_a = self._feed_stable_position(analyzer, frame_a, count=2)
                report_b = self._feed_stable_position(analyzer, frame_b, count=3)
                result = analyzer.finalize()

                self.assertEqual(report_a.motion_state, "sampling")
                self.assertEqual(report_b.motion_state, "tile_committed")
                self.assertFalse(result.image.isNull())
                self.assertEqual(result.tile_count, 2)
                self.assertGreater(result.image.width(), frame_a.width())
                self.assertEqual(result.metadata.get("edge_count"), 1)
                self.assertIn("registration_thresholds", result.metadata)
                self.assertEqual(result.metadata.get("map_build_interval_ms"), 90)
                self.assertEqual(result.metadata.get("stable_required_frames"), MAP_BUILD_STABLE_REQUIRED_FRAMES)
                self.assertEqual(result.metadata.get("max_tile_frames"), MAP_BUILD_MAX_TILE_FRAMES)
                self.assertIn("preview_refresh_interval_ms", result.metadata)
                self.assertIn("preview_render_count", result.metadata)
                self.assertIn("skipped_tile_frames", result.metadata)
                self.assertLess(abs(analyzer._tiles[1].x - shift), 8.0)  # noqa: SLF001
                self.assertLess(abs(analyzer._tiles[1].y), 6.0)  # noqa: SLF001

    def test_mosaic_strips_match_full_frame_reference_and_bound_float_allocations(self) -> None:
        first = self._make_map_base()
        second = np.roll(first, shift=35, axis=1)
        tiles = [
            preview_analysis._TileRecord(  # noqa: SLF001
                tile_id=0,
                bgr=first,
                gray=cv2.cvtColor(first, cv2.COLOR_BGR2GRAY),
                x=0.0,
                y=0.0,
            ),
            preview_analysis._TileRecord(  # noqa: SLF001
                tile_id=1,
                bgr=second,
                gray=cv2.cvtColor(second, cv2.COLOR_BGR2GRAY),
                x=140.0,
                y=45.0,
            ),
        ]
        width, height = preview_analysis._mosaic_dimensions(tiles)  # noqa: SLF001
        min_x = min(tile.x for tile in tiles)
        min_y = min(tile.y for tile in tiles)
        legacy_canvas = np.zeros((height, width, 3), dtype=np.float32)
        legacy_weights = np.zeros((height, width, 1), dtype=np.float32)
        for tile in tiles:
            x = int(round(tile.x - min_x))
            y = int(round(tile.y - min_y))
            y2 = min(height, y + tile.height)
            x2 = min(width, x + tile.width)
            crop = tile.bgr[: y2 - y, : x2 - x].astype(np.float32, copy=False)
            mask = preview_analysis._feather_mask(crop.shape[1], crop.shape[0])  # noqa: SLF001
            legacy_canvas[y:y2, x:x2] += crop * mask
            legacy_weights[y:y2, x:x2] += mask
        legacy = np.clip(legacy_canvas / np.clip(legacy_weights, 1e-6, None), 0, 255).astype(np.uint8)

        allocations: list[tuple[tuple[int, ...], np.dtype]] = []
        original_zeros = np.zeros

        def tracked_zeros(shape, *args, **kwargs):
            array = original_zeros(shape, *args, **kwargs)
            allocations.append((tuple(int(value) for value in shape), array.dtype))
            return array

        with patch.object(np, "zeros", side_effect=tracked_zeros):
            striped = preview_analysis._render_mosaic(tiles, strip_height=64)  # noqa: SLF001

        self.assertTrue(np.array_equal(striped, legacy))
        float_allocations = [shape for shape, dtype in allocations if dtype == np.dtype(np.float32)]
        self.assertTrue(float_allocations)
        self.assertLessEqual(max(shape[0] for shape in float_allocations), 64)
        self.assertNotIn((height, width, 3), float_allocations)
        self.assertNotIn((height, width, 1), float_allocations)

    def test_mosaic_working_set_estimate_scales_with_strip_height_not_full_height(self) -> None:
        width = 4096
        height = 4096
        estimate = preview_analysis._estimate_mosaic_render_working_bytes(  # noqa: SLF001
            width,
            height,
            strip_height=64,
        )
        old_full_frame_working_set = width * height * 19

        self.assertLess(estimate, old_full_frame_working_set // 3)
        self.assertEqual(
            estimate,
            (width * height * 3) + (width * 64 * 32),
        )

    def test_map_build_accepts_high_overlap_small_stage_moves(self) -> None:
        scene = self._make_map_scene()
        for shift in (24, 48, 64):
            with self.subTest(shift=shift):
                frame_a = self._crop_map_frame(scene, x=80)
                frame_b = self._crop_map_frame(scene, x=80 + shift)
                analyzer = MapBuildAnalyzer(device_id="microview:0", device_name="Microview #1")

                self._feed_stable_position(analyzer, frame_a, count=2)
                report_b = self._feed_stable_position(analyzer, frame_b, count=3)
                result = analyzer.finalize()

                self.assertEqual(report_b.motion_state, "tile_committed")
                self.assertFalse(result.image.isNull())
                self.assertEqual(result.tile_count, 2)
                self.assertLess(abs(analyzer._tiles[1].x - shift), 8.0)  # noqa: SLF001
                self.assertLess(abs(analyzer._tiles[1].y), 6.0)  # noqa: SLF001

    def test_map_build_caps_tile_fusion_frames_but_keeps_monitoring_motion(self) -> None:
        scene = self._make_map_scene()
        analyzer = MapBuildAnalyzer(device_id="microview:0", device_name="Microview #1")
        first_position = [
            self._crop_map_frame(scene, x=80),
            self._crop_map_frame(scene, x=80),
            self._crop_map_frame(scene, x=80),
            self._crop_map_frame(scene, x=80, blur_sigma=0.8),
            self._crop_map_frame(scene, x=80, blur_sigma=1.4),
            self._crop_map_frame(scene, x=80, blur_sigma=2.0),
            self._crop_map_frame(scene, x=80, blur_sigma=2.6),
            self._crop_map_frame(scene, x=80, blur_sigma=3.2),
        ]

        report = None
        for frame in first_position:
            report = analyzer.add_frame(frame)
        self.assertIsNotNone(report)
        self.assertEqual(report.accepted_frames, MAP_BUILD_MAX_TILE_FRAMES)
        self.assertIn("已采够", report.message)
        self.assertGreater(analyzer._skipped_tile_frames, 0)  # noqa: SLF001

        shifted = self._crop_map_frame(scene, x=288)
        committed = self._feed_stable_position(analyzer, shifted, count=3)
        result = analyzer.finalize()

        self.assertEqual(committed.motion_state, "tile_committed")
        self.assertEqual(result.tile_count, 2)
        self.assertEqual(result.metadata.get("skipped_tile_frames"), analyzer._skipped_tile_frames)  # noqa: SLF001

    def test_map_build_uses_light_motion_path_for_settling_and_cached_preview(self) -> None:
        scene = self._make_map_scene()
        stable = self._crop_map_frame(scene, x=80)
        moving = self._crop_map_frame(scene, x=245, blur_sigma=4.0)
        shifted = self._crop_map_frame(scene, x=288)
        analyzer = MapBuildAnalyzer(device_id="microview:0", device_name="Microview #1")

        with patch.object(preview_analysis, "_prepare_frame", wraps=preview_analysis._prepare_frame) as prepare_frame:
            analyzer.add_frame(stable)
            self.assertEqual(prepare_frame.call_count, 0)
            analyzer.add_frame(stable)
            self.assertEqual(prepare_frame.call_count, 1)

            preview_render_count = analyzer._preview_render_count  # noqa: SLF001
            analyzer.add_frame(moving)
            analyzer.add_frame(shifted)

            self.assertEqual(prepare_frame.call_count, 1)
            self.assertEqual(analyzer._preview_render_count, preview_render_count)  # noqa: SLF001

    def test_map_build_waits_for_required_stable_frames_before_sampling(self) -> None:
        base = self._make_map_base()
        frame = bgr_array_to_qimage(base)
        analyzer = MapBuildAnalyzer(device_id="microview:0", device_name="Microview #1")

        report_a = analyzer.add_frame(frame)
        report_b = analyzer.add_frame(frame)
        report_c = analyzer.add_frame(frame)

        self.assertEqual(report_a.motion_state, "settling")
        self.assertEqual(report_b.motion_state, "sampling")
        self.assertEqual(report_c.motion_state, "sampling")
        self.assertEqual(report_b.accepted_frames, 1)
        self.assertEqual(report_c.accepted_frames, 1)
        self.assertIn("已静止", report_b.message)

    def test_map_build_rejects_moving_blurred_frames_until_new_tile_is_stable(self) -> None:
        scene = self._make_map_scene()
        stable = self._crop_map_frame(scene, x=80)
        moving = self._crop_map_frame(scene, x=245, blur_sigma=4.0)
        shifted = self._crop_map_frame(scene, x=288)
        analyzer = MapBuildAnalyzer(device_id="microview:0", device_name="Microview #1")

        steady_report = self._feed_stable_position(analyzer, stable, count=2)
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
        self.assertEqual(settling_c.motion_state, "tile_committed")
        self.assertEqual(stable_report.motion_state, "sampling")
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
