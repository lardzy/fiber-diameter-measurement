from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import sys
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget

from fdm.canvas_benchmark import (
    BENCHMARK_OUTPUT_ROOT,
    PROJECT_ROOT,
    SCENARIOS,
    _BenchmarkCanvas,
    _ensure_application,
    _resolve_output_path,
    _settle_visible_canvas,
    _trace_overlay_drop_reasons,
    _wait_for_overlay_tiles,
    build_scenario,
    main,
    run_benchmark,
)
from fdm.ui.canvas_overlay_cache import (
    CanvasOverlayTileKey,
    canvas_overlay_tile_cache,
)


class CanvasBenchmarkTests(unittest.TestCase):
    def test_required_scenarios_are_registered(self) -> None:
        self.assertEqual(
            set(SCENARIOS),
            {
                "length_labels_500",
                "length_labels_1000",
                "length_no_labels_500",
                "length_no_labels_1000",
                "areas_holes_100",
                "areas_holes_300",
                "areas_holes_500",
                "area_coordinates_200000",
                "area_coordinates_600000",
                "offscreen_5000",
            },
        )

    def test_every_registered_scenario_builds_with_a_small_override(self) -> None:
        for name, definition in SCENARIOS.items():
            with self.subTest(name=name):
                scenario = build_scenario(
                    name,
                    object_count=1,
                    coordinate_count=(
                        8 if definition.family == "area" else None
                    ),
                    canvas_size=(160, 120),
                )
                self.assertEqual(scenario.object_count, 1)
                self.assertGreater(scenario.coordinate_count, 0)

    def test_area_builder_preserves_requested_object_and_coordinate_counts(self) -> None:
        scenario = build_scenario(
            "areas_holes_300",
            object_count=3,
            coordinate_count=30,
            canvas_size=(320, 240),
        )

        self.assertEqual(scenario.object_count, 3)
        self.assertEqual(scenario.coordinate_count, 30)
        self.assertTrue(scenario.settings.area_measurement_label_style.enabled)
        self.assertTrue(
            all(
                len(measurement.area_rings_px) == 2
                for measurement in scenario.document.measurements
            )
        )

    def test_visible_label_scenarios_generate_distinct_formatted_results(self) -> None:
        length = build_scenario(
            "length_labels_1000",
            object_count=50,
            canvas_size=(320, 240),
        )
        length_decimals = (
            length.settings.length_measurement_label_style.decimals
        )
        length_labels = {
            f"{measurement.diameter_unit:.{length_decimals}f}"
            for measurement in length.document.measurements
        }
        self.assertEqual(len(length_labels), length.object_count)

        area = build_scenario(
            "areas_holes_100",
            object_count=20,
            coordinate_count=160,
            canvas_size=(320, 240),
        )
        area_decimals = area.settings.area_measurement_label_style.decimals
        area_labels = {
            f"{measurement.area_unit:.{area_decimals}f}"
            for measurement in area.document.measurements
        }
        self.assertEqual(len(area_labels), area.object_count)

    def test_small_offscreen_run_returns_stable_json_schema(self) -> None:
        result = run_benchmark(
            "offscreen_5000",
            object_count=4,
            frames=2,
            warmup_frames=0,
            canvas_size=(320, 240),
            idle_ms=5,
        )

        self.assertEqual(result["schema_version"], 1)
        scenario = result["scenario"]
        self.assertEqual(scenario["name"], "offscreen_5000")
        self.assertEqual(scenario["object_count"], 4)
        self.assertEqual(scenario["coordinate_count"], 8)
        self.assertIn("environment", result)
        self.assertIn("rss", result)
        self.assertEqual(result["render_path"]["requested"], "direct")
        self.assertEqual(result["render_path"]["effective_hot"], "direct")
        self.assertEqual(scenario["canvas_kind"], "document")
        self.assertFalse(result["overlay_cache"]["enabled"])
        self.assertEqual(result["overlay_cache"]["tiles"]["entries"], 0)
        self.assertEqual(result["overlay_cache"]["tiles"]["pending_bytes"], 0)
        self.assertEqual(result["overlay_cache"]["late_drop_count"], 0)
        self.assertEqual(
            result["overlay_cache"]["generation_late_drop_count"],
            0,
        )
        self.assertEqual(
            result["overlay_cache"]["defensive_drop_count"],
            result["overlay_cache"]["generation_late_drop_count"]
            + result["overlay_cache"]["other_defensive_drop_count"],
        )
        timing = result["timing_ms"]
        self.assertEqual(timing["cold"]["frame_count"], 1)
        self.assertEqual(timing["hot"]["frame_count"], 2)
        for field in ("p50", "p95", "max"):
            self.assertIsInstance(timing["cold"][field], float)
            self.assertIsInstance(timing["hot"][field], float)
            self.assertGreaterEqual(timing["hot"][field], 0.0)
        interactions = result["interactions"]
        self.assertEqual(
            set(interactions),
            {"pan", "zoom", "selection", "area_point", "drag", "idle"},
        )
        for name in ("pan", "zoom", "selection", "drag"):
            self.assertTrue(interactions[name]["applicable"])
            self.assertEqual(interactions[name]["action_count"], 1)
            self.assertEqual(interactions[name]["render_count"], 1)
            self.assertGreaterEqual(interactions[name]["combined_ms"], 0.0)
        self.assertFalse(interactions["area_point"]["applicable"])
        self.assertEqual(interactions["idle"]["configured_duration_ms"], 5)
        self.assertTrue(interactions["idle"]["canvas_visible"])
        self.assertTrue(interactions["idle"]["settle"]["settled"])
        self.assertTrue(interactions["idle"]["valid"])
        self.assertTrue(interactions["idle"]["quiescent"])
        self.assertEqual(interactions["idle"]["pending_bytes_after"], 0)
        self.assertGreaterEqual(interactions["idle"]["paint_events_delta"], 0)
        workload = result["render_workload"]
        self.assertIn("unmodified QPainter", workload["instrumentation"])
        self.assertEqual(workload["classified_cached_frames"], 0)
        draw_calls = workload["draw_calls"]
        self.assertFalse(draw_calls["timed"])
        self.assertEqual(draw_calls["render_count"], 1)
        self.assertGreaterEqual(draw_calls["draw_path"], 0)
        self.assertGreaterEqual(draw_calls["draw_image"], 1)
        self.assertGreaterEqual(draw_calls["draw_pixmap"], 0)
        self.assertGreaterEqual(draw_calls["picture_play"], 0)

        runtime_caches = result["runtime_caches"]
        self.assertTrue(runtime_caches["cold_reset"]["drained"])
        for cache_name in (
            "area_paths",
            "label_sprites",
            "area_handles",
            "overlay_tiles",
        ):
            self.assertEqual(
                runtime_caches["before_render"][cache_name]["entries"],
                0,
            )

        generated = build_scenario(
            "offscreen_5000",
            object_count=4,
            canvas_size=(320, 240),
        )
        self.assertTrue(
            all(
                measurement.line_px is not None
                and measurement.line_px.start.x > 320
                for measurement in generated.document.measurements
            )
        )
        self.assertNotEqual(generated.document.view_state.zoom, 1.0)

    def test_cli_json_mode_prints_one_parseable_document(self) -> None:
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            exit_code = main(
                [
                    "--scenario",
                    "length_labels_500",
                    "--objects",
                    "2",
                    "--frames",
                    "1",
                    "--warmup",
                    "0",
                    "--width",
                    "320",
                    "--height",
                    "240",
                    "--idle-ms",
                    "5",
                    "--json",
                ]
            )

        self.assertEqual(exit_code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["scenario"]["name"], "length_labels_500")
        self.assertEqual(payload["scenario"]["object_count"], 2)
        self.assertTrue(payload["scenario"]["labels_enabled"])
        self.assertEqual(payload["render_path"]["requested"], "direct")

    def test_cli_lists_every_registered_scenario(self) -> None:
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            exit_code = main(["--list-scenarios"])

        self.assertEqual(exit_code, 0)
        listed = {
            line.split(":", 1)[0]
            for line in stdout.getvalue().splitlines()
            if line
        }
        self.assertEqual(listed, set(SCENARIOS))

    def test_overlay_cache_run_waits_for_tiles_and_reports_hot_hits(self) -> None:
        result = run_benchmark(
            "length_labels_500",
            object_count=2,
            frames=2,
            warmup_frames=0,
            canvas_size=(320, 240),
            overlay_cache=True,
            overlay_cache_timeout_ms=3_000,
            idle_ms=5,
        )

        self.assertEqual(result["render_path"]["requested"], "overlay_cache")
        wait = result["overlay_cache"]["wait"]
        self.assertTrue(wait["ready"], result)
        self.assertFalse(wait["timed_out"])
        self.assertGreater(wait["requested_tiles"], 0)
        tiles = result["overlay_cache"]["tiles"]
        self.assertGreater(tiles["entries"], 0)
        self.assertGreater(tiles["bytes"], 0)
        hot = result["overlay_cache"]["hot_activity"]
        self.assertGreater(hot["hits"], 0)
        self.assertEqual(hot["misses"], 0)
        self.assertEqual(hot["hit_rate"], 1.0)

    def test_cache_state_is_cold_for_each_run_and_labels_are_not_collapsed(self) -> None:
        results = [
            run_benchmark(
                "length_labels_500",
                object_count=3,
                frames=1,
                warmup_frames=0,
                canvas_size=(240, 180),
                idle_ms=5,
            )
            for _ in range(2)
        ]

        for result in results:
            runtime_caches = result["runtime_caches"]
            self.assertEqual(
                runtime_caches["before_render"]["label_sprites"]["entries"],
                0,
            )
            self.assertEqual(
                runtime_caches["after_cold"]["label_sprites"]["entries"],
                3,
            )
            self.assertEqual(
                runtime_caches["activity"]["label_sprites"]["misses"],
                3,
            )

    def test_unmeasured_trace_counts_area_paths_and_label_images(self) -> None:
        result = run_benchmark(
            "areas_holes_100",
            object_count=2,
            coordinate_count=16,
            frames=1,
            warmup_frames=0,
            canvas_size=(240, 180),
            idle_ms=5,
        )

        draw_calls = result["render_workload"]["draw_calls"]
        self.assertGreater(draw_calls["draw_path"], 0)
        self.assertGreater(draw_calls["draw_image"], 1)
        self.assertGreater(
            result["runtime_caches"]["after_cold"]["area_paths"]["entries"],
            0,
        )
        self.assertGreater(
            result["runtime_caches"]["after_interactions"]["area_handles"][
                "entries"
            ],
            0,
        )

    def test_no_label_scenario_does_not_populate_label_cache(self) -> None:
        result = run_benchmark(
            "length_no_labels_500",
            object_count=3,
            frames=1,
            warmup_frames=0,
            canvas_size=(240, 180),
            idle_ms=5,
        )

        self.assertEqual(
            result["runtime_caches"]["after_interactions"][
                "label_sprites"
            ]["entries"],
            0,
        )

    def test_shown_offscreen_canvas_has_no_unsolicited_paint_for_500ms(self) -> None:
        result = run_benchmark(
            "length_labels_500",
            object_count=1,
            frames=1,
            warmup_frames=0,
            canvas_size=(240, 180),
            idle_ms=500,
        )

        idle = result["interactions"]["idle"]
        self.assertTrue(idle["canvas_visible"])
        self.assertTrue(idle["settle"]["settled"], idle)
        self.assertTrue(idle["valid"], idle)
        self.assertTrue(idle["quiescent"], idle)
        self.assertEqual(idle["paint_events_delta"], 0, idle)

    def test_visible_paint_self_loop_cannot_be_reported_as_settled(self) -> None:
        class SelfUpdatingWidget(QWidget):
            def __init__(self) -> None:
                super().__init__()
                self.paint_event_count = 0

            def paintEvent(self, event) -> None:  # noqa: N802
                del event
                self.paint_event_count += 1
                self.update()

        app = _ensure_application()
        widget = SelfUpdatingWidget()
        widget.resize(80, 60)
        widget.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
        widget.show()
        try:
            result = _settle_visible_canvas(
                app,
                widget,
                quiet_ms=10,
                timeout_ms=50,
            )
            self.assertFalse(result["settled"], result)
            self.assertTrue(result["timed_out"], result)
        finally:
            widget.close()
            widget.deleteLater()
            app.processEvents()

    def test_generation_late_drop_observer_is_separate_from_aggregate(self) -> None:
        app = _ensure_application()
        scenario = build_scenario(
            "length_labels_500",
            object_count=1,
            canvas_size=(160, 120),
        )
        canvas = _BenchmarkCanvas()
        canvas.resize(160, 120)
        canvas.set_document(scenario.document, scenario.image)
        stale_key = CanvasOverlayTileKey(
            document_token=id(scenario.document),
            document_id=scenario.document.id,
            zoom=round(float(canvas._zoom), 8),
            device_pixel_ratio=round(
                max(1.0, float(canvas.devicePixelRatioF())),
                4,
            ),
            tile_x=99,
            tile_y=99,
            style_generation=canvas._overlay_style_generation,
            tile_epoch=0,
            show_area_fill=True,
        )
        dropped_before = canvas_overlay_tile_cache.stats().dropped
        try:
            with _trace_overlay_drop_reasons(canvas) as trace:
                canvas_overlay_tile_cache._drop_completion(stale_key, -1)
            self.assertEqual(trace.generation_late, 1)
            self.assertEqual(trace.other_defensive, 0)
            self.assertEqual(
                canvas_overlay_tile_cache.stats().dropped - dropped_before,
                1,
            )
        finally:
            canvas.clear_document()
            canvas.close()
            canvas.deleteLater()
            canvas_overlay_tile_cache.clear()
            app.processEvents()

    def test_overlay_cache_timeout_must_be_positive(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be greater than zero"):
            run_benchmark(
                "length_labels_500",
                object_count=1,
                frames=1,
                canvas_size=(160, 120),
                overlay_cache=True,
                overlay_cache_timeout_ms=0,
                idle_ms=5,
            )

    def test_area_interactions_and_digital_slide_canvas_are_selectable(self) -> None:
        result = run_benchmark(
            "areas_holes_100",
            object_count=1,
            coordinate_count=12,
            frames=1,
            warmup_frames=0,
            canvas_size=(240, 180),
            canvas_kind="digital_slide",
            idle_ms=5,
        )

        self.assertEqual(result["scenario"]["canvas_kind"], "digital_slide")
        slide = result["scenario"]["digital_slide"]
        self.assertTrue(slide["set_slide_document_used"])
        self.assertNotEqual(slide["viewport_origin"], {"x": 0.0, "y": 0.0})
        self.assertGreaterEqual(slide["store_render_calls"], 3)
        self.assertGreater(slide["viewport_buffer_requests"], 0)
        self.assertTrue(result["interactions"]["area_point"]["applicable"])
        self.assertEqual(result["interactions"]["area_point"]["action_count"], 1)
        self.assertIn(
            "scalar preview offset",
            result["interactions"]["drag"]["workload"],
        )

    def test_canvas_kind_and_idle_window_are_validated(self) -> None:
        with self.assertRaisesRegex(ValueError, "canvas_kind"):
            run_benchmark(
                "length_labels_500",
                object_count=1,
                frames=1,
                canvas_kind="unknown",
                idle_ms=5,
            )
        with self.assertRaisesRegex(ValueError, "idle_ms"):
            run_benchmark(
                "length_labels_500",
                object_count=1,
                frames=1,
                idle_ms=-1,
            )

    def test_output_suffix_is_applied_before_ignored_root_validation(self) -> None:
        relative = _resolve_output_path("nested/run")
        self.assertEqual(
            relative,
            (BENCHMARK_OUTPUT_ROOT / "nested" / "run.json").resolve(),
        )
        with self.assertRaisesRegex(ValueError, "ignored directory"):
            _resolve_output_path(str((PROJECT_ROOT / ".tmp").resolve()))
        with self.assertRaisesRegex(ValueError, "ignored directory"):
            _resolve_output_path(str((PROJECT_ROOT / "outside").resolve()))

    def test_overlay_wait_timeout_is_bounded_and_reported(self) -> None:
        class NeverSettledCanvas:
            _overlay_tile_queue = [object()]
            _overlay_tile_active = None
            _overlay_tile_build_scheduled = False
            _overlay_tile_failed = set()

        result = _wait_for_overlay_tiles(
            _ensure_application(),
            NeverSettledCanvas(),
            baseline=canvas_overlay_tile_cache.stats(),
            requested_tile_count=1,
            timeout_ms=2,
        )

        self.assertFalse(result["ready"])
        self.assertTrue(result["timed_out"])
        self.assertEqual(result["timeout_ms"], 2)
        self.assertGreaterEqual(result["elapsed_ms"], 2.0)
        self.assertEqual(result["remaining_queue"], 1)


if __name__ == "__main__":
    unittest.main()
