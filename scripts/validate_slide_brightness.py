"""Synthetic slide viewing audit, with real Qt paints and isolated user state.

Run each DPI in a fresh process. Width/height are requested physical screenshot
dimensions; no monitor resolution or system DPI settings are changed. Native
window-manager minimum sizes can enlarge the requested client area, so actual
sizes and devicePixelRatio are recorded. This does not validate camera hardware.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--platform", default="windows" if sys.platform == "win32" else "offscreen")
    parser.add_argument("--width", type=int, default=1366)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--scale", type=float, default=1)
    parser.add_argument("--theme", choices=("light", "dark"), default="light")
    parser.add_argument("--adjustment-mode", choices=("brightness", "exposure", "combined"), default="brightness")
    parser.add_argument("--samples", type=int, default=41)
    parser.add_argument("--require-performance", action="store_true")
    args = parser.parse_args()
    if min(args.width, args.height, args.samples) < 1 or args.scale <= 0:
        parser.error("width, height, scale and samples must be positive")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "result.json").exists() or (output / "synthetic.fdmslide").exists():
        parser.error("use a fresh output directory to retain prior evidence")
    os.environ["QT_QPA_PLATFORM"] = args.platform
    probe_env = os.environ.copy()
    probe_env.pop("QT_SCALE_FACTOR", None)
    probe_env.pop("QT_SCREEN_SCALE_FACTORS", None)
    baseline_dpr = float(subprocess.check_output([
        sys.executable, "-B", "-c",
        "from PySide6.QtWidgets import QApplication; a=QApplication([]); print(a.primaryScreen().devicePixelRatio())",
    ], env=probe_env, text=True).strip())
    os.environ.pop("QT_SCREEN_SCALE_FACTORS", None)
    os.environ["QT_SCALE_FACTOR"] = str(args.scale / baseline_dpr)
    sys.path.insert(0, str(ROOT / "src"))
    source_paths = [ROOT / "src/fdm/ui/digital_slide_canvas.py", ROOT / "src/fdm/ui/slide_brightness.py",
        ROOT / "src/fdm/ui/slide_brightness_control.py", ROOT / "src/fdm/ui/main_window.py",
        ROOT / "src/fdm/ui/scale_overlay_editor.py", ROOT / "src/fdm/settings.py", ROOT / "src/fdm/ui/icons.py",
        ROOT / "src/fdm/ui/dialogs.py", Path(__file__)]
    source_hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source_paths}

    import cv2
    import numpy as np
    from PySide6.QtCore import QCoreApplication, QEvent, Qt
    from PySide6.QtGui import QColor, QImage, QPainter, QPen
    from PySide6.QtWidgets import QApplication
    from fdm import runtime_logging, screenshot_settings, settings
    from fdm.geometry import Line, Point
    from fdm.models import Calibration, ImageDocument, Measurement, OverlayAnnotation
    from fdm.services.digital_slide_store import DigitalSlideManifest, DigitalSlideStore, DigitalSlideTile
    from fdm.services.export_service import ExportImageRenderMode, ExportRenderContext
    from fdm.services.segmentation_source import digital_slide_segmentation_snapshot
    from fdm.settings import AppSettings
    from fdm.ui import digital_slide_canvas, theme
    from fdm.ui.main_window import MainWindow
    from fdm.version import __version__

    app = QApplication.instance() or QApplication([])
    report = dict(schema_version=2, passed=False, adjustment_mode=args.adjustment_mode, scope="Synthetic Qt widget rendering; no hardware or physical monitor acceptance",
        version=__version__, system=platform.platform(), python=sys.version, opencv=cv2.__version__,
        qt=__import__("PySide6").__version__, platform=args.platform, theme=args.theme,
        requested_physical_size=[args.width, args.height], requested_scale=args.scale,
        baseline_screen_dpr=baseline_dpr, qt_scale_factor=os.environ["QT_SCALE_FACTOR"],
        scenarios=[], boundaries=[], screenshots=[], performance={})
    failures = []
    canvas = window = None
    paint_samples = []
    source = output / "synthetic.fdmslide"
    original_paint = digital_slide_canvas.DigitalSlideCanvas.paintEvent

    def paint(widget, event):
        started = time.perf_counter()
        original_paint(widget, event)
        if widget is canvas:
            paint_samples.append(dict(brightness=widget.display_brightness, exposure_tenths=widget.display_exposure_tenths, started=started,
                completed=time.perf_counter()))

    def settle(predicate=lambda: True, timeout=15, minimum=.02):
        deadline = time.monotonic() + timeout
        earliest = time.monotonic() + minimum
        while time.monotonic() < deadline:
            app.processEvents()
            if time.monotonic() >= earliest and predicate():
                return
            time.sleep(.002)
        raise AssertionError("Qt/renderer wait timed out")

    def check(name, condition, **details):
        report["boundaries"].append(dict(name=name, passed=bool(condition), **details))
        if not condition:
            failures.append(name)

    def image_hash(image):
        rgba = image.convertToFormat(QImage.Format.Format_RGBA8888)
        return hashlib.sha256(bytes(rgba.constBits())).hexdigest()

    def capture(name, widget):
        settle(minimum=.05)
        path = output / f"{name}.png"
        pixmap = widget.grab()
        if not pixmap.save(str(path)):
            raise OSError(f"cannot save {path}")
        report["screenshots"].append(dict(name=name, file=path.name,
            size=[pixmap.width(), pixmap.height()], sha256=hashlib.sha256(path.read_bytes()).hexdigest()))

    def cache_state():
        # This explicit list is updated with the implementation's stats API.
        stats = getattr(canvas, "display_brightness_cache_stats", None)
        if stats is not None:
            return stats()
        cache = getattr(canvas, "_brightness_cache", None)
        if cache is not None and hasattr(cache, "stats"):
            return cache.stats()
        return {"bytes": cache.bytes_used, "limit_bytes": cache.max_bytes, "items": len(cache._images)}

    def record_view(name):
        settle(canvas.pixel_work_enabled if not canvas.large_area_browse_active() else lambda: True, minimum=.15)
        rect = canvas.visible_slide_rect()
        report["scenarios"].append(dict(name=name, brightness=canvas.display_brightness, exposure_tenths=canvas.display_exposure_tenths,
            focus=canvas.focus_index(), pixel_work_enabled=canvas.pixel_work_enabled(),
            browse=canvas.large_area_browse_active(), view=[rect.x(), rect.y(), rect.width(), rect.height()],
            cache=cache_state()))

    # The scalar is brightness for brightness/combined runs, or EV tenths for
    # exposure-only runs. Every observation also stores the full pair.
    extent = 40 if args.adjustment_mode == "exposure" else 100

    def state_for(value):
        value = int(value)
        if args.adjustment_mode == "brightness":
            return value, 0
        if args.adjustment_mode == "exposure":
            return 0, value
        return value, int(round(value * .4))

    def state_of(item):
        return item["brightness"], item["exposure_tenths"]

    def values_between(low, high, samples):
        # More samples than slider ticks must not introduce no-op edits.
        return list(dict.fromkeys(np.linspace(low, high, samples).round().astype(int).tolist()))

    def set_canvas(value):
        canvas.set_display_adjustments(*state_for(value))

    def set_slider(popup, value):
        brightness, exposure = state_for(value)
        # Both edits reach the same coalescing batch, without pumping events
        # between them. This tests the real atomic pair application.
        popup.slider.setValue(brightness)
        popup.exposure_slider.setValue(exposure)

    def release_slider(popup):
        slider = popup.exposure_slider if args.adjustment_mode == "exposure" else popup.slider
        slider.sliderReleased.emit()

    def continuous_slider(popup, applied_events):
        """Feed input independently of paint and match repeated values by time."""
        settle(minimum=.05)
        applied_events.clear()
        paint_start = len(paint_samples)
        started = time.perf_counter()
        inputs = []
        next_input = started
        # Exposure has 81 ticks. Repeated ticks are deliberately associated
        # with the latest matching input before each apply, not a value dict.
        while time.perf_counter() - started < 1.0 and len(inputs) < 200:
            app.processEvents()
            now = time.perf_counter()
            if now >= next_input:
                identity = len(inputs)
                value = -extent + identity % (2 * extent)
                brightness, exposure = state_for(value)
                inputs.append(dict(input_id=identity, value=value, brightness=brightness,
                    exposure_tenths=exposure, at=now))
                set_slider(popup, value)
                next_input = now + .005
            time.sleep(.001)

        release_started = time.perf_counter()
        final_state = state_for(extent)
        inputs.append(dict(input_id=len(inputs), value=extent, brightness=final_state[0],
            exposure_tenths=final_state[1], at=release_started))
        set_slider(popup, extent)
        release_slider(popup)
        flush_ms = (time.perf_counter() - release_started) * 1000
        final_flushed = canvas.display_adjustments == final_state and not popup.has_pending_edit
        settle(lambda: any(state_of(item) == final_state and item["started"] >= release_started
            for item in paint_samples[paint_start:]), timeout=5, minimum=0)
        phase_paints = paint_samples[paint_start:]
        updates, unpainted = [], []
        for index, event in enumerate(applied_events):
            state, applied = event["state"], event["at"]
            source_input = next((item for item in reversed(inputs)
                if state_of(item) == state and item["at"] <= applied), None)
            if source_input is None or applied < started:
                continue
            next_applied = applied_events[index + 1]["at"] if index + 1 < len(applied_events) else float("inf")
            painted = next((item for item in phase_paints if state_of(item) == state
                and applied <= item["started"] < next_applied), None)
            if painted is None:
                unpainted.append(dict(input_id=source_input["input_id"], value=source_input["value"],
                    brightness=state[0], exposure_tenths=state[1], applied_at_ms=(applied - started) * 1000))
                continue
            latest_input = next(item for item in reversed(inputs) if item["at"] <= painted["completed"])
            updates.append(dict(input_id=source_input["input_id"], value=source_input["value"],
                brightness=state[0], exposure_tenths=state[1],
                applied_at_ms=(applied - started) * 1000,
                paint_at_ms=(painted["completed"] - started) * 1000,
                apply_to_paint_ms=(painted["completed"] - applied) * 1000,
                displayed_input_to_paint_ms=(painted["completed"] - source_input["at"]) * 1000,
                latest_input_id=latest_input["input_id"],
                latest_input_to_paint_ms=(painted["completed"] - latest_input["at"]) * 1000))
        intervals = [b["paint_at_ms"] - a["paint_at_ms"] for a, b in zip(updates, updates[1:])]
        first_ms = updates[0]["paint_at_ms"] if updates else None
        def p95(key):
            return float(np.percentile([item[key] for item in updates], 95)) if updates else None
        response_p95 = p95("displayed_input_to_paint_ms")
        interval_p95 = float(np.percentile(intervals, 95)) if intervals else None
        final_update = next((item for item in updates if item["input_id"] == inputs[-1]["input_id"]), None)
        passed = (final_flushed and final_update is not None and len(updates) >= 2
            and first_ms <= 100 and response_p95 <= 50 and interval_p95 <= 50)
        return dict(metric="1s input stream at approximately 5ms; natural Qt paints only; product coalescing unchanged",
            adjustment_mode=args.adjustment_mode, target_input_interval_ms=5,
            input_duration_ms=(release_started - started) * 1000,
            inputs=[dict(input_id=item["input_id"], value=item["value"], brightness=item["brightness"],
                exposure_tenths=item["exposure_tenths"], at_ms=(item["at"] - started) * 1000) for item in inputs],
            updates=updates, applied_without_paint=unpainted,
            first_response_ms=first_ms, response_p95_ms=response_p95,
            apply_to_paint_p95_ms=p95("apply_to_paint_ms"), latest_input_to_paint_p95_ms=p95("latest_input_to_paint_ms"),
            update_intervals_ms=intervals, update_interval_p95_ms=interval_p95,
            final_flush_ms=flush_ms, final_value_applied_synchronously=final_flushed,
            final_paint_ms=final_update["displayed_input_to_paint_ms"] if final_update else None,
            first_target_ms=100, p95_target_ms=50, passed=bool(passed))

    with ExitStack() as stack:
        stack.enter_context(patch.object(settings, "settings_file_path", lambda: output / "settings.json"))
        stack.enter_context(patch.object(screenshot_settings, "screenshot_settings_file_path", lambda: output / "screenshot-settings.json"))
        stack.enter_context(patch.object(runtime_logging, "runtime_log_path", lambda: output / "runtime.log"))
        stack.enter_context(patch.object(digital_slide_canvas, "digital_slide_render_cache_directory", lambda: output / "cache"))
        stack.enter_context(patch.object(digital_slide_canvas.DigitalSlideCanvas, "paintEvent", paint))
        try:
            tile_width, tile_height = 1024, 768
            manifest = DigitalSlideManifest(1, tile_width * 3, tile_height * 3,
                tile_width, tile_height, [-4, -2, 0, 2, 4])
            store = DigitalSlideStore.create(source, manifest)
            yy, xx = np.mgrid[:tile_height, :tile_width]
            try:
                for focus in range(5):
                    for row in range(3):
                        for column in range(3):
                            field = np.empty((tile_height, tile_width, 3), dtype=np.uint8)
                            field[:, :, 0] = (xx // 5 + row * 16 + focus * 11) % 220 + 15
                            field[:, :, 1] = (yy // 4 + column * 20 + focus * 15) % 220 + 15
                            field[:, :, 2] = ((xx + yy) // 8 + focus * 21) % 220 + 15
                            image = QImage(field.data, tile_width, tile_height, field.strides[0], QImage.Format.Format_RGB888).copy()
                            painter = QPainter(image)
                            painter.setPen(QPen(QColor("#202020"), 9))
                            for offset in range(50, tile_width, 120):
                                painter.drawLine(offset, 100, offset + 80, 650)
                            painter.setPen(QColor("white"))
                            font = painter.font()
                            font.setPixelSize(42)
                            painter.setFont(font)
                            painter.drawText(35, 60, f"F{focus + 1} / XY {column + 1},{row + 1}")
                            painter.end()
                            store.write_tile(DigitalSlideTile(focus, column * tile_width, row * tile_height,
                                tile_width, tile_height), image, update_manifest=False)
                store.write_manifest(manifest)
            finally:
                store.close()
            source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
            document = ImageDocument(id="brightness-validation", path=str(source),
                image_size=(manifest.width, manifest.height), document_kind="digital_slide",
                calibration=Calibration("image_scale", 2, "um", "synthetic"),
                metadata={"digital_slide": {"viewport_origin": [tile_width, tile_height], "focus_index": 2}},
                measurements=[Measurement(id="line", image_id="brightness-validation", fiber_group_id=None,
                    mode="manual", line_px=Line(Point(1200, 1100), Point(1680, 1100)))],
                overlay_annotations=[OverlayAnnotation(id="label", image_id="brightness-validation", kind="text",
                    content="查看亮度与曝光 / 原始导出", anchor_px=Point(1250, 1250))])
            document.initialize_runtime_state()
            document.measurements[0].recalculate(document.calibration)
            window = MainWindow()
            window._app_settings = AppSettings(theme_mode=args.theme, digital_slide_render_cache_gib=0,
                selected_capture_device_id="slide-adjustments-validation-unavailable", digital_slide_motor_output_enabled=False)
            theme.apply_application_theme(app, args.theme)
            window.resize(round(args.width / args.scale), round(args.height / args.scale))
            window.show()
            # Windows clamps the initial show to the desktop work area. Apply
            # the requested client size after native creation, without changing
            # monitor resolution or desktop DPI. Parts may be off-screen; grab()
            # below verifies the full widget's raster layout, not a 4K monitor.
            app.processEvents()
            window.resize(round(args.width / args.scale), round(args.height / args.scale))
            window._add_digital_slide_document_from_path(source, document=document)
            canvas = window._canvases[document.id]
            settle(canvas.pixel_work_enabled, minimum=.25)
            canvas.set_overview_enabled(True)
            canvas.request_overview()
            settle(lambda: not canvas.overview_image().isNull(), minimum=.1)
            report["actual_logical_size"] = [window.width(), window.height()]
            report["actual_dpr"] = canvas.devicePixelRatioF()
            report["screen_logical_size"] = [window.screen().size().width(), window.screen().size().height()]
            report["window_exposed"] = window.windowHandle().isExposed()
            report["native_viewport_size"] = [tile_width, tile_height]
            report["synthetic_source_sha256_before"] = source_hash
            document_before = deepcopy(document.to_dict())
            dirty_before = asdict(document.dirty_flags)
            image_before = image_hash(canvas._image)
            overview_before = image_hash(canvas.overview_image())
            button = window._slide_brightness_button
            geometry_before = (canvas.width(), canvas.height(), button.width(), button.height())
            check("fixed_label_and_icon", button.text() == "亮度/曝光" and not button.icon().isNull(),
                  text=button.text(), icon_present=not button.icon().isNull())
            processing_before = window._create_processing_source_context()[0].plane.sha256()
            mounted_store = window._slide_stores[document.id]
            def segmentation_hash():
                return digital_slide_segmentation_snapshot(document, mounted_store,
                    origin_px=canvas.viewport_origin(), width=tile_width, height=tile_height,
                    focus_index=canvas.focus_index()).source_version
            segment_before = segmentation_hash()
            capture("native-original", window)
            timings = []
            values = values_between(-extent, extent, args.samples)
            for value in values:
                count = len(paint_samples)
                started = time.perf_counter()
                set_canvas(value)
                settle(lambda: len(paint_samples) > count, timeout=5, minimum=0)
                completed = next(item for item in paint_samples[count:] if state_of(item) == state_for(value))
                timings.append(dict(value=value, brightness=state_for(value)[0], exposure_tenths=state_for(value)[1], response_ms=(completed["completed"] - started) * 1000,
                    paint_ms=(completed["completed"] - completed["started"]) * 1000, cache=cache_state()))
            response = [item["response_ms"] for item in timings]
            report["performance"] = dict(metric="setter-to-real-Qt-paint-completion; event pumping interval 2ms; no physical display latency",
                samples=timings, first_response_ms=response[0], response_p50_ms=float(np.percentile(response, 50)),
                response_p95_ms=float(np.percentile(response, 95)), first_target_ms=100, p95_target_ms=50,
                passed=response[0] <= 100 and float(np.percentile(response, 95)) <= 50)
            window.digital_slide_brightness_action.trigger()
            popup = window._slide_brightness_popup
            settle(minimum=.05)
            ui_timings = []
            applied_events = []
            canvas.displayAdjustmentsChanged.connect(lambda brightness, exposure: applied_events.append(
                dict(state=(brightness, exposure), at=time.perf_counter())))
            for value in values_between(round(-extent * .9), round(extent * .9), min(args.samples, 31)):
                count = len(paint_samples)
                apply_start = len(applied_events)
                started = time.perf_counter()
                set_slider(popup, value)
                expected_state = state_for(value)
                settle(lambda: any(state_of(item) == expected_state for item in paint_samples[count:]), timeout=5, minimum=0)
                completed = next(item for item in paint_samples[count:] if state_of(item) == expected_state)
                applied = next(item["at"] for item in applied_events[apply_start:] if item["state"] == expected_state)
                ui_timings.append(dict(value=value, brightness=expected_state[0], exposure_tenths=expected_state[1],
                    response_ms=(completed["completed"] - started) * 1000,
                    apply_ms=(applied - started) * 1000,
                    paint_after_apply_ms=(completed["completed"] - applied) * 1000))
            release_started = time.perf_counter()
            released_value = round(extent * .77)
            set_slider(popup, released_value)
            release_slider(popup)
            check("slider_release_flushes_final_value", canvas.display_adjustments == state_for(released_value),
                  elapsed_ms=(time.perf_counter() - release_started) * 1000)
            ui_response = [item["response_ms"] for item in ui_timings]
            report["ui_performance"] = dict(metric="slider-valueChanged-to-real-Qt-paint-completion; includes 33ms coalescing; event pumping interval 2ms",
                samples=ui_timings, first_response_ms=ui_response[0], response_p95_ms=float(np.percentile(ui_response, 95)),
                passed=ui_response[0] <= 100 and float(np.percentile(ui_response, 95)) <= 50)
            report["continuous_ui_performance"] = continuous_slider(popup, applied_events)
            # A failed continuous interaction target must not be hidden by the
            # existing correctness-only passed/exit-code behavior.
            check("continuous_ui_response_targets", report["continuous_ui_performance"]["passed"])
            popup.close()
            settle()
            check("adjustment_geometry_stable", geometry_before == (canvas.width(), canvas.height(), button.width(), button.height()),
                  before=geometry_before, after=(canvas.width(), canvas.height(), button.width(), button.height()))
            check("overview_reflects_adjustments", image_hash(canvas.overview_image()) != overview_before)
            check("mounted_raw_image_unchanged", image_hash(canvas._image) == image_before)
            check("document_serialization_unchanged", document.to_dict() == document_before)
            check("dirty_flags_unchanged", asdict(document.dirty_flags) == dirty_before)
            check("processing_source_unchanged", window._create_processing_source_context()[0].plane.sha256() == processing_before)
            check("segmentation_source_unchanged", segmentation_hash() == segment_before)
            context = ExportRenderContext(document_id=document.id, render_mode=ExportImageRenderMode.CURRENT_VIEWPORT,
                focus_index=2, origin_x=tile_width, origin_y=tile_height, viewport_width=tile_width, viewport_height=tile_height)
            for name, measure, scale in (("raw", False, False), ("annotations", True, False), ("scale", True, True)):
                hashes = []
                for value in (0, round(-extent * .7), round(extent * .7)):
                    set_canvas(value)
                    path = output / f"export-{name}-{value}.png"
                    window._render_overlay_image(document, path, include_measurements=measure, include_scale=scale,
                        include_annotations=measure, render_mode=ExportImageRenderMode.CURRENT_VIEWPORT, render_context=context)
                    hashes.append(image_hash(QImage(str(path))))
                check(f"export_{name}_original_pixels", len(set(hashes)) == 1, pixel_sha256=hashes)
            set_canvas(round(extent * .6))
            record_view("native_bright")
            capture("native-bright", window)
            for focus in (4, 0, 1, 3, 2):
                canvas.set_focus_index(focus)
            record_view("rapid_focus_final")
            check("focus_change_keeps_adjustments", canvas.display_adjustments == state_for(round(extent * .6)))
            for dx, dy in ((tile_width, 0), (-tile_width, 0), (64, 0), (64, 0), (0, 64), (-128, -64)):
                canvas.move_viewport_by(dx, dy, throttled=abs(dx) < tile_width)
                record_view(f"move_{dx}_{dy}")
                check(f"move_keeps_adjustments_{len(report['scenarios'])}", canvas.display_adjustments == state_for(round(extent * .6)))
            canvas.fit_to_view()
            record_view("whole_slide_browse")
            capture("overview-bright", window)
            canvas.fit_native_viewport()
            record_view("native_return")
            window.scale_preview.begin()
            settle(canvas.pixel_work_enabled, minimum=.2)
            capture("scale-export-preview-original-brightness", window)
            check("scale_preview_keeps_runtime_adjustments", canvas.display_adjustments == state_for(round(extent * .6)))
            window.scale_preview.cancel()
            window.scale_preview.hide()
            check("leaving_scale_preview_restores_view_adjustments", canvas.display_adjustments == state_for(round(extent * .6)))
            set_canvas(round(-extent * .6))
            capture("native-darkened", window)
            set_canvas(round(extent * .6))
            # Popup screenshots are added through the feature's public action.
            window.digital_slide_brightness_action.trigger()
            settle(minimum=.1)
            capture("brightness-popup-window", window)
            popup = window._slide_brightness_popup
            check("popup_is_visible", popup is not None and popup.isVisible())
            if popup is not None:
                capture("brightness-popup", popup)
                check("remember_initially_unchecked", not popup.remember_checkbox.isChecked())
                def persisted_state():
                    saved = settings.AppSettingsIO.load()
                    return (saved.digital_slide_remember_view_adjustments,
                        saved.digital_slide_view_brightness, saved.digital_slide_view_exposure_tenths)
                popup.remember_checkbox.setChecked(True)
                expected_saved = (True, *canvas.display_adjustments)
                check("remember_saves_pair", persisted_state() == expected_saved,
                      persisted=persisted_state(), expected=expected_saved)
                changed_value = round(extent * .5)
                set_slider(popup, changed_value)
                settle(lambda: canvas.display_adjustments == state_for(changed_value))
                check("preview_does_not_persist", persisted_state() == expected_saved)
                release_slider(popup)
                check("release_persists_pair", persisted_state() == (True, *state_for(changed_value)))
                capture("adjustments-popup-remembered", popup)
                popup.reset_button.click()
                check("reset_clears_both_and_keeps_remember", canvas.display_adjustments == (0, 0)
                    and popup.remember_checkbox.isChecked() and persisted_state() == (True, 0, 0))
                check("reset_restores_raw_overview", image_hash(canvas.overview_image()) == image_hash(canvas._overview_image))
                capture("adjustments-popup-reset", popup)
                set_slider(popup, changed_value)
                release_slider(popup)
                popup.remember_checkbox.setChecked(False)
                check("uncheck_clears_defaults_keeps_view", persisted_state() == (False, 0, 0)
                    and canvas.display_adjustments == state_for(changed_value))
                popup.close()
                capture("remember-disabled-current-view", window)
            check("source_file_unchanged", hashlib.sha256(source.read_bytes()).hexdigest() == source_hash)
            report["synthetic_source_sha256_after"] = hashlib.sha256(source.read_bytes()).hexdigest()
            report["final_cache"] = cache_state()
            report["passed"] = not failures
        except Exception as exc:
            import traceback
            report["exception"] = repr(exc)
            report["traceback"] = traceback.format_exc()
            failures.append(repr(exc))
        finally:
            if window is not None:
                window._reset_workspace()
                window.close()
                window.deleteLater()
            QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
            report["failures"] = failures
            report["source_files"] = source_hashes
            report["source_files_after"] = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in source_paths}
            if report["source_files_after"] != source_hashes:
                failures.append("source_changed_during_validation")
                report["passed"] = False
            (output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    summary = dict(adjustment_mode=args.adjustment_mode, passed=report["passed"], performance_passed=report["performance"].get("passed", False)
        and report.get("ui_performance", {}).get("passed", False)
        and report.get("continuous_ui_performance", {}).get("passed", False),
        first_response_ms=report["performance"].get("first_response_ms"), p95_ms=report["performance"].get("response_p95_ms"),
        slider_p95_ms=report.get("ui_performance", {}).get("response_p95_ms"),
        continuous_p95_ms=report.get("continuous_ui_performance", {}).get("response_p95_ms"),
        continuous_update_interval_p95_ms=report.get("continuous_ui_performance", {}).get("update_interval_p95_ms"),
        failures=failures, output=str(output))
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if report["passed"] and (not args.require_performance or summary["performance_passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
