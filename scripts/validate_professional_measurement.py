"""Isolated synthetic Qt audit for professional measurement and linked labels.

This renders real MainWindow widgets without an application-instance server,
camera, device or user-profile writes. Run each requested DPI in a fresh process.
Screenshots describe widget rasters, not physical monitor or metrology accuracy.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--platform", default="windows" if sys.platform == "win32" else "offscreen")
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=1000)
    parser.add_argument("--scale", type=float, default=1)
    parser.add_argument("--theme", choices=("light", "dark", "both"), default="both")
    parser.add_argument("--samples", type=int, default=40)
    parser.add_argument("--require-performance", action="store_true")
    args = parser.parse_args()
    if min(args.width, args.height, args.samples) < 1 or args.scale <= 0:
        parser.error("width, height, samples and scale must be positive")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "result.json").exists():
        parser.error("use a fresh directory to preserve prior evidence")
    os.environ["QT_QPA_PLATFORM"] = args.platform
    env = os.environ.copy()
    env.pop("QT_SCALE_FACTOR", None)
    env.pop("QT_SCREEN_SCALE_FACTORS", None)
    baseline = float(subprocess.check_output([
        sys.executable, "-B", "-c",
        "from PySide6.QtWidgets import QApplication; a=QApplication([]); print(a.primaryScreen().devicePixelRatio())",
    ], text=True, env=env).strip())
    os.environ.pop("QT_SCREEN_SCALE_FACTORS", None)
    os.environ["QT_SCALE_FACTOR"] = str(args.scale / baseline)
    sys.path.insert(0, str(ROOT / "src"))

    import numpy as np
    from PySide6.QtCore import QCoreApplication, QEvent, QPointF, QRectF, Qt
    from PySide6.QtGui import QColor, QImage, QMouseEvent, QPainter, QPen
    from PySide6.QtWidgets import QApplication
    from fdm import runtime_logging, screenshot_settings, settings
    from fdm.geometry import Line, Point
    from fdm.measurement_annotation import MeasurementAnnotationLayout
    from fdm.metrology import MeasurementDefinition, MeasurementFeature
    from fdm.models import Calibration, ImageDocument, Measurement
    from fdm.services.metrology_service import evaluate_document
    from fdm.settings import AppSettings
    from fdm.ui import theme
    from fdm.ui.canvas import DocumentCanvas
    from fdm.ui.image_loader import ImageLoadRequest
    from fdm.ui.main_window import MainWindow
    from fdm.ui.rendering import resolve_measurement_annotation
    from fdm.version import __version__

    app = QApplication.instance() or QApplication([])
    paths = [ROOT / "src/fdm" / path for path in (
        "measurement_annotation.py", "metrology.py", "services/metrology_service.py",
        "services/metrology_fitting.py", "ui/measurement_annotation_layout.py", "ui/rendering.py",
        "ui/canvas.py", "ui/professional_canvas.py", "ui/professional_measurement.py", "ui/professional_rendering.py",
        "models.py", "history.py", "ui/main_window.py", "ui/digital_slide_canvas.py",
        "ui/measurement_export_preview.py", "ui/scale_overlay_editor.py",
        "services/metrology_source_identity.py", "ui/measurement_results_model.py",
    )] + [Path(__file__)]
    hashes = lambda: {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    report = dict(schema_version=1, passed=False, version=__version__, system=platform.platform(),
                  python=sys.version, qt=__import__("PySide6").__version__, platform=args.platform,
                  scope="Synthetic ordinary image, real Qt widget paints; no physical accuracy or device acceptance",
                  requested_physical_size=[args.width, args.height], requested_scale=args.scale,
                  baseline_screen_dpr=baseline, qt_scale_factor=os.environ["QT_SCALE_FACTOR"],
                  source_hashes=hashes(), screenshots=[], checks=[], performance={}, failures=[])
    window = canvas = None
    paints = []
    original_paint = DocumentCanvas.paintEvent

    def measured_paint(widget, event):
        started = time.perf_counter()
        original_paint(widget, event)
        if widget is canvas:
            paints.append((started, time.perf_counter()))

    def settle(predicate=lambda: True, *, minimum=.03, timeout=15):
        started = time.monotonic()
        while time.monotonic() - started < timeout:
            app.processEvents()
            if time.monotonic() - started >= minimum and predicate():
                return
            time.sleep(.002)
        raise TimeoutError("Qt paint/worker did not settle")

    def check(name, passed, **details):
        report["checks"].append(dict(name=name, passed=bool(passed), **details))
        if not passed:
            report["failures"].append(name)

    def capture(name):
        settle(minimum=.1)
        pixmap = window.grab()
        path = output / f"{name}.png"
        if not pixmap.save(str(path)):
            raise OSError(f"Cannot save {path}")
        report["screenshots"].append(dict(file=path.name, size=[pixmap.width(), pixmap.height()],
            logical_window_size=[window.width(), window.height()], canvas_size=[canvas.width(), canvas.height()],
            device_pixel_ratio=pixmap.devicePixelRatioF(), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))

    def image_hash(image):
        return hashlib.sha256(bytes(image.constBits())).hexdigest()

    def event(kind, position):
        return QMouseEvent(kind, position, position, Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
                           Qt.KeyboardModifier.NoModifier)

    def summary(samples):
        return dict(samples_ms=samples, p50_ms=float(np.percentile(samples, 50)),
                    p95_ms=float(np.percentile(samples, 95)), max_ms=max(samples))

    isolated = AppSettings(theme_mode="light", selected_capture_device_id="metrology-audit-unavailable",
                           digital_slide_motor_output_enabled=False)
    with ExitStack() as stack:
        stack.enter_context(patch.object(settings, "settings_file_path", lambda: output / "settings.json"))
        stack.enter_context(patch.object(screenshot_settings, "screenshot_settings_file_path", lambda: output / "screenshot-settings.json"))
        stack.enter_context(patch.object(runtime_logging, "runtime_log_path", lambda: output / "runtime.log"))
        stack.enter_context(patch("fdm.ui.main_window.AppSettingsIO.load", return_value=isolated))
        stack.enter_context(patch.object(DocumentCanvas, "paintEvent", measured_paint))
        try:
            image = QImage(1200, 800, QImage.Format.Format_RGB32)
            image.fill(QColor("#444d59"))
            painter = QPainter(image)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setPen(QPen(QColor("#b8c4d1"), 2))
            painter.setBrush(QColor("#c5ccd4"))
            painter.drawRoundedRect(QRectF(110, 110, 420, 110), 12, 12)
            painter.drawEllipse(QPointF(850, 235), 100, 100)
            painter.setPen(QPen(QColor("#b8c4d1"), 14))
            painter.drawLine(QPointF(150, 570), QPointF(330, 570))
            painter.drawLine(QPointF(330, 570), QPointF(440, 410))
            painter.drawLine(QPointF(620, 575), QPointF(1050, 640))
            painter.end()
            source = output / "synthetic-metrology.png"
            image.save(str(source))
            document = ImageDocument(id="metrology-validation", path=str(source), image_size=(1200, 800),
                calibration=Calibration("image_scale", 4, "um", "synthetic known geometry"))
            def add(identity, kind, points, offset):
                feature = MeasurementFeature(identity + "-feature", document.id,
                    "polyline" if kind == "angle" else "line" if kind == "distance" else kind,
                    tuple(Point(*p) for p in points), parameters={"robust": True})
                definition = MeasurementDefinition(identity, document.id, kind, (feature.id,),
                    annotation_layout=MeasurementAnnotationLayout(offset_px=Point(*offset),
                        leader_style="dual" if kind in {"distance", "circle", "fit_line"} else "single"))
                document.professional_features.append(feature)
                document.professional_measurements.append(definition)
            add("distance", "distance", [(110, 110), (530, 110)], (0, -62))
            add("circle", "circle", [(950, 235), (850, 335), (750, 235)], (0, -150))
            add("angle", "angle", [(150, 570), (330, 570), (440, 410)], (-40, 80))
            add("fit", "fit_line", [(620+i*22, 575+i*3.3+math.sin(i)*1.3) for i in range(20)], (0, 90))
            legacy = Measurement(id="legacy-diameter", image_id=document.id, fiber_group_id=None,
                mode="manual", line_px=Line(Point(300, 110), Point(300, 220)),
                annotation_layout=MeasurementAnnotationLayout(offset_px=Point(100, 100), leader_style="dual"))
            legacy.recalculate(document.calibration); document.measurements.append(legacy)
            document.initialize_runtime_state()
            window = MainWindow(); window.show()
            window.resize(round(args.width / args.scale), round(args.height / args.scale))
            window._add_loaded_document(ImageLoadRequest(path=str(source), document=document), image)
            canvas = window._canvases[document.id]
            workspace = window.professional_workspace
            workspace.show()
            workspace.select("fit")
            settle(minimum=.15)
            results = evaluate_document(document)
            check("synthetic_dimensions_valid", all(r.valid for r in results.values()), results={k: asdict(v) for k, v in results.items()})
            before_hash = image_hash(canvas._image)
            for chosen_theme in (["light", "dark"] if args.theme == "both" else [args.theme]):
                theme.apply_application_theme(app, chosen_theme)
                window._app_settings.theme_mode = chosen_theme
                window._refresh_theme_sensitive_icons()
                for name, width, height in (("wide", args.width, args.height), ("narrow", min(args.width, 1100), min(args.height, 800))):
                    window.resize(round(width / args.scale), round(height / args.scale))
                    settle(minimum=.1); canvas.fit_to_view(); settle()
                    capture(f"professional-{name}-{chosen_theme}")
            # Exercise the actual signal/history path, preserving values/raw data.
            linked = resolve_measurement_annotation(legacy, document, canvas._settings, canvas.image_to_widget)
            start = linked.text_rect.center()
            old_layout = legacy.annotation_layout
            before_values = {key: asdict(value) for key, value in evaluate_document(document).items()}
            canvas.mousePressEvent(event(QEvent.Type.MouseButtonPress, start))
            canvas.mouseMoveEvent(event(QEvent.Type.MouseMove, start + QPointF(55, 38)))
            canvas.mouseReleaseEvent(event(QEvent.Type.MouseButtonRelease, start + QPointF(55, 38)))
            settle()
            current_legacy = document.get_measurement(legacy.id)
            check("legacy_drag_commits_layout", current_legacy.annotation_layout != old_layout)
            check("drag_preserves_raw_pixels", image_hash(canvas._image) == before_hash)
            check("drag_preserves_professional_values", {k: asdict(v) for k, v in evaluate_document(document).items()} == before_values)
            capture("linked-label-after-drag")

            # Exactly 200 professional labels have centers inside the canvas.
            # Dense overlap is deliberately retained as a load case and counted.
            workspace.dock.hide()
            window.resize(round(args.width / args.scale), round(args.height / args.scale))
            document.measurements.clear()
            document.professional_features.clear(); document.professional_measurements.clear()
            for i in range(200):
                x, y = 45 + (i % 20) * 58, 55 + (i // 20) * 72
                add(f"load-{i}", "distance", [(x-10, y+15), (x+10, y+15)], (0, -15))
            document.mark_professional_geometry_changed()
            document.select_professional(None)
            canvas._annotation_preview_revision += 1
            canvas.fit_to_view(); canvas.update(); settle(minimum=.2)
            controller = canvas._professional_controller
            rect = QRectF(canvas.rect())
            visible = sum(rect.contains(label.text_rect.center()) for label in controller.label_hits.values())
            check("load_case_contains_200_label_centers", visible == 200, visible=visible, count=len(controller.label_hits))
            render_times = []
            for _ in range(args.samples):
                count = len(paints); canvas.update()
                settle(lambda: len(paints) > count, minimum=0)
                render_times.append((paints[count][1]-paints[count][0])*1000)
            identity = "load-199"
            pos = controller.label_hits[identity].text_rect.center()
            count = len(paints); began = time.perf_counter()
            controller.mouse_press(event(QEvent.Type.MouseButtonPress, pos))
            check("professional_label_drag_started", controller.drag is not None)
            canvas.update(); settle(lambda: len(paints) > count, minimum=0)
            professional_press = dict(
                response_ms=(paints[count][1]-began)*1000,
                paint_ms=(paints[count][1]-paints[count][0])*1000,
                scope="mouse press through first paint, including static layer construction",
            )
            drag_times = []; response_times = []
            for i in range(args.samples):
                count = len(paints); began = time.perf_counter()
                controller.mouse_move(event(QEvent.Type.MouseMove, pos + QPointF(5+(i % 15) * .8, math.sin(i*.4)*5)))
                settle(lambda: len(paints) > count, minimum=0)
                drag_times.append((paints[count][1]-paints[count][0])*1000)
                response_times.append((paints[count][1]-began)*1000)
            controller.mouse_release(event(QEvent.Type.MouseButtonRelease, pos + QPointF(12, 0)))
            settle()
            report["performance"] = dict(metric="natural Qt canvas paint duration and event-to-paint completion; excludes physical display latency",
                visible_label_centers=visible, dense_overlap=True, annotation_style="dual", draw=summary(render_times), drag=summary(drag_times),
                drag_response=summary(response_times), press_first_paint=professional_press, target_p95_ms=33,
                passed=float(np.percentile(drag_times, 95)) <= 33,
                actual_canvas_size=[canvas.width(), canvas.height()], device_pixel_ratio=canvas.devicePixelRatioF())
            capture("200-linked-labels")
            document.professional_features.clear(); document.professional_measurements.clear()
            document.mark_professional_geometry_changed()
            for i in range(200):
                x, y = 45 + (i % 20) * 58, 55 + (i // 20) * 72
                item = Measurement(id=f"legacy-load-{i}", image_id=document.id, fiber_group_id=None,
                    mode="manual", line_px=Line(Point(x-10, y+15), Point(x+10, y+15)),
                    annotation_layout=MeasurementAnnotationLayout(offset_px=Point(0, -15), leader_style="dual"))
                item.recalculate(document.calibration)
                document.measurements.append(item)
            document.mark_measurement_geometry_changed()
            canvas._annotation_preview_revision += 1
            canvas.update(); settle(minimum=.3)
            legacy_rects = [resolve_measurement_annotation(m, document, canvas._settings, canvas.image_to_widget).text_rect
                            for m in document.measurements]
            check("legacy_load_contains_200_label_centers", sum(rect.contains(r.center()) for r in legacy_rects) == 200)
            start = legacy_rects[-1].center()
            count = len(paints); began = time.perf_counter()
            canvas.mousePressEvent(event(QEvent.Type.MouseButtonPress, start))
            check("legacy_load_drag_started", canvas._annotation_drag is not None)
            canvas.update(); settle(lambda: len(paints) > count, minimum=0)
            legacy_press = dict(
                response_ms=(paints[count][1]-began)*1000,
                paint_ms=(paints[count][1]-paints[count][0])*1000,
                scope="mouse press through first paint, including static layer construction",
            )
            legacy_paints = []; legacy_responses = []
            for i in range(args.samples):
                count = len(paints); began = time.perf_counter()
                canvas.mouseMoveEvent(event(QEvent.Type.MouseMove, start + QPointF(5+(i%15)*.8, math.sin(i*.4)*5)))
                settle(lambda: len(paints) > count, minimum=0)
                legacy_paints.append((paints[count][1]-paints[count][0])*1000)
                legacy_responses.append((paints[count][1]-began)*1000)
            canvas.mouseReleaseEvent(event(QEvent.Type.MouseButtonRelease, start + QPointF(12, 0)))
            settle()
            report["performance"]["legacy"] = dict(drag=summary(legacy_paints), drag_response=summary(legacy_responses),
                press_first_paint=legacy_press, passed=float(np.percentile(legacy_paints, 95)) <= 33)
            report["performance"]["passed"] &= report["performance"]["legacy"]["passed"]
            capture("200-legacy-linked-labels")
            report["passed"] = not report["failures"]
        except Exception as exc:
            report["failures"].append(repr(exc))
            report["traceback"] = traceback.format_exc()
        finally:
            if window is not None:
                workspace = getattr(window, "professional_workspace", None)
                if workspace is not None:
                    workspace.cancel()
                    for worker in workspace._workers:
                        worker.join(timeout=5)
                window._reset_workspace()
                with patch.object(window, "_confirm_close_documents", return_value=True):
                    window.close()
                window.deleteLater()
            QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
            app.processEvents()
            report["source_hashes_after"] = hashes()
            if report["source_hashes_after"] != report["source_hashes"]:
                report["source_changed_during_run"] = True
            report["passed"] = bool(report["passed"] and not report["failures"])
            (output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(dict(passed=report["passed"], performance=report["performance"], failures=report["failures"], output=str(output)), ensure_ascii=False))
    return 0 if report["passed"] and (not args.require_performance or report["performance"].get("passed")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
