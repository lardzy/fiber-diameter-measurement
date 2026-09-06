"""Reproducible visible-canvas workload; no microscope/model/files are required.

QT_QPA_PLATFORM=offscreen python scripts/benchmark_canvas_interaction.py \
    --image-size 4096 --objects 100 --vertices 10000 --subtracts 10 \
    --output .tmp/canvas-benchmark/interaction-4k.json

Input-to-image timings stop at QImage rendering, not the OS display compositor.
"""

from __future__ import annotations
import argparse
import ast
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import types

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QImage, QMouseEvent, QPainter
from PySide6.QtWidgets import QApplication
from fdm.geometry import Point
from fdm.models import ImageDocument, Measurement
from fdm.settings import AppSettings, MagicSegmentToolMode, MeasurementLabelStyleSettings
from fdm.ui import canvas as canvas_module
from fdm.ui.canvas import DocumentCanvas
from fdm.ui.canvas_overlay_cache import canvas_overlay_tile_cache, canvas_overlay_preview_cache
from fdm.ui.draft_preview_cache import draft_preview_cache


def summary(samples):
    return dict(
        p50_ms=float(np.percentile(samples, 50)),
        p95_ms=float(np.percentile(samples, 95)),
        max_ms=max(samples),
        samples=len(samples),
    )


def ring(vertices, x, y, radius):
    return [
        Point(
            x + radius * (1 + 0.035 * math.sin(i * 1.3)) * math.cos(i * 2 * math.pi / vertices),
            y + radius * (1 + 0.035 * math.sin(i * 1.3)) * math.sin(i * 2 * math.pi / vertices),
        )
        for i in range(vertices)
    ]


def timed(callback):
    start = time.perf_counter()
    callback()
    return (time.perf_counter() - start) * 1000


def run(args):
    app = QApplication.instance() or QApplication([])
    os.environ["FDM_ENABLE_CANVAS_OVERLAY_CACHE"] = "1"
    size = args.image_size
    doc = ImageDocument(id="interaction", path="synthetic.png", image_size=(size, size))
    for i in range(args.objects):
        x = 80 + (i % 10) * (size - 160) / 10
        y = 80 + (i // 10) * (size - 160) / 10
        points = ring(args.vertices, x, y, min(60, size / 30))
        doc.measurements.append(
            Measurement(
                id=str(i),
                image_id=doc.id,
                fiber_group_id=None,
                mode="magic_segment",
                measurement_kind="area",
                polygon_px=points,
                area_rings_px=[points],
                exact_area_px=10000,
            )
        )
    doc.initialize_runtime_state()
    source = QImage(size, size, QImage.Format.Format_RGB32)
    source.fill(0xFFBDBDBD)
    failures = []
    canvas_overlay_tile_cache.tileFailed.connect(lambda key, error: failures.append(error))
    draft_preview_cache._raster_cache.tileFailed.connect(lambda key, error: failures.append(error))
    canvas = DocumentCanvas()
    canvas.resize(1024, 768)
    canvas.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
    canvas.set_document(doc, source)
    canvas.set_settings(
        AppSettings(area_measurement_label_style=MeasurementLabelStyleSettings(enabled=True))
    )
    canvas.set_tool_mode(MagicSegmentToolMode.STANDARD)
    canvas._zoom = min(1, 768 / size)
    canvas._pan = Point(0, 0)
    canvas.show()
    surface = QImage(canvas.size(), QImage.Format.Format_ARGB32_Premultiplied)
    # The app has already rendered its Chinese labels before users open images.
    warm = QImage(200, 50, QImage.Format.Format_RGB32)
    p = QPainter(warm)
    p.drawText(0, 25, "正在载入测量显示… 100.00 px")
    p.end()
    paint = lambda: canvas.render(surface)
    first = timed(paint)
    deadline = time.perf_counter() + 20
    pumps = []
    while time.perf_counter() < deadline:
        pumps.append(timed(app.processEvents))
        current = canvas._scene_preview_key()
        keys = canvas._visible_overlay_tile_keys(canvas._paint_context())
        if canvas_overlay_preview_cache.contains(current) and all(
            canvas_overlay_tile_cache.contains(key) for key in keys
        ):
            break
        time.sleep(0.001)
    ready = time.perf_counter() < deadline
    primary = ring(args.vertices, size / 2, size / 2, min(size / 4, 256))
    session = canvas._magic_segment
    session.primary_polygon = primary
    session.primary_rings = [primary]
    for index in range(args.subtracts):
        points = ring(max(100, args.vertices // 10), size / 2 + index * 10, size / 2 + 20, 30)
        session.confirmed_subtract_polygons.append(points)
        session.confirmed_subtract_rings.append([points])
    # Compare the exact old draft routine on the same scene and same Qt build.
    # All baseline code is read from the explicit local Git ref supplied here.
    baseline = None
    canvas._panning = True

    def preview():
        painter = QPainter(surface)
        try:
            canvas._draw_magic_segment_preview(painter)
        finally:
            painter.end()

    if args.preview_baseline_ref:
        source_text = subprocess.check_output(
            ["git", "show", f"{args.preview_baseline_ref}:src/fdm/ui/canvas.py"], text=True
        )
        tree = ast.parse(source_text)
        node = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "_draw_magic_area_preview"
        )
        scope = dict(vars(canvas_module))
        exec(compile(ast.Module(body=[node], type_ignores=[]), "baseline_preview", "exec"), scope)
        saved = canvas._draw_magic_area_preview
        baseline_method = types.MethodType(scope[node.name], canvas)

        def old_preview(*args, **kwargs):
            kwargs.pop("layer_id", None)
            return baseline_method(*args, **kwargs)

        canvas._draw_magic_area_preview = old_preview
        baseline = summary([timed(preview) for _ in range(12)])
        canvas._draw_magic_area_preview = saved
    draft_preview_cache.discard(id(canvas))
    draft_cold = timed(preview)
    start = time.perf_counter()
    while draft_preview_cache._requests and time.perf_counter() - start < 10:
        app.processEvents()
        time.sleep(0.001)
    draft_ready_ms = (time.perf_counter() - start) * 1000
    draft_hot = summary([timed(preview) for _ in range(30)])
    canvas._panning = False
    position = QPointF(300, 300)

    def event(kind, buttons):
        return QMouseEvent(
            kind,
            position,
            position,
            Qt.MouseButton.RightButton
            if kind != QEvent.Type.MouseMove
            else Qt.MouseButton.NoButton,
            buttons,
            Qt.KeyboardModifier.NoModifier,
        )

    canvas.mousePressEvent(event(QEvent.Type.MouseButtonPress, Qt.MouseButton.RightButton))
    handlers = []
    paints = []
    combined = []
    installs = []
    additions = []
    dispatches = []
    for i in range(args.frames):
        if i == args.frames // 3:
            measurement = doc.measurements[0]
            edited = ring(args.vertices, 160, 160, 65)

            def install():
                measurement.replace_area_geometry(
                    polygon_px=edited, area_rings_px=[edited], exact_area_px=12000
                )
                doc.mark_measurement_geometry_changed()
                doc.mark_session_dirty()
                canvas.notify_document_visual_changed()

            installs.append(timed(install))
        if i == args.frames // 2:
            added = Measurement(
                id="accepted-now",
                image_id=doc.id,
                fiber_group_id=None,
                mode="magic_segment",
                measurement_kind="area",
                polygon_px=primary,
                area_rings_px=[primary],
                exact_area_px=10000,
            )

            def append():
                canvas._overlay_accepted_previews[added.id] = (
                    canvas._preserve_magic_display_preview()
                )
                doc.insert_measurement_incremental(added)
                canvas.notify_document_visual_changed(added_measurement_ids=(added.id,))

            additions.append(timed(append))
        if i == 2 * args.frames // 3:
            canvas._zoom *= 1.25
        position += QPointF(18 if (i // 10) % 2 == 0 else -18, 4)
        start = time.perf_counter()
        handlers.append(
            timed(
                lambda: canvas.mouseMoveEvent(
                    event(QEvent.Type.MouseMove, Qt.MouseButton.RightButton)
                )
            )
        )
        paints.append(timed(paint))
        combined.append((time.perf_counter() - start) * 1000)
        dispatches.append(timed(app.processEvents))
        if args.frame_interval_ms:
            time.sleep(args.frame_interval_ms / 1000)
    canvas.mouseReleaseEvent(event(QEvent.Type.MouseButtonRelease, Qt.MouseButton.NoButton))
    settle_start = time.perf_counter()
    final_ready = False
    while time.perf_counter() - settle_start < 20:
        pumps.append(timed(app.processEvents))
        keys = canvas._visible_overlay_tile_keys(canvas._paint_context())
        if (
            all(canvas_overlay_tile_cache.contains(key) for key in keys)
            and not draft_preview_cache._requests
        ):
            final_ready = True
            break
        time.sleep(0.001)
    stationary = timed(paint)
    result = dict(
        image_size=size,
        objects=args.objects,
        vertices_per_object=args.vertices,
        subtracts=args.subtracts,
        exact_cache_ready=ready,
        cache_errors=failures,
        first_visible_ui_ms=first,
        draft_first_ui_ms=draft_cold,
        draft_raster_ready_ms=draft_ready_ms,
        draft_preview=draft_hot,
        baseline_draft_preview=baseline,
        mouse_handler=summary(handlers),
        paint=summary(paints),
        input_to_qimage=summary(combined),
        event_loop_dispatch=summary(pumps),
        interaction_dispatch=summary(dispatches),
        stationary_exact_paint_ms=stationary,
        final_exact_cache_ready=final_ready,
        settle_ms=(time.perf_counter() - settle_start) * 1000,
        frame_interval_ms=args.frame_interval_ms,
        geometry_install_ms=installs,
        new_object_install_ms=additions,
        preview_frames=canvas._overlay_preview_frames,
        synchronous_missing_tile_fallbacks=canvas._overlay_sync_fallbacks,
        draft_path_builds=draft_preview_cache.path_builds,
        draft_raster_builds=draft_preview_cache.raster_builds,
    )
    canvas.clear_document()
    canvas.close()
    app.processEvents()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--image-size", type=int, default=2048, choices=(2048, 4096, 8192))
    parser.add_argument("--objects", type=int, default=50)
    parser.add_argument("--vertices", type=int, default=10000)
    parser.add_argument("--subtracts", type=int, default=3)
    parser.add_argument("--frames", type=int, default=90)
    parser.add_argument("--frame-interval-ms", type=float, default=16)
    parser.add_argument("--preview-baseline-ref")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {key: value for key, value in result.items() if key not in {"event_loop_dispatch"}},
            ensure_ascii=False,
            allow_nan=False,
        )
    )
