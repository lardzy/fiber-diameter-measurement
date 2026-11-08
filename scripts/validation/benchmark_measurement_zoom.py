"""Benchmark ordinary measurement zoom through the canvas's public inputs.

Captures Qt-painted device pixels, not the display compositor. Run each source
revision with the same interpreter, window size, DPI and camera-free fixture.
Images and JSON are written only to the requested evidence directory.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import psutil
from PySide6.QtCore import QEvent, QPoint, QPointF, QSize, Qt, qVersion
from PySide6.QtGui import QImage, QMouseEvent, QWheelEvent
from PySide6.QtWidgets import QApplication

from fdm.geometry import Line, Point
from fdm.measurement_annotation import MeasurementAnnotationLayout
from fdm.models import ImageDocument, Measurement
from fdm.settings import AppSettings, MeasurementLabelStyleSettings
from fdm.ui.canvas import DocumentCanvas
from fdm.ui.canvas_overlay_cache import canvas_overlay_tile_cache
from fdm.ui.rendering import resolve_measurement_annotation


def now_ms():
    return time.perf_counter() * 1000


def distribution(values):
    return {
        "samples": len(values),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "max_ms": max(values),
    }


def working_set_mb():
    process = psutil.Process()
    total = 0
    try:
        children = process.children(recursive=True)
    except (psutil.AccessDenied, PermissionError):
        children = []
    for child in [process, *children]:
        try:
            total += child.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return total / 1024**2


def send_mouse(canvas, kind, point, button, held):
    QApplication.sendEvent(
        canvas,
        QMouseEvent(
            kind,
            point,
            canvas.mapToGlobal(point),
            button,
            held,
            Qt.KeyboardModifier.NoModifier,
        ),
    )


def wheel(canvas, ticks):
    point = QPointF(canvas.width() / 2, canvas.height() / 2)
    QApplication.sendEvent(
        canvas,
        QWheelEvent(
            point,
            canvas.mapToGlobal(point),
            QPoint(),
            QPoint(0, 120 * ticks),
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
            Qt.ScrollPhase.ScrollUpdate,
            False,
        ),
    )


def make_scene(stress=False):
    width, height = (4096, 4096) if stress else (1024, 768)
    anchor = Point(width / 2, height / 2)
    document = ImageDocument(
        id="stress" if stress else "lines",
        path="zoom-fixture.png",
        image_size=(width, height),
    )
    if stress:
        angles = np.arange(10000) * (2 * np.pi / 10000)
        radius = 34 * (1 + np.sin(np.arange(10000) * 1.3) * 0.035)
        xx, yy = radius * np.cos(angles), radius * np.sin(angles)
        for i in range(50):
            cx, cy = anchor.x - 405 + i % 10 * 90, anchor.y - 250 + (i // 10) * 85
            outline = [Point(float(x + cx), float(y + cy)) for x, y in zip(xx, yy)]
            document.measurements.append(
                Measurement(
                    f"a{i}",
                    document.id,
                    None,
                    "magic_segment",
                    measurement_kind="area",
                    polygon_px=outline,
                    area_rings_px=[outline],
                    exact_area_px=3600.0,
                )
            )
    for i in range(64):
        y = anchor.y if i == 63 else anchor.y - 345 + i % 18 * 37
        if i < 63 and abs(y - anchor.y) < 72:
            y += 100 if y >= anchor.y else -100
        x = anchor.x if i == 63 else anchor.x - 390 + i // 18 * 205
        half = 135 if i == 63 else 66
        document.measurements.append(
            Measurement(
                f"l{i}",
                document.id,
                None,
                "manual",
                line_px=Line(Point(x - half, y), Point(x + half, y)),
                annotation_layout=(
                    MeasurementAnnotationLayout.ordinary_near().with_manual_offset(
                        Point(0.0, 68.0)
                    )
                    if i == 63
                    else MeasurementAnnotationLayout.ordinary_near()
                ),
            )
        )
    document.initialize_runtime_state()
    source = QImage(width, height, QImage.Format.Format_RGB32)
    source.fill(Qt.GlobalColor.white)
    return document, source, anchor


class ViewProbe:
    def __init__(self, app, out, stress=False, labels=False):
        self.app, self.out, self.stress = app, out, stress
        self.doc, image, self.anchor = make_scene(stress)
        self.canvas = DocumentCanvas()
        self.canvas.resize(820, 576)
        self.canvas.set_settings(
            AppSettings(
                length_measurement_label_style=MeasurementLabelStyleSettings(
                    enabled=labels
                ),
                area_measurement_label_style=MeasurementLabelStyleSettings(
                    enabled=False
                ),
            )
        )
        self.canvas.set_document(self.doc, image)
        self.canvas.set_tool_mode("select")
        self.canvas.show()
        self.app.processEvents()
        self.canvas.set_view_zoom(1)
        self.canvas.center_on_image_point(self.anchor)
        self.dpr = self.canvas.devicePixelRatioF()
        self.labelled = labels
        dimensions = QSize(
            math.ceil(self.canvas.width() * self.dpr),
            math.ceil(self.canvas.height() * self.dpr),
        )
        self.surface = QImage(dimensions, QImage.Format.Format_ARGB32_Premultiplied)
        self.surface.setDevicePixelRatio(self.dpr)
        self.frames = []
        self.peak = working_set_mb()
        self.last_rss_probe = now_ms()
        self.warm()
        self.reference_width = self.stroke()

    def draw(self):
        self.canvas.render(self.surface)

    def warm(self):
        deadline = now_ms() + (18000 if self.stress else 8000)
        while now_ms() < deadline:
            self.app.processEvents()
            self.draw()
            keys = self.canvas._visible_overlay_tile_keys(self.canvas._paint_context())
            if keys and all(canvas_overlay_tile_cache.contains(key) for key in keys):
                return True
            time.sleep(0.004)
        return False

    def stroke(self):
        point = self.canvas.image_to_widget(self.anchor)
        # Thickness of the connected foreground run through this line; other
        # parallel objects or letters elsewhere in the image do not enter it.
        # Sample away from the label at the centre of the line.
        logical_x = (
            point.x() - min(66.0, self.canvas.view_zoom() * 90.0)
            if self.labelled
            else point.x()
        )
        x, y = round(logical_x * self.dpr), round(point.y() * self.dpr)
        if not (0 <= x < self.surface.width() and 0 <= y < self.surface.height()):
            return 0
        data = np.frombuffer(self.surface.constBits(), np.uint8).reshape(
            self.surface.height(), self.surface.bytesPerLine()
        )
        foreground = data[:, x * 4 + 2] < 185
        if not foreground[y]:
            return 0
        low, high = y, y + 1
        while low > 0 and foreground[low - 1]:
            low -= 1
        while high < self.surface.height() and foreground[high]:
            high += 1
        return high - low

    def cursor_zoom(self, target):
        if target == "fit":
            self.canvas.fit_to_view()
            w, h = self.doc.image_size
            return min((self.canvas.width() - 40) / w, (self.canvas.height() - 40) / h)
        self.canvas.set_view_zoom(target)
        return target

    def record(self, target, start, entry, presentation_time=None):
        self.draw()
        timestamp = now_ms()
        width = self.stroke()
        row = {
            "reason": entry,
            "target": target,
            "presented": self.canvas.view_zoom(),
            "elapsed_ms": timestamp - start,
            "stroke_px": width,
            "sharp": 0 < width and abs(width - self.reference_width) <= 1,
        }
        if presentation_time is not None:
            row["trace_ms"] = timestamp - presentation_time
        if self.stress and timestamp - self.last_rss_probe >= 100.0:
            self.peak = max(self.peak, working_set_mb())
            self.last_rss_probe = timestamp
        self.frames.append(row)
        return row

    def single(self, command, number):
        start = now_ms()
        target = self.cursor_zoom(command)
        row = self.record(target, start, command)
        first = row.copy()
        selected = command in (4.0, 8.0, 16.0) and number < 4
        if selected:
            self.surface.save(str(self.out / f"zoom-{command:g}-first.png"))
        while (
            not row["sharp"] or abs(row["presented"] - target) > 1e-6
        ) and now_ms() - start < 4000:
            self.app.processEvents()
            row = self.record(target, start, command)
            if not row["sharp"]:
                time.sleep(0.003)
        if selected:
            self.surface.save(str(self.out / f"zoom-{command:g}-last.png"))
        return {
            "command": command,
            "target": target,
            "first": first,
            "clear_ms": row["elapsed_ms"],
            "reached": row["sharp"] and abs(row["presented"] - target) < 1e-6,
        }

    def burst(self):
        self.canvas.set_view_zoom(1.0)
        self.canvas.center_on_image_point(self.anchor)
        self.warm()
        self.peak = max(self.peak, working_set_mb())
        sequence = [1] * 6 + [-1] * 3 + [1] * 5
        target = 1.0
        start = now_ms()
        samples = []
        for tick in sequence:
            target *= 1.15**tick
            sent = now_ms()
            wheel(self.canvas, tick)
            samples.append(self.record(target, sent, "wheel", start))
            self.app.processEvents()
            remaining = 0.016 - (now_ms() - sent) / 1000
            if remaining > 0:
                time.sleep(remaining)
        stop = now_ms()
        row = samples[-1]
        while (
            not row["sharp"] or abs(row["presented"] - target) > 1e-6
        ) and now_ms() - stop < 6000:
            self.app.processEvents()
            row = self.record(target, stop, "wheel-settle", start)
            samples.append(row)
            time.sleep(0.002)
        self.peak = max(self.peak, working_set_mb())
        changes = [0.0]
        previous = 1.0
        for frame in samples:
            if abs(previous - frame["presented"]) > 1e-6:
                changes.append(frame["trace_ms"])
                previous = frame["presented"]
        changes.append(samples[-1]["trace_ms"])
        return {
            "ticks": len(sequence),
            "sampled_presented_changes": len(changes) - 2,
            "skipped_intermediate": max(0, len(sequence) - len(changes) + 2),
            "longest_unchanged_ms": max(np.diff(changes)),
            "settle_ms": max(0.0, now_ms() - stop),
            "reached": row["sharp"] and abs(row["presented"] - target) < 1e-6,
            "peak_sampled_tree_rss_mib": self.peak,
        }

    def moves(self, drag):
        canvas = self.canvas
        if drag:
            canvas.set_settings(AppSettings())
            canvas.set_tool_mode("select")
            self.warm()
            target = self.doc.measurements[-1]
            resolved = resolve_measurement_annotation(
                target, self.doc, canvas._settings, canvas.image_to_widget
            )
            point = resolved.text_rect.center()
            button = Qt.MouseButton.LeftButton
        else:
            point = QPointF(canvas.width() / 2, canvas.height() / 2)
            button = Qt.MouseButton.RightButton
        send_mouse(canvas, QEvent.Type.MouseButtonPress, point, button, button)
        if drag:
            assert canvas._annotation_drag is not None
        times = []
        for i in range(88):
            position = point + QPointF(
                ((i % 16) if i % 32 < 16 else 15 - (i % 16)) * 2, 0
            )
            start = now_ms()
            send_mouse(
                canvas, QEvent.Type.MouseMove, position, Qt.MouseButton.NoButton, button
            )
            self.draw()
            elapsed = now_ms() - start
            if i >= 24:
                times.append(elapsed)
            self.app.processEvents()
            time.sleep(0.003)
        send_mouse(
            canvas,
            QEvent.Type.MouseButtonRelease,
            position,
            button,
            Qt.MouseButton.NoButton,
        )
        self.app.processEvents()
        return {**distribution(times), "frame_ms": times}

    def close(self):
        self.canvas.clear_document()
        self.canvas.close()
        self.canvas.deleteLater()
        self.app.processEvents()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=True, parents=True)
    app = QApplication.instance() or QApplication([])
    if os.environ.get("QT_QPA_PLATFORM") == "offscreen":
        os.environ.pop("QT_QPA_PLATFORM")
    result = {"platform": app.platformName(), "qt": qVersion(), "rounds": []}
    for repeat in range(args.repeats):
        path = args.output / f"round-{repeat + 1}"
        path.mkdir(exist_ok=True)
        probe = ViewProbe(app, path)
        steps = [4.0, 8.0, 16.0, "fit", 1.0, 3.5, 7.25, 1.0, 4.0, 8.0, 16.0, "fit"]
        measured = [probe.single(value, i) for i, value in enumerate(steps)]
        probe.canvas.set_view_zoom(1)
        probe.canvas.center_on_image_point(probe.anchor)
        probe.warm()
        pan = probe.moves(False)
        drag = probe.moves(True)
        ordinary_frames = probe.frames
        reference = probe.reference_width
        dpr = probe.dpr
        probe.close()
        labels = ViewProbe(app, path, labels=True)
        labelled = [labels.single(value, i + 8) for i, value in enumerate(steps)]
        labelled_frames = labels.frames
        labels.close()
        stress = ViewProbe(app, path, stress=True)
        movement = stress.burst()
        stress_frames = stress.frames
        stress.close()
        summary = {
            "zoom": [m["clear_ms"] for m in measured if m["reached"]],
            "unreached": sum(not m["reached"] for m in measured),
            "first_strokes": [m["first"]["stroke_px"] for m in measured],
            "reference_width": reference,
            "labelled_zoom": [m["clear_ms"] for m in labelled if m["reached"]],
            "labelled_unreached": sum(not m["reached"] for m in labelled),
            "pan": pan,
            "drag": drag,
            "stress": movement,
            "dpr": dpr,
        }
        print(
            f"Round {repeat + 1}: zoom {distribution(summary['zoom'])}, pan {pan['p95_ms']:.2f} ms, drag {drag['p95_ms']:.2f} ms, stress {movement}",
            flush=True,
        )
        (path / "frames.json").write_text(
            json.dumps(
                {
                    "commands": measured,
                    "ordinary": ordinary_frames,
                    "labelled_commands": labelled,
                    "labelled": labelled_frames,
                    "stress": stress_frames,
                }
            ),
            encoding="utf8",
        )
        result["rounds"].append(summary)
        (args.output / "metrics.json").write_text(
            json.dumps(result, indent=2), encoding="utf8"
        )


if __name__ == "__main__":
    main()
