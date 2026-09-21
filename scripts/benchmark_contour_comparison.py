"""Synthetic acceptance aid, not a real garment/camera accuracy certificate.

uv run --no-sync python scripts/benchmark_contour_comparison.py --output /tmp/contour-qa --screenshots
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import time

import cv2
import numpy as np

from fdm.services.contour_comparison import (
    ContourAxis, compare_contours, frame_from_rgba, save_comparison,
)


def synthetic_photo(height, *, after=False):
    width = round(height * .7)
    image = np.full((height, width, 4), (236, 233, 225, 255), np.uint8)
    # Entire leg gap remains exterior. A known transverse/longitudinal shrink
    # is anchored at the SAME origin, never fitted to the after bounding box.
    points = np.array([
        (.20, .10), (.80, .10), (.83, .22), (.77, .52), (.72, .92),
        (.54, .92), (.53, .57), (.50, .35), (.47, .57), (.46, .92),
        (.28, .92), (.23, .52), (.17, .22),
    ])
    if after:
        points[:, 0] = .5 + (points[:, 0] - .5) * .97
        points[:, 1] = .1 + (points[:, 1] - .1) * .98
    vertices = np.rint(points * [width, height]).astype(np.int32)
    cv2.fillPoly(image, [vertices], (44, 78, 116, 255))
    cv2.polylines(image, [vertices], True, (35, 66, 95, 255), max(1, height // 700))
    # A permanent table reference is outside the specimen; initial segmentation
    # may find it as a small component, which must not be called fabric.
    cv2.line(image, (width//12, height//10), (width//12, height*4//10), (95, 100, 98, 255), max(2, height//350))
    for i in range(7):
        y = height//10 + i * height//20
        cv2.line(image, (width//12-7, y), (width//12+7, y), (95, 100, 98, 255), max(1, height//1000))
    return image


def run(output, screenshots, heights=(2000, 4000, 6000)):
    output.mkdir(parents=True, exist_ok=True)
    metrics = []
    for height in heights:
        frames, segment_ms = [], []
        for after in (False, True):
            image = synthetic_photo(height, after=after)
            t = time.perf_counter()
            frame = frame_from_rgba(image, "合成验证 · " + ("处理后" if after else "处理前"), mm_per_pixel=1200 / height)
            segment_ms.append((time.perf_counter() - t) * 1000)
            frames.append(replace(frame, axis=ContourAxis((image.shape[1]/2, height*.1), (image.shape[1]/2, height*.9)), axis_confirmed=True))
        times = []
        for _ in range(5):
            start = time.perf_counter()
            result = compare_contours(*frames, 5)
            times.append((time.perf_counter() - start) * 1000)
        metrics.append({"size": [frames[0].mask.shape[1], height], "segmentation_ms": segment_ms, "profile_p50_ms": float(np.median(times)), "profile_max_ms": max(times), "rows": len(result.sections), "length_change_mm": result.summary["纵向总长变化"]})
        if height == heights[0]:
            demo_frames = frames
        print(json.dumps(metrics[-1], ensure_ascii=False, allow_nan=False), flush=True)
    save_comparison(output / "合成样本-前后轮廓.fdmcompare", *demo_frames, 5)
    (output / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    if screenshots:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
        from PySide6.QtTest import QTest
        from fdm.ui.theme import apply_application_theme
        from fdm.ui.contour_comparison_dialog import ContourComparisonDialog
        from fdm.ui.contour_comparison_canvas import BEFORE_COLOR, AFTER_COLOR, prepare_presentation
        app = QApplication.instance() or QApplication([])
        d = ContourComparisonDialog()
        presentations = [prepare_presentation(f, color) for f, color in zip(demo_frames, (BEFORE_COLOR, AFTER_COLOR))]
        d.show()
        app.processEvents()
        d._install(demo_frames, presentations, reset_views=(0, 1))
        d.step.setValue(5)
        deadline = time.monotonic() + 30
        while (d._tasks.busy or d._timer.isActive()) and time.monotonic() < deadline:
            app.processEvents()
            QTest.qWait(5)
        assert d.result is not None, d.status.text()
        for theme in ("dark", "light"):
            apply_application_theme(app, theme)
            for size in ((1280, 880), (860, 640)):
                d.resize(*size)
                for tab in (0, 1):
                    d.tabs.setCurrentIndex(tab)
                    app.processEvents()
                    d.grab().save(str(output / f"{theme}-{size[0]}-tab{tab}.png"))
        d.dirty = False
        d.close()
        app.processEvents()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("/tmp/fdm-contour-qa"))
    parser.add_argument("--screenshots", action="store_true")
    parser.add_argument("--heights", type=int, nargs="+", default=[2000, 4000, 6000])
    args = parser.parse_args()
    run(args.output, args.screenshots, args.heights)
