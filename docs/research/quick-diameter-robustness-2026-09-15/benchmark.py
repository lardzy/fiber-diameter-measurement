"""Deterministic masks for evaluating false rejections and width errors."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
from time import perf_counter

import cv2
import numpy as np

from fdm.geometry import distance
from fdm.services.fiber_quick_geometry import FIBER_QUICK_GEOMETRY_REVISION, FiberQuickDiameterGeometryService, prepare_fiber_quick_geometry_backend


def rectangle(shape, center, length, width, angle):
    ys, xs = np.mgrid[:shape[0], :shape[1]]
    theta = np.deg2rad(angle)
    along = (xs - center[0]) * np.cos(theta) + (ys - center[1]) * np.sin(theta)
    across = -(xs - center[0]) * np.sin(theta) + (ys - center[1]) * np.cos(theta)
    return (np.abs(along) < length / 2) & (np.abs(across) < width / 2)


def cases():
    for width in (8, 16, 40):
        for ratio in (1.2, 1.5, 2.0):
            for angle in (0, 15, 30, 45, 75):
                yield f"short_w{width}_r{ratio}_a{angle}", rectangle((256, 256), (128, 128), width * ratio, width, angle), width, angle
    for angle in (0, 15, 45, 75, 90):
        yield f"thin_w4_a{angle}", rectangle((256, 256), (128, 128), 160, 4, angle), 4, angle
    for margin in (2, 4, 6, 10):
        yield f"near_border_m{margin}", rectangle((240, 320), (160, margin + 20), 220, 40, 0), 40, 0
    yield "frame_spanning", rectangle((256, 320), (160, 128), 350, 24, 0), 24, 0
    yield "large_diagonal", rectangle((512, 512), (256, 256), 640, 24, 45), 24, 45
    yield "long_thin", rectangle((160, 4096), (2048, 80), 3600, 6, 0), 6, 0
    for width in (16, 40):
        for ratio in (3.0, 4.0, 5.0):
            a = rectangle((320, 320), (160, 160), width * ratio, width, 25)
            b = rectangle((320, 320), (160, 160), width * ratio, width, 115)
            yield f"cross_w{width}_r{ratio}", a | b, width, None


def cross_stress_cases():
    for width in (16, 40):
        for ratio in (3, 4, 5):
            for angle in (0, 15, 25, 45, 75):
                a = rectangle((320, 320), (160, 160), width * ratio, width, angle)
                b = rectangle((320, 320), (160, 160), width * ratio, width, angle + 90)
                yield f"cross_w{width}_r{ratio}_a{angle}", a | b, width, (angle, angle + 90)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cross-stress", action="store_true")
    args = parser.parse_args()
    prepare_fiber_quick_geometry_backend()
    service = FiberQuickDiameterGeometryService()
    rows = []
    for name, mask, expected, angle in (cross_stress_cases() if args.cross_stress else cases()):
        started = perf_counter()
        try:
            result = service.measure_from_mask(mask)
            line = result.line_px
            width = distance(line.start, line.end)
            direction_error = None
            if angle is not None:
                axis = np.array([line.end.x - line.start.x, line.end.y - line.start.y]) / width
                angles = angle if isinstance(angle, tuple) else (angle,)
                direction_error = min(
                    float(np.degrees(np.arcsin(np.clip(abs(axis @ np.array([np.cos(np.deg2rad(a)), np.sin(np.deg2rad(a))])), 0, 1))))
                    for a in angles
                )
            row = dict(name=name, expected_width=expected, width_px=width, width_error=abs(width - expected), direction_error_deg=direction_error, debug=result.debug_payload, error=None)
        except RuntimeError as exc:
            row = dict(name=name, expected_width=expected, error=str(exc), code=getattr(exc, "code", None), debug=getattr(exc, "debug_payload", {}))
        row["elapsed_ms"] = (perf_counter() - started) * 1000
        rows.append(row)
    payload = dict(platform=platform.platform(), geometry_revision=FIBER_QUICK_GEOMETRY_REVISION, cases=rows, successful=sum(row["error"] is None for row in rows), total=len(rows))
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"successful": payload["successful"], "total": payload["total"], "failures": [r["name"] for r in rows if r["error"]], "incorrect_widths": [(r["name"], round(r["width_error"], 2)) for r in rows if not r["error"] and r["width_error"] > 2.1]}, ensure_ascii=False, allow_nan=False))
