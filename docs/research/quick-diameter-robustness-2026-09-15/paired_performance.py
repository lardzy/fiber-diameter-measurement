"""Compare both service implementations in one warmed process and alternating order.

The baseline is the local first-pass service snapshot, before this robustness
change. Its checksum is recorded so it cannot be confused with the old Python
skeleton implementation or a different checkout.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import statistics
import sys
from time import perf_counter

import cv2
import numpy as np

from fdm.geometry import Point, distance
import fdm.services.fiber_quick_geometry as current


def cases():
    small = np.zeros((120, 160), np.uint8)
    small[35:85, 50:110] = 1
    yield "small_rectangle", small
    horizontal = np.zeros((768, 1024), np.uint8)
    horizontal[350:410, 110:910] = 1
    yield "horizontal_60px", horizontal
    diagonal = np.zeros((768, 1024), np.uint8)
    cv2.line(diagonal, (170, 180), (840, 600), 1, 40)
    yield "diagonal_40px", diagonal
    thick = np.zeros((1536, 2048), np.uint8)
    cv2.line(thick, (300, 320), (1700, 1200), 1, 120)
    yield "diagonal_120px", thick
    curved = np.zeros((768, 1024), np.uint8)
    points = np.array([[120, 540], [250, 350], [430, 230], [600, 260], [750, 420], [880, 500]], np.int32)
    cv2.polylines(curved, [points], False, 1, 48)
    yield "curved_48px", curved
    cross = np.zeros((220, 220), np.uint8)
    cv2.rectangle(cross, (90, 20), (130, 200), 1, -1)
    cv2.rectangle(cross, (20, 90), (200, 130), 1, -1)
    yield "cross_fixture", cross


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-service", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("fdm_quick_geometry_baseline", args.baseline_service)
    baseline = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = baseline
    spec.loader.exec_module(baseline)
    modules = {"before": baseline, "after": current}
    services = {}
    for label, module in modules.items():
        module.prepare_fiber_quick_geometry_backend()
        services[label] = module.FiberQuickDiameterGeometryService()
    rows = []
    for name, raster in cases():
        mask = raster.astype(bool)
        contours, _ = cv2.findContours(raster, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polygon = [Point(float(x), float(y)) for x, y in max(contours, key=cv2.contourArea).reshape(-1, 2)]
        runs = {label: [] for label in services}
        for repeat in range(args.repeats + 2):
            order = ("before", "after") if repeat % 2 == 0 else ("after", "before")
            for label in order:
                start = perf_counter()
                result = services[label].measure_from_mask(mask, preview_polygon_points=polygon, cancel_check=lambda: False)
                elapsed = (perf_counter() - start) * 1000
                if repeat >= 2:
                    runs[label].append({"elapsed_ms": elapsed, "width_px": distance(result.line_px.start, result.line_px.end)})
        rows.append({"name": name, "runs": runs, "median_ms": {label: statistics.median(run["elapsed_ms"] for run in values) for label, values in runs.items()}})
    payload = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "scope": "warmed geometry only, same masks and precomputed contours; no segmentation or UI",
        "repeats": args.repeats,
        "warmups_per_service_per_case": 2,
        "baseline_service": str(args.baseline_service),
        "source_sha256": {label: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() for label, module in modules.items()},
        "cases": rows,
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps([{ "name": row["name"], **row["median_ms"] } for row in rows], allow_nan=False))
