"""Read-only geometry audit: imports project code; never edits it.

Run from repository root via uv run --no-sync python -B PATH.
Add --with scikit-image==0.26.0 for a temporary dependency overlay.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import platform
import statistics
import sys
from time import perf_counter

import cv2
import numpy as np

sys.path.insert(0, str(Path.cwd() / "src"))
from fdm.geometry import Point, distance
import fdm.services.fiber_quick_geometry as g


def cases():
    small = np.zeros((120, 160), np.uint8)
    small[35:85, 50:110] = 1
    yield "small_rectangle", small
    horiz = np.zeros((768, 1024), np.uint8)
    horiz[350:410, 110:910] = 1
    yield "horizontal_60px", horiz
    diagonal = np.zeros((768, 1024), np.uint8)
    cv2.line(diagonal, (170, 180), (840, 600), 1, 40)
    yield "diagonal_40px", diagonal
    thick = np.zeros((1536, 2048), np.uint8)
    cv2.line(thick, (300, 320), (1700, 1200), 1, 120)
    yield "diagonal_120px", thick
    curved = np.zeros((768, 1024), np.uint8)
    pts = np.array([[120,540],[250,350],[430,230],[600,260],[750,420],[880,500]], np.int32)
    cv2.polylines(curved, [pts], False, 1, 48)
    yield "curved_48px", curved
    cross = np.zeros((220, 220), np.uint8)
    cv2.rectangle(cross, (90,20), (130,200), 1, -1)
    cv2.rectangle(cross, (20,90), (200,130), 1, -1)
    yield "cross_fixture", cross


parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--repeats", type=int, default=3)
args = parser.parse_args()

# Resolve optional imports before warm measurements. Cold import is reported separately.
start = perf_counter()
has_skimage = importlib.util.find_spec("skimage") is not None
if has_skimage:
    from skimage.morphology import skeletonize
cold_import_ms = (perf_counter() - start) * 1000
selected_backend = "skimage" if has_skimage else "opencv_ximgproc" if hasattr(cv2, "ximgproc") else "python_fallback"
versions = {}
for name in ("numpy", "opencv-python", "opencv-contrib-python", "scikit-image", "scipy"):
    try:
        versions[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        pass

stage_times = {}
skeleton_info = {}
def instrument(name):
    original = getattr(g, name)
    def wrapped(*a, **kw):
        start = perf_counter()
        try:
            result = original(*a, **kw)
            if name == "_compute_skeleton":
                skeleton_info.update(shape=list(result.shape), pixels=int(result.sum()), components=int(cv2.connectedComponents(result.astype(np.uint8), connectivity=8)[0] - 1))
            return result
        finally:
            stage_times[name] = stage_times.get(name, 0) + (perf_counter() - start) * 1000
    setattr(g, name, wrapped)

for name in ("_prepare_target_mask", "_trim_border_connected_region", "_crop_mask_roi", "_resize_mask_for_geometry", "_compute_skeleton", "_branch_points", "_end_points", "_border_forbidden_zone", "_select_candidate_points", "_estimate_tangent", "_measure_candidate_line"):
    instrument(name)

payload = dict(backend=selected_backend, versions=versions, platform=platform.platform(), python=sys.version.split()[0], cv2_threads=cv2.getNumThreads(), cold_optional_import_ms=cold_import_ms, repeats=args.repeats, timeout_ms=3000, scope="geometry only; existing mask + precomputed preview; no UI or segmentation", cases=[])
service = g.FiberQuickDiameterGeometryService()
for case_name, mask in cases():
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygon = [Point(float(x), float(y)) for x, y in max(contours, key=cv2.contourArea).reshape(-1, 2)]
    runs = []
    for repeat in range(args.repeats):
        stage_times.clear()
        skeleton_info.clear()
        start = perf_counter()
        result = None
        error = None
        try:
            result = service.measure_from_mask(mask.astype(bool), preview_polygon_points=polygon, cancel_check=lambda: False)
        except Exception as exc:
            error = str(exc)
        elapsed = (perf_counter() - start) * 1000
        runs.append(dict(elapsed_ms=elapsed, error=error, width_px=distance(result.line_px.start,result.line_px.end) if result and result.line_px else None, stages_ms=dict(stage_times), skeleton=dict(skeleton_info), debug=result.debug_payload if result else None))
    row = dict(name=case_name, shape=list(mask.shape), runs=runs, median_ms=statistics.median(r["elapsed_ms"] for r in runs), success_count=sum(r["error"] is None for r in runs))
    payload["cases"].append(row)
    print(json.dumps(dict(backend=selected_backend, name=case_name, median_ms=round(row["median_ms"],3), success_count=row["success_count"], width_px=runs[-1]["width_px"], skeleton=runs[-1]["skeleton"], stages_ms={k:round(v,3) for k,v in runs[-1]["stages_ms"].items()}, error=runs[-1]["error"]),ensure_ascii=False),flush=True)
args.output.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\n")
