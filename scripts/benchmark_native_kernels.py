"""Paired native/Python measurements with fixed inputs and semantic checks.

Run from the checkout with its installed environment::

    .venv/bin/python scripts/benchmark_native_kernels.py --output .tmp/native.json

These synthetic geometry timings exclude segmentation, worker queueing, Qt
installation and display presentation. Neither the speed nor the known-width
fixtures establish real-image metrology or Windows package acceptance.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass, fields, is_dataclass
import fnmatch
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Callable

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmark_contour_comparison import synthetic_photo
from fdm.geometry import Point
from fdm.services import contour_comparison as contour
from fdm.services import fiber_quick_geometry as quick


@dataclass(frozen=True)
class CaseSpec:
    name: str
    category: str
    variant: str
    size: int = 0
    width: int = 0
    ratio: int = 0
    angle: int = 0

    @property
    def heavy_quick(self) -> bool:
        return self.category == "quick" and self.width == 40 and self.angle in (15, 25, 75)


@dataclass
class PreparedCase:
    stages: dict[str, Callable[[], object]]
    input_sha256: str
    details: dict[str, object]


def case_catalog() -> list[CaseSpec]:
    result = [CaseSpec("contour_128_normal", "contour", "normal", 128)]
    for height in (2000, 4000, 6000):
        for variant in ("normal", "rotation17", "holes", "multicomponent"):
            result.append(CaseSpec(f"contour_{height}_{variant}", "contour", variant, height))
    for name in ("small_rectangle", "horizontal_60px", "diagonal_40px", "diagonal_120px", "curved_48px", "cross_fixture"):
        result.append(CaseSpec(f"quick_{name}", "quick", name))
    for width in (16, 40):
        for ratio in (3, 4, 5):
            for angle in (0, 15, 25, 45, 75):
                result.append(CaseSpec(f"quick_cross_w{width}_r{ratio}_a{angle}", "quick", "cross_stress", width=width, ratio=ratio, angle=angle))
    return result


@contextmanager
def backend(name: str):
    previous = os.environ.get("FDM_NATIVE_BACKEND")
    os.environ["FDM_NATIVE_BACKEND"] = name
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("FDM_NATIVE_BACKEND", None)
        else:
            os.environ["FDM_NATIVE_BACKEND"] = previous


def digest_inputs(arrays: list[np.ndarray], parameters: object) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(parameters, sort_keys=True, allow_nan=False).encode())
    for array in arrays:
        digest.update(str(array.shape).encode())
        digest.update(array.dtype.str.encode())
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def prepare_contour(spec: CaseSpec) -> PreparedCase:
    frames = []
    for after in (False, True):
        image = synthetic_photo(spec.size, after=after)
        # Ground-truth cloth colors; table reference/background are excluded.
        # Segmentation is deliberately outside this geometry benchmark.
        mask = image[:, :, 0] < 60
        height, width = mask.shape
        if spec.variant == "holes":
            mask[round(height * .20):round(height * .25), round(width * .30):round(width * .40)] = False
            mask[round(height * .60):round(height * .68), round(width * .63):round(width * .69)] = False
        elif spec.variant == "multicomponent":
            mask[round(height * .44):round(height * .46), :] = False
            mask[round(height * .05):round(height * .08), round(width * .88):round(width * .92)] = True
        origin = (width / 2, height * .1)
        theta = math.radians(17 if spec.variant == "rotation17" else 0)
        direction = (origin[0] + height * .8 * math.sin(theta), origin[1] + height * .8 * math.cos(theta))
        frames.append(contour.ContourFrame("synthetic-after" if after else "synthetic-before", image, mask, contour.ContourAxis(origin, direction), 1200 / height))
    step = 5.0
    with backend("python"):
        worlds = [frame.axis.to_world(contour.boundary_segments(frame.mask), frame.scale) for frame in frames]
    low = min(float(world[:, :, 1].min()) for world in worlds)
    high = max(float(world[:, :, 1].max()) for world in worlds)
    start = math.ceil(low / step - .5 - 1e-10) * step + step / 2
    heights = start + np.arange(max(1, math.ceil((high - start) / step))) * step
    if len(heights) == 1 and heights[0] >= high:
        heights[0] = (low + high) / 2
    parameters = {"step": step, "axes": [(frame.axis.origin, frame.axis.direction_point) for frame in frames], "scale": frames[0].scale}
    return PreparedCase(
        stages={
            "complete": lambda: contour.compare_contours(*frames, step),
            "boundary": lambda: tuple(contour.boundary_segments(frame.mask) for frame in frames),
            "sections": lambda: tuple(contour._sections(world, heights, step, None) for world in worlds),
        },
        input_sha256=digest_inputs([frame.mask for frame in frames], parameters),
        details={**parameters, "shape": list(frames[0].mask.shape), "sample_rows": len(heights), "edges": [len(world) for world in worlds], "provenance": "scripts/benchmark_contour_comparison.py synthetic_photo; RGB cloth mask; segmentation excluded"},
    )


def quick_mask(spec: CaseSpec) -> np.ndarray:
    """Fixtures reproduced from the 2026-09-15 research scripts, no runtime import.

    Standard six: paired_performance.py/cases. Cross grid: benchmark.py/
    cross_stress_cases, under quick-diameter-robustness-2026-09-15.
    """
    if spec.variant == "cross_stress":
        ys, xs = np.mgrid[:320, :320]

        def rectangle(angle):
            theta = np.deg2rad(angle)
            along = (xs - 160) * np.cos(theta) + (ys - 160) * np.sin(theta)
            across = -(xs - 160) * np.sin(theta) + (ys - 160) * np.cos(theta)
            return (abs(along) < spec.width * spec.ratio / 2) & (abs(across) < spec.width / 2)

        # Keep the original two angle evaluations: replacing angle+90 by an
        # algebraic swap changes pixels exactly on floating-point boundaries.
        return rectangle(spec.angle) | rectangle(spec.angle + 90)
    shape = {"small_rectangle": (120, 160), "diagonal_120px": (1536, 2048), "cross_fixture": (220, 220)}.get(spec.variant, (768, 1024))
    raster = np.zeros(shape, np.uint8)
    if spec.variant == "small_rectangle":
        raster[35:85, 50:110] = 1
    elif spec.variant == "horizontal_60px":
        raster[350:410, 110:910] = 1
    elif spec.variant == "diagonal_40px":
        cv2.line(raster, (170, 180), (840, 600), 1, 40)
    elif spec.variant == "diagonal_120px":
        cv2.line(raster, (300, 320), (1700, 1200), 1, 120)
    elif spec.variant == "curved_48px":
        points = np.array([[120, 540], [250, 350], [430, 230], [600, 260], [750, 420], [880, 500]], np.int32)
        cv2.polylines(raster, [points], False, 1, 48)
    elif spec.variant == "cross_fixture":
        cv2.rectangle(raster, (90, 20), (130, 200), 1, -1)
        cv2.rectangle(raster, (20, 90), (200, 130), 1, -1)
    else:
        raise ValueError(f"Unknown fixture: {spec.variant}")
    return raster.astype(bool)


def invoke(function: Callable[[], object]) -> dict[str, object]:
    try:
        return {"outcome": "ok", "result": function()}
    except quick.FiberQuickGeometryError as exc:
        return {"outcome": "rejected", "error_type": type(exc).__name__, "code": exc.code, "message": str(exc), "debug_payload": exc.debug_payload}


def prepare_quick(spec: CaseSpec) -> PreparedCase:
    mask = quick_mask(spec)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygon = [Point(float(x), float(y)) for x, y in max(contours, key=cv2.contourArea).reshape(-1, 2)]
    service = quick.FiberQuickDiameterGeometryService()
    complete = lambda: service.measure_from_mask(mask, preview_polygon_points=polygon, cancel_check=lambda: False)
    captured = []
    original = quick._prune_short_spurs

    def capture(skeleton, distance_map, **kwargs):
        captured.append((skeleton.copy(), distance_map.copy()))
        return original(skeleton, distance_map, **kwargs)

    # Capture actual production intermediates once, outside all timing. The
    # benchmark is serial and restores the module even if preparation fails.
    try:
        quick._prune_short_spurs = capture
        with backend("python"):
            initial = invoke(complete)
    finally:
        quick._prune_short_spurs = original
    if len(captured) != 1:
        raise RuntimeError(f"Expected one production pruning call for {spec.name}, got {len(captured)}")
    skeleton, distance_map = captured[0]
    return PreparedCase(
        stages={"complete": complete, "pruner": lambda: quick._prune_short_spurs(skeleton, distance_map, cancel_check=lambda: False)},
        input_sha256=digest_inputs([mask, skeleton, distance_map], {"preview_polygon": [(point.x, point.y) for point in polygon]}),
        details={"shape": list(mask.shape), "working_shape": list(skeleton.shape), "skeleton_pixels": int(skeleton.sum()), "baseline_outcome": initial["outcome"], "baseline_failure_code": initial.get("code"), "heavy_group": spec.heavy_quick, "provenance": "2026-09-15 quick-diameter-robustness paired_performance.py/cases and benchmark.py/cross_stress_cases, copied fixture builders"},
    )


def prepare_case(spec: CaseSpec) -> PreparedCase:
    return prepare_contour(spec) if spec.category == "contour" else prepare_quick(spec)


def semantic_view(value):
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: semantic_view(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, dict):
        # Wall times and implementation identity are diagnostics, not geometry.
        return {key: semantic_view(item) for key, item in value.items() if not key.endswith("_ms") and key not in {"native_backend", "native_build_info", "native_kernel_backend"}}
    if isinstance(value, (tuple, list)):
        return tuple(semantic_view(item) for item in value)
    return value


def assert_equivalent(first, second, path="result") -> None:
    if isinstance(first, np.ndarray) or isinstance(second, np.ndarray):
        if not isinstance(first, np.ndarray) or not isinstance(second, np.ndarray):
            raise AssertionError(f"{path}: array type differs")
        # Integer pixel boundaries and boolean pruning outputs must be exact.
        if first.dtype != second.dtype or not np.array_equal(first, second):
            raise AssertionError(f"{path}: array shape/dtype/values differ")
    elif isinstance(first, dict) and isinstance(second, dict):
        if first.keys() != second.keys():
            raise AssertionError(f"{path}: keys differ: {first.keys() ^ second.keys()}")
        for key in first:
            assert_equivalent(first[key], second[key], f"{path}.{key}")
    elif isinstance(first, (list, tuple)) and isinstance(second, (list, tuple)):
        if len(first) != len(second):
            raise AssertionError(f"{path}: lengths differ")
        for index, (a, b) in enumerate(zip(first, second)):
            assert_equivalent(a, b, f"{path}[{index}]")
    elif isinstance(first, (float, np.floating)) and isinstance(second, (float, np.floating)):
        if not math.isclose(float(first), float(second), rel_tol=1e-12, abs_tol=1e-10):
            raise AssertionError(f"{path}: floats differ ({first!r}, {second!r})")
    elif first != second:
        raise AssertionError(f"{path}: values differ ({first!r}, {second!r})")


def stats(values: list[float]) -> dict[str, float]:
    return {"p50_ms": float(np.percentile(values, 50)), "p95_ms": float(np.percentile(values, 95)), "min_ms": min(values), "max_ms": max(values)}


def paired_stage(function: Callable[[], object], *, warmups: int, repeats: int) -> dict[str, object]:
    raw = []
    outcomes = set()
    for repeat in range(warmups + repeats):
        order = ("python", "native") if repeat % 2 == 0 else ("native", "python")
        results, timings = {}, {}
        for selected in order:
            with backend(selected):
                started = time.perf_counter_ns()
                result = invoke(function)
                timings[selected] = (time.perf_counter_ns() - started) / 1_000_000
            results[selected] = result
        assert_equivalent(semantic_view(results["python"]), semantic_view(results["native"]))
        outcomes.add(results["python"].get("code", results["python"]["outcome"]))
        if repeat >= warmups:
            raw.append({"pair": repeat - warmups, "order": list(order), "python_ms": timings["python"], "native_ms": timings["native"]})
    summaries = {selected: stats([row[f"{selected}_ms"] for row in raw]) for selected in ("python", "native")}
    return {"semantic_equal": True, "outcomes": sorted(outcomes), "samples": raw, "summary": summaries, "speedup": {key: summaries["python"][key] / summaries["native"][key] for key in ("p50_ms", "p95_ms")}}


def aggregate(rows: list[dict], stage: str) -> dict[str, object]:
    # One value is a whole fixed corpus pass, not a median of speedup ratios.
    length = len(rows[0]["stages"][stage]["samples"])
    samples = {selected: [sum(row["stages"][stage]["samples"][index][f"{selected}_ms"] for row in rows) for index in range(length)] for selected in ("python", "native")}
    summaries = {selected: stats(values) for selected, values in samples.items()}
    return {"cases": [row["name"] for row in rows], "aggregation": "sum matching formal pair indices across fixed cases", "summary": summaries, "speedup": {key: summaries["python"][key] / summaries["native"][key] for key in ("p50_ms", "p95_ms")}}


def acceptance(rows: list[dict], *, warmups: int, repeats: int) -> dict[str, object]:
    checks = []
    normal_names = {f"contour_{size}_normal" for size in (2000, 4000, 6000)}
    heavy_names = {spec.name for spec in case_catalog() if spec.heavy_quick}
    all_quick_names = {spec.name for spec in case_catalog() if spec.category == "quick"}

    def corpus_check(name, names, stage, minimum):
        subset = [row for row in rows if row["name"] in names]
        if {row["name"] for row in subset} != names:
            checks.append({"name": name, "status": "not_run", "missing_cases": sorted(names - {row["name"] for row in subset})})
            return
        result = aggregate(subset, stage)
        checks.append({"name": name, "status": "passed" if all(value >= minimum for value in result["speedup"].values()) else "failed", "minimum_speedup": minimum, **result})

    corpus_check("contour_complete_1.5x", normal_names, "complete", 1.5)
    corpus_check("quick_pruner_3x", all_quick_names, "pruner", 3.0)
    corpus_check("quick_heavy_complete_10pct", heavy_names, "complete", 1 / .9)
    required_names = {spec.name for spec in case_catalog()}
    for row in rows:
        summary = row["stages"]["complete"]["summary"]
        limits = {key: summary["python"][key] + max(.5, .05 * summary["python"][key]) for key in ("p50_ms", "p95_ms")}
        checks.append({"name": f"complete_no_regression:{row['name']}", "status": "passed" if all(summary["native"][key] <= limit for key, limit in limits.items()) else "failed", "native_limit_ms": limits})
    missing = sorted(required_names - {row["name"] for row in rows})
    if missing:
        checks.append({"name": "complete_no_regression:unrun_cases", "status": "not_run", "missing_cases": missing})
    qualified = warmups >= 5 and repeats >= 30
    statuses = {check["status"] for check in checks}
    overall = "incomplete"
    if qualified:
        overall = "failed" if "failed" in statuses else ("passed" if statuses == {"passed"} else "incomplete")
    return {"status": overall, "formal_sample_count_met": qualified, "requires": {"warmups": 5, "repeats": 30}, "checks": checks}


def peak_rss_bytes() -> int | None:
    try:
        import resource
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(value if sys.platform == "darwin" else value * 1024)
    except (ImportError, AttributeError):
        return None


def rss_worker(spec: CaseSpec, selected: str) -> dict[str, object]:
    from fdm import _native
    _native.require_native()
    quick.prepare_fiber_quick_geometry_backend()
    prepared = prepare_case(spec)
    before = peak_rss_bytes()
    with backend(selected):
        result = invoke(prepared.stages["complete"])
    after = peak_rss_bytes()
    return {"backend": selected, "before_call_peak_bytes": before, "after_call_peak_bytes": after, "peak_increase_bytes": None if before is None or after is None else max(0, after - before), "outcome": result.get("code", result["outcome"]), "scope": "isolated process high-water RSS including prepared inputs; incremental peak is a lower bound, not allocation count"}


def environment_metadata() -> dict[str, object]:
    from fdm import _native
    extension = _native.require_native()
    paths = {Path(__file__).resolve(), Path(contour.__file__), Path(quick.__file__), Path(_native.__file__), ROOT / "scripts/benchmark_contour_comparison.py", ROOT / "pyproject.toml", ROOT / "setup.py"}
    paths.update(ROOT.glob("native/**/*.cpp"))
    paths.update(ROOT.glob("native/**/*.h"))
    paths.update(ROOT.glob("native/**/*.hpp"))
    if getattr(extension, "__file__", None):
        paths.add(Path(extension.__file__))
    versions = {}
    for package in ("numpy", "opencv-python", "scikit-image", "PySide6", "pybind11"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False)
    state = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=False)
    return {"platform": platform.platform(), "machine": platform.machine(), "processor": platform.processor(), "python": sys.version, "executable": sys.executable, "versions": versions, "git_head": revision.stdout.strip() if revision.returncode == 0 else None, "git_dirty": bool(state.stdout.strip()) if state.returncode == 0 else None, "cpu_count": os.cpu_count(), "opencv_threads": cv2.getNumThreads(), "thread_environment": {name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")}, "native_build_info": _native.build_info(), "source_sha256": {str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(paths) if path.is_file()}, "clock": {"implementation": time.get_clock_info("perf_counter").implementation, "resolution_seconds": time.get_clock_info("perf_counter").resolution}}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", choices=("all", "contour", "quick"), default="all")
    parser.add_argument("--case", action="append", default=[], help="Shell-style name filter; repeat to select a union. Quote wildcards.")
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--measure-rss", action="store_true", help="Additional isolated process for each case/backend; outside timings.")
    parser.add_argument("--rss-worker", help=argparse.SUPPRESS)
    parser.add_argument("--rss-backend", choices=("python", "native"), help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.warmups < 0 or args.repeats < 1:
        parser.error("warmups must be >= 0 and repeats >= 1")
    catalog = case_catalog()
    if args.rss_worker:
        matching = [spec for spec in catalog if spec.name == args.rss_worker]
        if len(matching) != 1 or args.rss_backend is None:
            parser.error("invalid RSS worker case/backend")
        print(json.dumps(rss_worker(matching[0], args.rss_backend), allow_nan=False))
        return 0
    selected = [spec for spec in catalog if (args.category == "all" or spec.category == args.category) and (not args.case or any(fnmatch.fnmatchcase(spec.name, pattern) for pattern in args.case))]
    if not selected:
        parser.error("no cases match the selection")
    if args.list_cases:
        print("\n".join(spec.name for spec in selected))
        return 0
    if args.output is None:
        parser.error("--output is required unless --list-cases is used")
    payload = {"schema": "fdm.native-benchmark.v1", "scope": "synthetic service geometry only; no segmentation, GUI, queueing, real-device or Windows package acceptance", "environment": environment_metadata(), "protocol": {"warmups_per_backend_per_stage": args.warmups, "formal_pairs_per_stage": args.repeats, "order": "alternating per pair; serial execution; semantic check after each pair outside timed region", "timed_boundary": "production Python service call including dispatch, conversion, native execution and Python result reconstruction", "float_tolerance": {"absolute": 1e-10, "relative": 1e-12}, "array_equality": "exact shape, dtype and values", "fixtures": "fixed deterministic inputs, no random sampling"}, "cases": []}
    quick.prepare_fiber_quick_geometry_backend()
    for spec in selected:
        prepared = prepare_case(spec)
        row = {"name": spec.name, "category": spec.category, "variant": spec.variant, "input_sha256": prepared.input_sha256, "details": prepared.details, "stages": {}}
        for name, function in prepared.stages.items():
            try:
                row["stages"][name] = paired_stage(function, warmups=args.warmups, repeats=args.repeats)
            except Exception as exc:
                payload["failure"] = {"case": spec.name, "stage": name, "input_sha256": prepared.input_sha256, "error_type": type(exc).__name__, "message": str(exc)}
                payload["acceptance"] = {"status": "failed", "reason": "semantic mismatch or execution error; failing case not accepted"}
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
                raise
        row["process_peak_rss_bytes"] = peak_rss_bytes()
        if args.measure_rss:
            row["isolated_rss"] = {}
            for selected_backend in ("python", "native"):
                process = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--rss-worker", spec.name, "--rss-backend", selected_backend], cwd=ROOT, capture_output=True, text=True, check=True)
                row["isolated_rss"][selected_backend] = json.loads(process.stdout)
        payload["cases"].append(row)
        payload["acceptance"] = acceptance(payload["cases"], warmups=args.warmups, repeats=args.repeats)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps({"case": spec.name, "complete_speedup": row["stages"]["complete"]["speedup"], "outcomes": row["stages"]["complete"]["outcomes"]}, ensure_ascii=False, allow_nan=False), flush=True)
        del prepared
    print(json.dumps({"acceptance": payload["acceptance"]["status"], "output": str(args.output)}, allow_nan=False), flush=True)
    return 1 if payload["acceptance"]["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
