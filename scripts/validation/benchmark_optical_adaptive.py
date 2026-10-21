"""Repeatable algorithm-only optical comparison, not camera latency/accuracy.

Synthetic images are generated here independently of production preprocessing.
Each input and algorithm source is hashed. No parameter fitting is performed.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import platform
from pathlib import Path
import sys
from time import perf_counter

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def sha(data):
    return hashlib.sha256(data).hexdigest()


def inputs():
    h, w = 576, 768
    y, x = np.mgrid[:h, :w]
    noise = np.random.default_rng(314159).normal(0, .65, (h, w))
    for name, core, edge, gradient in (
        ("dark_solid", -170, -170, False),
        ("pale_solid", -18, -18, False),
        ("pale_core_dark_rims", -18, -170, False),
        ("near_clear_core_dark_rims", -3, -170, False),
        ("gradient_pale_core_dark_rims", -18, -120, True),
        ("gradient_pale_solid", -18, -18, True),
    ):
        background = np.broadcast_to(220 + ((x / (w - 1) - .5) * 90 if gradient else 0), (h, w)).astype(float)
        gray = background.copy()
        for center in (190., 390., 590.):
            across = (x - center - .16 * (y - h / 2)) / np.sqrt(1 + .16**2)
            body = np.abs(across) <= 16
            rim = body & (np.abs(across) > 13)
            gray[body] += core
            gray[rim] = background[rim] + edge
        gray = cv2.GaussianBlur(np.clip(gray + noise, 0, 255).astype(np.uint8), (3, 3), .65)
        yield name, cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), {"fiber_count": 3, "normal_width_px": 32.,
              "kind": "parallel_straight", "center_x_at_mid": [190., 390., 590.], "dx_dy": .16}


def complex_inputs():
    shape = (576, 768)
    blank = np.full(shape, 220, np.uint8)
    image = blank.copy()
    image[240:272, 80:688] = 40
    image[300:332, 80:688] = 40
    yield "nearby_two_real_fibers", image, {"criterion": "two_independent_widths", "count": 2, "width_px": 32}
    image = blank.copy()
    image[272:304, 80:688] = 40
    image[70:506, 368:400] = 40
    yield "crossing_two_fibers", image, {"criterion": "local_sections_away_from_intersection", "width_px": 32,
          "physical_fibers": 2, "note": "Multiple visible arms are local proposals, not independent ground-truth fibers"}
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    radius = np.hypot(xx - 190., yy - 288.)
    theta = np.arctan2(yy - 288., xx - 190.)
    image = blank.copy()
    image[(np.abs(radius - 190) <= 16) & (np.abs(theta) < 1.20)] = 40
    yield "curved_solid", image, {"criterion": "local_normal_width", "width_px": 32,
          "radius_px": 190, "note": "Rasterized circular annulus, not continuous optical edge calibration"}
    image = blank.copy()
    image[80:496, :16] = 40
    yield "border_cut_half_fiber", image, {"criterion": "no_complete_two_sided_width", "expected_candidates": 0}
    image = blank.copy()
    image[180:240, 240:300] = 40
    yield "square_particle", image, {"criterion": "not_a_fiber", "expected_candidates": 0}
    image = np.clip(blank.astype(float) + np.random.default_rng(912).normal(0, 2., shape), 0, 255).astype(np.uint8)
    yield "noise_only", image, {"criterion": "no_fiber", "expected_candidates": 0}


def compact(c):
    return {key: c.get(key) for key in ("width_px", "start", "end", "structure", "hypothesis_id", "warning_codes", "conflict_group_id", "conflicts_with")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()
    if args.repetitions < 1 or args.warmup < 0:
        raise ValueError("invalid repetition count")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "benchmark.json").exists():
        raise FileExistsError("Preserve previous runs; use another output directory")
    from fdm.services.cu_automatic_measurement import analyze_frame
    from fdm.services.optical_automatic import PARAMETERS
    cv2.setNumThreads(1)
    initial_opencv_threads = cv2.getNumThreads()
    sources = {str(path.relative_to(ROOT)): sha(path.read_bytes()) for path in (
        ROOT / "src/fdm/services/optical_automatic.py", ROOT / "src/fdm/services/optical_flat_field.py",
        ROOT / "src/fdm/services/cu_automatic_measurement.py", ROOT / "native/cu_automatic.cpp")}
    rows = []
    for name, image, truth in inputs():
        cv2.imwrite(str(output / f"{name}.png"), image)
        for algorithm in ("traditional", "multiscale", "optical_adaptive"):
            warmup_threads = []
            for _ in range(args.warmup):
                threads_before = cv2.getNumThreads()
                analyze_frame(image, algorithm)
                warmup_threads.append({"before": threads_before, "after": cv2.getNumThreads()})
            times, counts, widths, measured_threads = [], [], [], []
            for _ in range(args.repetitions):
                threads_before = cv2.getNumThreads()
                started = perf_counter()
                result = analyze_frame(image, algorithm)
                times.append((perf_counter() - started) * 1000)
                measured_threads.append({"before": threads_before, "after": cv2.getNumThreads()})
                counts.append(len(result["candidates"]))
                widths.append([c["width_px"] for c in result["candidates"]])
            row = dict(case=name, algorithm=algorithm, shape=list(image.shape), source_pixels_sha256=sha(image.tobytes()),
                source_png_sha256=sha((output / f"{name}.png").read_bytes()), truth=truth,
                warmup=args.warmup, repetitions=args.repetitions, p50_ms=float(np.percentile(times, 50)),
                p95_ms=float(np.percentile(times, 95)), timings_ms=times, counts=counts, widths_px=widths,
                opencv_threads_by_call={"warmup": warmup_threads, "measured": measured_threads},
                result_status=result["status"], candidates=[compact(c) for c in result["candidates"]])
            rows.append(row)
            print(f"{name}/{algorithm}: n={counts[-1]} P50={row['p50_ms']:.1f} P95={row['p95_ms']:.1f} ms", flush=True)
    complex_cases = []
    for name, image, expected in complex_inputs():
        cv2.imwrite(str(output / f"{name}.png"), image)
        threads_before = cv2.getNumThreads()
        result = analyze_frame(image, "optical_adaptive")
        complex_cases.append(dict(case=name, source_pixels_sha256=sha(image.tobytes()), shape=list(image.shape),
            opencv_threads_by_call={"before": threads_before, "after": cv2.getNumThreads()},
            expected=expected, status=result["status"], candidates=[compact(c) for c in result["candidates"]]))
        rendered = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        for candidate in result["candidates"]:
            a, b = (tuple(np.rint(candidate[key]).astype(int)) for key in ("start", "end"))
            cv2.line(rendered, a, b, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.putText(rendered, f"{candidate['width_px']:.2f}", (b[0]+5, b[1]), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 0, 255), 1)
        cv2.imwrite(str(output / f"{name}-results.png"), rendered)
    unchanged = all(sha((ROOT / path).read_bytes()) == digest for path, digest in sources.items())
    report = dict(schema_version=2, scope="Synthetic algorithm-only benchmark; no capture, IPC, UI, real camera, or physical metrology acceptance",
        version="0.4.10", runtime=platform.platform(), python=sys.version, opencv=cv2.__version__, numpy=np.__version__,
        requested_initial_opencv_threads=1, initial_opencv_threads=initial_opencv_threads,
        opencv_threads=cv2.getNumThreads(),
        thread_observation="OpenCV configured thread count at call boundaries; service runtime budgets may change it. Not a measurement of actual concurrent worker threads.",
        script_sha256=sha(Path(__file__).read_bytes()), source_sha256=sources,
        sources_unchanged_during_run=unchanged, parameters=asdict(PARAMETERS), rows=rows, complex_cases=complex_cases)
    (output / "benchmark.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return 0 if unchanged else 2


if __name__ == "__main__":
    raise SystemExit(main())
