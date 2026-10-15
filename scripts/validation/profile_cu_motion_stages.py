#!/usr/bin/env python3
"""Diagnostic motion-stage profile; run only after exclusive UI sampling ends.

    python scripts/validation/profile_cu_motion_stages.py --output motion-stages.json

Replays benchmark_cu_motion.py's seed, size order, random base and one-bit noise,
adding the UI harness's 16-bit sequence marker to every current frame. Five
warmups and thirty samples per size compare the unchanged production detector
with an instrumented decomposition of exactly the same CV2 operations. This is
not an alternative detector, a GUI benchmark, or a measured parallel pipeline.
The Sobel stage includes the production's exact uint8 low-difference shortcut;
this post-remediation decomposition is not an optimization-before baseline.
The normal macOS run is intended to take under a minute; actual time is reported.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

# Identical non-OpenCV thread defaults to the formal UI harness. OpenCV's
# application default remains unchanged unless --opencv-threads is explicit.
from scripts.benchmark_cu_live_measurement import distribution, environment
import cv2
import numpy as np

from fdm._native import require_native
from fdm.live_experiment.frames import FrameHandle
from fdm.live_experiment.motion import FieldMotionDetector, MotionResult
import fdm.live_experiment.motion as motion

SIZES = (2048, 4096, 6144)
STAGES = ("gray_reduce", "geometry_cache_check", "centroid", "normalization",
          "forward_dft", "mul_spectrums", "inverse_dft", "native_peak", "sobel")
PREPARATION = ("gray_reduce", "centroid", "normalization", "forward_dft")
DECISION = ("mul_spectrums", "inverse_dft", "native_peak", "sobel")


def digest(array):
    return hashlib.sha256(memoryview(array).cast("B")).hexdigest()


def measured(stages, name, function):
    start = time.perf_counter_ns()
    value = function()
    stages[name] = (time.perf_counter_ns() - start) / 1e6
    return value


def staged_compare(detector, frame):
    """Mirror compare/compare_reduced without changing thresholds or FFT flags."""
    stages = dict.fromkeys(STAGES)
    current = measured(stages, "gray_reduce", lambda: motion.reduced_gray(frame.pixels))
    key = (frame.source_id, frame.generation, frame.size)
    if detector.reference is None or key != detector.source_key:
        raise RuntimeError("Profiler requires the fixed, already initialized reference")
    cache_hit = measured(stages, "geometry_cache_check", lambda:
        detector._geometry_pixels is not None and np.array_equal(current, detector._geometry_pixels))
    if cache_hit:
        reason, delta, shift = detector._geometry_result
    else:
        center = measured(stages, "centroid", lambda: motion.dark_centroid(current))
        delta = (center[0] - detector.reference_centroid[0], center[1] - detector.reference_centroid[1])
        shift = (0, 0)
        if max(abs(delta[0]), abs(delta[1])) > detector.centroid_limit:
            reason = "centroid"
        else:
            normalized = measured(stages, "normalization", lambda: cv2.LUT(current, motion._GRAY_NORMALIZATION))
            spectrum = measured(stages, "forward_dft", lambda:
                cv2.dft(normalized, flags=cv2.DFT_COMPLEX_OUTPUT))
            del normalized
            product = measured(stages, "mul_spectrums", lambda:
                cv2.mulSpectrums(spectrum, detector.reference_fft, flags=0, conjB=True))
            correlated = measured(stages, "inverse_dft", lambda: cv2.idft(product, flags=cv2.DFT_SCALE))
            shift = measured(stages, "native_peak", lambda: tuple(require_native().cu_motion_peak(correlated)))
            reason = "translation" if max(abs(shift[0]), abs(shift[1])) > detector.translation_limit else "stable"
            # Match correlation_shift's temporary-buffer lifetime before Sobel.
            del spectrum, product, correlated
        detector._geometry_pixels = current
        detector._geometry_result = reason, delta, shift
    if reason != "stable":
        result = MotionResult(frame.frame_id, frame.sequence, False, reason, delta, shift)
    else:
        fraction = measured(stages, "sobel", lambda:
            motion.sobel_fraction(detector.previous, current, legacy_signed=detector.legacy_signed))
        detector.previous = current
        stable = fraction <= detector.fraction_limit
        result = MotionResult(frame.frame_id, frame.sequence, stable,
                              "stable" if stable else "appearance", delta, shift, fraction)
    return result, stages, current, bool(cache_hit)


def stage_sum(stages, names):
    # A skipped centroid/FFT branch must not appear as a fast complete pipeline.
    return sum(stages[name] for name in names) if all(stages[name] is not None for name in names) else None


def case(args, size, base, noisy):
    reference = FrameHandle.snapshot(base, source_id="motion-stage-profile", transfer_ownership=True)
    detectors = [FieldMotionDetector(), FieldMotionDetector()]
    for detector in detectors:
        detector.set_reference(reference)
    rows = []
    mismatch_count = cache_hits = duplicate_reduced = 0
    previous_reduced = None
    began = time.perf_counter()
    for index in range(args.warmup + args.samples):
        sequence = index + 1
        # Allocation, marker writing and hashes are deliberately outside timers,
        # just as acquisition/frame allocation are outside benchmark_cu_motion.
        owned = np.array(noisy if index % 2 else base, copy=True, order="C")
        for bit in range(16):
            owned[12:18, 12 + bit * 6:18 + bit * 6] = 200 + ((sequence >> bit) & 1)
        frame = FrameHandle.snapshot(owned, source_id=reference.source_id,
                                     sequence=sequence, transfer_ownership=True)
        results = [None, None]
        elapsed = [None, None]
        stages = current = cache_hit = None
        order = (0, 1) if index % 2 == 0 else (1, 0)
        for implementation in order:
            start = time.perf_counter_ns()
            if implementation == 0:
                results[0] = detectors[0].compare(frame)
            else:
                results[1], stages, current, cache_hit = staged_compare(detectors[1], frame)
            elapsed[implementation] = (time.perf_counter_ns() - start) / 1e6
        matches = results[0] == results[1]
        same_as_previous = previous_reduced is not None and np.array_equal(current, previous_reduced)
        mismatch_count += not matches
        cache_hits += cache_hit
        duplicate_reduced += same_as_previous
        previous_reduced = current
        rows.append({"sequence": sequence, "warmup": index < args.warmup,
            "execution_order": ["production" if which == 0 else "instrumented" for which in order],
            "frame_sha256": digest(frame.pixels), "reduced_sha256": digest(current),
            "equal_to_previous_reduced": same_as_previous, "geometry_cache_hit": cache_hit,
            "production_ms": elapsed[0], "instrumented_ms": elapsed[1], "stages_ms": stages,
            "preparation_sum_ms": stage_sum(stages, PREPARATION),
            "decision_sum_ms": stage_sum(stages, DECISION),
            "production_result": asdict(results[0]), "instrumented_result": asdict(results[1]),
            "result_exact": matches})
        # Keep only the two fixed source images and the next current full frame.
        del frame, owned
    samples = [row for row in rows if not row["warmup"]]
    metrics = {name: distribution([row["stages_ms"][name] for row in samples
                                 if row["stages_ms"][name] is not None]) for name in STAGES}
    for name in ("production_ms", "instrumented_ms", "preparation_sum_ms", "decision_sum_ms"):
        metrics[name] = distribution([row[name] for row in samples if row[name] is not None])
    return {"width": size, "shape": list(base.shape), "reduced_shape": list(previous_reduced.shape),
        "samples": args.samples, "warmup": args.warmup, "case_wall_seconds": time.perf_counter() - began,
        "reference_sha256": digest(base), "noise_source_sha256": digest(noisy),
        "reference_centroid": detectors[0].reference_centroid,
        "reference_fft_sha256": digest(detectors[0].reference_fft),
        "result_disagreements_including_warmup": mismatch_count,
        "geometry_cache_hits_including_warmup": cache_hits,
        "duplicate_consecutive_reduced_frames_including_warmup": duplicate_reduced,
        "all_frames_use_full_geometry": all(row["stages_ms"]["native_peak"] is not None for row in rows),
        "all_results_stable": all(row["production_result"]["stable"] for row in rows),
        "metrics": metrics, "trials": rows}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sizes", type=int, nargs="+", choices=SIZES, default=[2048, 4096])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--opencv-threads", type=int, default=None)
    args = parser.parse_args(argv)
    if args.warmup < 0 or args.samples < 1 or args.warmup + args.samples > 65535:
        parser.error("Require nonnegative warmup, positive samples, and at most 65535 unique markers")
    if args.opencv_threads is not None and args.opencv_threads < 1:
        parser.error("OpenCV thread override must be positive; omit it to preserve the production default")
    started = time.perf_counter()
    if args.opencv_threads is not None:
        cv2.setNumThreads(args.opencv_threads)
    build_environment = environment()
    build_environment.update(opencv_requested_threads=args.opencv_threads,
        opencv_reported_threads=cv2.getNumThreads(),
        opencv_parallel_framework=next((line.strip() for line in cv2.getBuildInformation().splitlines()
                                      if "Parallel framework:" in line), "unknown"))
    for relative in ("scripts/validation/profile_cu_motion_stages.py", "scripts/validation/benchmark_cu_motion.py",
                     "scripts/benchmark_cu_live_measurement.py", "scripts/benchmark_cu_live_ui.py"):
        build_environment["source_sha256"][relative] = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
    report = {"schema_version": 2, "purpose": "post-remediation sequential diagnostic; not before-baseline, GUI or parallel throughput acceptance",
        "environment": build_environment,
        "parameters": {name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()},
        "fixture": {"source": "scripts/validation/benchmark_cu_motion.py", "seed": 20261015,
            "canonical_rng_size_order": list(SIZES), "gray_reduction_factor": 3,
            "base": "uint8 RNG integers [40,216), shape (width*2//3,width,3)",
            "noise": "base XOR independently drawn uint8 integers [0,2)",
            "unique_marker": "16 adjacent 6x6 BGR blocks at (12,12), values 200/201 encode sequence",
            "reference": "unmodified base; every measured current includes the sequence marker"},
        "limits": ["Uses unchanged cv2.dft/mulSpectrums/idft flags and float32 normalization; no other FFT library.",
            "Sobel timing includes the exact uint8 max(absdiff)<=8 shortcut; native peak uses one contiguous parity-preserving scan.",
            "Reference initialization, acquisition/frame allocation and hashes are excluded from sample timings.",
            "Instrumented operations run sequentially. Preparation/decision sums are not evidence of parallel speedup or bounded frame age.",
            "Compare final MotionResult exactly; this is not a new intermediate-bit regression for another FFT implementation.",
            "Unique marker defeats consecutive exact-pixel cache hits; no claim about natural-camera content is implied.",
            "CPU/GPU frame-budget, GUI presentation, PNG and automatic-worker contention are outside this diagnostic.",
            "Run only when exclusive benchmark ownership is available; Windows x64 emulation may exceed the intended macOS runtime."],
        "cases": []}
    rng = np.random.default_rng(20261015)
    for size in SIZES:
        if size > max(args.sizes):
            break
        shape = (size * 2 // 3, size, 3)
        base = rng.integers(40, 216, shape, dtype=np.uint8)
        noisy = np.bitwise_xor(base, rng.integers(0, 2, shape, dtype=np.uint8))
        if size in args.sizes:
            result = case(args, size, base, noisy)
            report["cases"].append(result)
            print(json.dumps({"width": size, "result_disagreements": result["result_disagreements_including_warmup"],
                "cache_hits": result["geometry_cache_hits_including_warmup"],
                "production_p95_ms": result["metrics"]["production_ms"]["p95_ms"],
                "elapsed_seconds": time.perf_counter() - started}, allow_nan=False), flush=True)
        del base, noisy
    report["source_changed_during_run"] = [relative for relative, sha in build_environment["source_sha256"].items()
        if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != sha]
    report["wall_seconds"] = time.perf_counter() - started
    report["correctness_passed"] = not report["source_changed_during_run"] and all(
        not row["result_disagreements_including_warmup"] and not row["geometry_cache_hits_including_warmup"]
        and not row["duplicate_consecutive_reduced_frames_including_warmup"]
        and row["all_frames_use_full_geometry"] and row["all_results_stable"] for row in report["cases"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return 0 if report["correctness_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
