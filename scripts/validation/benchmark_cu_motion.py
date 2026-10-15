#!/usr/bin/env python3
"""Paired motion timing: original CV2 arithmetic versus exact-cache production.

The dynamic case alternates independent one-bit sensor noise. It cannot hit the
pixel-equality cache. Timing includes gray conversion/reduction and both motion
stages, but excludes acquisition, GUI scheduling and frame allocation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import time

import cv2
import numpy as np

from fdm.live_experiment.frames import FrameHandle
from fdm.live_experiment.motion import FieldMotionDetector
import fdm.live_experiment.motion as motion_module


def reference_shift(reference_fft, current):
    normalized = (current.astype(np.float64) / 255.0).astype(np.float32)
    spectrum = cv2.dft(normalized, flags=cv2.DFT_COMPLEX_OUTPUT)
    product = cv2.mulSpectrums(spectrum, reference_fft, flags=0, conjB=True)
    correlated = cv2.idft(product, flags=cv2.DFT_SCALE)
    return motion_module.correlation_peak_reference(correlated)


def reference_sobel_fraction(previous, current, *, legacy_signed=False):
    """Frozen pre-shortcut CV2 arithmetic; independent of production dispatch."""
    difference = cv2.absdiff(previous, current)
    dx = cv2.convertScaleAbs(cv2.Sobel(difference, cv2.CV_16S, 1, 0, ksize=3,
                                     borderType=cv2.BORDER_REPLICATE))
    dy = cv2.convertScaleAbs(cv2.Sobel(difference, cv2.CV_16S, 0, 1, ksize=3,
                                     borderType=cv2.BORDER_REPLICATE))
    edges = cv2.add(dx, dy)
    changed = edges > 70
    if legacy_signed:
        changed &= edges <= 127
    return float(np.count_nonzero(changed)) / edges.size


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--sizes", type=int, nargs="+", default=[2048, 4096, 6144])
    parser.add_argument("--opencv-threads", type=int, default=None)
    args = parser.parse_args()
    if args.opencv_threads is not None:
        cv2.setNumThreads(args.opencv_threads)
    rng = np.random.default_rng(20261015)
    production_shift = motion_module.correlation_shift
    production_sobel = motion_module.sobel_fraction
    rows = []
    for size in args.sizes:
        shape = (size * 2 // 3, size, 3)
        pixels = rng.integers(40, 216, shape, dtype=np.uint8)
        frame = FrameHandle.snapshot(pixels, source_id="benchmark", transfer_ownership=True)
        noisy = np.bitwise_xor(pixels, rng.integers(0, 2, shape, dtype=np.uint8))
        other = FrameHandle.snapshot(noisy, source_id="benchmark", sequence=1, transfer_ownership=True)
        for scenario in ("identical", "dynamic_one_bit_noise"):
            detectors = [FieldMotionDetector(), FieldMotionDetector()]
            for detector in detectors:
                detector.set_reference(frame)
            times = [[], []]
            disagreements = 0
            for step in range(args.warmup + args.samples):
                sample = other if scenario != "identical" and step % 2 else frame
                results = [None, None]
                # Alternate order to reduce frequency/temperature order bias.
                for index in ((0, 1) if step % 2 == 0 else (1, 0)):
                    if index == 0:
                        detectors[index]._geometry_pixels = None
                    motion_module.correlation_shift = reference_shift if index == 0 else production_shift
                    motion_module.sobel_fraction = reference_sobel_fraction if index == 0 else production_sobel
                    start = time.perf_counter_ns()
                    try:
                        results[index] = detectors[index].compare(sample)
                    finally:
                        motion_module.correlation_shift = production_shift
                        motion_module.sobel_fraction = production_sobel
                    elapsed = (time.perf_counter_ns() - start) / 1e6
                    if step >= args.warmup:
                        times[index].append(elapsed)
                disagreements += results[0] != results[1]
            rows.append({"size": list(frame.size), "scenario": scenario,
                "samples": args.samples, "warmup": args.warmup,
                "result_disagreements": disagreements,
                "reference_sha256": hashlib.sha256(pixels).hexdigest(),
                "current_sha256": hashlib.sha256(noisy).hexdigest(),
                "baseline_no_cache_ms": {"p50": float(np.percentile(times[0], 50)),
                    "p95": float(np.percentile(times[0], 95)), "samples": times[0]},
                "production_ms": {"p50": float(np.percentile(times[1], 50)),
                    "p95": float(np.percentile(times[1], 95)), "samples": times[1]}})
            print(json.dumps({key: value for key, value in rows[-1].items()
                  if key not in {"reference_sha256", "current_sha256"}}, allow_nan=False), flush=True)
    source = Path(__file__).resolve().parents[2] / "src/fdm/live_experiment/motion.py"
    report = {"schema_version": 2, "platform": platform.platform(),
        "opencv": cv2.__version__, "numpy": np.__version__, "opencv_requested_threads": args.opencv_threads,
        "opencv_reported_threads": cv2.getNumThreads(),
        "opencv_parallel_framework": next((line.strip() for line in cv2.getBuildInformation().splitlines()
            if "Parallel framework:" in line), "unknown"),
        "native_build_info": __import__("fdm._native", fromlist=["build_info"]).build_info(),
        "motion_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "benchmark_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "baseline_definition": "cache disabled; double-to-float normalization; original CV2 DFT/multiply/idft; independent NumPy parity peak; frozen pre-shortcut CV2 Sobel",
        "production_definition": "exact geometry cache, LUT normalization, contiguous native parity peak, proven uint8 low-difference Sobel shortcut",
        "limits": ["Includes reduction and motion; excludes GUI, capture, allocation and PNG.",
                   "Baseline disables cache and uses independent frozen CV2 Sobel, double normalization and NumPy peak; no production shortcut leaks into the baseline.",
                   "Dynamic independent sensor noise intentionally misses the equality cache.",
                   "Synthetic performance is not physical microscopy acceptance."], "records": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return int(any(row["result_disagreements"] for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
