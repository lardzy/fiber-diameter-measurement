#!/usr/bin/env python3
"""Lossless PNG strategy experiment; never changes production evidence policy."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import time

import cv2
import numpy as np


def inputs(width):
    height = width * 2 // 3
    image = np.full((height, width, 3), 240, np.uint8)
    for y in np.linspace(height // 8, height * 7 // 8, 8).astype(int):
        image[y - 10:y + 10, 20:width - 20] = 30
    yield "uniform_stripes", image
    rng = np.random.default_rng(20261015)
    yield "stripes_one_bit_noise", np.bitwise_xor(image, rng.integers(0, 2, image.shape, np.uint8))
    image = None
    gradient = np.linspace(24, 230, width, dtype=np.float32)[None, :, None]
    # Three gently varying color channels plus bounded sensor-like noise.
    noise = rng.integers(-12, 13, (height, width, 3), dtype=np.int16)
    natural = np.clip(noise + gradient + np.array([0, 4, -4], np.float32), 0, 255).astype(np.uint8)
    noise = gradient = None
    yield "gradient_with_noise", natural


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--samples", type=int, default=10)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    strategies = [
        ("compression1_default", 1, cv2.IMWRITE_PNG_STRATEGY_DEFAULT),
        ("compression1_rle", 1, cv2.IMWRITE_PNG_STRATEGY_RLE),
        ("compression1_huffman_only", 1, cv2.IMWRITE_PNG_STRATEGY_HUFFMAN_ONLY),
        ("compression0_default", 0, cv2.IMWRITE_PNG_STRATEGY_DEFAULT),
    ]
    records = []
    for scenario, pixels in inputs(args.size):
        times = {name: [] for name, *_ in strategies}
        sizes, hashes, exact = {}, {}, {name: 0 for name, *_ in strategies}
        for step in range(args.warmup + args.samples):
            # Rotate order while holding one immutable source for every pair.
            order = strategies[step % 4:] + strategies[:step % 4]
            for name, compression, strategy in order:
                params = [cv2.IMWRITE_PNG_COMPRESSION, compression, cv2.IMWRITE_PNG_STRATEGY, strategy]
                started = time.perf_counter_ns()
                success, encoded = cv2.imencode(".png", pixels, params)
                duration = (time.perf_counter_ns() - started) / 1e6
                if not success:
                    raise RuntimeError("PNG encoding failed")
                decoded = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
                if not np.array_equal(decoded, pixels):
                    raise AssertionError(f"PNG roundtrip differs: {scenario}/{name}")
                exact[name] += 1
                if step >= args.warmup:
                    times[name].append(duration)
                sizes[name] = encoded.nbytes
                hashes[name] = hashlib.sha256(encoded).hexdigest()
                encoded = decoded = None
        for name, compression, strategy in strategies:
            row = {"scenario": scenario, "strategy": name, "compression": compression,
                "strategy_code": strategy, "size": list(pixels.shape), "raw_bytes": pixels.nbytes,
                "input_sha256": hashlib.sha256(pixels).hexdigest(),
                "encoded_bytes": sizes[name], "encoded_sha256": hashes[name],
                "roundtrip_pixel_exact": exact[name], "warmup": args.warmup, "samples": args.samples,
                "p50_ms": float(np.percentile(times[name], 50)), "p95_ms": float(np.percentile(times[name], 95)),
                "samples_ms": times[name]}
            records.append(row)
            print(json.dumps({key: row[key] for key in ("scenario", "strategy", "encoded_bytes", "p50_ms", "p95_ms", "roundtrip_pixel_exact")}, allow_nan=False), flush=True)
    report = {"schema_version": 1, "platform": platform.platform(), "opencv": cv2.__version__,
        "numpy": np.__version__, "opencv_threads": cv2.getNumThreads(),
        "limits": ["Synthetic inputs, not real microscopy image storage ratios.",
                   "Times measure imencode only; readback verifies every output outside timing.",
                   "No production evidence strategy changes are made by this experiment."], "records": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
