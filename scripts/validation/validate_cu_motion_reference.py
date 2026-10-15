#!/usr/bin/env python3
"""Replay the recorded 37 motion helper cases per product using synthetic pixels."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform

import cv2
import numpy as np

from fdm.live_experiment.motion import compare_geometry, dark_centroid, reduced_gray, sobel_fraction

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests/fixtures/cu_motion_reference.json"
LIMIT = 0.04138815884775365


def xorshift_image() -> np.ndarray:
    state = 0x76543210
    pixels = []
    for _ in range(128 * 96):
        state ^= (state << 13) & 0xffffffff
        state ^= state >> 17
        state ^= (state << 5) & 0xffffffff
        pixels.append(state & 255)
    return np.asarray(pixels, np.uint8).reshape(96, 128)


def replay() -> list[dict]:
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    rows = []
    for product, groups in fixture["products"].items():
        reference = xorshift_image()
        for expected in groups["translation_cases"]:
            current = np.roll(reference, expected["input_wrap_shift_x"], axis=1)
            reason, _delta, shift = compare_geometry(reference, current)
            checks = {"centroid_a": list(dark_centroid(reference)) == expected["centroid_a"],
                      "centroid_b": list(dark_centroid(current)) == expected["centroid_b"],
                      "peak": [shift[1], shift[0]] == expected["peak_shift_yx"],
                      "result": (reason == "stable") == (expected["return"] == 1)}
            rows.append({"product": product, "group": "translation", "input": expected["input_wrap_shift_x"],
                         "passed": all(checks.values()), "checks": checks})
        for expected in groups["centroid_gate_cases"]:
            reference = np.full((96, 128), 255, np.uint8)
            reference[30:62, 44:76] = 0
            current = np.roll(reference, expected["input_shift"], axis=1 if expected["axis"] == "x" else 0)
            reason, _delta, _shift = compare_geometry(reference, current)
            checks = {"centroid_a": list(dark_centroid(reference)) == expected["centroid_a"],
                      "centroid_b": list(dark_centroid(current)) == expected["centroid_b"],
                      "result": (reason == "stable") == (expected["return"] == 1)}
            rows.append({"product": product, "group": "centroid", "input": [expected["axis"], expected["input_shift"]],
                         "passed": all(checks.values()), "checks": checks})
        for expected in groups["constant_cases"]:
            reference = np.full((96, 128), expected["gray"], np.uint8)
            reason, _delta, _shift = compare_geometry(reference, reference)
            checks = {"centroid": list(dark_centroid(reference)) == expected["centroid"],
                      "result": (reason == "stable") == (expected["return"] == 1)}
            rows.append({"product": product, "group": "constant", "input": expected["gray"],
                         "passed": all(checks.values()), "checks": checks})
        for expected in groups["sobel_cases"]:
            amplitude = expected["stripe_amplitude"]
            reference = np.zeros((288, 384), np.uint8)
            current = np.tile(((np.arange(384) // 6) % 2 * amplitude).astype(np.uint8), (288, 1))
            factor = 5 if product == "cu6mt" else 3
            a, b = reduced_gray(reference, factor), reduced_gray(current, factor)
            signed, unsigned = sobel_fraction(a, b, legacy_signed=True), sobel_fraction(a, b)
            signed_count, unsigned_count = round(signed * a.size), round(unsigned * a.size)
            checks = {"size": [a.shape[1], a.shape[0]] == expected["reduced_size"],
                      "signed_count": signed_count == expected["signed_71_127_count"],
                      "unsigned_count": unsigned_count == expected["unsigned_gt70_count"],
                      "legacy_result": (signed <= LIMIT) == (expected["return"] == 1)}
            rows.append({"product": product, "group": "sobel", "input": amplitude,
                "passed": all(checks.values()), "checks": checks, "legacy_stable": signed <= LIMIT,
                "corrected_stable": unsigned <= LIMIT, "corrected_differs": (signed <= LIMIT) != (unsigned <= LIMIT),
                "signed_count": signed_count, "unsigned_count": unsigned_count})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = replay()
    report = {"schema_version": 1, "cases": len(rows), "passed": sum(row["passed"] for row in rows),
        "corrected_sobel_changes": sum(row.get("corrected_differs", False) for row in rows),
        "platform": platform.platform(), "opencv": cv2.__version__, "numpy": np.__version__,
        "fixture_sha256": hashlib.sha256(FIXTURE.read_bytes()).hexdigest(),
        "motion_source_sha256": hashlib.sha256((ROOT / "src/fdm/live_experiment/motion.py").read_bytes()).hexdigest(),
        "limits": ["Replays recorded pure helper observations; no original binary executes.",
                   "Checks centroids, translation peak coordinates, decisions, signed/unsigned gradient counts; old FFT intermediate bits are not claimed.",
                   "Production uses the CU6 sampling profile and corrected unsigned Sobel count.",
                   "Physical camera movement and automatic UI field closure are separate acceptance boundaries."],
        "records": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("cases", "passed", "corrected_sobel_changes")}, allow_nan=False))
    return 0 if report["cases"] == report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
