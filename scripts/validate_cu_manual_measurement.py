#!/usr/bin/env python3
"""Replay the existing CU manual goldens; never execute CU software.

Only source images and recorded JSON are read from the reference directory.
Output is a standalone JSON audit with source/fixture identities. The references
are optional local research material and are not a production dependency.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import platform
import statistics
import sys
import time

import numpy as np
from PIL import Image


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percentile(values: list[float], fraction: float) -> float:
    return float(np.percentile(np.asarray(values), fraction * 100))


def replay(reference: Path, module) -> dict:
    audit_path = reference / "docs/validation/diameter_manual_20261011.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    records = []
    for reference_row in audit["differential_validation"]["records"]:
        raw = reference / reference_row["raw_run"]["path"].replace("\\", "/")
        actual_digest = digest(raw)
        if actual_digest != reference_row["raw_run"]["sha256"]:
            raise ValueError(f"Golden result digest differs: {raw}")
        run = json.loads(raw.read_text(encoding="utf-8"))
        expected = run["results"]["original"]["measurement"]
        image_path = raw.parent / "input.bmp"
        if digest(image_path) != reference_row["normalized_input_sha256"]:
            raise ValueError(f"Golden image digest differs: {image_path}")
        with Image.open(image_path) as source:
            image = np.array(source.convert("RGB"))[..., ::-1]
        line = reference_row["input_line"]
        actual = module.cu_manual_measure(image, line[:2], line[2:])
        valid = expected["fit_success"]
        checks = {
            "acceptance": (actual["status"] == "ok") == valid,
            "start": list(actual["start"]) == expected["fitted_line"][:2],
            "end": list(actual["end"]) == expected["fitted_line"][2:],
            "edge_a": [list(p) for p in actual["edge_a"]] == expected["edge_first"],
            "edge_b": [list(p) for p in actual["edge_b"]] == expected["edge_second"],
        }
        difference = None
        bit_equal = None
        if valid and actual["width_px"] is not None:
            difference = abs(actual["width_px"] - expected["pixel_return_code"])
            bit_equal = np.float64(actual["width_px"]).tobytes() == np.float64(expected["pixel_return_code"]).tobytes()
            checks["diameter"] = difference <= 1e-9 + 1e-12 * abs(expected["pixel_return_code"])
        else:
            checks["diameter"] = not valid and actual["width_px"] is None
        records.append({
            "product": reference_row["product"], "case": reference_row["case"],
            "passed": all(checks.values()), "checks": checks,
            "expected_accepted": valid, "actual_reason": actual["reason"],
            "width_absolute_difference_px": difference, "width_bits_equal": bit_equal,
            "golden_result_sha256": actual_digest, "input_sha256": digest(image_path),
            "diagnostics": actual["diagnostics"],
        })
    return {
        "audit_sha256": digest(audit_path), "cases": len(records),
        "passed": sum(r["passed"] for r in records),
        "positive": sum(r["expected_accepted"] for r in records),
        "positive_bits_equal": sum(r["width_bits_equal"] is True for r in records),
        "max_absolute_difference_px": max((r["width_absolute_difference_px"] or 0) for r in records),
        "acceptance": "exact states, endpoints and ordered edge traces; width atol=1e-9 px, rtol=1e-12; bit equality reported separately",
        "records": records,
    }


def benchmarks(module) -> list[dict]:
    records = []
    for width, height in ((320, 240), (2048, 1536), (4096, 3072), (6144, 4096)):
        image = np.full((height, width, 3), 240, np.uint8)
        middle = height // 2
        image[middle - 10:middle + 10, 20:width - 20] = 30
        line = ((width // 2, middle - 40), (width // 2, middle + 40))
        for _ in range(5):
            module.cu_manual_measure(image, *line)
        samples = []
        for _ in range(30):
            start = time.perf_counter_ns()
            result = module.cu_manual_measure(image, *line)
            samples.append((time.perf_counter_ns() - start) / 1_000_000)
            if result["status"] != "ok" or result["width_px"] != 19:
                raise AssertionError("Benchmark returned the wrong measurement")
        records.append({"frame_size": [width, height], "image_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
            "input_line": line, "warmups": 5, "samples": 30, "p50_ms": statistics.median(samples),
            "p95_ms": percentile(samples, .95), "max_ms": max(samples),
            "processed_pixels": result["diagnostics"]["processed_pixels"], "values_ms": samples})
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native-module", default="fdm._native._kernels", help="Development probe modules are recorded explicitly")
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()
    if args.native_module == "fdm._native._kernels":
        from fdm import _native
        module = _native.require_native(validate_source=True)
    else:
        module = importlib.import_module(args.native_module)
    root = Path(__file__).resolve().parents[1]
    paths = [root / name for name in ("native/cu_manual.cpp", "native/cu_manual.hpp", "src/fdm/services/cu_manual_measurement.py", "scripts/validate_cu_manual_measurement.py")]
    report = {
        "schema_version": 1, "algorithm_revision": "cu-manual-portable-v1",
        "host": {"platform": platform.platform(), "machine": platform.machine(), "python": sys.version, "numpy": np.__version__},
        "native_module": args.native_module, "native_file": str(module.__file__),
        "native_sha256": digest(Path(module.__file__)),
        "native_build_info": module.build_info() if hasattr(module, "build_info") else None,
        "sources": [{"path": str(p.relative_to(root)), "sha256": digest(p)} for p in paths],
        "limits": ["Recorded CU differential evidence is not physical measurement accuracy.",
            "Timing includes native ROI copying and Python result construction, excludes acquisition, task queue and GUI rendering.",
            "No original EXE or DLL is executed. Desktop architecture and installed-package acceptance are separate."],
    }
    if args.reference_root:
        report["differential"] = replay(args.reference_root, module)
    if args.benchmark:
        report["benchmark"] = benchmarks(module)
    report["passed"] = "differential" not in report or report["differential"]["passed"] == report["differential"]["cases"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "cases": report.get("differential", {}).get("cases"), "output": str(args.output)}, ensure_ascii=False, allow_nan=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
