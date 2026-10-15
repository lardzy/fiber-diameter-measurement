#!/usr/bin/env python3
"""Replay the 177 audited CU fixtures against source kernels; never runs old DLLs.

Usage: python scripts/validation/validate_cu_automatic_gold.py --cu-root /Volumes/LARD/cu_reverse --output /tmp/cu-automatic-gold.json
"""
from __future__ import annotations
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np


def compare_result(result, original, algorithm):
    differences = []
    expected = original.get("fibers", [])
    actual = result["candidates"]
    if len(actual) != len(expected):
        differences.append({"field": "count", "actual": len(actual), "expected": len(expected)})
    for index, (got, want) in enumerate(zip(actual, expected)):
        width = want.get("diameter_px", want.get("raw_diameter_px"))
        if width is None or not np.isclose(got["width_px"], width, rtol=1e-10, atol=1e-8):
            differences.append({"field": "width_px", "index": index, "actual": got["width_px"], "expected": width})
        if algorithm == "traditional":
            samples = [{"x1": s["start"][0], "y1": s["start"][1], "x2": s["end"][0], "y2": s["end"][1]} for s in got["samples"]]
            if samples != want.get("samples"):
                differences.append({"field": "sample_endpoints", "index": index})
            if got["circle_indices"] != want.get("circle_indices"):
                differences.append({"field": "circle_order", "index": index})
        else:
            # Original exposes centerline topology, but not width samples.
            if [s["points"] for s in got["segments"]] != [s["points"] for s in want.get("centerlines", [])]:
                differences.append({"field": "segments", "index": index})
            if got["pass"] != want.get("pass"):
                differences.append({"field": "pass", "index": index})
    if algorithm == "traditional" and result["diagnostics"]["circle_count"] != original["circle_count"]:
        differences.append({"field": "circle_count", "actual": result["diagnostics"]["circle_count"], "expected": original["circle_count"]})
    return differences


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cu-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native-module", default="fdm._native._kernels")
    parser.add_argument("--native-path", type=Path)
    args = parser.parse_args()
    if args.native_path:
        sys.path.insert(0, str(args.native_path))
    native = importlib.import_module(args.native_module)
    cv2.setNumThreads(1)
    manifest_path = args.cu_root / "docs/validation/diameter_demo_20261011.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    references = {}
    for suite in manifest["validation_suites"]:
        for case in suite["cases"]:
            run = json.loads((args.cu_root / case["raw_run"]).read_text(encoding="utf-8"))
            references[(run["source"]["normalized_bmp_sha256"], case["preset"], case["engine"])] = (run, case["raw_run"])
    report = {"cv2": cv2.__version__, "native_build_info": dict(native.build_info()) if hasattr(native, "build_info") else {"isolated_test_module": args.native_module}, "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(), "cases": []}
    for suite in manifest["validation_suites"]:
        for case in suite["cases"]:
            run_path = args.cu_root / case["raw_run"]
            run = json.loads(run_path.read_text(encoding="utf-8"))
            original = run["results"]["original"]
            image = cv2.imread(str(run_path.parent / "input.bmp"))
            if image is None:
                raise RuntimeError(f"Missing gold input: {run_path}")
            algorithm = "multiscale" if case["engine"] == "cu6mt" else "traditional"
            started = time.perf_counter()
            record = {"image": case["image"], "engine": case["engine"], "preset": case["preset"], "raw_run": case["raw_run"]}
            try:
                result = native._cu_automatic(image, algorithm, None, 10000, 2 * 1024**3, case["preset"], True)
                differences = compare_result(result, original, algorithm)
                image_diffs = {}
                image_details = {}
                for key, filename in original.get("images", {}).items():
                    if key not in result["diagnostics"] or key in {"original", "overlay", "gray", "grayscale"}:
                        continue
                    expected = cv2.imread(str(run_path.parent / "original" / filename), cv2.IMREAD_GRAYSCALE)
                    if expected is None:
                        continue
                    got = result["diagnostics"][key]
                    image_diffs[key] = int(np.count_nonzero(got != expected)) if got.shape == expected.shape else -1
                    if image_diffs[key] > 0:
                        coordinates = np.argwhere(got != expected)
                        image_details[key] = {
                            "bbox_xyxy": [int(coordinates[:, 1].min()), int(coordinates[:, 0].min()),
                                          int(coordinates[:, 1].max()), int(coordinates[:, 0].max())],
                            "pixels_preview": [{"x": int(x), "y": int(y), "actual": int(got[y, x]),
                                                "expected": int(expected[y, x])} for y, x in coordinates[:32]],
                            "preview_truncated": len(coordinates) > 32,
                            "role": {
                                "contour": "Computational input to legacy gather/opposite/contour_slope; also modified during traversal. This is not merely an overlay.",
                                "selection": "Computational width-sampling mask in modern passes 1 and 2. Drawn before width rejection; may include rejected centerlines.",
                                "mask": "Accepted pass 1/2 geometry used to exclude overlaps in passes 2/3.",
                                "edges": "Dilated edge map consumed by modern pass 1/2 width sampling.",
                                "mask_labeled": "Legacy computational component/circle label map used during candidate grouping.",
                            }.get(key, "Computational stage; downstream equivalence is not established by final results alone."),
                        }
                status = "equal" if not differences else "different"
                if algorithm == "traditional" and case["engine"] == "cu6" and differences and all(d["field"] == "count" for d in differences):
                    reference = references.get((run["source"]["normalized_bmp_sha256"], case["preset"], "cu6mt_legacy"))
                    if reference and not compare_result(result, reference[0]["results"]["original"], algorithm):
                        status = "capacity_extension"
                        record["capacity_reference_run"] = reference[1]
                if algorithm == "multiscale":
                    expected_passes = [{"count": p["count"], "valid": p["valid"]} for p in original["pass_candidates"]]
                    record["internal_passes_equal"] = result["diagnostics"]["passes"] == expected_passes
                    if not record["internal_passes_equal"]:
                        record["internal_passes"] = {"actual": result["diagnostics"]["passes"], "expected": expected_passes}
                record.update(status=status, differences=differences, intermediate_different_pixels=image_diffs,
                              intermediate_difference_details=image_details, count=len(result["candidates"]))
            except Exception as exc:
                record.update(status="error", error=f"{type(exc).__name__}: {exc}")
            record["milliseconds"] = (time.perf_counter() - started) * 1000
            report["cases"].append(record)
            print(len(report["cases"]), record["engine"], record["image"], record["status"], record.get("differences", record.get("error")), flush=True)
    report["summary"] = {key: sum(c["status"] == key for c in report["cases"]) for key in ("equal", "capacity_extension", "different", "error")}
    report["scope"] = "Final accepted results are compared separately from intermediate images and rejected pass candidates. Capacity extensions are accepted only against a matching uncapped CU6mt original result."
    report["intermediate_image_differences"] = sum(any(c.get("intermediate_different_pixels", {}).values()) for c in report["cases"])
    report["internal_pass_count_differences"] = sum(c.get("internal_passes_equal") is False for c in report["cases"])
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(report["summary"])
    return 0 if (report["summary"]["equal"] + report["summary"]["capacity_extension"] == 177
                 and report["intermediate_image_differences"] == 0
                 and report["internal_pass_count_differences"] == 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
