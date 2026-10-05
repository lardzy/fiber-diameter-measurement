"""Diagnose native input snapshots and contour result reconstruction separately.

Uses the exact fixed inputs of benchmark_native_kernels.py. These independently
timed stages are not additive components of the formal paired run: cache state,
allocation lifetimes and call context differ. Never subtract these numbers from
formal service timings or use them to replace the formal acceptance verdict.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import benchmark_native_kernels as benchmark
from fdm import _native


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def measure(function, warmups: int, repeats: int) -> dict[str, object]:
    samples = []
    for index in range(warmups + repeats):
        started = time.perf_counter_ns()
        result = function()
        elapsed = (time.perf_counter_ns() - started) / 1_000_000
        # Keep returned snapshots/tuples alive until after the timed interval.
        # Deallocation is not part of this deliberately isolated diagnostic.
        del result
        if index >= warmups:
            samples.append(elapsed)
    return {"samples_ms": samples, "summary": benchmark.stats(samples)}


def capture_inputs(prepared, category: str, stage: str, native):
    module = benchmark.contour if category == "contour" else benchmark.quick
    snapshot = module.array_snapshot
    sections = native.contour_sections
    inputs, section_outputs = [], []

    def record_snapshot(value, **kwargs):
        inputs.append((value, kwargs))
        return snapshot(value, **kwargs)

    def record_sections(*args, **kwargs):
        result = sections(*args, **kwargs)
        section_outputs.append(result)
        return result

    try:
        # Instrument only preparation, never any sampled call. These are
        # process-local assignments; production files remain unchanged.
        module.array_snapshot = record_snapshot
        native.contour_sections = record_sections
        with benchmark.backend("native"):
            expected = prepared.stages[stage]()
    finally:
        module.array_snapshot = snapshot
        native.contour_sections = sections
    return inputs, section_outputs, expected


def reconstruct_frames(outputs):
    result = []
    for offsets, intervals in outputs:
        # Exact production _sections() reconstruction, including Python floats
        # and the per-row tuple slices.
        pairs = tuple((float(left), float(right)) for left, right in intervals)
        result.append(tuple(pairs[int(start):int(stop)] for start, stop in zip(offsets[:-1], offsets[1:])))
    return tuple(result)


def snapshot_diagnostic(inputs, roles, warmups, repeats):
    if len(inputs) != len(roles):
        raise RuntimeError(f"Expected {len(roles)} production snapshots, got {len(inputs)}")
    details = []
    for role, (value, kwargs) in zip(roles, inputs):
        snapshot = _native.array_snapshot(value, **kwargs)
        np.testing.assert_array_equal(snapshot, value)
        copied = not np.shares_memory(snapshot, value)
        details.append({
            "role": role,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "strides_bytes": list(value.strides),
            "input_writeable": bool(value.flags.writeable),
            "input_c_contiguous": bool(value.flags.c_contiguous),
            "input_aligned": bool(value.flags.aligned),
            "copied": copied,
            "logical_input_bytes": int(value.nbytes),
            "copied_bytes": int(value.nbytes) if copied else 0,
            "output_writeable": bool(snapshot.flags.writeable),
        })

    def run():
        return tuple(_native.array_snapshot(value, **kwargs) for value, kwargs in inputs)

    return {
        "arrays": details,
        "logical_input_bytes": sum(item["logical_input_bytes"] for item in details),
        "copied_bytes": sum(item["copied_bytes"] for item in details),
        "dtype_conversion": "none; production snapshot validates dtype and copies for ownership/layout only",
        **measure(run, warmups, repeats),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-result", type=Path, default=ROOT / "build/native-kernel-migration/paired-performance.json")
    parser.add_argument("--output", type=Path, default=ROOT / "build/native-kernel-migration/native-overhead.json")
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    args = parser.parse_args(argv)
    if args.warmups < 0 or args.repeats < 1:
        parser.error("warmups must be >= 0; repeats must be >= 1")
    if args.output.resolve() == args.formal_result.resolve():
        parser.error("diagnostic output must not overwrite the formal result")

    started_utc = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    formal_sha = sha256(args.formal_result)
    formal = json.loads(args.formal_result.read_text(encoding="utf-8"))
    formal_cases = {row["name"]: row for row in formal["cases"]}
    original_hashes = formal["environment"]["source_sha256"]
    for name, digest in original_hashes.items():
        path = Path(name) if Path(name).is_absolute() else ROOT / name
        if not path.is_file() or sha256(path) != digest:
            raise RuntimeError(f"Source or binary differs from the formal run: {name}")

    native = _native.require_native(validate_source=True)
    benchmark.quick.prepare_fiber_quick_geometry_backend()
    names = {f"contour_{size}_normal" for size in (2000, 4000, 6000)}
    names.update({"quick_small_rectangle", "quick_diagonal_120px"})
    selected = [spec for spec in benchmark.case_catalog() if spec.name in names or spec.heavy_quick]
    payload = {
        "schema": "fdm.native-overhead-diagnostic.v1",
        "scope": "input snapshot ownership/layout copies and Python contour tuple reconstruction; standalone diagnostic, not additive formal timing components",
        "formal_result": {"path": str(args.formal_result), "sha256": formal_sha, "acceptance": formal["acceptance"]["status"]},
        "environment": benchmark.environment_metadata(),
        "diagnostic_source_sha256": sha256(Path(__file__).resolve()),
        "protocol": {
            "warmups": args.warmups,
            "repeats": args.repeats,
            "preparation": "capture actual production array_snapshot arguments and C++ sections outputs, restore instrumentation before sampling",
            "snapshot_timer": "all production snapshot calls for the stage plus a retaining tuple; returned arrays released outside the timed interval",
            "reconstruction_timer": "same float conversion, interval tuple creation and row slicing as production _sections; per-frame and paired-frame diagnostics",
            "limitations": "warm cache, standalone allocation lifetimes, no C++ compute in sampled stages; do not add/subtract these values to derive formal costs",
        },
        "started_utc": started_utc,
        "cases": [],
    }
    for spec in selected:
        prepared = benchmark.prepare_case(spec)
        if prepared.input_sha256 != formal_cases[spec.name]["input_sha256"]:
            raise RuntimeError(f"Input digest differs from the formal run: {spec.name}")
        stages = ("boundary", "sections") if spec.category == "contour" else ("pruner",)
        row = {"name": spec.name, "category": spec.category, "input_sha256": prepared.input_sha256, "formal_input_sha256_matches": True, "stages": {}}
        for stage in stages:
            inputs, outputs, expected = capture_inputs(prepared, spec.category, stage, native)
            roles = {
                "boundary": ("before_mask", "after_mask"),
                "sections": ("before_segments", "before_heights", "after_segments", "after_heights"),
                "pruner": ("skeleton", "distance_map"),
            }[stage]
            row["stages"][stage] = {"input_snapshots": snapshot_diagnostic(inputs, roles, args.warmups, args.repeats)}
            if stage == "sections":
                if len(outputs) != 2:
                    raise RuntimeError("Expected two native section outputs")
                benchmark.assert_equivalent(reconstruct_frames(outputs), expected)
                row["stages"][stage]["tuple_reconstruction"] = {
                    "production_result_equal": True,
                    "rows_per_frame": [len(offsets) - 1 for offsets, _ in outputs],
                    "intervals_per_frame": [len(intervals) for _, intervals in outputs],
                    "each_frame": [measure(lambda output=output: reconstruct_frames((output,))[0], args.warmups, args.repeats) for output in outputs],
                    "both_frames": measure(lambda: reconstruct_frames(outputs), args.warmups, args.repeats),
                }
        payload["cases"].append(row)
        print(json.dumps({"case": spec.name, "input_sha256_matches": True}, allow_nan=False), flush=True)
        del prepared

    if sha256(args.formal_result) != formal_sha:
        raise RuntimeError("The original formal result changed during this diagnostic")
    for name, digest in original_hashes.items():
        path = Path(name) if Path(name).is_absolute() else ROOT / name
        if sha256(path) != digest:
            raise RuntimeError(f"Formal source or binary changed during this diagnostic: {name}")
    payload["formal_result_preserved"] = True
    payload["formal_sources_preserved"] = True
    payload["finished_utc"] = datetime.now(timezone.utc).isoformat()
    payload["elapsed_seconds"] = time.perf_counter() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"cases": len(selected), "output": str(args.output), "elapsed_seconds": payload["elapsed_seconds"]}, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
