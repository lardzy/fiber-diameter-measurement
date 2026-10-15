#!/usr/bin/env python3
"""Compare saved R3 and rebuilt R4 native peak functions in one interpreter.

Example (run only in an exclusively assigned validation window):
  python benchmark_cu_motion_peak_pair.py --old-native /path/r3/_kernels.so \
    --new-native /path/r4/_kernels.so --output peak-pair.json

Both actual extension filenames are loaded under distinct package names ending
in ._kernels. No production dispatch, source fingerprint validation, or native
build is performed. Every timed call includes the real pybind function call;
input allocation, NumPy oracle, hashing, and output comparisons are not timed.
This is a peak-only microbenchmark, not motion/UI/throughput acceptance.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import sys
import sysconfig
import time
from types import ModuleType

import numpy as np


SIZES = (2048, 4096, 6144)
LABELS = ("r3", "r4")


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(array):
    # Invalid-layout boundary inputs are tiny; copying here is outside timing.
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def load_native(path, label):
    path = path.resolve(strict=True)
    package_name = "_fdm_peak_pair_" + label
    package = ModuleType(package_name)
    package.__path__ = []
    sys.modules[package_name] = package
    name = package_name + "._kernels"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot create native-module loader for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    if Path(module.__file__).resolve() != path:
        raise RuntimeError("Native loader resolved an unexpected binary")
    info = dict(module.build_info())
    if not callable(getattr(module, "cu_motion_peak", None)):
        raise RuntimeError("The selected native module has no cu_motion_peak")
    return module, {"path": str(path), "module_name": name,
                    "file_sha256": file_sha256(path), "build_info": info}


def numpy_peak_reference(correlation):
    """Frozen double-promoted magnitude and legacy parity tie oracle."""
    real = correlation[:, :, 0].astype(np.float64)
    imaginary = correlation[:, :, 1].astype(np.float64)
    magnitude = real * real + imaginary * imaginary
    best, best_y, best_x = 0.0, 0, 0
    for parity in (0, 1):
        part = magnitude[:, parity::2]
        if not part.size:
            continue
        row, column = np.unravel_index(int(part.argmax()), part.shape)
        value = float(part[row, column])
        if value > 0 and value >= best:
            best, best_y, best_x = value, int(row), int(column * 2 + parity)
    if best_x > correlation.shape[1] // 2:
        best_x -= correlation.shape[1]
    if best_y > correlation.shape[0] // 2:
        best_y -= correlation.shape[0]
    return [best_x, best_y]


def outcome(function, array):
    try:
        result = function(array)
    except Exception as exc:
        return {"kind": "error", "type": type(exc).__name__, "message": str(exc)}
    return {"kind": "ok", "point": [int(value) for value in result]}


def distribution(values):
    return {"count": len(values), "p50_ms": float(np.percentile(values, 50)),
            "p95_ms": float(np.percentile(values, 95)),
            "p99_ms": float(np.percentile(values, 99)),
            "max_ms": max(values), "samples_ms": values}


def boundary_fixtures():
    # Writing through uint32 preserves signaling/quiet bits without conversion.
    exceptional_bits = (0x7F800000, 0xFF800000,
        0x7FC00000, 0xFFC00000, 0x7FC00001, 0xFFC00001,
        0x7FFFFFFF, 0xFFFFFFFF, 0x7F800001, 0xFF800001,
        0x7FBFFFFF, 0xFFBFFFFF)
    for bits in exceptional_bits:
        for y, x in ((0, 0), (1, 2), (2, 4)):
            for component in (0, 1):
                array = np.zeros((3, 5, 2), np.float32)
                array.view(np.uint32)[y, x, component] = bits
                yield (f"nonfinite_{bits:08x}_at_{y}_{x}_{component}", array,
                       "ValueError", "correlation must contain finite values")
    finite_bits = (0x00000000, 0x80000000, 0x00000001, 0x80000001,
        0x007FFFFF, 0x807FFFFF, 0x00800000, 0x80800000,
        0x7F7FFFFF, 0xFF7FFFFF)
    for bits in finite_bits:
        array = np.zeros((3, 5, 2), np.float32)
        array.view(np.uint32)[2, 4, 0] = bits
        yield f"finite_{bits:08x}", array, None, None
    for shape in ((1, 1), (1, 5), (2, 2), (3, 5), (4, 6)):
        yield f"all_zero_{shape}", np.zeros((*shape, 2), np.float32), None, None
        yield f"all_equal_{shape}", np.ones((*shape, 2), np.float32), None, None
        for y in range(shape[0]):
            for x in range(shape[1]):
                array = np.zeros((*shape, 2), np.float32)
                array[y, x, 0] = np.finfo(np.float32).max
                array[-1, -1, 1] = -np.finfo(np.float32).max
                yield f"parity_wrap_{shape}_{y}_{x}", array, None, None
    # 2^-26 is below float32's half unit at 1 but above double's resolution:
    # (1,2^-13) must beat (1,0) under double arithmetic, despite float32's tie.
    # 2^-54 is below double's half unit at 1, so the odd column must win that tie.
    for exponent in (-13, -26, -27):
        for stronger_parity in (0, 1):
            array = np.zeros((3, 6, 2), np.float32)
            array[0, :2, 0] = 1
            array[0, stronger_parity, 1] = np.float32(2.0 ** exponent)
            yield f"near_tie_{exponent}_{stronger_parity}", array, None, None
    invalid = (
        ("float64", np.zeros((2, 3, 2), np.float64), "TypeError"),
        ("rank2", np.zeros((2, 3), np.float32), "ValueError"),
        ("empty", np.zeros((0, 3, 2), np.float32), "ValueError"),
        ("one_component", np.zeros((2, 3, 1), np.float32), "ValueError"),
        ("reversed", np.zeros((2, 3, 2), np.float32)[:, ::-1], "ValueError"),
        ("unaligned", np.ndarray((2, 3, 2), np.float32,
             buffer=bytearray(2 * 3 * 2 * 4 + 1), offset=1), "ValueError"))
    for name, array, error in invalid:
        yield name, array, error, None


def check_boundaries(functions):
    rows = []
    recovery = np.zeros((2, 3, 2), np.float32)
    recovery.setflags(write=False)
    for name, array, error, message in boundary_fixtures():
        array.setflags(write=False)
        before = array_sha256(array)
        outputs = {label: outcome(functions[label], array) for label in LABELS}
        expected = None if error else numpy_peak_reference(array)
        agrees = outputs["r3"] == outputs["r4"]
        valid = all((item["kind"] == "error" and item["type"] == error
                     and (message is None or item["message"] == message)) if error else
                    (item["kind"] == "ok" and item["point"] == expected)
                    for item in outputs.values())
        after_error = ({label: outcome(functions[label], recovery) for label in LABELS}
                       if error else None)
        recovered = after_error is None or all(item == {"kind": "ok", "point": [0, 0]}
                                               for item in after_error.values())
        unchanged = before == array_sha256(array)
        rows.append({"name": name, "shape": list(array.shape), "dtype": str(array.dtype),
            "strides": list(array.strides), "input_sha256": before, "outputs": outputs,
            "expected_exception": error, "expected_point": expected,
            "outputs_equal": agrees, "matches_contract": valid,
            "recovery_outputs": after_error, "recovery_ok": recovered,
            "input_unchanged": unchanged, "passed": agrees and valid and recovered and unchanged})
    return {"cases": rows, "case_count": len(rows),
            "failures": sum(not row["passed"] for row in rows)}


def benchmark_case(functions, size, warmup, samples):
    reduced_shape = ((size * 2 // 3) // 3, size // 3, 2)
    # A per-size seed preserves fixture identity when selecting one size only.
    rng = np.random.default_rng(20261015 + size)
    array = rng.standard_normal(reduced_shape, dtype=np.float32)
    array *= np.float32(2 ** 20)
    array.setflags(write=False)
    before = array_sha256(array)
    expected = numpy_peak_reference(array)
    rows = []
    timings = {label: [] for label in LABELS}
    for index in range(warmup + samples):
        order = LABELS if index % 2 == 0 else tuple(reversed(LABELS))
        outputs, elapsed = {}, {}
        for label in order:
            function = functions[label]
            started = time.perf_counter_ns()
            try:
                value = function(array)
            except Exception as exc:
                elapsed[label] = (time.perf_counter_ns() - started) / 1e6
                outputs[label] = {"kind": "error", "type": type(exc).__name__, "message": str(exc)}
            else:
                elapsed[label] = (time.perf_counter_ns() - started) / 1e6
                outputs[label] = {"kind": "ok", "point": [int(item) for item in value]}
            if index >= warmup:
                timings[label].append(elapsed[label])
        agrees = outputs["r3"] == outputs["r4"]
        valid = all(item == {"kind": "ok", "point": expected} for item in outputs.values())
        rows.append({"iteration": index, "warmup": index < warmup,
                     "execution_order": list(order), "elapsed_ms": elapsed,
                     "outputs": outputs, "outputs_equal": agrees,
                     "matches_reference": valid})
    metrics = {label: distribution(timings[label]) for label in LABELS}
    return {"input_width": size, "input_height": size * 2 // 3,
        "shape": list(array.shape), "dtype": str(array.dtype),
        "input_sha256": before, "input_unchanged": before == array_sha256(array),
        "input_bytes": array.nbytes, "expected_point": expected,
        "fixture": "fixed seeded float32 complex-normal components times 2^20; not a full motion input",
        "warmup": warmup, "samples": samples,
        "output_disagreements_including_warmup": sum(not row["outputs_equal"] for row in rows),
        "reference_mismatches_including_warmup": sum(not row["matches_reference"] for row in rows),
        "metrics": metrics, "p95_speedup_r3_over_r4": metrics["r3"]["p95_ms"] / metrics["r4"]["p95_ms"],
        "trials": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--old-native", required=True, type=Path)
    parser.add_argument("--new-native", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sizes", type=int, nargs="+", choices=SIZES, default=list(SIZES))
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--samples", type=int, default=30)
    args = parser.parse_args()
    if args.warmup < 0 or args.samples < 1:
        parser.error("Require nonnegative warmup and at least one formal sample")
    if args.old_native.resolve() == args.new_native.resolve():
        parser.error("Old and new modules must be distinct saved files")
    started = time.perf_counter()
    report = {"schema_version": 1, "purpose": "same-process paired native peak microbenchmark",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {"platform": platform.platform(), "machine": platform.machine(),
            "python": sys.version, "python_executable": sys.executable,
            "python_abi": sysconfig.get_config_var("SOABI") or sys.implementation.cache_tag,
            "numpy": np.__version__, "script_sha256": file_sha256(Path(__file__).resolve())},
        "parameters": {key: str(value) if isinstance(value, Path) else value
                       for key, value in vars(args).items()},
        "limits": ["Run only after all other formal workloads/builds stop.",
            "Each pair uses the identical immutable complex64 storage and includes the actual pybind call.",
            "Random complex component fixture isolates the peak scan; no DFT, acquisition, PNG, GUI or throughput is timed.",
            "Warmups are retained for correctness but excluded from timing percentiles.",
            "No change to floating-point exception masks/rounding/denormal state is requested.",
            "Default-process NaN/error contracts are tested; unmasked floating-point traps are outside this harness.",
            "No performance improvement is an automatic acceptance of full motion or UI thresholds."],
        "binaries": {}, "boundary_validation": None, "cases": [], "passed": False}
    status = 1
    try:
        modules = {}
        for label, path in zip(LABELS, (args.old_native, args.new_native)):
            modules[label], report["binaries"][label] = load_native(path, label)
        old_info, new_info = (report["binaries"][label]["build_info"] for label in LABELS)
        if old_info.get("source_sha256") == new_info.get("source_sha256"):
            raise RuntimeError("Both binaries report the same native source fingerprint; R3/R4 separation is unproven")
        if not old_info.get("source_sha256") or not new_info.get("source_sha256"):
            raise RuntimeError("Both binaries must report their source fingerprints")
        for key in ("api_version", "python_abi", "compiler", "pybind11_version", "build_type"):
            if old_info.get(key) != new_info.get(key):
                raise RuntimeError(f"Build environment mismatch: {key}")
        functions = {label: modules[label].cu_motion_peak for label in LABELS}
        report["boundary_validation"] = check_boundaries(functions)
        if report["boundary_validation"]["failures"]:
            raise RuntimeError("Boundary disagreement: performance sampling was not started")
        for size in args.sizes:
            row = benchmark_case(functions, size, args.warmup, args.samples)
            report["cases"].append(row)
            print(json.dumps({"input_width": size, "r3_p95_ms": row["metrics"]["r3"]["p95_ms"],
                "r4_p95_ms": row["metrics"]["r4"]["p95_ms"],
                "output_disagreements": row["output_disagreements_including_warmup"],
                "reference_mismatches": row["reference_mismatches_including_warmup"]}, allow_nan=False), flush=True)
        report["binary_files_unchanged"] = all(file_sha256(Path(item["path"])) == item["file_sha256"]
                                                  for item in report["binaries"].values())
        report["passed"] = report["binary_files_unchanged"] and all(
            row["input_unchanged"] and row["output_disagreements_including_warmup"] == 0
            and row["reference_mismatches_including_warmup"] == 0 for row in report["cases"])
        status = 0 if report["passed"] else 1
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
