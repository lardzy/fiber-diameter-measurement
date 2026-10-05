"""Small deterministic probes that require the shipped native implementation."""
from __future__ import annotations

from importlib.machinery import EXTENSION_SUFFIXES
import re


NATIVE_PROBE_CASES = frozenset({"boundary_segments", "contour_sections", "prune_short_spurs", "inputs_unchanged"})


def native_probe_passed(report: object) -> bool:
    """Validate the release probe independently of its aggregate success flag."""
    if not isinstance(report, dict):
        return False
    info, cases = report.get("build_info"), report.get("cases")
    return (
        report.get("ok") is True
        and report.get("backend") == "native"
        and report.get("api_version") == 1
        and report.get("compiled_extension") is True
        and isinstance(info, dict)
        and info.get("api_version") == 1
        and all(isinstance(info.get(key), str) and bool(info[key]) for key in ("compiler", "python_abi", "pybind11_version"))
        and isinstance(info.get("source_sha256"), str)
        and re.fullmatch(r"[0-9a-f]{64}", info["source_sha256"]) is not None
        and isinstance(cases, dict)
        and all(cases.get(name) is True for name in NATIVE_PROBE_CASES)
    )


def run_native_kernels_self_check(*, validate_source: bool = False) -> dict[str, object]:
    """Call all three C++ entrypoints even when source debug mode requests Python."""
    import numpy as np

    from fdm._native import require_native

    kernels = require_native(validate_source=validate_source)
    info = kernels.build_info()
    compiled = any(str(kernels.__file__).endswith(suffix) for suffix in EXTENSION_SUFFIXES)
    mask = np.zeros((3, 3), dtype=np.bool_)
    mask[1, 1] = True
    expected_edges = np.asarray([
        [[1, 1], [2, 1]], [[1, 2], [2, 2]],
        [[1, 1], [1, 2]], [[2, 1], [2, 2]],
    ], dtype=np.float64)
    edges = kernels.boundary_segments(mask)
    heights = np.asarray([0.5, 1.5, 2.5], dtype=np.float64)
    offsets, intervals = kernels.contour_sections(expected_edges, heights, 1.0)
    skeleton = np.zeros((24, 28), dtype=np.bool_)
    skeleton[12, 2:26] = True
    skeleton[9:12, 13] = True
    before_skeleton = skeleton.copy()
    distance_map = np.full(skeleton.shape, 2.0, dtype=np.float32)
    pruned = kernels.prune_short_spurs(skeleton, distance_map)
    expected_skeleton = before_skeleton.copy()
    expected_skeleton[9:12, 13] = False
    cases = {
        "boundary_segments": edges.dtype == np.float64 and np.array_equal(edges, expected_edges),
        "contour_sections": np.array_equal(offsets, [0, 0, 1, 1]) and np.array_equal(intervals, [[1.0, 2.0]]),
        "prune_short_spurs": pruned.dtype == np.bool_ and np.array_equal(pruned, expected_skeleton),
        "inputs_unchanged": int(mask.sum()) == 1 and bool(mask[1, 1])
            and np.array_equal(skeleton, before_skeleton)
            and bool(np.all(distance_map == 2.0))
            and np.array_equal(heights, [0.5, 1.5, 2.5]),
    }
    report = {
        "ok": compiled and all(cases.values()),
        "backend": "native",
        "api_version": info.get("api_version"),
        "compiled_extension": compiled,
        "build_info": info,
        "cases": {name: bool(value) for name, value in cases.items()},
    }
    report["ok"] = native_probe_passed(report)
    return report
