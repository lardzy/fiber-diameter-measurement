"""Check that the benchmark cannot report speed while hiding semantic drift."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_native_kernels.py"
SPEC = importlib.util.spec_from_file_location("fdm_native_benchmark_test_module", SCRIPT)
benchmark = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = benchmark
SPEC.loader.exec_module(benchmark)


def test_fixed_corpora_are_complete_and_heavy_group_is_predeclared():
    catalog = benchmark.case_catalog()
    assert len({spec.name for spec in catalog}) == 49
    assert sum(spec.category == "contour" for spec in catalog) == 13
    assert sum(spec.variant == "cross_stress" for spec in catalog) == 30
    heavy = [spec for spec in catalog if spec.heavy_quick]
    assert len(heavy) == 9
    assert {spec.ratio for spec in heavy} == {3, 4, 5}
    assert {spec.angle for spec in heavy} == {15, 25, 75}


def test_exact_array_comparison_rejects_single_pixel_or_dtype_change():
    first = np.zeros((12, 12), bool)
    changed = first.copy()
    changed[5, 7] = True
    with pytest.raises(AssertionError, match="values differ"):
        benchmark.assert_equivalent(first, changed)
    with pytest.raises(AssertionError, match="values differ"):
        benchmark.assert_equivalent(first, first.astype(np.uint8))


def test_float_comparison_uses_measurement_contract_tolerance():
    benchmark.assert_equivalent(0.0, 5e-11)
    with pytest.raises(AssertionError, match="floats differ"):
        benchmark.assert_equivalent(0.0, 5e-10)
    with pytest.raises(AssertionError, match="floats differ"):
        benchmark.assert_equivalent(100.0, 100.0 + 5e-9)


def test_semantic_comparison_ignores_timings_but_preserves_rejection_reason():
    before = {"outcome": "rejected", "code": "ambiguous_direction", "debug_payload": {"geometry_ms": 31.0, "candidate_count": 8}}
    after = {"outcome": "rejected", "code": "ambiguous_direction", "debug_payload": {"geometry_ms": 4.0, "candidate_count": 8}}
    benchmark.assert_equivalent(benchmark.semantic_view(before), benchmark.semantic_view(after))
    after["code"] = "unstable_width"
    with pytest.raises(AssertionError, match="code"):
        benchmark.assert_equivalent(benchmark.semantic_view(before), benchmark.semantic_view(after))


def test_paired_measurements_alternate_and_restore_requested_backend(monkeypatch):
    monkeypatch.setenv("FDM_NATIVE_BACKEND", "auto")
    calls = []

    def function():
        calls.append(os.environ["FDM_NATIVE_BACKEND"])
        return {"point": (3.0, 4.0)}

    result = benchmark.paired_stage(function, warmups=1, repeats=3)
    assert calls == ["python", "native", "native", "python", "python", "native", "native", "python"]
    assert len(result["samples"]) == 3
    assert result["samples"][0]["order"] == ["native", "python"]
    assert os.environ["FDM_NATIVE_BACKEND"] == "auto"


def test_paired_measurement_fails_on_native_drift_and_restores_environment(monkeypatch):
    monkeypatch.delenv("FDM_NATIVE_BACKEND", raising=False)

    def function():
        return 2 if os.environ["FDM_NATIVE_BACKEND"] == "native" else 1

    with pytest.raises(AssertionError, match="values differ"):
        benchmark.paired_stage(function, warmups=0, repeats=1)
    assert "FDM_NATIVE_BACKEND" not in os.environ


def fake_stage(python_ms, native_ms):
    samples = [{"python_ms": python_ms, "native_ms": native_ms} for _ in range(30)]
    return {"samples": samples, "summary": {"python": benchmark.stats([python_ms] * 30), "native": benchmark.stats([native_ms] * 30)}}


def test_fast_kernels_cannot_hide_missing_end_to_end_improvement():
    rows = []
    for spec in benchmark.case_catalog():
        rows.append({"name": spec.name, "stages": {"complete": fake_stage(10, 10), "pruner": fake_stage(9, 1)}})
    result = benchmark.acceptance(rows, warmups=5, repeats=30)
    checks = {check["name"]: check for check in result["checks"]}
    assert checks["quick_pruner_3x"]["status"] == "passed"
    assert checks["contour_complete_1.5x"]["status"] == "failed"
    assert checks["quick_heavy_complete_10pct"]["status"] == "failed"
    assert result["status"] == "failed"


def test_partial_or_smoke_run_cannot_claim_formal_acceptance():
    row = {"name": "contour_128_normal", "stages": {"complete": fake_stage(10, 1)}}
    result = benchmark.acceptance([row], warmups=0, repeats=1)
    assert result["status"] == "incomplete"
    assert not result["formal_sample_count_met"]
    assert any(check["status"] == "not_run" for check in result["checks"])
    row["stages"]["complete"] = fake_stage(1, 100)
    assert benchmark.acceptance([row], warmups=0, repeats=1)["status"] == "incomplete"


def test_corpus_speedup_cannot_hide_one_case_regression():
    rows = [{"name": spec.name, "stages": {"complete": fake_stage(10, 1), "pruner": fake_stage(9, 1)}} for spec in benchmark.case_catalog()]
    rows[0]["stages"]["complete"] = fake_stage(10, 11)
    result = benchmark.acceptance(rows, warmups=5, repeats=30)
    assert result["status"] == "failed"
    check = next(check for check in result["checks"] if check["name"] == "complete_no_regression:contour_128_normal")
    assert check["status"] == "failed"


def test_input_digest_covers_layout_type_and_parameters():
    data = np.ones((2, 6), dtype=np.uint8)
    first = benchmark.digest_inputs([data], {"scale": 1})
    assert first != benchmark.digest_inputs([data.reshape(3, 4)], {"scale": 1})
    assert first != benchmark.digest_inputs([data.astype(bool)], {"scale": 1})
    assert first != benchmark.digest_inputs([data], {"scale": 2})
