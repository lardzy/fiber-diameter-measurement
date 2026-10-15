"""Receipt-phase audit regressions; synthetic JSON only, no benchmark workload."""
from copy import deepcopy
from pathlib import Path

import pytest

from scripts.validation.audit_cu_live_performance import (
    TEARDOWN_HARNESS_SHA256, background_completion, ui_cases,
)


def document(sha="phase-instrumented-harness"):
    return {"environment": {"source_sha256": {"scripts/benchmark_cu_live_ui.py": sha}}}


def response(task, phase, received, status="ok", candidates=1):
    return {"task_id": task, "phase": phase, "received_monotonic": received,
            "status": status, "candidates": candidates, "error": None}


def traced_case():
    return {"background_automatic": "traditional",
        "background_timeline": {"phase_schema_version": 1, "clock": "time.perf_counter",
            "warmup_started_at": 10.0, "formal_started_at": 20.0,
            "teardown_started_at": 30.0, "close_requested_task_id": "closing-task"},
        "background_warmup_results": [response("warmup-task", "warmup", 15.0)],
        "background_results": [response("formal-task", "formal", 25.0),
            response("closing-task", "teardown", 31.0, "cancelled", 0)]}


def test_explicit_teardown_uses_receipt_phase_and_matching_closed_task():
    result = background_completion(document(), traced_case())
    assert result["classification"] == "EXPLICIT_TEARDOWN"
    assert result["phase_trace_valid"]
    assert result["teardown_cancel_explicit"] == 1
    assert result["teardown_cancel_inferred"] == 0
    assert result["normal_completed"] == 1
    assert result["warmup_normal_completed"] == 1
    assert result["total_responses"] == 3
    assert result["cancelled_total"] == 1
    assert result["background_errors"] == result["unexplained_cancelled"] == 0


@pytest.mark.parametrize("change", [
    "wrong_task", "wrong_close_task", "formal_cancel", "before_teardown", "missing_phase",
    "missing_time", "backwards_time", "duplicate_task", "missing_clock", "wrong_schema",
    "nonfinite_time", "nonzero_candidates", "error", "absent_error", "second_cancel",
])
def test_phase_trace_never_blanket_exempts_cancelled_responses(change):
    case = traced_case()
    last = case["background_results"][-1]
    if change == "wrong_task":
        last["task_id"] = "some-other-task"
    elif change == "wrong_close_task":
        case["background_timeline"]["close_requested_task_id"] = None
    elif change == "formal_cancel":
        last.update(phase="formal", received_monotonic=29.0)
    elif change == "before_teardown":
        last["received_monotonic"] = 29.0
    elif change == "missing_phase":
        del last["phase"]
    elif change == "missing_time":
        del last["received_monotonic"]
    elif change == "backwards_time":
        case["background_results"][0]["received_monotonic"] = 32.0
    elif change == "duplicate_task":
        case["background_results"][0]["task_id"] = "closing-task"
    elif change == "missing_clock":
        del case["background_timeline"]["clock"]
    elif change == "wrong_schema":
        case["background_timeline"]["phase_schema_version"] = 2
    elif change == "nonfinite_time":
        last["received_monotonic"] = float("inf")
    elif change == "nonzero_candidates":
        last["candidates"] = 1
    elif change == "error":
        last["error"] = {"message": "worker failed"}
    elif change == "absent_error":
        del last["error"]
    else:
        case["background_results"].insert(1, response("bad-formal", "formal", 27.0, "cancelled", 0))
    # Even the old reviewed SHA cannot override an invalid new phase trace.
    result = background_completion(document(TEARDOWN_HARNESS_SHA256), case)
    assert result["teardown_cancel_explicit"] == result["teardown_cancel_inferred"] == 0
    assert result["background_errors"] >= 1
    assert result["unexplained_cancelled"] >= 1


def test_warmup_or_teardown_success_does_not_count_as_formal_completion():
    case = traced_case()
    case["background_results"] = [response("closing-task", "teardown", 31.0)]
    result = background_completion(document(), case)
    assert result["normal_completed"] == 0
    assert result["warmup_normal_completed"] == result["teardown_normal_completed"] == 1
    doc = document()
    doc["cases"] = [case]
    checks = {item["name"]: item for item in ui_cases(Path("synthetic.json"), doc)[0]["checks"]}
    assert checks["background_executed"]["status"] == "FAIL"


def test_warmup_error_is_retained_and_prevents_teardown_exemption():
    case = traced_case()
    case["background_warmup_results"][0].update(status="cancelled", candidates=0)
    result = background_completion(document(), case)
    assert result["warmup_error_indices"] == [0]
    assert result["cancelled_total"] == result["unexplained_cancelled"] == 2
    assert result["background_errors"] == 2


def test_invalid_trace_fails_audit_even_when_all_responses_succeeded():
    case = traced_case()
    case["background_results"].pop()
    case["background_results"][0]["received_monotonic"] = 5.0
    doc = document()
    doc["cases"] = [case]
    row = ui_cases(Path("synthetic.json"), doc)[0]
    checks = {item["name"]: item for item in row["checks"]}
    assert checks["background_errors"]["status"] == "PASS"
    assert checks["background_phase_trace_valid"]["status"] == "FAIL"


def test_legacy_exact_sha_and_last_unique_cancel_rule_is_preserved():
    case = {"background_results": [
        {"status": "ok", "candidates": 1, "error": None},
        {"status": "cancelled", "candidates": 0, "error": None}]}
    original = deepcopy(case)
    result = background_completion(document(TEARDOWN_HARNESS_SHA256), case)
    assert result["teardown_cancel_inferred"] == 1 and result["teardown_cancel_explicit"] == 0
    assert result["background_errors"] == 0 and result["phase_trace_valid"] is None
    assert case == original
    assert background_completion(document("unknown"), case)["background_errors"] == 1
    case["background_results"].append({"status": "empty", "candidates": 0, "error": None})
    assert background_completion(document(TEARDOWN_HARNESS_SHA256), case)["background_errors"] == 1
