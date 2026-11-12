from copy import deepcopy
import json
import subprocess

import pytest

from fdm.validation.trial_template_flow import cleanup_failures, view_differences


def snapshot():
    return {"active_sheet": "特种毛(报出)", "sheets": [
        {"name": "原始数据", "visible": -1, "shapes": 0, "sample": []},
        {"name": "特种毛(报出)", "visible": -1, "shapes": 2,
         "sample": [{"formula": "=原始数据!D25", "font_size": 12.0}]},
        {"name": "织物纤维含量报告", "visible": -1, "shapes": 1, "sample": []}]}


def test_visibility_or_active_page_regression_fails_even_when_data_is_correct():
    before = snapshot()
    after = deepcopy(before)
    after["active_sheet"] = "原始数据"
    after["sheets"][1]["visible"] = 0
    after["first"] = 18.25
    assert set(view_differences(before, after)) == {"active worksheet changed", "特种毛(报出): visible changed"}


def test_formula_and_drawing_changes_are_reported_separately():
    before = snapshot()
    after = deepcopy(before)
    after["sheets"][1]["sample"][0]["formula"] = "0"
    after["sheets"][2]["shapes"] = 0
    assert set(view_differences(before, after)) == {"特种毛(报出): sample changed", "织物纤维含量报告: shapes changed"}


def test_measurement_values_and_workbook_names_are_not_template_layout_changes():
    before = snapshot()
    after = deepcopy(before)
    after.update(name="new workbook", first=18.25, second=25.5)
    assert view_differences(before, after) == []


def test_acceptance_requires_final_sync_and_confirmed_empty_owned_excel_instances():
    assert cleanup_failures({"cleanup": [{"remaining": 0}, {"remaining": 0}]}) == []
    report = {"close_error": "final inputs timed out", "cleanup": [
        {"remaining": 0}, {"error": "identity mismatch"}, {"remaining": 1}, {}]}
    problems = cleanup_failures(report)
    assert len(problems) == 4
    assert "final inputs timed out" in problems[0]
    assert "identity mismatch" in problems[1]
    assert "1" in problems[2]
    assert "None" in problems[3]


def test_explicit_application_entry_routes_arguments_before_normal_window_start(monkeypatch):
    from fdm import app
    from fdm.validation import trial_template_flow
    received = []
    monkeypatch.setattr(trial_template_flow, "main", lambda args: received.append(args) or 7)
    assert app.main(["fdm", "--verify-trial-template-flow", "--output", "isolated-run", "--require-installed"]) == 7
    assert received == [["--output", "isolated-run", "--require-installed"]]


def test_probe_timeout_without_independent_owner_never_stops_excel(tmp_path, monkeypatch):
    from fdm.validation.trial_template_flow import ExcelProbe
    calls = []

    def timeout(command, **kwargs):
        calls.append(command)
        raise subprocess.TimeoutExpired(command, 90, stderr=b"probe still waiting")

    monkeypatch.setattr(subprocess, "run", timeout)
    with pytest.raises(RuntimeError, match="timed out"):
        ExcelProbe(tmp_path).call("read", excel_pid=11784, excel_created="user-owned", name="User report")
    assert len(calls) == 1
    diagnostic = json.loads((tmp_path / "timeout-1.json").read_text())
    assert not diagnostic["cleanup"]["cleaned"]
    assert "no Excel process touched" in diagnostic["cleanup"]["reason"]
    assert (tmp_path / "timeout-1.stderr.txt").read_bytes() == b"probe still waiting"


def test_probe_timeout_runs_guarded_cleanup_only_with_its_owner_record(tmp_path, monkeypatch):
    from fdm.validation.trial_template_flow import ExcelProbe
    calls = []
    owner = tmp_path / "request-1.json.owner.json"

    def execute(command, **kwargs):
        calls.append(command)
        if len(calls) == 1:
            owner.write_text(json.dumps({"validation_probe": True, "id": 100, "created": "123",
                                         "operation": "variant", "preexisting_excel_ids": [11784]}))
            raise subprocess.TimeoutExpired(command, 90)
        assert command[-2:] == ["-OwnerPath", str(owner)]
        return subprocess.CompletedProcess(command, 0, stdout=b'{"cleaned":true,"forced":true,"id":100}', stderr=b"")

    monkeypatch.setattr(subprocess, "run", execute)
    with pytest.raises(RuntimeError, match="timed out"):
        ExcelProbe(tmp_path).call("variant", source="copy.xlt", destination="fixture.xlt")
    assert len(calls) == 2
    diagnostic = json.loads((tmp_path / "timeout-1.json").read_text())
    assert diagnostic["cleanup"] == {"cleaned": True, "forced": True, "id": 100}
