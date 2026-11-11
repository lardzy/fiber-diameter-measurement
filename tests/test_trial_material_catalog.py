"""Library edits, trial ranges and explicit future-trial defaults are isolated."""
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from fdm.live_experiment.controller import LiveExperimentController
from fdm.live_experiment.material_catalog import MaterialCatalog
from fdm.live_experiment.models import DiameterRangePolicy, ExperimentSession, MaterialSnapshot
from fdm.live_experiment.reminders import ReminderPolicy
from fdm.live_experiment.trial_defaults import (new_trial_defaults, save_diameter_range_default,
                                              save_reminder_defaults)
from fdm.ui.material_catalog import DiameterRangeDialog, MaterialCatalogDialog, MaterialEditDialog, TrialMaterialsDialog
from fdm.ui.quantity_targets import QuantityTargetDialog


def session(mode="content"):
    trial = ExperimentSession(workspace_options={"mode": mode})
    trial.add_operator("测试员")
    catalog = MaterialCatalog()
    trial.set_materials([catalog.get("cashmere"), catalog.get("wool")])
    trial.establish_history_baseline()
    return trial


def test_catalog_rename_copy_delete_and_restart_leave_trial_snapshots_intact(tmp_path):
    catalog = MaterialCatalog()
    trial = session()
    before = trial.to_dict()
    old = catalog.get("cashmere")
    catalog.upsert(replace(old, name="细绒", aliases=("山羊绒",), upper_limit=26,
                           provenance={"parameter_source": "实验室确认"}))
    copied = catalog.duplicate("cashmere", "另一种绒")
    assert copied.id != old.id and copied.upper_limit == 26
    assert copied.provenance["copied_from_id"] == old.id
    path = tmp_path / "中文" / "类别库.json"
    catalog.save(path)
    restored = MaterialCatalog.load(path)
    assert restored.get("cashmere").aliases == ("山羊绒",)
    assert restored.get("cashmere").provenance["parameter_source"] == "实验室确认"
    with pytest.raises(ValueError, match="引用"):
        restored.delete("goat_hair")
    restored.delete("goat_hair", clear_references=True)
    assert restored.get("cashmere").upper_suggestion_id is None
    assert trial.to_dict() == before
    assert trial.material("cashmere").upper_suggestion_id == "goat_hair"
    restored.restore_builtin()
    assert restored.get("cashmere").upper_limit == 30
    assert restored.get(copied.id) is not None


def test_catalog_validation_cannot_partially_replace_definition(tmp_path):
    catalog = MaterialCatalog()
    old = catalog.get("wool")
    for invalid in (replace(old, name="山羊绒"), replace(old, upper_suggestion_id="unknown"),
                    replace(old, lower_suggestion_id="wool")):
        with pytest.raises(ValueError):
            catalog.upsert(invalid)
        assert catalog.get("wool") == old
    path = tmp_path / "bad.json"
    path.write_text('{"version":8,"materials":[]}')
    with pytest.raises(ValueError, match="格式"):
        MaterialCatalog.load(path)


@pytest.mark.parametrize("low,high", [(0, None), (-1, None), (True, None), (float("nan"), None),
                                       (None, float("inf")), (20, 10)])
def test_diameter_range_rejects_invalid_inputs(low, high):
    with pytest.raises(ValueError):
        DiameterRangePolicy(low, high)


def test_trial_range_inclusive_undo_serialization_and_mode_isolation(tmp_path):
    trial = session("diameter")
    controller = LiveExperimentController(trial, evidence_directory=tmp_path)
    intent = SimpleNamespace(material_id="cashmere", operator_id=trial.active_operator_id, direction="warp")
    try:
        # The content library's cashmere upper limit of 30 is irrelevant here.
        assert not controller._out_of_range(intent, {"diameter_um": 40})
        trial.configure(diameter_range_policy=DiameterRangePolicy(10, 35))
        assert controller._out_of_range(intent, {"diameter_um": 40})
        for value in (10, 35):
            assert not controller._out_of_range(intent, {"diameter_um": value})
        assert ExperimentSession.from_dict(trial.to_dict()).diameter_range_policy == DiameterRangePolicy(10, 35)
        trial.undo()
        assert trial.diameter_range_policy == DiameterRangePolicy()
        trial.redo()
        assert trial.diameter_range_policy == DiameterRangePolicy(10, 35)
        content = session()
        assert content.measurement_range("cashmere")["upper_limit_um"] == 30
        assert "diameter_trial" not in content.measurement_range("cashmere").values()
    finally:
        controller.close()


def test_named_suggestion_requires_configuration_and_never_mutates_trial(tmp_path):
    trial = session()
    controller = LiveExperimentController(trial, evidence_directory=tmp_path)
    intent = SimpleNamespace(material_id="cashmere", operator_id=trial.active_operator_id, direction="warp")
    catalog = MaterialCatalog()
    try:
        before = trial.to_dict()
        suggestion = controller.pending_suggestion(intent, {"diameter_um": 40}, catalog=catalog)
        assert suggestion == {"material_id": "goat_hair", "material_name": "山羊毛", "available": False, "range_status": "above"}
        assert trial.to_dict() == before
        trial.set_materials([*trial.operators[trial.active_operator_id].materials["warp"], catalog.get("goat_hair")])
        assert controller.pending_suggestion(intent, {"diameter_um": 40}, catalog=catalog)["available"]
        assert controller.pending_suggestion(intent, {"diameter_um": 30}, catalog=catalog) is None
    finally:
        controller.close()


def test_defaults_are_explicit_mode_scoped_and_preserve_range_when_goals_change(tmp_path):
    path = tmp_path / "defaults.json"
    assert new_trial_defaults("diameter", path=path)["diameter_range_policy"] == DiameterRangePolicy()
    content = ReminderPolicy(content_measured_target=12, content_total_target=50, material_targets={"wool": 17})
    save_reminder_defaults("content", content, path=path)
    save_diameter_range_default(DiameterRangePolicy(5, 20), path=path)
    save_reminder_defaults("diameter", ReminderPolicy(diameter_measured_target=22), path=path)
    assert new_trial_defaults("content", path=path)["reminder_policy"] == content
    diameter = new_trial_defaults("diameter", path=path)
    assert diameter["reminder_policy"].diameter_measured_target == 22
    assert diameter["diameter_range_policy"] == DiameterRangePolicy(5, 20)
    assert json.loads(path.read_text())["version"] == 1


def test_goal_dialog_cancel_does_not_write_but_explicit_default_does(tmp_path):
    trial = session()
    path = tmp_path / "defaults.json"
    before = trial.to_dict()
    dialog = QuantityTargetDialog(trial, defaults_file=path)
    try:
        dialog.default.setValue(44)
        dialog.reject()
        assert not path.exists()
        dialog.save_default.click()
        assert new_trial_defaults("content", path=path)["reminder_policy"].content_measured_target == 44
        assert trial.to_dict() == before
    finally:
        dialog.deleteLater()


def test_range_dialog_validates_and_only_explicit_button_updates_future_trials(tmp_path):
    trial = session("diameter")
    before = trial.to_dict()
    path = tmp_path / "defaults.json"
    dialog = DiameterRangeDialog(trial, defaults_file=path)
    try:
        assert dialog.lower.text() == dialog.upper.text() == ""
        dialog.lower.setText("30")
        dialog.upper.setText("20")
        dialog.accept()
        assert "下限" in dialog.status.text()
        assert not path.exists()
        dialog.lower.setText("10.25")
        dialog.save_default.click()
        assert new_trial_defaults("diameter", path=path)["diameter_range_policy"] == DiameterRangePolicy(10.25, 20)
        assert trial.to_dict() == before
    finally:
        dialog.deleteLater()


def test_material_dialog_only_edits_draft_and_trial_refuses_data_bearing_removal(tmp_path):
    trial = session()
    trial.add_measurement(material_id="cashmere", diameter_um=20)
    before = trial.to_dict()
    catalog = MaterialCatalog()
    dialog = TrialMaterialsDialog(trial, catalog=catalog)
    try:
        dialog.list.setCurrentRow(0)
        dialog.remove()
        assert len(dialog.materials()) == 2 and "已有记录" in dialog.status.text()
        dialog.move(1)
        assert [m.id for m in dialog.materials()] == ["wool", "cashmere"]
        dialog.reject()
        assert trial.to_dict() == before
        editor = MaterialEditDialog(catalog.get("cashmere"), catalog)
        editor.name.setText("试验临时名称")
        editor.aliases.setText("绒一，绒二,绒一")
        assert editor.material().aliases == ("绒一", "绒二")
        assert catalog.get("cashmere").name == "山羊绒"
        editor.deleteLater()
    finally:
        dialog.deleteLater()


def test_library_save_error_remains_visible_and_dialog_open(tmp_path, monkeypatch):
    dialog = MaterialCatalogDialog(path=tmp_path / "catalog.json")
    try:
        def fail(*args):
            raise OSError("只读目录")
        monkeypatch.setattr(dialog.catalog, "save", fail)
        dialog.accept()
        assert dialog.result() == 0
        assert "未保存" in dialog.status.text() and "只读目录" in dialog.status.text()
    finally:
        dialog.deleteLater()
