"""Actual Qt workspace + domain policy, with Excel deliberately stubbed out."""
from concurrent.futures import Future
from dataclasses import replace
import time

import pytest
from PySide6.QtCore import QObject, Signal, Qt
from PySide6.QtWidgets import QApplication, QDialog, QInputDialog, QMessageBox

from fdm.live_experiment.controller import LiveExperimentController
from fdm.live_experiment.material_catalog import MaterialCatalog
from fdm.live_experiment.models import DiameterRangePolicy
from fdm.ui.live_experiment_workspace import CandidatesModel
from test_live_experiment_content_workspace import workspace, make_session, present, ready_result


class OfflineExcel(QObject):
    changed = Signal(str)

    def ensure(self, session):
        pass

    update = ensure
    resync = ensure
    show = ensure
    stop = ensure

    def status(self, session):
        return {"state": "idle", "message": "测试不连接 Excel", "active": False, "pending": False}

    def detach(self, *args, **kwargs):
        return True

    flush = detach
    close = detach


@pytest.fixture(autouse=True)
def no_excel_process(monkeypatch):
    from fdm.ui import trial_excel_lifecycle
    def manager(host):
        if not hasattr(host, "_trial_live_excel"):
            host._trial_live_excel = OfflineExcel(host)
            host._trial_excel_sessions = {}
        return host._trial_live_excel
    monkeypatch.setattr(trial_excel_lifecycle, "live_excel_manager", manager)
    monkeypatch.setattr(MaterialCatalog, "load", classmethod(lambda cls, path=None: cls()))


def _pending(workspace, diameter=40):
    controller = workspace.controller
    controller.start((30, 30))
    controller.finish((30, 30 + diameter), "cashmere", method="free")
    deadline = time.perf_counter() + 2
    while controller._jobs and time.perf_counter() < deadline:
        controller.poll()
        QApplication.processEvents()
        time.sleep(.001)
    assert controller.pending_classification
    workspace._refresh_summary()
    return controller.pending_classification[0][0]


def test_pending_suggestion_names_missing_target_then_adds_without_losing_measurement(workspace, monkeypatch):
    intent = _pending(workspace)
    assert "山羊毛" in workspace.pending_suggestion.text()
    assert "先配置" in workspace.pending_suggestion.text()
    from fdm.ui.material_catalog import TrialMaterialsDialog
    def add_target(dialog):
        dialog.set_selected("goat_hair", True)
        return QDialog.DialogCode.Accepted
    monkeypatch.setattr(TrialMaterialsDialog, "exec", add_target)
    workspace._materials()
    assert workspace.controller.pending_classification[0][0] is intent
    assert not workspace.session.measurements
    assert workspace.session.material("goat_hair").name == "山羊毛"
    workspace._refresh_summary()
    assert workspace.pending_suggestion.text() == "改为“山羊毛”"
    workspace._accept_suggested_material()
    workspace.controller.poll()
    record = next(iter(workspace.session.measurements.values()))
    assert record.material_id == "goat_hair"
    assert record.provenance["range_decision"] == "reclassified"
    assert record.provenance["range_reclassifications"][0]["from"]["upper_limit_um"] == 30


def test_reclassifying_to_still_outside_requires_second_explicit_confirmation(workspace):
    materials = list(workspace.session.operators[workspace.session.active_operator_id].materials["warp"])
    target = replace(MaterialCatalog().get("goat_hair"), upper_limit=35)
    workspace.session.set_materials([*materials, target])
    workspace._refresh_materials()
    intent = _pending(workspace)
    workspace._accept_suggested_material()
    assert not workspace.session.measurements
    assert workspace.controller.pending_classification[0][0] is intent
    assert intent.material_id == "goat_hair" and intent.range_confirmation is None
    workspace._refresh_summary()
    assert "35" in workspace.pending_label.text()
    workspace._resolve_displayed_pending()
    workspace.controller.poll()
    record = next(iter(workspace.session.measurements.values()))
    assert record.material_id == "goat_hair"
    assert record.provenance["range_decision"] == "keep_outside"
    assert record.provenance["range_confirmation"]["upper_limit_um"] == 35


def test_pending_material_cannot_be_removed_by_dialog_patch(workspace, monkeypatch):
    intent = _pending(workspace)
    from fdm.ui.material_catalog import TrialMaterialsDialog
    monkeypatch.setattr(TrialMaterialsDialog, "exec", lambda _: QDialog.DialogCode.Accepted)
    monkeypatch.setattr(TrialMaterialsDialog, "materials", lambda _: (workspace.session.material("wool"),))
    with pytest.raises(ValueError):
        workspace._materials()
    assert workspace.controller.pending_classification[0][0] is intent
    assert workspace.session.material("cashmere")


def test_materials_apply_to_captured_owner_after_active_owner_changes(workspace, monkeypatch):
    from fdm.ui.material_catalog import TrialMaterialsDialog
    session = workspace.session
    original = session.active_operator_id
    other = session.add_operator("另一位操作员")
    session.active_operator_id = original
    old_other = session.operators[other].materials["weft"]
    def choose(dialog):
        assert dialog.operator_id == original and dialog.direction == "warp"
        dialog.set_selected("cotton", True)
        session.active_operator_id, session.active_direction = other, "weft"
        dialog.accept()
        return dialog.result()
    monkeypatch.setattr(TrialMaterialsDialog, "exec", choose)
    workspace._materials()
    assert session.operators[original].materials["warp"][-1].id == "cotton"
    assert session.operators[other].materials["weft"] == old_other


def test_materials_unchanged_confirmation_does_not_emit_domain_update(workspace, monkeypatch):
    from fdm.ui.material_catalog import TrialMaterialsDialog
    before = workspace.session.to_dict()
    monkeypatch.setattr(TrialMaterialsDialog, "exec", lambda dialog: QDialog.DialogCode.Accepted)
    with monkeypatch.context() as changes:
        changes.setattr(workspace, "_changed", lambda: pytest.fail("unchanged selection updated Excel/project"))
        workspace._materials()
    assert workspace.session.to_dict() == before


def test_diameter_candidates_use_trial_limits_and_consent_invalidates_when_changed():
    session = make_session()
    session.configure(workspace_options={"mode": "diameter"})
    model = CandidatesModel()
    model.session = session
    model.set_candidates([{"width_px": 40, "start": [0, 0], "end": [0, 40]}], 1,
                         session.operators[session.active_operator_id].materials["warp"])
    model.setData(model.index(0, 3), "cashmere", Qt.ItemDataRole.EditRole)
    assert model.range_status(0) == "范围内"  # Ignore the fiber's 30 µm limit.
    session.configure(diameter_range_policy=DiameterRangePolicy(5, 35))
    assert model.range_status(0) == "越限，需核对"
    model.setData(model.index(0, 5), Qt.CheckState.Checked, Qt.ItemDataRole.CheckStateRole)
    assert model.range_confirmed(0)
    session.configure(diameter_range_policy=DiameterRangePolicy(5, 39))
    assert not model.range_confirmed(0)
    assert "39" in model.range_description(0)


def test_changed_diameter_range_during_motion_confirmation_returns_pending_without_losing_lease(tmp_path):
    session = make_session()
    session.configure(workspace_options={"mode": "diameter"}, diameter_range_policy=DiameterRangePolicy(None, 30))
    controller = LiveExperimentController(session, evidence_directory=tmp_path)
    try:
        ready_result(controller)
        intent, future = controller._jobs[0]
        done = Future()
        done.set_result(dict(future.result(), width_px=40, diameter_um=40))
        controller._jobs[0] = (intent, done)
        controller._motion_sequence = 1
        controller.poll()
        assert controller.pending_classification
        present(controller, 10)
        controller.resolve_pending(intent_id=intent.id)
        assert controller._confirmation and intent.range_confirmation["upper_limit_um"] == 30
        session.configure(diameter_range_policy=DiameterRangePolicy(None, 35))
        controller._motion_sequence = 10
        controller.poll()
        assert controller._confirmation is None
        assert controller.pending_classification[0][0] is intent
        assert intent.range_confirmation is None and not session.measurements
        controller.resolve_pending(intent_id=intent.id)
        record = next(iter(session.measurements.values()))
        assert record.provenance["measurement_range"] == {
            "scope": "diameter_trial", "lower_limit_um": None, "upper_limit_um": 35.0, "diameter_um": 40}
        assert record.provenance["range_decision"] == "keep_outside"
    finally:
        controller.close()


@pytest.mark.parametrize("mode", ["content", "diameter"])
def test_record_revision_checks_range_and_keeps_original_evidence_with_one_undo(workspace, monkeypatch, mode):
    session = workspace.session
    session.configure(workspace_options={"mode": mode}, diameter_range_policy=DiameterRangePolicy(None, 35))
    record = session.add_measurement(20, "cashmere", direction="weft", provenance={
        "original_diameter_um": 20, "measurement_range": {"upper_limit_um": 30, "diameter_um": 20}})
    monkeypatch.setattr(workspace, "_selected_record", lambda: session.measurements[record.id])
    monkeypatch.setattr(QInputDialog, "getDouble", lambda *_: (40, True))
    def choose(*args):
        assert mode == "content", "直径试验修订不应要求选择纤维类别"
        return args[3][1], True
    monkeypatch.setattr(QInputDialog, "getItem", choose)
    prompts = []
    def cancel(*args):
        prompts.append(args[2])
        return QMessageBox.StandardButton.No
    monkeypatch.setattr(QMessageBox, "question", cancel)
    before = session.to_dict()
    workspace._edit_record()
    assert session.to_dict() == before
    assert "纬向" in prompts[-1] and "40" in prompts[-1]
    monkeypatch.setattr(QMessageBox, "question", lambda *_: QMessageBox.StandardButton.Yes)
    workspace._edit_record()
    revised = session.measurements[record.id]
    assert revised.diameter_um == 40
    assert revised.provenance["original_diameter_um"] == 20
    assert revised.provenance["measurement_range"] == record.provenance["measurement_range"]
    revision = revised.provenance["range_revisions"][-1]
    assert revision["decision"] == "keep_outside" and revision["operation"] == "manual_revision"
    assert revision["measurement_range"]["upper_limit_um"] == (30 if mode == "content" else 35)
    session.undo()
    assert session.measurements[record.id] == record


def test_record_revision_cannot_apply_stale_consent_when_range_changes_in_dialog(workspace, monkeypatch):
    session = workspace.session
    session.configure(workspace_options={"mode": "diameter"}, diameter_range_policy=DiameterRangePolicy(None, 30))
    record = session.add_measurement(20, "wool")
    monkeypatch.setattr(workspace, "_selected_record", lambda: session.measurements[record.id])
    monkeypatch.setattr(QInputDialog, "getDouble", lambda *_: (40, True))
    def confirm(*args):
        session.configure(diameter_range_policy=DiameterRangePolicy(None, 25))
        return QMessageBox.StandardButton.Yes
    monkeypatch.setattr(QMessageBox, "question", confirm)
    with pytest.raises(ValueError, match="范围已变化"):
        workspace._edit_record()
    assert session.measurements[record.id] == record


def test_postselection_records_new_range_without_rewriting_original_acceptance(tmp_path):
    session = make_session()
    controller = LiveExperimentController(session, evidence_directory=tmp_path)
    try:
        ready_result(controller)
        controller._motion_sequence = 1
        controller.poll()
        record = next(iter(session.measurements.values()))
        assert record.material_id == "wool"
        controller.reclassify_latest("cashmere")
        revised = session.measurements[record.id]
        assert revised.material_id == "cashmere"
        assert revised.provenance["measurement_range"] == record.provenance["measurement_range"]
        history = revised.provenance["range_revisions"][-1]
        assert history["previous_material_id"] == "wool" and history["operation"] == "post_classification"
        assert history["measurement_range"]["upper_limit_um"] == 30 and history["decision"] == "within"
        session.undo()
        assert session.measurements[record.id] == record
    finally:
        controller.close()
