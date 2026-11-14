"""Verified content results outlive preview motion, with session-only range choices."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
from PySide6.QtWidgets import QApplication

from fdm.live_experiment import default_materials
from fdm.live_experiment.controller import CalibrationSnapshot, LiveExperimentController
from fdm.live_experiment.selection_retention import FieldEndCause
from test_live_experiment_runtime import session, frame, present, drain


@pytest.fixture
def controller(tmp_path):
    value = session()
    value.set_materials([replace(default_materials()[0], lower_limit=10, upper_limit=30),
                         replace(default_materials()[1], lower_limit=1, upper_limit=70)])
    result = LiveExperimentController(value, evidence_directory=tmp_path)
    result.set_calibration(CalibrationSnapshot(1, 1))
    present(result, frame())
    yield result
    result.close()
    assert result.budget.used_bytes == 0


def measure(controller, diameter=40, category="wool"):
    controller.start((10, 10))
    controller.finish((10, 10 + diameter), category, method="free")
    drain(controller)


def test_verified_decision_survives_motion_timeout_and_disconnect(controller):
    original_field = controller.field_id
    source = controller.presented.frame
    measure(controller)
    decision = controller.pending_range_decision
    assert decision.result["diameter_um"] == 40 and not controller.session.measurements
    decision.intent.submitted_at -= 5
    controller.end_field(cause=FieldEndCause.MOTION)
    present(controller, frame(1))
    controller.unavailable("camera disconnected")
    controller.poll()
    assert controller.pending_range_decision is decision
    assert source.frame_id in controller.budget._frames
    controller.resolve_range_decision("keep", intent_id=decision.intent.id)
    record = next(iter(controller.session.measurements.values()))
    assert record.field_id == original_field and record.frame_id == source.frame_id
    assert record.diameter_um == 40 and record.provenance["range_decision_source"] == "manual"
    assert record.provenance["calibration"]["x_um"] == 1
    assert controller.current_records == []
    assert not controller.has_range_decisions and controller.budget.used_bytes == 0
    assert controller.session.fields[original_field]["end_reason"] == "视场已变化，请重新测量"
    field = controller.session.fields[original_field]
    assert field["opened_at"] <= field["ended_at"] <= record.created_at


def test_keep_and_reclassify_rules_are_direction_specific_and_do_not_change_selection(controller):
    measure(controller)
    controller.resolve_range_decision("reclassify", material_id="cashmere", remember=True)
    before = controller.session.revision
    measure(controller, 45)
    assert not controller.pending_range_decision
    records = list(controller.session.measurements.values())
    assert [r.material_id for r in records] == ["cashmere", "cashmere"]
    assert records[-1].provenance["range_decision_source"] == "remembered_rule"
    measure(controller, 20)
    assert list(controller.session.measurements.values())[-1].material_id == "wool"
    measure(controller, 5)
    assert controller.pending_range_decision.direction == "below"
    controller.resolve_range_decision("discard")
    assert len(controller.session.measurements) == 3
    assert controller.session.revision > before


def test_rules_are_runtime_only_and_reused_by_recreated_controller(controller, tmp_path):
    measure(controller)
    controller.resolve_range_decision("keep", remember=True)
    state = controller.session.to_dict()
    count = len(controller.range_rules.list_rules(controller.session.id))
    revision = controller.session.revision
    controller.range_rules.remove(controller.session.id, ("wool", "above"))
    assert controller.session.to_dict() == state and controller.session.revision == revision
    assert count == 1
    measure(controller)
    controller.resolve_range_decision("keep", remember=True)
    second = LiveExperimentController(controller.session, evidence_directory=tmp_path / "second")
    try:
        assert second.range_rules is controller.range_rules
        assert len(second.range_rules.list_rules(controller.session.id)) == 1
        restored = type(controller.session).from_dict(controller.session.to_dict())
        assert not hasattr(restored, "_range_rules")
    finally:
        second.close()


def test_limit_change_invalidates_rule_and_requires_explicit_new_confirmation(controller):
    measure(controller)
    controller.resolve_range_decision("keep", remember=True)
    materials = controller.session.operators[controller.session.active_operator_id].materials["warp"]
    controller.session.set_materials([replace(materials[0], upper_limit=35), materials[1]])
    measure(controller)
    assert controller.pending_range_decision
    assert not controller.range_rules.list_rules(controller.session.id)
    controller.session.set_materials([replace(materials[0], upper_limit=38), materials[1]])
    assert not controller.resolve_range_decision("keep")
    assert len(controller.session.measurements) == 1
    assert controller.resolve_range_decision("keep")


def test_target_limit_and_availability_invalidate_automatic_reclassification(controller):
    measure(controller)
    controller.resolve_range_decision("reclassify", material_id="cashmere", remember=True)
    materials = controller.session.operators[controller.session.active_operator_id].materials["warp"]
    controller.session.set_materials([materials[0], replace(materials[1], upper_limit=35)])
    measure(controller)
    assert controller.pending_range_decision
    assert not controller.resolve_range_decision("reclassify", material_id="cashmere", remember=True)
    assert controller.pending_range_decision.requires_explicit_keep
    assert controller.materials_in_use(operator_id=controller.session.active_operator_id,
        direction="warp") == {"wool", "cashmere"}
    assert controller.materials_in_use(direction="weft") == set()
    assert len(controller.session.measurements) == 1
    controller.resolve_range_decision("keep", remember=True)
    assert not controller.range_rules.list_rules(controller.session.id)


def test_discard_has_no_ledger_revision_evidence_or_undo(controller):
    measure(controller)
    before = controller.session.to_dict()
    undo = len(controller.session._undo)
    controller.resolve_range_decision("discard", remember=True)
    assert controller.session.to_dict() == before
    assert len(controller.session._undo) == undo
    assert not controller.evidence.pending
    measure(controller)
    assert not controller.pending_range_decision and not controller.session.measurements


def prepare_candidates(controller, widths):
    controller._automatic = SimpleNamespace(cancel=lambda: None, close=lambda: None)
    controller._automatic_context = dict(token=controller.presented, calibration=controller.calibration,
        algorithm="traditional", operator_id=controller.session.active_operator_id, direction="warp",
        lease=controller.budget.acquire(controller.presented.frame), diagnostics={})
    controller.candidates = [dict(width_px=width, start=[20 + i * 20, 20], end=[20 + i * 20, 20 + width])
                             for i, width in enumerate(widths)]


def test_selected_mixed_batch_remains_atomic_across_motion_and_discard(controller):
    prepare_candidates(controller, [20, 40, 50])
    origin = controller.field_id
    undo = len(controller.session._undo)
    controller.accept_candidates([(0, "wool"), (1, "wool"), (2, "wool")])
    assert controller.range_decision_count == 2
    controller.end_field(cause=FieldEndCause.MOTION)
    present(controller, frame(1))
    controller.resolve_range_decision("keep")
    assert not controller.session.measurements
    controller.resolve_range_decision("discard")
    records = list(controller.session.measurements.values())
    assert [item.diameter_um for item in records] == [20, 40]
    assert all(item.field_id == origin for item in records)
    assert not controller.current_records
    assert len(controller.session._undo) == undo + 1
    controller.session.undo()
    assert not controller.session.measurements
    controller.session.redo()
    assert len(controller.session.measurements) == 2
    with pytest.raises(ValueError):
        controller.resolve_range_decision("keep")


def test_unselected_candidates_do_not_prompt_and_pending_blocks_new_input(controller):
    prepare_candidates(controller, [20, 40])
    controller.accept_candidates([(0, "wool")])
    assert len(controller.session.measurements) == 1 and not controller.has_range_decisions
    controller.accept_candidates([(1, "wool")])
    with pytest.raises(ValueError, match="越限"):
        controller.start((10, 10))
    with pytest.raises(ValueError, match="越限"):
        controller.accept_candidates([(0, "wool")])
    with pytest.raises(ValueError, match="越限"):
        controller.reclassify_latest("cashmere")
    controller.resolve_range_decision("discard")


def test_original_frame_and_frozen_archive_policy_are_used_after_motion(controller):
    import time
    import cv2
    import numpy as np
    controller.set_evidence_options({"mode": "full_png"})
    source = controller.presented.frame
    measure(controller)
    controller.set_evidence_options({"mode": "disabled"})
    controller.end_field(cause=FieldEndCause.MOTION)
    present(controller, frame(2, pixels=np.zeros_like(source.pixels)))
    controller.resolve_range_decision("keep")
    record = next(iter(controller.session.measurements.values()))
    deadline = time.monotonic() + 3
    while record.evidence["state"] == "pending" and time.monotonic() < deadline:
        controller._poll_evidence()
        record = controller.session.measurements[record.id]
        QApplication.processEvents()
        time.sleep(.002)
    assert record.evidence["state"] == "saved"
    assert record.evidence["frame_id"] == source.frame_id
    assert np.array_equal(cv2.imread(record.evidence["path"], cv2.IMREAD_UNCHANGED), source.pixels)
    restored = type(controller.session).from_dict(controller.session.to_dict())
    assert restored.measurements[record.id].provenance == record.provenance


def test_batch_keep_reclassify_discard_is_one_transaction(controller):
    prepare_candidates(controller, [40, 45, 50])
    before = len(controller.session._undo)
    controller.accept_candidates([(i, "wool") for i in range(3)])
    controller.resolve_range_decision("keep")
    controller.resolve_range_decision("reclassify", material_id="cashmere")
    assert not controller.session.measurements
    controller.resolve_range_decision("discard")
    assert [(record.diameter_um, record.material_id) for record in controller.session.measurements.values()] == [
        (40, "wool"), (45, "cashmere")]
    assert len(controller.session._undo) == before + 1
    controller.session.undo()
    assert not controller.session.measurements


def test_reentrant_commit_notifications_do_not_duplicate_ledger(controller):
    controller.recordAccepted.connect(lambda _: controller._advance_range_decisions())
    controller.recordsAccepted.connect(lambda _: controller._advance_range_decisions())
    measure(controller)
    controller.resolve_range_decision("keep")
    assert len(controller.session.measurements) == 1
    prepare_candidates(controller, [40, 45])
    controller.accept_candidates([(0, "wool"), (1, "wool")])
    controller.resolve_range_decision("keep")
    controller.resolve_range_decision("keep")
    assert len(controller.session.measurements) == 3


@pytest.mark.parametrize("retry", [True, False])
def test_evidence_admission_failure_keeps_actionable_snapshot(controller, monkeypatch, retry):
    controller.set_evidence_options({"mode": "full_png"})
    measure(controller)
    decision = controller.pending_range_decision
    reserve = controller.evidence.reserve
    monkeypatch.setattr(controller.evidence, "reserve", lambda *a, **k: (_ for _ in ()).throw(MemoryError("budget")))
    with pytest.raises(MemoryError, match="budget"):
        controller.resolve_range_decision("keep")
    assert controller.pending_range_decision is decision
    assert not controller.session.measurements
    assert decision.intent.token.frame.frame_id in controller.budget._frames
    monkeypatch.setattr(controller.evidence, "reserve", reserve)
    controller.resolve_range_decision("keep" if retry else "discard")
    assert len(controller.session.measurements) == int(retry)
    assert not controller.has_range_decisions


def test_automatic_reclassification_notice_is_after_generic_acceptance(controller):
    messages = []
    controller.statusChanged.connect(messages.append)
    controller.recordAccepted.connect(lambda _: controller.statusChanged.emit("generic accepted"))
    measure(controller)
    controller.resolve_range_decision("reclassify", material_id="cashmere", remember=True)
    measure(controller, 45)
    assert messages[-1] == "本条直径 45 μm，已由 绵羊毛 改为 山羊绒"


def test_rule_invalidates_for_operator_without_target_and_clears_without_edit(controller):
    measure(controller)
    controller.resolve_range_decision("reclassify", material_id="cashmere", remember=True)
    original_operator = controller.session.active_operator_id
    materials = controller.session.operators[original_operator].materials["warp"]
    controller.session.add_operator("B")
    controller.session.set_materials([materials[0]])
    present(controller, frame(2))
    measure(controller)
    assert controller.pending_range_decision
    assert not controller.range_rules.list_rules(controller.session.id)
    controller.resolve_range_decision("keep", remember=True)
    before = controller.session.to_dict()
    controller.range_rules.clear(controller.session.id)
    assert controller.session.to_dict() == before
    measure(controller)
    assert controller.pending_range_decision


def test_discard_remains_available_when_source_category_was_removed(controller):
    measure(controller)
    materials = controller.session.operators[controller.session.active_operator_id].materials["warp"]
    controller.session.set_materials([materials[1]])
    assert controller.resolve_range_decision("discard")
    assert not controller.has_range_decisions and not controller.session.measurements


def test_partial_snapshot_admission_releases_leases_and_restores_candidates(controller, monkeypatch):
    from fdm.live_experiment.frames import FrameBudgetExceeded
    prepare_candidates(controller, [40, 45])
    leases = set(controller.budget._leases)
    acquire = controller._acquire_intent_frame
    calls = []
    def fail_second(frame):
        calls.append(frame)
        if len(calls) == 2:
            raise FrameBudgetExceeded("test budget")
        return acquire(frame)
    monkeypatch.setattr(controller, "_acquire_intent_frame", fail_second)
    controller.accept_candidates([(0, "wool"), (1, "wool")])
    assert not controller.has_range_decisions and not controller.session.measurements
    assert set(controller.budget._leases) == leases
    assert all(item["state"] == "available" for item in controller.candidates)
    monkeypatch.setattr(controller, "_acquire_intent_frame", acquire)
    controller.accept_candidates([(0, "wool"), (1, "wool")])
    assert controller.range_decision_count == 2
    controller.resolve_range_decision("discard")
    controller.resolve_range_decision("discard")


def test_changed_then_restored_limits_do_not_revive_old_rule(controller):
    measure(controller)
    controller.resolve_range_decision("keep", remember=True)
    materials = controller.session.operators[controller.session.active_operator_id].materials["warp"]
    controller.session.set_materials([replace(materials[0], upper_limit=50), materials[1]])
    before = controller.session.to_dict()
    assert controller.refresh_range_rules() == 1
    assert controller.session.to_dict() == before
    measure(controller)  # 40 is now normal, so no range decision is created.
    assert not controller.pending_range_decision
    controller.session.set_materials(materials)
    controller.refresh_range_rules()
    measure(controller)
    assert controller.pending_range_decision
    assert not controller.range_rules.list_rules(controller.session.id)


def test_configuration_invalidation_only_removes_related_rule_sources_and_targets(controller):
    from types import SimpleNamespace
    rules = controller.range_rules
    decision = SimpleNamespace(original_material_id="wool", direction="above",
        source_range={"material_id": "wool", "lower_limit_um": 10, "upper_limit_um": 30})
    rules.remember(controller.session.id, decision, "reclassify", target_range={"material_id": "cashmere"})
    independent = SimpleNamespace(original_material_id="other", direction="below",
        source_range={"material_id": "other", "lower_limit_um": 10, "upper_limit_um": None})
    rules.remember(controller.session.id, independent, "keep")
    rules.remember("different trial", decision, "keep")
    materials = controller.session.operators[controller.session.active_operator_id].materials["warp"]
    controller.session.set_materials([materials[0], replace(materials[1], upper_limit=65)])
    assert controller.refresh_range_rules() == 1
    assert [r["original_material_id"] for r in rules.list_rules(controller.session.id)] == ["other"]
    assert len(rules.list_rules("different trial")) == 1
    controller.session.set_materials([replace(materials[0], name="New name", density=1.4), replace(materials[1], upper_limit=65)])
    assert controller.refresh_range_rules() == 0
