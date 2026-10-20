"""Post-selection identifies asynchronous gestures instead of older ledger rows."""
from concurrent.futures import Future
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QColor, QImage
from PySide6.QtTest import QTest, QSignalSpy
from PySide6.QtWidgets import QApplication, QInputDialog

from fdm.live_experiment import ExperimentSession, default_materials, summarize
from fdm.live_experiment.controller import LiveExperimentController
from test_live_experiment_content_workspace import workspace, make_session, ready_result, present, candidates_ready


@pytest.mark.parametrize("input_mode", ["digit", "right_click"])
def test_postselection_input_targets_computing_gesture_not_previous_record(workspace, monkeypatch, input_mode):
    session, controller = workspace.session, workspace.controller
    previous = session.add_measurement(21, "wool", field_id=controller.field_id)
    controller.current_records.append(previous.id)
    workspace._rebuild_records()
    workspace.classification_mode.setCurrentIndex(1)
    future = Future()
    monkeypatch.setattr(controller._manual_pool, "submit", lambda *args: future)
    controller.start((30, 30))
    controller.finish((30, 50), "wool", method="free")
    intent = controller._jobs[0][0]
    if input_mode == "digit":
        workspace.keyboard_mode.setCurrentIndex(0)  # NK selects; default digits count.
        workspace.canvas.setFocus()
        QTest.keyClick(workspace.canvas, Qt.Key.Key_2)
    else:
        QTest.mouseClick(workspace.canvas, Qt.MouseButton.RightButton, pos=QPoint(40, 40))
        menu = workspace._category_menu
        next(action for action in menu.actions() if action.data() == "cashmere").trigger()
        menu.close()
    assert intent.material_id == "cashmere"
    assert session.measurements[previous.id].material_id == "wool"
    future.set_result({"status": "ok", "width_px": 20, "diameter_um": 20,
                       "start": (30, 30), "end": (30, 50)})
    controller.poll()
    latest = session.measurements[controller.current_records[-1]]
    assert latest.material_id == "cashmere"
    assert latest.provenance["measurement_operation_id"] == intent.id
    assert session.measurements[previous.id].material_id == "wool"


@pytest.mark.parametrize("phase", ["motion_gate", "pending", "confirmation"])
def test_postselection_follows_latest_intent_across_publication_states(tmp_path, phase):
    controller = LiveExperimentController(make_session(), evidence_directory=tmp_path)
    try:
        ready_result(controller)
        previous = controller.session.add_measurement(21, "cashmere", field_id=controller.field_id)
        controller.current_records.append(previous.id)
        intent, future = controller._jobs[0]
        if phase != "motion_gate":
            intent.material_id = "cashmere"
            done = Future()
            done.set_result(dict(future.result(), width_px=40, diameter_um=40))
            controller._jobs[0] = (intent, done)
            controller._motion_sequence = 1
            controller.poll()
            assert controller.pending_classification
        present(controller, 10)
        if phase == "confirmation":
            controller.resolve_pending(intent_id=intent.id)
            assert controller._confirmation
        controller.reclassify_latest("wool")
        assert intent.material_id == "wool"
        assert controller.session.measurements[previous.id].material_id == "cashmere"
        controller._motion_sequence = 10
        if phase == "pending":
            controller.resolve_pending(intent_id=intent.id)
        controller.poll()
        accepted = controller.session.measurements[controller.current_records[-1]]
        assert accepted.material_id == "wool" and accepted.id != previous.id
    finally:
        controller.close()


@pytest.mark.parametrize("outcome", ["failed", "timeout", "discarded", "cancelled_draft"])
def test_postselection_after_failed_or_cancelled_latest_gesture_never_falls_back(tmp_path, outcome):
    controller = LiveExperimentController(make_session(), evidence_directory=tmp_path)
    try:
        ready_result(controller)
        previous = controller.session.add_measurement(21, "wool", field_id=controller.field_id)
        controller.current_records.append(previous.id)
        intent, future = controller._jobs[0]
        if outcome == "failed":
            done = Future()
            done.set_result({"status": "rejected", "reason": "no edges"})
            controller._jobs[0] = (intent, done)
            controller._motion_sequence = 1
            controller.poll()
        elif outcome == "timeout":
            controller.poll()
            intent.submission_barrier_at -= 3
            controller.poll()
        elif outcome == "discarded":
            done = Future()
            done.set_result(dict(future.result(), width_px=40, diameter_um=40))
            intent.material_id = "cashmere"
            controller._jobs[0] = (intent, done)
            controller._motion_sequence = 1
            controller.poll()
            controller.resolve_pending(discard=True, intent_id=intent.id)
        else:
            controller.start((35, 35))
            controller.cancel_draft()
        controller.reclassify_latest("cashmere")
        assert controller.session.measurements[previous.id].material_id == "wool"
    finally:
        controller.close()


def test_reclassification_during_confirmation_requires_new_overrange_consent(tmp_path):
    controller = LiveExperimentController(make_session(), evidence_directory=tmp_path)
    try:
        ready_result(controller)
        intent, future = controller._jobs[0]
        done = Future()
        done.set_result(dict(future.result(), width_px=40, diameter_um=40))
        intent.material_id = "cashmere"
        controller._jobs[0] = (intent, done)
        controller._motion_sequence = 1
        controller.poll()
        present(controller, 10)
        controller.resolve_pending(intent_id=intent.id)
        assert controller._confirmation and intent.range_confirmation
        controller.reclassify_latest("cashmere")
        assert controller._confirmation is None
        assert controller.pending_classification[0][0] is intent
        assert intent.range_confirmation is None
        controller._motion_sequence = 10
        controller.poll()
        assert not controller.session.measurements
    finally:
        controller.close()


def test_pending_action_rejects_stale_display_identity(workspace):
    controller = workspace.controller
    # Use actual queued intent identities while replacing only computation.
    controller.start((30, 30))
    controller.finish((30, 70), "cashmere", method="free")
    intent, future = controller._jobs.popleft()
    result = {"status": "ok", "width_px": 40, "diameter_um": 40,
              "start": (30, 30), "end": (30, 70)}
    controller.pending_classification.append((intent, result))
    workspace._refresh_summary()
    assert workspace._pending_display_id == intent.id
    assert "40.0000" in workspace.pending_label.text()
    assert "30" in workspace.pending_label.text() and "未入账" in workspace.pending_label.text()
    assert "测量 #" in workspace.pending_label.text()
    workspace.canvas.update()
    QApplication.processEvents()
    image = workspace.canvas.grab().toImage().convertToFormat(QImage.Format.Format_RGBA8888)
    pixels = np.frombuffer(image.constBits(), np.uint8).reshape(image.height(), image.bytesPerLine())
    rgb = pixels[:, :image.width() * 4].reshape(image.height(), image.width(), 4)[:, :, :3]
    assert np.count_nonzero(np.all(rgb == QColor("#a45300").getRgb()[:3], axis=2)) > 10
    assert not controller.session.measurements  # Painted pending geometry is not a record.
    controller.resolve_pending(discard=True, intent_id=intent.id)
    with pytest.raises(ValueError, match="失效"):
        workspace._resolve_displayed_pending()
    assert not controller.session.measurements


def test_postselection_identity_does_not_cross_operator_or_direction(tmp_path):
    controller = LiveExperimentController(make_session(), evidence_directory=tmp_path)
    try:
        ready_result(controller)
        intent = controller._jobs[0][0]
        controller.session.active_direction = "weft"
        controller.reclassify_latest("cashmere")
        assert intent.material_id == "wool"
        controller.session.active_direction = "warp"
        controller.session.add_operator("乙")
        controller.reclassify_latest("cashmere")
        assert intent.material_id == "wool"
        controller.end_field()
        assert controller._classification_target is None and not controller.current_records
    finally:
        controller.close()


def test_accepted_postselection_warns_and_preserves_record_when_new_category_out_of_range(tmp_path):
    controller = LiveExperimentController(make_session(), evidence_directory=tmp_path)
    try:
        ready_result(controller)
        intent, future = controller._jobs[0]
        done = Future()
        done.set_result(dict(future.result(), width_px=40, diameter_um=40))
        controller._jobs[0] = (intent, done)
        controller._motion_sequence = 1
        controller.poll()
        record = controller.session.measurements[controller.current_records[-1]]
        messages = QSignalSpy(controller.statusChanged)
        assert controller.reclassify_latest("cashmere") is None
        assert controller.session.measurements[record.id].material_id == "wool"
        assert len(controller.session.measurements) == 1
        assert "40" in messages.at(0)[0] and "30" in messages.at(0)[0]
        assert "原记录类别已保留" in messages.at(0)[0]
    finally:
        controller.close()


@pytest.mark.parametrize("newer_manual", [False, True])
def test_candidate_acceptance_ends_postselection_until_next_manual_gesture(tmp_path, newer_manual):
    controller = LiveExperimentController(make_session(), evidence_directory=tmp_path)
    try:
        ready_result(controller)
        controller._motion_sequence = 1
        controller.poll()
        old = controller.session.measurements[controller.current_records[-1]]
        frame = controller.presented.frame
        controller._automatic = SimpleNamespace(cancel=lambda: None, close=lambda: None, poll=lambda: [])
        controller._automatic_context = {"token": controller.presented, "calibration": controller.calibration,
            "algorithm": "traditional", "operator_id": controller.session.active_operator_id,
            "direction": "warp", "lease": controller.budget.acquire(frame), "diagnostics": {}}
        controller.candidates = [{"width_px": 20, "start": [50, 30], "end": [50, 50]}]
        controller._motion_sequence = 1
        controller.accept_candidates([(0, "wool")])
        last = controller.session.measurements[controller.current_records[-1]]
        assert last.id != old.id
        if newer_manual:
            controller.start((60, 30))
            controller.finish((60, 50), "wool", method="free")
            target_id = controller._classification_target["id"]
        changed_id = controller.reclassify_latest("cashmere")
        assert controller.session.measurements[old.id].material_id == "wool"
        if newer_manual:
            assert changed_id is None and controller._classification_target["id"] == target_id
            assert controller._jobs[-1][0].material_id == "cashmere"
            assert controller.session.measurements[last.id].material_id == "wool"
        else:
            assert changed_id is None
            assert controller.session.measurements[last.id].material_id == "wool"
    finally:
        controller.close()


@pytest.mark.parametrize("phase", ["draft", "computing", "pending", "confirmation"])
def test_history_action_cannot_undo_previous_record_while_manual_operation_unfinished(workspace, monkeypatch, phase):
    session, controller = workspace.session, workspace.controller
    record = session.add_measurement(20, "wool", field_id=controller.field_id)
    workspace._rebuild_records()
    future = Future()
    monkeypatch.setattr(controller._manual_pool, "submit", lambda *args: future)
    controller.start((30, 30))
    if phase != "draft":
        controller.finish((30, 70), "cashmere", method="free")
        if phase in {"pending", "confirmation"}:
            intent, _ = controller._jobs.popleft()
            result = {"width_px": 40, "diameter_um": 40, "start": (30, 30), "end": (30, 70)}
            controller.pending_classification.append((intent, result))
            if phase == "confirmation":
                controller.pending_classification.clear()
                controller._confirmation = {"kind": "manual", "intents": [(intent, result)]}
    try:
        with pytest.raises(ValueError, match="未撤销之前记录"):
            workspace._history_action("undo")
        assert record.id in session.measurements
    finally:
        if phase == "confirmation":
            controller._cancel_confirmation()


def test_evidence_writing_and_pure_automatic_analysis_do_not_block_undo(workspace, monkeypatch):
    session, controller = workspace.session, workspace.controller
    record = session.add_measurement(20, "wool", field_id=controller.field_id)
    workspace._rebuild_records()
    with monkeypatch.context() as patch:
        patch.setattr(type(controller.evidence), "pending", property(lambda self: True))
        lease = controller.budget.acquire(controller.presented.frame)
        controller._automatic_context = {"task_id": "uncommitted-analysis", "token": controller.presented,
            "lease": lease, "calibration": controller.calibration,
            "operator_id": session.active_operator_id, "direction": session.active_direction,
            "completed": False}
        try:
            workspace._history_action("undo")
        finally:
            controller._automatic_context = None
            controller.budget.release(lease)
    assert record.id not in session.measurements


def test_operator_rename_keeps_identity_and_allows_three_real_names_with_undo():
    session = ExperimentSession()
    first = session.add_operator("操作者 A")
    session.set_materials(default_materials())
    record = session.add_measurement(20, "wool")
    session.rename_operator("张三")
    assert record.operator_id == first and session.operators[first].name == "张三"
    session.undo()
    assert session.operators[first].name == "操作者 A"
    session.redo()
    second = session.add_operator("李四")
    third = session.add_operator("王五")
    assert len({first, second, third}) == 3
    assert [operator.name for operator in session.operators.values()] == ["张三", "李四", "王五"]
    with pytest.raises(ValueError, match="重复"):
        session.rename_operator("张三", operator_id=second)
    loaded = ExperimentSession.from_dict(session.to_dict())
    assert loaded.operators[first].name == "张三" and loaded.measurements[record.id].operator_id == first
    # Empty newly added people remain explicitly not ready; no reporting rule changes.
    assert summarize(loaded)["fiber"]["status"] == "not_ready"


def test_operator_rename_ui_updates_current_and_history_labels(workspace, monkeypatch):
    session = workspace.session
    original_id = session.active_operator_id
    record = session.add_measurement(20, "wool", field_id=workspace.controller.field_id)
    workspace._rebuild_records()
    history_changed = QSignalSpy(workspace.history_model.dataChanged)
    monkeypatch.setattr(QInputDialog, "getText", lambda *args, **kwargs: ("张三", True))
    workspace._rename_operator()
    assert session.active_operator_id == original_id
    assert len(session.operators) == 1
    assert workspace.operator.currentText() == "张三"
    assert workspace.history_model.data(workspace.history_model.index(0, 1)) == "张三"
    assert history_changed.count()
    assert session.measurements[record.id].operator_id == original_id
