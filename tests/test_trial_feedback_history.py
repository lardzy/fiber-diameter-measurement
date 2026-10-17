"""History and target changes cannot leave obsolete viewport feedback visible."""
from dataclasses import replace
from pathlib import Path
import importlib.util

import pytest
from PySide6.QtWidgets import QPushButton

from fdm.live_experiment.feedback import feedback_for_reason
from fdm.live_experiment.reminders import ReminderPolicy
from fdm.ui import live_experiment_workspace as workspace_module

_spec = importlib.util.spec_from_file_location("feedback_history_workflow",
    Path(__file__).resolve().parents[1] / "scripts/validation/validate_workflow_redesign.py")
workflow = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(workflow)


@pytest.fixture
def page(tmp_path):
    with workflow.WorkflowHarness(tmp_path / "profile") as harness:
        harness.enter()
        page = harness.select(1)
        page._configure(reminder_policy=ReminderPolicy(content_measured_target=2, content_total_target=3))
        yield page


def accept(page, diameter):
    record = page.session.add_measurement(diameter, "wool", field_id=page.controller.field_id)
    page.controller.recordAccepted.emit(record)
    page.controller.changed.emit()
    return record


def history_click(page, text):
    next(button for button in page.findChildren(QPushButton) if button.text() == text).click()
    workflow.settle()


def test_real_undo_redo_buttons_refresh_value_and_do_not_replay_goal(page):
    first, second = accept(page, 19), accept(page, 21)
    assert page.measurement_feedback.feedback.code == "target_reached"
    seen = set(page._reminder_tracker._seen)
    history_click(page, "撤销")
    assert not page.measurement_feedback.isVisible()
    assert page.controller.current_records == [first.id]
    assert page.latest_value.text() == "19.000 μm"
    assert page.measurement_outcome.text() == "最近有效测量"
    assert "1 / 2" in page.quantity_targets.progress.text()
    history_click(page, "撤销")
    assert not page.session.measurements
    assert page.latest_value.text() == "— μm" and not page.measurement_outcome.text()
    assert "暂无有效测量" in page.statusBar().currentMessage()
    history_click(page, "重做")
    history_click(page, "重做")
    assert set(page.session.measurements) == {first.id, second.id}
    assert page.latest_value.text() == "21.000 μm"
    assert not page.measurement_feedback.isVisible()
    assert page._reminder_tracker._seen == seen


def select_current_record(page, record):
    page.workflow_tabs.setCurrentIndex(1)
    page.tabs.setCurrentIndex(0)
    page.current_table.setCurrentIndex(page.current_model.index(page.current_model.ids.index(record.id), 0))


def test_delete_selected_record_removes_old_goal_and_last_value(page):
    first, second = accept(page, 19), accept(page, 21)
    select_current_record(page, second)
    page._delete_record()
    assert not page.measurement_feedback.isVisible()
    assert page.latest_value.text() == "19.000 μm"
    select_current_record(page, first)
    page._delete_record()
    assert not page.controller.current_records
    assert page.latest_value.text() == "— μm" and not page.measurement_outcome.text()


def test_edit_selected_record_refreshes_value_and_invalidates_rejection(page, monkeypatch):
    record = accept(page, 19)
    page.controller.measurementFeedback.emit(feedback_for_reason("no_complete_crossing",
                                                                 **page.controller.feedback_context()))
    select_current_record(page, record)
    monkeypatch.setattr(workspace_module.QInputDialog, "getDouble", lambda *a, **k: (23.5, True))
    monkeypatch.setattr(workspace_module.QInputDialog, "getItem", lambda *a, **k: (a[3][0], True))
    page._edit_record()
    assert page.session.measurements[record.id].diameter_um == 23.5
    assert page.latest_value.text() == "23.500 μm"
    assert page.measurement_outcome.text() == "最近有效测量"
    assert not page.measurement_feedback.isVisible()


@pytest.mark.parametrize("change", ["decrease_count", "increase_count", "target", "rename"])
def test_changed_goal_snapshot_dismisses_card_without_forgetting_delivery(page, change):
    page._count_material("wool", 3)
    assert page.measurement_feedback.feedback.code == "target_reached"
    seen = set(page._reminder_tracker._seen)
    if change == "decrease_count":
        page._count_material("wool", -1)
    elif change == "increase_count":
        page._count_material("wool", 1)
    elif change == "target":
        page._configure(reminder_policy=replace(page.session.reminder_policy, content_total_target=7))
    else:
        page.session.rename_operator("新姓名")
        page._changed()
    assert not page.measurement_feedback.isVisible()
    assert seen <= page._reminder_tracker._seen
    if change == "decrease_count":
        page._count_material("wool", 1)
        assert not page.measurement_feedback.isVisible()


def test_unrelated_state_refresh_does_not_dismiss_current_goal(page):
    page._count_material("wool", 3)
    feedback = page.measurement_feedback.feedback
    page.controller.statusChanged.emit("视场稳定，可以测量")
    page.session.set_weights(70, 30)
    page._changed()
    assert page.measurement_feedback.feedback is feedback
    assert page.measurement_feedback.isVisible()
