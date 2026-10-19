"""Real navigation must survive a cancelled project transition or failed save."""
from threading import Event
import time
from copy import deepcopy

import pytest
from PySide6.QtWidgets import QMessageBox

from test_acquisition_workflow_redesign import workflow


@pytest.fixture
def harness(tmp_path):
    with workflow.WorkflowHarness(tmp_path / "profile") as instance:
        instance.enter()
        page = instance.select(1)
        from fdm.live_experiment import default_materials
        page.session.set_materials(default_materials())
        page._refresh_materials()
        page.sample.setText("取消后继续测量")
        page.commit_pending_edits()
        page.ruler_choice.setCurrentIndex(1)
        page.ruler_choice.activated.emit(1)
        controller = page.controller
        record = page.session.add_measurement(20, "wool", field_id=controller.field_id,
            endpoints=((30, 30), (30, 70)))
        page.session.update_field(controller.field_id, first_frame_id=controller.latest_frame.frame_id)
        controller.current_records.append(record.id)
        page._record_accepted(record)
        page._changed()
        instance.accepted_field = controller.field_id
        instance.accepted_record = record.id
        instance.accepted_fields = deepcopy(page.session.fields)
        page.workflow_tabs.setCurrentIndex(1)
        yield instance


def choose_dialog_button(monkeypatch, label):
    def execute(dialog):
        # macOS message boxes intentionally expose an empty windowTitle.
        next(button for button in dialog.buttons() if button.text() == label).click()
        return 0
    monkeypatch.setattr(QMessageBox, "exec", execute)


def assert_retained(harness, page):
    workflow.settle()
    harness.backend.tick()
    assert harness.window.isVisible()
    assert harness.hub.navigation.currentIndex() == 1
    assert harness.hub.pages.currentWidget() is page
    assert harness.window._live_experiment_workspaces["content"] is page
    assert not page._closed
    assert page.workflow_tabs.currentIndex() == 1
    assert page.sample.text() == "取消后继续测量"
    assert page.canvas.frame is not None
    assert page.source is not None
    assert page.controller.field_id == harness.accepted_field
    assert page.controller.current_records == [harness.accepted_record]
    assert page.current_model.ids == [harness.accepted_record]
    assert page.session.fields == harness.accepted_fields
    assert harness.backend.starts == 1
    assert not harness.slot_errors


def wait_until(condition, timeout=6):
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "Qt condition timed out"
        workflow.settle()
        time.sleep(.003)
    workflow.settle()


def test_cancel_main_close_preserves_trial_then_discard_disposes(harness, monkeypatch):
    page = harness.hub.pages.currentWidget()
    choose_dialog_button(monkeypatch, "取消")
    assert not harness.window.close()
    assert_retained(harness, page)
    choose_dialog_button(monkeypatch, "放弃")
    assert harness.window.close()
    assert not harness.window.isVisible()
    assert not harness.window._live_experiment_workspaces
    assert not harness.window._capture_manager.is_preview_active()


def test_cancel_project_open_preserves_trial_then_retry_loads(harness, monkeypatch, tmp_path):
    from fdm.live_experiment import ExperimentSession
    from fdm.models import ProjectState
    from fdm.project_io import ProjectIO
    next_project = ProjectState.empty()
    next_project.live_experiments.append(ExperimentSession(sample_name="另一个项目"))
    path = tmp_path / "next.fdmproj"
    ProjectIO.save(next_project, path)
    page, project = harness.hub.pages.currentWidget(), harness.window.project
    choose_dialog_button(monkeypatch, "取消")
    result = harness.window._load_project_from_path(path)
    assert result.cancelled and not result.success
    assert harness.window.project is project
    assert_retained(harness, page)
    choose_dialog_button(monkeypatch, "放弃")
    assert harness.window._load_project_from_path(path).success
    assert harness.window.project is not project
    assert not harness.window._live_experiment_workspaces
    assert harness.hub.navigation.currentIndex() == 0


@pytest.mark.parametrize("failure", [False, True])
def test_save_before_close_retains_page_until_success_or_failure(harness, monkeypatch, tmp_path, failure):
    from fdm.project_io import ProjectIO
    from fdm.ui import project_save_coordinator as saving
    page = harness.hub.pages.currentWidget()
    path = tmp_path / "saved.fdmproj"
    monkeypatch.setattr(harness.window, "_select_project_save_path", lambda _: str(path))
    choose_dialog_button(monkeypatch, "保存")
    entered, release = Event(), Event()
    original = saving.write_project_save

    def delayed(request):
        entered.set()
        assert release.wait(5), "test must release save writer"
        if failure:
            raise OSError("validation disk full")
        return original(request)

    monkeypatch.setattr(saving, "write_project_save", delayed)
    try:
        assert not harness.window.close()
        wait_until(entered.is_set)
        assert harness.window.project_save_coordinator.busy
        assert_retained(harness, page)
    finally:
        release.set()
        wait_until(lambda: not harness.window.project_save_coordinator.busy)
    if failure:
        assert harness.window.project_save_coordinator.status.phase == "failed"
        assert_retained(harness, page)
        assert harness.window._project_dirty()
        assert not path.exists()
    else:
        wait_until(lambda: not harness.window.isVisible())
        assert not harness.window._live_experiment_workspaces
        assert not harness.window._capture_manager.is_preview_active()
        restored = ProjectIO.load(path).live_experiments[0]
        assert restored.sample_name == "取消后继续测量"
        assert restored.fields == page.session.fields == harness.accepted_fields
    assert not harness.slot_errors


def test_accepted_close_cancels_restart_already_queued_by_async_stop(tmp_path):
    backend = workflow.FakeCaptureBackend(delayed_stop=True)
    with workflow.WorkflowHarness(tmp_path / "async-profile", backend=backend) as h:
        h.enter()
        h.select(1)
        h.leave()
        h.enter(frame=False)
        # Complete the old stop, but deliberately leave the queued 0-ms
        # restart pending until the close event has been accepted.
        callback, backend.pending_stop = backend.pending_stop, None
        callback()
        assert h.hub._restart_pending
        assert h.window.close()
        workflow.settle()
        assert not h.window.isVisible()
        assert not h.window._capture_manager.is_preview_active()
        assert backend.starts == 1
        assert not h.hub._entered and not h.hub._restart_pending
        assert not h.slot_errors


def test_background_save_rechecks_focused_edits_before_resuming_close(harness, monkeypatch, tmp_path):
    from fdm.project_io import ProjectIO
    from fdm.ui import project_save_coordinator as saving
    page = harness.hub.pages.currentWidget()
    path = tmp_path / "saved-before-edit.fdmproj"
    monkeypatch.setattr(harness.window, "_select_project_save_path", lambda _: str(path))
    entered, release = Event(), Event()
    original = saving.write_project_save

    def delayed(request):
        entered.set()
        assert release.wait(5), "test must release save writer"
        return original(request)

    monkeypatch.setattr(saving, "write_project_save", delayed)
    choose_dialog_button(monkeypatch, "保存")
    try:
        assert not harness.window.close()
        wait_until(entered.is_set)
        # Leave the editor focused/uncommitted: the resumed close must flush
        # it and prompt again instead of trusting the older saved snapshot.
        page.sample_number.setText("save-in-progress-edit")
        choose_dialog_button(monkeypatch, "取消")
    finally:
        release.set()
        wait_until(lambda: not harness.window.project_save_coordinator.busy)
    workflow.settle()
    assert_retained(harness, page)
    assert page.session.sample_number == "save-in-progress-edit"
    assert harness.window._project_dirty()
    assert ProjectIO.load(path).live_experiments[0].sample_number == ""
