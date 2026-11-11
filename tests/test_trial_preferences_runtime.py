"""Apply/shortcut preferences exercise real workspaces and retained task policies."""
from concurrent.futures import Future
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import json
import sys
import threading
import time

import pytest
from PySide6.QtWidgets import QApplication

from fdm.live_experiment import default_materials
from fdm.live_experiment.frames import FrameHandle, PresentedFrameToken
from fdm.settings import AppSettingsIO
from fdm.ui.trial_preferences_binding import trial_preferences_binding
from test_live_experiment_content_workspace import workspace
from test_candidate_controller import controller, batch, frame, present
from test_trial_assistance_ui import candidates

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/validation"))
from validate_workflow_redesign import WorkflowHarness, settle


@pytest.fixture
def workspaces(tmp_path):
    with WorkflowHarness(tmp_path / "profile", size=(1366, 768)) as harness:
        harness.enter()
        content = harness.select(1)
        for direction in ("warp", "weft"):
            content.session.set_materials(default_materials(), direction=direction)
        content._refresh_materials()
        content._changed()
        try:
            diameter = harness.select(2)
        except KeyError:
            pytest.fail("\n".join(harness.slot_errors) or repr(harness.modal_messages))
        for view in (content, diameter):
            view.controller._timer.stop()
            view._refresh_timer.stop()
        yield harness, content, diameter
        assert not harness.slot_errors, "\n".join(harness.slot_errors)


def test_main_settings_activation_updates_hidden_and_visible_without_trial_edits(workspaces):
    harness, content, diameter = workspaces
    host = harness.window
    assert not content.isVisible() and diameter.isVisible()
    before = [(deepcopy(view.session.to_dict()), deepcopy(view.session._undo)) for view in (content, diameter)]
    preferences = replace(host._app_settings.trial_preferences, font_size=19,
        label_background=True, color_mode="custom", font_color="#12AB34", show_stability_status=True,
        evidence_enabled=True, evidence_format="full_png", gesture="drag", input_mode="manual")
    host._activate_app_settings(replace(host._app_settings, trial_preferences=preferences))
    for view, (payload, undo) in zip((content, diameter), before):
        assert view._applied_trial_preferences.font_size == 19
        assert view._applied_trial_preferences.font_color == "#12AB34"
        assert view.input_mode.currentData() == "manual" and view.gesture.currentIndex() == 1
        assert view.controller.effective_evidence_options["mode"] == "full_png"
        assert view.session.to_dict() == payload and view.session._undo == undo
    assert not trial_preferences_binding(host)._timer.isActive()


def test_shortcuts_share_preferences_and_restart_does_not_read_trial_image_flags(workspaces):
    harness, content, diameter = workspaces
    host = harness.window
    binding = trial_preferences_binding(host)
    assert not binding.preferences.evidence_enabled
    assert all(view.controller.effective_evidence_options["mode"] == "disabled" for view in (content, diameter))
    before = [deepcopy(view.session.to_dict()) for view in (content, diameter)]
    diameter.evidence_mode_actions["crop_jpeg"].trigger()
    diameter.manual_gesture_actions["drag"].trigger()
    diameter.candidate_opacity.setValue(43)
    assert binding.preferences.evidence_enabled and binding.preferences.gesture == "drag"
    assert binding.preferences.opacity == 43
    assert content.gesture.currentIndex() == 1 and content.candidate_opacity.value() == 43
    assert content.controller.effective_evidence_options["mode"] == "crop_jpeg"
    binding.flush()
    restored = AppSettingsIO.load(harness.profile / "settings.json")
    assert restored.trial_preferences == binding.preferences
    assert [view.session.to_dict() for view in (content, diameter)] == before
    diameter.evidence_mode_actions["disabled"].trigger()
    binding.flush()
    assert not AppSettingsIO.load(harness.profile / "settings.json").trial_preferences.evidence_enabled
    assert binding.preferences.evidence_format == "crop_jpeg"


def test_first_upgrade_navigation_preserves_legacy_preferences_without_settings_edit(tmp_path):
    from PySide6.QtGui import QAction
    from PySide6.QtWidgets import QMainWindow
    from fdm.models import ProjectState
    from fdm.settings import AppSettings, settings_file_path
    from fdm.ui.live_experiment_workspace import open_live_experiment
    path = settings_file_path()
    initial = AppSettings().to_dict()
    initial.pop("trial_preferences")
    path.write_text(json.dumps(initial), encoding="utf-8")
    old = {"version": 4, "algorithm": "multiscale", "manual_algorithm": "free",
        "input_mode": "manual", "automatic": False, "opacity": 37}
    legacy_path = path.parent / "trial-workbench.json"
    legacy_path.write_text(json.dumps({"assistance": old}), encoding="utf-8")
    class Host(QMainWindow):
        def current_document(self):
            return None
    host = Host()
    host.project = ProjectState.empty()
    host._images = {}
    host._app_settings = AppSettingsIO.load()
    host.save_project_action = QAction(host)
    view = None
    try:
        view = open_live_experiment(host)
        assert view.algorithm.currentData() == "multiscale"
        assert view.measure_mode.currentData() == "free"
        assert not view._trial_preferences._timer.isActive()
        view.hide()
        view._persist_view_preferences()
        view.close()
        assert "trial_preferences" not in json.loads(path.read_text())
        assert json.loads(legacy_path.read_text())["assistance"] == old
        reopened = AppSettingsIO.load()
        assert reopened.trial_preferences.algorithm == "multiscale"
        assert reopened.trial_preferences.manual_algorithm == "free"
        assert reopened.trial_preferences.input_mode == "manual"
        assert reopened.trial_preferences.opacity == 37
        assert not reopened.trial_preferences.evidence_enabled
    finally:
        if view is not None:
            view.close()
        host.close()


def test_routine_status_is_opt_in_but_operation_and_source_errors_remain(workspace):
    view = workspace
    view.controller._timer.stop()
    view._refresh_timer.stop()
    binding = trial_preferences_binding(view.host)
    view.statusBar().clearMessage()
    view.controller.state = "ready"
    view.controller.routineStatusChanged.emit("视场稳定，可以测量")
    view._refresh_summary()
    assert view.statusBar().currentMessage() == ""
    assert "就绪" not in view.live_state_label.text()
    assert "μm/px" in view.live_state_label.text()
    view.controller.statusChanged.emit("请等待视场稳定，本次未入账。")
    assert "本次未入账" in view.statusBar().currentMessage()
    binding.update(show_stability_status=True)
    # Opting in does not hide the newer rejected-measurement explanation.
    assert "本次未入账" in view.statusBar().currentMessage()
    view.controller.routineStatusChanged.emit("视场稳定，可以测量")
    assert view.statusBar().currentMessage() == "视场稳定，可以测量"
    assert "视场就绪" in view.live_state_label.text()
    binding.update(show_stability_status=False)
    assert view.statusBar().currentMessage() == ""
    view.controller.state = "unavailable"
    view.controller.statusChanged.emit("设备连接中断")
    view._refresh_summary()
    assert view.statusBar().currentMessage() == "设备连接中断"
    assert "来源不可用" in view.live_state_label.text()


def test_reenabling_stability_restores_ready_hint_without_another_frame(workspace):
    view = workspace
    view.controller._timer.stop()
    view._refresh_timer.stop()
    binding = trial_preferences_binding(view.host)
    view.statusBar().clearMessage()
    view.controller.routineStatusChanged.emit("视场稳定，可以测量")
    assert view.statusBar().currentMessage() == ""
    binding.update(show_stability_status=True)
    assert view.statusBar().currentMessage() == "视场稳定，可以测量"
    assert "视场就绪" in view.live_state_label.text()
    binding.update(show_stability_status=False)
    assert view.statusBar().currentMessage() == ""
    binding.update(show_stability_status=True)
    assert view.statusBar().currentMessage() == "视场稳定，可以测量"


@pytest.mark.parametrize("changed", ["unavailable", "stabilizing", "field", "source"])
def test_stability_enable_never_replays_ready_from_stale_source_or_field(workspace, changed):
    view = workspace
    controller = view.controller
    controller._timer.stop()
    view._refresh_timer.stop()
    binding = trial_preferences_binding(view.host)
    view.statusBar().clearMessage()
    controller.routineStatusChanged.emit("视场稳定，可以测量")
    if changed in {"unavailable", "stabilizing"}:
        controller.state = changed
    elif changed == "field":
        controller.field_epoch += 1
    else:
        controller.latest_frame = FrameHandle.snapshot(controller.latest_frame.pixels,
                                                       source_id="new-source")
    binding.update(show_stability_status=True)
    assert view.statusBar().currentMessage() == ""


@pytest.mark.parametrize("initial_enabled", [False, True])
@pytest.mark.parametrize("change_at", ["first_point", "submitted"])
def test_evidence_choice_is_captured_before_async_work(workspace, monkeypatch, initial_enabled, change_at):
    view, binding = workspace, trial_preferences_binding(workspace.host)
    controller = view.controller
    controller._timer.stop()
    binding.update(evidence_enabled=initial_enabled, evidence_format="crop_jpeg")
    blocker, entered = threading.Event(), threading.Event()
    compute = controller._compute
    def delayed(intent):
        entered.set()
        assert blocker.wait(3), "test did not release measurement"
        return compute(intent)
    monkeypatch.setattr(controller, "_compute", delayed)
    try:
        controller.start((30, 30))
        if change_at == "first_point":
            binding.update(evidence_enabled=not initial_enabled)
            assert controller.draft is not None
        controller.finish((30, 50), "wool", method="free")
        assert entered.wait(1)
        intent = controller._jobs[0][0]
        if change_at == "submitted":
            binding.update(evidence_enabled=not initial_enabled)
        assert not intent.cancel.is_set()
        assert intent.evidence_options["mode"] == ("crop_jpeg" if initial_enabled else "disabled")
        blocker.set()
        deadline = time.monotonic() + 3
        while (not view.session.measurements or controller.evidence.pending) and time.monotonic() < deadline:
            controller.poll()
            QApplication.processEvents()
            time.sleep(.002)
        assert len(view.session.measurements) == 1
        record = next(iter(view.session.measurements.values()))
        assert record.diameter_um == 20
        assert record.evidence["state"] == ("saved" if initial_enabled else "disabled")
        if initial_enabled:
            assert Path(record.evidence["path"]).is_file()
    finally:
        blocker.set()


def test_display_change_keeps_draft_but_gesture_change_cancels_unfinished_point(workspace):
    view = workspace
    binding = trial_preferences_binding(view.host)
    view.controller._timer.stop()
    before = deepcopy(view.session.to_dict())
    view.controller.start((30, 30))
    token = view.controller.draft
    binding.update(font_size=22, label_background=True, show_stability_status=True)
    assert view.controller.draft is token
    binding.update(gesture="drag")
    assert view.controller.draft is None
    assert view.session.to_dict() == before


def test_algorithm_change_clears_unsubmitted_preview_without_touching_completed_record(workspace):
    view = workspace
    view.controller._timer.stop()
    candidates(view)
    view.controller.submit_candidate("ui-batch:0", "wool")
    assert len(view.session.measurements) == 1
    before = deepcopy(view.session.to_dict())
    binding = trial_preferences_binding(view.host)
    # Unrelated interaction changes must not compare mismatched optical
    # algorithm IDs and accidentally discard the remaining candidate preview.
    binding.update(input_mode="manual")
    assert len(view.controller.candidates) == 3
    binding.update(algorithm="multiscale")
    assert view.controller._automatic_context is None and not view.controller.candidates
    assert view.session.to_dict() == before


def test_algorithm_change_preserves_pending_confirmation_and_hides_unsubmitted(controller):
    ids = batch(controller)
    context = controller._automatic_context
    controller._motion_future = (controller.field_epoch, Future())
    present(controller, frame(2, live=True))
    controller.state = "ready"
    controller._motion_sequence = 0
    controller.submit_candidate(ids[0], "wool")
    assert controller.candidate_pending_count == 1
    controller.invalidate_analysis_preferences()
    assert controller._automatic_context is context
    assert controller.candidates[0]["state"] == "confirming"
    assert all(item["state"] == "stale" for item in controller.candidates[1:])
    with pytest.raises(ValueError):
        controller.submit_candidate(ids[1], "wool")
    controller._motion_sequence = 2
    controller._poll_confirmation()
    assert len(controller.session.measurements) == 1
    controller._schedule_automatic()
    assert controller._automatic_context is None and not controller.candidates
    assert len(controller.session.measurements) == 1


def test_actual_settings_binding_keeps_submitted_candidate_policy_until_confirmation(workspace):
    view, controller = workspace, workspace.controller
    controller._timer.stop()
    view._refresh_timer.stop()
    ids = candidates(view)
    context = controller._automatic_context
    old = controller.presented
    incoming = FrameHandle.snapshot(old.frame.pixels, source_id=old.frame.source_id,
        generation=old.frame.generation, sequence=old.frame.sequence + 1,
        live=True, metadata=old.frame.metadata)
    controller._motion_future = (controller.field_epoch, Future())
    controller.receive_frame(incoming)
    controller.mark_presented(PresentedFrameToken(incoming, controller.field_epoch,
        old.display_revision, old.scale, old.offset))
    controller.state = "ready"
    controller._motion_sequence = 0
    controller.submit_candidate(ids[0], "wool")
    assert controller.candidate_pending_count == 1
    binding = trial_preferences_binding(view.host)
    binding.update(algorithm="multiscale", evidence_enabled=True)
    assert controller._automatic_context is context
    assert controller.candidates[0]["state"] == "confirming"
    assert all(item["state"] == "stale" for item in controller.candidates[1:])
    controller._motion_sequence = incoming.sequence
    controller._poll_confirmation()
    assert len(view.session.measurements) == 1
    record = next(iter(view.session.measurements.values()))
    assert record.measurement_method == "automatic-traditional"
    assert record.evidence["state"] == "disabled"
    controller._schedule_automatic()
    assert not controller.candidates and controller._automatic_context is None
