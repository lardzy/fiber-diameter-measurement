"""The advanced total goal is shared without editing any experiment payload."""
from copy import deepcopy
from dataclasses import replace

import pytest

from fdm.live_experiment import ExperimentSession
from fdm.live_experiment.models import DiameterRangePolicy
from fdm.live_experiment.reminders import ReminderPolicy, effective_content_total_target
from fdm.live_experiment.trial_defaults import (
    content_total_target_default, new_trial_defaults, save_content_total_target_default,
    save_diameter_range_default, save_reminder_defaults,
)
from fdm.ui.dialogs import SettingsDialog
from fdm.ui.trial_preferences_binding import TrialPreferencesBinding, trial_preferences_binding
from test_trial_preferences_runtime import workspaces


def test_total_goal_has_one_source_and_preserves_other_trial_defaults(tmp_path):
    path = tmp_path / "trial-defaults.json"
    assert content_total_target_default(path=path) == 1500
    policy = ReminderPolicy(content_measured_target=12, content_total_target=50,
                            material_targets={"wool": 17})
    save_reminder_defaults("content", policy, path=path)
    save_diameter_range_default(DiameterRangePolicy(5, 20), path=path)
    save_reminder_defaults("diameter", ReminderPolicy(diameter_measured_target=22), path=path)
    save_content_total_target_default(500, path=path)
    assert content_total_target_default(path=path) == 500
    assert new_trial_defaults("content", path=path)["reminder_policy"] == replace(policy, content_total_target=500)
    diameter = new_trial_defaults("diameter", path=path)
    assert diameter["reminder_policy"].diameter_measured_target == 22
    assert diameter["diameter_range_policy"] == DiameterRangePolicy(5, 20)
    before = path.read_bytes()
    with pytest.raises(ValueError):
        save_content_total_target_default(0, path=path)
    assert path.read_bytes() == before


def test_advanced_goal_apply_updates_all_content_sessions_without_project_edits(workspaces):
    harness, content, diameter = workspaces
    host = harness.window
    other = ExperimentSession(reminder_policy=ReminderPolicy(content_total_target=90))
    host.project.live_experiments.append(other)
    binding = trial_preferences_binding(host)
    before = [(deepcopy(view.to_dict()), deepcopy(view._undo), view.revision)
              for view in (content.session, diameter.session, other)]
    dialog = SettingsDialog(host._app_settings, document=None)
    try:
        binding.prepare_quantity_target_dialog(dialog)
        dialog.select_trial_section("targets")
        assert dialog._trial_settings_widget.tabs.currentIndex() == 1
        assert dialog.trial_total_target() == 1500
        dialog._trial_settings_widget.total_target.setValue(450)
        assert content_total_target_default() == 1500
        binding.commit_quantity_target_dialog(dialog)
        assert not dialog.trial_total_target_changed()
        assert content_total_target_default() == 450
        assert all(effective_content_total_target(item) == 450 for item in (content.session, other))
        assert not hasattr(diameter.session, "_content_total_target")
        assert "450" in content.quantity_targets.progress.text()
        for item, (payload, undo, revision) in zip((content.session, diameter.session, other), before):
            assert item.to_dict() == payload and item._undo == undo and item.revision == revision
        # Restart the binding and open a saved session whose snapshot held a
        # different goal. The shared value affects reminders, never the payload.
        from PySide6.QtWidgets import QMainWindow
        from fdm.models import ProjectState
        reopened_host = QMainWindow()
        reopened_host._app_settings = host._app_settings
        reopened_host.project = ProjectState.empty()
        restarted = TrialPreferencesBinding(reopened_host)
        loaded = ExperimentSession.from_dict(other.to_dict())
        loaded_before = loaded.to_dict()
        restarted.apply_quantity_target(loaded)
        assert effective_content_total_target(loaded) == 450
        assert loaded.to_dict() == loaded_before
        assert new_trial_defaults("content")["reminder_policy"].content_total_target == 450
        reopened_host.close()
        reopened_host.deleteLater()
    finally:
        dialog.deleteLater()


def test_cancel_reset_and_save_failure_keep_quantity_goal_draft(workspaces, monkeypatch):
    harness, content, _ = workspaces
    host = harness.window
    binding = trial_preferences_binding(host)
    dialog = SettingsDialog(host._app_settings, document=None)
    try:
        binding.prepare_quantity_target_dialog(dialog)
        page = dialog._trial_settings_widget
        page.total_target.setValue(321)
        dialog.reject()
        assert content_total_target_default() == 1500
        assert effective_content_total_target(content.session) == 1500
        monkeypatch.setattr("fdm.ui.trial_preferences_binding.save_content_total_target_default",
                            lambda *_: (_ for _ in ()).throw(PermissionError("read only")))
        with pytest.raises(PermissionError):
            binding.commit_quantity_target_dialog(dialog)
        assert page.total_target.value() == 321 and dialog.trial_total_target_changed()
        assert effective_content_total_target(content.session) == 1500
        page.restore_defaults()
        assert dialog.trial_total_target() == 1500
        assert not dialog.trial_total_target_changed()
    finally:
        dialog.deleteLater()


def test_main_window_apply_commits_shared_goal_and_keeps_measurements(workspaces, monkeypatch):
    harness, content, diameter = workspaces
    host = harness.window
    before = [deepcopy(view.session.to_dict()) for view in (content, diameter)]
    monkeypatch.setattr(host, "_apply_screenshot_settings", lambda *a, **k: None)
    dialog = SettingsDialog(host._app_settings, document=None)
    try:
        trial_preferences_binding(host).prepare_quantity_target_dialog(dialog)
        dialog._trial_settings_widget.total_target.setValue(725)
        host._apply_settings_dialog(dialog, close_after=False)
        assert content_total_target_default() == 725
        assert effective_content_total_target(content.session) == 725
        assert [view.session.to_dict() for view in (content, diameter)] == before
        assert not dialog.trial_total_target_changed()
    finally:
        dialog.deleteLater()
