"""The live diagnostics entry publishes captured references to shared settings."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from fdm.live_experiment.frames import FrameHandle
from fdm.settings import AppSettingsIO
from test_live_experiment_content_workspace import workspace


def open_diagnostics(view):
    action = next(action for action in view.analyze_button.menu().actions()
                  if action.text() == "光照检查与诊断…")
    action.trigger()
    dialog = view.optical_settings_dialog
    dialog._timer.stop()
    # The reference tab has no recurring light-check job, so the test observes
    # completion of exactly the explicitly requested reference operation.
    dialog.tabs.setCurrentIndex(0)
    assert dialog._diagnostics_only and dialog.isVisible()
    assert dialog.save_button.isHidden() and dialog.refresh_button.isHidden()
    return dialog


def collect_reference(dialog, monkeypatch, *, brightness=210):
    def frame(sequence):
        return FrameHandle.snapshot(np.full((120, 160, 3), brightness, np.uint8),
                                    source_id="camera", sequence=sequence, live=True)

    current = [frame(0)]
    monkeypatch.setattr(dialog, "_raw_frame", lambda: current[0])
    dialog.capture_button.click()
    dialog._collect_frame()
    dialog._collect_frame()
    assert len(dialog._capture_frames) == 1  # Repeated frames are not five samples.
    for sequence in range(1, 5):
        current[0] = frame(sequence)
        dialog._collect_frame()
    assert dialog._job is not None and dialog._job[0] == "capture"
    dialog._job[1].result(timeout=3)


def test_diagnostics_reference_is_published_saved_and_retained_without_enabling(workspace, monkeypatch):
    dialog = open_diagnostics(workspace)
    before = deepcopy(workspace.session.to_dict())
    used = workspace.controller.budget.used_bytes
    analyses = []
    monkeypatch.setattr(workspace, "_analyze_candidates", lambda: analyses.append(True))
    assert not workspace._trial_preferences.preferences.optical_analysis["flat_enabled"]

    collect_reference(dialog, monkeypatch)
    dialog._tick()
    preferences = workspace._trial_preferences.preferences
    reference = preferences.optical_analysis["flat_reference"]
    assert reference is not None and Path(reference["path"]).is_file()
    assert workspace.controller.optical_analysis_config["flat_reference"] == reference
    assert not preferences.optical_analysis["flat_enabled"]
    assert not dialog.flat_enabled.isChecked()
    assert "保存设置" not in dialog.status.text()
    assert "设置 → 采集与试验 → 全图分析" in dialog.status.text()
    assert workspace.session.to_dict() == before and analyses == []
    assert workspace.controller.budget.used_bytes == used

    workspace._trial_preferences.flush()
    assert AppSettingsIO.load().trial_preferences == preferences
    dialog.close()
    reopened = open_diagnostics(workspace)
    assert reopened._config["flat_reference"] == reference
    reopened.reference_preview_button.click()
    assert reopened._job is not None and reopened._job[0] == "preview"
    reopened._job[1].result(timeout=3)
    reopened._tick()
    assert reopened._preview_image is not None
    assert workspace.controller.budget.used_bytes == used


@pytest.mark.parametrize("latest_enabled", [False, True])
def test_capture_merges_only_reference_into_latest_global_choices(workspace, monkeypatch, latest_enabled):
    binding = workspace._trial_preferences
    binding.update(optical_analysis={**binding.preferences.optical_analysis,
                                    "flat_enabled": not latest_enabled})
    dialog = open_diagnostics(workspace)
    collect_reference(dialog, monkeypatch)
    # A main-settings Apply can complete while the background capture is in
    # flight. Its current choices must survive the delayed capture result.
    binding.update(optical_analysis={**binding.preferences.optical_analysis,
                                    "flat_enabled": latest_enabled,
                                    "imaging_mode": "double_edge"}, font_size=19)
    expected = deepcopy(binding.preferences.optical_analysis)
    before = deepcopy(workspace.session.to_dict())
    assert dialog._config["imaging_mode"] == "double_edge"
    assert dialog.flat_enabled.isChecked() is latest_enabled
    dialog._tick()

    actual = binding.preferences.optical_analysis
    assert actual["flat_reference"] is not None
    assert {key: value for key, value in actual.items() if key != "flat_reference"} == {
        key: value for key, value in expected.items() if key != "flat_reference"}
    assert actual["flat_enabled"] is latest_enabled
    assert workspace.controller.optical_analysis_config == actual
    assert binding.preferences.font_size == 19
    assert workspace.session.to_dict() == before


def test_invalid_blank_capture_does_not_publish_or_enable(workspace, monkeypatch):
    dialog = open_diagnostics(workspace)
    before = workspace._trial_preferences.preferences
    used = workspace.controller.budget.used_bytes
    collect_reference(dialog, monkeypatch, brightness=0)
    dialog._tick()
    assert workspace._trial_preferences.preferences == before
    assert workspace.controller.optical_analysis_config["flat_reference"] is None
    assert not workspace.controller.optical_analysis_config["flat_enabled"]
    assert "零亮度" in dialog.status.text()
    assert workspace.controller.budget.used_bytes == used
