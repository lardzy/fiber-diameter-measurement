"""Exercise the actual main Settings entry, including cleanup signal wiring."""
from types import SimpleNamespace

from PySide6.QtWidgets import QApplication

from test_trial_preferences_runtime import workspaces
from fdm.ui.dialogs import SettingsDialog
from fdm.ui.trial_evidence_lifecycle import trial_evidence_cleanup
from fdm.ui.trial_preferences_binding import trial_preferences_binding


def test_trial_advanced_opens_main_settings_and_cleanup_status(workspaces, monkeypatch, tmp_path):
    harness, content, diameter = workspaces
    host = harness.window
    monkeypatch.setattr(host._screenshot_agent_client, "status",
                        lambda **_: SimpleNamespace(running=False, error="", result={}))
    from fdm.ui.screenshot_hotkey_status import ScreenshotHotkeyStatusController
    monkeypatch.setattr(ScreenshotHotkeyStatusController, "for_client",
                        lambda *a: SimpleNamespace(close=lambda: None))
    cleanup = trial_evidence_cleanup(host)
    binding = trial_preferences_binding(host)
    observed = []

    def enter(dialog):
        assert dialog._settings_pages.currentWidget().property("settingsPageKey") == "trial"
        dialog.set_trial_cleanup_status("清理测试", True)
        cleanup.statusChanged.emit("已清理 2 张临时图片", False)
        page = dialog._trial_settings_widget
        page.controls["font_size"].setValue(23)
        reference = {"path": str(tmp_path / "late-reference.npz"), "sha256": "a" * 64, "context": {}}
        binding.update(optical_analysis={**binding.preferences.optical_analysis,
                                        "flat_reference": reference})
        draft = dialog.app_settings().trial_preferences
        assert draft.font_size == 23
        assert draft.optical_analysis["flat_reference"] == reference
        observed.append(True)
        dialog.reject()
        return dialog.DialogCode.Rejected

    monkeypatch.setattr(SettingsDialog, "exec", enter)
    content.advanced_toggle.trigger()
    assert observed == [True]
    # Closed Settings dialogs must not remain signal recipients.
    cleanup.statusChanged.emit("后续后台清理完成", False)
    binding.update(font_size=17)
    QApplication.processEvents()
    assert not harness.slot_errors
