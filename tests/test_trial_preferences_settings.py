"""Global trial preference persistence and draft-only main-settings editing."""
from dataclasses import replace
import json
from unittest.mock import patch

import pytest

from fdm.settings import AppSettings, AppSettingsIO
from fdm.trial_preferences import TrialPreferences


def test_trial_defaults_are_display_only_and_image_storage_is_opt_in():
    preferences = AppSettings().trial_preferences.normalized_copy()
    assert preferences.content_label == "material"
    assert preferences.diameter_label == "value"
    assert preferences.font_size == 12 and preferences.font_family == ""
    assert not preferences.label_background and not preferences.show_stability_status
    assert not preferences.evidence_enabled
    assert preferences.evidence_options == {"version": 1, "mode": "disabled"}
    assert replace(preferences, evidence_enabled=True).evidence_options["mode"] == "crop_jpeg"
    assert AppSettings().trial_preferences.optical_analysis["algorithm"] == "optical_adaptive"
    assert replace(preferences, algorithm="multiscale").normalized_copy().optical_analysis == preferences.optical_analysis


@pytest.mark.parametrize("value", [None, [], "invalid", 7])
def test_missing_or_invalid_preferences_use_independent_defaults(value):
    restored = AppSettings.from_dict({"trial_preferences": value, "measurement_label_font_size": 40})
    assert restored.trial_preferences == TrialPreferences().normalized_copy()
    assert restored.length_measurement_label_style.font_size == 40


def test_invalid_fields_are_normalized_without_losing_valid_fields():
    preferences = TrialPreferences.from_dict({
        "font_size": float("nan"), "font_color": "not-a-color", "font_family": 123,
        "automatic": "false", "evidence_enabled": "true", "evidence_format": "bad",
        "opacity": 999, "retention_seconds": -5, "gesture": "drag",
        "input_mode": "manual", "font_bold": True, "algorithm": "optical_adaptive",
        "optical_analysis": {"imaging_mode": "bad"}, "diameter_label": "material",
    })
    assert preferences.font_size == 12 and preferences.font_family == ""
    assert preferences.font_color == "#FFFFFF" and preferences.opacity == 100
    assert preferences.retention_seconds == 0 and preferences.font_bold
    assert not preferences.automatic and not preferences.evidence_enabled
    assert preferences.evidence_format == "crop_jpeg" and preferences.diameter_label == "value"
    assert preferences.gesture == "drag" and preferences.input_mode == "manual"
    assert preferences.optical_analysis["imaging_mode"] == "mixed"
    assert preferences.optical_analysis["algorithm"] == "optical_adaptive"
    json.dumps(preferences.to_dict(), allow_nan=False)


def test_shared_preferences_roundtrip_import_and_copy(tmp_path):
    original = TrialPreferences(evidence_enabled=True, evidence_format="full_png", gesture="drag",
        show_stability_status=True, content_label="number_material", font_size=24,
        color_mode="custom", font_color="#123abc", font_bold=True,
        optical_analysis={"imaging_mode": "double_edge", "flat_enabled": False})
    settings = AppSettings(trial_preferences=original)
    path, imported_path = tmp_path / "settings.json", tmp_path / "import.json"
    AppSettingsIO.save(settings, path)
    restored = AppSettingsIO.load(path)
    assert restored.trial_preferences == original.normalized_copy()
    imported, _ = AppSettingsIO.replace_with_file(path, imported_path)
    assert imported.trial_preferences == original.normalized_copy()
    copy = settings.normalized_copy()
    copy.trial_preferences.optical_analysis["imaging_mode"] = "solid"
    assert original.optical_analysis["imaging_mode"] == "double_edge"


def test_legacy_shared_preferences_migrate_once_without_enabling_evidence(tmp_path):
    legacy = tmp_path / "trial-workbench.json"
    before = {"content": {"splitter": [100, 300]}, "assistance": {
        "version": 4, "input_mode": "candidate", "manual_algorithm": "free",
        "algorithm": "multiscale", "automatic": True, "opacity": 77,
        "retention_seconds": 2.5, "evidence_enabled": True, "gesture": "drag"}}
    legacy.write_text(json.dumps(before), encoding="utf-8")
    path = tmp_path / "settings.json"
    preferences = AppSettingsIO.load(path).trial_preferences
    assert preferences.input_mode == "candidate" and preferences.manual_algorithm == "free"
    assert preferences.algorithm == "multiscale" and preferences.automatic
    assert preferences.opacity == 77 and preferences.retention_seconds == 2.5
    assert not preferences.evidence_enabled and preferences.gesture == "click"
    assert not path.exists() and json.loads(legacy.read_text()) == before
    AppSettingsIO.save(AppSettings(trial_preferences=replace(preferences, opacity=45)), path)
    assert AppSettingsIO.load(path).trial_preferences.opacity == 45
    path.write_text('{"trial_preferences":null}', encoding="utf-8")
    assert AppSettingsIO.load(path).trial_preferences.input_mode == "mixed"


def test_main_page_draft_cancel_restore_and_real_preview(desktop_application):
    from fdm.ui.dialogs import SettingsDialog
    from fdm.ui.trial_label_style import draw_trial_label
    source = AppSettings(trial_preferences=TrialPreferences(evidence_enabled=True, font_size=21),
                         professional_measurement_label_decimals=7)
    dialog = SettingsDialog(source, document=None)
    try:
        dialog.select_trial_section("display")
        page = dialog._trial_settings_widget
        assert page.font_combo.currentText() == ""
        assert page.font_combo.lineEdit().placeholderText() == "跟随应用字体"
        assert dialog._settings_pages.currentWidget() is page
        assert dialog._settings_page_title.text() == "采集与试验"
        page.controls["font_size"].setValue(18)
        page.controls["label_background"].setChecked(True)
        page.controls["content_label"].setCurrentIndex(page.controls["content_label"].findData("number_material"))
        page.preview.resize(380, 170)
        with patch("fdm.ui.trial_label_style.draw_trial_label", wraps=draw_trial_label) as draw:
            assert not page.preview.grab().isNull()
        assert draw.call_count == 2
        assert draw.call_args_list[0].args[2] == "#1 · 绵羊毛"
        assert draw.call_args_list[1].args[2] == "18.25 μm"
        changed = dialog.app_settings().trial_preferences
        assert changed.font_size == 18 and changed.label_background
        assert source.trial_preferences.font_size == 21 and not source.trial_preferences.label_background
        dialog._restore_page_defaults_button.click()
        restored = dialog.app_settings()
        assert restored.trial_preferences == TrialPreferences().normalized_copy()
        assert restored.professional_measurement_label_decimals == 7
        dialog.reject()
        assert source.trial_preferences.evidence_enabled
    finally:
        dialog.close()


def test_navigation_cleanup_action_and_optical_configuration_without_camera(desktop_application):
    from fdm.ui.dialogs import SettingsDialog
    dialog = SettingsDialog(AppSettings(), document=None)
    calls = []
    dialog.trialEvidenceCleanupRequested.connect(lambda: calls.append("clean"))
    try:
        dialog._settings_search.setText("视场 稳定")
        assert dialog._settings_page_title.text() == "采集与试验"
        dialog.select_trial_section("evidence")
        page = dialog._trial_settings_widget
        assert page.tabs.currentIndex() == 1
        page.cleanup_button.click()
        assert calls == ["clean"]
        dialog.set_trial_cleanup_status("正在清理", busy=True)
        assert not page.cleanup_button.isEnabled()
        dialog.set_trial_cleanup_status("已清理 3 张")
        assert page.cleanup_button.isEnabled() and page.cleanup_status.text() == "已清理 3 张"
        dialog.select_trial_section("analysis")
        page.imaging_mode.setCurrentIndex(page.imaging_mode.findData("double_edge"))
        page.flat_enabled.setChecked(True)
        assert dialog.app_settings().trial_preferences.optical_analysis["imaging_mode"] == "double_edge"
        assert dialog.app_settings().trial_preferences.optical_analysis["flat_enabled"]
        dialog.select_page("screenshot")
        assert dialog._settings_page_title.text() == "截图工具"
        assert dialog._tabs.count() == 7
        assert dialog._tabs.tabText(4) == "数字化切片"
    finally:
        dialog.close()


def test_new_optical_resource_merges_without_overwriting_unsaved_settings_draft(desktop_application, tmp_path):
    from fdm.ui.dialogs import SettingsDialog
    original = TrialPreferences().normalized_copy()
    dialog = SettingsDialog(AppSettings(trial_preferences=original), document=None)
    try:
        page = dialog._trial_settings_widget
        page.controls["font_size"].setValue(23)
        page.controls["algorithm"].setCurrentIndex(page.controls["algorithm"].findData("multiscale"))
        page.imaging_mode.setCurrentIndex(page.imaging_mode.findData("double_edge"))
        page.flat_enabled.setChecked(True)
        reference = {"path": str(tmp_path / "new-reference.npz"), "sha256": "a" * 64,
                     "context": {"camera": "one"}}
        captured = replace(original, font_size=17, algorithm="optical_adaptive",
            optical_analysis={**original.optical_analysis, "flat_reference": reference,
                              "reference_context": {"camera": "one"}}).normalized_copy()
        dialog.update_trial_optical_resources(captured)
        draft = dialog.app_settings().trial_preferences
        assert draft.font_size == 23 and draft.algorithm == "multiscale"
        assert draft.optical_analysis["imaging_mode"] == "double_edge"
        assert draft.optical_analysis["flat_enabled"]
        assert draft.optical_analysis["flat_reference"] == reference
        assert draft.optical_analysis["reference_context"] == {"camera": "one"}
        assert "已配置" in page.reference_status.text()
        assert original.optical_analysis["flat_reference"] is None
        dialog.reject()
        assert captured.font_size == 17 and captured.algorithm == "optical_adaptive"
    finally:
        dialog.close()


def test_restore_keeps_default_draft_until_a_different_reference_arrives(desktop_application, tmp_path):
    from fdm.ui.dialogs import SettingsDialog
    reference = {"path": str(tmp_path / "old.npz"), "sha256": "a" * 64, "context": {}}
    original = TrialPreferences(optical_analysis={"flat_reference": reference,
        "reference_context": {"camera": "one"}, "flat_enabled": True}).normalized_copy()
    dialog = SettingsDialog(AppSettings(trial_preferences=original), document=None)
    try:
        dialog.select_page("trial")
        page = dialog._trial_settings_widget
        dialog._restore_page_defaults_button.click()
        assert dialog.app_settings().trial_preferences.optical_analysis["flat_reference"] is None
        # A settings notification changes a regular preference but contains
        # exactly the same resources; it must not undo Restore Defaults.
        dialog.update_trial_optical_resources(replace(original, font_size=18))
        assert dialog.app_settings().trial_preferences == TrialPreferences().normalized_copy()
        assert "尚未" in page.reference_status.text()
        page.controls["font_size"].setValue(19)
        new_reference = {"path": str(tmp_path / "new.npz"), "sha256": "b" * 64, "context": {}}
        updated = replace(original, optical_analysis={**original.optical_analysis,
            "flat_reference": new_reference, "reference_context": {"camera": "two"}})
        dialog.update_trial_optical_resources(updated)
        draft = dialog.app_settings().trial_preferences
        assert draft.font_size == 19 and not draft.optical_analysis["flat_enabled"]
        assert draft.optical_analysis["flat_reference"] == new_reference
        assert draft.optical_analysis["reference_context"] == {"camera": "two"}
        assert "已配置" in page.reference_status.text()
    finally:
        dialog.close()
