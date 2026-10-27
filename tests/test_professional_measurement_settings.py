"""Independent application preference, stable navigation and real label preview."""
from __future__ import annotations

import os
from pathlib import Path
import sys
from unittest.mock import patch

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PySide6.QtWidgets import QApplication, QGroupBox

from fdm.settings import AppSettings, AppSettingsIO, MeasurementLabelStyleSettings
from fdm.ui.dialogs import SettingsDialog
from fdm.ui.professional_rendering import draw_professional_measurements


@pytest.fixture(scope="module")
def application():
    return QApplication.instance() or QApplication([])


@pytest.mark.parametrize("value,expected", [
    (0, 0), (2, 2), (8, 8), (-3, 0), (99, 8), ("6", 6),
    (None, 2), (True, 2), ("invalid", 2), ({}, 2), ([], 2),
    (float("nan"), 2), (float("inf"), 2), (-float("inf"), 2),
])
def test_precision_normalizes_corrupt_settings_without_affecting_other_labels(value, expected):
    settings = AppSettings.from_dict({
        "professional_measurement_label_decimals": value,
        "length_measurement_label_style": {"decimals": 7},
        "area_measurement_label_style": {"decimals": 5},
    })
    assert settings.professional_measurement_label_decimals == expected
    assert settings.length_measurement_label_style.decimals == 7
    assert settings.area_measurement_label_style.decimals == 5
    assert AppSettings(professional_measurement_label_decimals=value).normalized_copy().professional_measurement_label_decimals == expected


def test_missing_precision_uses_two_instead_of_inheriting_legacy_length_style():
    assert AppSettings.from_dict({"measurement_label_decimals": 8}).professional_measurement_label_decimals == 2


def test_precision_survives_save_load_and_settings_import(tmp_path):
    source, target = tmp_path / "source.json", tmp_path / "imported.json"
    AppSettingsIO.save(AppSettings(professional_measurement_label_decimals=8), source)
    assert AppSettingsIO.load(source).professional_measurement_label_decimals == 8
    imported, actual = AppSettingsIO.replace_with_file(source, target)
    assert actual == target and imported.professional_measurement_label_decimals == 8
    assert AppSettingsIO.load(target).professional_measurement_label_decimals == 8


def test_precision_page_draft_cancel_restore_and_real_preview(application):
    settings = AppSettings(
        professional_measurement_label_decimals=7,
        length_measurement_label_style=MeasurementLabelStyleSettings(decimals=6),
        area_measurement_label_style=MeasurementLabelStyleSettings(decimals=4),
    )
    dialog = SettingsDialog(settings, document=None)
    try:
        assert dialog.select_page("professional") == 2
        editor = dialog._professional_measurement_label_decimals
        preview = dialog._professional_measurement_style_preview
        assert editor.minimum() == 0 and editor.maximum() == 8
        assert editor.objectName() == "professionalMeasurementLabelDecimals"
        assert editor.accessibleName() == "结果文字小数位"
        editor.setValue(8)
        preview.resize(420, 140)
        with patch("fdm.ui.professional_rendering.draw_professional_measurements",
                   wraps=draw_professional_measurements) as draw:
            assert not preview.grab().isNull()
        assert draw.call_args.kwargs["label_decimals"] == 8
        definition = draw.call_args.args[1].professional_measurements[0]
        assert definition.annotation_layout.leader_style == "dual"
        assert dialog.app_settings().professional_measurement_label_decimals == 8
        assert settings.professional_measurement_label_decimals == 7
        dialog._restore_page_defaults_button.click()
        restored = dialog.app_settings()
        assert restored.professional_measurement_label_decimals == preview._decimals == 2
        assert restored.length_measurement_label_style.decimals == 6
        assert restored.area_measurement_label_style.decimals == 4
        dialog.reject()
        assert settings.professional_measurement_label_decimals == 7
    finally:
        dialog.close()


def test_page_keys_find_screenshot_and_restore_only_the_selected_category(application):
    dialog = SettingsDialog(AppSettings(professional_measurement_label_decimals=8), document=None)
    try:
        dialog._settings_search.setText("专业 精度")
        application.processEvents()
        assert dialog._settings_page_title.text() == "专业测量"
        dialog.select_page("screenshot")
        assert dialog._settings_search.text() == ""
        assert dialog._settings_page_title.text() == "截图工具"
        with patch.object(dialog._screenshot_settings_widget, "restore_defaults") as restore:
            dialog._restore_page_defaults_button.click()
        restore.assert_called_once()
        assert dialog.app_settings().professional_measurement_label_decimals == 8
        with pytest.raises(ValueError):
            dialog.select_page("missing-page")
        assert dialog._tabs.count() == 7
        assert dialog._tabs.tabText(4) == "数字化切片"
    finally:
        dialog.close()


def test_legacy_line_controls_share_the_result_preview_group(application):
    dialog = SettingsDialog(AppSettings(), document=None)
    try:
        preview = dialog._length_measurement_style_preview
        group = preview.parentWidget()
        assert isinstance(group, QGroupBox) and group.title() == "直径/长度结果"
        assert dialog._endpoint_style_combo.parentWidget() is group
        assert dialog._default_measurement_color.parentWidget() is group
    finally:
        dialog.close()
