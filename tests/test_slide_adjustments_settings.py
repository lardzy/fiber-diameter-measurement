from __future__ import annotations

import math
import pytest

from fdm.settings import AppSettings, AppSettingsIO, AppThemeMode, DIGITAL_SLIDE_PROFILE_FIELDS


def test_old_settings_default_to_unremembered_neutral_view():
    settings = AppSettings.from_dict({})
    assert not settings.digital_slide_remember_view_adjustments
    assert settings.digital_slide_view_brightness == 0
    assert settings.digital_slide_view_exposure_tenths == 0


def test_view_defaults_round_trip_without_entering_capture_profiles(tmp_path):
    original = AppSettings(digital_slide_remember_view_adjustments=True,
                           digital_slide_view_brightness=-37,
                           digital_slide_view_exposure_tenths=14)
    path = tmp_path / '设置.json'
    AppSettingsIO.save(original, path)
    restored = AppSettingsIO.load(path)
    assert restored.digital_slide_remember_view_adjustments
    assert restored.digital_slide_view_brightness == -37
    assert restored.digital_slide_view_exposure_tenths == 14
    fields = {'digital_slide_remember_view_adjustments', 'digital_slide_view_brightness',
              'digital_slide_view_exposure_tenths'}
    assert not fields.intersection(DIGITAL_SLIDE_PROFILE_FIELDS)
    for profile in restored.to_dict()['digital_slide_profiles']:
        assert not fields.intersection(profile['values'])


@pytest.mark.parametrize('invalid', [None, 'bad', math.nan, math.inf, -math.inf, [], {}])
def test_invalid_numeric_defaults_fall_back_to_zero(invalid):
    settings = AppSettings.from_dict(dict(digital_slide_remember_view_adjustments=True,
        digital_slide_view_brightness=invalid, digital_slide_view_exposure_tenths=invalid))
    assert settings.digital_slide_view_brightness == 0
    assert settings.digital_slide_view_exposure_tenths == 0
    direct = AppSettings(digital_slide_remember_view_adjustments=True,
        digital_slide_view_brightness=invalid, digital_slide_view_exposure_tenths=invalid).normalized_copy()
    assert direct.digital_slide_view_brightness == direct.digital_slide_view_exposure_tenths == 0


@pytest.mark.parametrize('remember', [False, None, 'false', 'true', 1, [], {}])
def test_only_explicit_boolean_true_can_enable_persistence(remember):
    result = AppSettings.from_dict(dict(digital_slide_remember_view_adjustments=remember,
        digital_slide_view_brightness=40, digital_slide_view_exposure_tenths=12)).to_dict()
    assert result['digital_slide_remember_view_adjustments'] is False
    assert result['digital_slide_view_brightness'] == result['digital_slide_view_exposure_tenths'] == 0


def test_out_of_range_settings_clamp_and_disabled_settings_drop_stale_values():
    settings = AppSettings(digital_slide_remember_view_adjustments=True,
        digital_slide_view_brightness=101, digital_slide_view_exposure_tenths=-41).normalized_copy()
    assert settings.digital_slide_view_brightness == 100
    assert settings.digital_slide_view_exposure_tenths == -40
    settings.digital_slide_remember_view_adjustments = False
    restored = AppSettings.from_dict(settings.to_dict())
    assert restored.digital_slide_view_brightness == restored.digital_slide_view_exposure_tenths == 0


@pytest.mark.parametrize('theme', [None, AppThemeMode.DARK, AppThemeMode.WARM])
def test_preferences_round_trip_and_theme_changes_preserve_view_defaults(
    desktop_application, tmp_path, theme,
):
    from fdm.ui.dialogs import SettingsDialog

    original = AppSettings(theme_mode=AppThemeMode.LIGHT,
        digital_slide_remember_view_adjustments=True,
        digital_slide_view_brightness=45, digital_slide_view_exposure_tenths=13)
    dialog = SettingsDialog(original, document=None)
    try:
        if theme is not None:
            dialog._theme_mode_combo.setCurrentIndex(dialog._theme_mode_combo.findData(theme))
        result = dialog.app_settings()
        path = tmp_path / '首选项.json'
        AppSettingsIO.save(result, path)
        restored = AppSettingsIO.load(path)
        assert restored.theme_mode == (theme or AppThemeMode.LIGHT)
        assert restored.digital_slide_remember_view_adjustments
        assert restored.digital_slide_view_brightness == 45
        assert restored.digital_slide_view_exposure_tenths == 13
        assert original.theme_mode == AppThemeMode.LIGHT
    finally:
        dialog.close()
        dialog.deleteLater()
