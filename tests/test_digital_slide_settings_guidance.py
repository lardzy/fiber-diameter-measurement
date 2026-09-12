from copy import deepcopy
from dataclasses import replace
from unittest.mock import patch

import pytest
from PySide6.QtWidgets import QApplication

from fdm.services.slide_capture_geometry import (
    calibrated_capture_settings, calibration_signature, scaled_capture_size,
)
from fdm.settings import AppSettings, DigitalSlideAcquisitionProfile
from fdm.ui.dialogs import SettingsDialog
from fdm.ui.digital_slide_settings import capture_settings_guidance


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def calibrated_settings():
    settings = AppSettings(
        selected_capture_device_id="test-camera",
        digital_slide_pixel_stride_mode="calibrated_overlap",
        digital_slide_overlap_percent=20,
        digital_slide_x_stage_step=-8500,
        digital_slide_y_stage_step=6600,
    ).normalized_copy()
    settings.digital_slide_xy_calibration = {
        "signature": calibration_signature(settings, (1600, 1198)),
        "capture_frame_size": [5280, 3956],
        "x": {"reliable": True, "pixels_per_step": .2, "cross_per_step": .001, "uncertainty_px": 2},
        "y": {"reliable": True, "pixels_per_step": .18, "cross_per_step": -.001, "uncertainty_px": 2},
    }
    settings.digital_slide_profiles[0].values = AppSettings._digital_slide_profile_values_from_settings(settings)
    return settings


def select_mode(dialog, mode):
    dialog._digital_slide_pixel_stride_mode_combo.setCurrentIndex(
        dialog._digital_slide_pixel_stride_mode_combo.findData(mode))


def test_guidance_uses_saved_resolution_and_actual_frozen_capture_steps(qapp):
    settings = calibrated_settings()
    original = deepcopy(settings.to_dict())
    dialog = SettingsDialog(settings, document=None, digital_slide_frame_size=(5280, 3956))
    try:
        assert "当前相机帧" in dialog._digital_slide_calibration_context.text()
        assert "1600 × 1198" in dialog._digital_slide_calibration_context.text()
        assert "-6400 steps → 1280 px" in dialog._digital_slide_step_summary.text()
        assert "5324 steps → 958 px" in dialog._digital_slide_step_summary.text()
        assert not dialog._digital_slide_x_stage_step_spin.isEnabled()
        for overlap in (10, 20, 25):
            dialog._digital_slide_overlap_spin.setValue(overlap)
            draft = dialog.app_settings().normalized_copy()
            width, height, _ = scaled_capture_size(5280, 3956, draft.digital_slide_capture_max_width)
            frozen = calibrated_capture_settings(draft, (width, height), source_frame_size=(5280, 3956))
            assert f"{frozen.digital_slide_x_stage_step} steps → {frozen.digital_slide_x_pixel_stride} px" in dialog._digital_slide_step_summary.text()
            assert draft.digital_slide_x_stage_step == -8500  # Manual fallback stays intact.
            assert draft.digital_slide_y_stage_step == 6600
            if overlap == 10:
                assert "有效重叠不足" in dialog._digital_slide_mode_note.text()
            elif overlap == 25:
                assert "14%" in dialog._digital_slide_mode_note.text()
        assert settings.to_dict() == original
    finally:
        dialog.close()


def test_mode_switches_keep_manual_values_and_explicit_legacy_overlap(qapp):
    settings = AppSettings(digital_slide_overlap_percent=2, digital_slide_x_stage_step=-1234,
        digital_slide_x_pixel_stride=1777, digital_slide_y_pixel_stride=1333).normalized_copy()
    dialog = SettingsDialog(settings, document=None, digital_slide_frame_size=(5280, 3956))
    try:
        assert dialog._digital_slide_overlap_label.text() == "排布重叠"
        assert "不改变电机步距" in dialog._digital_slide_mode_note.text()
        assert not dialog._digital_slide_x_pixel_stride_spin.isEnabled()
        select_mode(dialog, "calibrated_overlap")
        dialog._on_digital_slide_stride_mode_activated(1)
        assert dialog._digital_slide_overlap_spin.value() == 2
        assert "当前不能" in dialog._digital_slide_mode_note.text()
        select_mode(dialog, "manual_pixels")
        assert dialog._digital_slide_overlap_spin.isHidden()
        assert dialog._digital_slide_x_pixel_stride_spin.isEnabled()
        draft = dialog.app_settings()
        assert (draft.digital_slide_x_stage_step, draft.digital_slide_x_pixel_stride, draft.digital_slide_y_pixel_stride) == (-1234, 1777, 1333)
        assert draft.digital_slide_overlap_percent == 2
        dialog.reject()
        assert settings.digital_slide_pixel_stride_mode == "auto_overlap"
    finally:
        dialog.close()


def test_changing_resolution_or_direction_invalidates_guidance_without_losing_calibration(qapp):
    settings = calibrated_settings()
    dialog = SettingsDialog(settings, document=None, digital_slide_frame_size=(5280, 3956))
    try:
        combo = dialog._digital_slide_capture_width_combo
        combo.setCurrentIndex(combo.findData(3200))
        assert "保存尺寸" in dialog._digital_slide_calibration_status.text()
        assert "当前不能" in dialog._digital_slide_mode_note.text()
        combo.setCurrentIndex(combo.findData(1600))
        assert "-6400 steps" in dialog._digital_slide_step_summary.text()
        dialog._digital_slide_reverse_x_axis_checkbox.setChecked(True)
        assert "坐标方向" in dialog._digital_slide_calibration_status.text()
        dialog._digital_slide_reverse_x_axis_checkbox.setChecked(False)
        assert dialog.app_settings().digital_slide_xy_calibration == settings.digital_slide_xy_calibration
    finally:
        dialog.close()
    # A different original sensor size can produce the very same saved size.
    stale = capture_settings_guidance(settings, (2640, 1978))
    assert "相机原始尺寸" in stale.calibration
    assert "当前不能" in stale.notice


def test_missing_or_partial_calibration_never_claims_a_verified_live_camera():
    settings = calibrated_settings()
    guidance = capture_settings_guidance(settings)
    assert "历史档案估算" in guidance.context
    assert "尚未核对实时相机" in guidance.notice
    assert "-6400 steps" in guidance.result
    del settings.digital_slide_xy_calibration["y"]
    partial = capture_settings_guidance(settings, (5280, 3956))
    assert "X 已有校准" in partial.calibration
    assert "Y 待校准" in partial.calibration
    assert "当前不能" in partial.notice
    settings.digital_slide_xy_calibration["y"] = {"reliable": True, "pixels_per_step": "invalid"}
    assert "Y 待校准" in capture_settings_guidance(settings).calibration


def test_legacy_profile_without_original_camera_dimensions_does_not_invent_them():
    settings = calibrated_settings()
    del settings.digital_slide_xy_calibration["capture_frame_size"]
    guidance = capture_settings_guidance(settings)
    assert "历史档案估算" in guidance.context
    assert "-6400 steps" in guidance.result
    changed = capture_settings_guidance(replace(settings, digital_slide_capture_max_width=0))
    assert "当前不能" in changed.notice
    assert "连接相机" in changed.context


def test_profile_changes_and_copies_refresh_status_and_preserve_custom_widths(qapp):
    settings = calibrated_settings()
    other = AppSettings._digital_slide_profile_values_from_settings(settings)
    other.update(digital_slide_capture_max_width=1920, digital_slide_preview_max_width=1440,
        digital_slide_pixel_stride_mode="manual_pixels", digital_slide_x_pixel_stride=1701)
    settings.digital_slide_profiles.append(DigitalSlideAcquisitionProfile("other", "另一物镜", other))
    dialog = SettingsDialog(settings, document=None, digital_slide_frame_size=(5280, 3956))
    try:
        combo = dialog._digital_slide_profile_combo
        combo.setCurrentIndex(combo.findData("other"))
        assert dialog._digital_slide_capture_width_combo.currentData() == 1920
        assert dialog._digital_slide_preview_width_combo.currentData() == 1440
        assert dialog._digital_slide_x_pixel_stride_spin.value() == 1701
        assert "采集配置" in dialog._digital_slide_calibration_status.text()
        reopened = SettingsDialog(dialog.app_settings(), document=None)
        try:
            assert reopened._digital_slide_capture_width_combo.currentData() == 1920
            assert reopened._digital_slide_preview_width_combo.currentData() == 1440
        finally:
            reopened.close()
        combo.setCurrentIndex(combo.findData("default"))
        assert "-6400 steps" in dialog._digital_slide_step_summary.text()
        with patch.object(dialog, "_prompt_digital_slide_profile_name", return_value="新物镜"):
            dialog._duplicate_digital_slide_profile()
        assert "采集配置" in dialog._digital_slide_calibration_status.text()
        assert "当前不能" in dialog._digital_slide_mode_note.text()
    finally:
        dialog.close()


def test_reset_only_active_profile_clears_its_calibration_and_keeps_other_profiles(qapp):
    settings = calibrated_settings()
    other = AppSettings._digital_slide_profile_values_from_settings(settings)
    other["digital_slide_x_stage_step"] = -7000
    settings.digital_slide_profiles.append(DigitalSlideAcquisitionProfile("other", "另一物镜", other))
    dialog = SettingsDialog(settings, document=None)
    try:
        dialog._settings_navigation.setCurrentRow(5)
        dialog._digital_slide_profile_combo.setCurrentIndex(1)
        dialog._restore_current_page_defaults()
        saved = dialog.app_settings()
        assert saved.digital_slide_active_profile_id == "other"
        assert not saved.digital_slide_xy_calibration
        assert saved.digital_slide_overlap_percent == 0
        assert saved.digital_slide_pixel_stride_mode == "auto_overlap"
        original_profile = next(p for p in saved.digital_slide_profiles if p.profile_id == "default")
        assert original_profile.values["digital_slide_xy_calibration"] == settings.digital_slide_xy_calibration
        assert original_profile.values["digital_slide_x_stage_step"] == -8500
    finally:
        dialog.close()


def test_acquisition_lock_also_blocks_reset_and_keeps_frozen_parameters(qapp):
    settings = calibrated_settings()
    dialog = SettingsDialog(settings, document=None, digital_slide_locked=True)
    try:
        dialog._settings_navigation.setCurrentRow(5)
        assert not dialog._digital_slide_calibration_button.isEnabled()
        assert not dialog._digital_slide_overlap_spin.isEnabled()
        dialog._restore_current_page_defaults()
        saved = dialog.app_settings()
        assert saved.digital_slide_xy_calibration == settings.digital_slide_xy_calibration
        assert saved.digital_slide_overlap_percent == 20
        assert saved.digital_slide_x_stage_step == -8500
    finally:
        dialog.close()


def test_long_profile_name_does_not_push_settings_outside_a_small_window(qapp):
    settings = calibrated_settings()
    settings.digital_slide_profiles[0].name = "激光共聚焦物镜与相机采集参数配置用于验证小窗口显示" * 5
    settings = settings.normalized_copy()
    dialog = SettingsDialog(settings, document=None)
    try:
        dialog._preferred_size_applied = True
        dialog.resize(760, 620)
        dialog._settings_navigation.setCurrentRow(5)
        dialog.show()
        qapp.processEvents()
        page = dialog._settings_pages.currentWidget()
        assert page.horizontalScrollBar().maximum() == 0
        assert dialog.width() == 760
        assert dialog._digital_slide_profile_combo.toolTip() == settings.digital_slide_profiles[0].name
    finally:
        dialog.close()


def test_jpeg_quality_stays_visible_inside_its_row_and_survives_format_switch(qapp):
    dialog = SettingsDialog(AppSettings(), document=None)
    try:
        dialog._preferred_size_applied = True
        dialog.resize(760, 620)
        dialog._settings_navigation.setCurrentRow(5)
        dialog.show()
        combo = dialog._digital_slide_capture_codec_combo
        row = dialog._digital_slide_capture_quality_row
        assert row.isHidden()
        combo.setCurrentIndex(combo.findData("jpeg"))
        dialog._digital_slide_capture_quality_slider.setValue(95)
        qapp.processEvents()
        value = dialog._digital_slide_capture_quality_label
        dialog._settings_pages.currentWidget().ensureWidgetVisible(value, 0, 60)
        qapp.processEvents()
        assert value.text() == "95"
        assert row.rect().contains(value.geometry())
        assert not value.visibleRegion().isEmpty()
        combo.setCurrentIndex(combo.findData("png"))
        assert row.isHidden()
        assert dialog.app_settings().digital_slide_capture_jpeg_quality == 95
    finally:
        dialog.close()
