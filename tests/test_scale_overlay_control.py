import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QToolBar, QToolButton

from fdm.scale_overlay import ScaleOverlaySpec
from fdm.settings import AppSettings
from fdm.ui.scale_overlay_control import ScaleOverlayStatusButton
from fdm.services.export_service import ExportScope, ExportSelection
from fdm.ui.dialogs import ExportOptionsDialog
from test_scale_overlay_editor import window  # noqa: F401


def test_scale_status_button_distinguishes_open_editor_from_visibility():
    button = ScaleOverlayStatusButton()
    edits, visibility = [], []
    button.editRequested.connect(lambda: edits.append(True))
    button.visibilityRequested.connect(visibility.append)
    assert not button.isEnabled()
    button.setScaleState(True, False, True)
    button.click()
    assert edits == [True]
    assert button.isChecked() and visibility == []
    button._visible_action.trigger()
    assert visibility == [False]
    button.setScaleState(False, False, True)
    assert not button.isChecked() and button.text() == "比例尺：关"
    assert button.popupMode() == QToolButton.ToolButtonPopupMode.MenuButtonPopup
    assert button.focusPolicy() == Qt.FocusPolicy.NoFocus
    button.close()


def test_bottom_control_reopens_finished_or_closed_editor(window):
    window.resize(1440, 1000)
    window.show()
    QApplication.processEvents()
    control = window._scale_status_button
    assert control.parentWidget() == window.statusBar()
    assert control.x() < window._object_snap_status_button.x()
    for toolbar in window.findChildren(QToolBar):
        assert window.scale_preview_action not in toolbar.actions()
        assert not any(
            button.defaultAction() == window.scale_preview_action
            for button in toolbar.findChildren(QToolButton)
        )
    control.click()
    preview = window.scale_preview
    assert preview.visible and preview.editing and not preview.panel.isHidden()
    assert control.isChecked() and control.text() == "比例尺：编辑"
    preview.finish()
    assert preview.panel.isHidden() and control.text() == "比例尺：显示"
    control.click()
    assert preview.visible and preview.editing and not preview.panel.isHidden()
    window._inspector_dock.hide()
    control.click()
    assert not window._inspector_dock.isHidden()
    control._visible_action.trigger()
    assert not preview.visible and not control.isChecked()


def test_fresh_defaults_red_ticks_bottom_right_and_saved_colors_preserved():
    fresh = AppSettings.from_dict({})
    spec = ScaleOverlaySpec.from_legacy_settings(fresh)
    assert spec.color == spec.text_color == "#FF0000"
    assert spec.style == "ticks" and spec.position == "bottom_right"
    saved = AppSettings.from_dict(
        {
            "scale_overlay_color": "#345678",
            "scale_overlay_text_color": "#FFFFFF",
        }
    )
    migrated = ScaleOverlaySpec.from_legacy_settings(saved)
    assert migrated.color == "#345678" and migrated.text_color == "#FFFFFF"


@pytest.mark.parametrize("scope", [ExportScope.CURRENT, ExportScope.ALL_OPEN])
def test_export_range_obeys_selection_when_several_images_are_open(scope):
    dialog = ExportOptionsDialog(
        ExportSelection(include_scale_overlay=True, scope=scope),
        allow_all_scope=True,
        watermark_count_current=0,
        watermark_count_all=1,
    )
    try:
        assert dialog.selection().scope == scope
        assert dialog._watermark.isChecked() == (scope == ExportScope.ALL_OPEN)
    finally:
        dialog.close()


def test_scale_editor_export_does_not_reselect_declined_watermark(window, monkeypatch):
    preview = window.scale_preview
    preview.begin(
        ExportSelection(
            include_combined_overlay=True,
            include_csv=True,
            include_annotations=False,
            include_watermark=False,
            scope=ExportScope.ALL_OPEN,
        )
    )
    selections = []
    monkeypatch.setattr(window, "export_results", selections.append)
    preview.export()
    dialog = ExportOptionsDialog(
        selections[0],
        allow_all_scope=True,
        watermark_count_current=1,
        watermark_count_all=2,
    )
    try:
        actual = dialog.selection()
        assert actual.include_combined_overlay and actual.include_csv
        assert actual.scope == ExportScope.ALL_OPEN
        assert not actual.include_watermark and not actual.include_annotations
    finally:
        dialog.close()


def test_declined_watermark_stays_unchecked_after_scope_changes():
    dialog = ExportOptionsDialog(
        ExportSelection(include_scale_overlay=True),
        allow_all_scope=True,
        watermark_count_current=0,
        watermark_count_all=1,
    )
    try:
        dialog._scope_all.setChecked(True)
        assert dialog.selection().include_watermark
        dialog._watermark.setChecked(False)
        dialog._scope_current.setChecked(True)
        dialog._scope_all.setChecked(True)
        assert not dialog.selection().include_watermark
    finally:
        dialog.close()


def test_stroke_control_is_actual_rendered_width_for_every_style(window):
    preview = window.scale_preview
    preview.begin()
    panel = preview.panel
    for index in range(panel.style_combo.count()):
        panel.style_combo.setCurrentIndex(index)
        panel.stroke.setValue(6.0)
        layout, error = preview.current_layout()
        assert not error and layout.stroke == 6.0
        assert not panel.style_combo.itemIcon(index).isNull()


def test_unrelated_style_change_cannot_reinterpret_physical_length_as_pixels(window):
    preview = window.scale_preview
    preview.begin()
    preview.change(length_mode="custom", length=50, unit="um")
    window.current_document().calibration = None
    preview.refresh()
    assert preview.current_layout()[0] is None
    preview.panel.stroke.setValue(4.0)
    preview.panel.position.setCurrentIndex(1)
    preview.panel.bold.setChecked(False)
    assert preview.spec.unit == "um" and preview.spec.length == 50
    assert preview.current_layout()[0] is None
    assert not preview.panel.export_button.isEnabled()
    preview.panel.length_mode.setCurrentIndex(0)
    value = preview.current_layout()[0].value
    preview.panel.length_mode.setCurrentIndex(1)
    assert preview.spec.unit == "px" and preview.spec.length == value
    assert preview.current_layout()[0] is not None
