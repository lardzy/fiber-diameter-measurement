"""Alignment controls use accepted shared display state, never project edits."""
import json

import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QColor, QPalette
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QWidget

from fdm.ui.acquisition_alignment_button import ALIGNMENT_CHOICES, AcquisitionAlignmentButton
from test_acquisition_second_round import workflow


def click_direction(button, key):
    button.menu().popup(button.mapToGlobal(QPoint(0, button.height())))
    workflow.settle()
    QTest.mouseClick(button.direction_buttons[key], Qt.MouseButton.LeftButton)
    workflow.settle()


def test_picker_displays_only_accepted_state_and_keeps_toolbar_focus():
    host = QWidget()
    picker = AcquisitionAlignmentButton(host)
    selected = []
    picker.alignmentSelected.connect(selected.append)
    host.show()
    workflow.settle()
    try:
        assert picker.focusPolicy() == Qt.FocusPolicy.NoFocus
        assert picker.alignment() == "free"
        assert picker.text() == "自由位置"
        assert not any(button.isChecked() for button in picker.direction_buttons.values())
        for key, _, _ in ALIGNMENT_CHOICES:
            click_direction(picker, key)
            assert selected[-1] == key
            # The popup never assumes an intent was accepted or persisted.
            assert picker.alignment() == "free"
            assert not any(button.isChecked() for button in picker.direction_buttons.values())
            assert not picker.menu().isVisible()
        original_width = picker.sizeHint().width()
        picker.set_alignment("bottom_right")
        assert picker.text() == "画面对齐"
        assert "右下" in picker.toolTip()
        assert picker.direction_buttons["bottom_right"].isChecked()
        assert sum(button.isChecked() for button in picker.direction_buttons.values()) == 1
        assert picker.sizeHint().width() == original_width
        picker.set_alignment("free")
        assert len(selected) == 9
    finally:
        host.close()
        host.deleteLater()


def test_picker_theme_icons_and_narrow_fallback_preserve_accessible_state():
    # A top-level Windows tool button inherits the native minimum window
    # width. Test the actual toolbar-child geometry instead of window chrome.
    host = QWidget()
    host.resize(240, 100)
    picker = AcquisitionAlignmentButton(host)
    host.show()
    try:
        picker.set_alignment("top_left")
        palette = picker.palette()
        palette.setColor(QPalette.ColorRole.ButtonText, QColor("#f3f3f3"))
        picker.setPalette(palette)
        for key, _, _ in ALIGNMENT_CHOICES:
            icon = picker.direction_buttons[key].icon()
            assert not icon.isNull()
            assert not icon.pixmap(36, 36).isNull()
        picker.resize(picker.minimumSizeHint())
        workflow.settle()
        assert picker.toolButtonStyle() == Qt.ToolButtonStyle.ToolButtonIconOnly
        assert picker.accessibleName() == "画面对齐"
        assert "左上" in picker.accessibleDescription()
        picker.resize(picker.sizeHint())
        workflow.settle()
        assert picker.toolButtonStyle() == Qt.ToolButtonStyle.ToolButtonTextBesideIcon
    finally:
        host.close()
        host.deleteLater()


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_trial_alignment_shared_across_pages_without_project_edits(tmp_path, theme):
    with workflow.WorkflowHarness(tmp_path / "profile", theme=theme) as harness:
        harness.enter()
        content = harness.select(1)
        content_revision = content.session.revision
        click_direction(content.alignment_button, "top_left")
        assert content.alignment_button.alignment() == "top_left"
        diameter = harness.select(2)
        assert diameter.alignment_button.alignment() == "top_left"
        diameter_revision = diameter.session.revision
        click_direction(diameter.alignment_button, "bottom_right")
        assert content.alignment_button.alignment() == "bottom_right"
        harness.select(1)
        assert content.alignment_button.alignment() == "bottom_right"
        assert content.session.revision == content_revision
        assert diameter.session.revision == diameter_revision
        saved = json.loads(content.display_binding.store.path.read_text(encoding="utf-8"))
        assert saved["display"]["alignment"] == "bottom_right"
        # Manual pan is published by the core binding as free; hidden pages
        # must reflect it too, even before they have another source frame.
        content.display_binding.store.set_alignment("free")
        workflow.settle()
        assert content.alignment_button.text() == diameter.alignment_button.text() == "自由位置"
        harness.assert_clean()


@pytest.mark.parametrize("failure", ["busy", "write_error"])
def test_trial_alignment_rejection_restores_picker_and_shared_preference(tmp_path, monkeypatch, failure):
    with workflow.WorkflowHarness(tmp_path / "profile") as harness:
        harness.enter()
        page = harness.select(1)
        click_direction(page.alignment_button, "center")
        before = page.display_binding.store.path.read_bytes()
        with monkeypatch.context() as patcher:
            if failure == "busy":
                # Keep the busy interval deterministic across native event
                # loop speeds. UI refreshes can inspect a normal confirmation
                # shape, while this UI-only test does not complete a request.
                import time
                patcher.setattr(page.controller, "_confirmation", {
                    "kind": "manual", "intents": [], "context": None,
                    "candidates": None, "sequence": None, "started": time.perf_counter(),
                })
                patcher.setattr(page.controller, "_poll_confirmation", lambda: None)
            else:
                def fail_save(*_):
                    raise OSError("模拟显示偏好写入失败")
                patcher.setattr(page.display_binding.store, "_save", fail_save)
            click_direction(page.alignment_button, "top")
            assert page.alignment_button.alignment() == "center"
            assert page.display_binding.store.preferences.alignment == "center"
            assert page.alignment_button.direction_buttons["center"].isChecked()
            assert not page.alignment_button.direction_buttons["top"].isChecked()
            assert page.statusBar().currentMessage()
        assert page.display_binding.store.path.read_bytes() == before
        harness.assert_clean()


def test_alignment_does_not_grow_narrow_trial_with_left_records(tmp_path):
    with workflow.WorkflowHarness(tmp_path / "profile", size=(1067, 600)) as harness:
        harness.enter()
        page = harness.select(1)
        QTest.mouseClick(page.records_toggle, Qt.MouseButton.LeftButton)
        workflow.settle()
        harness.assert_size()
        controls = (page.analyze_button, page.auto_analysis, page.candidate_display_button,
                    page.display_choice, page.alignment_button)
        for control in controls:
            assert control.isVisible()
            assert page.view_controls.rect().contains(control.mapTo(page.view_controls, QPoint()))
            assert page.view_controls.rect().contains(control.mapTo(page.view_controls, control.rect().bottomRight()))
        assert page.display_choice.width() >= page.display_choice.minimumSizeHint().width()
        click_direction(page.alignment_button, "right")
        harness.assert_size()
        assert page.alignment_button.alignment() == "right"
        harness.assert_clean()
