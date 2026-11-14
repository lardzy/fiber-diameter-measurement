"""Real workbench routes the same content controls through the floating dock."""
from pathlib import Path
import sys

import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/validation"))
from validate_workflow_redesign import WorkflowHarness, settle


@pytest.fixture
def harness(tmp_path):
    with WorkflowHarness(tmp_path / "profile", size=(1600, 900)) as value:
        value.enter()
        yield value
        assert not value.slot_errors, "\n".join(value.slot_errors)
        assert not value.modal_messages


def test_content_sidebar_navigation_keeps_session_and_user_visibility(harness):
    page = harness.select(1)
    hub = harness.hub
    sidebar = hub.trial_sidebar
    assert sidebar.workspace is page and sidebar.is_open
    assert hub.trial_panel_button.isVisible() and hub.trial_panel_button.isChecked()
    assert page.body_splitter.count() == 1
    original_session = page.session
    sidebar.set_position("floating")
    settle()
    assert sidebar.owns_window(sidebar.dock)
    assert sidebar.contains_focus(page.operator)
    assert not sidebar.contains_focus(page.canvas)
    assert hub.select_page(0)
    settle()
    assert not sidebar.is_open and sidebar.preferences["visible"]
    assert not hub.trial_panel_button.isVisible()
    assert harness.select(1) is page
    assert page.session is original_session and sidebar.is_open
    QTest.mouseClick(hub.trial_panel_button, Qt.MouseButton.LeftButton)
    settle()
    assert not sidebar.is_open and not sidebar.preferences["visible"]
    harness.select(0)
    harness.select(1)
    assert not sidebar.is_open
    QTest.mouseClick(hub.trial_panel_button, Qt.MouseButton.LeftButton)
    settle()
    assert sidebar.is_open


def test_diameter_keeps_original_sidebar_and_rejected_switch_keeps_dock(harness, monkeypatch):
    content = harness.select(1)
    hub = harness.hub
    sidebar = hub.trial_sidebar
    monkeypatch.setattr(content, "_confirm_trial_navigation", lambda: False)
    assert hub.select_page(2) is False
    assert hub.pages.currentWidget() is content and sidebar.is_open
    assert hub.trial_panel_button.isVisible()
    monkeypatch.setattr(content, "_confirm_trial_navigation", lambda: True)
    diameter = harness.select(2)
    assert diameter.body_splitter.count() == 2
    assert diameter.trial_sidebar.parentWidget() is diameter.body_splitter
    assert not sidebar.is_open and not hub.trial_panel_button.isVisible()
    assert harness.select(1) is content
    assert sidebar.is_open and sidebar.workspace is content


def test_bottom_docking_keeps_top_navigation_and_window_size(harness):
    page = harness.select(1)
    window, hub = harness.window, harness.hub
    size = window.size()
    nav_y = hub.navigation.mapToGlobal(hub.navigation.rect().topLeft()).y()
    hub.trial_sidebar.set_position("bottom")
    settle()
    assert window.size() == size
    assert hub.navigation.mapToGlobal(hub.navigation.rect().topLeft()).y() == nav_y
    assert page.trial_sidebar.width() > 520
    assert page.trial_sidebar._dock_content_box.direction().name == "LeftToRight"
    assert page.save_button.isVisible() and page.export_button.isVisible()


def test_floating_controls_route_digits_history_and_editors_to_current_trial(harness):
    from PySide6.QtWidgets import QApplication
    from fdm.live_experiment import default_materials
    page = harness.select(1)
    page.session.set_materials(default_materials())
    page._refresh_materials()
    page._changed()
    page.keyboard_mode.setCurrentIndex(1)
    control = harness.hub.trial_sidebar
    control.set_position("floating")
    control.dock.activateWindow()
    page.records_toggle.setFocus()
    settle()
    assert QApplication.activeWindow() is control.dock
    assert page._measurement_focus_allowed()
    before = len(page.session.count_adjustments)
    QTest.keyClick(page.records_toggle, Qt.Key.Key_1)
    settle()
    assert len(page.session.count_adjustments) == before + 1
    QTest.keyClick(page.records_toggle, Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier)
    settle()
    assert len(page.session.count_adjustments) == before
    QTest.keyClick(page.records_toggle, Qt.Key.Key_Y, Qt.KeyboardModifier.ControlModifier)
    settle()
    assert len(page.session.count_adjustments) == before + 1
    page.sample_details_toggle.setChecked(True)
    control.scroll.ensureWidgetVisible(page.sample)
    page.sample.setFocus()
    settle()
    assert not page._measurement_focus_allowed()
    QTest.keyClicks(page.sample, "123")
    assert page.sample.text() == "123"
    assert len(page.session.count_adjustments) == before + 1
    QTest.keyClick(page.sample, Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier)
    assert page.sample.text() == ""
    assert len(page.session.count_adjustments) == before + 1


def test_floating_editor_wheel_does_not_switch_operator(harness):
    from PySide6.QtCore import QPoint, QPointF
    from PySide6.QtGui import QWheelEvent
    from PySide6.QtWidgets import QApplication
    page = harness.select(1)
    page.session.add_operator("乙")
    page._refresh_operators()
    control = harness.hub.trial_sidebar
    control.set_position("floating")
    control.dock.activateWindow()
    page.operator.setFocus()
    settle()
    original = page.operator.currentIndex()
    position = page.operator.rect().center()
    event = QWheelEvent(QPointF(position), QPointF(page.operator.mapToGlobal(position)),
        QPoint(), QPoint(0, -120), Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.ScrollUpdate, False)
    QApplication.sendEvent(page.operator, event)
    settle()
    assert page.operator.currentIndex() == original
