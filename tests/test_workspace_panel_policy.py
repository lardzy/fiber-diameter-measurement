"""Page capabilities win over dock actions, restored state, and camera state."""
from __future__ import annotations

import json

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QMainWindow

from fdm.ui.workspace import AdaptiveLayoutController, WorkspaceDockWidget, WorkspacePage


@pytest.fixture
def panel_window(desktop_application):
    window = QMainWindow()
    window.setCentralWidget(QLabel("Canvas"))
    window.resize(1500, 850)
    docks = {}
    for key in ("project", "inspector", "results", "professional"):
        dock = WorkspaceDockWidget(key, key + "Dock", window)
        dock.setWidget(QLabel(key))
        area = (Qt.DockWidgetArea.LeftDockWidgetArea if key == "project" else
                Qt.DockWidgetArea.BottomDockWidgetArea if key == "results" else Qt.DockWidgetArea.RightDockWidgetArea)
        window.addDockWidget(area, dock)
        docks[key] = dock
    layout = AdaptiveLayoutController(window, project_dock=docks["project"],
        inspector_dock=docks["inspector"], results_dock=docks["results"])
    layout.register_panel("professional", docks["professional"])
    window.tabifyDockWidget(docks["inspector"], docks["professional"])
    window.show()
    layout.apply_for_width(window.width(), force=True)
    desktop_application.processEvents()
    yield window, layout, docks
    window.close()
    window.deleteLater()


@pytest.mark.parametrize("page,allowed", [
    (WorkspacePage.MEASURE, {"project", "inspector", "results", "professional"}),
    (WorkspacePage.CAPTURE, set()),
    (WorkspacePage.CONTENT_TRIAL, set()),
    (WorkspacePage.DIAMETER_TRIAL, set()),
    (WorkspacePage.DIGITAL_SLIDE_ACQUIRE, {"project", "inspector"}),
])
def test_each_page_blocks_every_unsupported_direct_entry(panel_window, desktop_application, page, allowed):
    window, layout, docks = panel_window
    for key in docks:
        layout.set_panel_visible(key, True)
    old_state = window.saveState(2)
    layout.set_page(page)
    preferences = dict(layout._wide_visibility)
    for key, dock in docks.items():
        assert layout.panel_allowed(key) == (key in allowed)
        if key in allowed:
            continue
        dock.show()
        dock.toggleViewAction().trigger()
        assert not layout.set_panel_visible(key, True)
        desktop_application.processEvents()
        assert dock.isHidden()
        assert not dock.toggleViewAction().isEnabled()
        assert not dock.toggleViewAction().isVisible()
    window.restoreState(old_state, 2)
    layout.enforce_panel_policy()
    desktop_application.processEvents()
    assert all(dock.isHidden() for key, dock in docks.items() if key not in allowed)
    assert layout._wide_visibility == preferences


def test_trial_hiding_preserves_four_preferences_and_tab_group(panel_window, desktop_application):
    window, layout, docks = panel_window
    for key in docks:
        layout.set_panel_visible(key, True)
    layout.set_panel_visible("project", False)
    original = dict(layout._wide_visibility)
    window.resizeDocks([docks["inspector"]], [395], Qt.Orientation.Horizontal)
    desktop_application.processEvents()
    original_width = docks["professional"].width()
    layout.set_page(WorkspacePage.CONTENT_TRIAL)
    layout.persist_profiles()
    saved = json.loads(layout._profiles_path().read_text())
    assert saved["workspaces"]["measure"]["visibility"] == original
    layout.set_page(WorkspacePage.DIAMETER_TRIAL)
    layout.set_page(WorkspacePage.MEASURE)
    desktop_application.processEvents()
    assert layout._wide_visibility == original
    assert {key: not dock.isHidden() for key, dock in docks.items()} == original
    assert docks["professional"] in window.tabifiedDockWidgets(docks["inspector"])
    assert abs(docks["professional"].width() - original_width) <= 2


def test_hidden_restore_and_presentation_cannot_restore_unsupported_panels(panel_window, desktop_application):
    window, layout, docks = panel_window
    for key in docks:
        layout.set_panel_visible(key, True)
    snapshot = window.saveState(2)
    layout.begin_presentation_mode()
    layout.set_page(WorkspacePage.DIAMETER_TRIAL)
    window.restoreState(snapshot, 2)
    layout.end_presentation_mode()
    assert all(dock.isHidden() for dock in docks.values())
    window.hide()
    window.restoreState(snapshot, 2)
    window.resize(1000, 700)
    layout.apply_for_width(1000, force=True)
    window.show()
    layout.apply_pending_layout()
    desktop_application.processEvents()
    assert all(dock.isHidden() for dock in docks.values())
    layout.set_page(WorkspacePage.MEASURE)
    assert layout._wide_visibility["professional"]


def test_compact_professional_close_does_not_reopen_after_trial(panel_window, desktop_application):
    window, layout, docks = panel_window
    window.resize(1000, 700)
    layout.apply_for_width(1000, force=True)
    layout.set_panel_visible("professional", True)
    docks["professional"].close()
    desktop_application.processEvents()
    layout.set_page(WorkspacePage.CAPTURE)
    layout.set_page(WorkspacePage.MEASURE)
    assert docks["professional"].isHidden()


def test_compact_selected_side_panel_returns_after_trial(panel_window, desktop_application):
    window, layout, docks = panel_window
    window.resize(1000, 700)
    layout.apply_for_width(1000, force=True)
    layout.toggle_project()
    assert not docks["project"].isHidden() and docks["inspector"].isHidden()
    layout.set_page(WorkspacePage.CONTENT_TRIAL)
    layout.set_page(WorkspacePage.MEASURE)
    desktop_application.processEvents()
    assert not docks["project"].isHidden() and docks["inspector"].isHidden()
