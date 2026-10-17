from __future__ import annotations

import gc
from unittest.mock import patch

import pytest
from PySide6.QtCore import QCoreApplication, QEvent, Qt
from PySide6.QtWidgets import QApplication, QToolButton

from fdm.settings import AppSettings
from fdm.ui.main_window import MainWindow
from fdm.ui.theme import apply_application_theme


def settle():
    for _ in range(4):
        QApplication.processEvents()


@pytest.fixture
def window(desktop_application):
    with patch("fdm.ui.main_window.AppSettingsIO.load", return_value=AppSettings()), \
         patch("fdm.ui.main_window.AppSettingsIO.save"):
        win = MainWindow()
        win.show()
        settle()
        yield win
        win._confirm_close_documents_for_transition = lambda *_: True
        win.close()
        win.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        settle()


@pytest.mark.parametrize("width", (1093, 1280, 1366, 1920))
@pytest.mark.parametrize("mode", ("light", "dark", "navy", "warm"))
def test_main_commands_retain_labels_and_fit_without_extension(window, width, mode):
    apply_application_theme(QApplication.instance(), mode)
    window.resize(width, 768)
    settle()
    toolbar = window._file_toolbar
    assert toolbar.toolButtonStyle() == Qt.ToolButtonStyle.ToolButtonTextBesideIcon
    extension = toolbar.findChild(QToolButton, "qt_toolbar_ext_button")
    assert extension is not None and not extension.isVisible()
    fixed = [*window._workspace_buttons, window._settings_command_button,
             window._panel_command_button, window._more_command_button,
             window.findChild(QToolButton, "openCommandButton"), window._export_command_button,
             toolbar.widgetForAction(window.save_project_action),
             toolbar.widgetForAction(window.command_search_action)]
    for button in fixed:
        assert button is not None and button.isVisible(), button
        assert button.toolButtonStyle() == Qt.ToolButtonStyle.ToolButtonTextBesideIcon, button.text()
        assert button.text() and not button.icon().isNull(), button.text()
        assert toolbar.rect().contains(button.mapTo(toolbar, button.rect().bottomRight())), button.text()
    assert window._settings_command_button.objectName() == "settingsCommandButton"


def test_export_and_panel_commands_have_named_menu_homes(window):
    assert window.export_template_action in window._export_command_button.menu().actions()
    assert window.export_overlay_action in window._export_command_button.menu().actions()
    panel_actions = window._panel_command_button.menu().actions()
    assert window._panel_command_button.text() == "面板"
    for action in (window.toggle_project_panel_action, window.toggle_inspector_panel_action, window.toggle_results_panel_action):
        assert action in panel_actions
        assert window._file_toolbar.widgetForAction(action) is None
    assert window.switch_capture_device_action in window._command_bar_more_menu.actions()


def test_optional_commands_move_without_hiding_shared_actions(window):
    window.resize(1920, 768)
    settle()
    assert all(wrapper.isVisible() for wrapper, _, _ in window._command_bar_optional)
    # Use a measured width just below the full command bar to force the
    # priority path independently of platform-specific font metrics.
    full_width = window._file_toolbar.sizeHint().width()
    window.resize(full_width - 20, 768)
    settle()
    wrappers = window._command_bar_optional
    assert not wrappers[-1][0].isVisible()  # Capture is the first optional shortcut.
    for wrapper, button, action in wrappers:
        assert action.isVisible()
        assert action in window._command_bar_more_menu.actions()
        assert button.defaultAction() is action
        if not wrapper.isVisible():
            assert not button.isVisible()
    window.resize(1920, 768)
    settle()
    assert all(wrapper.isVisible() for wrapper, _, _ in wrappers)


def test_optional_menu_action_keeps_enabled_state_and_callback(window):
    action = window.capture_frame_action
    action.triggered.disconnect()
    calls = []
    action.triggered.connect(lambda *_: calls.append(True))
    window._update_main_command_bar_density(900)
    action.setEnabled(False)
    assert not window._command_bar_more_menu.actions()[2].isEnabled()
    action.setEnabled(True)
    window._command_bar_more_menu.actions()[2].trigger()
    assert calls == [True]


def test_window_theme_icon_lifecycle_survives_repeated_deletion_and_gc(desktop_application):
    # Exercise QIconEngine callbacks during live painting and actual QObject
    # teardown; GC remains enabled throughout this regression.
    assert gc.isenabled()
    with patch("fdm.ui.main_window.AppSettingsIO.load", return_value=AppSettings()), \
         patch("fdm.ui.main_window.AppSettingsIO.save"):
        for index in range(20):
            win = MainWindow()
            win.resize(1366, 768)
            win.show()
            for mode in ("light", "dark", "navy", "warm"):
                apply_application_theme(desktop_application, mode)
                settle()
                assert not win._file_toolbar.grab().isNull()
            win.close()
            assert not win.isVisible()
            win.deleteLater()
            QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
            settle()
            del win
            gc.collect()
