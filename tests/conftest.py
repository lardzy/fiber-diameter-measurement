"""Keep desktop tests independent of the user's real configuration files."""
from __future__ import annotations

import pytest


@pytest.fixture(scope="session", autouse=True)
def desktop_application():
    # A function-scoped QApplication may be collected while window wrappers
    # still exist. Keep one application alive for the whole desktop test run.
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture(autouse=True)
def dispose_closed_main_windows(desktop_application):
    """Finish Qt deletion for windows whose test already accepted a close.

    QWidget.close() normally only hides a window. Signal reference cycles can
    otherwise retain thousands of closed widgets across the suite, including
    their palette callbacks. Never force-delete a window that refused to close
    (for example, because a background task has not stopped).
    """
    from PySide6.QtCore import QCoreApplication, QEvent
    from shiboken6 import isValid

    from fdm.ui.main_window import MainWindow

    yield
    for window in desktop_application.topLevelWidgets():
        # MainWindow removes its application event filter only at the end of
        # an accepted closeEvent. An ignored close keeps this flag set.
        if (
            isinstance(window, MainWindow)
            and isValid(window)
            and not window.isVisible()
            and getattr(window, "_application_key_filter_installed", True) is False
        ):
            window.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


@pytest.fixture(autouse=True)
def isolated_desktop_profile(tmp_path, monkeypatch):
    from fdm import screenshot_settings, settings

    monkeypatch.setattr(settings, "settings_file_path", lambda: tmp_path / "settings.json")
    monkeypatch.setattr(screenshot_settings, "screenshot_settings_file_path", lambda: tmp_path / "screenshot-settings.json")
