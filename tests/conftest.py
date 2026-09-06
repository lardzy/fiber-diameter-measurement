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
def isolated_desktop_profile(tmp_path, monkeypatch):
    from fdm import screenshot_settings, settings

    monkeypatch.setattr(settings, "settings_file_path", lambda: tmp_path / "settings.json")
    monkeypatch.setattr(screenshot_settings, "screenshot_settings_file_path", lambda: tmp_path / "screenshot-settings.json")
