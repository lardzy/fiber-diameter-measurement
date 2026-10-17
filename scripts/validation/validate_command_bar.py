"""Capture the real MainWindow command bar without opening any capture device."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scale", type=float, default=1)
    args = parser.parse_args()
    os.environ["QT_SCALE_FACTOR"] = str(args.scale)
    from PySide6.QtWidgets import QApplication, QToolButton
    from fdm import runtime_logging, screenshot_settings, settings
    from fdm.settings import AppSettings
    args.output.mkdir(parents=True, exist_ok=True)
    profile = args.output / "isolated-profile"
    profile.mkdir(exist_ok=True)
    settings.settings_file_path = lambda: profile / "settings.json"
    screenshot_settings.screenshot_settings_file_path = lambda: profile / "screenshot-settings.json"
    runtime_logging.runtime_log_path = lambda: profile / "startup.log"
    from fdm.ui.main_window import MainWindow
    from fdm.ui.theme import apply_application_theme

    app = QApplication.instance() or QApplication([])
    report = {"platform": sys.platform, "qt_platform": app.platformName(),
              "boundary": "real MainWindow, no capture started, logical window sizes", "frames": []}
    with patch("fdm.ui.main_window.AppSettingsIO.load", return_value=AppSettings()), \
         patch("fdm.ui.main_window.AppSettingsIO.save"):
        window = MainWindow()
        window.resize(1366, 768)
        window.show()
        try:
            for mode in ("light", "dark", "navy", "warm"):
                apply_application_theme(app, mode)
                for width in (1366, 1280):
                    window.resize(width, 768)
                    for _ in range(5):
                        app.processEvents()
                    assert not window._capture_manager.is_preview_active()
                    toolbar = window._file_toolbar
                    extension = toolbar.findChild(QToolButton, "qt_toolbar_ext_button")
                    assert extension is not None and not extension.isVisible()
                    stem = f"main-{mode}-{width}@{args.scale:g}x"
                    window.grab().save(str(args.output / (stem + ".png")))
                    toolbar.grab().save(str(args.output / (stem + "-toolbar.png")))
                    report["frames"].append({"theme": mode, "logical_window": [window.width(), window.height()],
                        "dpr": window.devicePixelRatioF(), "toolbar_size": [toolbar.width(), toolbar.height()],
                        "optional_visible": [button.text() for wrapper, button, _ in window._command_bar_optional if wrapper.isVisible()],
                        "screenshot": stem + ".png", "toolbar_screenshot": stem + "-toolbar.png"})
        finally:
            window.close()
            window.deleteLater()
            app.processEvents()
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
