import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from fdm.ui import qt_raster_runtime as runtime


@pytest.mark.parametrize("platform, expected", [("win32", "windows"), ("darwin", "offscreen"), ("linux", "offscreen")])
def test_image_only_application_selects_a_platform_with_system_fonts(monkeypatch, platform, expected):
    class Application:
        existing = None

        @classmethod
        def instance(cls):
            return cls.existing

        def __init__(self, arguments):
            self.arguments = arguments

    monkeypatch.setattr(runtime, "sys", SimpleNamespace(platform=platform))
    monkeypatch.setattr(runtime, "QGuiApplication", Application)
    monkeypatch.setattr(runtime, "_application", None)
    app = runtime.ensure_raster_application("probe")
    assert app.arguments == ["probe", "-platform", expected]
    Application.existing = app
    assert runtime.ensure_raster_application("reuse") is app


def test_image_only_application_rejects_an_existing_core_application(monkeypatch):
    class Application:
        @classmethod
        def instance(cls):
            return object()

    monkeypatch.setattr(runtime, "QGuiApplication", Application)
    monkeypatch.setattr(runtime, "_application", None)
    with pytest.raises(TypeError, match="requires a Qt GUI application"):
        runtime.ensure_raster_application("probe")


@pytest.mark.parametrize("fontless", [False, True])
def test_fresh_process_renders_text_or_reports_missing_fonts(fontless):
    # Start without pytest's pre-existing QApplication. The minimal platform
    # reproduces the empty font database of Windows' old offscreen startup.
    code = """
import json
import sys
from PySide6.QtGui import QGuiApplication
from fdm.ui.watermark_self_check import run_watermark_self_check
if sys.argv[1] == 'fontless':
    application = QGuiApplication(['fontless-probe', '-platform', 'minimal'])
report = run_watermark_self_check()
report['top_level_windows'] = len(QGuiApplication.topLevelWindows())
print(json.dumps(report))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code, "fontless" if fontless else "normal"],
        cwd=Path(__file__).resolve().parents[1],
        # The initializer must override an unsuitable inherited platform.
        env={**os.environ, "QT_QPA_PLATFORM": "minimal"},
        capture_output=True, text=True, encoding="utf-8", timeout=30, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["top_level_windows"] == 0
    if fontless:
        assert not report["ok"]
        assert report["runtime"]["font_family_count"] == 0
        assert set(report["failed_cases"]) == {"font_database"} | {
            f"{name}@{dpr}"
            for name in ("text", "datetime_text", "datetime_logo")
            for dpr in ("1", "1.5", "2")
        }
        assert report["cases"]["codec_png"] and report["cases"]["tile@2"]
        assert report["details"]["datetime_logo@2"]["visible"] is False
        assert report["details"]["datetime_logo@2"]["repeat_equal"] is True
    else:
        assert report["ok"], report
        assert report["runtime"]["qt_platform"] == runtime.raster_platform_name()
        assert report["runtime"]["font_family_count"] > 0
        assert report["failed_cases"] == []
