"""Qt application for image-only rendering, without creating any windows."""
from __future__ import annotations

import sys

from PySide6.QtGui import QGuiApplication

_application = None


def raster_platform_name() -> str:
    # Qt's Windows offscreen plugin uses a FreeType database that looks in
    # Qt's fonts directory, not the Windows system font database. PySide does
    # not ship fonts there. The native plugin loads installed fonts lazily
    # and, without a QWindow/QWidget, still renders entirely into QImages.
    return "windows" if sys.platform == "win32" else "offscreen"


def ensure_raster_application(name: str) -> QGuiApplication:
    global _application
    _application = QGuiApplication.instance() or QGuiApplication(
        [name, "-platform", raster_platform_name()]
    )
    if not isinstance(_application, QGuiApplication):
        raise TypeError("raster rendering requires a Qt GUI application")
    return _application
