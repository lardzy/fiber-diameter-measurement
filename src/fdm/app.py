from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    try:
        from PySide6.QtWidgets import QApplication, QMessageBox
    except ImportError as exc:
        print(
            "PySide6 is not installed. Please create a virtual environment and install the "
            "project dependencies before launching the desktop application.",
            file=sys.stderr,
        )
        print(f"Import error: {exc}", file=sys.stderr)
        return 1

    from fdm.ui.main_window import MainWindow

    app = QApplication(argv or sys.argv)
    app.setApplicationName("Fiber Diameter Measurement")
    app.setOrganizationName("Codex")
    window = MainWindow()
    window.show()
    return app.exec()
