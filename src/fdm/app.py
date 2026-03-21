from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys
import traceback


APP_NAME = "Fiber Diameter Measurement"


def _log_directory() -> Path:
    local_app_data = Path.home()
    if sys.platform.startswith("win"):
        app_data = Path(
            (
                __import__("os").environ.get("LOCALAPPDATA")
                or __import__("os").environ.get("APPDATA")
                or str(Path.home())
            )
        )
        local_app_data = app_data
    return local_app_data / "FiberDiameterMeasurement" / "logs"


def _write_startup_log(title: str, details: str) -> Path | None:
    try:
        log_dir = _log_directory()
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "startup.log"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"[{timestamp}] {title}\n")
            handle.write(details.rstrip())
            handle.write("\n\n")
        return log_path
    except OSError:
        return None


def _show_fallback_error(title: str, message: str) -> None:
    if sys.platform.startswith("win"):
        try:
            import ctypes

            ctypes.windll.user32.MessageBoxW(None, message, title, 0x10)
            return
        except Exception:
            pass
    print(f"{title}: {message}", file=sys.stderr)


def _report_startup_exception(title: str, exc: BaseException) -> int:
    details = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    log_path = _write_startup_log(title, details)
    log_hint = f"\n\n详细日志: {log_path}" if log_path else ""
    message = f"{title}\n\n{exc}{log_hint}"
    try:
        from PySide6.QtWidgets import QMessageBox

        QMessageBox.critical(None, APP_NAME, message)
    except Exception:
        _show_fallback_error(APP_NAME, message)
    print(details, file=sys.stderr)
    return 1


def _install_global_exception_hook() -> None:
    default_hook = sys.excepthook

    def handle_exception(exc_type, exc_value, exc_traceback) -> None:
        details = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        _write_startup_log("Unhandled exception", details)
        default_hook(exc_type, exc_value, exc_traceback)

    sys.excepthook = handle_exception


def main(argv: list[str] | None = None) -> int:
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError as exc:
        details = (
            "PySide6 is not installed. Please create a virtual environment and install the "
            "project dependencies before launching the desktop application.\n\n"
            f"Import error: {exc}"
        )
        log_path = _write_startup_log("PySide6 import failed", details)
        if log_path is not None:
            details += f"\n\n详细日志: {log_path}"
        _show_fallback_error(APP_NAME, details)
        print(details, file=sys.stderr)
        return 1

    _install_global_exception_hook()

    try:
        from fdm.ui.main_window import MainWindow

        app = QApplication(argv or sys.argv)
        app.setApplicationName(APP_NAME)
        app.setOrganizationName("Codex")
        window = MainWindow()
        window.show()
        return app.exec()
    except Exception as exc:  # noqa: BLE001
        return _report_startup_exception("应用启动失败", exc)
