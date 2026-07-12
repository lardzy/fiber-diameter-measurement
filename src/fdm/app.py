from __future__ import annotations

import json
from pathlib import Path
import sys
import traceback


APP_NAME = "Fiber Diameter Measurement"


def _log_directory() -> Path:
    from fdm.runtime_logging import runtime_log_path

    return runtime_log_path().parent


def _write_startup_log(title: str, details: str) -> Path | None:
    try:
        from fdm.runtime_logging import append_runtime_log, runtime_log_path

        log_path = runtime_log_path()
        append_runtime_log(title, details)
        return log_path
    except (ImportError, OSError):
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
        try:
            from fdm.runtime_logging import flush_runtime_metrics

            flush_runtime_metrics()
        except Exception:
            pass
        _write_startup_log("Unhandled exception", details)
        default_hook(exc_type, exc_value, exc_traceback)

    sys.excepthook = handle_exception


def _run_release_self_check(*, json_output: bool) -> int:
    from fdm.release_manifest import format_self_check_report, run_release_self_check

    try:
        report = run_release_self_check()
    except Exception as exc:  # noqa: BLE001
        report = {
            "ok": False,
            "errors": [f"self-check failed unexpectedly: {exc}"],
            "warnings": [],
        }
    output = (
        json.dumps(report, ensure_ascii=False, sort_keys=True, allow_nan=False)
        if json_output
        else format_self_check_report(report)
    )
    _write_cli_output(output + "\n")
    return 0 if report.get("ok") else 1


def _write_cli_output(payload: str) -> None:
    if sys.stdout is not None:
        sys.stdout.write(payload)
        sys.stdout.flush()
        return
    if not sys.platform.startswith("win"):
        return
    # PyInstaller's windowed bootloader may set sys.stdout to None. Reuse an
    # inherited/parent console handle so `--self-check --json` remains usable.
    try:
        import ctypes
        import msvcrt
        import os

        kernel32 = ctypes.windll.kernel32
        get_std_handle = kernel32.GetStdHandle
        get_std_handle.argtypes = [ctypes.c_ulong]
        get_std_handle.restype = ctypes.c_void_p
        invalid_handle = ctypes.c_void_p(-1).value
        stdout_handle = get_std_handle(ctypes.c_ulong(-11).value)  # STD_OUTPUT_HANDLE
        if stdout_handle in (None, 0, invalid_handle):
            kernel32.AttachConsole(ctypes.c_ulong(-1).value)  # ATTACH_PARENT_PROCESS
            stdout_handle = get_std_handle(ctypes.c_ulong(-11).value)
        if stdout_handle in (None, 0, invalid_handle):
            return
        descriptor = msvcrt.open_osfhandle(int(stdout_handle), os.O_WRONLY)
        with os.fdopen(descriptor, "w", encoding="utf-8", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
    except Exception:
        return


def main(argv: list[str] | None = None) -> int:
    args = list(argv) if argv is not None else sys.argv
    if "--self-check" in args[1:]:
        return _run_release_self_check(json_output="--json" in args[1:])
    if len(args) > 1 and args[1] == "--microview-helper":
        try:
            from fdm.microview_helper import main as microview_helper_main

            return microview_helper_main(args[2:])
        except Exception as exc:  # noqa: BLE001
            return _report_startup_exception("Microview helper 启动失败", exc)

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
        from PySide6.QtCore import QTimer

        from fdm.application_launch import (
            ApplicationOpenRequest,
            ApplicationOpenRequestError,
            SingleInstanceCoordinator,
            parse_application_arguments,
        )

        try:
            qt_args, initial_open_request = parse_application_arguments(args)
        except ApplicationOpenRequestError as exc:
            _show_fallback_error(APP_NAME, str(exc))
            return 2

        app = QApplication(qt_args)
        app.setApplicationName(APP_NAME)
        app.setOrganizationName("Codex")
        instance_coordinator: SingleInstanceCoordinator | None = None
        pending_instance_requests: list[ApplicationOpenRequest] = []

        def buffer_instance_request(request: ApplicationOpenRequest) -> None:
            pending_instance_requests.append(request)

        if sys.platform.startswith("win"):
            instance_coordinator = SingleInstanceCoordinator.for_current_application(app)
            instance_result = instance_coordinator.start_or_forward(initial_open_request)
            if instance_result.forwarded:
                return 0
            if not instance_result.primary:
                _show_fallback_error(APP_NAME, instance_result.error or "无法启动软件实例。")
                return 1
            instance_coordinator.requestReceived.connect(buffer_instance_request)
            instance_coordinator.protocolError.connect(
                lambda message: _write_startup_log("Single-instance protocol error", message)
            )
        from fdm.settings import AppSettingsIO
        from fdm.ui.icons import application_icon
        from fdm.ui.main_window import MainWindow
        from fdm.ui.theme import apply_application_theme

        apply_application_theme(app, AppSettingsIO.load().theme_mode)
        app.setWindowIcon(application_icon())
        window = MainWindow()
        if instance_coordinator is not None:
            instance_coordinator.requestReceived.disconnect(buffer_instance_request)
            instance_coordinator.requestReceived.connect(window.enqueue_application_open_request)
        window.show()
        startup_requests: list[ApplicationOpenRequest] = []
        if initial_open_request.paths:
            startup_requests.append(initial_open_request)
        startup_requests.extend(pending_instance_requests)
        if startup_requests:

            def enqueue_startup_requests() -> None:
                for request in startup_requests:
                    window.enqueue_application_open_request(request)

            QTimer.singleShot(0, enqueue_startup_requests)
        exit_code = app.exec()
        if instance_coordinator is not None:
            instance_coordinator.close()
        from fdm.runtime_logging import flush_runtime_metrics

        flush_runtime_metrics()
        return exit_code
    except Exception as exc:  # noqa: BLE001
        return _report_startup_exception("应用启动失败", exc)


if __name__ == "__main__":
    raise SystemExit(main())
