"""Exercise the real Windows workbench against the compiled SDK test DLL.

This verifies window/session ownership and nonfreeze input. It does not certify
sensor images, field order, driver compatibility or physical measurement.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if sys.platform != "win32" or not args.sdk.is_file():
        parser.error("Windows and the compiled test SDK DLL are required")
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ["FDM_MICROVIEW_DLL_DIR"] = str(args.sdk.resolve().parent)
    os.environ["FDM_TEST_MICROVIEW_FIXED_PERIOD"] = "1"
    os.environ["FDM_TEST_MICROVIEW_REUSE_PROCESSED"] = "1"
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication
    from fdm import runtime_logging, screenshot_settings, settings
    profile = args.output / "isolated-profile"
    profile.mkdir(exist_ok=True)
    settings.settings_file_path = lambda: profile / "settings.json"
    screenshot_settings.screenshot_settings_file_path = lambda: profile / "screenshot-settings.json"
    runtime_logging.runtime_log_path = lambda: profile / "startup.log"
    from fdm.services import capture
    backend = capture.MicroviewIsolatedBackend()
    capture.available_capture_backends = lambda: [backend]
    from fdm.models import CalibrationPreset
    from fdm.ui.main_window import MainWindow
    app = QApplication.instance() or QApplication([])
    host = MainWindow()
    errors = []
    manager = host._capture_manager
    manager.errorOccurred.connect(errors.append)
    report = {"boundary": "native test DLL, real MainWindow, no physical Microview", "checks": [], "passed": False,
              "python": sys.version, "platform": platform.platform(),
              "qt_platform": app.platformName(), "sdk_sha256": hashlib.sha256(args.sdk.read_bytes()).hexdigest()}

    def until(predicate, timeout=15):
        deadline = time.perf_counter() + timeout
        while not predicate() and time.perf_counter() < deadline:
            QTest.qWait(5)
        assert predicate(), {"errors": errors, "session_state": manager._session_state}

    def capture_window(name):
        app.processEvents()
        image = host.screen().grabWindow(int(host.winId()))
        assert not image.isNull()
        image.save(str(args.output / (name + ".png")))

    def select_page(index):
        QTest.mouseClick(hub.navigation, Qt.MouseButton.LeftButton,
                         pos=hub.navigation.tabRect(index).center())
        app.processEvents()

    try:
        host._app_settings.calibration_presets = [CalibrationPreset("验证标尺 0.5 μm/px", 2, "um")]
        host.show()
        host.resize(1366, 768)
        report["window_size"] = [host.width(), host.height()]
        report["dpr"] = host.devicePixelRatioF()
        host.activateWindow()
        host.acquisition_workspace_action.trigger()
        hub = host._acquisition_workbench
        until(lambda: manager.latest_preview_frame() is not None)
        assert manager.preview_kind() == "native_embed"
        native_handle = host._microview_preview_host.native_preview_handle()
        assert not host._microview_preview_host.isVisible()
        assert host._preview_canvas.isVisible()
        assert not host._center_stack.testAttribute(Qt.WidgetAttribute.WA_NativeWindow)
        process = backend._preview_process
        capture_window("realtime-qt-pixels-hidden-sdk-host")
        select_page(1)
        page = host._live_experiment_workspace
        until(lambda: page.canvas.frame is not None and page.controller.state == "ready")
        page.ruler_choice.setCurrentIndex(1)
        page.ruler_choice.activated.emit(1)
        until(lambda: page.controller.state == "ready")
        assert page.canvas.frame.metadata["sdk_callback_stage"] == "after_process"
        assert not host._microview_preview_host.isVisible()
        assert not host._center_stack.testAttribute(Qt.WidgetAttribute.WA_NativeWindow)
        page.measure_mode.setCurrentIndex(page.measure_mode.findData("free"))
        page.canvas.setFocus()
        QTest.keyClick(page.canvas, Qt.Key.Key_2)
        assert page.material.currentData() == "wool"
        assert next(iter(page.session.count_adjustments.values())).material_id == "cashmere"
        for point in ((180, 200), (240, 200)):
            token = page.controller.presented
            position = QPoint(round(point[0] * token.scale + token.offset[0]),
                              round(point[1] * token.scale + token.offset[1]))
            QTest.mouseClick(page.canvas, Qt.MouseButton.LeftButton, pos=position)
            if point[0] == 180:
                assert page.controller.draft is not None
                pinned = page.controller.draft[0].frame
                until(lambda: page.controller.presented.frame.sequence > pinned.sequence)
                assert page.controller.draft[0].frame is pinned
        until(lambda: len(page.session.measurements) == 1)
        record = next(iter(page.session.measurements.values()))
        assert record.material_id == "wool" and record.diameter_um == 30
        trial, ruler = page.session, page.controller.calibration
        for _ in range(20):
            host.acquisition_workspace_action.trigger()
            QTest.qWait(5)
            assert host._live_experiment_workspace is page
            assert page.controller.current_records == [record.id]
            assert backend._preview_process is process
            assert host._microview_preview_host.native_preview_handle() == native_handle
        capture_window("nonfreeze-measured-reentry")
        for _ in range(5):
            for index in (0, 2, 1):
                select_page(index)
                assert backend._preview_process is process
                assert host._microview_preview_host.native_preview_handle() == native_handle
                assert not host._microview_preview_host.isVisible()
                if index:
                    page = host._live_experiment_workspace
                    until(lambda: page.canvas.frame is not None and page.controller.state == "ready")
                    assert page.controller.calibration == ruler
                    assert not host.statusBar().isVisible()
        assert page.session is trial and record.id in trial.measurements
        capture_window("nonfreeze-after-page-switches")
        for _ in range(2):
            hub.capture_button.click()
            until(lambda: not manager.is_preview_active() and not manager._stopping)
            hub.capture_button.click()
            until(lambda: page.canvas.frame is not None and page.controller.state == "ready")
            assert page.controller.calibration == ruler and record.id in trial.measurements
        report["checks"] = ["automatic capture entry", "hidden SDK host with Qt pixel display", "AFTER_PROCESS trial pixels", "shared helper across pages",
            "first-click frame pinned while preview updates", "numeric count does not select class",
            "free measurement 30 um", "20 main-entry repeats", "5 realtime/diameter/content roundtrips",
            "2 stop/restart cycles", "history and ruler retained", "single status surface", "stable SDK HWND"]
        report["errors"] = errors
        assert not errors
        report["frame_stats"] = manager.preview_frame_stats()
        report["passed"] = True
    finally:
        if manager.is_preview_active():
            manager.request_stop_preview()
            until(lambda: not manager.is_preview_active() and not manager._stopping)
        host._confirm_close_documents_for_transition = lambda *_: True
        host.close()
        app.processEvents()
        (args.output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
