"""Exercise the real desktop workflow with deterministic, hardware-free frames.

Run each theme / screen / DPI combination in its own process, for example::

    .venv/bin/python scripts/validation/validate_workflow_redesign.py \
        --theme navy --size 1366x768 --scale 1 --output /tmp/fdm-workflow-navy

``--size`` is the physical client raster. Qt's logical size is size / scale.
Also check 1366x690 and 1280x670 clients to reserve native window chrome.
Optional font overrides are diagnostic: the resolved family/metrics are logged,
and a missing Windows font on macOS is not equivalent to native Windows fonts.
All configuration, camera choices, and screen rulers use an isolated profile.
No real capture backend is instantiated or enumerated. The generated report is
UI regression evidence, not Microview SDK / camera throughput acceptance.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
import os
import platform
from pathlib import Path
import sys
import tempfile
import time
import traceback
from types import SimpleNamespace
from unittest.mock import patch


def source_fingerprints():
    """Fingerprint the production shell, rendering and state ownership paths."""
    root = Path(__file__).resolve().parents[2]
    paths = ["src/fdm/ui/" + name for name in (
        "main_window.py", "acquisition_workbench.py", "live_experiment_workspace.py",
        "live_experiment_view.py", "display_preferences.py", "live_display.py",
        "theme.py", "icons.py", "dialogs.py", "canvas.py", "workspace.py",
        "workspace_stack.py", "microview_preview_host.py", "project_save_coordinator.py",
        "project_session_controller.py")]
    paths += ["src/fdm/settings.py", "src/fdm/services/capture.py", "src/fdm/live_experiment/controller.py"]
    paths += [str(path.relative_to(root)) for path in sorted((root / "src/fdm/ui/assets").rglob("*.svg"))]
    return {path: hashlib.sha256((root / path).read_bytes()).hexdigest() for path in paths}


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--output", type=Path)
    result.add_argument("--theme", choices=("dark", "light", "navy", "warm"), default="dark")
    result.add_argument("--size", default="1366x768", help="physical client raster WIDTHxHEIGHT")
    result.add_argument("--scale", type=float, choices=(1, 1.25, 1.5, 2), default=1)
    result.add_argument("--frame-size", default="768x576")
    result.add_argument("--cycles", type=int, default=200)
    result.add_argument("--roundtrips", type=int, default=20)
    result.add_argument("--platform", default="windows" if sys.platform == "win32" else "offscreen")
    result.add_argument("--font-family", help="diagnostic font override; resolved family is recorded")
    result.add_argument("--font-point-size", type=float, help="diagnostic font size override")
    return result


if __name__ == "__main__":
    _args = parser().parse_args()
    os.environ["QT_QPA_PLATFORM"] = _args.platform
    os.environ["QT_SCALE_FACTOR"] = str(_args.scale)
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from PySide6.QtCore import QCoreApplication, QEvent, QObject, QPointF, Qt
from PySide6.QtGui import QColor, QFontDatabase, QFontInfo, QFontMetrics, QImage, QPainter, QPen
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMessageBox, QWidget

from fdm.services.capture import CaptureBackend, CaptureDevice


def dimensions(text):
    width, height = map(int, text.lower().split("x"))
    if min(width, height) <= 0:
        raise ValueError("image dimensions must be positive")
    return width, height


def settle(rounds=8):
    for _ in range(rounds):
        QApplication.processEvents()


def synthetic_frame(size=(768, 576), sequence=0):
    image = QImage(*size, QImage.Format.Format_RGB32)
    image.fill(QColor("#d6d2c4"))
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.scale(size[0] / 768, size[1] / 576)
    for index in range(9):
        painter.setPen(QPen(QColor(70 + index * 4, 65 + index * 4, 50 + index * 4), 8 + index % 4))
        painter.drawLine(QPointF(index * 100 - 130, 0), QPointF(index * 100 + 100, 576))
    painter.fillRect(10, 10, 12, 12, QColor(sequence % 255, 40, 120))
    painter.end()
    return image


class FakeCaptureBackend(CaptureBackend):
    """Real manager contract, explicit frame/error/late-stop delivery, no SDK."""

    backend_key = "workflow-validation"

    def __init__(self, size=(768, 576), kind="native_embed", delayed_stop=False):
        self.size, self.kind = size, kind
        self.device = CaptureDevice("workflow-validation:0", "验证相机（模拟帧）", self.backend_key, 0)
        self.starts = self.stops = self.sequence = 0
        self.available = True
        self.callbacks = []
        self.native_visibility = []
        self.pending_stop = None
        if delayed_stop:
            self.stop_preview_async = self._defer_stop

    def list_devices(self):
        return [self.device] if self.available else []

    def preview_kind(self, device):
        return self.kind

    def preview_resolution(self, device):
        return self.size

    def start_preview(self, device, *, preview_target=None, frame_callback, error_callback):
        self.starts += 1
        self.callbacks.append((frame_callback, error_callback))

    def stop_preview(self):
        self.stops += 1

    def set_native_preview_visible(self, visible):
        self.native_visibility.append(bool(visible))

    def tick(self, *, callback_index=-1, image=None):
        self.sequence += 1
        self.callbacks[callback_index][0](image or synthetic_frame(self.size, self.sequence))
        settle()

    def fail(self, message="验证：设备连接中断", *, callback_index=-1):
        self.callbacks[callback_index][1](message)
        settle()

    def _defer_stop(self, callback):
        self.pending_stop = callback

    def finish_stop(self):
        callback, self.pending_stop = self.pending_stop, None
        assert callback is not None
        callback()
        settle()


class WindowShowObserver(QObject):
    """Catch transient parentless widgets even when they disappear immediately."""

    def __init__(self):
        super().__init__()
        self.events = []

    def eventFilter(self, watched, event):
        if event.type() == QEvent.Type.Show and isinstance(watched, QWidget) and watched.isWindow():
            self.events.append({"class": type(watched).__name__, "name": watched.objectName(),
                "title": watched.windowTitle(), "size": [watched.width(), watched.height()]})
        return False

    def unexpected(self):
        return [entry for entry in self.events if entry["class"] not in {"MainWindow", "QTipLabel"}]


class WorkflowHarness:
    """Own the temporary profile and the actual MainWindow lifetime."""

    def __init__(self, profile, *, theme="dark", size=(1366, 768), backend=None,
                 font_family=None, font_point_size=None):
        self.profile, self.theme, self.size = Path(profile), theme, size
        self.font_family, self.font_point_size = font_family, font_point_size
        self.backend = backend if backend is not None else FakeCaptureBackend()
        self.stack = ExitStack()
        self.observer = WindowShowObserver()
        self.modal_messages = []
        self.slot_errors = []

    def __enter__(self):
        from fdm import runtime_logging, screenshot_settings, settings
        from fdm.models import CalibrationPreset
        from fdm.settings import AppSettings, AppSettingsIO
        from fdm.ui.main_window import MainWindow
        self.profile.mkdir(parents=True, exist_ok=True)
        for module, name, filename in ((settings, "settings_file_path", "settings.json"),
                (screenshot_settings, "screenshot_settings_file_path", "screenshot-settings.json"),
                (runtime_logging, "runtime_log_path", "startup.log")):
            self.stack.enter_context(patch.object(module, name, lambda filename=filename: self.profile / filename))
        self.stack.enter_context(patch("fdm.services.capture.available_capture_backends", return_value=[self.backend]))
        # Entering digital-slide mode probes its configured serial port. Make
        # the real controller report unavailable without opening any port.
        self.stack.enter_context(patch("fdm.services.motion_control.serial", None))
        self.stack.enter_context(patch("fdm.ui.main_window.list_motion_ports", lambda: []))
        self.stack.enter_context(patch("sys.excepthook", lambda *error:
            self.slot_errors.append("".join(traceback.format_exception(*error)))))
        # A regression must fail with evidence rather than block unattended QA.
        def modal(parent, title, text, *_args, **_kwargs):
            self.modal_messages.append({"title": title, "text": text})
            return QMessageBox.StandardButton.Cancel
        for name in ("warning", "critical", "information", "question"):
            self.stack.enter_context(patch.object(QMessageBox, name, modal))
        settings_value = AppSettings(theme_mode=self.theme)
        settings_value.calibration_presets = [CalibrationPreset("验证标尺 2px/μm", 2, "um")]
        AppSettingsIO.save(settings_value)
        self.app = QApplication.instance() or QApplication([])
        self.app.installEventFilter(self.observer)
        self.window = MainWindow()
        if self.font_family is not None or self.font_point_size is not None:
            font = self.window.font()
            if self.font_family is not None:
                font.setFamily(self.font_family)
            if self.font_point_size is not None:
                if self.font_point_size <= 0:
                    raise ValueError("font point size must be positive")
                font.setPointSizeF(self.font_point_size)
            self.window.setFont(font)
        self.stack.enter_context(patch.object(self.window._screenshot_agent_client, "status",
            lambda **_: SimpleNamespace(running=False, error="", result={})))
        self.stack.enter_context(patch.object(self.window._screenshot_agent_client, "send",
            lambda *_args, **_kwargs: SimpleNamespace(result={"running": False, "offline": True, "hotkeys": []})))
        self.window.show()
        self.window.resize(*self.size)
        settle()
        self.window.resize(*self.size)
        settle()
        self.observer.events.clear()
        return self

    @property
    def hub(self):
        return self.window._acquisition_workbench

    def font_report(self):
        font = self.window.font()
        info, metrics = QFontInfo(font), QFontMetrics(font)
        return {"requested_family": self.font_family, "requested_point_size": self.font_point_size,
                "requested_family_installed": (None if self.font_family is None else
                    self.font_family in QFontDatabase.families()),
                "resolved_family": info.family(), "point_size": info.pointSizeF(),
                "pixel_size": info.pixelSize(), "height": metrics.height(),
                "line_spacing": metrics.lineSpacing(),
                "chinese_sample": "样品编号含量试验",
                "chinese_sample_width": metrics.horizontalAdvance("样品编号含量试验"),
                "native_windows": sys.platform == "win32"}

    def enter(self, *, frame=True):
        self.window.acquisition_workspace_action.trigger()
        settle()
        if frame:
            self.backend.tick()
        self.assert_size()
        return self.hub

    def select(self, index, *, frame=True):
        QTest.mouseClick(self.hub.navigation, Qt.MouseButton.LeftButton,
                        pos=self.hub.navigation.tabRect(index).center())
        settle()
        if frame:
            self.backend.tick()
        assert self.hub.navigation.currentIndex() == index
        expected = (self.hub.capture_page if index == 0 else
            self.window._live_experiment_workspaces[("content", "diameter")[index - 1]])
        assert self.hub.pages.currentWidget() is expected
        self.assert_size()
        return self.hub.pages.currentWidget()

    def leave(self):
        self.window.measure_workspace_action.trigger()
        settle()
        self.assert_size()

    def assert_size(self):
        actual = (self.window.width(), self.window.height())
        assert actual == self.size, f"workspace grew beyond requested size: {actual}, requested {self.size}"

    def assert_clean(self):
        assert not self.window.project.live_experiments
        assert not self.window._project_dirty()
        assert not self.modal_messages
        assert not self.slot_errors, "\n".join(self.slot_errors)

    def assert_navigation(self):
        assert self.window._center_stack.currentWidget() is self.hub
        assert self.hub.navigation.isVisible()
        assert self.hub.navigation.visibleRegion().contains(self.hub.navigation.rect())
        assert self.hub.capture_button.isVisible()
        assert not self.window._microview_preview_host.isVisible()
        assert not self.window._center_stack.testAttribute(Qt.WidgetAttribute.WA_NativeWindow)
        assert not self.hub.testAttribute(Qt.WidgetAttribute.WA_NativeWindow)

    def __exit__(self, exc_type, *_):
        try:
            if exc_type is not None:
                self.window.grab().save(str(self.profile.parent / "validation-failure.png"))
            self.window._confirm_close_documents_for_transition = lambda *_: True
            if self.backend.pending_stop is not None:
                self.backend.finish_stop()
            self.window.close()
            self.app.removeEventFilter(self.observer)
            self.window.deleteLater()
            QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
            settle()
        finally:
            self.stack.close()


def dock_state(window):
    return {name: getattr(window, name).isVisible()
            for name in ("_project_dock", "_inspector_dock", "_results_dock")}


def validate_transitions(harness, cycles=200, roundtrips=20):
    from fdm.ui.live_experiment_workspace import LiveExperimentWorkspace
    w, backend = harness.window, harness.backend
    w._project_dock.show()
    w._inspector_dock.show()
    w._results_dock.hide()
    settle()
    docks = dock_state(w)
    harness.enter()
    display = w._preview_display_binding
    source = display._source
    initial_center = display.store.center(*source)
    center_drift = 0.0

    def check_center():
        nonlocal center_drift
        current = display.store.center(*source)
        error = max(abs(a - b) * dimension
                    for a, b, dimension in zip(current, initial_center, source[1]))
        center_drift = max(center_drift, error)
        # An integer scrollbar can round by half a logical pixel, but layout
        # transitions must never accumulate a remembered pan with no gesture.
        assert error <= w.devicePixelRatioF(), (initial_center, current, error)

    first_starts = backend.starts
    native_handle = w._microview_preview_host.native_preview_handle()
    # Include the exact live -> content -> live regression before cycling all pages.
    for index in (1, 0, 2):
        harness.select(index)
        harness.assert_navigation()
        check_center()
    for step in range(cycles):
        harness.select(step % 3)
        harness.assert_navigation()
        check_center()
    assert backend.starts == first_starts == 1
    assert harness.hub.pages.count() == 3
    assert len(w.findChildren(LiveExperimentWorkspace)) == 2
    assert not harness.observer.unexpected(), harness.observer.unexpected()
    for _ in range(roundtrips):
        harness.leave()
        assert not w._capture_manager.is_preview_active()
        assert dock_state(w) == docks
        harness.enter()
        harness.assert_navigation()
        check_center()
    assert backend.starts == 1 + roundtrips
    assert w._microview_preview_host.native_preview_handle() == native_handle
    assert not any(backend.native_visibility), "SDK display ownership must stay hidden"
    harness.leave()
    assert dock_state(w) == docks
    harness.assert_clean()
    return {"page_switches": cycles + 3, "workspace_roundtrips": roundtrips,
            "backend_starts": backend.starts, "backend_stops": backend.stops,
            "stable_native_handle": bool(native_handle), "native_display_requested": any(backend.native_visibility),
            "dock_state_restored": docks, "unexpected_windows": harness.observer.unexpected(),
            "max_center_drift_source_pixels": center_drift,
            "modal_messages": harness.modal_messages, "project_dirty": w._project_dirty()}


def validate_display_modes(harness):
    """Use all real page bindings; no project/calibration mutation is required."""
    from fdm.ui.display_preferences import AcquisitionDisplayStore
    from fdm.ui.view_transform import CanvasZoomMode
    harness.enter()
    w = harness.window
    requested = (("pixel", None), ("fit", None), ("percent", 137), ("magnification", 300))
    modes = []
    for mode, value in requested:
        harness.select(0)
        w._preview_display_binding.set_mode(mode, value)
        settle()
        for index in (1, 2, 0):
            page = harness.select(index)
            binding = w._preview_display_binding if index == 0 else page.display_binding
            assert binding.store.preferences.mode == mode
            if index:
                # A missing calibration falls back without losing the chosen mode.
                assert page.canvas.display_mode == ("pixel" if mode == "magnification" else mode)
                assert page.display_choice.currentData() == mode
            elif mode == "fit":
                assert w._preview_canvas.zoom_mode() is CanvasZoomMode.FIT
            else:
                expected = 1.37 if mode == "percent" else 1
                assert abs(w._preview_canvas.view_zoom() * w.devicePixelRatioF() - expected) < 1e-6
        harness.leave()
        harness.enter()
        harness.select(0)
        assert w._preview_display_binding.store.preferences.mode == mode
        restored = AcquisitionDisplayStore(w._preview_display_binding.store.path)
        assert restored.preferences == w._preview_display_binding.store.preferences
        modes.append({"mode": mode, "value": value, "reloaded": True})
    preferences = w._preview_display_binding.store.preferences
    assert preferences.percent == 137 and preferences.magnification == 300
    harness.assert_clean()
    return modes


def capture_screenshots(harness, output):
    w = harness.window
    screenshots = []
    def save(name, widget=None, *, live_tick=True):
        if live_tick and w._capture_manager.is_preview_active():
            harness.backend.tick()
        # Status/readiness are refreshed by an 80 ms UI timer. Capture a
        # settled presentation, not the old status between frame and timer.
        QTest.qWait(100)
        target = widget or w
        if target is w:
            harness.assert_size()
        path = output / f"{name}.png"
        pixmap = target.grab()
        assert pixmap.save(str(path))
        screenshots.append({"name": name, "path": str(path),
            "logical_size": [target.width(), target.height()],
            "raster_size": [pixmap.width(), pixmap.height()], "dpr": target.devicePixelRatioF()})
    harness.leave()
    save("measurement")
    harness.enter()
    harness.select(0)
    w._preview_display_binding.set_mode("fit")
    settle()
    save("realtime")
    for index, mode in ((1, "content"), (2, "diameter")):
        page = harness.select(index)
        assert page.sample_number.isVisible() and page.ruler_choice.isVisible()
        viewport = page.image_scroll.geometry()
        for tab, title in enumerate(("operation", "records", "report")):
            page.workflow_tabs.setCurrentIndex(tab)
            settle()
            assert page.image_scroll.geometry() == viewport, "records/report must not cover the image"
            save(f"{mode}-{title}")
        page.workflow_tabs.setCurrentIndex(0)
        page.advanced_toggle.setChecked(True)
        settle()
        for tab in range(page.settings_tabs.count()):
            page.settings_tabs.setCurrentIndex(tab)
            settle()
            scroll = page.settings_tabs.currentWidget()
            assert not scroll.horizontalScrollBar().isVisible()
            save(f"{mode}-advanced-{tab}", page.settings_dialog)
        page.settings_dialog.close()
    harness.leave()
    w.digital_slide_action.trigger()
    settle()
    harness.backend.tick()
    assert w._digital_slide_mode
    assert not w._slide_motion.enabled and w._slide_motion._serial is None
    save("digital-slide")
    harness.leave()
    assert not w._digital_slide_mode

    # Enter settings through the real QAction; replace only the modal event
    # loop so all real widgets can be inspected and the script can continue.
    from fdm.ui.dialogs import SettingsDialog
    def inspect_settings(dialog):
        dialog.show()
        settle()
        for index in range(dialog._settings_pages.count()):
            dialog._settings_navigation.setCurrentRow(index)
            settle()
            save(f"settings-{index}", dialog)
        dialog.reject()
        return dialog.DialogCode.Rejected
    with patch.object(SettingsDialog, "exec", inspect_settings):
        w.settings_action.trigger()
    settle()

    harness.backend.available = False
    harness.enter(frame=False)
    assert not w._capture_manager.is_preview_active()
    assert harness.hub.feedback.isVisible()
    save("capture-no-device")
    harness.backend.available = True
    harness.hub.capture_button.click()
    settle()
    page = harness.select(1, frame=False)
    page.source._last_frame_at -= 4
    page.source._check_arrivals()
    settle()
    assert page.controller.state == "unavailable"
    save("trial-no-frame", live_tick=False)
    harness.backend.tick()
    save("trial-frame-restored")
    harness.backend.fail()
    assert w._capture_manager.current_capture_state() == "faulted"
    assert harness.hub.feedback.isVisible()
    save("trial-capture-error")
    harness.select(0, frame=False)
    save("capture-error")
    assert not harness.modal_messages
    return screenshots


def main(args):
    output = (args.output or Path(tempfile.mkdtemp(prefix="fdm-workflow-validation-"))).resolve()
    output.mkdir(parents=True, exist_ok=True)
    profile = Path(tempfile.mkdtemp(prefix="isolated-profile-", dir=output))
    physical = dimensions(args.size)
    logical = tuple(round(value / args.scale) for value in physical)
    report = {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
        "source_fingerprints": source_fingerprints(), "theme": args.theme, "requested_raster": physical,
        "runtime": {"python": sys.version.split()[0], "platform": platform.platform()},
        "requested_logical": logical, "requested_scale": args.scale,
        "frame_size": dimensions(args.frame_size), "profile": str(profile),
        "platform": args.platform, "real_hardware": False,
        "limitations": ["Synthetic backend does not validate SDK HWND clipping, throughput or camera recovery.",
            "Mixed physical monitors require the documented native Windows acceptance pass."],
        "passed": False}
    began = time.perf_counter()
    harness = None
    try:
        backend = FakeCaptureBackend(dimensions(args.frame_size))
        with WorkflowHarness(profile, theme=args.theme, size=logical, backend=backend,
                             font_family=args.font_family, font_point_size=args.font_point_size) as harness:
            report["font"] = harness.font_report()
            report["actual_logical"] = [harness.window.width(), harness.window.height()]
            report["actual_dpr"] = harness.window.devicePixelRatioF()
            assert report["actual_logical"] == list(logical), "window exceeds requested client size"
            report["transitions"] = validate_transitions(harness, args.cycles, args.roundtrips)
            report["display_modes"] = validate_display_modes(harness)
            report["screenshots"] = capture_screenshots(harness, output)
            assert not harness.slot_errors, "\n".join(harness.slot_errors)
            report["passed"] = True
    except Exception:
        report["error"] = traceback.format_exc()
        if harness is not None:
            report["show_events"] = harness.observer.events
            report["modal_messages"] = harness.modal_messages
            report["slot_errors"] = harness.slot_errors
    finally:
        report["source_changed_during_run"] = source_fingerprints() != report["source_fingerprints"]
        report["duration_seconds"] = round(time.perf_counter() - began, 3)
        (output / "workflow.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "report": str(output / "workflow.json"),
                      "duration_seconds": report["duration_seconds"], "error": report.get("error")}, ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main(_args))
