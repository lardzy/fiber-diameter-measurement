"""Exercise screenshot hotkeys on an interactive Windows desktop.

Run from the checkout with its Windows Python environment::

    python scripts/validate_screenshot_hotkeys_windows.py --output build/hotkeys.json

This sends synthetic Ctrl+Alt+Shift+F9/F10 followed by PrintScreen checks,
only while the probe window owns the foreground. PrintScreen interception by
Windows or another application is reported as an unvalidated limitation.
It creates independent settings, never contacts the user's
resident companion, and never modifies autostart. Actual desktop PNGs are saved
to a temporary directory and removed after verification unless --keep-captures
is specified. Timings start at application receipt, not physical key press.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import ctypes
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import math
from pathlib import Path
import platform
import shutil
import statistics
import sys
import tempfile
import threading
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
MODIFIERS = 0x01 | 0x02 | 0x04
VK_F9 = 0x78
VK_F10 = 0x79
VK_PRINT = 0x2C
CHORD_F9 = "Ctrl+Alt+Shift+F9"
CHORD_F10 = "Ctrl+Alt+Shift+F10"


def summarize_ms(values: list[float]) -> dict[str, float | int | None]:
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "p50_ms": round(statistics.median(ordered), 3) if ordered else None,
        "p95_ms": round(ordered[max(0, math.ceil(len(ordered) * .95) - 1)], 3) if ordered else None,
        "max_ms": round(ordered[-1], 3) if ordered else None,
    }


class KeyboardInput(ctypes.Structure):
    _fields_ = [
        ("wVk", ctypes.c_uint16), ("wScan", ctypes.c_uint16),
        ("dwFlags", ctypes.c_uint32), ("time", ctypes.c_uint32),
        ("dwExtraInfo", ctypes.c_size_t),
    ]


class MouseInput(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_int32), ("dy", ctypes.c_int32),
        ("mouseData", ctypes.c_uint32), ("dwFlags", ctypes.c_uint32),
        ("time", ctypes.c_uint32), ("dwExtraInfo", ctypes.c_size_t),
    ]


class HardwareInput(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_uint32), ("wParamL", ctypes.c_uint16), ("wParamH", ctypes.c_uint16)]


class InputUnion(ctypes.Union):
    _fields_ = [("ki", KeyboardInput), ("mi", MouseInput), ("hi", HardwareInput)]


class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_uint32), ("data", InputUnion)]


class WindowsInput:
    def __init__(self) -> None:
        self.user32 = ctypes.WinDLL("user32", use_last_error=True)
        self.user32.SendInput.argtypes = [ctypes.c_uint32, ctypes.POINTER(Input), ctypes.c_int]
        self.user32.SendInput.restype = ctypes.c_uint32
        self.user32.GetForegroundWindow.argtypes = []
        self.user32.GetForegroundWindow.restype = ctypes.c_void_p
        self.user32.SetForegroundWindow.argtypes = [ctypes.c_void_p]
        self.user32.SetForegroundWindow.restype = ctypes.c_int
        self.user32.GetAsyncKeyState.argtypes = [ctypes.c_int]
        self.user32.GetAsyncKeyState.restype = ctypes.c_int16
        self.user32.GetClassNameW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_int]
        self.user32.GetClassNameW.restype = ctypes.c_int
        self.original_foreground = self.foreground()
        self.held: set[int] = set()

    def foreground(self) -> int:
        return int(self.user32.GetForegroundWindow() or 0)

    def foreground_snapshot(self) -> dict[str, object]:
        hwnd = self.foreground()
        name = ctypes.create_unicode_buffer(256)
        length = self.user32.GetClassNameW(ctypes.c_void_p(hwnd), name, len(name)) if hwnd else 0
        return {"hwnd": hwnd, "class_name": name.value if length else None}

    def assert_keyboard_idle(self) -> None:
        keys = (0x10, 0x11, 0x12, 0x5B, 0x5C, VK_F9, VK_F10, VK_PRINT)
        pressed = [key for key in keys if self.user32.GetAsyncKeyState(key) & 0x8000]
        if pressed:
            raise RuntimeError(f"Keyboard is not idle; refusing synthetic input: {pressed}")

    def send(self, events: list[tuple[int, bool]]) -> None:
        items = (Input * len(events))()
        for index, (key, released) in enumerate(events):
            items[index].type = 1  # INPUT_KEYBOARD
            items[index].data.ki = KeyboardInput(key, 0, 2 if released else 0, 0, 0)
        sent = int(self.user32.SendInput(len(items), items, ctypes.sizeof(Input)))
        for key, released in events[:sent]:
            if released:
                self.held.discard(key)
            else:
                self.held.add(key)
        if sent != len(events):
            raise RuntimeError(f"SendInput inserted {sent}/{len(events)} events; error {ctypes.get_last_error()}")

    def release_owned_keys(self) -> None:
        # Only release keys this probe inserted. Never blanket-release user keys.
        if self.held:
            self.send([(key, True) for key in sorted(self.held, reverse=True)])

    def restore_foreground(self) -> None:
        if self.original_foreground:
            self.user32.SetForegroundWindow(ctypes.c_void_p(self.original_foreground))


def run_probe(args: argparse.Namespace, report: dict[str, object]) -> None:
    from dataclasses import replace

    from PySide6.QtCore import QEvent, QEventLoop, QObject, Qt, QTimer
    from PySide6.QtGui import QImage
    from PySide6.QtWidgets import QApplication, QLabel, QPushButton, QScrollArea, QVBoxLayout, QWidget

    from fdm.platform.windows_global_hotkey import HotkeyBinding as NativeBinding
    from fdm.platform.windows_global_hotkey import WindowsGlobalHotkeyManager
    from fdm.screenshot_agent import ScreenshotAgent
    from fdm.screenshot_settings import HotkeyBinding, ScreenshotSettings, ScreenshotSettingsIO, default_hotkeys
    from fdm.services.screenshot_capture import CaptureCoordinator, CaptureMode, CaptureRect, CaptureRequest
    from fdm.services.screenshot_output import ScreenshotOutputService
    from fdm.ui.screenshot_hotkey_edit import ScreenshotHotkeyEdit
    from fdm.ui.screenshot_settings_page import ScreenshotSettingsPage

    app = QApplication.instance() or QApplication([])
    app.setQuitOnLastWindowClosed(False)
    if app.platformName().casefold() != "windows":
        raise RuntimeError(f"Real Windows Qt platform required, got {app.platformName()!r}")
    keyboard = WindowsInput()
    report["original_foreground"] = keyboard.foreground_snapshot()
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Control Panel\Keyboard", 0, winreg.KEY_READ) as key:
            value, kind = winreg.QueryValueEx(key, "PrintScreenKeyForSnippingEnabled")
        report["printscreen_snipping_preference"] = {"present": True, "value": value, "registry_type": kind}
    except FileNotFoundError:
        report["printscreen_snipping_preference"] = {"present": False, "value": None, "note": "Absent value; no Windows default is inferred"}
    except OSError as exc:
        report["printscreen_snipping_preference"] = {"present": None, "error": str(exc)}
    temporary_root = Path(tempfile.mkdtemp(prefix="fdm-hotkey-probe-", dir=args.output.parent))
    report["capture_directory"] = str(temporary_root)
    report["capture_files_retained"] = bool(args.keep_captures)
    report["qt_platform"] = app.platformName()
    report["input_struct_bytes"] = ctypes.sizeof(Input)
    all_traces: list[dict[str, object]] = []

    def pump(duration: float = 0.0) -> None:
        end = time.perf_counter() + duration
        while True:
            app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 5)
            if time.perf_counter() >= end:
                break
            time.sleep(.002)

    def wait_until(predicate, message: str, timeout: float = 10.0) -> None:
        end = time.perf_counter() + timeout
        while not predicate():
            if time.perf_counter() >= end:
                raise AssertionError(message)
            pump(.005)

    class NoAutostart:
        def set_enabled(self, enabled: bool) -> None:
            if enabled:
                raise AssertionError("Probe must never enable autostart")

    class ProbeOutput(ScreenshotOutputService):
        def __init__(self) -> None:
            super().__init__()
            self.delay = 0.0
            self.started = threading.Event()
            self.finished = threading.Event()
            self.saved: list[dict[str, object]] = []

        def save_image(self, image, settings, **kwargs):
            self.started.set()
            try:
                if self.delay:
                    time.sleep(self.delay)
                path = super().save_image(image, settings, **kwargs)
                self.saved.append({
                    "path": str(path), "bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "width": image.width(), "height": image.height(),
                })
                return path
            finally:
                self.finished.set()

    @contextmanager
    def scenario(name: str, *, occupied: bool = False):
        item: dict[str, object] = {"name": name, "ok": False}
        report["scenarios"].append(item)
        directory = temporary_root / name
        directory.mkdir()
        window = QWidget()
        window.setWindowTitle(f"FDM screenshot hotkey validation — {name}")
        layout = QVBoxLayout(window)
        layout.addWidget(QLabel("Synthetic hotkey probe is running.\nPlease do not type or switch windows until it finishes."))
        window.resize(520, 130)
        window.show()
        agent = None
        owner = None
        output = None
        receipts: list[dict[str, object]] = []
        entered = False
        try:
            hwnd = int(window.winId())
            window.raise_()
            window.activateWindow()
            keyboard.user32.SetForegroundWindow(ctypes.c_void_p(hwnd))
            pump(.15)
            if keyboard.foreground() != hwnd:
                raise RuntimeError("Probe window could not acquire foreground; refusing to send keys")
            keyboard.assert_keyboard_idle()
            coordinator = CaptureCoordinator()
            if "Windows" not in type(coordinator.backend).__name__:
                raise AssertionError(f"Expected Windows screenshot backend, got {type(coordinator.backend).__name__}")
            item["capture_backend"] = type(coordinator.backend).__name__
            screen = coordinator.screens()[0].physical_rect
            keys = {mode: HotkeyBinding(binding.sequence, False) for mode, binding in default_hotkeys().items()}
            keys[CaptureMode.FULL_SCREEN] = HotkeyBinding(CHORD_F9)
            keys[CaptureMode.LAST_REGION] = HotkeyBinding(CHORD_F10)
            settings = ScreenshotSettings(
                enabled=True, autostart=False, output_directory=str(directory / "captures"),
                show_editor=False, notification=False, hotkeys=keys, png_compression=1,
                last_region=CaptureRect(screen.x, screen.y, min(96, screen.width), min(64, screen.height)),
            ).normalized()
            settings_path = directory / "settings.json"
            ScreenshotSettingsIO.save(settings, settings_path)
            if occupied:
                owner = WindowsGlobalHotkeyManager(hwnd)
                owner.bind(NativeBinding(0x7001, MODIFIERS, VK_F9))
            output = ProbeOutput()
            agent = ScreenshotAgent(
                app, coordinator=coordinator, settings_path=settings_path,
                output_service=output, autostart_manager_factory=NoAutostart,
            )
            agent.start()
            def received(identifier: int, _lparam: int) -> None:
                status = agent.status()
                traces = status["recent_captures"]
                receipts.append({
                    "identifier": identifier, "time": time.perf_counter(),
                    "stage_at_signal": traces[-1]["stage"] if traces else None,
                    "slow_save_active": output.started.is_set() and not output.finished.is_set(),
                })

            if agent._hotkey_receiver is None:
                raise AssertionError(f"Native hotkey receiver missing: {agent.integration_errors}")
            agent._hotkey_receiver.hotkeyPressed.connect(received)
            item["initial_hotkeys"] = deepcopy(agent.status()["hotkeys"])
            for row in item["initial_hotkeys"]:
                if row["mode"] not in {CaptureMode.FULL_SCREEN.value, CaptureMode.LAST_REGION.value}:
                    continue
                if occupied and row["mode"] == CaptureMode.FULL_SCREEN.value:
                    continue
                if row["state"] != "registered":
                    raise RuntimeError(f"Test chord unavailable; no input sent: {row}")

            def inject(
                key: int, *, repetitions: int = 1, release: bool = True,
                modifiers: tuple[int, ...] = (0x11, 0x10, 0x12),
            ) -> None:
                if keyboard.foreground() != hwnd:
                    raise RuntimeError("Probe lost foreground; refusing to send keys")
                keyboard.assert_keyboard_idle()
                events = [(modifier, False) for modifier in modifiers]
                events.extend((key, False) for _ in range(repetitions))
                if release:
                    events.append((key, True))
                    events.extend((modifier, True) for modifier in reversed(modifiers))
                keyboard.send(events)

            def idle() -> bool:
                status = agent.status()
                return not status["capture_queue_depth"] and not status["output_pending"]

            def capture(
                key: int, expected_mode: CaptureMode,
                *, modifiers: tuple[int, ...] = (0x11, 0x10, 0x12),
            ) -> dict[str, object]:
                before = len(receipts)
                saved_before = len(output.saved)
                inject(key, modifiers=modifiers)
                wait_until(lambda: len(receipts) > before, "No WM_HOTKEY received after SendInput")
                wait_until(lambda: len(output.saved) > saved_before and idle(), "Screenshot did not finish saving")
                pump(.03)
                if len(receipts) != before + 1:
                    raise AssertionError(f"Expected one WM_HOTKEY, got {len(receipts) - before}")
                trace = dict(agent.status()["recent_captures"][-1])
                if trace["mode"] != expected_mode.value:
                    raise AssertionError(f"Wrong capture mode: {trace}")
                if "dispatched_ms" not in trace or "image_ready_ms" not in trace:
                    raise AssertionError(f"Capture trace lacks dispatch/image_ready timings: {trace}")
                if receipts[-1]["stage_at_signal"] != "received":
                    raise AssertionError(f"Capture was not queued before native callback returned: {receipts[-1]}")
                image = QImage(str(output.saved[-1]["path"]))
                if image.isNull():
                    raise AssertionError("Saved PNG cannot be decoded")
                return trace

            entered = True
            yield agent, output, receipts, inject, capture, idle, item, owner, window
            item["ok"] = True
        except Exception as exc:
            item["error"] = str(exc)
            item["traceback"] = traceback.format_exc()
            item["acceptance"] = "failed"
            if not entered:
                raise
        finally:
            cleanup_errors = []
            if agent is not None:
                try:
                    item["final_status"] = deepcopy(agent.status())
                    traces = deepcopy(item["final_status"]["recent_captures"])
                    all_traces.extend(traces)
                    item["timings"] = {
                        label: summarize_ms([float(entry[key]) for entry in traces if key in entry])
                        for label, key in (
                            ("received_to_dispatch", "dispatched_ms"),
                            ("received_to_image_ready", "image_ready_ms"),
                            ("received_to_completed", "completed_ms"),
                        )
                    }
                    item["saved_files"] = list(output.saved) if output else []
                    item["saved_file_count"] = len(item["saved_files"])
                    item["native_message_count"] = len(receipts)
                    item["native_receipts"] = receipts
                except Exception as exc:
                    cleanup_errors.append(str(exc))
            for cleanup in (
                keyboard.release_owned_keys,
                (lambda: agent.tray.hide()) if agent else lambda: None,
                agent.close if agent else lambda: None,
                owner.close if owner else lambda: None,
            ):
                try:
                    cleanup()
                except Exception as exc:
                    cleanup_errors.append(str(exc))
            if cleanup_errors:
                item["ok"] = False
                item["cleanup_errors"] = cleanup_errors
            window.close()
            window.deleteLater()
            pump(.03)
            print(json.dumps({"scenario": name, "ok": item["ok"], "error": item.get("error")}, allow_nan=False), flush=True)

    @contextmanager
    def recording_editor(agent, window, item):
        """Exercise the actual widget and its focus-owned recording signal."""
        editor = ScreenshotHotkeyEdit(window)
        finish = QPushButton("Finish shortcut recording", window)
        owner = "windows-probe-editor"
        native_events: list[dict[str, object]] = []
        recording_changes: list[dict[str, object]] = []
        sequence_changes: list[dict[str, object]] = []
        callback_errors: list[str] = []

        class EventObserver(QObject):
            def eventFilter(self, watched, event):
                if event.type() in (QEvent.Type.KeyPress, QEvent.Type.KeyRelease):
                    native_events.append({
                        "event": "press" if event.type() == QEvent.Type.KeyPress else "release",
                        "qt_key": event.key(), "native_virtual_key": event.nativeVirtualKey(),
                        "native_scan_code": event.nativeScanCode(),
                        "qt_modifiers": event.modifiers().value,
                        "auto_repeat": event.isAutoRepeat(),
                    })
                return False

        observer = EventObserver(editor)
        editor.installEventFilter(observer)
        if editor._line_edit is not None:
            editor._line_edit.installEventFilter(observer)

        def recording_changed(active: bool) -> None:
            try:
                status = agent.set_hotkey_recording(owner, active)
                recording_changes.append({
                    "active": active, "hotkeys_paused": status["hotkeys_paused"],
                    "registered_hotkeys": status["registered_hotkeys"],
                })
            except Exception as exc:
                callback_errors.append(str(exc))

        editor.recordingChanged.connect(recording_changed)

        def sequence_changed(sequence) -> None:
            sequence_changes.append({
                "sequence": sequence.toString(), "binding_at_signal": editor.binding().to_dict(),
                "is_recording": editor.is_recording,
            })

        editor.keySequenceChanged.connect(sequence_changed)
        window.layout().addWidget(editor)
        window.layout().addWidget(finish)
        window.resize(520, 220)
        editor.show()
        finish.show()
        try:
            editor.setFocus(Qt.FocusReason.OtherFocusReason)
            pump(.05)
            assert not callback_errors, callback_errors
            assert editor.is_recording, "Shortcut editor did not enter recording on focus"
            status = agent.status()
            assert status["hotkeys_paused"] and status["registered_hotkeys"] == 0, status
            yield editor, finish, native_events
            assert not callback_errors, callback_errors
        finally:
            item["editor_native_events"] = native_events
            item["editor_recording_transitions"] = recording_changes
            item["editor_sequence_changes"] = sequence_changes
            item["editor_callback_errors"] = callback_errors
            editor.hide()
            agent.set_hotkey_recording(owner, False)
            editor.recordingChanged.disconnect(recording_changed)
            editor.deleteLater()
            finish.deleteLater()
            pump(.02)

    def register_print_settings(agent, settings, item, *, result_key: str) -> bool:
        result = agent.reload_settings({"settings": settings.to_dict()})
        row = next(row for row in agent.status()["hotkeys"] if row["mode"] == CaptureMode.FULL_SCREEN.value)
        item[result_key] = deepcopy(row)
        if row["state"] == "registered" and row.get("virtual_key") == VK_PRINT:
            assert row["modifiers"] & 0x0F == 0, row
            assert not result["hotkey_errors"], result
            return True
        if row["error_code"] == 1409:
            item["acceptance"] = "limited"
            item.setdefault("limitations", []).append(
                "Bare PrintScreen RegisterHotKey failed with 1409. The existing owner was left untouched; "
                "PrintScreen global dispatch is not validated."
            )
            return False
        raise AssertionError(f"Unexpected PrintScreen registration failure: {row}")

    try:
        with scenario("registration_conflict", occupied=True) as state:
            agent, output, receipts, inject, capture, idle, item, owner, window = state
            rows = {row["mode"]: row for row in agent.status()["hotkeys"]}
            assert rows[CaptureMode.FULL_SCREEN.value]["error_code"] == 1409, rows
            assert rows[CaptureMode.LAST_REGION.value]["state"] == "registered", rows
            capture(VK_F10, CaptureMode.LAST_REGION)
            owner.close()
            rows = {row["mode"]: row for row in agent.retry_hotkeys()["hotkeys"]}
            assert rows[CaptureMode.FULL_SCREEN.value]["state"] == "registered", rows
            capture(VK_F9, CaptureMode.FULL_SCREEN)

        with scenario("normal_capture_latency") as state:
            agent, output, receipts, inject, capture, idle, item, owner, window = state
            for _ in range(args.samples):
                capture(VK_F9, CaptureMode.FULL_SCREEN)
            item["sample_count"] = args.samples

        with scenario("recording_pause_restore") as state:
            agent, output, receipts, inject, capture, idle, item, owner, window = state
            capture(VK_F9, CaptureMode.FULL_SCREEN)
            before = len(receipts)
            status = agent.set_hotkey_recording("windows-validation", True)
            assert status["hotkeys_paused"] and status["registered_hotkeys"] == 0, status
            inject(VK_F9)
            pump(.2)
            assert len(receipts) == before, receipts
            agent.set_hotkey_recording("windows-validation", False)
            capture(VK_F9, CaptureMode.FULL_SCREEN)

        with scenario("group_swap") as state:
            agent, output, receipts, inject, capture, idle, item, owner, window = state
            keys = dict(agent.settings.hotkeys)
            keys[CaptureMode.FULL_SCREEN] = HotkeyBinding(CHORD_F10)
            keys[CaptureMode.LAST_REGION] = HotkeyBinding(CHORD_F9)
            result = agent.reload_settings({"settings": replace(agent.settings, hotkeys=keys).to_dict()})
            assert not result["hotkey_errors"], result
            capture(VK_F9, CaptureMode.LAST_REGION)
            capture(VK_F10, CaptureMode.FULL_SCREEN)

        with scenario("editor_native_recording") as state:
            agent, output, receipts, inject, capture, idle, item, owner, window = state
            capture(VK_F9, CaptureMode.FULL_SCREEN)
            receipt_count = len(receipts)
            with recording_editor(agent, window, item) as (editor, finish, native_events):
                inject(VK_F9)
                wait_until(lambda: editor.binding().native_virtual_key == VK_F9, "Real Windows input was not recorded by ScreenshotHotkeyEdit", timeout=2)
                binding = editor.binding()
                assert binding.native_modifiers == MODIFIERS, binding
                assert len(receipts) == receipt_count, "Hotkey fired while the editor had focus"
                assert any(event["native_virtual_key"] == VK_F9 for event in native_events), native_events
                item["recorded_binding"] = binding.to_dict()
                finish.setFocus(Qt.FocusReason.OtherFocusReason)
                wait_until(lambda: not agent.status()["hotkeys_paused"], "Leaving editor did not restore hotkeys")
                pump(.05)
                blurred_binding = editor.binding()
                item["recorded_binding_after_blur"] = blurred_binding.to_dict()
                assert blurred_binding.native_virtual_key == VK_F9 and blurred_binding.native_modifiers == MODIFIERS, blurred_binding
                assert not editor.is_recording
                assert agent.status()["registered_hotkeys"] == 2
                capture(VK_F9, CaptureMode.FULL_SCREEN)
                item["focus_pause_and_restore_verified"] = True

        with scenario("repeat_keydown_one_stroke") as state:
            agent, output, receipts, inject, capture, idle, item, owner, window = state
            inject(VK_F9, repetitions=8, release=False)
            wait_until(lambda: len(receipts) >= 1 and len(output.saved) >= 1 and idle(), "Held key capture did not finish")
            keyboard.send([(VK_F9, False)] * 8)
            pump(.2)
            assert len(receipts) == 1 and len(output.saved) == 1, receipts
            keyboard.release_owned_keys()
            pump(.03)
            capture(VK_F9, CaptureMode.FULL_SCREEN)
            assert len(receipts) == 2, receipts

        with scenario("slow_output_sync_reference") as state:
            agent, output, receipts, inject, capture, idle, item, owner, window = state
            heartbeat: list[float] = []
            timer = QTimer()
            timer.setInterval(10)
            timer.timeout.connect(lambda: heartbeat.append(time.perf_counter()))
            timer.start()
            injected: dict[str, object] = {}
            sender = None
            try:
                frame = agent.coordinator.capture_now(CaptureRequest(CaptureMode.FULL_SCREEN, open_editor=False))
                pump(.06)

                def send_during_blocking_save() -> None:
                    time.sleep(.05)
                    try:
                        injected["at"] = time.perf_counter()
                        inject(VK_F9)
                    except Exception as exc:
                        injected["error"] = str(exc)

                sender = threading.Thread(target=send_during_blocking_save, name="fdm-probe-input")
                output.delay = .3
                sender.start()
                blocking_start = time.perf_counter()
                output.save_image(frame.image, agent.settings, mode=CaptureMode.FULL_SCREEN)
                item["synchronous_save_ms"] = round((time.perf_counter() - blocking_start) * 1000, 3)
                output.delay = 0
                sender.join(timeout=2)
                assert not sender.is_alive(), "Reference input thread did not stop"
                assert "error" not in injected, injected
                wait_until(lambda: len(receipts) == 1 and len(output.saved) == 2 and idle(), "Reference hotkey/capture did not complete")
                pump(.04)
                item["synthetic_send_to_received_ms"] = round((receipts[0]["time"] - float(injected["at"])) * 1000, 3)
                item["injected_save_delay_ms"] = 300
                item["reference_direct_save_count"] = 1
                item["heartbeat_gaps"] = summarize_ms([
                    (later - earlier) * 1000 for earlier, later in zip(heartbeat, heartbeat[1:])
                ])
            finally:
                timer.stop()
                if sender is not None:
                    sender.join(timeout=2)

        with scenario("slow_output_responsiveness") as state:
            agent, output, receipts, inject, capture, idle, item, owner, window = state
            heartbeat: list[float] = []
            timer = QTimer()
            timer.setInterval(10)
            timer.timeout.connect(lambda: heartbeat.append(time.perf_counter()))
            timer.start()
            try:
                output.delay = .3
                inject(VK_F9)
                wait_until(output.started.is_set, "Slow save did not begin")
                assert not output.finished.is_set(), "GUI loop was blocked until the slow save ended"
                during = len(heartbeat)
                injected_at = time.perf_counter()
                inject(VK_F10)
                wait_until(lambda: len(receipts) >= 2, "Second hotkey was not received during slow output")
                assert receipts[1]["slow_save_active"], receipts
                pump(.05)
                assert len(heartbeat) > during, "Qt heartbeat stopped while output was pending"
                wait_until(lambda: len(output.saved) == 2 and idle(), "Slow outputs did not finish", timeout=15)
                item["injected_save_delay_ms"] = 300
                item["synthetic_send_to_received_ms"] = round((receipts[1]["time"] - injected_at) * 1000, 3)
                item["heartbeat_count"] = len(heartbeat)
                item["heartbeat_gaps"] = summarize_ms([
                    (later - earlier) * 1000 for earlier, later in zip(heartbeat, heartbeat[1:])
                ])
            finally:
                timer.stop()

        with scenario("printscreen_global_registration") as state:
            agent, output, receipts, inject, capture, idle, item, owner, window = state
            keys = dict(agent.settings.hotkeys)
            keys[CaptureMode.FULL_SCREEN] = HotkeyBinding("Print", True, VK_PRINT, 0)
            if register_print_settings(agent, replace(agent.settings, hotkeys=keys), item, result_key="printscreen_registration"):
                capture(VK_PRINT, CaptureMode.FULL_SCREEN, modifiers=())
                item["printscreen_global_result"] = "RegisterHotKey_to_WM_HOTKEY_and_actual_saved_capture"
                item["foreground_after_global_printscreen"] = keyboard.foreground_snapshot()
            else:
                item["printscreen_global_result"] = "registration_conflict_1409_no_input_sent"

        # This is intentionally last. PrintScreen may invoke a system or another
        # application's capture UI; never send follow-up input to that window.
        with scenario("editor_printscreen_recording") as state:
            agent, output, receipts, inject, capture, idle, item, owner, window = state
            with recording_editor(agent, window, item) as (editor, finish, native_events):
                inject(VK_PRINT, modifiers=())
                pump(.5)
                binding = editor.binding()
                delivered = any(
                    event["native_virtual_key"] == VK_PRINT
                    or event["qt_key"] in (int(Qt.Key.Key_Print), int(Qt.Key.Key_SysReq))
                    for event in native_events
                )
                item["recorded_binding"] = binding.to_dict()
                item["foreground_after_printscreen"] = keyboard.foreground_snapshot()
                if delivered:
                    assert binding.native_virtual_key == VK_PRINT, binding
                    assert binding.native_modifiers == 0, binding
                    assert binding.sequence == "Print", binding
                    item["printscreen_result"] = "recorded_from_real_windows_event"
                    item["acceptance"] = "passed"
                else:
                    item["printscreen_result"] = "no_printscreen_event_reached_editor"
                    item["acceptance"] = "limited"
                    item["limitation"] = (
                        "No PrintScreen press/release reached the focused editor. System or another "
                        "application interception is possible; the interceptor is not identified. "
                        "PrintScreen recording compatibility is NOT validated in this run."
                    )
                assert not receipts, "A capture hotkey fired while recording PrintScreen"

            # Test the real settings-menu action separately from direct key
            # recording. Invoking this action is not evidence that Qt received
            # PrintScreen; its explicit native key identity is the fallback.
            window.raise_()
            window.activateWindow()
            keyboard.user32.SetForegroundWindow(ctypes.c_void_p(int(window.winId())))
            pump(.15)
            if keyboard.foreground() != int(window.winId()):
                item["acceptance"] = "limited"
                item["special_menu_result"] = "probe_could_not_reacquire_foreground_no_more_input_sent"
            else:
                page = ScreenshotSettingsPage(agent.settings)
                scroll = QScrollArea(window)
                scroll.setWidgetResizable(True)
                scroll.setWidget(page)
                window.layout().addWidget(scroll)
                window.resize(640, 550)
                scroll.show()
                menu_owner = "windows-probe-special-menu"

                def page_recording_changed(active: bool) -> None:
                    agent.set_hotkey_recording(menu_owner, active)

                page.hotkeyRecordingChanged.connect(page_recording_changed)
                try:
                    editor = page.hotkey_edits[CaptureMode.FULL_SCREEN]
                    sequence_changes: list[dict[str, object]] = []
                    item["special_menu_sequence_changes"] = sequence_changes
                    sequence_phase = "before_focus"

                    def menu_sequence_changed(sequence) -> None:
                        sequence_changes.append({
                            "phase": sequence_phase, "sequence": sequence.toString(),
                            "binding_at_signal": editor.binding().to_dict(),
                            "is_recording": editor.is_recording,
                        })

                    editor.keySequenceChanged.connect(menu_sequence_changed)
                    editor.setFocus(Qt.FocusReason.OtherFocusReason)
                    pump(.05)
                    assert agent.status()["hotkeys_paused"], agent.status()
                    button = page.hotkey_special_buttons[CaptureMode.FULL_SCREEN]
                    action = next(action for action in button.menu().actions() if action.data() == "Print")
                    item["special_menu_action"] = {"text": action.text(), "object_name": action.objectName(), "data": action.data()}
                    sequence_phase = "action_trigger"
                    action.trigger()
                    item["special_menu_binding_immediate"] = editor.binding().to_dict()
                    sequence_phase = "after_trigger_event_pump"
                    pump(.05)
                    item["special_menu_editor_binding_after_pump"] = editor.binding().to_dict()
                    binding = page.settings().hotkeys[CaptureMode.FULL_SCREEN]
                    item["special_menu_binding"] = binding.to_dict()
                    assert binding.sequence == "Print" and binding.native_virtual_key == VK_PRINT and binding.native_modifiers == 0, binding
                    assert not editor.is_recording and not agent.status()["hotkeys_paused"], agent.status()
                    if register_print_settings(agent, page.settings(), item, result_key="special_menu_registration"):
                        capture(VK_PRINT, CaptureMode.FULL_SCREEN, modifiers=())
                        item["special_menu_result"] = "menu_action_native_metadata_registration_WM_HOTKEY_and_actual_saved_capture"
                    else:
                        item["special_menu_result"] = "menu_metadata_passed_registration_conflict_1409"
                finally:
                    page.hotkeyRecordingChanged.disconnect(page_recording_changed)
                    agent.set_hotkey_recording(menu_owner, False)
                    scroll.hide()
                    scroll.deleteLater()
                    pump(.02)

    finally:
        try:
            keyboard.release_owned_keys()
        finally:
            keyboard.restore_foreground()
        report["foreground_restore_attempted"] = True
        report["timings"] = {
            "received_to_dispatch": summarize_ms([float(item["dispatched_ms"]) for item in all_traces if "dispatched_ms" in item]),
            "received_to_image_ready": summarize_ms([float(item["image_ready_ms"]) for item in all_traces if "image_ready_ms" in item]),
            "received_to_completed": summarize_ms([float(item["completed_ms"]) for item in all_traces if "completed_ms" in item]),
        }
        report["capture_trace_count"] = len(all_traces)
        report["limited_scenarios"] = [item["name"] for item in report["scenarios"] if item.get("acceptance") == "limited"]
        if not args.keep_captures:
            shutil.rmtree(temporary_root)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--keep-captures", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.samples <= 50:
        parser.error("--samples must be between 1 and 50 (trace history is bounded)")
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {
        "started_at": datetime.now(timezone.utc).isoformat(), "ok": False,
        "platform": platform.platform(), "python": sys.version,
        "perf_counter_resolution_seconds": time.get_clock_info("perf_counter").resolution,
        "input_kind": "synthetic Win32 SendInput through actual RegisterHotKey/WM_HOTKEY",
        "timing_origin": "application WM_HOTKEY receipt, not physical keyboard or host-to-guest latency",
        "limitations": [
            "No physical-key, Parallels host interception, elevated foreground, secure desktop, or keyboard-layout acceptance.",
            "Slow output is a controlled 300 ms delay before actual PNG encoding and saving.",
            "A registration conflict or inability to focus the probe fails that scenario; it is never counted as success.",
        ],
        "scenarios": [],
    }
    report["source_sha256"] = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (
            Path(__file__).resolve(), ROOT / "src/fdm/screenshot_agent.py",
            ROOT / "src/fdm/platform/windows_global_hotkey.py",
            ROOT / "src/fdm/services/screenshot_output_worker.py",
            ROOT / "src/fdm/ui/screenshot_hotkey_edit.py",
            ROOT / "src/fdm/ui/screenshot_settings_page.py",
        ) if path.is_file()
    }
    report["dependencies"] = {}
    for name in ("PySide6", "numpy", "mss"):
        try:
            report["dependencies"][name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            report["dependencies"][name] = None
    try:
        if sys.platform != "win32":
            raise RuntimeError("This probe requires an interactive Windows desktop")
        run_probe(args, report)
        report["ok"] = bool(report["scenarios"]) and all(item["ok"] for item in report["scenarios"])
        report["acceptance"] = (
            "failed" if not report["ok"] else "passed_with_limitations" if report.get("limited_scenarios") else "passed"
        )
    except Exception as exc:
        report["error"] = str(exc)
        report["traceback"] = traceback.format_exc()
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"ok": report["ok"], "report": str(args.output)}, ensure_ascii=False, allow_nan=False), flush=True)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
