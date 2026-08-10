from __future__ import annotations

import argparse
from collections import deque
import ctypes
from dataclasses import dataclass, replace
from pathlib import Path
import re
import sys
from typing import Callable, Mapping, Sequence

from PySide6.QtCore import QByteArray, QObject, QLockFile, QStandardPaths, QThread, QTimer, Qt, Signal
from PySide6.QtGui import QAction, QCursor, QImage
from PySide6.QtNetwork import QLocalServer, QLocalSocket
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QMenu,
    QScrollArea,
    QStyle,
    QSystemTrayIcon,
    QVBoxLayout,
    QWidget,
)

from fdm.screenshot_protocol import (
    CommandType,
    IPCCommand,
    IPCResponse,
    MAX_IPC_MESSAGE_BYTES,
    SCREENSHOT_EXECUTABLE_NAME,
    ScreenshotProtocolError,
    decode_command,
    decode_response,
    encode_ipc_message,
    screenshot_ipc_server_name,
)
from fdm.services.screenshot_capture import (
    CaptureCoordinator,
    CapturedFrame,
    CaptureMode,
    CaptureRequest,
    ScreenInfo,
)
from fdm.screenshot_settings import (
    ScreenshotSettings,
    ScreenshotSettingsIO,
    UnsupportedScreenshotSettingsVersion,
)
from fdm.services.screenshot_output import OutputResult, ScreenshotOutputService
from fdm.ui.screenshot_editor import ScreenshotEditor
from fdm.ui.screenshot_overlay import ScreenshotOverlay, logical_point_to_physical


SCREENSHOT_AUTOSTART_VALUE_NAME = "FiberDiameterMeasurementScreenshotTool"


class AgentCommandError(RuntimeError):
    def __init__(self, message: str, *, result: Mapping[str, object] | None = None) -> None:
        super().__init__(message)
        self.result = dict(result or {})


class _HotkeyReceiver(QWidget):
    """Hidden native HWND that receives process-global ``WM_HOTKEY`` messages."""

    hotkeyPressed = Signal(int, int)

    def nativeEvent(self, event_type: QByteArray, message: int):  # noqa: N802 - Qt API
        del event_type
        if sys.platform == "win32":
            try:
                from ctypes import wintypes
                from fdm.platform.windows_global_hotkey import WM_HOTKEY

                address = int(message)
                msg = ctypes.cast(address, ctypes.POINTER(wintypes.MSG)).contents
                if int(msg.message) == WM_HOTKEY:
                    self.hotkeyPressed.emit(int(msg.wParam), int(msg.lParam))
                    return True, 0
            except (AttributeError, TypeError, ValueError):
                pass
        return False, 0


def _parse_windows_hotkey(sequence: str, identifier: int):
    from fdm.platform.windows_global_hotkey import (
        HotkeyBinding as NativeHotkeyBinding,
        MOD_ALT,
        MOD_CONTROL,
        MOD_SHIFT,
        MOD_WIN,
    )

    modifiers = 0
    virtual_key = 0
    named_keys = {
        "backspace": 0x08,
        "tab": 0x09,
        "return": 0x0D,
        "enter": 0x0D,
        "pause": 0x13,
        "capslock": 0x14,
        "escape": 0x1B,
        "esc": 0x1B,
        "space": 0x20,
        "pageup": 0x21,
        "pgup": 0x21,
        "pagedown": 0x22,
        "pgdown": 0x22,
        "end": 0x23,
        "home": 0x24,
        "left": 0x25,
        "up": 0x26,
        "right": 0x27,
        "down": 0x28,
        "insert": 0x2D,
        "ins": 0x2D,
        "delete": 0x2E,
        "del": 0x2E,
    }
    for part in (token.strip() for token in str(sequence).split("+") if token.strip()):
        token = part.casefold()
        if token in {"ctrl", "control"}:
            modifiers |= MOD_CONTROL
        elif token == "alt":
            modifiers |= MOD_ALT
        elif token == "shift":
            modifiers |= MOD_SHIFT
        elif token in {"win", "windows", "meta"}:
            modifiers |= MOD_WIN
        elif token in {"print", "printscreen", "prtsc", "prtscn"}:
            virtual_key = 0x2C  # VK_SNAPSHOT
        elif token in named_keys:
            virtual_key = named_keys[token]
        elif len(token) == 1 and token.isascii() and token.isalnum():
            virtual_key = ord(token.upper())
        elif token.startswith("f") and token[1:].isdigit() and 1 <= int(token[1:]) <= 24:
            virtual_key = 0x70 + int(token[1:]) - 1
        else:
            raise ValueError(f"不支持的全局快捷键按键：{part}")
    if not virtual_key:
        raise ValueError("全局快捷键缺少主按键。")
    return NativeHotkeyBinding(identifier, modifiers, virtual_key).normalized()


@dataclass(frozen=True, slots=True)
class AgentStartResult:
    primary: bool
    forwarded: bool = False
    response: IPCResponse | None = None
    error: str = ""


class ScreenshotCommandStream:
    """Incremental newline-delimited UTF-8 JSON command decoder."""

    def __init__(self, *, maximum_bytes: int = MAX_IPC_MESSAGE_BYTES) -> None:
        self._buffer = bytearray()
        self._maximum_bytes = max(256, int(maximum_bytes))

    @property
    def pending_bytes(self) -> int:
        return len(self._buffer)

    def feed(self, payload: bytes | bytearray | memoryview | QByteArray) -> tuple[IPCCommand, ...]:
        self._buffer.extend(bytes(payload))
        result: list[IPCCommand] = []
        while True:
            newline = self._buffer.find(b"\n")
            if newline < 0:
                break
            if newline + 1 > self._maximum_bytes:
                self._buffer.clear()
                raise ScreenshotProtocolError("IPC message exceeds the size limit")
            line = bytes(self._buffer[: newline + 1])
            del self._buffer[: newline + 1]
            if line.strip():
                result.append(decode_command(line))
        if len(self._buffer) > self._maximum_bytes:
            self._buffer.clear()
            raise ScreenshotProtocolError("IPC message exceeds the size limit")
        return tuple(result)


class ScreenshotSingleInstance(QObject):
    """QLocalServer primary election plus request forwarding."""

    commandReceived = Signal(object)
    protocolError = Signal(str)

    def __init__(
        self,
        server_name: str | None = None,
        parent: QObject | None = None,
        *,
        lock_file_path: str | Path | None = None,
    ) -> None:
        super().__init__(parent)
        self._server_name = str(server_name or screenshot_ipc_server_name())
        if lock_file_path is None:
            temporary = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.TempLocation)
            directory = Path(temporary) if temporary else Path.cwd()
            safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", self._server_name)
            lock_file_path = directory / "fdm-screenshot-agent" / f"{safe_name}.lock"
        self._lock_path = Path(lock_file_path)
        self._lock = QLockFile(str(self._lock_path))
        self._server = QLocalServer(self)
        self._server.setSocketOptions(QLocalServer.SocketOption.UserAccessOption)
        self._server.newConnection.connect(self._accept_connections)
        self._connections: dict[int, tuple[QLocalSocket, ScreenshotCommandStream]] = {}
        self._command_handler: Callable[[IPCCommand], Mapping[str, object] | IPCResponse | None] | None = None

    @property
    def server_name(self) -> str:
        return self._server_name

    @property
    def is_listening(self) -> bool:
        return self._server.isListening()

    def set_command_handler(
        self,
        handler: Callable[[IPCCommand], Mapping[str, object] | IPCResponse | None] | None,
    ) -> None:
        self._command_handler = handler

    def start_or_forward(
        self,
        command: IPCCommand,
        *,
        timeout_ms: int = 2_000,
    ) -> AgentStartResult:
        first_timeout = min(300, max(1, int(timeout_ms)))
        response = self.send_command(command, timeout_ms=first_timeout)
        if response is not None:
            return AgentStartResult(False, forwarded=True, response=response)

        try:
            self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            return AgentStartResult(False, error=f"无法创建截图工具单实例锁：{exc}")
        if self._lock.tryLock(0):
            if self._server.listen(self._server_name):
                return AgentStartResult(True)
            # We own the process lock, so a local endpoint can only be stale.
            QLocalServer.removeServer(self._server_name)
            if self._server.listen(self._server_name):
                return AgentStartResult(True)
            error = self._server.errorString()
            self._lock.unlock()
            return AgentStartResult(False, error=f"无法建立截图工具本机服务：{error}")

        remaining = max(1, int(timeout_ms) - first_timeout)
        attempts = 4
        for _attempt in range(attempts):
            QThread.msleep(min(100, max(1, remaining // attempts)))
            response = self.send_command(command, timeout_ms=max(1, remaining // attempts))
            if response is not None:
                return AgentStartResult(False, forwarded=True, response=response)
        return AgentStartResult(False, error="截图工具实例正在启动，但本机通信暂不可用。")

    @staticmethod
    def _read_response(socket: QLocalSocket, *, timeout_ms: int) -> IPCResponse | None:
        buffer = bytearray()
        remaining = max(1, int(timeout_ms))
        while b"\n" not in buffer:
            if socket.bytesAvailable() <= 0 and not socket.waitForReadyRead(remaining):
                return None
            buffer.extend(bytes(socket.readAll()))
            if len(buffer) > MAX_IPC_MESSAGE_BYTES:
                return None
        line, _separator, _rest = buffer.partition(b"\n")
        try:
            return decode_response(line + b"\n")
        except ScreenshotProtocolError:
            return None

    def send_command(self, command: IPCCommand, *, timeout_ms: int = 2_000) -> IPCResponse | None:
        socket = QLocalSocket()
        socket.connectToServer(self._server_name)
        if not socket.waitForConnected(max(1, int(timeout_ms))):
            socket.abort()
            return None
        encoded = encode_ipc_message(command)
        if socket.write(encoded) != len(encoded):
            socket.abort()
            return None
        if socket.bytesToWrite() and not socket.waitForBytesWritten(max(1, int(timeout_ms))):
            socket.abort()
            return None
        response = self._read_response(socket, timeout_ms=max(1, int(timeout_ms)))
        socket.disconnectFromServer()
        return response

    def close(self) -> None:
        for socket, _stream in tuple(self._connections.values()):
            socket.abort()
            socket.deleteLater()
        self._connections.clear()
        self._server.close()
        if self._lock.isLocked():
            self._lock.unlock()

    def _accept_connections(self) -> None:
        while self._server.hasPendingConnections():
            socket = self._server.nextPendingConnection()
            if socket is None:
                continue
            key = id(socket)
            self._connections[key] = (socket, ScreenshotCommandStream())
            socket.readyRead.connect(lambda key=key: self._consume(key))
            socket.disconnected.connect(lambda key=key: self._discard(key))
            self._consume(key)

    def _consume(self, key: int) -> None:
        connection = self._connections.get(key)
        if connection is None:
            return
        socket, stream = connection
        try:
            commands = stream.feed(socket.readAll())
        except ScreenshotProtocolError as exc:
            self.protocolError.emit(str(exc))
            socket.abort()
            self._discard(key)
            return
        for command in commands:
            self.commandReceived.emit(command)
            response: IPCResponse
            try:
                result = self._command_handler(command) if self._command_handler is not None else None
                if isinstance(result, IPCResponse):
                    if result.request_id != command.request_id:
                        raise AgentCommandError("IPC handler returned a mismatched request_id")
                    response = result
                else:
                    response = IPCResponse.success(
                        command.request_id,
                        dict(result or {"accepted": True}),
                    )
            except AgentCommandError as exc:
                response = IPCResponse(
                    request_id=command.request_id,
                    ok=False,
                    result=exc.result,
                    error=str(exc) or type(exc).__name__,
                )
            except Exception as exc:  # noqa: BLE001 - IPC error boundary
                response = IPCResponse.failure(
                    command.request_id,
                    str(exc) or type(exc).__name__,
                )
            socket.write(encode_ipc_message(response))
        socket.flush()

    def _discard(self, key: int) -> None:
        connection = self._connections.pop(key, None)
        if connection is None:
            return
        socket, _stream = connection
        socket.deleteLater()


class ScreenshotAgent(QObject):
    """Long-lived tray controller for capture, selection, editing and IPC."""

    def __init__(
        self,
        application: QApplication,
        coordinator: CaptureCoordinator | None = None,
        parent: QObject | None = None,
        *,
        settings_path: str | Path | None = None,
        output_service: ScreenshotOutputService | None = None,
        cu5_locator: object | None = None,
        hotkey_manager_factory: Callable[[int], object] | None = None,
        autostart_manager_factory: Callable[[], object] | None = None,
    ) -> None:
        super().__init__(parent)
        self._application = application
        self._coordinator = coordinator or CaptureCoordinator(parent=self)
        self._settings_path = Path(settings_path) if settings_path is not None else None
        self._settings_load_error = ""
        self._settings_read_only = False
        try:
            self._settings = ScreenshotSettingsIO.load(self._settings_path)
        except UnsupportedScreenshotSettingsVersion as exc:
            self._settings = ScreenshotSettings()
            self._settings_load_error = str(exc) or type(exc).__name__
            self._settings_read_only = True
        except Exception as exc:  # noqa: BLE001 - keep tray available for recovery
            self._settings = ScreenshotSettings()
            self._settings_load_error = str(exc) or type(exc).__name__
        self._settings = self._settings.normalized()
        self._output_service = output_service or ScreenshotOutputService()
        if cu5_locator is None:
            from fdm.services.cu5_preview_locator import Cu5PreviewLocator

            cu5_locator = Cu5PreviewLocator(selector=self._settings.cu5_selector)
        self._cu5_locator = cu5_locator
        self._update_cu5_locator_selector()
        backend_setter = getattr(self._coordinator.backend, "set_cu5_locator", None)
        if callable(backend_setter):
            backend_setter(self._cu5_locator)
        self._hotkey_manager_factory = hotkey_manager_factory
        self._autostart_manager_factory = autostart_manager_factory
        self._hotkey_receiver: _HotkeyReceiver | None = None
        self._hotkey_manager: object | None = None
        self._autostart_manager: object | None = None
        self._hotkey_modes: dict[int, CaptureMode] = {}
        self._initialization_errors: list[str] = []
        self._integration_errors: list[str] = []
        self._overlay: ScreenshotOverlay | None = None
        self._settings_window: QDialog | None = None
        self._editors: list[ScreenshotEditor] = []
        self._ipc_capture_queue: deque[CaptureRequest] = deque()
        self._ipc_capture_timer = QTimer(self)
        self._ipc_capture_timer.setSingleShot(True)
        self._ipc_capture_timer.setInterval(0)
        self._ipc_capture_timer.timeout.connect(self._dispatch_next_ipc_capture)
        self._default_delay_ms = self._settings.delay_ms
        self._coordinator.set_last_region(self._settings.last_region)
        icon = application.style().standardIcon(QStyle.StandardPixmap.SP_ComputerIcon)
        self.tray = QSystemTrayIcon(icon, self)
        self.tray.setToolTip("Fiber Screenshot Tool")
        self.tray.setContextMenu(self._build_menu())
        self.tray.activated.connect(self._tray_activated)
        self._coordinator.selectionRequested.connect(self._show_selection_overlay)
        self._coordinator.captureReady.connect(self._capture_ready)
        self._coordinator.captureFailed.connect(self._capture_failed)
        self._application.aboutToQuit.connect(self.close)

    @property
    def coordinator(self) -> CaptureCoordinator:
        return self._coordinator

    @property
    def editors(self) -> tuple[ScreenshotEditor, ...]:
        return tuple(self._editors)

    @property
    def settings(self) -> ScreenshotSettings:
        return self._settings.normalized()

    @property
    def integration_errors(self) -> tuple[str, ...]:
        return tuple(self._integration_errors)

    def start(self) -> None:
        self._initialize_windows_integrations()
        self.tray.show()

    def close(self) -> None:
        self._ipc_capture_timer.stop()
        self._ipc_capture_queue.clear()
        manager = self._hotkey_manager
        self._hotkey_manager = None
        if manager is not None:
            try:
                manager.close()
            except Exception:  # noqa: BLE001 - process is already shutting down
                pass
        if self._hotkey_receiver is not None:
            self._hotkey_receiver.close()
            self._hotkey_receiver.deleteLater()
            self._hotkey_receiver = None

    def handle_command(self, command: IPCCommand) -> Mapping[str, object]:
        if command.command is CommandType.CAPTURE:
            payload = dict(command.payload)
            payload["mode"] = command.capture_mode.value if command.capture_mode is not None else CaptureMode.REGION.value
            request = CaptureRequest.from_mapping(payload)
            if "open_editor" not in command.payload:
                request = replace(request, open_editor=self._settings.show_editor)
            if "include_cursor" not in command.payload:
                request = replace(
                    request,
                    include_cursor=self._settings.include_cursor,
                )
            self._enqueue_ipc_capture(request)
            return {
                "accepted": True,
                "mode": request.mode.value,
                "queued": len(self._ipc_capture_queue),
            }
        elif command.command is CommandType.SHUTDOWN:
            QTimer.singleShot(0, self._application.quit)
            return {"accepted": True, "shutting_down": True}
        elif command.command is CommandType.SHOW_SETTINGS:
            QTimer.singleShot(0, self._show_settings_window)
            return {"accepted": True, "settings_window": "requested"}
        elif command.command is CommandType.UPDATE_SETTINGS:
            return self.reload_settings(command.payload)
        elif command.command is CommandType.DIAGNOSE_CU5:
            return self.diagnose_cu5()
        elif command.command in {CommandType.PING, CommandType.STATUS}:
            self.tray.show()
            return self.status()
        raise AgentCommandError(f"不支持的截图工具命令：{command.command.value}")

    def status(self) -> dict[str, object]:
        last_region = self._settings.last_region
        return {
            "running": True,
            "settings_loaded": not bool(self._settings_load_error),
            "settings_error": self._settings_load_error,
            "settings_read_only": self._settings_read_only,
            "background_resident": self._settings.background_resident,
            "autostart": self._settings.autostart,
            "registered_hotkeys": len(self._hotkey_modes),
            "integration_errors": list(self._integration_errors),
            "last_region": (
                {
                    "x": last_region.x,
                    "y": last_region.y,
                    "width": last_region.width,
                    "height": last_region.height,
                    "coordinate_space": "physical_pixels",
                }
                if last_region is not None
                else None
            ),
        }

    @staticmethod
    def _cu5_candidate_payload(candidate: object) -> dict[str, object]:
        from fdm.services.cu5_preview_locator import Cu5PreviewSelector

        record = getattr(candidate, "record", candidate)
        rect = record.rect
        payload = {
            "hwnd": int(record.hwnd),
            "rect": {
                "x": int(rect.left),
                "y": int(rect.top),
                "width": int(rect.width),
                "height": int(rect.height),
                "coordinate_space": "physical_pixels",
            },
            "score": float(getattr(candidate, "score", 0.0)),
            "reasons": list(getattr(candidate, "reasons", ())),
        }
        try:
            selector = getattr(candidate, "selector", None)
            payload["selector"] = Cu5PreviewSelector.from_value(
                selector or Cu5PreviewSelector.from_record(record)
            ).to_dict()
        except (AttributeError, TypeError, ValueError):
            pass
        return payload

    def diagnose_cu5(self) -> dict[str, object]:
        from fdm.services.cu5_preview_locator import Cu5PreviewAmbiguousError

        try:
            match = self._cu5_locator.locate()
        except Cu5PreviewAmbiguousError as exc:
            raise AgentCommandError(
                str(exc),
                result={
                    "diagnostic": "ambiguous",
                    "candidates": [
                        self._cu5_candidate_payload(candidate)
                        for candidate in exc.candidates
                    ],
                },
            ) from exc
        except Exception as exc:  # noqa: BLE001 - native diagnostic boundary
            raise AgentCommandError(str(exc) or type(exc).__name__) from exc
        self._remember_cu5_match(match)
        return {
            "diagnostic": "ok",
            **self._cu5_candidate_payload(match),
        }

    def reload_settings(self, payload: Mapping[str, object] | None = None) -> dict[str, object]:
        previous = self._settings.normalized()
        raw = dict(payload or {})
        supplied = raw.get("settings")
        if isinstance(supplied, Mapping):
            update = dict(supplied)
            # The standalone settings page edits user preferences only.  A
            # capture or CU-5 diagnosis may have refreshed these runtime-owned
            # values while the dialog was open.
            update.pop("last_region", None)
            update.pop("cu5_selector", None)
        elif raw and not bool(raw.get("reload", False)):
            update = {
                key: value
                for key, value in raw.items()
                if key not in {"reload"}
            }
        else:
            update = {}
        # The file schema is selected by this executable, not by an IPC caller.
        update.pop("schema_version", None)

        if update:
            if self._settings_read_only:
                raise AgentCommandError(
                    "截图设置由更高版本创建，当前版本以只读方式运行，未覆盖原文件。"
                )
        apply_errors: list[str] = []
        failed_hotkey_modes: set[CaptureMode] = set()
        native_settings_applied = False

        def apply_candidate(candidate: ScreenshotSettings) -> ScreenshotSettings:
            nonlocal apply_errors, native_settings_applied
            # Apply native integrations before committing the settings file. A
            # conflicting hotkey is transactional per capture mode: the manager
            # restores the old registration and the persisted settings retain
            # that same old chord across companion restarts.
            self._settings = candidate.normalized()
            failed_hotkey_modes.clear()
            apply_errors = self._apply_windows_settings(
                failed_hotkey_modes=failed_hotkey_modes,
            )
            native_settings_applied = True
            if failed_hotkey_modes:
                hotkeys = dict(self._settings.hotkeys)
                for mode in failed_hotkey_modes:
                    fallback = previous.hotkeys.get(mode)
                    if fallback is None:
                        hotkeys.pop(mode, None)
                    else:
                        hotkeys[mode] = fallback
                self._settings = replace(
                    self._settings,
                    hotkeys=hotkeys,
                ).normalized()
            return self._settings

        try:
            if update:
                def merge_payload(persisted: ScreenshotSettings) -> ScreenshotSettings:
                    merged = persisted.to_dict()
                    merged.update(update)
                    return apply_candidate(
                        ScreenshotSettings.from_dict(merged).normalized()
                    )

                settings = ScreenshotSettingsIO.update(
                    merge_payload,
                    self._settings_path,
                )
            else:
                settings = ScreenshotSettingsIO.load(self._settings_path).normalized()
                settings = apply_candidate(settings)
                if failed_hotkey_modes:
                    corrected_hotkeys = dict(settings.hotkeys)
                    settings = ScreenshotSettingsIO.update(
                        lambda persisted: replace(
                            persisted,
                            hotkeys=corrected_hotkeys,
                        ).normalized(),
                        self._settings_path,
                    )
        except UnsupportedScreenshotSettingsVersion as exc:
            if native_settings_applied:
                self._settings = previous
                self._apply_windows_settings()
            self._settings_read_only = True
            self._settings_load_error = str(exc) or type(exc).__name__
            raise AgentCommandError(
                "截图设置版本高于当前软件，已保持只读。"
            ) from exc
        except Exception as exc:  # noqa: BLE001 - atomic persistence boundary
            # Native changes have already happened. Restore both global
            # hotkeys and HKCU autostart to the last committed settings before
            # surfacing the save failure to IPC/the settings dialog.
            self._settings = previous
            rollback_errors = (
                self._apply_windows_settings()
                if native_settings_applied
                else []
            )
            self._update_cu5_locator_selector()
            self._default_delay_ms = previous.delay_ms
            self._coordinator.set_last_region(previous.last_region)
            details = [*apply_errors, *rollback_errors]
            self._integration_errors = [*self._initialization_errors, *details]
            suffix = (
                "；系统集成回滚提示：" + "；".join(rollback_errors)
                if rollback_errors
                else ""
            )
            raise AgentCommandError(f"截图设置保存失败：{exc}{suffix}") from exc

        self._update_cu5_locator_selector()
        self._settings_load_error = ""
        self._settings_read_only = False
        self._default_delay_ms = settings.delay_ms
        self._coordinator.set_last_region(settings.last_region)
        self._integration_errors = [
            *self._initialization_errors,
            *apply_errors,
        ]
        return {
            "reloaded": True,
            "hotkey_errors": [
                item for item in self._integration_errors if "快捷键" in item
            ],
            "integration_errors": list(self._integration_errors),
            "settings": self._settings.to_dict(),
        }

    def _initialize_windows_integrations(self) -> None:
        if sys.platform != "win32" and self._hotkey_manager_factory is None and self._autostart_manager_factory is None:
            return
        if self._hotkey_manager is None:
            try:
                self._hotkey_receiver = _HotkeyReceiver()
                self._hotkey_receiver.setAttribute(Qt.WidgetAttribute.WA_NativeWindow, True)
                hwnd = int(self._hotkey_receiver.winId())
                if self._hotkey_manager_factory is not None:
                    self._hotkey_manager = self._hotkey_manager_factory(hwnd)
                else:
                    from fdm.platform.windows_global_hotkey import WindowsGlobalHotkeyManager

                    self._hotkey_manager = WindowsGlobalHotkeyManager(hwnd)
                self._hotkey_receiver.hotkeyPressed.connect(self._hotkey_pressed)
            except Exception as exc:  # noqa: BLE001 - optional Windows integration
                self._initialization_errors.append(f"全局快捷键初始化失败：{exc}")
                self._hotkey_manager = None
        if self._autostart_manager is None:
            try:
                if self._autostart_manager_factory is not None:
                    self._autostart_manager = self._autostart_manager_factory()
                else:
                    from fdm.platform.windows_autostart import WindowsAutostartManager

                    arguments = (
                        ("--background",)
                        if getattr(sys, "frozen", False)
                        else ("-m", "fdm.screenshot_agent", "--background")
                    )
                    self._autostart_manager = WindowsAutostartManager(
                        value_name=SCREENSHOT_AUTOSTART_VALUE_NAME,
                        executable=Path(sys.executable).resolve(),
                        arguments=arguments,
                    )
            except Exception as exc:  # noqa: BLE001 - optional Windows integration
                self._initialization_errors.append(f"开机自启初始化失败：{exc}")
                self._autostart_manager = None
        self._integration_errors = [
            *self._initialization_errors,
            *self._apply_windows_settings(),
        ]

    def _apply_windows_settings(
        self,
        *,
        failed_hotkey_modes: set[CaptureMode] | None = None,
    ) -> list[str]:
        # A newer companion may have added fields or changed the meaning of
        # existing settings.  Loading defaults is sufficient to keep this
        # process usable for diagnostics, but must never be allowed to mutate
        # registrations that belong to that newer version.
        if self._settings_read_only:
            return []
        errors: list[str] = []
        manager = self._hotkey_manager
        self._hotkey_modes.clear()
        if manager is not None:
            desired: dict[int, tuple[CaptureMode, object]] = {}
            retain_after_error: dict[int, CaptureMode] = {}
            if self._settings.enabled:
                identifiers = {
                    mode: 0x5F00 + index
                    for index, mode in enumerate(
                        (
                            CaptureMode.REGION,
                            CaptureMode.WINDOW,
                            CaptureMode.FULL_SCREEN,
                            CaptureMode.LAST_REGION,
                            CaptureMode.CU5,
                        ),
                        start=1,
                    )
                }
                for mode, binding in self._settings.hotkeys.items():
                    if not binding.enabled or mode not in identifiers:
                        continue
                    identifier = identifiers[mode]
                    try:
                        desired[identifier] = (
                            mode,
                            _parse_windows_hotkey(binding.sequence, identifier),
                        )
                    except Exception as exc:  # noqa: BLE001 - surface parse errors
                        errors.append(f"{mode.value} 快捷键注册失败：{exc}")
                        if failed_hotkey_modes is not None:
                            failed_hotkey_modes.add(mode)
                        retain_after_error[identifier] = mode
            existing_ids = {binding.identifier for binding in tuple(manager.bindings)}
            for identifier, (mode, binding) in desired.items():
                try:
                    manager.bind(binding)
                except Exception as exc:  # noqa: BLE001 - manager restores old binding
                    errors.append(f"{mode.value} 快捷键注册失败：{exc}")
                    if failed_hotkey_modes is not None:
                        failed_hotkey_modes.add(mode)
                if manager.binding(identifier) is not None:
                    self._hotkey_modes[identifier] = mode
            for identifier, mode in retain_after_error.items():
                if manager.binding(identifier) is not None:
                    self._hotkey_modes[identifier] = mode
            retained_ids = set(retain_after_error)
            for identifier in sorted(existing_ids - desired.keys() - retained_ids):
                try:
                    manager.unbind(identifier)
                except Exception as exc:  # noqa: BLE001 - report stale registrations
                    errors.append(f"全局快捷键注销失败：{exc}")

        if self._autostart_manager is not None:
            try:
                self._autostart_manager.set_enabled(self._settings.autostart)
            except Exception as exc:  # noqa: BLE001 - registry error is surfaced over IPC
                errors.append(f"开机自启同步失败：{exc}")
        return errors

    def _hotkey_pressed(self, identifier: int, lparam: int) -> None:
        manager = self._hotkey_manager
        if manager is None:
            return
        try:
            from fdm.platform.windows_global_hotkey import WM_HOTKEY

            binding = manager.binding_for_message(WM_HOTKEY, identifier, lparam)
        except Exception:  # noqa: BLE001 - ignore malformed native messages
            return
        mode = self._hotkey_modes.get(binding.identifier) if binding is not None else None
        if mode is not None:
            self.begin_capture(self._request(mode))

    def _enqueue_ipc_capture(self, request: CaptureRequest) -> None:
        self._ipc_capture_queue.append(request)
        if not self._ipc_capture_timer.isActive():
            self._ipc_capture_timer.start()

    def _dispatch_next_ipc_capture(self) -> None:
        if not self._ipc_capture_queue:
            return
        request = self._ipc_capture_queue.popleft()
        self.begin_capture(request)
        if self._ipc_capture_queue:
            self._ipc_capture_timer.start()

    def begin_capture(self, request: CaptureRequest) -> None:
        previous_overlay = self._overlay
        if previous_overlay is not None:
            # Starting a second command must retire the old selector before an
            # immediate full-screen/window capture can include it.  Leave a few
            # compositor frames before capture and let the request-identity
            # guard reject the old overlay's delayed completion callback.
            self._overlay = None
            previous_overlay.hide()
            previous_overlay.close()
        metadata = {**request.metadata, "open_editor": request.open_editor}
        if request.delay_ms == 0 and self._default_delay_ms:
            request = replace(request, delay_ms=self._default_delay_ms, metadata=metadata)
        else:
            request = replace(request, metadata=metadata)
        if previous_overlay is not None and request.delay_ms < 80:
            request = replace(request, delay_ms=80)
        self._coordinator.start(request)

    def _request(self, mode: CaptureMode, *, delay_ms: int | None = None) -> CaptureRequest:
        physical_cursor = None
        if mode is CaptureMode.DISPLAY:
            logical_cursor = QCursor.pos()
            screens = self._coordinator.screens()
            physical_cursor = logical_point_to_physical(logical_cursor, screens)
        return CaptureRequest(
            mode=mode,
            delay_ms=self._default_delay_ms if delay_ms is None else max(0, int(delay_ms)),
            cursor_position=physical_cursor,
            open_editor=self._settings.show_editor,
            include_cursor=self._settings.include_cursor,
        )

    def _show_settings_window(self) -> None:
        if self._settings_window is not None:
            self._settings_window.show()
            self._settings_window.raise_()
            self._settings_window.activateWindow()
            return
        from fdm.ui.screenshot_settings_page import ScreenshotSettingsPage

        dialog = QDialog()
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        dialog.setWindowTitle("Fiber Screenshot Tool 设置")
        target_screen = self._application.screenAt(QCursor.pos())
        if target_screen is None:
            target_screen = self._application.primaryScreen()
        available = target_screen.availableGeometry() if target_screen is not None else None
        width = min(620, max(320, available.width() - 32)) if available is not None else 620
        height = min(720, max(360, available.height() - 48)) if available is not None else 720
        dialog.resize(width, height)
        layout = QVBoxLayout(dialog)
        scroll = QScrollArea(dialog)
        scroll.setWidgetResizable(True)
        scroll.setProperty("redirectEditorWheel", True)
        page = ScreenshotSettingsPage(self._settings)
        page.set_agent_status(True, f"已注册 {len(self._hotkey_modes)} 个全局快捷键")
        scroll.setWidget(page)
        layout.addWidget(scroll, 1)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel,
            dialog,
        )
        buttons.accepted.connect(lambda: self._save_settings_page(page, dialog))
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        def diagnose() -> None:
            try:
                result = self.diagnose_cu5()
                rect = result["rect"]
                page.set_cu5_diagnostic_status(
                    f"已识别 HWND {result['hwnd']}：{rect['width']}×{rect['height']}，"
                    f"得分 {result['score']:.1f}",
                    success=True,
                )
            except AgentCommandError as exc:
                page.set_cu5_diagnostic_status(str(exc), success=False)

        page.cu5DiagnosticRequested.connect(diagnose)
        dialog.destroyed.connect(lambda: self._clear_settings_window(dialog))
        self._settings_window = dialog
        dialog.show()

    def _save_settings_page(self, page: object, dialog: QDialog) -> None:
        try:
            settings = page.settings()
            result = self.reload_settings({"settings": settings.to_dict()})
        except Exception as exc:  # noqa: BLE001 - visible settings save boundary
            self.tray.showMessage(
                "设置保存失败",
                str(exc) or type(exc).__name__,
                QSystemTrayIcon.MessageIcon.Warning,
            )
            return
        errors = result.get("integration_errors", [])
        if errors:
            self.tray.showMessage(
                "设置已保存，部分系统集成未启用",
                "\n".join(str(item) for item in errors),
                QSystemTrayIcon.MessageIcon.Warning,
            )
        dialog.accept()
        if not self._settings.enabled:
            QTimer.singleShot(0, self._application.quit)

    def _clear_settings_window(self, dialog: QDialog) -> None:
        if self._settings_window is dialog:
            self._settings_window = None

    def _build_menu(self) -> QMenu:
        menu = QMenu()
        modes = (
            ("区域截图", CaptureMode.REGION),
            ("智能窗口/控件", CaptureMode.SMART),
            ("活动窗口", CaptureMode.ACTIVE_WINDOW),
            ("当前显示器", CaptureMode.DISPLAY),
            ("全屏", CaptureMode.FULL_SCREEN),
            ("上次区域", CaptureMode.LAST_REGION),
            ("CU-5 实时预览", CaptureMode.CU5),
        )
        for label, mode in modes:
            action = QAction(label, menu)
            action.triggered.connect(
                lambda _checked=False, item=mode: self._begin_tray_capture(item)
            )
            menu.addAction(action)
        delay_menu = menu.addMenu("延时截图")
        for seconds in (3, 5, 10):
            action = QAction(f"{seconds} 秒后区域截图", delay_menu)
            action.triggered.connect(
                lambda _checked=False, value=seconds: self.begin_capture(
                    self._request(CaptureMode.REGION, delay_ms=value * 1000)
                )
            )
            delay_menu.addAction(action)
        menu.addSeparator()
        settings_action = QAction("设置…", menu)
        settings_action.triggered.connect(
            lambda _checked=False: self._show_settings_window()
        )
        menu.addAction(settings_action)
        menu.addSeparator()
        quit_action = QAction("退出截图工具", menu)
        quit_action.triggered.connect(self._application.quit)
        menu.addAction(quit_action)
        return menu

    def _begin_tray_capture(self, mode: CaptureMode) -> None:
        request = self._request(mode)
        minimum_delay = 0
        if mode is CaptureMode.ACTIVE_WINDOW:
            # The tray menu itself is the foreground window while its action is
            # firing.  Resolve the active target only after the menu has closed.
            minimum_delay = 150
        elif mode in {
            CaptureMode.DISPLAY,
            CaptureMode.FULL_SCREEN,
            CaptureMode.LAST_REGION,
            CaptureMode.CU5,
        }:
            minimum_delay = 80
        if request.delay_ms < minimum_delay:
            request = replace(request, delay_ms=minimum_delay)
        self.begin_capture(request)

    def _tray_activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        if reason in {
            QSystemTrayIcon.ActivationReason.Trigger,
            QSystemTrayIcon.ActivationReason.DoubleClick,
        }:
            self.begin_capture(self._request(CaptureMode.REGION))

    def _show_selection_overlay(self, request: CaptureRequest, candidates: object) -> None:
        if self._overlay is not None:
            self._overlay.close()
        typed_candidates = tuple(candidates) if isinstance(candidates, (tuple, list)) else ()
        overlay = ScreenshotOverlay(self._coordinator.screens(), typed_candidates)
        self._overlay = overlay
        def accept_after_overlay_hides(selection: object) -> None:
            overlay.hide()

            def finish() -> None:
                self._coordinator.complete_selection(
                    selection,
                    expected_request=request,
                )
                overlay.close()

            finish_timer = QTimer(overlay)
            finish_timer.setSingleShot(True)
            finish_timer.timeout.connect(finish)
            # Keep both Qt and Python ownership tied to the overlay. Closing a
            # stale selector destroys the timer and cancels its callback before
            # the wrapper can be touched again.
            overlay._capture_finish_timer = finish_timer
            finish_timer.start(80)

        overlay.selectionAccepted.connect(accept_after_overlay_hides)
        overlay.cancelled.connect(self._coordinator.cancel)
        overlay.cancelled.connect(overlay.close)
        overlay.destroyed.connect(lambda: self._clear_overlay(overlay))
        overlay.begin()

    def _clear_overlay(self, overlay: ScreenshotOverlay) -> None:
        if self._overlay is overlay:
            self._overlay = None

    def _capture_ready(self, frame: CapturedFrame) -> None:
        if frame.mode is CaptureMode.CU5:
            self._remember_cu5_match(
                getattr(self._coordinator.backend, "last_cu5_match", None)
            )
        last_region = self._coordinator.last_region
        if (
            not self._settings_read_only
            and last_region is not None
            and last_region != self._settings.last_region
        ):
            try:
                self._settings = ScreenshotSettingsIO.update(
                    lambda persisted: replace(
                        persisted,
                        last_region=last_region,
                    ).normalized(),
                    self._settings_path,
                )
            except (OSError, UnsupportedScreenshotSettingsVersion) as exc:
                if isinstance(exc, UnsupportedScreenshotSettingsVersion):
                    self._settings_read_only = True
                    self._settings_load_error = str(exc) or type(exc).__name__
                self.tray.showMessage(
                    "截图设置保存失败",
                    f"上次截图区域未能持久化：{exc}",
                    QSystemTrayIcon.MessageIcon.Warning,
                )

        if bool(frame.metadata.get("open_editor", self._settings.show_editor)):
            editor = ScreenshotEditor(frame.image)
            editor.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
            self._editors.append(editor)
            editor.destroyed.connect(lambda: self._forget_editor(editor))
            editor.completed.connect(
                lambda image, item=editor, mode=frame.mode: self._complete_editor(
                    item,
                    image,
                    mode,
                )
            )
            editor.show()
            editor.raise_()
            editor.activateWindow()
            return
        self._publish_capture(frame.image, frame.mode)

    def _update_cu5_locator_selector(self) -> None:
        setter = getattr(self._cu5_locator, "set_selector", None)
        if callable(setter):
            setter(self._settings.cu5_selector)

    def _remember_cu5_match(self, match: object | None) -> None:
        """Persist only restart-stable CU-5 features, never native handles."""

        if match is None or self._settings_read_only:
            return
        try:
            from fdm.services.cu5_preview_locator import Cu5PreviewSelector

            selector = getattr(match, "selector", None)
            if selector is None:
                record = getattr(match, "record", None)
                if record is None or not hasattr(record, "class_name"):
                    return
                selector = Cu5PreviewSelector.from_record(record)
            payload = Cu5PreviewSelector.from_value(selector).to_dict()
        except (AttributeError, TypeError, ValueError):
            return
        if not payload or payload == self._settings.cu5_selector:
            return
        try:
            updated = ScreenshotSettingsIO.update(
                lambda persisted: replace(
                    persisted,
                    cu5_selector=payload,
                ).normalized(),
                self._settings_path,
            )
        except (OSError, UnsupportedScreenshotSettingsVersion) as exc:
            if isinstance(exc, UnsupportedScreenshotSettingsVersion):
                self._settings_read_only = True
                self._settings_load_error = str(exc) or type(exc).__name__
            self.tray.showMessage(
                "截图设置保存失败",
                f"CU-5 预览特征未能持久化：{exc}",
                QSystemTrayIcon.MessageIcon.Warning,
            )
            return
        self._settings = updated
        self._update_cu5_locator_selector()

    def _complete_editor(
        self,
        editor: ScreenshotEditor,
        image: object,
        mode: CaptureMode,
    ) -> None:
        if not isinstance(image, QImage) or image.isNull():
            self._capture_failed("编辑器返回了空截图。")
            return
        if self._publish_capture(image, mode):
            editor.close()

    def _publish_capture(self, image: QImage, mode: CaptureMode) -> bool:
        try:
            result: OutputResult = self._output_service.process_capture(
                image,
                self._settings,
                mode=mode,
            )
        except Exception as exc:  # noqa: BLE001 - user-facing output pipeline
            self._capture_failed(str(exc) or type(exc).__name__)
            return False
        if result.errors:
            completed: list[str] = []
            if result.saved_path is not None:
                completed.append(f"已保存到 {result.saved_path}")
            if result.copied_to_clipboard:
                completed.append("已复制到剪贴板")
            prefix = "；".join(completed) or "至少一项输出已完成"
            self.tray.showMessage(
                "截图部分完成",
                f"{prefix}；{result.failure_summary}",
                QSystemTrayIcon.MessageIcon.Warning,
            )
            return True
        if result.notification_requested:
            details: list[str] = []
            if result.saved_path is not None:
                details.append(f"已保存到 {result.saved_path}")
            if result.copied_to_clipboard:
                details.append("已复制到剪贴板")
            self.tray.showMessage("截图完成", "；".join(details) or "截图处理完成。")
        return True

    def _forget_editor(self, editor: ScreenshotEditor) -> None:
        if editor in self._editors:
            self._editors.remove(editor)

    def _capture_failed(self, message: str) -> None:
        self.tray.showMessage("截图失败", str(message), QSystemTrayIcon.MessageIcon.Warning)


def build_initial_command(arguments: argparse.Namespace) -> IPCCommand:
    if arguments.shutdown:
        return IPCCommand(CommandType.SHUTDOWN)
    if arguments.show_settings:
        return IPCCommand(CommandType.SHOW_SETTINGS)
    if arguments.capture:
        payload = {
            "delay_ms": max(0, int(arguments.delay_ms)),
            "open_editor": not bool(arguments.no_editor),
        }
        return IPCCommand.capture(arguments.capture, payload=payload)
    return IPCCommand(CommandType.STATUS)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=SCREENSHOT_EXECUTABLE_NAME)
    parser.add_argument("--capture", choices=[mode.value for mode in CaptureMode])
    parser.add_argument("--delay-ms", type=int, default=0)
    parser.add_argument("--no-editor", action="store_true")
    parser.add_argument("--background", action="store_true")
    parser.add_argument("--show-settings", action="store_true")
    parser.add_argument("--shutdown", action="store_true")
    parser.add_argument("--server-name", default=screenshot_ipc_server_name(), help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_argument_parser().parse_args(list(argv) if argv is not None else None)
    existing = QApplication.instance()
    app = existing if isinstance(existing, QApplication) else QApplication(sys.argv[:1])
    app.setApplicationName("FiberScreenshotTool")
    app.setApplicationDisplayName("Fiber Screenshot Tool")
    app.setQuitOnLastWindowClosed(False)

    command = build_initial_command(arguments)
    instance = ScreenshotSingleInstance(arguments.server_name)
    result = instance.start_or_forward(command)
    if not result.primary:
        if result.forwarded:
            return 0 if result.response is None or result.response.ok else 1
        if result.error:
            print(result.error, file=sys.stderr)
        return 1

    agent = ScreenshotAgent(app)
    instance.set_command_handler(agent.handle_command)
    app.aboutToQuit.connect(instance.close)
    agent.start()
    if command.command not in {CommandType.STATUS, CommandType.PING}:
        QTimer.singleShot(0, lambda: agent.handle_command(command))
    return int(app.exec())


__all__ = [
    "AgentStartResult",
    "ScreenshotAgent",
    "ScreenshotCommandStream",
    "ScreenshotSingleInstance",
    "build_argument_parser",
    "build_initial_command",
    "main",
]


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())
