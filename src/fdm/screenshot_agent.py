from __future__ import annotations

import argparse
from collections import OrderedDict, deque
import ctypes
from dataclasses import dataclass, replace
from pathlib import Path
import re
import sys
import time
from datetime import datetime
from uuid import uuid4
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
    should_open_annotation,
)
from fdm.screenshot_settings import (
    AfterCaptureTask,
    ScreenshotSettings,
    ScreenshotSettingsIO,
    UnsupportedScreenshotSettingsVersion,
)
from fdm.services.screenshot_output import OutputResult, ScreenshotOutputService
from fdm.ui.screenshot_editor import ScreenshotEditModel, ScreenshotEditor
from fdm.ui.screenshot_annotation_overlay import InlineAnnotationOverlay
from fdm.ui.screenshot_overlay import ScreenshotOverlay, logical_point_to_physical


SCREENSHOT_AUTOSTART_VALUE_NAME = "FiberDiameterMeasurementScreenshotTool"
_HOTKEY_MODES = (
    CaptureMode.REGION, CaptureMode.WINDOW, CaptureMode.FULL_SCREEN,
    CaptureMode.LAST_REGION, CaptureMode.CU5,
)
_HOTKEY_IDS = {mode: 0x5F00 + index for index, mode in enumerate(_HOTKEY_MODES, 1)}
_MAX_CAPTURE_QUEUE = 8
_RECORDING_LEASE_SECONDS = 15.0


# Qt's PortableText names are translated to the documented Win32 virtual-key
# codes consumed by RegisterHotKey.  Keep aliases for hand-edited legacy
# settings, but use Qt's canonical spellings as the primary keys.
_WINDOWS_NAMED_VIRTUAL_KEYS = {
    "cancel": 0x03,  # VK_CANCEL
    "backspace": 0x08,  # VK_BACK
    "tab": 0x09,  # VK_TAB
    "clear": 0x0C,  # VK_CLEAR
    "return": 0x0D,  # VK_RETURN
    "enter": 0x0D,
    "pause": 0x13,  # VK_PAUSE
    "break": 0x13,
    "capslock": 0x14,  # VK_CAPITAL
    "escape": 0x1B,  # VK_ESCAPE
    "esc": 0x1B,
    "space": 0x20,  # VK_SPACE
    "pageup": 0x21,  # VK_PRIOR
    "pgup": 0x21,
    "pagedown": 0x22,  # VK_NEXT
    "pgdown": 0x22,
    "end": 0x23,  # VK_END
    "home": 0x24,  # VK_HOME
    "left": 0x25,  # VK_LEFT
    "up": 0x26,  # VK_UP
    "right": 0x27,  # VK_RIGHT
    "down": 0x28,  # VK_DOWN
    "select": 0x29,  # VK_SELECT
    "execute": 0x2B,  # VK_EXECUTE
    "insert": 0x2D,  # VK_INSERT
    "ins": 0x2D,
    "delete": 0x2E,  # VK_DELETE
    "del": 0x2E,
    "help": 0x2F,  # VK_HELP
    "power off": 0x5E,  # VK_POWER
    "poweroff": 0x5E,
    # The physical context-menu key is VK_APPS.  VK_MENU is the Win32 name
    # for Alt and must not be used for Qt's Key_Menu.
    "menu": 0x5D,  # VK_APPS
    "apps": 0x5D,
    "application": 0x5D,
    "context menu": 0x5D,
    "application menu": 0x5D,
    "sleep": 0x5F,  # VK_SLEEP
    "standby": 0x5F,
    "numlock": 0x90,  # VK_NUMLOCK
    "scrolllock": 0x91,  # VK_SCROLL
    "back": 0xA6,  # VK_BROWSER_BACK
    "browser back": 0xA6,
    "forward": 0xA7,  # VK_BROWSER_FORWARD
    "browser forward": 0xA7,
    "refresh": 0xA8,  # VK_BROWSER_REFRESH
    "browser refresh": 0xA8,
    "stop": 0xA9,  # VK_BROWSER_STOP
    "browser stop": 0xA9,
    "search": 0xAA,  # VK_BROWSER_SEARCH
    "browser search": 0xAA,
    "favorites": 0xAB,  # VK_BROWSER_FAVORITES
    "browser favorites": 0xAB,
    "home page": 0xAC,  # VK_BROWSER_HOME
    "homepage": 0xAC,
    "browser home": 0xAC,
    "volume mute": 0xAD,  # VK_VOLUME_MUTE
    "volume down": 0xAE,  # VK_VOLUME_DOWN
    "volume up": 0xAF,  # VK_VOLUME_UP
    "media next": 0xB0,  # VK_MEDIA_NEXT_TRACK
    "media next track": 0xB0,
    "media previous": 0xB1,  # VK_MEDIA_PREV_TRACK
    "media previous track": 0xB1,
    "media prev": 0xB1,
    "media stop": 0xB2,  # VK_MEDIA_STOP
    "media play": 0xB3,  # VK_MEDIA_PLAY_PAUSE
    "media pause": 0xB3,
    "media play/pause": 0xB3,
    "toggle media play/pause": 0xB3,
    "launch mail": 0xB4,  # VK_LAUNCH_MAIL
    "launch media": 0xB5,  # VK_LAUNCH_MEDIA_SELECT
    "launch media select": 0xB5,
    "launch (0)": 0xB6,  # VK_LAUNCH_APP1
    "launch app 1": 0xB6,
    "launch (1)": 0xB7,  # VK_LAUNCH_APP2
    "launch app 2": 0xB7,
    "play": 0xFA,  # VK_PLAY
    "zoom": 0xFB,  # VK_ZOOM
}

# Qt preserves printable keys in PortableText instead of naming the physical
# Windows key. RegisterHotKey accepts virtual-key codes, so translate common
# US/Chinese-layout glyphs back to their OEM keys. In particular, the top-left
# ` key may be reported as the middle dot used by a Chinese IME.
_WINDOWS_PRINTABLE_VIRTUAL_KEYS = {
    "`": 0xC0,  # VK_OEM_3
    "~": 0xC0,
    "·": 0xC0,
    "-": 0xBD,  # VK_OEM_MINUS
    "_": 0xBD,
    "=": 0xBB,  # VK_OEM_PLUS
    "+": 0xBB,
    "[": 0xDB,  # VK_OEM_4
    "{": 0xDB,
    "]": 0xDD,  # VK_OEM_6
    "}": 0xDD,
    "\\": 0xDC,  # VK_OEM_5
    "|": 0xDC,
    ";": 0xBA,  # VK_OEM_1
    ":": 0xBA,
    "'": 0xDE,  # VK_OEM_7
    '"': 0xDE,
    ",": 0xBC,  # VK_OEM_COMMA
    "<": 0xBC,
    ".": 0xBE,  # VK_OEM_PERIOD
    ">": 0xBE,
    "/": 0xBF,  # VK_OEM_2
    "?": 0xBF,
}

_WINDOWS_KEYPAD_VIRTUAL_KEYS = {
    "0": 0x60,  # VK_NUMPAD0
    "1": 0x61,
    "2": 0x62,
    "3": 0x63,
    "4": 0x64,
    "5": 0x65,
    "6": 0x66,
    "7": 0x67,
    "8": 0x68,
    "9": 0x69,
    "*": 0x6A,  # VK_MULTIPLY
    "+": 0x6B,  # VK_ADD
    "-": 0x6D,  # VK_SUBTRACT
    ".": 0x6E,  # VK_DECIMAL
    "/": 0x6F,  # VK_DIVIDE
    "insert": 0x60,
    "ins": 0x60,
    "end": 0x61,
    "down": 0x62,
    "pagedown": 0x63,
    "pgdown": 0x63,
    "left": 0x64,
    "clear": 0x65,
    "right": 0x66,
    "home": 0x67,
    "up": 0x68,
    "pageup": 0x69,
    "pgup": 0x69,
    "delete": 0x6E,
    "del": 0x6E,
    "enter": 0x0D,  # RegisterHotKey cannot distinguish keypad Enter.
    "return": 0x0D,
}


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

    raw_sequence = str(sequence or "").strip()
    plus_key = raw_sequence.endswith("+")
    split_source = raw_sequence[:-1] if plus_key else raw_sequence
    parts = [part.strip() for part in split_source.split("+") if part.strip()]
    if plus_key:
        parts.append("+")

    modifiers = 0
    keypad = False
    key_token = ""
    key_label = ""
    for part in parts:
        token = part.casefold()
        if token in {"ctrl", "control"}:
            modifiers |= MOD_CONTROL
        elif token == "alt":
            modifiers |= MOD_ALT
        elif token == "shift":
            modifiers |= MOD_SHIFT
        elif token in {"win", "windows", "meta", "super", "cmd", "command"}:
            modifiers |= MOD_WIN
        elif token in {"num", "keypad", "numpad"}:
            keypad = True
        else:
            if key_token:
                raise ValueError("全局快捷键只能包含一个主按键。")
            key_token = token
            key_label = part

    if not key_token:
        raise ValueError("全局快捷键缺少主按键。")

    compact_key = re.sub(r"[\s_-]+", "", key_token)
    if keypad:
        virtual_key = _WINDOWS_KEYPAD_VIRTUAL_KEYS.get(key_token, 0)
    elif compact_key in {
        "print",
        "printscreen",
        "prtsc",
        "prtscn",
        "snapshot",
        "sysreq",
    }:
        virtual_key = 0x2C  # VK_SNAPSHOT
    elif key_token in _WINDOWS_NAMED_VIRTUAL_KEYS:
        virtual_key = _WINDOWS_NAMED_VIRTUAL_KEYS[key_token]
    elif key_token in _WINDOWS_PRINTABLE_VIRTUAL_KEYS:
        virtual_key = _WINDOWS_PRINTABLE_VIRTUAL_KEYS[key_token]
    elif len(key_token) == 1 and key_token.isascii() and key_token.isalnum():
        virtual_key = ord(key_token.upper())
    elif (
        key_token.startswith("f")
        and key_token[1:].isdigit()
        and 1 <= int(key_token[1:]) <= 24
    ):
        virtual_key = 0x70 + int(key_token[1:]) - 1
    else:
        virtual_key = 0
    if not virtual_key:
        raise ValueError(f"不支持的全局快捷键按键：{key_label}")
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
        self._hotkey_diagnostics: dict[CaptureMode, dict[str, object]] = {}
        self._effective_hotkey_text: dict[int, str] = {}
        self._recording_leases: dict[str, float] = {}
        self._recording_timer = QTimer(self)
        self._recording_timer.setInterval(1_000)
        self._recording_timer.timeout.connect(self._expire_recording_leases)
        self._capture_history: deque[dict[str, object]] = deque(maxlen=64)
        self._active_capture_request: CaptureRequest | None = None
        self._ipc_capture_receipts: OrderedDict[str, tuple[dict, dict]] = OrderedDict()
        self._shutting_down = False
        self._shutdown_output_failed = False
        self._output_queue = None
        self._output_jobs: dict[str, dict[str, object]] = {}
        self._output_sequence = 0
        self._last_successful_region_output_sequence = 0
        self._initialization_errors: list[str] = []
        self._integration_errors: list[str] = []
        self._overlay: ScreenshotOverlay | None = None
        self._settings_window: QDialog | None = None
        self._editors: list[ScreenshotEditor] = []
        self._annotation_session: QWidget | None = None
        self._annotation_frame: CapturedFrame | None = None
        self._last_output_error = ""
        self._pending_annotation_styles: dict[str, object] | None = None
        self._annotation_style_timer = QTimer(self)
        self._annotation_style_timer.setSingleShot(True)
        self._annotation_style_timer.setInterval(250)
        self._annotation_style_timer.timeout.connect(self._persist_annotation_styles)
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
        self._coordinator.captureFailed.connect(self._capture_backend_failed)
        self._coordinator.cancelled.connect(self._capture_cancelled)
        self._application.aboutToQuit.connect(self.close)

    @property
    def coordinator(self) -> CaptureCoordinator:
        return self._coordinator

    @property
    def editors(self) -> tuple[ScreenshotEditor, ...]:
        return tuple(self._editors)

    @property
    def annotation_session(self) -> QWidget | None:
        return self._annotation_session

    @property
    def settings(self) -> ScreenshotSettings:
        return self._settings.normalized()

    @property
    def integration_errors(self) -> tuple[str, ...]:
        return tuple(self._integration_errors)

    def start(self) -> None:
        self._initialize_windows_integrations()
        self.tray.show()

    def request_shutdown(self) -> None:
        if self._shutting_down:
            return
        self._shutting_down = True
        self._shutdown_output_failed = False
        self._ipc_capture_timer.stop()
        self._ipc_capture_queue.clear()
        self._coordinator.cancel(emit_signal=False)
        if self._overlay is not None:
            self._overlay.close()
            self._overlay = None
        queue = self._output_queue
        if queue is None or queue.is_shutdown_complete:
            self._application.quit()
            return
        queue.shutdownFinished.connect(self._finish_shutdown)
        queue.shutdown()

    def _finish_shutdown(self) -> None:
        if not self._shutting_down:
            return
        if self._output_jobs:
            # Process already posted completion notifications before deciding
            # whether it is safe to discard the annotation session.
            QTimer.singleShot(0, self, self._finish_shutdown)
            return
        session = self._annotation_session
        if self._shutdown_output_failed and session is not None:
            from shiboken6 import isValid

            if isValid(session):
                # shutdownFinished means every accepted file request and the
                # worker thread have ended. Retire this closed queue before a
                # retry lazily creates a fresh one; no old job can cross over.
                queue = self._output_queue
                self._output_queue = None
                if queue is not None:
                    queue.deleteLater()
                self._shutting_down = False
                self._shutdown_output_failed = False
                session.setEnabled(True)
                self.tray.showMessage(
                    "已取消退出",
                    "截图输出未全部成功，已保留标注。请重试保存、复制或另存为后再退出。",
                    QSystemTrayIcon.MessageIcon.Warning,
                )
                return
        self._application.quit()

    def close(self) -> None:
        self._shutting_down = True
        self._coordinator.cancel(emit_signal=False)
        self._active_capture_request = None
        self._recording_timer.stop()
        self._recording_leases.clear()
        if self._output_queue is not None:
            self._output_queue.shutdown()
            self._output_queue.wait_for_shutdown()
        self._annotation_style_timer.stop()
        if self._pending_annotation_styles is not None:
            self._persist_annotation_styles()
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
        session = self._annotation_session
        self._annotation_session = None
        self._annotation_frame = None
        if session is not None:
            session.close()

    def handle_command(self, command: IPCCommand) -> Mapping[str, object]:
        if command.command is CommandType.CAPTURE:
            receipt = self._ipc_capture_receipts.get(command.request_id)
            fingerprint = command.to_dict()
            if receipt is not None:
                if receipt[0] != fingerprint:
                    raise AgentCommandError("同一请求编号不能用于不同截图命令。")
                return {**receipt[1], "duplicate": True}
            payload = dict(command.payload)
            payload["mode"] = command.capture_mode.value if command.capture_mode is not None else CaptureMode.REGION.value
            request = CaptureRequest.from_mapping(payload)
            if "include_cursor" not in command.payload:
                request = replace(
                    request,
                    include_cursor=self._settings.include_cursor,
                )
            result = self._enqueue_capture(request, source="ipc", trigger_id=command.request_id)
            self._ipc_capture_receipts[command.request_id] = (fingerprint, result)
            while len(self._ipc_capture_receipts) > 128:
                self._ipc_capture_receipts.popitem(last=False)
            return result
        elif command.command is CommandType.SHUTDOWN:
            QTimer.singleShot(0, self, self.request_shutdown)
            return {"accepted": True, "shutting_down": True}
        elif command.command is CommandType.RETRY_HOTKEYS:
            return self.retry_hotkeys()
        elif command.command is CommandType.HOTKEY_RECORDING:
            owner = str(command.payload.get("owner", ""))
            active = command.payload.get("active")
            if not owner or len(owner) > 128 or not isinstance(active, bool):
                raise AgentCommandError("快捷键录入需要有效的会话编号和布尔状态。")
            return self.set_hotkey_recording(owner, active)
        elif command.command is CommandType.SHOW_SETTINGS:
            QTimer.singleShot(0, self, self._show_settings_window)
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
            "registered_hotkeys": len(self._hotkey_manager.bindings) if self._hotkey_manager is not None else 0,
            "hotkeys": list(self._hotkey_diagnostics.values()),
            "hotkeys_paused": bool(self._recording_leases),
            "capture_queue_depth": len(self._ipc_capture_queue),
            "capture_queue_limit": _MAX_CAPTURE_QUEUE,
            "output_pending": len(self._output_jobs),
            "recent_captures": list(self._capture_history),
            "shutting_down": self._shutting_down,
            "annotation_active": self._annotation_session is not None,
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
            "title": str(getattr(record, "title", "") or ""),
            "class_name": str(getattr(record, "class_name", "") or ""),
            "process_name": str(getattr(record, "process_name", "") or ""),
            "control_id": getattr(record, "control_id", None),
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
            locate_with_candidates = getattr(
                self._cu5_locator,
                "locate_with_candidates",
                None,
            )
            if callable(locate_with_candidates):
                match, candidates = locate_with_candidates()
            else:
                match = self._cu5_locator.locate()
                candidates = (match,)
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
            "candidates": [
                self._cu5_candidate_payload(candidate)
                for candidate in tuple(candidates)[:8]
            ],
        }

    def reload_settings(self, payload: Mapping[str, object] | None = None) -> dict[str, object]:
        if self._recording_leases:
            raise AgentCommandError("正在录入快捷键，请先移出录入框再保存设置。")
        previous = self._settings.normalized()
        raw = dict(payload or {})
        supplied = raw.get("settings")
        if isinstance(supplied, Mapping):
            update = dict(supplied)
            # The standalone settings page edits user preferences only.  A
            # capture or CU-family diagnosis may have refreshed these runtime-owned
            # values while the dialog was open.
            update.pop("last_region", None)
            update.pop("cu5_selector", None)
            update.pop("annotation_styles", None)
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
            "hotkeys": list(self._hotkey_diagnostics.values()),
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
        old_rows = self._hotkey_diagnostics
        self._hotkey_diagnostics = {}
        desired: dict[int, tuple[CaptureMode, object]] = {}
        paused = bool(self._recording_leases)
        for mode in _HOTKEY_MODES:
            binding = self._settings.hotkeys.get(mode)
            identifier = _HOTKEY_IDS[mode]
            enabled = bool(self._settings.enabled and binding and binding.enabled)
            row = {
                "mode": mode.value, "requested": binding.sequence if binding else "",
                "effective": "", "state": "disabled" if not enabled else "paused" if paused else "unavailable",
                "error_code": 0, "error": "", "last_received_at": old_rows.get(mode, {}).get("last_received_at"),
            }
            self._hotkey_diagnostics[mode] = row
            if not enabled or paused:
                continue
            try:
                from fdm.platform.windows_global_hotkey import HotkeyBinding as NativeBinding
                vk = getattr(binding, "native_virtual_key", None)
                modifiers = getattr(binding, "native_modifiers", None)
                native = (
                    NativeBinding(identifier, modifiers, vk).normalized()
                    if vk is not None and modifiers is not None
                    else _parse_windows_hotkey(binding.sequence, identifier)
                )
                desired[identifier] = (mode, native)
            except Exception as exc:
                row.update(state="invalid", error=str(exc), error_code=getattr(exc, "error_code", 0))
                errors.append(f"{mode.value} 快捷键无效：{exc}")
                if failed_hotkey_modes is not None:
                    failed_hotkey_modes.add(mode)
                # An invalid edit must not discard a working old registration.
                old = manager.binding(identifier) if manager is not None else None
                if old is not None:
                    desired[identifier] = (mode, old)
        if manager is not None:
            before = {binding.identifier: binding for binding in manager.bindings}
            group_bind = getattr(manager, "bind_all", None)
            if before and callable(group_bind):
                try:
                    group_bind(item[1] for item in desired.values())
                except Exception as exc:
                    affected = {
                        mode for mode in _HOTKEY_MODES
                        if before.get(_HOTKEY_IDS[mode]) != (
                            desired[_HOTKEY_IDS[mode]][1] if _HOTKEY_IDS[mode] in desired else None
                        )
                    }
                    for mode in affected:
                        self._hotkey_diagnostics[mode].update(
                            state="failed", error=str(exc), error_code=getattr(exc, "error_code", 0),
                        )
                    errors.append(f"全局快捷键更新失败：{exc}")
                    if failed_hotkey_modes is not None:
                        failed_hotkey_modes.update(affected)
            else:
                # First startup admits independent available keys even if one is
                # occupied. Existing groups use bind_all for swaps and rollback.
                for identifier, (mode, binding) in desired.items():
                    try:
                        manager.bind(binding)
                    except Exception as exc:
                        self._hotkey_diagnostics[mode].update(
                            state="failed", error=str(exc), error_code=getattr(exc, "error_code", 0),
                        )
                        errors.append(f"{mode.value} 快捷键注册失败：{exc}")
                        if failed_hotkey_modes is not None:
                            failed_hotkey_modes.add(mode)
                for identifier in sorted(before.keys() - desired.keys()):
                    try:
                        manager.unbind(identifier)
                    except Exception as exc:
                        mode = next((m for m, i in _HOTKEY_IDS.items() if i == identifier), None)
                        if mode is not None:
                            self._hotkey_diagnostics[mode].update(state="failed", error=str(exc), error_code=getattr(exc, "error_code", 0))
                        errors.append(f"全局快捷键注销失败：{exc}")
            for mode, identifier in _HOTKEY_IDS.items():
                actual = manager.binding(identifier)
                row = self._hotkey_diagnostics[mode]
                if actual is None:
                    self._effective_hotkey_text.pop(identifier, None)
                    continue
                requested = desired.get(identifier)
                if requested and actual == requested[1] and row["state"] != "invalid":
                    # A failed group rollback can leave this new registration
                    # active. Preserve the failure, but report the actual key.
                    self._effective_hotkey_text[identifier] = str(row["requested"])
                    if row["state"] != "failed":
                        row["state"] = "registered"
                elif actual != before.get(identifier):
                    self._effective_hotkey_text.pop(identifier, None)
                if row["state"] == "failed" and actual == before.get(identifier):
                    row["state"] = "restored"
                row["effective"] = self._effective_hotkey_text.get(identifier, f"VK {actual.virtual_key:#x}")
                row["virtual_key"] = actual.virtual_key
                row["modifiers"] = actual.modifiers
                if not paused and self._settings.enabled:
                    self._hotkey_modes[identifier] = mode

        if self._autostart_manager is not None:
            try:
                self._autostart_manager.set_enabled(self._settings.autostart)
            except Exception as exc:  # noqa: BLE001 - registry error is surfaced over IPC
                errors.append(f"开机自启同步失败：{exc}")
        return errors

    def _hotkey_pressed(self, identifier: int, lparam: int) -> None:
        if self._recording_leases or self._shutting_down or not self._settings.enabled:
            return
        manager = self._hotkey_manager
        if manager is None:
            return
        try:
            from fdm.platform.windows_global_hotkey import WM_HOTKEY

            binding = manager.binding_for_message(WM_HOTKEY, identifier, lparam)
        except Exception:  # noqa: BLE001 - ignore malformed native messages
            return
        mode = self._hotkey_modes.get(binding.identifier) if binding is not None else None
        configured = self._settings.hotkeys.get(mode) if mode is not None else None
        if configured is None or not configured.enabled:
            return
        if mode is not None:
            row = self._hotkey_diagnostics.get(mode)
            if row is not None:
                row["last_received_at"] = datetime.now().isoformat(timespec="milliseconds")
            try:
                self._enqueue_capture(self._request(mode), source="hotkey")
            except AgentCommandError as exc:
                self._capture_failed(str(exc))

    def retry_hotkeys(self) -> dict[str, object]:
        self._integration_errors = [*self._initialization_errors, *self._apply_windows_settings()]
        return self.status()

    def set_hotkey_recording(self, owner: str, active: bool) -> dict[str, object]:
        was_paused = bool(self._recording_leases)
        if active:
            if owner not in self._recording_leases and len(self._recording_leases) >= 16:
                raise AgentCommandError("快捷键录入会话过多，请关闭其它设置窗口后重试。")
            self._recording_leases[owner] = time.monotonic() + _RECORDING_LEASE_SECONDS
            self._recording_timer.start()
        else:
            self._recording_leases.pop(owner, None)
        if was_paused != bool(self._recording_leases):
            if active:
                pending = self._active_capture_request
                if pending is not None and pending.metadata.get("_trigger_source") == "hotkey":
                    # A delayed hotkey may already have left the dispatch queue.
                    # It must not open a selector over the recording editor.
                    self._coordinator.cancel(emit_signal=False)
                    self._trace_capture(pending, "recording_paused")
                    self._active_capture_request = None
                    if self._overlay is not None:
                        self._overlay.close()
                        self._overlay = None
                retained = []
                for request in self._ipc_capture_queue:
                    if request.metadata.get("_trigger_source") == "hotkey":
                        self._trace_capture(request, "recording_paused")
                    else:
                        retained.append(request)
                self._ipc_capture_queue.clear()
                self._ipc_capture_queue.extend(retained)
            self.retry_hotkeys()
        if not self._recording_leases:
            self._recording_timer.stop()
        return self.status()

    def _expire_recording_leases(self) -> None:
        now = time.monotonic()
        for owner, deadline in tuple(self._recording_leases.items()):
            if deadline <= now:
                self.set_hotkey_recording(owner, False)

    def _trace_capture(self, request: CaptureRequest, stage: str, **details: object) -> None:
        trigger_id = str(request.metadata.get("_trigger_id", ""))
        if not trigger_id:
            return
        entry = next((item for item in reversed(self._capture_history) if item["trigger_id"] == trigger_id), None)
        if entry is not None:
            entry["stage"] = stage
            entry[f"{stage}_ms"] = round((time.perf_counter() - float(entry["received_counter"])) * 1_000, 3)
            entry.update(details)

    def _enqueue_capture(self, request: CaptureRequest, *, source: str, trigger_id: str | None = None) -> dict[str, object]:
        if self._shutting_down:
            raise AgentCommandError("截图工具正在退出，未接受新的截图。")
        if source == "hotkey":
            for queued in self._ipc_capture_queue:
                if queued.mode is request.mode and queued.metadata.get("_trigger_source") == source:
                    return {"accepted": True, "coalesced": True, "trigger_id": queued.metadata["_trigger_id"], "mode": request.mode.value, "queued": len(self._ipc_capture_queue)}
        if len(self._ipc_capture_queue) >= _MAX_CAPTURE_QUEUE:
            raise AgentCommandError("截图请求队列已满，请稍后重试。", result={"accepted": False, "queued": len(self._ipc_capture_queue)})
        trigger_id = trigger_id or uuid4().hex
        request = replace(request, metadata={**request.metadata, "_trigger_id": trigger_id, "_trigger_source": source})
        # perf_counter uses QPC on Windows, including Python versions whose
        # monotonic clock only resolves roughly 15.6 ms intervals.
        self._capture_history.append({"trigger_id": trigger_id, "source": source, "mode": request.mode.value, "stage": "received", "received_counter": time.perf_counter(), "received_at": datetime.now().isoformat(timespec="milliseconds")})
        self._ipc_capture_queue.append(request)
        if not self._ipc_capture_timer.isActive():
            self._ipc_capture_timer.start()
        return {"accepted": True, "trigger_id": trigger_id, "mode": request.mode.value, "queued": len(self._ipc_capture_queue)}

    def _enqueue_ipc_capture(self, request: CaptureRequest) -> None:
        self._enqueue_capture(request, source="ipc")

    def _dispatch_next_ipc_capture(self) -> None:
        if not self._ipc_capture_queue:
            return
        request = self._ipc_capture_queue.popleft()
        try:
            self.begin_capture(request)
        except Exception as exc:  # noqa: BLE001 - one bad request must not stall later work
            message = str(exc) or type(exc).__name__
            self._trace_capture(request, "capture_failed", error=message)
            self._coordinator.cancel(emit_signal=False)
            self._active_capture_request = None
            self._capture_failed(message)
        if self._ipc_capture_queue:
            self._ipc_capture_timer.start()

    def _queue_ui_capture(self, request: CaptureRequest) -> None:
        try:
            self._enqueue_capture(request, source="tray")
        except AgentCommandError as exc:
            self._capture_failed(str(exc))

    def begin_capture(self, request: CaptureRequest) -> None:
        if self._shutting_down:
            self._trace_capture(request, "shutdown_rejected")
            return
        self._trace_capture(request, "dispatched")
        session = self._annotation_session
        if session is not None:
            self._trace_capture(request, "annotation_active")
            activate = getattr(session, "activate_session", None)
            if callable(activate):
                activate("已有截图正在标注，请先完成或取消。")
            else:
                session.show()
                session.raise_()
                session.activateWindow()
            self.tray.showMessage(
                "截图标注尚未完成",
                "请先完成或取消当前标注，再开始新的截图。",
                QSystemTrayIcon.MessageIcon.Information,
            )
            return
        previous_overlay = self._overlay
        if previous_overlay is not None:
            # Starting a second command must retire the old selector before an
            # immediate full-screen/window capture can include it.  Leave a few
            # compositor frames before capture and let the request-identity
            # guard reject the old overlay's delayed completion callback.
            self._overlay = None
            previous_overlay.hide()
            previous_overlay.close()
        open_editor = should_open_annotation(
            request.mode,
            request.open_editor,
            default=self._settings.show_editor,
        )
        metadata = {
            **request.metadata,
            "open_editor": open_editor,
            "open_editor_explicit": request.open_editor is not None,
        }
        if request.delay_ms == 0 and self._default_delay_ms:
            request = replace(request, delay_ms=self._default_delay_ms, metadata=metadata)
        else:
            request = replace(request, metadata=metadata)
        if previous_overlay is not None and request.delay_ms < 80:
            request = replace(request, delay_ms=80)
        if self._active_capture_request is not None:
            self._trace_capture(self._active_capture_request, "superseded")
        self._active_capture_request = request
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
            open_editor=None,
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
        from fdm.ui.screenshot_hotkey_status import ScreenshotHotkeyStatusController

        dialog._hotkey_status_controller = ScreenshotHotkeyStatusController(page, self.handle_command)
        dialog.finished.connect(dialog._hotkey_status_controller.close)
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

        def diagnose() -> bool:
            try:
                result = self.diagnose_cu5()
                rect = result["rect"]
                page.set_cu5_diagnostic_status(
                    f"已识别 HWND {result['hwnd']}：{rect['width']}×{rect['height']}，"
                    f"得分 {result['score']:.1f}",
                    success=True,
                )
                page.set_cu5_candidates(
                    result.get("candidates", ()),
                    selected_selector=result.get("selector"),
                )
                return True
            except AgentCommandError as exc:
                page.set_cu5_diagnostic_status(str(exc), success=False)
                page.set_cu5_candidates(
                    exc.result.get("candidates", ()),
                    selected_selector=None,
                )
                return False

        def select_candidate(selector: object) -> None:
            from fdm.services.cu5_preview_locator import Cu5PreviewSelector

            stable_selector = Cu5PreviewSelector.from_value(selector)
            if not stable_selector.active:
                return
            selector = stable_selector.to_dict()
            previous_selector = dict(self._settings.cu5_selector)
            try:
                self.reload_settings({"cu5_selector": dict(selector)})
            except AgentCommandError as exc:
                page.set_cu5_diagnostic_status(str(exc), success=False)
                return
            if diagnose():
                return
            failure_message = page.cu5_status_label.text()
            try:
                self.reload_settings({"cu5_selector": previous_selector})
            except AgentCommandError as exc:
                page.set_cu5_diagnostic_status(
                    f"所选对象验证失败，且恢复原预览对象失败：{exc}",
                    success=False,
                )
            else:
                page.set_cu5_diagnostic_status(
                    f"{failure_message}；已恢复原预览对象。",
                    success=False,
                )

        page.cu5DiagnosticRequested.connect(diagnose)
        page.cu5CandidateSelectionRequested.connect(select_candidate)
        dialog.destroyed.connect(lambda: self._clear_settings_window(dialog))
        self._settings_window = dialog
        dialog.show()
        dialog._hotkey_status_controller.refresh()

    def _save_settings_page(self, page: object, dialog: QDialog) -> None:
        try:
            dialog._hotkey_status_controller.finish_recording()
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
            QTimer.singleShot(0, self._application, self._application.quit)

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
            ("CU 系列实时预览", CaptureMode.CU5),
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
                lambda _checked=False, value=seconds: self._queue_ui_capture(
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
        quit_action.triggered.connect(self.request_shutdown)
        menu.addAction(quit_action)
        return menu

    def _begin_tray_capture(self, mode: CaptureMode) -> None:
        request = self._request(mode)
        minimum_delay = 0
        if mode in {
            CaptureMode.REGION,
            CaptureMode.SMART,
            CaptureMode.WINDOW,
            CaptureMode.ACTIVE_WINDOW,
        }:
            # The tray menu itself is the foreground window while its action is
            # firing. Resolve interactive targets only after the menu closes.
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
        self._queue_ui_capture(request)

    def _tray_activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        if reason in {
            QSystemTrayIcon.ActivationReason.Trigger,
            QSystemTrayIcon.ActivationReason.DoubleClick,
        }:
            self._queue_ui_capture(self._request(CaptureMode.REGION))

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
        self._trace_capture(request, "overlay_shown")

    def _clear_overlay(self, overlay: ScreenshotOverlay) -> None:
        if self._overlay is overlay:
            self._overlay = None

    def _capture_ready(self, frame: CapturedFrame) -> None:
        if self._shutting_down:
            return
        self._active_capture_request = None
        self._trace_capture(CaptureRequest(frame.mode, metadata=frame.metadata), "image_ready")
        if frame.mode is CaptureMode.CU5:
            self._remember_cu5_match(
                getattr(self._coordinator.backend, "last_cu5_match", None)
            )
        open_editor = should_open_annotation(
            frame.mode,
            bool(frame.metadata.get("open_editor", self._settings.show_editor)),
            default=self._settings.show_editor,
        )
        if open_editor:
            self._open_inline_annotation(frame)
            return
        if isinstance(self._output_service, ScreenshotOutputService):
            self._output_async(frame.image, frame.mode, frame.rect, trace_metadata=frame.metadata)
        else:
            self._publish_capture(frame.image, frame.mode, capture_rect=frame.rect)

    def _open_inline_annotation(self, frame: CapturedFrame) -> None:
        if self._annotation_session is not None:
            self._annotation_session.raise_()
            self._annotation_session.activateWindow()
            return
        screens = self._coordinator.screens()
        if not screens or not any(
            frame.rect.intersection(screen.physical_rect) is not None
            for screen in screens
        ):
            self._open_fallback_editor(
                ScreenshotEditModel(frame.image),
                frame,
                reason="当前显示器布局无法安全映射原截图位置。",
            )
            return
        overlay = InlineAnnotationOverlay(
            frame,
            screens,
            styles=self._settings.annotation_styles,
            screens_provider=self._coordinator.screens,
        )
        self._annotation_session = overlay
        self._annotation_frame = frame
        overlay.completed.connect(
            lambda image, item=overlay, captured=frame: self._complete_annotation(
                item, image, captured, "configured"
            )
        )
        overlay.copyRequested.connect(
            lambda image, item=overlay, captured=frame: self._complete_annotation(
                item, image, captured, "copy"
            )
        )
        overlay.saveRequested.connect(
            lambda image, item=overlay, captured=frame: self._complete_annotation(
                item, image, captured, "save"
            )
        )
        overlay.saveAsRequested.connect(
            lambda image, path, item=overlay, captured=frame: self._complete_annotation(
                item, image, captured, "save_as", path=path
            )
        )
        overlay.cancelled.connect(lambda item=overlay: self._cancel_annotation(item))
        overlay.stylesChanged.connect(self._queue_annotation_styles)
        overlay.fallbackRequested.connect(
            lambda model, item=overlay, captured=frame: self._fallback_annotation(
                item, model, captured
            )
        )
        overlay.destroyed.connect(lambda: self._clear_annotation_session(overlay))
        overlay.begin()

    def _complete_annotation(
        self,
        session: InlineAnnotationOverlay,
        image: object,
        frame: CapturedFrame,
        operation: str,
        *,
        path: str = "",
    ) -> None:
        if session is not self._annotation_session:
            return
        if not isinstance(image, QImage) or image.isNull():
            session.output_failed("编辑器返回了空截图。")
            return
        if isinstance(self._output_service, ScreenshotOutputService):
            self._output_async(image, frame.mode, frame.rect, operation=operation, path=path, session=session, trace_metadata=frame.metadata)
            return
        if operation == "configured":
            success = self._publish_capture(
                image,
                frame.mode,
                capture_rect=frame.rect,
                require_all_outputs=True,
            )
            if not success:
                session.output_failed(self._last_output_error or "截图输出失败。")
                return
            session.output_succeeded()
            return
        try:
            saved_path = None
            copied = False
            if operation == "copy":
                self._output_service.copy_to_clipboard(image)
                copied = True
            elif operation == "save":
                saved_path = self._output_service.save_image(
                    image,
                    self._settings,
                    mode=frame.mode,
                )
            elif operation == "save_as":
                saved_path = self._output_service.save_image_as(
                    image,
                    path,
                    self._settings,
                )
            else:
                raise ValueError(f"未知截图输出操作：{operation}")
        except Exception as exc:  # noqa: BLE001 - retain editable state for retry
            message = str(exc) or type(exc).__name__
            session.output_failed(message)
            self._capture_failed(message)
            return
        self._remember_last_region_after_success(frame.mode, frame.rect)
        if self._settings.notification:
            detail = f"已保存到 {saved_path}" if saved_path is not None else "已复制到剪贴板" if copied else "截图处理完成。"
            self.tray.showMessage("截图完成", detail)
        session.output_succeeded()

    def _cancel_annotation(self, session: InlineAnnotationOverlay) -> None:
        if session is not self._annotation_session:
            return
        self._clear_annotation_session(session)

    def _fallback_annotation(
        self,
        overlay: InlineAnnotationOverlay,
        model: object,
        frame: CapturedFrame,
    ) -> None:
        if overlay is not self._annotation_session or not isinstance(model, ScreenshotEditModel):
            return
        overlay.close()
        self._open_fallback_editor(
            model,
            frame,
            reason="显示器布局已变化。",
        )

    def _open_fallback_editor(
        self,
        model: ScreenshotEditModel,
        frame: CapturedFrame,
        *,
        reason: str,
    ) -> None:
        editor = ScreenshotEditor(frame.image, model=model, managed_output=True)
        editor.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self._annotation_session = editor
        self._annotation_frame = frame
        self._editors.append(editor)
        editor.destroyed.connect(lambda: self._clear_fallback_editor(editor))
        editor.completed.connect(
            lambda image, item=editor, captured=frame: self._complete_fallback_editor(
                item, image, captured
            )
        )
        editor.copyOutputRequested.connect(
            lambda image, item=editor, captured=frame: self._complete_fallback_output(
                item, image, captured, "copy"
            )
        )
        editor.saveAsOutputRequested.connect(
            lambda image, path, item=editor, captured=frame: self._complete_fallback_output(
                item, image, captured, "save_as", path=path
            )
        )
        editor.cancelled.connect(editor.close)
        editor.show()
        editor.raise_()
        editor.activateWindow()
        self.tray.showMessage(
            "已切换到独立标注窗口",
            f"{reason}已保留截图和全部标注。",
            QSystemTrayIcon.MessageIcon.Information,
        )

    def _complete_fallback_editor(
        self,
        editor: ScreenshotEditor,
        image: object,
        frame: CapturedFrame,
    ) -> None:
        if not isinstance(image, QImage) or image.isNull():
            self._capture_failed("编辑器返回了空截图。")
            return
        if isinstance(self._output_service, ScreenshotOutputService):
            self._output_async(image, frame.mode, frame.rect, session=editor, trace_metadata=frame.metadata)
            return
        if self._publish_capture(
            image,
            frame.mode,
            capture_rect=frame.rect,
            require_all_outputs=True,
        ):
            editor.close()

    def _complete_fallback_output(
        self,
        editor: ScreenshotEditor,
        image: object,
        frame: CapturedFrame,
        operation: str,
        *,
        path: str = "",
    ) -> None:
        if editor is not self._annotation_session:
            return
        if not isinstance(image, QImage) or image.isNull():
            self._capture_failed("编辑器返回了空截图。")
            return
        if isinstance(self._output_service, ScreenshotOutputService):
            self._output_async(image, frame.mode, frame.rect, operation=operation, path=path, session=editor, trace_metadata=frame.metadata)
            return
        try:
            if operation == "copy":
                self._output_service.copy_to_clipboard(image)
                detail = "已复制到剪贴板"
            elif operation == "save_as":
                saved = self._output_service.save_image_as(image, path, self._settings)
                detail = f"已保存到 {saved}"
            else:
                raise ValueError(f"未知截图输出操作：{operation}")
        except Exception as exc:  # noqa: BLE001 - keep fallback editor for retry
            self._capture_failed(str(exc) or type(exc).__name__)
            return
        self._remember_last_region_after_success(frame.mode, frame.rect)
        if self._settings.notification:
            self.tray.showMessage("截图完成", detail)
        editor.close()

    def _clear_fallback_editor(self, editor: ScreenshotEditor) -> None:
        self._forget_editor(editor)
        self._clear_annotation_session(editor)

    def _clear_annotation_session(self, session: QWidget) -> None:
        if self._annotation_session is session:
            self._annotation_session = None
            self._annotation_frame = None

    def _queue_annotation_styles(self, styles: object) -> None:
        if not isinstance(styles, dict):
            return
        self._pending_annotation_styles = dict(styles)
        self._annotation_style_timer.start()

    def _persist_annotation_styles(self) -> None:
        styles = self._pending_annotation_styles
        self._pending_annotation_styles = None
        if styles is None or self._settings_read_only:
            return
        try:
            self._settings = ScreenshotSettingsIO.update(
                lambda persisted: replace(
                    persisted,
                    annotation_styles=styles,
                ).normalized(),
                self._settings_path,
            )
        except (OSError, UnsupportedScreenshotSettingsVersion) as exc:
            if isinstance(exc, UnsupportedScreenshotSettingsVersion):
                self._settings_read_only = True
                self._settings_load_error = str(exc) or type(exc).__name__
            self.tray.showMessage(
                "截图设置保存失败",
                f"标注样式未能持久化：{exc}",
                QSystemTrayIcon.MessageIcon.Warning,
            )

    def _update_cu5_locator_selector(self) -> None:
        setter = getattr(self._cu5_locator, "set_selector", None)
        if callable(setter):
            setter(self._settings.cu5_selector)

    def _remember_cu5_match(self, match: object | None) -> None:
        """Persist only restart-stable CU-family features, never native handles."""

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
                f"CU 系列预览特征未能持久化：{exc}",
                QSystemTrayIcon.MessageIcon.Warning,
            )
            return
        self._settings = updated
        self._update_cu5_locator_selector()

    def _publish_capture(
        self,
        image: QImage,
        mode: CaptureMode,
        *,
        capture_rect: object | None = None,
        require_all_outputs: bool = False,
    ) -> bool:
        self._last_output_error = ""
        try:
            result: OutputResult = self._output_service.process_capture(
                image,
                self._settings,
                mode=mode,
            )
        except Exception as exc:  # noqa: BLE001 - user-facing output pipeline
            self._last_output_error = str(exc) or type(exc).__name__
            self._capture_failed(self._last_output_error)
            return False
        return self._report_output_result(result, mode, capture_rect, require_all_outputs)

    def _report_output_result(self, result: OutputResult, mode: CaptureMode, capture_rect: object, require_all_outputs: bool, *, region_sequence: int | None = None) -> bool:
        if result.errors:
            if result.saved_path is None and not result.copied_to_clipboard:
                self._last_output_error = result.failure_summary
                self._capture_failed(self._last_output_error)
                return False
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
            self._remember_last_region_after_success(mode, capture_rect, output_sequence=region_sequence)
            self._last_output_error = result.failure_summary
            return not require_all_outputs
        if result.notification_requested:
            details: list[str] = []
            if result.saved_path is not None:
                details.append(f"已保存到 {result.saved_path}")
            if result.copied_to_clipboard:
                details.append("已复制到剪贴板")
            self.tray.showMessage("截图完成", "；".join(details) or "截图处理完成。")
        self._remember_last_region_after_success(mode, capture_rect, output_sequence=region_sequence)
        return True

    def _output_async(
        self, image: QImage, mode: CaptureMode, capture_rect: object, *,
        operation: str = "configured", path: str = "", session: QWidget | None = None,
        trace_metadata: Mapping[str, object] | None = None,
    ) -> None:
        from fdm.services.screenshot_output_worker import ScreenshotOutputQueue

        if self._shutting_down:
            return
        if session is not None and any(job["session"] is session for job in self._output_jobs.values()):
            return
        self._output_sequence += 1
        settings = self._settings.normalized()
        tasks = settings.after_capture_tasks if operation == "configured" else (
            (AfterCaptureTask.COPY_CLIPBOARD,) if operation == "copy" else (AfterCaptureTask.SAVE,)
        )
        job: dict[str, object] = {
            "mode": mode, "rect": capture_rect, "session": session,
            "copied": False, "errors": [], "notification": settings.notification,
            "metadata": dict(trace_metadata or {}),
            "sequence": self._output_sequence,
        }
        if AfterCaptureTask.COPY_CLIPBOARD in tasks:
            try:
                self._output_service.copy_to_clipboard(image)
                job["copied"] = True
            except Exception as exc:
                job["errors"].append(f"复制到剪贴板失败：{exc}")
        if AfterCaptureTask.SAVE not in tasks:
            self._finish_async_output(job, OutputResult())
            return
        if self._output_queue is None:
            self._output_queue = ScreenshotOutputQueue(self, service=self._output_service)
            self._output_queue.completed.connect(self._output_completed)
            self._output_queue.failed.connect(self._output_failed)
        try:
            job_id = self._output_queue.submit(
                image, settings, mode=mode, captured_at=datetime.now().astimezone(),
                output_path=path if operation == "save_as" else None,
            )
        except Exception as exc:
            job["errors"].append(str(exc))
            self._finish_async_output(job, OutputResult())
            return
        self._output_jobs[job_id] = job
        if session is not None:
            session.setEnabled(False)
        self._trace_capture(CaptureRequest(mode, metadata=job["metadata"]), "output_queued")

    def _output_completed(self, job_id: str, result: OutputResult) -> None:
        job = self._output_jobs.pop(job_id, None)
        if job is not None:
            self._finish_async_output(job, result)

    def _output_failed(self, job_id: str, error: str) -> None:
        job = self._output_jobs.pop(job_id, None)
        if job is not None:
            job["errors"].append(error)
            self._finish_async_output(job, OutputResult())

    def _finish_async_output(self, job: dict[str, object], saved: OutputResult) -> None:
        from shiboken6 import isValid

        result = replace(saved, copied_to_clipboard=bool(job["copied"]), errors=tuple(job["errors"]), notification_requested=bool(job["notification"]))
        self._last_output_error = ""
        session = job["session"]
        success = self._report_output_result(
            result, job["mode"], job["rect"], session is not None,
            region_sequence=int(job["sequence"]),
        )
        self._trace_capture(CaptureRequest(job["mode"], metadata=job["metadata"]), "completed" if success else "output_failed", error=result.failure_summary)
        if session is None or not isValid(session) or session is not self._annotation_session:
            return
        if self._shutting_down and not success:
            self._shutdown_output_failed = True
        session.setEnabled(True)
        if isinstance(session, InlineAnnotationOverlay):
            if success:
                session.output_succeeded()
            else:
                session.output_failed(self._last_output_error or "截图输出失败。")
        else:
            session.setEnabled(True)
            if success:
                session.close()

    def _remember_last_region_after_success(
        self,
        mode: CaptureMode,
        capture_rect: object | None,
        *,
        output_sequence: int | None = None,
    ) -> None:
        from fdm.services.screenshot_capture import CaptureRect

        if mode is not CaptureMode.REGION or not isinstance(capture_rect, CaptureRect):
            return
        rect = capture_rect.normalized()
        if not rect.valid:
            return
        if output_sequence is not None:
            if output_sequence < self._last_successful_region_output_sequence:
                return
            self._last_successful_region_output_sequence = output_sequence
        self._coordinator.set_last_region(rect)
        if self._settings_read_only or rect == self._settings.last_region:
            return
        try:
            self._settings = ScreenshotSettingsIO.update(
                lambda persisted: replace(persisted, last_region=rect).normalized(),
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

    def _forget_editor(self, editor: ScreenshotEditor) -> None:
        if editor in self._editors:
            self._editors.remove(editor)

    def _capture_backend_failed(self, message: str) -> None:
        if self._active_capture_request is not None:
            self._trace_capture(self._active_capture_request, "capture_failed", error=str(message))
            self._active_capture_request = None
        self._capture_failed(message)

    def _capture_cancelled(self) -> None:
        if self._active_capture_request is not None:
            self._trace_capture(self._active_capture_request, "cancelled")
            self._active_capture_request = None

    def _capture_failed(self, message: str) -> None:
        self.tray.showMessage("截图失败", str(message), QSystemTrayIcon.MessageIcon.Warning)


def build_initial_command(arguments: argparse.Namespace) -> IPCCommand:
    if arguments.shutdown:
        return IPCCommand(CommandType.SHUTDOWN)
    if arguments.show_settings:
        return IPCCommand(CommandType.SHOW_SETTINGS)
    if arguments.capture:
        payload = {"delay_ms": max(0, int(arguments.delay_ms))}
        if arguments.editor is not None:
            payload["open_editor"] = bool(arguments.editor)
        return IPCCommand.capture(arguments.capture, payload=payload)
    return IPCCommand(CommandType.STATUS)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=SCREENSHOT_EXECUTABLE_NAME)
    parser.add_argument("--capture", choices=[mode.value for mode in CaptureMode])
    parser.add_argument("--delay-ms", type=int, default=0)
    editor_group = parser.add_mutually_exclusive_group()
    editor_group.add_argument("--editor", dest="editor", action="store_true")
    editor_group.add_argument("--no-editor", dest="editor", action="store_false")
    parser.set_defaults(editor=None)
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
        QTimer.singleShot(0, agent, lambda: agent.handle_command(command))
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
