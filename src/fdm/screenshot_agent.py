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
    should_open_annotation,
)
from fdm.screenshot_settings import (
    ScreenshotSettings,
    ScreenshotSettingsIO,
    UnsupportedScreenshotSettingsVersion,
)
from fdm.services.screenshot_output import OutputResult, ScreenshotOutputService
from fdm.ui.screenshot_editor import ScreenshotEditModel, ScreenshotEditor
from fdm.ui.screenshot_annotation_overlay import InlineAnnotationOverlay
from fdm.ui.screenshot_overlay import ScreenshotOverlay, logical_point_to_physical


SCREENSHOT_AUTOSTART_VALUE_NAME = "FiberDiameterMeasurementScreenshotTool"


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
        self._coordinator.captureFailed.connect(self._capture_failed)
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

    def close(self) -> None:
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
            payload = dict(command.payload)
            payload["mode"] = command.capture_mode.value if command.capture_mode is not None else CaptureMode.REGION.value
            request = CaptureRequest.from_mapping(payload)
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
                        mode_label = (
                            "CU 系列实时预览"
                            if mode is CaptureMode.CU5
                            else mode.value
                        )
                        errors.append(f"{mode_label} 快捷键注册失败：{exc}")
                        if failed_hotkey_modes is not None:
                            failed_hotkey_modes.add(mode)
                        retain_after_error[identifier] = mode
            existing_ids = {binding.identifier for binding in tuple(manager.bindings)}
            for identifier, (mode, binding) in desired.items():
                try:
                    manager.bind(binding)
                except Exception as exc:  # noqa: BLE001 - manager restores old binding
                    mode_label = (
                        "CU 系列实时预览"
                        if mode is CaptureMode.CU5
                        else mode.value
                    )
                    errors.append(f"{mode_label} 快捷键注册失败：{exc}")
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
        session = self._annotation_session
        if session is not None:
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
        open_editor = should_open_annotation(
            frame.mode,
            bool(frame.metadata.get("open_editor", self._settings.show_editor)),
            default=self._settings.show_editor,
        )
        if open_editor:
            self._open_inline_annotation(frame)
            return
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
            self._remember_last_region_after_success(mode, capture_rect)
            self._last_output_error = result.failure_summary
            return not require_all_outputs
        if result.notification_requested:
            details: list[str] = []
            if result.saved_path is not None:
                details.append(f"已保存到 {result.saved_path}")
            if result.copied_to_clipboard:
                details.append("已复制到剪贴板")
            self.tray.showMessage("截图完成", "；".join(details) or "截图处理完成。")
        self._remember_last_region_after_success(mode, capture_rect)
        return True

    def _remember_last_region_after_success(
        self,
        mode: CaptureMode,
        capture_rect: object | None,
    ) -> None:
        from fdm.services.screenshot_capture import CaptureRect

        if mode is not CaptureMode.REGION or not isinstance(capture_rect, CaptureRect):
            return
        rect = capture_rect.normalized()
        if not rect.valid:
            return
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
