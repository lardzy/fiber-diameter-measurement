from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import sys
import time
from typing import Callable, Mapping, Protocol

from fdm.screenshot_protocol import (
    CommandType,
    IPCCommand,
    IPCResponse,
    MAX_IPC_MESSAGE_BYTES,
    SCREENSHOT_EXECUTABLE_NAME,
    ScreenshotProtocolError,
    decode_response,
    encode_ipc_message,
    screenshot_ipc_server_name,
)
from fdm.services.screenshot_capture import CaptureMode


class ScreenshotAgentClientError(RuntimeError):
    pass


class ScreenshotAgentUnavailableError(ScreenshotAgentClientError):
    pass


class ScreenshotAgentTimeoutError(ScreenshotAgentClientError):
    pass


class ScreenshotAgentProtocolError(ScreenshotAgentClientError):
    pass


class ScreenshotAgentCommandError(ScreenshotAgentClientError):
    def __init__(self, message: str, *, response: IPCResponse) -> None:
        super().__init__(message)
        self.response = response


class ScreenshotAgentLaunchError(ScreenshotAgentClientError):
    pass


@dataclass(frozen=True, slots=True)
class ScreenshotAgentLaunchSpec:
    program: str
    arguments: tuple[str, ...]
    working_directory: str
    packaged: bool


@dataclass(frozen=True, slots=True)
class ScreenshotAgentLaunchResult:
    started: bool
    pid: int | None = None
    error: str = ""


@dataclass(frozen=True, slots=True)
class ScreenshotAgentStatus:
    running: bool
    result: Mapping[str, object] = field(default_factory=dict)
    error: str = ""


class ScreenshotAgentTransport(Protocol):
    def send(self, command: IPCCommand, *, timeout_ms: int) -> IPCResponse: ...


class ScreenshotAgentLauncher(Protocol):
    def start_detached(
        self,
        launch_spec: ScreenshotAgentLaunchSpec,
    ) -> ScreenshotAgentLaunchResult: ...


def resolve_screenshot_agent_launch_spec(
    *,
    executable: str | Path | None = None,
    frozen: bool | None = None,
) -> ScreenshotAgentLaunchSpec:
    executable_path = Path(executable or sys.executable).expanduser().resolve(strict=False)
    is_packaged = bool(getattr(sys, "frozen", False)) if frozen is None else bool(frozen)
    if is_packaged:
        tool_path = executable_path.parent / SCREENSHOT_EXECUTABLE_NAME
        return ScreenshotAgentLaunchSpec(
            program=str(tool_path),
            arguments=(),
            working_directory=str(tool_path.parent),
            packaged=True,
        )
    return ScreenshotAgentLaunchSpec(
        program=str(executable_path),
        arguments=("-m", "fdm.screenshot_agent"),
        working_directory=str(Path.cwd()),
        packaged=False,
    )


class QLocalSocketScreenshotTransport:
    """One-command-per-connection synchronous QLocalSocket transport."""

    def __init__(self, server_name: str | None = None) -> None:
        self.server_name = str(server_name or screenshot_ipc_server_name())

    def send(self, command: IPCCommand, *, timeout_ms: int) -> IPCResponse:
        from PySide6.QtNetwork import QLocalSocket

        timeout = max(1, int(timeout_ms))
        deadline = time.monotonic() + (timeout / 1000.0)
        socket = QLocalSocket()

        def remaining_ms() -> int:
            return max(1, int(round((deadline - time.monotonic()) * 1000.0)))

        try:
            socket.connectToServer(self.server_name)
            if not socket.waitForConnected(remaining_ms()):
                detail = socket.errorString().strip()
                suffix = f"：{detail}" if detail else ""
                raise ScreenshotAgentUnavailableError(
                    f"截图工具尚未运行或本机通信不可用{suffix}"
                )
            encoded = encode_ipc_message(command)
            written = int(socket.write(encoded))
            if written != len(encoded):
                raise ScreenshotAgentUnavailableError("截图工具本机通信写入失败。")
            if socket.bytesToWrite() and not socket.waitForBytesWritten(remaining_ms()):
                raise ScreenshotAgentTimeoutError("等待截图工具接收命令超时。")

            buffer = bytearray()
            while b"\n" not in buffer:
                if time.monotonic() >= deadline:
                    raise ScreenshotAgentTimeoutError("等待截图工具响应超时。")
                if socket.bytesAvailable() <= 0 and not socket.waitForReadyRead(remaining_ms()):
                    raise ScreenshotAgentTimeoutError("等待截图工具响应超时。")
                buffer.extend(bytes(socket.readAll()))
                if len(buffer) > MAX_IPC_MESSAGE_BYTES:
                    raise ScreenshotAgentProtocolError("截图工具响应超过协议大小限制。")
            line, _separator, _remainder = buffer.partition(b"\n")
            try:
                response = decode_response(line + b"\n")
            except ScreenshotProtocolError as exc:
                raise ScreenshotAgentProtocolError(
                    f"截图工具返回了无效响应：{exc}"
                ) from exc
            if response.request_id != command.request_id:
                raise ScreenshotAgentProtocolError(
                    "截图工具响应 request_id 与请求不一致。"
                )
            return response
        finally:
            socket.abort()


class QtDetachedScreenshotAgentLauncher:
    def start_detached(
        self,
        launch_spec: ScreenshotAgentLaunchSpec,
    ) -> ScreenshotAgentLaunchResult:
        from PySide6.QtCore import QProcess

        try:
            result = QProcess.startDetached(
                launch_spec.program,
                list(launch_spec.arguments),
                launch_spec.working_directory,
            )
        except Exception as exc:  # noqa: BLE001 - normalize Qt binding errors
            return ScreenshotAgentLaunchResult(False, error=str(exc))
        if isinstance(result, tuple):
            started = bool(result[0])
            pid = int(result[1]) if started and len(result) > 1 and result[1] else None
        else:  # pragma: no cover - compatibility with alternate PySide overloads
            started = bool(result)
            pid = None
        return ScreenshotAgentLaunchResult(
            started=started,
            pid=pid,
            error="" if started else "QProcess.startDetached 返回失败。",
        )


class ScreenshotAgentClient:
    """Synchronous main-application client for the detached screenshot tool.

    The client deliberately has no process ownership or destructor shutdown.
    Closing the measurement window therefore never stops the resident tool.
    """

    def __init__(
        self,
        *,
        transport: ScreenshotAgentTransport | None = None,
        launcher: ScreenshotAgentLauncher | None = None,
        launch_spec: ScreenshotAgentLaunchSpec | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._transport = transport or QLocalSocketScreenshotTransport()
        self._launcher = launcher or QtDetachedScreenshotAgentLauncher()
        self._launch_spec = launch_spec or resolve_screenshot_agent_launch_spec()
        self._monotonic = monotonic
        self._sleep = sleep

    @property
    def launch_spec(self) -> ScreenshotAgentLaunchSpec:
        return self._launch_spec

    def send(
        self,
        command: IPCCommand,
        *,
        timeout_ms: int = 1_000,
    ) -> IPCResponse:
        response = self._transport.send(command, timeout_ms=max(1, int(timeout_ms)))
        if response.request_id != command.request_id:
            raise ScreenshotAgentProtocolError(
                "截图工具响应 request_id 与请求不一致。"
            )
        if not response.ok:
            raise ScreenshotAgentCommandError(
                response.error or "截图工具拒绝了命令。",
                response=response,
            )
        return response

    def ping(self, *, timeout_ms: int = 300) -> bool:
        try:
            self.send(
                IPCCommand(command=CommandType.PING),
                timeout_ms=timeout_ms,
            )
        except ScreenshotAgentClientError:
            return False
        return True

    def status(self, *, timeout_ms: int = 500) -> ScreenshotAgentStatus:
        try:
            response = self.send(
                IPCCommand(command=CommandType.STATUS),
                timeout_ms=timeout_ms,
            )
        except ScreenshotAgentCommandError as exc:
            # A command-level rejection still proves the agent endpoint is up.
            return ScreenshotAgentStatus(True, error=str(exc))
        except ScreenshotAgentClientError as exc:
            return ScreenshotAgentStatus(False, error=str(exc))
        return ScreenshotAgentStatus(True, dict(response.result))

    def ensure_started(
        self,
        *,
        timeout_ms: int = 4_000,
        retry_interval_ms: int = 100,
        max_attempts: int = 30,
        probe_timeout_ms: int = 250,
    ) -> ScreenshotAgentStatus:
        total_timeout = max(1, int(timeout_ms))
        attempts = max(1, int(max_attempts))
        retry_interval = max(1, int(retry_interval_ms))
        probe_timeout = max(1, int(probe_timeout_ms))
        deadline = self._monotonic() + (total_timeout / 1000.0)

        if self.ping(timeout_ms=min(probe_timeout, total_timeout)):
            return ScreenshotAgentStatus(True, {"already_running": True})

        program_path = Path(self._launch_spec.program)
        if not program_path.is_file():
            raise ScreenshotAgentLaunchError(
                f"未找到截图工具程序：{program_path}"
            )
        launch_result = self._launcher.start_detached(self._launch_spec)
        if not launch_result.started:
            # A concurrent starter may have won after the first probe.
            if self.ping(timeout_ms=min(probe_timeout, total_timeout)):
                return ScreenshotAgentStatus(True, {"already_running": True})
            detail = launch_result.error.strip()
            suffix = f"：{detail}" if detail else ""
            raise ScreenshotAgentLaunchError(f"无法启动截图工具{suffix}")

        last_error = ""
        for attempt in range(attempts):
            remaining_ms = int(round((deadline - self._monotonic()) * 1000.0))
            if remaining_ms <= 0:
                break
            try:
                response = self.send(
                    IPCCommand(command=CommandType.PING),
                    timeout_ms=min(probe_timeout, remaining_ms),
                )
            except ScreenshotAgentClientError as exc:
                last_error = str(exc)
            else:
                result = dict(response.result)
                result.update(started=True, pid=launch_result.pid)
                return ScreenshotAgentStatus(True, result)
            if attempt + 1 >= attempts:
                break
            remaining_seconds = deadline - self._monotonic()
            if remaining_seconds <= 0:
                break
            self._sleep(min(retry_interval / 1000.0, remaining_seconds))

        suffix = f"；最后错误：{last_error}" if last_error else ""
        raise ScreenshotAgentTimeoutError(
            f"截图工具已启动，但在 {total_timeout} ms 内未就绪{suffix}"
        )

    def capture(
        self,
        mode: CaptureMode | str,
        *,
        payload: Mapping[str, object] | None = None,
        timeout_ms: int = 1_000,
    ) -> IPCResponse:
        return self.send(
            IPCCommand.capture(mode, payload=payload),
            timeout_ms=timeout_ms,
        )

    def show_settings(self, *, timeout_ms: int = 1_000) -> IPCResponse:
        return self.send(
            IPCCommand(command=CommandType.SHOW_SETTINGS),
            timeout_ms=timeout_ms,
        )

    def update_settings(
        self,
        settings: Mapping[str, object],
        *,
        timeout_ms: int = 1_000,
    ) -> IPCResponse:
        return self.send(
            IPCCommand(
                command=CommandType.UPDATE_SETTINGS,
                payload=dict(settings),
            ),
            timeout_ms=timeout_ms,
        )

    def shutdown(self, *, timeout_ms: int = 1_000) -> bool:
        try:
            self.send(
                IPCCommand(command=CommandType.SHUTDOWN),
                timeout_ms=timeout_ms,
            )
        except ScreenshotAgentUnavailableError:
            return False
        return True


__all__ = [
    "QLocalSocketScreenshotTransport",
    "QtDetachedScreenshotAgentLauncher",
    "ScreenshotAgentClient",
    "ScreenshotAgentClientError",
    "ScreenshotAgentCommandError",
    "ScreenshotAgentLaunchError",
    "ScreenshotAgentLauncher",
    "ScreenshotAgentLaunchResult",
    "ScreenshotAgentLaunchSpec",
    "ScreenshotAgentProtocolError",
    "ScreenshotAgentStatus",
    "ScreenshotAgentTimeoutError",
    "ScreenshotAgentTransport",
    "ScreenshotAgentUnavailableError",
    "resolve_screenshot_agent_launch_spec",
]
