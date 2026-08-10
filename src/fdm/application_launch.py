from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Iterable
import uuid

from PySide6.QtCore import QLockFile, QObject, QStandardPaths, Signal
from PySide6.QtNetwork import QLocalServer, QLocalSocket


ASSOCIATED_FILE_SUFFIXES = frozenset({".fdmproj", ".fdmslide"})
MAX_OPEN_PATHS = 32
MAX_IPC_MESSAGE_BYTES = 64 * 1024
IPC_PROTOCOL_VERSION = 1


class ApplicationOpenRequestError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ApplicationOpenRequest:
    request_id: str
    paths: tuple[Path, ...]
    source: str
    activate: bool = True


@dataclass(frozen=True, slots=True)
class InstanceStartResult:
    primary: bool
    forwarded: bool = False
    error: str = ""


def _path_key(path: Path) -> str:
    return os.path.normcase(str(path)).casefold()


def build_application_open_request(
    paths: Iterable[str | Path],
    *,
    source: str,
    cwd: str | Path | None = None,
    request_id: str | None = None,
    require_absolute: bool = False,
) -> ApplicationOpenRequest:
    base_dir = Path(cwd) if cwd is not None else Path.cwd()
    normalized: list[Path] = []
    seen: set[str] = set()
    for raw_path in paths:
        token = str(raw_path)
        if not token or "\x00" in token:
            raise ApplicationOpenRequestError("文件路径为空或包含非法字符。")
        candidate = Path(token).expanduser()
        if require_absolute and not candidate.is_absolute():
            raise ApplicationOpenRequestError(f"IPC 文件路径必须是绝对路径：{token}")
        if not candidate.is_absolute():
            candidate = base_dir / candidate
        candidate = candidate.resolve(strict=False)
        if candidate.suffix.lower() not in ASSOCIATED_FILE_SUFFIXES:
            raise ApplicationOpenRequestError(f"不支持的关联文件类型：{candidate.name}")
        key = _path_key(candidate)
        if key in seen:
            continue
        seen.add(key)
        normalized.append(candidate)

    if len(normalized) > MAX_OPEN_PATHS:
        raise ApplicationOpenRequestError(f"一次最多打开 {MAX_OPEN_PATHS} 个关联文件。")
    project_paths = [path for path in normalized if path.suffix.lower() == ".fdmproj"]
    slide_paths = [path for path in normalized if path.suffix.lower() == ".fdmslide"]
    if len(project_paths) > 1:
        raise ApplicationOpenRequestError("一次只能打开一个项目文件。")
    if project_paths and slide_paths:
        raise ApplicationOpenRequestError("项目文件和数字化切片不能在同一请求中混合打开。")

    source_token = str(source).strip()
    if not source_token or len(source_token) > 64:
        raise ApplicationOpenRequestError("打开请求来源无效。")
    request_token = str(request_id or uuid.uuid4().hex).strip()
    if not request_token or len(request_token) > 64:
        raise ApplicationOpenRequestError("打开请求 ID 无效。")
    return ApplicationOpenRequest(
        request_id=request_token,
        paths=tuple(normalized),
        source=source_token,
    )


def parse_application_arguments(
    argv: Iterable[str],
    *,
    cwd: str | Path | None = None,
) -> tuple[list[str], ApplicationOpenRequest]:
    args = [str(item) for item in argv]
    if not args:
        args = ["fdm"]
    qt_args = [args[0]]
    associated_paths: list[str] = []
    for token in args[1:]:
        if Path(token).suffix.lower() in ASSOCIATED_FILE_SUFFIXES:
            associated_paths.append(token)
        else:
            qt_args.append(token)
    request = build_application_open_request(
        associated_paths,
        source="command_line",
        cwd=cwd,
    )
    return qt_args, request


def encode_application_open_request(request: ApplicationOpenRequest) -> bytes:
    payload = {
        "version": IPC_PROTOCOL_VERSION,
        "request_id": request.request_id,
        "source": request.source,
        "activate": bool(request.activate),
        "paths": [str(path) for path in request.paths],
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    if len(encoded) > MAX_IPC_MESSAGE_BYTES:
        raise ApplicationOpenRequestError("关联文件请求超过本机通信大小限制。")
    return encoded


def decode_application_open_request(payload: bytes) -> ApplicationOpenRequest:
    if not payload or len(payload) > MAX_IPC_MESSAGE_BYTES:
        raise ApplicationOpenRequestError("本机通信消息为空或超过大小限制。")
    try:
        decoded = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ApplicationOpenRequestError(f"本机通信消息不是有效的 UTF-8 JSON：{exc}") from exc
    if not isinstance(decoded, dict):
        raise ApplicationOpenRequestError("本机通信消息根节点必须是对象。")
    allowed_keys = {"version", "request_id", "source", "activate", "paths"}
    unknown_keys = set(decoded) - allowed_keys
    if unknown_keys:
        raise ApplicationOpenRequestError(
            "本机通信消息包含未知字段：" + ", ".join(sorted(str(key) for key in unknown_keys))
        )
    if decoded.get("version") != IPC_PROTOCOL_VERSION:
        raise ApplicationOpenRequestError("本机通信协议版本不受支持。")
    if decoded.get("activate") is not True:
        raise ApplicationOpenRequestError("本机通信激活标记无效。")
    request_id = decoded.get("request_id")
    source = decoded.get("source")
    if not isinstance(request_id, str) or not isinstance(source, str):
        raise ApplicationOpenRequestError("本机通信请求 ID 和来源必须是字符串。")
    raw_paths = decoded.get("paths")
    if not isinstance(raw_paths, list) or any(not isinstance(item, str) for item in raw_paths):
        raise ApplicationOpenRequestError("本机通信路径字段必须是字符串列表。")
    return build_application_open_request(
        raw_paths,
        source=source,
        request_id=request_id,
        require_absolute=True,
    )


def single_instance_server_name(
    *,
    app_data_directory: str | Path,
    executable_path: str | Path,
) -> str:
    identity = "|".join(
        (
            str(Path(app_data_directory).expanduser().resolve(strict=False)).casefold(),
            str(Path(executable_path).expanduser().resolve(strict=False).parent).casefold(),
        )
    )
    return "fdm-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]


class SingleInstanceCoordinator(QObject):
    requestReceived = Signal(object)
    protocolError = Signal(str)

    def __init__(
        self,
        server_name: str,
        parent: QObject | None = None,
        *,
        lock_file_path: str | Path | None = None,
    ) -> None:
        super().__init__(parent)
        self._server_name = str(server_name)
        if lock_file_path is None:
            temporary_directory = QStandardPaths.writableLocation(
                QStandardPaths.StandardLocation.TempLocation
            ) or str(Path.home() / ".fdm")
            lock_file_path = Path(temporary_directory) / "fdm-instance-locks" / (
                f"{self._server_name}.lock"
            )
        self._lock_file_path = Path(lock_file_path).expanduser().resolve(strict=False)
        self._instance_lock = QLockFile(str(self._lock_file_path))
        self._server = QLocalServer(self)
        self._server.setSocketOptions(QLocalServer.SocketOption.UserAccessOption)
        self._server.newConnection.connect(self._accept_pending_connections)
        self._connections: dict[int, tuple[QLocalSocket, bytearray]] = {}

    @classmethod
    def for_current_application(cls, parent: QObject | None = None) -> SingleInstanceCoordinator:
        app_data_location = QStandardPaths.writableLocation(
            QStandardPaths.StandardLocation.AppLocalDataLocation
        )
        app_data_directory = Path(app_data_location) if app_data_location else Path.home() / ".fdm"
        executable_identity = (
            Path(sys.executable)
            if getattr(sys, "frozen", False)
            else Path(__file__).resolve().parents[2] / "fdm-development"
        )
        server_name = single_instance_server_name(
            app_data_directory=app_data_directory,
            executable_path=executable_identity,
        )
        return cls(
            server_name,
            parent,
            lock_file_path=app_data_directory / "instance" / f"{server_name}.lock",
        )

    @property
    def server_name(self) -> str:
        return self._server_name

    def start_or_forward(
        self,
        request: ApplicationOpenRequest,
        *,
        timeout_ms: int = 2_000,
    ) -> InstanceStartResult:
        first_timeout = min(250, max(1, int(timeout_ms)))
        if self._forward_request(request, timeout_ms=first_timeout):
            return InstanceStartResult(primary=False, forwarded=True)

        try:
            self._lock_file_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            return InstanceStartResult(
                primary=False,
                error=f"无法创建单实例锁目录：{self._lock_file_path.parent}\n\n{exc}",
            )
        owns_instance_lock = self._instance_lock.tryLock(0)
        listen_error = "另一个软件进程持有单实例锁。"
        if owns_instance_lock:
            if self._server.listen(self._server_name):
                return InstanceStartResult(primary=True)
            listen_error = self._server.errorString()
            self._instance_lock.unlock()

        remaining = max(1, int(timeout_ms) - first_timeout)
        attempts = 4
        per_attempt = max(1, remaining // attempts)
        for _attempt in range(attempts):
            if self._forward_request(request, timeout_ms=per_attempt):
                return InstanceStartResult(primary=False, forwarded=True)
        return InstanceStartResult(
            primary=False,
            error=(
                "无法连接到已运行的软件实例，也无法建立本机通信服务。\n\n"
                f"服务名：{self._server_name}\n"
                f"错误：{listen_error}"
            ),
        )

    def close(self) -> None:
        try:
            self._server.newConnection.disconnect(self._accept_pending_connections)
        except (RuntimeError, TypeError):
            pass
        for socket, _buffer in list(self._connections.values()):
            self._disconnect_connection_signals(socket)
            socket.abort()
        self._connections.clear()
        self._server.close()
        if self._instance_lock.isLocked():
            self._instance_lock.unlock()

    def _forward_request(self, request: ApplicationOpenRequest, *, timeout_ms: int) -> bool:
        socket = QLocalSocket()
        try:
            socket.connectToServer(self._server_name)
            if not socket.waitForConnected(max(1, int(timeout_ms))):
                return False
            payload = encode_application_open_request(request)
            written = socket.write(payload)
            if written != len(payload):
                return False
            if socket.bytesToWrite() > 0 and not socket.waitForBytesWritten(max(1, int(timeout_ms))):
                return False
            return True
        finally:
            # This is a synchronous one-shot transport.  Do not leave a
            # disconnect notifier queued after the temporary Python wrapper
            # has gone out of scope; on Qt's local-socket backends that stale
            # notifier can otherwise surface much later in an unrelated event
            # loop turn.
            socket.abort()

    def _accept_pending_connections(self) -> None:
        while self._server.hasPendingConnections():
            socket = self._server.nextPendingConnection()
            if socket is None:
                continue
            key = id(socket)
            self._connections[key] = (socket, bytearray())
            socket.readyRead.connect(lambda key=key: self._consume_connection(key))
            socket.disconnected.connect(lambda key=key: self._discard_connection(key))
            self._consume_connection(key)

    def _consume_connection(self, key: int) -> None:
        connection = self._connections.get(key)
        if connection is None:
            return
        socket, buffer = connection
        buffer.extend(bytes(socket.readAll()))
        while True:
            newline_index = buffer.find(b"\n")
            if newline_index < 0:
                break
            if newline_index + 1 > MAX_IPC_MESSAGE_BYTES:
                self.protocolError.emit("本机通信消息超过 64KiB，已拒绝。")
                self._discard_connection(key)
                return
            raw_message = bytes(buffer[:newline_index])
            del buffer[: newline_index + 1]
            if not raw_message:
                continue
            try:
                request = decode_application_open_request(raw_message)
            except ApplicationOpenRequestError as exc:
                self.protocolError.emit(str(exc))
                continue
            self.requestReceived.emit(
                ApplicationOpenRequest(
                    request_id=request.request_id,
                    paths=request.paths,
                    source="ipc",
                    activate=request.activate,
                )
            )
        if len(buffer) > MAX_IPC_MESSAGE_BYTES:
            self.protocolError.emit("本机通信消息超过 64KiB，已拒绝。")
            self._discard_connection(key)

    def _discard_connection(self, key: int) -> None:
        connection = self._connections.pop(key, None)
        if connection is None:
            return
        socket, _buffer = connection
        self._disconnect_connection_signals(socket)
        socket.abort()
        socket.deleteLater()

    @staticmethod
    def _disconnect_connection_signals(socket: QLocalSocket) -> None:
        for signal in (socket.readyRead, socket.disconnected):
            try:
                signal.disconnect()
            except (RuntimeError, TypeError):
                pass
