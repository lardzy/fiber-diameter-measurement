from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
from typing import Mapping
from uuid import uuid4

from fdm.services.screenshot_capture import CaptureMode


SCREENSHOT_PROTOCOL_VERSION = 1
MAX_IPC_MESSAGE_BYTES = 1024 * 1024
SCREENSHOT_IPC_SERVER_NAME = "FiberDiameterMeasurement.Screenshot.v1"
SCREENSHOT_EXECUTABLE_NAME = "FiberScreenshotTool.exe"


class ScreenshotProtocolError(ValueError):
    """Base class for malformed or unsupported local IPC messages."""


class UnsupportedScreenshotProtocolVersion(ScreenshotProtocolError):
    pass


class CommandType(str, Enum):
    PING = "ping"
    STATUS = "status"
    CAPTURE = "capture"
    DIAGNOSE_CU5 = "diagnose_cu5"
    SHOW_SETTINGS = "show_settings"
    UPDATE_SETTINGS = "update_settings"
    SHUTDOWN = "shutdown"

    @classmethod
    def parse(cls, value: object) -> "CommandType":
        if isinstance(value, cls):
            return value
        token = str(value or "").strip().lower().replace("-", "_")
        aliases = {
            "settings": cls.SHOW_SETTINGS,
            "quit": cls.SHUTDOWN,
            "detect_cu5": cls.DIAGNOSE_CU5,
        }
        if token in aliases:
            return aliases[token]
        try:
            return cls(token)
        except ValueError as exc:
            raise ScreenshotProtocolError(f"unknown screenshot command: {value!r}") from exc


ScreenshotCommandType = CommandType


@dataclass(frozen=True, slots=True)
class IPCCommand:
    command: CommandType
    request_id: str = field(default_factory=lambda: uuid4().hex)
    capture_mode: CaptureMode | None = None
    payload: dict[str, object] = field(default_factory=dict)
    protocol_version: int = SCREENSHOT_PROTOCOL_VERSION

    def __post_init__(self) -> None:
        command = CommandType.parse(self.command)
        request_id = _normalize_request_id(self.request_id)
        version = _validate_protocol_version(self.protocol_version)
        mode = _parse_capture_mode(self.capture_mode)
        if command is CommandType.CAPTURE and mode is None:
            raise ScreenshotProtocolError("capture command requires capture_mode")
        payload = _strict_json_object(self.payload, label="command payload")
        object.__setattr__(self, "command", command)
        object.__setattr__(self, "request_id", request_id)
        object.__setattr__(self, "capture_mode", mode)
        object.__setattr__(self, "payload", payload)
        object.__setattr__(self, "protocol_version", version)

    @classmethod
    def capture(
        cls,
        mode: CaptureMode | str,
        *,
        payload: Mapping[str, object] | None = None,
        request_id: str | None = None,
    ) -> "IPCCommand":
        kwargs: dict[str, object] = {
            "command": CommandType.CAPTURE,
            "capture_mode": mode,
            "payload": dict(payload or {}),
        }
        if request_id is not None:
            kwargs["request_id"] = request_id
        return cls(**kwargs)  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "message_type": "command",
            "protocol_version": self.protocol_version,
            "request_id": self.request_id,
            "command": self.command.value,
            "payload": self.payload,
        }
        if self.capture_mode is not None:
            result["capture_mode"] = self.capture_mode.value
        return result

    @classmethod
    def from_dict(cls, value: object) -> "IPCCommand":
        if not isinstance(value, dict):
            raise ScreenshotProtocolError("IPC command must be a JSON object")
        message_type = str(value.get("message_type", "command") or "command")
        if message_type != "command":
            raise ScreenshotProtocolError("JSON object is not an IPC command")
        return cls(
            command=CommandType.parse(value.get("command")),
            request_id=value.get("request_id", ""),
            capture_mode=value.get("capture_mode"),
            payload=value.get("payload", {}),
            protocol_version=value.get("protocol_version", SCREENSHOT_PROTOCOL_VERSION),
        )


@dataclass(frozen=True, slots=True)
class IPCResponse:
    request_id: str
    ok: bool
    result: dict[str, object] = field(default_factory=dict)
    error: str = ""
    protocol_version: int = SCREENSHOT_PROTOCOL_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.ok, bool):
            raise ScreenshotProtocolError("IPC response 'ok' must be a boolean")
        request_id = _normalize_request_id(self.request_id)
        version = _validate_protocol_version(self.protocol_version)
        result = _strict_json_object(self.result, label="response result")
        error = str(self.error or "").strip()
        if self.ok and error:
            raise ScreenshotProtocolError("successful IPC response cannot contain an error")
        if not self.ok and not error:
            raise ScreenshotProtocolError("failed IPC response requires an error message")
        object.__setattr__(self, "request_id", request_id)
        object.__setattr__(self, "ok", bool(self.ok))
        object.__setattr__(self, "result", result)
        object.__setattr__(self, "error", error)
        object.__setattr__(self, "protocol_version", version)

    @classmethod
    def success(
        cls,
        request_id: str,
        result: Mapping[str, object] | None = None,
    ) -> "IPCResponse":
        return cls(request_id=request_id, ok=True, result=dict(result or {}))

    @classmethod
    def failure(cls, request_id: str, error: object) -> "IPCResponse":
        return cls(request_id=request_id, ok=False, error=str(error or "unknown error"))

    def to_dict(self) -> dict[str, object]:
        return {
            "message_type": "response",
            "protocol_version": self.protocol_version,
            "request_id": self.request_id,
            "ok": self.ok,
            "result": self.result,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, value: object) -> "IPCResponse":
        if not isinstance(value, dict):
            raise ScreenshotProtocolError("IPC response must be a JSON object")
        if value.get("message_type", "response") != "response":
            raise ScreenshotProtocolError("JSON object is not an IPC response")
        raw_ok = value.get("ok")
        if not isinstance(raw_ok, bool):
            raise ScreenshotProtocolError("IPC response 'ok' must be a boolean")
        return cls(
            request_id=value.get("request_id", ""),
            ok=raw_ok,
            result=value.get("result", {}),
            error=value.get("error", ""),
            protocol_version=value.get("protocol_version", SCREENSHOT_PROTOCOL_VERSION),
        )


IPCMessage = IPCCommand | IPCResponse


def screenshot_ipc_server_name() -> str:
    """Return the stable QLocalServer/Windows named-pipe endpoint name."""

    return SCREENSHOT_IPC_SERVER_NAME


def encode_ipc_message(message: IPCMessage) -> bytes:
    if not isinstance(message, (IPCCommand, IPCResponse)):
        raise TypeError("message must be IPCCommand or IPCResponse")
    try:
        encoded = json.dumps(
            message.to_dict(),
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8") + b"\n"
    except (TypeError, ValueError) as exc:
        raise ScreenshotProtocolError("IPC message is not strict JSON") from exc
    if len(encoded) > MAX_IPC_MESSAGE_BYTES:
        raise ScreenshotProtocolError("IPC message exceeds the size limit")
    return encoded


def decode_ipc_message(data: bytes | bytearray | memoryview | str) -> IPCMessage:
    payload = decode_json_line(data)
    message_type = payload.get("message_type")
    if message_type == "command":
        return IPCCommand.from_dict(payload)
    if message_type == "response":
        return IPCResponse.from_dict(payload)
    raise ScreenshotProtocolError("IPC message_type must be 'command' or 'response'")


def decode_json_line(data: bytes | bytearray | memoryview | str) -> dict[str, object]:
    if isinstance(data, str):
        raw = data.encode("utf-8")
    else:
        raw = bytes(data)
    if not raw:
        raise ScreenshotProtocolError("empty IPC message")
    if len(raw) > MAX_IPC_MESSAGE_BYTES:
        raise ScreenshotProtocolError("IPC message exceeds the size limit")
    raw = raw[:-2] if raw.endswith(b"\r\n") else raw[:-1] if raw.endswith(b"\n") else raw
    if b"\n" in raw or b"\r" in raw:
        raise ScreenshotProtocolError("expected exactly one JSON line")
    try:
        text = raw.decode("utf-8")
        payload = json.loads(text, parse_constant=_reject_non_finite_json_constant)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ScreenshotProtocolError("invalid IPC JSON line") from exc
    if not isinstance(payload, dict):
        raise ScreenshotProtocolError("IPC JSON value must be an object")
    return payload


def decode_command(data: bytes | bytearray | memoryview | str) -> IPCCommand:
    message = decode_ipc_message(data)
    if not isinstance(message, IPCCommand):
        raise ScreenshotProtocolError("expected an IPC command")
    return message


def decode_response(data: bytes | bytearray | memoryview | str) -> IPCResponse:
    message = decode_ipc_message(data)
    if not isinstance(message, IPCResponse):
        raise ScreenshotProtocolError("expected an IPC response")
    return message


encode_message = encode_ipc_message
decode_message = decode_ipc_message


def _parse_capture_mode(value: object) -> CaptureMode | None:
    if value is None or value == "":
        return None
    if isinstance(value, CaptureMode):
        return value
    try:
        return CaptureMode.parse(value)
    except ValueError as exc:
        raise ScreenshotProtocolError(f"unknown capture mode: {value!r}") from exc


def _normalize_request_id(value: object) -> str:
    request_id = str(value or "").strip()
    if not request_id or len(request_id) > 128 or any(char in request_id for char in "\r\n"):
        raise ScreenshotProtocolError("request_id must contain 1 to 128 single-line characters")
    return request_id


def _validate_protocol_version(value: object) -> int:
    try:
        if isinstance(value, bool):
            raise ValueError
        version = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ScreenshotProtocolError("invalid screenshot protocol version") from exc
    if version != SCREENSHOT_PROTOCOL_VERSION:
        raise UnsupportedScreenshotProtocolVersion(
            f"unsupported screenshot protocol version: {version}"
        )
    return version


def _strict_json_object(value: object, *, label: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ScreenshotProtocolError(f"{label} must be a JSON object")
    try:
        serialized = json.dumps(
            dict(value),
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
        result = json.loads(serialized, parse_constant=_reject_non_finite_json_constant)
    except (TypeError, ValueError) as exc:
        raise ScreenshotProtocolError(f"{label} is not strict JSON") from exc
    if not isinstance(result, dict):
        raise ScreenshotProtocolError(f"{label} must be a JSON object")
    return result


def _reject_non_finite_json_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON constant is not allowed: {value}")


__all__ = [
    "CommandType",
    "IPCCommand",
    "IPCMessage",
    "IPCResponse",
    "MAX_IPC_MESSAGE_BYTES",
    "SCREENSHOT_EXECUTABLE_NAME",
    "SCREENSHOT_IPC_SERVER_NAME",
    "SCREENSHOT_PROTOCOL_VERSION",
    "ScreenshotCommandType",
    "ScreenshotProtocolError",
    "UnsupportedScreenshotProtocolVersion",
    "decode_command",
    "decode_ipc_message",
    "decode_json_line",
    "decode_message",
    "decode_response",
    "encode_ipc_message",
    "encode_message",
    "screenshot_ipc_server_name",
]
