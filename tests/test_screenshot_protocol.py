from __future__ import annotations

import pytest

from fdm.screenshot_protocol import (
    CommandType,
    IPCCommand,
    IPCResponse,
    MAX_IPC_MESSAGE_BYTES,
    SCREENSHOT_EXECUTABLE_NAME,
    SCREENSHOT_IPC_SERVER_NAME,
    ScreenshotProtocolError,
    UnsupportedScreenshotProtocolVersion,
    decode_command,
    decode_ipc_message,
    decode_response,
    encode_ipc_message,
    screenshot_ipc_server_name,
)
from fdm.services.screenshot_capture import CaptureMode


def test_capture_command_json_line_round_trip() -> None:
    command = IPCCommand.capture(
        CaptureMode.CU5,
        request_id="request-一",
        payload={"delay_ms": 100, "diagnostics": True},
    )

    encoded = encode_ipc_message(command)
    decoded = decode_command(encoded)

    assert encoded.endswith(b"\n")
    assert decoded == command
    assert decoded.command is CommandType.CAPTURE
    assert decoded.capture_mode is CaptureMode.CU5


def test_response_factories_and_round_trip() -> None:
    success = IPCResponse.success("abc", {"saved_path": "测试.png"})
    failure = IPCResponse.failure("def", "capture failed")

    assert decode_response(encode_ipc_message(success)) == success
    assert decode_ipc_message(encode_ipc_message(failure)) == failure
    with pytest.raises(ScreenshotProtocolError, match="must be a boolean"):
        IPCResponse("abc", ok=1)  # type: ignore[arg-type]


def test_protocol_rejects_invalid_commands_versions_and_json() -> None:
    with pytest.raises(ScreenshotProtocolError, match="requires capture_mode"):
        IPCCommand(CommandType.CAPTURE)
    with pytest.raises(ScreenshotProtocolError, match="strict JSON"):
        IPCCommand(CommandType.STATUS, payload={"value": float("nan")})
    with pytest.raises(UnsupportedScreenshotProtocolVersion):
        IPCCommand.from_dict(
            {
                "message_type": "command",
                "protocol_version": 2,
                "request_id": "abc",
                "command": "ping",
            }
        )
    with pytest.raises(ScreenshotProtocolError, match="exactly one"):
        decode_ipc_message(b'{"message_type":"command"}\n{}\n')
    with pytest.raises(ScreenshotProtocolError, match="size limit"):
        decode_ipc_message(b"x" * (MAX_IPC_MESSAGE_BYTES + 1))


def test_companion_endpoint_and_executable_names_are_stable() -> None:
    assert screenshot_ipc_server_name() == SCREENSHOT_IPC_SERVER_NAME
    assert SCREENSHOT_IPC_SERVER_NAME == "FiberDiameterMeasurement.Screenshot.v1"
    assert SCREENSHOT_EXECUTABLE_NAME == "FiberScreenshotTool.exe"


def test_cu5_diagnostic_command_and_legacy_alias_are_stable() -> None:
    command = IPCCommand(CommandType.DIAGNOSE_CU5, payload={"include_reasons": True})

    assert decode_command(encode_ipc_message(command)) == command
    assert CommandType.parse("detect_cu5") is CommandType.DIAGNOSE_CU5
    assert command.to_dict()["command"] == "diagnose_cu5"
