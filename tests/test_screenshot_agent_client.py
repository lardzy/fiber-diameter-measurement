from __future__ import annotations

from pathlib import Path

import pytest

from fdm.screenshot_protocol import CommandType, IPCResponse, SCREENSHOT_EXECUTABLE_NAME
from fdm.services.screenshot_agent_client import (
    ScreenshotAgentClient,
    ScreenshotAgentCommandError,
    ScreenshotAgentLaunchError,
    ScreenshotAgentLaunchResult,
    ScreenshotAgentLaunchSpec,
    ScreenshotAgentProtocolError,
    ScreenshotAgentTimeoutError,
    ScreenshotAgentUnavailableError,
    resolve_screenshot_agent_launch_spec,
)
from fdm.services.screenshot_capture import CaptureMode


class _FakeClock:
    def __init__(self) -> None:
        self.value = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.value

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.value += seconds


class _FakeTransport:
    def __init__(self) -> None:
        self.running = False
        self.starting = False
        self.failed_probes_after_launch = 0
        self.requests = []
        self.response_error = ""
        self.mismatched_request_id = False

    def send(self, command, *, timeout_ms):
        self.requests.append((command, timeout_ms))
        if not self.running:
            if self.starting and self.failed_probes_after_launch <= 0:
                self.running = True
            elif self.starting:
                self.failed_probes_after_launch -= 1
            if not self.running:
                raise ScreenshotAgentUnavailableError("not running")
        request_id = "wrong-id" if self.mismatched_request_id else command.request_id
        if self.response_error:
            return IPCResponse.failure(request_id, self.response_error)
        result = {"accepted": True, "command": command.command.value}
        return IPCResponse.success(request_id, result)


class _FakeLauncher:
    def __init__(
        self,
        transport: _FakeTransport,
        *,
        result: ScreenshotAgentLaunchResult | None = None,
    ) -> None:
        self.transport = transport
        self.result = result or ScreenshotAgentLaunchResult(True, pid=8123)
        self.calls: list[ScreenshotAgentLaunchSpec] = []

    def start_detached(self, launch_spec):
        self.calls.append(launch_spec)
        if self.result.started:
            self.transport.starting = True
        return self.result


def _launch_spec(tmp_path: Path) -> ScreenshotAgentLaunchSpec:
    executable = tmp_path / "FiberScreenshotTool.exe"
    executable.write_bytes(b"MZ fake")
    return ScreenshotAgentLaunchSpec(
        program=str(executable),
        arguments=(),
        working_directory=str(tmp_path),
        packaged=True,
    )


def test_resolves_packaged_tool_next_to_main_executable(tmp_path: Path) -> None:
    main_executable = tmp_path / "FiberDiameterMeasurement.exe"
    spec = resolve_screenshot_agent_launch_spec(
        executable=main_executable,
        frozen=True,
    )

    assert spec.program == str(tmp_path / SCREENSHOT_EXECUTABLE_NAME)
    assert spec.arguments == ()
    assert spec.working_directory == str(tmp_path)
    assert spec.packaged


def test_resolves_development_agent_as_python_module(tmp_path: Path) -> None:
    python = tmp_path / "venv" / "bin" / "python"
    spec = resolve_screenshot_agent_launch_spec(executable=python, frozen=False)

    assert spec.program == str(python)
    assert spec.arguments == ("-m", "fdm.screenshot_agent")
    assert not spec.packaged


def test_ensure_started_does_not_launch_an_existing_agent(tmp_path: Path) -> None:
    transport = _FakeTransport()
    transport.running = True
    launcher = _FakeLauncher(transport)
    client = ScreenshotAgentClient(
        transport=transport,
        launcher=launcher,
        launch_spec=_launch_spec(tmp_path),
    )

    status = client.ensure_started()

    assert status.running
    assert status.result["already_running"] is True
    assert launcher.calls == []
    assert [request.command for request, _timeout in transport.requests] == [CommandType.PING]


def test_ensure_started_uses_detached_launcher_and_bounded_retries(tmp_path: Path) -> None:
    transport = _FakeTransport()
    transport.failed_probes_after_launch = 2
    launcher = _FakeLauncher(transport)
    clock = _FakeClock()
    spec = _launch_spec(tmp_path)
    client = ScreenshotAgentClient(
        transport=transport,
        launcher=launcher,
        launch_spec=spec,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )

    status = client.ensure_started(
        timeout_ms=2_000,
        retry_interval_ms=25,
        max_attempts=5,
    )

    assert status.running
    assert status.result["started"] is True
    assert status.result["pid"] == 8123
    assert launcher.calls == [spec]
    assert len(clock.sleeps) == 2
    assert len(transport.requests) == 4  # initial probe + three post-launch probes


def test_ensure_started_times_out_after_max_attempts(tmp_path: Path) -> None:
    transport = _FakeTransport()
    transport.failed_probes_after_launch = 999
    launcher = _FakeLauncher(transport)
    clock = _FakeClock()
    client = ScreenshotAgentClient(
        transport=transport,
        launcher=launcher,
        launch_spec=_launch_spec(tmp_path),
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )

    with pytest.raises(ScreenshotAgentTimeoutError, match="未就绪"):
        client.ensure_started(
            timeout_ms=10_000,
            retry_interval_ms=10,
            max_attempts=3,
        )

    assert len(launcher.calls) == 1
    assert len(transport.requests) == 4  # initial probe + exactly three retries
    assert len(clock.sleeps) == 2


def test_ensure_started_reports_missing_or_failed_executable(tmp_path: Path) -> None:
    transport = _FakeTransport()
    missing = ScreenshotAgentLaunchSpec(
        program=str(tmp_path / "missing.exe"),
        arguments=(),
        working_directory=str(tmp_path),
        packaged=True,
    )
    client = ScreenshotAgentClient(
        transport=transport,
        launcher=_FakeLauncher(transport),
        launch_spec=missing,
    )
    with pytest.raises(ScreenshotAgentLaunchError, match="未找到"):
        client.ensure_started()

    failing_launcher = _FakeLauncher(
        transport,
        result=ScreenshotAgentLaunchResult(False, error="access denied"),
    )
    client = ScreenshotAgentClient(
        transport=transport,
        launcher=failing_launcher,
        launch_spec=_launch_spec(tmp_path),
    )
    with pytest.raises(ScreenshotAgentLaunchError, match="access denied"):
        client.ensure_started()


def test_all_protocol_commands_are_exposed_for_main_window(tmp_path: Path) -> None:
    transport = _FakeTransport()
    transport.running = True
    client = ScreenshotAgentClient(
        transport=transport,
        launcher=_FakeLauncher(transport),
        launch_spec=_launch_spec(tmp_path),
    )

    assert client.ping()
    assert client.status().running
    client.capture(CaptureMode.CU5, payload={"open_editor": False})
    client.show_settings()
    client.update_settings({"delay_ms": 500})
    assert client.shutdown()

    commands = [request.command for request, _timeout in transport.requests]
    assert commands == [
        CommandType.PING,
        CommandType.STATUS,
        CommandType.CAPTURE,
        CommandType.SHOW_SETTINGS,
        CommandType.UPDATE_SETTINGS,
        CommandType.SHUTDOWN,
    ]
    capture_command = transport.requests[2][0]
    assert capture_command.capture_mode is CaptureMode.CU5
    assert capture_command.payload == {"open_editor": False}
    assert transport.requests[4][0].payload == {"delay_ms": 500}


def test_failed_response_and_request_id_mismatch_are_not_silently_accepted(tmp_path: Path) -> None:
    transport = _FakeTransport()
    transport.running = True
    client = ScreenshotAgentClient(
        transport=transport,
        launcher=_FakeLauncher(transport),
        launch_spec=_launch_spec(tmp_path),
    )

    transport.response_error = "bad settings"
    with pytest.raises(ScreenshotAgentCommandError, match="bad settings"):
        client.update_settings({"delay_ms": -1})

    transport.response_error = ""
    transport.mismatched_request_id = True
    with pytest.raises(ScreenshotAgentProtocolError, match="request_id"):
        client.show_settings()


def test_friendly_status_and_shutdown_handle_agent_not_running(tmp_path: Path) -> None:
    transport = _FakeTransport()
    client = ScreenshotAgentClient(
        transport=transport,
        launcher=_FakeLauncher(transport),
        launch_spec=_launch_spec(tmp_path),
    )

    status = client.status()

    assert not status.running
    assert "not running" in status.error
    assert not client.shutdown()
