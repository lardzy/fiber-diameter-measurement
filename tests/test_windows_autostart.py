from __future__ import annotations

from pathlib import Path
import sys

import pytest

from fdm.platform.windows_autostart import (
    WindowsAutostartManager,
    WindowsAutostartUnavailableError,
    windows_command_line,
)


class _FakeRegistry:
    def __init__(self, values: dict[str, str] | None = None) -> None:
        self.values = dict(values or {})
        self.writes: list[tuple[str, str]] = []
        self.deletes: list[str] = []

    def read_run_value(self, value_name: str) -> str | None:
        return self.values.get(value_name)

    def write_run_value(self, value_name: str, command: str) -> None:
        self.writes.append((value_name, command))
        self.values[value_name] = command

    def delete_run_value(self, value_name: str) -> bool:
        self.deletes.append(value_name)
        return self.values.pop(value_name, None) is not None


def test_command_line_quotes_executable_and_arguments() -> None:
    command = windows_command_line(
        Path("/Program Files/Fiber Screenshot/FiberScreenshot.exe"),
        ("--background", "--profile", "常用 配置"),
    )

    assert command.startswith('"/Program Files/Fiber Screenshot/FiberScreenshot.exe"')
    assert '"常用 配置"' in command


def test_enable_and_disable_owned_hkcu_run_value() -> None:
    registry = _FakeRegistry()
    manager = WindowsAutostartManager(
        value_name="FiberScreenshot",
        executable="/opt/FiberScreenshot.exe",
        registry=registry,
    )

    assert not manager.status().enabled
    enabled = manager.enable()
    assert enabled.enabled and enabled.owned
    assert registry.writes == [("FiberScreenshot", manager.command)]

    disabled = manager.disable()
    assert not disabled.enabled
    assert registry.deletes == ["FiberScreenshot"]


def test_disable_does_not_delete_foreign_value_with_same_name() -> None:
    registry = _FakeRegistry({"FiberScreenshot": '"C:\\Other.exe" --background'})
    manager = WindowsAutostartManager(
        value_name="FiberScreenshot",
        executable="/opt/FiberScreenshot.exe",
        registry=registry,
    )

    status = manager.disable()

    assert not status.enabled
    assert not status.owned
    assert registry.deletes == []
    assert status.current_command.endswith("--background")


def test_native_registry_is_only_created_on_windows() -> None:
    if sys.platform == "win32":
        pytest.skip("non-Windows import guard")
    with pytest.raises(WindowsAutostartUnavailableError):
        WindowsAutostartManager(
            value_name="FiberScreenshot",
            executable="/opt/FiberScreenshot.exe",
        )
