from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
from typing import Protocol


RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"


class WindowsAutostartUnavailableError(RuntimeError):
    pass


class AutostartRegistryApi(Protocol):
    def read_run_value(self, value_name: str) -> str | None: ...

    def write_run_value(self, value_name: str, command: str) -> None: ...

    def delete_run_value(self, value_name: str) -> bool: ...


class _WinRegistryApi:
    def __init__(self) -> None:
        if sys.platform != "win32":
            raise WindowsAutostartUnavailableError("开机自启注册仅支持 Windows。")
        try:
            import winreg
        except ImportError as exc:  # pragma: no cover - Windows always provides it
            raise WindowsAutostartUnavailableError("当前 Python 缺少 winreg。") from exc
        self._winreg = winreg

    def read_run_value(self, value_name: str) -> str | None:
        try:
            with self._winreg.OpenKey(
                self._winreg.HKEY_CURRENT_USER,
                RUN_KEY,
                0,
                self._winreg.KEY_QUERY_VALUE,
            ) as key:
                value, _kind = self._winreg.QueryValueEx(key, value_name)
        except FileNotFoundError:
            return None
        return str(value)

    def write_run_value(self, value_name: str, command: str) -> None:
        with self._winreg.CreateKeyEx(
            self._winreg.HKEY_CURRENT_USER,
            RUN_KEY,
            0,
            self._winreg.KEY_SET_VALUE,
        ) as key:
            self._winreg.SetValueEx(
                key,
                value_name,
                0,
                self._winreg.REG_SZ,
                command,
            )

    def delete_run_value(self, value_name: str) -> bool:
        try:
            with self._winreg.OpenKey(
                self._winreg.HKEY_CURRENT_USER,
                RUN_KEY,
                0,
                self._winreg.KEY_SET_VALUE,
            ) as key:
                self._winreg.DeleteValue(key, value_name)
        except FileNotFoundError:
            return False
        return True


@dataclass(frozen=True, slots=True)
class AutostartStatus:
    enabled: bool
    owned: bool
    configured_command: str
    current_command: str = ""


def windows_command_line(executable: str | Path, arguments: tuple[str, ...]) -> str:
    executable_path = Path(executable).expanduser()
    if not executable_path.is_absolute():
        raise ValueError("开机自启程序路径必须是绝对路径。")
    return subprocess.list2cmdline(
        [str(executable_path), *(str(argument) for argument in arguments)]
    )


class WindowsAutostartManager:
    """Manages one per-user HKCU Run value without requiring elevation."""

    def __init__(
        self,
        *,
        value_name: str,
        executable: str | Path,
        arguments: tuple[str, ...] = ("--background",),
        registry: AutostartRegistryApi | None = None,
    ) -> None:
        normalized_name = str(value_name).strip()
        if not normalized_name or "\\" in normalized_name:
            raise ValueError("开机自启注册表值名称无效。")
        self._value_name = normalized_name
        self._command = windows_command_line(executable, arguments)
        self._registry = registry or _WinRegistryApi()

    @property
    def value_name(self) -> str:
        return self._value_name

    @property
    def command(self) -> str:
        return self._command

    def status(self) -> AutostartStatus:
        current = self._registry.read_run_value(self._value_name)
        if current is None:
            return AutostartStatus(False, False, self._command)
        owned = current.strip().casefold() == self._command.strip().casefold()
        return AutostartStatus(
            enabled=owned,
            owned=owned,
            configured_command=self._command,
            current_command=current,
        )

    def enable(self) -> AutostartStatus:
        self._registry.write_run_value(self._value_name, self._command)
        return self.status()

    def disable(self) -> AutostartStatus:
        current = self._registry.read_run_value(self._value_name)
        if current is not None and current.strip().casefold() == self._command.strip().casefold():
            self._registry.delete_run_value(self._value_name)
        return self.status()

    def set_enabled(self, enabled: bool) -> AutostartStatus:
        return self.enable() if enabled else self.disable()


__all__ = [
    "AutostartRegistryApi",
    "AutostartStatus",
    "RUN_KEY",
    "WindowsAutostartManager",
    "WindowsAutostartUnavailableError",
    "windows_command_line",
]
