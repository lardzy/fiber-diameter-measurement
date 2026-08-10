from __future__ import annotations

import ctypes
from dataclasses import dataclass
import sys
from typing import Protocol


WM_HOTKEY = 0x0312
MOD_ALT = 0x0001
MOD_CONTROL = 0x0002
MOD_SHIFT = 0x0004
MOD_WIN = 0x0008
MOD_NOREPEAT = 0x4000
_ALLOWED_MODIFIERS = MOD_ALT | MOD_CONTROL | MOD_SHIFT | MOD_WIN | MOD_NOREPEAT


class WindowsHotkeyUnavailableError(RuntimeError):
    """Raised when the Win32 hotkey API is requested on another platform."""


class HotkeyRegistrationError(RuntimeError):
    def __init__(self, message: str, *, error_code: int = 0) -> None:
        super().__init__(message)
        self.error_code = int(error_code)


@dataclass(frozen=True, slots=True)
class HotkeyBinding:
    identifier: int
    modifiers: int
    virtual_key: int

    def normalized(self) -> "HotkeyBinding":
        identifier = int(self.identifier)
        virtual_key = int(self.virtual_key)
        modifiers = int(self.modifiers)
        if identifier <= 0:
            raise ValueError("全局快捷键 ID 必须为正整数。")
        if not (1 <= virtual_key <= 0xFE):
            raise ValueError("全局快捷键虚拟键码无效。")
        if modifiers & ~_ALLOWED_MODIFIERS:
            raise ValueError("全局快捷键包含不支持的修饰键。")
        return HotkeyBinding(
            identifier=identifier,
            modifiers=modifiers | MOD_NOREPEAT,
            virtual_key=virtual_key,
        )


class GlobalHotkeyNativeApi(Protocol):
    def register_hot_key(
        self,
        hwnd: int,
        identifier: int,
        modifiers: int,
        virtual_key: int,
    ) -> tuple[bool, int]: ...

    def unregister_hot_key(self, hwnd: int, identifier: int) -> tuple[bool, int]: ...


class _CtypesGlobalHotkeyApi:
    def __init__(self) -> None:
        if sys.platform != "win32":
            raise WindowsHotkeyUnavailableError("全局快捷键仅能在 Windows 上注册。")
        win_dll = getattr(ctypes, "WinDLL", None)
        if win_dll is None:  # pragma: no cover - defensive platform guard
            raise WindowsHotkeyUnavailableError("当前 Python 运行时不提供 Win32 API。")
        self._user32 = win_dll("user32", use_last_error=True)
        self._user32.RegisterHotKey.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_uint,
            ctypes.c_uint,
        ]
        self._user32.RegisterHotKey.restype = ctypes.c_int
        self._user32.UnregisterHotKey.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self._user32.UnregisterHotKey.restype = ctypes.c_int

    @staticmethod
    def _last_error() -> int:
        getter = getattr(ctypes, "get_last_error", None)
        return int(getter()) if callable(getter) else 0

    def register_hot_key(
        self,
        hwnd: int,
        identifier: int,
        modifiers: int,
        virtual_key: int,
    ) -> tuple[bool, int]:
        result = bool(
            self._user32.RegisterHotKey(
                ctypes.c_void_p(int(hwnd)),
                int(identifier),
                int(modifiers),
                int(virtual_key),
            )
        )
        return result, 0 if result else self._last_error()

    def unregister_hot_key(self, hwnd: int, identifier: int) -> tuple[bool, int]:
        result = bool(
            self._user32.UnregisterHotKey(
                ctypes.c_void_p(int(hwnd)),
                int(identifier),
            )
        )
        return result, 0 if result else self._last_error()


class WindowsGlobalHotkeyManager:
    """Owns process-global Win32 hotkeys for one native message window.

    Rebinding is transactional where Win32 permits it: the old binding is
    unregistered, the new binding is attempted, and the old binding is restored
    if the new chord is already owned by another process.
    """

    def __init__(
        self,
        hwnd: int,
        *,
        api: GlobalHotkeyNativeApi | None = None,
    ) -> None:
        self._hwnd = int(hwnd)
        if self._hwnd <= 0:
            raise ValueError("全局快捷键需要有效的原生窗口句柄。")
        self._api = api or _CtypesGlobalHotkeyApi()
        self._bindings: dict[int, HotkeyBinding] = {}

    @property
    def bindings(self) -> tuple[HotkeyBinding, ...]:
        return tuple(self._bindings[key] for key in sorted(self._bindings))

    def binding(self, identifier: int) -> HotkeyBinding | None:
        return self._bindings.get(int(identifier))

    def bind(self, binding: HotkeyBinding) -> HotkeyBinding:
        requested = binding.normalized()
        previous = self._bindings.get(requested.identifier)
        if previous == requested:
            return requested

        if previous is not None:
            removed, error_code = self._api.unregister_hot_key(
                self._hwnd,
                previous.identifier,
            )
            if not removed:
                raise HotkeyRegistrationError(
                    "无法释放原全局快捷键，快捷键未修改。",
                    error_code=error_code,
                )

        registered, error_code = self._api.register_hot_key(
            self._hwnd,
            requested.identifier,
            requested.modifiers,
            requested.virtual_key,
        )
        if registered:
            self._bindings[requested.identifier] = requested
            return requested

        if previous is not None:
            restored, restore_error = self._api.register_hot_key(
                self._hwnd,
                previous.identifier,
                previous.modifiers,
                previous.virtual_key,
            )
            if restored:
                self._bindings[previous.identifier] = previous
                raise HotkeyRegistrationError(
                    "新快捷键已被其它程序占用，已恢复原快捷键。",
                    error_code=error_code,
                )
            self._bindings.pop(previous.identifier, None)
            raise HotkeyRegistrationError(
                "新快捷键注册失败，且原快捷键恢复失败。"
                f"（新错误 {error_code}，恢复错误 {restore_error}）",
                error_code=error_code or restore_error,
            )

        raise HotkeyRegistrationError(
            "快捷键已被其它程序占用或系统拒绝注册。",
            error_code=error_code,
        )

    def unbind(self, identifier: int) -> bool:
        identifier = int(identifier)
        existing = self._bindings.get(identifier)
        if existing is None:
            return False
        removed, error_code = self._api.unregister_hot_key(self._hwnd, identifier)
        if not removed:
            raise HotkeyRegistrationError(
                "无法注销全局快捷键。",
                error_code=error_code,
            )
        self._bindings.pop(identifier, None)
        return True

    def binding_for_message(
        self,
        message: int,
        wparam: int,
        lparam: int = 0,
    ) -> HotkeyBinding | None:
        if int(message) != WM_HOTKEY:
            return None
        binding = self._bindings.get(int(wparam))
        if binding is None or not lparam:
            return binding
        message_modifiers = (int(lparam) & 0xFFFF) & ~MOD_NOREPEAT
        message_virtual_key = (int(lparam) >> 16) & 0xFFFF
        expected_modifiers = binding.modifiers & ~MOD_NOREPEAT
        if (
            message_modifiers != expected_modifiers
            or message_virtual_key != binding.virtual_key
        ):
            return None
        return binding

    def close(self) -> None:
        first_error: HotkeyRegistrationError | None = None
        for identifier in tuple(sorted(self._bindings)):
            try:
                self.unbind(identifier)
            except HotkeyRegistrationError as exc:
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error

    def __enter__(self) -> "WindowsGlobalHotkeyManager":
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()


__all__ = [
    "HotkeyBinding",
    "HotkeyRegistrationError",
    "GlobalHotkeyNativeApi",
    "MOD_ALT",
    "MOD_CONTROL",
    "MOD_NOREPEAT",
    "MOD_SHIFT",
    "MOD_WIN",
    "WM_HOTKEY",
    "WindowsGlobalHotkeyManager",
    "WindowsHotkeyUnavailableError",
]
