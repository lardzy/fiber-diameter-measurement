from __future__ import annotations

import ctypes
from collections.abc import Iterable
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


def hotkey_error_message(error_code: int) -> str:
    """Describe the actual Win32 failure without guessing which app owns a key."""
    error_code = int(error_code)
    if error_code == 1409:
        return "该组合键已被系统或其它程序占用，请更换组合键或释放占用后重试"
    if error_code == 87:
        return "快捷键参数无效，请检查按键和修饰键设置"
    if error_code == 5:
        return "Windows 拒绝访问，系统错误码 5"
    if error_code:
        return f"Windows 操作失败，系统错误码 {error_code}"
    return "Windows 操作失败，未提供系统错误码"


@dataclass(frozen=True, slots=True)
class HotkeyOperationFailure:
    identifier: int
    operation: str
    error_code: int
    message: str


class HotkeyRegistrationError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        error_code: int = 0,
        identifier: int | None = None,
        operation: str = "register",
        rollback_errors: tuple[HotkeyOperationFailure, ...] = (),
    ) -> None:
        super().__init__(message)
        self.error_code = int(error_code)
        self.identifier = identifier
        self.operation = operation
        self.rollback_errors = rollback_errors


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

    Group changes release affected bindings before registering replacements,
    allowing two actions to exchange keys. Win32 offers no atomic replacement:
    failures trigger best-effort restoration, and ``bindings`` always reflects
    successful native operations, including any unsuccessful rollback.
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
        desired = dict(self._bindings)
        desired[requested.identifier] = requested
        self.bind_all(desired.values())
        return requested

    def bind_all(self, bindings: Iterable[HotkeyBinding]) -> tuple[HotkeyBinding, ...]:
        """Replace the complete binding set, restoring old keys on failure.

        Validation completes before any native changes. Unchanged bindings are
        retained. A failed replacement can leave a partial set if Windows also
        rejects rollback; callers must read ``bindings`` after an exception.
        Call this on the same thread that owns the native message window.
        """
        desired: dict[int, HotkeyBinding] = {}
        chords: dict[tuple[int, int], int] = {}
        for binding in bindings:
            requested = binding.normalized()
            chord = (requested.modifiers, requested.virtual_key)
            if requested.identifier in desired:
                raise HotkeyRegistrationError(
                    f"全局快捷键 ID {requested.identifier} 重复，快捷键未修改。",
                    error_code=87,
                    identifier=requested.identifier,
                    operation="validate",
                )
            if chord in chords:
                raise HotkeyRegistrationError(
                    f"快捷键 {chords[chord]} 与 {requested.identifier} 使用相同组合键，"
                    "请为不同操作设置不同组合键。快捷键未修改。",
                    error_code=87,
                    identifier=requested.identifier,
                    operation="validate",
                )
            desired[requested.identifier] = requested
            chords[chord] = requested.identifier

        previous = dict(self._bindings)
        for identifier in sorted(previous):
            if previous[identifier] == desired.get(identifier):
                continue
            removed, error_code = self._api.unregister_hot_key(self._hwnd, identifier)
            if not removed:
                self._raise_with_restoration(
                    previous, identifier, "unregister", error_code,
                )
            self._bindings.pop(identifier)

        for identifier, requested in sorted(desired.items()):
            if self._bindings.get(identifier) == requested:
                continue
            registered, error_code = self._api.register_hot_key(
                self._hwnd, identifier, requested.modifiers, requested.virtual_key,
            )
            if not registered:
                self._raise_with_restoration(
                    previous, identifier, "register", error_code,
                )
            self._bindings[identifier] = requested
        return self.bindings

    def _raise_with_restoration(
        self,
        previous: dict[int, HotkeyBinding],
        identifier: int,
        operation: str,
        error_code: int,
    ) -> None:
        failures: list[HotkeyOperationFailure] = []
        # Remove every newly registered key before restoring any old one, so a
        # partially completed exchange cannot obstruct restoration unnecessarily.
        for current_id, current in tuple(sorted(self._bindings.items())):
            if current == previous.get(current_id):
                continue
            removed, remove_error = self._api.unregister_hot_key(self._hwnd, current_id)
            if removed:
                self._bindings.pop(current_id)
            else:
                failures.append(HotkeyOperationFailure(
                    current_id, "unregister", remove_error,
                    f"无法释放新快捷键：{hotkey_error_message(remove_error)}。",
                ))
        for old_id, old in sorted(previous.items()):
            if self._bindings.get(old_id) == old:
                continue
            if old_id in self._bindings:
                # RegisterHotKey does not replace an existing registration with
                # the same id. Do not create multiple registrations for one id.
                continue
            restored, restore_error = self._api.register_hot_key(
                self._hwnd, old_id, old.modifiers, old.virtual_key,
            )
            if restored:
                self._bindings[old_id] = old
            else:
                failures.append(HotkeyOperationFailure(
                    old_id, "restore", restore_error,
                    f"原快捷键恢复失败：{hotkey_error_message(restore_error)}。",
                ))

        action = "无法释放原全局快捷键" if operation == "unregister" else "快捷键注册失败"
        message = f"{action}：{hotkey_error_message(error_code)}。"
        if failures:
            message += "未能完整恢复原快捷键设置，请检查当前生效状态并重试。"
        elif previous:
            message += "已恢复原快捷键设置。"
        raise HotkeyRegistrationError(
            message,
            error_code=error_code,
            identifier=identifier,
            operation=operation,
            rollback_errors=tuple(failures),
        )

    def unbind(self, identifier: int) -> bool:
        identifier = int(identifier)
        existing = self._bindings.get(identifier)
        if existing is None:
            return False
        removed, error_code = self._api.unregister_hot_key(self._hwnd, identifier)
        if not removed:
            raise HotkeyRegistrationError(
                f"无法注销全局快捷键：{hotkey_error_message(error_code)}。",
                error_code=error_code,
                identifier=identifier,
                operation="unregister",
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
    "HotkeyOperationFailure",
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
    "hotkey_error_message",
]
