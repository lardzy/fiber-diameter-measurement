from __future__ import annotations

import sys

import pytest

from fdm.platform.windows_global_hotkey import (
    HotkeyBinding,
    HotkeyRegistrationError,
    MOD_CONTROL,
    MOD_NOREPEAT,
    MOD_SHIFT,
    WM_HOTKEY,
    WindowsGlobalHotkeyManager,
    WindowsHotkeyUnavailableError,
)


class _FakeHotkeyApi:
    def __init__(self) -> None:
        self.registered: dict[int, tuple[int, int]] = {}
        self.calls: list[tuple[object, ...]] = []
        self.fail_register: set[tuple[int, int]] = set()
        self.fail_unregister: set[int] = set()

    def register_hot_key(self, hwnd, identifier, modifiers, virtual_key):
        self.calls.append(("register", hwnd, identifier, modifiers, virtual_key))
        chord = (modifiers, virtual_key)
        if chord in self.fail_register:
            return False, 1409
        self.registered[identifier] = chord
        return True, 0

    def unregister_hot_key(self, hwnd, identifier):
        self.calls.append(("unregister", hwnd, identifier))
        if identifier in self.fail_unregister:
            return False, 5
        self.registered.pop(identifier, None)
        return True, 0


def test_binding_always_uses_mod_norepeat_and_dispatches_matching_message() -> None:
    api = _FakeHotkeyApi()
    manager = WindowsGlobalHotkeyManager(100, api=api)

    binding = manager.bind(HotkeyBinding(7, MOD_CONTROL | MOD_SHIFT, ord("A")))

    assert binding.modifiers & MOD_NOREPEAT
    assert api.registered[7][0] & MOD_NOREPEAT
    lparam = ((ord("A") & 0xFFFF) << 16) | (MOD_CONTROL | MOD_SHIFT)
    assert manager.binding_for_message(WM_HOTKEY, 7, lparam) == binding
    assert manager.binding_for_message(WM_HOTKEY, 7, lparam + 1) is None
    assert manager.binding_for_message(0x1234, 7, lparam) is None


def test_rebind_conflict_restores_previous_binding() -> None:
    api = _FakeHotkeyApi()
    manager = WindowsGlobalHotkeyManager(100, api=api)
    previous = manager.bind(HotkeyBinding(1, MOD_CONTROL, ord("A")))
    requested = HotkeyBinding(1, MOD_CONTROL, ord("B")).normalized()
    api.fail_register.add((requested.modifiers, requested.virtual_key))

    with pytest.raises(HotkeyRegistrationError, match="恢复原快捷键") as captured:
        manager.bind(requested)

    assert captured.value.error_code == 1409
    assert manager.binding(1) == previous
    assert api.registered[1] == (previous.modifiers, previous.virtual_key)
    assert [call[0] for call in api.calls[-3:]] == ["unregister", "register", "register"]


def test_failed_initial_registration_does_not_create_binding() -> None:
    api = _FakeHotkeyApi()
    manager = WindowsGlobalHotkeyManager(100, api=api)
    requested = HotkeyBinding(2, MOD_SHIFT, ord("S")).normalized()
    api.fail_register.add((requested.modifiers, requested.virtual_key))

    with pytest.raises(HotkeyRegistrationError) as captured:
        manager.bind(requested)

    assert captured.value.error_code == 1409
    assert manager.binding(2) is None


def test_failed_unregister_keeps_owned_binding() -> None:
    api = _FakeHotkeyApi()
    manager = WindowsGlobalHotkeyManager(100, api=api)
    binding = manager.bind(HotkeyBinding(3, MOD_CONTROL, ord("C")))
    api.fail_unregister.add(3)

    with pytest.raises(HotkeyRegistrationError, match="无法注销"):
        manager.unbind(3)

    assert manager.binding(3) == binding


def test_module_imports_off_windows_and_native_constructor_fails_lazily() -> None:
    if sys.platform == "win32":
        pytest.skip("non-Windows import guard")
    with pytest.raises(WindowsHotkeyUnavailableError):
        WindowsGlobalHotkeyManager(1)
