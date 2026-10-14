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
    hotkey_error_message,
)


class _FakeHotkeyApi:
    def __init__(self) -> None:
        self.registered: dict[int, tuple[int, int]] = {}
        self.calls: list[tuple[object, ...]] = []
        self.fail_register: set[tuple[int, int]] = set()
        self.fail_unregister: set[int] = set()
        self.register_error = 1409
        self.unregister_results: dict[int, list[int]] = {}

    def register_hot_key(self, hwnd, identifier, modifiers, virtual_key):
        self.calls.append(("register", hwnd, identifier, modifiers, virtual_key))
        chord = (modifiers, virtual_key)
        if chord in self.fail_register:
            return False, self.register_error
        if identifier in self.registered or chord in self.registered.values():
            return False, 1409
        self.registered[identifier] = chord
        return True, 0

    def unregister_hot_key(self, hwnd, identifier):
        self.calls.append(("unregister", hwnd, identifier))
        results = self.unregister_results.get(identifier)
        if results:
            error_code = results.pop(0)
            if error_code:
                return False, error_code
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


def test_group_can_exchange_two_chords_without_releasing_unchanged_binding() -> None:
    api = _FakeHotkeyApi()
    manager = WindowsGlobalHotkeyManager(100, api=api)
    manager.bind_all([
        HotkeyBinding(1, MOD_CONTROL, ord("A")),
        HotkeyBinding(2, MOD_CONTROL, ord("B")),
        HotkeyBinding(3, MOD_CONTROL, ord("C")),
    ])
    api.calls.clear()

    result = manager.bind_all([
        HotkeyBinding(1, MOD_CONTROL, ord("B")),
        HotkeyBinding(2, MOD_CONTROL, ord("A")),
        HotkeyBinding(3, MOD_CONTROL, ord("C")),
    ])

    assert [binding.virtual_key for binding in result] == [ord("B"), ord("A"), ord("C")]
    assert [call[:3] for call in api.calls] == [
        ("unregister", 100, 1), ("unregister", 100, 2),
        ("register", 100, 1), ("register", 100, 2),
    ]
    assert api.registered == {
        binding.identifier: (binding.modifiers, binding.virtual_key) for binding in result
    }


@pytest.mark.parametrize("requested", [
    [HotkeyBinding(1, MOD_CONTROL, ord("B")), HotkeyBinding(1, MOD_SHIFT, ord("C"))],
    [HotkeyBinding(1, MOD_CONTROL, ord("B")),
     HotkeyBinding(2, MOD_CONTROL | MOD_NOREPEAT, ord("B"))],
])
def test_group_rejects_duplicate_ids_or_normalized_chords_before_changes(requested) -> None:
    api = _FakeHotkeyApi()
    manager = WindowsGlobalHotkeyManager(100, api=api)
    previous = manager.bind(HotkeyBinding(1, MOD_CONTROL, ord("A")))
    api.calls.clear()

    with pytest.raises(HotkeyRegistrationError) as captured:
        manager.bind_all(requested)

    assert captured.value.error_code == 87
    assert captured.value.operation == "validate"
    assert manager.bindings == (previous,)
    assert api.calls == []


def test_group_validates_every_binding_before_touching_existing_registration() -> None:
    api = _FakeHotkeyApi()
    manager = WindowsGlobalHotkeyManager(100, api=api)
    previous = manager.bind(HotkeyBinding(1, MOD_CONTROL, ord("A")))
    api.calls.clear()

    with pytest.raises(ValueError, match="虚拟键码"):
        manager.bind_all([
            HotkeyBinding(1, MOD_CONTROL, ord("B")), HotkeyBinding(2, MOD_CONTROL, 0),
        ])

    assert manager.bindings == (previous,)
    assert api.calls == []


def test_group_removes_missing_keys_and_empty_group_releases_everything() -> None:
    api = _FakeHotkeyApi()
    manager = WindowsGlobalHotkeyManager(100, api=api)
    manager.bind_all([
        HotkeyBinding(1, MOD_CONTROL, ord("A")), HotkeyBinding(2, MOD_CONTROL, ord("B")),
    ])
    api.calls.clear()

    manager.bind_all([HotkeyBinding(2, MOD_CONTROL, ord("B"))])
    assert api.calls == [("unregister", 100, 1)]
    assert manager.bind_all([]) == ()
    assert api.registered == {}


def test_group_release_failure_restores_already_released_keys() -> None:
    api = _FakeHotkeyApi()
    manager = WindowsGlobalHotkeyManager(100, api=api)
    old = manager.bind_all([
        HotkeyBinding(1, MOD_CONTROL, ord("A")), HotkeyBinding(2, MOD_CONTROL, ord("B")),
    ])
    api.fail_unregister.add(2)
    api.calls.clear()

    with pytest.raises(HotkeyRegistrationError) as captured:
        manager.bind_all([
            HotkeyBinding(1, MOD_CONTROL, ord("B")), HotkeyBinding(2, MOD_CONTROL, ord("A")),
        ])

    error = captured.value
    assert (error.identifier, error.operation, error.error_code) == (2, "unregister", 5)
    assert error.rollback_errors == ()
    assert manager.bindings == old
    assert [call[:3] for call in api.calls] == [
        ("unregister", 100, 1), ("unregister", 100, 2), ("register", 100, 1),
    ]
    assert api.registered == {binding.identifier: (binding.modifiers, binding.virtual_key)
                              for binding in old}


def test_second_registration_failure_removes_new_keys_and_restores_complete_group() -> None:
    api = _FakeHotkeyApi()
    manager = WindowsGlobalHotkeyManager(100, api=api)
    old = manager.bind_all([
        HotkeyBinding(1, MOD_CONTROL, ord("A")), HotkeyBinding(2, MOD_CONTROL, ord("B")),
    ])
    api.fail_register.add((MOD_CONTROL | MOD_NOREPEAT, ord("C")))
    api.calls.clear()

    with pytest.raises(HotkeyRegistrationError) as captured:
        manager.bind_all([
            HotkeyBinding(1, MOD_CONTROL, ord("B")), HotkeyBinding(2, MOD_CONTROL, ord("C")),
        ])

    assert captured.value.identifier == 2
    assert captured.value.error_code == 1409
    assert captured.value.rollback_errors == ()
    assert manager.bindings == old
    assert [call[:3] for call in api.calls] == [
        ("unregister", 100, 1), ("unregister", 100, 2),
        ("register", 100, 1), ("register", 100, 2),
        ("unregister", 100, 1), ("register", 100, 1), ("register", 100, 2),
    ]


def test_failed_restoration_reports_missing_binding_and_retry_keeps_working_keys() -> None:
    api = _FakeHotkeyApi()
    manager = WindowsGlobalHotkeyManager(100, api=api)
    old = manager.bind_all([
        HotkeyBinding(1, MOD_CONTROL, ord("A")), HotkeyBinding(2, MOD_CONTROL, ord("B")),
    ])
    api.fail_register.update({
        (MOD_CONTROL | MOD_NOREPEAT, ord("A")), (MOD_CONTROL | MOD_NOREPEAT, ord("D")),
    })

    with pytest.raises(HotkeyRegistrationError, match="未能完整恢复") as captured:
        manager.bind_all([
            HotkeyBinding(1, MOD_CONTROL, ord("C")), HotkeyBinding(2, MOD_CONTROL, ord("D")),
        ])

    assert manager.bindings == (old[1],)
    assert api.registered == {2: (old[1].modifiers, old[1].virtual_key)}
    assert [(failure.identifier, failure.operation, failure.error_code)
            for failure in captured.value.rollback_errors] == [(1, "restore", 1409)]
    api.fail_register.clear()
    api.calls.clear()
    assert manager.bind_all(old) == old
    assert [call[:3] for call in api.calls] == [("register", 100, 1)]


def test_rollback_unregister_failure_keeps_new_binding_and_does_not_double_register_id() -> None:
    api = _FakeHotkeyApi()
    manager = WindowsGlobalHotkeyManager(100, api=api)
    manager.bind_all([
        HotkeyBinding(1, MOD_CONTROL, ord("A")), HotkeyBinding(2, MOD_CONTROL, ord("B")),
    ])
    api.fail_register.add((MOD_CONTROL | MOD_NOREPEAT, ord("C")))
    api.unregister_results[1] = [0, 5]
    api.calls.clear()
    requested = HotkeyBinding(1, MOD_CONTROL, ord("B")).normalized()

    with pytest.raises(HotkeyRegistrationError, match="未能完整恢复") as captured:
        manager.bind_all([requested, HotkeyBinding(2, MOD_CONTROL, ord("C"))])

    assert manager.bindings == (requested,)
    assert api.registered == {1: (requested.modifiers, requested.virtual_key)}
    assert [(failure.identifier, failure.operation, failure.error_code)
            for failure in captured.value.rollback_errors] == [
        (1, "unregister", 5), (2, "restore", 1409),
    ]
    assert sum(call[0] == "register" and call[2] == 1 for call in api.calls) == 1


@pytest.mark.parametrize(("code", "message"), [
    (1409, "占用"), (87, "参数无效"), (5, "拒绝访问"),
    (1234, "系统错误码 1234"), (0, "未提供系统错误码"),
])
def test_registration_error_preserves_system_code_and_specific_reason(code, message) -> None:
    api = _FakeHotkeyApi()
    manager = WindowsGlobalHotkeyManager(100, api=api)
    api.fail_register.add((MOD_CONTROL | MOD_NOREPEAT, ord("A")))
    api.register_error = code

    with pytest.raises(HotkeyRegistrationError, match=message) as captured:
        manager.bind(HotkeyBinding(1, MOD_CONTROL, ord("A")))

    assert captured.value.error_code == code
    assert captured.value.identifier == 1
    assert captured.value.operation == "register"
    assert message in hotkey_error_message(code)
    if code != 1409:
        assert "占用" not in str(captured.value)


def test_module_imports_off_windows_and_native_constructor_fails_lazily() -> None:
    if sys.platform == "win32":
        pytest.skip("non-Windows import guard")
    with pytest.raises(WindowsHotkeyUnavailableError):
        WindowsGlobalHotkeyManager(1)
