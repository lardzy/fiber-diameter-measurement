from __future__ import annotations

import time
from dataclasses import replace

import pytest
from PySide6.QtTest import QTest

from fdm.platform.windows_global_hotkey import WindowsGlobalHotkeyManager
from fdm.screenshot_agent import AgentCommandError, ScreenshotAgent, _MAX_CAPTURE_QUEUE
from fdm.screenshot_protocol import IPCCommand
from fdm.screenshot_settings import HotkeyBinding, ScreenshotSettings, ScreenshotSettingsIO
from fdm.services.screenshot_capture import CaptureCoordinator, CaptureMode, CaptureRequest
from test_screenshot_agent import _CaptureBackend, _OutputService, _Locator, _Autostart


class NativeApi:
    def __init__(self):
        self.registered = {}
        self.occupied = set()

    def register_hot_key(self, hwnd, identifier, modifiers, key):
        chord = (modifiers, key)
        if chord in self.occupied or chord in self.registered.values():
            return False, 1409
        self.registered[identifier] = chord
        return True, 0

    def unregister_hot_key(self, hwnd, identifier):
        return self.registered.pop(identifier, None) is not None, 0


@pytest.fixture
def agent(tmp_path, desktop_application):
    path = tmp_path / 'settings.json'
    ScreenshotSettingsIO.save(ScreenshotSettings(enabled=True), path)
    api = NativeApi()
    item = ScreenshotAgent(desktop_application, CaptureCoordinator(_CaptureBackend()), settings_path=path,
                           output_service=_OutputService(), cu5_locator=_Locator(),
                           hotkey_manager_factory=lambda hwnd: WindowsGlobalHotkeyManager(hwnd, api=api),
                           autostart_manager_factory=_Autostart)
    item.start()
    yield item, api
    item.close()


def test_swap_registered_keys_as_group(agent):
    item, api = agent
    result = item.reload_settings({'hotkeys': {
        **{mode.value: binding.to_dict() for mode, binding in item.settings.hotkeys.items()},
        'region': HotkeyBinding('Alt+Print').to_dict(),
        'window': HotkeyBinding('Print').to_dict(),
    }})
    assert not result['hotkey_errors']
    assert api.registered[0x5F01] == (0x4001, 0x2C)
    assert api.registered[0x5F02] == (0x4000, 0x2C)


def test_failure_reports_requested_and_restored_actual_key(agent):
    item, api = agent
    api.occupied.add((0x4002, ord('K')))
    result = item.reload_settings({'hotkeys': {
        **{mode.value: binding.to_dict() for mode, binding in item.settings.hotkeys.items()},
        'region': HotkeyBinding('Ctrl+K').to_dict(),
    }})
    row = next(row for row in result['hotkeys'] if row['mode'] == 'region')
    assert row['error_code'] == 1409
    assert row['requested'] == 'Ctrl+K'
    assert row['effective'] == 'Print'
    assert row['state'] == 'restored'
    assert item.settings.hotkeys[CaptureMode.REGION].sequence == 'Print'
    assert len(api.registered) == 5


def test_failed_rollback_reports_new_key_that_could_not_be_unregistered(agent, monkeypatch):
    item, api = agent
    api.occupied.add((0x4002, ord('K')))
    original_unregister = api.unregister_hot_key

    def fail_to_release_new_region_key(hwnd, identifier):
        if identifier == 0x5F01 and api.registered.get(identifier) == (0x4001, 0x2C):
            return False, 5
        return original_unregister(hwnd, identifier)

    monkeypatch.setattr(api, 'unregister_hot_key', fail_to_release_new_region_key)
    result = item.reload_settings({'hotkeys': {
        **{mode.value: binding.to_dict() for mode, binding in item.settings.hotkeys.items()},
        'region': HotkeyBinding('Alt+Print').to_dict(),
        'window': HotkeyBinding('Ctrl+K').to_dict(),
    }})

    rows = {row['mode']: row for row in result['hotkeys']}
    assert rows['region']['state'] == 'failed'
    assert rows['region']['error_code'] == 1409
    assert rows['region']['requested'] == 'Alt+Print'
    assert rows['region']['effective'] == 'Alt+Print'
    assert rows['region']['virtual_key'] == 0x2C
    assert rows['region']['modifiers'] == 0x4001
    assert api.registered[0x5F01] == (0x4001, 0x2C)
    assert rows['window']['effective'] == ''
    assert 0x5F02 not in api.registered
    # Desired settings are rolled back, while diagnostics keep the actual
    # residual registration visible until the user successfully retries.
    assert item.settings.hotkeys[CaptureMode.REGION].sequence == 'Print'
    monkeypatch.setattr(api, 'unregister_hot_key', original_unregister)
    restored = {row['mode']: row for row in item.retry_hotkeys()['hotkeys']}
    assert restored['region']['state'] == 'registered'
    assert restored['region']['effective'] == 'Print'
    assert restored['window']['effective'] == 'Alt+Print'
    assert len(api.registered) == 5


def test_recording_leases_release_keys_and_expire_without_losing_settings(agent):
    item, api = agent
    settings = item.settings.to_dict()
    item.set_hotkey_recording('one', True)
    item.set_hotkey_recording('two', True)
    assert not api.registered
    assert item.status()['hotkeys_paused']
    item._hotkey_pressed(0x5F01, 0)
    assert not item._ipc_capture_queue
    item.set_hotkey_recording('one', False)
    assert not api.registered
    item._recording_leases['two'] = time.monotonic() - 1
    item._expire_recording_leases()
    assert len(api.registered) == 5
    assert not item.status()['hotkeys_paused']
    assert item.settings.to_dict() == settings


def test_hotkey_records_receipt_and_defers_coalesced_capture(agent):
    item, _ = agent
    calls = []
    item.begin_capture = lambda request: calls.append(request)
    item._hotkey_pressed(0x5F01, 0)
    item._hotkey_pressed(0x5F01, 0)
    assert not calls
    assert len(item._ipc_capture_queue) == 1
    assert item.status()['hotkeys'][0]['last_received_at']
    QTest.qWait(5)
    assert len(calls) == 1


def test_ipc_duplicate_and_queue_limit_are_explicit(agent):
    item, _ = agent
    command = IPCCommand.capture(CaptureMode.FULL_SCREEN, request_id='same')
    first = item.handle_command(command)
    assert item.handle_command(command)['duplicate']
    assert len(item._ipc_capture_queue) == 1
    with pytest.raises(AgentCommandError, match='不同截图命令'):
        item.handle_command(IPCCommand.capture(CaptureMode.REGION, request_id='same'))
    for _ in range(_MAX_CAPTURE_QUEUE - 1):
        item.handle_command(IPCCommand.capture(CaptureMode.FULL_SCREEN))
    with pytest.raises(AgentCommandError, match='队列已满'):
        item.handle_command(IPCCommand.capture(CaptureMode.FULL_SCREEN))
    assert first['trigger_id'] == 'same'


def test_native_key_identity_is_used_instead_of_glyph_fallback(agent):
    item, api = agent
    item.reload_settings({'hotkeys': {'region': HotkeyBinding('Ctrl+Y', True, ord('Z'), 2).to_dict()}})
    assert api.registered[0x5F01] == (0x4002, ord('Z'))


@pytest.mark.parametrize('payload', [None, {'reload': True}, {'enabled': False}])
def test_recording_rejects_all_settings_apply_paths(agent, payload):
    item, _ = agent
    previous = item.settings.to_dict()
    item.set_hotkey_recording('editor', True)
    with pytest.raises(AgentCommandError, match='正在录入'):
        item.reload_settings(payload)
    assert item.settings.to_dict() == previous


def test_failed_unregistration_is_reported_and_does_not_dispatch_disabled_key(agent):
    item, api = agent
    api.unregister_hot_key = lambda hwnd, identifier: (False, 5)
    item._settings = replace(item.settings, hotkeys={
        **item.settings.hotkeys, CaptureMode.REGION: HotkeyBinding('Print', False),
    })
    item.retry_hotkeys()
    assert item.status()['registered_hotkeys'] == 5
    row = item.status()['hotkeys'][0]
    assert row['effective'] == 'Print'
    assert row['error_code'] == 5
    item._hotkey_pressed(0x5F01, 0)
    assert not item._ipc_capture_queue


def test_capture_failure_and_cancellation_have_terminal_trace(agent):
    item, _ = agent
    item.coordinator.set_last_region(None)
    item._enqueue_capture(CaptureRequest(CaptureMode.LAST_REGION), source='test')
    QTest.qWait(5)
    row = item.status()['recent_captures'][-1]
    assert row['stage'] == 'capture_failed'
    assert '上次区域' in row['error']
    item._enqueue_capture(CaptureRequest(CaptureMode.REGION, delay_ms=1000), source='test')
    QTest.qWait(5)
    item.coordinator.cancel()
    assert item.status()['recent_captures'][-1]['stage'] == 'cancelled'


def test_dispatch_exception_does_not_leave_later_requests_stuck(agent):
    item, _ = agent
    calls = []

    def begin(request):
        calls.append(request)
        if len(calls) == 1:
            raise RuntimeError('window enumeration failed')

    item.begin_capture = begin
    item._enqueue_capture(CaptureRequest(CaptureMode.SMART), source='test')
    item._enqueue_capture(CaptureRequest(CaptureMode.FULL_SCREEN), source='test')
    deadline = time.perf_counter() + 1.0
    while len(calls) < 2 and time.perf_counter() < deadline:
        QTest.qWait(10)
    assert len(calls) == 2
    assert not item._ipc_capture_queue
    assert item.status()['recent_captures'][0]['stage'] == 'capture_failed'


def test_recording_cancels_already_dispatched_delayed_hotkey(agent):
    item, _ = agent
    item._enqueue_capture(CaptureRequest(CaptureMode.REGION, delay_ms=1000), source='hotkey')
    deadline = time.perf_counter() + 1.0
    while item._active_capture_request is None and time.perf_counter() < deadline:
        QTest.qWait(10)
    assert item._active_capture_request is not None
    item.set_hotkey_recording('editor', True)
    assert item._active_capture_request is None
    assert not item.coordinator._delay_timer.isActive()
    assert item.status()['recent_captures'][-1]['stage'] == 'recording_paused'
