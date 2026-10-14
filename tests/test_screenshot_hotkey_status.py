from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QCoreApplication, QEvent, Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QApplication

from fdm.screenshot_protocol import CommandType, IPCCommand, IPCResponse
from fdm.services.screenshot_agent_client import ScreenshotAgentClient, ScreenshotAgentUnavailableError
from fdm.services.screenshot_capture import CaptureMode
from fdm.ui.screenshot_hotkey_status import ScreenshotHotkeyStatusController
from fdm.ui.screenshot_settings_page import ScreenshotSettingsPage


def _dispose_page(page: ScreenshotSettingsPage) -> None:
    # close() only hides a widget. Qt signal closures keep the page/controller
    # cycle alive, including all combo-box popup windows, across later tests.
    page.close()
    page.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_recording_signal_renews_same_lease_and_close_always_releases_it() -> None:
    page = ScreenshotSettingsPage()
    requests: list[IPCCommand] = []

    def send(command: IPCCommand) -> dict[str, object]:
        requests.append(command)
        return {"running": True, "hotkeys": [{"mode": "region", "state": "paused"}]}

    controller = ScreenshotHotkeyStatusController(page, send)
    try:
        page.show()
        page.hotkeyRecordingChanged.emit(True)
        first = requests[-1]
        assert first.command is CommandType.HOTKEY_RECORDING
        assert first.payload["active"] is True
        assert first.payload["owner"]
        assert "录入期间暂停" in page.hotkey_status_labels[CaptureMode.REGION].text()
        controller.refresh()
        assert requests[-1].payload == first.payload
        assert requests[-1].request_id != first.request_id
        controller.close()
        assert requests[-1].payload == {**first.payload, "active": False}
        count = len(requests)
        controller.close()
        controller.refresh()
        controller.finish_recording()
        controller.retry()
        page.hotkeyRecordingChanged.emit(True)
        assert len(requests) == count
    finally:
        controller.close()
        _dispose_page(page)


def test_hidden_page_stops_renewal_and_focus_lease_ends_when_page_hides() -> None:
    page = ScreenshotSettingsPage()
    requests: list[IPCCommand] = []
    controller = ScreenshotHotkeyStatusController(page, lambda command: requests.append(command) or {})
    try:
        page.show()
        QApplication.processEvents()
        page.hotkey_edits[CaptureMode.REGION].setFocus()
        QApplication.processEvents()
        assert requests[-1].payload["active"] is True
        page.hide()
        assert requests[-1].payload["active"] is False
        count = len(requests)
        controller.refresh()
        assert len(requests) == count
    finally:
        controller.close()
        _dispose_page(page)


def test_retry_presents_restored_binding_without_hiding_native_error() -> None:
    page = ScreenshotSettingsPage()
    requests: list[IPCCommand] = []

    def send(command: IPCCommand) -> dict[str, object]:
        requests.append(command)
        return {"running": True, "hotkeys": [{
            "mode": "region", "requested": "Ctrl+F8", "effective": "Print",
            "state": "restored", "error_code": 1409,
            "error": "注册失败，已恢复原绑定。",
        }]}

    controller = ScreenshotHotkeyStatusController(page, send)
    try:
        page.retry_hotkeys_button.click()
        assert requests[-1].command is CommandType.RETRY_HOTKEYS
        text = page.hotkey_status_labels[CaptureMode.REGION].text()
        assert "已恢复原绑定" in text
        assert "当前生效：Print" in text
        assert "Ctrl+F8" in text
        assert "1409" in text
    finally:
        controller.close()
        _dispose_page(page)


def test_uncertain_recording_request_is_visible_and_close_still_releases_owner() -> None:
    page = ScreenshotSettingsPage()
    requests: list[IPCCommand] = []

    def send(command: IPCCommand) -> dict[str, object]:
        requests.append(command)
        if command.payload.get("active"):
            raise RuntimeError("等待暂停确认超时")
        return {}

    controller = ScreenshotHotkeyStatusController(page, send)
    try:
        page.hotkeyRecordingChanged.emit(True)
        assert "等待暂停确认超时" in page.agent_status_label.text()
        owner = requests[-1].payload["owner"]
        controller.close()
        assert requests[-1].payload == {"owner": owner, "active": False}
    finally:
        controller.close()
        _dispose_page(page)


def test_client_adapter_surfaces_rejected_commands_and_recovers_status() -> None:
    class Transport:
        fail = True

        def send(self, command: IPCCommand, *, timeout_ms: int) -> IPCResponse:
            assert timeout_ms == 250
            return IPCResponse(
                request_id=command.request_id,
                ok=not self.fail,
                result={} if self.fail else {"running": True, "hotkeys": []},
                error="旧版工具不支持快捷键诊断" if self.fail else "",
            )

    page = ScreenshotSettingsPage()
    transport = Transport()
    controller = ScreenshotHotkeyStatusController.for_client(page, ScreenshotAgentClient(transport=transport))
    try:
        page.show()
        controller.refresh()
        assert "旧版工具不支持快捷键诊断" in page.agent_status_label.text()
        transport.fail = False
        controller.refresh()
        assert "正在运行" in page.agent_status_label.text()
        assert "旧版工具" not in page.agent_status_label.text()
    finally:
        controller.close()
        _dispose_page(page)


def test_legacy_status_without_diagnostic_rows_does_not_erase_edited_binding() -> None:
    page = ScreenshotSettingsPage()
    controller = ScreenshotHotkeyStatusController(page, lambda command: {"running": True})
    try:
        page.show()
        before = page.settings().hotkeys
        controller.refresh()
        assert page.settings().hotkeys == before
        assert "尚未检测" in page.hotkey_status_labels[CaptureMode.REGION].text()
    finally:
        controller.close()
        _dispose_page(page)


def test_direct_page_destruction_releases_recording_owner_immediately() -> None:
    page = ScreenshotSettingsPage()
    requests: list[IPCCommand] = []
    controller = ScreenshotHotkeyStatusController(page, lambda command: requests.append(command) or {})
    page.hotkeyRecordingChanged.emit(True)
    owner = requests[-1].payload["owner"]
    page.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    assert requests[-1].payload == {"owner": owner, "active": False}
    # Keep the wrapper alive until after Qt destruction so this test exercises
    # parent-owned QObject teardown, rather than Python reference collection.
    assert controller is not None


def test_stopped_agent_allows_offline_recording_and_settings_save() -> None:
    class Transport:
        def send(self, command: IPCCommand, *, timeout_ms: int) -> IPCResponse:
            raise ScreenshotAgentUnavailableError("截图工具尚未运行")

    page = ScreenshotSettingsPage()
    controller = ScreenshotHotkeyStatusController.for_client(page, ScreenshotAgentClient(transport=Transport()))
    try:
        page.show()
        QApplication.processEvents()
        edit = page.hotkey_edits[CaptureMode.REGION]
        edit.setFocus()
        QApplication.processEvents()
        assert edit.hasFocus()
        QApplication.sendEvent(edit, QKeyEvent(
            QEvent.Type.KeyPress, Qt.Key.Key_F8, Qt.KeyboardModifier.ControlModifier,
        ))
        controller.finish_recording()
        assert page.settings().hotkeys[CaptureMode.REGION].sequence == "Ctrl+F8"
    finally:
        controller.close()
        _dispose_page(page)


def test_pause_failure_remains_visible_after_automatic_focus_out_and_status_refresh() -> None:
    page = ScreenshotSettingsPage()
    requests: list[IPCCommand] = []

    def send(command: IPCCommand) -> dict[str, object]:
        requests.append(command)
        if command.command is CommandType.HOTKEY_RECORDING and command.payload["active"]:
            raise RuntimeError("模拟暂停确认超时")
        return {"running": True, "registered_hotkeys": 5, "hotkeys_paused": False, "hotkeys": []}

    controller = ScreenshotHotkeyStatusController(page, send)
    try:
        page.show()
        QApplication.processEvents()
        edit = page.hotkey_edits[CaptureMode.REGION]
        edit.setFocus()
        QApplication.processEvents()
        assert not edit.hasFocus()
        assert requests[-1].command is CommandType.HOTKEY_RECORDING
        assert requests[-1].payload["active"] is False
        assert "模拟暂停确认超时" in page.agent_status_label.text()
        assert "模拟暂停确认超时" in page.hotkey_recording_hint.text()
        controller.refresh()
        assert "模拟暂停确认超时" in page.agent_status_label.text()
        assert "模拟暂停确认超时" in page.hotkey_recording_hint.text()
        page.retry_hotkeys_button.click()
        assert requests[-1].command is CommandType.RETRY_HOTKEYS
        assert "模拟暂停确认超时" not in page.agent_status_label.text()
        assert "模拟暂停确认超时" not in page.hotkey_recording_hint.text()
    finally:
        controller.close()
        _dispose_page(page)


def test_registration_left_in_place_is_not_treated_as_confirmed_pause() -> None:
    page = ScreenshotSettingsPage()

    def send(command: IPCCommand) -> dict[str, object]:
        return {
            "running": True, "hotkeys_paused": command.payload.get("active", False),
            "registered_hotkeys": 1,
            "hotkeys": [{"mode": "region", "effective": "Print", "state": "restored"}],
        }

    controller = ScreenshotHotkeyStatusController(page, send)
    try:
        page.show()
        QApplication.processEvents()
        edit = page.hotkey_edits[CaptureMode.REGION]
        edit.setFocus()
        QApplication.processEvents()
        assert not edit.hasFocus()
        assert "未能释放原快捷键" in page.hotkey_recording_hint.text()
    finally:
        controller.close()
        _dispose_page(page)


def test_no_lease_attempt_means_finish_and_close_never_contact_the_agent() -> None:
    page = ScreenshotSettingsPage()
    requests: list[IPCCommand] = []
    controller = ScreenshotHotkeyStatusController(page, lambda command: requests.append(command) or {})
    try:
        before = page.settings().to_dict()
        page.hotkeyRecordingChanged.emit(False)
        controller.finish_recording()
        controller.close()
        # Main-window Save runs after dialog.finished has closed the bridge.
        controller.finish_recording()
        controller.retry()
        assert page.settings().to_dict() == before
        assert not requests
    finally:
        controller.close()
        _dispose_page(page)


def test_uncertain_release_is_retried_before_registration_retry() -> None:
    page = ScreenshotSettingsPage()
    requests: list[IPCCommand] = []
    release_failures = 1

    def send(command: IPCCommand) -> dict[str, object]:
        nonlocal release_failures
        requests.append(command)
        if command.command is CommandType.HOTKEY_RECORDING and not command.payload["active"]:
            if release_failures:
                release_failures -= 1
                raise RuntimeError("恢复确认超时")
        return {"running": True, "hotkeys": []}

    controller = ScreenshotHotkeyStatusController(page, send)
    try:
        page.hotkeyRecordingChanged.emit(True)
        page.hotkeyRecordingChanged.emit(False)
        assert "恢复确认超时" in page.hotkey_recording_hint.text()
        controller.retry()
        assert requests[-2].command is CommandType.HOTKEY_RECORDING
        assert requests[-2].payload["active"] is False
        assert requests[-1].command is CommandType.RETRY_HOTKEYS
        assert "恢复确认超时" not in page.hotkey_recording_hint.text()
    finally:
        controller.close()
        _dispose_page(page)
