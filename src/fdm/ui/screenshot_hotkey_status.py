from __future__ import annotations

from collections.abc import Callable, Mapping
from uuid import uuid4

from PySide6.QtCore import QObject, QTimer

from fdm.screenshot_protocol import CommandType, IPCCommand


class ScreenshotHotkeyStatusController(QObject):
    """Keep settings diagnostics live and renew a bounded recording lease.

    The same bridge is used by the main app over IPC and by the tray dialog
    directly. A crashed settings process leaves at most a 15 second lease.
    """

    def __init__(self, page, send: Callable[[IPCCommand], Mapping[str, object]]) -> None:
        super().__init__(page)
        self._page = page
        self._send = send
        self._owner = uuid4().hex
        self._recording = False
        self._lease_attempted = False
        self._recording_error = ""
        self._closed = False
        self._timer = QTimer(self)
        self._timer.setInterval(2_000)
        self._timer.timeout.connect(self.refresh)
        page.hotkeyRecordingChanged.connect(self._recording_changed)
        page.retryHotkeysRequested.connect(self.retry)
        page.destroyed.connect(self.close)
        self._timer.start()

    @classmethod
    def for_client(cls, page, client):
        from fdm.services.screenshot_agent_client import ScreenshotAgentUnavailableError

        def send(command):
            try:
                return dict(client.send(command, timeout_ms=250).result)
            except ScreenshotAgentUnavailableError:
                # No connection was established. Preferences can be edited
                # before starting the companion. Response timeouts remain
                # errors because the pause request may already be in flight.
                return {"running": False, "hotkeys": [], "offline": True}

        return cls(page, send)

    def _lease(self, active: bool) -> Mapping[str, object]:
        # Keep an uncertain active request until an explicit release succeeds;
        # an acknowledgement timeout does not prove the request was rejected.
        if active:
            self._lease_attempted = True
        result = self._send(IPCCommand(CommandType.HOTKEY_RECORDING, payload={"owner": self._owner, "active": active}))
        if result.get("offline"):
            self._lease_attempted = False
            return result
        if active and (result.get("hotkeys_paused") is False or result.get("registered_hotkeys", 0)):
            raise RuntimeError("未能释放原快捷键，暂停录入尚未确认，请重试。")
        if not active:
            self._lease_attempted = False
        return result

    def _show(self, result: Mapping[str, object]) -> None:
        if result.get("offline"):
            detail = "截图工具未连接；当前可编辑配置，保存并启动后生效。"
            if self._recording_error:
                detail += f"；{self._recording_error}"
            self._page.set_agent_status(False, detail)
            self._page.set_hotkey_diagnostics(())
            self._restore_recording_error_hint()
            return
        detail = "正在录入，已暂停全局快捷键" if result.get("hotkeys_paused") else ""
        if self._recording_error:
            detail = f"{detail}；{self._recording_error}" if detail else self._recording_error
        self._page.set_agent_status(True, detail)
        self._page.set_hotkey_diagnostics(result.get("hotkeys", ()))
        self._restore_recording_error_hint()

    def _restore_recording_error_hint(self) -> None:
        if self._recording_error:
            self._page.hotkey_recording_hint.setText(self._recording_error)

    def _recording_failed(self, exc: Exception, *, active: bool) -> None:
        if active:
            self._recording_error = f"暂停状态未确认，已结束本次录入，请先重新检测快捷键：{exc}"
            self._recording = False
        elif not self._recording_error:
            self._recording_error = f"快捷键恢复状态未确认，请重新检测快捷键：{exc}"
        self._page.set_agent_status(False, self._recording_error)
        if active:
            focus = self._page.focusWidget()
            if focus is not None:
                focus.clearFocus()
        self._restore_recording_error_hint()

    def _recording_changed(self, active: bool) -> None:
        if self._closed:
            return
        self._recording = active
        if active:
            self._recording_error = ""
        elif not self._lease_attempted:
            # Focus-out may arrive after Save/Apply has already released it.
            self._restore_recording_error_hint()
            return
        try:
            self._show(self._lease(active))
        except Exception as exc:
            self._recording_failed(exc, active=active)

    def refresh(self) -> None:
        if self._closed or not self._page.isVisible():
            return
        try:
            result = self._lease(True) if self._recording else self._send(IPCCommand(CommandType.STATUS))
            self._show(result)
        except Exception as exc:
            if self._recording:
                self._recording_failed(exc, active=True)
            else:
                self._page.set_agent_status(False, str(exc))
                self._restore_recording_error_hint()

    def retry(self) -> None:
        if self._closed:
            return
        try:
            if not self._recording and self._lease_attempted:
                self._lease(False)
            result = self._send(IPCCommand(CommandType.RETRY_HOTKEYS))
            had_error = bool(self._recording_error)
            self._recording_error = ""
            self._show(result)
            if had_error:
                self._page.hotkey_recording_hint.setText("已重新检测快捷键，可点击输入框重新录入。")
        except Exception as exc:
            self._page.set_agent_status(False, str(exc))
            self._restore_recording_error_hint()

    def finish_recording(self) -> None:
        """Release before Save/Apply, including the focus-out debounce turn."""
        if self._closed:
            return
        focus = self._page.focusWidget()
        if focus is not None:
            focus.clearFocus()
        self._recording = False
        if self._lease_attempted:
            self._show(self._lease(False))

    def close(self, *_args) -> None:
        if self._closed:
            return
        self._closed = True
        self._timer.stop()
        # Release only a possibly outstanding lease; merely viewing settings
        # must not contact an absent agent during Save/Cancel. A timed-out
        # active request still counts. Lease expiry is the final fallback.
        if not self._lease_attempted:
            return
        try:
            self._lease(False)
        except Exception:
            pass
