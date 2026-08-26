from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QCoreApplication, QEvent, QPoint, Qt
from PySide6.QtGui import QColor, QImage
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog, QScrollArea

from fdm.platform.windows_window_locator import PhysicalRect
from fdm.screenshot_agent import (
    AgentCommandError,
    SCREENSHOT_AUTOSTART_VALUE_NAME,
    ScreenshotAgent,
    ScreenshotCommandStream,
    ScreenshotSingleInstance,
    build_argument_parser,
    build_initial_command,
)
from fdm.screenshot_protocol import (
    CommandType,
    IPCCommand,
    ScreenshotProtocolError,
    decode_response,
    encode_ipc_message,
)
from fdm.screenshot_settings import HotkeyBinding, ScreenshotSettings, ScreenshotSettingsIO
from fdm.services.cu5_preview_locator import (
    Cu5PreviewAmbiguousError,
    Cu5PreviewNotFoundError,
    Cu5PreviewSelector,
)
from fdm.services.screenshot_capture import (
    CaptureCoordinator,
    CapturedFrame,
    CaptureMode,
    CaptureRect,
    CaptureRequest,
    ScreenInfo,
    WindowCandidate,
)
from fdm.services.screenshot_output import OutputResult
from fdm.ui.screenshot_settings_page import ScreenshotSettingsPage


def _app() -> QApplication:
    app = QApplication.instance() or QApplication([])
    app.setQuitOnLastWindowClosed(False)
    return app


@pytest.fixture(scope="module", autouse=True)
def _screenshot_qt_application() -> QApplication:
    """Create Qt before any QLocalServer/QSocketNotifier in this module.

    Constructing ScreenshotSingleInstance before QApplication leaves a native
    socket notifier attached to an invalid event dispatcher.  It may only
    surface later when QTest.qWait() processes the first queued capture.
    """

    app = _app()
    yield app


def _image() -> QImage:
    image = QImage(12, 8, QImage.Format.Format_ARGB32)
    image.fill(QColor("#345678"))
    return image


class _CaptureBackend:
    def screens(self):
        return (
            ScreenInfo(
                "primary",
                CaptureRect(0, 0, 800, 600),
                CaptureRect(0, 0, 800, 600),
                1.0,
                True,
            ),
        )

    def windows(self, *, include_children: bool = True):
        return ()

    def active_window_handle(self) -> int:
        return 0

    def capture_rect(self, rect: CaptureRect) -> QImage:
        return _image()

    def capture_window(self, candidate) -> QImage:
        return _image()


class _OutputService:
    def __init__(self) -> None:
        self.calls: list[tuple[QImage, ScreenshotSettings, CaptureMode]] = []

    def process_capture(self, image, settings, *, mode):
        self.calls.append((image.copy(), settings.normalized(), mode))
        return OutputResult(notification_requested=False)


class _Hotkeys:
    def __init__(self, _hwnd: int) -> None:
        self._bindings: dict[int, object] = {}
        self.closed = False
        self.reject_virtual_key = 0

    @property
    def bindings(self):
        return tuple(self._bindings.values())

    def binding(self, identifier: int):
        return self._bindings.get(identifier)

    def bind(self, binding):
        if binding.virtual_key == self.reject_virtual_key:
            raise RuntimeError("occupied")
        self._bindings[binding.identifier] = binding
        return binding

    def unbind(self, identifier: int):
        return self._bindings.pop(identifier, None) is not None

    def binding_for_message(self, _message, identifier, _lparam):
        return self.binding(identifier)

    def close(self):
        self.closed = True
        self._bindings.clear()


class _Autostart:
    def __init__(self) -> None:
        self.values: list[bool] = []

    def set_enabled(self, enabled: bool):
        self.values.append(bool(enabled))
        return SimpleNamespace(enabled=enabled)


def _record(hwnd: int = 77):
    return SimpleNamespace(
        hwnd=hwnd,
        rect=PhysicalRect(-40, 20, 728, 596),
    )


class _Locator:
    def locate(self):
        return SimpleNamespace(
            record=_record(),
            score=211.5,
            reasons=("匹配 CWndForSDK", "匹配 4:3"),
        )


def test_command_stream_accepts_chunks_and_rejects_invalid_json() -> None:
    first = IPCCommand(CommandType.STATUS, request_id="one")
    second = IPCCommand.capture(CaptureMode.CU5, request_id="two")
    payload = encode_ipc_message(first) + encode_ipc_message(second)
    stream = ScreenshotCommandStream()

    assert stream.feed(payload[:11]) == ()
    assert stream.feed(payload[11:]) == (first, second)
    assert stream.pending_bytes == 0
    with pytest.raises(ScreenshotProtocolError):
        ScreenshotCommandStream().feed(b"{broken}\n")


def test_ipc_handler_response_keeps_request_id_and_surfaces_result(tmp_path: Path) -> None:
    instance = ScreenshotSingleInstance(
        f"test-screenshot-{uuid4().hex}",
        lock_file_path=tmp_path / "agent.lock",
    )

    class _Socket:
        def __init__(self, payload: bytes) -> None:
            self.payload = payload
            self.written = bytearray()

        def readAll(self):
            payload, self.payload = self.payload, b""
            return payload

        def write(self, payload):
            self.written.extend(bytes(payload))
            return len(payload)

        def flush(self):
            return True

        def abort(self):
            pass

        def deleteLater(self):
            pass

    command = IPCCommand(CommandType.STATUS, request_id="status-1")
    socket = _Socket(encode_ipc_message(command))
    instance._connections[id(socket)] = (socket, ScreenshotCommandStream())
    instance.set_command_handler(lambda _command: {"running": True})
    instance._consume(id(socket))
    response = decode_response(bytes(socket.written))

    assert response.request_id == "status-1"
    assert response.ok and response.result == {"running": True}

    failing = IPCCommand(CommandType.DIAGNOSE_CU5, request_id="diag-2")
    socket2 = _Socket(encode_ipc_message(failing))
    instance._connections[id(socket2)] = (socket2, ScreenshotCommandStream())

    def reject(_command):
        raise AgentCommandError("ambiguous", result={"candidates": [{"hwnd": 9}]})

    instance.set_command_handler(reject)
    instance._consume(id(socket2))
    failed = decode_response(bytes(socket2.written))
    assert failed.request_id == "diag-2"
    assert not failed.ok and failed.error == "ambiguous"
    assert failed.result["candidates"][0]["hwnd"] == 9
    instance.close()


def test_single_instance_elects_a_qlocalserver_primary(tmp_path: Path) -> None:
    instance = ScreenshotSingleInstance(
        f"fdm-shot-{uuid4().hex[:8]}",
        lock_file_path=tmp_path / "primary.lock",
    )
    try:
        result = instance.start_or_forward(
            IPCCommand(CommandType.STATUS),
            timeout_ms=5,
        )
        assert result.primary and not result.forwarded
        assert instance.is_listening
    finally:
        instance.close()


def test_agent_loads_settings_rebinds_hotkeys_syncs_autostart_and_persists_region(
    tmp_path: Path,
    monkeypatch,
) -> None:
    app = _app()
    path = tmp_path / "settings.json"
    settings = ScreenshotSettings(
        enabled=True,
        autostart=True,
        show_editor=False,
        notification=False,
        include_cursor=True,
    )
    ScreenshotSettingsIO.save(settings, path)
    output = _OutputService()
    hotkeys = _Hotkeys(1)
    autostart = _Autostart()
    coordinator = CaptureCoordinator(_CaptureBackend())
    agent = ScreenshotAgent(
        app,
        coordinator,
        settings_path=path,
        output_service=output,
        cu5_locator=_Locator(),
        hotkey_manager_factory=lambda _hwnd: hotkeys,
        autostart_manager_factory=lambda: autostart,
    )
    try:
        agent.start()
        assert len(hotkeys.bindings) == 5
        assert autostart.values[-1] is True
        assert agent.status()["registered_hotkeys"] == 5
        assert agent._request(CaptureMode.REGION).include_cursor is True
        ipc_requests = []
        monkeypatch.setattr(coordinator, "start", ipc_requests.append)
        response = agent.handle_command(IPCCommand.capture(CaptureMode.REGION))
        assert response["accepted"] is True and response["queued"] == 1
        assert ipc_requests == []
        QTest.qWait(1)
        assert ipc_requests[-1].include_cursor is True

        hotkeys.reject_virtual_key = 0x24  # VK_HOME
        updated = agent.handle_command(
            IPCCommand(
                CommandType.UPDATE_SETTINGS,
                payload={
                    "hotkeys": {
                        CaptureMode.REGION.value: HotkeyBinding("Ctrl+Home").to_dict(),
                    },
                    "autostart": False,
                },
            )
        )
        assert any("occupied" in item for item in updated["hotkey_errors"])
        assert len(hotkeys.bindings) == 5  # old region binding was retained
        assert (
            ScreenshotSettingsIO.load(path).hotkeys[CaptureMode.REGION].sequence
            == "Print"
        )
        assert autostart.values[-1] is False

        coordinator.set_last_region(CaptureRect(-300, 40, 120, 80))
        agent._capture_ready(
            CapturedFrame(
                _image(),
                CaptureRect(-300, 40, 120, 80),
                CaptureMode.REGION,
                metadata={"open_editor": False},
            )
        )
        assert len(output.calls) == 1
        assert ScreenshotSettingsIO.load(path).last_region == CaptureRect(-300, 40, 120, 80)
    finally:
        agent.close()
    assert hotkeys.closed


def test_ipc_capture_returns_before_backend_runs_and_dispatches_next_turn(
    tmp_path: Path,
) -> None:
    app = _app()

    class _CountingBackend(_CaptureBackend):
        def __init__(self) -> None:
            self.capture_calls: list[CaptureRect] = []

        def capture_rect(self, rect: CaptureRect) -> QImage:
            self.capture_calls.append(rect)
            return _image()

    backend = _CountingBackend()
    agent = ScreenshotAgent(
        app,
        CaptureCoordinator(backend),
        settings_path=tmp_path / "settings.json",
        output_service=_OutputService(),
        cu5_locator=_Locator(),
    )
    try:
        response = agent.handle_command(
            IPCCommand.capture(CaptureMode.FULL_SCREEN)
        )

        assert response == {
            "accepted": True,
            "mode": CaptureMode.FULL_SCREEN.value,
            "queued": 1,
        }
        assert backend.capture_calls == []

        QTest.qWait(1)

        assert backend.capture_calls == [CaptureRect(0, 0, 800, 600)]
    finally:
        agent.close()


def test_ipc_capture_queue_preserves_order_and_parented_timer_is_destroy_safe(
    tmp_path: Path,
) -> None:
    app = _app()
    agent = ScreenshotAgent(
        app,
        CaptureCoordinator(_CaptureBackend()),
        settings_path=tmp_path / "settings.json",
        output_service=_OutputService(),
        cu5_locator=_Locator(),
    )
    dispatched: list[CaptureMode] = []
    agent.begin_capture = lambda request: dispatched.append(request.mode)  # type: ignore[method-assign]

    agent.handle_command(IPCCommand.capture(CaptureMode.FULL_SCREEN))
    agent.handle_command(IPCCommand.capture(CaptureMode.DISPLAY))
    agent.handle_command(IPCCommand.capture(CaptureMode.CU5))

    assert dispatched == []
    assert agent._ipc_capture_timer.parent() is agent
    QTest.qWait(20)
    assert dispatched == [
        CaptureMode.FULL_SCREEN,
        CaptureMode.DISPLAY,
        CaptureMode.CU5,
    ]

    agent.handle_command(IPCCommand.capture(CaptureMode.FULL_SCREEN))
    assert dispatched[-1] is CaptureMode.CU5
    agent.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    QTest.qWait(2)

    assert dispatched[-1] is CaptureMode.CU5


def test_agent_ui_requests_keep_all_overlay_candidates_and_settle_tray_menu(
    tmp_path: Path,
) -> None:
    app = _app()
    coordinator = CaptureCoordinator(_CaptureBackend())
    agent = ScreenshotAgent(
        app,
        coordinator,
        settings_path=tmp_path / "settings.json",
        output_service=_OutputService(),
        cu5_locator=_Locator(),
    )
    try:
        assert agent._request(CaptureMode.REGION).cursor_position is None
        assert agent._request(CaptureMode.SMART).cursor_position is None
        assert agent._request(CaptureMode.WINDOW).cursor_position is None
        assert agent._request(CaptureMode.DISPLAY).cursor_position is not None

        with patch.object(coordinator, "start") as start:
            for mode in (
                CaptureMode.REGION,
                CaptureMode.SMART,
                CaptureMode.WINDOW,
                CaptureMode.ACTIVE_WINDOW,
            ):
                agent._begin_tray_capture(mode)
                assert start.call_args.args[0].delay_ms >= 150

        menu_labels = [action.text() for action in agent.tray.contextMenu().actions()]
        assert "设置…" in menu_labels
        assert "CU 系列实时预览" in menu_labels
    finally:
        agent.close()


def test_new_capture_retires_old_overlay_and_stale_finish_keeps_request_identity(
    tmp_path: Path,
) -> None:
    app = _app()
    coordinator = CaptureCoordinator(_CaptureBackend())
    agent = ScreenshotAgent(
        app,
        coordinator,
        settings_path=tmp_path / "settings.json",
        output_service=_OutputService(),
        cu5_locator=_Locator(),
    )

    class _Overlay:
        hidden = False
        closed = False

        def hide(self) -> None:
            self.hidden = True

        def close(self) -> None:
            self.closed = True

    old_overlay = _Overlay()
    agent._overlay = old_overlay  # type: ignore[assignment]
    try:
        with patch.object(coordinator, "start") as start:
            agent.begin_capture(CaptureRequest(CaptureMode.SMART))
        queued = start.call_args.args[0]
        assert old_overlay.hidden and old_overlay.closed
        assert queued.delay_ms >= 80

        request = CaptureRequest(CaptureMode.SMART)
        with patch.object(coordinator, "complete_selection") as complete:
            agent._show_selection_overlay(request, ())
            assert agent._overlay is not None
            agent._overlay.accept_region(CaptureRect(2, 3, 10, 8))
            QTest.qWait(110)
        assert complete.call_args.kwargs["expected_request"] is request
    finally:
        if agent._overlay is not None:
            agent._overlay.close()
        agent.close()


@pytest.mark.parametrize(
    ("candidates", "click_position", "expected_handle"),
    (
        (
            (
                WindowCandidate(
                    100,
                    CaptureRect(50, 50, 500, 400),
                    z_order=0,
                    metadata={"root_handle": 100},
                ),
                WindowCandidate(
                    101,
                    CaptureRect(100, 100, 200, 120),
                    parent_handle=100,
                    depth=1,
                    z_order=1,
                    metadata={
                        "root_handle": 100,
                        "ancestor_handles": (100,),
                    },
                ),
            ),
            QPoint(150, 150),
            101,
        ),
        (
            (
                WindowCandidate(
                    200,
                    CaptureRect(0, 560, 800, 40),
                    z_order=0,
                    metadata={"root_handle": 200},
                ),
            ),
            QPoint(200, 580),
            200,
        ),
    ),
    ids=("nested-window", "taskbar-style-root"),
)
def test_smart_overlay_real_click_reaches_capture_ready(
    tmp_path: Path,
    candidates: tuple[WindowCandidate, ...],
    click_position: QPoint,
    expected_handle: int,
) -> None:
    app = _app()

    class _CandidateBackend(_CaptureBackend):
        def windows(self, *, include_children: bool = True):
            return candidates

    coordinator = CaptureCoordinator(_CandidateBackend())
    agent = ScreenshotAgent(
        app,
        coordinator,
        settings_path=tmp_path / "settings.json",
        output_service=_OutputService(),
        cu5_locator=_Locator(),
    )
    ready: list[CapturedFrame] = []
    coordinator.captureReady.connect(ready.append)
    try:
        agent.begin_capture(CaptureRequest(CaptureMode.SMART))
        overlay = agent._overlay
        assert overlay is not None
        QTest.mouseMove(overlay, click_position)
        assert overlay.selected_candidate is not None
        assert overlay.selected_candidate.handle == expected_handle

        QTest.mouseClick(
            overlay,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            click_position,
        )
        QTest.qWait(110)

        assert len(ready) == 1
        assert ready[0].target_handle == expected_handle
        assert agent._overlay is None
    finally:
        if agent._overlay is not None:
            agent._overlay.close()
        agent.close()


def test_closing_accepted_overlay_destroys_its_deferred_capture_timer(
    tmp_path: Path,
) -> None:
    app = _app()
    coordinator = CaptureCoordinator(_CaptureBackend())
    agent = ScreenshotAgent(
        app,
        coordinator,
        settings_path=tmp_path / "settings.json",
        output_service=_OutputService(),
        cu5_locator=_Locator(),
    )
    request = CaptureRequest(CaptureMode.SMART)
    with patch.object(coordinator, "complete_selection") as complete:
        agent._show_selection_overlay(request, ())
        overlay = agent._overlay
        assert overlay is not None
        overlay.accept_region(CaptureRect(2, 3, 10, 8))
        overlay.close()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        QTest.qWait(110)

    assert not complete.called
    assert agent._overlay is None
    agent.close()


def test_standalone_settings_scroll_and_disabling_quits_after_accept(
    tmp_path: Path,
) -> None:
    app = _app()
    agent = ScreenshotAgent(
        app,
        CaptureCoordinator(_CaptureBackend()),
        settings_path=tmp_path / "settings.json",
        output_service=_OutputService(),
        cu5_locator=_Locator(),
    )
    try:
        agent._show_settings_window()
        dialog = agent._settings_window
        assert dialog is not None
        scroll = dialog.findChild(QScrollArea)
        assert scroll is not None and scroll.widget() is not None
        assert scroll.property("redirectEditorWheel") is True
        page = scroll.widget()
        page.resident_checkbox.setChecked(False)

        with patch("fdm.screenshot_agent.QTimer.singleShot") as single_shot:
            agent._save_settings_page(page, dialog)

        assert dialog.result() == QDialog.DialogCode.Accepted
        single_shot.assert_called_once()
        assert single_shot.call_args.args[0] == 0
        assert single_shot.call_args.args[1] == app.quit
    finally:
        if agent._settings_window is not None:
            agent._settings_window.close()
        agent.close()


def test_conflicting_hotkey_update_keeps_old_binding_after_agent_recreation(
    tmp_path: Path,
) -> None:
    app = _app()
    path = tmp_path / "settings.json"
    ScreenshotSettingsIO.save(ScreenshotSettings(enabled=True), path)

    first_hotkeys = _Hotkeys(1)
    first = ScreenshotAgent(
        app,
        CaptureCoordinator(_CaptureBackend()),
        settings_path=path,
        output_service=_OutputService(),
        cu5_locator=_Locator(),
        hotkey_manager_factory=lambda _hwnd: first_hotkeys,
        autostart_manager_factory=_Autostart,
    )
    first.start()
    try:
        first_hotkeys.reject_virtual_key = 0x24  # VK_HOME
        result = first.reload_settings(
            {
                "hotkeys": {
                    CaptureMode.REGION.value: HotkeyBinding("Ctrl+Home").to_dict(),
                }
            }
        )
        assert result["hotkey_errors"]
        assert first_hotkeys.binding(0x5F01).virtual_key == 0x2C
        assert ScreenshotSettingsIO.load(path).hotkeys[CaptureMode.REGION].sequence == "Print"
    finally:
        first.close()

    second_hotkeys = _Hotkeys(1)
    second_hotkeys.reject_virtual_key = 0x24
    second = ScreenshotAgent(
        app,
        CaptureCoordinator(_CaptureBackend()),
        settings_path=path,
        output_service=_OutputService(),
        cu5_locator=_Locator(),
        hotkey_manager_factory=lambda _hwnd: second_hotkeys,
        autostart_manager_factory=_Autostart,
    )
    second.start()
    try:
        assert second_hotkeys.binding(0x5F01).virtual_key == 0x2C
        assert not second.integration_errors
    finally:
        second.close()


def test_settings_save_failure_rolls_back_hotkeys_and_autostart(
    tmp_path: Path,
    monkeypatch,
) -> None:
    app = _app()
    path = tmp_path / "settings.json"
    ScreenshotSettingsIO.save(
        ScreenshotSettings(enabled=True, autostart=True),
        path,
    )
    hotkeys = _Hotkeys(1)
    autostart = _Autostart()
    agent = ScreenshotAgent(
        app,
        CaptureCoordinator(_CaptureBackend()),
        settings_path=path,
        output_service=_OutputService(),
        cu5_locator=_Locator(),
        hotkey_manager_factory=lambda _hwnd: hotkeys,
        autostart_manager_factory=lambda: autostart,
    )
    agent.start()
    original_save_unlocked = ScreenshotSettingsIO._save_unlocked

    def fail_save_unlocked(_settings, _path):
        raise OSError("disk full")

    monkeypatch.setattr(ScreenshotSettingsIO, "_save_unlocked", fail_save_unlocked)
    try:
        with pytest.raises(AgentCommandError, match="disk full"):
            agent.reload_settings(
                {
                    "hotkeys": {
                        CaptureMode.REGION.value: HotkeyBinding("Ctrl+Home").to_dict(),
                    },
                    "autostart": False,
                }
            )

        assert hotkeys.binding(0x5F01).virtual_key == 0x2C
        assert autostart.values[-2:] == [False, True]
        assert agent.settings.hotkeys[CaptureMode.REGION].sequence == "Print"
        assert agent.settings.autostart is True
    finally:
        monkeypatch.setattr(
            ScreenshotSettingsIO,
            "_save_unlocked",
            original_save_unlocked,
        )
        agent.close()


def test_cu5_diagnostic_success_and_ambiguity_include_physical_candidate_details(
    tmp_path: Path,
) -> None:
    app = _app()
    agent = ScreenshotAgent(
        app,
        CaptureCoordinator(_CaptureBackend()),
        settings_path=tmp_path / "settings.json",
        output_service=_OutputService(),
        cu5_locator=_Locator(),
    )
    success = agent.handle_command(IPCCommand(CommandType.DIAGNOSE_CU5))
    assert success["hwnd"] == 77
    assert success["rect"] == {
        "x": -40,
        "y": 20,
        "width": 768,
        "height": 576,
        "coordinate_space": "physical_pixels",
    }
    assert success["score"] == 211.5 and len(success["reasons"]) == 2
    assert success["candidates"][0]["hwnd"] == 77

    candidate = SimpleNamespace(record=_record(99), score=150.0, reasons=("候选",))

    class _Ambiguous:
        def locate(self):
            raise Cu5PreviewAmbiguousError("多个候选", candidates=(candidate,))

    agent._cu5_locator = _Ambiguous()
    with pytest.raises(AgentCommandError) as caught:
        agent.diagnose_cu5()
    assert caught.value.result["diagnostic"] == "ambiguous"
    assert caught.value.result["candidates"][0]["hwnd"] == 99
    agent.close()


def test_agent_updates_locator_from_settings_and_persists_only_stable_cu5_signature(
    tmp_path: Path,
) -> None:
    app = _app()
    path = tmp_path / "settings.json"
    ScreenshotSettingsIO.save(
        ScreenshotSettings(
            cu5_selector={
                "process_name": "CU-5.exe",
                "class_name": "CWndForSDK",
                "control_id": 1201,
                "hwnd": 999,
            }
        ),
        path,
    )
    stable = Cu5PreviewSelector(
        process_name="CU-5.exe",
        class_name="CWndForSDK",
        control_id=1301,
        width=768,
        height=576,
        ancestor_classes=("Afx:00400000:b:1", "MDIClient"),
    )

    class _TrackingLocator:
        def __init__(self) -> None:
            self.selectors: list[dict[str, object]] = []

        def set_selector(self, value) -> None:
            self.selectors.append(dict(value))

        def locate(self):
            return SimpleNamespace(
                record=_record(88),
                score=300.0,
                reasons=("selector",),
                selector=stable,
            )

    locator = _TrackingLocator()
    agent = ScreenshotAgent(
        app,
        CaptureCoordinator(_CaptureBackend()),
        settings_path=path,
        output_service=_OutputService(),
        cu5_locator=locator,
    )
    try:
        assert locator.selectors[-1]["control_id"] == 1201

        result = agent.diagnose_cu5()

        assert result["hwnd"] == 88
        saved = ScreenshotSettingsIO.load(path).cu5_selector
        assert saved == stable.normalized().to_dict()
        assert not ({"hwnd", "pid", "x", "y", "left", "top"} & saved.keys())
        assert locator.selectors[-1] == saved

        agent.reload_settings(
            {
                "cu5_selector": {
                    "process_name": "cu-5.exe",
                    "class_name": "cwndforsdk",
                    "control_id": 1401,
                }
            }
        )
        assert locator.selectors[-1]["control_id"] == 1401
    finally:
        agent.close()


def test_standalone_settings_restores_selector_when_adjustment_validation_fails(
    tmp_path: Path,
) -> None:
    app = _app()
    path = tmp_path / "settings.json"
    previous = {
        "process_name": "cu.exe",
        "class_name": "cwndforsdk",
        "control_id": 1201,
    }
    ScreenshotSettingsIO.save(ScreenshotSettings(cu5_selector=previous), path)

    class _FailingLocator:
        def __init__(self) -> None:
            self.selector: dict[str, object] = {}

        def set_selector(self, value) -> None:
            self.selector = dict(value)

        def locate(self):
            if self.selector.get("control_id") == 1501:
                raise Cu5PreviewNotFoundError("所选对象已不可用")
            return SimpleNamespace(
                record=_record(88),
                score=200.0,
                reasons=("原对象",),
                selector=Cu5PreviewSelector.from_value(self.selector),
            )

    locator = _FailingLocator()
    agent = ScreenshotAgent(
        app,
        CaptureCoordinator(_CaptureBackend()),
        settings_path=path,
        output_service=_OutputService(),
        cu5_locator=locator,
    )
    try:
        agent._show_settings_window()
        dialog = agent._settings_window
        assert dialog is not None
        page = dialog.findChild(ScreenshotSettingsPage)
        assert page is not None
        page.cu5CandidateSelectionRequested.emit(
            {
                "process_name": "cu-6.exe",
                "class_name": "static",
                "control_id": 1501,
            }
        )

        assert agent.settings.cu5_selector == previous
        assert locator.selector == previous
        assert ScreenshotSettingsIO.load(path).cu5_selector == previous
        assert "已恢复原预览对象" in page.cu5_status_label.text()
    finally:
        if agent._settings_window is not None:
            agent._settings_window.close()
        agent.close()


def test_successful_cu5_capture_persists_backend_locator_signature(tmp_path: Path) -> None:
    app = _app()
    path = tmp_path / "settings.json"
    backend = _CaptureBackend()
    backend.last_cu5_match = SimpleNamespace(
        selector=Cu5PreviewSelector(
            process_name="CU-5.exe",
            class_name="CWndForSDK",
            control_id=1501,
            width=768,
            height=576,
        )
    )
    agent = ScreenshotAgent(
        app,
        CaptureCoordinator(backend),
        settings_path=path,
        output_service=_OutputService(),
        cu5_locator=_Locator(),
    )
    try:
        agent._capture_ready(
            CapturedFrame(
                _image(),
                CaptureRect(0, 0, 768, 576),
                CaptureMode.CU5,
                metadata={"open_editor": False},
            )
        )
        assert ScreenshotSettingsIO.load(path).cu5_selector["control_id"] == 1501
    finally:
        agent.close()


def test_editor_completion_publishes_edited_image_through_output_service(
    tmp_path: Path,
) -> None:
    app = _app()
    output = _OutputService()
    agent = ScreenshotAgent(
        app,
        CaptureCoordinator(_CaptureBackend()),
        settings_path=tmp_path / "settings.json",
        output_service=output,
        cu5_locator=_Locator(),
    )
    agent._capture_ready(
        CapturedFrame(
            _image(),
            CaptureRect(0, 0, 12, 8),
            CaptureMode.REGION,
            metadata={"open_editor": True},
        )
    )
    editor = agent.annotation_session
    assert editor is not None
    editor.request_complete()

    assert len(output.calls) == 1
    assert output.calls[0][2] is CaptureMode.REGION
    agent.close()


def test_missing_screen_mapping_uses_managed_fallback_editor(tmp_path: Path) -> None:
    from fdm.ui.screenshot_editor import ScreenshotEditor

    app = _app()

    class _NoScreensBackend(_CaptureBackend):
        def screens(self):
            return ()

    agent = ScreenshotAgent(
        app,
        CaptureCoordinator(_NoScreensBackend()),
        settings_path=tmp_path / "settings.json",
        output_service=_OutputService(),
        cu5_locator=_Locator(),
    )
    try:
        agent._capture_ready(
            CapturedFrame(
                _image(),
                CaptureRect(0, 0, 12, 8),
                CaptureMode.REGION,
                metadata={"open_editor": True},
            )
        )

        assert isinstance(agent.annotation_session, ScreenshotEditor)
        assert agent.annotation_session._managed_output is True
    finally:
        agent.close()


def test_partial_output_failure_warns_but_still_completes_capture(
    tmp_path: Path,
) -> None:
    app = _app()

    class _PartialOutput:
        def process_capture(self, _image, _settings, *, mode):
            assert mode is CaptureMode.REGION
            return OutputResult(
                copied_to_clipboard=True,
                notification_requested=False,
                errors=("保存文件失败：disk full",),
            )

    agent = ScreenshotAgent(
        app,
        CaptureCoordinator(_CaptureBackend()),
        settings_path=tmp_path / "settings.json",
        output_service=_PartialOutput(),
        cu5_locator=_Locator(),
    )
    try:
        with patch.object(agent.tray, "showMessage") as show_message:
            assert agent._publish_capture(_image(), CaptureMode.REGION) is True
        assert show_message.call_args.args[0] == "截图部分完成"
        assert "disk full" in show_message.call_args.args[1]
    finally:
        agent.close()


def test_future_settings_schema_remains_read_only_during_capture(tmp_path: Path) -> None:
    app = _app()
    path = tmp_path / "future.json"
    original = '{"schema_version":999,"future":"keep"}'
    path.write_text(original, encoding="utf-8")
    coordinator = CaptureCoordinator(_CaptureBackend())
    hotkeys = _Hotkeys(1)
    autostart = _Autostart()
    agent = ScreenshotAgent(
        app,
        coordinator,
        settings_path=path,
        output_service=_OutputService(),
        cu5_locator=_Locator(),
        hotkey_manager_factory=lambda _hwnd: hotkeys,
        autostart_manager_factory=lambda: autostart,
    )
    agent.start()
    coordinator.set_last_region(CaptureRect(1, 2, 30, 40))
    agent._capture_ready(
        CapturedFrame(
            _image(),
            CaptureRect(1, 2, 30, 40),
            CaptureMode.REGION,
            metadata={"open_editor": False},
        )
    )

    assert agent.status()["settings_read_only"] is True
    assert hotkeys.bindings == ()
    assert autostart.values == []
    assert path.read_text(encoding="utf-8") == original
    with pytest.raises(AgentCommandError, match="只读"):
        agent.reload_settings({"delay_ms": 100})
    agent.close()


def test_runtime_region_update_preserves_main_process_fields(tmp_path: Path) -> None:
    app = _app()
    path = tmp_path / "settings.json"
    ScreenshotSettingsIO.save(
        ScreenshotSettings(enabled=True, output_directory="before"),
        path,
    )
    coordinator = CaptureCoordinator(_CaptureBackend())
    agent = ScreenshotAgent(
        app,
        coordinator,
        settings_path=path,
        output_service=_OutputService(),
        cu5_locator=_Locator(),
    )
    try:
        # Simulate the main process disabling the resident tool after this
        # agent loaded its own in-memory snapshot.
        ScreenshotSettingsIO.update(
            lambda persisted: replace(
                persisted,
                enabled=False,
                output_directory="updated-by-main",
            ),
            path,
        )
        coordinator.set_last_region(CaptureRect(4, 5, 60, 70))

        agent._capture_ready(
            CapturedFrame(
                _image(),
                CaptureRect(4, 5, 60, 70),
                CaptureMode.REGION,
                metadata={"open_editor": False},
            )
        )

        persisted = ScreenshotSettingsIO.load(path)
        assert persisted.enabled is False
        assert persisted.output_directory == "updated-by-main"
        assert persisted.last_region == CaptureRect(4, 5, 60, 70)
    finally:
        agent.close()


def test_cli_background_show_settings_and_stable_autostart_identity() -> None:
    parser = build_argument_parser()
    assert build_initial_command(parser.parse_args(["--background"])).command is CommandType.STATUS
    assert (
        build_initial_command(parser.parse_args(["--show-settings"])).command
        is CommandType.SHOW_SETTINGS
    )
    assert SCREENSHOT_AUTOSTART_VALUE_NAME == "FiberDiameterMeasurementScreenshotTool"


def test_cli_editor_switch_is_tristate_and_mutually_exclusive() -> None:
    parser = build_argument_parser()
    inherited = build_initial_command(parser.parse_args(["--capture", "region"]))
    enabled = build_initial_command(parser.parse_args(["--capture", "region", "--editor"]))
    disabled = build_initial_command(parser.parse_args(["--capture", "region", "--no-editor"]))

    assert "open_editor" not in inherited.payload
    assert enabled.payload["open_editor"] is True
    assert disabled.payload["open_editor"] is False
    with pytest.raises(SystemExit):
        parser.parse_args(["--capture", "region", "--editor", "--no-editor"])


def test_inline_annotation_mode_policy_and_instant_mode_bypass(tmp_path: Path) -> None:
    app = _app()
    path = tmp_path / "settings.json"
    ScreenshotSettingsIO.save(ScreenshotSettings(show_editor=True), path)
    output = _OutputService()
    agent = ScreenshotAgent(
        app,
        CaptureCoordinator(_CaptureBackend()),
        settings_path=path,
        output_service=output,
        cu5_locator=_Locator(),
    )
    try:
        for mode in (CaptureMode.CU5, CaptureMode.LAST_REGION):
            agent._capture_ready(
                CapturedFrame(
                    _image(),
                    CaptureRect(0, 0, 12, 8),
                    mode,
                    metadata={"open_editor": True},
                )
            )
            assert agent.annotation_session is None
        assert [call[2] for call in output.calls] == [CaptureMode.CU5, CaptureMode.LAST_REGION]

        agent._capture_ready(
            CapturedFrame(
                _image(),
                CaptureRect(0, 0, 12, 8),
                CaptureMode.FULL_SCREEN,
                metadata={"open_editor": True},
            )
        )
        session = agent.annotation_session
        assert session is not None
        session.request_cancel()
        QTest.qWait(1)
        assert agent.annotation_session is None
    finally:
        agent.close()


def test_explicit_editor_override_and_single_session_guard(tmp_path: Path, monkeypatch) -> None:
    app = _app()
    coordinator = CaptureCoordinator(_CaptureBackend())
    agent = ScreenshotAgent(
        app,
        coordinator,
        settings_path=tmp_path / "settings.json",
        output_service=_OutputService(),
        cu5_locator=_Locator(),
    )
    try:
        agent._capture_ready(
            CapturedFrame(
                _image(),
                CaptureRect(0, 0, 12, 8),
                CaptureMode.REGION,
                metadata={"open_editor": True},
            )
        )
        first = agent.annotation_session
        assert first is not None
        starts: list[CaptureRequest] = []
        monkeypatch.setattr(coordinator, "start", starts.append)

        agent.begin_capture(CaptureRequest(CaptureMode.FULL_SCREEN, open_editor=True))

        assert starts == []
        assert agent.annotation_session is first
        first.request_cancel()
        assert ScreenshotSettingsIO.load(tmp_path / "settings.json").last_region is None
    finally:
        agent.close()


def test_output_failure_retains_annotation_and_does_not_replace_last_region(tmp_path: Path) -> None:
    app = _app()

    class _FailingOutput:
        def process_capture(self, *_args, **_kwargs):
            raise RuntimeError("disk unavailable")

    path = tmp_path / "settings.json"
    agent = ScreenshotAgent(
        app,
        CaptureCoordinator(_CaptureBackend()),
        settings_path=path,
        output_service=_FailingOutput(),
        cu5_locator=_Locator(),
    )
    try:
        frame = CapturedFrame(
            _image(),
            CaptureRect(20, 30, 12, 8),
            CaptureMode.REGION,
            metadata={"open_editor": True},
        )
        agent._capture_ready(frame)
        session = agent.annotation_session
        assert session is not None

        session.request_complete()

        assert agent.annotation_session is session
        assert session.output_pending is False
        assert agent.coordinator.last_region is None
        assert ScreenshotSettingsIO.load(path).last_region is None
        session.request_cancel()
    finally:
        agent.close()


def test_partial_output_retains_annotation_for_retry_but_records_successful_region(
    tmp_path: Path,
) -> None:
    app = _app()

    class _PartialOutput:
        def process_capture(self, *_args, **_kwargs):
            return OutputResult(
                copied_to_clipboard=True,
                errors=("保存文件失败：disk full",),
            )

    path = tmp_path / "settings.json"
    agent = ScreenshotAgent(
        app,
        CaptureCoordinator(_CaptureBackend()),
        settings_path=path,
        output_service=_PartialOutput(),
        cu5_locator=_Locator(),
    )
    frame = CapturedFrame(
        _image(),
        CaptureRect(20, 30, 12, 8),
        CaptureMode.REGION,
        metadata={"open_editor": True},
    )
    try:
        agent._capture_ready(frame)
        session = agent.annotation_session
        assert session is not None

        session.request_complete()

        assert agent.annotation_session is session
        assert session.output_pending is False
        assert agent.coordinator.last_region == frame.rect
        assert ScreenshotSettingsIO.load(path).last_region == frame.rect
        session.request_cancel()
    finally:
        agent.close()


def test_recent_annotation_style_persists_across_sessions(tmp_path: Path) -> None:
    from fdm.ui.screenshot_editor import EditorTool

    app = _app()
    path = tmp_path / "settings.json"
    agent = ScreenshotAgent(
        app,
        CaptureCoordinator(_CaptureBackend()),
        settings_path=path,
        output_service=_OutputService(),
        cu5_locator=_Locator(),
    )
    try:
        agent._capture_ready(
            CapturedFrame(
                _image(),
                CaptureRect(0, 0, 12, 8),
                CaptureMode.REGION,
                metadata={"open_editor": True},
            )
        )
        session = agent.annotation_session
        assert session is not None
        session.set_tool(EditorTool.ARROW)
        session.arrow_spin.setValue(27)
        session.width_spin.setValue(6)
        QTest.qWait(300)

        persisted = ScreenshotSettingsIO.load(path).annotation_styles
        assert persisted["active_tool"] == "arrow"
        assert persisted["tools"]["arrow"]["arrow_size"] == 27
        assert persisted["tools"]["arrow"]["stroke_width"] == 6
        session.request_cancel()
    finally:
        agent.close()


def test_full_settings_reload_does_not_overwrite_agent_owned_annotation_styles(tmp_path: Path) -> None:
    app = _app()
    path = tmp_path / "settings.json"
    persisted = ScreenshotSettings(
        annotation_styles={
            "schema_version": 1,
            "active_tool": "arrow",
            "tools": {"arrow": {"color": "#123456", "arrow_size": 31}},
        }
    )
    ScreenshotSettingsIO.save(persisted, path)
    agent = ScreenshotAgent(
        app,
        CaptureCoordinator(_CaptureBackend()),
        settings_path=path,
        output_service=_OutputService(),
        cu5_locator=_Locator(),
    )
    try:
        dialog_draft = ScreenshotSettings(show_editor=True).to_dict()
        agent.reload_settings({"settings": dialog_draft})

        loaded = ScreenshotSettingsIO.load(path)
        assert loaded.show_editor is True
        assert loaded.annotation_styles["active_tool"] == "arrow"
        assert loaded.annotation_styles["tools"]["arrow"]["arrow_size"] == 31
    finally:
        agent.close()
