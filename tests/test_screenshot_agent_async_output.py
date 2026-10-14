from __future__ import annotations

from pathlib import Path
from dataclasses import replace
from threading import Event
from time import monotonic, sleep

import pytest
from PySide6.QtCore import QCoreApplication, QEvent, QThread, QTimer
from PySide6.QtGui import QColor, QImage

from fdm.screenshot_agent import ScreenshotAgent
from fdm.screenshot_settings import AfterCaptureTask, ScreenshotSettings, ScreenshotSettingsIO
from fdm.services.screenshot_capture import (
    CaptureCoordinator, CapturedFrame, CaptureMode, CaptureRect, CaptureRequest, ScreenInfo,
)
from fdm.services.screenshot_output import ScreenshotOutputService
from fdm.ui.screenshot_annotation_overlay import InlineAnnotationOverlay
from fdm.ui.screenshot_editor import ScreenshotEditor


def _image() -> QImage:
    image = QImage(12, 8, QImage.Format.Format_ARGB32)
    image.fill(QColor("#345678"))
    return image


def _until(app, predicate, timeout: float = 5) -> None:
    deadline = monotonic() + timeout
    while not predicate():
        app.processEvents()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        if monotonic() > deadline:
            pytest.fail("timed out waiting for screenshot agent output")
        sleep(0.001)


class _Backend:
    def __init__(self, *, no_screens=False):
        self.no_screens = no_screens
        self.capture_count = 0

    def screens(self):
        if self.no_screens:
            return ()
        rect = CaptureRect(0, 0, 800, 600)
        return (ScreenInfo("primary", rect, rect, 1, True),)

    def windows(self, *, include_children=True):
        return ()

    def active_window_handle(self):
        return 0

    def capture_rect(self, rect):
        self.capture_count += 1
        return _image()

    def capture_window(self, candidate):
        self.capture_count += 1
        return _image()


class _SlowOutput(ScreenshotOutputService):
    def __init__(self, *, fail_save=False, fail_copy=False):
        super().__init__(clipboard_setter=self._copy)
        self.fail_save = fail_save
        self.fail_copy = fail_copy
        self.entered = Event()
        self.release = Event()
        self.saved = []
        self.copied = []
        self.save_attempts = 0

    def _copy(self, image):
        if self.fail_copy:
            raise OSError("clipboard busy")
        self.copied.append((image.copy(), QThread.currentThread()))

    def _wait(self):
        self.save_attempts += 1
        self.entered.set()
        if not self.release.wait(5):
            raise RuntimeError("test did not release screenshot writer")
        if self.fail_save:
            raise OSError("disk unavailable")

    def save_image(self, image, settings, **kwargs):
        self._wait()
        path = super().save_image(image, settings, **kwargs)
        self.saved.append(path)
        return path

    def save_image_as(self, image, path, settings):
        self._wait()
        result = super().save_image_as(image, path, settings)
        self.saved.append(result)
        return result


@pytest.fixture
def agents(desktop_application, tmp_path, monkeypatch):
    created = []
    desktop_application.setQuitOnLastWindowClosed(False)

    def create(service=None, *, tasks=(AfterCaptureTask.SAVE,), no_screens=False):
        service = service or _SlowOutput()
        root = tmp_path / str(len(created))
        root.mkdir()
        path = root / "settings.json"
        ScreenshotSettingsIO.save(
            ScreenshotSettings(
                output_directory=str(root / "captures"), filename_template="same",
                after_capture_tasks=tasks, notification=False,
            ),
            path,
        )
        backend = _Backend(no_screens=no_screens)
        agent = ScreenshotAgent(
            desktop_application, CaptureCoordinator(backend), settings_path=path,
            output_service=service, cu5_locator=object(),
        )
        notices = []
        monkeypatch.setattr(agent.tray, "showMessage", lambda *args: notices.append(args))
        created.append((agent, service))
        return agent, service, notices, backend

    yield create
    for _, service in created:
        service.release.set()
    for agent, _ in created:
        agent.close()
        if agent._output_queue is not None:
            _until(desktop_application, lambda: agent._output_queue.is_shutdown_complete)
        agent.tray.hide()
        agent.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def _frame(*, editor=False) -> CapturedFrame:
    return CapturedFrame(
        _image(), CaptureRect(20, 30, 12, 8), CaptureMode.REGION,
        metadata={"open_editor": editor},
    )


def test_capture_ready_leaves_gui_responsive_and_saves_each_frame_once(
    agents, desktop_application
) -> None:
    agent, service, _, _ = agents()
    agent._capture_ready(_frame())
    _until(desktop_application, service.entered.is_set)
    ticks = []
    QTimer.singleShot(0, lambda: ticks.append(True))
    _until(desktop_application, lambda: bool(ticks))
    agent._capture_ready(_frame())
    assert agent.status()["output_pending"] == 2
    assert not service.saved

    service.release.set()
    _until(desktop_application, lambda: not agent._output_jobs)
    assert [path.name for path in service.saved] == ["same.png", "same_001.png"]
    assert agent.coordinator.last_region == _frame().rect


def test_partial_copy_success_retains_annotation_and_retry_saves_once(
    agents, desktop_application
) -> None:
    agent, service, notices, _ = agents(
        _SlowOutput(fail_save=True),
        tasks=(AfterCaptureTask.SAVE, AfterCaptureTask.COPY_CLIPBOARD),
    )
    frame = _frame(editor=True)
    agent._capture_ready(frame)
    session = agent.annotation_session
    assert isinstance(session, InlineAnnotationOverlay)
    session.request_complete()
    _until(desktop_application, service.entered.is_set)
    assert session.output_pending
    assert not session.isEnabled()  # Prevent edits after the output snapshot.
    session.request_complete()
    assert len(agent._output_jobs) == 1 and len(service.copied) == 1
    assert service.copied[0][1] == desktop_application.thread()

    service.release.set()
    _until(desktop_application, lambda: not agent._output_jobs)
    assert agent.annotation_session is session
    assert session.isEnabled() and not session.output_pending
    assert agent.coordinator.last_region == frame.rect
    assert any(item[0] == "截图部分完成" and "disk unavailable" in item[1] for item in notices)

    service.fail_save = False
    session.request_save()
    _until(desktop_application, lambda: agent.annotation_session is None)
    assert len(service.saved) == 1
    assert len(service.copied) == 1


def test_total_failure_keeps_annotation_and_previous_region_for_retry(
    agents, desktop_application
) -> None:
    agent, service, _, _ = agents(_SlowOutput(fail_save=True))
    original = CaptureRect(1, 2, 3, 4)
    agent.coordinator.set_last_region(original)
    agent._capture_ready(_frame(editor=True))
    session = agent.annotation_session
    session.request_complete()
    service.release.set()
    _until(desktop_application, lambda: not agent._output_jobs)
    assert agent.annotation_session is session
    assert session.isEnabled() and not session.output_pending
    assert agent.coordinator.last_region == original
    assert "disk unavailable" in agent._last_output_error


def test_fallback_failure_reenables_editor_and_save_as_retry_closes_it(
    agents, desktop_application, tmp_path
) -> None:
    agent, service, _, _ = agents(_SlowOutput(fail_save=True), no_screens=True)
    frame = _frame(editor=True)
    agent._capture_ready(frame)
    session = agent.annotation_session
    assert isinstance(session, ScreenshotEditor)
    session.complete()
    assert not session.isEnabled()
    service.release.set()
    _until(desktop_application, lambda: not agent._output_jobs)
    assert agent.annotation_session is session and session.isEnabled()

    service.fail_save = False
    path = tmp_path / "chosen.jpg"
    session.saveAsOutputRequested.emit(session.edited_image(), str(path))
    _until(desktop_application, lambda: agent.annotation_session is None)
    assert path.read_bytes().startswith(b"\xff\xd8")
    assert service.saved == [path]


def test_shutdown_drains_existing_output_and_cancels_delayed_capture(
    agents, desktop_application, monkeypatch
) -> None:
    agent, service, _, backend = agents()
    agent._capture_ready(_frame())
    _until(desktop_application, service.entered.is_set)
    agent.begin_capture(CaptureRequest(CaptureMode.FULL_SCREEN, delay_ms=20))
    quit_calls = []
    monkeypatch.setattr(desktop_application, "quit", lambda: quit_calls.append(True))
    agent.request_shutdown()
    assert not quit_calls
    ticks = []
    QTimer.singleShot(60, lambda: ticks.append(True))
    _until(desktop_application, lambda: bool(ticks))
    assert backend.capture_count == 0
    assert service.save_attempts == 1

    service.release.set()
    _until(desktop_application, lambda: bool(quit_calls))
    assert len(service.saved) == 1 and not agent._output_jobs
    assert agent._output_queue.is_shutdown_complete


def test_copy_failure_with_successful_save_keeps_annotation_editable(
    agents, desktop_application
) -> None:
    agent, service, notices, _ = agents(
        _SlowOutput(fail_copy=True),
        tasks=(AfterCaptureTask.SAVE, AfterCaptureTask.COPY_CLIPBOARD),
    )
    agent._capture_ready(_frame(editor=True))
    session = agent.annotation_session
    session.request_complete()
    service.release.set()
    _until(desktop_application, lambda: not agent._output_jobs)
    assert agent.annotation_session is session
    assert session.isEnabled() and not session.output_pending
    assert len(service.saved) == 1
    assert any("clipboard busy" in item[1] for item in notices)


@pytest.mark.parametrize("fallback", [False, True])
def test_shutdown_save_failure_preserves_annotation_and_allows_retry_on_new_queue(
    agents, desktop_application, monkeypatch, fallback
) -> None:
    agent, service, notices, _ = agents(_SlowOutput(fail_save=True), no_screens=fallback)
    agent._capture_ready(_frame(editor=True))
    session = agent.annotation_session
    if fallback:
        session.complete()
    else:
        session.request_complete()
    _until(desktop_application, service.entered.is_set)
    stopped_queue = agent._output_queue
    quit_calls = []
    monkeypatch.setattr(desktop_application, "quit", lambda: quit_calls.append(True))
    agent.request_shutdown()
    service.release.set()
    _until(desktop_application, lambda: bool(quit_calls) or not agent._shutting_down)

    assert not quit_calls
    assert agent.annotation_session is session and session.isEnabled()
    assert agent._output_queue is None
    assert not agent._output_jobs
    assert any(item[0] == "已取消退出" for item in notices)
    service.fail_save = False
    if fallback:
        session.complete()
    else:
        session.request_save()
    assert agent._output_queue is not stopped_queue
    _until(desktop_application, lambda: agent.annotation_session is None)
    assert len(service.saved) == 1
    agent.request_shutdown()
    _until(desktop_application, lambda: bool(quit_calls))
    assert quit_calls == [True]


def test_older_slow_save_does_not_replace_newer_clipboard_success_region(
    agents, desktop_application
) -> None:
    agent, service, _, _ = agents()
    first = _frame()
    latest = replace(first, rect=CaptureRect(100, 110, 12, 8))
    agent._capture_ready(first)
    _until(desktop_application, service.entered.is_set)
    agent._settings.after_capture_tasks = (AfterCaptureTask.COPY_CLIPBOARD,)
    agent._capture_ready(latest)
    assert agent.coordinator.last_region == latest.rect

    service.release.set()
    _until(desktop_application, lambda: not agent._output_jobs)
    assert len(service.saved) == 1 and len(service.copied) == 1
    assert agent.coordinator.last_region == latest.rect
    assert ScreenshotSettingsIO.load(agent._settings_path).last_region == latest.rect


def test_later_failed_clipboard_does_not_prevent_earlier_save_region_success(
    agents, desktop_application
) -> None:
    agent, service, _, _ = agents(_SlowOutput(fail_copy=True))
    first = _frame()
    later_failed = replace(first, rect=CaptureRect(100, 110, 12, 8))
    agent._capture_ready(first)
    _until(desktop_application, service.entered.is_set)
    agent._settings.after_capture_tasks = (AfterCaptureTask.COPY_CLIPBOARD,)
    agent._capture_ready(later_failed)
    assert agent.coordinator.last_region is None

    service.release.set()
    _until(desktop_application, lambda: not agent._output_jobs)
    assert agent.coordinator.last_region == first.rect


def test_full_output_queue_keeps_annotation_for_retry_and_latest_partial_region(
    agents, desktop_application
) -> None:
    agent, service, notices, _ = agents()
    frame = _frame()
    for _ in range(4):
        agent._capture_ready(frame)
    _until(desktop_application, service.entered.is_set)
    latest = replace(frame, rect=CaptureRect(100, 110, 12, 8), metadata={"open_editor": True})
    agent._settings.after_capture_tasks = (AfterCaptureTask.SAVE, AfterCaptureTask.COPY_CLIPBOARD)
    agent._capture_ready(latest)
    session = agent.annotation_session
    session.request_complete()
    assert len(agent._output_jobs) == 4
    assert session.isEnabled() and not session.output_pending
    assert agent.coordinator.last_region == latest.rect
    assert any("队列已满" in item[1] for item in notices)

    service.release.set()
    _until(desktop_application, lambda: not agent._output_jobs)
    assert len(service.saved) == 4 and len(service.copied) == 1
    assert agent.coordinator.last_region == latest.rect
    session.request_save()
    _until(desktop_application, lambda: agent.annotation_session is None)
    assert len(service.saved) == 5
