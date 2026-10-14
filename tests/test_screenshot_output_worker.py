from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from threading import Event
from time import monotonic, sleep

import pytest
from PySide6.QtCore import QCoreApplication, QEvent, QObject, QThread, QTimer
from PySide6.QtGui import QColor, QImage

from fdm.screenshot_settings import AfterCaptureTask, ScreenshotSettings
from fdm.services.screenshot_capture import CaptureMode
from fdm.services.screenshot_output import ScreenshotOutputError, ScreenshotOutputService
from fdm.services.screenshot_output_worker import (
    ScreenshotOutputQueue,
    ScreenshotOutputQueueFull,
)
from fdm.services import screenshot_output_worker


def _image() -> QImage:
    result = QImage(8, 6, QImage.Format.Format_ARGB32)
    result.fill(QColor("red"))
    return result


def _until(app, predicate, timeout: float = 5.0) -> None:
    deadline = monotonic() + timeout
    while not predicate():
        app.processEvents()
        if monotonic() > deadline:
            pytest.fail("timed out waiting for screenshot output")
        sleep(0.001)
    app.processEvents()


@pytest.fixture
def queues(desktop_application):
    created: list[ScreenshotOutputQueue] = []
    releases: list[Event] = []

    def create(*, service=None, max_pending=4):
        queue = ScreenshotOutputQueue(service=service, max_pending=max_pending)
        created.append(queue)
        if hasattr(service, "release"):
            releases.append(service.release)
        return queue

    yield create
    for release in releases:
        release.set()
    for queue in created:
        assert queue.wait_for_shutdown(5000)
    desktop_application.processEvents()
    for queue in created:
        queue.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


class _BlockedWriter(ScreenshotOutputService):
    def __init__(self) -> None:
        super().__init__()
        self.entered = Event()
        self.release = Event()
        self.observations = []

    def save_image(self, image, settings, **kwargs):
        self.entered.set()
        if not self.release.wait(5):
            raise RuntimeError("test did not release file writer")
        self.observations.append(
            (image.pixelColor(0, 0), settings, QThread.currentThread())
        )
        return super().save_image(image, settings, **kwargs)

    def copy_to_clipboard(self, image):
        pytest.fail("the output worker must never access the clipboard")


def test_slow_file_output_keeps_gui_timer_and_completion_on_gui_thread(
    queues, desktop_application, tmp_path: Path
) -> None:
    service = _BlockedWriter()
    queue = queues(service=service)
    completed = []
    queue.completed.connect(
        lambda job, result: completed.append((job, result, QThread.currentThread()))
    )
    job = queue.submit(_image(), ScreenshotSettings(output_directory=str(tmp_path)))
    _until(desktop_application, service.entered.is_set)

    ticks = []
    QTimer.singleShot(0, lambda: ticks.append(True))
    _until(desktop_application, lambda: bool(ticks))
    assert queue.busy and not completed

    service.release.set()
    _until(desktop_application, lambda: bool(completed))
    assert completed[0][0] == job
    assert completed[0][1].saved_path.exists()
    assert completed[0][2] == desktop_application.thread()
    assert service.observations[0][2] != desktop_application.thread()
    assert not queue.busy


def test_queue_limit_includes_active_output_and_shutdown_drains_in_order(
    queues, desktop_application, tmp_path: Path
) -> None:
    service = _BlockedWriter()
    queue = queues(service=service, max_pending=2)
    settings = ScreenshotSettings(output_directory=str(tmp_path), filename_template="same")
    completed = []
    stopped = []
    queue.completed.connect(lambda job, result: completed.append((job, result)))
    queue.shutdownFinished.connect(lambda: stopped.append(True))
    first = queue.submit(_image(), settings)
    _until(desktop_application, service.entered.is_set)
    second = queue.submit(_image(), settings)
    with pytest.raises(ScreenshotOutputQueueFull):
        queue.submit(_image(), settings)
    assert queue.pending_count == 2

    queue.shutdown()
    assert not stopped and queue.busy
    assert not queue.wait_for_shutdown(1)
    with pytest.raises(ScreenshotOutputError, match="正在关闭"):
        queue.submit(_image(), settings)
    service.release.set()
    _until(desktop_application, lambda: bool(stopped))
    assert [job for job, _ in completed] == [first, second]
    assert [result.saved_path.name for _, result in completed] == ["same.png", "same_001.png"]
    assert queue.is_shutdown_complete and queue.pending_count == 0
    assert queue.wait_for_shutdown(1)


def test_output_owns_image_and_deep_settings_snapshots(
    queues, desktop_application, tmp_path: Path
) -> None:
    service = _BlockedWriter()
    queue = queues(service=service)
    image = _image()
    settings = ScreenshotSettings(
        output_directory=str(tmp_path),
        filename_template="original",
        after_capture_tasks=(AfterCaptureTask.COPY_CLIPBOARD,),
        cu5_selector={"nested": {"value": "original"}},
    )
    completed = []
    queue.completed.connect(lambda job, result: completed.append(result))
    queue.submit(image, settings, mode=CaptureMode.WINDOW)
    _until(desktop_application, service.entered.is_set)
    image.fill(QColor("blue"))
    settings.filename_template = "changed"
    settings.cu5_selector["nested"]["value"] = "changed"
    service.release.set()
    _until(desktop_application, lambda: bool(completed))

    color, saved_settings, _ = service.observations[0]
    assert color == QColor("red")
    assert saved_settings.filename_template == "original"
    assert saved_settings.cu5_selector == {"nested": {"value": "original"}}
    assert saved_settings.after_capture_tasks == (AfterCaptureTask.SAVE,)
    assert completed[0].saved_path.name == "original.png"
    assert not completed[0].copied_to_clipboard
    assert QImage(str(completed[0].saved_path)).pixelColor(0, 0) == QColor("red")


def test_serial_queue_resolves_collisions_and_preserves_capture_timestamp(
    queues, desktop_application, tmp_path: Path
) -> None:
    queue = queues(max_pending=4)
    settings = ScreenshotSettings(output_directory=str(tmp_path), filename_template="{time}_{mode}")
    timestamp = datetime(2026, 10, 14, 12, 34, 56, tzinfo=timezone.utc)
    completed = []
    queue.completed.connect(lambda job, result: completed.append((job, result)))
    jobs = [
        queue.submit(_image(), settings, mode=CaptureMode.CU5, captured_at=timestamp)
        for _ in range(4)
    ]
    _until(desktop_application, lambda: len(completed) == 4)
    assert [job for job, _ in completed] == jobs
    assert [result.saved_path.name for _, result in completed] == [
        "12-34-56_cu5.png", "12-34-56_cu5_001.png",
        "12-34-56_cu5_002.png", "12-34-56_cu5_003.png",
    ]
    assert len(list(tmp_path.iterdir())) == 4


def test_one_failed_save_does_not_stop_following_job_or_explicit_save_as(
    queues, desktop_application, tmp_path: Path
) -> None:
    occupied = tmp_path / "occupied"
    occupied.write_text("file", encoding="utf-8")
    queue = queues()
    results, errors = [], []
    queue.completed.connect(lambda job, result: results.append((job, result)))
    queue.failed.connect(lambda job, error: errors.append((job, error)))
    bad = queue.submit(_image(), ScreenshotSettings(output_directory=str(occupied)))
    good = queue.submit(
        _image(), ScreenshotSettings(output_directory=str(tmp_path)),
        output_path=tmp_path / "chosen.jpg",
    )
    _until(desktop_application, lambda: len(results) == 1 and len(errors) == 1)
    assert errors[0][0] == bad and "保存文件失败" in errors[0][1]
    assert results[0][0] == good
    assert results[0][1].saved_path == tmp_path / "chosen.jpg"
    assert (tmp_path / "chosen.jpg").read_bytes().startswith(b"\xff\xd8")
    assert not queue.busy


def test_empty_shutdown_and_invalid_image_do_not_start_thread(queues) -> None:
    queue = queues()
    with pytest.raises(ScreenshotOutputError, match="null"):
        queue.submit(QImage(), ScreenshotSettings())
    assert not queue.busy
    stopped = []
    queue.shutdownFinished.connect(lambda: stopped.append(True))
    queue.shutdown()
    queue.shutdown()
    assert stopped == [True]
    assert queue.is_shutdown_complete


def test_blocking_shutdown_drains_without_gui_acknowledgements(
    queues, desktop_application, tmp_path: Path
) -> None:
    queue = queues()
    results = []
    queue.completed.connect(lambda job, result: results.append(result))
    settings = ScreenshotSettings(output_directory=str(tmp_path), filename_template="same")
    queue.submit(_image(), settings)
    queue.submit(_image(), settings)

    # Teardown cannot rely on callbacks after the GUI event loop has stopped.
    assert queue.wait_for_shutdown(5000)
    assert (tmp_path / "same.png").exists()
    assert (tmp_path / "same_001.png").exists()
    _until(desktop_application, lambda: queue.is_shutdown_complete)
    assert len(results) == 2 and not queue.busy


def test_deleting_owner_does_not_destroy_running_thread(
    desktop_application, tmp_path: Path
) -> None:
    owner = QObject()
    service = _BlockedWriter()
    queue = ScreenshotOutputQueue(owner, service=service)
    queue.submit(_image(), ScreenshotSettings(output_directory=str(tmp_path)))
    _until(desktop_application, service.entered.is_set)
    thread = queue._thread
    try:
        owner.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        assert thread in screenshot_output_worker._ACTIVE_WRITERS
    finally:
        service.release.set()
    _until(
        desktop_application,
        lambda: thread not in screenshot_output_worker._ACTIVE_WRITERS,
    )
    assert len(list(tmp_path.glob("*.png"))) == 1
