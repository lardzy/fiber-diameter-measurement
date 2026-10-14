from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from PySide6.QtCore import QCoreApplication, QObject, QThread, Qt, Signal, Slot
from PySide6.QtGui import QImage

from fdm.screenshot_settings import AfterCaptureTask, ScreenshotSettings
from fdm.services.screenshot_capture import CaptureMode
from fdm.services.screenshot_output import (
    OutputResult,
    ScreenshotOutputError,
    ScreenshotOutputService,
)


class ScreenshotOutputQueueFull(ScreenshotOutputError):
    pass


@dataclass(frozen=True, slots=True)
class _SaveRequest:
    job_id: str
    image: QImage
    settings: ScreenshotSettings
    mode: CaptureMode
    captured_at: datetime
    output_path: Path | None


class _FileWriter(QObject):
    completed = Signal(str, object)
    failed = Signal(str, str)

    def __init__(self, service: ScreenshotOutputService) -> None:
        super().__init__()
        self._service = service

    @Slot(object)
    def save(self, request: _SaveRequest) -> None:
        # Do not call process_capture here: clipboard ownership belongs to the
        # GUI thread even when the user's configured tasks include clipboard.
        try:
            if request.output_path is None:
                path = self._service.save_image(
                    request.image,
                    request.settings,
                    mode=request.mode,
                    captured_at=request.captured_at,
                )
            else:
                path = self._service.save_image_as(
                    request.image, request.output_path, request.settings
                )
            result = OutputResult(
                saved_path=path,
                open_editor_requested=request.settings.show_editor,
                notification_requested=request.settings.notification,
            )
        except Exception as exc:  # noqa: BLE001 - one job must not stop the queue
            self.failed.emit(
                request.job_id,
                f"保存文件失败：{str(exc).strip() or type(exc).__name__}",
            )
        else:
            self.completed.emit(request.job_id, result)

    @Slot()
    def stop(self) -> None:
        # This queued slot follows every accepted save request. No main-thread
        # acknowledgement is needed to drain, including during aboutToQuit.
        QThread.currentThread().quit()


# Keep unparented threads alive if a caller accidentally deletes its queue
# before draining. QObject.destroyed queues stop after the accepted requests;
# a running QThread is never destroyed together with the owner QObject.
_ACTIVE_WRITERS: dict[QThread, _FileWriter] = {}


class ScreenshotOutputQueue(QObject):
    """Bounded, sequential file output with results delivered on the GUI thread.

    All public methods must be called on this object's owning thread. submit()
    always saves a file; callers perform clipboard work separately. The limit
    includes the currently running job, bounding retained image snapshots.

    Normal exit calls shutdown(), keeps the event loop alive, and waits for
    shutdownFinished. wait_for_shutdown() is a blocking fallback for teardown
    or an externally initiated application quit, never the normal GUI flow.
    """

    completed = Signal(str, object)
    failed = Signal(str, str)
    busyChanged = Signal(bool)
    drained = Signal()
    shutdownFinished = Signal()
    _save_requested = Signal(object)
    _stop_requested = Signal()

    def __init__(
        self,
        parent: QObject | None = None,
        *,
        service: ScreenshotOutputService | None = None,
        max_pending: int = 4,
    ) -> None:
        super().__init__(parent)
        if max_pending < 1:
            raise ValueError("max_pending must be positive")
        self._service = service if service is not None else ScreenshotOutputService()
        self._max_pending = max_pending
        self._pending: set[str] = set()
        self._thread: QThread | None = None
        self._writer: _FileWriter | None = None
        self._closing = False
        self._shutdown_complete = False
        app = QCoreApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self._on_application_quit)

    @property
    def busy(self) -> bool:
        return bool(self._pending)

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    @property
    def is_shutdown_complete(self) -> bool:
        return self._shutdown_complete

    def submit(
        self,
        image: QImage,
        settings: ScreenshotSettings,
        *,
        mode: CaptureMode = CaptureMode.REGION,
        captured_at: datetime | None = None,
        output_path: str | Path | None = None,
    ) -> str:
        self._check_owner_thread()
        if self._closing:
            raise ScreenshotOutputError("截图文件输出正在关闭，无法接收新任务")
        if len(self._pending) >= self._max_pending:
            raise ScreenshotOutputQueueFull("截图文件保存队列已满，请稍后重试")
        if not isinstance(image, QImage) or image.isNull():
            raise ScreenshotOutputError("cannot save a null screenshot")
        # Settings contain mutable mappings. A shallow dataclass replacement
        # would let edits made after submit change the accepted job.
        snapshot = deepcopy(settings).normalized()
        snapshot.after_capture_tasks = (AfterCaptureTask.SAVE,)
        image_snapshot = image.copy()
        if image_snapshot.isNull():
            raise ScreenshotOutputError("could not allocate screenshot output snapshot")
        request = _SaveRequest(
            job_id=uuid4().hex,
            image=image_snapshot,
            settings=snapshot,
            mode=CaptureMode.parse(mode),
            captured_at=captured_at or datetime.now().astimezone(),
            output_path=Path(output_path) if output_path is not None else None,
        )
        self._ensure_writer()
        was_busy = self.busy
        self._pending.add(request.job_id)
        self._save_requested.emit(request)
        if not was_busy:
            self.busyChanged.emit(True)
        return request.job_id

    @Slot()
    def shutdown(self) -> None:
        self._check_owner_thread()
        if self._closing:
            return
        self._closing = True
        if self._thread is None:
            self._shutdown_complete = True
            self.shutdownFinished.emit()
        else:
            self._stop_requested.emit()

    def wait_for_shutdown(self, timeout_ms: int | None = None) -> bool:
        self.shutdown()
        thread = self._thread
        if thread is None:
            return True
        # save/stop are already enqueued, so waiting does not depend on GUI
        # callbacks. Queued result notifications are delivered after return.
        return thread.wait() if timeout_ms is None else thread.wait(max(0, timeout_ms))

    @Slot()
    def _on_application_quit(self) -> None:
        self.wait_for_shutdown()

    def _check_owner_thread(self) -> None:
        if QThread.currentThread() != self.thread():
            raise RuntimeError("ScreenshotOutputQueue must be used from its owner thread")

    def _ensure_writer(self) -> None:
        if self._thread is not None:
            return
        thread = QThread()
        thread.setObjectName("fdm-screenshot-file-output")
        writer = _FileWriter(self._service)
        writer.moveToThread(thread)
        self._save_requested.connect(writer.save, Qt.ConnectionType.QueuedConnection)
        self._stop_requested.connect(writer.stop, Qt.ConnectionType.QueuedConnection)
        self.destroyed.connect(writer.stop, Qt.ConnectionType.QueuedConnection)
        writer.completed.connect(self._on_completed)
        writer.failed.connect(self._on_failed)
        thread.finished.connect(writer.deleteLater)
        thread.finished.connect(self._on_thread_finished)
        thread.finished.connect(lambda: _ACTIVE_WRITERS.pop(thread, None))
        thread.finished.connect(thread.deleteLater)
        self._thread = thread
        self._writer = writer
        _ACTIVE_WRITERS[thread] = writer
        thread.start()

    @Slot(str, object)
    def _on_completed(self, job_id: str, result: OutputResult) -> None:
        self._pending.discard(job_id)
        self.completed.emit(job_id, result)
        # A completed slot may submit a new job; test the current state after
        # returning to avoid announcing a drain while the new job is pending.
        if not self._pending:
            self.busyChanged.emit(False)
            self.drained.emit()

    @Slot(str, str)
    def _on_failed(self, job_id: str, error: str) -> None:
        self._pending.discard(job_id)
        self.failed.emit(job_id, error)
        if not self._pending:
            self.busyChanged.emit(False)
            self.drained.emit()

    @Slot()
    def _on_thread_finished(self) -> None:
        self._thread = None
        self._writer = None
        self._shutdown_complete = True
        self.shutdownFinished.emit()
