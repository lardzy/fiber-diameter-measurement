from __future__ import annotations

from collections.abc import Callable
from contextvars import copy_context
from threading import Event, Lock, Thread
from typing import TypeVar

from PySide6.QtCore import QEventLoop, QObject, QThread, QTimer, Qt, Signal
from PySide6.QtWidgets import QProgressBar, QProgressDialog, QWidget

from fdm.operation_diagnostics import diagnose_operation, operation_phase

T = TypeVar("T")
ProgressCallback = Callable[[int, int], None]


class _IOCompletion(QObject):
    finished = Signal()


class _OwnedProgressDialog(QProgressDialog):
    _io_complete = False

    def reject(self) -> None:
        if self._io_complete:
            super().reject()
        else:
            self.canceled.emit()

    def closeEvent(self, event) -> None:
        if self._io_complete:
            super().closeEvent(event)
        else:
            event.ignore()
            self.canceled.emit()


@diagnose_operation("digital-slide-file-io/v2")
def run_responsive_io(
    parent: QWidget,
    *,
    title: str,
    label: str,
    operation: Callable[[ProgressCallback], T],
    cancellation_event: Event | None = None,
) -> T:
    """Run one owned I/O operation while the modal GUI keeps dispatching.

    The caller retains its synchronous transaction/exception boundary. No GUI
    calls occur in the worker, and progress is a bounded latest-value mailbox.
    The worker must finish before this scope (and its owner) can be released.
    Callers guard transitions/reentry for the duration of the operation.
    """

    if QThread.currentThread() != parent.thread():
        raise RuntimeError("切片进度窗口必须在界面线程中创建。")

    operation_phase(f"gui.setup: {label}")
    loop = QEventLoop(parent)
    completion = _IOCompletion(loop)
    completion.finished.connect(loop.quit, Qt.ConnectionType.QueuedConnection)
    dialog = _OwnedProgressDialog(label, "", 0, 0, parent)
    bar = QProgressBar(dialog)
    bar.setRange(0, 0)
    dialog.setBar(bar)
    dialog.setWindowTitle(title)
    dialog.setWindowModality(Qt.WindowModality.ApplicationModal)
    dialog.setWindowFlag(Qt.WindowType.WindowCloseButtonHint, False)
    if cancellation_event is None:
        dialog.setCancelButton(None)
    else:
        dialog.setCancelButtonText("取消")

        def cancel() -> None:
            cancellation_event.set()
            dialog.setLabelText("正在取消，请等待当前文件操作返回…")
            dialog.setCancelButton(None)
            dialog.show()

        dialog.canceled.connect(cancel)
    dialog.setMinimumDuration(250)
    dialog.setAutoClose(False)
    dialog.setAutoReset(False)
    lock = Lock()
    latest_progress: tuple[int, int] | None = None
    result: list[T] = []
    error: list[BaseException] = []
    work_finished = Event()

    def progress(completed: int, total: int) -> None:
        nonlocal latest_progress
        with lock:
            latest_progress = (int(completed), int(total))

    def work() -> None:
        operation_phase("worker.started")
        try:
            result.append(operation(progress))
        except BaseException as exc:
            error.append(exc)
        finally:
            operation_phase("worker.finished; posting completion")
            work_finished.set()
            # Completion must not depend on a low-priority progress timer.
            # In a busy native Windows message loop timers can be delayed
            # even though queued events and camera messages keep arriving.
            completion.finished.emit()

    worker_context = copy_context()
    thread = Thread(target=lambda: worker_context.run(work), name="fdm-slide-io", daemon=True)
    timer = QTimer(loop)
    timer.setInterval(20)

    def poll() -> None:
        nonlocal latest_progress
        if work_finished.is_set():
            return
        with lock:
            value, latest_progress = latest_progress, None
        if value is not None:
            completed, total = value
            bar.setRange(0, 1000)
            # QProgressDialog.setValue() processes events when modal. A
            # nested dispatch here can postpone the completion handoff or
            # reenter the caller; update only the owned bar instead.
            bar.setValue(min(999, int(1000 * completed / max(1, total))))

    timer.timeout.connect(poll)
    try:
        dialog.setValue(0)
        thread.start()
        if not work_finished.is_set():
            operation_phase("gui.waiting_for_completion")
            timer.start()
            while not work_finished.is_set():
                loop.exec()
        # Work has finished; only the queued signal emission/thread teardown
        # can remain. Joining preserves ownership and releases the Python GIL.
        operation_phase("gui.received_completion")
        thread.join()
        if error:
            raise error[0]
        return result[0]
    finally:
        timer.stop()
        if cancellation_event is not None:
            dialog.canceled.disconnect(cancel)
        dialog._io_complete = True
        # A not-yet-shown native QWindow can accept close() without delivering
        # QProgressDialog.closeEvent(), leaving its forceShow timer running.
        # A following QMessageBox.exec() also postpones deferred deletion, so
        # the completed loader can reappear and block that completion dialog.
        # reset() stops the internal timer; autoClose is false, so hide too.
        dialog.reset()
        dialog.hide()
        dialog.deleteLater()
        loop.deleteLater()
        operation_phase("gui.progress_closed")
