from __future__ import annotations

from collections.abc import Callable
from threading import Event, Lock, Thread
from typing import TypeVar

from PySide6.QtCore import QEventLoop, QTimer, Qt
from PySide6.QtWidgets import QProgressDialog, QWidget


T = TypeVar("T")
ProgressCallback = Callable[[int, int], None]


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

    loop = QEventLoop(parent)
    dialog = _OwnedProgressDialog(label, "", 0, 0, parent)
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

    def progress(completed: int, total: int) -> None:
        nonlocal latest_progress
        with lock:
            latest_progress = (int(completed), int(total))

    def work() -> None:
        try:
            result.append(operation(progress))
        except BaseException as exc:
            error.append(exc)

    thread = Thread(target=work, name="fdm-slide-io", daemon=True)
    timer = QTimer(loop)
    timer.setInterval(20)

    def poll() -> None:
        nonlocal latest_progress
        if not thread.is_alive():
            loop.quit()
            return
        with lock:
            value, latest_progress = latest_progress, None
        if value is not None:
            completed, total = value
            dialog.setRange(0, 1000)
            dialog.setValue(min(999, int(1000 * completed / max(1, total))))

    timer.timeout.connect(poll)
    try:
        thread.start()
        dialog.setValue(0)
        if thread.is_alive():
            timer.start()
            while thread.is_alive():
                loop.exec()
        # poll only exits after the actual thread has returned, so this join
        # cannot wait for network I/O while holding the GUI thread.
        thread.join()
        if error:
            raise error[0]
        return result[0]
    finally:
        timer.stop()
        if cancellation_event is not None:
            dialog.canceled.disconnect(cancel)
        dialog._io_complete = True
        dialog.close()
        dialog.deleteLater()
        loop.deleteLater()
