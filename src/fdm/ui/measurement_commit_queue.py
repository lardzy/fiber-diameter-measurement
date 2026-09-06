"""Ordered accepted measurement operations; workers never touch Qt widgets."""

import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from PySide6.QtCore import QObject, QTimer, QEventLoop, Signal

_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="fdm-area-finalize")


@dataclass
class _Commit:
    document: object
    future: Future
    apply: object
    cancelled: bool = False


class MeasurementCommitQueue(QObject):
    changed = Signal()
    failed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pending = deque()
        self._timer = QTimer(self)
        self._timer.setInterval(10)
        self._timer.timeout.connect(self.poll)
        self._publishing = False
        self._failures = []

    def pending(self, document=None):
        return any(
            not job.cancelled and (document is None or job.document is document)
            for job in self._pending
        )

    def submit(self, document, compute, apply):
        self._pending.append(_Commit(document, _executor.submit(compute), apply))
        self._timer.start()
        self.changed.emit()

    def cancel_last(self, document):
        for job in reversed(self._pending):
            if job.document is document and not job.cancelled:
                job.cancelled = True
                job.future.cancel()
                self.changed.emit()
                return True
        return False

    def poll(self):
        if self._publishing:
            return
        self._publishing = True
        deadline = time.perf_counter() + 0.004
        try:
            while self._pending and (self._pending[0].cancelled or self._pending[0].future.done()):
                job = self._pending.popleft()
                if job.cancelled:
                    continue
                try:
                    job.apply(job.future.result())
                except Exception as error:
                    self._failures.append((job.document, str(error)))
                    self.failed.emit(str(error))
                self.changed.emit()
                if time.perf_counter() >= deadline:
                    break
            if not self._pending:
                self._timer.stop()
        finally:
            self._publishing = False

    def flush(self, document=None):
        if self._publishing or not self.pending(document):
            return
        # Keep painting and cancellation live while save/export obtains the
        # accepted operations. No extra confirmation or blocking future.result.
        loop = QEventLoop()

        def check():
            if not self.pending(document):
                loop.quit()

        self.changed.connect(check)
        try:
            self.poll()
            if self.pending(document):
                loop.exec()
        finally:
            self.changed.disconnect(check)

    def raise_failures(self, document=None):
        errors = [
            message for owner, message in self._failures if document is None or owner is document
        ]
        self._failures = [
            (owner, message)
            for owner, message in self._failures
            if document is not None and owner is not document
        ]
        if errors:
            raise RuntimeError("已接受的测量未能完成：" + "；".join(errors))

    def cancel_all(self):
        for job in self._pending:
            job.cancelled = True
            job.future.cancel()
        self._pending.clear()
        self._failures.clear()
        self._timer.stop()
        self.changed.emit()
