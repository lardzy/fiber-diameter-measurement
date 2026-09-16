"""A bounded worker whose QObject signal carrier outlives every runnable."""
from __future__ import annotations

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal, Slot

from fdm.cancellation import CancellationError, CancellationTokenSource


class _Signals(QObject):
    finished = Signal(object, str, bool)


class _Task(QRunnable):
    def __init__(self, function, source, signals):
        super().__init__()
        self.function, self.source, self.signals = function, source, signals

    def run(self):
        result, error, cancelled = None, "", False
        try:
            self.source.token.raise_if_cancelled()
            result = self.function(self.source.token)
            self.source.token.raise_if_cancelled()
        except CancellationError:
            cancelled = True
        except Exception as exc:
            error = str(exc).strip() or type(exc).__name__
        self.signals.finished.emit(result, error, cancelled)


class ContourTaskController(QObject):
    finished = Signal(object, str, bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pool = QThreadPool(self)
        self._pool.setMaxThreadCount(1)
        self._pool.setExpiryTimeout(5000)
        self._signals = _Signals(self)
        self._signals.finished.connect(self._complete)
        self._source = None

    @property
    def busy(self):
        return self._source is not None

    def start(self, function):
        if self.busy:
            raise RuntimeError("请等待当前轮廓任务完成。")
        self._source = CancellationTokenSource()
        self._pool.start(_Task(function, self._source, self._signals))

    def cancel(self):
        if self._source is not None:
            self._source.cancel()

    @Slot(object, str, bool)
    def _complete(self, result, error, cancelled):
        # A user can cancel after run() emitted but before this queued slot is
        # delivered. The UI's accepted cancellation still wins that race.
        cancelled = cancelled or (self._source is not None and self._source.token.is_cancelled)
        if cancelled:
            result, error = None, ""
        self._source = None
        self.finished.emit(result, error, cancelled)

    def wait_for_done(self, timeout_ms=5000):
        return self._pool.waitForDone(timeout_ms)
