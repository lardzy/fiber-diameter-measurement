"""Single-worker Qt controller for deterministic analysis batches."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal, Slot

from fdm.cancellation import (
    CancellationError,
    CancellationToken,
    CancellationTokenSource,
)
from fdm.services.analysis_batch import (
    AnalysisBatchProgress,
    AnalysisBatchRequest,
    AnalysisBatchResult,
    execute_analysis_batch,
)


BatchExecutor = Callable[
    [AnalysisBatchRequest, CancellationToken, Callable[[AnalysisBatchProgress], None]],
    AnalysisBatchResult,
]


class _WorkerSignals(QObject):
    progress = Signal(object)
    completed = Signal(object)


@dataclass(frozen=True, slots=True)
class _Completion:
    request: AnalysisBatchRequest
    result: AnalysisBatchResult | None = None
    error: str | None = None
    cancelled: bool = False


class _BatchWorker(QRunnable):
    def __init__(
        self,
        request: AnalysisBatchRequest,
        token: CancellationToken,
        executor: BatchExecutor,
    ) -> None:
        super().__init__()
        self.request = request
        self.token = token
        self.executor = executor
        self.signals = _WorkerSignals()

    @Slot()
    def run(self) -> None:
        try:
            result = self.executor(
                self.request,
                self.token,
                self.signals.progress.emit,
            )
        except CancellationError:
            self.signals.completed.emit(
                _Completion(self.request, cancelled=True)
            )
        except Exception as exc:
            self.signals.completed.emit(
                _Completion(
                    self.request,
                    error=str(exc),
                    cancelled=self.token.is_cancelled,
                )
            )
        else:
            self.signals.completed.emit(_Completion(self.request, result=result))


def _default_executor(
    request: AnalysisBatchRequest,
    token: CancellationToken,
    progress: Callable[[AnalysisBatchProgress], None],
) -> AnalysisBatchResult:
    return execute_analysis_batch(
        request,
        cancellation_token=token,
        progress=progress,
    )


class AnalysisBatchController(QObject):
    batchStarted = Signal(str, int)
    progressChanged = Signal(object)
    batchReady = Signal(object)
    batchCancelled = Signal(str, int)
    batchFailed = Signal(str, int, str)
    staleResultDiscarded = Signal(str, int)
    busyChanged = Signal(bool)

    def __init__(
        self,
        parent: QObject | None = None,
        *,
        thread_pool: QThreadPool | None = None,
        executor: BatchExecutor = _default_executor,
    ) -> None:
        super().__init__(parent)
        self._thread_pool = thread_pool or QThreadPool(self)
        self._thread_pool.setMaxThreadCount(1)
        self._thread_pool.setExpiryTimeout(5_000)
        self._executor = executor
        self._generation = 0
        self._active_request: AnalysisBatchRequest | None = None
        self._cancellation: CancellationTokenSource | None = None
        self._closed = False

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def busy(self) -> bool:
        return self._active_request is not None

    @property
    def active_request(self) -> AnalysisBatchRequest | None:
        return self._active_request

    def next_generation(self) -> int:
        if self._closed:
            raise RuntimeError("批量分析控制器已经关闭")
        self.cancel()
        self._generation += 1
        return self._generation

    def start(self, request: AnalysisBatchRequest) -> bool:
        if not isinstance(request, AnalysisBatchRequest):
            raise TypeError("request 必须是 AnalysisBatchRequest")
        if self._closed:
            raise RuntimeError("批量分析控制器已经关闭")
        if self.busy:
            return False
        if request.generation < self._generation:
            raise ValueError("不能启动过期 generation")
        self._generation = request.generation
        source = CancellationTokenSource()
        worker = _BatchWorker(request, source.token, self._executor)
        worker.signals.progress.connect(self._on_progress)
        worker.signals.completed.connect(self._on_completed)
        self._active_request = request
        self._cancellation = source
        self.busyChanged.emit(True)
        self.batchStarted.emit(request.request_id, request.generation)
        self._thread_pool.start(worker)
        return True

    def cancel(self) -> bool:
        if self._cancellation is None:
            return False
        return self._cancellation.cancel()

    def wait_for_done(self, timeout_ms: int = 5_000) -> bool:
        return self._thread_pool.waitForDone(timeout_ms)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.cancel()
        self._generation += 1

    @Slot(object)
    def _on_progress(self, update: object) -> None:
        if not isinstance(update, AnalysisBatchProgress):
            return
        request = self._active_request
        if (
            request is not None
            and update.request_id == request.request_id
            and update.generation == self._generation
        ):
            self.progressChanged.emit(update)

    @Slot(object)
    def _on_completed(self, payload: object) -> None:
        if not isinstance(payload, _Completion):
            return
        request = payload.request
        active_matches = (
            self._active_request is not None
            and request.request_id == self._active_request.request_id
        )
        if not active_matches:
            self.staleResultDiscarded.emit(request.request_id, request.generation)
            return
        self._active_request = None
        self._cancellation = None
        self.busyChanged.emit(False)
        if request.generation != self._generation:
            self.staleResultDiscarded.emit(request.request_id, request.generation)
            return
        if payload.cancelled:
            self.batchCancelled.emit(request.request_id, request.generation)
        elif payload.error is not None:
            self.batchFailed.emit(
                request.request_id,
                request.generation,
                payload.error,
            )
        elif payload.result is not None and payload.result.cancelled:
            self.batchCancelled.emit(request.request_id, request.generation)
        elif payload.result is not None:
            self.batchReady.emit(payload.result)


__all__ = ["AnalysisBatchController", "BatchExecutor"]
