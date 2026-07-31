"""Qt orchestration for isolated image-recipe batch execution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, TypeAlias
from uuid import uuid4

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal, Slot

from fdm.cancellation import (
    CancellationError,
    CancellationToken,
    CancellationTokenSource,
)
from fdm.image_processing_models import ImageProcessingRecipe
from fdm.services.image_batch import (
    BatchExecutionLimits,
    BatchExecutionResult,
    BatchProgressCallback,
    BatchProgressUpdate,
    BatchRasterInput,
    BatchRecipeRequest,
    DEFAULT_BATCH_EXECUTION_LIMITS,
    execute_batch_recipe,
)


BatchTaskExecutor: TypeAlias = Callable[
    [
        BatchRecipeRequest,
        CancellationToken,
        BatchProgressCallback,
        BatchExecutionLimits,
    ],
    BatchExecutionResult,
]


def execute_image_batch_task(
    request: BatchRecipeRequest,
    token: CancellationToken,
    progress_callback: BatchProgressCallback,
    limits: BatchExecutionLimits,
) -> BatchExecutionResult:
    return execute_batch_recipe(
        request,
        cancellation_token=token,
        progress_callback=progress_callback,
        limits=limits,
    )


@dataclass(frozen=True, slots=True)
class _BatchTaskCompletion:
    request: BatchRecipeRequest
    result: BatchExecutionResult | None = None
    error: str | None = None
    cancelled: bool = False


class _BatchTaskSignals(QObject):
    completed = Signal(object)
    progress = Signal(object)


class _BatchTaskRunnable(QRunnable):
    def __init__(
        self,
        *,
        request: BatchRecipeRequest,
        token: CancellationToken,
        executor: BatchTaskExecutor,
        limits: BatchExecutionLimits,
        signals: _BatchTaskSignals,
    ) -> None:
        super().__init__()
        self._request = request
        self._token = token
        self._executor = executor
        self._limits = limits
        self._signals = signals

    @Slot()
    def run(self) -> None:
        def report(update: BatchProgressUpdate) -> None:
            self._token.raise_if_cancelled()
            if (
                update.request_id != self._request.request_id
                or update.generation != self._request.generation
            ):
                raise RuntimeError(
                    "批处理进度的 request_id/generation 与请求不一致"
                )
            self._signals.progress.emit(update)

        try:
            self._token.raise_if_cancelled()
            result = self._executor(
                self._request,
                self._token,
                report,
                self._limits,
            )
            self._token.raise_if_cancelled()
            if (
                result.request_id != self._request.request_id
                or result.generation != self._request.generation
            ):
                raise RuntimeError(
                    "批处理结果的 request_id/generation 与请求不一致"
                )
            completion = _BatchTaskCompletion(
                request=self._request,
                result=result,
            )
        except CancellationError:
            completion = _BatchTaskCompletion(
                request=self._request,
                cancelled=True,
            )
        except Exception as exc:
            if self._token.is_cancelled:
                completion = _BatchTaskCompletion(
                    request=self._request,
                    cancelled=True,
                )
            else:
                completion = _BatchTaskCompletion(
                    request=self._request,
                    error=str(exc).strip() or type(exc).__name__,
                )
        self._signals.completed.emit(completion)


class ImageBatchTaskController(QObject):
    """Single-worker controller with latest-request wins semantics."""

    batchReady = Signal(object)
    taskFailed = Signal(str, str)
    taskCancelled = Signal(str)
    progressChanged = Signal(object)
    busyChanged = Signal(bool)
    staleResultDiscarded = Signal(str, int)

    def __init__(
        self,
        *,
        executor: BatchTaskExecutor | None = None,
        limits: BatchExecutionLimits = DEFAULT_BATCH_EXECUTION_LIMITS,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._executor = executor or execute_image_batch_task
        self._limits = limits
        self._signals = _BatchTaskSignals(self)
        self._signals.completed.connect(self._on_completed)
        self._signals.progress.connect(self._on_progress)
        self._pool = QThreadPool(self)
        self._pool.setMaxThreadCount(1)
        self._pool.setExpiryTimeout(5_000)
        self._generation = 0
        self._active: BatchRecipeRequest | None = None
        self._pending: BatchRecipeRequest | None = None
        self._cancellation: CancellationTokenSource | None = None
        self._busy = False
        self._closed = False

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def active_request(self) -> BatchRecipeRequest | None:
        return self._active

    def is_busy(self) -> bool:
        return self._active is not None or self._pending is not None

    def start(
        self,
        *,
        recipe: ImageProcessingRecipe,
        inputs: tuple[BatchRasterInput, ...],
        resource_directory: str | Path | None = None,
        available_disk_bytes: int | None = None,
    ) -> BatchRecipeRequest:
        if self._closed:
            raise RuntimeError("图像批处理控制器已经关闭")
        self._generation += 1
        request = BatchRecipeRequest(
            request_id=uuid4().hex,
            generation=self._generation,
            recipe=recipe,
            inputs=inputs,
            resource_directory=(
                None
                if resource_directory is None
                else str(Path(resource_directory).expanduser())
            ),
            available_disk_bytes=available_disk_bytes,
        )
        if self._active is not None:
            if self._cancellation is not None:
                self._cancellation.cancel()
            self._pending = request
        else:
            self._launch(request)
        return request

    def cancel(self) -> None:
        self._pending = None
        if self._cancellation is not None:
            self._cancellation.cancel()
        if self._active is None:
            self._set_busy(False)

    def close(self) -> None:
        self._closed = True
        self.cancel()

    def wait_for_done(self, timeout_ms: int = 5_000) -> bool:
        return self._pool.waitForDone(timeout_ms)

    def _launch(self, request: BatchRecipeRequest) -> None:
        cancellation = CancellationTokenSource()
        self._active = request
        self._cancellation = cancellation
        self._set_busy(True)
        self._pool.start(
            _BatchTaskRunnable(
                request=request,
                token=cancellation.token,
                executor=self._executor,
                limits=self._limits,
                signals=self._signals,
            )
        )

    def _set_busy(self, busy: bool) -> None:
        value = bool(busy)
        if self._busy == value:
            return
        self._busy = value
        self.busyChanged.emit(value)

    @Slot(object)
    def _on_progress(self, payload: object) -> None:
        if not isinstance(payload, BatchProgressUpdate):
            return
        if (
            self._active is not None
            and self._active.request_id == payload.request_id
            and payload.generation == self._generation
            and self._pending is None
            and not self._closed
        ):
            self.progressChanged.emit(payload)

    @Slot(object)
    def _on_completed(self, payload: object) -> None:
        if not isinstance(payload, _BatchTaskCompletion):
            return
        request = payload.request
        if self._active is None or self._active.request_id != request.request_id:
            self.staleResultDiscarded.emit(
                request.request_id,
                request.generation,
            )
            return
        self._active = None
        self._cancellation = None
        current = (
            not self._closed
            and request.generation == self._generation
            and self._pending is None
        )
        if current:
            if payload.cancelled or (
                payload.result is not None and payload.result.cancelled
            ):
                self.taskCancelled.emit(request.request_id)
            elif payload.error is not None:
                self.taskFailed.emit(request.request_id, payload.error)
            elif payload.result is not None and payload.result.stale:
                self.staleResultDiscarded.emit(
                    request.request_id,
                    request.generation,
                )
            elif payload.result is not None:
                self.batchReady.emit(payload.result)
        elif not payload.cancelled:
            self.staleResultDiscarded.emit(
                request.request_id,
                request.generation,
            )

        pending = self._pending
        self._pending = None
        if pending is not None and not self._closed:
            self._launch(pending)
        else:
            self._set_busy(False)


__all__ = [
    "BatchTaskExecutor",
    "ImageBatchTaskController",
    "execute_image_batch_task",
]
