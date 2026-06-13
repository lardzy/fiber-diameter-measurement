from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from PySide6.QtCore import QObject, QThread


TASK_IMAGE_LOAD = "image_load"
TASK_AREA_INFERENCE = "area_inference"
TASK_PROMPT_SEGMENTATION = "prompt_segmentation"
TASK_FIBER_QUICK_GEOMETRY = "fiber_quick_geometry"
TASK_FIBER_QUICK_COMMIT_GEOMETRY = "fiber_quick_commit_geometry"
TASK_REFERENCE_INSTANCE = "reference_instance"
TASK_PREVIEW_ANALYSIS = "preview_analysis"

DEFAULT_WAIT_MS = 2000
REFERENCE_INSTANCE_WAIT_MS = 1500


@dataclass(slots=True)
class ManagedTaskHandle:
    name: str
    thread: QThread
    worker: QObject
    wait_ms: int = DEFAULT_WAIT_MS
    cancel_callback: Callable[[QObject], None] | None = None

    def cancel(self) -> None:
        if self.cancel_callback is not None:
            self.cancel_callback(self.worker)

    def stop(self, *, cancel: bool = True) -> None:
        if cancel:
            try:
                self.cancel()
            except Exception:
                pass
        if self.thread.isRunning():
            self.thread.quit()
            self.thread.wait(self.wait_ms)


class ThreadTaskManager:
    def __init__(self, *, parent: QObject | None = None) -> None:
        self._parent = parent
        self._handles: dict[str, ManagedTaskHandle] = {}

    def start_one_shot(
        self,
        name: str,
        *,
        worker_factory: Callable[[], QObject],
        connect_signals: Callable[[QObject], None] | None = None,
        start_slot_name: str = "run",
        finished_signal_name: str = "finished",
        cancel_callback: Callable[[QObject], None] | None = None,
        wait_ms: int = DEFAULT_WAIT_MS,
    ) -> ManagedTaskHandle:
        self.stop(name, cancel=True)
        thread = QThread(self._parent)
        worker = worker_factory()
        worker.moveToThread(thread)
        if connect_signals is not None:
            connect_signals(worker)
        finished_signal = getattr(worker, finished_signal_name)
        finished_signal.connect(thread.quit)
        finished_signal.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        handle = ManagedTaskHandle(
            name=name,
            thread=thread,
            worker=worker,
            wait_ms=wait_ms,
            cancel_callback=cancel_callback or _default_cancel_callback,
        )
        thread.finished.connect(lambda _handle=handle: self._cleanup_handle(_handle))
        thread.started.connect(getattr(worker, start_slot_name))
        self._handles[name] = handle
        thread.start()
        return handle

    def ensure_persistent(
        self,
        name: str,
        *,
        worker_factory: Callable[[], QObject],
        connect_signals: Callable[[QObject], None] | None = None,
        cancel_callback: Callable[[QObject], None] | None = None,
        wait_ms: int = DEFAULT_WAIT_MS,
    ) -> ManagedTaskHandle:
        existing = self._handles.get(name)
        if existing is not None and existing.thread.isRunning():
            return existing
        self.stop(name, cancel=False)
        thread = QThread(self._parent)
        worker = worker_factory()
        worker.moveToThread(thread)
        if connect_signals is not None:
            connect_signals(worker)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        handle = ManagedTaskHandle(
            name=name,
            thread=thread,
            worker=worker,
            wait_ms=wait_ms,
            cancel_callback=cancel_callback or _default_cancel_callback,
        )
        thread.finished.connect(lambda _handle=handle: self._cleanup_handle(_handle))
        self._handles[name] = handle
        thread.start()
        return handle

    def register_external(
        self,
        name: str,
        *,
        thread: QThread,
        worker: QObject | None = None,
        cancel_callback: Callable[[QObject], None] | None = None,
        wait_ms: int = DEFAULT_WAIT_MS,
    ) -> ManagedTaskHandle:
        self.stop(name, cancel=False)
        if worker is None:
            worker = QObject()
        handle = ManagedTaskHandle(
            name=name,
            thread=thread,
            worker=worker,
            wait_ms=wait_ms,
            cancel_callback=cancel_callback,
        )
        self._handles[name] = handle
        return handle

    def worker(self, name: str) -> QObject | None:
        handle = self._handles.get(name)
        return handle.worker if handle is not None else None

    def thread(self, name: str) -> QThread | None:
        handle = self._handles.get(name)
        return handle.thread if handle is not None else None

    def is_running(self, name: str) -> bool:
        handle = self._handles.get(name)
        return bool(handle is not None and handle.thread.isRunning())

    def stop(self, name: str, *, cancel: bool = True) -> None:
        handle = self._handles.pop(name, None)
        if handle is None:
            return
        handle.stop(cancel=cancel)

    def shutdown_all(self, *, cancel: bool = True) -> None:
        for name in list(self._handles):
            self.stop(name, cancel=cancel)

    def _cleanup_handle(self, handle: ManagedTaskHandle) -> None:
        if self._handles.get(handle.name) is handle:
            self._handles.pop(handle.name, None)


def _default_cancel_callback(worker: QObject) -> None:
    cancel = getattr(worker, "cancel", None)
    if callable(cancel):
        cancel()
