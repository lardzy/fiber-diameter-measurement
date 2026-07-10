from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from threading import Lock
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


class TaskPhase(str, Enum):
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    TIMED_OUT = "timed_out"
    FINISHED = "finished"


@dataclass(frozen=True, slots=True)
class TaskStopResult:
    name: str
    phase_before: TaskPhase | None
    phase_after: TaskPhase | None
    cancel_requested: bool
    was_running: bool
    stopped: bool
    timed_out: bool

    @property
    def can_restart(self) -> bool:
        return self.stopped and not self.timed_out


@dataclass(slots=True, weakref_slot=True)
class ManagedTaskHandle:
    name: str
    thread: QThread
    worker: QObject
    wait_ms: int = DEFAULT_WAIT_MS
    cancel_callback: Callable[[QObject], None] | None = None
    phase: TaskPhase = TaskPhase.STARTING
    _phase_lock: Lock = field(default_factory=Lock, repr=False)

    def mark_running(self) -> None:
        with self._phase_lock:
            if self.phase == TaskPhase.STARTING:
                self.phase = TaskPhase.RUNNING

    def mark_finished(self) -> None:
        with self._phase_lock:
            self.phase = TaskPhase.FINISHED

    def _set_stopping(self) -> TaskPhase:
        with self._phase_lock:
            previous = self.phase
            if self.phase not in {TaskPhase.FINISHED, TaskPhase.TIMED_OUT}:
                self.phase = TaskPhase.STOPPING
            return previous

    def mark_timed_out(self) -> None:
        with self._phase_lock:
            if self.phase != TaskPhase.FINISHED:
                self.phase = TaskPhase.TIMED_OUT

    def cancel(self) -> None:
        if self.cancel_callback is not None:
            self.cancel_callback(self.worker)

    def stop(self, *, cancel: bool = True) -> TaskStopResult:
        phase_before = self._set_stopping()
        was_running = self.thread.isRunning()
        if cancel:
            try:
                self.cancel()
            except Exception:
                pass
        if was_running:
            self.thread.quit()
            stopped = bool(self.thread.wait(max(0, int(self.wait_ms))))
        else:
            stopped = True
        if was_running and not stopped:
            self.mark_timed_out()
        return TaskStopResult(
            name=self.name,
            phase_before=phase_before,
            phase_after=self.phase,
            cancel_requested=cancel,
            was_running=was_running,
            stopped=stopped,
            timed_out=was_running and not stopped,
        )


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
        self._stop_existing_or_raise(name, cancel=True)
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
        thread.started.connect(handle.mark_running)
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
            if existing.phase in {TaskPhase.STOPPING, TaskPhase.TIMED_OUT}:
                raise RuntimeError(f"任务 {name!r} 正在停止，不能同名重启。")
            return existing
        if existing is not None:
            raise RuntimeError(
                f"任务 {name!r} 已退出但尚未收到 thread.finished；不能同名重启。"
            )
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
        thread.started.connect(handle.mark_running)
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
        self._stop_existing_or_raise(name, cancel=False)
        if worker is None:
            worker = QObject()
        handle = ManagedTaskHandle(
            name=name,
            thread=thread,
            worker=worker,
            wait_ms=wait_ms,
            cancel_callback=cancel_callback,
            phase=TaskPhase.RUNNING if thread.isRunning() else TaskPhase.STARTING,
        )
        thread.started.connect(handle.mark_running)
        thread.finished.connect(lambda _handle=handle: self._cleanup_handle(_handle))
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

    def phase(self, name: str) -> TaskPhase | None:
        handle = self._handles.get(name)
        return handle.phase if handle is not None else None

    def stop(self, name: str, *, cancel: bool = True) -> TaskStopResult:
        handle = self._handles.get(name)
        if handle is None:
            return TaskStopResult(
                name=name,
                phase_before=None,
                phase_after=None,
                cancel_requested=False,
                was_running=False,
                stopped=True,
                timed_out=False,
            )
        result = handle.stop(cancel=cancel)
        return result

    def shutdown_all(self, *, cancel: bool = True) -> list[TaskStopResult]:
        results: list[TaskStopResult] = []
        for name in list(self._handles):
            results.append(self.stop(name, cancel=cancel))
        return results

    def _stop_existing_or_raise(self, name: str, *, cancel: bool) -> None:
        existing = self._handles.get(name)
        if existing is None:
            return
        result = self.stop(name, cancel=cancel)
        if not result.can_restart:
            raise RuntimeError(
                f"任务 {name!r} 在 {existing.wait_ms} ms 内未停止；"
                "旧任务句柄仍被保留，不能同名重启。"
            )
        if self._handles.get(name) is existing:
            raise RuntimeError(
                f"任务 {name!r} 已停止但 thread.finished 尚未送达；不能同名重启。"
            )

    def _cleanup_handle(self, handle: ManagedTaskHandle) -> None:
        handle.mark_finished()
        if self._handles.get(handle.name) is handle:
            self._handles.pop(handle.name, None)


def _default_cancel_callback(worker: QObject) -> None:
    cancel = getattr(worker, "cancel", None)
    if callable(cancel):
        cancel()
