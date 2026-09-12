"""Bounded, console-independent traces for slow desktop operations.

The GUI only updates a small in-memory phase record. File writes and stack
snapshots run on a bounded number of observers, never a Qt timer or GUI slot.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Callable
from contextvars import ContextVar
from functools import wraps
from itertools import count
import sys
from threading import BoundedSemaphore, Event, Lock, Thread, enumerate as threads, get_ident
from time import monotonic
from typing import ParamSpec, TypeVar

from fdm.runtime_logging import append_runtime_log
from fdm.version import __version__


_SLOW_AT_SECONDS = (5.0, 30.0, 120.0)
_OBSERVER_SLOTS = BoundedSemaphore(4)
_IDS = count(1)
_CURRENT: ContextVar[_OperationTrace | None] = ContextVar("fdm_operation_trace", default=None)
P = ParamSpec("P")
T = TypeVar("T")


def _thread_stacks() -> str:
    # Do not load source files through linecache: frozen executables and source
    # trees on a share must not introduce more I/O while diagnosing a wait.
    names = {thread.ident: thread.name for thread in threads()}
    lines = []
    for ident, frame in list(sys._current_frames().items())[:32]:
        lines.append(f"thread={ident} name={names.get(ident, 'unknown')}")
        for _ in range(24):
            if frame is None:
                break
            lines.append(f"  {frame.f_code.co_filename}:{frame.f_lineno} in {frame.f_code.co_name}")
            frame = frame.f_back
    return "\n".join(lines)


class _OperationTrace:
    def __init__(self, name: str) -> None:
        self.name = name
        self.id = next(_IDS)
        parent = _CURRENT.get()
        self.parent_id = parent.id if parent is not None else None
        self.started_at = monotonic()
        self.owner_thread = get_ident()
        self.done = Event()
        self.lock = Lock()
        self.phases: deque[str] = deque(maxlen=16)
        self.outcome = "running"
        self.observer: Thread | None = None

    def phase(self, phase: str) -> None:
        with self.lock:
            self.phases.append(f"{monotonic() - self.started_at:.3f}s thread={get_ident()} {phase[:1000]}")

    def snapshot(self) -> str:
        with self.lock:
            phases, outcome = tuple(self.phases), self.outcome
        return (
            f"trace={self.id} parent={self.parent_id} operation={self.name} "
            f"version={__version__} elapsed_s={monotonic() - self.started_at:.3f} "
            f"owner_thread={self.owner_thread} outcome={outcome}\n"
            + "\n".join(phases)
        )

    def start(self) -> None:
        if not _OBSERVER_SLOTS.acquire(blocking=False):
            return

        def observe() -> None:
            try:
                append_runtime_log("Digital slide operation started", self.snapshot())
                for slow_at in _SLOW_AT_SECONDS:
                    remaining = self.started_at + slow_at - monotonic()
                    if self.done.wait(max(0.0, remaining)):
                        break
                    append_runtime_log("Slow digital slide operation", self.snapshot() + "\n" + _thread_stacks())
                # At most three stack snapshots, even for an indefinitely slow
                # OS call. This observer holds no QObject or operation inputs.
                self.done.wait()
                append_runtime_log("Digital slide operation finished", self.snapshot())
            except Exception:
                # Diagnostics are best effort and must never affect the
                # operation, including when the local log is unavailable.
                pass
            finally:
                _OBSERVER_SLOTS.release()

        try:
            self.observer = Thread(target=observe, name=f"fdm-slide-diagnostic-{self.id}", daemon=True)
            self.observer.start()
        except Exception:
            _OBSERVER_SLOTS.release()
            self.observer = None

    def finish(self, error: BaseException | None) -> None:
        with self.lock:
            self.outcome = "completed" if error is None else f"{type(error).__name__}: {str(error)[:1000]}"
        self.done.set()


def operation_phase(phase: str) -> None:
    trace = _CURRENT.get()
    if trace is not None:
        trace.phase(phase)


def diagnose_operation(name: str) -> Callable[[Callable[P, T]], Callable[P, T]]:
    def decorate(function: Callable[P, T]) -> Callable[P, T]:
        @wraps(function)
        def wrapped(*args: P.args, **kwargs: P.kwargs) -> T:
            trace = _OperationTrace(name)
            token = _CURRENT.set(trace)
            trace.start()
            error = None
            try:
                return function(*args, **kwargs)
            except BaseException as exc:
                error = exc
                raise
            finally:
                trace.finish(error)
                _CURRENT.reset(token)
        return wrapped
    return decorate
