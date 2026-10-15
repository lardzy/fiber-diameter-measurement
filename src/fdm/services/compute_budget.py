"""A process-wide FIFO admission gate for CPU-heavy analyses.

This controls when work starts. It does not preempt an ORT/PyTorch invocation
already executing, and manual diameter feedback never needs this permit.
"""
from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from functools import wraps
import threading


class ComputeBusyError(RuntimeError):
    pass


class ComputeRequest:
    def __init__(self, budget: "ComputeBudget", kind: str) -> None:
        self._budget = budget
        self.kind = str(kind)
        self.state = "queued"

    def try_acquire(self) -> bool:
        with self._budget._condition:
            if self.state == "acquired":
                return True
            if self.state != "queued":
                return False
            if self._budget._owner is None and self._budget._queue and self._budget._queue[0] is self:
                self._budget._queue.popleft()
                self._budget._owner = self
                self.state = "acquired"
                return True
            return False

    def release(self) -> None:
        with self._budget._condition:
            if self._budget._owner is self:
                self._budget._owner = None
            else:
                try:
                    self._budget._queue.remove(self)
                except ValueError:
                    pass
            self.state = "released"
            self._budget._condition.notify_all()

    cancel = release

    def wait(self, checkpoint=None) -> None:
        try:
            while True:
                if checkpoint is not None:
                    checkpoint()
                if self.try_acquire():
                    return
                if self.state != "queued":
                    raise ComputeBusyError("计算请求已取消。")
                # Never add a blocking admission wait to a GUI-thread service call.
                if threading.current_thread() is threading.main_thread():
                    raise ComputeBusyError("其他分析正在运行，请等待当前分析完成。")
                with self._budget._condition:
                    self._budget._condition.wait(timeout=0.025)
        except BaseException:
            self.release()
            raise


class ComputeBudget:
    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._queue = deque()
        self._owner = None
        self._local = threading.local()

    def request(self, kind: str) -> ComputeRequest:
        request = ComputeRequest(self, kind)
        with self._condition:
            self._queue.append(request)
            self._condition.notify_all()
        return request

    @property
    def active_kind(self) -> str | None:
        with self._condition:
            return self._owner.kind if self._owner is not None else None

    @contextmanager
    def slot(self, kind: str, checkpoint=None):
        if getattr(self._local, "depth", 0):
            self._local.depth += 1
            try:
                if checkpoint is not None:
                    checkpoint()
                yield
            finally:
                self._local.depth -= 1
            return
        request = self.request(kind)
        request.wait(checkpoint)
        self._local.depth = 1
        try:
            yield
        finally:
            self._local.depth = 0
            request.release()


HEAVY_COMPUTE_BUDGET = ComputeBudget()


def heavy_compute(kind: str, *, checkpoint_factory=None):
    """Wrap a worker-thread service entry, retaining its cancellation exception."""
    def decorate(method):
        @wraps(method)
        def admitted(*args, **kwargs):
            if checkpoint_factory is not None:
                checkpoint = checkpoint_factory(kwargs)
            else:
                cancel = kwargs.get("cancel_check")

                def checkpoint():
                    if cancel is not None and cancel():
                        raise RuntimeError("请求已取消。")

            with HEAVY_COMPUTE_BUDGET.slot(kind, checkpoint):
                return method(*args, **kwargs)
        return admitted
    return decorate
