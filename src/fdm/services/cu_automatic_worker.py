"""One persistent spawn worker for expensive automatic CU analysis.

The GUI only copies a frame to shared memory and polls a local deque. Pipe reads
and result deserialization happen on a reader thread, never in the Qt event loop.
There is one in-flight job, no hidden unbounded queue. Timeout/crash/cancel drops
all unconfirmed work and rebuilds the process on the next submission.
"""
from __future__ import annotations

from collections import deque
import multiprocessing as mp
from multiprocessing import shared_memory
import os
import threading
from time import monotonic
import uuid


from fdm.services.compute_budget import HEAVY_COMPUTE_BUDGET

DEFAULT_MEMORY_BUDGET = 2 * 1024**3


def _worker_main(connection, cancelled) -> None:
    # Configure before importing the native image stack in this fresh process.
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    import numpy as np
    import cv2
    cv2.setNumThreads(1)
    from fdm.services.cu_automatic_measurement import analyze_frame, CuAutomaticCancelled
    try:
        while True:
            command = connection.recv()
            if command is None:
                return
            task_id, name, shape, algorithm, limits = command
            segment = None
            try:
                segment = shared_memory.SharedMemory(name=name)
                image = np.ndarray(shape, dtype=np.uint8, buffer=segment.buf)
                image.setflags(write=False)
                result = analyze_frame(image, algorithm, cancelled.is_set, **limits)
                response = {"task_id": task_id, "result": result, "error": None}
            except CuAutomaticCancelled:
                response = {"task_id": task_id, "result": {"status": "cancelled", "reason": "cancelled", "candidates": [], "diagnostics": {}}, "error": None}
            except Exception as exc:
                response = {"task_id": task_id, "result": {"status": "failed", "reason": "algorithm_error", "candidates": [], "diagnostics": {}}, "error": {"type": type(exc).__name__, "message": str(exc)}}
            finally:
                if segment is not None:
                    segment.close()
            connection.send(response)
    except (EOFError, BrokenPipeError, OSError):
        return
    finally:
        connection.close()


class CuAutomaticWorker:
    """Create once per application and reuse; no camera or Qt object crosses IPC."""

    def __init__(self, *, timeout_seconds: float = 30.0) -> None:
        if not 0 < timeout_seconds <= 30:
            raise ValueError("自动分析超时必须大于 0 且不超过 30 秒。")
        self._timeout = float(timeout_seconds)
        self._context = mp.get_context("spawn")
        self._process = None
        self._connection = None
        self._cancelled = None
        self._segment = None
        self._active = None
        self._pending = None
        self._permit = None
        self._started = 0.0
        self._generation = 0
        self._closed = False
        self._responses = deque()
        self._received = deque()
        self._lock = threading.RLock()
        self._wake = threading.Event()

        def supervise() -> None:
            while not self._wake.wait(0.025):
                with self._lock:
                    if self._closed:
                        return
                    self._harvest()

        threading.Thread(target=supervise, name="CU automatic watchdog", daemon=True).start()

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._active is not None

    @property
    def state(self) -> str:
        with self._lock:
            if self._closed:
                return "closed"
            if self._active is None:
                return "idle"
            return "queued" if self._pending is not None else "running"

    def _start(self) -> None:
        parent, child = self._context.Pipe(duplex=True)
        self._cancelled = self._context.Event()
        process = self._context.Process(target=_worker_main, args=(child, self._cancelled), daemon=True, name="CU automatic measurement")
        process.start()
        child.close()
        self._process, self._connection = process, parent
        self._generation += 1
        generation = self._generation

        def receive() -> None:
            try:
                while True:
                    response = parent.recv()
                    with self._lock:
                        if generation != self._generation:
                            return
                        self._received.append(response)
            except (EOFError, OSError):
                return

        threading.Thread(target=receive, name="CU automatic result reader", daemon=True).start()

    def submit(self, image, algorithm: str = "traditional", task_id=None, **limits):
        import numpy as np
        source = np.asarray(image)
        if source.dtype != np.uint8 or source.ndim not in (2, 3) or (source.ndim == 3 and source.shape[2] not in (3, 4)):
            raise ValueError("自动分析输入必须为 uint8 灰度、BGR 或 BGRA 图像。")
        if min(source.shape[:2]) < 16:
            raise ValueError("自动分析图像宽高至少为 16 像素。")
        if algorithm not in {"traditional", "multiscale"}:
            raise ValueError("未知的全图分析算法。")
        allowed = {"max_candidates", "memory_budget", "preset"}
        if set(limits) - allowed:
            raise TypeError(f"不支持的全图分析参数：{sorted(set(limits) - allowed)}")
        budget = limits.get("memory_budget", DEFAULT_MEMORY_BUDGET)
        if not isinstance(budget, int) or budget <= 0:
            raise ValueError("memory_budget 必须为正整数。")
        if source.nbytes > budget:
            raise MemoryError("automatic_memory_budget")
        with self._lock:
            if self._closed:
                raise RuntimeError("自动分析工作进程已关闭。")
            self._harvest()
            if self._active is not None:
                raise RuntimeError("automatic_busy")
            identifier = task_id if task_id is not None else uuid.uuid4().hex
            segment = shared_memory.SharedMemory(create=True, size=source.nbytes)
            try:
                np.copyto(np.ndarray(source.shape, dtype=np.uint8, buffer=segment.buf), source)
            except BaseException:
                segment.close()
                segment.unlink()
                raise
            self._segment, self._active = segment, identifier
            self._pending = (identifier, segment.name, tuple(source.shape), algorithm, limits)
            self._permit = HEAVY_COMPUTE_BUDGET.request("cu_automatic")
            self._dispatch()
            return identifier

    def _dispatch(self) -> None:
        if self._pending is None or self._permit is None or not self._permit.try_acquire():
            return
        try:
            if self._process is None or not self._process.is_alive():
                # A crashed idle process has no pending computation to clean up.
                old = self._permit
                self._permit = None
                self._stop_process()
                self._permit = old
                self._start()
            self._cancelled.clear()
            self._connection.send(self._pending)
            self._pending = None
            self._started = monotonic()
        except Exception as exc:
            self._fail("worker_start_failed", str(exc))

    def _release_frame(self) -> None:
        if self._segment is not None:
            self._segment.close()
            try:
                self._segment.unlink()
            except FileNotFoundError:
                pass
            self._segment = None

    def _harvest(self) -> None:
        while self._received:
            response = self._received.popleft()
            if self._active == response.get("task_id"):
                self._responses.append(response)
                self._active = None
                self._release_frame()
                if self._permit is not None:
                    self._permit.release()
                    self._permit = None
        if self._active is None:
            return
        if self._pending is not None:
            self._dispatch()
            if self._pending is not None or self._active is None:
                return
        if self._process is None or not self._process.is_alive():
            self._fail("worker_crashed", "工作进程异常退出。")
        elif monotonic() - self._started >= self._timeout:
            self._fail("timeout", "全图分析超过 30 秒硬时限。")

    def poll(self) -> list[dict]:
        with self._lock:
            self._harvest()
            responses = list(self._responses)
            self._responses.clear()
            return responses

    def _fail(self, status: str, message: str) -> None:
        identifier = self._active
        self._stop_process()
        self._active = None
        self._pending = None
        self._release_frame()
        if identifier is not None:
            self._responses.append({"task_id": identifier, "result": {"status": status, "reason": status, "candidates": [], "diagnostics": {}}, "error": {"type": status, "message": message} if status != "cancelled" else None})

    def cancel(self, task_id=None) -> bool:
        with self._lock:
            self._harvest()
            if self._active is None or (task_id is not None and task_id != self._active):
                return False
            if self._cancelled is not None:
                self._cancelled.set()
            # A synchronous cv2 stage may not call a checkpoint. Killing this
            # isolated process is the hard guarantee; the live stream continues.
            self._fail("cancelled", "全图分析已取消。")
            return True

    def _stop_process(self) -> None:
        self._generation += 1
        process, connection = self._process, self._connection
        permit, self._permit = self._permit, None
        self._process = self._connection = None
        if permit is not None and permit.state != "acquired":
            permit.release()
            permit = None
        if process is not None:
            if process.is_alive():
                process.terminate()
            def reap() -> None:
                process.join(timeout=0.5)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=1.0)
                if not process.is_alive():
                    process.close()
                    if permit is not None:
                        permit.release()
            threading.Thread(target=reap, name="CU automatic process cleanup", daemon=True).start()
        elif permit is not None:
            permit.release()
        if connection is not None:
            connection.close()
        self._received.clear()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._wake.set()
            self._fail("cancelled", "自动分析工作进程已关闭。")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
