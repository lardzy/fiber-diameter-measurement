"""One persistent spawn worker for expensive automatic CU analysis.

The GUI only copies a frame to shared memory and polls a local deque. Pipe reads
and result deserialization happen on a reader thread, never in the Qt event loop.
There is one in-flight job, no hidden unbounded queue. Timeout/crash/cancel drops
all unconfirmed work and rebuilds the process on the next submission.
"""
from __future__ import annotations

from collections import deque
from copy import deepcopy
import multiprocessing as mp
from multiprocessing import shared_memory
import os
import sys
import math
import threading
from time import monotonic, perf_counter
import uuid


from fdm.services.compute_budget import HEAVY_COMPUTE_BUDGET

DEFAULT_MEMORY_BUDGET = 2 * 1024**3


def _configure_worker_scheduling():
    """Keep the isolated Windows analysis below interactive work priority.

    This runs only inside the child. No GUI/capture thread priority changes,
    affinity restrictions, algorithm changes or timeout extensions are made.
    Failure is returned in worker diagnostics, never mistaken for success.
    """
    if sys.platform != "win32":
        return {"policy": "platform_default", "applied": True}
    import ctypes
    from ctypes import wintypes
    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.argtypes = []
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel32.SetPriorityClass.restype = wintypes.BOOL
        kernel32.GetPriorityClass.argtypes = [wintypes.HANDLE]
        kernel32.GetPriorityClass.restype = wintypes.DWORD
        process = kernel32.GetCurrentProcess()
        if not kernel32.SetPriorityClass(process, 0x4000):  # BELOW_NORMAL_PRIORITY_CLASS
            raise ctypes.WinError(ctypes.get_last_error())
        actual = int(kernel32.GetPriorityClass(process))
        return {"policy": "below_normal", "applied": actual == 0x4000, "priority_class": actual}
    except OSError as exc:
        return {"policy": "below_normal", "applied": False, "error": str(exc)}


def _worker_main(connection, cancelled) -> None:
    scheduling = _configure_worker_scheduling()
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
            response["worker_runtime"] = {"scheduling": scheduling, "opencv_requested_threads": 1,
                                          "opencv_threads": cv2.getNumThreads()}
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
        self._queued_at = 0.0
        self._dispatch_at = None
        self._task_timing = {}
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
                    # Timestamp before acquiring the GUI/watchdog lock. This
                    # includes recv deserialization, not GUI polling latency.
                    received_at = perf_counter()
                    with self._lock:
                        if generation != self._generation:
                            return
                        response["_parent_received_at"] = received_at
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
        from fdm.services.optical_config import ALGORITHM_IDS
        if algorithm not in ALGORITHM_IDS:
            raise ValueError("未知的全图分析算法。")
        allowed = {"max_candidates", "memory_budget", "preset", "config"}
        if set(limits) - allowed:
            raise TypeError(f"不支持的全图分析参数：{sorted(set(limits) - allowed)}")
        budget = limits.get("memory_budget", DEFAULT_MEMORY_BUDGET)
        if not isinstance(budget, int) or budget <= 0:
            raise ValueError("memory_budget 必须为正整数。")
        if source.nbytes > budget:
            raise MemoryError("automatic_memory_budget")
        # Pending jobs may wait for the shared compute permit. Own scalar
        # configuration now, not when a later dispatcher happens to send it.
        limits = deepcopy(limits)
        with self._lock:
            if self._closed:
                raise RuntimeError("自动分析工作进程已关闭。")
            self._harvest()
            if self._active is not None:
                raise RuntimeError("automatic_busy")
            identifier = task_id if task_id is not None else uuid.uuid4().hex
            copy_started = perf_counter()
            segment = shared_memory.SharedMemory(create=True, size=source.nbytes)
            try:
                np.copyto(np.ndarray(source.shape, dtype=np.uint8, buffer=segment.buf), source)
            except BaseException:
                segment.close()
                segment.unlink()
                raise
            self._segment, self._active = segment, identifier
            self._pending = (identifier, segment.name, tuple(source.shape), algorithm, limits)
            self._queued_at = perf_counter()
            self._dispatch_at = None
            self._task_timing = {"version": 1, "clock": "perf_counter",
                                 "shared_memory_ms": (self._queued_at - copy_started) * 1000}
            self._permit = HEAVY_COMPUTE_BUDGET.request("cu_automatic")
            self._dispatch()
            return identifier

    def _dispatch(self) -> None:
        if self._pending is None or self._permit is None or not self._permit.try_acquire():
            return
        try:
            self._task_timing["budget_wait_ms"] = (perf_counter() - self._queued_at) * 1000
            if self._process is None or not self._process.is_alive():
                # A crashed idle process has no pending computation to clean up.
                old = self._permit
                self._permit = None
                self._stop_process()
                self._permit = old
                start_called = perf_counter()
                self._start()
                self._task_timing["process_start_call_ms"] = (perf_counter() - start_called) * 1000
            self._cancelled.clear()
            self._dispatch_at = perf_counter()
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
                received_at = response.pop("_parent_received_at", None)
                timings = dict(self._task_timing)
                if self._dispatch_at is not None and received_at is not None:
                    elapsed = (received_at - self._dispatch_at) * 1000
                    timings["dispatch_to_receive_ms"] = elapsed
                    service = ((response.get("result") or {}).get("diagnostics") or {}).get("total_ms")
                    if isinstance(service, (int, float)) and math.isfinite(service) and 0 <= service <= elapsed:
                        # Duration subtraction only, never clocks across
                        # processes. Cold imports/startup and transport remain
                        # an explicitly combined overhead measurement.
                        timings["startup_transport_overhead_ms"] = elapsed - service
                    else:
                        timings["startup_transport_overhead_unavailable"] = (
                            "duration_difference_negative" if isinstance(service, (int, float))
                            and math.isfinite(service) and service > elapsed else "service_duration_unavailable")
                response["timings"] = timings
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
