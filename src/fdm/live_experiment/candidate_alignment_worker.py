"""A lazy, serial alignment process with two reusable shared frame buffers.

The monitor reserves 64 B/source pixel for the serial comparison group, including
the shared inputs and the child's reference/FFT scratch. Reset retains that
reservation until invalidate() has dropped the old child cache. No Qt object or
camera handle crosses the process boundary.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict
import multiprocessing as mp
from multiprocessing import shared_memory
import os
import threading
import time
import weakref


def _alignment_main(connection, threads):
    import cv2
    import numpy as np
    from .candidate_alignment import CandidateReferenceCache, compare_candidate_frames

    cv2.setNumThreads(threads)  # Child only; never change the application's pool.
    segments = []
    reference = current = prepared = identity = None
    try:
        while True:
            command = connection.recv()
            if command is None:
                return
            kind, request_id, payload = command
            if kind == "reset":
                reference = current = prepared = identity = None
                for segment in segments:
                    segment.close()
                segments.clear()
                connection.send((request_id, None, None))
                continue
            generation, reference_name, current_name, shape = payload
            try:
                if identity != (generation, reference_name, current_name, shape):
                    reference = current = prepared = None
                    for segment in segments:
                        segment.close()
                    segments.clear()
                    segments.append(shared_memory.SharedMemory(name=reference_name))
                    segments.append(shared_memory.SharedMemory(name=current_name))
                    reference = np.ndarray(shape, dtype=np.uint8, buffer=segments[0].buf)
                    current = np.ndarray(shape, dtype=np.uint8, buffer=segments[1].buf)
                    reference.setflags(write=False)
                    current.setflags(write=False)
                    prepared = CandidateReferenceCache()
                    identity = (generation, reference_name, current_name, shape)
                result = compare_candidate_frames(reference, current, prepared=prepared)
                connection.send((request_id, asdict(result),
                                 {"pid": os.getpid(), "opencv_threads": cv2.getNumThreads()}))
            except Exception as exc:
                connection.send((request_id, None, {"error": type(exc).__name__ + ": " + str(exc)}))
    except (EOFError, BrokenPipeError, OSError):
        pass
    finally:
        reference = current = prepared = None
        for segment in segments:
            segment.close()
        connection.close()


class AlignmentProcessWorker:
    """Called exclusively by the monitor's one worker thread.

    request_stop() is the only cross-thread operation. It wakes the bounded IPC
    wait without blocking Qt; close() runs after the serial executor has drained.
    A failed/crashed process is disposed and lazily replaced on the next job.
    """
    def __init__(self, *, timeout_seconds=5.):
        if not 0 < timeout_seconds <= 5:
            raise ValueError("候选核验工作进程等待上限必须在 0–5 秒内。")
        self.timeout_seconds = float(timeout_seconds)
        self.process = self.connection = None
        self.reference_segment = self.current_segment = None
        self._reference = None
        self._generation = self._request = 0
        self._stop = threading.Event()
        self.closed = False
        self.timings = deque(maxlen=512)
        self.runtime = {}

    @staticmethod
    def _release(segment):
        if segment is not None:
            segment.close()
            try:
                segment.unlink()
            except FileNotFoundError:
                pass

    def _release_buffers(self):
        self._reference = None
        self._release(self.reference_segment)
        self._release(self.current_segment)
        self.reference_segment = self.current_segment = None

    def _start(self):
        if self.closed or self._stop.is_set():
            raise RuntimeError("候选核验工作进程已关闭。")
        import cv2
        context = mp.get_context("spawn")
        parent, child = context.Pipe()
        process = context.Process(target=_alignment_main, args=(child, cv2.getNumThreads()),
                                  daemon=True, name="FDM candidate alignment")
        try:
            process.start()
        except BaseException:
            parent.close()
            child.close()
            raise
        child.close()
        self.process, self.connection = process, parent

    def _exchange(self, kind, payload):
        if self.closed or self._stop.is_set():
            raise RuntimeError("候选核验已取消。")
        self._request += 1
        request_id = self._request
        self.connection.send((kind, request_id, payload))
        deadline = time.perf_counter() + self.timeout_seconds
        while True:
            if self._stop.is_set():
                raise RuntimeError("候选核验已取消。")
            if not self.process.is_alive():
                raise RuntimeError("候选核验工作进程异常退出。")
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                raise TimeoutError("候选核验工作进程未及时返回。")
            if self.connection.poll(min(.02, remaining)):
                response = self.connection.recv()
                if not isinstance(response, tuple) or len(response) != 3 or response[0] != request_id:
                    raise RuntimeError("候选核验返回了不匹配的任务。")
                if response[2] and response[2].get("error"):
                    raise RuntimeError(response[2]["error"])
                return response[1:]

    def compare(self, reference, current):
        import numpy as np
        from .candidate_alignment import AlignmentResult
        if reference.dtype != np.uint8 or current.dtype != np.uint8 or reference.shape != current.shape:
            raise ValueError("候选核验帧格式不匹配。")
        began = time.perf_counter()
        try:
            if self.process is None or not self.process.is_alive():
                self._dispose()
                self._start()
            if self._reference is None or self._reference() is not reference:
                # The previous job has completed: no child can still read the
                # current slot while it is reused or replaced.
                self.invalidate()
                if self.process is None:
                    self._start()
                self.reference_segment = shared_memory.SharedMemory(create=True, size=reference.nbytes)
                try:
                    self.current_segment = shared_memory.SharedMemory(create=True, size=current.nbytes)
                except BaseException:
                    self._release_buffers()
                    raise
                np.copyto(np.ndarray(reference.shape, dtype=np.uint8, buffer=self.reference_segment.buf), reference)
                self._reference = weakref.ref(reference)
                self._generation += 1
            np.copyto(np.ndarray(current.shape, dtype=np.uint8, buffer=self.current_segment.buf), current)
            copied = time.perf_counter()
            result, runtime = self._exchange("compare", (self._generation, self.reference_segment.name,
                                            self.current_segment.name, tuple(current.shape)))
            completed = time.perf_counter()
            if not isinstance(result, dict):
                raise RuntimeError("候选核验未返回测量结果。")
            value = AlignmentResult(**result)
            self.runtime = dict(runtime or {})
            self.timings.append({"copy_startup_ms": (copied - began) * 1000,
                                 "worker_ms": value.elapsed_ms, "total_ms": (completed - began) * 1000,
                                 "transport_scheduling_ms": max(0., (completed - began) * 1000 - value.elapsed_ms)})
            return value
        except BaseException:
            self._dispose()
            raise

    def invalidate(self):
        """Drop cached source pixels before releasing their monitor reservation."""
        try:
            if self.process is not None and self.process.is_alive() and self.reference_segment is not None:
                self._exchange("reset", None)
        except Exception:
            self._dispose()  # If reaping fails, retain mapped buffers/ownership.
            return
        self._release_buffers()

    def request_stop(self):
        self._stop.set()

    def _dispose(self):
        process, connection = self.process, self.connection
        self.process = self.connection = None
        if connection is not None:
            try:
                connection.send(None)
            except (EOFError, BrokenPipeError, OSError):
                pass
        if process is not None:
            process.join(timeout=.15)
            if process.is_alive():
                process.terminate()  # Only the Process object created above.
                process.join(timeout=.5)
            if process.is_alive():
                process.kill()
                process.join(timeout=.5)
            if process.is_alive():
                # Retain ownership rather than silently orphan a child or
                # unlink pixels it may still be reading.
                self.process, self.connection = process, connection
                raise RuntimeError("候选核验工作进程未能退出。")
            process.close()
        if connection is not None:
            connection.close()
        self._release_buffers()

    def close(self):
        self.closed = True
        self.request_stop()
        self._dispose()


def run_alignment_worker_self_check():
    """Exercise the actual spawn target from source or the frozen executable."""
    import cv2
    import numpy as np
    from .candidate_alignment import compare_candidate_frames
    rng = np.random.default_rng(117)
    source = cv2.GaussianBlur(rng.integers(0, 256, (96, 128), dtype=np.uint8), (5, 5), 0)
    noisy = np.clip(source.astype(np.int16) + rng.integers(-1, 2, source.shape), 0, 255).astype(np.uint8)
    worker = AlignmentProcessWorker()
    cases = {}
    runtime = {}
    names = []
    child = None
    try:
        for name, ref, current in (("same", source, source), ("noisy", source, noisy),
                                   ("drift", source, np.roll(source, 3, axis=1)),
                                   ("reference_reset", noisy, noisy)):
            expected, actual = compare_candidate_frames(ref, current), worker.compare(ref, current)
            cases[name] = all(getattr(expected, field) == getattr(actual, field)
                              for field in ("valid", "reason", "shift", "response", "correlation", "gradient_ratio"))
            if name in {"same", "noisy", "reference_reset"}:
                cases[name] = cases[name] and actual.valid
            elif name == "drift":
                cases[name] = cases[name] and not actual.valid
        runtime = dict(worker.runtime)
        cases["child_process"] = runtime.get("pid") not in {None, os.getpid()}
        names = [worker.reference_segment.name, worker.current_segment.name]
        child = worker.process
    finally:
        worker.close()
    released = True
    for name in names:
        try:
            segment = shared_memory.SharedMemory(name=name)
        except FileNotFoundError:
            continue
        released = False
        segment.close()
    # Process.close itself rejects a running process. Confirm the handle is
    # closed rather than only testing that the worker dropped its reference.
    child_closed = False
    if child is not None:
        try:
            child.is_alive()
        except ValueError:
            child_closed = True
    cases["closed"] = bool(released and child_closed and worker.process is None
                           and worker.reference_segment is None and worker.current_segment is None)
    return {"ok": all(cases.values()), **cases, "worker_runtime": runtime}
