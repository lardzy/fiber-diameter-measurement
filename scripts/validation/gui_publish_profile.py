"""Opt-in phase and GC tracing for the real Qt measurement benchmark.

No production scheduling changes. Inclusive phase times can overlap; use the
raw per-operation trace to locate a spike, then measure again without tracing.
"""
from __future__ import annotations

from contextlib import contextmanager
from functools import wraps
import gc
import sys
import threading
import time


class PublishProfile:
    def __init__(self):
        self.active = None
        self.operations = {}
        self._patches = []
        self._gui_thread = threading.get_ident()
        self._gc_started = {}
        self._cycles = self._cycle_counter()

    @staticmethod
    def _cycle_counter():
        if sys.platform != "win32":
            return lambda: None
        import ctypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        current = kernel.GetCurrentThread
        current.argtypes, current.restype = [], ctypes.c_void_p
        query = kernel.QueryThreadCycleTime
        query.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulonglong)]
        query.restype = ctypes.c_int
        handle = current()

        def read():
            value = ctypes.c_ulonglong()
            return value.value if query(handle, ctypes.byref(value)) else None

        # Counts are diagnostic only, never converted into elapsed time: CPU
        # frequency and Windows-on-ARM emulation can affect their interpretation.
        return read

    def wrap(self, owner, name, label=None):
        original = getattr(owner, name)

        @wraps(original)
        def measured(*args, **kwargs):
            profile = self.active
            if profile is None or threading.get_ident() != self._gui_thread:
                return original(*args, **kwargs)
            started = time.perf_counter()
            cycles = self._cycles()
            try:
                return original(*args, **kwargs)
            finally:
                phase = label or name
                finished_cycles = self._cycles()
                profile["phases"].append({"name": phase,
                    "start_ms": (started - profile["started_at"]) * 1000,
                    "duration_ms": (time.perf_counter() - started) * 1000,
                    "thread_cycles": (finished_cycles - cycles)
                        if cycles is not None and finished_cycles is not None else None})

        self._patches.append((owner, name, original))
        setattr(owner, name, measured)

    def restore(self):
        for owner, name, original in reversed(self._patches):
            setattr(owner, name, original)
        self._patches.clear()

    def _gc(self, phase, info):
        thread_id = threading.get_ident()
        if phase == "start":
            self._gc_started[thread_id] = time.perf_counter()
        elif self.active is not None:
            started = self._gc_started.pop(thread_id, None)
            if started is not None:
                self.active["gc"].append({"generation": info["generation"],
                    "gui_thread": thread_id == self._gui_thread,
                    "duration_ms": (time.perf_counter() - started) * 1000})

    @contextmanager
    def operation(self, identity):
        profile = {"started_at": time.perf_counter(), "phases": [], "gc": []}
        cycles = self._cycles()
        self.active = profile
        gc.callbacks.append(self._gc)
        try:
            yield
        finally:
            gc.callbacks.remove(self._gc)
            self.active = None
            finished_cycles = self._cycles()
            profile["thread_cycles"] = (finished_cycles - cycles) if cycles is not None and finished_cycles is not None else None
            profile["duration_ms"] = (time.perf_counter() - profile["started_at"]) * 1000
            self.operations[identity] = profile
