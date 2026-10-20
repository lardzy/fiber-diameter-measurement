"""Actual spawn transport, cache ownership and serial retirement invariants."""
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import shared_memory
from threading import Event
import gc
import os
import time
import weakref

import numpy as np
import pytest

from fdm.live_experiment.candidate_alignment import CandidateAlignmentMonitor, compare_candidate_frames
from fdm.live_experiment.candidate_alignment_worker import AlignmentProcessWorker, run_alignment_worker_self_check
from fdm.live_experiment.frames import FrameBudget, FrameHandle


def pixels():
    return np.random.default_rng(114).integers(0, 256, (96, 128), np.uint8)


def frame(sequence=1):
    return FrameHandle.snapshot(pixels(), source_id="source", sequence=sequence, live=False)


def assert_unlinked(name):
    with pytest.raises(FileNotFoundError):
        shared_memory.SharedMemory(name=name)


def test_real_worker_self_check_valid_drift_reset_and_cleanup():
    result = run_alignment_worker_self_check()
    assert result["ok"], result
    assert all(result[key] for key in ("same", "noisy", "drift", "reference_reset", "child_process", "closed"))
    assert result["worker_runtime"]["pid"] != os.getpid()


def test_lazy_process_reuses_buffers_and_restarts_after_its_own_child_crashes():
    worker = AlignmentProcessWorker()
    assert worker.process is None
    source = pixels()
    try:
        assert worker.compare(source, source).valid
        names = (worker.reference_segment.name, worker.current_segment.name)
        pid = worker.process.pid
        noise = np.clip(source.astype(np.int16) + 1, 0, 255).astype(np.uint8)
        expected = compare_candidate_frames(source, noise)
        actual = worker.compare(source, noise)
        assert actual.valid == expected.valid
        assert actual.shift == expected.shift and actual.correlation == expected.correlation
        assert (worker.reference_segment.name, worker.current_segment.name) == names
        worker.process.terminate()
        worker.process.join(timeout=3)
        assert worker.compare(source, source).valid
        assert worker.process.pid != pid
        for name in names:
            assert_unlinked(name)
        new_names = (worker.reference_segment.name, worker.current_segment.name)
        worker.invalidate()
        for name in new_names:
            assert_unlinked(name)
    finally:
        worker.close()


def test_worker_does_not_retain_parent_pixels_after_monitor_leases_are_released():
    worker = AlignmentProcessWorker()
    source = pixels()
    reference = weakref.ref(source)
    try:
        worker.compare(source, source)
        del source
        gc.collect()
        assert reference() is None
    finally:
        worker.close()


def sleeping_child(connection, threads):
    # An unresponsive child must not prevent bounded close or IPC failure recovery.
    time.sleep(10)


def test_ipc_timeout_reaps_only_owned_child_and_unlinks_shared_inputs(monkeypatch):
    from fdm.live_experiment import candidate_alignment_worker as module
    monkeypatch.setattr(module, "_alignment_main", sleeping_child)
    worker = AlignmentProcessWorker(timeout_seconds=.08)
    source = pixels()
    before = time.perf_counter()
    with pytest.raises(TimeoutError):
        worker.compare(source, source)
    assert time.perf_counter() - before < 2
    assert worker.process is None and worker.reference_segment is None
    worker.close()


def test_stop_interrupts_ipc_and_releases_monitor_budget(monkeypatch):
    from fdm.live_experiment import candidate_alignment_worker as module
    monkeypatch.setattr(module, "_alignment_main", sleeping_child)
    monitor = CandidateAlignmentMonitor(FrameBudget(), lambda *_: None)
    current = frame()
    monitor.reset(current)
    monitor.poll(current)
    deadline = time.perf_counter() + 1
    while monitor.worker.process is None and time.perf_counter() < deadline:
        time.sleep(.001)
    before = time.perf_counter()
    monitor.close()
    assert time.perf_counter() - before < 2
    assert monitor.budget.used_bytes == 0
    assert monitor.worker.process is None


class SerialWorker:
    def __init__(self):
        self.cached = None
        self.calls = []
        self.first_cleanup_started = Event()
        self.release_first_cleanup = Event()
        self.second_started = Event()
        self.release_second = Event()
        self.resets = 0
    def compare(self, a, b):
        self.calls.append("work")
        if self.resets:
            self.second_started.set()
            assert self.release_second.wait(3)
        self.cached = a
        return compare_candidate_frames(a, b)
    def invalidate(self):
        self.resets += 1
        self.calls.append("cleanup")
        if self.resets == 1:
            self.first_cleanup_started.set()
            assert self.release_first_cleanup.wait(3)
        self.cached = None
    def request_stop(self):
        self.release_first_cleanup.set()
        self.release_second.set()
    def close(self):
        self.cached = None


def test_retirement_runs_after_queued_generation_not_inside_previous_cleanup():
    worker = SerialWorker()
    monitor = CandidateAlignmentMonitor(FrameBudget(), lambda *_: None, worker=worker)
    a, b = frame(1), frame(2)
    try:
        monitor.reset(a)
        monitor.poll(a)
        monitor.future[0].result(3)
        monitor.reset(b)  # Cleanup A runs while GUI consumes A and queues B.
        assert worker.first_cleanup_started.wait(3)
        monitor.poll(b)
        monitor.reset()  # Retirement B must be queued *after* B work.
        worker.release_first_cleanup.set()
        assert worker.second_started.wait(3)
        assert monitor.budget.used_bytes >= b.size[0] * b.size[1] * 64
        worker.release_second.set()
        monitor.pool.submit(lambda: None).result(3)
        assert worker.calls == ["work", "cleanup", "work", "cleanup"]
        assert worker.cached is None
        assert monitor.poll(None) is None  # Both old generations discarded.
        assert monitor.budget.used_bytes == 0
    finally:
        monitor.close()


def test_many_unused_resets_do_not_enqueue_cleanup_or_create_a_process():
    worker = AlignmentProcessWorker()
    monitor = CandidateAlignmentMonitor(FrameBudget(), lambda *_: None, worker=worker)
    try:
        for _ in range(200):
            monitor.reset(frame())
        assert worker.process is None
        assert monitor.pool._work_queue.qsize() == 0
        monitor.reset()
        assert monitor.budget.used_bytes == 0
    finally:
        monitor.close()


def test_completed_ready_timestamp_precedes_gui_poll_and_includes_work_delay(monkeypatch):
    worker = SerialWorker()
    worker.release_first_cleanup.set()
    monitor = CandidateAlignmentMonitor(FrameBudget(), lambda *_: None, worker=worker)
    current = frame()
    original = monitor._work
    def delayed(*args):
        time.sleep(.04)
        return original(*args)
    monkeypatch.setattr(monitor, "_work", delayed)
    try:
        monitor.reset(current)
        monitor.poll(current)
        monitor.future[0].result(3)
        time.sleep(.04)
        monitor.poll(current)
        assert monitor.completed_ready_at - monitor.completed_submitted_at >= .04
        assert time.perf_counter() - monitor.completed_ready_at >= .04
    finally:
        monitor.close()


def test_failed_retirement_keeps_budget_and_blocks_new_work_until_confirmed_close():
    worker = SerialWorker()
    worker.release_first_cleanup.set()
    worker.release_second.set()
    monitor = CandidateAlignmentMonitor(FrameBudget(), lambda *_: None, worker=worker)
    current = frame()
    try:
        monitor.reset(current)
        monitor.poll(current)
        monitor.future[0].result(3)
        def unreaped():
            raise RuntimeError("child still owns buffers")
        worker.invalidate = unreaped
        monitor.reset(frame(2))
        monitor.pool.submit(lambda: None).result(3)
        assert monitor._failed_retirements
        assert monitor.budget.used_bytes >= current.size[0] * current.size[1] * 64
        monitor.poll(frame(3))
        assert monitor.future is None
    finally:
        monitor.close()  # Fake confirms resource release here; only now free tickets.
    assert monitor.budget.used_bytes == 0
