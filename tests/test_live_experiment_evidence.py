from pathlib import Path
import hashlib
import threading
import time
import weakref

import numpy as np
import cv2
import pytest

from fdm.live_experiment.evidence import EvidenceStore
from fdm.live_experiment.frames import FrameBudget, FrameHandle
from fdm.live_experiment.png_stream import ENCODER_REVISION, png_workspace_bytes
import fdm.live_experiment.evidence as evidence_module


@pytest.mark.parametrize("channels", [None, 3, 4])
def test_saved_png_is_lossless_and_shared_for_same_frame(tmp_path, channels):
    shape = (64, 80) if channels is None else (64, 80, channels)
    pixels = np.random.default_rng(45).integers(0, 256, shape, dtype=np.uint8)
    frame = FrameHandle.snapshot(pixels, source_id="lossless")
    store = EvidenceStore(tmp_path)
    first = store.reserve(frame)
    assert store.reserve(frame) is first
    completed = store.close()
    assert len(completed) == 1 and completed[0]["state"] == "saved"
    assert completed[0]["encoder"] == ENCODER_REVISION
    decoded = cv2.imdecode(np.fromfile(completed[0]["path"], dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    np.testing.assert_array_equal(decoded, pixels)
    assert len(list(tmp_path.glob("*.png"))) == 1
    assert store.budget.used_bytes == 0
    assert not store.budget._scratch


def test_write_and_cleanup_failure_preserves_failed_record_and_releases_frame(tmp_path, monkeypatch):
    original_open, original_unlink = Path.open, Path.unlink

    def denied_open(path, *args, **kwargs):
        if path.suffix == ".pending":
            raise PermissionError("disk became read-only")
        return original_open(path, *args, **kwargs)

    def denied_unlink(path, *args, **kwargs):
        if path.suffix == ".pending":
            raise PermissionError("cleanup denied")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", denied_open)
    monkeypatch.setattr(Path, "unlink", denied_unlink)
    budget = FrameBudget()
    store = EvidenceStore(tmp_path, budget=budget)
    frame = FrameHandle.snapshot(np.zeros((16, 16), np.uint8), source_id="failure")
    try:
        store.reserve(frame)
        deadline = time.monotonic() + 3
        completed = []
        while store.pending and time.monotonic() < deadline:
            completed.extend(store.poll())
            time.sleep(.001)
        assert len(completed) == 1
        assert completed[0]["state"] == "failed"
        assert "disk became read-only" in completed[0]["error"]
        assert "cleanup denied" in completed[0]["error"]
        assert not store.pending
        assert budget.used_bytes == 0
        assert store.reserve(frame)["state"] == "failed"
    finally:
        store.close()


def static_frame(value=7, sequence=0):
    return FrameHandle.snapshot(np.full((18, 23, 3), value, np.uint8),
        source_id="evidence-transaction", sequence=sequence, live=False)


def test_submit_queues_then_thread_start_fails_aborted_work_never_reads_pixels(tmp_path, monkeypatch):
    store = EvidenceStore(tmp_path)
    adjust = store._pool._adjust_thread_count
    encoder = evidence_module.write_png_stream
    encoded_values = []

    def fail_start():
        # ThreadPoolExecutor.submit has already queued its _WorkItem here.
        raise RuntimeError("cannot start new thread")

    def capture(pixels, handle):
        encoded_values.append(int(pixels.flat[0]))
        return encoder(pixels, handle)

    monkeypatch.setattr(evidence_module, "write_png_stream", capture)
    monkeypatch.setattr(store._pool, "_adjust_thread_count", fail_start)
    rejected = static_frame()
    try:
        with pytest.raises(RuntimeError, match="cannot start"):
            store.reserve(rejected)
        assert not store._entries and not store._futures and not store._tasks
        assert store.budget.used_bytes == 0
        assert not store.budget._leases and not store.budget._scratch
        monkeypatch.setattr(store._pool, "_adjust_thread_count", adjust)
        accepted = static_frame(9, 1)
        store.reserve(accepted)
        completed = store.close()
        assert encoded_values == [9]
        assert len(completed) == 1 and completed[0]["state"] == "saved"
        rejected_name = hashlib.sha256(rejected.frame_id.encode()).hexdigest() + ".png"
        assert not (tmp_path / rejected_name).exists()
        assert store.budget.used_bytes == 0
    finally:
        store.close()


def test_registration_failure_after_submit_aborts_worker_and_rolls_back(tmp_path, monkeypatch):
    class FailOnceDict(dict):
        fail = True

        def __setitem__(self, key, value):
            if self.fail:
                self.fail = False
                raise RuntimeError("registration failed")
            return super().__setitem__(key, value)

    store = EvidenceStore(tmp_path)
    store._tasks = FailOnceDict()
    encoder = evidence_module.write_png_stream
    encoded_values = []

    def capture(pixels, handle):
        encoded_values.append(int(pixels.flat[0]))
        return encoder(pixels, handle)

    monkeypatch.setattr(evidence_module, "write_png_stream", capture)
    try:
        with pytest.raises(RuntimeError, match="registration failed"):
            store.reserve(static_frame(7))
        assert not store._entries and not store._futures and not store._tasks
        assert store.budget.used_bytes == 0
        store.reserve(static_frame(9, 1))
        completed = store.close()
        assert encoded_values == [9]
        assert len(completed) == 1 and completed[0]["state"] == "saved"
        assert store.budget.used_bytes == 0
    finally:
        store.close()


@pytest.mark.parametrize("other_consumer", [False, True])
def test_raw_owner_released_before_fsync_but_other_leases_and_workspace_remain(
        tmp_path, monkeypatch, other_consumer):
    store = EvidenceStore(tmp_path)
    entered = threading.Event()
    release = threading.Event()
    fsync = evidence_module.os.fsync

    def gated_fsync(fd):
        entered.set()
        if not release.wait(3):
            raise RuntimeError("test fsync gate timed out")
        return fsync(fd)

    monkeypatch.setattr(evidence_module.os, "fsync", gated_fsync)
    value = static_frame()
    pixels = weakref.ref(value.pixels)
    raw_bytes = value.nbytes
    keeper = store.budget.acquire(value) if other_consumer else None
    workspace = png_workspace_bytes(value.size[0], 3)
    try:
        first = store.reserve(value)
        assert store.reserve(value) is first
        del value
        assert entered.wait(3)
        assert store.budget.used_bytes == workspace + (raw_bytes if other_consumer else 0)
        assert (pixels() is not None) is other_consumer
        if keeper is not None:
            store.budget.release(keeper)
            keeper = None
            assert pixels() is None
        release.set()
        completed = store.close()
        assert len(completed) == 1 and completed[0]["state"] == "saved"
        assert store.budget.used_bytes == 0
        assert not store.budget._frames and not store.budget._scratch
    finally:
        release.set()
        if keeper is not None:
            store.budget.release(keeper)
        store.close()


@pytest.mark.parametrize("failure", ["fsync", "replace"])
def test_commit_failure_retained_and_next_frame_can_save(tmp_path, monkeypatch, failure):
    store = EvidenceStore(tmp_path)
    original = getattr(evidence_module.os, failure)

    def failed(*args, **kwargs):
        raise OSError("injected commit failure")

    monkeypatch.setattr(evidence_module.os, failure, failed)
    first = static_frame()
    try:
        store.reserve(first)
        failures = store.flush()
        assert len(failures) == 1 and failures[0]["state"] == "failed"
        assert "injected commit failure" in failures[0]["error"]
        assert store.reserve(first)["state"] == "failed"
        assert store.budget.used_bytes == 0
        assert not list(tmp_path.glob("*.pending"))
        monkeypatch.setattr(evidence_module.os, failure, original)
        second = static_frame(9, 1)
        store.reserve(second)
        completed = store.close()
        assert len(completed) == 1 and completed[0]["state"] == "saved"
        assert store.budget.used_bytes == 0 and not store.budget._scratch
    finally:
        store.close()


def test_evidence_stream_handles_reverse_strided_frame_without_changing_input(tmp_path):
    owner = np.random.default_rng(5).integers(0, 256, (40, 60, 4), dtype=np.uint8)
    view = owner[::-2, ::-3]
    view.flags.writeable = False
    value = FrameHandle("strided", 0, 0, time.perf_counter(), view, "BGRA", False)
    store = EvidenceStore(tmp_path)
    try:
        store.reserve(value)
        completed = store.close()
        data = Path(completed[0]["path"]).read_bytes()
        np.testing.assert_array_equal(cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED), view)
        assert completed[0]["sha256"] == hashlib.sha256(data).hexdigest()
        assert not view.flags.writeable
        assert store.budget.used_bytes == 0
    finally:
        store.close()


def test_close_blocks_new_admission_while_draining_existing_writer(tmp_path, monkeypatch):
    store = EvidenceStore(tmp_path)
    entered, release = threading.Event(), threading.Event()
    encoder = evidence_module.write_png_stream
    closed = []

    def gated(pixels, handle):
        entered.set()
        if not release.wait(3):
            raise RuntimeError("test encoder gate timed out")
        return encoder(pixels, handle)

    monkeypatch.setattr(evidence_module, "write_png_stream", gated)
    thread = threading.Thread(target=lambda: closed.extend(store.close()))
    try:
        store.reserve(static_frame())
        assert entered.wait(3)
        thread.start()
        deadline = time.monotonic() + 3
        while not store._closed and time.monotonic() < deadline:
            time.sleep(.001)
        assert store._closed
        with pytest.raises(RuntimeError, match="存储已关闭"):
            store.reserve(static_frame(9, 1))
        release.set()
        thread.join(3)
        assert not thread.is_alive()
        assert len(closed) == 1 and closed[0]["state"] == "saved"
        assert store.budget.used_bytes == 0
    finally:
        release.set()
        if thread.ident is not None:
            thread.join(3)
        store.close()
