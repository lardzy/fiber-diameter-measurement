"""Exact shortcut and completion-notification regressions (no timing thresholds)."""
from concurrent.futures import Future
import threading
import time

import cv2
import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from fdm.live_experiment import ExperimentSession, default_materials
from fdm.live_experiment.controller import CalibrationSnapshot, LiveExperimentController
from fdm.live_experiment.frames import FrameHandle, PresentedFrameToken
from fdm.live_experiment.motion import MotionResult
import fdm.live_experiment.motion as motion
from scripts.validation.benchmark_cu_motion import reference_sobel_fraction


def make_frame(sequence=0, *, live=False):
    return FrameHandle.snapshot(np.full((36, 48, 3), 100, np.uint8),
        source_id="wake-test", sequence=sequence, live=live)


@pytest.fixture
def controller(tmp_path):
    session = ExperimentSession(save_images=False)
    session.add_operator("wake-test")
    session.set_materials(default_materials())
    value = LiveExperimentController(session, evidence_directory=tmp_path)
    value._timer.stop()  # Every asynchronous assertion must use the new wake.
    value.set_calibration(CalibrationSnapshot(1, 1))
    yield value
    value.close()


def present(controller, frame):
    controller.receive_frame(frame)
    controller.mark_presented(PresentedFrameToken(frame, controller.field_epoch, 0, 1, (0, 0)))


def pump_until(predicate):
    deadline = time.perf_counter() + 3
    while not predicate() and time.perf_counter() < deadline:
        QApplication.processEvents()
        time.sleep(.001)
    assert predicate()


class ControlledPool:
    def __init__(self):
        self.calls = []

    def submit(self, function, *args):
        future = Future()
        self.calls.append([future, function, args])
        return future

    def run(self, index):
        future, function, args = self.calls[index]
        if not future.done():
            try:
                result = function(*args)
            except BaseException as error:
                future.set_exception(error)
            else:
                future.set_result(result)
            self.calls[index][2] = None
        return future

    def shutdown(self, **_kwargs):
        for index in range(len(self.calls)):
            self.run(index)


class StubMotion:
    def __init__(self, moving=()):
        self.moving = set(moving)
        self.resets = 0

    def reset(self):
        self.resets += 1

    def compare_reduced(self, _gray, _key, frame_id, sequence):
        stable = sequence not in self.moving
        return MotionResult(frame_id, sequence, stable, "stable" if stable else "translation")

    def set_reference_reduced(self, *_args):
        raise AssertionError("Unstable completion must not build a soon-discarded reference")


def controlled_motion(controller, moving=()):
    controller._motion_pool.shutdown(wait=True)
    controller._motion_pool = ControlledPool()
    controller._motion = StubMotion(moving)
    return controller._motion_pool


def test_manual_completion_publishes_on_qt_without_watchdog_tick(controller, monkeypatch):
    present(controller, make_frame())
    released = threading.Event()
    original = controller._compute

    def blocked(intent):
        assert released.wait(2)
        return original(intent)

    monkeypatch.setattr(controller, "_compute", blocked)
    published_threads = []
    controller.recordAccepted.connect(lambda _record: published_threads.append(threading.get_ident()))
    controller.start((10, 5))
    controller.finish((10, 25), "wool", method="free")
    assert not controller.session.measurements
    released.set()
    pump_until(lambda: bool(controller.session.measurements))
    assert published_threads == [threading.get_ident()]


def test_worker_notifications_coalesce_and_queued_wake_is_harmless_after_close(controller, monkeypatch):
    calls = []
    monkeypatch.setattr(controller, "poll", lambda: calls.append(threading.get_ident()))
    worker = threading.Thread(target=lambda: [controller._request_poll() for _ in range(30)])
    worker.start()
    worker.join(2)
    assert not worker.is_alive() and not calls
    QApplication.processEvents()
    assert calls == [threading.get_ident()]
    controller._request_poll()
    controller.close()
    QApplication.processEvents()
    controller._request_poll()
    QApplication.processEvents()
    assert len(calls) == 1


def test_arrival_submits_immediately_and_pending_measurement_bypasses_idle_spacing(controller):
    pool = controlled_motion(controller)
    present(controller, make_frame(0, live=True))
    assert len(pool.calls) == 1  # No event processing or timer tick.
    pool.run(0)
    controller.poll()
    controller.state = "ready"
    controller._stable_count = 2
    controller._last_motion_submit = time.perf_counter()
    present(controller, make_frame(1, live=True))
    assert len(pool.calls) == 1  # Existing no-job spacing is retained.
    controller.start((10, 5))
    controller.finish((10, 25), "wool", method="free")
    assert len(pool.calls) == 2  # The job bypasses spacing before the next tick.
    controller._jobs[0][1].result(timeout=2)
    QApplication.processEvents()
    assert not controller.session.measurements  # Presented frame 1 is unchecked.
    pool.run(1)
    pump_until(lambda: bool(controller.session.measurements))


def test_completion_wake_does_not_publish_past_latest_moving_presented_frame(controller):
    pool = controlled_motion(controller, moving=(12,))
    present(controller, make_frame(10, live=True))
    pool.run(0)
    controller.poll()
    controller.state = "ready"
    controller._stable_count = 2
    epoch = controller.field_epoch
    controller.start((10, 5))
    controller.finish((10, 25), "wool", method="free")
    controller._jobs[0][1].result(timeout=2)
    controller._last_motion_submit = -1
    present(controller, make_frame(12, live=True))
    QApplication.processEvents()
    assert not controller.session.measurements
    pool.run(1)
    pump_until(lambda: controller.field_epoch > epoch)
    assert not controller.session.measurements
    # poll may already have queued the next field's first job; either way its
    # reset is mandatory and no reference is built inside the unstable job.
    assert controller._motion_reset or (len(pool.calls) > 2 and pool.calls[2][2][1] is True)


def test_motion_work_releases_holder_and_defers_reference_reset_to_next_job(controller):
    controlled_motion(controller, moving=(1,))
    for sequence, reset in ((1, False), (2, True)):
        frame = make_frame(sequence, live=True)
        holder = [frame, controller.budget.acquire(frame)]
        result = controller._motion_work(holder, reset)
        assert result.stable is (sequence == 2)
        assert holder == [] and controller.budget.used_bytes == 0
    assert controller._motion.resets == 1


def test_queued_poll_does_not_reenter_publication(controller, monkeypatch):
    calls = []

    def reentrant():
        calls.append(1)
        controller.poll()

    monkeypatch.setattr(controller, "_poll_once", reentrant)
    controller.poll()
    assert calls == [1]
    assert not controller._polling


@pytest.mark.parametrize("legacy_signed", [False, True])
@pytest.mark.parametrize("shape", [(1, 1), (1, 17), (19, 1), (23, 31)])
def test_low_difference_shortcut_is_exact_for_readonly_sliced_inputs(shape, legacy_signed, monkeypatch):
    rng = np.random.default_rng(20261015)
    previous = rng.integers(8, 248, (shape[0], shape[1] * 2), dtype=np.uint8)[:, ::2]
    delta = rng.integers(-8, 9, previous.shape, dtype=np.int16)
    current = (previous.astype(np.int16) + delta).astype(np.uint8)
    previous.setflags(write=False)
    current.setflags(write=False)
    expected = reference_sobel_fraction(previous, current, legacy_signed=legacy_signed)
    assert expected == 0.0
    monkeypatch.setattr(cv2, "Sobel", lambda *_a, **_kw: pytest.fail("Proven zero must skip Sobel"))
    assert motion.sobel_fraction(previous, current, legacy_signed=legacy_signed) == expected


@pytest.mark.parametrize("contrast", [9, 12, 28, 29, 127, 255])
@pytest.mark.parametrize("legacy_signed", [False, True])
def test_above_shortcut_boundary_preserves_complete_sobel_pipeline(contrast, legacy_signed, monkeypatch):
    previous = np.zeros((3, 3), np.uint8)
    current = previous.copy()
    current[:, 2] = [min(7, contrast), contrast, min(7, contrast)]
    expected = reference_sobel_fraction(previous, current, legacy_signed=legacy_signed)
    original = cv2.Sobel
    calls = []

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(cv2, "Sobel", counted)
    assert motion.sobel_fraction(previous, current, legacy_signed=legacy_signed) == expected
    assert len(calls) == 2


def test_shortcut_does_not_change_non_uint8_opencv_error_path(monkeypatch):
    monkeypatch.setattr(cv2, "Sobel", lambda *_a, **_kw: (_ for _ in ()).throw(RuntimeError("original path")))
    with pytest.raises(RuntimeError, match="original path"):
        motion.sobel_fraction(np.zeros((3, 3), np.float32), np.zeros((3, 3), np.float32))
