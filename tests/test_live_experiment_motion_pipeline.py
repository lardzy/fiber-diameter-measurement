"""Ordered field decisions and bounded presentation handoff under delayed work."""
from concurrent.futures import Future
import time

import numpy as np
import pytest

from fdm.live_experiment import ExperimentSession, default_materials
from fdm.live_experiment.controller import CalibrationSnapshot, LiveExperimentController
from fdm.live_experiment.frames import FrameHandle, PresentedFrameToken
from fdm.live_experiment.motion import FieldMotionDetector, MotionPreparer, MotionResult


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
                future.set_result(function(*args))
            except BaseException as error:
                future.set_exception(error)
            self.calls[index][2] = None

    def shutdown(self, **kwargs):
        for index in range(len(self.calls)):
            self.run(index)


def frame(sequence, generation=0):
    return FrameHandle.snapshot(np.full((48, 72, 3), 100 + sequence % 2, np.uint8),
        source_id='pipeline', sequence=sequence, generation=generation, live=True)


@pytest.fixture
def controller(tmp_path):
    session = ExperimentSession(save_images=False)
    session.add_operator('pipeline')
    session.set_materials(default_materials())
    c = LiveExperimentController(session, evidence_directory=tmp_path, verify_before_present=True)
    c._timer.stop()
    c.set_calibration(CalibrationSnapshot(1, 1))
    c._prepare_pool.shutdown(wait=True)
    c._motion_pool.shutdown(wait=True)
    c._prepare_pool = ControlledPool()
    c._motion_pool = ControlledPool()
    yield c
    c.close()
    assert c.budget.used_bytes == 0
    assert not c.budget._frames and not c.budget._leases


def receive(c, value):
    c._last_motion_submit = -1
    assert c.receive_frame(value)


def paint(c, value):
    c.mark_presented(PresentedFrameToken(value, c.field_epoch, 0, 1, (0, 0)))


def finish_motion(c, index):
    c._prepare_pool.run(index)
    c.poll()
    c._motion_pool.run(index)
    c.poll()


def ready(c):
    c.frameReady.connect(lambda value: paint(c, value))
    for sequence in range(2):
        receive(c, frame(sequence))
        finish_motion(c, sequence)
    assert c.state == 'ready' and c.presented.frame.sequence == 1


def test_arrival_and_preparation_cannot_promote_an_unverified_frame(controller):
    c = controller
    published = []
    c.frameReady.connect(published.append)
    receive(c, frame(0))
    assert not published and c.presented is None
    c._prepare_pool.run(0)
    c.poll()
    assert not published and c.presented is None
    c._motion_pool.run(0)
    c.poll()
    assert [f.sequence for f in published] == [0]
    assert c.presented is None  # A Qt paint is still required.
    assert c._presentation_lease is not None
    paint(c, published[0])
    assert c.presented.frame is published[0] and c._presentation_lease is None
    assert c.state == 'stabilizing'  # One stable decision is insufficient.


def test_preparation_overlaps_ordered_decision_and_keeps_only_newest_arrival(controller):
    c = controller
    receive(c, frame(0))
    c._prepare_pool.run(0)
    c.poll()  # Decision 0 is deliberately blocked.
    receive(c, frame(1))
    assert len(c._prepare_pool.calls) == 2
    c._prepare_pool.run(1)
    for sequence in range(2, 12):
        receive(c, frame(sequence))
    assert len(c._prepare_pool.calls) == 2 and len(c._motion_pool.calls) == 1
    c._motion_pool.run(0)
    c.poll()
    assert len(c._motion_pool.calls) == 2
    assert len(c._prepare_pool.calls) == 3
    assert c._prepare_future[2].frame.sequence == 11
    assert len(c.budget._frames) <= 4


def test_submitted_measurement_does_not_chase_unpresented_arrivals(controller):
    c = controller
    ready(c)
    c.start((20, 10))
    receive(c, frame(2))  # New source pixels are still being verified.
    intent_id = c.finish((20, 30), 'wool', method='free')
    c._jobs[0][1].result(timeout=2)
    for sequence in range(3, 20):
        receive(c, frame(sequence))
    c.poll()
    assert len(c.session.measurements) == 1
    record = next(iter(c.session.measurements.values()))
    assert record.frame_id == 'pipeline:0:1'
    assert record.provenance['submission_barrier_sequence'] == 1
    assert record.diameter_um == 20
    assert c.presented.frame.sequence == 1


def test_moving_frame_clears_draft_pending_and_markers_before_presentation(controller):
    c = controller
    ready(c)
    c.start((20, 10))
    c.finish((20, 30), 'wool', method='free')
    c._jobs[0][1].result(timeout=2)
    c.poll()
    record_ids = list(c.current_records)
    c.start((21, 10))
    epoch = c.field_epoch
    c._motion.compare_prepared = lambda _p, _k, identity, sequence: MotionResult(identity, sequence, False, 'translation')
    observed = []
    c.frameReady.connect(lambda value: observed.append((value.sequence, c.field_epoch, c.draft, list(c.current_records))))
    receive(c, frame(2))
    finish_motion(c, 2)
    assert observed == [(2, epoch + 1, None, [])]
    assert set(record_ids) <= c.session.measurements.keys()
    assert c.state == 'stabilizing'


def test_old_source_prepare_and_decision_are_discarded(controller):
    c = controller
    published = []
    c.frameReady.connect(published.append)
    receive(c, frame(0))
    c._prepare_pool.run(0)
    c.poll()
    receive(c, frame(1))
    receive(c, frame(0, generation=1))
    c._prepare_pool.run(1)
    c._motion_pool.run(0)
    c.poll()
    assert published == []
    assert c.calibration is None
    assert c._prepare_future[2].frame.generation == 1


def test_preparation_failure_releases_task_and_invalidates_source(controller):
    c = controller
    receive(c, frame(0))
    c._prepare_pool.calls[0][0].set_exception(RuntimeError('prepare failed'))
    c.poll()
    assert c.state == 'unavailable'
    assert c.latest_frame is None and c._prepare_future is None
    assert c.budget.used_bytes == 0
    # Avoid executing the intentionally failed job during controlled shutdown.
    c._prepare_pool.calls[0][2] = None


@pytest.mark.parametrize('shape', [(1, 1), (97, 139), (300, 420, 3), (2730, 4096, 3)])
def test_prepared_pipeline_matches_serial_detector_for_motion_and_reference_reset(shape):
    rng = np.random.default_rng(20151015)
    base = rng.integers(15, 241, shape, np.uint8)
    serial, pipelined, preparer = FieldMotionDetector(), FieldMotionDetector(), MotionPreparer()
    for sequence, pixels in enumerate((base, base.copy(), base ^ 1, np.roll(base, 93, 0), base)):
        f = FrameHandle.snapshot(pixels, source_id='parity', sequence=sequence, live=True)
        expected = serial.compare(f)
        prepared = preparer.prepare(f)
        actual = pipelined.compare_prepared(prepared, (f.source_id, f.generation, f.size), f.frame_id, f.sequence)
        assert expected == actual
        if not expected.stable:
            serial.reset()
            pipelined.reset()


def test_pipeline_still_rejects_a_late_result_for_an_unverified_display(controller):
    c = controller
    ready(c)
    c.start((20, 10))
    incoming = frame(2)
    receive(c, incoming)
    c.finish((20, 30), 'wool', method='free')
    c._jobs[0][1].result(timeout=2)
    # A renderer which violates the preferred handoff must not weaken the
    # ledger's independent latest-presented-frame guard or reset its deadline.
    paint(c, incoming)
    c.poll()
    assert not c.session.measurements
    intent = c._jobs[0][0]
    intent.submission_barrier_at = time.perf_counter() - 3
    rejected = []
    c.resultRejected.connect(rejected.append)
    c.poll()
    assert not c.session.measurements and not c._jobs
    assert any('两秒' in reason for reason in rejected)


@pytest.mark.parametrize('stage', ['prepare', 'decision'])
def test_enqueue_then_thread_start_failure_leaves_no_live_pixels_or_late_work(controller, stage):
    c = controller

    class FailedStartPool(ControlledPool):
        def submit(self, function, *args):
            super().submit(function, *args)
            raise RuntimeError('injected Thread.start failure')

    failed = FailedStartPool()
    if stage == 'prepare':
        c._prepare_pool = failed
        receive(c, frame(0))
    else:
        c._motion_pool = failed
        receive(c, frame(0))
        c._prepare_pool.run(0)
        c.poll()
    assert c.state == 'unavailable'
    assert c.budget.used_bytes == 0
    failed.run(0)  # Model the executor later draining its rejected WorkItem.
    assert failed.calls[0][0].result() is None
    assert c._preparer.previous is None or stage == 'decision'
