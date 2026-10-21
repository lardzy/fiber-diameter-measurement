"""Retention is an ephemeral, separately timed view of successful canvas picks."""
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtTest import QSignalSpy, QTest

from fdm.live_experiment import ExperimentSession, default_materials
from fdm.live_experiment.controller import CalibrationSnapshot, LiveExperimentController
from fdm.live_experiment.frames import FrameHandle, PresentedFrameToken
from fdm.live_experiment.selection_retention import FieldEndCause, SelectionRetention


@pytest.fixture
def session():
    value = ExperimentSession(save_images=False)
    value.add_operator('A')
    value.set_materials(default_materials())
    return value


def record(session, number=1):
    return session.add_measurement(20, session.operators[session.active_operator_id].materials['warp'][0].id,
        endpoints=[[10, 20], [30, 20]], provenance={'measurement_number': number})


def test_multiple_groups_keep_original_deadlines_and_only_geometry(session):
    clock = [100.]
    layer = SelectionRetention(session, clock=lambda: clock[0])
    a, b, c = [record(session, i) for i in range(1, 4)]
    revision = session.revision
    layer.remember(a); layer.remember(b)
    assert not layer.visible_marks()
    layer.end_field(FieldEndCause.MOTION)
    assert {m['record_id'] for m in layer.visible_marks()} == {a.id, b.id}
    assert set(layer.visible_marks()[0]) == {'record_id', 'start', 'end', 'number', 'color'}
    clock[0] += .5
    layer.end_field(FieldEndCause.MOTION)  # Repeated refocusing cannot renew.
    layer.remember(c)
    layer.end_field(FieldEndCause.NEW_FIELD)
    clock[0] += .51
    assert [m['record_id'] for m in layer.visible_marks()] == [c.id]
    clock[0] += .5
    assert layer.visible_marks() == ()
    layer.prune()
    assert not layer._timer.isActive()
    assert session.revision == revision


def test_duration_changes_only_future_groups_zero_clears(session):
    clock = [0.]
    layer = SelectionRetention(session, clock=lambda: clock[0])
    layer.remember(record(session)); layer.end_field(FieldEndCause.MOTION)
    layer.set_duration(10)
    clock[0] = 1.01
    assert not layer.visible_marks()
    layer.remember(record(session, 2)); layer.end_field(FieldEndCause.MOTION)
    clock[0] += 2
    assert len(layer.visible_marks()) == 1
    layer.set_duration(0)
    assert not layer.visible_marks() and not layer._timer.isActive()


@pytest.mark.parametrize('value', [-1, 10.1, float('nan'), float('inf')])
def test_invalid_duration_rejected(session, value):
    with pytest.raises(ValueError):
        SelectionRetention(session).set_duration(value)


def test_context_and_deleted_records_clear_without_undo_replay(session):
    layer = SelectionRetention(session)
    a = record(session)
    layer.remember(a); layer.end_field(FieldEndCause.MOTION)
    session.delete_measurement(a.id); layer.prune()
    assert not layer.visible_marks()
    session.undo(); layer.prune()
    assert not layer.visible_marks()
    layer.remember(session.measurements[a.id]); layer.end_field(FieldEndCause.MOTION)
    layer.end_field(FieldEndCause.CONTEXT)
    assert not layer.visible_marks() and not layer._picked


def test_real_timer_expires_without_new_capture_frames(session):
    layer = SelectionRetention(session)
    layer.set_duration(.1)
    layer.remember(record(session)); layer.end_field(FieldEndCause.MOTION)
    signal = QSignalSpy(layer.changed)
    QTest.qWait(160)
    assert signal.count() >= 1 and not layer.visible_marks()
    assert not layer._timer.isActive()


@pytest.fixture
def controller(session, tmp_path):
    c = LiveExperimentController(session, evidence_directory=tmp_path)
    c._timer.stop()
    c.set_calibration(CalibrationSnapshot(1, 1))
    f = FrameHandle.snapshot(np.full((80, 120, 3), 120, np.uint8), source_id='retention', live=False)
    c.receive_frame(f)
    c.mark_presented(PresentedFrameToken(f, c.field_epoch, 0, 1, (0, 0)))
    c._automatic_context = dict(task_id='test', token=c.presented, calibration=c.calibration,
        operator_id=session.active_operator_id, direction='warp', algorithm='traditional',
        lease=c.budget.acquire(f), completed=True, batch_id='retention-test')
    c.candidates = [dict(start=[10 + i * 30, 10], end=[10 + i * 30, 30], width_px=20) for i in range(3)]
    c._ensure_candidate_ids()
    yield c
    c.close()
    assert c.budget.used_bytes == 0


def test_only_successful_canvas_picks_retained_not_bulk_or_manual(controller):
    c = controller
    material = c.session.operators[c.session.active_operator_id].materials['warp'][0].id
    c.submit_candidate(c.candidates[0]['candidate_id'], material)
    c.accept_candidates([(1, material)])
    manual = c.session.add_measurement(12, material, field_id=c.field_id, endpoints=[[1, 1], [13, 1]])
    c.current_records.append(manual.id)
    assert len(c.session.measurements) == 3
    picked = c.candidates[0]['record_id']
    c.new_field()
    assert [m['record_id'] for m in c.selection_retention.visible_marks()] == [picked]
    assert len(c.session.measurements) == 3
    assert not c.current_records and not c.candidates


def test_refresh_does_not_forget_successful_picks(controller):
    c = controller
    material = c.session.operators[c.session.active_operator_id].materials['warp'][0].id
    c.submit_candidate(c.candidates[0]['candidate_id'], material)
    c.discard_candidates()
    c.end_field(cause=FieldEndCause.MOTION)
    assert len(c.selection_retention.visible_marks()) == 1


def test_source_change_clears_retained_geometry(controller):
    c = controller
    material = c.session.operators[c.session.active_operator_id].materials['warp'][0].id
    c.submit_candidate(c.candidates[0]['candidate_id'], material)
    c.new_field()
    assert c.selection_retention.visible_marks()
    c.observe_source_identity(('other', 0, (120, 80)))
    assert not c.selection_retention.visible_marks()
