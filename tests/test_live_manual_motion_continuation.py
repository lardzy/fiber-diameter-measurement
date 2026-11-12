"""A changing field must not steal a content trial's unfinished manual input."""
from copy import deepcopy

import numpy as np
import pytest
from PySide6.QtCore import QEvent, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from fdm.live_experiment.frames import FrameHandle
from fdm.live_experiment.motion import MotionResult
from fdm.live_experiment.selection_retention import FieldEndCause
from test_live_experiment_content_workspace import workspace
from test_live_experiment_motion_pipeline import ControlledPool
from test_live_explicit_manual_gesture import click, mouse, prepare, wait_record
from test_live_mixed_presented_frame import _prepare_live_stripe, _pump


def controlled_preview(view, *, mode="manual", gesture="click", method="free"):
    """Keep actual delivery/paint/input; control only worker completion order."""
    prepare(view, mode, gesture)
    view.measure_mode.setCurrentIndex(view.measure_mode.findData(method))
    c = view.controller
    assert c._motion_future is None and c._prepare_future is None
    for pool in (c._prepare_pool, c._motion_pool):
        if pool is not None:
            pool.shutdown(wait=True)
    c._prepare_pool = ControlledPool()
    c._motion_pool = ControlledPool()
    c.verify_before_present = True
    pixels = np.full_like(c.presented.frame.pixels, 240)
    pixels[20:220, 130:146] = 30
    if pixels.shape[2] == 4:
        pixels[:, :, 3] = 255
    step(view, pixels)
    step(view, pixels)
    assert c.state == "ready"
    return pixels


def step(view, pixels, *, stable=True, paint=True):
    c = view.controller
    previous = c.latest_frame
    frame = FrameHandle.snapshot(pixels, source_id=previous.source_id,
        generation=previous.generation, sequence=previous.sequence + 1,
        live=True, metadata=previous.metadata)
    c._motion.compare_prepared = lambda _p, _k, identity, sequence: MotionResult(
        identity, sequence, stable, "stable" if stable else "translation")
    c._last_motion_submit = -1
    view._receive_frame(frame)
    c._prepare_pool.run(len(c._prepare_pool.calls) - 1)
    c.poll()
    c._motion_pool.run(len(c._motion_pool.calls) - 1)
    c.poll()
    if paint:
        QApplication.processEvents()
        view.canvas.repaint()
    return frame


@pytest.mark.parametrize("mode", ["manual", "mixed"])
@pytest.mark.parametrize("method", ["free", "assisted"])
def test_two_points_survive_motion_and_finish_on_one_new_visible_frame(workspace, monkeypatch, mode, method):
    view, c = workspace, workspace.controller
    pixels = controlled_preview(view, mode=mode, method=method)
    c.set_evidence_options({"version": 1, "mode": "full_png"})
    click(view, (100, 100))
    original_token = c.draft[0]
    identity = deepcopy(c._classification_target)
    options = deepcopy(c._draft_evidence_options)
    # The new image has a wider fibre; fitting must use this frame, not the
    # image retained when the first endpoint was selected.
    moved = np.full_like(pixels, 240)
    moved[20:220, 150:176] = 30
    if moved.shape[2] == 4:
        moved[:, :, 3] = 255
    epoch = c.field_epoch
    step(view, moved, stable=False)
    assert c.field_epoch == epoch + 1
    assert c.draft[1] == pytest.approx((100, 100))
    assert c._classification_target["id"] == identity["id"]
    assert c._classification_target["number"] == identity["number"]
    assert c._classification_target["field_epoch"] == c.field_epoch
    assert c.state == "stabilizing"
    click(view, (200, 100))
    assert c.draft is not None and not c._jobs
    step(view, moved)
    step(view, moved)
    visible = c.presented
    assert visible.frame is not original_token.frame and c.state == "ready"
    c.set_evidence_options({"version": 1, "mode": "disabled"})
    evidence_calls = []

    def reserve(frame, **kwargs):
        evidence_calls.append((frame, kwargs))
        return {"state": "disabled"}

    monkeypatch.setattr(c.evidence, "reserve", reserve)
    click(view, (200, 100))
    assert c.draft is None and len(c._jobs) == 1
    intent = c._jobs[0][0]
    assert intent.id == identity["id"]
    assert intent.token is visible
    assert intent.start == pytest.approx((100, 100))
    assert intent.end == pytest.approx((200, 100))
    assert intent.evidence_options == options
    record = wait_record(view)
    assert record.frame_id == visible.frame.frame_id
    assert record.diameter_um == pytest.approx(25 if method == "assisted" else 100, abs=.1)
    assert evidence_calls[0][0] is visible.frame
    assert evidence_calls[0][1]["options"] == options
    assert not c._pointer_frames
    click(view, (80, 130))
    assert c._classification_target["number"] == identity["number"] + 1
    assert len(c.session.measurements) == 1


def test_ready_decision_without_new_paint_keeps_start_and_does_not_submit(workspace):
    view, c = workspace, workspace.controller
    pixels = controlled_preview(view)
    click(view, (100, 100))
    visible = c.presented
    step(view, pixels, stable=False, paint=False)
    step(view, pixels, paint=False)
    step(view, pixels, paint=False)
    assert c.state == "ready" and c.presented is visible
    click(view, (200, 100))
    assert c.draft is not None and not c._jobs
    QApplication.processEvents()
    view.canvas.repaint()
    assert c.presented.field_epoch == c.field_epoch
    click(view, (200, 100))
    assert wait_record(view).diameter_um == pytest.approx(100)


@pytest.mark.parametrize("release_while_moving", [False, True])
def test_started_drag_survives_motion_and_can_finish(workspace, release_while_moving):
    view, c = workspace, workspace.controller
    pixels = controlled_preview(view, gesture="drag")
    mouse(view, QEvent.Type.MouseButtonPress, (100, 100), held=True)
    mouse(view, QEvent.Type.MouseMove, (150, 100), held=True)
    assert c.draft is not None
    step(view, pixels, stable=False)
    mouse(view, QEvent.Type.MouseMove, (180, 100), held=True)
    assert c.draft is not None
    if release_while_moving:
        mouse(view, QEvent.Type.MouseButtonRelease, (200, 100))
        assert c.draft is not None and not c._jobs
    step(view, pixels)
    step(view, pixels)
    visible = c.presented
    if release_while_moving:
        click(view, (200, 100))
    else:
        mouse(view, QEvent.Type.MouseButtonRelease, (200, 100))
    assert c.draft is None and len(c._jobs) == 1
    assert c._jobs[0][0].token is visible
    assert wait_record(view).diameter_um == pytest.approx(100)
    assert not c._pointer_frames


@pytest.mark.parametrize("cancel", ["escape", "context", "unavailable"])
def test_preserved_start_still_respects_explicit_cancel_and_context_changes(workspace, cancel):
    view, c = workspace, workspace.controller
    pixels = controlled_preview(view)
    click(view, (100, 100))
    step(view, pixels, stable=False)
    assert c.draft is not None
    if cancel == "escape":
        QTest.keyClick(view.canvas, Qt.Key.Key_Escape)
    elif cancel == "context":
        c.end_field(cause=FieldEndCause.CONTEXT)
    else:
        c.unavailable("设备已断开")
    assert c.draft is None and c._draft_lease is None
    assert not c.motion_draft_is_current()


def test_repeated_motion_keeps_one_draft_and_releases_all_leases_on_close(workspace):
    view, c = workspace, workspace.controller
    pixels = controlled_preview(view)
    click(view, (100, 100))
    identity = c._classification_target["id"]
    for _ in range(8):
        step(view, pixels, stable=False)
        assert c.draft[1] == pytest.approx((100, 100))
        assert c._classification_target["id"] == identity
        assert len(c.budget._frames) <= 4
    c.close()
    assert c.budget.used_bytes == 0
    assert not c.budget._leases and not c.budget._frames


def test_real_detector_motion_preserves_first_point(workspace):
    view, c = workspace, workspace.controller
    original = _prepare_live_stripe(view, "manual")
    click(view, (100, 100))
    epoch = c.field_epoch
    pixels = np.roll(original.frame.pixels, 30, axis=1)
    for sequence in range(1, 9):
        frame = FrameHandle.snapshot(pixels, source_id=original.frame.source_id,
            generation=original.frame.generation, sequence=original.frame.sequence + sequence,
            live=True, metadata=original.frame.metadata)
        view._receive_frame(frame)
        _pump(view, .05)
        assert c.draft is not None, "Actual detector must not discard the user's first point"
    assert c.field_epoch > epoch and c.state == "ready"
    click(view, (210, 100))
    assert wait_record(view).diameter_um == pytest.approx(15, abs=.1)
