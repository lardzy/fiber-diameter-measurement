"""Explicit two-point/drag contracts through the real experiment workspace."""
from concurrent.futures import Future
import time

import pytest
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication

from fdm.live_experiment.frames import FrameBudgetExceeded
from test_live_experiment_content_workspace import workspace
from test_live_mixed_presented_frame import _arrive_without_paint, _prepare_live_stripe
from test_trial_assistance_ui import candidates, point


def mouse(view, event_type, xy, *, held=False, shift=False):
    position = point(view, *xy)
    event = QMouseEvent(
        event_type, QPointF(position), QPointF(view.canvas.mapToGlobal(position)),
        Qt.MouseButton.NoButton if event_type == QEvent.Type.MouseMove else Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton if held else Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.ShiftModifier if shift else Qt.KeyboardModifier.NoModifier,
    )
    QApplication.sendEvent(view.canvas, event)


def click(view, xy, *, shift=False):
    mouse(view, QEvent.Type.MouseButtonPress, xy, held=True, shift=shift)
    mouse(view, QEvent.Type.MouseButtonRelease, xy, shift=shift)


def drag(view, start, end, *, shift=False):
    mouse(view, QEvent.Type.MouseButtonPress, start, held=True, shift=shift)
    mouse(view, QEvent.Type.MouseMove, end, held=True, shift=shift)
    mouse(view, QEvent.Type.MouseButtonRelease, end, shift=shift)


def prepare(view, mode, gesture):
    view.controller._timer.stop()
    view.input_mode.setCurrentIndex(view.input_mode.findData(mode))
    view.gesture.setCurrentIndex(int(gesture == "drag"))
    view.measure_mode.setCurrentIndex(view.measure_mode.findData("free"))
    QApplication.processEvents()
    view.canvas.repaint()
    assert view.canvas.input_mode == mode
    assert view.canvas.manual_gesture == gesture


def wait_record(view, count=1):
    deadline = time.monotonic() + 3
    while len(view.session.measurements) < count and time.monotonic() < deadline:
        QApplication.processEvents()
        view.controller.poll()
        time.sleep(.001)
    assert len(view.session.measurements) == count
    assert not view.controller._jobs
    return list(view.session.measurements.values())[-1]


@pytest.mark.parametrize("mode", ["mixed", "manual"])
@pytest.mark.parametrize("automatic", [False, True])
def test_two_point_first_click_and_unpressed_movement_never_submit(workspace, mode, automatic):
    prepare(workspace, mode, "click")
    workspace.auto_analysis.setChecked(automatic)
    click(workspace, (100, 100))
    draft = workspace.controller.draft
    assert draft is not None and draft[1] == (100., 100.)
    for index in range(1000):
        mouse(workspace, QEvent.Type.MouseMove, (120 + index % 30, 140))
    assert workspace.controller.draft is draft
    assert not workspace.controller._jobs and not workspace.session.measurements
    click(workspace, (180, 100))
    intent = workspace.controller._jobs[0][0]
    assert intent.start == (100., 100.) and intent.end == (180., 100.)
    assert wait_record(workspace).diameter_um == pytest.approx(80)
    click(workspace, (80, 130))
    assert workspace.controller.draft[1] == (80., 130.)
    assert not workspace.controller._jobs


@pytest.mark.parametrize("mode", ["mixed", "manual"])
def test_two_point_mode_places_only_on_press_never_on_move_or_release(workspace, mode):
    prepare(workspace, mode, "click")
    controller = workspace.controller
    mouse(workspace, QEvent.Type.MouseButtonPress, (100, 100), held=True)
    assert controller.draft is not None
    draft = controller.draft
    assert draft[1] == (100., 100.)
    for index in range(30):
        mouse(workspace, QEvent.Type.MouseMove, (140 + index, 120), held=True)
    mouse(workspace, QEvent.Type.MouseButtonRelease, (180, 120))
    assert controller.draft is draft
    assert not controller._jobs and not workspace.session.measurements
    mouse(workspace, QEvent.Type.MouseButtonPress, (180, 100), held=True)
    assert controller.draft is None
    assert len(controller._jobs) == 1
    intent = controller._jobs[0][0]
    assert intent.start == (100., 100.) and intent.end == (180., 100.)
    mouse(workspace, QEvent.Type.MouseMove, (220, 130), held=True)
    mouse(workspace, QEvent.Type.MouseButtonRelease, (220, 130))
    assert len(controller._jobs) == 1
    assert controller.draft is None
    record = wait_record(workspace)
    assert record.diameter_um == pytest.approx(80)
    trace = controller.performance_frame_context()["input_trace"]["events"]
    finish = [item for item in trace if item["event"] == "manual_finish"]
    assert len(finish) == 1 and finish[0]["submit_reason"] == "second_point"


@pytest.mark.parametrize("mode", ["mixed", "manual"])
def test_same_point_second_click_keeps_first_point_without_submission(workspace, mode):
    prepare(workspace, mode, "click")
    click(workspace, (100, 100))
    draft = workspace.controller.draft
    click(workspace, (100, 100))
    assert workspace.controller.draft is draft
    assert not workspace.controller._jobs
    click(workspace, (180, 100))
    assert wait_record(workspace).diameter_um == pytest.approx(80)


@pytest.mark.parametrize("mode", ["mixed", "manual"])
@pytest.mark.parametrize("error", [ValueError("测量条件已变化"), FrameBudgetExceeded("测量帧预算已满")])
def test_rejected_second_point_cannot_be_completed_by_next_intended_first_point(workspace, monkeypatch, mode, error):
    prepare(workspace, mode, "click")
    controller = workspace.controller
    click(workspace, (100, 100))
    assert controller.draft is not None
    finish = controller.finish

    def reject(*args, **kwargs):
        raise error

    monkeypatch.setattr(controller, "finish", reject)
    click(workspace, (180, 100))
    assert controller.draft is None
    assert not controller._jobs and not workspace.session.measurements
    monkeypatch.setattr(controller, "finish", finish)
    click(workspace, (80, 130))
    assert controller.draft is not None
    assert controller.draft[1] == (80., 130.)
    assert not controller._jobs and not workspace.session.measurements


@pytest.mark.parametrize("mode", ["mixed", "manual"])
@pytest.mark.parametrize("rejection", ["frame_admission", "outside_image"])
def test_rejected_second_press_does_not_keep_an_invisible_old_start(workspace, monkeypatch, mode, rejection):
    prepare(workspace, mode, "click")
    controller = workspace.controller
    click(workspace, (100, 100))
    assert controller.draft is not None
    with monkeypatch.context() as context:
        if rejection == "frame_admission":
            def reject(_):
                raise FrameBudgetExceeded("按下帧预算已满")
            context.setattr(controller, "retain_pointer_frame", reject)
            click(workspace, (180, 100))
        else:
            click(workspace, (-5, 100))
    assert controller.draft is None
    assert workspace.canvas._pointer_gesture is None
    assert not controller._pointer_frames and not controller._jobs
    click(workspace, (80, 130))
    assert controller.draft[1] == (80., 130.)
    assert not controller._jobs and not workspace.session.measurements


@pytest.mark.parametrize("mode", ["mixed", "manual"])
def test_drag_mode_short_clicks_never_leave_a_first_point(workspace, mode):
    prepare(workspace, mode, "drag")
    for xy in ((100, 100), (180, 100), (80, 130)):
        click(workspace, xy)
        assert workspace.controller.draft is None
        assert workspace.canvas._pointer_gesture is None
        assert not workspace.controller._jobs and not workspace.session.measurements
    drag(workspace, (100, 100), (180, 100))
    intent = workspace.controller._jobs[0][0]
    assert intent.start == (100., 100.) and intent.end == (180., 100.)
    assert wait_record(workspace).diameter_um == pytest.approx(80)
    assert workspace.controller.draft is None


@pytest.mark.parametrize("mode", ["mixed", "manual"])
def test_drag_requires_held_move_not_a_different_release_coordinate(workspace, mode):
    prepare(workspace, mode, "drag")
    mouse(workspace, QEvent.Type.MouseButtonPress, (100, 100), held=True)
    mouse(workspace, QEvent.Type.MouseButtonRelease, (180, 100))
    assert workspace.controller.draft is None
    assert not workspace.controller._jobs and not workspace.session.measurements


@pytest.mark.parametrize("gesture", ["click", "drag"])
def test_mixed_candidate_short_click_is_still_a_candidate(workspace, gesture):
    prepare(workspace, "mixed", gesture)
    ids = candidates(workspace)
    click(workspace, (90, 40))
    assert wait_record(workspace).diameter_um == pytest.approx(20)
    assert workspace.controller.draft is None
    assert [item["candidate_id"] for item in workspace.controller.candidates] == ids
    assert [item["state"] for item in workspace.controller.candidates] == ["available", "accepted", "available"]
    click(workspace, (90, 40))
    assert len(workspace.session.measurements) == 1
    assert workspace.controller.draft is None


def test_mixed_drag_can_start_over_candidate_without_accepting_it(workspace):
    prepare(workspace, "mixed", "drag")
    candidates(workspace)
    drag(workspace, (90, 40), (140, 70))
    intent = workspace.controller._jobs[0][0]
    assert intent.start == (90., 40.) and intent.end == (140., 70.)
    record = wait_record(workspace)
    assert record.diameter_um == pytest.approx((50**2 + 30**2)**.5)
    assert all(item["state"] == "available" for item in workspace.controller.candidates)


def test_two_point_candidate_drag_is_cancelled_without_becoming_manual(workspace):
    prepare(workspace, "mixed", "click")
    candidates(workspace)
    drag(workspace, (90, 40), (140, 70))
    assert workspace.controller.draft is None
    assert not workspace.controller._jobs and not workspace.session.measurements
    assert all(item["state"] == "available" for item in workspace.controller.candidates)
    # The next ordinary click must establish its own first point, not finish
    # a hidden manual draft created while moving away from the candidate.
    click(workspace, (180, 120))
    assert workspace.controller.draft[1] == (180., 120.)
    assert not workspace.controller._jobs


def test_two_point_existing_first_point_wins_over_candidate(workspace):
    prepare(workspace, "mixed", "click")
    candidates(workspace)
    # Stay clear of both the candidate line and its screen-space label.
    click(workspace, (90, 100))
    assert workspace.controller.draft[1] == (90., 100.)
    click(workspace, (90, 40))
    intent = workspace.controller._jobs[0][0]
    assert intent.start == (90., 100.) and intent.end == (90., 40.)
    assert wait_record(workspace).diameter_um == pytest.approx(60)
    assert all(item["state"] == "available" for item in workspace.controller.candidates)


@pytest.mark.parametrize("gesture", ["click", "drag"])
def test_shift_forces_manual_without_changing_selected_gesture(workspace, gesture):
    prepare(workspace, "mixed", gesture)
    candidates(workspace)
    click(workspace, (90, 40), shift=True)
    assert not workspace.controller._jobs and not workspace.session.measurements
    if gesture == "click":
        assert workspace.controller.draft[1] == (90., 40.)
        click(workspace, (140, 40))
    else:
        assert workspace.controller.draft is None
        drag(workspace, (90, 40), (140, 40), shift=True)
    assert wait_record(workspace).diameter_um == pytest.approx(50)
    assert all(item["state"] == "available" for item in workspace.controller.candidates)


@pytest.mark.parametrize("mode", ["mixed", "manual"])
def test_changing_manual_gesture_cancels_old_two_point_draft(workspace, mode):
    prepare(workspace, mode, "click")
    click(workspace, (100, 100))
    assert workspace.controller.draft is not None
    workspace.gesture.setCurrentIndex(1)
    assert workspace.canvas.manual_gesture == "drag"
    assert workspace.controller.draft is None
    click(workspace, (180, 100))
    assert workspace.controller.draft is None
    assert not workspace.controller._jobs
    workspace.gesture.setCurrentIndex(0)
    click(workspace, (80, 130))
    assert workspace.controller.draft[1] == (80., 130.)
    assert not workspace.controller._jobs


@pytest.mark.parametrize("mode", ["mixed", "manual"])
@pytest.mark.parametrize("gesture", ["click", "drag"])
def test_explicit_gestures_keep_presented_frame_during_preview_arrival(workspace, mode, gesture):
    _prepare_live_stripe(workspace, mode)
    workspace.gesture.setCurrentIndex(int(gesture == "drag"))
    QApplication.processEvents()
    workspace.canvas.repaint()
    visible = _arrive_without_paint(workspace)
    if gesture == "click":
        click(workspace, (100, 100))
        assert workspace.controller.draft[0] is visible
        QApplication.processEvents()
        workspace.canvas.repaint()
        _arrive_without_paint(workspace)
        click(workspace, (180, 100))
    else:
        mouse(workspace, QEvent.Type.MouseButtonPress, (100, 100), held=True)
        QApplication.processEvents()
        workspace.canvas.repaint()
        mouse(workspace, QEvent.Type.MouseMove, (180, 100), held=True)
        mouse(workspace, QEvent.Type.MouseButtonRelease, (180, 100))
    intent = workspace.controller._jobs[0][0]
    assert intent.token is visible
    assert intent.start == (100., 100.) and intent.end == (180., 100.)
    record = wait_record(workspace)
    assert record.frame_id == visible.frame.frame_id
    assert record.diameter_um == pytest.approx(15, abs=.1)


@pytest.mark.parametrize("mode", ["mixed", "manual"])
@pytest.mark.parametrize("automatic", [False, True])
def test_old_failure_between_new_press_and_release_cannot_cover_new_first_point(workspace, monkeypatch, mode, automatic):
    prepare(workspace, mode, "click")
    workspace.auto_analysis.setChecked(automatic)
    controller = workspace.controller
    monkeypatch.setattr(controller._manual_pool, "submit", lambda *_: Future())
    click(workspace, (30, 30))
    click(workspace, (60, 30))
    intent, future = controller._jobs[-1]
    mouse(workspace, QEvent.Type.MouseButtonPress, (120, 100), held=True)
    before = workspace.measurement_outcome.text()
    future.set_result({"status": "rejected", "reason": "no_complete_crossing"})
    controller.poll()
    assert not workspace.measurement_feedback.isVisible()
    assert workspace.measurement_outcome.text() == before
    mouse(workspace, QEvent.Type.MouseButtonRelease, (120, 100))
    assert controller.draft is not None
    assert controller.draft[1] == (120., 100.)
    assert not workspace.measurement_feedback.isVisible()
    assert not controller._jobs and not workspace.session.measurements
    feedback = [entry for entry in controller.performance_frame_context()["input_trace"]["events"]
                if entry["event"] == "feedback"]
    assert feedback[-1]["task_id"] == intent.id
    assert feedback[-1]["current_interaction"] is False


@pytest.mark.parametrize("mode", ["mixed", "manual"])
def test_new_drag_start_dismisses_failure_that_arrived_before_held_move(workspace, monkeypatch, mode):
    prepare(workspace, mode, "drag")
    controller = workspace.controller
    monkeypatch.setattr(controller._manual_pool, "submit", lambda *_: Future())
    drag(workspace, (30, 100), (80, 100))
    old_intent, old_future = controller._jobs[-1]
    mouse(workspace, QEvent.Type.MouseButtonPress, (120, 100), held=True)
    old_future.set_result({"status": "rejected", "reason": "no_complete_crossing"})
    controller.poll()
    mouse(workspace, QEvent.Type.MouseMove, (180, 100), held=True)
    assert controller.draft is not None
    assert controller.draft[1] == (120., 100.)
    assert not workspace.measurement_feedback.isVisible()
    mouse(workspace, QEvent.Type.MouseButtonRelease, (180, 100))
    assert controller.draft is None
    assert len(controller._jobs) == 1
    new_intent = controller._jobs[0][0]
    assert new_intent.id != old_intent.id
    assert new_intent.start == (120., 100.) and new_intent.end == (180., 100.)
