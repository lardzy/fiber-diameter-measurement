"""Real mouse routes for mixed candidate picking and manual measurement."""
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtCore import QEvent, QEventLoop, QPoint, QPointF, Qt, QTimer
from PySide6.QtGui import QMouseEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QTabWidget, QWidget

from fdm.live_experiment.frames import FrameBudget, FrameHandle
from test_live_measurement_candidate_canvas import add_candidates, canvas, pointer
from test_live_experiment_content_workspace import workspace


def mouse_move(view, position, *, held=True):
    event = QMouseEvent(QEvent.Type.MouseMove, QPointF(position), QPointF(view.mapToGlobal(position)),
        Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton if held else Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier)
    QApplication.sendEvent(view, event)


def manual_routes(view):
    observed = SimpleNamespace(starts=[], finishes=[], candidates=[], errors=[])

    def start(point, token):
        observed.starts.append((point, token))
        view.controller.draft = (token, point)

    def finish(point):
        observed.finishes.append(point)
        view.controller.draft = None

    view.manualGestureStarted.connect(start)
    view.manualGestureFinished.connect(finish)
    view.candidateClicked.connect(observed.candidates.append)
    view.gestureRejected.connect(observed.errors.append)
    view.gestureCancelled.connect(lambda: setattr(view.controller, "draft", None))
    return observed


def test_mixed_is_default_and_candidate_commits_only_on_release(canvas):
    observed = manual_routes(canvas)
    add_candidates(canvas)
    position = pointer(canvas, (215, 200))
    QTest.mousePress(canvas, Qt.MouseButton.LeftButton, pos=position)
    assert canvas.input_mode == "mixed"
    assert not observed.candidates and not observed.starts
    assert canvas.controller.budget.used_bytes == canvas.frame.nbytes
    QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton, pos=position)
    assert observed.candidates == ["batch:0"]
    assert canvas.controller.candidate_interactions == 1
    assert canvas.controller.budget.used_bytes == 0


def test_mixed_two_points_finish_manual_even_when_second_point_hits_candidate(canvas):
    observed = manual_routes(canvas)
    add_candidates(canvas)
    QTest.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=pointer(canvas, (150, 200)))
    assert len(observed.starts) == 1 and canvas.controller.draft is not None
    QTest.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=pointer(canvas, (215, 200)))
    assert observed.finishes == [(215., 200.)]
    assert not observed.candidates and canvas.controller.draft is None


@pytest.mark.parametrize("start", [(215, 200), (100, 100)])
def test_mixed_drag_on_candidate_or_empty_space_is_manual_and_keeps_press_frame(canvas, start):
    observed = manual_routes(canvas)
    canvas.set_manual_gesture("drag")
    add_candidates(canvas)
    original = canvas.controller.presented
    QTest.mousePress(canvas, Qt.MouseButton.LeftButton, pos=pointer(canvas, start))
    canvas.set_frame(FrameHandle.snapshot(np.full((576, 768, 3), 181, np.uint8),
                                         source_id="image", sequence=1, live=False))
    canvas.repaint()
    assert canvas.controller.presented.frame is not original.frame
    end = (start[0] + 45, start[1] + 30)
    mouse_move(canvas, pointer(canvas, end))
    QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton, pos=pointer(canvas, end))
    assert len(observed.starts) == 1
    assert observed.starts[0][0] == tuple(map(float, start))
    assert observed.starts[0][1] is original
    assert observed.finishes == [tuple(map(float, end))]
    assert not observed.candidates
    assert canvas.controller.budget.used_bytes == 0


def test_shift_candidate_click_is_a_manual_first_point(canvas):
    observed = manual_routes(canvas)
    add_candidates(canvas)
    QTest.mouseClick(canvas, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.ShiftModifier,
                     pos=pointer(canvas, (215, 200)))
    assert len(observed.starts) == 1 and not observed.candidates
    QTest.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=pointer(canvas, (250, 200)))
    assert observed.finishes == [(250., 200.)]


def test_pending_candidate_does_not_turn_repeated_clicks_into_manual_drafts(canvas):
    observed = manual_routes(canvas)
    add_candidates(canvas)
    canvas.controller.candidates[0]["state"] = "confirming"
    canvas.controller.candidatesChanged.emit(canvas.controller.candidates)
    QTest.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=pointer(canvas, (215, 200)))
    assert not observed.starts and not observed.finishes and not observed.candidates
    assert observed.errors == ["该候选正在核验，请等待计入。"]
    assert canvas.controller.budget.used_bytes == 0


def test_accepted_candidate_second_click_and_double_click_are_not_manual(canvas):
    observed = manual_routes(canvas)
    add_candidates(canvas)

    def accept(_):
        canvas.controller.candidates[0]["state"] = "accepted"
        canvas.controller.candidatesChanged.emit(canvas.controller.candidates)

    canvas.candidateClicked.connect(accept)
    position = pointer(canvas, (215, 200))
    QTest.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=position)
    QTest.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=position)
    QTest.mouseDClick(canvas, Qt.MouseButton.LeftButton, pos=position)
    QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton, pos=position)
    assert observed.candidates == ["batch:0"]
    assert not observed.starts and not observed.finishes
    assert canvas.controller.budget.used_bytes == 0


def test_candidate_ambiguity_is_resolved_on_release_without_manual_fallback(canvas):
    observed = manual_routes(canvas)
    ambiguous = []
    canvas.candidateAmbiguous.connect(ambiguous.append)
    add_candidates(canvas, [((200, 200), (230, 200)), ((200, 202), (230, 202))])
    position = pointer(canvas, (215, 201))
    QTest.mousePress(canvas, Qt.MouseButton.LeftButton, pos=position)
    assert not ambiguous
    QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton, pos=position)
    assert ambiguous == [["batch:0", "batch:1"]]
    assert not observed.starts and not observed.candidates


@pytest.mark.parametrize("transition", ["hidden", "unaligned", "refresh", "field", "source", "resize", "mode"])
def test_candidate_pointer_cancels_on_context_transition_without_becoming_manual(canvas, transition):
    observed = manual_routes(canvas)
    add_candidates(canvas)
    position = pointer(canvas, (215, 200))
    QTest.mousePress(canvas, Qt.MouseButton.LeftButton, pos=position)
    if transition == "hidden":
        canvas.set_candidate_presentation(False)
    elif transition == "unaligned":
        canvas.controller.candidate_overlay_ready = False
        canvas.controller.candidateStateChanged.emit()
    elif transition == "refresh":
        canvas.controller.candidates[0]["candidate_id"] = "next-batch:0"
        canvas.controller.candidatesChanged.emit(canvas.controller.candidates)
    elif transition == "field":
        canvas.controller.field_epoch += 1
        canvas.controller.fieldChanged.emit()
    elif transition == "source":
        canvas.set_frame(FrameHandle.snapshot(np.zeros((576, 768, 3), np.uint8),
                                             source_id="different", live=False))
    elif transition == "resize":
        canvas.resize(820, 620)
    else:
        canvas.set_input_mode("candidate")
    QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton, pos=position)
    assert not observed.starts and not observed.candidates
    assert canvas.controller.budget.used_bytes == 0


def test_actual_tab_navigation_cancels_mouse_down_and_releases_frame(canvas):
    observed = manual_routes(canvas)
    tabs = QTabWidget()
    tabs.addTab(canvas, "试验")
    tabs.addTab(QWidget(), "采集")
    tabs.show()
    add_candidates(canvas)
    position = pointer(canvas, (215, 200))
    QTest.mousePress(canvas, Qt.MouseButton.LeftButton, pos=position)
    QTest.mouseClick(tabs.tabBar(), Qt.MouseButton.LeftButton, pos=tabs.tabBar().tabRect(1).center())
    QTest.mouseClick(tabs.tabBar(), Qt.MouseButton.LeftButton, pos=tabs.tabBar().tabRect(0).center())
    QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton, pos=position)
    assert not observed.candidates and not observed.starts
    assert canvas.controller.budget.used_bytes == 0
    canvas.setParent(None)
    tabs.close()
    tabs.deleteLater()


def test_hidden_candidates_allow_normal_manual_input(canvas):
    observed = manual_routes(canvas)
    add_candidates(canvas)
    canvas.set_candidate_presentation(False)
    QTest.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=pointer(canvas, (215, 200)))
    assert len(observed.starts) == 1 and not observed.candidates


def test_hiding_candidates_clears_recent_click_interception(canvas):
    observed = manual_routes(canvas)
    add_candidates(canvas)
    position = pointer(canvas, (215, 200))
    QTest.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=position)
    assert observed.candidates == ["batch:0"]
    canvas.set_candidate_presentation(False)
    QTest.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=position)
    assert len(observed.starts) == 1


def test_manual_only_uses_the_same_explicit_gesture_route_without_legacy_duplicates(canvas):
    observed = manual_routes(canvas)
    add_candidates(canvas)
    canvas.set_input_mode("manual")
    canvas.set_manual_gesture("drag")
    pressed, released = [], []
    canvas.pointPressed.connect(pressed.append)
    canvas.pointReleased.connect(released.append)
    QTest.mousePress(canvas, Qt.MouseButton.LeftButton, pos=pointer(canvas, (215, 200)))
    assert not observed.starts
    mouse_move(canvas, pointer(canvas, (245, 230)))
    QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton, pos=pointer(canvas, (245, 230)))
    assert len(observed.starts) == 1 and observed.starts[0][0] == (215., 200.)
    assert observed.finishes == [(245., 230.)]
    assert not pressed and not released and not observed.candidates


def test_candidate_only_blank_and_drag_do_not_create_manual_measurement(canvas):
    observed = manual_routes(canvas)
    add_candidates(canvas)
    canvas.set_input_mode("candidate")
    QTest.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=pointer(canvas, (100, 100)))
    QTest.mousePress(canvas, Qt.MouseButton.LeftButton, pos=pointer(canvas, (215, 200)))
    QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton, pos=pointer(canvas, (260, 230)))
    assert not observed.starts and not observed.finishes and not observed.candidates


def test_failed_frame_admission_does_not_create_gesture_or_draft(canvas):
    observed = manual_routes(canvas)
    canvas.controller.budget = FrameBudget(limit_bytes=1)
    QTest.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=pointer(canvas, (100, 100)))
    assert observed.errors and not observed.starts and canvas._pointer_gesture is None
    assert canvas.controller.budget.used_bytes == 0


@pytest.mark.parametrize("change", ["source", "generation", "size", "cleared", "display"])
def test_pending_context_change_rejects_old_painted_input_before_retaining_frame(canvas, change, monkeypatch):
    observed = manual_routes(canvas)
    events = []
    monkeypatch.setattr(canvas.controller, "trace_input_event",
                        lambda event, **details: events.append({"event": event, **details}), raising=False)
    presented = canvas.controller.presented
    position = pointer(canvas, (100, 100))
    if change == "cleared":
        canvas.set_frame(None)
    elif change == "display":
        canvas.set_display_mode("percent", 125.)
    else:
        canvas.set_frame(FrameHandle.snapshot(
            np.full((580 if change == "size" else 576, 768, 3), 180, np.uint8),
            source_id="new-camera" if change == "source" else "image",
            generation=1 if change == "generation" else 0, sequence=1, live=False))
    # Queue a real context change without drawing it. Input cannot consume
    # the stale coordinate/source token even though the old image is visible.
    assert canvas.controller.presented is presented
    QTest.mousePress(canvas, Qt.MouseButton.LeftButton, pos=position)
    assert canvas._pointer_gesture is None
    assert canvas.controller.budget.used_bytes == 0
    QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton, pos=position)
    assert not observed.starts and not observed.finishes and not observed.candidates
    assert events[0]["event"] == "press_blocked"
    assert events[0]["event_type"] == "MouseButtonPress"
    assert events[0]["reason"] == ("no_presented_image" if change == "cleared"
                                    else "presentation_context_changed")


def _await_records(workspace, count):
    from time import monotonic
    deadline = monotonic() + 2
    loop = QEventLoop()
    while len(workspace.session.measurements) < count and monotonic() < deadline:
        QTimer.singleShot(10, loop.quit)
        loop.exec()
        workspace.controller.poll()
    assert len(workspace.session.measurements) == count


def test_workspace_mixed_candidate_then_two_points_preserves_both_workflows(workspace):
    from test_trial_assistance_ui import candidates, point
    workspace.input_mode.setCurrentIndex(workspace.input_mode.findData("mixed"))
    workspace.measure_mode.setCurrentIndex(workspace.measure_mode.findData("free"))
    candidates(workspace)
    workspace.material.setCurrentIndex(workspace.material.findData("wool"))
    QTest.mouseClick(workspace.canvas, Qt.MouseButton.LeftButton, pos=point(workspace, 90, 40))
    _await_records(workspace, 1)
    workspace.material.setCurrentIndex(workspace.material.findData("cashmere"))
    QTest.mouseClick(workspace.canvas, Qt.MouseButton.LeftButton, pos=point(workspace, 220, 180))
    assert workspace.controller.draft is not None
    QTest.mouseClick(workspace.canvas, Qt.MouseButton.LeftButton, pos=point(workspace, 240, 180))
    _await_records(workspace, 2)
    records = list(workspace.session.measurements.values())
    assert [record.material_id for record in records] == ["wool", "cashmere"]
    assert records[-1].diameter_um == pytest.approx(20)
    assert [item["state"] for item in workspace.controller.candidates] == ["available", "accepted", "available"]
    assert not workspace.controller._pointer_frames
    assert workspace.workflow_tabs.currentIndex() == 0


def test_workspace_mixed_drag_from_candidate_measures_manually_without_accepting_candidate(workspace):
    from test_trial_assistance_ui import candidates, point
    workspace.input_mode.setCurrentIndex(workspace.input_mode.findData("mixed"))
    workspace.gesture.setCurrentIndex(1)
    workspace.measure_mode.setCurrentIndex(workspace.measure_mode.findData("free"))
    candidates(workspace)
    source = workspace.controller.presented.frame
    QTest.mousePress(workspace.canvas, Qt.MouseButton.LeftButton, pos=point(workspace, 30, 40))
    mouse_move(workspace.canvas, point(workspace, 60, 80))
    QTest.mouseRelease(workspace.canvas, Qt.MouseButton.LeftButton, pos=point(workspace, 60, 80))
    _await_records(workspace, 1)
    record = next(iter(workspace.session.measurements.values()))
    assert record.diameter_um == pytest.approx(50)
    assert record.endpoints == ((30., 40.), (60., 80.))
    assert record.frame_id == source.frame_id
    assert all(item["state"] == "available" for item in workspace.controller.candidates)
    assert not workspace.controller._pointer_frames


@pytest.mark.parametrize("start", [(30, 40), (220, 180)])
def test_workspace_refresh_after_drag_starts_preserves_the_manual_intent(workspace, start):
    from test_trial_assistance_ui import candidates, point
    workspace.input_mode.setCurrentIndex(workspace.input_mode.findData("mixed"))
    workspace.gesture.setCurrentIndex(1)
    workspace.measure_mode.setCurrentIndex(workspace.measure_mode.findData("free"))
    candidates(workspace)
    end = (start[0] + 30, start[1] + 20)
    QTest.mousePress(workspace.canvas, Qt.MouseButton.LeftButton, pos=point(workspace, *start))
    mouse_move(workspace.canvas, point(workspace, *end))
    QApplication.processEvents()
    draft = workspace.controller.draft
    assert draft is not None
    workspace.controller.discard_candidates()
    assert workspace.controller.draft is draft
    QTest.mouseRelease(workspace.canvas, Qt.MouseButton.LeftButton, pos=point(workspace, *end))
    _await_records(workspace, 1)
    record = next(iter(workspace.session.measurements.values()))
    assert record.endpoints == (tuple(map(float, start)), tuple(map(float, end)))
    assert not workspace.controller._jobs
    assert not workspace.controller._pointer_frames


def test_same_batch_state_change_does_not_cancel_a_manual_drag(canvas):
    observed = manual_routes(canvas)
    canvas.set_manual_gesture("drag")
    add_candidates(canvas)
    QTest.mousePress(canvas, Qt.MouseButton.LeftButton, pos=pointer(canvas, (215, 200)))
    mouse_move(canvas, pointer(canvas, (260, 230)))
    QApplication.processEvents()
    assert canvas.controller.draft is not None
    # A completion callback changes an unrelated candidate in this batch. A
    # new explicit table-submit action has its own controller cancellation rule.
    for state in ("confirming", "accepted"):
        canvas.controller.candidates[1]["state"] = state
        canvas.controller.candidatesChanged.emit(canvas.controller.candidates)
        assert canvas.controller.draft is not None
    QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton, pos=pointer(canvas, (260, 230)))
    assert observed.finishes == [(260., 230.)]
    assert not observed.candidates and canvas.controller.budget.used_bytes == 0


def test_offset_release_without_held_move_is_only_first_point(canvas):
    observed = manual_routes(canvas)
    QTest.mousePress(canvas, Qt.MouseButton.LeftButton, pos=pointer(canvas, (100, 100)))
    QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton, pos=pointer(canvas, (145, 130)))
    assert len(observed.starts) == 1
    assert observed.starts[0][0] == (100., 100.)
    assert not observed.finishes
    assert canvas.controller.draft[1] == (100., 100.)


def test_first_point_and_one_thousand_unpressed_moves_never_submit(canvas):
    observed = manual_routes(canvas)
    QTest.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=pointer(canvas, (100, 100)))
    for index in range(1000):
        mouse_move(canvas, pointer(canvas, (120 + index % 40, 120)), held=False)
    assert len(observed.starts) == 1 and not observed.finishes
    assert canvas.controller.draft[1] == (100., 100.)


def test_unpressed_move_cancels_stale_press_instead_of_starting_a_drag(canvas):
    observed = manual_routes(canvas)
    canvas.set_manual_gesture("drag")
    QTest.mousePress(canvas, Qt.MouseButton.LeftButton, pos=pointer(canvas, (100, 100)))
    mouse_move(canvas, pointer(canvas, (145, 130)), held=False)
    QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton, pos=pointer(canvas, (145, 130)))
    assert not observed.starts and not observed.finishes
    assert canvas._pointer_gesture is None and canvas.controller.budget.used_bytes == 0


@pytest.mark.parametrize("event_type", [QEvent.Type.UngrabMouse, QEvent.Type.FocusOut,
                                       QEvent.Type.WindowDeactivate])
def test_lost_press_preserves_established_first_point_and_ignores_late_release(canvas, event_type):
    observed = manual_routes(canvas)
    QTest.mousePress(canvas, Qt.MouseButton.LeftButton, pos=pointer(canvas, (100, 100)))
    draft = canvas.controller.draft
    assert draft is not None and draft[1] == (100., 100.)
    QApplication.sendEvent(canvas, QEvent(event_type))
    QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton, pos=pointer(canvas, (150, 130)))
    assert canvas.controller.draft is draft
    assert not observed.finishes and canvas.controller.budget.used_bytes == 0


@pytest.mark.parametrize("event_type", [QEvent.Type.UngrabMouse, QEvent.Type.FocusOut,
                                       QEvent.Type.WindowDeactivate])
def test_interruption_after_second_press_cannot_repeat_submission_on_release(canvas, event_type):
    observed = manual_routes(canvas)
    QTest.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=pointer(canvas, (100, 100)))
    QTest.mousePress(canvas, Qt.MouseButton.LeftButton, pos=pointer(canvas, (140, 120)))
    assert observed.finishes == [(140., 120.)]
    assert canvas.controller.draft is None
    QApplication.sendEvent(canvas, QEvent(event_type))
    QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton, pos=pointer(canvas, (150, 130)))
    assert len(observed.starts) == 1 and observed.finishes == [(140., 120.)]
    assert canvas.controller.draft is None and canvas.controller.budget.used_bytes == 0


def test_repeated_same_point_and_double_click_do_not_submit_zero_length(canvas):
    observed = manual_routes(canvas)
    position = pointer(canvas, (100, 100))
    QTest.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=position)
    QTest.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=position)
    QTest.mouseDClick(canvas, Qt.MouseButton.LeftButton, pos=position)
    QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton, pos=position)
    assert len(observed.starts) == 1 and not observed.finishes
    assert canvas.controller.draft[1] == (100., 100.)


@pytest.mark.parametrize("automatic", [False, True])
@pytest.mark.parametrize("mode", ["mixed", "manual"])
def test_late_failed_task_cannot_cover_new_first_point_in_workspace(workspace, monkeypatch, automatic, mode):
    from concurrent.futures import Future
    from test_trial_assistance_ui import point
    controller = workspace.controller
    controller._timer.stop()
    workspace.input_mode.setCurrentIndex(workspace.input_mode.findData(mode))
    workspace.gesture.setCurrentIndex(0)
    controller.automatic_enabled = automatic
    monkeypatch.setattr(controller._manual_pool, "submit", lambda *_: Future())
    QTest.mouseClick(workspace.canvas, Qt.MouseButton.LeftButton, pos=point(workspace, 30, 30))
    QTest.mouseClick(workspace.canvas, Qt.MouseButton.LeftButton, pos=point(workspace, 60, 30))
    intent, future = controller._jobs[-1]
    QTest.mouseClick(workspace.canvas, Qt.MouseButton.LeftButton, pos=point(workspace, 120, 100))
    draft = controller.draft
    assert draft is not None
    before = workspace.measurement_outcome.text()
    future.set_result({"status": "rejected", "reason": "endpoints_not_background"})
    controller.poll()
    assert controller.draft is draft
    assert not workspace.measurement_feedback.isVisible()
    assert workspace.measurement_outcome.text() == before
    assert workspace.statusBar().currentMessage().startswith("测量 #1 未通过")
    trace = controller.performance_frame_context()["input_trace"]
    submits = [item for item in trace["events"] if item["event"] == "task_submitted"]
    assert len(submits) == 1 and submits[0]["task_id"] == intent.id


def test_input_trace_is_bounded_and_links_pointer_to_task(workspace):
    from test_trial_assistance_ui import point
    workspace.input_mode.setCurrentIndex(workspace.input_mode.findData("mixed"))
    workspace.measure_mode.setCurrentIndex(workspace.measure_mode.findData("free"))
    for index in range(600):
        mouse_move(workspace.canvas, point(workspace, 100 + index % 30, 100), held=False)
    QTest.mouseClick(workspace.canvas, Qt.MouseButton.LeftButton, pos=point(workspace, 30, 30))
    QTest.mouseClick(workspace.canvas, Qt.MouseButton.LeftButton, pos=point(workspace, 60, 30))
    _await_records(workspace, 1)
    trace = workspace.controller.performance_frame_context()["input_trace"]
    assert trace["capacity"] == len(trace["events"]) == 512
    assert trace["total_events"] > 600
    finish = next(item for item in trace["events"] if item["event"] == "manual_finish")
    task = next(item for item in trace["events"] if item["event"] == "task_submitted")
    assert finish["submit_reason"] == "second_point"
    assert finish["pointer_id"] and finish["interaction_id"] == task["task_id"]
