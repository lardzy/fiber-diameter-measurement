"""Real mouse routes for mixed candidate picking and manual measurement."""
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtCore import QEventLoop, QPoint, Qt, QTimer
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QTabWidget, QWidget

from fdm.live_experiment.frames import FrameBudget, FrameHandle
from test_live_measurement_candidate_canvas import add_candidates, canvas, pointer
from test_live_experiment_content_workspace import workspace


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
    add_candidates(canvas)
    original = canvas.controller.presented
    QTest.mousePress(canvas, Qt.MouseButton.LeftButton, pos=pointer(canvas, start))
    canvas.set_frame(FrameHandle.snapshot(np.full((576, 768, 3), 181, np.uint8),
                                         source_id="image", sequence=1, live=False))
    canvas.repaint()
    assert canvas.controller.presented.frame is not original.frame
    end = (start[0] + 45, start[1] + 30)
    QTest.mouseMove(canvas, pointer(canvas, end))
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


def test_manual_only_keeps_legacy_gesture_signals(canvas):
    observed = manual_routes(canvas)
    add_candidates(canvas)
    canvas.set_input_mode("manual")
    canvas.set_manual_gesture("drag")
    pressed, released = [], []
    canvas.pointPressed.connect(pressed.append)
    canvas.pointReleased.connect(released.append)
    QTest.mousePress(canvas, Qt.MouseButton.LeftButton, pos=pointer(canvas, (215, 200)))
    QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton, pos=pointer(canvas, (245, 230)))
    assert pressed == [(215., 200.)] and released == [(245., 230.)]
    assert not observed.starts and not observed.candidates


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
    workspace.measure_mode.setCurrentIndex(workspace.measure_mode.findData("free"))
    candidates(workspace)
    source = workspace.controller.presented.frame
    QTest.mousePress(workspace.canvas, Qt.MouseButton.LeftButton, pos=point(workspace, 30, 40))
    QTest.mouseRelease(workspace.canvas, Qt.MouseButton.LeftButton, pos=point(workspace, 60, 80))
    _await_records(workspace, 1)
    record = next(iter(workspace.session.measurements.values()))
    assert record.diameter_um == pytest.approx(50)
    assert record.endpoints == ((30., 40.), (60., 80.))
    assert record.frame_id == source.frame_id
    assert all(item["state"] == "available" for item in workspace.controller.candidates)
    assert not workspace.controller._pointer_frames


@pytest.mark.parametrize("start", [(30, 40), (220, 180)])
def test_workspace_refresh_after_drag_starts_cancels_the_pointer_intent(workspace, start):
    from test_trial_assistance_ui import candidates, point
    workspace.input_mode.setCurrentIndex(workspace.input_mode.findData("mixed"))
    workspace.measure_mode.setCurrentIndex(workspace.measure_mode.findData("free"))
    candidates(workspace)
    end = (start[0] + 30, start[1] + 20)
    QTest.mousePress(workspace.canvas, Qt.MouseButton.LeftButton, pos=point(workspace, *start))
    QTest.mouseMove(workspace.canvas, point(workspace, *end))
    QApplication.processEvents()
    assert workspace.controller.draft is not None
    workspace.controller.discard_candidates()
    assert workspace.controller.draft is None
    QTest.mouseRelease(workspace.canvas, Qt.MouseButton.LeftButton, pos=point(workspace, *end))
    workspace.controller.poll()
    assert not workspace.session.measurements
    assert not workspace.controller._jobs
    assert not workspace.controller._pointer_frames


def test_same_batch_state_change_does_not_cancel_a_manual_drag(canvas):
    observed = manual_routes(canvas)
    add_candidates(canvas)
    QTest.mousePress(canvas, Qt.MouseButton.LeftButton, pos=pointer(canvas, (215, 200)))
    QTest.mouseMove(canvas, pointer(canvas, (260, 230)))
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
