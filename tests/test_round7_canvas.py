"""Paint-owned candidate presses, independent manual drafts and retained marks."""
from types import SimpleNamespace
import time

import pytest
from PySide6.QtCore import QPointF, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from test_live_measurement_candidate_canvas import add_candidates, canvas, pointer
from test_live_measurement_mixed_canvas import manual_routes, mouse_move
from test_live_experiment_content_workspace import workspace
from fdm.ui.live_measurement_annotations import segment_intersects_rect


def test_paused_painted_candidate_cannot_silently_become_manual_first_point(canvas):
    observed = manual_routes(canvas)
    add_candidates(canvas)
    target = pointer(canvas, (215, 200))
    assert canvas._painted_candidates
    canvas.controller.candidate_overlay_ready = False
    canvas.controller.candidateStateChanged.emit()
    # Do not process a repaint: the user can still see the old candidate.
    QTest.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=target)
    assert not observed.starts and not observed.finishes and not observed.candidates
    assert observed.errors and canvas.controller.draft is None
    canvas.controller.candidate_overlay_ready = True
    canvas.controller.candidateStateChanged.emit()
    QTest.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=pointer(canvas, (150, 200)))
    assert len(observed.starts) == 1 and not observed.finishes
    assert canvas.controller.draft[1] == (150., 200.)


@pytest.mark.parametrize("change", ["refresh", "accepted", "excluded", "removed"])
def test_stale_painted_candidate_blocks_before_repaint_but_gone_geometry_is_blank(canvas, change):
    observed = manual_routes(canvas)
    add_candidates(canvas)
    target = pointer(canvas, (215, 200))
    if change == "refresh":
        canvas.controller.candidates[0]["candidate_id"] = "new:0"
    elif change == "removed":
        canvas.controller.candidates = []
    else:
        canvas.controller.candidates[0]["state"] = change
    canvas.controller.candidatesChanged.emit(canvas.controller.candidates)
    QTest.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=target)
    assert canvas.controller.draft is None and not observed.candidates
    assert observed.errors
    if change != "refresh":
        canvas.repaint()
        # Once removed, a different blank location is available immediately.
        QTest.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=pointer(canvas, (150, 200)))
        assert len(observed.starts) == 1


@pytest.mark.parametrize("phase", ["first_press", "first_point", "second_press", "drag"])
def test_candidate_batch_refresh_cannot_cancel_or_complete_manual_input(canvas, phase):
    observed = manual_routes(canvas)
    add_candidates(canvas)
    start, end = pointer(canvas, (100, 100)), pointer(canvas, (150, 130))
    if phase in {"first_point", "second_press"}:
        QTest.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=start)
        if phase == "second_press":
            QTest.mousePress(canvas, Qt.MouseButton.LeftButton, pos=end)
    else:
        QTest.mousePress(canvas, Qt.MouseButton.LeftButton, pos=start)
        if phase == "drag":
            mouse_move(canvas, end)
    draft = canvas.controller.draft
    canvas.controller.candidates[0]["candidate_id"] = "replacement:0"
    canvas.controller.candidatesChanged.emit(canvas.controller.candidates)
    assert canvas.controller.draft is draft
    assert not observed.finishes
    if phase == "first_point":
        for _ in range(20):
            mouse_move(canvas, end, held=False)
        assert canvas.controller.draft is draft
        QTest.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=end)
    else:
        QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton, pos=end if phase != "first_press" else start)
    if phase == "first_press":
        assert len(observed.starts) == 1 and not observed.finishes
    else:
        assert observed.finishes == [(150., 130.)]


def test_drag_promoted_on_candidate_survives_batch_refresh(canvas):
    observed = manual_routes(canvas)
    add_candidates(canvas)
    QTest.mousePress(canvas, Qt.MouseButton.LeftButton, pos=pointer(canvas, (215, 200)))
    mouse_move(canvas, pointer(canvas, (260, 230)))
    draft = canvas.controller.draft
    canvas.controller.candidates = []
    canvas.controller.candidatesChanged.emit([])
    assert canvas.controller.draft is draft
    QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton, pos=pointer(canvas, (260, 230)))
    assert observed.finishes == [(260., 230.)] and not observed.candidates


def test_retained_marks_paint_under_current_geometry_and_never_intercept_input(canvas):
    observed = manual_routes(canvas)
    mark = dict(record_id="old-picked", start=(200., 200.), end=(230., 200.), number=3, color="#d33a28")
    canvas.controller.selection_retention = SimpleNamespace(visible_marks=lambda: [mark])
    canvas.repaint()
    items, _ = canvas._overlay_items()
    assert items[0]["kind"] == "retained" and items[0]["label"] == "#3"
    assert not canvas._painted_candidates
    target = pointer(canvas, (215, 200))
    assert not canvas.candidate_hits(QPointF(target))
    QTest.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=target)
    assert len(observed.starts) == 1 and not observed.candidates
    mouse_move(canvas, pointer(canvas, (250, 230)), held=False)
    canvas.repaint()
    items, protected = canvas._overlay_items()
    placement = canvas._arrange_labels(items, protected).get("retained:old-picked")
    assert placement is not None
    assert not segment_intersects_rect(*protected[0], placement.rect.adjusted(-12., -12., 12., 12.))
    assert canvas._hovered_record_id is None


def test_retained_marks_do_not_follow_ordinary_candidate_visibility(canvas):
    mark = dict(record_id="picked", start=(200., 200.), end=(230., 200.), number=1, color="#d33a28")
    canvas.controller.selection_retention = SimpleNamespace(visible_marks=lambda: [mark])
    add_candidates(canvas)
    canvas.set_candidate_presentation(False)
    assert [item["kind"] for item in canvas._overlay_items()[0]] == ["retained"]
    canvas.controller.candidates = []
    canvas.controller.candidatesChanged.emit([])
    assert [item["kind"] for item in canvas._overlay_items()[0]] == ["retained"]


def test_retention_preference_roundtrip_defaults_and_legacy_mode(workspace):
    before = workspace.session.to_dict()
    assert workspace.selection_retention_seconds.value() == 1.
    workspace.selection_retention_seconds.setValue(2.3)
    saved = workspace._assistance_preferences()
    assert saved["version"] == 3 and saved["retention_seconds"] == 2.3
    assert workspace.controller.selection_retention.duration == 2.3
    workspace._apply_assistance_preferences({"version": 2, "input_mode": "manual", "manual_algorithm": "free"})
    assert workspace.input_mode.currentData() == "manual"
    assert workspace.measure_mode.currentData() == "free"
    assert workspace.selection_retention_seconds.value() == 1.
    workspace._apply_assistance_preferences(saved)
    assert workspace.controller.selection_retention.duration == 2.3
    assert workspace.session.to_dict() == before


@pytest.mark.parametrize("value,expected", [(None, 1.), ("bad", 1.), (float("nan"), 1.),
                                             (-5, 0.), (20, 10.), (0, 0.)])
def test_retention_preference_validation(workspace, value, expected):
    workspace._apply_assistance_preferences({"retention_seconds": value})
    assert workspace.selection_retention_seconds.value() == expected
    assert workspace.controller.selection_retention.duration == expected


def test_blocked_candidate_double_click_cannot_start_draft_after_repaint(canvas):
    observed = manual_routes(canvas)
    add_candidates(canvas)
    target = pointer(canvas, (215, 200))
    canvas.controller.candidates = []
    canvas.controller.candidatesChanged.emit([])
    QTest.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=target)
    canvas.repaint()
    assert not canvas._painted_candidates
    QTest.mouseDClick(canvas, Qt.MouseButton.LeftButton, pos=target)
    QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton, pos=target)
    assert not observed.starts and not observed.finishes and not observed.candidates
    # The ordinary system double-click interval, not a new manual-input delay.
    stamp, position, batch = canvas._last_candidate_click
    canvas._last_candidate_click = (stamp - QApplication.doubleClickInterval() / 1000. - .01, position, batch)
    QTest.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=target)
    assert len(observed.starts) == 1 and not observed.finishes
