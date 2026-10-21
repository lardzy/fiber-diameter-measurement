"""Clicks between live-frame delivery and painting use the visible frame.

Events are delivered without pumping Qt between frame arrival and mouse press:
processing queued paints there would hide the production timing regression.
"""
import time

import numpy as np
import pytest
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication

from fdm.live_experiment.frames import FrameHandle
from test_live_experiment_content_workspace import workspace
from test_trial_assistance_ui import candidates, point


def _mouse(view, event_type, xy, *, held=False):
    position = point(view, *xy)
    event = QMouseEvent(
        event_type, QPointF(position), QPointF(view.canvas.mapToGlobal(position)),
        Qt.MouseButton.NoButton if event_type == QEvent.Type.MouseMove else Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton if held else Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
    )
    QApplication.sendEvent(view.canvas, event)


def _click(view, xy):
    _mouse(view, QEvent.Type.MouseButtonPress, xy, held=True)
    _mouse(view, QEvent.Type.MouseButtonRelease, xy)


def _pump(view, duration):
    deadline = time.monotonic() + duration
    while time.monotonic() < deadline:
        QApplication.processEvents()
        view.controller.poll()
        time.sleep(.001)


def _prepare_live_stripe(view, mode):
    controller = view.controller
    controller._timer.stop()
    view.input_mode.setCurrentIndex(view.input_mode.findData(mode))
    view.gesture.setCurrentIndex(0)
    view.measure_mode.setCurrentIndex(view.measure_mode.findData("assisted"))
    QApplication.processEvents()
    view.canvas.repaint()
    original = controller.presented.frame
    pixels = np.full_like(original.pixels, 240)
    pixels[20:220, 130:146] = 30
    if pixels.ndim == 3 and pixels.shape[2] == 4:
        pixels[:, :, 3] = 255
    # Keep real motion detection and assisted measurement enabled. Repeated
    # camera frames establish stability instead of assigning controller.state.
    for sequence in range(1, 12):
        frame = FrameHandle.snapshot(
            pixels, source_id=original.source_id, generation=original.generation,
            sequence=original.sequence + sequence, live=True, metadata=original.metadata,
        )
        view._receive_frame(frame)
        _pump(view, .04)
    view.canvas.repaint()
    assert controller.state == "ready"
    assert controller.presented.frame.live
    return controller.presented


def _arrive_without_paint(view):
    visible = view.controller.presented
    frame = visible.frame
    incoming = FrameHandle.snapshot(
        frame.pixels, source_id=frame.source_id, generation=frame.generation,
        sequence=frame.sequence + 1, live=True, metadata=frame.metadata,
    )
    view._receive_frame(incoming)
    assert view.controller.presented is visible
    assert view.canvas.frame is incoming
    assert frame is not incoming
    assert np.array_equal(frame.pixels, incoming.pixels)
    assert view.controller.state == "ready"
    return visible


def _await_record(view):
    deadline = time.monotonic() + 3
    while not view.session.measurements and time.monotonic() < deadline:
        _pump(view, .01)
    assert len(view.session.measurements) == 1
    return next(iter(view.session.measurements.values()))


@pytest.mark.parametrize("mode", ["mixed", "manual"])
@pytest.mark.parametrize("gap_at", ["first", "second"])
def test_live_two_point_clicks_survive_unpainted_frame(workspace, mode, gap_at):
    view = workspace
    controller = view.controller
    _prepare_live_stripe(view, mode)
    if gap_at == "first":
        visible = _arrive_without_paint(view)
    else:
        visible = controller.presented
    _click(view, (100, 100))
    assert controller.draft is not None, "The first visible-frame click must not disappear"
    assert controller.draft[0] is visible
    assert controller.draft[1] == pytest.approx((100, 100))
    assert not controller._jobs
    QApplication.processEvents()
    view.canvas.repaint()
    if gap_at == "second":
        _arrive_without_paint(view)
    _click(view, (180, 100))
    assert controller.draft is None, "The second click must finish the existing draft"
    assert len(controller._jobs) == 1
    intent = controller._jobs[0][0]
    assert intent.token is visible
    assert intent.start == pytest.approx((100, 100))
    assert intent.end == pytest.approx((180, 100))
    record = _await_record(view)
    assert record.frame_id == visible.frame.frame_id
    assert record.diameter_um == pytest.approx(15, abs=.1)
    assert not controller._jobs
    # This next intended first point previously completed the abandoned old
    # draft across blank background, producing no_complete_crossing.
    _click(view, (80, 130))
    assert controller.draft is not None
    assert controller.draft[1] == pytest.approx((80, 130))
    assert not controller._jobs
    assert len(view.session.measurements) == 1


def test_mixed_drag_after_frame_arrival_measures_the_pressed_visible_frame(workspace):
    view = workspace
    controller = view.controller
    _prepare_live_stripe(view, "mixed")
    view.gesture.setCurrentIndex(1)
    visible = _arrive_without_paint(view)
    _mouse(view, QEvent.Type.MouseButtonPress, (100, 100), held=True)
    assert view.canvas._pointer_gesture is not None
    assert view.canvas._pointer_gesture["token"] is visible
    _mouse(view, QEvent.Type.MouseMove, (180, 100), held=True)
    assert controller.draft is not None
    assert controller.draft[0] is visible
    # Let the next frame become visible while the button is still held. The
    # measurement must retain the original press frame and its coordinates.
    QApplication.processEvents()
    view.canvas.repaint()
    assert controller.presented.frame is not visible.frame
    _mouse(view, QEvent.Type.MouseButtonRelease, (180, 100))
    assert controller.draft is None
    assert len(controller._jobs) == 1
    intent = controller._jobs[0][0]
    assert intent.token is visible
    assert intent.start == pytest.approx((100, 100))
    assert intent.end == pytest.approx((180, 100))
    record = _await_record(view)
    assert record.frame_id == visible.frame.frame_id
    assert record.diameter_um == pytest.approx(15, abs=.1)


def test_mixed_candidate_click_after_frame_arrival_keeps_other_candidates(workspace):
    view = workspace
    controller = view.controller
    _prepare_live_stripe(view, "mixed")
    ids = candidates(view)
    visible = _arrive_without_paint(view)
    _mouse(view, QEvent.Type.MouseButtonPress, (90, 40), held=True)
    assert view.canvas._pointer_gesture is not None
    assert view.canvas._pointer_gesture["kind"] == "candidate"
    assert view.canvas._pointer_gesture["token"] is visible
    assert not view.session.measurements
    _mouse(view, QEvent.Type.MouseButtonRelease, (90, 40))
    record = _await_record(view)
    assert record.frame_id == visible.frame.frame_id
    assert record.diameter_um == pytest.approx(20)
    assert controller.draft is None
    assert not controller._jobs
    assert [item["candidate_id"] for item in controller.candidates] == ids
    assert [item["state"] for item in controller.candidates] == ["available", "accepted", "available"]
