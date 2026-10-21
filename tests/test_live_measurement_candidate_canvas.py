"""Actual Qt pointer, tab and scrolling routes for candidate presentation."""
from types import SimpleNamespace
from time import monotonic

import numpy as np
import pytest
from PySide6.QtCore import QEvent, QEventLoop, QObject, QPoint, QPointF, Qt, QTimer, Signal
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QScrollArea, QTabWidget, QWidget

from fdm.live_experiment import ExperimentSession, default_materials
from fdm.live_experiment.frames import FrameBudget, FrameHandle
from fdm.ui.live_measurement_canvas import LiveMeasurementCanvas


class CanvasController(QObject):
    changed = Signal()
    fieldChanged = Signal()
    candidatesChanged = Signal(object)
    candidateStateChanged = Signal()

    def __init__(self):
        super().__init__()
        self.session = ExperimentSession(save_images=False)
        self.session.add_operator("测试")
        self.session.set_materials(default_materials())
        self.calibration = None
        self.candidates = []
        self.current_records = []
        self.pending_classification = []
        self._confirmation = None
        self._jobs = []
        self.draft = None
        self.presented = None
        self.field_epoch = 1
        self.candidate_overlay_ready = True
        self.budget = FrameBudget()
        self.candidate_interactions = 0

    def mark_presented(self, token):
        self.presented = token

    def candidates_aligned_for(self, frame):
        return self.candidate_overlay_ready

    def record_candidate_painted(self, frame):
        pass

    def record_candidate_visibility(self, frame, visible, *, display_enabled=True):
        pass

    def should_defer_canvas_business_update(self):
        return False

    def pointer_token_is_current(self, token):
        current = self.presented
        return (current is not None and token.field_epoch == self.field_epoch
                and token.frame.source_id == current.frame.source_id
                and token.frame.generation == current.frame.generation
                and token.display_revision == current.display_revision)

    def retain_pointer_frame(self, token):
        if not self.pointer_token_is_current(token):
            raise ValueError("测量画面已变化")
        return self.budget.acquire(token.frame)

    def release_pointer_frame(self, lease):
        self.budget.release(lease)

    def candidate_interaction_started(self):
        self.candidate_interactions += 1


@pytest.fixture
def canvas():
    controller = CanvasController()
    widget = LiveMeasurementCanvas(controller)
    widget.resize(800, 600)
    widget.set_frame(FrameHandle.snapshot(np.full((576, 768, 3), 180, np.uint8), source_id="image", live=False))
    widget.show()
    QApplication.processEvents()
    yield widget
    widget.close()
    widget.deleteLater()


def add_candidates(canvas, endpoints=None):
    endpoints = endpoints or [((200, 200), (230, 200)), ((400, 300), (425, 315))]
    canvas.controller.candidates = [dict(candidate_id=f"batch:{index}", ordinal=index + 1,
                                        start=start, end=end, state="available")
                                    for index, (start, end) in enumerate(endpoints)]
    canvas.controller.candidatesChanged.emit(canvas.controller.candidates)
    QApplication.processEvents()


def pointer(canvas, image_point):
    scale, offset = canvas._transform()
    return QPoint(round(image_point[0] * scale + offset[0]), round(image_point[1] * scale + offset[1]))


def test_candidate_click_is_explicit_and_manual_mouse_route_is_preserved(canvas):
    add_candidates(canvas)
    canvas.set_input_mode("manual")
    accepted, starts, finishes = [], [], []

    def start(point, token):
        starts.append((point, token))
        canvas.controller.draft = (token, point)

    def finish(point):
        finishes.append(point)
        canvas.controller.draft = None

    canvas.candidateClicked.connect(accepted.append)
    canvas.manualGestureStarted.connect(start)
    canvas.manualGestureFinished.connect(finish)
    QTest.mousePress(canvas, Qt.MouseButton.LeftButton, pos=pointer(canvas, (215, 200)))
    assert len(starts) == 1 and starts[0][0] == (215., 200.) and not finishes
    QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton, pos=pointer(canvas, (215, 200)))
    assert not accepted and not finishes
    QTest.mousePress(canvas, Qt.MouseButton.LeftButton, pos=pointer(canvas, (250, 200)))
    assert finishes == [(250., 200.)]
    QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton, pos=pointer(canvas, (250, 200)))
    canvas.set_candidate_input_enabled(True)
    QTest.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=pointer(canvas, (215, 200)))
    QTest.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=pointer(canvas, (700, 500)))
    assert accepted == ["batch:0"]
    assert len(starts) == len(finishes) == 1


def test_hidden_invalid_and_confirming_candidates_cannot_be_clicked(canvas):
    add_candidates(canvas)
    accepted = []
    canvas.candidateClicked.connect(accepted.append)
    canvas.set_candidate_input_enabled(True)
    for reason in ("hidden", "invalid", "confirming", "accepted"):
        canvas.set_candidate_presentation(reason != "hidden")
        canvas.controller.candidate_overlay_ready = reason != "invalid"
        canvas.controller.candidates[0]["state"] = reason if reason in {"confirming", "accepted"} else "available"
        QApplication.processEvents()
        QTest.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=pointer(canvas, (215, 200)))
    assert not accepted


def test_newly_painted_frame_is_gated_before_presentation_token_publication(canvas, monkeypatch):
    add_candidates(canvas)
    previous = canvas.controller.presented
    monkeypatch.setattr(canvas.controller, "candidates_aligned_for", lambda frame: frame.sequence == 0)
    canvas.set_frame(FrameHandle.snapshot(np.full((576, 768, 3), 180, np.uint8),
                                         source_id="image", sequence=1, live=False))
    # The old presented token is still verified, but paint is about to use the
    # newer image. No old overlay may appear for even this one repaint.
    assert canvas.controller.presented is previous
    assert not any(item["kind"] == "candidate" for item in canvas._overlay_items()[0])
    canvas.repaint()
    assert canvas.controller.presented.frame.sequence == 1
    assert not canvas.candidate_hits(QPointF(pointer(canvas, (215, 200))))


def test_candidate_label_is_hit_target_and_hover_links_without_manual_interception(canvas):
    add_candidates(canvas)
    accepted, hovered = [], []
    canvas.candidateClicked.connect(accepted.append)
    canvas.candidateHovered.connect(hovered.append)
    canvas.set_candidate_input_enabled(True)
    canvas.activateWindow()
    canvas.raise_()
    # Native QPA mouse movement is asynchronous and may not emit a move when
    # the OS cursor is already at the target from a preceding test. Move away
    # first and drive the real event loop, rather than changing selection.
    QTest.mouseMove(canvas, QPoint(30, 30))
    loop = QEventLoop()
    QTimer.singleShot(30, loop.quit)
    loop.exec()
    rect = canvas._label_placements["candidate:batch:0"].rect
    QTest.mouseMove(canvas, rect.center().toPoint())
    deadline = monotonic() + 1.
    while (not hovered or hovered[-1] != "batch:0") and monotonic() < deadline:
        QTimer.singleShot(20, loop.quit)
        loop.exec()
    assert hovered and hovered[-1] == "batch:0"
    # A highlight can rearrange dense labels; use its visible label location.
    QApplication.processEvents()
    rect = canvas._label_placements["candidate:batch:0"].rect
    QTest.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=rect.center().toPoint())
    assert accepted == ["batch:0"]


def test_overlapping_candidates_request_disambiguation_without_accepting(canvas):
    add_candidates(canvas, [((200, 200), (230, 200)), ((200, 202), (230, 202))])
    ambiguous, accepted = [], []
    canvas.candidateAmbiguous.connect(ambiguous.append)
    canvas.candidateClicked.connect(accepted.append)
    canvas.set_candidate_input_enabled(True)
    QTest.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=pointer(canvas, (215, 201)))
    assert ambiguous == [["batch:0", "batch:1"]]
    assert not accepted


@pytest.mark.parametrize("mode,value", [("pixel", 100), ("percent", 50), ("percent", 200), ("fit", 100)])
def test_hit_distance_is_in_logical_pixels_and_candidates_do_not_block_zoom(canvas, mode, value):
    add_candidates(canvas)
    canvas.set_display_mode(mode, value)
    QApplication.processEvents()
    position = pointer(canvas, (215, 200))
    assert canvas.candidate_hits(QPointF(position + QPoint(0, 7))) == ["batch:0"]
    assert canvas.candidate_hits(QPointF(position + QPoint(0, 10))) == []


def test_actual_tab_navigation_and_scroll_keep_labels_in_visible_viewport(canvas):
    tabs = QTabWidget()
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setWidget(canvas)
    tabs.addTab(scroll, "含量试验")
    tabs.addTab(QWidget(), "实时采集")
    tabs.resize(500, 380)
    tabs.show()
    add_candidates(canvas, [((500, 400), (530, 420))])
    for _ in range(5):
        QTest.mouseClick(tabs.tabBar(), Qt.MouseButton.LeftButton, pos=tabs.tabBar().tabRect(1).center())
        QTest.mouseClick(tabs.tabBar(), Qt.MouseButton.LeftButton, pos=tabs.tabBar().tabRect(0).center())
    scroll.horizontalScrollBar().setValue(scroll.horizontalScrollBar().maximum())
    scroll.verticalScrollBar().setValue(scroll.verticalScrollBar().maximum())
    QApplication.processEvents()
    canvas.repaint()
    rect = canvas._label_placements["candidate:batch:0"].rect
    assert canvas._visible_viewport().contains(rect)
    assert canvas.controller.presented.frame is canvas.frame
    scroll.takeWidget()
    canvas.setParent(None)
    tabs.close()
    tabs.deleteLater()


def test_accepted_ordinals_survive_deletion_and_frame_arrival(canvas):
    session = canvas.controller.session
    record_a = session.add_measurement(20, "wool", endpoints=((200, 200), (230, 210)),
                                       provenance={"measurement_number": 12})
    record_b = session.add_measurement(21, "wool", endpoints=((400, 300), (425, 315)),
                                       provenance={"measurement_number": 15})
    canvas.controller.current_records[:] = [record_a.id, record_b.id]
    canvas.repaint()
    first = dict(canvas._label_placements)
    canvas.set_frame(FrameHandle.snapshot(np.full((576, 768, 3), 182, np.uint8),
                                         source_id="image", sequence=1, live=False))
    canvas.repaint()
    assert canvas._label_placements == first
    canvas.controller.current_records.remove(record_a.id)
    assert canvas._overlay_items()[0][0]["label"] == "#15"


def test_display_change_preserves_confirmation_guard_and_opacity_is_clamped(canvas):
    add_candidates(canvas)
    canvas.set_candidate_presentation(True, 60)
    assert canvas.candidate_opacity == .6
    canvas.set_candidate_presentation(True, .1)
    assert canvas.candidate_opacity == .2
    with pytest.raises(ValueError):
        canvas.set_candidate_presentation(True, float("nan"))
    canvas.controller._confirmation = {"kind": "automatic"}
    with pytest.raises(ValueError, match="待确认"):
        canvas.set_display_mode("fit")


def test_alignment_visibility_alternation_reuses_geometry_without_enabling_hidden_hits(canvas, monkeypatch):
    import fdm.ui.live_measurement_canvas as canvas_module

    add_candidates(canvas)
    record = canvas.controller.session.add_measurement(
        20, "wool", endpoints=((450, 400), (470, 420)), provenance={"measurement_number": 1})
    canvas.controller.current_records[:] = [record.id]
    layouts = []
    original = canvas_module.layout_annotations

    def counted_layout(*args, **kwargs):
        layouts.append(args[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(canvas_module, "layout_annotations", counted_layout)
    visible_placements = None
    for index in range(20):
        visible = index % 2 == 0
        canvas.controller.candidate_overlay_ready = visible
        canvas.repaint()
        if visible:
            if visible_placements is None:
                visible_placements = canvas._label_placements
            assert canvas._label_placements is visible_placements
            assert canvas.candidate_hits(QPointF(pointer(canvas, (215, 200)))) == ["batch:0"]
        else:
            assert "candidate:batch:0" not in canvas._label_placements
            assert not canvas.candidate_hits(QPointF(pointer(canvas, (215, 200))))
        assert "record:" + record.id in canvas._label_placements
    assert len(layouts) == 2


def test_annotation_geometry_cache_is_bounded_and_field_change_discards_old_geometry(canvas, monkeypatch):
    from PySide6.QtCore import QRectF

    add_candidates(canvas)
    items, protected = canvas._overlay_items()
    viewport = QRectF(0, 0, 800, 600)
    monkeypatch.setattr(canvas, "_visible_viewport", lambda: viewport)
    canvas._arrange_labels(items, protected)
    initial_key = canvas._annotation_cache_key
    for index in range(1, 7):
        viewport = QRectF(index, 0, 800 - index, 600)
        canvas._arrange_labels(items, protected)
        assert len(canvas._annotation_layout_cache) <= 4
    assert len(canvas._annotation_layout_cache) == 4
    assert initial_key not in canvas._annotation_layout_cache
    # Keys and values have only scalar geometry/labels, never frame/image leases.
    assert all(isinstance(key, tuple) and isinstance(value, dict)
               for key, value in canvas._annotation_layout_cache.items())
    canvas.controller.fieldChanged.emit()
    assert not canvas._annotation_layout_cache
    assert not canvas._label_placements


class PaintEvents(QObject):
    def __init__(self, canvas):
        super().__init__(canvas)
        self.count = 0
        canvas.installEventFilter(self)

    def eventFilter(self, watched, event):
        if event.type() == QEvent.Type.Paint:
            self.count += 1
        return False


def test_business_redraw_joins_next_verified_preview_without_repainting_old_frame(canvas, monkeypatch):
    QTest.qWait(20)
    paints = PaintEvents(canvas)
    previous = canvas.controller.presented
    monkeypatch.setattr(canvas.controller, "should_defer_canvas_business_update", lambda: True)
    record = canvas.controller.session.add_measurement(
        20, "wool", endpoints=((200, 200), (230, 210)), provenance={"measurement_number": 12})
    canvas.controller.current_records.append(record.id)
    canvas.controller.changed.emit()
    QTest.qWait(20)
    assert paints.count == 0
    assert canvas.controller.presented is previous
    assert "record:" + record.id not in canvas._label_placements

    next_frame = FrameHandle.snapshot(np.full((576, 768, 3), 181, np.uint8),
                                      source_id="image", sequence=1, live=True)
    # The controller's already scheduled frameReady delivery is set_frame.
    canvas.set_frame(next_frame)
    QTest.qWait(20)
    assert paints.count >= 1
    assert canvas.controller.presented.frame is next_frame
    assert "record:" + record.id in canvas._label_placements


@pytest.mark.parametrize("notification", ["state", "candidate", "selection", "field"])
def test_nonbusiness_and_candidate_notifications_are_not_deferred(canvas, monkeypatch, notification):
    add_candidates(canvas)
    QTest.qWait(20)
    paints = PaintEvents(canvas)
    monkeypatch.setattr(canvas.controller, "should_defer_canvas_business_update", lambda: True)
    if notification == "state":
        canvas.controller.changed.emit()
    elif notification == "candidate":
        canvas.controller.candidateStateChanged.emit()
    elif notification == "selection":
        canvas.set_selected_candidate("batch:0")
    else:
        canvas.controller.fieldChanged.emit()
    QTest.qWait(20)
    assert paints.count >= 1


def test_active_pointer_and_unavailable_defer_gate_keep_business_updates_immediate(canvas, monkeypatch):
    add_candidates(canvas)
    position = pointer(canvas, (215, 200))
    QTest.mousePress(canvas, Qt.MouseButton.LeftButton, pos=position)
    assert canvas._pointer_gesture is not None
    QTest.qWait(20)
    paints = PaintEvents(canvas)
    monkeypatch.setattr(canvas.controller, "should_defer_canvas_business_update", lambda: True)
    canvas.controller.session.adjust_count("wool", 1)
    canvas.controller.changed.emit()
    QTest.qWait(20)
    assert paints.count >= 1
    QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton, pos=position)
    QTest.qWait(20)
    paints.count = 0
    monkeypatch.setattr(canvas.controller, "should_defer_canvas_business_update", lambda: False)
    canvas.controller.session.adjust_count("wool", 1)
    canvas.controller.changed.emit()
    QTest.qWait(20)
    assert paints.count >= 1
