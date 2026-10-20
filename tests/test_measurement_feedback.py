"""Current-task feedback and a viewport-only, click-through presentation card."""
from concurrent.futures import Future
from dataclasses import FrozenInstanceError, asdict
from types import SimpleNamespace
import time

import numpy as np
import pytest
from PySide6.QtCore import QCoreApplication, QEvent, QPoint, Qt
from PySide6.QtGui import QColor, QPainter
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QScrollArea, QWidget

from fdm.live_experiment import ExperimentSession, default_materials
from fdm.live_experiment.controller import CalibrationSnapshot, LiveExperimentController
from fdm.live_experiment.feedback import MeasurementFeedback, feedback_for_reason
from fdm.live_experiment.frames import FrameHandle, PresentedFrameToken
from fdm.ui.measurement_feedback_overlay import MeasurementFeedbackOverlay


@pytest.fixture
def controller(tmp_path, monkeypatch):
    session = ExperimentSession(save_images=False)
    session.add_operator("反馈测试")
    session.set_materials(default_materials())
    instance = LiveExperimentController(session, evidence_directory=tmp_path)
    instance._timer.stop()
    instance.set_calibration(CalibrationSnapshot(1, 1))
    frame = FrameHandle.snapshot(np.full((120, 160, 3), 180, np.uint8),
                                 source_id="feedback-camera", generation=4, sequence=7, live=False)
    instance.receive_frame(frame)
    instance.mark_presented(PresentedFrameToken(frame, instance.field_epoch, 0, 1, (0, 0)))
    monkeypatch.setattr(instance._manual_pool, "submit", lambda *_: Future())
    yield instance
    instance.close()
    assert instance.budget.used_bytes == 0


def queue(controller):
    controller.start((10, 10))
    controller.finish((10, 30), "wool", method="assisted")
    return controller._jobs[-1]


@pytest.mark.parametrize("reason", ["line_too_short", "endpoints_not_background", "no_complete_crossing",
    "edge_tracking_failed", "edge_span_too_short", "edges_not_parallel", "crossing_too_oblique", "invalid_width"])
def test_native_rejections_publish_chinese_with_exact_task_context(controller, reason):
    structured, compatible = [], []
    controller.measurementFeedback.connect(structured.append)
    controller.resultRejected.connect(compatible.append)
    before = controller.session.to_dict()
    intent, future = queue(controller)
    future.set_result({"status": "rejected", "reason": reason})
    controller.poll()
    assert len(structured) == len(compatible) == 1
    event = structured[0]
    assert event.code == reason and event.message == compatible[0]
    assert reason not in event.message
    assert any("\u3400" <= char <= "\u9fff" for char in event.message)
    assert event.intent_id == intent.id
    assert (event.field_id, event.field_epoch) == (controller.field_id, controller.field_epoch)
    assert (event.source_id, event.source_generation, event.frame_id) == (
        "feedback-camera", 4, "feedback-camera:4:7")
    assert not controller._jobs and controller.session.to_dict() == before
    assert all(value is None or isinstance(value, (str, int)) for value in asdict(event).values())
    with pytest.raises(FrozenInstanceError):
        event.message = "不能修改"


@pytest.mark.parametrize("mutation", ["field", "source", "calibration", "operator", "direction", "cancel"])
def test_invalidated_manual_task_emits_neither_feedback_nor_legacy_rejection(controller, mutation):
    structured, compatible = [], []
    controller.measurementFeedback.connect(structured.append)
    controller.resultRejected.connect(compatible.append)
    intent, future = queue(controller)
    if mutation == "field":
        controller.end_field()
    elif mutation == "source":
        controller.observe_source_identity(("other-camera", 5, (160, 120)))
    elif mutation == "calibration":
        controller.set_calibration(CalibrationSnapshot(2, 2))
    elif mutation == "operator":
        controller.session.add_operator("另一操作者")
    elif mutation == "direction":
        controller.session.active_direction = "weft"
    else:
        intent.cancel.set()
    future.set_exception(RuntimeError("obsolete worker exception"))
    controller.poll()
    assert not structured and not compatible and not controller._jobs


def test_compatibility_callback_cannot_place_old_feedback_on_a_new_field(controller):
    structured = []
    controller.measurementFeedback.connect(structured.append)
    controller.resultRejected.connect(lambda _: controller.end_field())
    _, future = queue(controller)
    future.set_result({"status": "rejected", "reason": "no_complete_crossing"})
    controller.poll()
    assert not structured


def test_unknown_worker_details_are_localized_without_hiding_diagnostics(controller):
    events = []
    controller.measurementFeedback.connect(events.append)
    _, future = queue(controller)
    future.set_exception(RuntimeError("injected native failure"))
    controller.poll()
    assert len(events) == 1 and events[0].severity == "error"
    assert events[0].code == "measurement_failed"
    assert events[0].detail == "injected native failure"
    assert "injected" not in events[0].message
    assert feedback_for_reason(ValueError("类别或范围已变化，请重新确认。")).message.startswith("类别或范围已变化")


def test_nonpositive_success_payload_is_rejected_as_invalid_width(controller):
    events = []
    controller.measurementFeedback.connect(events.append)
    _, future = queue(controller)
    future.set_result({"status": "ok", "reason": "ok", "diameter_um": 0})
    controller.poll()
    assert len(events) == 1 and events[0].code == "invalid_width"
    assert not controller.session.measurements


def test_cancelled_future_is_not_a_new_failure(controller):
    structured, compatible = [], []
    controller.measurementFeedback.connect(structured.append)
    controller.resultRejected.connect(compatible.append)
    _, future = queue(controller)
    future.cancel()
    controller.poll()
    assert not structured and not compatible and not controller._jobs


def automatic_context(controller, response):
    token = controller.presented
    lease = controller.budget.acquire(token.frame)
    controller._automatic_context = {"task_id": "analysis-7", "token": token, "lease": lease,
        "calibration": controller.calibration, "algorithm": "traditional",
        "started_at": time.perf_counter(), "completed": False,
        "operator_id": controller.session.active_operator_id, "direction": controller.session.active_direction}
    responses = [response]
    controller._automatic = SimpleNamespace(poll=lambda: [responses.pop()] if responses else [],
                                            cancel=lambda: None, close=lambda: None)


@pytest.mark.parametrize("scenario", ["current", "old_task", "old_operator", "cancelled", "empty"])
def test_automatic_feedback_requires_current_task_and_context(controller, scenario):
    events = []
    controller.measurementFeedback.connect(events.append)
    automatic_context(controller, {"task_id": "old-7" if scenario == "old_task" else "analysis-7",
        "result": {"status": "cancelled" if scenario == "cancelled" else "empty" if scenario == "empty" else "failed",
                   "reason": "worker_crashed", "candidates": []}})
    if scenario == "old_operator":
        controller.session.add_operator("新操作者")
    controller.poll()
    if scenario == "current":
        assert len(events) == 1 and events[0].intent_id == "analysis-7"
        assert events[0].code == "worker_crashed"
    else:
        assert not events
    if scenario == "empty":
        # Empty is a completed stable cycle, not an active task or a reason to
        # continually repeat the same feedback on every controller poll.
        context = controller._automatic_context
        assert context["completed"] and not controller.busy
        assert not controller.candidates and not controller.session.measurements
        controller.poll()
        assert not events and controller._automatic_context is context


@pytest.mark.parametrize("change", ["field", "source", "unavailable", "cancel"])
def test_context_changes_invalidate_feedback(controller, change):
    reasons = []
    controller.feedbackInvalidated.connect(reasons.append)
    if change == "field":
        controller.end_field("视场移动")
    elif change == "source":
        controller.observe_source_identity(("next", 0, (160, 120)))
    elif change == "unavailable":
        controller.unavailable("来源不可用")
    else:
        controller.cancel_pending_measurements()
    assert reasons


class SolidCanvas(QWidget):
    def __init__(self):
        super().__init__()
        self.presses = []

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#56779A"))

    def mousePressEvent(self, event):
        self.presses.append(event.button())
        event.accept()


@pytest.fixture
def viewport():
    scroll = QScrollArea()
    canvas = SolidCanvas()
    canvas.resize(1200, 900)
    scroll.setWidget(canvas)
    scroll.resize(640, 480)
    scroll.show()
    QApplication.processEvents()
    overlay = MeasurementFeedbackOverlay(scroll.viewport(), canvas=canvas)
    yield scroll, canvas, overlay
    scroll.close()
    scroll.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def show(overlay):
    overlay.show_feedback(feedback_for_reason("no_complete_crossing"))
    QApplication.processEvents()


def test_overlay_is_a_mouse_transparent_viewport_child_and_not_image_content(viewport):
    scroll, canvas, overlay = viewport
    original = canvas.grab().toImage()
    show(overlay)
    assert overlay.isVisible() and not overlay.isWindow()
    assert overlay.parentWidget() is scroll.viewport()
    assert overlay.title_label.text() == "本次测量未通过，未计入记录"
    assert overlay.testAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
    assert not overlay.testAttribute(Qt.WidgetAttribute.WA_NativeWindow)
    assert canvas.grab().toImage() == original
    assert QApplication.widgetAt(overlay.mapToGlobal(overlay.rect().center())) is canvas


@pytest.mark.parametrize("button", [Qt.MouseButton.LeftButton, Qt.MouseButton.MiddleButton, Qt.MouseButton.RightButton])
def test_dismissal_never_consumes_the_measurement_or_pan_click(viewport, button):
    scroll, canvas, overlay = viewport
    show(overlay)
    point = overlay.mapTo(canvas, overlay.rect().center())
    QTest.mouseClick(canvas, button, pos=point)
    assert not overlay.isVisible() and overlay.feedback is None
    assert canvas.presses == [button]
    show(overlay)
    QTest.mouseClick(scroll.viewport(), button, pos=QPoint(3, 3))
    assert overlay.feedback is None


def test_overlay_stays_in_visible_viewport_when_panned_and_resized(viewport):
    scroll, canvas, overlay = viewport
    show(overlay)
    before = overlay.geometry()
    scroll.horizontalScrollBar().setValue(500)
    scroll.verticalScrollBar().setValue(350)
    QApplication.processEvents()
    assert overlay.geometry() == before
    for size in ((360, 240), (800, 600)):
        scroll.resize(*size)
        QApplication.processEvents()
        assert scroll.viewport().rect().contains(overlay.geometry())
        assert abs(overlay.geometry().center().x() - scroll.viewport().rect().center().x()) <= 1
        assert overlay.y() == 16


def test_success_card_replaces_failure_and_old_timeout_cannot_hide_it(viewport):
    scroll, canvas, overlay = viewport
    overlay.show_feedback(feedback_for_reason("no_complete_crossing"), duration_ms=20)
    QTest.qWait(5)
    event = MeasurementFeedback("target_reached", "已完成 100 根测量，可生成报告。", "success")
    overlay.show_feedback(event)
    QTest.qWait(35)
    assert overlay.feedback is event and overlay.isVisible()
    assert overlay.title_label.text() == "已达到目标"
    scroll.hide()
    QApplication.processEvents()
    assert overlay.feedback is None
