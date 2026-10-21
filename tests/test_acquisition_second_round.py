"""Second-round acceptance through real navigation and input dispatch."""
from pathlib import Path
import importlib.util

import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest

from fdm.live_experiment.feedback import feedback_for_reason
from fdm.live_experiment.reminders import ReminderPolicy
from fdm.ui.quantity_targets import QuantityTargetDialog

_spec = importlib.util.spec_from_file_location("second_round_workflow",
    Path(__file__).resolve().parents[1] / "scripts/validation/validate_workflow_redesign.py")
workflow = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(workflow)


def rect(widget, window):
    origin = widget.mapTo(window, QPoint(0, 0))
    return (origin.x(), origin.y(), widget.width(), widget.height())


@pytest.fixture
def harness(tmp_path):
    with workflow.WorkflowHarness(tmp_path / "profile") as item:
        yield item


def test_real_navigation_keeps_identical_image_frame_and_sidebar_width(harness):
    harness.enter()
    window = harness.window
    reference = rect(window._preview_display_stack, window)
    for index in (1, 2, 0) * 67:
        page = harness.select(index)
        image = page.image_scroll if index else window._preview_display_stack
        assert rect(image, window) == reference
        assert not window._measure_toolbar.isVisible()
        assert not window._inspector_dock.isVisible()
    assert not harness.observer.unexpected()
    harness.assert_clean()


def test_shared_sidebar_width_survives_pages_and_new_window(tmp_path):
    profile = tmp_path / "profile"
    with workflow.WorkflowHarness(profile) as h:
        h.enter()
        page = h.select(1)
        handle = page.body_splitter.handle(1)
        origin = handle.rect().center()
        QTest.mousePress(handle, Qt.MouseButton.LeftButton, pos=origin)
        QTest.mouseMove(handle, origin - QPoint(35, 0))
        QTest.mouseRelease(handle, Qt.MouseButton.LeftButton, pos=origin - QPoint(35, 0))
        workflow.settle()
        expected = page.body_splitter.sizes()[1]
        for index in (2, 0, 1):
            page = h.select(index)
            splitter = page.body_splitter if index else h.window._preview_body_splitter
            assert splitter.sizes()[1] == expected
    with workflow.WorkflowHarness(profile) as h:
        h.enter()
        for index in (0, 2, 1):
            page = h.select(index)
            splitter = page.body_splitter if index else h.window._preview_body_splitter
            assert splitter.sizes()[1] == expected
        h.assert_clean()


def test_motion_status_never_inserts_a_readiness_row(harness):
    harness.enter()
    page = harness.select(1)
    page.ruler_choice.setCurrentIndex(1)
    page.ruler_choice.activated.emit(1)
    page.session.rename_operator("测试员", operator_id=page.session.active_operator_id)
    from fdm.live_experiment import default_materials
    page.session.set_materials(default_materials())
    page._refresh_operators()
    page._refresh_measurement_readiness()
    workflow.settle()
    baseline = [rect(w, harness.window) for w in (page.workflow_tabs, page.image_scroll, page.sample, page.save_button)]
    # Navigation above is the actual route. Exercise the controller's motion
    # transition rather than changing the selected page/internal layout state.
    for _ in range(100):
        page.controller.end_field("视场已变化，请重新测量")
        page._refresh_measurement_readiness()
        workflow.settle(2)
        assert not page.readiness_label.isVisible()
        assert baseline == [rect(w, harness.window) for w in (page.workflow_tabs, page.image_scroll, page.sample, page.save_button)]
    assert "持续采集" not in page.readiness_label.text()


def test_rejected_feedback_is_viewport_anchored_and_click_passes_through(harness):
    harness.enter()
    page = harness.select(1)
    page.ruler_choice.setCurrentIndex(1)
    page.ruler_choice.activated.emit(1)
    page.display_binding.set_mode("fit")
    page.input_mode.setCurrentIndex(page.input_mode.findData("manual"))
    page.gesture.setCurrentIndex(0)
    # The explicit input route retains a valid presented frame before emitting
    # a manual-start signal. Establish readiness through actual capture frames.
    frame = workflow.synthetic_frame()
    for _ in range(12):
        harness.backend.tick(image=frame)
        QTest.qWait(40)
    workflow.settle()
    assert page.controller.state == "ready"
    before = page.session.to_dict()
    feedback = feedback_for_reason("no_complete_crossing", **page.controller.feedback_context())
    page.controller.measurementFeedback.emit(feedback)
    overlay = page.measurement_feedback
    assert overlay.isVisible()
    assert "本次未通过" in page.measurement_outcome.text()
    assert overlay.parentWidget() is page.image_scroll.viewport()
    assert page.session.to_dict() == before
    clicked = []
    page.canvas.manualGestureStarted.connect(lambda point, token: clicked.append((point, token)))
    # Receive click without initiating an asynchronous measurement in this
    # presentation test; controller validity/ledger gates have separate tests.
    page.canvas.manualGestureStarted.disconnect(page._manual_start)
    QTest.mouseClick(page.canvas, Qt.MouseButton.LeftButton, pos=page.canvas.rect().center())
    assert clicked and not overlay.isVisible()
    page.controller.measurementFeedback.emit(feedback)
    page.controller.end_field("视场变化")
    assert not overlay.isVisible()
    page.controller.measurementFeedback.emit(feedback)  # stale epoch
    assert not overlay.isVisible()


def test_goal_notice_progress_and_undo_are_separate_from_status(harness):
    harness.enter()
    page = harness.select(1)
    from fdm.live_experiment import default_materials
    page.session.set_materials(default_materials())
    page._refresh_materials()
    page._configure(reminder_policy=ReminderPolicy(content_measured_target=2, content_total_target=3))
    page.statusBar().showMessage("上次测量未通过", 10000)
    for _ in range(2):
        record = page.session.add_measurement(20, "wool")
        page.controller.recordAccepted.emit(record)
        page.controller.changed.emit()
    assert page.measurement_feedback.feedback.code == "target_reached"
    assert "已计入" in page.statusBar().currentMessage()
    assert "未通过" not in page.statusBar().currentMessage()
    page._refresh_summary()
    assert page.material_rows["wool"]["measured_count"].text() == "✓ 2/2"
    page.controller.statusChanged.emit("画面采集中")
    assert page.measurement_feedback.isVisible()
    before = page.session.revision
    page.measurement_feedback.dismiss()
    assert page.session.revision == before
    page._history_action("undo")
    page._history_action("redo")
    assert not page.measurement_feedback.isVisible()
    for index in (2, 0, 1):
        harness.select(index)
    assert not page.measurement_feedback.isVisible()


def test_legacy_target_editor_requires_explicit_opt_in(harness):
    harness.enter()
    page = harness.select(2)
    page.session.configure(reminder_policy=None, measured_reminder=17, total_reminder=29)
    before = page.session.to_dict()
    dialog = QuantityTargetDialog(page.session, page)
    assert not dialog.enable_new.isChecked()
    assert dialog.changes() == {"measured_reminder": 17, "total_reminder": 29}
    dialog.enable_new.setChecked(True)
    assert dialog.changes()["reminder_policy"].diameter_measured_target == 17
    dialog.reject()
    assert page.session.to_dict() == before


def test_acquisition_action_errors_reach_the_visible_status_bar(harness):
    harness.enter()
    harness.window.statusBar().showMessage("显示偏好保存失败", 8000)
    assert harness.window._preview_inline_status.currentMessage() == "显示偏好保存失败"
    page = harness.select(1)
    harness.window.statusBar().showMessage("采集操作失败", 8000)
    assert page.statusBar().currentMessage() == "采集操作失败"
    assert page.statusBar().isVisible()
    harness.assert_clean()
