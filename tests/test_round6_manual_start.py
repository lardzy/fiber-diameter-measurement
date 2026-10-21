"""The first point never submits, with or without a previous accepted task."""
import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from test_live_experiment_content_workspace import workspace
from test_live_measurement_mixed_canvas import mouse_move, _await_records
from test_trial_assistance_ui import point


@pytest.mark.parametrize("automatic", [False, True])
@pytest.mark.parametrize("mode", ["mixed", "manual"])
@pytest.mark.parametrize("previous_measurement", [False, True])
def test_first_point_unpressed_motion_cannot_submit(workspace, automatic, mode, previous_measurement):
    controller = workspace.controller
    workspace.input_mode.setCurrentIndex(workspace.input_mode.findData(mode))
    workspace.measure_mode.setCurrentIndex(workspace.measure_mode.findData("free"))
    workspace.gesture.setCurrentIndex(0)
    controller.automatic_enabled = automatic
    if previous_measurement:
        QTest.mouseClick(workspace.canvas, Qt.MouseButton.LeftButton, pos=point(workspace, 30, 30))
        QTest.mouseClick(workspace.canvas, Qt.MouseButton.LeftButton, pos=point(workspace, 60, 30))
        _await_records(workspace, 1)
    count = len(controller.session.measurements)
    QTest.mouseClick(workspace.canvas, Qt.MouseButton.LeftButton, pos=point(workspace, 100, 100))
    draft = controller.draft
    assert draft is not None
    for index in range(1000):
        mouse_move(workspace.canvas, point(workspace, 120 + index % 40, 120), held=False)
        if index % 100 == 0:
            QApplication.processEvents()
    controller.poll()
    assert controller.draft is draft
    assert not controller._jobs
    assert len(controller.session.measurements) == count
    assert not workspace.measurement_feedback.isVisible()
    assert not any(item["event"] == "task_submitted"
                   for item in controller.performance_frame_context()["input_trace"]["events"])
