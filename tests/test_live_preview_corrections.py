from concurrent.futures import Future
from dataclasses import replace
import numpy as np

from PySide6.QtWidgets import QApplication, QMainWindow, QScrollArea

from fdm.live_experiment.calibration_store import load_rulers, save_ruler, source_key
from fdm.live_experiment.controller import CalibrationSnapshot
from fdm.live_experiment.frames import FrameHandle
from test_live_experiment_content_workspace import workspace


def live_frame(sequence=1, generation=1, width=768):
    return FrameHandle.snapshot(np.zeros((576, width, 3), np.uint8), source_id="camera:microview:1",
                                sequence=sequence, generation=generation, live=True)


def test_ruler_survives_restart_but_not_different_device_or_geometry(workspace):
    frame = live_frame()
    value = CalibrationSnapshot(.681234567890123, .681234567890123, source="CU 40x")
    workspace._receive_frame(frame)
    workspace.controller.set_calibration(value)
    workspace._remember_ruler(value)
    assert load_rulers()[source_key(frame)] == value
    workspace.controller.unavailable("停止采集")
    workspace._receive_frame(live_frame(generation=2))
    assert workspace.controller.calibration == value
    workspace.controller.unavailable("换分辨率")
    workspace._receive_frame(live_frame(generation=3, width=640))
    assert workspace.controller.calibration is None
    workspace._receive_frame(live_frame(generation=4))
    assert workspace.controller.calibration == value
    workspace._receive_frame(replace(live_frame(), source_id="camera:another"))
    assert workspace.controller.calibration is None


def test_high_precision_saved_ruler_reloads_from_application_profile(tmp_path):
    frame = live_frame()
    value = CalibrationSnapshot(.6712345678901234, .6712345678901234, source="CU .scl")
    save_ruler(source_key(frame), value)
    assert load_rulers()[source_key(frame)] == value
    assert source_key(frame) == source_key(live_frame(generation=20))
    assert (tmp_path / "capture-rulers.json").exists()


def test_preview_paints_latest_frame_while_motion_is_waiting(workspace):
    controller = workspace.controller
    controller._timer.stop()
    controller._motion_future = (controller.field_epoch, Future())
    for sequence in (1, 2, 3, 4):
        frame = live_frame(sequence=sequence)
        workspace._receive_frame(frame)
        controller._motion_future = (controller.field_epoch, Future())
        QApplication.processEvents()
        assert workspace.canvas.frame is frame
        assert controller.presented.frame is frame
        assert controller.presented.frame.sequence == sequence
    assert not controller.verify_before_present
    assert not controller._motion_future[1].done()
    assert not controller.session.measurements
    controller._motion_future = None


def test_direction_switch_requires_new_field_and_preserves_history(workspace):
    controller = workspace.controller
    record = workspace.session.add_measurement(20, "wool", field_id=controller.field_id)
    controller.current_records.append(record.id)
    workspace.direction.setCurrentIndex(1)
    assert workspace.session.active_direction == "warp"
    assert workspace.direction.currentData() == "warp"
    assert "新视场" in workspace.statusBar().currentMessage()
    controller.end_field()
    workspace.direction.setCurrentIndex(1)
    assert workspace.session.active_direction == "weft"
    assert record.id in workspace.session.measurements


def test_settings_tabs_fit_without_horizontal_scroll_or_hidden_fields(workspace):
    workspace.settings_dialog.resize(620, 480)
    workspace.advanced_toggle.click()
    QApplication.processEvents()
    assert workspace.settings_tabs.count() == 2
    assert workspace.workflow_tabs.count() == 3
    assert not workspace.settings_dialog.isAncestorOf(workspace.sample)
    assert not workspace.settings_dialog.isAncestorOf(workspace.sample_number)
    for index in range(workspace.settings_tabs.count()):
        workspace.settings_tabs.setCurrentIndex(index)
        QApplication.processEvents()
        scroll = workspace.settings_tabs.currentWidget()
        assert isinstance(scroll, QScrollArea)
        assert not scroll.horizontalScrollBar().isVisible()
        assert scroll.widget().width() <= scroll.viewport().width()
    assert not isinstance(workspace, QMainWindow)
