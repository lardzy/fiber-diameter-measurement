"""Display modes are explicit; incidental mouse gestures cannot rescale video."""
import numpy as np
import pytest
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from fdm.live_experiment import ExperimentSession, default_materials
from fdm.live_experiment.controller import CalibrationSnapshot, LiveExperimentController
from fdm.live_experiment.frames import FrameHandle
from fdm.ui.live_experiment_workspace import LiveMeasurementCanvas


@pytest.fixture
def canvas(tmp_path):
    session = ExperimentSession(save_images=False)
    session.add_operator("test")
    session.set_materials(default_materials())
    controller = LiveExperimentController(session, evidence_directory=tmp_path)
    view = LiveMeasurementCanvas(controller)
    view.gestureCancelled.connect(controller.cancel_draft)
    view.resize(640, 480)
    view.show()
    frame = FrameHandle.snapshot(np.zeros((240, 320, 3), np.uint8), source_id="image", live=False)
    controller.receive_frame(frame)
    controller.set_calibration(CalibrationSnapshot(.5, .5))
    view.set_frame(frame)
    QApplication.processEvents()
    yield view
    view.close()
    controller.close()


@pytest.mark.parametrize("dpr", [1, 1.25, 1.5, 2])
def test_actual_pixels_ignore_window_size_and_dpi(canvas, monkeypatch, dpr):
    monkeypatch.setattr(canvas, "devicePixelRatioF", lambda: dpr)
    for width, height in ((640, 480), (900, 700)):
        canvas.resize(width, height)
        scale, offset = canvas._transform()
        assert scale * dpr == 1
        assert np.allclose(np.array(offset) * dpr, np.round(np.array(offset) * dpr))
    assert canvas.toolTip() == ""


def test_wheel_and_middle_drag_do_not_change_display_or_draft(canvas):
    controller = canvas.controller
    original = controller.presented
    controller.start((40, 40))
    QTest.mousePress(canvas, Qt.MouseButton.MiddleButton, pos=QPoint(120, 120))
    QTest.mouseMove(canvas, QPoint(170, 150))
    QTest.mouseRelease(canvas, Qt.MouseButton.MiddleButton, pos=QPoint(170, 150))
    for angle, pixels in ((QPoint(0, 120), QPoint()), (QPoint(), QPoint(0, -15))):
        event = QWheelEvent(QPointF(250, 180), QPointF(250, 180), pixels, angle,
            Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.ScrollUpdate, False)
        QApplication.sendEvent(canvas, event)
    QApplication.processEvents()
    assert controller.draft is not None
    assert (controller.presented.scale, controller.presented.offset) == (original.scale, original.offset)


def test_cu_modes_cancel_draft_and_preserve_measurement_geometry(canvas):
    controller = canvas.controller
    epoch = controller.field_epoch
    controller.start((40, 40))
    canvas.set_display_mode("fit")
    QApplication.processEvents()
    assert controller.draft is None and controller.field_epoch == epoch
    assert canvas._transform()[0] == 2
    canvas.set_display_mode("percent", 150)
    assert canvas._transform()[0] * canvas.devicePixelRatioF() == 1.5
    canvas.set_display_mode("magnification", 200, screen_pixel_mm=.25)
    assert canvas._transform()[0] * canvas.devicePixelRatioF() == .4
    assert controller.calibration.x_um == .5


@pytest.mark.parametrize("mode", ["pixel", "fit", "percent", "magnification"])
def test_nonfreeze_display_switch_preserves_accepted_records_and_cancels_draft(canvas, mode):
    controller = canvas.controller
    record = controller.session.add_measurement(20, "wool", field_id=controller.field_id,
        endpoints=((20, 20), (20, 40)))
    controller.current_records.append(record.id)
    revision = controller.session.revision
    controller.start((40, 40))
    canvas.set_display_mode(mode, 100, .25)
    assert controller.draft is None
    assert controller.current_records == [record.id]
    assert controller.session.measurements[record.id].diameter_um == 20
    assert controller.session.measurements[record.id].endpoints == ((20, 20), (20, 40))
    assert controller.session.revision == revision


def test_oversize_pixels_have_extent_instead_of_silent_fit(canvas):
    frame = FrameHandle.snapshot(np.zeros((576, 768, 3), np.uint8), source_id="image", live=False)
    canvas.set_frame(frame)
    assert canvas.minimumWidth() >= 768 / canvas.devicePixelRatioF()
    assert canvas.minimumHeight() >= 576 / canvas.devicePixelRatioF()
    assert canvas._transform()[0] * canvas.devicePixelRatioF() == 1


def test_invalid_physical_magnification_does_not_change_display(canvas):
    with pytest.raises(ValueError, match="显示范围过大"):
        canvas.set_display_mode("magnification", 100000, screen_pixel_mm=.0000001)
    assert canvas.display_mode == "pixel"
    assert canvas._transform()[0] * canvas.devicePixelRatioF() == 1
