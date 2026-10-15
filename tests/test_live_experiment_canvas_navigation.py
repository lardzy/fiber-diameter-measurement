import numpy as np
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from fdm.live_experiment import ExperimentSession, default_materials
from fdm.live_experiment.controller import CalibrationSnapshot, LiveExperimentController
from fdm.live_experiment.frames import FrameHandle
from fdm.ui.live_experiment_workspace import LiveMeasurementCanvas


def test_pan_zoom_cancel_draft_preserve_field_and_publish_painted_transform(tmp_path):
    session = ExperimentSession(save_images=False)
    session.add_operator("test")
    session.set_materials(default_materials())
    controller = LiveExperimentController(session, evidence_directory=tmp_path)
    canvas = LiveMeasurementCanvas(controller)
    canvas.gestureCancelled.connect(controller.cancel_draft)
    try:
        canvas.resize(640, 480)
        canvas.show()
        frame = FrameHandle.snapshot(np.zeros((240, 320, 3), np.uint8), source_id="image", live=False)
        controller.receive_frame(frame)
        controller.set_calibration(CalibrationSnapshot(1, 1))
        canvas.set_frame(frame)
        QApplication.processEvents()
        epoch = controller.field_epoch
        original = controller.presented
        controller.start((40, 40))
        QTest.mousePress(canvas, Qt.MouseButton.MiddleButton, pos=QPoint(120, 120))
        QTest.mouseMove(canvas, QPoint(170, 150))
        QTest.mouseRelease(canvas, Qt.MouseButton.MiddleButton, pos=QPoint(170, 150))
        QApplication.processEvents()
        assert controller.draft is None
        assert controller.field_epoch == epoch
        assert controller.presented.frame is frame
        assert controller.presented.offset == (original.offset[0] + 50, original.offset[1] + 30)
        location = QPointF(250, 180)
        anchor = controller.presented.image_point(location.x(), location.y())
        controller.start((40, 40))
        wheel = QWheelEvent(location, location, QPoint(), QPoint(0, 120),
            Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase, False)
        QApplication.sendEvent(canvas, wheel)
        QApplication.processEvents()
        assert controller.draft is None
        assert np.allclose(controller.presented.image_point(location.x(), location.y()), anchor)
        assert controller.field_epoch == epoch
        canvas.fit_view()
        QApplication.processEvents()
        assert controller.presented.scale == original.scale
        assert controller.presented.offset == original.offset
    finally:
        canvas.close()
        controller.close()
