import time
from dataclasses import replace

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication, QMainWindow
from PySide6.QtGui import QAction, QImage

from fdm.live_experiment import ExperimentSession, default_materials
from fdm.live_experiment.controller import CalibrationSnapshot, LiveExperimentController
from fdm.live_experiment.frames import FrameBudget, FrameBudgetExceeded, FrameHandle, PresentedFrameToken
from fdm.live_experiment.motion import FieldMotionDetector, sobel_fraction
from fdm.models import Calibration, ImageDocument, ProjectState
from fdm.project_io import ProjectIO


def session():
    value = ExperimentSession(save_images=False)
    value.add_operator("A")
    value.set_materials(default_materials())
    return value


def frame(seq=0, live=False, pixels=None):
    if pixels is None:
        pixels = np.full((120, 160, 3), 200, np.uint8)
        pixels[:, 70:90] = 30
    return FrameHandle.snapshot(pixels, source_id="test", sequence=seq, live=live)


def present(controller, value):
    controller.receive_frame(value)
    controller.mark_presented(PresentedFrameToken(value, controller.field_epoch, 0, 1.0, (0, 0)))


def drain(controller, timeout=2):
    end = time.perf_counter() + timeout
    while controller._jobs and time.perf_counter() < end:
        QApplication.processEvents()
        controller.poll()
        time.sleep(.002)
    assert not controller._jobs


def test_snapshot_owns_pixels_and_budget_shares_identity():
    source = np.zeros((40, 50), np.uint8)
    value = frame(pixels=source)
    source[:] = 10
    assert not value.pixels.any()
    assert not value.pixels.flags.writeable
    budget = FrameBudget(value.nbytes)
    first, second = budget.acquire(value), budget.acquire(value)
    assert budget.used_bytes == value.nbytes
    with pytest.raises(FrameBudgetExceeded):
        budget.acquire(frame(1, pixels=source))
    budget.release(first)
    assert budget.used_bytes == value.nbytes
    budget.release(second)
    assert budget.used_bytes == 0


def test_free_measurement_uses_first_frame_and_anisotropic_scale(tmp_path):
    controller = LiveExperimentController(session(), evidence_directory=tmp_path)
    try:
        controller.set_calibration(CalibrationSnapshot(2, 3))
        present(controller, frame())
        controller.start((10, 10))
        present(controller, frame(1))
        controller.finish((13, 14), "wool", method="free")
        drain(controller)
        record = next(iter(controller.session.measurements.values()))
        assert record.diameter_um == pytest.approx(np.hypot(6, 12))
        assert record.frame_id == "test:0:0"
    finally:
        controller.close()


def test_movement_cancels_computing_result_but_preserves_accepted_history(tmp_path):
    controller = LiveExperimentController(session(), evidence_directory=tmp_path)
    try:
        controller.set_calibration(CalibrationSnapshot(1, 1))
        present(controller, frame())
        controller.start((10, 10)); controller.finish((10, 30), "wool", method="free")
        drain(controller)
        controller.start((20, 10)); controller.finish((20, 40), "wool", method="free")
        controller.end_field()
        drain(controller)
        assert len(controller.session.measurements) == 1
        assert not controller.current_records
        assert controller.budget.used_bytes == controller.latest_frame.nbytes
    finally:
        controller.close()


def test_pending_classification_invalidated_on_move(tmp_path):
    value = session()
    value.set_materials([replace(default_materials()[0], upper_limit=15)])
    controller = LiveExperimentController(value, evidence_directory=tmp_path)
    try:
        controller.set_calibration(CalibrationSnapshot(1, 1))
        present(controller, frame())
        controller.start((10, 10)); controller.finish((10, 30), "wool", method="free")
        drain(controller)
        assert len(controller.pending_classification) == 1
        controller.end_field()
        controller.resolve_pending()
        assert not value.measurements
        assert controller.budget.used_bytes == controller.latest_frame.nbytes
    finally:
        controller.close()


def test_second_click_uses_actual_event_point(tmp_path):
    controller = LiveExperimentController(session(), evidence_directory=tmp_path)
    try:
        controller.set_calibration(CalibrationSnapshot(1, 1))
        present(controller, frame())
        controller.start((10, 10)); controller.finish((10, 33), "wool", method="free")
        drain(controller)
        assert next(iter(controller.session.measurements.values())).diameter_um == 23
    finally:
        controller.close()


def test_source_size_change_clears_calibration_and_draft(tmp_path):
    controller = LiveExperimentController(session(), evidence_directory=tmp_path)
    try:
        controller.set_calibration(CalibrationSnapshot(1, 1))
        present(controller, frame())
        controller.start((10, 10))
        controller.receive_frame(frame(1, pixels=np.zeros((80, 80), np.uint8)))
        assert controller.calibration is None
        assert controller.draft is None
        assert controller.presented is None
    finally:
        controller.close()


def test_unsigned_sobel_does_not_lose_strong_changes():
    a = np.zeros((96, 128), np.uint8)
    b = np.tile(((np.arange(128) // 2) % 2 * 255).astype(np.uint8), (96, 1))
    assert sobel_fraction(a, b) > .04
    assert sobel_fraction(a, b, legacy_signed=True) == 0


def test_motion_static_and_clear_translation():
    detector = FieldMotionDetector()
    first = frame(live=True)
    assert detector.compare(first).stable
    assert detector.compare(first).stable
    shifted = frame(1, True, np.roll(first.pixels, 25, axis=1))
    assert not detector.compare(shifted).stable


def test_project_contains_typed_experiment_and_roundtrips(tmp_path):
    project = ProjectState.empty()
    project.live_experiments = [session()]
    project.live_experiments[0].add_measurement(12, "wool")
    path = tmp_path / "sample.fdm.json"
    ProjectIO.save(project, path)
    loaded = ProjectIO.load(path)
    assert "live-experiments/v1" in loaded.effective_required_features()
    assert loaded.live_experiments[0].to_dict() == project.live_experiments[0].to_dict()


def test_workspace_advances_frame_while_draft_is_held(tmp_path):
    from fdm.ui.live_experiment_workspace import open_live_experiment
    class Host(QMainWindow):
        def current_document(self):
            return self.project.documents[0]
    host = Host()
    host.project = ProjectState.empty()
    host.project.documents.append(ImageDocument(id="a", path="a.png", image_size=(160, 120),
        calibration=Calibration("manual", 1, "um", "test")))
    image = QImage(160, 120, QImage.Format.Format_RGB32); image.fill(0xffffffff)
    host._images = {"a": image}
    host.save_project_action = QAction(host)
    workspace = open_live_experiment(host)
    try:
        QApplication.processEvents()
        controller = workspace.controller
        first = controller.presented.frame
        controller.start((10, 10))
        later = FrameHandle.snapshot(first.pixels, source_id=first.source_id,
             generation=first.generation, sequence=1, live=False)
        workspace._receive_frame(later)
        QApplication.processEvents()
        assert controller.presented.frame is later
        assert controller.draft[0].frame is first
        controller.cancel_draft()
    finally:
        workspace.close(); host.close(); QApplication.processEvents()
