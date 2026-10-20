"""Acquisition view choices survive page, device and application lifetimes."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtCore import QCoreApplication, QEvent, QSize, Qt
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication, QScrollArea, QVBoxLayout, QWidget

from fdm.live_experiment import ExperimentSession, default_materials
from fdm.live_experiment.controller import CalibrationSnapshot, LiveExperimentController
from fdm.live_experiment.frames import FrameHandle
from fdm.models import ImageDocument, ProjectState
from fdm.ui.canvas import DocumentCanvas
from fdm.ui.display_preferences import (
    AcquisitionDisplayBinding, AcquisitionDisplayPreferences, AcquisitionDisplayStore,
    acquisition_display_store, resolve_display,
)
from fdm.ui.live_display import _screen_key, save_screen_ruler, screen_pixel_mm
from fdm.ui.live_experiment_workspace import LiveMeasurementCanvas


def settle():
    for _ in range(5):
        QApplication.processEvents()


@pytest.fixture
def trial_factory(tmp_path):
    created = []

    def create(store, *, source="camera:test", size=(1600, 1200), calibration=True, dpr=1):
        session = ExperimentSession(save_images=False)
        session.add_operator("test")
        session.set_materials(default_materials())
        session.establish_history_baseline()
        controller = LiveExperimentController(session, evidence_directory=tmp_path)
        host = QWidget()
        host.project = ProjectState.empty()
        layout = QVBoxLayout(host)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        canvas = LiveMeasurementCanvas(controller)
        canvas.devicePixelRatioF = lambda: dpr
        canvas.gestureCancelled.connect(controller.cancel_draft)
        scroll.setWidget(canvas)
        layout.addWidget(scroll)
        notices = []
        binding = AcquisitionDisplayBinding(canvas, store=store, scroll_area=scroll,
            calibration_provider=lambda: controller.calibration,
            screen_pixel_mm_provider=lambda: .25, status_callback=notices.append, parent=host)
        host.resize(650, 490)
        host.show()
        frame = FrameHandle.snapshot(np.zeros((size[1], size[0], 3), np.uint8),
            source_id=source, live=False)
        controller.receive_frame(frame)
        if calibration:
            controller.set_calibration(CalibrationSnapshot(.5, .5))
        canvas.set_frame(frame)
        binding.set_source(source, size)
        settle()
        result = SimpleNamespace(host=host, canvas=canvas, scroll=scroll, controller=controller,
            binding=binding, notices=notices, frame=frame, session=session)
        created.append(result)
        return result

    yield create
    for item in created:
        item.binding.close()
        item.host.close()
        item.controller.close()
        item.host.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    settle()


def test_mode_values_and_device_geometry_centers_survive_reload(tmp_path):
    path = tmp_path / "display.json"
    store = AcquisitionDisplayStore(path)
    store.set_mode("percent", 137.5)
    store.set_mode("magnification", 320)
    store.set_mode("fit")
    store.set_center("camera:a", (1600, 1200), (.25, .7))
    store.set_center("camera:a", (1920, 1080), (.8, .3))
    store.set_center("camera:b", (1600, 1200), (-1, 2))
    restored = AcquisitionDisplayStore(path)
    assert restored.preferences == AcquisitionDisplayPreferences("fit", 137.5, 320)
    assert restored.center("camera:a", (1600, 1200)) == (.25, .7)
    assert restored.center("camera:a", (1920, 1080)) == (.8, .3)
    assert restored.center("camera:b", (1600, 1200)) == (0, 1)
    assert restored.center("camera:c", (1600, 1200)) == (.5, .5)
    restored.set_mode("percent")
    assert restored.preferences.percent == 137.5
    restored.set_mode("magnification")
    assert restored.preferences.magnification == 320


@pytest.mark.parametrize("mode,value", [("pixel", None), ("fit", None),
                                         ("percent", 137), ("magnification", 300)])
def test_real_trial_canvas_rebuild_and_restart_restore_four_modes_without_dirty(
        tmp_path, trial_factory, mode, value):
    path = tmp_path / "display.json"
    store = AcquisitionDisplayStore(path)
    first = trial_factory(store)
    revision, extension = first.session.revision, first.host.project.extension_state_id
    first.binding.set_mode(mode, value)
    settle()
    first.binding.flush()
    assert first.session.revision == revision
    assert first.host.project.extension_state_id == extension
    first.binding.close()
    first.host.hide()
    second = trial_factory(store)
    third = trial_factory(AcquisitionDisplayStore(path))
    for page in (second, third):
        assert page.canvas.display_mode == mode
        if value is not None:
            assert page.canvas.display_value == value
        assert page.host.project.extension_state_id == 0
        assert not page.session.measurements


def test_shared_store_updates_visible_pages_without_recreating_them(tmp_path, trial_factory):
    store = AcquisitionDisplayStore(tmp_path / "display.json")
    first, second = trial_factory(store), trial_factory(store)
    first.binding.set_mode("percent", 175)
    settle()
    assert second.canvas.display_mode == "percent"
    assert second.canvas.display_value == 175
    first.binding.set_mode("fit")
    settle()
    assert second.canvas.display_mode == "fit"
    assert store.preferences.percent == 175


def test_controller_change_during_display_apply_does_not_reenter(tmp_path, trial_factory, monkeypatch):
    import sys
    errors = []
    monkeypatch.setattr(sys, "excepthook", lambda *args: errors.append(args))
    page = trial_factory(AcquisitionDisplayStore(tmp_path / "display.json"))
    page.controller.changed.connect(page.binding.apply)
    page.binding.set_mode("fit")
    settle()
    page.binding.set_mode("percent", 150)
    settle()
    assert page.canvas.display_mode == "percent"
    assert page.canvas.display_value == 150
    assert not errors


def test_scroll_center_survives_page_rebuild_and_fit_roundtrip(tmp_path, trial_factory):
    store = AcquisitionDisplayStore(tmp_path / "display.json")
    first = trial_factory(store)
    first.binding.set_mode("percent", 125)
    settle()
    first.scroll.horizontalScrollBar().setValue(420)
    first.scroll.verticalScrollBar().setValue(560)
    first.binding.flush()
    saved = store.center("camera:test", first.frame.size)
    first.binding.set_mode("fit")
    settle()
    assert store.center("camera:test", first.frame.size) == saved
    first.binding.set_mode("percent")
    settle()
    first.binding.flush()
    assert store.center("camera:test", first.frame.size) == pytest.approx(saved, abs=.002)
    first.binding.close()
    first.host.hide()
    second = trial_factory(AcquisitionDisplayStore(store.path))
    second.binding.flush()
    assert store.center("camera:test", first.frame.size) == pytest.approx(
        second.binding.store.center("camera:test", first.frame.size), abs=.002)


@pytest.mark.parametrize("dpr", [1, 1.25, 1.5, 2])
def test_trial_physical_pixels_and_percent_are_independent_of_ui_dpi(tmp_path, trial_factory, dpr):
    store = AcquisitionDisplayStore(tmp_path / "display.json")
    page = trial_factory(store, dpr=dpr)
    assert page.canvas._transform()[0] * dpr == 1
    page.binding.set_mode("percent", 150)
    settle()
    assert page.canvas._transform()[0] * dpr == 1.5
    page.binding.set_mode("magnification", 200)
    settle()
    assert page.canvas._transform()[0] * dpr == .4


def test_missing_calibration_falls_back_without_replacing_remembered_mode(tmp_path, trial_factory):
    store = AcquisitionDisplayStore(tmp_path / "display.json")
    store.set_mode("magnification", 300)
    page = trial_factory(store, calibration=False)
    assert page.canvas.display_mode == "pixel"
    assert store.preferences.mode == "magnification"
    assert any("显微镜标尺" in message for message in page.notices)
    page.controller.set_calibration(CalibrationSnapshot(.5, .5))
    page.binding.set_source(page.frame.source_id, page.frame.size)
    settle()
    assert page.canvas.display_mode == "magnification"
    assert page.canvas.display_value == 300


def test_pending_work_blocks_manual_change_without_persisting(tmp_path, trial_factory):
    store = AcquisitionDisplayStore(tmp_path / "display.json")
    page = trial_factory(store)
    page.controller._confirmation = object()
    try:
        with pytest.raises(ValueError):
            page.binding.set_mode("percent", 150)
        assert store.preferences.mode == "pixel"
        assert page.canvas.display_mode == "pixel"
    finally:
        page.controller._confirmation = None


def test_unaccepted_candidates_allow_view_changes(tmp_path, trial_factory):
    store = AcquisitionDisplayStore(tmp_path / "display.json")
    page = trial_factory(store)
    page.controller.candidates = [{"candidate_id": "candidate", "state": "available"}]
    page.binding.set_mode("percent", 150)
    assert store.preferences.mode == "percent"
    assert page.canvas.display_mode == "percent"
    assert page.canvas.display_value == 150


def test_document_preview_adapter_does_not_redefine_measurement_actual_size(tmp_path):
    store = AcquisitionDisplayStore(tmp_path / "display.json")
    preview, measurement = DocumentCanvas(), DocumentCanvas()
    image = QImage(1600, 1200, QImage.Format.Format_RGB32)
    image.fill(Qt.GlobalColor.black)
    document = ImageDocument(id="preview", path="preview.png", image_size=(1600, 1200))
    measured = ImageDocument(id="measurement", path="saved.png", image_size=(1600, 1200))
    preview.devicePixelRatioF = lambda: 2
    measurement.devicePixelRatioF = lambda: 2
    binding = AcquisitionDisplayBinding(preview, store=store)
    try:
        for canvas, item in ((preview, document), (measurement, measured)):
            canvas.resize(640, 480)
            canvas.set_document(item, image)
            canvas.show()
        measurement.actual_size()
        existing_view = measured.view_state.to_dict()
        binding.set_source("camera:test", (1600, 1200))
        settle()
        assert preview.view_zoom() == .5
        assert measurement.view_zoom() == 1
        binding.set_mode("percent", 150)
        settle()
        assert preview.view_zoom() == .75
        assert measured.view_state.to_dict() == existing_view
        assert not measured.dirty_flags.session_dirty
    finally:
        binding.close()
        preview.close()
        measurement.close()
        preview.deleteLater()
        measurement.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_preview_same_source_after_document_clear_reapplies_preferences(tmp_path):
    store = AcquisitionDisplayStore(tmp_path / "display.json")
    canvas = DocumentCanvas()
    binding = AcquisitionDisplayBinding(canvas, store=store)
    image = QImage(1600, 1200, QImage.Format.Format_RGB32)
    image.fill(Qt.GlobalColor.black)
    try:
        canvas.resize(640, 480)
        canvas.show()
        for index in range(2):
            with binding.mounting_frame():
                canvas.set_document(ImageDocument(id=str(index), path="preview.png",
                    image_size=(1600, 1200)), image)
            binding.set_source("camera:test", (1600, 1200))
            if index == 0:
                binding.set_mode("percent", 137)
            settle()
            assert canvas.view_zoom() == pytest.approx(1.37)
            if index == 0:
                binding.clear_source()
                canvas.clear_document()
                settle()
                assert binding._source is None and not binding._restore_timer.isActive()
        assert store.preferences.percent == 137
    finally:
        binding.close()
        canvas.close()
        canvas.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def fake_screen(name="test", size=(1920, 1080), dpr=1):
    return SimpleNamespace(name=lambda: name, manufacturer=lambda: "vendor", model=lambda: "panel",
        serialNumber=lambda: "serial", size=lambda: QSize(*size), devicePixelRatio=lambda: dpr)


def test_screen_ruler_legacy_compatibility_and_resolution_invalidation(tmp_path, monkeypatch):
    from fdm.ui import live_display
    path = tmp_path / "screen-rulers.json"
    monkeypatch.setattr(live_display, "_settings", lambda: path)
    screen = fake_screen()
    path.write_text(json.dumps({_screen_key(screen): 125}), encoding="utf-8")
    assert screen_pixel_mm(screen) == .25
    save_screen_ruler(screen, 125)
    assert screen_pixel_mm(screen) == .25
    # Same physical raster, different OS UI scaling retains the measurement.
    assert screen_pixel_mm(fake_screen(size=(1536, 864), dpr=1.25)) == .25
    assert screen_pixel_mm(fake_screen(size=(1280, 720))) is None
    assert screen_pixel_mm(fake_screen(name="other")) is None
    preference = AcquisitionDisplayPreferences("magnification", magnification=200)
    result = resolve_display(preference, calibration=CalibrationSnapshot(.5, .5), screen_pixel_mm=None)
    assert result.mode == "pixel" and "显示器" in result.message
    assert preference.mode == "magnification"


@pytest.mark.parametrize("payload", ["{broken", '{"version":2}',
    '{"version":1,"display":{"percent":"NaN"}}'])
def test_invalid_local_preferences_never_block_workspace(tmp_path, payload):
    path = tmp_path / "display.json"
    path.write_text(payload)
    store = AcquisitionDisplayStore(path)
    assert store.preferences == AcquisitionDisplayPreferences()
    assert store.load_warning


def test_store_is_shared_per_host_and_separate_from_project(tmp_path):
    host = QWidget()
    host.project = ProjectState.empty()
    first = acquisition_display_store(host)
    first.set_mode("fit")
    assert acquisition_display_store(host) is first
    assert host.project.extension_state_id == 0
    assert not host.project.live_experiments
    host.close()
    host.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
