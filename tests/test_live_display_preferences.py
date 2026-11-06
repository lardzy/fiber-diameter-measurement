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
            assert canvas.view_zoom() * canvas.devicePixelRatioF() == pytest.approx(1.37)
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


@pytest.mark.parametrize("dpr", [1, 1.25, 1.5, 2])
@pytest.mark.parametrize("size", [(160, 120), (1600, 1200), (1600, 120), (160, 1200)])
def test_nine_anchors_cover_small_large_and_mixed_axes(tmp_path, trial_factory, dpr, size):
    from fdm.ui.acquisition_alignment import ALIGNMENT_FACTORS
    store = AcquisitionDisplayStore(tmp_path / "alignment.json")
    page = trial_factory(store, size=size, dpr=dpr)
    before_revision = page.session.revision
    for mode, value in (("pixel", None), ("fit", None), ("percent", 125), ("magnification", 300)):
        page.binding.set_mode(mode, value)
        settle()
        for alignment, factors in ALIGNMENT_FACTORS.items():
            page.binding.set_alignment(alignment)
            settle()
            scale, offset = page.canvas._transform()
            bars = (page.scroll.horizontalScrollBar(), page.scroll.verticalScrollBar())
            viewport = page.scroll.viewport()
            # Content in viewport coordinates must meet the selected edge or
            # center, regardless of which axes actually need scrollbars.
            observed = tuple(start - bar.value() for start, bar in zip(offset, bars))
            expected = tuple((view - extent * scale) * factor for view, extent, factor
                             in zip((viewport.width(), viewport.height()), size, factors))
            assert observed == pytest.approx(expected, abs=1 / dpr + 1)
            assert all(abs(start * dpr - round(start * dpr)) < 1e-6 for start in observed)
            assert store.preferences.alignment == alignment
            assert scale * dpr == pytest.approx(1 if mode == "pixel" else 1.25 if mode == "percent"
                                               else .6 if mode == "magnification" else scale * dpr)
    assert page.session.revision == before_revision
    assert page.host.project.extension_state_id == 0


def test_alignment_reload_legacy_shared_and_hidden_pages(tmp_path, trial_factory):
    path = tmp_path / "alignment.json"
    path.write_text('{"version":1,"display":{"mode":"percent","percent":125}}')
    store = AcquisitionDisplayStore(path)
    assert store.preferences.alignment == "free"
    first = trial_factory(store, size=(160, 120))
    second = trial_factory(store, size=(160, 120))
    second.host.hide()
    first.binding.set_alignment("bottom_right")
    settle()
    assert second.canvas.display_alignment == "free"
    second.host.resize(800, 620)
    second.host.show()
    settle()
    assert second.canvas.display_alignment == "bottom_right"
    restored = AcquisitionDisplayStore(path)
    assert restored.preferences.alignment == "bottom_right"
    assert restored.preferences.mode == "percent"
    # Same geometry video updates must not reset scroll or display generation.
    revision = first.canvas.display_revision
    for _ in range(3):
        first.canvas.set_frame(first.frame)
        first.binding.set_source(first.frame.source_id, first.frame.size)
        settle()
    assert first.canvas.display_revision == revision
    first.host.resize(920, 700)
    settle()
    scale, offset = first.canvas._transform()
    assert offset == pytest.approx((first.canvas.width() - 160 * scale,
                                    first.canvas.height() - 120 * scale), abs=1)


def test_only_real_scroll_releases_anchor_and_saves_position(tmp_path, trial_factory):
    from PySide6.QtWidgets import QAbstractSlider
    store = AcquisitionDisplayStore(tmp_path / "alignment.json")
    page = trial_factory(store)
    page.binding.set_alignment("top_left")
    settle()
    bar = page.scroll.horizontalScrollBar()
    # A Qt layout/restore update must not be mistaken for an input gesture.
    bar.setValue(50)
    settle()
    assert store.preferences.alignment == "top_left"
    assert store.center("camera:test", page.frame.size) == (.5, .5)
    page.binding.set_alignment("top_left")
    settle()
    prior_revision = page.canvas.display_revision
    bar.triggerAction(QAbstractSlider.SliderAction.SliderSingleStepAdd)
    value = bar.value()
    settle()
    assert value > 0 and bar.value() == value
    assert store.preferences.alignment == "free"
    assert page.canvas.display_alignment == "free"
    assert page.canvas.display_revision > prior_revision
    page.binding.flush()
    assert store.center("camera:test", page.frame.size) != (.5, .5)
    assert AcquisitionDisplayStore(store.path).preferences.alignment == "free"


@pytest.mark.parametrize("busy", ["_confirmation", "_jobs"])
def test_busy_alignment_and_scroll_are_rejected_without_mutation(tmp_path, trial_factory, busy):
    from PySide6.QtWidgets import QAbstractSlider
    store = AcquisitionDisplayStore(tmp_path / "alignment.json")
    page = trial_factory(store)
    page.binding.set_alignment("top_left")
    settle()
    old = getattr(page.controller, busy)
    setattr(page.controller, busy, object())
    try:
        with pytest.raises(ValueError):
            page.binding.set_alignment("bottom_right")
        bar = page.scroll.horizontalScrollBar()
        before = bar.value()
        bar.triggerAction(QAbstractSlider.SliderAction.SliderSingleStepAdd)
        settle()
        assert bar.value() == before
        assert store.preferences.alignment == "top_left"
        assert page.canvas.display_alignment == "top_left"
    finally:
        setattr(page.controller, busy, old)


def test_alignment_cancels_draft_and_changes_presented_generation(tmp_path, trial_factory):
    page = trial_factory(AcquisitionDisplayStore(tmp_path / "alignment.json"))
    page.controller.start((40, 40))
    revision = page.canvas.display_revision
    page.binding.set_alignment("bottom_right")
    settle()
    assert page.controller.draft is None
    assert page.canvas.display_revision > revision
    assert page.controller.presented.display_revision == page.canvas.display_revision
    point = page.controller.presented.image_point(*page.canvas._transform()[1])
    assert point == pytest.approx((0, 0))


@pytest.mark.parametrize("dpr", [1, 1.25, 1.5, 2])
def test_preview_anchor_zoom_resize_pan_and_same_frame(tmp_path, dpr):
    from fdm.geometry import Point
    from fdm.ui.acquisition_alignment import ALIGNMENT_FACTORS
    from PySide6.QtCore import QPointF
    store = AcquisitionDisplayStore(tmp_path / "alignment.json")
    canvas = DocumentCanvas()
    canvas.devicePixelRatioF = lambda: dpr
    binding = AcquisitionDisplayBinding(canvas, store=store)
    image = QImage(1000, 800, QImage.Format.Format_RGB32)
    image.fill(Qt.GlobalColor.black)
    try:
        canvas.resize(640, 480)
        canvas.show()
        with binding.mounting_frame():
            canvas.set_document(ImageDocument(id="preview", path="preview.png", image_size=(1000, 800)), image)
        binding.set_source("camera:a", (1000, 800))
        settle()
        for alignment, factors in ALIGNMENT_FACTORS.items():
            binding.set_alignment(alignment)
            binding.set_mode("percent", 45)
            settle()
            origin = canvas.image_to_widget(Point(0, 0))
            expected = ((640 - 450 / dpr) * factors[0], (480 - 360 / dpr) * factors[1])
            assert (origin.x(), origin.y()) == pytest.approx(expected, abs=1 / dpr)
        binding.set_mode("pixel")
        canvas.resize(800, 600)
        settle()
        assert canvas.view_zoom() * dpr == pytest.approx(1)
        assert canvas._pan == Point(round((800 - 1000 / dpr) * dpr) / dpr,
                                    round((600 - 800 / dpr) * dpr) / dpr)
        # No movement or a subpixel move below the pixel phase does not release.
        canvas._last_mouse_pos = QPointF(100, 100)
        canvas._begin_canvas_pan(Qt.MouseButton.MiddleButton)
        canvas._apply_pan_position(QPointF(100, 100))
        assert store.preferences.alignment == "bottom_right"
        canvas._apply_pan_position(QPointF(120, 130))
        canvas._end_canvas_pan()
        pan = canvas._pan
        settle()
        assert store.preferences.alignment == "free"
        assert canvas._pan == pan
        binding.set_source("camera:a", (1000, 800))
        settle()
        assert canvas._pan == pan
    finally:
        binding.close()
        canvas.close()
        canvas.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


@pytest.mark.parametrize("size", [(32, 24), (131072, 24)])
def test_drag_from_fit_keeps_actual_scale_and_center_outside_entry_range(tmp_path, monkeypatch, size):
    from PySide6.QtCore import QPoint, QPointF
    from PySide6.QtTest import QTest
    from fdm.ui.view_transform import CanvasZoomMode

    path = tmp_path / "preview.json"
    image = QImage(*size, QImage.Format.Format_Grayscale8)
    image.fill(20)
    canvases = []

    def mount(view_size):
        store = AcquisitionDisplayStore(path)
        canvas = DocumentCanvas()
        canvas.devicePixelRatioF = lambda: 1.25
        binding = AcquisitionDisplayBinding(canvas, store=store)
        canvases.append((canvas, binding))
        canvas.resize(*view_size)
        canvas.show()
        with binding.mounting_frame():
            canvas.set_document(ImageDocument(id="preview", path="p.png", image_size=size), image)
        binding.set_source("camera:roi", size)
        settle()
        return canvas, binding, store

    def image_center(canvas):
        point = canvas.widget_to_image(QPointF(canvas.width() / 2, canvas.height() / 2))
        return point.x, point.y

    def drag(canvas):
        QTest.mousePress(canvas, Qt.MouseButton.MiddleButton, pos=QPoint(80, 80))
        QTest.mouseMove(canvas, QPoint(110, 130))
        QTest.mouseRelease(canvas, Qt.MouseButton.MiddleButton, pos=QPoint(110, 130))
        settle()

    def fail_save(*_args):
        raise OSError("test disk full")

    try:
        canvas, binding, store = mount((640, 480))
        binding.set_mode("fit")
        binding.set_alignment("top_left")
        settle()
        initial_zoom = canvas.view_zoom()
        canvas.resize(720, 550)
        settle()
        zoom = canvas.view_zoom()
        assert zoom > initial_zoom
        assert canvas.zoom_mode() is CanvasZoomMode.FIT
        percent = zoom * canvas.devicePixelRatioF() * 100
        assert percent > 1600 or 0 < percent < 1
        prior = store.preferences
        prior_center = image_center(canvas)
        with monkeypatch.context() as patch:
            patch.setattr(store, "_save", fail_save)
            drag(canvas)
            assert store.preferences == prior
            assert canvas.zoom_mode() is CanvasZoomMode.FIT
            assert image_center(canvas) == pytest.approx(prior_center)
        drag(canvas)
        center = image_center(canvas)
        assert center != prior_center
        assert store.preferences.alignment == "free"
        assert store.preferences.mode == "percent"
        assert store.preferences.percent == pytest.approx(percent)
        assert canvas.view_zoom() == pytest.approx(zoom)
        canvas.resize(760, 565)
        settle()
        assert image_center(canvas) == pytest.approx(center)
        assert canvas.view_zoom() == pytest.approx(zoom)
        binding.close()
        canvas.hide()
        restored, new_binding, new_store = mount((760, 565))
        assert new_store.preferences == store.preferences
        assert image_center(restored) == pytest.approx(center)
        assert restored.view_zoom() == pytest.approx(zoom)
        # Selecting Fit explicitly uses the new viewport again.
        new_binding.set_mode("fit")
        settle()
        assert restored.zoom_mode() is CanvasZoomMode.FIT
        assert restored.view_zoom() > zoom
    finally:
        for canvas, binding in canvases:
            binding.close()
            canvas.close()
            canvas.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_real_trial_controller_reentry_does_not_snap_user_scroll_back(tmp_path, trial_factory):
    from PySide6.QtWidgets import QAbstractSlider
    store = AcquisitionDisplayStore(tmp_path / "alignment.json")
    page = trial_factory(store)
    page.controller.changed.connect(page.binding.apply)
    page.binding.set_alignment("bottom_right")
    settle()
    bar = page.scroll.horizontalScrollBar()
    maximum = bar.maximum()
    bar.triggerAction(QAbstractSlider.SliderAction.SliderPageStepSub)
    value = bar.value()
    settle()
    assert value < maximum and bar.value() == value
    assert store.preferences.alignment == "free"
    assert page.canvas.display_alignment == "free"


@pytest.mark.parametrize("size,axis", [((1600, 120), 0), ((160, 1200), 1)])
@pytest.mark.parametrize("alignment", ["top_left", "bottom_right"])
def test_release_keeps_small_axis_position_across_apply_and_restart(tmp_path, trial_factory, size, axis, alignment):
    from PySide6.QtWidgets import QAbstractSlider
    store = AcquisitionDisplayStore(tmp_path / "alignment.json")
    page = trial_factory(store, size=size)
    page.binding.set_alignment(alignment)
    settle()
    before = page.canvas._transform()[1][1 - axis]
    bar = (page.scroll.horizontalScrollBar(), page.scroll.verticalScrollBar())[axis]
    action = (QAbstractSlider.SliderAction.SliderSingleStepSub if alignment == "bottom_right"
              else QAbstractSlider.SliderAction.SliderSingleStepAdd)
    bar.triggerAction(action)
    settle()
    assert store.preferences.alignment == "free"
    page.binding.flush()
    center = store.center("camera:test", size)
    assert center[1 - axis] < 0 or center[1 - axis] > 1
    page.binding.apply()
    settle()
    assert page.canvas._transform()[1][1 - axis] == pytest.approx(before, abs=1)
    reloaded = AcquisitionDisplayStore(store.path)
    assert reloaded.center("camera:test", size) == center
    other = trial_factory(reloaded, size=size)
    assert other.canvas._transform()[1][1 - axis] == pytest.approx(before, abs=1)


def test_scroll_disk_failure_keeps_confirmed_anchor(tmp_path, trial_factory, monkeypatch):
    from PySide6.QtWidgets import QAbstractSlider
    store = AcquisitionDisplayStore(tmp_path / "alignment.json")
    page = trial_factory(store)
    page.controller.changed.connect(page.binding.apply)
    page.binding.set_alignment("bottom_right")
    settle()
    def fail(*_args, **_kwargs):
        raise OSError("test disk full")
    monkeypatch.setattr(store, "_save", fail)
    bar = page.scroll.horizontalScrollBar()
    bar.triggerAction(QAbstractSlider.SliderAction.SliderPageStepSub)
    settle()
    assert store.preferences.alignment == page.canvas.display_alignment == "bottom_right"
    assert bar.value() == bar.maximum()
    assert any("test disk full" in message for message in page.notices)


def test_invalid_large_viewport_centers_are_ignored_on_load(tmp_path):
    from fdm.ui.display_preferences import viewport_key
    path = tmp_path / "display.json"
    key = viewport_key("camera:test", (100, 100))
    path.write_text(json.dumps({"version": 1, "center_coordinates": "viewport",
        "centers": {key: [1e300, -1e300]}}))
    store = AcquisitionDisplayStore(path)
    assert store.center("camera:test", (100, 100)) == (.5, .5)
    with pytest.raises(ValueError):
        store.set_center("camera:test", (100, 100), (1e300, 0), allow_outside_image=True)


def test_alignment_survives_source_geometry_and_dpr_change(tmp_path, trial_factory):
    store = AcquisitionDisplayStore(tmp_path / "alignment.json")
    page = trial_factory(store)
    page.binding.set_alignment("right")
    settle()
    frame = FrameHandle.snapshot(np.zeros((180, 220, 3), np.uint8), source_id="camera:other", live=False)
    page.controller.receive_frame(frame)
    page.canvas.set_frame(frame)
    page.binding.set_source(frame.source_id, frame.size)
    page.canvas.devicePixelRatioF = lambda: 1.5
    QApplication.sendEvent(page.canvas, QEvent(QEvent.Type.DevicePixelRatioChange))
    settle()
    scale, offset = page.canvas._transform()
    assert scale * 1.5 == pytest.approx(1)
    assert offset == pytest.approx((page.canvas.width() - 220 * scale,
                                   (page.canvas.height() - 180 * scale) / 2), abs=1 / 1.5)
    assert store.preferences.alignment == "right"
    assert store.center("camera:other", frame.size) == (.5, .5)


def test_drag_position_writes_are_debounced(tmp_path, trial_factory, monkeypatch):
    from PySide6.QtWidgets import QAbstractSlider
    store = AcquisitionDisplayStore(tmp_path / "alignment.json")
    page = trial_factory(store)
    page.controller.changed.connect(page.binding.apply)
    page.binding.set_alignment("top_left")
    settle()
    writes = []
    original = store._save
    def save(*args):
        writes.append(args)
        return original(*args)
    monkeypatch.setattr(store, "_save", save)
    bar = page.scroll.horizontalScrollBar()
    for _ in range(10):
        bar.triggerAction(QAbstractSlider.SliderAction.SliderSingleStepAdd)
    assert len(writes) == 1  # free preference, not ten viewport snapshots
    page.binding.flush()
    assert len(writes) == 2
