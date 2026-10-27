from __future__ import annotations

import csv
from dataclasses import replace
import hashlib
from threading import Event
import time

import numpy as np
import pytest
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QImage, QKeyEvent, QMouseEvent, QPainter

from fdm.construction_geometry import ConstructionEntity, LineDefinition
from fdm.geometry import Point
from fdm.image_processing_models import DisplayTransform
from fdm.models import Calibration, ImageDocument, new_id
from fdm.raster import RasterPixelType, RasterPlane
from fdm.services.metrology_fitting import CaliperOptions
from fdm.services.raster_io import raster_plane_to_qimage
from fdm.services.digital_slide_store import DigitalSlideManifest, DigitalSlideStore, DigitalSlideTile
from fdm.services.export_service import ExportImageRenderMode, ExportRenderContext
from fdm.settings import AppSettings
from fdm.ui.main_window import MainWindow
from fdm.ui.professional_measurement import export_professional_results, raw_caliper_source


@pytest.fixture
def mounted_window(monkeypatch, desktop_application):
    monkeypatch.setattr("fdm.ui.main_window.AppSettingsIO.load", lambda: AppSettings(theme_mode="dark"))
    monkeypatch.setattr("fdm.ui.main_window.AppSettingsIO.save", lambda *_: None)
    windows = []

    def mount(array=None):
        if array is None:
            y, _x = np.mgrid[:192, :256]
            array = np.rint(255 * (0.5 + 0.5 * np.tanh((y - 90.3) / 0.8))).astype(np.uint8)
        kind = RasterPixelType.GRAY16 if array.dtype == np.uint16 else RasterPixelType.GRAY8
        if array.ndim == 3:
            kind = RasterPixelType.RGBA8 if array.shape[2] == 4 else RasterPixelType.RGB8
        plane = RasterPlane(array.shape[1], array.shape[0], kind, array.tobytes())
        window = MainWindow()
        windows.append(window)
        document = ImageDocument(new_id("image"), "/tmp/professional-test.tif", (plane.width, plane.height))
        document.initialize_runtime_state()
        document.mark_session_saved()
        document.mark_calibration_saved()
        window._mount_document(document, raster_plane_to_qimage(plane), tooltip=document.path, raster_plane=plane)
        window.resize(1100, 760)
        window.show()
        desktop_application.processEvents()
        workspace = window.professional_workspace
        workspace.show()
        desktop_application.processEvents()
        return window, document, workspace, plane

    yield mount
    for window in windows:
        window.professional_workspace.cancel()
        for thread in window.professional_workspace._workers:
            thread.join(timeout=1)
        window._reset_workspace()
        window.close()
    desktop_application.processEvents()


@pytest.fixture
def professional_slide(mounted_window, tmp_path, monkeypatch, desktop_application):
    from fdm.ui import digital_slide_canvas
    monkeypatch.setattr(digital_slide_canvas, "digital_slide_render_cache_directory", lambda: tmp_path / "cache")
    path = tmp_path / "professional-five-layer.fdmslide"
    store = DigitalSlideStore.create(path, DigitalSlideManifest(1, 480, 360, 160, 120, [-4, -2, 0, 2, 4]))
    try:
        y, _x = np.mgrid[:360, :480]
        for focus in range(5):
            gray = np.rint(20 + focus * 5 + 160 * (0.5 + 0.5 * np.tanh((y - 178 - focus) / 0.8))).astype(np.uint8)
            for row in range(3):
                for column in range(3):
                    pixels = gray[row * 120:(row + 1) * 120, column * 160:(column + 1) * 160].copy()
                    plane = RasterPlane(160, 120, RasterPixelType.GRAY8, pixels.tobytes())
                    store.write_tile(DigitalSlideTile(focus, column * 160, row * 120, 160, 120), raster_plane_to_qimage(plane))
    finally:
        store.close()
    source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    document = ImageDocument(new_id("slide"), str(path), (480, 360), document_kind="digital_slide",
                             metadata={"digital_slide": {"viewport_origin": [160, 120], "focus_index": 2}})
    document.initialize_runtime_state()
    window = MainWindow()
    try:
        window.resize(1100, 760)
        window.show()
        window._add_digital_slide_document_from_path(path, document=document)
        workspace = window.professional_workspace
        workspace.show()
        canvas = workspace.controller().canvas
        _wait_until(desktop_application, lambda: canvas.pixel_work_enabled() and not canvas._initial_fit_pending, timeout=5)
        canvas.grab()
        # Initial native layout can queue one further render after the first frame.
        _wait_until(desktop_application, canvas.pixel_work_enabled, timeout=5)
        yield window, document, workspace, path, source_hash
    finally:
        window.professional_workspace.cancel()
        for thread in window.professional_workspace._workers:
            thread.join(timeout=1)
        window._reset_workspace()
        window.close()
        desktop_application.processEvents()


def _mouse(kind, position):
    return QMouseEvent(kind, position, position, Qt.MouseButton.LeftButton,
                       Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)


def _click(canvas, point):
    position = canvas.image_to_widget(Point(*point))
    canvas.mousePressEvent(_mouse(QEvent.Type.MouseButtonPress, position))
    canvas.mouseReleaseEvent(_mouse(QEvent.Type.MouseButtonRelease, position))


def _begin(workspace, key):
    workspace.panel.tool.setCurrentIndex(workspace.panel.tool.findData(key))
    workspace.begin()


def _paint(controller):
    image = QImage(controller.canvas.size(), QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(Qt.GlobalColor.black)
    painter = QPainter(image)
    try:
        controller.paint(painter)
    finally:
        painter.end()
    return image


def _wait_until(app, predicate, timeout=3):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return
        time.sleep(0.005)
    assert predicate(), "后台拟合未在测试期限内完成"


def _distance(workspace, first=(30, 40), second=(70, 70)):
    _begin(workspace, "distance")
    canvas = workspace.controller().canvas
    _click(canvas, first)
    _click(canvas, second)
    return canvas._document.professional_measurements[-1]


def test_main_window_mount_and_real_canvas_distance_clicks(mounted_window):
    window, document, workspace, plane = mounted_window()
    controller = workspace.controller()
    assert controller.canvas is window.current_canvas()
    assert window._mode_actions["professional"].isEnabled()
    definition = _distance(workspace)
    result = controller.results()[definition.id]
    assert result.valid
    assert result.primary.value == pytest.approx(50)
    assert document.measurements == []
    assert document.dirty_flags.session_dirty
    assert window._rasters[document.id].sha256() == plane.sha256()
    _paint(controller)
    assert definition.id in controller.label_hits
    label = controller.label_hits[definition.id]
    assert len(label.witness_lines) == 2
    assert label.source_points == tuple(controller.canvas.image_to_widget(p) for p in document.get_professional_feature(definition.feature_ids[0]).points_px)
    assert workspace.panel.table.rowCount() == 3


def test_manual_circle_and_arc_commit_from_canvas(mounted_window):
    _window, document, workspace, _plane = mounted_window()
    _begin(workspace, "circle")
    for point in [(140, 90), (110, 120), (80, 90)]:
        _click(workspace.controller().canvas, point)
    definition = document.professional_measurements[-1]
    metrics = {m.key: m.value for m in workspace.controller().results()[definition.id].metrics}
    assert metrics["diameter"] == pytest.approx(60)
    assert metrics["radius"] == pytest.approx(30)
    _paint(workspace.controller())
    _begin(workspace, "arc")
    for point in [(140, 90), (110, 60), (110, 120)]:
        _click(workspace.controller().canvas, point)
    result = workspace.controller().results()[document.professional_measurements[-1].id]
    assert result.valid and result.primary.value == pytest.approx(1.5 * np.pi * 30)


def test_live_construction_source_recomputes_delete_and_undo(mounted_window):
    window, document, workspace, _plane = mounted_window()
    line = ConstructionEntity("baseline", "测量基准", LineDefinition(Point(20, 60), Point(80, 60)))
    document.add_construction_entity(line, mark_dirty=False)
    workspace.refresh()
    box = workspace.panel.sources[0]
    index = next(i for i in range(box.count()) if box.itemData(i) == ("construction", "baseline"))
    box.setCurrentIndex(index)
    workspace.begin()
    definition = document.professional_measurements[-1]
    assert workspace.active is None
    assert workspace.controller().results()[definition.id].primary.value == pytest.approx(60)
    replacement = replace(line, definition=LineDefinition(Point(20, 60), Point(120, 60)))
    workspace.change(document, "调整测试基准", lambda: document.replace_construction_entity("baseline", replacement, mark_dirty=False))
    assert workspace.controller().results()[definition.id].primary.value == pytest.approx(100)
    workspace.change(document, "删除测试基准", lambda: document.remove_construction_entity("baseline", mark_dirty=False))
    assert not workspace.controller().results()[definition.id].valid
    window.undo_current_document()
    assert workspace.controller().results()[definition.id].primary.value == pytest.approx(100)


def test_label_drag_escape_release_and_undo_keep_numeric_value(mounted_window):
    window, document, workspace, plane = mounted_window()
    definition = _distance(workspace)
    controller = workspace.controller()
    _paint(controller)
    original = document.get_professional_measurement(definition.id).annotation_layout
    initial_value = controller.results()[definition.id].primary.value
    initial_revision = document.professional_geometry_revision
    start = controller.label_hits[definition.id].text_rect.center()
    end = start + QPointF(35, -20)
    controller.canvas.mousePressEvent(_mouse(QEvent.Type.MouseButtonPress, start))
    controller.canvas.mouseMoveEvent(_mouse(QEvent.Type.MouseMove, end))
    assert controller.preview_layouts
    _paint(controller)
    controller.canvas.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier))
    assert not controller.preview_layouts and controller.drag is None
    assert document.get_professional_measurement(definition.id).annotation_layout == original
    _paint(controller)
    start = controller.label_hits[definition.id].text_rect.center()
    controller.canvas.mousePressEvent(_mouse(QEvent.Type.MouseButtonPress, start))
    controller.canvas.mouseMoveEvent(_mouse(QEvent.Type.MouseMove, start + QPointF(35, -20)))
    controller.canvas.mouseReleaseEvent(_mouse(QEvent.Type.MouseButtonRelease, start + QPointF(35, -20)))
    moved = document.get_professional_measurement(definition.id).annotation_layout
    assert moved != original
    assert moved.offset_px.x == pytest.approx(original.offset_px.x + 35 / controller.canvas._zoom)
    assert controller.results()[definition.id].primary.value == initial_value
    assert document.professional_geometry_revision == initial_revision
    assert window._rasters[document.id].sha256() == plane.sha256()
    window.undo_current_document()
    assert document.get_professional_measurement(definition.id).annotation_layout == original
    window.redo_current_document()
    assert document.get_professional_measurement(definition.id).annotation_layout == moved


def test_raw_crop_ignores_view_display_transform_and_is_a_copy(mounted_window):
    y, x = np.mgrid[:192, :256]
    array = (x * 111 + y * 143).astype(np.uint16)
    window, document, workspace, plane = mounted_window(array)
    canvas = workspace.controller().canvas
    points = (Point(35, 90), Point(220, 90))
    before, origin = raw_caliper_source(window, canvas, points, CaliperOptions(), "caliper_line")
    window._preview_document_display_transform(document.id, plane.sha256(), DisplayTransform(channel_ranges=((5000., 60000.),), gamma=2.1))
    after, next_origin = raw_caliper_source(window, canvas, points, CaliperOptions(), "caliper_line")
    assert after.dtype == np.uint16 and origin == next_origin
    np.testing.assert_array_equal(before, after)
    np.testing.assert_array_equal(after, array[origin[1]:origin[1] + after.shape[0], origin[0]:origin[0] + after.shape[1]])
    after[:] = 0
    assert window._rasters[document.id].sha256() == plane.sha256()


def test_async_line_caliper_preview_requires_confirmation(mounted_window, desktop_application):
    _window, document, workspace, _plane = mounted_window()
    _begin(workspace, "caliper_line")
    _click(workspace.controller().canvas, (30, 90))
    _click(workspace.controller().canvas, (220, 90))
    _wait_until(desktop_application, lambda: workspace._preview is not None)
    assert not document.professional_measurements
    assert workspace.panel.confirm.isEnabled()
    assert workspace.controller().key is None
    _paint(workspace.controller())
    # Unrelated clicks after the preview cannot continue a completed point slot.
    _click(workspace.controller().canvas, (210, 140))
    workspace.confirm_fit()
    assert len(document.professional_measurements) == 1
    definition = document.professional_measurements[0]
    result = workspace.controller().results()[definition.id]
    assert result.valid, result.reason
    assert not workspace.panel.confirm.isEnabled()
    assert document.professional_features[0].quality["profiles"]


def test_async_cancel_and_late_delivery_cannot_apply(mounted_window, monkeypatch, desktop_application):
    import fdm.services.metrology_fitting as fitting
    _window, document, workspace, _plane = mounted_window()
    entered, release = Event(), Event()
    real = fitting.run_line_caliper
    captured = []

    def delayed(*args, **kwargs):
        entered.set()
        assert release.wait(2)
        kwargs.pop("cancel", None)
        value = real(*args, **kwargs)
        captured.append(value)
        return value

    monkeypatch.setattr(fitting, "run_line_caliper", delayed)
    _begin(workspace, "caliper_line")
    _click(workspace.controller().canvas, (30, 90))
    _click(workspace.controller().canvas, (220, 90))
    assert entered.wait(1)
    job = workspace._job
    workspace.controller().canvas.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier))
    assert workspace._job is None
    release.set()
    _wait_until(desktop_application, lambda: bool(captured))
    workspace._fit_finished(job, captured[0])
    assert workspace._preview is None
    assert not document.professional_measurements
    assert not workspace.panel.confirm.isEnabled()


def test_caliper_changed_calibration_rejects_pending_confirmation(mounted_window, desktop_application):
    _window, document, workspace, _plane = mounted_window()
    _begin(workspace, "caliper_line")
    _click(workspace.controller().canvas, (30, 90))
    _click(workspace.controller().canvas, (220, 90))
    _wait_until(desktop_application, lambda: workspace._preview is not None)
    document.calibration = Calibration("manual", 2, "µm", "test")
    workspace.confirm_fit()
    assert not document.professional_measurements
    assert workspace._preview is None
    assert "来源已改变" in workspace.panel.prompt.text()


def test_caliper_replaced_raw_plane_rejects_pending_confirmation(mounted_window, desktop_application):
    window, document, workspace, plane = mounted_window()
    _begin(workspace, "caliper_line")
    _click(workspace.controller().canvas, (30, 90))
    _click(workspace.controller().canvas, (220, 90))
    _wait_until(desktop_application, lambda: workspace._preview is not None)
    window._rasters[document.id] = RasterPlane(plane.width, plane.height, plane.pixel_type, bytes(len(plane.data)))
    workspace.confirm_fit()
    assert not document.professional_measurements
    assert workspace._preview is None


def test_opaque_rgba_caliper_dispatches_explicit_channel_order(mounted_window, desktop_application):
    y, _x = np.mgrid[:192, :256]
    edge = np.rint(255 * (0.5 + 0.5 * np.tanh((y - 90.3) / 0.8))).astype(np.uint8)
    array = np.dstack((edge, edge, edge, np.full_like(edge, 255)))
    _window, document, workspace, _plane = mounted_window(array)
    _begin(workspace, "caliper_line")
    _click(workspace.controller().canvas, (30, 90))
    _click(workspace.controller().canvas, (220, 90))
    _wait_until(desktop_application, lambda: workspace._preview is not None)
    assert workspace.panel.confirm.isEnabled(), workspace.panel.prompt.text()
    workspace.confirm_fit()
    assert len(document.professional_measurements) == 1


def test_circle_caliper_from_real_clicks_confirms_valid_circle(mounted_window, desktop_application):
    y, x = np.mgrid[:192, :256]
    array = np.rint(255 * (0.5 + 0.5 * np.tanh((np.hypot(x - 128, y - 96) - 40.3) / 0.8))).astype(np.uint8)
    _window, document, workspace, _plane = mounted_window(array)
    _begin(workspace, "caliper_circle")
    _click(workspace.controller().canvas, (128, 96))
    _click(workspace.controller().canvas, (168, 96))
    _wait_until(desktop_application, lambda: workspace._preview is not None)
    assert workspace.panel.confirm.isEnabled(), workspace.panel.prompt.text()
    workspace.confirm_fit()
    result = workspace.controller().results()[document.professional_measurements[-1].id]
    assert result.valid, result.reason
    assert {m.key: m.value for m in result.metrics}["radius"] == pytest.approx(40.3, abs=0.3)
    _paint(workspace.controller())


def test_csv_export_contains_independent_metrics_and_layout_cannot_change_values(mounted_window, tmp_path):
    _window, document, workspace, _plane = mounted_window()
    definition = _distance(workspace)
    path = tmp_path / "专业测量.csv"
    export_professional_results(document, workspace.controller().results(), path)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        before = list(csv.DictReader(handle))
    assert len(before) == 3
    assert before[0]["编号"] == "P1" and float(before[0]["数值"]) == pytest.approx(50)
    assert before[0]["单位"] == "px"
    workspace.edit_layout(workspace.controller(), definition.id, definition.annotation_layout.with_manual_offset(Point(150, -100)))
    export_professional_results(document, workspace.controller().results(), path)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        assert list(csv.DictReader(handle)) == before


def test_slide_focus_cancels_unfinished_points_and_scopes_saved_labels(professional_slide, desktop_application):
    _window, document, workspace, _path, _hash = professional_slide
    canvas = workspace.controller().canvas
    _begin(workspace, "distance")
    _click(canvas, (190, 160))
    assert workspace.controller().points
    canvas.set_focus_index(3)
    _wait_until(desktop_application, canvas.pixel_work_enabled)
    _paint(workspace.controller())
    assert workspace.controller().key is None and not workspace.controller().points
    assert not document.professional_measurements
    definition = _distance(workspace, (190, 160), (280, 160))
    assert definition.focus_index == 3
    _paint(workspace.controller())
    assert definition.id in workspace.controller().label_hits
    canvas.set_focus_index(2)
    _wait_until(desktop_application, canvas.pixel_work_enabled)
    _paint(workspace.controller())
    assert definition.id not in workspace.controller().label_hits
    workspace.select(definition.id, navigate=True)
    _wait_until(desktop_application, canvas.pixel_work_enabled)
    assert canvas.focus_index() == 3


def test_slide_raw_crop_and_confirm_ignore_view_brightness_exposure(professional_slide, desktop_application):
    _window, document, workspace, path, source_hash = professional_slide
    canvas = workspace.controller().canvas
    points = (Point(190, 180), Point(285, 180))
    before, origin = raw_caliper_source(workspace.window, canvas, points, CaliperOptions(), "caliper_line")
    original = canvas._image.copy()
    canvas.set_display_adjustments(60, 17)
    canvas.grab()
    after, next_origin = raw_caliper_source(workspace.window, canvas, points, CaliperOptions(), "caliper_line")
    assert origin == next_origin
    assert canvas._image == original
    np.testing.assert_array_equal(before, after)
    _begin(workspace, "caliper_line")
    for p in points:
        _click(canvas, (p.x, p.y))
    _wait_until(desktop_application, lambda: workspace._preview is not None)
    assert workspace.panel.confirm.isEnabled(), workspace.panel.prompt.text()
    canvas.set_display_adjustments(-40, -15)
    workspace.confirm_fit()
    definition = document.professional_measurements[-1]
    assert definition.focus_index == 2
    assert workspace.controller().results()[definition.id].valid
    assert hashlib.sha256(path.read_bytes()).hexdigest() == source_hash


def test_slide_raw_crop_rejects_gap_even_when_both_endpoints_are_covered(professional_slide, monkeypatch):
    window, _document, workspace, _path, _hash = professional_slide
    canvas = workspace.controller().canvas
    state = canvas.presentation_state()
    # Simulate a renderer reporting native pixels on both sides of a missing
    # tile strip. The crop must require the complete search band, not overlap.
    partial = replace(state, coverage_rects=((160, 120, 75, 120), (245, 120, 75, 120)))
    monkeypatch.setattr(canvas, "presentation_state", lambda: partial)
    assert canvas.pixel_work_enabled()
    with pytest.raises(ValueError, match="缺失瓦片"):
        raw_caliper_source(window, canvas, (Point(190, 180), Point(285, 180)), CaliperOptions(), "caliper_line")


def test_slide_manual_label_remains_visible_when_source_geometry_leaves_view(professional_slide, desktop_application):
    _window, _document, workspace, _path, _hash = professional_slide
    canvas = workspace.controller().canvas
    definition = _distance(workspace, (190, 180), (280, 180))
    workspace.edit_layout(workspace.controller(), definition.id, definition.annotation_layout.with_manual_offset(Point(145, 0)))
    canvas.move_viewport_by(160, 0)
    _wait_until(desktop_application, canvas.pixel_work_enabled)
    _paint(workspace.controller())
    label = workspace.controller().label_hits[definition.id]
    result = workspace.controller().results()[definition.id]
    assert not canvas.rect().contains(canvas.image_to_widget(result.anchor_px).toPoint())
    assert canvas.rect().contains(label.text_rect.center().toPoint())
    assert workspace.controller()._hit_label(label.text_rect.center()) == definition.id
    assert label.leader


def test_slide_export_uses_frozen_focus_and_original_pixels(professional_slide, tmp_path, desktop_application):
    window, document, workspace, _path, _hash = professional_slide
    canvas = workspace.controller().canvas
    _distance(workspace, (190, 160), (280, 160))
    context = ExportRenderContext(document_id=document.id, render_mode=ExportImageRenderMode.CURRENT_VIEWPORT,
                                  focus_index=2, origin_x=160, origin_y=120, viewport_width=160, viewport_height=120)
    images = []
    for index, focus in enumerate([2, 3]):
        canvas.set_focus_index(focus)
        _wait_until(desktop_application, canvas.pixel_work_enabled)
        canvas.set_display_adjustments(0 if index == 0 else 65, 0 if index == 0 else 20)
        target = tmp_path / f"frozen-{focus}.png"
        window._render_overlay_image(document, target, include_measurements=True, include_scale=False,
                                     include_annotations=True, render_mode=ExportImageRenderMode.CURRENT_VIEWPORT,
                                     render_context=context)
        images.append(QImage(str(target)))
    assert not images[0].isNull()
    assert images[0] == images[1]
    target = tmp_path / "raw.png"
    window._render_overlay_image(document, target, include_measurements=False, include_scale=False,
                                 include_annotations=False, render_mode=ExportImageRenderMode.CURRENT_VIEWPORT,
                                 render_context=context)
    assert images[0] != QImage(str(target))
