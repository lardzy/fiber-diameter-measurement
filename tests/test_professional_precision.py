"""Professional label precision changes presentation, never measurement data."""
from __future__ import annotations

from dataclasses import replace
import hashlib

import pytest
from PySide6.QtCore import QPointF
from PySide6.QtGui import QImage

from fdm.geometry import Point
from fdm.measurement_annotation import MeasurementAnnotationLayout
from fdm.metrology import MeasurementDefinition, MeasurementMetric, MeasurementResult
from fdm.models import ImageDocument
from fdm.services.export_service import ExportImageRenderMode, ExportRenderContext
from fdm.services.raster_io import raster_plane_to_qimage
from fdm.ui.measurement_export_preview import clipped_manual_labels
from fdm.ui.professional_measurement import export_professional_results
from fdm.ui.professional_rendering import result_caption

from test_metrology_drag_cache import pixels
from test_metrology_workspace import mounted_window, professional_slide, _distance, _paint, _wait_until


@pytest.mark.parametrize("value,decimals,number", [
    (266.55098, 0, "267"),
    (266.55098, 2, "266.55"),
    (266.55098, 8, "266.55098000"),
    (25, 0, "25"),
    (25, 2, "25.00"),
    (25, 8, "25.00000000"),
    (0.000123456, 2, "0.00"),
    (0.000123456, 8, "0.00012346"),
    (1234567890.125, 2, "1234567890.12"),
    (-12.3456, 2, "-12.35"),
    (-0.0, 0, "0"),
    (-0.0, 8, "0.00000000"),
    (-0.004, 2, "0.00"),
    (-0.000000004, 8, "0.00000000"),
    (-0.000000006, 8, "-0.00000001"),
])
def test_caption_uses_fixed_decimals_and_normalizes_only_rounded_negative_zero(value, decimals, number):
    definition = MeasurementDefinition("dimension", "image", "distance", ("feature",))
    metric = MeasurementMetric("distance", "距离", value, "µm")
    result = MeasurementResult(definition.id, "ready", metrics=(metric,))
    assert result_caption(definition, result, 12, decimals=decimals) == f"P12 · 距离 {number} µm"
    assert result.primary is metric and metric.value == value


@pytest.mark.parametrize("key,label,unit,symbol", [
    ("diameter", "直径", "µm", "⌀ "),
    ("radius", "半径", "px", "R "),
    ("area", "面积", "µm²", ""),
    ("angle", "夹角", "°", ""),
])
def test_all_metrics_share_label_precision_without_losing_selected_metric_identity(key, label, unit, symbol):
    definition = MeasurementDefinition("dimension", "image", "distance", ("feature",),
        annotation_layout=MeasurementAnnotationLayout(show_identifier=False, metric_key=key))
    result = MeasurementResult(definition.id, "ready", metrics=(
        MeasurementMetric("distance", "距离", 99.123, "px"),
        MeasurementMetric(key, label, 25, unit),
    ))
    assert result_caption(definition, result, 1, decimals=2) == f"{label} {symbol}25.00 {unit}"


def test_invalid_and_missing_metrics_retain_explanation():
    definition = MeasurementDefinition("dimension", "image", "distance", ("feature",))
    assert result_caption(definition, MeasurementResult(definition.id, "invalid", reason="零长度直线"),
                          2, decimals=8) == "P2 · 无效：零长度直线"
    assert result_caption(definition, MeasurementResult(definition.id, "ready"),
                          2, decimals=0) == "P2 · 无有效指标"


@pytest.mark.parametrize("leader_style", ["single", "dual"])
def test_precision_refreshes_hits_without_moving_labels_or_changing_results(mounted_window, monkeypatch, tmp_path, leader_style):
    window, document, workspace, plane = mounted_window()
    definition = _distance(workspace, (30, 40), (97, 85))
    controller = workspace.controller()
    layout = replace(definition.annotation_layout.with_manual_offset(Point(30, -20)), leader_style=leader_style)
    workspace.edit_layout(controller, definition.id, layout)
    document.mark_session_saved()
    before = document.to_dict()
    history_count = document.history.command_count
    results = controller.results()
    table_before = [workspace.panel.table.item(row, 2).text() for row in range(workspace.panel.table.rowCount())]
    details_before = workspace.panel.details.toPlainText()
    path = tmp_path / "专业数值.csv"
    export_professional_results(document, results, path)
    csv_before = path.read_bytes()
    canvas = controller.canvas

    def no_recalculation(*_args, **_kwargs):
        raise AssertionError("Changing precision must not evaluate or fit geometry")

    monkeypatch.setattr("fdm.services.metrology_service.evaluate_document", no_recalculation)
    canvas.set_settings(replace(canvas._settings, professional_measurement_label_decimals=0))
    _paint(controller)
    low = controller.label_hits[definition.id]
    export_frame = low.text_rect.adjusted(-1, -1, 1, 1)
    assert clipped_manual_labels(canvas, export_frame) == ()
    canvas.set_settings(replace(canvas._settings, professional_measurement_label_decimals=8))
    assert controller.label_hits == {}
    # Clipping checks must use the new width even before the next paint.
    assert clipped_manual_labels(canvas, export_frame) == ("P1",)
    _paint(controller)
    high = controller.label_hits[definition.id]
    assert high.text_rect.width() > low.text_rect.width()
    assert high.text_rect.center() == low.text_rect.center()
    assert high.source_points == low.source_points
    added_text_region = QPointF(high.text_rect.right() - 1, high.text_rect.center().y())
    assert not low.text_rect.contains(added_text_region)
    assert controller._hit_label(added_text_region) == definition.id
    assert controller.results() is results
    workspace.refresh()
    assert [workspace.panel.table.item(row, 2).text() for row in range(workspace.panel.table.rowCount())] == table_before
    assert workspace.panel.details.toPlainText() == details_before
    export_professional_results(document, controller.results(), path)
    assert path.read_bytes() == csv_before
    assert document.to_dict() == before
    assert not document.dirty_flags.session_dirty
    assert document.history.command_count == history_count
    assert window._rasters[document.id].sha256() == plane.sha256()


def test_applying_precision_updates_hidden_canvases_without_mutating_either_document(mounted_window, desktop_application):
    window, first_document, workspace, plane = mounted_window()
    first_definition = _distance(workspace)
    first = workspace.controller()
    _paint(first)
    first_width = first.label_hits[first_definition.id].text_rect.width()
    second_document = ImageDocument("second-precision", "/tmp/second-precision.tif", (plane.width, plane.height))
    second_document.initialize_runtime_state()
    window._mount_document(second_document, raster_plane_to_qimage(plane), tooltip=second_document.path, raster_plane=plane)
    desktop_application.processEvents()
    workspace.show()
    second_definition = _distance(workspace, (50, 60), (110, 120))
    second = workspace.controller()
    assert second is not first and not first.canvas.isVisible()
    first_document.mark_session_saved(); second_document.mark_session_saved()
    before = (first_document.to_dict(), second_document.to_dict())
    window._activate_app_settings(replace(window._app_settings, professional_measurement_label_decimals=8))
    assert first.canvas._settings.professional_measurement_label_decimals == 8
    assert first.label_hits == {} and second.label_hits == {}
    _paint(first); _paint(second)
    assert first.label_hits[first_definition.id].text_rect.width() > first_width
    assert second_definition.id in second.label_hits
    assert (first_document.to_dict(), second_document.to_dict()) == before
    assert not first_document.dirty_flags.session_dirty and not second_document.dirty_flags.session_dirty


def test_precision_during_drag_invalidates_cached_background_without_cancelling_drag(mounted_window):
    _window, document, workspace, _plane = mounted_window()
    definition = _distance(workspace)
    _distance(workspace, (130, 110), (190, 140))
    controller = workspace.controller(); canvas = controller.canvas
    initial = document.get_professional_measurement(definition.id).annotation_layout
    controller.drag = (definition.id, Point(0, 0), initial, initial.offset_px)
    controller.preview_layouts[definition.id] = initial.with_manual_offset(Point(-15, 20))
    assert pixels(controller, cached=True) == pixels(controller, cached=False)
    old_layer = controller._drag_layer
    results = controller.results()
    document_before = document.to_dict()
    for decimals in (8, 0, 2):
        canvas.set_settings(replace(canvas._settings, professional_measurement_label_decimals=decimals))
        assert controller._drag_layer is None and controller._drag_layer_hits == {}
        assert controller.drag is not None and controller.preview_layouts
        assert pixels(controller, cached=True) == pixels(controller, cached=False)
        assert controller._drag_layer is not old_layer
        old_layer = controller._drag_layer
        assert controller.results() is results
        assert document.to_dict() == document_before
    controller.cancel()


def test_slide_precision_survives_focus_changes_without_touching_source(professional_slide, desktop_application):
    _window, document, workspace, path, source_hash = professional_slide
    definition = _distance(workspace, (190, 160), (283, 187))
    controller = workspace.controller(); canvas = controller.canvas
    canvas.set_settings(replace(canvas._settings, professional_measurement_label_decimals=0))
    _paint(controller)
    low = controller.label_hits[definition.id]
    before = document.to_dict()
    canvas.set_settings(replace(canvas._settings, professional_measurement_label_decimals=8))
    _paint(controller)
    high = controller.label_hits[definition.id]
    assert high.text_rect.width() > low.text_rect.width()
    assert high.source_points == low.source_points
    assert document.to_dict() == before
    canvas.set_focus_index(3)
    _wait_until(desktop_application, canvas.pixel_work_enabled)
    _paint(controller)
    assert controller.label_hits == {}
    canvas.set_focus_index(2)
    _wait_until(desktop_application, canvas.pixel_work_enabled)
    _paint(controller)
    assert canvas._settings.professional_measurement_label_decimals == 8
    assert controller.label_hits[definition.id].text_rect.width() == pytest.approx(high.text_rect.width())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == source_hash


def _verify_overlay_export_precision(window, document, workspace, tmp_path, monkeypatch, points, *, context=None):
    definition = _distance(workspace, *points)
    controller = workspace.controller()
    workspace.edit_layout(controller, definition.id, definition.annotation_layout.with_manual_offset(Point(0, -18)))
    document.mark_session_saved()
    before = document.to_dict()
    history_count = document.history.command_count
    original = controller.canvas._image.copy()
    calls = []
    from fdm.ui import professional_rendering
    original_caption = professional_rendering.result_caption

    def caption(*args, **kwargs):
        calls.append(kwargs["decimals"])
        return original_caption(*args, **kwargs)

    monkeypatch.setattr(professional_rendering, "result_caption", caption)
    mode = ExportImageRenderMode.CURRENT_VIEWPORT if context else ExportImageRenderMode.FULL_RESOLUTION
    images = []; raw = []; csv_contents = []; excel_values = []
    for decimals in (2, 8):
        window._activate_app_settings(replace(window._app_settings, professional_measurement_label_decimals=decimals))
        calls.clear(); _paint(controller)
        assert calls and set(calls) == {decimals}
        calls.clear()
        output = tmp_path / f"overlay-{decimals}.png"
        window._render_overlay_image(document, output, include_measurements=True, include_scale=False,
            include_annotations=True, render_mode=mode, render_context=context)
        assert calls and set(calls) == {decimals}
        images.append(QImage(str(output)))
        raw_output = tmp_path / f"raw-{decimals}.png"
        window._render_overlay_image(document, raw_output, include_measurements=False, include_scale=False,
            include_annotations=False, render_mode=mode, render_context=context)
        raw.append(QImage(str(raw_output)))
        csv_output = tmp_path / f"numeric-{decimals}.csv"
        export_professional_results(document, controller.results(), csv_output)
        csv_contents.append(csv_output.read_bytes())
        if context is None:
            import pandas as pd
            excel_output = tmp_path / f"numeric-{decimals}.xlsx"
            export_professional_results(document, controller.results(), excel_output)
            excel_values.append(pd.read_excel(excel_output)["数值"].tolist())
        assert controller.canvas._image == original
        assert document.to_dict() == before
        assert document.history.command_count == history_count
        assert not document.dirty_flags.session_dirty
    assert not any(image.isNull() for image in images + raw)
    assert images[0] != images[1]
    assert raw[0] == raw[1]
    assert csv_contents[0] == csv_contents[1]
    if excel_values:
        assert excel_values[0] == excel_values[1]
        assert excel_values[0] == pytest.approx([m.value for m in controller.results()[definition.id].metrics])


def test_main_window_image_export_uses_current_label_precision_with_original_numeric_values(mounted_window, tmp_path, monkeypatch):
    window, document, workspace, plane = mounted_window()
    _verify_overlay_export_precision(window, document, workspace, tmp_path, monkeypatch,
                                     ((30, 80), (170, 100)))
    assert window._rasters[document.id].sha256() == plane.sha256()


def test_main_window_slide_export_uses_current_label_precision_and_frozen_focus(professional_slide, tmp_path, monkeypatch):
    window, document, workspace, path, source_hash = professional_slide
    context = ExportRenderContext(document_id=document.id, render_mode=ExportImageRenderMode.CURRENT_VIEWPORT,
        focus_index=2, origin_x=160, origin_y=120, viewport_width=160, viewport_height=120)
    _verify_overlay_export_precision(window, document, workspace, tmp_path, monkeypatch,
                                     ((190, 180), (283, 187)), context=context)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == source_hash
