"""Ordinary labels retain their defaults through creation and stay beside the measured geometry."""
import math
from dataclasses import replace

import pytest
from PySide6.QtCore import QEvent, QPointF
from PySide6.QtGui import QImage

from fdm.geometry import Line, Point
from fdm.measurement_annotation import MeasurementAnnotationLayout
from fdm.models import Measurement
from fdm.services.snap_service import SnapResult
from fdm.settings import AppSettings
from fdm.ui.canvas import DocumentCanvas
from fdm.ui.rendering import resolve_measurement_annotation
from test_measurement_annotations import make_document, mouse
from test_metrology_workspace import mounted_window


@pytest.mark.parametrize("end", [Point(220, 80), Point(210, 170), Point(120.5, 210)])
def test_ordinary_text_stays_centered_on_the_effective_line_when_rotated(end):
    document, measurement = make_document()
    measurement.snapped_line_px = Line(Point(120, 80), end)
    settings = AppSettings()
    measurement.annotation_layout = settings.new_annotation_layout("line", dual_supported=True)
    transform = lambda point: QPointF(1.25 * point.x + 15., 1.25 * point.y + 45.)
    a, b = (transform(point) for point in (measurement.snapped_line_px.start, end))
    axis = b - a
    centres = []
    for parallel in (False, True):
        settings.length_measurement_label_style.parallel_to_line = parallel
        label = resolve_measurement_annotation(measurement, document, settings, transform)
        centre = label.text_rect.center()
        delta = centre - (a + b) * .5
        assert not label.leader and not label.witness_lines
        assert QPointF.dotProduct(delta, axis) == pytest.approx(0, abs=1e-8)
        assert 0 < math.hypot(delta.x(), delta.y()) < label.paint_rect.height()
        assert label.contains_text(centre)
        centres.append(centre)
    assert centres[0] == centres[1]


def test_polyline_label_uses_half_path_length_and_compatible_line_defaults():
    document, measurement = make_document("polyline")
    measurement.polyline_px = [Point(0, 0), Point(12, 0), Point(12, 100)]
    settings = AppSettings(annotation_style_defaults={"line": {
        "label_position": "near", "show_identifier": True, "show_symbol": True, "leader_style": "dual"}})
    measurement.annotation_layout = settings.new_annotation_layout("polyline")
    assert measurement.annotation_layout.show_identifier
    assert not measurement.annotation_layout.show_symbol
    assert measurement.annotation_layout.leader_style == "single"
    label = resolve_measurement_annotation(measurement, document, settings, lambda p: QPointF(p.x, p.y))
    assert label.anchor == QPointF(12, 44)
    assert label.text_rect.center().y() == pytest.approx(44)
    settings.annotation_style_defaults["polyline"] = {"show_identifier": False}
    assert not settings.new_annotation_layout("polyline").show_identifier


@pytest.mark.parametrize("tool", ["continuous_manual", "manual", "snap"])
def test_label_covering_shared_endpoint_leaves_point_input_to_active_tool(tool):
    document, measurement = make_document()
    measurement.annotation_layout = MeasurementAnnotationLayout.ordinary_near().with_manual_offset(Point(70, 0))
    canvas = DocumentCanvas()
    try:
        canvas.resize(1000, 600)
        canvas.set_document(document, QImage(2000, 1000, QImage.Format.Format_RGB32))
        end = canvas.image_to_widget(measurement.effective_line().end)
        assert canvas._measurement_annotation_hit(end)[0] is measurement
        requests = []
        canvas.measurementAnnotationPropertiesRequested.connect(lambda *args: requests.append(args))
        canvas.set_tool_mode(tool)
        canvas.mousePressEvent(mouse(QEvent.Type.MouseButtonPress, end))
        assert canvas._annotation_drag is None
        assert canvas._drawing_polygon_points if tool == "continuous_manual" else canvas._drawing_line is not None
        canvas.mouseReleaseEvent(mouse(QEvent.Type.MouseButtonRelease, end))
        assert not requests
        canvas.set_tool_mode("select")
        canvas.mousePressEvent(mouse(QEvent.Type.MouseButtonPress, end))
        canvas.mouseReleaseEvent(mouse(QEvent.Type.MouseButtonRelease, end))
        assert requests == [(document.id, measurement.id)]
    finally:
        canvas.clear_document()
        canvas.close()


@pytest.mark.parametrize("old_mode", ["manual", "automatic"])
def test_opening_old_line_discards_offsets_but_preserves_data_and_new_drag(old_mode):
    _, original = make_document()
    old = original.to_dict()
    old["annotation_layout"].update(placement_mode=old_mode, offset_px={"x": 1250, "y": 450},
                                    leader_style="dual", show_identifier=True, show_metric_name=True)
    old["annotation_layout"].pop("label_position")
    loaded = Measurement.from_dict(old)
    persisted = loaded.to_dict()
    style = loaded.annotation_layout
    assert style.label_position == "near" and style.offset_px is None and not style.show_leader
    assert style.show_identifier and style.show_metric_name
    assert {k: v for k, v in persisted.items() if k != "annotation_layout"} == {
        k: v for k, v in old.items() if k != "annotation_layout"}
    loaded.annotation_layout = replace(style, label_position="external", show_leader=True).with_manual_offset(Point(-90, 35))
    assert Measurement.from_dict(loaded.to_dict()).annotation_layout == loaded.annotation_layout


def test_manual_snapped_quick_and_batch_insert_keep_loaded_line_style(mounted_window, monkeypatch):
    window, document, _, _ = mounted_window()
    settings = AppSettings()
    saved_style = MeasurementAnnotationLayout(label_position="external", leader_style="dual", show_leader=True,
                                             show_identifier=True, show_symbol=True, show_metric_name=True)
    settings.annotation_style_defaults["line"] = saved_style.style_fields()
    window._app_settings.annotation_style_defaults = AppSettings.from_dict(settings.to_dict()).annotation_style_defaults
    line = Line(Point(55, 70), Point(145, 97))
    snapped = Line(Point(63, 72), Point(134, 93))
    monkeypatch.setattr(window.snap_service, "snap_measurement",
                        lambda image, original: SnapResult("snapped", original, snapped, 74., .98))
    for mode, payload in (("manual", line), ("snap", line), ("fiber_quick", {"line_px": line})):
        window._on_canvas_line_committed(document.id, mode, payload)
    bulk = Measurement("batch-line", document.id, None, "manual", line_px=line)
    window._apply_document_change(document, "批量新增线段", lambda: document.add_measurement(bulk))
    assert len(document.measurements) == 4
    assert document.measurements[1].effective_line() == snapped
    for measurement in document.measurements:
        assert measurement.annotation_layout.style_fields() == saved_style.style_fields()
        assert measurement.annotation_layout.offset_px is not None


def test_settings_previews_show_separate_ordinary_and_professional_defaults():
    from fdm.ui.dialogs import SettingsDialog
    from fdm.ui.professional_rendering import result_caption

    ordinary = MeasurementAnnotationLayout.ordinary_near()
    professional = MeasurementAnnotationLayout(leader_style="single", show_identifier=True, metric_key="dy")
    settings = AppSettings(annotation_style_defaults={"line": ordinary.style_fields(),
                                                      "professional:distance": professional.style_fields()})
    dialog = SettingsDialog(settings, document=None)
    try:
        preview = dialog._length_measurement_style_preview
        line = preview._sample.measurements[0]
        assert line.annotation_layout.style_fields() == ordinary.style_fields()
        assert line.annotation_layout.offset_px is None

        preview = dialog._professional_measurement_style_preview
        preview.resize(420, 140)
        assert not preview.grab().isNull()
        example = preview._document.professional_measurements[0]
        assert example.annotation_layout.style_fields() == professional.style_fields()
        assert example.annotation_layout.offset_px is not None
        assert result_caption(example, preview._results[example.id], 1, decimals=2).startswith("P1 · -74.63")
    finally:
        dialog.close()
