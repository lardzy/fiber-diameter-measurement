"""Linked dimension preferences, rotated hit geometry and document compatibility."""
from dataclasses import replace

import pytest
from PySide6.QtCore import QPointF
from PySide6.QtGui import QFontMetricsF
from PySide6.QtWidgets import QCheckBox, QDialog

from fdm.geometry import Line, Point
from fdm.measurement_annotation import MeasurementAnnotationLayout
from fdm.models import Calibration, ProjectState
from fdm.settings import AppSettings, AppSettingsIO
from fdm.ui.measurement_annotation_layout import readable_angle
from fdm.ui.professional_rendering import professional_label_layout, professional_text_scale, result_caption
from fdm.ui.rendering import measurement_display_text_with_settings, measurement_label_font, resolve_measurement_annotation
from test_measurement_annotations import make_document
from test_metrology_workspace import _distance, mounted_window


def test_old_caption_roundtrip_and_required_feature_only_when_needed():
    document, measurement = make_document()
    document.calibration = Calibration("manual", 14., "mm", "sample")
    measurement.recalculate(document.calibration)
    settings = AppSettings()
    assert measurement_display_text_with_settings(measurement, document, settings) == "10.00 mm"
    assert "measurement-caption-fields/v1" in ProjectState("test", [document]).effective_required_features()
    old = measurement.annotation_layout.to_dict()
    for key in ("show_symbol", "show_metric_name"):
        old.pop(key)
    old["show_identifier"] = True
    measurement.annotation_layout = MeasurementAnnotationLayout.from_dict(old)
    assert measurement.annotation_layout.to_dict() == old
    assert measurement_display_text_with_settings(measurement, document, settings) == "M1 · 直径 10.00 mm"
    assert "measurement-caption-fields/v1" not in ProjectState("test", [document]).effective_required_features()
    measurement.annotation_layout = replace(measurement.annotation_layout, show_identifier=False, show_symbol=True)
    assert measurement_display_text_with_settings(measurement, document, settings) == "直径 ⌀ 10.00 mm"


@pytest.mark.parametrize("style", ["single", "dual"])
def test_sloped_reversed_label_uses_exact_hit_area_and_endpoint_witness(style):
    document, measurement = make_document()
    measurement.line_px = Line(Point(160, 210), Point(85, 120))
    measurement.annotation_layout = MeasurementAnnotationLayout(offset_px=Point(30, -20), leader_style=style)
    settings = AppSettings()
    settings.length_measurement_label_style.parallel_to_line = True
    transform = lambda point: QPointF(point.x, point.y)
    resolved = resolve_measurement_annotation(measurement, document, settings, transform)
    assert 0 < resolved.angle < 90
    assert resolved.contains_text(resolved.text_rect.center())
    assert not resolved.contains_text(resolved.text_rect.topRight() - QPointF(1, -1))
    if style == "dual":
        assert len(resolved.witness_lines) == 2
        for p, raw in zip(resolved.source_points, (measurement.line_px.start, measurement.line_px.end)):
            assert (p - transform(raw)).manhattanLength() < 1e-8
    assert readable_angle(QPointF(0, 100), QPointF(0, 1)) == -90


@pytest.mark.parametrize("space,ratio", [("image_px", 2.), ("legacy_output_px", 1.)])
def test_professional_respects_shared_size_mode_and_stable_center(mounted_window, space, ratio):
    _, document, workspace, _ = mounted_window()
    definition = _distance(workspace, (30, 40), (97, 100))
    result = workspace.controller().results()[definition.id]
    settings = AppSettings(measurement_text_size_space=space)
    settings.length_measurement_label_style.parallel_to_line = True
    boxes = []
    for zoom in (1., 2.):
        transform = lambda p: QPointF(p.x * zoom, p.y * zoom)
        _, label = professional_label_layout(definition, result, 1, transform, definition.annotation_layout,
            decimals=2, metrics=QFontMetricsF(measurement_label_font(settings)), scale=professional_text_scale(settings, transform),
            parallel=True)
        assert 0 < label.angle < 90
        boxes.append(label)
    assert boxes[1].paint_rect.width() == pytest.approx(boxes[0].paint_rect.width() * ratio)
    assert boxes[1].text_rect.center() == boxes[0].text_rect.center() * 2


def test_remember_only_after_explicit_default_and_only_for_same_kind(mounted_window, monkeypatch):
    window, doc, workspace, _ = mounted_window()
    first = _distance(workspace)
    controller = workspace.controller()
    def dialog_with(remember):
        def edit(dialog):
            boxes = {box.text(): box for box in dialog.findChildren(QCheckBox)}
            boxes["显示指标名称"].setChecked(True)
            boxes["显示测量编号"].setChecked(True)
            boxes["设为同类默认"].setChecked(remember)
            return QDialog.DialogCode.Accepted
        return edit
    monkeypatch.setattr(QDialog, "exec", dialog_with(False))
    workspace.professional_properties(controller, first)
    assert not window._app_settings.annotation_style_defaults
    assert doc.get_professional_measurement(first.id).annotation_layout.show_identifier
    second = _distance(workspace, (70, 60), (90, 90))
    assert not second.annotation_layout.show_identifier
    before = doc.to_dict()
    monkeypatch.setattr(QDialog, "exec", dialog_with(True))
    workspace.professional_properties(controller, doc.get_professional_measurement(first.id))
    assert doc.to_dict() == before  # saving the style alone isn't a project edit
    reloaded = AppSettings.from_dict(window._app_settings.to_dict())
    fields = reloaded.annotation_style_defaults["professional:distance"]
    assert not {"offset_px", "revision", "placement_mode"}.intersection(fields)
    assert reloaded.new_annotation_layout("professional:distance", dual_supported=True).show_identifier
    assert not reloaded.new_annotation_layout("line", dual_supported=True).show_identifier
    third = _distance(workspace, (110, 80), (170, 100))
    assert third.annotation_layout.show_identifier and third.annotation_layout.show_metric_name
    assert "P3 · 距离" in result_caption(third, controller.results()[third.id], 3, decimals=2)
    assert not doc.get_professional_measurement(second.id).annotation_layout.show_identifier

def test_defaults_survive_preferences_file_but_never_restore_offsets(tmp_path):
    path = tmp_path / "设置.json"
    style = MeasurementAnnotationLayout(show_metric_name=True, show_identifier=True, metric_key="dx",
                                        offset_px=Point(150, 40), revision=8, leader_style="dual")
    settings = AppSettings(annotation_style_defaults={"professional:distance": style.style_fields()})
    AppSettingsIO.save(settings, path)
    layout = AppSettingsIO.load(path).new_annotation_layout("professional:distance", dual_supported=True)
    assert layout.style_fields() == style.style_fields()
    assert layout.offset_px is None and layout.revision == 1
    assert AppSettings.from_dict({"annotation_style_defaults": {"professional:distance": {"leader_style": "bogus"}}}).annotation_style_defaults == {}
    imported = AppSettings.from_dict({"annotation_style_defaults": {"line": {"show_identifier": True, "offset_px": {"x": 50, "y": 20}}}})
    line = imported.new_annotation_layout("line", dual_supported=True)
    assert line.show_identifier and line.offset_px is None


def test_painted_background_and_text_do_not_fill_the_rotated_envelope():
    from PySide6.QtGui import QImage, QPainter
    from fdm.ui.rendering import draw_linked_measurement_label
    doc, line = make_document()
    line.line_px = Line(Point(50, 50), Point(130, 130))
    line.annotation_layout = MeasurementAnnotationLayout(show_leader=False, offset_px=Point(60, -20))
    settings = AppSettings()
    settings.length_measurement_label_style.parallel_to_line = True
    transform = lambda point: QPointF(point.x, point.y)
    target = QImage(320, 220, QImage.Format.Format_ARGB32_Premultiplied)
    target.fill(0)
    box = resolve_measurement_annotation(line, doc, settings, transform)
    with QPainter(target) as painter:
        draw_linked_measurement_label(painter, line, doc, settings, transform, use_sprite_cache=False)
    assert target.pixelColor(box.text_rect.center().toPoint()).alpha() > 0
    corner = box.text_rect.topLeft() + QPointF(2, 2)
    assert not box.contains_text(corner)
    assert target.pixelColor(corner.toPoint()).alpha() == 0


def test_failed_default_write_preserves_old_preferences_and_current_object_edit(mounted_window, monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    window, doc, workspace, _ = mounted_window()
    first = _distance(workspace)
    def edit(dialog):
        dialog.findChild(QCheckBox, "annotationMetricName").setChecked(True)
        dialog.findChild(QCheckBox, "annotationRememberForKind").setChecked(True)
        return QDialog.DialogCode.Accepted
    messages = []
    def disk_full(*_):
        raise OSError("disk full")
    monkeypatch.setattr(QDialog, "exec", edit)
    monkeypatch.setattr(AppSettingsIO, "save", disk_full)
    monkeypatch.setattr(QMessageBox, "warning", lambda _w, title, detail: messages.append((title, detail)))
    workspace.professional_properties(workspace.controller(), first)
    assert messages and "disk full" in messages[0][1]
    assert not window._app_settings.annotation_style_defaults
    assert doc.get_professional_measurement(first.id).annotation_layout.show_metric_name
    assert not _distance(workspace, (70, 60), (90, 90)).annotation_layout.show_metric_name
