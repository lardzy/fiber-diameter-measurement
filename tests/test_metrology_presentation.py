"""Dimensions, screen modes and export notices use the same live semantics."""
from __future__ import annotations

from dataclasses import replace
import math

import pytest
from PySide6.QtCore import QRectF

from fdm.construction_geometry import ConstructionEntity, LineDefinition
from fdm.geometry import Point
from fdm.measurement_annotation import MeasurementAnnotationLayout
from fdm.metrology import MeasurementDefinition, MeasurementFeature
from fdm.models import Calibration, ImageDocument
from fdm.services.metrology_service import evaluate_measurement
from fdm.ui.measurement_export_preview import clipped_manual_labels
from fdm.ui.professional_rendering import presentation_result, result_caption

from test_metrology_workspace import (mounted_window, professional_slide, _begin,
                                      _click, _distance, _paint, _wait_until)


def _document_feature(kind, points, key):
    document = ImageDocument("presentation", "", (400, 300), calibration=Calibration("manual", 0.5, "µm", "test", 1))
    document.initialize_runtime_state()
    feature = MeasurementFeature("feature", document.id, kind, tuple(Point(*p) for p in points))
    definition = MeasurementDefinition("dimension", document.id, kind if kind == "circle" else "distance", (feature.id,),
                                       annotation_layout=MeasurementAnnotationLayout(metric_key=key))
    document.apply_professional_state([feature], [definition], mark_dirty=False)
    return document, definition, evaluate_measurement(document, definition)


@pytest.mark.parametrize("key,start,end", [
    ("radius", Point(80, 90), Point(95, 90)),
    ("diameter", Point(65, 90), Point(95, 90)),
])
def test_circle_selected_metric_has_matching_physical_dimension_witness(key, start, end):
    document, definition, result = _document_feature("circle", [(95, 90), (80, 120), (65, 90)], key)
    assert result.valid
    shown = presentation_result(document, definition, result)
    dimension = [p for p in shown.geometry if p.role == "measurement"]
    assert len(dimension) == 1
    assert dimension[0].points_px == (start, end)
    assert any(p.kind == "polyline" and p.role == "source" for p in shown.geometry)
    physical_length = math.hypot((end.x - start.x) * 2, end.y - start.y)
    selected = next(m for m in result.metrics if m.key == key)
    assert selected.value == pytest.approx(physical_length)
    assert ("R " if key == "radius" else "⌀ ") in result_caption(definition, result, 1, decimals=2)
    assert shown.metrics == result.metrics


@pytest.mark.parametrize("key,finish,anchor", [
    ("dx", Point(40, 140), Point(70, 140)),
    ("dy", Point(100, 50), Point(100, 95)),
])
def test_signed_distance_components_have_axis_witness_and_projection_line(key, finish, anchor):
    document, definition, result = _document_feature("line", [(100, 140), (40, 50)], key)
    shown = presentation_result(document, definition, result)
    dimension, extension = shown.geometry
    assert dimension.points_px == (Point(100, 140), finish)
    assert dimension.role == "measurement"
    assert extension.points_px == (finish, Point(40, 50))
    assert extension.role == "witness"
    assert shown.anchor_px == anchor
    metric = next(m for m in shown.metrics if m.key == key)
    assert metric.value == (-120 if key == "dx" else -90)
    assert str(int(metric.value)) in result_caption(definition, result, 2, decimals=2)


def test_circle_center_metric_is_a_point_marker_not_a_diameter_line():
    document, definition, result = _document_feature("circle", [(95, 90), (80, 120), (65, 90)], "center_x")
    shown = presentation_result(document, definition, result)
    measured = [p for p in shown.geometry if p.role == "measurement"]
    assert len(measured) == 1
    assert measured[0].kind == "point" and measured[0].points_px == (Point(80, 90),)
    assert shown.anchor_px == Point(80, 90)


def test_real_professional_canvas_clicks_snap_to_auxiliary_endpoints(mounted_window):
    _window, document, workspace, _plane = mounted_window()
    source = ConstructionEntity("guide", "辅助定位线", LineDefinition(Point(40, 50), Point(180, 110)))
    document.add_construction_entity(source, mark_dirty=False)
    workspace.refresh()
    _begin(workspace, "distance")
    canvas = workspace.controller().canvas
    assert canvas._object_snap_pointer_enabled()
    _click(canvas, (40.1, 50.1))
    _click(canvas, (180.1, 110.1))
    definition = document.professional_measurements[-1]
    assert document.get_professional_feature(definition.feature_ids[0]).points_px == (Point(40, 50), Point(180, 110))


def test_screen_labels_all_selected_hidden_keep_geometry_and_values(mounted_window):
    _window, document, workspace, _plane = mounted_window()
    first = _distance(workspace)
    second = _distance(workspace, (150, 130), (200, 160))
    controller = workspace.controller()
    initial = controller.results()
    document.select_professional(second.id)
    for mode, expected in [("all", {first.id, second.id}), ("selected", {second.id}), ("hidden", set())]:
        controller.canvas.set_screen_measurement_labels(mode)
        _paint(controller)
        assert set(controller.label_hits) == expected
        assert controller.results() is initial
    controller.canvas.set_screen_measurement_labels("selected")
    document.select_professional(first.id)
    _paint(controller)
    assert set(controller.label_hits) == {first.id}


def test_export_clipped_manual_labels_are_reported_even_with_hidden_screen_labels(mounted_window):
    _window, document, workspace, _plane = mounted_window()
    definition = _distance(workspace)
    controller = workspace.controller()
    layout = definition.annotation_layout.with_manual_offset(Point(1000, 0))
    workspace.edit_layout(controller, definition.id, layout)
    before = document.to_dict()
    for mode in ["all", "selected", "hidden"]:
        controller.canvas.set_screen_measurement_labels(mode)
        _paint(controller)
        assert clipped_manual_labels(controller.canvas, QRectF(controller.canvas.rect())) == ("P1",)
    assert document.to_dict() == before


def test_export_clip_check_uses_current_layout_without_waiting_for_canvas_paint(mounted_window):
    _window, document, workspace, _plane = mounted_window()
    definition = _distance(workspace)
    controller = workspace.controller()
    _paint(controller)
    assert clipped_manual_labels(controller.canvas, QRectF(controller.canvas.rect())) == ()
    layout = definition.annotation_layout.with_manual_offset(Point(1000, 0))
    workspace.edit_layout(controller, definition.id, layout)
    assert clipped_manual_labels(controller.canvas, QRectF(controller.canvas.rect())) == ("P1",)
    assert document.get_professional_measurement(definition.id).annotation_layout == layout


def test_export_clip_warning_excludes_other_focus_labels(professional_slide, desktop_application):
    _window, _document, workspace, _path, _source_hash = professional_slide
    definition = _distance(workspace, (190, 160), (280, 160))
    controller = workspace.controller()
    workspace.edit_layout(controller, definition.id, definition.annotation_layout.with_manual_offset(Point(1000, 0)))
    _paint(controller)
    assert clipped_manual_labels(controller.canvas, QRectF(controller.canvas.rect())) == ("P1",)
    controller.canvas.set_focus_index(3)
    _wait_until(desktop_application, controller.canvas.pixel_work_enabled)
    assert clipped_manual_labels(controller.canvas, QRectF(controller.canvas.rect())) == ()
