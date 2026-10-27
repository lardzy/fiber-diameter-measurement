"""CAD dimensions must retain their own two measured endpoints in every view."""
from __future__ import annotations

from dataclasses import replace
import math

import pytest
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QImage, QPainter

from fdm.geometry import Line, Point
from fdm.measurement_annotation import MeasurementAnnotationLayout
from fdm.metrology import MeasurementDefinition, MeasurementFeature
from fdm.models import ImageDocument, Measurement, ProjectState
from fdm.project_io import ProjectIO
from fdm.services.metrology_service import evaluate_document
from fdm.settings import AppSettings
from fdm.ui.measurement_annotation_layout import resolve_dual_dimension
from fdm.ui.professional_rendering import draw_professional_measurements
from fdm.ui.rendering import resolve_measurement_annotation

from test_metrology_presentation import _document_feature
from test_metrology_workspace import mounted_window, _distance, _mouse, _paint


def _chain(points):
    doc = ImageDocument("dual-chain", "", (1200, 1000))
    doc.initialize_runtime_state()
    tangent = points[1] - points[0] if isinstance(points[0], QPointF) else QPointF(points[1].x - points[0].x, points[1].y - points[0].y)
    norm = math.hypot(tangent.x(), tangent.y())
    normal = QPointF(-tangent.y() / norm, tangent.x() / norm)
    for index, (start, end) in enumerate(zip(points, points[1:])):
        layout = MeasurementAnnotationLayout(leader_style="dual", offset_px=Point(normal.x() * (70 + 50 * index), normal.y() * (70 + 50 * index)))
        measurement = Measurement(f"m{index}", doc.id, None, "manual", line_px=Line(start, end), annotation_layout=layout)
        measurement.recalculate(None)
        doc.measurements.append(measurement)
        feature = MeasurementFeature(f"f{index}", doc.id, "line", (start, end))
        definition = MeasurementDefinition(f"p{index}", doc.id, "distance", (feature.id,), annotation_layout=layout)
        doc.professional_features.append(feature)
        doc.professional_measurements.append(definition)
    return doc


def _draw(doc, transform, *, font=None):
    image = QImage(1600, 1200, QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    try:
        if font:
            painter.setFont(font)
        labels = draw_professional_measurements(painter, doc, transform, label_decimals=2)
    finally:
        painter.end()
    return image, labels


@pytest.mark.parametrize("points", [
    [Point(100, 300), Point(320, 300), Point(540, 300), Point(760, 300)],
    [Point(450, 100), Point(450, 320), Point(450, 540), Point(450, 760)],
    [Point(140, 100), Point(320, 235), Point(500, 370), Point(680, 505)],
])
@pytest.mark.parametrize("zoom", [0.8, 1.75])
def test_contiguous_old_and_professional_dimensions_do_not_join_neighbor_nodes(points, zoom):
    doc = _chain(points)
    transform = lambda p: QPointF(p.x * zoom + 25, p.y * zoom + 35)
    _image, professional = _draw(doc, transform)
    for index, (start, end) in enumerate(zip(points, points[1:])):
        expected = (transform(start), transform(end))
        old = resolve_measurement_annotation(doc.measurements[index], doc, AppSettings(), transform)
        for label in (old, professional[f"p{index}"]):
            assert label.source_points == expected
            assert len(label.witness_lines) == len(label.arrowheads) == 2
            tangent = expected[1] - expected[0]
            for source, witness, arrow in zip(expected, label.witness_lines, label.arrowheads):
                # Each extension projects from its own source, not a shared
                # text anchor or the next dimension's other endpoint.
                for endpoint in (*witness, arrow[0]):
                    delta = endpoint - source
                    assert abs(delta.x() * tangent.x() + delta.y() * tangent.y()) < 1e-7
            assert label.arrowheads[0][0] != label.arrowheads[1][0]


@pytest.mark.parametrize("end", [QPointF(106, 100), QPointF(100, 106), QPointF(104.8, 103.6)])
def test_short_span_preserves_manual_text_position_and_external_arrows(end):
    start, center = QPointF(100, 100), QPointF(103, 70)
    label = resolve_dual_dimension(start, end, center, 220, 30)
    assert label.text_rect.center() == center
    assert label.source_points == (start, end)
    assert label.dimension_lines and len(label.arrowheads) == 2
    tangent = (end - start) / math.hypot(end.x() - start.x(), end.y() - start.y())
    first, last = label.arrowheads
    for polygon, direction in ((first, -1), (last, 1)):
        base = (polygon[1] + polygon[2]) / 2
        delta = base - polygon[0]
        assert direction * (delta.x() * tangent.x() + delta.y() * tangent.y()) > 0
        assert not label.text_rect.contains(polygon[0])


def test_far_drag_style_switch_and_undo_preserve_pair_and_value(mounted_window):
    window, document, workspace, plane = mounted_window()
    definition = _distance(workspace)
    controller = workspace.controller()
    _paint(controller)
    initial = document.get_professional_measurement(definition.id).annotation_layout
    assert initial.leader_style == "dual"
    source = controller.label_hits[definition.id].source_points
    value = controller.results()[definition.id].primary.value
    start = controller.label_hits[definition.id].text_rect.center()
    finish = start + QPointF(260, -130)
    controller.canvas.mousePressEvent(_mouse(QEvent.Type.MouseButtonPress, start))
    controller.canvas.mouseMoveEvent(_mouse(QEvent.Type.MouseMove, finish))
    controller.canvas.mouseReleaseEvent(_mouse(QEvent.Type.MouseButtonRelease, finish))
    _paint(controller)
    moved = document.get_professional_measurement(definition.id).annotation_layout
    assert moved.placement_mode == "manual" and moved.offset_px != initial.offset_px
    assert controller.label_hits[definition.id].source_points == source
    assert controller.label_hits[definition.id].leader
    workspace.edit_layout(controller, definition.id, replace(moved, leader_style="single"))
    _paint(controller)
    assert not controller.label_hits[definition.id].source_points
    window.undo_current_document()
    _paint(controller)
    assert document.get_professional_measurement(definition.id).annotation_layout == moved
    assert controller.label_hits[definition.id].source_points == source
    window.undo_current_document()
    assert document.get_professional_measurement(definition.id).annotation_layout == initial
    assert controller.results()[definition.id].primary.value == value
    assert window._rasters[document.id].sha256() == plane.sha256()


def test_dual_project_roundtrip_restores_three_separate_source_pairs(tmp_path):
    points = [Point(100, 300), Point(320, 300), Point(540, 300), Point(760, 300)]
    doc = _chain(points)
    doc.path = str(tmp_path / "原图.png")
    doc.professional_measurements[1] = replace(doc.professional_measurements[1], annotation_layout=doc.professional_measurements[1].annotation_layout.with_manual_offset(Point(400, -200)))
    path = tmp_path / "连续双端尺寸.fdmproj"
    ProjectIO.save(ProjectState("dual", [doc]), path)
    restored = ProjectIO.load(path).documents[0]
    assert restored.professional_measurements == doc.professional_measurements
    assert [m.annotation_layout for m in restored.measurements] == [m.annotation_layout for m in doc.measurements]
    transform = lambda p: QPointF(p.x, p.y)
    original_image, original = _draw(doc, transform)
    restored_image, labels = _draw(restored, transform)
    assert restored_image == original_image
    assert labels == original
    assert all(d.annotation_layout.leader_style == "dual" for d in restored.professional_measurements)


def test_canvas_and_export_painter_receive_identical_dual_layout(mounted_window):
    _window, document, workspace, _plane = mounted_window()
    first = _distance(workspace, (30, 80), (100, 80))
    second = _distance(workspace, (100, 80), (180, 80))
    controller = workspace.controller()
    workspace.edit_layout(controller, first.id, first.annotation_layout.with_manual_offset(Point(10, -50)))
    workspace.edit_layout(controller, second.id, second.annotation_layout.with_manual_offset(Point(-10, 60)))
    _paint(controller)
    _export, labels = _draw(document, controller.canvas.image_to_widget, font=controller.canvas.font())
    assert labels == controller.label_hits


@pytest.mark.parametrize("kind,points,key,expected", [
    ("circle", [(95, 90), (80, 120), (65, 90)], "radius", (Point(80, 90), Point(95, 90))),
    ("line", [(100, 140), (40, 50)], "dx", (Point(100, 140), Point(40, 140))),
    ("line", [(100, 140), (40, 50)], "dy", (Point(100, 140), Point(100, 50))),
])
def test_dual_sources_follow_selected_metric_instead_of_original_diagonal_or_diameter(kind, points, key, expected):
    document, definition, result = _document_feature(kind, points, key)
    document.professional_measurements[0] = replace(definition, annotation_layout=replace(definition.annotation_layout, leader_style="dual", offset_px=Point(40, -50)))
    transform = lambda p: QPointF(p.x * 2 + 7, p.y * 2 + 9)
    _image, labels = _draw(document, transform)
    assert labels[definition.id].source_points == tuple(transform(p) for p in expected)
    assert len(labels[definition.id].witness_lines) == 2
    assert evaluate_document(document)[definition.id].metrics == result.metrics
