from __future__ import annotations

import math

import pytest

from fdm.construction_geometry import CircleCenterRadiusDefinition, ConstructionEntity, LineDefinition, LiveFeatureRef
from fdm.geometry import Point
from fdm.metrology import MeasurementDefinition, MeasurementFeature
from fdm.models import Calibration, ImageDocument
from fdm.services.metrology_service import evaluate_document, evaluate_measurement, freeze_feature, resolve_feature


def document(*, scale=(1.0, 1.0), calibrated=True):
    doc = ImageDocument("image", "", (2000, 2000))
    if calibrated:
        doc.calibration = Calibration("manual", 1 / scale[0], "µm", "test", 1 / scale[1])
    doc.initialize_runtime_state()
    return doc


def feature(doc, key, kind, *points, **kwargs):
    item = MeasurementFeature(key, doc.id, kind, tuple(Point(*p) for p in points), **kwargs)
    doc.professional_features.append(item)
    return item


def measure(doc, kind, *features, **params):
    definition = MeasurementDefinition("m", doc.id, kind, tuple(item.id for item in features), parameters=params)
    return evaluate_measurement(doc, definition)


def values(result):
    assert result.valid, result.reason
    return {item.key: item.value for item in result.metrics}


def circle(doc, key, center, radius):
    x, y = center
    return feature(doc, key, "circle", (x + radius, y), (x, y + radius), (x - radius, y))


def test_anisotropic_distance_components_and_no_legacy_statistics():
    doc = document(scale=(2, 3))
    line = feature(doc, "line", "line", (0, 0), (3, 4))
    result = measure(doc, "distance", line)
    assert values(result) == pytest.approx({"distance": math.sqrt(180), "dx": 6, "dy": 12})
    assert result.metrics[0].unit == "µm"
    assert doc.measurements == []
    assert not doc.dirty_flags.session_dirty
    assert not doc.dirty_flags.calibration_dirty


def test_point_infinite_line_and_segment_have_different_witnesses():
    doc = document()
    point = feature(doc, "point", "point", (12, 4))
    line = feature(doc, "line", "line", (0, 0), (10, 0))
    assert values(measure(doc, "point_line_distance", point, line))["distance"] == 4
    segment = measure(doc, "point_segment_distance", point, line)
    assert values(segment)["distance"] == pytest.approx(math.sqrt(20))
    assert segment.geometry[-1].points_px[-1] == Point(10, 0)


def test_parallel_spacing_rejects_nonparallel_lines_and_uses_physical_plane():
    doc = document(scale=(2, 3))
    a = feature(doc, "a", "line", (0, 0), (10, 0))
    b = feature(doc, "b", "line", (0, 2), (10, 2))
    c = feature(doc, "c", "line", (0, 2), (10, 3))
    assert values(measure(doc, "parallel_distance", a, b))["distance"] == 6
    result = measure(doc, "parallel_distance", a, c)
    assert not result.valid and "不平行" in result.reason


def test_projection_and_relative_coordinates_use_reference_direction():
    doc = document()
    p = feature(doc, "p", "point", (3, 4))
    origin = feature(doc, "o", "point", (1, 1))
    baseline = feature(doc, "base", "line", (0, 0), (0, 5))
    line = feature(doc, "line", "line", (1, 1), (3, 4))
    assert values(measure(doc, "projection", line, baseline))["projection"] == 3
    assert values(measure(doc, "coordinates", p, origin, baseline)) == {"x": 3, "y": -2}


def test_angles_anisotropy_and_supplement_have_matching_arc_domain():
    doc = document(scale=(2, 1))
    points = feature(doc, "angle", "polyline", (1, 0), (0, 0), (1, 1))
    expected = math.degrees(math.atan(0.5))
    assert values(measure(doc, "angle", points))["angle"] == pytest.approx(expected)
    supplement = measure(doc, "angle", points, supplement=True)
    assert values(supplement)["angle"] == pytest.approx(180 - expected)
    assert abs(math.degrees(supplement.quality["angle_domain"]["sweep_rad"])) == pytest.approx(180 - expected)
    a = feature(doc, "a", "line", (0, 0), (1, 0))
    b = feature(doc, "b", "line", (0, 0), (-1, -1))
    assert values(measure(doc, "line_angle", a, b))["angle"] == pytest.approx(expected)


def test_physical_circle_maps_back_to_ellipse_and_rejects_collinear():
    doc = document(scale=(2, 1))
    c = feature(doc, "c", "circle", (5, 0), (0, 10), (-5, 0))
    result = measure(doc, "circle", c)
    assert values(result)["diameter"] == pytest.approx(20)
    curve = result.geometry[0].points_px
    assert max(p.x for p in curve) == pytest.approx(5)
    assert max(p.y for p in curve) == pytest.approx(10)
    degenerate = feature(doc, "bad", "circle", (1, 1), (2, 2), (3, 3))
    assert "共线" in measure(doc, "circle", degenerate).reason


def test_arc_uses_middle_point_to_select_long_arc():
    doc = document()
    arc = feature(doc, "arc", "arc", (1, 0), (0, -1), (0, 1))
    result = measure(doc, "arc", arc)
    assert values(result) == pytest.approx({"arc_length": 1.5 * math.pi, "radius": 1, "angle": 270})
    assert result.quality["arc_sweep_rad"] == pytest.approx(-1.5 * math.pi)


@pytest.mark.parametrize("center,radius,expected", [((10, 0), 2, 5), ((2, 0), 2, 0), ((1, 0), 1, 1), ((0, 0), 1, 2), ((0, 0), 3, 0)])
def test_circle_boundary_gap_external_intersection_containment(center, radius, expected):
    doc = document()
    first, second = circle(doc, "a", (0, 0), 3), circle(doc, "b", center, radius)
    result = measure(doc, "circle_gap", first, second)
    assert values(result)["distance"] == pytest.approx(expected)
    witness = result.geometry[-1].points_px
    assert math.dist((witness[0].x, witness[0].y), (witness[1].x, witness[1].y)) == pytest.approx(expected)
    assert values(measure(doc, "circle_center_distance", first, second))["distance"] == pytest.approx(math.hypot(*center))


def test_point_circle_distance_is_boundary_not_center_distance():
    doc = document()
    point = feature(doc, "p", "point", (0, 0))
    c = circle(doc, "c", (0, 0), 3)
    assert values(measure(doc, "point_circle_distance", point, c))["distance"] == 3


def test_polygon_area_perimeter_self_intersection_and_large_origin():
    doc = document(scale=(2, 3))
    origin = 1e9
    rectangle = feature(doc, "rectangle", "polygon", (origin, origin), (origin + 2, origin), (origin + 2, origin + 3), (origin, origin + 3))
    assert values(measure(doc, "polygon", rectangle)) == pytest.approx({"area": 36, "perimeter": 26})
    crossing = feature(doc, "crossing", "polygon", (0, 0), (4, 4), (0, 4), (4, 0))
    assert "自交" in measure(doc, "polygon", crossing).reason
    assert measure(doc, "polygon", rectangle).metrics[0].unit == "µm²"


def test_polyline_and_multi_section_widths_keep_each_section():
    doc = document(calibrated=False)
    line = feature(doc, "p", "polyline", (0, 0), (3, 0), (3, 4))
    assert values(measure(doc, "polyline", line))["length"] == 7
    width = feature(doc, "w", "caliper_width", (0, 0), (3, 0), (0, 1), (5, 1))
    result = measure(doc, "width", width)
    assert values(result) == pytest.approx({"mean": 4, "min": 3, "max": 5, "stddev": math.sqrt(2), "count": 2, "section_1": 3, "section_2": 5})
    assert result.quality["stddev_basis"] == "sample_ddof_1"
    assert result.metrics[0].unit == "px"


def test_live_reference_recompute_delete_and_explicit_freeze():
    doc = document()
    source = ConstructionEntity("source", "baseline", LineDefinition(Point(0, 0), Point(5, 0)))
    doc.add_construction_entity(source)
    ref = MeasurementFeature("ref", doc.id, "line", source_ref=LiveFeatureRef(doc.id, "source"))
    doc.professional_features = [ref]
    definition = MeasurementDefinition("m", doc.id, "distance", (ref.id,))
    doc.professional_measurements = [definition]
    assert values(evaluate_document(doc)["m"])["distance"] == 5
    frozen = freeze_feature(doc, ref)
    doc.replace_construction_entity("source", ConstructionEntity("other", "new", LineDefinition(Point(0, 0), Point(9, 0))))
    assert values(evaluate_document(doc)["m"])["distance"] == 9
    assert frozen.source_ref is None and frozen.parameters["frozen_source"]["object_id"] == "source"
    doc.remove_construction_entity("source")
    assert not evaluate_document(doc)["m"].valid
    doc.professional_features = [frozen]
    assert values(evaluate_document(doc)["m"])["distance"] == 5


def test_old_pixel_circle_rejected_under_anisotropic_calibration_even_frozen():
    doc = document()
    doc.add_construction_entity(ConstructionEntity("circle", "circle", CircleCenterRadiusDefinition(Point(0, 0), 5)))
    ref = MeasurementFeature("ref", doc.id, "circle", source_ref=LiveFeatureRef(doc.id, "circle"))
    frozen = freeze_feature(doc, ref)
    doc.calibration = Calibration("manual", 1, "µm", "test", 2)
    assert "唯一物理半径" in resolve_feature(doc, ref).error
    assert "唯一物理半径" in resolve_feature(doc, frozen).error


def test_focus_and_source_revision_are_validated_without_using_current_layer():
    doc = document()
    a = feature(doc, "a", "point", (0, 0), focus_index=2)
    b = feature(doc, "b", "point", (0, 2), focus_index=3)
    assert "不同焦层" in measure(doc, "distance", a, b).reason
    c = feature(doc, "c", "line", (0, 0), (0, 3), source_revision="old")
    doc.metadata["digital_slide"] = {"stitching_revision": "new"}
    assert "版本已改变" in measure(doc, "distance", c).reason


def test_fitted_line_and_circle_follow_calibration_and_preserve_evidence():
    doc = document(scale=(2, 1))
    line = feature(doc, "line", "fit_line", (0, 0), (1, 2), (2, 4), (3, 6))
    result = measure(doc, "fit_line", line)
    assert values(result)["length"] == pytest.approx(math.sqrt(72))
    points = tuple((5 * math.cos(t), 10 * math.sin(t)) for t in [i * math.pi / 8 for i in range(16)])
    fitted_circle = feature(doc, "circle", "fit_circle", *points)
    result = measure(doc, "fit_circle", fitted_circle)
    assert values(result)["diameter"] == pytest.approx(20, abs=1e-6)
    assert len(result.quality["features"]["circle"]["evaluated_fit"]["accepted_indices"]) == 16


def test_digital_slide_missing_source_layer_is_invalid_instead_of_visible_everywhere():
    doc = document()
    doc.document_kind = "digital_slide"
    doc.professional_source_revision = "verified-test-source"
    line = feature(doc, "line", "line", (0, 0), (3, 4), source_revision="verified-test-source")
    result = measure(doc, "distance", line)
    assert not result.valid and "来源焦层" in result.reason
