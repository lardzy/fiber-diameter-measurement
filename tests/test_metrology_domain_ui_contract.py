"""Contracts across extraction, persisted metrology, and independent exports."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from fdm.geometry import Line, Point
from fdm.metrology import MeasurementDefinition, MeasurementFeature
from fdm.models import Calibration, ImageDocument, Measurement
from fdm.services.metrology_fitting import CaliperOptions, run_circle_caliper, run_line_caliper
from fdm.services.metrology_service import evaluate_document, resolve_feature
from fdm.ui.professional_measurement import export_professional_results


@pytest.mark.parametrize("kind", ["caliper_line", "caliper_circle", "caliper_width"])
def test_caliper_evidence_domain_roundtrip_and_original_pixels(kind):
    y, x = np.mgrid[:160, :180]
    edge = lambda positions, location: (0.5 + 0.5 * np.tanh((positions - location) / 0.8)).astype(np.float32)
    image = edge(y, 70) if kind == "caliper_line" else edge(np.hypot(x - 90, y - 80), 40)
    options = CaliperOptions(search_half_width_px=20, robust=False)
    if kind == "caliper_width":
        image = np.clip(edge(y, 60) - edge(y, 82), 0, 1)
    before = hashlib.sha256(image.tobytes()).hexdigest()
    extraction = run_circle_caliper(image, (90, 80), 40, options=options) if kind == "caliper_circle" else run_line_caliper(image, (20, 70), (160, 70), options=options, double_edge=kind == "caliper_width")
    assert extraction.success, extraction.reason
    points = [Point(*p) for section in extraction.sections if section.accepted for p in (section.first_px, section.second_px)] if kind == "caliper_width" else [Point(*p) for p in extraction.fit.points_px]
    doc = ImageDocument("image", "", (180, 160))
    feature = MeasurementFeature("feature", doc.id, kind, tuple(points), parameters={"caliper": options.to_dict()}, quality=extraction.to_dict())
    definition = MeasurementDefinition("M1", doc.id, {"caliper_line": "fit_line", "caliper_circle": "fit_circle", "caliper_width": "width"}[kind], (feature.id,))
    doc.apply_professional_state([feature], [definition])
    loaded = ImageDocument.from_dict(json.loads(json.dumps(doc.to_dict(), allow_nan=False)))
    result = evaluate_document(loaded)[definition.id]
    assert result.valid, result.reason
    assert hashlib.sha256(image.tobytes()).hexdigest() == before
    if kind == "caliper_width":
        assert result.primary.value == pytest.approx(extraction.statistics["mean"])
        assert next(metric.value for metric in result.metrics if metric.key == "stddev") == pytest.approx(extraction.statistics["stddev"])
    else:
        quality = result.quality["features"][feature.id]
        assert quality["source_evidence"] == feature.quality
        assert quality["evaluated_fit"]["rejected_indices"] == ()
        if kind == "caliper_circle":
            assert result.primary.value == pytest.approx(2 * extraction.fit.geometry["radius"])
    # Changing physical calibration refits the stored point cloud and keeps
    # the original acquisition evidence available under its original scale.
    loaded.calibration = Calibration("manual", .5, "µm", "changed", .5)
    changed = evaluate_document(loaded)[definition.id]
    assert changed.primary.value == pytest.approx(result.primary.value * 2)
    assert loaded.professional_features[0].quality == feature.quality


def test_failed_extraction_never_becomes_success_through_plain_point_fitting():
    doc = ImageDocument("image", "", (100, 100))
    feature = MeasurementFeature("bad", doc.id, "caliper_line", (Point(0, 0), Point(5, 0)), quality={"status": "insufficient_edges", "reason": "有效点不足"})
    assert not resolve_feature(doc, feature).valid
    assert "有效点不足" in resolve_feature(doc, feature).error


def test_calibration_change_rechecks_caliper_fit_quality_in_current_physical_plane():
    doc = ImageDocument("image", "", (100, 100))
    points = tuple(Point(50 + 20 * np.cos(angle), 50 + 20 * np.sin(angle)) for angle in np.linspace(0, 2 * np.pi, 64, endpoint=False))
    feature = MeasurementFeature("circle", doc.id, "caliper_circle", points,
        parameters={"caliper": CaliperOptions(robust=False, max_residual_px=1).to_dict()},
        quality={"status": "ready", "profiles": [{} for _ in points]})
    assert resolve_feature(doc, feature).valid
    doc.calibration = Calibration("manual", 0.5, "µm", "changed", 1)
    changed = resolve_feature(doc, feature)
    assert not changed.valid
    assert "残差" in changed.error
    assert changed.quality["source_evidence"]["status"] == "ready"
    assert changed.quality["evaluated_fit"]["max_residual"] > 1


@pytest.mark.parametrize("extension", ["csv", "xlsx"])
def test_independent_numeric_export_excludes_legacy_statistics_and_label_layout(tmp_path, extension):
    doc = ImageDocument("image", "", (100, 100))
    legacy = Measurement("legacy", doc.id, None, "manual", line_px=Line(Point(0, 0), Point(99, 0)))
    doc.add_measurement(legacy)
    feature = MeasurementFeature("f", doc.id, "line", (Point(0, 0), Point(3, 4)))
    definition = MeasurementDefinition("M1", doc.id, "distance", (feature.id,))
    doc.apply_professional_state([feature], [definition])
    before = tmp_path / f"before.{extension}"
    after = tmp_path / f"after.{extension}"
    export_professional_results(doc, evaluate_document(doc), before)
    doc.apply_professional_state([feature], [definition.clone(annotation_layout=definition.annotation_layout.with_manual_offset(Point(1000, 2000)))])
    export_professional_results(doc, evaluate_document(doc), after)
    read = pd.read_csv if extension == "csv" else pd.read_excel
    first, second = read(before), read(after)
    pd.testing.assert_frame_equal(first, second)
    assert first["测量ID"].tolist() == ["M1", "M1", "M1"]
    assert first["数值"].tolist() == [5, 3, 4]
    assert legacy.diameter_px == 99
