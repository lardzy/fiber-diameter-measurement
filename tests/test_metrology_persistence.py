from __future__ import annotations

from dataclasses import replace
import json

import pytest

from fdm.geometry import Line, Point
from fdm.history import DocumentChangeImpact, DocumentHistoryState
from fdm.measurement_annotation import MeasurementAnnotationLayout
from fdm.metrology import MeasurementDefinition, MeasurementFeature
from fdm.models import Calibration, ImageDocument, Measurement, ProjectState
from fdm.project_io import ProjectCompatibilityError, ProjectIO
from fdm.services.metrology_service import evaluate_document


def document():
    doc = ImageDocument("image", "", (640, 480))
    doc.initialize_runtime_state()
    return doc


def populate(doc):
    feature = MeasurementFeature("feature", doc.id, "line", (Point(1, 2), Point(4, 6)), parameters={"source_identity": {"focus": 2}}, focus_index=2,
        source_revision=doc.metadata.get("digital_slide", {}).get("stitching_revision") if doc.is_digital_slide() else None)
    definition = MeasurementDefinition("M1", doc.id, "distance", (feature.id,), focus_index=2)
    doc.apply_professional_state([feature], [definition], selected_id=definition.id)
    return feature, definition


def test_sparse_legacy_document_and_measurement_remain_compatible():
    doc = document()
    measurement = Measurement("legacy", doc.id, None, "manual", line_px=Line(Point(0, 0), Point(3, 4)))
    doc.add_measurement(measurement)
    payload = doc.to_dict()
    assert "professional_features" not in payload
    assert "professional_measurements" not in payload
    assert "annotation_layout" not in payload["measurements"][0]
    loaded = ImageDocument.from_dict(payload)
    assert loaded.professional_features == []
    assert loaded.measurements[0].annotation_layout is None
    assert ProjectState("test", [doc]).effective_required_features() == ()


def test_project_and_background_save_payload_roundtrip_and_detach_nested_parameters():
    doc = document()
    feature, definition = populate(doc)
    payload = ProjectIO.persistent_payload(ProjectState("test", [doc]))
    assert "professional-measurements/v1" in payload["required_features"]
    loaded = ProjectState.from_dict(payload)
    copy = loaded.documents[0]
    assert copy.professional_features == [feature]
    assert copy.professional_measurements == [definition]
    assert copy.selected_professional_id == definition.id
    assert evaluate_document(copy)[definition.id].primary.value == 5
    payload["documents"][0]["professional_features"][0]["parameters"]["source_identity"]["focus"] = 100
    assert feature.parameters["source_identity"]["focus"] == 2
    assert loaded.documents[0].professional_features[0].parameters["source_identity"]["focus"] == 2


def test_legacy_annotation_layout_persists_and_auto_declares_feature():
    doc = document()
    measurement = Measurement("legacy", doc.id, None, "manual", line_px=Line(Point(0, 0), Point(3, 4)), annotation_layout=MeasurementAnnotationLayout().with_manual_offset(Point(40, 10)))
    doc.add_measurement(measurement)
    payload = ProjectState("test", [doc]).to_dict()
    assert "measurement-annotations/v1" in payload["required_features"]
    copy = ProjectState.from_dict(payload).documents[0]
    assert copy.measurements[0].annotation_layout == measurement.annotation_layout


def test_delta_history_restores_professional_branch_selection_and_dirty_stamp():
    doc = document()
    before, before_stamp = DocumentHistoryState.capture(doc), doc.state_stamp
    feature, definition = populate(doc)
    after, after_stamp = DocumentHistoryState.capture(doc), doc.state_stamp
    assert doc.history.push_delta("专业测量", before=before, after=after, before_stamp=before_stamp, after_stamp=after_stamp, impact=DocumentChangeImpact.SESSION)
    assert doc.history.undo(doc)
    assert doc.professional_features == [] and doc.professional_measurements == []
    assert not doc.dirty_flags.session_dirty
    assert doc.history.redo(doc)
    assert doc.professional_features == [feature]
    assert doc.professional_measurements == [definition]
    assert doc.selected_professional_id == definition.id
    assert doc.dirty_flags.session_dirty


def test_annotation_only_undo_restores_positions_without_geometry_revision_or_value_changes():
    doc = document()
    feature, definition = populate(doc)
    old = Measurement("legacy", doc.id, None, "manual", line_px=Line(Point(0, 0), Point(3, 4)), annotation_layout=MeasurementAnnotationLayout())
    doc.add_measurement(old)
    before = DocumentHistoryState.capture(doc)
    old_geometry_revision = doc.measurement_geometry_revision
    professional_revision = doc.professional_geometry_revision
    result = evaluate_document(doc)[definition.id]
    moved = definition.clone(annotation_layout=definition.annotation_layout.with_manual_offset(Point(80, 90)))
    old.annotation_layout = old.annotation_layout.with_manual_offset(Point(-20, 30))
    doc.apply_professional_state([feature], [moved], selected_id=definition.id)
    after = DocumentHistoryState.capture(doc)
    assert before != after
    assert doc.professional_geometry_revision == professional_revision
    assert evaluate_document(doc)[definition.id] is result
    before.restore(doc)
    assert doc.professional_measurements[0].annotation_layout == definition.annotation_layout
    assert old.annotation_layout.offset_px is None
    after.restore(doc)
    assert doc.professional_measurements[0].annotation_layout.offset_px == Point(80, 90)
    assert old.annotation_layout.offset_px == Point(-20, 30)
    assert doc.measurement_geometry_revision == old_geometry_revision
    assert doc.professional_geometry_revision == professional_revision
    assert evaluate_document(doc)[definition.id] is result


def test_snapshot_restore_and_session_snapshot_include_complete_professional_data():
    doc = document()
    feature, definition = populate(doc)
    snapshot = doc.snapshot_state()
    assert doc.session_snapshot()["professional_features"] == [feature.to_dict()]
    doc.apply_professional_state([], [])
    doc.restore_snapshot(snapshot)
    assert doc.professional_features == [feature]
    assert doc.professional_measurements == [definition]
    assert doc.selected_professional_id == definition.id


@pytest.mark.parametrize("mutate", [
    lambda payload: payload["professional_features"].append(payload["professional_features"][0]),
    lambda payload: payload["professional_features"][0].update(image_id="wrong"),
    lambda payload: payload["professional_features"][0].update(kind="future-kind"),
    lambda payload: payload["professional_features"][0].update(points_px=[{"x": float("nan"), "y": 0}]),
    lambda payload: payload["professional_measurements"][0].update(focus_index=-1),
    lambda payload: payload["professional_measurements"][0].update(parameters={"invalid": float("inf")}),
])
def test_invalid_professional_payloads_are_rejected(mutate):
    doc = document()
    populate(doc)
    payload = doc.to_dict()
    mutate(payload)
    with pytest.raises((ValueError, TypeError)):
        ImageDocument.from_dict(payload)


def test_missing_feature_is_retained_as_invalid_and_unknown_version_read_only():
    doc = document()
    feature, definition = populate(doc)
    payload = ProjectState("test", [doc]).to_dict()
    payload["documents"][0]["professional_features"] = []
    payload["required_features"].append("professional-measurements/v999")
    loaded = ProjectState.from_dict(payload)
    assert loaded.is_read_only_compatible
    assert loaded.documents[0].professional_measurements[0].id == definition.id
    assert not evaluate_document(loaded.documents[0])[definition.id].valid


def test_calibration_recomputes_original_points_without_modifying_previous_snapshot():
    doc = document()
    feature, definition = populate(doc)
    before = DocumentHistoryState.capture(doc)
    doc.calibration = Calibration("manual", 0.5, "µm", "test", 1)
    assert evaluate_document(doc)[definition.id].primary.value == pytest.approx((36 + 16) ** 0.5)
    assert doc.professional_features[0].points_px == feature.points_px
    before.restore(doc)
    assert evaluate_document(doc)[definition.id].primary.value == 5
    assert doc.professional_features[0].points_px == feature.points_px


def test_selections_are_exclusive_and_nonexistent_professional_selection_is_cleared():
    doc = document()
    _, definition = populate(doc)
    doc.select_measurement("legacy")
    assert doc.selected_professional_id is None
    doc.select_professional(definition.id)
    assert doc.view_state.selected_measurement_id is None
    payload = doc.to_dict()
    payload["selected_professional_id"] = "missing"
    assert ImageDocument.from_dict(payload).selected_professional_id is None


@pytest.mark.parametrize("kind", ["image", "digital_slide"])
def test_actual_project_file_roundtrip_with_source_layer_and_layout(tmp_path, kind):
    doc = document()
    doc.document_kind = kind
    doc.path = str(tmp_path / ("原图.png" if kind == "image" else "切片.fdmslide"))
    doc.metadata = {"digital_slide": {"focus_index": 2, "stitching_revision": "registration-v1"}} if kind == "digital_slide" else {}
    feature, definition = populate(doc)
    moved = definition.clone(annotation_layout=definition.annotation_layout.with_manual_offset(Point(300, -120)))
    doc.apply_professional_state([feature], [moved], selected_id=moved.id)
    path = tmp_path / "专业测量.fdmproj"
    ProjectIO.save(ProjectState("test", [doc]), path)
    loaded = ProjectIO.load(path)
    restored = loaded.documents[0]
    assert restored.document_kind == kind
    assert restored.professional_features == [feature]
    assert restored.professional_measurements == [moved]
    assert restored.professional_measurements[0].focus_index == 2
    assert evaluate_document(restored)[moved.id].primary.value == 5
    assert restored.measurements == []
    assert "professional-measurements/v1" in loaded.required_features


def test_older_reader_required_features_block_original_file_overwrite(tmp_path, monkeypatch):
    import fdm.models as models

    doc = document()
    populate(doc)
    source = tmp_path / "professional.fdmproj"
    ProjectIO.save(ProjectState("test", [doc]), source)
    original = source.read_bytes()
    monkeypatch.setattr(models, "SUPPORTED_PROJECT_REQUIRED_FEATURES", frozenset(models.SUPPORTED_PROJECT_REQUIRED_FEATURES - {"professional-measurements/v1", "measurement-annotations/v1"}))
    older_reader = ProjectIO.load(source)
    assert older_reader.compatibility.read_only
    with pytest.raises(ProjectCompatibilityError, match="不能覆盖"):
        ProjectIO.save(older_reader, source)
    assert source.read_bytes() == original


def test_legacy_insert_undo_restores_professional_selection():
    doc = document()
    _, definition = populate(doc)
    before_stamp = doc.state_stamp
    previous = (None, None, None, definition.id)
    old = Measurement("legacy", doc.id, None, "manual", line_px=Line(Point(0, 0), Point(3, 4)))
    doc.add_measurement(old)
    doc.history.push_measurement_insert(old, index=0, previous_selection=previous, before_stamp=before_stamp, after_stamp=doc.state_stamp)
    assert doc.selected_professional_id is None
    assert doc.history.undo(doc)
    assert doc.selected_professional_id == definition.id
    assert doc.history.redo(doc)
    assert doc.selected_professional_id is None
    assert doc.view_state.selected_measurement_id == old.id


def test_shared_feature_deletion_and_undo_preserve_dependencies_without_stale_success():
    doc = document()
    feature, first = populate(doc)
    second = MeasurementDefinition("M2", doc.id, "distance", (feature.id,), parameters={"metric": "dx"})
    doc.apply_professional_state([feature], [first, second])
    before = DocumentHistoryState.capture(doc)
    doc.apply_professional_state([], [first, second])
    assert all(not result.valid for result in evaluate_document(doc).values())
    before.restore(doc)
    results = evaluate_document(doc)
    assert results[first.id].primary.value == 5
    assert results[second.id].primary.value == 3
    doc.apply_professional_state([feature.clone(points_px=(Point(1, 2), Point(10, 14)))], [first, second])
    results = evaluate_document(doc)
    assert results[first.id].primary.value == 15
    assert results[second.id].primary.value == 9


@pytest.mark.parametrize("metric_key", [None, "distance", "dx", "radius", "section_12"])
def test_annotation_metric_key_roundtrips_project_and_history(metric_key):
    doc = document()
    feature, definition = populate(doc)
    layout = replace(definition.annotation_layout, metric_key=metric_key)
    doc.apply_professional_state([feature], [definition.clone(annotation_layout=layout)])
    payload = ProjectState("test", [doc]).to_dict()
    restored = ProjectState.from_dict(payload).documents[0]
    assert restored.professional_measurements[0].annotation_layout.metric_key == metric_key
    state = DocumentHistoryState.capture(restored)
    restored.apply_professional_state([], [])
    state.restore(restored)
    assert restored.professional_measurements[0].annotation_layout.metric_key == metric_key


@pytest.mark.parametrize("metric_key", ["", "   ", 4, True, {}, []])
def test_annotation_metric_key_rejects_wrong_type_or_empty_values(metric_key):
    doc = document()
    populate(doc)
    payload = doc.to_dict()
    payload["professional_measurements"][0]["annotation_layout"]["metric_key"] = metric_key
    with pytest.raises((TypeError, ValueError)):
        ImageDocument.from_dict(payload)


def test_switching_label_metric_changes_caption_only_and_never_evaluates_geometry(monkeypatch):
    from fdm.services import metrology_service
    from fdm.ui.professional_rendering import result_caption

    doc = document()
    feature, definition = populate(doc)
    first = evaluate_document(doc)[definition.id]
    revision = doc.professional_geometry_revision
    before = DocumentHistoryState.capture(doc)
    def forbidden(*_args, **_kwargs):
        raise AssertionError("Changing label metric must not resolve or fit geometry")
    monkeypatch.setattr(metrology_service, "resolve_feature", forbidden)
    changed = definition.clone(annotation_layout=replace(definition.annotation_layout, metric_key="dx"))
    doc.apply_professional_state([feature], [changed])
    current = evaluate_document(doc)[definition.id]
    assert current is first
    assert current.primary.value == 5
    assert "水平分量 3 px" in result_caption(changed, current, 1)
    assert doc.professional_geometry_revision == revision
    before.restore(doc)
    assert evaluate_document(doc)[definition.id] is first
    assert doc.professional_measurements[0].annotation_layout.metric_key is None


@pytest.mark.parametrize("branch", ["legacy", "professional"])
@pytest.mark.parametrize("style", ["single", "dual"])
def test_dual_dimension_feature_is_derived_only_from_dual_layouts_and_roundtrips(tmp_path, branch, style):
    doc = document()
    layout = MeasurementAnnotationLayout(leader_style=style).with_manual_offset(Point(30, -40))
    if branch == "legacy":
        doc.add_measurement(Measurement("legacy", doc.id, None, "manual", line_px=Line(Point(0, 0), Point(3, 4)), annotation_layout=layout))
    else:
        feature, definition = populate(doc)
        doc.apply_professional_state([feature], [definition.clone(annotation_layout=layout)])
    project = ProjectState("test", [doc])
    required = project.effective_required_features()
    assert ("measurement-dual-dimensions/v1" in required) is (style == "dual")
    assert ("measurement-annotations/v1" if branch == "legacy" else "professional-measurements/v1") in required
    path = tmp_path / f"{branch}-{style}.fdmproj"
    ProjectIO.save(project, path)
    restored = ProjectIO.load(path)
    restored_doc = restored.documents[0]
    record = restored_doc.measurements[0] if branch == "legacy" else restored_doc.professional_measurements[0]
    assert record.annotation_layout == layout
    assert not restored.is_read_only_compatible
    assert ("measurement-dual-dimensions/v1" in restored.required_features) is (style == "dual")


@pytest.mark.parametrize("branch", ["legacy", "professional"])
def test_reader_without_dual_dimension_support_cannot_overwrite_source(tmp_path, monkeypatch, branch):
    import fdm.models as models

    doc = document()
    layout = MeasurementAnnotationLayout(leader_style="dual")
    if branch == "legacy":
        doc.add_measurement(Measurement("legacy", doc.id, None, "manual", line_px=Line(Point(0, 0), Point(3, 4)), annotation_layout=layout))
    else:
        feature, definition = populate(doc)
        doc.apply_professional_state([feature], [definition.clone(annotation_layout=layout)])
    path = tmp_path / f"dual-{branch}.fdmproj"
    ProjectIO.save(ProjectState("test", [doc]), path)
    before = path.read_bytes()
    # This reader still supports the original annotation/metrology fields;
    # only the independently introduced dual endpoint layout is unsupported.
    monkeypatch.setattr(models, "SUPPORTED_PROJECT_REQUIRED_FEATURES", models.SUPPORTED_PROJECT_REQUIRED_FEATURES - {"measurement-dual-dimensions/v1"})
    older_reader = ProjectIO.load(path)
    assert older_reader.compatibility.unknown_required_features == ("measurement-dual-dimensions/v1",)
    assert older_reader.is_read_only_compatible
    with pytest.raises(ProjectCompatibilityError, match="measurement-dual-dimensions/v1"):
        ProjectIO.save(older_reader, path)
    assert path.read_bytes() == before


def test_dual_layout_does_not_serialize_runtime_source_verification_state(tmp_path):
    doc = document()
    feature, definition = populate(doc)
    doc.apply_professional_state([feature], [definition.clone(annotation_layout=MeasurementAnnotationLayout(leader_style="dual"))])
    doc.professional_source_revision = "runtime-only-current-source"
    doc.professional_source_revision_pending = True
    doc.professional_source_revision_error = "runtime-only-read-error"
    runtime_names = {"professional_source_revision", "professional_source_revision_pending", "professional_source_revision_error"}
    for payload in (doc.to_dict(), doc.session_snapshot(), doc.snapshot_state()):
        assert not runtime_names.intersection(payload)
    path = tmp_path / "dual-no-runtime.fdmproj"
    ProjectIO.save(ProjectState("test", [doc]), path)
    serialized = path.read_text(encoding="utf-8")
    assert "runtime-only-current-source" not in serialized
    assert "runtime-only-read-error" not in serialized
    restored = ProjectIO.load(path).documents[0]
    assert restored.professional_source_revision is None
    assert restored.professional_source_revision_pending is False
    assert restored.professional_source_revision_error is None
    assert restored.professional_measurements[0].annotation_layout.leader_style == "dual"
