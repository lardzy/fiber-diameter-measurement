from __future__ import annotations

from types import SimpleNamespace

from PySide6.QtGui import QImage

from fdm.construction_geometry import ConstructionEntity, LineDefinition, LiveFeatureRef
from fdm.geometry import Point
from fdm.history import DocumentChangeImpact, DocumentHistoryState
from fdm.metrology import MeasurementDefinition, MeasurementFeature
from fdm.models import ImageDocument
from fdm.services.metrology_service import evaluate_document
from fdm.ui.canvas import DocumentCanvas
from fdm.ui.professional_measurement import ProfessionalWorkspace


def test_freeze_selected_measurement_keeps_other_shared_dimensions_live():
    document = ImageDocument("image", "", (100, 100))
    document.initialize_runtime_state()
    source = ConstructionEntity("source", "baseline", LineDefinition(Point(0, 0), Point(3, 4)))
    document.add_construction_entity(source)
    feature = MeasurementFeature("shared", document.id, "line", source_ref=LiveFeatureRef(document.id, source.id))
    first = MeasurementDefinition("first", document.id, "distance", (feature.id,))
    second = MeasurementDefinition("second", document.id, "distance", (feature.id,))
    document.apply_professional_state([feature], [first, second], selected_id=first.id)

    def change(doc, label, mutator):
        before, before_stamp = DocumentHistoryState.capture(doc), doc.state_stamp
        mutator()
        doc.mark_session_dirty()
        doc.history.push_delta(label, before=before, after=DocumentHistoryState.capture(doc), before_stamp=before_stamp, after_stamp=doc.state_stamp, impact=DocumentChangeImpact.SESSION)

    def error(message):
        raise AssertionError(message)

    workspace = SimpleNamespace(selected=lambda: (document, document.get_professional_measurement(first.id)), change=change, message=error)
    ProfessionalWorkspace.freeze_selected(workspace)
    frozen = document.get_professional_feature(document.get_professional_measurement(first.id).feature_ids[0])
    live = document.get_professional_feature(document.get_professional_measurement(second.id).feature_ids[0])
    assert frozen.id != live.id
    assert frozen.source_ref is None
    assert live.source_ref == feature.source_ref
    assert frozen.parameters["frozen_source"]["object_id"] == source.id
    assert document.get_professional_measurement(first.id).annotation_layout == first.annotation_layout

    change(document, "move source", lambda: document.replace_construction_entity(source.id, ConstructionEntity("temp", "moved", LineDefinition(Point(0, 0), Point(9, 12))), mark_dirty=False))
    results = evaluate_document(document)
    assert results[first.id].primary.value == 5
    assert results[second.id].primary.value == 15
    assert document.history.undo(document)
    assert evaluate_document(document)[second.id].primary.value == 5
    assert document.history.undo(document)
    assert document.get_professional_measurement(first.id).feature_ids == (feature.id,)
    assert document.get_professional_measurement(second.id).feature_ids == (feature.id,)
    assert document.professional_features == [feature]
    assert document.history.redo(document)
    assert document.get_professional_measurement(first.id).feature_ids != (feature.id,)


def test_professional_mode_object_snap_is_enabled_and_returns_baseline_endpoint():
    document = ImageDocument("image", "", (100, 100))
    document.initialize_runtime_state()
    document.add_construction_entity(ConstructionEntity("line", "baseline", LineDefinition(Point(20, 20), Point(80, 20))))
    image = QImage(100, 100, QImage.Format.Format_RGB32)
    image.fill(0xff888888)
    canvas = DocumentCanvas()
    try:
        canvas.resize(500, 500)
        canvas.set_document(document, image)
        canvas.set_tool_mode("professional")
        assert canvas._object_snap_pointer_enabled()
        candidate = canvas._query_object_snap(Point(20.05, 20.05))
        assert candidate is not None
        assert candidate.point_px == Point(20, 20)
    finally:
        canvas.close()
