from __future__ import annotations

from dataclasses import replace

from fdm.construction_geometry import (
    ConstructionEntity,
    FreePointDefinition,
    LineDefinition,
    LineExtent,
)
from fdm.geometry import Point
from fdm.history import (
    ConstructionHistoryPayload,
    DocumentChangeImpact,
    DocumentDeltaCommand,
    DocumentHistoryState,
    _shared_construction_payload_bytes,
    dirty_domains_for_impact,
)
from fdm.models import (
    PROJECT_SCHEMA_VERSION,
    DirtyDomain,
    ImageDocument,
    ProjectState,
)


def _document(document_id: str = "doc") -> ImageDocument:
    document = ImageDocument(
        id=document_id,
        path=f"/tmp/{document_id}.png",
        image_size=(320, 240),
    )
    document.initialize_runtime_state()
    return document


def _line(entity_id: str, *, y: float = 0.0) -> ConstructionEntity:
    return ConstructionEntity(
        id=entity_id,
        name=f"辅助线 {entity_id}",
        definition=LineDefinition(
            Point(0.0, y),
            Point(10.0, y),
            LineExtent.SEGMENT,
        ),
    )


def test_legacy_document_defaults_to_empty_sparse_construction_layer() -> None:
    document = _document()

    payload = document.to_dict()

    assert "construction_entities" not in payload
    assert "selected_construction_id" not in payload
    loaded = ImageDocument.from_dict(payload)
    assert loaded.construction_entities == []
    assert loaded.selected_construction_id is None
    assert loaded.construction_geometry_revision == 0

    legacy_project = ProjectState.from_dict(
        {"version": "legacy", "documents": [payload]}
    )
    assert legacy_project.documents[0].construction_entities == []
    project_payload = ProjectState(
        version="test",
        documents=[legacy_project.documents[0]],
    ).to_dict()
    assert project_payload["project_schema_version"] == PROJECT_SCHEMA_VERSION == 2
    assert "construction-geometry/v1" not in project_payload["required_features"]


def test_construction_roundtrip_auto_declares_supported_required_feature() -> None:
    document = _document()
    entity = _line("c1")
    document.add_construction_entity(entity)
    project = ProjectState(version="test", documents=[document])

    payload = project.to_dict()

    assert payload["project_schema_version"] == 2
    assert payload["required_features"] == ["construction-geometry/v1"]
    document_payload = payload["documents"][0]
    assert document_payload["construction_entities"] == [entity.to_dict()]
    assert document_payload["selected_construction_id"] == entity.id

    loaded = ProjectState.from_dict(payload)
    loaded_document = loaded.documents[0]
    assert not loaded.is_read_only_compatible
    assert loaded.required_features == ("construction-geometry/v1",)
    assert [item.to_dict() for item in loaded_document.construction_entities] == [
        entity.to_dict()
    ]
    assert loaded_document.selected_construction_id == entity.id


def test_construction_selection_is_validated_and_mutually_exclusive() -> None:
    document = _document()
    entity = _line("c1")
    document.add_construction_entity(entity)
    document_payload = document.to_dict()
    document_payload["selected_construction_id"] = "missing"

    loaded = ImageDocument.from_dict(document_payload)

    assert loaded.selected_construction_id is None
    loaded.select_construction(entity.id)
    assert loaded.selected_construction_id == entity.id
    assert loaded.view_state.selected_measurement_id is None
    assert loaded.selected_overlay_id is None


def test_construction_mutations_advance_revision_dirty_state_and_selection() -> None:
    document = _document()
    entity = _line("c1")
    assert document.construction_geometry_revision == 0
    assert not document.dirty_flags.session_dirty

    document.add_construction_entity(entity)

    assert document.construction_geometry_revision == 1
    assert document.selected_construction_id == entity.id
    assert document.dirty_flags.session_dirty

    assert document.remove_construction_entity(entity.id)
    assert document.construction_geometry_revision == 2
    assert document.selected_construction_id is None
    assert document.construction_entities == []


def test_replace_construction_entity_preserves_id_and_bumps_revisions() -> None:
    document = _document()
    original = _line("stable")
    document.add_construction_entity(original)
    document.mark_session_saved()
    document_revision = document.construction_geometry_revision
    replacement = _line("temporary", y=8.0)

    assert document.replace_construction_entity(original.id, replacement)

    stored = document.get_construction_entity(original.id)
    assert stored is not None
    assert stored.id == original.id
    assert stored.name == replacement.name
    assert stored.revision == original.revision + 1
    assert document.construction_geometry_revision == document_revision + 1
    assert document.selected_construction_id == original.id
    assert document.dirty_flags.session_dirty


def test_metadata_restore_does_not_advance_construction_geometry_revision() -> None:
    document = _document()
    entity = _line("metadata-only")
    document.add_construction_entity(entity)
    before = DocumentHistoryState.capture(document)
    geometry_revision = document.construction_geometry_revision
    metadata_revision = document.construction_metadata_revision

    assert document.replace_construction_entity(
        entity.id,
        replace(entity, visible=False),
        mark_dirty=False,
    )
    after = DocumentHistoryState.capture(document)
    assert document.construction_geometry_revision == geometry_revision
    assert document.construction_metadata_revision == metadata_revision + 1

    before.restore(document)
    assert document.get_construction_entity(entity.id).visible is True
    assert document.construction_geometry_revision == geometry_revision
    undo_metadata_revision = document.construction_metadata_revision

    after.restore(document)
    assert document.get_construction_entity(entity.id).visible is False
    assert document.construction_geometry_revision == geometry_revision
    assert document.construction_metadata_revision == undo_metadata_revision + 1


def test_snapshot_restore_restores_construction_geometry_and_valid_selection() -> None:
    document = _document()
    first = _line("first")
    second = _line("second", y=5.0)
    document.add_construction_entity(first)
    snapshot = document.snapshot_state()
    revision = document.construction_geometry_revision

    document.add_construction_entity(second)
    document.select_construction(second.id)
    document.restore_snapshot(snapshot)

    assert [entity.id for entity in document.construction_entities] == [first.id]
    assert document.selected_construction_id == first.id
    assert document.construction_geometry_revision > revision


def test_construction_delta_history_restores_entities_selection_and_revision() -> None:
    document = _document()
    before_stamp = document.state_stamp
    before = DocumentHistoryState.capture(document)
    entity = _line("history")
    document.add_construction_entity(entity)
    after = DocumentHistoryState.capture(document)
    after_stamp = document.state_stamp
    revision_after_add = document.construction_geometry_revision

    assert dirty_domains_for_impact(DocumentChangeImpact.CONSTRUCTION) == frozenset(
        {DirtyDomain.SESSION}
    )
    assert after.estimated_bytes >= len(after.construction_payload)
    assert document.history.push_delta(
        "新增辅助线",
        before=before,
        after=after,
        before_stamp=before_stamp,
        after_stamp=after_stamp,
        impact=DocumentChangeImpact.CONSTRUCTION,
    )

    assert document.history.undo(document)
    assert document.construction_entities == []
    assert document.selected_construction_id is None
    assert document.construction_geometry_revision > revision_after_add
    assert document.history.last_affected_domains == frozenset({DirtyDomain.SESSION})

    undo_revision = document.construction_geometry_revision
    assert document.history.redo(document)
    assert [item.to_dict() for item in document.construction_entities] == [
        entity.to_dict()
    ]
    assert document.selected_construction_id == entity.id
    assert document.construction_geometry_revision > undo_revision


def test_non_construction_history_reuses_large_construction_payload_and_budget(
    monkeypatch,
) -> None:
    entities = [_line(f"cached-{index}", y=float(index)) for index in range(1024)]
    document = ImageDocument(
        id="large-construction-history-cache",
        path="/tmp/large-construction-history-cache.png",
        image_size=(320, 240),
        construction_entities=entities,
    )
    document.initialize_runtime_state()
    original_to_dict = ConstructionEntity.to_dict
    serialization_calls = 0

    def counted_to_dict(entity: ConstructionEntity) -> dict[str, object]:
        nonlocal serialization_calls
        serialization_calls += 1
        return original_to_dict(entity)

    monkeypatch.setattr(ConstructionEntity, "to_dict", counted_to_dict)

    first_before_stamp = document.state_stamp
    first_before = DocumentHistoryState.capture(document)
    document.metadata["ordinary-edit"] = 1
    first_after = DocumentHistoryState.capture(document)
    first_after_stamp = document.advance_state({DirtyDomain.SESSION})
    assert document.history.push_delta(
        "普通命令一",
        before=first_before,
        after=first_after,
        before_stamp=first_before_stamp,
        after_stamp=first_after_stamp,
    )

    second_before_stamp = document.state_stamp
    second_before = DocumentHistoryState.capture(document)
    document.metadata["ordinary-edit"] = 2
    second_after = DocumentHistoryState.capture(document)
    second_after_stamp = document.advance_state({DirtyDomain.SESSION})
    payload = first_before.construction_payload
    assert serialization_calls == len(entities)
    assert first_after.construction_payload is payload
    assert second_before.construction_payload is payload
    assert second_after.construction_payload is payload
    first_command_bytes = (
        first_before.estimated_bytes
        + first_after.estimated_bytes
        + 256
        - payload.estimated_bytes
    )
    second_command_bytes = (
        second_before.estimated_bytes
        + second_after.estimated_bytes
        + 256
        - payload.estimated_bytes
    )
    deduplicated_history_bytes = (
        first_command_bytes + second_command_bytes - payload.estimated_bytes
    )
    document.history._max_bytes = deduplicated_history_bytes
    assert document.history.push_delta(
        "普通命令二",
        before=second_before,
        after=second_after,
        before_stamp=second_before_stamp,
        after_stamp=second_after_stamp,
    )
    assert document.history.command_count == 2
    assert document.history.total_bytes == deduplicated_history_bytes
    first_entity = document.construction_entities[0]
    assert document.history.undo(document)
    assert document.history.undo(document)
    assert document.history.redo(document)
    assert document.history.redo(document)
    assert serialization_calls == len(entities)
    assert document.construction_entities[0] is first_entity


def test_shared_history_payload_is_scanned_once_across_many_commands() -> None:
    class CountingPayloadTuple(tuple[bytes, ...]):
        iteration_count = 0

        def __iter__(self):
            self.iteration_count += 1
            return super().__iter__()

    entity_payloads = CountingPayloadTuple(
        bytes(f'{{"id":"entity-{index}"}}', "utf-8")
        for index in range(1024)
    )
    payload = ConstructionHistoryPayload(entity_payloads, ())
    document = _document("shared-history-scan")
    state = replace(
        DocumentHistoryState.capture(document),
        construction_payload=payload,
    )
    command = DocumentDeltaCommand(
        "普通命令",
        state,
        state,
        document.state_stamp,
        document.state_stamp,
        DocumentChangeImpact.SESSION,
    )
    document.history._undo_stack = [(index, command) for index in range(200)]

    single_budget = _shared_construction_payload_bytes((payload,))
    entity_payloads.iteration_count = 0
    repeated_budget = document.history.total_bytes

    assert repeated_budget == (
        command.estimated_bytes_without_construction * 200 + single_budget
    )
    assert entity_payloads.iteration_count == 1


def test_single_construction_edit_structurally_shares_large_history_payload() -> None:
    document = _document("construction-sharing")
    document.construction_entities = [
        ConstructionEntity(
            id=f"point-{index}",
            name=f"点 {index}",
            definition=FreePointDefinition(Point(float(index), 0.0)),
        )
        for index in range(2048)
    ]
    document.mark_construction_geometry_changed()
    before = DocumentHistoryState.capture(document)
    original = document.construction_entities[1024]

    assert document.replace_construction_entity(
        original.id,
        replace(
            original,
            definition=FreePointDefinition(Point(1024.0, 12.0)),
        ),
        mark_dirty=False,
    )
    after = DocumentHistoryState.capture(document)

    assert before.construction_payload is not after.construction_payload
    shared = sum(
        left is right
        for left, right in zip(
            before.construction_payload.entity_payloads,
            after.construction_payload.entity_payloads,
            strict=True,
        )
    )
    assert shared == 2047
    assert document.history.push_delta(
        "编辑单个辅助点",
        before=before,
        after=after,
        before_stamp=document.state_stamp,
        after_stamp=document.state_stamp,
        impact=DocumentChangeImpact.CONSTRUCTION,
    )
    assert document.history.total_bytes < len(before.construction_payload) * 1.2


def test_digital_slide_construction_history_preserves_live_navigation_for_save() -> None:
    document = ImageDocument(
        id="slide-history",
        path="/tmp/slide-history.fdmslide",
        image_size=(4096, 4096),
        document_kind="digital_slide",
        metadata={
            "digital_slide": {
                "working_path": "/tmp/slide-history.fdmslide",
                "viewport_origin": [100, 200],
                "focus_index": 1,
                "capture_scale": 1.0,
            }
        },
    )
    document.initialize_runtime_state()
    before_stamp = document.state_stamp
    before = DocumentHistoryState.capture(document)
    entity = ConstructionEntity(
        id="slide-point",
        name="辅助点",
        definition=FreePointDefinition(Point(120.0, 230.0)),
    )
    document.add_construction_entity(entity)
    document.metadata["digital_slide"]["capture_scale"] = 2.0
    after = DocumentHistoryState.capture(document)
    after_stamp = document.state_stamp
    assert document.history.push_delta(
        "新增辅助点",
        before=before,
        after=after,
        before_stamp=before_stamp,
        after_stamp=after_stamp,
        impact=DocumentChangeImpact.CONSTRUCTION,
    )

    # Navigation and focus changes are live view state, not part of the
    # construction command that was just recorded.
    document.metadata["digital_slide"]["viewport_origin"] = [1200, 2200]
    document.metadata["digital_slide"]["focus_index"] = 4

    assert document.history.undo(document)
    assert document.construction_entities == []
    assert document.metadata["digital_slide"]["capture_scale"] == 1.0
    assert document.metadata["digital_slide"]["viewport_origin"] == [1200, 2200]
    assert document.metadata["digital_slide"]["focus_index"] == 4

    assert document.history.redo(document)
    assert [item.id for item in document.construction_entities] == [entity.id]
    assert document.metadata["digital_slide"]["capture_scale"] == 2.0
    assert document.metadata["digital_slide"]["viewport_origin"] == [1200, 2200]
    assert document.metadata["digital_slide"]["focus_index"] == 4

    save_payload = document.to_dict()
    saved_slide_metadata = save_payload["metadata"]["digital_slide"]
    assert saved_slide_metadata["viewport_origin"] == [1200, 2200]
    assert saved_slide_metadata["focus_index"] == 4


def test_session_snapshot_is_sparse_until_construction_content_exists() -> None:
    document = _document()
    assert "construction_entities" not in document.session_snapshot()

    entity = _line("c1")
    document.add_construction_entity(entity)

    snapshot = document.session_snapshot()
    assert snapshot["construction_entities"] == [entity.to_dict()]
    assert snapshot["selected_construction_id"] == entity.id
