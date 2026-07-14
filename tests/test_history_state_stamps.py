from __future__ import annotations

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm.geometry import Line, Point
from fdm.history import (
    DocumentChangeImpact,
    DocumentHistory,
    DocumentHistoryState,
    WorkspaceHistoryBudget,
    dirty_domains_for_impact,
)
from fdm.models import (
    Calibration,
    DirtyDomain,
    ImageDocument,
    Measurement,
    ProjectGroupTemplate,
    ProjectState,
)
from fdm.services.group_manager import GroupManager


def _document(document_id: str = "doc") -> ImageDocument:
    document = ImageDocument(
        id=document_id,
        path=f"/tmp/{document_id}.png",
        image_size=(640, 480),
    )
    document.initialize_runtime_state()
    return document


def _record(
    document: ImageDocument,
    label: str,
    mutator,
    *,
    impact: DocumentChangeImpact = DocumentChangeImpact.SESSION,
    geometry_ids: tuple[str, ...] = (),
) -> None:
    before_stamp = document.state_stamp
    before = DocumentHistoryState.capture(
        document,
        geometry_measurement_ids=geometry_ids,
    )
    mutator()
    after = DocumentHistoryState.capture(
        document,
        geometry_measurement_ids=geometry_ids,
    )
    assert before != after
    after_stamp = document.advance_state(dirty_domains_for_impact(impact))
    assert document.history.push_delta(
        label,
        before=before,
        after=after,
        before_stamp=before_stamp,
        after_stamp=after_stamp,
        impact=impact,
    )


def test_refresh_dirty_flags_is_constant_time_and_does_not_serialize(monkeypatch) -> None:
    document = _document()

    def forbidden(_self):  # pragma: no cover - failure explains the regression
        raise AssertionError("dirty refresh serialized the document")

    monkeypatch.setattr(ImageDocument, "session_snapshot", forbidden)
    monkeypatch.setattr(ImageDocument, "calibration_snapshot", forbidden)
    document.refresh_dirty_flags()
    document.mark_session_dirty()
    assert document.dirty_flags.session_dirty
    document.mark_session_saved()
    assert not document.dirty_flags.session_dirty


def test_savepoint_undo_redo_and_branch_use_restorable_monotonic_stamps() -> None:
    document = _document()
    first = Measurement(
        id="first",
        image_id=document.id,
        fiber_group_id=None,
        mode="manual",
        line_px=Line(Point(0, 0), Point(20, 0)),
    )
    _record(document, "first", lambda: document.add_measurement(first))
    saved_stamp = document.state_stamp
    document.mark_session_saved()

    assert document.history.undo(document)
    assert document.dirty_flags.session_dirty
    assert document.history.redo(document)
    assert document.state_stamp == saved_stamp
    assert not document.dirty_flags.session_dirty

    assert document.history.undo(document)
    second = Measurement(
        id="second",
        image_id=document.id,
        fiber_group_id=None,
        mode="manual",
        line_px=Line(Point(0, 0), Point(40, 0)),
    )
    _record(document, "branch", lambda: document.add_measurement(second))
    assert document.state_stamp.session_state_id > saved_stamp.session_state_id
    assert not document.history.can_redo()


def test_category_history_does_not_copy_or_invalidate_dense_area_geometry() -> None:
    document = _document()
    first_group = document.create_group(color="#112233", label="A")
    second_group = document.create_group(color="#445566", label="B")
    ring = [Point(float(index), float(index % 17)) for index in range(20_000)]
    measurement = Measurement(
        id="dense",
        image_id=document.id,
        fiber_group_id=first_group.id,
        mode="polygon_area",
        measurement_kind="area",
        polygon_px=list(ring),
        area_rings_px=[list(ring)],
    )
    document.add_measurement(measurement)
    document.mark_session_saved()
    geometry_revision = document.measurement_geometry_revision
    raw_ring = measurement.area_rings_px

    _record(
        document,
        "category",
        lambda: document.set_measurement_group(measurement.id, second_group.id),
    )

    assert document.measurement_geometry_revision == geometry_revision
    assert document.history.total_bytes < 100_000
    assert document.history.undo(document)
    assert document.get_measurement(measurement.id).fiber_group_id == first_group.id
    assert document.get_measurement(measurement.id).area_rings_px is raw_ring
    assert document.measurement_geometry_revision == geometry_revision


def test_geometry_history_restores_raw_rings_exact_area_and_revision() -> None:
    document = _document()
    original_rings = [
        [Point(0, 0), Point(20, 0), Point(20, 20), Point(0, 20)],
        [Point(5, 5), Point(10, 5), Point(10, 10), Point(5, 10)],
    ]
    measurement = Measurement(
        id="area",
        image_id=document.id,
        fiber_group_id=None,
        mode="auto_instance",
        measurement_kind="area",
        polygon_px=list(original_rings[0]),
        area_rings_px=[list(ring) for ring in original_rings],
        exact_area_px=321.5,
    )
    measurement.recalculate(None)
    document.add_measurement(measurement)
    document.mark_session_saved()
    before_revision = measurement.geometry_revision
    replacement = [Point(0, 0), Point(30, 0), Point(30, 10), Point(0, 10)]

    _record(
        document,
        "geometry",
        lambda: measurement.replace_area_geometry(
            polygon_px=replacement,
            area_rings_px=[replacement],
            exact_area_px=None,
            calibration=None,
        ),
        impact=DocumentChangeImpact.SESSION | DocumentChangeImpact.GEOMETRY,
        geometry_ids=(measurement.id,),
    )
    assert measurement.exact_area_px is None
    assert document.history.undo(document)
    restored = document.get_measurement(measurement.id)
    assert restored.exact_area_px == 321.5
    assert [[point.to_dict() for point in ring] for ring in restored.area_rings_px] == [
        [point.to_dict() for point in ring] for ring in original_rings
    ]
    assert restored.geometry_revision > before_revision
    assert document.history.redo(document)
    assert document.get_measurement(measurement.id).exact_area_px is None


@pytest.mark.parametrize("calibration", [None, Calibration("preset", 8.0, "um", "test")])
def test_loaded_document_recalculation_remains_clean(calibration: Calibration | None) -> None:
    source = _document("source")
    source.calibration = calibration
    source.add_measurement(
        Measurement(
            id="line",
            image_id=source.id,
            fiber_group_id=None,
            mode="manual",
            line_px=Line(Point(0, 0), Point(16, 0)),
        )
    )
    loaded = ImageDocument.from_dict(source.to_dict())
    loaded.recalculate_measurements()
    assert not loaded.dirty_flags.session_dirty
    assert not loaded.dirty_flags.calibration_dirty


def test_group_template_sync_rolls_back_all_documents_on_late_failure(monkeypatch) -> None:
    first = _document("first")
    second = _document("second")
    project = ProjectState.empty()
    project.documents = [first, second]
    project.project_group_templates = [ProjectGroupTemplate(label="A", color="#112233")]
    manager = GroupManager(project, color_palette=["#112233"])
    first_before = first.to_dict()
    second_before = second.to_dict()
    first_stamp = first.state_stamp
    second_stamp = second.state_stamp
    original = manager.apply_project_group_templates_to_document

    def fail_second(document: ImageDocument, *, labels=None):
        if document is second:
            document.create_group(color="#445566", label="partial")
            raise RuntimeError("injected")
        return original(document, labels=labels)

    monkeypatch.setattr(manager, "apply_project_group_templates_to_document", fail_second)
    with pytest.raises(RuntimeError, match="injected"):
        manager.sync_project_group_templates(history_label="sync")

    assert first.to_dict() == first_before
    assert second.to_dict() == second_before
    assert first.state_stamp == first_stamp
    assert second.state_stamp == second_stamp
    assert not first.history.can_undo()
    assert not second.history.can_undo()


def test_history_budgets_and_workspace_unregister() -> None:
    budget = WorkspaceHistoryBudget(max_bytes=1_500)
    first = DocumentHistory(max_commands=2, max_bytes=10_000)
    second = DocumentHistory(max_commands=2, max_bytes=10_000)
    first.set_workspace_budget(budget)
    second.set_workspace_budget(budget)
    for index in range(3):
        first.push(str(index), {"value": index}, {"value": index + 1})
    assert first.command_count <= 2
    second.push("other", {"value": 1}, {"value": 2})
    assert first.total_bytes + second.total_bytes <= budget.max_bytes
    first.set_workspace_budget(None)
    # The detached history is no longer considered or evicted by the workspace.
    detached_count = first.command_count
    for index in range(4):
        second.push(str(index), {"value": index}, {"value": index + 10})
    assert first.command_count == detached_count


def test_legacy_snapshot_history_restores_savepoint_stamps() -> None:
    document = _document("legacy")
    # Exercise the snapshot metadata itself rather than relying on the normal
    # owner-bound compatibility fallback. Third-party histories may be created
    # before they are attached to a document.
    history = DocumentHistory()
    before = document.snapshot_state()
    document.add_measurement(
        Measurement(
            id="legacy-line",
            image_id=document.id,
            fiber_group_id=None,
            mode="manual",
            line_px=Line(Point(0, 0), Point(25, 0)),
        )
    )
    after = document.snapshot_state()
    history.push("legacy", before, after)
    document.mark_session_saved()
    document.mark_calibration_saved()

    assert history.undo(document)
    assert document.dirty_flags.session_dirty
    assert history.redo(document)
    assert not document.dirty_flags.session_dirty
    assert not document.dirty_flags.calibration_dirty


def test_owner_bound_legacy_push_advances_missing_session_stamp() -> None:
    measurement = Measurement(
        id="legacy-direct",
        image_id="legacy-owner",
        fiber_group_id=None,
        mode="manual",
        line_px=Line(Point(0, 0), Point(10, 0)),
    )
    document = ImageDocument(
        id="legacy-owner",
        path="/tmp/legacy-owner.png",
        image_size=(100, 80),
        measurements=[measurement],
    )
    document.initialize_runtime_state()
    before = document.snapshot_state()
    measurement.status = "edited-by-legacy-integration"
    after = document.snapshot_state()

    # The legacy integration changed persisted state directly, so both
    # snapshots carry the same pre-change stamp. The owner-bound history must
    # allocate a new SESSION state without falsely dirtying calibration.
    document.history.push("legacy direct", before, after)
    assert document.dirty_flags.session_dirty
    assert not document.dirty_flags.calibration_dirty
    assert document.history.last_affected_domains == frozenset()

    edited_stamp = document.state_stamp
    document.mark_session_saved()
    assert document.history.undo(document)
    assert document.dirty_flags.session_dirty
    assert document.history.last_affected_domains == frozenset({DirtyDomain.SESSION})
    assert document.history.redo(document)
    assert document.state_stamp == edited_stamp
    assert not document.dirty_flags.session_dirty


def test_oversized_latest_history_command_is_retained_and_notified_once() -> None:
    history = DocumentHistory(max_commands=200, max_bytes=64)
    history.push("oversized", {"value": "a" * 128}, {"value": "b" * 128})

    assert history.can_undo()
    assert history.command_count == 1
    assert history.consume_budget_notice()
    assert not history.consume_budget_notice()
