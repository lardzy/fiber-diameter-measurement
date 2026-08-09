from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntFlag
import json
import sys
from typing import Any, Iterable

from fdm.construction_geometry import ConstructionEntity
from fdm.geometry import Line, Point
from fdm.image_processing_models import DisplayTransform
from fdm.models import (
    Calibration,
    DirtyDomain,
    DocumentStateStamp,
    FiberGroup,
    Measurement,
    ObjectAppearanceOverride,
    OverlayAnnotation,
)


MAX_DOCUMENT_HISTORY_COMMANDS = 200
MAX_DOCUMENT_HISTORY_BYTES = 128 * 1024 * 1024
MAX_WORKSPACE_HISTORY_BYTES = 512 * 1024 * 1024


class DocumentChangeImpact(IntFlag):
    """Runtime effects of a history command.

    ``GEOMETRY`` is deliberately separate from ``SESSION`` so category,
    appearance and calibration commands do not invalidate area path caches.
    """

    SESSION = 1
    CALIBRATION = 2
    GEOMETRY = 4
    CONSTRUCTION = 8


def dirty_domains_for_impact(impact: DocumentChangeImpact) -> frozenset[DirtyDomain]:
    domains = {DirtyDomain.SESSION}
    if impact & DocumentChangeImpact.CALIBRATION:
        domains.add(DirtyDomain.CALIBRATION)
    return frozenset(domains)


def _json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _decode_json(payload: bytes) -> Any:
    return json.loads(payload.decode("utf-8"))


@dataclass(frozen=True, slots=True)
class ConstructionHistoryPayload:
    """Structurally shared immutable construction snapshot.

    Each entity is serialized independently, so editing one object reuses the
    unchanged bytes of every other object across before/after states and across
    history commands.  The shallow tuple preserves deterministic order.
    """

    entity_payloads: tuple[bytes, ...]
    entities: tuple[ConstructionEntity, ...] = field(compare=False, repr=False)

    def __len__(self) -> int:
        # Equivalent compact JSON-list size without materializing one large
        # contiguous byte string.
        return (
            sum(len(payload) for payload in self.entity_payloads)
            + max(0, len(self.entity_payloads) - 1)
            + 2
        )

    @property
    def estimated_bytes(self) -> int:
        return len(self) + sys.getsizeof(self.entity_payloads) + sys.getsizeof(
            self.entities
        )


def _serialize_construction_payload(document: Any) -> ConstructionHistoryPayload:
    entities = tuple(document.construction_entities)
    return ConstructionHistoryPayload(
        tuple(_json_bytes(entity.to_dict()) for entity in entities),
        entities,
    )


def _construction_payload_for_document(document: Any) -> ConstructionHistoryPayload:
    history = getattr(document, "history", None)
    provider = getattr(history, "_construction_payload_for", None)
    if callable(provider):
        return provider(document)
    return _serialize_construction_payload(document)


def _remember_construction_payload(
    document: Any,
    payload: ConstructionHistoryPayload,
) -> None:
    history = getattr(document, "history", None)
    remember = getattr(history, "_remember_construction_payload", None)
    if callable(remember):
        remember(document, payload)


_DIGITAL_SLIDE_VIEW_METADATA_KEYS = ("viewport_origin", "focus_index")


def _current_digital_slide_view_metadata(document: Any) -> dict[str, Any] | None:
    """Capture view-only metadata that must not participate in undo/redo.

    The live digital-slide canvas owns navigation and focus state.  Restoring
    those fields from a geometry command would leave the live canvas at one
    viewport while the document (and the next save) points at another.
    """

    is_digital_slide = False
    checker = getattr(document, "is_digital_slide", None)
    if callable(checker):
        is_digital_slide = bool(checker())
    else:
        is_digital_slide = getattr(document, "document_kind", None) == "digital_slide"
    if not is_digital_slide:
        return None
    metadata = document.metadata if isinstance(document.metadata, dict) else {}
    slide_metadata = metadata.get("digital_slide")
    if not isinstance(slide_metadata, dict):
        return {}
    return {
        key: _decode_json(_json_bytes(slide_metadata[key]))
        for key in _DIGITAL_SLIDE_VIEW_METADATA_KEYS
        if key in slide_metadata
    }


def _restore_current_digital_slide_view_metadata(
    document: Any,
    view_metadata: dict[str, Any] | None,
) -> None:
    if view_metadata is None:
        return
    metadata = dict(document.metadata) if isinstance(document.metadata, dict) else {}
    existing_slide_metadata = metadata.get("digital_slide")
    slide_metadata = (
        dict(existing_slide_metadata)
        if isinstance(existing_slide_metadata, dict)
        else {}
    )
    for key in _DIGITAL_SLIDE_VIEW_METADATA_KEYS:
        slide_metadata.pop(key, None)
    slide_metadata.update(view_metadata)
    if slide_metadata or isinstance(existing_slide_metadata, dict):
        metadata["digital_slide"] = slide_metadata
    else:
        metadata.pop("digital_slide", None)
    document.metadata = metadata


@dataclass(frozen=True, slots=True)
class MeasurementRuntimeState:
    measurement_id: str
    fiber_group_id: str | None
    mode: str
    measurement_kind: str
    diameter_px: float | None
    diameter_unit: float | None
    area_px: float | None
    area_unit: float | None
    confidence: float
    status: str
    appearance_payload: bytes

    @classmethod
    def capture(cls, measurement: Measurement) -> "MeasurementRuntimeState":
        return cls(
            measurement_id=measurement.id,
            fiber_group_id=measurement.fiber_group_id,
            mode=measurement.mode,
            measurement_kind=measurement.measurement_kind,
            diameter_px=measurement.diameter_px,
            diameter_unit=measurement.diameter_unit,
            area_px=measurement.area_px,
            area_unit=measurement.area_unit,
            confidence=float(measurement.confidence),
            status=measurement.status,
            appearance_payload=_json_bytes(
                measurement.appearance.to_dict()
                if measurement.appearance is not None
                else None
            ),
        )

    def restore(self, measurement: Measurement) -> None:
        measurement.fiber_group_id = self.fiber_group_id
        measurement.mode = self.mode
        measurement.measurement_kind = self.measurement_kind
        measurement.diameter_px = self.diameter_px
        measurement.diameter_unit = self.diameter_unit
        measurement.area_px = self.area_px
        measurement.area_unit = self.area_unit
        measurement.confidence = self.confidence
        measurement.status = self.status
        appearance_payload = _decode_json(self.appearance_payload)
        measurement.appearance = (
            ObjectAppearanceOverride.from_dict(appearance_payload)
            if isinstance(appearance_payload, dict)
            else None
        )


def _measurement_geometry_payload(measurement: Measurement) -> bytes:
    return _json_bytes(
        {
            "measurement_kind": measurement.measurement_kind,
            "line_px": measurement.line_px.to_dict() if measurement.line_px else None,
            "snapped_line_px": (
                measurement.snapped_line_px.to_dict()
                if measurement.snapped_line_px
                else None
            ),
            "polyline_px": [point.to_dict() for point in measurement.polyline_px],
            "point_px": measurement.point_px.to_dict() if measurement.point_px else None,
            "polygon_px": [point.to_dict() for point in measurement.polygon_px],
            "area_rings_px": [
                [point.to_dict() for point in ring]
                for ring in measurement.area_rings_px
            ],
            "exact_area_px": measurement.exact_area_px,
        }
    )


def _restore_measurement_geometry(measurement: Measurement, payload: bytes) -> None:
    decoded = _decode_json(payload)
    kind = str(decoded.get("measurement_kind", measurement.measurement_kind))
    if kind == "area":
        measurement.replace_area_geometry(
            polygon_px=[Point.from_dict(item) for item in decoded.get("polygon_px", [])],
            area_rings_px=[
                [Point.from_dict(item) for item in ring]
                for ring in decoded.get("area_rings_px", [])
            ],
            exact_area_px=decoded.get("exact_area_px"),
            calibration=None,
        )
        return
    if kind == "polyline":
        measurement.replace_polyline_geometry(
            polyline_px=[Point.from_dict(item) for item in decoded.get("polyline_px", [])],
            calibration=None,
        )
        return
    if kind == "line" and decoded.get("line_px"):
        measurement.replace_line_geometry(
            line_px=Line.from_dict(decoded["line_px"]),
            snapped_line_px=(
                Line.from_dict(decoded["snapped_line_px"])
                if decoded.get("snapped_line_px")
                else None
            ),
            calibration=None,
        )
        return
    measurement.measurement_kind = kind
    point_payload = decoded.get("point_px")
    measurement.point_px = Point.from_dict(point_payload) if point_payload else None
    measurement._advance_geometry_revision()  # noqa: SLF001 - centralized history restore
    measurement.recalculate(None)


@dataclass(frozen=True, slots=True)
class DocumentHistoryState:
    """Compact restorable state for one document command.

    Dense measurement geometry remains opt-in through ``geometry_measurement_ids``.
    Construction geometry is intentionally frozen as one small JSON branch because
    definitions may reference one another and must be restored atomically.
    """

    calibration_payload: bytes
    groups_payload: bytes
    overlay_payload: bytes
    construction_payload: ConstructionHistoryPayload
    metadata_payload: bytes
    display_transform_payload: bytes
    measurement_order: tuple[str, ...]
    measurement_states: tuple[MeasurementRuntimeState, ...]
    geometry_payloads: tuple[tuple[str, bytes], ...]
    detached_measurement_payloads: tuple[tuple[str, bytes], ...]
    active_group_id: str | None = field(compare=False)
    selected_measurement_id: str | None = field(compare=False)
    selected_overlay_id: str | None = field(compare=False)
    selected_construction_id: str | None = field(compare=False)
    scale_overlay_anchor: tuple[float, float] | None
    suppressed_project_group_labels: tuple[str, ...]
    measurement_objects: tuple[Measurement, ...] = field(
        compare=False,
        repr=False,
    )

    @classmethod
    def capture(
        cls,
        document: Any,
        *,
        geometry_measurement_ids: Iterable[str] = (),
    ) -> "DocumentHistoryState":
        geometry_ids = frozenset(str(item) for item in geometry_measurement_ids if item)
        geometry_payloads = tuple(
            (measurement.id, _measurement_geometry_payload(measurement))
            for measurement in document.measurements
            if measurement.id in geometry_ids
        )
        return cls(
            calibration_payload=_json_bytes(
                document.calibration.to_dict() if document.calibration else None
            ),
            groups_payload=_json_bytes(
                [group.to_dict() for group in document.sorted_groups()]
            ),
            overlay_payload=_json_bytes(
                [annotation.to_dict() for annotation in document.overlay_annotations]
            ),
            construction_payload=_construction_payload_for_document(document),
            metadata_payload=_json_bytes(document.metadata),
            display_transform_payload=_json_bytes(
                document.display_transform.to_dict()
                if document.display_transform is not None
                else None
            ),
            measurement_order=tuple(
                measurement.id for measurement in document.measurements
            ),
            measurement_states=tuple(
                MeasurementRuntimeState.capture(measurement)
                for measurement in document.measurements
            ),
            geometry_payloads=geometry_payloads,
            detached_measurement_payloads=(),
            active_group_id=document.active_group_id,
            selected_measurement_id=document.view_state.selected_measurement_id,
            selected_overlay_id=document.selected_overlay_id,
            selected_construction_id=document.selected_construction_id,
            scale_overlay_anchor=(
                (document.scale_overlay_anchor.x, document.scale_overlay_anchor.y)
                if document.scale_overlay_anchor is not None
                else None
            ),
            suppressed_project_group_labels=tuple(
                document.suppressed_project_group_labels
            ),
            measurement_objects=tuple(document.measurements),
        )

    def detach_measurements(self, measurement_ids: frozenset[str]) -> "DocumentHistoryState":
        if not measurement_ids:
            return self
        object_map = {measurement.id: measurement for measurement in self.measurement_objects}
        detached = tuple(
            (measurement_id, _json_bytes(object_map[measurement_id].to_dict()))
            for measurement_id in self.measurement_order
            if measurement_id in measurement_ids and measurement_id in object_map
        )
        return DocumentHistoryState(
            calibration_payload=self.calibration_payload,
            groups_payload=self.groups_payload,
            overlay_payload=self.overlay_payload,
            construction_payload=self.construction_payload,
            metadata_payload=self.metadata_payload,
            display_transform_payload=self.display_transform_payload,
            measurement_order=self.measurement_order,
            measurement_states=self.measurement_states,
            geometry_payloads=self.geometry_payloads,
            detached_measurement_payloads=detached,
            active_group_id=self.active_group_id,
            selected_measurement_id=self.selected_measurement_id,
            selected_overlay_id=self.selected_overlay_id,
            selected_construction_id=self.selected_construction_id,
            scale_overlay_anchor=self.scale_overlay_anchor,
            suppressed_project_group_labels=self.suppressed_project_group_labels,
            measurement_objects=tuple(
                measurement
                for measurement in self.measurement_objects
                if measurement.id not in measurement_ids
            ),
        )

    @property
    def estimated_bytes_without_construction(self) -> int:
        return (
            len(self.calibration_payload)
            + len(self.groups_payload)
            + len(self.overlay_payload)
            + len(self.metadata_payload)
            + len(self.display_transform_payload)
            + sum(len(state.appearance_payload) + 128 for state in self.measurement_states)
            + sum(len(payload) for _measurement_id, payload in self.geometry_payloads)
            + sum(
                len(payload)
                for _measurement_id, payload in self.detached_measurement_payloads
            )
            + len(self.measurement_order) * 24
        )

    @property
    def estimated_bytes(self) -> int:
        return (
            self.estimated_bytes_without_construction
            + self.construction_payload.estimated_bytes
        )

    def restore(self, document: Any) -> None:
        current_slide_view = _current_digital_slide_view_metadata(document)
        current_order = tuple(measurement.id for measurement in document.measurements)
        current_construction_payload = _construction_payload_for_document(document)
        construction_changed = current_construction_payload != self.construction_payload
        construction_geometry_changed = construction_changed and (
            len(document.construction_entities)
            != len(self.construction_payload.entities)
            or any(
                current.id != restored.id
                or current.definition != restored.definition
                for current, restored in zip(
                    document.construction_entities,
                    self.construction_payload.entities,
                    strict=True,
                )
            )
        )
        object_map = {measurement.id: measurement for measurement in self.measurement_objects}
        object_map.update(
            {
                measurement_id: Measurement.from_dict(_decode_json(payload))
                for measurement_id, payload in self.detached_measurement_payloads
            }
        )
        document.measurements = [
            object_map[measurement_id]
            for measurement_id in self.measurement_order
            if measurement_id in object_map
        ]

        calibration_payload = _decode_json(self.calibration_payload)
        document.calibration = (
            Calibration.from_dict(calibration_payload)
            if isinstance(calibration_payload, dict)
            else None
        )
        document.fiber_groups = [
            FiberGroup.from_dict(item, fallback_number=index + 1)
            for index, item in enumerate(_decode_json(self.groups_payload))
        ]
        document.overlay_annotations = [
            OverlayAnnotation.from_dict(item)
            for item in _decode_json(self.overlay_payload)
        ]
        if construction_changed:
            document.construction_entities = list(self.construction_payload.entities)
        _remember_construction_payload(document, self.construction_payload)
        document.metadata = dict(_decode_json(self.metadata_payload))
        _restore_current_digital_slide_view_metadata(document, current_slide_view)
        display_transform_payload = _decode_json(
            self.display_transform_payload
        )
        document.display_transform = (
            DisplayTransform.from_dict(display_transform_payload)
            if isinstance(display_transform_payload, dict)
            else None
        )

        measurement_map = {
            measurement.id: measurement for measurement in document.measurements
        }
        for measurement_id, payload in self.geometry_payloads:
            measurement = measurement_map.get(measurement_id)
            if measurement is not None:
                _restore_measurement_geometry(measurement, payload)
        for state in self.measurement_states:
            measurement = measurement_map.get(state.measurement_id)
            if measurement is not None:
                state.restore(measurement)

        document.active_group_id = self.active_group_id
        document.view_state.selected_measurement_id = self.selected_measurement_id
        document.selected_overlay_id = self.selected_overlay_id
        document.selected_construction_id = self.selected_construction_id
        document.scale_overlay_anchor = (
            Point(*self.scale_overlay_anchor)
            if self.scale_overlay_anchor is not None
            else None
        )
        document.suppressed_project_group_labels = list(
            self.suppressed_project_group_labels
        )
        document.rebuild_group_memberships()
        if document.active_group_id is not None and document.get_group(document.active_group_id) is None:
            document.active_group_id = document.fiber_groups[0].id if document.fiber_groups else None
        if document.get_measurement(document.view_state.selected_measurement_id) is None:
            document.view_state.selected_measurement_id = None
        if document.get_overlay_annotation(document.selected_overlay_id) is None:
            document.selected_overlay_id = None
        if document.get_construction_entity(document.selected_construction_id) is None:
            document.selected_construction_id = None
        if document.selected_construction_id is not None:
            document.view_state.selected_measurement_id = None
            document.selected_overlay_id = None
        if current_order != self.measurement_order or self.geometry_payloads:
            document.mark_measurement_geometry_changed()
        if construction_changed:
            if construction_geometry_changed:
                document.mark_construction_geometry_changed()
            else:
                document.mark_construction_metadata_changed()


def _shared_construction_payload_bytes(
    payloads: Iterable[ConstructionHistoryPayload],
) -> int:
    """Count shared entity bytes once while retaining tuple/list overhead."""

    seen_payloads: set[int] = set()
    seen_entities: set[int] = set()
    total = 0
    for payload in payloads:
        payload_identity = id(payload)
        if payload_identity in seen_payloads:
            continue
        seen_payloads.add(payload_identity)
        total += (
            max(0, len(payload.entity_payloads) - 1)
            + 2
            + sys.getsizeof(payload.entity_payloads)
            + sys.getsizeof(payload.entities)
        )
        for entity_payload in payload.entity_payloads:
            identity = id(entity_payload)
            if identity in seen_entities:
                continue
            seen_entities.add(identity)
            total += len(entity_payload)
    return total


@dataclass(slots=True)
class DocumentDeltaCommand:
    label: str
    before: DocumentHistoryState
    after: DocumentHistoryState
    before_stamp: DocumentStateStamp
    after_stamp: DocumentStateStamp
    impact: DocumentChangeImpact

    @property
    def construction_payloads(self) -> tuple[ConstructionHistoryPayload, ...]:
        if self.before.construction_payload is self.after.construction_payload:
            return (self.before.construction_payload,)
        return (
            self.before.construction_payload,
            self.after.construction_payload,
        )

    @property
    def estimated_bytes_without_construction(self) -> int:
        return (
            self.before.estimated_bytes_without_construction
            + self.after.estimated_bytes_without_construction
            + 256
        )

    @property
    def estimated_bytes(self) -> int:
        return self.estimated_bytes_without_construction + (
            _shared_construction_payload_bytes(self.construction_payloads)
        )

    @property
    def construction_estimated_bytes(self) -> int:
        return _shared_construction_payload_bytes(self.construction_payloads)

    @property
    def affected_domains(self) -> frozenset[DirtyDomain]:
        return dirty_domains_for_impact(self.impact)

    def undo(self, document: Any) -> None:
        self.before.restore(document)
        document.restore_state_stamp(self.before_stamp)

    def redo(self, document: Any) -> None:
        self.after.restore(document)
        document.restore_state_stamp(self.after_stamp)


@dataclass(slots=True)
class UndoCommand:
    """Compatibility command for third-party callers using legacy snapshots."""

    label: str
    before: dict[str, Any]
    after: dict[str, Any]
    before_stamp: DocumentStateStamp | None = None
    after_stamp: DocumentStateStamp | None = None
    domains: frozenset[DirtyDomain] = frozenset(
        {DirtyDomain.SESSION, DirtyDomain.CALIBRATION}
    )

    @property
    def estimated_bytes(self) -> int:
        return len(_json_bytes(self.before)) + len(_json_bytes(self.after)) + 128

    @property
    def affected_domains(self) -> frozenset[DirtyDomain]:
        return self.domains

    def undo(self, document: Any) -> None:
        current_slide_view = _current_digital_slide_view_metadata(document)
        document.restore_snapshot(self.before)
        _restore_current_digital_slide_view_metadata(document, current_slide_view)
        if self.before_stamp is not None:
            document.restore_state_stamp(self.before_stamp)
        else:
            document.advance_state({DirtyDomain.SESSION, DirtyDomain.CALIBRATION})

    def redo(self, document: Any) -> None:
        current_slide_view = _current_digital_slide_view_metadata(document)
        document.restore_snapshot(self.after)
        _restore_current_digital_slide_view_metadata(document, current_slide_view)
        if self.after_stamp is not None:
            document.restore_state_stamp(self.after_stamp)
        else:
            document.advance_state({DirtyDomain.SESSION, DirtyDomain.CALIBRATION})


def _snapshot_state_stamp(snapshot: dict[str, Any]) -> DocumentStateStamp | None:
    payload = snapshot.get("_runtime_state_stamp")
    if not isinstance(payload, dict):
        return None
    try:
        return DocumentStateStamp(
            session_state_id=int(payload["session_state_id"]),
            calibration_state_id=int(payload["calibration_state_id"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _legacy_snapshot_domains(
    before: dict[str, Any],
    after: dict[str, Any],
) -> frozenset[DirtyDomain]:
    domains = {DirtyDomain.SESSION}
    before_metadata = before.get("metadata")
    after_metadata = after.get("metadata")
    before_line = before_metadata.get("calibration_line") if isinstance(before_metadata, dict) else None
    after_line = after_metadata.get("calibration_line") if isinstance(after_metadata, dict) else None
    if before.get("calibration") != after.get("calibration") or before_line != after_line:
        domains.add(DirtyDomain.CALIBRATION)
    return frozenset(domains)


HistoryCommand = DocumentDeltaCommand | UndoCommand


class WorkspaceHistoryBudget:
    """Optional aggregate budget shared by all document histories in a window."""

    def __init__(self, max_bytes: int = MAX_WORKSPACE_HISTORY_BYTES) -> None:
        self.max_bytes = max(1, int(max_bytes))
        self._histories: list["DocumentHistory"] = []

    def register(self, history: "DocumentHistory") -> None:
        if history not in self._histories:
            self._histories.append(history)

    def unregister(self, history: "DocumentHistory") -> None:
        self._histories = [item for item in self._histories if item is not history]

    def enforce(self, newest: "DocumentHistory") -> None:
        while sum(history.total_bytes for history in self._histories) > self.max_bytes:
            candidates = [
                history
                for history in self._histories
                if history is not newest or history.command_count > 1
            ]
            if not candidates:
                break
            victim = min(candidates, key=lambda history: history.oldest_sequence)
            if not victim.evict_oldest():
                break


class DocumentHistory:
    _sequence = 0

    def __init__(
        self,
        *,
        max_commands: int = MAX_DOCUMENT_HISTORY_COMMANDS,
        max_bytes: int = MAX_DOCUMENT_HISTORY_BYTES,
        owner: Any | None = None,
    ) -> None:
        self._undo_stack: list[tuple[int, HistoryCommand]] = []
        self._redo_stack: list[tuple[int, HistoryCommand]] = []
        self._max_commands = max(1, int(max_commands))
        self._max_bytes = max(1, int(max_bytes))
        self._workspace_budget: WorkspaceHistoryBudget | None = None
        self.last_affected_domains: frozenset[DirtyDomain] = frozenset()
        self.budget_evicted = False
        self._budget_notice_pending = False
        self._owner = owner
        self._construction_entities_cache: tuple[ConstructionEntity, ...] | None = None
        self._construction_payload_cache: ConstructionHistoryPayload | None = None

    def bind_document(self, document: Any) -> None:
        self._owner = document
        self._construction_entities_cache = None
        self._construction_payload_cache = None

    def _construction_payload_for(
        self,
        document: Any,
    ) -> ConstructionHistoryPayload:
        """Return a frozen construction payload, reusing it while entities match."""

        entities = document.construction_entities
        cached_entities = self._construction_entities_cache
        cached_payload = self._construction_payload_cache
        if (
            cached_entities is not None
            and cached_payload is not None
            and len(entities) == len(cached_entities)
            and all(
                entity is cached
                for entity, cached in zip(entities, cached_entities, strict=True)
            )
        ):
            return cached_payload
        reusable = (
            {
                id(entity): entity_payload
                for entity, entity_payload in zip(
                    cached_entities,
                    cached_payload.entity_payloads,
                    strict=True,
                )
            }
            if cached_entities is not None and cached_payload is not None
            else {}
        )
        entity_tuple = tuple(entities)
        payload = ConstructionHistoryPayload(
            tuple(
                reusable.get(id(entity)) or _json_bytes(entity.to_dict())
                for entity in entity_tuple
            ),
            entity_tuple,
        )
        self._remember_construction_payload(document, payload)
        return payload

    def _remember_construction_payload(
        self,
        document: Any,
        payload: ConstructionHistoryPayload,
    ) -> None:
        self._construction_entities_cache = payload.entities
        self._construction_payload_cache = payload

    def set_workspace_budget(self, budget: WorkspaceHistoryBudget | None) -> None:
        if self._workspace_budget is not None and self._workspace_budget is not budget:
            self._workspace_budget.unregister(self)
        self._workspace_budget = budget
        if budget is not None:
            budget.register(self)

    @property
    def command_count(self) -> int:
        return len(self._undo_stack) + len(self._redo_stack)

    @property
    def total_bytes(self) -> int:
        total = 0
        construction_payloads: list[ConstructionHistoryPayload] = []
        for _sequence, command in self._undo_stack + self._redo_stack:
            if isinstance(command, DocumentDeltaCommand):
                total += command.estimated_bytes_without_construction
                construction_payloads.extend(command.construction_payloads)
            else:
                total += command.estimated_bytes
        return total + _shared_construction_payload_bytes(construction_payloads)

    @property
    def oldest_sequence(self) -> int:
        candidates = self._undo_stack + self._redo_stack
        return min((sequence for sequence, _command in candidates), default=2**63 - 1)

    def clear(self) -> None:
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._construction_entities_cache = None
        self._construction_payload_cache = None
        self.last_affected_domains = frozenset()
        self.budget_evicted = False
        self._budget_notice_pending = False

    @classmethod
    def _next_sequence(cls) -> int:
        cls._sequence += 1
        return cls._sequence

    def _append(self, command: HistoryCommand) -> None:
        self._undo_stack.append((self._next_sequence(), command))
        self._redo_stack.clear()
        self._enforce_limits()
        if command.estimated_bytes > self._max_bytes:
            # Keep the newest oversized command as the one guaranteed undo,
            # but surface the memory-budget tradeoff once in the workspace UI.
            self._mark_budget_evicted()
        if self._workspace_budget is not None:
            self._workspace_budget.enforce(self)

    def _mark_budget_evicted(self) -> None:
        self.budget_evicted = True
        self._budget_notice_pending = True

    def consume_budget_notice(self) -> bool:
        pending = self._budget_notice_pending
        self._budget_notice_pending = False
        return pending

    def _enforce_limits(self) -> None:
        while len(self._undo_stack) > self._max_commands:
            self._undo_stack.pop(0)
            self._mark_budget_evicted()
        while self.total_bytes > self._max_bytes and len(self._undo_stack) > 1:
            self._undo_stack.pop(0)
            self._mark_budget_evicted()

    def evict_oldest(self) -> bool:
        candidates: list[tuple[str, int, int]] = [
            ("undo", index, sequence)
            for index, (sequence, _command) in enumerate(self._undo_stack)
        ]
        candidates.extend(
            ("redo", index, sequence)
            for index, (sequence, _command) in enumerate(self._redo_stack)
        )
        if not candidates:
            return False
        stack_name, index, _sequence = min(candidates, key=lambda item: item[2])
        target = self._undo_stack if stack_name == "undo" else self._redo_stack
        target.pop(index)
        self._mark_budget_evicted()
        return True

    def push(self, label: str, before: dict[str, Any], after: dict[str, Any]) -> None:
        if before == after:
            return
        domains = _legacy_snapshot_domains(before, after)
        before_stamp = _snapshot_state_stamp(before)
        after_stamp = _snapshot_state_stamp(after)
        owner = self._owner
        if owner is not None:
            if before_stamp is None:
                before_stamp = owner.state_stamp
            if after_stamp is None or after_stamp == before_stamp:
                after_stamp = owner.advance_state(domains)
        self._append(
            UndoCommand(
                label=label,
                before=before,
                after=after,
                before_stamp=before_stamp,
                after_stamp=after_stamp,
                domains=domains,
            )
        )

    def push_delta(
        self,
        label: str,
        *,
        before: DocumentHistoryState,
        after: DocumentHistoryState,
        before_stamp: DocumentStateStamp,
        after_stamp: DocumentStateStamp,
        impact: DocumentChangeImpact = DocumentChangeImpact.SESSION,
    ) -> bool:
        if before == after:
            return False
        changed_membership = frozenset(before.measurement_order) ^ frozenset(after.measurement_order)
        before = before.detach_measurements(changed_membership)
        after = after.detach_measurements(changed_membership)
        self._append(
            DocumentDeltaCommand(
                label=label,
                before=before,
                after=after,
                before_stamp=before_stamp,
                after_stamp=after_stamp,
                impact=impact,
            )
        )
        return True

    def can_undo(self) -> bool:
        return bool(self._undo_stack)

    def can_redo(self) -> bool:
        return bool(self._redo_stack)

    @property
    def latest_undo_sequence(self) -> int | None:
        return self._undo_stack[-1][0] if self._undo_stack else None

    @property
    def latest_redo_sequence(self) -> int | None:
        return self._redo_stack[-1][0] if self._redo_stack else None

    def contains_undo_sequence(self, sequence: int) -> bool:
        return any(item_sequence == sequence for item_sequence, _ in self._undo_stack)

    def contains_redo_sequence(self, sequence: int) -> bool:
        return any(item_sequence == sequence for item_sequence, _ in self._redo_stack)

    def discard_sequences(self, sequences: Iterable[int]) -> int:
        """Drop coordinator-owned commands without executing either state.

        Workspace commands are represented by one local history command per
        participating document.  If one participant loses its command to a
        branch or a history budget, the remaining commands must be removed as
        well so they cannot later be replayed as only half of the operation.
        """

        targets = frozenset(int(sequence) for sequence in sequences)
        if not targets:
            return 0
        before_count = len(self._undo_stack) + len(self._redo_stack)
        self._undo_stack[:] = [
            item for item in self._undo_stack if item[0] not in targets
        ]
        self._redo_stack[:] = [
            item for item in self._redo_stack if item[0] not in targets
        ]
        return before_count - len(self._undo_stack) - len(self._redo_stack)

    def undo(self, document: Any) -> bool:
        if not self._undo_stack:
            return False
        sequence, command = self._undo_stack.pop()
        command.undo(document)
        self._redo_stack.append((sequence, command))
        self.last_affected_domains = command.affected_domains
        return True

    def redo(self, document: Any) -> bool:
        if not self._redo_stack:
            return False
        sequence, command = self._redo_stack.pop()
        command.redo(document)
        self._undo_stack.append((sequence, command))
        self.last_affected_domains = command.affected_domains
        return True
