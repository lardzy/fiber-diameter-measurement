from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, replace
import math

from fdm.construction_geometry import (
    CommonTangentDefinition,
    ConcentricCircleDefinition,
    ConstructionDefinition,
    ConstructionEntity,
    ConstructionIssue,
    ConstructionResolver,
    ConstructionValidationError,
    FeatureSource,
    FrozenFeatureSnapshot,
    IntersectionDefinition,
    LineExtent,
    LiveFeatureRef,
    MidpointDefinition,
    OffsetParallelDefinition,
    OffsetCircleDefinition,
    ParallelArrayDefinition,
    ParallelLineSequence,
    ParallelThroughPointDefinition,
    PerpendicularBisectorDefinition,
    PerpendicularDefinition,
    PointCircleTangentDefinition,
    ResolvedCircle,
    ResolvedGeometry,
    ResolvedLine,
    ResolvedPoint,
    SourceObjectIdentity,
    SourceObjectKind,
    TangentTangentRadiusCircleDefinition,
    ThreeTangentCircleDefinition,
    transitive_dependents as core_transitive_dependents,
    validate_construction_graph,
)
from fdm.construction_document import resolve_measurement_geometry
from fdm.geometry import Point
from fdm.models import ImageDocument, new_id


@dataclass(frozen=True, slots=True)
class ConstructionDeletionPlan:
    """An atomic, non-mutating cascade-deletion plan."""

    remaining_entities: tuple[ConstructionEntity, ...]
    requested_ids: frozenset[str]
    dependent_ids: frozenset[str]

    @property
    def removed_ids(self) -> frozenset[str]:
        return self.requested_ids | self.dependent_ids


@dataclass(frozen=True, slots=True)
class ConstructionCopyBoundsSummary:
    """Visibility and coordinate compatibility summary for a copy preview."""

    source_image_size: tuple[int, int]
    target_image_size: tuple[int, int]
    calibration_differs: bool
    inside_ids: tuple[str, ...]
    partially_outside_ids: tuple[str, ...]
    fully_outside_ids: tuple[str, ...]
    unresolved_ids: tuple[str, ...]

    @property
    def out_of_bounds_ids(self) -> tuple[str, ...]:
        return self.partially_outside_ids + self.fully_outside_ids

    @property
    def has_out_of_bounds(self) -> bool:
        return bool(self.out_of_bounds_ids)


@dataclass(frozen=True, slots=True)
class ConstructionCopyResult:
    """A complete copy payload ready to append as one history command."""

    entities: tuple[ConstructionEntity, ...]
    id_map: Mapping[str, str]
    requested_source_ids: tuple[str, ...]
    included_source_ids: tuple[str, ...]
    bounds_summary: ConstructionCopyBoundsSummary


def iter_live_refs(
    entity_or_definition: ConstructionEntity | ConstructionDefinition,
) -> Iterator[LiveFeatureRef]:
    """Yield the live references owned directly by one construction object."""

    definition = (
        entity_or_definition.definition
        if isinstance(entity_or_definition, ConstructionEntity)
        else entity_or_definition
    )
    if isinstance(definition, IntersectionDefinition):
        sources: tuple[FeatureSource, ...] = (definition.first, definition.second)
    elif isinstance(
        definition,
        (ParallelThroughPointDefinition, PerpendicularDefinition),
    ):
        sources = (definition.source,)
        if definition.point_source is not None:
            sources += (definition.point_source,)
    elif isinstance(
        definition,
        (
            MidpointDefinition,
            OffsetParallelDefinition,
            ParallelArrayDefinition,
            PerpendicularBisectorDefinition,
            ConcentricCircleDefinition,
            OffsetCircleDefinition,
        ),
    ):
        sources = (definition.source,)
    elif isinstance(
        definition,
        (
            CommonTangentDefinition,
            TangentTangentRadiusCircleDefinition,
        ),
    ):
        sources = (definition.first, definition.second)
    elif isinstance(definition, PointCircleTangentDefinition):
        sources = (definition.point_source, definition.circle_source)
    elif isinstance(definition, ThreeTangentCircleDefinition):
        sources = (definition.first, definition.second, definition.third)
    else:
        sources = ()
    yield from (source for source in sources if isinstance(source, LiveFeatureRef))


def transitive_dependents(
    entities: Iterable[ConstructionEntity],
    source_ids: Iterable[str],
    *,
    source_kind: SourceObjectKind,
) -> frozenset[str]:
    """Return every construction transitively invalidated by the sources.

    ``source_kind`` is mandatory because measurement and construction IDs live
    in separate namespaces and may intentionally have the same value.
    """

    sequence = _validated_entities(entities)
    normalized_ids = tuple(
        _required_id(value, field_name="source_ids") for value in source_ids
    )
    return frozenset(
        core_transitive_dependents(
            sequence,
            normalized_ids,
            source_kind=SourceObjectKind(source_kind),
        )
    )


def cascade_deletion_ids(
    entities: Iterable[ConstructionEntity],
    construction_ids: Iterable[str],
) -> frozenset[str]:
    """Return requested construction IDs plus their dependent closure."""

    sequence = _validated_entities(entities)
    requested = frozenset(
        _required_id(value, field_name="construction_ids")
        for value in construction_ids
    )
    known_ids = {entity.id for entity in sequence}
    missing = requested - known_ids
    if missing:
        raise KeyError(f"要删除的辅助几何不存在: {', '.join(sorted(missing))}")
    return requested | transitive_dependents(
        sequence,
        requested,
        source_kind=SourceObjectKind.CONSTRUCTION,
    )


def plan_cascade_deletion(
    entities: Iterable[ConstructionEntity],
    construction_ids: Iterable[str],
) -> ConstructionDeletionPlan:
    """Build a validated deletion payload without mutating the document."""

    sequence = _validated_entities(entities)
    requested = frozenset(
        _required_id(value, field_name="construction_ids")
        for value in construction_ids
    )
    removed = cascade_deletion_ids(sequence, requested)
    return ConstructionDeletionPlan(
        remaining_entities=tuple(entity for entity in sequence if entity.id not in removed),
        requested_ids=requested,
        dependent_ids=removed - requested,
    )


def detach_live_sources(
    entity: ConstructionEntity,
    resolver: ConstructionResolver,
    *,
    refs: Iterable[LiveFeatureRef] | None = None,
    source_ids: Iterable[str] = (),
    source_kind: SourceObjectKind | None = None,
) -> ConstructionEntity:
    """Freeze selected live references at their currently resolved geometry.

    Omitting both selectors detaches every live reference.  Passing an empty
    ``refs`` iterable explicitly detaches none.  Selecting by ``source_ids``
    requires ``source_kind`` so equal IDs from different namespaces stay
    distinct.  Revision advancement remains the owning document's
    responsibility so the returned entity can be used by
    ``ImageDocument.replace_construction_entity`` without double-incrementing.
    """

    if not isinstance(entity, ConstructionEntity):
        raise TypeError("entity 必须是 ConstructionEntity")
    if not isinstance(resolver, ConstructionResolver):
        raise TypeError("resolver 必须是 ConstructionResolver")
    selected_refs = None if refs is None else frozenset(refs)
    selected_ids = frozenset(
        _required_id(value, field_name="source_ids") for value in source_ids
    )
    if selected_ids and source_kind is None:
        raise ValueError("使用 source_ids 冻结来源时必须显式指定 source_kind")
    selected_kind = (
        SourceObjectKind(source_kind)
        if source_kind is not None
        else None
    )
    selected_identities: frozenset[SourceObjectIdentity] = frozenset(
        (selected_kind, source_id)
        for source_id in selected_ids
        if selected_kind is not None
    )
    detach_all = refs is None and not selected_identities

    def freeze(source: FeatureSource) -> FeatureSource:
        if not isinstance(source, LiveFeatureRef):
            return source
        if (
            not detach_all
            and source not in (selected_refs or ())
            and (source.object_kind, source.object_id) not in selected_identities
        ):
            return source
        geometry = resolver.resolve_feature(source, owner_id=entity.id)
        if isinstance(geometry, ConstructionIssue):
            raise ConstructionValidationError(
                "detach_unresolved",
                f"无法冻结来源 {source.object_id}：{geometry.message}",
                (entity.id, source.object_id),
            )
        return FrozenFeatureSnapshot(geometry)

    definition = _replace_definition_sources(entity.definition, freeze)
    if definition == entity.definition:
        return entity
    return replace(entity, definition=definition)


def copy_constructions(
    source_document: ImageDocument,
    target_document: ImageDocument,
    construction_ids: Iterable[str],
    *,
    id_factory: Callable[[str], str] | None = None,
) -> ConstructionCopyResult:
    """Prepare a self-contained cross-document construction copy.

    Construction dependencies are included recursively and remain live after
    their IDs and document IDs are remapped.  Measurement references are
    frozen at their selected source feature.  No object in the returned graph
    can retain a live reference to the source document.
    """

    if not isinstance(source_document, ImageDocument) or not isinstance(
        target_document, ImageDocument
    ):
        raise TypeError("source_document 和 target_document 必须是 ImageDocument")
    source_entities = _validated_entities(source_document.construction_entities)
    source_by_id = {entity.id: entity for entity in source_entities}
    requested = tuple(
        dict.fromkeys(
            _required_id(value, field_name="construction_ids")
            for value in construction_ids
        )
    )
    missing = set(requested) - source_by_id.keys()
    if missing:
        raise KeyError(f"要复制的辅助几何不存在: {', '.join(sorted(missing))}")

    ordered_source_ids = _dependency_closure(
        source_document.id,
        source_by_id,
        requested,
    )
    occupied_ids = {entity.id for entity in target_document.construction_entities}
    generated_ids: set[str] = set()
    id_map: dict[str, str] = {}
    for source_id in ordered_source_ids:
        copied_id = _copy_id(
            source_id,
            occupied_ids | generated_ids,
            id_factory=id_factory,
        )
        generated_ids.add(copied_id)
        id_map[source_id] = copied_id

    source_resolver = ConstructionResolver(
        source_document.id,
        source_entities,
        external_feature_resolver=_measurement_resolver(source_document),
    )

    def remap_source(source: FeatureSource, *, owner_id: str) -> FeatureSource:
        if not isinstance(source, LiveFeatureRef):
            return source
        if source.document_id != source_document.id:
            raise ConstructionValidationError(
                "cross_document_reference",
                "复制源包含跨文档实时引用",
                (source.object_id,),
            )
        if source.object_kind is SourceObjectKind.MEASUREMENT:
            resolved = source_resolver.resolve_feature(source, owner_id=owner_id)
            if isinstance(resolved, ConstructionIssue):
                raise ConstructionValidationError(
                    "copy_unresolved_source",
                    f"无法冻结对象 {owner_id} 的测量来源 {source.object_id}：{resolved.message}",
                    (owner_id, source.object_id),
                )
            return FrozenFeatureSnapshot(resolved)
        mapped_id = id_map.get(source.object_id)
        if mapped_id is None:
            raise ConstructionValidationError(
                "missing_source",
                f"复制闭包缺少依赖对象 {source.object_id}",
                (source.object_id,),
            )
        return replace(
            source,
            document_id=target_document.id,
            object_id=mapped_id,
        )

    copied_entities = tuple(
        replace(
            source_by_id[source_id],
            id=id_map[source_id],
            definition=_replace_definition_sources(
                source_by_id[source_id].definition,
                lambda source, owner_id=source_id: remap_source(
                    source,
                    owner_id=owner_id,
                ),
            ),
            revision=0,
        )
        for source_id in ordered_source_ids
    )
    validate_construction_graph(target_document.id, copied_entities)
    bounds_summary = summarize_copy_bounds(
        copied_entities,
        target_document,
        source_document=source_document,
    )
    return ConstructionCopyResult(
        entities=copied_entities,
        id_map=dict(id_map),
        requested_source_ids=requested,
        included_source_ids=ordered_source_ids,
        bounds_summary=bounds_summary,
    )


def summarize_copy_bounds(
    entities: Iterable[ConstructionEntity],
    target_document: ImageDocument,
    *,
    source_document: ImageDocument | None = None,
) -> ConstructionCopyBoundsSummary:
    """Classify copied geometry against the destination image rectangle."""

    sequence = _validated_entities(entities)
    resolver = ConstructionResolver(
        target_document.id,
        sequence,
        external_feature_resolver=_measurement_resolver(target_document),
    )
    inside: list[str] = []
    partial: list[str] = []
    outside: list[str] = []
    unresolved: list[str] = []
    width, height = _normalized_image_size(target_document.image_size)
    for entity in sequence:
        result = resolver.resolve(entity)
        if not result.valid:
            unresolved.append(entity.id)
            continue
        assert result.geometry is not None
        classification = _classify_geometry(
            result.geometry,
            width - 1,
            height - 1,
        )
        if classification == "inside":
            inside.append(entity.id)
        elif classification == "partial":
            partial.append(entity.id)
        else:
            outside.append(entity.id)
    source_size = (
        _normalized_image_size(source_document.image_size)
        if source_document is not None
        else _normalized_image_size(target_document.image_size)
    )
    return ConstructionCopyBoundsSummary(
        source_image_size=source_size,
        target_image_size=(width, height),
        calibration_differs=(
            _calibration_key(source_document) != _calibration_key(target_document)
            if source_document is not None
            else False
        ),
        inside_ids=tuple(inside),
        partially_outside_ids=tuple(partial),
        fully_outside_ids=tuple(outside),
        unresolved_ids=tuple(unresolved),
    )


def measurement_geometry_resolver(
    document: ImageDocument,
) -> Callable[[LiveFeatureRef], ResolvedGeometry | None]:
    """Return the external resolver expected by ``ConstructionResolver``."""

    return _measurement_resolver(document)


def _replace_definition_sources(
    definition: ConstructionDefinition,
    transform: Callable[[FeatureSource], FeatureSource],
) -> ConstructionDefinition:
    if isinstance(definition, IntersectionDefinition):
        return replace(
            definition,
            first=transform(definition.first),
            second=transform(definition.second),
        )
    if isinstance(
        definition,
        (ParallelThroughPointDefinition, PerpendicularDefinition),
    ):
        return replace(
            definition,
            source=transform(definition.source),
            point_source=(
                transform(definition.point_source)
                if definition.point_source is not None
                else None
            ),
        )
    if isinstance(
        definition,
        (
            MidpointDefinition,
            OffsetParallelDefinition,
            ParallelArrayDefinition,
            PerpendicularBisectorDefinition,
            ConcentricCircleDefinition,
            OffsetCircleDefinition,
        ),
    ):
        return replace(definition, source=transform(definition.source))
    if isinstance(
        definition,
        (
            CommonTangentDefinition,
            TangentTangentRadiusCircleDefinition,
        ),
    ):
        return replace(
            definition,
            first=transform(definition.first),
            second=transform(definition.second),
        )
    if isinstance(definition, PointCircleTangentDefinition):
        return replace(
            definition,
            point_source=transform(definition.point_source),
            circle_source=transform(definition.circle_source),
        )
    if isinstance(definition, ThreeTangentCircleDefinition):
        return replace(
            definition,
            first=transform(definition.first),
            second=transform(definition.second),
            third=transform(definition.third),
        )
    return definition


def _dependency_closure(
    source_document_id: str,
    entities: Mapping[str, ConstructionEntity],
    requested_ids: Iterable[str],
) -> tuple[str, ...]:
    ordered: list[str] = []
    state: dict[str, int] = {}
    active: list[str] = []
    active_positions: dict[str, int] = {}

    def references(entity_id: str) -> tuple[LiveFeatureRef, ...]:
        entity = entities.get(entity_id)
        if entity is None:
            raise ConstructionValidationError(
                "missing_source",
                f"复制源缺少辅助对象 {entity_id}",
                (entity_id,),
            )
        return tuple(iter_live_refs(entity))

    for requested_id in requested_ids:
        if state.get(requested_id) == 2:
            continue
        requested_refs = references(requested_id)
        state[requested_id] = 1
        active_positions[requested_id] = len(active)
        active.append(requested_id)
        frames: list[tuple[str, tuple[LiveFeatureRef, ...], int]] = [
            (requested_id, requested_refs, 0)
        ]
        while frames:
            entity_id, entity_refs, next_index = frames[-1]
            if next_index < len(entity_refs):
                reference = entity_refs[next_index]
                frames[-1] = (entity_id, entity_refs, next_index + 1)
                if reference.document_id != source_document_id:
                    raise ConstructionValidationError(
                        "cross_document_reference",
                        f"对象 {entity_id} 包含跨文档实时引用",
                        (entity_id, reference.object_id),
                    )
                if reference.object_kind is not SourceObjectKind.CONSTRUCTION:
                    continue
                dependency_id = reference.object_id
                dependency_state = state.get(dependency_id, 0)
                if dependency_state == 2:
                    continue
                if dependency_state == 1:
                    cycle_start = active_positions[dependency_id]
                    cycle = tuple(active[cycle_start:] + [dependency_id])
                    raise ConstructionValidationError(
                        "dependency_cycle",
                        "复制源的辅助几何依赖形成环",
                        cycle,
                    )
                dependency_refs = references(dependency_id)
                state[dependency_id] = 1
                active_positions[dependency_id] = len(active)
                active.append(dependency_id)
                frames.append((dependency_id, dependency_refs, 0))
                continue
            frames.pop()
            active.pop()
            active_positions.pop(entity_id, None)
            state[entity_id] = 2
            ordered.append(entity_id)
    return tuple(ordered)


def _copy_id(
    source_id: str,
    occupied_ids: set[str],
    *,
    id_factory: Callable[[str], str] | None,
) -> str:
    if id_factory is not None:
        candidate = _required_id(id_factory(source_id), field_name="id_factory result")
        if candidate in occupied_ids:
            raise ValueError(f"复制辅助几何生成了重复 ID: {candidate}")
        return candidate
    for _attempt in range(100):
        candidate = new_id("construction")
        if candidate not in occupied_ids:
            return candidate
    raise RuntimeError("无法为复制的辅助几何生成唯一 ID")


def _measurement_resolver(
    document: ImageDocument,
) -> Callable[[LiveFeatureRef], ResolvedGeometry | None]:
    def resolve(reference: LiveFeatureRef) -> ResolvedGeometry | None:
        if (
            reference.document_id != document.id
            or reference.object_kind is not SourceObjectKind.MEASUREMENT
        ):
            return None
        return resolve_measurement_geometry(document, reference)

    return resolve


def _classify_geometry(
    geometry: ResolvedGeometry,
    width: int,
    height: int,
) -> str:
    if isinstance(geometry, ResolvedPoint):
        return "inside" if _point_inside(geometry.point, width, height) else "outside"
    if isinstance(geometry, ResolvedCircle):
        minimum_x = geometry.center.x - geometry.radius
        maximum_x = geometry.center.x + geometry.radius
        minimum_y = geometry.center.y - geometry.radius
        maximum_y = geometry.center.y + geometry.radius
        if (
            minimum_x >= 0.0
            and minimum_y >= 0.0
            and maximum_x <= width
            and maximum_y <= height
        ):
            return "inside"
        if maximum_x < 0.0 or maximum_y < 0.0 or minimum_x > width or minimum_y > height:
            return "outside"
        return "partial"
    if isinstance(geometry, ResolvedLine):
        intersects = _line_intersects_rect(geometry, width, height)
        if not intersects:
            return "outside"
        if geometry.extent is not LineExtent.SEGMENT:
            # Infinite construction geometry is intentional.  It is useful in
            # the destination whenever it crosses the visible image rectangle.
            return "inside"
        if _point_inside(geometry.start, width, height) and _point_inside(
            geometry.end, width, height
        ):
            return "inside"
        return "partial"
    if isinstance(geometry.lines, ParallelLineSequence):
        nearby = geometry.lines.indexed_intersecting_rect(
            (0.0, 0.0, float(width), float(height))
        )
        child_states = [
            _classify_geometry(line, width, height)
            for _index, line in nearby
        ]
        if not child_states or all(state == "outside" for state in child_states):
            return "outside"
        if (
            len(nearby) == len(geometry.lines)
            and all(state == "inside" for state in child_states)
        ):
            return "inside"
        return "partial"
    child_states = [
        _classify_geometry(line, width, height) for line in geometry.lines
    ]
    if all(state == "inside" for state in child_states):
        return "inside"
    if all(state == "outside" for state in child_states):
        return "outside"
    return "partial"


def _line_intersects_rect(line: ResolvedLine, width: int, height: int) -> bool:
    dx = line.end.x - line.start.x
    dy = line.end.y - line.start.y
    lower = 0.0 if line.extent is not LineExtent.INFINITE else -math.inf
    upper = 1.0 if line.extent is LineExtent.SEGMENT else math.inf
    for origin, delta, minimum, maximum in (
        (line.start.x, dx, 0.0, float(width)),
        (line.start.y, dy, 0.0, float(height)),
    ):
        if abs(delta) <= 1e-12:
            if origin < minimum or origin > maximum:
                return False
            continue
        first = (minimum - origin) / delta
        second = (maximum - origin) / delta
        if first > second:
            first, second = second, first
        lower = max(lower, first)
        upper = min(upper, second)
        if lower > upper + 1e-12:
            return False
    return True


def _point_inside(point: Point, width: int, height: int) -> bool:
    return 0.0 <= point.x <= width and 0.0 <= point.y <= height


def _calibration_key(document: ImageDocument | None) -> tuple[object, ...]:
    if document is None or document.calibration is None:
        return (None,)
    calibration = document.calibration
    return (
        calibration.unit,
        float(calibration.pixels_per_unit),
    )


def _normalized_image_size(value: object) -> tuple[int, int]:
    try:
        width = int(value[0])  # type: ignore[index]
        height = int(value[1])  # type: ignore[index]
    except (IndexError, TypeError, ValueError) as exc:
        raise ValueError("image_size 必须包含正整数宽高") from exc
    if width <= 0 or height <= 0:
        raise ValueError("image_size 必须包含正整数宽高")
    return width, height


def _validated_entities(
    entities: Iterable[ConstructionEntity],
) -> tuple[ConstructionEntity, ...]:
    sequence = tuple(entities)
    if any(not isinstance(entity, ConstructionEntity) for entity in sequence):
        raise TypeError("entities 必须全部是 ConstructionEntity")
    identifiers = [entity.id for entity in sequence]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("entities 包含重复 ID")
    return sequence


def _required_id(value: object, *, field_name: str) -> str:
    identifier = str(value or "").strip()
    if not identifier:
        raise ValueError(f"{field_name} 不能包含空 ID")
    return identifier


__all__ = [
    "ConstructionCopyBoundsSummary",
    "ConstructionCopyResult",
    "ConstructionDeletionPlan",
    "cascade_deletion_ids",
    "copy_constructions",
    "detach_live_sources",
    "iter_live_refs",
    "measurement_geometry_resolver",
    "plan_cascade_deletion",
    "summarize_copy_bounds",
    "transitive_dependents",
]
