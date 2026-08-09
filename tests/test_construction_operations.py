from __future__ import annotations

import pytest

from fdm.construction_geometry import (
    CircleCenterRadiusDefinition,
    CommonTangentDefinition,
    ConcentricCircleDefinition,
    ConstructionEntity,
    ConstructionResolver,
    ConstructionValidationError,
    FreePointDefinition,
    FrozenFeatureSnapshot,
    IntersectionDefinition,
    LineDefinition,
    LineExtent,
    LiveFeatureRef,
    MidpointDefinition,
    OffsetParallelDefinition,
    ParallelThroughPointDefinition,
    PerpendicularDefinition,
    PointCircleTangentDefinition,
    ResolvedLine,
    ResolvedPoint,
    SourceObjectKind,
    ThreeTangentCircleDefinition,
)
from fdm.geometry import Line, Point
from fdm.models import Calibration, ImageDocument, Measurement
from fdm.services.construction_operations import (
    cascade_deletion_ids,
    copy_constructions,
    detach_live_sources,
    iter_live_refs,
    measurement_geometry_resolver,
    plan_cascade_deletion,
    summarize_copy_bounds,
    transitive_dependents,
)


def _document(
    identifier: str,
    *,
    size: tuple[int, int] = (100, 100),
    entities: list[ConstructionEntity] | None = None,
    measurements: list[Measurement] | None = None,
    calibration: Calibration | None = None,
) -> ImageDocument:
    return ImageDocument(
        id=identifier,
        path=f"/tmp/{identifier}.png",
        image_size=size,
        construction_entities=list(entities or []),
        measurements=list(measurements or []),
        calibration=calibration,
    )


def _entity(identifier: str, definition: object, **changes: object) -> ConstructionEntity:
    return ConstructionEntity(
        id=identifier,
        name=identifier,
        definition=definition,  # type: ignore[arg-type]
        **changes,  # type: ignore[arg-type]
    )


def _construction_ref(
    document_id: str,
    object_id: str,
    *,
    feature: str = "geometry",
) -> LiveFeatureRef:
    return LiveFeatureRef(
        document_id=document_id,
        object_id=object_id,
        object_kind=SourceObjectKind.CONSTRUCTION,
        feature=feature,
    )


def _measurement_ref(
    document_id: str,
    object_id: str,
    *,
    feature: str = "geometry",
) -> LiveFeatureRef:
    return LiveFeatureRef(
        document_id=document_id,
        object_id=object_id,
        object_kind=SourceObjectKind.MEASUREMENT,
        feature=feature,
    )


def _line_measurement(identifier: str, line: Line) -> Measurement:
    return Measurement(
        id=identifier,
        image_id="source",
        fiber_group_id=None,
        mode="manual",
        measurement_kind="line",
        line_px=line,
    )


def test_iter_live_refs_preserves_order_and_ignores_frozen_sources() -> None:
    first = _construction_ref("doc", "first", feature="line:2")
    second = _measurement_ref("doc", "measurement")
    frozen = FrozenFeatureSnapshot(
        ResolvedLine(Point(0.0, 0.0), Point(10.0, 0.0))
    )
    intersection = _entity(
        "intersection",
        IntersectionDefinition(first, second, branch=1),
    )
    partially_frozen = _entity(
        "partially-frozen",
        IntersectionDefinition(first, frozen),
    )

    assert tuple(iter_live_refs(intersection)) == (first, second)
    assert tuple(iter_live_refs(partially_frozen.definition)) == (first,)
    assert first.feature == "line:2"


def test_iter_live_refs_covers_all_phase_two_source_shapes() -> None:
    first = _construction_ref("doc", "first")
    second = _construction_ref("doc", "second")
    third = _measurement_ref("doc", "third")
    frozen = FrozenFeatureSnapshot(ResolvedPoint(Point(1.0, 2.0)))

    assert tuple(iter_live_refs(ConcentricCircleDefinition(first, 12.0))) == (first,)
    assert tuple(iter_live_refs(CommonTangentDefinition(first, second))) == (
        first,
        second,
    )
    assert tuple(
        iter_live_refs(PointCircleTangentDefinition(frozen, second))
    ) == (second,)
    assert tuple(
        iter_live_refs(ThreeTangentCircleDefinition(first, second, third))
    ) == (first, second, third)


def test_transitive_dependents_follows_construction_and_measurement_roots() -> None:
    base = _entity(
        "base",
        LineDefinition(Point(0.0, 0.0), Point(10.0, 0.0)),
    )
    offset = _entity(
        "offset",
        OffsetParallelDefinition(_construction_ref("doc", "base"), 5.0),
    )
    perpendicular = _entity(
        "perpendicular",
        PerpendicularDefinition(
            _construction_ref("doc", "offset"),
            Point(2.0, 3.0),
        ),
    )
    measured = _entity(
        "measured",
        MidpointDefinition(_measurement_ref("doc", "measurement")),
    )
    measured_child = _entity(
        "measured-child",
        ParallelThroughPointDefinition(
            _construction_ref("doc", "measured"),
            Point(0.0, 0.0),
        ),
    )
    independent = _entity(
        "independent",
        LineDefinition(Point(0.0, 20.0), Point(10.0, 20.0)),
    )
    entities = (base, offset, perpendicular, measured, measured_child, independent)

    assert transitive_dependents(
        entities,
        ("base",),
        source_kind=SourceObjectKind.CONSTRUCTION,
    ) == frozenset(
        {"offset", "perpendicular"}
    )
    assert transitive_dependents(
        entities,
        ("measurement",),
        source_kind=SourceObjectKind.MEASUREMENT,
    ) == frozenset(
        {"measured", "measured-child"}
    )


def test_transitive_dependents_handles_deep_chain_in_one_graph_pass() -> None:
    document_id = "deep-chain"
    entities: list[ConstructionEntity] = []
    previous_id = "measurement-root"
    previous_kind = SourceObjectKind.MEASUREMENT
    for index in range(5000):
        entity_id = f"dependent-{index}"
        entities.append(
            _entity(
                entity_id,
                MidpointDefinition(
                    LiveFeatureRef(
                        document_id,
                        previous_id,
                        object_kind=previous_kind,
                    )
                ),
            )
        )
        previous_id = entity_id
        previous_kind = SourceObjectKind.CONSTRUCTION

    dependents = transitive_dependents(
        entities,
        ("measurement-root",),
        source_kind=SourceObjectKind.MEASUREMENT,
    )

    assert len(dependents) == 5000
    assert "dependent-0" in dependents
    assert "dependent-4999" in dependents


def test_dependency_identity_separates_same_id_across_object_kinds() -> None:
    shared = _entity(
        "shared",
        LineDefinition(Point(0.0, 0.0), Point(20.0, 0.0)),
    )
    construction_child = _entity(
        "construction-child",
        OffsetParallelDefinition(_construction_ref("doc", "shared"), 2.0),
    )
    construction_grandchild = _entity(
        "construction-grandchild",
        PerpendicularDefinition(
            _construction_ref("doc", construction_child.id),
            Point(3.0, 4.0),
        ),
    )
    measurement_child = _entity(
        "measurement-child",
        OffsetParallelDefinition(_measurement_ref("doc", "shared"), 4.0),
    )
    measurement_grandchild = _entity(
        "measurement-grandchild",
        PerpendicularDefinition(
            _construction_ref("doc", measurement_child.id),
            Point(5.0, 6.0),
        ),
    )
    entities = (
        shared,
        construction_child,
        construction_grandchild,
        measurement_child,
        measurement_grandchild,
    )

    assert transitive_dependents(
        entities,
        ("shared",),
        source_kind=SourceObjectKind.CONSTRUCTION,
    ) == frozenset({"construction-child", "construction-grandchild"})
    assert transitive_dependents(
        entities,
        ("shared",),
        source_kind=SourceObjectKind.MEASUREMENT,
    ) == frozenset({"measurement-child", "measurement-grandchild"})
    assert cascade_deletion_ids(entities, ("shared",)) == frozenset(
        {"shared", "construction-child", "construction-grandchild"}
    )


def test_cascade_deletion_plan_is_atomic_and_validates_missing_ids() -> None:
    base = _entity(
        "base",
        LineDefinition(Point(0.0, 0.0), Point(10.0, 0.0)),
    )
    child = _entity(
        "child",
        OffsetParallelDefinition(_construction_ref("doc", "base"), 2.0),
    )
    independent = _entity(
        "independent",
        LineDefinition(Point(0.0, 10.0), Point(10.0, 10.0)),
    )
    entities = [base, child, independent]

    assert cascade_deletion_ids(entities, ("base",)) == frozenset(
        {"base", "child"}
    )
    plan = plan_cascade_deletion(entities, ("base",))
    assert plan.requested_ids == frozenset({"base"})
    assert plan.dependent_ids == frozenset({"child"})
    assert [entity.id for entity in plan.remaining_entities] == ["independent"]
    assert [entity.id for entity in entities] == ["base", "child", "independent"]

    with pytest.raises(KeyError, match="不存在"):
        cascade_deletion_ids(entities, ("missing",))


def test_detach_live_sources_freezes_only_selected_feature_without_revision_change() -> None:
    horizontal = _entity(
        "horizontal",
        LineDefinition(Point(0.0, 0.0), Point(20.0, 0.0)),
    )
    vertical = _entity(
        "vertical",
        LineDefinition(Point(10.0, -10.0), Point(10.0, 10.0)),
    )
    horizontal_ref = _construction_ref("doc", "horizontal", feature="midpoint")
    vertical_ref = _construction_ref("doc", "vertical")
    dependent = _entity(
        "dependent",
        IntersectionDefinition(horizontal_ref, vertical_ref),
        revision=7,
    )
    resolver = ConstructionResolver("doc", [horizontal, vertical, dependent])

    detached = detach_live_sources(
        dependent,
        resolver,
        refs=(horizontal_ref,),
    )

    assert detached is not dependent
    assert detached.revision == 7
    assert isinstance(detached.definition, IntersectionDefinition)
    assert isinstance(detached.definition.first, FrozenFeatureSnapshot)
    assert detached.definition.first.geometry == ResolvedPoint(Point(10.0, 0.0))
    assert detached.definition.second == vertical_ref
    assert horizontal_ref.feature == "midpoint"


def test_detach_all_supports_measurement_features_through_external_resolver() -> None:
    measurement = _line_measurement(
        "measurement",
        Line(Point(4.0, 6.0), Point(12.0, 6.0)),
    )
    entity = _entity(
        "midpoint",
        MidpointDefinition(_measurement_ref("doc", "measurement")),
    )
    document = _document("doc", entities=[entity], measurements=[measurement])
    resolver = ConstructionResolver(
        document.id,
        document.construction_entities,
        external_feature_resolver=measurement_geometry_resolver(document),
    )

    detached = detach_live_sources(entity, resolver)

    assert isinstance(detached.definition, MidpointDefinition)
    assert isinstance(detached.definition.source, FrozenFeatureSnapshot)
    assert detached.definition.source.geometry == ResolvedLine(
        Point(4.0, 6.0),
        Point(12.0, 6.0),
    )
    detached_result = ConstructionResolver("doc", [detached]).resolve(detached)
    assert detached_result.valid
    assert detached_result.geometry == ResolvedPoint(Point(8.0, 6.0))


def test_detach_source_ids_freezes_only_matching_kind_when_ids_collide() -> None:
    shared = _entity(
        "shared",
        LineDefinition(Point(0.0, 0.0), Point(20.0, 0.0)),
    )
    measurement = _line_measurement(
        "shared",
        Line(Point(6.0, -5.0), Point(6.0, 15.0)),
    )
    construction_ref = _construction_ref("doc", "shared")
    measurement_ref = _measurement_ref("doc", "shared", feature="midpoint")
    mixed = _entity(
        "mixed-dependent",
        ParallelThroughPointDefinition(
            construction_ref,
            Point(0.0, 0.0),
            point_source=measurement_ref,
        ),
    )
    document = _document(
        "doc",
        entities=[shared, mixed],
        measurements=[measurement],
    )
    resolver = ConstructionResolver(
        document.id,
        document.construction_entities,
        external_feature_resolver=measurement_geometry_resolver(document),
    )

    measurement_detached = detach_live_sources(
        mixed,
        resolver,
        source_ids=("shared",),
        source_kind=SourceObjectKind.MEASUREMENT,
    )
    assert isinstance(measurement_detached.definition, ParallelThroughPointDefinition)
    assert measurement_detached.definition.source == construction_ref
    assert isinstance(
        measurement_detached.definition.point_source,
        FrozenFeatureSnapshot,
    )

    construction_detached = detach_live_sources(
        mixed,
        resolver,
        source_ids=("shared",),
        source_kind=SourceObjectKind.CONSTRUCTION,
    )
    assert isinstance(construction_detached.definition, ParallelThroughPointDefinition)
    assert isinstance(construction_detached.definition.source, FrozenFeatureSnapshot)
    assert construction_detached.definition.point_source == measurement_ref


def test_copy_includes_dependency_closure_and_remaps_live_references() -> None:
    base = _entity(
        "base",
        LineDefinition(Point(1.0, 2.0), Point(21.0, 2.0)),
        locked=True,
        snap_enabled=False,
    )
    offset = _entity(
        "offset",
        OffsetParallelDefinition(
            _construction_ref("source", "base", feature="geometry"),
            5.0,
        ),
    )
    perpendicular = _entity(
        "perpendicular",
        PerpendicularDefinition(
            _construction_ref("source", "offset", feature="geometry"),
            Point(10.0, 10.0),
        ),
    )
    source = _document("source", entities=[base, offset, perpendicular])
    target = _document("target", size=(200, 200))

    result = copy_constructions(
        source,
        target,
        ("perpendicular",),
        id_factory=lambda source_id: f"copy-{source_id}",
    )

    assert result.requested_source_ids == ("perpendicular",)
    assert result.included_source_ids == ("base", "offset", "perpendicular")
    assert result.id_map == {
        "base": "copy-base",
        "offset": "copy-offset",
        "perpendicular": "copy-perpendicular",
    }
    assert [entity.id for entity in result.entities] == [
        "copy-base",
        "copy-offset",
        "copy-perpendicular",
    ]
    copied_base, copied_offset, copied_perpendicular = result.entities
    assert copied_base.locked
    assert not copied_base.snap_enabled
    assert copied_base.revision == 0
    assert isinstance(copied_offset.definition, OffsetParallelDefinition)
    assert copied_offset.definition.source == _construction_ref(
        "target",
        "copy-base",
        feature="geometry",
    )
    assert isinstance(copied_perpendicular.definition, PerpendicularDefinition)
    assert copied_perpendicular.definition.source == _construction_ref(
        "target",
        "copy-offset",
        feature="geometry",
    )
    assert source.construction_entities == [base, offset, perpendicular]
    assert target.construction_entities == []

    resolved = ConstructionResolver("target", result.entities).resolve(
        "copy-perpendicular"
    )
    assert resolved.valid


def test_copy_handles_dependency_chain_deeper_than_python_recursion_limit() -> None:
    depth = 1_600
    entities = [
        _entity(
            "deep-0",
            LineDefinition(Point(0.0, 0.0), Point(10.0, 0.0)),
        )
    ]
    for index in range(1, depth + 1):
        entities.append(
            _entity(
                f"deep-{index}",
                OffsetParallelDefinition(
                    _construction_ref("source", f"deep-{index - 1}"),
                    1.0,
                ),
            )
        )
    source = _document("source", size=(2_000, 2_000), entities=entities)
    target = _document("target", size=(2_000, 2_000))

    result = copy_constructions(
        source,
        target,
        (f"deep-{depth}",),
        id_factory=lambda source_id: f"copy-{source_id}",
    )

    assert len(result.entities) == depth + 1
    assert result.included_source_ids[0] == "deep-0"
    assert result.included_source_ids[-1] == f"deep-{depth}"
    resolved = ConstructionResolver("target", result.entities).resolve(
        f"copy-deep-{depth}"
    )
    assert resolved.valid
    assert isinstance(resolved.geometry, ResolvedLine)
    assert resolved.geometry.start == Point(0.0, float(depth))


def test_copy_dependency_closure_preserves_cycle_error() -> None:
    first = _entity(
        "cycle-first",
        OffsetParallelDefinition(
            _construction_ref("source", "cycle-second"),
            1.0,
        ),
    )
    second = _entity(
        "cycle-second",
        OffsetParallelDefinition(
            _construction_ref("source", "cycle-first"),
            1.0,
        ),
    )

    with pytest.raises(ConstructionValidationError) as error:
        copy_constructions(
            _document("source", entities=[first, second]),
            _document("target"),
            (first.id,),
        )

    assert error.value.code == "dependency_cycle"
    assert error.value.entity_ids[0] == error.value.entity_ids[-1]


def test_copy_freezes_measurement_reference_and_preserves_pixel_coordinates() -> None:
    measurement = _line_measurement(
        "measurement",
        Line(Point(10.0, 20.0), Point(30.0, 20.0)),
    )
    midpoint = _entity(
        "midpoint",
        MidpointDefinition(_measurement_ref("source", "measurement")),
    )
    source = _document(
        "source",
        size=(100, 100),
        entities=[midpoint],
        measurements=[measurement],
        calibration=Calibration("manual", 10.0, "um", "source"),
    )
    target = _document(
        "target",
        size=(50, 50),
        calibration=Calibration("manual", 20.0, "um", "target"),
    )

    result = copy_constructions(
        source,
        target,
        ("midpoint",),
        id_factory=lambda _source_id: "copy-midpoint",
    )

    copied = result.entities[0]
    assert isinstance(copied.definition, MidpointDefinition)
    assert isinstance(copied.definition.source, FrozenFeatureSnapshot)
    assert copied.definition.source.geometry == ResolvedLine(
        Point(10.0, 20.0),
        Point(30.0, 20.0),
    )
    assert tuple(iter_live_refs(copied)) == ()
    resolved = ConstructionResolver("target", [copied]).resolve(copied)
    assert resolved.geometry == ResolvedPoint(Point(20.0, 20.0))
    assert result.bounds_summary.calibration_differs
    assert result.bounds_summary.source_image_size == (100, 100)
    assert result.bounds_summary.target_image_size == (50, 50)


def test_copy_freezes_selected_polyline_segment_feature() -> None:
    measurement = Measurement(
        id="polyline",
        image_id="source",
        fiber_group_id=None,
        mode="continuous_manual",
        measurement_kind="polyline",
        polyline_px=[Point(0.0, 0.0), Point(10.0, 0.0), Point(10.0, 20.0)],
    )
    parallel = _entity(
        "parallel",
        ParallelThroughPointDefinition(
            _measurement_ref("source", "polyline", feature="line:1"),
            Point(30.0, 40.0),
        ),
    )
    source = _document("source", entities=[parallel], measurements=[measurement])
    target = _document("target")

    result = copy_constructions(
        source,
        target,
        ("parallel",),
        id_factory=lambda _source_id: "copy-parallel",
    )

    copied = result.entities[0]
    assert isinstance(copied.definition, ParallelThroughPointDefinition)
    assert isinstance(copied.definition.source, FrozenFeatureSnapshot)
    assert copied.definition.source.geometry == ResolvedLine(
        Point(10.0, 0.0),
        Point(10.0, 20.0),
    )


def test_phase_two_tangent_copy_and_detach_remap_every_source_field() -> None:
    point = _entity(
        "point",
        FreePointDefinition(Point(20.0, 0.0)),
    )
    circle = _entity(
        "circle",
        CircleCenterRadiusDefinition(Point(0.0, 0.0), 5.0),
    )
    point_ref = _construction_ref("source", "point", feature="point")
    circle_ref = _construction_ref("source", "circle", feature="geometry")
    tangent = _entity(
        "tangent",
        PointCircleTangentDefinition(point_ref, circle_ref),
    )
    source = _document("source", entities=[point, circle, tangent])
    target = _document("target")

    result = copy_constructions(
        source,
        target,
        ("tangent",),
        id_factory=lambda source_id: f"copy-{source_id}",
    )

    assert result.included_source_ids == ("point", "circle", "tangent")
    copied_tangent = result.entities[-1]
    assert isinstance(copied_tangent.definition, PointCircleTangentDefinition)
    assert copied_tangent.definition.point_source == _construction_ref(
        "target",
        "copy-point",
        feature="point",
    )
    assert copied_tangent.definition.circle_source == _construction_ref(
        "target",
        "copy-circle",
        feature="geometry",
    )

    resolver = ConstructionResolver("source", source.construction_entities)
    detached = detach_live_sources(tangent, resolver, refs=(circle_ref,))
    assert isinstance(detached.definition, PointCircleTangentDefinition)
    assert detached.definition.point_source == point_ref
    assert isinstance(detached.definition.circle_source, FrozenFeatureSnapshot)


def test_copy_rejects_cross_document_live_references_and_id_collisions() -> None:
    foreign = _entity(
        "foreign",
        MidpointDefinition(_construction_ref("another-document", "base")),
    )
    source = _document("source", entities=[foreign])
    target = _document("target")

    with pytest.raises(ConstructionValidationError, match="跨文档") as error:
        copy_constructions(source, target, ("foreign",))
    assert error.value.code == "cross_document_reference"

    base = _entity(
        "base",
        LineDefinition(Point(0.0, 0.0), Point(10.0, 0.0)),
    )
    occupied = _entity(
        "copy-base",
        LineDefinition(Point(0.0, 5.0), Point(10.0, 5.0)),
    )
    with pytest.raises(ValueError, match="重复 ID"):
        copy_constructions(
            _document("source", entities=[base]),
            _document("target", entities=[occupied]),
            ("base",),
            id_factory=lambda _source_id: "copy-base",
        )


def test_copy_bounds_summary_distinguishes_partial_full_and_infinite_visibility() -> None:
    inside = _entity("inside", LineDefinition(Point(5.0, 5.0), Point(20.0, 5.0)))
    partial = _entity(
        "partial",
        LineDefinition(Point(40.0, 10.0), Point(70.0, 10.0)),
    )
    outside = _entity(
        "outside",
        CircleCenterRadiusDefinition(Point(100.0, 100.0), 5.0),
    )
    partial_circle = _entity(
        "partial-circle",
        CircleCenterRadiusDefinition(Point(45.0, 45.0), 10.0),
    )
    visible_infinite = _entity(
        "infinite",
        LineDefinition(
            Point(-100.0, 25.0),
            Point(-90.0, 25.0),
            LineExtent.INFINITE,
        ),
    )
    hidden_infinite = _entity(
        "infinite-outside",
        LineDefinition(
            Point(-100.0, 80.0),
            Point(-90.0, 80.0),
            LineExtent.INFINITE,
        ),
    )
    unresolved = _entity(
        "unresolved",
        MidpointDefinition(_construction_ref("target", "missing")),
    )
    target = _document("target", size=(50, 50))

    summary = summarize_copy_bounds(
        [
            inside,
            partial,
            outside,
            partial_circle,
            visible_infinite,
            hidden_infinite,
            unresolved,
        ],
        target,
    )

    assert summary.inside_ids == ("inside", "infinite")
    assert summary.partially_outside_ids == ("partial", "partial-circle")
    assert summary.fully_outside_ids == ("outside", "infinite-outside")
    assert summary.unresolved_ids == ("unresolved",)
    assert summary.out_of_bounds_ids == (
        "partial",
        "partial-circle",
        "outside",
        "infinite-outside",
    )
    assert summary.has_out_of_bounds
