from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import json
import math

import pytest

from fdm.construction_geometry import (
    ArraySide,
    CircleCenterDiameterDefinition,
    CircleCenterRadiusDefinition,
    CircleTangency,
    CircleThreePointDefinition,
    CircleTwoPointDefinition,
    CommonTangentDefinition,
    CommonTangentMode,
    ConcentricCircleDefinition,
    ConstructionEntity,
    ConstructionResolver,
    ConstructionValidationError,
    FreePointDefinition,
    FrozenFeatureSnapshot,
    IntersectionDefinition,
    LineDefinition,
    LineAxisConstraint,
    LineExtent,
    LiveFeatureRef,
    MidpointDefinition,
    OffsetCircleDefinition,
    OffsetParallelDefinition,
    ParallelArrayDefinition,
    ParallelLineSequence,
    ParallelThroughPointDefinition,
    PerpendicularBisectorDefinition,
    PerpendicularDefinition,
    PointCircleTangentDefinition,
    ResolvedCircle,
    ResolvedLine,
    ResolvedLineArray,
    ResolvedPoint,
    SourceObjectKind,
    TangentTangentRadiusCircleDefinition,
    TangencyConstraint,
    ThreeTangentCircleDefinition,
    common_tangent_lines,
    detach_sources,
    geometry_intersections,
    intersection_branch_hint,
    iter_live_refs,
    live_dependency_ids,
    select_feature,
    point_circle_tangent_lines,
    tangent_tangent_radius_solutions,
    three_tangent_circle_solutions,
    transitive_dependents,
    validate_construction_graph,
)
from fdm.geometry import Point


DOCUMENT_ID = "document-1"


def _entity(
    entity_id: str,
    definition: object,
    *,
    name: str | None = None,
) -> ConstructionEntity:
    return ConstructionEntity(
        id=entity_id,
        name=name or entity_id,
        definition=definition,  # type: ignore[arg-type]
    )


def _ref(entity_id: str, feature: str = "geometry") -> LiveFeatureRef:
    return LiveFeatureRef(DOCUMENT_ID, entity_id, feature=feature)


def _assert_point(point: Point, x: float, y: float) -> None:
    assert point.x == pytest.approx(x)
    assert point.y == pytest.approx(y)


def test_axis_constrained_line_normalizes_geometry_and_round_trips_sparsely() -> None:
    horizontal = LineDefinition(
        Point(3.0, 7.0),
        Point(3.0, 99.0),
        LineExtent.INFINITE,
        LineAxisConstraint.HORIZONTAL,
    )
    vertical = LineDefinition(
        Point(5.0, 8.0),
        Point(22.0, 8.0),
        LineExtent.INFINITE,
        LineAxisConstraint.VERTICAL,
    )

    assert horizontal.end == Point(4.0, 7.0)
    assert vertical.end == Point(5.0, 9.0)
    horizontal_payload = ConstructionEntity(
        id="horizontal",
        name="水平线",
        definition=horizontal,
    ).to_dict()
    assert horizontal_payload["definition"]["axis_constraint"] == "horizontal"
    restored = ConstructionEntity.from_dict(horizontal_payload)
    assert restored.definition == horizontal

    ordinary_payload = ConstructionEntity(
        id="ordinary",
        name="普通线",
        definition=LineDefinition(Point(0.0, 0.0), Point(2.0, 1.0)),
    ).to_dict()
    assert "axis_constraint" not in ordinary_payload["definition"]


@pytest.mark.parametrize(
    "definition",
    [
        FreePointDefinition(Point(1.25, 2.5)),
        LineDefinition(Point(1, 2), Point(3, 4), LineExtent.RAY),
        LineDefinition(
            Point(1, 2),
            Point(3, 9),
            LineExtent.INFINITE,
            LineAxisConstraint.HORIZONTAL,
        ),
        CircleCenterRadiusDefinition(Point(2, 3), 4.5),
        CircleCenterDiameterDefinition(Point(2, 3), 9),
        CircleTwoPointDefinition(Point(-1, 0), Point(1, 0)),
        CircleThreePointDefinition(Point(0, 1), Point(-1, 0), Point(1, 0)),
        MidpointDefinition(_ref("source")),
        IntersectionDefinition(_ref("one"), _ref("two"), branch=1, extend=True),
        ParallelThroughPointDefinition(_ref("source"), Point(4, 5), LineExtent.RAY),
        ParallelThroughPointDefinition(
            _ref("source"),
            Point(4, 5),
            LineExtent.RAY,
            point_source=_ref("through-point"),
        ),
        OffsetParallelDefinition(_ref("source"), -2.5, LineExtent.SEGMENT),
        ParallelArrayDefinition(_ref("source"), 3.0, 4, ArraySide.BOTH),
        PerpendicularDefinition(_ref("source"), Point(4, 5), LineExtent.RAY),
        PerpendicularDefinition(
            _ref("source"),
            Point(4, 5),
            LineExtent.RAY,
            point_source=FrozenFeatureSnapshot(ResolvedPoint(Point(4, 5))),
        ),
        PerpendicularBisectorDefinition(_ref("source")),
        ConcentricCircleDefinition(_ref("circle"), 8.5),
        OffsetCircleDefinition(_ref("circle"), -1.25),
        PointCircleTangentDefinition(
            _ref("point"),
            _ref("circle"),
            branch=1,
            extent=LineExtent.RAY,
        ),
        CommonTangentDefinition(
            _ref("circle-one"),
            _ref("circle-two"),
            CommonTangentMode.INTERNAL,
            branch=1,
            extent=LineExtent.RAY,
        ),
        TangentTangentRadiusCircleDefinition(
            _ref("line"),
            _ref("circle"),
            4.5,
            TangencyConstraint(-1),
            TangencyConstraint(circle_relation=CircleTangency.SOURCE_CONTAINS),
            branch=1,
            extend=True,
        ),
        ThreeTangentCircleDefinition(
            _ref("first"),
            _ref("second"),
            _ref("third"),
            TangencyConstraint(1),
            TangencyConstraint(-1),
            TangencyConstraint(circle_relation=CircleTangency.SOLUTION_CONTAINS),
            branch=1,
            extend=True,
        ),
        MidpointDefinition(
            FrozenFeatureSnapshot(
                ResolvedLine(Point(0, 0), Point(10, 0), LineExtent.SEGMENT)
            )
        ),
    ],
)
def test_entity_schema_v1_roundtrip_for_every_definition(definition: object) -> None:
    original = ConstructionEntity(
        id="guide-1",
        name="定位辅助对象",
        definition=definition,  # type: ignore[arg-type]
        visible=False,
        locked=True,
        snap_enabled=False,
        revision=7,
    )

    encoded = json.dumps(original.to_dict(), allow_nan=False)
    restored = ConstructionEntity.from_dict(json.loads(encoded))

    assert restored == original
    assert restored.definition.kind == definition.kind  # type: ignore[attr-defined]
    assert restored.snappable is False


def test_entity_is_frozen_and_revision_can_be_replaced_explicitly() -> None:
    entity = _entity("point", FreePointDefinition(Point(1, 2)))

    with pytest.raises(FrozenInstanceError):
        entity.name = "changed"  # type: ignore[misc]

    changed = replace(entity, name="changed", revision=entity.revision + 1)
    assert entity.name == "point"
    assert changed.name == "changed"
    assert changed.revision == 1


def test_unknown_entity_or_definition_schema_is_rejected() -> None:
    entity = _entity("point", FreePointDefinition(Point(1, 2)))
    future_entity = entity.to_dict()
    future_entity["schema_version"] = 2
    with pytest.raises(ValueError, match="schema"):
        ConstructionEntity.from_dict(future_entity)

    unknown_definition = entity.to_dict()
    unknown_definition["definition"] = {"kind": "ellipse"}
    with pytest.raises(ValueError, match="未知"):
        ConstructionEntity.from_dict(unknown_definition)


def test_legacy_live_reference_payload_defaults_to_whole_geometry_feature() -> None:
    entity = _entity("midpoint", MidpointDefinition(_ref("source", "midpoint")))
    payload = entity.to_dict()
    source_payload = payload["definition"]["source"]  # type: ignore[index]
    del source_payload["feature"]  # type: ignore[index]

    restored = ConstructionEntity.from_dict(payload)

    assert isinstance(restored.definition, MidpointDefinition)
    assert isinstance(restored.definition.source, LiveFeatureRef)
    assert restored.definition.source.feature == "geometry"


def test_resolves_basic_point_line_and_all_circle_creation_methods() -> None:
    entities = [
        _entity("point", FreePointDefinition(Point(2, 3))),
        _entity("line", LineDefinition(Point(1, 2), Point(5, 2), LineExtent.RAY)),
        _entity("radius", CircleCenterRadiusDefinition(Point(0, 0), 4)),
        _entity("diameter", CircleCenterDiameterDefinition(Point(0, 0), 8)),
        _entity("two-point", CircleTwoPointDefinition(Point(-4, 0), Point(4, 0))),
        _entity(
            "three-point",
            CircleThreePointDefinition(Point(0, 4), Point(-4, 0), Point(4, 0)),
        ),
    ]
    resolved = ConstructionResolver(DOCUMENT_ID, entities).resolve_all()

    assert isinstance(resolved["point"].geometry, ResolvedPoint)
    assert isinstance(resolved["line"].geometry, ResolvedLine)
    assert resolved["line"].geometry.extent is LineExtent.RAY
    for entity_id in ("radius", "diameter", "two-point", "three-point"):
        circle = resolved[entity_id].geometry
        assert isinstance(circle, ResolvedCircle)
        _assert_point(circle.center, 0, 0)
        assert circle.radius == pytest.approx(4)


@pytest.mark.parametrize(
    ("entity", "message"),
    [
        (_entity("line", LineDefinition(Point(1, 1), Point(1, 1))), "两个定义点"),
        (_entity("circle", CircleCenterRadiusDefinition(Point(0, 0), 0)), "半径"),
        (
            _entity(
                "circle",
                CircleThreePointDefinition(Point(0, 0), Point(1, 1), Point(2, 2)),
            ),
            "共线",
        ),
    ],
)
def test_degenerate_geometry_is_retained_as_an_invalid_result(
    entity: ConstructionEntity,
    message: str,
) -> None:
    result = ConstructionResolver(DOCUMENT_ID, [entity]).resolve(entity.id)

    assert not result.valid
    assert result.error is not None
    assert result.error.code == "degenerate_geometry"
    assert message in result.error.message


def test_midpoint_is_live_and_updates_when_source_geometry_changes() -> None:
    source = _entity("source", LineDefinition(Point(0, 0), Point(10, 0)))
    midpoint = _entity("midpoint", MidpointDefinition(_ref("source")))

    first = ConstructionResolver(DOCUMENT_ID, [source, midpoint]).resolve("midpoint")
    assert isinstance(first.geometry, ResolvedPoint)
    _assert_point(first.geometry.point, 5, 0)

    moved = replace(source, definition=LineDefinition(Point(4, 2), Point(14, 2)), revision=1)
    second = ConstructionResolver(DOCUMENT_ID, [moved, midpoint]).resolve("midpoint")
    assert isinstance(second.geometry, ResolvedPoint)
    _assert_point(second.geometry.point, 9, 2)


def test_midpoint_and_bisector_reject_unbounded_source_lines() -> None:
    source = _entity(
        "source",
        LineDefinition(Point(0, 0), Point(10, 0), LineExtent.INFINITE),
    )
    midpoint = _entity("midpoint", MidpointDefinition(_ref("source")))
    bisector = _entity("bisector", PerpendicularBisectorDefinition(_ref("source")))
    resolver = ConstructionResolver(DOCUMENT_ID, [source, midpoint, bisector])

    assert resolver.resolve("midpoint").error.code == "degenerate_geometry"  # type: ignore[union-attr]
    assert resolver.resolve("bisector").error.code == "degenerate_geometry"  # type: ignore[union-attr]


def test_parallel_offset_array_and_perpendicular_constructions() -> None:
    source = _entity("source", LineDefinition(Point(0, 0), Point(10, 0)))
    entities = [
        source,
        _entity(
            "through",
            ParallelThroughPointDefinition(_ref("source"), Point(5, 3)),
        ),
        _entity("offset", OffsetParallelDefinition(_ref("source"), -4)),
        _entity(
            "array",
            ParallelArrayDefinition(_ref("source"), 2, 2, ArraySide.BOTH),
        ),
        _entity(
            "perpendicular",
            PerpendicularDefinition(_ref("source"), Point(3, 7)),
        ),
        _entity("bisector", PerpendicularBisectorDefinition(_ref("source"))),
    ]
    resolved = ConstructionResolver(DOCUMENT_ID, entities).resolve_all()

    through = resolved["through"].geometry
    offset = resolved["offset"].geometry
    array = resolved["array"].geometry
    perpendicular = resolved["perpendicular"].geometry
    bisector = resolved["bisector"].geometry
    assert isinstance(through, ResolvedLine)
    assert isinstance(offset, ResolvedLine)
    assert isinstance(array, ResolvedLineArray)
    assert isinstance(perpendicular, ResolvedLine)
    assert isinstance(bisector, ResolvedLine)
    _assert_point(through.start, 5, 3)
    _assert_point(offset.start, 0, -4)
    assert [line.start.y for line in array.lines] == pytest.approx([-2, 2, -4, 4])
    assert perpendicular.direction == pytest.approx((0, 1))
    _assert_point(perpendicular.start, 3, 7)
    _assert_point(bisector.start, 5, 0)
    assert bisector.direction == pytest.approx((0, 1))


def test_large_parallel_array_resolves_to_lazy_stably_indexed_sequence() -> None:
    source = _entity("source", LineDefinition(Point(0, 0), Point(10, 0)))
    array = _entity(
        "large-array",
        ParallelArrayDefinition(_ref("source"), 2.0, 10_000, ArraySide.BOTH),
    )

    resolved = ConstructionResolver(DOCUMENT_ID, [source, array]).resolve(array)

    assert isinstance(resolved.geometry, ResolvedLineArray)
    assert isinstance(resolved.geometry.lines, ParallelLineSequence)
    assert len(resolved.geometry.lines) == 20_000
    assert resolved.geometry.lines[0].start == Point(0, -2)
    assert resolved.geometry.lines[1].start == Point(0, 2)
    assert resolved.geometry.lines[-1].start == Point(0, 20_000)
    visible = resolved.geometry.lines.indexed_intersecting_rect(
        (0.0, -5.0, 100.0, 5.0)
    )
    assert [index for index, _line in visible] == [0, 1, 2, 3]


def test_through_point_parallel_and_perpendicular_follow_live_point_source() -> None:
    line = _entity("line", LineDefinition(Point(0, 0), Point(10, 0)))
    point = _entity("point", FreePointDefinition(Point(3, 4)))
    parallel = _entity(
        "parallel",
        ParallelThroughPointDefinition(
            _ref("line"),
            Point(-1, -1),
            point_source=_ref("point"),
        ),
    )
    perpendicular = _entity(
        "perpendicular",
        PerpendicularDefinition(
            _ref("line"),
            Point(-1, -1),
            point_source=_ref("point"),
        ),
    )
    first = ConstructionResolver(
        DOCUMENT_ID,
        [line, point, parallel, perpendicular],
    ).resolve_all()
    assert isinstance(first["parallel"].geometry, ResolvedLine)
    assert isinstance(first["perpendicular"].geometry, ResolvedLine)
    _assert_point(first["parallel"].geometry.start, 3, 4)
    _assert_point(first["perpendicular"].geometry.start, 3, 4)

    moved_point = replace(
        point,
        definition=FreePointDefinition(Point(8, 9)),
        revision=1,
    )
    second = ConstructionResolver(
        DOCUMENT_ID,
        [line, moved_point, parallel, perpendicular],
    ).resolve_all()
    _assert_point(second["parallel"].geometry.start, 8, 9)  # type: ignore[union-attr]
    _assert_point(second["perpendicular"].geometry.start, 8, 9)  # type: ignore[union-attr]
    assert live_dependency_ids(
        parallel,
        source_kind=SourceObjectKind.CONSTRUCTION,
    ) == ("line", "point")


def test_line_line_intersection_obeys_extent_unless_extension_is_enabled() -> None:
    horizontal = ResolvedLine(Point(0, 0), Point(2, 0), LineExtent.SEGMENT)
    vertical = ResolvedLine(Point(3, -2), Point(3, 2), LineExtent.SEGMENT)

    points, issue = geometry_intersections(horizontal, vertical)
    assert points == ()
    assert issue is not None and issue.code == "no_intersection"

    points, issue = geometry_intersections(horizontal, vertical, extend=True)
    assert issue is None
    assert len(points) == 1
    _assert_point(points[0], 3, 0)


def test_line_circle_intersections_have_stable_line_parameter_order() -> None:
    line = ResolvedLine(Point(-10, 0), Point(10, 0), LineExtent.INFINITE)
    circle = ResolvedCircle(Point(0, 0), 5)

    points, issue = geometry_intersections(line, circle)

    assert issue is None
    assert len(points) == 2
    _assert_point(points[0], -5, 0)
    _assert_point(points[1], 5, 0)


def test_line_circle_tangent_is_a_single_intersection() -> None:
    line = ResolvedLine(Point(-10, 5), Point(10, 5), LineExtent.INFINITE)
    circle = ResolvedCircle(Point(0, 0), 5)

    points, issue = geometry_intersections(line, circle)

    assert issue is None
    assert len(points) == 1
    _assert_point(points[0], 0, 5)


def test_circle_circle_branch_follows_stable_signed_side() -> None:
    first = ResolvedCircle(Point(0, 0), 5)
    second = ResolvedCircle(Point(6, 0), 5)
    points, issue = geometry_intersections(first, second)
    assert issue is None
    assert len(points) == 2
    assert points[0].y < 0 < points[1].y

    moved_second = ResolvedCircle(Point(7, 1), 5)
    moved, issue = geometry_intersections(first, moved_second)
    assert issue is None
    direction = (
        moved_second.center.x - first.center.x,
        moved_second.center.y - first.center.y,
    )
    side0 = direction[0] * moved[0].y - direction[1] * moved[0].x
    side1 = direction[0] * moved[1].y - direction[1] * moved[1].x
    assert side0 < 0 < side1


def test_intersection_definition_keeps_selected_branch_after_parent_move() -> None:
    line = _entity(
        "line",
        LineDefinition(Point(-10, 0), Point(10, 0), LineExtent.INFINITE),
    )
    circle = _entity("circle", CircleCenterRadiusDefinition(Point(0, 0), 5))
    intersection = _entity(
        "intersection",
        IntersectionDefinition(_ref("line"), _ref("circle"), branch=1),
    )
    first = ConstructionResolver(DOCUMENT_ID, [line, circle, intersection]).resolve(
        "intersection"
    )
    assert isinstance(first.geometry, ResolvedPoint)
    _assert_point(first.geometry.point, 5, 0)

    moved_circle = replace(
        circle,
        definition=CircleCenterRadiusDefinition(Point(1, 0), 5),
        revision=1,
    )
    moved = ConstructionResolver(
        DOCUMENT_ID, [line, moved_circle, intersection]
    ).resolve("intersection")
    assert isinstance(moved.geometry, ResolvedPoint)
    _assert_point(moved.geometry.point, 6, 0)


def test_line_circle_intersection_hint_survives_endpoint_reversal_and_translation() -> None:
    original_line = ResolvedLine(
        Point(-10, 0),
        Point(10, 0),
        LineExtent.SEGMENT,
    )
    original_circle = ResolvedCircle(Point(0, 0), 5)
    candidates, issue = geometry_intersections(original_line, original_circle)
    assert issue is None and len(candidates) == 2
    hint = intersection_branch_hint(original_line, original_circle, candidates[1])
    assert hint is not None

    line = _entity(
        "stable-line",
        LineDefinition(Point(110, 0), Point(90, 0), LineExtent.SEGMENT),
    )
    circle = _entity(
        "stable-circle",
        CircleCenterRadiusDefinition(Point(100, 0), 5),
    )
    intersection = _entity(
        "stable-intersection",
        IntersectionDefinition(
            _ref(line.id),
            _ref(circle.id),
            branch=1,
            branch_hint=hint,
        ),
    )
    resolved = ConstructionResolver(
        DOCUMENT_ID,
        [line, circle, intersection],
    ).resolve(intersection)
    assert isinstance(resolved.geometry, ResolvedPoint)
    _assert_point(resolved.geometry.point, 105, 0)

    shortened = replace(
        line,
        definition=LineDefinition(Point(100, 0), Point(90, 0), LineExtent.SEGMENT),
    )
    unavailable = ConstructionResolver(
        DOCUMENT_ID,
        [shortened, circle, intersection],
    ).resolve(intersection)
    assert not unavailable.valid
    assert unavailable.error is not None
    assert unavailable.error.code == "intersection_branch_missing"

    restored = ConstructionEntity.from_dict(intersection.to_dict())
    assert restored.definition == intersection.definition


def test_circle_circle_intersection_hint_uses_source_local_signed_side() -> None:
    original_first = ResolvedCircle(Point(0, 0), 5)
    original_second = ResolvedCircle(Point(6, 0), 5)
    candidates, issue = geometry_intersections(original_first, original_second)
    assert issue is None and len(candidates) == 2
    selected = candidates[1]
    hint = intersection_branch_hint(original_first, original_second, selected)
    assert hint is not None

    first = _entity("rotated-first", CircleCenterRadiusDefinition(Point(100, 100), 5))
    second = _entity("rotated-second", CircleCenterRadiusDefinition(Point(100, 106), 5))
    intersection = _entity(
        "rotated-intersection",
        IntersectionDefinition(
            _ref(first.id),
            _ref(second.id),
            branch=1,
            branch_hint=hint,
        ),
    )
    resolved = ConstructionResolver(
        DOCUMENT_ID,
        [first, second, intersection],
    ).resolve(intersection)
    assert isinstance(resolved.geometry, ResolvedPoint)
    # The original branch is on the positive signed side of first -> second;
    # after a 90-degree rotation that side is the left-hand intersection.
    assert resolved.geometry.point.x < 100.0


@pytest.mark.parametrize(
    ("first", "second", "expected_code"),
    [
        (
            ResolvedLine(Point(0, 0), Point(1, 0), LineExtent.INFINITE),
            ResolvedLine(Point(0, 1), Point(1, 1), LineExtent.INFINITE),
            "no_intersection",
        ),
        (
            ResolvedLine(Point(0, 0), Point(1, 0), LineExtent.INFINITE),
            ResolvedLine(Point(2, 0), Point(3, 0), LineExtent.INFINITE),
            "coincident_geometry",
        ),
        (ResolvedCircle(Point(0, 0), 2), ResolvedCircle(Point(10, 0), 2), "no_intersection"),
        (
            ResolvedCircle(Point(0, 0), 2),
            ResolvedCircle(Point(0, 0), 2),
            "coincident_geometry",
        ),
    ],
)
def test_non_unique_or_missing_intersections_report_typed_issue(
    first: object,
    second: object,
    expected_code: str,
) -> None:
    points, issue = geometry_intersections(first, second)  # type: ignore[arg-type]
    assert points == ()
    assert issue is not None
    assert issue.code == expected_code


def test_feature_selection_covers_endpoints_midpoint_center_quadrants_and_array_lines() -> None:
    line = ResolvedLine(Point(0, 0), Point(10, 0))
    circle = ResolvedCircle(Point(4, 5), 3)
    array = ResolvedLineArray((line, ResolvedLine(Point(0, 2), Point(10, 2))))

    assert select_feature(line, "start") == ResolvedPoint(Point(0, 0))
    assert select_feature(line, "end") == ResolvedPoint(Point(10, 0))
    assert select_feature(line, "midpoint") == ResolvedPoint(Point(5, 0))
    assert select_feature(circle, "center") == ResolvedPoint(Point(4, 5))
    assert select_feature(circle, "quadrant:1") == ResolvedPoint(Point(4, 8))
    assert select_feature(array, "line:1") == array.lines[1]
    assert select_feature(array, "geometry:line:1:start") == ResolvedPoint(
        Point(0, 2)
    )
    assert select_feature(array, "segment:0:end") == ResolvedPoint(Point(10, 0))
    assert select_feature(array, "vertex:1") == ResolvedPoint(Point(0, 2))
    with pytest.raises(ValueError, match="不包含"):
        select_feature(circle, "endpoint")


def test_parallel_array_child_feature_identity_is_stable_when_side_changes() -> None:
    source = _entity(
        "stable-array-source",
        LineDefinition(Point(0, 0), Point(10, 0), LineExtent.SEGMENT),
    )
    positive_array = _entity(
        "stable-array",
        ParallelArrayDefinition(
            _ref(source.id),
            spacing=5.0,
            count=2,
            side=ArraySide.POSITIVE,
            extent=LineExtent.SEGMENT,
        ),
    )
    dependent = _entity(
        "stable-array-dependent",
        MidpointDefinition(_ref(positive_array.id, feature="line:+1")),
    )
    positive = ConstructionResolver(
        DOCUMENT_ID,
        [source, positive_array, dependent],
    ).resolve(dependent)
    assert isinstance(positive.geometry, ResolvedPoint)
    _assert_point(positive.geometry.point, 5, 5)

    both_array = replace(
        positive_array,
        definition=replace(positive_array.definition, side=ArraySide.BOTH),
    )
    both = ConstructionResolver(
        DOCUMENT_ID,
        [source, both_array, dependent],
    ).resolve(dependent)
    assert isinstance(both.geometry, ResolvedPoint)
    _assert_point(both.geometry.point, 5, 5)

    negative_array = replace(
        positive_array,
        definition=replace(positive_array.definition, side=ArraySide.NEGATIVE),
    )
    unavailable = ConstructionResolver(
        DOCUMENT_ID,
        [source, negative_array, dependent],
    ).resolve(dependent)
    assert not unavailable.valid
    assert unavailable.error is not None
    assert unavailable.error.code == "missing_feature"


def test_external_measurement_feature_resolver_can_supply_live_geometry() -> None:
    measurement_ref = LiveFeatureRef(
        DOCUMENT_ID,
        "measurement-1",
        SourceObjectKind.MEASUREMENT,
        "geometry",
    )
    midpoint = _entity("midpoint", MidpointDefinition(measurement_ref))
    seen: list[LiveFeatureRef] = []

    def external(source: LiveFeatureRef) -> ResolvedLine:
        seen.append(source)
        return ResolvedLine(Point(2, 4), Point(8, 4))

    result = ConstructionResolver(
        DOCUMENT_ID,
        [midpoint],
        external_feature_resolver=external,
    ).resolve("midpoint")

    assert result.valid
    assert isinstance(result.geometry, ResolvedPoint)
    _assert_point(result.geometry.point, 5, 4)
    assert seen == [measurement_ref]


def test_cross_document_live_reference_is_invalid_even_with_external_resolver() -> None:
    source = LiveFeatureRef("another-document", "measurement-1", SourceObjectKind.MEASUREMENT)
    midpoint = _entity("midpoint", MidpointDefinition(source))
    result = ConstructionResolver(
        DOCUMENT_ID,
        [midpoint],
        external_feature_resolver=lambda _source: ResolvedLine(Point(0, 0), Point(1, 0)),
    ).resolve("midpoint")

    assert result.error is not None
    assert result.error.code == "cross_document_reference"


def test_graph_validation_rejects_duplicate_missing_cross_document_and_cycle() -> None:
    point = _entity("point", FreePointDefinition(Point(0, 0)))
    with pytest.raises(ConstructionValidationError) as duplicate:
        validate_construction_graph(DOCUMENT_ID, [point, point])
    assert duplicate.value.code == "duplicate_id"

    missing = _entity("missing-user", MidpointDefinition(_ref("does-not-exist")))
    with pytest.raises(ConstructionValidationError) as missing_error:
        validate_construction_graph(DOCUMENT_ID, [missing])
    assert missing_error.value.code == "missing_source"

    cross_document = _entity(
        "cross-document",
        MidpointDefinition(LiveFeatureRef("other", "measurement", SourceObjectKind.MEASUREMENT)),
    )
    with pytest.raises(ConstructionValidationError) as cross_error:
        validate_construction_graph(DOCUMENT_ID, [cross_document])
    assert cross_error.value.code == "cross_document_reference"

    first = _entity("first", MidpointDefinition(_ref("second")))
    second = _entity("second", MidpointDefinition(_ref("first")))
    with pytest.raises(ConstructionValidationError) as cycle_error:
        validate_construction_graph(DOCUMENT_ID, [first, second])
    assert cycle_error.value.code == "dependency_cycle"
    assert cycle_error.value.entity_ids[0] == cycle_error.value.entity_ids[-1]


def test_resolver_preserves_cycle_as_invalid_geometry_instead_of_recursing() -> None:
    first = _entity("first", MidpointDefinition(_ref("second")))
    second = _entity("second", MidpointDefinition(_ref("first")))

    result = ConstructionResolver(DOCUMENT_ID, [first, second]).resolve("first")

    assert not result.valid
    assert result.error is not None
    assert result.error.code == "unresolved_dependency"
    assert "形成环" in result.error.message


def test_resolver_and_graph_validation_handle_very_deep_dependency_chain() -> None:
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
                OffsetParallelDefinition(_ref(f"deep-{index - 1}"), 1.0),
            )
        )

    result = ConstructionResolver(DOCUMENT_ID, entities).resolve(f"deep-{depth}")

    assert result.valid
    assert isinstance(result.geometry, ResolvedLine)
    assert result.geometry.start == Point(0.0, float(depth))
    assert result.geometry.end == Point(10.0, float(depth))
    validate_construction_graph(DOCUMENT_ID, entities)


def test_dependency_helpers_report_direct_and_transitive_relations() -> None:
    source = _entity("source", LineDefinition(Point(0, 0), Point(10, 0)))
    midpoint = _entity("midpoint", MidpointDefinition(_ref("source")))
    through = _entity(
        "through",
        ParallelThroughPointDefinition(_ref("source"), Point(0, 2)),
    )
    chained = _entity(
        "chained",
        ParallelThroughPointDefinition(_ref("through"), Point(0, 4)),
    )
    entities = [source, midpoint, through, chained]

    assert iter_live_refs(midpoint.definition) == (_ref("source"),)
    assert live_dependency_ids(
        through,
        source_kind=SourceObjectKind.CONSTRUCTION,
    ) == ("source",)
    assert transitive_dependents(
        entities,
        ["source"],
        source_kind=SourceObjectKind.CONSTRUCTION,
    ) == (
        "midpoint",
        "through",
        "chained",
    )


def test_core_dependency_and_detach_helpers_separate_same_id_object_kinds() -> None:
    shared = _entity("shared", LineDefinition(Point(0, 0), Point(10, 0)))
    construction_child = _entity(
        "construction-child",
        MidpointDefinition(_ref("shared")),
    )
    measurement_ref = LiveFeatureRef(
        DOCUMENT_ID,
        "shared",
        SourceObjectKind.MEASUREMENT,
    )
    measurement_child = _entity(
        "measurement-child",
        MidpointDefinition(measurement_ref),
    )
    entities = (shared, construction_child, measurement_child)
    resolver = ConstructionResolver(
        DOCUMENT_ID,
        entities,
        external_feature_resolver=lambda _source: ResolvedLine(
            Point(0, 2),
            Point(10, 2),
        ),
    )

    assert transitive_dependents(
        entities,
        ("shared",),
        source_kind=SourceObjectKind.CONSTRUCTION,
    ) == ("construction-child",)
    assert transitive_dependents(
        entities,
        ("shared",),
        source_kind=SourceObjectKind.MEASUREMENT,
    ) == ("measurement-child",)

    detached = detach_sources(
        entities,
        ("shared",),
        resolver,
        source_kind=SourceObjectKind.CONSTRUCTION,
    )
    detached_construction_child = detached[1]
    assert isinstance(
        detached_construction_child.definition,
        MidpointDefinition,
    )
    assert isinstance(
        detached_construction_child.definition.source,
        FrozenFeatureSnapshot,
    )
    detached_measurement_child = detached[2]
    assert detached_measurement_child is measurement_child
    assert isinstance(detached_measurement_child.definition, MidpointDefinition)
    assert detached_measurement_child.definition.source == measurement_ref


def test_detach_sources_freezes_exact_selected_feature_without_owning_revision() -> None:
    source = _entity("source", LineDefinition(Point(0, 0), Point(10, 0)))
    midpoint = _entity("midpoint", MidpointDefinition(_ref("source")))
    other = _entity("other", FreePointDefinition(Point(3, 4)))
    resolver = ConstructionResolver(DOCUMENT_ID, [source, midpoint, other])

    detached = detach_sources(
        [source, midpoint, other],
        ["source"],
        resolver,
        source_kind=SourceObjectKind.CONSTRUCTION,
    )
    detached_midpoint = detached[1]

    assert isinstance(detached_midpoint.definition, MidpointDefinition)
    assert isinstance(detached_midpoint.definition.source, FrozenFeatureSnapshot)
    assert detached_midpoint.revision == midpoint.revision
    assert detached[0] is source
    assert detached[2] is other

    without_source = [detached_midpoint, other]
    result = ConstructionResolver(DOCUMENT_ID, without_source).resolve("midpoint")
    assert isinstance(result.geometry, ResolvedPoint)
    _assert_point(result.geometry.point, 5, 0)


def test_detach_rejects_an_already_unresolved_source() -> None:
    invalid_source = _entity("source", LineDefinition(Point(0, 0), Point(0, 0)))
    midpoint = _entity("midpoint", MidpointDefinition(_ref("source")))
    resolver = ConstructionResolver(DOCUMENT_ID, [invalid_source, midpoint])

    with pytest.raises(ConstructionValidationError) as error:
        detach_sources(
            [invalid_source, midpoint],
            ["source"],
            resolver,
            source_kind=SourceObjectKind.CONSTRUCTION,
        )
    assert error.value.code == "detach_unresolved"


def test_concentric_and_offset_circles_follow_the_live_source_center_and_radius() -> None:
    source = _entity("source", CircleCenterRadiusDefinition(Point(3, 4), 5))
    concentric = _entity("concentric", ConcentricCircleDefinition(_ref("source"), 8))
    offset = _entity("offset", OffsetCircleDefinition(_ref("source"), -2))
    resolver = ConstructionResolver(DOCUMENT_ID, [source, concentric, offset])

    resolved_concentric = resolver.resolve("concentric").geometry
    resolved_offset = resolver.resolve("offset").geometry
    assert isinstance(resolved_concentric, ResolvedCircle)
    assert isinstance(resolved_offset, ResolvedCircle)
    _assert_point(resolved_concentric.center, 3, 4)
    _assert_point(resolved_offset.center, 3, 4)
    assert resolved_concentric.radius == pytest.approx(8)
    assert resolved_offset.radius == pytest.approx(3)

    moved = replace(
        source,
        definition=CircleCenterRadiusDefinition(Point(10, 12), 7),
        revision=1,
    )
    moved_resolver = ConstructionResolver(DOCUMENT_ID, [moved, concentric, offset])
    moved_offset = moved_resolver.resolve("offset").geometry
    assert isinstance(moved_offset, ResolvedCircle)
    _assert_point(moved_offset.center, 10, 12)
    assert moved_offset.radius == pytest.approx(5)


def test_circle_offset_that_collapses_radius_is_structurally_invalid() -> None:
    source = _entity("source", CircleCenterRadiusDefinition(Point(0, 0), 3))
    offset = _entity("offset", OffsetCircleDefinition(_ref("source"), -3))

    result = ConstructionResolver(DOCUMENT_ID, [source, offset]).resolve("offset")

    assert not result.valid
    assert result.error is not None
    assert result.error.code == "degenerate_geometry"


def test_point_to_circle_has_two_stable_tangents_outside_one_on_and_none_inside() -> None:
    circle = ResolvedCircle(Point(0, 0), 5)

    outside, issue = point_circle_tangent_lines(Point(13, 0), circle)
    assert issue is None
    assert len(outside) == 2
    assert outside[0].end.y < 0 < outside[1].end.y
    for line in outside:
        _assert_point(line.start, 13, 0)
        assert math.hypot(line.end.x, line.end.y) == pytest.approx(5)
        radius = (line.end.x, line.end.y)
        tangent = (line.start.x - line.end.x, line.start.y - line.end.y)
        assert radius[0] * tangent[0] + radius[1] * tangent[1] == pytest.approx(0)

    on_circle, issue = point_circle_tangent_lines(Point(5, 0), circle)
    assert issue is None
    assert len(on_circle) == 1
    assert on_circle[0].direction == pytest.approx((0, 1))

    inside, issue = point_circle_tangent_lines(Point(1, 0), circle)
    assert inside == ()
    assert issue is not None and issue.code == "no_tangent_solution"


def test_point_circle_tangent_definition_associates_both_sources_and_branch() -> None:
    point = _entity("point", FreePointDefinition(Point(13, 0)))
    circle = _entity("circle", CircleCenterRadiusDefinition(Point(0, 0), 5))
    tangent = _entity(
        "tangent",
        PointCircleTangentDefinition(_ref("point"), _ref("circle"), branch=1),
    )

    result = ConstructionResolver(DOCUMENT_ID, [point, circle, tangent]).resolve("tangent")

    assert isinstance(result.geometry, ResolvedLine)
    assert result.geometry.end.y > 0
    assert result.dependencies == ("point", "circle")
    assert live_dependency_ids(
        tangent,
        source_kind=SourceObjectKind.CONSTRUCTION,
    ) == ("point", "circle")


def test_external_and_internal_common_tangents_are_analytic_and_branch_stable() -> None:
    first = ResolvedCircle(Point(0, 0), 2)
    second = ResolvedCircle(Point(10, 0), 2)

    external, issue = common_tangent_lines(first, second, CommonTangentMode.EXTERNAL)
    assert issue is None
    assert len(external) == 2
    assert external[0].start.y == pytest.approx(-2)
    assert external[1].start.y == pytest.approx(2)
    assert external[0].end.y == pytest.approx(-2)
    assert external[1].end.y == pytest.approx(2)

    internal, issue = common_tangent_lines(first, second, CommonTangentMode.INTERNAL)
    assert issue is None
    assert len(internal) == 2
    for line in internal:
        assert _distance_to_infinite_line(first.center, line) == pytest.approx(first.radius)
        assert _distance_to_infinite_line(second.center, line) == pytest.approx(second.radius)
    assert internal[0].start.y < 0 < internal[1].start.y


def test_common_tangent_reports_containment_and_coincident_locus() -> None:
    contained, issue = common_tangent_lines(
        ResolvedCircle(Point(0, 0), 5),
        ResolvedCircle(Point(1, 0), 1),
        CommonTangentMode.EXTERNAL,
    )
    assert contained == ()
    assert issue is not None and issue.code == "no_tangent_solution"

    coincident, issue = common_tangent_lines(
        ResolvedCircle(Point(0, 0), 5),
        ResolvedCircle(Point(0, 0), 5),
        CommonTangentMode.EXTERNAL,
    )
    assert coincident == ()
    assert issue is not None and issue.code == "coincident_tangent_locus"


def test_common_tangent_definition_resolves_selected_internal_branch() -> None:
    first = _entity("first", CircleCenterRadiusDefinition(Point(0, 0), 2))
    second = _entity("second", CircleCenterRadiusDefinition(Point(10, 0), 2))
    tangent = _entity(
        "tangent",
        CommonTangentDefinition(
            _ref("first"),
            _ref("second"),
            CommonTangentMode.INTERNAL,
            branch=1,
        ),
    )

    result = ConstructionResolver(DOCUMENT_ID, [first, second, tangent]).resolve("tangent")

    assert isinstance(result.geometry, ResolvedLine)
    assert result.geometry.start.y > 0


def test_fixed_radius_tangent_circle_supports_line_line_line_circle_and_circle_circle() -> None:
    horizontal = ResolvedLine(Point(0, 0), Point(10, 0), LineExtent.INFINITE)
    vertical = ResolvedLine(Point(0, 0), Point(0, 10), LineExtent.INFINITE)
    line_line, issue = tangent_tangent_radius_solutions(
        horizontal,
        vertical,
        2,
        TangencyConstraint(1),
        TangencyConstraint(1),
    )
    assert issue is None
    assert len(line_line) == 1
    _assert_point(line_line[0].circle.center, -2, 2)
    assert line_line[0].tangent_points == (Point(-2, 0), Point(0, 2))

    circle = ResolvedCircle(Point(0, 4), 1)
    line_circle, issue = tangent_tangent_radius_solutions(
        horizontal,
        circle,
        2,
        TangencyConstraint(1),
        TangencyConstraint(circle_relation=CircleTangency.EXTERNAL),
    )
    assert issue is None
    assert len(line_circle) == 2
    assert line_circle[0].circle.center.x < 0 < line_circle[1].circle.center.x
    for solution in line_circle:
        assert solution.circle.center.y == pytest.approx(2)
        assert math.hypot(solution.circle.center.x, -2) == pytest.approx(3)

    first_circle = ResolvedCircle(Point(0, 0), 2)
    second_circle = ResolvedCircle(Point(4, 0), 2)
    circle_circle, issue = tangent_tangent_radius_solutions(
        first_circle,
        second_circle,
        1,
    )
    assert issue is None
    assert len(circle_circle) == 2
    assert circle_circle[0].circle.center.y < 0 < circle_circle[1].circle.center.y


def test_fixed_radius_line_circle_branches_are_stable_across_world_rotation() -> None:
    def rotate(point: Point, degrees: float) -> Point:
        angle = math.radians(degrees)
        cosine = math.cos(angle)
        sine = math.sin(angle)
        return Point(
            point.x * cosine - point.y * sine,
            point.x * sine + point.y * cosine,
        )

    for degrees in (89.0, 91.0):
        line = ResolvedLine(
            rotate(Point(-10.0, 0.0), degrees),
            rotate(Point(10.0, 0.0), degrees),
            LineExtent.INFINITE,
        )
        source_circle = ResolvedCircle(rotate(Point(0.0, 4.0), degrees), 1.0)
        solutions, issue = tangent_tangent_radius_solutions(
            line,
            source_circle,
            2.0,
            TangencyConstraint(line_side=1),
            TangencyConstraint(circle_relation=CircleTangency.EXTERNAL),
        )
        assert issue is None and len(solutions) == 2
        local_centers = {
            solution.branch: rotate(solution.circle.center, -degrees)
            for solution in solutions
        }
        assert local_centers[0].x < 0.0 < local_centers[1].x
        assert local_centers[0].y == pytest.approx(2.0)
        assert local_centers[1].y == pytest.approx(2.0)


def test_fixed_radius_tangent_circle_supports_both_internal_containment_directions() -> None:
    horizontal = ResolvedLine(Point(-10, 0), Point(10, 0), LineExtent.INFINITE)
    source = ResolvedCircle(Point(0, 5), 5)

    source_contains, issue = tangent_tangent_radius_solutions(
        horizontal,
        source,
        2,
        TangencyConstraint(1),
        TangencyConstraint(circle_relation=CircleTangency.SOURCE_CONTAINS),
    )
    assert issue is None
    assert len(source_contains) == 1
    _assert_point(source_contains[0].circle.center, 0, 2)

    small_source = ResolvedCircle(Point(0, -1), 1)
    solution_contains, issue = tangent_tangent_radius_solutions(
        horizontal,
        small_source,
        6,
        TangencyConstraint(-1),
        TangencyConstraint(circle_relation=CircleTangency.SOLUTION_CONTAINS),
    )
    assert issue is None
    assert len(solution_contains) == 1
    _assert_point(solution_contains[0].circle.center, 0, -6)


def test_fixed_radius_tangent_circle_respects_finite_line_domain_unless_extended() -> None:
    horizontal_segment = ResolvedLine(Point(0, 0), Point(1, 0), LineExtent.SEGMENT)
    vertical = ResolvedLine(Point(10, 0), Point(10, 10), LineExtent.INFINITE)
    constraints = (TangencyConstraint(1), TangencyConstraint(1))

    solutions, issue = tangent_tangent_radius_solutions(
        horizontal_segment,
        vertical,
        2,
        *constraints,
    )
    assert solutions == ()
    assert issue is not None and issue.code == "no_tangent_solution"

    extended, issue = tangent_tangent_radius_solutions(
        horizontal_segment,
        vertical,
        2,
        *constraints,
        extend=True,
    )
    assert issue is None
    assert len(extended) == 1
    _assert_point(extended[0].circle.center, 8, 2)


def test_tangent_tangent_radius_definition_resolves_and_tracks_sources() -> None:
    horizontal = _entity(
        "horizontal",
        LineDefinition(Point(0, 0), Point(10, 0), LineExtent.INFINITE),
    )
    vertical = _entity(
        "vertical",
        LineDefinition(Point(0, 0), Point(0, 10), LineExtent.INFINITE),
    )
    tangent_circle = _entity(
        "ttr",
        TangentTangentRadiusCircleDefinition(
            _ref("horizontal"),
            _ref("vertical"),
            2,
            TangencyConstraint(1),
            TangencyConstraint(-1),
        ),
    )

    result = ConstructionResolver(
        DOCUMENT_ID,
        [horizontal, vertical, tangent_circle],
    ).resolve("ttr")

    assert isinstance(result.geometry, ResolvedCircle)
    _assert_point(result.geometry.center, 2, 2)


def test_three_line_tangent_circle_solves_triangle_incircle_analytically() -> None:
    bottom = ResolvedLine(Point(0, 0), Point(10, 0), LineExtent.INFINITE)
    left = ResolvedLine(Point(0, 0), Point(0, 10), LineExtent.INFINITE)
    diagonal = ResolvedLine(Point(10, 0), Point(0, 10), LineExtent.INFINITE)

    solutions, issue = three_tangent_circle_solutions(
        (bottom, left, diagonal),
        (TangencyConstraint(1), TangencyConstraint(-1), TangencyConstraint(1)),
    )

    assert issue is None
    assert len(solutions) == 1
    expected_radius = 10.0 / (2.0 + math.sqrt(2.0))
    circle = solutions[0].circle
    _assert_point(circle.center, expected_radius, expected_radius)
    assert circle.radius == pytest.approx(expected_radius)
    assert solutions[0].tangent_points[2] == Point(5, 5)


def test_three_circle_tangent_circle_solves_external_apollonius_case() -> None:
    sources = (
        ResolvedCircle(Point(0, 0), 1),
        ResolvedCircle(Point(10, 0), 1),
        ResolvedCircle(Point(5, 5 * math.sqrt(3)), 1),
    )

    solutions, issue = three_tangent_circle_solutions(sources)

    assert issue is None
    assert len(solutions) == 1
    # The smaller algebraic root has a non-positive radius and is filtered,
    # but the surviving solution retains its raw branch identity.
    assert solutions[0].branch == 1
    _assert_point(solutions[0].circle.center, 5, 5 / math.sqrt(3))
    assert solutions[0].circle.radius == pytest.approx(10 / math.sqrt(3) - 1)


def test_three_source_solver_handles_mixed_line_and_circles() -> None:
    line = ResolvedLine(Point(-20, 0), Point(20, 0), LineExtent.INFINITE)
    first = ResolvedCircle(Point(0, 5), 1)
    second = ResolvedCircle(Point(8, 5), 1)

    solutions, issue = three_tangent_circle_solutions(
        (line, first, second),
        (
            TangencyConstraint(1),
            TangencyConstraint(circle_relation=CircleTangency.EXTERNAL),
            TangencyConstraint(circle_relation=CircleTangency.EXTERNAL),
        ),
    )

    assert issue is None
    assert len(solutions) == 1
    _assert_point(solutions[0].circle.center, 4, 10 / 3)
    assert solutions[0].circle.radius == pytest.approx(10 / 3)


def test_three_source_solver_handles_two_lines_and_one_circle_with_stable_branches() -> None:
    horizontal = ResolvedLine(Point(0, 0), Point(10, 0), LineExtent.INFINITE)
    vertical = ResolvedLine(Point(0, 0), Point(0, 10), LineExtent.INFINITE)
    source_circle = ResolvedCircle(Point(5, 5), 1)

    solutions, issue = three_tangent_circle_solutions(
        (horizontal, vertical, source_circle),
        (
            TangencyConstraint(1),
            TangencyConstraint(-1),
            TangencyConstraint(circle_relation=CircleTangency.EXTERNAL),
        ),
    )

    assert issue is None
    assert [solution.branch for solution in solutions] == [0, 1]
    assert solutions[0].circle.radius < solutions[1].circle.radius
    for solution in solutions:
        circle = solution.circle
        _assert_point(circle.center, circle.radius, circle.radius)
        assert math.hypot(circle.center.x - 5, circle.center.y - 5) == pytest.approx(
            circle.radius + 1
        )


def test_three_tangent_branch_survives_a_new_smaller_positive_root() -> None:
    horizontal_geometry = ResolvedLine(
        Point(0, 0),
        Point(10, 0),
        LineExtent.INFINITE,
    )
    vertical_geometry = ResolvedLine(
        Point(0, 0),
        Point(0, 10),
        LineExtent.INFINITE,
    )
    constraints = (
        TangencyConstraint(1),
        TangencyConstraint(-1),
        TangencyConstraint(circle_relation=CircleTangency.EXTERNAL),
    )

    initial_solutions, issue = three_tangent_circle_solutions(
        (
            horizontal_geometry,
            vertical_geometry,
            ResolvedCircle(Point(0.707, 0.707), 1),
        ),
        constraints,
    )

    assert issue is None
    assert len(initial_solutions) == 1
    stable_branch = initial_solutions[0].branch
    assert stable_branch == 1
    assert initial_solutions[0].circle.radius == pytest.approx(4.828062550970872)

    moved_solutions, issue = three_tangent_circle_solutions(
        (
            horizontal_geometry,
            vertical_geometry,
            ResolvedCircle(Point(0.708, 0.708), 1),
        ),
        constraints,
    )

    assert issue is None
    assert [solution.branch for solution in moved_solutions] == [0, 1]
    continued = next(
        solution for solution in moved_solutions if solution.branch == stable_branch
    )
    assert continued.circle.radius == pytest.approx(4.831476764533247)

    horizontal = _entity(
        "horizontal",
        LineDefinition(Point(0, 0), Point(10, 0), LineExtent.INFINITE),
    )
    vertical = _entity(
        "vertical",
        LineDefinition(Point(0, 0), Point(0, 10), LineExtent.INFINITE),
    )
    moved_circle = _entity(
        "circle",
        CircleCenterRadiusDefinition(Point(0.708, 0.708), 1),
    )
    persisted = ConstructionEntity.from_dict(
        _entity(
            "three",
            ThreeTangentCircleDefinition(
                _ref("horizontal"),
                _ref("vertical"),
                _ref("circle"),
                *constraints,
                branch=stable_branch,
            ),
        ).to_dict()
    )

    resolved = ConstructionResolver(
        DOCUMENT_ID,
        [horizontal, vertical, moved_circle, persisted],
    ).resolve("three")

    assert isinstance(resolved.geometry, ResolvedCircle)
    assert resolved.geometry.radius == pytest.approx(continued.circle.radius)
    _assert_point(
        resolved.geometry.center,
        continued.circle.center.x,
        continued.circle.center.y,
    )


def test_three_tangent_branch_is_stable_across_affine_direction_flip() -> None:
    constraints = (
        TangencyConstraint(1),
        TangencyConstraint(-1),
        TangencyConstraint(circle_relation=CircleTangency.EXTERNAL),
    )

    def rotate(point: Point, degrees: float) -> Point:
        angle = math.radians(degrees)
        cosine = math.cos(angle)
        sine = math.sin(angle)
        return Point(
            point.x * cosine - point.y * sine,
            point.x * sine + point.y * cosine,
        )

    def source_geometry(degrees: float):
        return (
            ResolvedLine(
                rotate(Point(0, 0), degrees),
                rotate(Point(10, 0), degrees),
                LineExtent.INFINITE,
            ),
            ResolvedLine(
                rotate(Point(0, 0), degrees),
                rotate(Point(0, 10), degrees),
                LineExtent.INFINITE,
            ),
            ResolvedCircle(rotate(Point(5, 5), degrees), 1),
        )

    initial_solutions, issue = three_tangent_circle_solutions(
        source_geometry(90.0),
        constraints,
    )
    assert issue is None
    assert [solution.branch for solution in initial_solutions] == [0, 1]
    stable_branch = initial_solutions[0].branch
    assert initial_solutions[0].circle.radius == pytest.approx(2.51471862576143)

    moved_solutions, issue = three_tangent_circle_solutions(
        source_geometry(90.01),
        constraints,
    )
    assert issue is None
    continued = next(
        solution for solution in moved_solutions if solution.branch == stable_branch
    )
    assert continued.circle.radius == pytest.approx(
        initial_solutions[0].circle.radius
    )

    moved_sources = source_geometry(90.01)
    horizontal = _entity(
        "horizontal",
        LineDefinition(
            moved_sources[0].start,
            moved_sources[0].end,
            LineExtent.INFINITE,
        ),
    )
    vertical = _entity(
        "vertical",
        LineDefinition(
            moved_sources[1].start,
            moved_sources[1].end,
            LineExtent.INFINITE,
        ),
    )
    circle = _entity(
        "circle",
        CircleCenterRadiusDefinition(moved_sources[2].center, 1),
    )
    persisted = ConstructionEntity.from_dict(
        _entity(
            "three",
            ThreeTangentCircleDefinition(
                _ref("horizontal"),
                _ref("vertical"),
                _ref("circle"),
                *constraints,
                branch=stable_branch,
            ),
        ).to_dict()
    )

    resolved = ConstructionResolver(
        DOCUMENT_ID,
        [horizontal, vertical, circle, persisted],
    ).resolve("three")

    assert isinstance(resolved.geometry, ResolvedCircle)
    assert resolved.geometry.radius == pytest.approx(continued.circle.radius)
    _assert_point(
        resolved.geometry.center,
        continued.circle.center.x,
        continued.circle.center.y,
    )


def test_three_tangent_definition_resolves_branch_and_serializes_all_sources() -> None:
    bottom = _entity(
        "bottom",
        LineDefinition(Point(0, 0), Point(10, 0), LineExtent.INFINITE),
    )
    left = _entity(
        "left",
        LineDefinition(Point(0, 0), Point(0, 10), LineExtent.INFINITE),
    )
    diagonal = _entity(
        "diagonal",
        LineDefinition(Point(10, 0), Point(0, 10), LineExtent.INFINITE),
    )
    definition = ThreeTangentCircleDefinition(
        _ref("bottom"),
        _ref("left"),
        _ref("diagonal"),
        TangencyConstraint(1),
        TangencyConstraint(-1),
        TangencyConstraint(1),
    )
    entity = _entity("three", definition)

    result = ConstructionResolver(
        DOCUMENT_ID,
        [bottom, left, diagonal, entity],
    ).resolve("three")

    assert isinstance(result.geometry, ResolvedCircle)
    assert live_dependency_ids(
        entity,
        source_kind=SourceObjectKind.CONSTRUCTION,
    ) == ("bottom", "left", "diagonal")
    assert ConstructionEntity.from_dict(entity.to_dict()) == entity


def test_three_tangent_solver_reports_underdetermined_and_no_positive_solution() -> None:
    parallel = (
        ResolvedLine(Point(0, 0), Point(10, 0), LineExtent.INFINITE),
        ResolvedLine(Point(0, 2), Point(10, 2), LineExtent.INFINITE),
        ResolvedLine(Point(0, 4), Point(10, 4), LineExtent.INFINITE),
    )

    solutions, issue = three_tangent_circle_solutions(parallel)

    assert solutions == ()
    assert issue is not None
    assert issue.code in {"underdetermined_tangent_system", "no_tangent_solution"}


def test_tangent_solvers_return_structured_issue_for_point_or_array_sources() -> None:
    point = ResolvedPoint(Point(0, 0))
    line = ResolvedLine(Point(0, 0), Point(10, 0))
    array = ResolvedLineArray((line,))

    fixed, issue = tangent_tangent_radius_solutions(point, line, 2)
    assert fixed == ()
    assert issue is not None and issue.code == "unsupported_tangent_source"

    three, issue = three_tangent_circle_solutions((line, array, line))
    assert three == ()
    assert issue is not None and issue.code == "unsupported_tangent_source"


def test_all_serialized_numeric_values_are_finite() -> None:
    with pytest.raises(ValueError, match="有限"):
        FreePointDefinition(Point(math.nan, 0))
    with pytest.raises(ValueError, match="有限"):
        CircleCenterRadiusDefinition(Point(0, 0), math.inf)
    with pytest.raises(ValueError, match="有限"):
        OffsetParallelDefinition(
            FrozenFeatureSnapshot(ResolvedLine(Point(0, 0), Point(1, 0))),
            math.nan,
        )


def _distance_to_infinite_line(point: Point, line: ResolvedLine) -> float:
    direction_x, direction_y = line.direction
    return abs(
        (point.x - line.start.x) * (-direction_y)
        + (point.y - line.start.y) * direction_x
    )
