from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

import pytest

import fdm.services.object_snap_service as object_snap_service

from fdm.construction_geometry import (
    ConstructionEntity,
    ConstructionResolver,
    FreePointDefinition,
    IntersectionDefinition,
    LineDefinition,
    LineExtent,
    LiveFeatureRef,
    MidpointDefinition,
)
from fdm.geometry import Line, Point
from fdm.models import Measurement, OverlayAnnotation, OverlayAnnotationKind
from fdm.services.object_snap_service import (
    DEFAULT_SNAP_KINDS,
    ObjectSnapEngine,
    ObjectSnapSettings,
    SnapCandidate,
    SnapKind,
    contextual_snap_candidate,
)
from fdm.settings import AppSettings


class _Extent(str, Enum):
    SEGMENT = "segment"
    RAY = "ray"
    INFINITE = "infinite"


@dataclass(frozen=True)
class _Entity:
    id: str
    visible: bool = True
    locked: bool = False
    snappable: bool = True


@dataclass(frozen=True)
class _ResolvedPoint:
    point: Point


@dataclass(frozen=True)
class _ResolvedLine:
    start: Point
    end: Point
    extent: _Extent = _Extent.SEGMENT


@dataclass(frozen=True)
class _ResolvedCircle:
    center: Point
    radius: float


@dataclass(frozen=True)
class _ResolvedLineArray:
    lines: tuple[_ResolvedLine, ...]


@dataclass(frozen=True)
class _ResolvedConstruction:
    entity_id: str
    geometry: object | None
    error: str | None = None

    @property
    def valid(self) -> bool:
        return self.geometry is not None and self.error is None


def _resolved(entity: _Entity, geometry: object) -> tuple[_Entity, _ResolvedConstruction]:
    return entity, _ResolvedConstruction(entity.id, geometry)


def _measurement(
    identifier: str,
    *,
    kind: str = "line",
    mode: str = "manual",
    line: Line | None = None,
    points: list[Point] | None = None,
    point: Point | None = None,
) -> Measurement:
    return Measurement(
        id=identifier,
        image_id="image",
        fiber_group_id=None,
        mode=mode,
        measurement_kind=kind,
        line_px=line,
        polyline_px=list(points or []) if kind == "polyline" else [],
        polygon_px=list(points or []) if kind == "area" else [],
        point_px=point,
    )


def test_default_settings_enable_semantic_targets_but_not_nearest() -> None:
    settings = ObjectSnapSettings()

    assert settings.enabled_kinds == DEFAULT_SNAP_KINDS
    assert settings.allows(SnapKind.ENDPOINT)
    assert settings.allows(SnapKind.INTERSECTION)
    assert not settings.allows(SnapKind.NEAREST)
    assert ObjectSnapSettings.from_dict(settings.to_dict()) == settings


def test_contextual_candidate_bypasses_kind_filter_but_uses_engine_aperture() -> None:
    engine = ObjectSnapEngine(
        ObjectSnapSettings(enabled_kinds=frozenset(), aperture_px=6.0)
    )
    inside = contextual_snap_candidate(
        Point(10.0, 10.0),
        kind=SnapKind.PERPENDICULAR,
        source_id="line:source",
        feature_key="perpendicular:foot",
        cursor_image_px=Point(15.0, 10.0),
    )

    candidate = engine.query(
        Point(15.0, 10.0),
        contextual_candidates=(inside,),
    )

    assert candidate is not None
    assert candidate.kind is SnapKind.PERPENDICULAR
    assert candidate.label == "垂足"

    engine.clear_hysteresis()
    outside = contextual_snap_candidate(
        Point(10.0, 10.0),
        kind=SnapKind.TANGENT,
        source_id="circle:source",
        feature_key="tangent:0",
        cursor_image_px=Point(17.0, 10.0),
    )
    assert (
        engine.query(
            Point(17.0, 10.0),
            contextual_candidates=(outside,),
        )
        is None
    )


def test_real_construction_resolver_output_is_consumed_without_conversion() -> None:
    entity = ConstructionEntity(
        id="real-point",
        name="基准点",
        definition=FreePointDefinition(Point(12.0, 8.0)),
        locked=True,
    )
    resolved = ConstructionResolver("document", [entity]).resolve(entity)
    engine = ObjectSnapEngine(ObjectSnapSettings(aperture_px=2.0))

    candidate = engine.query(
        Point(12.5, 8.0),
        constructions=[(entity, resolved)],
    )

    assert candidate is not None
    assert candidate.kind is SnapKind.POINT
    assert candidate.source_id == entity.id
    assert candidate.point_px == Point(12.0, 8.0)


def test_measurement_line_emits_endpoints_and_midpoint_in_screen_pixels() -> None:
    measurement = _measurement(
        "m1",
        line=Line(Point(10.0, 10.0), Point(30.0, 10.0)),
    )
    engine = ObjectSnapEngine(ObjectSnapSettings(aperture_px=5.0))

    candidates = engine.candidates(
        Point(10.9, 10.0),
        image_to_screen=lambda point: (point.x * 2.0, point.y * 2.0),
        measurements=[measurement],
    )

    assert candidates[0].kind is SnapKind.ENDPOINT
    assert candidates[0].point_px == Point(10.0, 10.0)
    assert candidates[0].screen_distance_px == pytest.approx(1.8)
    midpoint_candidate = engine.query(
        Point(20.0, 10.0),
        image_to_screen=lambda point: (point.x * 2.0, point.y * 2.0),
        measurements=[measurement],
        previous=None,
    )
    assert midpoint_candidate is not None
    assert midpoint_candidate.kind is SnapKind.MIDPOINT


def test_distance_dominates_semantic_priority_and_semantics_break_ties() -> None:
    point_entity = _resolved(_Entity("point"), _ResolvedPoint(Point(10.0, 0.0)))
    line_entity = _resolved(
        _Entity("line"),
        _ResolvedLine(Point(11.0, 0.0), Point(21.0, 0.0)),
    )
    engine = ObjectSnapEngine(ObjectSnapSettings(aperture_px=20.0))

    closest = engine.query(Point(10.8, 0.0), constructions=[point_entity, line_entity])
    assert closest is not None
    assert closest.kind is SnapKind.ENDPOINT
    assert closest.source_id == "line"

    engine.clear_hysteresis()
    tied = engine.query(
        Point(10.5, 0.0),
        constructions=[point_entity, line_entity],
        previous=None,
    )
    # Explicit point has the lower semantic priority value at equal distance.
    assert tied is not None
    assert tied.kind is SnapKind.POINT


def test_hidden_and_no_snap_constructions_are_excluded_but_locked_is_snappable() -> None:
    constructions = [
        _resolved(_Entity("hidden", visible=False), _ResolvedPoint(Point(1.0, 1.0))),
        _resolved(_Entity("disabled", snappable=False), _ResolvedPoint(Point(2.0, 1.0))),
        _resolved(_Entity("locked", locked=True), _ResolvedPoint(Point(3.0, 1.0))),
        (
            _Entity("invalid"),
            _ResolvedConstruction("invalid", None, "退化几何"),
        ),
    ]
    engine = ObjectSnapEngine(ObjectSnapSettings(aperture_px=20.0))

    candidates = engine.candidates(Point(0.0, 0.0), constructions=constructions)

    assert {candidate.source_id for candidate in candidates} == {"locked"}


def test_circle_emits_center_and_four_quadrants() -> None:
    construction = _resolved(
        _Entity("circle"),
        _ResolvedCircle(Point(50.0, 50.0), 10.0),
    )
    engine = ObjectSnapEngine(ObjectSnapSettings(aperture_px=1.0))

    center = engine.query(Point(50.0, 50.0), constructions=[construction])
    assert center is not None
    assert center.kind is SnapKind.CENTER
    engine.clear_hysteresis()

    points: set[tuple[float, float]] = set()
    for point in (
        Point(60.0, 50.0),
        Point(50.0, 60.0),
        Point(40.0, 50.0),
        Point(50.0, 40.0),
    ):
        candidate = engine.query(point, constructions=[construction])
        assert candidate is not None
        points.add((candidate.point_px.x, candidate.point_px.y))
    assert points == {
        (60.0, 50.0),
        (50.0, 60.0),
        (40.0, 50.0),
        (50.0, 40.0),
    }


def test_intersections_cover_line_line_line_circle_and_circle_circle() -> None:
    horizontal = _resolved(
        _Entity("horizontal"),
        _ResolvedLine(Point(-20.0, 0.0), Point(20.0, 0.0), _Extent.INFINITE),
    )
    vertical = _resolved(
        _Entity("vertical"),
        _ResolvedLine(Point(0.0, -20.0), Point(0.0, 20.0), _Extent.INFINITE),
    )
    circle = _resolved(_Entity("circle"), _ResolvedCircle(Point(0.0, 0.0), 10.0))
    second_circle = _resolved(
        _Entity("circle2"),
        _ResolvedCircle(Point(10.0, 0.0), 10.0),
    )
    engine = ObjectSnapEngine(ObjectSnapSettings(aperture_px=0.5))

    line_line = engine.query(Point(0.0, 0.0), constructions=[horizontal, vertical])
    assert line_line is not None
    assert line_line.kind is SnapKind.INTERSECTION

    engine.clear_hysteresis()
    line_circle = engine.query(Point(10.0, 0.0), constructions=[horizontal, circle])
    assert line_circle is not None
    # A circle quadrant occupies the same coordinate.  The complete candidate
    # list still exposes the analytically derived intersection for disambiguation.
    assert any(
        candidate.kind is SnapKind.INTERSECTION
        for candidate in engine.candidates(
            Point(10.0, 0.0), constructions=[horizontal, circle]
        )
    )

    engine.clear_hysteresis()
    upper = Point(5.0, 5.0 * 3.0**0.5)
    circle_circle_candidates = engine.candidates(
        upper,
        constructions=[circle, second_circle],
    )
    assert any(
        candidate.kind is SnapKind.INTERSECTION
        and candidate.point_px.x == pytest.approx(upper.x)
        and candidate.point_px.y == pytest.approx(upper.y)
        for candidate in circle_circle_candidates
    )


def test_intersection_respects_segment_and_ray_domains() -> None:
    short_segment = _resolved(
        _Entity("segment"),
        _ResolvedLine(Point(0.0, 0.0), Point(2.0, 0.0), _Extent.SEGMENT),
    )
    backward_ray = _resolved(
        _Entity("ray"),
        _ResolvedLine(Point(5.0, 5.0), Point(5.0, 10.0), _Extent.RAY),
    )
    horizontal = _resolved(
        _Entity("horizontal"),
        _ResolvedLine(Point(0.0, 0.0), Point(1.0, 0.0), _Extent.INFINITE),
    )
    vertical = _resolved(
        _Entity("vertical"),
        _ResolvedLine(Point(5.0, -1.0), Point(5.0, 1.0), _Extent.INFINITE),
    )
    settings = ObjectSnapSettings(
        aperture_px=0.5,
        enabled_kinds=frozenset({SnapKind.INTERSECTION}),
    )
    engine = ObjectSnapEngine(settings)

    assert engine.query(Point(5.0, 0.0), constructions=[short_segment, vertical]) is None
    assert engine.query(Point(5.0, 0.0), constructions=[backward_ray, horizontal]) is None


def test_nearest_is_opt_in_for_lines_and_circles() -> None:
    line = _resolved(
        _Entity("line"),
        _ResolvedLine(Point(0.0, 0.0), Point(20.0, 0.0), _Extent.SEGMENT),
    )
    default_engine = ObjectSnapEngine(ObjectSnapSettings(aperture_px=3.0))
    assert default_engine.query(Point(7.0, 2.0), constructions=[line]) is None

    nearest_engine = ObjectSnapEngine(
        ObjectSnapSettings(
            aperture_px=3.0,
            enabled_kinds=frozenset({SnapKind.NEAREST}),
        )
    )
    candidate = nearest_engine.query(Point(7.0, 2.0), constructions=[line])
    assert candidate is not None
    assert candidate.kind is SnapKind.NEAREST
    assert candidate.point_px == Point(7.0, 0.0)


def test_hysteresis_retains_identity_until_a_materially_closer_target_exists() -> None:
    first = _resolved(_Entity("first"), _ResolvedPoint(Point(0.0, 0.0)))
    second = _resolved(_Entity("second"), _ResolvedPoint(Point(4.0, 0.0)))
    engine = ObjectSnapEngine(
        ObjectSnapSettings(aperture_px=5.0, hysteresis_px=2.0)
    )

    initial = engine.query(Point(0.5, 0.0), constructions=[first, second])
    assert initial is not None and initial.source_id == "first"

    retained = engine.query(Point(2.5, 0.0), constructions=[first, second])
    assert retained is not None and retained.source_id == "first"

    switched = engine.query(Point(3.8, 0.0), constructions=[first, second])
    assert switched is not None and switched.source_id == "second"


def test_polyline_polygon_and_count_are_analytical_but_freehand_magic_and_overlay_are_not() -> None:
    polyline = _measurement(
        "polyline",
        kind="polyline",
        mode="continuous_manual",
        points=[Point(0.0, 0.0), Point(10.0, 0.0), Point(10.0, 10.0)],
    )
    polygon = _measurement(
        "polygon",
        kind="area",
        mode="polygon_area",
        points=[Point(20.0, 0.0), Point(30.0, 0.0), Point(25.0, 10.0)],
    )
    count = _measurement(
        "count",
        kind="count",
        mode="count",
        point=Point(40.0, 0.0),
    )
    freehand = _measurement(
        "freehand",
        kind="area",
        mode="freehand_area",
        points=[Point(50.0, 0.0), Point(55.0, 0.0), Point(50.0, 5.0)],
    )
    magic = _measurement(
        "magic",
        kind="area",
        mode="magic_segment",
        points=[Point(60.0, 0.0), Point(65.0, 0.0), Point(60.0, 5.0)],
    )
    overlay = OverlayAnnotation(
        id="overlay",
        image_id="image",
        kind=OverlayAnnotationKind.LINE,
        start_px=Point(70.0, 0.0),
        end_px=Point(80.0, 0.0),
    )
    engine = ObjectSnapEngine(ObjectSnapSettings(aperture_px=0.5))

    assert engine.query(Point(10.0, 0.0), measurements=[polyline]) is not None
    engine.clear_hysteresis()
    assert engine.query(Point(25.0, 0.0), measurements=[polygon]) is not None
    engine.clear_hysteresis()
    assert engine.query(Point(40.0, 0.0), measurements=[count]) is not None
    engine.clear_hysteresis()
    assert engine.query(Point(50.0, 0.0), measurements=[freehand]) is None
    assert engine.query(Point(60.0, 0.0), measurements=[magic]) is None
    assert engine.query(Point(70.0, 0.0), measurements=[overlay]) is None  # type: ignore[list-item]


def test_line_array_children_are_queryable_without_materializing_entities() -> None:
    array = _resolved(
        _Entity("array", locked=True),
        _ResolvedLineArray(
            (
                _ResolvedLine(Point(0.0, 0.0), Point(20.0, 0.0), _Extent.INFINITE),
                _ResolvedLine(Point(0.0, 5.0), Point(20.0, 5.0), _Extent.INFINITE),
            )
        ),
    )
    crossing = _resolved(
        _Entity("crossing"),
        _ResolvedLine(Point(10.0, -10.0), Point(10.0, 10.0), _Extent.INFINITE),
    )
    engine = ObjectSnapEngine(
        ObjectSnapSettings(
            aperture_px=0.5,
            enabled_kinds=frozenset({SnapKind.INTERSECTION}),
        )
    )

    candidates = engine.candidates(Point(10.0, 5.0), constructions=[array, crossing])

    assert len(candidates) == 1
    assert candidates[0].point_px == Point(10.0, 5.0)
    assert candidates[0].kind is SnapKind.INTERSECTION


def test_resolved_midpoint_and_intersection_entities_keep_typed_snap_markers() -> None:
    document_id = "typed-snaps"
    horizontal = ConstructionEntity(
        id="horizontal",
        name="水平源线",
        definition=LineDefinition(Point(0.0, 0.0), Point(20.0, 0.0)),
        snap_enabled=False,
    )
    vertical = ConstructionEntity(
        id="vertical",
        name="垂直源线",
        definition=LineDefinition(
            Point(10.0, -10.0),
            Point(10.0, 10.0),
            LineExtent.INFINITE,
        ),
        snap_enabled=False,
    )
    horizontal_ref = LiveFeatureRef(document_id, horizontal.id)
    vertical_ref = LiveFeatureRef(document_id, vertical.id)
    midpoint = ConstructionEntity(
        id="midpoint",
        name="关联中点",
        definition=MidpointDefinition(horizontal_ref),
    )
    intersection = ConstructionEntity(
        id="intersection",
        name="关联交点",
        definition=IntersectionDefinition(horizontal_ref, vertical_ref),
    )
    entities = (horizontal, vertical, midpoint, intersection)
    resolver = ConstructionResolver(document_id, entities)
    entries = tuple((entity, resolver.resolve(entity)) for entity in entities)

    midpoint_engine = ObjectSnapEngine(
        ObjectSnapSettings(
            aperture_px=0.5,
            enabled_kinds=frozenset({SnapKind.MIDPOINT}),
        )
    )
    midpoint_candidate = midpoint_engine.query(
        Point(10.0, 0.0),
        constructions=entries,
    )
    assert midpoint_candidate is not None
    assert midpoint_candidate.kind is SnapKind.MIDPOINT
    assert midpoint_candidate.source_id == midpoint.id

    intersection_engine = ObjectSnapEngine(
        ObjectSnapSettings(
            aperture_px=0.5,
            enabled_kinds=frozenset({SnapKind.INTERSECTION}),
        )
    )
    intersection_candidate = intersection_engine.query(
        Point(10.0, 0.0),
        constructions=entries,
    )
    assert intersection_candidate is not None
    assert intersection_candidate.kind is SnapKind.INTERSECTION
    assert intersection_candidate.source_id == intersection.id
def test_explicit_empty_snap_filter_round_trips_across_sessions() -> None:
    settings = AppSettings(object_snap_enabled=True, object_snap_kinds=[])

    payload = settings.to_dict()
    restored = AppSettings.from_dict(payload)

    assert payload["object_snap_kinds"] == []
    assert restored.object_snap_kinds == []


def test_self_crossing_polyline_emits_intersection_but_not_adjacent_vertices() -> None:
    measurement = _measurement(
        "self-crossing-polyline",
        kind="polyline",
        mode="continuous_manual",
        points=[
            Point(0.0, 0.0),
            Point(10.0, 10.0),
            Point(0.0, 10.0),
            Point(10.0, 0.0),
        ],
    )
    engine = ObjectSnapEngine(
        ObjectSnapSettings(
            aperture_px=0.5,
            enabled_kinds=frozenset({SnapKind.INTERSECTION}),
        )
    )

    crossing = engine.candidates(
        Point(5.0, 5.0),
        measurements=[measurement],
    )
    adjacent_vertex = engine.candidates(
        Point(10.0, 10.0),
        measurements=[measurement],
    )

    assert len(crossing) == 1
    assert crossing[0].kind is SnapKind.INTERSECTION
    assert crossing[0].point_px == Point(5.0, 5.0)
    assert adjacent_vertex == ()


def test_dense_crossing_intersections_are_bounded_and_coalesced(monkeypatch) -> None:
    constructions = []
    for index in range(1000):
        angle = math.pi * index / 1000.0
        direction_x = math.cos(angle) * 100.0
        direction_y = math.sin(angle) * 100.0
        constructions.append(
            _resolved(
                _Entity(f"line-{index}"),
                _ResolvedLine(
                    Point(-direction_x, -direction_y),
                    Point(direction_x, direction_y),
                    _Extent.INFINITE,
                ),
            )
        )

    intersection_calls = 0
    original_intersections = object_snap_service._intersections

    def counted_intersections(first, second):
        nonlocal intersection_calls
        intersection_calls += 1
        return original_intersections(first, second)

    monkeypatch.setattr(
        object_snap_service,
        "_intersections",
        counted_intersections,
    )
    engine = ObjectSnapEngine(
        ObjectSnapSettings(
            aperture_px=0.5,
            enabled_kinds=frozenset({SnapKind.INTERSECTION}),
        )
    )

    candidates = engine.candidates(Point(0.0, 0.0), constructions=constructions)

    assert len(candidates) == 1
    assert candidates[0].point_px == Point(0.0, 0.0)
    assert candidates[0].kind is SnapKind.INTERSECTION
    assert intersection_calls <= 64 * 63 // 2


def test_parallel_primitives_cannot_starve_a_transverse_nearby_intersection() -> None:
    constructions = [
        _resolved(
            _Entity(f"coincident-{index}"),
            _ResolvedLine(
                Point(-10.0, 0.0),
                Point(10.0, 0.0),
                _Extent.INFINITE,
            ),
        )
        for index in range(64)
    ]
    constructions.extend(
        (
            _resolved(
                _Entity("vertical"),
                _ResolvedLine(
                    Point(0.1, -10.0),
                    Point(0.1, 10.0),
                    _Extent.INFINITE,
                ),
            ),
            _resolved(
                _Entity("offset-horizontal"),
                _ResolvedLine(
                    Point(-10.0, 0.1),
                    Point(10.0, 0.1),
                    _Extent.INFINITE,
                ),
            ),
        )
    )
    engine = ObjectSnapEngine(
        ObjectSnapSettings(
            aperture_px=0.5,
            enabled_kinds=frozenset({SnapKind.INTERSECTION}),
        )
    )

    candidates = engine.candidates(Point(0.0, 0.0), constructions=constructions)

    assert any(
        candidate.point_px.x == pytest.approx(0.1)
        and candidate.point_px.y == pytest.approx(0.1)
        for candidate in candidates
    )


def test_short_segments_cannot_starve_same_angle_infinite_construction_line() -> None:
    constructions = [
        _resolved(
            _Entity(f"short-segment-{index}"),
            _ResolvedLine(
                Point(-0.05, (index - 31.5) * 0.002),
                Point(0.05, (index - 31.5) * 0.002),
                _Extent.SEGMENT,
            ),
        )
        for index in range(64)
    ]
    constructions.extend(
        (
            _resolved(
                _Entity("infinite-vertical"),
                _ResolvedLine(
                    Point(0.1, -10.0),
                    Point(0.1, 10.0),
                    _Extent.INFINITE,
                ),
            ),
            _resolved(
                _Entity("infinite-horizontal"),
                _ResolvedLine(
                    Point(-10.0, 0.1),
                    Point(10.0, 0.1),
                    _Extent.INFINITE,
                ),
            ),
        )
    )
    engine = ObjectSnapEngine(
        ObjectSnapSettings(
            aperture_px=0.5,
            enabled_kinds=frozenset({SnapKind.INTERSECTION}),
        )
    )

    candidates = engine.candidates(Point(0.0, 0.0), constructions=constructions)

    assert any(
        candidate.point_px.x == pytest.approx(0.1)
        and candidate.point_px.y == pytest.approx(0.1)
        for candidate in candidates
    )


def test_distinct_parallel_lines_cannot_starve_nearby_fine_angle_pair() -> None:
    constructions = [
        _resolved(
            _Entity(f"near-horizontal-{index}"),
            _ResolvedLine(
                Point(-10.0, (index + 1) * 0.0009),
                Point(10.0, (index + 1) * 0.0009),
                _Extent.INFINITE,
            ),
        )
        for index in range(64)
    ]
    target = Point(0.1, 0.1)
    for angle_degrees in (1.0, 2.0):
        angle = math.radians(angle_degrees)
        direction = Point(math.cos(angle) * 10.0, math.sin(angle) * 10.0)
        constructions.append(
            _resolved(
                _Entity(f"fine-angle-{angle_degrees}"),
                _ResolvedLine(
                    Point(target.x - direction.x, target.y - direction.y),
                    Point(target.x + direction.x, target.y + direction.y),
                    _Extent.INFINITE,
                ),
            )
        )
    engine = ObjectSnapEngine(
        ObjectSnapSettings(
            aperture_px=0.2,
            enabled_kinds=frozenset({SnapKind.INTERSECTION}),
        )
    )

    candidates = engine.candidates(Point(0.0, 0.0), constructions=constructions)

    assert any(
        candidate.point_px.x == pytest.approx(target.x)
        and candidate.point_px.y == pytest.approx(target.y)
        for candidate in candidates
    )


def test_aggregated_intersection_identity_ignores_bounded_contributor_changes() -> None:
    first = SnapCandidate(
        point_px=Point(5.0, 5.0),
        kind=SnapKind.INTERSECTION,
        source_type="derived",
        source_id="50000000:50000000",
        feature_key="intersection",
        screen_distance_px=0.0,
        semantic_priority=40,
        related_source_ids=("construction:a", "construction:b"),
    )
    second = SnapCandidate(
        point_px=first.point_px,
        kind=first.kind,
        source_type=first.source_type,
        source_id=first.source_id,
        feature_key=first.feature_key,
        screen_distance_px=first.screen_distance_px,
        semantic_priority=first.semantic_priority,
        related_source_ids=("construction:c", "construction:d"),
    )

    assert first.identity == second.identity
