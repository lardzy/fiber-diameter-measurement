from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any

from fdm.geometry import Point
from fdm.models import Measurement


class SnapKind(str, Enum):
    """Semantic object-snap targets.

    ``NEAREST``, ``PERPENDICULAR`` and ``TANGENT`` are intentionally present
    even though only ``NEAREST`` is calculated by the general-purpose engine
    today.  The latter two are contextual construction aids and therefore must
    be enabled explicitly by the command that knows their source geometry.
    """

    POINT = "point"
    ENDPOINT = "endpoint"
    MIDPOINT = "midpoint"
    CENTER = "center"
    QUADRANT = "quadrant"
    INTERSECTION = "intersection"
    NEAREST = "nearest"
    PERPENDICULAR = "perpendicular"
    TANGENT = "tangent"


DEFAULT_SNAP_KINDS = frozenset(
    {
        SnapKind.POINT,
        SnapKind.ENDPOINT,
        SnapKind.MIDPOINT,
        SnapKind.CENTER,
        SnapKind.QUADRANT,
        SnapKind.INTERSECTION,
    }
)


@dataclass(frozen=True, slots=True)
class ObjectSnapSettings:
    """User-level object-snap preferences expressed in logical screen pixels."""

    enabled: bool = True
    enabled_kinds: frozenset[SnapKind] = field(
        default_factory=lambda: DEFAULT_SNAP_KINDS
    )
    aperture_px: float = 10.0
    hysteresis_px: float = 3.0
    include_measurements: bool = True

    def __post_init__(self) -> None:
        normalized_kinds: set[SnapKind] = set()
        for kind in self.enabled_kinds:
            try:
                normalized_kinds.add(kind if isinstance(kind, SnapKind) else SnapKind(str(kind)))
            except ValueError:
                continue
        object.__setattr__(self, "enabled_kinds", frozenset(normalized_kinds))
        object.__setattr__(self, "aperture_px", _finite_nonnegative(self.aperture_px, 10.0))
        object.__setattr__(self, "hysteresis_px", _finite_nonnegative(self.hysteresis_px, 3.0))

    def allows(self, kind: SnapKind) -> bool:
        return self.enabled and kind in self.enabled_kinds

    def to_dict(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "enabled_kinds": sorted(kind.value for kind in self.enabled_kinds),
            "aperture_px": self.aperture_px,
            "hysteresis_px": self.hysteresis_px,
            "include_measurements": self.include_measurements,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "ObjectSnapSettings":
        raw_kinds = payload.get("enabled_kinds", DEFAULT_SNAP_KINDS)
        kinds: set[SnapKind] = set()
        if isinstance(raw_kinds, (list, tuple, set, frozenset)):
            for value in raw_kinds:
                try:
                    kinds.add(value if isinstance(value, SnapKind) else SnapKind(str(value)))
                except ValueError:
                    continue
        else:
            kinds.update(DEFAULT_SNAP_KINDS)
        return cls(
            enabled=bool(payload.get("enabled", True)),
            enabled_kinds=frozenset(kinds),
            aperture_px=_coerce_float(payload.get("aperture_px"), 10.0),
            hysteresis_px=_coerce_float(payload.get("hysteresis_px"), 3.0),
            include_measurements=bool(payload.get("include_measurements", True)),
        )


_SEMANTIC_PRIORITY: dict[SnapKind, int] = {
    SnapKind.POINT: 0,
    SnapKind.ENDPOINT: 10,
    SnapKind.MIDPOINT: 20,
    SnapKind.CENTER: 30,
    SnapKind.INTERSECTION: 40,
    SnapKind.QUADRANT: 50,
    SnapKind.PERPENDICULAR: 60,
    SnapKind.TANGENT: 70,
    SnapKind.NEAREST: 100,
}

_SNAP_LABELS: dict[SnapKind, str] = {
    SnapKind.POINT: "点",
    SnapKind.ENDPOINT: "端点",
    SnapKind.MIDPOINT: "中点",
    SnapKind.CENTER: "圆心",
    SnapKind.QUADRANT: "象限点",
    SnapKind.INTERSECTION: "交点",
    SnapKind.NEAREST: "最近点",
    SnapKind.PERPENDICULAR: "垂足",
    SnapKind.TANGENT: "切点",
}

# Intersections are evaluated on demand around the cursor.  A dense family of
# construction lines can all pass through that aperture, so the spatial index
# alone cannot prevent an O(N²) pair explosion.  Keep a bounded, distance-first
# working set and coalesce coincident results below visual precision.
_MAX_INTERSECTION_PRIMITIVES = 64
_INTERSECTION_DISTANCE_HEAD = 16
_INTERSECTION_POINT_KEY_SCALE = 10_000_000.0
_INTERSECTION_DIRECTION_KEY_SCALE = 100_000.0


@dataclass(frozen=True, slots=True)
class SnapCandidate:
    """One screen-ranked snap target with a stable source identity."""

    point_px: Point
    kind: SnapKind
    source_type: str
    source_id: str
    feature_key: str
    screen_distance_px: float
    semantic_priority: int
    label: str = ""
    related_source_ids: tuple[str, ...] = ()

    @property
    def identity(self) -> tuple[str, str, str, str, tuple[str, ...]]:
        # The contributing primitive set is bounded and distance-ranked, so it
        # may change slightly while the pointer moves even though the derived
        # intersection coordinate is unchanged.  Coordinate-keyed derived
        # intersections therefore keep contributor IDs as metadata rather than
        # making them part of hysteresis identity.
        identity_sources = (
            ()
            if self.kind is SnapKind.INTERSECTION and self.source_type == "derived"
            else self.related_source_ids
        )
        return (
            self.kind.value,
            self.source_type,
            self.source_id,
            self.feature_key,
            identity_sources,
        )


@dataclass(frozen=True, slots=True)
class _PointPrimitive:
    point: Point
    kind: SnapKind
    source_type: str
    source_id: str
    feature_key: str


@dataclass(frozen=True, slots=True)
class _LinePrimitive:
    start: Point
    end: Point
    domain: str
    source_type: str
    source_id: str
    feature_key: str
    emit_endpoints: bool = True
    emit_midpoint: bool = True


@dataclass(frozen=True, slots=True)
class _CirclePrimitive:
    center: Point
    radius: float
    source_type: str
    source_id: str
    feature_key: str


_Primitive = _PointPrimitive | _LinePrimitive | _CirclePrimitive
_ScreenTransform = Callable[[Point], Point | tuple[float, float] | Sequence[float] | object]


class ObjectSnapEngine:
    """Collect and rank transient geometric snap candidates.

    Construction input is intentionally read through a small structural
    adapter rather than importing a concrete resolver.  A caller may pass a
    resolved object directly, ``(ConstructionEntity, ResolvedConstruction)``
    pairs, or a mapping of IDs to resolved objects.  This keeps the service
    independent from persistence and dependency resolution while still
    honoring entity visibility and snap-enabled flags.
    """

    def __init__(self, settings: ObjectSnapSettings | None = None) -> None:
        self.settings = settings or ObjectSnapSettings()
        self._active_candidate: SnapCandidate | None = None

    @property
    def active_candidate(self) -> SnapCandidate | None:
        return self._active_candidate

    def update_settings(self, settings: ObjectSnapSettings) -> None:
        self.settings = settings
        if not settings.enabled:
            self.clear_hysteresis()

    def clear_hysteresis(self) -> None:
        self._active_candidate = None

    def candidates(
        self,
        cursor_image_px: Point,
        *,
        image_to_screen: _ScreenTransform | None = None,
        constructions: Iterable[object] | Mapping[object, object] = (),
        measurements: Iterable[Measurement] = (),
        contextual_candidates: Iterable[SnapCandidate] = (),
    ) -> tuple[SnapCandidate, ...]:
        """Return all candidates inside the configured aperture in rank order."""

        if not self.settings.enabled or self.settings.aperture_px <= 0.0:
            return ()
        return self._collect_candidates(
            cursor_image_px,
            image_to_screen=image_to_screen,
            constructions=constructions,
            measurements=measurements,
            contextual_candidates=contextual_candidates,
            maximum_distance_px=self.settings.aperture_px,
        )

    def query(
        self,
        cursor_image_px: Point,
        *,
        image_to_screen: _ScreenTransform | None = None,
        constructions: Iterable[object] | Mapping[object, object] = (),
        measurements: Iterable[Measurement] = (),
        contextual_candidates: Iterable[SnapCandidate] = (),
        previous: SnapCandidate | None = None,
    ) -> SnapCandidate | None:
        """Return the preferred candidate, retaining the previous one briefly.

        Hysteresis is measured in logical screen pixels.  A previous target is
        retained while it remains in the enlarged aperture and is no more than
        ``hysteresis_px`` worse than the newly ranked target.  This suppresses
        cursor flicker without making a materially closer target difficult to
        select.
        """

        if not self.settings.enabled or self.settings.aperture_px <= 0.0:
            self.clear_hysteresis()
            return None
        sticky = previous if previous is not None else self._active_candidate
        maximum_distance = self.settings.aperture_px
        if sticky is not None:
            maximum_distance += self.settings.hysteresis_px
        expanded = self._collect_candidates(
            cursor_image_px,
            image_to_screen=image_to_screen,
            constructions=constructions,
            measurements=measurements,
            contextual_candidates=contextual_candidates,
            maximum_distance_px=maximum_distance,
        )
        normal = tuple(
            candidate
            for candidate in expanded
            if candidate.screen_distance_px <= self.settings.aperture_px + 1e-9
        )
        preferred = normal[0] if normal else None
        if sticky is not None:
            retained = next(
                (candidate for candidate in expanded if candidate.identity == sticky.identity),
                None,
            )
            if retained is not None and (
                preferred is None
                or retained.screen_distance_px
                <= preferred.screen_distance_px + self.settings.hysteresis_px
            ):
                preferred = retained
        self._active_candidate = preferred
        return preferred

    def _collect_candidates(
        self,
        cursor_image_px: Point,
        *,
        image_to_screen: _ScreenTransform | None,
        constructions: Iterable[object] | Mapping[object, object],
        measurements: Iterable[Measurement],
        contextual_candidates: Iterable[SnapCandidate],
        maximum_distance_px: float,
    ) -> tuple[SnapCandidate, ...]:
        transform = image_to_screen or (lambda point: point)
        cursor_screen = _screen_xy(transform(cursor_image_px))
        primitives = list(_construction_primitives(constructions))
        if self.settings.include_measurements:
            primitives.extend(_measurement_primitives(measurements))

        candidates: list[SnapCandidate] = []
        for primitive in primitives:
            candidates.extend(
                self._static_candidates(
                    primitive,
                    cursor_screen=cursor_screen,
                    image_to_screen=transform,
                    maximum_distance_px=maximum_distance_px,
                )
            )
        if self.settings.allows(SnapKind.INTERSECTION):
            candidates.extend(
                self._intersection_candidates(
                    primitives,
                    cursor_screen=cursor_screen,
                    image_to_screen=transform,
                    maximum_distance_px=maximum_distance_px,
                )
            )
        # Perpendicular feet and tangent points are command-scoped aids.  The
        # construction command supplies them explicitly, so they deliberately
        # bypass the user's persistent kind filter while still sharing the
        # normal aperture, distance ordering and hysteresis rules.
        candidates.extend(
            candidate
            for candidate in contextual_candidates
            if candidate.kind in {SnapKind.PERPENDICULAR, SnapKind.TANGENT}
            and candidate.screen_distance_px <= maximum_distance_px + 1e-9
        )
        unique: dict[tuple[str, str, str, str, tuple[str, ...]], SnapCandidate] = {}
        for candidate in candidates:
            existing = unique.get(candidate.identity)
            if existing is None or candidate.screen_distance_px < existing.screen_distance_px:
                unique[candidate.identity] = candidate
        return tuple(sorted(unique.values(), key=_candidate_sort_key))

    def _static_candidates(
        self,
        primitive: _Primitive,
        *,
        cursor_screen: tuple[float, float],
        image_to_screen: _ScreenTransform,
        maximum_distance_px: float,
    ) -> list[SnapCandidate]:
        result: list[SnapCandidate] = []
        if isinstance(primitive, _PointPrimitive):
            if self.settings.allows(primitive.kind):
                candidate = _candidate_for_point(
                    primitive.point,
                    kind=primitive.kind,
                    source_type=primitive.source_type,
                    source_id=primitive.source_id,
                    feature_key=primitive.feature_key,
                    cursor_screen=cursor_screen,
                    image_to_screen=image_to_screen,
                )
                if candidate.screen_distance_px <= maximum_distance_px + 1e-9:
                    result.append(candidate)
            return result

        if isinstance(primitive, _LinePrimitive):
            if primitive.emit_endpoints and self.settings.allows(SnapKind.ENDPOINT):
                endpoint_values: list[tuple[str, Point]] = []
                if primitive.domain == "segment":
                    endpoint_values = [("start", primitive.start), ("end", primitive.end)]
                elif primitive.domain == "ray":
                    endpoint_values = [("origin", primitive.start)]
                for suffix, point in endpoint_values:
                    candidate = _candidate_for_point(
                        point,
                        kind=SnapKind.ENDPOINT,
                        source_type=primitive.source_type,
                        source_id=primitive.source_id,
                        feature_key=f"{primitive.feature_key}:{suffix}",
                        cursor_screen=cursor_screen,
                        image_to_screen=image_to_screen,
                    )
                    if candidate.screen_distance_px <= maximum_distance_px + 1e-9:
                        result.append(candidate)
            if (
                primitive.domain == "segment"
                and primitive.emit_midpoint
                and self.settings.allows(SnapKind.MIDPOINT)
            ):
                candidate = _candidate_for_point(
                    Point(
                        (primitive.start.x + primitive.end.x) / 2.0,
                        (primitive.start.y + primitive.end.y) / 2.0,
                    ),
                    kind=SnapKind.MIDPOINT,
                    source_type=primitive.source_type,
                    source_id=primitive.source_id,
                    feature_key=f"{primitive.feature_key}:midpoint",
                    cursor_screen=cursor_screen,
                    image_to_screen=image_to_screen,
                )
                if candidate.screen_distance_px <= maximum_distance_px + 1e-9:
                    result.append(candidate)
            if self.settings.allows(SnapKind.NEAREST):
                # Nearest must be calculated in image space when a transform is
                # present; use a screen-space projection for affine viewport
                # mapping and convert the ratio back onto the image primitive.
                nearest = _nearest_on_line_screen(
                    cursor_screen,
                    primitive,
                    image_to_screen=image_to_screen,
                )
                candidate = _candidate_for_point(
                    nearest,
                    kind=SnapKind.NEAREST,
                    source_type=primitive.source_type,
                    source_id=primitive.source_id,
                    feature_key=f"{primitive.feature_key}:nearest",
                    cursor_screen=cursor_screen,
                    image_to_screen=image_to_screen,
                )
                if candidate.screen_distance_px <= maximum_distance_px + 1e-9:
                    result.append(candidate)
            return result

        if self.settings.allows(SnapKind.CENTER):
            candidate = _candidate_for_point(
                primitive.center,
                kind=SnapKind.CENTER,
                source_type=primitive.source_type,
                source_id=primitive.source_id,
                feature_key=f"{primitive.feature_key}:center",
                cursor_screen=cursor_screen,
                image_to_screen=image_to_screen,
            )
            if candidate.screen_distance_px <= maximum_distance_px + 1e-9:
                result.append(candidate)
        if self.settings.allows(SnapKind.QUADRANT):
            quadrants = (
                Point(primitive.center.x + primitive.radius, primitive.center.y),
                Point(primitive.center.x, primitive.center.y + primitive.radius),
                Point(primitive.center.x - primitive.radius, primitive.center.y),
                Point(primitive.center.x, primitive.center.y - primitive.radius),
            )
            for index, point in enumerate(quadrants):
                candidate = _candidate_for_point(
                    point,
                    kind=SnapKind.QUADRANT,
                    source_type=primitive.source_type,
                    source_id=primitive.source_id,
                    feature_key=f"{primitive.feature_key}:quadrant:{index}",
                    cursor_screen=cursor_screen,
                    image_to_screen=image_to_screen,
                )
                if candidate.screen_distance_px <= maximum_distance_px + 1e-9:
                    result.append(candidate)
        if self.settings.allows(SnapKind.NEAREST):
            nearest = _nearest_on_circle_screen(
                cursor_screen,
                primitive,
                image_to_screen=image_to_screen,
            )
            candidate = _candidate_for_point(
                nearest,
                kind=SnapKind.NEAREST,
                source_type=primitive.source_type,
                source_id=primitive.source_id,
                feature_key=f"{primitive.feature_key}:nearest",
                cursor_screen=cursor_screen,
                image_to_screen=image_to_screen,
            )
            if candidate.screen_distance_px <= maximum_distance_px + 1e-9:
                result.append(candidate)
        return result

    def _intersection_candidates(
        self,
        primitives: list[_Primitive],
        *,
        cursor_screen: tuple[float, float],
        image_to_screen: _ScreenTransform,
        maximum_distance_px: float,
    ) -> list[SnapCandidate]:
        geometry = [
            primitive
            for primitive in primitives
            if isinstance(primitive, (_LinePrimitive, _CirclePrimitive))
        ]
        neighborhood_px = max(32.0, maximum_distance_px * 4.0)
        ranked_nearby: list[tuple[float, int, _Primitive]] = []
        for source_order, primitive in enumerate(geometry):
            screen_distance = _primitive_screen_distance(
                primitive,
                cursor_screen=cursor_screen,
                image_to_screen=image_to_screen,
            )
            if screen_distance <= neighborhood_px:
                ranked_nearby.append((screen_distance, source_order, primitive))
        ranked_nearby.sort(key=lambda item: (item[0], item[1]))
        nearby = _bounded_intersection_primitives(
            ranked_nearby,
            cursor_screen=cursor_screen,
            image_to_screen=image_to_screen,
        )

        # Several source pairs commonly describe the same visual intersection
        # (grids, radial guides, coincident measurement edges).  Object snap
        # needs one coordinate candidate; source identities remain available as
        # metadata for commands that need to disambiguate their parents.
        aggregated: dict[
            tuple[int, int],
            tuple[Point, set[str]],
        ] = {}
        for left_index, left in enumerate(nearby):
            for right in nearby[left_index + 1 :]:
                if (
                    left.source_type == right.source_type
                    and left.source_id == right.source_id
                    and left.feature_key == right.feature_key
                ):
                    continue
                points = _intersections(left, right)
                if (
                    left.source_type == right.source_type
                    and left.source_id == right.source_id
                    and isinstance(left, _LinePrimitive)
                    and isinstance(right, _LinePrimitive)
                ):
                    # Adjacent segments of one polyline/polygon meet at an
                    # ordinary endpoint, not a self-intersection.  Non-adjacent
                    # segments from that same analytical object must still be
                    # allowed to generate an INTERSECTION candidate.
                    points = tuple(
                        point
                        for point in points
                        if not _is_shared_line_endpoint(point, left, right)
                    )
                related = tuple(
                    sorted(
                        {
                            f"{left.source_type}:{left.source_id}",
                            f"{right.source_type}:{right.source_id}",
                        }
                    )
                )
                for point in points:
                    candidate = _candidate_for_point(
                        point,
                        kind=SnapKind.INTERSECTION,
                        source_type="derived",
                        source_id="intersection",
                        feature_key="intersection",
                        cursor_screen=cursor_screen,
                        image_to_screen=image_to_screen,
                        related_source_ids=related,
                    )
                    if candidate.screen_distance_px <= maximum_distance_px + 1e-9:
                        point_key = (
                            int(round(point.x * _INTERSECTION_POINT_KEY_SCALE)),
                            int(round(point.y * _INTERSECTION_POINT_KEY_SCALE)),
                        )
                        existing = aggregated.get(point_key)
                        if existing is None:
                            aggregated[point_key] = (point, set(related))
                        else:
                            existing[1].update(related)

        result: list[SnapCandidate] = []
        for point_key, (point, related) in aggregated.items():
            stable_key = f"{point_key[0]}:{point_key[1]}"
            result.append(
                _candidate_for_point(
                    point,
                    kind=SnapKind.INTERSECTION,
                    source_type="derived",
                    source_id=stable_key,
                    feature_key="intersection",
                    cursor_screen=cursor_screen,
                    image_to_screen=image_to_screen,
                    related_source_ids=tuple(sorted(related)),
                )
            )
        return result


def _bounded_intersection_primitives(
    ranked: Sequence[tuple[float, int, _Primitive]],
    *,
    cursor_screen: tuple[float, float],
    image_to_screen: _ScreenTransform,
) -> list[_Primitive]:
    """Choose a bounded, distance-first but direction-diverse working set.

    A simple first-N cutoff can be monopolized by coincident or parallel guide
    lines and omit the one transverse primitive that creates the useful nearby
    intersection.  Exact geometric duplicates are collapsed first; a small
    closest-distance head is then combined with round-robin line-angle/circle-
    scale buckets.  The pair count remains strictly bounded while a single
    family cannot starve every other geometry class.
    """

    unique_ranked: list[tuple[float, int, _Primitive]] = []
    seen_geometry: set[tuple[object, ...]] = set()
    for item in ranked:
        geometry_key = _intersection_primitive_geometry_key(item[2])
        if geometry_key in seen_geometry:
            continue
        seen_geometry.add(geometry_key)
        unique_ranked.append(item)
    if len(unique_ranked) <= _MAX_INTERSECTION_PRIMITIVES:
        return [item[2] for item in unique_ranked]

    selected: list[_Primitive] = []
    selected_orders: set[int] = set()
    for _distance, source_order, primitive in unique_ranked[
        :_INTERSECTION_DISTANCE_HEAD
    ]:
        selected.append(primitive)
        selected_orders.add(source_order)

    buckets: dict[tuple[object, ...], list[tuple[float, int, _Primitive]]] = {}
    bucket_order: list[tuple[object, ...]] = []
    for item in unique_ranked:
        if item[1] in selected_orders:
            continue
        bucket_key = _intersection_primitive_bucket_key(
            item[2],
            cursor_screen=cursor_screen,
            image_to_screen=image_to_screen,
        )
        if bucket_key not in buckets:
            buckets[bucket_key] = []
            bucket_order.append(bucket_key)
        buckets[bucket_key].append(item)

    bucket_offsets = {bucket_key: 0 for bucket_key in bucket_order}
    while len(selected) < _MAX_INTERSECTION_PRIMITIVES:
        added = False
        for bucket_key in bucket_order:
            offset = bucket_offsets[bucket_key]
            bucket = buckets[bucket_key]
            if offset >= len(bucket):
                continue
            _distance, _source_order, primitive = bucket[offset]
            bucket_offsets[bucket_key] = offset + 1
            selected.append(primitive)
            added = True
            if len(selected) >= _MAX_INTERSECTION_PRIMITIVES:
                break
        if not added:
            break
    return selected


def _intersection_primitive_geometry_key(
    primitive: _Primitive,
) -> tuple[object, ...]:
    def point_key(point: Point) -> tuple[int, int]:
        return (
            int(round(point.x * _INTERSECTION_POINT_KEY_SCALE)),
            int(round(point.y * _INTERSECTION_POINT_KEY_SCALE)),
        )

    if isinstance(primitive, _CirclePrimitive):
        return (
            "circle",
            *point_key(primitive.center),
            int(round(primitive.radius * _INTERSECTION_POINT_KEY_SCALE)),
        )
    assert isinstance(primitive, _LinePrimitive)
    start_key = point_key(primitive.start)
    end_key = point_key(primitive.end)
    if primitive.domain == "segment":
        first, second = sorted((start_key, end_key))
        return ("line", "segment", *first, *second)

    direction_x = primitive.end.x - primitive.start.x
    direction_y = primitive.end.y - primitive.start.y
    length = math.hypot(direction_x, direction_y)
    if length <= 1e-12:
        return ("line", primitive.domain, *start_key, *end_key)
    direction_x /= length
    direction_y /= length
    if primitive.domain == "ray":
        return (
            "line",
            "ray",
            *start_key,
            int(round(direction_x * _INTERSECTION_POINT_KEY_SCALE)),
            int(round(direction_y * _INTERSECTION_POINT_KEY_SCALE)),
        )
    if direction_x < 0.0 or (abs(direction_x) <= 1e-12 and direction_y < 0.0):
        direction_x = -direction_x
        direction_y = -direction_y
    signed_offset = -direction_y * primitive.start.x + direction_x * primitive.start.y
    return (
        "line",
        "infinite",
        int(round(direction_x * _INTERSECTION_POINT_KEY_SCALE)),
        int(round(direction_y * _INTERSECTION_POINT_KEY_SCALE)),
        int(round(signed_offset * _INTERSECTION_POINT_KEY_SCALE)),
    )


def _intersection_primitive_bucket_key(
    primitive: _Primitive,
    *,
    cursor_screen: tuple[float, float],
    image_to_screen: _ScreenTransform,
) -> tuple[object, ...]:
    if isinstance(primitive, _CirclePrimitive):
        center_screen = _screen_xy(image_to_screen(primitive.center))
        radius_screen_point = _screen_xy(
            image_to_screen(
                Point(primitive.center.x + primitive.radius, primitive.center.y)
            )
        )
        radius_screen = math.hypot(
            radius_screen_point[0] - center_screen[0],
            radius_screen_point[1] - center_screen[1],
        )
        scale_bucket = int(math.floor(math.log2(max(radius_screen, 1e-6))))
        center_direction_x = center_screen[0] - cursor_screen[0]
        center_direction_y = center_screen[1] - cursor_screen[1]
        center_distance = math.hypot(center_direction_x, center_direction_y)
        if center_distance > 1e-9:
            center_direction_x /= center_distance
            center_direction_y /= center_distance
        return (
            "circle",
            scale_bucket,
            int(round(center_direction_x * _INTERSECTION_DIRECTION_KEY_SCALE)),
            int(round(center_direction_y * _INTERSECTION_DIRECTION_KEY_SCALE)),
        )

    assert isinstance(primitive, _LinePrimitive)
    start_screen = _screen_xy(image_to_screen(primitive.start))
    end_screen = _screen_xy(image_to_screen(primitive.end))
    direction_x = end_screen[0] - start_screen[0]
    direction_y = end_screen[1] - start_screen[1]
    screen_length = math.hypot(direction_x, direction_y)
    if screen_length > 1e-9:
        direction_x /= screen_length
        direction_y /= screen_length
    if direction_x < 0.0 or (abs(direction_x) <= 1e-12 and direction_y < 0.0):
        direction_x = -direction_x
        direction_y = -direction_y
    length_bucket = (
        int(math.floor(math.log2(max(screen_length, 1e-6))))
        if primitive.domain == "segment"
        else 0
    )
    # Definition domains are material to pair usefulness: dozens of short
    # segments can be near the cursor yet end before a transverse line, while
    # a same-angle infinite construction line still creates the intended
    # intersection.  Keep those families in separate round-robin buckets.
    return (
        "line",
        primitive.domain,
        int(round(direction_x * _INTERSECTION_DIRECTION_KEY_SCALE)),
        int(round(direction_y * _INTERSECTION_DIRECTION_KEY_SCALE)),
        length_bucket,
    )


def _is_shared_line_endpoint(
    point: Point,
    first: _LinePrimitive,
    second: _LinePrimitive,
) -> bool:
    tolerance = 1e-7
    return any(
        math.hypot(candidate.x - point.x, candidate.y - point.y) <= tolerance
        for candidate in (first.start, first.end)
    ) and any(
        math.hypot(candidate.x - point.x, candidate.y - point.y) <= tolerance
        for candidate in (second.start, second.end)
    )


def _candidate_sort_key(candidate: SnapCandidate) -> tuple[float, int, str, str, str]:
    # Physical cursor distance intentionally dominates semantic preference.
    return (
        candidate.screen_distance_px,
        candidate.semantic_priority,
        candidate.source_type,
        candidate.source_id,
        candidate.feature_key,
    )


def _candidate_for_point(
    point: Point,
    *,
    kind: SnapKind,
    source_type: str,
    source_id: str,
    feature_key: str,
    cursor_screen: tuple[float, float],
    image_to_screen: _ScreenTransform,
    related_source_ids: tuple[str, ...] = (),
) -> SnapCandidate:
    target_screen = _screen_xy(image_to_screen(point))
    return SnapCandidate(
        point_px=Point(float(point.x), float(point.y)),
        kind=kind,
        source_type=source_type,
        source_id=source_id,
        feature_key=feature_key,
        screen_distance_px=math.hypot(
            target_screen[0] - cursor_screen[0],
            target_screen[1] - cursor_screen[1],
        ),
        semantic_priority=_SEMANTIC_PRIORITY[kind],
        label=_SNAP_LABELS[kind],
        related_source_ids=related_source_ids,
    )


def contextual_snap_candidate(
    point: Point,
    *,
    kind: SnapKind,
    source_id: str,
    feature_key: str,
    cursor_image_px: Point,
    image_to_screen: _ScreenTransform | None = None,
    source_type: str = "derived",
    related_source_ids: tuple[str, ...] = (),
) -> SnapCandidate:
    """Build one command-scoped perpendicular or tangent snap candidate."""

    kind = SnapKind(kind)
    if kind not in {SnapKind.PERPENDICULAR, SnapKind.TANGENT}:
        raise ValueError("上下文捕捉仅支持垂足和切点")
    transform = image_to_screen or (lambda value: value)
    return _candidate_for_point(
        point,
        kind=kind,
        source_type=str(source_type),
        source_id=str(source_id),
        feature_key=str(feature_key),
        cursor_screen=_screen_xy(transform(cursor_image_px)),
        image_to_screen=transform,
        related_source_ids=tuple(str(value) for value in related_source_ids),
    )


def _measurement_primitives(measurements: Iterable[Measurement]) -> Iterable[_Primitive]:
    for measurement in measurements:
        if not isinstance(measurement, Measurement):
            # OverlayAnnotation and other decorative records deliberately do
            # not enter the analytical snap graph.
            continue
        mode = str(measurement.mode or "").strip().lower()
        if mode.startswith("freehand") or mode.startswith("magic"):
            continue
        source_id = str(measurement.id)
        kind = str(measurement.measurement_kind or "").strip().lower()
        if kind == "line" and measurement.line_px is not None:
            try:
                line = measurement.effective_line()
            except ValueError:
                continue
            yield _LinePrimitive(
                start=_copy_point(line.start),
                end=_copy_point(line.end),
                domain="segment",
                source_type="measurement",
                source_id=source_id,
                feature_key="line",
            )
            continue
        if kind == "polyline" and len(measurement.polyline_px) >= 2:
            points = measurement.polyline_px
            for index, point in enumerate(points):
                yield _PointPrimitive(
                    point=_copy_point(point),
                    kind=SnapKind.ENDPOINT,
                    source_type="measurement",
                    source_id=source_id,
                    feature_key=f"vertex:{index}",
                )
            for index in range(len(points) - 1):
                yield _LinePrimitive(
                    start=_copy_point(points[index]),
                    end=_copy_point(points[index + 1]),
                    domain="segment",
                    source_type="measurement",
                    source_id=source_id,
                    feature_key=f"segment:{index}",
                    emit_endpoints=False,
                )
            continue
        if kind == "count" and measurement.point_px is not None:
            yield _PointPrimitive(
                point=_copy_point(measurement.point_px),
                kind=SnapKind.POINT,
                source_type="measurement",
                source_id=source_id,
                feature_key="point",
            )
            continue
        if kind != "area" or mode not in {"polygon", "polygon_area"}:
            continue
        points = measurement.polygon_px
        if len(points) < 3:
            continue
        for index, point in enumerate(points):
            yield _PointPrimitive(
                point=_copy_point(point),
                kind=SnapKind.ENDPOINT,
                source_type="measurement",
                source_id=source_id,
                feature_key=f"vertex:{index}",
            )
        for index, point in enumerate(points):
            yield _LinePrimitive(
                start=_copy_point(point),
                end=_copy_point(points[(index + 1) % len(points)]),
                domain="segment",
                source_type="measurement",
                source_id=source_id,
                feature_key=f"edge:{index}",
                emit_endpoints=False,
            )


def _construction_primitives(
    constructions: Iterable[object] | Mapping[object, object],
) -> Iterable[_Primitive]:
    values: Iterable[object]
    if isinstance(constructions, Mapping):
        values = constructions.values()
    else:
        values = constructions
    for entry in values:
        entity, resolved = _split_construction_entry(entry)
        if not _construction_is_available(entity, resolved):
            continue
        source_id = str(
            _read(entity, "id", None)
            or _read(resolved, "entity_id", None)
            or _read(resolved, "source_id", None)
            or _read(resolved, "id", None)
            or "construction"
        )
        # A ConstructionSpatialItem represents one already-nearby primitive.
        # Preserve that narrow result instead of expanding its owning array (or
        # other compound geometry) again inside the snap engine.
        indexed_feature_key = _read(entry, "feature_key", None)
        indexed_geometry = _read(entry, "geometry", None)
        if indexed_feature_key is not None and indexed_geometry is not None:
            geometry = indexed_geometry
            feature_prefix = str(indexed_feature_key)
        else:
            geometry = _read(resolved, "geometry", resolved)
            feature_prefix = "geometry"
        definition = _read(entity, "definition", None)
        definition_kind = _enum_token(_read(definition, "kind", ""))
        resolved_point_kind = {
            "midpoint": SnapKind.MIDPOINT,
            "intersection": SnapKind.INTERSECTION,
        }.get(definition_kind)
        yield from _geometry_primitives(
            geometry,
            source_id=source_id,
            feature_prefix=feature_prefix,
            resolved_point_kind=resolved_point_kind,
        )


def _split_construction_entry(entry: object) -> tuple[object, object]:
    if isinstance(entry, tuple) and len(entry) == 2:
        return entry[0], entry[1]
    entity = _read(entry, "entity", None)
    resolved = _read(entry, "resolved", None)
    if entity is not None and resolved is not None and not isinstance(resolved, bool):
        return entity, resolved
    return entry, entry


def _construction_is_available(entity: object, resolved: object) -> bool:
    visible = _read(entity, "visible", _read(resolved, "visible", True))
    snap_enabled = _read(
        entity,
        "snap_enabled",
        _read(
            entity,
            "snappable",
            _read(resolved, "snap_enabled", _read(resolved, "snappable", True)),
        ),
    )
    valid = _read(
        resolved,
        "valid",
        _read(resolved, "is_valid", _read(resolved, "resolved", True)),
    )
    if callable(valid):
        try:
            valid = valid()
        except TypeError:
            valid = True
    return bool(visible) and bool(snap_enabled) and valid is not False


def _geometry_primitives(
    geometry: object,
    *,
    source_id: str,
    feature_prefix: str,
    inherited_domain: str | None = None,
    resolved_point_kind: SnapKind | None = None,
) -> Iterable[_Primitive]:
    if geometry is None:
        return
    nested = _read(geometry, "primitives", None)
    if isinstance(nested, Iterable) and not isinstance(nested, (str, bytes, Mapping)):
        for index, item in enumerate(nested):
            yield from _geometry_primitives(
                item,
                source_id=source_id,
                feature_prefix=f"{feature_prefix}:{index}",
                resolved_point_kind=resolved_point_kind,
            )
        return
    lines = _read(geometry, "lines", None)
    if isinstance(lines, Iterable) and not isinstance(lines, (str, bytes, Mapping)):
        for index, line in enumerate(lines):
            multiplier_at = _read(lines, "multiplier_at", None)
            child_key = (
                f"{int(multiplier_at(index)):+d}"
                if callable(multiplier_at)
                else str(index)
            )
            yield from _geometry_primitives(
                line,
                source_id=source_id,
                feature_prefix=f"{feature_prefix}:line:{child_key}",
                inherited_domain=_domain_value(_read(geometry, "domain", "line")),
                resolved_point_kind=resolved_point_kind,
            )
        return

    raw_kind = _read(geometry, "kind", _read(geometry, "geometry_kind", ""))
    kind = _enum_token(raw_kind)
    domain = _domain_value(
        _read(geometry, "domain", _read(geometry, "extent", inherited_domain or kind))
    )

    center = _as_point(
        _read(geometry, "center_px", _read(geometry, "center", None))
    )
    radius = _read(geometry, "radius_px", _read(geometry, "radius", None))
    if center is not None and radius is not None:
        numeric_radius = _coerce_float(radius, -1.0)
        if math.isfinite(numeric_radius) and numeric_radius > 1e-9:
            yield _CirclePrimitive(
                center=center,
                radius=numeric_radius,
                source_type="construction",
                source_id=source_id,
                feature_key=feature_prefix,
            )
        return

    raw_line = _read(geometry, "line", None)
    if raw_line is not None:
        start = _as_point(_read(raw_line, "start", _read(raw_line, "start_px", None)))
        end = _as_point(_read(raw_line, "end", _read(raw_line, "end_px", None)))
    else:
        start = _as_point(
            _read(
                geometry,
                "start_px",
                _read(geometry, "start", _read(geometry, "origin", _read(geometry, "origin_px", None))),
            )
        )
        end = _as_point(_read(geometry, "end_px", _read(geometry, "end", None)))
    if start is not None and end is None:
        direction = _vector_xy(
            _read(geometry, "direction", _read(geometry, "direction_px", None))
        )
        if direction is not None:
            end = Point(start.x + direction[0], start.y + direction[1])
    if start is not None and end is not None and _point_distance(start, end) > 1e-9:
        yield _LinePrimitive(
            start=start,
            end=end,
            domain=domain,
            source_type="construction",
            source_id=source_id,
            feature_key=feature_prefix,
        )
        return

    point = _as_point(
        _read(geometry, "point_px", _read(geometry, "point", None))
    )
    if point is None and kind in {"point", "node", "midpoint", "intersection"}:
        point = _as_point(_read(geometry, "position", None))
    if point is not None:
        yield _PointPrimitive(
            point=point,
            kind=resolved_point_kind or SnapKind.POINT,
            source_type="construction",
            source_id=source_id,
            feature_key=feature_prefix,
        )


def _domain_value(value: object) -> str:
    token = _enum_token(value)
    if token in {"ray", "half_line", "halfline"}:
        return "ray"
    if token in {"line", "infinite", "infinite_line", "xline"}:
        return "line"
    return "segment"


def _intersections(left: _Primitive, right: _Primitive) -> list[Point]:
    if isinstance(left, _LinePrimitive) and isinstance(right, _LinePrimitive):
        return _line_line_intersections(left, right)
    if isinstance(left, _LinePrimitive) and isinstance(right, _CirclePrimitive):
        return _line_circle_intersections(left, right)
    if isinstance(left, _CirclePrimitive) and isinstance(right, _LinePrimitive):
        return _line_circle_intersections(right, left)
    if isinstance(left, _CirclePrimitive) and isinstance(right, _CirclePrimitive):
        return _circle_circle_intersections(left, right)
    return []


def _line_line_intersections(left: _LinePrimitive, right: _LinePrimitive) -> list[Point]:
    p = left.start
    q = right.start
    r = (left.end.x - left.start.x, left.end.y - left.start.y)
    s = (right.end.x - right.start.x, right.end.y - right.start.y)
    denominator = _cross(r, s)
    if abs(denominator) <= 1e-12:
        return []
    delta = (q.x - p.x, q.y - p.y)
    t = _cross(delta, s) / denominator
    u = _cross(delta, r) / denominator
    if not _parameter_in_domain(t, left.domain) or not _parameter_in_domain(u, right.domain):
        return []
    return [Point(p.x + t * r[0], p.y + t * r[1])]


def _line_circle_intersections(line: _LinePrimitive, circle: _CirclePrimitive) -> list[Point]:
    dx = line.end.x - line.start.x
    dy = line.end.y - line.start.y
    fx = line.start.x - circle.center.x
    fy = line.start.y - circle.center.y
    a = dx * dx + dy * dy
    if a <= 1e-18:
        return []
    b = 2.0 * (fx * dx + fy * dy)
    c = fx * fx + fy * fy - circle.radius * circle.radius
    discriminant = b * b - 4.0 * a * c
    tolerance = 1e-10 * max(1.0, b * b, abs(4.0 * a * c))
    if discriminant < -tolerance:
        return []
    roots: list[float]
    if abs(discriminant) <= tolerance:
        roots = [-b / (2.0 * a)]
    else:
        root = math.sqrt(max(0.0, discriminant))
        roots = [(-b - root) / (2.0 * a), (-b + root) / (2.0 * a)]
    return [
        Point(line.start.x + value * dx, line.start.y + value * dy)
        for value in roots
        if _parameter_in_domain(value, line.domain)
    ]


def _circle_circle_intersections(
    left: _CirclePrimitive,
    right: _CirclePrimitive,
) -> list[Point]:
    dx = right.center.x - left.center.x
    dy = right.center.y - left.center.y
    center_distance = math.hypot(dx, dy)
    if center_distance <= 1e-12:
        return []
    if center_distance > left.radius + right.radius + 1e-10:
        return []
    if center_distance < abs(left.radius - right.radius) - 1e-10:
        return []
    along = (
        left.radius * left.radius
        - right.radius * right.radius
        + center_distance * center_distance
    ) / (2.0 * center_distance)
    height_sq = left.radius * left.radius - along * along
    if height_sq < -1e-9:
        return []
    base_x = left.center.x + along * dx / center_distance
    base_y = left.center.y + along * dy / center_distance
    if abs(height_sq) <= 1e-9:
        return [Point(base_x, base_y)]
    height = math.sqrt(max(0.0, height_sq))
    offset_x = -dy * height / center_distance
    offset_y = dx * height / center_distance
    return [
        Point(base_x + offset_x, base_y + offset_y),
        Point(base_x - offset_x, base_y - offset_y),
    ]


def _parameter_in_domain(value: float, domain: str) -> bool:
    tolerance = 1e-9
    if domain == "segment":
        return -tolerance <= value <= 1.0 + tolerance
    if domain == "ray":
        return value >= -tolerance
    return True


def _primitive_screen_distance(
    primitive: _LinePrimitive | _CirclePrimitive,
    *,
    cursor_screen: tuple[float, float],
    image_to_screen: _ScreenTransform,
) -> float:
    if isinstance(primitive, _LinePrimitive):
        nearest = _nearest_on_line_screen(
            cursor_screen,
            primitive,
            image_to_screen=image_to_screen,
        )
    else:
        nearest = _nearest_on_circle_screen(
            cursor_screen,
            primitive,
            image_to_screen=image_to_screen,
        )
    nearest_screen = _screen_xy(image_to_screen(nearest))
    return math.hypot(
        nearest_screen[0] - cursor_screen[0],
        nearest_screen[1] - cursor_screen[1],
    )


def _nearest_on_line_screen(
    cursor_screen: tuple[float, float],
    line: _LinePrimitive,
    *,
    image_to_screen: _ScreenTransform,
) -> Point:
    start_screen = _screen_xy(image_to_screen(line.start))
    end_screen = _screen_xy(image_to_screen(line.end))
    dx = end_screen[0] - start_screen[0]
    dy = end_screen[1] - start_screen[1]
    denominator = dx * dx + dy * dy
    if denominator <= 1e-18:
        return _copy_point(line.start)
    parameter = (
        (cursor_screen[0] - start_screen[0]) * dx
        + (cursor_screen[1] - start_screen[1]) * dy
    ) / denominator
    if line.domain == "segment":
        parameter = max(0.0, min(1.0, parameter))
    elif line.domain == "ray":
        parameter = max(0.0, parameter)
    return Point(
        line.start.x + parameter * (line.end.x - line.start.x),
        line.start.y + parameter * (line.end.y - line.start.y),
    )


def _nearest_on_circle_screen(
    cursor_screen: tuple[float, float],
    circle: _CirclePrimitive,
    *,
    image_to_screen: _ScreenTransform,
) -> Point:
    center_screen = _screen_xy(image_to_screen(circle.center))
    x_axis_screen = _screen_xy(
        image_to_screen(Point(circle.center.x + circle.radius, circle.center.y))
    )
    screen_radius = math.hypot(
        x_axis_screen[0] - center_screen[0],
        x_axis_screen[1] - center_screen[1],
    )
    dx = cursor_screen[0] - center_screen[0]
    dy = cursor_screen[1] - center_screen[1]
    length = math.hypot(dx, dy)
    if length <= 1e-12 or screen_radius <= 1e-12:
        return Point(circle.center.x + circle.radius, circle.center.y)
    # View transforms in the canvas are uniform scale + translation.  Mapping
    # the screen-space direction back as a unit image direction is therefore
    # exact for every supported image and digital-slide viewport.
    return Point(
        circle.center.x + circle.radius * dx / length,
        circle.center.y + circle.radius * dy / length,
    )


def _screen_xy(value: object) -> tuple[float, float]:
    point = _as_point(value)
    if point is not None:
        return float(point.x), float(point.y)
    raise TypeError("image_to_screen must return Point, QPointF, or a two-value sequence")


def _as_point(value: object) -> Point | None:
    if value is None:
        return None
    if isinstance(value, Point):
        if math.isfinite(value.x) and math.isfinite(value.y):
            return Point(float(value.x), float(value.y))
        return None
    if isinstance(value, Mapping):
        try:
            x = float(value["x"])
            y = float(value["y"])
        except (KeyError, TypeError, ValueError):
            return None
        return Point(x, y) if math.isfinite(x) and math.isfinite(y) else None
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) >= 2:
        try:
            x = float(value[0])
            y = float(value[1])
        except (TypeError, ValueError):
            return None
        return Point(x, y) if math.isfinite(x) and math.isfinite(y) else None
    x = _read(value, "x", None)
    y = _read(value, "y", None)
    if callable(x):
        x = x()
    if callable(y):
        y = y()
    try:
        numeric_x = float(x)
        numeric_y = float(y)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric_x) or not math.isfinite(numeric_y):
        return None
    return Point(numeric_x, numeric_y)


def _vector_xy(value: object) -> tuple[float, float] | None:
    point = _as_point(value)
    if point is None:
        return None
    return point.x, point.y


def _read(value: object, name: str, default: Any) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _enum_token(value: object) -> str:
    raw = getattr(value, "value", value)
    return str(raw or "").strip().lower().replace("-", "_")


def _copy_point(point: Point) -> Point:
    return Point(float(point.x), float(point.y))


def _point_distance(left: Point, right: Point) -> float:
    return math.hypot(left.x - right.x, left.y - right.y)


def _cross(left: tuple[float, float], right: tuple[float, float]) -> float:
    return left[0] * right[1] - left[1] * right[0]


def _coerce_float(value: object, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _finite_nonnegative(value: object, default: float) -> float:
    numeric = _coerce_float(value, default)
    if not math.isfinite(numeric):
        return default
    return max(0.0, numeric)


__all__ = [
    "DEFAULT_SNAP_KINDS",
    "ObjectSnapEngine",
    "ObjectSnapSettings",
    "SnapCandidate",
    "SnapKind",
    "contextual_snap_candidate",
]
