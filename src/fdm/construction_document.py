from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import math

from fdm.construction_geometry import (
    ConstructionEntity,
    ConstructionResolver,
    LineExtent,
    LiveFeatureRef,
    ParallelLineSequence,
    ResolvedCircle,
    ResolvedConstruction,
    ResolvedGeometry,
    ResolvedLine,
    ResolvedLineArray,
    ResolvedPoint,
    SourceObjectKind,
)
from fdm.geometry import Point
from fdm.models import ImageDocument, Measurement


@dataclass(frozen=True, slots=True)
class ResolvedSourceCandidate:
    """One analytical object/feature available to a construction command."""

    object_id: str
    object_kind: str
    feature: str
    geometry: ResolvedGeometry
    distance_px: float


def resolve_measurement_geometry(
    document: ImageDocument,
    reference: LiveFeatureRef,
) -> ResolvedGeometry | None:
    """Expose only analytical measurement geometry to construction refs."""

    measurement = document.get_measurement(reference.object_id)
    if measurement is None:
        return None
    kind = str(measurement.measurement_kind or "").strip().lower()
    mode = str(measurement.mode or "").strip().lower()
    if mode.startswith("freehand") or mode.startswith("magic"):
        return None
    if kind == "line" and measurement.line_px is not None:
        line = measurement.effective_line()
        return ResolvedLine(line.start, line.end, LineExtent.SEGMENT)
    if kind == "polyline" and len(measurement.polyline_px) >= 2:
        return ResolvedLineArray(
            tuple(
                ResolvedLine(start, end, LineExtent.SEGMENT)
                for start, end in zip(
                    measurement.polyline_px,
                    measurement.polyline_px[1:],
                )
            )
        )
    if kind == "area" and mode in {"polygon", "polygon_area"}:
        points = measurement.polygon_px
        if len(points) >= 3:
            return ResolvedLineArray(
                tuple(
                    ResolvedLine(
                        point,
                        points[(index + 1) % len(points)],
                        LineExtent.SEGMENT,
                    )
                    for index, point in enumerate(points)
                )
            )
    if kind == "count" and measurement.point_px is not None:
        return ResolvedPoint(measurement.point_px)
    return None


def make_construction_resolver(document: ImageDocument) -> ConstructionResolver:
    return ConstructionResolver(
        document.id,
        document.construction_entities,
        external_feature_resolver=lambda reference: resolve_measurement_geometry(
            document,
            reference,
        ),
    )


def resolved_construction_entries(
    document: ImageDocument,
) -> tuple[tuple[ConstructionEntity, ResolvedConstruction], ...]:
    resolver = make_construction_resolver(document)
    return tuple(
        (entity, resolver.resolve(entity))
        for entity in document.construction_entities
    )


def resolved_geometry_center(geometry: ResolvedGeometry) -> Point:
    if isinstance(geometry, ResolvedPoint):
        return geometry.point
    if isinstance(geometry, ResolvedCircle):
        return geometry.center
    if isinstance(geometry, ResolvedLine):
        return Point(
            (geometry.start.x + geometry.end.x) / 2.0,
            (geometry.start.y + geometry.end.y) / 2.0,
        )
    if isinstance(geometry.lines, ParallelLineSequence):
        lines = geometry.lines
        if lines.side.value == "both":
            mean_multiplier = 0.0
        else:
            sign = 1.0 if lines.side.value == "positive" else -1.0
            mean_multiplier = sign * (lines.per_side_count + 1) / 2.0
        base_center = resolved_geometry_center(lines.base_line)
        dx, dy = lines.base_line.direction
        return Point(
            base_center.x + (-dy) * lines.spacing * mean_multiplier,
            base_center.y + dx * lines.spacing * mean_multiplier,
        )
    points = [resolved_geometry_center(line) for line in geometry.lines]
    return Point(
        sum(point.x for point in points) / len(points),
        sum(point.y for point in points) / len(points),
    )


def construction_center(
    document: ImageDocument,
    construction_id: str,
) -> Point | None:
    resolved = make_construction_resolver(document).resolve(construction_id)
    if not resolved.valid or resolved.geometry is None:
        return None
    return resolved_geometry_center(resolved.geometry)


def resolved_geometry_bounds(
    geometry: ResolvedGeometry,
    *,
    image_size: tuple[int, int] | None = None,
) -> tuple[float, float, float, float] | None:
    if isinstance(geometry, ResolvedPoint):
        return (geometry.point.x, geometry.point.y, geometry.point.x, geometry.point.y)
    if isinstance(geometry, ResolvedCircle):
        return (
            geometry.center.x - geometry.radius,
            geometry.center.y - geometry.radius,
            geometry.center.x + geometry.radius,
            geometry.center.y + geometry.radius,
        )
    if isinstance(geometry, ResolvedLine):
        if geometry.extent is LineExtent.SEGMENT or image_size is None:
            return (
                min(geometry.start.x, geometry.end.x),
                min(geometry.start.y, geometry.end.y),
                max(geometry.start.x, geometry.end.x),
                max(geometry.start.y, geometry.end.y),
            )
        clipped = clip_line_to_rect(
            geometry,
            (0.0, 0.0, float(image_size[0] - 1), float(image_size[1] - 1)),
        )
        if clipped is None:
            return None
        return (
            min(clipped[0].x, clipped[1].x),
            min(clipped[0].y, clipped[1].y),
            max(clipped[0].x, clipped[1].x),
            max(clipped[0].y, clipped[1].y),
        )
    lines_to_bound: Iterable[ResolvedLine]
    if isinstance(geometry.lines, ParallelLineSequence):
        sequence = geometry.lines
        extreme_indices = {
            index
            for multiplier in (-sequence.per_side_count, -1, 1, sequence.per_side_count)
            if (index := sequence.index_for_multiplier(multiplier)) is not None
        }
        lines_to_bound = (sequence[index] for index in sorted(extreme_indices))
    else:
        lines_to_bound = geometry.lines
    child_bounds = [
        resolved_geometry_bounds(line, image_size=image_size)
        for line in lines_to_bound
    ]
    valid = [bounds for bounds in child_bounds if bounds is not None]
    if not valid:
        return None
    return (
        min(bounds[0] for bounds in valid),
        min(bounds[1] for bounds in valid),
        max(bounds[2] for bounds in valid),
        max(bounds[3] for bounds in valid),
    )


def clip_line_to_rect(
    line: ResolvedLine,
    rect: tuple[float, float, float, float],
) -> tuple[Point, Point] | None:
    """Clip a segment, ray or xline without manufacturing snap endpoints."""

    left, top, right, bottom = rect
    dx = line.end.x - line.start.x
    dy = line.end.y - line.start.y
    t_min = 0.0 if line.extent is not LineExtent.INFINITE else -math.inf
    t_max = 1.0 if line.extent is LineExtent.SEGMENT else math.inf
    for p, q in (
        (-dx, line.start.x - left),
        (dx, right - line.start.x),
        (-dy, line.start.y - top),
        (dy, bottom - line.start.y),
    ):
        if abs(p) <= 1e-12:
            if q < 0.0:
                return None
            continue
        ratio = q / p
        if p < 0.0:
            t_min = max(t_min, ratio)
        else:
            t_max = min(t_max, ratio)
        if t_min > t_max:
            return None
    if not math.isfinite(t_min) or not math.isfinite(t_max):
        return None
    return (
        Point(line.start.x + dx * t_min, line.start.y + dy * t_min),
        Point(line.start.x + dx * t_max, line.start.y + dy * t_max),
    )


def closest_resolved_source(
    document: ImageDocument,
    point: Point,
    *,
    tolerance_px: float,
    require_line: bool = False,
) -> tuple[str, str, ResolvedGeometry] | None:
    """Return an analytical source under the pointer for construction tools."""

    candidates = resolved_source_candidates(
        document,
        point,
        tolerance_px=tolerance_px,
        require_line=require_line,
    )
    if not candidates:
        return None
    best = candidates[0]
    return best.object_id, best.feature, best.geometry


def resolved_source_candidates(
    document: ImageDocument,
    point: Point,
    *,
    tolerance_px: float,
    require_line: bool = False,
    accepted_geometry_types: tuple[type, ...] = (),
    construction_entries: Iterable[
        tuple[ConstructionEntity, ResolvedConstruction]
    ] | None = None,
    measurements: Iterable[Measurement] | None = None,
) -> tuple[ResolvedSourceCandidate, ...]:
    """Return all nearby source identities in stable distance/order ranking.

    This is separate from ordinary coordinate snapping: derived commands need
    to preserve the exact object/feature identity and may therefore present a
    choice when two analytical sources occupy the same screen location.
    """

    candidates: list[ResolvedSourceCandidate] = []
    if construction_entries is None:
        resolver = make_construction_resolver(document)
        entries = tuple(
            (entity, resolver.resolve(entity))
            for entity in document.construction_entities
        )
    else:
        entries = tuple(construction_entries)
    for entity, resolved in reversed(entries):
        if not entity.visible or not entity.snap_enabled:
            continue
        if not resolved.valid or resolved.geometry is None:
            continue
        for feature, geometry in _source_geometries_near(
            resolved.geometry,
            point,
            tolerance_px,
        ):
            if require_line and not isinstance(geometry, ResolvedLine):
                continue
            if accepted_geometry_types and not isinstance(
                geometry,
                accepted_geometry_types,
            ):
                continue
            distance_px = _distance_to_geometry(point, geometry)
            if distance_px <= tolerance_px:
                candidates.append(
                    ResolvedSourceCandidate(
                        object_id=entity.id,
                        object_kind=SourceObjectKind.CONSTRUCTION,
                        feature=feature,
                        geometry=geometry,
                        distance_px=distance_px,
                    )
                )
    measurement_entries = (
        tuple(document.measurements)
        if measurements is None
        else tuple(measurements)
    )
    for measurement in reversed(measurement_entries):
        geometry = resolve_measurement_geometry(
            document,
            LiveFeatureRef(
                document.id,
                measurement.id,
                object_kind="measurement",
            ),
        )
        if geometry is None:
            continue
        for feature, child in _source_geometries_near(
            geometry,
            point,
            tolerance_px,
        ):
            if require_line and not isinstance(child, ResolvedLine):
                continue
            if accepted_geometry_types and not isinstance(
                child,
                accepted_geometry_types,
            ):
                continue
            distance_px = _distance_to_geometry(point, child)
            if distance_px <= tolerance_px:
                candidates.append(
                    ResolvedSourceCandidate(
                        object_id=measurement.id,
                        object_kind=SourceObjectKind.MEASUREMENT,
                        feature=feature,
                        geometry=child,
                        distance_px=distance_px,
                    )
                )
    candidates.sort(key=lambda candidate: candidate.distance_px)
    return tuple(candidates)


def _source_geometries(
    geometry: ResolvedGeometry,
) -> Iterable[tuple[str, ResolvedGeometry]]:
    if isinstance(geometry, ResolvedLineArray):
        for index, line in enumerate(geometry.lines):
            yield _line_array_feature_key(geometry.lines, index), line
        return
    yield "geometry", geometry


def _source_geometries_near(
    geometry: ResolvedGeometry,
    point: Point,
    tolerance_px: float,
) -> Iterable[tuple[str, ResolvedGeometry]]:
    if (
        isinstance(geometry, ResolvedLineArray)
        and isinstance(geometry.lines, ParallelLineSequence)
    ):
        for index, line in geometry.lines.indexed_near_point(
            point,
            tolerance_px,
        ):
            feature = _line_array_feature_key(geometry.lines, index)
            yield feature, line
            yield from _line_point_features(line, prefix=f"{feature}:")
        return
    if isinstance(geometry, ResolvedLineArray):
        for index, line in enumerate(geometry.lines):
            feature = _line_array_feature_key(geometry.lines, index)
            yield feature, line
            yield from _line_point_features(line, prefix=f"{feature}:")
        return
    yield "geometry", geometry
    if isinstance(geometry, ResolvedLine):
        yield from _line_point_features(geometry)
    elif isinstance(geometry, ResolvedCircle):
        yield "center", ResolvedPoint(geometry.center)
        for quadrant in range(4):
            angle = quadrant * math.pi / 2.0
            yield (
                f"quadrant:{quadrant}",
                ResolvedPoint(
                    Point(
                        geometry.center.x + geometry.radius * math.cos(angle),
                        geometry.center.y + geometry.radius * math.sin(angle),
                    )
                ),
            )


def _line_array_feature_key(lines: object, index: int) -> str:
    if isinstance(lines, ParallelLineSequence):
        return f"line:{lines.multiplier_at(index):+d}"
    return f"line:{index}"


def _line_point_features(
    line: ResolvedLine,
    *,
    prefix: str = "",
) -> Iterable[tuple[str, ResolvedPoint]]:
    if line.extent is not LineExtent.INFINITE:
        yield f"{prefix}start", ResolvedPoint(line.start)
    if line.extent is LineExtent.SEGMENT:
        yield f"{prefix}end", ResolvedPoint(line.end)
        yield (
            f"{prefix}midpoint",
            ResolvedPoint(
                Point(
                    (line.start.x + line.end.x) / 2.0,
                    (line.start.y + line.end.y) / 2.0,
                )
            ),
        )


def _distance_to_geometry(point: Point, geometry: ResolvedGeometry) -> float:
    if isinstance(geometry, ResolvedPoint):
        return math.hypot(point.x - geometry.point.x, point.y - geometry.point.y)
    if isinstance(geometry, ResolvedCircle):
        return abs(
            math.hypot(point.x - geometry.center.x, point.y - geometry.center.y)
            - geometry.radius
        )
    if isinstance(geometry, ResolvedLine):
        dx = geometry.end.x - geometry.start.x
        dy = geometry.end.y - geometry.start.y
        denominator = dx * dx + dy * dy
        if denominator <= 1e-12:
            return math.inf
        parameter = (
            (point.x - geometry.start.x) * dx
            + (point.y - geometry.start.y) * dy
        ) / denominator
        if geometry.extent is LineExtent.SEGMENT:
            parameter = max(0.0, min(1.0, parameter))
        elif geometry.extent is LineExtent.RAY:
            parameter = max(0.0, parameter)
        projected = Point(
            geometry.start.x + parameter * dx,
            geometry.start.y + parameter * dy,
        )
        return math.hypot(point.x - projected.x, point.y - projected.y)
    if isinstance(geometry.lines, ParallelLineSequence):
        candidates = geometry.lines.indexed_nearest(point)
        return min(
            (_distance_to_geometry(point, line) for _index, line in candidates),
            default=math.inf,
        )
    return min(_distance_to_geometry(point, line) for line in geometry.lines)


__all__ = [
    "clip_line_to_rect",
    "closest_resolved_source",
    "construction_center",
    "make_construction_resolver",
    "resolve_measurement_geometry",
    "resolved_source_candidates",
    "ResolvedSourceCandidate",
    "resolved_construction_entries",
    "resolved_geometry_bounds",
    "resolved_geometry_center",
]
