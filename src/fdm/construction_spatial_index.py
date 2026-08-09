"""Viewport-independent spatial index for resolved construction geometry.

The index is deliberately a broad geometry service rather than a canvas
cache.  Coordinates stay in full image/slide pixel space, and callers decide
when to rebuild it by comparing the document construction revision.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
import math
from typing import TypeAlias

from fdm.construction_geometry import (
    ConstructionEntity,
    LineExtent,
    ParallelLineSequence,
    ResolvedCircle,
    ResolvedConstruction,
    ResolvedLine,
    ResolvedLineArray,
    ResolvedPoint,
)
from fdm.geometry import Point


SpatialPrimitive: TypeAlias = ResolvedPoint | ResolvedLine | ResolvedCircle
Bounds: TypeAlias = tuple[float, float, float, float]
CellKey: TypeAlias = tuple[int, int]
SpatialPredicate: TypeAlias = Callable[["ConstructionSpatialItem"], bool]


@dataclass(frozen=True, slots=True)
class ConstructionSpatialItem:
    """One indexed primitive retaining its owning entity and resolution."""

    ordinal: int
    owner_order: int
    feature_order: int
    entity: ConstructionEntity
    resolved: ResolvedConstruction
    geometry: SpatialPrimitive
    feature_key: str
    bounds: Bounds | None
    unbounded: bool

    @property
    def entity_id(self) -> str:
        return self.entity.id


@dataclass(frozen=True, slots=True)
class ConstructionSpatialIndexStats:
    item_count: int
    finite_item_count: int
    unbounded_item_count: int
    oversized_item_count: int
    grid_cell_count: int
    grid_reference_count: int


@dataclass(frozen=True, slots=True)
class _ParametricArrayEntry:
    owner_order: int
    entity: ConstructionEntity
    resolved: ResolvedConstruction
    lines: ParallelLineSequence


class ConstructionSpatialIndex:
    """Uniform-grid index with a separate analytical unbounded collection."""

    DEFAULT_CELL_SIZE_PX = 256.0
    DEFAULT_MAX_CELLS_PER_ITEM = 4096

    def __init__(
        self,
        *,
        revision: int | None,
        cell_size_px: float,
        max_cells_per_item: int,
        items: tuple[ConstructionSpatialItem, ...],
        grid: dict[CellKey, tuple[int, ...]],
        unbounded_ordinals: tuple[int, ...],
        oversized_ordinals: tuple[int, ...],
        parametric_arrays: tuple[_ParametricArrayEntry, ...],
    ) -> None:
        self.revision = revision
        self.cell_size_px = cell_size_px
        self.max_cells_per_item = max_cells_per_item
        self._items = items
        self._grid = grid
        self._unbounded_ordinals = unbounded_ordinals
        self._oversized_ordinals = oversized_ordinals
        self._parametric_arrays = parametric_arrays
        finite_count = sum(not item.unbounded for item in items)
        parametric_count = sum(len(entry.lines) for entry in parametric_arrays)
        parametric_unbounded_count = sum(
            len(entry.lines)
            for entry in parametric_arrays
            if entry.lines.extent is not LineExtent.SEGMENT
        )
        self.stats = ConstructionSpatialIndexStats(
            item_count=len(items) + parametric_count,
            finite_item_count=(
                finite_count + parametric_count - parametric_unbounded_count
            ),
            unbounded_item_count=(
                len(unbounded_ordinals) + parametric_unbounded_count
            ),
            oversized_item_count=len(oversized_ordinals),
            grid_cell_count=len(grid),
            grid_reference_count=sum(len(ordinals) for ordinals in grid.values()),
        )

    @classmethod
    def build(
        cls,
        entries: Iterable[tuple[ConstructionEntity, ResolvedConstruction]],
        *,
        revision: int | None = None,
        cell_size_px: float = DEFAULT_CELL_SIZE_PX,
        max_cells_per_item: int = DEFAULT_MAX_CELLS_PER_ITEM,
    ) -> "ConstructionSpatialIndex":
        cell_size = _positive_finite(cell_size_px, "cell_size_px")
        max_cells = int(max_cells_per_item)
        if max_cells < 1:
            raise ValueError("max_cells_per_item 必须至少为 1")

        items: list[ConstructionSpatialItem] = []
        grid_lists: dict[CellKey, list[int]] = {}
        unbounded: list[int] = []
        oversized: list[int] = []
        parametric_arrays: list[_ParametricArrayEntry] = []
        for owner_order, (entity, resolved) in enumerate(entries):
            if not isinstance(entity, ConstructionEntity):
                raise TypeError("空间索引条目的 entity 必须是 ConstructionEntity")
            if not isinstance(resolved, ResolvedConstruction):
                raise TypeError("空间索引条目的 resolved 必须是 ResolvedConstruction")
            if not resolved.valid or resolved.geometry is None:
                continue
            if (
                isinstance(resolved.geometry, ResolvedLineArray)
                and isinstance(resolved.geometry.lines, ParallelLineSequence)
            ):
                parametric_arrays.append(
                    _ParametricArrayEntry(
                        owner_order,
                        entity,
                        resolved,
                        resolved.geometry.lines,
                    )
                )
                continue
            for feature_order, (feature_key, primitive) in enumerate(
                _resolved_primitives(resolved.geometry)
            ):
                is_unbounded = (
                    isinstance(primitive, ResolvedLine)
                    and primitive.extent is not LineExtent.SEGMENT
                )
                bounds = None if is_unbounded else _primitive_bounds(primitive)
                ordinal = len(items)
                item = ConstructionSpatialItem(
                    ordinal=ordinal,
                    owner_order=owner_order,
                    feature_order=feature_order,
                    entity=entity,
                    resolved=resolved,
                    geometry=primitive,
                    feature_key=feature_key,
                    bounds=bounds,
                    unbounded=is_unbounded,
                )
                items.append(item)
                if is_unbounded:
                    unbounded.append(ordinal)
                    continue
                assert bounds is not None
                if not all(math.isfinite(value) for value in bounds):
                    oversized.append(ordinal)
                    continue
                cell_bounds = _bounds_cell_range(bounds, cell_size)
                cell_count = (
                    (cell_bounds[2] - cell_bounds[0] + 1)
                    * (cell_bounds[3] - cell_bounds[1] + 1)
                )
                if cell_count > max_cells:
                    # A gigapixel-scale circle must not allocate millions of
                    # hash buckets.  It remains finite and is checked using
                    # the same exact query predicate from this small side list.
                    oversized.append(ordinal)
                    continue
                for cell_x in range(cell_bounds[0], cell_bounds[2] + 1):
                    for cell_y in range(cell_bounds[1], cell_bounds[3] + 1):
                        grid_lists.setdefault((cell_x, cell_y), []).append(ordinal)
        grid = {key: tuple(value) for key, value in grid_lists.items()}
        return cls(
            revision=int(revision) if revision is not None else None,
            cell_size_px=cell_size,
            max_cells_per_item=max_cells,
            items=tuple(items),
            grid=grid,
            unbounded_ordinals=tuple(unbounded),
            oversized_ordinals=tuple(oversized),
            parametric_arrays=tuple(parametric_arrays),
        )

    @classmethod
    def build_for_revision(
        cls,
        entries: Iterable[tuple[ConstructionEntity, ResolvedConstruction]],
        *,
        revision: int,
        previous: "ConstructionSpatialIndex | None" = None,
        cell_size_px: float = DEFAULT_CELL_SIZE_PX,
        max_cells_per_item: int = DEFAULT_MAX_CELLS_PER_ITEM,
    ) -> "ConstructionSpatialIndex":
        """Reuse a matching index or rebuild it for a document revision."""

        cell_size = _positive_finite(cell_size_px, "cell_size_px")
        max_cells = int(max_cells_per_item)
        if (
            previous is not None
            and previous.is_current(revision)
            and previous.cell_size_px == cell_size
            and previous.max_cells_per_item == max_cells
        ):
            return previous
        return cls.build(
            entries,
            revision=revision,
            cell_size_px=cell_size,
            max_cells_per_item=max_cells,
        )

    @property
    def items(self) -> tuple[ConstructionSpatialItem, ...]:
        return self._items

    def is_current(self, revision: int | None) -> bool:
        normalized = int(revision) if revision is not None else None
        return self.revision == normalized

    def query(
        self,
        cursor_image_px: Point,
        radius_px: float,
        *,
        predicate: SpatialPredicate | None = None,
    ) -> tuple[ConstructionSpatialItem, ...]:
        """Return nearby primitives in stable document/feature order.

        Only grid cells touched by the query aperture are inspected.  Rays and
        infinite lines are kept outside the grid and tested analytically.
        Circle proximity includes both its center and circumference because
        both are object-snap sources.  No intersection is generated here.
        """

        cursor = _point(cursor_image_px)
        radius = _nonnegative_finite(radius_px, "radius_px")
        query_bounds = (
            cursor.x - radius,
            cursor.y - radius,
            cursor.x + radius,
            cursor.y + radius,
        )
        candidate_ordinals: set[int] = set(self._unbounded_ordinals)
        candidate_ordinals.update(self._oversized_ordinals)
        if not all(math.isfinite(value) for value in query_bounds):
            candidate_ordinals.update(range(len(self._items)))
        else:
            cell_bounds = _bounds_cell_range(query_bounds, self.cell_size_px)
            query_cell_count = (
                (cell_bounds[2] - cell_bounds[0] + 1)
                * (cell_bounds[3] - cell_bounds[1] + 1)
            )
            if query_cell_count > max(4096, len(self._grid) * 2):
                candidate_ordinals.update(range(len(self._items)))
            else:
                for cell_x in range(cell_bounds[0], cell_bounds[2] + 1):
                    for cell_y in range(cell_bounds[1], cell_bounds[3] + 1):
                        candidate_ordinals.update(self._grid.get((cell_x, cell_y), ()))

        result: list[ConstructionSpatialItem] = []
        for ordinal in sorted(candidate_ordinals):
            item = self._items[ordinal]
            if predicate is not None and not predicate(item):
                continue
            if item.bounds is not None and not _bounds_overlap(item.bounds, query_bounds):
                continue
            if _primitive_query_distance(item.geometry, cursor) > radius + 1e-9:
                continue
            result.append(item)
        for entry in self._parametric_arrays:
            for feature_order, line in entry.lines.indexed_near_point(
                cursor,
                radius,
            ):
                is_unbounded = line.extent is not LineExtent.SEGMENT
                item = ConstructionSpatialItem(
                    ordinal=-1,
                    owner_order=entry.owner_order,
                    feature_order=feature_order,
                    entity=entry.entity,
                    resolved=entry.resolved,
                    geometry=line,
                    feature_key=(
                        f"line:{entry.lines.multiplier_at(feature_order):+d}"
                    ),
                    bounds=None if is_unbounded else _primitive_bounds(line),
                    unbounded=is_unbounded,
                )
                if predicate is not None and not predicate(item):
                    continue
                if _primitive_query_distance(line, cursor) > radius + 1e-9:
                    continue
                result.append(item)
        result.sort(key=lambda item: (item.owner_order, item.feature_order))
        return tuple(result)

    def query_pairs(
        self,
        cursor_image_px: Point,
        radius_px: float,
        *,
        predicate: SpatialPredicate | None = None,
    ) -> tuple[tuple[ConstructionEntity, ResolvedConstruction], ...]:
        """Return owner pairs once each for consumers that resolve whole entities."""

        seen: set[str] = set()
        result: list[tuple[ConstructionEntity, ResolvedConstruction]] = []
        for item in self.query(cursor_image_px, radius_px, predicate=predicate):
            if item.entity.id in seen:
                continue
            seen.add(item.entity.id)
            result.append((item.entity, item.resolved))
        return tuple(result)

    def query_snappable(
        self,
        cursor_image_px: Point,
        radius_px: float,
    ) -> tuple[ConstructionSpatialItem, ...]:
        """Convenience query honoring entity visibility and snap-enabled state."""

        return self.query(
            cursor_image_px,
            radius_px,
            predicate=lambda item: item.entity.visible and item.entity.snap_enabled,
        )


def _resolved_primitives(
    geometry: ResolvedPoint | ResolvedLine | ResolvedCircle | ResolvedLineArray,
) -> Iterable[tuple[str, SpatialPrimitive]]:
    if isinstance(geometry, ResolvedLineArray):
        for index, line in enumerate(geometry.lines):
            feature = (
                f"line:{geometry.lines.multiplier_at(index):+d}"
                if isinstance(geometry.lines, ParallelLineSequence)
                else f"line:{index}"
            )
            yield feature, line
        return
    yield "geometry", geometry


def _primitive_bounds(primitive: SpatialPrimitive) -> Bounds:
    if isinstance(primitive, ResolvedPoint):
        return (
            primitive.point.x,
            primitive.point.y,
            primitive.point.x,
            primitive.point.y,
        )
    if isinstance(primitive, ResolvedCircle):
        return (
            primitive.center.x - primitive.radius,
            primitive.center.y - primitive.radius,
            primitive.center.x + primitive.radius,
            primitive.center.y + primitive.radius,
        )
    return (
        min(primitive.start.x, primitive.end.x),
        min(primitive.start.y, primitive.end.y),
        max(primitive.start.x, primitive.end.x),
        max(primitive.start.y, primitive.end.y),
    )


def _primitive_query_distance(primitive: SpatialPrimitive, cursor: Point) -> float:
    if isinstance(primitive, ResolvedPoint):
        return math.hypot(cursor.x - primitive.point.x, cursor.y - primitive.point.y)
    if isinstance(primitive, ResolvedCircle):
        center_distance = math.hypot(
            cursor.x - primitive.center.x,
            cursor.y - primitive.center.y,
        )
        return min(center_distance, abs(center_distance - primitive.radius))
    direction_x, direction_y = primitive.direction
    parameter = (
        (cursor.x - primitive.start.x) * direction_x
        + (cursor.y - primitive.start.y) * direction_y
    )
    if primitive.extent is LineExtent.SEGMENT:
        parameter = max(0.0, min(primitive.length, parameter))
    elif primitive.extent is LineExtent.RAY:
        parameter = max(0.0, parameter)
    projected = primitive.point_at(parameter)
    return math.hypot(cursor.x - projected.x, cursor.y - projected.y)


def _bounds_cell_range(
    bounds: Bounds,
    cell_size_px: float,
) -> tuple[int, int, int, int]:
    return (
        math.floor(bounds[0] / cell_size_px),
        math.floor(bounds[1] / cell_size_px),
        math.floor(bounds[2] / cell_size_px),
        math.floor(bounds[3] / cell_size_px),
    )


def _bounds_overlap(first: Bounds, second: Bounds) -> bool:
    return not (
        first[2] < second[0]
        or first[0] > second[2]
        or first[3] < second[1]
        or first[1] > second[3]
    )


def _point(value: object) -> Point:
    try:
        x = float(getattr(value, "x"))
        y = float(getattr(value, "y"))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("cursor_image_px 必须是有限坐标点") from exc
    if not math.isfinite(x) or not math.isfinite(y):
        raise ValueError("cursor_image_px 必须是有限坐标点")
    return Point(x, y)


def _positive_finite(value: object, field_name: str) -> float:
    numeric = _nonnegative_finite(value, field_name)
    if numeric <= 0.0:
        raise ValueError(f"{field_name} 必须大于 0")
    return numeric


def _nonnegative_finite(value: object, field_name: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} 必须是非负有限数") from exc
    if not math.isfinite(numeric) or numeric < 0.0:
        raise ValueError(f"{field_name} 必须是非负有限数")
    return numeric


__all__ = [
    "Bounds",
    "ConstructionSpatialIndex",
    "ConstructionSpatialIndexStats",
    "ConstructionSpatialItem",
    "SpatialPredicate",
    "SpatialPrimitive",
]
