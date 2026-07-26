"""Project ROI domain model and exact pixel-centre rasterisation.

The objects in this module deliberately do not depend on Qt widgets or the
canvas display path.  Coordinates are authoritative image-pixel coordinates;
screen simplification must never be fed back into these records.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from enum import StrEnum
import math
import re
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray


PROJECT_ROI_SCHEMA_VERSION = 1
DEFAULT_ROI_COLOR = "#2A9D8F"
_ID_PATTERN = re.compile(r"^[^\x00-\x1f\x7f]{1,256}$")
_COLOR_PATTERN = re.compile(r"^#[0-9A-Fa-f]{6}$")

RoiBounds: TypeAlias = tuple[float, float, float, float]
RoiMask: TypeAlias = NDArray[np.bool_]


class ProjectRoiKind(StrEnum):
    RECTANGLE = "rect"
    ELLIPSE = "ellipse"
    POLYGON = "polygon"
    FREEHAND = "free"
    COMPOSITE = "composite"


class RoiBooleanOperator(StrEnum):
    UNION = "union"
    INTERSECTION = "intersection"
    DIFFERENCE = "difference"
    XOR = "xor"


@dataclass(frozen=True, slots=True)
class RoiPoint:
    x: float
    y: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "x", _finite_number(self.x, field_name="point.x"))
        object.__setattr__(self, "y", _finite_number(self.y, field_name="point.y"))

    def to_dict(self) -> dict[str, float]:
        return {"x": self.x, "y": self.y}

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "RoiPoint":
        _require_mapping(payload, field_name="point")
        _require_exact_keys(payload, required={"x", "y"}, field_name="point")
        return cls(x=payload["x"], y=payload["y"])  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class RectangleRoiGeometry:
    x: float
    y: float
    width: float
    height: float

    kind = ProjectRoiKind.RECTANGLE

    def __post_init__(self) -> None:
        object.__setattr__(self, "x", _finite_number(self.x, field_name="x"))
        object.__setattr__(self, "y", _finite_number(self.y, field_name="y"))
        object.__setattr__(
            self,
            "width",
            _positive_finite(self.width, field_name="width"),
        )
        object.__setattr__(
            self,
            "height",
            _positive_finite(self.height, field_name="height"),
        )

    @property
    def bounds(self) -> RoiBounds:
        return self.x, self.y, self.x + self.width, self.y + self.height

    def to_dict(self) -> dict[str, float]:
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "RectangleRoiGeometry":
        _require_mapping(payload, field_name="geometry")
        _require_exact_keys(
            payload,
            required={"x", "y", "width", "height"},
            field_name="geometry",
        )
        return cls(
            x=payload["x"],  # type: ignore[arg-type]
            y=payload["y"],  # type: ignore[arg-type]
            width=payload["width"],  # type: ignore[arg-type]
            height=payload["height"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class EllipseRoiGeometry:
    """Ellipse represented by its axis-aligned image-pixel bounding box."""

    x: float
    y: float
    width: float
    height: float

    kind = ProjectRoiKind.ELLIPSE

    def __post_init__(self) -> None:
        object.__setattr__(self, "x", _finite_number(self.x, field_name="x"))
        object.__setattr__(self, "y", _finite_number(self.y, field_name="y"))
        object.__setattr__(
            self,
            "width",
            _positive_finite(self.width, field_name="width"),
        )
        object.__setattr__(
            self,
            "height",
            _positive_finite(self.height, field_name="height"),
        )

    @property
    def bounds(self) -> RoiBounds:
        return self.x, self.y, self.x + self.width, self.y + self.height

    def to_dict(self) -> dict[str, float]:
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "EllipseRoiGeometry":
        _require_mapping(payload, field_name="geometry")
        _require_exact_keys(
            payload,
            required={"x", "y", "width", "height"},
            field_name="geometry",
        )
        return cls(
            x=payload["x"],  # type: ignore[arg-type]
            y=payload["y"],  # type: ignore[arg-type]
            width=payload["width"],  # type: ignore[arg-type]
            height=payload["height"],  # type: ignore[arg-type]
        )


def _freeze_rings(
    rings: Iterable[Iterable[RoiPoint | Mapping[str, object] | tuple[float, float]]],
) -> tuple[tuple[RoiPoint, ...], ...]:
    frozen: list[tuple[RoiPoint, ...]] = []
    for ring_index, ring in enumerate(rings):
        points: list[RoiPoint] = []
        for point_index, point in enumerate(ring):
            if isinstance(point, RoiPoint):
                normalized = point
            elif isinstance(point, Mapping):
                normalized = RoiPoint.from_dict(point)
            else:
                try:
                    x, y = point
                except (TypeError, ValueError) as error:
                    raise TypeError(
                        f"rings[{ring_index}][{point_index}] 必须是二维坐标"
                    ) from error
                normalized = RoiPoint(x, y)
            points.append(normalized)
        if len(points) < 3:
            raise ValueError(f"rings[{ring_index}] 至少需要 3 个点")
        if len({(point.x, point.y) for point in points}) < 3:
            raise ValueError(f"rings[{ring_index}] 至少需要 3 个不同的点")
        frozen.append(tuple(points))
    if not frozen:
        raise ValueError("rings 至少需要一个环")
    return tuple(frozen)


@dataclass(frozen=True, slots=True)
class PolygonRoiGeometry:
    rings: tuple[tuple[RoiPoint, ...], ...]

    kind = ProjectRoiKind.POLYGON

    def __post_init__(self) -> None:
        object.__setattr__(self, "rings", _freeze_rings(self.rings))

    @property
    def bounds(self) -> RoiBounds:
        return _rings_bounds(self.rings)

    def to_dict(self) -> dict[str, object]:
        return {
            "rings": [
                [point.to_dict() for point in ring]
                for ring in self.rings
            ]
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "PolygonRoiGeometry":
        _require_mapping(payload, field_name="geometry")
        _require_exact_keys(payload, required={"rings"}, field_name="geometry")
        rings = payload["rings"]
        if not isinstance(rings, list):
            raise TypeError("geometry.rings 必须是列表")
        if any(not isinstance(ring, list) for ring in rings):
            raise TypeError("geometry.rings 中的每个环必须是列表")
        return cls(rings=_freeze_rings(rings))  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class FreehandRoiGeometry:
    rings: tuple[tuple[RoiPoint, ...], ...]

    kind = ProjectRoiKind.FREEHAND

    def __post_init__(self) -> None:
        object.__setattr__(self, "rings", _freeze_rings(self.rings))

    @property
    def bounds(self) -> RoiBounds:
        return _rings_bounds(self.rings)

    def to_dict(self) -> dict[str, object]:
        return {
            "rings": [
                [point.to_dict() for point in ring]
                for ring in self.rings
            ]
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "FreehandRoiGeometry":
        _require_mapping(payload, field_name="geometry")
        _require_exact_keys(payload, required={"rings"}, field_name="geometry")
        rings = payload["rings"]
        if not isinstance(rings, list):
            raise TypeError("geometry.rings 必须是列表")
        if any(not isinstance(ring, list) for ring in rings):
            raise TypeError("geometry.rings 中的每个环必须是列表")
        return cls(rings=_freeze_rings(rings))  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class RoiBooleanExpression:
    operator: RoiBooleanOperator
    operand_ids: tuple[str, ...]

    kind = ProjectRoiKind.COMPOSITE

    def __post_init__(self) -> None:
        try:
            operator = RoiBooleanOperator(self.operator)
        except (TypeError, ValueError) as error:
            raise ValueError(f"不支持的 ROI 布尔运算: {self.operator!r}") from error
        operand_ids = tuple(
            _required_id(value, field_name=f"operand_ids[{index}]")
            for index, value in enumerate(self.operand_ids)
        )
        if len(operand_ids) < 2:
            raise ValueError("ROI 布尔表达式至少需要两个成员")
        if len(set(operand_ids)) != len(operand_ids):
            raise ValueError("ROI 布尔表达式不能重复引用同一个成员")
        object.__setattr__(self, "operator", operator)
        object.__setattr__(self, "operand_ids", operand_ids)

    def to_dict(self) -> dict[str, object]:
        return {
            "operator": self.operator.value,
            "operand_ids": list(self.operand_ids),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "RoiBooleanExpression":
        _require_mapping(payload, field_name="geometry")
        _require_exact_keys(
            payload,
            required={"operator", "operand_ids"},
            field_name="geometry",
        )
        operand_ids = payload["operand_ids"]
        if not isinstance(operand_ids, list):
            raise TypeError("geometry.operand_ids 必须是列表")
        return cls(
            operator=payload["operator"],  # type: ignore[arg-type]
            operand_ids=tuple(operand_ids),  # type: ignore[arg-type]
        )


ProjectRoiGeometry: TypeAlias = (
    RectangleRoiGeometry
    | EllipseRoiGeometry
    | PolygonRoiGeometry
    | FreehandRoiGeometry
    | RoiBooleanExpression
)


@dataclass(frozen=True, slots=True)
class ProjectRoi:
    id: str
    document_id: str
    name: str
    geometry: ProjectRoiGeometry
    group: str | None = None
    visible: bool = True
    locked: bool = False
    color: str = DEFAULT_ROI_COLOR
    revision: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _required_id(self.id, field_name="id"))
        object.__setattr__(
            self,
            "document_id",
            _required_id(self.document_id, field_name="document_id"),
        )
        object.__setattr__(
            self,
            "name",
            _required_text(self.name, field_name="name", maximum_length=256),
        )
        if not isinstance(
            self.geometry,
            (
                RectangleRoiGeometry,
                EllipseRoiGeometry,
                PolygonRoiGeometry,
                FreehandRoiGeometry,
                RoiBooleanExpression,
            ),
        ):
            raise TypeError("geometry 不是受支持的 ROI 几何")
        group = None
        if self.group is not None:
            group = _required_text(
                self.group,
                field_name="group",
                maximum_length=256,
            )
        if not isinstance(self.visible, bool):
            raise TypeError("visible 必须是布尔值")
        if not isinstance(self.locked, bool):
            raise TypeError("locked 必须是布尔值")
        color = str(self.color or "").strip().upper()
        if not _COLOR_PATTERN.fullmatch(color):
            raise ValueError("color 必须是 #RRGGBB")
        revision = _non_negative_int(self.revision, field_name="revision")
        object.__setattr__(self, "group", group)
        object.__setattr__(self, "color", color)
        object.__setattr__(self, "revision", revision)

    @property
    def kind(self) -> ProjectRoiKind:
        return self.geometry.kind

    def replace_geometry(self, geometry: ProjectRoiGeometry) -> "ProjectRoi":
        """Return a new ROI and advance only its geometry revision."""

        if geometry == self.geometry:
            return self
        return replace(self, geometry=geometry, revision=self.revision + 1)

    def with_metadata(
        self,
        *,
        name: str | None = None,
        group: str | None | object = ...,
        visible: bool | None = None,
        locked: bool | None = None,
        color: str | None = None,
    ) -> "ProjectRoi":
        changes: dict[str, object] = {}
        if name is not None:
            changes["name"] = name
        if group is not ...:
            changes["group"] = group
        if visible is not None:
            changes["visible"] = visible
        if locked is not None:
            changes["locked"] = locked
        if color is not None:
            changes["color"] = color
        return replace(self, **changes)

    def bounds(
        self,
        roi_lookup: Mapping[str, "ProjectRoi"] | None = None,
    ) -> RoiBounds:
        return roi_bounds(self, roi_lookup=roi_lookup)

    def rasterize_mask(
        self,
        width: int,
        height: int,
        *,
        roi_lookup: Mapping[str, "ProjectRoi"] | None = None,
    ) -> RoiMask:
        return rasterize_roi_mask(
            self,
            width,
            height,
            roi_lookup=roi_lookup,
        )

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": PROJECT_ROI_SCHEMA_VERSION,
            "id": self.id,
            "document_id": self.document_id,
            "name": self.name,
            "kind": self.kind.value,
            "geometry": self.geometry.to_dict(),
            "visible": self.visible,
            "locked": self.locked,
            "color": self.color,
            "revision": self.revision,
        }
        if self.group is not None:
            payload["group"] = self.group
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "ProjectRoi":
        _require_mapping(payload, field_name="ProjectRoi")
        _require_exact_keys(
            payload,
            required={
                "schema_version",
                "id",
                "document_id",
                "name",
                "kind",
                "geometry",
                "visible",
                "locked",
                "color",
                "revision",
            },
            optional={"group"},
            field_name="ProjectRoi",
        )
        schema_version = payload["schema_version"]
        if (
            isinstance(schema_version, bool)
            or not isinstance(schema_version, int)
            or schema_version != PROJECT_ROI_SCHEMA_VERSION
        ):
            raise ValueError(
                "不支持的 ProjectRoi schema_version: "
                f"{schema_version!r}"
            )
        try:
            kind = ProjectRoiKind(payload["kind"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"不支持的 ROI kind: {payload['kind']!r}") from error
        geometry_payload = payload["geometry"]
        if not isinstance(geometry_payload, Mapping):
            raise TypeError("ProjectRoi.geometry 必须是对象")
        geometry_factories = {
            ProjectRoiKind.RECTANGLE: RectangleRoiGeometry.from_dict,
            ProjectRoiKind.ELLIPSE: EllipseRoiGeometry.from_dict,
            ProjectRoiKind.POLYGON: PolygonRoiGeometry.from_dict,
            ProjectRoiKind.FREEHAND: FreehandRoiGeometry.from_dict,
            ProjectRoiKind.COMPOSITE: RoiBooleanExpression.from_dict,
        }
        return cls(
            id=payload["id"],  # type: ignore[arg-type]
            document_id=payload["document_id"],  # type: ignore[arg-type]
            name=payload["name"],  # type: ignore[arg-type]
            geometry=geometry_factories[kind](geometry_payload),
            group=payload.get("group"),  # type: ignore[arg-type]
            visible=payload["visible"],  # type: ignore[arg-type]
            locked=payload["locked"],  # type: ignore[arg-type]
            color=payload["color"],  # type: ignore[arg-type]
            revision=payload["revision"],  # type: ignore[arg-type]
        )


def rasterize_roi_mask(
    roi: ProjectRoi,
    width: int,
    height: int,
    *,
    roi_lookup: Mapping[str, ProjectRoi] | None = None,
) -> RoiMask:
    """Rasterize an ROI using original-pixel centre sampling.

    Pixel ``(column, row)`` is tested at image coordinate
    ``(column + 0.5, row + 0.5)``.  Polygon and freehand rings use odd-even
    fill regardless of ring order or winding direction.
    """

    width = _positive_int(width, field_name="width")
    height = _positive_int(height, field_name="height")
    return _rasterize_roi_mask(
        roi,
        width,
        height,
        roi_lookup=roi_lookup or {},
        stack=(),
    )


def roi_bounds(
    roi: ProjectRoi,
    *,
    roi_lookup: Mapping[str, ProjectRoi] | None = None,
) -> RoiBounds:
    return _roi_bounds(roi, roi_lookup=roi_lookup or {}, stack=())


def _rasterize_roi_mask(
    roi: ProjectRoi,
    width: int,
    height: int,
    *,
    roi_lookup: Mapping[str, ProjectRoi],
    stack: tuple[str, ...],
) -> RoiMask:
    if roi.id in stack:
        cycle = " -> ".join((*stack, roi.id))
        raise ValueError(f"ROI 布尔表达式存在循环引用: {cycle}")
    geometry = roi.geometry
    if isinstance(geometry, RectangleRoiGeometry):
        return _rasterize_rectangle(geometry, width, height)
    if isinstance(geometry, EllipseRoiGeometry):
        return _rasterize_ellipse(geometry, width, height)
    if isinstance(geometry, (PolygonRoiGeometry, FreehandRoiGeometry)):
        return _rasterize_rings(geometry.rings, width, height)

    operands = _resolve_operands(
        roi,
        geometry,
        roi_lookup=roi_lookup,
        stack=stack,
    )
    nested_stack = (*stack, roi.id)
    result = _rasterize_roi_mask(
        operands[0],
        width,
        height,
        roi_lookup=roi_lookup,
        stack=nested_stack,
    ).copy()
    for operand in operands[1:]:
        mask = _rasterize_roi_mask(
            operand,
            width,
            height,
            roi_lookup=roi_lookup,
            stack=nested_stack,
        )
        if geometry.operator is RoiBooleanOperator.UNION:
            np.logical_or(result, mask, out=result)
        elif geometry.operator is RoiBooleanOperator.INTERSECTION:
            np.logical_and(result, mask, out=result)
        elif geometry.operator is RoiBooleanOperator.DIFFERENCE:
            np.logical_and(result, np.logical_not(mask), out=result)
        else:
            np.logical_xor(result, mask, out=result)
    result.setflags(write=False)
    return result


def _roi_bounds(
    roi: ProjectRoi,
    *,
    roi_lookup: Mapping[str, ProjectRoi],
    stack: tuple[str, ...],
) -> RoiBounds:
    if roi.id in stack:
        cycle = " -> ".join((*stack, roi.id))
        raise ValueError(f"ROI 布尔表达式存在循环引用: {cycle}")
    geometry = roi.geometry
    if not isinstance(geometry, RoiBooleanExpression):
        return geometry.bounds
    operands = _resolve_operands(
        roi,
        geometry,
        roi_lookup=roi_lookup,
        stack=stack,
    )
    operand_bounds = [
        _roi_bounds(
            operand,
            roi_lookup=roi_lookup,
            stack=(*stack, roi.id),
        )
        for operand in operands
    ]
    if geometry.operator is RoiBooleanOperator.INTERSECTION:
        left = max(bounds[0] for bounds in operand_bounds)
        top = max(bounds[1] for bounds in operand_bounds)
        right = min(bounds[2] for bounds in operand_bounds)
        bottom = min(bounds[3] for bounds in operand_bounds)
        if right < left or bottom < top:
            return left, top, left, top
        return left, top, right, bottom
    if geometry.operator is RoiBooleanOperator.DIFFERENCE:
        return operand_bounds[0]
    return (
        min(bounds[0] for bounds in operand_bounds),
        min(bounds[1] for bounds in operand_bounds),
        max(bounds[2] for bounds in operand_bounds),
        max(bounds[3] for bounds in operand_bounds),
    )


def _resolve_operands(
    owner: ProjectRoi,
    expression: RoiBooleanExpression,
    *,
    roi_lookup: Mapping[str, ProjectRoi],
    stack: tuple[str, ...],
) -> tuple[ProjectRoi, ...]:
    operands: list[ProjectRoi] = []
    for operand_id in expression.operand_ids:
        operand = roi_lookup.get(operand_id)
        if operand is None:
            raise KeyError(f"ROI 布尔表达式引用了不存在的成员: {operand_id}")
        if operand.id != operand_id:
            raise ValueError(
                f"ROI 索引键 {operand_id} 与对象 ID {operand.id} 不一致"
            )
        if operand.document_id != owner.document_id:
            raise ValueError(
                f"ROI {owner.id} 不能引用其他文档的 ROI {operand.id}"
            )
        if operand.id in (*stack, owner.id):
            cycle = " -> ".join((*stack, owner.id, operand.id))
            raise ValueError(f"ROI 布尔表达式存在循环引用: {cycle}")
        operands.append(operand)
    return tuple(operands)


def _rasterize_rectangle(
    geometry: RectangleRoiGeometry,
    width: int,
    height: int,
) -> RoiMask:
    columns = np.arange(width, dtype=np.float64) + 0.5
    rows = np.arange(height, dtype=np.float64) + 0.5
    mask = (
        (rows[:, None] >= geometry.y)
        & (rows[:, None] < geometry.y + geometry.height)
        & (columns[None, :] >= geometry.x)
        & (columns[None, :] < geometry.x + geometry.width)
    )
    mask.setflags(write=False)
    return mask


def _rasterize_ellipse(
    geometry: EllipseRoiGeometry,
    width: int,
    height: int,
) -> RoiMask:
    center_x = geometry.x + (geometry.width / 2.0)
    center_y = geometry.y + (geometry.height / 2.0)
    radius_x = geometry.width / 2.0
    radius_y = geometry.height / 2.0
    columns = np.arange(width, dtype=np.float64) + 0.5
    rows = np.arange(height, dtype=np.float64) + 0.5
    normalized_x = (columns - center_x) / radius_x
    normalized_y = (rows - center_y) / radius_y
    mask = (
        (normalized_y[:, None] * normalized_y[:, None])
        + (normalized_x[None, :] * normalized_x[None, :])
        <= 1.0
    )
    mask.setflags(write=False)
    return mask


def _rasterize_rings(
    rings: tuple[tuple[RoiPoint, ...], ...],
    width: int,
    height: int,
) -> RoiMask:
    mask = np.zeros((height, width), dtype=np.bool_)
    left, top, right, bottom = _rings_bounds(rings)
    column_start = max(0, int(math.floor(left - 0.5)))
    column_stop = min(width, int(math.ceil(right - 0.5)) + 1)
    row_start = max(0, int(math.floor(top - 0.5)))
    row_stop = min(height, int(math.ceil(bottom - 0.5)) + 1)
    if column_start >= column_stop or row_start >= row_stop:
        mask.setflags(write=False)
        return mask

    columns = np.arange(column_start, column_stop, dtype=np.float64) + 0.5
    edges = [
        (previous, current)
        for ring in rings
        for previous, current in zip((ring[-1], *ring[:-1]), ring)
    ]
    x1 = np.asarray([edge[0].x for edge in edges], dtype=np.float64)
    y1 = np.asarray([edge[0].y for edge in edges], dtype=np.float64)
    x2 = np.asarray([edge[1].x for edge in edges], dtype=np.float64)
    y2 = np.asarray([edge[1].y for edge in edges], dtype=np.float64)
    dx = x2 - x1
    dy = y2 - y1
    epsilon = 1e-10
    for row_index in range(row_start, row_stop):
        y = row_index + 0.5
        on_boundary = np.zeros(columns.shape, dtype=np.bool_)
        crosses = (y1 > y) != (y2 > y)
        intersections = x1[crosses] + (
            ((y - y1[crosses]) * dx[crosses]) / dy[crosses]
        )
        intersections.sort()
        # Odd-even ray casting: a point is inside when the number of
        # intersections strictly to its right is odd.
        insertion_indices = np.searchsorted(
            intersections,
            columns,
            side="right",
        )
        inside = ((intersections.size - insertion_indices) % 2) == 1

        # Include pixel centres exactly on a boundary.  For non-horizontal
        # edges this can only be one centre per edge on the current scanline;
        # horizontal edges contribute one contiguous interval.
        horizontal = np.abs(dy) <= epsilon
        horizontal_here = horizontal & (np.abs(y1 - y) <= epsilon)
        for edge_index in np.flatnonzero(horizontal_here):
            on_boundary |= (
                (columns >= min(x1[edge_index], x2[edge_index]) - epsilon)
                & (columns <= max(x1[edge_index], x2[edge_index]) + epsilon)
            )
        non_horizontal_here = (
            ~horizontal
            & (y >= np.minimum(y1, y2) - epsilon)
            & (y <= np.maximum(y1, y2) + epsilon)
        )
        boundary_x = x1[non_horizontal_here] + (
            ((y - y1[non_horizontal_here]) * dx[non_horizontal_here])
            / dy[non_horizontal_here]
        )
        boundary_columns = np.rint(boundary_x - 0.5).astype(np.int64)
        exact_centres = np.abs(
            (boundary_columns.astype(np.float64) + 0.5) - boundary_x
        ) <= epsilon
        for absolute_column in boundary_columns[exact_centres]:
            if column_start <= absolute_column < column_stop:
                on_boundary[absolute_column - column_start] = True
        mask[row_index, column_start:column_stop] = inside | on_boundary
    mask.setflags(write=False)
    return mask


def _rings_bounds(rings: tuple[tuple[RoiPoint, ...], ...]) -> RoiBounds:
    points = [point for ring in rings for point in ring]
    return (
        min(point.x for point in points),
        min(point.y for point in points),
        max(point.x for point in points),
        max(point.y for point in points),
    )


def _require_mapping(
    payload: object,
    *,
    field_name: str,
) -> Mapping[str, object]:
    if not isinstance(payload, Mapping):
        raise TypeError(f"{field_name} 必须是对象")
    if any(not isinstance(key, str) for key in payload):
        raise TypeError(f"{field_name} 的键必须是字符串")
    return payload


def _require_exact_keys(
    payload: Mapping[str, object],
    *,
    required: set[str],
    optional: set[str] | None = None,
    field_name: str,
) -> None:
    actual = set(payload)
    missing = required - actual
    unknown = actual - required - (optional or set())
    if missing:
        raise ValueError(f"{field_name} 缺少字段: {', '.join(sorted(missing))}")
    if unknown:
        raise ValueError(f"{field_name} 包含未知字段: {', '.join(sorted(unknown))}")


def _finite_number(value: object, *, field_name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{field_name} 必须是数值")
    try:
        normalized = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{field_name} 必须是数值") from error
    if not math.isfinite(normalized):
        raise ValueError(f"{field_name} 必须是有限数")
    return normalized


def _positive_finite(value: object, *, field_name: str) -> float:
    normalized = _finite_number(value, field_name=field_name)
    if normalized <= 0.0:
        raise ValueError(f"{field_name} 必须大于 0")
    return normalized


def _required_text(
    value: object,
    *,
    field_name: str,
    maximum_length: int,
) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} 必须是字符串")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} 不能为空")
    if len(normalized) > maximum_length:
        raise ValueError(f"{field_name} 不能超过 {maximum_length} 个字符")
    if any(ord(character) < 32 or ord(character) == 127 for character in normalized):
        raise ValueError(f"{field_name} 不能包含控制字符")
    return normalized


def _required_id(value: object, *, field_name: str) -> str:
    normalized = _required_text(value, field_name=field_name, maximum_length=256)
    if not _ID_PATTERN.fullmatch(normalized):
        raise ValueError(f"{field_name} 包含无效字符")
    return normalized


def _non_negative_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} 必须是整数")
    if value < 0:
        raise ValueError(f"{field_name} 不能小于 0")
    return value


def _positive_int(value: object, *, field_name: str) -> int:
    normalized = _non_negative_int(value, field_name=field_name)
    if normalized <= 0:
        raise ValueError(f"{field_name} 必须大于 0")
    return normalized


__all__ = [
    "DEFAULT_ROI_COLOR",
    "EllipseRoiGeometry",
    "FreehandRoiGeometry",
    "PolygonRoiGeometry",
    "PROJECT_ROI_SCHEMA_VERSION",
    "ProjectRoi",
    "ProjectRoiGeometry",
    "ProjectRoiKind",
    "RectangleRoiGeometry",
    "RoiBooleanExpression",
    "RoiBooleanOperator",
    "RoiBounds",
    "RoiMask",
    "RoiPoint",
    "rasterize_roi_mask",
    "roi_bounds",
]
