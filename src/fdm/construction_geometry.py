"""Immutable construction-geometry domain model and analytic resolver.

This module deliberately has no dependency on :mod:`fdm.models`.  A project
document owns the entities, while this module owns their schema, dependency
rules and deterministic analytic geometry.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace
from enum import Enum
import math
from typing import Callable, ClassVar, Iterable, Mapping, Sequence, TypeAlias

from fdm.geometry import Point


CONSTRUCTION_SCHEMA_VERSION = 1
_EPSILON = 1e-9


class LineExtent(str, Enum):
    SEGMENT = "segment"
    RAY = "ray"
    INFINITE = "infinite"


class LineAxisConstraint(str, Enum):
    """Optional persistent orientation constraint for an explicitly axis line."""

    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"


class ArraySide(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    BOTH = "both"


class SourceObjectKind(str, Enum):
    CONSTRUCTION = "construction"
    MEASUREMENT = "measurement"


SourceObjectIdentity: TypeAlias = tuple[SourceObjectKind, str]


class CommonTangentMode(str, Enum):
    EXTERNAL = "external"
    INTERNAL = "internal"


class CircleTangency(str, Enum):
    """Oriented relation between a solution circle and a source circle."""

    EXTERNAL = "external"
    SOURCE_CONTAINS = "source_contains"
    SOLUTION_CONTAINS = "solution_contains"


class IntersectionBranchKind(str, Enum):
    LINE_CIRCLE = "line_circle"
    CIRCLE_CIRCLE = "circle_circle"


@dataclass(frozen=True, slots=True)
class IntersectionBranchHint:
    """Source-local identity for one root of a multi-intersection solution."""

    kind: IntersectionBranchKind
    radial: Point | None = None
    axis: Point | None = None
    side: int = 0

    def __post_init__(self) -> None:
        kind = IntersectionBranchKind(self.kind)
        object.__setattr__(self, "kind", kind)
        if self.radial is not None:
            object.__setattr__(self, "radial", _point(self.radial, "radial"))
        if self.axis is not None:
            object.__setattr__(self, "axis", _point(self.axis, "axis"))
        side = int(self.side)
        if side not in {-1, 0, 1}:
            raise ValueError("交点分支侧向必须为 -1、0 或 1")
        object.__setattr__(self, "side", side)
        if kind is IntersectionBranchKind.LINE_CIRCLE:
            if self.radial is None or self.axis is None:
                raise ValueError("线圆交点分支必须包含径向和线轴提示")
        elif side == 0:
            raise ValueError("双圆交点分支必须包含有效侧向")

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {"kind": self.kind.value}
        if self.radial is not None:
            payload["radial"] = _point_to_dict(self.radial)
        if self.axis is not None:
            payload["axis"] = _point_to_dict(self.axis)
        if self.side:
            payload["side"] = self.side
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "IntersectionBranchHint":
        return cls(
            kind=IntersectionBranchKind(str(payload["kind"])),
            radial=(
                _point_from_payload(payload, "radial")
                if isinstance(payload.get("radial"), Mapping)
                else None
            ),
            axis=(
                _point_from_payload(payload, "axis")
                if isinstance(payload.get("axis"), Mapping)
                else None
            ),
            side=int(payload.get("side", 0)),
        )


@dataclass(frozen=True, slots=True)
class TangencyConstraint:
    """Stable side/relation choice for one line-or-circle tangent source."""

    line_side: int = 1
    circle_relation: CircleTangency = CircleTangency.EXTERNAL

    def __post_init__(self) -> None:
        side = int(self.line_side)
        if side not in {-1, 1}:
            raise ValueError("切线侧向必须为 -1 或 1")
        object.__setattr__(self, "line_side", side)
        object.__setattr__(self, "circle_relation", CircleTangency(self.circle_relation))

    def to_dict(self) -> dict[str, object]:
        return {
            "line_side": self.line_side,
            "circle_relation": self.circle_relation.value,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "TangencyConstraint":
        return cls(
            line_side=int(payload.get("line_side", 1)),
            circle_relation=CircleTangency(
                str(payload.get("circle_relation", CircleTangency.EXTERNAL.value))
            ),
        )


@dataclass(frozen=True, slots=True)
class ConstructionStyle:
    stroke_color: str = "#29B6C8"
    stroke_width: float = 1.0
    dashed: bool = True
    opacity: float = 0.9

    def __post_init__(self) -> None:
        width = float(self.stroke_width)
        opacity = float(self.opacity)
        if not math.isfinite(width) or width <= 0.0:
            raise ValueError("辅助几何线宽必须是正有限数")
        if not math.isfinite(opacity) or not 0.0 <= opacity <= 1.0:
            raise ValueError("辅助几何透明度必须在 0 到 1 之间")
        object.__setattr__(self, "stroke_width", width)
        object.__setattr__(self, "opacity", opacity)

    def to_dict(self) -> dict[str, object]:
        return {
            "stroke_color": self.stroke_color,
            "stroke_width": self.stroke_width,
            "dashed": self.dashed,
            "opacity": self.opacity,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "ConstructionStyle":
        return cls(
            stroke_color=str(payload.get("stroke_color", "#29B6C8")),
            stroke_width=float(payload.get("stroke_width", 1.0)),
            dashed=bool(payload.get("dashed", True)),
            opacity=float(payload.get("opacity", 0.9)),
        )


@dataclass(frozen=True, slots=True)
class ResolvedPoint:
    point: Point

    def __post_init__(self) -> None:
        object.__setattr__(self, "point", _point(self.point, "point"))


@dataclass(frozen=True, slots=True)
class ResolvedLine:
    start: Point
    end: Point
    extent: LineExtent = LineExtent.SEGMENT

    def __post_init__(self) -> None:
        start = _point(self.start, "start")
        end = _point(self.end, "end")
        extent = LineExtent(self.extent)
        if _distance(start, end) <= _EPSILON:
            raise ValueError("线的两个定义点不能重合")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)
        object.__setattr__(self, "extent", extent)

    @property
    def direction(self) -> tuple[float, float]:
        length = self.length
        return (
            (self.end.x - self.start.x) / length,
            (self.end.y - self.start.y) / length,
        )

    @property
    def length(self) -> float:
        return _distance(self.start, self.end)

    def point_at(self, parameter: float) -> Point:
        dx, dy = self.direction
        return Point(self.start.x + dx * parameter, self.start.y + dy * parameter)

    def project_parameter(self, point: Point) -> float:
        dx, dy = self.direction
        return (point.x - self.start.x) * dx + (point.y - self.start.y) * dy

    def contains_parameter(self, parameter: float, *, epsilon: float = _EPSILON) -> bool:
        if self.extent is LineExtent.INFINITE:
            return True
        if self.extent is LineExtent.RAY:
            return parameter >= -epsilon
        return -epsilon <= parameter <= self.length + epsilon


@dataclass(frozen=True, slots=True)
class ResolvedCircle:
    center: Point
    radius: float

    def __post_init__(self) -> None:
        center = _point(self.center, "center")
        radius = float(self.radius)
        if not math.isfinite(radius) or radius <= _EPSILON:
            raise ValueError("圆半径必须是正有限数")
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "radius", radius)


@dataclass(frozen=True, slots=True)
class ParallelLineSequence(Sequence[ResolvedLine]):
    """Lazily expanded, stably indexed parallel-array children."""

    base_line: ResolvedLine
    spacing: float
    per_side_count: int
    side: ArraySide
    extent: LineExtent

    def __post_init__(self) -> None:
        spacing = float(self.spacing)
        count = int(self.per_side_count)
        if not math.isfinite(spacing) or spacing <= _EPSILON:
            raise ValueError("阵列间距必须大于 0")
        if count < 1:
            raise ValueError("阵列数量必须至少为 1")
        object.__setattr__(self, "spacing", spacing)
        object.__setattr__(self, "per_side_count", count)
        object.__setattr__(self, "side", ArraySide(self.side))
        object.__setattr__(self, "extent", LineExtent(self.extent))

    def __len__(self) -> int:
        return self.per_side_count * (2 if self.side is ArraySide.BOTH else 1)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return tuple(self[item] for item in range(*index.indices(len(self))))
        normalized = int(index)
        if normalized < 0:
            normalized += len(self)
        if normalized < 0 or normalized >= len(self):
            raise IndexError(index)
        return _offset_line(
            self.base_line,
            self.spacing * self.multiplier_at(normalized),
            self.extent,
        )

    def multiplier_at(self, index: int) -> int:
        if self.side is ArraySide.POSITIVE:
            return index + 1
        if self.side is ArraySide.NEGATIVE:
            return -(index + 1)
        magnitude = index // 2 + 1
        return -magnitude if index % 2 == 0 else magnitude

    def index_for_multiplier(self, multiplier: int) -> int | None:
        value = int(multiplier)
        if value == 0 or abs(value) > self.per_side_count:
            return None
        if self.side is ArraySide.POSITIVE:
            return value - 1 if value > 0 else None
        if self.side is ArraySide.NEGATIVE:
            return -value - 1 if value < 0 else None
        return 2 * (abs(value) - 1) + (1 if value > 0 else 0)

    def indices_for_offset_range(
        self,
        minimum_offset: float,
        maximum_offset: float,
    ) -> tuple[int, ...]:
        lower = math.ceil(min(minimum_offset, maximum_offset) / self.spacing - 1e-9)
        upper = math.floor(max(minimum_offset, maximum_offset) / self.spacing + 1e-9)
        indices: list[int] = []
        negative_start = max(lower, -self.per_side_count)
        negative_end = min(upper, -1)
        for multiplier in range(negative_start, negative_end + 1):
            index = self.index_for_multiplier(multiplier)
            if index is not None:
                indices.append(index)
        positive_start = max(lower, 1)
        positive_end = min(upper, self.per_side_count)
        for multiplier in range(positive_start, positive_end + 1):
            index = self.index_for_multiplier(multiplier)
            if index is not None:
                indices.append(index)
        indices.sort()
        return tuple(indices)

    def indexed_near_point(
        self,
        point: Point,
        radius: float,
    ) -> tuple[tuple[int, ResolvedLine], ...]:
        dx, dy = self.base_line.direction
        signed_offset = (
            (point.x - self.base_line.start.x) * (-dy)
            + (point.y - self.base_line.start.y) * dx
        )
        return tuple(
            (index, self[index])
            for index in self.indices_for_offset_range(
                signed_offset - radius,
                signed_offset + radius,
            )
        )

    def indexed_nearest(
        self,
        point: Point,
    ) -> tuple[tuple[int, ResolvedLine], ...]:
        dx, dy = self.base_line.direction
        signed_offset = (
            (point.x - self.base_line.start.x) * (-dy)
            + (point.y - self.base_line.start.y) * dx
        )
        ideal = int(round(signed_offset / self.spacing))
        multipliers = {
            ideal - 1,
            ideal,
            ideal + 1,
            -self.per_side_count,
            -1,
            1,
            self.per_side_count,
        }
        indices = sorted(
            index
            for multiplier in multipliers
            if (index := self.index_for_multiplier(multiplier)) is not None
        )
        return tuple((index, self[index]) for index in indices)

    def indexed_intersecting_rect(
        self,
        rect: tuple[float, float, float, float],
        *,
        padding: float = 0.0,
    ) -> tuple[tuple[int, ResolvedLine], ...]:
        left, top, right, bottom = rect
        dx, dy = self.base_line.direction
        offsets = tuple(
            (x - self.base_line.start.x) * (-dy)
            + (y - self.base_line.start.y) * dx
            for x, y in (
                (left, top),
                (right, top),
                (right, bottom),
                (left, bottom),
            )
        )
        return tuple(
            (index, self[index])
            for index in self.indices_for_offset_range(
                min(offsets) - padding,
                max(offsets) + padding,
            )
        )


@dataclass(frozen=True, slots=True)
class ResolvedLineArray:
    lines: tuple[ResolvedLine, ...] | ParallelLineSequence

    def __post_init__(self) -> None:
        if not isinstance(self.lines, ParallelLineSequence):
            object.__setattr__(self, "lines", tuple(self.lines))
        if not self.lines:
            raise ValueError("平行线阵列不能为空")


ResolvedGeometry: TypeAlias = ResolvedPoint | ResolvedLine | ResolvedCircle | ResolvedLineArray


@dataclass(frozen=True, slots=True)
class TangentCircleSolution:
    """One deterministically indexed circle tangent to selected sources."""

    branch: int
    circle: ResolvedCircle
    tangent_points: tuple[Point, ...]

    def __post_init__(self) -> None:
        branch = int(self.branch)
        if branch < 0:
            raise ValueError("相切圆分支不能为负数")
        object.__setattr__(self, "branch", branch)
        object.__setattr__(
            self,
            "tangent_points",
            tuple(_point(point, "tangent_point") for point in self.tangent_points),
        )


@dataclass(frozen=True, slots=True)
class ConstructionIssue:
    code: str
    message: str
    dependency_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ResolvedConstruction:
    entity_id: str
    geometry: ResolvedGeometry | None = None
    error: ConstructionIssue | None = None
    dependencies: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if (self.geometry is None) == (self.error is None):
            raise ValueError("解析结果必须且只能包含几何或错误")

    @property
    def valid(self) -> bool:
        return self.geometry is not None


@dataclass(frozen=True, slots=True)
class LiveFeatureRef:
    document_id: str
    object_id: str
    object_kind: SourceObjectKind = SourceObjectKind.CONSTRUCTION
    feature: str = "geometry"

    def __post_init__(self) -> None:
        if not str(self.document_id).strip():
            raise ValueError("实时特征引用缺少文档 ID")
        if not str(self.object_id).strip():
            raise ValueError("实时特征引用缺少对象 ID")
        object.__setattr__(self, "object_kind", SourceObjectKind(self.object_kind))

    @property
    def entity_id(self) -> str:
        return self.object_id

    def to_dict(self) -> dict[str, object]:
        return {
            "source_type": "live",
            "document_id": self.document_id,
            "object_id": self.object_id,
            "object_kind": self.object_kind.value,
            "feature": self.feature,
        }


@dataclass(frozen=True, slots=True)
class FrozenFeatureSnapshot:
    geometry: ResolvedGeometry

    def to_dict(self) -> dict[str, object]:
        return {"source_type": "frozen", "geometry": _resolved_geometry_to_dict(self.geometry)}


FeatureSource: TypeAlias = LiveFeatureRef | FrozenFeatureSnapshot


class _Definition:
    kind: ClassVar[str]


@dataclass(frozen=True, slots=True)
class FreePointDefinition(_Definition):
    kind: ClassVar[str] = "free_point"
    point: Point

    def __post_init__(self) -> None:
        object.__setattr__(self, "point", _point(self.point, "point"))


@dataclass(frozen=True, slots=True)
class LineDefinition(_Definition):
    kind: ClassVar[str] = "line"
    start: Point
    end: Point
    extent: LineExtent = LineExtent.SEGMENT
    axis_constraint: LineAxisConstraint | None = None

    def __post_init__(self) -> None:
        start = _point(self.start, "start")
        end = _point(self.end, "end")
        constraint = (
            None
            if self.axis_constraint is None
            else LineAxisConstraint(self.axis_constraint)
        )
        if constraint is LineAxisConstraint.HORIZONTAL:
            end = Point(end.x, start.y)
            if abs(end.x - start.x) <= _EPSILON:
                end = Point(start.x + 1.0, start.y)
        elif constraint is LineAxisConstraint.VERTICAL:
            end = Point(start.x, end.y)
            if abs(end.y - start.y) <= _EPSILON:
                end = Point(start.x, start.y + 1.0)
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)
        object.__setattr__(self, "extent", LineExtent(self.extent))
        object.__setattr__(self, "axis_constraint", constraint)


@dataclass(frozen=True, slots=True)
class CircleCenterRadiusDefinition(_Definition):
    kind: ClassVar[str] = "circle_center_radius"
    center: Point
    radius: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "center", _point(self.center, "center"))
        object.__setattr__(self, "radius", _finite(self.radius, "radius"))


@dataclass(frozen=True, slots=True)
class CircleCenterDiameterDefinition(_Definition):
    kind: ClassVar[str] = "circle_center_diameter"
    center: Point
    diameter: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "center", _point(self.center, "center"))
        object.__setattr__(self, "diameter", _finite(self.diameter, "diameter"))


@dataclass(frozen=True, slots=True)
class CircleTwoPointDefinition(_Definition):
    kind: ClassVar[str] = "circle_two_point"
    first: Point
    second: Point

    def __post_init__(self) -> None:
        object.__setattr__(self, "first", _point(self.first, "first"))
        object.__setattr__(self, "second", _point(self.second, "second"))


@dataclass(frozen=True, slots=True)
class CircleThreePointDefinition(_Definition):
    kind: ClassVar[str] = "circle_three_point"
    first: Point
    second: Point
    third: Point

    def __post_init__(self) -> None:
        object.__setattr__(self, "first", _point(self.first, "first"))
        object.__setattr__(self, "second", _point(self.second, "second"))
        object.__setattr__(self, "third", _point(self.third, "third"))


@dataclass(frozen=True, slots=True)
class MidpointDefinition(_Definition):
    kind: ClassVar[str] = "midpoint"
    source: FeatureSource


@dataclass(frozen=True, slots=True)
class IntersectionDefinition(_Definition):
    kind: ClassVar[str] = "intersection"
    first: FeatureSource
    second: FeatureSource
    branch: int = 0
    extend: bool = False
    # Vector from a source-local origin to the chosen solution.  New objects
    # persist it so reordering a line's endpoints cannot swap a two-root
    # line/circle intersection.  ``branch`` remains for schema-v1 payloads.
    branch_hint: IntersectionBranchHint | None = None

    def __post_init__(self) -> None:
        branch = int(self.branch)
        if branch < 0:
            raise ValueError("交点分支不能为负数")
        object.__setattr__(self, "branch", branch)
        if self.branch_hint is not None and not isinstance(
            self.branch_hint,
            IntersectionBranchHint,
        ):
            raise TypeError("交点分支提示类型无效")


@dataclass(frozen=True, slots=True)
class ParallelThroughPointDefinition(_Definition):
    kind: ClassVar[str] = "parallel_through_point"
    source: FeatureSource
    point: Point
    extent: LineExtent = LineExtent.INFINITE
    point_source: FeatureSource | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "point", _point(self.point, "point"))
        object.__setattr__(self, "extent", LineExtent(self.extent))
        if self.point_source is not None and not isinstance(
            self.point_source,
            (LiveFeatureRef, FrozenFeatureSnapshot),
        ):
            raise TypeError("过点来源必须是实时引用或冻结几何")


@dataclass(frozen=True, slots=True)
class OffsetParallelDefinition(_Definition):
    kind: ClassVar[str] = "offset_parallel"
    source: FeatureSource
    offset: float
    extent: LineExtent = LineExtent.INFINITE

    def __post_init__(self) -> None:
        object.__setattr__(self, "offset", _finite(self.offset, "offset"))
        object.__setattr__(self, "extent", LineExtent(self.extent))


@dataclass(frozen=True, slots=True)
class ParallelArrayDefinition(_Definition):
    kind: ClassVar[str] = "parallel_array"
    source: FeatureSource
    spacing: float
    count: int
    side: ArraySide = ArraySide.POSITIVE
    extent: LineExtent = LineExtent.INFINITE

    def __post_init__(self) -> None:
        spacing = _finite(self.spacing, "spacing")
        count = int(self.count)
        if spacing <= _EPSILON:
            raise ValueError("阵列间距必须大于 0")
        if count < 1:
            raise ValueError("阵列数量必须至少为 1")
        object.__setattr__(self, "spacing", spacing)
        object.__setattr__(self, "count", count)
        object.__setattr__(self, "side", ArraySide(self.side))
        object.__setattr__(self, "extent", LineExtent(self.extent))


@dataclass(frozen=True, slots=True)
class PerpendicularDefinition(_Definition):
    kind: ClassVar[str] = "perpendicular"
    source: FeatureSource
    point: Point
    extent: LineExtent = LineExtent.INFINITE
    point_source: FeatureSource | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "point", _point(self.point, "point"))
        object.__setattr__(self, "extent", LineExtent(self.extent))
        if self.point_source is not None and not isinstance(
            self.point_source,
            (LiveFeatureRef, FrozenFeatureSnapshot),
        ):
            raise TypeError("过点来源必须是实时引用或冻结几何")


@dataclass(frozen=True, slots=True)
class PerpendicularBisectorDefinition(_Definition):
    kind: ClassVar[str] = "perpendicular_bisector"
    source: FeatureSource
    extent: LineExtent = LineExtent.INFINITE

    def __post_init__(self) -> None:
        object.__setattr__(self, "extent", LineExtent(self.extent))


@dataclass(frozen=True, slots=True)
class ConcentricCircleDefinition(_Definition):
    kind: ClassVar[str] = "concentric_circle"
    source: FeatureSource
    radius: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "radius", _finite(self.radius, "radius"))


@dataclass(frozen=True, slots=True)
class OffsetCircleDefinition(_Definition):
    kind: ClassVar[str] = "offset_circle"
    source: FeatureSource
    offset: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "offset", _finite(self.offset, "offset"))


@dataclass(frozen=True, slots=True)
class PointCircleTangentDefinition(_Definition):
    kind: ClassVar[str] = "point_circle_tangent"
    point_source: FeatureSource
    circle_source: FeatureSource
    branch: int = 0
    extent: LineExtent = LineExtent.INFINITE

    def __post_init__(self) -> None:
        branch = int(self.branch)
        if branch < 0:
            raise ValueError("切线分支不能为负数")
        object.__setattr__(self, "branch", branch)
        object.__setattr__(self, "extent", LineExtent(self.extent))


@dataclass(frozen=True, slots=True)
class CommonTangentDefinition(_Definition):
    kind: ClassVar[str] = "common_tangent"
    first: FeatureSource
    second: FeatureSource
    mode: CommonTangentMode = CommonTangentMode.EXTERNAL
    branch: int = 0
    extent: LineExtent = LineExtent.INFINITE

    def __post_init__(self) -> None:
        branch = int(self.branch)
        if branch < 0:
            raise ValueError("公切线分支不能为负数")
        object.__setattr__(self, "mode", CommonTangentMode(self.mode))
        object.__setattr__(self, "branch", branch)
        object.__setattr__(self, "extent", LineExtent(self.extent))


@dataclass(frozen=True, slots=True)
class TangentTangentRadiusCircleDefinition(_Definition):
    kind: ClassVar[str] = "tangent_tangent_radius_circle"
    first: FeatureSource
    second: FeatureSource
    radius: float
    first_constraint: TangencyConstraint = field(default_factory=TangencyConstraint)
    second_constraint: TangencyConstraint = field(default_factory=TangencyConstraint)
    branch: int = 0
    extend: bool = False

    def __post_init__(self) -> None:
        branch = int(self.branch)
        if branch < 0:
            raise ValueError("相切圆分支不能为负数")
        object.__setattr__(self, "radius", _finite(self.radius, "radius"))
        object.__setattr__(self, "first_constraint", _constraint(self.first_constraint))
        object.__setattr__(self, "second_constraint", _constraint(self.second_constraint))
        object.__setattr__(self, "branch", branch)


@dataclass(frozen=True, slots=True)
class ThreeTangentCircleDefinition(_Definition):
    kind: ClassVar[str] = "three_tangent_circle"
    first: FeatureSource
    second: FeatureSource
    third: FeatureSource
    first_constraint: TangencyConstraint = field(default_factory=TangencyConstraint)
    second_constraint: TangencyConstraint = field(default_factory=TangencyConstraint)
    third_constraint: TangencyConstraint = field(default_factory=TangencyConstraint)
    branch: int = 0
    extend: bool = False

    def __post_init__(self) -> None:
        branch = int(self.branch)
        if branch < 0:
            raise ValueError("三相切圆分支不能为负数")
        object.__setattr__(self, "first_constraint", _constraint(self.first_constraint))
        object.__setattr__(self, "second_constraint", _constraint(self.second_constraint))
        object.__setattr__(self, "third_constraint", _constraint(self.third_constraint))
        object.__setattr__(self, "branch", branch)


ConstructionDefinition: TypeAlias = (
    FreePointDefinition
    | LineDefinition
    | CircleCenterRadiusDefinition
    | CircleCenterDiameterDefinition
    | CircleTwoPointDefinition
    | CircleThreePointDefinition
    | MidpointDefinition
    | IntersectionDefinition
    | ParallelThroughPointDefinition
    | OffsetParallelDefinition
    | ParallelArrayDefinition
    | PerpendicularDefinition
    | PerpendicularBisectorDefinition
    | ConcentricCircleDefinition
    | OffsetCircleDefinition
    | PointCircleTangentDefinition
    | CommonTangentDefinition
    | TangentTangentRadiusCircleDefinition
    | ThreeTangentCircleDefinition
)


@dataclass(frozen=True, slots=True)
class ConstructionEntity:
    id: str
    name: str
    definition: ConstructionDefinition
    visible: bool = True
    locked: bool = False
    snap_enabled: bool = True
    style: ConstructionStyle = field(default_factory=ConstructionStyle)
    revision: int = 0
    schema_version: int = CONSTRUCTION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not str(self.id).strip():
            raise ValueError("辅助几何对象缺少 ID")
        revision = int(self.revision)
        if revision < 0:
            raise ValueError("辅助几何修订号不能为负数")
        if int(self.schema_version) != CONSTRUCTION_SCHEMA_VERSION:
            raise ValueError(f"不支持的辅助几何 schema: {self.schema_version}")
        object.__setattr__(self, "revision", revision)
        object.__setattr__(self, "schema_version", CONSTRUCTION_SCHEMA_VERSION)

    @property
    def snappable(self) -> bool:
        return self.snap_enabled

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "id": self.id,
            "name": self.name,
            "definition": definition_to_dict(self.definition),
            "visible": self.visible,
            "locked": self.locked,
            "snap_enabled": self.snap_enabled,
            "style": self.style.to_dict(),
            "revision": self.revision,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "ConstructionEntity":
        schema_version = int(payload.get("schema_version", CONSTRUCTION_SCHEMA_VERSION))
        if schema_version != CONSTRUCTION_SCHEMA_VERSION:
            raise ValueError(f"不支持的辅助几何 schema: {schema_version}")
        raw_definition = payload.get("definition")
        if not isinstance(raw_definition, Mapping):
            raise ValueError("辅助几何对象缺少 definition")
        raw_style = payload.get("style", {})
        return cls(
            id=str(payload.get("id", "")),
            name=str(payload.get("name", "")),
            definition=definition_from_dict(raw_definition),
            visible=bool(payload.get("visible", True)),
            locked=bool(payload.get("locked", False)),
            snap_enabled=bool(payload.get("snap_enabled", payload.get("snappable", True))),
            style=ConstructionStyle.from_dict(raw_style if isinstance(raw_style, Mapping) else {}),
            revision=int(payload.get("revision", 0)),
            schema_version=schema_version,
        )


class ConstructionValidationError(ValueError):
    def __init__(self, code: str, message: str, entity_ids: Iterable[str] = ()) -> None:
        super().__init__(message)
        self.code = code
        self.entity_ids = tuple(entity_ids)


ExternalFeatureResolver: TypeAlias = Callable[
    [LiveFeatureRef], ResolvedConstruction | ResolvedGeometry | None
]


class ConstructionResolver:
    """Resolve construction definitions while preserving dependency failures."""

    def __init__(
        self,
        document_id: str,
        entities: Iterable[ConstructionEntity],
        external_feature_resolver: ExternalFeatureResolver | None = None,
    ) -> None:
        self.document_id = str(document_id)
        self.entities = {entity.id: entity for entity in entities}
        self.external_feature_resolver = external_feature_resolver
        self._cache: dict[str, ResolvedConstruction] = {}
        self._resolving: list[str] = []

    def resolve(self, entity: str | ConstructionEntity) -> ResolvedConstruction:
        entity_id = entity.id if isinstance(entity, ConstructionEntity) else str(entity)
        cached = self._cache.get(entity_id)
        if cached is not None:
            return cached
        target = self.entities.get(entity_id)
        if target is None:
            return _failure(entity_id, "missing_source", "找不到辅助几何对象", (entity_id,))
        if entity_id in self._resolving:
            cycle = tuple(self._resolving[self._resolving.index(entity_id) :] + [entity_id])
            return _failure(entity_id, "dependency_cycle", "辅助几何依赖形成环", cycle)

        resolution_order, cycle_placeholders = self._resolution_order(entity_id)
        try:
            for pending_id in resolution_order:
                pending = self.entities[pending_id]
                self._resolving.append(pending_id)
                try:
                    result = self._resolve_definition(pending)
                except (ValueError, TypeError) as exc:
                    result = _failure(pending_id, "degenerate_geometry", str(exc))
                finally:
                    self._resolving.pop()
                self._cache[pending_id] = result
        finally:
            # Back edges need a temporary cycle result so the first node that
            # unwinds observes the same invalid dependency as recursive DFS.
            # Every participating node is normally overwritten above; remove a
            # placeholder defensively if an unexpected exception interrupted it.
            for placeholder_id, placeholder in cycle_placeholders.items():
                if self._cache.get(placeholder_id) is placeholder:
                    self._cache.pop(placeholder_id, None)
        return self._cache.get(entity_id) or _failure(
            entity_id,
            "missing_source",
            "找不到辅助几何对象",
            (entity_id,),
        )

    def _resolution_order(
        self,
        root_id: str,
    ) -> tuple[list[str], dict[str, ResolvedConstruction]]:
        """Return dependency-first order without consuming the Python stack."""

        state: dict[str, int] = {}
        active: list[str] = []
        active_positions: dict[str, int] = {}
        frames: list[tuple[str, tuple[str, ...], int]] = []
        ordered: list[str] = []
        placeholders: dict[str, ResolvedConstruction] = {}

        def push(entity_id: str) -> bool:
            target = self.entities.get(entity_id)
            if target is None:
                return False
            dependencies = tuple(
                reference.object_id
                for reference in _definition_live_refs(target.definition)
                if reference.document_id == self.document_id
                and reference.object_kind is SourceObjectKind.CONSTRUCTION
            )
            state[entity_id] = 1
            active_positions[entity_id] = len(active)
            active.append(entity_id)
            frames.append((entity_id, dependencies, 0))
            return True

        push(root_id)
        while frames:
            entity_id, dependencies, next_index = frames[-1]
            if next_index < len(dependencies):
                dependency_id = dependencies[next_index]
                frames[-1] = (entity_id, dependencies, next_index + 1)
                cached = self._cache.get(dependency_id)
                if cached is not None and dependency_id not in placeholders:
                    continue
                dependency_state = state.get(dependency_id, 0)
                if dependency_state == 0:
                    push(dependency_id)
                    continue
                if dependency_state == 1:
                    cycle_start = active_positions[dependency_id]
                    cycle = tuple(active[cycle_start:] + [dependency_id])
                    if dependency_id not in placeholders:
                        placeholder = _failure(
                            dependency_id,
                            "dependency_cycle",
                            "辅助几何依赖形成环",
                            cycle,
                        )
                        placeholders[dependency_id] = placeholder
                        self._cache[dependency_id] = placeholder
                continue

            frames.pop()
            active.pop()
            active_positions.pop(entity_id, None)
            state[entity_id] = 2
            ordered.append(entity_id)
        return ordered, placeholders

    def resolve_all(self) -> dict[str, ResolvedConstruction]:
        return {entity_id: self.resolve(entity_id) for entity_id in self.entities}

    def resolve_feature(
        self,
        source: FeatureSource,
        *,
        owner_id: str = "",
    ) -> ResolvedGeometry | ConstructionIssue:
        """Resolve one feature source without exposing resolver internals."""

        return self._source_geometry(source, owner_id)

    def _resolve_definition(self, entity: ConstructionEntity) -> ResolvedConstruction:
        definition = entity.definition
        dependencies = _definition_dependency_ids(definition)
        if isinstance(definition, FreePointDefinition):
            geometry: ResolvedGeometry = ResolvedPoint(definition.point)
        elif isinstance(definition, LineDefinition):
            geometry = ResolvedLine(definition.start, definition.end, definition.extent)
        elif isinstance(definition, CircleCenterRadiusDefinition):
            geometry = ResolvedCircle(definition.center, definition.radius)
        elif isinstance(definition, CircleCenterDiameterDefinition):
            geometry = ResolvedCircle(definition.center, definition.diameter / 2.0)
        elif isinstance(definition, CircleTwoPointDefinition):
            geometry = ResolvedCircle(
                _midpoint(definition.first, definition.second),
                _distance(definition.first, definition.second) / 2.0,
            )
        elif isinstance(definition, CircleThreePointDefinition):
            center = _circumcenter(definition.first, definition.second, definition.third)
            geometry = ResolvedCircle(center, _distance(center, definition.first))
        elif isinstance(definition, MidpointDefinition):
            source = self._source_geometry(definition.source, entity.id)
            if isinstance(source, ConstructionIssue):
                return ResolvedConstruction(entity.id, error=source, dependencies=dependencies)
            line = _require_line(source)
            if line.extent is not LineExtent.SEGMENT:
                raise ValueError("只有有限线段具有中点")
            geometry = ResolvedPoint(_midpoint(line.start, line.end))
        elif isinstance(definition, IntersectionDefinition):
            first = self._source_geometry(definition.first, entity.id)
            if isinstance(first, ConstructionIssue):
                return ResolvedConstruction(entity.id, error=first, dependencies=dependencies)
            second = self._source_geometry(definition.second, entity.id)
            if isinstance(second, ConstructionIssue):
                return ResolvedConstruction(entity.id, error=second, dependencies=dependencies)
            candidates, issue = geometry_intersections(
                first,
                second,
                # A hinted branch must first be identified in the complete
                # analytical solution set.  Its domain is checked below so a
                # disappearing branch becomes unresolved instead of silently
                # jumping to the remaining root.
                extend=(True if definition.branch_hint is not None else definition.extend),
            )
            if issue is not None:
                return ResolvedConstruction(entity.id, error=issue, dependencies=dependencies)
            if definition.branch_hint is not None and candidates:
                selected = _select_intersection_by_hint(
                    first,
                    second,
                    candidates,
                    definition.branch_hint,
                )
                if selected is None:
                    return _failure(
                        entity.id,
                        "intersection_branch_hint_mismatch",
                        "交点分支提示与当前来源类型不匹配",
                        dependencies,
                    )
                if not definition.extend and not _intersection_point_in_domains(
                    first,
                    second,
                    selected,
                ):
                    return _failure(
                        entity.id,
                        "intersection_branch_missing",
                        "指定的交点分支当前不在线或射线的有效范围内",
                        dependencies,
                    )
                geometry = ResolvedPoint(selected)
            elif definition.branch >= len(candidates):
                return _failure(
                    entity.id,
                    "intersection_branch_missing",
                    "指定的交点分支当前不存在",
                    dependencies,
                )
            else:
                geometry = ResolvedPoint(candidates[definition.branch])
        elif isinstance(definition, ParallelThroughPointDefinition):
            source = self._source_geometry(definition.source, entity.id)
            if isinstance(source, ConstructionIssue):
                return ResolvedConstruction(entity.id, error=source, dependencies=dependencies)
            through_point = definition.point
            if definition.point_source is not None:
                point_source = self._source_geometry(
                    definition.point_source,
                    entity.id,
                )
                if isinstance(point_source, ConstructionIssue):
                    return ResolvedConstruction(
                        entity.id,
                        error=point_source,
                        dependencies=dependencies,
                    )
                through_point = _require_point(point_source).point
            geometry = _line_through(
                through_point,
                _require_line(source).direction,
                definition.extent,
            )
        elif isinstance(definition, OffsetParallelDefinition):
            source = self._source_geometry(definition.source, entity.id)
            if isinstance(source, ConstructionIssue):
                return ResolvedConstruction(entity.id, error=source, dependencies=dependencies)
            line = _require_line(source)
            geometry = _offset_line(line, definition.offset, definition.extent)
        elif isinstance(definition, ParallelArrayDefinition):
            source = self._source_geometry(definition.source, entity.id)
            if isinstance(source, ConstructionIssue):
                return ResolvedConstruction(entity.id, error=source, dependencies=dependencies)
            line = _require_line(source)
            geometry = ResolvedLineArray(
                ParallelLineSequence(
                    line,
                    definition.spacing,
                    definition.count,
                    definition.side,
                    definition.extent,
                )
            )
        elif isinstance(definition, PerpendicularDefinition):
            source = self._source_geometry(definition.source, entity.id)
            if isinstance(source, ConstructionIssue):
                return ResolvedConstruction(entity.id, error=source, dependencies=dependencies)
            through_point = definition.point
            if definition.point_source is not None:
                point_source = self._source_geometry(
                    definition.point_source,
                    entity.id,
                )
                if isinstance(point_source, ConstructionIssue):
                    return ResolvedConstruction(
                        entity.id,
                        error=point_source,
                        dependencies=dependencies,
                    )
                through_point = _require_point(point_source).point
            dx, dy = _require_line(source).direction
            geometry = _line_through(through_point, (-dy, dx), definition.extent)
        elif isinstance(definition, PerpendicularBisectorDefinition):
            source = self._source_geometry(definition.source, entity.id)
            if isinstance(source, ConstructionIssue):
                return ResolvedConstruction(entity.id, error=source, dependencies=dependencies)
            line = _require_line(source)
            if line.extent is not LineExtent.SEGMENT:
                raise ValueError("垂直平分线要求有限源线段")
            dx, dy = line.direction
            geometry = _line_through(_midpoint(line.start, line.end), (-dy, dx), definition.extent)
        elif isinstance(definition, ConcentricCircleDefinition):
            source = self._source_geometry(definition.source, entity.id)
            if isinstance(source, ConstructionIssue):
                return ResolvedConstruction(entity.id, error=source, dependencies=dependencies)
            geometry = ResolvedCircle(_require_circle(source).center, definition.radius)
        elif isinstance(definition, OffsetCircleDefinition):
            source = self._source_geometry(definition.source, entity.id)
            if isinstance(source, ConstructionIssue):
                return ResolvedConstruction(entity.id, error=source, dependencies=dependencies)
            circle = _require_circle(source)
            geometry = ResolvedCircle(circle.center, circle.radius + definition.offset)
        elif isinstance(definition, PointCircleTangentDefinition):
            point_source = self._source_geometry(definition.point_source, entity.id)
            if isinstance(point_source, ConstructionIssue):
                return ResolvedConstruction(entity.id, error=point_source, dependencies=dependencies)
            circle_source = self._source_geometry(definition.circle_source, entity.id)
            if isinstance(circle_source, ConstructionIssue):
                return ResolvedConstruction(entity.id, error=circle_source, dependencies=dependencies)
            lines, issue = point_circle_tangent_lines(
                _require_point(point_source).point,
                _require_circle(circle_source),
                extent=definition.extent,
            )
            if issue is not None:
                return ResolvedConstruction(entity.id, error=issue, dependencies=dependencies)
            if definition.branch >= len(lines):
                return _failure(
                    entity.id,
                    "tangent_branch_missing",
                    "指定的点圆切线分支当前不存在",
                    dependencies,
                )
            geometry = lines[definition.branch]
        elif isinstance(definition, CommonTangentDefinition):
            first = self._source_geometry(definition.first, entity.id)
            if isinstance(first, ConstructionIssue):
                return ResolvedConstruction(entity.id, error=first, dependencies=dependencies)
            second = self._source_geometry(definition.second, entity.id)
            if isinstance(second, ConstructionIssue):
                return ResolvedConstruction(entity.id, error=second, dependencies=dependencies)
            lines, issue = common_tangent_lines(
                _require_circle(first),
                _require_circle(second),
                definition.mode,
                extent=definition.extent,
            )
            if issue is not None:
                return ResolvedConstruction(entity.id, error=issue, dependencies=dependencies)
            if definition.branch >= len(lines):
                return _failure(
                    entity.id,
                    "tangent_branch_missing",
                    "指定的两圆公切线分支当前不存在",
                    dependencies,
                )
            geometry = lines[definition.branch]
        elif isinstance(definition, TangentTangentRadiusCircleDefinition):
            first = self._source_geometry(definition.first, entity.id)
            if isinstance(first, ConstructionIssue):
                return ResolvedConstruction(entity.id, error=first, dependencies=dependencies)
            second = self._source_geometry(definition.second, entity.id)
            if isinstance(second, ConstructionIssue):
                return ResolvedConstruction(entity.id, error=second, dependencies=dependencies)
            solutions, issue = tangent_tangent_radius_solutions(
                first,
                second,
                definition.radius,
                definition.first_constraint,
                definition.second_constraint,
                extend=definition.extend,
            )
            if issue is not None:
                return ResolvedConstruction(entity.id, error=issue, dependencies=dependencies)
            solution = next(
                (item for item in solutions if item.branch == definition.branch),
                None,
            )
            if solution is None:
                return _failure(
                    entity.id,
                    "tangent_branch_missing",
                    "指定的相切—相切—半径圆分支当前不存在",
                    dependencies,
                )
            geometry = solution.circle
        elif isinstance(definition, ThreeTangentCircleDefinition):
            first = self._source_geometry(definition.first, entity.id)
            if isinstance(first, ConstructionIssue):
                return ResolvedConstruction(entity.id, error=first, dependencies=dependencies)
            second = self._source_geometry(definition.second, entity.id)
            if isinstance(second, ConstructionIssue):
                return ResolvedConstruction(entity.id, error=second, dependencies=dependencies)
            third = self._source_geometry(definition.third, entity.id)
            if isinstance(third, ConstructionIssue):
                return ResolvedConstruction(entity.id, error=third, dependencies=dependencies)
            solutions, issue = three_tangent_circle_solutions(
                (first, second, third),
                (
                    definition.first_constraint,
                    definition.second_constraint,
                    definition.third_constraint,
                ),
                extend=definition.extend,
            )
            if issue is not None:
                return ResolvedConstruction(entity.id, error=issue, dependencies=dependencies)
            solution = next(
                (item for item in solutions if item.branch == definition.branch),
                None,
            )
            if solution is None:
                return _failure(
                    entity.id,
                    "tangent_branch_missing",
                    "指定的三相切圆分支当前不存在",
                    dependencies,
                )
            geometry = solution.circle
        else:  # pragma: no cover - guarded by serializer/type union
            return _failure(entity.id, "unsupported_definition", "不支持的辅助几何定义")
        return ResolvedConstruction(
            entity.id,
            geometry=geometry,
            dependencies=dependencies,
        )

    def _source_geometry(
        self,
        source: FeatureSource,
        owner_id: str,
    ) -> ResolvedGeometry | ConstructionIssue:
        if isinstance(source, FrozenFeatureSnapshot):
            return source.geometry
        if source.document_id != self.document_id:
            return ConstructionIssue(
                "cross_document_reference",
                "辅助几何不能实时引用其它文档",
                (source.object_id,),
            )
        if source.object_kind is SourceObjectKind.CONSTRUCTION:
            resolved = self.resolve(source.object_id)
            if not resolved.valid:
                assert resolved.error is not None
                return ConstructionIssue(
                    "unresolved_dependency",
                    f"依赖对象 {source.object_id} 当前不可解：{resolved.error.message}",
                    (source.object_id, *resolved.error.dependency_ids),
                )
            assert resolved.geometry is not None
            try:
                return select_feature(resolved.geometry, source.feature)
            except ValueError as exc:
                return ConstructionIssue("missing_feature", str(exc), (source.object_id,))
        if self.external_feature_resolver is None:
            return ConstructionIssue(
                "external_resolver_missing",
                "没有可用于解析测量对象的特征解析器",
                (source.object_id,),
            )
        resolved_external = self.external_feature_resolver(source)
        if resolved_external is None:
            return ConstructionIssue("missing_source", "找不到被引用的测量对象", (source.object_id,))
        if isinstance(resolved_external, ResolvedConstruction):
            if not resolved_external.valid:
                return resolved_external.error or ConstructionIssue("unresolved_dependency", "源对象不可解")
            assert resolved_external.geometry is not None
            geometry = resolved_external.geometry
        else:
            geometry = resolved_external
        try:
            return select_feature(geometry, source.feature)
        except ValueError as exc:
            return ConstructionIssue("missing_feature", str(exc), (source.object_id,))


def validate_construction_graph(
    document_id: str,
    entities: Iterable[ConstructionEntity],
) -> None:
    """Reject duplicate IDs, cross-document references, missing nodes and cycles."""

    sequence = tuple(entities)
    by_id: dict[str, ConstructionEntity] = {}
    for entity in sequence:
        if entity.id in by_id:
            raise ConstructionValidationError("duplicate_id", f"辅助几何 ID 重复：{entity.id}", (entity.id,))
        by_id[entity.id] = entity
    graph: dict[str, tuple[str, ...]] = {}
    for entity in sequence:
        refs = tuple(_definition_live_refs(entity.definition))
        for ref in refs:
            if ref.document_id != document_id:
                raise ConstructionValidationError(
                    "cross_document_reference",
                    f"对象 {entity.id} 包含跨文档实时引用",
                    (entity.id, ref.object_id),
                )
            if ref.object_kind is SourceObjectKind.CONSTRUCTION and ref.object_id not in by_id:
                raise ConstructionValidationError(
                    "missing_source",
                    f"对象 {entity.id} 引用了不存在的辅助对象 {ref.object_id}",
                    (entity.id, ref.object_id),
                )
        graph[entity.id] = tuple(
            ref.object_id
            for ref in refs
            if ref.object_kind is SourceObjectKind.CONSTRUCTION
        )
    state: dict[str, int] = {}
    active: list[str] = []
    active_positions: dict[str, int] = {}
    for root_id in graph:
        if state.get(root_id) == 2:
            continue
        state[root_id] = 1
        active_positions[root_id] = len(active)
        active.append(root_id)
        frames: list[tuple[str, int]] = [(root_id, 0)]
        while frames:
            entity_id, next_index = frames[-1]
            dependencies = graph[entity_id]
            if next_index < len(dependencies):
                dependency_id = dependencies[next_index]
                frames[-1] = (entity_id, next_index + 1)
                dependency_state = state.get(dependency_id, 0)
                if dependency_state == 2:
                    continue
                if dependency_state == 1:
                    cycle_start = active_positions[dependency_id]
                    cycle = tuple(active[cycle_start:] + [dependency_id])
                    raise ConstructionValidationError(
                        "dependency_cycle",
                        "辅助几何依赖形成环",
                        cycle,
                    )
                state[dependency_id] = 1
                active_positions[dependency_id] = len(active)
                active.append(dependency_id)
                frames.append((dependency_id, 0))
                continue
            frames.pop()
            active.pop()
            active_positions.pop(entity_id, None)
            state[entity_id] = 2


def iter_live_refs(definition: ConstructionDefinition) -> tuple[LiveFeatureRef, ...]:
    """Return the definition's live feature references in stable field order."""

    return tuple(_definition_live_refs(definition))


def live_dependency_identities(
    entity: ConstructionEntity,
) -> tuple[SourceObjectIdentity, ...]:
    """Return kind-qualified live dependencies in stable definition order."""

    return tuple(
        dict.fromkeys(
            (reference.object_kind, reference.object_id)
            for reference in iter_live_refs(entity.definition)
        )
    )


def live_dependency_ids(
    entity: ConstructionEntity,
    *,
    source_kind: SourceObjectKind,
) -> tuple[str, ...]:
    """Return unique dependency IDs for one explicitly selected object kind."""

    kind = SourceObjectKind(source_kind)
    return tuple(
        identity[1]
        for identity in live_dependency_identities(entity)
        if identity[0] is kind
    )


def transitive_dependents(
    entities: Iterable[ConstructionEntity],
    source_ids: Iterable[str],
    *,
    source_kind: SourceObjectKind,
) -> tuple[str, ...]:
    """Return all downstream construction IDs in breadth-first stable order."""

    sequence = tuple(entities)
    reverse: dict[SourceObjectIdentity, list[str]] = {}
    for entity in sequence:
        for identity in live_dependency_identities(entity):
            reverse.setdefault(identity, []).append(entity.id)
    kind = SourceObjectKind(source_kind)
    root_ids = tuple(dict.fromkeys(str(source_id).strip() for source_id in source_ids))
    if any(not source_id for source_id in root_ids):
        raise ValueError("source_ids 不能包含空 ID")
    roots = tuple((kind, source_id) for source_id in root_ids)
    seen = set(roots)
    queue = deque(roots)
    result: list[str] = []
    while queue:
        source_identity = queue.popleft()
        for dependent_id in reverse.get(source_identity, ()):
            dependent_identity = (SourceObjectKind.CONSTRUCTION, dependent_id)
            if dependent_identity in seen:
                continue
            seen.add(dependent_identity)
            result.append(dependent_id)
            queue.append(dependent_identity)
    return tuple(result)


def detach_sources(
    entities: Iterable[ConstructionEntity],
    removed_ids: Iterable[str],
    resolver: ConstructionResolver,
    *,
    source_kind: SourceObjectKind,
) -> tuple[ConstructionEntity, ...]:
    """Freeze references to removed objects without mutating revision counters.

    The resolver must describe the state *before* removal.  If a referenced
    feature is already invalid, detaching is refused rather than silently
    changing the construction's meaning.  Revision advancement belongs to the
    owning document so passing a returned entity to its replace operation does
    not double-increment the counter.
    """

    kind = SourceObjectKind(source_kind)
    removed = {
        (kind, str(entity_id).strip())
        for entity_id in removed_ids
    }
    if any(not entity_id for _kind, entity_id in removed):
        raise ValueError("removed_ids 不能包含空 ID")
    result: list[ConstructionEntity] = []
    for entity in entities:
        changed = False

        def freeze(source: FeatureSource) -> FeatureSource:
            nonlocal changed
            if (
                not isinstance(source, LiveFeatureRef)
                or (source.object_kind, source.object_id) not in removed
            ):
                return source
            resolved = resolver.resolve_feature(source, owner_id=entity.id)
            if isinstance(resolved, ConstructionIssue):
                raise ConstructionValidationError(
                    "detach_unresolved",
                    f"无法冻结对象 {entity.id} 的来源 {source.object_id}：{resolved.message}",
                    (entity.id, source.object_id),
                )
            changed = True
            return FrozenFeatureSnapshot(resolved)

        definition = _map_definition_sources(entity.definition, freeze)
        result.append(replace(entity, definition=definition) if changed else entity)
    return tuple(result)


def geometry_intersections(
    first: ResolvedGeometry,
    second: ResolvedGeometry,
    *,
    extend: bool = False,
) -> tuple[tuple[Point, ...], ConstructionIssue | None]:
    """Return deterministically ordered intersections for line/circle primitives."""

    if isinstance(first, ResolvedLineArray) or isinstance(second, ResolvedLineArray):
        return (), ConstructionIssue("unsupported_intersection", "阵列不能整体作为交点源，请选择其中一条线")
    if isinstance(first, ResolvedPoint) or isinstance(second, ResolvedPoint):
        return (), ConstructionIssue("unsupported_intersection", "交点命令要求线或圆")
    if isinstance(first, ResolvedLine) and isinstance(second, ResolvedLine):
        return _line_line_intersections(first, second, extend=extend)
    if isinstance(first, ResolvedLine) and isinstance(second, ResolvedCircle):
        return _line_circle_intersections(first, second, extend=extend)
    if isinstance(first, ResolvedCircle) and isinstance(second, ResolvedLine):
        points, issue = _line_circle_intersections(second, first, extend=extend)
        return points, issue
    if isinstance(first, ResolvedCircle) and isinstance(second, ResolvedCircle):
        return _circle_circle_intersections(first, second)
    return (), ConstructionIssue("unsupported_intersection", "不支持这两个对象求交")


def intersection_branch_hint(
    first: ResolvedGeometry,
    second: ResolvedGeometry,
    selected: Point,
) -> IntersectionBranchHint | None:
    """Encode a multi-root intersection in the sources' local frame."""

    point = _point(selected, "selected")
    line: ResolvedLine | None = None
    circle: ResolvedCircle | None = None
    if isinstance(first, ResolvedLine) and isinstance(second, ResolvedCircle):
        line, circle = first, second
    elif isinstance(first, ResolvedCircle) and isinstance(second, ResolvedLine):
        line, circle = second, first
    if line is not None and circle is not None:
        radial_x, radial_y = _normalize(
            (point.x - circle.center.x, point.y - circle.center.y)
        )
        axis_x, axis_y = line.direction
        return IntersectionBranchHint(
            IntersectionBranchKind.LINE_CIRCLE,
            radial=Point(radial_x, radial_y),
            axis=Point(axis_x, axis_y),
        )
    if isinstance(first, ResolvedCircle) and isinstance(second, ResolvedCircle):
        baseline = (
            second.center.x - first.center.x,
            second.center.y - first.center.y,
        )
        side_value = _cross(
            baseline,
            (point.x - first.center.x, point.y - first.center.y),
        )
        if abs(side_value) <= _EPSILON:
            return None
        return IntersectionBranchHint(
            IntersectionBranchKind.CIRCLE_CIRCLE,
            side=(-1 if side_value < 0.0 else 1),
        )
    return None


def point_circle_tangent_lines(
    point: Point,
    circle: ResolvedCircle,
    *,
    extent: LineExtent = LineExtent.INFINITE,
) -> tuple[tuple[ResolvedLine, ...], ConstructionIssue | None]:
    """Return point-to-circle tangents in stable signed-side order."""

    point = _point(point, "point")
    extent = LineExtent(extent)
    vx = point.x - circle.center.x
    vy = point.y - circle.center.y
    distance_sq = vx * vx + vy * vy
    radius_sq = circle.radius * circle.radius
    tolerance = _scaled_epsilon(distance_sq, radius_sq)
    if distance_sq < radius_sq - tolerance:
        return (), ConstructionIssue("no_tangent_solution", "点位于圆内，不能作实切线")
    if abs(distance_sq - radius_sq) <= tolerance:
        radial = _normalize((vx, vy))
        tangent = (-radial[1], radial[0])
        return (
            ResolvedLine(
                point,
                Point(point.x + tangent[0], point.y + tangent[1]),
                extent,
            ),
        ), None
    base_factor = radius_sq / distance_sq
    offset_factor = circle.radius * math.sqrt(max(0.0, distance_sq - radius_sq)) / distance_sq
    base = Point(
        circle.center.x + base_factor * vx,
        circle.center.y + base_factor * vy,
    )
    offset = (-vy * offset_factor, vx * offset_factor)
    tangent_points = (
        Point(base.x - offset[0], base.y - offset[1]),
        Point(base.x + offset[0], base.y + offset[1]),
    )
    return tuple(ResolvedLine(point, tangent, extent) for tangent in tangent_points), None


def common_tangent_lines(
    first: ResolvedCircle,
    second: ResolvedCircle,
    mode: CommonTangentMode = CommonTangentMode.EXTERNAL,
    *,
    extent: LineExtent = LineExtent.INFINITE,
) -> tuple[tuple[ResolvedLine, ...], ConstructionIssue | None]:
    """Return the two external or internal common tangents, branch-stably."""

    mode = CommonTangentMode(mode)
    extent = LineExtent(extent)
    vx = second.center.x - first.center.x
    vy = second.center.y - first.center.y
    distance_sq = vx * vx + vy * vy
    if distance_sq <= _EPSILON * _EPSILON:
        message = (
            "同心等半径圆有无穷多条外公切线"
            if mode is CommonTangentMode.EXTERNAL and abs(first.radius - second.radius) <= _EPSILON
            else "同心圆没有所选类型的唯一公切线"
        )
        return (), ConstructionIssue("coincident_tangent_locus", message)
    second_sign = 1.0 if mode is CommonTangentMode.EXTERNAL else -1.0
    normal_projection = first.radius - second_sign * second.radius
    feasibility = distance_sq - normal_projection * normal_projection
    tolerance = _scaled_epsilon(distance_sq, normal_projection * normal_projection)
    if feasibility < -tolerance:
        return (), ConstructionIssue("no_tangent_solution", "两个圆不存在所选类型的公切线")
    inverse_distance_sq = 1.0 / distance_sq
    base_x = vx * normal_projection * inverse_distance_sq
    base_y = vy * normal_projection * inverse_distance_sq
    height_factor = math.sqrt(max(0.0, feasibility)) * inverse_distance_sq
    offset_x = -vy * height_factor
    offset_y = vx * height_factor
    normals = (
        (base_x - offset_x, base_y - offset_y),
        (base_x + offset_x, base_y + offset_y),
    )
    if feasibility <= tolerance:
        normals = normals[:1]
    lines: list[ResolvedLine] = []
    for normal_x, normal_y in normals:
        first_point = Point(
            first.center.x + first.radius * normal_x,
            first.center.y + first.radius * normal_y,
        )
        second_point = Point(
            second.center.x + second_sign * second.radius * normal_x,
            second.center.y + second_sign * second.radius * normal_y,
        )
        if _distance(first_point, second_point) <= _EPSILON:
            direction = (-normal_y, normal_x)
            second_point = Point(first_point.x + direction[0], first_point.y + direction[1])
        lines.append(ResolvedLine(first_point, second_point, extent))
    return tuple(lines), None


def tangent_tangent_radius_solutions(
    first: ResolvedGeometry,
    second: ResolvedGeometry,
    radius: float,
    first_constraint: TangencyConstraint = TangencyConstraint(),
    second_constraint: TangencyConstraint = TangencyConstraint(),
    *,
    extend: bool = False,
) -> tuple[tuple[TangentCircleSolution, ...], ConstructionIssue | None]:
    """Solve a fixed-radius circle tangent to two selected line/circle sources."""

    radius = _finite(radius, "radius")
    if radius <= _EPSILON:
        return (), ConstructionIssue("degenerate_geometry", "相切圆半径必须大于 0")
    try:
        sources = (_require_tangent_source(first), _require_tangent_source(second))
        constraints = (_constraint(first_constraint), _constraint(second_constraint))
    except ValueError as exc:
        return (), ConstructionIssue("unsupported_tangent_source", str(exc))
    loci: list[ResolvedLine | ResolvedCircle] = []
    for source, constraint in zip(sources, constraints, strict=True):
        locus, issue = _fixed_radius_locus(source, constraint, radius)
        if issue is not None:
            return (), issue
        assert locus is not None
        loci.append(locus)
    centers, issue = _stable_tangent_locus_intersections(loci[0], loci[1])
    if issue is not None:
        code = (
            "coincident_tangent_locus"
            if issue.code == "coincident_geometry"
            else "no_tangent_solution"
        )
        return (), ConstructionIssue(code, issue.message)
    solutions: list[TangentCircleSolution] = []
    for branch, center in enumerate(centers):
        circle = ResolvedCircle(center, radius)
        tangent_points = _validated_tangent_points(
            circle,
            sources,
            constraints,
            extend=extend,
        )
        if tangent_points is None:
            continue
        solutions.append(TangentCircleSolution(branch, circle, tangent_points))
    if not solutions:
        return (), ConstructionIssue(
            "no_tangent_solution",
            "相切点不在所选线段或射线的有效范围内",
        )
    return tuple(solutions), None


def _stable_tangent_locus_intersections(
    first: ResolvedLine | ResolvedCircle,
    second: ResolvedLine | ResolvedCircle,
) -> tuple[tuple[Point, ...], ConstructionIssue | None]:
    """Order fixed-radius locus roots in their source-local frame."""

    centers, issue = geometry_intersections(first, second, extend=True)
    if issue is not None or len(centers) <= 1:
        return centers, issue
    line = (
        first
        if isinstance(first, ResolvedLine) and isinstance(second, ResolvedCircle)
        else (
            second
            if isinstance(second, ResolvedLine) and isinstance(first, ResolvedCircle)
            else None
        )
    )
    if line is not None:
        centers = tuple(sorted(centers, key=line.project_parameter))
    # Circle/circle ordering already uses the signed side of the ordered
    # first-centre -> second-centre baseline and is rigid-transform invariant.
    return centers, None


def three_tangent_circle_solutions(
    sources: tuple[ResolvedGeometry, ResolvedGeometry, ResolvedGeometry],
    constraints: tuple[TangencyConstraint, TangencyConstraint, TangencyConstraint] = (
        TangencyConstraint(),
        TangencyConstraint(),
        TangencyConstraint(),
    ),
    *,
    extend: bool = False,
) -> tuple[tuple[TangentCircleSolution, ...], ConstructionIssue | None]:
    """Solve the oriented Apollonius problem for any line/circle combination."""

    try:
        if len(sources) != 3 or len(constraints) != 3:
            raise ValueError("三相切圆必须恰好提供三个来源和三个约束")
        normalized_sources = tuple(_require_tangent_source(source) for source in sources)
        normalized_constraints = tuple(_constraint(value) for value in constraints)
    except ValueError as exc:
        return (), ConstructionIssue("unsupported_tangent_source", str(exc))
    circle_indices = [
        index
        for index, source in enumerate(normalized_sources)
        if isinstance(source, ResolvedCircle)
    ]
    raw_solutions: tuple[tuple[int, Point, float], ...]
    if not circle_indices:
        rows = tuple(
            _line_tangency_equation(source, constraint)
            for source, constraint in zip(
                normalized_sources,
                normalized_constraints,
                strict=True,
            )
            if isinstance(source, ResolvedLine)
        )
        solved = _solve_3x3(rows)
        if solved is None:
            return (), ConstructionIssue(
                "underdetermined_tangent_system",
                "三条线的相切约束无唯一解",
            )
        raw_solutions = ((0, Point(solved[0], solved[1]), solved[2]),)
    else:
        base_index = circle_indices[0]
        base = normalized_sources[base_index]
        assert isinstance(base, ResolvedCircle)
        base_constraint = normalized_constraints[base_index]
        rows: list[tuple[float, float, float, float]] = []
        for index, (source, constraint) in enumerate(
            zip(normalized_sources, normalized_constraints, strict=True)
        ):
            if index == base_index:
                continue
            if isinstance(source, ResolvedLine):
                rows.append(_line_tangency_equation(source, constraint))
            else:
                rows.append(
                    _circle_difference_equation(
                        base,
                        base_constraint,
                        source,
                        constraint,
                    )
                )
        affine = _two_plane_affine_line(rows[0], rows[1])
        if affine is None:
            return (), ConstructionIssue(
                "underdetermined_tangent_system",
                "三源相切约束相关或重合，没有唯一有限解",
            )
        origin, direction = affine
        roots = _apollonius_roots(origin, direction, base, base_constraint)
        if roots is None:
            return (), ConstructionIssue("no_tangent_solution", "三源相切约束没有实数解")
        raw_solutions = tuple(
            (
                branch,
                Point(
                    origin[0] + direction[0] * root,
                    origin[1] + direction[1] * root,
                ),
                origin[2] + direction[2] * root,
            )
            for branch, root in enumerate(roots)
        )
    stable_raw_solutions = tuple(
        sorted(
            raw_solutions,
            key=lambda item: _three_tangent_branch_sort_key(
                item[1],
                item[2],
                normalized_sources,
            ),
        )
    )
    analytic: list[tuple[int, ResolvedCircle, tuple[Point, ...]]] = []
    for stable_branch, (_raw_branch, center, radius) in enumerate(
        stable_raw_solutions
    ):
        if not math.isfinite(radius) or radius <= _EPSILON:
            continue
        circle = ResolvedCircle(center, radius)
        unconstrained_points = _validated_tangent_points(
            circle,
            normalized_sources,
            normalized_constraints,
            extend=True,
        )
        if unconstrained_points is None:
            continue
        if any(
            _circles_close(circle, existing_circle)
            for _branch, existing_circle, _points in analytic
        ):
            continue
        analytic.append((stable_branch, circle, unconstrained_points))
    solutions: list[TangentCircleSolution] = []
    for raw_branch, circle, unconstrained_points in analytic:
        tangent_points = (
            unconstrained_points
            if extend
            else _validated_tangent_points(
                circle,
                normalized_sources,
                normalized_constraints,
                extend=False,
            )
        )
        if tangent_points is not None:
            solutions.append(
                TangentCircleSolution(raw_branch, circle, tangent_points)
            )
    if not solutions:
        return (), ConstructionIssue(
            "no_tangent_solution",
            "三源相切约束没有满足方向和定义域的正半径解",
        )
    return tuple(solutions), None


def select_feature(geometry: ResolvedGeometry, feature: str = "geometry") -> ResolvedGeometry:
    """Select a stable feature from resolved geometry for a live reference."""

    token = str(feature or "geometry").strip().lower()
    if token == "geometry":
        return geometry
    if token.startswith("geometry:"):
        token = token.partition(":")[2]
    if isinstance(geometry, ResolvedLineArray):
        if token.startswith(("segment:", "edge:")):
            _prefix, _separator, remainder = token.partition(":")
            token = f"line:{remainder}"
        if token.startswith("vertex:"):
            index = int(token.partition(":")[2])
            if 0 <= index < len(geometry.lines):
                return ResolvedPoint(geometry.lines[index].start)
            if index == len(geometry.lines) and geometry.lines:
                return ResolvedPoint(geometry.lines[-1].end)
            raise ValueError(f"解析几何不包含特征 {feature!r}")
        if token.startswith("line:"):
            parts = token.split(":")
            selector = parts[1]
            if (
                isinstance(geometry.lines, ParallelLineSequence)
                and selector.startswith(("+", "-"))
            ):
                index = geometry.lines.index_for_multiplier(int(selector))
                if index is None:
                    raise ValueError(f"解析几何不包含特征 {feature!r}")
            else:
                # Schema-v1 compatibility for early array references that
                # encoded a positional child index.
                index = int(selector)
            if 0 <= index < len(geometry.lines):
                line = geometry.lines[index]
                return (
                    line
                    if len(parts) == 2
                    else select_feature(line, ":".join(parts[2:]))
                )
            raise ValueError(f"解析几何不包含特征 {feature!r}")
    if isinstance(geometry, ResolvedPoint):
        if token in {"point", "node"}:
            return geometry
    elif isinstance(geometry, ResolvedLine):
        if token.startswith("line:"):
            token = token.partition(":")[2]
        if token in {"start", "origin"} and geometry.extent is not LineExtent.INFINITE:
            return ResolvedPoint(geometry.start)
        if token == "end" and geometry.extent is LineExtent.SEGMENT:
            return ResolvedPoint(geometry.end)
        if token == "midpoint" and geometry.extent is LineExtent.SEGMENT:
            return ResolvedPoint(_midpoint(geometry.start, geometry.end))
    elif isinstance(geometry, ResolvedCircle):
        if token == "center":
            return ResolvedPoint(geometry.center)
        if token.startswith("quadrant:"):
            quadrant = int(token.partition(":")[2]) % 4
            angle = quadrant * math.pi / 2.0
            return ResolvedPoint(
                Point(
                    geometry.center.x + geometry.radius * math.cos(angle),
                    geometry.center.y + geometry.radius * math.sin(angle),
                )
            )
    raise ValueError(f"解析几何不包含特征 {feature!r}")


def definition_to_dict(definition: ConstructionDefinition) -> dict[str, object]:
    payload: dict[str, object] = {"kind": definition.kind}
    if isinstance(definition, FreePointDefinition):
        payload["point"] = _point_to_dict(definition.point)
    elif isinstance(definition, LineDefinition):
        payload.update(start=_point_to_dict(definition.start), end=_point_to_dict(definition.end), extent=definition.extent.value)
        if definition.axis_constraint is not None:
            payload["axis_constraint"] = definition.axis_constraint.value
    elif isinstance(definition, CircleCenterRadiusDefinition):
        payload.update(center=_point_to_dict(definition.center), radius=definition.radius)
    elif isinstance(definition, CircleCenterDiameterDefinition):
        payload.update(center=_point_to_dict(definition.center), diameter=definition.diameter)
    elif isinstance(definition, CircleTwoPointDefinition):
        payload.update(first=_point_to_dict(definition.first), second=_point_to_dict(definition.second))
    elif isinstance(definition, CircleThreePointDefinition):
        payload.update(first=_point_to_dict(definition.first), second=_point_to_dict(definition.second), third=_point_to_dict(definition.third))
    elif isinstance(definition, MidpointDefinition):
        payload["source"] = _source_to_dict(definition.source)
    elif isinstance(definition, IntersectionDefinition):
        payload.update(first=_source_to_dict(definition.first), second=_source_to_dict(definition.second), branch=definition.branch, extend=definition.extend)
        if definition.branch_hint is not None:
            payload["branch_hint"] = definition.branch_hint.to_dict()
    elif isinstance(definition, ParallelThroughPointDefinition):
        payload.update(source=_source_to_dict(definition.source), point=_point_to_dict(definition.point), extent=definition.extent.value)
        if definition.point_source is not None:
            payload["point_source"] = _source_to_dict(definition.point_source)
    elif isinstance(definition, OffsetParallelDefinition):
        payload.update(source=_source_to_dict(definition.source), offset=definition.offset, extent=definition.extent.value)
    elif isinstance(definition, ParallelArrayDefinition):
        payload.update(source=_source_to_dict(definition.source), spacing=definition.spacing, count=definition.count, side=definition.side.value, extent=definition.extent.value)
    elif isinstance(definition, PerpendicularDefinition):
        payload.update(source=_source_to_dict(definition.source), point=_point_to_dict(definition.point), extent=definition.extent.value)
        if definition.point_source is not None:
            payload["point_source"] = _source_to_dict(definition.point_source)
    elif isinstance(definition, PerpendicularBisectorDefinition):
        payload.update(source=_source_to_dict(definition.source), extent=definition.extent.value)
    elif isinstance(definition, ConcentricCircleDefinition):
        payload.update(source=_source_to_dict(definition.source), radius=definition.radius)
    elif isinstance(definition, OffsetCircleDefinition):
        payload.update(source=_source_to_dict(definition.source), offset=definition.offset)
    elif isinstance(definition, PointCircleTangentDefinition):
        payload.update(
            point_source=_source_to_dict(definition.point_source),
            circle_source=_source_to_dict(definition.circle_source),
            branch=definition.branch,
            extent=definition.extent.value,
        )
    elif isinstance(definition, CommonTangentDefinition):
        payload.update(
            first=_source_to_dict(definition.first),
            second=_source_to_dict(definition.second),
            mode=definition.mode.value,
            branch=definition.branch,
            extent=definition.extent.value,
        )
    elif isinstance(definition, TangentTangentRadiusCircleDefinition):
        payload.update(
            first=_source_to_dict(definition.first),
            second=_source_to_dict(definition.second),
            radius=definition.radius,
            first_constraint=definition.first_constraint.to_dict(),
            second_constraint=definition.second_constraint.to_dict(),
            branch=definition.branch,
            extend=definition.extend,
        )
    elif isinstance(definition, ThreeTangentCircleDefinition):
        payload.update(
            first=_source_to_dict(definition.first),
            second=_source_to_dict(definition.second),
            third=_source_to_dict(definition.third),
            first_constraint=definition.first_constraint.to_dict(),
            second_constraint=definition.second_constraint.to_dict(),
            third_constraint=definition.third_constraint.to_dict(),
            branch=definition.branch,
            extend=definition.extend,
        )
    else:  # pragma: no cover
        raise TypeError(f"不支持的辅助几何定义：{type(definition).__name__}")
    return payload


def definition_from_dict(payload: Mapping[str, object]) -> ConstructionDefinition:
    kind = str(payload.get("kind", ""))
    if kind == FreePointDefinition.kind:
        return FreePointDefinition(_point_from_payload(payload, "point"))
    if kind == LineDefinition.kind:
        axis_constraint = payload.get("axis_constraint")
        return LineDefinition(
            _point_from_payload(payload, "start"),
            _point_from_payload(payload, "end"),
            LineExtent(str(payload.get("extent", "segment"))),
            (
                LineAxisConstraint(str(axis_constraint))
                if axis_constraint is not None
                else None
            ),
        )
    if kind == CircleCenterRadiusDefinition.kind:
        return CircleCenterRadiusDefinition(_point_from_payload(payload, "center"), float(payload["radius"]))
    if kind == CircleCenterDiameterDefinition.kind:
        return CircleCenterDiameterDefinition(_point_from_payload(payload, "center"), float(payload["diameter"]))
    if kind == CircleTwoPointDefinition.kind:
        return CircleTwoPointDefinition(_point_from_payload(payload, "first"), _point_from_payload(payload, "second"))
    if kind == CircleThreePointDefinition.kind:
        return CircleThreePointDefinition(_point_from_payload(payload, "first"), _point_from_payload(payload, "second"), _point_from_payload(payload, "third"))
    if kind == MidpointDefinition.kind:
        return MidpointDefinition(_source_from_payload(payload, "source"))
    if kind == IntersectionDefinition.kind:
        return IntersectionDefinition(
            _source_from_payload(payload, "first"),
            _source_from_payload(payload, "second"),
            int(payload.get("branch", 0)),
            bool(payload.get("extend", False)),
            (
                IntersectionBranchHint.from_dict(payload["branch_hint"])
                if isinstance(payload.get("branch_hint"), Mapping)
                else None
            ),
        )
    if kind == ParallelThroughPointDefinition.kind:
        return ParallelThroughPointDefinition(
            _source_from_payload(payload, "source"),
            _point_from_payload(payload, "point"),
            LineExtent(str(payload.get("extent", "infinite"))),
            (
                _source_from_payload(payload, "point_source")
                if isinstance(payload.get("point_source"), Mapping)
                else None
            ),
        )
    if kind == OffsetParallelDefinition.kind:
        return OffsetParallelDefinition(_source_from_payload(payload, "source"), float(payload["offset"]), LineExtent(str(payload.get("extent", "infinite"))))
    if kind == ParallelArrayDefinition.kind:
        return ParallelArrayDefinition(_source_from_payload(payload, "source"), float(payload["spacing"]), int(payload["count"]), ArraySide(str(payload.get("side", "positive"))), LineExtent(str(payload.get("extent", "infinite"))))
    if kind == PerpendicularDefinition.kind:
        return PerpendicularDefinition(
            _source_from_payload(payload, "source"),
            _point_from_payload(payload, "point"),
            LineExtent(str(payload.get("extent", "infinite"))),
            (
                _source_from_payload(payload, "point_source")
                if isinstance(payload.get("point_source"), Mapping)
                else None
            ),
        )
    if kind == PerpendicularBisectorDefinition.kind:
        return PerpendicularBisectorDefinition(_source_from_payload(payload, "source"), LineExtent(str(payload.get("extent", "infinite"))))
    if kind == ConcentricCircleDefinition.kind:
        return ConcentricCircleDefinition(_source_from_payload(payload, "source"), float(payload["radius"]))
    if kind == OffsetCircleDefinition.kind:
        return OffsetCircleDefinition(_source_from_payload(payload, "source"), float(payload["offset"]))
    if kind == PointCircleTangentDefinition.kind:
        return PointCircleTangentDefinition(
            _source_from_payload(payload, "point_source"),
            _source_from_payload(payload, "circle_source"),
            int(payload.get("branch", 0)),
            LineExtent(str(payload.get("extent", "infinite"))),
        )
    if kind == CommonTangentDefinition.kind:
        return CommonTangentDefinition(
            _source_from_payload(payload, "first"),
            _source_from_payload(payload, "second"),
            CommonTangentMode(str(payload.get("mode", "external"))),
            int(payload.get("branch", 0)),
            LineExtent(str(payload.get("extent", "infinite"))),
        )
    if kind == TangentTangentRadiusCircleDefinition.kind:
        return TangentTangentRadiusCircleDefinition(
            _source_from_payload(payload, "first"),
            _source_from_payload(payload, "second"),
            float(payload["radius"]),
            _constraint_from_payload(payload, "first_constraint"),
            _constraint_from_payload(payload, "second_constraint"),
            int(payload.get("branch", 0)),
            bool(payload.get("extend", False)),
        )
    if kind == ThreeTangentCircleDefinition.kind:
        return ThreeTangentCircleDefinition(
            _source_from_payload(payload, "first"),
            _source_from_payload(payload, "second"),
            _source_from_payload(payload, "third"),
            _constraint_from_payload(payload, "first_constraint"),
            _constraint_from_payload(payload, "second_constraint"),
            _constraint_from_payload(payload, "third_constraint"),
            int(payload.get("branch", 0)),
            bool(payload.get("extend", False)),
        )
    raise ValueError(f"未知的辅助几何定义类型：{kind!r}")


def _source_to_dict(source: FeatureSource) -> dict[str, object]:
    return source.to_dict()


def _constraint(value: object) -> TangencyConstraint:
    if isinstance(value, TangencyConstraint):
        return value
    if isinstance(value, Mapping):
        return TangencyConstraint.from_dict(value)
    raise ValueError("相切约束格式无效")


def _constraint_from_payload(
    payload: Mapping[str, object],
    key: str,
) -> TangencyConstraint:
    value = payload.get(key, {})
    if not isinstance(value, Mapping):
        raise ValueError(f"相切定义缺少约束 {key}")
    return TangencyConstraint.from_dict(value)


def _source_from_payload(payload: Mapping[str, object], key: str) -> FeatureSource:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"辅助几何定义缺少来源 {key}")
    source_type = str(value.get("source_type", ""))
    if source_type == "live":
        return LiveFeatureRef(
            document_id=str(value.get("document_id", "")),
            object_id=str(value.get("object_id", "")),
            object_kind=SourceObjectKind(str(value.get("object_kind", "construction"))),
            feature=str(value.get("feature", "geometry")),
        )
    if source_type == "frozen":
        geometry = value.get("geometry")
        if not isinstance(geometry, Mapping):
            raise ValueError("冻结特征缺少 geometry")
        return FrozenFeatureSnapshot(_resolved_geometry_from_dict(geometry))
    raise ValueError(f"未知的特征来源类型：{source_type!r}")


def _resolved_geometry_to_dict(geometry: ResolvedGeometry) -> dict[str, object]:
    if isinstance(geometry, ResolvedPoint):
        return {"kind": "point", "point": _point_to_dict(geometry.point)}
    if isinstance(geometry, ResolvedLine):
        return {"kind": "line", "start": _point_to_dict(geometry.start), "end": _point_to_dict(geometry.end), "extent": geometry.extent.value}
    if isinstance(geometry, ResolvedCircle):
        return {"kind": "circle", "center": _point_to_dict(geometry.center), "radius": geometry.radius}
    if isinstance(geometry, ResolvedLineArray):
        return {"kind": "line_array", "lines": [_resolved_geometry_to_dict(line) for line in geometry.lines]}
    raise TypeError(f"不支持的解析几何：{type(geometry).__name__}")


def _resolved_geometry_from_dict(payload: Mapping[str, object]) -> ResolvedGeometry:
    kind = str(payload.get("kind", ""))
    if kind == "point":
        return ResolvedPoint(_point_from_payload(payload, "point"))
    if kind == "line":
        return ResolvedLine(_point_from_payload(payload, "start"), _point_from_payload(payload, "end"), LineExtent(str(payload.get("extent", "segment"))))
    if kind == "circle":
        return ResolvedCircle(_point_from_payload(payload, "center"), float(payload["radius"]))
    if kind == "line_array":
        raw_lines = payload.get("lines")
        if not isinstance(raw_lines, list):
            raise ValueError("冻结阵列缺少 lines")
        lines = tuple(_resolved_geometry_from_dict(line) for line in raw_lines if isinstance(line, Mapping))
        if not all(isinstance(line, ResolvedLine) for line in lines):
            raise ValueError("冻结阵列只能包含直线")
        return ResolvedLineArray(lines)  # type: ignore[arg-type]
    raise ValueError(f"未知的解析几何类型：{kind!r}")


def _definition_sources(definition: ConstructionDefinition) -> tuple[FeatureSource, ...]:
    sources: tuple[FeatureSource, ...]
    if isinstance(
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
            IntersectionDefinition,
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
    return sources


def _definition_live_refs(definition: ConstructionDefinition) -> Iterable[LiveFeatureRef]:
    return (
        source
        for source in _definition_sources(definition)
        if isinstance(source, LiveFeatureRef)
    )


def _definition_dependency_ids(definition: ConstructionDefinition) -> tuple[str, ...]:
    return tuple(ref.object_id for ref in _definition_live_refs(definition))


def _map_definition_sources(
    definition: ConstructionDefinition,
    mapper: Callable[[FeatureSource], FeatureSource],
) -> ConstructionDefinition:
    if isinstance(
        definition,
        (ParallelThroughPointDefinition, PerpendicularDefinition),
    ):
        return replace(
            definition,
            source=mapper(definition.source),
            point_source=(
                mapper(definition.point_source)
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
        return replace(definition, source=mapper(definition.source))
    if isinstance(
        definition,
        (
            IntersectionDefinition,
            CommonTangentDefinition,
            TangentTangentRadiusCircleDefinition,
        ),
    ):
        return replace(
            definition,
            first=mapper(definition.first),
            second=mapper(definition.second),
        )
    if isinstance(definition, PointCircleTangentDefinition):
        return replace(
            definition,
            point_source=mapper(definition.point_source),
            circle_source=mapper(definition.circle_source),
        )
    if isinstance(definition, ThreeTangentCircleDefinition):
        return replace(
            definition,
            first=mapper(definition.first),
            second=mapper(definition.second),
            third=mapper(definition.third),
        )
    return definition


def _line_line_intersections(first: ResolvedLine, second: ResolvedLine, *, extend: bool) -> tuple[tuple[Point, ...], ConstructionIssue | None]:
    rx, ry = first.direction
    sx, sy = second.direction
    denominator = _cross((rx, ry), (sx, sy))
    delta = (second.start.x - first.start.x, second.start.y - first.start.y)
    if abs(denominator) <= _EPSILON:
        if abs(_cross(delta, (rx, ry))) <= _EPSILON:
            return (), ConstructionIssue("coincident_geometry", "两条线重合，没有唯一交点")
        return (), ConstructionIssue("no_intersection", "两条线平行")
    t = _cross(delta, (sx, sy)) / denominator
    u = _cross(delta, (rx, ry)) / denominator
    if not extend and (not first.contains_parameter(t) or not second.contains_parameter(u)):
        return (), ConstructionIssue("no_intersection", "交点不在线或射线的有效范围内")
    return (first.point_at(t),), None


def _line_circle_intersections(line: ResolvedLine, circle: ResolvedCircle, *, extend: bool) -> tuple[tuple[Point, ...], ConstructionIssue | None]:
    t_center = line.project_parameter(circle.center)
    closest = line.point_at(t_center)
    distance_sq = (closest.x - circle.center.x) ** 2 + (closest.y - circle.center.y) ** 2
    discriminant = circle.radius * circle.radius - distance_sq
    if discriminant < -_EPSILON:
        return (), ConstructionIssue("no_intersection", "直线与圆不相交")
    half = math.sqrt(max(0.0, discriminant))
    parameters = (t_center,) if half <= _EPSILON else (t_center - half, t_center + half)
    points = tuple(
        sorted(
            (
                line.point_at(parameter)
                for parameter in parameters
                if extend or line.contains_parameter(parameter)
            ),
            key=lambda point: (point.x, point.y),
        )
    )
    if not points:
        return (), ConstructionIssue("no_intersection", "交点不在线或射线的有效范围内")
    return points, None


def _select_intersection_by_hint(
    first: ResolvedGeometry,
    second: ResolvedGeometry,
    candidates: Sequence[Point],
    hint: IntersectionBranchHint,
) -> Point | None:
    if not candidates:
        return None
    line: ResolvedLine | None = None
    circle: ResolvedCircle | None = None
    if isinstance(first, ResolvedLine) and isinstance(second, ResolvedCircle):
        line, circle = first, second
    elif isinstance(first, ResolvedCircle) and isinstance(second, ResolvedLine):
        line, circle = second, first
    if (
        hint.kind is IntersectionBranchKind.LINE_CIRCLE
        and line is not None
        and circle is not None
        and hint.axis is not None
        and hint.radial is not None
    ):
        old_axis = _normalize((hint.axis.x, hint.axis.y))
        current_axis = line.direction
        axis_dot = old_axis[0] * current_axis[0] + old_axis[1] * current_axis[1]
        if axis_dot < 0.0:
            current_axis = (-current_axis[0], -current_axis[1])
            axis_dot = -axis_dot
        axis_cross = _cross(old_axis, current_axis)
        radial = _normalize((hint.radial.x, hint.radial.y))
        rotated_radial = (
            radial[0] * axis_dot - radial[1] * axis_cross,
            radial[0] * axis_cross + radial[1] * axis_dot,
        )
        return max(
            candidates,
            key=lambda candidate: (
                (candidate.x - circle.center.x) * rotated_radial[0]
                + (candidate.y - circle.center.y) * rotated_radial[1]
            ),
        )
    if (
        hint.kind is IntersectionBranchKind.CIRCLE_CIRCLE
        and isinstance(first, ResolvedCircle)
        and isinstance(second, ResolvedCircle)
        and hint.side in {-1, 1}
    ):
        baseline = (
            second.center.x - first.center.x,
            second.center.y - first.center.y,
        )
        return max(
            candidates,
            key=lambda candidate: hint.side
            * _cross(
                baseline,
                (
                    candidate.x - first.center.x,
                    candidate.y - first.center.y,
                ),
            ),
        )
    return None


def _intersection_point_in_domains(
    first: ResolvedGeometry,
    second: ResolvedGeometry,
    point: Point,
) -> bool:
    for geometry in (first, second):
        if not isinstance(geometry, ResolvedLine):
            continue
        parameter = geometry.project_parameter(point)
        projected = geometry.point_at(parameter)
        if _distance(projected, point) > 1e-6 or not geometry.contains_parameter(parameter):
            return False
    return True


def _circle_circle_intersections(first: ResolvedCircle, second: ResolvedCircle) -> tuple[tuple[Point, ...], ConstructionIssue | None]:
    dx = second.center.x - first.center.x
    dy = second.center.y - first.center.y
    center_distance = math.hypot(dx, dy)
    if center_distance <= _EPSILON and abs(first.radius - second.radius) <= _EPSILON:
        return (), ConstructionIssue("coincident_geometry", "两个圆重合，没有唯一交点")
    if center_distance > first.radius + second.radius + _EPSILON or center_distance < abs(first.radius - second.radius) - _EPSILON or center_distance <= _EPSILON:
        return (), ConstructionIssue("no_intersection", "两个圆不相交")
    along = (first.radius**2 - second.radius**2 + center_distance**2) / (2.0 * center_distance)
    height_sq = first.radius**2 - along**2
    if height_sq < -_EPSILON:
        return (), ConstructionIssue("no_intersection", "两个圆不相交")
    base_x = first.center.x + along * dx / center_distance
    base_y = first.center.y + along * dy / center_distance
    height = math.sqrt(max(0.0, height_sq))
    if height <= _EPSILON:
        return (Point(base_x, base_y),), None
    # Branch 0/1 are the negative/positive signed sides of first -> second.
    offset_x = -dy * height / center_distance
    offset_y = dx * height / center_distance
    return (Point(base_x - offset_x, base_y - offset_y), Point(base_x + offset_x, base_y + offset_y)), None


def _require_line(geometry: ResolvedGeometry) -> ResolvedLine:
    if not isinstance(geometry, ResolvedLine):
        raise ValueError("该构造命令要求线对象或线特征")
    return geometry


def _require_point(geometry: ResolvedGeometry) -> ResolvedPoint:
    if not isinstance(geometry, ResolvedPoint):
        raise ValueError("该构造命令要求点对象或点特征")
    return geometry


def _require_circle(geometry: ResolvedGeometry) -> ResolvedCircle:
    if not isinstance(geometry, ResolvedCircle):
        raise ValueError("该构造命令要求圆对象或圆特征")
    return geometry


def _require_tangent_source(
    geometry: ResolvedGeometry,
) -> ResolvedLine | ResolvedCircle:
    if not isinstance(geometry, (ResolvedLine, ResolvedCircle)):
        raise ValueError("相切构造只支持线或圆；阵列请先选择具体子线")
    return geometry


def _fixed_radius_locus(
    source: ResolvedLine | ResolvedCircle,
    constraint: TangencyConstraint,
    radius: float,
) -> tuple[ResolvedLine | ResolvedCircle | None, ConstructionIssue | None]:
    if isinstance(source, ResolvedLine):
        return (
            _offset_line(
                source,
                float(constraint.line_side) * radius,
                LineExtent.INFINITE,
            ),
            None,
        )
    relation = constraint.circle_relation
    if relation is CircleTangency.EXTERNAL:
        locus_radius = source.radius + radius
    elif relation is CircleTangency.SOURCE_CONTAINS:
        locus_radius = source.radius - radius
    else:
        locus_radius = radius - source.radius
    if locus_radius <= _EPSILON:
        return None, ConstructionIssue(
            "no_tangent_solution",
            "所选内切关系要求一个圆严格包含另一个圆",
        )
    return ResolvedCircle(source.center, locus_radius), None


def _validated_tangent_points(
    circle: ResolvedCircle,
    sources: tuple[ResolvedLine | ResolvedCircle, ...],
    constraints: tuple[TangencyConstraint, ...],
    *,
    extend: bool,
) -> tuple[Point, ...] | None:
    points: list[Point] = []
    for source, constraint in zip(sources, constraints, strict=True):
        point = _tangent_point(circle, source, constraint, extend=extend)
        if point is None:
            return None
        points.append(point)
    return tuple(points)


def tangent_points_for_circle(
    circle: ResolvedCircle,
    sources: tuple[ResolvedLine | ResolvedCircle, ...],
    constraints: tuple[TangencyConstraint, ...],
    *,
    extend: bool = False,
) -> tuple[Point, ...] | None:
    """Return validated contact points for a solved tangent circle.

    This read-only helper is used by interaction layers to expose temporary
    tangent object snaps without duplicating the domain solver's containment
    and tangency-relation rules.
    """

    if len(sources) != len(constraints):
        raise ValueError("相切来源与约束数量必须一致")
    return _validated_tangent_points(
        circle,
        tuple(_require_tangent_source(source) for source in sources),
        tuple(_constraint(constraint) for constraint in constraints),
        extend=bool(extend),
    )


def _tangent_point(
    circle: ResolvedCircle,
    source: ResolvedLine | ResolvedCircle,
    constraint: TangencyConstraint,
    *,
    extend: bool,
) -> Point | None:
    if isinstance(source, ResolvedLine):
        parameter = source.project_parameter(circle.center)
        dx, dy = source.direction
        signed_distance = (
            (circle.center.x - source.start.x) * (-dy)
            + (circle.center.y - source.start.y) * dx
        )
        expected = float(constraint.line_side) * circle.radius
        if abs(signed_distance - expected) > _scaled_epsilon(
            signed_distance,
            expected,
            circle.radius,
        ):
            return None
        if not extend and not source.contains_parameter(parameter, epsilon=1e-7):
            return None
        return source.point_at(parameter)
    vx = circle.center.x - source.center.x
    vy = circle.center.y - source.center.y
    center_distance = math.hypot(vx, vy)
    tolerance = _scaled_epsilon(
        center_distance,
        source.radius,
        circle.radius,
    )
    relation = constraint.circle_relation
    if relation is CircleTangency.EXTERNAL:
        expected = source.radius + circle.radius
    elif relation is CircleTangency.SOURCE_CONTAINS:
        if source.radius <= circle.radius + tolerance:
            return None
        expected = source.radius - circle.radius
    else:
        if circle.radius <= source.radius + tolerance:
            return None
        expected = circle.radius - source.radius
    if center_distance <= _EPSILON or abs(center_distance - expected) > tolerance:
        return None
    unit_x = vx / center_distance
    unit_y = vy / center_distance
    sign = -1.0 if relation is CircleTangency.SOLUTION_CONTAINS else 1.0
    return Point(
        source.center.x + sign * source.radius * unit_x,
        source.center.y + sign * source.radius * unit_y,
    )


def _circle_relation_sign(constraint: TangencyConstraint) -> float:
    return 1.0 if constraint.circle_relation is CircleTangency.EXTERNAL else -1.0


def _line_tangency_equation(
    line: ResolvedLine,
    constraint: TangencyConstraint,
) -> tuple[float, float, float, float]:
    dx, dy = line.direction
    normal_x, normal_y = -dy, dx
    return (
        normal_x,
        normal_y,
        -float(constraint.line_side),
        normal_x * line.start.x + normal_y * line.start.y,
    )


def _circle_difference_equation(
    base: ResolvedCircle,
    base_constraint: TangencyConstraint,
    other: ResolvedCircle,
    other_constraint: TangencyConstraint,
) -> tuple[float, float, float, float]:
    base_sign = _circle_relation_sign(base_constraint)
    other_sign = _circle_relation_sign(other_constraint)
    base_constant = (
        base.center.x * base.center.x
        + base.center.y * base.center.y
        - base.radius * base.radius
    )
    other_constant = (
        other.center.x * other.center.x
        + other.center.y * other.center.y
        - other.radius * other.radius
    )
    return (
        2.0 * (base.center.x - other.center.x),
        2.0 * (base.center.y - other.center.y),
        2.0 * (base_sign * base.radius - other_sign * other.radius),
        base_constant - other_constant,
    )


def _two_plane_affine_line(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    first = _normalized_equation(first)
    second = _normalized_equation(second)
    a = first[:3]
    b = second[:3]
    direction = _cross3(a, b)
    direction_length = math.sqrt(_dot3(direction, direction))
    if direction_length <= 1e-10:
        return None
    direction = tuple(value / direction_length for value in direction)
    dominant = max(range(3), key=lambda index: abs(direction[index]))
    if direction[dominant] < 0.0:
        direction = tuple(-value for value in direction)
    aa = _dot3(a, a)
    ab = _dot3(a, b)
    bb = _dot3(b, b)
    determinant = aa * bb - ab * ab
    if determinant <= 1e-14:
        return None
    lambda_first = (first[3] * bb - second[3] * ab) / determinant
    lambda_second = (second[3] * aa - first[3] * ab) / determinant
    origin = tuple(
        lambda_first * a[index] + lambda_second * b[index]
        for index in range(3)
    )
    return origin, direction


def _apollonius_roots(
    origin: tuple[float, float, float],
    direction: tuple[float, float, float],
    base: ResolvedCircle,
    constraint: TangencyConstraint,
) -> tuple[float, ...] | None:
    sigma = _circle_relation_sign(constraint)
    x = origin[0] - base.center.x
    y = origin[1] - base.center.y
    signed_radius = origin[2] + sigma * base.radius
    coefficient_a = (
        direction[0] * direction[0]
        + direction[1] * direction[1]
        - direction[2] * direction[2]
    )
    coefficient_b = 2.0 * (
        x * direction[0]
        + y * direction[1]
        - signed_radius * direction[2]
    )
    coefficient_c = x * x + y * y - signed_radius * signed_radius
    return _real_quadratic_roots(coefficient_a, coefficient_b, coefficient_c)


def _three_tangent_branch_sort_key(
    center: Point,
    radius: float,
    sources: tuple[
        ResolvedLine | ResolvedCircle,
        ResolvedLine | ResolvedCircle,
        ResolvedLine | ResolvedCircle,
    ],
) -> tuple[float, float, float]:
    """Return a rigid-motion-invariant ordering key for Apollonius roots.

    The quadratic parameter direction is an implementation detail and can flip
    sign when a rotated affine line changes its dominant component.  Radius is
    invariant under translation/rotation and separates the usual two roots.
    Equal-radius solutions are ordered in a frame carried by the ordered source
    geometry: the first oriented line when present, otherwise the first pair of
    distinct source-circle centers.
    """

    local_x = 0.0
    local_y = 0.0
    for source in sources:
        if not isinstance(source, ResolvedLine):
            continue
        direction_x, direction_y = source.direction
        relative_x = center.x - source.start.x
        relative_y = center.y - source.start.y
        local_x = relative_x * direction_x + relative_y * direction_y
        local_y = _cross(
            (direction_x, direction_y),
            (relative_x, relative_y),
        )
        break
    else:
        circle_centers = tuple(
            source.center
            for source in sources
            if isinstance(source, ResolvedCircle)
        )
        if circle_centers:
            origin = circle_centers[0]
            relative_x = center.x - origin.x
            relative_y = center.y - origin.y
            for axis_target in circle_centers[1:]:
                axis_x = axis_target.x - origin.x
                axis_y = axis_target.y - origin.y
                if math.hypot(axis_x, axis_y) <= _EPSILON:
                    continue
                axis_x, axis_y = _normalize((axis_x, axis_y))
                local_x = relative_x * axis_x + relative_y * axis_y
                local_y = _cross(
                    (axis_x, axis_y),
                    (relative_x, relative_y),
                )
                break
            else:
                # A fully concentric three-circle system is normally rejected
                # as underdetermined before roots are produced.  Retain a
                # deterministic radial fallback for numerical edge cases.
                local_x = math.hypot(relative_x, relative_y)
                local_y = 0.0
    return (
        _stable_branch_scalar(radius),
        _stable_branch_scalar(local_x),
        _stable_branch_scalar(local_y),
    )


def _stable_branch_scalar(value: float) -> float:
    """Suppress insignificant solver noise before comparing branch keys."""

    numeric = float(value)
    if abs(numeric) <= _EPSILON:
        return 0.0
    return float(f"{numeric:.12g}")


def _real_quadratic_roots(a: float, b: float, c: float) -> tuple[float, ...] | None:
    coefficient_scale = max(1.0, abs(a), abs(b), abs(c))
    a /= coefficient_scale
    b /= coefficient_scale
    c /= coefficient_scale
    coefficient_tolerance = 1e-14
    if abs(a) <= coefficient_tolerance:
        if abs(b) <= coefficient_tolerance:
            return None
        return (-c / b,)
    discriminant = b * b - 4.0 * a * c
    discriminant_tolerance = 1e-12 * max(1.0, abs(b * b), abs(4.0 * a * c))
    if discriminant < -discriminant_tolerance:
        return None
    if abs(discriminant) <= discriminant_tolerance:
        return (-b / (2.0 * a),)
    root = math.sqrt(max(0.0, discriminant))
    # The q form avoids cancellation for highly asymmetric solutions.
    q = -0.5 * (b + math.copysign(root, b))
    if abs(q) <= coefficient_tolerance:
        roots = ((-b - root) / (2.0 * a), (-b + root) / (2.0 * a))
    else:
        roots = (q / a, c / q)
    return tuple(sorted(roots))


def _solve_3x3(
    rows: tuple[
        tuple[float, float, float, float],
        tuple[float, float, float, float],
        tuple[float, float, float, float],
    ],
) -> tuple[float, float, float] | None:
    matrix = [list(_normalized_equation(row)) for row in rows]
    for column in range(3):
        pivot = max(range(column, 3), key=lambda row: abs(matrix[row][column]))
        if abs(matrix[pivot][column]) <= 1e-10:
            return None
        matrix[column], matrix[pivot] = matrix[pivot], matrix[column]
        divisor = matrix[column][column]
        matrix[column] = [value / divisor for value in matrix[column]]
        for row in range(3):
            if row == column:
                continue
            factor = matrix[row][column]
            matrix[row] = [
                matrix[row][index] - factor * matrix[column][index]
                for index in range(4)
            ]
    return matrix[0][3], matrix[1][3], matrix[2][3]


def _normalized_equation(
    row: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    norm = math.sqrt(row[0] * row[0] + row[1] * row[1] + row[2] * row[2])
    if norm <= _EPSILON:
        return row
    return tuple(value / norm for value in row)  # type: ignore[return-value]


def _cross3(
    first: tuple[float, float, float],
    second: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    )


def _dot3(
    first: tuple[float, float, float],
    second: tuple[float, float, float],
) -> float:
    return sum(left * right for left, right in zip(first, second, strict=True))


def _circles_close(first: ResolvedCircle, second: ResolvedCircle) -> bool:
    tolerance = _scaled_epsilon(
        first.center.x,
        first.center.y,
        first.radius,
        second.center.x,
        second.center.y,
        second.radius,
    )
    return (
        _distance(first.center, second.center) <= tolerance
        and abs(first.radius - second.radius) <= tolerance
    )


def _line_through(point: Point, direction: tuple[float, float], extent: LineExtent) -> ResolvedLine:
    dx, dy = _normalize(direction)
    return ResolvedLine(point, Point(point.x + dx, point.y + dy), extent)


def _offset_line(line: ResolvedLine, offset: float, extent: LineExtent) -> ResolvedLine:
    dx, dy = line.direction
    shift_x, shift_y = -dy * offset, dx * offset
    return ResolvedLine(
        Point(line.start.x + shift_x, line.start.y + shift_y),
        Point(line.end.x + shift_x, line.end.y + shift_y),
        extent,
    )


def _array_multipliers(count: int, side: ArraySide) -> tuple[int, ...]:
    if side is ArraySide.POSITIVE:
        return tuple(range(1, count + 1))
    if side is ArraySide.NEGATIVE:
        return tuple(range(-1, -count - 1, -1))
    return tuple(value for index in range(1, count + 1) for value in (-index, index))


def _circumcenter(first: Point, second: Point, third: Point) -> Point:
    determinant = 2.0 * (
        first.x * (second.y - third.y)
        + second.x * (third.y - first.y)
        + third.x * (first.y - second.y)
    )
    scale = max(
        1.0,
        _distance(first, second),
        _distance(second, third),
        _distance(third, first),
    )
    if abs(determinant) <= _EPSILON * scale * scale:
        raise ValueError("三点共线或过于接近，无法构造圆")
    first_sq = first.x * first.x + first.y * first.y
    second_sq = second.x * second.x + second.y * second.y
    third_sq = third.x * third.x + third.y * third.y
    x = (
        first_sq * (second.y - third.y)
        + second_sq * (third.y - first.y)
        + third_sq * (first.y - second.y)
    ) / determinant
    y = (
        first_sq * (third.x - second.x)
        + second_sq * (first.x - third.x)
        + third_sq * (second.x - first.x)
    ) / determinant
    return Point(x, y)


def _failure(entity_id: str, code: str, message: str, dependencies: Iterable[str] = ()) -> ResolvedConstruction:
    dependency_ids = tuple(dict.fromkeys(dependencies))
    return ResolvedConstruction(
        entity_id,
        error=ConstructionIssue(code, message, dependency_ids),
        dependencies=dependency_ids,
    )


def _point(value: object, field_name: str) -> Point:
    try:
        x = float(getattr(value, "x"))
        y = float(getattr(value, "y"))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} 必须是坐标点") from exc
    if not math.isfinite(x) or not math.isfinite(y):
        raise ValueError(f"{field_name} 坐标必须是有限数")
    return Point(x, y)


def _finite(value: object, field_name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} 必须是有限数") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field_name} 必须是有限数")
    return number


def _scaled_epsilon(*values: float) -> float:
    return 1e-9 * max(1.0, *(abs(float(value)) for value in values))


def _point_to_dict(point: Point) -> dict[str, float]:
    return {"x": point.x, "y": point.y}


def _point_from_payload(payload: Mapping[str, object], key: str) -> Point:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"缺少坐标字段 {key}")
    return Point(float(value["x"]), float(value["y"]))


def _distance(first: Point, second: Point) -> float:
    return math.hypot(second.x - first.x, second.y - first.y)


def _midpoint(first: Point, second: Point) -> Point:
    return Point((first.x + second.x) / 2.0, (first.y + second.y) / 2.0)


def _cross(first: tuple[float, float], second: tuple[float, float]) -> float:
    return first[0] * second[1] - first[1] * second[0]


def _normalize(vector: tuple[float, float]) -> tuple[float, float]:
    length = math.hypot(*vector)
    if not math.isfinite(length) or length <= _EPSILON:
        raise ValueError("方向向量不能为零")
    return vector[0] / length, vector[1] / length


__all__ = [
    "ArraySide",
    "CONSTRUCTION_SCHEMA_VERSION",
    "CircleCenterDiameterDefinition",
    "CircleCenterRadiusDefinition",
    "CircleTangency",
    "CircleThreePointDefinition",
    "CircleTwoPointDefinition",
    "CommonTangentDefinition",
    "CommonTangentMode",
    "ConcentricCircleDefinition",
    "ConstructionDefinition",
    "ConstructionEntity",
    "ConstructionIssue",
    "ConstructionResolver",
    "ConstructionStyle",
    "ConstructionValidationError",
    "FeatureSource",
    "FreePointDefinition",
    "FrozenFeatureSnapshot",
    "IntersectionDefinition",
    "IntersectionBranchHint",
    "IntersectionBranchKind",
    "LineDefinition",
    "LineExtent",
    "LiveFeatureRef",
    "MidpointDefinition",
    "OffsetParallelDefinition",
    "OffsetCircleDefinition",
    "ParallelArrayDefinition",
    "ParallelLineSequence",
    "ParallelThroughPointDefinition",
    "PerpendicularBisectorDefinition",
    "PerpendicularDefinition",
    "PointCircleTangentDefinition",
    "ResolvedCircle",
    "ResolvedConstruction",
    "ResolvedGeometry",
    "ResolvedLine",
    "ResolvedLineArray",
    "ResolvedPoint",
    "SourceObjectIdentity",
    "SourceObjectKind",
    "TangentCircleSolution",
    "TangentTangentRadiusCircleDefinition",
    "TangencyConstraint",
    "ThreeTangentCircleDefinition",
    "common_tangent_lines",
    "definition_from_dict",
    "definition_to_dict",
    "detach_sources",
    "geometry_intersections",
    "intersection_branch_hint",
    "iter_live_refs",
    "live_dependency_identities",
    "live_dependency_ids",
    "select_feature",
    "point_circle_tangent_lines",
    "tangent_tangent_radius_solutions",
    "tangent_points_for_circle",
    "three_tangent_circle_solutions",
    "transitive_dependents",
    "validate_construction_graph",
]
