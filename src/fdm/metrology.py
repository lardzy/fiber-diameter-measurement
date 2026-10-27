"""Persisted definitions and derived results for calibrated 2-D metrology.

This module has no dependency on the document model or Qt. Pixel control
points are the source of truth; values are evaluated with the current
calibration and never stored in the legacy fibre diameter fields.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
import json
import math
from typing import Any, Mapping

from fdm.construction_geometry import LiveFeatureRef, SourceObjectKind
from fdm.geometry import Point
from fdm.measurement_annotation import MeasurementAnnotationLayout


FEATURE_KINDS = frozenset({"point", "line", "circle", "arc", "polyline", "polygon", "fit_line", "fit_circle", "caliper_line", "caliper_circle", "caliper_width"})
MEASUREMENT_KINDS = frozenset({"distance", "point_line_distance", "point_segment_distance", "parallel_distance", "projection", "angle", "line_angle", "circle", "radius", "diameter", "arc", "circle_center_distance", "circle_gap", "point_circle_distance", "polyline", "polygon", "coordinates", "width", "fit_line", "fit_circle"})


def _identity(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} 必须是非空字符串")
    return value


def _points(points: object) -> tuple[Point, ...]:
    if not isinstance(points, (list, tuple)):
        raise TypeError("points_px 必须是坐标列表")
    result = []
    for point in points:
        if not isinstance(point, Point):
            raise TypeError("控制点必须为 Point")
        if not math.isfinite(point.x) or not math.isfinite(point.y):
            raise ValueError("控制点必须是有限数")
        result.append(Point(float(point.x), float(point.y)))
    return tuple(result)


def _json_dict(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{name} 必须是对象")
    # Round-trip isolates caller-owned nested arrays and rejects NaN/Infinity.
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


def _focus(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("focus_index 必须是非负整数")
    return value


def _parse_points(payload: Mapping[str, Any]) -> tuple[Point, ...]:
    raw = payload.get("points_px", [])
    if not isinstance(raw, list) or any(not isinstance(p, dict) for p in raw):
        raise TypeError("points_px 必须是坐标对象列表")
    if any(isinstance(p.get(axis), bool) or not isinstance(p.get(axis), (int, float)) for p in raw for axis in ("x", "y")):
        raise TypeError("控制点 X/Y 必须是数值")
    return tuple(Point.from_dict(p) for p in raw)


@dataclass(frozen=True, slots=True)
class MeasurementFeature:
    id: str
    image_id: str
    kind: str
    points_px: tuple[Point, ...] = ()
    source_ref: LiveFeatureRef | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    focus_index: int | None = None
    source_revision: str | None = None
    quality: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _identity(self.id, "特征 ID")
        _identity(self.image_id, "图像 ID")
        if self.kind not in FEATURE_KINDS:
            raise ValueError(f"不支持的专业几何类型：{self.kind}")
        object.__setattr__(self, "points_px", _points(self.points_px))
        if self.source_ref is not None and not isinstance(self.source_ref, LiveFeatureRef):
            raise TypeError("source_ref 必须是实时辅助几何引用")
        object.__setattr__(self, "parameters", _json_dict(self.parameters, "parameters"))
        object.__setattr__(self, "quality", _json_dict(self.quality, "quality"))
        object.__setattr__(self, "focus_index", _focus(self.focus_index))
        if self.source_revision is not None:
            _identity(self.source_revision, "source_revision")

    def clone(self, **changes: Any) -> "MeasurementFeature":
        return replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "image_id": self.image_id, "kind": self.kind,
                "points_px": [p.to_dict() for p in self.points_px],
                "source_ref": self.source_ref.to_dict() if self.source_ref else None,
                "parameters": _json_dict(self.parameters, "parameters"),
                "focus_index": self.focus_index, "source_revision": self.source_revision,
                "quality": _json_dict(self.quality, "quality")}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MeasurementFeature":
        if not isinstance(payload, Mapping):
            raise TypeError("专业特征必须是对象")
        ref = payload.get("source_ref")
        if ref is not None and not isinstance(ref, dict):
            raise TypeError("source_ref 必须是对象")
        if ref is not None:
            if ref.get("source_type", "live") != "live":
                raise ValueError("source_ref 仅支持实时引用；冻结后应保存控制点")
            for name in ("document_id", "object_id", "feature"):
                _identity(ref.get(name, "geometry" if name == "feature" else None), name)
        source = None if ref is None else LiveFeatureRef(
            document_id=ref["document_id"], object_id=ref["object_id"],
            object_kind=SourceObjectKind(ref.get("object_kind", "construction")),
            feature=ref.get("feature", "geometry"))
        return cls(id=payload["id"], image_id=payload["image_id"], kind=payload["kind"],
                   points_px=_parse_points(payload), source_ref=source,
                   parameters=payload.get("parameters", {}), focus_index=payload.get("focus_index"),
                   source_revision=payload.get("source_revision"), quality=payload.get("quality", {}))


@dataclass(frozen=True, slots=True)
class MeasurementDefinition:
    id: str
    image_id: str
    kind: str
    feature_ids: tuple[str, ...]
    parameters: dict[str, Any] = field(default_factory=dict)
    name: str = ""
    annotation_layout: MeasurementAnnotationLayout | None = field(default_factory=MeasurementAnnotationLayout)
    focus_index: int | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self) -> None:
        _identity(self.id, "测量 ID")
        _identity(self.image_id, "图像 ID")
        if self.kind not in MEASUREMENT_KINDS:
            raise ValueError(f"不支持的专业测量类型：{self.kind}")
        if not isinstance(self.feature_ids, (tuple, list)) or not self.feature_ids:
            raise ValueError("测量至少需要一个来源特征")
        object.__setattr__(self, "feature_ids", tuple(_identity(p, "来源特征 ID") for p in self.feature_ids))
        object.__setattr__(self, "parameters", _json_dict(self.parameters, "parameters"))
        object.__setattr__(self, "focus_index", _focus(self.focus_index))
        if not isinstance(self.name, str):
            raise TypeError("测量名称必须是字符串")
        if not isinstance(self.created_at, str):
            raise TypeError("created_at 必须是字符串")
        if self.annotation_layout is not None and not isinstance(self.annotation_layout, MeasurementAnnotationLayout):
            raise TypeError("annotation_layout 类型错误")

    def clone(self, **changes: Any) -> "MeasurementDefinition":
        return replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "image_id": self.image_id, "kind": self.kind,
                "feature_ids": list(self.feature_ids), "parameters": _json_dict(self.parameters, "parameters"),
                "name": self.name, "annotation_layout": self.annotation_layout.to_dict() if self.annotation_layout else None,
                "focus_index": self.focus_index, "created_at": self.created_at}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MeasurementDefinition":
        if not isinstance(payload, Mapping):
            raise TypeError("专业测量必须是对象")
        layout = payload.get("annotation_layout")
        if layout is not None and not isinstance(layout, dict):
            raise TypeError("annotation_layout 必须是对象")
        return cls(id=payload["id"], image_id=payload["image_id"], kind=payload["kind"],
                   feature_ids=payload["feature_ids"], parameters=payload.get("parameters", {}),
                   name=payload.get("name", ""),
                   annotation_layout=MeasurementAnnotationLayout.from_dict(layout) if layout is not None else None,
                   focus_index=payload.get("focus_index"), created_at=payload.get("created_at", ""))


@dataclass(frozen=True, slots=True)
class MeasurementMetric:
    key: str
    label: str
    value: float
    unit: str


Metric = MeasurementMetric


@dataclass(frozen=True, slots=True)
class MeasurementPrimitive:
    """Image-space overlay. Curves are sampled polylines; never analysis input."""
    kind: str
    points_px: tuple[Point, ...]
    role: str = "measurement"


@dataclass(frozen=True, slots=True)
class MeasurementResult:
    measurement_id: str
    status: str
    reason: str = ""
    metrics: tuple[MeasurementMetric, ...] = ()
    anchor_px: Point = field(default_factory=lambda: Point(0.0, 0.0))
    geometry: tuple[MeasurementPrimitive, ...] = ()
    quality: dict[str, Any] = field(default_factory=dict)

    @property
    def valid(self) -> bool:
        return self.status == "ready"

    @property
    def primary(self) -> MeasurementMetric | None:
        return self.metrics[0] if self.metrics else None


@dataclass(frozen=True, slots=True)
class ResolvedMeasurementFeature:
    feature_id: str
    kind: str
    points: tuple[Point, ...]  # calibrated physical plane (px when uncalibrated)
    points_px: tuple[Point, ...]
    center: Point | None = None
    radius: float | None = None
    quality: dict[str, Any] = field(default_factory=dict)
    error: str = ""

    @property
    def valid(self) -> bool:
        return not self.error


def parse_professional_state(payload: Mapping[str, Any], image_id: str) -> tuple[list[MeasurementFeature], list[MeasurementDefinition]]:
    collections = []
    for key, cls in (("professional_features", MeasurementFeature), ("professional_measurements", MeasurementDefinition)):
        raw = payload.get(key, [])
        if not isinstance(raw, list) or any(not isinstance(v, dict) for v in raw):
            raise TypeError(f"{key} 必须是对象列表")
        rows = [cls.from_dict(v) for v in raw]
        if len({r.id for r in rows}) != len(rows):
            raise ValueError(f"{key} 包含重复 ID")
        if any(r.image_id != image_id for r in rows):
            raise ValueError(f"{key} 的来源图像不匹配")
        collections.append(rows)
    # Missing dependencies are retained as explicitly invalid measurements so
    # deletion/undo and portable partial projects do not silently drop records.
    return collections[0], collections[1]
