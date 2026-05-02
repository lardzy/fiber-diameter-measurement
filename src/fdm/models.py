from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import statistics
import uuid

from fdm.geometry import (
    Line,
    Point,
    area_rings_area,
    area_rings_centroid,
    line_length,
    midpoint,
    polygon_area,
    polygon_centroid,
    polyline_centroid,
    polyline_length,
)
from fdm.version import __version__

UNCATEGORIZED_LABEL = "未分类"
UNCATEGORIZED_COLOR = "#98A2B3"


def utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def project_assets_root(project_path: str | Path) -> Path:
    return Path(project_path).with_suffix(".assets")


def project_capture_root(project_path: str | Path) -> Path:
    return project_assets_root(project_path) / "captures"


def format_measurement_label_value(value: float, unit: str, decimals: int) -> str:
    decimals = max(0, min(8, int(decimals)))
    formatted = f"{value:.{decimals}f}"
    if not formatted:
        formatted = "0"
    return f"{formatted} {unit}"


def square_unit(unit: str) -> str:
    return f"{unit}²"


def normalize_group_label(label: str) -> str:
    return str(label or "").strip()


@dataclass(slots=True)
class Calibration:
    mode: str
    pixels_per_unit: float
    unit: str
    source_label: str

    def clone(self, *, mode: str | None = None, source_label: str | None = None) -> "Calibration":
        return Calibration(
            mode=mode or self.mode,
            pixels_per_unit=self.pixels_per_unit,
            unit=self.unit,
            source_label=self.source_label if source_label is None else source_label,
        )

    def as_project_default(self) -> "Calibration":
        return self.clone(mode="project_default")

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "pixels_per_unit": self.pixels_per_unit,
            "unit": self.unit,
            "source_label": self.source_label,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Calibration":
        return cls(
            mode=str(payload["mode"]),
            pixels_per_unit=float(payload["pixels_per_unit"]),
            unit=str(payload["unit"]),
            source_label=str(payload["source_label"]),
        )

    def px_to_unit(self, value_px: float) -> float:
        if self.pixels_per_unit <= 0:
            return value_px
        return value_px / self.pixels_per_unit

    def unit_to_px(self, value: float) -> float:
        return value * self.pixels_per_unit

    def px_area_to_unit(self, value_px: float) -> float:
        if self.pixels_per_unit <= 0:
            return value_px
        return value_px / (self.pixels_per_unit ** 2)


@dataclass(slots=True)
class CalibrationPreset:
    name: str
    pixels_per_unit: float
    unit: str
    pixel_distance: float | None = None
    actual_distance: float | None = None
    computed_pixels_per_unit: float | None = None

    def resolved_pixels_per_unit(self) -> float:
        return self.computed_pixels_per_unit or self.pixels_per_unit

    def to_calibration(self) -> Calibration:
        return Calibration(
            mode="preset",
            pixels_per_unit=self.resolved_pixels_per_unit(),
            unit=self.unit,
            source_label=self.name,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "pixels_per_unit": self.resolved_pixels_per_unit(),
            "unit": self.unit,
            "pixel_distance": self.pixel_distance,
            "actual_distance": self.actual_distance,
            "computed_pixels_per_unit": self.resolved_pixels_per_unit(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CalibrationPreset":
        computed_pixels_per_unit = payload.get("computed_pixels_per_unit")
        pixels_per_unit = float(computed_pixels_per_unit or payload["pixels_per_unit"])
        return cls(
            name=str(payload["name"]),
            pixels_per_unit=pixels_per_unit,
            unit=str(payload["unit"]),
            pixel_distance=float(payload["pixel_distance"]) if payload.get("pixel_distance") is not None else None,
            actual_distance=float(payload["actual_distance"]) if payload.get("actual_distance") is not None else None,
            computed_pixels_per_unit=pixels_per_unit,
        )


@dataclass(slots=True)
class DirtyFlags:
    session_dirty: bool = False
    calibration_dirty: bool = False

    def copy(self) -> "DirtyFlags":
        return DirtyFlags(
            session_dirty=self.session_dirty,
            calibration_dirty=self.calibration_dirty,
        )


@dataclass(slots=True)
class FiberGroup:
    id: str
    image_id: str
    number: int
    color: str
    label: str = ""
    measurement_ids: list[str] = field(default_factory=list)

    def display_name(self) -> str:
        return f"{self.number} {self.label}".strip()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "image_id": self.image_id,
            "number": self.number,
            "label": self.label,
            "color": self.color,
            "measurement_ids": list(self.measurement_ids),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any], *, fallback_number: int = 1) -> "FiberGroup":
        return cls(
            id=str(payload["id"]),
            image_id=str(payload["image_id"]),
            number=int(payload.get("number", fallback_number)),
            label=normalize_group_label(str(payload.get("label", payload.get("name", "")))),
            color=str(payload["color"]),
            measurement_ids=list(payload.get("measurement_ids", [])),
        )


@dataclass(slots=True)
class ProjectGroupTemplate:
    label: str
    color: str

    def normalized_label(self) -> str:
        return normalize_group_label(self.label)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": normalize_group_label(self.label),
            "color": self.color,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ProjectGroupTemplate":
        return cls(
            label=normalize_group_label(str(payload.get("label", ""))),
            color=str(payload.get("color", "#1F7A8C")),
        )


@dataclass(slots=True)
class Measurement:
    id: str
    image_id: str
    fiber_group_id: str | None
    mode: str
    measurement_kind: str = "line"
    line_px: Line | None = None
    polyline_px: list[Point] = field(default_factory=list)
    point_px: Point | None = None
    polygon_px: list[Point] = field(default_factory=list)
    area_rings_px: list[list[Point]] = field(default_factory=list)
    snapped_line_px: Line | None = None
    diameter_px: float | None = None
    diameter_unit: float | None = None
    exact_area_px: float | None = None
    area_px: float | None = None
    area_unit: float | None = None
    confidence: float = 0.0
    status: str = "ready"
    created_at: str = field(default_factory=utc_now_iso)
    debug_payload: dict[str, Any] = field(default_factory=dict)
    display_polygon_px: list[Point] = field(default_factory=list, repr=False)
    display_area_rings_px: list[list[Point]] = field(default_factory=list, repr=False)
    display_bounds_px: tuple[float, float, float, float] | None = field(default=None, repr=False)

    def effective_line(self) -> Line:
        if self.measurement_kind != "line" or self.line_px is None:
            raise ValueError("Only straight line measurements expose line geometry.")
        return self.snapped_line_px or self.line_px

    def display_value(self) -> float:
        if self.measurement_kind == "area":
            return self.area_unit if self.area_unit is not None else self.area_px or 0.0
        if self.measurement_kind == "count":
            return 1.0
        return self.diameter_unit if self.diameter_unit is not None else self.diameter_px or 0.0

    def display_unit(self, calibration: Calibration | None) -> str:
        if self.measurement_kind == "area":
            return square_unit(calibration.unit if calibration else "px")
        if self.measurement_kind == "count":
            return "个"
        return calibration.unit if calibration else "px"

    def display_label(self, calibration: Calibration | None) -> str:
        return format_measurement_label_value(
            self.display_value(),
            self.display_unit(calibration),
            4,
        )

    def polygon_center(self) -> Point:
        if self.measurement_kind == "area" and self.area_rings_px:
            return area_rings_centroid(self.area_rings_px)
        return polygon_centroid(self.polygon_px)

    def geometry_center(self) -> Point:
        if self.measurement_kind == "area":
            return self.polygon_center()
        if self.measurement_kind == "polyline":
            return polyline_centroid(self.polyline_px)
        if self.measurement_kind == "count" and self.point_px is not None:
            return Point(self.point_px.x, self.point_px.y)
        if self.measurement_kind == "line" and self.line_px is not None:
            return midpoint(self.effective_line())
        return Point(0.0, 0.0)

    def recalculate(self, calibration: Calibration | None) -> None:
        if self.measurement_kind == "area":
            self.area_px = (
                float(self.exact_area_px)
                if self.exact_area_px is not None
                else (
                    area_rings_area(self.area_rings_px)
                    if self.area_rings_px
                    else polygon_area(self.polygon_px)
                )
            )
            if calibration is None:
                self.area_unit = self.area_px
            else:
                self.area_unit = calibration.px_area_to_unit(self.area_px)
            self.diameter_px = None
            self.diameter_unit = None
            return
        if self.measurement_kind == "count":
            self.diameter_px = None
            self.diameter_unit = None
            self.area_px = None
            self.area_unit = None
            return
        if self.measurement_kind == "polyline":
            self.diameter_px = polyline_length(self.polyline_px)
        else:
            self.diameter_px = line_length(self.effective_line())
        if calibration is None:
            self.diameter_unit = self.diameter_px
        else:
            self.diameter_unit = calibration.px_to_unit(self.diameter_px)
        self.area_px = None
        self.area_unit = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "image_id": self.image_id,
            "fiber_group_id": self.fiber_group_id,
            "measurement_kind": self.measurement_kind,
            "mode": self.mode,
            "line_px": self.line_px.to_dict() if self.line_px else None,
            "polyline_px": [point.to_dict() for point in self.polyline_px],
            "point_px": self.point_px.to_dict() if self.point_px else None,
            "polygon_px": [point.to_dict() for point in self.polygon_px],
            "area_rings_px": [
                [point.to_dict() for point in ring]
                for ring in self.area_rings_px
            ],
            "snapped_line_px": self.snapped_line_px.to_dict() if self.snapped_line_px else None,
            "diameter_px": self.diameter_px,
            "diameter_unit": self.diameter_unit,
            "exact_area_px": self.exact_area_px,
            "area_px": self.area_px,
            "area_unit": self.area_unit,
            "confidence": self.confidence,
            "status": self.status,
            "created_at": self.created_at,
            "debug_payload": self.debug_payload,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Measurement":
        snapped_line = payload.get("snapped_line_px")
        line_payload = payload.get("line_px")
        kind = str(payload.get("measurement_kind", "line"))
        mode = str(payload.get("mode", "manual"))
        if mode == "fiber_auto":
            mode = "fiber_quick"
        if mode == "continuous":
            mode = "continuous_manual"
        status = str(payload.get("status", "ready"))
        if status == "fiber_auto":
            status = "fiber_quick"
        return cls(
            id=str(payload["id"]),
            image_id=str(payload["image_id"]),
            fiber_group_id=payload.get("fiber_group_id"),
            measurement_kind=kind,
            mode=mode,
            line_px=Line.from_dict(line_payload) if line_payload else None,
            polyline_px=[
                Point.from_dict(item)
                for item in payload.get("polyline_px", [])
                if isinstance(item, dict)
            ],
            point_px=Point.from_dict(payload["point_px"]) if isinstance(payload.get("point_px"), dict) else None,
            polygon_px=[
                Point.from_dict(item)
                for item in payload.get("polygon_px", [])
                if isinstance(item, dict)
            ],
            area_rings_px=[
                [
                    Point.from_dict(item)
                    for item in ring
                    if isinstance(item, dict)
                ]
                for ring in payload.get("area_rings_px", [])
                if isinstance(ring, list)
            ],
            snapped_line_px=Line.from_dict(snapped_line) if snapped_line else None,
            diameter_px=payload.get("diameter_px"),
            diameter_unit=payload.get("diameter_unit"),
            exact_area_px=float(payload["exact_area_px"]) if payload.get("exact_area_px") is not None else None,
            area_px=payload.get("area_px"),
            area_unit=payload.get("area_unit"),
            confidence=float(payload.get("confidence", 0.0)),
            status=status,
            created_at=str(payload.get("created_at", utc_now_iso())),
            debug_payload=dict(payload.get("debug_payload", {})),
        )


@dataclass(slots=True)
class TextAnnotation:
    id: str
    image_id: str
    content: str
    anchor_px: Point
    created_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "image_id": self.image_id,
            "content": self.content,
            "anchor_px": self.anchor_px.to_dict(),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TextAnnotation":
        return cls(
            id=str(payload["id"]),
            image_id=str(payload["image_id"]),
            content=str(payload.get("content", "")),
            anchor_px=Point.from_dict(payload.get("anchor_px", {"x": 0.0, "y": 0.0})),
            created_at=str(payload.get("created_at", utc_now_iso())),
        )

    def to_overlay(self) -> "OverlayAnnotation":
        return OverlayAnnotation(
            id=self.id,
            image_id=self.image_id,
            kind=OverlayAnnotationKind.TEXT,
            content=self.content,
            anchor_px=self.anchor_px,
            created_at=self.created_at,
        )


class OverlayAnnotationKind:
    TEXT = "text"
    RECT = "rect"
    CIRCLE = "circle"
    LINE = "line"
    ARROW = "arrow"


@dataclass(slots=True)
class OverlayAnnotation:
    id: str
    image_id: str
    kind: str
    content: str = ""
    anchor_px: Point = field(default_factory=lambda: Point(0.0, 0.0))
    start_px: Point = field(default_factory=lambda: Point(0.0, 0.0))
    end_px: Point = field(default_factory=lambda: Point(0.0, 0.0))
    created_at: str = field(default_factory=utc_now_iso)

    def normalized_kind(self) -> str:
        if self.kind in {
            OverlayAnnotationKind.TEXT,
            OverlayAnnotationKind.RECT,
            OverlayAnnotationKind.CIRCLE,
            OverlayAnnotationKind.LINE,
            OverlayAnnotationKind.ARROW,
        }:
            return self.kind
        return OverlayAnnotationKind.TEXT

    def is_text(self) -> bool:
        return self.normalized_kind() == OverlayAnnotationKind.TEXT

    def clone(self, **changes) -> "OverlayAnnotation":
        return replace(self, **changes)

    def translated(self, dx: float, dy: float) -> "OverlayAnnotation":
        if self.is_text():
            return self.clone(anchor_px=Point(self.anchor_px.x + dx, self.anchor_px.y + dy))
        return self.clone(
            start_px=Point(self.start_px.x + dx, self.start_px.y + dy),
            end_px=Point(self.end_px.x + dx, self.end_px.y + dy),
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.id,
            "image_id": self.image_id,
            "kind": self.normalized_kind(),
            "created_at": self.created_at,
        }
        if self.is_text():
            payload["content"] = self.content
            payload["anchor_px"] = self.anchor_px.to_dict()
        else:
            payload["start_px"] = self.start_px.to_dict()
            payload["end_px"] = self.end_px.to_dict()
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "OverlayAnnotation":
        kind = str(payload.get("kind", OverlayAnnotationKind.TEXT)).strip() or OverlayAnnotationKind.TEXT
        if kind not in {
            OverlayAnnotationKind.TEXT,
            OverlayAnnotationKind.RECT,
            OverlayAnnotationKind.CIRCLE,
            OverlayAnnotationKind.LINE,
            OverlayAnnotationKind.ARROW,
        }:
            kind = OverlayAnnotationKind.TEXT
        if kind == OverlayAnnotationKind.TEXT:
            return cls(
                id=str(payload["id"]),
                image_id=str(payload["image_id"]),
                kind=kind,
                content=str(payload.get("content", "")),
                anchor_px=Point.from_dict(payload.get("anchor_px", {"x": 0.0, "y": 0.0})),
                created_at=str(payload.get("created_at", utc_now_iso())),
            )
        return cls(
            id=str(payload["id"]),
            image_id=str(payload["image_id"]),
            kind=kind,
            start_px=Point.from_dict(payload.get("start_px", {"x": 0.0, "y": 0.0})),
            end_px=Point.from_dict(payload.get("end_px", {"x": 0.0, "y": 0.0})),
            created_at=str(payload.get("created_at", utc_now_iso())),
        )


@dataclass(slots=True)
class ImageViewState:
    zoom: float = 1.0
    pan: Point = field(default_factory=lambda: Point(0.0, 0.0))
    selected_measurement_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "zoom": self.zoom,
            "pan": self.pan.to_dict(),
            "selected_measurement_id": self.selected_measurement_id,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ImageViewState":
        return cls(
            zoom=float(payload.get("zoom", 1.0)),
            pan=Point.from_dict(payload.get("pan", {"x": 0.0, "y": 0.0})),
            selected_measurement_id=payload.get("selected_measurement_id"),
        )


@dataclass(slots=True)
class CalibrationSidecar:
    image_path: str
    calibration: Calibration
    calibration_line: Line | None = None
    updated_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": "1",
            "image_path": self.image_path,
            "calibration": self.calibration.to_dict(),
            "calibration_line": self.calibration_line.to_dict() if self.calibration_line else None,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CalibrationSidecar":
        line_payload = payload.get("calibration_line")
        return cls(
            image_path=str(payload.get("image_path", "")),
            calibration=Calibration.from_dict(payload["calibration"]),
            calibration_line=Line.from_dict(line_payload) if line_payload else None,
            updated_at=str(payload.get("updated_at", utc_now_iso())),
        )


@dataclass(slots=True)
class ImageDocument:
    id: str
    path: str
    image_size: tuple[int, int]
    source_type: str = "filesystem"
    calibration: Calibration | None = None
    fiber_groups: list[FiberGroup] = field(default_factory=list)
    measurements: list[Measurement] = field(default_factory=list)
    overlay_annotations: list[OverlayAnnotation] = field(default_factory=list)
    view_state: ImageViewState = field(default_factory=ImageViewState)
    metadata: dict[str, Any] = field(default_factory=dict)
    active_group_id: str | None = None
    selected_overlay_id: str | None = None
    scale_overlay_anchor: Point | None = None
    suppressed_project_group_labels: list[str] = field(default_factory=list)
    sidecar_path: str | None = None
    dirty_flags: DirtyFlags = field(default_factory=DirtyFlags)
    history: Any = field(default=None, repr=False, compare=False)
    _session_clean_snapshot: dict[str, Any] | None = field(default=None, repr=False, compare=False)
    _calibration_clean_snapshot: dict[str, Any] | None = field(default=None, repr=False, compare=False)
    _measurement_geometry_revision: int = field(default=0, init=False, repr=False, compare=False)

    def initialize_runtime_state(self) -> None:
        from fdm.history import DocumentHistory

        if self.history is None:
            self.history = DocumentHistory()
        if self.sidecar_path is None and self.path and self.uses_sidecar():
            self.sidecar_path = self.default_sidecar_path()
        self.fiber_groups.sort(key=lambda group: group.number)
        self.suppressed_project_group_labels = self._normalized_suppressed_project_group_labels(self.suppressed_project_group_labels)
        self.rebuild_group_memberships()
        if self.active_group_id is None or self.get_group(self.active_group_id) is None:
            self.active_group_id = self.fiber_groups[0].id if self.fiber_groups else None
        if self.selected_overlay_id and self.get_overlay_annotation(self.selected_overlay_id) is None:
            self.selected_overlay_id = None
        if self._session_clean_snapshot is None:
            self.mark_session_saved()
        if self._calibration_clean_snapshot is None:
            self.mark_calibration_saved()
        self.refresh_dirty_flags()

    def default_sidecar_path(self) -> str:
        return f"{self.resolved_path()}.fdm.json"

    def uses_sidecar(self) -> bool:
        return self.source_type == "filesystem" and bool(str(self.path).strip())

    def is_project_asset(self) -> bool:
        return self.source_type == "project_asset"

    def resolved_path(self, project_path: str | Path | None = None) -> Path:
        token = str(self.path or "").strip()
        if not token:
            return Path()
        if self.is_project_asset():
            base = project_assets_root(project_path) if project_path is not None else Path()
            return (base / token).resolve() if base else Path(token)
        return Path(token).expanduser().resolve()

    def sorted_groups(self) -> list[FiberGroup]:
        return sorted(self.fiber_groups, key=lambda group: group.number)

    def uncategorized_measurements(self) -> list[Measurement]:
        return [measurement for measurement in self.measurements if measurement.fiber_group_id is None]

    def line_measurements(self) -> list[Measurement]:
        return [measurement for measurement in self.measurements if measurement.measurement_kind == "line"]

    def polyline_measurements(self) -> list[Measurement]:
        return [measurement for measurement in self.measurements if measurement.measurement_kind == "polyline"]

    def area_measurements(self) -> list[Measurement]:
        return [measurement for measurement in self.measurements if measurement.measurement_kind == "area"]

    def count_measurements(self) -> list[Measurement]:
        return [measurement for measurement in self.measurements if measurement.measurement_kind == "count"]

    def uncategorized_measurement_count(self) -> int:
        return len(self.uncategorized_measurements())

    def should_show_uncategorized_entry(self) -> bool:
        return (
            not self.fiber_groups
            or self.uncategorized_measurement_count() > 0
            or self.active_group_id is None
        )

    def can_delete_uncategorized_entry(self) -> bool:
        return self.uncategorized_measurement_count() == 0 and bool(self.fiber_groups)

    def next_group_number(self) -> int:
        if not self.fiber_groups:
            return 1
        return max(group.number for group in self.fiber_groups) + 1

    def create_group(self, *, color: str, label: str = "") -> FiberGroup:
        group = FiberGroup(
            id=new_id("group"),
            image_id=self.id,
            number=self.next_group_number(),
            label=normalize_group_label(label),
            color=color,
        )
        self.fiber_groups.append(group)
        self.fiber_groups.sort(key=lambda item: item.number)
        if self.active_group_id is None:
            self.active_group_id = group.id
        return group

    def find_group_by_label(self, label: str) -> FiberGroup | None:
        token = normalize_group_label(label)
        if not token:
            return None
        for group in self.sorted_groups():
            if normalize_group_label(group.label) == token:
                return group
        return None

    def groups_by_label(self, label: str) -> list[FiberGroup]:
        token = normalize_group_label(label)
        if not token:
            return []
        return [
            group
            for group in self.sorted_groups()
            if normalize_group_label(group.label) == token
        ]

    def ensure_group_for_label(self, label: str, *, color: str) -> FiberGroup:
        existing = self.find_group_by_label(label)
        if existing is not None:
            return existing
        active_group_id = self.active_group_id
        group = self.create_group(color=color, label=label)
        self.active_group_id = active_group_id
        return group

    def ensure_default_group(self) -> FiberGroup:
        if self.fiber_groups:
            return self.sorted_groups()[0]
        return self.create_group(color="#1F7A8C")

    def get_group(self, group_id: str | None) -> FiberGroup | None:
        if group_id is None:
            return None
        for group in self.fiber_groups:
            if group.id == group_id:
                return group
        return None

    def get_group_by_number(self, number: int) -> FiberGroup | None:
        for group in self.fiber_groups:
            if group.number == number:
                return group
        return None

    def set_active_group(self, group_id: str | None) -> None:
        if group_id is None:
            self.active_group_id = None
            return
        if self.get_group(group_id) is None:
            return
        self.active_group_id = group_id

    def get_measurement(self, measurement_id: str | None) -> Measurement | None:
        if measurement_id is None:
            return None
        for measurement in self.measurements:
            if measurement.id == measurement_id:
                return measurement
        return None

    @property
    def measurement_geometry_revision(self) -> int:
        return self._measurement_geometry_revision

    def mark_measurement_geometry_changed(self) -> None:
        self._measurement_geometry_revision += 1

    @property
    def text_annotations(self) -> list[OverlayAnnotation]:
        return [
            annotation
            for annotation in self.overlay_annotations
            if annotation.normalized_kind() == OverlayAnnotationKind.TEXT
        ]

    @property
    def selected_text_id(self) -> str | None:
        annotation = self.get_overlay_annotation(self.selected_overlay_id)
        if annotation is None or annotation.normalized_kind() != OverlayAnnotationKind.TEXT:
            return None
        return annotation.id

    @selected_text_id.setter
    def selected_text_id(self, value: str | None) -> None:
        self.selected_overlay_id = value

    def get_overlay_annotation(self, overlay_id: str | None) -> OverlayAnnotation | None:
        if overlay_id is None:
            return None
        for annotation in self.overlay_annotations:
            if annotation.id == overlay_id:
                return annotation
        return None

    def get_text_annotation(self, text_id: str | None) -> OverlayAnnotation | None:
        annotation = self.get_overlay_annotation(text_id)
        if annotation is None or annotation.normalized_kind() != OverlayAnnotationKind.TEXT:
            return None
        return annotation

    def select_measurement(self, measurement_id: str | None) -> None:
        self.view_state.selected_measurement_id = measurement_id
        if measurement_id is not None:
            self.selected_overlay_id = None

    def select_overlay_annotation(self, overlay_id: str | None) -> None:
        self.selected_overlay_id = overlay_id
        if overlay_id is not None:
            self.view_state.selected_measurement_id = None

    def select_text_annotation(self, text_id: str | None) -> None:
        annotation = self.get_text_annotation(text_id)
        self.select_overlay_annotation(annotation.id if annotation is not None else None)

    def add_measurement(self, measurement: Measurement) -> None:
        self.insert_measurement_incremental(measurement, mark_dirty=False)
        self.rebuild_group_memberships()
        self.refresh_dirty_flags()

    def insert_measurement_incremental(
        self,
        measurement: Measurement,
        *,
        index: int | None = None,
        select: bool = True,
        mark_dirty: bool = True,
    ) -> None:
        if measurement.fiber_group_id is None:
            measurement.fiber_group_id = self.active_group_id
        measurement.recalculate(self.calibration)
        if index is None or index < 0 or index >= len(self.measurements):
            self.measurements.append(measurement)
        else:
            self.measurements.insert(index, measurement)
        group = self.get_group(measurement.fiber_group_id)
        if group is not None and measurement.id not in group.measurement_ids:
            group.measurement_ids.append(measurement.id)
        if select:
            self.select_measurement(measurement.id)
        self.mark_measurement_geometry_changed()
        if mark_dirty:
            self.mark_session_dirty()

    def remove_measurement(self, measurement_id: str) -> None:
        self.remove_measurements([measurement_id])

    def remove_measurement_incremental(
        self,
        measurement_id: str,
        *,
        select_measurement_id: str | None = None,
        select_overlay_id: str | None = None,
        mark_dirty: bool = True,
    ) -> Measurement | None:
        removed: Measurement | None = None
        kept: list[Measurement] = []
        for measurement in self.measurements:
            if measurement.id == measurement_id and removed is None:
                removed = measurement
                continue
            kept.append(measurement)
        if removed is None:
            return None
        self.measurements = kept
        group = self.get_group(removed.fiber_group_id)
        if group is not None:
            group.measurement_ids = [
                item
                for item in group.measurement_ids
                if item != measurement_id
            ]
        if select_overlay_id is not None:
            self.select_overlay_annotation(select_overlay_id)
        else:
            self.select_measurement(
                select_measurement_id
                if self.get_measurement(select_measurement_id) is not None
                else None
            )
        self.mark_measurement_geometry_changed()
        if mark_dirty:
            self.mark_session_dirty()
        return removed

    def remove_measurements(self, measurement_ids: list[str] | set[str] | tuple[str, ...]) -> int:
        targets = {measurement_id for measurement_id in measurement_ids if measurement_id}
        if not targets:
            return 0
        original_count = len(self.measurements)
        self.measurements = [
            measurement
            for measurement in self.measurements
            if measurement.id not in targets
        ]
        removed_count = original_count - len(self.measurements)
        if removed_count <= 0:
            return 0
        if self.view_state.selected_measurement_id in targets:
            self.select_measurement(None)
        self.rebuild_group_memberships()
        self.mark_measurement_geometry_changed()
        self.refresh_dirty_flags()
        return removed_count

    def remove_measurements_for_group(self, group_id: str | None) -> int:
        if group_id is None:
            targets = [measurement.id for measurement in self.measurements if measurement.fiber_group_id is None]
        else:
            targets = [measurement.id for measurement in self.measurements if measurement.fiber_group_id == group_id]
        return self.remove_measurements(targets)

    def clear_measurements(self) -> int:
        return self.remove_measurements([measurement.id for measurement in self.measurements])

    def clear_measurements_by_group_label(self, label: str) -> int:
        token = normalize_group_label(label)
        if not token:
            return 0
        group = self.find_group_by_label(token)
        if group is not None:
            return self.remove_measurements_for_group(group.id)
        if token == normalize_group_label(UNCATEGORIZED_LABEL):
            return self.remove_measurements_for_group(None)
        return 0

    def has_measurements_for_group_label(self, label: str) -> bool:
        token = normalize_group_label(label)
        if not token:
            return False
        group = self.find_group_by_label(token)
        if group is not None:
            return any(measurement.fiber_group_id == group.id for measurement in self.measurements)
        if token == normalize_group_label(UNCATEGORIZED_LABEL):
            return any(measurement.fiber_group_id is None for measurement in self.measurements)
        return False

    def measurement_group_labels(self) -> list[str]:
        labels: list[str] = []
        seen: set[str] = set()
        for group in self.sorted_groups():
            token = normalize_group_label(group.label)
            if not token or token in seen or not any(measurement.fiber_group_id == group.id for measurement in self.measurements):
                continue
            seen.add(token)
            labels.append(token)
        if any(measurement.fiber_group_id is None for measurement in self.measurements):
            labels.append(UNCATEGORIZED_LABEL)
        return labels

    def remove_auto_area_measurements(self) -> None:
        auto_ids = {
            measurement.id
            for measurement in self.measurements
            if measurement.measurement_kind == "area" and measurement.mode == "auto_instance"
        }
        if not auto_ids:
            return
        self.remove_measurements(auto_ids)

    def set_measurement_group(self, measurement_id: str, group_id: str | None) -> None:
        measurement = self.get_measurement(measurement_id)
        if measurement is None:
            return
        if group_id is not None and self.get_group(group_id) is None:
            return
        measurement.fiber_group_id = group_id
        self.rebuild_group_memberships()
        self.refresh_dirty_flags()

    def rebuild_group_memberships(self) -> None:
        group_map = {group.id: group for group in self.fiber_groups}
        for group in self.fiber_groups:
            group.measurement_ids = []
        for measurement in self.measurements:
            group = group_map.get(measurement.fiber_group_id or "")
            if group is not None and measurement.id not in group.measurement_ids:
                group.measurement_ids.append(measurement.id)
        self.fiber_groups.sort(key=lambda group: group.number)

    def renumber_groups(self) -> None:
        for index, group in enumerate(self.sorted_groups(), start=1):
            group.number = index
        self.fiber_groups.sort(key=lambda group: group.number)

    def merge_group_into(self, source_group_id: str, target_group_id: str) -> bool:
        if source_group_id == target_group_id:
            return False
        source_group = self.get_group(source_group_id)
        target_group = self.get_group(target_group_id)
        if source_group is None or target_group is None:
            return False
        for measurement in self.measurements:
            if measurement.fiber_group_id == source_group_id:
                measurement.fiber_group_id = target_group_id
        self.fiber_groups = [group for group in self.fiber_groups if group.id != source_group_id]
        if self.active_group_id == source_group_id:
            self.active_group_id = target_group_id
        self.rebuild_group_memberships()
        self.renumber_groups()
        self.refresh_dirty_flags()
        return True

    def add_overlay_annotation(self, annotation: OverlayAnnotation) -> None:
        self.overlay_annotations.append(annotation)
        self.select_overlay_annotation(annotation.id)
        self.refresh_dirty_flags()

    def add_text_annotation(self, annotation: TextAnnotation) -> None:
        self.add_overlay_annotation(annotation.to_overlay())

    def replace_overlay_annotation(self, overlay_id: str, replacement: OverlayAnnotation) -> None:
        for index, annotation in enumerate(self.overlay_annotations):
            if annotation.id == overlay_id:
                self.overlay_annotations[index] = replacement
                self.select_overlay_annotation(replacement.id)
                self.refresh_dirty_flags()
                return

    def move_overlay_annotation(self, overlay_id: str, dx: float, dy: float) -> None:
        annotation = self.get_overlay_annotation(overlay_id)
        if annotation is None:
            return
        self.replace_overlay_annotation(overlay_id, annotation.translated(dx, dy))

    def move_text_annotation(self, text_id: str, anchor_px: Point) -> None:
        annotation = self.get_text_annotation(text_id)
        if annotation is None:
            return
        self.replace_overlay_annotation(text_id, annotation.clone(anchor_px=anchor_px))

    def remove_overlay_annotation(self, overlay_id: str) -> None:
        self.overlay_annotations = [
            annotation for annotation in self.overlay_annotations
            if annotation.id != overlay_id
        ]
        if self.selected_overlay_id == overlay_id:
            self.select_overlay_annotation(None)
        self.refresh_dirty_flags()

    def remove_text_annotation(self, text_id: str) -> None:
        self.remove_overlay_annotation(text_id)

    def update_overlay_annotation_geometry(
        self,
        overlay_id: str,
        *,
        anchor_px: Point | None = None,
        start_px: Point | None = None,
        end_px: Point | None = None,
    ) -> None:
        annotation = self.get_overlay_annotation(overlay_id)
        if annotation is None:
            return
        replacement = annotation.clone(
            anchor_px=anchor_px if anchor_px is not None else annotation.anchor_px,
            start_px=start_px if start_px is not None else annotation.start_px,
            end_px=end_px if end_px is not None else annotation.end_px,
        )
        self.replace_overlay_annotation(overlay_id, replacement)

    def remove_group_to_uncategorized(self, group_id: str) -> bool:
        group = self.get_group(group_id)
        if group is None:
            return False
        moved_measurements = False
        for measurement in self.measurements:
            if measurement.fiber_group_id == group_id:
                measurement.fiber_group_id = None
                moved_measurements = True
        self.fiber_groups = [item for item in self.fiber_groups if item.id != group_id]
        if self.active_group_id == group_id:
            self.active_group_id = None if moved_measurements else (self.sorted_groups()[0].id if self.fiber_groups else None)
        self.rebuild_group_memberships()
        self.renumber_groups()
        self.refresh_dirty_flags()
        return True

    def hide_uncategorized_entry(self) -> bool:
        if not self.can_delete_uncategorized_entry():
            return False
        if self.active_group_id is None:
            self.active_group_id = self.sorted_groups()[0].id
        self.refresh_dirty_flags()
        return True

    def is_project_group_label_suppressed(self, label: str) -> bool:
        token = normalize_group_label(label)
        return bool(token) and token in self.suppressed_project_group_labels

    def suppress_project_group_label(self, label: str) -> bool:
        token = normalize_group_label(label)
        if not token or token in self.suppressed_project_group_labels:
            return False
        self.suppressed_project_group_labels.append(token)
        self.suppressed_project_group_labels.sort()
        self.refresh_dirty_flags()
        return True

    def unsuppress_project_group_label(self, label: str) -> bool:
        token = normalize_group_label(label)
        if not token or token not in self.suppressed_project_group_labels:
            return False
        self.suppressed_project_group_labels = [
            item
            for item in self.suppressed_project_group_labels
            if item != token
        ]
        self.refresh_dirty_flags()
        return True

    @staticmethod
    def _normalized_suppressed_project_group_labels(labels: list[str]) -> list[str]:
        return sorted({token for token in (normalize_group_label(item) for item in labels) if token})

    def measurement_values(self) -> list[float]:
        return [
            measurement.diameter_unit
            for measurement in self.line_measurements()
            if measurement.diameter_unit is not None
        ]

    def area_values(self) -> list[float]:
        return [
            measurement.area_unit
            for measurement in self.area_measurements()
            if measurement.area_unit is not None
        ]

    def stats(self) -> dict[str, float | None]:
        values = self.measurement_values()
        if not values:
            return {
                "mean": None,
                "min": None,
                "max": None,
                "stddev": None,
            }
        return {
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
            "stddev": statistics.pstdev(values) if len(values) > 1 else 0.0,
        }

    def recalculate_measurements(self) -> None:
        for measurement in self.measurements:
            measurement.recalculate(self.calibration)
        self.refresh_dirty_flags()

    def session_snapshot(self) -> dict[str, Any]:
        return {
            "fiber_groups": [group.to_dict() for group in self.sorted_groups()],
            "measurements": [measurement.to_dict() for measurement in self.measurements],
            "overlay_annotations": [annotation.to_dict() for annotation in self.overlay_annotations],
            "scale_overlay_anchor": self.scale_overlay_anchor.to_dict() if self.scale_overlay_anchor else None,
            "suppressed_project_group_labels": list(self.suppressed_project_group_labels),
        }

    def calibration_snapshot(self) -> dict[str, Any]:
        calibration_line = self.metadata.get("calibration_line")
        return {
            "calibration": self.calibration.to_dict() if self.calibration else None,
            "calibration_line": calibration_line,
        }

    def snapshot_state(self) -> dict[str, Any]:
        return {
            "calibration": self.calibration.to_dict() if self.calibration else None,
            "fiber_groups": [group.to_dict() for group in self.sorted_groups()],
            "measurements": [measurement.to_dict() for measurement in self.measurements],
            "overlay_annotations": [annotation.to_dict() for annotation in self.overlay_annotations],
            "metadata": dict(self.metadata),
            "active_group_id": self.active_group_id,
            "selected_measurement_id": self.view_state.selected_measurement_id,
            "selected_overlay_id": self.selected_overlay_id,
            "scale_overlay_anchor": self.scale_overlay_anchor.to_dict() if self.scale_overlay_anchor else None,
            "suppressed_project_group_labels": list(self.suppressed_project_group_labels),
        }

    def restore_snapshot(self, snapshot: dict[str, Any]) -> None:
        self.calibration = Calibration.from_dict(snapshot["calibration"]) if snapshot.get("calibration") else None
        self.fiber_groups = [
            FiberGroup.from_dict(item, fallback_number=index + 1)
            for index, item in enumerate(snapshot.get("fiber_groups", []))
        ]
        self.measurements = [
            Measurement.from_dict(item)
            for item in snapshot.get("measurements", [])
        ]
        overlay_payload = snapshot.get("overlay_annotations")
        if isinstance(overlay_payload, list):
            self.overlay_annotations = [
                OverlayAnnotation.from_dict(item)
                for item in overlay_payload
                if isinstance(item, dict)
            ]
        else:
            self.overlay_annotations = [
                TextAnnotation.from_dict(item).to_overlay()
                for item in snapshot.get("text_annotations", [])
                if isinstance(item, dict)
            ]
        self.metadata = dict(snapshot.get("metadata", {}))
        self.active_group_id = snapshot.get("active_group_id")
        self.view_state.selected_measurement_id = snapshot.get("selected_measurement_id")
        self.selected_overlay_id = snapshot.get("selected_overlay_id", snapshot.get("selected_text_id"))
        scale_overlay_anchor = snapshot.get("scale_overlay_anchor")
        self.scale_overlay_anchor = Point.from_dict(scale_overlay_anchor) if scale_overlay_anchor else None
        self.suppressed_project_group_labels = self._normalized_suppressed_project_group_labels(
            list(snapshot.get("suppressed_project_group_labels", []))
        )
        self.rebuild_group_memberships()
        if self.active_group_id is None or self.get_group(self.active_group_id) is None:
            self.active_group_id = self.fiber_groups[0].id if self.fiber_groups else None
        if self.view_state.selected_measurement_id and self.get_measurement(self.view_state.selected_measurement_id) is None:
            self.view_state.selected_measurement_id = None
        if self.selected_overlay_id and self.get_overlay_annotation(self.selected_overlay_id) is None:
            self.selected_overlay_id = None
        self.mark_measurement_geometry_changed()
        self.refresh_dirty_flags()

    def mark_session_saved(self) -> None:
        self._session_clean_snapshot = self.session_snapshot()
        self.refresh_dirty_flags()

    def mark_calibration_saved(self) -> None:
        self._calibration_clean_snapshot = self.calibration_snapshot()
        self.refresh_dirty_flags()

    def mark_session_dirty(self) -> None:
        self.dirty_flags = DirtyFlags(
            session_dirty=True,
            calibration_dirty=self.dirty_flags.calibration_dirty,
        )

    def refresh_dirty_flags(self) -> None:
        session_dirty = self._session_clean_snapshot is not None and self.session_snapshot() != self._session_clean_snapshot
        calibration_dirty = self._calibration_clean_snapshot is not None and self.calibration_snapshot() != self._calibration_clean_snapshot
        self.dirty_flags = DirtyFlags(
            session_dirty=session_dirty,
            calibration_dirty=calibration_dirty,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "path": self.path,
            "source_type": self.source_type,
            "image_size": list(self.image_size),
            "calibration": self.calibration.to_dict() if self.calibration else None,
            "fiber_groups": [group.to_dict() for group in self.sorted_groups()],
            "measurements": [measurement.to_dict() for measurement in self.measurements],
            "overlay_annotations": [annotation.to_dict() for annotation in self.overlay_annotations],
            "view_state": self.view_state.to_dict(),
            "metadata": self.metadata,
            "active_group_id": self.active_group_id,
            "selected_overlay_id": self.selected_overlay_id,
            "scale_overlay_anchor": self.scale_overlay_anchor.to_dict() if self.scale_overlay_anchor else None,
            "suppressed_project_group_labels": list(self.suppressed_project_group_labels),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ImageDocument":
        overlay_payload = payload.get("overlay_annotations")
        overlay_annotations = [
            OverlayAnnotation.from_dict(item)
            for item in overlay_payload
            if isinstance(item, dict)
        ] if isinstance(overlay_payload, list) else [
            TextAnnotation.from_dict(item).to_overlay()
            for item in payload.get("text_annotations", [])
            if isinstance(item, dict)
        ]
        image_document = cls(
            id=str(payload["id"]),
            path=str(payload["path"]),
            source_type=str(payload.get("source_type", "filesystem")),
            image_size=(int(payload["image_size"][0]), int(payload["image_size"][1])),
            calibration=Calibration.from_dict(payload["calibration"]) if payload.get("calibration") else None,
            fiber_groups=[
                FiberGroup.from_dict(item, fallback_number=index + 1)
                for index, item in enumerate(payload.get("fiber_groups", []))
            ],
            measurements=[Measurement.from_dict(item) for item in payload.get("measurements", [])],
            overlay_annotations=overlay_annotations,
            view_state=ImageViewState.from_dict(payload.get("view_state", {})),
            metadata=dict(payload.get("metadata", {})),
            active_group_id=payload.get("active_group_id"),
            selected_overlay_id=payload.get("selected_overlay_id", payload.get("selected_text_id")),
            scale_overlay_anchor=Point.from_dict(payload["scale_overlay_anchor"]) if payload.get("scale_overlay_anchor") else None,
            suppressed_project_group_labels=list(payload.get("suppressed_project_group_labels", [])),
        )
        image_document.initialize_runtime_state()
        return image_document


@dataclass(slots=True)
class ProjectState:
    version: str
    documents: list[ImageDocument]
    calibration_presets: list[CalibrationPreset] = field(default_factory=list)
    project_default_calibration: Calibration | None = None
    project_group_templates: list[ProjectGroupTemplate] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_document(self, document_id: str) -> ImageDocument | None:
        for document in self.documents:
            if document.id == document_id:
                return document
        return None

    def to_dict(self) -> dict[str, Any]:
        seen_template_labels: set[str] = set()
        serialized_templates: list[dict[str, Any]] = []
        for template in self.project_group_templates:
            token = template.normalized_label()
            if not token or token in seen_template_labels:
                continue
            seen_template_labels.add(token)
            serialized_templates.append(template.to_dict())
        return {
            "version": self.version,
            "documents": [document.to_dict() for document in self.documents],
            "project_default_calibration": self.project_default_calibration.to_dict() if self.project_default_calibration else None,
            "project_group_templates": serialized_templates,
            "metadata": self.metadata,
        }

    @classmethod
    def empty(cls) -> "ProjectState":
        return cls(version=__version__, documents=[])

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ProjectState":
        seen_template_labels: set[str] = set()
        project_group_templates: list[ProjectGroupTemplate] = []
        for item in payload.get("project_group_templates", []):
            template = ProjectGroupTemplate.from_dict(item)
            token = template.normalized_label()
            if not token or token in seen_template_labels:
                continue
            seen_template_labels.add(token)
            project_group_templates.append(template)
        return cls(
            version=str(payload.get("version", __version__)),
            documents=[ImageDocument.from_dict(item) for item in payload.get("documents", [])],
            calibration_presets=[
                CalibrationPreset.from_dict(item)
                for item in payload.get("calibration_presets", [])
            ],
            project_default_calibration=Calibration.from_dict(payload["project_default_calibration"]) if payload.get("project_default_calibration") else None,
            project_group_templates=project_group_templates,
            metadata=dict(payload.get("metadata", {})),
        )
