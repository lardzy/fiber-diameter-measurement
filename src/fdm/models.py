from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping
import hashlib
import json
import math
import statistics
import uuid

import numpy as np

from fdm.analysis_artifacts import (
    AnalysisArtifact,
    AnalysisDependencySignature,
    AnalysisSourceDescriptor,
    calibration_signature_from_values,
    refresh_artifacts_validity,
)
from fdm.construction_geometry import ConstructionEntity
from fdm.geometry import (
    Line,
    Point,
    area_rings_area,
    line_length,
    midpoint,
    polygon_area,
    polygon_centroid,
    polyline_centroid,
    polyline_length,
)
from fdm.image_processing_models import (
    DisplayTransform,
    ImageDerivation,
    RasterSemantic,
    RasterTypeState,
)
from fdm.project_roi import (
    ProjectRoi,
    ProjectRoiDeletionResult,
    remove_rois_with_dependents,
)
from fdm.raster import RasterPixelType
from fdm.version import __version__

UNCATEGORIZED_LABEL = "未分类"
UNCATEGORIZED_COLOR = "#98A2B3"

PROJECT_SCHEMA_VERSION = 2
PROJECT_MIN_READER_VERSION = 2
SUPPORTED_PROJECT_REQUIRED_FEATURES = frozenset(
    {
        "analysis-artifacts/v1",
        "analysis-artifacts/v2",
        "construction-geometry/v1",
        "project-rois/v1",
    }
)


@dataclass(frozen=True, slots=True)
class ProjectCompatibilityState:
    """Runtime compatibility decision for a loaded project file."""

    source_schema_version: int = PROJECT_SCHEMA_VERSION
    min_reader_version: int = PROJECT_MIN_READER_VERSION
    required_features: tuple[str, ...] = ()
    unknown_required_features: tuple[str, ...] = ()
    source_path: str | None = None

    @property
    def read_only(self) -> bool:
        return bool(self.unknown_required_features)

    @property
    def overwrite_allowed(self) -> bool:
        return not self.read_only

    @property
    def requires_upgrade(self) -> bool:
        return self.source_schema_version < PROJECT_SCHEMA_VERSION

    def can_overwrite(self, path: str | Path | None = None) -> bool:
        if not self.read_only:
            return True
        if path is None or self.source_path is None:
            return False
        try:
            return Path(path).expanduser().resolve() != Path(
                self.source_path
            ).expanduser().resolve()
        except OSError:
            return str(path) != self.source_path


def utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


_DEBUG_PAYLOAD_MAX_DEPTH = 64


def _json_safe_debug_key(value: object) -> str | None:
    """Return a deterministic JSON object key, or reject an unsafe key."""

    if isinstance(value, np.generic):
        if isinstance(value, np.bool_):
            value = bool(value)
        elif isinstance(value, np.integer):
            value = int(value)
        elif isinstance(value, np.floating):
            value = float(value)
        elif isinstance(value, np.str_):
            value = str(value)
        else:
            return None
    if isinstance(value, str):
        return value
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float) and math.isfinite(value):
        return str(value)
    return None


def _json_safe_debug_value(
    value: object,
    *,
    active_container_ids: set[int] | None = None,
    depth: int = 0,
) -> object:
    """Normalize diagnostic data without inventing scientific values.

    Debug payloads are supplied by image-processing code and may contain NumPy
    scalar values, non-finite intermediates or accidentally retained runtime
    objects.  Project JSON must remain strict.  Finite scalar values and the
    list/dict shape are preserved, while values that have no truthful JSON
    representation become ``None``.  Container cycles and excessive nesting
    are truncated in the same explicit way instead of blocking project save.
    """

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.generic):
        if isinstance(value, np.bool_):
            return bool(value)
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            scalar = float(value)
            return scalar if math.isfinite(scalar) else None
        if isinstance(value, np.str_):
            return str(value)
        return None
    if depth >= _DEBUG_PAYLOAD_MAX_DEPTH:
        return None

    active_ids = active_container_ids if active_container_ids is not None else set()
    if isinstance(value, dict):
        identity = id(value)
        if identity in active_ids:
            return None
        active_ids.add(identity)
        try:
            normalized: dict[str, object] = {}
            for key, item in value.items():
                normalized_key = _json_safe_debug_key(key)
                if normalized_key is None:
                    continue
                normalized[normalized_key] = _json_safe_debug_value(
                    item,
                    active_container_ids=active_ids,
                    depth=depth + 1,
                )
            return normalized
        finally:
            active_ids.remove(identity)
    if isinstance(value, (list, tuple)):
        identity = id(value)
        if identity in active_ids:
            return None
        active_ids.add(identity)
        try:
            return [
                _json_safe_debug_value(
                    item,
                    active_container_ids=active_ids,
                    depth=depth + 1,
                )
                for item in value
            ]
        finally:
            active_ids.remove(identity)
    return None


def _json_safe_debug_payload(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        return {}
    normalized = _json_safe_debug_value(value)
    return normalized if isinstance(normalized, dict) else {}


def project_assets_root(project_path: str | Path) -> Path:
    return Path(project_path).with_suffix(".assets")


def project_capture_root(project_path: str | Path) -> Path:
    return project_assets_root(project_path) / "captures"


def project_processed_root(project_path: str | Path) -> Path:
    return project_assets_root(project_path) / "processed"


def project_slide_root(project_path: str | Path) -> Path:
    return project_assets_root(project_path) / "slides"


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


def require_positive_finite(value: float, *, field_name: str = "pixels_per_unit") -> float:
    """Return a normalized scale value or reject values unsafe for math/JSON."""
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"{field_name} 必须是大于 0 的有限数值")
    return normalized


def _normalize_appearance_color(value: object) -> str | None:
    token = str(value or "").strip()
    if not token.startswith("#"):
        return None
    hex_value = token[1:]
    if len(hex_value) == 3:
        hex_value = "".join(character * 2 for character in hex_value)
    if len(hex_value) != 6:
        return None
    if any(character not in "0123456789abcdefABCDEF" for character in hex_value):
        return None
    return f"#{hex_value.upper()}"


def _normalize_optional_finite(
    value: object,
    *,
    minimum: float,
    maximum: float,
) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return max(minimum, min(maximum, numeric))


@dataclass(slots=True)
class ObjectAppearanceOverride:
    """Optional per-object visual values layered over category/application defaults."""

    stroke_color: str | None = None
    stroke_width: float | None = None
    text_color: str | None = None
    font_family: str | None = None
    font_size: int | None = None
    marker_scale: float | None = None

    def __post_init__(self) -> None:
        self.stroke_color = _normalize_appearance_color(self.stroke_color)
        self.stroke_width = _normalize_optional_finite(
            self.stroke_width,
            minimum=0.5,
            maximum=24.0,
        )
        self.text_color = _normalize_appearance_color(self.text_color)
        family = str(self.font_family or "").strip()
        self.font_family = family[:128] or None
        normalized_font_size = _normalize_optional_finite(
            self.font_size,
            minimum=8.0,
            maximum=144.0,
        )
        self.font_size = int(round(normalized_font_size)) if normalized_font_size is not None else None
        self.marker_scale = _normalize_optional_finite(
            self.marker_scale,
            minimum=0.25,
            maximum=4.0,
        )

    def is_empty(self) -> bool:
        return all(
            value is None
            for value in (
                self.stroke_color,
                self.stroke_width,
                self.text_color,
                self.font_family,
                self.font_size,
                self.marker_scale,
            )
        )

    def clone(self, **changes: object) -> "ObjectAppearanceOverride":
        return replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for key in (
            "stroke_color",
            "stroke_width",
            "text_color",
            "font_family",
            "font_size",
            "marker_scale",
        ):
            value = getattr(self, key)
            if value is not None:
                payload[key] = value
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ObjectAppearanceOverride":
        return cls(
            stroke_color=payload.get("stroke_color"),
            stroke_width=payload.get("stroke_width"),
            text_color=payload.get("text_color"),
            font_family=payload.get("font_family"),
            font_size=payload.get("font_size"),
            marker_scale=payload.get("marker_scale"),
        )


def _appearance_from_payload(payload: object) -> ObjectAppearanceOverride | None:
    if not isinstance(payload, dict):
        return None
    appearance = ObjectAppearanceOverride.from_dict(payload)
    return None if appearance.is_empty() else appearance


@dataclass(slots=True)
class Calibration:
    mode: str
    pixels_per_unit: float
    unit: str
    source_label: str

    def __post_init__(self) -> None:
        self.pixels_per_unit = require_positive_finite(self.pixels_per_unit)

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
        return value_px / self.pixels_per_unit

    def unit_to_px(self, value: float) -> float:
        return value * self.pixels_per_unit

    def px_area_to_unit(self, value_px: float) -> float:
        return value_px / (self.pixels_per_unit ** 2)


@dataclass(slots=True)
class CalibrationPreset:
    name: str
    pixels_per_unit: float
    unit: str
    pixel_distance: float | None = None
    actual_distance: float | None = None
    computed_pixels_per_unit: float | None = None

    def __post_init__(self) -> None:
        self.pixels_per_unit = require_positive_finite(self.pixels_per_unit)
        if self.computed_pixels_per_unit is not None:
            self.computed_pixels_per_unit = require_positive_finite(
                self.computed_pixels_per_unit,
                field_name="computed_pixels_per_unit",
            )
        if self.pixel_distance is not None:
            self.pixel_distance = require_positive_finite(
                self.pixel_distance,
                field_name="pixel_distance",
            )
        if self.actual_distance is not None:
            self.actual_distance = require_positive_finite(
                self.actual_distance,
                field_name="actual_distance",
            )

    def resolved_pixels_per_unit(self) -> float:
        if self.computed_pixels_per_unit is not None:
            return self.computed_pixels_per_unit
        return self.pixels_per_unit

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
        pixels_per_unit = float(
            computed_pixels_per_unit
            if computed_pixels_per_unit is not None
            else payload["pixels_per_unit"]
        )
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


class DirtyDomain(str, Enum):
    """Independent persistence domains tracked by the runtime savepoint."""

    SESSION = "session"
    CALIBRATION = "calibration"


@dataclass(frozen=True, slots=True)
class DocumentStateStamp:
    """O(1) dirty-state identity restored verbatim by undo/redo.

    The values are runtime-only monotonic identities, not project revisions and
    therefore deliberately never serialized into project or sidecar files.
    """

    session_state_id: int = 0
    calibration_state_id: int = 0


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
    appearance: ObjectAppearanceOverride | None = None
    display_polygon_px: list[Point] = field(default_factory=list, repr=False)
    display_area_rings_px: list[list[Point]] = field(default_factory=list, repr=False)
    display_bounds_px: tuple[float, float, float, float] | None = field(default=None, repr=False)
    _geometry_revision: int = field(default=0, init=False, repr=False, compare=False)

    @property
    def geometry_revision(self) -> int:
        """Runtime-only revision for geometry-derived caches.

        The revision is deliberately excluded from project serialization.  A
        freshly loaded measurement starts at revision zero and receives a new
        cache namespace through its object identity.
        """

        return self._geometry_revision

    @staticmethod
    def _copy_point(point: Point) -> Point:
        return Point(float(point.x), float(point.y))

    @classmethod
    def _copy_line(cls, line: Line | None) -> Line | None:
        if line is None:
            return None
        return Line(cls._copy_point(line.start), cls._copy_point(line.end))

    @classmethod
    def _copy_points(cls, points: list[Point]) -> list[Point]:
        return [cls._copy_point(point) for point in points]

    @classmethod
    def _copy_rings(cls, rings: list[list[Point]]) -> list[list[Point]]:
        return [cls._copy_points(ring) for ring in rings]

    def _advance_geometry_revision(self) -> None:
        self._geometry_revision += 1
        # Keep the legacy display fields empty after production mutations.
        # The bounded derived-geometry service owns all new runtime caches.
        self.display_polygon_px = []
        self.display_area_rings_px = []
        self.display_bounds_px = None

    def replace_area_geometry(
        self,
        *,
        polygon_px: list[Point],
        area_rings_px: list[list[Point]] | None = None,
        exact_area_px: float | None = None,
        calibration: Calibration | None = None,
    ) -> None:
        """Atomically replace area geometry using private coordinate copies."""

        if self.measurement_kind != "area":
            self.measurement_kind = "area"
        copied_polygon = self._copy_points(polygon_px)
        copied_rings = self._copy_rings(area_rings_px or [])
        if len(copied_polygon) < 3 and copied_rings:
            copied_polygon = self._copy_points(copied_rings[0])
        self.polygon_px = copied_polygon
        self.area_rings_px = copied_rings
        self.exact_area_px = float(exact_area_px) if exact_area_px is not None else None
        self._advance_geometry_revision()
        self.recalculate(calibration)

    def replace_line_geometry(
        self,
        *,
        line_px: Line,
        snapped_line_px: Line | None = None,
        calibration: Calibration | None = None,
    ) -> None:
        """Atomically replace a straight-line measurement's geometry."""

        self.measurement_kind = "line"
        copied_line = self._copy_line(line_px)
        if copied_line is None:  # pragma: no cover - guarded by the signature
            raise ValueError("line_px is required")
        self.line_px = copied_line
        self.snapped_line_px = self._copy_line(snapped_line_px)
        self._advance_geometry_revision()
        self.recalculate(calibration)

    def replace_polyline_geometry(
        self,
        *,
        polyline_px: list[Point],
        calibration: Calibration | None = None,
    ) -> None:
        """Atomically replace continuous length geometry."""

        self.measurement_kind = "polyline"
        self.polyline_px = self._copy_points(polyline_px)
        self._advance_geometry_revision()
        self.recalculate(calibration)

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
        if self.measurement_kind == "area" and (
            self.area_rings_px or len(self.polygon_px) >= 3
        ):
            # Import lazily to keep the model independent from Qt cache setup
            # during module initialization.  The cached value is always
            # derived from RAW geometry and keyed by this measurement's
            # geometry revision.  Polygon-only inference objects use the same
            # path; requesting a label center does not build the independent
            # hole-area cache.
            from fdm.area_display import area_derived_geometry_service

            return area_derived_geometry_service.centroid(self)
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
        payload = {
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
            "debug_payload": _json_safe_debug_payload(self.debug_payload),
        }
        if self.appearance is not None and not self.appearance.is_empty():
            payload["appearance"] = self.appearance.to_dict()
        return payload

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
            debug_payload=_json_safe_debug_payload(payload.get("debug_payload", {})),
            appearance=_appearance_from_payload(payload.get("appearance")),
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


class OverlayTextAnchorAlignment:
    """Anchor point placement within an overlay text layout box."""

    TOP_LEFT = "top_left"
    TOP_CENTER = "top_center"
    TOP_RIGHT = "top_right"
    CENTER_LEFT = "center_left"
    CENTER = "center"
    CENTER_RIGHT = "center_right"
    BOTTOM_LEFT = "bottom_left"
    BOTTOM_CENTER = "bottom_center"
    BOTTOM_RIGHT = "bottom_right"

    # Readable aliases for callers that describe the second row as "middle".
    MIDDLE_LEFT = CENTER_LEFT
    MIDDLE_CENTER = CENTER
    MIDDLE_RIGHT = CENTER_RIGHT

    @classmethod
    def normalize(cls, value: object) -> str:
        token = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            cls.TOP_LEFT: cls.TOP_LEFT,
            cls.TOP_CENTER: cls.TOP_CENTER,
            cls.TOP_RIGHT: cls.TOP_RIGHT,
            cls.CENTER_LEFT: cls.CENTER_LEFT,
            cls.CENTER: cls.CENTER,
            cls.CENTER_RIGHT: cls.CENTER_RIGHT,
            cls.BOTTOM_LEFT: cls.BOTTOM_LEFT,
            cls.BOTTOM_CENTER: cls.BOTTOM_CENTER,
            cls.BOTTOM_RIGHT: cls.BOTTOM_RIGHT,
            "middle_left": cls.CENTER_LEFT,
            "middle": cls.CENTER,
            "middle_center": cls.CENTER,
            "middle_right": cls.CENTER_RIGHT,
        }
        return aliases.get(token, cls.CENTER)


class OverlayTextSizeSpace:
    """Coordinate space used to interpret an overlay text font size."""

    LEGACY_OUTPUT_PX = "legacy_output_px"
    IMAGE_PX = "image_px"

    @classmethod
    def normalize(cls, value: object) -> str:
        token = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
        if token == cls.LEGACY_OUTPUT_PX:
            return cls.LEGACY_OUTPUT_PX
        return cls.IMAGE_PX


@dataclass(slots=True)
class OverlayTextLayoutSpec:
    """Explicit placement and size semantics for a text overlay.

    A missing spec on :class:`OverlayAnnotation` retains the legacy fixed-output
    pixel behavior.  New annotations use image-space sizing and a centered
    anchor unless the caller requests otherwise.
    """

    anchor_alignment: str = OverlayTextAnchorAlignment.CENTER
    size_space: str = OverlayTextSizeSpace.IMAGE_PX
    image_font_size_px: float = 18.0

    def __post_init__(self) -> None:
        self.anchor_alignment = OverlayTextAnchorAlignment.normalize(self.anchor_alignment)
        self.size_space = OverlayTextSizeSpace.normalize(self.size_space)
        normalized_size = _normalize_optional_finite(
            self.image_font_size_px,
            minimum=1.0,
            maximum=8192.0,
        )
        self.image_font_size_px = 18.0 if normalized_size is None else float(normalized_size)

    def to_dict(self) -> dict[str, Any]:
        return {
            "anchor_alignment": self.anchor_alignment,
            "size_space": self.size_space,
            "image_font_size_px": self.image_font_size_px,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "OverlayTextLayoutSpec":
        return cls(
            anchor_alignment=payload.get("anchor_alignment", OverlayTextAnchorAlignment.CENTER),
            size_space=payload.get("size_space", OverlayTextSizeSpace.IMAGE_PX),
            image_font_size_px=payload.get("image_font_size_px", 18.0),
        )


def _overlay_text_layout_from_payload(payload: object) -> OverlayTextLayoutSpec | None:
    if not isinstance(payload, dict):
        return None
    return OverlayTextLayoutSpec.from_dict(payload)


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
    appearance: ObjectAppearanceOverride | None = None
    text_layout: OverlayTextLayoutSpec | None = None

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
            if self.text_layout is not None:
                payload["text_layout"] = self.text_layout.to_dict()
        else:
            payload["start_px"] = self.start_px.to_dict()
            payload["end_px"] = self.end_px.to_dict()
        if self.appearance is not None and not self.appearance.is_empty():
            payload["appearance"] = self.appearance.to_dict()
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
                appearance=_appearance_from_payload(payload.get("appearance")),
                text_layout=_overlay_text_layout_from_payload(payload.get("text_layout")),
            )
        return cls(
            id=str(payload["id"]),
            image_id=str(payload["image_id"]),
            kind=kind,
            start_px=Point.from_dict(payload.get("start_px", {"x": 0.0, "y": 0.0})),
            end_px=Point.from_dict(payload.get("end_px", {"x": 0.0, "y": 0.0})),
            created_at=str(payload.get("created_at", utc_now_iso())),
            appearance=_appearance_from_payload(payload.get("appearance")),
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
    document_kind: str = "image"
    absolute_path: str | None = None
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
    calibration_load_error: str | None = field(default=None, repr=False, compare=False)
    calibration_load_payload: dict[str, Any] | None = field(default=None, repr=False, compare=False)
    dirty_flags: DirtyFlags = field(default_factory=DirtyFlags)
    history: Any = field(default=None, repr=False, compare=False)
    raster_pixel_type: RasterPixelType | None = None
    display_transform: DisplayTransform | None = None
    derivation: ImageDerivation | None = None
    # Keep the legacy positional order through ``derivation`` intact.
    raster_semantic: RasterSemantic | None = None
    construction_entities: list[ConstructionEntity] = field(default_factory=list)
    selected_construction_id: str | None = None
    _current_state_stamp: DocumentStateStamp = field(
        default_factory=DocumentStateStamp,
        init=False,
        repr=False,
        compare=False,
    )
    _saved_state_stamp: DocumentStateStamp | None = field(default=None, init=False, repr=False, compare=False)
    _saved_calibration_signature: tuple[object, ...] | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )
    _next_state_id: int = field(default=1, init=False, repr=False, compare=False)
    _measurement_geometry_revision: int = field(default=0, init=False, repr=False, compare=False)
    _construction_geometry_revision: int = field(default=0, init=False, repr=False, compare=False)
    _construction_metadata_revision: int = field(default=0, init=False, repr=False, compare=False)

    def initialize_runtime_state(self) -> None:
        from fdm.history import DocumentHistory

        if self.history is None:
            self.history = DocumentHistory(owner=self)
        else:
            self.history.bind_document(self)
        if self.sidecar_path is None and self.path and self.uses_sidecar():
            self.sidecar_path = self.default_sidecar_path()
        self.fiber_groups.sort(key=lambda group: group.number)
        self.suppressed_project_group_labels = self._normalized_suppressed_project_group_labels(self.suppressed_project_group_labels)
        self.rebuild_group_memberships()
        if self.active_group_id is None or self.get_group(self.active_group_id) is None:
            self.active_group_id = self.fiber_groups[0].id if self.fiber_groups else None
        if self.selected_overlay_id and self.get_overlay_annotation(self.selected_overlay_id) is None:
            self.selected_overlay_id = None
        if (
            self.selected_construction_id
            and self.get_construction_entity(self.selected_construction_id) is None
        ):
            self.selected_construction_id = None
        if self.selected_construction_id is not None:
            self.view_state.selected_measurement_id = None
            self.selected_overlay_id = None
        if self._saved_state_stamp is None:
            self._saved_state_stamp = self._current_state_stamp
        if self._saved_calibration_signature is None:
            self._saved_calibration_signature = self.calibration_signature()
        self.refresh_dirty_flags()

    def default_sidecar_path(self) -> str:
        return f"{self.resolved_path()}.fdm.json"

    def uses_sidecar(self) -> bool:
        return self.document_kind == "image" and self.source_type == "filesystem" and bool(str(self.path).strip())

    def is_project_asset(self) -> bool:
        return self.source_type == "project_asset"

    def is_digital_slide(self) -> bool:
        return self.document_kind == "digital_slide"

    def resolved_path(self, project_path: str | Path | None = None) -> Path:
        token = str(self.path or "").strip()
        if not token:
            return Path()
        if self.is_project_asset():
            base = project_assets_root(project_path) if project_path is not None else Path()
            return (base / token).resolve() if base else Path(token)
        image_path = Path(token).expanduser()
        if image_path.is_absolute():
            return image_path.resolve()
        if project_path is not None:
            return (Path(project_path).expanduser().resolve().parent / image_path).resolve()
        return image_path.resolve()

    def sorted_groups(self) -> list[FiberGroup]:
        return sorted(self.fiber_groups, key=lambda group: group.number)

    def uncategorized_measurements(self) -> list[Measurement]:
        return [measurement for measurement in self.measurements if measurement.fiber_group_id is None]

    def line_measurements(self) -> list[Measurement]:
        return [measurement for measurement in self.measurements if measurement.measurement_kind == "line"]

    def polyline_measurements(self) -> list[Measurement]:
        return [measurement for measurement in self.measurements if measurement.measurement_kind == "polyline"]

    def length_measurements(self) -> list[Measurement]:
        return [
            measurement
            for measurement in self.measurements
            if measurement.measurement_kind in {"line", "polyline"}
        ]

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

    def get_construction_entity(
        self,
        construction_id: str | None,
    ) -> ConstructionEntity | None:
        if construction_id is None:
            return None
        for entity in self.construction_entities:
            if entity.id == construction_id:
                return entity
        return None

    @property
    def construction_geometry_revision(self) -> int:
        return self._construction_geometry_revision

    def mark_construction_geometry_changed(self) -> None:
        self._construction_geometry_revision += 1

    @property
    def construction_metadata_revision(self) -> int:
        return self._construction_metadata_revision

    def mark_construction_metadata_changed(self) -> None:
        self._construction_metadata_revision += 1

    def select_construction(self, construction_id: str | None) -> None:
        entity = self.get_construction_entity(construction_id)
        self.selected_construction_id = entity.id if entity is not None else None
        if entity is not None:
            self.view_state.selected_measurement_id = None
            self.selected_overlay_id = None

    def add_construction_entity(
        self,
        entity: ConstructionEntity,
        *,
        select: bool = True,
        mark_dirty: bool = True,
    ) -> None:
        if self.get_construction_entity(entity.id) is not None:
            raise ValueError(f"辅助几何包含重复 ID: {entity.id}")
        self.construction_entities.append(entity)
        if select:
            self.select_construction(entity.id)
        self.mark_construction_geometry_changed()
        if mark_dirty:
            self.mark_session_dirty()

    def remove_construction_entities(
        self,
        construction_ids: Iterable[str],
        *,
        mark_dirty: bool = True,
    ) -> int:
        targets = {str(item) for item in construction_ids if item}
        if not targets:
            return 0
        original_count = len(self.construction_entities)
        self.construction_entities = [
            entity
            for entity in self.construction_entities
            if entity.id not in targets
        ]
        removed_count = original_count - len(self.construction_entities)
        if removed_count <= 0:
            return 0
        if self.selected_construction_id in targets:
            self.selected_construction_id = None
        self.mark_construction_geometry_changed()
        if mark_dirty:
            self.mark_session_dirty()
        return removed_count

    def replace_construction_entity(
        self,
        construction_id: str,
        entity: ConstructionEntity,
        *,
        select: bool = True,
        mark_dirty: bool = True,
    ) -> bool:
        for index, current in enumerate(self.construction_entities):
            if current.id != construction_id:
                continue
            replacement = replace(
                entity,
                id=current.id,
                revision=current.revision + 1,
            )
            self.construction_entities[index] = replacement
            if select:
                self.select_construction(current.id)
            if current.definition != replacement.definition:
                self.mark_construction_geometry_changed()
            else:
                self.mark_construction_metadata_changed()
            if mark_dirty:
                self.mark_session_dirty()
            return True
        return False

    def remove_construction_entity(
        self,
        construction_id: str,
        *,
        mark_dirty: bool = True,
    ) -> bool:
        return bool(
            self.remove_construction_entities(
                (construction_id,),
                mark_dirty=mark_dirty,
            )
        )

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
            self.selected_construction_id = None

    def select_overlay_annotation(self, overlay_id: str | None) -> None:
        self.selected_overlay_id = overlay_id
        if overlay_id is not None:
            self.view_state.selected_measurement_id = None
            self.selected_construction_id = None

    def select_text_annotation(self, text_id: str | None) -> None:
        annotation = self.get_text_annotation(text_id)
        self.select_overlay_annotation(annotation.id if annotation is not None else None)

    def add_measurement(self, measurement: Measurement) -> None:
        self.insert_measurement_incremental(measurement, mark_dirty=True)
        self.rebuild_group_memberships()

    def insert_measurement_incremental(
        self,
        measurement: Measurement,
        *,
        index: int | None = None,
        select: bool = True,
        mark_dirty: bool = True,
        assign_active_group: bool = True,
    ) -> None:
        if assign_active_group and measurement.fiber_group_id is None:
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
        self.mark_session_dirty()
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
        if measurement.fiber_group_id == group_id:
            return
        measurement.fiber_group_id = group_id
        self.rebuild_group_memberships()
        self.mark_session_dirty()

    def move_uncategorized_measurements_to_group(self, group_id: str) -> int:
        if self.get_group(group_id) is None:
            return 0
        moved_count = 0
        for measurement in self.measurements:
            if measurement.fiber_group_id is None:
                measurement.fiber_group_id = group_id
                moved_count += 1
        if self.active_group_id is None:
            self.active_group_id = group_id
        self.rebuild_group_memberships()
        if moved_count:
            self.mark_session_dirty()
        return moved_count

    def rebuild_group_memberships(self) -> None:
        group_map = {group.id: group for group in self.fiber_groups}
        seen = {group.id: set() for group in self.fiber_groups}
        for group in self.fiber_groups:
            group.measurement_ids = []
        for measurement in self.measurements:
            group = group_map.get(measurement.fiber_group_id or "")
            if group is not None and measurement.id not in seen[group.id]:
                seen[group.id].add(measurement.id)
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
        self.mark_session_dirty()
        return True

    def add_overlay_annotation(self, annotation: OverlayAnnotation) -> None:
        self.overlay_annotations.append(annotation)
        self.select_overlay_annotation(annotation.id)
        self.mark_session_dirty()

    def add_text_annotation(self, annotation: TextAnnotation) -> None:
        self.add_overlay_annotation(annotation.to_overlay())

    def replace_overlay_annotation(self, overlay_id: str, replacement: OverlayAnnotation) -> None:
        for index, annotation in enumerate(self.overlay_annotations):
            if annotation.id == overlay_id:
                if annotation == replacement:
                    return
                self.overlay_annotations[index] = replacement
                self.select_overlay_annotation(replacement.id)
                self.mark_session_dirty()
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
        if self.get_overlay_annotation(overlay_id) is None:
            return
        self.overlay_annotations = [
            annotation for annotation in self.overlay_annotations
            if annotation.id != overlay_id
        ]
        if self.selected_overlay_id == overlay_id:
            self.select_overlay_annotation(None)
        self.mark_session_dirty()

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
        self.mark_session_dirty()
        return True

    def hide_uncategorized_entry(self) -> bool:
        if not self.can_delete_uncategorized_entry():
            return False
        if self.active_group_id is None:
            self.active_group_id = self.sorted_groups()[0].id
        self.mark_session_dirty()
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
        self.mark_session_dirty()
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
        self.mark_session_dirty()
        return True

    @staticmethod
    def _normalized_suppressed_project_group_labels(labels: list[str]) -> list[str]:
        return sorted({token for token in (normalize_group_label(item) for item in labels) if token})

    def measurement_values(self) -> list[float]:
        return [
            measurement.diameter_unit
            for measurement in self.length_measurements()
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
        # Recalculation is a derived-value operation.  The command that changes
        # calibration owns both dirty domains; load-time recalculation must not
        # manufacture a dirty sidecar state.
        self.refresh_dirty_flags()

    def session_snapshot(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "fiber_groups": [group.to_dict() for group in self.sorted_groups()],
            "measurements": [measurement.to_dict() for measurement in self.measurements],
            "overlay_annotations": [annotation.to_dict() for annotation in self.overlay_annotations],
            "scale_overlay_anchor": self.scale_overlay_anchor.to_dict() if self.scale_overlay_anchor else None,
            "suppressed_project_group_labels": list(self.suppressed_project_group_labels),
        }
        if self.construction_entities:
            payload["construction_entities"] = [
                entity.to_dict() for entity in self.construction_entities
            ]
        if self.selected_construction_id is not None:
            payload["selected_construction_id"] = self.selected_construction_id
        return payload

    def calibration_snapshot(self) -> dict[str, Any]:
        calibration_line = self.metadata.get("calibration_line")
        return {
            "calibration": self.calibration.to_dict() if self.calibration else None,
            "calibration_line": calibration_line,
        }

    def snapshot_state(self) -> dict[str, Any]:
        return {
            # Runtime-only compatibility metadata for legacy History.push().
            # Project/sidecar serializers never call snapshot_state().
            "_runtime_state_stamp": {
                "session_state_id": self._current_state_stamp.session_state_id,
                "calibration_state_id": self._current_state_stamp.calibration_state_id,
            },
            "calibration": self.calibration.to_dict() if self.calibration else None,
            "fiber_groups": [group.to_dict() for group in self.sorted_groups()],
            "measurements": [measurement.to_dict() for measurement in self.measurements],
            "overlay_annotations": [annotation.to_dict() for annotation in self.overlay_annotations],
            "metadata": dict(self.metadata),
            "active_group_id": self.active_group_id,
            "selected_measurement_id": self.view_state.selected_measurement_id,
            "selected_overlay_id": self.selected_overlay_id,
            "construction_entities": [
                entity.to_dict() for entity in self.construction_entities
            ],
            "selected_construction_id": self.selected_construction_id,
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
        construction_payload = snapshot.get("construction_entities", [])
        self.construction_entities = [
            ConstructionEntity.from_dict(item)
            for item in construction_payload
            if isinstance(item, dict)
        ]
        self.selected_construction_id = snapshot.get("selected_construction_id")
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
        if self.get_construction_entity(self.selected_construction_id) is None:
            self.selected_construction_id = None
        if self.selected_construction_id is not None:
            self.view_state.selected_measurement_id = None
            self.selected_overlay_id = None
        self.mark_measurement_geometry_changed()
        self.mark_construction_geometry_changed()
        self.refresh_dirty_flags()

    def mark_session_saved(self) -> None:
        saved = self._saved_state_stamp or self._current_state_stamp
        self._saved_state_stamp = DocumentStateStamp(
            session_state_id=self._current_state_stamp.session_state_id,
            calibration_state_id=saved.calibration_state_id,
        )
        self.refresh_dirty_flags()

    def mark_calibration_saved(self) -> None:
        saved = self._saved_state_stamp or self._current_state_stamp
        self._saved_state_stamp = DocumentStateStamp(
            session_state_id=saved.session_state_id,
            calibration_state_id=self._current_state_stamp.calibration_state_id,
        )
        self._saved_calibration_signature = self.calibration_signature()
        self.refresh_dirty_flags()

    def calibration_signature(self) -> tuple[object, ...]:
        calibration = self.calibration
        if calibration is None:
            calibration_token: tuple[object, ...] = (None,)
        else:
            calibration_token = (
                calibration.mode,
                float(calibration.pixels_per_unit),
                calibration.unit,
                calibration.source_label,
            )
        line_payload = self.metadata.get("calibration_line")
        line_token: tuple[object, ...] | None = None
        if isinstance(line_payload, Line):
            line_token = (
                float(line_payload.start.x),
                float(line_payload.start.y),
                float(line_payload.end.x),
                float(line_payload.end.y),
            )
        elif isinstance(line_payload, dict):
            try:
                line = Line.from_dict(line_payload)
            except (KeyError, TypeError, ValueError):
                line_token = (repr(line_payload),)
            else:
                line_token = (
                    float(line.start.x),
                    float(line.start.y),
                    float(line.end.x),
                    float(line.end.y),
                )
        return calibration_token + (line_token,)

    def ensure_external_calibration_change_is_dirty(self) -> None:
        """Bridge direct legacy assignments at the sidecar persistence edge."""

        if (
            not self.dirty_flags.calibration_dirty
            and self._saved_calibration_signature is not None
            and self.calibration_signature() != self._saved_calibration_signature
        ):
            self.mark_calibration_dirty()

    def mark_session_dirty(self) -> None:
        self.advance_state({DirtyDomain.SESSION})

    def mark_calibration_dirty(self) -> None:
        # A calibration changes the physical values persisted in the project,
        # in addition to the independently persisted sidecar calibration.
        self.advance_state({DirtyDomain.SESSION, DirtyDomain.CALIBRATION})

    @property
    def state_stamp(self) -> DocumentStateStamp:
        return self._current_state_stamp

    @property
    def saved_state_stamp(self) -> DocumentStateStamp:
        return self._saved_state_stamp or self._current_state_stamp

    def advance_state(self, domains: set[DirtyDomain] | frozenset[DirtyDomain]) -> DocumentStateStamp:
        normalized = frozenset(domains)
        if not normalized:
            return self._current_state_stamp
        state_id = self._next_state_id
        self._next_state_id += 1
        current = self._current_state_stamp
        self._current_state_stamp = DocumentStateStamp(
            session_state_id=(
                state_id
                if DirtyDomain.SESSION in normalized
                else current.session_state_id
            ),
            calibration_state_id=(
                state_id
                if DirtyDomain.CALIBRATION in normalized
                else current.calibration_state_id
            ),
        )
        self.refresh_dirty_flags()
        return self._current_state_stamp

    def restore_state_stamp(self, stamp: DocumentStateStamp) -> None:
        self._current_state_stamp = stamp
        self._next_state_id = max(
            self._next_state_id,
            stamp.session_state_id + 1,
            stamp.calibration_state_id + 1,
        )
        self.refresh_dirty_flags()

    def refresh_dirty_flags(self) -> None:
        saved = self._saved_state_stamp or self._current_state_stamp
        self.dirty_flags = DirtyFlags(
            session_dirty=self._current_state_stamp.session_state_id != saved.session_state_id,
            calibration_dirty=self._current_state_stamp.calibration_state_id != saved.calibration_state_id,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "id": self.id,
            "path": self.path,
            "source_type": self.source_type,
            "document_kind": self.document_kind,
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
        if self.source_type == "filesystem" and self.absolute_path:
            payload["absolute_path"] = self.absolute_path
        if self.raster_pixel_type is not None:
            payload["raster_pixel_type"] = self.raster_pixel_type.value
        if self.raster_semantic is not None:
            payload["raster_semantic"] = self.raster_semantic.value
        if self.display_transform is not None:
            payload["display_transform"] = self.display_transform.to_dict()
        if self.derivation is not None:
            payload["derivation"] = self.derivation.to_dict()
        if self.construction_entities:
            payload["construction_entities"] = [
                entity.to_dict() for entity in self.construction_entities
            ]
        if self.selected_construction_id is not None:
            payload["selected_construction_id"] = self.selected_construction_id
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ImageDocument":
        overlay_payload = payload.get("overlay_annotations")
        construction_payload = payload.get("construction_entities", [])
        if not isinstance(construction_payload, list) or any(
            not isinstance(item, dict) for item in construction_payload
        ):
            raise TypeError("construction_entities 必须是对象列表")
        construction_entities = [
            ConstructionEntity.from_dict(item)
            for item in construction_payload
        ]
        construction_ids = [entity.id for entity in construction_entities]
        if len(construction_ids) != len(set(construction_ids)):
            raise ValueError("construction_entities 包含重复 ID")
        display_transform_payload = payload.get("display_transform")
        if (
            display_transform_payload is not None
            and not isinstance(display_transform_payload, dict)
        ):
            raise TypeError("display_transform 必须是对象")
        derivation_payload = payload.get("derivation")
        if (
            derivation_payload is not None
            and not isinstance(derivation_payload, dict)
        ):
            raise TypeError("derivation 必须是对象")
        overlay_annotations = [
            OverlayAnnotation.from_dict(item)
            for item in overlay_payload
            if isinstance(item, dict)
        ] if isinstance(overlay_payload, list) else [
            TextAnnotation.from_dict(item).to_overlay()
            for item in payload.get("text_annotations", [])
            if isinstance(item, dict)
        ]
        parsed_pixel_type = (
            RasterPixelType.parse(payload["raster_pixel_type"])
            if payload.get("raster_pixel_type") is not None
            else None
        )
        parsed_derivation = (
            ImageDerivation.from_dict(derivation_payload)
            if isinstance(derivation_payload, dict)
            else None
        )
        semantic_payload = payload.get("raster_semantic")
        derivation_semantic = (
            parsed_derivation.result_semantic
            if parsed_derivation is not None
            else None
        )
        if semantic_payload is not None and derivation_semantic is not None:
            root_semantic = (
                semantic_payload
                if isinstance(semantic_payload, RasterSemantic)
                else RasterSemantic(str(semantic_payload))
            )
            if root_semantic is not derivation_semantic:
                raise ValueError(
                    "文档 raster_semantic 与 derivation.result.semantic "
                    "不一致，已拒绝按不确定语义加载"
                )
        if semantic_payload is None:
            semantic_payload = derivation_semantic
        parsed_semantic = (
            (
                semantic_payload
                if isinstance(semantic_payload, RasterSemantic)
                else RasterSemantic(str(semantic_payload))
            )
            if semantic_payload is not None
            else None
        )
        if parsed_semantic is not None:
            if parsed_pixel_type is None:
                raise ValueError(
                    "raster_semantic 需要同时记录 raster_pixel_type"
                )
            RasterTypeState(
                pixel_type=parsed_pixel_type,
                semantic=parsed_semantic,
            )
        image_document = cls(
            id=str(payload["id"]),
            path=str(payload["path"]),
            source_type=str(payload.get("source_type", "filesystem")),
            document_kind=str(payload.get("document_kind", "image")),
            absolute_path=str(payload["absolute_path"]) if payload.get("absolute_path") else None,
            raster_pixel_type=parsed_pixel_type,
            raster_semantic=parsed_semantic,
            display_transform=(
                DisplayTransform.from_dict(display_transform_payload)
                if isinstance(display_transform_payload, dict)
                else None
            ),
            derivation=parsed_derivation,
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
            construction_entities=construction_entities,
            selected_construction_id=(
                str(payload["selected_construction_id"])
                if payload.get("selected_construction_id")
                else None
            ),
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
    project_rois: list[ProjectRoi] = field(default_factory=list)
    analysis_artifacts: list[AnalysisArtifact] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    load_issues: list[dict[str, Any]] = field(default_factory=list, repr=False, compare=False)
    project_schema_version: int = PROJECT_SCHEMA_VERSION
    min_reader_version: int = PROJECT_MIN_READER_VERSION
    required_features: tuple[str, ...] = ()
    compatibility: ProjectCompatibilityState = field(
        default_factory=ProjectCompatibilityState,
        repr=False,
        compare=False,
    )
    _extension_state_id: int = field(
        default=0,
        init=False,
        repr=False,
        compare=False,
    )
    _next_extension_state_id: int = field(
        default=1,
        init=False,
        repr=False,
        compare=False,
    )

    def get_document(self, document_id: str) -> ImageDocument | None:
        for document in self.documents:
            if document.id == document_id:
                return document
        return None

    def get_project_roi(self, roi_id: str) -> ProjectRoi | None:
        for roi in self.project_rois:
            if roi.id == roi_id:
                return roi
        return None

    def get_analysis_artifact(self, artifact_id: str) -> AnalysisArtifact | None:
        for artifact in self.analysis_artifacts:
            if artifact.id == artifact_id:
                return artifact
        return None

    @property
    def is_read_only_compatible(self) -> bool:
        return self.compatibility.read_only

    def effective_required_features(self) -> tuple[str, ...]:
        """Return declared features plus those required by persisted content."""

        features = list(self.required_features)
        if self.project_rois:
            features.append("project-rois/v1")
        if self.analysis_artifacts:
            features.append("analysis-artifacts/v2")
        if any(document.construction_entities for document in self.documents):
            features.append("construction-geometry/v1")
        return tuple(dict.fromkeys(features))

    def remove_project_rois(
        self,
        roi_ids: Iterable[str],
    ) -> ProjectRoiDeletionResult:
        """Atomically remove ROIs and every now-dangling composite."""

        result = remove_rois_with_dependents(self.project_rois, roi_ids)
        if result.removed_ids:
            self.project_rois = list(result.remaining_rois)
            self.mark_extension_changed()
        return result

    @property
    def extension_state_id(self) -> int:
        """O(1) dirty identity for ROI and independent analysis branches."""

        return self._extension_state_id

    def mark_extension_changed(self) -> int:
        state_id = self._next_extension_state_id
        self._next_extension_state_id += 1
        self._extension_state_id = state_id
        return state_id

    def refresh_analysis_validity(
        self,
        document_id: str,
        *,
        source_document_exists: bool | None = None,
        current_pixel_revision: int | None = None,
        current_source_descriptor: AnalysisSourceDescriptor | None | object = ...,
        current_source_descriptors: (
            Mapping[str, AnalysisSourceDescriptor | None] | None
        ) = None,
        current_calibration_signature: str | None | object = ...,
        roi_revisions: Mapping[str, int] | None = None,
        measurement_revisions: Mapping[str, int] | None = None,
    ) -> int:
        """Mark outdated analysis results stale without deleting audit data.

        The caller should pass ``current_pixel_revision`` whenever authoritative
        pixels may have changed.  ROI and measurement revision registries
        default to the currently mounted project objects.  A previously stale
        artifact is never revived; recalculation must append a new artifact.
        """

        normalized_document_id = str(document_id or "").strip()
        if not normalized_document_id:
            raise ValueError("document_id 不能为空")
        document = self.get_document(normalized_document_id)
        document_exists = (
            document is not None
            if source_document_exists is None
            else source_document_exists
        )
        if not isinstance(document_exists, bool):
            raise TypeError("source_document_exists 必须是布尔值")
        if roi_revisions is None:
            roi_revisions = {
                roi.id: roi.revision
                for roi in self.project_rois
                if roi.document_id == normalized_document_id
            }
        if measurement_revisions is None:
            measurement_revisions = (
                {}
                if document is None
                else {
                    measurement.id: measurement.geometry_revision
                    for measurement in document.measurements
                }
            )
        if current_calibration_signature is ...:
            calibration = None if document is None else document.calibration
            calibration_signature = (
                None
                if calibration is None
                else calibration_signature_from_values(
                    pixels_per_unit=calibration.pixels_per_unit,
                    unit=calibration.unit,
                )
            )
        else:
            calibration_signature = current_calibration_signature
        before = tuple(self.analysis_artifacts)
        current_dependency_signatures = {
            artifact.id: (
                None
                if rebuilt is None
                else rebuilt.sha256
            )
            for artifact in before
            if artifact.source_document_id == normalized_document_id
            and artifact.dependency_signature is not None
            for rebuilt in (
                _rebuild_analysis_dependency_signature(
                    self,
                    document,
                    artifact.dependency_signature,
                ),
            )
        }
        refresh_kwargs: dict[str, object] = {
            "document_id": normalized_document_id,
            "source_document_exists": document_exists,
            "current_pixel_revision": current_pixel_revision,
            "current_calibration_signature": calibration_signature,
            "roi_revisions": roi_revisions,
            "measurement_revisions": measurement_revisions,
            "current_dependency_signatures": current_dependency_signatures,
        }
        if current_source_descriptor is not ...:
            refresh_kwargs["current_source_descriptor"] = (
                current_source_descriptor
            )
        if current_source_descriptors is not None:
            refresh_kwargs["current_source_descriptors"] = (
                current_source_descriptors
            )
        refreshed = refresh_artifacts_validity(
            before,
            **refresh_kwargs,  # type: ignore[arg-type]
        )
        self.analysis_artifacts = list(refreshed)
        return sum(
            previous is not current
            for previous, current in zip(before, refreshed, strict=True)
        )

    def analysis_dependency_is_current(
        self,
        document_id: str,
        dependency_signature: AnalysisDependencySignature | None,
    ) -> bool:
        """Verify a frozen analysis dependency against current project state.

        This is intentionally side-effect free so result-commit paths can call
        it before and after writing assets.  ``None`` represents an old v1
        analysis request that did not declare dependency provenance.
        """

        if dependency_signature is None:
            return True
        if not isinstance(dependency_signature, AnalysisDependencySignature):
            raise TypeError(
                "dependency_signature 必须是 AnalysisDependencySignature"
            )
        normalized_document_id = str(document_id or "").strip()
        if not normalized_document_id:
            raise ValueError("document_id 不能为空")
        document = self.get_document(normalized_document_id)
        rebuilt = _rebuild_analysis_dependency_signature(
            self,
            document,
            dependency_signature,
        )
        return (
            rebuilt is not None
            and rebuilt.sha256 == dependency_signature.sha256
        )

    def to_dict(self) -> dict[str, Any]:
        seen_template_labels: set[str] = set()
        serialized_templates: list[dict[str, Any]] = []
        for template in self.project_group_templates:
            token = template.normalized_label()
            if not token or token in seen_template_labels:
                continue
            seen_template_labels.add(token)
            serialized_templates.append(template.to_dict())
        payload = {
            "project_schema_version": PROJECT_SCHEMA_VERSION,
            "min_reader_version": PROJECT_MIN_READER_VERSION,
            "required_features": list(self.effective_required_features()),
            "version": self.version,
            "documents": [document.to_dict() for document in self.documents],
            "project_default_calibration": self.project_default_calibration.to_dict() if self.project_default_calibration else None,
            "project_group_templates": serialized_templates,
            "metadata": self.metadata,
        }
        serialized_rois = _serialize_project_rois(self.project_rois)
        if serialized_rois:
            payload["project_rois"] = serialized_rois
        serialized_artifacts = _serialize_analysis_artifacts(
            self.analysis_artifacts
        )
        if serialized_artifacts:
            payload["analysis_artifacts"] = serialized_artifacts
        return payload

    @classmethod
    def empty(cls) -> "ProjectState":
        return cls(version=__version__, documents=[])

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ProjectState":
        if not isinstance(payload, dict):
            raise TypeError("ProjectState 必须是对象")
        compatibility = _project_compatibility_from_payload(payload)
        seen_template_labels: set[str] = set()
        project_group_templates: list[ProjectGroupTemplate] = []
        for item in payload.get("project_group_templates", []):
            template = ProjectGroupTemplate.from_dict(item)
            token = template.normalized_label()
            if not token or token in seen_template_labels:
                continue
            seen_template_labels.add(token)
            project_group_templates.append(template)
        project_rois_payload = payload.get("project_rois", [])
        if not isinstance(project_rois_payload, list) or any(
            not isinstance(item, dict) for item in project_rois_payload
        ):
            raise TypeError("project_rois 必须是对象列表")
        analysis_artifacts_payload = payload.get("analysis_artifacts", [])
        if not isinstance(analysis_artifacts_payload, list) or any(
            not isinstance(item, dict) for item in analysis_artifacts_payload
        ):
            raise TypeError("analysis_artifacts 必须是对象列表")
        project_rois = [
            ProjectRoi.from_dict(item)
            for item in project_rois_payload
        ]
        analysis_artifacts = [
            AnalysisArtifact.from_dict(item)
            for item in analysis_artifacts_payload
        ]
        _ensure_unique_ids(project_rois, field_name="project_rois")
        _ensure_unique_ids(
            analysis_artifacts,
            field_name="analysis_artifacts",
        )
        return cls(
            version=str(payload.get("version", __version__)),
            documents=[ImageDocument.from_dict(item) for item in payload.get("documents", [])],
            calibration_presets=[
                CalibrationPreset.from_dict(item)
                for item in payload.get("calibration_presets", [])
            ],
            project_default_calibration=Calibration.from_dict(payload["project_default_calibration"]) if payload.get("project_default_calibration") else None,
            project_group_templates=project_group_templates,
            project_rois=project_rois,
            analysis_artifacts=analysis_artifacts,
            metadata=dict(payload.get("metadata", {})),
            project_schema_version=compatibility.source_schema_version,
            min_reader_version=compatibility.min_reader_version,
            required_features=compatibility.required_features,
            compatibility=compatibility,
        )


def _analysis_dependency_json_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _rebuild_analysis_dependency_signature(
    project: ProjectState,
    document: ImageDocument | None,
    dependency_signature: AnalysisDependencySignature | None,
) -> AnalysisDependencySignature | None:
    """Rebuild a v2 dependency signature from current authoritative objects.

    The stored dependency descriptor defines the original analysis scope. Current
    revisions, point coordinates, category membership and calibration values
    are substituted into that descriptor. Missing dependencies deliberately
    return ``None`` so validity refresh marks the artifact stale.
    """

    original = dependency_signature
    if original is None or document is None:
        return None
    dependencies = original.dependencies

    calibration = document.calibration
    current_calibration = (
        {
            "signature": None,
            "pixel_size_x": 1.0,
            "pixel_size_y": 1.0,
            "unit": "px",
        }
        if calibration is None
        else {
            "signature": calibration_signature_from_values(
                pixels_per_unit=calibration.pixels_per_unit,
                unit=calibration.unit,
            ),
            "pixel_size_x": 1.0 / calibration.pixels_per_unit,
            "pixel_size_y": 1.0 / calibration.pixels_per_unit,
            "unit": calibration.unit,
        }
    )

    stored_roi_refs = dependencies.get("roi_transitive_refs")
    if not isinstance(stored_roi_refs, Mapping):
        return None
    roi_lookup = {
        roi.id: roi
        for roi in project.project_rois
        if roi.document_id == document.id
    }
    current_roi_refs: dict[str, object] = {}
    for roi_id, stored_entry in stored_roi_refs.items():
        if not isinstance(roi_id, str) or not isinstance(stored_entry, Mapping):
            return None
        roi = roi_lookup.get(roi_id)
        if roi is None:
            return None
        expected_revision = stored_entry.get("revision")
        if (
            isinstance(expected_revision, bool)
            or not isinstance(expected_revision, int)
            or roi.revision != expected_revision
            or stored_entry.get("kind") != roi.kind.value
        ):
            return None
        current_roi_refs[roi_id] = dict(stored_entry)

    stored_measurements = dependencies.get("measurement_revisions")
    if not isinstance(stored_measurements, Mapping):
        return None
    measurement_lookup = {
        measurement.id: measurement
        for measurement in document.measurements
    }
    current_measurements: dict[str, int] = {}
    for measurement_id, expected_revision in stored_measurements.items():
        if (
            not isinstance(measurement_id, str)
            or isinstance(expected_revision, bool)
            or not isinstance(expected_revision, int)
        ):
            return None
        measurement = measurement_lookup.get(measurement_id)
        if (
            measurement is None
            or measurement.geometry_revision != expected_revision
        ):
            return None
        current_measurements[measurement_id] = measurement.geometry_revision

    stored_study_region = dependencies.get("study_region")
    current_point_set = dependencies.get("point_set")
    current_group = dependencies.get("group")
    if current_point_set is not None:
        if not isinstance(current_point_set, Mapping):
            return None
        measurement_ids = current_point_set.get("measurement_ids")
        if not isinstance(measurement_ids, list) or any(
            not isinstance(item, str) for item in measurement_ids
        ):
            return None
        origin_x = 0.0
        origin_y = 0.0
        if isinstance(stored_study_region, Mapping):
            raw_origin = stored_study_region.get("viewport_origin")
            if (
                isinstance(raw_origin, list)
                and len(raw_origin) == 2
                and all(
                    isinstance(item, (int, float)) and not isinstance(item, bool)
                    for item in raw_origin
                )
            ):
                origin_x = float(raw_origin[0])
                origin_y = float(raw_origin[1])
        point_records: list[dict[str, object]] = []
        for measurement_id in measurement_ids:
            measurement = measurement_lookup.get(measurement_id)
            if (
                measurement is None
                or measurement.measurement_kind != "count"
                or measurement.point_px is None
            ):
                return None
            point_records.append(
                {
                    "measurement_id": measurement.id,
                    "revision": measurement.geometry_revision,
                    "point": [
                        float(measurement.point_px.x) - origin_x,
                        float(measurement.point_px.y) - origin_y,
                    ],
                    "group_id": measurement.fiber_group_id,
                }
            )
        rebuilt_point_set = dict(current_point_set)
        rebuilt_point_set.update(
            {
                "sha256": _analysis_dependency_json_sha256(point_records),
                "count": len(point_records),
                "measurement_ids": list(measurement_ids),
            }
        )
        current_point_set = rebuilt_point_set

        if not isinstance(current_group, Mapping):
            return None
        scope = str(current_group.get("scope", "all"))
        rebuilt_group: dict[str, object]
        if scope == "active_group":
            group_id = str(current_group.get("id") or "")
            group = document.get_group(group_id)
            if group is None:
                return None
            membership = sorted(
                measurement.id
                for measurement in document.count_measurements()
                if measurement.fiber_group_id == group.id
            )
            rebuilt_group = {
                "scope": "active_group",
                "id": group.id,
                "number": group.number,
                "label": group.label,
                "color": group.color,
                "membership_sha256": _analysis_dependency_json_sha256(
                    membership
                ),
            }
        elif scope == "all":
            membership = sorted(
                measurement.id
                for measurement in document.count_measurements()
            )
            rebuilt_group = {
                "scope": "all",
                "membership_sha256": _analysis_dependency_json_sha256(
                    membership
                ),
            }
            # Early v2 artifacts did not yet persist all-scope membership. Keep
            # their original canonical shape for read compatibility.
            if "membership_sha256" not in current_group:
                rebuilt_group.pop("membership_sha256")
        else:
            return None
        current_group = rebuilt_group

    return AnalysisDependencySignature(
        calibration=current_calibration,
        roi_transitive_refs=current_roi_refs,
        measurement_revisions=current_measurements,
        point_set=current_point_set,
        group=current_group,
        study_region=stored_study_region,
    )


def _project_compatibility_from_payload(
    payload: Mapping[str, Any],
    *,
    source_path: str | Path | None = None,
) -> ProjectCompatibilityState:
    raw_schema_version = payload.get("project_schema_version", 1)
    if isinstance(raw_schema_version, bool) or not isinstance(raw_schema_version, int):
        raise TypeError("project_schema_version 必须是整数")
    if raw_schema_version < 1:
        raise ValueError("project_schema_version 不能小于 1")
    if raw_schema_version > PROJECT_SCHEMA_VERSION:
        raise ValueError(
            "项目格式版本过新，当前程序不支持: "
            f"{raw_schema_version} > {PROJECT_SCHEMA_VERSION}"
        )

    raw_min_reader = payload.get("min_reader_version", 1)
    if isinstance(raw_min_reader, bool) or not isinstance(raw_min_reader, int):
        raise TypeError("min_reader_version 必须是整数")
    if raw_min_reader < 1:
        raise ValueError("min_reader_version 不能小于 1")
    if raw_min_reader > PROJECT_SCHEMA_VERSION:
        raise ValueError(
            "项目要求更高版本的读取器: "
            f"{raw_min_reader} > {PROJECT_SCHEMA_VERSION}"
        )

    raw_features = payload.get("required_features", [])
    if not isinstance(raw_features, list) or any(
        not isinstance(feature, str) for feature in raw_features
    ):
        raise TypeError("required_features 必须是字符串列表")
    normalized_features = tuple(
        dict.fromkeys(
            feature.strip()
            for feature in raw_features
            if feature.strip()
        )
    )
    unknown_features = tuple(
        feature
        for feature in normalized_features
        if feature not in SUPPORTED_PROJECT_REQUIRED_FEATURES
    )
    return ProjectCompatibilityState(
        source_schema_version=raw_schema_version,
        min_reader_version=raw_min_reader,
        required_features=normalized_features,
        unknown_required_features=unknown_features,
        source_path=None if source_path is None else str(Path(source_path)),
    )


def _ensure_unique_ids(
    items: list[ProjectRoi] | list[AnalysisArtifact],
    *,
    field_name: str,
) -> None:
    seen: set[str] = set()
    for item in items:
        if item.id in seen:
            raise ValueError(f"{field_name} 包含重复 ID: {item.id}")
        seen.add(item.id)


def _serialize_project_rois(
    rois: list[ProjectRoi],
) -> list[dict[str, object]]:
    if any(not isinstance(roi, ProjectRoi) for roi in rois):
        raise TypeError("project_rois 必须全部是 ProjectRoi")
    _ensure_unique_ids(rois, field_name="project_rois")
    return [roi.to_dict() for roi in rois]


def _serialize_analysis_artifacts(
    artifacts: list[AnalysisArtifact],
) -> list[dict[str, object]]:
    if any(not isinstance(artifact, AnalysisArtifact) for artifact in artifacts):
        raise TypeError("analysis_artifacts 必须全部是 AnalysisArtifact")
    _ensure_unique_ids(artifacts, field_name="analysis_artifacts")
    return [artifact.to_dict() for artifact in artifacts]
