from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
import statistics
import uuid

from fdm.geometry import Line, Point, line_length


def utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


@dataclass(slots=True)
class Calibration:
    mode: str
    pixels_per_unit: float
    unit: str
    source_label: str

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


@dataclass(slots=True)
class CalibrationPreset:
    name: str
    pixels_per_unit: float
    unit: str

    def to_calibration(self) -> Calibration:
        return Calibration(
            mode="preset",
            pixels_per_unit=self.pixels_per_unit,
            unit=self.unit,
            source_label=self.name,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "pixels_per_unit": self.pixels_per_unit,
            "unit": self.unit,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CalibrationPreset":
        return cls(
            name=str(payload["name"]),
            pixels_per_unit=float(payload["pixels_per_unit"]),
            unit=str(payload["unit"]),
        )


@dataclass(slots=True)
class FiberGroup:
    id: str
    image_id: str
    name: str
    color: str
    measurement_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "image_id": self.image_id,
            "name": self.name,
            "color": self.color,
            "measurement_ids": list(self.measurement_ids),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FiberGroup":
        return cls(
            id=str(payload["id"]),
            image_id=str(payload["image_id"]),
            name=str(payload["name"]),
            color=str(payload["color"]),
            measurement_ids=list(payload.get("measurement_ids", [])),
        )


@dataclass(slots=True)
class Measurement:
    id: str
    image_id: str
    fiber_group_id: str | None
    mode: str
    line_px: Line
    snapped_line_px: Line | None = None
    diameter_px: float | None = None
    diameter_unit: float | None = None
    confidence: float = 0.0
    status: str = "ready"
    created_at: str = field(default_factory=utc_now_iso)
    debug_payload: dict[str, Any] = field(default_factory=dict)

    def effective_line(self) -> Line:
        return self.snapped_line_px or self.line_px

    def recalculate(self, calibration: Calibration | None) -> None:
        self.diameter_px = line_length(self.effective_line())
        if calibration is None:
            self.diameter_unit = self.diameter_px
        else:
            self.diameter_unit = calibration.px_to_unit(self.diameter_px)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "image_id": self.image_id,
            "fiber_group_id": self.fiber_group_id,
            "mode": self.mode,
            "line_px": self.line_px.to_dict(),
            "snapped_line_px": self.snapped_line_px.to_dict() if self.snapped_line_px else None,
            "diameter_px": self.diameter_px,
            "diameter_unit": self.diameter_unit,
            "confidence": self.confidence,
            "status": self.status,
            "created_at": self.created_at,
            "debug_payload": self.debug_payload,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Measurement":
        snapped_line = payload.get("snapped_line_px")
        return cls(
            id=str(payload["id"]),
            image_id=str(payload["image_id"]),
            fiber_group_id=payload.get("fiber_group_id"),
            mode=str(payload["mode"]),
            line_px=Line.from_dict(payload["line_px"]),
            snapped_line_px=Line.from_dict(snapped_line) if snapped_line else None,
            diameter_px=payload.get("diameter_px"),
            diameter_unit=payload.get("diameter_unit"),
            confidence=float(payload.get("confidence", 0.0)),
            status=str(payload.get("status", "ready")),
            created_at=str(payload.get("created_at", utc_now_iso())),
            debug_payload=dict(payload.get("debug_payload", {})),
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
class ImageDocument:
    id: str
    path: str
    image_size: tuple[int, int]
    calibration: Calibration | None = None
    fiber_groups: list[FiberGroup] = field(default_factory=list)
    measurements: list[Measurement] = field(default_factory=list)
    view_state: ImageViewState = field(default_factory=ImageViewState)
    metadata: dict[str, Any] = field(default_factory=dict)

    def ensure_default_group(self) -> FiberGroup:
        if self.fiber_groups:
            return self.fiber_groups[0]
        group = FiberGroup(
            id=new_id("group"),
            image_id=self.id,
            name="默认分组",
            color="#1F7A8C",
        )
        self.fiber_groups.append(group)
        return group

    def get_group(self, group_id: str | None) -> FiberGroup | None:
        if group_id is None:
            return None
        for group in self.fiber_groups:
            if group.id == group_id:
                return group
        return None

    def get_measurement(self, measurement_id: str | None) -> Measurement | None:
        if measurement_id is None:
            return None
        for measurement in self.measurements:
            if measurement.id == measurement_id:
                return measurement
        return None

    def add_measurement(self, measurement: Measurement) -> None:
        measurement.recalculate(self.calibration)
        self.measurements.append(measurement)
        if measurement.fiber_group_id:
            group = self.get_group(measurement.fiber_group_id)
            if group and measurement.id not in group.measurement_ids:
                group.measurement_ids.append(measurement.id)
        self.view_state.selected_measurement_id = measurement.id

    def measurement_values(self) -> list[float]:
        return [
            measurement.diameter_unit
            for measurement in self.measurements
            if measurement.diameter_unit is not None
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "path": self.path,
            "image_size": list(self.image_size),
            "calibration": self.calibration.to_dict() if self.calibration else None,
            "fiber_groups": [group.to_dict() for group in self.fiber_groups],
            "measurements": [measurement.to_dict() for measurement in self.measurements],
            "view_state": self.view_state.to_dict(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ImageDocument":
        image_document = cls(
            id=str(payload["id"]),
            path=str(payload["path"]),
            image_size=(int(payload["image_size"][0]), int(payload["image_size"][1])),
            calibration=Calibration.from_dict(payload["calibration"]) if payload.get("calibration") else None,
            fiber_groups=[FiberGroup.from_dict(item) for item in payload.get("fiber_groups", [])],
            measurements=[Measurement.from_dict(item) for item in payload.get("measurements", [])],
            view_state=ImageViewState.from_dict(payload.get("view_state", {})),
            metadata=dict(payload.get("metadata", {})),
        )
        if not image_document.fiber_groups:
            image_document.ensure_default_group()
        return image_document


@dataclass(slots=True)
class ProjectState:
    version: str
    documents: list[ImageDocument]
    calibration_presets: list[CalibrationPreset] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_document(self, document_id: str) -> ImageDocument | None:
        for document in self.documents:
            if document.id == document_id:
                return document
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "documents": [document.to_dict() for document in self.documents],
            "calibration_presets": [preset.to_dict() for preset in self.calibration_presets],
            "metadata": self.metadata,
        }

    @classmethod
    def empty(cls) -> "ProjectState":
        return cls(version="0.1.0", documents=[])

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ProjectState":
        return cls(
            version=str(payload.get("version", "0.1.0")),
            documents=[ImageDocument.from_dict(item) for item in payload.get("documents", [])],
            calibration_presets=[
                CalibrationPreset.from_dict(item)
                for item in payload.get("calibration_presets", [])
            ],
            metadata=dict(payload.get("metadata", {})),
        )
