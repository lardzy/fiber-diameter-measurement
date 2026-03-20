from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
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
            label=str(payload.get("label", payload.get("name", ""))),
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
    calibration: Calibration | None = None
    fiber_groups: list[FiberGroup] = field(default_factory=list)
    measurements: list[Measurement] = field(default_factory=list)
    view_state: ImageViewState = field(default_factory=ImageViewState)
    metadata: dict[str, Any] = field(default_factory=dict)
    active_group_id: str | None = None
    sidecar_path: str | None = None
    dirty_flags: DirtyFlags = field(default_factory=DirtyFlags)
    history: Any = field(default=None, repr=False, compare=False)
    _session_clean_snapshot: dict[str, Any] | None = field(default=None, repr=False, compare=False)
    _calibration_clean_snapshot: dict[str, Any] | None = field(default=None, repr=False, compare=False)

    def initialize_runtime_state(self) -> None:
        from fdm.history import DocumentHistory

        if self.history is None:
            self.history = DocumentHistory()
        if self.sidecar_path is None and self.path:
            self.sidecar_path = self.default_sidecar_path()
        if not self.fiber_groups:
            self.ensure_default_group()
        else:
            self.fiber_groups.sort(key=lambda group: group.number)
        self.rebuild_group_memberships()
        if self.active_group_id is None or self.get_group(self.active_group_id) is None:
            self.active_group_id = self.fiber_groups[0].id if self.fiber_groups else None
        if self._session_clean_snapshot is None:
            self.mark_session_saved()
        if self._calibration_clean_snapshot is None:
            self.mark_calibration_saved()
        self.refresh_dirty_flags()

    def default_sidecar_path(self) -> str:
        return f"{self.path}.fdm.json"

    def sorted_groups(self) -> list[FiberGroup]:
        return sorted(self.fiber_groups, key=lambda group: group.number)

    def next_group_number(self) -> int:
        if not self.fiber_groups:
            return 1
        return max(group.number for group in self.fiber_groups) + 1

    def create_group(self, *, color: str, label: str = "") -> FiberGroup:
        group = FiberGroup(
            id=new_id("group"),
            image_id=self.id,
            number=self.next_group_number(),
            label=label,
            color=color,
        )
        self.fiber_groups.append(group)
        self.fiber_groups.sort(key=lambda item: item.number)
        if self.active_group_id is None:
            self.active_group_id = group.id
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
        if group_id is None or self.get_group(group_id) is None:
            return
        self.active_group_id = group_id

    def get_measurement(self, measurement_id: str | None) -> Measurement | None:
        if measurement_id is None:
            return None
        for measurement in self.measurements:
            if measurement.id == measurement_id:
                return measurement
        return None

    def add_measurement(self, measurement: Measurement) -> None:
        if measurement.fiber_group_id is None:
            measurement.fiber_group_id = self.active_group_id or self.ensure_default_group().id
        measurement.recalculate(self.calibration)
        self.measurements.append(measurement)
        self.rebuild_group_memberships()
        self.view_state.selected_measurement_id = measurement.id
        self.refresh_dirty_flags()

    def remove_measurement(self, measurement_id: str) -> None:
        self.measurements = [
            measurement for measurement in self.measurements
            if measurement.id != measurement_id
        ]
        if self.view_state.selected_measurement_id == measurement_id:
            self.view_state.selected_measurement_id = None
        self.rebuild_group_memberships()
        self.refresh_dirty_flags()

    def set_measurement_group(self, measurement_id: str, group_id: str) -> None:
        measurement = self.get_measurement(measurement_id)
        if measurement is None or self.get_group(group_id) is None:
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
        self.refresh_dirty_flags()

    def session_snapshot(self) -> dict[str, Any]:
        return {
            "fiber_groups": [group.to_dict() for group in self.sorted_groups()],
            "measurements": [measurement.to_dict() for measurement in self.measurements],
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
            "metadata": dict(self.metadata),
            "active_group_id": self.active_group_id,
            "selected_measurement_id": self.view_state.selected_measurement_id,
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
        self.metadata = dict(snapshot.get("metadata", {}))
        self.active_group_id = snapshot.get("active_group_id")
        self.view_state.selected_measurement_id = snapshot.get("selected_measurement_id")
        if not self.fiber_groups:
            self.ensure_default_group()
        self.rebuild_group_memberships()
        if self.active_group_id is None or self.get_group(self.active_group_id) is None:
            self.active_group_id = self.fiber_groups[0].id if self.fiber_groups else None
        if self.view_state.selected_measurement_id and self.get_measurement(self.view_state.selected_measurement_id) is None:
            self.view_state.selected_measurement_id = None
        self.refresh_dirty_flags()

    def mark_session_saved(self) -> None:
        self._session_clean_snapshot = self.session_snapshot()
        self.refresh_dirty_flags()

    def mark_calibration_saved(self) -> None:
        self._calibration_clean_snapshot = self.calibration_snapshot()
        self.refresh_dirty_flags()

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
            "image_size": list(self.image_size),
            "calibration": self.calibration.to_dict() if self.calibration else None,
            "fiber_groups": [group.to_dict() for group in self.sorted_groups()],
            "measurements": [measurement.to_dict() for measurement in self.measurements],
            "view_state": self.view_state.to_dict(),
            "metadata": self.metadata,
            "active_group_id": self.active_group_id,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ImageDocument":
        image_document = cls(
            id=str(payload["id"]),
            path=str(payload["path"]),
            image_size=(int(payload["image_size"][0]), int(payload["image_size"][1])),
            calibration=Calibration.from_dict(payload["calibration"]) if payload.get("calibration") else None,
            fiber_groups=[
                FiberGroup.from_dict(item, fallback_number=index + 1)
                for index, item in enumerate(payload.get("fiber_groups", []))
            ],
            measurements=[Measurement.from_dict(item) for item in payload.get("measurements", [])],
            view_state=ImageViewState.from_dict(payload.get("view_state", {})),
            metadata=dict(payload.get("metadata", {})),
            active_group_id=payload.get("active_group_id"),
        )
        image_document.initialize_runtime_state()
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
