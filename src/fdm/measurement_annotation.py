"""Persistent presentation state shared by legacy and professional measurements.

Offsets are image coordinates relative to a measurement's semantic anchor.  They
never contain a copied measurement value, screen coordinates or fitted geometry.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any

from fdm.geometry import Point


@dataclass(frozen=True, slots=True)
class MeasurementAnnotationLayout:
    placement_mode: str = "automatic"
    offset_px: Point | None = None
    show_leader: bool = True
    show_identifier: bool = True
    revision: int = 1
    metric_key: str | None = None
    leader_style: str = "single"

    def __post_init__(self) -> None:
        if self.placement_mode not in {"automatic", "manual"}:
            raise ValueError("Unknown measurement annotation placement mode")
        if self.leader_style not in {"single", "dual"}:
            raise ValueError("Unknown measurement annotation leader style")
        if self.metric_key is not None and (not isinstance(self.metric_key, str) or not self.metric_key.strip()):
            raise ValueError("Annotation metric key must be a nonempty string")
        if self.offset_px is not None:
            if not math.isfinite(self.offset_px.x) or not math.isfinite(self.offset_px.y):
                raise ValueError("Annotation offset must be finite")
            object.__setattr__(self, "offset_px", Point(float(self.offset_px.x), float(self.offset_px.y)))
        if self.placement_mode == "manual" and self.offset_px is None:
            raise ValueError("Manual annotation requires an image-space offset")
        if type(self.show_leader) is not bool or type(self.show_identifier) is not bool:
            raise ValueError("Annotation display flags must be boolean")
        if type(self.revision) is not int or self.revision < 1:
            raise ValueError("Annotation revision must be a positive integer")

    def with_manual_offset(self, offset: Point) -> MeasurementAnnotationLayout:
        return replace(self, placement_mode="manual", offset_px=offset, revision=self.revision + 1)

    def with_automatic_position(self) -> MeasurementAnnotationLayout:
        return replace(self, placement_mode="automatic", offset_px=None, revision=self.revision + 1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "placement_mode": self.placement_mode,
            "offset_px": None if self.offset_px is None else self.offset_px.to_dict(),
            "show_leader": self.show_leader,
            "show_identifier": self.show_identifier,
            "revision": self.revision,
            "metric_key": self.metric_key,
            "leader_style": self.leader_style,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> MeasurementAnnotationLayout:
        if not isinstance(payload, dict):
            raise ValueError("Annotation layout must be an object")
        raw = payload.get("offset_px")
        if raw is not None and not isinstance(raw, dict):
            raise ValueError("Annotation offset must be an object or null")
        return cls(
            placement_mode=payload.get("placement_mode", "automatic"),
            offset_px=Point.from_dict(raw) if isinstance(raw, dict) else None,
            show_leader=payload.get("show_leader", True),
            show_identifier=payload.get("show_identifier", True),
            revision=payload.get("revision", 1),
            metric_key=payload.get("metric_key"),
            leader_style=payload.get("leader_style", "single"),
        )
