"""Persistent presentation state shared by legacy and professional measurements.

Offsets are image coordinates relative to a measurement's semantic anchor.  They
never contain a copied measurement value, screen coordinates or fitted geometry.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any

from fdm.geometry import Point



ANNOTATION_STYLE_FIELDS = (
    "label_position", "show_leader", "show_identifier", "leader_style", "show_metric_name", "show_symbol", "metric_key",
)


@dataclass(frozen=True, slots=True)
class MeasurementAnnotationLayout:
    placement_mode: str = "automatic"
    offset_px: Point | None = None
    show_leader: bool = True
    show_identifier: bool = False
    revision: int = 1
    metric_key: str | None = None
    leader_style: str = "single"
    show_metric_name: bool = False
    # None preserves the v1 convention: professional dimensions used symbols, ordinary lines did not.
    show_symbol: bool | None = False
    # Placement policy is a style. A particular drag offset belongs to one object.
    label_position: str = "external"

    def __post_init__(self) -> None:
        if self.label_position not in {"near", "external"}:
            raise ValueError("Unknown measurement label position")
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
        flags = (self.show_leader, self.show_identifier, self.show_metric_name)
        if self.show_symbol is not None:
            flags += (self.show_symbol,)
        if not all(type(flag) is bool for flag in flags):
            raise ValueError("Annotation display flags must be boolean")
        if type(self.revision) is not int or self.revision < 1:
            raise ValueError("Annotation revision must be a positive integer")

    def with_manual_offset(self, offset: Point) -> MeasurementAnnotationLayout:
        return replace(self, placement_mode="manual", offset_px=offset, revision=self.revision + 1)

    def with_automatic_position(self) -> MeasurementAnnotationLayout:
        return replace(self, placement_mode="automatic", offset_px=None, revision=self.revision + 1)

    def with_near_position(self) -> MeasurementAnnotationLayout:
        return replace(self.with_automatic_position(), label_position="near", show_leader=False)

    @classmethod
    def ordinary_near(cls) -> MeasurementAnnotationLayout:
        return cls(label_position="near", show_leader=False)

    @classmethod
    def legacy_professional(cls) -> MeasurementAnnotationLayout:
        """v1 professional definitions with no layout still displayed verbose text."""
        return cls(show_identifier=True, show_metric_name=True, show_symbol=None)

    def style_fields(self) -> dict[str, Any]:
        """Only these choices may be shared between new objects."""
        return {key: getattr(self, key) for key in ANNOTATION_STYLE_FIELDS}

    def to_dict(self) -> dict[str, Any]:
        return {
            "label_position": self.label_position,
            "placement_mode": self.placement_mode,
            "offset_px": None if self.offset_px is None else self.offset_px.to_dict(),
            "show_leader": self.show_leader,
            "show_identifier": self.show_identifier,
            "revision": self.revision,
            # A v1 layout omitted both flags and always printed metric names
            # and supported symbols. Preserve its serialized appearance.
            **({"show_metric_name": False} if not self.show_metric_name else {}),
            **({"show_symbol": self.show_symbol} if self.show_symbol is not None else {}),
            "metric_key": self.metric_key,
            "leader_style": self.leader_style,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any], *, ordinary_line: bool = False) -> MeasurementAnnotationLayout:
        if not isinstance(payload, dict):
            raise ValueError("Annotation layout must be an object")
        if ordinary_line and "label_position" not in payload:
            # Pre-release line placements were calculated with the external
            # dimension solver. Keep their captions and discard stale positions.
            payload = {**payload, "label_position": "near", "placement_mode": "automatic",
                       "offset_px": None, "show_leader": False, "leader_style": "single"}
        raw = payload.get("offset_px")
        if raw is not None and not isinstance(raw, dict):
            raise ValueError("Annotation offset must be an object or null")
        return cls(
            label_position=payload.get("label_position", "external"),
            placement_mode=payload.get("placement_mode", "automatic"),
            offset_px=Point.from_dict(raw) if isinstance(raw, dict) else None,
            show_leader=payload.get("show_leader", True),
            show_identifier=payload.get("show_identifier", True),
            show_metric_name=payload.get("show_metric_name", True),
            show_symbol=payload.get("show_symbol"),
            revision=payload.get("revision", 1),
            metric_key=payload.get("metric_key"),
            leader_style=payload.get("leader_style", "single"),
        )
