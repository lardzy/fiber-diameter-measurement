"""Shared trial UI/runtime preferences, independent of experiment records."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
import math
import re

from fdm.services.optical_config import ALGORITHM_IDS, OpticalAnalysisConfig


def _choice(value, values, default):
    return value if isinstance(value, str) and value in values else default


def _number(value, default, low, high, *, integer=False):
    try:
        result = float(value)
        if not math.isfinite(result) or isinstance(value, bool):
            raise ValueError
    except (TypeError, ValueError, OverflowError):
        result = default
    result = max(low, min(high, result))
    return int(round(result)) if integer else round(result, 1)


@dataclass(slots=True)
class TrialPreferences:
    show_stability_status: bool = False
    content_label: str = "material"
    diameter_label: str = "value"
    label_background: bool = False
    font_family: str = ""
    font_size: int = 12
    font_bold: bool = False
    font_italic: bool = False
    color_mode: str = "follow"
    font_color: str = "#FFFFFF"
    evidence_enabled: bool = False
    evidence_format: str = "crop_jpeg"
    input_mode: str = "mixed"
    gesture: str = "click"
    manual_algorithm: str = "assisted"
    algorithm: str = "traditional"
    automatic: bool = False
    visible: bool = True
    opacity: int = 60
    retention_seconds: float = 1.
    optical_analysis: dict = field(default_factory=dict)

    def normalized_copy(self) -> "TrialPreferences":
        defaults = type(self)()
        values = {item.name: getattr(self, item.name) for item in fields(self)}
        for name in ("show_stability_status", "label_background", "font_bold", "font_italic",
                     "evidence_enabled", "automatic", "visible"):
            values[name] = values[name] if type(values[name]) is bool else getattr(defaults, name)
        for name, allowed in {
            "content_label": {"material", "number", "value", "number_material"},
            "diameter_label": {"value", "number"},
            "color_mode": {"follow", "custom"},
            "evidence_format": {"crop_jpeg", "full_png"},
            "input_mode": {"mixed", "manual", "candidate"},
            "gesture": {"click", "drag"},
            "manual_algorithm": {"assisted", "free"},
            "algorithm": ALGORITHM_IDS,
        }.items():
            values[name] = _choice(values[name], allowed, getattr(defaults, name))
        values["font_family"] = self.font_family.strip()[:200] if isinstance(self.font_family, str) else ""
        values["font_color"] = (self.font_color.upper() if isinstance(self.font_color, str)
                               and re.fullmatch(r"#[0-9a-fA-F]{6}", self.font_color) else defaults.font_color)
        values["font_size"] = _number(self.font_size, 12, 6, 72, integer=True)
        values["opacity"] = _number(self.opacity, 60, 20, 100, integer=True)
        values["retention_seconds"] = _number(self.retention_seconds, 1., 0., 10.)
        try:
            values["optical_analysis"] = OpticalAnalysisConfig.from_dict(
                self.optical_analysis, algorithm="optical_adaptive").to_dict()
        except (ValueError, TypeError, OverflowError):
            values["optical_analysis"] = OpticalAnalysisConfig(algorithm="optical_adaptive").to_dict()
        return type(self)(**values)

    @classmethod
    def from_dict(cls, payload) -> "TrialPreferences":
        if isinstance(payload, cls):
            return payload.normalized_copy()
        source = payload if isinstance(payload, dict) else {}
        return cls(**{item.name: source[item.name] for item in fields(cls) if item.name in source}).normalized_copy()

    def to_dict(self) -> dict:
        return asdict(self.normalized_copy())

    def assistance_dict(self) -> dict:
        values = self.to_dict()
        return {"version": 4, **{name: values[name] for name in (
            "input_mode", "manual_algorithm", "algorithm", "automatic", "visible",
            "opacity", "retention_seconds", "optical_analysis")}}

    @property
    def evidence_options(self) -> dict:
        settings = self.normalized_copy()
        return {"version": 1, "mode": settings.evidence_format if settings.evidence_enabled else "disabled"}

    @classmethod
    def from_legacy_assistance(cls, payload) -> "TrialPreferences":
        """Only shared preferences migrate; old session image flags never opt in."""
        if not isinstance(payload, dict):
            return cls().normalized_copy()
        values = {name: payload[name] for name in (
            "algorithm", "automatic", "visible", "opacity", "retention_seconds", "optical_analysis")
                  if name in payload}
        if payload.get("version") in (2, 3, 4):
            values.update(input_mode=payload.get("input_mode", "mixed"),
                          manual_algorithm=payload.get("manual_algorithm", "assisted"))
        else:
            values.update(input_mode="mixed", manual_algorithm=payload.get("input_mode", "assisted"))
        return cls.from_dict(values)
