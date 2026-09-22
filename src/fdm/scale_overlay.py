"""Shared, session-level scale-bar preferences (never document geometry)."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace

from fdm.units import millimeters_per_unit, resolve_length_unit


@dataclass(frozen=True, slots=True)
class ScaleOverlaySpec:
    length_mode: str = "auto"
    length: float = 50.0
    unit: str = "um"
    position: str = "bottom_right"
    # Fraction of the available travel of the COMPLETE bar + label bounds.
    relative_x: float = 1.0
    relative_y: float = 1.0
    font_mode: str = "auto"
    font_size: float = 18.0
    font_family: str = "Microsoft YaHei UI"
    bold: bool = True
    color: str = "#FF0000"
    text_color: str = "#FF0000"
    line_width: float = 2.5
    style: str = "ticks"
    text_position: str = "above"
    preview_region: str = "image"

    def __post_init__(self) -> None:
        for field, choices in (
            ("length_mode", ("auto", "custom")),
            ("font_mode", ("auto", "custom")),
            (
                "position",
                ("top_left", "top_right", "bottom_left", "bottom_right", "manual"),
            ),
            ("style", ("line", "ticks", "bar", "ticks_up", "ticks_down", "divisions")),
            ("text_position", ("above", "below")),
            ("preview_region", ("image", "viewport")),
        ):
            if getattr(self, field) not in choices:
                raise ValueError(f"无效比例尺设置：{field}")
        definition = resolve_length_unit(self.unit)
        if definition is None and self.unit != "px":
            raise ValueError("比例尺单位必须是 nm、μm、mm、cm、m 或 px。")
        if definition is not None:
            object.__setattr__(self, "unit", definition.code)
        for field in ("length", "font_size", "line_width"):
            if not math.isfinite(getattr(self, field)) or getattr(self, field) <= 0:
                raise ValueError(f"比例尺 {field} 必须是正数。")
        for field in ("relative_x", "relative_y"):
            if (
                not math.isfinite(getattr(self, field))
                or not 0 <= getattr(self, field) <= 1
            ):
                raise ValueError("比例尺相对位置必须在 0–1 之间。")
        for color in (self.color, self.text_color):
            if len(color) != 7 or color[0] != "#":
                raise ValueError("比例尺颜色必须是 #RRGGBB。")
            try:
                int(color[1:], 16)
            except ValueError as exc:
                raise ValueError("无效比例尺颜色。") from exc

    def with_unit(self, unit: str) -> ScaleOverlaySpec:
        """A unit change never changes an explicitly selected physical length."""
        source, target = millimeters_per_unit(self.unit), millimeters_per_unit(unit)
        if source is None or target is None:
            if unit != self.unit:
                return replace(self, unit=unit, length_mode="auto")
            return self
        return replace(self, unit=unit, length=self.length * source / target)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict | None) -> ScaleOverlaySpec | None:
        if payload is None:
            return None
        if not isinstance(payload, dict):
            raise ValueError("比例尺设置必须是对象。")
        return cls(
            **{
                key: value
                for key, value in payload.items()
                if key in cls.__dataclass_fields__
            }
        )

    @classmethod
    def from_legacy_settings(cls, settings) -> ScaleOverlaySpec:
        return cls(
            length=settings.scale_overlay_length_value,
            position=settings.scale_overlay_placement_mode,
            color=settings.scale_overlay_color,
            text_color=settings.scale_overlay_text_color,
            font_family=settings.scale_overlay_font_family,
            font_size=settings.scale_overlay_font_size,
            style=settings.scale_overlay_style,
        )


@dataclass(frozen=True, slots=True)
class ScaleOverlayLayout:
    """Source-pixel geometry; start/end are the OUTER horizontal bar edges.

    Their difference is the calibrated length, independent of line width.
    Fractional image pixels remain fractional until the raster paint device
    computes coverage; they are never rounded to change the measured length.
    """

    spec: ScaleOverlaySpec
    target: tuple[float, float, float, float]
    bounds: tuple[float, float, float, float]
    text_rect: tuple[float, float, float, float]
    start: tuple[float, float]
    end: tuple[float, float]
    label: str
    value: float
    unit: str
    font_size: int
    stroke: float
    tick_height: float

    @property
    def fraction(self) -> float:
        return (self.end[0] - self.start[0]) / self.target[2]


def nice_length(target: float) -> float:
    if not math.isfinite(target) or target <= 0:
        raise ValueError("比例尺目标长度无效。")
    exponent = math.floor(math.log10(target))
    return min(
        (
            factor * 10.0**power
            for power in range(exponent - 1, exponent + 2)
            for factor in (1, 2, 5)
        ),
        key=lambda value: abs(value - target),
    )
