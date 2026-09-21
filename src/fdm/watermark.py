"""Persistent, image-relative watermark settings (no GUI dependencies)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from datetime import datetime
import math
import re


@dataclass(frozen=True, slots=True)
class WatermarkSpec:
    enabled: bool = False
    kind: str = "text"
    text: str = ""
    font_family: str = ""
    bold: bool = False
    italic: bool = False
    color: str = "#666666"
    logo_sha256: str = ""
    layout: str = "single"
    anchor: str = "bottom_right"
    opacity: float = 0.25
    width_ratio: float = 0.25
    rotation: float = 0.0
    offset_x: float = 0.02
    offset_y: float = 0.02
    gap_x: float = 1.0
    gap_y: float = 1.0
    include_datetime: bool = False
    datetime_text: str = ""

    def __post_init__(self) -> None:
        if self.kind not in {"text", "logo"} or self.layout not in {"single", "tile"}:
            raise ValueError("水印内容或布局类型无效")
        if self.anchor not in ANCHORS:
            raise ValueError("水印对齐位置无效")
        for name in ("enabled", "bold", "italic", "include_datetime"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"水印 {name} 必须为布尔值")
        for name in ("text", "font_family", "color", "logo_sha256", "datetime_text"):
            if not isinstance(getattr(self, name), str):
                raise ValueError(f"水印 {name} 必须为文字")
        if not re.fullmatch(r"#[0-9a-fA-F]{6}", self.color):
            raise ValueError("水印颜色必须为 RGB 颜色")
        if self.logo_sha256 and not re.fullmatch(r"[0-9a-f]{64}", self.logo_sha256):
            raise ValueError("水印 Logo 内容标识无效")
        if self.datetime_text:
            if not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}", self.datetime_text):
                raise ValueError("水印日期时间格式应为 YYYY-MM-DD HH:mm:ss")
            try:
                datetime.fromisoformat(self.datetime_text)
            except ValueError as exc:
                raise ValueError("水印日期或时间无效") from exc
        for name, low, high in (
            ("opacity", 0, 1), ("width_ratio", 0.01, 2),
            ("rotation", -180, 180), ("offset_x", -1, 1),
            ("offset_y", -1, 1), ("gap_x", 0, 10), ("gap_y", 0, 10),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"水印 {name} 必须为数值")
            if not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f"水印 {name} 超出有效范围")

    @property
    def asset_path(self) -> str:
        return f"watermarks/{self.logo_sha256}.png" if self.logo_sha256 else ""

    def validate_content(self) -> None:
        if self.enabled:
            if self.kind == "text" and not self.text.strip():
                raise ValueError("请输入水印文字")
            if self.kind == "logo" and not self.logo_sha256:
                raise ValueError("请选择水印 Logo")
            if self.include_datetime and not self.datetime_text:
                raise ValueError("请选择水印日期和时间")

    def to_dict(self) -> dict:
        payload = asdict(self)
        if not self.include_datetime and not self.datetime_text:
            payload.pop("include_datetime")
            payload.pop("datetime_text")
        return payload

    @classmethod
    def from_dict(cls, payload: dict | None) -> WatermarkSpec | None:
        if payload is None:
            return None
        if not isinstance(payload, dict):
            raise ValueError("水印配置必须为对象")
        names = {item.name for item in fields(cls)}
        return cls(**{key: value for key, value in payload.items() if key in names})


ANCHORS = {
    "top_left": (0, 0), "top_center": (1, 0), "top_right": (2, 0),
    "middle_left": (0, 1), "center": (1, 1), "middle_right": (2, 1),
    "bottom_left": (0, 2), "bottom_center": (1, 2), "bottom_right": (2, 2),
}
