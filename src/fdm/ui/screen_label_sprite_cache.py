from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import math

from PySide6.QtCore import QPointF, QRectF
from PySide6.QtGui import (
    QColor,
    QFont,
    QFontMetricsF,
    QImage,
    QPainter,
    QStaticText,
    QTransform,
)


DEFAULT_SCREEN_LABEL_SPRITE_CACHE_BYTES = 32 * 1024 * 1024
_HORIZONTAL_PADDING = 6.0
_VERTICAL_PADDING = 3.0


def _enum_value(value: object) -> int:
    raw_value = getattr(value, "value", value)
    return int(raw_value)


def _normalized_dpr(device_pixel_ratio: float) -> float:
    try:
        value = float(device_pixel_ratio)
    except (TypeError, ValueError):
        return 1.0
    if not math.isfinite(value) or value <= 0.0:
        return 1.0
    return min(8.0, max(0.25, value))


def _color_key(color: QColor | None) -> int | None:
    if color is None:
        return None
    normalized = QColor(color)
    return int(normalized.rgba()) if normalized.isValid() else int(QColor("#000000").rgba())


@dataclass(frozen=True, slots=True)
class ScreenLabelSpriteKey:
    """All display inputs that can change a cached label's pixels."""

    text: str
    font_family: str
    font_style_name: str
    font_pixel_size: int
    font_point_size: float
    font_weight: int
    font_italic: bool
    font_underline: bool
    font_strike_out: bool
    font_stretch: int
    letter_spacing_type: int
    letter_spacing: float
    text_rgba: int
    outline_rgba: int | None
    background_rgba: int | None
    device_pixel_ratio: float
    arrangement_mode: str

    @classmethod
    def from_values(
        cls,
        *,
        text: str,
        font: QFont,
        text_color: QColor,
        outline_color: QColor | None,
        background_color: QColor | None,
        device_pixel_ratio: float,
        arrangement_mode: str,
    ) -> ScreenLabelSpriteKey:
        return cls(
            text=str(text),
            font_family=font.family(),
            font_style_name=font.styleName(),
            font_pixel_size=font.pixelSize(),
            font_point_size=round(font.pointSizeF(), 6),
            font_weight=_enum_value(font.weight()),
            font_italic=font.italic(),
            font_underline=font.underline(),
            font_strike_out=font.strikeOut(),
            font_stretch=font.stretch(),
            letter_spacing_type=_enum_value(font.letterSpacingType()),
            letter_spacing=round(font.letterSpacing(), 6),
            text_rgba=int(QColor(text_color).rgba()),
            outline_rgba=_color_key(outline_color),
            background_rgba=_color_key(background_color),
            device_pixel_ratio=_normalized_dpr(device_pixel_ratio),
            arrangement_mode=str(arrangement_mode),
        )


@dataclass(frozen=True, slots=True)
class ScreenLabelSprite:
    """A complete transparent label ready for one drawImage() call."""

    image: QImage
    content_width: float
    content_height: float
    logical_width: float
    logical_height: float
    byte_size: int

    @property
    def logical_rect(self) -> QRectF:
        return QRectF(0.0, 0.0, self.logical_width, self.logical_height)


@dataclass(frozen=True, slots=True)
class ScreenLabelSpriteCacheStats:
    entries: int
    bytes: int
    max_bytes: int
    hits: int
    misses: int
    evictions: int


class ScreenLabelSpriteCache:
    """Byte-bounded LRU for complete screen-space measurement labels."""

    def __init__(self, max_bytes: int = DEFAULT_SCREEN_LABEL_SPRITE_CACHE_BYTES) -> None:
        self._max_bytes = max(1, int(max_bytes))
        self._entries: OrderedDict[ScreenLabelSpriteKey, ScreenLabelSprite] = OrderedDict()
        self._bytes = 0
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    @property
    def max_bytes(self) -> int:
        return self._max_bytes

    @property
    def byte_size(self) -> int:
        return self._bytes

    def __len__(self) -> int:
        return len(self._entries)

    def clear(self, *, reset_stats: bool = False) -> None:
        self._entries.clear()
        self._bytes = 0
        if reset_stats:
            self._hits = 0
            self._misses = 0
            self._evictions = 0

    def stats(self) -> ScreenLabelSpriteCacheStats:
        return ScreenLabelSpriteCacheStats(
            entries=len(self._entries),
            bytes=self._bytes,
            max_bytes=self._max_bytes,
            hits=self._hits,
            misses=self._misses,
            evictions=self._evictions,
        )

    def get_or_create(
        self,
        *,
        text: str,
        font: QFont,
        text_color: QColor,
        outline_color: QColor | None,
        background_color: QColor | None,
        device_pixel_ratio: float,
        arrangement_mode: str,
    ) -> ScreenLabelSprite:
        key = ScreenLabelSpriteKey.from_values(
            text=text,
            font=font,
            text_color=text_color,
            outline_color=outline_color,
            background_color=background_color,
            device_pixel_ratio=device_pixel_ratio,
            arrangement_mode=arrangement_mode,
        )
        cached = self._entries.get(key)
        if cached is not None:
            self._hits += 1
            self._entries.move_to_end(key)
            return cached

        self._misses += 1
        sprite = self._render_sprite(
            key,
            font=font,
            text_color=text_color,
            outline_color=outline_color,
            background_color=background_color,
        )
        if sprite.byte_size > self._max_bytes:
            return sprite

        while self._entries and self._bytes + sprite.byte_size > self._max_bytes:
            _old_key, old_sprite = self._entries.popitem(last=False)
            self._bytes -= old_sprite.byte_size
            self._evictions += 1
        self._entries[key] = sprite
        self._bytes += sprite.byte_size
        return sprite

    @staticmethod
    def _render_sprite(
        key: ScreenLabelSpriteKey,
        *,
        font: QFont,
        text_color: QColor,
        outline_color: QColor | None,
        background_color: QColor | None,
    ) -> ScreenLabelSprite:
        metrics = QFontMetricsF(font)
        source_lines = key.text.splitlines() or [""]
        line_spacing = max(1.0, metrics.lineSpacing())
        line_widths = tuple(metrics.horizontalAdvance(line or " ") for line in source_lines)
        content_width = max(line_widths, default=1.0)
        content_height = max(
            1.0,
            ((len(source_lines) - 1) * line_spacing) + metrics.height(),
        )
        logical_width = max(1.0, content_width + (_HORIZONTAL_PADDING * 2.0))
        logical_height = max(1.0, content_height + (_VERTICAL_PADDING * 2.0))
        dpr = key.device_pixel_ratio
        physical_width = max(1, int(math.ceil(logical_width * dpr)))
        physical_height = max(1, int(math.ceil(logical_height * dpr)))

        image = QImage(
            physical_width,
            physical_height,
            QImage.Format.Format_ARGB32_Premultiplied,
        )
        image.setDevicePixelRatio(dpr)
        image.fill(0)

        painter = QPainter(image)
        if not painter.isActive():
            raise RuntimeError("Unable to create a painter for a label sprite.")
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
            if background_color is not None:
                painter.fillRect(
                    QRectF(0.0, 0.0, logical_width, logical_height),
                    QColor(background_color),
                )
            painter.setFont(font)
            for index, (line, line_width) in enumerate(zip(source_lines, line_widths, strict=True)):
                static_text = QStaticText(line)
                try:
                    static_text.setPerformanceHint(QStaticText.PerformanceHint.AggressiveCaching)
                except AttributeError:
                    pass
                static_text.prepare(QTransform(), font)
                anchor = QPointF(
                    (logical_width - line_width) / 2.0,
                    _VERTICAL_PADDING + (index * line_spacing),
                )
                if outline_color is not None:
                    painter.setPen(QColor(outline_color))
                    for dx, dy in (
                        (1.0, 0.0),
                        (-1.0, 0.0),
                        (0.0, 1.0),
                        (0.0, -1.0),
                    ):
                        painter.drawStaticText(
                            QPointF(anchor.x() + dx, anchor.y() + dy),
                            static_text,
                        )
                painter.setPen(QColor(text_color))
                painter.drawStaticText(anchor, static_text)
        finally:
            painter.end()

        size_in_bytes = getattr(image, "sizeInBytes", None)
        byte_size = (
            int(size_in_bytes())
            if callable(size_in_bytes)
            else int(image.bytesPerLine() * image.height())
        )
        return ScreenLabelSprite(
            image=image,
            content_width=content_width,
            content_height=content_height,
            logical_width=logical_width,
            logical_height=logical_height,
            byte_size=byte_size,
        )


screen_label_sprite_cache = ScreenLabelSpriteCache()
