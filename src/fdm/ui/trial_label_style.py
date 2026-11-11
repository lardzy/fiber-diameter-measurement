"""Screen-space trial labels shared by the live canvas and settings preview.

Only text presentation lives here. Record identities, measurement values and
source pixels remain owned by the trial model and controller.
"""
from __future__ import annotations

from functools import lru_cache
import math

from PySide6.QtCore import QPointF, QRectF, QSizeF
from PySide6.QtGui import QColor, QFont, QTextLayout, QTextOption
from PySide6.QtWidgets import QApplication


def trial_label_font(preferences, base_font: QFont | None = None) -> QFont:
    font = QFont(base_font if base_font is not None else QApplication.font())
    if preferences.font_family:
        font.setFamily(preferences.font_family)
    font.setPointSizeF(float(preferences.font_size))
    font.setBold(preferences.font_bold)
    font.setItalic(preferences.font_italic)
    return font


def trial_label_text(preferences, *, mode: str, number, material_name: str,
                     diameter_um: float, decimals: int = 2) -> str:
    """Format the same accepted record in a canvas or a settings preview."""
    content = preferences.diameter_label if mode == "diameter" else preferences.content_label
    ordinal = f"#{number}" if number is not None else "#—"
    if content == "number":
        return ordinal
    if content == "number_material":
        return f"{ordinal} · {material_name or '未知类别'}"
    if content == "value":
        if not math.isfinite(diameter_um):
            return "无效直径"
        text = f"{diameter_um:.{max(0, min(8, int(decimals)))}f}"
        if text.startswith("-") and float(text) == 0:
            text = text[1:]
        return f"{text} μm"
    return material_name or "未知类别"


@lru_cache(maxsize=512)
def _text_layout(text: str, font_key: str, max_width: float):
    """Bounded GUI-thread cache: preview frames do not reshape the same text."""
    font = QFont()
    font.fromString(font_key)
    layout = QTextLayout(text, font)
    options = QTextOption()
    options.setWrapMode(QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
    layout.setTextOption(options)
    layout.setCacheEnabled(True)
    layout.beginLayout()
    height, width = 0., 0.
    while True:
        line = layout.createLine()
        if not line.isValid():
            break
        line.setLineWidth(max_width)
        line.setPosition(QPointF(0., height))
        height += line.height()
        width = max(width, line.naturalTextWidth())
    layout.endLayout()
    return layout, QSizeF(math.ceil(width) + 10., math.ceil(height) + 4.)


def trial_label_size(text: str, font: QFont, max_width: float = 240.) -> QSizeF:
    """Full text extents, including padding, with long names wrapped."""
    return QSizeF(_text_layout(text, font.toString(), max(1., round(max_width - 10., 3)))[1])


def draw_trial_label(painter, rect: QRectF, text: str, color, preferences, *,
                     opacity: float = 1., base_font: QFont | None = None,
                     semantic_color: bool = False, max_width: float = 240.,
                     font: QFont | None = None) -> None:
    """Draw at screen scale using the exact layout measured above.

    Candidate/pending warnings retain their semantic color even when accepted
    labels have a custom color. No background or category bar is implicit.
    """
    font = font if font is not None else trial_label_font(preferences, base_font)
    layout, _ = _text_layout(text, font.toString(), max(1., round(max_width - 10., 3)))
    painter.save()
    painter.setOpacity(opacity)
    if preferences.label_background:
        painter.fillRect(rect, QColor(255, 255, 255, 232))
    actual_color = (color if semantic_color or preferences.color_mode == "follow"
                    else preferences.font_color)
    painter.setPen(QColor(actual_color))
    layout.draw(painter, rect.topLeft() + QPointF(5., 2.))
    painter.restore()
