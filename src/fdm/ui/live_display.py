"""CU display magnification uses an explicitly measured screen ruler.

This is independent of microscope calibration and never changes measurement
pixels. OS-reported DPI is not a substitute for measuring the displayed line.
"""
import json
import math

from PySide6.QtCore import QPointF
from PySide6.QtGui import QPainter, QPen
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QFormLayout, QLabel, QVBoxLayout, QWidget

from fdm.ui.widgets import NoWheelDoubleSpinBox


class ScreenRuler(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(56)
        self.setMinimumWidth(round(500 / self.devicePixelRatioF()) + 24)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setPen(QPen(self.palette().windowText().color(), 1))
        length = 500 / self.devicePixelRatioF()
        x, y = (self.width() - length) / 2, self.height() / 2
        painter.drawLine(QPointF(x, y), QPointF(x + length, y))
        for endpoint in (x, x + length):
            painter.drawLine(QPointF(endpoint, y - 7), QPointF(endpoint, y + 7))


def _settings():
    from fdm.settings import settings_file_path
    return settings_file_path().parent / "screen-rulers.json"


def _screen_key(screen):
    return json.dumps([screen.name(), screen.manufacturer(), screen.model(), screen.serialNumber()], allow_nan=False)


def _screen_geometry(screen):
    """Physical raster size distinguishes display modes from OS UI scaling."""
    size = screen.size()
    dpr = screen.devicePixelRatio()
    return [round(size.width() * dpr), round(size.height() * dpr)]


def _read_screen_rulers():
    try:
        saved = json.loads(_settings().read_text(encoding="utf-8"))
        return saved if isinstance(saved, dict) else {}
    except (OSError, ValueError):
        return {}


def screen_pixel_mm(screen):
    """Return measured mm per physical pixel, never an OS-DPI estimate.

    Legacy entries stored a bare 500-pixel ruler length and remain readable.
    New entries also store raster size so a resolution change needs a fresh
    check instead of silently reusing an invalid physical magnification.
    """
    if screen is None:
        return None
    entry = _read_screen_rulers().get(_screen_key(screen))
    if isinstance(entry, dict):
        if entry.get("raster_size") != _screen_geometry(screen):
            return None
        entry = entry.get("length_mm")
    try:
        length = float(entry)
    except (ValueError, TypeError):
        return None
    return length / 500 if math.isfinite(length) and 0 < length <= 2000 else None


def save_screen_ruler(screen, length_mm):
    length = float(length_mm)
    if not math.isfinite(length) or not 0 < length <= 2000:
        raise ValueError("显示器标线长度必须大于零且不超过 2000 mm")
    saved = _read_screen_rulers()
    saved[_screen_key(screen)] = {"length_mm": length, "raster_size": _screen_geometry(screen)}
    from fdm.atomic_io import atomic_write_text
    _settings().parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(_settings(), json.dumps(saved, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def request_magnification(parent, current=100.0):
    screen = parent.screen()
    dialog = QDialog(parent)
    dialog.setWindowTitle("指定物理倍率")
    layout = QVBoxLayout(dialog)
    explanation = QLabel("先用普通直尺测量下方标线的毫米长度。此设置只需对显示器设置一次，\n与显微镜标尺分开保存；更换显示器或显示模式后请复核。")
    explanation.setWordWrap(True)
    layout.addWidget(explanation)
    layout.addWidget(ScreenRuler(dialog))
    form = QFormLayout()
    length = NoWheelDoubleSpinBox()
    length.setRange(0, 2000)
    length.setDecimals(2)
    length.setSuffix(" mm")
    length.setSpecialValueText("请输入标线实测长度")
    length.setValue((screen_pixel_mm(screen) or 0) * 500)
    form.addRow("500 像素标线", length)
    factor = NoWheelDoubleSpinBox()
    factor.setRange(0.01, 100000)
    factor.setDecimals(2)
    factor.setValue(current)
    factor.setSuffix(" 倍")
    form.addRow("显示倍率", factor)
    layout.addLayout(form)
    buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
    ok = buttons.button(QDialogButtonBox.StandardButton.Ok)
    ok.setEnabled(length.value() > 0)
    length.valueChanged.connect(lambda value: ok.setEnabled(value > 0))
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    layout.addWidget(buttons)
    if dialog.exec() != QDialog.DialogCode.Accepted:
        return None
    save_screen_ruler(dialog.screen(), length.value())
    return factor.value(), length.value() / 500
