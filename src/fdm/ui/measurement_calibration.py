from __future__ import annotations

from PySide6.QtCore import QEvent, Qt
from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QHBoxLayout, QLabel, QMenu, QPushButton,
    QSizePolicy, QToolButton, QVBoxLayout, QWidget,
)

from fdm.models import Calibration
from fdm.ui.theme import refresh_widget_theme
from fdm.ui.widgets import NoWheelComboBox
from fdm.units import resolve_length_unit


def calibration_scale_text(calibration: Calibration) -> str:
    definition = resolve_length_unit(calibration.unit)
    unit = definition.symbol if definition else calibration.unit
    if not calibration.is_isotropic:
        return f"X {calibration.pixel_size_x:.6g} / Y {calibration.pixel_size_y:.6g} {unit}/px"
    return f"{calibration.pixel_size_x:.6g} {unit}/px"


class _CalibrationStatusButton(QToolButton):
    """Keep the physical ratio visible even with a long user-defined preset name."""

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self._source = ""
        self._scale = ""
        self.setMinimumWidth(0)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.setMenu(QMenu(self))

    def set_status(self, source: str, scale: str, details: str) -> None:
        self._source, self._scale = source, scale
        self.setToolTip(details)
        self.setAccessibleName(f"{source} · {scale}")
        self._fit_text()

    def _fit_text(self) -> None:
        metrics = self.fontMetrics()
        suffix = f" · {self._scale}"
        available = max(0, self.width() - 36)
        source = metrics.elidedText(self._source, Qt.TextElideMode.ElideRight,
                                    max(0, available - metrics.horizontalAdvance(suffix)))
        self.setText(source + suffix if source else self._scale)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._fit_text()

    def changeEvent(self, event) -> None:
        super().changeEvent(event)
        if event.type() == QEvent.Type.FontChange and hasattr(self, "_source"):
            self._fit_text()


class MeasurementCalibrationBar(QWidget):
    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setObjectName("measurementCalibrationBar")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)
        self.notice = QLabel(self)
        layout.addWidget(self.notice)
        self.status = _CalibrationStatusButton(self)
        layout.addWidget(self.status)
        self.actions = QWidget(self)
        buttons = QHBoxLayout(self.actions)
        buttons.setContentsMargins(0, 0, 0, 0)
        buttons.setSpacing(6)
        self.choose = QPushButton("应用标尺", self.actions)
        self.choose.setProperty("emphasis", "primary")
        self.choose.setMenu(QMenu(self.choose))
        self.draw = QPushButton("手动标定", self.actions)
        self.draw.setToolTip("在图像上画一段已知长度的线，再输入实际长度。")
        buttons.addWidget(self.choose, 1)
        buttons.addWidget(self.draw, 1)
        layout.addWidget(self.actions)
        self.show_notice("请先打开图像")

    def show_notice(self, message: str, *, uncalibrated: bool = False) -> None:
        tone = "warning" if uncalibrated else None
        if self.notice.property("statusTone") != tone:
            self.notice.setProperty("statusTone", tone)
            refresh_widget_theme(self.notice)
        self.notice.setText(message)
        self.notice.show()
        self.actions.setVisible(uncalibrated)
        self.status.hide()

    def show_scale(self, source: str, scale: str, details: str) -> None:
        self.status.set_status(source, scale, details)
        self.notice.hide()
        self.actions.hide()
        self.status.show()


class CalibrationManagerDialog(QDialog):
    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setWindowTitle("标定详情与标尺管理")
        self.resize(460, 280)
        layout = QVBoxLayout(self)
        self.details = QLabel(self)
        self.details.setTextFormat(Qt.TextFormat.PlainText)
        self.details.setWordWrap(True)
        self.details.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.details.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.details)
        layout.addWidget(QLabel("已有标尺", self))
        self.presets = NoWheelComboBox(self)
        self.presets.setMinimumContentsLength(10)
        self.presets.setSizeAdjustPolicy(NoWheelComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.presets.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.presets.currentTextChanged.connect(self.presets.setToolTip)
        layout.addWidget(self.presets)
        management = QHBoxLayout()
        self.add = QPushButton("新增", self)
        self.edit = QPushButton("编辑", self)
        self.delete = QPushButton("删除", self)
        self.import_cu = QPushButton("导入 CU 标尺", self)
        for button in (self.add, self.edit, self.delete, self.import_cu):
            management.addWidget(button)
        layout.addLayout(management)
        layout.addStretch(1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, parent=self)
        self.apply = buttons.addButton("应用所选标尺…", QDialogButtonBox.ButtonRole.ActionRole)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
