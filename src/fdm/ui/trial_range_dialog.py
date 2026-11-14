"""Explicit adjudication of an already verified, immutable trial result."""
from __future__ import annotations

import math

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPainter, QPen, QColor, QPixmap
from PySide6.QtWidgets import (QApplication, QCheckBox, QDialog, QHBoxLayout,
    QLabel, QPushButton, QVBoxLayout)

from fdm.ui.widgets import NoWheelComboBox


def source_preview(decision) -> QImage:
    """Crop only the owned source frame, never the currently displayed frame."""
    import numpy as np
    intent = decision.intent
    frame = intent.token.frame
    points = decision.result.get("points") or (intent.start, intent.end)
    # Computation adapters also expose the final endpoints individually.
    start = decision.result.get("start", points[0])
    end = decision.result.get("end", points[-1])
    x0, y0 = min(start[0], end[0]), min(start[1], end[1])
    x1, y1 = max(start[0], end[0]), max(start[1], end[1])
    margin = max(24, math.hypot(x1 - x0, y1 - y0) * .35)
    left, top = max(0, int(x0 - margin)), max(0, int(y0 - margin))
    right, bottom = min(frame.size[0], math.ceil(x1 + margin)), min(frame.size[1], math.ceil(y1 + margin))
    pixels = np.ascontiguousarray(frame.pixels[top:bottom, left:right])
    if pixels.size == 0:
        return QImage()
    if pixels.ndim == 2:
        image = QImage(pixels.data, pixels.shape[1], pixels.shape[0], pixels.strides[0], QImage.Format.Format_Grayscale8).copy()
    elif pixels.shape[2] == 3:
        image = QImage(pixels.data, pixels.shape[1], pixels.shape[0], pixels.strides[0], QImage.Format.Format_BGR888).copy()
    else:
        rgba = np.ascontiguousarray(pixels[..., [2, 1, 0, 3]])
        image = QImage(rgba.data, rgba.shape[1], rgba.shape[0], rgba.strides[0], QImage.Format.Format_RGBA8888).copy()
    image = image.convertToFormat(QImage.Format.Format_RGB32)
    painter = QPainter(image)
    painter.setPen(QPen(QColor("#ff9f28"), max(1, image.width() / 280)))
    from PySide6.QtCore import QPointF
    painter.drawLine(QPointF(start[0] - left, start[1] - top), QPointF(end[0] - left, end[1] - top))
    painter.end()
    return image


class TrialRangeDialog(QDialog):
    choice = Signal(str, object, bool)
    configureRequested = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("直径越限 · 请确认本条测量")
        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.setWindowFlag(Qt.WindowType.WindowCloseButtonHint, False)
        self.setMinimumWidth(420)
        self._finished = False
        self._key = None
        self._preview_id = None
        self.intent_id = None
        layout = QVBoxLayout(self)
        self.details = QLabel(self)
        self.details.setWordWrap(True)
        self.details.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self.details)
        self.preview = QLabel(self)
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setFixedHeight(180)
        layout.addWidget(self.preview)
        note = QLabel("画面可继续变化。本条按测量完成时的原图、标定和视场入账；请选择后继续。", self)
        note.setWordWrap(True)
        layout.addWidget(note)
        self.target = NoWheelComboBox(self)
        self.target.setAccessibleName("本条测量改为类别")
        layout.addWidget(self.target)
        self.suggestion = QLabel(self)
        self.suggestion.setWordWrap(True)
        self.suggestion.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self.suggestion)
        self.configure = QPushButton("配置建议类别…", self)
        self.configure.clicked.connect(lambda: self.configureRequested.emit(self._suggestion_id))
        layout.addWidget(self.configure)
        self.remember = QCheckBox("记住本次选择（仅本试验、原类别及当前越限方向）", self)
        layout.addWidget(self.remember)
        self.error = QLabel(self)
        self.error.setWordWrap(True)
        self.error.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self.error)
        buttons = QHBoxLayout()
        self.keep = QPushButton("保留原类别", self)
        self.reclassify = QPushButton("改为指定类别", self)
        self.discard = QPushButton("放弃本条", self)
        for button in (self.keep, self.reclassify, self.discard):
            button.setAutoDefault(False)
            buttons.addWidget(button)
        self.keep.clicked.connect(lambda: self.choice.emit("keep", None, self.remember.isChecked()))
        self.reclassify.clicked.connect(lambda: self.choice.emit("reclassify", self.target.currentData(), self.remember.isChecked()))
        self.discard.clicked.connect(lambda: self.choice.emit("discard", None, self.remember.isChecked()))
        layout.addLayout(buttons)

    def update_decision(self, decision, session, suggestion=None):
        intent = decision.intent
        self.intent_id = intent.id
        try:
            material = session.material(intent.material_id, operator_id=intent.operator_id, direction=intent.direction)
            name = material.name
            limits = session.measurement_range(intent.material_id, operator_id=intent.operator_id, direction=intent.direction)
            self.keep.setEnabled(True)
        except ValueError:
            name = "类别已不可用，请重新改类或放弃"
            limits = decision.source_range
            self.keep.setEnabled(False)
        try:
            original_name = session.material(decision.original_material_id,
                operator_id=intent.operator_id, direction=intent.direction).name
        except ValueError:
            original_name = decision.source_range.get("material_name", decision.original_material_id)
        low, high = limits["lower_limit_um"], limits["upper_limit_um"]
        bounds = f"{'不限' if low is None else f'{low:g}'} ～ {'不限' if high is None else f'{high:g}'} μm"
        repeated = decision.requires_explicit_keep
        self.details.setText(f"测量 #{intent.measurement_number or intent.id[:6]} · {decision.result['diameter_um']:.4f} μm\n"
            f"原类别：{original_name}\n当前待入账类别：{name} · 允许范围：{bounds}"
            + ("\n改类后仍然越限，请再次明确确认；不会继续自动改类。" if repeated else ""))
        self.keep.setText("保留当前类别" if repeated else "保留原类别")
        materials = session.operators[intent.operator_id].materials[intent.direction]
        key = (intent.id, intent.material_id, repr(materials), repeated)
        if key != self._key:
            previous = self.target.currentData()
            new_intent = self._key is None or self._key[0] != intent.id
            self.target.clear()
            for item in materials:
                if item.id != intent.material_id:
                    self.target.addItem(item.name, item.id)
            preferred = suggestion["material_id"] if suggestion and suggestion["available"] else previous
            index = self.target.findData(preferred)
            if index >= 0:
                self.target.setCurrentIndex(index)
            self.reclassify.setEnabled(self.target.count() > 0)
            if new_intent or repeated:
                self.remember.setChecked(False)
            self._key = key
        if self._preview_id != intent.id:
            preview = source_preview(decision)
            self.preview.setPixmap(QPixmap.fromImage(preview).scaled(480, 180,
                Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
            self._preview_id = intent.id
        self._suggestion_id = suggestion["material_id"] if suggestion else None
        self.configure.setVisible(bool(suggestion and not suggestion["available"]))
        self.suggestion.setText((f"建议类别：{suggestion['material_name']}" +
            ("（尚未加入当前方向，请先配置）" if not suggestion["available"] else "")) if suggestion else "")

    def show_centered(self, origin=None):
        self.adjustSize()
        origin = origin or self.parentWidget()
        screen = origin.screen() if origin is not None else QApplication.primaryScreen()
        bounds = screen.availableGeometry()
        self.resize(min(self.width(), bounds.width()), min(self.height(), bounds.height()))
        self.move(bounds.center() - self.rect().center())
        self.show()
        self.move(self.pos() + bounds.center() - self.frameGeometry().center())
        self.raise_()
        self.activateWindow()

    def finish(self):
        self._finished = True
        super().done(QDialog.DialogCode.Accepted)
        self.deleteLater()

    def reject(self):
        # Esc and window-manager close cannot silently discard a measurement.
        if self._finished:
            super().reject()

    def closeEvent(self, event):
        if self._finished:
            super().closeEvent(event)
        else:
            event.ignore()
