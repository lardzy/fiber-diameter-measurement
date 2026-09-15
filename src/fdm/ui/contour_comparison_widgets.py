"""Small, palette-aware presentation widgets for the contour workspace."""
from PySide6.QtCore import QPointF, QRectF, QSize, Qt
from PySide6.QtGui import QPainter, QPainterPath, QPalette, QPen
from PySide6.QtWidgets import QCheckBox, QFrame, QLabel, QSizePolicy, QStyle, QStyleOptionButton, QVBoxLayout


class WorkspaceCheckBox(QCheckBox):
    """Native checkbox semantics with an indicator visible in either palette."""

    def paintEvent(self, event):
        super().paintEvent(event)
        option = QStyleOptionButton()
        self.initStyleOption(option)
        rect = QRectF(self.style().subElementRect(QStyle.SubElement.SE_CheckBoxIndicator, option, self)).adjusted(.5, .5, -.5, -.5)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        group = QPalette.ColorGroup.Active if self.isEnabled() else QPalette.ColorGroup.Disabled
        palette = self.palette()
        checked = self.isChecked()
        border = palette.color(group, QPalette.ColorRole.Highlight if checked else QPalette.ColorRole.PlaceholderText)
        painter.setPen(QPen(border, 1))
        painter.setBrush(palette.color(group, QPalette.ColorRole.Highlight if checked else QPalette.ColorRole.Base))
        painter.drawRoundedRect(rect, 3, 3)
        if checked:
            path = QPainterPath(QPointF(rect.left()+rect.width()*.22, rect.center().y()))
            path.lineTo(rect.left()+rect.width()*.43, rect.top()+rect.height()*.72)
            path.lineTo(rect.left()+rect.width()*.79, rect.top()+rect.height()*.27)
            painter.setPen(QPen(palette.color(group, QPalette.ColorRole.HighlightedText), 1.8, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPath(path)


class ElidedLabel(QLabel):
    """Keep full text accessible without letting a filename resize the workspace."""

    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.setMinimumWidth(0)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.setTextFormat(Qt.TextFormat.PlainText)
        self.setToolTip(text)

    def setText(self, text):
        super().setText(text)
        self.setToolTip(text)
        self.setAccessibleName(text)

    def minimumSizeHint(self):
        return QSize(0, self.sizeHint().height())

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setPen(self.palette().color(self.foregroundRole()))
        rect = self.contentsRect()
        text = self.fontMetrics().elidedText(self.text(), Qt.TextElideMode.ElideRight, rect.width())
        painter.drawText(rect, self.alignment() | Qt.AlignmentFlag.AlignVCenter, text)


class MetricCard(QFrame):
    def __init__(self, title, description, parent=None):
        super().__init__(parent)
        self.setObjectName("comparisonMetric")
        self.setMinimumWidth(0)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(3)
        self.title = ElidedLabel(title)
        self.title.setProperty("muted", True)
        self.value = ElidedLabel("—")
        font = self.value.font()
        font.setPointSizeF(font.pointSizeF() + 5)
        font.setBold(True)
        self.value.setFont(font)
        self.description = ElidedLabel(description)
        self.description.setProperty("muted", True)
        layout.addWidget(self.title)
        layout.addWidget(self.value)
        layout.addWidget(self.description)

    def set_value(self, value):
        self.value.setText(value)


WORKSPACE_STYLE = """
QDialog#contourWorkspace { background: palette(window); }
QDialog#contourWorkspace QLabel { background: transparent; }
QDialog#contourWorkspace QLabel[muted="true"] { color: palette(placeholder-text); }
QFrame#comparisonToolbar, QFrame#comparisonReference {
    background: palette(alternate-base); border: 1px solid palette(mid); border-radius: 7px;
}
QFrame#comparisonContext { background: palette(alternate-base); border-radius: 6px; }
QFrame#comparisonImagePanel, QFrame#comparisonRecords, QFrame#comparisonMetric {
    background: palette(base); border: 1px solid palette(mid); border-radius: 8px;
}
QFrame#comparisonImagePanel[active="true"] { border: 1px solid palette(highlight); }
QDialog#contourWorkspace QTabWidget::pane { border: 0; }
QDialog#contourWorkspace QTabBar::tab {
    min-height: 30px; padding: 3px 16px; border: 0; border-bottom: 2px solid transparent;
}
QDialog#contourWorkspace QTabBar::tab:selected {
    border-bottom: 2px solid palette(highlight); font-weight: 600;
}
QDialog#contourWorkspace QToolButton {
    min-height: 28px; padding: 3px 6px; border: 1px solid transparent; border-radius: 5px;
}
QDialog#contourWorkspace QToolButton:hover {
    background: palette(midlight); border-color: palette(mid);
}
QDialog#contourWorkspace QToolButton:checked {
    background: palette(highlight); color: palette(highlighted-text);
}
QDialog#contourWorkspace QToolButton:disabled { color: palette(placeholder-text); }
QDialog#contourWorkspace QToolButton[badge="true"] {
    background: palette(alternate-base); border-color: palette(mid); font-weight: 600; padding: 3px 10px;
}
QDialog#contourWorkspace QPushButton[primary="true"] {
    background: palette(highlight); color: palette(highlighted-text); border-color: palette(highlight);
}
QDialog#contourWorkspace QTableView {
    border: 0; background: palette(base); selection-background-color: palette(highlight);
    selection-color: palette(highlighted-text);
}
QDialog#contourWorkspace QHeaderView::section {
    border: 0; border-bottom: 1px solid palette(mid); padding: 6px 5px;
    background: palette(alternate-base);
}
"""
