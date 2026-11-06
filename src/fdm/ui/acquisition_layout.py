"""Shared geometry for capture and trial pages; image scale stays independent."""
from PySide6.QtCore import Qt, QSize
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QSizePolicy, QToolButton


SIDEBAR_MIN = 390
SIDEBAR_MAX = 520
SIDEBAR_DEFAULT = 432


class CompactAcquisitionToolButton(QToolButton):
    """Keep a toolbar action reachable when a records dock narrows the view."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)

    def sizeHint(self):
        hint = super().sizeHint()
        return QSize(self.fontMetrics().horizontalAdvance(self.text()) + self.iconSize().width() + 32,
                     hint.height())

    def minimumSizeHint(self):
        return QSize(self.iconSize().width() + 18, self.sizeHint().height())

    def resizeEvent(self, event):
        super().resizeEvent(event)
        style = (Qt.ToolButtonStyle.ToolButtonTextBesideIcon
                 if self.width() >= self.sizeHint().width()
                 else Qt.ToolButtonStyle.ToolButtonIconOnly)
        if self.toolButtonStyle() != style:
            self.setToolButtonStyle(style)


def view_controls(parent, choice, title, *, alignment_button=None):
    bar = QFrame(parent)
    bar.setObjectName("acquisitionViewControls")
    bar.setFixedHeight(38)
    row = QHBoxLayout(bar)
    row.setContentsMargins(4, 0, 4, 0)
    caption = QLabel(title, bar)
    caption.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
    row.addWidget(caption, 1)
    choice.setMinimumWidth(150)
    row.addWidget(choice)
    if alignment_button is not None:
        row.addWidget(alignment_button)
    return bar


def configure_splitter(splitter, sidebar):
    splitter.setChildrenCollapsible(False)
    splitter.setHandleWidth(6)
    sidebar.setMinimumWidth(SIDEBAR_MIN)
    sidebar.setMaximumWidth(SIDEBAR_MAX)
    splitter.setStretchFactor(0, 1)
    splitter.setStretchFactor(1, 0)
    splitter.setSizes([800, SIDEBAR_DEFAULT])


def configure_status(status):
    status.setFixedHeight(30)
    status.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
    status.setSizeGripEnabled(False)
