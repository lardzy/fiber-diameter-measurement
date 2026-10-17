"""Shared geometry for capture and trial pages; image scale stays independent."""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QSizePolicy


SIDEBAR_MIN = 390
SIDEBAR_MAX = 520
SIDEBAR_DEFAULT = 432


def view_controls(parent, choice, title):
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
