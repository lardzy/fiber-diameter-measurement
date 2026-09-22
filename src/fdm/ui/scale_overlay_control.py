from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QWheelEvent
from PySide6.QtWidgets import QMenu, QSizePolicy, QToolButton


class ScaleOverlayStatusButton(QToolButton):
    """A view control whose main action always opens the scale editor.

    The checked state describes visibility, not a mutually exclusive tool mode.
    Hiding is explicit in the adjacent menu (or the editor), so a visible bar can
    be edited again with a single click after finishing or closing the dock.
    """

    editRequested = Signal()
    visibilityRequested = Signal(bool)
    locateRequested = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("scaleOverlayStatusButton")
        self.setCheckable(True)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)
        self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        self.setAccessibleName("比例尺，点击编辑，展开菜单显示或隐藏")
        self.setToolTip("点击打开或返回比例尺编辑；箭头菜单可显示、隐藏或定位比例尺")
        menu = QMenu(self)
        menu.setObjectName("scaleOverlayStatusMenu")
        self._visible_action = menu.addAction("显示比例尺")
        self._visible_action.setCheckable(True)
        self._visible_action.triggered.connect(self.visibilityRequested)
        menu.addAction("编辑比例尺…").triggered.connect(self.editRequested)
        self._locate_action = menu.addAction("定位比例尺")
        self._locate_action.triggered.connect(self.locateRequested)
        self.setMenu(menu)
        self.clicked.connect(self.editRequested)
        self.setScaleState(False, False, False)

    def nextCheckState(self) -> None:
        # QAbstractButton's default click would toggle the visibility indicator
        # before the controller can open the editor. Only the controller owns it.
        pass

    def setScaleState(self, visible: bool, editing: bool, has_document: bool) -> None:
        self.setEnabled(has_document)
        self.setChecked(visible)
        state = "编辑" if editing else "显示" if visible else "关"
        self.setText(f"比例尺：{state}")
        self._visible_action.setChecked(visible)
        self._locate_action.setEnabled(visible and has_document)

    def wheelEvent(self, event: QWheelEvent) -> None:
        event.accept()
