"""Shared, view-only alignment picker for acquisition canvases."""
from __future__ import annotations

from PySide6.QtCore import QEvent, QSize, Qt, Signal
from PySide6.QtGui import QPalette
from PySide6.QtWidgets import (
    QGridLayout, QLabel, QMenu, QSizePolicy, QToolButton, QVBoxLayout,
    QWidget, QWidgetAction,
)

from fdm.ui.icons import themed_icon
from fdm.ui.acquisition_layout import CompactAcquisitionToolButton


ALIGNMENT_CHOICES = (
    ("top_left", "左上", "direction_top_left"),
    ("top", "上中", "direction_up"),
    ("top_right", "右上", "direction_top_right"),
    ("left", "左中", "direction_left"),
    ("center", "居中", "actual_size"),
    ("right", "右中", "direction_right"),
    ("bottom_left", "左下", "direction_bottom_left"),
    ("bottom", "下中", "direction_down"),
    ("bottom_right", "右下", "direction_bottom_right"),
)
_LABELS = {key: label for key, label, _ in ALIGNMENT_CHOICES}
_ICONS = {key: icon for key, _, icon in ALIGNMENT_CHOICES}


class AcquisitionAlignmentButton(CompactAcquisitionToolButton):
    """Emit an intent; only the binding's accepted state updates the picker.

    The entry does not take canvas focus. A narrow trial toolbar may collapse
    its text, while the tooltip and the popup retain the full current state.
    """

    alignmentSelected = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._alignment = None
        self.setObjectName("acquisitionAlignment")
        self.setAccessibleName("画面对齐")
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setIconSize(QSize(18, 18))
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        menu = QMenu(self)
        self.setMenu(menu)
        panel = QWidget(menu)
        column = QVBoxLayout(panel)
        column.setContentsMargins(8, 8, 8, 8)
        self.current_label = QLabel(panel)
        column.addWidget(self.current_label)
        grid = QGridLayout()
        grid.setSpacing(4)
        self.direction_buttons = {}
        for index, (key, label, _) in enumerate(ALIGNMENT_CHOICES):
            button = QToolButton(panel)
            button.setObjectName(f"acquisitionAlignment_{key}")
            button.setText(label)
            button.setAccessibleName(f"画面对齐：{label}")
            button.setToolTip(f"持续对齐到{label}，不改变显示比例")
            button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
            button.setIconSize(QSize(18, 18))
            button.setCheckable(True)
            button.setMinimumSize(52, 48)
            button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            button.clicked.connect(lambda checked=False, value=key: self._select(value))
            grid.addWidget(button, index // 3, index % 3)
            self.direction_buttons[key] = button
        column.addLayout(grid)
        note = QLabel("手动移动画面后转为自由位置", panel)
        column.addWidget(note)
        action = QWidgetAction(menu)
        action.setDefaultWidget(panel)
        menu.addAction(action)
        self.set_alignment("free")

    def alignment(self):
        return self._alignment

    def set_alignment(self, value):
        if value not in _LABELS and value != "free":
            raise ValueError("未知画面对齐方向")
        changed = value != self._alignment
        self._alignment = value
        for key, button in self.direction_buttons.items():
            button.setChecked(key == value)
        if changed:
            label = _LABELS.get(value, "自由位置")
            self.setText("自由位置" if value == "free" else "画面对齐")
            self.setToolTip(f"画面对齐：{label}\n对齐只改变位置，不改变显示比例。\n手动移动画面后转为自由位置。")
            self.setAccessibleDescription(f"当前：{label}")
            self.current_label.setText(f"画面对齐 · {label}")
            self._refresh_icons()

    def _select(self, value):
        # A rejected command must not leave the clicked cell checked. The
        # owner synchronizes accepted state after save/validation completes.
        self.set_alignment(self._alignment)
        self.menu().close()
        self.alignmentSelected.emit(value)

    def _refresh_icons(self):
        color = self.palette().color(QPalette.ColorRole.ButtonText).name()
        self.setIcon(themed_icon(_ICONS.get(self._alignment, "navigator"), color=color))
        for key, button in self.direction_buttons.items():
            button.setIcon(themed_icon(_ICONS[key], color=color))

    def sizeHint(self):
        # Keep geometry unchanged between free and anchored states. Qt's
        # platform style supplies the arrow, icon padding and control height.
        hint = super().sizeHint()
        text_width = max(self.fontMetrics().horizontalAdvance(text)
                         for text in ("自由位置", "画面对齐"))
        return QSize(text_width + self.iconSize().width() + 32, hint.height())

    def changeEvent(self, event):
        super().changeEvent(event)
        if not hasattr(self, "direction_buttons"):
            return
        if event.type() in (QEvent.Type.PaletteChange, QEvent.Type.ApplicationPaletteChange):
            self._refresh_icons()
        elif event.type() in (QEvent.Type.FontChange, QEvent.Type.StyleChange):
            self.updateGeometry()
