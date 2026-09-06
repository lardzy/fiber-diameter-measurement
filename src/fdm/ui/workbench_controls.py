"""Small presentation components for the high-frequency measurement workflow.

These widgets emit intent only. Documents, tools, calibration and exports remain
owned by the existing main-window controllers.
"""
from __future__ import annotations

from PySide6.QtCore import QEvent, QSize, Qt, Signal
from PySide6.QtGui import QAction, QPainter, QPalette
from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QFrame, QGridLayout, QHBoxLayout, QLabel, QLayout, QLineEdit,
    QListWidget, QListWidgetItem, QMenu, QPushButton, QSizePolicy,
    QToolButton, QVBoxLayout, QWidget,
)

from fdm.ui.widgets import NoWheelComboBox


def action_button(action: QAction, parent: QWidget, *, text: str | None = None) -> QToolButton:
    # QAction's native iconText keeps toolbar labels stable on enabled/icon
    # changes without Python callbacks participating in QObject destruction.
    if text is not None:
        action.setIconText(text)
    button = QToolButton(parent)
    button.setDefaultAction(action)
    button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
    button.setIconSize(QSize(16, 16))
    # Clicking a workflow command must leave single-key canvas shortcuts usable.
    button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
    return button


class MeasurementContextBar(QFrame):
    documentActivated = Signal(int)
    groupActivated = Signal(object)

    def __init__(self, previous: QAction, following: QAction, snap: QAction, parent=None):
        super().__init__(parent)
        self.setObjectName("measurementContextBar")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._grid = QGridLayout(self)
        self._grid.setContentsMargins(8, 1, 8, 1)
        self._grid.setSpacing(6)
        self._grid.setSizeConstraint(QLayout.SizeConstraint.SetNoConstraint)
        self._navigation = QFrame(self)
        self._navigation.setObjectName("imageNavigationGroup")
        row = QHBoxLayout(self._navigation)
        row.setContentsMargins(3, 0, 3, 0)
        row.setSpacing(2)
        self.previousButton = action_button(previous, self, text="上一张")
        self.nextButton = action_button(following, self, text="下一张")
        row.addWidget(self.previousButton)
        self.documents = NoWheelComboBox(self)
        self.documents.setObjectName("quickDocumentSelector")
        self.documents.setAccessibleName("切换图片")
        self.documents.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.documents.setSizeAdjustPolicy(NoWheelComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.documents.setMinimumContentsLength(8)
        self.documents.setMinimumWidth(100)
        self.documents.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.documents.activated.connect(self.documentActivated)
        row.addWidget(self.documents, 1)
        row.addWidget(self.nextButton)
        self._editing = QWidget(self)
        row = QHBoxLayout(self._editing)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        category = QFrame(self._editing)
        category.setObjectName("measurementCategoryGroup")
        category_row = QHBoxLayout(category)
        category_row.setContentsMargins(10, 0, 3, 0)
        category_row.setSpacing(6)
        label = QLabel("新测量归类", category)
        label.setObjectName("measurementCategoryLabel")
        category_row.addWidget(label)
        self.groups = NoWheelComboBox(self)
        self.groups.setObjectName("quickMeasurementGroup")
        self.groups.setAccessibleName("新测量归类")
        self.groups.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.groups.setMinimumContentsLength(7)
        self.groups.setSizeAdjustPolicy(NoWheelComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.groups.setMinimumWidth(95)
        self.groups.setMaximumWidth(180)
        self.groups.activated.connect(lambda index: self.groupActivated.emit(self.groups.itemData(index)))
        category_row.addWidget(self.groups)
        row.addWidget(category)
        self.areaTools = QWidget(self)
        self.areaTools.setObjectName("quickAreaTools")
        self.areaLayout = QHBoxLayout(self.areaTools)
        self.areaLayout.setContentsMargins(0, 0, 0, 0)
        self.areaLayout.setSpacing(2)
        self.areaTools.hide()
        row.addWidget(self.areaTools)
        self.snapButton = action_button(snap, self)
        row.addWidget(self.snapButton)
        row.addStretch(1)
        self.calibrationButton = QToolButton(self)
        self.calibrationButton.setObjectName("persistentCalibrationButton")
        self.calibrationButton.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.calibrationButton.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.calibrationButton.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.calibrationButton.setIconSize(QSize(16, 16))
        self.calibrationMenu = QMenu(self.calibrationButton)
        self.calibrationButton.setMenu(self.calibrationMenu)
        self._compact = None
        self._arrange_controls(compact=True)

    def minimumSizeHint(self) -> QSize:
        # Advertise the wrapped minimum even while expanded, so QMainWindow
        # can shrink far enough to deliver the resize that switches rows.
        margins = self._grid.contentsMargins()
        width = max(
            self._navigation.minimumSizeHint().width() + self.calibrationButton.minimumSizeHint().width() + 6,
            self._editing.minimumSizeHint().width(),
        )
        return QSize(width + margins.left() + margins.right(), super().minimumSizeHint().height())

    def _arrange_controls(self, *, compact: bool) -> None:
        if self._compact == compact:
            return
        self._compact = compact
        for widget in (self._navigation, self._editing, self.calibrationButton):
            self._grid.removeWidget(widget)
        self._grid.addWidget(self._navigation, 0, 0)
        if compact:
            self._grid.addWidget(self.calibrationButton, 0, 1)
            self._grid.addWidget(self._editing, 1, 0, 1, 2)
        else:
            self._grid.addWidget(self._editing, 0, 1)
            self._grid.addWidget(self.calibrationButton, 0, 2)
        self._grid.setColumnStretch(0, 1)
        self.updateGeometry()

    def resizeEvent(self, event) -> None:
        self._arrange_controls(compact=event.size().width() < 1000)
        super().resizeEvent(event)

    def set_area_actions(self, actions: list[tuple[QAction, str]]) -> None:
        self.areaButtons = []
        for action, label in actions:
            button = action_button(action, self.areaTools, text=label)
            button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
            self.areaLayout.addWidget(button)
            self.areaButtons.append(button)

    def set_calibration(self, text: str, details: str, *, missing: bool, enabled: bool) -> None:
        button = self.calibrationButton
        button.setText(text)
        button.setToolTip(details)
        button.setAccessibleName(text)
        button.setEnabled(enabled)
        if button.property("uncalibrated") != missing:
            button.setProperty("uncalibrated", missing)
            button.style().unpolish(button)
            button.style().polish(button)


class WelcomePanel(QFrame):
    def __init__(self, open_images: QAction, open_project: QAction, capture: QAction, parent=None):
        super().__init__(parent)
        self.setObjectName("workspaceWelcome")
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.addStretch(1)
        title = QLabel("开始测量", self)
        title.setObjectName("welcomeTitle")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        hint = QLabel("可一次打开多张图片，逐张测量并统一导出", self)
        hint.setWordWrap(True)
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(hint)
        actions = QHBoxLayout()
        actions.addStretch(1)
        for action, label in ((open_images, "打开"), (open_project, "打开项目"), (capture, "实时采集")):
            actions.addWidget(action_button(action, self, text=label))
        actions.addStretch(1)
        layout.addLayout(actions)
        steps = QLabel("打开图片  →  确认标定  →  选择类别  →  开始测量", self)
        steps.setObjectName("welcomeHint")
        steps.setWordWrap(True)
        steps.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(steps)
        layout.addStretch(1)


class FixedLineLabel(QLabel):
    """Reserve a stable number of lines; long text remains in the tooltip."""

    def __init__(self, text: str = "", parent=None, *, lines: int = 1):
        self._lines = lines
        super().__init__(text, parent)
        self.setTextFormat(Qt.TextFormat.PlainText)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.setToolTip(text)

    def setText(self, text: str) -> None:
        super().setText(text)
        self.setToolTip(text)

    def sizeHint(self) -> QSize:
        return QSize(0, self.fontMetrics().lineSpacing() * self._lines + 2)

    def minimumSizeHint(self) -> QSize:
        return self.sizeHint()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setFont(self.font())
        painter.setPen(self.palette().color(QPalette.ColorRole.WindowText))
        rect = self.contentsRect()
        metrics = self.fontMetrics()
        for index, text in enumerate(self.text().splitlines()[:self._lines]):
            line = metrics.elidedText(text, Qt.TextElideMode.ElideRight, rect.width())
            painter.drawText(rect.left(), rect.top() + 1 + metrics.ascent() + index * metrics.lineSpacing(), line)


class CurrentMeasurementSummary(QFrame):
    editRequested = Signal()
    groupChangeRequested = Signal(str, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("currentMeasurementSummary")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.measurement_id: str | None = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(4)
        heading = QHBoxLayout()
        heading.addWidget(QLabel("当前对象", self))
        self.editButton = QPushButton("属性", self)
        self.editButton.setObjectName("currentObjectProperties")
        self.editButton.setCheckable(True)
        self.editButton.setToolTip("展开或收起当前对象的详细属性")
        self.editButton.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.editButton.clicked.connect(self.editRequested)
        heading.addWidget(self.editButton)
        layout.addLayout(heading)
        self.valueLabel = FixedLineLabel("未选中对象", self)
        self.valueLabel.setObjectName("currentMeasurementValue")
        layout.addWidget(self.valueLabel)
        self.sourceLabel = FixedLineLabel("点击画布中的测量对象\n或在测量记录中选择", self, lines=2)
        layout.addWidget(self.sourceLabel)
        self.groupCombo = NoWheelComboBox(self)
        self.groupCombo.setAccessibleName("当前对象类别")
        self.groupCombo.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.groupCombo.setMinimumWidth(0)
        self.groupCombo.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.groupCombo.setSizeAdjustPolicy(NoWheelComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.groupCombo.activated.connect(self._activate_group)
        self.groupCombo.currentTextChanged.connect(self.groupCombo.setToolTip)
        heading.insertWidget(1, self.groupCombo, 1)
        self.groupCombo.addItem("对象类别", None)
        self.groupCombo.setEnabled(False)

    def _activate_group(self, index: int) -> None:
        if self.measurement_id:
            self.groupChangeRequested.emit(self.measurement_id, self.groupCombo.itemData(index))


class CommandSearchDialog(QDialog):
    """Discover existing QActions; execution still uses their original guards."""

    def __init__(self, entries: list[tuple[QAction, str, str]], parent=None):
        super().__init__(parent)
        self.setWindowTitle("查找功能")
        self.resize(560, 460)
        self.entries = entries
        self.chosen_action: QAction | None = None
        layout = QVBoxLayout(self)
        self.search = QLineEdit(self)
        self.search.setPlaceholderText("输入功能名称，如：模板、直径、吸附、面积")
        self.search.setClearButtonEnabled(True)
        self.search.installEventFilter(self)
        layout.addWidget(self.search)
        self.results = QListWidget(self)
        self.results.setUniformItemSizes(True)
        layout.addWidget(self.results, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Open | QDialogButtonBox.StandardButton.Cancel, self)
        self.run_button = buttons.button(QDialogButtonBox.StandardButton.Open)
        self.run_button.setText("执行")
        buttons.accepted.connect(self._choose)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.search.textChanged.connect(self._filter)
        self.results.currentItemChanged.connect(self._selection_changed)
        self.results.itemDoubleClicked.connect(lambda _item: self._choose())
        self._filter("")
        self.search.setFocus()

    def _filter(self, query: str) -> None:
        tokens = query.casefold().split()
        self.results.clear()
        for action, path, aliases in self.entries:
            if not action.isVisible():
                continue
            title = action.text().replace("&", "")
            if not all(token in f"{title} {path} {aliases}".casefold() for token in tokens):
                continue
            shortcut = action.shortcut().toString()
            state = "" if action.isEnabled() else " · 当前不可用"
            item = QListWidgetItem(f"{title}{state}\n{path}" + (f"  ·  {shortcut}" if shortcut else ""))
            item.setData(Qt.ItemDataRole.UserRole, action)
            item.setToolTip(action.toolTip())
            self.results.addItem(item)
        if self.results.count():
            self.results.setCurrentRow(0)
        else:
            self.run_button.setEnabled(False)

    def _selection_changed(self, current, _previous) -> None:
        action = current.data(Qt.ItemDataRole.UserRole) if current else None
        self.run_button.setEnabled(action is not None and action.isEnabled() and action.isVisible())

    def _choose(self) -> None:
        item = self.results.currentItem()
        action = item.data(Qt.ItemDataRole.UserRole) if item else None
        if action is not None and action.isEnabled() and action.isVisible():
            self.chosen_action = action
            self.accept()

    def eventFilter(self, watched, event):
        if watched is self.search and event.type() == QEvent.Type.KeyPress:
            if event.key() in (Qt.Key.Key_Down, Qt.Key.Key_Up):
                delta = 1 if event.key() == Qt.Key.Key_Down else -1
                row = max(0, min(self.results.count() - 1, self.results.currentRow() + delta))
                self.results.setCurrentRow(row)
                return True
        return super().eventFilter(watched, event)
