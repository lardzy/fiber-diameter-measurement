from __future__ import annotations

from PySide6.QtCore import QEvent, QPoint, QRect, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QAction, QColor, QFont, QFontMetrics, QIcon, QMouseEvent, QPainter, QPalette, QPen
from PySide6.QtWidgets import (
    QAbstractScrollArea,
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QFontComboBox,
    QHBoxLayout,
    QLabel,
    QLayout,
    QLayoutItem,
    QMenu,
    QSizePolicy,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QWidgetItem,
)


def _repolish(widget: QWidget) -> None:
    style = widget.style()
    style.unpolish(widget)
    style.polish(widget)
    widget.update()


def _is_dark_palette(widget: QWidget) -> bool:
    return widget.palette().color(QPalette.ColorRole.Window).lightnessF() < 0.5


def _application_palette(widget: QWidget) -> QPalette:
    app = QApplication.instance()
    return app.palette() if app is not None else widget.palette()


def _application_palette_is_dark(widget: QWidget) -> bool:
    return _application_palette(widget).color(QPalette.ColorRole.Window).lightnessF() < 0.5


def _redirect_wheel_to_inspector_scroll(widget: QWidget, event) -> None:
    """Scroll the inspector page without letting an editor mutate its value."""

    parent = widget.parentWidget()
    while parent is not None:
        if (
            isinstance(parent, QAbstractScrollArea)
            and parent.objectName() == "measurementInspectorScroll"
        ):
            bar = parent.verticalScrollBar()
            pixel_delta = event.pixelDelta().y() if hasattr(event, "pixelDelta") else 0
            angle_delta = event.angleDelta().y() if hasattr(event, "angleDelta") else 0
            if pixel_delta:
                amount = -int(pixel_delta)
            elif angle_delta:
                amount = -int(round(angle_delta / 120.0 * max(12, bar.singleStep() * 3)))
            else:
                amount = 0
            if amount:
                bar.setValue(bar.value() + amount)
            event.accept()
            return
        parent = parent.parentWidget()
    event.ignore()


class CollapsibleSection(QFrame):
    """Restrained, keyboard-accessible section with a persistent header."""

    expandedChanged = Signal(bool)
    contentHeightChanged = Signal(int)

    def __init__(
        self,
        title: str,
        *,
        expanded: bool = False,
        summary: str = "",
        resizable: bool = False,
        content_height: int | None = None,
        minimum_content_height: int = 120,
        maximum_content_height: int = 1200,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._resizable = bool(resizable)
        self._minimum_content_height = max(40, int(minimum_content_height))
        self._maximum_content_height = max(
            self._minimum_content_height,
            int(maximum_content_height),
        )
        initial_height = (
            self._minimum_content_height
            if content_height is None
            else int(content_height)
        )
        self._remembered_content_height = max(
            self._minimum_content_height,
            min(self._maximum_content_height, initial_height),
        )
        self.setObjectName("collapsibleSection")
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 6, 8, 8)
        root.setSpacing(6)
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(6)
        self.toggleButton = QToolButton(self)
        self.toggleButton.setText(title)
        self.toggleButton.setCheckable(True)
        self.toggleButton.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.toggleButton.setAccessibleName(title)
        self.summaryLabel = QLabel(summary, self)
        self.summaryLabel.setObjectName("collapsibleSectionSummary")
        self.summaryLabel.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Fixed,
        )
        self.summaryLabel.setMinimumWidth(0)
        header.addWidget(self.toggleButton)
        header.addStretch(1)
        header.addWidget(self.summaryLabel)
        root.addLayout(header)
        self.contentWidget = QWidget(self)
        self.contentLayout = QVBoxLayout(self.contentWidget)
        self.contentLayout.setContentsMargins(0, 0, 0, 0)
        self.contentLayout.setSpacing(6)
        root.addWidget(self.contentWidget)
        self.resizeHandle = _SectionResizeHandle(title, self)
        self.resizeHandle.setVisible(self._resizable)
        self.resizeHandle.dragDelta.connect(self._resize_content_by)
        root.addWidget(self.resizeHandle)
        self.toggleButton.toggled.connect(self._on_toggled)
        if self._resizable:
            self.setContentHeight(self._remembered_content_height, emit_signal=False)
        self.setExpanded(expanded, emit_signal=False)
        self.setStyleSheet(
            "QFrame#collapsibleSection { border: 1px solid palette(mid); border-radius: 7px; }"
            "QLabel#collapsibleSectionSummary { color: palette(placeholder-text); }"
        )

    def setContentWidget(self, widget: QWidget) -> None:
        while self.contentLayout.count():
            item = self.contentLayout.takeAt(0)
            old_widget = item.widget()
            if old_widget is not None and old_widget is not widget:
                old_widget.setParent(None)
        self.contentLayout.addWidget(widget)

    def isExpanded(self) -> bool:
        return self.toggleButton.isChecked()

    def contentHeight(self) -> int:
        return int(self._remembered_content_height)

    def setContentHeight(self, height: int, *, emit_signal: bool = True) -> None:
        if not self._resizable:
            return
        normalized = max(
            self._minimum_content_height,
            min(self._maximum_content_height, int(height)),
        )
        changed = normalized != self._remembered_content_height
        self._remembered_content_height = normalized
        self.contentWidget.setFixedHeight(normalized)
        self.contentWidget.updateGeometry()
        self.updateGeometry()
        if emit_signal and changed:
            self.contentHeightChanged.emit(normalized)

    def setExpanded(self, expanded: bool, *, emit_signal: bool = True) -> None:
        expanded = bool(expanded)
        self.toggleButton.blockSignals(True)
        self.toggleButton.setChecked(expanded)
        self.toggleButton.blockSignals(False)
        self._apply_expanded(expanded)
        if emit_signal:
            self.expandedChanged.emit(expanded)

    def setSummary(self, text: str) -> None:
        self.summaryLabel.setText(str(text or ""))
        self.summaryLabel.setToolTip(str(text or ""))
        self.summaryLabel.setVisible(bool(str(text or "").strip()))

    def _on_toggled(self, expanded: bool) -> None:
        self._apply_expanded(expanded)
        self.expandedChanged.emit(bool(expanded))

    def _apply_expanded(self, expanded: bool) -> None:
        self.toggleButton.setArrowType(
            Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow
        )
        self.contentWidget.setVisible(bool(expanded))
        self.resizeHandle.setVisible(bool(expanded) and self._resizable)
        self.summaryLabel.setVisible(bool(self.summaryLabel.text().strip()))
        self.updateGeometry()

    def _resize_content_by(self, delta: int) -> None:
        self.setContentHeight(self._remembered_content_height + int(delta))


class _SectionResizeHandle(QWidget):
    """Small bottom grip that grows only its owning section downward."""

    dragDelta = Signal(int)

    def __init__(self, section_title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._last_global_y: float | None = None
        self.setFixedHeight(10)
        self.setCursor(Qt.CursorShape.SizeVerCursor)
        self.setAccessibleName(f"调整{section_title}高度")
        self.setToolTip("上下拖动以调整此区域高度；下次启动会恢复")

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._last_global_y = float(event.globalPosition().y())
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._last_global_y is None:
            super().mouseMoveEvent(event)
            return
        current_y = float(event.globalPosition().y())
        delta = int(round(current_y - self._last_global_y))
        if delta:
            self._last_global_y = current_y
            self.dragDelta.emit(delta)
        event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._last_global_y is not None:
            self._last_global_y = None
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        painter = QPainter(self)
        pen = QPen(self.palette().color(QPalette.ColorRole.Mid))
        pen.setWidth(1)
        painter.setPen(pen)
        y = self.rect().center().y()
        half_width = min(34, max(8, self.width() // 5))
        painter.drawLine(self.rect().center().x() - half_width, y, self.rect().center().x() + half_width, y)


class NoWheelComboBox(QComboBox):
    """Combo box that never changes selection from an incidental wheel."""

    def wheelEvent(self, event) -> None:
        _redirect_wheel_to_inspector_scroll(self, event)


class NoWheelSpinBox(QSpinBox):
    """Integer editor that leaves wheel gestures to the containing scroller."""

    def wheelEvent(self, event) -> None:
        _redirect_wheel_to_inspector_scroll(self, event)


class NoWheelDoubleSpinBox(QDoubleSpinBox):
    """Float editor that leaves wheel gestures to the containing scroller."""

    def wheelEvent(self, event) -> None:
        _redirect_wheel_to_inspector_scroll(self, event)


class NoWheelFontComboBox(QFontComboBox):
    """Font selector protected from accidental wheel-based changes."""

    def wheelEvent(self, event) -> None:
        _redirect_wheel_to_inspector_scroll(self, event)


class MeasurementGroupComboBox(QComboBox):
    """Category combo used inside the measurement table.

    It should only change by explicit click/open interactions, not by wheel
    scrolling while the user is browsing the canvas.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)

    def wheelEvent(self, event) -> None:
        _redirect_wheel_to_inspector_scroll(self, event)


class FlowLayout(QLayout):
    def __init__(self, parent: QWidget | None = None, *, h_spacing: int = 6, v_spacing: int = 6) -> None:
        super().__init__(parent)
        self._items: list[QLayoutItem] = []
        self._h_spacing = h_spacing
        self._v_spacing = v_spacing
        self.setContentsMargins(0, 0, 0, 0)

    def __del__(self) -> None:
        while self.count():
            self.takeAt(0)

    def addItem(self, item: QLayoutItem) -> None:
        self._items.append(item)

    def addWidget(self, widget: QWidget) -> None:
        layout_parent = self.parentWidget()
        if layout_parent is not None and widget.parent() is not layout_parent:
            widget.setParent(layout_parent)
        self.addItem(QWidgetItem(widget))

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index: int) -> QLayoutItem | None:
        if 0 <= index < len(self._items):
            return self._items[index]
        return None

    def takeAt(self, index: int) -> QLayoutItem | None:
        if 0 <= index < len(self._items):
            return self._items.pop(index)
        return None

    def expandingDirections(self) -> Qt.Orientations:
        return Qt.Orientations()

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        return self._do_layout(QRect(0, 0, width, 0), test_only=True)

    def setGeometry(self, rect: QRect) -> None:
        super().setGeometry(rect)
        self._do_layout(rect, test_only=False)

    def sizeHint(self) -> QSize:
        return self._preferred_size()

    def minimumSize(self) -> QSize:
        size = QSize()
        for item in self._items:
            widget = item.widget()
            if widget is not None and widget.isHidden():
                continue
            size = size.expandedTo(item.minimumSize())
        left, top, right, bottom = self.getContentsMargins()
        size += QSize(left + right, top + bottom)
        return size

    def _preferred_size(self) -> QSize:
        visible_items = [
            item
            for item in self._items
            if item.widget() is None or not item.widget().isHidden()
        ]
        if not visible_items:
            left, top, right, bottom = self.getContentsMargins()
            return QSize(left + right, top + bottom)
        total_width = 0
        max_height = 0
        for item in visible_items:
            hint = item.sizeHint()
            total_width += hint.width()
            max_height = max(max_height, hint.height())
        total_width += self._h_spacing * max(0, len(visible_items) - 1)
        left, top, right, bottom = self.getContentsMargins()
        return QSize(total_width + left + right, max_height + top + bottom)

    def _do_layout(self, rect: QRect, *, test_only: bool) -> int:
        left, top, right, bottom = self.getContentsMargins()
        effective_rect = rect.adjusted(left, top, -right, -bottom)
        x = effective_rect.x()
        y = effective_rect.y()
        line_height = 0

        for item in self._items:
            widget = item.widget()
            if widget is not None and widget.isHidden():
                continue
            hint = item.sizeHint()
            next_x = x + hint.width() + self._h_spacing
            if line_height > 0 and next_x - self._h_spacing > effective_rect.right() + 1:
                x = effective_rect.x()
                y = y + line_height + self._v_spacing
                next_x = x + hint.width() + self._h_spacing
                line_height = 0
            if not test_only:
                item.setGeometry(QRect(QPoint(x, y), hint))
            x = next_x
            line_height = max(line_height, hint.height())

        used_height = y + line_height - rect.y() + bottom
        return max(0, used_height)


class FiberGroupListItemWidget(QWidget):
    HEIGHT = 38
    DOT_SIZE = 10
    COUNT_COLUMN_WIDTH = 76
    RIGHT_MARGIN = 12

    def __init__(
        self,
        label: str,
        current_count: int,
        project_count: int,
        color: str,
        *,
        selected: bool = False,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._label = label
        self._current_count = max(0, int(current_count))
        self._project_count = max(0, int(project_count))
        self._color = QColor(color)
        self._selected = bool(selected)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setFixedHeight(self.HEIGHT)

    def setSelected(self, selected: bool) -> None:
        selected = bool(selected)
        if self._selected == selected:
            return
        self._selected = selected
        self.update()

    def setCounts(self, current_count: int, project_count: int) -> None:
        current_count = max(0, int(current_count))
        project_count = max(0, int(project_count))
        if self._current_count == current_count and self._project_count == project_count:
            return
        self._current_count = current_count
        self._project_count = project_count
        self.update()

    def labelText(self) -> str:
        return self._label

    def currentCountValue(self) -> int:
        return self._current_count

    def projectCountValue(self) -> int:
        return self._project_count

    def countText(self) -> str:
        return f"{self._current_count}/{self._project_count}"

    def sizeHint(self) -> QSize:
        return QSize(256, self.HEIGHT)

    def minimumSizeHint(self) -> QSize:
        return QSize(188, self.HEIGHT)

    def _resolved_colors(self) -> tuple[QColor, QColor, QColor, QColor, QColor, QColor]:
        dark_palette = _is_dark_palette(self)
        if self._selected:
            return (
                QColor("#12343B"),
                QColor("#00A6A6"),
                QColor("#F4FBFF"),
                QColor(255, 255, 255, 34),
                QColor(255, 255, 255, 48),
                QColor("#F4FBFF"),
            )
        if dark_palette:
            return (
                QColor(255, 255, 255, 20),
                QColor(255, 255, 255, 34),
                QColor("#E7ECF2"),
                QColor(255, 255, 255, 18),
                QColor(255, 255, 255, 28),
                QColor("#D9E2EC"),
            )
        return (
            QColor(15, 23, 42, 10),
            QColor(15, 23, 42, 34),
            QColor("#182430"),
            QColor(15, 23, 42, 12),
            QColor(15, 23, 42, 20),
            QColor("#223142"),
        )

    def paintEvent(self, event) -> None:
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        # Keep the rounded border off the widget edge so the left/top strokes
        # do not get clipped by the list viewport on dense layouts.
        rect = self.rect().adjusted(1, 1, -2, -2)
        background, border, text_color, badge_background, badge_border, badge_text = self._resolved_colors()

        painter.setPen(QPen(border, 1))
        painter.setBrush(background)
        painter.drawRoundedRect(rect, 10, 10)

        dot_x = rect.x() + 14
        dot_y = rect.y() + (rect.height() - self.DOT_SIZE) // 2
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(self._color if self._color.isValid() else QColor("#7BD389"))
        painter.drawEllipse(QRectF(dot_x, dot_y, self.DOT_SIZE, self.DOT_SIZE))

        badge_font = QFont(self.font())
        badge_font.setPointSizeF(max(8.0, badge_font.pointSizeF() - 0.25))
        badge_rect = QRect(
            rect.right() - self.RIGHT_MARGIN - self.COUNT_COLUMN_WIDTH,
            rect.y() + 8,
            self.COUNT_COLUMN_WIDTH,
            rect.height() - 16,
        )
        painter.setPen(QPen(badge_border, 1))
        painter.setBrush(badge_background)
        painter.drawRoundedRect(QRectF(badge_rect), badge_rect.height() / 2, badge_rect.height() / 2)

        text_font = QFont(self.font())
        text_font.setWeight(QFont.Weight.DemiBold if self._selected else QFont.Weight.Medium)
        painter.setFont(text_font)
        painter.setPen(text_color)
        text_left = dot_x + self.DOT_SIZE + 14
        text_rect = QRect(text_left, rect.y(), max(0, badge_rect.left() - text_left - 10), rect.height())
        text = QFontMetrics(text_font).elidedText(self._label, Qt.TextElideMode.ElideRight, text_rect.width())
        painter.drawText(text_rect, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, text)

        painter.setFont(badge_font)
        painter.setPen(badge_text)
        painter.drawText(badge_rect, Qt.AlignmentFlag.AlignCenter, self.countText())


class ToolStripActionButton(QToolButton):
    HEIGHT = 40
    COMPACT_WIDTH = 40
    ICON_SIZE = 16

    def __init__(self, action: QAction, parent=None) -> None:
        super().__init__(parent)
        self._full_text = action.text()
        self._compact_mode = False
        self.setDefaultAction(action)
        self.setProperty("primaryTool", True)
        self.setProperty("compactTool", False)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setIconSize(QSize(self.ICON_SIZE, self.ICON_SIZE))
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self.setMinimumWidth(self.COMPACT_WIDTH)
        self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.setFixedHeight(self.HEIGHT)
        self._expanded_width_hint = max(86, self._calculate_expanded_width())

    def _calculate_expanded_width(self) -> int:
        metrics = QFontMetrics(self.font())
        return 14 + self.ICON_SIZE + 8 + metrics.horizontalAdvance(self._full_text) + 14

    def expandedWidthHint(self) -> int:
        return self._expanded_width_hint

    def isCompactMode(self) -> bool:
        return self._compact_mode

    def setCompactMode(self, enabled: bool) -> None:
        enabled = bool(enabled)
        if self._compact_mode == enabled:
            return
        self._compact_mode = enabled
        self.setProperty("compactTool", enabled)
        self.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonIconOnly if enabled else Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )
        self.setToolTip(self._full_text)
        self.updateGeometry()
        _repolish(self)

    def sizeHint(self) -> QSize:
        width = self.COMPACT_WIDTH if self._compact_mode else self._expanded_width_hint
        return QSize(width, self.HEIGHT)

    def minimumSizeHint(self) -> QSize:
        return QSize(self.COMPACT_WIDTH, self.HEIGHT)


class OverlayToolSplitButton(QToolButton):
    """Standard split tool button used by grouped measurement tools.

    The previous implementation painted and dispatched both halves manually.
    Using ``QToolButton.MenuButtonPopup`` keeps font metrics, keyboard focus,
    menu indicators, disabled states and platform accessibility in Qt's normal
    control path while retaining the small compatibility API used by
    ``MainWindow`` and ``MeasurementToolStrip``.
    """

    primaryTriggered = Signal()

    HEIGHT = ToolStripActionButton.HEIGHT
    EXPANDED_MIN_WIDTH = 108
    COMPACT_MIN_WIDTH = 56
    MENU_WIDTH = 28
    ICON_SIZE = ToolStripActionButton.ICON_SIZE

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._current_kind = ""
        self._compact_mode = False
        self._compat_pressed_part = "none"
        self.setProperty("primaryTool", True)
        self.setProperty("splitTool", True)
        self.setProperty("compactTool", False)
        self.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)
        self.setCheckable(True)
        self.setText("叠加标注")
        self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.setIconSize(QSize(self.ICON_SIZE, self.ICON_SIZE))
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self.setMinimumWidth(self.COMPACT_MIN_WIDTH)
        self.setFixedHeight(self.HEIGHT)
        self.clicked.connect(self._emit_primary_triggered)
        self._update_accessible_name()

    def setText(self, text: str) -> None:
        if self.text() == text:
            return
        super().setText(text)
        self._update_accessible_name()
        self.updateGeometry()

    def currentToolKind(self) -> str:
        return self._current_kind

    def currentToolIcon(self) -> QIcon:
        return self.icon()

    def setCurrentTool(self, kind: str, icon: QIcon) -> None:
        self._current_kind = kind
        self.setIcon(icon)
        self._update_accessible_name()

    def setMenu(self, menu: QMenu | None) -> None:
        super().setMenu(menu)
        self.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)
        self._update_accessible_name()

    def _emit_primary_triggered(self, _checked: bool = False) -> None:
        self.primaryTriggered.emit()

    def _update_accessible_name(self) -> None:
        label = self.text().strip() or "测量"
        suffix = "，可展开选择其他工具" if self.menu() is not None else ""
        self.setAccessibleName(f"{label}工具{suffix}")

    def isCompactMode(self) -> bool:
        return self._compact_mode

    def setCompactMode(self, enabled: bool) -> None:
        enabled = bool(enabled)
        if self._compact_mode == enabled:
            return
        self._compact_mode = enabled
        self.setProperty("compactTool", enabled)
        self.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonIconOnly if enabled else Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )
        self.setToolTip(self.toolTip() or self.text())
        self.updateGeometry()
        _repolish(self)

    def expandedWidthHint(self) -> int:
        metrics = QFontMetrics(self.font())
        width = 10 + self.ICON_SIZE + 6 + metrics.horizontalAdvance(self.text()) + 8 + self.menuAreaWidth()
        return max(self.EXPANDED_MIN_WIDTH, width)

    def compactWidthHint(self) -> int:
        return self.COMPACT_MIN_WIDTH

    def menuAreaWidth(self) -> int:
        return self.MENU_WIDTH

    def primaryRect(self) -> QRect:
        width = max(0, self.width() - self.MENU_WIDTH)
        return QRect(0, 0, width, self.height())

    def menuRect(self) -> QRect:
        return QRect(max(0, self.width() - self.MENU_WIDTH), 0, self.MENU_WIDTH, self.height())

    def sizeHint(self) -> QSize:
        width = self.compactWidthHint() if self._compact_mode else self.expandedWidthHint()
        return QSize(width, self.HEIGHT)

    def minimumSizeHint(self) -> QSize:
        return QSize(self.compactWidthHint(), self.HEIGHT)

    def _point_from_event(self, event) -> QPoint:
        position = event.position()
        if hasattr(position, "toPoint"):
            return position.toPoint()
        return QPoint(round(position.x()), round(position.y()))

    def _hit_part(self, position: QPoint) -> str:
        if not self.rect().contains(position) or not self.isEnabled():
            return "none"
        if self.menuRect().contains(position):
            return "menu"
        if self.primaryRect().contains(position):
            return "primary"
        return "none"

    def _popup_menu(self) -> None:
        menu = self.menu()
        if menu is None or not self.isEnabled():
            return
        menu.setMinimumWidth(max(self.width() + 8, menu.sizeHint().width()))
        self.showMenu()

    def mousePressEvent(self, event) -> None:
        if isinstance(event, QMouseEvent):
            super().mousePressEvent(event)
            return
        # Some legacy unit tests dispatch a small mouse-event test double
        # directly. Real UI events always follow QToolButton's native path.
        if not self.isEnabled() or event.button() != Qt.MouseButton.LeftButton:
            return
        self._compat_pressed_part = self._hit_part(self._point_from_event(event))
        if hasattr(event, "accept"):
            event.accept()

    def mouseReleaseEvent(self, event) -> None:
        if isinstance(event, QMouseEvent):
            super().mouseReleaseEvent(event)
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return
        released_part = self._hit_part(self._point_from_event(event))
        pressed_part = self._compat_pressed_part
        self._compat_pressed_part = "none"
        if not self.isEnabled():
            return
        if pressed_part == "primary" and released_part == "primary":
            self.primaryTriggered.emit()
        elif pressed_part == "menu" and released_part == "menu":
            self._popup_menu()
        if hasattr(event, "accept"):
            event.accept()


class MeasurementToolStrip(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._mode_buttons: dict[str, ToolStripActionButton] = {}
        self._split_buttons: dict[str, OverlayToolSplitButton] = {}
        self._split_mode_lookup: dict[str, OverlayToolSplitButton] = {}
        self._primary_order: list[str] = []
        self._magic_tool_button: OverlayToolSplitButton | None = None
        self._overlay_button: OverlayToolSplitButton | None = None
        self._magic_context_widget: QWidget | None = None
        self._count_context_widget: QWidget | None = None
        self._preview_context_widget: QWidget | None = None
        self._path_context_widget: QWidget | None = None
        self._compact_mode = False
        self._primary_tools_visible = True
        self._active_mode = "select"
        self._context_placement = "hidden"
        self._theme_updating = False
        self.setObjectName("measurementToolStrip")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        self.setMinimumWidth(0)
        self._apply_theme_styles()
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(8, 8, 8, 12)
        root_layout.setSpacing(6)

        self._top_row = QWidget(self)
        self._top_row.setMinimumWidth(0)
        self._top_row_layout = QHBoxLayout(self._top_row)
        self._top_row_layout.setContentsMargins(0, 0, 0, 0)
        self._top_row_layout.setSpacing(12)

        self._primary_row = QWidget(self)
        self._primary_row.setObjectName("measurementPrimaryRow")
        self._primary_row.setMinimumWidth(0)
        self._primary_row.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        self._primary_row.setFixedHeight(ToolStripActionButton.HEIGHT + 2)
        self._primary_row_layout = QHBoxLayout(self._primary_row)
        self._primary_row_layout.setContentsMargins(0, 0, 0, 0)
        self._primary_row_layout.setSpacing(6)
        self._top_row_layout.addWidget(self._primary_row, 0, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self._top_row_layout.addStretch(1)
        root_layout.addWidget(self._top_row)

        self._context_host = QWidget(self)
        self._context_host.setMinimumWidth(0)
        self._context_host.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Maximum)
        self._context_layout = QVBoxLayout(self._context_host)
        self._context_layout.setContentsMargins(0, 0, 0, 0)
        self._context_layout.setSpacing(6)
        self._context_host.setVisible(False)
        root_layout.addWidget(self._context_host)

    def _build_stylesheet(self) -> str:
        if _application_palette_is_dark(self):
            strip_background = "#34373C"
            strip_border = "rgba(255, 255, 255, 18)"
            primary_text = "#F3F4F6"
            primary_disabled_text = "#9AA5B1"
            primary_hover = "rgba(255, 255, 255, 14)"
            primary_pressed = "rgba(255, 255, 255, 20)"
            primary_checked_background = "#12343B"
            primary_checked_text = "#F7F4EA"
            primary_checked_border = "#2A9D8F"
            context_tool_border = "rgba(255, 255, 255, 24)"
            context_tool_background = "rgba(255, 255, 255, 8)"
            context_tool_text = "#F3F4F6"
            context_tool_disabled_text = "#9AA5B1"
            context_tool_hover = "rgba(255, 255, 255, 16)"
            context_tool_pressed = "rgba(255, 255, 255, 20)"
            positive_prompt_background = "#064E3B"
            positive_prompt_border = "#34D399"
            positive_prompt_text = "#ECFDF5"
            positive_prompt_hover = "#0F6B50"
            positive_prompt_pressed = "#075A43"
            negative_prompt_background = "#7F1D1D"
            negative_prompt_border = "#F87171"
            negative_prompt_text = "#FEF2F2"
            negative_prompt_hover = "#991F1F"
            negative_prompt_pressed = "#861A1A"
            chip_background = "#F6F1E8"
            chip_text = "#182430"
            header_background = "#E8F1F2"
            header_text = "#12343B"
            status_text = "#9C6B2F"
        else:
            strip_background = "#F5F7FA"
            strip_border = "rgba(17, 24, 39, 22)"
            primary_text = "#1F2933"
            primary_disabled_text = "#51606F"
            primary_hover = "rgba(31, 41, 51, 10)"
            primary_pressed = "rgba(31, 41, 51, 16)"
            primary_checked_background = "#DDF3EF"
            primary_checked_text = "#16363D"
            primary_checked_border = "#2A9D8F"
            context_tool_border = "rgba(17, 24, 39, 16)"
            context_tool_background = "rgba(31, 41, 51, 4)"
            context_tool_text = "#1F2933"
            context_tool_disabled_text = "#51606F"
            context_tool_hover = "rgba(31, 41, 51, 9)"
            context_tool_pressed = "rgba(31, 41, 51, 14)"
            positive_prompt_background = "#D1FAE5"
            positive_prompt_border = "#059669"
            positive_prompt_text = "#064E3B"
            positive_prompt_hover = "#A7F3D0"
            positive_prompt_pressed = "#6EE7B7"
            negative_prompt_background = "#FEE2E2"
            negative_prompt_border = "#DC2626"
            negative_prompt_text = "#7F1D1D"
            negative_prompt_hover = "#FECACA"
            negative_prompt_pressed = "#FCA5A5"
            chip_background = "#F5EFD9"
            chip_text = "#4D3B1F"
            header_background = "#E7F1F4"
            header_text = "#204650"
            status_text = "#8A5A1F"
        return f"""
            QWidget#measurementToolStrip {{
                background: {strip_background};
                border-top: 1px solid {strip_border};
                border-bottom: 1px solid {strip_border};
            }}
            QToolButton[primaryTool="true"] {{
                min-height: 38px;
                padding: 0 12px;
                border-radius: 10px;
                border: 1px solid transparent;
                background: transparent;
                color: {primary_text};
                font-weight: 600;
            }}
            QToolButton[primaryTool="true"]:hover {{
                background: {primary_hover};
            }}
            QToolButton[primaryTool="true"]:pressed {{
                background: {primary_pressed};
            }}
            QToolButton[primaryTool="true"]:checked {{
                background: {primary_checked_background};
                color: {primary_checked_text};
                border: 1px solid {primary_checked_border};
            }}
            QToolButton[primaryTool="true"]:focus {{
                border: 1px solid {primary_checked_border};
            }}
            QToolButton[primaryTool="true"]:disabled {{
                color: {primary_disabled_text};
            }}
            QToolButton[primaryTool="true"][compactTool="true"] {{
                padding: 0;
            }}
            QToolButton[primaryTool="true"][splitTool="true"]::menu-button {{
                width: 28px;
                border: none;
            }}
            QLabel[contextChip="true"] {{
                padding: 6px 10px;
                border-radius: 8px;
                background: {chip_background};
                color: {chip_text};
                font-weight: 600;
            }}
            QLabel[contextHeader="true"] {{
                padding: 6px 10px;
                border-radius: 8px;
                background: {header_background};
                color: {header_text};
                font-weight: 600;
            }}
            QLabel[contextStatus="true"] {{
                color: {status_text};
                font-weight: 600;
                padding: 8px 2px 0 2px;
            }}
            QToolButton[contextTool="true"] {{
                min-height: 36px;
                padding: 0 12px;
                border-radius: 8px;
                border: 1px solid {context_tool_border};
                background: {context_tool_background};
                color: {context_tool_text};
                font-weight: 600;
            }}
            QToolButton[contextTool="true"]:hover {{
                background: {context_tool_hover};
            }}
            QToolButton[contextTool="true"]:pressed {{
                background: {context_tool_pressed};
            }}
            QToolButton[contextTool="true"]:checked {{
                background: {primary_checked_background};
                color: {primary_checked_text};
                border: 1px solid {primary_checked_border};
            }}
            QToolButton[contextTool="true"]:disabled {{
                color: {context_tool_disabled_text};
            }}
            QToolButton[contextTool="true"][magicPrompt="positive"] {{
                background: {positive_prompt_background};
                border: 1px solid {positive_prompt_border};
                color: {positive_prompt_text};
            }}
            QToolButton[contextTool="true"][magicPrompt="positive"]:hover {{
                background: {positive_prompt_hover};
            }}
            QToolButton[contextTool="true"][magicPrompt="positive"]:pressed {{
                background: {positive_prompt_pressed};
            }}
            QToolButton[contextTool="true"][magicPrompt="negative"] {{
                background: {negative_prompt_background};
                border: 1px solid {negative_prompt_border};
                color: {negative_prompt_text};
            }}
            QToolButton[contextTool="true"][magicPrompt="negative"]:hover {{
                background: {negative_prompt_hover};
            }}
            QToolButton[contextTool="true"][magicPrompt="negative"]:pressed {{
                background: {negative_prompt_pressed};
            }}
            QToolButton[contextTool="true"][magicPrompt="positive"]:disabled,
            QToolButton[contextTool="true"][magicPrompt="negative"]:disabled {{
                border: 1px solid {context_tool_border};
                background: {context_tool_background};
                color: {context_tool_disabled_text};
            }}
        """

    def _apply_button_palette(self, button: QToolButton) -> None:
        dark_theme = _application_palette_is_dark(self)
        if button.property("primaryTool"):
            normal_color = "#F3F4F6" if dark_theme else "#1F2933"
            disabled_color = "#9AA5B1" if dark_theme else "#51606F"
        elif button.property("contextTool"):
            magic_prompt = button.property("magicPrompt")
            if magic_prompt == "positive":
                normal_color = "#ECFDF5" if dark_theme else "#064E3B"
            elif magic_prompt == "negative":
                normal_color = "#FEF2F2" if dark_theme else "#7F1D1D"
            else:
                normal_color = "#F3F4F6" if dark_theme else "#1F2933"
            disabled_color = "#9AA5B1" if dark_theme else "#51606F"
        else:
            return

        palette = QPalette(button.palette())
        for role in (
            QPalette.ColorRole.ButtonText,
            QPalette.ColorRole.WindowText,
            QPalette.ColorRole.Text,
        ):
            palette.setColor(QPalette.ColorGroup.Active, role, QColor(normal_color))
            palette.setColor(QPalette.ColorGroup.Inactive, role, QColor(normal_color))
            palette.setColor(QPalette.ColorGroup.Disabled, role, QColor(disabled_color))
        button.setPalette(palette)

    def _apply_theme_styles(self) -> None:
        if self._theme_updating:
            return
        self._theme_updating = True
        try:
            self.setStyleSheet(self._build_stylesheet())
            for button in self.findChildren(QToolButton):
                self._apply_button_palette(button)
                _repolish(button)
            for button in self._mode_buttons.values():
                button.update()
            for button in set(self._split_mode_lookup.values()):
                button.update()
            if self._magic_tool_button is not None:
                self._magic_tool_button.update()
            if self._overlay_button is not None:
                self._overlay_button.update()
            self.update()
        finally:
            self._theme_updating = False

    def addModeAction(self, mode: str, action: QAction) -> ToolStripActionButton:
        button = ToolStripActionButton(action, self._primary_row)
        self._mode_buttons[mode] = button
        self._primary_order.append(mode)
        self._primary_row_layout.addWidget(button)
        return button

    def addSplitModeButton(
        self,
        mode: str,
        button: OverlayToolSplitButton,
        *,
        aliases: list[str] | tuple[str, ...] | None = None,
    ) -> OverlayToolSplitButton:
        self._split_buttons[mode] = button
        self._split_mode_lookup[mode] = button
        for alias in aliases or []:
            self._split_mode_lookup[alias] = button
        self._primary_order.append(mode)
        self._primary_row_layout.addWidget(button)
        self._sync_auto_compact_mode()
        return button

    def buttonForMode(self, mode: str):
        return self._mode_buttons.get(mode) or self._split_mode_lookup.get(mode)

    def primaryModeLabels(self) -> list[str]:
        labels: list[str] = []
        for mode in self._primary_order:
            if mode == "__magic_tool__":
                if self._magic_tool_button is not None:
                    labels.append(self._magic_tool_button.text())
                continue
            if mode in self._split_buttons:
                labels.append(self._split_buttons[mode].text())
                continue
            if mode in self._mode_buttons:
                labels.append(self._mode_buttons[mode].defaultAction().text())
        if self._overlay_button is not None:
            labels.append(self._overlay_button.text())
        return labels

    def setMagicToolButton(self, button: OverlayToolSplitButton) -> None:
        self._magic_tool_button = button
        self._primary_order.append("__magic_tool__")
        self._primary_row_layout.addWidget(button)
        self._sync_auto_compact_mode()

    def setMagicTool(self, kind: str, checked: bool, *, icon: QIcon | None = None, tooltip: str | None = None) -> None:
        if self._magic_tool_button is None:
            return
        if icon is not None:
            self._magic_tool_button.setCurrentTool(kind, icon)
        self._magic_tool_button.setChecked(checked)
        if tooltip is not None:
            self._magic_tool_button.setToolTip(tooltip)

    def setOverlayButton(self, button: OverlayToolSplitButton) -> None:
        self._overlay_button = button
        self._primary_row_layout.addWidget(button)
        self._sync_auto_compact_mode()

    def setMagicContextWidget(self, widget: QWidget) -> None:
        self._magic_context_widget = widget
        self._context_layout.addWidget(widget)
        widget.setVisible(False)
        self._refresh_context_visibility()

    def setCountContextWidget(self, widget: QWidget) -> None:
        self._count_context_widget = widget
        self._context_layout.addWidget(widget)
        widget.setVisible(False)
        self._refresh_context_visibility()

    def setPreviewContextWidget(self, widget: QWidget) -> None:
        self._preview_context_widget = widget
        self._context_layout.addWidget(widget)
        widget.setVisible(False)
        self._refresh_context_visibility()

    def setPathContextWidget(self, widget: QWidget) -> None:
        self._path_context_widget = widget
        self._context_layout.addWidget(widget)
        widget.setVisible(False)
        self._refresh_context_visibility()

    def setActiveMode(self, mode: str) -> None:
        self._active_mode = mode
        for button in set(self._split_mode_lookup.values()):
            button.setChecked(False)
        split_button = self._split_mode_lookup.get(mode)
        if split_button is not None:
            split_button.setChecked(True)
        if self._magic_tool_button is not None and mode not in {"magic_segment", "reference_propagation", "fiber_quick"}:
            self._magic_tool_button.setChecked(False)
        if self._overlay_button is not None and mode != "overlay":
            self._overlay_button.setChecked(False)

    def setOverlayTool(self, kind: str, checked: bool, *, icon: QIcon | None = None, tooltip: str | None = None) -> None:
        if self._overlay_button is None:
            return
        if icon is not None:
            self._overlay_button.setCurrentTool(kind, icon)
        self._overlay_button.setChecked(checked)
        if tooltip is not None:
            self._overlay_button.setToolTip(tooltip)

    def isCompactMode(self) -> bool:
        return self._compact_mode

    def setCompactMode(self, enabled: bool) -> None:
        enabled = bool(enabled)
        if self._compact_mode == enabled:
            return
        self._compact_mode = enabled
        for button in self._mode_buttons.values():
            button.setCompactMode(enabled)
        for button in set(self._split_mode_lookup.values()):
            button.setCompactMode(enabled)
        if self._magic_tool_button is not None:
            self._magic_tool_button.setCompactMode(enabled)
        if self._overlay_button is not None:
            self._overlay_button.setCompactMode(enabled)
        self.updateGeometry()

    def setPrimaryToolsVisible(self, visible: bool) -> None:
        visible = bool(visible)
        if self._primary_tools_visible == visible:
            return
        self._primary_tools_visible = visible
        self._primary_row.setVisible(visible)
        self._sync_auto_compact_mode()

    def primaryToolsVisible(self) -> bool:
        return self._primary_tools_visible

    def setMagicContextVisible(self, visible: bool) -> None:
        if self._magic_context_widget is not None:
            self._magic_context_widget.setVisible(bool(visible))
        self._refresh_context_visibility()

    def isMagicContextVisible(self) -> bool:
        return bool(self._magic_context_widget and not self._magic_context_widget.isHidden())

    def setCountContextVisible(self, visible: bool) -> None:
        if self._count_context_widget is not None:
            self._count_context_widget.setVisible(bool(visible))
        self._refresh_context_visibility()

    def isCountContextVisible(self) -> bool:
        return bool(self._count_context_widget and not self._count_context_widget.isHidden())

    def setPreviewContextVisible(self, visible: bool) -> None:
        if self._preview_context_widget is not None:
            self._preview_context_widget.setVisible(bool(visible))
        self._refresh_context_visibility()

    def isPreviewContextVisible(self) -> bool:
        return bool(self._preview_context_widget and not self._preview_context_widget.isHidden())

    def setPathContextVisible(self, visible: bool) -> None:
        if self._path_context_widget is not None:
            self._path_context_widget.setVisible(bool(visible))
        self._refresh_context_visibility()

    def isPathContextVisible(self) -> bool:
        return bool(self._path_context_widget and not self._path_context_widget.isHidden())

    def _refresh_context_visibility(self) -> None:
        visible = (
            self.isMagicContextVisible()
            or self.isCountContextVisible()
            or self.isPreviewContextVisible()
            or self.isPathContextVisible()
        )
        if not visible:
            self._context_placement = "hidden"
            self._apply_context_placement()
            self.updateGeometry()
            return
        self._sync_auto_compact_mode()
        self.updateGeometry()

    def _expanded_primary_width(self) -> int:
        if not self._primary_tools_visible:
            return 0
        widths: list[int] = []
        for mode in self._primary_order:
            if mode in self._mode_buttons:
                widths.append(self._mode_buttons[mode].expandedWidthHint())
            elif mode in self._split_buttons:
                widths.append(self._split_buttons[mode].expandedWidthHint())
        if self._magic_tool_button is not None:
            widths.append(self._magic_tool_button.expandedWidthHint())
        if self._overlay_button is not None:
            widths.append(self._overlay_button.expandedWidthHint())
        if not widths:
            return 0
        spacing = self._primary_row_layout.spacing()
        return sum(widths) + spacing * (len(widths) - 1)

    def _compact_primary_width(self) -> int:
        if not self._primary_tools_visible:
            return 0
        widths = [button.COMPACT_WIDTH for button in self._mode_buttons.values()]
        widths.extend(button.compactWidthHint() for button in self._split_buttons.values())
        if self._magic_tool_button is not None:
            widths.append(self._magic_tool_button.compactWidthHint())
        if self._overlay_button is not None:
            widths.append(self._overlay_button.compactWidthHint())
        if not widths:
            return 0
        spacing = self._primary_row_layout.spacing()
        return sum(widths) + spacing * (len(widths) - 1)

    def _current_context_width(self) -> int:
        return self._current_context_size().width()

    def _current_context_size(self) -> QSize:
        widget = self._current_context_widget()
        if widget is not None:
            return widget.sizeHint()
        return QSize()

    def _current_context_widget(self) -> QWidget | None:
        if self.isMagicContextVisible() and self._magic_context_widget is not None:
            return self._magic_context_widget
        if self.isCountContextVisible() and self._count_context_widget is not None:
            return self._count_context_widget
        if self.isPreviewContextVisible() and self._preview_context_widget is not None:
            return self._preview_context_widget
        if self.isPathContextVisible() and self._path_context_widget is not None:
            return self._path_context_widget
        return None

    def _context_height_for_width(self, width: int) -> int:
        widget = self._current_context_widget()
        if widget is None:
            return 0
        layout = widget.layout()
        target_width = max(0, width)
        if layout is not None:
            if layout.hasHeightForWidth():
                return max(widget.minimumSizeHint().height(), layout.heightForWidth(target_width))
            return max(widget.minimumSizeHint().height(), layout.sizeHint().height())
        if widget.hasHeightForWidth():
            return max(widget.minimumSizeHint().height(), widget.heightForWidth(target_width))
        return max(widget.minimumSizeHint().height(), widget.sizeHint().height())

    def _update_context_host_metrics(self) -> None:
        if self._context_placement == "hidden" or self._current_context_widget() is None:
            self._context_host.setMinimumWidth(0)
            self._context_host.setMaximumWidth(16777215)
            self._context_host.setMinimumHeight(0)
            self._context_host.setMaximumHeight(16777215)
            return
        if self._context_placement == "inline":
            context_width = self._current_context_width()
            self._context_host.setMinimumWidth(context_width)
            self._context_host.setMaximumWidth(context_width)
            self._context_host.setMinimumHeight(0)
            self._context_host.setMaximumHeight(16777215)
            return
        self._context_host.setMinimumWidth(0)
        self._context_host.setMaximumWidth(16777215)
        layout = self.layout()
        margins = layout.contentsMargins() if layout is not None else self.contentsMargins()
        available_width = max(0, self.width() - margins.left() - margins.right())
        context_height = self._context_height_for_width(available_width)
        self._context_host.setMinimumHeight(context_height)
        self._context_host.setMaximumHeight(context_height)

    def _preferred_strip_height(self) -> int:
        layout = self.layout()
        margins = layout.contentsMargins() if layout is not None else self.contentsMargins()
        primary_height = (ToolStripActionButton.HEIGHT + 2) if self._primary_tools_visible else 0
        height = margins.top() + primary_height + margins.bottom()
        if not self._primary_tools_visible and self._context_placement == "inline" and self._current_context_widget() is not None:
            height += self._context_height_for_width(self.width())
        if self._context_placement == "stacked" and self._current_context_widget() is not None:
            available_width = max(0, self.width() - margins.left() - margins.right())
            height += (layout.spacing() if layout is not None else 0) + self._context_height_for_width(available_width)
        return height

    def _apply_strip_height(self) -> None:
        target_height = self._preferred_strip_height()
        if self.height() != target_height or self.minimumHeight() != target_height or self.maximumHeight() != target_height:
            self.setFixedHeight(target_height)

    def _refresh_parent_layouts(self) -> None:
        layout = self.layout()
        if layout is not None:
            layout.activate()
        parent = self.parentWidget()
        if parent is not None and parent.layout() is not None:
            parent.layout().activate()
            parent.updateGeometry()
        self.updateGeometry()

    def _apply_context_placement(self) -> None:
        if self._context_placement == "hidden":
            self._top_row_layout.removeWidget(self._context_host)
            self.layout().removeWidget(self._context_host)
            self._context_host.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Maximum)
            self._context_host.setVisible(False)
            self.layout().insertWidget(1, self._context_host)
            self._update_context_host_metrics()
            self._apply_strip_height()
            self._refresh_parent_layouts()
            return
        self._top_row_layout.removeWidget(self._context_host)
        self.layout().removeWidget(self._context_host)
        if self._context_placement == "inline":
            self._context_host.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
            if self._primary_tools_visible:
                self._top_row_layout.addWidget(
                    self._context_host,
                    0,
                    Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                )
            else:
                self._top_row_layout.insertWidget(
                    0,
                    self._context_host,
                    0,
                    Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                )
        else:
            self._context_host.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            self.layout().insertWidget(1, self._context_host)
        self._context_host.setVisible(True)
        self._update_context_host_metrics()
        self._apply_strip_height()
        self._refresh_parent_layouts()

    def isContextInline(self) -> bool:
        return self._context_placement == "inline"

    def isContextStacked(self) -> bool:
        return self._context_placement == "stacked"

    def _sync_auto_compact_mode(self) -> None:
        if self.width() <= 0:
            return
        available_width = max(0, self.contentsRect().width())
        expanded_primary_width = self._expanded_primary_width()
        compact_primary_width = self._compact_primary_width()
        context_width = self._current_context_width()
        inline_gap = self._top_row_layout.spacing() if context_width > 0 else 0

        if context_width <= 0:
            self.setCompactMode(expanded_primary_width > available_width)
            self._context_placement = "hidden"
            self._apply_context_placement()
            return

        if expanded_primary_width + inline_gap + context_width <= available_width:
            self.setCompactMode(False)
            self._context_placement = "inline"
            self._apply_context_placement()
            return

        if expanded_primary_width <= available_width:
            self.setCompactMode(False)
            self._context_placement = "stacked"
            self._apply_context_placement()
            return

        if compact_primary_width + inline_gap + context_width <= available_width:
            self.setCompactMode(True)
            self._context_placement = "inline"
            self._apply_context_placement()
            return

        self.setCompactMode(expanded_primary_width > available_width)
        self._context_placement = "stacked"
        self._apply_context_placement()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._sync_auto_compact_mode()
        self._update_context_host_metrics()
        self._apply_strip_height()

    def changeEvent(self, event) -> None:
        super().changeEvent(event)
        if event.type() in {QEvent.Type.PaletteChange, QEvent.Type.ApplicationPaletteChange}:
            self._apply_theme_styles()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._sync_auto_compact_mode()
        self._update_context_host_metrics()
        self._apply_strip_height()

    def minimumSizeHint(self) -> QSize:
        return QSize(0, self._preferred_strip_height())
