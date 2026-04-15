from __future__ import annotations

from PySide6.QtCore import QEvent, QPoint, QRect, QRectF, QSize, Qt, QVariantAnimation, Signal
from PySide6.QtGui import QAction, QColor, QCursor, QFont, QFontMetrics, QIcon, QPainter, QPainterPath, QPalette, QPen
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLayout,
    QLayoutItem,
    QMenu,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QWidgetItem,
)


def _mix_channel(left: int, right: int, progress: float) -> int:
    return round(left + (right - left) * max(0.0, min(1.0, progress)))


def _mix_color(left: QColor, right: QColor, progress: float) -> QColor:
    return QColor(
        _mix_channel(left.red(), right.red(), progress),
        _mix_channel(left.green(), right.green(), progress),
        _mix_channel(left.blue(), right.blue(), progress),
        _mix_channel(left.alpha(), right.alpha(), progress),
    )


def _alpha_scaled(color: QColor, factor: float) -> QColor:
    scaled = QColor(color)
    scaled.setAlpha(round(color.alpha() * max(0.0, min(1.0, factor))))
    return scaled


def _repolish(widget: QWidget) -> None:
    style = widget.style()
    style.unpolish(widget)
    style.polish(widget)
    widget.update()


def _is_dark_palette(widget: QWidget) -> bool:
    return widget.palette().color(QPalette.ColorRole.Window).lightnessF() < 0.5


class MeasurementGroupComboBox(QComboBox):
    """Category combo used inside the measurement table.

    It should only change by explicit click/open interactions, not by wheel
    scrolling while the user is browsing the canvas.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)

    def wheelEvent(self, event) -> None:
        view = self.view()
        if view is not None and view.isVisible():
            super().wheelEvent(event)
            return
        event.ignore()


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

    def __init__(self, label: str, count: int, color: str, *, selected: bool = False, parent=None) -> None:
        super().__init__(parent)
        self._label = label
        self._count = max(0, int(count))
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

    def labelText(self) -> str:
        return self._label

    def countValue(self) -> int:
        return self._count

    def countText(self) -> str:
        return str(self._count)

    def sizeHint(self) -> QSize:
        return QSize(210, self.HEIGHT)

    def minimumSizeHint(self) -> QSize:
        return QSize(120, self.HEIGHT)

    def paintEvent(self, event) -> None:
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect().adjusted(0, 0, -1, -1)
        palette = self.palette()
        dark_palette = _is_dark_palette(self)

        if self._selected:
            background = QColor("#12343B")
            border = QColor("#00A6A6")
            text_color = QColor("#F4FBFF")
            badge_background = QColor(255, 255, 255, 34)
            badge_border = QColor(255, 255, 255, 48)
            badge_text = QColor("#F4FBFF")
        else:
            if dark_palette:
                background = QColor(255, 255, 255, 20)
                border = QColor(255, 255, 255, 34)
                badge_background = QColor(255, 255, 255, 18)
                badge_border = QColor(255, 255, 255, 28)
            else:
                background = QColor(15, 23, 42, 10)
                border = QColor(15, 23, 42, 34)
                badge_background = QColor(15, 23, 42, 12)
                badge_border = QColor(15, 23, 42, 20)
            text_color = palette.color(QPalette.ColorRole.ButtonText)
            badge_text = palette.color(QPalette.ColorRole.Text)

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
        badge_metrics = QFontMetrics(badge_font)
        badge_width = max(28, badge_metrics.horizontalAdvance(self.countText()) + 16)
        badge_rect = QRect(rect.right() - 12 - badge_width, rect.y() + 8, badge_width, rect.height() - 16)
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
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
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


class OverlayToolSplitButton(QWidget):
    primaryTriggered = Signal()

    HEIGHT = ToolStripActionButton.HEIGHT
    EXPANDED_MIN_WIDTH = 108
    COMPACT_MIN_WIDTH = 56
    MENU_WIDTH = 28
    RADIUS = 10
    ICON_SIZE = ToolStripActionButton.ICON_SIZE

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._text = "叠加标注"
        self._current_kind = ""
        self._icon = QIcon()
        self._menu: QMenu | None = None
        self._checked = False
        self._compact_mode = False
        self._hover_part = "none"
        self._pressed_part = "none"
        self._menu_visible = False
        self._primary_hover_strength = 0.0
        self._menu_hover_strength = 0.0
        self._checked_strength = 0.0
        self._primary_hover_animation = self._build_scalar_animation("_primary_hover_strength")
        self._menu_hover_animation = self._build_scalar_animation("_menu_hover_strength")
        self._checked_animation = self._build_scalar_animation("_checked_strength")
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self.setMinimumWidth(self.COMPACT_MIN_WIDTH)
        self.setFixedHeight(self.HEIGHT)
        self.setAccessibleName("叠加标注工具按钮")

    def _build_scalar_animation(self, attribute_name: str) -> QVariantAnimation:
        animation = QVariantAnimation(self)
        animation.setDuration(120)
        animation.valueChanged.connect(lambda value, name=attribute_name: self._on_scalar_animated(name, value))
        return animation

    def _on_scalar_animated(self, attribute_name: str, value) -> None:
        setattr(self, attribute_name, float(value))
        self.update()

    def _animate_scalar(self, animation: QVariantAnimation, attribute_name: str, target: float, *, immediate: bool = False) -> None:
        if immediate:
            animation.stop()
            setattr(self, attribute_name, target)
            self.update()
            return
        current = float(getattr(self, attribute_name))
        if abs(current - target) < 0.001:
            return
        animation.stop()
        animation.setStartValue(current)
        animation.setEndValue(target)
        animation.start()

    def text(self) -> str:
        return self._text

    def setText(self, text: str) -> None:
        if self._text == text:
            return
        self._text = text
        self.updateGeometry()
        self.update()

    def currentToolKind(self) -> str:
        return self._current_kind

    def currentToolIcon(self) -> QIcon:
        return self._icon

    def setCurrentTool(self, kind: str, icon: QIcon) -> None:
        self._current_kind = kind
        self._icon = icon
        self.update()

    def menu(self) -> QMenu | None:
        return self._menu

    def setMenu(self, menu: QMenu | None) -> None:
        if self._menu is menu:
            return
        if self._menu is not None:
            try:
                self._menu.aboutToShow.disconnect(self._on_menu_about_to_show)
            except (RuntimeError, TypeError):
                pass
            try:
                self._menu.aboutToHide.disconnect(self._on_menu_about_to_hide)
            except (RuntimeError, TypeError):
                pass
        self._menu = menu
        if self._menu is not None:
            self._menu.aboutToShow.connect(self._on_menu_about_to_show)
            self._menu.aboutToHide.connect(self._on_menu_about_to_hide)

    def isChecked(self) -> bool:
        return self._checked

    def setChecked(self, checked: bool) -> None:
        checked = bool(checked)
        if self._checked == checked:
            return
        self._checked = checked
        self._animate_scalar(self._checked_animation, "_checked_strength", 1.0 if checked else 0.0)

    def isCompactMode(self) -> bool:
        return self._compact_mode

    def setCompactMode(self, enabled: bool) -> None:
        enabled = bool(enabled)
        if self._compact_mode == enabled:
            return
        self._compact_mode = enabled
        self.updateGeometry()
        self.update()

    def expandedWidthHint(self) -> int:
        metrics = QFontMetrics(self.font())
        width = 10 + self.ICON_SIZE + 6 + metrics.horizontalAdvance(self._text) + 8 + self.MENU_WIDTH
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

    def _set_hover_part(self, part: str, *, immediate: bool = False) -> None:
        if part == self._hover_part and not immediate:
            return
        self._hover_part = part
        self._animate_scalar(
            self._primary_hover_animation,
            "_primary_hover_strength",
            1.0 if part == "primary" else 0.0,
            immediate=immediate,
        )
        self._animate_scalar(
            self._menu_hover_animation,
            "_menu_hover_strength",
            1.0 if part == "menu" or self._menu_visible else 0.0,
            immediate=immediate,
        )

    def _on_menu_about_to_show(self) -> None:
        self._menu_visible = True
        self._animate_scalar(self._menu_hover_animation, "_menu_hover_strength", 1.0)
        self.update()

    def _on_menu_about_to_hide(self) -> None:
        self._menu_visible = False
        hover_part = self._hit_part(self.mapFromGlobal(QCursor.pos())) if self.underMouse() else "none"
        self._set_hover_part(hover_part)
        self.update()

    def _popup_menu(self) -> None:
        if self._menu is None or not self.isEnabled():
            return
        self._menu.setMinimumWidth(max(self.width() + 8, self._menu.sizeHint().width()))
        self._menu.popup(self.mapToGlobal(QPoint(0, self.height() + 6)))

    def _theme_colors(self) -> dict[str, QColor]:
        if _is_dark_palette(self):
            return {
                "checked_fill": QColor("#12343B"),
                "checked_border": QColor("#2A9D8F"),
                "primary_hover": QColor(255, 255, 255, 14),
                "menu_hover": QColor(255, 255, 255, 14),
                "pressed": QColor(255, 255, 255, 20),
                "divider": QColor(255, 255, 255, 28),
                "text": QColor("#F3F4F6"),
                "checked_text": QColor("#FBFAFD"),
                "chevron": QColor("#D7D9DE"),
                "checked_chevron": QColor("#F3F4F6"),
                "hover_chevron": QColor("#C9B3E5"),
            }
        return {
            "checked_fill": QColor("#DDF3EF"),
            "checked_border": QColor("#2A9D8F"),
            "primary_hover": QColor(22, 54, 61, 18),
            "menu_hover": QColor(22, 54, 61, 18),
            "pressed": QColor(22, 54, 61, 28),
            "divider": QColor(22, 54, 61, 36),
            "text": QColor("#1F2933"),
            "checked_text": QColor("#16363D"),
            "chevron": QColor("#51606F"),
            "checked_chevron": QColor("#16363D"),
            "hover_chevron": QColor("#8B6FB4"),
        }

    def changeEvent(self, event) -> None:
        super().changeEvent(event)
        if event.type() == QEvent.Type.EnabledChange and not self.isEnabled():
            self._pressed_part = "none"
            self._set_hover_part("none", immediate=True)
        elif event.type() in {QEvent.Type.PaletteChange, QEvent.Type.ApplicationPaletteChange}:
            self.update()

    def enterEvent(self, event) -> None:
        super().enterEvent(event)
        self._set_hover_part(self._hit_part(self.mapFromGlobal(QCursor.pos())))

    def leaveEvent(self, event) -> None:
        super().leaveEvent(event)
        if self._menu_visible:
            self._set_hover_part("menu")
            return
        self._set_hover_part("none")

    def mouseMoveEvent(self, event) -> None:
        super().mouseMoveEvent(event)
        self._set_hover_part(self._hit_part(self._point_from_event(event)))

    def mousePressEvent(self, event) -> None:
        if not self.isEnabled() or event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        self._pressed_part = self._hit_part(self._point_from_event(event))
        self._set_hover_part(self._pressed_part, immediate=True)
        self.update()
        if hasattr(event, "accept"):
            event.accept()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            super().mouseReleaseEvent(event)
            return
        released_part = self._hit_part(self._point_from_event(event))
        pressed_part = self._pressed_part
        self._pressed_part = "none"
        self._set_hover_part(released_part if self.underMouse() else "none", immediate=True)
        if not self.isEnabled():
            return
        if pressed_part == "primary" and released_part == "primary":
            self.primaryTriggered.emit()
        elif pressed_part == "menu" and released_part == "menu":
            self._popup_menu()
        self.update()
        if hasattr(event, "accept"):
            event.accept()

    def keyPressEvent(self, event) -> None:
        if not self.isEnabled():
            super().keyPressEvent(event)
            return
        if event.key() in {Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space}:
            self.primaryTriggered.emit()
            event.accept()
            return
        if event.key() == Qt.Key.Key_Down and event.modifiers() in {Qt.KeyboardModifier.NoModifier, Qt.KeyboardModifier.AltModifier}:
            self._popup_menu()
            event.accept()
            return
        if event.key() == Qt.Key.Key_Escape and self._menu is not None and self._menu.isVisible():
            self._menu.close()
            event.accept()
            return
        super().keyPressEvent(event)

    def paintEvent(self, event) -> None:
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        if not self.isEnabled():
            painter.setOpacity(0.45)
        colors = self._theme_colors()

        outer_rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        outer_path = QPainterPath()
        outer_path.addRoundedRect(outer_rect, self.RADIUS, self.RADIUS)

        fill_color = _mix_color(QColor(255, 255, 255, 0), colors["checked_fill"], self._checked_strength)
        border_color = _mix_color(QColor(255, 255, 255, 0), colors["checked_border"], self._checked_strength)

        painter.fillPath(outer_path, fill_color)

        primary_hover_color = colors["primary_hover"]
        menu_hover_color = colors["menu_hover"]
        pressed_color = colors["pressed"]

        painter.save()
        painter.setClipPath(outer_path)
        if self._primary_hover_strength > 0.0:
            painter.fillRect(QRectF(self.primaryRect()), _alpha_scaled(primary_hover_color, self._primary_hover_strength))
        if self._menu_hover_strength > 0.0:
            painter.fillRect(QRectF(self.menuRect()), _alpha_scaled(menu_hover_color, self._menu_hover_strength))
        if self._pressed_part == "primary":
            painter.fillRect(QRectF(self.primaryRect()), pressed_color)
        elif self._pressed_part == "menu":
            painter.fillRect(QRectF(self.menuRect()), pressed_color)
        painter.restore()

        painter.setPen(QPen(border_color, 1.0))
        painter.drawPath(outer_path)

        divider_intensity = max(self._checked_strength * 0.9, self._menu_hover_strength)
        divider_color = _mix_color(QColor(255, 255, 255, 0), colors["divider"], divider_intensity)
        divider_x = self.menuRect().left()
        painter.setPen(QPen(divider_color, 1.0))
        painter.drawLine(QPoint(divider_x, 8), QPoint(divider_x, self.height() - 8))

        icon_size = self.ICON_SIZE
        primary_rect = self.primaryRect()
        icon_left = 10
        icon_rect = QRect(icon_left, (self.height() - icon_size) // 2, icon_size, icon_size)
        if self._compact_mode:
            icon_rect.moveLeft(max(0, (primary_rect.width() - icon_size) // 2))
        if not self._icon.isNull():
            self._icon.paint(painter, icon_rect)

        if not self._compact_mode:
            text_left = icon_rect.right() + 6
            text_right_padding = 8
            text_rect = QRect(text_left, 0, max(0, primary_rect.right() - text_left - text_right_padding), self.height())
            text_color = _mix_color(colors["text"], colors["checked_text"], self._checked_strength * 0.55)
            font = QFont(self.font())
            font.setWeight(QFont.Weight.DemiBold)
            painter.setFont(font)
            painter.setPen(text_color)
            text = QFontMetrics(font).elidedText(self._text, Qt.TextElideMode.ElideRight, text_rect.width())
            painter.drawText(text_rect, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, text)

        menu_rect = self.menuRect()
        chevron_color = _mix_color(colors["chevron"], colors["checked_chevron"], self._checked_strength * 0.6)
        chevron_color = _mix_color(chevron_color, colors["hover_chevron"], self._menu_hover_strength * 0.85)
        painter.setPen(QPen(chevron_color, 1.8, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        chevron_center_x = menu_rect.center().x()
        chevron_center_y = menu_rect.center().y() + 1
        painter.drawLine(
            QPoint(chevron_center_x - 4, chevron_center_y - 2),
            QPoint(chevron_center_x, chevron_center_y + 2),
        )
        painter.drawLine(
            QPoint(chevron_center_x, chevron_center_y + 2),
            QPoint(chevron_center_x + 4, chevron_center_y - 2),
        )


class MeasurementToolStrip(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._mode_buttons: dict[str, ToolStripActionButton] = {}
        self._primary_order: list[str] = []
        self._overlay_button: OverlayToolSplitButton | None = None
        self._magic_context_widget: QWidget | None = None
        self._preview_context_widget: QWidget | None = None
        self._compact_mode = False
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
        if _is_dark_palette(self):
            strip_background = "#34373C"
            strip_border = "rgba(255, 255, 255, 18)"
            primary_text = "#F3F4F6"
            primary_hover = "rgba(255, 255, 255, 14)"
            primary_pressed = "rgba(255, 255, 255, 20)"
            primary_checked_background = "#12343B"
            primary_checked_text = "#F7F4EA"
            primary_checked_border = "#2A9D8F"
            context_tool_border = "rgba(255, 255, 255, 24)"
            context_tool_background = "rgba(255, 255, 255, 8)"
            context_tool_text = "#F3F4F6"
            context_tool_hover = "rgba(255, 255, 255, 16)"
            context_tool_pressed = "rgba(255, 255, 255, 20)"
            chip_background = "#F6F1E8"
            chip_text = "#182430"
            header_background = "#E8F1F2"
            header_text = "#12343B"
            status_text = "#9C6B2F"
        else:
            strip_background = "#F5F7FA"
            strip_border = "rgba(17, 24, 39, 22)"
            primary_text = "#1F2933"
            primary_hover = "rgba(31, 41, 51, 10)"
            primary_pressed = "rgba(31, 41, 51, 16)"
            primary_checked_background = "#DDF3EF"
            primary_checked_text = "#16363D"
            primary_checked_border = "#2A9D8F"
            context_tool_border = "rgba(17, 24, 39, 16)"
            context_tool_background = "rgba(31, 41, 51, 4)"
            context_tool_text = "#1F2933"
            context_tool_hover = "rgba(31, 41, 51, 9)"
            context_tool_pressed = "rgba(31, 41, 51, 14)"
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
            QToolButton[primaryTool="true"][compactTool="true"] {{
                padding: 0;
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
        """

    def _apply_theme_styles(self) -> None:
        if self._theme_updating:
            return
        self._theme_updating = True
        try:
            self.setStyleSheet(self._build_stylesheet())
            if self._overlay_button is not None:
                self._overlay_button.update()
        finally:
            self._theme_updating = False

    def addModeAction(self, mode: str, action: QAction) -> ToolStripActionButton:
        button = ToolStripActionButton(action, self._primary_row)
        self._mode_buttons[mode] = button
        self._primary_order.append(mode)
        self._primary_row_layout.addWidget(button)
        return button

    def buttonForMode(self, mode: str) -> ToolStripActionButton | None:
        return self._mode_buttons.get(mode)

    def primaryModeLabels(self) -> list[str]:
        labels = [self._mode_buttons[mode].defaultAction().text() for mode in self._primary_order if mode in self._mode_buttons]
        if self._overlay_button is not None:
            labels.append(self._overlay_button.text())
        return labels

    def setOverlayButton(self, button: OverlayToolSplitButton) -> None:
        self._overlay_button = button
        self._primary_row_layout.addWidget(button)
        self._sync_auto_compact_mode()

    def setMagicContextWidget(self, widget: QWidget) -> None:
        self._magic_context_widget = widget
        self._context_layout.addWidget(widget)
        widget.setVisible(False)
        self._refresh_context_visibility()

    def setPreviewContextWidget(self, widget: QWidget) -> None:
        self._preview_context_widget = widget
        self._context_layout.addWidget(widget)
        widget.setVisible(False)
        self._refresh_context_visibility()

    def setActiveMode(self, mode: str) -> None:
        self._active_mode = mode
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
        if self._overlay_button is not None:
            self._overlay_button.setCompactMode(enabled)
        self.updateGeometry()

    def setMagicContextVisible(self, visible: bool) -> None:
        if self._magic_context_widget is not None:
            self._magic_context_widget.setVisible(bool(visible))
        self._refresh_context_visibility()

    def isMagicContextVisible(self) -> bool:
        return bool(self._magic_context_widget and not self._magic_context_widget.isHidden())

    def setPreviewContextVisible(self, visible: bool) -> None:
        if self._preview_context_widget is not None:
            self._preview_context_widget.setVisible(bool(visible))
        self._refresh_context_visibility()

    def isPreviewContextVisible(self) -> bool:
        return bool(self._preview_context_widget and not self._preview_context_widget.isHidden())

    def _refresh_context_visibility(self) -> None:
        visible = self.isMagicContextVisible() or self.isPreviewContextVisible()
        if not visible:
            self._context_placement = "hidden"
            self._apply_context_placement()
            self.updateGeometry()
            return
        self._sync_auto_compact_mode()
        self.updateGeometry()

    def _expanded_primary_width(self) -> int:
        widths = [self._mode_buttons[mode].expandedWidthHint() for mode in self._primary_order if mode in self._mode_buttons]
        if self._overlay_button is not None:
            widths.append(self._overlay_button.expandedWidthHint())
        if not widths:
            return 0
        spacing = self._primary_row_layout.spacing()
        return sum(widths) + spacing * (len(widths) - 1)

    def _compact_primary_width(self) -> int:
        widths = [button.COMPACT_WIDTH for button in self._mode_buttons.values()]
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
        if self.isPreviewContextVisible() and self._preview_context_widget is not None:
            return self._preview_context_widget
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
        height = margins.top() + (ToolStripActionButton.HEIGHT + 2) + margins.bottom()
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
            self._top_row_layout.addWidget(
                self._context_host,
                0,
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
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
