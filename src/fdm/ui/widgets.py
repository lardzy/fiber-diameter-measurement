from __future__ import annotations

from PySide6.QtCore import QEvent, QPoint, QRect, QRectF, QSize, Qt, QVariantAnimation, Signal
from PySide6.QtGui import QColor, QCursor, QFontMetrics, QIcon, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QComboBox, QMenu, QWidget


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


class OverlayToolSplitButton(QWidget):
    primaryTriggered = Signal()

    HEIGHT = 42
    MIN_WIDTH = 168
    MENU_WIDTH = 36
    RADIUS = 10
    ICON_SIZE = 18

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._text = "叠加标注"
        self._current_kind = ""
        self._icon = QIcon()
        self._menu: QMenu | None = None
        self._checked = False
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
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumWidth(self.MIN_WIDTH)
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
            self._menu.setParent(self)
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

    def menuAreaWidth(self) -> int:
        return self.MENU_WIDTH

    def primaryRect(self) -> QRect:
        width = max(0, self.width() - self.MENU_WIDTH)
        return QRect(0, 0, width, self.height())

    def menuRect(self) -> QRect:
        return QRect(max(0, self.width() - self.MENU_WIDTH), 0, self.MENU_WIDTH, self.height())

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
        self._menu.setMinimumWidth(max(self.width() + 10, self._menu.sizeHint().width()))
        self._menu.popup(self.mapToGlobal(QPoint(0, self.height() + 6)))

    def sizeHint(self) -> QSize:
        metrics = QFontMetrics(self.font())
        width = (
            14
            + self.ICON_SIZE
            + 8
            + metrics.horizontalAdvance(self._text)
            + 14
            + self.MENU_WIDTH
        )
        return QSize(max(self.MIN_WIDTH, width), self.HEIGHT)

    def minimumSizeHint(self) -> QSize:
        return QSize(self.MIN_WIDTH, self.HEIGHT)

    def changeEvent(self, event) -> None:
        super().changeEvent(event)
        if event.type() == QEvent.Type.EnabledChange and not self.isEnabled():
            self._pressed_part = "none"
            self._set_hover_part("none", immediate=True)

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

        outer_rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        outer_path = QPainterPath()
        outer_path.addRoundedRect(outer_rect, self.RADIUS, self.RADIUS)

        base_fill = QColor(255, 255, 255, 9)
        checked_fill = QColor("#1A222A")
        fill_color = _mix_color(base_fill, checked_fill, self._checked_strength)

        base_border = QColor(255, 255, 255, 23)
        checked_border = QColor(183, 154, 216, 128)
        border_color = _mix_color(base_border, checked_border, self._checked_strength)

        painter.fillPath(outer_path, fill_color)

        primary_hover_color = QColor(183, 154, 216, 26)
        menu_hover_color = QColor(183, 154, 216, 21)
        pressed_color = QColor(183, 154, 216, 40)

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

        divider_intensity = max(self._checked_strength, self._menu_hover_strength, self._primary_hover_strength * 0.5)
        divider_color = _mix_color(QColor(255, 255, 255, 20), QColor(183, 154, 216, 74), divider_intensity)
        divider_x = self.menuRect().left()
        painter.setPen(QPen(divider_color, 1.0))
        painter.drawLine(QPoint(divider_x, 8), QPoint(divider_x, self.height() - 8))

        if self.hasFocus():
            focus_color = QColor(183, 154, 216, 110)
            painter.setPen(QPen(focus_color, 1.0))
            painter.drawRoundedRect(outer_rect.adjusted(1.0, 1.0, -1.0, -1.0), self.RADIUS - 1, self.RADIUS - 1)

        icon_rect = QRect(14, (self.height() - self.ICON_SIZE) // 2, self.ICON_SIZE, self.ICON_SIZE)
        if not self._icon.isNull():
            self._icon.paint(painter, icon_rect)

        primary_rect = self.primaryRect()
        text_left = icon_rect.right() + 8
        text_right_padding = 14
        text_rect = QRect(text_left, 0, max(0, primary_rect.right() - text_left - text_right_padding), self.height())
        text_color = _mix_color(QColor("#F3F4F6"), QColor("#FBFAFD"), self._checked_strength * 0.8)
        painter.setPen(text_color)
        text = QFontMetrics(self.font()).elidedText(self._text, Qt.TextElideMode.ElideRight, text_rect.width())
        painter.drawText(text_rect, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, text)

        menu_rect = self.menuRect()
        chevron_color = _mix_color(QColor("#D7D9DE"), QColor("#C9B3E5"), max(self._menu_hover_strength, self._checked_strength))
        painter.setPen(QPen(chevron_color, 1.8, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        chevron_center_x = menu_rect.center().x()
        chevron_center_y = menu_rect.center().y() + 1
        painter.drawLine(
            QPoint(chevron_center_x - 5, chevron_center_y - 2),
            QPoint(chevron_center_x, chevron_center_y + 3),
        )
        painter.drawLine(
            QPoint(chevron_center_x, chevron_center_y + 3),
            QPoint(chevron_center_x + 5, chevron_center_y - 2),
        )
