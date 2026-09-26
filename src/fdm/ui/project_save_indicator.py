"""Compact non-modal save feedback next to the Save command."""

from __future__ import annotations

from PySide6.QtCore import QPointF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QPainter, QPalette, QPen
from PySide6.QtWidgets import QMenu, QToolButton

from fdm.ui.project_save_coordinator import SaveStatus


class ProjectSaveIndicator(QToolButton):
    retryRequested = Signal()
    detailsRequested = Signal()
    cancelTransitionRequested = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("projectSaveIndicator")
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setAutoRaise(True)
        self._status = SaveStatus()
        self._compact = False
        self._angle = 0
        self._spin = QTimer(self)
        self._spin.setInterval(50)
        self._spin.timeout.connect(self._advance)
        self._delay = QTimer(self)
        self._delay.setSingleShot(True)
        self._delay.setInterval(150)
        self._delay.timeout.connect(self._spin.start)
        self.clicked.connect(self._show_menu)
        self.set_compact(False)
        self.set_status(self._status)

    def set_compact(self, compact: bool) -> None:
        self._compact = compact
        self.setFixedWidth(28 if compact else 188)
        self.updateGeometry()
        self.update()

    def sizeHint(self) -> QSize:
        return QSize(28 if self._compact else 188, 32)

    def set_status(self, status: SaveStatus) -> None:
        was_saving = self._status.phase == "saving"
        self._status = status
        if status.phase == "saving":
            if not was_saving:
                self._angle = 0
                self._delay.start()
        else:
            self._delay.stop()
            self._spin.stop()
        details = [status.text]
        if status.path:
            details.append(status.path)
        if status.saved_at:
            details.append(f"上次保存：{status.saved_at}")
        if status.detail:
            details.append(status.detail)
        if status.phase == "failed":
            details.append("点击查看详情或重试")
        if status.transition:
            details.append(f"等待{status.transition}；点击可取消")
        self.setToolTip("\n".join(details))
        self.setAccessibleName(status.text)
        self.setAccessibleDescription(self.toolTip())
        self.setCursor(
            Qt.CursorShape.PointingHandCursor
            if status.transition or status.phase == "failed"
            else Qt.CursorShape.ArrowCursor
        )
        self.update()

    def _advance(self) -> None:
        self._angle = (self._angle + 18) % 360
        self.update()

    def _show_menu(self) -> None:
        if self._status.phase != "failed" and not self._status.transition:
            return
        menu = QMenu(self)
        menu.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        if self._status.transition:
            menu.addAction(
                f"取消{self._status.transition}", self.cancelTransitionRequested.emit
            )
        if self._status.phase == "failed":
            menu.addAction("重新保存", self.retryRequested.emit)
            menu.addAction("查看失败原因…", self.detailsRequested.emit)
        menu.popup(self.mapToGlobal(self.rect().bottomLeft()))

    def paintEvent(self, event) -> None:
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        phase = self._status.phase
        color = self.palette().color(QPalette.ColorRole.ButtonText)
        if phase in {"saved", "saved_newer"}:
            color = (
                QColor("#39B99B")
                if self.palette().color(QPalette.ColorRole.Window).lightness() < 128
                else QColor("#147D64")
            )
        elif phase == "failed":
            color = (
                QColor("#E87468")
                if self.palette().color(QPalette.ColorRole.Window).lightness() < 128
                else QColor("#B43B31")
            )
        elif phase in {"clean", "idle"}:
            color = self.palette().color(QPalette.ColorRole.PlaceholderText)
        painter.setPen(
            QPen(
                color,
                1.8,
                Qt.PenStyle.SolidLine,
                Qt.PenCapStyle.RoundCap,
                Qt.PenJoinStyle.RoundJoin,
            )
        )
        cy = self.height() / 2
        if phase == "saving":
            painter.drawArc(6, int(cy - 7), 14, 14, -self._angle * 16, 260 * 16)
        elif phase in {"saved", "saved_newer", "clean"}:
            painter.drawLine(QPointF(6, cy), QPointF(11, cy + 4))
            painter.drawLine(QPointF(11, cy + 4), QPointF(20, cy - 5))
        elif phase == "failed":
            painter.drawEllipse(QPointF(13, cy), 7, 7)
            painter.drawLine(QPointF(13, cy - 4), QPointF(13, cy + 1))
            painter.drawPoint(QPointF(13, cy + 4))
        else:
            painter.drawEllipse(QPointF(13, cy), 3, 3)
        if not self._compact:
            font = self.font()
            font.setBold(phase in {"saved", "saved_newer", "failed"})
            painter.setFont(font)
            painter.drawText(
                self.rect().adjusted(27, 0, -3, 0),
                Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                painter.fontMetrics().elidedText(
                    self._status.text, Qt.TextElideMode.ElideRight, self.width() - 30
                ),
            )
        painter.end()
