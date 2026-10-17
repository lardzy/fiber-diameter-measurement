"""A transient feedback card in the visible viewport, outside image rendering."""
from __future__ import annotations

from PySide6.QtCore import QEvent, QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPalette, QPen
from PySide6.QtWidgets import QLabel, QLayout, QVBoxLayout, QWidget

from fdm.live_experiment.feedback import MeasurementFeedback, feedback_for_reason


class MeasurementFeedbackOverlay(QWidget):
    """Mouse-transparent child of a viewport, not of the pannable image canvas.

    Connect controller.measurementFeedback to show_feedback and
    controller.feedbackInvalidated to dismiss. The optional canvas is observed
    only for clicks; neither its pixels nor its paint/export path are changed.
    """

    dismissed = Signal(object)

    def __init__(self, viewport: QWidget, *, canvas: QWidget | None = None):
        super().__init__(viewport)
        self.setObjectName("measurementFeedbackOverlay")
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.viewport, self.canvas = viewport, canvas
        self.feedback: MeasurementFeedback | None = None
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self.dismiss)
        layout = QVBoxLayout(self)
        layout.setSizeConstraint(QLayout.SizeConstraint.SetNoConstraint)
        layout.setContentsMargins(54, 16, 20, 16)
        layout.setSpacing(8)
        self.title_label = QLabel(self)
        self.title_label.setTextFormat(Qt.TextFormat.PlainText)
        self.title_label.setWordWrap(True)
        title_font = self.title_label.font()
        title_font.setBold(True)
        self.title_label.setFont(title_font)
        self.message_label = QLabel(self)
        self.message_label.setTextFormat(Qt.TextFormat.PlainText)
        self.message_label.setWordWrap(True)
        self.hint_label = QLabel("点击图像可继续操作", self)
        self.hint_label.setWordWrap(True)
        for label in (self.title_label, self.message_label, self.hint_label):
            label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
            layout.addWidget(label)
        viewport.installEventFilter(self)
        if canvas is not None and canvas is not viewport:
            canvas.installEventFilter(self)
        self.hide()

    def show_feedback(self, feedback: MeasurementFeedback | str, *, duration_ms: int = 0):
        """Replace the card; by default it stays until a click or invalidation."""
        if not isinstance(feedback, MeasurementFeedback):
            feedback = feedback_for_reason(feedback)
        self._timer.stop()
        self.feedback = feedback
        self.title_label.setText({"warning": "本次测量未通过，未计入记录", "error": "测量未完成，未计入记录",
                                 "success": "已达到目标", "info": "提示"}[feedback.severity])
        self.message_label.setText(feedback.message)
        self._position()
        self.show()
        self.raise_()
        self.update()
        if duration_ms > 0:
            self._timer.start(duration_ms)

    def dismiss(self, *_):
        self._timer.stop()
        previous, self.feedback = self.feedback, None
        self.hide()
        if previous is not None:
            self.dismissed.emit(previous)

    def _position(self):
        if self.feedback is None:
            return
        available = self.viewport.rect()
        margin = min(16, max(0, min(available.width(), available.height()) // 8))
        width = max(1, min(500, available.width() - 2 * margin))
        height = self.layout().heightForWidth(width)
        if height < 0:
            height = self.layout().sizeHint().height()
        height = max(1, min(height, available.height() - 2 * margin))
        self.setGeometry((available.width() - width) // 2, min(margin, max(0, available.height() - height)),
                         width, height)

    def eventFilter(self, watched, event):
        kind = event.type()
        if kind == QEvent.Type.MouseButtonPress and watched in (self.viewport, self.canvas):
            self.dismiss()
            # This exact press must still reach measurement, pan or context
            # menu handlers; dismissing the card never consumes a user click.
        elif watched is self.viewport:
            if kind == QEvent.Type.Resize:
                self._position()
            elif kind == QEvent.Type.Hide:
                self.dismiss()
        return False

    def changeEvent(self, event):
        super().changeEvent(event)
        if hasattr(self, "feedback") and event.type() in (
                QEvent.Type.PaletteChange, QEvent.Type.FontChange, QEvent.Type.StyleChange):
            self._position()
            self.update()

    def paintEvent(self, event):
        if self.feedback is None:
            return
        palette = self.palette()
        background = QColor(palette.color(QPalette.ColorRole.Window))
        background.setAlpha(235)
        accent = QColor({"warning": "#C28A20", "error": "#C65353",
                         "success": "#278267", "info": "#377FAB"}[self.feedback.severity])
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setBrush(background)
        painter.setPen(QPen(accent, 1.5))
        painter.drawRoundedRect(QRectF(self.rect()).adjusted(1, 1, -1, -1), 10, 10)
        # Small vector glyphs stay crisp under mixed DPR and need no fonts or
        # external assets. The icon occupies its own column beside the text.
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(accent, 1.8, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap,
                            Qt.PenJoinStyle.RoundJoin))
        if self.feedback.severity == "warning":
            path = QPainterPath(QPointF(30, 16))
            path.lineTo(19, 36)
            path.lineTo(41, 36)
            path.closeSubpath()
            painter.drawPath(path)
            painter.drawLine(QPointF(30, 23), QPointF(30, 28))
            painter.drawPoint(QPointF(30, 32))
        else:
            painter.drawEllipse(QPointF(30, 27), 10, 10)
            if self.feedback.severity == "success":
                path = QPainterPath(QPointF(24, 27))
                path.lineTo(28, 31)
                path.lineTo(36, 23)
                painter.drawPath(path)
            elif self.feedback.severity == "error":
                painter.drawLine(QPointF(26, 23), QPointF(34, 31))
                painter.drawLine(QPointF(34, 23), QPointF(26, 31))
            else:
                painter.drawPoint(QPointF(30, 23))
                painter.drawLine(QPointF(30, 27), QPointF(30, 32))
        painter.end()
