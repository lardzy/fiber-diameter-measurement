"""Transient display controls bound to one digital-slide canvas."""

from __future__ import annotations

from typing import TYPE_CHECKING
from time import monotonic_ns
import weakref

from PySide6.QtCore import QEvent, QSignalBlocker, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QHideEvent, QWheelEvent
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QSlider, QSpinBox, QStyle, QStyleOptionToolButton, QToolButton, QVBoxLayout, QWidget
from shiboken6 import isValid as is_qobject_valid

if TYPE_CHECKING:
    from fdm.ui.digital_slide_canvas import DigitalSlideCanvas


class SlideBrightnessButton(QToolButton):
    """Reserve the full numeric range so brightness cannot resize the viewport."""

    def __init__(self, parent: QWidget | None = None) -> None:
        # Qt can dispatch style/font events during the C++ constructor.
        self._size_hint_cache: QSize | None = None
        super().__init__(parent)
        self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)

    def sizeHint(self) -> QSize:
        self.ensurePolished()
        if self._size_hint_cache is not None:
            return QSize(self._size_hint_cache)
        metrics = self.fontMetrics()
        # Use actual font metrics, including proportional digits. This is a
        # text-only tool button: Qt reserves a space on either side of its text
        # before applying the active style's tool-button padding and borders.
        text = max(
            (f"亮度：{value}" for value in range(-100, 101)),
            key=lambda label: metrics.size(Qt.TextFlag.TextShowMnemonic, label).width(),
        )
        contents = metrics.size(Qt.TextFlag.TextShowMnemonic, text)
        contents.setWidth(contents.width() + 2 * metrics.horizontalAdvance(" "))
        option = QStyleOptionToolButton()
        self.initStyleOption(option)
        option.text = text
        option.rect.setSize(contents)
        self._size_hint_cache = self.style().sizeFromContents(
            QStyle.ContentsType.CT_ToolButton, option, contents, self,
        )
        return QSize(self._size_hint_cache)

    def minimumSizeHint(self) -> QSize:
        return self.sizeHint()

    def event(self, event: QEvent) -> bool:
        sizing_changed = event.type() in (
            QEvent.Type.FontChange, QEvent.Type.ApplicationFontChange,
            QEvent.Type.StyleChange, QEvent.Type.DevicePixelRatioChange,
        )
        if sizing_changed:
            self._size_hint_cache = None
        result = super().event(event)
        if sizing_changed:
            self.updateGeometry()
        return result


class _BrightnessSlider(QSlider):
    def wheelEvent(self, event: QWheelEvent) -> None:
        # Consume wheel input so it cannot reach the slide's focus-layer handler.
        event.accept()


class _BrightnessSpinBox(QSpinBox):
    def wheelEvent(self, event: QWheelEvent) -> None:
        event.accept()


class SlideBrightnessPopup(QFrame):
    """A popup owns a fixed canvas binding; queued edits never follow tab changes."""

    closed = Signal()
    _UPDATE_INTERVAL_MS = 33

    def __init__(self, canvas: DigitalSlideCanvas, parent: QWidget | None = None) -> None:
        super().__init__(parent, Qt.WindowType.Popup)
        self.setObjectName("slideBrightnessPopup")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self._canvas_ref = weakref.ref(canvas)
        self._document_id = canvas.document_id
        self._pending_value: int | None = None
        self._last_apply_ns: int | None = None
        self._closed = False
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._timer.setInterval(self._UPDATE_INTERVAL_MS)
        self._timer.timeout.connect(self.flush)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.addWidget(QLabel("切片查看亮度", self))
        row = QHBoxLayout()
        self.slider = _BrightnessSlider(Qt.Orientation.Horizontal, self)
        self.slider.setObjectName("slideBrightnessSlider")
        self.slider.setAccessibleName("切片查看亮度")
        self.slider.setRange(-100, 100)
        self.slider.setMinimumWidth(180)
        self.spin_box = _BrightnessSpinBox(self)
        self.spin_box.setObjectName("slideBrightnessValue")
        self.spin_box.setAccessibleName("切片查看亮度数值")
        self.spin_box.setRange(-100, 100)
        row.addWidget(self.slider, 1)
        row.addWidget(self.spin_box)
        layout.addLayout(row)
        hint = QLabel("仅调整当前切片的查看亮度，导出保持原亮度。", self)
        hint.setWordWrap(True)
        hint.setMaximumWidth(300)
        layout.addWidget(hint)
        self.reset_button = QPushButton("恢复原始亮度", self)
        layout.addWidget(self.reset_button)

        self.sync_value(canvas.display_brightness)
        self.slider.valueChanged.connect(self._queue_value)
        self.spin_box.valueChanged.connect(self._queue_value)
        self.slider.sliderReleased.connect(self.flush)
        self.spin_box.editingFinished.connect(self.flush)
        self.reset_button.clicked.connect(self._reset)

    @property
    def canvas(self) -> DigitalSlideCanvas | None:
        canvas = self._canvas_ref()
        return (
            canvas
            if canvas is not None and is_qobject_valid(canvas)
            and canvas.document_id == self._document_id
            else None
        )

    @property
    def has_pending_edit(self) -> bool:
        return self._pending_value is not None

    def sync_value(self, value: int) -> None:
        """Reflect an externally changed canvas value without queuing another edit."""
        self._timer.stop()
        self._pending_value = None
        with QSignalBlocker(self.slider), QSignalBlocker(self.spin_box):
            self.slider.setValue(value)
            self.spin_box.setValue(value)

    def _queue_value(self, value: int) -> None:
        with QSignalBlocker(self.slider), QSignalBlocker(self.spin_box):
            self.slider.setValue(value)
            self.spin_box.setValue(value)
        self._pending_value = value
        if not self._timer.isActive():
            # Keep a cadence measured from actual application, including time
            # spent mapping/painting, instead of adding 33 ms to each new batch.
            # An idle/first edit still goes through the event loop to coalesce
            # all inputs in the same turn; it never applies synchronously here.
            remaining_ns = (
                0 if self._last_apply_ns is None
                else self._UPDATE_INTERVAL_MS * 1_000_000
                - (monotonic_ns() - self._last_apply_ns)
            )
            delay_ms = max(0, (remaining_ns + 999_999) // 1_000_000)
            self._timer.start(delay_ms)

    def flush(self) -> None:
        self._timer.stop()
        value, self._pending_value = self._pending_value, None
        canvas = self.canvas
        if value is not None and canvas is not None:
            self._last_apply_ns = monotonic_ns()
            canvas.set_display_brightness(value)

    def _reset(self) -> None:
        self._queue_value(0)
        self.flush()

    def hideEvent(self, event: QHideEvent) -> None:
        self.flush()
        super().hideEvent(event)
        if not self._closed:
            self._closed = True
            self.closed.emit()

    def wheelEvent(self, event: QWheelEvent) -> None:
        event.accept()
