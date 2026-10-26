"""Display adjustments and explicit persistence requests for one slide canvas."""

from __future__ import annotations

from typing import TYPE_CHECKING
from time import monotonic_ns
import weakref

from PySide6.QtCore import QEvent, QSignalBlocker, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QHideEvent, QPalette, QWheelEvent
from PySide6.QtWidgets import (
    QCheckBox, QDoubleSpinBox, QFrame, QHBoxLayout, QLabel, QPushButton,
    QSlider, QSpinBox, QToolButton, QVBoxLayout, QWidget,
)
from shiboken6 import isValid as is_qobject_valid

from fdm.ui.icons import themed_icon

if TYPE_CHECKING:
    from fdm.ui.digital_slide_canvas import DigitalSlideCanvas


class SlideBrightnessButton(QToolButton):
    """Fixed text keeps adjustment values from changing viewport geometry."""

    def __init__(self, parent: QWidget | None = None) -> None:
        self._icon_ready = False
        super().__init__(parent)
        self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.setText("亮度/曝光")
        self.setAccessibleName("切片亮度与曝光")
        self.setIconSize(QSize(18, 18))
        self._icon_ready = True
        self._refresh_icon()

    def _refresh_icon(self) -> None:
        self.setIcon(themed_icon(
            "display_adjustments",
            color=self.palette().color(QPalette.ColorRole.ButtonText).name(),
        ))

    def event(self, event: QEvent) -> bool:
        result = super().event(event)
        if self._icon_ready and event.type() in (
            QEvent.Type.PaletteChange, QEvent.Type.ApplicationPaletteChange,
            QEvent.Type.StyleChange,
        ):
            self._refresh_icon()
        return result


class _BrightnessSlider(QSlider):
    def wheelEvent(self, event: QWheelEvent) -> None:
        # Consume wheel input so it cannot reach the slide's focus-layer handler.
        event.accept()


class _BrightnessSpinBox(QSpinBox):
    def wheelEvent(self, event: QWheelEvent) -> None:
        event.accept()


class _ExposureSpinBox(QDoubleSpinBox):
    def wheelEvent(self, event: QWheelEvent) -> None:
        event.accept()


class SlideBrightnessPopup(QFrame):
    """A fixed canvas binding separates frequent previews from user commits."""

    closed = Signal()
    adjustmentsCommitted = Signal(int, int)
    rememberToggled = Signal(bool)
    _UPDATE_INTERVAL_MS = 33

    def __init__(
        self, canvas: DigitalSlideCanvas, parent: QWidget | None = None, *,
        remember: bool = False,
    ) -> None:
        super().__init__(parent, Qt.WindowType.Popup)
        self.setObjectName("slideBrightnessPopup")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self._canvas_ref = weakref.ref(canvas)
        self._document_id = canvas.document_id
        self._pending_adjustments: tuple[int, int] | None = None
        self._requested_adjustments = canvas.display_adjustments
        self._edited_since_commit = False
        self._last_apply_ns: int | None = None
        self._closed = False
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._timer.setInterval(self._UPDATE_INTERVAL_MS)
        self._timer.timeout.connect(self.flush)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.addWidget(QLabel("切片查看亮度与曝光", self))
        layout.addWidget(QLabel("亮度", self))
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

        layout.addWidget(QLabel("曝光补偿", self))
        row = QHBoxLayout()
        self.exposure_slider = _BrightnessSlider(Qt.Orientation.Horizontal, self)
        self.exposure_slider.setObjectName("slideExposureSlider")
        self.exposure_slider.setAccessibleName("切片查看曝光补偿")
        self.exposure_slider.setRange(-40, 40)
        self.exposure_slider.setMinimumWidth(180)
        self.exposure_spin_box = _ExposureSpinBox(self)
        self.exposure_spin_box.setObjectName("slideExposureValue")
        self.exposure_spin_box.setAccessibleName("切片查看曝光补偿数值")
        self.exposure_spin_box.setDecimals(1)
        self.exposure_spin_box.setRange(-4.0, 4.0)
        self.exposure_spin_box.setSingleStep(0.1)
        self.exposure_spin_box.setSuffix(" EV")
        row.addWidget(self.exposure_slider, 1)
        row.addWidget(self.exposure_spin_box)
        layout.addLayout(row)

        hint = QLabel("仅影响当前切片的查看，原图、测量和导出保持不变。", self)
        hint.setWordWrap(True)
        hint.setMaximumWidth(320)
        layout.addWidget(hint)
        self.remember_checkbox = QCheckBox("记住当前设置", self)
        self.remember_checkbox.setObjectName("slideRememberAdjustments")
        self.remember_checkbox.setAccessibleName("记住当前切片查看设置")
        layout.addWidget(self.remember_checkbox)
        remember_hint = QLabel("用于之后打开的切片，重启软件后仍有效。", self)
        remember_hint.setWordWrap(True)
        remember_hint.setMaximumWidth(320)
        layout.addWidget(remember_hint)
        self.persistence_error_label = QLabel(self)
        self.persistence_error_label.setObjectName("slideAdjustmentsPersistenceError")
        self.persistence_error_label.setProperty("statusTone", "error")
        self.persistence_error_label.setWordWrap(True)
        self.persistence_error_label.setMaximumWidth(320)
        self.persistence_error_label.hide()
        layout.addWidget(self.persistence_error_label)
        self.reset_button = QPushButton("恢复原始显示", self)
        layout.addWidget(self.reset_button)

        self.sync_adjustments(*canvas.display_adjustments)
        self.sync_remember(remember)
        self.slider.valueChanged.connect(self._queue_value)
        self.spin_box.valueChanged.connect(self._queue_value)
        self.exposure_slider.valueChanged.connect(self._queue_exposure)
        self.exposure_spin_box.valueChanged.connect(
            lambda value: self._queue_exposure(round(value * 10))
        )
        self.slider.sliderReleased.connect(self.commit)
        self.exposure_slider.sliderReleased.connect(self.commit)
        self.spin_box.editingFinished.connect(self.commit)
        self.exposure_spin_box.editingFinished.connect(self.commit)
        self.remember_checkbox.toggled.connect(self._remember_toggled)
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
        return self._pending_adjustments is not None

    def _sync_controls(self, brightness: int, exposure_tenths: int) -> None:
        with (
            QSignalBlocker(self.slider), QSignalBlocker(self.spin_box),
            QSignalBlocker(self.exposure_slider), QSignalBlocker(self.exposure_spin_box),
        ):
            # A spin box may have accepted the numeric part of an unfinished
            # edit (for example "1." before "1.3"). Even setting its current
            # value again rewrites that text and moves the caret. Update only
            # the controls whose numeric value actually needs to change.
            for control, value in (
                (self.slider, brightness), (self.spin_box, brightness),
                (self.exposure_slider, exposure_tenths),
                (self.exposure_spin_box, exposure_tenths / 10),
            ):
                if control.value() != value:
                    control.setValue(value)

    def sync_adjustments(self, brightness: int, exposure_tenths: int) -> None:
        """Reflect external state without treating it as a user commit."""
        self._timer.stop()
        self._pending_adjustments = None
        self._edited_since_commit = False
        self._requested_adjustments = (brightness, exposure_tenths)
        self._sync_controls(brightness, exposure_tenths)

    def sync_value(self, value: int) -> None:
        """Compatibility entry for the original brightness-only control."""
        canvas = self.canvas
        self.sync_adjustments(value, canvas.display_exposure_tenths if canvas is not None else 0)

    def sync_remember(self, remember: bool) -> None:
        with QSignalBlocker(self.remember_checkbox):
            self.remember_checkbox.setChecked(remember)

    def set_persistence_error(self, message: str) -> None:
        self.persistence_error_label.setText(message)
        self.persistence_error_label.setVisible(bool(message))
        self.adjustSize()
        if self.isVisible() and self.screen() is not None:
            bounds = self.screen().availableGeometry()
            self.move(
                max(bounds.left(), min(self.x(), bounds.right() - self.width() + 1)),
                max(bounds.top(), min(self.y(), bounds.bottom() - self.height() + 1)),
            )

    def _queue_value(self, value: int) -> None:
        self._queue_adjustments(value, self.exposure_slider.value())

    def _queue_exposure(self, value: int) -> None:
        self._queue_adjustments(self.slider.value(), value)

    def _queue_adjustments(self, brightness: int, exposure_tenths: int) -> None:
        pair = (max(-100, min(100, int(brightness))), max(-40, min(40, int(exposure_tenths))))
        if pair == self._requested_adjustments:
            return
        self._requested_adjustments = pair
        self._sync_controls(*pair)
        self._pending_adjustments = pair
        self._edited_since_commit = True
        if not self._timer.isActive():
            # Keep cadence from actual application, including display work.
            # Initial/idle edits still merge asynchronously in the event loop.
            remaining_ns = (
                0 if self._last_apply_ns is None
                else self._UPDATE_INTERVAL_MS * 1_000_000
                - (monotonic_ns() - self._last_apply_ns)
            )
            self._timer.start(max(0, (remaining_ns + 999_999) // 1_000_000))

    def flush(self) -> None:
        """Apply the latest preview once; never emit a persistence request."""
        self._timer.stop()
        pair, self._pending_adjustments = self._pending_adjustments, None
        canvas = self.canvas
        if canvas is None:
            self._edited_since_commit = False
        elif pair is not None:
            self._last_apply_ns = monotonic_ns()
            canvas.set_display_adjustments(*pair)

    def commit(self) -> None:
        self.flush()
        edited, self._edited_since_commit = self._edited_since_commit, False
        canvas = self.canvas
        if edited and canvas is not None:
            self.adjustmentsCommitted.emit(*canvas.display_adjustments)

    def _remember_toggled(self, remember: bool) -> None:
        self.flush()
        # The toggle submits the complete current pair, so a subsequent hide
        # must not submit the same pending edit a second time.
        self._edited_since_commit = False
        if self.canvas is not None:
            self.rememberToggled.emit(remember)

    def _reset(self) -> None:
        self._queue_adjustments(0, 0)
        # Reset is also an explicit request to save neutral defaults, even if
        # this already-open canvas was neutral before another tab saved defaults.
        self._edited_since_commit = True
        self.commit()

    def hideEvent(self, event: QHideEvent) -> None:
        self.commit()
        super().hideEvent(event)
        if not self._closed:
            self._closed = True
            self.closed.emit()

    def wheelEvent(self, event: QWheelEvent) -> None:
        event.accept()
