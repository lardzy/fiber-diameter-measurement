from __future__ import annotations

from collections.abc import Mapping, Sequence
import math

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QKeyEvent,
    QMouseEvent,
    QPainter,
    QPaintEvent,
    QPalette,
    QPen,
    QPolygonF,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from fdm.ui.widgets import (
    NoWheelComboBox,
    NoWheelDoubleSpinBox,
    NoWheelSpinBox,
)


def _require_finite(value: float, name: str) -> float:
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name}必须是有限数")
    return normalized


def _set_widget_value_blocked(widget, value) -> None:
    blocked = widget.blockSignals(True)
    try:
        widget.setValue(value)
    finally:
        widget.blockSignals(blocked)


def _set_widget_range_blocked(widget, minimum, maximum) -> None:
    blocked = widget.blockSignals(True)
    try:
        widget.setRange(minimum, maximum)
    finally:
        widget.blockSignals(blocked)


class _HistogramCanvas(QWidget):
    """Palette-aware histogram surface with draggable threshold handles."""

    handleDragged = Signal(int, float)
    handleDragFinished = Signal(int, float)

    _HORIZONTAL_PADDING = 10.0
    _VERTICAL_PADDING = 9.0
    _HANDLE_SIZE = 6.0

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._counts: tuple[float, ...] = ()
        self._minimum = 0.0
        self._maximum = 1.0
        self._lower = 0.0
        self._upper = 1.0
        self._single_threshold = False
        self._active_handle: int | None = None
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMinimumHeight(112)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,
        )
        self.setAccessibleName("阈值直方图")
        self.setToolTip("拖动标记线调整阈值；数值框可用于精确输入")

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt API
        return QSize(228, 132)

    def setHistogram(self, counts: Sequence[float]) -> None:  # noqa: N802
        self._counts = tuple(float(value) for value in counts)
        self.update()

    def setState(  # noqa: N802 - Qt API
        self,
        *,
        minimum: float,
        maximum: float,
        lower: float,
        upper: float,
        single_threshold: bool,
    ) -> None:
        self._minimum = float(minimum)
        self._maximum = float(maximum)
        self._lower = float(lower)
        self._upper = float(upper)
        self._single_threshold = bool(single_threshold)
        self.update()

    def _plot_rect(self) -> QRectF:
        return QRectF(self.rect()).adjusted(
            self._HORIZONTAL_PADDING,
            self._VERTICAL_PADDING,
            -self._HORIZONTAL_PADDING,
            -self._VERTICAL_PADDING,
        )

    def _value_to_x(self, value: float) -> float:
        plot = self._plot_rect()
        span = self._maximum - self._minimum
        if span <= 0.0 or plot.width() <= 0.0:
            return plot.left()
        fraction = (float(value) - self._minimum) / span
        return plot.left() + min(1.0, max(0.0, fraction)) * plot.width()

    def _x_to_value(self, x: float) -> float:
        plot = self._plot_rect()
        if plot.width() <= 0.0:
            return self._minimum
        fraction = (float(x) - plot.left()) / plot.width()
        fraction = min(1.0, max(0.0, fraction))
        return self._minimum + fraction * (self._maximum - self._minimum)

    def _nearest_handle(self, x: float) -> int:
        if self._single_threshold:
            return 0
        lower_distance = abs(float(x) - self._value_to_x(self._lower))
        upper_distance = abs(float(x) - self._value_to_x(self._upper))
        return 0 if lower_distance <= upper_distance else 1

    def _emit_drag(self, x: float, *, finished: bool = False) -> None:
        if self._active_handle is None:
            return
        value = self._x_to_value(x)
        if finished:
            self.handleDragFinished.emit(self._active_handle, value)
        else:
            self.handleDragged.emit(self._active_handle, value)

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self._active_handle = self._nearest_handle(event.position().x())
            self.setFocus(Qt.FocusReason.MouseFocusReason)
            self._emit_drag(event.position().x())
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._active_handle is not None:
            self._emit_drag(event.position().x())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if (
            event.button() == Qt.MouseButton.LeftButton
            and self._active_handle is not None
        ):
            self._emit_drag(event.position().x(), finished=True)
            self._active_handle = None
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        if event.key() not in (Qt.Key.Key_Left, Qt.Key.Key_Right):
            super().keyPressEvent(event)
            return
        handle = 0 if self._active_handle is None else self._active_handle
        if self._single_threshold:
            handle = 0
        step = (self._maximum - self._minimum) / 1000.0
        if step <= 0.0:
            event.accept()
            return
        current = self._lower if handle == 0 else self._upper
        direction = -1.0 if event.key() == Qt.Key.Key_Left else 1.0
        self.handleDragFinished.emit(handle, current + direction * step)
        event.accept()

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        plot = self._plot_rect()
        palette = self.palette()

        painter.fillRect(self.rect(), palette.color(QPalette.ColorRole.Base))
        painter.setPen(QPen(palette.color(QPalette.ColorRole.Mid), 1.0))
        painter.drawRoundedRect(plot, 4.0, 4.0)

        if self._counts and plot.width() > 0.0 and plot.height() > 0.0:
            peak = max(self._counts)
            if peak > 0.0:
                bar_color = QColor(
                    palette.color(QPalette.ColorRole.PlaceholderText)
                )
                bar_color.setAlpha(150)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(bar_color)
                bin_width = plot.width() / len(self._counts)
                for index, count in enumerate(self._counts):
                    fraction = min(1.0, max(0.0, count / peak))
                    height = max(0.0, fraction * (plot.height() - 2.0))
                    left = plot.left() + index * bin_width
                    painter.drawRect(
                        QRectF(
                            left,
                            plot.bottom() - height,
                            max(1.0, bin_width),
                            height,
                        )
                    )

        lower_x = self._value_to_x(self._lower)
        upper_x = self._value_to_x(self._upper)
        selection_color = QColor(
            palette.color(QPalette.ColorRole.Highlight)
        )
        selection_color.setAlpha(42)
        painter.fillRect(
            QRectF(
                min(lower_x, upper_x),
                plot.top() + 1.0,
                abs(upper_x - lower_x),
                max(0.0, plot.height() - 2.0),
            ),
            selection_color,
        )

        marker_color = palette.color(QPalette.ColorRole.Highlight)
        painter.setPen(QPen(marker_color, 2.0))
        marker_positions = (lower_x,) if self._single_threshold else (lower_x, upper_x)
        painter.setBrush(marker_color)
        for x in marker_positions:
            painter.drawLine(QPointF(x, plot.top()), QPointF(x, plot.bottom()))
            painter.drawPolygon(
                QPolygonF(
                    (
                        QPointF(x - self._HANDLE_SIZE, plot.top()),
                        QPointF(x + self._HANDLE_SIZE, plot.top()),
                        QPointF(x, plot.top() + self._HANDLE_SIZE),
                    )
                )
            )


class HistogramRangeEditor(QWidget):
    """Histogram and exact numeric editor for one threshold or a value range."""

    DISPLAY_MODES: tuple[tuple[str, str], ...] = (
        ("bw", "黑白"),
        ("red_overlay", "红色覆盖"),
        ("over_under", "Over/Under"),
    )
    FOREGROUND_POLARITIES: tuple[tuple[str, str], ...] = (
        ("bright", "亮前景"),
        ("dark", "暗前景"),
    )

    rangeChanged = Signal(float, float)
    thresholdsChanged = Signal(float, float)
    thresholdChanged = Signal(float)
    validityChanged = Signal(bool)
    autoRequested = Signal()
    resetRequested = Signal()
    displayModeChanged = Signal(str)
    foregroundPolarityChanged = Signal(str)
    editFinished = Signal()
    interactionFinished = Signal()

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        single_threshold: bool = False,
        minimum: float = 0.0,
        maximum: float = 255.0,
        lower: float | None = None,
        upper: float | None = None,
        decimals: int = 6,
        suffix: str = "",
    ) -> None:
        super().__init__(parent)
        self._minimum = 0.0
        self._maximum = 255.0
        self._lower = 0.0
        self._upper = 255.0
        self._single_threshold = bool(single_threshold)
        self._valid = True
        self._display_mode = "bw"
        self._foreground_polarity = "bright"
        self._selection_statistics: tuple[int, int] | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(6)
        self.histogramCanvas = _HistogramCanvas(self)
        root.addWidget(self.histogramCanvas)

        values = QGridLayout()
        values.setContentsMargins(0, 0, 0, 0)
        values.setHorizontalSpacing(8)
        values.setVerticalSpacing(4)
        self.lowerLabel = QLabel(
            "阈值" if self._single_threshold else "下限",
            self,
        )
        self.upperLabel = QLabel("上限", self)
        self.lowerSpin = NoWheelDoubleSpinBox(self)
        self.upperSpin = NoWheelDoubleSpinBox(self)
        for spin in (self.lowerSpin, self.upperSpin):
            spin.setKeyboardTracking(False)
            spin.setDecimals(max(0, min(12, int(decimals))))
            spin.setSuffix(str(suffix))
            spin.setSizePolicy(
                QSizePolicy.Policy.Expanding,
                QSizePolicy.Policy.Fixed,
            )
        values.addWidget(self.lowerLabel, 0, 0)
        values.addWidget(self.lowerSpin, 0, 1)
        values.addWidget(self.upperLabel, 1, 0)
        values.addWidget(self.upperSpin, 1, 1)
        values.setColumnStretch(1, 1)
        root.addLayout(values)

        controls = QHBoxLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setSpacing(6)
        self.autoButton = QPushButton("自动", self)
        self.autoButton.setToolTip("请求使用当前图像统计自动计算阈值")
        self.resetButton = QPushButton("重置", self)
        self.resetButton.setToolTip("请求恢复当前操作的默认阈值")
        self.displayModeCombo = NoWheelComboBox(self)
        self.displayModeCombo.setAccessibleName("阈值显示模式")
        self.displayModeCombo.setToolTip(
            "只改变阈值预览的显示方式，不改变阈值和最终计算结果"
        )
        for value, label in self.DISPLAY_MODES:
            self.displayModeCombo.addItem(label, value)
        self.polarityCombo = NoWheelComboBox(self)
        self.polarityCombo.setAccessibleName("前景极性")
        self.polarityCombo.setToolTip(
            "选择高于阈值的亮像素或低于阈值的暗像素作为前景"
        )
        for value, label in self.FOREGROUND_POLARITIES:
            self.polarityCombo.addItem(label, value)
        controls.addWidget(self.autoButton)
        controls.addWidget(self.resetButton)
        controls.addStretch(1)
        root.addLayout(controls)

        options = QGridLayout()
        options.setContentsMargins(0, 0, 0, 0)
        options.setHorizontalSpacing(8)
        options.setVerticalSpacing(4)
        self.displayModeLabel = QLabel("显示", self)
        options.addWidget(self.displayModeLabel, 0, 0)
        options.addWidget(self.displayModeCombo, 0, 1)
        self.polarityLabel = QLabel("前景", self)
        options.addWidget(self.polarityLabel, 1, 0)
        options.addWidget(self.polarityCombo, 1, 1)
        options.setColumnStretch(1, 1)
        root.addLayout(options)

        self.selectionStatisticsLabel = QLabel("选中像素：—", self)
        self.selectionStatisticsLabel.setObjectName(
            "histogramSelectionStatistics"
        )
        self.selectionStatisticsLabel.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.selectionStatisticsLabel.setToolTip(
            "由当前预览样本统计；最终处理仍使用原始分辨率像素"
        )
        root.addWidget(self.selectionStatisticsLabel)

        self.lowerSpin.valueChanged.connect(self._lower_spin_changed)
        self.upperSpin.valueChanged.connect(self._upper_spin_changed)
        self.histogramCanvas.handleDragged.connect(self._canvas_handle_changed)
        self.histogramCanvas.handleDragFinished.connect(
            self._canvas_handle_finished
        )
        self.lowerSpin.editingFinished.connect(
            self._emit_interaction_finished
        )
        self.upperSpin.editingFinished.connect(
            self._emit_interaction_finished
        )
        self.autoButton.clicked.connect(self._request_auto)
        self.resetButton.clicked.connect(self._request_reset)
        self.displayModeCombo.currentIndexChanged.connect(
            self._display_mode_changed
        )
        self.polarityCombo.currentIndexChanged.connect(
            self._foreground_polarity_changed
        )

        self.setRange(minimum, maximum, emit_signal=False)
        initial_lower = minimum if lower is None else lower
        initial_upper = maximum if upper is None else upper
        if self._single_threshold:
            initial_upper = maximum
        self.setThresholds(initial_lower, initial_upper, emit_signal=False)
        self._update_mode_visibility()
        self._sync_children()

    def setHistogram(  # noqa: N802 - Qt API
        self,
        counts: Sequence[float],
        *,
        value_range: tuple[float, float] | None = None,
        emit_range_signal: bool = False,
    ) -> None:
        """Replace display statistics without silently editing the recipe.

        Histogram refreshes are driven by preview completion and ROI/channel
        changes.  Those data updates must not be mistaken for a user threshold
        edit, even when a new native value range needs to clamp the handles.
        Callers that intentionally treat a range replacement as an edit can
        opt in explicitly.
        """

        normalized: list[float] = []
        for value in counts:
            count = _require_finite(float(value), "直方图计数")
            if count < 0.0:
                raise ValueError("直方图计数不能为负数")
            normalized.append(count)
        if value_range is not None:
            self.setRange(
                *value_range,
                emit_signal=bool(emit_range_signal),
            )
        self.histogramCanvas.setHistogram(normalized)

    def setRange(  # noqa: N802 - Qt API
        self,
        minimum: float,
        maximum: float,
        *,
        emit_signal: bool = True,
    ) -> None:
        normalized_minimum = _require_finite(minimum, "范围下限")
        normalized_maximum = _require_finite(maximum, "范围上限")
        if normalized_maximum <= normalized_minimum:
            raise ValueError("范围上限必须大于范围下限")
        range_changed = (
            normalized_minimum != self._minimum
            or normalized_maximum != self._maximum
        )
        self._minimum = normalized_minimum
        self._maximum = normalized_maximum
        _set_widget_range_blocked(
            self.lowerSpin,
            normalized_minimum,
            normalized_maximum,
        )
        _set_widget_range_blocked(
            self.upperSpin,
            normalized_minimum,
            normalized_maximum,
        )
        lower = min(normalized_maximum, max(normalized_minimum, self._lower))
        upper = min(normalized_maximum, max(normalized_minimum, self._upper))
        if self._single_threshold:
            upper = normalized_maximum
        elif lower > upper:
            lower = upper
        threshold_changed = lower != self._lower or upper != self._upper
        self._lower, self._upper = lower, upper
        self._sync_children()
        if emit_signal and range_changed:
            self.rangeChanged.emit(self._minimum, self._maximum)
        if emit_signal and threshold_changed:
            self._emit_threshold_signals()

    def range(self) -> tuple[float, float]:
        return self._minimum, self._maximum

    def setThresholds(  # noqa: N802 - Qt API
        self,
        lower: float,
        upper: float | None = None,
        *,
        emit_signal: bool = True,
    ) -> None:
        normalized_lower = _require_finite(lower, "阈值下限")
        normalized_upper = (
            self._maximum
            if upper is None or self._single_threshold
            else _require_finite(upper, "阈值上限")
        )
        if normalized_lower < self._minimum or normalized_lower > self._maximum:
            raise ValueError("阈值下限超出有效范围")
        if normalized_upper < self._minimum or normalized_upper > self._maximum:
            raise ValueError("阈值上限超出有效范围")
        if normalized_lower > normalized_upper:
            raise ValueError("阈值下限不能大于阈值上限")
        changed = (
            normalized_lower != self._lower
            or normalized_upper != self._upper
        )
        self._lower = normalized_lower
        self._upper = normalized_upper
        self._set_valid(True)
        self._sync_children()
        if emit_signal and changed:
            self._emit_threshold_signals()

    def thresholds(self) -> tuple[float, float]:
        return self._lower, self._upper

    def setThreshold(  # noqa: N802 - Qt API
        self,
        value: float,
        *,
        emit_signal: bool = True,
    ) -> None:
        self.setThresholds(value, self._maximum, emit_signal=emit_signal)

    def threshold(self) -> float:
        return self._lower

    def setSingleThreshold(  # noqa: N802 - Qt API
        self,
        single_threshold: bool,
        *,
        emit_signal: bool = True,
    ) -> None:
        normalized = bool(single_threshold)
        if normalized == self._single_threshold:
            return
        self._single_threshold = normalized
        if normalized:
            self._upper = self._maximum
        self._update_mode_visibility()
        self._sync_children()
        if emit_signal:
            self._emit_threshold_signals()

    def isSingleThreshold(self) -> bool:  # noqa: N802 - Qt API
        return self._single_threshold

    def setDecimals(self, decimals: int) -> None:  # noqa: N802 - Qt API
        normalized = max(0, min(12, int(decimals)))
        for spin in (self.lowerSpin, self.upperSpin):
            blocked = spin.blockSignals(True)
            try:
                spin.setDecimals(normalized)
            finally:
                spin.blockSignals(blocked)
        self._lower = float(self.lowerSpin.value())
        self._upper = (
            self._maximum
            if self._single_threshold
            else float(self.upperSpin.value())
        )
        self._sync_children()

    def setSuffix(self, suffix: str) -> None:  # noqa: N802 - Qt API
        self.lowerSpin.setSuffix(str(suffix))
        self.upperSpin.setSuffix(str(suffix))

    def isValid(self) -> bool:  # noqa: N802 - Qt API
        return self._valid

    def requestAuto(self) -> None:  # noqa: N802 - Qt API
        """Request an automatic threshold from the owning workbench."""

        self._request_auto()

    def requestReset(self) -> None:  # noqa: N802 - Qt API
        """Request restoring defaults from the owning workbench."""

        self._request_reset()

    def setDisplayMode(  # noqa: N802 - Qt API
        self,
        mode: str,
        *,
        emit_signal: bool = True,
    ) -> None:
        normalized = str(mode).strip().lower()
        supported = {value for value, _label in self.DISPLAY_MODES}
        if normalized not in supported:
            raise ValueError(f"不支持的阈值显示模式：{mode}")
        changed = normalized != self._display_mode
        self._display_mode = normalized
        blocked = self.displayModeCombo.blockSignals(True)
        try:
            index = self.displayModeCombo.findData(normalized)
            self.displayModeCombo.setCurrentIndex(index)
        finally:
            self.displayModeCombo.blockSignals(blocked)
        if emit_signal and changed:
            self.displayModeChanged.emit(normalized)

    def displayMode(self) -> str:  # noqa: N802 - Qt API
        return self._display_mode

    def setForegroundPolarity(  # noqa: N802 - Qt API
        self,
        polarity: str,
        *,
        emit_signal: bool = True,
    ) -> None:
        normalized = str(polarity).strip().lower()
        supported = {value for value, _label in self.FOREGROUND_POLARITIES}
        if normalized not in supported:
            raise ValueError(f"不支持的前景极性：{polarity}")
        changed = normalized != self._foreground_polarity
        self._foreground_polarity = normalized
        blocked = self.polarityCombo.blockSignals(True)
        try:
            index = self.polarityCombo.findData(normalized)
            self.polarityCombo.setCurrentIndex(index)
        finally:
            self.polarityCombo.blockSignals(blocked)
        if emit_signal and changed:
            self.foregroundPolarityChanged.emit(normalized)

    def foregroundPolarity(self) -> str:  # noqa: N802 - Qt API
        return self._foreground_polarity

    def setSelectionStatistics(  # noqa: N802 - Qt API
        self,
        selected_count: int,
        total_count: int,
    ) -> None:
        selected = int(selected_count)
        total = int(total_count)
        if selected < 0 or total < 0:
            raise ValueError("像素计数不能为负数")
        if selected > total:
            raise ValueError("选中像素数不能大于总像素数")
        self._selection_statistics = (selected, total)
        percentage = 0.0 if total == 0 else selected * 100.0 / total
        self.selectionStatisticsLabel.setText(
            f"选中像素：{selected:,} / {total:,}（{percentage:.2f}%）"
        )

    def setBandStatistics(  # noqa: N802 - Qt API
        self,
        *,
        below_count: int,
        within_count: int,
        above_count: int,
        total_count: int,
        foreground_count: int,
    ) -> None:
        """Show exact low/in/high bands while preserving the public summary."""

        below = int(below_count)
        within = int(within_count)
        above = int(above_count)
        total = int(total_count)
        foreground = int(foreground_count)
        if min(below, within, above, total, foreground) < 0:
            raise ValueError("阈值区间像素数不能为负数")
        if below + within + above != total:
            raise ValueError("阈值区间像素数之和必须等于有效像素数")
        if foreground > total:
            raise ValueError("前景像素数不能大于有效像素数")
        self._selection_statistics = (foreground, total)
        percentage = (
            0.0
            if total == 0
            else foreground * 100.0 / total
        )
        if self._single_threshold:
            bands = (
                f"≤阈值 {below + within:,} · "
                f">阈值 {above:,}"
            )
        else:
            bands = (
                f"低于 {below:,} · 范围内 {within:,} · "
                f"高于 {above:,}"
            )
        self.selectionStatisticsLabel.setText(
            f"{bands} · 当前前景 {foreground:,}/{total:,}"
            f"（{percentage:.2f}%）"
        )

    def selectionStatistics(self) -> tuple[int, int] | None:  # noqa: N802
        return self._selection_statistics

    def clearSelectionStatistics(self) -> None:  # noqa: N802 - Qt API
        self._selection_statistics = None
        self.selectionStatisticsLabel.setText("选中像素：—")

    def _lower_spin_changed(self, value: float) -> None:
        normalized = min(float(value), self._upper)
        if normalized == self._lower:
            if normalized != value:
                _set_widget_value_blocked(self.lowerSpin, normalized)
            return
        self._lower = normalized
        self._set_valid(True)
        self._sync_children()
        self._emit_threshold_signals()

    def _upper_spin_changed(self, value: float) -> None:
        if self._single_threshold:
            return
        normalized = max(float(value), self._lower)
        if normalized == self._upper:
            if normalized != value:
                _set_widget_value_blocked(self.upperSpin, normalized)
            return
        self._upper = normalized
        self._set_valid(True)
        self._sync_children()
        self._emit_threshold_signals()

    def _canvas_handle_changed(self, handle: int, value: float) -> None:
        normalized = min(self._maximum, max(self._minimum, float(value)))
        if int(handle) == 0:
            normalized = min(normalized, self._upper)
            if normalized == self._lower:
                return
            self._lower = normalized
        elif not self._single_threshold:
            normalized = max(normalized, self._lower)
            if normalized == self._upper:
                return
            self._upper = normalized
        else:
            return
        self._set_valid(True)
        self._sync_children()
        self._emit_threshold_signals()

    def _canvas_handle_finished(self, handle: int, value: float) -> None:
        self._canvas_handle_changed(handle, value)
        self._emit_interaction_finished()

    def _request_auto(self) -> None:
        self.autoRequested.emit()
        self._emit_interaction_finished()

    def _request_reset(self) -> None:
        self.resetRequested.emit()
        self._emit_interaction_finished()

    def _display_mode_changed(self, index: int) -> None:
        mode = str(self.displayModeCombo.itemData(int(index)) or "")
        if not mode or mode == self._display_mode:
            return
        self._display_mode = mode
        self.displayModeChanged.emit(mode)
        self._emit_interaction_finished()

    def _foreground_polarity_changed(self, index: int) -> None:
        polarity = str(self.polarityCombo.itemData(int(index)) or "")
        if not polarity or polarity == self._foreground_polarity:
            return
        self._foreground_polarity = polarity
        self.foregroundPolarityChanged.emit(polarity)
        self._emit_interaction_finished()

    def _emit_interaction_finished(self) -> None:
        self.editFinished.emit()
        self.interactionFinished.emit()

    def _update_mode_visibility(self) -> None:
        self.lowerLabel.setText("阈值" if self._single_threshold else "下限")
        self.upperLabel.setVisible(not self._single_threshold)
        self.upperSpin.setVisible(not self._single_threshold)

    def _sync_children(self) -> None:
        _set_widget_value_blocked(self.lowerSpin, self._lower)
        _set_widget_value_blocked(self.upperSpin, self._upper)
        self.histogramCanvas.setState(
            minimum=self._minimum,
            maximum=self._maximum,
            lower=self._lower,
            upper=self._upper,
            single_threshold=self._single_threshold,
        )

    def _emit_threshold_signals(self) -> None:
        self.thresholdsChanged.emit(self._lower, self._upper)
        self.thresholdChanged.emit(self._lower)

    def _set_valid(self, valid: bool) -> None:
        normalized = bool(valid)
        if normalized == self._valid:
            return
        self._valid = normalized
        self.validityChanged.emit(normalized)


class _NoWheelSlider(QSlider):
    """Slider that ignores ordinary wheel gestures."""

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802 - Qt API
        event.ignore()


class SliderNumberEditor(QWidget):
    """Linearly mapped slider paired with an exact floating-point editor."""

    valueChanged = Signal(float)
    rangeChanged = Signal(float, float)
    editFinished = Signal()
    interactionFinished = Signal()

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        minimum: float = 0.0,
        maximum: float = 1.0,
        value: float | None = None,
        decimals: int = 3,
        suffix: str = "",
        resolution: int = 10_000,
    ) -> None:
        super().__init__(parent)
        self._minimum = 0.0
        self._maximum = 1.0
        self._value = 0.0
        self._resolution = max(100, int(resolution))

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(8)
        self.slider = _NoWheelSlider(Qt.Orientation.Horizontal, self)
        self.slider.setRange(0, self._resolution)
        self.slider.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        self.spinBox = NoWheelDoubleSpinBox(self)
        self.spinBox.setKeyboardTracking(False)
        self.spinBox.setDecimals(max(0, min(12, int(decimals))))
        self.spinBox.setSuffix(str(suffix))
        root.addWidget(self.slider, 1)
        root.addWidget(self.spinBox)

        self.slider.valueChanged.connect(self._slider_changed)
        self.spinBox.valueChanged.connect(self._spin_changed)
        self.slider.sliderReleased.connect(self._emit_interaction_finished)
        self.spinBox.editingFinished.connect(
            self._emit_interaction_finished
        )
        self.setRange(minimum, maximum, emit_signal=False)
        self.setValue(minimum if value is None else value, emit_signal=False)

    def setRange(  # noqa: N802 - Qt API
        self,
        minimum: float,
        maximum: float,
        *,
        emit_signal: bool = True,
    ) -> None:
        normalized_minimum = _require_finite(minimum, "范围下限")
        normalized_maximum = _require_finite(maximum, "范围上限")
        if normalized_maximum <= normalized_minimum:
            raise ValueError("范围上限必须大于范围下限")
        changed = (
            normalized_minimum != self._minimum
            or normalized_maximum != self._maximum
        )
        self._minimum = normalized_minimum
        self._maximum = normalized_maximum
        _set_widget_range_blocked(
            self.spinBox,
            normalized_minimum,
            normalized_maximum,
        )
        self._value = min(
            normalized_maximum,
            max(normalized_minimum, self._value),
        )
        self._sync_children()
        if emit_signal and changed:
            self.rangeChanged.emit(self._minimum, self._maximum)

    def range(self) -> tuple[float, float]:
        return self._minimum, self._maximum

    def setValue(  # noqa: N802 - Qt API
        self,
        value: float,
        *,
        emit_signal: bool = True,
    ) -> None:
        normalized = _require_finite(value, "参数值")
        normalized = min(self._maximum, max(self._minimum, normalized))
        decimals = self.spinBox.decimals()
        normalized = round(normalized, decimals)
        changed = normalized != self._value
        self._value = normalized
        self._sync_children()
        if emit_signal and changed:
            self.valueChanged.emit(self._value)

    def value(self) -> float:
        return self._value

    def setDecimals(self, decimals: int) -> None:  # noqa: N802 - Qt API
        self.spinBox.setDecimals(max(0, min(12, int(decimals))))
        self.setValue(self._value, emit_signal=False)

    def decimals(self) -> int:
        return self.spinBox.decimals()

    def setSuffix(self, suffix: str) -> None:  # noqa: N802 - Qt API
        self.spinBox.setSuffix(str(suffix))

    def suffix(self) -> str:
        return self.spinBox.suffix()

    def setSingleStep(self, step: float) -> None:  # noqa: N802 - Qt API
        normalized = _require_finite(step, "步长")
        if normalized <= 0.0:
            raise ValueError("步长必须大于零")
        self.spinBox.setSingleStep(normalized)
        slider_step = max(
            1,
            int(round(normalized / (self._maximum - self._minimum) * self._resolution)),
        )
        self.slider.setSingleStep(slider_step)
        self.slider.setPageStep(max(slider_step, slider_step * 10))

    def setSliderResolution(self, resolution: int) -> None:  # noqa: N802
        normalized = max(100, int(resolution))
        if normalized == self._resolution:
            return
        self._resolution = normalized
        self.slider.setRange(0, normalized)
        self._sync_children()

    def _value_to_slider(self, value: float) -> int:
        fraction = (float(value) - self._minimum) / (
            self._maximum - self._minimum
        )
        return int(round(min(1.0, max(0.0, fraction)) * self._resolution))

    def _slider_to_value(self, position: int) -> float:
        fraction = int(position) / self._resolution
        return self._minimum + fraction * (self._maximum - self._minimum)

    def _slider_changed(self, position: int) -> None:
        normalized = round(
            self._slider_to_value(position),
            self.spinBox.decimals(),
        )
        if normalized == self._value:
            return
        self._value = normalized
        _set_widget_value_blocked(self.spinBox, normalized)
        self.valueChanged.emit(normalized)

    def _spin_changed(self, value: float) -> None:
        normalized = float(value)
        if normalized == self._value:
            return
        self._value = normalized
        blocked = self.slider.blockSignals(True)
        try:
            self.slider.setValue(self._value_to_slider(normalized))
        finally:
            self.slider.blockSignals(blocked)
        self.valueChanged.emit(normalized)

    def _sync_children(self) -> None:
        _set_widget_value_blocked(self.spinBox, self._value)
        blocked = self.slider.blockSignals(True)
        try:
            self.slider.setValue(self._value_to_slider(self._value))
        finally:
            self.slider.blockSignals(blocked)

    def _emit_interaction_finished(self) -> None:
        self.editFinished.emit()
        self.interactionFinished.emit()


class PercentileRangeEditor(QWidget):
    """Exact lower/upper percentile editor with resolved-value feedback."""

    valueChanged = Signal(float, float)
    editFinished = Signal()
    validityChanged = Signal(bool)
    validationChanged = Signal(bool, str)

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        lower: float = 0.5,
        upper: float = 99.5,
        decimals: int = 3,
    ) -> None:
        super().__init__(parent)
        self._valid = True
        self._validation_message = ""
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(6)
        form = QGridLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setHorizontalSpacing(8)
        form.setVerticalSpacing(5)
        self.lowerLabel = QLabel("下百分位", self)
        self.lowerEditor = SliderNumberEditor(
            self,
            minimum=0.0,
            maximum=99.999,
            value=lower,
            decimals=decimals,
            suffix=" %",
        )
        self.lowerSpin = self.lowerEditor.spinBox
        self.upperLabel = QLabel("上百分位", self)
        self.upperEditor = SliderNumberEditor(
            self,
            minimum=0.001,
            maximum=100.0,
            value=upper,
            decimals=decimals,
            suffix=" %",
        )
        self.upperSpin = self.upperEditor.spinBox
        form.addWidget(self.lowerLabel, 0, 0)
        form.addWidget(self.lowerEditor, 0, 1)
        form.addWidget(self.upperLabel, 1, 0)
        form.addWidget(self.upperEditor, 1, 1)
        form.setColumnStretch(1, 1)
        root.addLayout(form)

        self.saturationLabel = QLabel(self)
        self.saturationLabel.setObjectName("percentileSaturationSummary")
        self.saturationLabel.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        root.addWidget(self.saturationLabel)
        self.resolvedValuesLabel = QLabel(
            "正在读取当前步骤输入的实际强度分位值…",
            self,
        )
        self.resolvedValuesLabel.setObjectName(
            "percentileResolvedValues"
        )
        self.resolvedValuesLabel.setWordWrap(True)
        self.resolvedValuesLabel.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        root.addWidget(self.resolvedValuesLabel)
        self.validationLabel = QLabel(self)
        self.validationLabel.setObjectName(
            "percentileRangeValidation"
        )
        self.validationLabel.setWordWrap(True)
        root.addWidget(self.validationLabel)

        self.lowerEditor.valueChanged.connect(
            self._values_changed
        )
        self.upperEditor.valueChanged.connect(
            self._values_changed
        )
        self.lowerEditor.editFinished.connect(
            self._emit_edit_finished
        )
        self.upperEditor.editFinished.connect(
            self._emit_edit_finished
        )
        self.setValue(lower, upper, emit_signal=False)

    def value(self) -> tuple[float, float]:
        return (
            float(self.lowerEditor.value()),
            float(self.upperEditor.value()),
        )

    def setValue(  # noqa: N802 - Qt API
        self,
        lower: float,
        upper: float,
        *,
        emit_signal: bool = True,
    ) -> None:
        previous = self.value()
        self.lowerEditor.setValue(float(lower), emit_signal=False)
        self.upperEditor.setValue(float(upper), emit_signal=False)
        self._refresh_state()
        if emit_signal and self.value() != previous:
            self.valueChanged.emit(*self.value())

    def isValid(self) -> bool:  # noqa: N802 - Qt API
        return self._valid

    def validationMessage(self) -> str:  # noqa: N802 - Qt API
        return self._validation_message

    def setResolvedText(self, text: str) -> None:  # noqa: N802
        self.resolvedValuesLabel.setText(str(text))

    def _values_changed(self, _value: float) -> None:
        self._refresh_state()
        self.valueChanged.emit(*self.value())

    def _refresh_state(self) -> None:
        lower, upper = self.value()
        valid = 0.0 <= lower < upper <= 100.0
        message = (
            "百分位范围有效"
            if valid
            else "下百分位必须小于上百分位"
        )
        low_tail = max(0.0, lower)
        high_tail = max(0.0, 100.0 - upper)
        self.saturationLabel.setText(
            f"预计裁剪低端 {low_tail:.3f}% · "
            f"高端 {high_tail:.3f}%"
        )
        state_changed = valid != self._valid
        message_changed = message != self._validation_message
        self._valid = valid
        self._validation_message = message
        self.validationLabel.setText(message)
        self.validationLabel.setVisible(not valid)
        if state_changed:
            self.validityChanged.emit(valid)
        if state_changed or message_changed:
            self.validationChanged.emit(valid, message)

    def _emit_edit_finished(self) -> None:
        self.editFinished.emit()


class _FrequencyResponseCanvas(QWidget):
    """Lightweight, palette-aware Butterworth response preview."""

    _LEFT_MARGIN = 34.0
    _TOP_MARGIN = 10.0
    _RIGHT_MARGIN = 10.0
    _BOTTOM_MARGIN = 24.0

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._minimum = 0.0
        self._maximum = 0.5
        self._mode = "lowpass"
        self._low_cutoff = 0.05
        self._high_cutoff = 0.15
        self._order = 2
        self.setMinimumHeight(112)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,
        )
        self.setAccessibleName("FFT 频率响应曲线")
        self.setToolTip(
            "显示当前 Butterworth 参数的理论幅频响应；"
            "仅用于参数预览，不执行 FFT"
        )

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt API
        return QSize(228, 132)

    def setState(  # noqa: N802 - Qt API
        self,
        *,
        minimum: float,
        maximum: float,
        mode: str,
        low_cutoff: float,
        high_cutoff: float,
        order: int,
    ) -> None:
        self._minimum = float(minimum)
        self._maximum = float(maximum)
        self._mode = str(mode)
        self._low_cutoff = float(low_cutoff)
        self._high_cutoff = float(high_cutoff)
        self._order = int(order)
        self.update()

    def _plot_rect(self) -> QRectF:
        return QRectF(self.rect()).adjusted(
            self._LEFT_MARGIN,
            self._TOP_MARGIN,
            -self._RIGHT_MARGIN,
            -self._BOTTOM_MARGIN,
        )

    def _frequency_to_x(self, frequency: float) -> float:
        plot = self._plot_rect()
        span = self._maximum - self._minimum
        if span <= 0.0 or plot.width() <= 0.0:
            return plot.left()
        fraction = (float(frequency) - self._minimum) / span
        return plot.left() + min(1.0, max(0.0, fraction)) * plot.width()

    @staticmethod
    def _lowpass_response(
        frequency: float,
        cutoff: float,
        order: int,
    ) -> float:
        epsilon = 1e-12
        if frequency <= 0.0:
            return 1.0
        ratio = frequency / max(float(cutoff), epsilon)
        exponent = 2.0 * max(1, int(order)) * math.log(max(ratio, epsilon))
        if exponent >= 700.0:
            return 0.0
        if exponent <= -700.0:
            return 1.0
        return 1.0 / (1.0 + math.exp(exponent))

    def _response_at(self, frequency: float) -> float:
        lowpass_high = self._lowpass_response(
            frequency,
            self._high_cutoff,
            self._order,
        )
        highpass_low = 1.0 - self._lowpass_response(
            frequency,
            self._low_cutoff,
            self._order,
        )
        if self._mode == "lowpass":
            return lowpass_high
        if self._mode == "highpass":
            return highpass_low
        bandpass = highpass_low * lowpass_high
        if self._mode == "bandpass":
            return bandpass
        return 1.0 - bandpass

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        palette = self.palette()
        plot = self._plot_rect()

        painter.fillRect(self.rect(), palette.color(QPalette.ColorRole.Base))
        painter.setPen(QPen(palette.color(QPalette.ColorRole.Mid), 1.0))
        painter.drawRoundedRect(plot, 4.0, 4.0)
        if plot.width() <= 0.0 or plot.height() <= 0.0:
            return

        points = QPolygonF()
        sample_count = max(64, min(384, int(plot.width())))
        span = self._maximum - self._minimum
        for index in range(sample_count + 1):
            fraction = index / sample_count
            frequency = self._minimum + fraction * span
            response = min(1.0, max(0.0, self._response_at(frequency)))
            points.append(
                QPointF(
                    plot.left() + fraction * plot.width(),
                    plot.bottom() - response * plot.height(),
                )
            )

        fill = QPolygonF((QPointF(plot.left(), plot.bottom()),))
        fill += points
        fill.append(QPointF(plot.right(), plot.bottom()))
        response_color = QColor(
            palette.color(QPalette.ColorRole.Highlight)
        )
        response_fill = QColor(response_color)
        response_fill.setAlpha(45)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(response_fill)
        painter.drawPolygon(fill)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(response_color, 2.0))
        painter.drawPolyline(points)

        cutoff_pen = QPen(
            palette.color(QPalette.ColorRole.Link),
            1.0,
            Qt.PenStyle.DashLine,
        )
        painter.setPen(cutoff_pen)
        cutoffs: tuple[float, ...]
        if self._mode == "lowpass":
            cutoffs = (self._high_cutoff,)
        elif self._mode == "highpass":
            cutoffs = (self._low_cutoff,)
        else:
            cutoffs = (self._low_cutoff, self._high_cutoff)
        for cutoff in cutoffs:
            x = self._frequency_to_x(cutoff)
            painter.drawLine(QPointF(x, plot.top()), QPointF(x, plot.bottom()))

        painter.setPen(
            QPen(palette.color(QPalette.ColorRole.PlaceholderText), 1.0)
        )
        painter.drawText(
            QRectF(0.0, plot.top() - 2.0, self._LEFT_MARGIN - 5.0, 18.0),
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
            "1",
        )
        painter.drawText(
            QRectF(
                plot.left(),
                plot.bottom() + 3.0,
                plot.width(),
                self._BOTTOM_MARGIN - 3.0,
            ),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
            f"{self._minimum:g}",
        )
        painter.drawText(
            QRectF(
                plot.left(),
                plot.bottom() + 3.0,
                plot.width(),
                self._BOTTOM_MARGIN - 3.0,
            ),
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop,
            f"{self._maximum:g}",
        )


class FrequencyResponseEditor(QWidget):
    """Professional FFT filter parameter editor with a response preview."""

    MODES: tuple[tuple[str, str], ...] = (
        ("lowpass", "低通"),
        ("highpass", "高通"),
        ("bandpass", "带通"),
        ("bandstop", "带阻"),
    )

    valueChanged = Signal(dict)
    editFinished = Signal()
    interactionFinished = Signal()
    validityChanged = Signal(bool)
    validationChanged = Signal(bool, str)

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        mode: str = "lowpass",
        low_cutoff: float = 0.05,
        high_cutoff: float = 0.15,
        order: int = 2,
        minimum: float = 0.0,
        maximum: float = 0.5,
        decimals: int = 6,
        suffix: str = " cycles/px",
    ) -> None:
        super().__init__(parent)
        normalized_minimum = _require_finite(minimum, "频率下限")
        normalized_maximum = _require_finite(maximum, "频率上限")
        if normalized_maximum <= normalized_minimum:
            raise ValueError("频率上限必须大于频率下限")
        self._minimum = normalized_minimum
        self._maximum = normalized_maximum
        self._valid = True
        self._validation_message = ""
        self._updating = False

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(8)
        form = QGridLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setHorizontalSpacing(8)
        form.setVerticalSpacing(6)

        self.modeLabel = QLabel("滤波模式", self)
        self.modeCombo = NoWheelComboBox(self)
        self.modeCombo.setAccessibleName("FFT 滤波模式")
        for value, label in self.MODES:
            self.modeCombo.addItem(label, value)
        form.addWidget(self.modeLabel, 0, 0)
        form.addWidget(self.modeCombo, 0, 1)

        self.lowCutoffLabel = QLabel("低截止", self)
        self.lowCutoffEditor = SliderNumberEditor(
            self,
            minimum=self._minimum,
            maximum=self._maximum,
            value=low_cutoff,
            decimals=decimals,
            suffix=suffix,
        )
        self.lowCutoffSpin = self.lowCutoffEditor.spinBox
        self.lowCutoffSlider = self.lowCutoffEditor.slider
        form.addWidget(self.lowCutoffLabel, 1, 0)
        form.addWidget(self.lowCutoffEditor, 1, 1)

        self.highCutoffLabel = QLabel("高截止", self)
        self.highCutoffEditor = SliderNumberEditor(
            self,
            minimum=self._minimum,
            maximum=self._maximum,
            value=high_cutoff,
            decimals=decimals,
            suffix=suffix,
        )
        self.highCutoffSpin = self.highCutoffEditor.spinBox
        self.highCutoffSlider = self.highCutoffEditor.slider
        form.addWidget(self.highCutoffLabel, 2, 0)
        form.addWidget(self.highCutoffEditor, 2, 1)

        self.orderLabel = QLabel("Butterworth 阶数", self)
        self.orderSpin = NoWheelSpinBox(self)
        self.orderSpin.setKeyboardTracking(False)
        self.orderSpin.setRange(1, 16)
        form.addWidget(self.orderLabel, 3, 0)
        form.addWidget(self.orderSpin, 3, 1)
        form.setColumnStretch(1, 1)
        root.addLayout(form)

        self.responseCanvas = _FrequencyResponseCanvas(self)
        root.addWidget(self.responseCanvas)
        self.validationLabel = QLabel(self)
        self.validationLabel.setObjectName("frequencyResponseValidation")
        self.validationLabel.setWordWrap(True)
        root.addWidget(self.validationLabel)

        step = max(
            10.0 ** -max(0, min(12, int(decimals))),
            (self._maximum - self._minimum) / 1000.0,
        )
        self.lowCutoffEditor.setSingleStep(step)
        self.highCutoffEditor.setSingleStep(step)

        self.modeCombo.currentIndexChanged.connect(
            self._mode_index_changed
        )
        self.lowCutoffEditor.valueChanged.connect(
            self._parameter_changed
        )
        self.highCutoffEditor.valueChanged.connect(
            self._parameter_changed
        )
        self.orderSpin.valueChanged.connect(self._parameter_changed)
        self.lowCutoffEditor.editFinished.connect(
            self._emit_interaction_finished
        )
        self.highCutoffEditor.editFinished.connect(
            self._emit_interaction_finished
        )
        self.orderSpin.editingFinished.connect(
            self._emit_interaction_finished
        )

        self.setValue(
            {
                "mode": mode,
                "low_cutoff": low_cutoff,
                "high_cutoff": high_cutoff,
                "order": order,
            },
            emit_signal=False,
        )

    def rawValue(self) -> dict[str, object]:  # noqa: N802 - Qt API
        return {
            "mode": self.mode(),
            "low_cutoff": self.lowCutoffEditor.value(),
            "high_cutoff": self.highCutoffEditor.value(),
            "order": int(self.orderSpin.value()),
        }

    def value(self) -> dict[str, object]:
        if not self._valid:
            raise ValueError(self._validation_message)
        return self.rawValue()

    def tryValue(self) -> dict[str, object] | None:  # noqa: N802
        return self.rawValue() if self._valid else None

    def setValue(  # noqa: N802 - Qt API
        self,
        value: Mapping[str, object],
        *,
        emit_signal: bool = True,
    ) -> None:
        if not isinstance(value, Mapping):
            raise TypeError("FFT 参数必须是映射")
        mode = str(value.get("mode", self.mode() or "lowpass")).strip().lower()
        supported_modes = {item for item, _label in self.MODES}
        if mode not in supported_modes:
            raise ValueError(f"不支持的 FFT 滤波模式：{mode}")
        low_cutoff = _require_finite(
            value.get("low_cutoff", self.lowCutoffEditor.value()),
            "低截止频率",
        )
        high_cutoff = _require_finite(
            value.get("high_cutoff", self.highCutoffEditor.value()),
            "高截止频率",
        )
        order_value = value.get("order", self.orderSpin.value())
        if isinstance(order_value, bool):
            raise ValueError("Butterworth 阶数必须是整数")
        try:
            order_float = float(order_value)
        except (TypeError, ValueError) as exc:
            raise ValueError("Butterworth 阶数必须是整数") from exc
        if not math.isfinite(order_float) or not order_float.is_integer():
            raise ValueError("Butterworth 阶数必须是整数")
        order = int(order_float)
        message = self._validate_values(
            mode,
            low_cutoff,
            high_cutoff,
            order,
        )
        if message:
            raise ValueError(message)

        previous = self.rawValue()
        self._updating = True
        try:
            mode_blocked = self.modeCombo.blockSignals(True)
            try:
                self.modeCombo.setCurrentIndex(self.modeCombo.findData(mode))
            finally:
                self.modeCombo.blockSignals(mode_blocked)
            self.lowCutoffEditor.setValue(low_cutoff, emit_signal=False)
            self.highCutoffEditor.setValue(high_cutoff, emit_signal=False)
            _set_widget_value_blocked(self.orderSpin, order)
        finally:
            self._updating = False
        self._update_visibility()
        self._refresh_state()
        if emit_signal and previous != self.rawValue():
            self.valueChanged.emit(self.rawValue())

    def setMode(  # noqa: N802 - Qt API
        self,
        mode: str,
        *,
        emit_signal: bool = True,
    ) -> None:
        updated = self.rawValue()
        updated["mode"] = str(mode)
        self.setValue(updated, emit_signal=emit_signal)

    def mode(self) -> str:
        return str(self.modeCombo.currentData() or "")

    def frequencyRange(self) -> tuple[float, float]:  # noqa: N802
        return self._minimum, self._maximum

    def setFrequencyRange(  # noqa: N802 - Qt API
        self,
        minimum: float,
        maximum: float,
    ) -> bool:
        """Update the displayed Nyquist range without clamping a recipe value.

        Returning ``False`` means one of the exact cutoffs is currently outside
        the requested scientific range.  The caller can then keep the old
        widget range long enough for the user to correct the value while the
        recipe validator presents the precise error.
        """

        normalized_minimum = _require_finite(minimum, "频率下限")
        normalized_maximum = _require_finite(maximum, "频率上限")
        if normalized_maximum <= normalized_minimum:
            raise ValueError("频率上限必须大于频率下限")
        raw = self.rawValue()
        low = float(raw["low_cutoff"])
        high = float(raw["high_cutoff"])
        if (
            low < normalized_minimum
            or low > normalized_maximum
            or high < normalized_minimum
            or high > normalized_maximum
        ):
            return False
        if (
            normalized_minimum == self._minimum
            and normalized_maximum == self._maximum
        ):
            return True
        self._minimum = normalized_minimum
        self._maximum = normalized_maximum
        self.lowCutoffEditor.setRange(
            normalized_minimum,
            normalized_maximum,
            emit_signal=False,
        )
        self.highCutoffEditor.setRange(
            normalized_minimum,
            normalized_maximum,
            emit_signal=False,
        )
        step = max(
            10.0 ** -max(0, self.lowCutoffSpin.decimals()),
            (normalized_maximum - normalized_minimum) / 1000.0,
        )
        self.lowCutoffEditor.setSingleStep(step)
        self.highCutoffEditor.setSingleStep(step)
        self._refresh_state()
        return True

    def isValid(self) -> bool:  # noqa: N802 - Qt API
        return self._valid

    def validationMessage(self) -> str:  # noqa: N802 - Qt API
        return self._validation_message

    def _validate_values(
        self,
        mode: str,
        low_cutoff: float,
        high_cutoff: float,
        order: int,
    ) -> str:
        for value, name in (
            (low_cutoff, "低截止频率"),
            (high_cutoff, "高截止频率"),
        ):
            if not math.isfinite(float(value)):
                return f"{name}必须是有限数"
            if value < self._minimum or value > self._maximum:
                return (
                    f"{name}必须在 {self._minimum:g} 到 "
                    f"{self._maximum:g} 之间"
                )
        if order < 1 or order > 16:
            return "Butterworth 阶数必须在 1 到 16 之间"
        if mode == "lowpass" and high_cutoff <= self._minimum:
            return "低通滤波的高截止频率必须大于频率下限"
        if mode == "highpass" and low_cutoff <= self._minimum:
            return "高通滤波的低截止频率必须大于频率下限"
        if mode in {"bandpass", "bandstop"} and high_cutoff <= low_cutoff:
            return "带通/带阻滤波的高截止频率必须大于低截止频率"
        return ""

    def _mode_index_changed(self, _index: int) -> None:
        if self._updating:
            return
        self._update_visibility()
        self._refresh_state(emit_value=True)
        self._emit_interaction_finished()

    def _parameter_changed(self, *_args) -> None:
        if self._updating:
            return
        self._refresh_state(emit_value=True)

    def _update_visibility(self) -> None:
        mode = self.mode()
        show_low = mode in {"highpass", "bandpass", "bandstop"}
        show_high = mode in {"lowpass", "bandpass", "bandstop"}
        self.lowCutoffLabel.setVisible(show_low)
        self.lowCutoffEditor.setVisible(show_low)
        self.highCutoffLabel.setVisible(show_high)
        self.highCutoffEditor.setVisible(show_high)

    def _refresh_state(self, *, emit_value: bool = False) -> None:
        raw = self.rawValue()
        mode = str(raw["mode"])
        low_cutoff = float(raw["low_cutoff"])
        high_cutoff = float(raw["high_cutoff"])
        order = int(raw["order"])
        message = self._validate_values(
            mode,
            low_cutoff,
            high_cutoff,
            order,
        )
        self._set_validation(not message, message or "频率范围有效")
        self.responseCanvas.setState(
            minimum=self._minimum,
            maximum=self._maximum,
            mode=mode,
            low_cutoff=low_cutoff,
            high_cutoff=high_cutoff,
            order=order,
        )
        if emit_value:
            self.valueChanged.emit(raw)

    def _set_validation(self, valid: bool, message: str) -> None:
        normalized_valid = bool(valid)
        normalized_message = str(message)
        state_changed = normalized_valid != self._valid
        message_changed = normalized_message != self._validation_message
        self._valid = normalized_valid
        self._validation_message = normalized_message
        self.validationLabel.setText(normalized_message)
        self.validationLabel.setProperty("valid", normalized_valid)
        style = self.validationLabel.style()
        style.unpolish(self.validationLabel)
        style.polish(self.validationLabel)
        if state_changed:
            self.validityChanged.emit(normalized_valid)
        if state_changed or message_changed:
            self.validationChanged.emit(
                normalized_valid,
                normalized_message,
            )

    def _emit_interaction_finished(self) -> None:
        self.editFinished.emit()
        self.interactionFinished.emit()


class _StripeFrequencyCanvas(QWidget):
    """Frequency-plane explanation for directional stripe suppression."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._direction = "horizontal"
        self._notch_width = 0.02
        self._protect_radius = 0.02
        self.setMinimumHeight(132)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,
        )
        self.setAccessibleName("条纹抑制频谱示意")
        self.setToolTip(
            "显示被抑制的方向频带和中心低频保护区；"
            "仅解释参数，不执行 FFT"
        )

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt API
        return QSize(228, 148)

    def setState(  # noqa: N802 - Qt API
        self,
        *,
        direction: str,
        notch_width: float,
        protect_radius: float,
    ) -> None:
        self._direction = str(direction)
        self._notch_width = float(notch_width)
        self._protect_radius = float(protect_radius)
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        palette = self.palette()
        painter.fillRect(self.rect(), palette.color(QPalette.ColorRole.Base))
        plot = QRectF(self.rect()).adjusted(12.0, 10.0, -12.0, -28.0)
        painter.setPen(QPen(palette.color(QPalette.ColorRole.Mid), 1.0))
        painter.drawRoundedRect(plot, 4.0, 4.0)
        if plot.width() <= 0.0 or plot.height() <= 0.0:
            return

        center = plot.center()
        painter.setPen(
            QPen(palette.color(QPalette.ColorRole.PlaceholderText), 1.0)
        )
        painter.drawLine(
            QPointF(plot.left(), center.y()),
            QPointF(plot.right(), center.y()),
        )
        painter.drawLine(
            QPointF(center.x(), plot.top()),
            QPointF(center.x(), plot.bottom()),
        )

        # Horizontal spatial stripes produce a vertical frequency axis, and
        # vertical stripes produce the horizontal counterpart.
        normalized_width = min(0.25, max(0.0, self._notch_width)) / 0.5
        band_color = QColor(palette.color(QPalette.ColorRole.Highlight))
        band_color.setAlpha(72)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(band_color)
        if self._direction == "horizontal":
            band = QRectF(
                center.x() - plot.width() * normalized_width / 2.0,
                plot.top(),
                plot.width() * normalized_width,
                plot.height(),
            )
        else:
            band = QRectF(
                plot.left(),
                center.y() - plot.height() * normalized_width / 2.0,
                plot.width(),
                plot.height() * normalized_width,
            )
        painter.drawRect(band)

        protect_fraction = min(
            0.25,
            max(0.0, self._protect_radius),
        ) / 0.5
        radius = (
            min(plot.width(), plot.height())
            * protect_fraction
            / 2.0
        )
        protect_color = QColor(palette.color(QPalette.ColorRole.Base))
        protect_color.setAlpha(235)
        painter.setBrush(protect_color)
        painter.setPen(
            QPen(palette.color(QPalette.ColorRole.Link), 1.5)
        )
        painter.drawEllipse(center, radius, radius)

        painter.setPen(
            QPen(palette.color(QPalette.ColorRole.Text), 1.0)
        )
        painter.drawText(
            QRectF(
                plot.left(),
                plot.bottom() + 4.0,
                plot.width(),
                20.0,
            ),
            Qt.AlignmentFlag.AlignCenter,
            (
                "水平条纹 → 抑制竖直频率轴"
                if self._direction == "horizontal"
                else "垂直条纹 → 抑制水平频率轴"
            ),
        )


class StripeSuppressionEditor(QWidget):
    """Directional notch controls with an explicit frequency-plane preview."""

    editFinished = Signal()
    interactionFinished = Signal()

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        direction: str = "horizontal",
        notch_width: float = 0.02,
        protect_radius: float = 0.02,
        decimals: int = 4,
    ) -> None:
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(7)
        form = QGridLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setHorizontalSpacing(8)
        form.setVerticalSpacing(5)

        self.directionLabel = QLabel("空间条纹方向", self)
        self.directionCombo = NoWheelComboBox(self)
        self.directionCombo.addItem("水平条纹", "horizontal")
        self.directionCombo.addItem("垂直条纹", "vertical")
        form.addWidget(self.directionLabel, 0, 0)
        form.addWidget(self.directionCombo, 0, 1)

        self.notchWidthLabel = QLabel("陷波宽度", self)
        self.notchWidthEditor = SliderNumberEditor(
            self,
            minimum=0.0001,
            maximum=0.25,
            value=notch_width,
            decimals=decimals,
            suffix=" 周期/像素",
        )
        self.notchWidthSpin = self.notchWidthEditor.spinBox
        form.addWidget(self.notchWidthLabel, 1, 0)
        form.addWidget(self.notchWidthEditor, 1, 1)

        self.protectRadiusLabel = QLabel("低频保护半径", self)
        self.protectRadiusEditor = SliderNumberEditor(
            self,
            minimum=0.0,
            maximum=0.25,
            value=protect_radius,
            decimals=decimals,
            suffix=" 周期/像素",
        )
        self.protectRadiusSpin = self.protectRadiusEditor.spinBox
        form.addWidget(self.protectRadiusLabel, 2, 0)
        form.addWidget(self.protectRadiusEditor, 2, 1)
        form.setColumnStretch(1, 1)
        root.addLayout(form)

        self.frequencyCanvas = _StripeFrequencyCanvas(self)
        root.addWidget(self.frequencyCanvas)
        self.explanationLabel = QLabel(
            "彩色区域为待抑制频带，中央圆为保留的低频区域；"
            "抑制强度在下方单独调整。",
            self,
        )
        self.explanationLabel.setWordWrap(True)
        self.explanationLabel.setObjectName(
            "stripeFrequencyExplanation"
        )
        root.addWidget(self.explanationLabel)

        selected = self.directionCombo.findData(str(direction))
        self.directionCombo.setCurrentIndex(max(0, selected))
        self.notchWidthEditor.setValue(
            notch_width,
            emit_signal=False,
        )
        self.protectRadiusEditor.setValue(
            protect_radius,
            emit_signal=False,
        )
        self.directionCombo.currentIndexChanged.connect(
            self._value_changed
        )
        self.notchWidthEditor.valueChanged.connect(
            self._value_changed
        )
        self.protectRadiusEditor.valueChanged.connect(
            self._value_changed
        )
        self.directionCombo.currentIndexChanged.connect(
            self._emit_finished
        )
        self.notchWidthEditor.editFinished.connect(
            self._emit_finished
        )
        self.protectRadiusEditor.editFinished.connect(
            self._emit_finished
        )
        self._refresh_canvas()

    def value(self) -> dict[str, object]:
        return {
            "direction": str(
                self.directionCombo.currentData() or "horizontal"
            ),
            "notch_width": float(
                self.notchWidthEditor.value()
            ),
            "protect_radius": float(
                self.protectRadiusEditor.value()
            ),
        }

    def _value_changed(self, *_args) -> None:
        self._refresh_canvas()

    def _refresh_canvas(self) -> None:
        values = self.value()
        self.frequencyCanvas.setState(
            direction=str(values["direction"]),
            notch_width=float(values["notch_width"]),
            protect_radius=float(values["protect_radius"]),
        )

    def _emit_finished(self, *_args) -> None:
        self._refresh_canvas()
        self.editFinished.emit()
        self.interactionFinished.emit()


class LinkedDimensionsEditor(QWidget):
    """Linked output-size editor for resize and canvas-size operations.

    Width and height are always stored as exact integer pixel dimensions.  The
    percentage editor is a convenience for applying one uniform scale to both
    source dimensions; it never replaces the exact width/height values.
    """

    valueChanged = Signal(int, int)
    editFinished = Signal()

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        source_width: int = 1,
        source_height: int = 1,
        width: int | None = None,
        height: int | None = None,
        lock_aspect: bool = True,
        aspect_lock_available: bool = True,
        maximum_dimension: int = 200_000,
    ) -> None:
        super().__init__(parent)
        self._source_width = 1
        self._source_height = 1
        self._width = 1
        self._height = 1
        self._maximum_dimension = self._normalize_maximum_dimension(
            maximum_dimension
        )
        self._aspect_lock_available = bool(aspect_lock_available)
        self._syncing = False

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(6)

        self.sourceSizeLabel = QLabel(self)
        self.sourceSizeLabel.setObjectName("linkedDimensionsSourceSize")
        self.sourceSizeLabel.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        root.addWidget(self.sourceSizeLabel)

        dimensions = QGridLayout()
        dimensions.setContentsMargins(0, 0, 0, 0)
        dimensions.setHorizontalSpacing(8)
        dimensions.setVerticalSpacing(5)

        self.widthLabel = QLabel("宽度", self)
        self.widthSpin = NoWheelSpinBox(self)
        self.widthSpin.setKeyboardTracking(False)
        self.widthSpin.setSuffix(" px")
        self.widthSpin.setAccessibleName("输出宽度")
        self.widthSpin.setRange(1, self._maximum_dimension)
        self.widthSpin.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        dimensions.addWidget(self.widthLabel, 0, 0)
        dimensions.addWidget(self.widthSpin, 0, 1)

        self.heightLabel = QLabel("高度", self)
        self.heightSpin = NoWheelSpinBox(self)
        self.heightSpin.setKeyboardTracking(False)
        self.heightSpin.setSuffix(" px")
        self.heightSpin.setAccessibleName("输出高度")
        self.heightSpin.setRange(1, self._maximum_dimension)
        self.heightSpin.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        dimensions.addWidget(self.heightLabel, 1, 0)
        dimensions.addWidget(self.heightSpin, 1, 1)
        dimensions.setColumnStretch(1, 1)
        root.addLayout(dimensions)

        controls = QGridLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setHorizontalSpacing(8)
        controls.setVerticalSpacing(5)
        self.lockAspectCheck = QCheckBox("锁定宽高比", self)
        self.lockAspectCheck.setAccessibleName("锁定源图片宽高比")
        self.lockAspectCheck.setToolTip(
            "开启后，修改宽度或高度会按源图片比例同步另一边"
        )
        self.percentLabel = QLabel("统一缩放", self)
        self.percentSpin = NoWheelDoubleSpinBox(self)
        self.percentSpin.setKeyboardTracking(False)
        self.percentSpin.setDecimals(4)
        self.percentSpin.setSuffix(" %")
        self.percentSpin.setSingleStep(1.0)
        self.percentSpin.setAccessibleName("相对源图片的统一缩放百分比")
        self.percentSpin.setToolTip(
            "修改后按同一百分比分别计算源图片的宽度和高度"
        )
        controls.addWidget(self.lockAspectCheck, 0, 0, 1, 2)
        controls.addWidget(self.percentLabel, 1, 0)
        controls.addWidget(self.percentSpin, 1, 1)
        controls.setColumnStretch(1, 1)
        root.addLayout(controls)

        self.outputSummaryLabel = QLabel(self)
        self.outputSummaryLabel.setObjectName("linkedDimensionsOutputSummary")
        self.outputSummaryLabel.setWordWrap(True)
        self.outputSummaryLabel.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        root.addWidget(self.outputSummaryLabel)

        self.widthSpin.valueChanged.connect(self._width_changed)
        self.heightSpin.valueChanged.connect(self._height_changed)
        self.percentSpin.valueChanged.connect(self._percent_changed)
        self.lockAspectCheck.toggled.connect(self._aspect_lock_changed)
        self.widthSpin.editingFinished.connect(self.editFinished.emit)
        self.heightSpin.editingFinished.connect(self.editFinished.emit)
        self.percentSpin.editingFinished.connect(self.editFinished.emit)

        self.setSourceSize(
            source_width,
            source_height,
            reset_value=True,
            emit_signal=False,
        )
        self.setAspectLockAvailable(self._aspect_lock_available)
        self.setAspectLocked(lock_aspect, emit_signal=False)
        self.setValue(
            source_width if width is None else width,
            source_height if height is None else height,
            emit_signal=False,
        )

    def sourceSize(self) -> tuple[int, int]:  # noqa: N802 - Qt API
        return self._source_width, self._source_height

    def setSourceSize(  # noqa: N802 - Qt API
        self,
        width: int,
        height: int,
        *,
        reset_value: bool = False,
        emit_signal: bool = True,
    ) -> None:
        normalized_width = self._normalize_source_dimension(width, "源宽度")
        normalized_height = self._normalize_source_dimension(height, "源高度")
        self._source_width = normalized_width
        self._source_height = normalized_height
        self._update_percent_range()
        if reset_value:
            self.setValue(
                normalized_width,
                normalized_height,
                emit_signal=emit_signal,
            )
            return
        self._sync_children()

    def value(self) -> tuple[int, int]:
        return self._width, self._height

    def setValue(  # noqa: N802 - Qt API
        self,
        width: int | tuple[int, int],
        height: int | None = None,
        *,
        emit_signal: bool = True,
    ) -> None:
        if height is None:
            if not isinstance(width, tuple) or len(width) != 2:
                raise TypeError("尺寸必须提供宽度和高度")
            width, height = width
        normalized_width = self._normalize_output_dimension(width, "输出宽度")
        normalized_height = self._normalize_output_dimension(
            height,
            "输出高度",
        )
        changed = (
            normalized_width != self._width
            or normalized_height != self._height
        )
        self._width = normalized_width
        self._height = normalized_height
        self._sync_children()
        if emit_signal and changed:
            self.valueChanged.emit(self._width, self._height)

    def setMaximumDimension(  # noqa: N802 - Qt API
        self,
        maximum: int,
        *,
        emit_signal: bool = True,
    ) -> None:
        normalized = self._normalize_maximum_dimension(maximum)
        if normalized == self._maximum_dimension:
            return
        self._maximum_dimension = normalized
        with self._synchronizing(self):
            self.widthSpin.setMaximum(normalized)
            self.heightSpin.setMaximum(normalized)
        self._update_percent_range()
        self.setValue(
            min(self._width, normalized),
            min(self._height, normalized),
            emit_signal=emit_signal,
        )

    def maximumDimension(self) -> int:  # noqa: N802 - Qt API
        return self._maximum_dimension

    def setAspectLocked(  # noqa: N802 - Qt API
        self,
        locked: bool,
        *,
        emit_signal: bool = True,
    ) -> None:
        normalized = bool(locked) and self._aspect_lock_available
        changed = normalized != self.lockAspectCheck.isChecked()
        with self._synchronizing(self):
            self.lockAspectCheck.setChecked(normalized)
        if not changed or not normalized:
            self._update_summary()
            return
        linked_height = self._height_for_width(self._width)
        self.setValue(
            self._width,
            linked_height,
            emit_signal=emit_signal,
        )

    def isAspectLocked(self) -> bool:  # noqa: N802 - Qt API
        return (
            self._aspect_lock_available
            and self.lockAspectCheck.isChecked()
        )

    def setAspectLockAvailable(  # noqa: N802 - Qt API
        self,
        available: bool,
    ) -> None:
        self._aspect_lock_available = bool(available)
        with self._synchronizing(self):
            if not self._aspect_lock_available:
                self.lockAspectCheck.setChecked(False)
            self.lockAspectCheck.setEnabled(self._aspect_lock_available)
            self.lockAspectCheck.setVisible(self._aspect_lock_available)
        self._update_summary()

    def isAspectLockAvailable(self) -> bool:  # noqa: N802 - Qt API
        return self._aspect_lock_available

    def _width_changed(self, value: int) -> None:
        if self._syncing:
            return
        width = int(value)
        height = self._height_for_width(width) if self.isAspectLocked() else self._height
        self._apply_user_value(width, height)

    def _height_changed(self, value: int) -> None:
        if self._syncing:
            return
        height = int(value)
        width = self._width_for_height(height) if self.isAspectLocked() else self._width
        self._apply_user_value(width, height)

    def _percent_changed(self, value: float) -> None:
        if self._syncing:
            return
        scale = float(value) / 100.0
        width = self._clamp_output(round(self._source_width * scale))
        height = self._clamp_output(round(self._source_height * scale))
        self._apply_user_value(width, height)

    def _aspect_lock_changed(self, locked: bool) -> None:
        if self._syncing:
            return
        if not self._aspect_lock_available:
            self.setAspectLocked(False, emit_signal=False)
            return
        if bool(locked):
            self._apply_user_value(
                self._width,
                self._height_for_width(self._width),
            )
        else:
            self._update_summary()
        self.editFinished.emit()

    def _apply_user_value(self, width: int, height: int) -> None:
        normalized_width = self._clamp_output(width)
        normalized_height = self._clamp_output(height)
        changed = (
            normalized_width != self._width
            or normalized_height != self._height
        )
        self._width = normalized_width
        self._height = normalized_height
        self._sync_children()
        if changed:
            self.valueChanged.emit(self._width, self._height)

    def _sync_children(self) -> None:
        with self._synchronizing(self):
            self.widthSpin.setValue(self._width)
            self.heightSpin.setValue(self._height)
            self.percentSpin.setValue(
                self._width * 100.0 / self._source_width
            )
        self._update_labels()

    def _update_labels(self) -> None:
        self.sourceSizeLabel.setText(
            f"源尺寸：{self._source_width:,} × "
            f"{self._source_height:,} px"
        )
        self._update_summary()

    def _update_summary(self) -> None:
        pixel_count = self._width * self._height
        width_percent = self._width * 100.0 / self._source_width
        height_percent = self._height * 100.0 / self._source_height
        scale_text = (
            f"等比 {width_percent:.2f}%"
            if math.isclose(width_percent, height_percent, abs_tol=0.01)
            else f"宽 {width_percent:.2f}% / 高 {height_percent:.2f}%"
        )
        self.outputSummaryLabel.setText(
            f"输出：{self._width:,} × {self._height:,} px"
            f" · {pixel_count:,} 像素 · {scale_text}"
        )

    def _update_percent_range(self) -> None:
        maximum_percent = min(
            self._maximum_dimension * 100.0 / self._source_width,
            self._maximum_dimension * 100.0 / self._source_height,
        )
        minimum_percent = max(
            0.0001,
            100.0 / self._source_width,
            100.0 / self._source_height,
        )
        if maximum_percent < minimum_percent:
            maximum_percent = minimum_percent
        with self._synchronizing(self):
            self.percentSpin.setRange(minimum_percent, maximum_percent)

    def _height_for_width(self, width: int) -> int:
        return self._clamp_output(
            round(int(width) * self._source_height / self._source_width)
        )

    def _width_for_height(self, height: int) -> int:
        return self._clamp_output(
            round(int(height) * self._source_width / self._source_height)
        )

    def _clamp_output(self, value: int | float) -> int:
        return min(self._maximum_dimension, max(1, int(value)))

    def _normalize_output_dimension(self, value: object, name: str) -> int:
        try:
            normalized = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name}必须是整数") from exc
        if normalized < 1 or normalized > self._maximum_dimension:
            raise ValueError(
                f"{name}必须在 1 到 {self._maximum_dimension} 之间"
            )
        return normalized

    @staticmethod
    def _normalize_source_dimension(value: object, name: str) -> int:
        try:
            normalized = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name}必须是正整数") from exc
        if normalized < 1:
            raise ValueError(f"{name}必须是正整数")
        return normalized

    @staticmethod
    def _normalize_maximum_dimension(value: object) -> int:
        try:
            normalized = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("最大尺寸必须是正整数") from exc
        if normalized < 1:
            raise ValueError("最大尺寸必须是正整数")
        return normalized

    class _synchronizing:
        """Small context manager that blocks recursive slot handling."""

        def __init__(self, owner: LinkedDimensionsEditor) -> None:
            self._owner = owner
            self._previous = False

        def __enter__(self) -> None:
            self._previous = self._owner._syncing
            self._owner._syncing = True

        def __exit__(self, exc_type, exc_value, traceback) -> None:
            self._owner._syncing = self._previous


class CropBoundsEditor(QWidget):
    """Source-bounded crop rectangle editor with exact pixel coordinates."""

    valueChanged = Signal(int, int, int, int)
    editFinished = Signal()

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        source_width: int,
        source_height: int,
        x: int = 0,
        y: int = 0,
        width: int | None = None,
        height: int | None = None,
    ) -> None:
        super().__init__(parent)
        self._source_width = max(1, int(source_width))
        self._source_height = max(1, int(source_height))
        self._syncing = False
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(6)
        self.sourceSizeLabel = QLabel(
            f"当前步骤输入：{self._source_width:,} × "
            f"{self._source_height:,} px",
            self,
        )
        self.sourceSizeLabel.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        root.addWidget(self.sourceSizeLabel)

        form = QGridLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setHorizontalSpacing(8)
        form.setVerticalSpacing(5)
        self.xSpin = NoWheelSpinBox(self)
        self.ySpin = NoWheelSpinBox(self)
        self.widthSpin = NoWheelSpinBox(self)
        self.heightSpin = NoWheelSpinBox(self)
        for widget in (
            self.xSpin,
            self.ySpin,
            self.widthSpin,
            self.heightSpin,
        ):
            widget.setKeyboardTracking(False)
            widget.setSuffix(" px")
            widget.setSizePolicy(
                QSizePolicy.Policy.Expanding,
                QSizePolicy.Policy.Fixed,
            )
        form.addWidget(QLabel("左（X）", self), 0, 0)
        form.addWidget(self.xSpin, 0, 1)
        form.addWidget(QLabel("上（Y）", self), 1, 0)
        form.addWidget(self.ySpin, 1, 1)
        form.addWidget(QLabel("宽度", self), 2, 0)
        form.addWidget(self.widthSpin, 2, 1)
        form.addWidget(QLabel("高度", self), 3, 0)
        form.addWidget(self.heightSpin, 3, 1)
        form.setColumnStretch(1, 1)
        root.addLayout(form)

        self.summaryLabel = QLabel(self)
        self.summaryLabel.setObjectName("cropBoundsSummary")
        self.summaryLabel.setWordWrap(True)
        self.summaryLabel.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        root.addWidget(self.summaryLabel)
        self.fullImageButton = QPushButton("使用整幅图片", self)
        self.fullImageButton.setToolTip(
            "把裁剪矩形恢复为当前步骤输入的完整像素范围"
        )
        root.addWidget(self.fullImageButton)

        self.xSpin.valueChanged.connect(self._origin_changed)
        self.ySpin.valueChanged.connect(self._origin_changed)
        self.widthSpin.valueChanged.connect(self._size_changed)
        self.heightSpin.valueChanged.connect(self._size_changed)
        for widget in (
            self.xSpin,
            self.ySpin,
            self.widthSpin,
            self.heightSpin,
        ):
            widget.editingFinished.connect(self.editFinished.emit)
        self.fullImageButton.clicked.connect(self._use_full_image)
        self.setValue(
            x,
            y,
            self._source_width if width is None else width,
            self._source_height if height is None else height,
            emit_signal=False,
        )

    def value(self) -> tuple[int, int, int, int]:
        return (
            int(self.xSpin.value()),
            int(self.ySpin.value()),
            int(self.widthSpin.value()),
            int(self.heightSpin.value()),
        )

    def setValue(  # noqa: N802 - Qt API
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        *,
        emit_signal: bool = True,
    ) -> None:
        normalized = tuple(int(value) for value in (x, y, width, height))
        nx, ny, nw, nh = normalized
        if nx < 0 or ny < 0 or nw < 1 or nh < 1:
            raise ValueError("裁剪坐标不能为负，宽度和高度必须为正数")
        if (
            nx + nw > self._source_width
            or ny + nh > self._source_height
        ):
            raise ValueError("裁剪矩形必须完全位于当前步骤输入范围内")
        previous = self.value()
        self._syncing = True
        try:
            self.xSpin.setRange(0, self._source_width - 1)
            self.ySpin.setRange(0, self._source_height - 1)
            self.xSpin.setValue(nx)
            self.ySpin.setValue(ny)
            self.widthSpin.setRange(1, self._source_width - nx)
            self.heightSpin.setRange(1, self._source_height - ny)
            self.widthSpin.setValue(nw)
            self.heightSpin.setValue(nh)
        finally:
            self._syncing = False
        self._update_summary()
        if emit_signal and self.value() != previous:
            self.valueChanged.emit(*self.value())

    def _origin_changed(self, _value: int) -> None:
        if self._syncing:
            return
        x, y, width, height = self.value()
        width = min(width, self._source_width - x)
        height = min(height, self._source_height - y)
        self.setValue(
            x,
            y,
            width,
            height,
            emit_signal=False,
        )
        self.valueChanged.emit(*self.value())

    def _size_changed(self, _value: int) -> None:
        if self._syncing:
            return
        self._update_summary()
        self.valueChanged.emit(*self.value())

    def _use_full_image(self) -> None:
        self.setValue(
            0,
            0,
            self._source_width,
            self._source_height,
        )
        self.editFinished.emit()

    def _update_summary(self) -> None:
        x, y, width, height = self.value()
        right = x + width
        bottom = y + height
        coverage = (
            width
            * height
            * 100.0
            / (self._source_width * self._source_height)
        )
        self.summaryLabel.setText(
            f"范围：X {x:,}–{right - 1:,} · "
            f"Y {y:,}–{bottom - 1:,} · "
            f"{width * height:,} 像素（{coverage:.2f}%）"
        )


class _StructuringElementCanvas(QWidget):
    """Palette-aware preview of a discrete morphology kernel."""

    _PADDING = 8.0

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._radius = 1
        self._shape = "ellipse"
        self.setMinimumSize(76, 76)
        self.setSizePolicy(
            QSizePolicy.Policy.Fixed,
            QSizePolicy.Policy.Fixed,
        )
        self.setAccessibleName("结构元素预览")
        self.setToolTip("显示当前结构元素形状及实际核尺寸")

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt API
        return QSize(92, 92)

    def setState(self, *, radius: int, shape: str) -> None:  # noqa: N802
        normalized_radius = max(1, int(radius))
        normalized_shape = str(shape)
        if (
            normalized_radius == self._radius
            and normalized_shape == self._shape
        ):
            return
        self._radius = normalized_radius
        self._shape = normalized_shape
        self.update()

    def kernelSize(self) -> int:  # noqa: N802 - Qt API
        return self._radius * 2 + 1

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        palette = self.palette()
        bounds = QRectF(self.rect()).adjusted(
            self._PADDING,
            self._PADDING,
            -self._PADDING,
            -self._PADDING,
        )
        painter.fillRect(self.rect(), palette.color(QPalette.ColorRole.Base))
        painter.setPen(QPen(palette.color(QPalette.ColorRole.Mid), 1.0))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(bounds, 4.0, 4.0)

        fill = QColor(palette.color(QPalette.ColorRole.Highlight))
        fill.setAlpha(175)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(fill)
        inset = max(5.0, min(bounds.width(), bounds.height()) * 0.12)
        kernel_bounds = bounds.adjusted(inset, inset, -inset, -inset)
        if self._shape == "rectangle":
            painter.drawRect(kernel_bounds)
        elif self._shape == "cross":
            arm = max(3.0, min(kernel_bounds.width(), kernel_bounds.height()) / 5.0)
            center = kernel_bounds.center()
            painter.drawRect(
                QRectF(
                    center.x() - arm / 2.0,
                    kernel_bounds.top(),
                    arm,
                    kernel_bounds.height(),
                )
            )
            painter.drawRect(
                QRectF(
                    kernel_bounds.left(),
                    center.y() - arm / 2.0,
                    kernel_bounds.width(),
                    arm,
                )
            )
        else:
            painter.drawEllipse(kernel_bounds)

        size = self.kernelSize()
        size_text = f"{size} × {size}"
        text_rect = QRectF(
            bounds.left(),
            bounds.bottom() - 24.0,
            bounds.width(),
            20.0,
        )
        painter.setPen(palette.color(QPalette.ColorRole.Text))
        painter.drawText(
            text_rect,
            Qt.AlignmentFlag.AlignCenter,
            size_text,
        )


class StructuringElementEditor(QWidget):
    """Compact editor for morphology radius, passes and kernel shape."""

    valueChanged = Signal(object)
    editFinished = Signal()

    SHAPES: tuple[tuple[str, str], ...] = (
        ("ellipse", "椭圆"),
        ("rectangle", "矩形"),
        ("cross", "十字"),
    )

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        radius: int = 1,
        iterations: int = 1,
        shape: str = "ellipse",
        maximum_radius: int = 255,
        maximum_iterations: int = 100,
    ) -> None:
        super().__init__(parent)
        self._maximum_radius = max(1, int(maximum_radius))
        self._maximum_iterations = max(1, int(maximum_iterations))
        self._syncing = False

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(8)

        controls = QGridLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setHorizontalSpacing(8)
        controls.setVerticalSpacing(6)

        self.radiusLabel = QLabel("半径", self)
        self.radiusSpin = NoWheelSpinBox(self)
        self.radiusSpin.setKeyboardTracking(False)
        self.radiusSpin.setRange(1, self._maximum_radius)
        self.radiusSpin.setSuffix(" px")
        self.radiusSpin.setAccessibleName("结构元素半径")
        self.radiusSpin.setToolTip(
            "核尺寸按 2 × 半径 + 1 计算；半径越大，邻域范围越大"
        )
        controls.addWidget(self.radiusLabel, 0, 0)
        controls.addWidget(self.radiusSpin, 0, 1)

        self.iterationsLabel = QLabel("迭代次数", self)
        self.iterationsSpin = NoWheelSpinBox(self)
        self.iterationsSpin.setKeyboardTracking(False)
        self.iterationsSpin.setRange(1, self._maximum_iterations)
        self.iterationsSpin.setAccessibleName("形态学迭代次数")
        self.iterationsSpin.setToolTip(
            "重复执行同一形态学操作；增加迭代会扩大处理效果"
        )
        controls.addWidget(self.iterationsLabel, 1, 0)
        controls.addWidget(self.iterationsSpin, 1, 1)

        self.shapeLabel = QLabel("结构元素", self)
        self.shapeCombo = NoWheelComboBox(self)
        self.shapeCombo.setAccessibleName("结构元素形状")
        self.shapeCombo.setToolTip("选择椭圆、矩形或十字形结构元素")
        for value, label in self.SHAPES:
            self.shapeCombo.addItem(label, value)
        controls.addWidget(self.shapeLabel, 2, 0)
        controls.addWidget(self.shapeCombo, 2, 1)
        controls.setColumnStretch(1, 1)
        root.addLayout(controls)

        self.preview = _StructuringElementCanvas(self)
        root.addWidget(
            self.preview,
            0,
            Qt.AlignmentFlag.AlignHCenter,
        )

        self.radiusSpin.valueChanged.connect(self._child_value_changed)
        self.iterationsSpin.valueChanged.connect(self._child_value_changed)
        self.shapeCombo.currentIndexChanged.connect(self._child_value_changed)
        self.radiusSpin.editingFinished.connect(self.editFinished.emit)
        self.iterationsSpin.editingFinished.connect(self.editFinished.emit)
        self.shapeCombo.activated.connect(
            lambda _index: self.editFinished.emit()
        )

        self.setValue(
            {
                "radius": radius,
                "iterations": iterations,
                "kernel": shape,
            },
            emit_signal=False,
        )

    def value(self) -> dict[str, int | str]:
        return {
            "radius": int(self.radiusSpin.value()),
            "iterations": int(self.iterationsSpin.value()),
            "kernel": str(self.shapeCombo.currentData()),
        }

    def setValue(  # noqa: N802 - Qt API
        self,
        value: Mapping[str, object],
        *,
        emit_signal: bool = True,
    ) -> None:
        if not isinstance(value, Mapping):
            raise ValueError("结构元素参数必须是映射")
        try:
            radius = int(value.get("radius", self.radiusSpin.value()))
            iterations = int(
                value.get("iterations", self.iterationsSpin.value())
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("半径和迭代次数必须是整数") from exc
        shape = str(
            value.get(
                "kernel",
                value.get("shape", self.shapeCombo.currentData()),
            )
        )
        supported_shapes = {item[0] for item in self.SHAPES}
        if radius < 1 or radius > self._maximum_radius:
            raise ValueError(
                f"半径必须在 1–{self._maximum_radius} 之间"
            )
        if iterations < 1 or iterations > self._maximum_iterations:
            raise ValueError(
                f"迭代次数必须在 1–{self._maximum_iterations} 之间"
            )
        if shape not in supported_shapes:
            raise ValueError(f"不支持的结构元素：{shape}")

        previous = self.value()
        self._syncing = True
        try:
            _set_widget_value_blocked(self.radiusSpin, radius)
            _set_widget_value_blocked(self.iterationsSpin, iterations)
            blocked = self.shapeCombo.blockSignals(True)
            try:
                self.shapeCombo.setCurrentIndex(
                    self.shapeCombo.findData(shape)
                )
            finally:
                self.shapeCombo.blockSignals(blocked)
        finally:
            self._syncing = False
        self._sync_preview()
        if emit_signal and self.value() != previous:
            self.valueChanged.emit(self.value())

    def radius(self) -> int:
        return int(self.radiusSpin.value())

    def iterations(self) -> int:
        return int(self.iterationsSpin.value())

    def shape(self) -> str:
        return str(self.shapeCombo.currentData())

    def kernelSize(self) -> tuple[int, int]:  # noqa: N802 - Qt API
        size = self.radius() * 2 + 1
        return size, size

    def _child_value_changed(self, *_args) -> None:
        if self._syncing:
            return
        self._sync_preview()
        self.valueChanged.emit(self.value())

    def _sync_preview(self) -> None:
        self.preview.setState(
            radius=self.radius(),
            shape=self.shape(),
        )


class AnchorGridEditor(QWidget):
    """Compact, keyboard-accessible 3×3 canvas-anchor selector."""

    valueChanged = Signal(str)
    anchorChanged = Signal(str)

    ANCHORS: tuple[str, ...] = (
        "top_left",
        "top_center",
        "top_right",
        "center_left",
        "center",
        "center_right",
        "bottom_left",
        "bottom_center",
        "bottom_right",
    )
    _BUTTON_TEXT = ("↖", "↑", "↗", "←", "•", "→", "↙", "↓", "↘")
    _ACCESSIBLE_NAMES = (
        "左上",
        "上中",
        "右上",
        "左中",
        "居中",
        "右中",
        "左下",
        "下中",
        "右下",
    )

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        value: str = "center",
    ) -> None:
        super().__init__(parent)
        self._value = "center"
        self.buttonGroup = QButtonGroup(self)
        self.buttonGroup.setExclusive(True)
        self.buttons: dict[str, QToolButton] = {}

        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setHorizontalSpacing(3)
        layout.setVerticalSpacing(3)
        for index, anchor in enumerate(self.ANCHORS):
            button = QToolButton(self)
            button.setText(self._BUTTON_TEXT[index])
            button.setCheckable(True)
            button.setAutoRaise(False)
            button.setAccessibleName(f"画布锚点：{self._ACCESSIBLE_NAMES[index]}")
            button.setToolTip(self._ACCESSIBLE_NAMES[index])
            button.setProperty("anchorValue", anchor)
            button.setSizePolicy(
                QSizePolicy.Policy.Expanding,
                QSizePolicy.Policy.Preferred,
            )
            button.setMinimumSize(30, 28)
            button.setStyleSheet(
                "QToolButton:checked {"
                " background: palette(highlight);"
                " color: palette(highlighted-text);"
                " border-color: palette(highlight);"
                "}"
            )
            self.buttonGroup.addButton(button, index)
            self.buttons[anchor] = button
            layout.addWidget(button, index // 3, index % 3)
        self.buttonGroup.idClicked.connect(self._button_clicked)
        self.setValue(value, emit_signal=False)

    def value(self) -> str:
        return self._value

    def anchor(self) -> str:
        return self.value()

    def setValue(  # noqa: N802 - Qt API
        self,
        value: str,
        *,
        emit_signal: bool = True,
    ) -> None:
        normalized = str(value)
        if normalized not in self.buttons:
            raise ValueError(f"不支持的锚点：{normalized}")
        changed = normalized != self._value
        self._value = normalized
        button = self.buttons[normalized]
        blocked = self.buttonGroup.blockSignals(True)
        try:
            button.setChecked(True)
        finally:
            self.buttonGroup.blockSignals(blocked)
        if emit_signal and changed:
            self.valueChanged.emit(normalized)
            self.anchorChanged.emit(normalized)

    def setAnchor(  # noqa: N802 - Qt API
        self,
        value: str,
        *,
        emit_signal: bool = True,
    ) -> None:
        self.setValue(value, emit_signal=emit_signal)

    def _button_clicked(self, button_id: int) -> None:
        self.setValue(self.ANCHORS[int(button_id)])


class KernelMatrixEditor(QWidget):
    """Structured finite-number convolution-kernel editor."""

    kernelChanged = Signal(object)
    dimensionsChanged = Signal(int, int)
    validityChanged = Signal(bool)
    validationChanged = Signal(bool, str)

    PRESETS: dict[str, tuple[tuple[float, ...], ...]] = {
        "identity": (
            (0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0),
        ),
        "sharpen": (
            (0.0, -1.0, 0.0),
            (-1.0, 5.0, -1.0),
            (0.0, -1.0, 0.0),
        ),
        "sobel_x": (
            (-1.0, 0.0, 1.0),
            (-2.0, 0.0, 2.0),
            (-1.0, 0.0, 1.0),
        ),
        "sobel_y": (
            (-1.0, -2.0, -1.0),
            (0.0, 0.0, 0.0),
            (1.0, 2.0, 1.0),
        ),
        "laplacian": (
            (0.0, 1.0, 0.0),
            (1.0, -4.0, 1.0),
            (0.0, 1.0, 0.0),
        ),
    }
    _PRESET_LABELS: tuple[tuple[str, str], ...] = (
        ("identity", "Identity（恒等）"),
        ("sharpen", "Sharpen（锐化）"),
        ("sobel_x", "Sobel X"),
        ("sobel_y", "Sobel Y"),
        ("laplacian", "Laplacian"),
    )

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        kernel: Sequence[Sequence[float]] | None = None,
        maximum_dimension: int = 31,
    ) -> None:
        super().__init__(parent)
        self._maximum_dimension = max(3, int(maximum_dimension))
        self._valid = True
        self._validation_message = ""

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(6)

        controls = QGridLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setHorizontalSpacing(6)
        controls.setVerticalSpacing(5)
        controls.addWidget(QLabel("宽", self), 0, 0)
        self.widthSpin = NoWheelSpinBox(self)
        self.widthSpin.setRange(1, self._maximum_dimension)
        self.widthSpin.setSingleStep(2)
        controls.addWidget(self.widthSpin, 0, 1)
        controls.addWidget(QLabel("高", self), 1, 0)
        self.heightSpin = NoWheelSpinBox(self)
        self.heightSpin.setRange(1, self._maximum_dimension)
        self.heightSpin.setSingleStep(2)
        controls.addWidget(self.heightSpin, 1, 1)
        self.presetCombo = NoWheelComboBox(self)
        for preset_id, label in self._PRESET_LABELS:
            self.presetCombo.addItem(label, preset_id)
        self.applyPresetButton = QPushButton("应用预设", self)
        controls.addWidget(self.presetCombo, 2, 0, 1, 2)
        controls.addWidget(self.applyPresetButton, 3, 0, 1, 2)
        controls.setColumnStretch(1, 1)
        root.addLayout(controls)

        self.table = QTableWidget(self)
        self.table.setAlternatingRowColors(True)
        self.table.horizontalHeader().setVisible(False)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.table.verticalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.table.setMinimumHeight(126)
        self.table.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        root.addWidget(self.table, 1)
        self.validationLabel = QLabel(self)
        self.validationLabel.setWordWrap(True)
        self.validationLabel.setObjectName("kernelValidationLabel")
        root.addWidget(self.validationLabel)

        self.widthSpin.valueChanged.connect(self._dimensions_edited)
        self.heightSpin.valueChanged.connect(self._dimensions_edited)
        self.applyPresetButton.clicked.connect(self._apply_selected_preset)
        self.table.itemChanged.connect(self._cell_changed)

        initial = kernel if kernel is not None else self.PRESETS["identity"]
        self.setKernel(initial, emit_signal=False)

    def setDimensions(  # noqa: N802 - Qt API
        self,
        width: int,
        height: int,
        *,
        preserve: bool = True,
        emit_signal: bool = True,
    ) -> None:
        normalized_width = int(width)
        normalized_height = int(height)
        if not 1 <= normalized_width <= self._maximum_dimension:
            raise ValueError("卷积核宽度超出允许范围")
        if not 1 <= normalized_height <= self._maximum_dimension:
            raise ValueError("卷积核高度超出允许范围")
        if normalized_width % 2 == 0 or normalized_height % 2 == 0:
            raise ValueError("卷积核宽度和高度必须为正奇数")
        old_width = self.table.columnCount()
        old_height = self.table.rowCount()
        if old_width == normalized_width and old_height == normalized_height:
            return
        previous = self._best_effort_matrix() if preserve else ()
        self._rebuild_table(
            normalized_width,
            normalized_height,
            previous=previous,
        )
        _set_widget_value_blocked(self.widthSpin, normalized_width)
        _set_widget_value_blocked(self.heightSpin, normalized_height)
        self._validate_and_emit(emit_kernel=True)
        if emit_signal:
            self.dimensionsChanged.emit(normalized_width, normalized_height)

    def dimensions(self) -> tuple[int, int]:
        return self.table.columnCount(), self.table.rowCount()

    def setKernel(  # noqa: N802 - Qt API
        self,
        kernel: Sequence[Sequence[float]],
        *,
        emit_signal: bool = True,
    ) -> None:
        normalized = self._normalize_kernel(kernel)
        height = len(normalized)
        width = len(normalized[0])
        self._rebuild_table(width, height, previous=normalized)
        _set_widget_value_blocked(self.widthSpin, width)
        _set_widget_value_blocked(self.heightSpin, height)
        self._set_validation(
            True,
            self._valid_kernel_summary(normalized),
        )
        if emit_signal:
            self.dimensionsChanged.emit(width, height)
            self.kernelChanged.emit(normalized)

    def kernel(self) -> tuple[tuple[float, ...], ...]:
        parsed, message = self._parse_table()
        if parsed is None:
            raise ValueError(message)
        return parsed

    def tryKernel(self) -> tuple[tuple[float, ...], ...] | None:  # noqa: N802
        parsed, _message = self._parse_table()
        return parsed

    def isValid(self) -> bool:  # noqa: N802 - Qt API
        return self._valid

    def validationMessage(self) -> str:  # noqa: N802 - Qt API
        return self._validation_message

    def applyPreset(self, preset: str) -> None:  # noqa: N802 - Qt API
        preset_id = str(preset)
        try:
            kernel = self.PRESETS[preset_id]
        except KeyError as exc:
            raise ValueError(f"未知卷积核预设：{preset_id}") from exc
        self.setKernel(kernel)
        index = self.presetCombo.findData(preset_id)
        if index >= 0:
            self.presetCombo.setCurrentIndex(index)

    def _apply_selected_preset(self) -> None:
        self.applyPreset(str(self.presetCombo.currentData()))

    def _dimensions_edited(self, _value: int) -> None:
        try:
            self.setDimensions(
                self.widthSpin.value(),
                self.heightSpin.value(),
                preserve=True,
            )
        except ValueError as exc:
            self._set_validation(False, str(exc))

    def _cell_changed(self, _item: QTableWidgetItem) -> None:
        self._validate_and_emit(emit_kernel=True)

    def _rebuild_table(
        self,
        width: int,
        height: int,
        *,
        previous: Sequence[Sequence[float]],
    ) -> None:
        blocked = self.table.blockSignals(True)
        try:
            self.table.clear()
            self.table.setRowCount(height)
            self.table.setColumnCount(width)
            for row in range(height):
                for column in range(width):
                    item = QTableWidgetItem("0")
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    self.table.setItem(row, column, item)

            previous_height = len(previous)
            previous_width = len(previous[0]) if previous_height else 0
            source_row = max(0, (previous_height - height) // 2)
            source_column = max(0, (previous_width - width) // 2)
            target_row = max(0, (height - previous_height) // 2)
            target_column = max(0, (width - previous_width) // 2)
            copied_height = min(previous_height, height)
            copied_width = min(previous_width, width)
            for row in range(copied_height):
                for column in range(copied_width):
                    value = float(
                        previous[source_row + row][source_column + column]
                    )
                    self.table.item(
                        target_row + row,
                        target_column + column,
                    ).setText(self._format_number(value))
        finally:
            self.table.blockSignals(blocked)

    def _best_effort_matrix(self) -> tuple[tuple[float, ...], ...]:
        rows: list[tuple[float, ...]] = []
        for row in range(self.table.rowCount()):
            values: list[float] = []
            for column in range(self.table.columnCount()):
                item = self.table.item(row, column)
                try:
                    value = float(item.text()) if item is not None else 0.0
                except (TypeError, ValueError):
                    value = 0.0
                values.append(value if math.isfinite(value) else 0.0)
            rows.append(tuple(values))
        return tuple(rows)

    def _parse_table(
        self,
    ) -> tuple[tuple[tuple[float, ...], ...] | None, str]:
        if self.table.rowCount() <= 0 or self.table.columnCount() <= 0:
            return None, "卷积核不能为空"
        rows: list[tuple[float, ...]] = []
        for row in range(self.table.rowCount()):
            values: list[float] = []
            for column in range(self.table.columnCount()):
                item = self.table.item(row, column)
                text = "" if item is None else item.text().strip()
                try:
                    value = float(text)
                except (TypeError, ValueError):
                    return (
                        None,
                        f"第 {row + 1} 行、第 {column + 1} 列不是有效数字",
                    )
                if not math.isfinite(value):
                    return (
                        None,
                        f"第 {row + 1} 行、第 {column + 1} 列必须是有限数",
                    )
                values.append(value)
            rows.append(tuple(values))
        return tuple(rows), "卷积核有效"

    def _validate_and_emit(self, *, emit_kernel: bool) -> None:
        parsed, message = self._parse_table()
        self._set_validation(
            parsed is not None,
            (
                self._valid_kernel_summary(parsed)
                if parsed is not None
                else message
            ),
        )
        if parsed is not None and emit_kernel:
            self.kernelChanged.emit(parsed)

    @staticmethod
    def _valid_kernel_summary(
        kernel: tuple[tuple[float, ...], ...],
    ) -> str:
        height = len(kernel)
        width = len(kernel[0])
        coefficient_sum = math.fsum(
            value for row in kernel for value in row
        )
        zero_note = (
            " · 零和核（不能启用归一化）"
            if math.isclose(coefficient_sum, 0.0, abs_tol=1e-12)
            else ""
        )
        return (
            f"卷积核有效 · {width}×{height} · "
            f"系数和 {coefficient_sum:.12g}{zero_note}"
        )

    def _set_validation(self, valid: bool, message: str) -> None:
        normalized_valid = bool(valid)
        normalized_message = str(message)
        state_changed = normalized_valid != self._valid
        message_changed = normalized_message != self._validation_message
        self._valid = normalized_valid
        self._validation_message = normalized_message
        self.validationLabel.setText(normalized_message)
        self.validationLabel.setProperty("valid", normalized_valid)
        style = self.validationLabel.style()
        style.unpolish(self.validationLabel)
        style.polish(self.validationLabel)
        if state_changed:
            self.validityChanged.emit(normalized_valid)
        if state_changed or message_changed:
            self.validationChanged.emit(
                normalized_valid,
                normalized_message,
            )

    def _normalize_kernel(
        self,
        kernel: Sequence[Sequence[float]],
    ) -> tuple[tuple[float, ...], ...]:
        rows = tuple(tuple(row) for row in kernel)
        if not rows or not rows[0]:
            raise ValueError("卷积核不能为空")
        width = len(rows[0])
        if len(rows) > self._maximum_dimension or width > self._maximum_dimension:
            raise ValueError("卷积核尺寸超出允许范围")
        if any(len(row) != width for row in rows):
            raise ValueError("卷积核每一行必须具有相同宽度")
        normalized: list[tuple[float, ...]] = []
        for row in rows:
            normalized.append(
                tuple(_require_finite(value, "卷积核系数") for value in row)
            )
        if len(rows) % 2 == 0 or width % 2 == 0:
            raise ValueError("卷积核宽度和高度必须为正奇数")
        return tuple(normalized)

    @staticmethod
    def _format_number(value: float) -> str:
        return f"{float(value):.12g}"


__all__ = [
    "AnchorGridEditor",
    "CropBoundsEditor",
    "FrequencyResponseEditor",
    "HistogramRangeEditor",
    "KernelMatrixEditor",
    "LinkedDimensionsEditor",
    "PercentileRangeEditor",
    "SliderNumberEditor",
    "StripeSuppressionEditor",
    "StructuringElementEditor",
]
