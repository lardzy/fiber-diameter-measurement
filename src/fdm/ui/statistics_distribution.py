from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import math

from PySide6.QtCore import QEvent, QObject, QRectF, QRunnable, QSize, QThreadPool, QTimer, Qt, Signal, Slot
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPalette, QPen
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QLabel,
    QLayout,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from fdm.models import ImageDocument, ProjectState
from fdm.services.measurement_statistics import (
    MeasurementMetric,
    MeasurementStatisticsService,
    MeasurementStatisticsSnapshot,
    StatisticsScope,
)
from fdm.ui.widgets import NoWheelComboBox


@dataclass(frozen=True, slots=True)
class DistributionRecordFilterRequest:
    """An explicit request to apply a chart category to current-image records."""

    document_id: str
    category_label: str
    metric: MeasurementMetric


@dataclass(frozen=True, slots=True)
class _DistributionTaskResult:
    snapshots: tuple[MeasurementStatisticsSnapshot, ...]
    category_comparisons: tuple[
        tuple[str, tuple[MeasurementStatisticsSnapshot, ...]], ...
    ]


class _DistributionTaskSignals(QObject):
    ready = Signal(int, object, object)


class _DistributionTask(QRunnable):
    def __init__(
        self,
        *,
        generation: int,
        signals: _DistributionTaskSignals,
        project: ProjectState,
        metric: MeasurementMetric,
        scope: StatisticsScope,
        document_id: str | None,
    ) -> None:
        super().__init__()
        self._generation = generation
        self._signals = signals
        self._project = project
        self._metric = metric
        self._scope = scope
        self._document_id = document_id

    @Slot()
    def run(self) -> None:
        try:
            service = MeasurementStatisticsService()
            snapshots = service.summarize(
                self._project,
                metric=self._metric,
                scope=self._scope,
                document_id=self._document_id,
            )
            if self._scope is StatisticsScope.PROJECT:
                documents = tuple(self._project.documents)
            else:
                document = self._project.get_document(self._document_id or "")
                documents = (document,) if document is not None else ()
            comparisons = service.summarize_by_category(
                documents,
                metric=self._metric,
                scope=self._scope,
            )
            result: object = _DistributionTaskResult(
                tuple(snapshots),
                tuple(comparisons),
            )
            error: object = None
        except Exception as exc:  # UI renders a non-destructive failure state.
            result = _DistributionTaskResult((), ())
            error = exc
        self._signals.ready.emit(self._generation, result, error)


@dataclass(frozen=True, slots=True)
class _CategoryDatum:
    label: str
    value: float
    color: QColor


def _draw_message(painter: QPainter, rect: QRectF, text: str) -> None:
    device = painter.device()
    palette = device.palette() if isinstance(device, QWidget) else QPalette()
    painter.setPen(palette.color(QPalette.ColorRole.PlaceholderText))
    painter.drawText(
        rect.adjusted(12, 8, -12, -8),
        Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
        text,
    )


class _HistogramCanvas(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.snapshot: MeasurementStatisticsSnapshot | None = None
        self.setMinimumHeight(120)

    def set_snapshot(self, snapshot: MeasurementStatisticsSnapshot | None) -> None:
        self.snapshot = snapshot
        self.update()

    def paintEvent(self, event) -> None:
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = QRectF(self.rect()).adjusted(10, 8, -10, -12)
        snapshot = self.snapshot
        if snapshot is not None and snapshot.metric is MeasurementMetric.COUNT:
            _draw_message(painter, rect, "计数记录表示对象数量，不绘制数值恒为 1 的假直方图。")
            return
        if snapshot is None or snapshot.valid_count < 2 or not snapshot.histogram_counts:
            _draw_message(painter, rect, "至少需要两个同单位有效结果才能显示直方图。")
            return
        palette = self.palette()
        border = palette.color(QPalette.ColorRole.Mid)
        muted = palette.color(QPalette.ColorRole.PlaceholderText)
        accent = QColor("#2A9D8F")
        plot = rect.adjusted(4, 8, -4, -22)
        painter.setPen(QPen(border, 1))
        painter.drawLine(plot.bottomLeft(), plot.bottomRight())
        maximum_count = max(snapshot.histogram_counts) or 1
        bar_width = plot.width() / len(snapshot.histogram_counts)
        for index, count in enumerate(snapshot.histogram_counts):
            height = plot.height() * (count / maximum_count)
            bar = QRectF(
                plot.left() + index * bar_width + 1,
                plot.bottom() - height,
                max(1.0, bar_width - 2),
                height,
            )
            painter.fillRect(bar, QColor(accent.red(), accent.green(), accent.blue(), 190))
        painter.setPen(muted)
        minimum = snapshot.histogram_edges[0]
        maximum = snapshot.histogram_edges[-1]
        painter.drawText(
            QRectF(rect.left(), plot.bottom() + 3, rect.width() / 2, 18),
            Qt.AlignmentFlag.AlignLeft,
            f"{minimum:.4g}",
        )
        painter.drawText(
            QRectF(rect.center().x(), plot.bottom() + 3, rect.width() / 2, 18),
            Qt.AlignmentFlag.AlignRight,
            f"{maximum:.4g} {snapshot.unit}",
        )


class _BoxPlotCanvas(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.snapshot: MeasurementStatisticsSnapshot | None = None
        self.setMinimumHeight(120)

    def set_snapshot(self, snapshot: MeasurementStatisticsSnapshot | None) -> None:
        self.snapshot = snapshot
        self.update()

    def paintEvent(self, event) -> None:
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = QRectF(self.rect()).adjusted(10, 8, -10, -10)
        snapshot = self.snapshot
        if snapshot is not None and snapshot.metric is MeasurementMetric.COUNT:
            _draw_message(painter, rect, "计数记录不适用连续数值箱线图；请查看类别计数柱状图。")
            return
        if snapshot is None or snapshot.valid_count < 2:
            _draw_message(painter, rect, "至少需要两个同单位有效结果才能显示箱线图。")
            return
        minimum = snapshot.minimum
        maximum = snapshot.maximum
        if minimum is None or maximum is None:
            _draw_message(painter, rect, "当前数据无法形成箱线图。")
            return
        palette = self.palette()
        border = palette.color(QPalette.ColorRole.Mid)
        text = palette.color(QPalette.ColorRole.PlaceholderText)
        accent = QColor("#2A9D8F")
        painter.setPen(text)
        if math.isclose(minimum, maximum, rel_tol=1e-12, abs_tol=1e-12):
            painter.drawText(
                rect,
                Qt.AlignmentFlag.AlignCenter,
                f"所有有效值相同：{minimum:.4g} {snapshot.unit}",
            )
            return
        axis = QRectF(rect.left() + 8, rect.center().y() - 18, rect.width() - 16, 36)

        def x_for(value: float | None) -> float:
            if value is None:
                return axis.left()
            return axis.left() + ((value - minimum) / (maximum - minimum)) * axis.width()

        center_y = axis.center().y()
        lower_x = x_for(snapshot.lower_whisker)
        upper_x = x_for(snapshot.upper_whisker)
        q1_x = x_for(snapshot.q1)
        q3_x = x_for(snapshot.q3)
        median_x = x_for(snapshot.median)
        painter.setPen(QPen(border, 1.5))
        painter.drawLine(lower_x, center_y, upper_x, center_y)
        painter.drawLine(lower_x, center_y - 8, lower_x, center_y + 8)
        painter.drawLine(upper_x, center_y - 8, upper_x, center_y + 8)
        box = QRectF(q1_x, center_y - 13, max(2.0, q3_x - q1_x), 26)
        painter.fillRect(box, QColor(accent.red(), accent.green(), accent.blue(), 80))
        painter.drawRect(box)
        painter.setPen(QPen(accent, 2))
        painter.drawLine(median_x, box.top(), median_x, box.bottom())
        painter.setBrush(accent)
        for value in snapshot.outlier_values:
            painter.drawEllipse(QRectF(x_for(value) - 3, center_y - 3, 6, 6))
        painter.setPen(text)
        painter.drawText(
            QRectF(rect.left(), axis.bottom() + 8, rect.width(), 18),
            Qt.AlignmentFlag.AlignCenter,
            f"Q1 {snapshot.q1:.4g} · 中位数 {snapshot.median:.4g} · Q3 {snapshot.q3:.4g} {snapshot.unit}",
        )


class _DonutCanvas(QWidget):
    categoryActivated = Signal(str)
    categoryVisibilityToggled = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._series: tuple[_CategoryDatum, ...] = ()
        self._selected_label = ""
        self._hidden_labels: frozenset[str] = frozenset()
        self._segment_hits: list[tuple[QPainterPath, str]] = []
        self._legend_hits: list[tuple[QRectF, str]] = []
        self.denominator_total = 0.0
        self._legend_offset = 0
        self.setMinimumHeight(120)
        self.setMouseTracking(True)

    def set_series(
        self,
        series: tuple[_CategoryDatum, ...],
        *,
        selected_label: str,
        hidden_labels: frozenset[str],
    ) -> None:
        self._series = series
        self._selected_label = selected_label
        self._hidden_labels = hidden_labels
        self.denominator_total = sum(item.value for item in series if item.value > 0)
        page_size = self._legend_page_size()
        self._legend_offset = min(
            self._legend_offset,
            max(0, len(series) - page_size),
        )
        self.update()

    def _legend_page_size(self) -> int:
        return 5 if len(self._series) > 6 else 6

    def visible_legend_labels(self) -> tuple[str, ...]:
        page_size = self._legend_page_size()
        return tuple(
            item.label
            for item in self._series[
                self._legend_offset:self._legend_offset + page_size
            ]
        )

    def paintEvent(self, event) -> None:
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = QRectF(self.rect()).adjusted(8, 7, -8, -7)
        self._segment_hits.clear()
        self._legend_hits.clear()
        positive = [item for item in self._series if item.value > 0]
        total = self.denominator_total
        if not positive or total <= 0:
            _draw_message(painter, rect, "当前单位没有可用于类别占比的有效记录。")
            return
        chart_side = min(rect.height() - 4, max(74.0, rect.width() * 0.46))
        chart_rect = QRectF(rect.left(), rect.center().y() - chart_side / 2, chart_side, chart_side)
        hole = chart_rect.adjusted(
            chart_rect.width() * 0.29,
            chart_rect.height() * 0.29,
            -chart_rect.width() * 0.29,
            -chart_rect.height() * 0.29,
        )
        hole_path = QPainterPath()
        hole_path.addEllipse(hole)
        start = 90.0
        border = self.palette().color(QPalette.ColorRole.Mid)
        for item in positive:
            span = 360.0 * item.value / total
            if item.label in self._hidden_labels:
                start += span
                continue
            painter.setBrush(item.color)
            painter.setPen(
                QPen(
                    QColor("#2A9D8F") if item.label == self._selected_label else border,
                    3 if item.label == self._selected_label else 1,
                )
            )
            painter.drawPie(chart_rect, round(start * 16), round(span * 16))
            path = QPainterPath()
            path.moveTo(chart_rect.center())
            path.arcTo(chart_rect, start, span)
            path.closeSubpath()
            self._segment_hits.append((path.subtracted(hole_path), item.label))
            start += span
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(self.palette().color(QPalette.ColorRole.AlternateBase))
        painter.drawEllipse(hole)
        painter.setPen(self.palette().color(QPalette.ColorRole.Text))
        painter.drawText(hole, Qt.AlignmentFlag.AlignCenter, f"N\n{int(total)}")

        legend_left = chart_rect.right() + 10
        legend_width = max(40.0, rect.right() - legend_left)
        font_metrics = painter.fontMetrics()
        page_size = self._legend_page_size()
        legend_items = self._series[
            self._legend_offset:self._legend_offset + page_size
        ]
        for index, item in enumerate(legend_items):
            row = QRectF(legend_left, rect.top() + index * 20, legend_width, 19)
            self._legend_hits.append((row, item.label))
            color = QColor(item.color)
            if item.label in self._hidden_labels:
                color.setAlpha(70)
            painter.fillRect(QRectF(row.left(), row.top() + 4, 10, 10), color)
            label = item.label + ("（隐藏）" if item.label in self._hidden_labels else "")
            label = font_metrics.elidedText(
                label,
                Qt.TextElideMode.ElideRight,
                max(20, round(row.width() - 16)),
            )
            painter.setPen(self.palette().color(QPalette.ColorRole.Text))
            painter.drawText(row.adjusted(16, 0, 0, 0), Qt.AlignmentFlag.AlignVCenter, label)
        if len(self._series) > 6:
            painter.setPen(self.palette().color(QPalette.ColorRole.PlaceholderText))
            start = self._legend_offset + 1
            end = self._legend_offset + len(legend_items)
            painter.drawText(
                QRectF(legend_left, rect.top() + 100, legend_width, 19),
                f"图例 {start}–{end}/{len(self._series)} · 滚轮浏览",
            )

    def mousePressEvent(self, event) -> None:
        point = event.position()
        for rect, label in self._legend_hits:
            if rect.contains(point):
                self.categoryVisibilityToggled.emit(label)
                return
        for path, label in self._segment_hits:
            if path.contains(point):
                self.categoryActivated.emit(label)
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        point = event.position()
        for rect, label in self._legend_hits:
            if rect.contains(point):
                self.setToolTip(
                    f"{self._category_value_text(label)}；点击仅隐藏或显示该类别，不会改变统计分母。"
                )
                return
        for path, label in self._segment_hits:
            if path.contains(point):
                self.setToolTip(f"{self._category_value_text(label)}；点击高亮该类别。")
                return
        self.setToolTip("")
        super().mouseMoveEvent(event)

    def wheelEvent(self, event) -> None:
        if len(self._series) <= 6:
            super().wheelEvent(event)
            return
        delta = event.angleDelta().y()
        if delta == 0:
            super().wheelEvent(event)
            return
        step = -1 if delta > 0 else 1
        maximum = max(0, len(self._series) - self._legend_page_size())
        self._legend_offset = max(0, min(maximum, self._legend_offset + step))
        self.update()
        event.accept()

    def _category_value_text(self, label: str) -> str:
        item = next((candidate for candidate in self._series if candidate.label == label), None)
        if item is None or self.denominator_total <= 0:
            return label
        percent = (item.value / self.denominator_total) * 100.0
        return f"{label}：N={int(item.value)}，占 {percent:.1f}%"


class _BarCanvas(QWidget):
    categoryActivated = Signal(str)
    MAX_VISIBLE_CATEGORIES = 12

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._series: tuple[_CategoryDatum, ...] = ()
        self._selected_label = ""
        self._bar_hits: list[tuple[QRectF, str, float]] = []
        self.truncated_count = 0
        self.setMinimumHeight(120)
        self.setMouseTracking(True)

    def set_series(
        self,
        series: tuple[_CategoryDatum, ...],
        *,
        selected_label: str,
        hidden_labels: frozenset[str],
    ) -> None:
        visible = [item for item in series if item.label not in hidden_labels]
        self.truncated_count = max(0, len(visible) - self.MAX_VISIBLE_CATEGORIES)
        if self.truncated_count:
            visible = sorted(visible, key=lambda item: (-item.value, item.label.casefold()))[
                : self.MAX_VISIBLE_CATEGORIES
            ]
        self._series = tuple(visible)
        self._selected_label = selected_label
        self.update()

    def paintEvent(self, event) -> None:
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = QRectF(self.rect()).adjusted(9, 7, -9, -8)
        self._bar_hits.clear()
        positive = [item for item in self._series if item.value >= 0]
        maximum = max((item.value for item in positive), default=0.0)
        if not positive or maximum <= 0:
            _draw_message(painter, rect, "当前单位没有可比较的类别数据。")
            return
        plot = rect.adjusted(4, 14, -4, -28)
        painter.setPen(QPen(self.palette().color(QPalette.ColorRole.Mid), 1))
        painter.drawLine(plot.bottomLeft(), plot.bottomRight())
        slot_width = plot.width() / len(positive)
        font_metrics = painter.fontMetrics()
        for index, item in enumerate(positive):
            height = plot.height() * item.value / maximum
            bar_width = max(3.0, slot_width * 0.62)
            bar = QRectF(
                plot.left() + index * slot_width + (slot_width - bar_width) / 2,
                plot.bottom() - height,
                bar_width,
                height,
            )
            painter.fillRect(bar, item.color)
            if item.label == self._selected_label:
                painter.setPen(QPen(QColor("#2A9D8F"), 2))
                painter.drawRect(bar)
            self._bar_hits.append((bar, item.label, item.value))
            painter.setPen(self.palette().color(QPalette.ColorRole.Text))
            value_text = f"{item.value:.4g}"
            painter.drawText(
                QRectF(bar.left() - slot_width * 0.2, bar.top() - 17, slot_width * 1.4, 16),
                Qt.AlignmentFlag.AlignCenter,
                value_text,
            )
            label = font_metrics.elidedText(
                item.label,
                Qt.TextElideMode.ElideRight,
                max(10, round(slot_width - 3)),
            )
            painter.drawText(
                QRectF(plot.left() + index * slot_width, plot.bottom() + 4, slot_width, 18),
                Qt.AlignmentFlag.AlignCenter,
                label,
            )

    def mousePressEvent(self, event) -> None:
        for rect, label, _value in self._bar_hits:
            if rect.contains(event.position()):
                self.categoryActivated.emit(label)
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        for rect, label, value in self._bar_hits:
            if rect.contains(event.position()):
                self.setToolTip(f"{label}：{value:.6g}；点击后高亮该类别。")
                return
        self.setToolTip("")
        super().mouseMoveEvent(event)


class _ChartCard(QFrame):
    def __init__(
        self,
        title: str,
        description: str,
        canvas: QWidget,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("distributionChartCard")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumHeight(182)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 9, 10, 9)
        layout.setSpacing(3)
        title_label = QLabel(title, self)
        title_label.setProperty("chartTitle", True)
        self.description_label = QLabel(description, self)
        self.description_label.setProperty("chartDescription", True)
        self.description_label.setWordWrap(True)
        layout.addWidget(title_label)
        layout.addWidget(self.description_label)
        layout.addWidget(canvas, 1)

    def set_description(self, text: str) -> None:
        self.description_label.setText(text)


def _control_group(label: str, control: QWidget, parent: QWidget) -> QWidget:
    group = QWidget(parent)
    group.setMinimumWidth(0)
    layout = QVBoxLayout(group)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(2)
    caption = QLabel(label, group)
    caption.setProperty("distributionControlCaption", True)
    control.setParent(group)
    control.setMinimumWidth(0)
    control.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    layout.addWidget(caption)
    layout.addWidget(control)
    return group


class StatisticsDistributionWidget(QWidget):
    """Dependency-free, responsive distribution dashboard.

    ``set_context`` is the preferred integration API.  The widget owns its
    metric/scope controls and recalculates from the supplied project using a
    200 ms debounce and generation-guarded background work.  ``set_snapshot``
    remains as a compatibility seam for older callers which only have one
    summary available.
    """

    recordFilterRequested = Signal(object)
    categoryHighlighted = Signal(str)
    BACKGROUND_THRESHOLD = 5000
    REFRESH_DELAY_MS = 200

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("statisticsDistributionDashboard")
        self.setMinimumHeight(120)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._service = MeasurementStatisticsService()
        self._project = ProjectState.empty()
        self._document: ImageDocument | None = None
        self._snapshots: tuple[MeasurementStatisticsSnapshot, ...] = ()
        self._category_comparisons: tuple[
            tuple[str, tuple[MeasurementStatisticsSnapshot, ...]], ...
        ] = ()
        self._category_colors: dict[str, QColor] = {}
        self._color_conflicts: frozenset[str] = frozenset()
        self._hidden_categories: set[str] = set()
        self._generation = 0
        self._metric_initialized = False
        self._unit_selection_explicit = False
        self._context_mode = False
        self._updating_controls = False
        self._chart_columns = 0
        self._control_columns = 0
        self._last_non_count_bar_metric = "mean"
        self._task_signals = _DistributionTaskSignals(self)
        self._task_signals.ready.connect(self._on_async_ready)
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.setInterval(self.REFRESH_DELAY_MS)
        self._refresh_timer.timeout.connect(self._refresh_now)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # The whole dashboard scrolls as one page.  Keeping the controls and
        # context outside a cards-only scroll area left almost no chart viewport
        # in the compact results drawer and made the final card unreachable.
        self._scroll = QScrollArea(self)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Ignored)
        self._scroll.setProperty("redirectEditorWheel", True)
        self._scroll.viewport().installEventFilter(self)
        self._scroll_content = QWidget(self._scroll)
        self._scroll_content.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,
        )
        self._scroll_content_layout = QVBoxLayout(self._scroll_content)
        self._scroll_content_layout.setContentsMargins(0, 0, 0, 0)
        self._scroll_content_layout.setSpacing(6)
        self._scroll_content_layout.setSizeConstraint(
            QLayout.SizeConstraint.SetMinimumSize
        )
        self._scroll.setWidget(self._scroll_content)
        root.addWidget(self._scroll, 1)

        self._controls_layout = QGridLayout()
        self._controls_layout.setContentsMargins(0, 0, 0, 0)
        self._controls_layout.setHorizontalSpacing(8)
        self._controls_layout.setVerticalSpacing(5)
        self._scroll_content_layout.addLayout(self._controls_layout)

        self.metric_combo = NoWheelComboBox(self._scroll_content)
        self.metric_combo.addItem("长度", MeasurementMetric.LENGTH)
        self.metric_combo.addItem("面积", MeasurementMetric.AREA)
        self.metric_combo.addItem("计数", MeasurementMetric.COUNT)
        self.scope_combo = NoWheelComboBox(self._scroll_content)
        self.scope_combo.addItem("当前图片", StatisticsScope.CURRENT_DOCUMENT)
        self.scope_combo.addItem("整个项目", StatisticsScope.PROJECT)
        self.target_combo = NoWheelComboBox(self._scroll_content)
        self.target_combo.addItem("整体", "overall")
        self.target_combo.addItem("指定类别", "category")
        self.unit_combo = NoWheelComboBox(self._scroll_content)
        self.category_combo = NoWheelComboBox(self._scroll_content)
        self.bar_metric_combo = NoWheelComboBox(self._scroll_content)
        self.bar_metric_combo.addItem("有效 N", "valid_count")
        self.bar_metric_combo.addItem("均值", "mean")
        self.bar_metric_combo.addItem("中位数", "median")
        self.bar_metric_combo.addItem("总量", "total_value")
        self.bar_metric_combo.setCurrentIndex(1)
        self.filter_records_button = QPushButton("筛选当前图片记录", self._scroll_content)
        self.filter_records_button.setEnabled(False)
        self._control_widgets = (
            _control_group("指标", self.metric_combo, self._scroll_content),
            _control_group("范围", self.scope_combo, self._scroll_content),
            _control_group("数据对象", self.target_combo, self._scroll_content),
            _control_group("单位", self.unit_combo, self._scroll_content),
            _control_group("类别", self.category_combo, self._scroll_content),
            _control_group("柱状图指标", self.bar_metric_combo, self._scroll_content),
            _control_group("记录联动", self.filter_records_button, self._scroll_content),
        )

        self.context_label = QLabel("等待统计数据。", self._scroll_content)
        self.context_label.setObjectName("distributionContextLabel")
        self.context_label.setWordWrap(True)
        self._scroll_content_layout.addWidget(self.context_label)

        self._cards_container = QWidget(self._scroll_content)
        self._cards_container.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,
        )
        self._cards_layout = QGridLayout(self._cards_container)
        self._cards_layout.setContentsMargins(0, 0, 0, 0)
        self._cards_layout.setHorizontalSpacing(8)
        self._cards_layout.setVerticalSpacing(8)
        self._scroll_content_layout.addWidget(self._cards_container)

        self.histogram_canvas = _HistogramCanvas(self)
        self.box_plot_canvas = _BoxPlotCanvas(self)
        self.donut_canvas = _DonutCanvas(self)
        self.bar_canvas = _BarCanvas(self)
        self.histogram_card = _ChartCard(
            "直方图",
            "显示所选整体或类别的数值频数；不同单位不会混合。",
            self.histogram_canvas,
            self,
        )
        self.box_plot_card = _ChartCard(
            "箱线图",
            "显示中位数、四分位数、1.5×IQR 须和建议复核的异常点。",
            self.box_plot_canvas,
            self,
        )
        self.donut_card = _ChartCard(
            "类别构成",
            "按有效记录数显示类别占比；点击图例仅切换可见性，类别较多时滚轮浏览图例。",
            self.donut_canvas,
            self,
        )
        self.bar_card = _ChartCard(
            "类别比较",
            "按所选指标比较类别；点击柱只高亮类别。",
            self.bar_canvas,
            self,
        )
        self._cards = (
            self.histogram_card,
            self.box_plot_card,
            self.donut_card,
            self.bar_card,
        )

        self.metric_combo.currentIndexChanged.connect(self._on_metric_changed)
        self.scope_combo.currentIndexChanged.connect(self._on_scope_changed)
        self.target_combo.currentIndexChanged.connect(self._render)
        self.unit_combo.currentIndexChanged.connect(self._render)
        self.unit_combo.activated.connect(self._on_unit_activated)
        self.category_combo.currentIndexChanged.connect(self._render)
        self.bar_metric_combo.currentIndexChanged.connect(self._on_bar_metric_changed)
        self.filter_records_button.clicked.connect(self._request_record_filter)
        self.donut_canvas.categoryActivated.connect(self._select_category_from_chart)
        self.bar_canvas.categoryActivated.connect(self._select_category_from_chart)
        self.donut_canvas.categoryVisibilityToggled.connect(self._toggle_category_visibility)
        self._apply_responsive_layout(960)
        self._apply_style()
        self._render()

    @property
    def snapshots(self) -> tuple[MeasurementStatisticsSnapshot, ...]:
        return self._snapshots

    @property
    def category_comparisons(
        self,
    ) -> tuple[tuple[str, tuple[MeasurementStatisticsSnapshot, ...]], ...]:
        return self._category_comparisons

    def active_metric(self) -> MeasurementMetric:
        return MeasurementMetric(self.metric_combo.currentData() or MeasurementMetric.LENGTH)

    def active_scope(self) -> StatisticsScope:
        return StatisticsScope(
            self.scope_combo.currentData() or StatisticsScope.CURRENT_DOCUMENT
        )

    def set_context(
        self,
        project: ProjectState,
        document: ImageDocument | None,
        *,
        suggested_metric: MeasurementMetric | None = None,
    ) -> None:
        """Set live project context without taking over the user's chart controls."""

        self._project = project
        self._document = document
        self._context_mode = True
        if suggested_metric is not None and not self._metric_initialized:
            self._set_combo_data(self.metric_combo, MeasurementMetric(suggested_metric))
            self._metric_initialized = True
        self._queue_refresh()

    def set_snapshot(self, snapshot: MeasurementStatisticsSnapshot | None) -> None:
        """Compatibility adapter for integrations that only provide one snapshot."""

        if self._context_mode:
            # Old main-window integrations may still have a delayed
            # ``set_snapshot`` timer.  Once full project context is available,
            # that compatibility update must not erase category data.
            return
        self._refresh_timer.stop()
        self._generation += 1
        if snapshot is None:
            self._snapshots = ()
            self._category_comparisons = ()
        else:
            self._set_combo_data(self.metric_combo, snapshot.metric)
            compatible_scope = (
                StatisticsScope.PROJECT
                if snapshot.scope is StatisticsScope.PROJECT
                else StatisticsScope.CURRENT_DOCUMENT
            )
            self._set_combo_data(self.scope_combo, compatible_scope)
            self._metric_initialized = True
            self._snapshots = (snapshot,)
            self._category_comparisons = ()
        self._category_colors = {}
        self._color_conflicts = frozenset()
        self._refresh_option_controls()
        self._render()

    def refresh(self) -> None:
        self._queue_refresh()

    def _queue_refresh(self) -> None:
        self._generation += 1
        self._refresh_timer.start()

    @Slot()
    def _refresh_now(self) -> None:
        generation = self._generation
        metric = self.active_metric()
        scope = self.active_scope()
        document = self._document
        if scope is StatisticsScope.CURRENT_DOCUMENT and document is None:
            self._apply_results(generation, _DistributionTaskResult((), ()), None)
            return
        documents = (
            tuple(self._project.documents)
            if scope is StatisticsScope.PROJECT
            else ((document,) if document is not None else ())
        )
        candidate_count = sum(len(item.measurements) for item in documents)
        if candidate_count > self.BACKGROUND_THRESHOLD:
            self.context_label.setText("正在后台计算分布；数据变化后晚到结果会自动丢弃。")
            task = _DistributionTask(
                generation=generation,
                signals=self._task_signals,
                project=copy.deepcopy(self._project),
                metric=metric,
                scope=scope,
                document_id=document.id if document is not None else None,
            )
            QThreadPool.globalInstance().start(task)
            return
        try:
            snapshots = self._service.summarize(
                self._project,
                metric=metric,
                scope=scope,
                document_id=document.id if document is not None else None,
            )
            comparisons = self._service.summarize_by_category(
                documents,
                metric=metric,
                scope=scope,
            )
            result = _DistributionTaskResult(tuple(snapshots), tuple(comparisons))
            error: object = None
        except (KeyError, ValueError) as exc:
            result = _DistributionTaskResult((), ())
            error = exc
        self._apply_results(generation, result, error)

    @Slot(int, object, object)
    def _on_async_ready(self, generation: int, result: object, error: object) -> None:
        if not isinstance(result, _DistributionTaskResult):
            result = _DistributionTaskResult((), ())
        self._apply_results(generation, result, error)

    def _apply_results(
        self,
        generation: int,
        result: _DistributionTaskResult,
        error: object,
    ) -> None:
        if generation != self._generation:
            return
        self._snapshots = result.snapshots if error is None else ()
        self._category_comparisons = result.category_comparisons if error is None else ()
        self._category_colors, conflicts = self._resolve_category_colors()
        self._color_conflicts = frozenset(conflicts)
        self._refresh_option_controls()
        self._render(error=error)

    def _refresh_option_controls(self) -> None:
        selected_unit = self.unit_combo.currentData()
        units = tuple(dict.fromkeys(item.unit for item in self._snapshots))
        if len(units) > 1:
            items = (("请选择单位…", None), *((unit, unit) for unit in units))
            preferred_unit = (
                selected_unit
                if self._unit_selection_explicit and selected_unit in units
                else None
            )
        else:
            self._unit_selection_explicit = False
            items = tuple((unit, unit) for unit in units)
            preferred_unit = selected_unit
        self._replace_combo_items(self.unit_combo, items, preferred_unit)
        self.unit_combo.setEnabled(len(units) > 1)
        selected_category = self.category_combo.currentData()
        categories = tuple(label for label, _snapshots in self._category_comparisons)
        self._replace_combo_items(
            self.category_combo,
            ((label, label) for label in categories),
            selected_category,
        )
        self._hidden_categories.intersection_update(categories)
        if not categories and self.target_combo.currentData() == "category":
            self._set_combo_data(self.target_combo, "overall")
        self._sync_control_states()

    def _sync_control_states(self) -> None:
        metric = self.active_metric()
        category_mode = self.target_combo.currentData() == "category"
        self.category_combo.setEnabled(category_mode and self.category_combo.count() > 0)
        self.filter_records_button.setEnabled(
            category_mode
            and self._document is not None
            and bool(self.category_combo.currentData())
        )
        if metric is MeasurementMetric.COUNT:
            self._set_combo_data(self.bar_metric_combo, "valid_count")
            self.bar_metric_combo.setEnabled(False)
        else:
            if (
                self.bar_metric_combo.currentData() == "valid_count"
                and self._last_non_count_bar_metric != "valid_count"
            ):
                self._set_combo_data(
                    self.bar_metric_combo,
                    self._last_non_count_bar_metric,
                )
            self.bar_metric_combo.setEnabled(True)

    def _render(self, *_args, error: object = None) -> None:
        self._sync_control_states()
        snapshot = self._active_snapshot()
        selected_category = (
            str(self.category_combo.currentData() or "")
            if self.target_combo.currentData() == "category"
            else ""
        )
        self.histogram_canvas.set_snapshot(snapshot)
        self.box_plot_canvas.set_snapshot(snapshot)
        series = self._category_series("valid_count")
        hidden = frozenset(self._hidden_categories)
        self.donut_canvas.set_series(
            series,
            selected_label=selected_category,
            hidden_labels=hidden,
        )
        bar_metric = str(self.bar_metric_combo.currentData() or "valid_count")
        bar_series = self._category_series(bar_metric)
        self.bar_canvas.set_series(
            bar_series,
            selected_label=selected_category,
            hidden_labels=hidden,
        )
        bar_label = {
            "valid_count": "有效 N",
            "mean": "均值",
            "median": "中位数",
            "total_value": "总量",
        }[bar_metric]
        if bar_metric == "total_value":
            if self.active_metric() is MeasurementMetric.AREA:
                bar_label = "对象净面积合计"
            elif self.active_metric() is MeasurementMetric.LENGTH:
                bar_label = "长度合计"
        bar_note = f"按“{bar_label}”比较同单位类别；点击柱只高亮类别。"
        if self.bar_canvas.truncated_count:
            bar_note += f" 类别较多，当前显示最高的 12 类，另有 {self.bar_canvas.truncated_count} 类。"
        self.bar_card.set_description(bar_note)

        metric_label = {
            MeasurementMetric.LENGTH: "长度",
            MeasurementMetric.AREA: "面积",
            MeasurementMetric.COUNT: "计数",
        }[self.active_metric()]
        scope_label = "整个项目" if self.active_scope() is StatisticsScope.PROJECT else "当前图片"
        target_label = selected_category or "整体"
        unit_data = self.unit_combo.currentData()
        unit = str(unit_data) if unit_data is not None else ("请选择单位" if len(self._snapshots) > 1 else "—")
        valid_count = snapshot.valid_count if snapshot is not None else 0
        context = f"{metric_label} · {scope_label} · {target_label} · {unit} · N={valid_count}"
        if len(self._snapshots) > 1:
            context += "；项目含多种单位，当前只显示所选单位"
        if self._color_conflicts:
            context += "；同名类别存在颜色冲突，图表使用稳定配色"
        if self.active_metric() is MeasurementMetric.COUNT:
            context += "；计数不绘制直方图和箱线图"
        if error is not None:
            context = "分布计算失败；测量数据未被修改。"
        self.context_label.setText(context)

    def _active_snapshot(self) -> MeasurementStatisticsSnapshot | None:
        unit = self.unit_combo.currentData()
        if self.target_combo.currentData() == "category":
            selected = str(self.category_combo.currentData() or "").casefold()
            for label, snapshots in self._category_comparisons:
                if label.casefold() != selected:
                    continue
                return next((item for item in snapshots if item.unit == unit), None)
            return None
        return next((item for item in self._snapshots if item.unit == unit), None)

    def _category_series(self, value_key: str) -> tuple[_CategoryDatum, ...]:
        unit = self.unit_combo.currentData()
        series: list[_CategoryDatum] = []
        for label, snapshots in self._category_comparisons:
            snapshot = next((item for item in snapshots if item.unit == unit), None)
            if snapshot is None:
                continue
            value = getattr(snapshot, value_key, None)
            if value is None or not math.isfinite(float(value)):
                continue
            series.append(
                _CategoryDatum(
                    label=label,
                    value=float(value),
                    color=self._category_colors.get(label.casefold(), self._stable_color(label)),
                )
            )
        return tuple(series)

    def _resolve_category_colors(self) -> tuple[dict[str, QColor], set[str]]:
        scope = self.active_scope()
        documents = (
            tuple(self._project.documents)
            if scope is StatisticsScope.PROJECT
            else ((self._document,) if self._document is not None else ())
        )
        collected: dict[str, tuple[str, set[str]]] = {}
        for document in documents:
            for group in document.fiber_groups:
                label = group.label.strip() or group.display_name()
                key = label.casefold()
                _display, colors = collected.setdefault(key, (label, set()))
                color = QColor(group.color)
                if color.isValid():
                    colors.add(color.name(QColor.NameFormat.HexRgb).casefold())
        resolved: dict[str, QColor] = {}
        conflicts: set[str] = set()
        comparison_labels = tuple(label for label, _snapshots in self._category_comparisons)
        for label in comparison_labels:
            key = label.casefold()
            colors = collected.get(key, (label, set()))[1]
            if len(colors) == 1:
                resolved[key] = QColor(next(iter(colors)))
            else:
                resolved[key] = self._stable_color(label)
                if len(colors) > 1:
                    conflicts.add(label)
        return resolved, conflicts

    @staticmethod
    def _stable_color(label: str) -> QColor:
        digest = hashlib.sha256(label.strip().casefold().encode("utf-8")).digest()
        hue = int.from_bytes(digest[:2], "big") % 360
        return QColor.fromHsv(hue, 155, 210)

    def _on_metric_changed(self, _index: int) -> None:
        if self._updating_controls:
            return
        self._metric_initialized = True
        self._unit_selection_explicit = False
        self._queue_refresh()

    def _on_scope_changed(self, _index: int) -> None:
        if not self._updating_controls:
            self._unit_selection_explicit = False
            self._queue_refresh()

    def _on_unit_activated(self, _index: int) -> None:
        self._unit_selection_explicit = self.unit_combo.currentData() is not None
        self._render()

    def _on_bar_metric_changed(self, _index: int) -> None:
        if self.active_metric() is not MeasurementMetric.COUNT:
            self._last_non_count_bar_metric = str(
                self.bar_metric_combo.currentData() or "mean"
            )
        self._render()

    def _select_category_from_chart(self, label: str) -> None:
        if not self._set_combo_data(self.category_combo, label):
            return
        self._set_combo_data(self.target_combo, "category")
        self.categoryHighlighted.emit(label)
        self._render()

    def _toggle_category_visibility(self, label: str) -> None:
        if label in self._hidden_categories:
            self._hidden_categories.remove(label)
        else:
            self._hidden_categories.add(label)
        self._render()

    def _request_record_filter(self) -> None:
        document = self._document
        category = str(self.category_combo.currentData() or "")
        if document is None or not category or self.target_combo.currentData() != "category":
            return
        self.recordFilterRequested.emit(
            DistributionRecordFilterRequest(
                document_id=document.id,
                category_label=category,
                metric=self.active_metric(),
            )
        )

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._apply_responsive_layout(max(1, self._scroll.viewport().width()))

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if watched is self._scroll.viewport() and event.type() == QEvent.Type.Resize:
            self._apply_responsive_layout(max(1, self._scroll.viewport().width()))
        return super().eventFilter(watched, event)

    def minimumSizeHint(self) -> QSize:
        # A collapsed results drawer must never force the supported 1093x576
        # main window to grow.  The complete dashboard is internally scrollable.
        return QSize(320, 112)

    def sizeHint(self) -> QSize:
        return QSize(960, 260)

    @staticmethod
    def chart_columns_for_width(width: int) -> int:
        if width >= 1200:
            return 4
        if width >= 640:
            return 2
        return 1

    @staticmethod
    def control_columns_for_width(width: int) -> int:
        if width >= 1200:
            return 7
        if width >= 640:
            return 4
        return 2

    def _apply_responsive_layout(self, width: int) -> None:
        chart_columns = self.chart_columns_for_width(width)
        if chart_columns != self._chart_columns:
            previous_columns = self._chart_columns
            self._chart_columns = chart_columns
            while self._cards_layout.count():
                self._cards_layout.takeAt(0)
            for column in range(max(previous_columns, chart_columns)):
                self._cards_layout.setColumnStretch(column, 0)
            for index, card in enumerate(self._cards):
                self._cards_layout.addWidget(
                    card,
                    index // chart_columns,
                    index % chart_columns,
                )
            for column in range(chart_columns):
                self._cards_layout.setColumnStretch(column, 1)
        control_columns = self.control_columns_for_width(width)
        if control_columns != self._control_columns:
            previous_columns = self._control_columns
            self._control_columns = control_columns
            while self._controls_layout.count():
                self._controls_layout.takeAt(0)
            for column in range(max(previous_columns, control_columns)):
                self._controls_layout.setColumnStretch(column, 0)
            for index, control in enumerate(self._control_widgets):
                self._controls_layout.addWidget(
                    control,
                    index // control_columns,
                    index % control_columns,
                )
            for column in range(control_columns):
                self._controls_layout.setColumnStretch(column, 1)

    def _replace_combo_items(self, combo: QComboBox, items, selected_data: object) -> None:
        self._updating_controls = True
        blocked = combo.blockSignals(True)
        try:
            combo.clear()
            for label, data in items:
                combo.addItem(str(label), data)
            if combo.count() and not self._set_combo_data(combo, selected_data):
                combo.setCurrentIndex(0)
        finally:
            combo.blockSignals(blocked)
            self._updating_controls = False

    def _set_combo_data(self, combo: QComboBox, value: object) -> bool:
        for index in range(combo.count()):
            if combo.itemData(index) == value:
                blocked = combo.blockSignals(True)
                combo.setCurrentIndex(index)
                combo.blockSignals(blocked)
                return True
        return False

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QWidget#statisticsDistributionDashboard { background: palette(window); }
            QFrame#distributionChartCard {
                border: 1px solid palette(mid);
                border-radius: 6px;
                background: palette(alternate-base);
            }
            QLabel[chartTitle="true"] { font-weight: 700; }
            QLabel[chartDescription="true"],
            QLabel[distributionControlCaption="true"],
            QLabel#distributionContextLabel { color: palette(placeholder-text); }
            """
        )
