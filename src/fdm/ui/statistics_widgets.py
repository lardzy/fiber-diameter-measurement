from __future__ import annotations

import copy
from dataclasses import dataclass

from PySide6.QtCore import QObject, QRectF, QRunnable, QThreadPool, Qt, Signal, Slot
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPalette, QPen
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from fdm.models import ImageDocument, Measurement, ProjectState
from fdm.services.measurement_statistics import (
    MeasurementMetric,
    MeasurementStatisticsService,
    MeasurementStatisticsSnapshot,
    StatisticsScope,
)


def metric_for_tool_mode(tool_mode: str) -> MeasurementMetric:
    token = str(tool_mode or "").strip()
    if token in {"polygon_area", "freehand_area", "magic_segment", "reference_propagation"}:
        return MeasurementMetric.AREA
    if token == "count":
        return MeasurementMetric.COUNT
    return MeasurementMetric.LENGTH


def metric_for_measurement(measurement: Measurement | None) -> MeasurementMetric | None:
    if measurement is None:
        return None
    if measurement.measurement_kind == "area":
        return MeasurementMetric.AREA
    if measurement.measurement_kind == "count":
        return MeasurementMetric.COUNT
    if measurement.measurement_kind in {"line", "polyline"}:
        return MeasurementMetric.LENGTH
    return None


class _StatisticsTaskSignals(QObject):
    ready = Signal(int, object, object)


@dataclass(frozen=True, slots=True)
class _StatisticsTaskResult:
    snapshots: tuple[MeasurementStatisticsSnapshot, ...]
    category_comparisons: tuple[
        tuple[str, tuple[MeasurementStatisticsSnapshot, ...]], ...
    ]


class _StatisticsTask(QRunnable):
    def __init__(
        self,
        *,
        generation: int,
        signals: _StatisticsTaskSignals,
        project: ProjectState,
        metric: MeasurementMetric,
        scope: StatisticsScope,
        document_id: str | None,
        fiber_group_id: str | None,
    ) -> None:
        super().__init__()
        self._generation = generation
        self._signals = signals
        self._project = project
        self._metric = metric
        self._scope = scope
        self._document_id = document_id
        self._fiber_group_id = fiber_group_id

    @Slot()
    def run(self) -> None:
        try:
            service = MeasurementStatisticsService()
            snapshots = service.summarize(
                self._project,
                metric=self._metric,
                scope=self._scope,
                document_id=self._document_id,
                fiber_group_id=self._fiber_group_id,
            )
            if self._scope is StatisticsScope.PROJECT:
                category_documents = self._project.documents
            else:
                selected_document = self._project.get_document(self._document_id or "")
                category_documents = (selected_document,) if selected_document is not None else ()
            category_comparisons = service.summarize_by_category(
                category_documents,
                metric=self._metric,
                scope=self._scope,
            )
            result: object = _StatisticsTaskResult(
                snapshots=tuple(snapshots),
                category_comparisons=tuple(category_comparisons),
            )
            error: object = None
        except Exception as exc:  # the UI receives a structured failure state
            result = _StatisticsTaskResult((), ())
            error = exc
        self._signals.ready.emit(self._generation, result, error)


class _MetricCell(QFrame):
    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("statisticsMetricCell")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(9, 7, 9, 7)
        layout.setSpacing(2)
        title_label = QLabel(title, self)
        title_label.setProperty("statisticsCaption", True)
        self.value_label = QLabel("—", self)
        self.value_label.setProperty("statisticsValue", True)
        self.value_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(title_label)
        layout.addWidget(self.value_label)


class MeasurementStatisticsPanel(QWidget):
    """Compact live statistics and selected-object inspector."""

    resultsRequested = Signal()
    statisticsChanged = Signal(object)
    BACKGROUND_THRESHOLD = 5000

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("measurementStatisticsPanel")
        self._service = MeasurementStatisticsService()
        self._project = ProjectState.empty()
        self._document: ImageDocument | None = None
        self._tool_mode = "select"
        self._selected_measurement: Measurement | None = None
        self._snapshots: tuple[MeasurementStatisticsSnapshot, ...] = ()
        self._category_comparisons: tuple[
            tuple[str, tuple[MeasurementStatisticsSnapshot, ...]], ...
        ] = ()
        self._generation = 0
        self._task_signals = _StatisticsTaskSignals()
        self._task_signals.ready.connect(self._on_async_statistics_ready)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(8)

        header = QHBoxLayout()
        title = QLabel("实时统计", self)
        title.setProperty("panelTitle", True)
        self._results_button = QPushButton("结果", self)
        self._results_button.clicked.connect(self.resultsRequested)
        header.addWidget(title)
        header.addStretch(1)
        header.addWidget(self._results_button)
        root.addLayout(header)

        filter_row = QHBoxLayout()
        self.metric_combo = QComboBox(self)
        self.metric_combo.addItem("自动", None)
        self.metric_combo.addItem("长度", MeasurementMetric.LENGTH)
        self.metric_combo.addItem("面积", MeasurementMetric.AREA)
        self.metric_combo.addItem("计数", MeasurementMetric.COUNT)
        self.metric_combo.setMinimumWidth(0)
        self.metric_combo.setMinimumContentsLength(4)
        self.metric_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.metric_combo.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.metric_combo.currentIndexChanged.connect(self.refresh)
        self.scope_combo = QComboBox(self)
        self.scope_combo.addItem("当前类别", StatisticsScope.CURRENT_CATEGORY)
        self.scope_combo.addItem("当前图片", StatisticsScope.CURRENT_DOCUMENT)
        self.scope_combo.addItem("整个项目", StatisticsScope.PROJECT)
        self.scope_combo.setMinimumWidth(0)
        self.scope_combo.setMinimumContentsLength(4)
        self.scope_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.scope_combo.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.scope_combo.currentIndexChanged.connect(self.refresh)
        filter_row.addWidget(self.metric_combo, 1)
        filter_row.addWidget(self.scope_combo, 1)
        root.addLayout(filter_row)

        self.current_value_label = QLabel("尚未选择测量对象", self)
        self.current_value_label.setObjectName("currentMeasurementValue")
        self.current_value_label.setWordWrap(True)
        self.current_value_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        root.addWidget(self.current_value_label)
        self._object_details_toggle = QToolButton(self)
        self._object_details_toggle.setText("当前对象属性")
        self._object_details_toggle.setCheckable(True)
        self._object_details_toggle.setArrowType(Qt.ArrowType.RightArrow)
        self._object_details_toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._object_details_toggle.toggled.connect(self._toggle_object_details)
        self._object_details_toggle.hide()
        root.addWidget(self._object_details_toggle)
        self._object_details_label = QLabel(self)
        self._object_details_label.setWordWrap(True)
        self._object_details_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._object_details_label.hide()
        root.addWidget(self._object_details_label)

        metrics = QGridLayout()
        metrics.setContentsMargins(0, 0, 0, 0)
        metrics.setHorizontalSpacing(6)
        metrics.setVerticalSpacing(6)
        self._n_cell = _MetricCell("有效 N", self)
        self._mean_cell = _MetricCell("均值", self)
        self._std_cell = _MetricCell("总体标准差", self)
        self._cv_cell = _MetricCell("CV", self)
        metrics.addWidget(self._n_cell, 0, 0)
        metrics.addWidget(self._mean_cell, 0, 1)
        metrics.addWidget(self._std_cell, 1, 0)
        metrics.addWidget(self._cv_cell, 1, 1)
        root.addLayout(metrics)

        self.details_label = QLabel("", self)
        self.details_label.setObjectName("statisticsDetails")
        self.details_label.setWordWrap(True)
        self.details_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        root.addWidget(self.details_label)
        self.quality_label = QLabel("有效 0 · 需复核 0 · 失败 0", self)
        self.quality_label.setObjectName("statisticsQuality")
        self.quality_label.setWordWrap(True)
        root.addWidget(self.quality_label)
        self._apply_style()

    @property
    def snapshots(self) -> tuple[MeasurementStatisticsSnapshot, ...]:
        return self._snapshots

    @property
    def category_comparisons(
        self,
    ) -> tuple[tuple[str, tuple[MeasurementStatisticsSnapshot, ...]], ...]:
        return self._category_comparisons

    def set_context(
        self,
        project: ProjectState,
        document: ImageDocument | None,
        *,
        tool_mode: str,
        selected_measurement: Measurement | None,
    ) -> None:
        self._project = project
        self._document = document
        self._tool_mode = str(tool_mode or "select")
        self._selected_measurement = selected_measurement
        self.refresh()

    def active_metric(self) -> MeasurementMetric:
        explicit = self.metric_combo.currentData()
        if explicit is not None:
            return MeasurementMetric(explicit)
        selected_metric = metric_for_measurement(self._selected_measurement)
        return selected_metric or metric_for_tool_mode(self._tool_mode)

    def active_scope(self) -> StatisticsScope:
        return StatisticsScope(self.scope_combo.currentData() or StatisticsScope.CURRENT_CATEGORY)

    def refresh(self) -> None:
        self._generation += 1
        generation = self._generation
        metric = self.active_metric()
        scope = self.active_scope()
        document = self._document
        if scope is not StatisticsScope.PROJECT and document is None:
            self._snapshots = ()
            self._category_comparisons = ()
            self._render_empty(metric)
            self.statisticsChanged.emit(self._snapshots)
            return
        candidate_count = sum(
            len(item.measurements)
            for item in (
                self._project.documents
                if scope is StatisticsScope.PROJECT
                else ([document] if document is not None else [])
            )
        )
        self._render_selected_measurement(document)
        if candidate_count > self.BACKGROUND_THRESHOLD:
            self._category_comparisons = ()
            self._render_pending(metric)
            task = _StatisticsTask(
                generation=generation,
                signals=self._task_signals,
                project=copy.deepcopy(self._project),
                metric=metric,
                scope=scope,
                document_id=document.id if document is not None else None,
                fiber_group_id=document.active_group_id if document is not None else None,
            )
            QThreadPool.globalInstance().start(task)
            return
        try:
            self._snapshots = self._service.summarize(
                self._project,
                metric=metric,
                scope=scope,
                document_id=document.id if document is not None else None,
                fiber_group_id=document.active_group_id if document is not None else None,
            )
            comparison_documents = (
                self._project.documents
                if scope is StatisticsScope.PROJECT
                else ([document] if document is not None else [])
            )
            self._category_comparisons = self._service.summarize_by_category(
                comparison_documents,
                metric=metric,
                scope=scope,
            )
        except (KeyError, ValueError):
            self._snapshots = ()
            self._category_comparisons = ()
        if not self._snapshots:
            self._render_empty(metric)
        else:
            self._render_snapshot(self._preferred_snapshot(metric, document))
        self.statisticsChanged.emit(self._snapshots)

    @Slot(int, object, object)
    def _on_async_statistics_ready(
        self,
        generation: int,
        result: object,
        error: object,
    ) -> None:
        if generation != self._generation:
            return
        if error is None and isinstance(result, _StatisticsTaskResult):
            self._snapshots = result.snapshots
            self._category_comparisons = result.category_comparisons
        else:
            # Preserve the small direct-call seam used by focused UI tests and
            # older integrations which supplied the snapshot tuple itself.
            self._snapshots = tuple(result or ()) if error is None else ()
            self._category_comparisons = ()
        metric = self.active_metric()
        document = self._document
        if not self._snapshots:
            self._render_empty(metric)
            if error is not None:
                self.details_label.setText("统计计算失败；测量数据未被修改。")
        else:
            self._render_snapshot(self._preferred_snapshot(metric, document))
        self.statisticsChanged.emit(self._snapshots)

    def _preferred_snapshot(
        self,
        metric: MeasurementMetric,
        document: ImageDocument | None,
    ) -> MeasurementStatisticsSnapshot:
        if document is not None:
            expected = self._service.display_unit_for(document, metric)
            for snapshot in self._snapshots:
                if snapshot.unit == expected:
                    return snapshot
        return self._snapshots[0]

    def _render_selected_measurement(self, document: ImageDocument | None) -> None:
        measurement = self._selected_measurement
        if measurement is None:
            group = document.get_group(document.active_group_id) if document is not None else None
            category = group.display_name() if group is not None else "未分类"
            self.current_value_label.setText(f"当前类别：{category}")
            self._object_details_toggle.hide()
            self._object_details_label.hide()
            return
        unit = measurement.display_unit(document.calibration if document is not None else None)
        value = measurement.display_value()
        manual_modes = {
            "manual",
            "continuous_manual",
            "polygon_area",
            "freehand_area",
            "count",
        }
        confidence = "手工" if measurement.mode in manual_modes else f"{measurement.confidence:.0%}"
        if measurement.measurement_kind == "area":
            pixel_value = measurement.area_px
            pixel_unit = "px²"
        elif measurement.measurement_kind == "count":
            pixel_value = 1.0
            pixel_unit = "个"
        else:
            pixel_value = measurement.diameter_px
            pixel_unit = "px"
        group = document.get_group(measurement.fiber_group_id) if document is not None else None
        category = group.display_name() if group is not None else "未分类"
        mode_label = {
            "manual": "手动线段",
            "continuous_manual": "连续折线",
            "snap": "边缘吸附",
            "polygon_area": "多边形面积",
            "freehand_area": "自由圈选",
            "count": "手工计数",
            "magic_segment": "标准魔棒",
            "reference_propagation": "同类扩选",
            "fiber_quick": "快速测径",
        }.get(measurement.mode, measurement.mode)
        pixel_text = "—" if pixel_value is None else f"{pixel_value:.4g} {pixel_unit}"
        created = str(measurement.created_at or "—").replace("T", " ").removesuffix("Z")
        self.current_value_label.setText(
            f"当前结果  {value:.4g} {unit}\n"
            f"{category} · {self._kind_label(measurement.measurement_kind)} · {confidence}"
        )
        self._object_details_label.setText(
            f"物理值：{value:.4g} {unit}\n"
            f"像素值：{pixel_text}\n"
            f"类别：{category}\n"
            f"模式：{mode_label}\n"
            f"状态：{measurement.status}\n"
            f"创建时间：{created}"
        )
        self._object_details_toggle.show()
        self._object_details_label.setVisible(self._object_details_toggle.isChecked())

    def _toggle_object_details(self, expanded: bool) -> None:
        self._object_details_toggle.setArrowType(
            Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow
        )
        self._object_details_label.setVisible(bool(expanded and self._selected_measurement is not None))

    @staticmethod
    def _kind_label(kind: str) -> str:
        return {"line": "线段", "polyline": "折线", "area": "面积", "count": "计数"}.get(kind, kind)

    def _render_empty(self, metric: MeasurementMetric) -> None:
        self._n_cell.value_label.setText("0")
        self._mean_cell.value_label.setText("—")
        self._std_cell.value_label.setText("—")
        self._cv_cell.value_label.setText("—")
        self.details_label.setText(f"当前作用域没有有效的{self._metric_label(metric)}结果。")
        self.quality_label.setText("有效 0 · 需复核 0 · 失败 0")

    def _render_pending(self, metric: MeasurementMetric) -> None:
        self._n_cell.value_label.setText("…")
        self._mean_cell.value_label.setText("…")
        self._std_cell.value_label.setText("…")
        self._cv_cell.value_label.setText("…")
        self.details_label.setText(f"正在后台计算{self._metric_label(metric)}统计…")
        self.quality_label.setText("数据变化时，晚到的旧统计结果会自动丢弃。")

    def _render_snapshot(self, snapshot: MeasurementStatisticsSnapshot) -> None:
        suffix = "" if snapshot.metric is MeasurementMetric.COUNT else f" {snapshot.unit}"
        self._n_cell.value_label.setText(str(snapshot.valid_count))
        self._mean_cell.value_label.setText(self._format_value(snapshot.mean, suffix=suffix))
        self._std_cell.value_label.setText(self._format_value(snapshot.stddev, suffix=suffix))
        self._cv_cell.value_label.setText(self._format_value(snapshot.cv_percent, suffix="%"))
        if snapshot.valid_count:
            details = (
                f"中位数 {self._format_value(snapshot.median, suffix=suffix)}\n"
                f"范围 {self._format_value(snapshot.minimum, suffix=suffix)} – "
                f"{self._format_value(snapshot.maximum, suffix=suffix)}\n"
                f"P10 / P90  {self._format_value(snapshot.p10, suffix=suffix)} / "
                f"{self._format_value(snapshot.p90, suffix=suffix)}"
            )
            if snapshot.metric is MeasurementMetric.AREA:
                details += f"\n对象净面积合计 {self._format_value(snapshot.total_value, suffix=suffix)}"
            elif snapshot.metric is MeasurementMetric.COUNT:
                details = f"计数合计 {snapshot.valid_count} 个"
            if len(self._snapshots) > 1:
                details += f"\n项目包含 {len(self._snapshots)} 种单位，已分开统计"
            self.details_label.setText(details)
        else:
            self.details_label.setText("当前作用域没有可用于数值统计的结果。")
        self.quality_label.setText(
            f"有效 {snapshot.valid_count} · 需复核 {snapshot.manual_review_count} · "
            f"失败 {snapshot.hard_failure_count} · 无效值 {snapshot.non_finite_count + snapshot.missing_value_count}"
        )

    @staticmethod
    def _format_value(value: float | None, *, suffix: str = "") -> str:
        if value is None:
            return "—"
        return f"{value:.4g}{suffix}"

    @staticmethod
    def _metric_label(metric: MeasurementMetric) -> str:
        return {
            MeasurementMetric.LENGTH: "长度",
            MeasurementMetric.AREA: "面积",
            MeasurementMetric.COUNT: "计数",
        }[metric]

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QWidget#measurementStatisticsPanel { background: transparent; }
            QLabel[panelTitle="true"] { font-weight: 700; font-size: 14px; }
            QFrame#statisticsMetricCell {
                border: 1px solid palette(mid);
                border-radius: 6px;
                background: palette(alternate-base);
            }
            QLabel[statisticsCaption="true"] { color: palette(placeholder-text); }
            QLabel[statisticsValue="true"] { font-size: 15px; font-weight: 700; }
            QLabel#currentMeasurementValue {
                padding: 8px;
                border-left: 3px solid #2A9D8F;
                background: palette(alternate-base);
            }
            QLabel#statisticsDetails, QLabel#statisticsQuality { color: palette(placeholder-text); }
            """
        )


class StatisticsDistributionWidget(QWidget):
    """Small dependency-free histogram and box plot for the active snapshot."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._snapshot: MeasurementStatisticsSnapshot | None = None
        # The results dock must remain usable at the supported 1093x576
        # logical viewport.  The plot scales its own drawing geometry, so a
        # compact minimum is preferable to forcing the main window taller.
        self.setMinimumHeight(96)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_snapshot(self, snapshot: MeasurementStatisticsSnapshot | None) -> None:
        self._snapshot = snapshot
        self.update()

    def paintEvent(self, event) -> None:
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        palette = self.palette()
        text = palette.color(QPalette.ColorRole.Text)
        muted = palette.color(QPalette.ColorRole.PlaceholderText)
        border = palette.color(QPalette.ColorRole.Mid)
        accent = QColor("#2A9D8F")
        rect = QRectF(self.rect()).adjusted(12, 10, -12, -10)
        painter.setPen(text)
        snapshot = self._snapshot
        if snapshot is None or snapshot.valid_count < 2 or not snapshot.histogram_counts:
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "至少需要两个有效结果才能显示分布。")
            return

        painter.drawText(QRectF(rect.left(), rect.top(), rect.width(), 20), "直方图")
        histogram = QRectF(rect.left(), rect.top() + 26, rect.width(), max(60.0, rect.height() * 0.55))
        painter.setPen(QPen(border, 1))
        painter.drawLine(histogram.bottomLeft(), histogram.bottomRight())
        maximum_count = max(snapshot.histogram_counts) or 1
        bar_width = histogram.width() / len(snapshot.histogram_counts)
        for index, count in enumerate(snapshot.histogram_counts):
            height = histogram.height() * (count / maximum_count)
            bar = QRectF(
                histogram.left() + index * bar_width + 1,
                histogram.bottom() - height,
                max(1.0, bar_width - 2),
                height,
            )
            painter.fillRect(bar, QColor(accent.red(), accent.green(), accent.blue(), 190))

        box_top = histogram.bottom() + 24
        box_rect = QRectF(rect.left(), box_top, rect.width(), max(36.0, rect.bottom() - box_top))
        painter.setPen(muted)
        painter.drawText(QRectF(box_rect.left(), box_rect.top(), box_rect.width(), 18), "箱线图（1.5×IQR）")
        minimum = snapshot.minimum
        maximum = snapshot.maximum
        if minimum is None or maximum is None or maximum <= minimum:
            return

        axis = QRectF(box_rect.left() + 8, box_rect.top() + 24, box_rect.width() - 16, 24)

        def x_for(value: float | None) -> float:
            if value is None:
                return axis.left()
            return axis.left() + ((value - minimum) / (maximum - minimum)) * axis.width()

        median_x = x_for(snapshot.median)
        q1_x = x_for(snapshot.q1)
        q3_x = x_for(snapshot.q3)
        lower_whisker_x = x_for(snapshot.lower_whisker)
        upper_whisker_x = x_for(snapshot.upper_whisker)
        center_y = axis.center().y()
        painter.setPen(QPen(border, 1.5))
        painter.drawLine(lower_whisker_x, center_y, upper_whisker_x, center_y)
        cap_half_height = axis.height() * 0.24
        painter.drawLine(
            lower_whisker_x,
            center_y - cap_half_height,
            lower_whisker_x,
            center_y + cap_half_height,
        )
        painter.drawLine(
            upper_whisker_x,
            center_y - cap_half_height,
            upper_whisker_x,
            center_y + cap_half_height,
        )
        painter.fillRect(QRectF(q1_x, axis.top() + 3, max(2.0, q3_x - q1_x), axis.height() - 6), QColor(accent.red(), accent.green(), accent.blue(), 85))
        painter.drawRect(QRectF(q1_x, axis.top() + 3, max(2.0, q3_x - q1_x), axis.height() - 6))
        painter.setPen(QPen(accent, 2))
        painter.drawLine(median_x, axis.top() + 2, median_x, axis.bottom() - 2)
        painter.setBrush(accent)
        for value in snapshot.outlier_values:
            painter.drawEllipse(QRectF(x_for(value) - 3, center_y - 3, 6, 6))
