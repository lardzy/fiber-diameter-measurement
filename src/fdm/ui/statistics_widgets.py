from __future__ import annotations

import copy
from dataclasses import dataclass

from PySide6.QtCore import (
    QObject,
    QRunnable,
    QThreadPool,
    Qt,
    Signal,
    Slot,
)
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
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
from fdm.ui.statistics_distribution import (
    DistributionRecordFilterRequest as DistributionRecordFilterRequest,
    StatisticsDistributionWidget,
)
from fdm.ui.widgets import NoWheelComboBox


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
    """Compact live statistics and the latest selected measurement result."""

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
        self.metric_combo = NoWheelComboBox(self)
        self.metric_combo.addItem("自动", None)
        self.metric_combo.addItem("长度", MeasurementMetric.LENGTH)
        self.metric_combo.addItem("面积", MeasurementMetric.AREA)
        self.metric_combo.addItem("计数", MeasurementMetric.COUNT)
        self.metric_combo.setMinimumWidth(0)
        self.metric_combo.setMinimumContentsLength(4)
        self.metric_combo.setSizeAdjustPolicy(
            NoWheelComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.metric_combo.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.metric_combo.currentIndexChanged.connect(self.refresh)
        self.scope_combo = NoWheelComboBox(self)
        self.scope_combo.addItem("当前类别", StatisticsScope.CURRENT_CATEGORY)
        self.scope_combo.addItem("当前图片", StatisticsScope.CURRENT_DOCUMENT)
        self.scope_combo.addItem("整个项目", StatisticsScope.PROJECT)
        self.scope_combo.setMinimumWidth(0)
        self.scope_combo.setMinimumContentsLength(4)
        self.scope_combo.setSizeAdjustPolicy(
            NoWheelComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
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
        group = document.get_group(measurement.fiber_group_id) if document is not None else None
        category = group.display_name() if group is not None else "未分类"
        self.current_value_label.setText(
            f"当前结果  {value:.4g} {unit}\n"
            f"{category} · {self._kind_label(measurement.measurement_kind)} · {confidence}"
        )

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
