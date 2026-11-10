from __future__ import annotations

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
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from fdm.models import ImageDocument, Measurement, ProjectState
from fdm.services.measurement_statistics import (
    MeasurementMetric,
    MeasurementStatisticsService,
    MeasurementStatisticsSnapshot,
    StatisticsInputSnapshot,
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
        input_snapshot: StatisticsInputSnapshot,
        metric: MeasurementMetric,
        scope: StatisticsScope,
        document_id: str | None,
        fiber_group_id: str | None,
    ) -> None:
        super().__init__()
        self._generation = generation
        self._signals = signals
        self._input_snapshot = input_snapshot
        self._metric = metric
        self._scope = scope
        self._document_id = document_id
        self._fiber_group_id = fiber_group_id

    @Slot()
    def run(self) -> None:
        try:
            service = MeasurementStatisticsService()
            snapshots = service.summarize_input(
                self._input_snapshot,
                metric=self._metric,
                scope=self._scope,
                document_id=self._document_id,
                fiber_group_id=self._fiber_group_id,
            )
            if self._scope is StatisticsScope.PROJECT:
                category_documents = self._input_snapshot.documents
            else:
                selected_document = self._input_snapshot.get_document(self._document_id or "")
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


class MeasurementStatisticsPanel(QWidget):
    """Compact live statistics and the latest selected measurement result."""

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
        root.setSpacing(3)
        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("指标", self))
        self.metric_combo = NoWheelComboBox(self)
        self.metric_combo.addItem("自动", None)
        self.metric_combo.addItem("长度", MeasurementMetric.LENGTH)
        self.metric_combo.addItem("面积", MeasurementMetric.AREA)
        self.metric_combo.addItem("计数", MeasurementMetric.COUNT)
        self.metric_combo.setAccessibleName("统计指标")
        self.metric_combo.currentIndexChanged.connect(self.refresh)
        self.scope_combo = NoWheelComboBox(self)
        self.scope_combo.addItem("当前类别", StatisticsScope.CURRENT_CATEGORY)
        self.scope_combo.addItem("当前图片", StatisticsScope.CURRENT_DOCUMENT)
        self.scope_combo.addItem("整个项目", StatisticsScope.PROJECT)
        self.scope_combo.setAccessibleName("统计范围")
        self.scope_combo.currentIndexChanged.connect(self.refresh)
        filter_row.addWidget(self.metric_combo)
        filter_row.addWidget(QLabel("范围", self))
        filter_row.addWidget(self.scope_combo)

        self.current_value_label = QLabel(self)
        self.current_value_label.setTextFormat(Qt.TextFormat.PlainText)
        self.current_value_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.current_value_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        filter_row.addWidget(self.current_value_label, 1)
        root.addLayout(filter_row)

        self.details_label = QLabel(self)
        self.details_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.details_label.setWordWrap(True)
        self.details_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        root.addWidget(self.details_label)

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
            self.current_value_label.clear()
            self._publish_statistics()
            return
        comparison_documents = (
            tuple(self._project.documents)
            if scope is StatisticsScope.PROJECT
            else ((document,) if document is not None else ())
        )
        candidate_count = sum(len(item.measurements) for item in comparison_documents)
        self._render_selected_measurement(document)
        if candidate_count > self.BACKGROUND_THRESHOLD:
            self._category_comparisons = ()
            self._snapshots = ()
            self._publish_statistics(f"正在后台计算{self._metric_label(metric)}统计…")
            task = _StatisticsTask(
                generation=generation,
                signals=self._task_signals,
                input_snapshot=StatisticsInputSnapshot.from_documents(
                    comparison_documents
                ),
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
            self._category_comparisons = self._service.summarize_by_category(
                comparison_documents,
                metric=metric,
                scope=scope,
            )
        except (KeyError, ValueError):
            self._snapshots = ()
            self._category_comparisons = ()
        self._publish_statistics()

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
        self._publish_statistics("统计计算失败。" if error is not None else "")

    def _publish_statistics(self, message: str = "") -> None:
        metric_label = self._metric_label(self.active_metric())
        if not message:
            if self._snapshots:
                comparison_scope = "整个项目" if self.active_scope() is StatisticsScope.PROJECT else "当前图片"
                message = f"{metric_label} · {self.scope_combo.currentText()} · 总体标准差口径 · 类别比较范围：{comparison_scope}"
                if len(self._snapshots) > 1:
                    message += " · 多种单位已分开展示"
            else:
                message = f"当前作用域没有有效的{metric_label}结果。"
        self.details_label.setText(message)
        self.statisticsChanged.emit(self._snapshots)

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
            f"当前结果  {value:.4g} {unit} · "
            f"{category} · {self._kind_label(measurement.measurement_kind)} · {confidence}"
        )

    @staticmethod
    def _kind_label(kind: str) -> str:
        return {"line": "线段", "polyline": "折线", "area": "面积", "count": "计数"}.get(kind, kind)

    @staticmethod
    def _metric_label(metric: MeasurementMetric) -> str:
        return {
            MeasurementMetric.LENGTH: "长度",
            MeasurementMetric.AREA: "面积",
            MeasurementMetric.COUNT: "计数",
        }[metric]
