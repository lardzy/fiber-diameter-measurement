from __future__ import annotations

from datetime import datetime
from enum import IntEnum
from functools import lru_cache
import math

from PySide6.QtCore import QAbstractTableModel, QModelIndex, QSortFilterProxyModel, Qt, Signal
from PySide6.QtGui import QColor, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import QAbstractItemDelegate, QComboBox, QStyledItemDelegate, QWidget

from fdm.area_display import area_derived_geometry_service
from fdm.models import ImageDocument, Measurement, UNCATEGORIZED_LABEL
from fdm.services.measurement_statistics import MeasurementStatisticsService
from fdm.ui.widgets import MeasurementGroupComboBox


class MeasurementResultColumn(IntEnum):
    RESULT_SEQUENCE = 0
    CATEGORY_SEQUENCE = 1
    GROUP = 2
    KIND = 3
    RESULT = 4
    UNIT = 5
    HOLE_AREA = 6
    MODE = 7
    CONFIDENCE = 8
    STATUS = 9
    CREATED_AT = 10
    ID = 11


MEASUREMENT_ID_ROLE = int(Qt.ItemDataRole.UserRole) + 1
GROUP_ID_ROLE = int(Qt.ItemDataRole.UserRole) + 2
GROUP_FILTER_ROLE = int(Qt.ItemDataRole.UserRole) + 3
RAW_KIND_ROLE = int(Qt.ItemDataRole.UserRole) + 4
RAW_STATUS_ROLE = int(Qt.ItemDataRole.UserRole) + 5
SORT_ROLE = int(Qt.ItemDataRole.UserRole) + 6


_HEADERS = (
    "纤维结果序号",
    "纤维类别结果序号",
    "纤维类别",
    "类型",
    "结果",
    "单位",
    "孔洞面积",
    "模式",
    "置信度",
    "状态",
    "创建时间",
    "ID",
)
_MANUAL_MODES = {"manual", "continuous_manual", "count", "polygon_area", "freehand_area"}


def format_measurement_kind(measurement: Measurement) -> str:
    return {
        "line": "线段",
        "polyline": "折线",
        "area": "面积",
        "count": "计数点",
    }.get(measurement.measurement_kind, measurement.measurement_kind)


def format_measurement_mode(mode: str) -> str:
    return {
        "manual": "手动线段",
        "continuous_manual": "连续测量",
        "count": "计数",
        "snap": "边缘吸附",
        "fiber_auto": "快速测径",
        "fiber_quick": "快速测径",
        "polygon_area": "多边形面积",
        "freehand_area": "自由形状面积",
        "magic_segment": "魔棒分割",
        "auto_instance": "实例分割",
        "reference_instance": "同类扩选",
    }.get(mode, mode)


def format_measurement_status(status: str) -> str:
    return {
        "manual": "手动测量",
        "continuous_manual": "连续测量",
        "ready": "已完成",
        "manual_review": "需人工复核",
        "snapped": "吸附成功",
        "edited": "已编辑",
        "line_too_short": "测量线过短",
        "profile_too_flat": "灰度变化不足",
        "edge_pair_not_found": "未找到有效边缘",
        "component_not_found": "未找到目标区域",
        "centerline_not_found": "未找到可靠中心线",
        "boundary_not_found": "未找到边界",
        "fiber_auto": "快速测径",
        "fiber_quick": "快速测径",
        "count": "计数",
        "auto_instance": "自动识别",
        "reference_instance": "同类扩选",
    }.get(status, status)


def _measurement_hole_area(
    document: ImageDocument,
    measurement: Measurement,
) -> tuple[float, str] | None:
    if measurement.measurement_kind != "area":
        return None
    value = area_derived_geometry_service.scalar_geometry(measurement).hole_area_px
    calibration = document.calibration
    if calibration is None:
        return value, "px²"
    return calibration.px_area_to_unit(value), f"{calibration.unit}²"


def _parse_created_at(value: str) -> datetime | None:
    token = str(value or "").strip()
    if not token:
        return None
    if token.endswith("Z"):
        token = f"{token[:-1]}+00:00"
    try:
        return datetime.fromisoformat(token)
    except ValueError:
        return None


def _format_created_at(value: str) -> str:
    parsed = _parse_created_at(value)
    if parsed is None:
        return str(value or "") or "—"
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone()
    return parsed.strftime("%Y-%m-%d %H:%M:%S")


def _created_at_sort_value(value: str) -> float | str:
    parsed = _parse_created_at(value)
    if parsed is None:
        return str(value or "").casefold()
    if parsed.tzinfo is None:
        parsed = parsed.astimezone()
    return parsed.timestamp()


class MeasurementResultsModel(QAbstractTableModel):
    """Twelve-column measurement model backed by one image document.

    The model keeps its own ordered reference list so incremental inserts obey
    Qt's model-change contract even though the domain document is mutated first.
    Domain edits are requested by stable measurement ID and remain owned by the
    main-window history/dirty-state coordinator.
    """

    groupChangeRequested = Signal(str, object)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._document: ImageDocument | None = None
        self._measurements: list[Measurement] = []
        self._row_by_id: dict[str, int] = {}
        self._result_sequence_by_id: dict[str, int] = {}
        self._category_sequence_by_id: dict[str, int] = {}
        self._sequence_totals = {}
        self._category_totals = {}

    @property
    def document(self) -> ImageDocument | None:
        return self._document

    def set_document(self, document: ImageDocument | None) -> None:
        self.beginResetModel()
        self._document = document
        self._measurements = list(document.measurements) if document is not None else []
        self._row_by_id = {
            measurement.id: row
            for row, measurement in enumerate(self._measurements)
        }
        self._rebuild_sequences()
        self.endResetModel()

    def append_measurement(self, document: ImageDocument, measurement: Measurement) -> bool:
        """Append one already-committed domain measurement without a model reset."""

        if document is not self._document:
            return False
        if len(document.measurements) != len(self._measurements) + 1:
            return False
        if not document.measurements or document.measurements[-1] is not measurement:
            return False
        if measurement.id in self._row_by_id:
            return False
        row = len(self._measurements)
        self.beginInsertRows(QModelIndex(), row, row)
        self._measurements.append(measurement)
        self._row_by_id[measurement.id] = row
        self._append_sequence(measurement)
        self.endInsertRows()
        return True

    def refresh_measurements(self, measurement_ids) -> None:
        for measurement_id in measurement_ids:
            row = self._row_by_id.get(measurement_id)
            if row is not None:
                self.dataChanged.emit(
                    self.index(row, 0), self.index(row, self.columnCount() - 1), []
                )

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._measurements)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else len(_HEADERS)

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.ItemDataRole.DisplayRole):
        if role in {Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.ToolTipRole} and orientation == Qt.Orientation.Horizontal:
            if 0 <= section < len(_HEADERS):
                return _HEADERS[section]
        return super().headerData(section, orientation, role)

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole):
        measurement = self.measurement_at(index.row()) if index.isValid() else None
        document = self._document
        if measurement is None or document is None:
            return None
        column = MeasurementResultColumn(index.column())
        group = document.get_group(measurement.fiber_group_id)
        group_display = group.display_name() if group is not None else UNCATEGORIZED_LABEL
        group_filter = (
            (group.label.strip() or group.display_name())
            if group is not None
            else UNCATEGORIZED_LABEL
        )

        if role in {Qt.ItemDataRole.UserRole, MEASUREMENT_ID_ROLE}:
            return measurement.id
        if role == GROUP_ID_ROLE:
            return measurement.fiber_group_id
        if role == GROUP_FILTER_ROLE:
            return group_filter
        if role == RAW_KIND_ROLE:
            return measurement.measurement_kind
        if role == RAW_STATUS_ROLE:
            return measurement.status
        if role == Qt.ItemDataRole.DecorationRole and column is MeasurementResultColumn.GROUP:
            return _color_icon(group.color if group is not None else "#98A2B3")
        if role == Qt.ItemDataRole.TextAlignmentRole:
            if column in {
                MeasurementResultColumn.RESULT_SEQUENCE,
                MeasurementResultColumn.CATEGORY_SEQUENCE,
                MeasurementResultColumn.RESULT,
                MeasurementResultColumn.HOLE_AREA,
                MeasurementResultColumn.CONFIDENCE,
            }:
                return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            return int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        if role == Qt.ItemDataRole.ToolTipRole:
            if column is MeasurementResultColumn.ID:
                return measurement.id
            if column is MeasurementResultColumn.CREATED_AT:
                return measurement.created_at
            if column is MeasurementResultColumn.HOLE_AREA:
                hole_area = _measurement_hole_area(document, measurement)
                if hole_area is not None:
                    value, unit = hole_area
                    return f"{value:.6g} {unit}"
        if role == SORT_ROLE:
            return self._sort_value(column, document, measurement, group_display)
        if role not in {Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole}:
            return None
        if role == Qt.ItemDataRole.EditRole and column is MeasurementResultColumn.GROUP:
            return measurement.fiber_group_id
        return self._display_value(
            column,
            document,
            measurement,
            group_display,
            self._result_sequence_by_id.get(measurement.id, 0),
            self._category_sequence_by_id.get(measurement.id, 0),
        )

    def flags(self, index: QModelIndex) -> Qt.ItemFlags:
        flags = super().flags(index)
        if index.isValid() and index.column() == MeasurementResultColumn.GROUP:
            flags |= Qt.ItemFlag.ItemIsEditable
        return flags

    def setData(self, index: QModelIndex, value, role: int = Qt.ItemDataRole.EditRole) -> bool:
        if (
            not index.isValid()
            or index.column() != MeasurementResultColumn.GROUP
            or role != Qt.ItemDataRole.EditRole
            or self._document is None
        ):
            return False
        measurement = self.measurement_at(index.row())
        if measurement is None:
            return False
        target_group_id = None if value in {None, ""} else str(value)
        if target_group_id is not None and self._document.get_group(target_group_id) is None:
            return False
        if measurement.fiber_group_id == target_group_id:
            return False
        self.groupChangeRequested.emit(measurement.id, target_group_id)
        return True

    def measurement_at(self, source_row: int) -> Measurement | None:
        if 0 <= source_row < len(self._measurements):
            return self._measurements[source_row]
        return None

    def measurement_id_at(self, source_row: int) -> str | None:
        measurement = self.measurement_at(source_row)
        return measurement.id if measurement is not None else None

    def source_row_for_id(self, measurement_id: str | None) -> int:
        if not measurement_id:
            return -1
        return self._row_by_id.get(measurement_id, -1)

    def group_options(self) -> tuple[tuple[str, str | None, str], ...]:
        document = self._document
        if document is None:
            return ((UNCATEGORIZED_LABEL, None, "#98A2B3"),)
        return (
            (UNCATEGORIZED_LABEL, None, "#98A2B3"),
            *tuple(
                (group.display_name(), group.id, group.color)
                for group in document.sorted_groups()
            ),
        )

    def _rebuild_sequences(self) -> None:
        self._result_sequence_by_id.clear()
        self._category_sequence_by_id.clear()
        self._sequence_totals.clear()
        self._category_totals.clear()
        document = self._document
        if document is None:
            return
        for measurement in self._measurements:
            self._append_sequence(measurement)

    def _append_sequence(self, measurement):
        kind = measurement.measurement_kind or ""
        group = self._document.get_group(measurement.fiber_group_id)
        category = (kind, group.label if group is not None else UNCATEGORIZED_LABEL)
        self._sequence_totals[kind] = self._sequence_totals.get(kind, 0) + 1
        self._category_totals[category] = self._category_totals.get(category, 0) + 1
        self._result_sequence_by_id[measurement.id] = self._sequence_totals[kind]
        self._category_sequence_by_id[measurement.id] = self._category_totals[category]

    @staticmethod
    def _display_value(
        column: MeasurementResultColumn,
        document: ImageDocument,
        measurement: Measurement,
        group_display: str,
        result_sequence: int,
        category_sequence: int,
    ):
        if column is MeasurementResultColumn.RESULT_SEQUENCE:
            return result_sequence
        if column is MeasurementResultColumn.CATEGORY_SEQUENCE:
            return category_sequence
        if column is MeasurementResultColumn.GROUP:
            return group_display
        if column is MeasurementResultColumn.KIND:
            return format_measurement_kind(measurement)
        if column is MeasurementResultColumn.RESULT:
            return f"{measurement.display_value():.4f}"
        if column is MeasurementResultColumn.UNIT:
            return measurement.display_unit(document.calibration)
        if column is MeasurementResultColumn.HOLE_AREA:
            hole_area = _measurement_hole_area(document, measurement)
            return "—" if hole_area is None else f"{hole_area[0]:.4f}"
        if column is MeasurementResultColumn.MODE:
            return format_measurement_mode(measurement.mode)
        if column is MeasurementResultColumn.CONFIDENCE:
            return "手工" if measurement.mode in _MANUAL_MODES else f"{measurement.confidence:.2f}"
        if column is MeasurementResultColumn.STATUS:
            return format_measurement_status(measurement.status)
        if column is MeasurementResultColumn.CREATED_AT:
            return _format_created_at(measurement.created_at)
        return measurement.id

    def _sort_value(
        self,
        column: MeasurementResultColumn,
        document: ImageDocument,
        measurement: Measurement,
        group_display: str,
    ):
        if column is MeasurementResultColumn.RESULT_SEQUENCE:
            return self._result_sequence_by_id.get(measurement.id, 0)
        if column is MeasurementResultColumn.CATEGORY_SEQUENCE:
            return self._category_sequence_by_id.get(measurement.id, 0)
        if column is MeasurementResultColumn.GROUP:
            return group_display.casefold()
        if column is MeasurementResultColumn.KIND:
            return format_measurement_kind(measurement).casefold()
        if column is MeasurementResultColumn.RESULT:
            value = float(measurement.display_value())
            return value if math.isfinite(value) else None
        if column is MeasurementResultColumn.UNIT:
            return measurement.display_unit(document.calibration).casefold()
        if column is MeasurementResultColumn.HOLE_AREA:
            hole_area = _measurement_hole_area(document, measurement)
            return None if hole_area is None else hole_area[0]
        if column is MeasurementResultColumn.MODE:
            return format_measurement_mode(measurement.mode).casefold()
        if column is MeasurementResultColumn.CONFIDENCE:
            return measurement.confidence
        if column is MeasurementResultColumn.STATUS:
            return format_measurement_status(measurement.status).casefold()
        if column is MeasurementResultColumn.CREATED_AT:
            return _created_at_sort_value(measurement.created_at)
        return measurement.id.casefold()


class MeasurementResultsProxyModel(QSortFilterProxyModel):
    """Combined text/type/category/status filtering with stable source mapping."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._query = ""
        self._kind_filter = ""
        self._group_filter = ""
        self._status_filter = ""
        self.setDynamicSortFilter(True)
        self.setSortRole(SORT_ROLE)
        self.setSortCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)

    def set_filters(
        self,
        *,
        query: str = "",
        kind: str = "",
        group: str = "",
        status: str = "",
    ) -> None:
        normalized = (
            str(query or "").strip().casefold(),
            str(kind or ""),
            str(group or ""),
            str(status or ""),
        )
        current = (self._query, self._kind_filter, self._group_filter, self._status_filter)
        if normalized == current:
            return
        self.beginFilterChange()
        self._query, self._kind_filter, self._group_filter, self._status_filter = normalized
        self.endFilterChange(QSortFilterProxyModel.Direction.Rows)

    def filterAcceptsRow(self, source_row: int, source_parent: QModelIndex) -> bool:
        source = self.sourceModel()
        if not isinstance(source, MeasurementResultsModel):
            return False
        measurement = source.measurement_at(source_row)
        document = source.document
        if measurement is None or document is None:
            return False
        group = document.get_group(measurement.fiber_group_id)
        group_display = group.display_name() if group is not None else UNCATEGORIZED_LABEL
        group_filter_label = (
            (group.label.strip() or group.display_name())
            if group is not None
            else UNCATEGORIZED_LABEL
        )
        if self._query:
            searchable = " ".join(
                (
                    group_display,
                    group_filter_label,
                    format_measurement_kind(measurement),
                    format_measurement_mode(measurement.mode),
                    format_measurement_status(measurement.status),
                    measurement.id,
                    measurement.id.split("_")[-1],
                )
            ).casefold()
            if self._query not in searchable:
                return False
        if self._kind_filter:
            if self._kind_filter == "length":
                if measurement.measurement_kind not in {"line", "polyline"}:
                    return False
            elif measurement.measurement_kind != self._kind_filter:
                return False
        if self._group_filter and group_filter_label.casefold() != self._group_filter.casefold():
            return False
        if self._status_filter:
            hard_failures = MeasurementStatisticsService.DEFAULT_HARD_FAILURE_STATUSES
            if self._status_filter == "review":
                if measurement.status != MeasurementStatisticsService.MANUAL_REVIEW_STATUS:
                    return False
            elif self._status_filter == "failed":
                if measurement.status not in hard_failures:
                    return False
            elif self._status_filter == "valid":
                if (
                    measurement.status in hard_failures
                    or measurement.status == MeasurementStatisticsService.MANUAL_REVIEW_STATUS
                ):
                    return False
        return True

    def lessThan(self, left: QModelIndex, right: QModelIndex) -> bool:
        left_value = left.data(SORT_ROLE)
        right_value = right.data(SORT_ROLE)
        if left_value is None:
            return False
        if right_value is None:
            return True
        if isinstance(left_value, (int, float)) and isinstance(right_value, (int, float)):
            return float(left_value) < float(right_value)
        return str(left_value).casefold() < str(right_value).casefold()


class MeasurementGroupDelegate(QStyledItemDelegate):
    """Creates the category combo only while the category cell is edited."""

    def createEditor(self, parent: QWidget, option, index: QModelIndex) -> QWidget | None:
        del option
        source, _source_index = _source_model_and_index(index)
        if source is None:
            return None
        editor = MeasurementGroupComboBox(parent)
        editor.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        for label, group_id, color in source.group_options():
            editor.addItem(_color_icon(color), label, group_id)
        editor.activated.connect(lambda _row, widget=editor: self._commit_and_close(widget))
        return editor

    def setEditorData(self, editor: QWidget, index: QModelIndex) -> None:
        if not isinstance(editor, QComboBox):
            return
        current_group_id = index.data(GROUP_ID_ROLE)
        target = editor.findData(current_group_id)
        editor.setCurrentIndex(0 if target < 0 else target)

    def setModelData(self, editor: QWidget, model, index: QModelIndex) -> None:
        if isinstance(editor, QComboBox):
            model.setData(index, editor.currentData(), Qt.ItemDataRole.EditRole)

    def _commit_and_close(self, editor: QWidget) -> None:
        self.commitData.emit(editor)
        self.closeEditor.emit(editor, QAbstractItemDelegate.EndEditHint.NoHint)


def _source_model_and_index(index: QModelIndex) -> tuple[MeasurementResultsModel | None, QModelIndex]:
    model = index.model()
    if isinstance(model, MeasurementResultsProxyModel):
        source_index = model.mapToSource(index)
        source = model.sourceModel()
        return (source if isinstance(source, MeasurementResultsModel) else None), source_index
    return (model if isinstance(model, MeasurementResultsModel) else None), index


@lru_cache(maxsize=128)
def _color_icon(color: str) -> QIcon:
    pixmap = QPixmap(14, 14)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setPen(QColor(color).darker(130))
    painter.setBrush(QColor(color))
    painter.drawEllipse(2, 2, 10, 10)
    painter.end()
    return QIcon(pixmap)
