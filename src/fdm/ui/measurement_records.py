from __future__ import annotations

from contextlib import contextmanager

from PySide6.QtCore import (
    QByteArray,
    QItemSelectionModel,
    QModelIndex,
    QObject,
    QPoint,
    Qt,
    Signal,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMenu,
    QPushButton,
    QSizePolicy,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from fdm.models import ImageDocument, Measurement, UNCATEGORIZED_LABEL
from fdm.ui.measurement_results_model import (
    MEASUREMENT_ID_ROLE,
    MeasurementGroupDelegate,
    MeasurementResultColumn,
    MeasurementResultsModel,
    MeasurementResultsProxyModel,
)
from fdm.ui.widgets import FlowLayout


class MeasurementRecordsController(QObject):
    """Shared record state for any number of synchronized table views.

    The domain document still belongs to the main-window coordinator.  This
    controller only owns view state: one source/proxy pair, one selection model,
    shared filters and shared sorting.  Each pane deliberately owns its own
    header layout so a compact inspector can coexist with a wide result drawer.
    """

    documentChanged = Signal(object)
    filtersChanged = Signal(str, str, str, str)
    countsChanged = Signal(int, int)
    sortChanged = Signal(int, int)
    groupChangeRequested = Signal(str, object)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.source_model = MeasurementResultsModel(self)
        self.proxy_model = MeasurementResultsProxyModel(self)
        self.proxy_model.setSourceModel(self.source_model)
        self.selection_model = QItemSelectionModel(self.proxy_model, self)
        self._query = ""
        self._kind = ""
        self._group = ""
        self._status = ""
        self._sort_column = -1
        self._sort_order = Qt.SortOrder.AscendingOrder
        self.source_model.modelReset.connect(self._emit_counts)
        self.proxy_model.modelReset.connect(self._emit_counts)
        self.proxy_model.rowsInserted.connect(self._emit_counts)
        self.proxy_model.rowsRemoved.connect(self._emit_counts)
        self.source_model.groupChangeRequested.connect(self.groupChangeRequested.emit)

    @property
    def model(self) -> MeasurementResultsModel:
        """Compatibility alias for integrations that call the source `model`."""

        return self.source_model

    @property
    def proxy(self) -> MeasurementResultsProxyModel:
        """Compatibility alias for integrations that call the proxy `proxy`."""

        return self.proxy_model

    @property
    def document(self) -> ImageDocument | None:
        return self.source_model.document

    @property
    def filters(self) -> tuple[str, str, str, str]:
        return self._query, self._kind, self._group, self._status

    @property
    def sort_state(self) -> tuple[int, Qt.SortOrder]:
        return self._sort_column, self._sort_order

    def set_document(self, document: ImageDocument | None) -> None:
        self.selection_model.clear()
        self.source_model.set_document(document)
        available_groups = {label.casefold() for label in self.group_labels()}
        if self._group and self._group.casefold() not in available_groups:
            self.set_filters(
                query=self._query,
                kind=self._kind,
                group="",
                status=self._status,
            )
        self.documentChanged.emit(document)
        self._emit_counts()

    def append_measurement(self, document: ImageDocument, measurement: Measurement) -> bool:
        appended = self.source_model.append_measurement(document, measurement)
        if appended:
            self.documentChanged.emit(document)
            self._emit_counts()
        return appended

    def group_labels(self) -> tuple[str, ...]:
        document = self.document
        if document is None:
            return ()
        labels: list[str] = []
        seen: set[str] = set()
        for measurement in document.measurements:
            group = document.get_group(measurement.fiber_group_id)
            label = (
                (group.label.strip() or group.display_name())
                if group is not None
                else UNCATEGORIZED_LABEL
            )
            key = label.casefold()
            if key not in seen:
                seen.add(key)
                labels.append(label)
        return tuple(labels)

    def set_filters(
        self,
        *,
        query: str = "",
        kind: str = "",
        group: str = "",
        status: str = "",
    ) -> None:
        normalized = (
            str(query or "").strip(),
            str(kind or ""),
            str(group or ""),
            str(status or ""),
        )
        if normalized == self.filters:
            return
        self._query, self._kind, self._group, self._status = normalized
        self.proxy_model.set_filters(
            query=self._query,
            kind=self._kind,
            group=self._group,
            status=self._status,
        )
        self.filtersChanged.emit(*normalized)
        self._emit_counts()

    def set_sort(self, column: int, order: Qt.SortOrder | int) -> None:
        normalized_column = int(column)
        try:
            normalized_order = Qt.SortOrder(order)
        except (TypeError, ValueError):
            normalized_order = Qt.SortOrder.AscendingOrder
        if (
            normalized_column == self._sort_column
            and normalized_order == self._sort_order
        ):
            return
        self._sort_column = normalized_column
        self._sort_order = normalized_order
        self.proxy_model.sort(normalized_column, normalized_order)
        self.sortChanged.emit(normalized_column, int(normalized_order.value))

    def selected_measurement_ids(self) -> list[str]:
        result: list[str] = []
        for index in self.selection_model.selectedRows(
            int(MeasurementResultColumn.RESULT_SEQUENCE)
        ):
            measurement_id = index.data(MEASUREMENT_ID_ROLE)
            if measurement_id:
                result.append(str(measurement_id))
        return result

    def measurement_id_for_index(self, proxy_index: QModelIndex) -> str | None:
        if not proxy_index.isValid() or proxy_index.model() is not self.proxy_model:
            return None
        measurement_id = proxy_index.data(MEASUREMENT_ID_ROLE)
        return str(measurement_id) if measurement_id else None

    def select_measurement_id(self, measurement_id: str | None) -> bool:
        self.selection_model.clearSelection()
        source_row = self.source_model.source_row_for_id(measurement_id)
        if source_row < 0:
            self.selection_model.setCurrentIndex(
                QModelIndex(),
                QItemSelectionModel.SelectionFlag.NoUpdate,
            )
            return False
        source_index = self.source_model.index(
            source_row,
            int(MeasurementResultColumn.RESULT_SEQUENCE),
        )
        proxy_index = self.proxy_model.mapFromSource(source_index)
        if not proxy_index.isValid():
            return False
        self.selection_model.select(
            proxy_index,
            QItemSelectionModel.SelectionFlag.ClearAndSelect
            | QItemSelectionModel.SelectionFlag.Rows,
        )
        self.selection_model.setCurrentIndex(
            proxy_index,
            QItemSelectionModel.SelectionFlag.NoUpdate,
        )
        return True

    def _emit_counts(self, *_args) -> None:
        self.countsChanged.emit(
            self.proxy_model.rowCount(),
            self.source_model.rowCount(),
        )


class MeasurementRecordsPane(QWidget):
    """Reusable filters, table and actions bound to a records controller."""

    measurementActivated = Signal(str)
    deleteSelectedRequested = Signal(object)
    deleteCategoryRequested = Signal()
    deleteAllRequested = Signal()
    headerStateChanged = Signal(str)

    _WIDE_WIDTHS = (105, 125, 150, 90, 120, 75, 110, 125, 90, 125, 160, 120)
    HEADER_STATE_SCHEMA = "measurement-records-v2"
    _COMPACT_WIDTHS = (48, 80, 58, 42, 54, 42, 70, 80, 60, 56, 110, 90)
    _WIDE_VISIBLE = frozenset(
        {
            MeasurementResultColumn.RESULT_SEQUENCE,
            MeasurementResultColumn.GROUP,
            MeasurementResultColumn.KIND,
            MeasurementResultColumn.RESULT,
            MeasurementResultColumn.UNIT,
            MeasurementResultColumn.HOLE_AREA,
            MeasurementResultColumn.MODE,
            MeasurementResultColumn.CONFIDENCE,
            MeasurementResultColumn.STATUS,
            MeasurementResultColumn.CREATED_AT,
        }
    )
    _COMPACT_VISIBLE = frozenset(
        {
            MeasurementResultColumn.RESULT_SEQUENCE,
            MeasurementResultColumn.GROUP,
            MeasurementResultColumn.KIND,
            MeasurementResultColumn.RESULT,
            MeasurementResultColumn.UNIT,
            MeasurementResultColumn.STATUS,
        }
    )

    def __init__(
        self,
        controller: MeasurementRecordsController,
        *,
        compact: bool = False,
        show_actions: bool = True,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.controller = controller
        self.compact = bool(compact)
        self._syncing_sort = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        filter_row = QHBoxLayout()
        filter_row.setContentsMargins(0, 0, 0, 0)
        filter_row.setSpacing(6)
        self.search_edit = QLineEdit(self)
        self.search_edit.setPlaceholderText("搜索类别、类型、状态或 ID")
        self.search_edit.setClearButtonEnabled(True)
        self.kind_filter = QComboBox(self)
        self.kind_filter.addItem("全部类型", "")
        self.kind_filter.addItem("长度", "length")
        self.kind_filter.addItem("面积", "area")
        self.kind_filter.addItem("计数", "count")
        self.group_filter = QComboBox(self)
        self.group_filter.addItem("全部类别", "")
        self.status_filter = QComboBox(self)
        self.status_filter.addItem("全部状态", "")
        self.status_filter.addItem("有效", "valid")
        self.status_filter.addItem("需复核", "review")
        self.status_filter.addItem("失败", "failed")
        if self.compact:
            layout.addWidget(self.search_edit)
            filter_row.addWidget(self.kind_filter, 1)
            filter_row.addWidget(self.group_filter, 1)
            filter_row.addWidget(self.status_filter, 1)
            layout.addLayout(filter_row)
        else:
            filter_row.addWidget(self.search_edit, 1)
            filter_row.addWidget(self.kind_filter)
            filter_row.addWidget(self.group_filter)
            filter_row.addWidget(self.status_filter)
            layout.addLayout(filter_row)

        self.table = QTableView(self)
        self.table.setModel(controller.proxy_model)
        self.table.setSelectionModel(controller.selection_model)
        self.table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Ignored)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked
            | QAbstractItemView.EditTrigger.EditKeyPressed
        )
        self.table.setItemDelegateForColumn(
            int(MeasurementResultColumn.GROUP),
            MeasurementGroupDelegate(self.table),
        )
        self.table.verticalHeader().setVisible(False)
        self.table.setSortingEnabled(True)
        initial_sort_column, initial_sort_order = controller.sort_state
        controller.proxy_model.sort(initial_sort_column, initial_sort_order)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        header.setStretchLastSection(True)
        header.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        header.customContextMenuRequested.connect(self._show_columns_menu)
        header.sortIndicatorChanged.connect(self._on_sort_indicator_changed)
        header.sectionResized.connect(lambda *_args: self._emit_header_state())
        header.sectionMoved.connect(lambda *_args: self._emit_header_state())
        layout.addWidget(self.table, 1)

        action_host = QWidget(self) if show_actions and self.compact else None
        action_parent = action_host or self
        self.count_label = QLabel(action_parent)
        self.count_label.setObjectName("measurementRecordsCount")
        if show_actions:
            self.delete_selected_button = QPushButton("删除选中", action_parent)
            self.delete_category_button = QPushButton("删除类别", action_parent)
            self.delete_all_button = QPushButton("删除全部", action_parent)
            self.delete_selected_button.clicked.connect(self._request_delete_selected)
            self.delete_category_button.clicked.connect(
                lambda _checked=False: self.deleteCategoryRequested.emit()
            )
            self.delete_all_button.clicked.connect(
                lambda _checked=False: self.deleteAllRequested.emit()
            )
            if self.compact:
                assert action_host is not None
                action_flow = FlowLayout(action_host, h_spacing=6, v_spacing=4)
                action_flow.addWidget(self.delete_selected_button)
                action_flow.addWidget(self.delete_category_button)
                action_flow.addWidget(self.delete_all_button)
                action_flow.addWidget(self.count_label)
                layout.addWidget(action_host)
            else:
                action_row = QHBoxLayout()
                action_row.setContentsMargins(0, 0, 0, 0)
                action_row.setSpacing(8)
                action_row.addWidget(self.delete_selected_button)
                action_row.addWidget(self.delete_category_button)
                action_row.addWidget(self.delete_all_button)
                action_row.addStretch(1)
                action_row.addWidget(self.count_label)
                layout.addLayout(action_row)
        else:
            layout.addWidget(self.count_label, 0, Qt.AlignmentFlag.AlignRight)

        self.search_edit.textChanged.connect(self._apply_filters)
        self.kind_filter.currentIndexChanged.connect(self._apply_filters)
        self.group_filter.currentIndexChanged.connect(self._apply_filters)
        self.status_filter.currentIndexChanged.connect(self._apply_filters)
        self.table.doubleClicked.connect(self._activate_index)
        controller.documentChanged.connect(self._refresh_group_filter)
        controller.filtersChanged.connect(self._sync_filters)
        controller.countsChanged.connect(self._update_count)
        controller.sortChanged.connect(self._sync_sort)

        self.reset_columns()
        self._refresh_group_filter(controller.document)
        self._sync_filters(*controller.filters)
        self._update_count(controller.proxy_model.rowCount(), controller.source_model.rowCount())
        sort_column, sort_order = controller.sort_state
        self._sync_sort(sort_column, int(sort_order.value))

    def save_header_state(self) -> str:
        encoded = bytes(self.table.horizontalHeader().saveState().toBase64()).decode("ascii")
        return f"{self.HEADER_STATE_SCHEMA}:{encoded}"

    def restore_header_state(self, state: str, *, restore_sort: bool = False) -> bool:
        token = str(state or "").strip()
        prefix = f"{self.HEADER_STATE_SCHEMA}:"
        if not token.startswith(prefix):
            if token:
                self.reset_columns()
            return False
        encoded = token[len(prefix):].strip()
        if not encoded:
            self.reset_columns()
            return False
        header = self.table.horizontalHeader()
        with _blocked(header):
            restored = header.restoreState(QByteArray.fromBase64(encoded.encode("ascii")))
        if not restored:
            self.reset_columns()
            return False
        if restored:
            if restore_sort and header.isSortIndicatorShown():
                self.controller.set_sort(
                    header.sortIndicatorSection(),
                    header.sortIndicatorOrder(),
                )
            else:
                sort_column, sort_order = self.controller.sort_state
                self._sync_sort(sort_column, int(sort_order.value))
            self._emit_header_state()
        return True

    def reset_columns(self) -> None:
        visible = self._COMPACT_VISIBLE if self.compact else self._WIDE_VISIBLE
        widths = self._COMPACT_WIDTHS if self.compact else self._WIDE_WIDTHS
        for column, width in enumerate(widths):
            self.table.setColumnHidden(column, MeasurementResultColumn(column) not in visible)
            self.table.setColumnWidth(column, width)
        self._emit_header_state()

    def scroll_to_measurement(self, measurement_id: str | None) -> bool:
        source_row = self.controller.source_model.source_row_for_id(measurement_id)
        if source_row < 0:
            return False
        source_index = self.controller.source_model.index(
            source_row,
            int(MeasurementResultColumn.RESULT_SEQUENCE),
        )
        proxy_index = self.controller.proxy_model.mapFromSource(source_index)
        if not proxy_index.isValid():
            return False
        self.table.scrollTo(proxy_index, QAbstractItemView.ScrollHint.PositionAtCenter)
        return True

    def _apply_filters(self, *_args) -> None:
        self.controller.set_filters(
            query=self.search_edit.text(),
            kind=str(self.kind_filter.currentData() or ""),
            group=str(self.group_filter.currentData() or ""),
            status=str(self.status_filter.currentData() or ""),
        )

    def _sync_filters(self, query: str, kind: str, group: str, status: str) -> None:
        with _blocked(self.search_edit, self.kind_filter, self.group_filter, self.status_filter):
            self.search_edit.setText(query)
            self._set_combo_data(self.kind_filter, kind)
            self._set_combo_data(self.group_filter, group)
            self._set_combo_data(self.status_filter, status)

    def _refresh_group_filter(self, _document: object = None) -> None:
        current = self.controller.filters[2]
        with _blocked(self.group_filter):
            self.group_filter.clear()
            self.group_filter.addItem("全部类别", "")
            for label in self.controller.group_labels():
                self.group_filter.addItem(label, label)
            self._set_combo_data(self.group_filter, current)

    @staticmethod
    def _set_combo_data(combo: QComboBox, value: str) -> None:
        index = combo.findData(value)
        if index < 0 and isinstance(value, str):
            target = value.casefold()
            for candidate in range(combo.count()):
                data = combo.itemData(candidate)
                if isinstance(data, str) and data.casefold() == target:
                    index = candidate
                    break
        combo.setCurrentIndex(index if index >= 0 else 0)

    def _on_sort_indicator_changed(self, column: int, order: Qt.SortOrder) -> None:
        if not self._syncing_sort:
            self.controller.set_sort(column, order)

    def _sync_sort(self, column: int, order_value: int) -> None:
        header = self.table.horizontalHeader()
        self._syncing_sort = True
        try:
            if column < 0:
                header.setSortIndicatorShown(False)
                return
            order = Qt.SortOrder(order_value)
            header.setSortIndicator(column, order)
            header.setSortIndicatorShown(True)
        finally:
            self._syncing_sort = False

    def _activate_index(self, index: QModelIndex) -> None:
        if not index.isValid() or index.column() == int(MeasurementResultColumn.GROUP):
            return
        measurement_id = self.controller.measurement_id_for_index(index)
        if measurement_id:
            self.measurementActivated.emit(measurement_id)

    def _request_delete_selected(self) -> None:
        self.deleteSelectedRequested.emit(self.controller.selected_measurement_ids())

    def _show_columns_menu(self, position: QPoint) -> None:
        header = self.table.horizontalHeader()
        menu = QMenu(header)
        model = self.table.model()
        for column in range(model.columnCount() if model is not None else 0):
            label = str(model.headerData(column, Qt.Orientation.Horizontal) or column)
            action = menu.addAction(label)
            action.setCheckable(True)
            action.setChecked(not self.table.isColumnHidden(column))
            action.toggled.connect(
                lambda visible, target=column: self._set_column_visible(target, visible)
            )
        menu.addSeparator()
        menu.addAction("恢复默认列", self.reset_columns)
        menu.exec(header.mapToGlobal(position))

    def _set_column_visible(self, column: int, visible: bool) -> None:
        self.table.setColumnHidden(column, not visible)
        self._emit_header_state()

    def _emit_header_state(self) -> None:
        if hasattr(self, "table"):
            self.headerStateChanged.emit(self.save_header_state())

    def _update_count(self, visible: int, total: int) -> None:
        self.count_label.setText(
            f"{visible}/{total} 条" if visible != total else f"{total} 条"
        )


@contextmanager
def _blocked(*widgets):
    previous = [widget.blockSignals(True) for widget in widgets]
    try:
        yield
    finally:
        for widget, state in zip(widgets, previous, strict=True):
            widget.blockSignals(state)
