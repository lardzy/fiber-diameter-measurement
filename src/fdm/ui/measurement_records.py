from __future__ import annotations

from contextlib import contextmanager

from PySide6.QtCore import (
    QByteArray,
    QItemSelection,
    QItemSelectionModel,
    QModelIndex,
    QObject,
    QPoint,
    QTimer,
    Qt,
    Signal,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMenu,
    QPushButton,
    QSizePolicy,
    QStyle,
    QStyledItemDelegate,
    QTableView,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
)

from fdm.models import ImageDocument, Measurement, UNCATEGORIZED_LABEL
from fdm.services.measurement_statistics import MeasurementStatisticsService
from fdm.units import resolve_length_unit
from fdm.ui.measurement_results_model import (
    MEASUREMENT_ID_ROLE,
    RAW_STATUS_ROLE,
    MeasurementGroupDelegate,
    MeasurementResultColumn,
    MeasurementResultsModel,
    MeasurementResultsProxyModel,
)
from fdm.ui.widgets import NoWheelComboBox


class _MeasurementRecordsHeader(QHeaderView):
    """Give the narrow inspector concise labels without changing shared data."""

    def __init__(self, *, compact: bool, parent: QWidget | None = None) -> None:
        super().__init__(Qt.Orientation.Horizontal, parent)
        self._compact = bool(compact)

    def display_label(self, logical_index: int, fallback: str) -> str:
        if self._compact:
            if logical_index == MeasurementResultColumn.RESULT_SEQUENCE:
                return "序号"
            if logical_index == MeasurementResultColumn.GROUP:
                return "类别"
        return fallback

    def initStyleOptionForIndex(self, option, logical_index: int) -> None:
        super().initStyleOptionForIndex(option, logical_index)
        option.text = self.display_label(logical_index, option.text)


class _CompactValueDelegate(QStyledItemDelegate):
    """Compose two display columns without changing the shared sort or edit roles."""

    def initStyleOption(self, option, index):
        super().initStyleOption(option, index)
        if index.column() == MeasurementResultColumn.RESULT:
            unit = index.siblingAtColumn(MeasurementResultColumn.UNIT).data() or ""
            length_unit = resolve_length_unit(unit.removesuffix("²"))
            if length_unit:
                unit = length_unit.symbol + ("²" if unit.endswith("²") else "")
            option.text = f"{option.text} {unit}".rstrip()
        elif index.column() == MeasurementResultColumn.RESULT_SEQUENCE:
            kind = index.siblingAtColumn(MeasurementResultColumn.KIND).data()
            option.text = f"{kind} {option.text}"
            option.displayAlignment = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
            status = index.data(RAW_STATUS_ROLE)
            if (status == MeasurementStatisticsService.MANUAL_REVIEW_STATUS
                    or status in MeasurementStatisticsService.DEFAULT_HARD_FAILURE_STATUSES):
                option.icon = QApplication.style().standardIcon(QStyle.StandardPixmap.SP_MessageBoxWarning)
                option.features |= option.ViewItemFeature.HasDecoration


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
    refreshRequested = Signal(object)
    newRecordsReady = Signal(object, object)

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
        self._category_editors = 0
        self._refresh_after_edit = False
        self._pending_new_records: list[str] = []
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
        if document is not self.document:
            self._pending_new_records.clear()
            self._refresh_after_edit = False
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

    def category_edit_started(self) -> None:
        self._category_editors += 1

    def category_edit_finished(self) -> None:
        # A menu can send aboutToHide before triggering its action, and a
        # delegate may still be committing data. Finish that transaction first.
        QTimer.singleShot(0, self, self._release_category_editor)

    def _release_category_editor(self) -> None:
        self._category_editors -= 1
        self._publish_pending_records()

    def defer_refresh_while_editing(self, document) -> bool:
        if self._category_editors and document is self.document and document is not None:
            self._refresh_after_edit = True
            return True
        return False

    def records_committed(self, document: ImageDocument, ids) -> None:
        if document is self.document:
            self._pending_new_records.extend(ids)
            self._publish_pending_records()

    def _publish_pending_records(self) -> None:
        if self._category_editors or self.document is None:
            return
        if self._refresh_after_edit:
            self._refresh_after_edit = False
            self.refreshRequested.emit(self.document)
        if self._pending_new_records:
            ids, self._pending_new_records = self._pending_new_records, []
            self.newRecordsReady.emit(self.document, ids)

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
        # Keep an existing category filter after its last row is reassigned.
        # Otherwise rebuilding the table silently selects “全部类别”.
        if self._group and self._group.casefold() not in seen:
            names = [UNCATEGORIZED_LABEL, *(group.label.strip() or group.display_name()
                                           for group in document.fiber_groups)]
            match = next((name for name in names if name.casefold() == self._group.casefold()), None)
            if match is not None:
                labels.append(match)
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
        # Canvas clicks explicitly replace table selection, even if the same
        # row was still selected while the canvas had no active object.
        self.selection_model.clearSelection()
        ids = (measurement_id,) if measurement_id is not None else ()
        return self.select_measurement_ids(ids, primary=measurement_id) is not None

    def select_measurement_ids(self, ids, *, primary=None):
        """Restore surviving visible records by ID after sorting or rebuilding."""
        selection = QItemSelection()
        current = QModelIndex()
        for identity in ids:
            row = self.source_model.source_row_for_id(identity)
            if row < 0:
                continue
            source = self.source_model.index(row, int(MeasurementResultColumn.RESULT_SEQUENCE))
            index = self.proxy_model.mapFromSource(source)
            if index.isValid():
                selection.select(index, index)
                if not current.isValid() or identity == primary:
                    current = index
        self.selection_model.select(
            selection,
            QItemSelectionModel.SelectionFlag.ClearAndSelect
            | QItemSelectionModel.SelectionFlag.Rows,
        )
        self.selection_model.setCurrentIndex(
            current,
            QItemSelectionModel.SelectionFlag.NoUpdate,
        )
        return self.measurement_id_for_index(current)

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
    propertiesRequested = Signal()
    batchGroupChangeRequested = Signal(str, object, object)

    _WIDE_WIDTHS = (105, 125, 150, 90, 120, 75, 110, 125, 90, 125, 160, 120)
    HEADER_STATE_SCHEMA = "measurement-records-v2"
    COMPACT_HEADER_STATE_SCHEMA = "measurement-records-compact-v5"
    _COMPACT_WIDTHS = (82, 80, 91, 48, 102, 40, 70, 80, 60, 48, 110, 90)
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
            MeasurementResultColumn.RESULT,
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
        self._selection_label = ""
        self._count_notice_timer = QTimer(self)
        self._count_notice_timer.setSingleShot(True)
        self._count_notice_timer.timeout.connect(self._refresh_count_label)
        self._notice_document = None
        # 启动恢复前禁止外发列状态：构造和初次布局期间的列宽信号是默认值，
        # 若写回设置会在 restore_header_state 读取之前覆盖已保存的用户列宽。
        self._header_state_persist_ready = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        filter_row = QHBoxLayout()
        filter_row.setContentsMargins(0, 0, 0, 0)
        filter_row.setSpacing(6)
        self.search_edit = QLineEdit(self)
        self.search_edit.setPlaceholderText("搜索类别、类型、状态或 ID")
        self.search_edit.setClearButtonEnabled(True)
        self.kind_filter = NoWheelComboBox(self)
        self.kind_filter.addItem("全部类型", "")
        self.kind_filter.addItem("长度", "length")
        self.kind_filter.addItem("面积", "area")
        self.kind_filter.addItem("计数", "count")
        self.group_filter = NoWheelComboBox(self)
        self.group_filter.addItem("全部类别", "")
        self.status_filter = NoWheelComboBox(self)
        self.status_filter.addItem("全部状态", "")
        self.status_filter.addItem("有效", "valid")
        self.status_filter.addItem("需复核", "review")
        self.status_filter.addItem("失败", "failed")
        for combo in (self.kind_filter, self.group_filter, self.status_filter):
            combo.setMinimumWidth(0)
            combo.setMinimumContentsLength(6)
            combo.setSizeAdjustPolicy(
                NoWheelComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
            )
            combo.setSizePolicy(
                QSizePolicy.Policy.Ignored,
                QSizePolicy.Policy.Fixed,
            )
        self.group_filter.currentTextChanged.connect(self.group_filter.setToolTip)
        if self.compact:
            self.search_edit.setPlaceholderText("搜索记录…")
            self.search_edit.setAccessibleName("搜索当前图的测量记录")
            filter_row.addWidget(self.search_edit, 1)
            self.filter_button = self._popup_button("筛选")
            filter_panel = QWidget()
            filters = QFormLayout(filter_panel)
            filters.addRow("类型", self.kind_filter)
            filters.addRow("类别", self.group_filter)
            filters.addRow("状态", self.status_filter)
            filter_panel.setMinimumWidth(210)
            filter_menu = self.filter_button.menu()
            filter_action = QWidgetAction(filter_menu)
            filter_action.setDefaultWidget(filter_panel)
            filter_menu.addAction(filter_action)
            filter_menu.addAction("清除搜索与筛选", lambda: controller.set_filters())
            self.more_button = self._popup_button("更多")
            self.more_menu = self.more_button.menu()
            filter_row.addWidget(self.filter_button)
            filter_row.addWidget(self.more_button)
            layout.addLayout(filter_row)
        else:
            filter_row.addWidget(self.search_edit, 1)
            filter_row.addWidget(self.kind_filter)
            filter_row.addWidget(self.group_filter)
            filter_row.addWidget(self.status_filter)
            layout.addLayout(filter_row)

        self.table = QTableView(self)
        self._header = _MeasurementRecordsHeader(compact=self.compact, parent=self.table)
        self.table.setHorizontalHeader(self._header)
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
        if self.compact:
            self.table.setItemDelegate(_CompactValueDelegate(self.table))
        group_delegate = MeasurementGroupDelegate(self.table)
        group_delegate.editStarted.connect(controller.category_edit_started)
        group_delegate.editFinished.connect(controller.category_edit_finished)
        self.table.setItemDelegateForColumn(int(MeasurementResultColumn.GROUP), group_delegate)
        # 类别是唯一可编辑列：单击单元格直接弹出下拉，省去双击。
        self.table.clicked.connect(self._edit_group_on_click)
        self.table.verticalHeader().setVisible(False)
        self.table.setSortingEnabled(True)
        initial_sort_column, initial_sort_order = controller.sort_state
        controller.proxy_model.sort(initial_sort_column, initial_sort_order)
        header = self._header
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        header.setStretchLastSection(True)
        header.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        header.customContextMenuRequested.connect(self._show_columns_menu)
        header.sortIndicatorChanged.connect(self._on_sort_indicator_changed)
        header.sectionResized.connect(lambda *_args: self._emit_header_state())
        header.sectionMoved.connect(lambda *_args: self._emit_header_state())
        layout.addWidget(self.table, 1)

        self.count_label = QLabel(self)
        self.count_label.setObjectName("measurementRecordsCount")
        if self.compact:
            footer = QHBoxLayout()
            footer.setSpacing(4)
            self.count_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
            footer.addWidget(self.count_label, 1)
            self.batch_group_button = self._popup_button("批量改类别")
            batch_menu = self.batch_group_button.menu()
            batch_menu.aboutToShow.connect(self._populate_batch_groups)
            batch_menu.aboutToShow.connect(controller.category_edit_started)
            batch_menu.aboutToHide.connect(controller.category_edit_finished)
            footer.addWidget(self.batch_group_button)
            self.properties_button = QPushButton("属性 →", self)
            self.properties_button.clicked.connect(self.propertiesRequested)
            footer.addWidget(self.properties_button)
            self.set_selection_context("", can_open=False, can_batch_edit=False)
            layout.addLayout(footer)
            if show_actions:
                self.delete_selected_button = self.more_menu.addAction("删除选中", self._request_delete_selected)
                self.delete_category_button = self.more_menu.addAction("按类别删除…", self.deleteCategoryRequested.emit)
                self.delete_all_button = self.more_menu.addAction("删除全部测量…", self.deleteAllRequested.emit)
        elif show_actions:
            self.delete_selected_button = QPushButton("删除选中", self)
            self.delete_category_button = QPushButton("删除类别", self)
            self.delete_all_button = QPushButton("删除全部", self)
            self.delete_selected_button.clicked.connect(self._request_delete_selected)
            self.delete_category_button.clicked.connect(
                lambda _checked=False: self.deleteCategoryRequested.emit()
            )
            self.delete_all_button.clicked.connect(
                lambda _checked=False: self.deleteAllRequested.emit()
            )
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

    def _popup_button(self, title: str) -> QToolButton:
        button = QToolButton(self)
        button.setText(title)
        button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        button.setMenu(QMenu(button))
        return button

    def _populate_batch_groups(self) -> None:
        menu = self.batch_group_button.menu()
        menu.clear()
        document = self.controller.document
        ids = tuple(self.controller.selected_measurement_ids())
        if document is None or len(ids) < 2:
            return
        for label, group_id, _color in self.controller.source_model.group_options():
            action = menu.addAction(label)
            action.triggered.connect(
                lambda checked=False, target=group_id: self.batchGroupChangeRequested.emit(
                    document.id, ids, {"fiber_group_id": target}
                )
            )

    def set_selection_context(self, label: str, *, can_open: bool, can_batch_edit: bool) -> None:
        self.properties_button.setEnabled(can_open)
        self.batch_group_button.setVisible(can_batch_edit)
        self._selection_label = label if can_open else ""
        self._refresh_count_label()

    def show_filtered_insert_notice(self, count: int) -> None:
        self._notice_document = self.controller.document
        self.count_label.setText(f"新增 {count} 条，当前筛选未显示")
        self.count_label.setToolTip(self.count_label.text())
        self._count_notice_timer.start(5000)

    def _refresh_count_label(self) -> None:
        self._update_count(self.controller.proxy_model.rowCount(), self.controller.source_model.rowCount())

    def save_header_state(self) -> str:
        encoded = bytes(self.table.horizontalHeader().saveState().toBase64()).decode("ascii")
        return f"{self.header_state_schema}:{encoded}"

    @property
    def header_state_schema(self) -> str:
        return self.COMPACT_HEADER_STATE_SCHEMA if self.compact else self.HEADER_STATE_SCHEMA

    def restore_header_state(self, state: str, *, restore_sort: bool = False) -> bool:
        # 无论恢复成败，本次恢复完成后才允许把列状态写回设置。
        self._header_state_persist_ready = True
        token = str(state or "").strip()
        prefix = f"{self.header_state_schema}:"
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
        if self.compact:
            count = sum(bool(value) for value in (kind, group, status))
            self.filter_button.setText(f"筛选 ({count})" if count else "筛选")
            self._count_notice_timer.stop()
            self._refresh_count_label()

    def _refresh_group_filter(self, _document: object = None) -> None:
        current = self.controller.filters[2]
        with _blocked(self.group_filter):
            self.group_filter.clear()
            self.group_filter.addItem("全部类别", "")
            for label in self.controller.group_labels():
                self.group_filter.addItem(label, label)
            self._set_combo_data(self.group_filter, current)
        self.group_filter.setToolTip(self.group_filter.currentText())

    @staticmethod
    def _set_combo_data(combo: NoWheelComboBox, value: str) -> None:
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

    def _edit_group_on_click(self, index: QModelIndex) -> None:
        if not index.isValid() or index.column() != int(MeasurementResultColumn.GROUP):
            return
        if (QApplication.keyboardModifiers() & (
            Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.MetaModifier | Qt.KeyboardModifier.ShiftModifier
        ) or len(self.controller.selected_measurement_ids()) > 1):
            return
        self.table.edit(index)
        editor = self.table.indexWidget(index)
        if isinstance(editor, QComboBox):
            editor.showPopup()

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
            model_label = str(model.headerData(column, Qt.Orientation.Horizontal) or column)
            label = self._header.display_label(column, model_label)
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
        if hasattr(self, "table") and self._header_state_persist_ready:
            self.headerStateChanged.emit(self.save_header_state())

    def showEvent(self, event) -> None:
        # 独立使用（未走启动恢复流程）时，最晚在显示后允许持久化。
        self._header_state_persist_ready = True
        super().showEvent(event)

    def _update_count(self, visible: int, total: int) -> None:
        if self._count_notice_timer.isActive():
            if self._notice_document is self.controller.document:
                return
            self._count_notice_timer.stop()
        count = f"{visible}/{total} 条" if visible != total else f"{total} 条"
        if self.compact:
            label = self._selection_label or "当前图"
            count = f"{label} · {count}"
        self.count_label.setText(count)
        self.count_label.setToolTip(count)


@contextmanager
def _blocked(*widgets):
    previous = [widget.blockSignals(True) for widget in widgets]
    try:
        yield
    finally:
        for widget, state in zip(widgets, previous, strict=True):
            widget.blockSignals(state)
