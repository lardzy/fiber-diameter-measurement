"""A movable view of a trial's two ledgers, independent of acquisition ownership.

The model never writes domain records. Hosts publish business changes explicitly;
camera frames and preview timers have no reason to refresh this table.
"""
from __future__ import annotations

from bisect import bisect_left
from datetime import datetime, timezone
import json
from pathlib import Path

from PySide6.QtCore import (
    QAbstractTableModel, QEvent, QModelIndex, QObject, QPoint, QRect,
    QSortFilterProxyModel, Qt, QTimer, Signal,
)
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QDockWidget, QGridLayout,
    QLabel, QMainWindow, QMenu, QPushButton, QSizePolicy, QTableView, QVBoxLayout, QWidget,
)

from fdm.atomic_io import atomic_write_text
from fdm.ui.icons import themed_icon
from fdm.ui.widgets import NoWheelComboBox


RecordKey = tuple[str, str]
KEY_ROLE = int(Qt.ItemDataRole.UserRole) + 1


def _record_time(value):
    try:
        result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        # The original producer used UTC. Do not reinterpret legacy naive text
        # according to whichever computer happens to open the project today.
        return (result.replace(tzinfo=timezone.utc) if result.tzinfo is None
                else result.astimezone(timezone.utc))
    except (ValueError, TypeError, OverflowError):
        return None


class TrialTimelineModel(QAbstractTableModel):
    """Typed references to accepted measurements and signed count adjustments."""

    headers = ("序号", "类型", "结果", "类别", "操作者", "方向", "时间", "方式", "图像证据")

    def __init__(self, session=None, parent=None):
        super().__init__(parent)
        self.session = None
        self.keys: list[RecordKey] = []
        self.rows_by_key: dict[RecordKey, int] = {}
        self._sort_keys = []
        self._record_refs = {}
        self._metadata = ()
        self.revision = -1
        if session is not None:
            self.set_session(session)

    def record(self, key):
        if self.session is None or not key or key[0] not in ("measurement", "count"):
            return None
        ledger = self.session.measurements if key[0] == "measurement" else self.session.count_adjustments
        return ledger.get(key[1])

    def _order(self, key):
        value = _record_time(getattr(self.record(key), "created_at", None))
        # Unknown timestamps sort first; tied/unknown records have stable order
        # across reload, undo and redo without modifying the saved project.
        return (value is not None, value.isoformat(timespec="microseconds") if value else "", *key)

    def _all_keys(self):
        if self.session is None:
            return []
        return ([('measurement', key) for key in self.session.measurements]
                + [('count', key) for key in self.session.count_adjustments])

    def _metadata_signature(self):
        return tuple((operator.id, operator.name,
                      tuple((direction, material.id, material.name)
                            for direction, materials in operator.materials.items() for material in materials))
                     for operator in self.session.operators.values()) if self.session else ()

    def set_session(self, session):
        self.beginResetModel()
        self.session = session
        self.keys = sorted(self._all_keys(), key=self._order)
        self._sort_keys = [self._order(key) for key in self.keys]
        self._record_refs = {key: self.record(key) for key in self.keys}
        self._metadata = self._metadata_signature()
        self.rows_by_key = {key: row for row, key in enumerate(self.keys)}
        self.revision = session.revision if session is not None else -1
        self.endResetModel()

    def sync(self, keys=None, *, rebuild=False):
        """Return added keys. A no-op revision check performs no ledger scan."""
        if self.session is None or (keys is None and not rebuild and self.revision == self.session.revision):
            return []
        old = set(self.keys) if rebuild else None
        if rebuild:
            self.set_session(self.session)
            return [key for key in self.keys if key not in old]
        metadata = self._metadata_signature()
        metadata_changed = metadata != self._metadata
        self._metadata = metadata
        if keys is None:
            live = self._all_keys()
            current = set(live)
            # Records are frozen dataclasses: ordinary evidence completion or
            # edits replace only their own row. Comparing references avoids
            # reparsing/sorting every timestamp and emitting N dataChanged
            # signals on each completed image write.
            keys = ([key for key in self.keys if key not in current]
                    + [key for key in live if self._record_refs.get(key) is not self.record(key)])
        keys = list(dict.fromkeys(tuple(key) for key in keys))
        # A confirmed full-field batch normally arrives in chronological order.
        # Publish its contiguous tail once instead of laying out each new row.
        if keys and all(key not in self.rows_by_key and self.record(key) is not None for key in keys):
            ordered = sorted((self._order(key), key) for key in keys)
            if not self._sort_keys or ordered[0][0] >= self._sort_keys[-1]:
                start = len(self.keys)
                self.beginInsertRows(QModelIndex(), start, start + len(ordered) - 1)
                self._sort_keys.extend(order for order, _ in ordered)
                added = [key for _, key in ordered]
                self.keys.extend(added)
                self._record_refs.update((key, self.record(key)) for key in added)
                self.rows_by_key.update((key, start + offset) for offset, key in enumerate(added))
                self.endInsertRows()
                self.revision = self.session.revision
                if metadata_changed and self.keys:
                    self.dataChanged.emit(self.index(0, 0), self.index(len(self.keys) - 1, self.columnCount() - 1))
                return added
        added = []
        for key in keys:
            record = self.record(key)
            row = self.rows_by_key.get(key)
            if record is None:
                if row is not None:
                    self.beginRemoveRows(QModelIndex(), row, row)
                    self.keys.pop(row)
                    self._sort_keys.pop(row)
                    self._record_refs.pop(key, None)
                    self.rows_by_key = {item: i for i, item in enumerate(self.keys)}
                    self.endRemoveRows()
                continue
            order = self._order(key)
            if row is not None and self._sort_keys[row] == order:
                self._record_refs[key] = record
                self.dataChanged.emit(self.index(row, 0), self.index(row, self.columnCount() - 1))
                continue
            if row is not None:
                self.beginRemoveRows(QModelIndex(), row, row)
                self.keys.pop(row)
                self._sort_keys.pop(row)
                self.rows_by_key = {item: i for i, item in enumerate(self.keys)}
                self.endRemoveRows()
            else:
                added.append(key)
            destination = bisect_left(self._sort_keys, order)
            self.beginInsertRows(QModelIndex(), destination, destination)
            self.keys.insert(destination, key)
            self._sort_keys.insert(destination, order)
            self._record_refs[key] = record
            if destination == len(self.keys) - 1:
                self.rows_by_key[key] = destination
            else:
                self.rows_by_key = {item: i for i, item in enumerate(self.keys)}
            self.endInsertRows()
        self.revision = self.session.revision
        if metadata_changed and self.keys:
            self.dataChanged.emit(self.index(0, 0), self.index(len(self.keys) - 1, self.columnCount() - 1))
        return added

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.keys)

    def columnCount(self, parent=QModelIndex()):
        return len(self.headers)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return self.headers[section] if 0 <= section < len(self.headers) else None
        return None

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or not 0 <= index.row() < len(self.keys):
            return None
        key = self.keys[index.row()]
        if role == KEY_ROLE:
            return key
        record = self.record(key)
        if record is None:
            return None
        if role == Qt.ItemDataRole.ToolTipRole:
            details = [f"记录 ID：{key[1]}", f"视场：{record.field_id or '未知'}", f"时间：{record.created_at}"]
            if key[0] == "count":
                details += ["补计只改变根数，不产生直径。", record.note]
            return "\n".join(filter(None, details))
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        operator = self.session.operators.get(record.operator_id)
        materials = operator.materials.get(record.direction, ()) if operator else ()
        material = next((item.name for item in materials if item.id == record.material_id), record.material_id)
        timestamp = _record_time(record.created_at)
        try:
            time_text = timestamp.astimezone().strftime("%m-%d %H:%M:%S") if timestamp else "时间未知"
        except (ValueError, OverflowError, OSError):
            time_text = "时间未知"
        measurement = key[0] == "measurement"
        method = ({"assisted": "贴边测径", "free": "自由测量", "automatic-traditional": "传统全图",
                   "automatic-multiscale": "多尺度全图"}.get(record.measurement_method, record.measurement_method)
                  if measurement else "补计调整")
        evidence = getattr(record, "evidence", None) or {}
        evidence_text = ({"saved": "已保存", "pending": "写入中", "failed": "写入失败"}.get(evidence.get("state"), "无原图")
                         if measurement else "不适用")
        return (str(index.row() + 1), "实测" if measurement else "补计",
                f"{record.diameter_um:.4f} μm" if measurement else f"{record.delta:+d} 根",
                material, operator.name if operator else record.operator_id,
                "经向" if record.direction == "warp" else "纬向", time_text, method, evidence_text)[index.column()]


class TrialTimelineFilter(QSortFilterProxyModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.operator_id = None
        self.direction = None
        self.record_type = None
        self.current_field_only = False
        self.field_id = ""

    def filterAcceptsRow(self, source_row, source_parent):
        model = self.sourceModel()
        key = model.keys[source_row]
        record = model.record(key)
        return record is not None and (
            (self.operator_id is None or record.operator_id == self.operator_id)
            and (self.direction is None or record.direction == self.direction)
            and (self.record_type is None or key[0] == self.record_type)
            and (not self.current_field_only or record.field_id == self.field_id))

    def apply(self, *, operator_id=None, direction=None, record_type=None, current_field_only=False, field_id=""):
        values = (operator_id, direction, record_type, bool(current_field_only), field_id)
        previous = (self.operator_id, self.direction, self.record_type, self.current_field_only, self.field_id)
        if values != previous:
            modern = hasattr(self, "endFilterChange")
            if modern:
                self.beginFilterChange()
            (self.operator_id, self.direction, self.record_type,
             self.current_field_only, self.field_id) = values
            if modern:
                self.endFilterChange(QSortFilterProxyModel.Direction.Rows)
            else:
                self.invalidateFilter()


class TrialRecordsPanel(QWidget):
    actionRequested = Signal(str, object)
    columnsChanged = Signal()
    positionRequested = Signal(str)
    resetLayoutRequested = Signal()

    DEFAULT_COLUMNS = (44, 44, 104, 116, 90, 46, 112, 86, 78)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("trialRecordsPanel")
        self.setMinimumSize(270, 120)
        self.model = TrialTimelineModel(parent=self)
        self.proxy = TrialTimelineFilter(self)
        self.proxy.setSourceModel(self.model)
        self.following = True
        self._updating = False
        self._pending_latest = False
        self._unseen = set()
        self._operator_signature = None
        self.field_id = ""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)
        self._controls = QGridLayout()
        self._controls.setContentsMargins(0, 0, 0, 0)
        self._controls.setSpacing(4)
        self.scope_label = QLabel("全部操作者 · 全部方向", self)
        self.scope_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.position_button = QPushButton("位置", self)
        self.position_button.setIcon(themed_icon("results"))
        menu = QMenu(self.position_button)
        for label, position in (("停靠左侧", "left"), ("停靠底部", "bottom"), ("浮动窗口", "floating")):
            action = menu.addAction(label)
            action.triggered.connect(lambda checked=False, value=position: self.positionRequested.emit(value))
        menu.addSeparator()
        menu.addAction("恢复默认布局", self.resetLayoutRequested.emit)
        self.position_button.setMenu(menu)
        self.operator_filter = NoWheelComboBox(self)
        self.operator_filter.setAccessibleName("记录操作者筛选")
        self.direction_filter = NoWheelComboBox(self)
        self.direction_filter.setAccessibleName("记录方向筛选")
        self.type_filter = NoWheelComboBox(self)
        self.type_filter.setAccessibleName("记录类型筛选")
        self.field_filter = NoWheelComboBox(self)
        self.field_filter.setAccessibleName("记录视场筛选")
        for label, value in (("全部方向", None), ("经向", "warp"), ("纬向", "weft")):
            self.direction_filter.addItem(label, value)
        for label, value in (("实测与补计", None), ("仅实测", "measurement"), ("仅补计", "count")):
            self.type_filter.addItem(label, value)
        self.field_filter.addItem("全部视场", False)
        self.field_filter.addItem("当前视场", True)
        for widget in (self.operator_filter, self.direction_filter, self.type_filter, self.field_filter):
            widget.setMinimumWidth(0)
            widget.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
            widget.currentIndexChanged.connect(self._filters_changed)
        layout.addLayout(self._controls)
        self.table = QTableView(self)
        self.table.setObjectName("trialTimelineTable")
        self.table.setModel(self.proxy)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.table.verticalHeader().hide()
        self.table.horizontalHeader().setStretchLastSection(False)
        self.table.horizontalHeader().sectionResized.connect(lambda *_: self.columnsChanged.emit())
        self.table.selectionModel().selectionChanged.connect(self._selection_changed)
        self.table.verticalScrollBar().valueChanged.connect(self._scrolled)
        layout.addWidget(self.table, 1)
        self.latest_button = QPushButton("回到最新", self)
        self.latest_button.clicked.connect(self.follow_latest)
        self._actions = QGridLayout()
        self._actions.setContentsMargins(0, 0, 0, 0)
        self._actions.setSpacing(3)
        self.action_buttons = {}
        for action, label, icon in (("edit", "修订", "rename"), ("delete", "删除", "delete"),
                                    ("evidence", "原图", "open_images"), ("correct_count", "调整补计", "count")):
            button = QPushButton(label, self)
            button.setIcon(themed_icon(icon))
            button.clicked.connect(lambda checked=False, value=action: self._request_action(value))
            self.action_buttons[action] = button
        layout.addLayout(self._actions)
        self._wide = None
        self.set_presentation("left")
        self._latest_timer = QTimer(self)
        self._latest_timer.setSingleShot(True)
        self._latest_timer.timeout.connect(self._scroll_latest)
        self.restore_columns(self.DEFAULT_COLUMNS)
        self._selection_changed()

    def set_presentation(self, position, *, condensed=False):
        """Use the width of a bottom dock instead of stacking five tool rows."""
        wide = position == "bottom"
        condensed = wide and condensed
        if (wide, condensed) == (self._wide, getattr(self, "_condensed", False)):
            return
        self._wide, self._condensed = wide, condensed
        filters = (self.operator_filter, self.direction_filter, self.type_filter, self.field_filter)
        widgets = (*filters, self.scope_label, self.position_button, self.latest_button, *self.action_buttons.values())
        for widget in widgets:
            self._controls.removeWidget(widget)
            self._actions.removeWidget(widget)
        for column in range(10):
            self._controls.setColumnStretch(column, 0)
            self._actions.setColumnStretch(column, 0)
        self.scope_label.setVisible(not wide)
        # Smaller control padding gives the bottom dock room for actual rows.
        # Colors and focus styling continue to come from the active theme.
        for widget in (*filters, self.position_button, self.latest_button, *self.action_buttons.values()):
            selector = "QComboBox" if widget in filters else "QPushButton"
            widget.setStyleSheet(f"{selector} {{ min-height: 0; padding: 2px 6px; }}" if wide else "")
            widget.setMinimumHeight(26 if wide else 0)
        self.table.horizontalHeader().setStyleSheet("QHeaderView::section { padding: 3px 5px; }" if wide else "")
        self.table.verticalHeader().setDefaultSectionSize(max(26, self.fontMetrics().height() + 8) if wide else 30)
        self.setMinimumHeight(120 if condensed else 160 if wide else 150)
        if wide:
            for column, widget in enumerate(filters):
                self._controls.addWidget(widget, 0, column)
                self._controls.setColumnStretch(column, 1)
            self._controls.addWidget(self.position_button, 0, 4)
            if condensed:
                for column, widget in enumerate((self.latest_button, *self.action_buttons.values()), 5):
                    self._controls.addWidget(widget, 0, column)
            else:
                self._actions.addWidget(self.latest_button, 0, 0)
                self._actions.setColumnStretch(0, 1)
                for column, widget in enumerate(self.action_buttons.values(), 1):
                    self._actions.addWidget(widget, 0, column)
        else:
            self._controls.addWidget(self.scope_label, 0, 0, 1, 3)
            self._controls.addWidget(self.position_button, 0, 3)
            for index, widget in enumerate(filters):
                self._controls.addWidget(widget, 1 + index // 2, 2 * (index % 2), 1, 2)
            self._actions.addWidget(self.latest_button, 0, 0, 1, 4)
            for column, widget in enumerate(self.action_buttons.values()):
                self._actions.addWidget(widget, 1, column)
                self._actions.setColumnStretch(column, 1)
        self.updateGeometry()

    def set_session(self, session, field_id=""):
        self._updating = True
        self.model.set_session(session)
        self.field_id = field_id
        self._operator_signature = None
        self.refresh_context(field_id)
        self._updating = False
        self.reset_filters()

    def refresh_context(self, field_id):
        session = self.model.session
        self.field_id = field_id
        signature = tuple((item.id, item.name) for item in session.operators.values()) if session else ()
        if signature != self._operator_signature:
            selected = self.operator_filter.currentData()
            self.operator_filter.blockSignals(True)
            self.operator_filter.clear()
            self.operator_filter.addItem("全部操作者", None)
            for key, name in signature:
                self.operator_filter.addItem(name, key)
            self.operator_filter.setCurrentIndex(max(0, self.operator_filter.findData(selected)))
            self.operator_filter.blockSignals(False)
            self._operator_signature = signature
        self._apply_filters()

    def reset_filters(self):
        self._updating = True
        for widget in (self.operator_filter, self.direction_filter, self.type_filter, self.field_filter):
            widget.setCurrentIndex(0)
        self._updating = False
        self._apply_filters()
        self.follow_latest()

    def _filters_changed(self):
        if not self._updating:
            self._apply_filters()
            self.follow_latest()

    def _apply_filters(self):
        self.proxy.apply(operator_id=self.operator_filter.currentData(), direction=self.direction_filter.currentData(),
                         record_type=self.type_filter.currentData(), current_field_only=self.field_filter.currentData(),
                         field_id=self.field_id)
        self.scope_label.setText(f"{self.operator_filter.currentText()} · {self.direction_filter.currentText()}")
        self.scope_label.setToolTip(self.scope_label.text())
        self._selection_changed()

    def selected_key(self):
        return self.table.currentIndex().data(KEY_ROLE)

    def select_key(self, key):
        row = self.model.rows_by_key.get(tuple(key)) if key else None
        index = self.proxy.mapFromSource(self.model.index(row, 0)) if row is not None else QModelIndex()
        if index.isValid():
            from PySide6.QtCore import QItemSelectionModel
            self.table.selectionModel().setCurrentIndex(index, QItemSelectionModel.SelectionFlag.ClearAndSelect |
                                                        QItemSelectionModel.SelectionFlag.Rows)
        else:
            self.table.clearSelection()
            self.table.setCurrentIndex(QModelIndex())

    def sync(self, keys=None, *, rebuild=False, field_id=None):
        selected = self.selected_key()
        position = self.table.verticalScrollBar().value()
        self._updating = True
        try:
            added = self.model.sync(keys, rebuild=rebuild)
            self.refresh_context(self.field_id if field_id is None else field_id)
            self.select_key(selected)
            visible_added = [key for key in added if self.proxy.mapFromSource(
                self.model.index(self.model.rows_by_key[key], 0)).isValid()]
            if self.following:
                if added or rebuild:
                    self._queue_latest()
            else:
                self.table.verticalScrollBar().setValue(position)
                self._unseen.intersection_update(self.model.rows_by_key)
                self._unseen.update(visible_added)
            self._update_latest_button()
        finally:
            self._updating = False

    def _selection_changed(self, *_):
        key = self.selected_key()
        valid = self.model.record(key) is not None
        for action, button in self.action_buttons.items():
            button.setEnabled(valid and ((key[0] == "count") if action == "correct_count" else (key[0] == "measurement")))

    def _request_action(self, action):
        key = self.selected_key()
        if key and self.action_buttons[action].isEnabled():
            self.actionRequested.emit(action, key)

    def _scrolled(self, value):
        if self._updating or self._pending_latest:
            return
        self.following = value >= self.table.verticalScrollBar().maximum() - 2
        if self.following:
            self._unseen.clear()
        self._update_latest_button()

    def _update_latest_button(self):
        count = len(self._unseen)
        self.latest_button.setText(f"新增 {count} 条 · 回到最新" if count else "回到最新")
        # A permanent row avoids changing table geometry when new data arrives.
        self.latest_button.setEnabled(not self.following or bool(count))

    def follow_latest(self):
        self.following = True
        self._unseen.clear()
        self._update_latest_button()
        self._queue_latest()

    def _queue_latest(self):
        self._pending_latest = True
        if self.isVisible():
            self._latest_timer.start(0)

    def _scroll_latest(self):
        if self._pending_latest and self.isVisible():
            self._updating = True
            self.table.scrollToBottom()
            self._updating = False
            self._pending_latest = False

    def showEvent(self, event):
        super().showEvent(event)
        if self._pending_latest:
            self._latest_timer.start(0)

    def resizeEvent(self, event):
        if self.following and hasattr(self, "_latest_timer"):
            self._queue_latest()
        super().resizeEvent(event)

    def column_widths(self):
        return [self.table.columnWidth(index) for index in range(self.model.columnCount())]

    def changeEvent(self, event):
        super().changeEvent(event)
        if (event.type() in (QEvent.Type.PaletteChange, QEvent.Type.ApplicationPaletteChange)
                and hasattr(self, "action_buttons")):
            color = self.palette().buttonText().color().name()
            self.position_button.setIcon(themed_icon("results", color=color))
            for action, icon in (("edit", "rename"), ("delete", "delete"),
                                 ("evidence", "open_images"), ("correct_count", "count")):
                self.action_buttons[action].setIcon(themed_icon(icon, color=color))

    def restore_columns(self, widths):
        for index, width in enumerate(widths[:self.model.columnCount()]):
            try:
                self.table.setColumnWidth(index, max(36, min(600, int(width))))
            except (TypeError, ValueError, OverflowError):
                pass


class AcquisitionDockHost(QMainWindow):
    """Only the acquisition body is dockable; navigation stays outside it."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.Widget)
        self.setObjectName("acquisitionDockHost")
        self.setDockNestingEnabled(False)
        self.setContentsMargins(0, 0, 0, 0)


class _RecordsDock(QDockWidget):
    closedByUser = Signal()

    def closeEvent(self, event):
        self.closedByUser.emit()
        super().closeEvent(event)


class RecordsPresentationController(QObject):
    """One app-owned records panel follows the currently active trial.

    Integration: bind on accepted navigation, suspend before leaving the trial,
    open on the records action, notify_change on business edits, shutdown once.
    Action callbacks receive (workspace, action, (type, id)); they own mutations.
    """
    visibilityChanged = Signal(bool)

    def __init__(self, dock_host, *, is_current=None, action_callback=None, preferences_path=None, parent=None):
        super().__init__(parent or dock_host)
        self.host = dock_host
        self.workspace = None
        self._active = False
        self._closed = False
        self._programmatic = False
        self._is_current = is_current or (lambda workspace: workspace is self.workspace and not getattr(workspace, "_closed", False))
        self._action_callback = action_callback
        if preferences_path is None:
            from fdm.settings import settings_file_path
            preferences_path = settings_file_path().parent / "trial-records-layout.json"
        self.preferences_path = Path(preferences_path)
        self.preferences = self._read_preferences()
        self.panel = TrialRecordsPanel(dock_host)
        self.model, self.proxy, self.table = self.panel.model, self.panel.proxy, self.panel.table
        self.dock = _RecordsDock("试验记录", dock_host)
        self.dock.setObjectName("trialRecordsDock")
        self.dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.BottomDockWidgetArea)
        self.dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetClosable |
                              QDockWidget.DockWidgetFeature.DockWidgetMovable |
                              QDockWidget.DockWidgetFeature.DockWidgetFloatable)
        self.dock.setWidget(self.panel)
        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(250)
        self._save_timer.timeout.connect(self.flush_preferences)
        self._fit_timer = QTimer(self)
        self._fit_timer.setSingleShot(True)
        self._fit_timer.timeout.connect(self._fit_bottom)
        self._bottom_target_height = None
        self._last_host_size = self.host.size()
        self._applied_position = None
        self._programmatic = True
        self.host.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.dock)
        self.panel.restore_columns(self.preferences["columns"])
        self.dock.hide()
        self._programmatic = False
        self.panel.actionRequested.connect(self._action)
        self.panel.columnsChanged.connect(self._columns_changed)
        self.panel.positionRequested.connect(self.set_position)
        self.panel.resetLayoutRequested.connect(self.reset_layout)
        self.dock.closedByUser.connect(self._user_closed)
        self.dock.topLevelChanged.connect(self._dock_position_changed)
        self.dock.dockLocationChanged.connect(self._dock_position_changed)
        self.dock.installEventFilter(self)
        self.host.installEventFilter(self)

    @staticmethod
    def _defaults():
        return {"version": 1, "position": "left", "visible": False, "left_width": 400,
                "bottom_height": 250, "floating_geometry": None,
                "columns": list(TrialRecordsPanel.DEFAULT_COLUMNS)}

    def _read_preferences(self):
        result = self._defaults()
        try:
            values = json.loads(self.preferences_path.read_text(encoding="utf-8"))
            if not isinstance(values, dict) or values.get("version") != 1:
                return result
            if values.get("position") in ("left", "bottom", "floating"):
                result["position"] = values["position"]
            result["visible"] = values.get("visible") is True
            for key, minimum, maximum in (("left_width", 270, 900), ("bottom_height", 120, 700)):
                result[key] = max(minimum, min(maximum, int(values.get(key, result[key]))))
            geometry = values.get("floating_geometry")
            if isinstance(geometry, list) and len(geometry) == 4:
                result["floating_geometry"] = [int(value) for value in geometry]
            columns = values.get("columns")
            if isinstance(columns, list) and len(columns) == len(TrialRecordsPanel.DEFAULT_COLUMNS):
                result["columns"] = [max(36, min(600, int(value))) for value in columns]
        except (OSError, ValueError, TypeError, OverflowError):
            return self._defaults()
        return result

    @property
    def is_open(self):
        return self._active and not self.dock.isHidden()

    def _current(self):
        return (not self._closed and self._active and self.workspace is not None
                and self._is_current(self.workspace))

    def contains_focus(self, widget):
        return bool(widget is not None and (widget is self.panel or self.panel.isAncestorOf(widget)))

    def bind(self, workspace):
        if self._closed:
            return
        different = self.workspace is not workspace or self.model.session is not workspace.session
        self.workspace, self._active = workspace, True
        self.panel.setEnabled(True)
        field = getattr(workspace.controller, "field_id", "")
        if different:
            self.panel.set_session(workspace.session, field)
        else:
            self.panel.sync(field_id=field)
        self._refresh_title()
        self._apply_position()
        self._programmatic = True
        self.dock.setVisible(self.preferences["visible"])
        self._programmatic = False
        self.visibilityChanged.emit(self.is_open)

    def suspend(self):
        if self._closed:
            return
        self._active = False
        self.panel.setEnabled(False)
        self._programmatic = True
        self.dock.hide()
        self._programmatic = False
        self.visibilityChanged.emit(False)

    def open(self, workspace=None):
        if workspace is not None and (workspace is not self.workspace or not self._active):
            self.bind(workspace)
        if not self._current():
            return False
        self.panel.sync(field_id=getattr(self.workspace.controller, "field_id", ""))
        self.panel.reset_filters()
        self.preferences["visible"] = True
        self._apply_position()
        self.dock.show()
        self._save_timer.start()
        self.visibilityChanged.emit(True)
        return True

    def close(self):
        """Explicit user action, unlike suspend or parent-window hiding."""
        if not self._closed:
            self.dock.close()

    def _user_closed(self):
        if not self._programmatic and not self._closed:
            self.preferences["visible"] = False
            self._save_timer.start()
            self.visibilityChanged.emit(False)

    def notify_change(self, workspace, keys=None, *, rebuild=False):
        if self._closed or workspace is not self.workspace:
            return
        self.panel.sync(keys, rebuild=rebuild, field_id=getattr(workspace.controller, "field_id", ""))
        self._refresh_title()

    def _refresh_title(self):
        if self.workspace is not None:
            session = self.workspace.session
            name = session.sample_name or session.sample_number or session.name or "新试验"
            self.dock.setWindowTitle(f"试验记录 · {name}")

    def _action(self, action, key):
        key = tuple(key)
        if (action not in ("edit", "delete", "evidence", "correct_count")
                or not self._current() or self.model.record(key) is None):
            return
        if (action == "correct_count") != (key[0] == "count"):
            return
        if self._action_callback is not None:
            self._action_callback(self.workspace, action, key)

    def set_position(self, position):
        if position not in ("left", "bottom", "floating"):
            raise ValueError("未知记录窗口位置")
        if self._closed:
            return
        self.preferences["position"] = position
        self._apply_position()
        self._save_timer.start()

    def _apply_position(self):
        position = self.preferences["position"]
        visible = self.preferences["visible"] and self._active
        self._programmatic = True
        try:
            # Retire the old dock constraints before changing the control
            # arrangement. A tall left/floating arrangement must never spend
            # one layout pass in the bottom area and enlarge the main window.
            if position != self._applied_position:
                self.dock.hide()
            central = self.host.centralWidget()
            minimum = central.minimumSizeHint().height() if central is not None else 0
            available = self.host.height() - minimum - 6
            condensed = position == "bottom" and self.host.width() >= 900
            if self.panel.following:
                self.panel._queue_latest()
            self.panel.set_presentation(position, condensed=condensed)
            if position == "floating":
                self.dock.setFloating(True)
                geometry = self.preferences["floating_geometry"]
                rect = QRect(*geometry) if geometry else QRect(self.host.mapToGlobal(QPoint(24, 24)), self.panel.sizeHint())
                screens = QApplication.screens()
                screen = next((screen for screen in screens if screen.availableGeometry().intersects(rect)),
                              self.host.screen() or QApplication.primaryScreen())
                if screen is not None:
                    # Leave space for the native tool-window frame/title bar;
                    # Qt geometry is the client rectangle, not frameGeometry.
                    available = screen.availableGeometry().adjusted(12, 36, -12, -12)
                    rect.setWidth(min(max(400, rect.width()), available.width()))
                    rect.setHeight(min(max(250, rect.height()), available.height()))
                    rect.moveLeft(max(available.left(), min(rect.left(), available.right() - rect.width() + 1)))
                    rect.moveTop(max(available.top(), min(rect.top(), available.bottom() - rect.height() + 1)))
                self.dock.setGeometry(rect)
            else:
                self.dock.setFloating(False)
                area = Qt.DockWidgetArea.LeftDockWidgetArea if position == "left" else Qt.DockWidgetArea.BottomDockWidgetArea
                self.host.addDockWidget(area, self.dock)
                dimension = self.preferences["left_width" if position == "left" else "bottom_height"]
                if position == "bottom":
                    dimension = min(dimension, max(self.dock.minimumSizeHint().height(),
                                                   available))
                    self._bottom_target_height = dimension
                self.host.resizeDocks([self.dock], [dimension], Qt.Orientation.Horizontal if position == "left" else Qt.Orientation.Vertical)
            self.dock.setVisible(visible)
            self._last_host_size = self.host.size()
            self._applied_position = position
        finally:
            self._programmatic = False

    def _dock_position_changed(self, *_):
        if self._programmatic or not self._active or self._closed:
            return
        if self.dock.isFloating():
            position = "floating"
        else:
            area = self.host.dockWidgetArea(self.dock)
            position = "bottom" if area == Qt.DockWidgetArea.BottomDockWidgetArea else "left"
        self.preferences["position"] = position
        self.panel.set_presentation(position, condensed=self.host.width() >= 900)
        self._capture_extent()

    def _capture_extent(self):
        if self._programmatic or not self._active or self._closed or not self.dock.isVisible():
            return
        if self.dock.isFloating():
            rect = self.dock.geometry()
            self.preferences["floating_geometry"] = [rect.x(), rect.y(), rect.width(), rect.height()]
        elif self.host.dockWidgetArea(self.dock) == Qt.DockWidgetArea.BottomDockWidgetArea:
            height = self.dock.height()
            if self.host.size() != self._last_host_size:
                self._fit_timer.start(0)
                return
            # A short window temporarily limits the dock. Do not replace the
            # user's normal height preference with that programmatic clamp.
            if height != self._bottom_target_height or height >= self.preferences["bottom_height"]:
                self.preferences["bottom_height"] = max(120, min(700, height))
        else:
            self.preferences["left_width"] = max(270, min(900, self.dock.width()))
        self._save_timer.start()

    def _columns_changed(self):
        if not self._programmatic and not self._closed:
            self.preferences["columns"] = self.panel.column_widths()
            self._save_timer.start()

    def eventFilter(self, watched, event):
        if watched is self.dock and event.type() in (QEvent.Type.Resize, QEvent.Type.Move):
            self._capture_extent()
        if (watched is self.host and event.type() == QEvent.Type.Resize
                and not self._closed and self._active and self.preferences["position"] == "bottom"):
            self._fit_timer.start(0)
        return super().eventFilter(watched, event)

    def _fit_bottom(self):
        if not self._closed and self._active and self.preferences["position"] == "bottom":
            self._apply_position()

    def reset_layout(self):
        visible = self.preferences["visible"]
        self.preferences = self._defaults()
        self.preferences["visible"] = visible
        self._programmatic = True
        self.panel.restore_columns(self.preferences["columns"])
        self._programmatic = False
        self._apply_position()
        self._save_timer.start()

    def flush_preferences(self):
        self._save_timer.stop()
        try:
            self.preferences_path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_text(self.preferences_path, json.dumps(self.preferences, ensure_ascii=False,
                                                               indent=2, allow_nan=False), encoding="utf-8")
        except OSError as exc:
            if self.workspace is not None and not getattr(self.workspace, "_closed", False):
                self.workspace.statusBar().showMessage(f"记录布局已应用，但未能保存偏好：{exc}", 8000)

    def shutdown(self):
        if self._closed:
            return
        if self._save_timer.isActive():
            self.flush_preferences()
        self.suspend()
        self._closed = True
        self._fit_timer.stop()
        self.workspace = None
        self.panel.set_session(None)
        self.dock.removeEventFilter(self)
        self.host.removeEventFilter(self)
