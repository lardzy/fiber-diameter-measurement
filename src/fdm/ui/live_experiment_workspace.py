"""Live experiment UI. The canvas keeps painting while a gesture owns a frame."""
from __future__ import annotations

from pathlib import Path
from concurrent.futures import CancelledError, ThreadPoolExecutor
from dataclasses import replace
import math
import threading
import time
import uuid

from PySide6.QtCore import QAbstractTableModel, QEvent, QModelIndex, QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QColor, QImage, QKeySequence, QPainter, QPen
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox,
    QDoubleSpinBox, QFileDialog, QFormLayout, QHBoxLayout, QInputDialog, QLabel,
    QLineEdit, QMainWindow, QMenu, QMessageBox, QPushButton, QSplitter, QTableView,
    QTabWidget, QToolBar, QVBoxLayout, QWidget, QTableWidget, QSpinBox, QScrollArea,
    QAbstractSpinBox, QPlainTextEdit, QTextEdit,
    QStyledItemDelegate, QGridLayout, QStatusBar,
)

from fdm.ui.widgets import NoWheelComboBox as QComboBox, NoWheelSpinBox as QSpinBox, NoWheelDoubleSpinBox as QDoubleSpinBox

from fdm.live_experiment.controller import CalibrationSnapshot, LiveExperimentController
from fdm.live_experiment.frames import FrameHandle, FrameBudgetExceeded, PresentedFrameToken, frame_from_qimage
from fdm.live_experiment.presentation import category_color, category_number


from fdm.ui.live_measurement_canvas import LiveMeasurementCanvas
from fdm.ui.trial_assistance import TrialAssistanceMixin

class ExperimentRecordsTable(QTableView):
    """Follow new records without laying out a collapsed/off-tab table."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self._latest_pending = False
        self._show_scroll_timer = QTimer(self)
        self._show_scroll_timer.setSingleShot(True)
        self._show_scroll_timer.timeout.connect(self._scroll_to_latest)

    def follow_latest(self):
        self._latest_pending = True
        if self.isVisible():
            self._scroll_to_latest()

    def _scroll_to_latest(self):
        if self._latest_pending and self.isVisible():
            self.scrollToBottom()
            self._latest_pending = False

    def showEvent(self, event):
        super().showEvent(event)
        if self._latest_pending:
            # Finish the show/resize layout before positioning the last row.
            self._show_scroll_timer.start(0)


class ExperimentRecordsModel(QAbstractTableModel):
    headers = ("序号", "操作者", "类别", "直径 μm", "方向", "方式", "图像证据")

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session
        self.ids = []
        self.rows_by_id = {}

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.ids)

    def columnCount(self, parent=QModelIndex()):
        return len(self.headers)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return self.headers[section]
        return None

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or role != Qt.ItemDataRole.DisplayRole:
            return None
        record = self.session.measurements.get(self.ids[index.row()])
        if record is None:
            return None
        operator = self.session.operators.get(record.operator_id)
        material = next((m.name for m in operator.materials[record.direction]
                         if m.id == record.material_id), record.material_id) if operator else record.material_id
        evidence = getattr(record, "evidence", None) or {}
        state = {"saved": "已保存", "pending": "写入中", "failed": "写入失败", "disabled": "无原图"}.get(evidence.get("state"), "无原图")
        method = {"assisted": "贴边测径", "free": "自由测量", "automatic-traditional": "传统全图",
                  "automatic-multiscale": "多尺度全图"}.get(record.measurement_method, record.measurement_method)
        return (str(index.row() + 1), operator.name if operator else record.operator_id, material, f"{record.diameter_um:.4f}",
                "经向" if record.direction == "warp" else "纬向", method, state)[index.column()]

    def append(self, record_id):
        row = len(self.ids)
        self.beginInsertRows(QModelIndex(), row, row)
        self.ids.append(record_id)
        self.rows_by_id[record_id] = row
        self.endInsertRows()

    def append_many(self, ids):
        ids = list(ids)
        if not ids:
            return
        row = len(self.ids)
        self.beginInsertRows(QModelIndex(), row, row + len(ids) - 1)
        self.ids.extend(ids)
        self.rows_by_id.update((record_id, row + offset) for offset, record_id in enumerate(ids))
        self.endInsertRows()

    def reset_ids(self, ids):
        self.beginResetModel()
        self.ids = list(ids)
        self.rows_by_id = {record_id: row for row, record_id in enumerate(self.ids)}
        self.endResetModel()

    def notify_record(self, record_id):
        row = self.rows_by_id.get(record_id)
        if row is not None:
            self.dataChanged.emit(self.index(row, 0), self.index(row, len(self.headers) - 1))


class CandidatesModel(QAbstractTableModel):
    exclusionChanged = Signal(object, bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.candidates = []
        self.excluded = set()
        self.materials = ()
        self.assignments = {}
        self.range_confirmations = {}
        self.scale = 1.0
        self.session = None

    def set_candidates(self, candidates, scale, materials=()):
        old_keys = getattr(self, "_candidate_keys", [])
        keys = [item.get("candidate_id", id(item)) for item in candidates]
        materials = tuple(materials)
        # Controller rows are mutated in place as acceptance progresses.
        # Keep presentation snapshots so a targeted state change is observable,
        # without invalidating selection, editors or the rest of the batch.
        rows = [dict(item) for item in candidates]
        if keys == old_keys:
            context_changed = self.scale != scale or self.materials != materials
            changed_rows = [row for row, (before, after) in enumerate(zip(self.candidates, rows))
                            if context_changed or before != after]
            self.candidates, self.scale, self.materials = rows, scale, materials
            for row in changed_rows:
                self.dataChanged.emit(self.index(row, 0), self.index(row, self.columnCount() - 1))
            return False
        assignments = {key: self.assignments[i] for i, key in enumerate(old_keys) if i in self.assignments}
        excluded = {key for i, key in enumerate(old_keys) if i in self.excluded}
        confirmations = {key: self.range_confirmations[i] for i, key in enumerate(old_keys) if i in self.range_confirmations}
        self.beginResetModel()
        self.candidates = rows
        self._candidate_keys = keys
        self.scale = scale
        self.materials = materials
        self.assignments = {i: assignments[key] for i, key in enumerate(self._candidate_keys) if key in assignments}
        self.excluded = {i for i, key in enumerate(self._candidate_keys) if key in excluded}
        self.range_confirmations = {i: confirmations[key] for i, key in enumerate(self._candidate_keys) if key in confirmations}
        self.endResetModel()
        return True

    def available(self, row):
        return self.candidates[row].get("state", "available") == "available"

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.candidates)

    def columnCount(self, parent=QModelIndex()):
        return 6

    def accepted_record(self, row):
        candidate = self.candidates[row]
        if self.session is not None and candidate.get("state") == "accepted":
            return self.session.measurements.get(candidate.get("record_id"))
        return None

    def material_id(self, row):
        record = self.accepted_record(row)
        return record.material_id if record is not None else self.assignments.get(row)

    def notify_accepted_records(self):
        # Preserve the review selection and assignments of unaccepted rows.
        for row in range(self.rowCount()):
            if self.candidates[row].get("state") == "accepted":
                self.dataChanged.emit(self.index(row, 2), self.index(row, 5))

    def assigned_material(self, row):
        material_id = self.material_id(row)
        return next((material for material in self.materials if material.id == material_id), None)

    def range_status(self, row):
        material = self.assigned_material(row)
        if material is None:
            return "未赋类"
        width = self.candidates[row]["width_px"] * self.scale
        if ((material.lower_limit is not None and width < material.lower_limit) or
                (material.upper_limit is not None and width > material.upper_limit)):
            return "越限，需核对"
        return "范围内"

    def range_signature(self, row):
        material = self.assigned_material(row)
        return None if material is None else (material.id, material.name, material.lower_limit,
            material.upper_limit, self.candidates[row]["width_px"] * self.scale)

    def range_confirmed(self, row):
        return self.range_status(row) == "越限，需核对" and self.range_confirmations.get(row) == self.range_signature(row)

    def range_description(self, row):
        material = self.assigned_material(row)
        if material is None:
            return "未赋类"
        lower = "不限" if material.lower_limit is None else f"{material.lower_limit:g}"
        upper = "不限" if material.upper_limit is None else f"{material.upper_limit:g}"
        return f"{self.range_status(row)} · {lower}～{upper} μm"

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        state = self.candidates[index.row()].get("state", "available")
        if index.column() == 0 and role == Qt.ItemDataRole.CheckStateRole:
            if state not in {"available", "excluded"}:
                return None
            return Qt.CheckState.Unchecked if index.row() in self.excluded or state == "excluded" else Qt.CheckState.Checked
        if index.column() == 5 and role == Qt.ItemDataRole.CheckStateRole and self.range_status(index.row()) == "越限，需核对":
            return Qt.CheckState.Checked if self.range_confirmed(index.row()) else Qt.CheckState.Unchecked
        if index.column() == 3 and role in (Qt.ItemDataRole.EditRole, Qt.ItemDataRole.UserRole):
            return self.material_id(index.row())
        if index.column() == 2 and role == Qt.ItemDataRole.ToolTipRole:
            original = self.candidates[index.row()]["width_px"] * self.scale
            record = self.accepted_record(index.row())
            return (f"原候选测值 {original:.4f} μm；当前记录 {record.diameter_um:.4f} μm。修订不会改变原始测量几何。"
                    if record is not None else f"候选测值 {original:.4f} μm，尚未入账。")
        if role == Qt.ItemDataRole.DisplayRole:
            material = self.assigned_material(index.row())
            original = self.candidates[index.row()]["width_px"] * self.scale
            record = self.accepted_record(index.row())
            diameter = f"{original:.4f}" + ("（原候选）" if record is not None and abs(record.diameter_um - original) > 1e-8 else "")
            return ({"accepted": "已计入", "confirming": "待计入", "excluded": "已排除"}.get(state, ""),
                    f"C{self.candidates[index.row()].get('ordinal', index.row() + 1)}", diameter,
                    material.name if material else "未赋类 · 双击选择", self.range_description(index.row()),
                    "确认保留" if self.range_status(index.row()) == "越限，需核对" else "")[index.column()]
        if role == Qt.ItemDataRole.ForegroundRole and index.column() in (3, 4) and self.range_status(index.row()) != "范围内":
            from fdm.ui.theme import semantic_colors
            colors = semantic_colors()
            return QColor(colors.warning if self.assigned_material(index.row()) is None else colors.error)
        return None

    def flags(self, index):
        flags = super().flags(index)
        if index.isValid() and self.candidates[index.row()].get("state") == "excluded" and index.column() == 0:
            return flags | Qt.ItemFlag.ItemIsUserCheckable
        if index.isValid() and not self.available(index.row()):
            return flags & ~Qt.ItemFlag.ItemIsEditable & ~Qt.ItemFlag.ItemIsUserCheckable
        if index.column() == 0:
            return flags | Qt.ItemFlag.ItemIsUserCheckable
        if index.column() == 5 and self.range_status(index.row()) == "越限，需核对":
            return flags | Qt.ItemFlag.ItemIsUserCheckable
        return flags | Qt.ItemFlag.ItemIsEditable if index.column() == 3 else flags

    def setData(self, index, value, role=Qt.ItemDataRole.EditRole):
        if not index.isValid():
            return False
        if not self.available(index.row()) and not (index.column() == 0 and self.candidates[index.row()].get("state") == "excluded"):
            return False
        if index.column() == 3 and role == Qt.ItemDataRole.EditRole:
            if value is not None and value not in {material.id for material in self.materials}:
                return False
            self.assignments[index.row()] = value
            self.range_confirmations.pop(index.row(), None)
            self.dataChanged.emit(index, self.index(index.row(), 5))
            return True
        if index.column() == 5 and role == Qt.ItemDataRole.CheckStateRole:
            if self.range_status(index.row()) != "越限，需核对":
                return False
            if value in (Qt.CheckState.Checked.value, Qt.CheckState.Checked):
                self.range_confirmations[index.row()] = self.range_signature(index.row())
            else:
                self.range_confirmations.pop(index.row(), None)
            self.dataChanged.emit(index, index)
            return True
        if index.column() != 0 or role != Qt.ItemDataRole.CheckStateRole:
            return False
        if value == Qt.CheckState.Checked.value or value == Qt.CheckState.Checked:
            self.excluded.discard(index.row())
        else:
            self.excluded.add(index.row())
        self.exclusionChanged.emit(self.candidates[index.row()].get("candidate_id"), index.row() in self.excluded)
        self.dataChanged.emit(index, index)
        return True

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        return ("接受", "候选", "直径 μm", "类别", "范围核对", "越限保留")[section] if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal else None


class CandidateMaterialDelegate(QStyledItemDelegate):
    """Store stable material IDs; translated names are display-only."""
    def createEditor(self, parent, option, index):
        editor = QComboBox(parent)
        editor.addItem("未赋类", None)
        for material in index.model().materials:
            editor.addItem(material.name, material.id)
        return editor

    def setEditorData(self, editor, index):
        editor.setCurrentIndex(max(0, editor.findData(index.data(Qt.ItemDataRole.EditRole))))

    def setModelData(self, editor, model, index):
        model.setData(index, editor.currentData(), Qt.ItemDataRole.EditRole)


class LiveExperimentWorkspace(TrialAssistanceMixin, QWidget):
    frameArrived = Signal()
    sourceFailed = Signal(object)

    def __init__(self, host, session, *, embedded=False):
        super().__init__(host)
        self.embedded = embedded
        self._bare_ctrl_pending = False
        self.setWindowFlags(Qt.WindowType.Widget if embedded else Qt.WindowType.Window)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setWindowTitle("实时试验 · 纤维直径与含量")
        self.resize(1240, 840)
        self.host = host
        self.project_identity = id(host.project)
        self.session = session
        self.source = None
        self._source_retiring = False
        self._closed = False
        self._source_epoch = 0
        self._frame_mailbox_lock = threading.Lock()
        self._frame_mailbox = None
        self._source_identity_mailbox = None
        self._frame_notification_pending = False
        self._frame_budget_pressure = False
        self._last_preview_pressure_at = -math.inf
        self._pending_display_id = None
        self._statistics_pool = None
        self._statistics_job = None
        # Preview never waits for field detection. The controller still checks
        # the latest painted frame before accepting a measurement into history.
        self.controller = LiveExperimentController(session, parent=self, verify_before_present=False)
        self.canvas = LiveMeasurementCanvas(self.controller)
        self.controller.frameReady.connect(self.canvas.set_frame)
        self._last_summary_revision = -1
        self._record_model_revision = -1
        self._static_generation = 0
        from fdm.live_experiment.reminders import ReminderTracker
        if not hasattr(session, "_quantity_notifications"):
            session._quantity_notifications = ReminderTracker(session)
        self._reminder_tracker = session._quantity_notifications
        self._last_dirty_state = (session.revision, session.active_operator_id, session.active_direction)
        # Opening a workbench creates a usable draft, not project data.
        self._draft_revision = session.revision
        self._feedback = QStatusBar(self)
        self._feedback.setSizeGripEnabled(False)
        self._calibration_key = None
        self._pending_calibration = None
        self._rulers = getattr(host, "_capture_rulers", None)
        if self._rulers is None:
            from fdm.live_experiment.calibration_store import load_rulers
            try:
                self._rulers = load_rulers()
            except (OSError, ValueError, TypeError) as exc:
                self._rulers = {}
                self._feedback.showMessage(str(exc))
            host._capture_rulers = self._rulers
        self._build()
        self._refresh_sample_summary()
        presenter = self._records_presenter()
        if presenter is not None:
            presenter.visibilityChanged.connect(self._sync_records_toggle)
        self._sync_records_toggle()
        from fdm.ui.measurement_feedback_overlay import MeasurementFeedbackOverlay
        self.measurement_feedback = MeasurementFeedbackOverlay(self.image_scroll.viewport(), canvas=self.canvas)
        self._restore_view_preferences()
        self._init_assistance()
        from fdm.ui.display_preferences import acquisition_display_store, AcquisitionDisplayBinding
        self.display_binding = AcquisitionDisplayBinding(self.canvas,
            store=acquisition_display_store(host), scroll_area=self.image_scroll,
            calibration_provider=lambda: self.controller.calibration,
            status_callback=lambda text: self.statusBar().showMessage(text, 8000), parent=self)
        self.display_binding.applied.connect(lambda _: self.display_choice.setCurrentIndex(
            self.display_choice.findData(self.display_binding.store.preferences.mode)))
        self._display_busy = False
        self._refresh_ruler_choices()
        from fdm.ui.trial_report_panel import TrialReportPanel
        self.report_panel = TrialReportPanel(self)
        self.live_state_label = QLabel(self)
        self.live_state_label.setMaximumWidth(360)
        self.statusBar().addPermanentWidget(self.live_state_label)
        self.statistics_progress = QLabel(self)
        self.statistics_cancel = QPushButton("取消重算", self)
        self.statistics_cancel.clicked.connect(self._cancel_statistics_rebuild)
        self.statusBar().addPermanentWidget(self.statistics_progress)
        self.statusBar().addPermanentWidget(self.statistics_cancel)
        self.statistics_progress.hide()
        self.statistics_cancel.hide()
        self.frameArrived.connect(self._drain_frame_mailbox, Qt.ConnectionType.QueuedConnection)
        self.sourceFailed.connect(self._receive_source_error, Qt.ConnectionType.QueuedConnection)
        self.canvas.pointPressed.connect(self._press)
        self.canvas.pointReleased.connect(self._release)
        self.canvas.manualGestureStarted.connect(self._manual_start)
        self.canvas.manualGestureFinished.connect(self._manual_finish)
        self.canvas.gestureRejected.connect(lambda text: self.statusBar().showMessage(text, 8000))
        self.canvas.gestureCancelled.connect(self.controller.cancel_draft)
        self.canvas.categoryRequested.connect(self._choose_material_at_cursor)
        self.controller.recordAccepted.connect(self._record_accepted)
        self.controller.recordsAccepted.connect(self._records_accepted)
        self.controller.fieldChanged.connect(lambda *_: self.current_model.reset_ids([]))
        self.controller.fieldChanged.connect(self._clear_latest_value)
        self.controller.statusChanged.connect(self._status_changed)
        self.controller.resultRejected.connect(lambda text: self.statusBar().showMessage(text, 8000))
        self.controller.measurementFeedback.connect(self._measurement_feedback_received)
        self.controller.feedbackInvalidated.connect(self.measurement_feedback.dismiss)
        self.measurement_feedback.dismissed.connect(self._measurement_feedback_dismissed)
        self.controller.candidatesChanged.connect(self._candidates_changed)
        self.controller.changed.connect(self._changed)
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setInterval(80)
        self._refresh_timer.timeout.connect(self._refresh_summary)
        self._refresh_timer.start()
        self._refresh_operators()
        self.history_model.reset_ids(session.measurements)
        self._record_model_revision = session.revision
        if not embedded:
            self._guard(self._load_static)
        self._check_reminders(announce=False)
        self._refresh_persistence_state()
        QApplication.instance().installEventFilter(self)

    def statusBar(self):
        """Inline feedback beneath the command row, never a nested main window."""
        return self._feedback

    def setCentralWidget(self, widget):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        if self.embedded:
            layout.addWidget(widget)
        else:
            from fdm.ui.trial_records import AcquisitionDockHost, RecordsPresentationController
            self._standalone_dock_host = AcquisitionDockHost(self)
            self._standalone_dock_host.setCentralWidget(widget)
            layout.addWidget(self._standalone_dock_host)
            self._standalone_records = RecordsPresentationController(self._standalone_dock_host,
                is_current=lambda workspace: (workspace is self and self.isVisible() and not self._closed
                    and self.project_identity == id(self.host.project)),
                action_callback=lambda workspace, action, key: self._guard(lambda: self._record_action(action, key)))

    def _button(self, label, callback, layout):
        button = QPushButton(label)
        button.clicked.connect(lambda: self._guard(callback))
        layout.addWidget(button)
        return button

    def _build(self):
        from fdm.ui.live_experiment_view import build_trial_view
        build_trial_view(self)

    def _restore_view_preferences(self):
        import json
        from fdm.settings import settings_file_path
        self._view_preferences_path = settings_file_path().parent / "trial-workbench.json"
        self._view_preferences_timer = QTimer(self)
        self._view_preferences_timer.setSingleShot(True)
        self._view_preferences_timer.setInterval(250)
        self._view_preferences_timer.timeout.connect(self._persist_view_preferences)
        try:
            preferences = json.loads(self._view_preferences_path.read_text(encoding="utf-8"))
            saved = preferences.get(self.session.workspace_options["mode"], {})
            tab = max(0, min(2, int(saved.get("tab", 0))))
            # Version 1's middle tab was records, now presented independently.
            if preferences.get("version", 1) < 2 and tab == 1:
                tab = 0
            width = max(390, min(520, int(saved.get("sidebar_width", 432))))
            expanded = bool(saved.get("metadata_expanded", False))
        except (OSError, ValueError, TypeError, AttributeError):
            tab, width, expanded = 0, 432, False
        self.sample_details_toggle.setChecked(expanded)
        self.workflow_tabs.setCurrentIndex(tab)
        self.body_splitter.setSizes([max(320, self.width() - width), width])
        self.workflow_tabs.currentChanged.connect(lambda _: self._view_preferences_timer.start())
        self.body_splitter.splitterMoved.connect(lambda *_: self._view_preferences_timer.start())
        hub = getattr(self.host, "_acquisition_workbench", None)
        if self.embedded and hub is not None:
            hub.register_splitter(self.body_splitter)

    def _persist_view_preferences(self):
        import json
        from fdm.atomic_io import atomic_write_text
        self._view_preferences_timer.stop()
        try:
            try:
                preferences = json.loads(self._view_preferences_path.read_text(encoding="utf-8"))
                if not isinstance(preferences, dict):
                    preferences = {}
            except (OSError, ValueError, TypeError):
                preferences = {}
            preferences[self.session.workspace_options["mode"]] = {
                "tab": self.workflow_tabs.currentIndex(),
                "sidebar_width": max(390, min(520, self.trial_sidebar.width())),
                "metadata_expanded": self.sample_details_toggle.isChecked(),
            }
            if hasattr(self, "auto_analysis"):
                preferences["assistance"] = self._assistance_preferences()
            preferences["version"] = 3
            self._view_preferences_path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_text(self._view_preferences_path, json.dumps(preferences, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        except OSError as exc:
            self.statusBar().showMessage(f"本次布局已应用，偏好未能保存：{exc}", 8000)

    def _metadata_expansion_changed(self, expanded):
        if hasattr(self, "_view_preferences_timer"):
            self._view_preferences_timer.start()

    def _refresh_sample_summary(self):
        if hasattr(self, "sample_summary"):
            text = f"{self.session.sample_number or '未填编号'} · {self.session.sample_name or '未命名样品'}"
            self.sample_summary.setText(text)
            self.sample_summary.setToolTip(text)

    def _records_presenter(self):
        hub = getattr(self.host, "_acquisition_workbench", None)
        return getattr(hub, "records", None) if self.embedded else getattr(self, "_standalone_records", None)

    def _notify_records(self, keys=None, *, rebuild=False):
        presenter = self._records_presenter()
        if presenter is not None:
            presenter.notify_change(self, keys=keys, rebuild=rebuild)

    def _queue_records_notification(self, keys=None):
        """Leave ledger/dirty state synchronous; merge secondary view work."""
        if not hasattr(self, "_records_notification_timer"):
            self._records_notification_keys = set()
            self._records_notification_all = False
            self._records_notification_timer = QTimer(self)
            self._records_notification_timer.setSingleShot(True)
            self._records_notification_timer.timeout.connect(self._flush_records_notification)
        if keys is None:
            self._records_notification_all = True
        else:
            self._records_notification_keys.update(map(tuple, keys))
        if not self._records_notification_timer.isActive():
            self._records_notification_timer.start(0)

    def _flush_records_notification(self):
        keys = None if self._records_notification_all else list(self._records_notification_keys)
        self._records_notification_keys.clear()
        self._records_notification_all = False
        if not self._closed:
            self._notify_records(keys)

    def _show_records(self):
        presenter = self._records_presenter()
        if presenter is not None:
            if presenter.workspace is self and presenter.is_open:
                presenter.close()
            else:
                presenter.open(self)

    def _sync_records_toggle(self, visible=None):
        presenter = self._records_presenter()
        self.records_toggle.setChecked(bool(presenter is not None
            and presenter.workspace is self and (presenter.is_open if visible is None else visible)))

    def _confirm_trial_navigation(self):
        controller = self.controller
        if not (controller.draft or controller._jobs or controller.pending_classification
                or controller.has_unaccepted_candidates or controller._confirmation):
            return True
        dialog = QMessageBox(self)
        dialog.setWindowTitle("还有未入账的测量")
        dialog.setText("离开当前试验会放弃未完成线段和未入账结果；已接受的实测与补计继续保留。")
        leave = dialog.addButton("放弃未入账结果并离开", QMessageBox.ButtonRole.DestructiveRole)
        stay = dialog.addButton("留在试验", QMessageBox.ButtonRole.RejectRole)
        dialog.setDefaultButton(stay)
        dialog.exec()
        return dialog.clickedButton() is leave

    def _return_to_capture(self):
        hub = getattr(self.host, "_acquisition_workbench", None)
        if self.embedded and hub is not None:
            hub.select_page(0)
        elif self._confirm_trial_navigation():
            self.close()

    def showEvent(self, event):
        super().showEvent(event)
        if hasattr(self, "evidence_review"):
            self.evidence_review.resume()
            self._refresh_assistance_state()
        gallery = getattr(self.host, "_evidence_gallery", None)
        if gallery is not None:
            gallery.resume(self)

    def hideEvent(self, event):
        self.controller.configure_automatic(active=False)
        if hasattr(self, "evidence_review"):
            self.evidence_review.suspend()
        gallery = getattr(self.host, "_evidence_gallery", None)
        if gallery is not None and gallery.workspace is self:
            gallery.suspend()
        self.canvas.cancel_pointer_gesture()
        if hasattr(self, "_view_preferences_timer") and self._view_preferences_timer.isActive():
            self._persist_view_preferences()
        super().hideEvent(event)

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() in (QEvent.Type.PaletteChange, QEvent.Type.ApplicationPaletteChange) and hasattr(self, "workflow_tabs"):
            from fdm.ui.live_experiment_view import apply_trial_theme
            apply_trial_theme(self)
            self.canvas.update()

    def _table(self, model):
        table = ExperimentRecordsTable()
        table.setModel(model)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        table.setAlternatingRowColors(True)
        table.horizontalHeader().setStretchLastSection(True)
        return table

    def _guard(self, callback):
        try:
            return callback()
        except Exception as exc:
            self.statusBar().showMessage(str(exc), 10000)
            return None

    def _status_changed(self, message):
        self.statusBar().showMessage(message)
        if self.controller.state == "unavailable":
            if hasattr(self, "display_binding"):
                self.display_binding.clear_source()
            self.canvas.set_frame(None)

    def _stop_source(self, *, preserve_field=False):
        with self._frame_mailbox_lock:
            if not self._source_retiring:
                self._source_epoch += 1
            self._source_identity_mailbox = None
            if self._frame_mailbox is not None:
                self.controller.budget.release(self._frame_mailbox[2])
            self._frame_mailbox = None
        if self.source is None:
            self._source_retiring = False
            return True
        self._source_retiring = True
        if not preserve_field:
            self.controller.unavailable("正在停止旧图像来源，测量已暂停。")
        try:
            stopped = self.source.stop() is True
            detail = ""
        except Exception as exc:
            stopped, detail = False, f"（{exc}）"
        if not stopped:
            if preserve_field:
                self.controller.unavailable("旧图像来源未能安全暂停，测量已停止。")
            # The old callback epoch is intentionally invalid, so its error
            # callback cannot be relied on to explain retirement failures.
            self.statusBar().showMessage("旧图像来源仍在释放资源，测量已暂停；请稍后再次切换来源或关闭窗口。" + detail)
            return False
        self.source = None
        self._source_retiring = False
        return True

    def _source_callbacks(self):
        epoch = self._source_epoch
        return (lambda frame: self._queue_frame(epoch, frame),
                lambda error: self._queue_source_error(epoch, error))

    def _queue_frame(self, epoch, frame):
        """Producer threads retain only the newest frame and queue one GUI wakeup."""
        with self._frame_mailbox_lock:
            if self._closed or epoch != self._source_epoch:
                return
            # Source identity is lightweight and must reach the GUI even when
            # the new larger pixel buffer cannot enter the bounded mailbox.
            self._source_identity_mailbox = (epoch, (frame.source_id, frame.generation, frame.size))
            old_lease = self._frame_mailbox[2] if self._frame_mailbox is not None else None
            try:
                lease = self.controller.budget.replace(old_lease, frame, preview=True,
                    reserve_bytes=self.controller._acquisition_reservation(frame))
            except FrameBudgetExceeded:
                # The newly produced frame is still source handoff memory;
                # do not retain it if another preview has yet to be painted.
                self._frame_budget_pressure = True
            else:
                self._frame_mailbox = (epoch, frame, lease)
            notify = not self._frame_notification_pending
            self._frame_notification_pending = True
        if notify:
            self.frameArrived.emit()

    def _drain_frame_mailbox(self):
        with self._frame_mailbox_lock:
            pending, self._frame_mailbox = self._frame_mailbox, None
            identity, self._source_identity_mailbox = self._source_identity_mailbox, None
            pressure, self._frame_budget_pressure = self._frame_budget_pressure, False
            self._frame_notification_pending = False
        if identity and not self._closed and identity[0] == self._source_epoch:
            self.controller.observe_source_identity(identity[1])
        if pending:
            try:
                frame_identity = (pending[1].source_id, pending[1].generation, pending[1].size)
                if (not self._closed and pending[0] == self._source_epoch
                        and (identity is None or frame_identity == identity[1])):
                    self._receive_frame(pending[1])
            finally:
                self.controller.budget.release(pending[2])
        if pressure and not self._closed and time.perf_counter() - self._last_preview_pressure_at >= 2:
            self._last_preview_pressure_at = time.perf_counter()
            self.statusBar().showMessage("图像缓冲暂满，已跳过过期预览；当前草稿和视场记录保留，原图写入完成后可继续测量。", 5000)

    def _queue_source_error(self, epoch, error):
        with self._frame_mailbox_lock:
            if self._closed or epoch != self._source_epoch:
                return
        self.sourceFailed.emit((epoch, str(error)))

    def _receive_source_error(self, failure):
        epoch, error = failure
        if not self._closed and epoch == self._source_epoch:
            self.controller.unavailable(error)

    def _load_static(self):
        document = self.host.current_document()
        if document is None:
            self.statusBar().showMessage("请在主窗口打开图片，或从上方开始采集。")
            return
        if not self._stop_source():
            return False
        metadata = {"document_id": document.id, "path": document.path}
        if hasattr(self.host, "_rasters"):
            from fdm.raster import RasterPixelType
            from fdm.services.raster_io import raster_plane_to_numpy
            plane = self.host._rasters.get(document.id)
            if plane is None or plane.is_empty:
                raise ValueError("当前文档没有可供实时试验使用的原始栅格。")
            if plane.pixel_type not in (RasterPixelType.GRAY8, RasterPixelType.RGB8, RasterPixelType.RGBA8):
                raise ValueError("实时试验需要原始 8 位图像；请先创建明确转换的 8 位图像副本。")
            pixels = raster_plane_to_numpy(plane)
            metadata.update(raster_pixel_type=plane.pixel_type.value, raster_sha256=plane.sha256())
            if plane.pixel_type is not RasterPixelType.GRAY8:
                import cv2
                code = cv2.COLOR_RGB2BGR if plane.pixel_type is RasterPixelType.RGB8 else cv2.COLOR_RGBA2BGRA
                pixels = cv2.cvtColor(pixels, code)
                pixels.setflags(write=False)
            # GRAY8 retains immutable RasterPlane.data through the array's base;
            # color planes own exactly one channel-order conversion.
            image = None
        else:
            image = self.host._images.get(document.id)
            if image is None or image.isNull():
                raise ValueError("当前文档没有可供实时试验使用的图像。")
        self.controller.end_field("已载入当前图片")
        self._static_generation += 1
        if image is None:
            fmt = "GRAY" if pixels.ndim == 2 else ("BGR" if pixels.shape[2] == 3 else "BGRA")
            frame = FrameHandle("document:" + document.id, self._static_generation, 0,
                time.perf_counter(), pixels, fmt, False, metadata)
        else:
            frame = frame_from_qimage(image, source_id="document:" + document.id,
                generation=self._static_generation, sequence=0, live=False, metadata=metadata)
        self._receive_frame(frame)
        if document.calibration:
            self.controller.set_calibration(CalibrationSnapshot.from_calibration(document.calibration, self._static_generation))
        self._refresh_ruler_choices()

    def _start_camera(self, *, preserve_field=False):
        from fdm.live_experiment.sources import CameraFrameSource
        if self.source is not None and self.source.__class__.__name__ == "CameraFrameSource":
            return True
        if not self._stop_source():
            return False
        manager = self.host._capture_manager
        if not manager.is_preview_active():
            self.host.start_live_preview()
        if not manager.is_preview_active():
            return False
        if not preserve_field:
            self.controller.unavailable("正在等待采集完整帧…")
        self.source = CameraFrameSource(manager, *self._source_callbacks())
        self.source.start()
        return True

    def _receive_frame(self, frame):
        if self._closed:
            return
        if id(self.host.project) != self.project_identity:
            self.close()
            return
        if self.controller.receive_frame(frame):
            self._restore_calibration(frame)
            self.controller.present_preview_frame(frame)
            self.display_binding.set_source(frame.source_id, frame.size)

    def _calibrate(self):
        # Reuse the application's ruler library, including imported CU .scl.
        if hasattr(self.host, "add_calibration_preset"):
            count = len(self.host._calibration_presets())
            self.host.add_calibration_preset()
            self._refresh_ruler_choices()
            if len(self.host._calibration_presets()) > count:
                self.ruler_choice.setCurrentIndex(count + 1)
                self._select_ruler(count + 1)
        else:
            self.statusBar().showMessage("请从主窗口的标尺库选择或导入 CU 标尺。")

    def _import_ruler(self):
        if hasattr(self.host, "import_cu_calibration_presets"):
            self.host.import_cu_calibration_presets()
            self._refresh_ruler_choices()
            self.statusBar().showMessage("标尺库已刷新，请从上方选择当前显微镜使用的标尺。", 6000)

    def _select_display_mode(self, index):
        mode = self.display_choice.itemData(index)
        try:
            if self.controller._jobs or self.controller.pending_classification or self.controller._confirmation or self.controller.candidate_pending_count:
                raise ValueError("请先完成或取消当前测量／候选，再更换显示模式。")
            if mode == "percent":
                value, accepted = QInputDialog.getDouble(self, "指定比例显示", "显示比例（%）",
                    self.display_binding.store.preferences.percent, 1, 1600, 1)
                if accepted:
                    self.display_binding.set_mode(mode, value)
            elif mode == "magnification":
                if not self.controller.calibration or not self.controller.calibration.isotropic:
                    raise ValueError("请先选择等比例显微镜标尺，再设置物理显示倍率。")
                from fdm.ui.live_display import request_magnification
                values = request_magnification(self, self.display_binding.store.preferences.magnification)
                if values:
                    self.display_binding.set_mode(mode, values[0])
            else:
                self.display_binding.set_mode(mode)
        except (ValueError, OSError) as exc:
            self.statusBar().showMessage(str(exc), 10000)
        finally:
            self.display_choice.setCurrentIndex(self.display_choice.findData(self.display_binding.store.preferences.mode))

    def _refresh_ruler_choices(self):
        combo = self.ruler_choice
        combo.blockSignals(True)
        combo.clear()
        combo.addItem("选择已有标尺…", None)
        current = self.controller.calibration or self._pending_calibration
        choices = getattr(self.host, "_calibration_presets", lambda: [])()
        for preset in choices:
            try:
                value = CalibrationSnapshot.from_calibration(preset.to_calibration())
            except ValueError:
                continue
            combo.addItem(f"{preset.name} · {value.x_um:.6g} μm/px", value)
        selected = 0
        if current:
            for i in range(1, combo.count()):
                value = combo.itemData(i)
                if (value.source, value.x_um, value.y_um) == (current.source, current.x_um, current.y_um):
                    selected = i
                    break
            if not selected:
                combo.addItem(f"{current.source or '已保存标尺'} · {current.x_um:.6g} μm/px", current)
                selected = combo.count() - 1
        combo.setCurrentIndex(selected)
        combo.blockSignals(False)
        self._refresh_measurement_readiness()

    def _select_ruler(self, index):
        value = self.ruler_choice.itemData(index)
        if value is None:
            return
        revision = self.controller.calibration.revision + 1 if self.controller.calibration else 1
        value = replace(value, revision=revision)
        self._pending_calibration = value
        self.controller.set_calibration(value)
        self._remember_ruler(value)
        self._refresh_measurement_readiness()
        self.canvas._update_pixel_extent()
        self.canvas.update()
        if self.canvas.frame:
            self.display_binding.set_source(self.canvas.frame.source_id, self.canvas.frame.size)

    def _remember_ruler(self, value):
        frame = self.controller.latest_frame
        if frame is not None and frame.live:
            from fdm.live_experiment.calibration_store import save_ruler, source_key
            key = source_key(frame)
            self._rulers[key] = value
            self._calibration_key = key
            self._pending_calibration = None
            try:
                save_ruler(key, value)
            except (OSError, ValueError, TypeError) as exc:
                self.statusBar().showMessage(f"本次标尺已应用，但未能保存到下次使用：{exc}")

    def _restore_calibration(self, frame):
        if not frame.live:
            return
        from fdm.live_experiment.calibration_store import source_key
        key = source_key(frame)
        if key == self._calibration_key and self._pending_calibration is None and (
                self.controller.calibration is not None or key not in self._rulers):
            return
        value = self._pending_calibration or self._rulers.get(key)
        self._calibration_key = key
        if value is not None:
            self.controller.set_calibration(value)
            if self._pending_calibration:
                self._remember_ruler(value)
        self._refresh_ruler_choices()
        self.canvas._update_pixel_extent()

    def _refresh_measurement_readiness(self):
        if not hasattr(self, "readiness_label"):
            return
        value = self.controller.calibration
        operator = self.session.operators.get(self.session.active_operator_id)
        if self.controller.latest_frame is None or self.controller.state == "unavailable":
            text = '等待画面 · <a href="device">选择或检查采集设备</a>'
        elif value is None:
            text = '下一步 · <a href="ruler">选择测量标尺</a>，或从“管理”导入 CU 标尺。'
        elif operator is None:
            text = '下一步 · <a href="operator">添加操作者</a>'
        elif operator.name == "操作者 A":
            text = '试验准备 · <a href="operator">填写实际操作者姓名</a>'
        elif self.mode.currentIndex() == 0 and self.material.count() == 0:
            text = '下一步 · <a href="materials">选择本方向的纤维类别</a>'
        else:
            text = ""
        self.readiness_label.setText(text)
        self.readiness_label.setVisible(bool(text))
        if self.readiness_label.property("needsRuler") != (value is None):
            self.readiness_label.setProperty("needsRuler", value is None)
            self.readiness_label.style().unpolish(self.readiness_label)
            self.readiness_label.style().polish(self.readiness_label)

    def _readiness_action(self, target):
        if target == "device":
            callback = getattr(self.host, "show_capture_device_menu", None)
            if callback is not None:
                self._guard(callback)
        elif target == "ruler":
            self.ruler_choice.setFocus()
            self.ruler_choice.showPopup()
        elif target == "operator":
            self._guard(self._rename_operator if self.session.active_operator_id in self.session.operators else self._add_operator)
        elif target == "materials":
            self._guard(self._materials)

    def _manual_ready(self):
        if self.mode.currentIndex() == 0 and self.material.currentData() is None:
            self.controller.cancel_draft()
            self._refresh_measurement_readiness()
            self.statusBar().showMessage("请先选择本方向的纤维类别，再在图像上测量。", 8000)
            if hasattr(self, "choose_categories_button"):
                self.choose_categories_button.setFocus()
            return False
        if self.controller.calibration is None:
            self._refresh_measurement_readiness()
            self.ruler_choice.setFocus()
            return False
        return True

    def _manual_start(self, point, token):
        if self._manual_ready():
            self._measurement_guard(lambda: self.controller.start(point, token=token))

    def _manual_finish(self, point):
        if self.controller.draft and self._manual_ready():
            self._measurement_guard(lambda: self.controller.finish(point, self.material.currentData(), method=self.measure_mode.currentData()))
        self.canvas.update()

    def _press(self, point):
        if not self._manual_ready():
            return
        if self.gesture.currentIndex() == 0 and self.controller.draft:
            self._measurement_guard(lambda: self.controller.finish(point, self.material.currentData(), method=self.measure_mode.currentData()))
        else:
            self._measurement_guard(lambda: self.controller.start(point))

    def _release(self, point):
        if self.gesture.currentIndex() == 1 and self.controller.draft:
            self._measurement_guard(lambda: self.controller.finish(point, self.material.currentData(), method=self.measure_mode.currentData()))
        self.canvas.update()

    def _measurement_guard(self, callback):
        try:
            return callback()
        except FrameBudgetExceeded:
            # The controller already published a structured resource_limit
            # event. Keep its reason code and avoid replacing the same card.
            return None
        except Exception as exc:
            from fdm.live_experiment.feedback import feedback_for_reason
            feedback = feedback_for_reason(exc, **self.controller.feedback_context())
            self._measurement_feedback_received(feedback)
            self.statusBar().showMessage(feedback.message, 8000)

    def _refresh_operators(self):
        self.operator.blockSignals(True)
        self.operator.clear()
        for operator in self.session.operators.values():
            self.operator.addItem(operator.name, operator.id)
        self.operator.setCurrentIndex(max(0, self.operator.findData(self.session.active_operator_id)))
        self.operator.blockSignals(False)
        self.direction.blockSignals(True)
        self.direction.setCurrentIndex(max(0, self.direction.findData(self.session.active_direction)))
        self.direction.blockSignals(False)
        self._refresh_materials()
        self._sync_weight_controls()

    def _refresh_materials(self):
        current = self.material.currentData()
        blocked = self.material.blockSignals(True)
        self.material.clear()
        operator = self.session.operators.get(self.session.active_operator_id)
        if operator:
            for index, material in enumerate(operator.materials[self.session.active_direction]):
                self.material.addItem(f"{index + 1} · {material.name}", material.id)
        self.material.setCurrentIndex(max(0, self.material.findData(current)))
        self.material.blockSignals(blocked)
        self._rebuild_material_panel()
        empty = self.mode.currentIndex() == 0 and self.material.count() == 0
        if hasattr(self, "empty_categories"):
            self.empty_categories.setVisible(empty)
            self.category_scroll.setVisible(not empty)
        self._sync_material_buttons()
        self._last_summary_revision = -1

    def _rebuild_material_panel(self):
        from fdm.live_experiment.presentation import category_color
        grid = self.material_panel_layout
        while grid.count():
            item = grid.takeAt(0)
            if item.widget():
                item.widget().hide()  # Deferred deletion must not leave old values painted.
                item.widget().deleteLater()
        self.material_rows = {}
        for column, text in enumerate(("类别 1–9", "", "补计", "", "均径", "含量%", "实测/目标")):
            label = QLabel(text, self.material_panel)
            label.setObjectName("trialColumnHeader")
            if column == 4:
                label.setToolTip("原始实测直径的算术均值；不使用分组中心，不受补计数影响。")
            grid.addWidget(label, 0, column)
            label.show()
        operator = self.session.operators.get(self.session.active_operator_id)
        for index, material in enumerate(operator.materials[self.session.active_direction] if operator else ()):
            select = QPushButton(f"{index + 1} · {material.name}")
            select.setCheckable(True)
            select.setMinimumWidth(78)
            select.setFixedHeight(32)
            select.setToolTip(f"{index + 1} · {material.name}；先选用于下一条，后选跟随最新测量；取消或失败后不改上一条。")
            color = category_color(self.session, operator.id, self.session.active_direction, material.id)
            marker = QLabel()
            marker.setFixedWidth(5)
            marker.setMinimumHeight(20)
            marker.setStyleSheet(f"background-color: {color};")
            selector = QWidget()
            selector_layout = QHBoxLayout(selector)
            selector_layout.setContentsMargins(0, 0, 0, 0)
            selector_layout.setSpacing(3)
            selector_layout.addWidget(marker)
            selector_layout.addWidget(select, 1)
            select.clicked.connect(lambda checked=False, key=material.id: self._guard(lambda: self._select_material(key)))
            add, subtract = QPushButton("+"), QPushButton("−")
            for button, delta in ((add, 1), (subtract, -1)):
                button.setFixedSize(28, 32)
                button.setToolTip("只增减本类别的补计数；不改变选中类别或实测直径。")
                button.clicked.connect(lambda checked=False, key=material.id, delta=delta:
                    self._guard(lambda: self._count_material(key, delta)))
            row = {"select": select, "color": marker, "add": add, "subtract": subtract,
                "extra_count": QLabel("0"), "mean": QLabel("—"), "content": QLabel("—"),
                "measured_count": QLabel("0")}
            for column, widget in enumerate((selector, add, row["extra_count"], subtract,
                                             row["mean"], row["content"], row["measured_count"])):
                if isinstance(widget, QLabel):
                    widget.setAlignment(Qt.AlignmentFlag.AlignCenter)
                grid.addWidget(widget, index + 1, column)
            subtract.setEnabled(False)
            self.material_rows[material.id] = row
        grid.setColumnStretch(0, 1)

    def _sync_material_buttons(self, *_):
        selected = self.material.currentData()
        self.canvas.active_material_id = selected
        if hasattr(self, "current_class_label"):
            self.current_class_label.setText("测径类：" + (self.material.currentText() or "未选择类别"))
        for material_id, row in getattr(self, "material_rows", {}).items():
            row["select"].setChecked(material_id == selected)
        self.canvas.update()

    def _select_material(self, material_id):
        self.controller.cancel_draft()
        index = self.material.findData(material_id)
        if index < 0:
            raise ValueError("当前方向未配置此类别。")
        self.material.setCurrentIndex(index)
        self._sync_material_buttons()
        self._after_select()

    def _choose_material_at_cursor(self, position):
        """CU's right-click shortcut, with the same explicit before/after rule."""
        self.controller.cancel_draft()
        if self.mode.currentIndex() != 0:
            return
        menu = QMenu(self.canvas)
        menu.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        menu.setTitle("快捷选类")
        operator = self.session.operators.get(self.session.active_operator_id)
        for index, material in enumerate(operator.materials[self.session.active_direction] if operator else (), 1):
            action = menu.addAction(f"{index} · {material.name}")
            action.setData(material.id)
            action.setCheckable(True)
            action.setChecked(material.id == self.material.currentData())
            action.triggered.connect(lambda checked=False, key=material.id:
                                     self._guard(lambda: self._select_material(key)))
        if not menu.actions():
            menu.deleteLater()
            return
        self._category_menu = menu
        menu.popup(position)

    def _classification_changed(self, index):
        self.after_select.blockSignals(True)
        self.after_select.setChecked(index == 1)
        self.after_select.blockSignals(False)
        self._configure(workspace_options={"classification": "after" if index else "before"})

    def _refresh_material_row_stats(self, summary):
        from fdm.live_experiment.reminders import reminder_progress
        targets = {item.material_id: item for item in reminder_progress(self.session) if item.kind == "measured"}
        operator = summary.get("operators", {}).get(self.session.active_operator_id, {})
        categories = operator.get(self.session.active_direction, {}).get("categories", [])
        if isinstance(categories, dict):
            categories = categories.values()
        for category in categories:
            row = self.material_rows.get(category["id"])
            if row is None:
                continue
            row["extra_count"].setText(str(category["extra_count"]))
            target = targets.get(category["id"])
            row["measured_count"].setText(
                f"{'✓ ' if target.reached else ''}{category['measured_count']}/{target.target}" if target
                else str(category["measured_count"]))
            row["measured_count"].setToolTip(f"实测 {category['measured_count']} 根" +
                (f"，目标 {target.target} 根；{'已达到目标' if target.reached else '可继续测量'}" if target else ""))
            row["mean"].setText("—" if category["mean"] is None else f"{category['mean']:.3f}")
            row["content"].setText("—" if category["content"] is None else f"{category['content']:.2f}")
            row["subtract"].setEnabled(category["extra_count"] > 0)

    def _add_operator(self):
        name, ok = QInputDialog.getText(self, "添加操作者", "姓名")
        if ok and name.strip():
            self.controller.end_field("已结束当前视场并切换操作者；历史记录保留。")
            self.session.active_operator_id = self.session.add_operator(name.strip())
            self._refresh_operators()
            self._changed()

    def _operator_changed(self, index):
        if index < 0 or self.operator.currentData() == self.session.active_operator_id:
            return
        self.controller.end_field("已结束当前视场并切换操作者；历史记录保留。")
        self.session.active_operator_id = self.operator.currentData()
        self._refresh_materials()
        self._sync_weight_controls()
        self._last_summary_revision = -1
        self._changed()

    def _rename_operator(self):
        operator = self.session.operators[self.session.active_operator_id]
        name, ok = QInputDialog.getText(self, "操作者改名", "姓名（保留此人的全部记录）", text=operator.name)
        if ok:
            self.session.rename_operator(name)
            self._refresh_operators()
            for model in (self.current_model, self.history_model):
                if model.rowCount():
                    model.dataChanged.emit(model.index(0, 1), model.index(model.rowCount() - 1, 1))
            self._changed()

    def _direction_changed(self, index):
        if self.direction.currentData() == self.session.active_direction:
            return
        if self.controller.current_records or self.controller.pending_classification or self.controller._jobs:
            self.direction.blockSignals(True)
            self.direction.setCurrentIndex(self.direction.findData(self.session.active_direction))
            self.direction.blockSignals(False)
            self.statusBar().showMessage("本场已有测量，请先点击“新视场”再切换经纬方向；历史记录保留。", 10000)
            return
        self.controller.end_field("已切换经纬方向")
        self.session.active_direction = self.direction.currentData()
        self._refresh_materials()
        self._last_summary_revision = -1
        self._changed()

    def _count(self, delta):
        self._count_material(self.material.currentData(), delta)

    def _count_material(self, material_id, delta):
        if self.mode.currentIndex() != 0:
            return
        event = self.session.adjust_count(material_id, delta, field_id=self.controller.field_id)
        self._notify_records([("count", event.id)])
        self._changed()

    def _after_select(self):
        if (self.mode.currentIndex() == 0 and self.after_select.isChecked()
                and (not hasattr(self, "input_mode") or self.input_mode.currentData() != "candidate")):
            record_id = self.controller.reclassify_latest(self.material.currentData())
            if record_id is not None:
                self.current_model.notify_record(record_id)
                self.history_model.notify_record(record_id)
                self._notify_records([("measurement", record_id)])
            self._refresh_summary()

    def _weights(self):
        operator = self.session.operators[self.session.active_operator_id]
        dialog = QDialog(self)
        dialog.setWindowTitle("经纬重量")
        form = QFormLayout(dialog)
        inputs = []
        for label, value in (("经向重量", operator.warp_weight), ("纬向重量", operator.weft_weight)):
            spin = QDoubleSpinBox()
            spin.setRange(0, 1000000)
            spin.setDecimals(6)
            spin.setValue(value or 0)
            form.addRow(label, spin)
            inputs.append(spin)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        if dialog.exec():
            self.session.set_weights(inputs[0].value(), inputs[1].value())
            self._sync_weight_controls()
            self._changed()

    def _sync_weight_controls(self):
        operator = self.session.operators.get(self.session.active_operator_id)
        if operator is None:
            return
        for attribute in ("warp_weight", "weft_weight"):
            editor = getattr(self, attribute)
            editor.blockSignals(True)
            editor.setValue(getattr(operator, attribute) or 0)
            editor.blockSignals(False)

    def _weights_edited(self):
        operator = self.session.operators.get(self.session.active_operator_id)
        if operator is None:
            return
        values = self.warp_weight.value(), self.weft_weight.value()
        if values != (operator.warp_weight or 0, operator.weft_weight or 0):
            self._guard(lambda: self.session.set_weights(*values))
            self._changed()

    def _materials(self):
        from fdm.live_experiment import MaterialSnapshot, built_in_materials
        current = self.session.operators[self.session.active_operator_id].materials[self.session.active_direction]
        catalog = {m.id: m for m in built_in_materials()}
        catalog.update({m.id: m for m in current})
        dialog = QDialog(self)
        dialog.setWindowTitle("本方向类别 · 材料参数保存在本试验内")
        dialog.resize(830, 470)
        layout = QVBoxLayout(dialog)
        note = QLabel("最多九类。范围只用于越限提醒，不用于自动认定材料；密度为 0 时含量将显示待完善。")
        note.setWordWrap(True)
        layout.addWidget(note)
        table = QTableWidget(9, 5)
        table.setHorizontalHeaderLabels(["材料", "名称", "密度 g/cm³", "下限 μm", "上限 μm"])
        table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(table)
        rows = []
        for row in range(9):
            selected = current[row] if row < len(current) else None
            choice = QComboBox()
            choice.addItem("不使用", None)
            for material in catalog.values():
                choice.addItem(material.name, material.id)
            choice.addItem("自定义材料", "__custom__")
            name = QLineEdit()
            density, low, high = QDoubleSpinBox(), QDoubleSpinBox(), QDoubleSpinBox()
            for spin in (density, low, high):
                spin.setDecimals(6)
                spin.setRange(0, 1000000)
            low.setSpecialValueText("不限")
            high.setSpecialValueText("不限")
            density.setMaximum(100)
            def load(index, choice=choice, name=name, density=density, low=low, high=high):
                material = catalog.get(choice.currentData())
                name.setText(material.name if material else "")
                density.setValue((material.density or 0) if material else 0)
                low.setValue((material.lower_limit or 0) if material else 0)
                high.setValue((material.upper_limit or 0) if material else 0)
            choice.currentIndexChanged.connect(load)
            choice.setCurrentIndex(max(0, choice.findData(selected.id if selected else None)))
            for col, widget in enumerate((choice, name, density, low, high)):
                table.setCellWidget(row, col, widget)
            rows.append((choice, name, density, low, high))
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if not dialog.exec():
            return
        materials = []
        for choice, name, density, low, high in rows:
            key = choice.currentData()
            if key is None:
                continue
            base = catalog.get(key)
            values = dict(name=name.text().strip(), density=density.value(),
                          lower_limit=low.value() or None, upper_limit=high.value() or None)
            material = replace(base, **values) if base else MaterialSnapshot(id=uuid.uuid4().hex, family="other", **values)
            materials.append(material)
        self.session.set_materials(materials, direction=self.session.active_direction)
        self.controller.end_field("类别设置已变化")
        self._refresh_materials()
        self._changed()

    def _statistics_settings(self):
        if self._statistics_job is not None:
            raise ValueError("组距重算尚未结束，当前设置仍生效；请等待或取消后再打开计算设置。")
        from fdm.live_experiment.statistics import METHODS
        dialog = QDialog(self)
        dialog.setWindowTitle("含量计算设置 · 恢复的 CU 计算版本")
        form = QFormLayout(dialog)
        method = QComboBox()
        method.addItems([m.label for m in METHODS])
        method.setCurrentText(self.session.method)
        grouping = QComboBox()
        for width in (1.0, 1.5, 2.0, 2.5):
            grouping.addItem(f"{width:g}", width)
        if grouping.findData(self.session.group_width) < 0:
            grouping.addItem(f"{self.session.group_width:g}（当前自定义）", self.session.group_width)
        grouping.addItem("不分组", None)
        grouping.setCurrentIndex(grouping.findData(self.session.group_width) if self.session.grouped else grouping.count() - 1)
        profile = QComboBox()
        profile.addItems(["普通", "免脱色"])
        profile.setCurrentIndex(0 if self.session.profile == "normal" else 1)
        for label, widget in (("方法", method), ("组距 μm", grouping), ("汇总舍入", profile)):
            form.addRow(label, widget)
        note = QLabel("方法名表示恢复的 CU 版本，不表示现行标准认证。原始均径与分组计算均径可能不同。")
        note.setWordWrap(True)
        form.addRow(note)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        if dialog.exec():
            selected_width = grouping.currentData()
            self._configure(method=method.currentText(), grouped=selected_width is not None,
                group_width=selected_width if selected_width is not None else self.session.group_width,
                profile="normal" if profile.currentIndex() == 0 else "mt")

    def _configure(self, **kwargs):
        def configure():
            if self._statistics_job is not None:
                raise ValueError("组距重算尚未结束，当前设置仍生效；请等待或取消后再更改设置。")
            request = self.session.configure_deferred(**kwargs)
            if request is not None:
                self._start_statistics_rebuild(request)
        self._guard(configure)
        # Toggle/edit signals may already have changed a widget. Reflect only
        # committed values, including when a second request was rejected.
        self._sync_configuration_controls()
        self._changed()

    def _start_statistics_rebuild(self, request):
        if self._statistics_pool is None:
            self._statistics_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="fdm-content-rebuild")
        cancelled = threading.Event()
        future = self._statistics_pool.submit(request.build, cancelled.is_set)
        self._statistics_job = (request, future, cancelled)
        self.statistics_button.setEnabled(False)
        self.statistics_progress.setText(f"组距 {request.group_width:g} μm 后台重算，尚未应用")
        self.statistics_progress.show()
        self.statistics_cancel.setEnabled(True)
        self.statistics_cancel.show()
        self.statusBar().showMessage("当前设置与实时画面保持有效；重算期间若记录改变，本次结果会丢弃，请稍后重试。", 10000)

    def _cancel_statistics_rebuild(self):
        if self._statistics_job is not None:
            _, future, cancelled = self._statistics_job
            cancelled.set()
            future.cancel()
            self.statistics_progress.setText("正在取消组距重算，当前设置保留")
            self.statistics_cancel.setEnabled(False)

    def _poll_statistics_rebuild(self):
        job = self._statistics_job
        if job is None or not job[1].done():
            return
        request, future, cancelled = job
        self._statistics_job = None
        self.statistics_button.setEnabled(True)
        self.statistics_progress.hide()
        self.statistics_cancel.hide()
        try:
            aggregates = future.result()
            if cancelled.is_set():
                raise CancelledError()
            if self._closed or id(self.host.project) != self.project_identity:
                raise CancelledError()
            self.session.apply_rebuild(request, aggregates)
        except CancelledError:
            self.statusBar().showMessage("组距重算已取消，已保留当前记录和原设置。", 8000)
        except Exception as exc:
            self.statusBar().showMessage(f"组距重算未应用：{exc}", 12000)
        else:
            self.statusBar().showMessage(f"组距 {self.session.group_width:g} μm 已应用。", 6000)
            self._sync_configuration_controls()
            self._changed()

    def _sample_changed(self):
        self._configure(sample_name=self.sample.text().strip())
        self.sample_title.setText(self.session.sample_name or "未命名样品")

    def _gesture_changed(self, index):
        if self._statistics_job is not None:
            self.statusBar().showMessage("组距重算尚未结束，未改变测量方式或草稿；请等待或取消后再切换。", 8000)
            self._sync_configuration_controls()
            return
        before = self.session.workspace_options["gesture"]
        desired = "drag" if index else "click"
        self._configure(workspace_options={"gesture": desired})
        # Cancel only after a real, accepted change. A rejected configuration
        # must not silently discard the measurement the user is still drawing.
        if before != desired and self.session.workspace_options["gesture"] == desired:
            self.controller.cancel_draft()
            if hasattr(self, "auto_analysis"):
                self._apply_assistance_controls()

    def _mode_changed(self, index, persist=True):
        content = index == 0
        if hasattr(self, "category_area"):
            self.category_area.setVisible(content)
        for widget in (self.direction, self.keyboard_mode, self.classification_mode, self.material_panel, self.summary, self.summary_scope,
                       self.weights_button, self.statistics_button, self.direction_row, self.category_actions,
                       self.current_class_label):
            widget.setVisible(content)
        self.diameter_summary.setVisible(not content)
        self.operation_help.setText("两次点击横跨纤维完成测径。Esc 取消草稿；右键选类；Ctrl 切换数字键的选类／补计数模式。"
            if content else "两次点击横跨纤维完成测径。Esc 取消草稿；已入账记录可在记录页修订、删除或查看原图。")
        if persist:
            self._configure(workspace_options={**getattr(self.session, "workspace_options", {}),
                                              "mode": "content" if content else "diameter"})

    def _record_accepted(self, record):
        self.measurement_feedback.dismiss()
        self._latest_value_is_previous = False
        self.measurement_outcome.setText("已计入")
        self.latest_value.setText(f"{record.diameter_um:.3f} μm")
        self.statusBar().showMessage(f"本次有效测量 {record.diameter_um:.3f} μm 已计入记录。", 8000)
        self.current_model.append(record.id)
        self.history_model.append(record.id)
        self.current_table.follow_latest()
        self._record_model_revision = self.session.revision
        self._queue_records_notification([("measurement", record.id)])

    def _records_accepted(self, records):
        self.measurement_feedback.dismiss()
        self._latest_value_is_previous = False
        self.measurement_outcome.setText("已计入")
        if records:
            self.latest_value.setText(f"{records[-1].diameter_um:.3f} μm")
            self.statusBar().showMessage(f"已接受 {len(records)} 条有效测量并计入记录。", 8000)
        ids = [record.id for record in records]
        self.current_model.append_many(ids)
        self.history_model.append_many(ids)
        self.current_table.follow_latest()
        self._queue_records_notification([("measurement", record.id) for record in records])
        self._record_model_revision = self.session.revision

    def _candidates_changed(self, candidates):
        current = self.candidate_table.currentIndex()
        selected = (self.candidate_model.candidates[current.row()].get("candidate_id")
                    if current.isValid() and current.row() < self.candidate_model.rowCount() else None)
        scale = self.controller.calibration.x_um if self.controller.calibration else 1
        operator = self.session.operators.get(self.session.active_operator_id)
        materials = operator.materials[self.session.active_direction] if operator else ()
        reset = self.candidate_model.set_candidates(candidates, scale, materials)
        if reset and selected is not None:
            self._candidate_hovered(selected)
        self.candidate_feedback.setText(f"待审核 {len(candidates)} 条候选；确认接受后才计入试验。" if candidates else "")
        self.candidate_feedback.hide()  # Runtime messages must not move the category list.
        self._refresh_assistance_state()
        self.canvas.update()

    def _assign_candidate_material(self):
        material_id = self.material.currentData()
        if material_id is None:
            raise ValueError("请先选择有效的当前类别。")
        for row in range(self.candidate_model.rowCount()):
            if row not in self.candidate_model.excluded and self.candidate_model.available(row):
                self.candidate_model.setData(self.candidate_model.index(row, 3), material_id, Qt.ItemDataRole.EditRole)

    def _accept_candidates(self):
        selections = [(i, self.candidate_model.assignments.get(i)) for i in range(len(self.controller.candidates))
                      if i not in self.candidate_model.excluded and self.candidate_model.available(i)]
        if any(material is None for _, material in selections):
            raise ValueError("仍有勾选候选未赋类；请逐条选择类别或批量设置后确认。")
        out_of_range = {index for index, _ in selections if self.candidate_model.range_status(index) == "越限，需核对"}
        confirmed = {index for index in out_of_range if self.candidate_model.range_confirmed(index)}
        if out_of_range != confirmed:
            raise ValueError("请核对表中越限候选的直径和范围，勾选确认保留，或改类／取消接受。")
        duplicates = self.controller.possible_duplicate_candidates([index for index, _ in selections])
        if duplicates and QMessageBox.question(self, "核对可能重复的候选",
                f"有 {len(duplicates)} 条候选的位置和宽度接近本视场已接受记录。请核对是否为同一根纤维；仍然全部入账？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No) != QMessageBox.StandardButton.Yes:
            return
        self.controller.accept_candidates(selections, confirmed_out_of_range=confirmed)

    def _changed(self):
        state = (self.session.revision, self.session.active_operator_id, self.session.active_direction)
        if id(self.host.project) == self.project_identity and state != self._last_dirty_state:
            registered = any(session is self.session for session in self.host.project.live_experiments)
            if not registered and self.session.revision != self._draft_revision:
                self.host.project.live_experiments.append(self.session)
                registered = True
            if registered:
                self.host.project.mark_extension_changed()
            self._last_dirty_state = state
        self._refresh_persistence_state()
        if hasattr(self, "display_binding"):
            self.display_binding.apply()
        ui_state = (*state, self.controller.field_epoch)
        if ui_state == getattr(self, "_last_ui_change_state", None):
            self.canvas.update()
            return
        self._last_ui_change_state = ui_state
        self._refresh_sample_summary()
        self._queue_records_notification()
        self._check_reminders()
        self.canvas.update()

    def _refresh_persistence_state(self):
        registered = any(session is self.session for session in self.host.project.live_experiments)
        self.save_button.setEnabled(registered)
        if not registered:
            text = "尚无需要保存的试验改动"
        elif getattr(self.host, "_project_dirty", lambda: True)() or not getattr(self.host, "_project_path", None):
            text = "试验数据有更改 · 保存项目可保留样品、记录和设置"
        else:
            text = "试验数据已保存到项目"
        self.persistence_label.setText(text)

    def _refresh_summary(self):
        self._poll_statistics_rebuild()
        if id(self.host.project) != self.project_identity:
            self.close()
            return
        frame, calibration = self.controller.latest_frame, self.controller.calibration
        self._refresh_measurement_readiness()
        self._refresh_persistence_state()
        self._refresh_assistance_state()
        mode = "含量试验" if self.mode.currentIndex() == 0 else "直径试验"
        self.sample_subtitle.setText(f"{mode} · 本场实测 {len(self.controller.current_records)} · 累计实测 {len(self.session.measurements)}")
        source = "相机" if frame and frame.live else "图片" if frame else "无来源"
        state = {"ready": "视场就绪", "stabilizing": "视场核验中", "unavailable": "来源不可用"}.get(
            self.controller.state, self.controller.state)
        ruler = ("未标定" if calibration is None else
                 f"{calibration.x_um:.6g} μm/px" if calibration.isotropic else
                 f"X {calibration.x_um:.6g} / Y {calibration.y_um:.6g} μm/px")
        self.live_state_label.setText(f"{source} · {state} · {ruler}")
        pending = self.controller.pending_classification
        previous_pending_id = self._pending_display_id
        self._pending_display_id = pending[0][0].id if pending else None
        if pending:
            intent, result = pending[0]
            material = self.session.material(intent.material_id, operator_id=intent.operator_id, direction=intent.direction)
            lower = "不限" if material.lower_limit is None else f"{material.lower_limit:g}"
            upper = "不限" if material.upper_limit is None else f"{material.upper_limit:g}"
            text = (f"待确认 1/{len(pending)} · 测量 #{intent.measurement_number or intent.id[:6]}\n"
                    f"{material.name} · {result['diameter_um']:.4f} μm · 范围 {lower}–{upper} μm\n"
                    "下方操作仅作用于此条；橙色虚线表示尚未入账。")
            target = self.controller._classification_target
            if self.after_select.isChecked() and target and target["id"] != intent.id:
                text += f"\n后选当前作用于最新测量 #{target['number']}，与首条待确认不同。"
            self.pending_label.setText(text)
        else:
            request = self.controller._confirmation
            self.pending_label.setText("确认正在核验最新视场，尚未入账。" if request else "")
        self.pending_panel.setVisible(bool(pending or (self.controller._confirmation
            and self.controller._confirmation["kind"] == "manual")))
        if pending and previous_pending_id != self._pending_display_id:
            self.workflow_tabs.setCurrentIndex(0)
            scroll = self.workflow_tabs.widget(0)
            QTimer.singleShot(0, self, lambda: scroll.ensureWidgetVisible(self.pending_panel, 0, 12))
        for button in self.pending_buttons:
            button.setEnabled(bool(self.controller.pending_classification) and self.controller._confirmation is None)
        if self.session.revision == self._last_summary_revision:
            return
        self.candidate_model.notify_accepted_records()
        from fdm.live_experiment.statistics import summarize
        summary = summarize(self.session)
        self._refresh_material_row_stats(summary)
        operator = summary.get("operators", {}).get(self.session.active_operator_id, {})
        measured_count, measured_sum = 0, 0.0
        for direction_name in ("warp", "weft"):
            groups = operator.get(direction_name, {}).get("categories", [])
            for group in groups.values() if isinstance(groups, dict) else groups:
                count = group.get("measured_count", 0)
                measured_count += count
                measured_sum += count * (group.get("mean") or 0.0)
        mean = f"{measured_sum / measured_count:.3f}" if measured_count else "—"
        self.diameter_summary.setText(f"当前操作者：累计实测 {measured_count} 根\n平均直径 {mean} μm")
        scope = self.summary_scope.currentIndex()
        direction = (operator.get(self.session.active_direction, {}) if scope == 0 else
                     operator.get("mixed", {}) if scope == 1 else summary["fiber" if scope == 2 else "fabric"])
        categories = direction.get("categories", [])
        if isinstance(categories, dict):
            categories = list(categories.values())
        lines = [f"{self.session.method}", f"{'分组 ' + str(self.session.group_width) if self.session.grouped else '原始直径'} · {self.summary_scope.currentText()}"]
        for category in categories:
            percentage = category.get("content")
            content = f"{percentage:.2f}%" if isinstance(percentage, (int, float)) else "待完善"
            counts = f"实测 {category['measured_count']} + 计数 {category['extra_count']} · " if "measured_count" in category else ""
            mean = category.get("calculation_mean", category.get("mean"))
            mean_text = f" · 计算均径 {mean:.3f} μm" if mean is not None else ""
            lines.append(f"{category['name']}：{counts}{content}{mean_text}")
        reasons = direction.get("reasons", [])
        lines.extend(reason.get("message", str(reason)) if isinstance(reason, dict) else str(reason) for reason in reasons[:3])
        self.summary.setText("\n\n".join(lines))
        self._last_summary_revision = self.session.revision
        self.quantity_targets.refresh()
        if self.history_model.rowCount():
            self.history_model.dataChanged.emit(self.history_model.index(0, 6), self.history_model.index(self.history_model.rowCount() - 1, 6))

    def _resolve_displayed_pending(self, **kwargs):
        if self._pending_display_id is None:
            raise ValueError("请先核对当前待确认测量。")
        self.controller.resolve_pending(intent_id=self._pending_display_id, **kwargs)
        self._refresh_summary()

    def _history_action(self, action):
        if self._statistics_job is not None:
            raise ValueError("组距重算尚未结束，未改动撤销历史；请等待或取消后再撤销／重做。")
        controller = self.controller
        if (controller.draft is not None
                or any(controller._valid(intent) for intent, _ in controller._jobs)
                or controller.pending_classification
                or controller.candidate_pending_count or controller._confirmation):
            raise ValueError("手工测量尚未完成，未撤销之前记录；请取消草稿，或等待计算／核验完成并处理待确认项后，再撤销或重做。")
        self.controller.cancel_draft()
        before = (self.session.active_operator_id, self.session.active_direction,
                  repr([operator.materials for operator in self.session.operators.values()]))
        request = self.session.history_deferred(action)
        if request is not None:
            self._start_statistics_rebuild(request)
            return
        after = (self.session.active_operator_id, self.session.active_direction,
                 repr([operator.materials for operator in self.session.operators.values()]))
        if before != after:
            self.controller.end_field("试验配置已撤销或恢复，请重新测量")
        self._refresh_operators()
        self._sync_configuration_controls()
        self._rebuild_records()

    def _sync_configuration_controls(self):
        for widget, setter, value in (
                (self.sample, "setText", self.session.sample_name),
                (self.sample_number, "setText", self.session.sample_number),
                (self.keep_images, "setChecked", self.session.save_images),
                (self.evidence_storage, "setCurrentIndex", self.evidence_storage.findData(self.session.effective_evidence_options["mode"])),
                (self.after_select, "setChecked", self.session.workspace_options["classification"] == "after"),
                (self.classification_mode, "setCurrentIndex", int(self.session.workspace_options["classification"] == "after")),
                (self.mode, "setCurrentIndex", int(self.session.workspace_options["mode"] == "diameter")),
                (self.gesture, "setCurrentIndex", int(self.session.workspace_options["gesture"] == "drag"))):
            widget.blockSignals(True)
            getattr(widget, setter)(value)
            widget.blockSignals(False)
        self._mode_changed(self.mode.currentIndex(), persist=False)
        self._sync_weight_controls()
        if hasattr(self, "auto_analysis"):
            self._apply_assistance_controls()
            self._refresh_assistance_state()

    def _rebuild_records(self):
        # History and record edits can invalidate both a rejected attempt's
        # reference value and a target notice. Keep notification history, but
        # rebuild this presentation from the records that are still effective.
        self.measurement_feedback.dismiss()
        self.controller.current_records[:] = [key for key, record in self.session.measurements.items()
                                               if record.field_id == self.controller.field_id]
        self.current_model.reset_ids(self.controller.current_records)
        self.history_model.reset_ids(self.session.measurements)
        if self.controller.current_records:
            latest = self.session.measurements[self.controller.current_records[-1]]
            self._latest_value_is_previous = False
            self.latest_value.setText(f"{latest.diameter_um:.3f} μm")
            self.measurement_outcome.setText("最近有效测量")
            self.statusBar().showMessage(f"记录已更新；本视场最近有效测量为 {latest.diameter_um:.3f} μm。", 8000)
        else:
            self._clear_latest_value()
            self.statusBar().showMessage("本视场暂无有效测量记录。", 8000)
        self.controller.sync_candidate_records()
        self.candidate_model.notify_accepted_records()
        self._notify_records(rebuild=True)
        self._changed()

    def _selected_record(self):
        key = getattr(self, "_record_action_key", None)
        if key is not None:
            return self.session.measurements.get(key[1]) if key[0] == "measurement" else None
        if self.workflow_tabs.currentWidget() is self.candidate_review_page or self.tabs.currentIndex() not in (0, 1):
            raise ValueError("请先在当前视场或试验历史页选择一条测量记录；全图候选尚未入账。")
        table = self.history_table if self.tabs.currentIndex() == 1 else self.current_table
        index = table.currentIndex()
        return self.session.measurements.get(table.model().ids[index.row()]) if index.isValid() else None

    def _record_action(self, action, key):
        if self._closed or id(self.host.project) != self.project_identity:
            return
        if self.embedded:
            hub = getattr(self.host, "_acquisition_workbench", None)
            if hub is None or hub.pages.currentWidget() is not self or not self.host._acquisition_workspace_selected:
                return
        if key[0] == "count":
            if action != "correct_count" or key[1] not in self.session.count_adjustments:
                return
            event = self.session.count_adjustments[key[1]]
            owner = self.session.operators[event.operator_id]
            material = self.session.material(event.material_id, operator_id=event.operator_id, direction=event.direction)
            balance = sum(row.delta for row in self.session.count_adjustments.values()
                          if (row.operator_id, row.direction, row.material_id) ==
                          (event.operator_id, event.direction, event.material_id))
            delta, accepted = QInputDialog.getInt(self, "更正补计", (
                f"{owner.name}／{'经向' if event.direction == 'warp' else '纬向'}／{material.name}\n"
                f"当前补计 {balance} 根。填写本次增减量（正数增加，负数减少）："),
                0, -min(balance, 2147483647), 2147483647)
            if accepted and delta:
                change = self.session.adjust_count(event.material_id, delta, operator_id=event.operator_id,
                    direction=event.direction, field_id=event.field_id, note=f"更正补计记录 {event.id}")
                self._notify_records([("count", change.id)])
                self._changed()
            return
        callbacks = {"edit": self._edit_record, "delete": self._delete_record, "evidence": self._review_record}
        if key[0] != "measurement" or key[1] not in self.session.measurements or action not in callbacks:
            return
        self._record_action_key = key
        try:
            callbacks[action]()
        finally:
            self._record_action_key = None

    def _delete_record(self):
        record = self._selected_record()
        if record:
            self.session.delete_measurement(record.id)
            self._rebuild_records()

    def _edit_record(self):
        record = self._selected_record()
        if not record:
            return
        value, ok = QInputDialog.getDouble(self, "修订测量", "直径 μm（原始算法结果保留在来源信息中）", record.diameter_um, 0.000001, 1000000, 6)
        if ok:
            materials = self.session.operators[record.operator_id].materials[record.direction]
            choices = [f"{index + 1} · {material.name}" for index, material in enumerate(materials)]
            selected, accepted = QInputDialog.getItem(self, "修订分类", "记录所属操作者／方向的类别", choices,
                next((i for i, m in enumerate(materials) if m.id == record.material_id), 0), False)
            if not accepted:
                return
            self.session.edit_measurement(record.id, diameter_um=value, material_id=materials[choices.index(selected)].id)
            self._rebuild_records()

    def _review_record(self):
        record = self._selected_record()
        if record is None:
            raise ValueError("请先选择一条测量记录。")
        self.evidence_review.show_record(record, latest=False)

    def _discard_failed_evidence(self):
        record = self._selected_record()
        if not record or (record.evidence or {}).get("state") != "failed":
            raise ValueError("请在记录表中选择一条原图写入失败或缺失的记录。")
        if QMessageBox.question(self, "保留测量并放弃失败原图", "测量数值和来源信息继续保留。确认将这条记录标为无原图？") != QMessageBox.StandardButton.Yes:
            return
        self.session.set_evidence(record.id, {"state": "disabled", "reason": "用户明确放弃失败原图", "previous_error": record.evidence.get("error")})
        self._changed()

    def _check_reminders(self, announce=True):
        notice = self._reminder_tracker.update(self.session)
        displayed = getattr(self, "_reminder_feedback_crossings", ())
        feedback = self.measurement_feedback.feedback
        if feedback is not None and feedback.code == "target_reached" and displayed:
            from fdm.live_experiment.reminders import all_reminder_progress
            current = {item.key: item for item in all_reminder_progress(self.session)}
            if any(current.get(item.key) != item for item in displayed):
                # A count, target, name or scope changed after this notice.
                # Dismiss obsolete text without clearing the tracker's seen
                # set, so undo/redo cannot replay an acknowledged target.
                self.measurement_feedback.dismiss()
        if hasattr(self, "quantity_targets"):
            self.quantity_targets.refresh()
        if notice and announce and self.isVisible():
            from fdm.live_experiment.feedback import MeasurementFeedback
            self._reminder_feedback_crossings = notice.crossings
            self.measurement_feedback.show_feedback(MeasurementFeedback(
                code="target_reached", severity="success", message=notice.text + "。可以继续测量，或进入报告页复核。",
                **self.controller.feedback_context()))

    def _measurement_feedback_received(self, feedback):
        if self._closed or not self.isVisible():
            return
        if feedback.field_epoch is not None and feedback.field_epoch != self.controller.field_epoch:
            return
        if not self.controller.feedback_is_current_interaction(feedback):
            return
        self._latest_value_is_previous = self.latest_value.text() != "— μm"
        self.measurement_outcome.setText("上一条有效值" if self._latest_value_is_previous else "本次未通过")
        self.measurement_outcome.setToolTip("本次测量未通过，未计入记录；" + feedback.message +
            ("；上方为上一条有效测量" if self._latest_value_is_previous else ""))
        self.measurement_feedback.show_feedback(feedback)

    def _measurement_feedback_dismissed(self, feedback):
        if feedback is not None and feedback.code == "target_reached":
            self._reminder_feedback_crossings = ()
        if feedback is not None and feedback.severity != "success":
            self.measurement_outcome.setText("上一条有效值" if getattr(self, "_latest_value_is_previous", False) else "")

    def _clear_latest_value(self, *_):
        review = getattr(self, "evidence_review", None)
        record = self.session.measurements.get(review.latest_record_id) if review is not None else None
        self.latest_value.setText(f"{record.diameter_um:.3f} μm" if record else "— μm")
        self.measurement_outcome.setText("上一条" if record else "")
        self._latest_value_is_previous = record is not None

    def _new_experiment(self):
        if self._confirm_trial_navigation():
            self._replace_experiment(_new_trial_session(self.session.workspace_options["mode"]))

    def _choose_experiment(self):
        mode = self.session.workspace_options["mode"]
        sessions = [s for s in self.host.project.live_experiments if s.workspace_options["mode"] == mode]
        if self.session not in sessions:
            sessions.append(self.session)
        choices = [f"{i + 1} · {s.sample_name or s.name}" for i, s in enumerate(sessions)] + ["＋ 新建试验"]
        selected, ok = QInputDialog.getItem(self, "试验", "选择已有试验，或建立独立样品记录", choices, sessions.index(self.session), False)
        if not ok or selected == choices[sessions.index(self.session)]:
            return
        target = None if selected == choices[-1] else sessions[choices.index(selected)]
        if target is None:
            self._new_experiment()
        elif self._confirm_trial_navigation():
            self._replace_experiment(target)

    def _replace_experiment(self, target):
        mode = target.workspace_options["mode"]
        self._gallery_rebinding = True
        try:
            if not self.close():
                return
        finally:
            self._gallery_rebinding = False
        if self.embedded:
            self.host._acquisition_workbench.pages.removeWidget(self)
        window = LiveExperimentWorkspace(self.host, target, embedded=self.embedded)
        self.host._live_experiment_workspace = window
        self.host._live_experiment_workspaces[mode] = window
        if self.embedded:
            hub = self.host._acquisition_workbench
            hub.pages.addWidget(window)
            hub.pages.setCurrentWidget(window)
            hub.records.bind(window)
            hub.apply_sidebar_width()
            if self.host._capture_manager.is_preview_active():
                window._start_camera()
        window.show()

    def commit_pending_edits(self):
        """Flush focused editors before a project snapshot or accepted close."""
        if self._closed or id(self.host.project) != self.project_identity:
            return
        if self._statistics_job is not None:
            raise ValueError("组距设置尚未应用，请等待或取消重算后保存。")
        sample_name, sample_number = self.sample.text().strip(), self.sample_number.text().strip()
        self.warp_weight.interpretText()
        self.weft_weight.interpretText()
        self._weights_edited()
        self._configure(sample_name=sample_name, sample_number=sample_number)
        self.sample_title.setText(self.session.sample_name or "新试验")

    def _save_project(self):
        self.commit_pending_edits()
        self._changed()
        if not any(session is self.session for session in self.host.project.live_experiments):
            self.statusBar().showMessage("尚无需要保存的试验改动；填写样品信息或开始测量后即可保存。", 8000)
            return
        if self.controller.evidence.pending or self.controller._jobs:
            raise ValueError("测量或原图尚在写入，请稍后保存；实时画面继续更新。")
        self.host.save_project_action.trigger()

    def _export(self, excel):
        if not self.session.measurements and not self.session.count_adjustments:
            raise ValueError("还没有试验记录；完成测量或补计数后再导出报告。")
        if excel:
            self.report_panel.export()
            return
        from fdm.live_experiment.exports import export_csv, session_export_xlsx
        path, _ = QFileDialog.getSaveFileName(self, "导出试验", "试验.xlsx" if excel else "试验.csv", "Excel (*.xlsx)" if excel else "CSV (*.csv)")
        if path:
            (session_export_xlsx if excel else export_csv)(self.session, path)
            self.statusBar().showMessage("已导出试验数据", 5000)

    def _open_last_report(self):
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices
        result = self.report_panel.last_report
        if not result or not Path(result["destination"]).is_file():
            raise ValueError("最近的报告文件不存在，请重新导出。")
        if not QDesktopServices.openUrl(QUrl.fromLocalFile(result["destination"])):
            raise ValueError("无法打开报告，请通过“所在位置”手动打开。")

    def _open_report_folder(self):
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices
        result = self.report_panel.last_report
        if result:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(result["destination"]).parent)))

    def _measurement_focus_allowed(self):
        focus = QApplication.focusWidget()
        presenter = self._records_presenter()
        if presenter is not None and presenter.contains_focus(focus):
            return False
        if any(focus is table or (focus is not None and table.isAncestorOf(focus))
               for table in (self.current_table, self.history_table)):
            return False
        hub = getattr(self.host, "_acquisition_workbench", None)
        if self.embedded and (hub is None or hub.pages.currentWidget() is not self):
            return False
        return (not self._closed and self.isVisible() and QApplication.activeWindow() is self.window()
            and QApplication.activePopupWidget() is None
            and QApplication.activeModalWidget() is None
            and not isinstance(focus, (QLineEdit, QAbstractSpinBox, QPlainTextEdit, QTextEdit))
            and not (isinstance(focus, QComboBox) and self.candidate_table.isAncestorOf(focus)))

    def _handle_digit_key(self, event):
        if (not self._measurement_focus_allowed() or self.mode.currentIndex() != 0
                or event.modifiers() & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.AltModifier
                                       | Qt.KeyboardModifier.MetaModifier | Qt.KeyboardModifier.ShiftModifier)):
            return False
        if Qt.Key.Key_1 <= event.key() <= Qt.Key.Key_9:
            if event.isAutoRepeat():
                return True
            index = event.key() - Qt.Key.Key_1
            if index < self.material.count():
                key = self.material.itemData(index)
                self._guard(lambda: self._count_material(key, 1) if self.keyboard_mode.currentIndex()
                            else self._select_material(key))
                return True
        if event.key() == Qt.Key.Key_Minus and self.keyboard_mode.currentIndex():
            if not event.isAutoRepeat():
                self._guard(lambda: self._count(-1))
            return True
        return False

    def eventFilter(self, watched, event):
        kind = event.type()
        if (watched is getattr(self, "latest_value", None) and kind == QEvent.Type.MouseButtonPress
                and event.button() == Qt.MouseButton.LeftButton):
            self._guard(self._show_latest_evidence)
            return True
        if kind == QEvent.Type.Wheel:
            # Also protect editors created inside Qt's static QInputDialog
            # helpers (calibration / record correction), not just our widgets.
            from PySide6.QtWidgets import QComboBox as QtComboBox
            if isinstance(watched, (QAbstractSpinBox, QtComboBox)):
                parent = watched.parent()
                while parent is not None and parent is not self:
                    parent = parent.parent()
                if parent is self:
                    from fdm.ui.widgets import _redirect_wheel_to_inspector_scroll
                    _redirect_wheel_to_inspector_scroll(watched, event)
                    self._bare_ctrl_pending = False
                    return True
        if kind in (QEvent.Type.WindowDeactivate, QEvent.Type.ApplicationDeactivate,
                     QEvent.Type.MouseButtonPress, QEvent.Type.MouseButtonDblClick, QEvent.Type.Wheel):
            self._bare_ctrl_pending = False
        elif kind in (QEvent.Type.KeyPress, QEvent.Type.ShortcutOverride):
            if event.key() != Qt.Key.Key_Control:
                self._bare_ctrl_pending = False
            elif kind == QEvent.Type.KeyPress and not event.isAutoRepeat():
                self._bare_ctrl_pending = self._measurement_focus_allowed()
            if kind == QEvent.Type.KeyPress and self._handle_digit_key(event):
                event.accept()
                return True
        elif kind == QEvent.Type.KeyRelease and event.key() == Qt.Key.Key_Control:
            if not event.isAutoRepeat():
                toggle = self._bare_ctrl_pending and self._measurement_focus_allowed()
                self._bare_ctrl_pending = False
                if toggle and self.mode.currentIndex() == 0:
                    self.keyboard_mode.setCurrentIndex(1 - self.keyboard_mode.currentIndex())
                    event.accept()
                    return True
        return super().eventFilter(watched, event)

    def keyPressEvent(self, event):
        if self._handle_digit_key(event):
            event.accept()
            return
        super().keyPressEvent(event)

    def prepare_close(self):
        """Quiesce producers and editors without discarding the visible page."""
        if self._statistics_job is not None:
            self._cancel_statistics_rebuild()
            if not self._statistics_job[1].done():
                self.statusBar().showMessage("正在取消组距重算，窗口与来源仍保留；请稍后再次关闭。")
                return False
            self._poll_statistics_rebuild()
        if self.controller.evidence.pending:
            self.statusBar().showMessage("正在保存测量原图，请完成后关闭。")
            return False
        if self.source is not None:
            self._transition_resume_camera = True
        if not self._stop_source(preserve_field=True):
            return False
        # Static-image trials have no source subscription to stop. Their
        # in-flight results must also be invalidated before a save snapshot.
        self.controller.cancel_pending_measurements("正在确认关闭或切换项目")
        self.commit_pending_edits()
        return True

    def resume_after_cancel(self):
        restart = getattr(self, "_transition_resume_camera", False)
        self._transition_resume_camera = False
        manager = getattr(self.host, "_capture_manager", None)
        if (restart and manager is not None and not self._closed and self.isVisible()
                and id(self.host.project) == self.project_identity
                and manager.is_preview_active() and not manager.is_stopping()):
            self._start_camera(preserve_field=True)

    def closeEvent(self, event):
        if not self.prepare_close():
            event.ignore()
            return
        self._closed = True
        if self._view_preferences_timer.isActive():
            self._persist_view_preferences()
        self._view_preferences_timer.stop()
        self.display_binding.close()
        self.settings_dialog.close()
        self.report_panel.close()
        self.evidence_review.close()
        gallery = getattr(self.host, "_evidence_gallery", None)
        if gallery is not None and gallery.workspace is self:
            if getattr(self, "_gallery_rebinding", False):
                gallery.suspend()
                gallery.bind(None)
            else:
                gallery.close()
                self.host._evidence_gallery = None
        presenter = self._records_presenter()
        if presenter is not None and presenter.workspace is self:
            if self.embedded:
                presenter.suspend()
            else:
                presenter.shutdown()
        QApplication.instance().removeEventFilter(self)
        self._refresh_timer.stop()
        if self._statistics_pool is not None:
            self._statistics_pool.shutdown(wait=False, cancel_futures=True)
            self._statistics_pool = None
        self.canvas.set_frame(None)
        if getattr(self.host, "_live_experiment_workspace", None) is self:
            self.host._live_experiment_workspace = None
        cache = getattr(self.host, "_live_experiment_workspaces", {})
        for mode, workspace in tuple(cache.items()):
            if workspace is self:
                del cache[mode]
        self.controller.close(record_end=getattr(self, "_close_record_end", True))
        super().closeEvent(event)


def _new_trial_session(mode):
    """Content categories are explicit user input; diameter keeps its defaults."""
    from fdm.live_experiment import ExperimentSession, default_materials
    session = ExperimentSession(workspace_options={"mode": mode},
        evidence_options={"version": 1, "mode": "crop_jpeg"})
    session.add_operator("操作者 A")
    if mode == "diameter":
        session.set_materials(default_materials(), direction="warp")
        session.set_materials(default_materials(), direction="weft")
    session.establish_history_baseline()
    return session


def open_live_experiment(host, *, embedded=False, trial_mode=None):
    """Reuse each trial page; a fresh untouched session remains outside the project."""
    if host.project.is_read_only_compatible:
        QMessageBox.information(host, "实时试验", "当前项目以只读兼容模式打开，请先另存可编辑项目。")
        return None
    cache = getattr(host, "_live_experiment_workspaces", None)
    if cache is None:
        cache = host._live_experiment_workspaces = {}
    current = getattr(host, "_live_experiment_workspace", None)
    mode = trial_mode or (current.session.workspace_options.get("mode") if current is not None else "content")
    existing = cache.get(mode)
    if existing is not None:
        if not existing._closed and existing.project_identity == id(host.project):
            host._live_experiment_workspace = existing
            if not embedded:
                existing.show()
                existing.raise_()
                existing.activateWindow()
            return existing
        cache.pop(mode, None)
    matching = [session for session in host.project.live_experiments
                if session.workspace_options.get("mode") == mode]
    if matching:
        session = matching[-1]
    else:
        session = _new_trial_session(mode)
    window = LiveExperimentWorkspace(host, session, embedded=embedded)
    cache[mode] = window
    host._live_experiment_workspace = window
    if not embedded:
        window.show()
    return window
