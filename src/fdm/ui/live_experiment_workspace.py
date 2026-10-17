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


class LiveMeasurementCanvas(QWidget):
    pointPressed = Signal(object)
    pointReleased = Signal(object)
    gestureCancelled = Signal()
    categoryRequested = Signal(object)

    def __init__(self, controller, parent=None):
        super().__init__(parent)
        self.controller = controller
        self.frame = None
        self.image = None
        self.display_mode = "pixel"
        self.display_value = 100.0
        self.screen_pixel_mm = None
        self.display_revision = 0
        self.cursor_point = None
        self.active_material_id = None
        self.setMinimumSize(400, 300)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setToolTip("")
        controller.fieldChanged.connect(self._field_changed)
        controller.changed.connect(self.update)
        controller.candidatesChanged.connect(self.update)

    def set_frame(self, frame):
        old_size = self.frame.size if self.frame is not None else None
        self.frame = frame
        if frame is None:
            self.image = None
            self._update_pixel_extent()
            self.update()
            return
        pixels = frame.pixels
        fmt = {"GRAY": QImage.Format.Format_Grayscale8,
               "BGR": QImage.Format.Format_BGR888,
               "BGRA": QImage.Format.Format_ARGB32}[frame.pixel_format]
        self.image = QImage(pixels.data, frame.size[0], frame.size[1], pixels.strides[0], fmt)
        if old_size != frame.size:
            self._update_pixel_extent()
        self.update()

    def _update_pixel_extent(self):
        # QScrollArea supplies explicit scrollbars for images larger than the
        # viewport. Never silently shrink a 768/2K/4K frame to make it fit.
        if self.frame is None or self.display_mode == "fit":
            self.setMinimumSize(320, 240)
        else:
            scale, _ = self._transform()
            self.setMinimumSize(max(320, math.ceil(self.frame.size[0] * scale)),
                                max(240, math.ceil(self.frame.size[1] * scale)))

    def _field_changed(self, *_):
        self.cursor_point = None
        self.update()

    def _transform(self):
        if self.frame is None:
            return 1.0, (0.0, 0.0)
        width, height = self.frame.size
        dpr = self.devicePixelRatioF()
        if self.display_mode == "fit":
            scale = min(self.width() / width, self.height() / height)
        elif self.display_mode == "percent":
            scale = self.display_value / 100 / dpr
        elif self.display_mode == "magnification" and self.screen_pixel_mm and self.controller.calibration:
            scale = self.controller.calibration.x_um * self.display_value / (self.screen_pixel_mm * 1000 * dpr)
        else:
            scale = 1 / dpr
        # Align image edges to device pixels. Actual-pixel mode never resamples
        # merely because the application is running at 125/150/200% DPI.
        return scale, (round((self.width() - width * scale) * dpr / 2) / dpr,
                       round((self.height() - height * scale) * dpr / 2) / dpr)

    def set_display_mode(self, mode, value=100.0, screen_pixel_mm=None):
        if mode not in {"pixel", "fit", "percent", "magnification"}:
            raise ValueError("未知显示模式")
        if self.controller.pending_classification or self.controller._confirmation:
            raise ValueError("请先核对待确认测量，再更换显示模式。")
        if self.controller._jobs or self.controller.candidates:
            raise ValueError("请先完成或取消当前测量／候选，再更换显示模式。")
        if not math.isfinite(value) or value <= 0:
            raise ValueError("显示比例必须大于零。")
        if mode == "magnification" and (not self.controller.calibration or not self.controller.calibration.isotropic
                or screen_pixel_mm is None or not math.isfinite(screen_pixel_mm) or screen_pixel_mm <= 0):
            raise ValueError("指定倍率需要等比例显微镜标尺和显示器标线长度。")
        if self.frame is not None and mode in {"percent", "magnification"}:
            scale = (value / 100 if mode == "percent" else
                     self.controller.calibration.x_um * value / (screen_pixel_mm * 1000))
            extent = max(self.frame.size) * scale / self.devicePixelRatioF()
            if not math.isfinite(extent) or extent > 16777215:
                raise ValueError("显示范围过大，请降低比例／倍率，并核对屏幕标线的毫米长度。")
        self.gestureCancelled.emit()
        self.display_mode, self.display_value = mode, value
        self.screen_pixel_mm = screen_pixel_mm
        self.cursor_point = None
        self.display_revision += 1
        self._update_pixel_extent()
        self.update()

    def fit_view(self):
        self.set_display_mode("fit")

    def paintEvent(self, event):
        painter = QPainter(self)
        from fdm.ui.theme import semantic_colors
        colors = semantic_colors(self.palette())
        painter.fillRect(self.rect(), QColor(colors.canvas))
        if self.frame is None:
            painter.setPen(QColor(colors.canvas_text))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "等待实时画面\n在采集栏连接设备，在右侧完成试验准备")
            return
        frame = self.frame
        scale, offset = self._transform()
        painter.translate(*offset)
        painter.scale(scale, scale)
        painter.drawImage(QPointF(0, 0), self.image)
        session = self.controller.session
        for number, record_id in enumerate(self.controller.current_records, 1):
            record = self.controller.session.measurements.get(record_id)
            endpoints = getattr(record, "endpoints", None) if record else None
            if endpoints and len(endpoints) == 2:
                color = QColor(category_color(session, record.operator_id, record.direction, record.material_id))
                painter.setPen(QPen(QColor(255, 255, 255, 180), 3.5 / scale))
                painter.drawLine(QPointF(*endpoints[0]), QPointF(*endpoints[1]))
                painter.setPen(QPen(color, 1.5 / scale))
                painter.drawLine(QPointF(*endpoints[0]), QPointF(*endpoints[1]))
                category = category_number(session, record.operator_id, record.direction, record.material_id)
                label = f"{number} [{category}]" if category else str(number)
                # Keep labels legible at every zoom without scaling the font.
                painter.save()
                painter.resetTransform()
                location = QPointF(endpoints[1][0] * scale + offset[0] + 5,
                                   endpoints[1][1] * scale + offset[1] - 5)
                box = QRectF(painter.fontMetrics().boundingRect(label)).translated(location).adjusted(-2, -1, 2, 1)
                painter.fillRect(box, QColor(255, 255, 255, 220))
                painter.fillRect(QRectF(box.left() - 3, box.top(), 3, box.height()), color)
                painter.setPen(QColor("#111820"))
                painter.drawText(location, label)
                painter.restore()
        painter.setPen(QPen(QColor("#ffcf65"), 1.2 / scale, Qt.PenStyle.DashLine))
        for candidate in self.controller.candidates:
            painter.drawLine(QPointF(*candidate["start"]), QPointF(*candidate["end"]))
        pending = [(intent, result, f"待确认 {index}") for index, (intent, result)
                   in enumerate(self.controller.pending_classification, 1)]
        request = self.controller._confirmation
        if request and request["kind"] == "manual":
            pending.extend((intent, result, "确认核验中") for intent, result in request["intents"])
        for intent, result, prefix in pending:
            painter.setPen(QPen(QColor("#ff9f43"), 2 / scale, Qt.PenStyle.DashLine))
            painter.drawLine(QPointF(*result["start"]), QPointF(*result["end"]))
            painter.save()
            painter.resetTransform()
            location = QPointF(result["end"][0] * scale + offset[0] + 5,
                               result["end"][1] * scale + offset[1] + 15)
            label = f"{prefix} · 测量 #{intent.measurement_number or intent.id[:6]}（未入账）"
            box = QRectF(painter.fontMetrics().boundingRect(label)).translated(location).adjusted(-2, -1, 2, 1)
            painter.fillRect(box, QColor("#fff0d5"))
            painter.setPen(QColor("#7a3900"))
            painter.drawText(location, label)
            painter.restore()
        if self.controller.draft:
            start = self.controller.draft[1]
            end = self.cursor_point or start
            color = category_color(session, session.active_operator_id,
                                   session.active_direction, self.active_material_id)
            painter.setPen(QPen(QColor(color), 2 / scale))
            painter.drawLine(QPointF(*start), QPointF(*end))
            painter.drawEllipse(QPointF(*start), 3 / scale, 3 / scale)
        painter.end()
        # Publish only after painting; set_frame merely schedules a repaint.
        self.controller.mark_presented(PresentedFrameToken(
            frame, self.controller.field_epoch, self.display_revision, scale, offset))

    def _point(self, event):
        token = self.controller.presented
        if token is None:
            return None
        return token.image_point(event.position().x(), event.position().y())

    def mousePressEvent(self, event):
        self.setFocus()
        if event.button() == Qt.MouseButton.LeftButton:
            point = self._point(event)
            if point:
                self.pointPressed.emit(point)
        self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.RightButton:
            self.gestureCancelled.emit()
            self.categoryRequested.emit(event.globalPosition().toPoint())
        elif event.button() == Qt.MouseButton.LeftButton:
            point = self._point(event)
            if point:
                self.pointReleased.emit(point)

    def mouseMoveEvent(self, event):
        self.cursor_point = self._point(event)
        if self.controller.draft:
            self.update()

    def wheelEvent(self, event):
        event.accept()

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton:
            event.accept()
        else:
            # A fast second left click remains the second measurement point.
            self.mousePressEvent(event)

    def resizeEvent(self, event):
        self.gestureCancelled.emit()
        self.display_revision += 1
        super().resizeEvent(event)

    def event(self, event):
        if event.type() == QEvent.Type.DevicePixelRatioChange:
            self.gestureCancelled.emit()
            self._update_pixel_extent()
            self.display_revision += 1
            self.update()
        return super().event(event)


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
    def __init__(self, parent=None):
        super().__init__(parent)
        self.candidates = []
        self.excluded = set()
        self.materials = ()
        self.assignments = {}
        self.range_confirmations = {}
        self.scale = 1.0

    def set_candidates(self, candidates, scale, materials=()):
        self.beginResetModel()
        self.candidates = candidates
        self.scale = scale
        self.excluded.clear()
        self.materials = tuple(materials)
        self.assignments.clear()
        self.range_confirmations.clear()
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.candidates)

    def columnCount(self, parent=QModelIndex()):
        return 6

    def assigned_material(self, row):
        material_id = self.assignments.get(row)
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
        if index.column() == 0 and role == Qt.ItemDataRole.CheckStateRole:
            return Qt.CheckState.Unchecked if index.row() in self.excluded else Qt.CheckState.Checked
        if index.column() == 5 and role == Qt.ItemDataRole.CheckStateRole and self.range_status(index.row()) == "越限，需核对":
            return Qt.CheckState.Checked if self.range_confirmed(index.row()) else Qt.CheckState.Unchecked
        if index.column() == 3 and role in (Qt.ItemDataRole.EditRole, Qt.ItemDataRole.UserRole):
            return self.assignments.get(index.row())
        if role == Qt.ItemDataRole.DisplayRole:
            material = self.assigned_material(index.row())
            return ("", str(index.row() + 1), f"{self.candidates[index.row()]['width_px'] * self.scale:.4f}",
                    material.name if material else "未赋类 · 双击选择", self.range_description(index.row()),
                    "确认保留" if self.range_status(index.row()) == "越限，需核对" else "")[index.column()]
        if role == Qt.ItemDataRole.ForegroundRole and index.column() in (3, 4) and self.range_status(index.row()) != "范围内":
            return QColor("#bd6d00" if self.assigned_material(index.row()) is None else "#d43a3a")
        return None

    def flags(self, index):
        flags = super().flags(index)
        if index.column() == 0:
            return flags | Qt.ItemFlag.ItemIsUserCheckable
        if index.column() == 5 and self.range_status(index.row()) == "越限，需核对":
            return flags | Qt.ItemFlag.ItemIsUserCheckable
        return flags | Qt.ItemFlag.ItemIsEditable if index.column() == 3 else flags

    def setData(self, index, value, role=Qt.ItemDataRole.EditRole):
        if not index.isValid():
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


class LiveExperimentWorkspace(QWidget):
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
        self._reminders_seen = set()
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
        self._restore_view_preferences()
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
        self.canvas.gestureCancelled.connect(self.controller.cancel_draft)
        self.canvas.categoryRequested.connect(self._choose_material_at_cursor)
        self.controller.recordAccepted.connect(self._record_accepted)
        self.controller.recordsAccepted.connect(self._records_accepted)
        self.controller.fieldChanged.connect(lambda *_: self.current_model.reset_ids([]))
        self.controller.fieldChanged.connect(lambda *_: self.latest_value.setText("— μm"))
        self.controller.statusChanged.connect(self._status_changed)
        self.controller.resultRejected.connect(lambda text: self.statusBar().showMessage(text, 8000))
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
        layout.addWidget(widget)

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
            width = max(390, min(520, int(saved.get("sidebar_width", 432))))
        except (OSError, ValueError, TypeError, AttributeError):
            tab, width = 0, 432
        self.workflow_tabs.setCurrentIndex(tab)
        self.body_splitter.setSizes([max(320, self.width() - width), width])
        self.workflow_tabs.currentChanged.connect(lambda _: self._view_preferences_timer.start())
        self.body_splitter.splitterMoved.connect(lambda *_: self._view_preferences_timer.start())

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
            }
            self._view_preferences_path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_text(self._view_preferences_path, json.dumps(preferences, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError as exc:
            self.statusBar().showMessage(f"本次布局已应用，偏好未能保存：{exc}", 8000)

    def hideEvent(self, event):
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
            self.canvas.set_frame(frame)
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
            if self.controller._jobs or self.controller.candidates or self.controller.pending_classification or self.controller._confirmation:
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
        elif self.controller.state != "ready":
            text = "画面持续采集中；视场稳定后继续测量。"
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

    def _press(self, point):
        if self.controller.calibration is None:
            self._refresh_measurement_readiness()
            self.ruler_choice.setFocus()
            return
        if self.gesture.currentIndex() == 0 and self.controller.draft:
            self._guard(lambda: self.controller.finish(point, self.material.currentData(), method=self.measure_mode.currentData()))
        else:
            self._guard(lambda: self.controller.start(point))

    def _release(self, point):
        if self.gesture.currentIndex() == 1 and self.controller.draft:
            self._guard(lambda: self.controller.finish(point, self.material.currentData(), method=self.measure_mode.currentData()))
        self.canvas.update()

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
        for column, text in enumerate(("类别 1–9", "", "补计", "", "均径", "含量%", "实测")):
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
        operator = summary.get("operators", {}).get(self.session.active_operator_id, {})
        categories = operator.get(self.session.active_direction, {}).get("categories", [])
        if isinstance(categories, dict):
            categories = categories.values()
        for category in categories:
            row = self.material_rows.get(category["id"])
            if row is None:
                continue
            row["extra_count"].setText(str(category["extra_count"]))
            row["measured_count"].setText(str(category["measured_count"]))
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
        self.session.adjust_count(material_id, delta, field_id=self.controller.field_id)
        self._changed()

    def _after_select(self):
        if self.mode.currentIndex() == 0 and self.after_select.isChecked():
            record_id = self.controller.reclassify_latest(self.material.currentData())
            if record_id is not None:
                self.current_model.notify_record(record_id)
                self.history_model.notify_record(record_id)
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
        dialog.setWindowTitle("计算与数量提醒 · 恢复的 CU 计算版本")
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
        measured, total = QSpinBox(), QSpinBox()
        for spin, value in ((measured, self.session.measured_reminder), (total, self.session.total_reminder)):
            spin.setRange(1, 100000000)
            spin.setValue(value)
        for label, widget in (("方法", method), ("组距 μm", grouping), ("汇总舍入", profile),
                              ("每类实测数量提醒", measured), ("本方向总根数提醒", total)):
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
                profile="normal" if profile.currentIndex() == 0 else "mt",
                measured_reminder=measured.value(), total_reminder=total.value())

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

    def _mode_changed(self, index, persist=True):
        content = index == 0
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
        self.latest_value.setText(f"{record.diameter_um:.3f} μm")
        self.current_model.append(record.id)
        self.history_model.append(record.id)
        self.current_table.follow_latest()
        self._record_model_revision = self.session.revision

    def _records_accepted(self, records):
        ids = [record.id for record in records]
        self.current_model.append_many(ids)
        self.history_model.append_many(ids)
        self.current_table.follow_latest()
        self._record_model_revision = self.session.revision

    def _candidates_changed(self, candidates):
        scale = self.controller.calibration.x_um if self.controller.calibration else 1
        operator = self.session.operators.get(self.session.active_operator_id)
        materials = operator.materials[self.session.active_direction] if operator else ()
        self.candidate_model.set_candidates(candidates, scale, materials)
        if candidates:
            self.records_toggle.setChecked(True)
            self.tabs.setCurrentIndex(2)
        self.candidate_feedback.setText(f"待审核 {len(candidates)} 条候选；确认接受后才计入试验。" if candidates else "")
        self.candidate_feedback.setVisible(bool(candidates))
        self.tabs.setTabText(2, f"候选 {len(candidates)}" if candidates else "候选审核")
        self.canvas.update()

    def _assign_candidate_material(self):
        material_id = self.material.currentData()
        if material_id is None:
            raise ValueError("请先选择有效的当前类别。")
        for row in range(self.candidate_model.rowCount()):
            if row not in self.candidate_model.excluded:
                self.candidate_model.setData(self.candidate_model.index(row, 3), material_id, Qt.ItemDataRole.EditRole)

    def _accept_candidates(self):
        selections = [(i, self.candidate_model.assignments.get(i)) for i in range(len(self.controller.candidates))
                      if i not in self.candidate_model.excluded]
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
        self.pending_panel.setVisible(bool(pending or self.controller._confirmation))
        if pending and previous_pending_id != self._pending_display_id:
            self.workflow_tabs.setCurrentIndex(0)
            scroll = self.workflow_tabs.widget(0)
            QTimer.singleShot(0, self, lambda: scroll.ensureWidgetVisible(self.pending_panel, 0, 12))
        for button in self.pending_buttons:
            button.setEnabled(bool(self.controller.pending_classification) and self.controller._confirmation is None)
        if self.session.revision == self._last_summary_revision:
            return
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
        self._check_reminders()
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
                or (controller._confirmation and controller._confirmation["kind"] == "manual")):
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
                (self.after_select, "setChecked", self.session.workspace_options["classification"] == "after"),
                (self.classification_mode, "setCurrentIndex", int(self.session.workspace_options["classification"] == "after")),
                (self.mode, "setCurrentIndex", int(self.session.workspace_options["mode"] == "diameter")),
                (self.gesture, "setCurrentIndex", int(self.session.workspace_options["gesture"] == "drag"))):
            widget.blockSignals(True)
            getattr(widget, setter)(value)
            widget.blockSignals(False)
        self._mode_changed(self.mode.currentIndex(), persist=False)
        self._sync_weight_controls()

    def _rebuild_records(self):
        self.controller.current_records[:] = [key for key, record in self.session.measurements.items()
                                               if record.field_id == self.controller.field_id]
        self.current_model.reset_ids(self.controller.current_records)
        self.history_model.reset_ids(self.session.measurements)
        self._changed()

    def _selected_record(self):
        if self.tabs.currentIndex() not in (0, 1):
            raise ValueError("请先在当前视场或试验历史页选择一条测量记录；全图候选尚未入账。")
        table = self.history_table if self.tabs.currentIndex() == 1 else self.current_table
        index = table.currentIndex()
        return self.session.measurements.get(table.model().ids[index.row()]) if index.isValid() else None

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
        evidence = getattr(record, "evidence", None) if record else None
        if not evidence or evidence.get("state") != "saved":
            raise ValueError("该记录没有可用原图。")
        image = QImage(evidence["path"])
        if image.isNull():
            raise ValueError("原图文件缺失或无法读取。")
        dialog = QDialog(self)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        dialog.setWindowTitle(f"历史原图 · {record.diameter_um:.4f} μm")
        layout = QVBoxLayout(dialog)
        label = QLabel()
        from PySide6.QtGui import QPixmap
        label.setPixmap(QPixmap.fromImage(image).scaled(1000, 700, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        layout.addWidget(label)
        dialog.show()

    def _discard_failed_evidence(self):
        record = self._selected_record()
        if not record or (record.evidence or {}).get("state") != "failed":
            raise ValueError("请在记录表中选择一条原图写入失败或缺失的记录。")
        if QMessageBox.question(self, "保留测量并放弃失败原图", "测量数值和来源信息继续保留。确认将这条记录标为无原图？") != QMessageBox.StandardButton.Yes:
            return
        self.session.set_evidence(record.id, {"state": "disabled", "reason": "用户明确放弃失败原图", "previous_error": record.evidence.get("error")})
        self._changed()

    def _check_reminders(self, announce=True):
        messages = []
        for operator_id, operator in self.session.operators.items():
            for direction in ("warp", "weft"):
                state = self.session.reminder_state(operator_id=operator_id, direction=direction)
                prefix = f"{operator.name}／{'经向' if direction == 'warp' else '纬向'}"
                for material_id in state["measured_reached"]:
                    key = (operator_id, direction, material_id, state["measured_target"])
                    if key not in self._reminders_seen:
                        name = next(m.name for m in operator.materials[direction] if m.id == material_id)
                        messages.append(f"{prefix} {name}实测已达到 {state['measured_target']} 根")
                        self._reminders_seen.add(key)
                key = (operator_id, direction, "total", state["total_target"])
                if state["total_reached"] and key not in self._reminders_seen:
                    messages.append(f"{prefix} 总根数已达到 {state['total_target']} 根")
                    self._reminders_seen.add(key)
        if messages and announce:
            self.statusBar().showMessage("；".join(messages), 15000)

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
            name, ok = QInputDialog.getText(self, "新建试验", "样品名称")
            if not ok:
                return
        if not self.close():
            return
        if self.embedded:
            self.host._acquisition_workbench.pages.removeWidget(self)
        if target is None:
            from fdm.live_experiment import ExperimentSession, default_materials
            target = ExperimentSession(sample_name=name.strip(), workspace_options={"mode": mode})
            target.add_operator("操作者 A")
            target.set_materials(default_materials(), direction="warp")
            target.set_materials(default_materials(), direction="weft")
            target.establish_history_baseline()
            if name.strip():
                self.host.project.live_experiments.append(target)
                self.host.project.mark_extension_changed()
        window = LiveExperimentWorkspace(self.host, target, embedded=self.embedded)
        self.host._live_experiment_workspace = window
        self.host._live_experiment_workspaces[mode] = window
        if self.embedded:
            hub = self.host._acquisition_workbench
            hub.pages.addWidget(window)
            hub.pages.setCurrentWidget(window)
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

    def _measurement_focus_allowed(self):
        focus = QApplication.focusWidget()
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
    from fdm.live_experiment import ExperimentSession, default_materials
    matching = [session for session in host.project.live_experiments
                if session.workspace_options.get("mode") == mode]
    if matching:
        session = matching[-1]
    else:
        session = ExperimentSession(workspace_options={"mode": mode})
        session.add_operator("操作者 A")
        session.set_materials(default_materials(), direction="warp")
        session.set_materials(default_materials(), direction="weft")
        session.establish_history_baseline()
    window = LiveExperimentWorkspace(host, session, embedded=embedded)
    cache[mode] = window
    host._live_experiment_workspace = window
    if not embedded:
        window.show()
    return window
