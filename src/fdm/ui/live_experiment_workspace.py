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
    QStyledItemDelegate, QGridLayout,
)

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
        self.zoom = 1.0
        self.pan_offset = QPointF()
        self._pan_anchor = None
        self.display_revision = 0
        self.cursor_point = None
        self.active_material_id = None
        self.setMinimumSize(400, 300)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setToolTip("右键取消草稿并快捷选类；标记为场内序号 [类别号]。滚轮缩放；中键平移；双击中键适应画面。变换显示会取消未完成测量。")
        controller.fieldChanged.connect(self._field_changed)
        controller.changed.connect(self.update)
        controller.candidatesChanged.connect(self.update)

    def set_frame(self, frame):
        if frame is None or self.frame is None or (
                frame.source_id, frame.generation, frame.size) != (
                self.frame.source_id, self.frame.generation, self.frame.size):
            self.zoom = 1.0
            self.pan_offset = QPointF()
            self._pan_anchor = None
        self.frame = frame
        if frame is None:
            self.image = None
            self.update()
            return
        pixels = frame.pixels
        fmt = {"GRAY": QImage.Format.Format_Grayscale8,
               "BGR": QImage.Format.Format_BGR888,
               "BGRA": QImage.Format.Format_ARGB32}[frame.pixel_format]
        self.image = QImage(pixels.data, frame.size[0], frame.size[1], pixels.strides[0], fmt)
        self.update()

    def _field_changed(self, *_):
        self.cursor_point = None
        self.update()

    def _transform(self):
        if self.frame is None:
            return 1.0, (0.0, 0.0)
        width, height = self.frame.size
        scale = min(self.width() / width, self.height() / height) * self.zoom
        return scale, ((self.width() - width * scale) / 2 + self.pan_offset.x(),
                       (self.height() - height * scale) / 2 + self.pan_offset.y())

    def fit_view(self):
        self.gestureCancelled.emit()
        self.zoom = 1.0
        self.pan_offset = QPointF()
        self._pan_anchor = None
        self.cursor_point = None
        self.display_revision += 1
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#111820"))
        if self.frame is None:
            painter.setPen(QColor("#a6b4c5"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "选择图像来源，开始实时试验")
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
        if event.button() == Qt.MouseButton.MiddleButton:
            self.gestureCancelled.emit()
            self._pan_anchor = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        elif event.button() == Qt.MouseButton.RightButton:
            self.gestureCancelled.emit()
        elif event.button() == Qt.MouseButton.LeftButton and self._pan_anchor is None:
            point = self._point(event)
            if point:
                self.pointPressed.emit(point)
        self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton:
            self._pan_anchor = None
            self.setCursor(Qt.CursorShape.CrossCursor)
        elif event.button() == Qt.MouseButton.RightButton:
            self.categoryRequested.emit(event.globalPosition().toPoint())
        elif event.button() == Qt.MouseButton.LeftButton and self._pan_anchor is None:
            point = self._point(event)
            if point:
                self.pointReleased.emit(point)

    def mouseMoveEvent(self, event):
        if self._pan_anchor is not None:
            self.pan_offset += event.position() - self._pan_anchor
            self._pan_anchor = event.position()
            self.display_revision += 1
            self.update()
            return
        self.cursor_point = self._point(event)
        if self.controller.draft:
            self.update()

    def wheelEvent(self, event):
        self.gestureCancelled.emit()
        position = event.position()
        old_scale, old_offset = self._transform()
        anchor = ((position.x() - old_offset[0]) / old_scale,
                  (position.y() - old_offset[1]) / old_scale)
        self.zoom = max(0.2, min(8.0, self.zoom * (1.15 if event.angleDelta().y() > 0 else 1 / 1.15)))
        new_scale, new_offset = self._transform()
        self.pan_offset += QPointF(position.x() - (anchor[0] * new_scale + new_offset[0]),
                                   position.y() - (anchor[1] * new_scale + new_offset[1]))
        self.display_revision += 1
        self.update()

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton:
            self.fit_view()
            event.accept()
        else:
            # A fast second left click remains the second measurement point.
            self.mousePressEvent(event)

    def resizeEvent(self, event):
        self.gestureCancelled.emit()
        self.display_revision += 1
        super().resizeEvent(event)


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


class LiveExperimentWorkspace(QMainWindow):
    frameArrived = Signal()
    sourceFailed = Signal(object)

    def __init__(self, host, session):
        super().__init__(host)
        self.setWindowFlag(Qt.WindowType.Window, True)
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
        self.controller = LiveExperimentController(session, parent=self, verify_before_present=True)
        self.canvas = LiveMeasurementCanvas(self.controller)
        self.controller.frameReady.connect(self.canvas.set_frame)
        self._last_summary_revision = -1
        self._record_model_revision = -1
        self._static_generation = 0
        self._reminders_seen = set()
        self._last_dirty_state = (session.revision, session.active_operator_id, session.active_direction)
        self._build()
        self.live_state_label = QLabel()
        self.statusBar().addPermanentWidget(self.live_state_label)
        self.statistics_progress = QLabel()
        self.statistics_cancel = QPushButton("取消重算")
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
        self._guard(self._load_static)
        self._check_reminders(announce=False)
        QApplication.instance().installEventFilter(self)

    def _button(self, label, callback, layout):
        button = QPushButton(label)
        button.clicked.connect(lambda: self._guard(callback))
        layout.addWidget(button)
        return button

    def _build(self):
        toolbar = QToolBar("实时试验")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        toolbar.addAction("切换／新建试验", lambda: self._guard(self._choose_experiment))
        for label, callback in (("当前图片", self._load_static), ("本机相机", self._start_camera),
                                ("CU 实时画面", self._start_external), ("标定", self._calibrate),
                                ("新视场", self.controller.end_field)):
            action = toolbar.addAction(label)
            action.triggered.connect(lambda checked=False, fn=callback: self._guard(fn))
        toolbar.addSeparator()
        self.measure_mode = QComboBox()
        self.measure_mode.addItem("贴边测径", "assisted")
        self.measure_mode.addItem("自由测量", "free")
        toolbar.addWidget(self.measure_mode)
        self.gesture = QComboBox()
        self.gesture.addItems(["两次点击", "拖动测量"])
        self.gesture.setCurrentIndex(1 if getattr(self.session, "workspace_options", {}).get("gesture") == "drag" else 0)
        self.gesture.currentIndexChanged.connect(self._gesture_changed)
        toolbar.addWidget(self.gesture)
        self.algorithm = QComboBox()
        self.algorithm.addItem("传统全图", "traditional")
        self.algorithm.addItem("多尺度全图", "multiscale")
        toolbar.addWidget(self.algorithm)
        toolbar.addAction("分析当前视场", lambda: self._guard(lambda: self.controller.analyze(self.algorithm.currentData())))
        toolbar.addAction("取消分析", self.controller.discard_candidates)
        toolbar.addAction("适应画面", self.canvas.fit_view)
        root = QSplitter(Qt.Orientation.Horizontal)
        left = QSplitter(Qt.Orientation.Vertical)
        left.addWidget(self.canvas)
        self.tabs = QTabWidget()
        self.current_model = ExperimentRecordsModel(self.session)
        self.history_model = ExperimentRecordsModel(self.session)
        self.candidate_model = CandidatesModel()
        self.current_table = self._table(self.current_model)
        self.history_table = self._table(self.history_model)
        self.candidate_table = self._table(self.candidate_model)
        self.candidate_table.setItemDelegateForColumn(3, CandidateMaterialDelegate(self.candidate_table))
        self.candidate_table.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked
            | QAbstractItemView.EditTrigger.SelectedClicked | QAbstractItemView.EditTrigger.EditKeyPressed)
        for column, width in enumerate((55, 55, 95, 150, 235, 145)):
            self.candidate_table.setColumnWidth(column, width)
        self.tabs.addTab(self.current_table, "当前视场")
        self.tabs.addTab(self.history_table, "试验历史（全部操作者）")
        candidate_panel = QWidget()
        candidate_layout = QVBoxLayout(candidate_panel)
        candidate_layout.addWidget(self.candidate_table)
        candidate_note = QLabel("双击类别逐条赋类，或批量设置；核对直径和范围后可勾选保留越限项。未勾选接受的候选将在确认时排除。")
        candidate_note.setWordWrap(True)
        candidate_layout.addWidget(candidate_note)
        candidate_actions = QHBoxLayout()
        self._button("勾选项批量设为当前类别", self._assign_candidate_material, candidate_actions)
        self._button("确认已赋类的勾选项", self._accept_candidates, candidate_actions)
        candidate_layout.addLayout(candidate_actions)
        self.tabs.addTab(candidate_panel, "全图候选")
        left.addWidget(self.tabs)
        left.setSizes([580, 230])
        root.addWidget(left)
        side = QWidget()
        side.setMinimumWidth(370)
        side.setMaximumWidth(445)
        layout = QVBoxLayout(side)
        self.mode = QComboBox()
        self.mode.addItems(["含量试验", "直径试验"])
        self.mode.setCurrentIndex(1 if getattr(self.session, "workspace_options", {}).get("mode") == "diameter" else 0)
        layout.addWidget(self.mode)
        self.sample = QLineEdit()
        self.sample.setPlaceholderText("样品名称")
        self.sample.setText(getattr(self.session, "sample_name", ""))
        self.sample.editingFinished.connect(self._sample_changed)
        layout.addWidget(self.sample)
        row = QHBoxLayout()
        self.operator = QComboBox()
        self.operator.currentIndexChanged.connect(self._operator_changed)
        row.addWidget(self.operator, 1)
        self._button("添加", self._add_operator, row)
        self._button("改名", self._rename_operator, row)
        layout.addLayout(row)
        self.direction = QComboBox()
        self.direction.addItem("经向", "warp")
        self.direction.addItem("纬向", "weft")
        self.direction.currentIndexChanged.connect(self._direction_changed)
        layout.addWidget(self.direction)
        self.material = QComboBox()
        layout.addWidget(self.material)
        self.classification_mode = QComboBox()
        self.classification_mode.addItems(["先选：先选类别，再测量", "后选：选类修改最新测量"])
        self.classification_mode.setToolTip("后选跟随本场最新测量，计算和待确认期间也可选类；取消或失败后只用于下一次，不改上一条。")
        self.classification_mode.setCurrentIndex(int(self.session.workspace_options["classification"] == "after"))
        self.classification_mode.currentIndexChanged.connect(self._classification_changed)
        layout.addWidget(self.classification_mode)
        self.keyboard_mode = QComboBox()
        self.keyboard_mode.addItems(["数字键选类", "数字键额外计数"])
        self.keyboard_mode.setToolTip("1–9 与小键盘使用此模式；补计数模式下减号只减少当前类别的补计数。")
        layout.addWidget(self.keyboard_mode)
        self.after_select = QCheckBox("后选", side)
        self.after_select.setChecked(self.classification_mode.currentIndex() == 1)
        self.after_select.hide()
        self.after_select.toggled.connect(lambda enabled: self.classification_mode.setCurrentIndex(int(enabled)))
        self.material.activated.connect(self._after_select)
        self.material.currentIndexChanged.connect(self._sync_material_buttons)
        self.material_panel = QWidget()
        self.material_panel_layout = QGridLayout(self.material_panel)
        self.material_panel_layout.setContentsMargins(0, 0, 0, 0)
        self.material_panel_layout.setHorizontalSpacing(5)
        self.material_panel_layout.setVerticalSpacing(3)
        self.material_rows = {}
        layout.addWidget(self.material_panel)
        self.pending_label = QLabel()
        self.pending_label.setWordWrap(True)
        layout.addWidget(self.pending_label)
        pending_row = QHBoxLayout()
        self.pending_buttons = [
            self._button("保留", lambda: self._resolve_displayed_pending(), pending_row),
            self._button("改为当前类", lambda: self._resolve_displayed_pending(material_id=self.material.currentData()), pending_row),
            self._button("放弃", lambda: self._resolve_displayed_pending(discard=True), pending_row)]
        layout.addLayout(pending_row)
        edit_row = QHBoxLayout()
        self._button("撤销", lambda: self._history_action("undo"), edit_row)
        self._button("重做", lambda: self._history_action("redo"), edit_row)
        self._button("删除", self._delete_record, edit_row)
        layout.addLayout(edit_row)
        edit_row2 = QHBoxLayout()
        self._button("改径／改类", self._edit_record, edit_row2)
        self._button("查看原图", self._review_record, edit_row2)
        layout.addLayout(edit_row2)
        self._button("保存项目", self._save_project, layout)
        self.advanced_toggle = QPushButton("更多设置与汇总 ▸")
        self.advanced_toggle.setCheckable(True)
        layout.addWidget(self.advanced_toggle)
        self.advanced_panel = QWidget()
        advanced = QVBoxLayout(self.advanced_panel)
        advanced.setContentsMargins(0, 0, 0, 0)
        self.advanced_toggle.toggled.connect(self.advanced_panel.setVisible)
        self.advanced_toggle.toggled.connect(lambda checked: self.advanced_toggle.setText("更多设置与汇总 ▾" if checked else "更多设置与汇总 ▸"))
        self.advanced_panel.hide()
        layout.addWidget(self.advanced_panel)
        self.sample_number = QLineEdit(self.session.sample_number)
        self.sample_number.setPlaceholderText("样品编号")
        self.sample_number.editingFinished.connect(lambda: self._configure(sample_number=self.sample_number.text()))
        advanced.addWidget(self.sample_number)
        self._button("设置本方向类别", self._materials, advanced)
        self.weights_button = self._button("经纬重量", self._weights, advanced)
        count_row = QHBoxLayout()
        self.count_add_button = self._button("额外计数 +", lambda: self._count(1), count_row)
        self.count_subtract_button = self._button("额外计数 −", lambda: self._count(-1), count_row)
        advanced.addLayout(count_row)
        self.keep_images = QCheckBox("保存测量原图")
        self.keep_images.setChecked(self.session.save_images)
        self.keep_images.toggled.connect(lambda value: self._configure(save_images=value))
        advanced.addWidget(self.keep_images)
        self.statistics_button = self._button("含量计算设置", self._statistics_settings, advanced)
        self.summary_scope = QComboBox()
        self.summary_scope.addItems(["当前操作者／方向", "当前操作者／经纬混合", "全部操作者／纤维汇总", "全部操作者／织物汇总"])
        self.summary_scope.currentIndexChanged.connect(lambda _: setattr(self, "_last_summary_revision", -1))
        advanced.addWidget(self.summary_scope)
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        self.summary.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        advanced.addWidget(self.summary)
        self._button("放弃所选失败原图", self._discard_failed_evidence, advanced)
        export_row = QHBoxLayout()
        self._button("导出 CSV", lambda: self._export(False), export_row)
        self._button("导出 Excel", lambda: self._export(True), export_row)
        advanced.addLayout(export_row)
        layout.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(side)
        scroll.setMinimumWidth(390)
        root.addWidget(scroll)
        root.setSizes([850, 390])
        self.setCentralWidget(root)
        self.mode.currentIndexChanged.connect(self._mode_changed)
        self._mode_changed(self.mode.currentIndex(), persist=False)
        for sequence, callback in (("Escape", self.controller.cancel_draft),
                                   ("Ctrl+Z", lambda: self._history_action("undo")),
                                   ("Ctrl+Y", lambda: self._history_action("redo"))):
            action = QAction(self)
            action.setShortcut(QKeySequence(sequence))
            action.setShortcutContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            action.triggered.connect(lambda checked=False, fn=callback: self._guard(fn))
            self.addAction(action)

    def _table(self, model):
        table = QTableView()
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
            self.canvas.set_frame(None)

    def _stop_source(self):
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
        self.controller.unavailable("正在停止旧图像来源，测量已暂停。")
        try:
            stopped = self.source.stop() is True
            detail = ""
        except Exception as exc:
            stopped, detail = False, f"（{exc}）"
        if not stopped:
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
            self.statusBar().showMessage("请在主窗口打开图片，或选择相机／CU 实时画面。")
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

    def _start_camera(self):
        from fdm.live_experiment.sources import CameraFrameSource
        if not self._stop_source():
            return False
        self.controller.unavailable("正在连接本机相机…")
        self.source = CameraFrameSource(self.host._capture_manager, *self._source_callbacks())
        self.source.start()

    def _start_external(self):
        from fdm.live_experiment.sources import ExternalCuFrameSource
        if not self._stop_source():
            return False
        self.controller.unavailable("正在连接 CU 实时画面…")
        self.source = ExternalCuFrameSource(*self._source_callbacks())
        self.source.start()

    def _receive_frame(self, frame):
        if self._closed:
            return
        if id(self.host.project) != self.project_identity:
            self.close()
            return
        if self.controller.receive_frame(frame) and not frame.live:
            self.canvas.set_frame(frame)

    def _calibrate(self):
        value, ok = QInputDialog.getDouble(self, "当前来源标定", "每像素微米数（μm/px）",
            self.controller.calibration.x_um if self.controller.calibration else 1.0, 0.000001, 1000000, 6)
        if ok:
            revision = (self.controller.calibration.revision + 1) if self.controller.calibration else 1
            self.controller.set_calibration(CalibrationSnapshot(value, value, revision, "实时试验来源标定"))
            self.canvas.update()

    def _press(self, point):
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
        for column, text in enumerate(("纤维类别 1–9", "", "补计", "", "均径 μm", "含量%", "实测")):
            label = QLabel(text)
            if column == 4:
                label.setToolTip("原始实测直径的算术均值；不使用分组中心，不受补计数影响。")
            grid.addWidget(label, 0, column)
        operator = self.session.operators.get(self.session.active_operator_id)
        for index, material in enumerate(operator.materials[self.session.active_direction] if operator else ()):
            select = QPushButton(f"{index + 1} · {material.name}")
            select.setCheckable(True)
            select.setMinimumWidth(75)
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
                button.setFixedWidth(25)
                button.setToolTip("只增减本类别的补计数；不改变选中类别或实测直径。")
                button.clicked.connect(lambda checked=False, key=material.id, delta=delta:
                    self._guard(lambda: self._count_material(key, delta)))
            row = {"select": select, "color": marker, "add": add, "subtract": subtract,
                "extra_count": QLabel("0"), "mean": QLabel("—"), "content": QLabel("待完善"),
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
        for material_id, row in getattr(self, "material_rows", {}).items():
            row["select"].setChecked(material_id == selected)
        self.canvas.update()

    def _select_material(self, material_id):
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
            row["content"].setText("待完善" if category["content"] is None else f"{category['content']:.2f}")
            row["subtract"].setEnabled(category["extra_count"] > 0)

    def _add_operator(self):
        name, ok = QInputDialog.getText(self, "添加操作者", "姓名")
        if ok and name.strip():
            self.controller.end_field("已切换操作者")
            self.session.active_operator_id = self.session.add_operator(name.strip())
            self._refresh_operators()
            self._changed()

    def _operator_changed(self, index):
        if index < 0:
            return
        self.controller.end_field("已切换操作者")
        self.session.active_operator_id = self.operator.currentData()
        self._refresh_materials()
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
        self._configure(sample_name=self.sample.text())

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
                       self.weights_button, self.count_add_button, self.count_subtract_button, self.statistics_button):
            widget.setVisible(content)
        if persist:
            self._configure(workspace_options={**getattr(self.session, "workspace_options", {}),
                                              "mode": "content" if content else "diameter"})

    def _record_accepted(self, record):
        self.current_model.append(record.id)
        self.history_model.append(record.id)
        self.current_table.scrollToBottom()
        self._record_model_revision = self.session.revision

    def _records_accepted(self, records):
        ids = [record.id for record in records]
        self.current_model.append_many(ids)
        self.history_model.append_many(ids)
        self.current_table.scrollToBottom()
        self._record_model_revision = self.session.revision

    def _candidates_changed(self, candidates):
        scale = self.controller.calibration.x_um if self.controller.calibration else 1
        operator = self.session.operators.get(self.session.active_operator_id)
        materials = operator.materials[self.session.active_direction] if operator else ()
        self.candidate_model.set_candidates(candidates, scale, materials)
        if candidates:
            self.tabs.setCurrentIndex(2)
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
            self.host.project.mark_extension_changed()
            self._last_dirty_state = state
        self.canvas.update()

    def _refresh_summary(self):
        self._poll_statistics_rebuild()
        if id(self.host.project) != self.project_identity:
            self.close()
            return
        frame, calibration = self.controller.latest_frame, self.controller.calibration
        source = ("CU 预览" if frame and frame.metadata.get("rendered_preview") else
                  "相机" if frame and frame.live else "图片" if frame else "无来源")
        state = {"ready": "视场就绪", "stabilizing": "视场核验中", "unavailable": "来源不可用"}.get(
            self.controller.state, self.controller.state)
        ruler = ("未标定" if calibration is None else
                 f"{calibration.x_um:.6g} μm/px" if calibration.isotropic else
                 f"X {calibration.x_um:.6g} / Y {calibration.y_um:.6g} μm/px")
        self.live_state_label.setText(f"{source} · {state} · {ruler}")
        pending = self.controller.pending_classification
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
        for button in self.pending_buttons:
            button.setEnabled(bool(self.controller.pending_classification) and self.controller._confirmation is None)
        if self.session.revision == self._last_summary_revision:
            return
        from fdm.live_experiment.statistics import summarize
        summary = summarize(self.session)
        self._refresh_material_row_stats(summary)
        operator = summary.get("operators", {}).get(self.session.active_operator_id, {})
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
        sessions = self.host.project.live_experiments
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
        if target is None:
            from fdm.live_experiment import ExperimentSession, default_materials
            target = ExperimentSession(sample_name=name.strip())
            target.add_operator("操作者 A")
            target.set_materials(default_materials(), direction="warp")
            target.set_materials(default_materials(), direction="weft")
            target.establish_history_baseline()
            sessions.append(target)
            self.host.project.mark_extension_changed()
        window = LiveExperimentWorkspace(self.host, target)
        self.host._live_experiment_workspace = window
        window.show()

    def _save_project(self):
        if self._statistics_job is not None:
            raise ValueError("组距设置尚未应用，请等待或取消重算后保存。")
        if self.controller.evidence.pending or self.controller._jobs:
            raise ValueError("测量或原图尚在写入，请稍后保存；实时画面继续更新。")
        self.host.save_project_action.trigger()

    def _export(self, excel):
        from fdm.live_experiment.exports import export_csv, session_export_xlsx
        path, _ = QFileDialog.getSaveFileName(self, "导出试验", "试验.xlsx" if excel else "试验.csv", "Excel (*.xlsx)" if excel else "CSV (*.csv)")
        if path:
            (session_export_xlsx if excel else export_csv)(self.session, path)
            self.statusBar().showMessage("已导出试验数据", 5000)

    def _handle_digit_key(self, event):
        focus = QApplication.focusWidget()
        if (self._closed or QApplication.activeWindow() is not self or QApplication.activePopupWidget() is not None
                or isinstance(focus, (QLineEdit, QAbstractSpinBox, QPlainTextEdit, QTextEdit))
                or (isinstance(focus, QComboBox) and self.candidate_table.isAncestorOf(focus))
                or event.modifiers() & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.AltModifier
                                       | Qt.KeyboardModifier.MetaModifier | Qt.KeyboardModifier.ShiftModifier)):
            return False
        if Qt.Key.Key_1 <= event.key() <= Qt.Key.Key_9:
            if event.isAutoRepeat():
                return True
            index = event.key() - Qt.Key.Key_1
            if index < self.material.count():
                self.material.setCurrentIndex(index)
                self._guard(lambda: self._count(1) if self.keyboard_mode.currentIndex() else self._after_select())
                return True
        if event.key() == Qt.Key.Key_Minus and self.keyboard_mode.currentIndex():
            if not event.isAutoRepeat():
                self._guard(lambda: self._count(-1))
            return True
        return False

    def eventFilter(self, watched, event):
        if event.type() == QEvent.Type.KeyPress and self._handle_digit_key(event):
            event.accept()
            return True
        return super().eventFilter(watched, event)

    def keyPressEvent(self, event):
        if self._handle_digit_key(event):
            event.accept()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event):
        if self._statistics_job is not None:
            self._cancel_statistics_rebuild()
            if not self._statistics_job[1].done():
                self.statusBar().showMessage("正在取消组距重算，窗口与来源仍保留；请稍后再次关闭。")
                event.ignore()
                return
            self._poll_statistics_rebuild()
        if self.controller.evidence.pending:
            self.statusBar().showMessage("正在保存测量原图，请完成后关闭。")
            event.ignore()
            return
        if not self._stop_source():
            event.ignore()
            return
        self._closed = True
        QApplication.instance().removeEventFilter(self)
        self._refresh_timer.stop()
        if self._statistics_pool is not None:
            self._statistics_pool.shutdown(wait=False, cancel_futures=True)
            self._statistics_pool = None
        self.canvas.set_frame(None)
        if getattr(self.host, "_live_experiment_workspace", None) is self:
            self.host._live_experiment_workspace = None
        self.controller.close()
        super().closeEvent(event)


def open_live_experiment(host):
    existing = getattr(host, "_live_experiment_workspace", None)
    if existing is not None:
        existing.show()
        existing.raise_()
        existing.activateWindow()
        return existing
    from fdm.live_experiment import ExperimentSession, default_materials
    if host.project.is_read_only_compatible:
        QMessageBox.information(host, "实时试验", "当前项目以只读兼容模式打开，请先另存可编辑项目。")
        return None
    if not host.project.live_experiments:
        session = ExperimentSession()
        session.add_operator("操作者 A")
        session.set_materials(default_materials(), direction="warp")
        session.set_materials(default_materials(), direction="weft")
        session.establish_history_baseline()
        host.project.live_experiments.append(session)
        host.project.mark_extension_changed()
    else:
        session = host.project.live_experiments[-1]
    window = LiveExperimentWorkspace(host, session)
    host._live_experiment_workspace = window
    window.show()
    return window
