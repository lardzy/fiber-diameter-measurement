"""Non-destructive before/after silhouette workspace, independent of documents."""
from __future__ import annotations

from dataclasses import replace
import math
import json

from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt, QTimer
from PySide6.QtGui import QAction, QCloseEvent, QImage, QKeySequence, QPainter
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QButtonGroup, QCheckBox, QDialog,
    QFileDialog, QFrame, QHBoxLayout, QHeaderView, QInputDialog,
    QLabel, QMenu, QMessageBox, QPushButton, QScrollArea, QSplitter,
    QTableView, QTabWidget, QToolButton, QVBoxLayout, QWidget,
)

from fdm.services.contour_comparison import (
    ContourAxis, automatic_mask, compare_contours, edit_mask, load_frame,
    load_comparison, save_comparison, export_comparison_excel,
)
from fdm.ui.contour_comparison_canvas import (
    BEFORE_COLOR, AFTER_COLOR, ContourImageCanvas, ContourOverlayCanvas,
    prepare_presentation,
)
from fdm.ui.contour_comparison_tasks import ContourTaskController
from fdm.ui.widgets import NoWheelComboBox, NoWheelDoubleSpinBox, NoWheelSpinBox


class _ProfileTableModel(QAbstractTableModel):
    """Format visible cells only; installing 5000 samples stays constant-time."""
    headers = ("高度", "左变化", "右变化", "跨度变化", "前左距", "后左距", "前右距", "后右距", "前跨度", "后跨度", "状态")
    fields = ("height", "left_change", "right_change", "span_change", "left_before", "left_after", "right_before", "right_after", "span_before", "span_after", "status")

    def __init__(self, parent=None):
        super().__init__(parent)
        self.result = None

    def set_result(self, result):
        self.beginResetModel()
        self.result = result
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() or self.result is None else len(self.result.sections)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.headers)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return self.headers[section] + (f" ({self.result.unit})" if self.result is not None and section == 0 else "")
        return super().headerData(section, orientation, role)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or self.result is None or index.row() >= len(self.result.sections):
            return None
        if role == Qt.ItemDataRole.TextAlignmentRole and index.column() < 10:
            return Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        if role == Qt.ItemDataRole.DisplayRole:
            value = getattr(self.result.sections[index.row()], self.fields[index.column()])
            if index.column() == 10:
                return value
            if value is None:
                return "—"
            return f"{value:+.3f}" if index.column() in (1, 2, 3) else f"{value:.3f}"
        return None


class ContourComparisonDialog(QDialog):
    """Callbacks enumerate documents and freeze a selected source on the UI thread.

    ``source_loader(id)`` returns a callable taking a CancellationToken. It must
    own immutable source pixels, so closing the source tab cannot affect a task.
    """

    def __init__(self, parent=None, *, available_images=None, source_loader=None, wand_model_variant="edge_sam_3x"):
        super().__init__(parent)
        self.setWindowTitle("前后轮廓对比")
        self.setModal(False)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.resize(1280, 880)
        self.setMinimumSize(860, 640)
        self.available_images = available_images or (lambda: [])
        self.source_loader = source_loader
        self.frames = [None, None]
        self.presentations = [None, None]
        self.result = None
        self.active = 0
        self.dirty = False
        self.session_path = ""
        self._history = []
        self._redo = []
        self._callback = None
        self._task_name = ""
        self._closing = False
        self._save_then_close = False
        self._deferred_undo = False
        self._last_step = 10.0
        self._revision = 0
        self._result_revision = -1
        self._selected_height = None
        self._picker_depth = 0
        self._picker_restore = QTimer(self)
        self._picker_restore.setSingleShot(True)
        self._picker_restore.timeout.connect(self._restore_picker_owner)
        from fdm.services.contour_comparison_wand import ContourWandService
        self._wand_service = ContourWandService(wand_model_variant)
        self._wand_sessions = [None, None]
        self._tasks = ContourTaskController(self)
        self._tasks.finished.connect(self._task_finished)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(160)
        self._timer.timeout.connect(self._compare)
        self._build_ui()
        self._update_controls()

    def _button(self, text, callback, layout):
        button = QPushButton(text, self)
        button.clicked.connect(callback)
        layout.addWidget(button)
        return button

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 12, 14, 12)
        root.setSpacing(9)
        top = QHBoxLayout()
        title = QLabel("前后轮廓对比")
        font = title.font()
        font.setPointSize(font.pointSize() + 4)
        font.setBold(True)
        title.setFont(font)
        top.addWidget(title)
        top.addStretch()
        self.open_session_button = self._button("打开对比…", self._open_session, top)
        self.save_button = self._button("保存对比…", self._save, top)
        self.export_button = self._button("导出数据…", self._export, top)
        self.overlay_export_button = self._button("导出叠加图…", self._export_overlay, top)
        root.addLayout(top)
        self.explanation = QLabel("同一中线、同一参考高度：正值向外伸展，负值向内收缩。外形差异包含摆放影响，不等同于材料应变。")
        self.explanation.setWordWrap(True)
        root.addWidget(self.explanation)

        controls = QHBoxLayout()
        self.same_capture = QCheckBox("固定机位：共用中线和标定")
        self.same_capture.setChecked(True)
        self.same_capture.setToolTip("适用于前后照片分辨率、相机位置、焦距、裁切和拍摄平面均相同。第一点是固定参考高度，不能分别对齐衣物新的上端。")
        self.same_capture.toggled.connect(self._shared_changed)
        controls.addWidget(self.same_capture)
        controls.addStretch()
        controls.addWidget(QLabel("沿中线每隔"))
        self.step = NoWheelDoubleSpinBox()
        self.step.setDecimals(3)
        self.step.setRange(.01, 10000)
        self.step.setValue(10)
        self.step.setSuffix(" px")
        self.step.setToolTip("截线之间的距离。不是测量精度，也不是材料应变的空间分辨率。")
        self.step.valueChanged.connect(self._step_changed)
        controls.addWidget(self.step)
        controls.addWidget(QLabel("取一条截线"))

        self.tabs = QTabWidget()
        edit_page = QWidget()
        edit_layout = QVBoxLayout(edit_page)
        edit_layout.setContentsMargins(4, 8, 4, 4)
        edit_layout.setSpacing(7)
        edit_layout.addLayout(controls)
        tool_scroll = QScrollArea()
        tool_scroll.setWidgetResizable(True)
        tool_scroll.setFrameShape(QFrame.Shape.NoFrame)
        tool_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        tool_scroll.setFixedHeight(52)
        tool_widget = QWidget()
        toolbar = QHBoxLayout(tool_widget)
        toolbar.setContentsMargins(0, 0, 0, 4)
        toolbar.setSpacing(5)
        self.tool_group = QButtonGroup(self)
        self.tool_group.setExclusive(True)
        self.tools = {}
        for mode, text, tip in (
            ("browse", "查看", "滚轮缩放，按住右键拖动；点击图片选择修正对象"),
            ("wand", "魔棒", "点击目标，可连续补点；Alt＋点击排除背景。自动应用，可撤销；Esc 结束点选"),
            ("brush_add", "补画", "按住左键补入衣物区域"),
            ("brush_remove", "剔除", "按住左键剔除背景或腿缝"),
            ("polygon_add", "圈入", "逐点圈入区域，Enter 或双击完成"),
            ("polygon_remove", "圈除", "逐点剔除区域，Enter 或双击完成"),
            ("axis", "设中线", "先点共同参考高度的原点，再沿中线向下点第二点；第二点只决定方向"),
            ("calibrate", "标定", "在与衣物同一平面的标尺上点两端，再输入实际距离"),
            ("roi", "框选识别", "拖框包含整个目标及少量背景，重新自动识别；可撤销"),
        ):
            button = QToolButton()
            button.setText(text)
            button.setToolTip(tip)
            button.setCheckable(True)
            button.setMinimumHeight(32)
            self.tool_group.addButton(button)
            button.clicked.connect(lambda _checked=False, selected=mode: self._set_mode(selected))
            toolbar.addWidget(button)
            self.tools[mode] = button
        self.tools["browse"].setChecked(True)
        toolbar.addWidget(QLabel("半径"))
        self.brush = NoWheelSpinBox()
        self.brush.setRange(1, 500)
        self.brush.setValue(8)
        self.brush.setSuffix(" px")
        self.brush.setFixedWidth(91)
        self.brush.valueChanged.connect(self._brush_changed)
        toolbar.addWidget(self.brush)
        self.undo_button = self._button("撤销", self.undo, toolbar)
        self.redo_button = self._button("重做", self.redo, toolbar)
        toolbar.addStretch()
        tool_scroll.setWidget(tool_widget)
        edit_layout.addWidget(tool_scroll)
        self.tool_hint = QLabel("选择照片后修正。滚轮缩放 · 右键拖动 · 多边形按 Enter 完成 · Esc 取消当前修正")
        self.tool_hint.setWordWrap(True)
        hint_row = QHBoxLayout()
        hint_row.addWidget(self.tool_hint, 1)
        self.coverage_visible = QCheckBox("显示覆盖")
        self.coverage_visible.setChecked(True)
        self.coverage_visible.setToolTip("取消勾选可对照原照片纹理；不会改变参与计算的轮廓")
        self.coverage_visible.toggled.connect(self._coverage_changed)
        hint_row.addWidget(self.coverage_visible)
        edit_layout.addLayout(hint_row)
        self.wand_options = QWidget()
        wand_row = QHBoxLayout(self.wand_options)
        wand_row.setContentsMargins(0, 0, 0, 0)
        wand_row.addWidget(QLabel("魔棒结果"))
        self.wand_operation = NoWheelComboBox()
        for label, value in (("替换轮廓", "replace"), ("补入轮廓", "add"), ("剔除区域", "remove")):
            self.wand_operation.addItem(label, value)
        self.wand_operation.setToolTip("替换：重新选整个物体；补入／剔除：只修改所选区域。连续补点始终基于本次点选前的轮廓。")
        self.wand_operation.currentIndexChanged.connect(lambda: self._reset_wand(all_sources=True))
        wand_row.addWidget(self.wand_operation)
        self.wand_negative = QCheckBox("点选排除背景")
        self.wand_negative.setToolTip("也可按住 Alt 点击背景；右键仍用于拖动画布。剔除区域模式下，此处指不需要剔除的区域。")
        wand_row.addWidget(self.wand_negative)
        self.wand_reset_button = self._button("重新选点", self._reset_wand, wand_row)
        wand_row.addStretch()
        self.wand_options.hide()
        edit_layout.addWidget(self.wand_options)
        self.image_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.canvases, self.name_labels, self.scale_labels, self.import_buttons, self.auto_buttons = [], [], [], [], []
        for index, (text, color) in enumerate((("处理前", BEFORE_COLOR), ("处理后", AFTER_COLOR))):
            panel = QWidget()
            panel_layout = QVBoxLayout(panel)
            panel_layout.setContentsMargins(2, 0, 2, 0)
            header = QHBoxLayout()
            label = QLabel(text)
            label.setStyleSheet(f"color: {color.name()}; font-weight: bold;")
            header.addWidget(label)
            header.addStretch()
            import_button = QToolButton()
            import_button.setText("导入照片")
            import_button.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)
            import_button.clicked.connect(lambda _checked=False, i=index: self._import_file(i))
            menu = QMenu(import_button)
            menu.aboutToShow.connect(lambda m=menu, i=index: self._populate_sources(m, i))
            import_button.setMenu(menu)
            header.addWidget(import_button)
            self.import_buttons.append(import_button)
            auto = QToolButton()
            auto.setText("自动识别")
            auto.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)
            auto.clicked.connect(lambda _checked=False, i=index: self._automatic(i))
            auto_menu = QMenu(auto)
            for method, name in (("auto", "按背景颜色"), ("dark", "深色物体"), ("light", "浅色物体")):
                auto_menu.addAction(name, lambda i=index, m=method: self._automatic(i, m))
            auto.setMenu(auto_menu)
            header.addWidget(auto)
            self.auto_buttons.append(auto)
            self._button("适合窗口", lambda _checked=False, i=index: self.canvases[i].fit(), header)
            panel_layout.addLayout(header)
            name_label = QLabel("尚未导入")
            name_label.setMinimumWidth(0)
            name_label.setWordWrap(True)
            name_label.setMaximumHeight(38)
            panel_layout.addWidget(name_label)
            self.name_labels.append(name_label)
            canvas = ContourImageCanvas(color)
            canvas.activated.connect(lambda i=index: self._activate(i))
            canvas.gesture.connect(lambda mode, points, i=index: self._gesture(i, mode, points))
            canvas.wand_finished.connect(lambda i=index: self._end_wand(i))
            panel_layout.addWidget(canvas, 1)
            self.canvases.append(canvas)
            scale_label = QLabel("未标定 · 结果只能使用像素")
            scale_label.setWordWrap(True)
            panel_layout.addWidget(scale_label)
            self.scale_labels.append(scale_label)
            self.image_splitter.addWidget(panel)
        self.image_splitter.setChildrenCollapsible(False)
        edit_layout.addWidget(self.image_splitter, 1)
        self.tabs.addTab(edit_page, "① 照片与轮廓")

        result_page = QWidget()
        result_layout = QVBoxLayout(result_page)
        result_layout.setContentsMargins(4, 8, 4, 4)
        self.summary_label = QLabel("导入前后照片后自动计算。")
        self.summary_label.setWordWrap(True)
        result_layout.addWidget(self.summary_label)
        result_tools = QHBoxLayout()
        result_tools.addWidget(QLabel("底图"))
        self.result_background = NoWheelComboBox()
        for label, value in (("处理前照片", "before"), ("处理后照片", "after"), ("仅看轮廓", "none")):
            self.result_background.addItem(label, value)
        result_tools.addWidget(self.result_background)
        self.result_fit_button = self._button("适合窗口", lambda: self.overlay.fit(), result_tools)
        result_tools.addStretch()
        result_tools.addWidget(QLabel("参考高度"))
        self.section_height = NoWheelDoubleSpinBox()
        self.section_height.setDecimals(3)
        self.section_height.setMinimumWidth(112)
        self.section_height.valueChanged.connect(self._select_height)
        result_tools.addWidget(self.section_height)
        self.previous_section = self._button("上一处", lambda: self._step_section(-1), result_tools)
        self.next_section = self._button("下一处", lambda: self._step_section(1), result_tools)
        self.more_columns = QCheckBox("详细数据")
        self.more_columns.toggled.connect(self._show_more_columns)
        result_tools.addWidget(self.more_columns)
        result_layout.addLayout(result_tools)
        split = QSplitter(Qt.Orientation.Horizontal)
        self.overlay = ContourOverlayCanvas()
        self.overlay.setMinimumWidth(320)
        self.overlay.height_selected.connect(self._select_height)
        self.result_background.currentIndexChanged.connect(lambda: self.overlay.set_background(self.result_background.currentData()))
        split.addWidget(self.overlay)
        self.table = QTableView()
        self.table_model = _ProfileTableModel(self)
        self.table.setModel(self.table_model)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.table.horizontalHeader().setDefaultSectionSize(76)
        self.table.setColumnWidth(10, 180)
        self.table.setMinimumWidth(280)
        self.table.verticalHeader().setVisible(False)
        self.table.selectionModel().selectionChanged.connect(self._select_section)
        split.addWidget(self.table)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        split.setChildrenCollapsible(False)
        split.setSizes([640, 440])
        result_layout.addWidget(split, 1)
        self.section_label = QLabel("点击样品或记录定位。左右变化对应图上外缘标记，外缘跨度包含腿缝；详细数据与分段交集可导出。")
        self.section_label.setWordWrap(True)
        result_layout.addWidget(self.section_label)
        self.tabs.addTab(result_page, "② 变化结果")
        self.tabs.currentChanged.connect(lambda index: self.explanation.setVisible(index == 0))
        root.addWidget(self.tabs, 1)

        self.warning_label = QLabel("中线为自动建议。请使用固定台面上的参考位置；标尺与衣物应处于同一平面，前后铺放一致。")
        self.warning_label.setWordWrap(True)
        root.addWidget(self.warning_label)
        bottom = QHBoxLayout()
        self.status = QLabel("准备就绪 · 原照片和现有测量保持独立")
        self.status.setWordWrap(True)
        bottom.addWidget(self.status, 1)
        self.cancel_button = self._button("取消计算", self._tasks.cancel, bottom)
        self.cancel_button.hide()
        self._button("关闭", self.close, bottom)
        root.addLayout(bottom)
        for standard, callback in ((QKeySequence.StandardKey.Undo, self.undo), (QKeySequence.StandardKey.Redo, self.redo), (QKeySequence.StandardKey.Save, self._save)):
            action = QAction(self)
            action.setShortcuts(QKeySequence.keyBindings(standard))
            action.triggered.connect(callback)
            self.addAction(action)
        self._activate(0)
        self._show_more_columns(False)

    def _activate(self, index):
        self.active = index
        for i, canvas in enumerate(self.canvases):
            canvas.selected = i == index
            canvas.update()

    def _set_mode(self, mode):
        self.tools[mode].setChecked(True)
        for canvas in self.canvases:
            canvas.set_mode(mode)
        self.tool_hint.setText(self.tools[mode].toolTip() + (" · 右键拖动" if mode == "wand" else " · 右键拖动 · Esc 取消"))
        self.wand_options.setVisible(mode == "wand")

    def _reset_wand(self, *args, all_sources=False):
        for i in (range(2) if all_sources else (self.active,)):
            self._end_wand(i)

    def _end_wand(self, index):
        if self._tasks.busy and self._task_name == "魔棒识别轮廓":
            self._tasks.cancel()
        self._wand_sessions[index] = None
        self.wand_negative.setChecked(False)
        self.canvases[index].wand_prompts = ()
        self.canvases[index].update()
        self.status.setText("已结束本次点选，已应用的轮廓保留；需要恢复请撤销。")

    def _wand(self, index, point, negative=False):
        frame = self.frames[index]
        operation = self.wand_operation.currentData()
        session = self._wand_sessions[index]
        if session is None or session["operation"] != operation or session["result_mask"] is not frame.mask or session["pixels"] is not frame.rgba:
            session = dict(base_mask=frame.mask, pixels=frame.rgba, positive=(), negative=())
        positive, excluded = session["positive"], session["negative"]
        if negative or self.wand_negative.isChecked():
            if not positive:
                self.status.setText("请先取消“点选排除背景”，在要识别的区域内点一下，再排除背景。")
                return
            excluded += (point,)
        else:
            positive += (point,)
        base = session["base_mask"]
        def done():
            self._wand_sessions[index] = dict(base_mask=base, pixels=frame.rgba, positive=positive, negative=excluded, result_mask=self.frames[index].mask, operation=operation)
            canvas = self.canvases[index]
            canvas.wand_prompts = tuple((p, True) for p in positive) + tuple((p, False) for p in excluded)
            canvas.update()
            self.status.setText("魔棒已应用；可继续点目标或 Alt＋点背景，或直接切换手动工具修边。")
        self._replace_frame(index, lambda token: self._wand_service.predict(frame, positive, excluded, operation=operation, base_mask=base, token=token), "魔棒识别轮廓", on_installed=done)

    def _brush_changed(self, value):
        for canvas in self.canvases:
            canvas.radius = value
            canvas.update()

    def _coverage_changed(self, enabled):
        for canvas in self.canvases:
            canvas.show_mask = enabled
            canvas.update()

    def _populate_sources(self, menu, index):
        menu.clear()
        menu.addAction("从文件导入…", lambda: self._import_file(index))
        menu.addSeparator()
        sources = self.available_images()
        for key, label in sources:
            menu.addAction(label, lambda k=key: self._import_source(index, k))
        if not sources:
            menu.addAction("没有已打开的普通图片").setEnabled(False)

    def _start(self, title, function, callback, *, cancellable=True):
        if self._tasks.busy:
            return False
        self._timer.stop()
        self._callback, self._task_name = callback, title
        self.status.setText(title + "…")
        self._tasks.start(function)
        self.cancel_button.setVisible(cancellable)
        self._update_controls()
        return True

    def _task_finished(self, value, error, cancelled):
        callback, self._callback = self._callback, None
        task_name = self._task_name
        self.cancel_button.hide()
        if self._closing:
            self.dirty = False  # Discard was explicitly chosen before cancellation.
            self.close()
            return
        if error:
            self._save_then_close = False
            self.status.setText(error)
            if self._task_name != "计算轮廓变化":
                QMessageBox.warning(self, "前后轮廓对比", error)
        elif cancelled:
            self._save_then_close = False
            self.status.setText("已取消；当前已完成的轮廓保留。")
        elif callback:
            try:
                callback(value)
            except Exception as exc:
                self.status.setText(str(exc))
                QMessageBox.warning(self, "前后轮廓对比", str(exc))
        self._update_controls()
        if self._deferred_undo:
            self._deferred_undo = False
            self.undo()
            return
        if task_name != "计算轮廓变化" and self.result is None and all(f is not None for f in self.frames) and not self._tasks.busy:
            self._timer.start()

    def _state(self):
        return (tuple(self.frames), self.same_capture.isChecked(), self.step.value())

    def _remember(self):
        self._history.append(self._state())
        self._redo.clear()
        self._trim_history()

    def _trim_history(self):
        # Shared immutable images/masks count once. Cap additional history data.
        current = {id(a) for f in self.frames if f for a in (f.rgba, f.mask)}
        while self._history or self._redo:
            arrays = {id(a): a.nbytes for state in self._history + self._redo for f in state[0] if f for a in (f.rgba, f.mask) if id(a) not in current}
            if len(self._history) + len(self._redo) <= 20 and sum(arrays.values()) <= 192 * 1024**2:
                break
            (self._history if self._history else self._redo).pop(0)

    def _changed(self):
        self.dirty = any(frame is not None for frame in self.frames)
        self._revision += 1
        self.result = None
        self._result_revision = -1
        self.overlay.clear()
        self.table_model.set_result(None)
        self.summary_label.setText("正在等待两张有效轮廓与一致标定…")
        self.step.setSuffix(" mm" if all(f is not None and f.mm_per_pixel is not None for f in self.frames) else " px")
        self.setWindowTitle("前后轮廓对比" + (" *" if self.dirty else ""))
        self._trim_history()
        self._refresh_labels()
        self._update_controls()
        if all(f is not None for f in self.frames):
            self._timer.start()

    def _refresh_labels(self):
        warnings = []
        for i, frame in enumerate(self.frames):
            if frame is None:
                self.name_labels[i].setText("尚未导入")
                self.scale_labels[i].setText("未标定 · 结果只能使用像素")
                continue
            self.name_labels[i].setText(frame.label)
            self.name_labels[i].setToolTip(frame.source_path or frame.label)
            if frame.mm_per_pixel is None:
                self.scale_labels[i].setText("未标定 · px / px²（不能作为毫米）")
                self.scale_labels[i].setStyleSheet("color: #cc7130; font-weight: bold;")
            else:
                self.scale_labels[i].setText(f"已标定 · {frame.mm_per_pixel:.6g} mm/px · 原点 ({frame.axis.origin[0]:.1f}, {frame.axis.origin[1]:.1f})")
                self.scale_labels[i].setStyleSheet("")
            warnings.extend(frame.warnings)
        message = "请复核中线与轮廓；参考高度不能分别追随处理前后衣物的上端。"
        if warnings:
            message += " " + " ".join(dict.fromkeys(warnings))
        self.warning_label.setText(message)

    def _update_controls(self):
        busy = self._tasks.busy
        for button in self.import_buttons:
            button.setEnabled(not busy)
        for i, button in enumerate(self.auto_buttons):
            button.setEnabled(not busy and self.frames[i] is not None)
        for button in self.tools.values():
            button.setEnabled(not busy)
        for canvas in self.canvases:
            canvas.editing_enabled = not busy
        self.open_session_button.setEnabled(not busy)
        self.save_button.setEnabled(not busy and any(f is not None for f in self.frames))
        ready = not busy and self.result is not None and self._result_revision == self._revision
        self.export_button.setEnabled(ready)
        self.overlay_export_button.setEnabled(ready)
        self.undo_button.setEnabled(not busy and bool(self._history))
        self.redo_button.setEnabled(not busy and bool(self._redo))
        self.same_capture.setEnabled(not busy)
        self.step.setEnabled(not busy)
        self.brush.setEnabled(not busy)
        self.wand_options.setEnabled(not busy)
        has_result = self.result is not None
        self.section_height.setEnabled(has_result)
        self.previous_section.setEnabled(has_result and self.table.currentIndex().row() > 0)
        self.next_section.setEnabled(has_result and self.table.currentIndex().row() < len(self.result.sections)-1)

    def _prepare(self, frames, previous_frames=None):
        # Call on the UI thread to capture immutable inputs, return worker code.
        old_frames = tuple(self.frames) if previous_frames is None else previous_frames
        old = tuple(self.presentations)
        def work(token):
            output = []
            for i, frame in enumerate(frames):
                token.raise_if_cancelled()
                if frame is None:
                    output.append(None)
                elif old_frames[i] is not None and frame.mask is old_frames[i].mask and frame.rgba is old_frames[i].rgba:
                    output.append(old[i])
                else:
                    output.append(prepare_presentation(frame, (BEFORE_COLOR, AFTER_COLOR)[i], old[i]))
            return output
        return work

    def _install(self, frames, presentations, *, remember=True, reset_views=()):
        if remember:
            self._remember()
        self.frames, self.presentations = list(frames), list(presentations)
        for i, session in enumerate(self._wand_sessions):
            if session is not None and (self.frames[i] is None or session["result_mask"] is not self.frames[i].mask or session["pixels"] is not self.frames[i].rgba):
                self._wand_sessions[i] = None
                self.canvases[i].wand_prompts = ()
        for i, canvas in enumerate(self.canvases):
            canvas.set_frame(self.frames[i], self.presentations[i], reset_view=i in reset_views)
        self.status.setText("轮廓已更新，可继续修正或查看变化结果。")
        self._changed()

    def _replace_frame(self, index, function, title, *, on_installed=None):
        frames = list(self.frames)
        previous = tuple(self.presentations)
        old_frames = tuple(self.frames)
        shared = self.same_capture.isChecked()
        def work(token):
            frame = function(token)
            frames[index] = frame
            compatible = all(f is not None for f in frames) and frames[0].mask.shape == frames[1].mask.shape
            if shared and compatible:
                frames[1] = replace(frames[1], axis=frames[0].axis, mm_per_pixel=frames[0].mm_per_pixel, axis_confirmed=frames[0].axis_confirmed)
            prepared = []
            for i, f in enumerate(frames):
                token.raise_if_cancelled()
                if f is None:
                    prepared.append(None)
                elif old_frames[i] is not None and f.rgba is old_frames[i].rgba and f.mask is old_frames[i].mask:
                    prepared.append(previous[i])
                else:
                    prepared.append(prepare_presentation(f, (BEFORE_COLOR, AFTER_COLOR)[i], previous[i]))
            return frames, prepared, compatible
        def done(value):
            new, prepared, compatible = value
            # Different image sizes imply a different pixel mapping. Never copy
            # an axis or scale into that frame merely because capture was fixed.
            self._remember()
            if shared and all(f is not None for f in new) and not compatible:
                self.same_capture.blockSignals(True)
                self.same_capture.setChecked(False)
                self.same_capture.blockSignals(False)
            reset = (index,) if old_frames[index] is None or old_frames[index].rgba is not new[index].rgba else ()
            self._install(new, prepared, remember=False, reset_views=reset)
            if on_installed is not None:
                on_installed()
        self._start(title, work, done)

    def _restore_picker_owner(self):
        # Native file dialogs can reactivate the main window behind a modeless
        # workspace (reproduced on Cocoa). Restore this owner after native
        # teardown, without ever raising it over another modal interaction.
        modal = QApplication.activeModalWidget()
        if self._closing or not self.isVisible() or self._picker_depth or (modal is not None and modal is not self):
            return
        self.raise_()
        self.activateWindow()

    def _choose_file(self, *, save=False, title, initial="", filters):
        pending_comparison = self._timer.isActive()
        self._timer.stop()
        self._picker_depth += 1
        try:
            chooser = QFileDialog.getSaveFileName if save else QFileDialog.getOpenFileName
            return chooser(self, title, initial, filters)[0]
        finally:
            self._picker_depth -= 1
            self._restore_picker_owner()
            self._picker_restore.start(0)
            if pending_comparison and not self._closing:
                self._timer.start()

    def _import_file(self, index):
        if self._tasks.busy:
            return
        path = self._choose_file(title="选择处理前照片" if index == 0 else "选择处理后照片", filters="照片 (*.png *.jpg *.jpeg *.tif *.tiff *.bmp *.webp);;所有文件 (*)")
        if path:
            self._replace_frame(index, lambda token: load_frame(path, token=token), "读取照片并提取轮廓")

    def _import_source(self, index, key):
        if self._tasks.busy or self.source_loader is None:
            return
        try:
            loader = self.source_loader(key)
            self._replace_frame(index, loader, "读取图片快照并提取轮廓")
        except Exception as exc:
            QMessageBox.warning(self, "无法导入图片", str(exc))

    def _automatic(self, index, method="auto", roi=None):
        frame = self.frames[index]
        if frame is None or self._tasks.busy:
            return
        def work(token):
            mask, warnings = automatic_mask(frame.rgba, method=method, roi=roi, token=token)
            return replace(frame, mask=mask, warnings=warnings, edited=True)
        self._replace_frame(index, work, "自动提取轮廓")

    def _gesture(self, index, mode, points):
        frame = self.frames[index]
        if frame is None or self._tasks.busy:
            return
        self._activate(index)
        if mode in ("wand", "wand_negative"):
            self._wand(index, points[0], negative=mode == "wand_negative")
        elif mode == "axis":
            try:
                axis = ContourAxis(*points)
            except ValueError as exc:
                self.status.setText(str(exc))
                return
            self._set_geometry(index, axis=axis)
        elif mode == "calibrate":
            distance = math.dist(*points)
            if distance < 1:
                self.status.setText("标尺两端过近，请重新选取。")
                return
            length, accepted = QInputDialog.getDouble(self, "设置实际距离", "所选标尺长度（mm，与衣物处于同一平面）：", 100, .001, 100000, 3)
            if accepted:
                self._set_geometry(index, mm_per_pixel=length / distance)
        elif mode == "roi":
            x = [p[0] for p in points]
            y = [p[1] for p in points]
            self._automatic(index, roi=(math.floor(min(x)), math.floor(min(y)), math.ceil(max(x)), math.ceil(max(y))))
        elif mode.startswith(("brush", "polygon")):
            radius = self.brush.value()
            self._replace_frame(index, lambda token: edit_mask(frame, points, add=mode.endswith("add"), radius=radius, polygon=mode.startswith("polygon")), "应用轮廓修正")

    def _set_geometry(self, index, **changes):
        if "axis" in changes:
            changes["axis_confirmed"] = True
        self._remember()
        self.frames[index] = replace(self.frames[index], **changes)
        if self.same_capture.isChecked():
            other = 1 - index
            if self.frames[other] is not None and self.frames[other].mask.shape == self.frames[index].mask.shape:
                self.frames[other] = replace(self.frames[other], **changes)
        for i, canvas in enumerate(self.canvases):
            canvas.set_frame(self.frames[i], self.presentations[i])
        self._changed()

    def _shared_changed(self, enabled):
        if self._tasks.busy:
            return
        # Record the previous checkbox state, before Qt's toggled value.
        previous = (tuple(self.frames), not enabled, self.step.value())
        self._history.append(previous)
        if enabled and all(f is not None for f in self.frames):
            if self.frames[0].mask.shape != self.frames[1].mask.shape:
                self.same_capture.blockSignals(True)
                self.same_capture.setChecked(False)
                self.same_capture.blockSignals(False)
                self._history.pop()
                self.status.setText("照片尺寸不同，请分别设置中线和标定；不能直接共用像素坐标。")
                return
            self.frames[1] = replace(self.frames[1], axis=self.frames[0].axis, mm_per_pixel=self.frames[0].mm_per_pixel, axis_confirmed=self.frames[0].axis_confirmed)
            self.canvases[1].set_frame(self.frames[1], self.presentations[1])
        self._redo.clear()
        self._changed()

    def _step_changed(self, value):
        self._history.append((tuple(self.frames), self.same_capture.isChecked(), self._last_step))
        self._redo.clear()
        self._last_step = value
        self._changed()

    def _restore(self, state, applied):
        frames, shared, step = state
        def done(prepared):
            applied()
            for widget, value in ((self.same_capture, shared), (self.step, step)):
                widget.blockSignals(True)
                widget.setChecked(value) if widget is self.same_capture else widget.setValue(value)
                widget.blockSignals(False)
            self._last_step = step
            self._install(frames, prepared, remember=False)
        self._start("恢复轮廓", self._prepare(frames), done, cancellable=False)

    def undo(self):
        if self._tasks.busy:
            if self._task_name == "计算轮廓变化":
                self._deferred_undo = True
                self._tasks.cancel()
            elif self._task_name in {"应用轮廓修正", "自动提取轮廓", "魔棒识别轮廓", "读取照片并提取轮廓", "读取图片快照并提取轮廓"}:
                self._tasks.cancel()
            return
        if any(canvas.has_gesture() for canvas in self.canvases):
            for canvas in self.canvases:
                canvas.cancel_gesture()
            return
        if self._history:
            state, current = self._history[-1], self._state()
            def applied():
                self._history.pop()
                self._redo.append(current)
            self._restore(state, applied)

    def redo(self):
        if not self._tasks.busy and self._redo:
            state, current = self._redo[-1], self._state()
            def applied():
                self._redo.pop()
                self._history.append(current)
            self._restore(state, applied)

    def _compare(self):
        if self._tasks.busy or self._picker_depth or not all(f is not None for f in self.frames):
            return
        if QApplication.activeModalWidget() is not None:
            # A calibration prompt or discard question also runs a nested
            # event loop. Do not start work that could consume its next action.
            self._timer.start()
            return
        before, after = self.frames
        step, revision = self.step.value(), self._revision
        def done(result):
            if revision != self._revision:
                return
            self.result, self._result_revision = result, revision
            self._show_result()
            self.status.setText(f"已计算 {len(result.sections)} 条截线 · 正值向外，负值向内 · {result.unit}")
        self._start("计算轮廓变化", lambda token: compare_contours(before, after, step, token=token), done)

    @staticmethod
    def _number(value, signed=False):
        if value is None:
            return "—"
        value = 0 if abs(value) < .0005 else value
        return f"{value:+.3f}" if signed else f"{value:.3f}"

    def _show_result(self):
        result = self.result
        self.overlay.set_comparison(self.frames, self.presentations, result)
        s = result.summary
        clipped = any("截断" in warning for warning in result.warnings)
        prefix = "当前选区（可能截断）" if clipped else "当前选区"
        self.summary_label.setText(f"{prefix}：上端 {s['上端向外变化']:+.3f} {result.unit}　下端 {s['下端向外变化']:+.3f} {result.unit}　纵向总长 {s['纵向总长变化']:+.3f} {result.unit}　投影面积 {s['投影面积变化率 (%)']:+.2f}%")
        self.table_model.set_result(result)
        self.warning_label.setText(" ".join(result.warnings))
        self.section_height.blockSignals(True)
        self.section_height.setRange(result.sections[0].height, result.sections[-1].height)
        self.section_height.setSingleStep(result.step)
        self.section_height.setSuffix(" " + result.unit)
        self.section_height.blockSignals(False)
        height = self._selected_height
        if height is None:
            height = result.sections[len(result.sections)//2].height
        self._select_height(height)

    def _show_more_columns(self, visible):
        for column in range(4, 10):
            self.table.setColumnHidden(column, not visible)

    def _select_height(self, height):
        if self.result is None:
            return
        index = min(range(len(self.result.sections)), key=lambda i: abs(self.result.sections[i].height-height))
        self.table.selectRow(index)
        self.table.scrollTo(self.table_model.index(index, 0), QAbstractItemView.ScrollHint.PositionAtCenter)
        self._select_section()

    def _step_section(self, direction):
        if self.result is None:
            return
        row = max(0, min(len(self.result.sections)-1, self.table.currentIndex().row() + direction))
        self._select_height(self.result.sections[row].height)

    def _select_section(self, *args):
        i = self.table.currentIndex().row()
        if self.result is None or not 0 <= i < len(self.result.sections):
            return
        row = self.result.sections[i]
        self._selected_height = row.height
        self.overlay.set_section(row)
        self.section_height.blockSignals(True)
        self.section_height.setValue(row.height)
        self.section_height.blockSignals(False)
        detail = "；".join(f"第 {i+1} 段左右边界 {self._number(left, True)} / {self._number(right, True)}" for i, (left, right) in enumerate(row.boundary_changes[:6])) if len(row.boundary_changes) > 1 else ""
        if len(row.boundary_changes) > 6:
            detail += "；更多分段见导出数据。"
        self.section_label.setText(f"高度 {row.height:.3f} {self.result.unit} · {row.status}。处理前 {len(row.before_intervals)} 段，处理后 {len(row.after_intervals)} 段；跨度变化 {self._number(row.span_change, True)} {self.result.unit}（含空隙）。" + detail)
        self.previous_section.setEnabled(i > 0)
        self.next_section.setEnabled(i < len(self.result.sections)-1)

    def _save(self):
        if self._tasks.busy or not any(f is not None for f in self.frames):
            return
        if not self._finish_gesture_first():
            self._save_then_close = False
            return
        path = self._choose_file(save=True, title="保存可继续修正的对比", initial=self.session_path or "前后轮廓对比.fdmcompare", filters="轮廓对比 (*.fdmcompare)")
        if not path:
            self._save_then_close = False
            return
        if not path.lower().endswith(".fdmcompare"):
            path += ".fdmcompare"
        before, after = self.frames
        step, shared, revision = self.step.value(), self.same_capture.isChecked(), self._revision
        def done(_):
            self.session_path = path
            self.dirty = revision != self._revision
            self.setWindowTitle("前后轮廓对比" + (" *" if self.dirty else ""))
            self.status.setText("已保存照片、修正轮廓、中线和标定：" + path)
            if self._save_then_close:
                self._save_then_close = False
                self.close()
            elif self.result is None and all(f is not None for f in self.frames):
                self._timer.start()
        self._start("保存轮廓对比", lambda token: save_comparison(path, before, after, step, same_capture=shared), done, cancellable=False)

    def _open_session(self):
        if self._tasks.busy:
            return
        if self.dirty:
            answer = QMessageBox.question(self, "打开另一个对比", "当前对比尚未保存。放弃当前更改并打开另一个对比？", QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel, QMessageBox.StandardButton.Cancel)
            if answer != QMessageBox.StandardButton.Discard:
                return
        path = self._choose_file(title="打开轮廓对比", filters="轮廓对比 (*.fdmcompare)")
        if not path:
            return
        def work(token):
            before, after, step, shared = load_comparison(path)
            prepared = [prepare_presentation(f, (BEFORE_COLOR, AFTER_COLOR)[i]) if f else None for i, f in enumerate((before, after))]
            token.raise_if_cancelled()
            return (before, after), prepared, step, shared
        def done(value):
            frames, prepared, step, shared = value
            if shared and all(f is not None for f in frames) and (frames[0].mask.shape != frames[1].mask.shape or frames[0].axis != frames[1].axis or frames[0].mm_per_pixel != frames[1].mm_per_pixel):
                shared = False
            self._history.clear()
            self._redo.clear()
            self._selected_height = None
            self.same_capture.blockSignals(True)
            self.same_capture.setChecked(shared)
            self.same_capture.blockSignals(False)
            self.step.blockSignals(True)
            self.step.setValue(step)
            self.step.blockSignals(False)
            self._last_step = step
            self._install(frames, prepared, remember=False, reset_views=(0, 1))
            self.session_path, self.dirty = path, False
            self.setWindowTitle("前后轮廓对比")
        self._start("打开轮廓对比", work, done)

    def _export(self):
        if self._tasks.busy or self.result is None or self._result_revision != self._revision:
            return
        if not self._finish_gesture_first():
            return
        path = self._choose_file(save=True, title="导出轮廓变化数据", initial="前后轮廓变化.xlsx", filters="Excel 工作簿 (*.xlsx)")
        if not path:
            return
        if not path.lower().endswith(".xlsx"):
            path += ".xlsx"
        before, after = self.frames
        result = self.result
        self._start("导出变化数据", lambda token: export_comparison_excel(path, before, after, result), lambda _: self.status.setText("已导出：" + path), cancellable=False)

    def _export_overlay(self):
        if self._tasks.busy or self.result is None or self._result_revision != self._revision:
            return
        if not self._finish_gesture_first():
            return
        path = self._choose_file(save=True, title="导出轮廓叠加示意图", initial="前后轮廓叠加.png", filters="PNG 图像 (*.png)")
        if not path:
            return
        if not path.lower().endswith(".png"):
            path += ".png"
        # Render an independent fitted widget so export never inherits a panned
        # viewport or changes the user's current view.
        preview = ContourOverlayCanvas()
        preview.resize(1400, 1600)
        preview.set_background(self.result_background.currentData())
        preview.set_comparison(self.frames, self.presentations, self.result)
        preview.set_section(self.overlay.section)
        image = QImage(preview.size(), QImage.Format.Format_ARGB32_Premultiplied)
        image.setText("Before", self.frames[0].label)
        image.setText("After", self.frames[1].label)
        image.setText("ComparisonUnit", self.result.unit)
        image.setText("MeasurementSummary", json.dumps(self.result.summary, ensure_ascii=False))
        image.setText("Notes", " ".join(self.result.warnings))
        image.setText("Background", preview.background)
        image.setText("SelectedHeight", "" if preview.height_line is None else str(preview.height_line))
        painter = QPainter(image)
        preview.paint_scene(painter, preview.size())
        painter.end()
        preview.deleteLater()
        def write(token):
            from fdm.services.contour_comparison import _atomic
            def save(temporary):
                if not image.save(temporary, "PNG"):
                    raise OSError("叠加图保存失败，请检查输出位置。")
            _atomic(path, save)
        self._start("导出叠加示意图", write, lambda _: self.status.setText("已导出：" + path), cancellable=False)

    def _finish_gesture_first(self):
        if any(canvas.has_gesture() for canvas in self.canvases):
            self.tabs.setCurrentIndex(0)
            self.status.setText("还有未完成的修正：多边形按 Enter 完成，或按 Esc 取消，然后保存／导出。")
            return False
        return True

    def reject(self):
        # Esc belongs to in-progress geometry, never silently discards a session.
        if self._picker_depth:
            return
        if self.tools["wand"].isChecked() and (self._wand_sessions[self.active] is not None or (self._tasks.busy and self._task_name == "魔棒识别轮廓")):
            self._end_wand(self.active)
            return
        if any(canvas.has_gesture() for canvas in self.canvases):
            for canvas in self.canvases:
                canvas.cancel_gesture()
        else:
            self.close()

    def closeEvent(self, event: QCloseEvent):
        if self._picker_depth:
            event.ignore()
            return
        if self._tasks.busy and not self.cancel_button.isVisible():
            self.status.setText("正在完成保存或导出，请完成后关闭。")
            event.ignore()
            return
        if self.dirty and not self._closing:
            answer = QMessageBox.question(self, "保存轮廓对比", "是否保存照片、修正轮廓、中线和标定，以便继续分析？", QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel, QMessageBox.StandardButton.Save)
            if answer == QMessageBox.StandardButton.Cancel:
                event.ignore()
                return
            if answer == QMessageBox.StandardButton.Save:
                if self._tasks.busy:
                    self.status.setText("请等待当前轮廓任务完成后保存。")
                else:
                    self._save_then_close = True
                    self._save()
                event.ignore()
                return
        self._timer.stop()
        if self._tasks.busy:
            self._closing = True
            self._tasks.cancel()
            self.status.setText("正在结束后台任务…")
            event.ignore()
            return
        # The queued result may arrive just before QRunnable.run returns.
        if not self._tasks.wait_for_done(100):
            QTimer.singleShot(50, self.close)
            event.ignore()
            return
        event.accept()
