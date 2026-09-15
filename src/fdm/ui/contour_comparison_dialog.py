"""Non-destructive before/after silhouette workspace, independent of documents."""
from __future__ import annotations

from dataclasses import replace
import math
import json

from PySide6.QtCore import QAbstractTableModel, QEvent, QModelIndex, QSize, Qt, QTimer
from PySide6.QtGui import QAction, QCloseEvent, QImage, QKeySequence, QPainter, QPalette
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QButtonGroup, QDialog,
    QFileDialog, QFrame, QHBoxLayout, QHeaderView, QInputDialog,
    QLabel, QMenu, QMessageBox, QPushButton, QScrollArea, QSplitter,
    QTableView, QTabWidget, QToolButton, QVBoxLayout, QWidget, QWidgetAction,
    QStackedWidget, QSlider, QSizePolicy,
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
from fdm.ui.contour_comparison_widgets import ElidedLabel, MetricCard, WorkspaceCheckBox, WORKSPACE_STYLE
from fdm.ui.icons import themed_icon
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
        if role == Qt.ItemDataRole.ToolTipRole and orientation == Qt.Orientation.Horizontal:
            return "所有距离使用同一单位。左右变化：正值向外伸展，负值向内收缩；跨度包含腿缝等空隙。"
        return super().headerData(section, orientation, role)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or self.result is None or index.row() >= len(self.result.sections):
            return None
        if role == Qt.ItemDataRole.TextAlignmentRole and index.column() < 10:
            return Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        if role == Qt.ItemDataRole.ToolTipRole:
            row = self.result.sections[index.row()]
            value = getattr(row, self.fields[index.column()])
            return f"{self.headers[index.column()]}：{'无对应轮廓' if value is None else value} {self.result.unit if index.column() < 10 else ''}\n{row.status}"
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
        self.setObjectName("contourWorkspace")
        self.setStyleSheet(WORKSPACE_STYLE)
        self._icon_buttons = []
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 12, 14, 10)
        root.setSpacing(8)
        top = QHBoxLayout()
        title = QLabel("前后轮廓对比")
        font = title.font()
        font.setPointSizeF(font.pointSizeF() + 5)
        font.setBold(True)
        title.setFont(font)
        top.addWidget(title)
        top.addStretch()
        self.open_session_button = self._button("打开对比…", self._open_session, top)
        self.save_button = self._button("保存对比…", self._save, top)
        self.export_button = self._button("导出数据…", self._export, top)
        self.overlay_export_button = self._button("导出叠加图…", self._export_overlay, top)
        for button, icon in ((self.open_session_button, "open_project"), (self.save_button, "save_project"), (self.export_button, "results"), (self.overlay_export_button, "export")):
            self._add_icon(button, icon)
        root.addLayout(top)
        self.explanation = ElidedLabel("识别前后轮廓，沿同一参考中线比较外形变化。可随时修正，结果自动更新。")
        self.explanation.setProperty("muted", True)
        root.addWidget(self.explanation)

        reference = QFrame()
        self.reference_bar = reference
        reference.setObjectName("comparisonReference")
        controls = QHBoxLayout(reference)
        controls.setContentsMargins(10, 4, 10, 4)
        controls.setSpacing(10)
        controls.addWidget(QLabel("测量基准"))
        self.same_capture = WorkspaceCheckBox("共用中线和标定")
        self.same_capture.setChecked(True)
        self.same_capture.setToolTip("固定机位且分辨率、焦距、裁切、拍摄平面相同时可共用。取消后可分别设置；不会清除现有中线。重新勾选以处理前为准。第一点必须对应同一固定参考高度。")
        self.same_capture.toggled.connect(self._shared_changed)
        controls.addWidget(self.same_capture)
        self.reference_hint = ElidedLabel("固定机位 · 两张同步设置")
        self.reference_hint.setProperty("muted", True)
        controls.addWidget(self.reference_hint, 1)
        controls.addWidget(QLabel("截线间隔"))
        self.step = NoWheelDoubleSpinBox()
        self.step.setDecimals(3)
        self.step.setRange(.01, 10000)
        self.step.setValue(10)
        self.step.setSuffix(" px")
        self.step.setFixedWidth(116)
        self.step.setToolTip("沿中线每隔此距离测一条截线。不是测量精度；间隔更密不代表更准确。")
        self.step.valueChanged.connect(self._step_changed)
        controls.addWidget(self.step)
        root.addWidget(reference)

        self.tabs = QTabWidget()
        self.quality_badge = QToolButton()
        self.quality_badge.setProperty("badge", True)
        self.quality_badge.clicked.connect(self._locate_reference)
        self.tabs.setCornerWidget(self.quality_badge)
        edit_page = QWidget()
        edit_layout = QVBoxLayout(edit_page)
        edit_layout.setContentsMargins(0, 8, 0, 0)
        edit_layout.setSpacing(7)
        tool_widget = QFrame()
        tool_widget.setObjectName("comparisonToolbar")
        toolbar = QHBoxLayout(tool_widget)
        toolbar.setContentsMargins(6, 4, 6, 4)
        toolbar.setSpacing(2)
        self.tool_group = QButtonGroup(self)
        self.tool_group.setExclusive(True)
        self.tools = {}
        definitions = (
            ("browse", "查看", "select", "滚轮缩放，右键拖动；点击任一照片即可切换编辑对象"),
            ("wand", "魔棒", "magic_segment", "点击目标，连续补点；Alt＋点击排除背景。完成后自动应用，Esc 结束点选"),
            ("brush_add", "补画", "freehand_area", "按住左键补入选区；[ / ] 调整画笔，X 切换补画与剔除"),
            ("brush_remove", "剔除", "mask_erase", "按住左键剔除选区；[ / ] 调整画笔，X 切换补画与剔除"),
            ("polygon_add", "圈入", "polygon_area", "逐点圈入，Enter 或双击完成；Backspace 退回一点，X 切换圈入与圈除"),
            ("polygon_remove", "圈除", "polygon_subtract", "逐点剔除，Enter 或双击完成；Backspace 退回一点，X 切换圈入与圈除"),
            ("roi", "框选识别", "area_auto", "拖框包含整个目标及少量背景，重新自动识别；可撤销"),
            ("axis", "设中线", "manual", "先点固定参考高度的原点，再沿中线向下点第二点；不能分别追随衣物新的上端"),
            ("calibrate", "标定", "calibration", "在与衣物同一平面的标尺上点两端，再输入实际距离（mm）"),
        )
        for mode, text, icon, tip in definitions:
            if mode in ("axis",):
                divider = QFrame()
                divider.setFrameShape(QFrame.Shape.VLine)
                toolbar.addWidget(divider)
            button = QToolButton()
            button.setText(text)
            shortcut = {"browse": "V", "wand": "W", "brush_add": "B", "brush_remove": "E", "polygon_add": "P"}.get(mode)
            button.setToolTip(tip + (f" · 快捷键 {shortcut}" if shortcut else ""))
            button.setAccessibleName(text)
            button.setCheckable(True)
            button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
            self._add_icon(button, icon)
            self.tool_group.addButton(button)
            button.clicked.connect(lambda _checked=False, selected=mode: self._set_mode(selected))
            toolbar.addWidget(button)
            self.tools[mode] = button
        toolbar.addStretch()
        self.undo_button = self._button("撤销", self.undo, toolbar)
        self.redo_button = self._button("重做", self.redo, toolbar)
        self.undo_button.setToolTip("撤销上一次修正（Ctrl+Z / ⌘Z）")
        self.redo_button.setToolTip("恢复已撤销的修正（Ctrl+Shift+Z / ⌘⇧Z）")
        edit_layout.addWidget(tool_widget)

        context = QFrame()
        context.setObjectName("comparisonContext")
        context_layout = QVBoxLayout(context)
        context_layout.setContentsMargins(10, 3, 10, 5)
        context_layout.setSpacing(2)
        # A stable two-line area prevents tool changes from moving either photo.
        self.tool_options = QStackedWidget()
        self.tool_options.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.default_options = QWidget()
        default_row = QHBoxLayout(self.default_options)
        default_row.setContentsMargins(0, 0, 0, 0)
        self.mode_caption = ElidedLabel()
        default_row.addWidget(self.mode_caption, 1)
        self.tool_options.addWidget(self.default_options)
        self.wand_options = QWidget()
        wand_row = QHBoxLayout(self.wand_options)
        wand_row.setContentsMargins(0, 0, 0, 0)
        wand_row.addWidget(QLabel("魔棒结果"))
        self.wand_operation = NoWheelComboBox()
        for label, value in (("替换轮廓", "replace"), ("补入轮廓", "add"), ("剔除区域", "remove")):
            self.wand_operation.addItem(label, value)
        self.wand_operation.setToolTip("替换：重新选择整个物体。补入／剔除：只修改所选区域，连续点选基于本次开始前的轮廓。")
        self.wand_operation.currentIndexChanged.connect(self._wand_operation_changed)
        wand_row.addWidget(self.wand_operation)
        self.wand_negative = WorkspaceCheckBox("点选排除背景")
        self.wand_negative.setToolTip("或按住 Alt 点击；右键仍拖动画布。剔除模式中，负点指不需要剔除的部分。")
        wand_row.addWidget(self.wand_negative)
        self.wand_reset_button = self._button("重新选点", self._reset_wand, wand_row)
        wand_row.addStretch()
        self.tool_options.addWidget(self.wand_options)
        self.brush_options = QWidget()
        brush_row = QHBoxLayout(self.brush_options)
        brush_row.setContentsMargins(0, 0, 0, 0)
        brush_row.addWidget(QLabel("画笔半径"))
        self.brush = NoWheelSpinBox()
        self.brush.setRange(1, 500)
        self.brush.setValue(8)
        self.brush.setSuffix(" px")
        self.brush.setFixedWidth(100)
        self.brush.valueChanged.connect(self._brush_changed)
        brush_row.addWidget(self.brush)
        brush_tip = ElidedLabel("原图像素 · [ 缩小 / ] 放大 · X 切换补画与剔除")
        brush_tip.setProperty("muted", True)
        brush_row.addWidget(brush_tip, 1)
        self.tool_options.addWidget(self.brush_options)
        options_row = QHBoxLayout()
        options_row.addWidget(self.tool_options, 1)
        self.coverage_visible = WorkspaceCheckBox("显示覆盖")
        self.coverage_visible.setChecked(True)
        self.coverage_visible.setToolTip("只切换选区填色；轮廓、原图和计算范围不变。")
        self.coverage_visible.toggled.connect(self._coverage_changed)
        options_row.addWidget(self.coverage_visible)
        context_layout.addLayout(options_row)
        self.tool_hint = ElidedLabel()
        self.tool_hint.setProperty("muted", True)
        context_layout.addWidget(self.tool_hint)
        edit_layout.addWidget(context)

        self.image_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.canvases, self.name_labels, self.scale_labels, self.import_buttons, self.auto_buttons = [], [], [], [], []
        self.image_panels, self.active_labels, self.axis_labels, self.source_labels = [], [], [], []
        for index, (text, color) in enumerate((("处理前", BEFORE_COLOR), ("处理后", AFTER_COLOR))):
            panel = QFrame()
            panel.setObjectName("comparisonImagePanel")
            panel_layout = QVBoxLayout(panel)
            panel_layout.setContentsMargins(9, 8, 9, 7)
            panel_layout.setSpacing(5)
            header = QHBoxLayout()
            label = QLabel(("●  " if index == 0 else "■  ") + text)
            label.setStyleSheet(f"color: {color.name()}; font-weight: 600;")
            self.source_labels.append(label)
            header.addWidget(label)
            active_label = ElidedLabel()
            active_label.setProperty("muted", True)
            header.addWidget(active_label, 1)
            self.active_labels.append(active_label)
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
            fit = QToolButton()
            fit.setText("适合窗口")
            fit.setToolTip("适合窗口（F）")
            self._add_icon(fit, "fit")
            fit.clicked.connect(lambda _checked=False, i=index: self.canvases[i].fit())
            header.addWidget(fit)
            panel_layout.addLayout(header)
            name_label = ElidedLabel("尚未导入照片")
            name_label.setProperty("muted", True)
            panel_layout.addWidget(name_label)
            self.name_labels.append(name_label)
            canvas = ContourImageCanvas(color)
            canvas.set_source_title(text)
            canvas.import_requested.connect(lambda i=index: self._import_file(i))
            canvas.activated.connect(lambda i=index: self._activate(i))
            canvas.gesture.connect(lambda mode, points, i=index: self._gesture(i, mode, points))
            canvas.wand_finished.connect(lambda i=index: self._end_wand(i))
            canvas.tool_requested.connect(self._set_mode)
            canvas.brush_adjusted.connect(lambda delta: self.brush.setValue(self.brush.value() + delta))
            panel_layout.addWidget(canvas, 1)
            self.canvases.append(canvas)
            footer = QHBoxLayout()
            scale_label = ElidedLabel("未标定 · 仅像素")
            footer.addWidget(scale_label, 1)
            self.scale_labels.append(scale_label)
            axis_label = QLabel("中线待设置")
            axis_label.setProperty("muted", True)
            footer.addWidget(axis_label)
            self.axis_labels.append(axis_label)
            panel_layout.addLayout(footer)
            self.image_panels.append(panel)
            self.image_splitter.addWidget(panel)
        self.image_splitter.setChildrenCollapsible(False)
        edit_layout.addWidget(self.image_splitter, 1)
        self.tabs.addTab(edit_page, "照片与轮廓")

        result_page = QWidget()
        result_layout = QVBoxLayout(result_page)
        result_layout.setContentsMargins(0, 8, 0, 0)
        result_layout.setSpacing(7)
        self.summary_label = ElidedLabel("当前选区 · 导入前后照片后自动计算")
        self.summary_label.setProperty("muted", True)
        result_layout.addWidget(self.summary_label)
        metrics = QHBoxLayout()
        metrics.setSpacing(8)
        self.metrics = {}
        for key, title, description in (("length", "纵向总长变化", "下端到上端的投影长度"), ("top", "上端变化", "相对固定零高度 · 向外为正"), ("bottom", "下端变化", "相对固定零高度 · 向外为正"), ("area", "投影面积变化", "当前选区 · 不等同材料应变")):
            card = MetricCard(title, description)
            metrics.addWidget(card, 1)
            self.metrics[key] = card
        result_layout.addLayout(metrics)
        result_tools = QHBoxLayout()
        result_tools.addWidget(QLabel("底图"))
        self.result_background = NoWheelComboBox()
        for label, value in (("处理前照片", "before"), ("处理后照片", "after"), ("仅看轮廓", "none")):
            self.result_background.addItem(label, value)
        result_tools.addWidget(self.result_background)
        self.result_fit_button = self._button("适合窗口", lambda: self.overlay.fit(), result_tools)
        result_tools.addStretch()
        result_tools.addWidget(QLabel("定位高度"))
        self.section_height = NoWheelDoubleSpinBox()
        self.section_height.setDecimals(3)
        self.section_height.setFixedWidth(124)
        self.section_height.valueChanged.connect(self._select_height)
        result_tools.addWidget(self.section_height)
        self.previous_section = self._button("上一处", lambda: self._step_section(-1), result_tools)
        self.next_section = self._button("下一处", lambda: self._step_section(1), result_tools)
        self.records_visible = WorkspaceCheckBox("显示记录")
        self.records_visible.setChecked(True)
        result_tools.addWidget(self.records_visible)
        result_layout.addLayout(result_tools)
        self.result_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.overlay = ContourOverlayCanvas()
        self.overlay.setMinimumWidth(300)
        self.overlay.height_selected.connect(self._select_height)
        self.overlay.section_stepped.connect(self._step_section)
        self.result_background.currentIndexChanged.connect(lambda: self.overlay.set_background(self.result_background.currentData()))
        self.result_splitter.addWidget(self.overlay)
        self.records_panel = QFrame()
        self.records_panel.setObjectName("comparisonRecords")
        records = QVBoxLayout(self.records_panel)
        records.setContentsMargins(1, 6, 1, 0)
        records.setSpacing(4)
        record_header = QHBoxLayout()
        record_header.setContentsMargins(9, 0, 9, 0)
        self.record_count = QLabel("逐高度记录")
        record_header.addWidget(self.record_count)
        record_header.addStretch()
        self.more_columns = WorkspaceCheckBox("详细数据")
        self.more_columns.toggled.connect(self._show_more_columns)
        record_header.addWidget(self.more_columns)
        records.addLayout(record_header)
        interval_row = QHBoxLayout()
        interval_row.setContentsMargins(9, 0, 9, 0)
        interval_row.addWidget(QLabel("截线间隔"))
        self.result_step = NoWheelDoubleSpinBox()
        self.result_step.setDecimals(3)
        self.result_step.setRange(.01, 10000)
        self.result_step.setValue(self.step.value())
        self.result_step.setFixedWidth(116)
        self.result_step.setToolTip(self.step.toolTip())
        self.result_step.valueChanged.connect(self.step.setValue)
        interval_row.addWidget(self.result_step)
        interval_row.addStretch()
        records.addLayout(interval_row)
        self.table = QTableView()
        self.table_model = _ProfileTableModel(self)
        self.table.setModel(self.table_model)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setShowGrid(False)
        self.table.setAlternatingRowColors(True)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.table.horizontalHeader().setDefaultSectionSize(78)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setColumnWidth(10, 180)
        self.table.setMinimumWidth(280)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(30)
        self.table.selectionModel().selectionChanged.connect(self._select_section)
        records.addWidget(self.table, 1)
        self.result_splitter.addWidget(self.records_panel)
        self.result_splitter.setStretchFactor(0, 3)
        self.result_splitter.setStretchFactor(1, 2)
        self.result_splitter.setChildrenCollapsible(False)
        self.result_splitter.setSizes([720, 460])
        self.records_visible.toggled.connect(self.records_panel.setVisible)
        result_layout.addWidget(self.result_splitter, 1)
        locator = QHBoxLayout()
        locator.addWidget(QLabel("上端"))
        self.section_slider = QSlider(Qt.Orientation.Horizontal)
        self.section_slider.setToolTip("连续浏览已计算的高度；不更改采样间隔或测量数据")
        self.section_slider.valueChanged.connect(self._select_section_index)
        locator.addWidget(self.section_slider, 1)
        locator.addWidget(QLabel("下端"))
        self.position_label = QLabel("— / —")
        locator.addWidget(self.position_label)
        result_layout.addLayout(locator)
        self.section_label = ElidedLabel("点击样品或记录，查看这一高度的左右外缘变化。")
        result_layout.addWidget(self.section_label)
        self.tabs.addTab(result_page, "变化结果")
        self.tabs.currentChanged.connect(self._page_changed)
        root.addWidget(self.tabs, 1)

        notice = QHBoxLayout()
        self.warning_label = ElidedLabel("请先导入前后照片；照片应完整包含样品和同平面标尺。")
        self.warning_label.setProperty("muted", True)
        notice.addWidget(self.warning_label, 1)
        self.notes_button = QToolButton()
        self.notes_button.setText("测量说明")
        self.notes_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.notes_menu = QMenu(self.notes_button)
        self.notes_menu.aboutToShow.connect(self._populate_notes)
        self.notes_button.setMenu(self.notes_menu)
        notice.addWidget(self.notes_button)
        root.addLayout(notice)
        bottom = QHBoxLayout()
        self.status = ElidedLabel("准备就绪 · 滚轮缩放 · 右键拖动")
        bottom.addWidget(self.status, 1)
        self.cancel_button = self._button("取消计算", self._tasks.cancel, bottom)
        self.cancel_button.hide()
        close_button = self._button("关闭", self.close, bottom)
        close_button.setAutoDefault(False)
        root.addLayout(bottom)
        # Enter completes polygons, never an accidentally focused default button.
        for button in self.findChildren(QPushButton):
            button.setAutoDefault(False)
        for standard, callback in ((QKeySequence.StandardKey.Undo, self.undo), (QKeySequence.StandardKey.Redo, self.redo), (QKeySequence.StandardKey.Save, self._save)):
            action = QAction(self)
            action.setShortcuts(QKeySequence.keyBindings(standard))
            action.triggered.connect(callback)
            self.addAction(action)
        self._set_mode("browse")
        self._activate(0)
        self._refresh_labels()
        self._show_more_columns(False)

    def _add_icon(self, button, name):
        self._refresh_icon(button, name)
        button.setIconSize(QSize(16, 16))
        self._icon_buttons.append((button, name))
        if button.isCheckable():
            button.toggled.connect(lambda: self._refresh_icon(button, name))

    def _refresh_icon(self, button, name):
        role = QPalette.ColorRole.HighlightedText if button.isChecked() else QPalette.ColorRole.WindowText
        button.setIcon(themed_icon(name, color=self.palette().color(role).name()))

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() in (QEvent.Type.PaletteChange, QEvent.Type.ApplicationPaletteChange):
            for button, name in getattr(self, "_icon_buttons", ()):
                self._refresh_icon(button, name)
            if hasattr(self, "quality_badge") and hasattr(self, "scale_labels"):
                self._refresh_labels()

    def _populate_notes(self):
        self.notes_menu.clear()
        label = QLabel("测量说明\n\n" + self.warning_label.text() + "\n\n同高度外形差异包含摆放影响，不代表同一材料点的应变。原图、原分辨率轮廓和标定参与计算；屏幕缩放不影响数值。")
        label.setTextFormat(Qt.TextFormat.PlainText)
        label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        label.setWordWrap(True)
        width = min(460, max(260, self.width() - 80))
        label.setFixedWidth(width)
        label.setContentsMargins(14, 12, 14, 12)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(label)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFixedSize(width + 22, min(max(160, label.heightForWidth(width) + 8), max(180, self.height() - 200)))
        action = QWidgetAction(self.notes_menu)
        action.setDefaultWidget(scroll)
        self.notes_menu.addAction(action)

    def _locate_reference(self):
        self.tabs.setCurrentIndex(0)
        loaded = [i for i, frame in enumerate(self.frames) if frame is not None]
        if not loaded:
            self.status.setText("点击处理前／处理后画布中的按钮导入照片。")
            return
        if len(loaded) == 1:
            self._import_file(1 - loaded[0])
            return
        index = next((i for i in loaded if self.frames[i].mm_per_pixel is None), None)
        if index is not None:
            self._activate(index)
            self._set_mode("calibrate")
        else:
            index = next((i for i in loaded if not self.frames[i].axis_confirmed), self.active)
            self._activate(index)
            self._set_mode("axis")
        self.canvases[index].setFocus()

    def _wand_operation_changed(self):
        self._reset_wand(all_sources=True)
        removing = self.wand_operation.currentData() == "remove"
        self.wand_negative.setText("点选保留部分" if removing else "点选排除背景")

    def _select_section_index(self, index):
        if self.result is not None and 0 <= index < len(self.result.sections):
            self._select_height(self.result.sections[index].height)

    def _page_changed(self, index):
        self.reference_bar.setVisible(index == 0)
        self.explanation.setVisible(index == 0 and self.height() >= 760)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        compact = event.size().height() < 760
        for card in getattr(self, "metrics", {}).values():
            card.description.setVisible(not compact)
        if hasattr(self, "tabs"):
            self.explanation.setVisible(not compact and self.tabs.currentIndex() == 0)

    def _activate(self, index):
        self.active = index
        for i, canvas in enumerate(self.canvases):
            canvas.selected = i == index
            canvas.update()
            self.active_labels[i].setText("正在编辑" if i == index else "点击编辑")
            panel = self.image_panels[i]
            if panel.property("active") != (i == index):
                panel.setProperty("active", i == index)
                panel.style().unpolish(panel)
                panel.style().polish(panel)
                panel.update()

    def _set_mode(self, mode):
        if self._tasks.busy:
            return
        self.tools[mode].setChecked(True)
        for canvas in self.canvases:
            canvas.set_mode(mode)
        self.tool_hint.setText(self.tools[mode].toolTip())
        self.tool_options.setCurrentWidget(self.wand_options if mode == "wand" else self.brush_options if mode.startswith("brush") else self.default_options)
        captions = {
            "browse": "点击照片切换编辑对象 · 滚轮缩放 · 右键拖动 · F 适合窗口",
            "axis": "两点设置：① 固定零高度　② 沿中线向下",
            "calibrate": "两点标定：① 标尺起点　② 标尺终点 → 输入实际毫米",
            "polygon_add": "圈入选区 · Enter 完成 · Backspace 退回一点 · Esc 取消",
            "polygon_remove": "圈除选区 · Enter 完成 · Backspace 退回一点 · Esc 取消",
            "roi": "框住整个目标，留少量背景 · 松开鼠标后重新识别",
        }
        self.mode_caption.setText(captions.get(mode, ""))

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
                self.status.setText(f"请先取消“{self.wand_negative.text()}”，在要识别的区域内点一下，再添加排除点。")
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
        for card in self.metrics.values():
            card.set_value("—")
        self.record_count.setText("逐高度记录")
        self.position_label.setText("— / —")
        self.step.setSuffix(" mm" if all(f is not None and f.mm_per_pixel is not None for f in self.frames) else " px")
        self.result_step.blockSignals(True)
        self.result_step.setValue(self.step.value())
        self.result_step.setSuffix(self.step.suffix())
        self.result_step.blockSignals(False)
        self.setWindowTitle("前后轮廓对比" + (" *" if self.dirty else ""))
        self._trim_history()
        self._refresh_labels()
        self._update_controls()
        if all(f is not None for f in self.frames):
            self._timer.start()

    def _refresh_labels(self):
        warnings = []
        dark = self.palette().color(QPalette.ColorRole.Window).lightness() < 128
        warning_color = "#F3BF70" if dark else "#935500"
        for label, color in zip(self.source_labels, ("#51c4dd", "#ffb571") if dark else ("#006f86", "#a84e02")):
            label.setStyleSheet(f"color: {color}; font-weight: 600;")
        for i, frame in enumerate(self.frames):
            if frame is None:
                self.name_labels[i].setText("导入后自动提取轮廓，可继续手动修正")
                self.scale_labels[i].setText("尚未导入")
                self.scale_labels[i].setStyleSheet("")
                self.axis_labels[i].setText("中线待设置")
                continue
            self.name_labels[i].setText(frame.label)
            self.name_labels[i].setToolTip(frame.source_path or frame.label)
            if frame.mm_per_pixel is None:
                self.scale_labels[i].setText("未标定 · 仅 px / px²")
                self.scale_labels[i].setStyleSheet(f"color: {warning_color}; font-weight: 600;")
            else:
                self.scale_labels[i].setText(f"已标定 · {frame.mm_per_pixel:.6g} mm/px")
                self.scale_labels[i].setStyleSheet("")
            self.scale_labels[i].setToolTip(f"{self.scale_labels[i].text()} · 原点 ({frame.axis.origin[0]:.1f}, {frame.axis.origin[1]:.1f})")
            self.axis_labels[i].setText("中线已设置" if frame.axis_confirmed else "中线待设置")
            warnings.extend(frame.warnings)
        self.reference_hint.setText("固定机位 · 两张同步设置" if self.same_capture.isChecked() else "分别设置 · 两图须对应同一零高度")
        loaded = [f for f in self.frames if f is not None]
        calibrated = len(loaded) == 2 and all(f.mm_per_pixel is not None for f in loaded)
        confirmed = calibrated and all(f.axis_confirmed for f in loaded)
        badge = "导入前后照片" if not loaded else "未标定 · 仅像素" if not calibrated else "mm · 中线待设置" if not confirmed else "mm · 中线已设置"
        if len(loaded) == 1:
            badge = "待导入处理后照片" if self.frames[1] is None else "待导入处理前照片"
        if len(loaded) == 2 and sum(f.mm_per_pixel is not None for f in loaded) == 1:
            badge = "标定不一致 · 待设置"
        self.quality_badge.setText(badge)
        self.quality_badge.setToolTip("点击定位到需要设置的照片和工具。共用设置只适用于相同机位、尺寸及拍摄平面。")
        self.quality_badge.setStyleSheet(f"color: {warning_color};" if loaded and not confirmed else "")
        message = "请复核中线与轮廓；两图的零高度必须对应同一固定参考位置。"
        if warnings:
            message += " " + " ".join(dict.fromkeys(warnings))
        self.warning_label.setText(" ".join(self.result.warnings) if self.result else message)

    def _update_controls(self):
        busy = self._tasks.busy
        for button in self.import_buttons:
            button.setEnabled(not busy)
        for i, button in enumerate(self.auto_buttons):
            button.setEnabled(not busy and self.frames[i] is not None)
        for button in self.tools.values():
            button.setEnabled(not busy and any(f is not None for f in self.frames))
        for canvas in self.canvases:
            canvas.editing_enabled = not busy
            canvas.import_button.setEnabled(not busy)
        self.open_session_button.setEnabled(not busy)
        self.save_button.setEnabled(not busy and any(f is not None for f in self.frames))
        ready = not busy and self.result is not None and self._result_revision == self._revision
        self.export_button.setEnabled(ready)
        self.overlay_export_button.setEnabled(ready)
        self.undo_button.setEnabled(not busy and bool(self._history))
        self.redo_button.setEnabled(not busy and bool(self._redo))
        self.same_capture.setEnabled(not busy)
        self.step.setEnabled(not busy)
        self.result_step.setEnabled(not busy)
        self.brush.setEnabled(not busy)
        self.wand_options.setEnabled(not busy)
        self.quality_badge.setEnabled(not busy)
        has_result = self.result is not None
        self.section_height.setEnabled(has_result)
        self.section_slider.setEnabled(has_result)
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
            self._activate(index)
            self._replace_frame(index, lambda token: load_frame(path, token=token), "读取照片并提取轮廓")

    def _import_source(self, index, key):
        if self._tasks.busy or self.source_loader is None:
            return
        try:
            loader = self.source_loader(key)
            self._activate(index)
            self._replace_frame(index, loader, "读取图片快照并提取轮廓")
        except Exception as exc:
            QMessageBox.warning(self, "无法导入图片", str(exc))

    def _automatic(self, index, method="auto", roi=None):
        frame = self.frames[index]
        if frame is None or self._tasks.busy:
            return
        self._activate(index)
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
        self.summary_label.setText(f"{prefix} · 处理后 − 处理前 · 下方截线定位局部变化")
        for name, key in (("length", "纵向总长变化"), ("top", "上端向外变化"), ("bottom", "下端向外变化")):
            self.metrics[name].set_value(f"{self._number(s[key], True)} {result.unit}")
        self.metrics["area"].set_value(f"{s['投影面积变化率 (%)']:+.2f}%")
        self.record_count.setText(f"逐高度记录 · {len(result.sections)} 条")
        self.table_model.set_result(result)
        self.warning_label.setText(" ".join(result.warnings))
        self.section_height.blockSignals(True)
        self.section_height.setRange(result.sections[0].height, result.sections[-1].height)
        self.section_height.setSingleStep(result.step)
        self.section_height.setSuffix(" " + result.unit)
        self.section_height.blockSignals(False)
        self.section_slider.blockSignals(True)
        self.section_slider.setRange(0, len(result.sections) - 1)
        self.section_slider.blockSignals(False)
        height = self._selected_height
        if height is None:
            height = result.sections[len(result.sections)//2].height
        self._select_height(height)

    def _show_more_columns(self, visible):
        for column in range(4, 11):
            self.table.setColumnHidden(column, not visible)
        header = self.table.horizontalHeader()
        header.setStretchLastSection(visible)
        for column in range(4):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.Interactive if visible else QHeaderView.ResizeMode.Stretch)

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
        self.section_slider.blockSignals(True)
        self.section_slider.setValue(i)
        self.section_slider.blockSignals(False)
        self.position_label.setText(f"{i+1} / {len(self.result.sections)}")
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
