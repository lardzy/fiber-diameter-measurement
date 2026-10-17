"""Presentation composition for the acquisition/trial page.

Device ownership, measurements and accounting remain in their controllers.
The fixed category matrix is always available; configuration lives in a drawer.
"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QDialog, QFormLayout, QGridLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QScrollArea, QSplitter, QTabWidget,
    QVBoxLayout, QWidget, QFrame, QSizePolicy, QDialogButtonBox, QMenu, QApplication,
)
from fdm.ui.widgets import NoWheelComboBox, NoWheelDoubleSpinBox


class SegmentedChoice(QWidget):
    currentIndexChanged = Signal(int)

    def __init__(self, labels, parent=None):
        super().__init__(parent)
        self._index = -1
        self.buttons = []
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        for index, label in enumerate(labels):
            button = QPushButton(label)
            button.setCheckable(True)
            button.setProperty("segment", True)
            button.clicked.connect(lambda checked=False, i=index: self.setCurrentIndex(i))
            self.buttons.append(button)
            layout.addWidget(button)
        self.setCurrentIndex(0)

    def currentIndex(self):
        return self._index

    def currentText(self):
        return self.buttons[self._index].text()

    def setCurrentIndex(self, index):
        if not 0 <= index < len(self.buttons):
            return
        changed = self._index != index
        self._index = index
        for i, button in enumerate(self.buttons):
            button.setChecked(i == index)
        if changed:
            self.currentIndexChanged.emit(index)


class WeightInput(NoWheelDoubleSpinBox):
    """Keep six-decimal input precision without padding every weight with zeros."""

    def textFromValue(self, value):
        return self.locale().toString(value, "f", self.decimals()).rstrip("0").rstrip(self.locale().decimalPoint()) or "0"


def apply_trial_theme(view):
    from fdm.ui.theme import semantic_colors
    palette = QApplication.palette()
    colors = semantic_colors(palette)
    bg = palette.color(palette.ColorRole.Window).name()
    surface = palette.color(palette.ColorRole.Base).name()
    border, ink, muted, select = colors.border, colors.text, colors.muted, colors.info_background
    view.setObjectName("acquisitionTrial")
    style = f"""
        #acquisitionTrial {{ background: {bg}; color: {ink}; font-size: 10.5pt; }}
        #acquisitionTrial QWidget {{ font-size: 10.5pt; }}
        #acquisitionTrial QLabel {{ color: {ink}; background: transparent; padding: 0; margin: 0; }}
        #trialSidebar, #trialHeader, #trialRecords {{ background: {surface}; border: 1px solid {border}; border-radius: 6px; }}
        #trialSidebar QLabel, #trialHeader QLabel, #trialRecords QLabel {{ border: none; }}
        #trialSidebar QPushButton, #trialHeader QPushButton {{ min-height: 30px; padding: 0 7px; border: 1px solid {border}; border-radius: 4px; background: {surface}; color: {ink}; }}
        #trialSidebar QPushButton:hover, #trialHeader QPushButton:hover {{ border-color: {colors.accent}; background: {select}; }}
        #trialSidebar QPushButton:checked, #trialHeader QPushButton:checked {{ background: {select}; border-color: {colors.accent}; color: {ink}; font-weight: 600; }}
        #trialSidebar QPushButton:disabled {{ color: {muted}; }}
        #trialSidebar QComboBox, #trialHeader QComboBox {{ min-height: 30px; border: 1px solid {border}; border-radius: 4px; padding-left: 7px; }}
        #acquisitionTrial #trialSubtitle, #acquisitionTrial #trialHint, #acquisitionTrial #trialColumnHeader {{ color: {muted}; font-size: 9pt; }}
        #acquisitionTrial #trialLastValue {{ font-size: 18pt; font-weight: 600; }}
        #acquisitionTrial #trialTitle {{ font-size: 12pt; font-weight: 600; }}
        #trialReadiness {{ padding: 4px 8px; border-radius: 4px; background: {surface}; }}
        #trialReadiness[needsRuler="true"] {{ color: {colors.warning}; background: {colors.warning_background}; }}
        #trialPending {{ background: {select}; border-radius: 4px; }}
        #acquisitionTrial QTableView {{ border: none; gridline-color: {border}; }}
    """
    if style != getattr(view, "_trial_theme_style", None):
        view._trial_theme_style = style
        view.setStyleSheet(style)


def build_trial_view(v):
    """One stable image area and a persistent, task-oriented trial console."""
    from fdm.ui.live_experiment_workspace import ExperimentRecordsModel, CandidatesModel, CandidateMaterialDelegate
    from fdm.ui.icons import themed_icon

    def button(text, callback, layout, icon=None):
        control = v._button(text, callback, layout)
        if icon:
            control.setIcon(themed_icon(icon))
        return control

    def scroll_page(tabs, title, description=None):
        scroll = QScrollArea(tabs)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setProperty("redirectEditorWheel", True)
        content = QWidget(scroll)
        content.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        layout = QVBoxLayout(content)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        if description:
            note = QLabel(description, content)
            note.setWordWrap(True)
            note.setObjectName("trialHint")
            layout.addWidget(note)
        scroll.setWidget(content)
        tabs.addTab(scroll, title)
        return content, layout

    root = QWidget(v)
    outer = QVBoxLayout(root)
    outer.setContentsMargins(8, 8, 8, 4)
    outer.setSpacing(6)
    header = QFrame(root)
    header.setObjectName("trialHeader")
    top = QHBoxLayout(header)
    top.setContentsMargins(10, 5, 10, 5)
    v.sample_title = QLabel(v.session.sample_name or "新试验", header)
    v.sample_title.setObjectName("trialTitle")
    v.sample_title.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
    top.addWidget(v.sample_title, 1)
    v.sample_subtitle = QLabel(header)
    v.sample_subtitle.setObjectName("trialSubtitle")
    v.sample_subtitle.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
    top.addWidget(v.sample_subtitle, 2)
    button("切换试验", v._choose_experiment, top, "open_project")
    button("新视场", v.controller.end_field, top, "next_field")
    v.measure_mode = NoWheelComboBox(header)
    v.measure_mode.addItem("贴边测径", "assisted")
    v.measure_mode.addItem("自由测量", "free")
    v.measure_mode.setToolTip("贴边测径自动定位纤维两侧；自由测量使用两个点击点的距离。")
    top.addWidget(v.measure_mode)
    v.display_choice = NoWheelComboBox(header)
    for label, mode in (("实际像素", "pixel"), ("当前窗口", "fit"), ("指定倍率…", "magnification"), ("指定比例…", "percent")):
        v.display_choice.addItem(label, mode)
    v.display_choice.activated.connect(v._select_display_mode)
    top.addWidget(v.display_choice)
    outer.addWidget(header)

    v.body_splitter = QSplitter(Qt.Orientation.Horizontal, root)
    v.body_splitter.setChildrenCollapsible(False)
    v.canvas.setMinimumSize(320, 240)
    v.image_scroll = QScrollArea(v.body_splitter)
    v.image_scroll.setFrameShape(QFrame.Shape.NoFrame)
    v.image_scroll.setWidgetResizable(True)
    v.image_scroll.setWidget(v.canvas)
    v.image_scroll.horizontalScrollBar().valueChanged.connect(v.controller.cancel_draft)
    v.image_scroll.verticalScrollBar().valueChanged.connect(v.controller.cancel_draft)
    v.body_splitter.addWidget(v.image_scroll)

    side = QFrame(v.body_splitter)
    side.setObjectName("trialSidebar")
    side.setMinimumWidth(390)
    side.setMaximumWidth(520)
    v.trial_sidebar = side
    sidebar = QVBoxLayout(side)
    sidebar.setContentsMargins(8, 8, 8, 8)
    sidebar.setSpacing(6)
    metadata = QFormLayout()
    metadata.setContentsMargins(0, 0, 0, 0)
    metadata.setHorizontalSpacing(6)
    metadata.setVerticalSpacing(5)
    metadata.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
    v.sample_number = QLineEdit(v.session.sample_number, side)
    v.sample_number.setPlaceholderText("填写样品编号")
    v.sample_number.editingFinished.connect(lambda: v._configure(sample_number=v.sample_number.text().strip()))
    v.sample = QLineEdit(v.session.sample_name, side)
    v.sample.setPlaceholderText("填写样品名称")
    v.sample.editingFinished.connect(v._sample_changed)
    sample_fields = QHBoxLayout()
    for label, editor in (("样品编号", v.sample_number), ("样品名称", v.sample)):
        field = QVBoxLayout()
        field.setSpacing(2)
        caption = QLabel(label, side)
        caption.setObjectName("trialHint")
        field.addWidget(caption)
        editor.setAccessibleName(label)
        field.addWidget(editor)
        sample_fields.addLayout(field, 1)
    sidebar.addLayout(sample_fields)
    row = QHBoxLayout()
    v.operator = NoWheelComboBox(side)
    v.operator.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    v.operator.setToolTip("切换操作者会结束当前视场，已入账的历史记录保留。")
    v.operator.currentIndexChanged.connect(v._operator_changed)
    row.addWidget(v.operator, 1)
    v.rename_operator_button = button("改名", v._rename_operator, row, "rename")
    v.add_operator_button = button("添加", v._add_operator, row, "add")
    metadata.addRow("操作者", row)
    v.ruler_choice = NoWheelComboBox(side)
    v.ruler_choice.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
    v.ruler_choice.activated.connect(v._select_ruler)
    ruler_row = QHBoxLayout()
    ruler_row.addWidget(v.ruler_choice, 1)
    ruler_menu = QMenu(side)
    for title, callback in (("导入 CU 标尺", v._import_ruler), ("新增标尺", v._calibrate)):
        action = ruler_menu.addAction(title)
        action.triggered.connect(lambda checked=False, fn=callback: v._guard(fn))
    ruler_manage = QPushButton("管理", side)
    ruler_manage.setMenu(ruler_menu)
    ruler_manage.setIcon(themed_icon("calibration"))
    ruler_row.addWidget(ruler_manage)
    metadata.addRow("测量标尺", ruler_row)
    sidebar.addLayout(metadata)
    v.readiness_label = QLabel(side)
    v.readiness_label.setObjectName("trialReadiness")
    v.readiness_label.setWordWrap(True)
    v.readiness_label.setTextInteractionFlags(Qt.TextInteractionFlag.LinksAccessibleByMouse | Qt.TextInteractionFlag.LinksAccessibleByKeyboard)
    v.readiness_label.setOpenExternalLinks(False)
    v.readiness_label.linkActivated.connect(v._readiness_action)
    sidebar.addWidget(v.readiness_label)
    v.workflow_tabs = QTabWidget(side)
    v.workflow_tabs.setObjectName("trialWorkflowTabs")
    v.workflow_tabs.setMinimumHeight(140)
    v.workflow_tabs.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Ignored)
    sidebar.addWidget(v.workflow_tabs, 1)

    operations, controls = scroll_page(v.workflow_tabs, "操作")
    latest = QHBoxLayout()
    v.latest_value = QLabel("— μm", operations)
    v.latest_value.setObjectName("trialLastValue")
    latest.addWidget(v.latest_value, 1)
    v.direction_row = QWidget(operations)
    row = QHBoxLayout(v.direction_row)
    row.setContentsMargins(0, 0, 0, 0)
    row.addWidget(QLabel("方向", v.direction_row))
    v.direction = NoWheelComboBox(v.direction_row)
    v.direction.addItem("经向", "warp")
    v.direction.addItem("纬向", "weft")
    v.direction.currentIndexChanged.connect(v._direction_changed)
    row.addWidget(v.direction)
    v.weights_button = QWidget(operations)
    weights = QHBoxLayout(v.weights_button)
    weights.setContentsMargins(0, 0, 0, 0)
    weights.setSpacing(3)
    for label, attribute in (("经向重量", "warp_weight"), ("纬向重量", "weft_weight")):
        weights.addWidget(QLabel(label, v.weights_button))
        editor = WeightInput(v.weights_button)
        editor.setRange(0, 1000000)
        editor.setDecimals(6)
        editor.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        editor.setMinimumWidth(100)
        editor.setMaximumWidth(180)
        editor.setButtonSymbols(editor.ButtonSymbols.NoButtons)
        editor.setAccessibleName(label)
        editor.setToolTip("经纬重量应使用相同单位，用于织物混合含量计算。")
        editor.editingFinished.connect(v._weights_edited)
        setattr(v, attribute, editor)
        weights.addWidget(editor, 1)
    latest.addWidget(v.direction_row)
    controls.addLayout(latest)
    controls.addWidget(v.weights_button)
    v.classification_mode = SegmentedChoice(["先选", "后选"], operations)
    v.classification_mode.setToolTip("先选类别用于下一根；测后改类修改本视场的最新有效测量。")
    v.classification_mode.setCurrentIndex(int(v.session.workspace_options["classification"] == "after"))
    v.classification_mode.currentIndexChanged.connect(v._classification_changed)
    v.keyboard_mode = SegmentedChoice(["数字选类", "数字补计"], operations)
    v.keyboard_mode.setCurrentIndex(1)
    v.keyboard_mode.setToolTip("单独按 Ctrl 切换。补计数只增加根数，不改变测量类别或实测直径。")
    modes = QHBoxLayout()
    modes.addWidget(v.classification_mode, 1)
    modes.addWidget(v.keyboard_mode, 1)
    controls.addLayout(modes)
    v.after_select = QCheckBox("后选", v)
    v.after_select.setChecked(v.classification_mode.currentIndex() == 1)
    v.after_select.hide()
    v.after_select.toggled.connect(lambda enabled: v.classification_mode.setCurrentIndex(int(enabled)))
    v.material = NoWheelComboBox(v)
    v.material.hide()
    v.material.activated.connect(v._after_select)
    v.material.currentIndexChanged.connect(v._sync_material_buttons)
    v.category_actions = QWidget(operations)
    row = QHBoxLayout(v.category_actions)
    row.setContentsMargins(0, 0, 0, 0)
    v.current_class_label = QLabel("当前测量类", v.category_actions)
    v.current_class_label.setWordWrap(True)
    v.current_class_label.setObjectName("trialHint")
    row.addWidget(v.current_class_label, 1)
    button("管理类别", v._materials, row, "settings")
    controls.addWidget(v.category_actions)
    v.material_panel = QWidget(operations)
    v.material_panel_layout = QGridLayout(v.material_panel)
    v.material_panel_layout.setContentsMargins(0, 0, 0, 0)
    v.material_panel_layout.setHorizontalSpacing(3)
    v.material_panel_layout.setVerticalSpacing(3)
    v.material_rows = {}
    controls.addWidget(v.material_panel)
    v.diameter_summary = QLabel("累计实测 0 根 · 均径 — μm", operations)
    v.diameter_summary.setWordWrap(True)
    controls.addWidget(v.diameter_summary)
    v.pending_panel = QFrame(operations)
    v.pending_panel.setObjectName("trialPending")
    pending = QVBoxLayout(v.pending_panel)
    pending.setContentsMargins(6, 6, 6, 6)
    v.pending_label = QLabel(v.pending_panel)
    v.pending_label.setWordWrap(True)
    pending.addWidget(v.pending_label)
    row = QHBoxLayout()
    v.pending_buttons = [button("保留", lambda: v._resolve_displayed_pending(), row),
        button("改为当前类", lambda: v._resolve_displayed_pending(material_id=v.material.currentData()), row),
        button("放弃", lambda: v._resolve_displayed_pending(discard=True), row)]
    pending.addLayout(row)
    v.pending_panel.hide()
    controls.addWidget(v.pending_panel)
    v.candidate_feedback = QLabel(operations)
    v.candidate_feedback.setWordWrap(True)
    v.candidate_feedback.hide()
    controls.addWidget(v.candidate_feedback)
    v.records_toggle = QPushButton("查看本场记录", operations)
    v.records_toggle.setCheckable(True)
    v.records_toggle.setIcon(themed_icon("results"))
    controls.addWidget(v.records_toggle)
    v.statistics_button = button("含量计算设置", v._statistics_settings, controls, "statistics")
    v.advanced_toggle = QPushButton("高级选项", operations)
    v.advanced_toggle.setIcon(themed_icon("settings"))
    v.advanced_toggle.setCheckable(True)
    controls.addWidget(v.advanced_toggle)
    v.operation_help = QLabel("两次点击横跨纤维完成测径。Esc 取消未完成线段；撤销可回退已入账操作。", operations)
    v.operation_help.setObjectName("trialHint")
    v.operation_help.setWordWrap(True)
    controls.addWidget(v.operation_help)
    controls.addStretch(1)

    v.tabs = QTabWidget(v.workflow_tabs)
    v.tabs.setObjectName("trialRecords")
    v.current_model = ExperimentRecordsModel(v.session)
    v.history_model = ExperimentRecordsModel(v.session)
    v.candidate_model = CandidatesModel()
    v.current_table = v._table(v.current_model)
    v.history_table = v._table(v.history_model)
    v.candidate_table = v._table(v.candidate_model)
    v.candidate_table.setItemDelegateForColumn(3, CandidateMaterialDelegate(v.candidate_table))
    v.candidate_table.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked |
        QAbstractItemView.EditTrigger.SelectedClicked | QAbstractItemView.EditTrigger.EditKeyPressed)
    for column, width in enumerate((45, 45, 80, 115, 150, 115)):
        v.candidate_table.setColumnWidth(column, width)
    for table, title in ((v.current_table, "本场"), (v.history_table, "全部操作者")):
        panel = QWidget(v.tabs)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(table, 1)
        actions = QHBoxLayout()
        button("修订", v._edit_record, actions, "rename")
        button("删除", v._delete_record, actions, "delete")
        button("原图", v._review_record, actions, "open_images")
        layout.addLayout(actions)
        v.tabs.addTab(panel, title)
    candidates = QWidget(v.tabs)
    candidates_layout = QVBoxLayout(candidates)
    note = QLabel("候选尚未入账。选择类别，核对越限项，再确认接受。", candidates)
    note.setWordWrap(True)
    candidates_layout.addWidget(note)
    candidates_layout.addWidget(v.candidate_table, 1)
    button("勾选项设为当前类别", v._assign_candidate_material, candidates_layout)
    button("确认已赋类项", v._accept_candidates, candidates_layout, "preset_apply")
    v.tabs.addTab(candidates, "候选审核")
    v.workflow_tabs.addTab(v.tabs, "记录")
    v.records_toggle.toggled.connect(lambda checked: v.workflow_tabs.setCurrentIndex(1 if checked else 0))
    def sync_record_tab(index):
        v.records_toggle.blockSignals(True)
        v.records_toggle.setChecked(index == 1)
        v.records_toggle.blockSignals(False)
    v.workflow_tabs.currentChanged.connect(sync_record_tab)

    report, report_layout = scroll_page(v.workflow_tabs, "报告", "保存项目保留试验记录与设置；生成报告可用于复核、编辑和打印。")
    v.summary_scope = NoWheelComboBox(report)
    v.summary_scope.addItems(["当前操作者／方向", "当前操作者／经纬混合", "全部操作者／纤维汇总", "全部操作者／织物汇总"])
    v.summary_scope.currentIndexChanged.connect(lambda _: setattr(v, "_last_summary_revision", -1))
    report_layout.addWidget(v.summary_scope)
    v.summary = QLabel(report)
    v.summary.setWordWrap(True)
    v.summary.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    report_layout.addWidget(v.summary)
    button("导出原始记录 CSV", lambda: v._export(False), report_layout, "export")
    button("生成模板报告", lambda: v._export(True), report_layout, "export")
    v.report_layout = QVBoxLayout()
    report_layout.addLayout(v.report_layout)
    report_layout.addStretch(1)

    actions = QHBoxLayout()
    button("撤销", lambda: v._history_action("undo"), actions, "undo")
    button("重做", lambda: v._history_action("redo"), actions, "redo")
    actions.addStretch(1)
    v.save_button = button("保存项目", v._save_project, actions, "save_project")
    sidebar.addLayout(actions)
    v.persistence_label = QLabel("尚无需要保存的试验改动", side)
    v.persistence_label.setObjectName("trialHint")
    v.persistence_label.setWordWrap(True)
    v.persistence_label.setMaximumHeight(34)
    sidebar.addWidget(v.persistence_label)
    v.body_splitter.addWidget(side)
    v.body_splitter.setStretchFactor(0, 1)
    v.body_splitter.setStretchFactor(1, 0)
    v.body_splitter.setSizes([800, 432])
    outer.addWidget(v.body_splitter, 1)
    outer.addWidget(v.statusBar())
    v.setCentralWidget(root)

    v.settings_dialog = QDialog(v)
    v.settings_dialog.setWindowTitle("试验高级选项")
    v.settings_dialog.resize(560, 420)
    settings_layout = QVBoxLayout(v.settings_dialog)
    v.settings_tabs = QTabWidget(v.settings_dialog)
    settings_layout.addWidget(v.settings_tabs)
    v.advanced_panel, advanced = scroll_page(v.settings_tabs, "测量与原图", "常用样品信息和类别可在右侧直接调整。这里设置测量手势及可选证据。")
    form = QFormLayout()
    v.gesture = NoWheelComboBox(v.advanced_panel)
    v.gesture.addItems(["两次点击", "拖动测量"])
    v.gesture.setCurrentIndex(int(v.session.workspace_options.get("gesture") == "drag"))
    v.gesture.currentIndexChanged.connect(v._gesture_changed)
    form.addRow("测量手势", v.gesture)
    advanced.addLayout(form)
    v.keep_images = QCheckBox("保存测量原图（可选）", v.advanced_panel)
    v.keep_images.setChecked(v.session.save_images)
    v.keep_images.toggled.connect(lambda value: v._configure(save_images=value))
    advanced.addWidget(v.keep_images)
    button("使用主窗口当前图片", v._load_static, advanced, "open_images")
    button("放弃所选失败原图", v._discard_failed_evidence, advanced)
    advanced.addStretch(1)
    optional, optional_layout = scroll_page(v.settings_tabs, "全图分析", "分析固定的一帧，结果在右侧记录页审核。确认前不计入试验结果。")
    v.algorithm = NoWheelComboBox(optional)
    v.algorithm.addItem("传统全图", "traditional")
    v.algorithm.addItem("多尺度全图", "multiscale")
    optional_layout.addWidget(v.algorithm)
    def analyze():
        v.controller.analyze(v.algorithm.currentData())
        v.settings_dialog.close()
        v.candidate_feedback.setText("正在分析当前帧，实时画面继续更新。")
        v.candidate_feedback.show()
    button("分析当前视场", analyze, optional_layout, "statistics")
    button("取消全图分析", v.controller.discard_candidates, optional_layout)
    optional_layout.addStretch(1)
    close = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, v.settings_dialog)
    close.button(QDialogButtonBox.StandardButton.Close).setText("关闭")
    close.rejected.connect(v.settings_dialog.close)
    settings_layout.addWidget(close)
    v.advanced_toggle.toggled.connect(v.settings_dialog.setVisible)
    v.settings_dialog.finished.connect(lambda _: v.advanced_toggle.setChecked(False))
    v.mode = NoWheelComboBox(v)
    v.mode.addItems(["含量试验", "直径试验"])
    v.mode.setCurrentIndex(int(v.session.workspace_options.get("mode") == "diameter"))
    v.mode.hide()
    v.count_add_button = QPushButton(v)
    v.count_subtract_button = QPushButton(v)
    v.count_add_button.hide()
    v.count_subtract_button.hide()
    v.mode.currentIndexChanged.connect(v._mode_changed)
    v._mode_changed(v.mode.currentIndex(), persist=False)
    for sequence, callback in (("Escape", v.controller.cancel_draft),
            ("Ctrl+Z", lambda: v._history_action("undo")), ("Ctrl+Y", lambda: v._history_action("redo"))):
        action = QAction(v)
        action.setShortcut(QKeySequence(sequence))
        action.setShortcutContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        action.triggered.connect(lambda checked=False, fn=callback: v._guard(fn))
        v.addAction(action)
    apply_trial_theme(v)
    v.statusBar().setSizeGripEnabled(False)
    v.statusBar().setMaximumHeight(26)
