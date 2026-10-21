"""Presentation composition for the acquisition/trial page.

Device ownership, measurements and accounting remain in their controllers.
The fixed category matrix is always available; configuration lives in a drawer.
"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal, QEvent, QSize, QTimer
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QDialog, QFormLayout, QGridLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QScrollArea, QSplitter, QTabWidget, QToolButton, QStylePainter, QStyleOptionButton, QStyle,
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


class ElidedLabel(QLabel):
    """Keep full diagnostics accessible without growing compact chrome."""
    def __init__(self, *args):
        super().__init__(*args)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.setMinimumWidth(0)

    def setText(self, text):
        super().setText(text)
        self.setToolTip(text)

    def paintEvent(self, event):
        painter = QStylePainter(self)
        text = self.fontMetrics().elidedText(self.text(), Qt.TextElideMode.ElideRight, self.contentsRect().width())
        painter.drawItemText(self.contentsRect(), self.alignment(), self.palette(), self.isEnabled(), text, self.foregroundRole())


class ElidedButton(QPushButton):
    def setText(self, text):
        super().setText(text)
        self.setToolTip(text)

    def paintEvent(self, event):
        painter = QStylePainter(self)
        option = QStyleOptionButton()
        self.initStyleOption(option)
        option.features |= QStyleOptionButton.ButtonFeature.HasMenu
        option.text = self.fontMetrics().elidedText(option.text, Qt.TextElideMode.ElideRight, max(0, self.width() - 32))
        painter.drawControl(QStyle.ControlElement.CE_PushButton, option)


class TrialMetadataScroll(QScrollArea):
    """Use natural metadata height normally; scroll locally in a short dock body."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setMinimumHeight(70)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

    def setWidget(self, widget):
        super().setWidget(widget)
        widget.installEventFilter(self)
        self._update_extent()

    def _update_extent(self):
        if self.widget() is not None:
            self.setMaximumHeight(max(70, self.widget().sizeHint().height()))
            self.updateGeometry()

    def sizeHint(self):
        return QSize(0, max(70, self.widget().sizeHint().height() if self.widget() else 70))

    def minimumSizeHint(self):
        return QSize(0, 70)

    def eventFilter(self, watched, event):
        if watched is self.widget() and event.type() == QEvent.Type.LayoutRequest:
            self._update_extent()
        return super().eventFilter(watched, event)


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
        #trialSidebar QToolButton {{ min-height: 20px; padding: 0; }}
        #trialSidebar QToolButton#trialEvidence {{ min-height: 20px; max-height: 20px; padding: 0 12px 0 0;
            border: 1px solid {border}; border-radius: 4px; background: {surface}; color: {ink}; }}
        #trialSidebar QToolButton#trialEvidence:hover {{ background: {select}; border-color: {colors.accent}; }}
        #trialSidebar QToolButton#trialEvidence:pressed {{ background: {select}; border-color: {colors.accent}; }}
        #trialSidebar QToolButton#trialEvidence:focus {{ border-color: {colors.accent}; }}
        #trialSidebar QToolButton#trialEvidence::menu-button {{ width: 11px; border-left: 1px solid {border};
            border-top-right-radius: 3px; border-bottom-right-radius: 3px; }}
        #trialSidebar QToolButton#trialEvidence::menu-button:hover {{ background: {select}; }}
        #trialSidebar QComboBox, #trialHeader QComboBox {{ min-height: 30px; border: 1px solid {border}; border-radius: 4px; padding-left: 7px; }}
        #acquisitionTrial #trialSubtitle, #acquisitionTrial #trialHint, #acquisitionTrial #trialColumnHeader {{ color: {muted}; font-size: 9pt; }}
        #acquisitionTrial #trialLastValue {{ font-size: 16pt; font-weight: 600; }}
        #acquisitionTrial #trialTitle {{ font-size: 12pt; font-weight: 600; }}
        #trialReadiness {{ padding: 4px 8px; border-radius: 4px; background: {surface}; }}
        #trialReadiness[needsRuler="true"] {{ color: {colors.warning}; background: {colors.warning_background}; }}
        #trialPending {{ background: {select}; border-radius: 4px; }}
        #acquisitionTrial QTableView {{ border: none; gridline-color: {border}; }}
    """
    if style != getattr(view, "_trial_theme_style", None):
        view._trial_theme_style = style
        view.setStyleSheet(style)
        if hasattr(view, "evidence_button"):
            from fdm.ui.icons import themed_icon
            view.evidence_button.setIcon(themed_icon("open_images"))


def build_trial_view(v):
    """Keep measurement tools compact and give categories their own viewport."""
    from fdm.ui.live_experiment_workspace import ExperimentRecordsModel, CandidatesModel, CandidateMaterialDelegate
    from fdm.ui.icons import themed_icon
    from fdm.ui.acquisition_layout import view_controls, configure_splitter, configure_status

    def button(text, callback, layout, icon=None):
        control = v._button(text, callback, layout)
        if icon:
            control.setIcon(themed_icon(icon))
        control.setAccessibleName(text)
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
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
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
    v.measure_mode = NoWheelComboBox(root)
    v.measure_mode.addItem("贴边测径", "assisted")
    v.measure_mode.addItem("自由测量", "free")
    v.measure_mode.setAccessibleName("测量方式")
    v.measure_mode.setToolTip("贴边测径自动定位纤维两侧；自由测量使用两个点击点的距离。")
    v.display_choice = NoWheelComboBox(root)
    for label, mode in (("实际像素", "pixel"), ("当前窗口", "fit"), ("指定倍率…", "magnification"), ("指定比例…", "percent")):
        v.display_choice.addItem(label, mode)
    v.display_choice.activated.connect(v._select_display_mode)
    v.body_splitter = QSplitter(Qt.Orientation.Horizontal, root)
    v.body_splitter.setChildrenCollapsible(False)
    image_panel = QWidget(v.body_splitter)
    image_layout = QVBoxLayout(image_panel)
    image_layout.setContentsMargins(0, 0, 0, 0)
    image_layout.setSpacing(6)
    v.view_controls = view_controls(image_panel, v.display_choice, "测量画面")
    image_layout.addWidget(v.view_controls)
    v.canvas.setMinimumSize(320, 240)
    v.image_scroll = QScrollArea(image_panel)
    v.image_scroll.setFrameShape(QFrame.Shape.NoFrame)
    v.image_scroll.setWidgetResizable(True)
    v.image_scroll.setWidget(v.canvas)
    v.image_scroll.horizontalScrollBar().valueChanged.connect(v.controller.cancel_draft)
    v.image_scroll.verticalScrollBar().valueChanged.connect(v.controller.cancel_draft)
    image_layout.addWidget(v.image_scroll, 1)
    v.body_splitter.addWidget(image_panel)

    side = QFrame(v.body_splitter)
    side.setObjectName("trialSidebar")
    v.trial_sidebar = side
    sidebar = QVBoxLayout(side)
    sidebar.setContentsMargins(6, 6, 6, 6)
    sidebar.setSpacing(4)
    v.metadata_scroll = TrialMetadataScroll(side)
    v.metadata_scroll.setObjectName("trialMetadataScroll")
    metadata_content = QWidget(v.metadata_scroll)
    metadata_layout = QVBoxLayout(metadata_content)
    metadata_layout.setContentsMargins(0, 0, 0, 0)
    metadata_layout.setSpacing(4)
    header = QHBoxLayout()
    header.setSpacing(3)
    v.sample_details_toggle = ElidedButton("样品信息 ▾", side)
    v.sample_details_toggle.setObjectName("trialSampleSummary")
    v.sample_details_toggle.setAccessibleName("展开或收起样品信息")
    v.sample_details_toggle.setCheckable(True)
    v.sample_details_toggle.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
    v.sample_details_toggle.setMinimumWidth(72)
    v.sample_summary = v.sample_details_toggle
    header.addWidget(v.sample_details_toggle, 1)
    v.new_trial_button = button("新建", lambda: v._new_experiment(), header, "add")
    v.switch_trial_button = button("切换", v._choose_experiment, header, "open_project")
    v.return_capture_button = button("返回采集", lambda: v._return_to_capture(), header, "acquisition")
    metadata_layout.addLayout(header)
    # Retain data-binding labels without introducing another title row.
    v.sample_title = QLabel(v.session.sample_name or "新试验", side)
    v.sample_subtitle = QLabel(side)
    v.sample_title.hide()
    v.sample_subtitle.hide()
    v.sample_details = QWidget(side)
    sample_fields = QHBoxLayout(v.sample_details)
    sample_fields.setContentsMargins(0, 0, 0, 0)
    sample_fields.setSpacing(6)
    v.sample_number = QLineEdit(v.session.sample_number, v.sample_details)
    v.sample_number.setPlaceholderText("填写样品编号")
    v.sample_number.editingFinished.connect(lambda: v._configure(sample_number=v.sample_number.text().strip()))
    v.sample = QLineEdit(v.session.sample_name, v.sample_details)
    v.sample.setPlaceholderText("填写样品名称")
    v.sample.editingFinished.connect(v._sample_changed)
    for label, editor in (("样品编号", v.sample_number), ("样品名称", v.sample)):
        field = QVBoxLayout()
        field.setSpacing(2)
        caption = QLabel(label, v.sample_details)
        caption.setObjectName("trialHint")
        field.addWidget(caption)
        editor.setAccessibleName(label)
        field.addWidget(editor)
        sample_fields.addLayout(field, 1)
    metadata_layout.addWidget(v.sample_details)
    v.sample_details.hide()
    def expand_sample(expanded):
        v.sample_details.setVisible(expanded)
        v._metadata_expansion_changed(expanded)
        v.metadata_scroll._update_extent()
        if expanded:
            QTimer.singleShot(0, v.metadata_scroll, lambda: v.metadata_scroll.ensureWidgetVisible(v.sample_details))
    v.sample_details_toggle.toggled.connect(expand_sample)

    row = QHBoxLayout()
    row.setSpacing(4)
    row.addWidget(QLabel("操作者", side))
    v.operator = NoWheelComboBox(side)
    v.operator.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
    v.operator.setAccessibleName("操作者")
    v.operator.setToolTip("切换操作者会结束当前视场，已入账的历史记录保留。")
    v.operator.currentIndexChanged.connect(v._operator_changed)
    row.addWidget(v.operator, 1)
    operator_manage = QPushButton("管理", side)
    operator_manage.setAccessibleName("管理操作者")
    operator_manage.setIcon(themed_icon("settings"))
    operator_menu = QMenu(operator_manage)
    v.rename_operator_button = operator_menu.addAction(themed_icon("rename"), "修改姓名")
    v.add_operator_button = operator_menu.addAction(themed_icon("add"), "添加操作者")
    v.rename_operator_button.triggered.connect(lambda: v._guard(v._rename_operator))
    v.add_operator_button.triggered.connect(lambda: v._guard(v._add_operator))
    operator_manage.setMenu(operator_menu)
    row.addWidget(operator_manage)
    v.direction_row = QWidget(side)
    direction_row = QHBoxLayout(v.direction_row)
    direction_row.setContentsMargins(0, 0, 0, 0)
    direction_row.setSpacing(4)
    direction_row.addWidget(QLabel("方向", v.direction_row))
    v.direction = NoWheelComboBox(v.direction_row)
    v.direction.addItem("经向", "warp")
    v.direction.addItem("纬向", "weft")
    v.direction.setAccessibleName("试验方向")
    v.direction.currentIndexChanged.connect(v._direction_changed)
    direction_row.addWidget(v.direction)
    row.addWidget(v.direction_row)
    metadata_layout.addLayout(row)
    ruler_row = QHBoxLayout()
    ruler_row.setSpacing(4)
    ruler_row.addWidget(QLabel("标尺", side))
    v.ruler_choice = NoWheelComboBox(side)
    v.ruler_choice.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
    v.ruler_choice.setAccessibleName("测量标尺")
    v.ruler_choice.activated.connect(v._select_ruler)
    ruler_row.addWidget(v.ruler_choice, 1)
    ruler_menu = QMenu(side)
    for title, callback in (("导入 CU 标尺", v._import_ruler), ("新增标尺", v._calibrate)):
        action = ruler_menu.addAction(title)
        action.triggered.connect(lambda checked=False, fn=callback: v._guard(fn))
    ruler_manage = QPushButton("管理", side)
    ruler_manage.setMenu(ruler_menu)
    ruler_manage.setIcon(themed_icon("calibration"))
    ruler_manage.setAccessibleName("管理测量标尺")
    ruler_row.addWidget(ruler_manage)
    metadata_layout.addLayout(ruler_row)
    v.readiness_label = QLabel(side)
    v.readiness_label.setObjectName("trialReadiness")
    v.readiness_label.setWordWrap(True)
    v.readiness_label.setTextInteractionFlags(Qt.TextInteractionFlag.LinksAccessibleByMouse | Qt.TextInteractionFlag.LinksAccessibleByKeyboard)
    v.readiness_label.setOpenExternalLinks(False)
    v.readiness_label.linkActivated.connect(v._readiness_action)
    metadata_layout.addWidget(v.readiness_label)
    v.metadata_scroll.setWidget(metadata_content)
    sidebar.addWidget(v.metadata_scroll)
    v.workflow_tabs = QTabWidget(side)
    v.workflow_tabs.setObjectName("trialWorkflowTabs")
    v.workflow_tabs.setMinimumHeight(80)
    v.workflow_tabs.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Ignored)
    sidebar.addWidget(v.workflow_tabs, 1)

    operations, controls = scroll_page(v.workflow_tabs, "操作")
    from fdm.ui.quantity_targets import QuantityTargets
    tools = QHBoxLayout()
    tools.setSpacing(4)
    latest = QVBoxLayout()
    latest.setSpacing(0)
    v.latest_value = ElidedLabel("— μm", operations)
    v.latest_value.setObjectName("trialLastValue")
    v.latest_value.setMinimumWidth(94)
    latest.addWidget(v.latest_value)
    v.measurement_outcome = ElidedLabel(operations)
    v.measurement_outcome.setObjectName("trialHint")
    v.measurement_outcome.setAccessibleName("本次测量状态")
    v.measurement_outcome.setFixedHeight(15)
    latest.addWidget(v.measurement_outcome)
    tools.addLayout(latest, 1)
    v.measure_mode.setMinimumWidth(110)
    v.measure_mode.setMaximumWidth(140)
    tools.addWidget(v.measure_mode)
    button("新视场", v.controller.new_field, tools, "next_field")
    v.trial_tools_button = QToolButton(operations)
    v.trial_tools_button.setIcon(themed_icon("settings"))
    v.trial_tools_button.setAccessibleName("试验工具与设置")
    v.trial_tools_button.setToolTip("含量计算设置与高级选项")
    v.trial_tools_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
    v.trial_tools_button.setFixedSize(28, 32)
    trial_tools = QMenu(v.trial_tools_button)
    v.statistics_button = trial_tools.addAction(themed_icon("statistics"), "含量计算设置")
    v.statistics_button.triggered.connect(lambda: v._guard(v._statistics_settings))
    v.advanced_toggle = trial_tools.addAction(themed_icon("settings"), "高级选项")
    v.advanced_toggle.setCheckable(True)
    v.trial_tools_button.setMenu(trial_tools)
    tools.addWidget(v.trial_tools_button)
    controls.addLayout(tools)
    v.weights_button = QWidget(operations)
    weights = QHBoxLayout(v.weights_button)
    weights.setContentsMargins(0, 0, 0, 0)
    weights.setSpacing(4)
    for label, attribute in (("经重", "warp_weight"), ("纬重", "weft_weight")):
        weights.addWidget(QLabel(label, v.weights_button))
        editor = WeightInput(v.weights_button)
        editor.setRange(0, 1000000)
        editor.setDecimals(6)
        editor.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        editor.setMinimumWidth(64)
        editor.setMaximumWidth(180)
        editor.setButtonSymbols(editor.ButtonSymbols.NoButtons)
        editor.setAccessibleName("经向重量" if attribute == "warp_weight" else "纬向重量")
        editor.setToolTip("经纬重量应使用相同单位，用于织物混合含量计算。")
        editor.editingFinished.connect(v._weights_edited)
        setattr(v, attribute, editor)
        weights.addWidget(editor, 1)
    controls.addWidget(v.weights_button)
    v.classification_mode = SegmentedChoice(["先选", "后选"], operations)
    v.classification_mode.setToolTip("先选类别用于下一根；测后改类修改本视场的最新有效测量。")
    v.classification_mode.setCurrentIndex(int(v.session.workspace_options["classification"] == "after"))
    v.classification_mode.currentIndexChanged.connect(v._classification_changed)
    v.keyboard_mode = SegmentedChoice(["数字选类", "数字补计"], operations)
    v.keyboard_mode.setCurrentIndex(1)
    v.keyboard_mode.setToolTip("单独按 Ctrl 切换。补计数只增加根数，不改变测量类别或实测直径。")
    v.category_actions = QWidget(operations)
    modes = QHBoxLayout(v.category_actions)
    modes.setContentsMargins(0, 0, 0, 0)
    modes.setSpacing(4)
    modes.addWidget(v.classification_mode, 2)
    modes.addWidget(v.keyboard_mode, 3)
    button("类别", v._materials, modes, "settings").setAccessibleName("管理纤维类别")
    controls.addWidget(v.category_actions)
    v.after_select = QCheckBox("后选", v)
    v.after_select.setChecked(v.classification_mode.currentIndex() == 1)
    v.after_select.hide()
    v.after_select.toggled.connect(lambda enabled: v.classification_mode.setCurrentIndex(int(enabled)))
    v.material = NoWheelComboBox(v)
    v.material.hide()
    v.material.activated.connect(v._after_select)
    v.material.currentIndexChanged.connect(v._sync_material_buttons)
    v.current_class_label = QLabel("当前测量类", v.category_actions)
    v.current_class_label.setObjectName("trialHint")
    v.current_class_label.setFixedSize(0, 0)

    v.category_area = QWidget(operations)
    category_layout = QVBoxLayout(v.category_area)
    category_layout.setContentsMargins(0, 0, 0, 0)
    category_layout.setSpacing(0)
    v.empty_categories = QFrame(v.category_area)
    empty_layout = QVBoxLayout(v.empty_categories)
    empty_layout.setContentsMargins(8, 8, 8, 8)
    empty_layout.addStretch(1)
    note = QLabel("选择本次试验的纤维类别后，即可开始测量。", v.empty_categories)
    note.setWordWrap(True)
    empty_layout.addWidget(note)
    v.choose_categories_button = button("选择纤维类别", v._materials, empty_layout, "add")
    empty_layout.addStretch(1)
    category_layout.addWidget(v.empty_categories)
    v.empty_categories.hide()
    v.category_scroll = QScrollArea(v.category_area)
    v.category_scroll.setObjectName("trialCategoryScroll")
    v.category_scroll.setFrameShape(QFrame.Shape.NoFrame)
    v.category_scroll.setWidgetResizable(True)
    v.category_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    v.category_scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Ignored)
    v.category_scroll.setMinimumHeight(70)
    v.material_panel = QWidget(v.category_scroll)
    v.material_panel_layout = QGridLayout(v.material_panel)
    v.material_panel_layout.setContentsMargins(0, 0, 0, 0)
    v.material_panel_layout.setHorizontalSpacing(3)
    v.material_panel_layout.setVerticalSpacing(3)
    v.material_panel_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
    v.material_rows = {}
    v.category_scroll.setWidget(v.material_panel)
    category_layout.addWidget(v.category_scroll, 1)
    controls.addWidget(v.category_area, 1)
    v.diameter_summary = QLabel("累计实测 0 根 · 均径 — μm", operations)
    v.diameter_summary.setWordWrap(True)
    controls.addWidget(v.diameter_summary)
    v.pending_panel = QFrame(operations)
    v.pending_panel.setObjectName("trialPending")
    pending = QVBoxLayout(v.pending_panel)
    pending.setContentsMargins(4, 4, 4, 4)
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
    v.quantity_targets = QuantityTargets(v.session, v._configure, operations)
    controls.addWidget(v.quantity_targets)
    v.operation_help = QLabel(operations)
    v.operation_help.setObjectName("trialHint")
    v.operation_help.hide()  # Available as the operation page's tooltip, without another row.
    operations.setToolTip("两次点击横跨纤维完成测径。Esc 取消未完成线段；撤销可回退已入账操作。")

    # Legacy selectors remain as model adapters while the dock owns record browsing.
    v.tabs = QTabWidget(v)
    v.tabs.setObjectName("trialRecords")
    v.current_model = ExperimentRecordsModel(v.session)
    v.history_model = ExperimentRecordsModel(v.session)
    v.candidate_model = CandidatesModel()
    v.current_table = v._table(v.current_model)
    v.history_table = v._table(v.history_model)
    for table, title in ((v.current_table, "本场"), (v.history_table, "全部操作者")):
        panel = QWidget(v.tabs)
        layout = QVBoxLayout(panel)
        layout.addWidget(table)
        v.tabs.addTab(panel, title)
    v.tabs.addTab(QWidget(v.tabs), "候选审核")
    v.tabs.hide()
    candidates = QWidget(v.workflow_tabs)
    v.candidate_review_page = candidates
    candidates_layout = QVBoxLayout(candidates)
    candidates_layout.setContentsMargins(4, 4, 4, 4)
    note = QLabel("候选尚未入账。选择类别，核对越限项，再确认接受。", candidates)
    note.setWordWrap(True)
    candidates_layout.addWidget(note)
    v.candidate_table = v._table(v.candidate_model)
    v.candidate_table.setItemDelegateForColumn(3, CandidateMaterialDelegate(v.candidate_table))
    v.candidate_table.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked |
        QAbstractItemView.EditTrigger.SelectedClicked | QAbstractItemView.EditTrigger.EditKeyPressed)
    for column, width in enumerate((45, 45, 80, 115, 150, 115)):
        v.candidate_table.setColumnWidth(column, width)
    candidates_layout.addWidget(v.candidate_table, 1)
    button("勾选项设为当前类别", v._assign_candidate_material, candidates_layout)
    button("确认已赋类项", v._accept_candidates, candidates_layout, "preset_apply")
    v.workflow_tabs.addTab(candidates, "审核")

    report, report_layout = scroll_page(v.workflow_tabs, "报告", "固定下方的“导出 Excel”可随时生成报告；这里设置模板、查看结果与诊断。")
    v.summary_scope = NoWheelComboBox(report)
    v.summary_scope.addItems(["当前操作者／方向", "当前操作者／经纬混合", "全部操作者／纤维汇总", "全部操作者／织物汇总"])
    v.summary_scope.currentIndexChanged.connect(lambda _: setattr(v, "_last_summary_revision", -1))
    report_layout.addWidget(v.summary_scope)
    v.summary = QLabel(report)
    v.summary.setWordWrap(True)
    v.summary.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    report_layout.addWidget(v.summary)
    button("导出原始记录 CSV", lambda: v._export(False), report_layout, "export")
    v.report_layout = QVBoxLayout()
    report_layout.addLayout(v.report_layout)
    report_layout.addStretch(1)

    actions = QHBoxLayout()
    actions.setSpacing(3)
    for label, callback, icon in (("撤销", lambda: v._history_action("undo"), "undo"),
                                  ("重做", lambda: v._history_action("redo"), "redo")):
        control = button("", callback, actions, icon)
        control.setFixedWidth(32)
        control.setAccessibleName(label)
        control.setToolTip(label)
    v.records_toggle = button("记录", lambda: v._show_records(), actions, "results")
    v.records_toggle.setCheckable(True)
    actions.addStretch(1)
    v.save_button = button("保存", v._save_project, actions, "save_project")
    v.save_button.setAccessibleName("保存项目")
    v.save_button.setToolTip("保存项目，保留试验记录、样品和配置。")
    v.export_button = button("导出 Excel", lambda: v._export(True), actions, "export")
    v.export_button.setAccessibleName("导出 Excel 报告")
    sidebar.addLayout(actions)
    status = QWidget(side)
    status.setFixedHeight(22)
    status_row = QHBoxLayout(status)
    status_row.setContentsMargins(0, 0, 0, 0)
    status_row.setSpacing(3)
    v.persistence_label = ElidedLabel("尚无需要保存的试验改动", status)
    v.persistence_label.setObjectName("trialHint")
    status_row.addWidget(v.persistence_label, 1)
    v.report_status_label = ElidedLabel(status)
    v.report_status_label.setObjectName("trialHint")
    v.report_status_label.setAccessibleName("Excel 报告状态")
    status_row.addWidget(v.report_status_label, 1)
    for name, title, icon, callback in (("open_report_button", "打开 Excel 报告", "open_images", lambda: v._open_last_report()),
            ("report_folder_button", "打开报告文件夹", "open_project", lambda: v._open_report_folder())):
        control = QToolButton(status)
        control.setIcon(themed_icon(icon))
        control.setAccessibleName(title)
        control.setToolTip(title)
        control.setFixedSize(22, 22)
        control.clicked.connect(lambda checked=False, fn=callback: v._guard(fn))
        control.setEnabled(False)
        status_row.addWidget(control)
        setattr(v, name, control)
    sidebar.addWidget(status)
    v.body_splitter.addWidget(side)
    configure_splitter(v.body_splitter, side)
    outer.addWidget(v.body_splitter, 1)
    configure_status(v.statusBar())
    outer.addWidget(v.statusBar())
    v.setCentralWidget(root)

    v.settings_dialog = QDialog(v)
    v.settings_dialog.setWindowTitle("试验高级选项")
    v.settings_dialog.resize(560, 420)
    settings_layout = QVBoxLayout(v.settings_dialog)
    v.settings_tabs = QTabWidget(v.settings_dialog)
    settings_layout.addWidget(v.settings_tabs)
    v.advanced_panel, advanced = scroll_page(v.settings_tabs, "测量与证据", "混合模式支持候选点选和所选手工手势。两点测量与按住拖线明确分开；按住 Shift 可临时强制手工测量。")
    form = QFormLayout()
    v.input_mode = NoWheelComboBox(v.advanced_panel)
    v.input_mode.setObjectName("trialInputMode")
    v.input_mode.setAccessibleName("试验交互模式")
    for label, value in (("混合：候选点选与手工绘制", "mixed"), ("仅手工测量", "manual"), ("仅候选点选", "candidate")):
        v.input_mode.addItem(label, value)
    form.addRow("交互模式", v.input_mode)
    v.gesture = NoWheelComboBox(v.advanced_panel)
    v.gesture.addItems(["两点测量", "按住拖线"])
    v.gesture.setCurrentIndex(int(v.session.workspace_options.get("gesture") == "drag"))
    v.gesture.currentIndexChanged.connect(v._gesture_changed)
    form.addRow("手工测量手势", v.gesture)
    advanced.addLayout(form)
    v.keep_images = QCheckBox("保存测量原图（可选）", v.advanced_panel)
    v.keep_images.setChecked(v.session.save_images)
    v.keep_images.toggled.connect(lambda value: v._configure(save_images=value))
    v.keep_images.hide()  # Retain the legacy configuration binding for old projects.
    v.evidence_storage = NoWheelComboBox(v.advanced_panel)
    v.evidence_storage.setObjectName("trialEvidenceStorage")
    v.evidence_storage.setAccessibleName("测量图片存档")
    for label, value in (("不保存", "disabled"), ("局部 JPEG · 质量 95", "crop_jpeg"), ("完整无损 PNG", "full_png")):
        v.evidence_storage.addItem(label, value)
    v.evidence_storage.setCurrentIndex(v.evidence_storage.findData(v.session.effective_evidence_options["mode"]))
    v.evidence_storage.currentIndexChanged.connect(lambda _: v._configure(evidence_options={"version": 1, "mode": v.evidence_storage.currentData()}))
    form.addRow("测量图片存档", v.evidence_storage)
    evidence_note = QLabel("仅影响之后接受的测量，已有图片保持原样。局部图片保留源像素，透明输入使用 PNG。点击操作页的“证据相册”浏览本试验图片。", v.advanced_panel)
    evidence_note.setWordWrap(True)
    advanced.addWidget(evidence_note)
    button("使用主窗口当前图片", v._load_static, advanced, "open_images")
    button("放弃所选失败证据", v._discard_failed_evidence, advanced)
    advanced.addStretch(1)
    optional, optional_layout = scroll_page(v.settings_tabs, "全图分析", "分析固定的一帧，在画布点选候选，或在审核页批量确认。确认前不计入试验结果。")
    v.algorithm = NoWheelComboBox(optional)
    v.algorithm.addItem("传统全图", "traditional")
    v.algorithm.addItem("多尺度全图", "multiscale")
    optional_layout.addWidget(v.algorithm)
    def analyze():
        v._analyze_candidates()
        v.settings_dialog.close()
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
    shortcuts = [("Escape", v._cancel_trial_gesture)]
    # Embedded pages share the main window's history router, including their
    # floating records view. A second Ctrl+Z action would be ambiguous.
    if not v.embedded:
        shortcuts.extend((("Ctrl+Z", lambda: v._history_action("undo")),
                          ("Ctrl+Y", lambda: v._history_action("redo")),
                          ("Ctrl+Shift+Z", lambda: v._history_action("redo"))))
    for sequence, callback in shortcuts:
        action = QAction(v)
        action.setShortcut(QKeySequence(sequence))
        action.setShortcutContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        action.triggered.connect(lambda checked=False, fn=callback: v._guard(fn))
        v.addAction(action)
    apply_trial_theme(v)
    v.statusBar().setSizeGripEnabled(False)
    v.statusBar().setMaximumHeight(26)
