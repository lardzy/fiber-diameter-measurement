"""Draft-only trial preferences editor shared by the application's settings UI."""
from __future__ import annotations

from dataclasses import replace
from copy import deepcopy

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QLinearGradient, QPainter, QPen
from PySide6.QtWidgets import (
    QColorDialog, QCheckBox, QFormLayout, QFrame, QGroupBox, QHBoxLayout,
    QLabel, QPushButton, QScrollArea, QTabWidget, QVBoxLayout, QWidget,
)

from fdm.services.optical_config import ALGORITHM_CHOICES, IMAGING_MODE_CHOICES
from fdm.trial_preferences import TrialPreferences
from fdm.ui.widgets import NoWheelComboBox, NoWheelDoubleSpinBox, NoWheelFontComboBox, NoWheelSpinBox


class TrialLabelPreview(QWidget):
    """Uses the same formatter, font metrics and label painter as the canvas."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.preferences = TrialPreferences()
        self.setObjectName("trialLabelPreview")
        self.setMinimumSize(260, 160)
        self.setAccessibleName("含量与直径试验标记实时预览")

    def set_preferences(self, preferences):
        self.preferences = preferences.normalized_copy()
        self.update()

    def paintEvent(self, event):
        from fdm.ui.trial_label_style import draw_trial_label, trial_label_font, trial_label_size, trial_label_text
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        background = QLinearGradient(0, 0, self.width(), 0)
        background.setColorAt(0, QColor("#282E35"))
        background.setColorAt(1, QColor("#C7CBCB"))
        painter.fillRect(self.rect(), background)
        font = trial_label_font(self.preferences, self.font())
        for row, (mode, color) in enumerate((("content", "#27B9BD"), ("diameter", "#FF9469"))):
            y = 42. + row * 78.
            left, right = 20., max(80., self.width() - 20.)
            painter.setPen(QPen(QColor(color), 2.))
            painter.drawLine(QPointF(left, y), QPointF(right, y))
            for x in (left, right):
                painter.drawLine(QPointF(x, y - 6), QPointF(x, y + 6))
            text = trial_label_text(self.preferences, mode=mode, number=row + 1,
                                   material_name="绵羊毛", diameter_um=18.25, decimals=2)
            max_width = max(80., min(240., self.width() - 32.))
            size = trial_label_size(text, font, max_width=max_width)
            rect = QRectF((self.width() - size.width()) / 2, y + 6, size.width(), size.height())
            draw_trial_label(painter, rect, text, color, self.preferences, base_font=self.font(), max_width=max_width)
        painter.end()


class TrialSettingsPage(QWidget):
    cleanupRequested = Signal()

    def __init__(self, preferences: TrialPreferences, parent=None):
        super().__init__(parent)
        self.setObjectName("trialSettingsPage")
        self._loading = True
        self._draft = preferences.normalized_copy()
        self._observed_optical_resources = self._optical_resources(self._draft)
        self.controls = {}
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        note = QLabel("全局偏好：含量与直径试验共用。显示立即更新，测量参数用于之后开始的测量。", self)
        note.setWordWrap(True)
        outer.addWidget(note)
        self.tabs = QTabWidget(self)
        outer.addWidget(self.tabs, 1)
        display = self._page("显示与标记")
        self._check(display, "show_stability_status", "显示视场稳定状态")
        self._note(display, "控制底部两侧日常就绪／核验提示；不关闭视场核验，异常及未入账原因仍显示。")
        group, form = self._group(display, "测量标记")
        self._combo(form, "content_label", "含量试验文字", (
            ("类别全称", "material"), ("测量编号", "number"), ("直径值与单位", "value"),
            ("编号与类别全称", "number_material")))
        self._combo(form, "diameter_label", "直径试验文字", (("直径值与单位", "value"), ("测量编号", "number")))
        self._check(form, "label_background", "显示白色文字背景")
        self.font_combo = NoWheelFontComboBox(group)
        self.font_combo.setObjectName("trialFontFamily")
        self.font_combo.setAccessibleName("试验标记字体")
        self.font_combo.setPlaceholderText("跟随应用字体")
        self.font_combo.lineEdit().setPlaceholderText("跟随应用字体")
        self.font_combo.currentFontChanged.connect(self._font_changed)
        family_row = QWidget(group)
        family_layout = QHBoxLayout(family_row)
        family_layout.setContentsMargins(0, 0, 0, 0)
        family_layout.addWidget(self.font_combo, 1)
        self.application_font = QPushButton("应用字体", family_row)
        self.application_font.clicked.connect(self._use_application_font)
        family_layout.addWidget(self.application_font)
        form.addRow("字体", family_row)
        self._spin(form, "font_size", "字号", 6, 72, suffix=" pt")
        self._check(form, "font_bold", "粗体")
        self._check(form, "font_italic", "斜体")
        self._combo(form, "color_mode", "文字颜色", (("跟随类别／测量线", "follow"), ("自定义颜色", "custom")))
        self.color_button = QPushButton(group)
        self.color_button.setObjectName("trialFontColor")
        self.color_button.clicked.connect(self._choose_color)
        form.addRow("自定义颜色", self.color_button)
        self.preview = TrialLabelPreview(group)
        form.addRow(self.preview)
        self._note(display, "预览上方为含量试验，下方为直径试验。字号保持屏幕尺寸；候选及待确认标记继续显示状态。")
        self._spin(display, "retention_seconds", "换场后保留标记", 0., 10., suffix=" 秒", floating=True)
        display.addStretch(1)

        storage = self._page("测量与图片存档")
        _, goals = self._group(storage, "数量提醒")
        self.total_target = NoWheelSpinBox(self)
        self.total_target.setObjectName("trialContentTotalTarget")
        self.total_target.setRange(1, 100_000_000)
        self.total_target.setValue(1500)
        self.total_target.setSuffix(" 根")
        goals.addRow("含量试验总根数", self.total_target)
        self._note(storage, "统一用于所有已有及以后新建的含量试验。按每位操作者的经／纬方向分别累计实测＋补计，达到后提醒，仍可继续。类别的实测和计数目标在纤维类别库中设置。")
        _, interaction = self._group(storage, "测量交互")
        self._combo(interaction, "input_mode", "交互模式", (("混合：候选点选与手工绘制", "mixed"),
            ("仅手工测量", "manual"), ("仅候选点选", "candidate")))
        self._combo(interaction, "gesture", "手工测量手势", (("两点测量", "click"), ("按住拖线", "drag")))
        self._combo(interaction, "manual_algorithm", "手工测径方式", (("贴边测径", "assisted"), ("自由测量", "free")))
        _, archive = self._group(storage, "测量图片存档")
        self._check(archive, "evidence_enabled", "保存测量图片")
        self._combo(archive, "evidence_format", "图片格式", (("局部 JPEG · 质量 95", "crop_jpeg"),
            ("完整无损 PNG", "full_png")))
        self._note(storage, "默认不保存。开启后跨重启记忆；局部按原分辨率裁剪，以 JPEG 有损压缩，透明图片使用 PNG。关闭不删除已有图片，正在处理的测量仍使用原策略。")
        cleanup_group, cleanup = self._group(storage, "临时图片清理")
        self.cleanup_button = QPushButton("立即清理无用临时图片", cleanup_group)
        self.cleanup_button.setObjectName("trialCleanupEvidence")
        self.cleanup_button.clicked.connect(self.cleanupRequested)
        cleanup.addRow(self.cleanup_button)
        self.cleanup_status = QLabel("自动清理已退出实例的临时图片；当前测量、撤销、复核与保存仍在使用的图片会保留。", cleanup_group)
        self.cleanup_status.setWordWrap(True)
        cleanup.addRow(self.cleanup_status)
        self._note(storage, "清理立即执行，与“应用／取消”无关。仅清理程序管理的无引用临时图片，不删除项目资源、另存图片或报告。")
        storage.addStretch(1)

        analysis = self._page("全图分析")
        _, analysis_form = self._group(analysis, "候选分析")
        self._combo(analysis_form, "algorithm", "分析算法", ALGORITHM_CHOICES)
        self._check(analysis_form, "automatic", "视场稳定后自动分析")
        self._check(analysis_form, "visible", "显示候选标记")
        self._spin(analysis_form, "opacity", "候选不透明度", 20, 100, suffix=" %")
        _, optical = self._group(analysis, "普通光学参数")
        self.imaging_mode = NoWheelComboBox(self)
        for label, value in IMAGING_MODE_CHOICES:
            self.imaging_mode.addItem(label, value)
        optical.addRow("成像结构", self.imaging_mode)
        self.flat_enabled = QCheckBox("启用已采集的空白平场参考", self)
        optical.addRow(self.flat_enabled)
        self.reference_status = QLabel(self)
        self.reference_status.setWordWrap(True)
        optical.addRow(self.reference_status)
        self._note(analysis, "分析、取消、采集空白参考与光照检查在试验界面执行；候选仍需确认后才计入。这里不启动设备或分析任务。")
        analysis.addStretch(1)
        self.set_preferences(preferences)

    def _page(self, title):
        scroll = QScrollArea(self.tabs)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        widget = QWidget(scroll)
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(6, 8, 6, 8)
        scroll.setWidget(widget)
        self.tabs.addTab(scroll, title)
        return layout

    @staticmethod
    def _group(layout, title):
        group = QGroupBox(title)
        form = QFormLayout(group)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        layout.addWidget(group)
        return group, form

    @staticmethod
    def _note(layout, text):
        label = QLabel(text)
        label.setWordWrap(True)
        layout.addWidget(label)

    def _check(self, layout, name, text):
        widget = QCheckBox(text, self)
        self._register(name, widget)
        (layout.addRow if isinstance(layout, QFormLayout) else layout.addWidget)(widget)
        widget.toggled.connect(self._update_preview)
        return widget

    def _combo(self, layout, name, label, choices):
        widget = NoWheelComboBox(self)
        widget.setSizeAdjustPolicy(NoWheelComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        widget.setMinimumContentsLength(10)
        for text, value in choices:
            widget.addItem(text, value)
        self._register(name, widget)
        layout.addRow(label, widget)
        widget.currentIndexChanged.connect(self._update_preview)
        return widget

    def _spin(self, layout, name, label, low, high, *, suffix="", floating=False):
        widget = NoWheelDoubleSpinBox(self) if floating else NoWheelSpinBox(self)
        widget.setRange(low, high)
        if floating:
            widget.setDecimals(1)
            widget.setSingleStep(.1)
        widget.setSuffix(suffix)
        self._register(name, widget)
        if isinstance(layout, QFormLayout):
            layout.addRow(label, widget)
        else:
            form = QFormLayout()
            form.addRow(label, widget)
            layout.addLayout(form)
        widget.valueChanged.connect(self._update_preview)
        return widget

    def _register(self, name, widget):
        self.controls[name] = widget
        widget.setObjectName("trial_" + name)

    @staticmethod
    def _optical_resources(preferences):
        optical = preferences.optical_analysis
        return {"flat_reference": deepcopy(optical.get("flat_reference")),
                "reference_context": deepcopy(optical.get("reference_context", {}))}

    def set_preferences(self, preferences, *, reset_resource_observation=True):
        self._loading = True
        self._draft = preferences.normalized_copy()
        if reset_resource_observation:
            self._observed_optical_resources = self._optical_resources(self._draft)
        for name, widget in self.controls.items():
            value = getattr(self._draft, name)
            if isinstance(widget, QCheckBox):
                widget.setChecked(value)
            elif isinstance(widget, NoWheelComboBox):
                widget.setCurrentIndex(max(0, widget.findData(value)))
            else:
                widget.setValue(value)
        if self._draft.font_family:
            self.font_combo.setCurrentFont(QFont(self._draft.font_family))
        else:
            # QFontComboBox can choose its alphabetically first font when the
            # application uses a platform alias absent from the family list.
            # An inherited preference must not look like an explicit choice.
            self.font_combo.setCurrentIndex(-1)
            self.font_combo.setEditText("")
        optical = self._draft.optical_analysis
        self.imaging_mode.setCurrentIndex(max(0, self.imaging_mode.findData(optical.get("imaging_mode", "mixed"))))
        self.flat_enabled.setChecked(optical.get("flat_enabled", False))
        self._refresh_reference_status()
        self._loading = False
        self._update_preview()

    def preferences(self):
        values = {}
        for name, widget in self.controls.items():
            values[name] = widget.isChecked() if isinstance(widget, QCheckBox) else (
                widget.currentData() if isinstance(widget, NoWheelComboBox) else widget.value())
        optical = dict(self._draft.optical_analysis)
        optical.update(imaging_mode=self.imaging_mode.currentData(), flat_enabled=self.flat_enabled.isChecked())
        return replace(self._draft, **values, optical_analysis=optical).normalized_copy()

    def restore_defaults(self):
        # A later unrelated preference notification must not resurrect the
        # unchanged global reference that the user's draft just cleared.
        self.set_preferences(TrialPreferences(), reset_resource_observation=False)
        self.total_target.setValue(1500)

    def update_optical_resources(self, preferences):
        """Merge asynchronous capture resources without overwriting UI drafts."""
        resources = self._optical_resources(TrialPreferences.from_dict(preferences))
        if resources == self._observed_optical_resources:
            return
        self._observed_optical_resources = resources
        self._draft.optical_analysis = {**self._draft.optical_analysis, **deepcopy(resources)}
        self._refresh_reference_status()

    def _refresh_reference_status(self):
        reference = self._draft.optical_analysis.get("flat_reference")
        self.reference_status.setText("已配置空白平场参考；实际可用性在试验光照检查中确认。" if reference else "尚未采集空白平场参考。")

    def set_cleanup_status(self, text, *, busy=False):
        self.cleanup_status.setText(str(text))
        self.cleanup_button.setEnabled(not busy)

    def _font_changed(self, font):
        if not self._loading:
            self._draft.font_family = font.family()
            self._update_preview()

    def _use_application_font(self):
        self._loading = True
        self.font_combo.setCurrentIndex(-1)
        self.font_combo.setEditText("")
        self._draft.font_family = ""
        self._loading = False
        self._update_preview()

    def _choose_color(self):
        selected = QColorDialog.getColor(QColor(self._draft.font_color), self, "试验标记文字颜色")
        if selected.isValid():
            self._draft.font_color = selected.name().upper()
            self._update_preview()

    def _update_preview(self, *_):
        if self._loading:
            return
        preferences = self.preferences()
        self.color_button.setText(preferences.font_color)
        self.color_button.setStyleSheet(f"border-left: 14px solid {preferences.font_color}; padding: 4px;")
        self.color_button.setEnabled(preferences.color_mode == "custom")
        self.application_font.setToolTip("当前跟随应用字体" if not preferences.font_family else "恢复跟随应用字体")
        self.controls["evidence_format"].setEnabled(preferences.evidence_enabled)
        self.controls["gesture"].setEnabled(preferences.input_mode != "candidate")
        self.controls["manual_algorithm"].setEnabled(preferences.input_mode != "candidate")
        self.preview.set_preferences(preferences)
