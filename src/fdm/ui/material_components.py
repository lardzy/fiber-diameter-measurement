"""Shared presentation for library definitions and trial-specific drafts."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QApplication, QComboBox, QFormLayout, QHeaderView,
    QLabel, QLineEdit, QScrollArea, QSplitter, QStackedWidget, QTableWidget,
    QTableWidgetItem, QTabWidget, QToolButton, QPlainTextEdit, QVBoxLayout, QWidget)
from fdm.ui.widgets import NoWheelComboBox


def number_text(value):
    return "未设置" if value is None else f"{value:g}"


def range_text(material):
    low, high = material.lower_limit, material.upper_limit
    if low is None and high is None:
        return "不限"
    if low is None:
        return f"≤{high:g}"
    if high is None:
        return f"≥{low:g}"
    return f"{low:g}～{high:g}"


def source_text(material):
    source = str(material.provenance.get("parameter_source", material.provenance.get("source", "")))
    return "CU 还原定义" if source.startswith("CU TYPES.LIST") else "用户自定义" if source == "user" else source


def material_summary(material):
    density = number_text(material.density)
    if material.density is None or material.density <= 0:
        density += "（待确认）"
    measured = f"{material.measured_target} 根" if material.measured_target is not None else "随试验默认"
    counted = f"{material.count_target} 根" if material.count_target is not None else "不提醒"
    return (f"密度 {density} g/cm³\n直径范围 {range_text(material)} μm"
            f"\n实测直径提醒 {measured} · 补计／计数提醒 {counted}")


def fit_dialog(dialog, width, height):
    screen = dialog.parentWidget().screen() if dialog.parentWidget() is not None else QApplication.primaryScreen()
    if screen is not None:
        available = screen.availableGeometry()
        width, height = min(width, max(320, available.width() - 48)), min(height, max(300, available.height() - 48))
    dialog.resize(width, height)


class ResponsiveMaterialPanes(QWidget):
    """Keep both content widgets alive when switching between splitter and tabs."""
    def __init__(self, left, right, parent=None, *, labels=("候选类别", "已选顺序"), breakpoint=760):
        super().__init__(parent)
        self.left, self.right, self.labels = left, right, labels
        self.breakpoint, self.narrow = breakpoint, False
        self.stack = QStackedWidget(self)
        self.splitter = QSplitter(Qt.Orientation.Horizontal, self)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.addWidget(left)
        self.splitter.addWidget(right)
        self.splitter.setSizes([580, 420])
        self.tabs = QTabWidget(self)
        self.stack.addWidget(self.splitter)
        self.stack.addWidget(self.tabs)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.stack)
        self.setMinimumSize(0, 0)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        narrow = event.size().width() < self.breakpoint
        if narrow == self.narrow:
            return
        self.narrow = narrow
        if narrow:
            self.tabs.addTab(self.left, self.labels[0])
            self.tabs.addTab(self.right, self.labels[1])
            self.stack.setCurrentWidget(self.tabs)
        else:
            while self.tabs.count():
                self.tabs.removeTab(0)
            self.splitter.addWidget(self.left)
            self.splitter.addWidget(self.right)
            self.splitter.setSizes([580, 420])
            self.stack.setCurrentWidget(self.splitter)

    def show_right(self):
        if self.narrow:
            self.tabs.setCurrentWidget(self.right)


class MaterialParameterEditor(QWidget):
    changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.original = None
        self._loading = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.title = QLabel(self)
        self.title.setWordWrap(True)
        font = self.title.font()
        font.setBold(True)
        self.title.setFont(font)
        layout.addWidget(self.title)
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        panel = QWidget(scroll)
        self.form = QFormLayout(panel)
        self.form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        for key, title in (("name", "名称"), ("aliases", "别名"), ("density", "密度 g/cm³"),
                           ("lower", "直径下限 μm"), ("upper", "直径上限 μm")):
            editor = QLineEdit(panel)
            editor.setMinimumWidth(0)
            editor.setAccessibleName(title)
            editor.textChanged.connect(self._changed)
            self.form.addRow(title, editor)
            setattr(self, key, editor)
        self.aliases.setPlaceholderText("多个别名用逗号分隔")
        self.density.setPlaceholderText("未设置")
        self.lower.setPlaceholderText("不限")
        self.upper.setPlaceholderText("不限")
        for key, title, placeholder in (("measured_target", "实测直径提醒根数", "留空随试验默认"),
                                         ("count_target", "补计／计数提醒根数", "留空不提醒")):
            editor = QLineEdit(panel)
            editor.setMinimumWidth(0)
            editor.setAccessibleName(title)
            editor.setPlaceholderText(placeholder)
            editor.setToolTip("填写正整数；达到时提醒，可以继续。实测和补计分别统计，补计不含实测。")
            editor.textChanged.connect(self._changed)
            self.form.addRow(title, editor)
            setattr(self, key, editor)
        target_note = QLabel("提醒按操作者及经／纬方向分别统计。实测留空沿用试验默认；补计留空不提醒。", panel)
        target_note.setWordWrap(True)
        self.form.addRow(target_note)
        for key, title in (("below", "低于下限时建议"), ("above", "高于上限时建议")):
            editor = NoWheelComboBox(panel)
            editor.setMinimumWidth(0)
            editor.setMinimumContentsLength(6)
            editor.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            editor.currentIndexChanged.connect(self._changed)
            self.form.addRow(title, editor)
            setattr(self, key, editor)
        self.source = QLineEdit(panel)
        self.source.setMinimumWidth(0)
        self.source.textChanged.connect(self._changed)
        self.form.addRow("参数来源", self.source)
        self.source_details = QToolButton(panel)
        self.source_details.setText("查看来源详情")
        self.source_details.setCheckable(True)
        self.form.addRow(self.source_details)
        self.provenance = QPlainTextEdit(panel)
        self.provenance.setReadOnly(True)
        self.provenance.setMaximumHeight(130)
        self.form.addRow(self.provenance)
        self.provenance.hide()
        self.source_details.toggled.connect(self.provenance.setVisible)
        scroll.setWidget(panel)
        layout.addWidget(scroll, 1)
        self.error = QLabel(self)
        self.error.setWordWrap(True)
        layout.addWidget(self.error)

    def _changed(self, *_):
        if not self._loading:
            self.error.clear()
            self.changed.emit()

    def load(self, material, catalog, *, scope):
        self._loading = True
        try:
            self.original = deepcopy(material)
            self.title.setText(f"{scope} · {material.name}")
            self.name.setText(material.name)
            self.aliases.setText("，".join(material.aliases))
            for editor, value in ((self.density, material.density), (self.lower, material.lower_limit),
                                  (self.upper, material.upper_limit),
                                  (self.measured_target, material.measured_target), (self.count_target, material.count_target)):
                editor.setText("" if value is None else str(value))
            for combo, value in ((self.below, material.lower_suggestion_id), (self.above, material.upper_suggestion_id)):
                combo.clear()
                combo.addItem("不建议改类", None)
                for candidate in catalog.materials():
                    if candidate.id != material.id:
                        combo.addItem(candidate.name, candidate.id)
                if value is not None and combo.findData(value) < 0:
                    combo.addItem("类别已不在库中", value)
                combo.setCurrentIndex(max(0, combo.findData(value)))
            self._source_original = source_text(material)
            self.source.setText(self._source_original)
            self.provenance.setPlainText(json.dumps(material.provenance, ensure_ascii=False, indent=2))
            self.source_details.setChecked(False)
            self.error.clear()
        finally:
            self._loading = False

    def material(self):
        if self.original is None:
            raise ValueError("请先选择类别")
        def numeric(editor):
            return float(editor.text().strip()) if editor.text().strip() else None
        def target(editor):
            text = editor.text().strip()
            if not text:
                return None
            try:
                value = int(text)
            except ValueError as error:
                raise ValueError("类别提醒根数必须是正整数或留空") from error
            if value < 1:
                raise ValueError("类别提醒根数必须是正整数或留空")
            return value
        provenance = deepcopy(self.original.provenance)
        if self.source.text().strip() != self._source_original:
            provenance["parameter_source"] = self.source.text().strip()
        return replace(self.original, name=self.name.text(),
            aliases=tuple(self.aliases.text().replace("，", ",").split(",")),
            density=numeric(self.density), lower_limit=numeric(self.lower), upper_limit=numeric(self.upper),
            lower_suggestion_id=self.below.currentData(), upper_suggestion_id=self.above.currentData(),
            provenance=provenance, measured_target=target(self.measured_target), count_target=target(self.count_target))


def material_table(parent=None, *, selectable=False):
    titles = ["名称", "密度 g/cm³", "直径范围 μm", "状态"]
    if selectable:
        titles.insert(0, "选用")
    table = QTableWidget(0, len(titles), parent)
    table.setHorizontalHeaderLabels(titles)
    table.verticalHeader().hide()
    table.verticalHeader().setDefaultSectionSize(34)
    table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
    table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
    table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
    table.setAlternatingRowColors(True)
    table.setShowGrid(False)
    table.setWordWrap(False)
    table.setMinimumWidth(0)
    header = table.horizontalHeader()
    header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
    header.setSectionResizeMode(1 if selectable else 0, QHeaderView.ResizeMode.Stretch)
    if selectable:
        table.setColumnHidden(4, True)  # status remains available in tooltips and the selected list
    return table


def table_item(text, identity, *, numeric=False):
    item = QTableWidgetItem(text)
    item.setData(Qt.ItemDataRole.UserRole, identity)
    if numeric:
        item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    return item
