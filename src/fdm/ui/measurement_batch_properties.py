"""Edit only the requested properties of a selection of measurement records."""
from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QColorDialog, QFontDialog, QFormLayout, QHBoxLayout, QInputDialog, QLabel,
    QPushButton, QSizePolicy, QVBoxLayout, QWidget,
)

from fdm.models import UNCATEGORIZED_LABEL
from fdm.ui.widgets import NoWheelComboBox


STYLE_FIELDS = {
    "stroke_color": "线条／点颜色", "stroke_width": "线条宽度", "marker_scale": "计数点尺寸",
    "text_color": "文字颜色", "font_family": "字体", "font_size": "字号",
}
MIXED = object()


def common_value(values):
    first = next(values)
    return first if all(value == first for value in values) else MIXED


def common_style_fields(records):
    fields = set(STYLE_FIELDS)
    if any(record.measurement_kind == "count" for record in records):
        fields.discard("stroke_width")
    if any(record.measurement_kind != "count" for record in records):
        fields.discard("marker_scale")
    return fields


def inherited_measurement_style(measurement, settings, group_color):
    count = measurement.measurement_kind == "count"
    label_style = (settings.area_measurement_label_style if measurement.measurement_kind == "area"
                   else settings.length_measurement_label_style)
    return {
        "stroke_color": group_color or settings.default_measurement_color,
        "stroke_width": 2.0, "marker_scale": 1.0,
        "text_color": settings.count_number_color if count else label_style.color,
        "font_family": settings.count_number_font_family if count else label_style.font_family,
        "font_size": settings.count_number_font_size if count else label_style.font_size,
    }


def populate_group_choices(combo, document, value):
    combo.clear()
    combo.addItem(UNCATEGORIZED_LABEL, None)
    for group in document.sorted_groups():
        combo.addItem(group.display_name(), group.id)
    combo.setPlaceholderText("多种值")
    combo.setCurrentIndex(-1 if value is MIXED else combo.findData(value))
    combo.setToolTip(combo.currentText() if combo.currentIndex() >= 0 else "多种值")


class MeasurementBatchProperties(QWidget):
    editRequested = Signal(str, object, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.target = None
        self._values = {}
        self._inherited = {}
        self._patch = {}
        self._buttons = {}
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        self._summary = QLabel(self)
        root.addWidget(self._summary)
        form = QFormLayout()
        root.addLayout(form)
        self._group = NoWheelComboBox(self)
        self._group.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self._group.activated.connect(self._choose_group)
        form.addRow("所属类别", self._group)
        self._form = form
        for field, label in STYLE_FIELDS.items():
            button = QPushButton(self)
            button.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
            button.setAccessibleName("批量编辑" + label)
            button.clicked.connect(lambda checked=False, key=field: self._choose_style(key))
            form.addRow(label, button)
            self._buttons[field] = button
        buttons = QHBoxLayout()
        self._inherit = QPushButton("恢复继承样式", self)
        self._inherit.setToolTip("将共同样式字段恢复为继承；点击应用后生效")
        self._inherit.clicked.connect(self._inherit_styles)
        buttons.addWidget(self._inherit)
        self._apply = QPushButton("应用到所选", self)
        self._apply.clicked.connect(self._submit)
        buttons.addWidget(self._apply)
        root.addLayout(buttons)
        root.addStretch(1)

    def clear_selection(self):
        self.target = None
        self._patch.clear()

    def set_selection(self, document, records, settings):
        ids = tuple(sorted(record.id for record in records))
        target = (document.id, ids)
        fields = common_style_fields(records)
        values = {key: common_value(getattr(record.appearance, key, None) for record in records)
                  for key in fields}
        values["fiber_group_id"] = common_value(record.fiber_group_id for record in records)
        self._group_colors = {group.id: group.color for group in document.fiber_groups}
        self._group_colors[None] = settings.default_measurement_color
        styles = [inherited_measurement_style(m, settings, self._group_colors.get(m.fiber_group_id)) for m in records]
        self._inherited = {field: common_value(s[field] for s in styles) for field in fields}
        # Refreshes with the same selection must not discard an uncommitted edit.
        # A changed selection, undo or concurrent edit abandons the old draft.
        if self.target != target:
            self._patch.clear()
        else:
            self._patch = {k: v for k, v in self._patch.items()
                           if k in values and self._values.get(k) == values[k]}
        self.target, self._values = target, values
        self._summary.setText(f"已选择 {len(records)} 个测量对象")
        populate_group_choices(self._group, document, self._patch.get("fiber_group_id", values["fiber_group_id"]))
        for field, button in self._buttons.items():
            self._form.setRowVisible(button, field in fields)
        self._refresh_values()

    def _refresh_values(self):
        for field, button in self._buttons.items():
            if field not in self._values:
                continue
            value = self._patch.get(field, self._values[field])
            if value is MIXED:
                display = "多种值"
            elif value is None:
                inherited = self._inherited_value(field)
                display = "继承（多种值）" if inherited is MIXED else f"继承（{inherited}）"
            else:
                display = str(value)
            button.setText(display)
            button.setToolTip(STYLE_FIELDS[field] + "：" + display)
        self._apply.setEnabled(bool(self._patch))
        self._inherit.setEnabled(any(self._patch.get(f, v) is not None
                                     for f, v in self._values.items() if f != "fiber_group_id"))

    def _inherited_value(self, field):
        if field == "stroke_color" and "fiber_group_id" in self._patch:
            return self._group_colors[self._patch["fiber_group_id"]]
        return self._inherited[field]

    def _set_field(self, field, value):
        if self.target is None or field not in self._values:
            return
        if value == self._values[field]:
            self._patch.pop(field, None)
        else:
            self._patch[field] = value
        self._refresh_values()

    def _choose_group(self, index):
        if index >= 0:
            self._set_field("fiber_group_id", self._group.itemData(index))

    def _choose_style(self, field):
        if self.target is None or field not in self._values:
            return
        target = self.target
        current = self._patch.get(field, self._values[field])
        if current is None:
            current = self._inherited_value(field)
        if current is MIXED:
            current = None
        label = STYLE_FIELDS[field]
        if field in {"stroke_color", "text_color"}:
            color = QColorDialog.getColor(QColor(current or "#ffffff"), self, label)
            value, accepted = color.name().upper(), color.isValid()
        elif field == "font_family":
            accepted, font = QFontDialog.getFont(QFont(current or ""), self, label)
            value = font.family()
        elif field == "font_size":
            value, accepted = QInputDialog.getInt(self, label, label, current or 16, 8, 144)
        else:
            low, high, default, decimals = (0.5, 24.0, 2.0, 1) if field == "stroke_width" else (0.25, 4.0, 1.0, 2)
            value, accepted = QInputDialog.getDouble(self, label, label, current or default, low, high, decimals)
        if accepted and self.target == target:
            self._set_field(field, value)

    def _inherit_styles(self):
        for field in self._values.keys() & STYLE_FIELDS.keys():
            self._set_field(field, None)

    def _submit(self):
        if self.target is not None and self._patch:
            self.editRequested.emit(*self.target, dict(self._patch))
