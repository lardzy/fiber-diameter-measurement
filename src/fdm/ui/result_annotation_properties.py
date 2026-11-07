"""Shared property editor for the presentation of ordinary and professional dimensions."""
from __future__ import annotations

from dataclasses import dataclass, replace

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QFormLayout, QGroupBox, QPushButton, QSizePolicy, QVBoxLayout

from fdm.measurement_annotation import MeasurementAnnotationLayout
from fdm.ui.measurement_batch_properties import MIXED, common_value
from fdm.ui.professional_rendering import dimension_endpoints, presentation_result
from fdm.ui.widgets import NoWheelComboBox


@dataclass(frozen=True)
class AnnotationTarget:
    identity: str
    family: str
    layout: MeasurementAnnotationLayout
    metrics: dict
    dual_metrics: frozenset
    symbol_metrics: frozenset

    @classmethod
    def ordinary(cls, record):
        is_line = record.measurement_kind == "line"
        supported = frozenset({None}) if is_line else frozenset()
        layout = record.annotation_layout or MeasurementAnnotationLayout.ordinary_near()
        return cls(record.id, record.measurement_kind, layout, {}, supported, supported)

    @classmethod
    def professional(cls, document, definition, result):
        layout = definition.annotation_layout or MeasurementAnnotationLayout.legacy_professional()
        metrics = {m.key: f"{m.label} ({m.unit})" for m in result.metrics}
        dual = set()
        symbols = set()
        for key in (None, *metrics):
            candidate = replace(definition, annotation_layout=replace(layout, metric_key=key))
            visual = presentation_result(document, candidate, result)
            ends = dimension_endpoints(visual)
            if result.valid and ends is not None and ends[0] != ends[1]:
                dual.add(key)
            effective_key = key if key is not None else (result.primary.key if result.primary else None)
            if effective_key in {"diameter", "radius"}:
                symbols.add(key)
        return cls(definition.id, "professional:" + definition.kind, layout, metrics, frozenset(dual), frozenset(symbols))


class ResultAnnotationProperties(QGroupBox):
    editRequested = Signal(object, object, str)

    def __init__(self, parent=None):
        super().__init__("结果标注", parent)
        self.setObjectName("resultAnnotationProperties")
        self._target = None
        self._objects = ()
        self._pending = {}
        self._values = {}
        self._fields = {}
        self._updating = False
        root = QVBoxLayout(self)
        self.form = QFormLayout()
        self.form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        root.addLayout(self.form)
        for key, title in (("metric_key", "显示指标"), ("style", "标注样式"), ("show_identifier", "显示测量编号"),
                           ("show_metric_name", "显示指标名称"), ("show_symbol", "显示直径／半径符号")):
            combo = NoWheelComboBox(self)
            combo.setAccessibleName(title)
            combo.setPlaceholderText("多种值")
            combo.setMinimumContentsLength(5)
            combo.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
            combo.currentIndexChanged.connect(lambda index, field=key: self._choose(field, index))
            self.form.addRow(title, combo)
            self._fields[key] = combo
        self.apply_button = QPushButton("应用到所选标注", self)
        self.restore_button = QPushButton(self)
        self.default_button = QPushButton(self)
        for button, action in ((self.apply_button, "apply"), (self.restore_button, "restore"), (self.default_button, "default")):
            button.clicked.connect(lambda checked=False, op=action: self._submit(op))
            root.addWidget(button)
        self.hide()

    @property
    def target(self):
        return self._target

    @property
    def targets(self):
        return self._objects

    @staticmethod
    def _style(obj):
        layout = obj.layout
        if not obj.family.startswith("professional:") and layout.label_position == "near":
            return "near"
        return layout.leader_style if layout.show_leader else "none"

    def set_selection(self, document_id, domain, objects, family_label):
        if not objects:
            self._target = None
            self._objects = ()
            self._pending.clear()
            self.hide()
            return
        target = (document_id, domain, tuple(sorted(obj.identity for obj in objects)))
        values = {}
        for key in self._fields:
            if key == "style":
                values[key] = common_value(self._style(obj) for obj in objects)
            elif key == "show_symbol":
                values[key] = common_value((obj.layout.show_symbol if obj.layout.show_symbol is not None
                                            else domain == "professional") for obj in objects)
            else:
                values[key] = common_value(getattr(obj.layout, key) for obj in objects)
        if target == self._target and self._objects == objects:
            return
        self._pending = {k: v for k, v in self._pending.items()
                         if target == self._target and values[k] == self._values.get(k)}
        self._target, self._objects, self._values = target, objects, values
        self._family_label = family_label
        self.restore_button.setText("恢复自动位置" if domain == "professional" else "恢复贴近位置")
        self._refresh()
        self.show()

    def _refresh(self):
        if not self._objects:
            return
        objects = self._objects
        professional = self._target[1] == "professional"
        metrics = [(k, label) for k, label in objects[0].metrics.items()
                   if all(k in obj.metrics for obj in objects)]
        effective_metrics = [self._pending.get("metric_key", obj.layout.metric_key) for obj in objects]
        dual = all(key in obj.dual_metrics for key, obj in zip(effective_metrics, objects))
        symbol = all(key in obj.symbol_metrics for key, obj in zip(effective_metrics, objects))
        styles = [] if professional else [("near", "贴近测量几何")]
        styles += [("none", "外置文字（无引线）"), ("single", "单引出")]
        if dual:
            styles.append(("dual", "双端尺寸线"))
        boolean = [(False, "关闭"), (True, "开启")]
        choices = {"style": styles, "show_identifier": boolean, "show_metric_name": boolean}
        if symbol:
            choices["show_symbol"] = boolean
        if professional and metrics:
            choices["metric_key"] = [(None, "主指标"), *metrics]
        self._updating = True
        try:
            for key, combo in self._fields.items():
                self.form.setRowVisible(combo, key in choices)
                if key not in choices:
                    self._pending.pop(key, None)
                    continue
                options = choices[key]
                value = self._pending.get(key, self._values[key])
                if key == "style" and not dual and value == "dual":
                    self._pending.pop(key, None)
                    value = "single"
                combo.clear()
                index = -1
                for i, (data, label) in enumerate(options):
                    combo.addItem(label, data)
                    if data == value:
                        index = i
                combo.setCurrentIndex(index)
        finally:
            self._updating = False
        self.apply_button.setEnabled(bool(self._pending))
        same_family = len({obj.family for obj in objects}) == 1
        clear_values = all(self._pending.get(k, self._values[k]) is not MIXED for k in choices)
        self.default_button.setEnabled(same_family and clear_values)
        self.default_button.setText(f"设为{self._family_label}默认" if same_family else "设为同类默认")
        self.default_button.setToolTip("应用到所选，并用于之后新建的同类标注。"
                                      if same_family and clear_values else "选择同类对象，并先统一有多种值的标注字段。")

    def _choose(self, key, index):
        if self._updating or self._target is None or index < 0:
            return
        value = self._fields[key].itemData(index)
        if value == self._values[key]:
            self._pending.pop(key, None)
        else:
            self._pending[key] = value
        self._refresh()

    def _submit(self, action):
        if self._target is None:
            return
        patch = dict(self._pending)
        if "style" in patch:
            style = patch.pop("style")
            if self._target[1] == "ordinary":
                patch["label_position"] = "near" if style == "near" else "external"
            patch["show_leader"] = style in {"single", "dual"}
            if patch["show_leader"]:
                patch["leader_style"] = style
        self.editRequested.emit(self._target, patch, action)
