"""Professional measurement workspace and document/canvas coordinator."""
from __future__ import annotations

from dataclasses import replace
import csv
import json
import math
from pathlib import Path
from threading import Event, Thread

import numpy as np
from PySide6.QtCore import QEvent, QObject, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFontMetricsF
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDockWidget, QDoubleSpinBox,
    QFileDialog, QFormLayout, QHBoxLayout, QLabel, QMessageBox, QPushButton,
    QScrollArea, QSpinBox, QTableWidget, QTableWidgetItem, QTextEdit, QVBoxLayout, QWidget)

from fdm.construction_document import make_construction_resolver
from fdm.construction_geometry import ResolvedCircle, ResolvedLine, ResolvedPoint
from fdm.geometry import Point
from fdm.history import DocumentChangeImpact
from fdm.measurement_annotation import MeasurementAnnotationLayout
from fdm.metrology import MeasurementDefinition, MeasurementFeature
from fdm.models import new_id
from fdm.services.raster_io import raster_plane_to_numpy, qimage_to_raster_plane
from fdm.ui.measurement_annotation_layout import choose_automatic_offset
from fdm.ui.professional_rendering import result_caption, presentation_result, dimension_endpoints


# Each slot is either a live reference selected explicitly or a sequence of
# image-space points. -1 means an explicit Enter/Finish terminates the sequence.
TOOLS = {
    "distance": ("两点距离 / X、Y 分量", (("line", 2),)),
    "point_line_distance": ("点到直线垂距", (("point", 1), ("line", 2))),
    "point_segment_distance": ("点到线段最短距离", (("point", 1), ("line", 2))),
    "parallel_distance": ("平行线间距", (("line", 2), ("line", 2))),
    "projection": ("沿基准方向投影", (("line", 2), ("line", 2))),
    "angle": ("三点角（中点为顶点）", (("polyline", 3),)),
    "line_angle": ("两直线夹角", (("line", 2), ("line", 2))),
    "circle": ("三点圆：半径 / 直径 / 圆心", (("circle", 3),)),
    "arc": ("圆弧：起点 → 经过点 → 终点", (("arc", 3),)),
    "circle_center_distance": ("两圆中心距", (("circle", 3), ("circle", 3))),
    "circle_gap": ("两圆边界最短距离", (("circle", 3), ("circle", 3))),
    "point_circle_distance": ("点到圆边界", (("point", 1), ("circle", 3))),
    "polyline": ("折线长度", (("polyline", -1),)),
    "polygon": ("多边形面积 / 周长", (("polygon", -1),)),
    "coordinates": ("相对坐标：目标 / 原点 / 方向", (("point", 1), ("point", 1), ("line", 2))),
    "fit_line": ("手动多点拟合直线", (("fit_line", -1),)),
    "fit_circle": ("手动多点拟合圆", (("fit_circle", -1),)),
    "caliper_line": ("直线卡尺：带状找边", (("line", 2),)),
    "caliper_circle": ("圆形卡尺：圆心 → 半径", (("line", 2),)),
    "caliper_width": ("双边卡尺：多截面宽度 / 厚度", (("line", 2),)),
}


def calibration_scale(document):
    c = document.calibration
    return (c.pixel_size_x, c.pixel_size_y) if c else (1., 1.)


def calibration_snapshot(document):
    """Detach the full unit/axis/source contract captured by a fitting job."""
    return document.calibration.to_dict() if document.calibration else None


def refit_target_snapshot(document, identity):
    """Compare measurement semantics while allowing label/selection edits."""
    definition = document.get_professional_measurement(identity)
    if definition is None:
        return None
    features = [document.get_professional_feature(fid) for fid in definition.feature_ids]
    if any(feature is None for feature in features):
        return None
    payload = definition.to_dict()
    for field in ("annotation_layout", "name", "created_at"):
        payload.pop(field, None)
    return {"definition": payload, "features": [feature.to_dict() for feature in features]}


def source_identity(canvas):
    """Keep focus/registration identity distinct from transient view generation."""
    document = canvas._document
    manifest = getattr(canvas, "_slide_manifest", None)
    if manifest is None:
        return json.dumps({"image": document.id}, sort_keys=True)
    # Manifest metadata includes registration/blending revision when present.
    return json.dumps({"image": document.id, "focus": canvas.focus_index(),
                       "manifest": manifest.to_dict(),
                       "source_revision": document.professional_source_revision,
                       "source_pending": document.professional_source_revision_pending}, sort_keys=True, default=str)


class ProfessionalPanel(QWidget):
    def __init__(self, owner):
        super().__init__(owner.window)
        self.owner = owner
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        self.tool = QComboBox()
        for key, (label, _) in TOOLS.items():
            self.tool.addItem(label, key)
        layout.addWidget(self.tool)
        self.calibration_info = QLabel()
        self.calibration_info.setWordWrap(True)
        layout.addWidget(self.calibration_info)
        self.form = QFormLayout()
        self.sources = []
        for index in range(3):
            box = QComboBox()
            box.setMinimumContentsLength(12)
            box.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            self.sources.append(box)
            self.form.addRow(f"来源 {index + 1}", box)
        layout.addLayout(self.form)
        self.supplement = QCheckBox("使用补角（180° − 当前角）")
        layout.addWidget(self.supplement)
        self.caliper_widget = QWidget()
        options = QFormLayout(self.caliper_widget)
        options.setContentsMargins(0, 0, 0, 0)
        self.method = QComboBox()
        self.method.addItem("梯度边缘", "gradient")
        self.method.addItem("阈值交点", "threshold")
        self.polarity = QComboBox()
        for title, key in (("任意明暗", "either"), ("暗 → 明", "rising"), ("明 → 暗", "falling")):
            self.polarity.addItem(title, key)
        self.reverse = QCheckBox("反向搜索")
        self.width = QDoubleSpinBox(); self.width.setRange(1, 1024); self.width.setValue(10); self.width.setSuffix(" px")
        self.count = QSpinBox(); self.count.setRange(3, 1024); self.count.setValue(32)
        self.threshold = QDoubleSpinBox(); self.threshold.setRange(.1, 99.9); self.threshold.setValue(50); self.threshold.setSuffix(" %")
        self.step = QDoubleSpinBox(); self.step.setRange(.125, 4); self.step.setDecimals(3); self.step.setValue(.5); self.step.setSuffix(" px")
        self.contrast = QDoubleSpinBox(); self.contrast.setRange(0, 100); self.contrast.setValue(1.57); self.contrast.setSuffix(" %")
        self.fraction = QDoubleSpinBox(); self.fraction.setRange(1, 100); self.fraction.setValue(50); self.fraction.setSuffix(" %")
        self.robust = QCheckBox("稳健剔除异常点"); self.robust.setChecked(True)
        for name, widget in (("找边方式", self.method), ("边缘极性", self.polarity), ("搜索方向", self.reverse),
                             ("搜索半宽", self.width), ("采样截面", self.count), ("采样间隔", self.step),
                             ("百分比阈值", self.threshold), ("最小对比", self.contrast), ("最低有效比例", self.fraction), ("拟合", self.robust)):
            options.addRow(name, widget)
        layout.addWidget(self.caliper_widget)
        row = QHBoxLayout()
        for text, callback in (("开始取点", owner.begin), ("完成 / Enter", owner.finish), ("取消 / Esc", owner.cancel)):
            button = QPushButton(text); button.clicked.connect(callback); row.addWidget(button)
        layout.addLayout(row)
        self.prompt = QLabel("选择工具，指定辅助来源或在画布取点。坐标为图像 / 标定平面。")
        self.prompt.setWordWrap(True); layout.addWidget(self.prompt)
        self.confirm = QPushButton("确认拟合并保存测量")
        self.confirm.setEnabled(False); self.confirm.clicked.connect(owner.confirm_fit); layout.addWidget(self.confirm)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["编号", "指标 / 状态", "数值", "单位"])
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().hide()
        self.table.setMinimumHeight(140)
        self.table.itemSelectionChanged.connect(owner.table_selection)
        self.table.setMouseTracking(True)
        self.table.itemEntered.connect(lambda item: owner.hover(item.data(Qt.ItemDataRole.UserRole)))
        self.table.viewport().installEventFilter(owner)
        layout.addWidget(self.table, 1)
        self.details = QTextEdit(); self.details.setReadOnly(True); self.details.setMaximumHeight(160)
        layout.addWidget(self.details)
        row = QHBoxLayout()
        for text, callback in (("删除", owner.delete_selected), ("解除关联并冻结", owner.freeze_selected), ("重新找边", owner.refit_selected)):
            button = QPushButton(text); button.clicked.connect(callback); row.addWidget(button)
        layout.addLayout(row)
        row = QHBoxLayout()
        for text, callback in (("整理标注", owner.arrange), ("导出明细", owner.export)):
            button = QPushButton(text); button.clicked.connect(callback); row.addWidget(button)
        layout.addLayout(row)
        self.tool.currentIndexChanged.connect(self.configure)
        for box in self.sources:
            box.currentIndexChanged.connect(owner.parameters_changed)
        for control in (self.method, self.polarity):
            control.currentIndexChanged.connect(owner.parameters_changed)
        for control in (self.width, self.count, self.threshold, self.step, self.contrast, self.fraction):
            control.valueChanged.connect(owner.parameters_changed)
        for control in (self.reverse, self.robust, self.supplement):
            control.toggled.connect(owner.parameters_changed)
        self.configure()

    def configure(self, *_):
        self.owner.cancel()
        key = self.tool.currentData()
        self.caliper_widget.setVisible(key.startswith("caliper_"))
        self.confirm.setVisible(key.startswith("caliper_"))
        self.supplement.setVisible(key in {"angle", "line_angle"})
        self.populate_sources()

    def populate_sources(self):
        key = self.tool.currentData()
        if not key:
            return
        slots = TOOLS[key][1]
        document = self.owner.window.current_document()
        resolver = make_construction_resolver(document) if document else None
        for index, box in enumerate(self.sources):
            previous = box.currentData()
            box.blockSignals(True); box.clear(); box.addItem("在画布手动取点", None)
            visible = index < len(slots) and not key.startswith("caliper_")
            self.form.setRowVisible(box, visible)
            if visible and document is not None:
                expected = slots[index][0]
                for entity in document.construction_entities:
                    result = resolver.resolve(entity)
                    kind = {ResolvedPoint: "point", ResolvedLine: "line", ResolvedCircle: "circle"}.get(type(result.geometry))
                    if kind == expected:
                        box.addItem(f"辅助 · {entity.name or entity.id}", ("construction", entity.id))
                for feature in document.professional_features:
                    compatible = feature.kind == expected or (expected == "line" and feature.kind in {"fit_line", "caliper_line"}) or (expected == "circle" and feature.kind in {"fit_circle", "caliper_circle"})
                    if compatible:
                        box.addItem(f"专业特征 · {feature.id[-8:]}", ("feature", feature.id))
                for measurement in document.measurements:
                    if expected == "line" and measurement.measurement_kind == "line":
                        box.addItem(f"直径线 · {measurement.id[-8:]}", ("measurement", measurement.id))
            for i in range(box.count()):
                if box.itemData(i) == previous:
                    box.setCurrentIndex(i); break
            box.blockSignals(False)

    def options(self):
        from fdm.services.metrology_fitting import CaliperOptions
        return CaliperOptions(method=self.method.currentData(), polarity=self.polarity.currentData(),
            reverse_search=self.reverse.isChecked(), search_half_width_px=self.width.value(),
            sample_count=self.count.value(), sample_step_px=self.step.value(), threshold_percent=self.threshold.value(),
            min_contrast=self.contrast.value() / 100, min_valid_fraction=self.fraction.value() / 100,
            robust=self.robust.isChecked())


class ProfessionalWorkspace(QObject):
    fitFinished = Signal(object, object)

    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.panel = None
        self.active = None
        self._job = None
        self._preview = None
        self._refreshing = False
        self._bound_document = None
        self._workers = []
        self._legacy_viewports = []
        self.fitFinished.connect(self._fit_finished)
        self.panel = ProfessionalPanel(self)
        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(self.panel)
        self.dock = QDockWidget("专业测量", window)
        self.dock.setObjectName("professionalMeasurementDock")
        self.dock.setWidget(scroll)
        window.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.dock)
        self.dock.hide()
        for pane in (getattr(window, "_inspector_records_pane", None), getattr(window, "_bottom_records_pane", None)):
            if pane is not None:
                pane.table.setMouseTracking(True)
                pane.table.entered.connect(self.legacy_table_hover)
                pane.table.viewport().installEventFilter(self)
                self._legacy_viewports.append(pane.table.viewport())

    def show(self):
        inspector = getattr(self.window, "_inspector_dock", None)
        if inspector is not None:
            self.window.tabifyDockWidget(inspector, self.dock)
        self.dock.show(); self.dock.raise_(); self.refresh()

    def hover(self, identity):
        controller = self.controller()
        if controller is not None:
            controller.hovered_id = identity
            controller.canvas.update()
        if self.panel is not None:
            table = self.panel.table
            for row in range(table.rowCount()):
                active = table.item(row, 0).data(Qt.ItemDataRole.UserRole) == identity
                color = QColor(40, 180, 190, 70) if active else QColor(Qt.GlobalColor.transparent)
                for col in range(table.columnCount()):
                    table.item(row, col).setBackground(color)

    def eventFilter(self, obj, event):
        if self.panel is not None and obj is self.panel.table.viewport() and event.type() == QEvent.Type.Leave:
            self.hover(None)
        if obj in self._legacy_viewports and event.type() == QEvent.Type.Leave:
            document = self.window.current_document()
            if document is not None:
                self.legacy_hover(document.id, None)
        return super().eventFilter(obj, event)

    def legacy_table_hover(self, index):
        document = self.window.current_document()
        if document is not None:
            identity = self.window._records_controller.measurement_id_for_index(index)
            self.legacy_hover(document.id, identity)

    def legacy_hover(self, document_id, identity):
        document = self.window.current_document()
        if document is None or document.id != document_id:
            return
        self.window._records_controller.source_model.set_hovered_measurement_id(identity)
        canvas = self.window._canvases.get(document_id)
        if canvas is not None and canvas._hovered_measurement_annotation_id != identity:
            canvas._hovered_measurement_annotation_id = identity
            canvas.update()

    def attach(self, canvas):
        from fdm.ui.professional_canvas import ProfessionalCanvasController
        canvas._professional_controller = ProfessionalCanvasController(canvas, self)
        canvas.measurementAnnotationEdited.connect(self.legacy_layout_changed)
        canvas.measurementAnnotationsEdited.connect(self.legacy_layouts_changed)
        canvas.measurementAnnotationPropertiesRequested.connect(self.legacy_properties)
        canvas.measurementAnnotationHoverChanged.connect(self.legacy_hover)

    def controller(self):
        return getattr(self.window.current_canvas(), "_professional_controller", None)

    def message(self, text):
        if self.panel is not None:
            self.panel.prompt.setText(text)
        self.window.statusBar().showMessage(text, 6000)

    def parameters_changed(self, *_):
        if self._job is not None or self._preview is not None or self.active is not None:
            self.cancel()
            self.message("参数已改变；请重新开始取点 / 找边，旧结果不会应用。")

    def begin(self):
        controller = self.controller()
        if controller is None or controller.canvas._read_only:
            self.message("请先打开图像；数字切片需放大至测量视场并等待加载完成。")
            return
        from fdm.services.metrology_service import professional_source_unavailable_reason
        reason = professional_source_unavailable_reason(controller.canvas._document)
        if reason:
            self.message(reason)
            return
        self.cancel()
        self.window.set_tool_mode("professional")
        key = self.panel.tool.currentData()
        selections = [box.currentData() for box in self.panel.sources[:len(TOOLS[key][1])]]
        self.active = controller
        controller.begin(key, selections)
        controller.canvas.setFocus()

    def finish(self):
        if self.active:
            self.active.finish()

    def cancel(self):
        if self._job:
            self._job["cancel"].set()
            self._job = None
        controller = self.active or (self.controller() if self.panel is not None else None)
        if controller:
            controller.cancel()
        self.active = None
        self._preview = None
        if self.panel is not None:
            self.panel.confirm.setEnabled(False)

    def refresh(self):
        if self.panel is None or self._refreshing:
            return
        self._refreshing = True
        try:
            document = self.window.current_document()
            if document is not self._bound_document:
                self.cancel()
                self._bound_document = document
            if document is None:
                self.panel.calibration_info.setText("请打开图像或数字切片。")
            elif document.calibration is None:
                self.panel.calibration_info.setText("未标定：长度 px，面积 px²；角度基于像素平面。")
            else:
                c = document.calibration
                self.panel.calibration_info.setText(f"标定：X {c.pixel_size_x:.6g} / Y {c.pixel_size_y:.6g} {c.unit}/px")
            self.panel.populate_sources()
            table = self.panel.table; table.blockSignals(True); table.setRowCount(0)
            if document is not None:
                controller = self.controller()
                results = controller.results() if controller else {}
                for i, definition in enumerate(document.professional_measurements, 1):
                    result = results.get(definition.id)
                    if result is None:
                        continue
                    metrics = result.metrics if result.valid else (None,)
                    for metric in metrics:
                        row = table.rowCount(); table.insertRow(row)
                        values = [f"P{i}", metric.label if metric else result.reason,
                                  f"{metric.value:.8g}" if metric else "无效", metric.unit if metric else ""]
                        for column, value in enumerate(values):
                            item = QTableWidgetItem(value); item.setData(Qt.ItemDataRole.UserRole, definition.id)
                            table.setItem(row, column, item)
                        if definition.id == document.selected_professional_id:
                            table.selectRow(row)
                table.resizeColumnsToContents()
                selected = document.get_professional_measurement(document.selected_professional_id)
                if self._preview is None and self._job is None:
                    if selected is not None and selected.id in results:
                        self.panel.details.setPlainText(measurement_details_text(document, selected, results[selected.id]))
                    else:
                        self.panel.details.clear()
                if controller:
                    controller.canvas.update()
            else:
                self.panel.details.clear()
            table.blockSignals(False)
        finally:
            self._refreshing = False

    def table_selection(self):
        selected = self.panel.table.selectedItems()
        if not selected:
            return
        self.select(selected[0].data(Qt.ItemDataRole.UserRole), navigate=True)

    def select(self, identity, *, navigate=False):
        document = self.window.current_document(); controller = self.controller()
        if document is None or controller is None:
            return
        definition = document.get_professional_measurement(identity)
        if definition is None:
            return
        document.select_professional(identity)
        if navigate and definition.focus_index is not None and hasattr(controller.canvas, "set_focus_index"):
            controller.canvas.set_focus_index(definition.focus_index)
        result = controller.results().get(identity)
        if result:
            self.panel.details.setPlainText(measurement_details_text(document, definition, result))
        controller.canvas.update()
        self.refresh()

    def change(self, document, label, mutator):
        self.window._apply_document_change(document, label=label,
                                            mutator=mutator, impact=DocumentChangeImpact.SESSION)
        self.refresh()

    def commit(self, controller, key, features, *, parameters=None, replace_id=None):
        document = controller.canvas._document
        from fdm.services.metrology_service import professional_source_unavailable_reason
        reason = professional_source_unavailable_reason(document)
        if reason:
            self.cancel(); self.message(reason)
            return
        known_features = {f.id for f in document.professional_features}
        features = [replace(f, source_revision=slide_revision(document))
                    if f.id not in known_features and f.source_revision is None else f for f in features]
        definition = MeasurementDefinition(new_id("professional"), document.id, key,
            tuple(f.id for f in features), parameters=parameters or {}, focus_index=controller.focus())
        from fdm.services.metrology_service import evaluate_measurement
        combined = list(document.professional_features)
        known = {f.id for f in combined}
        combined += [f for f in features if f.id not in known]
        # Evaluate on a lightweight temporary view; no document mutations while
        # selecting a label position or validating a draft.
        import copy
        scratch = copy.copy(document); scratch.professional_features = combined
        result = evaluate_measurement(scratch, definition)
        if result.valid:
            zoom = max(controller.canvas._zoom, .001)
            fm = QFontMetricsF(controller.canvas.font())
            caption = result_caption(definition, result, len(document.professional_measurements) + 1,
                                     decimals=controller.canvas._settings.professional_measurement_label_decimals)
            occupied = [QRectF(controller.canvas.widget_to_image(label.text_rect.topLeft()).x,
                               controller.canvas.widget_to_image(label.text_rect.topLeft()).y,
                               label.text_rect.width() / zoom, label.text_rect.height() / zoom)
                        for label in controller.label_hits.values()]
            offset = choose_automatic_offset(result.anchor_px, (fm.horizontalAdvance(caption) + 14) / zoom,
                (fm.height() + 8) / zoom, occupied=occupied,
                image_rect=QRectF(0, 0, *document.image_size), gap=18 / zoom)
            definition = replace(definition, annotation_layout=MeasurementAnnotationLayout(offset_px=offset,
                leader_style="dual" if dimension_endpoints(result) is not None else "single"))
        if replace_id:
            original = document.get_professional_measurement(replace_id)
            if original is None:
                self.cancel(); self.message("重新找边的测量已删除，已丢弃过期拟合。")
                return
            definition = replace(definition, id=original.id, name=original.name,
                                 annotation_layout=original.annotation_layout, created_at=original.created_at)
        def mutate():
            measurements = [definition if d.id == replace_id else d for d in document.professional_measurements]
            if not replace_id:
                measurements.append(definition)
            used = {fid for d in measurements for fid in d.feature_ids}
            document.apply_professional_state([f for f in combined if f.id in used], measurements,
                                               selected_id=definition.id, mark_dirty=False)
        self.change(document, "新增专业测量" if not replace_id else "重新拟合专业测量", mutate)
        controller.cancel(); self.active = None
        self.select(definition.id)
        self.message("已保存专业测量。" if result.valid else "测量无效，已保留定义以便复核：" + result.reason)

    def edit_layout(self, controller, identity, layout):
        document = controller.canvas._document
        if layout.placement_mode == "automatic" and layout.offset_px is None:
            definition = document.get_professional_measurement(identity)
            result = presentation_result(document, definition, controller.results()[identity])
            zoom = max(controller.canvas._zoom, .001)
            fm = QFontMetricsF(controller.canvas.font())
            text = result_caption(definition, result, document.professional_measurements.index(definition) + 1,
                                  decimals=controller.canvas._settings.professional_measurement_label_decimals)
            occupied = [QRectF(controller.canvas.widget_to_image(h.text_rect.topLeft()).x,
                               controller.canvas.widget_to_image(h.text_rect.topLeft()).y,
                               h.text_rect.width() / zoom, h.text_rect.height() / zoom)
                        for key, h in controller.label_hits.items() if key != identity]
            offset = choose_automatic_offset(result.anchor_px, (fm.horizontalAdvance(text) + 14) / zoom,
                (fm.height() + 8) / zoom, occupied=occupied, image_rect=QRectF(0, 0, *document.image_size), gap=18 / zoom)
            layout = replace(layout, offset_px=offset)
        def mutate():
            rows = [replace(d, annotation_layout=layout) if d.id == identity else d for d in document.professional_measurements]
            document.apply_professional_state(document.professional_features, rows, selected_id=identity, mark_dirty=False)
        self.change(document, "移动专业测量标注", mutate)

    def edit_parameters(self, controller, identity, parameters):
        document = controller.canvas._document
        self.change(document, "修改专业测量定义", lambda: document.apply_professional_state(
            document.professional_features,
            [replace(d, parameters=parameters) if d.id == identity else d for d in document.professional_measurements],
            selected_id=identity, mark_dirty=False))

    def legacy_layout_changed(self, document_id, identity, layout):
        document = self.window.project.get_document(document_id)
        if document is None:
            return
        measurement = document.get_measurement(identity)
        if measurement:
            canvas = self.window._canvases.get(document_id)
            if canvas is not None:
                layout = canvas.resolved_measurement_annotation_layout(measurement, layout)
            self.change(document, "调整测量标注", lambda: setattr(measurement, "annotation_layout", layout))

    def legacy_layouts_changed(self, document_id, layouts):
        document = self.window.project.get_document(document_id)
        if document is not None:
            def mutate():
                for identity, layout in layouts.items():
                    measurement = document.get_measurement(identity)
                    if measurement:
                        measurement.annotation_layout = layout
            self.change(document, "整理测量标注", mutate)

    def legacy_properties(self, document_id, identity):
        document = self.window.project.get_document(document_id)
        measurement = document.get_measurement(identity) if document else None
        if measurement:
            from PySide6.QtWidgets import QDialog, QDialogButtonBox
            dialog = QDialog(self.window); dialog.setWindowTitle("标注属性")
            layout = QVBoxLayout(dialog)
            leader = QCheckBox("显示虚线引出"); number = QCheckBox("显示测量编号")
            current = measurement.annotation_layout or MeasurementAnnotationLayout()
            leader.setChecked(current.show_leader); number.setChecked(current.show_identifier)
            layout.addWidget(leader); layout.addWidget(number)
            leader.setText("显示尺寸界线 / 引出线")
            style = QComboBox()
            style.addItem("单引出：连接结果与测量位置", "single")
            style.addItem("双端引出：两端尺寸界线 + 尺寸线", "dual")
            style.setCurrentIndex(style.findData(current.leader_style))
            style.setEnabled(measurement.measurement_kind == "line")
            layout.addWidget(style)
            buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
            buttons.accepted.connect(dialog.accept); buttons.rejected.connect(dialog.reject); layout.addWidget(buttons)
            if dialog.exec():
                self.legacy_layout_changed(document_id, identity, replace(current, show_leader=leader.isChecked(), show_identifier=number.isChecked(), leader_style=style.currentData()))

    def selected(self):
        document = self.window.current_document()
        return (document, document.get_professional_measurement(document.selected_professional_id)) if document else (None, None)

    def delete_selected(self):
        document, definition = self.selected()
        if definition:
            def mutate():
                rows = [d for d in document.professional_measurements if d.id != definition.id]
                used = {f for d in rows for f in d.feature_ids}
                document.apply_professional_state([f for f in document.professional_features if f.id in used], rows, mark_dirty=False)
            self.change(document, "删除专业测量", mutate)

    def freeze_selected(self):
        document, definition = self.selected()
        if not definition:
            return
        from fdm.services.metrology_service import freeze_feature
        try:
            # Detach only this definition; other dimensions keep live datums.
            replacements = {fid: replace(freeze_feature(document, document.get_professional_feature(fid)), id=new_id("feature")) for fid in definition.feature_ids}
            definitions = [replace(d, feature_ids=tuple(replacements[fid].id for fid in d.feature_ids)) if d.id == definition.id else d for d in document.professional_measurements]
            used = {fid for d in definitions for fid in d.feature_ids}
            features = [f for f in document.professional_features if f.id in used] + list(replacements.values())
            self.change(document, "解除测量关联并冻结", lambda: document.apply_professional_state(
                features, definitions,
                selected_id=definition.id, mark_dirty=False))
        except (ValueError, TypeError) as exc:
            self.message(str(exc))

    def arrange(self):
        controller = self.controller()
        if not controller:
            return
        controller.canvas.arrange_measurement_annotations()
        document = controller.canvas._document
        results = controller.results(); occupied = []
        rows = []
        zoom = max(controller.canvas._zoom, .001)
        fm = QFontMetricsF(controller.canvas.font())
        for i, definition in enumerate(document.professional_measurements, 1):
            result = presentation_result(document, definition, results[definition.id])
            text = result_caption(definition, result, i,
                                  decimals=controller.canvas._settings.professional_measurement_label_decimals)
            width, height = (fm.horizontalAdvance(text) + 14) / zoom, (fm.height() + 8) / zoom
            offset = choose_automatic_offset(result.anchor_px, width, height, occupied=occupied,
                image_rect=QRectF(0, 0, *document.image_size), gap=18 / zoom)
            occupied.append(QRectF(result.anchor_px.x + offset.x - width / 2, result.anchor_px.y + offset.y - height / 2, width, height))
            previous = definition.annotation_layout or MeasurementAnnotationLayout()
            rows.append(replace(definition, annotation_layout=replace(previous, placement_mode="automatic", offset_px=offset, revision=previous.revision + 1)))
        self.change(document, "整理专业测量标注", lambda: document.apply_professional_state(document.professional_features, rows, mark_dirty=False))

    def export(self):
        document = self.window.current_document(); controller = self.controller()
        if document is None or controller is None:
            return
        path, _ = QFileDialog.getSaveFileName(self.window, "导出专业测量明细", "专业测量.csv", "CSV (*.csv);;Excel (*.xlsx)")
        if not path:
            return
        try:
            export_professional_results(document, controller.results(), Path(path))
            self.message("已导出专业测量明细，未混入纤维直径统计。")
        except Exception as exc:
            QMessageBox.warning(self.window, "导出失败", str(exc))

    def start_caliper(self, controller, key, points, *, replace_id=None):
        self.cancel()
        self.active = controller
        controller.key = None; controller.points = list(points); controller.start_focus = controller.focus()
        document = controller.canvas._document
        target = refit_target_snapshot(document, replace_id) if replace_id else None
        if replace_id and target is None:
            self.cancel(); self.message("重新找边的测量或来源已删除，请重新选择测量。")
            return
        try:
            image, origin = raw_caliper_source(self.window, controller.canvas, points, self.panel.options(), key)
        except (ValueError, RuntimeError) as exc:
            self.message(str(exc)); return
        options = self.panel.options()
        cancel = Event()
        job = {"cancel": cancel, "controller": controller, "key": key, "points": points,
               "source": source_identity(controller.canvas), "scale": calibration_scale(document),
               "calibration": calibration_snapshot(document),
               "options": options, "replace_id": replace_id, "refit_target": target,
               "raw_plane": None if hasattr(controller.canvas, "pixel_work_enabled") else self.window._rasters.get(document.id)}
        self._job = job
        self.message("后台提取原始像素边缘…可按 Esc 取消。")
        channel_order = "RGBA" if image.ndim == 3 and image.shape[2] == 4 else "RGB"
        def work():
            from fdm.services.metrology_fitting import run_line_caliper, run_circle_caliper
            try:
                if key == "caliper_circle":
                    sx, sy = job["scale"]
                    radius = math.hypot((points[1].x - points[0].x) * sx, (points[1].y - points[0].y) * sy)
                    result = run_circle_caliper(image, points[0], radius / sx, radius_y_px=radius / sy,
                        options=options, scale=job["scale"], origin_px=origin, channel_order=channel_order, cancel=cancel.is_set)
                else:
                    result = run_line_caliper(image, *points, options=options, scale=job["scale"], origin_px=origin,
                        double_edge=key == "caliper_width", channel_order=channel_order, cancel=cancel.is_set)
            except Exception as exc:
                result = exc
            if not cancel.is_set():
                self.fitFinished.emit(job, result)
        self._workers = [t for t in self._workers if t.is_alive()]
        thread = Thread(target=work, name="professional-caliper", daemon=True)
        self._workers.append(thread); thread.start()

    def _fit_finished(self, job, result):
        if self._job is not job or job["cancel"].is_set():
            return
        controller = job["controller"]
        document = controller.canvas._document
        if (controller is not self.controller() or source_identity(controller.canvas) != job["source"]
                or calibration_scale(document) != job["scale"]
                or calibration_snapshot(document) != job["calibration"]
                or (job["raw_plane"] is not None and self.window._rasters.get(document.id) is not job["raw_plane"])):
            self.cancel(); self.message("图像、焦层或标定已改变；丢弃过期拟合。")
            return
        if job["replace_id"] and refit_target_snapshot(document, job["replace_id"]) != job["refit_target"]:
            self.cancel(); self.message("重新找边的测量或来源已改变 / 删除，已丢弃过期拟合。")
            return
        self._job = None
        if isinstance(result, Exception):
            self.message("拟合失败：" + str(result)); return
        controller.evidence = result
        controller.canvas.update()
        self.panel.details.setPlainText(fit_details_text(result.to_dict(), document.calibration.unit if document.calibration else "px"))
        self._preview = (job, result)
        self.panel.confirm.setEnabled(result.success)
        self.message("拟合预览已完成；请检查搜索区、绿色采用点、红色剔除点后确认。" if result.success else "拟合失败，未生成尺寸：" + result.reason)

    def confirm_fit(self):
        if not self._preview:
            return
        job, result = self._preview; controller = job["controller"]
        from fdm.services.metrology_service import professional_source_unavailable_reason
        reason = professional_source_unavailable_reason(controller.canvas._document)
        if reason:
            self.cancel(); self.message(reason)
            return
        if not result.success:
            return
        if (source_identity(controller.canvas) != job["source"] or calibration_scale(controller.canvas._document) != job["scale"]
                or calibration_snapshot(controller.canvas._document) != job["calibration"]
                or (job["raw_plane"] is not None and self.window._rasters.get(controller.canvas._document.id) is not job["raw_plane"])):
            self.cancel(); self.message("来源已改变，请重新拟合。"); return
        if job["replace_id"] and refit_target_snapshot(controller.canvas._document, job["replace_id"]) != job["refit_target"]:
            self.cancel(); self.message("重新找边的测量或来源已改变 / 删除，已丢弃过期拟合。")
            return
        points = [Point(*p) for section in result.sections if section.accepted for p in (section.first_px, section.second_px)] if job["key"] == "caliper_width" else [Point(*p) for p in result.fit.points_px]
        feature = MeasurementFeature(new_id("feature"), controller.canvas._document.id, job["key"], tuple(points),
            parameters={"caliper": job["options"].to_dict(), "roi_points": [p.to_dict() for p in job["points"]],
                        "source_identity": job["source"], "calibration_at_creation": job["calibration"]},
            quality=result.to_dict(), focus_index=controller.focus(),
            source_revision=slide_revision(controller.canvas._document))
        kind = {"caliper_line": "fit_line", "caliper_circle": "fit_circle", "caliper_width": "width"}[job["key"]]
        self.commit(controller, kind, [feature], replace_id=job["replace_id"])
        self._preview = None; self.panel.confirm.setEnabled(False)

    def refit_selected(self):
        document, definition = self.selected()
        if not definition:
            return
        feature = document.get_professional_feature(definition.feature_ids[0])
        if feature is None or not feature.kind.startswith("caliper_"):
            self.message("请选择卡尺测量；手动几何可通过控制点或辅助基准修改。")
            return
        if definition.focus_index != self.controller().focus():
            self.message("请先在结果表选择该测量并回到来源焦层。")
            return
        points = tuple(Point.from_dict(p) for p in feature.parameters.get("roi_points", []))
        if len(points) != 2:
            self.message("缺少原始搜索区，请重新建立卡尺。"); return
        index = self.panel.tool.findData(feature.kind); self.panel.tool.setCurrentIndex(index)
        self.start_caliper(self.controller(), feature.kind, points, replace_id=definition.id)


def raw_caliper_source(window, canvas, points, options, key):
    """Bounded immutable raw crop; never reads a viewing LUT result."""
    document = canvas._document
    margin = options.search_half_width_px + options.averaging_half_width_px + 4
    if key == "caliper_circle":
        sx, sy = calibration_scale(document)
        radius = math.hypot((points[1].x - points[0].x) * sx, (points[1].y - points[0].y) * sy)
        xs = [points[0].x - radius / sx, points[0].x + radius / sx]
        ys = [points[0].y - radius / sy, points[0].y + radius / sy]
    else:
        xs, ys = [p.x for p in points], [p.y for p in points]
    left, top = math.floor(min(xs) - margin), math.floor(min(ys) - margin)
    right, bottom = math.ceil(max(xs) + margin) + 1, math.ceil(max(ys) + margin) + 1
    if hasattr(canvas, "pixel_work_enabled"):
        if not canvas.pixel_work_enabled():
            raise ValueError(canvas.pixel_work_unavailable_reason())
        origin = canvas.mounted_image_origin()
        image = canvas._image
        if image is None or image.isNull():
            raise ValueError("原生像素尚未就绪")
        bounds = QRectF(origin.x, origin.y, image.width(), image.height())
        roi = QRectF(left, top, right - left, bottom - top)
        if not bounds.contains(roi):
            raise ValueError("搜索区域超出已加载原生视场；请移动视场或缩小搜索区域。")
        coverage = canvas.presentation_state().coverage_rects
        # Vector coverage may be discontinuous; require the full search region,
        # not only its endpoints, to be covered by exact native pixels.
        from PySide6.QtGui import QRegion
        region = QRegion()
        for rect in coverage:
            region |= QRegion(QRectF(*rect).toAlignedRect())
        if not QRegion(roi.toAlignedRect()).subtracted(region).isEmpty():
            raise ValueError("搜索区域含未加载或缺失瓦片，不能拟合。")
        cropped = image.copy(left - int(origin.x), top - int(origin.y), right - left, bottom - top)
        array = raster_plane_to_numpy(qimage_to_raster_plane(cropped))
    else:
        plane = window._rasters.get(document.id)
        if plane is None:
            raise ValueError("缺少原始像素")
        left, top = max(0, left), max(0, top)
        right, bottom = min(plane.width, right), min(plane.height, bottom)
        array = raster_plane_to_numpy(plane)[top:bottom, left:right]
    if array.size == 0 or array.nbytes > 128 * 1024 * 1024:
        raise ValueError("搜索区为空或超过 128 MiB，请缩小区域。")
    return np.array(array, copy=True), (left, top)


def export_professional_results(document, results, path: Path):
    rows = []
    for index, definition in enumerate(document.professional_measurements, 1):
        result = results[definition.id]
        for metric in result.metrics if result.valid else (None,):
            rows.append({"编号": f"P{index}", "测量ID": definition.id, "图像ID": document.id,
                "焦层": definition.focus_index, "类型": definition.kind, "指标": metric.label if metric else "",
                "数值": metric.value if metric else None, "单位": metric.unit if metric else "", "状态": result.status,
                "原因": result.reason, "来源特征": ";".join(definition.feature_ids),
                "标定": json.dumps(document.calibration.to_dict() if document.calibration else {}, ensure_ascii=False),
                "参数": json.dumps(definition.parameters, ensure_ascii=False), "质量": json.dumps(result.quality, ensure_ascii=False)})
    if path.suffix.lower() == ".xlsx":
        import pandas as pd
        pd.DataFrame(rows).to_excel(path, index=False)
    else:
        with path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["编号", "状态"])
            writer.writeheader(); writer.writerows(rows)


def slide_revision(document):
    if document.professional_source_revision is not None:
        return document.professional_source_revision
    metadata = document.metadata.get("digital_slide", {})
    revision = metadata.get("stitching_revision", metadata.get("source_revision"))
    return str(revision) if revision is not None else None


def fit_details_text(payload, unit):
    lines = ["拟合预览：" + ("可确认" if payload.get("status") == "ready" else "未通过")]
    if payload.get("reason"):
        lines.append("原因：" + payload["reason"])
    profiles = payload.get("profiles", [])
    if profiles:
        used = sum(p.get("status") == "ready" for p in profiles)
        lines.append(f"有效截面：{used} / {len(profiles)}")
    fit = payload.get("fit") or payload.get("evaluated_fit") or (payload if "accepted_indices" in payload else {})
    if fit:
        lines.append(f"采用点：{len(fit.get('accepted_indices', []))}；剔除点：{len(fit.get('rejected_indices', []))}")
        if fit.get("rms") is not None:
            lines.append(f"残差 RMS：{fit['rms']:.5g} {unit}")
        if fit.get("max_residual") is not None:
            lines.append(f"最大残差：{fit['max_residual']:.5g} {unit}")
        if fit.get("coverage_deg") is not None:
            lines.append(f"圆周覆盖：{fit['coverage_deg']:.4g}°")
    stats = payload.get("statistics", {})
    for key, label in (("minimum", "最小宽度"), ("maximum", "最大宽度"), ("mean", "平均宽度"), ("stddev", "样本标准差")):
        if key in stats:
            lines.append(f"{label}：{stats[key]:.5g} {unit}")
    lines.append("绿色为采用点，红色为未采用点；残差不等于实际计量精度。")
    return "\n".join(lines)


def measurement_details_text(document, definition, result):
    unit = document.calibration.unit if document.calibration else "px"
    lines = ["状态：有效" if result.valid else "状态：无效 — " + result.reason,
             "来源：普通图像" if definition.focus_index is None else f"来源：焦层 {definition.focus_index + 1}"]
    lines.extend(f"{m.label}：{m.value:.8g} {m.unit}" for m in result.metrics)
    for fid in definition.feature_ids:
        feature = document.get_professional_feature(fid)
        if feature is None:
            lines.append("来源特征已删除"); continue
        lines.append(f"控制点：{len(feature.points_px)}；" + ("关联辅助基准" if feature.source_ref else "手动 / 提取坐标"))
        quality = result.quality.get("features", {}).get(fid, {})
        if "evaluated_fit" in quality:
            lines.append(fit_details_text(quality, unit).replace("拟合预览：未通过\n", ""))
        options = feature.parameters.get("caliper")
        if options:
            method = "梯度边缘" if options.get("method") == "gradient" else f"阈值交点 {options.get('threshold_percent', 50):g}%"
            lines.append(f"找边：{method}；搜索半宽 {options.get('search_half_width_px', 10):g} px；{options.get('sample_count', 32)} 个截面")
    lines.append("控制点、搜索区和完整采样证据随项目保存。")
    return "\n".join(lines)
