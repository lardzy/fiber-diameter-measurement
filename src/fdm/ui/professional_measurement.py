"""Professional measurement workspace and document/canvas coordinator."""
from __future__ import annotations

from dataclasses import replace
import csv
import json
import math
from pathlib import Path
from threading import Event, Thread

import numpy as np
from PySide6.QtCore import QEvent, QItemSelectionModel, QObject, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor
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
from fdm.ui.result_annotation_properties import AnnotationTarget
from fdm.ui.professional_rendering import (presentation_result,
    place_new_professional_annotation, professional_label_image_size)


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

TOOL_GROUPS = {
    "距离与坐标": ("distance", "point_line_distance", "point_segment_distance", "parallel_distance",
                 "projection", "coordinates", "fit_line"),
    "角度": ("angle", "line_angle"),
    "圆与圆弧": ("circle", "arc", "circle_center_distance", "circle_gap", "point_circle_distance", "fit_circle"),
    "长度与面积": ("polyline", "polygon"),
    "卡尺": ("caliper_line", "caliper_circle", "caliper_width"),
}

SOURCE_TITLES = {
    "point_line_distance": ("目标点", "基准线"),
    "point_segment_distance": ("目标点", "基准线段"),
    "parallel_distance": ("第一条直线", "第二条直线"),
    "projection": ("待测线段", "投影基准线"),
    "line_angle": ("第一条直线", "第二条直线"),
    "circle_center_distance": ("第一个圆", "第二个圆"),
    "circle_gap": ("第一个圆", "第二个圆"),
    "point_circle_distance": ("目标点", "基准圆"),
    "coordinates": ("目标点", "坐标原点", "方向基准线"),
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


def feature_id_after_edit(document, definition, previous_id, new_id):
    """Keep references to the edited datum live; fork a directly shared legacy feature."""
    shared = any(d.id != definition.id and previous_id in d.feature_ids
                 for d in document.professional_measurements)
    linked = any(previous_id in f.professional_dependencies() for f in document.professional_features)
    return previous_id if linked and not shared else new_id


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
        self.tool.setAccessibleName("专业测量工具")
        for group, keys in TOOL_GROUPS.items():
            self.tool.addItem(group)
            heading = self.tool.model().item(self.tool.count() - 1)
            heading.setFlags(Qt.ItemFlag.NoItemFlags)
            font = heading.font()
            font.setBold(True)
            heading.setFont(font)
            for key in keys:
                self.tool.addItem(TOOLS[key][0], key)
        self.tool.setCurrentIndex(self.tool.findData("distance"))
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
        self.follow_sources = QCheckBox("跟随捕捉来源")
        self.follow_sources.setChecked(True)
        self.follow_sources.setToolTip("勾选后，能识别的端点、圆心等随来源更新。取消后，新取的点固定在当前坐标。")
        layout.addWidget(self.follow_sources)
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
        self.start_button = QPushButton("开始测量")
        self.start_button.clicked.connect(owner.begin)
        self.back_button = QPushButton("退一步")
        self.back_button.setEnabled(False)
        self.back_button.clicked.connect(owner.step_back)
        self.finish_button = QPushButton("完成")
        self.finish_button.setToolTip("完成不定点数的取点（Enter）")
        self.finish_button.clicked.connect(owner.finish)
        self.cancel_button = QPushButton("取消")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(owner.cancel_command)
        for button in (self.start_button, self.back_button, self.finish_button, self.cancel_button):
            row.addWidget(button)
        layout.addLayout(row)
        self.prompt = QLabel("选择工具，指定辅助来源或在画布取点。坐标为图像 / 标定平面。")
        self.prompt.setWordWrap(True); layout.addWidget(self.prompt)
        self.confirm = QPushButton("确认拟合并保存测量")
        self.confirm.setEnabled(False); self.confirm.clicked.connect(owner.confirm_fit); layout.addWidget(self.confirm)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["编号", "指标 / 状态", "数值", "单位"])
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.ExtendedSelection)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().hide()
        self.table.setMinimumHeight(140)
        self.table.itemSelectionChanged.connect(owner.table_selection)
        self.table.clicked.connect(owner.window._expand_selected_annotation_properties)
        self.table.setMouseTracking(True)
        self.table.itemEntered.connect(lambda item: owner.hover(item.data(Qt.ItemDataRole.UserRole)))
        self.table.viewport().installEventFilter(owner)
        layout.addWidget(self.table, 1)
        self.details = QTextEdit(); self.details.setReadOnly(True); self.details.setMaximumHeight(160)
        layout.addWidget(self.details)
        row = QHBoxLayout()
        self.single_result_buttons = []
        for text, callback in (("删除", owner.delete_selected), ("解除关联并冻结", owner.freeze_selected), ("重新找边", owner.refit_selected)):
            button = QPushButton(text)
            button.clicked.connect(callback)
            row.addWidget(button)
            self.single_result_buttons.append(button)
        layout.addLayout(row)
        row = QHBoxLayout()
        self.properties_button = QPushButton("标注属性")
        self.properties_button.clicked.connect(owner.window._show_result_annotation_properties)
        row.addWidget(self.properties_button)
        for text, callback in (("整理专业标注", owner.arrange), ("导出明细", owner.export)):
            button = QPushButton(text); button.clicked.connect(callback); row.addWidget(button)
        layout.addLayout(row)
        self.tool.currentIndexChanged.connect(self.configure)
        self.tool.activated.connect(lambda _index: owner.begin())
        for index, box in enumerate(self.sources):
            box.activated.connect(lambda _, slot=index: owner.choose_source(slot))
        for control in (self.method, self.polarity):
            control.currentIndexChanged.connect(owner.parameters_changed)
        for control in (self.width, self.count, self.threshold, self.step, self.contrast, self.fraction):
            control.valueChanged.connect(owner.parameters_changed)
        for control in (self.reverse, self.robust):
            control.toggled.connect(owner.parameters_changed)
        self.configure()

    def configure(self, *_):
        self.owner.cancel()
        key = self.tool.currentData()
        if key is None:
            return
        for box in self.sources:
            box.setCurrentIndex(0)
        self.caliper_widget.setVisible(key.startswith("caliper_"))
        self.confirm.setVisible(key.startswith("caliper_"))
        self.follow_sources.setVisible(not key.startswith("caliper_"))
        self.finish_button.setVisible(any(count < 0 for _, count in TOOLS[key][1]))
        self.start_button.setText("开始测量")
        self.supplement.setVisible(key in {"angle", "line_angle"})
        self.populate_sources()

    def populate_sources(self):
        key = self.tool.currentData()
        if not key:
            return
        slots = TOOLS[key][1]
        document = self.owner.window.current_document()
        resolver = make_construction_resolver(document) if document else None
        controller = self.owner.controller() if document and self.owner.panel is not None else None
        professional_sources = controller.snap_features() if controller else {}
        feature_names = {}
        if document:
            for n, definition in enumerate(document.professional_measurements, 1):
                title = definition.name or TOOLS.get(definition.kind, ("卡尺宽度",))[0]
                for slot, fid in enumerate(definition.feature_ids, 1):
                    suffix = f" · 来源 {slot}" if len(definition.feature_ids) > 1 else ""
                    feature_names.setdefault(fid, f"P{n} · {title}{suffix}")
        titles = SOURCE_TITLES.get(key, ("测量来源",))
        for index, box in enumerate(self.sources):
            previous = box.currentData()
            box.blockSignals(True); box.clear(); box.addItem("在画布选择", None)
            visible = index < len(slots) and not key.startswith("caliper_")
            self.form.setRowVisible(box, visible)
            if visible and document is not None:
                self.form.labelForField(box).setText(titles[index])
                expected = slots[index][0]
                for entity in document.construction_entities:
                    if not entity.visible or not entity.snappable:
                        continue
                    result = resolver.resolve(entity)
                    kind = {ResolvedPoint: "point", ResolvedLine: "line", ResolvedCircle: "circle"}.get(type(result.geometry))
                    if kind == expected:
                        box.addItem(f"辅助 · {entity.name or entity.id}", ("construction", entity.id))
                for fid, resolved in professional_sources.items():
                    if resolved.kind == expected:
                        box.addItem(feature_names.get(fid, fid), ("feature", fid))
                for number, measurement in enumerate(document.measurements, 1):
                    if expected == "line" and measurement.measurement_kind == "line":
                        box.addItem(f"M{number} · 直径线", ("measurement", measurement.id))
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
        self._table_viewport = None
        self.active = None
        self._job = None
        self._preview = None
        self._caliper_roi = None
        self._refit_timer = QTimer(self)
        self._refit_timer.setSingleShot(True)
        self._refit_timer.setInterval(120)
        self._refit_timer.timeout.connect(self._run_caliper)
        self._refreshing = False
        self._bound_document = None
        self._selected_ids = ()
        self._workers = []
        self._legacy_viewports = []
        self._workspace_active = True
        self._workspace_generation = 0
        self._registered_layout = None
        self.fitFinished.connect(self._fit_finished)
        self.panel = ProfessionalPanel(self)
        self._table_viewport = self.panel.table.viewport()
        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(self.panel)
        self.dock = QDockWidget("专业测量", window)
        self.dock.setObjectName("professionalMeasurementDock")
        self.dock.setWidget(scroll)
        window.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.dock)
        self.dock.hide()
        layout = getattr(window, "_adaptive_layout", None)
        if layout is not None:
            self.register_layout(layout)
        for pane in (getattr(window, "_inspector_records_pane", None), getattr(window, "_bottom_records_pane", None)):
            if pane is not None:
                pane.table.setMouseTracking(True)
                pane.table.entered.connect(self.legacy_table_hover)
                pane.table.viewport().installEventFilter(self)
                self._legacy_viewports.append(pane.table.viewport())

    def register_layout(self, layout):
        if self._registered_layout is layout:
            return
        from fdm.ui.workspace import WorkspacePage
        layout.register_panel("professional", self.dock, default_visible=False)
        layout.pageChanged.connect(lambda page: self.set_workspace_active(page is WorkspacePage.MEASURE))
        self._registered_layout = layout
        self.set_workspace_active(layout.page is WorkspacePage.MEASURE)

    def show(self):
        if not self.workspace_active:
            return
        layout = getattr(self.window, "_adaptive_layout", None)
        if layout is not None:
            self.register_layout(layout)
        inspector = getattr(self.window, "_inspector_dock", None)
        if inspector is not None:
            self.window.tabifyDockWidget(inspector, self.dock)
        if layout is not None:
            layout.set_panel_visible("professional", True)
        else:
            self.dock.show()
        self.dock.raise_(); self.refresh()

    @property
    def workspace_active(self):
        layout = getattr(self.window, "_adaptive_layout", None)
        return self._workspace_active and (layout is None or layout.panel_allowed("professional"))

    def set_workspace_active(self, active: bool):
        """Retire every unfinished operation before another page can own input."""
        active = bool(active)
        if active == self._workspace_active:
            return
        self._workspace_generation += 1
        if not active:
            self.cancel()
            self.dock.hide()
        self._workspace_active = active
        if active:
            self.refresh()

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
        if event.type() == QEvent.Type.Leave and obj is self._table_viewport:
            self.hover(None)
        if obj in self._legacy_viewports and event.type() == QEvent.Type.Leave:
            document = self.window.current_document()
            if document is not None:
                self.legacy_hover(document.id, None)
        return super().eventFilter(obj, event)

    def legacy_table_hover(self, index):
        if not self.workspace_active:
            return
        document = self.window.current_document()
        if document is not None:
            identity = self.window._records_controller.measurement_id_for_index(index)
            self.legacy_hover(document.id, identity)

    def legacy_hover(self, document_id, identity):
        if not self.workspace_active:
            return
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
        if not self.workspace_active:
            return None
        return getattr(self.window.current_canvas(), "_professional_controller", None)

    def message(self, text):
        if self.panel is not None:
            self.panel.prompt.setText(text)
        self.window.statusBar().showMessage(text, 6000)

    def parameters_changed(self, *_):
        if self._caliper_roi is None:
            return
        if self._job is not None:
            self._job["cancel"].set()
            self._job = None
        self._preview = None
        self.panel.confirm.setEnabled(False)
        controller = self._caliper_roi["controller"]
        controller.evidence = None
        controller.canvas.update()
        self.panel.details.clear()
        self._refit_timer.start()
        self.message("已保留搜索区，正在按新参数拟合…")

    def choose_source(self, slot):
        controller = self.active
        if controller is not None and controller.key:
            controller.set_slot_source(slot, self.panel.sources[slot].currentData())
            controller.canvas.setFocus()
        else:
            self.begin()

    def step_back(self):
        if self.active is not None and self.active.key:
            self.active.step_back()
            self.active.canvas.setFocus()

    def cancel_command(self):
        self.cancel()
        self.message("已取消本次专业测量。")

    def begin(self):
        if not self.workspace_active:
            return
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
        self.panel.start_button.setVisible(False)
        self.panel.back_button.setEnabled(True)
        self.panel.cancel_button.setEnabled(True)
        controller.begin(key, selections)
        controller.canvas.setFocus()

    def finish(self):
        if self.active:
            self.active.finish()

    def cancel(self):
        self._refit_timer.stop()
        self._caliper_roi = None
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
            self.panel.start_button.setVisible(True)
            self.panel.back_button.setEnabled(False)
            self.panel.cancel_button.setEnabled(False)

    def refresh(self):
        if self.panel is None or self._refreshing or not self.workspace_active:
            return
        self._refreshing = True
        try:
            document = self.window.current_document()
            self.panel.setEnabled(document is not None)
            if document is not self._bound_document:
                self.cancel()
                self._bound_document = document
                self._selected_ids = ()
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
                selected_ids = set(self.selected_ids(document))
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
                        if definition.id in selected_ids:
                            table.selectionModel().select(table.model().index(row, 0),
                                QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows)
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
            self._update_result_buttons(self.selected_ids(document))
            table.blockSignals(False)
        finally:
            self._refreshing = False

    def selected_ids(self, document):
        if document is None or not document.selected_professional_id:
            self._selected_ids = ()
            return ()
        primary = document.selected_professional_id
        if document is self._bound_document and primary in self._selected_ids:
            return tuple(identity for identity in self._selected_ids if document.get_professional_measurement(identity))
        return (primary,)

    def inspector_targets(self, document):
        identities = self.selected_ids(document)
        controller = self.controller() if identities else None
        if controller is None or controller.canvas._document is not document:
            return (), ""
        results = controller.results()
        targets = []
        for identity in identities:
            definition = document.get_professional_measurement(identity)
            if definition is None or identity not in results:
                return (), ""
            targets.append(AnnotationTarget.professional(document, definition, results[identity]))
        key = targets[0].family.removeprefix("professional:")
        label = TOOLS.get(key, (key,))[0].split(" / ")[0].split("：")[0]
        return targets, "专业" + label

    def _update_result_buttons(self, selected):
        self.panel.properties_button.setEnabled(bool(selected))
        for button in self.panel.single_result_buttons:
            button.setEnabled(len(selected) == 1)

    def table_selection(self):
        if self._refreshing:
            return
        table = self.panel.table
        selected = tuple(dict.fromkeys(index.data(Qt.ItemDataRole.UserRole)
                                       for index in table.selectionModel().selectedRows()))
        document = self.window.current_document()
        self._update_result_buttons(selected)
        if not selected:
            if document:
                document.select_professional(None)
            self._selected_ids = ()
            if self._preview is None and self._job is None:
                self.panel.details.clear()
            controller = self.controller()
            if controller:
                controller.canvas.update()
            self.window._refresh_object_inspector()
            return
        item = table.currentItem()
        primary = item.data(Qt.ItemDataRole.UserRole) if item else None
        self.select(primary if primary in selected else selected[0], navigate=len(selected) == 1, selected=selected)

    def select(self, identity, *, navigate=False, selected=None):
        document = self.window.current_document(); controller = self.controller()
        if document is None or controller is None:
            return
        definition = document.get_professional_measurement(identity)
        if definition is None:
            return
        self._selected_ids = selected or (identity,)
        document.select_professional(identity)
        self.window._select_measurement_table_id(None)
        if navigate and definition.focus_index is not None and hasattr(controller.canvas, "set_focus_index"):
            controller.canvas.set_focus_index(definition.focus_index)
        result = controller.results().get(identity)
        if result:
            self.panel.details.setPlainText(measurement_details_text(document, definition, result))
        controller.canvas.update()
        # Rebuilding the table on a click destroys Qt's Ctrl/Shift selection anchor.
        if selected is None:
            self.refresh()
        self.window._refresh_object_inspector()

    def change(self, document, label, mutator):
        if not self.workspace_active:
            return
        self.window._apply_document_change(document, label=label,
                                            mutator=mutator, impact=DocumentChangeImpact.SESSION)
        self.refresh()

    @staticmethod
    def _auto_label_size(canvas, definition, result, index):
        return professional_label_image_size(canvas._settings, definition, result, index, canvas.image_to_widget)

    def commit(self, controller, key, features, *, parameters=None, replace_id=None):
        if not self.workspace_active or controller is not self.controller():
            return
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
        if replace_id:
            original = document.get_professional_measurement(replace_id)
            if original is None:
                self.cancel(); self.message("重新找边的测量已删除，已丢弃过期拟合。")
                return
            if len(features) == len(original.feature_ids) == 1:
                edited_id = feature_id_after_edit(document, original, original.feature_ids[0], features[0].id)
                if edited_id == original.feature_ids[0]:
                    features = [replace(features[0], id=edited_id)]
                    combined = [f for f in combined if f.id != edited_id]
                    definition = replace(definition, feature_ids=(edited_id,))
        known = {f.id for f in combined}
        combined += [f for f in features if f.id not in known]
        # Evaluate on a lightweight temporary view; no document mutations while
        # selecting a label position or validating a draft.
        import copy
        scratch = copy.copy(document); scratch.professional_features = combined
        result = evaluate_measurement(scratch, definition)
        zoom = max(controller.canvas._zoom, .001)
        occupied = []
        for label in controller.label_hits.values():
            corner = controller.canvas.widget_to_image(label.text_rect.topLeft())
            occupied.append(QRectF(corner.x, corner.y, label.text_rect.width() / zoom, label.text_rect.height() / zoom))
        definition = place_new_professional_annotation(
            scratch, definition, result, controller.canvas._settings, len(document.professional_measurements) + 1,
            controller.canvas.image_to_widget, occupied)
        if replace_id:
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
        self.panel.start_button.setText("再测一次")
        self.panel.start_button.setVisible(True)
        self.panel.cancel_button.setEnabled(False)
        self.panel.back_button.setEnabled(False)
        self.select(definition.id)
        self.message("已保存专业测量。" if result.valid else "测量无效，已保留定义以便复核：" + result.reason)

    def edit_layout(self, controller, identity, layout):
        document = controller.canvas._document
        if layout.placement_mode == "automatic" and layout.offset_px is None:
            positions = {identity: layout}
            self._auto_professional_positions(controller, positions)
            layout = positions[identity]
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
        if not self.workspace_active:
            return
        document = self.window.current_document()
        if document is None or document.id != document_id or not document.get_measurement(identity):
            return
        if identity not in self.window._records_controller.selected_measurement_ids():
            document.select_measurement(identity)
            self.window._select_measurement_table_id(identity)
        self.window._show_result_annotation_properties()

    def professional_properties(self, controller, definition):
        if controller is not self.controller():
            return
        if definition.id not in self.selected_ids(controller.canvas._document):
            self.select(definition.id)
        self.window._show_result_annotation_properties()

    def _auto_professional_positions(self, controller, layouts):
        """Place the requested definitions around the existing labels on their own focus plane."""
        document = controller.canvas._document
        results = controller.results()
        zoom = max(controller.canvas._zoom, .001)
        occupied = {}
        pending = []
        for i, original in enumerate(document.professional_measurements, 1):
            layout = layouts.get(original.id, original.annotation_layout or MeasurementAnnotationLayout.legacy_professional())
            definition = replace(original, annotation_layout=layout)
            result = presentation_result(document, definition, results[definition.id])
            width, height = self._auto_label_size(controller.canvas, definition, result, i)
            boxes = occupied.setdefault(definition.focus_index, [])
            if definition.id in layouts:
                pending.append((definition.id, layout, result.anchor_px, width, height, boxes))
            else:
                offset = layout.offset_px or Point(65, -38)
                boxes.append(QRectF(result.anchor_px.x + offset.x - width / 2,
                                    result.anchor_px.y + offset.y - height / 2, width, height))
        for identity, layout, anchor, width, height, boxes in pending:
            offset = choose_automatic_offset(anchor, width, height, occupied=boxes,
                image_rect=QRectF(0, 0, *document.image_size), gap=18 / zoom)
            boxes.append(QRectF(anchor.x + offset.x - width / 2, anchor.y + offset.y - height / 2, width, height))
            layouts[identity] = replace(layout, placement_mode="automatic", offset_px=offset)

    def edit_annotation_selection(self, target, patch, operation):
        if not self.workspace_active or self.window.project.is_read_only_compatible:
            return
        document = self.window.current_document()
        controller = self.controller()
        if controller is None or not document or document.id != target[0]:
            return
        editor = self.window._object_inspector.annotation_editor
        if editor.target != target:
            return
        professional = target[1] == "professional"
        if professional:
            selected = self.selected_ids(document)
        else:
            selected = self.window._records_controller.selected_measurement_ids()
            if not selected:
                selected = (document.view_state.selected_measurement_id,)
        if set(selected) != set(target[2]):
            return
        objects = editor.targets
        layouts = {}
        styles = []
        for obj in objects:
            current = obj.layout
            layout = replace(current, **patch)
            if operation == "restore":
                layout = layout.with_automatic_position() if professional else layout.with_near_position()
            elif not professional and current.label_position != layout.label_position and current.placement_mode == "automatic":
                layout = replace(layout, offset_px=None)
            if not professional and layout != current:
                layout = controller.canvas.resolved_measurement_annotation_layout(document.get_measurement(obj.identity), layout)
            styles.append(layout.style_fields())
            if replace(layout, revision=current.revision) != current:
                layouts[obj.identity] = replace(layout, revision=current.revision + 1)
        if layouts and professional and operation == "restore":
            self._auto_professional_positions(controller, layouts)
        if layouts:
            def mutate():
                if professional:
                    definitions = [replace(d, annotation_layout=layouts[d.id]) if d.id in layouts else d
                                   for d in document.professional_measurements]
                    document.apply_professional_state(document.professional_features, definitions,
                        selected_id=document.selected_professional_id, mark_dirty=False)
                else:
                    for identity, layout in layouts.items():
                        document.get_measurement(identity).annotation_layout = layout
            self.change(document, "调整所选结果标注", mutate)
        if operation == "default" and len({obj.family for obj in objects}) == 1:
            family = objects[0].family
            settings = self.window._app_settings
            style = dict(settings.annotation_style_defaults.get(family, {}))
            style.update({k: v for k, v in styles[0].items() if all(other[k] == v for other in styles)})
            defaults = {**settings.annotation_style_defaults, family: style}
            from fdm.settings import AppSettingsIO
            try:
                AppSettingsIO.save(replace(settings, annotation_style_defaults=defaults))
            except OSError as exc:
                QMessageBox.warning(self.window, "保存标注默认值", f"已应用当前标注，无法保存默认值。\n{exc}")
            else:
                settings.annotation_style_defaults = defaults
                self.window.statusBar().showMessage("已保存，将用于之后新建的同类标注。", 4000)
        self.window._refresh_object_inspector()

    def selected(self):
        if not self.workspace_active:
            return None, None
        document = self.window.current_document()
        if document is None or len(self.selected_ids(document)) != 1:
            return None, None
        return document, document.get_professional_measurement(document.selected_professional_id)

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
            replacements = {}
            for fid in definition.feature_ids:
                frozen = freeze_feature(document, document.get_professional_feature(fid))
                identity = feature_id_after_edit(document, definition, fid, new_id("feature"))
                replacements[fid] = replace(frozen, id=identity)
            definitions = [replace(d, feature_ids=tuple(replacements[fid].id for fid in d.feature_ids)) if d.id == definition.id else d for d in document.professional_measurements]
            used = {fid for d in definitions for fid in d.feature_ids}
            retained = {f.id for f in replacements.values()}
            features = [f for f in document.professional_features if f.id in used and f.id not in retained] + list(replacements.values())
            self.change(document, "解除测量关联并冻结", lambda: document.apply_professional_state(
                features, definitions,
                selected_id=definition.id, mark_dirty=False))
        except (ValueError, TypeError) as exc:
            self.message(str(exc))

    def arrange(self):
        controller = self.controller()
        if not controller:
            return
        document = controller.canvas._document
        layouts = {}
        for definition in document.professional_measurements:
            layout = definition.annotation_layout or MeasurementAnnotationLayout.legacy_professional()
            if layout.placement_mode == "automatic":
                layouts[definition.id] = layout
        if not layouts:
            return
        self._auto_professional_positions(controller, layouts)
        rows = []
        for definition in document.professional_measurements:
            layout = layouts.get(definition.id)
            if layout is not None and layout != definition.annotation_layout:
                definition = replace(definition, annotation_layout=replace(layout, revision=layout.revision + 1))
            rows.append(definition)
        if rows != document.professional_measurements:
            self.change(document, "整理专业标注", lambda: document.apply_professional_state(
                document.professional_features, rows, selected_id=document.selected_professional_id, mark_dirty=False))

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
        if not self.workspace_active or controller is not self.controller():
            return
        if controller.canvas._read_only:
            self.message("当前图像为只读。")
            return
        if controller.canvas._tool_mode != "professional":
            self.window.set_tool_mode("professional")
        self.cancel()
        self.active = controller
        controller.key = None; controller.points = list(points); controller.start_focus = controller.focus()
        document = controller.canvas._document
        target = refit_target_snapshot(document, replace_id) if replace_id else None
        if replace_id and target is None:
            self.cancel(); self.message("重新找边的测量或来源已删除，请重新选择测量。")
            return
        self._caliper_roi = {"controller": controller, "key": key, "points": tuple(points),
               "workspace_generation": self._workspace_generation,
               "source": source_identity(controller.canvas), "scale": calibration_scale(document),
               "calibration": calibration_snapshot(document),
               "replace_id": replace_id, "refit_target": target,
               "raw_plane": None if hasattr(controller.canvas, "pixel_work_enabled") else self.window._rasters.get(document.id)}
        self.panel.start_button.setVisible(False)
        self.panel.cancel_button.setEnabled(True)
        self._run_caliper()

    def _run_caliper(self):
        roi = self._caliper_roi
        if roi is None:
            return
        controller = roi["controller"]
        document = controller.canvas._document
        if (not self.workspace_active or controller is not self.controller()
                or source_identity(controller.canvas) != roi["source"]
                or calibration_snapshot(document) != roi["calibration"]
                or (roi["raw_plane"] is not None and self.window._rasters.get(document.id) is not roi["raw_plane"])
                or (roi["replace_id"] and refit_target_snapshot(document, roi["replace_id"]) != roi["refit_target"])):
            self.cancel()
            self.message("来源或标定已改变，请重新选择搜索区。")
            return
        key, points = roi["key"], roi["points"]
        options = self.panel.options()
        try:
            image, origin = raw_caliper_source(self.window, controller.canvas, points, options, key)
        except (ValueError, RuntimeError) as exc:
            self.message(str(exc))
            return
        cancel = Event()
        job = {**roi, "cancel": cancel, "options": options}
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
        if (not self.workspace_active or self._job is not job or job["cancel"].is_set()
                or job.get("workspace_generation") != self._workspace_generation):
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
        if not self.workspace_active or not self._preview:
            return
        job, result = self._preview; controller = job["controller"]
        if job.get("workspace_generation") != self._workspace_generation or controller is not self.controller():
            self.cancel()
            return
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
        self._caliper_roi = None
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
        if feature.kind.startswith("caliper_"):
            lines.append(f"卡尺提取坐标 · {len(feature.points_px)} 个点")
        else:
            from fdm.services.metrology_service import control_point_source_labels
            sources = control_point_source_labels(document, feature)
            if feature.professional_source_id or feature.source_ref:
                lines.extend(sources)
            elif sources and all(source == "固定坐标" for source in sources):
                lines.append(f"{len(sources)} 个控制点 · 固定坐标")
            else:
                lines.extend(f"控制点 {index} · {source}" for index, source in enumerate(sources, 1))
        quality = result.quality.get("features", {}).get(fid, {})
        if "evaluated_fit" in quality:
            lines.append(fit_details_text(quality, unit).replace("拟合预览：未通过\n", ""))
        options = feature.parameters.get("caliper")
        if options:
            method = "梯度边缘" if options.get("method") == "gradient" else f"阈值交点 {options.get('threshold_percent', 50):g}%"
            lines.append(f"找边：{method}；搜索半宽 {options.get('search_half_width_px', 10):g} px；{options.get('sample_count', 32)} 个截面")
    lines.append("控制点、搜索区和完整采样证据随项目保存。")
    return "\n".join(lines)
