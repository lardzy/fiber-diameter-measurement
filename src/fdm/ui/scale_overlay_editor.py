"""Ephemeral canvas scale-bar editing, deliberately separate from project history."""

from __future__ import annotations

import weakref
from dataclasses import replace
from pathlib import Path

from PySide6.QtCore import QEvent, QObject, QPointF, QRectF, QSize, Qt, QTimer
from PySide6.QtGui import QColor, QIcon, QKeySequence, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QStyle,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from fdm.models import Point
from fdm.scale_overlay import ScaleOverlaySpec
from fdm.services.export_service import (
    ExportImageRenderMode,
    ExportRenderContext,
    ExportSelection,
)
from fdm.ui.scale_overlay_rendering import (
    layout_scale_overlay,
    moved_spec,
    paint_scale_overlay,
    scale_bar_path,
)
from fdm.ui.widgets import NoWheelComboBox, NoWheelDoubleSpinBox, NoWheelFontComboBox
from fdm.units import LENGTH_UNITS


def scale_target(host, document, render_mode: str, context=None):
    """Return source rectangle, output size, uniform scale, output translation."""
    canvas = host._canvases.get(document.id)
    if document.is_digital_slide():
        if context is None:
            if canvas is None:
                raise ValueError("数字切片尚未挂载。")
            rect = canvas.native_viewport_rect()
            return (
                (rect.x(), rect.y(), rect.width(), rect.height()),
                (round(rect.width()), round(rect.height())),
                1.0,
                (-rect.x(), -rect.y()),
            )
        return (
            (
                float(context.origin_x),
                float(context.origin_y),
                float(context.viewport_width),
                float(context.viewport_height),
            ),
            (context.viewport_width, context.viewport_height),
            1.0,
            (-context.origin_x, -context.origin_y),
        )
    image = host._images.get(document.id)
    if image is None or image.isNull():
        raise ValueError("图片尚未加载。")
    width, height = image.width(), image.height()
    zoom = (
        canvas.view_zoom()
        if canvas is not None
        else max(0.001, document.view_state.zoom)
    )
    if render_mode == ExportImageRenderMode.CURRENT_VIEWPORT:
        output = (
            (max(200, canvas.width()), max(160, canvas.height()))
            if canvas
            else (max(400, min(1400, width)), max(300, min(900, height)))
        )
        origin = (
            canvas.image_to_widget(Point(0, 0))
            if canvas
            else QPointF(document.view_state.pan.x, document.view_state.pan.y)
        )
        left, top = max(0.0, -origin.x() / zoom), max(0.0, -origin.y() / zoom)
        right, bottom = (
            min(width, (output[0] - origin.x()) / zoom),
            min(height, (output[1] - origin.y()) / zoom),
        )
        return (
            (left, top, max(0.0, right - left), max(0.0, bottom - top)),
            output,
            zoom,
            (origin.x(), origin.y()),
        )
    scale = (
        zoom if render_mode == ExportImageRenderMode.SCREEN_SCALE_FULL_IMAGE else 1.0
    )
    return (
        (0.0, 0.0, float(width), float(height)),
        (max(1, round(width * scale)), max(1, round(height * scale))),
        scale,
        (0.0, 0.0),
    )


def freeze_scale_contexts(host, documents, selection, contexts):
    result = dict(contexts or {})
    errors = []
    for document in documents:
        try:
            context = result.get(document.id) or ExportRenderContext(
                document.id, selection.render_mode
            )
            target, output, scale, offset = scale_target(
                host,
                document,
                selection.render_mode,
                context if document.is_digital_slide() else None,
            )
            calibration = document.calibration
            snapshot = (
                (calibration.pixels_per_unit, calibration.unit) if calibration else None
            )
            layout = layout_scale_overlay(
                selection.scale_overlay, target, *(snapshot or (None, None))
            )
            result[document.id] = replace(
                context,
                scale_overlay=selection.scale_overlay,
                scale_layout=layout,
                output_size=output,
                image_scale=scale,
                image_offset=offset,
                calibration_snapshot=snapshot,
            )
        except (ValueError, RuntimeError) as exc:
            errors.append(f"{Path(document.path).name}：{exc}")
    if errors:
        raise ValueError("以下图片的比例尺无法导出：\n" + "\n".join(errors))
    return result


class ScaleNumberInput(NoWheelDoubleSpinBox):
    def textFromValue(self, value):
        return f"{value:.12g}"


class ScaleOverlayPanel(QGroupBox):
    def __init__(self, controller, parent=None):
        super().__init__("比例尺", parent)
        self.controller = controller
        self.loading = False
        self.setObjectName("scaleOverlayEditor")
        outer = QVBoxLayout(self)
        outer.setSpacing(7)
        actions = QHBoxLayout()
        self.export_button = QPushButton("导出图片…")
        self.export_button.setObjectName("scaleExportButton")
        self.export_button.setProperty("primary", True)
        self.export_button.setStyleSheet(
            "QPushButton#scaleExportButton { background: #237F74; color: white; "
            "border: 1px solid #237F74; font-weight: 600; }"
            "QPushButton#scaleExportButton:hover { background: #1C6A61; }"
            "QPushButton#scaleExportButton:pressed { background: #16544D; }"
            "QPushButton#scaleExportButton:disabled { background: palette(button); "
            "color: palette(mid); border-color: palette(mid); }"
        )
        self.export_button.clicked.connect(controller.export)
        self.hide_button = QPushButton("隐藏比例尺")
        self.hide_button.clicked.connect(controller.hide)
        actions.addWidget(self.export_button, 1)
        actions.addWidget(self.hide_button, 1)
        outer.addLayout(actions)
        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        form.setVerticalSpacing(6)
        outer.addLayout(form)
        self.scope = self.combo((("全部图片", "all"), ("仅当前图片", "current")))
        form.addRow("显示范围", self.scope)
        self.length_mode = self.combo((("自动", "auto"), ("自定义", "custom")))
        self.length = self.number(0.000000000001, 1e15, 12)
        self.unit = self.combo(
            tuple((u.symbol, u.code) for u in LENGTH_UNITS) + (("px", "px"),)
        )
        row = QHBoxLayout()
        row.setSpacing(4)
        row.addWidget(self.length_mode)
        row.addWidget(self.length, 1)
        row.addWidget(self.unit)
        form.addRow("长度", row)
        self.position = self.combo(
            (
                ("右下角", "bottom_right"),
                ("左下角", "bottom_left"),
                ("右上角", "top_right"),
                ("左上角", "top_left"),
                ("图上拖动", "manual"),
            )
        )
        position = QHBoxLayout()
        position.addWidget(self.position, 1)
        self.locate_button = QToolButton()
        self.locate_button.setText("定位")
        self.locate_button.setToolTip("定位比例尺 / 适合预览范围")
        self.locate_button.clicked.connect(controller.locate)
        position.addWidget(self.locate_button)
        form.addRow("位置", position)
        self.style_combo = self.combo(
            (
                ("端点刻度", "ticks"),
                ("上端点刻度", "ticks_up"),
                ("下端点刻度", "ticks_down"),
                ("四等分刻度", "divisions"),
                ("纯线", "line"),
                ("实心条", "bar"),
            )
        )
        self.style_combo.setIconSize(QSize(48, 20))
        for index in range(self.style_combo.count()):
            self.style_combo.setItemIcon(
                index, self.style_icon(self.style_combo.itemData(index))
            )
        form.addRow("样式", self.style_combo)
        self.stroke = self.number(0.1, 2000, 1)
        self.stroke.setSuffix(" px")
        self.stroke.setToolTip("实际线宽或实心条高度，单位为原图像素；预览与导出一致。")
        self.stroke_label = QLabel("线宽")
        form.addRow(self.stroke_label, self.stroke)
        self.font_mode = self.combo((("自动", "auto"), ("自定义", "custom")))
        self.font_size = self.number(1, 10000, 1)
        self.font_size.setSuffix(" px")
        self.font_size.setToolTip("原图像素字号")
        size = QHBoxLayout()
        size.addWidget(self.font_mode)
        size.addWidget(self.font_size, 1)
        form.addRow("字号", size)
        colors = QHBoxLayout()
        colors.setSpacing(4)
        for title, color in (
            ("红色", "#FF0000"),
            ("黑色", "#000000"),
            ("白色", "#FFFFFF"),
        ):
            button = QPushButton(title)
            colors.addWidget(button)
            button.clicked.connect(
                lambda checked=False, c=color: controller.change(color=c, text_color=c)
            )
        self.color = QPushButton("其他…")
        colors.addWidget(self.color)
        self.color.clicked.connect(self.pick_color)
        form.addRow("配色", colors)
        self.region = self.combo((("整图", "image"), ("当前视窗", "viewport")))
        form.addRow("预览范围", self.region)
        self.hint = QLabel()
        self.hint.setWordWrap(True)
        outer.addWidget(self.hint)
        self.calibrate = QPushButton("去标定")
        self.calibrate.clicked.connect(controller.calibrate)
        outer.addWidget(self.calibrate)
        self.more_toggle = QCheckBox("更多样式")
        outer.addWidget(self.more_toggle)
        self.more = QWidget()
        more = QFormLayout(self.more)
        more.setContentsMargins(0, 0, 0, 0)
        more.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self.font_combo = NoWheelFontComboBox()
        self.font_combo.setMinimumWidth(0)
        self.bold = QCheckBox("粗体")
        self.text_position = self.combo((("文字在上", "above"), ("文字在下", "below")))
        self.text_color = QPushButton("文字颜色…")
        self.text_color.clicked.connect(lambda: self.pick_color(text_only=True))
        more.addRow("字体", self.font_combo)
        more.addRow("", self.bold)
        more.addRow("文字位置", self.text_position)
        more.addRow("", self.text_color)
        outer.addWidget(self.more)
        self.more.hide()
        self.more_toggle.toggled.connect(self.more.setVisible)
        controls = QHBoxLayout()
        self.undo_button = QToolButton()
        self.undo_button.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_ArrowBack)
        )
        self.undo_button.setToolTip("撤销比例尺调整")
        self.undo_button.setAccessibleName("撤销比例尺调整")
        self.redo_button = QToolButton()
        self.redo_button.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_ArrowForward)
        )
        self.redo_button.setToolTip("重做比例尺调整")
        self.redo_button.setAccessibleName("重做比例尺调整")
        controls.addWidget(self.undo_button)
        controls.addWidget(self.redo_button)
        self.undo_button.clicked.connect(controller.undo)
        self.redo_button.clicked.connect(controller.redo)
        controls.addStretch(1)
        for title, callback in (
            ("完成编辑", controller.finish),
            ("取消编辑", controller.cancel),
        ):
            button = QPushButton(title)
            button.clicked.connect(callback)
            controls.addWidget(button)
        outer.addLayout(controls)
        self.scope.currentIndexChanged.connect(
            lambda: (
                controller.set_scope(self.scope.currentData())
                if not self.loading
                else None
            )
        )
        for widget in (
            self.length_mode,
            self.position,
            self.font_mode,
            self.region,
            self.text_position,
        ):
            widget.currentIndexChanged.connect(self.apply)
        for widget in (self.length, self.font_size, self.stroke):
            widget.valueChanged.connect(self.apply)
        self.unit.currentIndexChanged.connect(self.change_unit)
        self.style_combo.currentIndexChanged.connect(self.change_style)
        self.font_combo.currentFontChanged.connect(self.change_font)
        self.bold.toggled.connect(self.apply)

    @staticmethod
    def style_icon(style):
        """Use the same geometry for style samples as for scientific output."""
        spec = ScaleOverlaySpec(style=style, unit="px")
        layout = layout_scale_overlay(spec, (0.0, 0.0, 400.0, 200.0))
        baseline = (
            12.0
            if style in ("ticks_up", "divisions")
            else 8.0
            if style == "ticks_down"
            else 10.0
        )
        layout = replace(
            layout,
            target=(0.0, 0.0, 48.0, 20.0),
            start=(4.0, baseline),
            end=(44.0, baseline),
            stroke=4.0 if style == "bar" else 2.0,
            tick_height=12.0
            if style in ("ticks", "ticks_up", "ticks_down", "divisions")
            else 0.0,
        )
        pixmap = QPixmap(96, 40)
        pixmap.setDevicePixelRatio(2)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillPath(scale_bar_path(layout), QColor("#E64A4A"))
        painter.end()
        return QIcon(pixmap)

    @staticmethod
    def combo(items):
        widget = NoWheelComboBox()
        for text, value in items:
            widget.addItem(text, value)
        return widget

    @staticmethod
    def number(minimum, maximum, decimals):
        widget = ScaleNumberInput()
        widget.setDecimals(decimals)
        widget.setRange(minimum, maximum)
        widget.setKeyboardTracking(False)
        widget.setMinimumWidth(64)
        widget.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        return widget

    def apply(self, *_):
        if self.loading:
            return
        spec = self.controller.spec
        unit = spec.unit
        if spec.length_mode == "auto" and self.length_mode.currentData() == "custom":
            layout, _ = self.controller.current_layout()
            if layout is not None:
                unit = layout.unit
        self.controller.change(
            length_mode=self.length_mode.currentData(),
            length=self.length.value(),
            # A disabled "px" hint on an uncalibrated image must never turn a
            # remembered physical length into pixels when changing its style.
            unit=unit,
            position=self.position.currentData(),
            font_mode=self.font_mode.currentData(),
            font_size=self.font_size.value(),
            bold=self.bold.isChecked(),
            # The legacy solid-bar style stores half its drawn height. Expose
            # the actual height to the user without changing saved preferences.
            line_width=self.stroke.value()
            / (2 if self.style_combo.currentData() == "bar" else 1),
            style=self.style_combo.currentData(),
            text_position=self.text_position.currentData(),
            preview_region=self.region.currentData(),
        )

    def change_style(self):
        if not self.loading:
            self.controller.change(style=self.style_combo.currentData())

    def change_unit(self):
        if not self.loading:
            self.controller.set_spec(
                self.controller.spec.with_unit(self.unit.currentData())
            )

    def change_font(self, font):
        if not self.loading:
            self.controller.change(font_family=font.family())

    def pick_color(self, checked=False, *, text_only=False):
        current = (
            self.controller.spec.text_color if text_only else self.controller.spec.color
        )
        color = QColorDialog.getColor(QColor(current), self, "比例尺颜色")
        if color.isValid():
            self.controller.change(
                **(
                    {"text_color": color.name()}
                    if text_only
                    else {"color": color.name(), "text_color": color.name()}
                )
            )

    def refresh(self):
        self.loading = True
        try:
            spec = self.controller.spec
            doc = self.controller.host.current_document()
            calibrated = doc is not None and doc.calibration is not None
            digital = doc is not None and doc.is_digital_slide()
            layout, error = self.controller.current_layout()
            for widget, value in (
                (self.scope, self.controller.scope),
                (self.length_mode, spec.length_mode),
                (self.position, spec.position),
                (self.font_mode, spec.font_mode),
                (self.region, "viewport" if digital else spec.preview_region),
                (self.style_combo, spec.style),
                (self.text_position, spec.text_position),
                (
                    self.unit,
                    layout.unit if layout else (spec.unit if calibrated else "px"),
                ),
            ):
                widget.setCurrentIndex(widget.findData(value))
            self.length.setValue(
                layout.value if layout and spec.length_mode == "auto" else spec.length
            )
            self.length.setEnabled(spec.length_mode == "custom")
            self.unit.setEnabled(calibrated)
            self.font_size.setValue(
                layout.font_size
                if layout and spec.font_mode == "auto"
                else spec.font_size
            )
            self.font_size.setEnabled(spec.font_mode == "custom")
            self.region.setEnabled(not digital)
            from PySide6.QtGui import QFont

            self.font_combo.setCurrentFont(QFont(spec.font_family))
            self.bold.setChecked(spec.bold)
            self.stroke.setValue(spec.line_width * (2 if spec.style == "bar" else 1))
            self.stroke_label.setText("条高" if spec.style == "bar" else "线宽")
            self.color.setToolTip(spec.color)
            self.text_color.setToolTip(spec.text_color)
            self.calibrate.setVisible(not calibrated)
            text = error or (
                f"{layout.label}，约占目标画面宽度 {layout.fraction:.0%}"
                if layout
                else "请打开图片"
            )
            if digital:
                text += "\n框线标出当前焦层原始视窗的导出范围。"
            if (
                doc
                and digital
                and not self.controller.native_ready(
                    self.controller.host._canvases.get(doc.id)
                )
            ):
                text += "\n正在等待当前焦层原始像素…"
            self.hint.setText(text)
            self.export_button.setEnabled(layout is not None)
            self.undo_button.setEnabled(bool(self.controller.undo_stack))
            self.redo_button.setEnabled(bool(self.controller.redo_stack))
        finally:
            self.loading = False


class ScaleOverlayPreviewController(QObject):
    def __init__(self, host):
        super().__init__(host)
        self.host = host
        self.spec = (
            host._app_settings.last_scale_overlay
            or ScaleOverlaySpec.from_legacy_settings(host._app_settings)
        )
        self.confirmed = self.spec
        self.visible = False
        self.editing = False
        self.scope = "all"
        self.only_document_id = None
        self.panel = None
        self.undo_stack = []
        self.redo_stack = []
        self._backup = self.spec
        self._drag = None
        self._canvases = weakref.WeakSet()
        self._painted_bounds = weakref.WeakKeyDictionary()
        self._native_status = weakref.WeakKeyDictionary()
        self._editing_document_id = None
        self._pending_export = None
        self._legacy_position_pending = host._app_settings.last_scale_overlay is None
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.setInterval(40)
        self._refresh_timer.timeout.connect(self.refresh_panel)

    def create_panel(self, parent):
        self.panel = ScaleOverlayPanel(self, parent)
        self.panel.hide()
        return self.panel

    def sync(self):
        for canvas in self.host._canvases.values():
            if canvas not in self._canvases:
                self._canvases.add(canvas)
                canvas._scale_overlay_preview = self
                canvas.installEventFilter(self)
                canvas.viewTransformChanged.connect(self.view_changed)
        doc = self.host.current_document()
        if doc is None and self.editing:
            self.cancel()
        if self.editing and doc is not None and doc.id != self._editing_document_id:
            self._drag = None
            self._editing_document_id = doc.id
            if self.scope == "current" and doc.id != self.only_document_id:
                self.cancel()
        action = getattr(self.host, "scale_preview_action", None)
        if action:
            action.setEnabled(doc is not None)
            action.setChecked(self.visible)
        self.sync_status_button()
        edit = getattr(self.host, "scale_edit_action", None)
        if edit:
            edit.setEnabled(doc is not None)
        if self.editing:
            self.host.undo_action.setEnabled(bool(self.undo_stack))
            self.host.redo_action.setEnabled(bool(self.redo_stack))
        if self.editing:
            self._refresh_timer.start()
        if self.visible and doc and doc.is_digital_slide():
            canvas = self.host._canvases.get(doc.id)
            if canvas and self.is_visible(doc):
                canvas._request_native_frame()

    def is_visible(self, document):
        return self.visible and (
            self.scope == "all" or document.id == self.only_document_id
        )

    def sync_status_button(self):
        button = getattr(self.host, "_scale_status_button", None)
        if button is not None:
            button.setScaleState(
                self.visible, self.editing, self.host.current_document() is not None
            )

    def begin(self, preset=None):
        doc, canvas = self.host.current_document(), self.host.current_canvas()
        if doc is None or canvas is None:
            return
        if not self.editing and (
            canvas.has_pending_path_drawing()
            or canvas.has_magic_segment_session()
            or canvas.has_fiber_quick_session()
            or canvas.has_reference_instance_session()
            or getattr(canvas, "_drawing_anchor_raw", None) is not None
            or getattr(canvas, "_dragging_handle", None) is not None
        ):
            self.host.statusBar().showMessage(
                "请先完成或取消当前测量 / 分割，再编辑比例尺。", 6000
            )
            self.sync()
            return
        if self._legacy_position_pending:
            self._legacy_position_pending = False
            if doc.calibration:
                self.spec = replace(self.spec, unit=doc.calibration.unit)
            if self.spec.position == "manual" and doc.scale_overlay_anchor is not None:
                layout, _ = self.current_layout()
                if layout:
                    self.spec = moved_spec(
                        layout,
                        doc.scale_overlay_anchor.x,
                        doc.scale_overlay_anchor.y - layout.bounds[3],
                    )
            self.confirmed = self.spec
        if not self.editing:
            self._backup = self.confirmed
            self.undo_stack.clear()
            self.redo_stack.clear()
            self._pending_export = None
        self.visible = self.editing = True
        self._editing_document_id = doc.id
        if self.scope == "current":
            self.only_document_id = doc.id
        if preset is not None:
            self._pending_export = replace(preset)
            if preset.scale_overlay is not None:
                self.spec = preset.scale_overlay
            self.spec = replace(
                self.spec,
                preview_region="viewport"
                if preset.render_mode == ExportImageRenderMode.CURRENT_VIEWPORT
                else "image",
            )
        self.panel.show()
        self.host._inspector_dock.show()
        self.host._inspector_dock.raise_()
        self.host._inspector_scroll.ensureWidgetVisible(self.panel)
        self.refresh()
        self.sync()
        canvas.setFocus()

    def toggle(self, checked=False):
        if self.visible:
            self.hide()
        else:
            self.begin()

    def confirm(self):
        """Accept a draft without changing editor visibility or project history."""
        if self.current_layout()[0] is None:
            self.refresh_panel()
            return False
        self.confirmed = self.spec
        self._backup = self.confirmed
        previous = self.host._app_settings.last_scale_overlay
        if previous != self.confirmed:
            self.host._app_settings.last_scale_overlay = self.confirmed
            if self.host._save_app_settings(context="比例尺设置") is False:
                # Keep the accepted session value, but retry persisting next time.
                self.host._app_settings.last_scale_overlay = previous
        return True

    def finish(self):
        if not self.confirm():
            return False
        self.editing = False
        self._drag = None
        if self.panel:
            self.panel.hide()
        self.refresh()
        self.host._update_action_states()
        return True

    def cancel(self):
        self.spec = self._backup
        self.editing = False
        self._drag = None
        if self.panel:
            self.panel.hide()
        self.refresh()
        self.host._update_action_states()

    def hide(self):
        if self.editing and not self.finish():
            self.cancel()
        self.visible = False
        self.refresh()
        self.sync()

    def set_scope(self, scope):
        self.scope = scope
        doc = self.host.current_document()
        self.only_document_id = doc.id if scope == "current" and doc else None
        self.refresh()

    def set_spec(self, spec, *, remember=True):
        if spec == self.spec:
            return
        if remember:
            self.undo_stack.append(self.spec)
            self.undo_stack = self.undo_stack[-100:]
            self.redo_stack.clear()
        self.spec = spec
        self.refresh()

    def change(self, **values):
        self.set_spec(replace(self.spec, **values))

    def undo(self):
        if self.undo_stack:
            self.redo_stack.append(self.spec)
            self.spec = self.undo_stack.pop()
            self.refresh()

    def redo(self):
        if self.redo_stack:
            self.undo_stack.append(self.spec)
            self.spec = self.redo_stack.pop()
            self.refresh()

    def current_layout(self, canvas=None, spec=None):
        canvas = canvas or self.host.current_canvas()
        doc = getattr(canvas, "_document", None)
        if doc is None:
            return None, "请打开图片。"
        spec = spec or self.spec
        mode = (
            ExportImageRenderMode.CURRENT_VIEWPORT
            if doc.is_digital_slide() or spec.preview_region == "viewport"
            else ExportImageRenderMode.FULL_RESOLUTION
        )
        try:
            target, _, _, _ = scale_target(self.host, doc, mode)
            calibration = doc.calibration
            return layout_scale_overlay(
                spec,
                target,
                calibration.pixels_per_unit if calibration else None,
                calibration.unit if calibration else None,
            ), ""
        except (ValueError, RuntimeError) as exc:
            return None, str(exc)

    @staticmethod
    def native_ready(canvas):
        return (
            canvas is not None
            and canvas._native_frame_key == canvas._native_request_key()
        )

    def view_changed(self, *_):
        if self.visible:
            self._refresh_timer.start()

    def refresh_panel(self):
        if self.panel and self.editing:
            self.panel.refresh()

    def refresh(self):
        self.refresh_panel()
        for canvas in self.host._canvases.values():
            old_bounds = self._painted_bounds.get(canvas)
            document = getattr(canvas, "_document", None)
            layout, _ = self.current_layout(canvas) if document else (None, "")
            if (
                document
                and not document.is_digital_slide()
                and layout is not None
                and old_bounds is not None
            ):
                top_left = canvas.image_to_widget(
                    Point(layout.bounds[0], layout.bounds[1])
                )
                bottom_right = canvas.image_to_widget(
                    Point(
                        layout.bounds[0] + layout.bounds[2],
                        layout.bounds[1] + layout.bounds[3],
                    )
                )
                dirty = old_bounds.united(QRectF(top_left, bottom_right)).adjusted(
                    -16, -16, 16, 16
                )
                canvas.update(dirty.toAlignedRect())
            else:
                canvas.update()
        action = getattr(self.host, "scale_preview_action", None)
        if action:
            action.setChecked(self.visible)
        self.sync_status_button()
        if self.editing:
            self.host.undo_action.setEnabled(bool(self.undo_stack))
            self.host.redo_action.setEnabled(bool(self.redo_stack))

    def paint(self, canvas, painter):
        doc = canvas._document
        if not self.is_visible(doc):
            return
        layout, error = self.current_layout(canvas)
        digital = doc.is_digital_slide()
        if digital:
            rect = canvas.native_viewport_rect()
            top_left = canvas.image_to_widget(Point(rect.x(), rect.y()))
            bottom_right = canvas.image_to_widget(Point(rect.right(), rect.bottom()))
            frame = QRectF(top_left, bottom_right)
            painter.save()
            painter.setPen(QPen(QColor("#E9C46A"), 1.5, Qt.PenStyle.DashLine))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(frame)
            pending = not self.native_ready(canvas)
            status = (canvas._native_request_key(), pending)
            if self._native_status.get(canvas) != status:
                self._native_status[canvas] = status
                self._refresh_timer.start()
            message = "等待当前焦层原始像素…" if pending else "原始视窗导出范围"
            painter.drawText(
                QRectF(
                    frame.x() + 6, frame.y() + 6, max(230.0, frame.width() - 12), 28
                ),
                message,
            )
            painter.restore()
            if pending:
                return
        if layout is None:
            # The error notice is outside the old bar bounds; recovery must
            # repaint it as well as the scale bar.
            self._painted_bounds.pop(canvas, None)
            painter.save()
            painter.setPen(QColor("#E9C46A"))
            painter.drawText(
                QRectF(12, 12, max(100, canvas.width() - 24), 70),
                Qt.TextFlag.TextWordWrap,
                error,
            )
            painter.restore()
            return
        origin = canvas.image_to_widget(Point(0, 0))
        scale = canvas.view_zoom()
        self._painted_bounds[canvas] = QRectF(
            canvas.image_to_widget(Point(layout.bounds[0], layout.bounds[1])),
            canvas.image_to_widget(
                Point(
                    layout.bounds[0] + layout.bounds[2],
                    layout.bounds[1] + layout.bounds[3],
                )
            ),
        )
        painter.save()
        painter.translate(origin)
        painter.scale(scale, scale)
        paint_scale_overlay(painter, layout)
        painter.restore()
        if self.editing and canvas is self.host.current_canvas():
            painter.save()
            pen = QPen(QColor("#2A9D8F"), 1, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            rect = QRectF(
                canvas.image_to_widget(Point(layout.bounds[0], layout.bounds[1])),
                canvas.image_to_widget(
                    Point(
                        layout.bounds[0] + layout.bounds[2],
                        layout.bounds[1] + layout.bounds[3],
                    )
                ),
            )
            painter.drawRect(rect.adjusted(-4, -4, 4, 4))
            point = canvas.image_to_widget(Point(*layout.end))
            painter.setBrush(QColor("#2A9D8F"))
            painter.drawRect(QRectF(point.x() - 5, point.y() - 5, 10, 10))
            painter.restore()

    def eventFilter(self, canvas, event):
        if not self.editing or canvas is not self.host.current_canvas():
            return False
        kind = event.type()
        if kind == QEvent.Type.ShortcutOverride and (
            event.matches(QKeySequence.StandardKey.Undo)
            or event.matches(QKeySequence.StandardKey.Redo)
            or event.key() in (Qt.Key.Key_Escape, Qt.Key.Key_Return, Qt.Key.Key_Enter)
        ):
            event.accept()
            return True
        if kind == QEvent.Type.KeyPress:
            if event.matches(QKeySequence.StandardKey.Undo):
                self.undo()
                return True
            if event.matches(QKeySequence.StandardKey.Redo):
                self.redo()
                return True
            if event.key() == Qt.Key.Key_Escape:
                self.cancel()
                return True
            if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                self.finish()
                return True
        if kind not in (
            QEvent.Type.MouseButtonPress,
            QEvent.Type.MouseMove,
            QEvent.Type.MouseButtonRelease,
            QEvent.Type.MouseButtonDblClick,
        ):
            return False
        if (
            kind == QEvent.Type.MouseButtonRelease
            and event.button() == Qt.MouseButton.LeftButton
            and self._drag
        ):
            self._drag = None
            canvas._update_cursor()
            return True
        if getattr(canvas, "_space_pressed", False) or getattr(
            canvas, "_panning", False
        ):
            self._drag = None
            return False
        if kind in (QEvent.Type.MouseButtonPress, QEvent.Type.MouseButtonDblClick):
            if event.button() != Qt.MouseButton.LeftButton:
                return False
            layout, _ = self.current_layout(canvas)
            if layout is None:
                return True
            source = canvas.widget_to_image(event.position())
            point = QPointF(source.x, source.y)
            end = canvas.image_to_widget(Point(*layout.end))
            resize = (end - event.position()).manhattanLength() <= 15
            if resize or QRectF(*layout.bounds).adjusted(
                -8 / canvas.view_zoom(),
                -8 / canvas.view_zoom(),
                8 / canvas.view_zoom(),
                8 / canvas.view_zoom(),
            ).contains(point):
                self.undo_stack.append(self.spec)
                self.undo_stack = self.undo_stack[-100:]
                self.redo_stack.clear()
                self._drag = (layout, source, "resize" if resize else "move")
                canvas.setCursor(
                    Qt.CursorShape.SizeHorCursor
                    if resize
                    else Qt.CursorShape.ClosedHandCursor
                )
            return True
        if kind == QEvent.Type.MouseMove:
            if self._drag:
                layout, start, mode = self._drag
                point = canvas.widget_to_image(event.position())
                if mode == "move":
                    spec = moved_spec(
                        layout,
                        layout.bounds[0] + point.x - start.x,
                        layout.bounds[1] + point.y - start.y,
                    )
                else:
                    value = max(
                        1e-9,
                        (point.x - layout.start[0])
                        * layout.value
                        / (layout.end[0] - layout.start[0]),
                    )
                    spec = replace(
                        layout.spec,
                        length_mode="custom",
                        length=value,
                        unit=layout.unit,
                    )
                    resized, _ = self.current_layout(canvas, spec)
                    if resized is None:
                        return True
                    spec = moved_spec(
                        resized,
                        layout.start[0] - (resized.start[0] - resized.bounds[0]),
                        layout.bounds[1],
                    )
                self.set_spec(spec, remember=False)
            return bool(event.buttons() & Qt.MouseButton.LeftButton)
        if (
            kind == QEvent.Type.MouseButtonRelease
            and event.button() == Qt.MouseButton.LeftButton
        ):
            self._drag = None
            canvas._update_cursor()
            return True
        return False

    def locate(self):
        canvas = self.host.current_canvas()
        if canvas is None:
            return
        if canvas._document.is_digital_slide():
            canvas.fit_native_viewport()
        elif self.spec.preview_region == "image":
            canvas.fit_to_view()
        else:
            layout, _ = self.current_layout()
            if layout is None:
                canvas.fit_to_view()
        self.refresh()

    def calibrate(self):
        self.cancel()
        self.host._show_calibration_controls()

    def export(self):
        if not self.confirm():
            return
        doc = self.host.current_document()
        if doc is None:
            return
        selection = (
            replace(self._pending_export)
            if self._pending_export
            else ExportSelection(
                include_scale_overlay=True,
                include_watermark=bool(
                    not doc.is_digital_slide()
                    and doc.watermark
                    and doc.watermark.enabled
                ),
            )
        )
        selection.scale_overlay = self.confirmed
        selection.render_mode = (
            ExportImageRenderMode.CURRENT_VIEWPORT
            if doc.is_digital_slide() or self.spec.preview_region == "viewport"
            else (
                selection.render_mode
                if selection.render_mode != ExportImageRenderMode.CURRENT_VIEWPORT
                else ExportImageRenderMode.FULL_RESOLUTION
            )
        )
        # Modal export owns output-content selection. Keeping the editor mounted
        # covers cancel at either dialog, failure, and "adjust on canvas" alike.
        self.host.export_results(selection)
        self.refresh()
        self.sync()
