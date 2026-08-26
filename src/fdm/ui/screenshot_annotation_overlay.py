from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

from PySide6.QtCore import QPoint, QPointF, QRect, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import (
    QAction,
    QActionGroup,
    QColor,
    QFont,
    QGuiApplication,
    QImage,
    QKeyEvent,
    QMouseEvent,
    QPaintEvent,
    QPainter,
    QPen,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QFileDialog,
    QFontComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMenu,
    QMessageBox,
    QPushButton,
    QSlider,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from fdm.services.screenshot_capture import CaptureRect, CapturedFrame, ScreenInfo, union_rect
from fdm.ui.screenshot_editor import (
    EditCommand,
    EditorTool,
    InlineTextEdit,
    ScreenshotEditModel,
    command_rect,
    draw_edit_command,
    resized_command,
    selection_resize_target,
    translated_command,
)
from fdm.ui.widgets import NoWheelSpinBox


def screen_topology_signature(screens: Sequence[ScreenInfo]) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (
            item.name,
            item.logical_rect.x,
            item.logical_rect.y,
            item.logical_rect.width,
            item.logical_rect.height,
            item.physical_rect.x,
            item.physical_rect.y,
            item.physical_rect.width,
            item.physical_rect.height,
            round(float(item.device_pixel_ratio), 4),
        )
        for item in screens
    )


class CaptureViewportMapping:
    """Piecewise physical-pixel/logical-pixel mapping for mixed-DPI desktops."""

    def __init__(self, frame_rect: CaptureRect, screens: Sequence[ScreenInfo]) -> None:
        self.frame_rect = frame_rect.normalized()
        self.screens = tuple(screens)
        logical = union_rect(item.logical_rect for item in self.screens)
        self.virtual_logical_rect = logical or CaptureRect(0, 0, frame_rect.width, frame_rect.height)

    def logical_capture_rect(self, physical_rect: CaptureRect | None = None) -> CaptureRect:
        target = physical_rect or self.frame_rect
        fragments = []
        for screen in self.screens:
            clipped = target.intersection(screen.physical_rect)
            if clipped is not None:
                fragments.append(screen.physical_fragment_to_logical(clipped))
        return union_rect(fragments) or CaptureRect(0, 0, 0, 0)

    def widget_to_physical(self, point: QPointF) -> QPointF | None:
        logical_x = point.x() + self.virtual_logical_rect.x
        logical_y = point.y() + self.virtual_logical_rect.y
        screen = next(
            (
                item
                for item in self.screens
                if item.logical_rect.x <= logical_x < item.logical_rect.right
                and item.logical_rect.y <= logical_y < item.logical_rect.bottom
            ),
            None,
        )
        if screen is None:
            return None
        ratio_x = screen.physical_rect.width / max(1, screen.logical_rect.width)
        ratio_y = screen.physical_rect.height / max(1, screen.logical_rect.height)
        physical_x = screen.physical_rect.x + (logical_x - screen.logical_rect.x) * ratio_x
        physical_y = screen.physical_rect.y + (logical_y - screen.logical_rect.y) * ratio_y
        return QPointF(physical_x, physical_y)

    def physical_to_widget(self, point: QPointF) -> QPointF | None:
        screen = next(
            (
                item
                for item in self.screens
                if item.physical_rect.x <= point.x() < item.physical_rect.right
                and item.physical_rect.y <= point.y() < item.physical_rect.bottom
            ),
            None,
        )
        if screen is None:
            return None
        ratio_x = screen.logical_rect.width / max(1, screen.physical_rect.width)
        ratio_y = screen.logical_rect.height / max(1, screen.physical_rect.height)
        logical_x = screen.logical_rect.x + (point.x() - screen.physical_rect.x) * ratio_x
        logical_y = screen.logical_rect.y + (point.y() - screen.physical_rect.y) * ratio_y
        return QPointF(
            logical_x - self.virtual_logical_rect.x,
            logical_y - self.virtual_logical_rect.y,
        )

    def image_to_widget(self, point: QPointF, visible_rect: QRect) -> QPointF | None:
        return self.physical_to_widget(
            QPointF(
                self.frame_rect.x + visible_rect.x() + point.x(),
                self.frame_rect.y + visible_rect.y() + point.y(),
            )
        )

    def widget_to_image(self, point: QPointF, visible_rect: QRect) -> QPointF | None:
        physical = self.widget_to_physical(point)
        if physical is None:
            return None
        local = QPointF(
            physical.x() - self.frame_rect.x - visible_rect.x(),
            physical.y() - self.frame_rect.y - visible_rect.y(),
        )
        if not QRectF(0, 0, visible_rect.width(), visible_rect.height()).adjusted(-1, -1, 1, 1).contains(local):
            return None
        return local

    def image_fragments(
        self,
        visible_rect: QRect,
    ) -> tuple[tuple[QRectF, QRectF], ...]:
        desktop_rect = CaptureRect(
            self.frame_rect.x + visible_rect.x(),
            self.frame_rect.y + visible_rect.y(),
            visible_rect.width(),
            visible_rect.height(),
        )
        result: list[tuple[QRectF, QRectF]] = []
        for screen in self.screens:
            clipped = desktop_rect.intersection(screen.physical_rect)
            if clipped is None:
                continue
            logical = screen.physical_fragment_to_logical(clipped)
            destination = QRectF(
                logical.x - self.virtual_logical_rect.x,
                logical.y - self.virtual_logical_rect.y,
                logical.width,
                logical.height,
            )
            source = QRectF(
                clipped.x - desktop_rect.x,
                clipped.y - desktop_rect.y,
                clipped.width,
                clipped.height,
            )
            result.append((source, destination))
        return tuple(result)


class InlineAnnotationOverlay(QWidget):
    """Frameless, in-place screenshot annotation host."""

    completed = Signal(object)
    copyRequested = Signal(object)
    saveRequested = Signal(object)
    saveAsRequested = Signal(object, str)
    cancelled = Signal()
    stylesChanged = Signal(object)
    fallbackRequested = Signal(object)

    _TOOLS = (
        (EditorTool.SELECT, "选择", "V"),
        (EditorTool.RECTANGLE, "矩形", "R"),
        (EditorTool.ELLIPSE, "椭圆", "E"),
        (EditorTool.LINE, "直线", "L"),
        (EditorTool.ARROW, "箭头", "A"),
        (EditorTool.PEN, "画笔", "P"),
        (EditorTool.TEXT, "文字", "T"),
        (EditorTool.NUMBER, "编号", "N"),
        (EditorTool.HIGHLIGHT, "高亮", "H"),
        (EditorTool.MOSAIC, "马赛克", "M"),
        (EditorTool.BLUR, "模糊", "B"),
        (EditorTool.CROP, "裁剪", "C"),
    )
    _PRIMARY_NARROW = {
        EditorTool.SELECT,
        EditorTool.RECTANGLE,
        EditorTool.ARROW,
        EditorTool.TEXT,
        EditorTool.BLUR,
        EditorTool.CROP,
    }
    _SHORTCUTS = {
        Qt.Key.Key_V: EditorTool.SELECT,
        Qt.Key.Key_R: EditorTool.RECTANGLE,
        Qt.Key.Key_E: EditorTool.ELLIPSE,
        Qt.Key.Key_L: EditorTool.LINE,
        Qt.Key.Key_A: EditorTool.ARROW,
        Qt.Key.Key_P: EditorTool.PEN,
        Qt.Key.Key_T: EditorTool.TEXT,
        Qt.Key.Key_N: EditorTool.NUMBER,
        Qt.Key.Key_H: EditorTool.HIGHLIGHT,
        Qt.Key.Key_M: EditorTool.MOSAIC,
        Qt.Key.Key_B: EditorTool.BLUR,
        Qt.Key.Key_C: EditorTool.CROP,
    }

    def __init__(
        self,
        frame: CapturedFrame,
        screens: Sequence[ScreenInfo],
        *,
        styles: Mapping[str, object] | None = None,
        screens_provider: Callable[[], Sequence[ScreenInfo]] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        if not frame.valid:
            raise ValueError("不能标注空截图。")
        self.frame = frame
        self.model = ScreenshotEditModel(frame.image, self)
        self._screens = tuple(screens)
        self._mapping = CaptureViewportMapping(frame.rect, self._screens)
        self._screens_provider = screens_provider
        self._topology_signature = screen_topology_signature(self._screens)
        self._tool = EditorTool.RECTANGLE
        self._styles = self._normalized_styles(styles)
        self._points: list[tuple[float, float]] = []
        self._draft_modifiers = Qt.KeyboardModifier.NoModifier
        self._drag_origin: tuple[float, float] | None = None
        self._drag_current: tuple[float, float] | None = None
        self._selection_modifiers = Qt.KeyboardModifier.NoModifier
        self._resize_handle = ""
        self._line_endpoint: tuple[str, int] | None = None
        self._text_edit: InlineTextEdit | None = None
        self._editing_text_id = ""
        self._editing_text_style: dict[str, object] | None = None
        self._number = 1
        self._number_initialized = False
        self._zoom = 1.0
        self._pan = QPointF()
        self._space_pan = False
        self._pan_origin: QPoint | None = None
        self._output_pending = False
        self._status_timer = QTimer(self)
        self._status_timer.setSingleShot(True)
        self._status_timer.timeout.connect(lambda: self.status_label.hide())
        self._topology_timer = QTimer(self)
        self._topology_timer.setInterval(1000)
        self._topology_timer.timeout.connect(self._check_topology)
        self.setWindowTitle("截图标注")
        self.setWindowFlags(
            Qt.WindowType.Tool
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        virtual = self._mapping.virtual_logical_rect
        self.setGeometry(virtual.x, virtual.y, virtual.width, virtual.height)
        self._tool_actions: dict[EditorTool, QAction] = {}
        self._tool_buttons: dict[EditorTool, QToolButton] = {}
        self._compact_redo_action: QAction | None = None
        self._build_toolbar()
        self._build_properties()
        self.model.changed.connect(self._model_changed)
        self.model.historyChanged.connect(self._update_history_controls)
        self.model.selectionChanged.connect(self._selection_changed)
        active = str(self._styles.get("active_tool", "rectangle"))
        try:
            self.set_tool(EditorTool.parse(active))
        except ValueError:
            self.set_tool(EditorTool.RECTANGLE)
        self._update_history_controls(self.model.can_undo, self.model.can_redo)
        self._topology_timer.start()

    @property
    def tool(self) -> EditorTool:
        return self._tool

    @property
    def output_pending(self) -> bool:
        return self._output_pending

    def _normalized_styles(self, styles: Mapping[str, object] | None) -> dict[str, object]:
        source = dict(styles or {})
        tools = source.get("tools")
        return {
            "schema_version": 1,
            "active_tool": str(source.get("active_tool", "rectangle")),
            "tools": dict(tools) if isinstance(tools, dict) else {},
        }

    def _build_toolbar(self) -> None:
        self.toolbar = QFrame(self)
        self.toolbar.setObjectName("annotationToolbar")
        self.toolbar.setStyleSheet(
            "QFrame#annotationToolbar { background: rgba(28, 32, 39, 244);"
            " border: 1px solid rgba(255,255,255,42); border-radius: 8px; }"
            "QToolButton, QPushButton { color: #f5f7fa; background: rgba(255,255,255,12);"
            " border: 1px solid rgba(255,255,255,24); padding: 6px 8px; border-radius: 5px; }"
            "QToolButton:checked { background: #087f8c; }"
            "QToolButton:hover, QPushButton:hover { background: rgba(255,255,255,28); }"
            "QPushButton#finishButton { background: #079a8f; font-weight: 600; }"
        )
        layout = QHBoxLayout(self.toolbar)
        layout.setContentsMargins(7, 6, 7, 6)
        layout.setSpacing(2)
        group = QActionGroup(self)
        group.setExclusive(True)
        for tool, label, shortcut in self._TOOLS:
            action = QAction(label, self)
            action.setCheckable(True)
            action.setToolTip(f"{label}（{shortcut}）")
            action.triggered.connect(lambda _checked=False, item=tool: self.set_tool(item))
            group.addAction(action)
            button = QToolButton(self.toolbar)
            button.setDefaultAction(action)
            button.setText(label)
            button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
            layout.addWidget(button)
            self._tool_actions[tool] = action
            self._tool_buttons[tool] = button
        self.more_button = QToolButton(self.toolbar)
        self.more_button.setText("更多")
        self.more_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.more_menu = QMenu(self.more_button)
        self.more_button.setMenu(self.more_menu)
        layout.addWidget(self.more_button)
        layout.addSpacing(5)
        self.undo_button = QToolButton(self.toolbar)
        self.undo_button.setText("撤销")
        self.undo_button.clicked.connect(self.model.undo)
        layout.addWidget(self.undo_button)
        self.redo_button = QToolButton(self.toolbar)
        self.redo_button.setText("重做")
        self.redo_button.clicked.connect(self.model.redo)
        layout.addWidget(self.redo_button)
        self.zoom_button = QToolButton(self.toolbar)
        self.zoom_button.setText("视图")
        self.zoom_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        zoom_menu = QMenu(self.zoom_button)
        zoom_menu.addAction("放大", lambda: self.set_zoom(self._zoom * 1.25))
        zoom_menu.addAction("缩小", lambda: self.set_zoom(self._zoom / 1.25))
        zoom_menu.addAction("适合窗口", self.fit_to_window)
        zoom_menu.addAction("1:1", self.one_to_one)
        self.zoom_button.setMenu(zoom_menu)
        layout.addWidget(self.zoom_button)
        layout.addSpacing(5)
        self.copy_button = QPushButton("复制", self.toolbar)
        self.copy_button.clicked.connect(self.request_copy)
        layout.addWidget(self.copy_button)
        self.save_button = QToolButton(self.toolbar)
        self.save_button.setText("保存")
        self.save_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        save_menu = QMenu(self.save_button)
        save_menu.addAction("保存到默认目录", self.request_save)
        save_menu.addAction("另存为…", self.request_save_as)
        self.save_button.setMenu(save_menu)
        layout.addWidget(self.save_button)
        self.cancel_button = QPushButton("取消", self.toolbar)
        self.cancel_button.clicked.connect(self.request_cancel)
        layout.addWidget(self.cancel_button)
        self.finish_button = QPushButton("完成", self.toolbar)
        self.finish_button.setObjectName("finishButton")
        self.finish_button.setToolTip("按截图工具设置保存或复制（Enter）")
        self.finish_button.clicked.connect(self.request_complete)
        layout.addWidget(self.finish_button)

        self.status_label = QLabel(self)
        self.status_label.setStyleSheet(
            "QLabel { color: white; background: rgba(20,24,30,235);"
            " border: 1px solid rgba(255,255,255,38); border-radius: 5px; padding: 6px 10px; }"
        )
        self.status_label.hide()

    def _build_properties(self) -> None:
        self.properties = QFrame(self)
        self.properties.setObjectName("annotationProperties")
        self.properties.setStyleSheet(
            "QFrame#annotationProperties { background: rgba(38, 43, 52, 244);"
            " border: 1px solid rgba(255,255,255,35); border-radius: 7px; }"
            "QLabel, QCheckBox { color: #eef2f6; }"
            "QPushButton { color: #eef2f6; background: rgba(255,255,255,12);"
            " border: 1px solid rgba(255,255,255,28); border-radius: 4px; padding: 4px 8px; }"
        )
        layout = QHBoxLayout(self.properties)
        layout.setContentsMargins(8, 5, 8, 5)
        layout.setSpacing(6)
        self.color_button = QPushButton("描边", self.properties)
        self.color_button.clicked.connect(lambda: self._choose_color(fill=False))
        layout.addWidget(self.color_button)
        self.fill_button = QPushButton("填充：无", self.properties)
        self.fill_button.clicked.connect(lambda: self._choose_color(fill=True))
        self.fill_button.setToolTip("左键选择填充颜色；右键清除填充。实色填充可用于遮挡敏感信息。")
        self.fill_button.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.fill_button.customContextMenuRequested.connect(lambda _point: self.clear_fill())
        layout.addWidget(self.fill_button)
        self.width_label = QLabel("线宽", self.properties)
        layout.addWidget(self.width_label)
        self.width_spin = NoWheelSpinBox(self.properties)
        self.width_spin.setRange(1, 64)
        self.width_spin.setSuffix(" px")
        self.width_spin.valueChanged.connect(self._property_changed)
        layout.addWidget(self.width_spin)
        self.opacity_label = QLabel("透明度", self.properties)
        layout.addWidget(self.opacity_label)
        self.opacity_slider = QSlider(Qt.Orientation.Horizontal, self.properties)
        self.opacity_slider.setRange(5, 100)
        self.opacity_slider.setFixedWidth(86)
        self.opacity_slider.valueChanged.connect(self._property_changed)
        layout.addWidget(self.opacity_slider)
        self.strength_label = QLabel("强度", self.properties)
        layout.addWidget(self.strength_label)
        self.strength_spin = NoWheelSpinBox(self.properties)
        self.strength_spin.setRange(2, 96)
        self.strength_spin.valueChanged.connect(self._property_changed)
        layout.addWidget(self.strength_spin)
        self.arrow_label = QLabel("箭头", self.properties)
        layout.addWidget(self.arrow_label)
        self.arrow_spin = NoWheelSpinBox(self.properties)
        self.arrow_spin.setRange(4, 96)
        self.arrow_spin.setSuffix(" px")
        self.arrow_spin.valueChanged.connect(self._property_changed)
        layout.addWidget(self.arrow_spin)
        self.font_combo = QFontComboBox(self.properties)
        self.font_combo.setMinimumWidth(118)
        self.font_combo.currentFontChanged.connect(self._property_changed)
        layout.addWidget(self.font_combo)
        self.font_size_spin = NoWheelSpinBox(self.properties)
        self.font_size_spin.setRange(8, 160)
        self.font_size_spin.setSuffix(" px")
        self.font_size_spin.valueChanged.connect(self._property_changed)
        layout.addWidget(self.font_size_spin)
        self.bold_box = QCheckBox("粗体", self.properties)
        self.bold_box.toggled.connect(self._property_changed)
        layout.addWidget(self.bold_box)
        self.italic_box = QCheckBox("斜体", self.properties)
        self.italic_box.toggled.connect(self._property_changed)
        layout.addWidget(self.italic_box)
        self.number_label = QLabel("起始", self.properties)
        layout.addWidget(self.number_label)
        self.number_spin = NoWheelSpinBox(self.properties)
        self.number_spin.setRange(1, 9999)
        self.number_spin.valueChanged.connect(self._property_changed)
        layout.addWidget(self.number_spin)
        self.redaction_hint = QLabel("提示：模糊不是安全脱敏；敏感信息请使用实色填充矩形。", self.properties)
        self.redaction_hint.setStyleSheet("color: #ffcc80;")
        layout.addWidget(self.redaction_hint)
        self.selection_hint = QLabel("单击选择对象；Shift 可多选。", self.properties)
        layout.addWidget(self.selection_hint)
        self.crop_hint = QLabel("拖动选择保留区域；裁剪不会扩展到原截图之外。", self.properties)
        layout.addWidget(self.crop_hint)

    def begin(self) -> None:
        self.show()
        self.raise_()
        self.activateWindow()
        self.setFocus(Qt.FocusReason.ActiveWindowFocusReason)
        self._layout_floating_controls()

    def activate_session(self, message: str = "请先完成或取消当前标注。") -> None:
        self.show()
        self.raise_()
        self.activateWindow()
        self.show_status(message)

    def show_status(self, message: str, *, timeout_ms: int = 2600) -> None:
        self.status_label.setText(str(message))
        self.status_label.adjustSize()
        capture = self._display_capture_bounds()
        x = max(8, min(self.width() - self.status_label.width() - 8, round(capture.center().x() - self.status_label.width() / 2)))
        y = max(8, min(self.height() - self.status_label.height() - 8, round(capture.top() + 14)))
        self.status_label.move(x, y)
        self.status_label.show()
        self.status_label.raise_()
        self._status_timer.start(timeout_ms)

    def set_tool(self, tool: EditorTool | str) -> None:
        parsed = EditorTool.parse(tool)
        self.cancel_current_operation()
        self._tool = parsed
        self._styles["active_tool"] = parsed.value
        action = self._tool_actions.get(parsed)
        if action is not None:
            action.setChecked(True)
        self._apply_style_for_tool(parsed)
        self._update_property_visibility()
        self.setCursor(Qt.CursorShape.ArrowCursor if parsed is EditorTool.SELECT else Qt.CursorShape.CrossCursor)
        self.stylesChanged.emit(self._styles)
        if parsed in {EditorTool.BLUR, EditorTool.MOSAIC} and self.width() < 900:
            self.show_status("模糊与马赛克不是安全脱敏；敏感信息请使用实色填充矩形。")
        self.update()

    def _tool_style(self, tool: EditorTool | None = None) -> dict[str, object]:
        tools = self._styles.setdefault("tools", {})
        assert isinstance(tools, dict)
        return tools.setdefault((tool or self._tool).value, {})  # type: ignore[return-value]

    def _apply_style_for_tool(self, tool: EditorTool) -> None:
        style = self._tool_style(tool)
        widgets = (self.width_spin, self.opacity_slider, self.strength_spin, self.arrow_spin, self.font_size_spin, self.number_spin)
        for widget in widgets:
            widget.blockSignals(True)
        self.font_combo.blockSignals(True)
        self.bold_box.blockSignals(True)
        self.italic_box.blockSignals(True)
        self.width_spin.setValue(round(float(style.get("stroke_width", 3))))
        self.opacity_slider.setValue(round(float(style.get("opacity", 1.0)) * 100))
        self.strength_spin.setValue(int(style.get("block_size", 12)))
        self.arrow_spin.setValue(round(float(style.get("arrow_size", 12))))
        self.font_size_spin.setValue(int(style.get("font_size", 18)))
        number_value = int(style.get("number_start", 1))
        if tool is EditorTool.NUMBER:
            if not self._number_initialized:
                self._number = number_value
                self._number_initialized = True
            number_value = self._number
        self.number_spin.setValue(number_value)
        family = str(style.get("font_family", ""))
        if family:
            self.font_combo.setCurrentFont(QFont(family))
        self.bold_box.setChecked(bool(style.get("bold", False)))
        self.italic_box.setChecked(bool(style.get("italic", False)))
        for widget in widgets:
            widget.blockSignals(False)
        self.font_combo.blockSignals(False)
        self.bold_box.blockSignals(False)
        self.italic_box.blockSignals(False)
        self._refresh_color_buttons()

    def _property_changed(self, *_args: object) -> None:
        style = self._tool_style()
        style.update(
            stroke_width=self.width_spin.value(),
            opacity=self.opacity_slider.value() / 100.0,
            block_size=self.strength_spin.value(),
            arrow_size=self.arrow_spin.value(),
            font_family=self.font_combo.currentFont().family(),
            font_size=self.font_size_spin.value(),
            bold=self.bold_box.isChecked(),
            italic=self.italic_box.isChecked(),
            number_start=self.number_spin.value(),
        )
        if self._tool is EditorTool.NUMBER:
            self._number = self.number_spin.value()
        if self.model.selected_ids:
            selected = self.model.selected_commands
            if selected:
                changes: dict[str, object] = dict(
                    stroke_width=self.width_spin.value(),
                    opacity=self.opacity_slider.value() / 100.0,
                    block_size=self.strength_spin.value(),
                    arrow_size=self.arrow_spin.value(),
                    font_family=self.font_combo.currentFont().family(),
                    font_size=self.font_size_spin.value(),
                    bold=self.bold_box.isChecked(),
                    italic=self.italic_box.isChecked(),
                )
                if selected[0].tool is EditorTool.NUMBER:
                    changes["number"] = self.number_spin.value()
                self.model.update_selected(**changes)
        self.stylesChanged.emit(self._styles)
        self.update()

    def _choose_color(self, *, fill: bool) -> None:
        style = self._tool_style()
        key = self._secondary_color_key() if fill else "color"
        initial = QColor(str(style.get(key, "#e53935") or "#e53935"))
        selected = QColorDialog.getColor(
            initial,
            self,
            (
                "选择文字背景"
                if fill and key == "background_color"
                else "选择填充颜色"
                if fill
                else "选择标注颜色"
            ),
            QColorDialog.ColorDialogOption.ShowAlphaChannel,
        )
        if not selected.isValid():
            return
        style[key] = selected.name(QColor.NameFormat.HexArgb)
        if self.model.selected_ids:
            self.model.update_selected(**{key: style[key]})
        self._refresh_color_buttons()
        self.stylesChanged.emit(self._styles)

    def clear_fill(self) -> None:
        key = self._secondary_color_key()
        self._tool_style()[key] = ""
        if self.model.selected_ids:
            self.model.update_selected(**{key: ""})
        self._refresh_color_buttons()
        self.stylesChanged.emit(self._styles)

    def _refresh_color_buttons(self) -> None:
        style = self._tool_style()
        color = QColor(str(style.get("color", "#e53935")))
        key = self._secondary_color_key()
        fill = str(style.get(key, ""))
        self.color_button.setStyleSheet(f"background: {color.name(QColor.NameFormat.HexArgb)};")
        label = "背景" if key == "background_color" else "填充"
        self.fill_button.setText(label if fill else f"{label}：无")
        self.fill_button.setStyleSheet(f"background: {fill};" if fill else "")

    def _secondary_color_key(self) -> str:
        if self._tool is EditorTool.TEXT:
            return "background_color"
        if self._tool is EditorTool.SELECT:
            selected = self.model.selected_commands
            if len(selected) == 1 and selected[0].tool is EditorTool.TEXT:
                return "background_color"
        return "fill_color"

    def _update_property_visibility(self) -> None:
        selected = self.model.selected_commands
        effective = (
            selected[0].tool
            if self._tool is EditorTool.SELECT and len(selected) == 1
            else self._tool
        )
        shape = effective in {EditorTool.RECTANGLE, EditorTool.ELLIPSE}
        effect = effective in {EditorTool.MOSAIC, EditorTool.BLUR}
        text = effective is EditorTool.TEXT
        number = effective is EditorTool.NUMBER
        arrow = effective is EditorTool.ARROW
        selection_empty = self._tool is EditorTool.SELECT and not selected
        color_enabled = (
            not effect
            and effective is not EditorTool.CROP
            and not selection_empty
        )
        self.color_button.setVisible(color_enabled)
        self.fill_button.setVisible(shape or text)
        self.width_label.setVisible(color_enabled)
        self.width_spin.setVisible(color_enabled)
        self.opacity_label.setVisible(color_enabled)
        self.opacity_slider.setVisible(color_enabled)
        self.strength_label.setVisible(effect)
        self.strength_spin.setVisible(effect)
        self.arrow_label.setVisible(arrow)
        self.arrow_spin.setVisible(arrow)
        self.font_combo.setVisible(text)
        self.font_size_spin.setVisible(text or number)
        self.bold_box.setVisible(text)
        self.italic_box.setVisible(text)
        self.number_label.setVisible(number)
        self.number_spin.setVisible(number)
        self.redaction_hint.setVisible(
            self.width() >= 900
            and effective in {EditorTool.BLUR, EditorTool.MOSAIC, EditorTool.RECTANGLE}
        )
        self.selection_hint.setVisible(selection_empty)
        self.crop_hint.setVisible(effective is EditorTool.CROP)
        self.properties.adjustSize()
        self._layout_floating_controls()

    def _selection_changed(self, _ids: object) -> None:
        selected = self.model.selected_commands
        if self._tool is EditorTool.SELECT and len(selected) == 1:
            command = selected[0]
            tools = self._styles.setdefault("tools", {})
            assert isinstance(tools, dict)
            tools[EditorTool.SELECT.value] = {
                **self._tool_style(EditorTool.SELECT),
                "color": command.color,
                "fill_color": command.fill_color,
                "stroke_width": command.stroke_width,
                "opacity": command.opacity,
                "block_size": command.block_size,
                "arrow_size": command.arrow_size,
                "font_family": command.font_family,
                "font_size": command.font_size,
                "bold": command.bold,
                "italic": command.italic,
                "background_color": command.background_color,
                "number_start": command.number or 1,
            }
            self._apply_style_for_tool(EditorTool.SELECT)
        self._update_property_visibility()
        self.update()

    def _model_changed(self) -> None:
        self._layout_floating_controls()
        self.update()

    def _update_history_controls(self, can_undo: bool, can_redo: bool) -> None:
        self.undo_button.setEnabled(bool(can_undo))
        self.redo_button.setEnabled(bool(can_redo))
        action = self._compact_redo_action
        if action is not None:
            action.setEnabled(bool(can_redo))

    def _layout_floating_controls(self) -> None:
        capture = self._display_capture_bounds()
        available = self._control_available_rect().adjusted(8, 8, -8, -8)
        compact = available.width() < 760
        # Zooming can make the displayed capture wider than the desktop.  Base
        # overflow decisions on the usable control width as well, otherwise a
        # compact toolbar can unexpectedly expand and clip after zooming in.
        narrow = min(capture.width(), available.width()) < 920
        primary_tools = (
            {EditorTool.SELECT, EditorTool.RECTANGLE, EditorTool.ARROW, EditorTool.TEXT}
            if compact
            else self._PRIMARY_NARROW
        )
        self.more_menu.clear()
        self._compact_redo_action = None
        for tool, _label, _shortcut in self._TOOLS:
            button = self._tool_buttons[tool]
            visible = not narrow or tool in primary_tools
            button.setVisible(visible)
            if not visible:
                self.more_menu.addAction(self._tool_actions[tool])
        self.redo_button.setVisible(not compact)
        self.zoom_button.setVisible(not compact)
        self.save_button.setVisible(not compact)
        if compact:
            self.more_menu.addSeparator()
            self._compact_redo_action = self.more_menu.addAction(
                "重做", self.model.redo
            )
            self._compact_redo_action.setEnabled(self.model.can_redo)
            view_menu = self.more_menu.addMenu("视图")
            view_menu.addAction("放大", lambda: self.set_zoom(self._zoom * 1.25))
            view_menu.addAction("缩小", lambda: self.set_zoom(self._zoom / 1.25))
            view_menu.addAction("适合窗口", self.fit_to_window)
            view_menu.addAction("1:1", self.one_to_one)
            save_menu = self.more_menu.addMenu("保存")
            save_menu.addAction("保存到默认目录", self.request_save)
            save_menu.addAction("另存为…", self.request_save_as)
        self.more_button.setVisible(narrow)
        self.toolbar.adjustSize()
        self.properties.adjustSize()
        width = min(self.toolbar.width(), available.width())
        self.toolbar.resize(width, self.toolbar.height())
        x = round(capture.center().x() - width / 2)
        x = max(available.left(), min(available.right() - width, x))
        below = round(capture.bottom() + 10)
        above = round(capture.top() - self.toolbar.height() - 10)
        y = below if below + self.toolbar.height() <= available.bottom() else above
        y = max(available.top(), min(available.bottom() - self.toolbar.height(), y))
        self.toolbar.move(x, y)
        prop_width = min(self.properties.width(), available.width())
        self.properties.resize(prop_width, self.properties.height())
        prop_x = max(available.left(), min(available.right() - prop_width, round(capture.center().x() - prop_width / 2)))
        toolbar_below = y >= capture.bottom()
        if toolbar_below:
            candidate = y + self.toolbar.height() + 6
            prop_y = (
                candidate
                if candidate + self.properties.height() <= available.bottom()
                else round(capture.top() - self.properties.height() - 10)
            )
        else:
            candidate = y - self.properties.height() - 6
            prop_y = (
                candidate
                if candidate >= available.top()
                else y + self.toolbar.height() + 6
            )
        prop_y = max(available.top(), min(available.bottom() - self.properties.height(), prop_y))
        self.properties.move(prop_x, prop_y)
        self.toolbar.raise_()
        self.properties.raise_()

    def _control_available_rect(self) -> QRect:
        """Return the usable logical area of the capture's dominant screen."""

        visible = self.model.visible_rect
        desktop = CaptureRect(
            self.frame.rect.x + visible.x(),
            self.frame.rect.y + visible.y(),
            visible.width(),
            visible.height(),
        )
        target = max(
            self._screens,
            key=lambda item: (
                (desktop.intersection(item.physical_rect) or CaptureRect(0, 0, 0, 0)).area,
                1 if item.primary else 0,
            ),
            default=None,
        )
        if target is None:
            return self.rect()
        logical = target.logical_rect
        for qt_screen in QGuiApplication.screens():
            if qt_screen.name() == target.name:
                available = qt_screen.availableGeometry()
                logical = CaptureRect.from_qrect(available)
                break
        origin = self._mapping.virtual_logical_rect
        candidate = logical.translated(-origin.x, -origin.y).to_qrect()
        clipped = candidate.intersected(self.rect())
        return clipped if not clipped.isEmpty() else self.rect()

    def _view_point(self, raw: QPointF) -> QPointF:
        return QPointF(raw.x() * self._zoom + self._pan.x(), raw.y() * self._zoom + self._pan.y())

    def _raw_point(self, viewed: QPointF) -> QPointF:
        return QPointF((viewed.x() - self._pan.x()) / self._zoom, (viewed.y() - self._pan.y()) / self._zoom)

    def _view_rect(self, raw: QRectF) -> QRectF:
        return QRectF(self._view_point(raw.topLeft()), self._view_point(raw.bottomRight())).normalized()

    def _display_capture_bounds(self) -> QRectF:
        visible = self.model.visible_rect
        desktop = CaptureRect(
            self.frame.rect.x + visible.x(), self.frame.rect.y + visible.y(), visible.width(), visible.height()
        )
        logical = self._mapping.logical_capture_rect(desktop)
        raw = QRectF(
            logical.x - self._mapping.virtual_logical_rect.x,
            logical.y - self._mapping.virtual_logical_rect.y,
            logical.width,
            logical.height,
        )
        return self._view_rect(raw)

    def _widget_to_image(self, point: QPointF) -> QPointF | None:
        return self._mapping.widget_to_image(self._raw_point(point), self.model.visible_rect)

    def _image_to_widget(self, point: QPointF) -> QPointF | None:
        raw = self._mapping.image_to_widget(point, self.model.visible_rect)
        return self._view_point(raw) if raw is not None else None

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802 - Qt API
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(8, 12, 18, 112))
        rendered = self.model.render_cached()
        for source, destination in self._mapping.image_fragments(self.model.visible_rect):
            painter.drawImage(self._view_rect(destination), rendered, source)
        capture = self._display_capture_bounds()
        painter.setPen(QPen(QColor("#16d6c7"), 1.2))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(capture)
        self._paint_draft(painter)
        self._paint_selection(painter)
        painter.end()
        del event

    def _paint_draft(self, painter: QPainter) -> None:
        if len(self._points) < 2:
            return
        command = self._command_from_points(self._points, self._draft_modifiers)
        if command is None:
            return
        bounds = command_rect(command)
        if command.tool in {EditorTool.CROP, EditorTool.MOSAIC, EditorTool.BLUR}:
            widget_rect = self._image_rect_to_widget(bounds)
            painter.setPen(QPen(QColor("#2db4ff"), 1.5, Qt.PenStyle.DashLine))
            painter.setBrush(QColor(45, 180, 255, 35))
            painter.drawRect(widget_rect)
            return
        self._draw_local_command(painter, command)

    def _draw_local_command(self, painter: QPainter, command: EditCommand) -> None:
        """Draw one screenshot-local vector correctly across mixed-DPI screens."""

        for screen in self._screens:
            frame_fragment = self.frame.rect.intersection(screen.physical_rect)
            if frame_fragment is None:
                continue
            ratio_x = screen.logical_rect.width / max(1, screen.physical_rect.width)
            ratio_y = screen.logical_rect.height / max(1, screen.physical_rect.height)
            logical = screen.physical_fragment_to_logical(frame_fragment)
            raw_clip = QRectF(
                logical.x - self._mapping.virtual_logical_rect.x,
                logical.y - self._mapping.virtual_logical_rect.y,
                logical.width,
                logical.height,
            )
            painter.save()
            painter.setClipRect(self._view_rect(raw_clip))
            painter.translate(self._pan)
            painter.scale(self._zoom * ratio_x, self._zoom * ratio_y)
            painter.translate(
                (screen.logical_rect.x - self._mapping.virtual_logical_rect.x) / ratio_x - screen.physical_rect.x + self.frame.rect.x + self.model.visible_rect.x(),
                (screen.logical_rect.y - self._mapping.virtual_logical_rect.y) / ratio_y - screen.physical_rect.y + self.frame.rect.y + self.model.visible_rect.y(),
            )
            draw_edit_command(painter, command)
            painter.restore()

    def _image_rect_to_widget(self, rect: QRectF) -> QRectF:
        points = [
            self._image_to_widget(rect.topLeft()), self._image_to_widget(rect.topRight()),
            self._image_to_widget(rect.bottomLeft()), self._image_to_widget(rect.bottomRight()),
        ]
        valid = [point for point in points if point is not None]
        if not valid:
            return QRectF()
        left = min(point.x() for point in valid)
        top = min(point.y() for point in valid)
        right = max(point.x() for point in valid)
        bottom = max(point.y() for point in valid)
        return QRectF(left, top, right - left, bottom - top)

    def _paint_selection(self, painter: QPainter) -> None:
        if self._tool is not EditorTool.SELECT or not self.model.selected_ids:
            return
        bounds, previews = self._selection_preview()
        if self._drag_origin is not None and self._drag_current is not None:
            painter.setOpacity(0.72)
            for command in previews:
                self._draw_local_command(painter, command)
            painter.setOpacity(1.0)
        widget_rect = self._image_rect_to_widget(bounds)
        painter.setPen(QPen(QColor("#2db4ff"), 1.25, Qt.PenStyle.DashLine))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(widget_rect)
        painter.setBrush(QColor("white"))
        for point in self._selection_handles(widget_rect).values():
            painter.drawRect(QRectF(point.x() - 4, point.y() - 4, 8, 8))
        if len(previews) == 1 and previews[0].tool in {EditorTool.LINE, EditorTool.ARROW}:
            for value in (previews[0].points[0], previews[0].points[-1]):
                point = self._image_to_widget(QPointF(*value))
                if point is not None:
                    painter.drawEllipse(point, 5, 5)

    def _selection_preview(self) -> tuple[QRectF, tuple[EditCommand, ...]]:
        selected = self.model.selected_commands
        visible = self.model.visible_rect
        bounds = self.model.selection_bounds()
        previews = tuple(
            translated_command(command, -visible.x(), -visible.y())
            for command in selected
        )
        if self._drag_origin is None or self._drag_current is None:
            return bounds, previews
        if self._line_endpoint is not None and len(selected) == 1:
            identifier, endpoint = self._line_endpoint
            command = selected[0]
            points = list(command.points)
            if command.id == identifier and len(points) >= 2:
                points[0 if endpoint == 0 else -1] = (
                    self._drag_current[0] + visible.x(),
                    self._drag_current[1] + visible.y(),
                )
            preview = replace(command, points=tuple(points))
            bounds = command_rect(preview)
            bounds.translate(-visible.x(), -visible.y())
            return bounds, (
                translated_command(preview, -visible.x(), -visible.y()),
            )
        if self._resize_handle:
            target = selection_resize_target(
                bounds,
                self._resize_handle,
                self._drag_current,
                self._selection_modifiers,
            )
            absolute_target = QRectF(target)
            absolute_target.translate(visible.x(), visible.y())
            old_bounds = self.model.selection_bounds(local=False)
            previews = tuple(
                translated_command(
                    resized_command(command, old_bounds, absolute_target),
                    -visible.x(),
                    -visible.y(),
                )
                for command in selected
            )
            return target, previews
        dx = self._drag_current[0] - self._drag_origin[0]
        dy = self._drag_current[1] - self._drag_origin[1]
        bounds.translate(dx, dy)
        return bounds, tuple(
            translated_command(command, -visible.x() + dx, -visible.y() + dy)
            for command in selected
        )

    @staticmethod
    def _selection_handles(bounds: QRectF) -> dict[str, QPointF]:
        return {
            "nw": bounds.topLeft(), "n": QPointF(bounds.center().x(), bounds.top()), "ne": bounds.topRight(),
            "e": QPointF(bounds.right(), bounds.center().y()), "se": bounds.bottomRight(),
            "s": QPointF(bounds.center().x(), bounds.bottom()), "sw": bounds.bottomLeft(),
            "w": QPointF(bounds.left(), bounds.center().y()),
        }

    def _handle_at(self, point: QPointF) -> str:
        bounds = self._image_rect_to_widget(self.model.selection_bounds())
        for name, center in self._selection_handles(bounds).items():
            if math.hypot(point.x() - center.x(), point.y() - center.y()) <= 8:
                return name
        return ""

    def _endpoint_at(self, point: QPointF) -> tuple[str, int] | None:
        selected = self.model.selected_commands
        if len(selected) != 1 or selected[0].tool not in {EditorTool.LINE, EditorTool.ARROW}:
            return None
        offset = self.model.visible_rect.topLeft()
        for index, value in ((0, selected[0].points[0]), (-1, selected[0].points[-1])):
            widget = self._image_to_widget(QPointF(value[0] - offset.x(), value[1] - offset.y()))
            if widget is not None and math.hypot(point.x() - widget.x(), point.y() - widget.y()) <= 9:
                return selected[0].id, index
        return None

    def _command_from_points(
        self,
        points: Sequence[tuple[float, float]],
        modifiers: Qt.KeyboardModifier = Qt.KeyboardModifier.NoModifier,
    ) -> EditCommand | None:
        if not points:
            return None
        style = self._tool_style()
        common = {
            "color": str(style.get("color", "#e53935")),
            "fill_color": str(style.get("fill_color", "")),
            "stroke_width": float(style.get("stroke_width", 3)),
            "opacity": float(style.get("opacity", 1.0)),
            "block_size": int(style.get("block_size", 12)),
            "arrow_size": float(style.get("arrow_size", 12)),
            "font_family": str(style.get("font_family", "")),
            "font_size": int(style.get("font_size", 18)),
            "bold": bool(style.get("bold", False)),
            "italic": bool(style.get("italic", False)),
            "background_color": str(style.get("background_color", "")),
        }
        if self._tool is EditorTool.PEN:
            return EditCommand(self._tool, points=tuple(points), **common)
        if self._tool is EditorTool.NUMBER:
            return EditCommand(self._tool, points=(points[0],), number=self._number, **common)
        if len(points) < 2:
            return None
        start, end = points[0], points[-1]
        if modifiers & Qt.KeyboardModifier.ShiftModifier:
            if self._tool in {EditorTool.LINE, EditorTool.ARROW}:
                length = math.hypot(end[0] - start[0], end[1] - start[1])
                angle = round(math.atan2(end[1] - start[1], end[0] - start[0]) / (math.pi / 4)) * math.pi / 4
                end = (start[0] + length * math.cos(angle), start[1] + length * math.sin(angle))
            elif self._tool in {EditorTool.RECTANGLE, EditorTool.ELLIPSE}:
                size = max(abs(end[0] - start[0]), abs(end[1] - start[1]))
                end = (start[0] + math.copysign(size, end[0] - start[0] or 1), start[1] + math.copysign(size, end[1] - start[1] or 1))
        if modifiers & Qt.KeyboardModifier.ControlModifier and self._tool not in {EditorTool.LINE, EditorTool.ARROW}:
            dx, dy = end[0] - start[0], end[1] - start[1]
            start = (start[0] - dx, start[1] - dy)
        return EditCommand.from_drag(self._tool, start, end, **common)

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt API
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        if event.button() == Qt.MouseButton.MiddleButton or (
            event.button() == Qt.MouseButton.LeftButton and self._space_pan
        ):
            self._pan_origin = event.globalPosition().toPoint()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        if event.button() == Qt.MouseButton.RightButton:
            if self.cancel_current_operation():
                event.accept()
                return
            if self.model.selected_ids:
                self._show_object_menu(event.globalPosition().toPoint())
                event.accept()
                return
        if event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return
        image = self._widget_to_image(event.position())
        if image is None:
            event.accept()
            return
        point = (image.x(), image.y())
        if self._tool is EditorTool.SELECT:
            self._selection_modifiers = event.modifiers()
            endpoint = self._endpoint_at(event.position())
            handle = self._handle_at(event.position()) if endpoint is None else ""
            if endpoint is not None:
                self._line_endpoint = endpoint
                self._drag_origin = point
                self._drag_current = point
            elif handle:
                self._resize_handle = handle
                self._drag_origin = point
                self._drag_current = point
            else:
                hit = self.model.select_at(point, additive=bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier))
                if hit is not None and hit.id in self.model.selected_ids:
                    self._drag_origin = point
                    self._drag_current = point
            self.update()
            event.accept()
            return
        if self._tool is EditorTool.TEXT:
            self._begin_text_edit(image)
            event.accept()
            return
        self._points = [point]
        self._draft_modifiers = event.modifiers()
        event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt API
        if self._pan_origin is not None:
            current = event.globalPosition().toPoint()
            delta = current - self._pan_origin
            self._pan_origin = current
            self._pan += QPointF(delta)
            self._layout_floating_controls()
            self.update()
            event.accept()
            return
        image = self._widget_to_image(event.position())
        if image is None:
            return
        current = (image.x(), image.y())
        if self._tool is EditorTool.SELECT and self._drag_origin is not None:
            self._drag_current = current
            self._selection_modifiers = event.modifiers()
            self.update()
            event.accept()
            return
        if self._points and event.buttons() & Qt.MouseButton.LeftButton:
            self._draft_modifiers = event.modifiers()
            if self._tool is EditorTool.PEN:
                self._points.append(current)
            elif len(self._points) == 1:
                self._points.append(current)
            else:
                self._points[-1] = current
            self.update()
            event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt API
        if self._pan_origin is not None:
            self._pan_origin = None
            self.setCursor(Qt.CursorShape.ArrowCursor if self._tool is EditorTool.SELECT else Qt.CursorShape.CrossCursor)
            event.accept()
            return
        image = self._widget_to_image(event.position())
        if image is None:
            self.cancel_current_operation()
            return
        current = (image.x(), image.y())
        if self._tool is EditorTool.SELECT and self._drag_origin is not None and event.button() == Qt.MouseButton.LeftButton:
            if self._line_endpoint is not None:
                self.model.set_line_endpoint(self._line_endpoint[0], self._line_endpoint[1], current)
            elif self._resize_handle:
                self._finish_resize(current, event.modifiers())
            else:
                self.model.move_selected(current[0] - self._drag_origin[0], current[1] - self._drag_origin[1])
            self._drag_origin = None
            self._drag_current = None
            self._selection_modifiers = Qt.KeyboardModifier.NoModifier
            self._resize_handle = ""
            self._line_endpoint = None
            self.update()
            event.accept()
            return
        if event.button() != Qt.MouseButton.LeftButton or not self._points:
            return
        if self._tool is not EditorTool.NUMBER:
            if len(self._points) == 1:
                self._points.append(current)
            else:
                self._points[-1] = current
        command = self._command_from_points(self._points, event.modifiers())
        self._points = []
        self._draft_modifiers = Qt.KeyboardModifier.NoModifier
        if command is not None and self.model.add_command(command):
            if command.tool is EditorTool.NUMBER:
                self._number += 1
                self.number_spin.blockSignals(True)
                self.number_spin.setValue(self._number)
                self.number_spin.blockSignals(False)
        self._layout_floating_controls()
        self.update()
        event.accept()

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt API
        image = self._widget_to_image(event.position())
        if image is not None:
            command = self.model.hit_test(image)
            if command is not None and command.tool is EditorTool.TEXT:
                self.model.set_selection((command.id,))
                self._begin_text_edit(image, command=command)
                event.accept()
                return
        super().mouseDoubleClickEvent(event)

    def _finish_resize(self, current: tuple[float, float], modifiers: Qt.KeyboardModifier) -> None:
        bounds = self.model.selection_bounds()
        if bounds.isEmpty():
            return
        target = selection_resize_target(
            bounds,
            self._resize_handle,
            current,
            modifiers,
        )
        if target.width() >= 1 and target.height() >= 1:
            self.model.resize_selected(target)

    def _begin_text_edit(self, point: QPointF, command: EditCommand | None = None) -> None:
        self._finish_text_edit(cancel=True)
        editor = InlineTextEdit(self)
        editor.setPlaceholderText("输入文字；Shift+Enter 换行")
        editor.setStyleSheet(
            "QPlainTextEdit { background: rgba(20,24,30,238); color: white;"
            " border: 1px solid #2db4ff; border-radius: 4px; padding: 5px; }"
        )
        style = dict(self._tool_style(EditorTool.TEXT))
        target = command_rect(command) if command is not None else QRectF(point.x(), point.y(), 260, 90)
        if command is not None:
            offset = self.model.visible_rect.topLeft()
            target.translate(-offset.x(), -offset.y())
            editor.setPlainText(command.text)
            self._editing_text_id = command.id
            style.update(
                color=command.color,
                opacity=command.opacity,
                font_family=command.font_family,
                font_size=command.font_size,
                bold=command.bold,
                italic=command.italic,
                background_color=command.background_color,
            )
        else:
            self._editing_text_id = ""
        self._editing_text_style = style
        top_left = self._image_to_widget(target.topLeft()) or QPointF(20, 20)
        bottom_right = self._image_to_widget(target.bottomRight()) or QPointF(top_left.x() + 260, top_left.y() + 90)
        geometry = QRectF(top_left, bottom_right).normalized().toAlignedRect()
        geometry.setWidth(max(190, geometry.width()))
        geometry.setHeight(max(74, geometry.height()))
        capture_bounds = self._display_capture_bounds().toAlignedRect().intersected(
            self.rect().adjusted(4, 4, -4, -4)
        )
        if capture_bounds.isEmpty():
            capture_bounds = self.rect().adjusted(4, 4, -4, -4)
        geometry.setWidth(min(geometry.width(), capture_bounds.width()))
        geometry.setHeight(min(geometry.height(), capture_bounds.height()))
        if geometry.right() > capture_bounds.right():
            geometry.moveRight(capture_bounds.right())
        if geometry.bottom() > capture_bounds.bottom():
            geometry.moveBottom(capture_bounds.bottom())
        if geometry.left() < capture_bounds.left():
            geometry.moveLeft(capture_bounds.left())
        if geometry.top() < capture_bounds.top():
            geometry.moveTop(capture_bounds.top())
        editor.setGeometry(geometry)
        font = QFont(str(style.get("font_family", ""))) if style.get("font_family") else QFont()
        font.setPixelSize(max(10, round(int(style.get("font_size", 18)) * self._zoom)))
        font.setBold(bool(style.get("bold", False)))
        font.setItalic(bool(style.get("italic", False)))
        editor.setFont(font)
        editor.submitted.connect(self._commit_text_edit)
        editor.cancelled.connect(lambda: self._finish_text_edit(cancel=True))
        self._text_edit = editor
        editor.show()
        editor.raise_()
        editor.setFocus(Qt.FocusReason.MouseFocusReason)
        editor.selectAll()

    def _commit_text_edit(self) -> None:
        editor = self._text_edit
        if editor is None:
            return
        text = editor.toPlainText().rstrip()
        top_left = self._widget_to_image(QPointF(editor.geometry().topLeft()))
        bottom_right = self._widget_to_image(QPointF(editor.geometry().bottomRight()))
        if top_left is None or bottom_right is None:
            self.show_status("文字区域超出截图，已保留输入供调整。")
            return
        rect = QRectF(top_left, bottom_right).normalized()
        style = self._editing_text_style or self._tool_style(EditorTool.TEXT)
        command = EditCommand(
            EditorTool.TEXT,
            points=((rect.left(), rect.top() + int(style.get("font_size", 18))),),
            rect=(rect.x(), rect.y(), rect.width(), rect.height()),
            text=text,
            color=str(style.get("color", "#e53935")),
            opacity=float(style.get("opacity", 1.0)),
            font_family=str(style.get("font_family", "")),
            font_size=int(style.get("font_size", 18)),
            bold=bool(style.get("bold", False)),
            italic=bool(style.get("italic", False)),
            background_color=str(style.get("background_color", "")),
        )
        if self._editing_text_id:
            if text:
                visible = self.model.visible_rect
                self.model.replace_command(
                    self._editing_text_id,
                    replace(
                        command,
                        points=tuple(
                            (x + visible.x(), y + visible.y())
                            for x, y in command.points
                        ),
                        rect=(
                            command.rect[0] + visible.x(),
                            command.rect[1] + visible.y(),
                            command.rect[2],
                            command.rect[3],
                        ) if command.rect is not None else None,
                    ),
                )
            else:
                self.model.set_selection((self._editing_text_id,))
                self.model.delete_selected()
        elif text:
            self.model.add_command(command)
        self._finish_text_edit(cancel=False)

    def _finish_text_edit(self, *, cancel: bool) -> bool:
        editor = self._text_edit
        if editor is None:
            return False
        self._text_edit = None
        self._editing_text_id = ""
        self._editing_text_style = None
        editor.hide()
        editor.deleteLater()
        self.setFocus(Qt.FocusReason.OtherFocusReason)
        del cancel
        return True

    def cancel_current_operation(self) -> bool:
        if self._text_edit is not None:
            return self._finish_text_edit(cancel=True)
        if self._points or self._drag_origin is not None:
            self._points = []
            self._draft_modifiers = Qt.KeyboardModifier.NoModifier
            self._drag_origin = None
            self._drag_current = None
            self._selection_modifiers = Qt.KeyboardModifier.NoModifier
            self._resize_handle = ""
            self._line_endpoint = None
            self.update()
            return True
        return False

    def _show_object_menu(self, global_point: QPoint) -> None:
        menu = QMenu(self)
        duplicate = menu.addAction("复制对象")
        front = menu.addAction("置于顶层")
        forward = menu.addAction("上移一层")
        backward = menu.addAction("下移一层")
        back = menu.addAction("置于底层")
        menu.addSeparator()
        remove = menu.addAction("删除")
        chosen = menu.exec(global_point)
        if chosen is duplicate:
            self.model.duplicate_selected()
        elif chosen is front:
            self.model.bring_to_front()
        elif chosen is forward:
            self.model.bring_forward()
        elif chosen is backward:
            self.model.send_backward()
        elif chosen is back:
            self.model.send_to_back()
        elif chosen is remove:
            self.model.delete_selected()

    def set_zoom(self, zoom: float, anchor: QPointF | None = None) -> None:
        target = max(0.1, min(8.0, float(zoom)))
        if abs(target - self._zoom) < 1e-6:
            return
        anchor = anchor or self._display_capture_bounds().center()
        raw = self._raw_point(anchor)
        self._zoom = target
        self._pan = QPointF(anchor.x() - raw.x() * target, anchor.y() - raw.y() * target)
        self._layout_floating_controls()
        self.update()

    def fit_to_window(self) -> None:
        self._zoom = 1.0
        self._pan = QPointF()
        bounds = self._display_capture_bounds()
        available = self.rect().adjusted(24, 24, -24, -110)
        if bounds.width() > 0 and bounds.height() > 0:
            factor = min(1.0, available.width() / bounds.width(), available.height() / bounds.height())
            self.set_zoom(factor, bounds.center())

    def one_to_one(self) -> None:
        self._zoom = 1.0
        self._pan = QPointF()
        self._layout_floating_controls()
        self.update()

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802 - Qt API
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.set_zoom(self._zoom * (1.15 if event.angleDelta().y() > 0 else 1 / 1.15), event.position())
            event.accept()
            return
        super().wheelEvent(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802 - Qt API
        if self._text_edit is not None:
            super().keyPressEvent(event)
            return
        modifiers = event.modifiers()
        control = bool(modifiers & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.MetaModifier))
        if control and event.key() == Qt.Key.Key_Z:
            (self.model.redo if modifiers & Qt.KeyboardModifier.ShiftModifier else self.model.undo)()
            event.accept()
            return
        if control and event.key() == Qt.Key.Key_Y:
            self.model.redo()
            event.accept()
            return
        if control and event.key() == Qt.Key.Key_D:
            self.model.duplicate_selected()
            event.accept()
            return
        if event.key() == Qt.Key.Key_Space:
            self._space_pan = True
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            event.accept()
            return
        if event.key() == Qt.Key.Key_Delete:
            self.model.delete_selected()
            event.accept()
            return
        if event.key() in {Qt.Key.Key_Left, Qt.Key.Key_Right, Qt.Key.Key_Up, Qt.Key.Key_Down} and self.model.selected_ids:
            amount = 10 if modifiers & Qt.KeyboardModifier.ShiftModifier else 1
            dx = -amount if event.key() == Qt.Key.Key_Left else amount if event.key() == Qt.Key.Key_Right else 0
            dy = -amount if event.key() == Qt.Key.Key_Up else amount if event.key() == Qt.Key.Key_Down else 0
            self.model.move_selected(dx, dy)
            event.accept()
            return
        if event.key() in {Qt.Key.Key_Return, Qt.Key.Key_Enter}:
            self.request_complete()
            event.accept()
            return
        if event.key() == Qt.Key.Key_Escape:
            if self.cancel_current_operation():
                event.accept()
                return
            if self.model.selected_ids:
                self.model.clear_selection()
                event.accept()
                return
            self.request_cancel()
            event.accept()
            return
        if not control and not modifiers & Qt.KeyboardModifier.AltModifier and event.key() in self._SHORTCUTS:
            self.set_tool(self._SHORTCUTS[event.key()])
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent) -> None:  # noqa: N802 - Qt API
        if event.key() == Qt.Key.Key_Space:
            self._space_pan = False
            if self._pan_origin is None:
                self.setCursor(Qt.CursorShape.ArrowCursor if self._tool is EditorTool.SELECT else Qt.CursorShape.CrossCursor)
            event.accept()
            return
        super().keyReleaseEvent(event)

    def request_complete(self) -> None:
        if self._text_edit is not None:
            self._commit_text_edit()
            if self._text_edit is not None:
                return
        if not self._begin_output():
            return
        self.completed.emit(self.model.render())

    def request_copy(self) -> None:
        if not self._begin_output():
            return
        self.copyRequested.emit(self.model.render())

    def request_save(self) -> None:
        if not self._begin_output():
            return
        self.saveRequested.emit(self.model.render())

    def request_save_as(self) -> None:
        if self._output_pending:
            self.show_status("输出处理中，请稍候。")
            return
        path, _filter = QFileDialog.getSaveFileName(
            self,
            "另存截图为",
            str(Path.home() / "screenshot.png"),
            "PNG 图片 (*.png);;JPEG 图片 (*.jpg *.jpeg);;WebP 图片 (*.webp)",
        )
        if not path:
            return
        if not self._begin_output():
            return
        self.saveAsRequested.emit(self.model.render(), path)

    def _begin_output(self) -> bool:
        if self._output_pending:
            self.show_status("输出处理中，请稍候。")
            return False
        self._output_pending = True
        self.toolbar.setEnabled(False)
        self.properties.setEnabled(False)
        self.show_status("正在输出截图…", timeout_ms=10_000)
        return True

    def output_succeeded(self) -> None:
        self._output_pending = False
        self.close()

    def output_failed(self, message: str) -> None:
        self._output_pending = False
        self.toolbar.setEnabled(True)
        self.properties.setEnabled(True)
        self.show_status(f"输出失败：{message}", timeout_ms=5000)

    def request_cancel(self) -> None:
        if self._output_pending:
            self.show_status("输出处理中，请稍候。")
            return
        if self.model.has_annotations:
            answer = QMessageBox.question(self, "取消截图", "已有标注，确认放弃本次截图吗？")
            if answer != QMessageBox.StandardButton.Yes:
                return
        self.cancelled.emit()
        self.close()

    def _check_topology(self) -> None:
        if self._screens_provider is None or self._output_pending:
            return
        try:
            current = tuple(self._screens_provider())
        except Exception:
            return
        if screen_topology_signature(current) == self._topology_signature:
            return
        self._topology_timer.stop()
        self.hide()
        self.model.setParent(None)
        self.fallbackRequested.emit(self.model)

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API
        self._topology_timer.stop()
        super().closeEvent(event)


__all__ = [
    "CaptureViewportMapping",
    "InlineAnnotationOverlay",
    "screen_topology_signature",
]
