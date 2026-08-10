from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import math
from pathlib import Path
from typing import Iterable, Sequence

from PySide6.QtCore import QObject, QPointF, QRect, QRectF, QSize, Qt, Signal
from PySide6.QtGui import (
    QAction,
    QActionGroup,
    QColor,
    QGuiApplication,
    QImage,
    QKeySequence,
    QMouseEvent,
    QPaintEvent,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QPolygonF,
)
from PySide6.QtWidgets import QFileDialog, QMainWindow, QScrollArea, QToolBar, QWidget
from PySide6.QtWidgets import QColorDialog, QInputDialog, QLabel, QPushButton, QSpinBox


class EditorTool(str, Enum):
    RECTANGLE = "rectangle"
    ELLIPSE = "ellipse"
    ARROW = "arrow"
    LINE = "line"
    PEN = "pen"
    TEXT = "text"
    NUMBER = "number"
    HIGHLIGHT = "highlight"
    MOSAIC = "mosaic"
    BLUR = "blur"
    CROP = "crop"

    @classmethod
    def parse(cls, value: object) -> "EditorTool":
        if isinstance(value, cls):
            return value
        return cls(str(value).strip().lower().replace("-", "_"))


PointTuple = tuple[float, float]
RectTuple = tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class EditCommand:
    tool: EditorTool
    points: tuple[PointTuple, ...] = ()
    rect: RectTuple | None = None
    text: str = ""
    color: str = "#e53935"
    fill_color: str = ""
    stroke_width: float = 3.0
    opacity: float = 1.0
    number: int = 0
    block_size: int = 10

    def __post_init__(self) -> None:
        object.__setattr__(self, "tool", EditorTool.parse(self.tool))
        object.__setattr__(self, "points", tuple((float(x), float(y)) for x, y in self.points))
        if self.rect is not None:
            object.__setattr__(self, "rect", tuple(float(value) for value in self.rect))
        object.__setattr__(self, "stroke_width", max(0.5, float(self.stroke_width)))
        object.__setattr__(self, "opacity", min(1.0, max(0.0, float(self.opacity))))
        object.__setattr__(self, "block_size", max(2, int(self.block_size)))

    @classmethod
    def from_drag(
        cls,
        tool: EditorTool,
        start: PointTuple,
        end: PointTuple,
        **kwargs: object,
    ) -> "EditCommand":
        x1, y1 = start
        x2, y2 = end
        normalized = (min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))
        if tool in {EditorTool.LINE, EditorTool.ARROW}:
            return cls(tool=tool, points=(start, end), **kwargs)
        return cls(tool=tool, points=(start, end), rect=normalized, **kwargs)

    def to_dict(self) -> dict[str, object]:
        return {
            "tool": self.tool.value,
            "points": [list(point) for point in self.points],
            "rect": list(self.rect) if self.rect is not None else None,
            "text": self.text,
            "color": self.color,
            "fill_color": self.fill_color,
            "stroke_width": self.stroke_width,
            "opacity": self.opacity,
            "number": self.number,
            "block_size": self.block_size,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "EditCommand":
        raw_points = payload.get("points", ())
        points = tuple(
            (float(item[0]), float(item[1]))
            for item in raw_points
            if isinstance(item, (list, tuple)) and len(item) >= 2
        ) if isinstance(raw_points, (list, tuple)) else ()
        raw_rect = payload.get("rect")
        rect = (
            tuple(float(value) for value in raw_rect[:4])
            if isinstance(raw_rect, (list, tuple)) and len(raw_rect) >= 4
            else None
        )
        return cls(
            tool=EditorTool.parse(payload.get("tool")),
            points=points,
            rect=rect,  # type: ignore[arg-type]
            text=str(payload.get("text", "")),
            color=str(payload.get("color", "#e53935")),
            fill_color=str(payload.get("fill_color", "")),
            stroke_width=float(payload.get("stroke_width", 3.0)),
            opacity=float(payload.get("opacity", 1.0)),
            number=int(payload.get("number", 0)),
            block_size=int(payload.get("block_size", 10)),
        )


def _command_rect(command: EditCommand) -> QRectF:
    if command.rect is not None:
        return QRectF(*command.rect).normalized()
    if not command.points:
        return QRectF()
    xs = [point[0] for point in command.points]
    ys = [point[1] for point in command.points]
    return QRectF(min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)).normalized()


def _translated_command(command: EditCommand, dx: float, dy: float) -> EditCommand:
    points = tuple((x + float(dx), y + float(dy)) for x, y in command.points)
    rect = command.rect
    translated_rect = (
        (rect[0] + float(dx), rect[1] + float(dy), rect[2], rect[3])
        if rect is not None
        else None
    )
    return replace(command, points=points, rect=translated_rect)


def _color(value: str, opacity: float = 1.0) -> QColor:
    color = QColor(value)
    if not color.isValid():
        color = QColor("#e53935")
    color.setAlphaF(min(1.0, max(0.0, color.alphaF() * opacity)))
    return color


def _draw_arrow(painter: QPainter, start: QPointF, end: QPointF, width: float) -> None:
    painter.drawLine(start, end)
    angle = math.atan2(end.y() - start.y(), end.x() - start.x())
    size = max(8.0, width * 4.0)
    left = QPointF(
        end.x() - size * math.cos(angle - math.pi / 6.0),
        end.y() - size * math.sin(angle - math.pi / 6.0),
    )
    right = QPointF(
        end.x() - size * math.cos(angle + math.pi / 6.0),
        end.y() - size * math.sin(angle + math.pi / 6.0),
    )
    painter.setBrush(painter.pen().color())
    painter.drawPolygon(QPolygonF((end, left, right)))


def _pixelate(image: QImage, rect: QRectF, block_size: int, *, smooth: bool) -> None:
    clipped = rect.toAlignedRect().intersected(image.rect())
    if clipped.isEmpty():
        return
    source = image.copy(clipped)
    width = max(1, source.width() // max(2, block_size))
    height = max(1, source.height() // max(2, block_size))
    down_mode = Qt.TransformationMode.SmoothTransformation if smooth else Qt.TransformationMode.FastTransformation
    reduced = source.scaled(width, height, Qt.AspectRatioMode.IgnoreAspectRatio, down_mode)
    enlarged = reduced.scaled(
        source.size(),
        Qt.AspectRatioMode.IgnoreAspectRatio,
        Qt.TransformationMode.SmoothTransformation if smooth else Qt.TransformationMode.FastTransformation,
    )
    painter = QPainter(image)
    painter.drawImage(clipped.topLeft(), enlarged)
    painter.end()


def render_edit_commands(base: QImage, commands: Iterable[EditCommand]) -> QImage:
    image = base.convertToFormat(QImage.Format.Format_ARGB32_Premultiplied).copy()
    normalized = tuple(commands)
    crop: QRectF | None = None
    for command in normalized:
        if command.tool is EditorTool.CROP:
            crop = _command_rect(command)
            continue
        if command.tool in {EditorTool.MOSAIC, EditorTool.BLUR}:
            _pixelate(
                image,
                _command_rect(command),
                command.block_size * (2 if command.tool is EditorTool.BLUR else 1),
                smooth=command.tool is EditorTool.BLUR,
            )
            continue

        painter = QPainter(image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        color = _color(command.color, command.opacity)
        pen = QPen(color, command.stroke_width, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        rect = _command_rect(command)
        points = tuple(QPointF(x, y) for x, y in command.points)

        if command.tool is EditorTool.RECTANGLE:
            if command.fill_color:
                painter.setBrush(_color(command.fill_color, command.opacity))
            painter.drawRect(rect)
        elif command.tool is EditorTool.ELLIPSE:
            if command.fill_color:
                painter.setBrush(_color(command.fill_color, command.opacity))
            painter.drawEllipse(rect)
        elif command.tool is EditorTool.LINE and len(points) >= 2:
            painter.drawLine(points[0], points[-1])
        elif command.tool is EditorTool.ARROW and len(points) >= 2:
            _draw_arrow(painter, points[0], points[-1], command.stroke_width)
        elif command.tool is EditorTool.PEN and len(points) >= 2:
            path = QPainterPath(points[0])
            for point in points[1:]:
                path.lineTo(point)
            painter.drawPath(path)
        elif command.tool is EditorTool.TEXT:
            origin = points[0] if points else rect.topLeft()
            font = painter.font()
            font.setPixelSize(max(12, round(command.stroke_width * 6)))
            painter.setFont(font)
            painter.drawText(origin, command.text)
        elif command.tool is EditorTool.NUMBER:
            center = points[0] if points else rect.center()
            radius = max(10.0, command.stroke_width * 4.0)
            badge = QRectF(center.x() - radius, center.y() - radius, radius * 2, radius * 2)
            painter.setBrush(color)
            painter.drawEllipse(badge)
            painter.setPen(QPen(Qt.GlobalColor.white, max(1.0, command.stroke_width / 2)))
            painter.drawText(badge, Qt.AlignmentFlag.AlignCenter, str(command.number or 1))
        elif command.tool is EditorTool.HIGHLIGHT:
            highlight = _color(command.color or "#fff176", min(command.opacity, 0.45))
            if rect.width() > 1 and rect.height() > 1:
                painter.fillRect(rect, highlight)
            elif len(points) >= 2:
                painter.setPen(QPen(highlight, max(8.0, command.stroke_width * 4), Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
                painter.drawLine(points[0], points[-1])
        painter.end()

    if crop is not None:
        aligned = crop.toAlignedRect().intersected(image.rect())
        if not aligned.isEmpty():
            image = image.copy(aligned)
    return image


class ScreenshotEditModel(QObject):
    """Small immutable-command history shared by the editor UI and tests."""

    changed = Signal()
    historyChanged = Signal(bool, bool)

    def __init__(self, image: QImage, parent: QObject | None = None) -> None:
        super().__init__(parent)
        if image.isNull():
            raise ValueError("编辑器不能打开空截图。")
        self._base_image = image.copy()
        self._commands: tuple[EditCommand, ...] = ()
        self._undo: list[tuple[EditCommand, ...]] = []
        self._redo: list[tuple[EditCommand, ...]] = []

    @property
    def base_image(self) -> QImage:
        return self._base_image.copy()

    @property
    def commands(self) -> tuple[EditCommand, ...]:
        return self._commands

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    def _commit(self, commands: Sequence[EditCommand]) -> None:
        state = tuple(commands)
        if state == self._commands:
            return
        self._undo.append(self._commands)
        self._commands = state
        self._redo.clear()
        self.changed.emit()
        self.historyChanged.emit(self.can_undo, self.can_redo)

    def _effective_crop_rect(self, command: EditCommand | None) -> QRect:
        if command is None or command.tool is not EditorTool.CROP:
            return self._base_image.rect()
        return _command_rect(command).toAlignedRect().intersected(
            self._base_image.rect()
        )

    def add_command(self, command: EditCommand) -> None:
        crop = next(
            (item for item in self._commands if item.tool is EditorTool.CROP),
            None,
        )
        if command.tool is EditorTool.CROP:
            visible = self._effective_crop_rect(crop)
            local_bounds = QRect(0, 0, visible.width(), visible.height())
            local_crop = _command_rect(command).toAlignedRect().intersected(
                local_bounds
            )
            # A click without a drag, or a release wholly outside the canvas,
            # is not a crop.  Keeping an empty crop would make rendering ignore
            # it while still offsetting every later annotation.
            if visible.isEmpty() or local_crop.isEmpty():
                return
            absolute = local_crop.translated(visible.topLeft()).intersected(
                self._base_image.rect()
            )
            if absolute.isEmpty():
                return
            normalized_crop = replace(
                command,
                points=(),
                rect=(
                    float(absolute.x()),
                    float(absolute.y()),
                    float(absolute.width()),
                    float(absolute.height()),
                ),
            )
            self._commit(
                (
                    *[
                        item
                        for item in self._commands
                        if item.tool is not EditorTool.CROP
                    ],
                    normalized_crop,
                )
            )
            return
        if crop is not None:
            # Once the canvas is cropped, pointer coordinates are relative to
            # the visible cropped image. Persist commands in the original base
            # image coordinate space so render order remains annotations first,
            # crop last.
            visible = self._effective_crop_rect(crop)
            command = _translated_command(command, visible.x(), visible.y())
        crop_commands = tuple(item for item in self._commands if item.tool is EditorTool.CROP)
        annotations = tuple(item for item in self._commands if item.tool is not EditorTool.CROP)
        self._commit((*annotations, command, *crop_commands))

    def set_crop(self, rect: RectTuple | QRectF) -> None:
        values = (
            (rect.x(), rect.y(), rect.width(), rect.height())
            if isinstance(rect, QRectF)
            else tuple(float(value) for value in rect)
        )
        self.add_command(EditCommand(EditorTool.CROP, rect=values))  # type: ignore[arg-type]

    def clear(self) -> None:
        self._commit(())

    def undo(self) -> bool:
        if not self._undo:
            return False
        self._redo.append(self._commands)
        self._commands = self._undo.pop()
        self.changed.emit()
        self.historyChanged.emit(self.can_undo, self.can_redo)
        return True

    def redo(self) -> bool:
        if not self._redo:
            return False
        self._undo.append(self._commands)
        self._commands = self._redo.pop()
        self.changed.emit()
        self.historyChanged.emit(self.can_undo, self.can_redo)
        return True

    def render(self) -> QImage:
        return render_edit_commands(self._base_image, self._commands)


class EditorCanvas(QWidget):
    commandCommitted = Signal(object)

    def __init__(self, model: ScreenshotEditModel, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._model = model
        self._tool = EditorTool.RECTANGLE
        self._color = "#e53935"
        self._stroke_width = 3.0
        self._pending_text = "文字"
        self._number = 1
        self._points: list[PointTuple] = []
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self._model.changed.connect(self._sync_size)
        self._sync_size()

    @property
    def tool(self) -> EditorTool:
        return self._tool

    @property
    def pending_text(self) -> str:
        return self._pending_text

    def set_tool(self, tool: EditorTool | str) -> None:
        self._tool = EditorTool.parse(tool)

    def set_pending_text(self, text: str) -> None:
        self._pending_text = str(text)

    def set_color(self, color: str | QColor) -> None:
        parsed = QColor(color)
        if parsed.isValid():
            self._color = parsed.name(QColor.NameFormat.HexArgb if parsed.alpha() < 255 else QColor.NameFormat.HexRgb)

    def set_stroke_width(self, width: float) -> None:
        self._stroke_width = max(0.5, float(width))

    def _sync_size(self) -> None:
        size = self._model.render().size()
        self.setMinimumSize(size)
        self.resize(size)
        self.update()

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt API
        return self._model.render().size()

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802 - Qt API
        del event
        painter = QPainter(self)
        painter.drawImage(0, 0, self._model.render())
        if len(self._points) >= 2:
            draft = self._command_from_points(self._points)
            if draft is not None:
                if draft.tool in {
                    EditorTool.CROP,
                    EditorTool.MOSAIC,
                    EditorTool.BLUR,
                }:
                    # These tools modify existing pixels, so rendering them on
                    # a transparent scratch image produces no visible draft.
                    # Keep a clear selection outline under the pointer instead.
                    painter.setPen(
                        QPen(
                            QColor("#2db4ff"),
                            1.5,
                            Qt.PenStyle.DashLine,
                        )
                    )
                    painter.setBrush(QColor(45, 180, 255, 32))
                    painter.drawRect(_command_rect(draft))
                else:
                    preview_base = QImage(
                        self.size(),
                        QImage.Format.Format_ARGB32_Premultiplied,
                    )
                    preview_base.fill(Qt.GlobalColor.transparent)
                    preview = render_edit_commands(preview_base, (draft,))
                    painter.drawImage(0, 0, preview)

    def _command_from_points(self, points: Sequence[PointTuple]) -> EditCommand | None:
        if not points:
            return None
        common = {"color": self._color, "stroke_width": self._stroke_width}
        if self._tool is EditorTool.PEN:
            return EditCommand(self._tool, points=tuple(points), **common)
        if self._tool is EditorTool.TEXT:
            return EditCommand(self._tool, points=(points[0],), text=self._pending_text, **common)
        if self._tool is EditorTool.NUMBER:
            return EditCommand(self._tool, points=(points[0],), number=self._number, **common)
        if len(points) < 2:
            return None
        extra: dict[str, object] = {}
        if self._tool is EditorTool.HIGHLIGHT:
            extra.update(color="#fff176", opacity=0.45)
        return EditCommand.from_drag(self._tool, points[0], points[-1], **common, **extra)

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt API
        if event.button() == Qt.MouseButton.LeftButton:
            point = event.position()
            self._points = [(point.x(), point.y())]
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt API
        if self._points and event.buttons() & Qt.MouseButton.LeftButton:
            point = event.position()
            current = (point.x(), point.y())
            if self._tool is EditorTool.PEN:
                self._points.append(current)
            elif len(self._points) == 1:
                self._points.append(current)
            else:
                self._points[-1] = current
            self.update()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt API
        if event.button() != Qt.MouseButton.LeftButton or not self._points:
            super().mouseReleaseEvent(event)
            return
        point = event.position()
        current = (point.x(), point.y())
        if self._tool not in {EditorTool.TEXT, EditorTool.NUMBER}:
            if len(self._points) == 1:
                self._points.append(current)
            elif self._points[-1] != current:
                self._points[-1] = current
        command = self._command_from_points(self._points)
        self._points = []
        if command is not None:
            self._model.add_command(command)
            if command.tool is EditorTool.NUMBER:
                self._number += 1
            self.commandCommitted.emit(command)
        self.update()
        event.accept()


class ScreenshotEditor(QMainWindow):
    saved = Signal(str)
    copied = Signal()
    completed = Signal(object)

    _TOOL_LABELS = {
        EditorTool.RECTANGLE: "矩形",
        EditorTool.ELLIPSE: "椭圆",
        EditorTool.ARROW: "箭头",
        EditorTool.LINE: "线",
        EditorTool.PEN: "画笔",
        EditorTool.TEXT: "文字",
        EditorTool.NUMBER: "编号",
        EditorTool.HIGHLIGHT: "高亮",
        EditorTool.MOSAIC: "马赛克",
        EditorTool.BLUR: "模糊",
        EditorTool.CROP: "裁剪",
    }

    def __init__(self, image: QImage, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Fiber Screenshot Tool")
        self.model = ScreenshotEditModel(image, self)
        self.canvas = EditorCanvas(self.model)
        scroll = QScrollArea(self)
        scroll.setWidget(self.canvas)
        scroll.setWidgetResizable(False)
        scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setCentralWidget(scroll)
        self.tool_actions: dict[EditorTool, QAction] = {}
        self._current_color = QColor("#e53935")
        self._build_toolbar()
        self.model.historyChanged.connect(self._update_history_actions)
        self.resize(min(1100, image.width() + 80), min(800, image.height() + 100))

    def _build_toolbar(self) -> None:
        toolbar = QToolBar("截图标注", self)
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        tools = QActionGroup(self)
        tools.setExclusive(True)
        for tool, label in self._TOOL_LABELS.items():
            action = QAction(label, self)
            action.setCheckable(True)
            action.setData(tool.value)
            action.triggered.connect(lambda _checked=False, item=tool: self.set_tool(item))
            tools.addAction(action)
            toolbar.addAction(action)
            self.tool_actions[tool] = action
        self.tool_actions[EditorTool.RECTANGLE].setChecked(True)
        toolbar.addSeparator()

        self.color_button = QPushButton("颜色", self)
        self.color_button.setToolTip("设置后续标注颜色")
        self.color_button.clicked.connect(self._choose_color)
        toolbar.addWidget(self.color_button)
        self.width_spin = QSpinBox(self)
        self.width_spin.setRange(1, 24)
        self.width_spin.setValue(3)
        self.width_spin.setSuffix(" px")
        self.width_spin.setToolTip("设置后续标注线宽")
        self.width_spin.valueChanged.connect(self.canvas.set_stroke_width)
        toolbar.addWidget(QLabel("线宽", self))
        toolbar.addWidget(self.width_spin)
        self._refresh_color_button()
        toolbar.addSeparator()

        self.undo_action = QAction("撤销", self)
        self.undo_action.setShortcut(QKeySequence.StandardKey.Undo)
        self.undo_action.triggered.connect(self.undo)
        toolbar.addAction(self.undo_action)
        self.redo_action = QAction("重做", self)
        self.redo_action.setShortcut(QKeySequence.StandardKey.Redo)
        self.redo_action.triggered.connect(self.redo)
        toolbar.addAction(self.redo_action)
        toolbar.addSeparator()

        complete_action = QAction("完成", self)
        complete_action.setToolTip("按截图工具设置执行保存和复制到剪贴板")
        complete_action.triggered.connect(self.complete)
        toolbar.addAction(complete_action)
        save_action = QAction("另存为", self)
        save_action.setShortcut(QKeySequence.StandardKey.Save)
        save_action.triggered.connect(lambda: self.save())
        toolbar.addAction(save_action)
        copy_action = QAction("复制", self)
        copy_action.setShortcut(QKeySequence.StandardKey.Copy)
        copy_action.triggered.connect(self.copy_to_clipboard)
        toolbar.addAction(copy_action)
        self._update_history_actions(False, False)

    def set_tool(self, tool: EditorTool | str) -> None:
        parsed = EditorTool.parse(tool)
        previous = self.canvas.tool
        if parsed is EditorTool.TEXT:
            text, accepted = QInputDialog.getText(
                self,
                "截图文字",
                "文字内容",
                text=self.canvas.pending_text,
            )
            if not accepted or not text.strip():
                old_action = self.tool_actions.get(previous)
                if old_action is not None:
                    old_action.setChecked(True)
                return
            self.canvas.set_pending_text(text.strip())
        self.canvas.set_tool(parsed)
        action = self.tool_actions.get(parsed)
        if action is not None:
            action.setChecked(True)

    def add_command(self, command: EditCommand) -> None:
        self.model.add_command(command)

    def undo(self) -> bool:
        return self.model.undo()

    def redo(self) -> bool:
        return self.model.redo()

    def edited_image(self) -> QImage:
        return self.model.render()

    def complete(self) -> None:
        self.completed.emit(self.edited_image())

    def _choose_color(self) -> None:
        selected = QColorDialog.getColor(
            self._current_color,
            self,
            "选择标注颜色",
            QColorDialog.ColorDialogOption.ShowAlphaChannel,
        )
        if not selected.isValid():
            return
        self._current_color = selected
        self.canvas.set_color(selected)
        self._refresh_color_button()

    def _refresh_color_button(self) -> None:
        foreground = "#000000" if self._current_color.lightnessF() > 0.62 else "#ffffff"
        self.color_button.setStyleSheet(
            "QPushButton {"
            f"background: {self._current_color.name(QColor.NameFormat.HexArgb)};"
            f"color: {foreground};"
            "padding: 3px 10px; border: 1px solid palette(mid);"
            "}"
        )

    def save(self, path: str | Path | None = None) -> bool:
        target = str(path) if path is not None else ""
        if not target:
            target, _selected_filter = QFileDialog.getSaveFileName(
                self,
                "保存截图",
                "screenshot.png",
                "PNG 图片 (*.png);;JPEG 图片 (*.jpg *.jpeg);;BMP 图片 (*.bmp)",
            )
        if not target:
            return False
        suffix = Path(target).suffix
        if not suffix:
            target += ".png"
        success = self.edited_image().save(target)
        if success:
            self.saved.emit(target)
        return bool(success)

    def copy_to_clipboard(self) -> bool:
        app = QGuiApplication.instance()
        if app is None:
            return False
        app.clipboard().setPixmap(QPixmap.fromImage(self.edited_image()))
        self.copied.emit()
        return True

    def _update_history_actions(self, can_undo: bool, can_redo: bool) -> None:
        self.undo_action.setEnabled(bool(can_undo))
        self.redo_action.setEnabled(bool(can_redo))


__all__ = [
    "EditCommand",
    "EditorCanvas",
    "EditorTool",
    "ScreenshotEditModel",
    "ScreenshotEditor",
    "render_edit_commands",
]
