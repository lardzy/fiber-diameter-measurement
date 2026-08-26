from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence
from uuid import uuid4

from PySide6.QtCore import QObject, QPoint, QPointF, QRect, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import (
    QAction,
    QActionGroup,
    QColor,
    QFont,
    QFontMetricsF,
    QGuiApplication,
    QImage,
    QKeyEvent,
    QKeySequence,
    QMouseEvent,
    QPaintEvent,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QPolygonF,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QColorDialog,
    QFileDialog,
    QFontComboBox,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QToolBar,
    QToolButton,
    QWidget,
)


class EditorTool(str, Enum):
    SELECT = "select"
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
    """One immutable annotation object in screenshot physical-pixel space."""

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
    arrow_size: float = 12.0
    font_family: str = ""
    font_size: int = 18
    bold: bool = False
    italic: bool = False
    background_color: str = ""
    id: str = field(default_factory=lambda: uuid4().hex)

    def __post_init__(self) -> None:
        object.__setattr__(self, "tool", EditorTool.parse(self.tool))
        object.__setattr__(
            self,
            "points",
            tuple((float(x), float(y)) for x, y in self.points),
        )
        if self.rect is not None:
            values = tuple(float(value) for value in self.rect[:4])
            object.__setattr__(self, "rect", values if len(values) == 4 else None)
        object.__setattr__(self, "stroke_width", max(0.5, float(self.stroke_width)))
        object.__setattr__(self, "opacity", min(1.0, max(0.0, float(self.opacity))))
        object.__setattr__(self, "block_size", max(2, min(96, int(self.block_size))))
        object.__setattr__(self, "arrow_size", max(4.0, min(96.0, float(self.arrow_size))))
        object.__setattr__(self, "font_size", max(8, min(160, int(self.font_size))))
        object.__setattr__(self, "number", max(0, int(self.number)))
        object.__setattr__(self, "id", str(self.id or uuid4().hex))

    @classmethod
    def from_drag(
        cls,
        tool: EditorTool,
        start: PointTuple,
        end: PointTuple,
        **kwargs: object,
    ) -> "EditCommand":
        parsed = EditorTool.parse(tool)
        x1, y1 = start
        x2, y2 = end
        normalized = (min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))
        if parsed in {EditorTool.LINE, EditorTool.ARROW}:
            return cls(tool=parsed, points=(start, end), **kwargs)
        return cls(tool=parsed, points=(start, end), rect=normalized, **kwargs)

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
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
            "arrow_size": self.arrow_size,
            "font_family": self.font_family,
            "font_size": self.font_size,
            "bold": self.bold,
            "italic": self.italic,
            "background_color": self.background_color,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "EditCommand":
        raw_points = payload.get("points", ())
        points = (
            tuple(
                (float(item[0]), float(item[1]))
                for item in raw_points
                if isinstance(item, (list, tuple)) and len(item) >= 2
            )
            if isinstance(raw_points, (list, tuple))
            else ()
        )
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
            arrow_size=float(payload.get("arrow_size", 12.0)),
            font_family=str(payload.get("font_family", "")),
            font_size=int(payload.get("font_size", 18)),
            bold=bool(payload.get("bold", False)),
            italic=bool(payload.get("italic", False)),
            background_color=str(payload.get("background_color", "")),
            id=str(payload.get("id", "") or uuid4().hex),
        )


def _font_for(command: EditCommand) -> QFont:
    font = QFont(command.font_family) if command.font_family else QFont()
    font.setPixelSize(command.font_size)
    font.setBold(command.bold)
    font.setItalic(command.italic)
    return font


def command_rect(command: EditCommand) -> QRectF:
    if command.rect is not None:
        return QRectF(*command.rect).normalized()
    if command.tool is EditorTool.NUMBER and command.points:
        radius = max(10.0, command.font_size * 0.7, command.stroke_width * 4.0)
        x, y = command.points[0]
        return QRectF(x - radius, y - radius, radius * 2, radius * 2)
    if command.tool is EditorTool.TEXT and command.points:
        font = _font_for(command)
        metrics = QFontMetricsF(font)
        lines = command.text.splitlines() or [""]
        width = max((metrics.horizontalAdvance(line or " ") for line in lines), default=1.0)
        height = max(metrics.height(), metrics.lineSpacing() * len(lines))
        x, y = command.points[0]
        return QRectF(x, y - metrics.ascent(), width + 6, height + 4)
    if not command.points:
        return QRectF()
    xs = [point[0] for point in command.points]
    ys = [point[1] for point in command.points]
    return QRectF(
        min(xs),
        min(ys),
        max(xs) - min(xs),
        max(ys) - min(ys),
    ).normalized()


# Backwards-compatible private name used by older integrations.
_command_rect = command_rect


def translated_command(command: EditCommand, dx: float, dy: float) -> EditCommand:
    points = tuple((x + float(dx), y + float(dy)) for x, y in command.points)
    rect = command.rect
    translated_rect = (
        (rect[0] + float(dx), rect[1] + float(dy), rect[2], rect[3])
        if rect is not None
        else None
    )
    return replace(command, points=points, rect=translated_rect)


_translated_command = translated_command


def resized_command(
    command: EditCommand,
    old_bounds: QRectF,
    new_bounds: QRectF,
) -> EditCommand:
    old = old_bounds.normalized()
    new = new_bounds.normalized()
    sx = new.width() / old.width() if old.width() > 1e-6 else 1.0
    sy = new.height() / old.height() if old.height() > 1e-6 else 1.0

    def map_point(point: PointTuple) -> PointTuple:
        return (
            new.left() + (point[0] - old.left()) * sx,
            new.top() + (point[1] - old.top()) * sy,
        )

    points = tuple(map_point(point) for point in command.points)
    rect = command.rect
    mapped_rect = None
    if rect is not None:
        source = QRectF(*rect).normalized()
        top_left = map_point((source.left(), source.top()))
        bottom_right = map_point((source.right(), source.bottom()))
        mapped = QRectF(QPointF(*top_left), QPointF(*bottom_right)).normalized()
        mapped_rect = (mapped.x(), mapped.y(), mapped.width(), mapped.height())
    width_scale = max(0.25, math.sqrt(abs(sx * sy)))
    return replace(
        command,
        points=points,
        rect=mapped_rect,
        stroke_width=max(0.5, command.stroke_width * width_scale),
        font_size=max(8, round(command.font_size * width_scale)),
    )


def selection_resize_target(
    bounds: QRectF,
    handle: str,
    current: PointTuple,
    modifiers: Qt.KeyboardModifier,
) -> QRectF:
    """Build a constrained local selection rectangle for handle previews."""

    source = QRectF(bounds).normalized()
    left, top, right, bottom = (
        source.left(),
        source.top(),
        source.right(),
        source.bottom(),
    )
    if "w" in handle:
        left = current[0]
    if "e" in handle:
        right = current[0]
    if "n" in handle:
        top = current[1]
    if "s" in handle:
        bottom = current[1]
    target = QRectF(QPointF(left, top), QPointF(right, bottom)).normalized()
    if modifiers & Qt.KeyboardModifier.ShiftModifier and source.height() > 0:
        ratio = source.width() / source.height()
        if target.width() / max(1e-6, target.height()) > ratio:
            target.setHeight(target.width() / ratio)
        else:
            target.setWidth(target.height() * ratio)
    if modifiers & Qt.KeyboardModifier.ControlModifier:
        center = source.center()
        dx = max(abs(target.left() - center.x()), abs(target.right() - center.x()))
        dy = max(abs(target.top() - center.y()), abs(target.bottom() - center.y()))
        target = QRectF(center.x() - dx, center.y() - dy, dx * 2, dy * 2)
    return target


def _color(value: str, opacity: float = 1.0) -> QColor:
    color = QColor(value)
    if not color.isValid():
        color = QColor("#e53935")
    color.setAlphaF(min(1.0, max(0.0, color.alphaF() * opacity)))
    return color


def _draw_arrow(
    painter: QPainter,
    start: QPointF,
    end: QPointF,
    width: float,
    arrow_size: float = 12.0,
) -> None:
    painter.drawLine(start, end)
    angle = math.atan2(end.y() - start.y(), end.x() - start.x())
    size = max(8.0, float(arrow_size), width * 3.0)
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


def draw_edit_command(painter: QPainter, command: EditCommand) -> None:
    """Draw a vector command on an already active painter."""

    if command.tool in {EditorTool.SELECT, EditorTool.CROP, EditorTool.MOSAIC, EditorTool.BLUR}:
        return
    painter.save()
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    color = _color(command.color, command.opacity)
    painter.setPen(
        QPen(
            color,
            command.stroke_width,
            Qt.PenStyle.SolidLine,
            Qt.PenCapStyle.RoundCap,
            Qt.PenJoinStyle.RoundJoin,
        )
    )
    painter.setBrush(Qt.BrushStyle.NoBrush)
    rect = command_rect(command)
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
        _draw_arrow(
            painter,
            points[0],
            points[-1],
            command.stroke_width,
            command.arrow_size,
        )
    elif command.tool is EditorTool.PEN and len(points) >= 2:
        path = QPainterPath(points[0])
        for point in points[1:]:
            path.lineTo(point)
        painter.drawPath(path)
    elif command.tool is EditorTool.TEXT:
        painter.setFont(_font_for(command))
        if command.background_color:
            painter.fillRect(rect.adjusted(-3, -2, 3, 2), _color(command.background_color, command.opacity))
        if command.rect is not None:
            painter.drawText(
                rect,
                int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop | Qt.TextFlag.TextWordWrap),
                command.text,
            )
        else:
            origin = points[0] if points else rect.bottomLeft()
            painter.drawText(origin, command.text)
    elif command.tool is EditorTool.NUMBER:
        center = points[0] if points else rect.center()
        painter.setBrush(color)
        painter.drawEllipse(rect)
        font = _font_for(command)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QPen(Qt.GlobalColor.white, max(1.0, command.stroke_width / 2)))
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, str(command.number or 1))
    elif command.tool is EditorTool.HIGHLIGHT:
        highlight = _color(command.color or "#fff176", min(command.opacity, 0.55))
        if rect.width() > 1 and rect.height() > 1:
            painter.fillRect(rect, highlight)
        elif len(points) >= 2:
            painter.setPen(
                QPen(
                    highlight,
                    max(8.0, command.stroke_width),
                    Qt.PenStyle.SolidLine,
                    Qt.PenCapStyle.RoundCap,
                )
            )
            painter.drawLine(points[0], points[-1])
    painter.restore()


def _pixelate(image: QImage, rect: QRectF, block_size: int) -> None:
    clipped = rect.toAlignedRect().intersected(image.rect())
    if clipped.isEmpty():
        return
    source = image.copy(clipped)
    width = max(1, source.width() // max(2, block_size))
    height = max(1, source.height() // max(2, block_size))
    reduced = source.scaled(
        width,
        height,
        Qt.AspectRatioMode.IgnoreAspectRatio,
        Qt.TransformationMode.FastTransformation,
    )
    enlarged = reduced.scaled(
        source.size(),
        Qt.AspectRatioMode.IgnoreAspectRatio,
        Qt.TransformationMode.FastTransformation,
    )
    painter = QPainter(image)
    painter.drawImage(clipped.topLeft(), enlarged)
    painter.end()


def _gaussian_blur(image: QImage, rect: QRectF, strength: int) -> None:
    # Keep the resident screenshot agent light when annotation is disabled;
    # the relatively large image-processing modules are needed only when the
    # user actually commits a blur command.
    import cv2
    import numpy as np

    clipped = rect.toAlignedRect().intersected(image.rect())
    if clipped.isEmpty():
        return
    source = image.copy(clipped).convertToFormat(QImage.Format.Format_RGBA8888)
    bytes_per_line = source.bytesPerLine()
    array = np.frombuffer(source.constBits(), dtype=np.uint8, count=source.sizeInBytes())
    array = array.reshape((source.height(), bytes_per_line))[:, : source.width() * 4]
    rgba = array.reshape((source.height(), source.width(), 4))
    kernel = max(3, int(strength) * 2 + 1)
    if kernel % 2 == 0:
        kernel += 1
    blurred = cv2.GaussianBlur(
        rgba,
        (kernel, kernel),
        sigmaX=0,
        sigmaY=0,
        borderType=cv2.BORDER_REFLECT_101,
    )
    result = QImage(
        blurred.data,
        source.width(),
        source.height(),
        int(blurred.strides[0]),
        QImage.Format.Format_RGBA8888,
    ).copy()
    painter = QPainter(image)
    painter.drawImage(clipped.topLeft(), result)
    painter.end()


def render_edit_commands(base: QImage, commands: Iterable[EditCommand]) -> QImage:
    image = base.convertToFormat(QImage.Format.Format_ARGB32_Premultiplied).copy()
    crop: QRectF | None = None
    for command in tuple(commands):
        if command.tool is EditorTool.CROP:
            crop = command_rect(command)
            continue
        if command.tool is EditorTool.MOSAIC:
            _pixelate(image, command_rect(command), command.block_size)
            continue
        if command.tool is EditorTool.BLUR:
            _gaussian_blur(image, command_rect(command), command.block_size)
            continue
        painter = QPainter(image)
        draw_edit_command(painter, command)
        painter.end()
    if crop is not None:
        aligned = crop.toAlignedRect().intersected(image.rect())
        if not aligned.isEmpty():
            image = image.copy(aligned)
    return image


def _distance_to_segment(point: QPointF, start: QPointF, end: QPointF) -> float:
    dx = end.x() - start.x()
    dy = end.y() - start.y()
    denominator = dx * dx + dy * dy
    if denominator <= 1e-12:
        return math.hypot(point.x() - start.x(), point.y() - start.y())
    t = max(0.0, min(1.0, ((point.x() - start.x()) * dx + (point.y() - start.y()) * dy) / denominator))
    closest_x = start.x() + t * dx
    closest_y = start.y() + t * dy
    return math.hypot(point.x() - closest_x, point.y() - closest_y)


def command_hit_test(command: EditCommand, point: QPointF, tolerance: float = 6.0) -> bool:
    rect = command_rect(command)
    expanded = rect.adjusted(-tolerance, -tolerance, tolerance, tolerance)
    if command.tool in {EditorTool.LINE, EditorTool.ARROW, EditorTool.PEN}:
        points = [QPointF(*item) for item in command.points]
        return any(
            _distance_to_segment(point, start, end) <= tolerance + command.stroke_width / 2
            for start, end in zip(points, points[1:])
        )
    if command.tool is EditorTool.ELLIPSE and rect.width() > 0 and rect.height() > 0:
        cx, cy = rect.center().x(), rect.center().y()
        rx = max(1.0, rect.width() / 2)
        ry = max(1.0, rect.height() / 2)
        normalized = ((point.x() - cx) / rx) ** 2 + ((point.y() - cy) / ry) ** 2
        if command.fill_color:
            return normalized <= 1.0
        band = max(0.05, tolerance / max(rx, ry))
        return 1.0 - band <= normalized <= 1.0 + band
    if command.tool is EditorTool.RECTANGLE and not command.fill_color:
        inner = rect.adjusted(tolerance, tolerance, -tolerance, -tolerance)
        return expanded.contains(point) and (inner.isEmpty() or not inner.contains(point))
    return expanded.contains(point)


class ScreenshotEditModel(QObject):
    """Immutable object history with selection and a committed-render cache."""

    changed = Signal()
    historyChanged = Signal(bool, bool)
    selectionChanged = Signal(object)

    def __init__(self, image: QImage, parent: QObject | None = None) -> None:
        super().__init__(parent)
        if image.isNull():
            raise ValueError("编辑器不能打开空截图。")
        self._base_image = image.copy()
        self._commands: tuple[EditCommand, ...] = ()
        self._selected_ids: tuple[str, ...] = ()
        self._undo: list[tuple[EditCommand, ...]] = []
        self._redo: list[tuple[EditCommand, ...]] = []
        self._render_cache: QImage | None = None
        self._render_count = 0

    @property
    def base_image(self) -> QImage:
        return self._base_image.copy()

    @property
    def commands(self) -> tuple[EditCommand, ...]:
        return self._commands

    @property
    def selected_ids(self) -> tuple[str, ...]:
        return self._selected_ids

    @property
    def selected_commands(self) -> tuple[EditCommand, ...]:
        selected = set(self._selected_ids)
        return tuple(item for item in self._commands if item.id in selected)

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    @property
    def has_annotations(self) -> bool:
        return bool(self._commands)

    @property
    def render_count(self) -> int:
        return self._render_count

    @property
    def visible_rect(self) -> QRect:
        crop = next((item for item in self._commands if item.tool is EditorTool.CROP), None)
        return self._effective_crop_rect(crop)

    def _invalidate_render(self) -> None:
        self._render_cache = None

    def _emit_state(self) -> None:
        self._invalidate_render()
        valid = {item.id for item in self._commands}
        selection = tuple(item for item in self._selected_ids if item in valid)
        selection_changed = selection != self._selected_ids
        self._selected_ids = selection
        self.changed.emit()
        self.historyChanged.emit(self.can_undo, self.can_redo)
        if selection_changed:
            self.selectionChanged.emit(self._selected_ids)

    def _commit(self, commands: Sequence[EditCommand]) -> bool:
        state = tuple(commands)
        if state == self._commands:
            return False
        self._undo.append(self._commands)
        self._commands = state
        self._redo.clear()
        self._emit_state()
        return True

    def _effective_crop_rect(self, command: EditCommand | None) -> QRect:
        if command is None or command.tool is not EditorTool.CROP:
            return self._base_image.rect()
        return command_rect(command).toAlignedRect().intersected(self._base_image.rect())

    def add_command(self, command: EditCommand, *, select: bool = False) -> bool:
        crop = next((item for item in self._commands if item.tool is EditorTool.CROP), None)
        if command.tool is EditorTool.CROP:
            visible = self._effective_crop_rect(crop)
            local_bounds = QRect(0, 0, visible.width(), visible.height())
            local_crop = command_rect(command).toAlignedRect().intersected(local_bounds)
            if visible.isEmpty() or local_crop.isEmpty():
                return False
            absolute = local_crop.translated(visible.topLeft()).intersected(self._base_image.rect())
            if absolute.isEmpty():
                return False
            normalized_crop = replace(
                command,
                points=(),
                rect=(float(absolute.x()), float(absolute.y()), float(absolute.width()), float(absolute.height())),
            )
            changed = self._commit(
                (*[item for item in self._commands if item.tool is not EditorTool.CROP], normalized_crop)
            )
            if changed:
                self.clear_selection()
            return changed
        if command.tool is EditorTool.SELECT:
            return False
        if crop is not None:
            visible = self._effective_crop_rect(crop)
            command = translated_command(command, visible.x(), visible.y())
        crop_commands = tuple(item for item in self._commands if item.tool is EditorTool.CROP)
        annotations = tuple(item for item in self._commands if item.tool is not EditorTool.CROP)
        changed = self._commit((*annotations, command, *crop_commands))
        if changed and select:
            self.set_selection((command.id,))
        return changed

    def set_crop(self, rect: RectTuple | QRectF) -> bool:
        values = (
            (rect.x(), rect.y(), rect.width(), rect.height())
            if isinstance(rect, QRectF)
            else tuple(float(value) for value in rect)
        )
        return self.add_command(EditCommand(EditorTool.CROP, rect=values))  # type: ignore[arg-type]

    def clear(self) -> None:
        self._commit(())

    def undo(self) -> bool:
        if not self._undo:
            return False
        self._redo.append(self._commands)
        self._commands = self._undo.pop()
        self._emit_state()
        return True

    def redo(self) -> bool:
        if not self._redo:
            return False
        self._undo.append(self._commands)
        self._commands = self._redo.pop()
        self._emit_state()
        return True

    def render_cached(self) -> QImage:
        if self._render_cache is None:
            self._render_cache = render_edit_commands(self._base_image, self._commands)
            self._render_count += 1
        return self._render_cache

    def render(self) -> QImage:
        return self.render_cached().copy()

    def set_selection(self, identifiers: Iterable[str]) -> None:
        valid = {item.id for item in self._commands if item.tool is not EditorTool.CROP}
        selected = tuple(dict.fromkeys(str(item) for item in identifiers if str(item) in valid))
        if selected == self._selected_ids:
            return
        self._selected_ids = selected
        self.selectionChanged.emit(self._selected_ids)
        self.changed.emit()

    def clear_selection(self) -> None:
        self.set_selection(())

    def hit_test(self, point: PointTuple | QPointF, tolerance: float = 6.0) -> EditCommand | None:
        target = point if isinstance(point, QPointF) else QPointF(*point)
        offset = self.visible_rect.topLeft()
        absolute = target + QPointF(offset)
        for command in reversed(self._commands):
            if command.tool is EditorTool.CROP:
                continue
            if command_hit_test(command, absolute, tolerance):
                return command
        return None

    def select_at(self, point: PointTuple | QPointF, *, additive: bool = False) -> EditCommand | None:
        command = self.hit_test(point)
        if command is None:
            if not additive:
                self.clear_selection()
            return None
        if additive:
            selected = list(self._selected_ids)
            if command.id in selected:
                selected.remove(command.id)
            else:
                selected.append(command.id)
            self.set_selection(selected)
        else:
            self.set_selection((command.id,))
        return command

    def selection_bounds(self, *, local: bool = True) -> QRectF:
        selected = self.selected_commands
        if not selected:
            return QRectF()
        result = command_rect(selected[0])
        for command in selected[1:]:
            result = result.united(command_rect(command))
        if local:
            offset = self.visible_rect.topLeft()
            result.translate(-offset.x(), -offset.y())
        return result

    def replace_command(self, identifier: str, replacement: EditCommand) -> bool:
        commands = tuple(
            replace(replacement, id=item.id) if item.id == identifier else item
            for item in self._commands
        )
        return self._commit(commands)

    def update_selected(self, **changes: object) -> bool:
        selected = set(self._selected_ids)
        if not selected:
            return False
        allowed = {
            "color", "fill_color", "stroke_width", "opacity", "block_size",
            "arrow_size", "font_family", "font_size", "bold", "italic",
            "background_color", "text", "number",
        }
        values = {key: value for key, value in changes.items() if key in allowed}
        if not values:
            return False
        return self._commit(
            tuple(replace(item, **values) if item.id in selected else item for item in self._commands)
        )

    def move_selected(self, dx: float, dy: float) -> bool:
        selected = set(self._selected_ids)
        if not selected or (abs(dx) < 1e-9 and abs(dy) < 1e-9):
            return False
        return self._commit(
            tuple(translated_command(item, dx, dy) if item.id in selected else item for item in self._commands)
        )

    def resize_selected(self, new_bounds: QRectF) -> bool:
        selected = set(self._selected_ids)
        old_bounds = self.selection_bounds(local=False)
        if not selected or old_bounds.isEmpty() or new_bounds.isEmpty():
            return False
        target = QRectF(new_bounds)
        offset = self.visible_rect.topLeft()
        target.translate(offset.x(), offset.y())
        return self._commit(
            tuple(
                resized_command(item, old_bounds, target) if item.id in selected else item
                for item in self._commands
            )
        )

    def set_line_endpoint(self, identifier: str, endpoint: int, point: PointTuple) -> bool:
        command = next((item for item in self._commands if item.id == identifier), None)
        if command is None or command.tool not in {EditorTool.LINE, EditorTool.ARROW} or len(command.points) < 2:
            return False
        points = list(command.points)
        offset = self.visible_rect.topLeft()
        points[0 if endpoint == 0 else -1] = (point[0] + offset.x(), point[1] + offset.y())
        return self.replace_command(identifier, replace(command, points=tuple(points)))

    def delete_selected(self) -> bool:
        selected = set(self._selected_ids)
        if not selected:
            return False
        changed = self._commit(tuple(item for item in self._commands if item.id not in selected))
        if changed:
            self.clear_selection()
        return changed

    def duplicate_selected(self, offset: PointTuple = (10.0, 10.0)) -> tuple[str, ...]:
        selected = self.selected_commands
        if not selected:
            return ()
        duplicates = tuple(
            replace(translated_command(item, *offset), id=uuid4().hex)
            for item in selected
        )
        crop = tuple(item for item in self._commands if item.tool is EditorTool.CROP)
        annotations = tuple(item for item in self._commands if item.tool is not EditorTool.CROP)
        if not self._commit((*annotations, *duplicates, *crop)):
            return ()
        identifiers = tuple(item.id for item in duplicates)
        self.set_selection(identifiers)
        return identifiers

    def _reorder_selected(self, operation: str) -> bool:
        crop = tuple(item for item in self._commands if item.tool is EditorTool.CROP)
        annotations = [item for item in self._commands if item.tool is not EditorTool.CROP]
        selected = set(self._selected_ids)
        if not selected:
            return False
        if operation == "front":
            annotations = [item for item in annotations if item.id not in selected] + [item for item in annotations if item.id in selected]
        elif operation == "back":
            annotations = [item for item in annotations if item.id in selected] + [item for item in annotations if item.id not in selected]
        elif operation == "forward":
            for index in range(len(annotations) - 2, -1, -1):
                if annotations[index].id in selected and annotations[index + 1].id not in selected:
                    annotations[index], annotations[index + 1] = annotations[index + 1], annotations[index]
        elif operation == "backward":
            for index in range(1, len(annotations)):
                if annotations[index].id in selected and annotations[index - 1].id not in selected:
                    annotations[index], annotations[index - 1] = annotations[index - 1], annotations[index]
        return self._commit((*annotations, *crop))

    def bring_to_front(self) -> bool:
        return self._reorder_selected("front")

    def send_to_back(self) -> bool:
        return self._reorder_selected("back")

    def bring_forward(self) -> bool:
        return self._reorder_selected("forward")

    def send_backward(self) -> bool:
        return self._reorder_selected("backward")


class InlineTextEdit(QPlainTextEdit):
    submitted = Signal()
    cancelled = Signal()

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802 - Qt API
        if event.key() in {Qt.Key.Key_Return, Qt.Key.Key_Enter} and not (
            event.modifiers() & Qt.KeyboardModifier.ShiftModifier
        ):
            self.submitted.emit()
            event.accept()
            return
        if event.key() == Qt.Key.Key_Escape:
            self.cancelled.emit()
            event.accept()
            return
        super().keyPressEvent(event)


class EditorCanvas(QWidget):
    commandCommitted = Signal(object)
    toolChanged = Signal(object)
    zoomChanged = Signal(float)
    styleChanged = Signal(object, object)
    textEditingChanged = Signal(bool)

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

    def __init__(self, model: ScreenshotEditModel, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._model = model
        self._tool = EditorTool.RECTANGLE
        self._color = "#e53935"
        self._fill_color = ""
        self._stroke_width = 3.0
        self._opacity = 1.0
        self._block_size = 12
        self._arrow_size = 12.0
        self._font_family = ""
        self._font_size = 18
        self._bold = False
        self._italic = False
        self._background_color = ""
        self._pending_text = "文字"
        self._number = 1
        self._points: list[PointTuple] = []
        self._draft_modifiers = Qt.KeyboardModifier.NoModifier
        self._zoom = 1.0
        self._drag_origin: PointTuple | None = None
        self._drag_current: PointTuple | None = None
        self._selection_modifiers = Qt.KeyboardModifier.NoModifier
        self._resize_handle = ""
        self._line_endpoint: tuple[str, int] | None = None
        self._text_edit: InlineTextEdit | None = None
        self._editing_text_id = ""
        self._editing_text_style: dict[str, object] | None = None
        self._space_pan = False
        self._pan_origin: QPoint | None = None
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self._sync_cursor()
        self._model.changed.connect(self._sync_size)
        self._model.selectionChanged.connect(lambda _ids: self.update())
        self._sync_size()

    @property
    def tool(self) -> EditorTool:
        return self._tool

    @property
    def pending_text(self) -> str:
        return self._pending_text

    @property
    def text_editor_active(self) -> bool:
        return self._text_edit is not None

    @property
    def has_draft(self) -> bool:
        return bool(self._points or self._drag_origin or self._text_edit)

    @property
    def zoom(self) -> float:
        return self._zoom

    def set_tool(self, tool: EditorTool | str) -> None:
        parsed = EditorTool.parse(tool)
        if parsed == self._tool:
            return
        self.cancel_current_operation()
        self._tool = parsed
        self._sync_cursor()
        self.toolChanged.emit(parsed)
        self.update()

    def _sync_cursor(self) -> None:
        self.setCursor(
            Qt.CursorShape.ArrowCursor
            if self._tool is EditorTool.SELECT
            else Qt.CursorShape.CrossCursor
        )

    def set_pending_text(self, text: str) -> None:
        self._pending_text = str(text)

    def set_color(self, color: str | QColor) -> None:
        parsed = QColor(color)
        if parsed.isValid():
            self._color = parsed.name(
                QColor.NameFormat.HexArgb if parsed.alpha() < 255 else QColor.NameFormat.HexRgb
            )
            self.styleChanged.emit(self._tool, self.current_style())

    def set_fill_color(self, color: str | QColor) -> None:
        if not color:
            self._fill_color = ""
        else:
            parsed = QColor(color)
            if parsed.isValid():
                self._fill_color = parsed.name(QColor.NameFormat.HexArgb)
        self.styleChanged.emit(self._tool, self.current_style())

    def set_stroke_width(self, width: float) -> None:
        self._stroke_width = max(0.5, min(64.0, float(width)))
        self.styleChanged.emit(self._tool, self.current_style())

    def set_opacity(self, opacity: float) -> None:
        self._opacity = max(0.05, min(1.0, float(opacity)))
        self.styleChanged.emit(self._tool, self.current_style())

    def set_block_size(self, size: int) -> None:
        self._block_size = max(2, min(96, int(size)))
        self.styleChanged.emit(self._tool, self.current_style())

    def set_arrow_size(self, size: float) -> None:
        self._arrow_size = max(4.0, min(96.0, float(size)))
        self.styleChanged.emit(self._tool, self.current_style())

    def set_text_style(
        self,
        *,
        family: str | None = None,
        size: int | None = None,
        bold: bool | None = None,
        italic: bool | None = None,
        background_color: str | None = None,
    ) -> None:
        if family is not None:
            self._font_family = str(family)
        if size is not None:
            self._font_size = max(8, min(160, int(size)))
        if bold is not None:
            self._bold = bool(bold)
        if italic is not None:
            self._italic = bool(italic)
        if background_color is not None:
            self._background_color = str(background_color)
        self.styleChanged.emit(self._tool, self.current_style())

    def set_number_start(self, number: int) -> None:
        self._number = max(1, min(9999, int(number)))
        self.styleChanged.emit(self._tool, self.current_style())

    def current_style(self) -> dict[str, object]:
        return {
            "color": self._color,
            "fill_color": self._fill_color,
            "stroke_width": self._stroke_width,
            "opacity": self._opacity,
            "block_size": self._block_size,
            "arrow_size": self._arrow_size,
            "font_family": self._font_family,
            "font_size": self._font_size,
            "bold": self._bold,
            "italic": self._italic,
            "background_color": self._background_color,
            "number_start": self._number,
        }

    def apply_style(self, values: Mapping[str, object]) -> None:
        if "color" in values:
            self.set_color(str(values["color"]))
        if "fill_color" in values:
            self.set_fill_color(str(values["fill_color"] or ""))
        if "stroke_width" in values:
            self.set_stroke_width(float(values["stroke_width"]))
        if "opacity" in values:
            self.set_opacity(float(values["opacity"]))
        if "block_size" in values:
            self.set_block_size(int(values["block_size"]))
        if "arrow_size" in values:
            self.set_arrow_size(float(values["arrow_size"]))
        self.set_text_style(
            family=str(values["font_family"]) if "font_family" in values else None,
            size=int(values["font_size"]) if "font_size" in values else None,
            bold=bool(values["bold"]) if "bold" in values else None,
            italic=bool(values["italic"]) if "italic" in values else None,
            background_color=(str(values["background_color"]) if "background_color" in values else None),
        )
        if "number_start" in values:
            self.set_number_start(int(values["number_start"]))

    def set_zoom(self, zoom: float, *, anchor: QPointF | None = None) -> None:
        target = max(0.1, min(8.0, float(zoom)))
        if abs(target - self._zoom) < 1e-6:
            return
        old = self._zoom
        image_anchor = self.widget_to_image(anchor) if anchor is not None else None
        self._zoom = target
        self._sync_size()
        if anchor is not None and image_anchor is not None:
            scroll = self._scroll_area()
            if scroll is not None:
                old_global = self.mapToGlobal(anchor.toPoint())

                def restore_anchor() -> None:
                    new_widget = self.image_to_widget(image_anchor).toPoint()
                    new_global = self.mapToGlobal(new_widget)
                    delta = new_global - old_global
                    scroll.horizontalScrollBar().setValue(scroll.horizontalScrollBar().value() + delta.x())
                    scroll.verticalScrollBar().setValue(scroll.verticalScrollBar().value() + delta.y())

                QTimer.singleShot(0, restore_anchor)
        self.zoomChanged.emit(self._zoom)
        del old

    def fit_to_view(self) -> None:
        scroll = self._scroll_area()
        viewport = scroll.viewport().size() if scroll is not None else self.parentWidget().size() if self.parentWidget() else self.size()
        visible = self._model.render_cached().size()
        if visible.width() <= 0 or visible.height() <= 0:
            return
        self.set_zoom(min(viewport.width() / visible.width(), viewport.height() / visible.height(), 1.0))

    def one_to_one(self) -> None:
        self.set_zoom(1.0)

    def _scroll_area(self) -> QScrollArea | None:
        parent = self.parentWidget()
        while parent is not None:
            if isinstance(parent, QScrollArea):
                return parent
            parent = parent.parentWidget()
        return None

    def image_to_widget(self, point: PointTuple | QPointF) -> QPointF:
        value = point if isinstance(point, QPointF) else QPointF(*point)
        return QPointF(value.x() * self._zoom, value.y() * self._zoom)

    def widget_to_image(self, point: QPointF | None) -> QPointF:
        if point is None:
            return QPointF()
        return QPointF(point.x() / self._zoom, point.y() / self._zoom)

    def _sync_size(self) -> None:
        size = self._model.render_cached().size()
        scaled = QSize(max(1, round(size.width() * self._zoom)), max(1, round(size.height() * self._zoom)))
        self.setMinimumSize(scaled)
        self.resize(scaled)
        self.update()

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt API
        size = self._model.render_cached().size()
        return QSize(round(size.width() * self._zoom), round(size.height() * self._zoom))

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802 - Qt API
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, self._zoom < 1.0)
        painter.scale(self._zoom, self._zoom)
        painter.drawImage(0, 0, self._model.render_cached())
        self._paint_drag_preview(painter)
        self._paint_selection(painter)
        painter.end()
        del event

    def _paint_drag_preview(self, painter: QPainter) -> None:
        if self._tool is EditorTool.SELECT and self._drag_origin and self._drag_current:
            selected = self._model.selected_commands
            visible = self._model.visible_rect
            previews: tuple[EditCommand, ...]
            if self._line_endpoint is not None and len(selected) == 1:
                identifier, endpoint = self._line_endpoint
                command = selected[0]
                points = list(command.points)
                if command.id == identifier and len(points) >= 2:
                    points[0 if endpoint == 0 else -1] = (
                        self._drag_current[0] + visible.x(),
                        self._drag_current[1] + visible.y(),
                    )
                previews = (replace(command, points=tuple(points)),)
            elif self._resize_handle:
                local_bounds = self._model.selection_bounds()
                target = selection_resize_target(
                    local_bounds,
                    self._resize_handle,
                    self._drag_current,
                    self._selection_modifiers,
                )
                old_bounds = self._model.selection_bounds(local=False)
                absolute_target = QRectF(target)
                absolute_target.translate(visible.x(), visible.y())
                previews = tuple(
                    resized_command(command, old_bounds, absolute_target)
                    for command in selected
                )
            else:
                dx = self._drag_current[0] - self._drag_origin[0]
                dy = self._drag_current[1] - self._drag_origin[1]
                previews = tuple(
                    translated_command(command, dx, dy) for command in selected
                )
            painter.setOpacity(0.72)
            for command in previews:
                local = translated_command(command, -visible.x(), -visible.y())
                draw_edit_command(painter, local)
            painter.setOpacity(1.0)
            return
        if len(self._points) < 2:
            return
        draft = self._command_from_points(self._points, self._draft_modifiers)
        if draft is None:
            return
        if draft.tool in {EditorTool.CROP, EditorTool.MOSAIC, EditorTool.BLUR}:
            painter.setPen(QPen(QColor("#2db4ff"), 1.5 / self._zoom, Qt.PenStyle.DashLine))
            painter.setBrush(QColor(45, 180, 255, 32))
            painter.drawRect(command_rect(draft))
            return
        draw_edit_command(painter, draft)

    def _paint_selection(self, painter: QPainter) -> None:
        if self._tool is not EditorTool.SELECT or not self._model.selected_ids:
            return
        bounds = self._preview_selection_bounds()
        painter.setPen(QPen(QColor("#2db4ff"), 1.25 / self._zoom, Qt.PenStyle.DashLine))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(bounds)
        painter.setBrush(QColor("#ffffff"))
        for _name, point in self._selection_handles(bounds).items():
            radius = 4.0 / self._zoom
            painter.drawRect(QRectF(point.x() - radius, point.y() - radius, radius * 2, radius * 2))
        selected = self._model.selected_commands
        if len(selected) == 1 and selected[0].tool in {EditorTool.LINE, EditorTool.ARROW}:
            offset = self._model.visible_rect.topLeft()
            for point in (selected[0].points[0], selected[0].points[-1]):
                center = QPointF(point[0] - offset.x(), point[1] - offset.y())
                radius = 5.0 / self._zoom
                painter.drawEllipse(center, radius, radius)

    def _preview_selection_bounds(self) -> QRectF:
        bounds = self._model.selection_bounds()
        if not self._drag_origin or not self._drag_current:
            return bounds
        if self._line_endpoint is not None:
            selected = self._model.selected_commands
            if len(selected) == 1:
                command = selected[0]
                points = list(command.points)
                endpoint = self._line_endpoint[1]
                visible = self._model.visible_rect
                points[0 if endpoint == 0 else -1] = (
                    self._drag_current[0] + visible.x(),
                    self._drag_current[1] + visible.y(),
                )
                bounds = command_rect(replace(command, points=tuple(points)))
                bounds.translate(-visible.x(), -visible.y())
        elif self._resize_handle:
            bounds = selection_resize_target(
                bounds,
                self._resize_handle,
                self._drag_current,
                self._selection_modifiers,
            )
        else:
            bounds.translate(
                self._drag_current[0] - self._drag_origin[0],
                self._drag_current[1] - self._drag_origin[1],
            )
        return bounds

    @staticmethod
    def _selection_handles(bounds: QRectF) -> dict[str, QPointF]:
        return {
            "nw": bounds.topLeft(), "n": QPointF(bounds.center().x(), bounds.top()), "ne": bounds.topRight(),
            "e": QPointF(bounds.right(), bounds.center().y()), "se": bounds.bottomRight(),
            "s": QPointF(bounds.center().x(), bounds.bottom()), "sw": bounds.bottomLeft(),
            "w": QPointF(bounds.left(), bounds.center().y()),
        }

    def _handle_at(self, point: QPointF) -> str:
        tolerance = 7.0 / self._zoom
        bounds = self._model.selection_bounds()
        for name, center in self._selection_handles(bounds).items():
            if math.hypot(point.x() - center.x(), point.y() - center.y()) <= tolerance:
                return name
        return ""

    def _endpoint_at(self, point: QPointF) -> tuple[str, int] | None:
        selected = self._model.selected_commands
        if len(selected) != 1 or selected[0].tool not in {EditorTool.LINE, EditorTool.ARROW}:
            return None
        offset = self._model.visible_rect.topLeft()
        for index, value in ((0, selected[0].points[0]), (-1, selected[0].points[-1])):
            candidate = QPointF(value[0] - offset.x(), value[1] - offset.y())
            if math.hypot(point.x() - candidate.x(), point.y() - candidate.y()) <= 8.0 / self._zoom:
                return selected[0].id, index
        return None

    def _constrained_point(self, start: PointTuple, end: PointTuple, modifiers: Qt.KeyboardModifier) -> PointTuple:
        x1, y1 = start
        x2, y2 = end
        if modifiers & Qt.KeyboardModifier.ShiftModifier:
            if self._tool in {EditorTool.LINE, EditorTool.ARROW}:
                distance = math.hypot(x2 - x1, y2 - y1)
                angle = round(math.atan2(y2 - y1, x2 - x1) / (math.pi / 4)) * (math.pi / 4)
                x2 = x1 + distance * math.cos(angle)
                y2 = y1 + distance * math.sin(angle)
            elif self._tool in {EditorTool.RECTANGLE, EditorTool.ELLIPSE}:
                size = max(abs(x2 - x1), abs(y2 - y1))
                x2 = x1 + math.copysign(size, x2 - x1 or 1)
                y2 = y1 + math.copysign(size, y2 - y1 or 1)
        return x2, y2

    def _command_from_points(self, points: Sequence[PointTuple], modifiers: Qt.KeyboardModifier = Qt.KeyboardModifier.NoModifier) -> EditCommand | None:
        if not points:
            return None
        common = {
            "color": self._color,
            "fill_color": self._fill_color,
            "stroke_width": self._stroke_width,
            "opacity": self._opacity,
            "block_size": self._block_size,
            "arrow_size": self._arrow_size,
            "font_family": self._font_family,
            "font_size": self._font_size,
            "bold": self._bold,
            "italic": self._italic,
            "background_color": self._background_color,
        }
        if self._tool is EditorTool.PEN:
            return EditCommand(self._tool, points=tuple(points), **common)
        if self._tool is EditorTool.NUMBER:
            return EditCommand(self._tool, points=(points[0],), number=self._number, **common)
        if len(points) < 2:
            return None
        start = points[0]
        end = self._constrained_point(start, points[-1], modifiers)
        if modifiers & Qt.KeyboardModifier.ControlModifier and self._tool in {
            EditorTool.RECTANGLE, EditorTool.ELLIPSE, EditorTool.HIGHLIGHT,
            EditorTool.MOSAIC, EditorTool.BLUR, EditorTool.CROP,
        }:
            dx, dy = end[0] - start[0], end[1] - start[1]
            start = (start[0] - dx, start[1] - dy)
        if self._tool is EditorTool.HIGHLIGHT:
            common.update(color=self._color or "#fff176", opacity=min(self._opacity, 0.55))
        return EditCommand.from_drag(self._tool, start, end, **common)

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt API
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        image_point = self.widget_to_image(event.position())
        point = (image_point.x(), image_point.y())
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
            if self._model.selected_ids:
                self._show_object_menu(event.globalPosition().toPoint())
                event.accept()
                return
        if event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return
        if self._tool is EditorTool.SELECT:
            self._selection_modifiers = event.modifiers()
            endpoint = self._endpoint_at(image_point)
            handle = self._handle_at(image_point) if endpoint is None else ""
            if endpoint is not None:
                self._line_endpoint = endpoint
                self._drag_origin = point
                self._drag_current = point
            elif handle:
                self._resize_handle = handle
                self._drag_origin = point
                self._drag_current = point
            else:
                hit = self._model.select_at(
                    point,
                    additive=bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier),
                )
                if hit is not None and hit.id in self._model.selected_ids:
                    self._drag_origin = point
                    self._drag_current = point
            self.update()
            event.accept()
            return
        if self._tool is EditorTool.TEXT:
            self._begin_text_edit(image_point)
            event.accept()
            return
        self._points = [point]
        self._draft_modifiers = event.modifiers()
        event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt API
        if self._pan_origin is not None:
            scroll = self._scroll_area()
            current = event.globalPosition().toPoint()
            delta = current - self._pan_origin
            self._pan_origin = current
            if scroll is not None:
                scroll.horizontalScrollBar().setValue(scroll.horizontalScrollBar().value() - delta.x())
                scroll.verticalScrollBar().setValue(scroll.verticalScrollBar().value() - delta.y())
            event.accept()
            return
        image_point = self.widget_to_image(event.position())
        current = (image_point.x(), image_point.y())
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
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt API
        if self._pan_origin is not None and event.button() in {Qt.MouseButton.MiddleButton, Qt.MouseButton.LeftButton}:
            self._pan_origin = None
            self._sync_cursor()
            event.accept()
            return
        image_point = self.widget_to_image(event.position())
        current = (image_point.x(), image_point.y())
        if self._tool is EditorTool.SELECT and self._drag_origin is not None and event.button() == Qt.MouseButton.LeftButton:
            self._drag_current = current
            if self._line_endpoint is not None:
                identifier, endpoint = self._line_endpoint
                self._model.set_line_endpoint(identifier, endpoint, current)
            elif self._resize_handle:
                self._finish_resize(current, event.modifiers())
            else:
                dx = current[0] - self._drag_origin[0]
                dy = current[1] - self._drag_origin[1]
                self._model.move_selected(dx, dy)
            self._drag_origin = None
            self._drag_current = None
            self._selection_modifiers = Qt.KeyboardModifier.NoModifier
            self._resize_handle = ""
            self._line_endpoint = None
            self.update()
            event.accept()
            return
        if event.button() != Qt.MouseButton.LeftButton or not self._points:
            super().mouseReleaseEvent(event)
            return
        if self._tool is not EditorTool.NUMBER:
            if len(self._points) == 1:
                self._points.append(current)
            else:
                self._points[-1] = current
        command = self._command_from_points(self._points, event.modifiers())
        self._points = []
        self._draft_modifiers = Qt.KeyboardModifier.NoModifier
        if command is not None:
            self._model.add_command(command)
            if command.tool is EditorTool.NUMBER:
                self._number += 1
            self.commandCommitted.emit(command)
        self.update()
        event.accept()

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt API
        if event.button() == Qt.MouseButton.LeftButton:
            point = self.widget_to_image(event.position())
            command = self._model.hit_test(point)
            if command is not None and command.tool is EditorTool.TEXT:
                self._model.set_selection((command.id,))
                self._begin_text_edit(point, command=command)
                event.accept()
                return
        super().mouseDoubleClickEvent(event)

    def _finish_resize(self, current: PointTuple, modifiers: Qt.KeyboardModifier) -> None:
        bounds = self._model.selection_bounds()
        if bounds.isEmpty():
            return
        target = selection_resize_target(
            bounds,
            self._resize_handle,
            current,
            modifiers,
        )
        if target.width() >= 1 and target.height() >= 1:
            self._model.resize_selected(target)

    def _begin_text_edit(self, point: QPointF, command: EditCommand | None = None) -> None:
        self._finish_text_edit(cancel=True)
        editor = InlineTextEdit(self)
        editor.setTabChangesFocus(False)
        editor.setPlaceholderText("输入文字；Shift+Enter 换行")
        editor.setStyleSheet(
            "QPlainTextEdit { background: rgba(20, 24, 30, 225); color: white;"
            " border: 1px solid #2db4ff; border-radius: 4px; padding: 4px; }"
        )
        target = command_rect(command) if command is not None else QRectF(point.x(), point.y(), 260, 90)
        if command is not None:
            offset = self._model.visible_rect.topLeft()
            target.translate(-offset.x(), -offset.y())
            editor.setPlainText(command.text)
            self._editing_text_id = command.id
            style: dict[str, object] = {
                "color": command.color,
                "opacity": command.opacity,
                "font_family": command.font_family,
                "font_size": command.font_size,
                "bold": command.bold,
                "italic": command.italic,
                "background_color": command.background_color,
            }
        else:
            editor.setPlainText("")
            self._editing_text_id = ""
            style = self.current_style()
        self._editing_text_style = style
        top_left = self.image_to_widget(target.topLeft()).toPoint()
        width = max(180, round(max(220.0, target.width()) * self._zoom))
        height = max(70, round(max(70.0, target.height()) * self._zoom))
        editor.setGeometry(QRect(top_left, QSize(width, height)))
        family = str(style.get("font_family", ""))
        font = QFont(family) if family else QFont()
        font.setPixelSize(
            max(10, round(int(style.get("font_size", 18)) * self._zoom))
        )
        font.setBold(bool(style.get("bold", False)))
        font.setItalic(bool(style.get("italic", False)))
        editor.setFont(font)
        editor.submitted.connect(self._commit_text_edit)
        editor.cancelled.connect(lambda: self._finish_text_edit(cancel=True))
        self._text_edit = editor
        self.textEditingChanged.emit(True)
        editor.show()
        editor.setFocus(Qt.FocusReason.MouseFocusReason)
        editor.selectAll()

    def _commit_text_edit(self) -> None:
        editor = self._text_edit
        if editor is None:
            return
        text = editor.toPlainText().rstrip()
        style = self._editing_text_style or self.current_style()
        rect = editor.geometry()
        top_left = self.widget_to_image(QPointF(rect.topLeft()))
        size = QSize(max(1, round(rect.width() / self._zoom)), max(1, round(rect.height() / self._zoom)))
        command = EditCommand(
            EditorTool.TEXT,
            points=(
                (
                    top_left.x(),
                    top_left.y() + int(style.get("font_size", 18)),
                ),
            ),
            rect=(top_left.x(), top_left.y(), float(size.width()), float(size.height())),
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
                visible = self._model.visible_rect
                self._model.replace_command(
                    self._editing_text_id,
                    translated_command(command, visible.x(), visible.y()),
                )
            else:
                self._model.set_selection((self._editing_text_id,))
                self._model.delete_selected()
        elif text:
            self._model.add_command(command)
            self.commandCommitted.emit(command)
            self._pending_text = text
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
        self.textEditingChanged.emit(False)
        self.setFocus(Qt.FocusReason.OtherFocusReason)
        self.update()
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
            self._model.duplicate_selected()
        elif chosen is front:
            self._model.bring_to_front()
        elif chosen is forward:
            self._model.bring_forward()
        elif chosen is backward:
            self._model.send_backward()
        elif chosen is back:
            self._model.send_to_back()
        elif chosen is remove:
            self._model.delete_selected()

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802 - Qt API
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
            self.set_zoom(self._zoom * factor, anchor=event.position())
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
            (self._model.redo if modifiers & Qt.KeyboardModifier.ShiftModifier else self._model.undo)()
            event.accept()
            return
        if control and event.key() == Qt.Key.Key_Y:
            self._model.redo()
            event.accept()
            return
        if control and event.key() == Qt.Key.Key_D:
            self._model.duplicate_selected()
            event.accept()
            return
        if event.key() == Qt.Key.Key_Space:
            self._space_pan = True
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            event.accept()
            return
        if event.key() == Qt.Key.Key_Delete:
            self._model.delete_selected()
            event.accept()
            return
        if event.key() in {Qt.Key.Key_Left, Qt.Key.Key_Right, Qt.Key.Key_Up, Qt.Key.Key_Down} and self._model.selected_ids:
            amount = 10 if modifiers & Qt.KeyboardModifier.ShiftModifier else 1
            dx = -amount if event.key() == Qt.Key.Key_Left else amount if event.key() == Qt.Key.Key_Right else 0
            dy = -amount if event.key() == Qt.Key.Key_Up else amount if event.key() == Qt.Key.Key_Down else 0
            self._model.move_selected(dx, dy)
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
                self._sync_cursor()
            event.accept()
            return
        super().keyReleaseEvent(event)


class ScreenshotEditor(QMainWindow):
    """Standalone fallback host sharing the same object model as inline editing."""

    saved = Signal(str)
    copied = Signal()
    completed = Signal(object)
    cancelled = Signal()
    copyOutputRequested = Signal(object)
    saveAsOutputRequested = Signal(object, str)

    _TOOL_LABELS = {
        EditorTool.SELECT: "选择",
        EditorTool.RECTANGLE: "矩形",
        EditorTool.ELLIPSE: "椭圆",
        EditorTool.LINE: "直线",
        EditorTool.ARROW: "箭头",
        EditorTool.PEN: "画笔",
        EditorTool.TEXT: "文字",
        EditorTool.NUMBER: "编号",
        EditorTool.HIGHLIGHT: "高亮",
        EditorTool.MOSAIC: "马赛克",
        EditorTool.BLUR: "模糊",
        EditorTool.CROP: "裁剪",
    }

    def __init__(
        self,
        image: QImage,
        parent: QWidget | None = None,
        *,
        model: ScreenshotEditModel | None = None,
        managed_output: bool = False,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Fiber Screenshot Tool · 标注")
        self.model = model or ScreenshotEditModel(image, self)
        self._managed_output = bool(managed_output)
        if self.model.parent() is None:
            self.model.setParent(self)
        self.canvas = EditorCanvas(self.model)
        self.scroll = QScrollArea(self)
        self.scroll.setWidget(self.canvas)
        self.scroll.setWidgetResizable(False)
        self.scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setCentralWidget(self.scroll)
        self.tool_actions: dict[EditorTool, QAction] = {}
        self._current_color = QColor("#e53935")
        self._current_fill = QColor()
        self._current_background = QColor()
        self._build_toolbar()
        self._build_property_toolbar()
        self.model.historyChanged.connect(self._update_history_actions)
        self.model.selectionChanged.connect(self._selection_properties_changed)
        self.canvas.textEditingChanged.connect(self._text_editing_changed)
        self.canvas.commandCommitted.connect(self._canvas_command_committed)
        screen = QGuiApplication.primaryScreen()
        available = screen.availableGeometry() if screen is not None else QRect(0, 0, 1180, 860)
        width = min(max(320, available.width() - 32), max(720, image.width() + 100), 1180)
        height = min(max(280, available.height() - 32), max(520, image.height() + 140), 860)
        self.resize(width, height)

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
        self.color_button.setToolTip("描边或当前对象颜色")
        self.color_button.clicked.connect(self._choose_color)
        toolbar.addWidget(self.color_button)
        self.width_spin = QSpinBox(self)
        self.width_spin.setRange(1, 64)
        self.width_spin.setValue(3)
        self.width_spin.setSuffix(" px")
        self.width_spin.setToolTip("线宽")
        self.width_spin.valueChanged.connect(self.canvas.set_stroke_width)
        self.width_label = QLabel("线宽", self)
        toolbar.addWidget(self.width_label)
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
        fit_action = QAction("适合窗口", self)
        fit_action.triggered.connect(self.canvas.fit_to_view)
        toolbar.addAction(fit_action)
        one_action = QAction("1:1", self)
        one_action.triggered.connect(self.canvas.one_to_one)
        toolbar.addAction(one_action)
        toolbar.addSeparator()
        complete_action = QAction("完成", self)
        complete_action.setToolTip("按截图工具设置执行保存和复制")
        complete_action.triggered.connect(self.complete)
        toolbar.addAction(complete_action)
        self.save_action = QAction("另存为", self)
        self.save_action.setShortcut(QKeySequence.StandardKey.Save)
        self.save_action.triggered.connect(lambda: self.save())
        toolbar.addAction(self.save_action)
        self.copy_action = QAction("复制", self)
        self.copy_action.setShortcut(QKeySequence.StandardKey.Copy)
        self.copy_action.triggered.connect(self.copy_to_clipboard)
        toolbar.addAction(self.copy_action)
        cancel_action = QAction("取消", self)
        cancel_action.triggered.connect(self.cancel)
        toolbar.addAction(cancel_action)
        self._update_history_actions(False, False)

    def set_tool(self, tool: EditorTool | str) -> None:
        parsed = EditorTool.parse(tool)
        self.canvas.set_tool(parsed)
        action = self.tool_actions.get(parsed)
        if action is not None:
            action.setChecked(True)
        self._sync_property_visibility()

    def add_command(self, command: EditCommand) -> None:
        self.model.add_command(command)

    def undo(self) -> bool:
        return self.model.undo()

    def redo(self) -> bool:
        return self.model.redo()

    def edited_image(self) -> QImage:
        return self.model.render()

    def complete(self) -> None:
        if self.canvas.text_editor_active:
            self.canvas._commit_text_edit()
        self.completed.emit(self.edited_image())

    def cancel(self) -> None:
        self.cancelled.emit()
        self.close()

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
        if self.model.selected_ids:
            self.model.update_selected(color=selected.name(QColor.NameFormat.HexArgb))
        self._refresh_color_button()

    def _refresh_color_button(self) -> None:
        foreground = "#000000" if self._current_color.lightnessF() > 0.62 else "#ffffff"
        self.color_button.setStyleSheet(
            "QPushButton {"
            f"background: {self._current_color.name(QColor.NameFormat.HexArgb)};"
            f"color: {foreground}; padding: 3px 10px; border: 1px solid palette(mid);"
            "}"
        )

    def _build_property_toolbar(self) -> None:
        self.addToolBarBreak(Qt.ToolBarArea.TopToolBarArea)
        toolbar = QToolBar("标注属性", self)
        toolbar.setMovable(False)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, toolbar)
        self.property_toolbar = toolbar
        self.fill_button = QPushButton("填充：无", self)
        self.fill_button.clicked.connect(self._choose_secondary_color)
        self.fill_button.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.fill_button.customContextMenuRequested.connect(
            lambda _point: self._clear_secondary_color()
        )
        toolbar.addWidget(self.fill_button)
        self.opacity_label = QLabel("透明度", self)
        toolbar.addWidget(self.opacity_label)
        self.opacity_spin = QSpinBox(self)
        self.opacity_spin.setRange(5, 100)
        self.opacity_spin.setValue(100)
        self.opacity_spin.setSuffix("%")
        self.opacity_spin.valueChanged.connect(self._apply_property_controls)
        toolbar.addWidget(self.opacity_spin)
        self.strength_label = QLabel("强度", self)
        toolbar.addWidget(self.strength_label)
        self.strength_spin = QSpinBox(self)
        self.strength_spin.setRange(2, 96)
        self.strength_spin.setValue(12)
        self.strength_spin.valueChanged.connect(self._apply_property_controls)
        toolbar.addWidget(self.strength_spin)
        self.arrow_label = QLabel("箭头", self)
        toolbar.addWidget(self.arrow_label)
        self.arrow_spin = QSpinBox(self)
        self.arrow_spin.setRange(4, 96)
        self.arrow_spin.setValue(12)
        self.arrow_spin.setSuffix(" px")
        self.arrow_spin.valueChanged.connect(self._apply_property_controls)
        toolbar.addWidget(self.arrow_spin)
        self.font_combo = QFontComboBox(self)
        self.font_combo.currentFontChanged.connect(self._apply_property_controls)
        toolbar.addWidget(self.font_combo)
        self.font_size_spin = QSpinBox(self)
        self.font_size_spin.setRange(8, 160)
        self.font_size_spin.setValue(18)
        self.font_size_spin.setSuffix(" px")
        self.font_size_spin.valueChanged.connect(self._apply_property_controls)
        toolbar.addWidget(self.font_size_spin)
        self.bold_button = QToolButton(self)
        self.bold_button.setText("粗体")
        self.bold_button.setCheckable(True)
        self.bold_button.toggled.connect(self._apply_property_controls)
        toolbar.addWidget(self.bold_button)
        self.italic_button = QToolButton(self)
        self.italic_button.setText("斜体")
        self.italic_button.setCheckable(True)
        self.italic_button.toggled.connect(self._apply_property_controls)
        toolbar.addWidget(self.italic_button)
        self.number_label = QLabel("编号起始", self)
        toolbar.addWidget(self.number_label)
        self.number_spin = QSpinBox(self)
        self.number_spin.setRange(1, 9999)
        self.number_spin.setValue(1)
        self.number_spin.valueChanged.connect(self._apply_property_controls)
        toolbar.addWidget(self.number_spin)
        self.selection_hint = QLabel("单击选择对象；Shift 可多选。", self)
        toolbar.addWidget(self.selection_hint)
        self.crop_hint = QLabel("拖动选择保留区域；不会超出原截图。", self)
        toolbar.addWidget(self.crop_hint)
        self.width_spin.valueChanged.connect(self._apply_property_controls)
        self._sync_property_visibility()

    def _effective_property_tool(self) -> EditorTool:
        selected = self.model.selected_commands
        if self.canvas.tool is EditorTool.SELECT and len(selected) == 1:
            return selected[0].tool
        return self.canvas.tool

    def _sync_property_visibility(self) -> None:
        tool = self._effective_property_tool()
        shape = tool in {EditorTool.RECTANGLE, EditorTool.ELLIPSE}
        effect = tool in {EditorTool.MOSAIC, EditorTool.BLUR}
        text = tool is EditorTool.TEXT
        number = tool is EditorTool.NUMBER
        arrow = tool is EditorTool.ARROW
        selection_empty = (
            self.canvas.tool is EditorTool.SELECT
            and not self.model.selected_ids
        )
        color_enabled = (
            not effect
            and tool is not EditorTool.CROP
            and not selection_empty
        )
        self.color_button.setVisible(color_enabled)
        self.fill_button.setVisible(shape or text)
        self.width_label.setVisible(color_enabled)
        self.width_spin.setVisible(not effect and tool is not EditorTool.CROP)
        self.opacity_label.setVisible(not effect and tool is not EditorTool.CROP)
        self.opacity_spin.setVisible(not effect and tool is not EditorTool.CROP)
        self.strength_label.setVisible(effect)
        self.strength_spin.setVisible(effect)
        self.arrow_label.setVisible(arrow)
        self.arrow_spin.setVisible(arrow)
        self.font_combo.setVisible(text)
        self.font_size_spin.setVisible(text or number)
        self.bold_button.setVisible(text)
        self.italic_button.setVisible(text)
        self.number_label.setVisible(number)
        self.number_spin.setVisible(number)
        self.selection_hint.setVisible(selection_empty)
        self.crop_hint.setVisible(tool is EditorTool.CROP)

    def _selection_properties_changed(self, _identifiers: object) -> None:
        selected = self.model.selected_commands
        if len(selected) == 1:
            command = selected[0]
            widgets = (
                self.width_spin,
                self.opacity_spin,
                self.strength_spin,
                self.arrow_spin,
                self.font_size_spin,
                self.number_spin,
            )
            for widget in widgets:
                widget.blockSignals(True)
            self.font_combo.blockSignals(True)
            self.bold_button.blockSignals(True)
            self.italic_button.blockSignals(True)
            self.width_spin.setValue(round(command.stroke_width))
            self.opacity_spin.setValue(round(command.opacity * 100))
            self.strength_spin.setValue(command.block_size)
            self.arrow_spin.setValue(round(command.arrow_size))
            if command.font_family:
                self.font_combo.setCurrentFont(QFont(command.font_family))
            self.font_size_spin.setValue(command.font_size)
            self.bold_button.setChecked(command.bold)
            self.italic_button.setChecked(command.italic)
            self.number_spin.setValue(command.number or 1)
            self._current_color = QColor(command.color)
            self._current_fill = QColor(command.fill_color)
            self._current_background = QColor(command.background_color)
            for widget in widgets:
                widget.blockSignals(False)
            self.font_combo.blockSignals(False)
            self.bold_button.blockSignals(False)
            self.italic_button.blockSignals(False)
            self._refresh_color_button()
        self._sync_secondary_button()
        self._sync_property_visibility()

    def _apply_property_controls(self, *_args: object) -> None:
        self.canvas.set_opacity(self.opacity_spin.value() / 100.0)
        self.canvas.set_block_size(self.strength_spin.value())
        self.canvas.set_arrow_size(self.arrow_spin.value())
        self.canvas.set_text_style(
            family=self.font_combo.currentFont().family(),
            size=self.font_size_spin.value(),
            bold=self.bold_button.isChecked(),
            italic=self.italic_button.isChecked(),
            background_color=(
                self._current_background.name(QColor.NameFormat.HexArgb)
                if self._current_background.isValid()
                else ""
            ),
        )
        self.canvas.set_number_start(self.number_spin.value())
        if self.model.selected_ids:
            changes: dict[str, object] = {
                "stroke_width": self.width_spin.value(),
                "opacity": self.opacity_spin.value() / 100.0,
                "block_size": self.strength_spin.value(),
                "arrow_size": self.arrow_spin.value(),
                "font_family": self.font_combo.currentFont().family(),
                "font_size": self.font_size_spin.value(),
                "bold": self.bold_button.isChecked(),
                "italic": self.italic_button.isChecked(),
            }
            if self._effective_property_tool() is EditorTool.TEXT:
                changes["background_color"] = (
                    self._current_background.name(QColor.NameFormat.HexArgb)
                    if self._current_background.isValid()
                    else ""
                )
            elif self._effective_property_tool() is EditorTool.NUMBER:
                changes["number"] = self.number_spin.value()
            self.model.update_selected(**changes)

    def _choose_secondary_color(self) -> None:
        text = self._effective_property_tool() is EditorTool.TEXT
        current = self._current_background if text else self._current_fill
        selected = QColorDialog.getColor(
            current if current.isValid() else QColor("#80000000"),
            self,
            "选择文字背景" if text else "选择填充颜色",
            QColorDialog.ColorDialogOption.ShowAlphaChannel,
        )
        if not selected.isValid():
            return
        if text:
            self._current_background = selected
            self.canvas.set_text_style(
                background_color=selected.name(QColor.NameFormat.HexArgb)
            )
            if self.model.selected_ids:
                self.model.update_selected(
                    background_color=selected.name(QColor.NameFormat.HexArgb)
                )
        else:
            self._current_fill = selected
            self.canvas.set_fill_color(selected)
            if self.model.selected_ids:
                self.model.update_selected(
                    fill_color=selected.name(QColor.NameFormat.HexArgb)
                )
        self._sync_secondary_button()

    def _clear_secondary_color(self) -> None:
        if self._effective_property_tool() is EditorTool.TEXT:
            self._current_background = QColor()
            self.canvas.set_text_style(background_color="")
            if self.model.selected_ids:
                self.model.update_selected(background_color="")
        else:
            self._current_fill = QColor()
            self.canvas.set_fill_color("")
            if self.model.selected_ids:
                self.model.update_selected(fill_color="")
        self._sync_secondary_button()

    def _sync_secondary_button(self) -> None:
        text = self._effective_property_tool() is EditorTool.TEXT
        color = self._current_background if text else self._current_fill
        label = "背景" if text else "填充"
        self.fill_button.setText(label if color.isValid() else f"{label}：无")
        self.fill_button.setStyleSheet(
            f"background: {color.name(QColor.NameFormat.HexArgb)};"
            if color.isValid()
            else ""
        )

    def save(self, path: str | Path | None = None) -> bool:
        target = str(path) if path is not None else ""
        if not target:
            target, _selected_filter = QFileDialog.getSaveFileName(
                self,
                "保存截图",
                "screenshot.png",
                "PNG 图片 (*.png);;JPEG 图片 (*.jpg *.jpeg);;WebP 图片 (*.webp)",
            )
        if not target:
            return False
        if not Path(target).suffix:
            target += ".png"
        if self._managed_output:
            self.saveAsOutputRequested.emit(self.edited_image(), target)
            return True
        success = self.edited_image().save(target)
        if success:
            self.saved.emit(target)
        return bool(success)

    def copy_to_clipboard(self) -> bool:
        if self._managed_output:
            self.copyOutputRequested.emit(self.edited_image())
            return True
        app = QGuiApplication.instance()
        if app is None:
            return False
        app.clipboard().setPixmap(QPixmap.fromImage(self.edited_image()))
        self.copied.emit()
        return True

    def _update_history_actions(self, can_undo: bool, can_redo: bool) -> None:
        if self.canvas.text_editor_active:
            can_undo = False
            can_redo = False
        self.undo_action.setEnabled(bool(can_undo))
        self.redo_action.setEnabled(bool(can_redo))

    def _text_editing_changed(self, active: bool) -> None:
        self.save_action.setEnabled(not active)
        self.copy_action.setEnabled(not active)
        self._update_history_actions(self.model.can_undo, self.model.can_redo)

    def _canvas_command_committed(self, command: object) -> None:
        if not isinstance(command, EditCommand) or command.tool is not EditorTool.NUMBER:
            return
        self.number_spin.blockSignals(True)
        self.number_spin.setValue(
            int(self.canvas.current_style().get("number_start", 1))
        )
        self.number_spin.blockSignals(False)

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802 - Qt API
        if event.key() in {Qt.Key.Key_Return, Qt.Key.Key_Enter} and not self.canvas.text_editor_active:
            self.complete()
            event.accept()
            return
        if event.key() == Qt.Key.Key_Escape and not self.canvas.text_editor_active:
            if self.canvas.cancel_current_operation():
                event.accept()
                return
            if self.model.selected_ids:
                self.model.clear_selection()
                event.accept()
                return
            if self.model.has_annotations:
                result = QMessageBox.question(self, "取消截图", "已有标注，确认放弃本次截图吗？")
                if result != QMessageBox.StandardButton.Yes:
                    event.accept()
                    return
            self.cancel()
            event.accept()
            return
        super().keyPressEvent(event)


__all__ = [
    "EditCommand",
    "EditorCanvas",
    "EditorTool",
    "InlineTextEdit",
    "ScreenshotEditModel",
    "ScreenshotEditor",
    "command_hit_test",
    "command_rect",
    "draw_edit_command",
    "render_edit_commands",
    "resized_command",
    "translated_command",
]
