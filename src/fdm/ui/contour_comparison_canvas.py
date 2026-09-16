"""Image-coordinate editing; display paths never supply measurement geometry."""
from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPainterPath, QPalette, QPen, QPolygonF, QTransform
from PySide6.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget

from fdm.services.contour_comparison import ContourFrame

BEFORE_COLOR = QColor("#039fc0")
AFTER_COLOR = QColor("#e88127")


@dataclass(frozen=True)
class ContourPresentation:
    image: QImage
    mask: QImage
    # Cached display geometry, deliberately separate from native pixel edges.
    contours: tuple[np.ndarray, ...]
    pixel_owner: np.ndarray


def prepare_presentation(frame: ContourFrame, color: QColor, previous=None) -> ContourPresentation:
    h, w = frame.mask.shape
    image = previous.image if previous is not None and previous.pixel_owner is frame.rgba else QImage(frame.rgba.data, w, h, frame.rgba.strides[0], QImage.Format.Format_RGBA8888).copy()
    binary = frame.mask.astype(np.uint8)
    overlay = QImage(binary.data, w, h, binary.strides[0], QImage.Format.Format_Indexed8).copy()
    overlay.setColorTable([QColor(0, 0, 0, 0).rgba(), QColor(color.red(), color.green(), color.blue(), 65).rgba()])
    # At most a 2048 px display mask supplies outline decoration. The opaque
    # pixel overlay above remains native resolution, including manual holes.
    ratio = min(1.0, 2048 / max(w, h))
    small = cv2.resize(binary, (max(1, round(w * ratio)), max(1, round(h * ratio))), interpolation=cv2.INTER_NEAREST) if ratio < 1 else binary
    contours, _ = cv2.findContours(small, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # Extremely fragmented suggestions still have their complete native raster
    # overlay; do not construct hundreds of thousands of Qt points on the UI.
    paths = tuple((c.reshape(-1, 2).astype(float) + .5) / ratio for c in contours if len(c) >= 3) if sum(len(c) for c in contours) <= 50000 else ()
    return ContourPresentation(image, overlay, paths, frame.rgba)


def contour_path(presentation: ContourPresentation, frame: ContourFrame | None = None) -> QPainterPath:
    path = QPainterPath()
    path.setFillRule(Qt.FillRule.OddEvenFill)
    for contour in presentation.contours:
        points = frame.axis.to_world(contour, frame.scale) if frame is not None else contour
        path.addPolygon(QPolygonF([QPointF(float(x), float(y)) for x, y in points]))
        path.closeSubpath()
    return path


class _PanZoomCanvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.zoom = 1.0
        self.offset = QPointF()
        self._pan = None
        self._pan_button = None
        self._space_pan = False
        self._auto_fit = True
        self.setMinimumSize(180, 220)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def content_rect(self):
        return QRectF(0, 0, 1, 1)

    def transform(self):
        return QTransform(self.zoom, 0, 0, self.zoom, self.offset.x(), self.offset.y())

    def to_content(self, point):
        return self.transform().inverted()[0].map(point)

    def fit(self):
        bounds = self.content_rect()
        if bounds.width() <= 0 or bounds.height() <= 0:
            return
        self.zoom = max(.00001, min((self.width() - 40) / bounds.width(), (self.height() - 40) / bounds.height()))
        self.offset = QPointF(self.width() / 2 - bounds.center().x() * self.zoom, self.height() / 2 - bounds.center().y() * self.zoom)
        self._auto_fit = True
        self.update()

    def resizeEvent(self, event):
        if self._auto_fit:
            self.fit()
        super().resizeEvent(event)

    def wheelEvent(self, event):
        steps = event.angleDelta().y() / 120
        if not steps:
            return
        anchor = self.to_content(event.position())
        self.zoom = min(64, max(.00001, self.zoom * 1.2 ** max(-4, min(4, steps))))
        self.offset = event.position() - anchor * self.zoom
        self._auto_fit = False
        self.update()
        event.accept()

    def mousePressEvent(self, event):
        self.setFocus()
        left_pan = event.button() == Qt.MouseButton.LeftButton and (self._space_pan or getattr(self, "mode", None) == "browse")
        if event.button() in (Qt.MouseButton.RightButton, Qt.MouseButton.MiddleButton) or left_pan:
            self._pan = event.position()
            self._pan_button = event.button()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()

    def mouseMoveEvent(self, event):
        if self._pan is not None:
            self.offset += event.position() - self._pan
            self._pan = event.position()
            self._auto_fit = False
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == self._pan_button:
            self._pan = None
            self._pan_button = None
            self.unsetCursor()
            event.accept()

    def focusOutEvent(self, event):
        self._space_pan = False
        self._pan = None
        self._pan_button = None
        self.unsetCursor()
        super().focusOutEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._space_pan = True
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            event.accept()
        elif event.key() == Qt.Key.Key_F and event.modifiers() == Qt.KeyboardModifier.NoModifier:
            self.fit()
            event.accept()
        else:
            super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._space_pan = False
            if self._pan is None:
                self.unsetCursor()
            event.accept()
        else:
            super().keyReleaseEvent(event)


class ContourImageCanvas(_PanZoomCanvas):
    activated = Signal()
    gesture = Signal(str, object)
    wand_finished = Signal()
    import_requested = Signal()
    tool_requested = Signal(str)
    brush_adjusted = Signal(int)

    def __init__(self, color=BEFORE_COLOR, parent=None):
        super().__init__(parent)
        # Leave room for the wand controls and calibration row at 860×640.
        # The view remains zoomable; its old 220 px minimum overlapped labels.
        self.setMinimumHeight(140)
        self.color = color
        self.frame = None
        self.presentation = None
        self.path = QPainterPath()
        self.mode = "browse"
        self.radius = 8.0
        self.editing_enabled = True
        self.show_mask = True
        self.points = []
        self.hover = None
        self._dragging = False
        self.selected = False
        self.wand_prompts = ()
        empty_layout = QVBoxLayout(self)
        empty_layout.addStretch()
        self.empty_panel = QWidget()
        empty = QVBoxLayout(self.empty_panel)
        empty.setSpacing(9)
        self.empty_title = QLabel("导入照片")
        self.empty_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = self.empty_title.font()
        font.setPointSizeF(font.pointSizeF() + 4)
        font.setBold(True)
        self.empty_title.setFont(font)
        empty.addWidget(self.empty_title)
        self.import_button = QPushButton("选择照片…")
        self.import_button.setProperty("primary", True)
        self.import_button.setAutoDefault(False)
        self.import_button.clicked.connect(self.import_requested)
        empty.addWidget(self.import_button, 0, Qt.AlignmentFlag.AlignHCenter)
        tip = QLabel("导入后自动提取轮廓，再用魔棒或手动修边")
        tip.setWordWrap(True)
        tip.setAlignment(Qt.AlignmentFlag.AlignCenter)
        tip.setProperty("muted", True)
        empty.addWidget(tip)
        empty_layout.addWidget(self.empty_panel)
        empty_layout.addStretch()
        self.setToolTip("滚轮缩放 · 右键拖动 · Esc 取消当前修正 · 多边形按 Enter 完成")

    def set_source_title(self, title):
        self.empty_title.setText(title)
        self.import_button.setText("导入" + title + "照片…")

    def content_rect(self):
        if self.frame is None:
            return super().content_rect()
        h, w = self.frame.mask.shape
        return QRectF(0, 0, w, h)

    def set_frame(self, frame, presentation, *, reset_view=False):
        self.cancel_gesture()
        previous = self.presentation
        self.frame, self.presentation = frame, presentation
        self.empty_panel.setVisible(frame is None)
        if presentation is not previous:
            self.path = contour_path(presentation) if presentation else QPainterPath()
        if reset_view:
            self.fit()
        self.update()

    def set_mode(self, mode):
        if self.mode != mode:
            self.cancel_gesture()
        self.mode = mode
        self.update()

    def cancel_gesture(self):
        self.points = []
        self._dragging = False
        self.update()

    def has_gesture(self):
        return bool(self.points)

    def _finish(self):
        points = [(p.x(), p.y()) for p in self.points]
        self.cancel_gesture()
        self.gesture.emit(self.mode, points)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), self.palette().brush(QPalette.ColorRole.Base))
        if self.frame is None:
            return
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, self.zoom < 1)
        painter.setTransform(self.transform())
        painter.drawImage(QPointF(), self.presentation.image)
        if self.show_mask:
            painter.drawImage(QPointF(), self.presentation.mask)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen = QPen(self.color, 1.5)
        pen.setCosmetic(True)
        painter.setPen(pen)
        painter.drawPath(self.path)
        axis = self.frame.axis
        o, d = QPointF(*axis.origin), QPointF(*axis.direction_point)
        direction = (d - o) / math.hypot(d.x() - o.x(), d.y() - o.y())
        extent = max(self.frame.mask.shape) * 3
        pen = QPen(QColor("#ffdf70"), 1.5, Qt.PenStyle.DashLine)
        pen.setCosmetic(True)
        painter.setPen(pen)
        painter.drawLine(o - direction * extent, o + direction * extent)
        right = QPointF(direction.y(), -direction.x())
        painter.drawLine(o - right * extent, o + right * extent)
        painter.setBrush(QColor("#ffdf70"))
        painter.drawEllipse(o, 4 / self.zoom, 4 / self.zoom)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        if self.points:
            pen = QPen(QColor("#f9e871"), 2)
            pen.setCosmetic(True)
            painter.setPen(pen)
            if self.mode == "roi" and len(self.points) == 2:
                painter.drawRect(QRectF(self.points[0], self.points[-1]).normalized())
            elif self.mode.startswith("brush"):
                painter.setOpacity(.65)
                painter.setPen(QPen(QColor("#f9e871") if self.mode.endswith("add") else QColor("#f25365"), self.radius * 2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
                if len(self.points) > 1:
                    painter.drawPolyline(QPolygonF(self.points))
                else:
                    painter.drawPoint(self.points[0])
                painter.setOpacity(1)
            else:
                painter.drawPolyline(QPolygonF(self.points + ([self.hover] if self.hover is not None else [])))
        if self.hover is not None and self.mode.startswith("brush") and self.editing_enabled:
            pen = QPen(QColor("#ffffff"), 1)
            pen.setCosmetic(True)
            painter.setPen(pen)
            painter.drawEllipse(self.hover, self.radius, self.radius)
        painter.resetTransform()
        # Label the fixed reference origin, not the changing garment's top.
        origin_screen = self.transform().map(o)
        if self.rect().adjusted(8, 8, -8, -8).contains(origin_screen.toPoint()):
            label = "零高度 · 已设置" if self.frame.axis_confirmed else "建议中线 · 待设置"
            label_width = painter.fontMetrics().horizontalAdvance(label) + 16
            rect = QRectF(max(5, min(self.width()-label_width-5, origin_screen.x()+10)), max(5, origin_screen.y()-26), label_width, 21)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(31, 38, 47, 225))
            painter.drawRoundedRect(rect, 4, 4)
            painter.setPen(QColor("#fff0b4"))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, label)
        if self.mode == "wand":
            for point, positive in self.wand_prompts:
                position = self.transform().map(QPointF(*point))
                painter.setPen(QPen(QColor("white"), 1.5))
                painter.setBrush(QColor("#157c53") if positive else QColor("#ba3149"))
                painter.drawEllipse(position, 7, 7)
                painter.drawLine(position + QPointF(-3, 0), position + QPointF(3, 0))
                if positive:
                    painter.drawLine(position + QPointF(0, -3), position + QPointF(0, 3))
            painter.setBrush(Qt.BrushStyle.NoBrush)
        if self.selected:
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(self.color, 2))
            painter.drawRect(self.rect().adjusted(1, 1, -1, -1))

    def mousePressEvent(self, event):
        self.activated.emit()
        super().mousePressEvent(event)
        if self._pan is not None or event.button() != Qt.MouseButton.LeftButton or self.frame is None or not self.editing_enabled:
            return
        p = self.to_content(event.position())
        if not self.content_rect().contains(p):
            return
        if self.mode == "wand":
            mode = "wand_negative" if event.modifiers() & Qt.KeyboardModifier.AltModifier else "wand"
            self.gesture.emit(mode, [(p.x(), p.y())])
        elif self.mode.startswith("brush") or self.mode == "roi":
            self.points = [p]
            self._dragging = True
        elif self.mode in ("axis", "calibrate"):
            self.points.append(p)
            if len(self.points) == 2:
                self._finish()
        elif self.mode.startswith("polygon"):
            self.points.append(p)
        self.update()

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        self.hover = self.to_content(event.position())
        if self._pan is not None:
            return
        if self._dragging:
            if self.mode == "roi":
                self.points = [self.points[0], self.hover]
            else:
                self.points.append(self.hover)
        self.update()

    def mouseReleaseEvent(self, event):
        was_panning = self._pan is not None
        super().mouseReleaseEvent(event)
        if not was_panning and event.button() == Qt.MouseButton.LeftButton and self._dragging:
            p = self.to_content(event.position())
            if self.mode == "roi":
                self.points = [self.points[0], p]
            else:
                self.points.append(p)
            self._finish()

    def mouseDoubleClickEvent(self, event):
        if self.editing_enabled and self.mode.startswith("polygon") and len(self.points) >= 3:
            self._finish()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.cancel_gesture()
            if self.mode == "wand":
                self.wand_finished.emit()
            event.accept()
        elif event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and self.mode.startswith("polygon"):
            if len(self.points) >= 3:
                self._finish()
            event.accept()
        elif event.key() == Qt.Key.Key_Backspace and self.mode.startswith("polygon") and self.points:
            self.points.pop()
            self.update()
            event.accept()
        elif self.editing_enabled and not self.points and event.modifiers() == Qt.KeyboardModifier.NoModifier and event.key() in (Qt.Key.Key_V, Qt.Key.Key_W, Qt.Key.Key_B, Qt.Key.Key_E, Qt.Key.Key_P, Qt.Key.Key_X):
            modes = {Qt.Key.Key_V: "browse", Qt.Key.Key_W: "wand", Qt.Key.Key_B: "brush_add", Qt.Key.Key_E: "brush_remove", Qt.Key.Key_P: "polygon_add"}
            mode = modes.get(event.key())
            if event.key() == Qt.Key.Key_X:
                mode = {"brush_add": "brush_remove", "brush_remove": "brush_add", "polygon_add": "polygon_remove", "polygon_remove": "polygon_add"}.get(self.mode)
            if mode:
                self.tool_requested.emit(mode)
            event.accept()
        elif self.editing_enabled and self.mode.startswith("brush") and not self.points and event.key() in (Qt.Key.Key_BracketLeft, Qt.Key.Key_BracketRight):
            self.brush_adjusted.emit(max(1, round(self.radius * .2)) * (-1 if event.key() == Qt.Key.Key_BracketLeft else 1))
            event.accept()
        else:
            super().keyPressEvent(event)

    def leaveEvent(self, event):
        self.hover = None
        self.update()
        super().leaveEvent(event)


class ContourOverlayCanvas(_PanZoomCanvas):
    """A common-axis photo overlay; annotations use measured native crossings."""
    height_selected = Signal(float)
    section_stepped = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.paths = []
        self.rasters = []
        self.photos = []
        self.bounds = QRectF(0, 0, 1, 1)
        self.height_line = None
        self.section = None
        self.unit = "px"
        self.background = "before"
        self._source_geometry = None
        self.setToolTip("点击样品选择对应高度 · 滚轮缩放 · 右键拖动 · 蓝色圆点为处理前，橙色方点为处理后")

    def content_rect(self):
        bounds = QRectF(self.bounds)
        if self.photos and self.background != "none":
            image, transform = self.photos[0 if self.background == "before" else 1]
            bounds = bounds.united(transform.mapRect(QRectF(0, 0, image.width(), image.height())))
        return bounds

    @staticmethod
    def _plot_layout(size):
        # Preserve a useful specimen area on small laptop windows, while the
        # same two callouts remain readable and never cover the fitted photo.
        return (30, 88) if size.height() < 360 else (38, 120)

    def fit(self):
        bounds = self.content_rect()
        if bounds.width() <= 0 or bounds.height() <= 0:
            return
        # A separate lower band holds location-linked values, never covers the
        # specimen when fitted. All labels/markers retain their screen size.
        top, bottom = self._plot_layout(self.size())
        self.zoom = max(.00001, min((self.width() - 40) / bounds.width(), (self.height() - top - bottom - 20) / bounds.height()))
        self.offset = QPointF(self.width() / 2 - bounds.center().x() * self.zoom, top + (self.height() - top - bottom) / 2 - bounds.center().y() * self.zoom)
        self._auto_fit = True
        self.update()

    def set_background(self, background):
        if background not in ("before", "after", "none"):
            raise ValueError("未知的对比底图。")
        self.background = background
        if self._auto_fit:
            self.fit()
        else:
            self.update()

    def set_comparison(self, frames, presentations, result):
        source_geometry = tuple((id(f.rgba), f.axis, f.scale) for f in frames)
        reset_view = self._auto_fit or source_geometry != self._source_geometry
        self._source_geometry = source_geometry
        self.paths = [contour_path(p, f) for f, p in zip(frames, presentations)]
        self.rasters = []
        self.photos = []
        for frame, presentation in zip(frames, presentations):
            right, down = frame.axis.basis() * frame.scale
            origin = np.asarray(frame.axis.origin)
            transform = QTransform(right[0], down[0], right[1], down[1], -float(right @ origin), -float(down @ origin))
            self.rasters.append((presentation.mask, transform))
            self.photos.append((presentation.image, transform))
        b, a = result.before_bounds, result.after_bounds
        self.bounds = QRectF(min(b[0], a[0], 0), min(b[1], a[1], 0), max(b[2], a[2], 0) - min(b[0], a[0], 0), max(b[3], a[3], 0) - min(b[1], a[1], 0))
        self.height_line = None
        self.section = None
        self.unit = result.unit
        if reset_view:
            self.fit()
        else:
            self.update()

    def set_section(self, section):
        self.section = section
        self.height_line = section.height if section is not None else None
        self.update()

    def clear(self):
        self.paths = []
        self.rasters = []
        self.photos = []
        self.section = None
        self.height_line = None
        self.update()

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        top, bottom = self._plot_layout(self.size())
        if self._pan is None and event.button() == Qt.MouseButton.LeftButton and self.paths and top <= event.position().y() < self.height() - bottom:
            height = self.to_content(event.position()).y()
            if self.bounds.top() <= height <= self.bounds.bottom():
                self.height_selected.emit(height)
                event.accept()

    def keyPressEvent(self, event):
        if self.paths and event.key() in (Qt.Key.Key_Up, Qt.Key.Key_Down) and event.modifiers() == Qt.KeyboardModifier.NoModifier:
            self.section_stepped.emit(-1 if event.key() == Qt.Key.Key_Up else 1)
            event.accept()
        else:
            super().keyPressEvent(event)

    @staticmethod
    def _number(value):
        return "—" if value is None else f"{value:.4g}"

    @classmethod
    def change_text(cls, value, unit):
        if value is None:
            return "此侧无法比较"
        if abs(value) < .0005:
            return "无变化"
        return ("向外 " if value > 0 else "向内 ") + cls._number(abs(value)) + " " + unit

    def _marker(self, painter, point, before, *, small=False):
        r = 3 if small else 5
        painter.setPen(QPen(QColor("white"), 1.6))
        painter.setBrush(BEFORE_COLOR if before else AFTER_COLOR)
        if before:
            painter.drawEllipse(point, r, r)
        else:
            painter.drawRect(QRectF(point.x()-r, point.y()-r, r*2, r*2))

    def _paint_section(self, painter, plot):
        row = self.section
        if row is None:
            return
        transform = self.transform()
        y = transform.map(QPointF(0, row.height)).y()
        if not plot.top() <= y <= plot.bottom():
            return
        painter.setPen(QPen(QColor(255, 255, 255, 200), 3))
        painter.drawLine(QPointF(plot.left(), y), QPointF(plot.right(), y))
        painter.setPen(QPen(QColor("#8b389b"), 1.2, Qt.PenStyle.DashLine))
        painter.drawLine(QPointF(plot.left(), y), QPointF(plot.right(), y))
        # Exact crossings, including inner holes/gaps; never derive these from
        # downsampled decorative paths. Front/back symbols stay distinguishable.
        for before, intervals in ((True, row.before_intervals), (False, row.after_intervals)):
            for i, interval in enumerate(intervals):
                for j, x in enumerate(interval):
                    point = transform.map(QPointF(x, row.height))
                    outer = (i == 0 and j == 0) or (i == len(intervals)-1 and j == 1)
                    self._marker(painter, point, before, small=not outer)
                if 1 < len(intervals) <= 6 and (interval[1]-interval[0])*self.zoom >= 42:
                    x = transform.map(QPointF(sum(interval)/2, row.height)).x()
                    label = f"{'前' if before else '后'}{i+1}"
                    rect = QRectF(x-18, y+(-25 if before else 10), 36, 17)
                    painter.fillRect(rect, QColor(255, 255, 255, 225))
                    painter.setPen(QColor("#176b82") if before else QColor("#9a4c0b"))
                    painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, label)
        painter.setPen(QPen(QColor("#8b389b"), 1.5))
        origin = transform.map(QPointF(0, row.height))
        painter.setBrush(QColor("white"))
        painter.drawEllipse(origin, 3, 3)

    def _paint_callouts(self, painter, size, plot):
        row = self.section
        _, bottom = self._plot_layout(size)
        if row is None:
            painter.setPen(QColor("#536473"))
            painter.drawText(QRectF(12, size.height()-bottom+10, size.width()-24, bottom-30), Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap, "点击样品或表格，定位同一高度的前后边界。")
            return
        font = painter.font()
        gap, margin = 10, 10
        width = (size.width() - 2*margin - gap) / 2
        for side in (0, 1):
            card = QRectF(margin + side*(width+gap), size.height()-bottom+10, width, bottom-38)
            # A leader links each card to its visible specimen edge. Leaders
            # are clipped to the plot so panning cannot paint over the legend.
            points = []
            for intervals in (row.before_intervals, row.after_intervals):
                if intervals:
                    x = intervals[0][0] if side == 0 else intervals[-1][1]
                    point = self.transform().map(QPointF(x, row.height))
                    if plot.contains(point):
                        points.append(point)
            if points:
                anchor = sum(points, QPointF()) / len(points)
                painter.save()
                painter.setClipRect(QRectF(0, plot.top(), size.width(), card.top()-plot.top()))
                painter.setPen(QPen(QColor("#8b389b"), 1, Qt.PenStyle.DotLine))
                painter.drawLine(anchor, QPointF(card.center().x(), card.top()))
                painter.restore()
            painter.setPen(QPen(QColor("#d2dce5"), 1))
            painter.setBrush(QColor("white"))
            painter.drawRoundedRect(card, 7, 7)
            painter.setPen(QColor("#34495d"))
            title = "左外缘" if side == 0 else "右外缘"
            compact = bottom < 120
            painter.drawText(QRectF(card.left()+10, card.top()+(0 if compact else 3), card.width()-18, 16 if compact else 18), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, title)
            change = row.left_change if side == 0 else row.right_change
            bold = painter.font()
            bold.setBold(True)
            painter.setFont(bold)
            painter.setPen(QColor("#80388c"))
            painter.drawText(QRectF(card.left()+10, card.top()+(16 if compact else 26), card.width()-18, 18 if compact else 22), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, self.change_text(change, self.unit))
            painter.setFont(font)
            before = row.left_before if side == 0 else row.right_before
            after = row.left_after if side == 0 else row.right_after
            painter.setPen(QColor("#566b7b"))
            painter.drawText(QRectF(card.left()+10, card.top()+(32 if compact else 52), card.width()-18, 18 if compact else 23), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, f"距中线：{self._number(before)} → {self._number(after)} {self.unit}")
        painter.setPen(QColor("#566b7b"))
        painter.drawText(QRectF(10, size.height()-24, size.width()-20, 20), Qt.AlignmentFlag.AlignCenter, f"参考高度 {self._number(row.height)} {self.unit} · 位置按真实比例，变化未放大")

    def paint_scene(self, painter, size):
        painter.fillRect(QRectF(0, 0, size.width(), size.height()), QColor("#f7f9fb"))
        if not self.paths:
            painter.setPen(QColor("#536473"))
            painter.drawText(QRectF(12, 12, size.width() - 24, size.height() - 24), Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap, "导入前后照片，完成轮廓与标定后显示对比")
            return
        top, bottom = self._plot_layout(size)
        plot = QRectF(0, top, size.width(), max(1, size.height()-top-bottom))
        painter.save()
        painter.setClipRect(plot)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, self.zoom < 1)
        painter.setTransform(self.transform())
        if self.background != "none":
            photo, image_transform = self.photos[0 if self.background == "before" else 1]
            painter.save()
            painter.setWorldTransform(image_transform, True)
            painter.drawImage(QPointF(), photo)
            painter.restore()
        for (mask, transform), path, color in zip(self.rasters, self.paths, (BEFORE_COLOR, AFTER_COLOR)):
            painter.save()
            painter.setWorldTransform(transform, True)
            painter.setOpacity(.4 if self.background == "none" else .25)
            painter.drawImage(QPointF(), mask)
            painter.restore()
            pen = QPen(QColor(255, 255, 255, 170), 3.5)
            pen.setCosmetic(True)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPath(path)
            pen = QPen(color, 1.8)
            pen.setCosmetic(True)
            painter.setPen(pen)
            painter.drawPath(path)
        b = self.content_rect()
        pen = QPen(QColor("#5b4479"), 1, Qt.PenStyle.DashLine)
        pen.setCosmetic(True)
        painter.setPen(pen)
        painter.drawLine(QPointF(0, b.top()), QPointF(0, b.bottom()))
        painter.drawLine(QPointF(b.left(), 0), QPointF(b.right(), 0))
        painter.resetTransform()
        self._paint_section(painter, plot)
        painter.restore()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        legend_y = top / 2
        self._marker(painter, QPointF(18, legend_y), True)
        painter.setPen(QColor("#176b82"))
        painter.drawText(QPointF(29, legend_y+4), "处理前")
        self._marker(painter, QPointF(103, legend_y), False)
        painter.setPen(QColor("#9a4c0b"))
        painter.drawText(QPointF(114, legend_y+4), "处理后")
        painter.setPen(QColor("#566b7b"))
        painter.drawText(QRectF(180, 0, max(0, size.width()-190), top), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, "点击样品定位")
        self._paint_callouts(painter, size, plot)

    def paintEvent(self, event):
        painter = QPainter(self)
        self.paint_scene(painter, self.size())
