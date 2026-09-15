"""Image-coordinate editing; display paths never supply measurement geometry."""
from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPainterPath, QPalette, QPen, QPolygonF, QTransform
from PySide6.QtWidgets import QWidget

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
        if event.button() in (Qt.MouseButton.RightButton, Qt.MouseButton.MiddleButton):
            self._pan = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()

    def mouseMoveEvent(self, event):
        if self._pan is not None:
            self.offset += event.position() - self._pan
            self._pan = event.position()
            self._auto_fit = False
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() in (Qt.MouseButton.RightButton, Qt.MouseButton.MiddleButton):
            self._pan = None
            self.unsetCursor()
            event.accept()


class ContourImageCanvas(_PanZoomCanvas):
    activated = Signal()
    gesture = Signal(str, object)
    wand_finished = Signal()

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
        self.setToolTip("滚轮缩放 · 右键拖动 · Esc 取消当前修正 · 多边形按 Enter 完成")

    def content_rect(self):
        if self.frame is None:
            return super().content_rect()
        h, w = self.frame.mask.shape
        return QRectF(0, 0, w, h)

    def set_frame(self, frame, presentation, *, reset_view=False):
        self.cancel_gesture()
        previous = self.presentation
        self.frame, self.presentation = frame, presentation
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
            painter.setPen(self.palette().color(QPalette.ColorRole.PlaceholderText))
            painter.drawText(self.rect().adjusted(14, 14, -14, -14), Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap, "导入照片后自动提取轮廓\n也可使用已打开的图片")
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
            painter.setPen(QPen(self.color, 2))
            painter.drawRect(self.rect().adjusted(1, 1, -1, -1))

    def mousePressEvent(self, event):
        self.activated.emit()
        super().mousePressEvent(event)
        if event.button() != Qt.MouseButton.LeftButton or self.frame is None or not self.editing_enabled:
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
        super().mouseReleaseEvent(event)
        if event.button() == Qt.MouseButton.LeftButton and self._dragging:
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
        else:
            super().keyPressEvent(event)

    def leaveEvent(self, event):
        self.hover = None
        self.update()
        super().leaveEvent(event)


class ContourOverlayCanvas(_PanZoomCanvas):
    """A common-axis photo overlay; annotations use measured native crossings."""
    height_selected = Signal(float)

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
        self.setToolTip("点击样品选择对应高度 · 滚轮缩放 · 右键拖动 · 蓝色圆点为处理前，橙色方点为处理后")

    def content_rect(self):
        bounds = QRectF(self.bounds)
        if self.photos and self.background != "none":
            image, transform = self.photos[0 if self.background == "before" else 1]
            bounds = bounds.united(transform.mapRect(QRectF(0, 0, image.width(), image.height())))
        return bounds

    def fit(self):
        bounds = self.content_rect()
        if bounds.width() <= 0 or bounds.height() <= 0:
            return
        # A separate lower band holds location-linked values, never covers the
        # specimen when fitted. All labels/markers retain their screen size.
        top, bottom = 42, 120
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
        self.fit()

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
        if event.button() == Qt.MouseButton.LeftButton and self.paths and 38 <= event.position().y() < self.height() - 120:
            height = self.to_content(event.position()).y()
            if self.bounds.top() <= height <= self.bounds.bottom():
                self.height_selected.emit(height)
                event.accept()

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
        if row is None:
            painter.setPen(QColor("#536473"))
            painter.drawText(QRectF(12, size.height()-105, size.width()-24, 75), Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap, "点击样品或表格，定位同一高度的前后边界。")
            return
        font = painter.font()
        gap, margin = 10, 10
        width = (size.width() - 2*margin - gap) / 2
        for side in (0, 1):
            card = QRectF(margin + side*(width+gap), size.height()-110, width, 82)
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
            painter.drawText(card.adjusted(10, 5, -8, -57), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, title)
            change = row.left_change if side == 0 else row.right_change
            bold = painter.font()
            bold.setBold(True)
            painter.setFont(bold)
            painter.setPen(QColor("#80388c"))
            painter.drawText(card.adjusted(10, 26, -8, -32), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, self.change_text(change, self.unit))
            painter.setFont(font)
            before = row.left_before if side == 0 else row.right_before
            after = row.left_after if side == 0 else row.right_after
            painter.setPen(QColor("#566b7b"))
            painter.drawText(card.adjusted(10, 50, -8, -4), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter | Qt.TextFlag.TextWordWrap, f"距中线：{self._number(before)} → {self._number(after)} {self.unit}")
        painter.setPen(QColor("#566b7b"))
        painter.drawText(QRectF(10, size.height()-24, size.width()-20, 20), Qt.AlignmentFlag.AlignCenter, f"参考高度 {self._number(row.height)} {self.unit} · 位置按真实比例，变化未放大")

    def paint_scene(self, painter, size):
        painter.fillRect(QRectF(0, 0, size.width(), size.height()), QColor("#f7f9fb"))
        if not self.paths:
            painter.setPen(QColor("#536473"))
            painter.drawText(QRectF(12, 12, size.width() - 24, size.height() - 24), Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap, "导入前后照片，完成轮廓与标定后显示对比")
            return
        plot = QRectF(0, 38, size.width(), max(1, size.height()-158))
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
        self._marker(painter, QPointF(18, 20), True)
        painter.setPen(QColor("#176b82"))
        painter.drawText(29, 24, "处理前")
        self._marker(painter, QPointF(103, 20), False)
        painter.setPen(QColor("#9a4c0b"))
        painter.drawText(114, 24, "处理后")
        painter.setPen(QColor("#566b7b"))
        painter.drawText(QRectF(180, 8, max(0, size.width()-190), 25), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, "点击样品定位")
        self._paint_callouts(painter, size, plot)

    def paintEvent(self, event):
        painter = QPainter(self)
        self.paint_scene(painter, self.size())
