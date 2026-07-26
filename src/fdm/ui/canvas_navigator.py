from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import QEvent, QObject, QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QEnterEvent,
    QImage,
    QMouseEvent,
    QPaintEvent,
    QPainter,
    QPalette,
    QPen,
    QWheelEvent,
)
from PySide6.QtWidgets import QSizePolicy, QWidget

from fdm.geometry import Point

if TYPE_CHECKING:
    from fdm.ui.view_transform import CanvasViewportSnapshot


_THUMBNAIL_MAX_EDGE = 256
_WIDGET_SIZE = QSize(176, 120)
_CONTENT_MARGIN = 8.0
_COORDINATE_EPSILON = 1e-6


class CanvasNavigatorWidget(QWidget):
    """Small overview used to locate and recenter the active image viewport.

    The widget deliberately owns only a small derived thumbnail.  View changes
    update the rectangles painted over that thumbnail and never rescale the
    source image.  A digital slide can therefore use the same control without
    asking its store to render the complete slide.
    """

    centerRequested = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("canvasNavigator")
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.setFixedSize(_WIDGET_SIZE)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip("导航概览：单击或拖动可移动到图片中的对应位置")

        self._snapshot: CanvasViewportSnapshot | None = None
        self._thumbnail = QImage()
        self._source_image_key: tuple[int, int, int, int] | None = None
        self._navigator_enabled = True
        self._dragging = False
        self._hovered = False
        self._thumbnail_build_count = 0
        if parent is not None:
            parent.installEventFilter(self)
        self.hide()

    @property
    def navigator_enabled(self) -> bool:
        return self._navigator_enabled

    @property
    def thumbnail_build_count(self) -> int:
        """Number of thumbnail derivations, useful for diagnostics/benchmarks."""

        return self._thumbnail_build_count

    def sizeHint(self) -> QSize:
        return QSize(_WIDGET_SIZE)

    def minimumSizeHint(self) -> QSize:
        return QSize(120, 84)

    def set_navigator_enabled(self, enabled: bool) -> None:
        enabled = bool(enabled)
        if self._navigator_enabled == enabled:
            return
        self._navigator_enabled = enabled
        self._sync_visibility()

    def set_viewport_snapshot(
        self,
        snapshot: CanvasViewportSnapshot | None,
    ) -> None:
        if snapshot == self._snapshot:
            return
        self._snapshot = snapshot
        self._sync_visibility()
        if not self.isHidden():
            self.update()

    def set_source_image(self, image: QImage | None) -> None:
        """Cache a small thumbnail only when the source image really changes."""

        if image is None or image.isNull():
            if self._source_image_key is None and self._thumbnail.isNull():
                return
            self._source_image_key = None
            self._thumbnail = QImage()
            self.update()
            return

        image_key = (
            int(image.cacheKey()),
            int(image.width()),
            int(image.height()),
            int(image.format().value),
        )
        if image_key == self._source_image_key:
            return

        self._source_image_key = image_key
        self._thumbnail = image.scaled(
            _THUMBNAIL_MAX_EDGE,
            _THUMBNAIL_MAX_EDGE,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._thumbnail_build_count += 1
        self.update()

    def clear(self) -> None:
        self._snapshot = None
        self._source_image_key = None
        self._thumbnail = QImage()
        self._dragging = False
        self.hide()

    def place_at_top_right(self, *, margin: int = 12) -> None:
        """Position this child at the conventional top-right canvas corner."""

        parent = self.parentWidget()
        if parent is None:
            return
        x = max(margin, parent.width() - self.width() - margin)
        self.move(x, max(0, int(margin)))
        self.raise_()

    def content_rect(self) -> QRectF:
        snapshot = self._snapshot
        if snapshot is None:
            return QRectF()
        full_rect = QRectF(snapshot.full_image_rect)
        if not _valid_rect(full_rect):
            return QRectF()

        available = QRectF(self.rect()).adjusted(
            _CONTENT_MARGIN,
            _CONTENT_MARGIN,
            -_CONTENT_MARGIN,
            -_CONTENT_MARGIN,
        )
        if not _valid_rect(available):
            return QRectF()
        scale = min(
            available.width() / full_rect.width(),
            available.height() / full_rect.height(),
        )
        width = full_rect.width() * scale
        height = full_rect.height() * scale
        return QRectF(
            available.center().x() - width / 2.0,
            available.center().y() - height / 2.0,
            width,
            height,
        )

    def map_image_point_to_widget(self, point: Point | QPointF) -> QPointF:
        snapshot = self._snapshot
        content = self.content_rect()
        if snapshot is None or not _valid_rect(content):
            return QPointF()
        full_rect = QRectF(snapshot.full_image_rect)
        x = float(point.x if isinstance(point, Point) else point.x())
        y = float(point.y if isinstance(point, Point) else point.y())
        return QPointF(
            content.left()
            + ((x - full_rect.left()) / full_rect.width()) * content.width(),
            content.top()
            + ((y - full_rect.top()) / full_rect.height()) * content.height(),
        )

    def map_widget_point_to_image(self, point: QPointF) -> Point | None:
        snapshot = self._snapshot
        content = self.content_rect()
        if snapshot is None or not _valid_rect(content):
            return None
        full_rect = QRectF(snapshot.full_image_rect)
        clamped_x = min(content.right(), max(content.left(), point.x()))
        clamped_y = min(content.bottom(), max(content.top(), point.y()))
        return Point(
            x=full_rect.left()
            + ((clamped_x - content.left()) / content.width())
            * full_rect.width(),
            y=full_rect.top()
            + ((clamped_y - content.top()) / content.height())
            * full_rect.height(),
        )

    def map_image_rect_to_widget(self, image_rect: QRectF) -> QRectF:
        snapshot = self._snapshot
        content = self.content_rect()
        if snapshot is None or not _valid_rect(content):
            return QRectF()
        full_rect = QRectF(snapshot.full_image_rect)
        clipped = QRectF(image_rect).intersected(full_rect)
        if not _valid_rect(clipped):
            return QRectF()
        top_left = self.map_image_point_to_widget(clipped.topLeft())
        bottom_right = self.map_image_point_to_widget(clipped.bottomRight())
        return QRectF(top_left, bottom_right).normalized()

    def paintEvent(self, event: QPaintEvent) -> None:
        del event
        snapshot = self._snapshot
        content = self.content_rect()
        if snapshot is None or not _valid_rect(content):
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

        palette = self.palette()
        background, border, foreground, highlight = _navigator_colors(
            palette,
            hovered=self._hovered or self._dragging,
        )
        painter.setPen(QPen(border, 1.0))
        painter.setBrush(background)
        painter.drawRoundedRect(QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5), 8.0, 8.0)

        painter.save()
        painter.setClipRect(content)
        if not self._thumbnail.isNull():
            painter.drawImage(content, self._thumbnail)
            painter.fillRect(content, QColor(0, 0, 0, 18))
        else:
            self._draw_schematic_background(
                painter,
                content,
                foreground=foreground,
                border=border,
            )
        painter.restore()

        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(border, 1.0))
        painter.drawRect(content)

        mounted = self.map_image_rect_to_widget(snapshot.mounted_image_rect)
        if _valid_rect(mounted):
            mounted_pen = QPen(foreground, 1.0, Qt.PenStyle.DashLine)
            mounted_pen.setCosmetic(True)
            painter.setPen(mounted_pen)
            painter.setBrush(QColor(foreground.red(), foreground.green(), foreground.blue(), 18))
            painter.drawRect(mounted)

        visible = self.map_image_rect_to_widget(snapshot.visible_image_rect)
        if _valid_rect(visible):
            viewport_pen = QPen(highlight, 2.0)
            viewport_pen.setCosmetic(True)
            painter.setPen(viewport_pen)
            painter.setBrush(QColor(highlight.red(), highlight.green(), highlight.blue(), 32))
            painter.drawRect(visible)

        painter.end()

    def _draw_schematic_background(
        self,
        painter: QPainter,
        content: QRectF,
        *,
        foreground: QColor,
        border: QColor,
    ) -> None:
        """Draw a cheap, honest overview when no whole-slide thumbnail exists."""

        base = self.palette().color(QPalette.ColorRole.Base)
        base.setAlpha(225)
        painter.fillRect(content, base)

        grid_color = QColor(border)
        grid_color.setAlpha(95)
        grid_pen = QPen(grid_color, 1.0)
        grid_pen.setCosmetic(True)
        painter.setPen(grid_pen)
        for fraction in (0.25, 0.5, 0.75):
            x = content.left() + content.width() * fraction
            y = content.top() + content.height() * fraction
            painter.drawLine(QPointF(x, content.top()), QPointF(x, content.bottom()))
            painter.drawLine(QPointF(content.left(), y), QPointF(content.right(), y))

        center_color = QColor(foreground)
        center_color.setAlpha(80)
        painter.setPen(QPen(center_color, 1.0, Qt.PenStyle.DotLine))
        painter.drawLine(content.topLeft(), content.bottomRight())
        painter.drawLine(content.topRight(), content.bottomLeft())

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = self.content_rect().contains(event.position())
            if self._dragging:
                self._emit_center_request(event.position())
                self.update()
        event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._dragging and event.buttons() & Qt.MouseButton.LeftButton:
            self._emit_center_request(event.position())
        event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._dragging:
            self._emit_center_request(event.position())
            self._dragging = False
            self.update()
        event.accept()

    def wheelEvent(self, event: QWheelEvent) -> None:
        # The navigator is a location control.  Consuming the wheel prevents an
        # accidental zoom change while the user scrolls over the translucent UI.
        event.accept()

    def enterEvent(self, event: QEnterEvent) -> None:
        self._hovered = True
        self.update()
        super().enterEvent(event)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if watched is self.parentWidget() and event.type() in {
            QEvent.Type.Resize,
            QEvent.Type.Show,
        }:
            self.place_at_top_right()
        return super().eventFilter(watched, event)

    def leaveEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        self._hovered = False
        if not self._dragging:
            self.update()
        super().leaveEvent(event)

    def _emit_center_request(self, widget_point: QPointF) -> None:
        image_point = self.map_widget_point_to_image(widget_point)
        if image_point is not None:
            self.centerRequested.emit(image_point)

    def _sync_visibility(self) -> None:
        should_show = (
            self._navigator_enabled
            and self._snapshot is not None
            and not _full_image_is_visible(self._snapshot)
        )
        self.setVisible(should_show)
        if should_show:
            self.place_at_top_right()


def _valid_rect(rect: QRectF) -> bool:
    return (
        rect.isValid()
        and rect.width() > _COORDINATE_EPSILON
        and rect.height() > _COORDINATE_EPSILON
    )


def _full_image_is_visible(snapshot: CanvasViewportSnapshot) -> bool:
    full = QRectF(snapshot.full_image_rect)
    visible = QRectF(snapshot.visible_image_rect)
    if not _valid_rect(full) or not _valid_rect(visible):
        return False
    tolerance = max(
        _COORDINATE_EPSILON,
        max(full.width(), full.height()) * 1e-9,
    )
    return (
        visible.left() <= full.left() + tolerance
        and visible.top() <= full.top() + tolerance
        and visible.right() >= full.right() - tolerance
        and visible.bottom() >= full.bottom() - tolerance
    )


def _navigator_colors(
    palette: QPalette,
    *,
    hovered: bool,
) -> tuple[QColor, QColor, QColor, QColor]:
    background = QColor(palette.color(QPalette.ColorRole.Window))
    background.setAlpha(236 if hovered else 205)
    border = QColor(palette.color(QPalette.ColorRole.Mid))
    border.setAlpha(240)
    foreground = QColor(palette.color(QPalette.ColorRole.WindowText))
    foreground.setAlpha(190)
    highlight = QColor(palette.color(QPalette.ColorRole.Highlight))
    highlight.setAlpha(255)
    return background, border, foreground, highlight
