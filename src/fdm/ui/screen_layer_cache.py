"""Shared, bounded display rasters for stable active-object emphasis."""

from collections import OrderedDict
import math
from PySide6.QtCore import QPointF, QRectF
from PySide6.QtGui import QImage, QPainter, QTransform


class ScreenLayerCache:
    def __init__(self, max_bytes=32 * 1024 * 1024):
        self.max_bytes = max_bytes
        self.bytes = 0
        self.builds = 0
        self._entries = OrderedDict()

    def discard(self, owner):
        for key in [
            key
            for key in self._entries
            if key[0] == owner or (isinstance(key[0], tuple) and key[0][0] == owner)
        ]:
            self.bytes -= self._entries.pop(key).sizeInBytes()

    def draw(self, painter, *, owner, version, bounds, origin, zoom, dpr, viewport, render):
        transform = QTransform.fromTranslate(origin.x(), origin.y())
        transform.scale(zoom, zoom)
        inverse, _ = transform.inverted()
        visible = inverse.mapRect(viewport.adjusted(-256, -256, 256, 256))
        quantum = 512 / zoom
        left = math.floor(visible.left() / quantum) * quantum
        top = math.floor(visible.top() / quantum) * quantum
        clip = QRectF(
            left,
            top,
            math.ceil(visible.right() / quantum) * quantum - left,
            math.ceil(visible.bottom() / quantum) * quantum - top,
        )
        bounds = bounds.intersected(clip)
        if bounds.isEmpty():
            return True
        screen = transform.mapRect(bounds)
        left = math.floor(screen.left() * dpr) / dpr
        top = math.floor(screen.top() * dpr) / dpr
        width = math.ceil(screen.right() * dpr) - round(left * dpr)
        height = math.ceil(screen.bottom() * dpr) - round(top * dpr)
        if width * height * 4 > self.max_bytes or width <= 0 or height <= 0:
            return False
        key = (
            owner,
            version,
            zoom,
            dpr,
            round(left - origin.x(), 7),
            round(top - origin.y(), 7),
            width,
            height,
        )
        raster = self._entries.get(key)
        if raster is None:
            raster = QImage(width, height, QImage.Format.Format_ARGB32_Premultiplied)
            raster.setDevicePixelRatio(dpr)
            raster.fill(0)
            layer = QPainter(raster)
            try:
                layer.setRenderHint(QPainter.RenderHint.Antialiasing, True)
                layer.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
                layer.translate(-left, -top)
                layer.setClipRect(QRectF(left, top, width / dpr, height / dpr))
                render(layer)
            finally:
                layer.end()
            self.builds += 1
            for previous in [previous for previous in self._entries if previous[0] == owner]:
                self.bytes -= self._entries.pop(previous).sizeInBytes()
            while self._entries and self.bytes + raster.sizeInBytes() > self.max_bytes:
                self.bytes -= self._entries.popitem(last=False)[1].sizeInBytes()
            self._entries[key] = raster
            self.bytes += raster.sizeInBytes()
        self._entries.move_to_end(key)
        painter.drawImage(QPointF(left, top), raster)
        return True


screen_layer_cache = ScreenLayerCache()
