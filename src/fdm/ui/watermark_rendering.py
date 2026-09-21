"""Shared canvas/export watermark compositor; never paints into source pixels."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import struct

from PySide6.QtCore import QByteArray, QBuffer, QIODevice, QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QFont, QFontMetricsF, QImage, QImageReader, QPainter, QTransform

from fdm.geometry import Point
from fdm.services.watermark_assets import verify_logo
from fdm.watermark import ANCHORS, WatermarkSpec


class WatermarkRasterCache:
    def __init__(self, budget: int = 32 * 1024 * 1024) -> None:
        self.budget = budget
        self.clear()

    def clear(self) -> None:
        self._items: OrderedDict[tuple, QImage] = OrderedDict()
        self.bytes = self.hits = self.misses = self.logo_decodes = 0

    def get(self, key: tuple) -> QImage | None:
        value = self._items.get(key)
        if value is None:
            self.misses += 1
        else:
            self._items.move_to_end(key)
            self.hits += 1
        return value

    def put(self, key: tuple, image: QImage) -> QImage:
        size = image.sizeInBytes()
        if size > self.budget:
            return image
        old = self._items.pop(key, None)
        if old is not None:
            self.bytes -= old.sizeInBytes()
        while self._items and self.bytes + size > self.budget:
            _, removed = self._items.popitem(last=False)
            self.bytes -= removed.sizeInBytes()
        self._items[key] = image
        self.bytes += size
        return image


watermark_raster_cache = WatermarkRasterCache()


def import_logo(path: str | Path) -> tuple[str, bytes]:
    reader = QImageReader(str(path))
    reader.setAutoTransform(True)
    if bytes(reader.format()).lower() not in {b"png", b"jpg", b"jpeg", b"webp"}:
        raise ValueError("Logo 支持 PNG、JPEG、WebP 图片")
    image = reader.read()
    if image.isNull():
        raise ValueError(f"无法读取 Logo：{reader.errorString()}")
    buffer = QBuffer()
    buffer.open(QIODevice.OpenModeFlag.WriteOnly)
    if not image.save(buffer, "PNG"):
        raise ValueError("无法生成水印 Logo PNG")
    data = bytes(buffer.data())
    return hashlib.sha256(data).hexdigest(), data


def _logo_data(document, spec: WatermarkSpec) -> bytes:
    data = document.watermark_assets.get(spec.logo_sha256)
    if data is None:
        raise ValueError("水印 Logo 缺失，请通过“图像 → 水印”重新选择 Logo")
    return data


def _logo_dimensions(document, spec: WatermarkSpec) -> tuple[int, int]:
    # Imported and loaded assets are normalized PNGs. Reading their mandatory
    # first IHDR chunk needs only 24 bytes, even when the decoded image is too
    # large for the LRU. Cached stamps can then be reused without decoding the
    # original Logo or copying its encoded bytes on every frame.
    data = _logo_data(document, spec)
    if len(data) < 24 or not data.startswith(b"\x89PNG\r\n\x1a\n") or data[12:16] != b"IHDR":
        raise ValueError("水印 Logo PNG 头损坏，请重新选择图片")
    width, height = struct.unpack_from(">II", data, 16)
    if width == 0 or height == 0:
        raise ValueError("水印 Logo 尺寸无效，请重新选择图片")
    return width, height


def logo_image(document, spec: WatermarkSpec) -> QImage:
    data = _logo_data(document, spec)
    key = ("logo", spec.logo_sha256)
    image = watermark_raster_cache.get(key)
    if image is None:
        verify_logo(data, spec.logo_sha256)
        image = QImage.fromData(QByteArray(data), "PNG")
        if image.isNull():
            raise ValueError("水印 Logo PNG 无法解码，请重新选择图片")
        image = image.convertToFormat(QImage.Format.Format_ARGB32_Premultiplied)
        watermark_raster_cache.logo_decodes += 1
        watermark_raster_cache.put(key, image)
    return image


@dataclass(frozen=True)
class WatermarkGeometry:
    width: float
    height: float
    rotated: QRectF
    font: QFont
    lines: tuple[str, ...]
    text_width: float
    line_height: float
    ascent: float
    content_height: float
    datetime_line: str
    datetime_scale: float
    datetime_top: float


def watermark_geometry(document, spec: WatermarkSpec) -> WatermarkGeometry:
    spec.validate_content()
    font = QFont(spec.font_family) if spec.font_family else QFont()
    font.setPixelSize(64)
    font.setBold(spec.bold)
    font.setItalic(spec.italic)
    metrics = QFontMetricsF(font)
    lines = tuple(spec.text.split("\n"))
    # Include bearings so italic glyphs do not get cut at the stamp edges.
    text_width = max((metrics.boundingRect(line).width() for line in lines), default=1) + 8
    width = max(0.01, min(document.image_size) * spec.width_ratio)
    if spec.kind == "logo":
        logo_width, logo_height = _logo_dimensions(document, spec)
        height = width * logo_height / logo_width
    else:
        height = width * (metrics.lineSpacing() * len(lines) + 8) / text_width
    content_height = height
    datetime_line = spec.datetime_text if spec.include_datetime else ""
    datetime_scale = 0.0
    datetime_top = height
    if datetime_line:
        datetime_width = metrics.boundingRect(datetime_line).width() + 8
        datetime_scale = width / datetime_width
        if spec.kind == "text":
            datetime_scale = min(datetime_scale, 0.8 * width / text_width)
        gap = min(width * 0.03, metrics.lineSpacing() * datetime_scale * 0.3)
        datetime_top += gap
        height = datetime_top + (metrics.lineSpacing() + 8) * datetime_scale
    rotated = QTransform().rotate(spec.rotation).mapRect(QRectF(0, 0, width, height))
    return WatermarkGeometry(
        width, height, rotated, font, lines, text_width, metrics.lineSpacing(), metrics.ascent(),
        content_height, datetime_line, datetime_scale, datetime_top,
    )


def _draw_content(painter, document, spec, geometry):
    painter.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.TextAntialiasing | QPainter.RenderHint.SmoothPixmapTransform)
    if spec.kind == "logo":
        painter.drawImage(QRectF(0, 0, geometry.width, geometry.content_height), logo_image(document, spec))
    else:
        painter.save()
        try:
            painter.scale(geometry.width / geometry.text_width, geometry.width / geometry.text_width)
            painter.setFont(geometry.font)
            painter.setPen(QColor(spec.color))
            metrics = QFontMetricsF(geometry.font)
            for index, line in enumerate(geometry.lines):
                left = metrics.boundingRect(line).left()
                painter.drawText(QPointF(4 - left, 4 + geometry.ascent + index * geometry.line_height), line)
        finally:
            painter.restore()
    if geometry.datetime_line:
        painter.save()
        try:
            painter.translate(0, geometry.datetime_top)
            painter.scale(geometry.datetime_scale, geometry.datetime_scale)
            painter.setFont(geometry.font)
            painter.setPen(QColor(spec.color))
            left = QFontMetricsF(geometry.font).boundingRect(geometry.datetime_line).left()
            painter.drawText(QPointF(4 - left, 4 + geometry.ascent), geometry.datetime_line)
        finally:
            painter.restore()


def _stamp(document, spec, geometry, resolution: float) -> tuple[QImage | None, QRectF]:
    key = ("stamp", spec, tuple(document.image_size), round(resolution, 8))
    bounds = geometry.rotated
    source = QRectF(2, 2, bounds.width() * resolution, bounds.height() * resolution)
    image = watermark_raster_cache.get(key)
    if image is not None:
        return image, source
    width, height = max(1, math.ceil(source.width()) + 4), max(1, math.ceil(source.height()) + 4)
    if width * height * 4 > watermark_raster_cache.budget:
        # A zoomed-in watermark can be larger than the whole viewport. Draw
        # its clipped primitives directly instead of allocating a huge mask.
        return None, source
    image = QImage(width, height, QImage.Format.Format_ARGB32_Premultiplied)
    if image.isNull():
        raise ValueError("水印尺寸过大，无法创建绘制缓存")
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    try:
        painter.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.TextAntialiasing | QPainter.RenderHint.SmoothPixmapTransform)
        painter.translate(2, 2)
        painter.scale(resolution, resolution)
        painter.translate(-bounds.left(), -bounds.top())
        painter.rotate(spec.rotation)
        _draw_content(painter, document, spec, geometry)
    finally:
        painter.end()
    return watermark_raster_cache.put(key, image), source


_CURRENT_SPEC = object()


def draw_watermark(painter: QPainter, document, image_to_output, *, spec=_CURRENT_SPEC, strict: bool = False) -> bool:
    """Paint one watermark group. Asset errors are fatal only for exports.

    The mapper returns logical output coordinates. QImage's DPR and an existing
    painter transform are preserved; image-relative repeat periods never depend
    on viewport origin, cache pixel rounding, or the annotation text size mode.
    """
    if document.document_kind != "image":
        return False
    if spec is _CURRENT_SPEC:
        spec = document.watermark
    if spec is None or not spec.enabled:
        return False
    try:
        geometry = watermark_geometry(document, spec)
        if spec.opacity == 0:
            return True
        origin = image_to_output(Point(0, 0))
        unit = image_to_output(Point(1, 1))
        sx, sy = unit.x() - origin.x(), unit.y() - origin.y()
        if sx <= 0 or sy <= 0:
            return False
        dpr = painter.device().devicePixelRatioF()
        resolution = max(sx, sy) * dpr
        image, source = _stamp(document, spec, geometry, resolution)
        width, height = document.image_size
        bounds = geometry.rotated
        bw, bh = bounds.width(), bounds.height()
        output_bounds = QRectF(origin.x(), origin.y(), width * sx, height * sy)

        def draw_instance(x, y):
            if image is not None:
                painter.drawImage(QRectF(origin.x() + x * sx, origin.y() + y * sy, bw * sx, bh * sy), image, source)
            else:
                painter.save()
                try:
                    painter.translate(origin.x() + x * sx, origin.y() + y * sy)
                    painter.scale(sx, sy)
                    painter.translate(-bounds.left(), -bounds.top())
                    painter.rotate(spec.rotation)
                    _draw_content(painter, document, spec, geometry)
                finally:
                    painter.restore()

        painter.save()
        try:
            painter.setClipRect(output_bounds, Qt.ClipOperation.IntersectClip)
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
            painter.setOpacity(painter.opacity() * spec.opacity)
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
            if spec.layout == "single":
                column, row = ANCHORS[spec.anchor]
                x = (width - bw) * column / 2 + width * spec.offset_x * (-1 if column == 2 else 1)
                y = (height - bh) * row / 2 + height * spec.offset_y * (-1 if row == 2 else 1)
                draw_instance(x, y)
            else:
                period_x, period_y = bw * (1 + spec.gap_x), bh * (1 + spec.gap_y)
                tw, th = max(1, math.ceil(period_x * resolution)), max(1, math.ceil(period_y * resolution))
                key = ("tile", spec, tuple(document.image_size), round(resolution, 8))
                texture = watermark_raster_cache.get(key)
                if image is not None and texture is None and tw * th * 4 <= watermark_raster_cache.budget:
                    texture = QImage(tw, th, QImage.Format.Format_ARGB32_Premultiplied)
                    if texture.isNull():
                        raise ValueError("无法创建水印平铺缓存")
                    texture.fill(Qt.GlobalColor.transparent)
                    tile_painter = QPainter(texture)
                    tile_painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
                    tile_painter.drawImage(QRectF(0, 0, tw * bw / period_x, th * bh / period_y), image, source)
                    tile_painter.end()
                    watermark_raster_cache.put(key, texture)
                x0, y0 = width * spec.offset_x, height * spec.offset_y
                if texture is not None:
                    brush = QBrush(texture)
                    brush.setTransform(QTransform(period_x * sx / tw, 0, 0, period_y * sy / th, origin.x() + x0 * sx, origin.y() + y0 * sy))
                    painter.fillRect(output_bounds, brush)
                else:
                    # Very large/sparse cells use visible stamps, without an
                    # oversized transparent repeat texture or quality reduction.
                    visible = painter.clipBoundingRect().intersected(output_bounds)
                    device = painter.device()
                    if device.width() > 0 and device.height() > 0:
                        inverse, invertible = painter.deviceTransform().inverted()
                        if invertible:
                            visible = visible.intersected(inverse.mapRect(QRectF(0, 0, device.width(), device.height())))
                    left, top = (visible.left() - origin.x()) / sx, (visible.top() - origin.y()) / sy
                    right, bottom = (visible.right() - origin.x()) / sx, (visible.bottom() - origin.y()) / sy
                    for row in range(math.floor((top - y0 - bh) / period_y) + 1, math.ceil((bottom - y0) / period_y)):
                        for column in range(math.floor((left - x0 - bw) / period_x) + 1, math.ceil((right - x0) / period_x)):
                            draw_instance(x0 + column * period_x, y0 + row * period_y)
        finally:
            painter.restore()
        return True
    except (ValueError, OSError) as exc:
        if strict:
            raise
        document.watermark_asset_error = str(exc)
        return False
