"""Byte-bounded, paint-only exposure and gamma mapping for slide images."""
from __future__ import annotations

from collections import OrderedDict
from functools import lru_cache
import math

import cv2
import numpy as np
from PySide6.QtCore import QRect, QRectF, Qt
from PySide6.QtGui import QImage, QPainter


DISPLAY_BRIGHTNESS_CACHE_BYTES = 96 * 1024 * 1024
_TILE_EDGE = 1024


@lru_cache(maxsize=201)
def brightness_lut(value: int) -> np.ndarray:
    """Positive values brighten; zero is exactly the identity curve."""
    value = max(-100, min(100, int(value)))
    gamma = 2.0 ** (value / 50.0)
    lut = np.rint(255.0 * (np.arange(256) / 255.0) ** (1.0 / gamma)).astype(np.uint8)
    lut.flags.writeable = False
    return lut


@lru_cache(maxsize=256)
def display_adjustments_lut(value: int, exposure_tenths: int = 0) -> np.ndarray:
    """Compose sRGB viewing exposure and legacy gamma with one quantization.

    The display raster is already eight-bit RGB, not sensor-linear data. Treat
    its components as sRGB for this viewing compensation, without changing its
    color-space tag or introducing a separate color-management pipeline.
    """
    value = max(-100, min(100, int(value)))
    exposure_tenths = max(-40, min(40, int(exposure_tenths)))
    if exposure_tenths == 0:
        # Avoid numerical round-trip differences from the existing gamma LUT.
        return brightness_lut(value)
    encoded = np.arange(256, dtype=np.float64) / 255.0
    linear = np.where(encoded <= 0.04045, encoded / 12.92,
                      ((encoded + 0.055) / 1.055) ** 2.4)
    linear = np.clip(linear * 2.0 ** (exposure_tenths / 10.0), 0.0, 1.0)
    encoded = np.where(linear <= 0.0031308, linear * 12.92,
                       1.055 * linear ** (1.0 / 2.4) - 0.055)
    gamma = 2.0 ** (value / 50.0)
    lut = np.rint(255.0 * encoded ** (1.0 / gamma)).astype(np.uint8)
    lut.flags.writeable = False
    return lut


class SlideBrightnessCache:
    def __init__(self, max_bytes: int = DISPLAY_BRIGHTNESS_CACHE_BYTES) -> None:
        self.max_bytes = max(0, int(max_bytes))
        self.bytes_used = 0
        self._images: OrderedDict[tuple, QImage] = OrderedDict()

    def clear(self) -> None:
        self._images.clear()
        self.bytes_used = 0

    def mapped_image(self, image: QImage, value: int, source_rect: QRect | None = None,
                     *, exposure_tenths: int = 0) -> QImage:
        value = max(-100, min(100, int(value)))
        exposure_tenths = max(-40, min(40, int(exposure_tenths)))
        if image.isNull():
            return image
        rect = image.rect() if source_rect is None else source_rect.intersected(image.rect())
        if rect.isEmpty():
            return QImage()
        # No pixel copy, conversion, or cache allocation on the default path.
        if value == 0 and exposure_tenths == 0:
            return image if rect == image.rect() else image.copy(rect)
        key = (image.cacheKey(), value, exposure_tenths,
               rect.x(), rect.y(), rect.width(), rect.height())
        cached = self._images.get(key)
        if cached is not None:
            self._images.move_to_end(key)
            return QImage(cached)
        fmt = image.format()
        channels = {
            QImage.Format.Format_Grayscale8: 1,
            QImage.Format.Format_RGB888: 3,
            QImage.Format.Format_BGR888: 3,
            QImage.Format.Format_RGB32: 4,
            QImage.Format.Format_ARGB32: 4,
            QImage.Format.Format_RGBA8888: 4,
            QImage.Format.Format_RGBX8888: 4,
        }.get(fmt)
        if channels is None:
            # Unpremultiply before applying a nonlinear curve. Crop before
            # conversion so an oversized source never requires a full copy.
            source = image.copy(rect).convertToFormat(QImage.Format.Format_RGBA8888)
            fmt, channels = source.format(), 4
            source_rect = source.rect()
        else:
            source, source_rect = image, rect
        pixels = np.ndarray(
            (source.height(), source.width(), channels), dtype=np.uint8,
            buffer=source.constBits(), strides=(source.bytesPerLine(), channels, 1),
        )
        pixels = pixels[source_rect.y():source_rect.y() + source_rect.height(),
                        source_rect.x():source_rect.x() + source_rect.width()]
        mapped = QImage(rect.width(), rect.height(), fmt)
        output = np.ndarray(
            (mapped.height(), mapped.width(), channels), dtype=np.uint8,
            buffer=mapped.bits(), strides=(mapped.bytesPerLine(), channels, 1),
        )
        lut = display_adjustments_lut(value, exposure_tenths)
        if channels == 4:
            # RGB32/ARGB32 byte order on the supported little-endian hosts is
            # B,G,R,A; RGBA8888 is R,G,B,A. Both preserve alpha in byte three.
            lut = np.repeat(lut[:, None, None], 4, axis=2)
            lut[:, 0, 3] = np.arange(256, dtype=np.uint8)
        cv2.LUT(pixels, lut, dst=output)
        mapped.setColorSpace(image.colorSpace())
        mapped.setDevicePixelRatio(image.devicePixelRatio())
        size = mapped.sizeInBytes()
        if size <= self.max_bytes:
            while self._images and self.bytes_used + size > self.max_bytes:
                _, evicted = self._images.popitem(last=False)
                self.bytes_used -= evicted.sizeInBytes()
            self._images[key] = mapped
            self.bytes_used += size
        return QImage(mapped)

    def draw_image(self, painter: QPainter, target: QRectF, image: QImage,
                   value: int, visible: QRectF, *, exposure_tenths: int = 0) -> None:
        if (value == 0 and exposure_tenths == 0) or image.isNull():
            painter.drawImage(target, image)
            return
        # Four bytes per output pixel also bounds fallback format conversion.
        if image.width() * image.height() * 4 <= self.max_bytes:
            painter.drawImage(target, self.mapped_image(
                image, value, exposure_tenths=exposure_tenths,
            ))
            return
        clipped = target.intersected(visible)
        if clipped.isEmpty() or target.isEmpty():
            return
        sx, sy = image.width() / target.width(), image.height() / target.height()
        left = max(0, math.floor((clipped.left() - target.left()) * sx))
        top = max(0, math.floor((clipped.top() - target.top()) * sy))
        right = min(image.width(), math.ceil((clipped.right() - target.left()) * sx))
        bottom = min(image.height(), math.ceil((clipped.bottom() - target.top()) * sy))
        # Anchor tiles to source coordinates, so panning reuses cached pixels.
        for y in range(top // _TILE_EDGE * _TILE_EDGE, bottom, _TILE_EDGE):
            for x in range(left // _TILE_EDGE * _TILE_EDGE, right, _TILE_EDGE):
                core = QRect(x, y, min(_TILE_EDGE, image.width() - x),
                             min(_TILE_EDGE, image.height() - y))
                roi = core.adjusted(-1, -1, 1, 1).intersected(image.rect())
                def destination(rect: QRect) -> QRectF:
                    return QRectF(target.x() + rect.x() / sx, target.y() + rect.y() / sy,
                                  rect.width() / sx, rect.height() / sy)
                painter.save()
                # One source pixel of halo keeps smooth interpolation at tile
                # boundaries identical to the uncropped source.
                painter.setClipRect(destination(core), Qt.ClipOperation.IntersectClip)
                painter.drawImage(destination(roi), self.mapped_image(
                    image, value, roi, exposure_tenths=exposure_tenths,
                ))
                painter.restore()
