"""Microview pixel decoding shared by the existing capture helper and tests.

SDK metadata uses fixed-width Windows types. SkipPixel is a row prefix, not
extra padding. This module never opens a second camera/device session.
"""
from __future__ import annotations


import numpy as np

MAX_FRAME_BYTES = 128 * 1024 * 1024


def _microview_buffer_view(buffer, info):
    from fdm.services.capture import _microview_normalize_pixel_format, _microview_bytes_per_pixel
    width, height, length = int(info.Width), int(info.Heigth), int(info.Length)
    pixel_format = _microview_normalize_pixel_format(int(info.nColor))
    bpp = _microview_bytes_per_pixel(pixel_format)
    if not bpp or width <= 0 or height <= 0 or length <= 0 or length > MAX_FRAME_BYTES:
        raise ValueError("Microview 帧尺寸或像素格式无效。")
    if length > len(buffer) or width * height * bpp > length:
        raise ValueError("Microview 帧缓冲长度与元数据不一致。")
    skip = int(info.SkipPixel)
    if not 0 <= skip < width:
        raise ValueError("Microview SkipPixel 必须是有效的行首像素数。")
    stride = width * bpp
    valid_width = width - skip
    data = np.frombuffer(buffer, np.uint8, count=length)
    rows = np.ndarray((height, valid_width * bpp), dtype=np.uint8,
                      buffer=data, offset=skip * bpp, strides=(stride, 1))
    rows.setflags(write=False)
    width = valid_width
    return rows, pixel_format, width, height, bpp


def copy_microview_to_shared(buffer, info, destination) -> tuple[int, ...]:
    """Copy raw SDK pixels directly into the shared slot with row-sized scratch.

    No full decoded frame exists alongside the SDK allocation and shared slot.
    Row-prefix semantics follow the SDK header, manual and MV_ReadLine.
    """
    rows, pixel_format, width, height, bpp = _microview_buffer_view(buffer, info)
    shape = (height, width) if pixel_format == 0 else (height, width, bpp if pixel_format in (2, 3) else 3)
    image = np.ndarray(shape, np.uint8, buffer=destination)
    if pixel_format in (0, 2, 3):
        np.copyto(image, rows.reshape(shape))
        if pixel_format == 3:
            image[..., 3] = 255
        return shape
    green_bits = 6 if pixel_format == 5 else 5
    for y in range(height):
        words = np.ascontiguousarray(rows[y]).view("<u2").astype(np.uint32)
        blue, green, red = words & 31, (words >> 5) & ((1 << green_bits) - 1), (words >> (5 + green_bits)) & 31
        image[y, :, 0] = (blue << 3) | (blue >> 2)
        image[y, :, 1] = (green << (8 - green_bits)) | (green >> (2 * green_bits - 8))
        image[y, :, 2] = (red << 3) | (red >> 2)
    return shape


def decode_microview_buffer(buffer, info) -> np.ndarray:
    """Owned conversion utility; streaming writes directly to its shared slot."""
    rows, pixel_format, width, height, bpp = _microview_buffer_view(buffer, info)
    if pixel_format == 0:
        return np.array(rows, copy=True, order="C")
    if pixel_format in (2, 3):
        image = np.array(rows.reshape(height, width, bpp), copy=True, order="C")
        if bpp == 4:
            image[..., 3] = 255
        return image
    # Existing SDK declarations call these RGB565/RGB555; expand the little-
    # endian words directly so Qt GUI types are unnecessary in the transport.
    words = np.array(rows, copy=True).view("<u2").reshape(height, width).astype(np.uint32)
    green_bits = 6 if pixel_format == 5 else 5
    blue = words & 31
    green = (words >> 5) & ((1 << green_bits) - 1)
    red = (words >> (5 + green_bits)) & 31
    image = np.empty((height, width, 3), np.uint8)
    image[..., 0] = (blue << 3) | (blue >> 2)
    image[..., 1] = (green << (8 - green_bits)) | (green >> (2 * green_bits - 8))
    image[..., 2] = (red << 3) | (red >> 2)
    return image
