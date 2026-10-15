"""Lossless uint8 PNG stream with bounded scanline and compressor buffers.

EvidenceStore uses this encoder. Its workspace allowance is a
logical buffer budget, not a process RSS bound: Python/zlib allocators, OS file
caches, Qt and capture drivers have separate memory costs. File creation,
flush/fsync, atomic replacement and temporary-file cleanup belong to the caller.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import struct
from typing import BinaryIO, Callable
from zlib_ng import zlib_ng as zlib

import numpy as np


CODEC_ALLOWANCE_BYTES = 2 * 1024**2
ENCODER_REVISION = "png-block-fast-v2"
COMPRESS_INPUT_BYTES = 1024 * 1024
IDAT_CHUNK_BYTES = 2 * 1024 * 1024
SCANLINE_BLOCK_BYTES = 1024 * 1024
COMPRESSION_SAMPLE_BYTES = 64 * 1024
PNG_MAX_DIMENSION = 2**31 - 1


@dataclass(frozen=True, slots=True)
class PngStreamResult:
    sha256: str
    size_bytes: int
    workspace_bytes: int


def png_workspace_bytes(width: int, channels: int) -> int:
    """Two bounded scanline blocks plus codec allowance, independent of height.

    The allowance covers bounded compressor calls/output, checksum state and
    small Python buffers for the pinned zlib-ng build; it is deliberately
    not advertised as an allocator upper bound across arbitrary library builds.
    """
    if type(width) is not int or not 0 < width <= PNG_MAX_DIMENSION:
        raise ValueError("PNG 图像宽度必须是有效正整数。")
    if type(channels) is not int or channels not in (1, 3, 4):
        raise ValueError("PNG 图像必须是灰度、BGR 或 BGRA。")
    row_bytes = width * channels
    rows = max(1, SCANLINE_BLOCK_BYTES // (row_bytes + 1))
    return CODEC_ALLOWANCE_BYTES + rows * (2 * row_bytes + 1)


def _compression_level(pixels: np.ndarray) -> int:
    """Deterministic bounded probe: avoid costly expansion of noisy evidence.

    This changes only PNG storage, never pixels or measurement. Fast level 1
    retains compression on flat images. Moderate sensor noise already makes
    level 1 too slow for sustained 4K evidence, even when it saves some bytes;
    use stored DEFLATE unless the sample shrinks to less than one quarter.
    This trades disk space for bounded CPU work. Samples predict
    compressibility, not a latency guarantee.
    """
    sample = encoded = None
    try:
        height, width = pixels.shape[:2]
        channels = 1 if pixels.ndim == 2 else pixels.shape[2]
        columns = min(width, max(1, COMPRESSION_SAMPLE_BYTES // (channels * min(height, 8))))
        rows = min(height, max(1, COMPRESSION_SAMPLE_BYTES // (columns * channels)))
        ys = np.linspace(0, height - 1, rows, dtype=np.intp)
        xs = np.linspace(0, width - 1, columns, dtype=np.intp)
        sample = np.ascontiguousarray(pixels[ys[:, None], xs])
        encoded = zlib.compress(memoryview(sample).cast("B"), level=1)
        return 0 if len(encoded) >= sample.nbytes * .25 else 1
    finally:
        pixels = sample = encoded = None


class _ChunkWriter:
    def __init__(self, target: BinaryIO):
        self.target = target
        self.digest = hashlib.sha256()
        self.size_bytes = 0

    def write(self, data: bytes | memoryview) -> None:
        count = self.target.write(data)
        if type(count) is not int or count != len(data):
            raise OSError("原图写入不完整。")
        self.digest.update(data)
        self.size_bytes += count

    def chunk(self, kind: bytes, data: bytes | memoryview) -> None:
        self.write(struct.pack(">I", len(data)) + kind)
        self.write(data)
        checksum = zlib.crc32(data, zlib.crc32(kind)) & 0xffffffff
        self.write(struct.pack(">I", checksum))

    def idat(self, data: bytes) -> None:
        view = memoryview(data)
        for offset in range(0, len(view), IDAT_CHUNK_BYTES):
            self.chunk(b"IDAT", view[offset:offset + IDAT_CHUNK_BYTES])


def write_png_stream(pixels: np.ndarray, target: BinaryIO, *,
                     checkpoint: Callable[[], None] | None = None) -> PngStreamResult:
    """Write GRAY/BGR/BGRA pixels without a full image or encoded-file copy.

    Non-contiguous/read-only input is borrowed in small blocks and never modified.
    Alpha is preserved. A single standard zlib stream spans all unfiltered
    scanlines, split into PNG IDAT chunks. Exceptions propagate; no completed
    result is returned after a short write, I/O failure or cancellation.
    """
    if not isinstance(pixels, np.ndarray) or pixels.dtype != np.uint8:
        raise ValueError("PNG 输入必须是 uint8 数组。")
    if pixels.ndim == 2:
        channels = 1
    elif pixels.ndim == 3 and pixels.shape[2] in (3, 4):
        channels = int(pixels.shape[2])
    else:
        raise ValueError("PNG 输入必须是二维灰度或三维 BGR/BGRA 数组。")
    height, width = map(int, pixels.shape[:2])
    if not 0 < height <= PNG_MAX_DIMENSION:
        raise ValueError("PNG 图像高度必须是有效正整数。")
    workspace_bytes = png_workspace_bytes(width, channels)
    raw = filtered = colors = compressor = scanline = None
    try:
        if checkpoint is not None:
            checkpoint()
        # Batch color conversion and filtering to avoid thousands of Python /
        # NumPy calls competing with Qt for the GIL. Cancellation retains the
        # per-row checkpoints; the evidence writer has no cancellation callback.
        rows_per_block = (max(1, SCANLINE_BLOCK_BYTES // (width * channels + 1))
                          if checkpoint is None else 1)
        filtered = np.empty((rows_per_block, width * channels + 1), dtype=np.uint8)
        raw = filtered[:, 1:]
        colors = raw.reshape(rows_per_block, width, channels)
        filtered[:, 0] = 0
        compressor = zlib.compressobj(level=_compression_level(pixels), method=zlib.DEFLATED, wbits=15,
                                      memLevel=8, strategy=zlib.Z_DEFAULT_STRATEGY)
        writer = _ChunkWriter(target)
        writer.write(b"\x89PNG\r\n\x1a\n")
        color_type = {1: 0, 3: 2, 4: 6}[channels]
        writer.chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0))
        for y in range(0, height, rows_per_block):
            if checkpoint is not None:
                checkpoint()
            rows = min(rows_per_block, height - y)
            if channels == 1:
                np.copyto(raw[:rows], pixels[y:y + rows])
            else:
                colors[:rows, :, 0] = pixels[y:y + rows, :, 2]
                colors[:rows, :, 1] = pixels[y:y + rows, :, 1]
                colors[:rows, :, 2] = pixels[y:y + rows, :, 0]
                if channels == 4:
                    colors[:rows, :, 3] = pixels[y:y + rows, :, 3]
            scanline = memoryview(filtered[:rows]).cast("B")
            for offset in range(0, len(scanline), COMPRESS_INPUT_BYTES):
                writer.idat(compressor.compress(scanline[offset:offset + COMPRESS_INPUT_BYTES]))
        writer.idat(compressor.flush(zlib.Z_FINISH))
        writer.chunk(b"IEND", b"")
        return PngStreamResult(writer.digest.hexdigest(), writer.size_bytes, workspace_bytes)
    finally:
        # A retained exception traceback must not pin the input image or codec
        # through this frame after the caller has handled an encoding failure.
        pixels = raw = filtered = colors = compressor = scanline = None
