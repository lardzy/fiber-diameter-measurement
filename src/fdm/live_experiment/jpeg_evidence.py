"""Streaming JPEG sink with a declared, conservative transient buffer budget."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

from PIL import Image
import numpy as np


ENCODER_REVISION = "jpeg-evidence-444-v1"


def jpeg_workspace_bytes(width: int, height: int) -> int:
    # Contiguous source copy, Pillow's RGB owner and codec/output chunks.
    # This is the same logical buffer contract as PNG, not an RSS guarantee.
    return width * height * 12 + 2 * 1024**2


@dataclass(frozen=True, slots=True)
class JpegStreamResult:
    sha256: str
    size_bytes: int


class _HashingSink:
    """No fileno: Pillow must write through the hashed Python sink."""
    def __init__(self, target):
        self.target = target
        self.digest = hashlib.sha256()
        self.size_bytes = 0

    def write(self, chunk):
        written = self.target.write(chunk)
        if written != len(chunk):
            raise OSError("测量证据写入不完整。")
        self.digest.update(chunk)
        self.size_bytes += written
        return written

    def flush(self):
        self.target.flush()


def write_jpeg_evidence(pixels: np.ndarray, target, *, quality: int = 95) -> JpegStreamResult:
    if pixels.ndim == 2:
        mode, raw = "L", "L"
    else:
        mode, raw = "RGB", "BGR"
        pixels = pixels[:, :, :3]
    image = Image.frombytes(mode, (pixels.shape[1], pixels.shape[0]), pixels.tobytes(), "raw", raw)
    sink = _HashingSink(target)
    try:
        image.save(sink, format="JPEG", quality=quality, subsampling=0,
                   optimize=False, progressive=False)
    finally:
        image.close()
    return JpegStreamResult(sink.digest.hexdigest(), sink.size_bytes)
