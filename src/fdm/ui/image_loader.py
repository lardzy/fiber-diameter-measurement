from __future__ import annotations

from dataclasses import dataclass, field
from uuid import uuid4

import numpy as np
from PySide6.QtCore import QObject, Signal, Slot
from PySide6.QtGui import QImage, QImageReader

from fdm.cancellation import CancellationSource, CancellationToken
from fdm.raster import RasterImage


@dataclass(slots=True)
class ImageLoadRequest:
    path: str
    document: object | None = None
    request_id: str = field(default_factory=lambda: uuid4().hex)
    generation: int = 0


def qimage_to_raster(image: QImage) -> RasterImage:
    grayscale = image.convertToFormat(QImage.Format.Format_Grayscale8)
    width = grayscale.width()
    height = grayscale.height()
    ptr = grayscale.constBits()
    bpl = grayscale.bytesPerLine()
    arr = np.frombuffer(ptr, dtype=np.uint8, count=height * bpl).reshape(height, bpl)
    pixels = arr[:, :width].astype(int).ravel().tolist()
    return RasterImage(width=width, height=height, pixels=pixels)


class ImageBatchLoaderWorker(QObject):
    progress = Signal(int, int, str)
    loaded = Signal(object, object)
    failed = Signal(str, str)
    failedRequest = Signal(object, str)
    finished = Signal(bool, int, int, int)

    def __init__(self, requests: list[ImageLoadRequest], *, skipped_count: int = 0) -> None:
        super().__init__()
        self._requests = list(requests)
        self._skipped_count = skipped_count
        self._cancellation = CancellationSource()

    @property
    def cancellation_token(self) -> CancellationToken:
        return self._cancellation.token

    def cancel(self) -> None:
        self._cancellation.cancel()

    def request_cancel(self) -> None:
        """Thread-safe entry point; it must not wait for the worker event loop."""

        self._cancellation.cancel()

    @Slot()
    def run(self) -> None:
        total = len(self._requests)
        loaded_count = 0
        failed_count = 0
        for index, request in enumerate(self._requests, start=1):
            if self._cancellation.token.is_cancelled:
                break
            self.progress.emit(index, total, request.path)
            if self._cancellation.token.is_cancelled:
                break
            reader = QImageReader(request.path)
            reader.setAutoTransform(True)
            image = reader.read()
            if self._cancellation.token.is_cancelled:
                break
            if image.isNull():
                reason = reader.errorString() or "无法读取图片"
                failed_count += 1
                self.failed.emit(request.path, reason)
                self.failedRequest.emit(request, reason)
                continue
            loaded_count += 1
            self.loaded.emit(request, image)
        self.finished.emit(
            self._cancellation.token.is_cancelled,
            loaded_count,
            self._skipped_count,
            failed_count,
        )
