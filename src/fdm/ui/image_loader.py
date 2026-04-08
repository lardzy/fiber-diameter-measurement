from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import QObject, Signal, Slot
from PySide6.QtGui import QImage, QImageReader

from fdm.raster import RasterImage


@dataclass(slots=True)
class ImageLoadRequest:
    path: str
    document: object | None = None


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
    finished = Signal(bool, int, int, int)

    def __init__(self, requests: list[ImageLoadRequest], *, skipped_count: int = 0) -> None:
        super().__init__()
        self._requests = list(requests)
        self._skipped_count = skipped_count
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    @Slot()
    def run(self) -> None:
        total = len(self._requests)
        loaded_count = 0
        failed_count = 0
        for index, request in enumerate(self._requests, start=1):
            if self._cancelled:
                break
            self.progress.emit(index, total, request.path)
            reader = QImageReader(request.path)
            reader.setAutoTransform(True)
            image = reader.read()
            if image.isNull():
                reason = reader.errorString() or "无法读取图片"
                failed_count += 1
                self.failed.emit(request.path, reason)
                continue
            loaded_count += 1
            self.loaded.emit(request, image)
        self.finished.emit(self._cancelled, loaded_count, self._skipped_count, failed_count)
