from __future__ import annotations

from dataclasses import dataclass, field
from uuid import uuid4

import numpy as np
from PySide6.QtCore import QObject, Signal, Slot
from PySide6.QtGui import QImage

from fdm.cancellation import CancellationSource, CancellationToken
from fdm.raster import RasterImage, RasterPlane
from fdm.services.raster_io import (
    RasterMetadata,
    raster_plane_to_qimage,
    read_raster_file,
)


def raster_document_contract_error(
    document: object | None,
    plane: RasterPlane,
) -> str:
    """Return a mismatch message without mutating saved project geometry."""

    if document is None:
        return ""
    expected_pixel_type = getattr(document, "raster_pixel_type", None)
    if (
        expected_pixel_type is not None
        and expected_pixel_type is not plane.pixel_type
    ):
        return (
            "项目记录的像素类型与资产实际类型不一致："
            f"期望 {expected_pixel_type.value}，实际 "
            f"{plane.pixel_type.value}"
        )
    expected_size = getattr(document, "image_size", None)
    if expected_size is not None:
        try:
            normalized_size = (int(expected_size[0]), int(expected_size[1]))
        except (IndexError, TypeError, ValueError, OverflowError):
            return "项目记录的图片尺寸无效。"
        actual_size = (plane.width, plane.height)
        if normalized_size != actual_size:
            return (
                "项目记录的图片尺寸与资产实际尺寸不一致："
                f"期望 {normalized_size[0]}×{normalized_size[1]}，实际 "
                f"{actual_size[0]}×{actual_size[1]}"
            )
    return ""


@dataclass(slots=True)
class ImageLoadRequest:
    path: str
    document: object | None = None
    request_id: str = field(default_factory=lambda: uuid4().hex)
    generation: int = 0
    raster_plane: RasterPlane | None = None
    raster_metadata: RasterMetadata | None = None


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
        try:
            for index, request in enumerate(self._requests, start=1):
                if self._cancellation.token.is_cancelled:
                    break
                self.progress.emit(index, total, request.path)
                if self._cancellation.token.is_cancelled:
                    break
                try:
                    loaded = read_raster_file(request.path)
                    if self._cancellation.token.is_cancelled:
                        break
                    if (
                        not loaded.success
                        or loaded.plane is None
                        or loaded.metadata is None
                    ):
                        failure = loaded.failure
                        reason = (
                            f"{failure.message}{': ' + failure.detail if failure and failure.detail else ''}"
                            if failure is not None
                            else "无法读取图片"
                        )
                        failed_count += 1
                        self.failed.emit(request.path, reason)
                        self.failedRequest.emit(request, reason)
                        continue
                    contract_error = raster_document_contract_error(
                        request.document,
                        loaded.plane,
                    )
                    if contract_error:
                        failed_count += 1
                        self.failed.emit(request.path, contract_error)
                        self.failedRequest.emit(request, contract_error)
                        continue
                    display_transform = getattr(
                        request.document,
                        "display_transform",
                        None,
                    )
                    image = (
                        raster_plane_to_qimage(loaded.plane)
                        if display_transform is None
                        else raster_plane_to_qimage(
                            loaded.plane,
                            display_transform=display_transform,
                        )
                    )
                    if image.isNull():
                        raise ValueError(
                            "图片像素已解码，但无法创建画布显示缓存"
                        )
                    if self._cancellation.token.is_cancelled:
                        break
                    request.raster_plane = loaded.plane
                    request.raster_metadata = loaded.metadata
                    if self._cancellation.token.is_cancelled:
                        break
                    self.loaded.emit(request, image)
                    loaded_count += 1
                except Exception as exc:  # noqa: BLE001 - worker reports per-file failures
                    if self._cancellation.token.is_cancelled:
                        break
                    reason = f"创建图片显示缓存失败：{exc}"
                    failed_count += 1
                    self.failed.emit(request.path, reason)
                    self.failedRequest.emit(request, reason)
        finally:
            self.finished.emit(
                self._cancellation.token.is_cancelled,
                loaded_count,
                self._skipped_count,
                failed_count,
            )
