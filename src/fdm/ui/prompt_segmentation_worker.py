from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QObject, Qt, Signal, Slot
from PySide6.QtGui import QImage

from fdm.geometry import Point
from fdm.services.prompt_segmentation import PromptSegmentationService


@dataclass(slots=True)
class PromptSegmentationRequest:
    document_id: str
    image: QImage
    cache_key: str
    request_id: int
    positive_points: list[Point]
    negative_points: list[Point]


class PromptSegmentationWorker(QObject):
    requested = Signal(object)
    clearRequested = Signal()
    succeeded = Signal(str, int, object)
    failed = Signal(str, int, str)

    def __init__(self) -> None:
        super().__init__()
        self._service: PromptSegmentationService | None = None
        self.requested.connect(self.infer, Qt.ConnectionType.QueuedConnection)
        self.clearRequested.connect(self.clear_cache, Qt.ConnectionType.QueuedConnection)

    @Slot(object)
    def infer(self, request: PromptSegmentationRequest) -> None:
        try:
            if self._service is None:
                self._service = PromptSegmentationService()
            result = self._service.predict_polygon(
                image=request.image,
                cache_key=request.cache_key,
                positive_points=list(request.positive_points),
                negative_points=list(request.negative_points),
            )
            self.succeeded.emit(request.document_id, request.request_id, result.polygon_px)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(request.document_id, request.request_id, str(exc))

    @Slot()
    def clear_cache(self) -> None:
        if self._service is not None:
            self._service.clear_cache()
