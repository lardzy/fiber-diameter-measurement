from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QObject, Qt, Signal, Slot

from fdm.geometry import Point
from fdm.services.prompt_segmentation import PromptSegmentationService


@dataclass(slots=True)
class PromptSegmentationRequest:
    document_id: str
    image_path: str
    request_id: int
    positive_points: list[Point]
    negative_points: list[Point]


class PromptSegmentationWorker(QObject):
    requested = Signal(object)
    succeeded = Signal(str, int, object)
    failed = Signal(str, int, str)

    def __init__(self) -> None:
        super().__init__()
        self._service: PromptSegmentationService | None = None
        self.requested.connect(self.infer, Qt.ConnectionType.QueuedConnection)

    @Slot(object)
    def infer(self, request: PromptSegmentationRequest) -> None:
        try:
            if self._service is None:
                self._service = PromptSegmentationService()
            result = self._service.predict_polygon(
                image_path=request.image_path,
                positive_points=list(request.positive_points),
                negative_points=list(request.negative_points),
            )
            self.succeeded.emit(request.document_id, request.request_id, result.polygon_px)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(request.document_id, request.request_id, str(exc))
