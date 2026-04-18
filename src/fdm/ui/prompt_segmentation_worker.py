from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QObject, Qt, Signal, Slot
from PySide6.QtGui import QImage

from fdm.geometry import Point
from fdm.services.prompt_segmentation import PromptSegmentationService, resolve_magic_segment_model_variant


@dataclass(slots=True)
class PromptSegmentationRequest:
    document_id: str
    image: QImage
    cache_key: str
    request_id: int
    positive_points: list[Point]
    negative_points: list[Point]
    model_variant: str
    auto_small_object_enabled: bool


class PromptSegmentationWorker(QObject):
    requested = Signal(object)
    clearRequested = Signal()
    succeeded = Signal(str, int, object)
    failed = Signal(str, int, str)

    def __init__(self) -> None:
        super().__init__()
        self._services: dict[str, PromptSegmentationService] = {}
        self.requested.connect(self.infer, Qt.ConnectionType.QueuedConnection)
        self.clearRequested.connect(self.clear_cache, Qt.ConnectionType.QueuedConnection)

    @Slot(object)
    def infer(self, request: PromptSegmentationRequest) -> None:
        try:
            resolved_variant, fallback_message = resolve_magic_segment_model_variant(request.model_variant)
            service = self._services.get(resolved_variant)
            if service is None:
                service = PromptSegmentationService(model_variant=resolved_variant)
                self._services[resolved_variant] = service
            result = service.predict_polygon(
                image=request.image,
                cache_key=request.cache_key,
                positive_points=list(request.positive_points),
                negative_points=list(request.negative_points),
                auto_small_object_enabled=bool(request.auto_small_object_enabled),
            )
            result.metadata["requested_model_variant"] = request.model_variant
            result.metadata["resolved_model_variant"] = resolved_variant
            if fallback_message:
                result.metadata["model_fallback_message"] = fallback_message
            self.succeeded.emit(request.document_id, request.request_id, result)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(request.document_id, request.request_id, str(exc))

    @Slot()
    def clear_cache(self) -> None:
        for service in self._services.values():
            service.clear_cache()
