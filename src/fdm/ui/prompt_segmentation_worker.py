from __future__ import annotations

from dataclasses import dataclass
from threading import Lock

from PySide6.QtCore import QObject, Qt, Signal, Slot
from PySide6.QtGui import QImage

from fdm.geometry import Point
from fdm.services.prompt_segmentation import (
    PromptSegmentationService,
    create_interactive_segmentation_service,
    resolve_interactive_segmentation_backend,
)


@dataclass(slots=True)
class PromptSegmentationRequest:
    document_id: str
    image: QImage
    cache_key: str
    request_id: int
    positive_points: list[Point]
    negative_points: list[Point]
    tool_mode: str
    active_stage: str
    model_variant: str
    roi_enabled: bool


class PromptSegmentationWorker(QObject):
    requested = Signal(object)
    clearRequested = Signal()
    succeeded = Signal(str, int, object)
    failed = Signal(str, int, str)

    def __init__(self) -> None:
        super().__init__()
        self._services: dict[str, PromptSegmentationService] = {}
        self._cancelled_documents: set[str] = set()
        self._lock = Lock()
        self.requested.connect(self.infer, Qt.ConnectionType.QueuedConnection)
        self.clearRequested.connect(self.clear_cache, Qt.ConnectionType.QueuedConnection)

    def register_request(self, document_id: str, request_id: int) -> None:
        with self._lock:
            self._cancelled_documents.discard(document_id)

    def cancel_document(self, document_id: str) -> None:
        with self._lock:
            self._cancelled_documents.add(document_id)

    def _is_request_cancelled(self, document_id: str) -> bool:
        with self._lock:
            return document_id in self._cancelled_documents

    @Slot(object)
    def infer(self, request: PromptSegmentationRequest) -> None:
        if self._is_request_cancelled(request.document_id):
            return
        try:
            resolved_variant, fallback_message = resolve_interactive_segmentation_backend(request.model_variant)
            service = self._services.get(resolved_variant)
            if service is None:
                service = create_interactive_segmentation_service(resolved_variant)
                self._services[resolved_variant] = service
            result = service.predict_polygon(
                image=request.image,
                cache_key=request.cache_key,
                positive_points=list(request.positive_points),
                negative_points=list(request.negative_points),
                tool_mode=request.tool_mode,
                roi_enabled=bool(request.roi_enabled),
                cancel_check=lambda: self._is_request_cancelled(request.document_id),
            )
            if self._is_request_cancelled(request.document_id):
                return
            result.metadata["tool_mode"] = request.tool_mode
            result.metadata["active_stage"] = request.active_stage
            result.metadata["requested_model_variant"] = request.model_variant
            result.metadata["resolved_model_variant"] = resolved_variant
            result.metadata["positive_points_px"] = list(request.positive_points)
            result.metadata["negative_points_px"] = list(request.negative_points)
            if fallback_message:
                result.metadata["model_fallback_message"] = fallback_message
            self.succeeded.emit(request.document_id, request.request_id, result)
        except Exception as exc:  # noqa: BLE001
            if self._is_request_cancelled(request.document_id):
                return
            self.failed.emit(request.document_id, request.request_id, str(exc))

    @Slot()
    def clear_cache(self) -> None:
        for service in self._services.values():
            service.clear_cache()
