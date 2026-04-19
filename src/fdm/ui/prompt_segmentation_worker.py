from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QObject, Qt, Signal, Slot
from PySide6.QtGui import QImage

from fdm.geometry import Point
from fdm.services.fiber_quick_geometry import FiberQuickDiameterGeometryService
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


class PromptSegmentationWorker(QObject):
    requested = Signal(object)
    clearRequested = Signal()
    succeeded = Signal(str, int, object)
    failed = Signal(str, int, str)

    def __init__(self) -> None:
        super().__init__()
        self._services: dict[str, PromptSegmentationService] = {}
        self._fiber_quick_geometry = FiberQuickDiameterGeometryService()
        self.requested.connect(self.infer, Qt.ConnectionType.QueuedConnection)
        self.clearRequested.connect(self.clear_cache, Qt.ConnectionType.QueuedConnection)

    @Slot(object)
    def infer(self, request: PromptSegmentationRequest) -> None:
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
            )
            result.metadata["tool_mode"] = request.tool_mode
            result.metadata["active_stage"] = request.active_stage
            result.metadata["requested_model_variant"] = request.model_variant
            result.metadata["resolved_model_variant"] = resolved_variant
            if fallback_message:
                result.metadata["model_fallback_message"] = fallback_message
            if request.tool_mode == "fiber_quick":
                geometry_result = self._fiber_quick_geometry.measure_from_mask(
                    result.mask,
                    positive_points=list(request.positive_points),
                    negative_points=list(request.negative_points),
                )
                result.metadata["fiber_quick_line_px"] = geometry_result.line_px
                result.metadata["fiber_quick_confidence"] = geometry_result.confidence
                result.metadata["fiber_quick_status"] = geometry_result.status
                result.metadata["fiber_quick_debug_payload"] = geometry_result.debug_payload
            self.succeeded.emit(request.document_id, request.request_id, result)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(request.document_id, request.request_id, str(exc))

    @Slot()
    def clear_cache(self) -> None:
        for service in self._services.values():
            service.clear_cache()
