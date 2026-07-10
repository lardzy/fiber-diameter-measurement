from __future__ import annotations

from dataclasses import dataclass
from threading import Lock

from PySide6.QtCore import QObject, Qt, Signal, Slot
from PySide6.QtGui import QImage

from fdm.geometry import Point
from fdm.services.prompt_segmentation import resolve_magic_segment_model_variant
from fdm.services.reference_instance_propagation import ReferenceInstancePropagationService


@dataclass(slots=True)
class ReferenceInstancePropagationRequest:
    document_id: str
    image: QImage
    cache_key: str
    request_id: int
    model_variant: str
    reference_box: tuple[Point, Point] | None = None
    reference_polygon_px: list[Point] | None = None
    reference_area_rings_px: list[list[Point]] | None = None


class ReferenceInstancePropagationWorker(QObject):
    requested = Signal(object)
    clearRequested = Signal()
    succeeded = Signal(str, int, object)
    failed = Signal(str, int, str)

    def __init__(self) -> None:
        super().__init__()
        self._services: dict[str, ReferenceInstancePropagationService] = {}
        self._latest_request_ids: dict[str, int] = {}
        self._cancelled_documents: set[str] = set()
        self._lock = Lock()
        self.requested.connect(self.infer, Qt.ConnectionType.QueuedConnection)
        self.clearRequested.connect(self.clear_cache, Qt.ConnectionType.QueuedConnection)

    def register_request(self, document_id: str, request_id: int) -> None:
        with self._lock:
            self._latest_request_ids[document_id] = max(
                int(request_id),
                int(self._latest_request_ids.get(document_id, 0)),
            )
            self._cancelled_documents.discard(document_id)

    def cancel_document(self, document_id: str) -> None:
        with self._lock:
            self._cancelled_documents.add(document_id)

    def _is_request_stale(self, document_id: str, request_id: int) -> bool:
        with self._lock:
            return (
                document_id in self._cancelled_documents
                or int(request_id) < int(self._latest_request_ids.get(document_id, request_id))
            )

    @Slot(object)
    def infer(self, request: ReferenceInstancePropagationRequest) -> None:
        if self._is_request_stale(request.document_id, request.request_id):
            return
        try:
            resolved_variant, fallback_message = resolve_magic_segment_model_variant(request.model_variant)
            service = self._services.get(resolved_variant)
            if service is None:
                service = ReferenceInstancePropagationService(model_variant=resolved_variant)
                self._services[resolved_variant] = service
            result = service.propagate_from_reference(
                image=request.image,
                cache_key=request.cache_key,
                reference_box=request.reference_box,
                reference_polygon_px=list(request.reference_polygon_px or []),
                reference_area_rings_px=[list(ring) for ring in (request.reference_area_rings_px or [])],
                cancel_check=lambda: self._is_request_stale(request.document_id, request.request_id),
            )
            if self._is_request_stale(request.document_id, request.request_id):
                return
            result.metadata["requested_model_variant"] = request.model_variant
            result.metadata["resolved_model_variant"] = resolved_variant
            if fallback_message:
                result.metadata["model_fallback_message"] = fallback_message
            self.succeeded.emit(request.document_id, request.request_id, result)
        except Exception as exc:  # noqa: BLE001
            if self._is_request_stale(request.document_id, request.request_id):
                return
            self.failed.emit(request.document_id, request.request_id, str(exc))

    @Slot()
    def clear_cache(self) -> None:
        for service in self._services.values():
            service.clear_cache()
