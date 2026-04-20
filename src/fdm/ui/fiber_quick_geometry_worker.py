from __future__ import annotations

from dataclasses import dataclass
from threading import Lock

from PySide6.QtCore import QObject, Qt, Signal, Slot

from fdm.geometry import Point
from fdm.services.fiber_quick_geometry import FiberQuickDiameterGeometryService


@dataclass(slots=True)
class FiberQuickGeometryRequest:
    document_id: str
    request_id: int
    mask: object | None
    preview_polygon_px: list[Point]
    preview_area_rings_px: list[list[Point]]
    positive_points: list[Point]
    negative_points: list[Point]


class FiberQuickGeometryWorker(QObject):
    requested = Signal(object)
    succeeded = Signal(str, int, object)
    failed = Signal(str, int, str)

    def __init__(self) -> None:
        super().__init__()
        self._service = FiberQuickDiameterGeometryService()
        self._latest_request_ids: dict[str, int] = {}
        self._cancelled_documents: set[str] = set()
        self._lock = Lock()
        self.requested.connect(self.measure, Qt.ConnectionType.QueuedConnection)

    def register_request(self, document_id: str, request_id: int) -> None:
        with self._lock:
            self._latest_request_ids[document_id] = max(int(request_id), int(self._latest_request_ids.get(document_id, 0)))
            self._cancelled_documents.discard(document_id)

    def cancel_document(self, document_id: str) -> None:
        with self._lock:
            self._cancelled_documents.add(document_id)

    def _is_request_stale(self, document_id: str, request_id: int) -> bool:
        with self._lock:
            if document_id in self._cancelled_documents:
                return True
            return int(request_id) < int(self._latest_request_ids.get(document_id, request_id))

    @Slot(object)
    def measure(self, request: FiberQuickGeometryRequest) -> None:
        if self._is_request_stale(request.document_id, request.request_id):
            return
        try:
            result = self._service.measure_from_mask(
                request.mask,
                positive_points=list(request.positive_points),
                negative_points=list(request.negative_points),
                preview_polygon_points=list(request.preview_polygon_px),
                preview_area_rings_points=[list(ring) for ring in request.preview_area_rings_px],
                cancel_check=lambda: self._is_request_stale(request.document_id, request.request_id),
            )
            if self._is_request_stale(request.document_id, request.request_id):
                return
            self.succeeded.emit(request.document_id, request.request_id, result)
        except Exception as exc:  # noqa: BLE001
            if self._is_request_stale(request.document_id, request.request_id):
                return
            self.failed.emit(request.document_id, request.request_id, str(exc))
