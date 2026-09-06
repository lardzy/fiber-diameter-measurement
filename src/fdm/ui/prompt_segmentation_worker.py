from __future__ import annotations

from dataclasses import dataclass
from threading import Lock

from PySide6.QtCore import QObject, Qt, Signal, Slot
from PySide6.QtGui import QImage

from fdm.geometry import Point
from fdm.services.mask_region import MaskRegion, mask_region
from fdm.settings import is_magic_segment_tool_mode
from fdm.services.prompt_segmentation import (
    PromptSegmentationService,
    create_interactive_segmentation_service,
    magic_mask_area_px,
    magic_mask_to_geometry,
    resolve_interactive_segmentation_backend,
    fill_magic_draft_internal_holes,
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
    roi_constraint_box: tuple[int, int, int, int] | None = None
    small_object_enhancement_enabled: bool = False
    small_object_roi_area_threshold_px: int = 160000
    small_object_workspace_box: tuple[int, int, int, int] | None = None
    source_token: str = ""
    valid_coverage: object | None = None
    fill_draft_holes: bool = False


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
            compact = is_magic_segment_tool_mode(request.tool_mode)
            service.local_masks = compact
            result = service.predict_polygon(
                image=request.image,
                cache_key=request.cache_key,
                positive_points=list(request.positive_points),
                negative_points=list(request.negative_points),
                tool_mode=request.tool_mode,
                active_stage=request.active_stage,
                roi_enabled=bool(request.roi_enabled),
                roi_constraint_box=request.roi_constraint_box,
                small_object_enhancement_enabled=bool(request.small_object_enhancement_enabled),
                small_object_roi_area_threshold_px=int(request.small_object_roi_area_threshold_px),
                small_object_workspace_box=request.small_object_workspace_box,
                cancel_check=lambda: self._is_request_cancelled(request.document_id),
            )
            if result.mask is not None and request.valid_coverage is not None:
                import numpy as np

                region = result.mask if isinstance(result.mask, MaskRegion) else None
                mask = region.data if region is not None else np.asarray(result.mask, dtype=bool)
                coverage = np.asarray(request.valid_coverage, dtype=bool)
                if region is not None:
                    if coverage.shape != region.extent:
                        raise RuntimeError("分割结果与有效图块覆盖尺寸不一致")
                    x, y = region.origin
                    coverage = coverage[y : y + mask.shape[0], x : x + mask.shape[1]]
                if mask.shape != coverage.shape:
                    raise RuntimeError(
                        f"分割结果与有效图块覆盖尺寸不一致：{mask.shape} != {coverage.shape}。"
                    )
                clipped = np.ascontiguousarray(mask & coverage)
                if region is not None:
                    clipped = mask_region(clipped, origin=region.origin, extent=region.extent)
                selected_mask, rings, polygon, stats = magic_mask_to_geometry(
                    clipped,
                    positive_points=list(request.positive_points),
                    negative_points=list(request.negative_points),
                )
                result.mask = selected_mask
                result.area_rings_px = rings
                result.polygon_px = polygon
                result.area_px = (
                    magic_mask_area_px(selected_mask)
                    if selected_mask is not None
                    else 0.0
                )
                result.metadata["coverage_clipped"] = bool(
                    np.any(mask & ~coverage)
                )
                result.metadata.update(stats)
            if compact and result.mask is not None:
                result.mask = mask_region(result.mask)
                if request.fill_draft_holes:
                    result.mask, result.area_rings_px, result.polygon_px, stats = (
                        magic_mask_to_geometry(fill_magic_draft_internal_holes(result.mask))
                    )
                    result.metadata.update(stats)
                result.area_px = magic_mask_area_px(result.mask)
                result.metadata["holes_processed"] = bool(request.fill_draft_holes)
                result.metadata["geometry_final"] = True
            if self._is_request_cancelled(request.document_id):
                return
            result.metadata["tool_mode"] = request.tool_mode
            result.metadata["active_stage"] = request.active_stage
            result.metadata["requested_model_variant"] = request.model_variant
            result.metadata["resolved_model_variant"] = resolved_variant
            result.metadata["positive_points_px"] = list(request.positive_points)
            result.metadata["negative_points_px"] = list(request.negative_points)
            result.metadata["source_token"] = request.source_token
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
