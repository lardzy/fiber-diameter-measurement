from __future__ import annotations

from dataclasses import dataclass
import json
from threading import Lock
from time import perf_counter

from PySide6.QtCore import QObject, Qt, Signal, Slot
from PySide6.QtGui import QImage

from fdm.geometry import Point
from fdm.runtime_logging import append_runtime_log
from fdm.services.segmentation_performance import log_slow_segmentation
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
    roi_workspace_box: tuple[int, int, int, int] | None = None
    roi_workspace_context: tuple[object, ...] | None = None
    session_token: str = ""
    generation: int = 0
    source_prepare_ms: float = 0.0
    submitted_at: float = 0.0


class PromptSegmentationWorker(QObject):
    requested = Signal(object)
    clearRequested = Signal()
    warmupRequested = Signal(str)
    succeeded = Signal(str, int, object)
    failed = Signal(str, int, str)

    def __init__(self) -> None:
        super().__init__()
        self._services: dict[str, PromptSegmentationService] = {}
        self._cancelled_documents: set[str] = set()
        self._lock = Lock()
        self._generation = 0
        self._registered_requests: dict[str, tuple[int, int]] = {}
        self.requested.connect(self.infer, Qt.ConnectionType.QueuedConnection)
        self.clearRequested.connect(self.clear_cache, Qt.ConnectionType.QueuedConnection)
        self.warmupRequested.connect(self.warmup, Qt.ConnectionType.QueuedConnection)

    def register_request(self, document_id: str, request_id: int) -> int:
        with self._lock:
            self._generation += 1
            self._registered_requests[document_id] = (request_id, self._generation)
            self._cancelled_documents.discard(document_id)
            return self._generation

    def cancel_document(self, document_id: str) -> None:
        with self._lock:
            self._cancelled_documents.add(document_id)
            self._registered_requests.pop(document_id, None)

    def _is_request_cancelled(self, document_id: str, request_id: int | None = None, generation: int = 0) -> bool:
        with self._lock:
            registered = self._registered_requests.get(document_id)
            return document_id in self._cancelled_documents or (
                request_id is not None and registered is not None
                and (registered[0] != request_id or (generation > 0 and registered[1] != generation))
            )

    def _service(self, variant: str):
        service = self._services.get(variant)
        if service is None:
            service = create_interactive_segmentation_service(variant)
            self._services[variant] = service
        return service

    @staticmethod
    def _request_performance(request, started: float, performance: dict[str, object]) -> dict[str, object]:
        result = dict(performance)
        stages = dict(result.get("stages_ms", {}))
        stages["source_prepare_ms"] = request.source_prepare_ms
        stages["queue_ms"] = max(0.0, (started - request.submitted_at) * 1000.0) if request.submitted_at else 0.0
        result["stages_ms"] = stages
        result["worker_ms"] = (perf_counter() - started) * 1000.0
        result["total_ms"] = request.source_prepare_ms + stages["queue_ms"] + result["worker_ms"]
        return result

    @Slot(str)
    def warmup(self, requested_variant: str) -> None:
        try:
            variant, _message = resolve_interactive_segmentation_backend(requested_variant)
            service = self._service(variant)
            if isinstance(service, PromptSegmentationService):
                cold = service._encoder_session is None or service._decoder_session is None
                started = perf_counter()
                service.warmup()
                if cold:
                    append_runtime_log("Magic segmentation warmup", json.dumps({
                        "model": variant, "session_init_ms": (perf_counter() - started) * 1000.0,
                        "encoder_calls": 0,
                    }, allow_nan=False))
        except Exception as exc:  # noqa: BLE001 - retry with the normal UI error path on use
            append_runtime_log("Magic segmentation warmup failed", str(exc))

    @Slot(object)
    def infer(self, request: PromptSegmentationRequest) -> None:
        started = perf_counter()
        cancelled = lambda: self._is_request_cancelled(request.document_id, request.request_id, request.generation)
        if cancelled():
            performance = self._request_performance(request, started, {
                "encoder_calls": 0, "decoder_calls": 0, "cache_hits": 0, "cache_misses": 0,
                "roi_crops": [], "expansion_count": 0, "stop_reason": "cancelled_before_start",
            })
            log_slow_segmentation(performance, document_id=request.document_id, request_id=request.request_id,
                                  source_token=request.source_token, model=request.model_variant,
                                  session_token=request.session_token, tool_mode=request.tool_mode,
                                  active_stage=request.active_stage, status="cancelled")
            return
        service = None
        result = None
        reported = False
        try:
            resolved_variant, fallback_message = resolve_interactive_segmentation_backend(request.model_variant)
            service = self._service(resolved_variant)
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
                roi_workspace_box=request.roi_workspace_box,
                small_object_enhancement_enabled=bool(request.small_object_enhancement_enabled),
                small_object_roi_area_threshold_px=int(request.small_object_roi_area_threshold_px),
                small_object_workspace_box=request.small_object_workspace_box,
                cancel_check=cancelled,
            )
            geometry_started = perf_counter()
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
            if cancelled():
                return
            result.metadata["tool_mode"] = request.tool_mode
            result.metadata["active_stage"] = request.active_stage
            result.metadata["requested_model_variant"] = request.model_variant
            result.metadata["resolved_model_variant"] = resolved_variant
            result.metadata["positive_points_px"] = list(request.positive_points)
            result.metadata["negative_points_px"] = list(request.negative_points)
            result.metadata["source_token"] = request.source_token
            result.metadata["session_token"] = request.session_token
            result.metadata["roi_workspace_context"] = request.roi_workspace_context
            performance = dict(result.metadata.get("segmentation_performance", {}))
            stages = dict(performance.get("stages_ms", {}))
            stages["worker_geometry_ms"] = (perf_counter() - geometry_started) * 1000.0
            performance["stages_ms"] = stages
            result.metadata["segmentation_performance"] = self._request_performance(request, started, performance)
            result.metadata["_segmentation_emitted_at"] = perf_counter()
            if fallback_message:
                result.metadata["model_fallback_message"] = fallback_message
            self.succeeded.emit(request.document_id, request.request_id, result)
            reported = True
        except Exception as exc:  # noqa: BLE001
            if cancelled():
                return
            performance = self._request_performance(request, started, getattr(service, "last_performance", {}))
            log_slow_segmentation(performance, document_id=request.document_id, request_id=request.request_id,
                                  source_token=request.source_token, model=request.model_variant,
                                  session_token=request.session_token, tool_mode=request.tool_mode,
                                  active_stage=request.active_stage, status="failed", error=str(exc))
            reported = True
            self.failed.emit(request.document_id, request.request_id, str(exc))
        finally:
            if cancelled() and not reported:
                performance = self._request_performance(request, started, getattr(service, "last_performance", {}))
                log_slow_segmentation(performance, document_id=request.document_id, request_id=request.request_id,
                                      source_token=request.source_token, model=request.model_variant,
                                      session_token=request.session_token, tool_mode=request.tool_mode,
                                      active_stage=request.active_stage, status="cancelled")

    @Slot()
    def clear_cache(self) -> None:
        for service in self._services.values():
            service.clear_cache()
