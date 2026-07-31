"""Isolated batch execution for image-processing recipes.

This service deliberately stops at immutable *commit candidates*.  It never
adds a document, writes a project or updates dirty state.  Therefore a caller
can safely discard the whole result when cancellation is requested or when a
newer generation supersedes the request.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import json
import math
from pathlib import Path
import shutil
import tempfile
from typing import Callable, Mapping

import cv2
import numpy as np

from fdm.cancellation import CancellationError, CancellationToken
from fdm.image_processing_models import (
    ImageDerivation,
    ImageOperationSpec,
    ImageProcessingRecipe,
    RasterSemantic,
    RasterTypeState,
)
from fdm.raster import RasterPlane
from fdm.services.image_processing import (
    ImageOperation,
    execute_image_operation_tiled,
    flat_field_reference_levels,
    resolve_image_operation_capability,
    validate_image_processing_recipe,
)
from fdm.services.raster_io import numpy_to_raster_plane, raster_plane_to_numpy


BATCH_PROCESSING_TILE_EDGE = 1024
_DYNAMIC_OPERATION_METADATA_KEYS = frozenset(
    {
        "nonfinite_replacement_count",
        "repaired_count",
        "computed_threshold",
        "cropped_right",
        "cropped_bottom",
        "roi_bounds",
    }
)


class BatchItemStatus(StrEnum):
    SUCCESS = "success"
    FAILED = "failed"
    RESOURCE_BLOCKED = "resource_blocked"
    CANCELLED = "cancelled"
    STALE = "stale"


class BatchProgressPhase(StrEnum):
    PREFLIGHT = "preflight"
    PROCESSING = "processing"
    PACKAGING = "packaging"


@dataclass(frozen=True, slots=True)
class BatchProgressUpdate:
    request_id: str
    generation: int
    phase: BatchProgressPhase
    item_index: int
    item_total: int
    document_id: str = ""
    display_name: str = ""
    completed_operations: int = 0
    total_operations: int = 0
    message: str = ""

    def __post_init__(self) -> None:
        if not str(self.request_id or "").strip():
            raise ValueError("进度 request_id 不能为空")
        generation = int(self.generation)
        if generation < 0:
            raise ValueError("进度 generation 不能为负数")
        item_index = int(self.item_index)
        item_total = int(self.item_total)
        completed = int(self.completed_operations)
        total = int(self.total_operations)
        if (
            item_index < 0
            or item_total < 0
            or item_index > item_total
            or completed < 0
            or total < 0
            or completed > total
        ):
            raise ValueError("批处理进度计数不合法")
        object.__setattr__(self, "request_id", str(self.request_id).strip())
        object.__setattr__(self, "generation", generation)
        object.__setattr__(self, "phase", BatchProgressPhase(self.phase))
        object.__setattr__(self, "item_index", item_index)
        object.__setattr__(self, "item_total", item_total)
        object.__setattr__(self, "document_id", str(self.document_id))
        object.__setattr__(self, "display_name", str(self.display_name))
        object.__setattr__(self, "completed_operations", completed)
        object.__setattr__(self, "total_operations", total)
        object.__setattr__(self, "message", str(self.message))


@dataclass(frozen=True, slots=True)
class BatchExecutionLimits:
    max_working_bytes: int = 1 << 30
    min_free_disk_bytes: int = 2 << 30
    max_documents: int = 10_000

    def __post_init__(self) -> None:
        for name in ("max_working_bytes", "min_free_disk_bytes", "max_documents"):
            value = int(getattr(self, name))
            if value < 1:
                raise ValueError(f"{name} 必须为正整数")
            object.__setattr__(self, name, value)


DEFAULT_BATCH_EXECUTION_LIMITS = BatchExecutionLimits()


@dataclass(frozen=True, slots=True)
class BatchRasterInput:
    document_id: str
    display_name: str
    raster: RasterPlane
    source_pixel_revision: int = 0
    source_path: str | None = None
    roi_mask: np.ndarray | None = None
    secondary_raster: RasterPlane | None = None
    semantic: RasterSemantic | None = None
    secondary_semantic: RasterSemantic | None = None

    def __post_init__(self) -> None:
        document_id = str(self.document_id or "").strip()
        if not document_id:
            raise ValueError("document_id 不能为空")
        display_name = str(self.display_name or "").strip() or document_id
        if not isinstance(self.raster, RasterPlane):
            raise TypeError("raster 必须是 RasterPlane")
        semantic = RasterTypeState(
            pixel_type=self.raster.pixel_type,
            semantic=self.semantic,
        ).semantic
        revision = int(self.source_pixel_revision)
        if revision < 0:
            raise ValueError("source_pixel_revision 不能为负数")
        if self.secondary_raster is not None and not isinstance(
            self.secondary_raster,
            RasterPlane,
        ):
            raise TypeError("secondary_raster 必须是 RasterPlane")
        if (
            self.secondary_raster is None
            and self.secondary_semantic is not None
        ):
            raise ValueError("没有第二幅图像时不能声明第二图像语义")
        secondary_semantic = (
            None
            if self.secondary_raster is None
            else RasterTypeState(
                pixel_type=self.secondary_raster.pixel_type,
                semantic=self.secondary_semantic,
            ).semantic
        )
        roi_mask = self.roi_mask
        if roi_mask is not None:
            normalized = np.array(roi_mask, dtype=bool, copy=True, order="C")
            expected = (self.raster.height, self.raster.width)
            if normalized.shape != expected:
                raise ValueError(
                    f"ROI 尺寸 {normalized.shape} 与图像尺寸 {expected} 不一致"
                )
            normalized.setflags(write=False)
            roi_mask = normalized
        object.__setattr__(self, "document_id", document_id)
        object.__setattr__(self, "display_name", display_name)
        object.__setattr__(self, "semantic", semantic)
        object.__setattr__(self, "source_pixel_revision", revision)
        object.__setattr__(
            self,
            "source_path",
            None if self.source_path is None else str(self.source_path),
        )
        object.__setattr__(self, "roi_mask", roi_mask)
        object.__setattr__(
            self,
            "secondary_semantic",
            secondary_semantic,
        )


@dataclass(frozen=True, slots=True)
class BatchRecipeRequest:
    request_id: str
    generation: int
    recipe: ImageProcessingRecipe
    inputs: tuple[BatchRasterInput, ...]
    resource_directory: str | None = None
    available_disk_bytes: int | None = None

    def __post_init__(self) -> None:
        request_id = str(self.request_id or "").strip()
        if not request_id:
            raise ValueError("request_id 不能为空")
        generation = int(self.generation)
        if generation < 0:
            raise ValueError("generation 不能为负数")
        if not isinstance(self.recipe, ImageProcessingRecipe):
            raise TypeError("recipe 必须是 ImageProcessingRecipe")
        inputs = tuple(self.inputs)
        if not inputs:
            raise ValueError("批处理至少需要一张图片")
        if not all(isinstance(item, BatchRasterInput) for item in inputs):
            raise TypeError("inputs 必须全部是 BatchRasterInput")
        ids = tuple(item.document_id for item in inputs)
        if len(set(ids)) != len(ids):
            raise ValueError("批处理文档 ID 不能重复")
        available = self.available_disk_bytes
        if available is not None and int(available) < 0:
            raise ValueError("available_disk_bytes 不能为负数")
        object.__setattr__(self, "request_id", request_id)
        object.__setattr__(self, "generation", generation)
        object.__setattr__(self, "inputs", inputs)
        object.__setattr__(
            self,
            "resource_directory",
            None
            if self.resource_directory is None
            else str(Path(self.resource_directory).expanduser()),
        )
        object.__setattr__(
            self,
            "available_disk_bytes",
            None if available is None else int(available),
        )


@dataclass(frozen=True, slots=True)
class BatchItemResourceEstimate:
    document_id: str
    source_bytes: int
    estimated_output_bytes: int
    estimated_peak_bytes: int
    allowed: bool
    reason: str = ""


@dataclass(frozen=True, slots=True)
class BatchResourceEstimate:
    items: tuple[BatchItemResourceEstimate, ...]
    estimated_total_output_bytes: int
    available_disk_bytes: int
    reserve_disk_bytes: int
    disk_allowed: bool
    reason: str = ""

    @property
    def allowed(self) -> bool:
        return self.disk_allowed and any(item.allowed for item in self.items)


@dataclass(frozen=True, slots=True)
class DerivedRasterCandidate:
    source_document_id: str
    source_display_name: str
    raster: RasterPlane
    derivation: ImageDerivation
    operation_reports: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class BatchItemResult:
    document_id: str
    display_name: str
    status: BatchItemStatus
    message: str
    completed_operations: int = 0
    candidate: DerivedRasterCandidate | None = None


@dataclass(frozen=True, slots=True)
class BatchExecutionResult:
    request_id: str
    generation: int
    items: tuple[BatchItemResult, ...]
    preflight: BatchResourceEstimate
    cancelled: bool = False
    stale: bool = False

    @property
    def commit_allowed(self) -> bool:
        return not self.cancelled and not self.stale

    @property
    def commit_candidates(self) -> tuple[DerivedRasterCandidate, ...]:
        if not self.commit_allowed:
            return ()
        return tuple(
            item.candidate
            for item in self.items
            if item.status is BatchItemStatus.SUCCESS and item.candidate is not None
        )

    @property
    def success_count(self) -> int:
        return sum(item.status is BatchItemStatus.SUCCESS for item in self.items)

    @property
    def failure_count(self) -> int:
        return sum(
            item.status
            in {BatchItemStatus.FAILED, BatchItemStatus.RESOURCE_BLOCKED}
            for item in self.items
        )

    @property
    def summary_text(self) -> str:
        if self.cancelled:
            prefix = "批处理已取消，所有候选结果均未提交"
        elif self.stale:
            prefix = "批处理结果已过期，所有候选结果均已丢弃"
        else:
            prefix = "批处理完成"
        return (
            f"{prefix}：成功 {self.success_count} 张，"
            f"失败 {self.failure_count} 张，共 {len(self.items)} 张。"
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "request_id": self.request_id,
            "generation": self.generation,
            "cancelled": self.cancelled,
            "stale": self.stale,
            "commit_allowed": self.commit_allowed,
            "summary": self.summary_text,
            "counts": {
                "total": len(self.items),
                "success": self.success_count,
                "failure": self.failure_count,
            },
            "preflight": {
                "estimated_total_output_bytes": (
                    self.preflight.estimated_total_output_bytes
                ),
                "available_disk_bytes": self.preflight.available_disk_bytes,
                "reserve_disk_bytes": self.preflight.reserve_disk_bytes,
                "disk_allowed": self.preflight.disk_allowed,
                "reason": self.preflight.reason,
            },
            "items": [
                {
                    "document_id": item.document_id,
                    "display_name": item.display_name,
                    "status": item.status.value,
                    "message": item.message,
                    "completed_operations": item.completed_operations,
                    "result_sha256": (
                        ""
                        if item.candidate is None
                        else item.candidate.raster.sha256()
                    ),
                }
                for item in self.items
            ],
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
        )


GenerationPredicate = Callable[[int], bool]
BatchProgressCallback = Callable[[BatchProgressUpdate], None]


def preflight_batch_recipe(
    request: BatchRecipeRequest,
    *,
    limits: BatchExecutionLimits = DEFAULT_BATCH_EXECUTION_LIMITS,
) -> BatchResourceEstimate:
    if len(request.inputs) > limits.max_documents:
        item_estimates = tuple(
            BatchItemResourceEstimate(
                document_id=item.document_id,
                source_bytes=item.raster.byte_count,
                estimated_output_bytes=item.raster.byte_count,
                estimated_peak_bytes=item.raster.byte_count,
                allowed=False,
                reason=f"批处理最多允许 {limits.max_documents} 张图片。",
            )
            for item in request.inputs
        )
    else:
        item_estimates = tuple(
            _estimate_item_resources(item, request.recipe, limits)
            for item in request.inputs
        )
    output_bytes = sum(item.estimated_output_bytes for item in item_estimates)
    available = (
        request.available_disk_bytes
        if request.available_disk_bytes is not None
        else _available_disk_bytes(request.resource_directory)
    )
    disk_allowed = (
        available >= limits.min_free_disk_bytes
        and output_bytes <= available - limits.min_free_disk_bytes
    )
    reason = ""
    if not disk_allowed:
        reason = (
            "预计派生资产写入后无法保留至少 "
            f"{_format_bytes(limits.min_free_disk_bytes)} 可用磁盘空间。"
        )
    return BatchResourceEstimate(
        items=item_estimates,
        estimated_total_output_bytes=output_bytes,
        available_disk_bytes=available,
        reserve_disk_bytes=limits.min_free_disk_bytes,
        disk_allowed=disk_allowed,
        reason=reason,
    )


def execute_batch_recipe(
    request: BatchRecipeRequest,
    *,
    cancellation_token: CancellationToken | None = None,
    generation_is_current: GenerationPredicate | None = None,
    progress_callback: BatchProgressCallback | None = None,
    limits: BatchExecutionLimits = DEFAULT_BATCH_EXECUTION_LIMITS,
) -> BatchExecutionResult:
    """Execute selected images sequentially with per-document isolation."""

    _emit_progress(
        progress_callback,
        BatchProgressUpdate(
            request_id=request.request_id,
            generation=request.generation,
            phase=BatchProgressPhase.PREFLIGHT,
            item_index=0,
            item_total=len(request.inputs),
            message="正在检查内存、磁盘空间和处理配方。",
        ),
    )
    validation_errors = _validate_batch_inputs(request)
    preflight = preflight_batch_recipe(request, limits=limits)
    if not preflight.disk_allowed:
        _emit_progress(
            progress_callback,
            BatchProgressUpdate(
                request_id=request.request_id,
                generation=request.generation,
                phase=BatchProgressPhase.PACKAGING,
                item_index=0,
                item_total=len(request.inputs),
                message="磁盘空间预检未通过，正在整理阻断结果。",
            ),
        )
        return BatchExecutionResult(
            request_id=request.request_id,
            generation=request.generation,
            items=tuple(
                BatchItemResult(
                    item.document_id,
                    item.display_name,
                    (
                        BatchItemStatus.FAILED
                        if item.document_id in validation_errors
                        else BatchItemStatus.RESOURCE_BLOCKED
                    ),
                    validation_errors.get(item.document_id, preflight.reason),
                )
                for item in request.inputs
            ),
            preflight=preflight,
        )

    item_estimates = {
        item.document_id: item for item in preflight.items
    }
    results: list[BatchItemResult] = []
    cancelled = False
    stale = False
    for input_index, item in enumerate(request.inputs):
        if _is_cancelled(cancellation_token):
            cancelled = True
            results.extend(
                _remaining_results(
                    request.inputs[input_index:],
                    BatchItemStatus.CANCELLED,
                    "批处理已取消，未提交任何派生图片。",
                )
            )
            break
        if not _generation_current(request.generation, generation_is_current):
            stale = True
            results.extend(
                _remaining_results(
                    request.inputs[input_index:],
                    BatchItemStatus.STALE,
                    "请求 generation 已过期，结果已丢弃。",
                )
            )
            break
        validation_error = validation_errors.get(item.document_id)
        if validation_error is not None:
            results.append(
                BatchItemResult(
                    item.document_id,
                    item.display_name,
                    BatchItemStatus.FAILED,
                    validation_error,
                )
            )
            continue
        estimate = item_estimates[item.document_id]
        if not estimate.allowed:
            results.append(
                BatchItemResult(
                    item.document_id,
                    item.display_name,
                    BatchItemStatus.RESOURCE_BLOCKED,
                    estimate.reason,
                )
            )
            continue
        try:
            candidate, completed = _execute_item(
                request,
                item,
                item_index=input_index + 1,
                item_total=len(request.inputs),
                cancellation_token=cancellation_token,
                generation_is_current=generation_is_current,
                progress_callback=progress_callback,
            )
        except CancellationError:
            cancelled = True
            results.append(
                BatchItemResult(
                    item.document_id,
                    item.display_name,
                    BatchItemStatus.CANCELLED,
                    "处理期间收到取消请求，当前图片未生成派生结果。",
                )
            )
            results.extend(
                _remaining_results(
                    request.inputs[input_index + 1 :],
                    BatchItemStatus.CANCELLED,
                    "批处理已取消，未开始处理。",
                )
            )
            break
        except _StaleGenerationError:
            stale = True
            results.append(
                BatchItemResult(
                    item.document_id,
                    item.display_name,
                    BatchItemStatus.STALE,
                    "处理期间 generation 已过期，当前结果已丢弃。",
                )
            )
            results.extend(
                _remaining_results(
                    request.inputs[input_index + 1 :],
                    BatchItemStatus.STALE,
                    "请求 generation 已过期，未开始处理。",
                )
            )
            break
        except Exception as exc:
            results.append(
                BatchItemResult(
                    item.document_id,
                    item.display_name,
                    BatchItemStatus.FAILED,
                    f"处理失败：{exc}",
                )
            )
            continue
        results.append(
            BatchItemResult(
                item.document_id,
                item.display_name,
                BatchItemStatus.SUCCESS,
                "已生成待提交的派生图片候选。",
                completed_operations=completed,
                candidate=candidate,
            )
        )

    if not cancelled and _is_cancelled(cancellation_token):
        cancelled = True
    if not stale and not _generation_current(
        request.generation,
        generation_is_current,
    ):
        stale = True
    _emit_progress(
        progress_callback,
        BatchProgressUpdate(
            request_id=request.request_id,
            generation=request.generation,
            phase=BatchProgressPhase.PACKAGING,
            item_index=len(results),
            item_total=len(request.inputs),
            completed_operations=0,
            total_operations=0,
            message=(
                "正在整理取消结果，所有候选均保持未提交状态。"
                if cancelled
                else (
                    "请求已过期，正在丢弃所有候选结果。"
                    if stale
                    else "正在整理批处理结果和逐图片报告。"
                )
            ),
        ),
    )
    return BatchExecutionResult(
        request_id=request.request_id,
        generation=request.generation,
        items=tuple(results),
        preflight=preflight,
        cancelled=cancelled,
        stale=stale,
    )


class _StaleGenerationError(RuntimeError):
    pass


def _execute_item(
    request: BatchRecipeRequest,
    item: BatchRasterInput,
    *,
    item_index: int,
    item_total: int,
    cancellation_token: CancellationToken | None,
    generation_is_current: GenerationPredicate | None,
    progress_callback: BatchProgressCallback | None,
) -> tuple[DerivedRasterCandidate, int]:
    output_state = _resolve_batch_output_state(
        item.raster,
        request.recipe.operations,
        source_semantic=item.semantic,
        roi_requested=item.roi_mask is not None,
        secondary_raster=item.secondary_raster,
        secondary_semantic=item.secondary_semantic,
    )
    image = np.asarray(raster_plane_to_numpy(item.raster))
    working_roi_mask = item.roi_mask
    operation_reports: list[tuple[str, str]] = []
    executed_operations: list[ImageOperationSpec] = []
    completed = 0
    for operation_spec in request.recipe.operations:
        _require_current(
            request.generation,
            cancellation_token,
            generation_is_current,
        )
        parameters = operation_spec.parameters
        secondary_image = None
        flat_field_source = str(
            parameters.get("flat_field_source", "estimated")
        ).strip().lower()
        uses_secondary_image = (
            operation_spec.operation_id
            == ImageOperation.IMAGE_CALCULATOR.value
            or (
                operation_spec.operation_id
                == ImageOperation.FLAT_FIELD_CORRECTION.value
                and flat_field_source == "reference"
            )
        )
        if uses_secondary_image:
            if item.secondary_raster is None:
                message = (
                    "参考图平场校正缺少与当前图片严格兼容的参考图像"
                    if operation_spec.operation_id
                    == ImageOperation.FLAT_FIELD_CORRECTION.value
                    else "图像计算器缺少与当前图片对齐的第二幅图像"
                )
                raise ValueError(message)
            secondary_image = np.asarray(
                raster_plane_to_numpy(item.secondary_raster)
            )
            if (
                operation_spec.operation_id
                == ImageOperation.FLAT_FIELD_CORRECTION.value
            ):
                actual_sha256 = item.secondary_raster.sha256()
                expected_sha256 = str(
                    parameters.get("secondary_sha256", "")
                ).strip()
                if expected_sha256 and expected_sha256 != actual_sha256:
                    raise ValueError(
                        "参考平场像素摘要与当前批处理参考图像不一致"
                    )
                parameters["secondary_sha256"] = actual_sha256
                parameters["reference_levels"] = (
                    flat_field_reference_levels(
                        secondary_image,
                        preserve_mean=bool(
                            parameters.get("preserve_mean", True)
                        ),
                    )
                )
            else:
                parameters.pop("secondary_document_id", None)
        _emit_progress(
            progress_callback,
            BatchProgressUpdate(
                request_id=request.request_id,
                generation=request.generation,
                phase=BatchProgressPhase.PROCESSING,
                item_index=item_index,
                item_total=item_total,
                document_id=item.document_id,
                display_name=item.display_name,
                completed_operations=completed,
                total_operations=len(request.recipe.operations),
                message=(
                    f"正在处理 {item.display_name}："
                    f"{operation_spec.operation_id}"
                ),
            ),
        )

        def check_current() -> None:
            _require_current(
                request.generation,
                cancellation_token,
                generation_is_current,
            )

        operation_result = execute_image_operation_tiled(
            ImageOperation(operation_spec.operation_id),
            image,
            parameters=parameters,
            secondary_image=secondary_image,
            roi_mask=working_roi_mask,
            request_id=request.request_id,
            generation=request.generation,
            tile_size=BATCH_PROCESSING_TILE_EDGE,
            cancellation_check=check_current,
        )
        _require_current(
            request.generation,
            cancellation_token,
            generation_is_current,
        )
        if (
            operation_result.request_id != request.request_id
            or operation_result.generation != request.generation
        ):
            raise RuntimeError("图像处理结果的 request_id/generation 与请求不一致")
        image = np.asarray(operation_result.image)
        if operation_result.roi_mask is not None:
            working_roi_mask = operation_result.roi_mask
        dynamic_metadata = {
            key: value
            for key, value in operation_result.metadata_map.items()
            if key in _DYNAMIC_OPERATION_METADATA_KEYS
        }
        merged_metadata = operation_spec.result_metadata
        merged_metadata.update(dynamic_metadata)
        executed_operations.append(
            ImageOperationSpec(
                operation_spec.operation_id,
                parameters,
                implementation=operation_spec.implementation,
                implementation_version=operation_spec.implementation_version,
                result_metadata=merged_metadata,
            )
        )
        completed += 1
        operation_reports.append(
            (
                operation_spec.operation_id,
                "完成"
                if not operation_result.warnings
                else "完成；" + "；".join(operation_result.warnings),
            )
        )
        _emit_progress(
            progress_callback,
            BatchProgressUpdate(
                request_id=request.request_id,
                generation=request.generation,
                phase=BatchProgressPhase.PROCESSING,
                item_index=item_index,
                item_total=item_total,
                document_id=item.document_id,
                display_name=item.display_name,
                completed_operations=completed,
                total_operations=len(request.recipe.operations),
                message=(
                    f"{item.display_name} 已完成 "
                    f"{completed}/{len(request.recipe.operations)} 个步骤。"
                ),
            ),
        )
    _require_current(
        request.generation,
        cancellation_token,
        generation_is_current,
    )
    raster = numpy_to_raster_plane(image)
    derivation = ImageDerivation(
        source_document_id=item.document_id,
        source_path=item.source_path,
        source_sha256=item.raster.sha256(),
        source_image_size=(item.raster.width, item.raster.height),
        source_pixel_revision=item.source_pixel_revision,
        source_pixel_type=item.raster.pixel_type,
        source_semantic=item.semantic,
        recipe=ImageProcessingRecipe.from_operations(executed_operations),
        result_pixel_type=raster.pixel_type,
        result_semantic=output_state.semantic,
        result_image_size=(raster.width, raster.height),
        result_sha256=raster.sha256(),
        library_versions=(
            ("OpenCV", str(cv2.__version__)),
            ("NumPy", str(np.__version__)),
            ("fdm.image_processing", "1"),
        ),
    )
    return (
        DerivedRasterCandidate(
            source_document_id=item.document_id,
            source_display_name=item.display_name,
            raster=raster,
            derivation=derivation,
            operation_reports=tuple(operation_reports),
        ),
        completed,
    )


def _validate_batch_inputs(
    request: BatchRecipeRequest,
) -> dict[str, str]:
    errors: dict[str, str] = {}
    for item in request.inputs:
        try:
            _validate_batch_operation_sequence(
                item.raster,
                request.recipe.operations,
                source_semantic=item.semantic,
                roi_requested=item.roi_mask is not None,
                secondary_raster=item.secondary_raster,
                secondary_semantic=item.secondary_semantic,
            )
        except (TypeError, ValueError) as exc:
            errors[item.document_id] = (
                f"处理配方与图片像素类型不兼容：{exc}"
            )
    return errors


def _validate_batch_operation_sequence(
    source: RasterPlane,
    operations: tuple[ImageOperationSpec, ...],
    *,
    source_semantic: RasterSemantic | None = None,
    roi_requested: bool = False,
    secondary_raster: RasterPlane | None = None,
    secondary_semantic: RasterSemantic | None = None,
) -> RasterTypeState:
    """Validate a batch recipe through the same registry as the workbench."""

    return _resolve_batch_output_state(
        source,
        operations,
        source_semantic=source_semantic,
        roi_requested=roi_requested,
        secondary_raster=secondary_raster,
        secondary_semantic=secondary_semantic,
    )


def _resolve_batch_output_state(
    source: RasterPlane,
    operations: tuple[ImageOperationSpec, ...],
    *,
    source_semantic: RasterSemantic | None = None,
    roi_requested: bool = False,
    secondary_raster: RasterPlane | None = None,
    secondary_semantic: RasterSemantic | None = None,
) -> RasterTypeState:
    """Resolve the already-preflighted batch output semantic.

    Execution calls this directly so the public preflight hook remains exactly
    once per input, while the commit candidate still carries the authoritative
    semantic derived from the same registry contract.
    """

    source_state = RasterTypeState(
        pixel_type=source.pixel_type,
        semantic=source_semantic,
        width=source.width,
        height=source.height,
    )
    secondary_states: dict[str, RasterTypeState] = {}
    if secondary_raster is not None:
        secondary_state = RasterTypeState(
            pixel_type=secondary_raster.pixel_type,
            semantic=secondary_semantic,
            width=secondary_raster.width,
            height=secondary_raster.height,
        )
        for operation in operations:
            flat_field_source = str(
                operation.parameters.get(
                    "flat_field_source",
                    "estimated",
                )
            ).strip().lower()
            if (
                operation.operation_id
                == ImageOperation.IMAGE_CALCULATOR.value
                or (
                    operation.operation_id
                    == ImageOperation.FLAT_FIELD_CORRECTION.value
                    and flat_field_source == "reference"
                )
            ):
                document_id = str(
                    operation.parameters.get("secondary_document_id", "")
                )
                secondary_states[document_id] = secondary_state
    validation = validate_image_processing_recipe(
        ImageProcessingRecipe.from_operations(operations),
        source_state,
        roi_requested=roi_requested,
        secondary_states=secondary_states,
    )
    return validation.output_state


def _estimate_item_resources(
    item: BatchRasterInput,
    recipe: ImageProcessingRecipe,
    limits: BatchExecutionLimits,
) -> BatchItemResourceEstimate:
    width = item.raster.width
    height = item.raster.height
    channels = item.raster.pixel_type.channel_count
    bytes_per_channel = item.raster.pixel_type.bytes_per_channel
    output_bytes = item.raster.byte_count
    secondary_bytes = (
        0 if item.secondary_raster is None else item.secondary_raster.byte_count
    )
    roi_bytes = 0 if item.roi_mask is None else int(item.roi_mask.nbytes)
    current_bytes = item.raster.byte_count
    peak_bytes = current_bytes * 2 + secondary_bytes + roi_bytes
    for operation in recipe.operations:
        input_channels = channels
        input_bytes_per_channel = bytes_per_channel
        (
            width,
            height,
            channels,
            bytes_per_channel,
        ) = _estimated_output_layout(
            width,
            height,
            channels,
            bytes_per_channel,
            operation,
        )
        output_bytes = max(
            1,
            width * height * channels * bytes_per_channel,
        )
        multiplier = _operation_working_multiplier(operation.operation_id)
        capability = resolve_image_operation_capability(
            operation.operation_id,
            operation.parameters,
        )
        if (
            capability.tileable
            and capability.preserves_spatial_extent
            and (item.roi_mask is None or capability.supports_roi)
        ):
            tile_width = min(
                width,
                BATCH_PROCESSING_TILE_EDGE + 2 * capability.halo_x,
            )
            tile_height = min(
                height,
                BATCH_PROCESSING_TILE_EDGE + 2 * capability.halo_y,
            )
            tile_bytes = max(
                1,
                tile_width
                * tile_height
                * max(
                    input_channels * input_bytes_per_channel,
                    channels * bytes_per_channel,
                ),
            )
            operation_peak = (
                current_bytes
                + output_bytes
                + secondary_bytes
                + roi_bytes
                + tile_bytes * multiplier
            )
        else:
            operation_peak = (
                current_bytes
                + secondary_bytes
                + roi_bytes
                + output_bytes * multiplier
            )
        peak_bytes = max(peak_bytes, operation_peak)
        current_bytes = output_bytes
    allowed = peak_bytes <= limits.max_working_bytes
    reason = ""
    if not allowed:
        reason = (
            f"预计工作集 {_format_bytes(peak_bytes)} 超过 "
            f"{_format_bytes(limits.max_working_bytes)} 上限。"
        )
    return BatchItemResourceEstimate(
        document_id=item.document_id,
        source_bytes=item.raster.byte_count,
        estimated_output_bytes=output_bytes,
        estimated_peak_bytes=peak_bytes,
        allowed=allowed,
        reason=reason,
    )


def _estimated_output_layout(
    width: int,
    height: int,
    channels: int,
    bytes_per_channel: int,
    spec: ImageOperationSpec,
) -> tuple[int, int, int, int]:
    params: Mapping[str, object] = spec.parameters
    operation = spec.operation_id
    if operation == ImageOperation.CROP.value:
        width = max(1, int(params.get("width", width)))
        height = max(1, int(params.get("height", height)))
    elif operation in {
        ImageOperation.RESIZE.value,
        ImageOperation.RESIZE_CANVAS.value,
    }:
        width = max(1, int(params.get("width", width)))
        height = max(1, int(params.get("height", height)))
    elif operation == ImageOperation.PIXEL_BIN.value:
        factor = max(1, int(params.get("factor", 2)))
        width = max(1, width // factor)
        height = max(1, height // factor)
    elif operation in {
        ImageOperation.ROTATE_90_CLOCKWISE.value,
        ImageOperation.ROTATE_90_COUNTERCLOCKWISE.value,
    }:
        width, height = height, width
    elif operation == ImageOperation.ROTATE.value and bool(
        params.get("expand", True)
    ):
        angle = math.radians(float(params.get("angle_degrees", 0.0)))
        cosine = abs(math.cos(angle))
        sine = abs(math.sin(angle))
        original_width = width
        original_height = height
        width = max(
            1,
            int(math.ceil(original_width * cosine + original_height * sine)),
        )
        height = max(
            1,
            int(math.ceil(original_width * sine + original_height * cosine)),
        )
    if operation == ImageOperation.CONVERT_TYPE.value:
        target = str(params.get("target_type", "uint8"))
        bytes_per_channel = {"uint8": 1, "uint16": 2, "float32": 4}.get(
            target,
            bytes_per_channel,
        )
    elif operation == ImageOperation.CONVERT_COLOR.value:
        target = str(params.get("target_model", "grayscale"))
        channels = 1 if target == "grayscale" else 3
        if target != "grayscale":
            bytes_per_channel = 1
    elif operation == ImageOperation.COPY.value and bool(
        params.get("transparent_outside", False)
    ):
        channels = 4
        bytes_per_channel = 1
    elif operation in {
        ImageOperation.LOG_V2.value,
        ImageOperation.EXP_V2.value,
        ImageOperation.SQRT_V2.value,
    } and str(params.get("result_mode", "float32")) == "float32":
        channels = 1
        bytes_per_channel = 4
    elif operation == ImageOperation.IMAGE_CALCULATOR.value and str(
        params.get("result_mode", "preserve")
    ) == "float32":
        channels = 1
        bytes_per_channel = 4
    elif operation == ImageOperation.RANK_FILTER.value and str(
        params.get("method", "minimum")
    ) == "variance":
        channels = 1
        bytes_per_channel = 4
    elif operation == ImageOperation.MORPHOLOGY_DERIVATIVE.value and str(
        params.get("method", "gradient")
    ) == "laplacian":
        channels = 1
        bytes_per_channel = 4
    elif operation == ImageOperation.FFT_POWER_SPECTRUM.value:
        channels = 1
        bytes_per_channel = 4
    elif operation in {
        ImageOperation.SOBEL_EDGES.value,
        ImageOperation.LAPLACIAN_EDGES.value,
        ImageOperation.DISTANCE_TRANSFORM.value,
    }:
        channels = 1
        bytes_per_channel = 4
    elif operation in {
        ImageOperation.THRESHOLD.value,
        ImageOperation.AUTO_THRESHOLD.value,
        ImageOperation.BINARIZE.value,
        ImageOperation.CANNY_EDGES.value,
        ImageOperation.ERODE.value,
        ImageOperation.DILATE.value,
        ImageOperation.MORPHOLOGY_OPEN.value,
        ImageOperation.MORPHOLOGY_CLOSE.value,
        ImageOperation.FILL_HOLES.value,
        ImageOperation.CONTOUR_EXTRACT.value,
        ImageOperation.REMOVE_SMALL_OBJECTS.value,
        ImageOperation.FILL_SMALL_HOLES.value,
        ImageOperation.SKELETONIZE.value,
        ImageOperation.WATERSHED.value,
        ImageOperation.WATERSHED_V2.value,
        ImageOperation.ADAPTIVE_THRESHOLD.value,
        ImageOperation.REGIONAL_EXTREMA.value,
        ImageOperation.CLEAR_BORDER.value,
        ImageOperation.TOP_HAT.value,
        ImageOperation.BLACK_HAT.value,
    }:
        channels = 1
    return width, height, channels, bytes_per_channel


def _operation_working_multiplier(operation_id: str) -> int:
    if operation_id in {
        ImageOperation.FFT_FILTER.value,
        ImageOperation.FFT_POWER_SPECTRUM.value,
        ImageOperation.STRIPE_SUPPRESSION.value,
    }:
        return 24
    if operation_id in {
        ImageOperation.WATERSHED.value,
        ImageOperation.WATERSHED_V2.value,
        ImageOperation.MORPHOLOGICAL_RECONSTRUCTION.value,
        ImageOperation.DISTANCE_TRANSFORM.value,
        ImageOperation.SKELETONIZE.value,
    }:
        return 12
    return 6


def _available_disk_bytes(resource_directory: str | None) -> int:
    directory = Path(resource_directory or tempfile.gettempdir()).expanduser()
    probe = directory
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        return int(shutil.disk_usage(probe).free)
    except OSError as exc:
        raise RuntimeError(f"无法检查批处理磁盘空间：{exc}") from exc


def _remaining_results(
    inputs: tuple[BatchRasterInput, ...],
    status: BatchItemStatus,
    message: str,
) -> tuple[BatchItemResult, ...]:
    return tuple(
        BatchItemResult(
            item.document_id,
            item.display_name,
            status,
            message,
        )
        for item in inputs
    )


def _is_cancelled(token: CancellationToken | None) -> bool:
    return token is not None and token.is_cancelled


def _generation_current(
    generation: int,
    predicate: GenerationPredicate | None,
) -> bool:
    return predicate is None or bool(predicate(generation))


def _require_current(
    generation: int,
    token: CancellationToken | None,
    predicate: GenerationPredicate | None,
) -> None:
    if token is not None:
        token.raise_if_cancelled()
    if not _generation_current(generation, predicate):
        raise _StaleGenerationError


def _emit_progress(
    callback: BatchProgressCallback | None,
    update: BatchProgressUpdate,
) -> None:
    if callback is not None:
        callback(update)


def _format_bytes(value: int) -> str:
    amount = float(value)
    for suffix in ("B", "KiB", "MiB", "GiB", "TiB"):
        if amount < 1024.0 or suffix == "TiB":
            return f"{amount:.1f} {suffix}"
        amount /= 1024.0
    return f"{amount:.1f} TiB"  # pragma: no cover


__all__ = [
    "BATCH_PROCESSING_TILE_EDGE",
    "BatchExecutionLimits",
    "BatchExecutionResult",
    "BatchItemResourceEstimate",
    "BatchItemResult",
    "BatchItemStatus",
    "BatchProgressCallback",
    "BatchProgressPhase",
    "BatchProgressUpdate",
    "BatchRasterInput",
    "BatchRecipeRequest",
    "BatchResourceEstimate",
    "DEFAULT_BATCH_EXECUTION_LIMITS",
    "DerivedRasterCandidate",
    "execute_batch_recipe",
    "preflight_batch_recipe",
]
