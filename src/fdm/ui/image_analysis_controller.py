"""Background orchestration for image-analysis tools.

This module deliberately owns no project or main-window state.  It freezes the
pixels, ROI and RAW measurement geometry needed by one analysis request, runs
at most one worker at a time, rejects late generations, and converts kernel
results into the small immutable records used by :mod:`fdm.analysis_artifacts`.

Large arrays are returned as :class:`AnalysisAssetPayload` values.  They contain
only non-object NumPy arrays plus a versioned schema and are intended to be
written as ``allow_pickle=False`` NPZ assets by the project persistence layer.
No file is written from the worker thread.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
import json
import math
import re
from types import MappingProxyType
from typing import Any, TypeAlias
from uuid import uuid4

import cv2
import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal, Slot

from fdm.analysis_artifacts import (
    AnalysisArtifact,
    AnalysisAssetKind,
    AnalysisAssetReference,
    AnalysisCurve,
    AnalysisObjectReference,
    AnalysisTable,
)
from fdm.cancellation import (
    CancellationError,
    CancellationToken,
    CancellationTokenSource,
)
from fdm.raster import RasterPixelType, RasterPlane
from fdm.services.advanced_image_analysis import (
    DEFAULT_ADVANCED_ANALYSIS_LIMITS,
    AdvancedAnalysisResourceEstimate,
    DirectionalityRequest,
    GlcmHaralickRequest,
    IntensitySurfaceRequest,
    LocalThicknessRequest,
    SkeletonNetworkRequest,
    SpatialDistributionRequest,
    TubenessRequest,
    analyze_fiber_directionality,
    analyze_skeleton_network,
    analyze_spatial_distribution,
    build_intensity_surface,
    calculate_glcm_haralick,
    calculate_local_thickness,
    calculate_multiscale_tubeness,
    estimate_advanced_analysis_resources,
)
from fdm.services.image_analysis import (
    FindMaximaRequest,
    HistogramRequest,
    IntensityAnalysisRequest,
    IntensityProfileRequest,
    ParticleAnalysisRequest,
    ShapeAnalysisRequest,
    analyze_intensity,
    analyze_particles,
    analyze_shape,
    calculate_histogram,
    find_local_maxima,
    sample_intensity_profile,
)


MAX_ANALYSIS_WORKING_BYTES = 1 << 30
_INLINE_DETAIL_ROWS = 5_000
_INLINE_CURVE_POINTS = 50_000
_ASSET_MEMBER_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")
_ASSET_SCHEMA_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

Coordinate: TypeAlias = tuple[float, float]
ImmutableRing: TypeAlias = tuple[Coordinate, ...]
ImmutableRings: TypeAlias = tuple[ImmutableRing, ...]
JsonScalar: TypeAlias = str | int | float | bool | None


class AnalysisTool(StrEnum):
    SHAPE = "shape"
    INTENSITY = "intensity"
    HISTOGRAM = "histogram"
    PROFILE = "profile"
    PARTICLES = "particles"
    MAXIMA = "maxima"
    DIRECTIONALITY = "directionality"
    SKELETON = "skeleton"
    LOCAL_THICKNESS = "local_thickness"
    TUBENESS = "tubeness"
    GLCM = "glcm"
    SPATIAL_DISTRIBUTION = "spatial_distribution"
    SURFACE = "surface"

    @property
    def chinese_name(self) -> str:
        return {
            AnalysisTool.SHAPE: "形状测量",
            AnalysisTool.INTENSITY: "灰度与颜色统计",
            AnalysisTool.HISTOGRAM: "直方图",
            AnalysisTool.PROFILE: "强度剖面",
            AnalysisTool.PARTICLES: "粒子分析",
            AnalysisTool.MAXIMA: "极值检测",
            AnalysisTool.DIRECTIONALITY: "纤维方向性",
            AnalysisTool.SKELETON: "骨架网络",
            AnalysisTool.LOCAL_THICKNESS: "局部厚度",
            AnalysisTool.TUBENESS: "Tubeness",
            AnalysisTool.GLCM: "Haralick GLCM 纹理",
            AnalysisTool.SPATIAL_DISTRIBUTION: "最近邻与空间密度",
            AnalysisTool.SURFACE: "二维强度表面",
        }[self]


class AnalysisTaskPhase(StrEnum):
    PREPARING = "preparing"
    ANALYZING = "analyzing"
    PACKAGING = "packaging"

    @property
    def chinese_name(self) -> str:
        return {
            AnalysisTaskPhase.PREPARING: "预扫描与资源检查",
            AnalysisTaskPhase.ANALYZING: "执行分析",
            AnalysisTaskPhase.PACKAGING: "整理分析结果",
        }[self]


@dataclass(frozen=True, slots=True)
class AnalysisCalibrationSnapshot:
    pixel_size_x: float = 1.0
    pixel_size_y: float = 1.0
    unit: str = "px"
    signature: str | None = None

    def __post_init__(self) -> None:
        x = _positive_finite(self.pixel_size_x, field_name="pixel_size_x")
        y = _positive_finite(self.pixel_size_y, field_name="pixel_size_y")
        unit = str(self.unit or "px").strip()
        if not unit:
            raise ValueError("标定单位不能为空")
        signature = None if self.signature is None else str(self.signature).strip()
        if signature == "":
            signature = None
        object.__setattr__(self, "pixel_size_x", x)
        object.__setattr__(self, "pixel_size_y", y)
        object.__setattr__(self, "unit", unit)
        object.__setattr__(self, "signature", signature)


@dataclass(frozen=True, slots=True, init=False)
class ImageAnalysisTaskRequest:
    tool: AnalysisTool
    request_id: str
    generation: int
    document_id: str
    source_pixel_revision: int
    plane: RasterPlane
    roi_mask: NDArray[np.bool_] | None = field(
        default=None,
        compare=False,
        repr=False,
    )
    raw_rings: ImmutableRings = ()
    exact_area_px: float | None = None
    source_reference: AnalysisObjectReference | None = None
    calibration: AnalysisCalibrationSnapshot = field(
        default_factory=AnalysisCalibrationSnapshot,
    )
    _parameters_json: str = field(default="{}", repr=False, compare=True)

    def __init__(
        self,
        *,
        tool: AnalysisTool | str,
        request_id: str,
        generation: int,
        document_id: str,
        source_pixel_revision: int,
        plane: RasterPlane,
        roi_mask: NDArray[np.bool_] | None = None,
        raw_rings: Iterable[Iterable[Any]] = (),
        exact_area_px: float | None = None,
        source_reference: AnalysisObjectReference | None = None,
        calibration: AnalysisCalibrationSnapshot | None = None,
        parameters: Mapping[str, object] | None = None,
    ) -> None:
        try:
            resolved_tool = AnalysisTool(tool)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"不支持的分析工具：{tool!r}") from exc
        request_token = str(request_id or "").strip()
        document_token = str(document_id or "").strip()
        if not request_token:
            raise ValueError("request_id 不能为空")
        if not document_token:
            raise ValueError("document_id 不能为空")
        generation_value = _non_negative_int(generation, field_name="generation")
        revision = _non_negative_int(
            source_pixel_revision,
            field_name="source_pixel_revision",
        )
        if not isinstance(plane, RasterPlane) or plane.is_empty:
            raise ValueError("分析任务必须包含非空 RasterPlane")
        mask = _freeze_optional_mask(roi_mask, (plane.height, plane.width))
        rings = _freeze_rings(raw_rings)
        exact = exact_area_px
        if exact is not None:
            exact = _non_negative_finite(exact, field_name="exact_area_px")
        if source_reference is not None and not isinstance(
            source_reference,
            AnalysisObjectReference,
        ):
            raise TypeError("source_reference 必须是 AnalysisObjectReference")
        parameters_json = json.dumps(
            dict(parameters or {}),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        if not isinstance(json.loads(parameters_json), dict):
            raise TypeError("parameters 必须是 JSON 对象")

        object.__setattr__(self, "tool", resolved_tool)
        object.__setattr__(self, "request_id", request_token)
        object.__setattr__(self, "generation", generation_value)
        object.__setattr__(self, "document_id", document_token)
        object.__setattr__(self, "source_pixel_revision", revision)
        object.__setattr__(self, "plane", plane)
        object.__setattr__(self, "roi_mask", mask)
        object.__setattr__(self, "raw_rings", rings)
        object.__setattr__(self, "exact_area_px", exact)
        object.__setattr__(self, "source_reference", source_reference)
        object.__setattr__(
            self,
            "calibration",
            calibration or AnalysisCalibrationSnapshot(),
        )
        object.__setattr__(self, "_parameters_json", parameters_json)

    @property
    def parameters(self) -> dict[str, object]:
        return json.loads(self._parameters_json)


@dataclass(frozen=True, slots=True, init=False)
class AnalysisAssetPayload:
    """Safe in-memory payload for a future ``analysis/*.npz`` asset."""

    kind: AnalysisAssetKind
    schema: str
    suggested_stem: str
    arrays: tuple[tuple[str, NDArray[np.generic]], ...] = field(
        compare=False,
        repr=False,
    )
    _metadata_json: str = field(repr=False)

    def __init__(
        self,
        *,
        kind: AnalysisAssetKind | str,
        schema: str,
        suggested_stem: str,
        arrays: Mapping[str, NDArray[Any]],
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        try:
            resolved_kind = AnalysisAssetKind(kind)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"不支持的分析资产类型：{kind!r}") from exc
        schema_token = str(schema or "").strip()
        if not _ASSET_SCHEMA_PATTERN.fullmatch(schema_token):
            raise ValueError("分析资产 schema 不合法")
        stem = re.sub(r"[^A-Za-z0-9._-]+", "-", str(suggested_stem)).strip(".-")
        if not stem:
            raise ValueError("分析资产建议文件名不能为空")
        frozen_arrays: list[tuple[str, NDArray[np.generic]]] = []
        for name, value in arrays.items():
            member = str(name)
            if not _ASSET_MEMBER_PATTERN.fullmatch(member):
                raise ValueError(f"NPZ 成员名称不合法：{member!r}")
            array = np.asarray(value)
            if array.dtype.hasobject:
                raise TypeError("分析资产禁止 object dtype")
            if array.dtype.kind not in "biufc":
                raise TypeError(f"分析资产不支持 dtype：{array.dtype}")
            frozen = np.ascontiguousarray(array).copy()
            frozen.setflags(write=False)
            frozen_arrays.append((member, frozen))
        if not frozen_arrays:
            raise ValueError("分析资产至少需要一个数组")
        if len({name for name, _array in frozen_arrays}) != len(frozen_arrays):
            raise ValueError("分析资产 NPZ 成员名称不能重复")
        metadata_payload = dict(metadata or {})
        metadata_payload.update(
            {
                "schema": schema_token,
                "allow_pickle": False,
                "members": {
                    name: {
                        "dtype": str(array.dtype),
                        "shape": list(array.shape),
                    }
                    for name, array in frozen_arrays
                },
            }
        )
        metadata_json = json.dumps(
            metadata_payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        object.__setattr__(self, "kind", resolved_kind)
        object.__setattr__(self, "schema", schema_token)
        object.__setattr__(self, "suggested_stem", stem)
        object.__setattr__(self, "arrays", tuple(frozen_arrays))
        object.__setattr__(self, "_metadata_json", metadata_json)

    @property
    def metadata(self) -> dict[str, object]:
        return json.loads(self._metadata_json)

    @property
    def byte_count(self) -> int:
        return sum(int(array.nbytes) for _name, array in self.arrays)

    def array_mapping(self) -> Mapping[str, NDArray[np.generic]]:
        return MappingProxyType(dict(self.arrays))


@dataclass(frozen=True, slots=True)
class ParticleMeasurementCandidate:
    index: int
    exact_area_px: int
    centroid_px: Coordinate
    rings: ImmutableRings


@dataclass(frozen=True, slots=True)
class ParticleConversionPayload:
    candidates: tuple[ParticleMeasurementCandidate, ...]


@dataclass(frozen=True, slots=True)
class MaximaConversionPayload:
    points: tuple[tuple[float, float, float], ...]


ConversionPayload: TypeAlias = ParticleConversionPayload | MaximaConversionPayload


@dataclass(frozen=True, slots=True)
class ImageAnalysisTaskResult:
    tool: AnalysisTool
    request_id: str
    generation: int
    document_id: str
    source_pixel_revision: int
    source_reference: AnalysisObjectReference | None
    calibration_signature: str | None
    parameters: Mapping[str, object] = field(compare=False)
    scalars: Mapping[str, JsonScalar] = field(compare=False)
    tables: tuple[AnalysisTable, ...] = ()
    curves: tuple[AnalysisCurve, ...] = ()
    asset_payloads: tuple[AnalysisAssetPayload, ...] = ()
    conversion_payload: ConversionPayload | None = field(
        default=None,
        compare=False,
        repr=False,
    )
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        parameters_json = json.dumps(
            dict(self.parameters),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        scalars_json = json.dumps(
            dict(self.scalars),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        object.__setattr__(
            self,
            "parameters",
            MappingProxyType(json.loads(parameters_json)),
        )
        object.__setattr__(
            self,
            "scalars",
            MappingProxyType(json.loads(scalars_json)),
        )

    def to_analysis_artifact(
        self,
        *,
        artifact_id: str | None = None,
        asset_references: Iterable[AnalysisAssetReference] = (),
    ) -> AnalysisArtifact:
        references = tuple(asset_references)
        if len(references) != len(self.asset_payloads):
            raise ValueError(
                "分析资产引用数量与待落盘资产数量不一致；"
                "必须先由项目保存层原子写入全部资产"
            )
        return AnalysisArtifact(
            id=artifact_id or f"analysis_{uuid4().hex}",
            source_document_id=self.document_id,
            source_pixel_revision=self.source_pixel_revision,
            source_reference=self.source_reference,
            tool_id=f"fdm.{self.tool.value}",
            tool_version="1",
            parameters=dict(self.parameters),
            calibration_signature=self.calibration_signature,
            scalars=dict(self.scalars),
            tables=self.tables,
            curves=self.curves,
            assets=references,
        )


@dataclass(frozen=True, slots=True)
class AnalysisTaskPhaseUpdate:
    request_id: str
    generation: int
    tool: AnalysisTool
    phase: AnalysisTaskPhase

    @property
    def message(self) -> str:
        return f"{self.tool.chinese_name}：{self.phase.chinese_name}"


@dataclass(frozen=True, slots=True)
class AnalysisResourceEstimate:
    estimated_peak_bytes: int
    allowed: bool
    reason: str = ""


@dataclass(frozen=True, slots=True)
class _TaskCompletion:
    request: ImageAnalysisTaskRequest
    result: ImageAnalysisTaskResult | None = None
    error: str | None = None
    cancelled: bool = False


AnalysisPhaseCallback: TypeAlias = Callable[[AnalysisTaskPhase], None]
AnalysisTaskExecutor: TypeAlias = Callable[
    [ImageAnalysisTaskRequest, CancellationToken, AnalysisPhaseCallback],
    ImageAnalysisTaskResult,
]


class _TaskSignals(QObject):
    completed = Signal(object)
    phase = Signal(object)


class _AnalysisRunnable(QRunnable):
    def __init__(
        self,
        *,
        request: ImageAnalysisTaskRequest,
        token: CancellationToken,
        executor: AnalysisTaskExecutor,
        signals: _TaskSignals,
    ) -> None:
        super().__init__()
        self._request = request
        self._token = token
        self._executor = executor
        self._signals = signals

    @Slot()
    def run(self) -> None:
        def emit_phase(phase: AnalysisTaskPhase) -> None:
            self._signals.phase.emit(
                AnalysisTaskPhaseUpdate(
                    request_id=self._request.request_id,
                    generation=self._request.generation,
                    tool=self._request.tool,
                    phase=phase,
                )
            )

        try:
            self._token.raise_if_cancelled()
            result = self._executor(self._request, self._token, emit_phase)
            self._token.raise_if_cancelled()
            completion = _TaskCompletion(request=self._request, result=result)
        except CancellationError:
            completion = _TaskCompletion(request=self._request, cancelled=True)
        except Exception as exc:
            if self._token.is_cancelled:
                completion = _TaskCompletion(request=self._request, cancelled=True)
            else:
                completion = _TaskCompletion(
                    request=self._request,
                    error=str(exc).strip() or type(exc).__name__,
                )
        self._signals.completed.emit(completion)


class ImageAnalysisTaskController(QObject):
    """Single-worker controller with cancellation and late-result rejection."""

    analysisReady = Signal(object)
    taskFailed = Signal(str, str)
    taskCancelled = Signal(str)
    phaseChanged = Signal(object)
    busyChanged = Signal(bool)
    staleResultDiscarded = Signal(str, int)

    def __init__(
        self,
        *,
        executor: AnalysisTaskExecutor | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._executor = executor or execute_analysis_task
        self._signals = _TaskSignals(self)
        self._signals.completed.connect(self._on_completed)
        self._signals.phase.connect(self._on_phase)
        self._pool = QThreadPool(self)
        self._pool.setMaxThreadCount(1)
        self._pool.setExpiryTimeout(5_000)
        self._generation = 0
        self._active: ImageAnalysisTaskRequest | None = None
        self._pending: ImageAnalysisTaskRequest | None = None
        self._cancellation: CancellationTokenSource | None = None
        self._busy = False
        self._closed = False

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def active_request(self) -> ImageAnalysisTaskRequest | None:
        return self._active

    def is_busy(self) -> bool:
        return self._active is not None or self._pending is not None

    def start(
        self,
        *,
        tool: AnalysisTool | str,
        document_id: str,
        source_pixel_revision: int,
        plane: RasterPlane,
        roi_mask: NDArray[np.bool_] | None = None,
        raw_rings: Iterable[Iterable[Any]] = (),
        exact_area_px: float | None = None,
        source_reference: AnalysisObjectReference | None = None,
        calibration: AnalysisCalibrationSnapshot | None = None,
        parameters: Mapping[str, object] | None = None,
    ) -> ImageAnalysisTaskRequest:
        if self._closed:
            raise RuntimeError("图像分析任务控制器已经关闭")
        self._generation += 1
        request = ImageAnalysisTaskRequest(
            tool=tool,
            request_id=uuid4().hex,
            generation=self._generation,
            document_id=document_id,
            source_pixel_revision=source_pixel_revision,
            plane=plane,
            roi_mask=roi_mask,
            raw_rings=raw_rings,
            exact_area_px=exact_area_px,
            source_reference=source_reference,
            calibration=calibration,
            parameters=parameters,
        )
        if self._active is not None:
            if self._cancellation is not None:
                self._cancellation.cancel()
            self._pending = request
        else:
            self._launch(request)
        return request

    def cancel(self) -> None:
        self._pending = None
        if self._cancellation is not None:
            self._cancellation.cancel()
        if self._active is None:
            self._set_busy(False)

    def close(self) -> None:
        self._closed = True
        self.cancel()

    def wait_for_done(self, timeout_ms: int = 5_000) -> bool:
        return self._pool.waitForDone(timeout_ms)

    def _launch(self, request: ImageAnalysisTaskRequest) -> None:
        cancellation = CancellationTokenSource()
        self._active = request
        self._cancellation = cancellation
        self._set_busy(True)
        self._pool.start(
            _AnalysisRunnable(
                request=request,
                token=cancellation.token,
                executor=self._executor,
                signals=self._signals,
            )
        )

    def _set_busy(self, busy: bool) -> None:
        value = bool(busy)
        if self._busy == value:
            return
        self._busy = value
        self.busyChanged.emit(value)

    @Slot(object)
    def _on_phase(self, update: object) -> None:
        if not isinstance(update, AnalysisTaskPhaseUpdate):
            return
        if (
            self._active is not None
            and self._active.request_id == update.request_id
            and update.generation == self._generation
            and self._pending is None
            and not self._closed
        ):
            self.phaseChanged.emit(update)

    @Slot(object)
    def _on_completed(self, completion: object) -> None:
        if not isinstance(completion, _TaskCompletion):
            return
        request = completion.request
        if self._active is None or self._active.request_id != request.request_id:
            self.staleResultDiscarded.emit(request.request_id, request.generation)
            return
        self._active = None
        self._cancellation = None
        current = (
            not self._closed
            and request.generation == self._generation
            and self._pending is None
        )
        if current:
            if completion.cancelled:
                self.taskCancelled.emit(request.request_id)
            elif completion.error is not None:
                self.taskFailed.emit(request.request_id, completion.error)
            elif completion.result is not None:
                self.analysisReady.emit(completion.result)
        elif not completion.cancelled:
            self.staleResultDiscarded.emit(request.request_id, request.generation)

        pending = self._pending
        self._pending = None
        if pending is not None and not self._closed:
            self._launch(pending)
        else:
            self._set_busy(False)


def execute_analysis_task(
    request: ImageAnalysisTaskRequest,
    token: CancellationToken,
    phase_callback: AnalysisPhaseCallback,
) -> ImageAnalysisTaskResult:
    phase_callback(AnalysisTaskPhase.PREPARING)
    token.raise_if_cancelled()
    estimate = estimate_analysis_resources(request)
    if not estimate.allowed:
        raise MemoryError(estimate.reason)
    prepared = _prepare_kernel_request(request)
    if isinstance(prepared, tuple) and isinstance(
        prepared[1],
        AdvancedAnalysisResourceEstimate,
    ):
        kernel_request, advanced_estimate = prepared
        if not advanced_estimate.allowed:
            raise MemoryError(
                advanced_estimate.reason
                or "高级分析资源估算超过安全上限。"
            )
    else:
        kernel_request = prepared

    phase_callback(AnalysisTaskPhase.ANALYZING)
    token.raise_if_cancelled()
    kernel_result = _execute_kernel(
        request.tool,
        kernel_request,
        cancellation_token=token,
    )
    token.raise_if_cancelled()

    phase_callback(AnalysisTaskPhase.PACKAGING)
    result = _package_kernel_result(request, kernel_result)
    token.raise_if_cancelled()
    return result


def estimate_analysis_resources(
    request: ImageAnalysisTaskRequest,
) -> AnalysisResourceEstimate:
    pixels = request.plane.width * request.plane.height
    points = sum(len(ring) for ring in request.raw_rings)
    multiplier = {
        AnalysisTool.SHAPE: 2,
        AnalysisTool.INTENSITY: 12,
        AnalysisTool.HISTOGRAM: 12,
        AnalysisTool.PROFILE: 10,
        AnalysisTool.PARTICLES: 28,
        AnalysisTool.MAXIMA: 24,
        AnalysisTool.DIRECTIONALITY: 60,
        AnalysisTool.SKELETON: 72,
        AnalysisTool.LOCAL_THICKNESS: 80,
        AnalysisTool.TUBENESS: 96,
        AnalysisTool.GLCM: 32,
        AnalysisTool.SPATIAL_DISTRIBUTION: 2,
        AnalysisTool.SURFACE: 56,
    }[request.tool]
    estimated = request.plane.byte_count + pixels * multiplier + points * 64
    if request.roi_mask is not None:
        estimated += int(request.roi_mask.nbytes)
    if request.tool is AnalysisTool.SPATIAL_DISTRIBUTION:
        raw_points = request.parameters.get("points", ())
        count = len(raw_points) if isinstance(raw_points, Sequence) else 0
        estimated += count * count * 32 + count * 80
    if request.tool is AnalysisTool.GLCM:
        parameters = request.parameters
        levels = int(parameters.get("levels", 32))
        distances = parameters.get("distances", (1,))
        directions = parameters.get("directions_degrees", (0, 45, 90, 135))
        count = (
            len(distances) * len(directions)
            if isinstance(distances, Sequence) and isinstance(directions, Sequence)
            else 4
        )
        estimated += levels * levels * 16 * count
    allowed = estimated <= MAX_ANALYSIS_WORKING_BYTES
    return AnalysisResourceEstimate(
        estimated_peak_bytes=int(estimated),
        allowed=allowed,
        reason=(
            ""
            if allowed
            else (
                f"预计分析工作集 {estimated / (1 << 20):.1f} MiB 超过 "
                f"{MAX_ANALYSIS_WORKING_BYTES / (1 << 20):.0f} MiB 安全上限；"
                "请缩小 ROI、裁剪图片或分区分析。"
            )
        ),
    )


def _prepare_kernel_request(
    request: ImageAnalysisTaskRequest,
) -> object | tuple[object, AdvancedAnalysisResourceEstimate]:
    image = _raster_plane_to_array(request.plane)
    parameters = request.parameters
    calibration = request.calibration
    tool = request.tool

    if tool is AnalysisTool.SHAPE:
        _reject_unknown(parameters, ())
        if not request.raw_rings or len(request.raw_rings[0]) < 3:
            raise ValueError("形状测量需要包含至少三个点的 RAW 外环")
        return ShapeAnalysisRequest(
            rings=request.raw_rings,
            exact_area_px=request.exact_area_px,
            pixel_size_x=calibration.pixel_size_x,
            pixel_size_y=calibration.pixel_size_y,
            unit=calibration.unit,
            request_id=request.request_id,
            generation=request.generation,
        )
    if tool is AnalysisTool.INTENSITY:
        allowed = {"channel", "percentile_levels"}
        _reject_unknown(parameters, allowed)
        return IntensityAnalysisRequest(
            image=image,
            roi_mask=request.roi_mask,
            rings=request.raw_rings,
            channel=str(parameters.get("channel", "luminance")),
            percentile_levels=tuple(parameters.get(
                "percentile_levels",
                (10.0, 25.0, 50.0, 75.0, 90.0),
            )),
            request_id=request.request_id,
            generation=request.generation,
        )
    if tool is AnalysisTool.HISTOGRAM:
        allowed = {"channel", "bins", "value_range"}
        _reject_unknown(parameters, allowed)
        value_range = parameters.get("value_range")
        return HistogramRequest(
            image=image,
            roi_mask=request.roi_mask,
            rings=request.raw_rings,
            channel=str(parameters.get("channel", "luminance")),
            bins=int(parameters.get("bins", 256)),
            value_range=(
                None
                if value_range is None
                else (float(value_range[0]), float(value_range[1]))  # type: ignore[index]
            ),
            request_id=request.request_id,
            generation=request.generation,
        )
    if tool is AnalysisTool.PROFILE:
        allowed = {"points", "line_width", "sample_spacing", "channel"}
        _reject_unknown(parameters, allowed)
        return IntensityProfileRequest(
            image=image,
            points=_freeze_ring(parameters.get("points", ())),
            line_width=float(parameters.get("line_width", 1.0)),
            sample_spacing=float(parameters.get("sample_spacing", 1.0)),
            pixel_size_x=calibration.pixel_size_x,
            pixel_size_y=calibration.pixel_size_y,
            channel=str(parameters.get("channel", "luminance")),
            request_id=request.request_id,
            generation=request.generation,
        )
    if tool is AnalysisTool.PARTICLES:
        allowed = {
            "threshold",
            "foreground",
            "channel",
            "connectivity",
            "min_area_px",
            "max_area_px",
            "min_circularity",
            "max_circularity",
            "include_holes",
            "exclude_edge",
        }
        _reject_unknown(parameters, allowed)
        mask = _binary_input_mask(request, image, parameters)
        return ParticleAnalysisRequest(
            mask=mask,
            connectivity=int(parameters.get("connectivity", 8)),
            min_area_px=int(parameters.get("min_area_px", 1)),
            max_area_px=(
                None
                if parameters.get("max_area_px") is None
                else int(parameters["max_area_px"])
            ),
            min_circularity=float(parameters.get("min_circularity", 0.0)),
            max_circularity=float(parameters.get("max_circularity", 1.0)),
            include_holes=bool(parameters.get("include_holes", False)),
            exclude_edge=bool(parameters.get("exclude_edge", False)),
            pixel_size_x=calibration.pixel_size_x,
            pixel_size_y=calibration.pixel_size_y,
            unit=calibration.unit,
            request_id=request.request_id,
            generation=request.generation,
        )
    if tool is AnalysisTool.MAXIMA:
        allowed = {
            "channel",
            "minimum_value",
            "prominence",
            "neighborhood_radius",
            "min_distance",
            "exclude_edge",
            "max_points",
        }
        _reject_unknown(parameters, allowed)
        return FindMaximaRequest(
            image=image,
            roi_mask=_combined_analysis_mask(request),
            channel=str(parameters.get("channel", "luminance")),
            minimum_value=parameters.get("minimum_value"),  # type: ignore[arg-type]
            prominence=float(parameters.get("prominence", 0.0)),
            neighborhood_radius=int(parameters.get("neighborhood_radius", 1)),
            min_distance=float(parameters.get("min_distance", 1.0)),
            exclude_edge=bool(parameters.get("exclude_edge", False)),
            max_points=(
                None
                if parameters.get("max_points") is None
                else int(parameters["max_points"])
            ),
            request_id=request.request_id,
            generation=request.generation,
        )

    scalar = _select_scalar_channel(
        image,
        str(parameters.get("channel", "luminance")),
    )
    mask = _combined_analysis_mask(request)
    if tool is AnalysisTool.DIRECTIONALITY:
        allowed = {
            "channel",
            "bins",
            "gradient_sigma",
            "minimum_gradient",
            "histogram_smoothing_bins",
            "peak_min_fraction",
            "max_peaks",
        }
        _reject_unknown(parameters, allowed)
        kernel_request = DirectionalityRequest(
            image=scalar,
            roi_mask=mask,
            bins=int(parameters.get("bins", 180)),
            gradient_sigma=float(parameters.get("gradient_sigma", 1.0)),
            minimum_gradient=float(parameters.get("minimum_gradient", 0.0)),
            histogram_smoothing_bins=float(
                parameters.get("histogram_smoothing_bins", 1.0)
            ),
            peak_min_fraction=float(parameters.get("peak_min_fraction", 0.1)),
            max_peaks=int(parameters.get("max_peaks", 8)),
            request_id=request.request_id,
            generation=request.generation,
        )
    elif tool is AnalysisTool.SKELETON:
        allowed = {"threshold", "foreground", "channel", "already_skeletonized"}
        _reject_unknown(parameters, allowed)
        kernel_request = SkeletonNetworkRequest(
            mask=_binary_input_mask(request, image, parameters),
            already_skeletonized=bool(
                parameters.get("already_skeletonized", False)
            ),
            pixel_size_x=calibration.pixel_size_x,
            pixel_size_y=calibration.pixel_size_y,
            unit=calibration.unit,
            request_id=request.request_id,
            generation=request.generation,
        )
    elif tool is AnalysisTool.LOCAL_THICKNESS:
        allowed = {"threshold", "foreground", "channel"}
        _reject_unknown(parameters, allowed)
        kernel_request = LocalThicknessRequest(
            mask=_binary_input_mask(request, image, parameters),
            request_id=request.request_id,
            generation=request.generation,
        )
    elif tool is AnalysisTool.TUBENESS:
        allowed = {
            "channel",
            "scales",
            "beta",
            "structure_scale",
            "bright_ridges",
        }
        _reject_unknown(parameters, allowed)
        kernel_request = TubenessRequest(
            image=scalar,
            roi_mask=mask,
            scales=tuple(parameters.get("scales", (1.0, 2.0, 4.0))),
            beta=float(parameters.get("beta", 0.5)),
            structure_scale=parameters.get("structure_scale"),  # type: ignore[arg-type]
            bright_ridges=bool(parameters.get("bright_ridges", True)),
            request_id=request.request_id,
            generation=request.generation,
        )
    elif tool is AnalysisTool.GLCM:
        allowed = {
            "channel",
            "levels",
            "distances",
            "directions_degrees",
            "value_range",
            "symmetric",
        }
        _reject_unknown(parameters, allowed)
        value_range = parameters.get("value_range")
        kernel_request = GlcmHaralickRequest(
            image=scalar,
            roi_mask=mask,
            levels=int(parameters.get("levels", 32)),
            distances=tuple(parameters.get("distances", (1,))),
            directions_degrees=tuple(
                parameters.get("directions_degrees", (0.0, 45.0, 90.0, 135.0))
            ),
            value_range=(
                None
                if value_range is None
                else (float(value_range[0]), float(value_range[1]))  # type: ignore[index]
            ),
            symmetric=bool(parameters.get("symmetric", True)),
            request_id=request.request_id,
            generation=request.generation,
        )
    elif tool is AnalysisTool.SPATIAL_DISTRIBUTION:
        allowed = {"points", "study_area"}
        _reject_unknown(parameters, allowed)
        kernel_request = SpatialDistributionRequest(
            points=_freeze_ring(parameters.get("points", ())),
            study_area=parameters.get("study_area"),  # type: ignore[arg-type]
            pixel_size_x=calibration.pixel_size_x,
            pixel_size_y=calibration.pixel_size_y,
            unit=calibration.unit,
            request_id=request.request_id,
            generation=request.generation,
        )
    elif tool is AnalysisTool.SURFACE:
        allowed = {"channel", "sample_step_x", "sample_step_y"}
        _reject_unknown(parameters, allowed)
        kernel_request = IntensitySurfaceRequest(
            image=scalar,
            roi_mask=mask,
            sample_step_x=int(parameters.get("sample_step_x", 1)),
            sample_step_y=int(parameters.get("sample_step_y", 1)),
            pixel_size_x=calibration.pixel_size_x,
            pixel_size_y=calibration.pixel_size_y,
            unit=calibration.unit,
            request_id=request.request_id,
            generation=request.generation,
        )
    else:  # pragma: no cover - exhaustive enum guard
        raise ValueError(f"不支持的分析工具：{tool.value}")
    return (
        kernel_request,
        estimate_advanced_analysis_resources(
            kernel_request,
            limits=DEFAULT_ADVANCED_ANALYSIS_LIMITS,
        ),
    )


def _execute_kernel(
    tool: AnalysisTool,
    kernel_request: object,
    *,
    cancellation_token: CancellationToken,
) -> object:
    if tool is AnalysisTool.SHAPE:
        return analyze_shape(kernel_request)  # type: ignore[arg-type]
    if tool is AnalysisTool.INTENSITY:
        return analyze_intensity(kernel_request)  # type: ignore[arg-type]
    if tool is AnalysisTool.HISTOGRAM:
        return calculate_histogram(kernel_request)  # type: ignore[arg-type]
    if tool is AnalysisTool.PROFILE:
        return sample_intensity_profile(kernel_request)  # type: ignore[arg-type]
    if tool is AnalysisTool.PARTICLES:
        return analyze_particles(kernel_request)  # type: ignore[arg-type]
    if tool is AnalysisTool.MAXIMA:
        return find_local_maxima(kernel_request)  # type: ignore[arg-type]
    if tool is AnalysisTool.DIRECTIONALITY:
        return analyze_fiber_directionality(
            kernel_request,  # type: ignore[arg-type]
            cancellation_token=cancellation_token,
        )
    if tool is AnalysisTool.SKELETON:
        return analyze_skeleton_network(
            kernel_request,  # type: ignore[arg-type]
            cancellation_token=cancellation_token,
        )
    if tool is AnalysisTool.LOCAL_THICKNESS:
        return calculate_local_thickness(
            kernel_request,  # type: ignore[arg-type]
            cancellation_token=cancellation_token,
        )
    if tool is AnalysisTool.TUBENESS:
        return calculate_multiscale_tubeness(
            kernel_request,  # type: ignore[arg-type]
            cancellation_token=cancellation_token,
        )
    if tool is AnalysisTool.GLCM:
        return calculate_glcm_haralick(
            kernel_request,  # type: ignore[arg-type]
            cancellation_token=cancellation_token,
        )
    if tool is AnalysisTool.SPATIAL_DISTRIBUTION:
        return analyze_spatial_distribution(
            kernel_request,  # type: ignore[arg-type]
            cancellation_token=cancellation_token,
        )
    if tool is AnalysisTool.SURFACE:
        return build_intensity_surface(
            kernel_request,  # type: ignore[arg-type]
            cancellation_token=cancellation_token,
        )
    raise ValueError(f"不支持的分析工具：{tool.value}")


def _package_kernel_result(
    request: ImageAnalysisTaskRequest,
    result: object,
) -> ImageAnalysisTaskResult:
    tool = request.tool
    scalars: dict[str, JsonScalar] = {}
    tables: list[AnalysisTable] = []
    curves: list[AnalysisCurve] = []
    assets: list[AnalysisAssetPayload] = []
    conversion: ConversionPayload | None = None
    warnings: tuple[str, ...] = ()

    if tool is AnalysisTool.SHAPE:
        scalars = {
            "area_px": result.area_px,
            "vector_area_px": result.vector_area_px,
            "area": result.area,
            "outer_perimeter_px": result.outer_perimeter_px,
            "hole_perimeter_px": result.hole_perimeter_px,
            "total_perimeter_px": result.total_perimeter_px,
            "outer_perimeter": result.outer_perimeter,
            "hole_perimeter": result.hole_perimeter,
            "total_perimeter": result.total_perimeter,
            "hole_count": result.hole_count,
            "hole_area_px": result.hole_area_px,
            "equivalent_circle_diameter": result.equivalent_circle_diameter,
            "feret_max": result.feret_max,
            "feret_min": result.feret_min,
            "feret_angle_degrees": result.feret_angle_degrees,
            "ellipse_major": result.ellipse_major,
            "ellipse_minor": result.ellipse_minor,
            "ellipse_angle_degrees": result.ellipse_angle_degrees,
            "circularity": result.circularity,
            "aspect_ratio": result.aspect_ratio,
            "roundness": result.roundness,
            "solidity": result.solidity,
            "unit": result.unit,
            "area_from_exact_mask": result.area_from_exact_mask,
        }
        tables.append(
            AnalysisTable(
                name="位置与边界",
                columns=("项目", "X", "Y", "宽", "高"),
                rows=(
                    (
                        "质心",
                        result.centroid_px[0],
                        result.centroid_px[1],
                        None,
                        None,
                    ),
                    (
                        "边界框",
                        result.bounds_px[0],
                        result.bounds_px[1],
                        result.bounds_px[2] - result.bounds_px[0],
                        result.bounds_px[3] - result.bounds_px[1],
                    ),
                ),
            )
        )
        warnings = tuple(result.warnings)
    elif tool is AnalysisTool.INTENSITY:
        scalars = {
            "included_pixel_count": result.included_pixel_count,
            "valid_pixel_count": result.valid_pixel_count,
            "non_finite_count": result.non_finite_count,
            "mean": result.mean,
            "median": result.median,
            "stddev": result.stddev,
            "minimum": result.minimum,
            "maximum": result.maximum,
            "integrated_density": result.integrated_density,
            "channel": result.channel,
        }
        if result.intensity_centroid_px is not None:
            scalars["intensity_centroid_x_px"] = result.intensity_centroid_px[0]
            scalars["intensity_centroid_y_px"] = result.intensity_centroid_px[1]
        tables.append(
            AnalysisTable(
                name="分位数",
                columns=("分位数(%)", "数值"),
                rows=tuple(result.percentiles),
            )
        )
    elif tool is AnalysisTool.HISTOGRAM:
        centers = tuple(
            (result.edges[index] + result.edges[index + 1]) / 2.0
            for index in range(len(result.counts))
        )
        scalars = {
            "included_pixel_count": result.included_pixel_count,
            "non_finite_count": result.non_finite_count,
            "channel": result.channel,
            "bins": len(result.counts),
        }
        curves.append(
            AnalysisCurve(
                name="直方图",
                x=centers,
                y=tuple(float(value) for value in result.counts),
                x_unit="强度",
                y_unit="频数",
            )
        )
    elif tool is AnalysisTool.PROFILE:
        scalars = {
            "valid_sample_count": result.valid_sample_count,
            "sample_count": len(result.values),
            "channel": result.channel,
        }
        if len(result.values) <= _INLINE_CURVE_POINTS:
            curves.append(
                AnalysisCurve(
                    name="强度剖面",
                    x=result.distances,
                    y=result.values,
                    x_unit=request.calibration.unit,
                    y_unit="强度",
                )
            )
        else:
            assets.append(
                AnalysisAssetPayload(
                    kind=AnalysisAssetKind.CURVE,
                    schema="fdm.intensity-profile.v1",
                    suggested_stem="intensity-profile",
                    arrays={
                        "distance_px": np.asarray(result.distances_px),
                        "distance": np.asarray(result.distances),
                        "value": np.asarray(
                            [
                                np.nan if value is None else value
                                for value in result.values
                            ],
                            dtype=np.float64,
                        ),
                    },
                )
            )
    elif tool is AnalysisTool.PARTICLES:
        scalars = {
            "total_component_count": result.total_component_count,
            "accepted_count": result.accepted_count,
            "rejected_by_area_count": result.rejected_by_area_count,
            "rejected_by_circularity_count": result.rejected_by_circularity_count,
            "rejected_edge_count": result.rejected_edge_count,
            "foreground_pixel_count": result.foreground_pixel_count,
            "include_holes": result.include_holes,
            "connectivity": result.connectivity,
        }
        rows = tuple(
            (
                particle.index,
                particle.exact_area_px,
                particle.area,
                particle.centroid_px[0],
                particle.centroid_px[1],
                particle.perimeter_px,
                particle.circularity,
                particle.hole_count,
                particle.touches_edge,
            )
            for particle in result.particles
        )
        if len(rows) <= _INLINE_DETAIL_ROWS:
            tables.append(
                AnalysisTable(
                    name="粒子明细",
                    columns=(
                        "序号",
                        "精确面积(px²)",
                        "面积",
                        "质心X(px)",
                        "质心Y(px)",
                        "周长(px)",
                        "圆度",
                        "孔洞数",
                        "接触边缘",
                    ),
                    rows=rows,
                )
            )
        else:
            assets.append(
                AnalysisAssetPayload(
                    kind=AnalysisAssetKind.TABLE,
                    schema="fdm.particle-table.v1",
                    suggested_stem="particles",
                    arrays={
                        "values": np.asarray(rows, dtype=np.float64),
                    },
                    metadata={
                        "columns": [
                            "index",
                            "exact_area_px",
                            "area",
                            "centroid_x_px",
                            "centroid_y_px",
                            "perimeter_px",
                            "circularity",
                            "hole_count",
                            "touches_edge",
                        ]
                    },
                )
            )
        conversion = ParticleConversionPayload(
            candidates=tuple(
                ParticleMeasurementCandidate(
                    index=particle.index,
                    exact_area_px=particle.exact_area_px,
                    centroid_px=particle.centroid_px,
                    rings=particle.rings,
                )
                for particle in result.particles
            )
        )
    elif tool is AnalysisTool.MAXIMA:
        scalars = {
            "accepted_count": len(result.maxima),
            "candidate_plateau_count": result.candidate_plateau_count,
            "suppressed_count": result.suppressed_count,
            "channel": result.channel,
        }
        rows = tuple(
            (
                index,
                maximum.x,
                maximum.y,
                maximum.value,
                maximum.local_prominence,
            )
            for index, maximum in enumerate(result.maxima, start=1)
        )
        if len(rows) <= _INLINE_DETAIL_ROWS:
            tables.append(
                AnalysisTable(
                    name="极值点",
                    columns=("序号", "X(px)", "Y(px)", "强度", "局部 prominence"),
                    rows=rows,
                )
            )
        else:
            assets.append(
                AnalysisAssetPayload(
                    kind=AnalysisAssetKind.TABLE,
                    schema="fdm.maxima-table.v1",
                    suggested_stem="maxima",
                    arrays={"values": np.asarray(rows, dtype=np.float64)},
                    metadata={
                        "columns": [
                            "index",
                            "x_px",
                            "y_px",
                            "value",
                            "local_prominence",
                        ]
                    },
                )
            )
        conversion = MaximaConversionPayload(
            points=tuple(
                (maximum.x, maximum.y, maximum.value)
                for maximum in result.maxima
            )
        )
    elif tool is AnalysisTool.DIRECTIONALITY:
        scalars = {
            "valid_gradient_pixels": result.valid_gradient_pixels,
            "total_weight": result.total_weight,
            "convention": result.convention,
            "peak_count": len(result.peaks),
        }
        curves.append(
            AnalysisCurve(
                name="轴向方向分布",
                x=result.bin_centers_degrees,
                y=result.normalized_weights,
                x_unit="°",
                y_unit="归一化权重",
            )
        )
        tables.append(
            AnalysisTable(
                name="方向峰",
                columns=("角度(°)", "权重", "相对权重", "区间索引"),
                rows=tuple(
                    (
                        peak.angle_degrees,
                        peak.weight,
                        peak.relative_weight,
                        peak.bin_index,
                    )
                    for peak in result.peaks
                ),
            )
        )
    elif tool is AnalysisTool.SKELETON:
        scalars = {
            "endpoint_count": result.endpoint_count,
            "branchpoint_count": result.branchpoint_count,
            "connected_component_count": result.connected_component_count,
            "isolated_point_count": result.isolated_point_count,
            "loop_count": result.loop_count,
            "total_length": result.total_length,
            "maximum_geodesic_distance": result.maximum_geodesic_distance,
            "unit": result.unit,
        }
        branch_rows = tuple(
            (
                index,
                None if branch.start_px is None else branch.start_px[0],
                None if branch.start_px is None else branch.start_px[1],
                None if branch.end_px is None else branch.end_px[0],
                None if branch.end_px is None else branch.end_px[1],
                branch.length,
                branch.closed,
            )
            for index, branch in enumerate(result.branches, start=1)
        )
        if len(branch_rows) <= _INLINE_DETAIL_ROWS:
            tables.append(
                AnalysisTable(
                    name="骨架分支",
                    columns=(
                        "序号",
                        "起点X",
                        "起点Y",
                        "终点X",
                        "终点Y",
                        "长度",
                        "闭环",
                    ),
                    rows=branch_rows,
                )
            )
        assets.append(
            AnalysisAssetPayload(
                kind=AnalysisAssetKind.GRAPH,
                schema="fdm.skeleton-network.v1",
                suggested_stem="skeleton-network",
                arrays={
                    "skeleton": result.skeleton.astype(np.uint8),
                    "endpoints_xy": np.asarray(
                        result.endpoint_coordinates_px,
                        dtype=np.float64,
                    ).reshape((-1, 2)),
                    "branchpoints_xy": np.asarray(
                        result.branchpoint_coordinates_px,
                        dtype=np.float64,
                    ).reshape((-1, 2)),
                    "branches": (
                        np.asarray(
                            [
                                [
                                    row[0],
                                    np.nan if row[1] is None else row[1],
                                    np.nan if row[2] is None else row[2],
                                    np.nan if row[3] is None else row[3],
                                    np.nan if row[4] is None else row[4],
                                    row[5],
                                    float(row[6]),
                                ]
                                for row in branch_rows
                            ],
                            dtype=np.float64,
                        ).reshape((-1, 7))
                    ),
                },
                metadata={
                    "unit": result.unit,
                    "branch_columns": [
                        "index",
                        "start_x_px",
                        "start_y_px",
                        "end_x_px",
                        "end_y_px",
                        "length",
                        "closed",
                    ],
                },
            )
        )
    elif tool is AnalysisTool.LOCAL_THICKNESS:
        scalars = {
            "foreground_pixel_count": result.foreground_pixel_count,
            "maximum_thickness_px": result.maximum_thickness_px,
            "mean_thickness_px": result.mean_thickness_px,
            "definition": result.definition,
        }
        circle_rows = tuple(
            (
                index,
                circle.center_x,
                circle.center_y,
                circle.radius_px,
            )
            for index, circle in enumerate(result.maximal_circles, start=1)
        )
        if len(circle_rows) <= _INLINE_DETAIL_ROWS:
            tables.append(
                AnalysisTable(
                    name="最大内切圆",
                    columns=("序号", "中心X(px)", "中心Y(px)", "半径(px)"),
                    rows=circle_rows,
                )
            )
        assets.append(
            AnalysisAssetPayload(
                kind=AnalysisAssetKind.OTHER,
                schema="fdm.local-thickness.v1",
                suggested_stem="local-thickness",
                arrays={
                    "thickness_px": result.thickness_px,
                    "maximal_circles": np.asarray(
                        circle_rows,
                        dtype=np.float64,
                    ).reshape((-1, 4)),
                },
                metadata={
                    "circle_columns": [
                        "index",
                        "center_x_px",
                        "center_y_px",
                        "radius_px",
                    ]
                },
            )
        )
    elif tool is AnalysisTool.TUBENESS:
        scalars = {
            "maximum_response": result.maximum_response,
            "scale_count": len(result.scales),
        }
        assets.append(
            AnalysisAssetPayload(
                kind=AnalysisAssetKind.OTHER,
                schema="fdm.tubeness.v1",
                suggested_stem="tubeness",
                arrays={
                    "response": result.response,
                    "best_scale": result.best_scale,
                    "scales": np.asarray(result.scales, dtype=np.float64),
                },
            )
        )
    elif tool is AnalysisTool.GLCM:
        scalars = {
            "levels": result.levels,
            "quantization_minimum": result.quantization_range[0],
            "quantization_maximum": result.quantization_range[1],
            "symmetric": result.symmetric,
            "valid_pixel_count": result.valid_pixel_count,
            "non_finite_pixel_count": result.non_finite_pixel_count,
        }
        tables.append(
            AnalysisTable(
                name="Haralick 特征",
                columns=(
                    "距离(px)",
                    "方向(°)",
                    "像素对数",
                    "Contrast",
                    "Dissimilarity",
                    "Homogeneity",
                    "ASM",
                    "Energy",
                    "Correlation",
                    "Entropy",
                    "Maximum Probability",
                ),
                rows=tuple(
                    (
                        item.distance_px,
                        item.direction_degrees,
                        item.pair_count,
                        item.contrast,
                        item.dissimilarity,
                        item.homogeneity,
                        item.angular_second_moment,
                        item.energy,
                        item.correlation,
                        item.entropy,
                        item.maximum_probability,
                    )
                    for item in result.features
                ),
            )
        )
        assets.append(
            AnalysisAssetPayload(
                kind=AnalysisAssetKind.TABLE,
                schema="fdm.glcm-matrices.v1",
                suggested_stem="glcm-matrices",
                arrays={
                    "matrices": np.stack(
                        [item.matrix for item in result.features],
                        axis=0,
                    )
                    if result.features
                    else np.zeros((0, result.levels, result.levels), dtype=np.float64)
                },
            )
        )
    elif tool is AnalysisTool.SPATIAL_DISTRIBUTION:
        scalars = {
            "point_count": len(result.nearest_neighbor_distances),
            "mean_nearest_neighbor_distance": result.mean_nearest_neighbor_distance,
            "median_nearest_neighbor_distance": result.median_nearest_neighbor_distance,
            "minimum_nearest_neighbor_distance": result.minimum_nearest_neighbor_distance,
            "maximum_nearest_neighbor_distance": result.maximum_nearest_neighbor_distance,
            "study_area": result.study_area,
            "area_source": result.area_source,
            "spatial_density": result.spatial_density,
            "unit": result.unit,
        }
        tables.append(
            AnalysisTable(
                name="最近邻明细",
                columns=("点序号", "最近邻序号", "距离"),
                rows=tuple(
                    (index + 1, neighbor + 1, distance)
                    for index, (neighbor, distance) in enumerate(
                        zip(
                            result.nearest_neighbor_indices,
                            result.nearest_neighbor_distances,
                            strict=True,
                        )
                    )
                ),
            )
        )
    elif tool is AnalysisTool.SURFACE:
        scalars = {
            "finite_sample_count": result.finite_sample_count,
            "masked_sample_count": result.masked_sample_count,
            "non_finite_sample_count": result.non_finite_sample_count,
            "z_minimum": result.z_minimum,
            "z_maximum": result.z_maximum,
            "coordinate_unit": result.coordinate_unit,
            "intensity_unit": result.intensity_unit,
        }
        z = np.asarray(
            [
                [np.nan if value is None else value for value in row]
                for row in result.z_values
            ],
            dtype=np.float64,
        )
        assets.append(
            AnalysisAssetPayload(
                kind=AnalysisAssetKind.OTHER,
                schema="fdm.intensity-surface.v1",
                suggested_stem="intensity-surface",
                arrays={
                    "x": np.asarray(result.x_coordinates, dtype=np.float64),
                    "y": np.asarray(result.y_coordinates, dtype=np.float64),
                    "z": z,
                },
                metadata={
                    "coordinate_unit": result.coordinate_unit,
                    "intensity_unit": result.intensity_unit,
                },
            )
        )
    else:  # pragma: no cover - exhaustive enum guard
        raise ValueError(f"不支持的分析工具：{tool.value}")

    return ImageAnalysisTaskResult(
        tool=tool,
        request_id=request.request_id,
        generation=request.generation,
        document_id=request.document_id,
        source_pixel_revision=request.source_pixel_revision,
        source_reference=request.source_reference,
        calibration_signature=request.calibration.signature,
        parameters=request.parameters,
        scalars=scalars,
        tables=tuple(tables),
        curves=tuple(curves),
        asset_payloads=tuple(assets),
        conversion_payload=conversion,
        warnings=warnings,
    )


def _raster_plane_to_array(plane: RasterPlane) -> NDArray[np.generic]:
    dtype: np.dtype[Any]
    channels = plane.pixel_type.channel_count
    if plane.pixel_type is RasterPixelType.GRAY8:
        dtype = np.dtype(np.uint8)
    elif plane.pixel_type is RasterPixelType.GRAY16:
        dtype = np.dtype("<u2")
    elif plane.pixel_type is RasterPixelType.GRAY32_FLOAT:
        dtype = np.dtype("<f4")
    else:
        dtype = np.dtype(np.uint8)
    array = np.frombuffer(plane.data, dtype=dtype)
    if channels == 1:
        return array.reshape((plane.height, plane.width))
    return array.reshape((plane.height, plane.width, channels))


def _select_scalar_channel(
    image: NDArray[Any],
    channel: str,
) -> NDArray[np.generic]:
    if image.ndim == 2:
        return image
    if image.shape[2] == 1:
        return image[..., 0]
    token = str(channel).strip().lower()
    index = {
        "red": 0,
        "r": 0,
        "green": 1,
        "g": 1,
        "blue": 2,
        "b": 2,
    }.get(token)
    if index is not None:
        return image[..., index]
    if token in {"luminance", "gray", "grayscale"}:
        rgb = image[..., :3].astype(np.float64)
        return (
            rgb[..., 0] * 0.2126
            + rgb[..., 1] * 0.7152
            + rgb[..., 2] * 0.0722
        ).astype(np.float32)
    raise ValueError(f"不支持的分析通道：{channel}")


def _combined_analysis_mask(
    request: ImageAnalysisTaskRequest,
) -> NDArray[np.bool_] | None:
    mask: NDArray[np.bool_] | None = None
    if request.raw_rings:
        mask = _rings_to_odd_even_mask(
            (request.plane.height, request.plane.width),
            request.raw_rings,
        )
    if request.roi_mask is not None:
        mask = (
            np.asarray(request.roi_mask, dtype=bool).copy()
            if mask is None
            else mask & request.roi_mask
        )
    if mask is not None:
        mask = np.ascontiguousarray(mask, dtype=bool)
        mask.setflags(write=False)
    return mask


def _binary_input_mask(
    request: ImageAnalysisTaskRequest,
    image: NDArray[Any],
    parameters: Mapping[str, object],
) -> NDArray[np.bool_]:
    channel = str(parameters.get("channel", "luminance"))
    scalar = _select_scalar_channel(image, channel)
    selection = _combined_analysis_mask(request)
    threshold = parameters.get("threshold")
    if threshold is None:
        if selection is None:
            raise ValueError("该分析需要二值掩膜、ROI/面积对象或显式阈值")
        result = np.asarray(selection, dtype=bool).copy()
    else:
        value = float(threshold)
        if not math.isfinite(value):
            raise ValueError("二值阈值必须是有限数")
        foreground = str(parameters.get("foreground", "above")).strip().lower()
        if foreground == "above":
            result = np.asarray(scalar >= value, dtype=bool)
        elif foreground == "below":
            result = np.asarray(scalar <= value, dtype=bool)
        else:
            raise ValueError("foreground 只能是 above 或 below")
        if selection is not None:
            result &= selection
    result = np.ascontiguousarray(result, dtype=bool)
    result.setflags(write=False)
    return result


def _rings_to_odd_even_mask(
    shape: tuple[int, int],
    rings: ImmutableRings,
) -> NDArray[np.bool_]:
    mask = np.zeros(shape, dtype=bool)
    for ring in rings:
        if len(ring) < 3:
            continue
        contour = np.rint(np.asarray(ring, dtype=np.float64)).astype(np.int32)
        temporary = np.zeros(shape, dtype=np.uint8)
        cv2.fillPoly(temporary, [contour], 1)
        mask ^= temporary.astype(bool)
    return mask


def _freeze_optional_mask(
    mask: NDArray[np.bool_] | None,
    shape: tuple[int, int],
) -> NDArray[np.bool_] | None:
    if mask is None:
        return None
    normalized = np.asarray(mask, dtype=bool)
    if normalized.shape != shape:
        raise ValueError(f"ROI 掩膜尺寸 {normalized.shape!r} 与图像尺寸 {shape!r} 不一致")
    frozen = np.ascontiguousarray(normalized).copy()
    frozen.setflags(write=False)
    return frozen


def _freeze_rings(rings: Iterable[Iterable[Any]]) -> ImmutableRings:
    return tuple(_freeze_ring(ring) for ring in rings)


def _freeze_ring(ring: Iterable[Any]) -> ImmutableRing:
    result: list[Coordinate] = []
    for point in ring:
        if hasattr(point, "x") and hasattr(point, "y"):
            x = float(point.x)
            y = float(point.y)
        else:
            x = float(point[0])
            y = float(point[1])
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("RAW 几何坐标必须是有限数")
        result.append((x, y))
    return tuple(result)


def _reject_unknown(
    parameters: Mapping[str, object],
    allowed: Iterable[str],
) -> None:
    unknown = sorted(set(parameters) - set(allowed))
    if unknown:
        raise ValueError(f"分析参数包含未知字段：{', '.join(unknown)}")


def _positive_finite(value: object, *, field_name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{field_name} 必须是正有限数")
    return number


def _non_negative_finite(value: object, *, field_name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{field_name} 必须是非负有限数")
    return number


def _non_negative_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{field_name} 必须是非负整数")
    number = int(value)
    if number != value or number < 0:
        raise ValueError(f"{field_name} 必须是非负整数")
    return number
