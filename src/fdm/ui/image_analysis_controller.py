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
from pathlib import Path
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
    AnalysisDependencySignature,
    AnalysisObjectReference,
    AnalysisRegionSnapshot,
    AnalysisSourceDescriptor,
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
from fdm.services.analysis_asset_io import (
    ANALYSIS_NPZ_MANIFEST_MEMBER,
    validate_analysis_asset_reference,
)
from fdm.services.analysis_profiles import (
    ANALYSIS_OUTPUT_FIELDS_PARAMETER,
    analysis_output_field_schema,
    normalize_analysis_output_fields,
)
from fdm.services.image_analysis import (
    FftPowerSpectrumRequest,
    FindMaximaRequest,
    HistogramRequest,
    IntensityAnalysisRequest,
    IntensityProfileRequest,
    ParticleAnalysisRequest,
    ShapeAnalysisRequest,
    analyze_intensity,
    analyze_particles,
    analyze_shape,
    calculate_fft_power_spectrum,
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
    FFT_POWER_SPECTRUM = "fft_power_spectrum"
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
            AnalysisTool.FFT_POWER_SPECTRUM: "FFT 功率谱",
            AnalysisTool.PROFILE: "强度剖面",
            AnalysisTool.PARTICLES: "粒子分析",
            AnalysisTool.MAXIMA: "极值检测",
            AnalysisTool.DIRECTIONALITY: "纤维方向性",
            AnalysisTool.SKELETON: "骨架网络",
            AnalysisTool.LOCAL_THICKNESS: "局部厚度",
            AnalysisTool.TUBENESS: "Tubeness",
            AnalysisTool.GLCM: "Haralick GLCM 纹理",
            AnalysisTool.SPATIAL_DISTRIBUTION: "空间分布（最近邻 / Ripley K/L）",
            AnalysisTool.SURFACE: "二维强度表面",
        }[self]


_TOOL_VERSIONS: Mapping[AnalysisTool, str] = MappingProxyType(
    {
        AnalysisTool.SHAPE: "2",
        AnalysisTool.INTENSITY: "2",
        AnalysisTool.HISTOGRAM: "2",
        AnalysisTool.FFT_POWER_SPECTRUM: "1",
        AnalysisTool.PROFILE: "2",
        AnalysisTool.PARTICLES: "2",
        AnalysisTool.MAXIMA: "1",
        AnalysisTool.DIRECTIONALITY: "2",
        AnalysisTool.SKELETON: "2",
        AnalysisTool.LOCAL_THICKNESS: "2",
        AnalysisTool.TUBENESS: "1",
        AnalysisTool.GLCM: "2",
        AnalysisTool.SPATIAL_DISTRIBUTION: "1",
        AnalysisTool.SURFACE: "1",
    }
)


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
    viewport_origin: tuple[int, int] = (0, 0)
    source_reference: AnalysisObjectReference | None = None
    region_snapshot: AnalysisRegionSnapshot | None = None
    source_descriptor: AnalysisSourceDescriptor | None = None
    dependency_signature: AnalysisDependencySignature | None = None
    calibration: AnalysisCalibrationSnapshot = field(
        default_factory=AnalysisCalibrationSnapshot,
    )
    output_fields: tuple[str, ...] | None = None
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
        viewport_origin: Sequence[int] = (0, 0),
        source_reference: AnalysisObjectReference | None = None,
        region_snapshot: AnalysisRegionSnapshot | None = None,
        source_descriptor: AnalysisSourceDescriptor | None = None,
        dependency_signature: AnalysisDependencySignature | None = None,
        calibration: AnalysisCalibrationSnapshot | None = None,
        parameters: Mapping[str, object] | None = None,
        output_fields: Iterable[str] | None = None,
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
        if (
            isinstance(viewport_origin, (str, bytes))
            or len(viewport_origin) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in viewport_origin
            )
        ):
            raise TypeError("viewport_origin 必须是两个整数")
        normalized_origin = (int(viewport_origin[0]), int(viewport_origin[1]))
        if source_reference is not None and not isinstance(
            source_reference,
            AnalysisObjectReference,
        ):
            raise TypeError("source_reference 必须是 AnalysisObjectReference")
        if region_snapshot is not None and not isinstance(
            region_snapshot,
            AnalysisRegionSnapshot,
        ):
            raise TypeError("region_snapshot 必须是 AnalysisRegionSnapshot")
        if source_descriptor is not None and not isinstance(
            source_descriptor,
            AnalysisSourceDescriptor,
        ):
            raise TypeError("source_descriptor 必须是 AnalysisSourceDescriptor")
        if dependency_signature is not None and not isinstance(
            dependency_signature,
            AnalysisDependencySignature,
        ):
            raise TypeError(
                "dependency_signature 必须是 AnalysisDependencySignature"
            )
        parameter_payload = dict(parameters or {})
        embedded_output_fields = parameter_payload.pop(
            ANALYSIS_OUTPUT_FIELDS_PARAMETER,
            None,
        )
        normalized_explicit_output_fields = normalize_analysis_output_fields(
            f"fdm.{resolved_tool.value}",
            output_fields,
            legacy_defaults=True,
        )
        normalized_embedded_output_fields = normalize_analysis_output_fields(
            f"fdm.{resolved_tool.value}",
            embedded_output_fields,  # type: ignore[arg-type]
            legacy_defaults=True,
        )
        if (
            output_fields is not None
            and embedded_output_fields is not None
            and normalized_explicit_output_fields
            != normalized_embedded_output_fields
        ):
            raise ValueError(
                "parameters 中保存的输出字段选择与显式 output_fields 不一致"
            )
        normalized_output_fields = (
            normalized_explicit_output_fields
            if output_fields is not None
            else normalized_embedded_output_fields
        )
        parameters_json = json.dumps(
            parameter_payload,
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
        object.__setattr__(self, "viewport_origin", normalized_origin)
        object.__setattr__(self, "source_reference", source_reference)
        object.__setattr__(self, "region_snapshot", region_snapshot)
        object.__setattr__(self, "source_descriptor", source_descriptor)
        object.__setattr__(self, "dependency_signature", dependency_signature)
        object.__setattr__(
            self,
            "calibration",
            calibration or AnalysisCalibrationSnapshot(),
        )
        object.__setattr__(self, "output_fields", normalized_output_fields)
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
    viewport_origin: tuple[int, int] = (0, 0)


@dataclass(frozen=True, slots=True)
class MaximaConversionPayload:
    points: tuple[tuple[float, float, float], ...]
    viewport_origin: tuple[int, int] = (0, 0)


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
    region_snapshot: AnalysisRegionSnapshot | None = None
    source_descriptor: AnalysisSourceDescriptor | None = None
    dependency_signature: AnalysisDependencySignature | None = None
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
        if self.region_snapshot is not None and not isinstance(
            self.region_snapshot,
            AnalysisRegionSnapshot,
        ):
            raise TypeError("region_snapshot 必须是 AnalysisRegionSnapshot")
        if self.source_descriptor is not None and not isinstance(
            self.source_descriptor,
            AnalysisSourceDescriptor,
        ):
            raise TypeError("source_descriptor 必须是 AnalysisSourceDescriptor")
        if self.dependency_signature is not None and not isinstance(
            self.dependency_signature,
            AnalysisDependencySignature,
        ):
            raise TypeError(
                "dependency_signature 必须是 AnalysisDependencySignature"
            )
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
        tool_version = _TOOL_VERSIONS[self.tool]
        if self.tool in {
            AnalysisTool.MAXIMA,
            AnalysisTool.DIRECTIONALITY,
            AnalysisTool.SKELETON,
            AnalysisTool.SPATIAL_DISTRIBUTION,
        }:
            tool_version = str(self.parameters.get("algorithm_version", "1"))
        return AnalysisArtifact(
            id=artifact_id or f"analysis_{uuid4().hex}",
            source_document_id=self.document_id,
            source_pixel_revision=self.source_pixel_revision,
            source_reference=self.source_reference,
            region_snapshot=self.region_snapshot,
            source_descriptor=self.source_descriptor,
            dependency_signature=self.dependency_signature,
            tool_id=f"fdm.{self.tool.value}",
            tool_version=tool_version,
            parameters=dict(self.parameters),
            calibration_signature=self.calibration_signature,
            scalars=dict(self.scalars),
            tables=self.tables,
            curves=self.curves,
            assets=references,
            warnings=self.warnings,
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
        viewport_origin: Sequence[int] = (0, 0),
        source_reference: AnalysisObjectReference | None = None,
        region_snapshot: AnalysisRegionSnapshot | None = None,
        source_descriptor: AnalysisSourceDescriptor | None = None,
        dependency_signature: AnalysisDependencySignature | None = None,
        calibration: AnalysisCalibrationSnapshot | None = None,
        parameters: Mapping[str, object] | None = None,
        output_fields: Iterable[str] | None = None,
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
            viewport_origin=viewport_origin,
            source_reference=source_reference,
            region_snapshot=region_snapshot,
            source_descriptor=source_descriptor,
            dependency_signature=dependency_signature,
            calibration=calibration,
            parameters=parameters,
            output_fields=output_fields,
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
        AnalysisTool.FFT_POWER_SPECTRUM: 56,
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
        allowed = {
            "channel",
            "percentile_levels",
            "threshold_low",
            "threshold_high",
        }
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
            threshold_low=parameters.get("threshold_low"),  # type: ignore[arg-type]
            threshold_high=parameters.get("threshold_high"),  # type: ignore[arg-type]
            request_id=request.request_id,
            generation=request.generation,
        )
    if tool is AnalysisTool.HISTOGRAM:
        allowed = {"channel", "bins", "value_range", "log_counts"}
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
            log_counts=bool(parameters.get("log_counts", False)),
            request_id=request.request_id,
            generation=request.generation,
        )
    if tool is AnalysisTool.FFT_POWER_SPECTRUM:
        allowed = {
            "channel",
            "logarithmic",
            "centered",
            "window",
            "tukey_alpha",
        }
        _reject_unknown(parameters, allowed)
        return FftPowerSpectrumRequest(
            image=image,
            roi_mask=request.roi_mask,
            rings=request.raw_rings,
            channel=str(parameters.get("channel", "luminance")),
            logarithmic=bool(parameters.get("logarithmic", True)),
            centered=bool(parameters.get("centered", True)),
            window=str(parameters.get("window", "none")),
            tukey_alpha=float(parameters.get("tukey_alpha", 0.25)),
            request_id=request.request_id,
            generation=request.generation,
        )
    if tool is AnalysisTool.PROFILE:
        allowed = {
            "points",
            "line_width",
            "sample_spacing",
            "channel",
            "aggregation",
        }
        _reject_unknown(parameters, allowed)
        return IntensityProfileRequest(
            image=image,
            points=_freeze_ring(parameters.get("points", ())),
            line_width=float(parameters.get("line_width", 1.0)),
            sample_spacing=float(parameters.get("sample_spacing", 1.0)),
            pixel_size_x=calibration.pixel_size_x,
            pixel_size_y=calibration.pixel_size_y,
            channel=str(parameters.get("channel", "luminance")),
            aggregation=str(parameters.get("aggregation", "line")),
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
            "watershed",
            "watershed_min_distance",
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
            watershed=bool(parameters.get("watershed", False)),
            watershed_min_distance=int(
                parameters.get("watershed_min_distance", 3)
            ),
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
            "algorithm_version",
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
            algorithm_version=str(parameters.get("algorithm_version", "1")),
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
            "algorithm_version",
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
            algorithm_version=int(parameters.get("algorithm_version", 2)),
            request_id=request.request_id,
            generation=request.generation,
        )
    elif tool is AnalysisTool.SKELETON:
        allowed = {
            "threshold",
            "foreground",
            "channel",
            "already_skeletonized",
            "algorithm_version",
            "prune_terminal_branches_below",
            # Audit-only provenance for a Tubeness threshold-mask chain.
            # These fields are persisted on the resulting Artifact but do not
            # alter the skeleton kernel.
            "chain_parent_artifact_id",
            "chain_source_tubeness_artifact_id",
            "chain_threshold",
            "chain_mask_sha256",
            "chain_response_asset_sha256",
        }
        _reject_unknown(parameters, allowed)
        kernel_request = SkeletonNetworkRequest(
            mask=_binary_input_mask(request, image, parameters),
            already_skeletonized=bool(
                parameters.get("already_skeletonized", False)
            ),
            pixel_size_x=calibration.pixel_size_x,
            pixel_size_y=calibration.pixel_size_y,
            unit=calibration.unit,
            algorithm_version=int(parameters.get("algorithm_version", 2)),
            prune_terminal_branches_below=float(
                parameters.get("prune_terminal_branches_below", 0.0)
            ),
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
        allowed = {
            "points",
            "study_area",
            "study_bounds",
            "ripley_radii",
            "algorithm_version",
            "point_scope",
            "point_group_id",
            "point_group_label",
            "study_area_mode",
        }
        _reject_unknown(parameters, allowed)
        kernel_request = SpatialDistributionRequest(
            points=_freeze_ring(parameters.get("points", ())),
            study_area=parameters.get("study_area"),  # type: ignore[arg-type]
            study_bounds=(
                None
                if parameters.get("study_bounds") is None
                else tuple(parameters["study_bounds"])  # type: ignore[arg-type]
            ),
            ripley_radii=tuple(parameters.get("ripley_radii", ())),  # type: ignore[arg-type]
            algorithm_version=int(parameters.get("algorithm_version", 1)),
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
        return analyze_shape(  # type: ignore[arg-type]
            kernel_request,
            cancellation_check=cancellation_token.raise_if_cancelled,
        )
    if tool is AnalysisTool.INTENSITY:
        return analyze_intensity(  # type: ignore[arg-type]
            kernel_request,
            cancellation_check=cancellation_token.raise_if_cancelled,
        )
    if tool is AnalysisTool.HISTOGRAM:
        return calculate_histogram(  # type: ignore[arg-type]
            kernel_request,
            cancellation_check=cancellation_token.raise_if_cancelled,
        )
    if tool is AnalysisTool.FFT_POWER_SPECTRUM:
        return calculate_fft_power_spectrum(  # type: ignore[arg-type]
            kernel_request,
            cancellation_check=cancellation_token.raise_if_cancelled,
        )
    if tool is AnalysisTool.PROFILE:
        return sample_intensity_profile(  # type: ignore[arg-type]
            kernel_request,
            cancellation_check=cancellation_token.raise_if_cancelled,
        )
    if tool is AnalysisTool.PARTICLES:
        return analyze_particles(  # type: ignore[arg-type]
            kernel_request,
            cancellation_check=cancellation_token.raise_if_cancelled,
        )
    if tool is AnalysisTool.MAXIMA:
        return find_local_maxima(  # type: ignore[arg-type]
            kernel_request,
            cancellation_check=cancellation_token.raise_if_cancelled,
        )
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


def package_analysis_task_result(
    request: ImageAnalysisTaskRequest,
    kernel_result: object,
) -> ImageAnalysisTaskResult:
    """Package an already computed kernel result through the canonical path."""

    if not isinstance(request, ImageAnalysisTaskRequest):
        raise TypeError("request 必须是 ImageAnalysisTaskRequest")
    result_request_id = getattr(kernel_result, "request_id", request.request_id)
    result_generation = getattr(kernel_result, "generation", request.generation)
    if (
        str(result_request_id) != request.request_id
        or int(result_generation) != request.generation
    ):
        raise ValueError("分析内核结果与冻结请求不匹配")
    return _package_kernel_result(request, kernel_result)


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
        warnings = tuple(result.warnings)
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
            "component_count": result.component_count,
            "euler_number": result.euler_number,
            "extent": result.extent,
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
        tables.append(
            AnalysisTable(
                name="分组件形状指标",
                columns=(
                    "组件",
                    "面积(px²)",
                    "面积",
                    "质心X(px)",
                    "质心Y(px)",
                    "孔洞数",
                    "总周长(px)",
                    "Extent",
                    "Solidity",
                ),
                rows=tuple(
                    (
                        item.index,
                        item.area_px,
                        item.area,
                        item.centroid_px[0],
                        item.centroid_px[1],
                        item.hole_count,
                        item.total_perimeter_px,
                        item.extent,
                        item.solidity,
                    )
                    for item in result.component_table
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
            "mode": result.mode,
            "stddev": result.stddev,
            "skewness": result.skewness,
            "excess_kurtosis": result.excess_kurtosis,
            "minimum": result.minimum,
            "maximum": result.maximum,
            "integrated_density": result.integrated_density,
            "threshold_area_fraction": result.threshold_area_fraction,
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
        if result.channel_statistics:
            tables.append(
                AnalysisTable(
                    name="通道统计",
                    columns=(
                        "通道",
                        "有效像素",
                        "均值",
                        "中位数",
                        "众数",
                        "总体标准差",
                        "偏度",
                        "超额峰度",
                        "最小值",
                        "最大值",
                        "积分密度",
                        "阈值面积分数",
                    ),
                    rows=tuple(
                        (
                            item.channel,
                            item.valid_pixel_count,
                            item.mean,
                            item.median,
                            item.mode,
                            item.stddev,
                            item.skewness,
                            item.excess_kurtosis,
                            item.minimum,
                            item.maximum,
                            item.integrated_density,
                            item.threshold_area_fraction,
                        )
                        for item in result.channel_statistics
                    ),
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
            "log_counts": result.log_counts,
            "range_minimum": result.edges[0],
            "range_maximum": result.edges[-1],
        }
        curves.append(
            AnalysisCurve(
                name="直方图",
                x=centers,
                y=result.display_counts,
                x_unit="强度",
                y_unit="log(1+频数)" if result.log_counts else "频数",
            )
        )
        tables.append(
            AnalysisTable(
                name="直方图明细",
                columns=("下界", "上界", "中心", "原始频数", "显示值"),
                rows=tuple(
                    (
                        result.edges[index],
                        result.edges[index + 1],
                        centers[index],
                        result.counts[index],
                        result.display_counts[index],
                    )
                    for index in range(len(result.counts))
                ),
            )
        )
    elif tool is AnalysisTool.FFT_POWER_SPECTRUM:
        power = np.asarray(result.power, dtype=np.float32)
        finite = power[np.isfinite(power)]
        scalars = {
            "width": int(power.shape[1]),
            "height": int(power.shape[0]),
            "finite_value_count": int(finite.size),
            "non_finite_value_count": int(power.size - finite.size),
            "minimum": None if not finite.size else float(np.min(finite)),
            "maximum": None if not finite.size else float(np.max(finite)),
            "mean": (
                None
                if not finite.size
                else float(np.mean(finite, dtype=np.float64))
            ),
            "channel": result.channel,
            "logarithmic": result.logarithmic,
            "centered": result.centered,
            "window": result.window,
            "tukey_alpha": result.tukey_alpha,
            "roi_applied": result.roi_applied,
            "mask_policy": result.mask_policy,
        }
        assets.append(
            AnalysisAssetPayload(
                kind=AnalysisAssetKind.OTHER,
                schema="fdm.fft-power-spectrum.v1",
                suggested_stem="fft-power-spectrum",
                arrays={"power": power},
                metadata={
                    "channel": result.channel,
                    "logarithmic": result.logarithmic,
                    "centered": result.centered,
                    "window": result.window,
                    "tukey_alpha": result.tukey_alpha,
                    "source_size": list(result.source_size),
                    "analysis_bounds": list(result.analysis_bounds),
                    "roi_applied": result.roi_applied,
                    "mask_policy": result.mask_policy,
                    "frequency_axis_unit": "cycles_per_pixel",
                    "power_normalization": "unnormalized",
                },
            )
        )
    elif tool is AnalysisTool.PROFILE:
        scalars = {
            "valid_sample_count": result.valid_sample_count,
            "sample_count": len(result.values),
            "channel": result.channel,
            "aggregation": result.aggregation,
            "sample_spacing": request.parameters.get("sample_spacing", 1.0),
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
            "accepted_foreground_pixel_count": result.accepted_foreground_pixel_count,
            "area_fraction": result.area_fraction,
            "include_holes": result.include_holes,
            "connectivity": result.connectivity,
            "watershed": result.watershed,
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
        tables.append(
            AnalysisTable(
                name="粒子面积汇总",
                columns=("统计量", "数值"),
                rows=result.area_summary,
            )
        )
        assets.extend(
            (
                AnalysisAssetPayload(
                    kind=AnalysisAssetKind.LABEL_IMAGE,
                    schema="fdm.particle-labels.v2",
                    suggested_stem="particle-labels",
                    arrays={
                        "labels": result.label_image,
                        **_particle_conversion_arrays(
                            result.particles,
                            viewport_origin=request.viewport_origin,
                        ),
                    },
                    metadata={
                        "background_label": 0,
                        "coordinate_space": "viewport_pixel",
                        "conversion_schema": "fdm.particle-conversion.v2",
                    },
                ),
                AnalysisAssetPayload(
                    kind=AnalysisAssetKind.MASK,
                    schema="fdm.particle-contours.v2",
                    suggested_stem="particle-contours",
                    arrays={
                        "contours": result.contour_image.astype(np.uint8)
                    },
                ),
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
            ),
            viewport_origin=request.viewport_origin,
        )
    elif tool is AnalysisTool.MAXIMA:
        scalars = {
            "accepted_count": len(result.maxima),
            "candidate_plateau_count": result.candidate_plateau_count,
            "suppressed_count": result.suppressed_count,
            "channel": result.channel,
            "algorithm_version": result.algorithm_version,
            "conversion_schema": "fdm.maxima-conversion.v2",
            "conversion_viewport_origin_x": request.viewport_origin[0],
            "conversion_viewport_origin_y": request.viewport_origin[1],
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
                    columns=(
                        "序号",
                        "X(px)",
                        "Y(px)",
                        "强度",
                        (
                            "地形 prominence"
                            if result.algorithm_version == "2"
                            else "局部 prominence"
                        ),
                    ),
                    rows=rows,
                )
            )
        else:
            assets.append(
                AnalysisAssetPayload(
                    kind=AnalysisAssetKind.TABLE,
                    schema="fdm.maxima-table.v2",
                    suggested_stem="maxima",
                    arrays={
                        "values": np.asarray(rows, dtype=np.float64),
                        "viewport_origin": np.asarray(
                            request.viewport_origin,
                            dtype=np.int64,
                        ),
                    },
                    metadata={
                        "columns": [
                            "index",
                            "x_px",
                            "y_px",
                            "value",
                            "local_prominence",
                        ],
                        "coordinate_space": "viewport_pixel",
                        "conversion_schema": "fdm.maxima-conversion.v2",
                    },
                )
            )
        conversion = MaximaConversionPayload(
            points=tuple(
                (maximum.x, maximum.y, maximum.value)
                for maximum in result.maxima
            ),
            viewport_origin=request.viewport_origin,
        )
    elif tool is AnalysisTool.DIRECTIONALITY:
        scalars = {
            "valid_gradient_pixels": result.valid_gradient_pixels,
            "total_weight": result.total_weight,
            "convention": result.convention,
            "peak_count": len(result.peaks),
            "algorithm_version": result.algorithm_version,
            "concentration": result.concentration,
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
                columns=(
                    "角度(°)",
                    "权重",
                    "相对权重",
                    "区间索引",
                    "峰宽(°)",
                ),
                rows=tuple(
                    (
                        peak.angle_degrees,
                        peak.weight,
                        peak.relative_weight,
                        peak.bin_index,
                        peak.width_degrees,
                    )
                    for peak in result.peaks
                ),
            )
        )
        if result.algorithm_version == 2:
            directionality_arrays = {
                name: array
                for name, array in (
                    (
                        "gradient_magnitude_squared",
                        result.gradient_magnitude_squared,
                    ),
                    ("fourier_power", result.fourier_power),
                    (
                        "orientation_map_degrees",
                        result.orientation_map_degrees,
                    ),
                )
                if array is not None
            }
            if directionality_arrays:
                assets.append(
                    AnalysisAssetPayload(
                        kind=AnalysisAssetKind.OTHER,
                        schema="fdm.directionality.v2",
                        suggested_stem="directionality-v2",
                        arrays=directionality_arrays,
                        metadata={
                            "convention": result.convention,
                            "orientation_map": "HSB 方向图的轴向角度源数据",
                        },
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
            "algorithm_version": result.algorithm_version,
            "slab_pixel_count": result.slab_pixel_count,
            "junction_pixel_count": result.junction_pixel_count,
            "triple_junction_count": result.triple_junction_count,
            "quadruple_or_higher_junction_count": (
                result.quadruple_or_higher_junction_count
            ),
            "mean_branch_length": result.mean_branch_length,
            "maximum_branch_length": result.maximum_branch_length,
            "pruned_terminal_branch_count": len(result.pruning_audit),
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
                schema=(
                    "fdm.skeleton-network.v2"
                    if result.algorithm_version == 2
                    else "fdm.skeleton-network.v1"
                ),
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
                    **(
                        {
                            "classification_map": result.classification_map,
                            "pruning_audit": np.asarray(
                                [
                                    (
                                        item.start_px[0],
                                        item.start_px[1],
                                        item.length,
                                        item.removed_pixel_count,
                                    )
                                    for item in result.pruning_audit
                                ],
                                dtype=np.float64,
                            ).reshape((-1, 4)),
                        }
                        if (
                            result.algorithm_version == 2
                            and result.classification_map is not None
                        )
                        else {}
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
        selected_thickness = np.asarray(result.thickness_px, dtype=np.float64)
        selected_thickness = selected_thickness[selected_thickness > 0]
        isotropic_calibration = math.isclose(
            request.calibration.pixel_size_x,
            request.calibration.pixel_size_y,
            rel_tol=1e-12,
            abs_tol=1e-15,
        )
        if isotropic_calibration:
            thickness_scale = request.calibration.pixel_size_x
            thickness_unit = request.calibration.unit
            scale_rule = "isotropic_pixel_size"
        else:
            # The current maximal-circle kernel operates in pixel coordinates.
            # A geometric-mean scale would look plausible but is not a valid
            # physical distance under anisotropic calibration.
            thickness_scale = 1.0
            thickness_unit = "px"
            scale_rule = "pixel_only_anisotropic_calibration"
            warnings += (
                "横向与纵向像素尺寸不同；局部厚度内核按像素圆定义，"
                "因此仅输出 px，未生成可能失真的物理单位厚度。",
            )
        reported_thickness = selected_thickness * thickness_scale
        percentile_levels = (10.0, 25.0, 50.0, 75.0, 90.0)
        percentile_values = (
            np.percentile(reported_thickness, percentile_levels)
            if reported_thickness.size
            else np.asarray((), dtype=np.float64)
        )
        scalars = {
            "foreground_pixel_count": result.foreground_pixel_count,
            "maximum_thickness_px": result.maximum_thickness_px,
            "mean_thickness_px": result.mean_thickness_px,
            "maximum_thickness": (
                result.maximum_thickness_px * thickness_scale
            ),
            "mean_thickness": (
                None
                if result.mean_thickness_px is None
                else result.mean_thickness_px * thickness_scale
            ),
            "unit": thickness_unit,
            "physical_unit_available": isotropic_calibration,
            "definition": result.definition,
        }
        tables.append(
            AnalysisTable(
                name="局部厚度分位数",
                columns=("分位数(%)", f"厚度({thickness_unit})"),
                rows=tuple(
                    (level, float(value))
                    for level, value in zip(
                        percentile_levels,
                        percentile_values,
                        strict=True,
                    )
                ),
            )
        )
        if reported_thickness.size:
            counts, edges = np.histogram(reported_thickness, bins=64)
            centers = tuple(
                float((edges[index] + edges[index + 1]) / 2.0)
                for index in range(len(counts))
            )
            curves.append(
                AnalysisCurve(
                    name="局部厚度分布",
                    x=centers,
                    y=tuple(float(value) for value in counts),
                    x_unit=thickness_unit,
                    y_unit="频数",
                )
            )
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
                schema="fdm.local-thickness.v2",
                suggested_stem="local-thickness",
                arrays={
                    "thickness_px": result.thickness_px,
                    "thickness": (
                        np.asarray(result.thickness_px, dtype=np.float32)
                        * np.float32(thickness_scale)
                    ),
                    "maximal_circles": np.asarray(
                        circle_rows,
                        dtype=np.float64,
                    ).reshape((-1, 4)),
                },
                metadata={
                    "unit": thickness_unit,
                    "physical_scale_rule": scale_rule,
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
        feature_names = (
            ("Contrast", "contrast"),
            ("Dissimilarity", "dissimilarity"),
            ("Homogeneity", "homogeneity"),
            ("ASM", "angular_second_moment"),
            ("Energy", "energy"),
            ("Correlation", "correlation"),
            ("Entropy", "entropy"),
            ("Maximum Probability", "maximum_probability"),
        )
        tables.append(
            AnalysisTable(
                name="Haralick 聚合",
                columns=("特征", "均值", "总体标准差", "最小值", "最大值"),
                rows=tuple(
                    (
                        label,
                        float(np.mean(values)),
                        float(np.std(values)),
                        float(np.min(values)),
                        float(np.max(values)),
                    )
                    for label, attribute in feature_names
                    if (
                        values := np.asarray(
                            [
                                getattr(item, attribute)
                                for item in result.features
                            ],
                            dtype=np.float64,
                        )
                    ).size
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
            "algorithm_version": result.algorithm_version,
            "boundary_correction": result.boundary_correction,
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
        if result.ripley_radii:
            area_unit = (
                "px²"
                if result.unit == "px"
                else f"{result.unit}²"
            )
            tables.append(
                AnalysisTable(
                    name="Ripley K/L",
                    columns=("半径", "K(r)", "L(r)"),
                    rows=tuple(
                        (radius, k_value, l_value)
                        for radius, k_value, l_value in zip(
                            result.ripley_radii,
                            result.ripley_k,
                            result.ripley_l,
                            strict=True,
                        )
                    ),
                )
            )
            curves.extend(
                (
                    AnalysisCurve(
                        name="Ripley K(r)",
                        x=result.ripley_radii,
                        y=result.ripley_k,
                        x_unit=result.unit,
                        y_unit=area_unit,
                    ),
                    AnalysisCurve(
                        name="Ripley L(r)",
                        x=result.ripley_radii,
                        y=result.ripley_l,
                        x_unit=result.unit,
                        y_unit=result.unit,
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

    scalars, tables, curves, assets = _filter_selected_analysis_outputs(
        request,
        scalars=scalars,
        tables=tables,
        curves=curves,
        assets=assets,
    )
    artifact_parameters = request.parameters
    if request.output_fields is not None:
        artifact_parameters[ANALYSIS_OUTPUT_FIELDS_PARAMETER] = list(
            request.output_fields
        )
    return ImageAnalysisTaskResult(
        tool=tool,
        request_id=request.request_id,
        generation=request.generation,
        document_id=request.document_id,
        source_pixel_revision=request.source_pixel_revision,
        source_reference=request.source_reference,
        region_snapshot=request.region_snapshot,
        source_descriptor=request.source_descriptor,
        dependency_signature=request.dependency_signature,
        calibration_signature=request.calibration.signature,
        parameters=artifact_parameters,
        scalars=scalars,
        tables=tuple(tables),
        curves=tuple(curves),
        asset_payloads=tuple(assets),
        conversion_payload=conversion,
        warnings=warnings,
    )


def _filter_selected_analysis_outputs(
    request: ImageAnalysisTaskRequest,
    *,
    scalars: Mapping[str, JsonScalar],
    tables: Sequence[AnalysisTable],
    curves: Sequence[AnalysisCurve],
    assets: Sequence[AnalysisAssetPayload],
) -> tuple[
    dict[str, JsonScalar],
    list[AnalysisTable],
    list[AnalysisCurve],
    list[AnalysisAssetPayload],
]:
    schema = analysis_output_field_schema(f"fdm.{request.tool.value}")
    if schema is None or request.output_fields is None:
        return (
            dict(scalars),
            list(tables),
            list(curves),
            list(assets),
        )
    selected_keys = set(request.output_fields)
    selected_specs = tuple(
        field for field in schema.fields if field.key in selected_keys
    )
    scalar_keys = set(schema.required_scalar_keys)
    whole_table_names: set[str] = set()
    asset_schemas: set[str] = set()
    selected_columns: dict[str, set[str]] = {}
    selected_rows: dict[str, set[str]] = {}
    for field in selected_specs:
        scalar_keys.update(field.scalar_keys)
        whole_table_names.update(field.table_names)
        asset_schemas.update(field.asset_schemas)
        for table_name, column_name in field.table_columns:
            selected_columns.setdefault(table_name, set()).add(column_name)
        for table_name, row_label in field.table_row_labels:
            selected_rows.setdefault(table_name, set()).add(row_label)

    missing_required_scalars = set(schema.required_scalar_keys) - set(scalars)
    if missing_required_scalars:
        raise ValueError(
            "分析输出字段 schema 与实现不一致，缺少必要标量："
            + "、".join(sorted(missing_required_scalars))
        )
    filtered_scalars = {
        key: value for key, value in scalars.items() if key in scalar_keys
    }
    required_columns = dict(schema.required_table_columns)
    table_by_name = {table.name: table for table in tables}
    if len(table_by_name) != len(tables):
        raise ValueError("分析实现返回了重名表格，无法可靠应用输出字段选择")
    missing_whole_tables = whole_table_names - set(table_by_name)
    if missing_whole_tables:
        raise ValueError(
            "分析输出字段 schema 与实现不一致，缺少表格："
            + "、".join(sorted(missing_whole_tables))
        )
    for table_name, requested in selected_columns.items():
        table = table_by_name.get(table_name)
        if table is None:
            raise ValueError(
                f"分析输出字段 schema 与实现不一致，缺少表格：{table_name}"
            )
        missing_columns = (
            set(required_columns.get(table_name, ())) | requested
        ) - set(table.columns)
        if missing_columns:
            raise ValueError(
                f"分析输出字段 schema 与表格“{table_name}”不一致，缺少列："
                + "、".join(sorted(missing_columns))
            )
    for table_name, requested in selected_rows.items():
        table = table_by_name.get(table_name)
        if table is None:
            raise ValueError(
                f"分析输出字段 schema 与实现不一致，缺少表格：{table_name}"
            )
        available_labels = {
            str(row[0]) for row in table.rows if row
        }
        missing_rows = requested - available_labels
        if missing_rows:
            raise ValueError(
                f"分析输出字段 schema 与表格“{table_name}”不一致，缺少行："
                + "、".join(sorted(missing_rows))
            )
    available_asset_schemas = {asset.schema for asset in assets}
    missing_assets = asset_schemas - available_asset_schemas
    if missing_assets:
        raise ValueError(
            "分析输出字段 schema 与实现不一致，缺少资产："
            + "、".join(sorted(missing_assets))
        )

    filtered_tables: list[AnalysisTable] = []
    for table in tables:
        if table.name in whole_table_names:
            filtered_tables.append(table)
            continue
        requested_columns = selected_columns.get(table.name)
        if requested_columns:
            column_names = (
                *required_columns.get(table.name, ()),
                *(
                    column
                    for column in table.columns
                    if column in requested_columns
                ),
            )
            indices = tuple(
                index
                for index, column in enumerate(table.columns)
                if column in column_names
            )
            if indices:
                filtered_tables.append(
                    AnalysisTable(
                        name=table.name,
                        columns=tuple(table.columns[index] for index in indices),
                        rows=tuple(
                            tuple(row[index] for index in indices)
                            for row in table.rows
                        ),
                    )
                )
            continue
        requested_rows = selected_rows.get(table.name)
        if requested_rows:
            rows = tuple(
                row
                for row in table.rows
                if row and str(row[0]) in requested_rows
            )
            if rows:
                filtered_tables.append(
                    AnalysisTable(
                        name=table.name,
                        columns=table.columns,
                        rows=rows,
                    )
                )
    filtered_assets = [
        asset for asset in assets if asset.schema in asset_schemas
    ]
    # None of the first output-field schemas currently selects curve outputs.
    # Keeping this explicit prevents future tools from silently leaking a
    # complete curve when their schema has not declared one.
    filtered_curves: list[AnalysisCurve] = []
    return (
        filtered_scalars,
        filtered_tables,
        filtered_curves,
        filtered_assets,
    )


def _particle_conversion_arrays(
    particles: Sequence[object],
    *,
    viewport_origin: tuple[int, int],
) -> dict[str, NDArray[np.generic]]:
    coordinates: list[Coordinate] = []
    ring_offsets = [0]
    particle_ring_offsets = [0]
    indices: list[int] = []
    exact_areas: list[int] = []
    centroids: list[Coordinate] = []
    ring_count = 0
    for particle in particles:
        indices.append(int(particle.index))
        exact_areas.append(int(particle.exact_area_px))
        centroids.append(
            (
                float(particle.centroid_px[0]),
                float(particle.centroid_px[1]),
            )
        )
        for ring in particle.rings:
            coordinates.extend((float(x), float(y)) for x, y in ring)
            ring_offsets.append(len(coordinates))
            ring_count += 1
        particle_ring_offsets.append(ring_count)
    return {
        "coordinates": np.asarray(coordinates, dtype=np.float64).reshape((-1, 2)),
        "ring_offsets": np.asarray(ring_offsets, dtype=np.int64),
        "particle_ring_offsets": np.asarray(
            particle_ring_offsets,
            dtype=np.int64,
        ),
        "particle_index": np.asarray(indices, dtype=np.int64),
        "exact_area_px": np.asarray(exact_areas, dtype=np.int64),
        "centroid_px": np.asarray(centroids, dtype=np.float64).reshape((-1, 2)),
        "viewport_origin": np.asarray(viewport_origin, dtype=np.int64),
    }


def rebuild_particle_conversion_payload(
    artifact: AnalysisArtifact,
    *,
    asset_root: str | Path | None = None,
    asset_source_paths: Mapping[str, str | Path] | None = None,
) -> ParticleConversionPayload:
    """Rebuild exact particle conversion geometry from a saved safe NPZ."""

    if not isinstance(artifact, AnalysisArtifact):
        raise TypeError("artifact 必须是 AnalysisArtifact")
    references = tuple(
        reference
        for reference in artifact.assets
        if reference.metadata.get("schema") == "fdm.particle-labels.v2"
    )
    if len(references) != 1:
        raise ValueError("粒子 v2 分析结果必须恰好包含一个标签几何资产")
    reference = references[0]
    if (
        reference.kind is not AnalysisAssetKind.LABEL_IMAGE
        or reference.metadata.get("conversion_schema")
        != "fdm.particle-conversion.v2"
        or reference.metadata.get("coordinate_space") != "viewport_pixel"
        or reference.metadata.get("allow_pickle") is not False
    ):
        raise ValueError("粒子 v2 标签资产 metadata 不满足转换契约")
    candidate = _resolve_analysis_asset_path(
        reference.path,
        asset_root=asset_root,
        asset_source_paths=asset_source_paths,
    )
    validate_analysis_asset_reference(candidate, reference)
    required_members = {
        "labels",
        "coordinates",
        "ring_offsets",
        "particle_ring_offsets",
        "particle_index",
        "exact_area_px",
        "centroid_px",
        "viewport_origin",
    }
    declared_members = reference.metadata.get("members")
    if (
        not isinstance(declared_members, Mapping)
        or set(declared_members) != required_members
    ):
        raise ValueError("粒子 v2 标签资产缺少完整 members metadata")
    with np.load(candidate, allow_pickle=False) as archive:
        data_members = set(archive.files) - {ANALYSIS_NPZ_MANIFEST_MEMBER}
        if data_members != required_members:
            missing = required_members - data_members
            unknown = data_members - required_members
            details = []
            if missing:
                details.append(f"缺少 {', '.join(sorted(missing))}")
            if unknown:
                details.append(f"未知 {', '.join(sorted(unknown))}")
            raise ValueError("粒子转换资产成员不匹配：" + "；".join(details))
        arrays = {name: np.asarray(archive[name]) for name in required_members}
    if any(array.dtype.hasobject for array in arrays.values()):
        raise TypeError("粒子转换资产禁止 object dtype")
    labels = _required_integer_array(arrays["labels"], "labels", dimensions=2)
    coordinates = _required_float_matrix(
        arrays["coordinates"],
        "coordinates",
        columns=2,
    )
    ring_offsets = _required_offsets(
        arrays["ring_offsets"],
        "ring_offsets",
        maximum=len(coordinates),
    )
    particle_ring_offsets = _required_offsets(
        arrays["particle_ring_offsets"],
        "particle_ring_offsets",
        maximum=len(ring_offsets) - 1,
    )
    particle_indices = _required_integer_array(
        arrays["particle_index"],
        "particle_index",
        dimensions=1,
    )
    exact_areas = _required_integer_array(
        arrays["exact_area_px"],
        "exact_area_px",
        dimensions=1,
    )
    centroids = _required_float_matrix(
        arrays["centroid_px"],
        "centroid_px",
        columns=2,
    )
    viewport_origin_array = _required_integer_array(
        arrays["viewport_origin"],
        "viewport_origin",
        dimensions=1,
    )
    if viewport_origin_array.shape != (2,):
        raise ValueError("viewport_origin 必须包含两个整数")
    particle_count = len(particle_indices)
    if (
        len(exact_areas) != particle_count
        or len(centroids) != particle_count
        or len(particle_ring_offsets) != particle_count + 1
    ):
        raise ValueError("粒子转换资产的粒子数组长度不一致")
    if not np.array_equal(
        particle_indices,
        np.arange(1, particle_count + 1, dtype=particle_indices.dtype),
    ):
        raise ValueError("particle_index 必须从 1 连续递增")
    if np.any(exact_areas <= 0):
        raise ValueError("exact_area_px 必须全部大于 0")
    if labels.size and (
        int(np.min(labels)) < 0 or int(np.max(labels)) > particle_count
    ):
        raise ValueError("labels 包含超出粒子数量的标签")
    candidates: list[ParticleMeasurementCandidate] = []
    for particle_offset in range(particle_count):
        first_ring = int(particle_ring_offsets[particle_offset])
        last_ring = int(particle_ring_offsets[particle_offset + 1])
        rings: list[ImmutableRing] = []
        for ring_index in range(first_ring, last_ring):
            first_point = int(ring_offsets[ring_index])
            last_point = int(ring_offsets[ring_index + 1])
            if last_point - first_point < 3:
                raise ValueError("粒子转换资产中的环至少需要三个点")
            rings.append(
                tuple(
                    (float(x), float(y))
                    for x, y in coordinates[first_point:last_point]
                )
            )
        if not rings:
            raise ValueError("每个可转换粒子至少需要一个环")
        candidates.append(
            ParticleMeasurementCandidate(
                index=int(particle_indices[particle_offset]),
                exact_area_px=int(exact_areas[particle_offset]),
                centroid_px=(
                    float(centroids[particle_offset, 0]),
                    float(centroids[particle_offset, 1]),
                ),
                rings=tuple(rings),
            )
        )
    return ParticleConversionPayload(
        candidates=tuple(candidates),
        viewport_origin=(
            int(viewport_origin_array[0]),
            int(viewport_origin_array[1]),
        ),
    )


def rebuild_maxima_conversion_payload(
    artifact: AnalysisArtifact,
    *,
    asset_root: str | Path | None = None,
    asset_source_paths: Mapping[str, str | Path] | None = None,
) -> MaximaConversionPayload:
    """Rebuild maxima points from inline data or a validated v1/v2 NPZ."""

    if not isinstance(artifact, AnalysisArtifact):
        raise TypeError("artifact 必须是 AnalysisArtifact")
    table = next(
        (candidate for candidate in artifact.tables if candidate.name == "极值点"),
        None,
    )
    if table is not None:
        if len(table.columns) < 4:
            raise ValueError("极值点表至少需要序号、X、Y 和强度四列")
        rows = table.rows
        points = tuple(
            _validated_maxima_point(row[1], row[2], row[3])
            for row in rows
        )
        return MaximaConversionPayload(
            points=points,
            viewport_origin=_inline_maxima_viewport_origin(artifact),
        )
    references = tuple(
        reference
        for reference in artifact.assets
        if reference.metadata.get("schema")
        in {"fdm.maxima-table.v1", "fdm.maxima-table.v2"}
    )
    if len(references) != 1:
        raise ValueError(
            "极值分析结果缺少唯一的 inline 表或 maxima-table.v1/v2 资产"
        )
    reference = references[0]
    expected_columns = [
        "index",
        "x_px",
        "y_px",
        "value",
        "local_prominence",
    ]
    metadata = reference.metadata
    schema = metadata.get("schema")
    required_members = (
        {"values", "viewport_origin"}
        if schema == "fdm.maxima-table.v2"
        else {"values"}
    )
    if (
        reference.kind is not AnalysisAssetKind.TABLE
        or metadata.get("allow_pickle") is not False
        or metadata.get("columns") != expected_columns
        or not isinstance(metadata.get("members"), Mapping)
        or set(metadata["members"]) != required_members  # type: ignore[arg-type]
    ):
        raise ValueError(f"{schema} 的 metadata 不满足转换契约")
    if schema == "fdm.maxima-table.v2" and (
        metadata.get("conversion_schema") != "fdm.maxima-conversion.v2"
        or metadata.get("coordinate_space") != "viewport_pixel"
    ):
        raise ValueError("maxima-table.v2 缺少坐标空间或转换 schema")
    candidate = _resolve_analysis_asset_path(
        reference.path,
        asset_root=asset_root,
        asset_source_paths=asset_source_paths,
    )
    validate_analysis_asset_reference(candidate, reference)
    with np.load(candidate, allow_pickle=False) as archive:
        data_members = {
            name
            for name in archive.files
            if name != ANALYSIS_NPZ_MANIFEST_MEMBER
        }
        if data_members != required_members:
            raise ValueError(f"{schema} 的资产成员不匹配")
        values = np.asarray(archive["values"])
        viewport_origin_array = (
            np.asarray(archive["viewport_origin"])
            if schema == "fdm.maxima-table.v2"
            else None
        )
    matrix = _required_float_matrix(values, "values", columns=5)
    points = tuple(
        _validated_maxima_point(row[1], row[2], row[3])
        for row in matrix
    )
    viewport_origin = (0, 0)
    if viewport_origin_array is not None:
        normalized_origin = _required_integer_array(
            viewport_origin_array,
            "viewport_origin",
            dimensions=1,
        )
        if normalized_origin.shape != (2,):
            raise ValueError("viewport_origin 必须包含两个整数")
        viewport_origin = (
            int(normalized_origin[0]),
            int(normalized_origin[1]),
        )
    return MaximaConversionPayload(
        points=points,
        viewport_origin=viewport_origin,
    )


def _inline_maxima_viewport_origin(
    artifact: AnalysisArtifact,
) -> tuple[int, int]:
    schema = artifact.scalars.get("conversion_schema")
    if schema is None:
        return (0, 0)
    if schema != "fdm.maxima-conversion.v2":
        raise ValueError(f"不支持的极值转换 schema：{schema}")
    values = (
        artifact.scalars.get("conversion_viewport_origin_x"),
        artifact.scalars.get("conversion_viewport_origin_y"),
    )
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in values
    ):
        raise TypeError("极值转换 viewport_origin 必须包含两个整数")
    return int(values[0]), int(values[1])  # type: ignore[arg-type]


def rebuild_analysis_conversion_payload(
    artifact: AnalysisArtifact,
    *,
    asset_root: str | Path | None = None,
    asset_source_paths: Mapping[str, str | Path] | None = None,
) -> ConversionPayload:
    """Unified persisted-artifact conversion boundary for particle/maxima."""

    token = artifact.tool_id.casefold()
    if "particle" in token:
        return rebuild_particle_conversion_payload(
            artifact,
            asset_root=asset_root,
            asset_source_paths=asset_source_paths,
        )
    if "maxima" in token or "extrema" in token:
        return rebuild_maxima_conversion_payload(
            artifact,
            asset_root=asset_root,
            asset_source_paths=asset_source_paths,
        )
    raise ValueError(f"分析工具不支持转换重建：{artifact.tool_id}")


def _validated_maxima_point(
    x_value: object,
    y_value: object,
    intensity_value: object,
) -> tuple[float, float, float]:
    point = (float(x_value), float(y_value), float(intensity_value))
    if any(not math.isfinite(value) for value in point):
        raise ValueError("极值转换坐标和强度必须是有限数")
    return point


def _resolve_analysis_asset_path(
    asset_path: str,
    *,
    asset_root: str | Path | None,
    asset_source_paths: Mapping[str, str | Path] | None,
) -> Path:
    mapped = dict(asset_source_paths or {}).get(asset_path)
    if mapped is not None:
        candidate = Path(mapped).resolve()
    elif asset_root is not None:
        root = Path(asset_root).resolve()
        candidate = (root / asset_path).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise ValueError("分析资产路径逃逸项目目录") from exc
    else:
        raise ValueError("必须提供 asset_root 或 asset_source_paths")
    if not candidate.is_file():
        raise FileNotFoundError(f"分析资产不存在：{asset_path}")
    return candidate


def _required_integer_array(
    array: NDArray[Any],
    name: str,
    *,
    dimensions: int,
) -> NDArray[np.integer[Any]]:
    if array.ndim != dimensions or array.dtype.kind not in "iu":
        raise TypeError(f"{name} 必须是 {dimensions} 维整数数组")
    return array


def _required_float_matrix(
    array: NDArray[Any],
    name: str,
    *,
    columns: int,
) -> NDArray[np.floating[Any]]:
    if (
        array.ndim != 2
        or array.shape[1] != columns
        or array.dtype.kind not in "fiu"
    ):
        raise TypeError(f"{name} 必须是 N×{columns} 数值数组")
    normalized = np.asarray(array, dtype=np.float64)
    if np.any(~np.isfinite(normalized)):
        raise ValueError(f"{name} 必须全部是有限数")
    return normalized


def _required_offsets(
    array: NDArray[Any],
    name: str,
    *,
    maximum: int,
) -> NDArray[np.integer[Any]]:
    normalized = _required_integer_array(array, name, dimensions=1)
    if (
        not len(normalized)
        or int(normalized[0]) != 0
        or int(normalized[-1]) != maximum
        or np.any(np.diff(normalized) < 0)
    ):
        raise ValueError(f"{name} 必须从 0 单调递增并终止于 {maximum}")
    return normalized


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
        foreground = {
            "bright": "above",
            "dark": "below",
        }.get(foreground, foreground)
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
