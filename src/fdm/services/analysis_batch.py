"""Deterministic recipe/batch execution for plane-based analyses.

An :class:`AnalysisRecipe` may contain one or more independent basic or
advanced analysis tool invocations.  Every source item runs the recipe steps
sequentially and is published only after all of its steps succeed.  The legacy
single-tool constructor and result ``execution`` attribute remain available to
callers that have not yet adopted multi-tool recipes.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from enum import StrEnum
import json
import math
from types import MappingProxyType
from typing import Any, TypeAlias

import cv2
import numpy as np
from numpy.typing import NDArray

from fdm.analysis_artifacts import AnalysisToolSpec
from fdm.cancellation import CancellationError, CancellationToken
from fdm.raster import RasterPixelType, RasterPlane
from fdm.services.advanced_analysis_registry import (
    AdvancedArrayDescriptor,
    AdvancedAnalysisExecution,
    AdvancedAnalysisInvocation,
    AdvancedAnalysisRegistry,
)
from fdm.services.advanced_image_analysis import (
    AdvancedAnalysisKind,
    AdvancedAnalysisLimits,
    DEFAULT_ADVANCED_ANALYSIS_LIMITS,
)
from fdm.services.image_analysis import (
    FftPowerSpectrumRequest,
    FindMaximaRequest,
    HistogramRequest,
    IntensityAnalysisRequest,
    ParticleAnalysisRequest,
    analyze_intensity,
    analyze_particles,
    calculate_fft_power_spectrum,
    calculate_histogram,
    find_local_maxima,
)
from fdm.services.raster_io import raster_plane_to_numpy


class BasicAnalysisKind(StrEnum):
    """Plane-only basic tools that are safe to repeat across many sources."""

    INTENSITY = "intensity"
    HISTOGRAM = "histogram"
    FFT_POWER_SPECTRUM = "fft_power_spectrum"
    PARTICLES = "particles"
    MAXIMA = "maxima"


AnalysisBatchKind: TypeAlias = BasicAnalysisKind | AdvancedAnalysisKind

_BASIC_TOOL_VERSIONS: Mapping[BasicAnalysisKind, str] = MappingProxyType(
    {
        BasicAnalysisKind.INTENSITY: "2",
        BasicAnalysisKind.HISTOGRAM: "2",
        BasicAnalysisKind.FFT_POWER_SPECTRUM: "1",
        BasicAnalysisKind.PARTICLES: "2",
        BasicAnalysisKind.MAXIMA: "1",
    }
)

_BASIC_TOOL_NAMES: Mapping[BasicAnalysisKind, str] = MappingProxyType(
    {
        BasicAnalysisKind.INTENSITY: "灰度与颜色统计",
        BasicAnalysisKind.HISTOGRAM: "直方图",
        BasicAnalysisKind.FFT_POWER_SPECTRUM: "FFT 功率谱",
        BasicAnalysisKind.PARTICLES: "粒子分析",
        BasicAnalysisKind.MAXIMA: "极值检测",
    }
)

_BASIC_RESOURCE_MULTIPLIERS: Mapping[BasicAnalysisKind, int] = MappingProxyType(
    {
        BasicAnalysisKind.INTENSITY: 12,
        BasicAnalysisKind.HISTOGRAM: 12,
        BasicAnalysisKind.FFT_POWER_SPECTRUM: 56,
        BasicAnalysisKind.PARTICLES: 28,
        BasicAnalysisKind.MAXIMA: 24,
    }
)

_MAX_BASIC_ANALYSIS_WORKING_BYTES = 1 << 30


def _analysis_batch_kind(kind: AnalysisBatchKind | str) -> AnalysisBatchKind:
    if isinstance(kind, (BasicAnalysisKind, AdvancedAnalysisKind)):
        return kind
    try:
        return AdvancedAnalysisKind(kind)
    except ValueError:
        return BasicAnalysisKind(kind)


class AnalysisSourceKind(StrEnum):
    IMAGE = "image"
    DIGITAL_SLIDE = "digital_slide"


@dataclass(frozen=True, slots=True, init=False)
class BasicAnalysisInvocation:
    """Immutable source inputs for one plane-based basic analysis."""

    kind: BasicAnalysisKind
    request_id: str
    generation: int
    plane: RasterPlane | None
    roi_mask: NDArray[np.bool_] | None
    raw_rings: tuple[tuple[tuple[float, float], ...], ...]
    binary_mask: NDArray[np.bool_] | None
    points: tuple[tuple[float, float], ...]
    pixel_size_x: float
    pixel_size_y: float
    unit: str
    _parameters_json: str = field(repr=False)

    def __init__(
        self,
        kind: BasicAnalysisKind | str,
        *,
        request_id: str,
        generation: int,
        plane: RasterPlane | None = None,
        roi_mask: NDArray[np.bool_] | None = None,
        raw_rings: Iterable[Iterable[tuple[float, float]]] = (),
        binary_mask: NDArray[np.bool_] | None = None,
        points: Iterable[tuple[float, float]] = (),
        pixel_size_x: float = 1.0,
        pixel_size_y: float = 1.0,
        unit: str = "px",
        parameters: Mapping[str, object] | None = None,
    ) -> None:
        normalized_request_id = str(request_id or "").strip()
        if not normalized_request_id:
            raise ValueError("request_id 不能为空")
        normalized_generation = int(generation)
        if normalized_generation < 0:
            raise ValueError("generation 不能为负数")
        if plane is not None and not isinstance(plane, RasterPlane):
            raise TypeError("plane 必须是 RasterPlane")
        expected_shape = (
            None if plane is None else (int(plane.height), int(plane.width))
        )
        normalized_roi = _freeze_basic_mask(roi_mask, expected_shape, "ROI")
        normalized_rings = tuple(
            tuple(
                (
                    _finite_coordinate(x, "RAW 点 X"),
                    _finite_coordinate(y, "RAW 点 Y"),
                )
                for x, y in ring
            )
            for ring in raw_rings
        )
        normalized_binary = _freeze_basic_mask(
            binary_mask,
            expected_shape,
            "二值掩膜",
        )
        normalized_points = tuple(
            (
                _finite_coordinate(x, "点 X"),
                _finite_coordinate(y, "点 Y"),
            )
            for x, y in points
        )
        normalized_pixel_x = _positive_finite(
            pixel_size_x,
            "横向像素尺寸",
        )
        normalized_pixel_y = _positive_finite(
            pixel_size_y,
            "纵向像素尺寸",
        )
        parameters_json = json.dumps(
            dict(parameters or {}),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        object.__setattr__(self, "kind", BasicAnalysisKind(kind))
        object.__setattr__(self, "request_id", normalized_request_id)
        object.__setattr__(self, "generation", normalized_generation)
        object.__setattr__(self, "plane", plane)
        object.__setattr__(self, "roi_mask", normalized_roi)
        object.__setattr__(self, "raw_rings", normalized_rings)
        object.__setattr__(self, "binary_mask", normalized_binary)
        object.__setattr__(self, "points", normalized_points)
        object.__setattr__(self, "pixel_size_x", normalized_pixel_x)
        object.__setattr__(self, "pixel_size_y", normalized_pixel_y)
        object.__setattr__(self, "unit", str(unit or "px").strip() or "px")
        object.__setattr__(self, "_parameters_json", parameters_json)

    @property
    def parameters(self) -> Mapping[str, object]:
        return MappingProxyType(json.loads(self._parameters_json))


@dataclass(frozen=True, slots=True)
class BasicAnalysisExecution:
    kind: BasicAnalysisKind
    chinese_name: str
    algorithm_version: str
    request_id: str
    generation: int
    tool_spec: AnalysisToolSpec
    result: object
    scalar_report: tuple[tuple[str, str | int | float | bool | None], ...]
    arrays: tuple[AdvancedArrayDescriptor, ...]

    @property
    def scalar_report_map(
        self,
    ) -> Mapping[str, str | int | float | bool | None]:
        return MappingProxyType(dict(self.scalar_report))


AnalysisBatchExecution: TypeAlias = (
    BasicAnalysisExecution | AdvancedAnalysisExecution
)


@dataclass(frozen=True, slots=True)
class AnalysisViewport:
    x: int
    y: int
    width: int
    height: int
    level: int = 0

    def __post_init__(self) -> None:
        for name in ("x", "y", "level"):
            value = int(getattr(self, name))
            if value < 0:
                raise ValueError(f"viewport.{name} 不能为负数")
            object.__setattr__(self, name, value)
        for name in ("width", "height"):
            value = int(getattr(self, name))
            if value < 1:
                raise ValueError(f"viewport.{name} 必须为正整数")
            object.__setattr__(self, name, value)


@dataclass(frozen=True, slots=True, init=False)
class AnalysisToolInvocation:
    """One immutable tool step inside an :class:`AnalysisRecipe`."""

    kind: AnalysisBatchKind
    required_inputs: tuple[str, ...]
    _parameters_json: str = field(repr=False)

    def __init__(
        self,
        kind: AnalysisBatchKind | str,
        *,
        parameters: Mapping[str, object] | None = None,
        required_inputs: Iterable[str] = (),
    ) -> None:
        normalized_inputs = tuple(
            dict.fromkeys(str(item).strip() for item in required_inputs)
        )
        if any(not item for item in normalized_inputs):
            raise ValueError("required_inputs 不能包含空值")
        parameters_json = json.dumps(
            dict(parameters or {}),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        object.__setattr__(self, "kind", _analysis_batch_kind(kind))
        object.__setattr__(self, "required_inputs", normalized_inputs)
        object.__setattr__(self, "_parameters_json", parameters_json)

    @property
    def parameters(self) -> Mapping[str, object]:
        return MappingProxyType(json.loads(self._parameters_json))


@dataclass(frozen=True, slots=True, init=False)
class AnalysisRecipe:
    """An ordered, immutable collection of batch-analysis tool steps.

    ``kind``/``parameters``/``required_inputs`` construct a legacy single-step
    recipe.  New callers can instead pass ``invocations``.  Compatibility
    properties with the legacy names expose the first step because existing UI
    code uses it to freeze the source request and to choose the result adapter.
    """

    recipe_id: str
    name: str
    invocations: tuple[AnalysisToolInvocation, ...]

    def __init__(
        self,
        recipe_id: str,
        name: str,
        kind: AnalysisBatchKind | str | None = None,
        *,
        parameters: Mapping[str, object] | None = None,
        required_inputs: Iterable[str] = (),
        invocations: Iterable[AnalysisToolInvocation] | None = None,
    ) -> None:
        recipe_token = str(recipe_id or "").strip()
        name_token = str(name or "").strip()
        if not recipe_token or not name_token:
            raise ValueError("recipe_id 和 name 不能为空")
        if invocations is None:
            if kind is None:
                raise ValueError("单工具配方必须指定 kind")
            normalized_invocations = (
                AnalysisToolInvocation(
                    kind,
                    parameters=parameters,
                    required_inputs=required_inputs,
                ),
            )
        else:
            normalized_invocations = tuple(invocations)
            if not normalized_invocations:
                raise ValueError("分析配方至少需要一个工具步骤")
            if any(
                not isinstance(item, AnalysisToolInvocation)
                for item in normalized_invocations
            ):
                raise TypeError(
                    "invocations 必须全部是 AnalysisToolInvocation"
                )
            if kind is not None and (
                _analysis_batch_kind(kind)
                is not normalized_invocations[0].kind
            ):
                raise ValueError("kind 必须与配方首个工具步骤一致")
            if parameters:
                raise ValueError(
                    "多工具配方请在 AnalysisToolInvocation 中设置 parameters"
                )
            if tuple(required_inputs):
                raise ValueError(
                    "多工具配方请在 AnalysisToolInvocation 中设置 required_inputs"
                )
        object.__setattr__(self, "recipe_id", recipe_token)
        object.__setattr__(self, "name", name_token)
        object.__setattr__(self, "invocations", normalized_invocations)

    @property
    def tool_invocations(self) -> tuple[AnalysisToolInvocation, ...]:
        """Explicit alias for callers that prefer the longer domain name."""

        return self.invocations

    @property
    def kind(self) -> AnalysisBatchKind:
        """Legacy first-step analysis kind."""

        return self.invocations[0].kind

    @property
    def parameters(self) -> Mapping[str, object]:
        """Legacy first-step parameters."""

        return self.invocations[0].parameters

    @property
    def required_inputs(self) -> tuple[str, ...]:
        """Legacy first-step required inputs."""

        return self.invocations[0].required_inputs

    @property
    def all_required_inputs(self) -> tuple[str, ...]:
        """Union of dependencies required by every recipe step."""

        return tuple(
            dict.fromkeys(
                required_input
                for invocation in self.invocations
                for required_input in invocation.required_inputs
            )
        )


@dataclass(frozen=True, slots=True)
class AnalysisInvocation:
    item_id: str
    display_name: str
    analysis: BasicAnalysisInvocation | AdvancedAnalysisInvocation
    source_kind: AnalysisSourceKind = AnalysisSourceKind.IMAGE
    viewport: AnalysisViewport | None = None

    def __post_init__(self) -> None:
        item_id = str(self.item_id or "").strip()
        display_name = str(self.display_name or "").strip()
        if not item_id or not display_name:
            raise ValueError("item_id 和 display_name 不能为空")
        if not isinstance(
            self.analysis,
            (BasicAnalysisInvocation, AdvancedAnalysisInvocation),
        ):
            raise TypeError(
                "analysis 必须是 BasicAnalysisInvocation 或 "
                "AdvancedAnalysisInvocation"
            )
        source_kind = AnalysisSourceKind(self.source_kind)
        if source_kind is AnalysisSourceKind.DIGITAL_SLIDE and self.viewport is None:
            raise ValueError("数字切片批量分析必须显式指定 viewport")
        if self.viewport is not None and not isinstance(
            self.viewport,
            AnalysisViewport,
        ):
            raise TypeError("viewport 必须是 AnalysisViewport")
        object.__setattr__(self, "item_id", item_id)
        object.__setattr__(self, "display_name", display_name)
        object.__setattr__(self, "source_kind", source_kind)


@dataclass(frozen=True, slots=True)
class AnalysisBatchRequest:
    request_id: str
    generation: int
    recipe: AnalysisRecipe
    invocations: tuple[AnalysisInvocation, ...]
    continue_on_error: bool = True

    def __post_init__(self) -> None:
        request_id = str(self.request_id or "").strip()
        if not request_id:
            raise ValueError("request_id 不能为空")
        generation = int(self.generation)
        if generation < 0:
            raise ValueError("generation 不能为负数")
        if not isinstance(self.recipe, AnalysisRecipe):
            raise TypeError("recipe 必须是 AnalysisRecipe")
        invocations = tuple(self.invocations)
        if not invocations:
            raise ValueError("批量分析至少需要一个项目")
        if any(not isinstance(item, AnalysisInvocation) for item in invocations):
            raise TypeError("invocations 必须全部是 AnalysisInvocation")
        item_ids = tuple(item.item_id for item in invocations)
        if len(set(item_ids)) != len(item_ids):
            raise ValueError("批量分析 item_id 不能重复")
        for item in invocations:
            if _analysis_batch_kind(item.analysis.kind) is not self.recipe.kind:
                raise ValueError("批量项目的分析类型必须与配方首个步骤一致")
            if item.analysis.generation != generation:
                raise ValueError("批量项目 generation 必须与批次一致")
            available_inputs = {
                "pixel_size",
                "unit",
            }
            if item.analysis.plane is not None:
                available_inputs.add("plane")
            if item.analysis.roi_mask is not None:
                available_inputs.add("roi_mask")
            if item.analysis.binary_mask is not None:
                available_inputs.add("binary_mask")
            if item.analysis.points:
                available_inputs.add("points")
            if item.viewport is not None:
                available_inputs.add("viewport")
            for step_index, invocation in enumerate(
                self.recipe.invocations,
                start=1,
            ):
                missing_inputs = (
                    set(invocation.required_inputs) - available_inputs
                )
                if missing_inputs:
                    raise ValueError(
                        f"批量项目 {item.item_id} 的配方步骤 "
                        f"{step_index}（{invocation.kind.value}）"
                        "缺少配方依赖输入: "
                        + "、".join(sorted(missing_inputs))
                    )
        object.__setattr__(self, "request_id", request_id)
        object.__setattr__(self, "generation", generation)
        object.__setattr__(self, "invocations", invocations)
        object.__setattr__(self, "continue_on_error", bool(self.continue_on_error))


@dataclass(frozen=True, slots=True)
class AnalysisBatchItemResult:
    item_id: str
    display_name: str
    success: bool
    execution: AnalysisBatchExecution | None = None
    error_type: str | None = None
    error_message: str | None = None
    executions: tuple[AnalysisBatchExecution, ...] = ()

    def __post_init__(self) -> None:
        normalized_executions = tuple(self.executions)
        if self.execution is not None and not normalized_executions:
            normalized_executions = (self.execution,)
        elif self.execution is None and normalized_executions:
            object.__setattr__(self, "execution", normalized_executions[0])
        elif (
            self.execution is not None
            and normalized_executions
            and self.execution is not normalized_executions[0]
        ):
            raise ValueError("execution 必须是 executions 中的首个结果")
        object.__setattr__(self, "executions", normalized_executions)


@dataclass(frozen=True, slots=True)
class AnalysisBatchResult:
    request_id: str
    generation: int
    recipe_id: str
    item_results: tuple[AnalysisBatchItemResult, ...]
    cancelled: bool = False

    @property
    def success_count(self) -> int:
        return sum(item.success for item in self.item_results)

    @property
    def failure_count(self) -> int:
        return sum(not item.success for item in self.item_results)


@dataclass(frozen=True, slots=True)
class AnalysisBatchProgress:
    request_id: str
    generation: int
    completed: int
    total: int
    item_id: str


def builtin_plane_analysis_recipes() -> tuple[AnalysisRecipe, ...]:
    """Return safe batch recipes that need only one immutable raster plane."""

    return (
        AnalysisRecipe(
            "directionality-v2",
            "纤维方向性 v2",
            AdvancedAnalysisKind.DIRECTIONALITY,
            parameters={"algorithm_version": 2},
            required_inputs=("plane",),
        ),
        AnalysisRecipe(
            "tubeness-v1",
            "Tubeness",
            AdvancedAnalysisKind.TUBENESS,
            required_inputs=("plane",),
        ),
        AnalysisRecipe(
            "glcm-haralick-v1",
            "Haralick GLCM 纹理",
            AdvancedAnalysisKind.GLCM_HARALICK,
            required_inputs=("plane",),
        ),
        AnalysisRecipe(
            "intensity-surface-v1",
            "二维强度表面",
            AdvancedAnalysisKind.INTENSITY_SURFACE,
            required_inputs=("plane",),
        ),
        AnalysisRecipe(
            "intensity-v2",
            "灰度与颜色统计（亮度）",
            BasicAnalysisKind.INTENSITY,
            required_inputs=("plane",),
        ),
        AnalysisRecipe(
            "histogram-v2",
            "直方图（256 分箱）",
            BasicAnalysisKind.HISTOGRAM,
            required_inputs=("plane",),
        ),
        AnalysisRecipe(
            "intensity-and-histogram-v2",
            "灰度统计 + 直方图",
            invocations=(
                AnalysisToolInvocation(
                    BasicAnalysisKind.INTENSITY,
                    required_inputs=("plane",),
                ),
                AnalysisToolInvocation(
                    BasicAnalysisKind.HISTOGRAM,
                    required_inputs=("plane",),
                ),
            ),
        ),
        AnalysisRecipe(
            "fft-power-spectrum-v1",
            "FFT 功率谱",
            BasicAnalysisKind.FFT_POWER_SPECTRUM,
            required_inputs=("plane",),
        ),
        AnalysisRecipe(
            "particles-v2",
            "粒子分析（默认阈值）",
            BasicAnalysisKind.PARTICLES,
            required_inputs=("plane",),
        ),
        AnalysisRecipe(
            "maxima-v1",
            "极值检测（最多 10000 点）",
            BasicAnalysisKind.MAXIMA,
            parameters={"algorithm_version": "1", "max_points": 10_000},
            required_inputs=("plane",),
        ),
        AnalysisRecipe(
            "directionality-and-glcm-v2",
            "方向性 v2 + Haralick GLCM",
            invocations=(
                AnalysisToolInvocation(
                    AdvancedAnalysisKind.DIRECTIONALITY,
                    parameters={"algorithm_version": 2},
                    required_inputs=("plane",),
                ),
                AnalysisToolInvocation(
                    AdvancedAnalysisKind.GLCM_HARALICK,
                    required_inputs=("plane",),
                ),
            ),
        ),
    )


def analysis_step_request_id(
    source_request_id: str,
    step_index: int,
    *,
    step_count: int,
) -> str:
    """Return the stable request id used by one recipe execution.

    A one-step recipe keeps the historical source request id.  Multi-step
    recipes suffix every step, including the first, so callbacks can never be
    accidentally paired with a sibling result from the same frozen source.
    """

    token = str(source_request_id or "").strip()
    if not token:
        raise ValueError("source_request_id 不能为空")
    index = int(step_index)
    count = int(step_count)
    if count < 1 or index < 0 or index >= count:
        raise ValueError("分析配方步骤索引超出范围")
    if count == 1:
        return token
    return f"{token}:step-{index + 1}"


def resolve_basic_analysis_parameters(
    kind: BasicAnalysisKind | str,
    parameters: Mapping[str, object],
    plane: RasterPlane,
) -> dict[str, object]:
    """Resolve per-pixel-type defaults shared by batch execution and commit."""

    normalized_kind = BasicAnalysisKind(kind)
    if not isinstance(plane, RasterPlane):
        raise TypeError("plane 必须是 RasterPlane")
    resolved = dict(parameters)
    if (
        normalized_kind is BasicAnalysisKind.PARTICLES
        and resolved.get("threshold") is None
    ):
        resolved["threshold"] = {
            RasterPixelType.GRAY16: 32767.0,
            RasterPixelType.GRAY32_FLOAT: 0.5,
        }.get(plane.pixel_type, 127.0)
    return resolved


def execute_basic_analysis(
    invocation: BasicAnalysisInvocation,
    *,
    cancellation_token: CancellationToken | None = None,
) -> BasicAnalysisExecution:
    """Execute one registered basic analysis without importing Qt state."""

    if not isinstance(invocation, BasicAnalysisInvocation):
        raise TypeError("invocation 必须是 BasicAnalysisInvocation")
    if cancellation_token is not None:
        cancellation_token.raise_if_cancelled()
    plane = invocation.plane
    if plane is None:
        raise ValueError(f"{invocation.kind.value} 需要输入图像")
    _ensure_basic_resource_limit(invocation, plane)
    parameters = resolve_basic_analysis_parameters(
        invocation.kind,
        invocation.parameters,
        plane,
    )
    image = np.asarray(raster_plane_to_numpy(plane))
    cancellation_check = (
        None
        if cancellation_token is None
        else cancellation_token.raise_if_cancelled
    )

    if invocation.kind is BasicAnalysisKind.INTENSITY:
        selected = _selected_basic_parameters(
            parameters,
            {
                "channel",
                "percentile_levels",
                "threshold_low",
                "threshold_high",
            },
        )
        result = analyze_intensity(
            IntensityAnalysisRequest(
                image=image,
                roi_mask=invocation.roi_mask,
                rings=invocation.raw_rings,
                channel=str(selected.get("channel", "luminance")),
                percentile_levels=tuple(
                    selected.get(
                        "percentile_levels",
                        (10.0, 25.0, 50.0, 75.0, 90.0),
                    )
                ),
                threshold_low=selected.get("threshold_low"),  # type: ignore[arg-type]
                threshold_high=selected.get("threshold_high"),  # type: ignore[arg-type]
                request_id=invocation.request_id,
                generation=invocation.generation,
            ),
            cancellation_check=cancellation_check,
        )
    elif invocation.kind is BasicAnalysisKind.HISTOGRAM:
        selected = _selected_basic_parameters(
            parameters,
            {"channel", "bins", "value_range", "log_counts"},
        )
        value_range = selected.get("value_range")
        result = calculate_histogram(
            HistogramRequest(
                image=image,
                roi_mask=invocation.roi_mask,
                rings=invocation.raw_rings,
                channel=str(selected.get("channel", "luminance")),
                bins=int(selected.get("bins", 256)),
                value_range=(
                    None
                    if value_range is None
                    else (
                        float(value_range[0]),  # type: ignore[index]
                        float(value_range[1]),  # type: ignore[index]
                    )
                ),
                log_counts=bool(selected.get("log_counts", False)),
                request_id=invocation.request_id,
                generation=invocation.generation,
            ),
            cancellation_check=cancellation_check,
        )
    elif invocation.kind is BasicAnalysisKind.FFT_POWER_SPECTRUM:
        selected = _selected_basic_parameters(
            parameters,
            {"channel", "logarithmic", "centered", "window", "tukey_alpha"},
        )
        result = calculate_fft_power_spectrum(
            FftPowerSpectrumRequest(
                image=image,
                roi_mask=invocation.roi_mask,
                rings=invocation.raw_rings,
                channel=str(selected.get("channel", "luminance")),
                logarithmic=bool(selected.get("logarithmic", True)),
                centered=bool(selected.get("centered", True)),
                window=str(selected.get("window", "none")),
                tukey_alpha=float(selected.get("tukey_alpha", 0.25)),
                request_id=invocation.request_id,
                generation=invocation.generation,
            ),
            cancellation_check=cancellation_check,
        )
    elif invocation.kind is BasicAnalysisKind.PARTICLES:
        selected = _selected_basic_parameters(
            parameters,
            {
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
            },
        )
        mask = _basic_particle_mask(invocation, image, selected)
        result = analyze_particles(
            ParticleAnalysisRequest(
                mask=mask,
                connectivity=int(selected.get("connectivity", 8)),
                min_area_px=int(selected.get("min_area_px", 1)),
                max_area_px=(
                    None
                    if selected.get("max_area_px") is None
                    else int(selected["max_area_px"])
                ),
                min_circularity=float(selected.get("min_circularity", 0.0)),
                max_circularity=float(selected.get("max_circularity", 1.0)),
                include_holes=bool(selected.get("include_holes", False)),
                exclude_edge=bool(selected.get("exclude_edge", False)),
                watershed=bool(selected.get("watershed", False)),
                watershed_min_distance=int(
                    selected.get("watershed_min_distance", 3)
                ),
                pixel_size_x=invocation.pixel_size_x,
                pixel_size_y=invocation.pixel_size_y,
                unit=invocation.unit,
                request_id=invocation.request_id,
                generation=invocation.generation,
            ),
            cancellation_check=cancellation_check,
        )
    elif invocation.kind is BasicAnalysisKind.MAXIMA:
        selected = _selected_basic_parameters(
            parameters,
            {
                "channel",
                "minimum_value",
                "prominence",
                "neighborhood_radius",
                "min_distance",
                "exclude_edge",
                "max_points",
                "algorithm_version",
            },
        )
        result = find_local_maxima(
            FindMaximaRequest(
                image=image,
                roi_mask=_basic_combined_mask(
                    invocation,
                    image.shape[:2],
                ),
                channel=str(selected.get("channel", "luminance")),
                minimum_value=selected.get("minimum_value"),  # type: ignore[arg-type]
                prominence=float(selected.get("prominence", 0.0)),
                neighborhood_radius=int(
                    selected.get("neighborhood_radius", 1)
                ),
                min_distance=float(selected.get("min_distance", 1.0)),
                exclude_edge=bool(selected.get("exclude_edge", False)),
                max_points=(
                    None
                    if selected.get("max_points") is None
                    else int(selected["max_points"])
                ),
                algorithm_version=str(selected.get("algorithm_version", "1")),
                request_id=invocation.request_id,
                generation=invocation.generation,
            ),
            cancellation_check=cancellation_check,
        )
    else:  # pragma: no cover - exhaustive enum guard
        raise ValueError(f"不支持的基础批量分析：{invocation.kind.value}")

    if cancellation_token is not None:
        cancellation_token.raise_if_cancelled()
    result_request_id = getattr(result, "request_id", invocation.request_id)
    result_generation = getattr(result, "generation", invocation.generation)
    if (
        str(result_request_id) != invocation.request_id
        or int(result_generation) != invocation.generation
    ):
        raise RuntimeError("基础分析结果的 request_id/generation 与请求不一致")
    scalars, arrays = _describe_basic_analysis_result(result)
    version = _BASIC_TOOL_VERSIONS[invocation.kind]
    name = _BASIC_TOOL_NAMES[invocation.kind]
    return BasicAnalysisExecution(
        kind=invocation.kind,
        chinese_name=name,
        algorithm_version=version,
        request_id=invocation.request_id,
        generation=invocation.generation,
        tool_spec=_basic_analysis_tool_spec(
            invocation.kind,
            version=version,
            chinese_name=name,
        ),
        result=result,
        scalar_report=scalars,
        arrays=arrays,
    )


def _basic_analysis_tool_spec(
    kind: BasicAnalysisKind,
    *,
    version: str,
    chinese_name: str,
) -> AnalysisToolSpec:
    return AnalysisToolSpec(
        tool_id=f"fdm.{kind.value}",
        version=version,
        chinese_name=chinese_name,
        parameter_schema={
            "type": "object",
            "schema_id": f"fdm.{kind.value}.parameters.v{version}",
        },
        output_schema={
            "type": "object",
            "schema_id": f"fdm.{kind.value}.output.v{version}",
        },
        convertible_kinds=("image", "roi"),
    )


def _ensure_basic_resource_limit(
    invocation: BasicAnalysisInvocation,
    plane: RasterPlane,
) -> None:
    pixels = int(plane.width) * int(plane.height)
    estimated = (
        int(plane.byte_count)
        + pixels * _BASIC_RESOURCE_MULTIPLIERS[invocation.kind]
    )
    if invocation.roi_mask is not None:
        estimated += int(invocation.roi_mask.nbytes)
    if invocation.binary_mask is not None:
        estimated += int(invocation.binary_mask.nbytes)
    if estimated > _MAX_BASIC_ANALYSIS_WORKING_BYTES:
        raise MemoryError(
            f"预计分析工作集 {estimated / (1 << 20):.1f} MiB 超过 "
            f"{_MAX_BASIC_ANALYSIS_WORKING_BYTES / (1 << 20):.0f} MiB "
            "安全上限；请裁剪图片或分区分析。"
        )


def _selected_basic_parameters(
    parameters: Mapping[str, object],
    allowed: set[str],
) -> dict[str, Any]:
    unknown = sorted(set(parameters) - allowed)
    if unknown:
        raise ValueError(f"基础批量分析包含未知参数：{'、'.join(unknown)}")
    return dict(parameters)


def _basic_particle_mask(
    invocation: BasicAnalysisInvocation,
    image: NDArray[Any],
    parameters: Mapping[str, object],
) -> NDArray[np.bool_]:
    if invocation.binary_mask is not None:
        result = np.asarray(invocation.binary_mask, dtype=bool).copy()
    else:
        scalar = _basic_scalar_image(
            image,
            str(parameters.get("channel", "luminance")),
        )
        threshold = float(parameters["threshold"])
        if not math.isfinite(threshold):
            raise ValueError("二值阈值必须是有限数")
        foreground = str(parameters.get("foreground", "above")).strip().lower()
        foreground = {
            "bright": "above",
            "dark": "below",
        }.get(foreground, foreground)
        if foreground == "above":
            result = np.asarray(scalar >= threshold, dtype=bool)
        elif foreground == "below":
            result = np.asarray(scalar <= threshold, dtype=bool)
        else:
            raise ValueError("foreground 只能是 above 或 below")
    selection = _basic_combined_mask(invocation, result.shape)
    if selection is not None:
        result &= selection
    result = np.ascontiguousarray(result, dtype=bool)
    result.setflags(write=False)
    return result


def _basic_combined_mask(
    invocation: BasicAnalysisInvocation,
    shape: tuple[int, int],
) -> NDArray[np.bool_] | None:
    mask: NDArray[np.bool_] | None = None
    if invocation.raw_rings:
        mask = np.zeros(shape, dtype=bool)
        for ring in invocation.raw_rings:
            if len(ring) < 3:
                continue
            contour = np.rint(np.asarray(ring, dtype=np.float64)).astype(
                np.int32
            )
            temporary = np.zeros(shape, dtype=np.uint8)
            cv2.fillPoly(temporary, [contour], 1)
            mask ^= temporary.astype(bool)
    if invocation.roi_mask is not None:
        mask = (
            np.asarray(invocation.roi_mask, dtype=bool).copy()
            if mask is None
            else mask & invocation.roi_mask
        )
    if mask is not None:
        mask = np.ascontiguousarray(mask, dtype=bool)
        mask.setflags(write=False)
    return mask


def _basic_scalar_image(
    image: NDArray[Any],
    channel: str,
) -> NDArray[Any]:
    if image.ndim == 2:
        return image
    normalized_channel = str(channel).strip().casefold()
    channel_indices = {
        "red": 0,
        "r": 0,
        "green": 1,
        "g": 1,
        "blue": 2,
        "b": 2,
    }
    if normalized_channel in channel_indices:
        return image[..., channel_indices[normalized_channel]]
    color = image[..., :3].astype(np.float64)
    if normalized_channel in {"average", "mean", "平均"}:
        return np.mean(color, axis=2)
    if normalized_channel not in {"luminance", "gray", "grey", "亮度"}:
        raise ValueError("通道只支持 luminance、average、red、green 或 blue")
    return (
        color[..., 0] * 0.2126
        + color[..., 1] * 0.7152
        + color[..., 2] * 0.0722
    )


def _describe_basic_analysis_result(
    result: object,
) -> tuple[
    tuple[tuple[str, str | int | float | bool | None], ...],
    tuple[AdvancedArrayDescriptor, ...],
]:
    scalar_rows: list[tuple[str, str | int | float | bool | None]] = []
    arrays: list[AdvancedArrayDescriptor] = []
    if not is_dataclass(result):
        return (("结果类型", type(result).__name__),), ()
    for field_info in fields(result):
        value = getattr(result, field_info.name)
        if isinstance(value, np.ndarray):
            arrays.append(
                AdvancedArrayDescriptor(
                    name=field_info.name,
                    shape=tuple(int(item) for item in value.shape),
                    dtype=str(value.dtype),
                    byte_count=int(value.nbytes),
                )
            )
        elif isinstance(value, (str, int, float, bool)) or value is None:
            scalar_rows.append((field_info.name, value))
        elif isinstance(value, tuple) and len(value) <= 16 and all(
            isinstance(item, (str, int, float, bool)) or item is None
            for item in value
        ):
            scalar_rows.append(
                (
                    field_info.name,
                    json.dumps(
                        value,
                        ensure_ascii=False,
                        allow_nan=False,
                    ),
                )
            )
    return tuple(scalar_rows), tuple(arrays)


def _freeze_basic_mask(
    mask: NDArray[np.bool_] | None,
    expected_shape: tuple[int, int] | None,
    label: str,
) -> NDArray[np.bool_] | None:
    if mask is None:
        return None
    normalized = np.array(mask, dtype=bool, copy=True, order="C")
    if normalized.ndim != 2:
        raise ValueError(f"{label} 必须是二维数组")
    if expected_shape is not None and normalized.shape != expected_shape:
        raise ValueError(
            f"{label} 尺寸 {normalized.shape} 与图像尺寸 {expected_shape} 不一致"
        )
    normalized.setflags(write=False)
    return normalized


def _positive_finite(value: object, label: str) -> float:
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"{label} 必须是正有限数")
    return normalized


def _finite_coordinate(value: object, label: str) -> float:
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{label} 必须是有限数")
    return normalized


def execute_analysis_batch(
    request: AnalysisBatchRequest,
    *,
    registry: AdvancedAnalysisRegistry | None = None,
    cancellation_token: CancellationToken | None = None,
    limits: AdvancedAnalysisLimits = DEFAULT_ADVANCED_ANALYSIS_LIMITS,
    progress: Callable[[AnalysisBatchProgress], None] | None = None,
) -> AnalysisBatchResult:
    """Execute sources and recipe steps sequentially.

    A source item is appended to the staged result only after every recipe step
    succeeds.  An exception in a later step therefore cannot leak partial
    executions for that source.
    """

    if not isinstance(request, AnalysisBatchRequest):
        raise TypeError("request 必须是 AnalysisBatchRequest")
    active_registry = registry or AdvancedAnalysisRegistry()
    staged: list[AnalysisBatchItemResult] = []
    cancelled = False
    for index, item in enumerate(request.invocations, start=1):
        try:
            if cancellation_token is not None:
                cancellation_token.raise_if_cancelled()
            executions: list[AnalysisBatchExecution] = []
            for step_index, tool_invocation in enumerate(
                request.recipe.invocations,
            ):
                if cancellation_token is not None:
                    cancellation_token.raise_if_cancelled()
                merged_parameters = dict(tool_invocation.parameters)
                # Preserve the old per-source override contract for the first
                # step.  Later steps are fully described by their immutable
                # recipe invocation and share only the frozen source inputs.
                if step_index == 0:
                    merged_parameters.update(item.analysis.parameters)
                step_request_id = analysis_step_request_id(
                    item.analysis.request_id,
                    step_index,
                    step_count=len(request.recipe.invocations),
                )
                if isinstance(tool_invocation.kind, BasicAnalysisKind):
                    invocation = BasicAnalysisInvocation(
                        tool_invocation.kind,
                        request_id=step_request_id,
                        generation=item.analysis.generation,
                        plane=item.analysis.plane,
                        roi_mask=item.analysis.roi_mask,
                        raw_rings=getattr(item.analysis, "raw_rings", ()),
                        binary_mask=item.analysis.binary_mask,
                        points=item.analysis.points,
                        pixel_size_x=item.analysis.pixel_size_x,
                        pixel_size_y=item.analysis.pixel_size_y,
                        unit=item.analysis.unit,
                        parameters=merged_parameters,
                    )
                    executions.append(
                        execute_basic_analysis(
                            invocation,
                            cancellation_token=cancellation_token,
                        )
                    )
                else:
                    invocation = AdvancedAnalysisInvocation(
                        tool_invocation.kind,
                        request_id=step_request_id,
                        generation=item.analysis.generation,
                        plane=item.analysis.plane,
                        roi_mask=item.analysis.roi_mask,
                        binary_mask=item.analysis.binary_mask,
                        points=item.analysis.points,
                        pixel_size_x=item.analysis.pixel_size_x,
                        pixel_size_y=item.analysis.pixel_size_y,
                        unit=item.analysis.unit,
                        parameters=merged_parameters,
                    )
                    executions.append(
                        active_registry.execute(
                            invocation,
                            cancellation_token=cancellation_token,
                            limits=limits,
                        )
                    )
            staged.append(
                AnalysisBatchItemResult(
                    item_id=item.item_id,
                    display_name=item.display_name,
                    success=True,
                    execution=executions[0],
                    executions=tuple(executions),
                )
            )
        except CancellationError:
            cancelled = True
            break
        except Exception as exc:
            staged.append(
                AnalysisBatchItemResult(
                    item_id=item.item_id,
                    display_name=item.display_name,
                    success=False,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
            )
            if not request.continue_on_error:
                break
        if progress is not None:
            progress(
                AnalysisBatchProgress(
                    request_id=request.request_id,
                    generation=request.generation,
                    completed=index,
                    total=len(request.invocations),
                    item_id=item.item_id,
                )
            )
    return AnalysisBatchResult(
        request_id=request.request_id,
        generation=request.generation,
        recipe_id=request.recipe.recipe_id,
        item_results=tuple(staged),
        cancelled=cancelled,
    )


# Concise public names used by recipe/batch callers.  The prefixed names remain
# available for clarity at mixed image-processing/analysis call sites.
Invocation = AnalysisInvocation
BatchRequest = AnalysisBatchRequest
ItemResult = AnalysisBatchItemResult
Result = AnalysisBatchResult


__all__ = [
    "AnalysisBatchExecution",
    "AnalysisBatchItemResult",
    "AnalysisBatchProgress",
    "AnalysisBatchRequest",
    "AnalysisBatchResult",
    "AnalysisInvocation",
    "AnalysisRecipe",
    "AnalysisSourceKind",
    "AnalysisToolInvocation",
    "AnalysisViewport",
    "BatchRequest",
    "BasicAnalysisExecution",
    "BasicAnalysisInvocation",
    "BasicAnalysisKind",
    "Invocation",
    "ItemResult",
    "Result",
    "analysis_step_request_id",
    "builtin_plane_analysis_recipes",
    "execute_basic_analysis",
    "execute_analysis_batch",
    "resolve_basic_analysis_parameters",
]
