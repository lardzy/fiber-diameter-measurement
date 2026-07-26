"""高级二维图像分析内核。

本模块只包含无界面的确定性算法。请求会复制并冻结输入数组，结果同样不暴露
可写数组，因此可以安全地交给带 ``generation`` 的后台任务执行。所有需要较大
内存或可能退化为高复杂度的算法都提供执行前资源估算，并在算法内部继续执行
动态工作量检查。

坐标约定
--------

* 图像坐标的 ``x`` 向右、``y`` 向下。
* 对外报告的方向角使用数学方向：0° 向右，逆时针为正。
* 纤维方向是轴向量，范围为 ``[0°, 180°)``，不区分首尾。
* 局部厚度采用“最大内切圆传播”定义。距离变换只用于生成候选圆，最终每个
  前景像素的厚度是覆盖它的最大内切圆直径，绝不是逐点 ``2×EDT``。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import heapq
import math
from typing import Any, Iterable, Sequence, TypeAlias

import cv2
import numpy as np
from numpy.typing import NDArray

from fdm.cancellation import CancellationToken


Coordinate: TypeAlias = tuple[float, float]


class AdvancedAnalysisKind(StrEnum):
    DIRECTIONALITY = "directionality"
    SKELETON_NETWORK = "skeleton_network"
    LOCAL_THICKNESS = "local_thickness"
    TUBENESS = "tubeness"
    GLCM_HARALICK = "glcm_haralick"
    SPATIAL_DISTRIBUTION = "spatial_distribution"
    INTENSITY_SURFACE = "intensity_surface"


class AdvancedAnalysisErrorCode(StrEnum):
    INVALID_INPUT = "invalid_input"
    NON_FINITE_INPUT = "non_finite_input"
    EMPTY_SELECTION = "empty_selection"
    RESOURCE_LIMIT = "resource_limit"
    UNSUPPORTED_CONFIGURATION = "unsupported_configuration"


class AdvancedAnalysisError(RuntimeError):
    """带机器可读代码和中文说明的高级分析错误。"""

    def __init__(
        self,
        code: AdvancedAnalysisErrorCode,
        message: str,
        *,
        details: Iterable[tuple[str, str | int | float]] = (),
    ) -> None:
        super().__init__(str(message))
        self.code = code
        self.message = str(message)
        self.details = tuple((str(key), value) for key, value in details)

    def __str__(self) -> str:
        return self.message


@dataclass(frozen=True, slots=True)
class AdvancedAnalysisLimits:
    max_working_bytes: int = 1 << 30
    max_work_units: int = 1_000_000_000
    max_local_thickness_work_units: int = 200_000_000
    max_output_values: int = 16_000_000
    max_skeleton_pixels: int = 2_000_000
    max_local_thickness_centers: int = 250_000

    def __post_init__(self) -> None:
        for name in (
            "max_working_bytes",
            "max_work_units",
            "max_local_thickness_work_units",
            "max_output_values",
            "max_skeleton_pixels",
            "max_local_thickness_centers",
        ):
            value = int(getattr(self, name))
            if value < 1:
                raise AdvancedAnalysisError(
                    AdvancedAnalysisErrorCode.INVALID_INPUT,
                    f"资源限制 {name} 必须为正整数。",
                )
            object.__setattr__(self, name, value)


DEFAULT_ADVANCED_ANALYSIS_LIMITS = AdvancedAnalysisLimits()


@dataclass(frozen=True, slots=True)
class AdvancedAnalysisResourceEstimate:
    operation: AdvancedAnalysisKind
    input_pixels: int
    estimated_peak_bytes: int
    estimated_work_units: int
    estimated_output_values: int
    allowed: bool
    reason: str = ""


@dataclass(frozen=True, slots=True)
class DirectionalityRequest:
    image: NDArray[Any]
    roi_mask: NDArray[np.bool_] | None = None
    bins: int = 180
    gradient_sigma: float = 1.0
    minimum_gradient: float = 0.0
    histogram_smoothing_bins: float = 1.0
    peak_min_fraction: float = 0.1
    max_peaks: int = 8
    request_id: str = ""
    generation: int = 0

    def __post_init__(self) -> None:
        image = _freeze_scalar_image(self.image, name="方向性图像")
        object.__setattr__(self, "image", image)
        object.__setattr__(
            self,
            "roi_mask",
            _freeze_optional_mask(self.roi_mask, image.shape),
        )
        if not 4 <= int(self.bins) <= 4096:
            _invalid("方向性直方图的区间数必须在 4 到 4096 之间。")
        _require_finite_nonnegative("梯度平滑标准差", self.gradient_sigma)
        _require_finite_nonnegative("最小梯度", self.minimum_gradient)
        _require_finite_nonnegative("直方图平滑宽度", self.histogram_smoothing_bins)
        fraction = _finite_float("方向峰值阈值比例", self.peak_min_fraction)
        if not 0.0 <= fraction <= 1.0:
            _invalid("方向峰值阈值比例必须在 0 到 1 之间。")
        if int(self.max_peaks) < 1:
            _invalid("最多方向峰数量必须至少为 1。")
        object.__setattr__(self, "bins", int(self.bins))
        object.__setattr__(self, "gradient_sigma", float(self.gradient_sigma))
        object.__setattr__(self, "minimum_gradient", float(self.minimum_gradient))
        object.__setattr__(
            self,
            "histogram_smoothing_bins",
            float(self.histogram_smoothing_bins),
        )
        object.__setattr__(self, "peak_min_fraction", fraction)
        object.__setattr__(self, "max_peaks", int(self.max_peaks))
        object.__setattr__(self, "request_id", str(self.request_id))
        object.__setattr__(self, "generation", int(self.generation))


@dataclass(frozen=True, slots=True)
class OrientationPeak:
    angle_degrees: float
    weight: float
    relative_weight: float
    bin_index: int


@dataclass(frozen=True, slots=True)
class DirectionalityResult:
    bin_centers_degrees: tuple[float, ...]
    histogram_weights: tuple[float, ...]
    normalized_weights: tuple[float, ...]
    peaks: tuple[OrientationPeak, ...]
    valid_gradient_pixels: int
    total_weight: float
    convention: str = "0°向右，逆时针为正，轴向范围[0°,180°)"
    request_id: str = ""
    generation: int = 0


@dataclass(frozen=True, slots=True)
class SkeletonNetworkRequest:
    mask: NDArray[np.bool_]
    already_skeletonized: bool = False
    pixel_size_x: float = 1.0
    pixel_size_y: float = 1.0
    unit: str = "px"
    request_id: str = ""
    generation: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "mask", _freeze_binary_mask(self.mask, "骨架掩膜"))
        _require_finite_positive("横向像素尺寸", self.pixel_size_x)
        _require_finite_positive("纵向像素尺寸", self.pixel_size_y)
        object.__setattr__(self, "already_skeletonized", bool(self.already_skeletonized))
        object.__setattr__(self, "pixel_size_x", float(self.pixel_size_x))
        object.__setattr__(self, "pixel_size_y", float(self.pixel_size_y))
        object.__setattr__(self, "unit", str(self.unit or "px"))
        object.__setattr__(self, "request_id", str(self.request_id))
        object.__setattr__(self, "generation", int(self.generation))


@dataclass(frozen=True, slots=True)
class SkeletonBranch:
    start_px: Coordinate | None
    end_px: Coordinate | None
    length: float
    closed: bool = False


@dataclass(frozen=True, slots=True)
class SkeletonNetworkResult:
    skeleton: NDArray[np.bool_]
    endpoint_coordinates_px: tuple[Coordinate, ...]
    branchpoint_coordinates_px: tuple[Coordinate, ...]
    branches: tuple[SkeletonBranch, ...]
    endpoint_count: int
    branchpoint_count: int
    connected_component_count: int
    isolated_point_count: int
    loop_count: int
    total_length: float
    maximum_geodesic_distance: float
    unit: str
    request_id: str = ""
    generation: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "skeleton", _freeze_bool_output(self.skeleton))


@dataclass(frozen=True, slots=True)
class LocalThicknessRequest:
    mask: NDArray[np.bool_]
    request_id: str = ""
    generation: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "mask", _freeze_binary_mask(self.mask, "局部厚度掩膜"))
        object.__setattr__(self, "request_id", str(self.request_id))
        object.__setattr__(self, "generation", int(self.generation))


@dataclass(frozen=True, slots=True)
class MaximalInscribedCircle:
    center_x: int
    center_y: int
    radius_px: float


@dataclass(frozen=True, slots=True)
class LocalThicknessResult:
    thickness_px: NDArray[np.float32]
    maximal_circles: tuple[MaximalInscribedCircle, ...]
    foreground_pixel_count: int
    maximum_thickness_px: float
    mean_thickness_px: float | None
    definition: str = "覆盖该像素的最大内切圆直径（像素中心欧氏距离约定）"
    request_id: str = ""
    generation: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "thickness_px",
            _freeze_float_output(self.thickness_px),
        )


@dataclass(frozen=True, slots=True)
class TubenessRequest:
    image: NDArray[Any]
    roi_mask: NDArray[np.bool_] | None = None
    scales: tuple[float, ...] = (1.0, 2.0, 4.0)
    beta: float = 0.5
    structure_scale: float | None = None
    bright_ridges: bool = True
    request_id: str = ""
    generation: int = 0

    def __post_init__(self) -> None:
        image = _freeze_scalar_image(self.image, name="Tubeness 图像")
        object.__setattr__(self, "image", image)
        object.__setattr__(
            self,
            "roi_mask",
            _freeze_optional_mask(self.roi_mask, image.shape),
        )
        scales = tuple(float(value) for value in self.scales)
        if not scales or len(scales) > 64:
            _invalid("Tubeness 尺度数量必须在 1 到 64 之间。")
        for value in scales:
            _require_finite_positive("Tubeness 尺度", value)
        if len(set(scales)) != len(scales):
            _invalid("Tubeness 尺度不能重复。")
        _require_finite_positive("Tubeness beta", self.beta)
        if self.structure_scale is not None:
            _require_finite_positive("Tubeness 结构响应尺度", self.structure_scale)
        object.__setattr__(self, "scales", tuple(sorted(scales)))
        object.__setattr__(self, "beta", float(self.beta))
        object.__setattr__(
            self,
            "structure_scale",
            None if self.structure_scale is None else float(self.structure_scale),
        )
        object.__setattr__(self, "bright_ridges", bool(self.bright_ridges))
        object.__setattr__(self, "request_id", str(self.request_id))
        object.__setattr__(self, "generation", int(self.generation))


@dataclass(frozen=True, slots=True)
class TubenessResult:
    response: NDArray[np.float32]
    best_scale: NDArray[np.float32]
    scales: tuple[float, ...]
    maximum_response: float
    request_id: str = ""
    generation: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "response", _freeze_float_output(self.response))
        object.__setattr__(self, "best_scale", _freeze_float_output(self.best_scale))


@dataclass(frozen=True, slots=True)
class GlcmHaralickRequest:
    image: NDArray[Any]
    roi_mask: NDArray[np.bool_] | None = None
    levels: int = 32
    distances: tuple[int, ...] = (1,)
    directions_degrees: tuple[float, ...] = (0.0, 45.0, 90.0, 135.0)
    value_range: tuple[float, float] | None = None
    symmetric: bool = True
    request_id: str = ""
    generation: int = 0

    def __post_init__(self) -> None:
        image = _freeze_scalar_image(
            self.image,
            name="GLCM 图像",
            require_finite=False,
        )
        object.__setattr__(self, "image", image)
        object.__setattr__(
            self,
            "roi_mask",
            _freeze_optional_mask(self.roi_mask, image.shape),
        )
        if not 2 <= int(self.levels) <= 256:
            _invalid("GLCM 量化级数必须在 2 到 256 之间。")
        distances = tuple(int(value) for value in self.distances)
        if not distances or len(distances) > 32 or any(value < 1 for value in distances):
            _invalid("GLCM 距离必须包含 1 到 32 个正整数。")
        directions = tuple(float(value) for value in self.directions_degrees)
        if not directions or len(directions) > 64:
            _invalid("GLCM 方向数量必须在 1 到 64 之间。")
        if any(not math.isfinite(value) for value in directions):
            _non_finite("GLCM 方向必须是有限数。")
        normalized_directions = tuple(value % 180.0 for value in directions)
        offsets = {
            _direction_offset(distance, direction)
            for distance in distances
            for direction in normalized_directions
        }
        if len(offsets) != len(distances) * len(directions):
            _invalid("GLCM 距离与方向在像素取整后存在重复偏移，请调整参数。")
        value_range = self.value_range
        if value_range is not None:
            low = _finite_float("GLCM 数值下限", value_range[0])
            high = _finite_float("GLCM 数值上限", value_range[1])
            if high <= low:
                _invalid("GLCM 数值范围的上限必须大于下限。")
            value_range = (low, high)
        object.__setattr__(self, "levels", int(self.levels))
        object.__setattr__(self, "distances", distances)
        object.__setattr__(
            self,
            "directions_degrees",
            normalized_directions,
        )
        object.__setattr__(self, "value_range", value_range)
        object.__setattr__(self, "symmetric", bool(self.symmetric))
        object.__setattr__(self, "request_id", str(self.request_id))
        object.__setattr__(self, "generation", int(self.generation))


@dataclass(frozen=True, slots=True)
class HaralickFeatures:
    distance_px: int
    direction_degrees: float
    pair_count: int
    contrast: float
    dissimilarity: float
    homogeneity: float
    angular_second_moment: float
    energy: float
    correlation: float | None
    entropy: float
    maximum_probability: float
    matrix: NDArray[np.float64]

    def __post_init__(self) -> None:
        object.__setattr__(self, "matrix", _freeze_double_output(self.matrix))


@dataclass(frozen=True, slots=True)
class GlcmHaralickResult:
    features: tuple[HaralickFeatures, ...]
    levels: int
    quantization_range: tuple[float, float]
    symmetric: bool
    valid_pixel_count: int
    non_finite_pixel_count: int
    request_id: str = ""
    generation: int = 0


@dataclass(frozen=True, slots=True)
class SpatialDistributionRequest:
    points: tuple[Coordinate, ...]
    study_area: float | None = None
    pixel_size_x: float = 1.0
    pixel_size_y: float = 1.0
    unit: str = "px"
    request_id: str = ""
    generation: int = 0

    def __post_init__(self) -> None:
        points = _freeze_points(self.points)
        if len(points) < 2:
            _invalid("最近邻分析至少需要两个点。")
        _require_finite_positive("横向像素尺寸", self.pixel_size_x)
        _require_finite_positive("纵向像素尺寸", self.pixel_size_y)
        if self.study_area is not None:
            _require_finite_positive("研究区域面积", self.study_area)
        object.__setattr__(self, "points", points)
        object.__setattr__(
            self,
            "study_area",
            None if self.study_area is None else float(self.study_area),
        )
        object.__setattr__(self, "pixel_size_x", float(self.pixel_size_x))
        object.__setattr__(self, "pixel_size_y", float(self.pixel_size_y))
        object.__setattr__(self, "unit", str(self.unit or "px"))
        object.__setattr__(self, "request_id", str(self.request_id))
        object.__setattr__(self, "generation", int(self.generation))


@dataclass(frozen=True, slots=True)
class SpatialDistributionResult:
    nearest_neighbor_distances: tuple[float, ...]
    nearest_neighbor_indices: tuple[int, ...]
    mean_nearest_neighbor_distance: float
    median_nearest_neighbor_distance: float
    minimum_nearest_neighbor_distance: float
    maximum_nearest_neighbor_distance: float
    study_area: float
    area_source: str
    spatial_density: float
    unit: str
    request_id: str = ""
    generation: int = 0


@dataclass(frozen=True, slots=True)
class IntensitySurfaceRequest:
    image: NDArray[Any]
    roi_mask: NDArray[np.bool_] | None = None
    sample_step_x: int = 1
    sample_step_y: int = 1
    pixel_size_x: float = 1.0
    pixel_size_y: float = 1.0
    unit: str = "px"
    request_id: str = ""
    generation: int = 0

    def __post_init__(self) -> None:
        image = _freeze_scalar_image(
            self.image,
            name="二维强度表面图像",
            require_finite=False,
        )
        object.__setattr__(self, "image", image)
        object.__setattr__(
            self,
            "roi_mask",
            _freeze_optional_mask(self.roi_mask, image.shape),
        )
        if int(self.sample_step_x) < 1 or int(self.sample_step_y) < 1:
            _invalid("二维强度表面的采样步长必须为正整数。")
        _require_finite_positive("横向像素尺寸", self.pixel_size_x)
        _require_finite_positive("纵向像素尺寸", self.pixel_size_y)
        object.__setattr__(self, "sample_step_x", int(self.sample_step_x))
        object.__setattr__(self, "sample_step_y", int(self.sample_step_y))
        object.__setattr__(self, "pixel_size_x", float(self.pixel_size_x))
        object.__setattr__(self, "pixel_size_y", float(self.pixel_size_y))
        object.__setattr__(self, "unit", str(self.unit or "px"))
        object.__setattr__(self, "request_id", str(self.request_id))
        object.__setattr__(self, "generation", int(self.generation))


@dataclass(frozen=True, slots=True)
class IntensitySurfaceResult:
    x_coordinates: tuple[float, ...]
    y_coordinates: tuple[float, ...]
    z_values: tuple[tuple[float | None, ...], ...]
    finite_sample_count: int
    masked_sample_count: int
    non_finite_sample_count: int
    z_minimum: float | None
    z_maximum: float | None
    coordinate_unit: str
    intensity_unit: str = "原始强度"
    request_id: str = ""
    generation: int = 0


AdvancedRequest: TypeAlias = (
    DirectionalityRequest
    | SkeletonNetworkRequest
    | LocalThicknessRequest
    | TubenessRequest
    | GlcmHaralickRequest
    | SpatialDistributionRequest
    | IntensitySurfaceRequest
)


def estimate_advanced_analysis_resources(
    request: AdvancedRequest,
    *,
    limits: AdvancedAnalysisLimits = DEFAULT_ADVANCED_ANALYSIS_LIMITS,
) -> AdvancedAnalysisResourceEstimate:
    """返回保守但不会偷偷降采样的资源估算。"""

    if isinstance(request, DirectionalityRequest):
        pixels = int(request.image.size)
        peak_bytes = pixels * 8 * 7 + request.bins * 8 * 4
        work = pixels * 14
        output = request.bins * 3
        kind = AdvancedAnalysisKind.DIRECTIONALITY
    elif isinstance(request, SkeletonNetworkRequest):
        pixels = int(request.mask.size)
        foreground = int(np.count_nonzero(request.mask))
        peak_bytes = pixels * 12 + foreground * 160
        work = pixels * (12 if not request.already_skeletonized else 2) + foreground * 24
        output = pixels + foreground * 6
        kind = AdvancedAnalysisKind.SKELETON_NETWORK
        if foreground > limits.max_skeleton_pixels:
            return _rejected_estimate(
                kind,
                pixels,
                peak_bytes,
                work,
                output,
                f"骨架前景像素 {foreground:,} 超过上限 {limits.max_skeleton_pixels:,}。",
            )
    elif isinstance(request, LocalThicknessRequest):
        pixels = int(request.mask.size)
        foreground = int(np.count_nonzero(request.mask))
        peak_bytes = pixels * 24 + foreground * 40
        work = pixels * 8 + foreground * max(1, int(math.sqrt(foreground)))
        output = pixels + foreground * 3
        kind = AdvancedAnalysisKind.LOCAL_THICKNESS
        if foreground > limits.max_local_thickness_centers:
            return _rejected_estimate(
                kind,
                pixels,
                peak_bytes,
                work,
                output,
                "局部厚度候选中心数量超过安全上限；请先裁剪 ROI 或分区分析。",
            )
    elif isinstance(request, TubenessRequest):
        pixels = int(request.image.size)
        peak_bytes = pixels * 8 * 11
        work = pixels * 34 * len(request.scales)
        output = pixels * 2
        kind = AdvancedAnalysisKind.TUBENESS
    elif isinstance(request, GlcmHaralickRequest):
        pixels = int(request.image.size)
        matrix_count = len(request.distances) * len(request.directions_degrees)
        peak_bytes = pixels * 10 + request.levels * request.levels * 16
        work = pixels * matrix_count + request.levels * request.levels * matrix_count
        output = request.levels * request.levels * matrix_count
        kind = AdvancedAnalysisKind.GLCM_HARALICK
    elif isinstance(request, SpatialDistributionRequest):
        count = len(request.points)
        pixels = count
        peak_bytes = count * 80 + min(count, 1024) * count * 32
        work = count * count
        output = count * 2
        kind = AdvancedAnalysisKind.SPATIAL_DISTRIBUTION
    elif isinstance(request, IntensitySurfaceRequest):
        height, width = request.image.shape
        output_height = (height + request.sample_step_y - 1) // request.sample_step_y
        output_width = (width + request.sample_step_x - 1) // request.sample_step_x
        output = output_height * output_width
        pixels = int(request.image.size)
        peak_bytes = pixels * 8 + output * 48
        work = output
        kind = AdvancedAnalysisKind.INTENSITY_SURFACE
    else:
        raise TypeError(f"不支持的高级分析请求类型：{type(request).__name__}")

    reason = ""
    if peak_bytes > limits.max_working_bytes:
        reason = (
            f"预计峰值内存 {peak_bytes / (1 << 20):.1f} MiB 超过 "
            f"{limits.max_working_bytes / (1 << 20):.1f} MiB 上限。"
        )
    elif work > limits.max_work_units:
        reason = f"预计工作量 {work:,} 超过 {limits.max_work_units:,} 上限。"
    elif output > limits.max_output_values:
        reason = f"预计输出值 {output:,} 超过 {limits.max_output_values:,} 上限。"
    return AdvancedAnalysisResourceEstimate(
        operation=kind,
        input_pixels=pixels,
        estimated_peak_bytes=int(peak_bytes),
        estimated_work_units=int(work),
        estimated_output_values=int(output),
        allowed=not reason,
        reason=reason,
    )


def analyze_fiber_directionality(
    request: DirectionalityRequest,
    *,
    cancellation_token: CancellationToken | None = None,
    limits: AdvancedAnalysisLimits = DEFAULT_ADVANCED_ANALYSIS_LIMITS,
) -> DirectionalityResult:
    estimate = estimate_advanced_analysis_resources(request, limits=limits)
    _enforce_estimate(estimate)
    _check_cancel(cancellation_token)
    source = np.asarray(request.image, dtype=np.float64)
    mask = (
        np.ones(source.shape, dtype=bool)
        if request.roi_mask is None
        else np.asarray(request.roi_mask, dtype=bool)
    )
    if not np.any(mask):
        _empty("方向性分析的 ROI 中没有像素。")
    if np.any(~np.isfinite(source[mask])):
        _non_finite("方向性分析的 ROI 中包含 NaN 或 Inf，请先修复非有限像素。")
    if request.gradient_sigma > 0:
        source = cv2.GaussianBlur(
            source,
            (0, 0),
            sigmaX=request.gradient_sigma,
            sigmaY=request.gradient_sigma,
            borderType=cv2.BORDER_REFLECT_101,
        )
    _check_cancel(cancellation_token)
    gradient_x = cv2.Sobel(
        source,
        cv2.CV_64F,
        1,
        0,
        ksize=3,
        borderType=cv2.BORDER_REFLECT_101,
    )
    gradient_y_image = cv2.Sobel(
        source,
        cv2.CV_64F,
        0,
        1,
        ksize=3,
        borderType=cv2.BORDER_REFLECT_101,
    )
    magnitude = np.hypot(gradient_x, gradient_y_image)
    valid = mask & np.isfinite(magnitude) & (magnitude > request.minimum_gradient)
    valid_count = int(np.count_nonzero(valid))
    bin_width = 180.0 / request.bins
    centers = (np.arange(request.bins, dtype=np.float64) + 0.5) * bin_width
    if valid_count == 0:
        zeros = tuple(0.0 for _ in range(request.bins))
        return DirectionalityResult(
            bin_centers_degrees=tuple(float(value) for value in centers),
            histogram_weights=zeros,
            normalized_weights=zeros,
            peaks=(),
            valid_gradient_pixels=0,
            total_weight=0.0,
            request_id=request.request_id,
            generation=request.generation,
        )
    # atan2 的 y 分量取反，将向下的图像 y 轴转换为向上的数学 y 轴。
    gradient_angle = np.arctan2(-gradient_y_image[valid], gradient_x[valid])
    fiber_angle_degrees = np.mod(np.degrees(gradient_angle + math.pi / 2.0), 180.0)
    histogram, _edges = np.histogram(
        fiber_angle_degrees,
        bins=request.bins,
        range=(0.0, 180.0),
        weights=magnitude[valid],
    )
    histogram = histogram.astype(np.float64)
    if request.histogram_smoothing_bins > 0:
        histogram = _smooth_circular_histogram(
            histogram,
            sigma=request.histogram_smoothing_bins,
        )
    total_weight = float(np.sum(histogram, dtype=np.float64))
    normalized = (
        histogram / total_weight
        if total_weight > 0
        else np.zeros_like(histogram)
    )
    peaks = _directionality_peaks(
        histogram,
        centers,
        minimum_fraction=request.peak_min_fraction,
        maximum_count=request.max_peaks,
    )
    _check_cancel(cancellation_token)
    return DirectionalityResult(
        bin_centers_degrees=tuple(float(value) for value in centers),
        histogram_weights=tuple(float(value) for value in histogram),
        normalized_weights=tuple(float(value) for value in normalized),
        peaks=peaks,
        valid_gradient_pixels=valid_count,
        total_weight=total_weight,
        request_id=request.request_id,
        generation=request.generation,
    )


def analyze_skeleton_network(
    request: SkeletonNetworkRequest,
    *,
    cancellation_token: CancellationToken | None = None,
    limits: AdvancedAnalysisLimits = DEFAULT_ADVANCED_ANALYSIS_LIMITS,
) -> SkeletonNetworkResult:
    estimate = estimate_advanced_analysis_resources(request, limits=limits)
    _enforce_estimate(estimate)
    _check_cancel(cancellation_token)
    if request.already_skeletonized:
        skeleton = np.asarray(request.mask, dtype=bool).copy()
    else:
        skeleton = _skeletonize_with_cancellation(
            np.asarray(request.mask, dtype=bool),
            cancellation_token=cancellation_token,
            max_work_units=limits.max_work_units,
        )
    _check_cancel(cancellation_token)
    coordinates_yx = np.argwhere(skeleton)
    count = int(coordinates_yx.shape[0])
    if count == 0:
        return SkeletonNetworkResult(
            skeleton=skeleton,
            endpoint_coordinates_px=(),
            branchpoint_coordinates_px=(),
            branches=(),
            endpoint_count=0,
            branchpoint_count=0,
            connected_component_count=0,
            isolated_point_count=0,
            loop_count=0,
            total_length=0.0,
            maximum_geodesic_distance=0.0,
            unit=request.unit,
            request_id=request.request_id,
            generation=request.generation,
        )
    height, width = skeleton.shape
    vertex_by_linear = {
        int(y) * width + int(x): index
        for index, (y, x) in enumerate(coordinates_yx)
    }
    adjacency: list[list[tuple[int, float]]] = [[] for _ in range(count)]
    edge_count = 0
    total_length = 0.0
    half_directions = ((0, 1), (1, -1), (1, 0), (1, 1))
    for vertex, (raw_y, raw_x) in enumerate(coordinates_yx):
        y = int(raw_y)
        x = int(raw_x)
        for dy, dx in half_directions:
            ny = y + dy
            nx = x + dx
            if not (0 <= ny < height and 0 <= nx < width and skeleton[ny, nx]):
                continue
            if dy != 0 and dx != 0:
                # 若斜边两侧存在正交骨架像素，则斜边只是 8 邻域产生的角部捷径。
                if skeleton[y, nx] or skeleton[ny, x]:
                    continue
            neighbor = vertex_by_linear[ny * width + nx]
            weight = math.hypot(
                dx * request.pixel_size_x,
                dy * request.pixel_size_y,
            )
            adjacency[vertex].append((neighbor, weight))
            adjacency[neighbor].append((vertex, weight))
            edge_count += 1
            total_length += weight
    degrees = np.asarray([len(neighbors) for neighbors in adjacency], dtype=np.int32)
    endpoint_vertices = tuple(int(value) for value in np.flatnonzero(degrees == 1))
    isolated_vertices = tuple(int(value) for value in np.flatnonzero(degrees == 0))
    branch_vertices = set(int(value) for value in np.flatnonzero(degrees >= 3))
    branch_clusters = _connected_vertex_clusters(branch_vertices, adjacency)
    branch_cluster_by_vertex = {
        vertex: cluster_index
        for cluster_index, cluster in enumerate(branch_clusters)
        for vertex in cluster
    }
    branchpoint_coordinates = tuple(
        (
            float(np.mean(coordinates_yx[list(cluster), 1])),
            float(np.mean(coordinates_yx[list(cluster), 0])),
        )
        for cluster in branch_clusters
    )
    endpoint_coordinates = tuple(
        (float(coordinates_yx[index, 1]), float(coordinates_yx[index, 0]))
        for index in endpoint_vertices
    )
    component_labels, components = _graph_components(adjacency)
    loop_count = max(0, edge_count - count + len(components))
    branches = _trace_skeleton_branches(
        coordinates_yx,
        adjacency,
        endpoint_vertices=endpoint_vertices,
        branch_clusters=branch_clusters,
        branch_cluster_by_vertex=branch_cluster_by_vertex,
        components=components,
    )
    _check_cancel(cancellation_token)
    maximum_geodesic = _maximum_skeleton_geodesic(
        adjacency,
        components,
        endpoint_vertices,
        component_labels,
        cancellation_token=cancellation_token,
        max_work_units=limits.max_work_units,
    )
    return SkeletonNetworkResult(
        skeleton=skeleton,
        endpoint_coordinates_px=endpoint_coordinates,
        branchpoint_coordinates_px=branchpoint_coordinates,
        branches=branches,
        endpoint_count=len(endpoint_vertices),
        branchpoint_count=len(branch_clusters),
        connected_component_count=len(components),
        isolated_point_count=len(isolated_vertices),
        loop_count=loop_count,
        total_length=float(total_length),
        maximum_geodesic_distance=float(maximum_geodesic),
        unit=request.unit,
        request_id=request.request_id,
        generation=request.generation,
    )


def calculate_local_thickness(
    request: LocalThicknessRequest,
    *,
    cancellation_token: CancellationToken | None = None,
    limits: AdvancedAnalysisLimits = DEFAULT_ADVANCED_ANALYSIS_LIMITS,
) -> LocalThicknessResult:
    """以最大内切圆传播法计算二维局部厚度。

    候选圆按半径从大到小处理；若一个圆完全包含在已接受的更大圆中，它不是
    最大内切圆。随后将每个最大圆的直径传播给其覆盖的全部前景像素，并对重叠
    区域取最大值。
    """

    estimate = estimate_advanced_analysis_resources(request, limits=limits)
    _enforce_estimate(estimate)
    _check_cancel(cancellation_token)
    foreground = np.asarray(request.mask, dtype=bool)
    foreground_count = int(np.count_nonzero(foreground))
    if foreground_count == 0:
        return LocalThicknessResult(
            thickness_px=np.zeros(foreground.shape, dtype=np.float32),
            maximal_circles=(),
            foreground_pixel_count=0,
            maximum_thickness_px=0.0,
            mean_thickness_px=None,
            request_id=request.request_id,
            generation=request.generation,
        )
    # 一圈背景保证触边对象也按有限图像边界计算，而不是被 OpenCV 当作无限前景。
    padded = np.pad(foreground.astype(np.uint8), 1, mode="constant")
    distance = cv2.distanceTransform(
        padded,
        cv2.DIST_L2,
        cv2.DIST_MASK_PRECISE,
    )[1:-1, 1:-1].astype(np.float64)
    ys, xs = np.nonzero(foreground)
    radii = distance[ys, xs]
    order = np.argsort(-radii, kind="stable")
    accepted_x = np.empty(foreground_count, dtype=np.int32)
    accepted_y = np.empty(foreground_count, dtype=np.int32)
    accepted_radius = np.empty(foreground_count, dtype=np.float64)
    accepted_count = 0
    containment_work = 0
    tolerance = 1e-6
    for rank, candidate_index in enumerate(order):
        if rank % 256 == 0:
            _check_cancel(cancellation_token)
        x = int(xs[candidate_index])
        y = int(ys[candidate_index])
        radius = float(radii[candidate_index])
        contained = False
        if accepted_count:
            ax = accepted_x[:accepted_count]
            ay = accepted_y[:accepted_count]
            ar = accepted_radius[:accepted_count]
            eligible = ar + tolerance >= radius
            containment_work += accepted_count
            if containment_work > limits.max_local_thickness_work_units:
                _resource_limit(
                    "局部厚度的最大内切圆筛选工作量超过安全上限；"
                    "请裁剪 ROI 或提高显式资源上限。",
                    operation=AdvancedAnalysisKind.LOCAL_THICKNESS,
                )
            if np.any(eligible):
                distance_to_centers = np.hypot(ax[eligible] - x, ay[eligible] - y)
                contained = bool(
                    np.any(distance_to_centers + radius <= ar[eligible] + tolerance)
                )
        if not contained:
            accepted_x[accepted_count] = x
            accepted_y[accepted_count] = y
            accepted_radius[accepted_count] = radius
            accepted_count += 1
    _check_cancel(cancellation_token)
    propagation_work = 0
    for radius in accepted_radius[:accepted_count]:
        diameter = int(math.ceil(radius * 2.0 + 1.0))
        propagation_work += diameter * diameter
    if propagation_work > limits.max_local_thickness_work_units:
        _resource_limit(
            "局部厚度的最大圆传播工作量超过安全上限；请裁剪 ROI 或分区分析。",
            operation=AdvancedAnalysisKind.LOCAL_THICKNESS,
        )
    thickness = np.zeros(foreground.shape, dtype=np.float32)
    height, width = foreground.shape
    for index, (x, y, radius) in enumerate(
        zip(
            accepted_x[:accepted_count],
            accepted_y[:accepted_count],
            accepted_radius[:accepted_count],
            strict=True,
        )
    ):
        if index % 128 == 0:
            _check_cancel(cancellation_token)
        x0 = max(0, int(math.floor(x - radius)))
        x1 = min(width, int(math.ceil(x + radius)) + 1)
        y0 = max(0, int(math.floor(y - radius)))
        y1 = min(height, int(math.ceil(y + radius)) + 1)
        local_y, local_x = np.ogrid[y0:y1, x0:x1]
        covered = (
            (local_x - x) * (local_x - x) + (local_y - y) * (local_y - y)
            <= (radius + tolerance) * (radius + tolerance)
        )
        covered &= foreground[y0:y1, x0:x1]
        target = thickness[y0:y1, x0:x1]
        np.maximum(target, np.float32(2.0 * radius), out=target, where=covered)
    selected = thickness[foreground]
    circles = tuple(
        MaximalInscribedCircle(
            center_x=int(x),
            center_y=int(y),
            radius_px=float(radius),
        )
        for x, y, radius in zip(
            accepted_x[:accepted_count],
            accepted_y[:accepted_count],
            accepted_radius[:accepted_count],
            strict=True,
        )
    )
    return LocalThicknessResult(
        thickness_px=thickness,
        maximal_circles=circles,
        foreground_pixel_count=foreground_count,
        maximum_thickness_px=float(np.max(selected)) if selected.size else 0.0,
        mean_thickness_px=float(np.mean(selected)) if selected.size else None,
        request_id=request.request_id,
        generation=request.generation,
    )


def calculate_multiscale_tubeness(
    request: TubenessRequest,
    *,
    cancellation_token: CancellationToken | None = None,
    limits: AdvancedAnalysisLimits = DEFAULT_ADVANCED_ANALYSIS_LIMITS,
) -> TubenessResult:
    estimate = estimate_advanced_analysis_resources(request, limits=limits)
    _enforce_estimate(estimate)
    _check_cancel(cancellation_token)
    image = np.asarray(request.image, dtype=np.float64)
    mask = (
        np.ones(image.shape, dtype=bool)
        if request.roi_mask is None
        else np.asarray(request.roi_mask, dtype=bool)
    )
    if not np.any(mask):
        _empty("Tubeness 分析的 ROI 中没有像素。")
    if np.any(~np.isfinite(image[mask])):
        _non_finite("Tubeness 分析的 ROI 中包含 NaN 或 Inf，请先修复非有限像素。")
    response = np.zeros(image.shape, dtype=np.float64)
    best_scale = np.zeros(image.shape, dtype=np.float64)
    beta_denominator = 2.0 * request.beta * request.beta
    for scale in request.scales:
        _check_cancel(cancellation_token)
        smoothed = cv2.GaussianBlur(
            image,
            (0, 0),
            sigmaX=scale,
            sigmaY=scale,
            borderType=cv2.BORDER_REFLECT_101,
        )
        normalization = scale * scale
        hxx = cv2.Sobel(
            smoothed,
            cv2.CV_64F,
            2,
            0,
            ksize=3,
            borderType=cv2.BORDER_REFLECT_101,
        ) * normalization
        hyy = cv2.Sobel(
            smoothed,
            cv2.CV_64F,
            0,
            2,
            ksize=3,
            borderType=cv2.BORDER_REFLECT_101,
        ) * normalization
        hxy = cv2.Sobel(
            smoothed,
            cv2.CV_64F,
            1,
            1,
            ksize=3,
            borderType=cv2.BORDER_REFLECT_101,
        ) * normalization
        discriminant = np.sqrt(np.maximum(0.0, (hxx - hyy) ** 2 + 4.0 * hxy * hxy))
        eigen_a = 0.5 * (hxx + hyy - discriminant)
        eigen_b = 0.5 * (hxx + hyy + discriminant)
        swap = np.abs(eigen_a) > np.abs(eigen_b)
        lambda_small = np.where(swap, eigen_b, eigen_a)
        lambda_large = np.where(swap, eigen_a, eigen_b)
        ratio = np.abs(lambda_small) / (np.abs(lambda_large) + np.finfo(np.float64).eps)
        structure = np.sqrt(lambda_small * lambda_small + lambda_large * lambda_large)
        structure_scale = (
            request.structure_scale
            if request.structure_scale is not None
            else max(float(np.max(structure[mask])) * 0.5, np.finfo(np.float64).eps)
        )
        vesselness = np.exp(-(ratio * ratio) / beta_denominator)
        vesselness *= 1.0 - np.exp(
            -(structure * structure) / (2.0 * structure_scale * structure_scale)
        )
        ridge_sign = lambda_large < 0 if request.bright_ridges else lambda_large > 0
        vesselness[~ridge_sign] = 0.0
        vesselness[~mask] = 0.0
        better = vesselness > response
        response[better] = vesselness[better]
        best_scale[better] = scale
    _check_cancel(cancellation_token)
    return TubenessResult(
        response=response.astype(np.float32),
        best_scale=best_scale.astype(np.float32),
        scales=request.scales,
        maximum_response=float(np.max(response)) if response.size else 0.0,
        request_id=request.request_id,
        generation=request.generation,
    )


def calculate_glcm_haralick(
    request: GlcmHaralickRequest,
    *,
    cancellation_token: CancellationToken | None = None,
    limits: AdvancedAnalysisLimits = DEFAULT_ADVANCED_ANALYSIS_LIMITS,
) -> GlcmHaralickResult:
    estimate = estimate_advanced_analysis_resources(request, limits=limits)
    _enforce_estimate(estimate)
    _check_cancel(cancellation_token)
    image = np.asarray(request.image, dtype=np.float64)
    roi = (
        np.ones(image.shape, dtype=bool)
        if request.roi_mask is None
        else np.asarray(request.roi_mask, dtype=bool)
    )
    finite = np.isfinite(image)
    valid = roi & finite
    valid_count = int(np.count_nonzero(valid))
    non_finite_count = int(np.count_nonzero(roi & ~finite))
    if valid_count == 0:
        _empty("GLCM 分析范围中没有有限像素。")
    if request.value_range is None:
        low = float(np.min(image[valid]))
        high = float(np.max(image[valid]))
        if math.isclose(low, high):
            high = low + 1.0
    else:
        low, high = request.value_range
    quantized = np.zeros(image.shape, dtype=np.int32)
    scaled = (image[valid] - low) / (high - low)
    quantized[valid] = np.clip(
        np.floor(scaled * request.levels),
        0,
        request.levels - 1,
    ).astype(np.int32)
    features: list[HaralickFeatures] = []
    for distance in request.distances:
        for direction in request.directions_degrees:
            _check_cancel(cancellation_token)
            dy, dx = _direction_offset(distance, direction)
            left_slice, right_slice = _paired_slices(image.shape, dy=dy, dx=dx)
            pair_mask = valid[left_slice] & valid[right_slice]
            first = quantized[left_slice][pair_mask]
            second = quantized[right_slice][pair_mask]
            pair_count = int(first.size)
            matrix = np.bincount(
                first * request.levels + second,
                minlength=request.levels * request.levels,
            ).reshape(request.levels, request.levels).astype(np.float64)
            if request.symmetric:
                matrix += matrix.T
            matrix_total = float(np.sum(matrix, dtype=np.float64))
            probability = (
                matrix / matrix_total
                if matrix_total > 0
                else np.zeros_like(matrix)
            )
            features.append(
                _haralick_features(
                    probability,
                    distance=distance,
                    direction=direction,
                    pair_count=pair_count,
                )
            )
    return GlcmHaralickResult(
        features=tuple(features),
        levels=request.levels,
        quantization_range=(low, high),
        symmetric=request.symmetric,
        valid_pixel_count=valid_count,
        non_finite_pixel_count=non_finite_count,
        request_id=request.request_id,
        generation=request.generation,
    )


def analyze_spatial_distribution(
    request: SpatialDistributionRequest,
    *,
    cancellation_token: CancellationToken | None = None,
    limits: AdvancedAnalysisLimits = DEFAULT_ADVANCED_ANALYSIS_LIMITS,
) -> SpatialDistributionResult:
    estimate = estimate_advanced_analysis_resources(request, limits=limits)
    _enforce_estimate(estimate)
    _check_cancel(cancellation_token)
    points = np.asarray(request.points, dtype=np.float64)
    points[:, 0] *= request.pixel_size_x
    points[:, 1] *= request.pixel_size_y
    count = len(points)
    nearest_squared = np.full(count, np.inf, dtype=np.float64)
    nearest_indices = np.full(count, -1, dtype=np.int64)
    block_size = min(1024, count)
    for start in range(0, count, block_size):
        _check_cancel(cancellation_token)
        stop = min(count, start + block_size)
        delta = points[start:stop, np.newaxis, :] - points[np.newaxis, :, :]
        squared = np.sum(delta * delta, axis=2, dtype=np.float64)
        local_rows = np.arange(stop - start)
        squared[local_rows, np.arange(start, stop)] = np.inf
        local_indices = np.argmin(squared, axis=1)
        nearest_indices[start:stop] = local_indices
        nearest_squared[start:stop] = squared[local_rows, local_indices]
    distances = np.sqrt(nearest_squared)
    if request.study_area is not None:
        area = request.study_area
        area_source = "用户指定"
    else:
        width = float(np.max(points[:, 0]) - np.min(points[:, 0]))
        height = float(np.max(points[:, 1]) - np.min(points[:, 1]))
        area = width * height
        area_source = "点集轴对齐包围框"
        if not math.isfinite(area) or area <= 0:
            _invalid("点集包围框面积为零，请显式提供研究区域面积。")
    density = count / area
    return SpatialDistributionResult(
        nearest_neighbor_distances=tuple(float(value) for value in distances),
        nearest_neighbor_indices=tuple(int(value) for value in nearest_indices),
        mean_nearest_neighbor_distance=float(np.mean(distances)),
        median_nearest_neighbor_distance=float(np.median(distances)),
        minimum_nearest_neighbor_distance=float(np.min(distances)),
        maximum_nearest_neighbor_distance=float(np.max(distances)),
        study_area=float(area),
        area_source=area_source,
        spatial_density=float(density),
        unit=request.unit,
        request_id=request.request_id,
        generation=request.generation,
    )


def build_intensity_surface(
    request: IntensitySurfaceRequest,
    *,
    cancellation_token: CancellationToken | None = None,
    limits: AdvancedAnalysisLimits = DEFAULT_ADVANCED_ANALYSIS_LIMITS,
) -> IntensitySurfaceResult:
    estimate = estimate_advanced_analysis_resources(request, limits=limits)
    _enforce_estimate(estimate)
    _check_cancel(cancellation_token)
    image = np.asarray(request.image, dtype=np.float64)
    ys = np.arange(0, image.shape[0], request.sample_step_y, dtype=np.int64)
    xs = np.arange(0, image.shape[1], request.sample_step_x, dtype=np.int64)
    sampled = image[np.ix_(ys, xs)]
    roi = (
        np.ones(sampled.shape, dtype=bool)
        if request.roi_mask is None
        else np.asarray(request.roi_mask, dtype=bool)[np.ix_(ys, xs)]
    )
    finite = np.isfinite(sampled)
    valid = roi & finite
    rows: list[tuple[float | None, ...]] = []
    for row_index in range(sampled.shape[0]):
        if row_index % 128 == 0:
            _check_cancel(cancellation_token)
        rows.append(
            tuple(
                float(sampled[row_index, column_index])
                if valid[row_index, column_index]
                else None
                for column_index in range(sampled.shape[1])
            )
        )
    selected = sampled[valid]
    return IntensitySurfaceResult(
        x_coordinates=tuple(float(value * request.pixel_size_x) for value in xs),
        y_coordinates=tuple(float(value * request.pixel_size_y) for value in ys),
        z_values=tuple(rows),
        finite_sample_count=int(selected.size),
        masked_sample_count=int(np.count_nonzero(~roi)),
        non_finite_sample_count=int(np.count_nonzero(roi & ~finite)),
        z_minimum=float(np.min(selected)) if selected.size else None,
        z_maximum=float(np.max(selected)) if selected.size else None,
        coordinate_unit=request.unit,
        request_id=request.request_id,
        generation=request.generation,
    )


def _smooth_circular_histogram(
    histogram: NDArray[np.float64],
    *,
    sigma: float,
) -> NDArray[np.float64]:
    radius = max(1, int(math.ceil(sigma * 3.0)))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-(offsets * offsets) / (2.0 * sigma * sigma))
    kernel /= np.sum(kernel)
    result = np.zeros_like(histogram, dtype=np.float64)
    for offset, weight in zip(range(-radius, radius + 1), kernel, strict=True):
        result += np.roll(histogram, offset) * float(weight)
    return result


def _skeletonize_with_cancellation(
    mask: NDArray[np.bool_],
    *,
    cancellation_token: CancellationToken | None,
    max_work_units: int,
) -> NDArray[np.bool_]:
    """可取消的 Zhang-Suen 同步细化。"""

    skeleton = np.asarray(mask, dtype=np.uint8).copy()
    if min(skeleton.shape) < 3:
        return skeleton.astype(bool)
    changed = True
    work = 0
    while changed:
        changed = False
        for first_subiteration in (True, False):
            _check_cancel(cancellation_token)
            work += skeleton.size
            if work > max_work_units:
                _resource_limit(
                    "骨架细化迭代超过安全工作量上限。",
                    operation=AdvancedAnalysisKind.SKELETON_NETWORK,
                )
            padded = np.pad(skeleton, 1, mode="constant")
            p2 = padded[:-2, 1:-1]
            p3 = padded[:-2, 2:]
            p4 = padded[1:-1, 2:]
            p5 = padded[2:, 2:]
            p6 = padded[2:, 1:-1]
            p7 = padded[2:, :-2]
            p8 = padded[1:-1, :-2]
            p9 = padded[:-2, :-2]
            neighbors = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
            transitions = sum(
                transition.astype(np.uint8)
                for transition in (
                    (p2 == 0) & (p3 == 1),
                    (p3 == 0) & (p4 == 1),
                    (p4 == 0) & (p5 == 1),
                    (p5 == 0) & (p6 == 1),
                    (p6 == 0) & (p7 == 1),
                    (p7 == 0) & (p8 == 1),
                    (p8 == 0) & (p9 == 1),
                    (p9 == 0) & (p2 == 1),
                )
            )
            if first_subiteration:
                condition_a = p2 * p4 * p6 == 0
                condition_b = p4 * p6 * p8 == 0
            else:
                condition_a = p2 * p4 * p8 == 0
                condition_b = p2 * p6 * p8 == 0
            remove = (
                (skeleton == 1)
                & (neighbors >= 2)
                & (neighbors <= 6)
                & (transitions == 1)
                & condition_a
                & condition_b
            )
            if np.any(remove):
                skeleton[remove] = 0
                changed = True
    return skeleton.astype(bool)


def _directionality_peaks(
    histogram: NDArray[np.float64],
    centers: NDArray[np.float64],
    *,
    minimum_fraction: float,
    maximum_count: int,
) -> tuple[OrientationPeak, ...]:
    maximum = float(np.max(histogram)) if histogram.size else 0.0
    if maximum <= 0:
        return ()
    candidates = np.flatnonzero(
        (histogram >= np.roll(histogram, 1))
        & (histogram >= np.roll(histogram, -1))
        & (histogram >= maximum * minimum_fraction)
    )
    # 平台峰只保留最小索引，首尾相连的平台也视作一个轴向峰。
    candidate_set = set(int(value) for value in candidates)
    selected: list[int] = []
    for candidate in sorted(candidate_set):
        previous = (candidate - 1) % len(histogram)
        if previous in candidate_set and math.isclose(
            float(histogram[previous]),
            float(histogram[candidate]),
            rel_tol=1e-12,
            abs_tol=1e-15,
        ):
            continue
        selected.append(candidate)
    if not selected:
        selected = [int(np.argmax(histogram))]
    selected.sort(key=lambda index: (-float(histogram[index]), index))
    return tuple(
        OrientationPeak(
            angle_degrees=float(centers[index]),
            weight=float(histogram[index]),
            relative_weight=float(histogram[index] / maximum),
            bin_index=int(index),
        )
        for index in selected[:maximum_count]
    )


def _connected_vertex_clusters(
    vertices: set[int],
    adjacency: Sequence[Sequence[tuple[int, float]]],
) -> tuple[frozenset[int], ...]:
    remaining = set(vertices)
    clusters: list[frozenset[int]] = []
    while remaining:
        seed = min(remaining)
        remaining.remove(seed)
        stack = [seed]
        cluster = {seed}
        while stack:
            current = stack.pop()
            for neighbor, _weight in adjacency[current]:
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    cluster.add(neighbor)
                    stack.append(neighbor)
        clusters.append(frozenset(cluster))
    return tuple(clusters)


def _graph_components(
    adjacency: Sequence[Sequence[tuple[int, float]]],
) -> tuple[tuple[int, ...], tuple[tuple[int, ...], ...]]:
    labels = [-1] * len(adjacency)
    components: list[tuple[int, ...]] = []
    for seed in range(len(adjacency)):
        if labels[seed] >= 0:
            continue
        label = len(components)
        stack = [seed]
        labels[seed] = label
        component: list[int] = []
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor, _weight in adjacency[current]:
                if labels[neighbor] < 0:
                    labels[neighbor] = label
                    stack.append(neighbor)
        components.append(tuple(component))
    return tuple(labels), tuple(components)


def _trace_skeleton_branches(
    coordinates_yx: NDArray[np.int64],
    adjacency: Sequence[Sequence[tuple[int, float]]],
    *,
    endpoint_vertices: tuple[int, ...],
    branch_clusters: tuple[frozenset[int], ...],
    branch_cluster_by_vertex: dict[int, int],
    components: tuple[tuple[int, ...], ...],
) -> tuple[SkeletonBranch, ...]:
    node_vertices = set(endpoint_vertices) | set(branch_cluster_by_vertex)
    visited_edges: set[tuple[int, int]] = set()
    branches: list[SkeletonBranch] = []

    def point_for_vertex(vertex: int) -> Coordinate:
        return (
            float(coordinates_yx[vertex, 1]),
            float(coordinates_yx[vertex, 0]),
        )

    def node_point(vertex: int) -> Coordinate:
        cluster_index = branch_cluster_by_vertex.get(vertex)
        if cluster_index is None:
            return point_for_vertex(vertex)
        members = list(branch_clusters[cluster_index])
        return (
            float(np.mean(coordinates_yx[members, 1])),
            float(np.mean(coordinates_yx[members, 0])),
        )

    for start in sorted(node_vertices):
        for neighbor, first_weight in adjacency[start]:
            if (
                start in branch_cluster_by_vertex
                and neighbor in branch_cluster_by_vertex
                and branch_cluster_by_vertex[start] == branch_cluster_by_vertex[neighbor]
            ):
                continue
            edge_key = (min(start, neighbor), max(start, neighbor))
            if edge_key in visited_edges:
                continue
            visited_edges.add(edge_key)
            length = first_weight
            previous = start
            current = neighbor
            while current not in node_vertices and len(adjacency[current]) == 2:
                next_candidates = [
                    (candidate, weight)
                    for candidate, weight in adjacency[current]
                    if candidate != previous
                ]
                if not next_candidates:
                    break
                following, weight = next_candidates[0]
                next_key = (min(current, following), max(current, following))
                if next_key in visited_edges:
                    break
                visited_edges.add(next_key)
                length += weight
                previous, current = current, following
            branches.append(
                SkeletonBranch(
                    start_px=node_point(start),
                    end_px=node_point(current),
                    length=float(length),
                )
            )

    # 没有端点或分支点的连通分量是闭环；将整环作为一个闭合分支。
    for component in components:
        if any(vertex in node_vertices for vertex in component):
            continue
        component_set = set(component)
        length = 0.0
        for vertex in component:
            for neighbor, weight in adjacency[vertex]:
                if neighbor in component_set and vertex < neighbor:
                    length += weight
        if length > 0:
            branches.append(
                SkeletonBranch(
                    start_px=None,
                    end_px=None,
                    length=float(length),
                    closed=True,
                )
            )
    branches.sort(
        key=lambda item: (
            item.closed,
            item.start_px or (-1.0, -1.0),
            item.end_px or (-1.0, -1.0),
        )
    )
    return tuple(branches)


def _maximum_skeleton_geodesic(
    adjacency: Sequence[Sequence[tuple[int, float]]],
    components: tuple[tuple[int, ...], ...],
    endpoint_vertices: tuple[int, ...],
    component_labels: tuple[int, ...],
    *,
    cancellation_token: CancellationToken | None,
    max_work_units: int,
) -> float:
    endpoints_by_component: dict[int, list[int]] = {}
    for vertex in endpoint_vertices:
        endpoints_by_component.setdefault(component_labels[vertex], []).append(vertex)
    maximum = 0.0
    work = 0
    for component_index, component in enumerate(components):
        if len(component) <= 1:
            continue
        endpoints = endpoints_by_component.get(component_index, [])
        component_edge_count = (
            sum(len(adjacency[vertex]) for vertex in component) // 2
        )
        component_loop_count = max(
            0,
            component_edge_count - len(component) + 1,
        )
        if component_loop_count > 0 and not all(
            len(adjacency[vertex]) == 2 for vertex in component
        ):
            # 带支路的环图，直径端点可能位于环内的二度顶点。为避免低估，
            # 对全部顶点做精确最短路；超过预算时明确拒绝而不返回近似值。
            sources = list(component)
        elif endpoints:
            sources = endpoints
        else:
            # 纯单环用累积弧长精确求顶点测地直径，避免加权环上的双扫描近似。
            maximum = max(
                maximum,
                _weighted_cycle_diameter(adjacency, component),
            )
            work += len(component) * 8
            continue
        for source_index, source in enumerate(sources):
            if source_index % 32 == 0:
                _check_cancel(cancellation_token)
            work += len(component) * 4
            if work > max_work_units:
                _resource_limit(
                    "骨架最大测地距离计算超过安全工作量上限。",
                    operation=AdvancedAnalysisKind.SKELETON_NETWORK,
                )
            distances = _dijkstra(adjacency, source)
            maximum = max(
                maximum,
                max(distances[target] for target in endpoints),
            )
    return maximum


def _weighted_cycle_diameter(
    adjacency: Sequence[Sequence[tuple[int, float]]],
    component: tuple[int, ...],
) -> float:
    start = min(component)
    previous = -1
    current = start
    edge_weights: list[float] = []
    visited_vertices = {start}
    while True:
        candidates = [
            (neighbor, weight)
            for neighbor, weight in adjacency[current]
            if neighbor != previous
        ]
        if not candidates:
            return 0.0
        candidates.sort(key=lambda item: item[0])
        following, weight = candidates[0]
        if following in visited_vertices and following != start:
            if len(candidates) < 2:
                return 0.0
            following, weight = candidates[1]
        edge_weights.append(float(weight))
        if following == start:
            break
        visited_vertices.add(following)
        previous, current = current, following
        if len(visited_vertices) > len(component):
            return 0.0
    if len(edge_weights) != len(component):
        return 0.0
    weights = np.asarray(edge_weights * 2, dtype=np.float64)
    prefix = np.concatenate(([0.0], np.cumsum(weights, dtype=np.float64)))
    circumference = float(sum(edge_weights))
    half = circumference / 2.0
    maximum = 0.0
    count = len(edge_weights)
    for start_index in range(count):
        target = prefix[start_index] + half
        insertion = int(
            np.searchsorted(
                prefix,
                target,
                side="left",
            )
        )
        for end_index in (insertion - 1, insertion):
            if start_index < end_index <= start_index + count:
                arc = float(prefix[end_index] - prefix[start_index])
                maximum = max(maximum, min(arc, circumference - arc))
    return maximum


def _dijkstra(
    adjacency: Sequence[Sequence[tuple[int, float]]],
    source: int,
) -> list[float]:
    distances = [math.inf] * len(adjacency)
    distances[source] = 0.0
    queue: list[tuple[float, int]] = [(0.0, source)]
    while queue:
        distance, vertex = heapq.heappop(queue)
        if distance != distances[vertex]:
            continue
        for neighbor, weight in adjacency[vertex]:
            candidate = distance + weight
            if candidate < distances[neighbor]:
                distances[neighbor] = candidate
                heapq.heappush(queue, (candidate, neighbor))
    return distances


def _haralick_features(
    probability: NDArray[np.float64],
    *,
    distance: int,
    direction: float,
    pair_count: int,
) -> HaralickFeatures:
    levels = probability.shape[0]
    row, column = np.indices((levels, levels), dtype=np.float64)
    difference = row - column
    contrast = float(np.sum(probability * difference * difference))
    dissimilarity = float(np.sum(probability * np.abs(difference)))
    homogeneity = float(np.sum(probability / (1.0 + difference * difference)))
    angular_second_moment = float(np.sum(probability * probability))
    energy = math.sqrt(angular_second_moment)
    row_probability = np.sum(probability, axis=1)
    column_probability = np.sum(probability, axis=0)
    row_mean = float(np.sum(np.arange(levels) * row_probability))
    column_mean = float(np.sum(np.arange(levels) * column_probability))
    row_std = math.sqrt(
        float(
            np.sum(
                ((np.arange(levels) - row_mean) ** 2) * row_probability
            )
        )
    )
    column_std = math.sqrt(
        float(
            np.sum(
                ((np.arange(levels) - column_mean) ** 2) * column_probability
            )
        )
    )
    correlation = (
        float(
            np.sum(
                probability
                * (row - row_mean)
                * (column - column_mean)
            )
            / (row_std * column_std)
        )
        if row_std > 0 and column_std > 0
        else None
    )
    positive = probability[probability > 0]
    entropy = float(-np.sum(positive * np.log(positive)))
    maximum_probability = float(np.max(probability)) if probability.size else 0.0
    return HaralickFeatures(
        distance_px=int(distance),
        direction_degrees=float(direction),
        pair_count=int(pair_count),
        contrast=contrast,
        dissimilarity=dissimilarity,
        homogeneity=homogeneity,
        angular_second_moment=angular_second_moment,
        energy=energy,
        correlation=correlation,
        entropy=entropy,
        maximum_probability=maximum_probability,
        matrix=probability,
    )


def _direction_offset(distance: int, direction_degrees: float) -> tuple[int, int]:
    radians = math.radians(float(direction_degrees))
    dx = int(round(int(distance) * math.cos(radians)))
    dy = int(round(-int(distance) * math.sin(radians)))
    if dx == 0 and dy == 0:
        _invalid("GLCM 方向与距离取整后得到零偏移。")
    return dy, dx


def _paired_slices(
    shape: tuple[int, int],
    *,
    dy: int,
    dx: int,
) -> tuple[tuple[slice, slice], tuple[slice, slice]]:
    height, width = shape
    if abs(dy) >= height or abs(dx) >= width:
        return (
            (slice(0, 0), slice(0, 0)),
            (slice(0, 0), slice(0, 0)),
        )
    if dy >= 0:
        first_y = slice(0, height - dy)
        second_y = slice(dy, height)
    else:
        first_y = slice(-dy, height)
        second_y = slice(0, height + dy)
    if dx >= 0:
        first_x = slice(0, width - dx)
        second_x = slice(dx, width)
    else:
        first_x = slice(-dx, width)
        second_x = slice(0, width + dx)
    return (first_y, first_x), (second_y, second_x)


def _freeze_scalar_image(
    image: NDArray[Any],
    *,
    name: str,
    require_finite: bool = True,
) -> NDArray[Any]:
    array = np.asarray(image)
    if array.ndim == 3 and array.shape[2] == 1:
        array = array[..., 0]
    if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] < 1:
        _invalid(f"{name}必须是非空二维数组。")
    if array.dtype.kind not in {"u", "i", "f", "b"}:
        _invalid(f"{name}的数据类型必须是整数、浮点数或布尔值。")
    frozen = np.ascontiguousarray(array).copy()
    if require_finite and frozen.dtype.kind == "f" and not np.all(np.isfinite(frozen)):
        _non_finite(f"{name}包含 NaN 或 Inf。")
    frozen.setflags(write=False)
    return frozen


def _freeze_binary_mask(mask: NDArray[Any], name: str) -> NDArray[np.bool_]:
    array = np.asarray(mask)
    if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] < 1:
        _invalid(f"{name}必须是非空二维数组。")
    frozen = np.ascontiguousarray(array, dtype=bool).copy()
    frozen.setflags(write=False)
    return frozen


def _freeze_optional_mask(
    mask: NDArray[np.bool_] | None,
    shape: tuple[int, int],
) -> NDArray[np.bool_] | None:
    if mask is None:
        return None
    frozen = _freeze_binary_mask(mask, "ROI 掩膜")
    if frozen.shape != shape:
        _invalid(f"ROI 掩膜尺寸 {frozen.shape!r} 与图像尺寸 {shape!r} 不一致。")
    return frozen


def _freeze_points(points: Iterable[Sequence[float]]) -> tuple[Coordinate, ...]:
    frozen: list[Coordinate] = []
    for raw in points:
        if len(raw) != 2:
            _invalid("空间点坐标必须包含 x 和 y 两个数值。")
        x = _finite_float("点坐标 x", raw[0])
        y = _finite_float("点坐标 y", raw[1])
        frozen.append((x, y))
    return tuple(frozen)


def _freeze_bool_output(array: NDArray[Any]) -> NDArray[np.bool_]:
    frozen = np.ascontiguousarray(array, dtype=bool).copy()
    frozen.setflags(write=False)
    return frozen


def _freeze_float_output(array: NDArray[Any]) -> NDArray[np.float32]:
    frozen = np.ascontiguousarray(array, dtype=np.float32).copy()
    if not np.all(np.isfinite(frozen)):
        _non_finite("高级分析生成了非有限浮点结果。")
    frozen.setflags(write=False)
    return frozen


def _freeze_double_output(array: NDArray[Any]) -> NDArray[np.float64]:
    frozen = np.ascontiguousarray(array, dtype=np.float64).copy()
    if not np.all(np.isfinite(frozen)):
        _non_finite("高级分析生成了非有限浮点结果。")
    frozen.setflags(write=False)
    return frozen


def _finite_float(name: str, value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise AdvancedAnalysisError(
            AdvancedAnalysisErrorCode.INVALID_INPUT,
            f"{name}必须是数值。",
        ) from exc
    if not math.isfinite(result):
        _non_finite(f"{name}必须是有限数。")
    return result


def _require_finite_positive(name: str, value: Any) -> None:
    if _finite_float(name, value) <= 0:
        _invalid(f"{name}必须大于 0。")


def _require_finite_nonnegative(name: str, value: Any) -> None:
    if _finite_float(name, value) < 0:
        _invalid(f"{name}不能为负数。")


def _check_cancel(cancellation_token: CancellationToken | None) -> None:
    if cancellation_token is not None:
        cancellation_token.raise_if_cancelled()


def _enforce_estimate(estimate: AdvancedAnalysisResourceEstimate) -> None:
    if not estimate.allowed:
        _resource_limit(estimate.reason, operation=estimate.operation)


def _rejected_estimate(
    kind: AdvancedAnalysisKind,
    pixels: int,
    peak_bytes: int,
    work: int,
    output: int,
    reason: str,
) -> AdvancedAnalysisResourceEstimate:
    return AdvancedAnalysisResourceEstimate(
        operation=kind,
        input_pixels=int(pixels),
        estimated_peak_bytes=int(peak_bytes),
        estimated_work_units=int(work),
        estimated_output_values=int(output),
        allowed=False,
        reason=str(reason),
    )


def _invalid(message: str) -> None:
    raise AdvancedAnalysisError(AdvancedAnalysisErrorCode.INVALID_INPUT, message)


def _non_finite(message: str) -> None:
    raise AdvancedAnalysisError(AdvancedAnalysisErrorCode.NON_FINITE_INPUT, message)


def _empty(message: str) -> None:
    raise AdvancedAnalysisError(AdvancedAnalysisErrorCode.EMPTY_SELECTION, message)


def _resource_limit(
    message: str,
    *,
    operation: AdvancedAnalysisKind,
) -> None:
    raise AdvancedAnalysisError(
        AdvancedAnalysisErrorCode.RESOURCE_LIMIT,
        message,
        details=(("operation", operation.value),),
    )
