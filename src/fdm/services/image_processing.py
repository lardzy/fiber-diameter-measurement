"""Deterministic, UI-independent image-processing kernels.

The desktop UI deliberately does not own pixel algorithms.  This module accepts
immutable request snapshots and returns immutable result snapshots so callers
can safely execute work in a background thread and discard late generations.

The implementation keeps native ``uint8``, ``uint16`` and ``float32`` data
separate from display conversion.  Except for geometric transforms and explicit
type conversion, operations preserve the input shape and dtype.  When a region
mask is supplied, pixels outside that region are copied from the source without
modification.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from types import MappingProxyType
from typing import Any, Callable, Mapping, TypeAlias

import cv2
import numpy as np
from numpy.typing import NDArray

from fdm.image_processing_models import (
    ImageOperationDescriptor,
    ImageOperationParameterSchema,
    ImageOperationSpec,
    ImageProcessingRecipe,
    RasterSemantic,
    RasterTypeState,
    RoiProcessingSemantics,
)
from fdm.raster import RasterPixelType


ParameterScalar: TypeAlias = bool | int | float | str | None
ParameterValue: TypeAlias = ParameterScalar | tuple[ParameterScalar, ...]


class PixelType(str, Enum):
    UINT8 = "uint8"
    UINT16 = "uint16"
    FLOAT32 = "float32"

    @property
    def dtype(self) -> np.dtype[Any]:
        return np.dtype(self.value)


class ConversionScaleMode(str, Enum):
    """How samples are mapped when changing pixel type."""

    PRESERVE_VALUES = "preserve_values"
    FULL_TYPE_RANGE = "full_type_range"
    DATA_RANGE = "data_range"


class NonfiniteIntegerPolicy(str, Enum):
    """How float NaN/Inf samples are handled before an integer conversion."""

    REJECT = "reject"
    ZERO = "zero"
    RANGE_BOUNDS = "range_bounds"


class ColorTarget(str, Enum):
    GRAYSCALE = "grayscale"
    RGB = "rgb"


class GrayscaleMethod(str, Enum):
    """The explicit RGB-to-gray transfer formula."""

    REC601 = "rec601"
    AVERAGE = "average"


class CanvasAnchor(str, Enum):
    TOP_LEFT = "top_left"
    TOP_CENTER = "top_center"
    TOP_RIGHT = "top_right"
    CENTER_LEFT = "center_left"
    CENTER = "center"
    CENTER_RIGHT = "center_right"
    BOTTOM_LEFT = "bottom_left"
    BOTTOM_CENTER = "bottom_center"
    BOTTOM_RIGHT = "bottom_right"


class PixelBinMethod(str, Enum):
    MEAN = "mean"
    MINIMUM = "minimum"
    MAXIMUM = "maximum"
    SUM = "sum"


class PixelBinRemainderPolicy(str, Enum):
    REJECT = "reject"
    CROP = "crop"


class ImageOperation(str, Enum):
    COPY = "copy"
    CONVERT_TYPE = "convert_type"
    CONVERT_COLOR = "convert_color"
    COLOR_BALANCE = "color_balance"
    BRIGHTNESS_CONTRAST = "brightness_contrast"
    ADJUST_LEVELS = "adjust_levels"
    THRESHOLD = "threshold"
    FLIP_HORIZONTAL = "flip_horizontal"
    FLIP_VERTICAL = "flip_vertical"
    ROTATE_90_CLOCKWISE = "rotate_90_clockwise"
    ROTATE_90_COUNTERCLOCKWISE = "rotate_90_counterclockwise"
    ROTATE_180 = "rotate_180"
    ROTATE = "rotate"
    CROP = "crop"
    RESIZE = "resize"
    TRANSLATE = "translate"
    RESIZE_CANVAS = "resize_canvas"
    PIXEL_BIN = "pixel_bin"
    GAUSSIAN_BLUR = "gaussian_blur"
    MEDIAN_FILTER = "median_filter"
    MEAN_FILTER = "mean_filter"
    BILATERAL_FILTER = "bilateral_filter"
    UNSHARP_MASK = "unsharp_mask"
    SOBEL_EDGES = "sobel_edges"
    LAPLACIAN_EDGES = "laplacian_edges"
    CANNY_EDGES = "canny_edges"
    NORMALIZE = "normalize"
    HISTOGRAM_EQUALIZATION = "histogram_equalization"
    CLAHE = "clahe"
    REMOVE_OUTLIERS = "remove_outliers"
    REPAIR_NONFINITE = "repair_nonfinite"
    AUTO_THRESHOLD = "auto_threshold"
    BINARIZE = "binarize"
    ERODE = "erode"
    DILATE = "dilate"
    MORPHOLOGY_OPEN = "morphology_open"
    MORPHOLOGY_CLOSE = "morphology_close"
    FILL_HOLES = "fill_holes"
    CONTOUR_EXTRACT = "contour_extract"
    REMOVE_SMALL_OBJECTS = "remove_small_objects"
    FILL_SMALL_HOLES = "fill_small_holes"
    DISTANCE_TRANSFORM = "distance_transform"
    SKELETONIZE = "skeletonize"
    WATERSHED = "watershed"
    WATERSHED_V2 = "watershed_v2"
    TOP_HAT = "top_hat"
    BLACK_HAT = "black_hat"
    BACKGROUND_SUBTRACT = "background_subtract"
    ROLLING_BALL_BACKGROUND_SUBTRACT = "rolling_ball_background_subtract"
    CUSTOM_CONVOLUTION = "custom_convolution"
    INVERT = "invert"
    ADD = "add"
    SUBTRACT = "subtract"
    MULTIPLY = "multiply"
    DIVIDE = "divide"
    GAMMA = "gamma"
    LOG = "log"
    LOG_V2 = "log_v2"
    EXP = "exp"
    EXP_V2 = "exp_v2"
    SQRT = "sqrt"
    SQRT_V2 = "sqrt_v2"
    ABS = "abs"
    CLAMP = "clamp"
    IMAGE_CALCULATOR = "image_calculator"
    FFT_FILTER = "fft_filter"
    FFT_POWER_SPECTRUM = "fft_power_spectrum"
    STRIPE_SUPPRESSION = "stripe_suppression"
    ADAPTIVE_THRESHOLD = "adaptive_threshold"
    PERCENTILE_SATURATION = "percentile_saturation"
    RANK_FILTER = "rank_filter"
    MORPHOLOGY_DERIVATIVE = "morphology_derivative"
    MORPHOLOGICAL_RECONSTRUCTION = "morphological_reconstruction"
    REGIONAL_EXTREMA = "regional_extrema"
    CLEAR_BORDER = "clear_border"
    FLAT_FIELD_CORRECTION = "flat_field_correction"


@dataclass(frozen=True, slots=True)
class ImageOperationCapability:
    """Execution constraints for one resolved operation invocation.

    ``halo_x``/``halo_y`` describe the maximum finite neighborhood dependency
    in source-image pixels.  A tiled executor may only use operations whose
    spatial extent is preserved and whose complete dependency is represented
    by this halo.  Global histogram, connectivity, hysteresis, FFT and geometry
    operations intentionally remain non-tileable.
    """

    tileable: bool
    preserves_spatial_extent: bool
    supports_roi: bool
    halo_x: int = 0
    halo_y: int = 0
    requires_full_image_prescan: bool = False
    reason: str = ""


class ImageExecutionMode(str, Enum):
    TILED = "tiled"
    WHOLE_IMAGE = "whole_image"


MAX_TILED_HALO_FRACTION = 0.25
MAX_TILED_CPU_AMPLIFICATION = 2.0


@dataclass(frozen=True, slots=True)
class TiledExecutionEstimate:
    """Deterministic pixel-work estimate for one resolved execution.

    ``estimated_*_cpu_work_units`` are deliberately hardware-independent
    relative units, not elapsed-time promises.  They combine the number of
    source samples handed to the kernel with a conservative operation cost
    factor.  This is sufficient to detect when overlapping halo patches would
    do materially more CPU work than one exact whole-image invocation.
    """

    mode: ImageExecutionMode
    tile_count: int
    halo_x: int
    halo_y: int
    source_pixels: int
    tiled_patch_pixels: int
    overlap_multiplier: float
    operation_cost_factor: int
    estimated_cpu_work_units: int
    estimated_tiled_cpu_work_units: int
    estimated_whole_cpu_work_units: int
    reason: str = ""

    @property
    def uses_tiled_execution(self) -> bool:
        return self.mode is ImageExecutionMode.TILED


def estimate_tiled_execution(
    operation: ImageOperation | str,
    image_shape: tuple[int, ...],
    *,
    parameters: Mapping[str, object] | None = None,
    roi_requested: bool = False,
    tile_size: int = 1024,
) -> TiledExecutionEstimate:
    """Choose tiled versus exact whole-image execution before allocating output."""

    resolved = (
        operation
        if isinstance(operation, ImageOperation)
        else _coerce_enum(ImageOperation, operation, "图像操作")
    )
    if len(image_shape) not in {2, 3}:
        raise ValueError("图像尺寸必须包含高度、宽度和可选通道。")
    height = int(image_shape[0])
    width = int(image_shape[1])
    if height <= 0 or width <= 0:
        raise ValueError("图像宽高必须为正整数。")
    resolved_tile_size = int(tile_size)
    if resolved_tile_size < 32:
        raise ValueError("处理图块边长必须至少为 32 像素。")

    params = dict(parameters or {})
    capability = resolve_image_operation_capability(resolved, params)
    halo_x = max(0, int(capability.halo_x))
    halo_y = max(0, int(capability.halo_y))
    tile_columns = (width + resolved_tile_size - 1) // resolved_tile_size
    tile_rows = (height + resolved_tile_size - 1) // resolved_tile_size
    tile_count = tile_columns * tile_rows

    def patch_axis_total(length: int, halo: int) -> int:
        total = 0
        for core_start in range(0, length, resolved_tile_size):
            core_end = min(length, core_start + resolved_tile_size)
            patch_start = max(0, core_start - halo)
            patch_end = min(length, core_end + halo)
            total += patch_end - patch_start
        return total

    tiled_patch_pixels = (
        patch_axis_total(height, halo_y)
        * patch_axis_total(width, halo_x)
    )
    source_pixels = height * width
    overlap_multiplier = tiled_patch_pixels / source_pixels
    cost_factor = _operation_cpu_cost_factor(resolved, params)
    whole_work = source_pixels * cost_factor
    tiled_work = tiled_patch_pixels * cost_factor

    mode = ImageExecutionMode.TILED
    reason = ""
    if not capability.tileable:
        mode = ImageExecutionMode.WHOLE_IMAGE
        reason = capability.reason or "该操作具有全局像素依赖。"
    elif not capability.preserves_spatial_extent:
        mode = ImageExecutionMode.WHOLE_IMAGE
        reason = "该操作会改变输出空间范围。"
    elif roi_requested and not capability.supports_roi:
        mode = ImageExecutionMode.WHOLE_IMAGE
        reason = "该操作不支持按 ROI 分块写回。"
    elif tile_count <= 1:
        mode = ImageExecutionMode.WHOLE_IMAGE
        reason = "图像未超过单个处理图块。"
    elif (
        halo_x > resolved_tile_size * MAX_TILED_HALO_FRACTION
        or halo_y > resolved_tile_size * MAX_TILED_HALO_FRACTION
    ):
        mode = ImageExecutionMode.WHOLE_IMAGE
        reason = (
            "邻域 halo 超过图块边长的 25%，整图单次执行可避免"
            "高重叠和重复边界计算。"
        )
    elif overlap_multiplier > MAX_TILED_CPU_AMPLIFICATION:
        mode = ImageExecutionMode.WHOLE_IMAGE
        reason = (
            "图块 halo 重叠会使预计 CPU 像素工作量超过整图执行的"
            f" {MAX_TILED_CPU_AMPLIFICATION:.1f} 倍。"
        )

    selected_work = tiled_work if mode is ImageExecutionMode.TILED else whole_work
    return TiledExecutionEstimate(
        mode=mode,
        tile_count=tile_count,
        halo_x=halo_x,
        halo_y=halo_y,
        source_pixels=source_pixels,
        tiled_patch_pixels=tiled_patch_pixels,
        overlap_multiplier=float(overlap_multiplier),
        operation_cost_factor=cost_factor,
        estimated_cpu_work_units=selected_work,
        estimated_tiled_cpu_work_units=tiled_work,
        estimated_whole_cpu_work_units=whole_work,
        reason=reason,
    )


def _operation_cpu_cost_factor(
    operation: ImageOperation,
    parameters: Mapping[str, object],
) -> int:
    """Return conservative relative kernel work per submitted source pixel."""

    if operation in {
        ImageOperation.MEDIAN_FILTER,
        ImageOperation.MEAN_FILTER,
        ImageOperation.REMOVE_OUTLIERS,
        ImageOperation.REPAIR_NONFINITE,
        ImageOperation.ADAPTIVE_THRESHOLD,
        ImageOperation.RANK_FILTER,
        ImageOperation.MORPHOLOGY_DERIVATIVE,
    }:
        radius = max(0, int(parameters.get("radius", 1)))
        return max(1, (2 * radius + 1) ** 2)
    if operation is ImageOperation.CUSTOM_CONVOLUTION:
        width = max(1, int(parameters.get("kernel_width", 1)))
        height = max(1, int(parameters.get("kernel_height", 1)))
        return width * height
    if operation in {
        ImageOperation.ERODE,
        ImageOperation.DILATE,
        ImageOperation.MORPHOLOGY_OPEN,
        ImageOperation.MORPHOLOGY_CLOSE,
        ImageOperation.TOP_HAT,
        ImageOperation.BLACK_HAT,
    }:
        radius = max(0, int(parameters.get("radius", 1)))
        iterations = max(1, int(parameters.get("iterations", 1)))
        passes = (
            1
            if operation in {ImageOperation.ERODE, ImageOperation.DILATE}
            else 2
        )
        return max(1, (2 * radius + 1) ** 2 * iterations * passes)
    if operation is ImageOperation.GAUSSIAN_BLUR:
        sigma_x = max(
            0.0,
            float(parameters.get("sigma_x", parameters.get("sigma", 1.0))),
        )
        sigma_y = max(0.0, float(parameters.get("sigma_y", sigma_x)))
        return max(
            1,
            (2 * int(math.ceil(3.0 * sigma_x)) + 1)
            + (2 * int(math.ceil(3.0 * sigma_y)) + 1),
        )
    if operation is ImageOperation.UNSHARP_MASK:
        sigma = max(0.0, float(parameters.get("sigma", 1.0)))
        kernel = 2 * int(math.ceil(3.0 * sigma)) + 1
        return max(2, kernel * 2 + 2)
    if operation in {ImageOperation.SOBEL_EDGES, ImageOperation.LAPLACIAN_EDGES}:
        kernel_size = max(1, int(parameters.get("kernel_size", 3)))
        return kernel_size * kernel_size
    if operation is ImageOperation.BACKGROUND_SUBTRACT:
        radius = max(0, int(parameters.get("radius", 25)))
        return max(1, (2 * radius + 1) ** 2 * 2)
    return 1


def resolve_image_operation_capability(
    operation: ImageOperation | str,
    parameters: Mapping[str, object] | None = None,
) -> ImageOperationCapability:
    """Resolve the safe execution capability for concrete parameters.

    The declaration is deliberately conservative.  An operation is marked
    tileable only when cropping a source patch with the declared halo produces
    the exact same core pixels as running the operation over the complete
    image.
    """

    resolved = (
        operation
        if isinstance(operation, ImageOperation)
        else _coerce_enum(ImageOperation, operation, "图像操作")
    )
    params = dict(parameters or {})
    if resolved is ImageOperation.COPY:
        return ImageOperationCapability(
            False,
            False,
            True,
            reason="COPY 在启用 ROI 时按包围框改变输出尺寸，必须整图执行。",
        )
    border_mode = str(params.get("border_mode", BorderMode.REFLECT.value))
    finite_neighborhood_with_border = {
        ImageOperation.GAUSSIAN_BLUR,
        ImageOperation.MEAN_FILTER,
        ImageOperation.BILATERAL_FILTER,
        ImageOperation.ERODE,
        ImageOperation.DILATE,
        ImageOperation.MORPHOLOGY_OPEN,
        ImageOperation.MORPHOLOGY_CLOSE,
        ImageOperation.TOP_HAT,
        ImageOperation.BLACK_HAT,
        ImageOperation.BACKGROUND_SUBTRACT,
        ImageOperation.CUSTOM_CONVOLUTION,
    }
    if (
        resolved in finite_neighborhood_with_border
        and border_mode == BorderMode.WRAP.value
    ):
        return ImageOperationCapability(
            False,
            True,
            True,
            reason=(
                "循环边界需要读取整幅图像对侧像素，"
                "不能用局部图块保证逐位一致。"
            ),
        )
    pointwise = {
        ImageOperation.COLOR_BALANCE,
        ImageOperation.BRIGHTNESS_CONTRAST,
        ImageOperation.BINARIZE,
        ImageOperation.ADD,
        ImageOperation.SUBTRACT,
        ImageOperation.MULTIPLY,
        ImageOperation.DIVIDE,
        ImageOperation.LOG,
        ImageOperation.EXP,
        ImageOperation.SQRT,
        ImageOperation.ABS,
        ImageOperation.IMAGE_CALCULATOR,
        ImageOperation.LOG_V2,
        ImageOperation.EXP_V2,
        ImageOperation.SQRT_V2,
    }
    if resolved in pointwise:
        if (
            resolved
            in {
                ImageOperation.LOG_V2,
                ImageOperation.EXP_V2,
                ImageOperation.SQRT_V2,
            }
            and str(params.get("result_mode", "float32")) == "remap"
        ):
            return ImageOperationCapability(
                False,
                True,
                True,
                requires_full_image_prescan=True,
                reason="重映射科学变换需要完整结果的有限值范围。",
            )
        return ImageOperationCapability(True, True, True)
    if resolved is ImageOperation.CONVERT_COLOR:
        return ImageOperationCapability(
            True,
            True,
            False,
            reason="颜色模型转换保持宽高，但不接受 ROI。",
        )
    if resolved is ImageOperation.CONVERT_TYPE:
        scale_mode = str(
            params.get("scale_mode", ConversionScaleMode.PRESERVE_VALUES.value)
        )
        if scale_mode == ConversionScaleMode.DATA_RANGE.value:
            return ImageOperationCapability(
                False,
                True,
                True,
                requires_full_image_prescan=True,
                reason="按数据范围转换需要扫描整幅图像的有限值范围。",
            )
        return ImageOperationCapability(True, True, True)
    if resolved is ImageOperation.ADJUST_LEVELS:
        explicit_range = "black_point" in params and "white_point" in params
        return ImageOperationCapability(
            explicit_range,
            True,
            True,
            requires_full_image_prescan=not explicit_range,
            reason=(
                ""
                if explicit_range
                else "未指定黑白场时需要扫描整幅图像的有限值范围。"
            ),
        )
    if resolved is ImageOperation.THRESHOLD:
        explicit_range = "lower" in params and "upper" in params
        return ImageOperationCapability(
            explicit_range,
            True,
            True,
            requires_full_image_prescan=not explicit_range,
            reason=(
                ""
                if explicit_range
                else "未指定阈值范围时需要扫描整幅图像。"
            ),
        )
    if resolved in {ImageOperation.INVERT, ImageOperation.GAMMA}:
        explicit_range = "minimum" in params and "maximum" in params
        return ImageOperationCapability(
            explicit_range,
            True,
            True,
            requires_full_image_prescan=not explicit_range,
            reason=(
                ""
                if explicit_range
                else "未指定运算范围时需要扫描整幅图像。"
            ),
        )
    if resolved is ImageOperation.CLAMP:
        explicit_range = "minimum" in params and "maximum" in params
        return ImageOperationCapability(
            explicit_range,
            True,
            True,
            requires_full_image_prescan=not explicit_range,
            reason=(
                ""
                if explicit_range
                else "未指定截断范围时需要扫描整幅图像。"
            ),
        )
    if resolved is ImageOperation.GAUSSIAN_BLUR:
        sigma_x = max(0.0, float(params.get("sigma_x", params.get("sigma", 1.0))))
        sigma_y = max(0.0, float(params.get("sigma_y", sigma_x)))
        # OpenCV derives a finite kernel for ksize=(0, 0).  Six sigma plus a
        # two-pixel guard is intentionally wider than all supported kernels.
        return ImageOperationCapability(
            True,
            True,
            True,
            halo_x=int(math.ceil(6.0 * sigma_x)) + 2,
            halo_y=int(math.ceil(6.0 * sigma_y)) + 2,
        )
    if resolved in {
        ImageOperation.MEDIAN_FILTER,
        ImageOperation.MEAN_FILTER,
        ImageOperation.REMOVE_OUTLIERS,
        ImageOperation.REPAIR_NONFINITE,
    }:
        radius = max(0, int(params.get("radius", 1)))
        return ImageOperationCapability(True, True, True, radius, radius)
    if resolved is ImageOperation.BILATERAL_FILTER:
        return ImageOperationCapability(
            False,
            True,
            True,
            reason=(
                "OpenCV 双边滤波在不同图块尺寸下可能产生浮点归约差异，"
                "为保证逐位一致而整图执行。"
            ),
        )
    if resolved in {
        ImageOperation.ADAPTIVE_THRESHOLD,
        ImageOperation.RANK_FILTER,
        ImageOperation.MORPHOLOGY_DERIVATIVE,
    }:
        radius = max(0, int(params.get("radius", 1)))
        return ImageOperationCapability(
            True,
            True,
            True,
            halo_x=radius,
            halo_y=radius,
        )
    if resolved is ImageOperation.UNSHARP_MASK:
        sigma = max(0.0, float(params.get("sigma", 1.0)))
        halo = int(math.ceil(6.0 * sigma)) + 2
        return ImageOperationCapability(True, True, True, halo, halo)
    if resolved in {ImageOperation.SOBEL_EDGES, ImageOperation.LAPLACIAN_EDGES}:
        kernel_size = max(1, int(params.get("kernel_size", 3)))
        radius = max(1, kernel_size // 2)
        return ImageOperationCapability(True, True, True, radius, radius)
    if resolved in {
        ImageOperation.ERODE,
        ImageOperation.DILATE,
        ImageOperation.MORPHOLOGY_OPEN,
        ImageOperation.MORPHOLOGY_CLOSE,
        ImageOperation.TOP_HAT,
        ImageOperation.BLACK_HAT,
    }:
        radius = max(0, int(params.get("radius", 1)))
        iterations = max(0, int(params.get("iterations", 1)))
        passes = (
            1
            if resolved in {ImageOperation.ERODE, ImageOperation.DILATE}
            else 2
        )
        halo = radius * iterations * passes
        return ImageOperationCapability(True, True, True, halo, halo)
    if resolved is ImageOperation.BACKGROUND_SUBTRACT:
        if bool(params.get("preserve_offset", False)):
            return ImageOperationCapability(
                False,
                True,
                True,
                requires_full_image_prescan=True,
                reason="保留背景偏移需要整幅图像的背景中位数。",
            )
        halo = max(0, int(params.get("radius", 25))) * 2
        return ImageOperationCapability(True, True, True, halo, halo)
    if resolved is ImageOperation.FLAT_FIELD_CORRECTION:
        source_mode = str(
            params.get("flat_field_source", "estimated")
        ).strip().lower()
        return ImageOperationCapability(
            False,
            True,
            True,
            requires_full_image_prescan=True,
            reason=(
                "参考平场需要使用完整参考图像的通道归一化值，"
                "必须整图单次执行。"
                if source_mode == "reference"
                else "估算照明场依赖整幅图像，必须整图单次执行。"
            ),
        )
    if resolved is ImageOperation.CUSTOM_CONVOLUTION:
        width = max(0, int(params.get("kernel_width", 0)))
        height = max(0, int(params.get("kernel_height", 0)))
        if width < 1 or height < 1 or width % 2 == 0 or height % 2 == 0:
            return ImageOperationCapability(
                False,
                True,
                True,
                reason="卷积核宽高尚未形成有效的正奇数尺寸。",
            )
        return ImageOperationCapability(
            True,
            True,
            True,
            width // 2,
            height // 2,
        )

    geometry_operations = {
        ImageOperation.FLIP_HORIZONTAL,
        ImageOperation.FLIP_VERTICAL,
        ImageOperation.ROTATE_90_CLOCKWISE,
        ImageOperation.ROTATE_90_COUNTERCLOCKWISE,
        ImageOperation.ROTATE_180,
        ImageOperation.ROTATE,
        ImageOperation.RESIZE,
        ImageOperation.TRANSLATE,
        ImageOperation.RESIZE_CANVAS,
        ImageOperation.PIXEL_BIN,
    }
    if resolved is ImageOperation.CROP:
        return ImageOperationCapability(
            False,
            False,
            True,
            reason="裁剪改变输出尺寸，必须整图执行，但可同步裁剪 ROI。",
        )
    if resolved in geometry_operations:
        return ImageOperationCapability(
            False,
            False,
            False,
            reason="几何操作改变坐标或输出尺寸，必须整图执行。",
        )

    return ImageOperationCapability(
        False,
        True,
        True,
        requires_full_image_prescan=True,
        reason="该操作依赖全局直方图、连通关系、边缘追踪或频域信息。",
    )


class BorderMode(str, Enum):
    REFLECT = "reflect"
    REPLICATE = "replicate"
    CONSTANT = "constant"
    WRAP = "wrap"


class MorphologyKernel(str, Enum):
    ELLIPSE = "ellipse"
    RECTANGLE = "rectangle"
    CROSS = "cross"


class InterpolationMode(str, Enum):
    AUTO = "auto"
    NEAREST = "nearest"
    LINEAR = "linear"
    CUBIC = "cubic"
    AREA = "area"
    LANCZOS = "lanczos"


@dataclass(frozen=True, slots=True)
class ImageOperationRequest:
    """An immutable copy of one image operation request.

    ``parameters`` is stored as a tuple to prevent caller mutation.  Prefer
    :meth:`create` at call sites for readable keyword arguments.
    """

    operation: ImageOperation
    image: NDArray[Any]
    secondary_image: NDArray[Any] | None = None
    parameters: tuple[tuple[str, ParameterValue], ...] = ()
    roi_mask: NDArray[np.bool_] | None = None
    request_id: str = ""
    generation: int = 0

    def __post_init__(self) -> None:
        operation = (
            self.operation
            if isinstance(self.operation, ImageOperation)
            else _coerce_enum(ImageOperation, self.operation, "图像操作")
        )
        image = _freeze_raster(self.image)
        secondary_image = (
            _freeze_raster(self.secondary_image)
            if self.secondary_image is not None
            else None
        )
        parameters = _freeze_parameters(self.parameters)
        roi_mask = (
            _freeze_roi_mask(self.roi_mask, image.shape[:2])
            if self.roi_mask is not None
            else None
        )
        object.__setattr__(self, "operation", operation)
        object.__setattr__(self, "image", image)
        object.__setattr__(self, "secondary_image", secondary_image)
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "roi_mask", roi_mask)
        object.__setattr__(self, "request_id", str(self.request_id))
        object.__setattr__(self, "generation", int(self.generation))

    @classmethod
    def create(
        cls,
        operation: ImageOperation | str,
        image: NDArray[Any],
        *,
        secondary_image: NDArray[Any] | None = None,
        roi_mask: NDArray[np.bool_] | None = None,
        request_id: str = "",
        generation: int = 0,
        **parameters: ParameterValue,
    ) -> "ImageOperationRequest":
        return cls(
            operation=(
                operation
                if isinstance(operation, ImageOperation)
                else _coerce_enum(ImageOperation, operation, "图像操作")
            ),
            image=image,
            secondary_image=secondary_image,
            parameters=tuple(sorted(parameters.items())),
            roi_mask=roi_mask,
            request_id=request_id,
            generation=generation,
        )

    @property
    def parameter_map(self) -> Mapping[str, ParameterValue]:
        return MappingProxyType(dict(self.parameters))


@dataclass(frozen=True, slots=True)
class ImageOperationResult:
    operation: ImageOperation
    image: NDArray[Any]
    source_dtype: str
    output_dtype: str
    warnings: tuple[str, ...] = ()
    metadata: tuple[tuple[str, ParameterValue], ...] = ()
    request_id: str = ""
    generation: int = 0
    roi_mask: NDArray[np.bool_] | None = None

    def __post_init__(self) -> None:
        image = _freeze_raster(self.image)
        roi_mask = (
            _freeze_roi_mask(self.roi_mask, image.shape[:2])
            if self.roi_mask is not None
            else None
        )
        object.__setattr__(self, "image", image)
        object.__setattr__(self, "roi_mask", roi_mask)
        object.__setattr__(self, "warnings", tuple(str(item) for item in self.warnings))
        object.__setattr__(self, "metadata", _freeze_parameters(self.metadata))

    @property
    def metadata_map(self) -> Mapping[str, ParameterValue]:
        return MappingProxyType(dict(self.metadata))


def execute_image_operation(
    request: ImageOperationRequest,
    *,
    cancellation_check: Callable[[], None] | None = None,
) -> ImageOperationResult:
    """Execute one validated image operation."""

    image = np.asarray(request.image)
    params = dict(request.parameters)
    operation = request.operation
    warnings: list[str] = []
    metadata: dict[str, ParameterValue] = dict(request.parameters)
    output_roi_mask: NDArray[np.bool_] | None = None

    if operation is ImageOperation.COPY:
        processed = image.copy()
        roi_mode = str(params.get("roi_mode", "bounds")).strip().lower()
        if roi_mode not in {"bounds", "mask"}:
            raise ValueError("复制 ROI 模式必须为 bounds 或 mask。")
        transparent = bool(params.get("transparent_outside", False))
        if transparent and roi_mode != "mask":
            raise ValueError(
                "transparent_outside 仅适用于 roi_mode=mask。"
            )
        if request.roi_mask is not None:
            rows, columns = np.nonzero(request.roi_mask)
            if rows.size == 0:
                raise ValueError("复制操作的 ROI 不能为空。")
            x0 = int(np.min(columns))
            x1 = int(np.max(columns)) + 1
            y0 = int(np.min(rows))
            y1 = int(np.max(rows)) + 1
            processed = np.ascontiguousarray(image[y0:y1, x0:x1])
            output_roi_mask = np.ascontiguousarray(
                request.roi_mask[y0:y1, x0:x1],
                dtype=bool,
            )
            if roi_mode == "mask":
                if transparent:
                    processed = _crop_with_transparent_outside(
                        processed,
                        output_roi_mask,
                    )
                else:
                    processed = _fill_outside_mask(
                        processed,
                        output_roi_mask,
                        params.get(
                            "fill_value",
                            params.get("outside_value", 0.0),
                        ),
                    )
            metadata.update(
                roi_mode=roi_mode,
                transparent_outside=transparent,
                roi_bounds=(x0, y0, x1 - x0, y1 - y0),
            )
        elif roi_mode == "mask" or transparent:
            raise ValueError("mask/transparent_outside 复制需要 ROI。")
    elif operation is ImageOperation.CONVERT_TYPE:
        target = _coerce_enum(
            PixelType,
            params.get("target_type", PixelType.UINT8.value),
            "目标位深",
        )
        mode = _coerce_enum(
            ConversionScaleMode,
            params.get("scale_mode", ConversionScaleMode.PRESERVE_VALUES.value),
            "位深转换缩放模式",
        )
        nonfinite_policy = _coerce_enum(
            NonfiniteIntegerPolicy,
            params.get(
                "nonfinite_policy",
                NonfiniteIntegerPolicy.REJECT.value,
            ),
            "非有限数替代规则",
        )
        replacement_count = _integer_conversion_nonfinite_count(
            image,
            target,
        )
        processed = convert_pixel_type(
            image,
            target,
            mode=mode,
            nonfinite_policy=nonfinite_policy,
            statistics_mask=request.roi_mask,
        )
        if request.roi_mask is not None:
            source_as_target = convert_pixel_type(
                image,
                target,
                mode=ConversionScaleMode.PRESERVE_VALUES,
                nonfinite_policy=nonfinite_policy,
            )
            processed = _blend_roi(source_as_target, processed, request.roi_mask)
        metadata.update(
            target_type=target.value,
            scale_mode=mode.value,
            nonfinite_policy=nonfinite_policy.value,
            nonfinite_replacement_count=replacement_count,
        )
    elif operation is ImageOperation.CONVERT_COLOR:
        _reject_roi_for_geometry(request)
        target_model = _coerce_enum(
            ColorTarget,
            params.get("target_model", ColorTarget.GRAYSCALE.value),
            "颜色模型",
        )
        grayscale_method = _coerce_enum(
            GrayscaleMethod,
            params.get("grayscale_method", GrayscaleMethod.REC601.value),
            "灰度换算方式",
        )
        drop_alpha = bool(params.get("drop_alpha", False))
        processed = convert_color_model(
            image,
            target=target_model,
            grayscale_method=grayscale_method,
            drop_alpha=drop_alpha,
        )
        metadata.update(
            target_model=target_model.value,
            grayscale_method=grayscale_method.value,
            drop_alpha=drop_alpha,
        )
    elif operation is ImageOperation.COLOR_BALANCE:
        processed = adjust_color_balance(
            image,
            red_gain=float(params.get("red_gain", 1.0)),
            green_gain=float(params.get("green_gain", 1.0)),
            blue_gain=float(params.get("blue_gain", 1.0)),
            red_offset=float(params.get("red_offset", 0.0)),
            green_offset=float(params.get("green_offset", 0.0)),
            blue_offset=float(params.get("blue_offset", 0.0)),
        )
        processed = _blend_roi(image, processed, request.roi_mask)
    elif operation is ImageOperation.BRIGHTNESS_CONTRAST:
        processed = adjust_brightness_contrast(
            image,
            brightness=float(params.get("brightness", 0.0)),
            contrast=float(params.get("contrast", 1.0)),
            gamma=float(params.get("gamma", 1.0)),
        )
        processed = _blend_roi(image, processed, request.roi_mask)
    elif operation is ImageOperation.ADJUST_LEVELS:
        statistics_values = _roi_statistics_values(image, request.roi_mask)
        processed = adjust_levels(
            image,
            black_point=float(
                params.get("black_point", _finite_min(statistics_values))
            ),
            white_point=float(
                params.get("white_point", _finite_max(statistics_values))
            ),
            output_min=(
                None
                if params.get("output_min") is None
                else float(params["output_min"])
            ),
            output_max=(
                None
                if params.get("output_max") is None
                else float(params["output_max"])
            ),
            gamma=float(params.get("gamma", 1.0)),
        )
        processed = _blend_roi(image, processed, request.roi_mask)
    elif operation is ImageOperation.THRESHOLD:
        threshold_channel = params.get("channel")
        threshold_scalar = _require_scalar_image(image, threshold_channel)
        threshold_statistics = _roi_statistics_values(
            threshold_scalar,
            request.roi_mask,
        )
        processed = threshold_image(
            image,
            lower=float(params.get("lower", _finite_min(threshold_statistics))),
            upper=float(params.get("upper", _finite_max(threshold_statistics))),
            invert=bool(params.get("invert", False)),
            foreground_value=(
                None
                if params.get("foreground_value") is None
                else float(params["foreground_value"])
            ),
            background_value=(
                None
                if params.get("background_value") is None
                else float(params["background_value"])
            ),
            channel=None if threshold_channel is None else str(threshold_channel),
        )
        threshold_source = _roi_source_for_output(
            image,
            processed,
            channel=threshold_channel,
        )
        processed = _blend_roi(threshold_source, processed, request.roi_mask)
    elif operation is ImageOperation.FLIP_HORIZONTAL:
        _reject_roi_for_geometry(request)
        processed = np.flip(image, axis=1)
    elif operation is ImageOperation.FLIP_VERTICAL:
        _reject_roi_for_geometry(request)
        processed = np.flip(image, axis=0)
    elif operation is ImageOperation.ROTATE_90_CLOCKWISE:
        _reject_roi_for_geometry(request)
        processed = np.rot90(image, k=3)
    elif operation is ImageOperation.ROTATE_90_COUNTERCLOCKWISE:
        _reject_roi_for_geometry(request)
        processed = np.rot90(image, k=1)
    elif operation is ImageOperation.ROTATE_180:
        _reject_roi_for_geometry(request)
        processed = np.rot90(image, k=2)
    elif operation is ImageOperation.ROTATE:
        _reject_roi_for_geometry(request)
        angle = float(params.get("angle_degrees", 0.0))
        expand = bool(params.get("expand", True))
        interpolation = _coerce_enum(
            InterpolationMode,
            params.get("interpolation", InterpolationMode.LINEAR.value),
            "插值模式",
        )
        border_mode = _coerce_enum(
            BorderMode,
            params.get("border_mode", BorderMode.CONSTANT.value),
            "边界模式",
        )
        border_value = float(params.get("border_value", 0.0))
        processed = rotate_image(
            image,
            angle_degrees=angle,
            expand=expand,
            interpolation=interpolation,
            border_mode=border_mode,
            border_value=border_value,
        )
        metadata.update(
            angle_degrees=angle,
            expand=expand,
            interpolation=interpolation.value,
            border_mode=border_mode.value,
        )
    elif operation is ImageOperation.CROP:
        x = int(params.get("x", 0))
        y = int(params.get("y", 0))
        width = int(params.get("width", image.shape[1]))
        height = int(params.get("height", image.shape[0]))
        processed = crop_image(
            image,
            x=x,
            y=y,
            width=width,
            height=height,
        )
        roi_mode = str(params.get("roi_mode", "bounds")).strip().lower()
        if roi_mode not in {"bounds", "mask"}:
            raise ValueError("裁剪 ROI 模式必须为 bounds 或 mask。")
        transparent = bool(params.get("transparent_outside", False))
        if transparent and roi_mode != "mask":
            raise ValueError(
                "transparent_outside 仅适用于 roi_mode=mask。"
            )
        if request.roi_mask is None and (roi_mode == "mask" or transparent):
            raise ValueError("mask/transparent_outside 裁剪需要 ROI。")
        if request.roi_mask is not None:
            output_roi_mask = np.ascontiguousarray(
                request.roi_mask[y : y + height, x : x + width],
                dtype=bool,
            )
            if roi_mode == "mask":
                if transparent:
                    processed = _crop_with_transparent_outside(
                        processed,
                        output_roi_mask,
                    )
                else:
                    processed = _fill_outside_mask(
                        processed,
                        output_roi_mask,
                        params.get(
                            "fill_value",
                            params.get("outside_value", 0.0),
                        ),
                    )
            metadata.update(
                roi_mode=roi_mode,
                transparent_outside=transparent,
            )
    elif operation is ImageOperation.RESIZE:
        _reject_roi_for_geometry(request)
        width = int(params.get("width", image.shape[1]))
        height = int(params.get("height", image.shape[0]))
        requested_interpolation = _coerce_enum(
            InterpolationMode,
            params.get("interpolation", InterpolationMode.AUTO.value),
            "插值模式",
        )
        interpolation = resolve_resize_interpolation(
            source_width=int(image.shape[1]),
            source_height=int(image.shape[0]),
            width=width,
            height=height,
            requested=requested_interpolation,
        )
        processed = resize_image(
            image,
            width=width,
            height=height,
            interpolation=interpolation,
        )
        metadata.update(width=width, height=height, interpolation=interpolation.value)
    elif operation is ImageOperation.TRANSLATE:
        _reject_roi_for_geometry(request)
        offset_x = float(params.get("offset_x", 0.0))
        offset_y = float(params.get("offset_y", 0.0))
        interpolation = _coerce_enum(
            InterpolationMode,
            params.get("interpolation", InterpolationMode.LINEAR.value),
            "插值模式",
        )
        border_mode = _coerce_enum(
            BorderMode,
            params.get("border_mode", BorderMode.CONSTANT.value),
            "边界模式",
        )
        border_value = params.get("border_value", 0.0)
        processed = translate_image(
            image,
            offset_x=offset_x,
            offset_y=offset_y,
            interpolation=interpolation,
            border_mode=border_mode,
            border_value=border_value,
        )
        metadata.update(
            offset_x=offset_x,
            offset_y=offset_y,
            interpolation=interpolation.value,
            border_mode=border_mode.value,
        )
    elif operation is ImageOperation.RESIZE_CANVAS:
        _reject_roi_for_geometry(request)
        width = _positive_integer_parameter(
            params.get("width", image.shape[1]),
            field_name="画布宽度",
        )
        height = _positive_integer_parameter(
            params.get("height", image.shape[0]),
            field_name="画布高度",
        )
        anchor = _coerce_enum(
            CanvasAnchor,
            params.get("anchor", CanvasAnchor.CENTER.value),
            "画布锚点",
        )
        processed = resize_canvas(
            image,
            width=width,
            height=height,
            anchor=anchor,
            fill_value=params.get("fill_value", 0.0),
        )
        metadata.update(width=width, height=height, anchor=anchor.value)
    elif operation is ImageOperation.PIXEL_BIN:
        _reject_roi_for_geometry(request)
        factor = _positive_integer_parameter(
            params.get("factor", 2),
            field_name="像素合并系数",
        )
        method = _coerce_enum(
            PixelBinMethod,
            params.get("method", PixelBinMethod.MEAN.value),
            "像素合并方式",
        )
        remainder_policy = _coerce_enum(
            PixelBinRemainderPolicy,
            params.get(
                "remainder_policy",
                PixelBinRemainderPolicy.REJECT.value,
            ),
            "余数处理方式",
        )
        processed, cropped_right, cropped_bottom = pixel_bin(
            image,
            factor=factor,
            method=method,
            remainder_policy=remainder_policy,
        )
        metadata.update(
            factor=factor,
            method=method.value,
            remainder_policy=remainder_policy.value,
            cropped_right=cropped_right,
            cropped_bottom=cropped_bottom,
        )
    elif operation is ImageOperation.GAUSSIAN_BLUR:
        sigma_x = float(params.get("sigma_x", params.get("sigma", 1.0)))
        sigma_y = float(params.get("sigma_y", sigma_x))
        border_mode = _coerce_enum(
            BorderMode,
            params.get("border_mode", BorderMode.REFLECT.value),
            "边界模式",
        )
        processed = gaussian_blur(
            image,
            sigma_x=sigma_x,
            sigma_y=sigma_y,
            border_mode=border_mode,
        )
        processed = _blend_roi(image, processed, request.roi_mask)
    elif operation is ImageOperation.MEDIAN_FILTER:
        radius = int(params.get("radius", 1))
        processed = median_filter(image, radius=radius)
        processed = _blend_roi(image, processed, request.roi_mask)
    elif operation is ImageOperation.MEAN_FILTER:
        radius = int(params.get("radius", 1))
        border_mode = _coerce_enum(
            BorderMode,
            params.get("border_mode", BorderMode.REFLECT.value),
            "边界模式",
        )
        processed = mean_filter(image, radius=radius, border_mode=border_mode)
        processed = _blend_roi(image, processed, request.roi_mask)
    elif operation is ImageOperation.BILATERAL_FILTER:
        processed = bilateral_filter(
            image,
            diameter=int(params.get("diameter", 5)),
            sigma_color=float(params.get("sigma_color", 25.0)),
            sigma_space=float(params.get("sigma_space", 2.0)),
            border_mode=_coerce_enum(
                BorderMode,
                params.get("border_mode", BorderMode.REFLECT.value),
                "边界模式",
            ),
        )
        processed = _blend_roi(image, processed, request.roi_mask)
    elif operation is ImageOperation.UNSHARP_MASK:
        sigma = float(params.get("sigma", 1.0))
        amount = float(params.get("amount", 1.0))
        threshold = float(params.get("threshold", 0.0))
        processed = unsharp_mask(
            image,
            sigma=sigma,
            amount=amount,
            threshold=threshold,
        )
        processed = _blend_roi(image, processed, request.roi_mask)
    elif operation is ImageOperation.SOBEL_EDGES:
        processed = sobel_edges(
            image,
            kernel_size=int(params.get("kernel_size", 3)),
            channel=str(params.get("channel", "luminance")),
            output_float=bool(params.get("output_float", True)),
        )
        if request.roi_mask is not None:
            if image.ndim != processed.ndim:
                source = _select_scalar_channel(image, str(params.get("channel", "luminance")))
            else:
                source = image
            processed = _blend_roi(
                _cast_like(source, processed.dtype),
                processed,
                request.roi_mask,
            )
    elif operation is ImageOperation.LAPLACIAN_EDGES:
        channel = params.get("channel")
        processed = laplacian_edges(
            image,
            kernel_size=int(params.get("kernel_size", 3)),
            channel=None if channel is None else str(channel),
            output_float=bool(params.get("output_float", True)),
        )
        source = _roi_source_for_output(image, processed, channel=channel)
        processed = _blend_roi(
            _cast_like(source, processed.dtype),
            processed,
            request.roi_mask,
        )
    elif operation is ImageOperation.CANNY_EDGES:
        channel = params.get("channel")
        processed = canny_edges(
            image,
            threshold_low=float(params.get("threshold_low", 50.0)),
            threshold_high=float(params.get("threshold_high", 150.0)),
            aperture_size=int(params.get("aperture_size", 3)),
            l2_gradient=bool(params.get("l2_gradient", True)),
            channel=None if channel is None else str(channel),
        )
        source = _roi_source_for_output(image, processed, channel=channel)
        processed = _blend_roi(
            _cast_like(source, processed.dtype),
            processed,
            request.roi_mask,
        )
    elif operation is ImageOperation.NORMALIZE:
        processed = normalize_image(
            image,
            output_min=float(params.get("output_min", _working_range(image)[0])),
            output_max=float(params.get("output_max", _working_range(image)[1])),
            per_channel=bool(params.get("per_channel", True)),
            statistics_mask=request.roi_mask,
        )
        processed = _blend_roi(image, processed, request.roi_mask)
    elif operation is ImageOperation.HISTOGRAM_EQUALIZATION:
        processed = equalize_histogram(
            image,
            statistics_mask=request.roi_mask,
        )
        processed = _blend_roi(image, processed, request.roi_mask)
    elif operation is ImageOperation.CLAHE:
        processed = clahe(
            image,
            clip_limit=float(params.get("clip_limit", 2.0)),
            tile_grid_size=int(params.get("tile_grid_size", 8)),
        )
        processed = _blend_roi(image, processed, request.roi_mask)
    elif operation is ImageOperation.REMOVE_OUTLIERS:
        processed = remove_outliers(
            image,
            radius=int(params.get("radius", 1)),
            threshold=float(params.get("threshold", 25.0)),
            polarity=str(params.get("polarity", "both")),
        )
        processed = _blend_roi(image, processed, request.roi_mask)
    elif operation is ImageOperation.REPAIR_NONFINITE:
        processed, repaired_count = repair_nonfinite(
            image,
            radius=int(params.get("radius", 1)),
            fallback_value=float(params.get("fallback_value", 0.0)),
        )
        processed = _blend_roi(image, processed, request.roi_mask)
        metadata["repaired_count"] = repaired_count
    elif operation is ImageOperation.AUTO_THRESHOLD:
        channel = params.get("channel")
        processed, threshold = auto_threshold(
            image,
            method=str(params.get("method", "otsu")),
            channel=None if channel is None else str(channel),
            invert=bool(params.get("invert", False)),
            statistics_mask=request.roi_mask,
        )
        source = _roi_source_for_output(image, processed, channel=channel)
        processed = _blend_roi(
            _cast_like(source, processed.dtype),
            processed,
            request.roi_mask,
        )
        metadata["computed_threshold"] = threshold
    elif operation is ImageOperation.BINARIZE:
        channel = params.get("channel")
        processed = binarize_image(
            image,
            threshold=float(params.get("threshold", 0.0)),
            channel=None if channel is None else str(channel),
            invert=bool(params.get("invert", False)),
        )
        source = _roi_source_for_output(image, processed, channel=channel)
        processed = _blend_roi(
            _cast_like(source, processed.dtype),
            processed,
            request.roi_mask,
        )
    elif operation in {
        ImageOperation.ERODE,
        ImageOperation.DILATE,
        ImageOperation.MORPHOLOGY_OPEN,
        ImageOperation.MORPHOLOGY_CLOSE,
        ImageOperation.TOP_HAT,
        ImageOperation.BLACK_HAT,
    }:
        processed = morphology_filter(
            image,
            operation=operation,
            radius=int(params.get("radius", 1)),
            iterations=int(params.get("iterations", 1)),
            kernel=_coerce_enum(
                MorphologyKernel,
                params.get("kernel", MorphologyKernel.ELLIPSE.value),
                "形态学核",
            ),
            border_mode=_coerce_enum(
                BorderMode,
                params.get("border_mode", BorderMode.REFLECT.value),
                "边界模式",
            ),
            channel=(
                None
                if params.get("channel") is None
                else str(params.get("channel"))
            ),
        )
        source = _roi_source_for_output(image, processed, channel=params.get("channel"))
        processed = _blend_roi(source, processed, request.roi_mask)
    elif operation is ImageOperation.FILL_HOLES:
        scalar = _require_scalar_image(image, params.get("channel"))
        scalar_for_processing = _isolate_roi_domain(
            scalar,
            request.roi_mask,
            foreground_is_high=bool(params.get("foreground_is_high", True)),
        )
        processed = fill_binary_holes(
            scalar_for_processing,
            foreground_is_high=bool(params.get("foreground_is_high", True)),
            cancellation_check=cancellation_check,
        )
        processed = _blend_roi(scalar, processed, request.roi_mask)
    elif operation is ImageOperation.CONTOUR_EXTRACT:
        scalar = _require_scalar_image(image, params.get("channel"))
        scalar_for_processing = _isolate_roi_domain(
            scalar,
            request.roi_mask,
            foreground_is_high=bool(params.get("foreground_is_high", True)),
        )
        processed = extract_binary_contours(
            scalar_for_processing,
            foreground_is_high=bool(params.get("foreground_is_high", True)),
        )
        processed = _blend_roi(scalar, processed, request.roi_mask)
    elif operation is ImageOperation.REMOVE_SMALL_OBJECTS:
        scalar = _require_scalar_image(image, params.get("channel"))
        scalar_for_processing = _isolate_roi_domain(
            scalar,
            request.roi_mask,
            foreground_is_high=bool(params.get("foreground_is_high", True)),
        )
        processed = remove_small_objects(
            scalar_for_processing,
            minimum_area=int(params.get("minimum_area", 10)),
            connectivity=int(params.get("connectivity", 8)),
            foreground_is_high=bool(params.get("foreground_is_high", True)),
        )
        processed = _blend_roi(scalar, processed, request.roi_mask)
    elif operation is ImageOperation.FILL_SMALL_HOLES:
        scalar = _require_scalar_image(image, params.get("channel"))
        scalar_for_processing = _isolate_roi_domain(
            scalar,
            request.roi_mask,
            foreground_is_high=bool(params.get("foreground_is_high", True)),
        )
        processed = fill_small_holes(
            scalar_for_processing,
            maximum_area=int(params.get("maximum_area", 10)),
            connectivity=int(params.get("connectivity", 8)),
            foreground_is_high=bool(params.get("foreground_is_high", True)),
        )
        processed = _blend_roi(scalar, processed, request.roi_mask)
    elif operation is ImageOperation.DISTANCE_TRANSFORM:
        scalar = _require_scalar_image(image, params.get("channel"))
        scalar_for_processing = _isolate_roi_domain(
            scalar,
            request.roi_mask,
            foreground_is_high=bool(params.get("foreground_is_high", True)),
        )
        processed = distance_transform(
            scalar_for_processing,
            foreground_is_high=bool(params.get("foreground_is_high", True)),
            distance_type=str(params.get("distance_type", "l2")),
        )
        processed = _blend_roi(
            _cast_like(scalar, processed.dtype),
            processed,
            request.roi_mask,
        )
    elif operation is ImageOperation.SKELETONIZE:
        scalar = _require_scalar_image(image, params.get("channel"))
        scalar_for_processing = _isolate_roi_domain(
            scalar,
            request.roi_mask,
            foreground_is_high=bool(params.get("foreground_is_high", True)),
        )
        processed = skeletonize_binary(
            scalar_for_processing,
            foreground_is_high=bool(params.get("foreground_is_high", True)),
        )
        processed = _blend_roi(scalar, processed, request.roi_mask)
    elif operation is ImageOperation.WATERSHED:
        scalar = _require_scalar_image(image, params.get("channel"))
        scalar_for_processing = _isolate_roi_domain(
            scalar,
            request.roi_mask,
            foreground_is_high=bool(params.get("foreground_is_high", True)),
        )
        processed = watershed_split(
            scalar_for_processing,
            foreground_is_high=bool(params.get("foreground_is_high", True)),
            seed_threshold=float(params.get("seed_threshold", 0.45)),
        )
        processed = _blend_roi(scalar, processed, request.roi_mask)
    elif operation is ImageOperation.WATERSHED_V2:
        scalar = _require_scalar_image(image, params.get("channel"))
        scalar_for_processing = _isolate_roi_domain(
            scalar,
            request.roi_mask,
            foreground_is_high=bool(params.get("foreground_is_high", True)),
        )
        processed = watershed_split_v2(
            scalar_for_processing,
            foreground_is_high=bool(params.get("foreground_is_high", True)),
            seed_threshold=float(params.get("seed_threshold", 0.35)),
            minimum_seed_area=int(params.get("minimum_seed_area", 1)),
        )
        processed = _blend_roi(scalar, processed, request.roi_mask)
    elif operation is ImageOperation.BACKGROUND_SUBTRACT:
        processed = subtract_background(
            image,
            radius=int(params.get("radius", 25)),
            light_background=bool(params.get("light_background", False)),
            preserve_offset=bool(params.get("preserve_offset", False)),
            border_mode=_coerce_enum(
                BorderMode,
                params.get("border_mode", BorderMode.REFLECT.value),
                "边界模式",
            ),
        )
        processed = _blend_roi(image, processed, request.roi_mask)
    elif operation is ImageOperation.ROLLING_BALL_BACKGROUND_SUBTRACT:
        processed = rolling_ball_background_subtract(
            image,
            radius=float(params.get("radius", 25.0)),
            ball_height=float(params.get("ball_height", 255.0)),
            light_background=bool(params.get("light_background", False)),
            preserve_offset=bool(params.get("preserve_offset", False)),
        )
        processed = _blend_roi(image, processed, request.roi_mask)
    elif operation is ImageOperation.CUSTOM_CONVOLUTION:
        processed = custom_convolution(
            image,
            kernel=params.get("kernel", ()),
            kernel_width=int(params.get("kernel_width", 0)),
            kernel_height=int(params.get("kernel_height", 0)),
            normalize_kernel=bool(params.get("normalize_kernel", False)),
            offset=float(params.get("offset", 0.0)),
            border_mode=_coerce_enum(
                BorderMode,
                params.get("border_mode", BorderMode.REFLECT.value),
                "边界模式",
            ),
        )
        processed = _blend_roi(image, processed, request.roi_mask)
    elif operation in {
        ImageOperation.INVERT,
        ImageOperation.ADD,
        ImageOperation.SUBTRACT,
        ImageOperation.MULTIPLY,
        ImageOperation.DIVIDE,
        ImageOperation.GAMMA,
        ImageOperation.LOG,
        ImageOperation.EXP,
        ImageOperation.SQRT,
        ImageOperation.ABS,
        ImageOperation.CLAMP,
    }:
        processed = apply_math_operation(image, operation=operation, **params)
        processed = _blend_roi(image, processed, request.roi_mask)
    elif operation in {
        ImageOperation.LOG_V2,
        ImageOperation.EXP_V2,
        ImageOperation.SQRT_V2,
    }:
        processed = apply_scientific_math_transform(
            image,
            operation=operation,
            result_mode=str(params.get("result_mode", "float32")),
            output_min=float(params.get("output_min", 0.0)),
            output_max=float(params.get("output_max", 1.0)),
        )
        roi_source = (
            image.astype(np.float32)
            if processed.dtype == np.dtype(np.float32)
            else image
        )
        processed = _blend_roi(roi_source, processed, request.roi_mask)
    elif operation is ImageOperation.IMAGE_CALCULATOR:
        if request.secondary_image is None:
            raise ValueError("图像计算器需要提供第二幅图像。")
        processed = image_calculator(
            image,
            request.secondary_image,
            operation=str(params.get("calculator_operation", "add")),
            result_mode=str(params.get("result_mode", "preserve")),
        )
        roi_source = (
            image.astype(np.float32)
            if processed.dtype == np.dtype(np.float32)
            else image
        )
        processed = _blend_roi(roi_source, processed, request.roi_mask)
    elif operation is ImageOperation.FFT_FILTER:
        processed = fft_filter(
            image,
            mode=str(params.get("mode", "lowpass")),
            low_cutoff=float(params.get("low_cutoff", 0.0)),
            high_cutoff=float(params.get("high_cutoff", 0.15)),
            order=int(params.get("order", 2)),
            channel=str(params.get("channel", "per_channel")),
            output_float=bool(params.get("output_float", False)),
            boundary=str(params.get("boundary", "periodic")),
            tukey_alpha=float(params.get("tukey_alpha", 0.25)),
            frequency_unit=str(
                params.get("frequency_unit", "cycles_per_pixel")
            ),
            pixel_size=(
                None
                if params.get("pixel_size") is None
                else float(params["pixel_size"])
            ),
        )
        if request.roi_mask is not None:
            source = image if image.ndim == processed.ndim else _select_scalar_channel(
                image,
                str(params.get("channel", "luminance")),
            )
            processed = _blend_roi(
                _cast_like(source, processed.dtype),
                processed,
                request.roi_mask,
            )
    elif operation is ImageOperation.STRIPE_SUPPRESSION:
        processed = suppress_stripes(
            image,
            direction=str(params.get("direction", "horizontal")),
            notch_width=float(params.get("notch_width", 0.02)),
            protect_radius=float(params.get("protect_radius", 0.02)),
            strength=float(params.get("strength", 1.0)),
        )
        processed = _blend_roi(image, processed, request.roi_mask)
    elif operation is ImageOperation.ADAPTIVE_THRESHOLD:
        scalar = _require_scalar_image(image, params.get("channel"))
        processed = adaptive_threshold(
            scalar,
            method=str(params.get("method", "gaussian")),
            radius=int(params.get("radius", 7)),
            offset=float(params.get("offset", 0.0)),
            k=float(params.get("k", 0.2)),
            r=float(params.get("r", 128.0)),
            p=float(params.get("p", 2.0)),
            q=float(params.get("q", 10.0)),
            foreground_is_high=bool(params.get("foreground_is_high", True)),
        )
        processed = _blend_roi(scalar, processed, request.roi_mask)
    elif operation is ImageOperation.PERCENTILE_SATURATION:
        processed = percentile_saturation_enhance(
            image,
            lower_percentile=float(params.get("lower_percentile", 0.5)),
            upper_percentile=float(params.get("upper_percentile", 99.5)),
            per_channel=bool(params.get("per_channel", True)),
            statistics_mask=request.roi_mask,
        )
        processed = _blend_roi(image, processed, request.roi_mask)
    elif operation is ImageOperation.RANK_FILTER:
        processed = rank_filter(
            image,
            method=str(params.get("method", "minimum")),
            radius=int(params.get("radius", 1)),
        )
        roi_source = (
            image.astype(np.float32)
            if processed.dtype == np.dtype(np.float32)
            else image
        )
        processed = _blend_roi(roi_source, processed, request.roi_mask)
    elif operation is ImageOperation.MORPHOLOGY_DERIVATIVE:
        processed = morphology_derivative(
            image,
            method=str(params.get("method", "gradient")),
            radius=int(params.get("radius", 1)),
            channel=(
                None
                if params.get("channel") is None
                else str(params["channel"])
            ),
        )
        source = _roi_source_for_output(
            image,
            processed,
            channel=params.get("channel"),
        )
        processed = _blend_roi(source, processed, request.roi_mask)
    elif operation is ImageOperation.MORPHOLOGICAL_RECONSTRUCTION:
        scalar = _require_scalar_image(image, params.get("channel"))
        scalar_for_processing = _isolate_roi_domain(
            scalar,
            request.roi_mask,
            foreground_is_high=True,
        )
        processed = morphological_reconstruction(
            scalar_for_processing,
            method=str(params.get("method", "opening")),
            radius=int(params.get("radius", 1)),
            connectivity=int(params.get("connectivity", 8)),
        )
        processed = _blend_roi(scalar, processed, request.roi_mask)
    elif operation is ImageOperation.REGIONAL_EXTREMA:
        scalar = _require_scalar_image(image, params.get("channel"))
        extrema_kind = str(params.get("kind", "maxima"))
        scalar_for_processing = _isolate_roi_domain(
            scalar,
            request.roi_mask,
            foreground_is_high=extrema_kind == "maxima",
        )
        processed = regional_extrema(
            scalar_for_processing,
            kind=extrema_kind,
            h=float(params.get("h", 0.0)),
            connectivity=int(params.get("connectivity", 8)),
        )
        processed = _blend_roi(
            _cast_like(scalar, processed.dtype),
            processed,
            request.roi_mask,
        )
    elif operation is ImageOperation.CLEAR_BORDER:
        scalar = _require_scalar_image(image, params.get("channel"))
        scalar_for_processing = _isolate_roi_domain(
            scalar,
            request.roi_mask,
            foreground_is_high=bool(params.get("foreground_is_high", True)),
        )
        processed = clear_border_objects(
            scalar_for_processing,
            foreground_is_high=bool(params.get("foreground_is_high", True)),
            connectivity=int(params.get("connectivity", 8)),
        )
        processed = _blend_roi(scalar, processed, request.roi_mask)
    elif operation is ImageOperation.FLAT_FIELD_CORRECTION:
        source_mode = str(
            params.get("flat_field_source", "estimated")
        ).strip().lower()
        if source_mode not in {"estimated", "reference"}:
            raise ValueError("平场来源必须为 estimated 或 reference。")
        if source_mode == "reference" and request.secondary_image is None:
            raise ValueError("参考图平场校正需要提供第二幅参考图像。")
        processed = flat_field_correction(
            image,
            radius=float(params.get("radius", 25.0)),
            method=str(params.get("method", "gaussian")),
            preserve_mean=bool(params.get("preserve_mean", True)),
            reference_image=(
                request.secondary_image
                if source_mode == "reference"
                else None
            ),
            reference_levels=(
                params.get("reference_levels")
                if source_mode == "reference"
                else None
            ),
        )
        processed = _blend_roi(image, processed, request.roi_mask)
    elif operation is ImageOperation.FFT_POWER_SPECTRUM:
        scalar = _require_scalar_image(image, params.get("channel"))
        processed = fft_power_spectrum(
            scalar,
            logarithmic=bool(params.get("logarithmic", True)),
            centered=bool(params.get("centered", True)),
            window=str(params.get("window", "none")),
            tukey_alpha=float(params.get("tukey_alpha", 0.25)),
        )
        processed = _blend_roi(
            scalar.astype(np.float32),
            processed,
            request.roi_mask,
        )
    else:  # pragma: no cover - exhaustive enum guard
        raise ValueError(f"不支持的图像操作：{operation.value}")

    output = np.ascontiguousarray(processed)
    return ImageOperationResult(
        operation=operation,
        image=output,
        source_dtype=str(image.dtype),
        output_dtype=str(output.dtype),
        warnings=tuple(warnings),
        metadata=tuple(sorted(metadata.items())),
        request_id=request.request_id,
        generation=request.generation,
        roi_mask=output_roi_mask,
    )


def execute_image_operation_tiled(
    operation: ImageOperation | str,
    image: NDArray[Any],
    *,
    parameters: Mapping[str, object] | None = None,
    secondary_image: NDArray[Any] | None = None,
    roi_mask: NDArray[np.bool_] | None = None,
    request_id: str = "",
    generation: int = 0,
    tile_size: int = 1024,
    cancellation_check: Callable[[], None] | None = None,
) -> ImageOperationResult:
    """Execute a safe operation in bounded source-image tiles.

    Neighboring operations receive an expanded source patch and only their
    exact core is committed to the destination.  Consequently ROI pixels can
    read source samples outside the ROI, while pixels outside the ROI remain
    byte-for-byte unchanged.  Operations with global or geometric dependencies
    automatically use :func:`execute_image_operation`.
    """

    resolved = (
        operation
        if isinstance(operation, ImageOperation)
        else _coerce_enum(ImageOperation, operation, "图像操作")
    )
    source = _validate_raster(image)
    params = dict(parameters or {})
    resolved_tile_size = int(tile_size)
    if resolved_tile_size < 32:
        raise ValueError("处理图块边长必须至少为 32 像素。")
    if roi_mask is not None:
        mask = np.asarray(roi_mask, dtype=bool)
        if mask.shape != source.shape[:2]:
            raise ValueError(
                f"ROI 掩膜尺寸 {mask.shape!r} 与图像尺寸 "
                f"{source.shape[:2]!r} 不一致。"
            )
    else:
        mask = None
    secondary = None
    if secondary_image is not None:
        secondary = _validate_raster(secondary_image)
        if secondary.shape != source.shape:
            raise ValueError("第二幅图像的尺寸和通道必须与源图像完全一致。")
        if secondary.dtype != source.dtype:
            raise ValueError("第二幅图像的像素类型必须与源图像完全一致。")

    def check_cancelled() -> None:
        if cancellation_check is not None:
            cancellation_check()

    check_cancelled()
    height, width = source.shape[:2]
    estimate = estimate_tiled_execution(
        resolved,
        tuple(source.shape),
        parameters=params,
        roi_requested=mask is not None,
        tile_size=resolved_tile_size,
    )
    if not estimate.uses_tiled_execution:
        full_request = ImageOperationRequest.create(
            resolved,
            source,
            secondary_image=secondary,
            roi_mask=mask,
            request_id=request_id,
            generation=generation,
            **params,
        )
        result = execute_image_operation(
            full_request,
            cancellation_check=check_cancelled,
        )
        check_cancelled()
        return _with_tiled_execution_estimate(result, estimate)

    output: NDArray[Any] | None = None
    metadata: dict[str, ParameterValue] | None = None
    warnings: list[str] = []
    warning_set: set[str] = set()
    halo_x = estimate.halo_x
    halo_y = estimate.halo_y
    for core_y0 in range(0, height, resolved_tile_size):
        core_y1 = min(height, core_y0 + resolved_tile_size)
        patch_y0 = max(0, core_y0 - halo_y)
        patch_y1 = min(height, core_y1 + halo_y)
        for core_x0 in range(0, width, resolved_tile_size):
            check_cancelled()
            core_x1 = min(width, core_x0 + resolved_tile_size)
            patch_x0 = max(0, core_x0 - halo_x)
            patch_x1 = min(width, core_x1 + halo_x)
            patch = source[patch_y0:patch_y1, patch_x0:patch_x1]
            secondary_patch = (
                None
                if secondary is None
                else secondary[patch_y0:patch_y1, patch_x0:patch_x1]
            )
            mask_patch = (
                None
                if mask is None
                else mask[patch_y0:patch_y1, patch_x0:patch_x1]
            )
            tile_request = ImageOperationRequest.create(
                resolved,
                patch,
                secondary_image=secondary_patch,
                roi_mask=mask_patch,
                request_id=request_id,
                generation=generation,
                **params,
            )
            tile_result = execute_image_operation(tile_request)
            tile_image = np.asarray(tile_result.image)
            if tile_image.shape[:2] != patch.shape[:2]:
                raise RuntimeError(
                    f"操作 {resolved.value} 声明保持空间范围，"
                    "但图块输出尺寸发生变化。"
                )
            local_y0 = core_y0 - patch_y0
            local_y1 = local_y0 + (core_y1 - core_y0)
            local_x0 = core_x0 - patch_x0
            local_x1 = local_x0 + (core_x1 - core_x0)
            core = tile_image[local_y0:local_y1, local_x0:local_x1]
            if output is None:
                output_shape = (height, width) + tuple(tile_image.shape[2:])
                output = np.empty(output_shape, dtype=tile_image.dtype)
                metadata = dict(tile_result.metadata)
            elif (
                output.dtype != tile_image.dtype
                or output.shape[2:] != tile_image.shape[2:]
            ):
                raise RuntimeError(
                    f"操作 {resolved.value} 的不同图块返回了不一致的"
                    "位深或通道数。"
                )
            output[core_y0:core_y1, core_x0:core_x1] = core
            for warning in tile_result.warnings:
                if warning not in warning_set:
                    warning_set.add(warning)
                    warnings.append(warning)

    check_cancelled()
    if output is None:  # pragma: no cover - validated positive image dimensions
        raise RuntimeError("分块图像处理未产生输出。")
    if resolved is ImageOperation.REPAIR_NONFINITE and metadata is not None:
        metadata["repaired_count"] = int(np.count_nonzero(~np.isfinite(source)))
    if resolved is ImageOperation.CONVERT_TYPE and metadata is not None:
        target = _coerce_enum(
            PixelType,
            params.get("target_type", PixelType.UINT8.value),
            "目标位深",
        )
        metadata["nonfinite_replacement_count"] = (
            _integer_conversion_nonfinite_count(source, target)
        )
    result = ImageOperationResult(
        operation=resolved,
        image=np.ascontiguousarray(output),
        source_dtype=str(source.dtype),
        output_dtype=str(output.dtype),
        warnings=tuple(warnings),
        metadata=tuple(sorted((metadata or params).items())),
        request_id=str(request_id),
        generation=int(generation),
    )
    return _with_tiled_execution_estimate(result, estimate)


def _with_tiled_execution_estimate(
    result: ImageOperationResult,
    estimate: TiledExecutionEstimate,
) -> ImageOperationResult:
    metadata = dict(result.metadata)
    metadata.update(
        {
            "execution_mode": estimate.mode.value,
            "execution_tile_count": estimate.tile_count,
            "execution_halo_x": estimate.halo_x,
            "execution_halo_y": estimate.halo_y,
            "execution_overlap_multiplier": estimate.overlap_multiplier,
            "estimated_cpu_work_units": estimate.estimated_cpu_work_units,
            "estimated_tiled_cpu_work_units": (
                estimate.estimated_tiled_cpu_work_units
            ),
            "estimated_whole_cpu_work_units": (
                estimate.estimated_whole_cpu_work_units
            ),
        }
    )
    if estimate.reason:
        metadata["execution_decision_reason"] = estimate.reason
    return ImageOperationResult(
        operation=result.operation,
        image=result.image,
        source_dtype=result.source_dtype,
        output_dtype=result.output_dtype,
        warnings=result.warnings,
        metadata=tuple(sorted(metadata.items())),
        request_id=result.request_id,
        generation=result.generation,
        roi_mask=result.roi_mask,
    )


def convert_pixel_type(
    image: NDArray[Any],
    target: PixelType | str,
    *,
    mode: ConversionScaleMode | str = ConversionScaleMode.PRESERVE_VALUES,
    nonfinite_policy: NonfiniteIntegerPolicy | str = (
        NonfiniteIntegerPolicy.REJECT
    ),
    statistics_mask: NDArray[np.bool_] | None = None,
) -> NDArray[Any]:
    source = _validate_raster(image)
    target_type = (
        target
        if isinstance(target, PixelType)
        else _coerce_enum(PixelType, target, "目标位深")
    )
    scale_mode = (
        mode
        if isinstance(mode, ConversionScaleMode)
        else _coerce_enum(ConversionScaleMode, mode, "位深转换缩放模式")
    )
    resolved_nonfinite_policy = (
        nonfinite_policy
        if isinstance(nonfinite_policy, NonfiniteIntegerPolicy)
        else _coerce_enum(
            NonfiniteIntegerPolicy,
            nonfinite_policy,
            "非有限数替代规则",
        )
    )
    target_dtype = target_type.dtype
    if source.dtype == target_dtype:
        return source.copy()
    if source.ndim == 3 and target_type is not PixelType.UINT8:
        raise ValueError(
            "RGB/RGBA 图像只能保持为 8 位；"
            "请先显式转换为灰度，再转换为 16 位或 32 位浮点。"
        )

    work = source.astype(np.float64)
    active_statistics = _expanded_statistics_mask(work, statistics_mask)
    source_nonfinite = (
        ~np.isfinite(work)
        if target_dtype.kind in {"u", "i"} and source.dtype.kind == "f"
        else np.zeros(work.shape, dtype=bool)
    )
    replacement_count = int(np.count_nonzero(source_nonfinite))
    if (
        replacement_count
        and resolved_nonfinite_policy is NonfiniteIntegerPolicy.REJECT
    ):
        raise ValueError(
            "浮点图像包含 "
            f"{replacement_count} 个 NaN/Inf；"
            "转换为整数前必须明确选择“置零”或“按范围边界替代”。"
        )
    if scale_mode is ConversionScaleMode.PRESERVE_VALUES:
        mapped = work
    elif scale_mode is ConversionScaleMode.DATA_RANGE:
        finite = np.isfinite(work) & active_statistics
        if not np.any(finite):
            mapped = np.zeros_like(work)
        else:
            source_low = float(np.min(work[finite]))
            source_high = float(np.max(work[finite]))
            target_low, target_high = _dtype_range(target_dtype)
            if math.isclose(source_low, source_high):
                mapped = np.full_like(work, target_low)
            else:
                mapped = (
                    (work - source_low)
                    * ((target_high - target_low) / (source_high - source_low))
                    + target_low
                )
    else:
        source_low, source_high = _dtype_range(source.dtype)
        target_low, target_high = _dtype_range(target_dtype)
        if source.dtype.kind == "f":
            source_low, source_high = 0.0, 1.0
        mapped = (
            (work - source_low)
            * ((target_high - target_low) / (source_high - source_low))
            + target_low
        )
    if replacement_count:
        mapped = mapped.copy()
        if resolved_nonfinite_policy is NonfiniteIntegerPolicy.ZERO:
            mapped[source_nonfinite] = 0.0
        else:
            target_low, target_high = _dtype_range(target_dtype)
            positive_infinity = np.isposinf(work)
            mapped[source_nonfinite] = target_low
            mapped[positive_infinity] = target_high
    return _cast_like(mapped, target_dtype)


def _integer_conversion_nonfinite_count(
    image: NDArray[Any],
    target: PixelType,
) -> int:
    source = np.asarray(image)
    if source.dtype.kind != "f" or target.dtype.kind not in {"u", "i"}:
        return 0
    return int(np.count_nonzero(~np.isfinite(source)))


def convert_color_model(
    image: NDArray[Any],
    *,
    target: ColorTarget | str,
    grayscale_method: GrayscaleMethod | str = GrayscaleMethod.REC601,
    drop_alpha: bool = False,
) -> NDArray[Any]:
    """Convert between the supported scalar and RGB layouts.

    RGB-to-gray uses either the requested Rec. 601 formula
    ``0.299R + 0.587G + 0.114B`` or an unweighted channel average.  Conversion
    to RGB deliberately accepts only 8-bit scalar input: callers must first
    select an explicit 16-bit/float-to-8-bit mapping instead of receiving an
    implicit data-range normalization here.

    Removing an Alpha channel is destructive, so RGBA input is rejected unless
    ``drop_alpha`` is explicitly true.
    """

    source = _validate_raster(image)
    target_model = (
        target
        if isinstance(target, ColorTarget)
        else _coerce_enum(ColorTarget, target, "颜色模型")
    )
    method = (
        grayscale_method
        if isinstance(grayscale_method, GrayscaleMethod)
        else _coerce_enum(GrayscaleMethod, grayscale_method, "灰度换算方式")
    )

    if target_model is ColorTarget.GRAYSCALE:
        if source.ndim == 2:
            return source.copy()
        if source.shape[2] == 1:
            return source[..., 0].copy()
        if source.shape[2] == 4 and not drop_alpha:
            raise ValueError("RGBA 转灰度会移除 Alpha；必须显式启用 drop_alpha。")
        rgb = source[..., :3].astype(np.float64)
        if method is GrayscaleMethod.REC601:
            gray = (
                rgb[..., 0] * 0.299
                + rgb[..., 1] * 0.587
                + rgb[..., 2] * 0.114
            )
        else:
            gray = np.mean(rgb, axis=2)
        return _restore_dtype(gray, source.dtype)

    if source.ndim == 3 and source.shape[2] == 3:
        if source.dtype != np.uint8:
            raise ValueError("RGB 权威像素只支持 8 位；请先显式转换位深。")
        return source.copy()
    if source.ndim == 3 and source.shape[2] == 4:
        if not drop_alpha:
            raise ValueError("RGBA 转 RGB 会移除 Alpha；必须显式启用 drop_alpha。")
        if source.dtype != np.uint8:
            raise ValueError("RGB 权威像素只支持 8 位；请先显式转换位深。")
        return source[..., :3].copy()

    scalar = source if source.ndim == 2 else source[..., 0]
    if scalar.dtype != np.uint8:
        raise ValueError(
            "灰度转 RGB 只接受 8 位灰度；"
            "请先显式转换位深并明确选择数值映射规则。"
        )
    return np.repeat(scalar[..., np.newaxis], 3, axis=2)


def split_rgb_channels(
    image: NDArray[Any],
) -> tuple[NDArray[np.uint8], NDArray[np.uint8], NDArray[np.uint8]]:
    """Dedicated multi-output RGB split service.

    This intentionally is not an ``ImageOperation`` because the unified
    operation contract has exactly one raster output.
    """

    source = _validate_raster(image)
    if (
        source.dtype != np.dtype(np.uint8)
        or source.ndim != 3
        or source.shape[2] not in {3, 4}
    ):
        raise ValueError("RGB 通道拆分只支持 RGB8 或 RGBA8。")
    outputs = tuple(
        np.ascontiguousarray(source[..., index]).copy()
        for index in range(3)
    )
    return outputs  # type: ignore[return-value]


def merge_rgb_channels(
    red: NDArray[Any],
    green: NDArray[Any],
    blue: NDArray[Any],
    *,
    alpha: NDArray[Any] | None = None,
) -> NDArray[np.uint8]:
    """Dedicated multi-input RGB/RGBA merge service."""

    channels = [_validate_raster(item) for item in (red, green, blue)]
    if any(
        item.dtype != np.dtype(np.uint8) or item.ndim != 2
        for item in channels
    ):
        raise ValueError("RGB 通道合并只接受三个 GRAY8 平面。")
    if len({item.shape for item in channels}) != 1:
        raise ValueError("RGB 通道合并要求所有平面尺寸一致。")
    if alpha is not None:
        alpha_channel = _validate_raster(alpha)
        if (
            alpha_channel.dtype != np.dtype(np.uint8)
            or alpha_channel.ndim != 2
            or alpha_channel.shape != channels[0].shape
        ):
            raise ValueError("Alpha 通道必须是同尺寸 GRAY8 平面。")
        channels.append(alpha_channel)
    return np.ascontiguousarray(np.stack(channels, axis=2), dtype=np.uint8)


def adjust_color_balance(
    image: NDArray[Any],
    *,
    red_gain: float = 1.0,
    green_gain: float = 1.0,
    blue_gain: float = 1.0,
    red_offset: float = 0.0,
    green_offset: float = 0.0,
    blue_offset: float = 0.0,
) -> NDArray[Any]:
    """Apply explicit per-channel gain and offset without touching Alpha."""

    source = _validate_raster(image)
    if source.ndim != 3 or source.shape[2] not in {3, 4}:
        raise ValueError("色彩平衡只适用于 RGB 或 RGBA 图像。")
    if source.dtype != np.uint8:
        raise ValueError("当前权威 RGB/RGBA 像素只支持 8 位。")

    gains = (float(red_gain), float(green_gain), float(blue_gain))
    offsets = (float(red_offset), float(green_offset), float(blue_offset))
    for label, value in zip(("红色增益", "绿色增益", "蓝色增益"), gains):
        _require_finite(label, value)
        if value < 0.0:
            raise ValueError(f"{label}不能为负数。")
    for label, value in zip(("红色偏移", "绿色偏移", "蓝色偏移"), offsets):
        _require_finite(label, value)

    color = source[..., :3].astype(np.float64)
    balanced = color * np.asarray(gains) + np.asarray(offsets)
    restored = _restore_dtype(balanced, source.dtype)
    if source.shape[2] == 3:
        return restored
    return np.dstack((restored, source[..., 3].copy()))


def adjust_brightness_contrast(
    image: NDArray[Any],
    *,
    brightness: float = 0.0,
    contrast: float = 1.0,
    gamma: float = 1.0,
) -> NDArray[Any]:
    source = _validate_raster(image)
    _require_finite("亮度", brightness)
    _require_finite("对比度", contrast)
    _require_positive("Gamma", gamma)
    if contrast < 0:
        raise ValueError("对比度不能为负数。")
    if source.ndim == 3:
        return _apply_color_channels(
            source,
            lambda plane: adjust_brightness_contrast(
                plane,
                brightness=brightness,
                contrast=contrast,
                gamma=gamma,
            ),
        )
    low, high = _working_range(source)
    midpoint = (low + high) / 2.0
    work = source.astype(np.float64)
    work = (work - midpoint) * contrast + midpoint + brightness
    if not math.isclose(gamma, 1.0):
        normalized = np.clip((work - low) / max(high - low, np.finfo(float).eps), 0.0, 1.0)
        work = low + np.power(normalized, 1.0 / gamma) * (high - low)
    return _restore_dtype(work, source.dtype)


def adjust_levels(
    image: NDArray[Any],
    *,
    black_point: float,
    white_point: float,
    output_min: float | None = None,
    output_max: float | None = None,
    gamma: float = 1.0,
) -> NDArray[Any]:
    source = _validate_raster(image)
    _require_finite("黑场值", black_point)
    _require_finite("白场值", white_point)
    _require_positive("Gamma", gamma)
    if white_point <= black_point:
        raise ValueError("白场值必须大于黑场值。")
    default_low, default_high = _working_range(source)
    low = default_low if output_min is None else float(output_min)
    high = default_high if output_max is None else float(output_max)
    _require_finite("输出下限", low)
    _require_finite("输出上限", high)
    if high < low:
        raise ValueError("输出上限必须大于或等于输出下限。")
    if source.ndim == 3:
        return _apply_color_channels(
            source,
            lambda plane: adjust_levels(
                plane,
                black_point=black_point,
                white_point=white_point,
                output_min=low,
                output_max=high,
                gamma=gamma,
            ),
        )
    normalized = np.clip(
        (source.astype(np.float64) - black_point) / (white_point - black_point),
        0.0,
        1.0,
    )
    normalized = np.power(normalized, 1.0 / gamma)
    return _restore_dtype(low + normalized * (high - low), source.dtype)


def threshold_image(
    image: NDArray[Any],
    *,
    lower: float,
    upper: float,
    invert: bool = False,
    foreground_value: float | None = None,
    background_value: float | None = None,
    channel: str | None = None,
) -> NDArray[Any]:
    source = _validate_raster(image)
    _require_finite("阈值下限", lower)
    _require_finite("阈值上限", upper)
    if upper < lower:
        raise ValueError("阈值上限必须大于或等于阈值下限。")
    scalar = _require_scalar_image(source, channel)
    finite = np.isfinite(scalar)
    selected = finite & (scalar >= lower) & (scalar <= upper)
    if invert:
        selected = finite & ~selected
    low, high = _working_range(source)
    foreground = high if foreground_value is None else float(foreground_value)
    background = low if background_value is None else float(background_value)
    result = np.where(selected, foreground, background)
    return _restore_dtype(result, source.dtype)


def crop_image(
    image: NDArray[Any],
    *,
    x: int,
    y: int,
    width: int,
    height: int,
) -> NDArray[Any]:
    source = _validate_raster(image)
    if width <= 0 or height <= 0:
        raise ValueError("裁剪宽度和高度必须为正数。")
    if x < 0 or y < 0 or x + width > source.shape[1] or y + height > source.shape[0]:
        raise ValueError("裁剪矩形必须完全位于图像范围内。")
    return source[y : y + height, x : x + width].copy()


def resize_image(
    image: NDArray[Any],
    *,
    width: int,
    height: int,
    interpolation: InterpolationMode | str = InterpolationMode.AUTO,
) -> NDArray[Any]:
    source = _validate_raster(image)
    if width <= 0 or height <= 0:
        raise ValueError("调整后的宽度和高度必须为正数。")
    interpolation_mode = (
        interpolation
        if isinstance(interpolation, InterpolationMode)
        else _coerce_enum(InterpolationMode, interpolation, "插值模式")
    )
    interpolation_mode = resolve_resize_interpolation(
        source_width=int(source.shape[1]),
        source_height=int(source.shape[0]),
        width=int(width),
        height=int(height),
        requested=interpolation_mode,
    )
    resized = cv2.resize(
        source,
        (int(width), int(height)),
        interpolation=_cv_interpolation(interpolation_mode),
    )
    if source.ndim == 3 and source.shape[2] == 1 and resized.ndim == 2:
        return resized[..., np.newaxis]
    return resized


def resolve_resize_interpolation(
    *,
    source_width: int,
    source_height: int,
    width: int,
    height: int,
    requested: InterpolationMode | str,
    semantic: RasterSemantic | None = None,
) -> InterpolationMode:
    """Resolve ``auto`` to a concrete algorithm before execution.

    Binary masks and label images use nearest-neighbour interpolation so
    resizing cannot invent intermediate classes. Continuous rasters use Area
    only for a true downscale; enlargement and mixed-axis resizing use
    bilinear interpolation. The resolved value is persisted by the workbench.
    """

    mode = (
        requested
        if isinstance(requested, InterpolationMode)
        else _coerce_enum(InterpolationMode, requested, "插值模式")
    )
    if mode is not InterpolationMode.AUTO:
        return mode
    if semantic in {RasterSemantic.BINARY_MASK, RasterSemantic.LABELS}:
        return InterpolationMode.NEAREST
    if (
        width <= source_width
        and height <= source_height
        and (width < source_width or height < source_height)
    ):
        return InterpolationMode.AREA
    return InterpolationMode.LINEAR


def translate_image(
    image: NDArray[Any],
    *,
    offset_x: float,
    offset_y: float,
    interpolation: InterpolationMode | str = InterpolationMode.LINEAR,
    border_mode: BorderMode | str = BorderMode.CONSTANT,
    border_value: ParameterValue = 0.0,
) -> NDArray[Any]:
    """Translate pixels inside an unchanged canvas using explicit edge rules."""

    source = _validate_raster(image)
    _require_finite("水平平移量", offset_x)
    _require_finite("垂直平移量", offset_y)
    interpolation_mode = (
        interpolation
        if isinstance(interpolation, InterpolationMode)
        else _coerce_enum(InterpolationMode, interpolation, "插值模式")
    )
    border = (
        border_mode
        if isinstance(border_mode, BorderMode)
        else _coerce_enum(BorderMode, border_mode, "边界模式")
    )
    fill = _normalize_fill_value(source, border_value, field_name="边界填充值")
    matrix = np.asarray(
        [[1.0, 0.0, float(offset_x)], [0.0, 1.0, float(offset_y)]],
        dtype=np.float64,
    )
    translated = cv2.warpAffine(
        source,
        matrix,
        (int(source.shape[1]), int(source.shape[0])),
        flags=_cv_interpolation(interpolation_mode),
        borderMode=_cv_border(border),
        borderValue=fill,
    )
    if source.ndim == 3 and source.shape[2] == 1 and translated.ndim == 2:
        return translated[..., np.newaxis]
    return translated


def resize_canvas(
    image: NDArray[Any],
    *,
    width: int,
    height: int,
    anchor: CanvasAnchor | str = CanvasAnchor.CENTER,
    fill_value: ParameterValue = 0.0,
) -> NDArray[Any]:
    """Place the source on a new canvas, cropping only as requested by size."""

    source = _validate_raster(image)
    if width <= 0 or height <= 0:
        raise ValueError("画布宽度和高度必须为正数。")
    resolved_anchor = (
        anchor
        if isinstance(anchor, CanvasAnchor)
        else _coerce_enum(CanvasAnchor, anchor, "画布锚点")
    )
    fill = _normalize_fill_value(source, fill_value, field_name="画布填充值")
    output_shape = (
        (int(height), int(width))
        if source.ndim == 2
        else (int(height), int(width), int(source.shape[2]))
    )
    output = np.empty(output_shape, dtype=source.dtype)
    output[...] = fill

    horizontal, vertical = {
        CanvasAnchor.TOP_LEFT: (0.0, 0.0),
        CanvasAnchor.TOP_CENTER: (0.5, 0.0),
        CanvasAnchor.TOP_RIGHT: (1.0, 0.0),
        CanvasAnchor.CENTER_LEFT: (0.0, 0.5),
        CanvasAnchor.CENTER: (0.5, 0.5),
        CanvasAnchor.CENTER_RIGHT: (1.0, 0.5),
        CanvasAnchor.BOTTOM_LEFT: (0.0, 1.0),
        CanvasAnchor.BOTTOM_CENTER: (0.5, 1.0),
        CanvasAnchor.BOTTOM_RIGHT: (1.0, 1.0),
    }[resolved_anchor]
    origin_x = math.floor((int(width) - int(source.shape[1])) * horizontal)
    origin_y = math.floor((int(height) - int(source.shape[0])) * vertical)
    source_x = max(0, -origin_x)
    source_y = max(0, -origin_y)
    destination_x = max(0, origin_x)
    destination_y = max(0, origin_y)
    copy_width = min(
        int(source.shape[1]) - source_x,
        int(width) - destination_x,
    )
    copy_height = min(
        int(source.shape[0]) - source_y,
        int(height) - destination_y,
    )
    output[
        destination_y : destination_y + copy_height,
        destination_x : destination_x + copy_width,
        ...,
    ] = source[
        source_y : source_y + copy_height,
        source_x : source_x + copy_width,
        ...,
    ]
    return output


def pixel_bin(
    image: NDArray[Any],
    *,
    factor: int,
    method: PixelBinMethod | str = PixelBinMethod.MEAN,
    remainder_policy: PixelBinRemainderPolicy | str = PixelBinRemainderPolicy.REJECT,
) -> tuple[NDArray[Any], int, int]:
    """Combine non-overlapping ``factor × factor`` pixel blocks.

    Dimensions that are not divisible by ``factor`` are rejected unless the
    caller explicitly selects ``crop``.  Cropping then removes only the right
    and bottom remainder.  Scalar ``sum`` returns float32 to avoid silent
    integer saturation; RGB/RGBA sum is rejected because the authoritative
    pixel model has no floating-point color layout.
    """

    source = _validate_raster(image)
    if isinstance(factor, bool) or int(factor) != factor or int(factor) <= 0:
        raise ValueError("像素合并系数必须为正整数。")
    resolved_factor = int(factor)
    resolved_method = (
        method
        if isinstance(method, PixelBinMethod)
        else _coerce_enum(PixelBinMethod, method, "像素合并方式")
    )
    policy = (
        remainder_policy
        if isinstance(remainder_policy, PixelBinRemainderPolicy)
        else _coerce_enum(
            PixelBinRemainderPolicy,
            remainder_policy,
            "余数处理方式",
        )
    )
    height, width = (int(source.shape[0]), int(source.shape[1]))
    cropped_right = width % resolved_factor
    cropped_bottom = height % resolved_factor
    if cropped_right or cropped_bottom:
        if policy is PixelBinRemainderPolicy.REJECT:
            raise ValueError(
                "图像宽高不能被像素合并系数整除；"
                "如需裁去右侧和底部余数，必须显式选择 remainder_policy='crop'。"
            )
    usable_width = width - cropped_right
    usable_height = height - cropped_bottom
    if usable_width <= 0 or usable_height <= 0:
        raise ValueError("像素合并系数不能大于裁切后的图像尺寸。")
    cropped = source[:usable_height, :usable_width, ...]
    output_height = usable_height // resolved_factor
    output_width = usable_width // resolved_factor
    if source.ndim == 2:
        blocks = cropped.reshape(
            output_height,
            resolved_factor,
            output_width,
            resolved_factor,
        )
    else:
        blocks = cropped.reshape(
            output_height,
            resolved_factor,
            output_width,
            resolved_factor,
            int(source.shape[2]),
        )
    axes = (1, 3)
    if resolved_method is PixelBinMethod.MEAN:
        reduced = np.mean(blocks, axis=axes, dtype=np.float64)
        result = _restore_dtype(reduced, source.dtype)
    elif resolved_method is PixelBinMethod.MINIMUM:
        result = np.min(blocks, axis=axes)
    elif resolved_method is PixelBinMethod.MAXIMUM:
        result = np.max(blocks, axis=axes)
    else:
        if source.ndim == 3 and source.shape[2] > 1:
            raise ValueError("RGB/RGBA 像素合并不支持求和，请选择均值、最小值或最大值。")
        reduced = np.sum(blocks, axis=axes, dtype=np.float64)
        result = reduced.astype(np.float32)
    return np.ascontiguousarray(result), cropped_right, cropped_bottom


def rotate_image(
    image: NDArray[Any],
    *,
    angle_degrees: float,
    expand: bool = True,
    interpolation: InterpolationMode | str = InterpolationMode.LINEAR,
    border_mode: BorderMode | str = BorderMode.CONSTANT,
    border_value: float = 0.0,
) -> NDArray[Any]:
    source = _validate_raster(image)
    _require_finite("旋转角度", angle_degrees)
    interpolation_mode = (
        interpolation
        if isinstance(interpolation, InterpolationMode)
        else _coerce_enum(InterpolationMode, interpolation, "插值模式")
    )
    border = (
        border_mode
        if isinstance(border_mode, BorderMode)
        else _coerce_enum(BorderMode, border_mode, "边界模式")
    )
    height, width = source.shape[:2]
    center = ((width - 1) / 2.0, (height - 1) / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)
    output_width = width
    output_height = height
    if expand:
        cosine = abs(float(matrix[0, 0]))
        sine = abs(float(matrix[0, 1]))
        output_width = max(1, int(math.ceil(height * sine + width * cosine)))
        output_height = max(1, int(math.ceil(height * cosine + width * sine)))
        matrix[0, 2] += (output_width - width) / 2.0
        matrix[1, 2] += (output_height - height) / 2.0
    rotated = cv2.warpAffine(
        source,
        matrix,
        (output_width, output_height),
        flags=_cv_interpolation(interpolation_mode),
        borderMode=_cv_border(border),
        borderValue=float(border_value),
    )
    if source.ndim == 3 and source.shape[2] == 1 and rotated.ndim == 2:
        return rotated[..., np.newaxis]
    return rotated


def gaussian_blur(
    image: NDArray[Any],
    *,
    sigma_x: float,
    sigma_y: float | None = None,
    border_mode: BorderMode | str = BorderMode.REFLECT,
) -> NDArray[Any]:
    source = _validate_raster(image)
    _require_positive("横向标准差", sigma_x)
    resolved_sigma_y = sigma_x if sigma_y is None else float(sigma_y)
    _require_positive("纵向标准差", resolved_sigma_y)
    border = (
        border_mode
        if isinstance(border_mode, BorderMode)
        else _coerce_enum(BorderMode, border_mode, "边界模式")
    )
    return _apply_color_channels(
        source,
        lambda plane: cv2.GaussianBlur(
            plane,
            (0, 0),
            sigmaX=float(sigma_x),
            sigmaY=resolved_sigma_y,
            borderType=_cv_border(border),
        ),
    )


def median_filter(image: NDArray[Any], *, radius: int = 1) -> NDArray[Any]:
    source = _validate_raster(image)
    if radius < 1:
        raise ValueError("中值滤波半径必须至少为 1。")
    kernel_size = radius * 2 + 1
    if source.dtype != np.uint8 and kernel_size > 5:
        raise ValueError("只有 8 位图像支持大于 2 的中值滤波半径。")
    return _apply_color_channels(
        source,
        lambda plane: _median_blur_reflect(plane, radius),
    )


def mean_filter(
    image: NDArray[Any],
    *,
    radius: int = 1,
    border_mode: BorderMode | str = BorderMode.REFLECT,
) -> NDArray[Any]:
    source = _validate_raster(image)
    if radius < 1:
        raise ValueError("均值滤波半径必须至少为 1。")
    border = (
        border_mode
        if isinstance(border_mode, BorderMode)
        else _coerce_enum(BorderMode, border_mode, "边界模式")
    )
    if border is BorderMode.WRAP:
        raise ValueError(
            "均值滤波不支持循环边界；"
            "请选择 Reflect101、复制边缘或常量边界。"
        )
    kernel_size = radius * 2 + 1
    return _apply_color_channels(
        source,
        lambda plane: cv2.blur(
            plane,
            (kernel_size, kernel_size),
            borderType=_cv_border(border),
        ),
    )


def bilateral_filter(
    image: NDArray[Any],
    *,
    diameter: int = 5,
    sigma_color: float = 25.0,
    sigma_space: float = 2.0,
    border_mode: BorderMode | str = BorderMode.REFLECT,
) -> NDArray[Any]:
    """Preserve edges while smoothing each color channel independently."""

    source = _validate_raster(image)
    if diameter < 1 or diameter % 2 == 0:
        raise ValueError("双边滤波直径必须是正奇数。")
    _require_positive("颜色域标准差", sigma_color)
    _require_positive("空间域标准差", sigma_space)
    border = (
        border_mode
        if isinstance(border_mode, BorderMode)
        else _coerce_enum(BorderMode, border_mode, "边界模式")
    )

    def filter_plane(plane: NDArray[Any]) -> NDArray[Any]:
        work = plane if plane.dtype in {np.dtype(np.uint8), np.dtype(np.float32)} else plane.astype(np.float32)
        filtered = cv2.bilateralFilter(
            work,
            diameter,
            float(sigma_color),
            float(sigma_space),
            borderType=_cv_border(border),
        )
        return _restore_dtype(filtered, plane.dtype)

    return _apply_color_channels(source, filter_plane)


def normalize_image(
    image: NDArray[Any],
    *,
    output_min: float,
    output_max: float,
    per_channel: bool = True,
    statistics_mask: NDArray[np.bool_] | None = None,
) -> NDArray[Any]:
    """Map the finite data range to an explicit output range."""

    source = _validate_raster(image)
    _require_finite("输出下限", output_min)
    _require_finite("输出上限", output_max)
    if output_max < output_min:
        raise ValueError("归一化输出上限不能小于输出下限。")

    def normalize_plane(plane: NDArray[Any]) -> NDArray[Any]:
        work = plane.astype(np.float64)
        writable = np.isfinite(work)
        statistics = writable & _expanded_statistics_mask(
            work,
            statistics_mask,
        )
        if not np.any(statistics):
            return plane.copy()
        low = float(np.min(work[statistics]))
        high = float(np.max(work[statistics]))
        result = work.copy()
        if math.isclose(low, high):
            result[writable] = output_min
        else:
            result[writable] = (
                (work[writable] - low)
                * ((output_max - output_min) / (high - low))
                + output_min
            )
        return _restore_dtype(result, plane.dtype)

    if per_channel or source.ndim != 3 or source.shape[2] == 1:
        return _apply_color_channels(source, normalize_plane)
    alpha = source[..., 3].copy() if source.shape[2] == 4 else None
    color = source[..., :3]
    normalized = normalize_plane(color)
    if alpha is None:
        return normalized
    return np.dstack((normalized, alpha))


def equalize_histogram(
    image: NDArray[Any],
    *,
    statistics_mask: NDArray[np.bool_] | None = None,
) -> NDArray[Any]:
    """Equalize the finite histogram without changing dtype or Alpha."""

    source = _validate_raster(image)

    def equalize_plane(plane: NDArray[Any]) -> NDArray[Any]:
        work = np.asarray(plane)
        finite = np.isfinite(work)
        statistics = finite & _expanded_statistics_mask(
            work,
            statistics_mask,
        )
        if not np.any(statistics):
            return work.copy()
        low, high = _working_range(work)
        if work.dtype == np.uint8:
            histogram = np.bincount(
                work[statistics].ravel(),
                minlength=256,
            )
            cdf = histogram.cumsum()
            occupied = cdf[histogram > 0]
            if occupied.size <= 1:
                return work.copy()
            denominator = int(cdf[-1]) - int(occupied[0])
            if denominator <= 0:
                return work.copy()
            lut = np.rint(
                np.clip(
                    (cdf - int(occupied[0])) / denominator,
                    0.0,
                    1.0,
                )
                * 255.0
            ).astype(np.uint8)
            return lut[work]
        if work.dtype == np.uint16:
            histogram = np.bincount(
                work[statistics].ravel(),
                minlength=65536,
            )
            cdf = histogram.cumsum()
            occupied = cdf[histogram > 0]
            if occupied.size <= 1:
                return work.copy()
            cdf_min = int(occupied[0])
            denominator = int(cdf[-1]) - cdf_min
            if denominator <= 0:
                return work.copy()
            lut = np.rint(
                np.clip((cdf - cdf_min) / denominator, 0.0, 1.0) * 65535.0
            ).astype(np.uint16)
            return lut[work]
        values = work[statistics].astype(np.float64)
        value_min = float(np.min(values))
        value_max = float(np.max(values))
        if math.isclose(value_min, value_max):
            return work.copy()
        histogram, edges = np.histogram(values, bins=4096, range=(value_min, value_max))
        cdf = histogram.cumsum().astype(np.float64)
        nonzero = cdf[histogram > 0]
        if nonzero.size <= 1:
            return work.copy()
        cdf = np.clip((cdf - nonzero[0]) / max(cdf[-1] - nonzero[0], 1.0), 0.0, 1.0)
        indices = np.clip(np.searchsorted(edges, work, side="right") - 1, 0, len(cdf) - 1)
        result = work.copy()
        result[finite] = low + cdf[indices[finite]] * (high - low)
        return result.astype(np.float32)

    return _apply_color_channels(source, equalize_plane)


def clahe(
    image: NDArray[Any],
    *,
    clip_limit: float = 2.0,
    tile_grid_size: int = 8,
) -> NDArray[Any]:
    source = _validate_raster(image)
    _require_positive("CLAHE 对比度限制", clip_limit)
    if tile_grid_size < 2 or tile_grid_size > 64:
        raise ValueError("CLAHE 网格大小必须在 2 到 64 之间。")
    if source.dtype not in {np.dtype(np.uint8), np.dtype(np.uint16)}:
        raise TypeError("CLAHE 仅支持 8 位或 16 位整数图像。")
    operator = cv2.createCLAHE(
        clipLimit=float(clip_limit),
        tileGridSize=(tile_grid_size, tile_grid_size),
    )
    return _apply_color_channels(source, operator.apply)


def remove_outliers(
    image: NDArray[Any],
    *,
    radius: int = 1,
    threshold: float = 25.0,
    polarity: str = "both",
) -> NDArray[Any]:
    """Replace isolated hot/dark pixels with the local median."""

    source = _validate_raster(image)
    if radius < 1:
        raise ValueError("热点/坏点剔除半径必须至少为 1。")
    if source.dtype != np.uint8 and radius > 2:
        raise ValueError("16 位或浮点图像的热点/坏点剔除半径最大为 2。")
    _require_finite("热点/坏点阈值", threshold)
    if threshold < 0:
        raise ValueError("热点/坏点阈值不能为负数。")
    resolved_polarity = str(polarity).strip().lower()
    if resolved_polarity not in {"both", "bright", "dark"}:
        raise ValueError("热点/坏点极性必须为 both、bright 或 dark。")

    def process_plane(plane: NDArray[Any]) -> NDArray[Any]:
        median = _median_blur_reflect(plane, radius)
        delta = plane.astype(np.float64) - median.astype(np.float64)
        if resolved_polarity == "bright":
            replace = delta > threshold
        elif resolved_polarity == "dark":
            replace = delta < -threshold
        else:
            replace = np.abs(delta) > threshold
        return np.where(replace, median, plane).astype(plane.dtype)

    return _apply_color_channels(source, process_plane)


def repair_nonfinite(
    image: NDArray[Any],
    *,
    radius: int = 1,
    fallback_value: float = 0.0,
) -> tuple[NDArray[Any], int]:
    """Repair NaN/Inf samples from finite neighboring samples."""

    source = _validate_raster(image)
    if source.dtype != np.float32:
        raise TypeError("NaN/Inf 修复仅适用于 32 位浮点图像。")
    if radius < 1 or radius > 32:
        raise ValueError("NaN/Inf 修复半径必须在 1 到 32 之间。")
    _require_finite("NaN/Inf 修复回退值", fallback_value)
    kernel_size = radius * 2 + 1
    repaired_count = 0

    def process_plane(plane: NDArray[Any]) -> NDArray[Any]:
        nonlocal repaired_count
        finite = np.isfinite(plane)
        repaired_count += int(np.count_nonzero(~finite))
        if np.all(finite):
            return plane.copy()
        values = np.where(finite, plane, 0.0).astype(np.float32)
        weights = finite.astype(np.float32)
        sums = cv2.boxFilter(
            values,
            cv2.CV_32F,
            (kernel_size, kernel_size),
            normalize=False,
            borderType=cv2.BORDER_REFLECT_101,
        )
        counts = cv2.boxFilter(
            weights,
            cv2.CV_32F,
            (kernel_size, kernel_size),
            normalize=False,
            borderType=cv2.BORDER_REFLECT_101,
        )
        local = np.full_like(values, float(fallback_value))
        np.divide(sums, counts, out=local, where=counts > 0)
        return np.where(finite, plane, local).astype(np.float32)

    repaired = _apply_color_channels(source, process_plane)
    return repaired, repaired_count


def laplacian_edges(
    image: NDArray[Any],
    *,
    kernel_size: int = 3,
    channel: str | None = None,
    output_float: bool = True,
) -> NDArray[Any]:
    source = _validate_raster(image)
    if kernel_size not in {1, 3, 5, 7}:
        raise ValueError("Laplacian 核大小必须是 1、3、5 或 7。")
    scalar = _require_scalar_image(source, channel).astype(np.float32)
    result = cv2.Laplacian(
        scalar,
        cv2.CV_32F,
        ksize=kernel_size,
        borderType=cv2.BORDER_REFLECT_101,
    )
    if output_float:
        return result.astype(np.float32, copy=False)
    return _restore_dtype(np.abs(result), source.dtype)


def canny_edges(
    image: NDArray[Any],
    *,
    threshold_low: float,
    threshold_high: float,
    aperture_size: int = 3,
    l2_gradient: bool = True,
    channel: str | None = None,
) -> NDArray[np.uint8]:
    source = _validate_raster(image)
    if source.dtype != np.uint8:
        raise TypeError("Canny 边缘检测仅支持 8 位图像；请先显式转换位深。")
    _require_finite("Canny 低阈值", threshold_low)
    _require_finite("Canny 高阈值", threshold_high)
    if threshold_low < 0 or threshold_high <= threshold_low:
        raise ValueError("Canny 阈值必须满足 0 ≤ 低阈值 < 高阈值。")
    if aperture_size not in {3, 5, 7}:
        raise ValueError("Canny 孔径大小必须是 3、5 或 7。")
    scalar = _canny_scalar_uint8(source, channel)
    radius = aperture_size // 2
    padded = cv2.copyMakeBorder(
        scalar,
        radius,
        radius,
        radius,
        radius,
        cv2.BORDER_REFLECT_101,
    )
    result = cv2.Canny(
        padded,
        threshold1=threshold_low,
        threshold2=threshold_high,
        apertureSize=aperture_size,
        L2gradient=l2_gradient,
    )
    return result[radius:-radius, radius:-radius]


def canny_gradient_magnitude(
    image: NDArray[Any],
    *,
    aperture_size: int = 3,
    l2_gradient: bool = True,
    channel: str | None = None,
) -> NDArray[np.float32]:
    """Return the gradient domain to which Canny thresholds are applied.

    This helper exists for the parameter editor and tests.  Showing an input
    intensity histogram next to Canny thresholds is scientifically misleading:
    the thresholds operate on Sobel gradient magnitude.  The 7×7 scale mirrors
    OpenCV Canny's threshold normalization.
    """

    source = _validate_raster(image)
    if source.dtype != np.uint8:
        raise TypeError("Canny 梯度仅支持 8 位图像；请先显式转换位深。")
    if aperture_size not in {3, 5, 7}:
        raise ValueError("Canny 孔径大小必须是 3、5 或 7。")
    scalar = _canny_scalar_uint8(source, channel)
    radius = aperture_size // 2
    padded = cv2.copyMakeBorder(
        scalar,
        radius,
        radius,
        radius,
        radius,
        cv2.BORDER_REFLECT_101,
    )
    # OpenCV's Canny computes CV_16S Sobel derivatives and, for a 7×7
    # aperture, scales both the derivatives and the user thresholds by 1/16.
    # The editor displays values in the *user threshold* domain, so recreate
    # the same saturated integer derivatives and multiply the resulting
    # magnitude back by 16 for aperture 7.
    derivative_scale = 1.0 / 16.0 if aperture_size == 7 else 1.0
    dx = cv2.Sobel(
        padded,
        cv2.CV_16S,
        1,
        0,
        ksize=aperture_size,
        scale=derivative_scale,
        borderType=cv2.BORDER_CONSTANT,
    )
    dy = cv2.Sobel(
        padded,
        cv2.CV_16S,
        0,
        1,
        ksize=aperture_size,
        scale=derivative_scale,
        borderType=cv2.BORDER_CONSTANT,
    )
    dx_float = dx.astype(np.float32)
    dy_float = dy.astype(np.float32)
    magnitude = (
        cv2.magnitude(dx_float, dy_float)
        if l2_gradient
        else np.abs(dx_float) + np.abs(dy_float)
    )
    if aperture_size == 7:
        magnitude *= 16.0
    return np.ascontiguousarray(
        magnitude[radius:-radius, radius:-radius],
        dtype=np.float32,
    )


def _canny_scalar_uint8(
    source: NDArray[Any],
    channel: str | None,
) -> NDArray[np.uint8]:
    scalar = _require_scalar_image(source, channel)
    if scalar.dtype != np.dtype(np.uint8):
        # Selecting weighted luminance from an 8-bit RGB/RGBA source produces
        # a float32 scalar plane.  Canny itself requires CV_8U, so materialise
        # the explicitly selected scalar channel without changing the working
        # red/green/blue paths.
        scalar = np.clip(
            np.rint(scalar),
            0,
            255,
        ).astype(np.uint8)
    return np.ascontiguousarray(scalar, dtype=np.uint8)


def auto_threshold(
    image: NDArray[Any],
    *,
    method: str = "otsu",
    channel: str | None = None,
    invert: bool = False,
    statistics_mask: NDArray[np.bool_] | None = None,
) -> tuple[NDArray[Any], float]:
    source = _validate_raster(image)
    scalar = _require_scalar_image(source, channel)
    active = _expanded_statistics_mask(scalar, statistics_mask)
    finite = scalar[np.isfinite(scalar) & active]
    if finite.size == 0:
        raise ValueError("自动阈值无法处理不含有限像素的图像。")
    resolved_method = str(method).strip().lower()
    if resolved_method not in {"otsu", "isodata", "triangle"}:
        raise ValueError("自动阈值方法必须为 Otsu、IsoData 或 Triangle。")
    histogram, centers = _threshold_histogram(finite)
    if resolved_method == "otsu":
        threshold = _otsu_threshold(histogram, centers)
    elif resolved_method == "isodata":
        threshold = _isodata_threshold(histogram, centers)
    else:
        threshold = _triangle_threshold(histogram, centers)
    return binarize_image(scalar, threshold=threshold, invert=invert), threshold


def adaptive_threshold(
    image: NDArray[Any],
    *,
    method: str = "gaussian",
    radius: int = 7,
    offset: float = 0.0,
    k: float = 0.2,
    r: float = 128.0,
    p: float = 2.0,
    q: float = 10.0,
    foreground_is_high: bool = True,
) -> NDArray[Any]:
    """Local Mean/Gaussian/Sauvola/Phansalkar thresholding."""

    source = _validate_raster(image)
    if source.ndim != 2:
        raise ValueError("局部阈值需要单通道图像。")
    if radius < 1 or radius > 255:
        raise ValueError("局部阈值半径必须在 1 到 255 之间。")
    for label, value in (
        ("局部阈值偏移", offset),
        ("局部阈值 k", k),
        ("局部阈值 R", r),
        ("Phansalkar p", p),
        ("Phansalkar q", q),
    ):
        _require_finite(label, value)
    resolved = str(method).strip().lower()
    if resolved not in {"mean", "gaussian", "sauvola", "phansalkar"}:
        raise ValueError(
            "局部阈值方法必须为 mean、gaussian、sauvola 或 phansalkar。"
        )
    if resolved in {"sauvola", "phansalkar"} and r <= 0:
        raise ValueError("Sauvola/Phansalkar 的 R 必须为正数。")
    work = source.astype(np.float32)
    size = radius * 2 + 1
    if resolved == "gaussian":
        mean = cv2.GaussianBlur(
            work,
            (size, size),
            sigmaX=max(radius / 3.0, 0.1),
            borderType=cv2.BORDER_REFLECT_101,
        )
    else:
        mean = cv2.boxFilter(
            work,
            cv2.CV_32F,
            (size, size),
            normalize=True,
            borderType=cv2.BORDER_REFLECT_101,
        )
    if resolved in {"sauvola", "phansalkar"}:
        mean_square = cv2.boxFilter(
            work * work,
            cv2.CV_32F,
            (size, size),
            normalize=True,
            borderType=cv2.BORDER_REFLECT_101,
        )
        standard_deviation = np.sqrt(
            np.maximum(mean_square - mean * mean, 0.0)
        )
        threshold = mean * (1.0 + k * (standard_deviation / r - 1.0))
        if resolved == "phansalkar":
            _low, native_high = _working_range(source)
            normalized_mean = mean / max(
                native_high,
                np.finfo(np.float32).eps,
            )
            threshold = mean * (
                1.0
                + p * np.exp(-q * normalized_mean)
                + k * (standard_deviation / r - 1.0)
            )
    else:
        threshold = mean
    threshold = threshold - float(offset)
    mask = work >= threshold if foreground_is_high else work <= threshold
    return _mask_to_binary_values(
        mask,
        source.dtype,
        foreground_is_high=foreground_is_high,
    )


def percentile_saturation_enhance(
    image: NDArray[Any],
    *,
    lower_percentile: float = 0.5,
    upper_percentile: float = 99.5,
    per_channel: bool = True,
    statistics_mask: NDArray[np.bool_] | None = None,
) -> NDArray[Any]:
    """Clip percentile tails and remap them to the native numeric range."""

    source = _validate_raster(image)
    if not 0.0 <= lower_percentile < upper_percentile <= 100.0:
        raise ValueError("百分位范围必须满足 0 ≤ lower < upper ≤ 100。")

    def enhance_plane(plane: NDArray[Any]) -> NDArray[Any]:
        work = plane.astype(np.float64)
        finite = np.isfinite(work)
        statistics = finite & _expanded_statistics_mask(
            work,
            statistics_mask,
        )
        if not np.any(statistics):
            return plane.copy()
        low, high = np.percentile(
            work[statistics],
            [lower_percentile, upper_percentile],
        )
        if math.isclose(float(low), float(high)):
            return plane.copy()
        output_low, output_high = _working_range(plane)
        mapped = (
            (np.clip(work, low, high) - low)
            * ((output_high - output_low) / (high - low))
            + output_low
        )
        return _restore_dtype(mapped, plane.dtype)

    if per_channel or source.ndim == 2:
        return _apply_color_channels(source, enhance_plane)
    alpha = source[..., 3].copy() if source.shape[2] == 4 else None
    enhanced = enhance_plane(source[..., :3])
    if alpha is None:
        return enhanced
    return np.dstack((enhanced, alpha))


def rank_filter(
    image: NDArray[Any],
    *,
    method: str = "minimum",
    radius: int = 1,
) -> NDArray[Any]:
    """Minimum, maximum, or local population-variance rank filter."""

    source = _validate_raster(image)
    if radius < 1 or radius > 255:
        raise ValueError("Rank 滤波半径必须在 1 到 255 之间。")
    resolved = str(method).strip().lower()
    if resolved not in {"minimum", "maximum", "variance"}:
        raise ValueError("Rank 方法必须为 minimum、maximum 或 variance。")
    size = radius * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))

    def filter_plane(plane: NDArray[Any]) -> NDArray[Any]:
        if resolved == "minimum":
            return cv2.erode(
                plane,
                kernel,
                borderType=cv2.BORDER_REFLECT_101,
            )
        if resolved == "maximum":
            return cv2.dilate(
                plane,
                kernel,
                borderType=cv2.BORDER_REFLECT_101,
            )
        work = plane.astype(np.float32)
        mean = cv2.blur(
            work,
            (size, size),
            borderType=cv2.BORDER_REFLECT_101,
        )
        mean_square = cv2.blur(
            work * work,
            (size, size),
            borderType=cv2.BORDER_REFLECT_101,
        )
        return np.maximum(mean_square - mean * mean, 0.0).astype(np.float32)

    if resolved == "variance" and source.ndim == 3 and source.shape[2] > 1:
        raise ValueError("Variance Rank 的 float32 结果仅支持单通道权威栅格。")
    return _apply_color_channels(source, filter_plane)


def binarize_image(
    image: NDArray[Any],
    *,
    threshold: float,
    channel: str | None = None,
    invert: bool = False,
) -> NDArray[Any]:
    source = _validate_raster(image)
    _require_finite("二值化阈值", threshold)
    scalar = _require_scalar_image(source, channel)
    finite = np.isfinite(scalar)
    foreground = finite & (scalar > threshold)
    if invert:
        foreground = finite & ~foreground
    low, high = _binary_values(scalar.dtype)
    return np.where(foreground, high, low).astype(scalar.dtype)


def unsharp_mask(
    image: NDArray[Any],
    *,
    sigma: float = 1.0,
    amount: float = 1.0,
    threshold: float = 0.0,
) -> NDArray[Any]:
    source = _validate_raster(image)
    _require_positive("反锐化标准差", sigma)
    _require_finite("反锐化强度", amount)
    _require_finite("反锐化阈值", threshold)
    if amount < 0 or threshold < 0:
        raise ValueError("反锐化强度和阈值不能为负数。")
    blurred = gaussian_blur(source, sigma_x=sigma)
    original_float = source.astype(np.float64)
    difference = original_float - blurred.astype(np.float64)
    if threshold > 0:
        difference = np.where(np.abs(difference) >= threshold, difference, 0.0)
    return _restore_dtype(original_float + amount * difference, source.dtype)


def sobel_edges(
    image: NDArray[Any],
    *,
    kernel_size: int = 3,
    channel: str = "luminance",
    output_float: bool = True,
) -> NDArray[Any]:
    source = _validate_raster(image)
    if kernel_size not in {1, 3, 5, 7}:
        raise ValueError("Sobel 核大小必须是 1、3、5 或 7。")
    scalar = _select_scalar_channel(source, channel).astype(np.float32)
    dx = cv2.Sobel(scalar, cv2.CV_32F, 1, 0, ksize=kernel_size, borderType=cv2.BORDER_REFLECT_101)
    dy = cv2.Sobel(scalar, cv2.CV_32F, 0, 1, ksize=kernel_size, borderType=cv2.BORDER_REFLECT_101)
    magnitude = cv2.magnitude(dx, dy)
    if output_float:
        return magnitude.astype(np.float32, copy=False)
    return _restore_dtype(magnitude, source.dtype)


def morphology_filter(
    image: NDArray[Any],
    *,
    operation: ImageOperation | str,
    radius: int = 1,
    iterations: int = 1,
    kernel: MorphologyKernel | str = MorphologyKernel.ELLIPSE,
    border_mode: BorderMode | str = BorderMode.REFLECT,
    channel: str | None = None,
) -> NDArray[Any]:
    source = _validate_raster(image)
    scalar = _require_scalar_image(source, channel)
    resolved_operation = (
        operation
        if isinstance(operation, ImageOperation)
        else _coerce_enum(ImageOperation, operation, "形态学操作")
    )
    operation_map = {
        ImageOperation.ERODE: cv2.MORPH_ERODE,
        ImageOperation.DILATE: cv2.MORPH_DILATE,
        ImageOperation.MORPHOLOGY_OPEN: cv2.MORPH_OPEN,
        ImageOperation.MORPHOLOGY_CLOSE: cv2.MORPH_CLOSE,
        ImageOperation.TOP_HAT: cv2.MORPH_TOPHAT,
        ImageOperation.BLACK_HAT: cv2.MORPH_BLACKHAT,
    }
    if resolved_operation not in operation_map:
        raise ValueError(f"{resolved_operation.value} 不是形态学操作。")
    if radius < 1 or iterations < 1:
        raise ValueError("形态学半径和迭代次数必须至少为 1。")
    kernel_shape = (
        kernel
        if isinstance(kernel, MorphologyKernel)
        else _coerce_enum(MorphologyKernel, kernel, "形态学核")
    )
    border = (
        border_mode
        if isinstance(border_mode, BorderMode)
        else _coerce_enum(BorderMode, border_mode, "边界模式")
    )
    if border is BorderMode.WRAP:
        raise ValueError(
            "形态学处理不支持循环边界；"
            "请选择 Reflect101、复制边缘或常量边界。"
        )
    shape_map = {
        MorphologyKernel.ELLIPSE: cv2.MORPH_ELLIPSE,
        MorphologyKernel.RECTANGLE: cv2.MORPH_RECT,
        MorphologyKernel.CROSS: cv2.MORPH_CROSS,
    }
    structuring_element = cv2.getStructuringElement(
        shape_map[kernel_shape],
        (radius * 2 + 1, radius * 2 + 1),
    )
    return cv2.morphologyEx(
        scalar,
        operation_map[resolved_operation],
        structuring_element,
        iterations=iterations,
        borderType=_cv_border(border),
    )


def morphology_derivative(
    image: NDArray[Any],
    *,
    method: str = "gradient",
    radius: int = 1,
    channel: str | None = None,
) -> NDArray[Any]:
    """Morphological gradient or signed morphological Laplacian."""

    source = _validate_raster(image)
    if radius < 1 or radius > 255:
        raise ValueError("形态学微分半径必须在 1 到 255 之间。")
    resolved = str(method).strip().lower()
    if resolved not in {"gradient", "laplacian"}:
        raise ValueError("形态学微分方法必须为 gradient 或 laplacian。")
    if channel is not None:
        scalar = _require_scalar_image(source, channel)
        planes = [scalar]
        rebuild_color = False
    else:
        planes, alpha = _split_color_channels(source)
        rebuild_color = source.ndim == 3 and source.shape[2] > 1
    size = radius * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    results: list[NDArray[Any]] = []
    for plane in planes:
        minimum = cv2.erode(
            plane,
            kernel,
            borderType=cv2.BORDER_REFLECT_101,
        )
        maximum = cv2.dilate(
            plane,
            kernel,
            borderType=cv2.BORDER_REFLECT_101,
        )
        if resolved == "gradient":
            results.append(
                _restore_dtype(
                    maximum.astype(np.float64) - minimum.astype(np.float64),
                    plane.dtype,
                )
            )
        else:
            results.append(
                (
                    maximum.astype(np.float32)
                    + minimum.astype(np.float32)
                    - 2.0 * plane.astype(np.float32)
                ).astype(np.float32)
            )
    if not rebuild_color:
        return results[0]
    if resolved == "laplacian":
        raise ValueError("形态学 Laplacian 的 float32 结果仅支持单通道。")
    if alpha is not None:
        results.append(alpha.copy())
    return np.stack(results, axis=2)


def morphological_reconstruction(
    image: NDArray[Any],
    *,
    method: str = "opening",
    radius: int = 1,
    connectivity: int = 8,
) -> NDArray[Any]:
    """Opening/closing by geodesic morphological reconstruction."""

    source = _validate_raster(image)
    if source.ndim != 2:
        raise ValueError("形态学重建需要单通道图像。")
    if radius < 1 or radius > 255:
        raise ValueError("形态学重建半径必须在 1 到 255 之间。")
    _validate_connectivity(connectivity)
    resolved = str(method).strip().lower()
    if resolved not in {"opening", "closing"}:
        raise ValueError("形态学重建方法必须为 opening 或 closing。")
    seed_size = radius * 2 + 1
    seed_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (seed_size, seed_size),
    )
    geodesic_kernel = (
        cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        if connectivity == 4
        else np.ones((3, 3), dtype=np.uint8)
    )
    if resolved == "opening":
        marker = cv2.erode(
            source,
            seed_kernel,
            borderType=cv2.BORDER_REFLECT_101,
        )
        for _iteration in range(source.size):
            updated = np.minimum(
                cv2.dilate(
                    marker,
                    geodesic_kernel,
                    borderType=cv2.BORDER_REFLECT_101,
                ),
                source,
            )
            if np.array_equal(updated, marker):
                return updated
            marker = updated
    else:
        marker = cv2.dilate(
            source,
            seed_kernel,
            borderType=cv2.BORDER_REFLECT_101,
        )
        for _iteration in range(source.size):
            updated = np.maximum(
                cv2.erode(
                    marker,
                    geodesic_kernel,
                    borderType=cv2.BORDER_REFLECT_101,
                ),
                source,
            )
            if np.array_equal(updated, marker):
                return updated
            marker = updated
    raise RuntimeError("形态学重建未在有限迭代内收敛。")


def regional_extrema(
    image: NDArray[Any],
    *,
    kind: str = "maxima",
    h: float = 0.0,
    connectivity: int = 8,
) -> NDArray[Any]:
    """Regional or h-extended extrema as a binary raster."""

    source = _validate_raster(image)
    if source.ndim != 2:
        raise ValueError("区域极值需要单通道图像。")
    _validate_connectivity(connectivity)
    _require_finite("扩展极值高度 h", h)
    if h < 0:
        raise ValueError("扩展极值高度 h 不能为负数。")
    resolved = str(kind).strip().lower()
    if resolved not in {"maxima", "minima"}:
        raise ValueError("区域极值类型必须为 maxima 或 minima。")
    kernel = (
        cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        if connectivity == 4
        else np.ones((3, 3), dtype=np.uint8)
    )
    work = source.astype(np.float32)
    if h > 0:
        if resolved == "maxima":
            marker = work - float(h)
            for _iteration in range(source.size):
                updated = np.minimum(cv2.dilate(marker, kernel), work)
                if np.array_equal(updated, marker):
                    break
                marker = updated
            extrema = marker == cv2.dilate(marker, kernel)
        else:
            marker = work + float(h)
            for _iteration in range(source.size):
                updated = np.maximum(cv2.erode(marker, kernel), work)
                if np.array_equal(updated, marker):
                    break
                marker = updated
            extrema = marker == cv2.erode(marker, kernel)
    elif resolved == "maxima":
        extrema = work == cv2.dilate(work, kernel)
    else:
        extrema = work == cv2.erode(work, kernel)
    return _mask_to_binary_values(
        extrema,
        source.dtype,
        foreground_is_high=True,
    )


def fill_binary_holes(
    image: NDArray[Any],
    *,
    foreground_is_high: bool = True,
    cancellation_check: Callable[[], None] | None = None,
) -> NDArray[Any]:
    source = _validate_raster(image)
    if source.ndim != 2:
        raise ValueError("填充孔洞需要单通道图像。")
    if cancellation_check is not None:
        cancellation_check()
    foreground = _binary_mask(source, foreground_is_high=foreground_is_high)
    inverse = (~foreground).astype(np.uint8)
    _count, labels = cv2.connectedComponents(inverse, connectivity=4)
    if cancellation_check is not None:
        cancellation_check()
    border_labels = np.unique(
        np.concatenate(
            (labels[0], labels[-1], labels[:, 0], labels[:, -1])
        )
    )
    holes = (labels > 0) & ~np.isin(labels, border_labels)
    if cancellation_check is not None:
        cancellation_check()
    return _mask_to_binary_values(
        foreground | holes,
        source.dtype,
        foreground_is_high=foreground_is_high,
    )


def extract_binary_contours(
    image: NDArray[Any],
    *,
    foreground_is_high: bool = True,
) -> NDArray[Any]:
    source = _validate_raster(image)
    mask = _binary_mask(source, foreground_is_high=foreground_is_high)
    kernel = np.ones((3, 3), dtype=np.uint8)
    eroded = cv2.erode(
        mask.astype(np.uint8),
        kernel,
        iterations=1,
        borderType=cv2.BORDER_REFLECT_101,
    ).astype(bool)
    contours = mask & ~eroded
    low, high = _binary_values(source.dtype)
    return np.where(contours, high, low).astype(source.dtype)


def remove_small_objects(
    image: NDArray[Any],
    *,
    minimum_area: int,
    connectivity: int = 8,
    foreground_is_high: bool = True,
) -> NDArray[Any]:
    source = _validate_raster(image)
    if minimum_area < 1:
        raise ValueError("最小对象面积必须至少为 1 个像素。")
    _validate_connectivity(connectivity)
    mask = _binary_mask(source, foreground_is_high=foreground_is_high)
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8),
        connectivity=connectivity,
    )
    keep = np.zeros(count, dtype=bool)
    if count > 1:
        keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= minimum_area
    cleaned = keep[labels]
    return _mask_to_binary_values(
        cleaned,
        source.dtype,
        foreground_is_high=foreground_is_high,
    )


def fill_small_holes(
    image: NDArray[Any],
    *,
    maximum_area: int,
    connectivity: int = 8,
    foreground_is_high: bool = True,
) -> NDArray[Any]:
    source = _validate_raster(image)
    if maximum_area < 1:
        raise ValueError("最大孔洞面积必须至少为 1 个像素。")
    _validate_connectivity(connectivity)
    foreground = _binary_mask(source, foreground_is_high=foreground_is_high)
    background = ~foreground
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        background.astype(np.uint8),
        connectivity=connectivity,
    )
    border_labels = set(
        np.concatenate(
            (labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1])
        ).tolist()
    )
    holes = np.zeros_like(foreground)
    for label in range(1, count):
        if (
            label not in border_labels
            and int(stats[label, cv2.CC_STAT_AREA]) <= maximum_area
        ):
            holes |= labels == label
    return _mask_to_binary_values(
        foreground | holes,
        source.dtype,
        foreground_is_high=foreground_is_high,
    )


def clear_border_objects(
    image: NDArray[Any],
    *,
    foreground_is_high: bool = True,
    connectivity: int = 8,
) -> NDArray[Any]:
    """Remove connected foreground components touching any image border."""

    source = _validate_raster(image)
    if source.ndim != 2:
        raise ValueError("清除边界对象需要单通道图像。")
    _validate_connectivity(connectivity)
    foreground = _binary_mask(
        source,
        foreground_is_high=foreground_is_high,
    )
    _count, labels = cv2.connectedComponents(
        foreground.astype(np.uint8),
        connectivity=connectivity,
    )
    border_labels = np.unique(
        np.concatenate(
            (labels[0], labels[-1], labels[:, 0], labels[:, -1])
        )
    )
    cleaned = foreground & ~np.isin(labels, border_labels)
    return _mask_to_binary_values(
        cleaned,
        source.dtype,
        foreground_is_high=foreground_is_high,
    )


def distance_transform(
    image: NDArray[Any],
    *,
    foreground_is_high: bool = True,
    distance_type: str = "l2",
) -> NDArray[np.float32]:
    source = _validate_raster(image)
    resolved = str(distance_type).strip().lower()
    type_map = {"l1": cv2.DIST_L1, "l2": cv2.DIST_L2, "chessboard": cv2.DIST_C}
    if resolved not in type_map:
        raise ValueError("距离类型必须为 l1、l2 或 chessboard。")
    mask_size = 5 if resolved == "l2" else 3
    mask = _binary_mask(source, foreground_is_high=foreground_is_high)
    return cv2.distanceTransform(
        mask.astype(np.uint8),
        type_map[resolved],
        mask_size,
    ).astype(np.float32)


def skeletonize_binary(
    image: NDArray[Any],
    *,
    foreground_is_high: bool = True,
) -> NDArray[Any]:
    """Zhang-Suen thinning with deterministic simultaneous deletions."""

    source = _validate_raster(image)
    skeleton = _binary_mask(source, foreground_is_high=foreground_is_high).astype(np.uint8)
    if min(skeleton.shape) < 3:
        return _mask_to_binary_values(
            skeleton.astype(bool),
            source.dtype,
            foreground_is_high=foreground_is_high,
        )
    changed = True
    while changed:
        changed = False
        for first_subiteration in (True, False):
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
                c1 = p2 * p4 * p6 == 0
                c2 = p4 * p6 * p8 == 0
            else:
                c1 = p2 * p4 * p8 == 0
                c2 = p2 * p6 * p8 == 0
            remove = (
                (skeleton == 1)
                & (neighbors >= 2)
                & (neighbors <= 6)
                & (transitions == 1)
                & c1
                & c2
            )
            if np.any(remove):
                skeleton[remove] = 0
                changed = True
    return _mask_to_binary_values(
        skeleton.astype(bool),
        source.dtype,
        foreground_is_high=foreground_is_high,
    )


def watershed_split(
    image: NDArray[Any],
    *,
    foreground_is_high: bool = True,
    seed_threshold: float = 0.45,
) -> NDArray[Any]:
    source = _validate_raster(image)
    _require_finite("分水岭种子阈值", seed_threshold)
    if not 0.0 < seed_threshold < 1.0:
        raise ValueError("分水岭种子阈值必须在 0 与 1 之间。")
    foreground = _binary_mask(source, foreground_is_high=foreground_is_high)
    if not np.any(foreground):
        return source.copy()
    distance = cv2.distanceTransform(foreground.astype(np.uint8), cv2.DIST_L2, 5)
    local_max = distance == cv2.dilate(distance, np.ones((3, 3), dtype=np.uint8))
    seeds = local_max & (distance >= float(np.max(distance)) * seed_threshold)
    seed_count, seed_labels = cv2.connectedComponents(seeds.astype(np.uint8), connectivity=8)
    if seed_count <= 2:
        return source.copy()
    markers = seed_labels.astype(np.int32) + 1
    markers[~foreground] = 1
    unknown = foreground & ~seeds
    markers[unknown] = 0
    normalized = cv2.normalize(
        source,
        None,
        0,
        255,
        cv2.NORM_MINMAX,
        dtype=cv2.CV_8U,
    )
    rgb = cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)
    cv2.watershed(rgb, markers)
    separated = foreground & (markers > 1)
    separated[markers == -1] = False
    return _mask_to_binary_values(
        separated,
        source.dtype,
        foreground_is_high=foreground_is_high,
    )


def watershed_split_v2(
    image: NDArray[Any],
    *,
    foreground_is_high: bool = True,
    seed_threshold: float = 0.35,
    minimum_seed_area: int = 1,
) -> NDArray[Any]:
    """Marker-controlled watershed with plateau-safe regional-maxima seeds."""

    source = _validate_raster(image)
    if source.ndim != 2:
        raise ValueError("分水岭 v2 需要单通道图像。")
    _require_finite("分水岭 v2 种子阈值", seed_threshold)
    if not 0.0 < seed_threshold < 1.0:
        raise ValueError("分水岭 v2 种子阈值必须在 0 与 1 之间。")
    if minimum_seed_area < 1:
        raise ValueError("分水岭 v2 最小种子面积必须至少为 1。")
    foreground = _binary_mask(source, foreground_is_high=foreground_is_high)
    if not np.any(foreground):
        return source.copy()
    distance = cv2.distanceTransform(
        foreground.astype(np.uint8),
        cv2.DIST_L2,
        5,
    )
    maximum = float(np.max(distance))
    candidate = (
        foreground
        & (distance >= maximum * seed_threshold)
        & (distance == cv2.dilate(distance, np.ones((3, 3), np.float32)))
    )
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        candidate.astype(np.uint8),
        connectivity=8,
    )
    keep = np.zeros(count, dtype=bool)
    if count > 1:
        keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= minimum_seed_area
    seeds = keep[labels]
    seed_count, seed_labels = cv2.connectedComponents(
        seeds.astype(np.uint8),
        connectivity=8,
    )
    if seed_count <= 2:
        return source.copy()
    markers = seed_labels.astype(np.int32) + 1
    markers[~foreground] = 1
    markers[foreground & ~seeds] = 0
    normalized = cv2.normalize(
        distance,
        None,
        0,
        255,
        cv2.NORM_MINMAX,
        dtype=cv2.CV_8U,
    )
    terrain = cv2.cvtColor(255 - normalized, cv2.COLOR_GRAY2BGR)
    cv2.watershed(terrain, markers)
    separated = foreground & (markers > 1)
    separated[markers == -1] = False
    return _mask_to_binary_values(
        separated,
        source.dtype,
        foreground_is_high=foreground_is_high,
    )


def subtract_background(
    image: NDArray[Any],
    *,
    radius: int = 25,
    light_background: bool = False,
    preserve_offset: bool = False,
    border_mode: BorderMode | str = BorderMode.REFLECT,
) -> NDArray[Any]:
    """Subtract a rolling-ball-equivalent morphological background estimate."""

    source = _validate_raster(image)
    if radius < 1 or radius > 2048:
        raise ValueError("背景扣除半径必须在 1 到 2048 像素之间。")
    border = (
        border_mode
        if isinstance(border_mode, BorderMode)
        else _coerce_enum(BorderMode, border_mode, "边界模式")
    )
    if border is BorderMode.WRAP:
        raise ValueError(
            "背景扣除不支持循环边界；"
            "请选择 Reflect101、复制边缘或常量边界。"
        )
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (radius * 2 + 1, radius * 2 + 1),
    )

    def process_plane(plane: NDArray[Any]) -> NDArray[Any]:
        operation = cv2.MORPH_CLOSE if light_background else cv2.MORPH_OPEN
        background = cv2.morphologyEx(
            plane,
            operation,
            kernel,
            borderType=_cv_border(border),
        )
        work = (
            background.astype(np.float64) - plane.astype(np.float64)
            if light_background
            else plane.astype(np.float64) - background.astype(np.float64)
        )
        if preserve_offset:
            work += float(np.median(background[np.isfinite(background)]))
        return _restore_dtype(work, plane.dtype)

    return _apply_color_channels(source, process_plane)


def rolling_ball_background_subtract(
    image: NDArray[Any],
    *,
    radius: float = 25.0,
    ball_height: float = 255.0,
    light_background: bool = False,
    preserve_offset: bool = False,
) -> NDArray[Any]:
    """Subtract a parabolic-opening background without changing v1 behavior."""

    source = _validate_raster(image)
    _require_positive("滑动抛物面半径", radius)
    _require_positive("滑动抛物面高度", ball_height)
    integer_radius = max(1, int(math.ceil(radius)))
    coefficient = float(ball_height) / (float(radius) ** 2)

    def parabola_envelope(
        plane: NDArray[Any],
        *,
        lower: bool,
    ) -> NDArray[np.float64]:
        work = plane.astype(np.float64)
        if not lower:
            work = -work
        for axis in (0, 1):
            padding = [(0, 0), (0, 0)]
            padding[axis] = (integer_radius, integer_radius)
            padded = np.pad(
                work,
                padding,
                mode="reflect" if work.shape[axis] > 1 else "edge",
            )
            length = work.shape[axis]
            envelope = np.full(work.shape, np.inf, dtype=np.float64)
            for offset in range(-integer_radius, integer_radius + 1):
                start = integer_radius + offset
                selection = [slice(None), slice(None)]
                selection[axis] = slice(start, start + length)
                envelope = np.minimum(
                    envelope,
                    padded[tuple(selection)]
                    + coefficient * float(offset * offset),
                )
            work = envelope
        return work if lower else -work

    def process_plane(plane: NDArray[Any]) -> NDArray[Any]:
        if light_background:
            dilated = parabola_envelope(plane, lower=False)
            background = parabola_envelope(dilated, lower=True)
            corrected = background - plane.astype(np.float64)
        else:
            eroded = parabola_envelope(plane, lower=True)
            background = parabola_envelope(eroded, lower=False)
            corrected = plane.astype(np.float64) - background
        if preserve_offset:
            finite_background = background[np.isfinite(background)]
            if finite_background.size:
                corrected += float(np.median(finite_background))
        return _restore_dtype(corrected, plane.dtype)

    return _apply_color_channels(source, process_plane)


def flat_field_correction(
    image: NDArray[Any],
    *,
    radius: float = 25.0,
    method: str = "gaussian",
    preserve_mean: bool = True,
    reference_image: NDArray[Any] | None = None,
    reference_levels: object | None = None,
) -> NDArray[Any]:
    """Correct multiplicative illumination using an estimate or reference.

    The legacy estimated-field path is retained byte-for-byte.  When a
    reference is supplied, dimensions, channels and dtype must match exactly;
    RGB/RGBA channels are normalized independently and the source Alpha channel
    is never modified.
    """

    source = _validate_raster(image)
    reference = (
        None if reference_image is None else _validate_raster(reference_image)
    )
    if reference is not None:
        if reference.shape != source.shape:
            raise ValueError("参考平场与源图像的尺寸和通道必须完全一致。")
        if reference.dtype != source.dtype:
            raise ValueError("参考平场与源图像的像素类型必须完全一致。")
    else:
        _require_positive("平场估计半径", radius)
        resolved = str(method).strip().lower()
        if resolved not in {"gaussian", "morphology"}:
            raise ValueError("平场估计方法必须为 gaussian 或 morphology。")

    expected_channels = 1 if source.ndim == 2 else min(3, source.shape[2])
    explicit_levels = _normalize_flat_field_reference_levels(
        reference_levels,
        expected_channels=expected_channels,
    )

    def correct_estimated_plane(plane: NDArray[Any]) -> NDArray[Any]:
        work = plane.astype(np.float32)
        if resolved == "gaussian":
            flat = cv2.GaussianBlur(
                work,
                (0, 0),
                sigmaX=float(radius),
                sigmaY=float(radius),
                borderType=cv2.BORDER_REFLECT_101,
            )
        else:
            integer_radius = max(1, int(round(radius)))
            size = integer_radius * 2 + 1
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (size, size),
            )
            flat = cv2.morphologyEx(
                work,
                cv2.MORPH_OPEN,
                kernel,
                borderType=cv2.BORDER_REFLECT_101,
            )
        finite_positive = flat[np.isfinite(flat) & (flat > 0)]
        if finite_positive.size == 0:
            raise ValueError("平场估计不包含正有限值，无法执行除法校正。")
        reference = (
            float(np.mean(finite_positive)) if preserve_mean else 1.0
        )
        epsilon = max(
            float(np.percentile(finite_positive, 0.1)),
            np.finfo(np.float32).eps,
        )
        corrected = work * reference / np.maximum(flat, epsilon)
        return _restore_dtype(corrected, plane.dtype)

    if reference is None:
        return _apply_color_channels(source, correct_estimated_plane)

    source_planes = (
        (source,)
        if source.ndim == 2
        else tuple(source[..., index] for index in range(expected_channels))
    )
    reference_planes = (
        (reference,)
        if reference.ndim == 2
        else tuple(
            reference[..., index] for index in range(expected_channels)
        )
    )
    corrected_planes: list[NDArray[Any]] = []
    for index, (source_plane, flat_plane) in enumerate(
        zip(source_planes, reference_planes, strict=True)
    ):
        flat = flat_plane.astype(np.float64)
        invalid = ~np.isfinite(flat) | (flat <= 0)
        if np.any(invalid):
            raise ValueError(
                "参考平场必须全部为正有限值；"
                f"通道 {index + 1} 含 {int(np.count_nonzero(invalid))} 个无效像素。"
            )
        level = (
            explicit_levels[index]
            if explicit_levels is not None
            else (
                float(np.mean(flat, dtype=np.float64))
                if preserve_mean
                else 1.0
            )
        )
        corrected = (
            source_plane.astype(np.float64)
            * level
            / flat
        )
        corrected_planes.append(_restore_dtype(corrected, source_plane.dtype))

    if source.ndim == 2:
        return np.ascontiguousarray(corrected_planes[0])
    result = source.copy()
    for index, corrected in enumerate(corrected_planes):
        result[..., index] = corrected
    return np.ascontiguousarray(result)


def flat_field_reference_levels(
    reference_image: NDArray[Any],
    *,
    preserve_mean: bool = True,
) -> tuple[float, ...]:
    """Freeze the exact per-channel normalization used by reference correction."""

    reference = _validate_raster(reference_image)
    channel_count = 1 if reference.ndim == 2 else min(3, reference.shape[2])
    planes = (
        (reference,)
        if reference.ndim == 2
        else tuple(reference[..., index] for index in range(channel_count))
    )
    levels: list[float] = []
    for index, plane in enumerate(planes):
        values = plane.astype(np.float32)
        invalid = ~np.isfinite(values) | (values <= 0)
        if np.any(invalid):
            raise ValueError(
                "参考平场必须全部为正有限值；"
                f"通道 {index + 1} 含 {int(np.count_nonzero(invalid))} 个无效像素。"
            )
        levels.append(
            float(np.mean(values, dtype=np.float64))
            if preserve_mean
            else 1.0
        )
    return tuple(levels)


def _normalize_flat_field_reference_levels(
    levels: object | None,
    *,
    expected_channels: int,
) -> tuple[float, ...] | None:
    if levels is None:
        return None
    raw = levels if isinstance(levels, (list, tuple)) else (levels,)
    if len(raw) != expected_channels:
        raise ValueError(
            f"参考平场归一化值必须包含 {expected_channels} 个通道。"
        )
    normalized = tuple(float(value) for value in raw)
    if any(not math.isfinite(value) or value <= 0 for value in normalized):
        raise ValueError("参考平场归一化值必须全部为正有限数。")
    return normalized


def custom_convolution(
    image: NDArray[Any],
    *,
    kernel: ParameterValue,
    kernel_width: int,
    kernel_height: int,
    normalize_kernel: bool = False,
    offset: float = 0.0,
    border_mode: BorderMode | str = BorderMode.REFLECT,
) -> NDArray[Any]:
    source = _validate_raster(image)
    if not isinstance(kernel, tuple):
        raise ValueError("自定义卷积核必须是不可变数值序列。")
    if (
        kernel_width < 1
        or kernel_height < 1
        or kernel_width % 2 == 0
        or kernel_height % 2 == 0
    ):
        raise ValueError("自定义卷积核的宽和高必须是正奇数。")
    if len(kernel) != kernel_width * kernel_height:
        raise ValueError("自定义卷积核元素数量与宽高不匹配。")
    try:
        matrix = np.asarray([float(value) for value in kernel], dtype=np.float64).reshape(
            kernel_height,
            kernel_width,
        )
    except (TypeError, ValueError) as error:
        raise ValueError("自定义卷积核只能包含有限数值。") from error
    if not np.all(np.isfinite(matrix)):
        raise ValueError("自定义卷积核只能包含有限数值。")
    _require_finite("自定义卷积偏移量", offset)
    if normalize_kernel:
        total = float(np.sum(matrix))
        if math.isclose(total, 0.0, abs_tol=1e-15):
            raise ValueError("卷积核总和为零，不能执行归一化。")
        matrix = matrix / total
    border = (
        border_mode
        if isinstance(border_mode, BorderMode)
        else _coerce_enum(BorderMode, border_mode, "边界模式")
    )
    if border is BorderMode.WRAP:
        raise ValueError(
            "自定义卷积不支持循环边界；"
            "请选择 Reflect101、复制边缘或常量边界。"
        )

    def process_plane(plane: NDArray[Any]) -> NDArray[Any]:
        destination_depth = cv2.CV_32F if plane.dtype == np.float32 else cv2.CV_64F
        filtered = cv2.filter2D(
            plane,
            destination_depth,
            matrix,
            borderType=_cv_border(border),
        )
        return _restore_dtype(filtered + offset, plane.dtype)

    return _apply_color_channels(source, process_plane)


def apply_math_operation(
    image: NDArray[Any],
    *,
    operation: ImageOperation | str,
    **parameters: ParameterValue,
) -> NDArray[Any]:
    source = _validate_raster(image)
    resolved = (
        operation
        if isinstance(operation, ImageOperation)
        else _coerce_enum(ImageOperation, operation, "数学操作")
    )

    def process_plane(plane: NDArray[Any]) -> NDArray[Any]:
        work = plane.astype(np.float64)
        if resolved is ImageOperation.INVERT:
            low = float(parameters.get("minimum", _working_range(plane)[0]))
            high = float(parameters.get("maximum", _working_range(plane)[1]))
            _validate_range(low, high, "反相范围")
            result = low + high - work
        elif resolved in {
            ImageOperation.ADD,
            ImageOperation.SUBTRACT,
            ImageOperation.MULTIPLY,
            ImageOperation.DIVIDE,
        }:
            default_value = (
                0.0
                if resolved in {ImageOperation.ADD, ImageOperation.SUBTRACT}
                else 1.0
            )
            value = float(parameters.get("value", default_value))
            _require_finite("数学运算常数", value)
            if resolved is ImageOperation.ADD:
                result = work + value
            elif resolved is ImageOperation.SUBTRACT:
                result = work - value
            elif resolved is ImageOperation.MULTIPLY:
                result = work * value
            else:
                if math.isclose(value, 0.0):
                    raise ValueError("除数不能为零。")
                result = work / value
        elif resolved is ImageOperation.GAMMA:
            gamma = float(parameters.get("gamma", 1.0))
            _require_positive("Gamma", gamma)
            low = float(parameters.get("minimum", _working_range(plane)[0]))
            high = float(parameters.get("maximum", _working_range(plane)[1]))
            _validate_range(low, high, "Gamma 输入范围")
            normalized = np.clip((work - low) / (high - low), 0.0, 1.0)
            result = low + np.power(normalized, gamma) * (high - low)
        elif resolved is ImageOperation.LOG:
            if np.any(np.isfinite(work) & (work < 0)):
                raise ValueError("Log 运算要求所有有限像素均不小于零。")
            result = np.log1p(work)
        elif resolved is ImageOperation.EXP:
            with np.errstate(over="ignore", invalid="ignore"):
                result = np.exp(work)
            if np.any(np.isfinite(work) & ~np.isfinite(result)):
                raise ValueError("Exp 运算产生溢出，请先缩小输入范围。")
        elif resolved is ImageOperation.SQRT:
            if np.any(np.isfinite(work) & (work < 0)):
                raise ValueError("Sqrt 运算要求所有有限像素均不小于零。")
            result = np.sqrt(work)
        elif resolved is ImageOperation.ABS:
            result = np.abs(work)
        elif resolved is ImageOperation.CLAMP:
            minimum = float(parameters.get("minimum", _working_range(plane)[0]))
            maximum = float(parameters.get("maximum", _working_range(plane)[1]))
            _validate_range(minimum, maximum, "截断范围")
            result = np.clip(work, minimum, maximum)
        else:  # pragma: no cover - guarded by dispatcher
            raise ValueError(f"不支持的数学操作：{resolved.value}")
        return _restore_dtype(result, plane.dtype)

    return _apply_color_channels(source, process_plane)


def apply_scientific_math_transform(
    image: NDArray[Any],
    *,
    operation: ImageOperation | str,
    result_mode: str = "float32",
    output_min: float = 0.0,
    output_max: float = 1.0,
) -> NDArray[Any]:
    """Log/Exp/Sqrt v2 with explicit scientific output semantics."""

    source = _validate_raster(image)
    resolved = (
        operation
        if isinstance(operation, ImageOperation)
        else _coerce_enum(ImageOperation, operation, "科学数学操作")
    )
    if resolved not in {
        ImageOperation.LOG_V2,
        ImageOperation.EXP_V2,
        ImageOperation.SQRT_V2,
    }:
        raise ValueError("科学数学变换仅支持 Log/Exp/Sqrt v2。")
    mode = str(result_mode).strip().lower()
    if mode not in {"float32", "preserve", "remap"}:
        raise ValueError("结果模式必须为 float32、preserve 或 remap。")
    if mode == "remap":
        _validate_range(output_min, output_max, "科学变换重映射范围")
    if mode == "float32" and source.ndim == 3 and source.shape[2] > 1:
        raise ValueError("float32 科学变换结果仅支持单通道权威栅格。")

    def transform_plane(plane: NDArray[Any]) -> NDArray[Any]:
        work = plane.astype(np.float64)
        finite = np.isfinite(work)
        if resolved in {ImageOperation.LOG_V2, ImageOperation.SQRT_V2} and np.any(
            finite & (work < 0)
        ):
            label = "Log" if resolved is ImageOperation.LOG_V2 else "Sqrt"
            raise ValueError(f"{label} v2 要求所有有限像素均不小于零。")
        if resolved is ImageOperation.LOG_V2:
            result = np.log1p(work)
        elif resolved is ImageOperation.SQRT_V2:
            result = np.sqrt(work)
        else:
            with np.errstate(over="ignore", invalid="ignore"):
                result = np.exp(work)
            if np.any(finite & ~np.isfinite(result)):
                raise ValueError("Exp v2 产生溢出，请先缩小输入范围。")
        if mode == "float32":
            return result.astype(np.float32)
        if mode == "preserve":
            return _restore_dtype(result, plane.dtype)
        result_finite = np.isfinite(result)
        if not np.any(result_finite):
            mapped = np.zeros_like(result)
        else:
            low = float(np.min(result[result_finite]))
            high = float(np.max(result[result_finite]))
            mapped = result.copy()
            if math.isclose(low, high):
                mapped[result_finite] = output_min
            else:
                mapped[result_finite] = (
                    (result[result_finite] - low)
                    * ((output_max - output_min) / (high - low))
                    + output_min
                )
        return _restore_dtype(mapped, plane.dtype)

    return _apply_color_channels(source, transform_plane)


def image_calculator(
    first: NDArray[Any],
    second: NDArray[Any],
    *,
    operation: str,
    result_mode: str = "preserve",
) -> NDArray[Any]:
    left = _validate_raster(first)
    right = _validate_raster(second)
    if left.shape != right.shape:
        raise ValueError("图像计算器要求两幅图像的宽高和通道数完全一致。")
    if left.dtype != right.dtype:
        raise TypeError("图像计算器要求两幅图像的位深完全一致。")
    resolved = str(operation).strip().lower()
    supported = {
        "add",
        "subtract",
        "multiply",
        "divide",
        "difference",
        "minimum",
        "maximum",
        "mean",
        "and",
        "or",
        "xor",
        "copy",
    }
    if resolved not in supported:
        raise ValueError("不支持的图像计算器运算。")
    if resolved in {"and", "or", "xor"} and left.dtype.kind not in {"u", "i"}:
        raise TypeError("AND、OR、XOR 仅支持整数图像。")
    resolved_result_mode = str(result_mode).strip().lower()
    if resolved_result_mode not in {"preserve", "float32"}:
        raise ValueError("图像计算器结果模式必须为 preserve 或 float32。")
    if resolved_result_mode == "float32" and left.ndim == 3 and left.shape[2] > 1:
        raise ValueError("图像计算器 float32 结果仅支持单通道权威栅格。")

    def calculate(a: NDArray[Any], b: NDArray[Any]) -> NDArray[Any]:
        if resolved == "copy":
            result = b.astype(np.float64)
            if resolved_result_mode == "float32":
                return result.astype(np.float32)
            return b.copy()
        if resolved == "and":
            result = np.bitwise_and(a, b)
            return (
                result.astype(np.float32)
                if resolved_result_mode == "float32"
                else result
            )
        if resolved == "or":
            result = np.bitwise_or(a, b)
            return (
                result.astype(np.float32)
                if resolved_result_mode == "float32"
                else result
            )
        if resolved == "xor":
            result = np.bitwise_xor(a, b)
            return (
                result.astype(np.float32)
                if resolved_result_mode == "float32"
                else result
            )
        aw = a.astype(np.float64)
        bw = b.astype(np.float64)
        if resolved == "add":
            result = aw + bw
        elif resolved == "subtract":
            result = aw - bw
        elif resolved == "multiply":
            result = aw * bw
        elif resolved == "divide":
            if np.any(bw == 0):
                raise ValueError("图像计算器除法的第二幅图像包含零值。")
            result = aw / bw
        elif resolved == "difference":
            result = np.abs(aw - bw)
        elif resolved == "minimum":
            result = np.minimum(aw, bw)
        elif resolved == "maximum":
            result = np.maximum(aw, bw)
        else:
            result = (aw + bw) / 2.0
        if resolved_result_mode == "float32":
            return result.astype(np.float32)
        return _restore_dtype(result, a.dtype)

    if left.ndim == 3 and left.shape[2] == 4:
        color = np.stack(
            [calculate(left[..., index], right[..., index]) for index in range(3)],
            axis=2,
        )
        return np.dstack((color, left[..., 3].copy()))
    if left.ndim == 3:
        return np.stack(
            [calculate(left[..., index], right[..., index]) for index in range(left.shape[2])],
            axis=2,
        )
    return calculate(left, right)


def suppress_stripes(
    image: NDArray[Any],
    *,
    direction: str = "horizontal",
    notch_width: float = 0.02,
    protect_radius: float = 0.02,
    strength: float = 1.0,
) -> NDArray[Any]:
    """Suppress directional periodic stripes with a frequency-axis notch."""

    source = _validate_raster(image)
    resolved_direction = str(direction).strip().lower()
    if resolved_direction not in {"horizontal", "vertical"}:
        raise ValueError("条纹方向必须为 horizontal 或 vertical。")
    for name, value in (
        ("条纹抑制陷波宽度", notch_width),
        ("条纹抑制低频保护半径", protect_radius),
        ("条纹抑制强度", strength),
    ):
        _require_finite(name, value)
    if not 0.0 < notch_width <= 0.25:
        raise ValueError("条纹抑制陷波宽度必须在 0 与 0.25 之间。")
    if not 0.0 <= protect_radius <= 0.25:
        raise ValueError("条纹抑制低频保护半径必须在 0 与 0.25 之间。")
    if not 0.0 <= strength <= 1.0:
        raise ValueError("条纹抑制强度必须在 0 与 1 之间。")

    def process_plane(plane: NDArray[Any]) -> NDArray[Any]:
        height, width = plane.shape
        fy = np.fft.fftfreq(height).reshape(-1, 1)
        fx = np.fft.fftfreq(width).reshape(1, -1)
        radius = np.sqrt(fx * fx + fy * fy)
        axis_distance = np.abs(fx if resolved_direction == "horizontal" else fy)
        response = np.ones((height, width), dtype=np.float64)
        notch = (axis_distance <= notch_width) & (radius > protect_radius)
        response[notch] = 1.0 - strength
        filtered = np.fft.ifft2(np.fft.fft2(plane.astype(np.float32)) * response).real
        return _restore_dtype(filtered, plane.dtype)

    return _apply_color_channels(source, process_plane)


def _threshold_histogram(
    image: NDArray[Any],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    finite = np.asarray(image)[np.isfinite(image)].astype(np.float64)
    value_min = float(np.min(finite))
    value_max = float(np.max(finite))
    if math.isclose(value_min, value_max):
        return np.asarray([float(finite.size)]), np.asarray([value_min])
    if image.dtype == np.uint8:
        bins = 256
        histogram = np.bincount(finite.astype(np.uint8), minlength=bins).astype(np.float64)
        centers = np.arange(bins, dtype=np.float64)
        return histogram, centers
    if image.dtype == np.uint16:
        # 4096 bins avoid a 65K scan while preserving the native threshold scale.
        bins = 4096
    else:
        bins = 2048
    histogram, edges = np.histogram(finite, bins=bins, range=(value_min, value_max))
    centers = (edges[:-1] + edges[1:]) / 2.0
    return histogram.astype(np.float64), centers.astype(np.float64)


def _otsu_threshold(
    histogram: NDArray[np.float64],
    centers: NDArray[np.float64],
) -> float:
    if histogram.size == 1:
        return float(centers[0])
    weights = histogram.astype(np.float64)
    total = float(np.sum(weights))
    cumulative_weight = np.cumsum(weights)
    cumulative_mean = np.cumsum(weights * centers)
    denominator = cumulative_weight * (total - cumulative_weight)
    between = np.zeros_like(denominator)
    valid = denominator > 0
    between[valid] = (
        (cumulative_mean[-1] * cumulative_weight[valid] - cumulative_mean[valid] * total)
        ** 2
        / denominator[valid]
    )
    return float(centers[int(np.argmax(between))])


def _isodata_threshold(
    histogram: NDArray[np.float64],
    centers: NDArray[np.float64],
) -> float:
    if histogram.size == 1:
        return float(centers[0])
    threshold = float(np.average(centers, weights=histogram))
    for _iteration in range(256):
        lower = centers <= threshold
        upper = ~lower
        lower_weight = float(np.sum(histogram[lower]))
        upper_weight = float(np.sum(histogram[upper]))
        if lower_weight <= 0 or upper_weight <= 0:
            break
        lower_mean = float(np.sum(histogram[lower] * centers[lower]) / lower_weight)
        upper_mean = float(np.sum(histogram[upper] * centers[upper]) / upper_weight)
        updated = (lower_mean + upper_mean) / 2.0
        if abs(updated - threshold) <= max(np.spacing(threshold), 1e-12):
            threshold = updated
            break
        threshold = updated
    return threshold


def _triangle_threshold(
    histogram: NDArray[np.float64],
    centers: NDArray[np.float64],
) -> float:
    nonzero = np.flatnonzero(histogram)
    if nonzero.size <= 1:
        return float(centers[int(nonzero[0]) if nonzero.size else 0])
    left = int(nonzero[0])
    right = int(nonzero[-1])
    peak = int(np.argmax(histogram))
    # Work on the longer tail and reverse when the long tail is on the left.
    reversed_axis = peak - left > right - peak
    values = histogram[left : right + 1]
    sample_centers = centers[left : right + 1]
    if reversed_axis:
        values = values[::-1]
        sample_centers = sample_centers[::-1]
        peak = len(values) - 1 - (peak - left)
    else:
        peak -= left
    end = len(values) - 1
    x1, y1 = float(peak), float(values[peak])
    x2, y2 = float(end), float(values[end])
    denominator = math.hypot(y2 - y1, x2 - x1)
    if denominator <= 0:
        return float(sample_centers[peak])
    indices = np.arange(peak, end + 1, dtype=np.float64)
    distances = np.abs(
        (y2 - y1) * indices
        - (x2 - x1) * values[peak : end + 1]
        + x2 * y1
        - y2 * x1
    ) / denominator
    selected = peak + int(np.argmax(distances))
    return float(sample_centers[selected])


def fft_filter(
    image: NDArray[Any],
    *,
    mode: str = "lowpass",
    low_cutoff: float = 0.0,
    high_cutoff: float = 0.15,
    order: int = 2,
    channel: str = "per_channel",
    output_float: bool = False,
    boundary: str = "periodic",
    tukey_alpha: float = 0.25,
    frequency_unit: str = "cycles_per_pixel",
    pixel_size: float | None = None,
) -> NDArray[Any]:
    """Apply a Butterworth radial frequency filter.

    Cutoffs are cycles per pixel in ``[0, 0.5]``.  ``bandpass`` keeps
    ``low_cutoff <= f <= high_cutoff``; ``bandstop`` removes that interval.
    """

    source = _validate_raster(image)
    resolved_boundary = str(boundary).strip().lower()
    if resolved_boundary not in {"periodic", "mirror_pad", "tukey"}:
        raise ValueError("FFT 边界策略必须为 periodic、mirror_pad 或 tukey。")
    resolved_frequency_unit = str(frequency_unit).strip().lower()
    if resolved_frequency_unit not in {
        "cycles_per_pixel",
        "cycles_per_unit",
    }:
        raise ValueError(
            "FFT 频率单位必须为 cycles_per_pixel 或 cycles_per_unit。"
        )
    if resolved_frequency_unit == "cycles_per_unit":
        if pixel_size is None:
            raise ValueError(
                "cycles_per_unit 需要显式 pixel_size；"
                "当前处理请求不携带标定上下文。"
            )
        _require_positive("FFT 像素尺寸", pixel_size)
        low_cutoff *= float(pixel_size)
        high_cutoff *= float(pixel_size)
    _require_finite("Tukey alpha", tukey_alpha)
    if not 0.0 <= tukey_alpha <= 1.0:
        raise ValueError("Tukey alpha 必须在 0 到 1 之间。")
    resolved_mode = str(mode).strip().lower()
    if resolved_mode not in {"lowpass", "highpass", "bandpass", "bandstop"}:
        raise ValueError("FFT 模式必须为 lowpass、highpass、bandpass 或 bandstop。")
    if not 0.0 <= low_cutoff <= 0.5 or not 0.0 <= high_cutoff <= 0.5:
        raise ValueError("FFT 截止频率必须在 0 到 0.5 周期/像素之间。")
    if resolved_mode in {"bandpass", "bandstop"} and high_cutoff <= low_cutoff:
        raise ValueError("带通或带阻滤波的高截止频率必须大于低截止频率。")
    if resolved_mode == "lowpass" and high_cutoff <= 0:
        raise ValueError("低通滤波的高截止频率必须为正数。")
    if resolved_mode == "highpass" and low_cutoff <= 0:
        raise ValueError("高通滤波的低截止频率必须为正数。")
    if order < 1 or order > 16:
        raise ValueError("Butterworth 阶数必须在 1 到 16 之间。")

    if source.ndim == 3 and channel != "per_channel":
        planes = [_select_scalar_channel(source, channel)]
        collapse = True
    else:
        planes, alpha = _split_color_channels(source)
        collapse = False
    filtered_planes = [
        _fft_filter_plane(
            plane.astype(np.float32),
            mode=resolved_mode,
            low_cutoff=low_cutoff,
            high_cutoff=high_cutoff,
            order=order,
            boundary=resolved_boundary,
            tukey_alpha=tukey_alpha,
        )
        for plane in planes
    ]
    if collapse:
        result_float = filtered_planes[0]
    elif source.ndim == 2:
        result_float = filtered_planes[0]
    else:
        if alpha is not None:
            filtered_planes.append(alpha.astype(np.float32))
        result_float = np.stack(filtered_planes, axis=2)
    if output_float:
        return result_float.astype(np.float32)
    return _restore_dtype(result_float, source.dtype)


def _fft_filter_plane(
    plane: NDArray[np.float32],
    *,
    mode: str,
    low_cutoff: float,
    high_cutoff: float,
    order: int,
    boundary: str = "periodic",
    tukey_alpha: float = 0.25,
) -> NDArray[np.float32]:
    original_height, original_width = plane.shape
    crop_y = 0
    crop_x = 0
    work = plane
    if boundary == "mirror_pad":
        pad_y = max(1, original_height // 2)
        pad_x = max(1, original_width // 2)
        work = np.pad(
            work,
            ((pad_y, pad_y), (pad_x, pad_x)),
            mode="reflect",
        )
        crop_y = pad_y
        crop_x = pad_x
    elif boundary == "tukey":
        window = np.outer(
            _tukey_window(original_height, tukey_alpha),
            _tukey_window(original_width, tukey_alpha),
        )
        work = work * window.astype(np.float32)
    height, width = work.shape
    fy = np.fft.fftfreq(height).reshape(-1, 1)
    fx = np.fft.fftfreq(width).reshape(1, -1)
    radius = np.sqrt(fx * fx + fy * fy)
    epsilon = np.finfo(np.float64).eps

    def lowpass(cutoff: float) -> NDArray[np.float64]:
        return 1.0 / (1.0 + np.power(radius / max(cutoff, epsilon), 2 * order))

    def highpass(cutoff: float) -> NDArray[np.float64]:
        result = 1.0 - lowpass(cutoff)
        result[0, 0] = 0.0
        return result

    if mode == "lowpass":
        response = lowpass(high_cutoff)
    elif mode == "highpass":
        response = highpass(low_cutoff)
    elif mode == "bandpass":
        response = highpass(low_cutoff) * lowpass(high_cutoff)
    else:
        response = 1.0 - highpass(low_cutoff) * lowpass(high_cutoff)
    spectrum = np.fft.fft2(work)
    result = np.fft.ifft2(spectrum * response).real.astype(np.float32)
    if boundary == "mirror_pad":
        result = result[
            crop_y : crop_y + original_height,
            crop_x : crop_x + original_width,
        ]
    return result


def _tukey_window(length: int, alpha: float) -> NDArray[np.float64]:
    if length <= 1 or alpha <= 0.0:
        return np.ones(length, dtype=np.float64)
    if alpha >= 1.0:
        return np.hanning(length).astype(np.float64)
    x = np.linspace(0.0, 1.0, length)
    window = np.ones(length, dtype=np.float64)
    first = x < alpha / 2.0
    last = x >= 1.0 - alpha / 2.0
    window[first] = 0.5 * (
        1.0 + np.cos(2.0 * np.pi / alpha * (x[first] - alpha / 2.0))
    )
    window[last] = 0.5 * (
        1.0
        + np.cos(
            2.0 * np.pi / alpha * (x[last] - 1.0 + alpha / 2.0)
        )
    )
    return window


def fft_power_spectrum(
    image: NDArray[Any],
    *,
    logarithmic: bool = True,
    centered: bool = True,
    window: str = "none",
    tukey_alpha: float = 0.25,
) -> NDArray[np.float32]:
    """Return an unnormalized float32 FFT power spectrum."""

    source = _validate_raster(image)
    if source.ndim != 2:
        raise ValueError("FFT 功率谱需要单通道图像。")
    resolved_window = str(window).strip().lower()
    if resolved_window not in {"none", "tukey"}:
        raise ValueError("FFT 功率谱窗函数必须为 none 或 tukey。")
    _require_finite("Tukey alpha", tukey_alpha)
    if not 0.0 <= tukey_alpha <= 1.0:
        raise ValueError("Tukey alpha 必须在 0 到 1 之间。")
    work = source.astype(np.float32)
    if resolved_window == "tukey":
        work = work * np.outer(
            _tukey_window(work.shape[0], tukey_alpha),
            _tukey_window(work.shape[1], tukey_alpha),
        ).astype(np.float32)
    spectrum = np.fft.fft2(work)
    power = np.square(np.abs(spectrum), dtype=np.float64)
    if logarithmic:
        power = np.log1p(power)
    if centered:
        power = np.fft.fftshift(power)
    return power.astype(np.float32)


def _coerce_enum(enum_type, value: Any, label: str):
    try:
        return enum_type(str(value))
    except (TypeError, ValueError) as error:
        allowed = "、".join(str(item.value) for item in enum_type)
        raise ValueError(f"不支持的{label}“{value}”；可选值为：{allowed}。") from error


def _freeze_raster(image: NDArray[Any]) -> NDArray[Any]:
    array = np.ascontiguousarray(_validate_raster(image)).copy()
    array.setflags(write=False)
    return array


def _freeze_roi_mask(
    mask: NDArray[np.bool_],
    expected_shape: tuple[int, int],
) -> NDArray[np.bool_]:
    array = np.asarray(mask, dtype=bool)
    if array.shape != expected_shape:
        raise ValueError(
            f"ROI 掩膜尺寸 {array.shape!r} 与图像尺寸 {expected_shape!r} 不一致。"
        )
    frozen = np.ascontiguousarray(array).copy()
    frozen.setflags(write=False)
    return frozen


def _expanded_statistics_mask(
    values: NDArray[Any],
    statistics_mask: NDArray[np.bool_] | None,
) -> NDArray[np.bool_] | np.bool_:
    """Return a broadcastable mask selecting samples used for global stats."""

    if statistics_mask is None:
        return np.bool_(True)
    mask = np.asarray(statistics_mask, dtype=bool)
    if mask.shape != values.shape[:2]:
        raise ValueError(
            f"统计 ROI 掩膜尺寸 {mask.shape!r} 与图像尺寸 "
            f"{values.shape[:2]!r} 不一致。"
        )
    if values.ndim == 3:
        return mask[..., np.newaxis]
    return mask


def _roi_statistics_values(
    image: NDArray[Any],
    statistics_mask: NDArray[np.bool_] | None,
) -> NDArray[Any]:
    """Return samples inside ROI without changing the image being processed."""

    source = np.asarray(image)
    if statistics_mask is None:
        return source
    mask = np.asarray(statistics_mask, dtype=bool)
    if mask.shape != source.shape[:2]:
        raise ValueError(
            f"统计 ROI 掩膜尺寸 {mask.shape!r} 与图像尺寸 "
            f"{source.shape[:2]!r} 不一致。"
        )
    if not np.any(mask):
        raise ValueError("统计 ROI 不能为空。")
    return source[mask]


def _isolate_roi_domain(
    image: NDArray[Any],
    roi_mask: NDArray[np.bool_] | None,
    *,
    foreground_is_high: bool,
) -> NDArray[Any]:
    """Treat every pixel outside ROI as binary-domain background."""

    source = np.asarray(image)
    if roi_mask is None:
        return source
    mask = np.asarray(roi_mask, dtype=bool)
    if source.ndim != 2 or mask.shape != source.shape:
        raise ValueError("连通域 ROI 必须与单通道图像尺寸一致。")
    low, high = _binary_values(source.dtype)
    background = low if foreground_is_high else high
    isolated = source.copy()
    isolated[~mask] = background
    return isolated


def _freeze_parameters(
    parameters: tuple[tuple[str, ParameterValue], ...],
) -> tuple[tuple[str, ParameterValue], ...]:
    frozen: list[tuple[str, ParameterValue]] = []
    seen: set[str] = set()
    for raw_key, raw_value in parameters:
        key = str(raw_key)
        if key in seen:
            raise ValueError(f"图像操作参数重复：{key}")
        seen.add(key)
        value: ParameterValue
        if isinstance(raw_value, list):
            value = tuple(_parameter_scalar(item) for item in raw_value)
        elif isinstance(raw_value, tuple):
            value = tuple(_parameter_scalar(item) for item in raw_value)
        else:
            value = _parameter_scalar(raw_value)
        frozen.append((key, value))
    return tuple(frozen)


def _parameter_scalar(value: Any) -> ParameterScalar:
    if isinstance(value, Enum):
        return str(value.value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("图像操作参数不能包含 NaN 或无穷大。")
        return value
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, np.generic):
        return _parameter_scalar(value.item())
    raise TypeError(f"不支持的不可变操作参数类型：{type(value).__name__}")


def _validate_raster(image: NDArray[Any]) -> NDArray[Any]:
    array = np.asarray(image)
    if array.ndim not in {2, 3}:
        raise ValueError("图像数据必须采用 H×W 或 H×W×C 形状。")
    if array.shape[0] <= 0 or array.shape[1] <= 0:
        raise ValueError("图像宽度和高度必须为正数。")
    if array.ndim == 3 and array.shape[2] not in {1, 3, 4}:
        raise ValueError("图像通道数必须为 1、3 或 4。")
    if array.dtype not in {np.dtype(np.uint8), np.dtype(np.uint16), np.dtype(np.float32)}:
        raise TypeError("图像位深必须为 uint8、uint16 或 float32。")
    return array


def _blend_roi(
    source: NDArray[Any],
    processed: NDArray[Any],
    roi_mask: NDArray[np.bool_] | None,
) -> NDArray[Any]:
    if roi_mask is None:
        return np.asarray(processed)
    if source.shape != processed.shape:
        raise ValueError("ROI 处理结果必须与源图像尺寸和通道数一致。")
    mask = roi_mask if source.ndim == 2 else roi_mask[..., np.newaxis]
    return np.where(mask, processed, source)


def _fill_outside_mask(
    image: NDArray[Any],
    mask: NDArray[np.bool_],
    fill_value: ParameterValue,
) -> NDArray[Any]:
    source = _validate_raster(image)
    if mask.shape != source.shape[:2]:
        raise ValueError("裁剪后的 ROI 掩膜尺寸与输出图像不一致。")
    fill = _normalize_fill_value(
        source,
        fill_value,
        field_name="ROI 外填充值",
    )
    result = source.copy()
    result[~mask] = fill
    return result


def _crop_with_transparent_outside(
    image: NDArray[Any],
    mask: NDArray[np.bool_],
) -> NDArray[np.uint8]:
    source = _validate_raster(image)
    if mask.shape != source.shape[:2]:
        raise ValueError("裁剪后的 ROI 掩膜尺寸与输出图像不一致。")
    if source.dtype != np.dtype(np.uint8):
        raise ValueError(
            "透明 ROI 输出只支持 8 位图像；"
            "请先显式选择数值映射并转换为 8 位，或使用数值填充。"
        )
    if source.ndim == 2:
        rgb = np.repeat(source[..., np.newaxis], 3, axis=2)
        alpha = np.where(mask, 255, 0).astype(np.uint8)
    elif source.shape[2] == 3:
        rgb = source
        alpha = np.where(mask, 255, 0).astype(np.uint8)
    else:
        rgb = source[..., :3]
        alpha = np.where(mask, source[..., 3], 0).astype(np.uint8)
    return np.ascontiguousarray(np.dstack((rgb, alpha)), dtype=np.uint8)


def _require_scalar_image(
    image: NDArray[Any],
    channel: ParameterValue = None,
) -> NDArray[Any]:
    source = _validate_raster(image)
    if source.ndim == 2:
        return source
    if source.shape[2] == 1:
        return source[..., 0]
    if channel is None:
        raise ValueError("彩色图像执行此操作前必须显式选择 RGB 通道或亮度。")
    return _select_scalar_channel(source, str(channel))


def _roi_source_for_output(
    source: NDArray[Any],
    output: NDArray[Any],
    *,
    channel: ParameterValue = None,
) -> NDArray[Any]:
    if source.shape == output.shape:
        return source
    scalar = _require_scalar_image(source, channel)
    if scalar.shape != output.shape:
        raise ValueError("ROI 操作的输出尺寸与源图像不一致。")
    return scalar


def _binary_mask(
    image: NDArray[Any],
    *,
    foreground_is_high: bool,
) -> NDArray[np.bool_]:
    source = _validate_raster(image)
    if source.ndim != 2:
        raise ValueError("二值处理需要单通道图像。")
    finite = np.isfinite(source)
    if not np.any(finite):
        return np.zeros(source.shape, dtype=bool)
    threshold = (_finite_min(source) + _finite_max(source)) / 2.0
    mask = finite & (source > threshold)
    return mask if foreground_is_high else finite & ~mask


def _binary_values(dtype: np.dtype[Any]) -> tuple[float, float]:
    resolved = np.dtype(dtype)
    if resolved.kind in {"u", "i"}:
        low, high = _dtype_range(resolved)
        return low, high
    return 0.0, 1.0


def _mask_to_binary_values(
    mask: NDArray[np.bool_],
    dtype: np.dtype[Any],
    *,
    foreground_is_high: bool,
) -> NDArray[Any]:
    low, high = _binary_values(dtype)
    foreground_value, background_value = (
        (high, low) if foreground_is_high else (low, high)
    )
    return np.where(mask, foreground_value, background_value).astype(dtype)


def _validate_connectivity(connectivity: int) -> None:
    if connectivity not in {4, 8}:
        raise ValueError("连通性必须为 4 或 8。")


def _validate_range(low: float, high: float, name: str) -> None:
    _require_finite(f"{name}下限", low)
    _require_finite(f"{name}上限", high)
    if high <= low:
        raise ValueError(f"{name}的上限必须大于下限。")


def _reject_roi_for_geometry(request: ImageOperationRequest) -> None:
    if request.roi_mask is not None:
        raise ValueError("几何变换不接受像素 ROI 掩膜。")


def _dtype_range(dtype: np.dtype[Any]) -> tuple[float, float]:
    resolved = np.dtype(dtype)
    if resolved.kind in {"u", "i"}:
        info = np.iinfo(resolved)
        return float(info.min), float(info.max)
    if resolved.kind == "f":
        return 0.0, 1.0
    raise TypeError(f"不支持的栅格位深：{resolved}")


def _working_range(image: NDArray[Any]) -> tuple[float, float]:
    if image.dtype.kind in {"u", "i"}:
        return _dtype_range(image.dtype)
    return 0.0, 1.0


def _restore_dtype(values: NDArray[Any], dtype: np.dtype[Any]) -> NDArray[Any]:
    target = np.dtype(dtype)
    work = np.asarray(values)
    if target.kind in {"u", "i"}:
        low, high = _dtype_range(target)
        work = np.nan_to_num(work, nan=low, posinf=high, neginf=low)
        return np.rint(np.clip(work, low, high)).astype(target)
    return work.astype(target)


def _cast_like(values: NDArray[Any], dtype: np.dtype[Any]) -> NDArray[Any]:
    return _restore_dtype(values, np.dtype(dtype))


def _normalize_fill_value(
    image: NDArray[Any],
    value: ParameterValue,
    *,
    field_name: str,
) -> float | tuple[float, ...]:
    channels = 1 if image.ndim == 2 else int(image.shape[2])
    raw_values = value if isinstance(value, tuple) else (value,)
    if len(raw_values) not in {1, channels}:
        raise ValueError(f"{field_name}必须是一个数值或与图像通道数一致的数值序列。")
    normalized: list[float] = []
    low, high = _dtype_range(image.dtype)
    for raw in raw_values:
        if isinstance(raw, bool) or raw is None:
            raise TypeError(f"{field_name}必须是有限数值。")
        try:
            numeric = float(raw)
        except (TypeError, ValueError, OverflowError) as error:
            raise TypeError(f"{field_name}必须是有限数值。") from error
        _require_finite(field_name, numeric)
        if image.dtype.kind in {"u", "i"} and not low <= numeric <= high:
            raise ValueError(f"{field_name}必须位于 {low:g}–{high:g} 范围内。")
        normalized.append(numeric)
    if len(normalized) == 1:
        return normalized[0]
    return tuple(normalized)


def _positive_integer_parameter(value: ParameterValue, *, field_name: str) -> int:
    if isinstance(value, bool) or isinstance(value, tuple) or value is None:
        raise TypeError(f"{field_name}必须为正整数。")
    try:
        normalized = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise TypeError(f"{field_name}必须为正整数。") from error
    if normalized != value or normalized <= 0:
        raise ValueError(f"{field_name}必须为正整数。")
    return normalized


def _finite_min(image: NDArray[Any]) -> float:
    finite = np.asarray(image)[np.isfinite(image)]
    return float(np.min(finite)) if finite.size else 0.0


def _finite_max(image: NDArray[Any]) -> float:
    finite = np.asarray(image)[np.isfinite(image)]
    return float(np.max(finite)) if finite.size else 0.0


def _require_finite(name: str, value: float) -> None:
    if not math.isfinite(float(value)):
        raise ValueError(f"{name}必须是有限数。")


def _require_positive(name: str, value: float) -> None:
    _require_finite(name, value)
    if float(value) <= 0:
        raise ValueError(f"{name}必须为正数。")


def _select_scalar_channel(image: NDArray[Any], channel: str) -> NDArray[Any]:
    if image.ndim == 2:
        return image
    if image.shape[2] == 1:
        return image[..., 0]
    resolved = str(channel).strip().lower()
    indices = {"red": 0, "r": 0, "green": 1, "g": 1, "blue": 2, "b": 2}
    if resolved in indices:
        return image[..., indices[resolved]]
    if resolved in {"luminance", "gray", "grayscale"}:
        rgb = image[..., :3].astype(np.float64)
        return (rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722).astype(
            np.float32
        )
    raise ValueError(f"不支持的标量通道：{channel}")


def _split_color_channels(
    image: NDArray[Any],
) -> tuple[list[NDArray[Any]], NDArray[Any] | None]:
    if image.ndim == 2:
        return [image], None
    if image.shape[2] == 1:
        return [image[..., 0]], None
    alpha = image[..., 3] if image.shape[2] == 4 else None
    return [image[..., index] for index in range(3)], alpha


def _apply_color_channels(image: NDArray[Any], function) -> NDArray[Any]:
    planes, alpha = _split_color_channels(image)
    results = [function(plane) for plane in planes]
    if image.ndim == 2:
        return results[0]
    if image.shape[2] == 1:
        return results[0][..., np.newaxis]
    if alpha is not None:
        results.append(alpha.copy())
    return np.stack(results, axis=2)


def _median_blur_reflect(
    plane: NDArray[Any],
    radius: int,
) -> NDArray[Any]:
    padded = cv2.copyMakeBorder(
        plane,
        radius,
        radius,
        radius,
        radius,
        cv2.BORDER_REFLECT_101,
    )
    filtered = cv2.medianBlur(padded, radius * 2 + 1)
    return filtered[radius:-radius, radius:-radius]


def _cv_border(mode: BorderMode) -> int:
    return {
        BorderMode.REFLECT: cv2.BORDER_REFLECT_101,
        BorderMode.REPLICATE: cv2.BORDER_REPLICATE,
        BorderMode.CONSTANT: cv2.BORDER_CONSTANT,
        BorderMode.WRAP: cv2.BORDER_WRAP,
    }[mode]


def _cv_interpolation(mode: InterpolationMode) -> int:
    if mode is InterpolationMode.AUTO:
        raise ValueError("auto 插值必须在进入 OpenCV 前解析为具体算法。")
    return {
        InterpolationMode.NEAREST: cv2.INTER_NEAREST,
        InterpolationMode.LINEAR: cv2.INTER_LINEAR,
        InterpolationMode.CUBIC: cv2.INTER_CUBIC,
        InterpolationMode.AREA: cv2.INTER_AREA,
        InterpolationMode.LANCZOS: cv2.INTER_LANCZOS4,
    }[mode]


@dataclass(frozen=True, slots=True)
class RecipeValidationStep:
    """One successfully resolved link in a recipe type chain."""

    index: int
    operation: ImageOperationSpec
    descriptor: ImageOperationDescriptor
    input_state: RasterTypeState
    output_state: RasterTypeState
    capability: ImageOperationCapability


@dataclass(frozen=True, slots=True)
class RecipeValidationResult:
    """Validated operation sequence without executing or allocating pixels."""

    input_state: RasterTypeState
    output_state: RasterTypeState
    steps: tuple[RecipeValidationStep, ...]


class ImageRecipeValidationError(ValueError):
    """A recipe error with a stable operation index and operation id."""

    def __init__(
        self,
        message: str,
        *,
        operation_index: int | None = None,
        operation_id: str | None = None,
    ) -> None:
        self.operation_index = operation_index
        self.operation_id = operation_id
        prefix = ""
        if operation_index is not None:
            prefix = f"配方步骤 {operation_index + 1}"
            if operation_id:
                prefix += f"（{operation_id}）"
            prefix += "："
        super().__init__(prefix + str(message))


def raster_type_state_from_array(image: NDArray[Any]) -> RasterTypeState:
    """Describe an execution raster using the persisted type vocabulary."""

    source = _validate_raster(image)
    if source.dtype == np.dtype(np.uint8):
        pixel_type = {
            1: RasterPixelType.GRAY8,
            3: RasterPixelType.RGB8,
            4: RasterPixelType.RGBA8,
        }[1 if source.ndim == 2 else int(source.shape[2])]
    elif source.dtype == np.dtype(np.uint16) and source.ndim == 2:
        pixel_type = RasterPixelType.GRAY16
    elif source.dtype == np.dtype(np.float32) and source.ndim == 2:
        pixel_type = RasterPixelType.GRAY32_FLOAT
    else:  # pragma: no cover - guarded by _validate_raster
        raise ValueError("无法描述当前栅格类型")
    return RasterTypeState(
        pixel_type=pixel_type,
        width=int(source.shape[1]),
        height=int(source.shape[0]),
    )


def image_operation_registry() -> Mapping[str, ImageOperationDescriptor]:
    """Return the immutable registry for every executable operation."""

    return IMAGE_OPERATION_REGISTRY


def get_image_operation_descriptor(
    operation: ImageOperation | str,
) -> ImageOperationDescriptor:
    operation_id = (
        operation.value if isinstance(operation, ImageOperation) else str(operation)
    )
    try:
        return IMAGE_OPERATION_REGISTRY[operation_id]
    except KeyError as exc:
        raise ImageRecipeValidationError(
            f"未注册的图像操作：{operation_id}",
            operation_id=operation_id,
        ) from exc


def validate_image_operation_spec(
    operation: ImageOperationSpec,
    input_state: RasterTypeState,
    *,
    roi_requested: bool = False,
    secondary_state: RasterTypeState | None = None,
) -> RecipeValidationStep:
    """Validate one persisted operation and resolve its output type."""

    if not isinstance(operation, ImageOperationSpec):
        raise TypeError("operation 必须是 ImageOperationSpec")
    if not isinstance(input_state, RasterTypeState):
        raise TypeError("input_state 必须是 RasterTypeState")
    descriptor = get_image_operation_descriptor(operation.operation_id)
    if operation.implementation != "fdm":
        raise ImageRecipeValidationError(
            "不支持的图像处理实现："
            f"{operation.implementation}；当前版本只允许重放 fdm 实现",
            operation_id=operation.operation_id,
        )
    supported_versions = {"1", descriptor.version}
    if operation.implementation_version not in supported_versions:
        raise ImageRecipeValidationError(
            "不支持的算法版本："
            f"{operation.operation_id} v{operation.implementation_version}；"
            "当前版本仅支持 "
            + "、".join(
                f"v{version}"
                for version in sorted(supported_versions, key=int)
            ),
            operation_id=operation.operation_id,
        )
    resolved_operation = ImageOperation(operation.operation_id)
    parameters = operation.parameters
    unknown_parameters = sorted(set(parameters) - set(descriptor.parameters))
    if unknown_parameters:
        raise ImageRecipeValidationError(
            "包含未声明参数：" + "、".join(unknown_parameters),
            operation_id=operation.operation_id,
        )
    if roi_requested and not descriptor.roi_semantics.supports_roi:
        raise ImageRecipeValidationError(
            "该操作不支持 ROI",
            operation_id=operation.operation_id,
        )
    if operation.operation_id in {
        ImageOperation.CROP.value,
        ImageOperation.COPY.value,
    }:
        roi_mode = str(parameters.get("roi_mode", "bounds")).strip().lower()
        if roi_mode not in {"bounds", "mask"}:
            raise ImageRecipeValidationError(
                "ROI 模式必须为 bounds 或 mask",
                operation_id=operation.operation_id,
            )
        if (
            not roi_requested
            and (
                roi_mode == "mask"
                or bool(parameters.get("transparent_outside", False))
            )
        ):
            raise ImageRecipeValidationError(
                "mask/transparent_outside 裁剪需要 ROI",
                operation_id=operation.operation_id,
            )
        if (
            bool(parameters.get("transparent_outside", False))
            and input_state.pixel_type
            not in {
                RasterPixelType.GRAY8,
                RasterPixelType.RGB8,
                RasterPixelType.RGBA8,
            }
        ):
            raise ImageRecipeValidationError(
                "透明 ROI 输出只支持 8 位图像；"
                "16 位或浮点图像请先显式选择数值映射并转换为 8 位，"
                "或改用数值填充",
                operation_id=operation.operation_id,
            )
    if (
        resolved_operation in _VERSIONED_STRICT_BINARY_OPERATIONS
        and operation.implementation_version != "1"
        and input_state.semantic is not RasterSemantic.BINARY_MASK
    ):
        raise ImageRecipeValidationError(
            "该操作要求显式二值掩膜输入；"
            "请先添加“二值化”“自动阈值”或“局部自适应阈值”。"
            "旧版 v1 配方仍可按原算法只读重放",
            operation_id=operation.operation_id,
        )
    if (
        resolved_operation in _VERSIONED_EXPLICIT_FLOAT_RANGE_OPERATIONS
        and operation.implementation_version != "1"
        and input_state.pixel_type is RasterPixelType.GRAY32_FLOAT
    ):
        raise ImageRecipeValidationError(
            "32 位浮点图像没有可安全推断的 0–1 工作范围；"
            "请先用“色阶”或“归一化”显式限定输入/输出范围，"
            "并转换为 8 位或 16 位后再执行该操作。"
            "旧版 v1 配方仍可按原 0–1 假设只读重放",
            operation_id=operation.operation_id,
        )
    if (
        resolved_operation in _VERSIONED_NEAREST_GEOMETRY_OPERATIONS
        and operation.implementation_version != "1"
        and input_state.semantic
        in {
            RasterSemantic.BINARY_MASK,
            RasterSemantic.LABELS,
        }
    ):
        interpolation = _coerce_enum(
            InterpolationMode,
            parameters.get(
                "interpolation",
                (
                    InterpolationMode.AUTO.value
                    if resolved_operation is ImageOperation.RESIZE
                    else InterpolationMode.LINEAR.value
                ),
            ),
            "插值模式",
        )
        if (
            interpolation is not InterpolationMode.NEAREST
            and not (
                resolved_operation is ImageOperation.RESIZE
                and interpolation is InterpolationMode.AUTO
            )
        ):
            raise ImageRecipeValidationError(
                "二值图和标签图只允许最近邻插值；"
                "其他插值会产生不存在的中间类别值",
                operation_id=operation.operation_id,
            )
    flat_field_source = str(
        parameters.get("flat_field_source", "estimated")
    ).strip().lower()
    needs_secondary_state = (
        operation.operation_id == ImageOperation.IMAGE_CALCULATOR.value
        or (
            operation.operation_id
            == ImageOperation.FLAT_FIELD_CORRECTION.value
            and flat_field_source == "reference"
        )
    )
    if needs_secondary_state:
        if secondary_state is None:
            raise ImageRecipeValidationError(
                (
                    "参考图平场校正缺少第二幅参考图像的类型状态"
                    if operation.operation_id
                    == ImageOperation.FLAT_FIELD_CORRECTION.value
                    else "图像计算器缺少第二幅图像的类型状态"
                ),
                operation_id=operation.operation_id,
            )
        if (
            secondary_state.pixel_type is not input_state.pixel_type
            or (
                input_state.width is not None
                and secondary_state.width is not None
                and (
                    secondary_state.width != input_state.width
                    or secondary_state.height != input_state.height
                )
            )
        ):
            raise ImageRecipeValidationError(
                (
                    "参考平场与源图像的尺寸、通道和像素类型必须完全一致"
                    if operation.operation_id
                    == ImageOperation.FLAT_FIELD_CORRECTION.value
                    else "图像计算器要求两幅图像的尺寸和像素类型完全一致"
                ),
                operation_id=operation.operation_id,
            )
    try:
        output_state = descriptor.resolve_output(input_state, parameters)
        if (
            resolved_operation is ImageOperation.IMAGE_CALCULATOR
            and secondary_state is not None
        ):
            calculator_operation = str(
                parameters.get("calculator_operation", "add")
            ).strip().lower()
            if calculator_operation == "copy":
                # ``copy`` is the sole calculator operation whose pixels and
                # scientific meaning both come entirely from the right-hand
                # raster.  Pixel type/dimensions have already been required to
                # match above, so only its semantic needs to be transferred.
                output_state = output_state.replace(
                    semantic=secondary_state.semantic,
                )
            elif calculator_operation in {"and", "or", "xor"}:
                # Integer bitwise arithmetic is still a valid intensity
                # operation.  It is a binary-mask operation only when both
                # operands explicitly carry that contract; otherwise allowing
                # a following strict morphology step would silently reinterpret
                # arbitrary sample bits as foreground/background.
                output_state = output_state.replace(
                    semantic=(
                        RasterSemantic.COLOR
                        if not output_state.is_grayscale
                        else (
                            RasterSemantic.BINARY_MASK
                            if (
                                input_state.semantic
                                is RasterSemantic.BINARY_MASK
                                and secondary_state.semantic
                                is RasterSemantic.BINARY_MASK
                            )
                            else RasterSemantic.INTENSITY
                        )
                    ),
                )
            else:
                # Add/subtract/multiply/etc. produce a new quantitative raster;
                # even min/max/mean of masks are not authoritative masks.
                output_state = output_state.replace(
                    semantic=(
                        RasterSemantic.INTENSITY
                        if output_state.is_grayscale
                        else RasterSemantic.COLOR
                    ),
                )
        if (
            operation.operation_id == ImageOperation.COPY.value
            and not roi_requested
        ):
            output_state = input_state
        if (
            roi_requested
            and output_state.is_grayscale
            and output_state.semantic is not input_state.semantic
        ):
            # ROI operations are blended back into the original source.  When
            # the inside and outside carry different meanings (binary, label,
            # distance or intensity), the complete raster no longer satisfies
            # either specialized global contract.
            output_state = output_state.replace(
                semantic=RasterSemantic.INTENSITY,
            )
        capability = descriptor.tile(parameters)
    except (TypeError, ValueError) as exc:
        raise ImageRecipeValidationError(
            str(exc),
            operation_id=operation.operation_id,
        ) from exc
    if not isinstance(capability, ImageOperationCapability):
        raise TypeError(
            f"操作 {operation.operation_id} 的 tile 解析器返回类型无效"
        )
    return RecipeValidationStep(
        index=0,
        operation=operation,
        descriptor=descriptor,
        input_state=input_state,
        output_state=output_state,
        capability=capability,
    )


def validate_image_processing_recipe(
    recipe: ImageProcessingRecipe,
    input_state: RasterTypeState,
    *,
    roi_requested: bool = False,
    secondary_states: Mapping[str, RasterTypeState] | None = None,
) -> RecipeValidationResult:
    """Validate a persisted recipe's complete type chain without pixels."""

    if not isinstance(recipe, ImageProcessingRecipe):
        raise TypeError("recipe 必须是 ImageProcessingRecipe")
    if not isinstance(input_state, RasterTypeState):
        raise TypeError("input_state 必须是 RasterTypeState")
    resolved_secondary_states = dict(secondary_states or {})
    state = input_state
    steps: list[RecipeValidationStep] = []
    spatial_alignment_changed = False
    for index, operation in enumerate(recipe.operations):
        secondary_state = None
        flat_field_source = str(
            operation.parameters.get("flat_field_source", "estimated")
        ).strip().lower()
        if (
            operation.operation_id == ImageOperation.IMAGE_CALCULATOR.value
            or (
                operation.operation_id
                == ImageOperation.FLAT_FIELD_CORRECTION.value
                and flat_field_source == "reference"
            )
        ):
            secondary_id = str(
                operation.parameters.get("secondary_document_id", "")
            ).strip()
            secondary_state = resolved_secondary_states.get(secondary_id)
            if (
                spatial_alignment_changed
                and operation.implementation_version != "1"
            ):
                raise ImageRecipeValidationError(
                    "配方前序已执行空间几何变换，第二幅图像尚无可审计的"
                    "同变换或重新对齐记录；请先生成派生图片并重新选择"
                    "与其对齐的第二幅图像。旧版 v1 配方仍可只读重放",
                    operation_index=index,
                    operation_id=operation.operation_id,
                )
        try:
            step = validate_image_operation_spec(
                operation,
                state,
                roi_requested=roi_requested,
                secondary_state=secondary_state,
            )
        except ImageRecipeValidationError as exc:
            raise ImageRecipeValidationError(
                str(exc),
                operation_index=index,
                operation_id=operation.operation_id,
            ) from exc
        step = RecipeValidationStep(
            index=index,
            operation=step.operation,
            descriptor=step.descriptor,
            input_state=step.input_state,
            output_state=step.output_state,
            capability=step.capability,
        )
        steps.append(step)
        state = step.output_state
        if (
            ImageOperation(operation.operation_id)
            in _SPATIAL_ALIGNMENT_CHANGING_OPERATIONS
        ):
            spatial_alignment_changed = True
    return RecipeValidationResult(
        input_state=input_state,
        output_state=state,
        steps=tuple(steps),
    )


def _same_output(
    state: RasterTypeState,
    _parameters: Mapping[str, object],
) -> RasterTypeState:
    return state


def _gray_type(pixel_type: RasterPixelType) -> RasterPixelType:
    if pixel_type is RasterPixelType.GRAY16:
        return RasterPixelType.GRAY16
    if pixel_type is RasterPixelType.GRAY32_FLOAT:
        return RasterPixelType.GRAY32_FLOAT
    return RasterPixelType.GRAY8


def _scalar_output(
    state: RasterTypeState,
    _parameters: Mapping[str, object],
) -> RasterTypeState:
    return state.replace(
        pixel_type=_gray_type(state.pixel_type),
        semantic=(
            RasterSemantic.BINARY_MASK
            if state.semantic is RasterSemantic.BINARY_MASK
            else RasterSemantic.INTENSITY
        ),
    )


def _binary_output(
    state: RasterTypeState,
    _parameters: Mapping[str, object],
) -> RasterTypeState:
    return state.replace(
        pixel_type=_gray_type(state.pixel_type),
        semantic=RasterSemantic.BINARY_MASK,
    )


def _float_scalar_output(
    state: RasterTypeState,
    _parameters: Mapping[str, object],
) -> RasterTypeState:
    return state.replace(
        pixel_type=RasterPixelType.GRAY32_FLOAT,
        semantic=RasterSemantic.DISTANCE,
    )


def _convert_type_output(
    state: RasterTypeState,
    parameters: Mapping[str, object],
) -> RasterTypeState:
    target = _coerce_enum(
        PixelType,
        parameters.get("target_type", PixelType.UINT8.value),
        "目标位深",
    )
    if state.channel_count > 1 and target is not PixelType.UINT8:
        raise ValueError(
            "RGB/RGBA 图像不能直接转换为 16 位或 32 位浮点；"
            "请先转换为灰度：先添加“转换颜色模型 → 灰度”步骤"
        )
    pixel_type = {
        PixelType.UINT8: (
            state.pixel_type
            if state.channel_count > 1
            else RasterPixelType.GRAY8
        ),
        PixelType.UINT16: RasterPixelType.GRAY16,
        PixelType.FLOAT32: RasterPixelType.GRAY32_FLOAT,
    }[target]
    return state.replace(pixel_type=pixel_type)


def _convert_color_output(
    state: RasterTypeState,
    parameters: Mapping[str, object],
) -> RasterTypeState:
    target = _coerce_enum(
        ColorTarget,
        parameters.get("target_model", ColorTarget.GRAYSCALE.value),
        "颜色模型",
    )
    drop_alpha = bool(parameters.get("drop_alpha", False))
    if state.pixel_type is RasterPixelType.RGBA8 and not drop_alpha:
        raise ValueError("RGBA 转换会移除 Alpha；必须显式启用 drop_alpha")
    if target is ColorTarget.RGB:
        if state.pixel_type in {
            RasterPixelType.GRAY16,
            RasterPixelType.GRAY32_FLOAT,
        }:
            raise ValueError("RGB 权威像素只支持 8 位")
        return state.replace(
            pixel_type=RasterPixelType.RGB8,
            semantic=RasterSemantic.COLOR,
        )
    return state.replace(
        pixel_type=_gray_type(state.pixel_type),
        semantic=RasterSemantic.INTENSITY,
    )


def _color_input(
    state: RasterTypeState,
    _parameters: Mapping[str, object],
) -> str | None:
    if state.pixel_type not in {RasterPixelType.RGB8, RasterPixelType.RGBA8}:
        return "该操作只适用于 RGB 或 RGBA 图像"
    return None


def _integer_input(
    state: RasterTypeState,
    _parameters: Mapping[str, object],
) -> str | None:
    if state.pixel_type is RasterPixelType.GRAY32_FLOAT:
        return "该操作仅支持 8 位或 16 位整数图像"
    return None


def _single_channel_input(
    state: RasterTypeState,
    _parameters: Mapping[str, object],
) -> str | None:
    if not state.is_grayscale:
        return "该操作需要单通道图像"
    return None


def _scalar_channel_input(
    state: RasterTypeState,
    parameters: Mapping[str, object],
) -> str | None:
    if not state.is_grayscale and not str(parameters.get("channel", "")).strip():
        return "彩色图像必须显式选择一个标量通道"
    return None


def _authoritative_float_output_input(
    state: RasterTypeState,
    parameters: Mapping[str, object],
) -> str | None:
    if state.is_grayscale:
        return None
    operation_mode = str(
        parameters.get(
            "result_mode",
            parameters.get("method", ""),
        )
    ).strip().lower()
    if operation_mode in {"float32", "variance", "laplacian"}:
        return "float32 结果仅支持单通道权威栅格"
    return None


def _canny_input(
    state: RasterTypeState,
    parameters: Mapping[str, object],
) -> str | None:
    if state.pixel_type not in {
        RasterPixelType.GRAY8,
        RasterPixelType.RGB8,
        RasterPixelType.RGBA8,
    }:
        return "Canny 边缘检测仅支持 8 位图像；请先显式转换位深"
    return _scalar_channel_input(state, parameters)


def _repair_nonfinite_input(
    state: RasterTypeState,
    _parameters: Mapping[str, object],
) -> str | None:
    if state.pixel_type is not RasterPixelType.GRAY32_FLOAT:
        return "NaN/Inf 修复仅适用于 32 位浮点单通道图像"
    return None


def _pixel_bin_input(
    state: RasterTypeState,
    parameters: Mapping[str, object],
) -> str | None:
    method = str(
        parameters.get("method", PixelBinMethod.MEAN.value)
    ).strip().lower()
    if method == PixelBinMethod.SUM.value and not state.is_grayscale:
        return (
            "RGB/RGBA 像素合并不支持求和；"
            "请先显式转换为单通道图像，或选择均值、最小值、最大值"
        )
    return None


def _median_filter_input(
    state: RasterTypeState,
    parameters: Mapping[str, object],
) -> str | None:
    radius = int(parameters.get("radius", 1))
    if state.pixel_type is not RasterPixelType.GRAY8 and radius > 2:
        return "16 位或浮点图像的中值滤波半径最大为 2"
    return None


def _remove_outliers_input(
    state: RasterTypeState,
    parameters: Mapping[str, object],
) -> str | None:
    radius = int(parameters.get("radius", 1))
    if state.pixel_type is not RasterPixelType.GRAY8 and radius > 2:
        return "16 位或浮点图像的热点/坏点剔除半径最大为 2"
    return None


def _image_calculator_input(
    state: RasterTypeState,
    parameters: Mapping[str, object],
) -> str | None:
    operation = str(
        parameters.get("calculator_operation", "add")
    ).strip().lower()
    if (
        operation in {"and", "or", "xor"}
        and state.pixel_type is RasterPixelType.GRAY32_FLOAT
    ):
        return "AND、OR、XOR 仅支持 8 位或 16 位整数图像"
    return _authoritative_float_output_input(state, parameters)


def _geometry_output(
    operation: ImageOperation,
    state: RasterTypeState,
    parameters: Mapping[str, object],
) -> RasterTypeState:
    width = state.width
    height = state.height
    if operation in {
        ImageOperation.ROTATE_90_CLOCKWISE,
        ImageOperation.ROTATE_90_COUNTERCLOCKWISE,
    }:
        width, height = height, width
    elif operation in {
        ImageOperation.CROP,
        ImageOperation.RESIZE,
        ImageOperation.RESIZE_CANVAS,
    }:
        if "width" in parameters:
            width = _positive_integer_parameter(
                parameters["width"],
                field_name="输出宽度",
            )
        if "height" in parameters:
            height = _positive_integer_parameter(
                parameters["height"],
                field_name="输出高度",
            )
        if (
            operation is ImageOperation.CROP
            and bool(parameters.get("transparent_outside", False))
        ):
            return state.replace(
                pixel_type=RasterPixelType.RGBA8,
                semantic=RasterSemantic.COLOR,
                width=width,
                height=height,
            )
    elif operation is ImageOperation.PIXEL_BIN:
        factor = _positive_integer_parameter(
            parameters.get("factor", 2),
            field_name="像素合并系数",
        )
        if width is not None and height is not None:
            remainder = str(
                parameters.get(
                    "remainder_policy",
                    PixelBinRemainderPolicy.REJECT.value,
                )
            )
            if remainder == PixelBinRemainderPolicy.REJECT.value and (
                width % factor or height % factor
            ):
                raise ValueError("图片宽高不能被像素合并系数整除")
            width = width // factor
            height = height // factor
    semantic = state.semantic
    pixel_type = state.pixel_type
    if operation in {ImageOperation.ROTATE, ImageOperation.TRANSLATE}:
        interpolation = _coerce_enum(
            InterpolationMode,
            parameters.get(
                "interpolation",
                InterpolationMode.LINEAR.value,
            ),
            "插值模式",
        )
        if interpolation is InterpolationMode.AUTO:
            raise ValueError(
                "任意角度旋转和平移不支持 auto 插值；"
                "请显式选择 nearest、linear、cubic、area 或 lanczos"
            )
        if (
            semantic in {
                RasterSemantic.BINARY_MASK,
                RasterSemantic.LABELS,
            }
            and interpolation is not InterpolationMode.NEAREST
        ):
            semantic = RasterSemantic.INTENSITY
    elif operation is ImageOperation.RESIZE:
        interpolation = _coerce_enum(
            InterpolationMode,
            parameters.get(
                "interpolation",
                InterpolationMode.AUTO.value,
            ),
            "插值模式",
        )
        if semantic in {
            RasterSemantic.BINARY_MASK,
            RasterSemantic.LABELS,
        }:
            resolved_interpolation = (
                InterpolationMode.NEAREST
                if interpolation is InterpolationMode.AUTO
                else interpolation
            )
            if resolved_interpolation is not InterpolationMode.NEAREST:
                semantic = RasterSemantic.INTENSITY
    elif operation is ImageOperation.PIXEL_BIN:
        method = _coerce_enum(
            PixelBinMethod,
            parameters.get("method", PixelBinMethod.MEAN.value),
            "像素合并方式",
        )
        if method is PixelBinMethod.SUM:
            pixel_type = RasterPixelType.GRAY32_FLOAT
            semantic = RasterSemantic.INTENSITY
        elif (
            method is PixelBinMethod.MEAN
            and semantic
            in {
                RasterSemantic.BINARY_MASK,
                RasterSemantic.LABELS,
            }
        ):
            # Averaging invents intermediate samples that are neither binary
            # classes nor authoritative label identifiers.
            semantic = RasterSemantic.INTENSITY
    return state.replace(
        pixel_type=pixel_type,
        semantic=semantic,
        width=width,
        height=height,
    )


def _fft_output(
    state: RasterTypeState,
    parameters: Mapping[str, object],
) -> RasterTypeState:
    if bool(parameters.get("output_float", False)):
        if (
            state.channel_count > 1
            and str(parameters.get("channel", "per_channel")) == "per_channel"
        ):
            # float32 RGB/RGBA cannot cross the authoritative raster boundary.
            raise ValueError("逐颜色通道的浮点 FFT 输出不能保存为权威栅格")
        return state.replace(
            pixel_type=RasterPixelType.GRAY32_FLOAT,
            semantic=RasterSemantic.INTENSITY,
        )
    if str(parameters.get("channel", "per_channel")) == "per_channel":
        return state
    return _scalar_output(state, parameters)


_PARAMETERS: dict[ImageOperation, tuple[str, ...]] = {
    ImageOperation.COPY: (
        "roi_mode", "fill_value", "outside_value", "transparent_outside",
    ),
    ImageOperation.CONVERT_TYPE: (
        "target_type", "scale_mode", "nonfinite_policy",
    ),
    ImageOperation.CONVERT_COLOR: (
        "target_model", "grayscale_method", "drop_alpha",
    ),
    ImageOperation.COLOR_BALANCE: (
        "red_gain", "green_gain", "blue_gain",
        "red_offset", "green_offset", "blue_offset",
    ),
    ImageOperation.BRIGHTNESS_CONTRAST: ("brightness", "contrast", "gamma"),
    ImageOperation.ADJUST_LEVELS: (
        "black_point", "white_point", "output_min", "output_max", "gamma",
    ),
    ImageOperation.THRESHOLD: (
        "lower", "upper", "invert", "foreground_value",
        "background_value", "channel",
    ),
    ImageOperation.ROTATE: (
        "angle_degrees", "expand", "interpolation", "border_mode", "border_value",
    ),
    ImageOperation.CROP: (
        "x", "y", "width", "height", "roi_mode",
        "fill_value", "outside_value", "transparent_outside",
    ),
    ImageOperation.RESIZE: ("width", "height", "interpolation"),
    ImageOperation.TRANSLATE: (
        "offset_x", "offset_y", "interpolation", "border_mode", "border_value",
    ),
    ImageOperation.RESIZE_CANVAS: ("width", "height", "anchor", "fill_value"),
    ImageOperation.PIXEL_BIN: ("factor", "method", "remainder_policy"),
    ImageOperation.GAUSSIAN_BLUR: ("sigma", "sigma_x", "sigma_y", "border_mode"),
    ImageOperation.MEDIAN_FILTER: ("radius",),
    ImageOperation.MEAN_FILTER: ("radius", "border_mode"),
    ImageOperation.BILATERAL_FILTER: (
        "diameter", "sigma_color", "sigma_space", "border_mode",
    ),
    ImageOperation.UNSHARP_MASK: ("sigma", "amount", "threshold"),
    ImageOperation.SOBEL_EDGES: ("kernel_size", "channel", "output_float"),
    ImageOperation.LAPLACIAN_EDGES: ("kernel_size", "channel", "output_float"),
    ImageOperation.CANNY_EDGES: (
        "threshold_low", "threshold_high", "aperture_size",
        "l2_gradient", "channel",
    ),
    ImageOperation.NORMALIZE: ("output_min", "output_max", "per_channel"),
    ImageOperation.CLAHE: ("clip_limit", "tile_grid_size"),
    ImageOperation.REMOVE_OUTLIERS: ("radius", "threshold", "polarity"),
    ImageOperation.REPAIR_NONFINITE: ("radius", "fallback_value"),
    ImageOperation.AUTO_THRESHOLD: ("method", "invert", "channel"),
    ImageOperation.BINARIZE: ("threshold", "invert", "channel"),
    ImageOperation.ERODE: (
        "radius", "iterations", "kernel", "border_mode", "channel",
    ),
    ImageOperation.DILATE: (
        "radius", "iterations", "kernel", "border_mode", "channel",
    ),
    ImageOperation.MORPHOLOGY_OPEN: (
        "radius", "iterations", "kernel", "border_mode", "channel",
    ),
    ImageOperation.MORPHOLOGY_CLOSE: (
        "radius", "iterations", "kernel", "border_mode", "channel",
    ),
    ImageOperation.TOP_HAT: (
        "radius", "iterations", "kernel", "border_mode", "channel",
    ),
    ImageOperation.BLACK_HAT: (
        "radius", "iterations", "kernel", "border_mode", "channel",
    ),
    ImageOperation.FILL_HOLES: ("foreground_is_high", "channel"),
    ImageOperation.CONTOUR_EXTRACT: ("foreground_is_high", "channel"),
    ImageOperation.REMOVE_SMALL_OBJECTS: (
        "minimum_area", "connectivity", "foreground_is_high", "channel",
    ),
    ImageOperation.FILL_SMALL_HOLES: (
        "maximum_area", "connectivity", "foreground_is_high", "channel",
    ),
    ImageOperation.DISTANCE_TRANSFORM: (
        "foreground_is_high", "distance_type", "channel",
    ),
    ImageOperation.SKELETONIZE: ("foreground_is_high", "channel"),
    ImageOperation.WATERSHED: (
        "foreground_is_high", "seed_threshold", "channel",
    ),
    ImageOperation.WATERSHED_V2: (
        "foreground_is_high", "seed_threshold",
        "minimum_seed_area", "channel",
    ),
    ImageOperation.BACKGROUND_SUBTRACT: (
        "radius", "light_background", "preserve_offset", "border_mode",
    ),
    ImageOperation.ROLLING_BALL_BACKGROUND_SUBTRACT: (
        "radius", "ball_height", "light_background", "preserve_offset",
    ),
    ImageOperation.CUSTOM_CONVOLUTION: (
        "kernel", "kernel_width", "kernel_height",
        "normalize_kernel", "offset", "border_mode",
    ),
    ImageOperation.INVERT: ("minimum", "maximum"),
    ImageOperation.ADD: ("value",),
    ImageOperation.SUBTRACT: ("value",),
    ImageOperation.MULTIPLY: ("value",),
    ImageOperation.DIVIDE: ("value",),
    ImageOperation.GAMMA: ("gamma", "minimum", "maximum"),
    ImageOperation.CLAMP: ("minimum", "maximum"),
    ImageOperation.IMAGE_CALCULATOR: (
        "secondary_document_id", "calculator_operation", "result_mode",
    ),
    ImageOperation.FFT_FILTER: (
        "mode", "low_cutoff", "high_cutoff", "order", "channel",
        "output_float", "boundary", "tukey_alpha",
        "frequency_unit", "pixel_size",
    ),
    ImageOperation.FFT_POWER_SPECTRUM: (
        "channel", "logarithmic", "centered", "window", "tukey_alpha",
    ),
    ImageOperation.STRIPE_SUPPRESSION: (
        "direction", "notch_width", "protect_radius", "strength",
    ),
    ImageOperation.LOG_V2: (
        "result_mode", "output_min", "output_max",
    ),
    ImageOperation.EXP_V2: (
        "result_mode", "output_min", "output_max",
    ),
    ImageOperation.SQRT_V2: (
        "result_mode", "output_min", "output_max",
    ),
    ImageOperation.ADAPTIVE_THRESHOLD: (
        "method", "radius", "offset", "k", "r", "p", "q",
        "foreground_is_high", "channel",
    ),
    ImageOperation.PERCENTILE_SATURATION: (
        "lower_percentile", "upper_percentile", "per_channel",
    ),
    ImageOperation.RANK_FILTER: ("method", "radius"),
    ImageOperation.MORPHOLOGY_DERIVATIVE: (
        "method", "radius", "channel",
    ),
    ImageOperation.MORPHOLOGICAL_RECONSTRUCTION: (
        "method", "radius", "connectivity", "channel",
    ),
    ImageOperation.REGIONAL_EXTREMA: (
        "kind", "h", "connectivity", "channel",
    ),
    ImageOperation.CLEAR_BORDER: (
        "foreground_is_high", "connectivity", "channel",
    ),
    ImageOperation.FLAT_FIELD_CORRECTION: (
        "flat_field_source", "secondary_document_id", "secondary_sha256",
        "reference_levels", "radius", "method", "preserve_mean",
    ),
}


def _parameter_schema(
    key: str,
    kind: str,
    default: object = None,
    minimum: float | None = None,
    maximum: float | None = None,
    *,
    choices: tuple[object, ...] = (),
    required: bool = False,
    required_when: tuple[tuple[str, object], ...] = (),
) -> ImageOperationParameterSchema:
    return ImageOperationParameterSchema(
        key=key,
        kind=kind,
        default=default,
        minimum=minimum,
        maximum=maximum,
        choices=choices,
        required=required,
        required_when=required_when,
    )


_SCALAR_CHANNELS = ("luminance", "red", "green", "blue")
_FFT_CHANNELS = ("per_channel",) + _SCALAR_CHANNELS
_BORDER_MODES = ("reflect", "replicate", "constant", "wrap")
_LOCAL_BORDER_MODES = ("reflect", "replicate", "constant")
_INTERPOLATION_MODES = ("auto", "nearest", "linear", "cubic", "area", "lanczos")
_CONNECTIVITY_MODES = (4, 8)


_COMMON_PARAMETER_SCHEMAS: dict[str, ImageOperationParameterSchema] = {
    "roi_mode": _parameter_schema(
        "roi_mode", "choice", "bounds", choices=("bounds", "mask")
    ),
    "fill_value": _parameter_schema(
        "fill_value", "float", 0.0, -1e12, 1e12
    ),
    "outside_value": _parameter_schema(
        "outside_value", "float", 0.0, -1e12, 1e12
    ),
    "transparent_outside": _parameter_schema(
        "transparent_outside", "bool", False
    ),
    "target_type": _parameter_schema(
        "target_type",
        "choice",
        "uint8",
        choices=("uint8", "uint16", "float32"),
    ),
    "scale_mode": _parameter_schema(
        "scale_mode",
        "choice",
        "full_type_range",
        choices=("preserve_values", "full_type_range", "data_range"),
    ),
    "nonfinite_policy": _parameter_schema(
        "nonfinite_policy",
        "choice",
        "reject",
        choices=("reject", "zero", "range_bounds"),
    ),
    "target_model": _parameter_schema(
        "target_model", "choice", "grayscale", choices=("grayscale", "rgb")
    ),
    "grayscale_method": _parameter_schema(
        "grayscale_method",
        "choice",
        "rec601",
        choices=("rec601", "average"),
    ),
    "drop_alpha": _parameter_schema("drop_alpha", "bool", False),
    "red_gain": _parameter_schema("red_gain", "float", 1.0, 0, 100),
    "green_gain": _parameter_schema("green_gain", "float", 1.0, 0, 100),
    "blue_gain": _parameter_schema("blue_gain", "float", 1.0, 0, 100),
    "red_offset": _parameter_schema(
        "red_offset", "float", 0.0, -65535, 65535
    ),
    "green_offset": _parameter_schema(
        "green_offset", "float", 0.0, -65535, 65535
    ),
    "blue_offset": _parameter_schema(
        "blue_offset", "float", 0.0, -65535, 65535
    ),
    "brightness": _parameter_schema(
        "brightness", "float", 0.0, -65535, 65535
    ),
    "contrast": _parameter_schema("contrast", "float", 1.0, 0, 20),
    "gamma": _parameter_schema("gamma", "float", 1.0, 0.001, 100),
    "black_point": _parameter_schema(
        "black_point", "float", 0.0, -1e12, 1e12
    ),
    "white_point": _parameter_schema(
        "white_point", "float", "$working_max", -1e12, 1e12
    ),
    "lower": _parameter_schema("lower", "float", 0.0, -1e12, 1e12),
    "upper": _parameter_schema(
        "upper", "float", "$working_max", -1e12, 1e12
    ),
    "invert": _parameter_schema("invert", "bool", False),
    "foreground_value": _parameter_schema(
        "foreground_value", "float", 255.0, -1e12, 1e12
    ),
    "background_value": _parameter_schema(
        "background_value", "float", 0.0, -1e12, 1e12
    ),
    "channel": _parameter_schema(
        "channel", "choice", "luminance", choices=_SCALAR_CHANNELS
    ),
    "angle_degrees": _parameter_schema(
        "angle_degrees", "float", 0.0, -360, 360
    ),
    "expand": _parameter_schema("expand", "bool", True),
    "interpolation": _parameter_schema(
        "interpolation",
        "choice",
        "linear",
        choices=_INTERPOLATION_MODES,
    ),
    "border_mode": _parameter_schema(
        "border_mode", "choice", "reflect", choices=_BORDER_MODES
    ),
    "border_value": _parameter_schema(
        "border_value", "float", 0.0, -1e12, 1e12
    ),
    "x": _parameter_schema("x", "int", 0, 0, 1_000_000),
    "y": _parameter_schema("y", "int", 0, 0, 1_000_000),
    "width": _parameter_schema("width", "int", -1, 1, 1_000_000),
    "height": _parameter_schema("height", "int", -1, 1, 1_000_000),
    "offset_x": _parameter_schema(
        "offset_x", "float", 0.0, -1_000_000, 1_000_000
    ),
    "offset_y": _parameter_schema(
        "offset_y", "float", 0.0, -1_000_000, 1_000_000
    ),
    "anchor": _parameter_schema(
        "anchor",
        "choice",
        "center",
        choices=(
            "top_left", "top_center", "top_right",
            "center_left", "center", "center_right",
            "bottom_left", "bottom_center", "bottom_right",
        ),
    ),
    "factor": _parameter_schema("factor", "int", 2, 1, 4096),
    "method": _parameter_schema("method", "string", ""),
    "remainder_policy": _parameter_schema(
        "remainder_policy", "choice", "reject", choices=("reject", "crop")
    ),
    "sigma": _parameter_schema("sigma", "float", 1.0, 0.01, 100),
    "sigma_x": _parameter_schema("sigma_x", "float", 1.0, 0.01, 100),
    "sigma_y": _parameter_schema("sigma_y", "float", 1.0, 0.01, 100),
    "radius": _parameter_schema("radius", "int", 1, 1, 99),
    "diameter": _parameter_schema("diameter", "int", 5, 1, 99),
    "sigma_color": _parameter_schema(
        "sigma_color", "float", 25.0, 0.01, 1e6
    ),
    "sigma_space": _parameter_schema(
        "sigma_space", "float", 2.0, 0.01, 1e6
    ),
    "amount": _parameter_schema("amount", "float", 1.0, 0, 20),
    "threshold": _parameter_schema(
        "threshold", "float", 0.0, -1e12, 1e12
    ),
    "kernel_size": _parameter_schema(
        "kernel_size", "choice", 3, choices=(1, 3, 5, 7)
    ),
    "output_float": _parameter_schema("output_float", "bool", True),
    "threshold_low": _parameter_schema(
        "threshold_low", "float", 50.0, 0, 65535
    ),
    "threshold_high": _parameter_schema(
        "threshold_high", "float", 150.0, 0, 65535
    ),
    "aperture_size": _parameter_schema(
        "aperture_size", "choice", 3, choices=(3, 5, 7)
    ),
    "l2_gradient": _parameter_schema("l2_gradient", "bool", True),
    "output_min": _parameter_schema(
        "output_min", "float", "$working_min", -1e12, 1e12
    ),
    "output_max": _parameter_schema(
        "output_max", "float", "$working_max", -1e12, 1e12
    ),
    "per_channel": _parameter_schema("per_channel", "bool", True),
    "clip_limit": _parameter_schema(
        "clip_limit", "float", 2.0, 0.01, 1000
    ),
    "tile_grid_size": _parameter_schema(
        "tile_grid_size", "int", 8, 2, 64
    ),
    "polarity": _parameter_schema(
        "polarity",
        "choice",
        "both",
        choices=("both", "bright", "dark"),
    ),
    "fallback_value": _parameter_schema(
        "fallback_value", "float", 0.0, -1e12, 1e12
    ),
    "iterations": _parameter_schema("iterations", "int", 1, 1, 100),
    "kernel": _parameter_schema(
        "kernel",
        "choice",
        "ellipse",
        choices=("ellipse", "rectangle", "cross"),
    ),
    "foreground_is_high": _parameter_schema(
        "foreground_is_high", "bool", True
    ),
    "minimum_area": _parameter_schema(
        "minimum_area", "int", 10, 1, 2_147_483_647
    ),
    "maximum_area": _parameter_schema(
        "maximum_area", "int", 10, 1, 2_147_483_647
    ),
    "connectivity": _parameter_schema(
        "connectivity", "choice", 8, choices=_CONNECTIVITY_MODES
    ),
    "distance_type": _parameter_schema(
        "distance_type",
        "choice",
        "l2",
        choices=("l2", "l1", "chessboard"),
    ),
    "seed_threshold": _parameter_schema(
        "seed_threshold", "float", 0.45, 0.001, 0.999
    ),
    "minimum_seed_area": _parameter_schema(
        "minimum_seed_area", "int", 1, 1, 2_147_483_647
    ),
    "light_background": _parameter_schema(
        "light_background", "bool", False
    ),
    "preserve_offset": _parameter_schema(
        "preserve_offset", "bool", False
    ),
    "ball_height": _parameter_schema(
        "ball_height", "float", 255.0, 0.001, 1e12
    ),
    "kernel_width": _parameter_schema(
        "kernel_width", "int", 3, 1, 99
    ),
    "kernel_height": _parameter_schema(
        "kernel_height", "int", 3, 1, 99
    ),
    "normalize_kernel": _parameter_schema(
        "normalize_kernel", "bool", False
    ),
    "offset": _parameter_schema("offset", "float", 0.0, -1e12, 1e12),
    "minimum": _parameter_schema(
        "minimum", "float", "$working_min", -1e12, 1e12
    ),
    "maximum": _parameter_schema(
        "maximum", "float", "$working_max", -1e12, 1e12
    ),
    "value": _parameter_schema("value", "float", 0.0, -1e12, 1e12),
    "secondary_document_id": _parameter_schema(
        "secondary_document_id", "secondary_image", ""
    ),
    "secondary_sha256": _parameter_schema(
        "secondary_sha256", "string", ""
    ),
    "reference_levels": _parameter_schema(
        "reference_levels", "number_list", (1.0,)
    ),
    "calculator_operation": _parameter_schema(
        "calculator_operation",
        "choice",
        "add",
        choices=(
            "add", "subtract", "multiply", "divide", "difference",
            "minimum", "maximum", "mean", "and", "or", "xor", "copy",
        ),
    ),
    "result_mode": _parameter_schema(
        "result_mode",
        "choice",
        "float32",
        choices=("float32", "preserve", "remap"),
    ),
    "mode": _parameter_schema(
        "mode",
        "choice",
        "lowpass",
        choices=("lowpass", "highpass", "bandpass", "bandstop"),
    ),
    "low_cutoff": _parameter_schema(
        "low_cutoff", "float", 0.0, 0, 0.5
    ),
    "high_cutoff": _parameter_schema(
        "high_cutoff", "float", 0.15, 0, 0.5
    ),
    "order": _parameter_schema("order", "int", 2, 1, 16),
    "boundary": _parameter_schema(
        "boundary",
        "choice",
        "periodic",
        choices=("periodic", "mirror_pad", "tukey"),
    ),
    "tukey_alpha": _parameter_schema(
        "tukey_alpha", "float", 0.25, 0, 1
    ),
    "frequency_unit": _parameter_schema(
        "frequency_unit",
        "choice",
        "cycles_per_pixel",
        choices=("cycles_per_pixel", "cycles_per_unit"),
    ),
    "pixel_size": _parameter_schema(
        "pixel_size", "float", 1.0, 1e-9, 1e12
    ),
    "logarithmic": _parameter_schema("logarithmic", "bool", True),
    "centered": _parameter_schema("centered", "bool", True),
    "window": _parameter_schema(
        "window", "choice", "none", choices=("none", "tukey")
    ),
    "k": _parameter_schema("k", "float", 0.2, -10, 10),
    "r": _parameter_schema("r", "float", 128.0, 1e-6, 1e12),
    "p": _parameter_schema("p", "float", 2.0, -100, 100),
    "q": _parameter_schema("q", "float", 10.0, -100, 100),
    "lower_percentile": _parameter_schema(
        "lower_percentile", "float", 0.5, 0, 99.999
    ),
    "upper_percentile": _parameter_schema(
        "upper_percentile", "float", 99.5, 0.001, 100
    ),
    "kind": _parameter_schema(
        "kind", "choice", "maxima", choices=("maxima", "minima")
    ),
    "h": _parameter_schema("h", "float", 0.0, 0, 1e12),
    "flat_field_source": _parameter_schema(
        "flat_field_source",
        "choice",
        "estimated",
        choices=("estimated", "reference"),
    ),
    "preserve_mean": _parameter_schema("preserve_mean", "bool", True),
    "direction": _parameter_schema(
        "direction",
        "choice",
        "horizontal",
        choices=("horizontal", "vertical"),
    ),
    "notch_width": _parameter_schema(
        "notch_width", "float", 0.02, 0.0001, 0.25
    ),
    "protect_radius": _parameter_schema(
        "protect_radius", "float", 0.02, 0, 0.25
    ),
    "strength": _parameter_schema("strength", "float", 1.0, 0, 1),
}


def _with_parameter_schema(
    base: ImageOperationParameterSchema,
    **changes: object,
) -> ImageOperationParameterSchema:
    payload = {
        "key": base.key,
        "kind": base.kind,
        "default": base.default,
        "minimum": base.minimum,
        "maximum": base.maximum,
        "choices": base.choices,
        "required": base.required,
        "required_when": base.required_when,
    }
    payload.update(changes)
    return ImageOperationParameterSchema(**payload)  # type: ignore[arg-type]


_PARAMETER_SCHEMA_OVERRIDES: dict[
    tuple[ImageOperation, str],
    ImageOperationParameterSchema,
] = {
    (ImageOperation.COPY, "fill_value"): _with_parameter_schema(
        _COMMON_PARAMETER_SCHEMAS["fill_value"]
    ),
    (ImageOperation.ADJUST_LEVELS, "gamma"): _with_parameter_schema(
        _COMMON_PARAMETER_SCHEMAS["gamma"], minimum=0.01, maximum=20
    ),
    (ImageOperation.CONVERT_TYPE, "scale_mode"): _with_parameter_schema(
        _COMMON_PARAMETER_SCHEMAS["scale_mode"],
        default=ConversionScaleMode.PRESERVE_VALUES.value,
    ),
    (ImageOperation.BRIGHTNESS_CONTRAST, "gamma"): _with_parameter_schema(
        _COMMON_PARAMETER_SCHEMAS["gamma"], minimum=0.01, maximum=20
    ),
    (ImageOperation.ROTATE, "interpolation"): _with_parameter_schema(
        _COMMON_PARAMETER_SCHEMAS["interpolation"], default="linear"
    ),
    (ImageOperation.ROTATE, "border_mode"): _with_parameter_schema(
        _COMMON_PARAMETER_SCHEMAS["border_mode"], default="constant"
    ),
    (ImageOperation.RESIZE, "interpolation"): _with_parameter_schema(
        _COMMON_PARAMETER_SCHEMAS["interpolation"], default="auto"
    ),
    (ImageOperation.TRANSLATE, "interpolation"): _with_parameter_schema(
        _COMMON_PARAMETER_SCHEMAS["interpolation"], default="linear"
    ),
    (ImageOperation.TRANSLATE, "border_mode"): _with_parameter_schema(
        _COMMON_PARAMETER_SCHEMAS["border_mode"], default="constant"
    ),
    (ImageOperation.PIXEL_BIN, "method"): _parameter_schema(
        "method",
        "choice",
        "mean",
        choices=("mean", "minimum", "maximum", "sum"),
    ),
    (ImageOperation.MEAN_FILTER, "border_mode"): _with_parameter_schema(
        _COMMON_PARAMETER_SCHEMAS["border_mode"],
        choices=_LOCAL_BORDER_MODES,
    ),
    (ImageOperation.UNSHARP_MASK, "threshold"): _with_parameter_schema(
        _COMMON_PARAMETER_SCHEMAS["threshold"],
        default=0.0,
        minimum=0,
        maximum=65535,
    ),
    (ImageOperation.SOBEL_EDGES, "kernel_size"): _with_parameter_schema(
        _COMMON_PARAMETER_SCHEMAS["kernel_size"], choices=(3, 5, 7)
    ),
    (ImageOperation.CANNY_EDGES, "threshold"): _with_parameter_schema(
        _COMMON_PARAMETER_SCHEMAS["threshold"],
        default=127.0,
    ),
    (ImageOperation.REMOVE_OUTLIERS, "threshold"): _with_parameter_schema(
        _COMMON_PARAMETER_SCHEMAS["threshold"],
        default=25.0,
        minimum=0,
        maximum=1e12,
    ),
    (ImageOperation.REPAIR_NONFINITE, "radius"): _with_parameter_schema(
        _COMMON_PARAMETER_SCHEMAS["radius"], maximum=32
    ),
    (ImageOperation.AUTO_THRESHOLD, "method"): _parameter_schema(
        "method",
        "choice",
        "otsu",
        choices=("otsu", "isodata", "triangle"),
    ),
    (ImageOperation.BINARIZE, "threshold"): _with_parameter_schema(
        _COMMON_PARAMETER_SCHEMAS["threshold"], default=127.0
    ),
    (ImageOperation.ERODE, "border_mode"): _with_parameter_schema(
        _COMMON_PARAMETER_SCHEMAS["border_mode"],
        choices=_LOCAL_BORDER_MODES,
    ),
    (ImageOperation.DILATE, "border_mode"): _with_parameter_schema(
        _COMMON_PARAMETER_SCHEMAS["border_mode"],
        choices=_LOCAL_BORDER_MODES,
    ),
    (ImageOperation.MORPHOLOGY_OPEN, "border_mode"): _with_parameter_schema(
        _COMMON_PARAMETER_SCHEMAS["border_mode"],
        choices=_LOCAL_BORDER_MODES,
    ),
    (ImageOperation.MORPHOLOGY_CLOSE, "border_mode"): _with_parameter_schema(
        _COMMON_PARAMETER_SCHEMAS["border_mode"],
        choices=_LOCAL_BORDER_MODES,
    ),
    (ImageOperation.TOP_HAT, "border_mode"): _with_parameter_schema(
        _COMMON_PARAMETER_SCHEMAS["border_mode"],
        choices=_LOCAL_BORDER_MODES,
    ),
    (ImageOperation.BLACK_HAT, "border_mode"): _with_parameter_schema(
        _COMMON_PARAMETER_SCHEMAS["border_mode"],
        choices=_LOCAL_BORDER_MODES,
    ),
    (ImageOperation.WATERSHED_V2, "seed_threshold"): _with_parameter_schema(
        _COMMON_PARAMETER_SCHEMAS["seed_threshold"], default=0.35
    ),
    (ImageOperation.BACKGROUND_SUBTRACT, "radius"): _with_parameter_schema(
        _COMMON_PARAMETER_SCHEMAS["radius"], default=25, maximum=2048
    ),
    (ImageOperation.BACKGROUND_SUBTRACT, "border_mode"): _with_parameter_schema(
        _COMMON_PARAMETER_SCHEMAS["border_mode"],
        choices=_LOCAL_BORDER_MODES,
    ),
    (
        ImageOperation.ROLLING_BALL_BACKGROUND_SUBTRACT,
        "radius",
    ): _parameter_schema("radius", "float", 25.0, 0.1, 2048),
    (ImageOperation.CUSTOM_CONVOLUTION, "kernel"): _parameter_schema(
        "kernel",
        "number_list",
        (-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0),
    ),
    (ImageOperation.CUSTOM_CONVOLUTION, "border_mode"): _with_parameter_schema(
        _COMMON_PARAMETER_SCHEMAS["border_mode"],
        choices=_LOCAL_BORDER_MODES,
    ),
    (ImageOperation.MULTIPLY, "value"): _with_parameter_schema(
        _COMMON_PARAMETER_SCHEMAS["value"], default=1.0
    ),
    (ImageOperation.DIVIDE, "value"): _with_parameter_schema(
        _COMMON_PARAMETER_SCHEMAS["value"], default=1.0
    ),
    (ImageOperation.IMAGE_CALCULATOR, "result_mode"): _parameter_schema(
        "result_mode",
        "choice",
        "preserve",
        choices=("preserve", "float32"),
    ),
    (ImageOperation.LOG_V2, "output_min"): _with_parameter_schema(
        _COMMON_PARAMETER_SCHEMAS["output_min"], default=0.0
    ),
    (ImageOperation.LOG_V2, "output_max"): _with_parameter_schema(
        _COMMON_PARAMETER_SCHEMAS["output_max"], default=1.0
    ),
    (ImageOperation.EXP_V2, "output_min"): _with_parameter_schema(
        _COMMON_PARAMETER_SCHEMAS["output_min"], default=0.0
    ),
    (ImageOperation.EXP_V2, "output_max"): _with_parameter_schema(
        _COMMON_PARAMETER_SCHEMAS["output_max"], default=1.0
    ),
    (ImageOperation.SQRT_V2, "output_min"): _with_parameter_schema(
        _COMMON_PARAMETER_SCHEMAS["output_min"], default=0.0
    ),
    (ImageOperation.SQRT_V2, "output_max"): _with_parameter_schema(
        _COMMON_PARAMETER_SCHEMAS["output_max"], default=1.0
    ),
    (ImageOperation.FFT_FILTER, "channel"): _with_parameter_schema(
        _COMMON_PARAMETER_SCHEMAS["channel"],
        default="per_channel",
        choices=_FFT_CHANNELS,
    ),
    (ImageOperation.FFT_FILTER, "output_float"): _with_parameter_schema(
        _COMMON_PARAMETER_SCHEMAS["output_float"], default=False
    ),
    (ImageOperation.FFT_FILTER, "boundary"): _with_parameter_schema(
        _COMMON_PARAMETER_SCHEMAS["boundary"],
        default="mirror_pad",
    ),
    (ImageOperation.FFT_FILTER, "low_cutoff"): _with_parameter_schema(
        _COMMON_PARAMETER_SCHEMAS["low_cutoff"],
        maximum=1e12,
    ),
    (ImageOperation.FFT_FILTER, "high_cutoff"): _with_parameter_schema(
        _COMMON_PARAMETER_SCHEMAS["high_cutoff"],
        maximum=1e12,
    ),
    (ImageOperation.FFT_FILTER, "pixel_size"): _with_parameter_schema(
        _COMMON_PARAMETER_SCHEMAS["pixel_size"],
        required_when=(("frequency_unit", "cycles_per_unit"),),
    ),
    (ImageOperation.ADAPTIVE_THRESHOLD, "method"): _parameter_schema(
        "method",
        "choice",
        "gaussian",
        choices=("mean", "gaussian", "sauvola", "phansalkar"),
    ),
    (ImageOperation.ADAPTIVE_THRESHOLD, "radius"): _with_parameter_schema(
        _COMMON_PARAMETER_SCHEMAS["radius"], default=7, maximum=255
    ),
    (ImageOperation.RANK_FILTER, "method"): _parameter_schema(
        "method",
        "choice",
        "minimum",
        choices=("minimum", "maximum", "variance"),
    ),
    (ImageOperation.RANK_FILTER, "radius"): _with_parameter_schema(
        _COMMON_PARAMETER_SCHEMAS["radius"], maximum=255
    ),
    (ImageOperation.MORPHOLOGY_DERIVATIVE, "method"): _parameter_schema(
        "method",
        "choice",
        "gradient",
        choices=("gradient", "laplacian"),
    ),
    (ImageOperation.MORPHOLOGY_DERIVATIVE, "radius"): _with_parameter_schema(
        _COMMON_PARAMETER_SCHEMAS["radius"], maximum=255
    ),
    (ImageOperation.MORPHOLOGICAL_RECONSTRUCTION, "method"): _parameter_schema(
        "method",
        "choice",
        "opening",
        choices=("opening", "closing"),
    ),
    (
        ImageOperation.MORPHOLOGICAL_RECONSTRUCTION,
        "radius",
    ): _with_parameter_schema(
        _COMMON_PARAMETER_SCHEMAS["radius"], maximum=255
    ),
    (ImageOperation.FLAT_FIELD_CORRECTION, "radius"): _parameter_schema(
        "radius", "float", 25.0, 0.1, 2048
    ),
    (ImageOperation.FLAT_FIELD_CORRECTION, "method"): _parameter_schema(
        "method",
        "choice",
        "gaussian",
        choices=("gaussian", "morphology"),
    ),
    (
        ImageOperation.FLAT_FIELD_CORRECTION,
        "secondary_document_id",
    ): _with_parameter_schema(
        _COMMON_PARAMETER_SCHEMAS["secondary_document_id"],
        required_when=(("flat_field_source", "reference"),),
    ),
}


def _image_operation_parameter_schema(
    operation: ImageOperation,
) -> tuple[ImageOperationParameterSchema, ...]:
    schemas: list[ImageOperationParameterSchema] = []
    for key in _PARAMETERS.get(operation, ()):
        schema = _PARAMETER_SCHEMA_OVERRIDES.get(
            (operation, key),
            _COMMON_PARAMETER_SCHEMAS.get(key),
        )
        if schema is None:
            raise RuntimeError(
                f"图像操作 {operation.value} 的参数 {key} 缺少 schema"
            )
        schemas.append(schema)
    return tuple(schemas)


_CHINESE_NAMES: dict[ImageOperation, str] = {
    ImageOperation.COPY: "复制像素",
    ImageOperation.CONVERT_TYPE: "转换像素类型",
    ImageOperation.CONVERT_COLOR: "转换颜色模型",
    ImageOperation.COLOR_BALANCE: "色彩平衡",
    ImageOperation.BRIGHTNESS_CONTRAST: "亮度/对比度",
    ImageOperation.ADJUST_LEVELS: "色阶",
    ImageOperation.THRESHOLD: "阈值",
    ImageOperation.FLIP_HORIZONTAL: "水平翻转",
    ImageOperation.FLIP_VERTICAL: "垂直翻转",
    ImageOperation.ROTATE_90_CLOCKWISE: "顺时针旋转 90°",
    ImageOperation.ROTATE_90_COUNTERCLOCKWISE: "逆时针旋转 90°",
    ImageOperation.ROTATE_180: "旋转 180°",
    ImageOperation.ROTATE: "任意角度旋转",
    ImageOperation.CROP: "裁剪",
    ImageOperation.RESIZE: "缩放",
    ImageOperation.TRANSLATE: "平移",
    ImageOperation.RESIZE_CANVAS: "调整画布",
    ImageOperation.PIXEL_BIN: "像素合并",
    ImageOperation.GAUSSIAN_BLUR: "高斯模糊",
    ImageOperation.MEDIAN_FILTER: "中值滤波",
    ImageOperation.MEAN_FILTER: "均值滤波",
    ImageOperation.BILATERAL_FILTER: "双边滤波",
    ImageOperation.UNSHARP_MASK: "反锐化遮罩",
    ImageOperation.SOBEL_EDGES: "Sobel 边缘",
    ImageOperation.LAPLACIAN_EDGES: "Laplacian 边缘",
    ImageOperation.CANNY_EDGES: "Canny 边缘",
    ImageOperation.NORMALIZE: "归一化",
    ImageOperation.HISTOGRAM_EQUALIZATION: "直方图均衡",
    ImageOperation.CLAHE: "局部直方图均衡",
    ImageOperation.REMOVE_OUTLIERS: "去除异常点",
    ImageOperation.REPAIR_NONFINITE: "修复非有限值",
    ImageOperation.AUTO_THRESHOLD: "自动阈值",
    ImageOperation.BINARIZE: "二值化",
    ImageOperation.ERODE: "腐蚀",
    ImageOperation.DILATE: "膨胀",
    ImageOperation.MORPHOLOGY_OPEN: "开运算",
    ImageOperation.MORPHOLOGY_CLOSE: "闭运算",
    ImageOperation.FILL_HOLES: "填充孔洞",
    ImageOperation.CONTOUR_EXTRACT: "提取轮廓",
    ImageOperation.REMOVE_SMALL_OBJECTS: "移除小对象",
    ImageOperation.FILL_SMALL_HOLES: "填充小孔洞",
    ImageOperation.DISTANCE_TRANSFORM: "距离变换",
    ImageOperation.SKELETONIZE: "骨架化",
    ImageOperation.WATERSHED: "分水岭",
    ImageOperation.WATERSHED_V2: "标记控制分水岭 v2",
    ImageOperation.TOP_HAT: "顶帽",
    ImageOperation.BLACK_HAT: "黑帽",
    ImageOperation.BACKGROUND_SUBTRACT: "形态学背景扣除",
    ImageOperation.ROLLING_BALL_BACKGROUND_SUBTRACT: "滑动抛物面背景扣除",
    ImageOperation.CUSTOM_CONVOLUTION: "自定义卷积",
    ImageOperation.INVERT: "反相",
    ImageOperation.ADD: "加常数",
    ImageOperation.SUBTRACT: "减常数",
    ImageOperation.MULTIPLY: "乘常数",
    ImageOperation.DIVIDE: "除常数",
    ImageOperation.GAMMA: "伽马",
    ImageOperation.LOG: "对数",
    ImageOperation.LOG_V2: "对数变换 v2",
    ImageOperation.EXP: "指数",
    ImageOperation.EXP_V2: "指数变换 v2",
    ImageOperation.SQRT: "平方根",
    ImageOperation.SQRT_V2: "平方根变换 v2",
    ImageOperation.ABS: "绝对值",
    ImageOperation.CLAMP: "截断",
    ImageOperation.IMAGE_CALCULATOR: "图像计算器",
    ImageOperation.FFT_FILTER: "频域滤波",
    ImageOperation.FFT_POWER_SPECTRUM: "FFT 功率谱",
    ImageOperation.STRIPE_SUPPRESSION: "条纹抑制",
    ImageOperation.ADAPTIVE_THRESHOLD: "局部自适应阈值",
    ImageOperation.PERCENTILE_SATURATION: "百分位饱和增强",
    ImageOperation.RANK_FILTER: "Rank 滤波",
    ImageOperation.MORPHOLOGY_DERIVATIVE: "形态学微分",
    ImageOperation.MORPHOLOGICAL_RECONSTRUCTION: "形态学重建",
    ImageOperation.REGIONAL_EXTREMA: "区域/扩展极值",
    ImageOperation.CLEAR_BORDER: "清除边界对象",
    ImageOperation.FLAT_FIELD_CORRECTION: "平场校正",
}


_GEOMETRY_OPERATIONS = {
    ImageOperation.FLIP_HORIZONTAL,
    ImageOperation.FLIP_VERTICAL,
    ImageOperation.ROTATE_90_CLOCKWISE,
    ImageOperation.ROTATE_90_COUNTERCLOCKWISE,
    ImageOperation.ROTATE_180,
    ImageOperation.ROTATE,
    ImageOperation.CROP,
    ImageOperation.RESIZE,
    ImageOperation.TRANSLATE,
    ImageOperation.RESIZE_CANVAS,
    ImageOperation.PIXEL_BIN,
}
_BINARY_OUTPUT_OPERATIONS = {
    ImageOperation.THRESHOLD,
    ImageOperation.CANNY_EDGES,
    ImageOperation.AUTO_THRESHOLD,
    ImageOperation.BINARIZE,
    ImageOperation.FILL_HOLES,
    ImageOperation.CONTOUR_EXTRACT,
    ImageOperation.REMOVE_SMALL_OBJECTS,
    ImageOperation.FILL_SMALL_HOLES,
    ImageOperation.SKELETONIZE,
    ImageOperation.WATERSHED,
    ImageOperation.WATERSHED_V2,
    ImageOperation.ADAPTIVE_THRESHOLD,
    ImageOperation.REGIONAL_EXTREMA,
    ImageOperation.CLEAR_BORDER,
}
_SCALAR_OUTPUT_OPERATIONS = {
    ImageOperation.ERODE,
    ImageOperation.DILATE,
    ImageOperation.MORPHOLOGY_OPEN,
    ImageOperation.MORPHOLOGY_CLOSE,
    ImageOperation.TOP_HAT,
    ImageOperation.BLACK_HAT,
}
_BINARY_INPUT_OPERATIONS = {
    ImageOperation.FILL_HOLES,
    ImageOperation.CONTOUR_EXTRACT,
    ImageOperation.REMOVE_SMALL_OBJECTS,
    ImageOperation.FILL_SMALL_HOLES,
    ImageOperation.DISTANCE_TRANSFORM,
    ImageOperation.SKELETONIZE,
    ImageOperation.WATERSHED,
    ImageOperation.WATERSHED_V2,
    ImageOperation.CLEAR_BORDER,
}
# These operations historically accepted any scalar image and silently split
# it at ``(minimum + maximum) / 2``.  Version 2 makes the scientific contract
# explicit while retaining deterministic replay for persisted v1 recipes.
_VERSIONED_STRICT_BINARY_OPERATIONS = _BINARY_INPUT_OPERATIONS - {
    ImageOperation.WATERSHED,
}
_VERSIONED_NEAREST_GEOMETRY_OPERATIONS = {
    ImageOperation.ROTATE,
    ImageOperation.TRANSLATE,
    ImageOperation.RESIZE,
}
_VERSIONED_EXPLICIT_FLOAT_RANGE_OPERATIONS = {
    ImageOperation.BRIGHTNESS_CONTRAST,
    ImageOperation.HISTOGRAM_EQUALIZATION,
}
_VERSIONED_SECONDARY_ALIGNMENT_OPERATIONS = {
    ImageOperation.IMAGE_CALCULATOR,
    ImageOperation.FLAT_FIELD_CORRECTION,
}
_SPATIAL_ALIGNMENT_CHANGING_OPERATIONS = {
    ImageOperation.FLIP_HORIZONTAL,
    ImageOperation.FLIP_VERTICAL,
    ImageOperation.ROTATE_90_CLOCKWISE,
    ImageOperation.ROTATE_90_COUNTERCLOCKWISE,
    ImageOperation.ROTATE_180,
    ImageOperation.ROTATE,
    ImageOperation.CROP,
    ImageOperation.RESIZE,
    ImageOperation.TRANSLATE,
    ImageOperation.RESIZE_CANVAS,
    ImageOperation.PIXEL_BIN,
}
_VERSION_2_OPERATIONS = (
    {
        ImageOperation.WATERSHED_V2,
        ImageOperation.ROLLING_BALL_BACKGROUND_SUBTRACT,
        ImageOperation.LOG_V2,
        ImageOperation.EXP_V2,
        ImageOperation.SQRT_V2,
    }
    | _VERSIONED_STRICT_BINARY_OPERATIONS
    | _VERSIONED_NEAREST_GEOMETRY_OPERATIONS
    | _VERSIONED_EXPLICIT_FLOAT_RANGE_OPERATIONS
    | _VERSIONED_SECONDARY_ALIGNMENT_OPERATIONS
)
_ROI_STATISTICS_OPERATIONS = {
    ImageOperation.CONVERT_TYPE,
    ImageOperation.ADJUST_LEVELS,
    ImageOperation.THRESHOLD,
    ImageOperation.NORMALIZE,
    ImageOperation.HISTOGRAM_EQUALIZATION,
    ImageOperation.AUTO_THRESHOLD,
    ImageOperation.PERCENTILE_SATURATION,
}
_ISOLATED_DOMAIN_OPERATIONS = _BINARY_INPUT_OPERATIONS | {
    ImageOperation.MORPHOLOGICAL_RECONSTRUCTION,
    ImageOperation.REGIONAL_EXTREMA,
}


def _operation_category(operation: ImageOperation) -> str:
    if operation in {
        ImageOperation.COPY,
        ImageOperation.CONVERT_TYPE,
        ImageOperation.CONVERT_COLOR,
    }:
        return "类型"
    if operation in _GEOMETRY_OPERATIONS:
        return "几何"
    if operation in {
        ImageOperation.GAUSSIAN_BLUR,
        ImageOperation.MEDIAN_FILTER,
        ImageOperation.MEAN_FILTER,
        ImageOperation.BILATERAL_FILTER,
        ImageOperation.UNSHARP_MASK,
        ImageOperation.REMOVE_OUTLIERS,
        ImageOperation.BACKGROUND_SUBTRACT,
        ImageOperation.CUSTOM_CONVOLUTION,
        ImageOperation.ROLLING_BALL_BACKGROUND_SUBTRACT,
        ImageOperation.RANK_FILTER,
        ImageOperation.FLAT_FIELD_CORRECTION,
        ImageOperation.MORPHOLOGICAL_RECONSTRUCTION,
    }:
        return "滤波"
    if operation in {
        ImageOperation.FFT_FILTER,
        ImageOperation.STRIPE_SUPPRESSION,
        ImageOperation.FFT_POWER_SPECTRUM,
    }:
        return "频域"
    if operation in _BINARY_OUTPUT_OPERATIONS | _SCALAR_OUTPUT_OPERATIONS | {
        ImageOperation.DISTANCE_TRANSFORM
    }:
        return "分割/形态学"
    return "调整/算术"


def _registered_parameter_condition(
    operation: ImageOperation,
    state: RasterTypeState,
    parameters: Mapping[str, object],
) -> str | None:
    choices: dict[ImageOperation, tuple[str, str, set[str]]] = {
        ImageOperation.ADAPTIVE_THRESHOLD: (
            "method",
            "局部阈值方法",
            {"mean", "gaussian", "sauvola", "phansalkar"},
        ),
        ImageOperation.RANK_FILTER: (
            "method",
            "Rank 方法",
            {"minimum", "maximum", "variance"},
        ),
        ImageOperation.MORPHOLOGY_DERIVATIVE: (
            "method",
            "形态学微分方法",
            {"gradient", "laplacian"},
        ),
        ImageOperation.MORPHOLOGICAL_RECONSTRUCTION: (
            "method",
            "形态学重建方法",
            {"opening", "closing"},
        ),
        ImageOperation.REGIONAL_EXTREMA: (
            "kind",
            "区域极值类型",
            {"maxima", "minima"},
        ),
    }
    if operation in choices:
        key, label, supported = choices[operation]
        value = str(parameters.get(key, next(iter(supported)))).strip().lower()
        if value not in supported:
            return f"{label}不受支持：{value}"
    if operation is ImageOperation.FLAT_FIELD_CORRECTION:
        source_mode = str(
            parameters.get("flat_field_source", "estimated")
        ).strip().lower()
        if source_mode not in {"estimated", "reference"}:
            return "平场来源必须为 estimated 或 reference"
        if source_mode == "reference":
            if not str(parameters.get("secondary_document_id", "")).strip():
                return "参考图平场校正必须选择第二幅参考图像"
            secondary_sha256 = str(
                parameters.get("secondary_sha256", "")
            ).strip().lower()
            if secondary_sha256 and (
                len(secondary_sha256) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in secondary_sha256
                )
            ):
                return "参考平场像素摘要必须是 64 位 SHA256"
            reference_levels = parameters.get("reference_levels")
            if reference_levels is not None:
                raw_levels = (
                    reference_levels
                    if isinstance(reference_levels, (list, tuple))
                    else (reference_levels,)
                )
                corrected_channel_count = min(3, state.channel_count)
                if len(raw_levels) != corrected_channel_count:
                    return (
                        "参考平场归一化值数量必须与源图像通道数一致"
                    )
                try:
                    normalized_levels = tuple(
                        float(value) for value in raw_levels
                    )
                except (TypeError, ValueError):
                    return "参考平场归一化值必须是正有限数"
                if any(
                    not math.isfinite(value) or value <= 0
                    for value in normalized_levels
                ):
                    return "参考平场归一化值必须是正有限数"
        else:
            method = str(
                parameters.get("method", "gaussian")
            ).strip().lower()
            if method not in {"gaussian", "morphology"}:
                return f"平场方法不受支持：{method}"
    if operation in {
        ImageOperation.LOG_V2,
        ImageOperation.EXP_V2,
        ImageOperation.SQRT_V2,
    }:
        mode = str(parameters.get("result_mode", "float32")).strip().lower()
        if mode not in {"float32", "preserve", "remap"}:
            return "科学变换结果模式必须为 float32、preserve 或 remap"
    if operation is ImageOperation.FFT_FILTER:
        boundary = str(parameters.get("boundary", "periodic")).strip().lower()
        if boundary not in {"periodic", "mirror_pad", "tukey"}:
            return "FFT 边界策略必须为 periodic、mirror_pad 或 tukey"
        unit = str(
            parameters.get("frequency_unit", "cycles_per_pixel")
        ).strip().lower()
        if unit not in {"cycles_per_pixel", "cycles_per_unit"}:
            return "FFT 频率单位不受支持"
        pixel_size = parameters.get("pixel_size")
        if unit == "cycles_per_unit":
            if pixel_size is None:
                return "cycles_per_unit 需要显式 pixel_size"
            resolved_pixel_size = float(pixel_size)
            nyquist = 0.5 / resolved_pixel_size
            unit_label = "周期/物理单位"
        else:
            nyquist = 0.5
            unit_label = "周期/像素"
        low_cutoff = float(parameters.get("low_cutoff", 0.0))
        high_cutoff = float(parameters.get("high_cutoff", 0.15))
        if not 0.0 <= low_cutoff <= nyquist:
            return (
                f"FFT 低截止频率必须在 0 到 Nyquist 频率 "
                f"{nyquist:g} {unit_label}之间"
            )
        if not 0.0 <= high_cutoff <= nyquist:
            return (
                f"FFT 高截止频率必须在 0 到 Nyquist 频率 "
                f"{nyquist:g} {unit_label}之间"
            )
        mode = str(parameters.get("mode", "lowpass")).strip().lower()
        if mode in {"bandpass", "bandstop"} and high_cutoff <= low_cutoff:
            return "带通或带阻滤波的高截止频率必须大于低截止频率"
        if mode == "lowpass" and high_cutoff <= 0:
            return "低通滤波的高截止频率必须为正数"
        if mode == "highpass" and low_cutoff <= 0:
            return "高通滤波的低截止频率必须为正数"
    if operation is ImageOperation.CANNY_EDGES:
        low = float(parameters.get("threshold_low", 50.0))
        high = float(parameters.get("threshold_high", 150.0))
        if not math.isfinite(low) or not math.isfinite(high):
            return "Canny 低阈值和高阈值必须是有限数"
        if low < 0.0 or high <= low:
            return "Canny 阈值必须满足 0 ≤ 低阈值 < 高阈值"
    return None


def _build_image_operation_registry() -> Mapping[str, ImageOperationDescriptor]:
    descriptors: dict[str, ImageOperationDescriptor] = {}
    for operation in ImageOperation:
        conditions: tuple[
            Callable[[RasterTypeState, Mapping[str, object]], str | None],
            ...,
        ] = ()
        if operation is ImageOperation.COLOR_BALANCE:
            conditions = (_color_input,)
        elif operation is ImageOperation.CLAHE:
            conditions = (_integer_input,)
        elif operation is ImageOperation.CANNY_EDGES:
            conditions = (_canny_input,)
        elif operation is ImageOperation.MEDIAN_FILTER:
            conditions = (_median_filter_input,)
        elif operation is ImageOperation.REMOVE_OUTLIERS:
            conditions = (_remove_outliers_input,)
        elif operation is ImageOperation.REPAIR_NONFINITE:
            conditions = (_repair_nonfinite_input,)
        elif operation is ImageOperation.PIXEL_BIN:
            conditions = (_pixel_bin_input,)
        elif operation is ImageOperation.IMAGE_CALCULATOR:
            conditions = (_image_calculator_input,)
        elif operation in _BINARY_INPUT_OPERATIONS:
            conditions = (_scalar_channel_input,)
        elif operation in {
            ImageOperation.ADAPTIVE_THRESHOLD,
            ImageOperation.MORPHOLOGICAL_RECONSTRUCTION,
            ImageOperation.REGIONAL_EXTREMA,
            ImageOperation.FFT_POWER_SPECTRUM,
        }:
            conditions = (_scalar_channel_input,)
        elif operation in {
            ImageOperation.LOG_V2,
            ImageOperation.EXP_V2,
            ImageOperation.SQRT_V2,
            ImageOperation.RANK_FILTER,
            ImageOperation.MORPHOLOGY_DERIVATIVE,
        }:
            conditions = (_authoritative_float_output_input,)
        conditions += (
            lambda state, parameters, resolved=operation: (
                _registered_parameter_condition(
                    resolved,
                    state,
                    parameters,
                )
            ),
        )

        if operation is ImageOperation.CONVERT_TYPE:
            output_resolver = _convert_type_output
        elif operation is ImageOperation.CONVERT_COLOR:
            output_resolver = _convert_color_output
        elif operation in _GEOMETRY_OPERATIONS:
            output_resolver = (
                lambda state, parameters, resolved=operation: _geometry_output(
                    resolved,
                    state,
                    parameters,
                )
            )
        elif operation in _BINARY_OUTPUT_OPERATIONS:
            output_resolver = _binary_output
        elif operation in _SCALAR_OUTPUT_OPERATIONS:
            output_resolver = _scalar_output
        elif operation is ImageOperation.DISTANCE_TRANSFORM:
            output_resolver = _float_scalar_output
        elif operation in {
            ImageOperation.SOBEL_EDGES,
            ImageOperation.LAPLACIAN_EDGES,
        }:
            output_resolver = (
                lambda state, parameters: (
                    state.replace(
                        pixel_type=RasterPixelType.GRAY32_FLOAT,
                        semantic=RasterSemantic.INTENSITY,
                    )
                    if bool(parameters.get("output_float", True))
                    else _scalar_output(state, parameters)
                )
            )
        elif operation is ImageOperation.FFT_FILTER:
            output_resolver = _fft_output
        elif operation is ImageOperation.COPY:
            output_resolver = (
                lambda state, parameters: (
                    state.replace(
                        pixel_type=RasterPixelType.RGBA8,
                        semantic=RasterSemantic.COLOR,
                        preserve_dimensions=False,
                    )
                    if bool(parameters.get("transparent_outside", False))
                    else state.replace(preserve_dimensions=False)
                )
            )
        elif operation in {
            ImageOperation.LOG_V2,
            ImageOperation.EXP_V2,
            ImageOperation.SQRT_V2,
        }:
            output_resolver = (
                lambda state, parameters: (
                    state.replace(
                        pixel_type=RasterPixelType.GRAY32_FLOAT,
                        semantic=RasterSemantic.INTENSITY,
                    )
                    if str(parameters.get("result_mode", "float32"))
                    == "float32"
                    else state
                )
            )
        elif operation is ImageOperation.IMAGE_CALCULATOR:
            output_resolver = (
                lambda state, parameters: (
                    state.replace(
                        pixel_type=RasterPixelType.GRAY32_FLOAT,
                        semantic=RasterSemantic.INTENSITY,
                    )
                    if str(parameters.get("result_mode", "preserve"))
                    == "float32"
                    else state
                )
            )
        elif operation is ImageOperation.RANK_FILTER:
            output_resolver = (
                lambda state, parameters: (
                    state.replace(
                        pixel_type=RasterPixelType.GRAY32_FLOAT,
                        semantic=RasterSemantic.INTENSITY,
                    )
                    if str(parameters.get("method", "minimum")) == "variance"
                    else state
                )
            )
        elif operation is ImageOperation.MORPHOLOGY_DERIVATIVE:
            output_resolver = (
                lambda state, parameters: (
                    state.replace(
                        pixel_type=RasterPixelType.GRAY32_FLOAT,
                        semantic=RasterSemantic.INTENSITY,
                    )
                    if str(parameters.get("method", "gradient"))
                    == "laplacian"
                    else (
                        state
                        if not state.is_grayscale
                        and parameters.get("channel") is None
                        else _scalar_output(state, parameters)
                    )
                )
            )
        elif operation is ImageOperation.FFT_POWER_SPECTRUM:
            output_resolver = (
                lambda state, _parameters: state.replace(
                    pixel_type=RasterPixelType.GRAY32_FLOAT,
                    semantic=RasterSemantic.INTENSITY,
                )
            )
        else:
            output_resolver = _same_output

        if operation in {ImageOperation.CROP, ImageOperation.COPY}:
            roi_semantics = RoiProcessingSemantics.CROP_BOUNDS_OR_MASK
        elif (
            operation in _GEOMETRY_OPERATIONS
            or operation is ImageOperation.CONVERT_COLOR
        ):
            roi_semantics = RoiProcessingSemantics.UNSUPPORTED
        elif operation in _ROI_STATISTICS_OPERATIONS:
            roi_semantics = RoiProcessingSemantics.ROI_STATISTICS
        elif operation in _ISOLATED_DOMAIN_OPERATIONS:
            roi_semantics = RoiProcessingSemantics.ISOLATED_DOMAIN
        else:
            roi_semantics = RoiProcessingSemantics.WRITE_MASK_WITH_HALO
        resource = (
            "cpu_memory_intensive"
            if operation
            in {
                ImageOperation.FFT_FILTER,
                ImageOperation.FFT_POWER_SPECTRUM,
                ImageOperation.STRIPE_SUPPRESSION,
                ImageOperation.WATERSHED,
                ImageOperation.WATERSHED_V2,
                ImageOperation.ROLLING_BALL_BACKGROUND_SUBTRACT,
                ImageOperation.MORPHOLOGICAL_RECONSTRUCTION,
            }
            else "cpu"
        )
        descriptors[operation.value] = ImageOperationDescriptor(
            operation_id=operation.value,
            chinese_name=_CHINESE_NAMES[operation],
            category=_operation_category(operation),
            parameter_schema=_image_operation_parameter_schema(operation),
            input_conditions=conditions,
            output_resolver=output_resolver,
            roi_semantics=roi_semantics,
            resource=resource,
            tile=(
                lambda parameters, resolved=operation: (
                    resolve_image_operation_capability(resolved, parameters)
                )
            ),
            executor=execute_image_operation,
            version="2" if operation in _VERSION_2_OPERATIONS else "1",
        )
    return MappingProxyType(descriptors)


IMAGE_OPERATION_REGISTRY = _build_image_operation_registry()


class ImageProcessingService:
    """Stateless façade used by UI and background-task controllers."""

    @staticmethod
    def execute(request: ImageOperationRequest) -> ImageOperationResult:
        return execute_image_operation(request)

    @staticmethod
    def registry() -> Mapping[str, ImageOperationDescriptor]:
        return image_operation_registry()

    @staticmethod
    def validate_recipe(
        recipe: ImageProcessingRecipe,
        input_state: RasterTypeState,
        *,
        roi_requested: bool = False,
        secondary_states: Mapping[str, RasterTypeState] | None = None,
    ) -> RecipeValidationResult:
        return validate_image_processing_recipe(
            recipe,
            input_state,
            roi_requested=roi_requested,
            secondary_states=secondary_states,
        )
