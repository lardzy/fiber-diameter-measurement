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
from typing import Any, Mapping, TypeAlias

import cv2
import numpy as np
from numpy.typing import NDArray


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
    TOP_HAT = "top_hat"
    BLACK_HAT = "black_hat"
    BACKGROUND_SUBTRACT = "background_subtract"
    CUSTOM_CONVOLUTION = "custom_convolution"
    INVERT = "invert"
    ADD = "add"
    SUBTRACT = "subtract"
    MULTIPLY = "multiply"
    DIVIDE = "divide"
    GAMMA = "gamma"
    LOG = "log"
    EXP = "exp"
    SQRT = "sqrt"
    ABS = "abs"
    CLAMP = "clamp"
    IMAGE_CALCULATOR = "image_calculator"
    FFT_FILTER = "fft_filter"
    STRIPE_SUPPRESSION = "stripe_suppression"


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

    def __post_init__(self) -> None:
        image = _freeze_raster(self.image)
        object.__setattr__(self, "image", image)
        object.__setattr__(self, "warnings", tuple(str(item) for item in self.warnings))
        object.__setattr__(self, "metadata", _freeze_parameters(self.metadata))

    @property
    def metadata_map(self) -> Mapping[str, ParameterValue]:
        return MappingProxyType(dict(self.metadata))


def execute_image_operation(request: ImageOperationRequest) -> ImageOperationResult:
    """Execute one validated image operation."""

    image = np.asarray(request.image)
    params = dict(request.parameters)
    operation = request.operation
    warnings: list[str] = []
    metadata: dict[str, ParameterValue] = dict(request.parameters)

    if operation is ImageOperation.CONVERT_TYPE:
        target = _coerce_enum(
            PixelType,
            params.get("target_type", PixelType.UINT8.value),
            "目标位深",
        )
        mode = _coerce_enum(
            ConversionScaleMode,
            params.get("scale_mode", ConversionScaleMode.FULL_TYPE_RANGE.value),
            "位深转换缩放模式",
        )
        processed = convert_pixel_type(image, target, mode=mode)
        if request.roi_mask is not None:
            source_as_target = convert_pixel_type(
                image,
                target,
                mode=ConversionScaleMode.PRESERVE_VALUES,
            )
            processed = _blend_roi(source_as_target, processed, request.roi_mask)
        metadata.update(target_type=target.value, scale_mode=mode.value)
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
        processed = adjust_levels(
            image,
            black_point=float(params.get("black_point", _finite_min(image))),
            white_point=float(params.get("white_point", _finite_max(image))),
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
        processed = threshold_image(
            image,
            lower=float(params.get("lower", _finite_min(image))),
            upper=float(params.get("upper", _finite_max(image))),
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
        _reject_roi_for_geometry(request)
        processed = crop_image(
            image,
            x=int(params.get("x", 0)),
            y=int(params.get("y", 0)),
            width=int(params.get("width", image.shape[1])),
            height=int(params.get("height", image.shape[0])),
        )
    elif operation is ImageOperation.RESIZE:
        _reject_roi_for_geometry(request)
        width = int(params.get("width", image.shape[1]))
        height = int(params.get("height", image.shape[0]))
        interpolation = _coerce_enum(
            InterpolationMode,
            params.get("interpolation", InterpolationMode.LINEAR.value),
            "插值模式",
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
        )
        processed = _blend_roi(image, processed, request.roi_mask)
    elif operation is ImageOperation.HISTOGRAM_EQUALIZATION:
        processed = equalize_histogram(image)
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
        processed = fill_binary_holes(
            scalar,
            foreground_is_high=bool(params.get("foreground_is_high", True)),
        )
        processed = _blend_roi(scalar, processed, request.roi_mask)
    elif operation is ImageOperation.CONTOUR_EXTRACT:
        scalar = _require_scalar_image(image, params.get("channel"))
        processed = extract_binary_contours(
            scalar,
            foreground_is_high=bool(params.get("foreground_is_high", True)),
        )
        processed = _blend_roi(scalar, processed, request.roi_mask)
    elif operation is ImageOperation.REMOVE_SMALL_OBJECTS:
        scalar = _require_scalar_image(image, params.get("channel"))
        processed = remove_small_objects(
            scalar,
            minimum_area=int(params.get("minimum_area", 10)),
            connectivity=int(params.get("connectivity", 8)),
            foreground_is_high=bool(params.get("foreground_is_high", True)),
        )
        processed = _blend_roi(scalar, processed, request.roi_mask)
    elif operation is ImageOperation.FILL_SMALL_HOLES:
        scalar = _require_scalar_image(image, params.get("channel"))
        processed = fill_small_holes(
            scalar,
            maximum_area=int(params.get("maximum_area", 10)),
            connectivity=int(params.get("connectivity", 8)),
            foreground_is_high=bool(params.get("foreground_is_high", True)),
        )
        processed = _blend_roi(scalar, processed, request.roi_mask)
    elif operation is ImageOperation.DISTANCE_TRANSFORM:
        scalar = _require_scalar_image(image, params.get("channel"))
        processed = distance_transform(
            scalar,
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
        processed = skeletonize_binary(
            scalar,
            foreground_is_high=bool(params.get("foreground_is_high", True)),
        )
        processed = _blend_roi(scalar, processed, request.roi_mask)
    elif operation is ImageOperation.WATERSHED:
        scalar = _require_scalar_image(image, params.get("channel"))
        processed = watershed_split(
            scalar,
            foreground_is_high=bool(params.get("foreground_is_high", True)),
            seed_threshold=float(params.get("seed_threshold", 0.45)),
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
    elif operation is ImageOperation.IMAGE_CALCULATOR:
        if request.secondary_image is None:
            raise ValueError("图像计算器需要提供第二幅图像。")
        processed = image_calculator(
            image,
            request.secondary_image,
            operation=str(params.get("calculator_operation", "add")),
        )
        processed = _blend_roi(image, processed, request.roi_mask)
    elif operation is ImageOperation.FFT_FILTER:
        processed = fft_filter(
            image,
            mode=str(params.get("mode", "lowpass")),
            low_cutoff=float(params.get("low_cutoff", 0.0)),
            high_cutoff=float(params.get("high_cutoff", 0.15)),
            order=int(params.get("order", 2)),
            channel=str(params.get("channel", "per_channel")),
            output_float=bool(params.get("output_float", False)),
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
    )


def convert_pixel_type(
    image: NDArray[Any],
    target: PixelType | str,
    *,
    mode: ConversionScaleMode | str = ConversionScaleMode.FULL_TYPE_RANGE,
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
    target_dtype = target_type.dtype
    if source.dtype == target_dtype:
        return source.copy()

    work = source.astype(np.float64)
    if scale_mode is ConversionScaleMode.PRESERVE_VALUES:
        mapped = work
    elif scale_mode is ConversionScaleMode.DATA_RANGE:
        finite = np.isfinite(work)
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
    return _cast_like(mapped, target_dtype)


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
    interpolation: InterpolationMode | str = InterpolationMode.LINEAR,
) -> NDArray[Any]:
    source = _validate_raster(image)
    if width <= 0 or height <= 0:
        raise ValueError("调整后的宽度和高度必须为正数。")
    interpolation_mode = (
        interpolation
        if isinstance(interpolation, InterpolationMode)
        else _coerce_enum(InterpolationMode, interpolation, "插值模式")
    )
    resized = cv2.resize(
        source,
        (int(width), int(height)),
        interpolation=_cv_interpolation(interpolation_mode),
    )
    if source.ndim == 3 and source.shape[2] == 1 and resized.ndim == 2:
        return resized[..., np.newaxis]
    return resized


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
) -> NDArray[Any]:
    """Map the finite data range to an explicit output range."""

    source = _validate_raster(image)
    _require_finite("输出下限", output_min)
    _require_finite("输出上限", output_max)
    if output_max < output_min:
        raise ValueError("归一化输出上限不能小于输出下限。")

    def normalize_plane(plane: NDArray[Any]) -> NDArray[Any]:
        work = plane.astype(np.float64)
        finite = np.isfinite(work)
        if not np.any(finite):
            return plane.copy()
        low = float(np.min(work[finite]))
        high = float(np.max(work[finite]))
        result = work.copy()
        if math.isclose(low, high):
            result[finite] = output_min
        else:
            result[finite] = (
                (work[finite] - low)
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


def equalize_histogram(image: NDArray[Any]) -> NDArray[Any]:
    """Equalize the finite histogram without changing dtype or Alpha."""

    source = _validate_raster(image)

    def equalize_plane(plane: NDArray[Any]) -> NDArray[Any]:
        work = np.asarray(plane)
        finite = np.isfinite(work)
        if not np.any(finite):
            return work.copy()
        low, high = _working_range(work)
        if work.dtype == np.uint8:
            return cv2.equalizeHist(work)
        if work.dtype == np.uint16:
            histogram = np.bincount(work.ravel(), minlength=65536)
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
        values = work[finite].astype(np.float64)
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
    scalar = _require_scalar_image(source, channel)
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


def auto_threshold(
    image: NDArray[Any],
    *,
    method: str = "otsu",
    channel: str | None = None,
    invert: bool = False,
) -> tuple[NDArray[Any], float]:
    source = _validate_raster(image)
    scalar = _require_scalar_image(source, channel)
    finite = scalar[np.isfinite(scalar)]
    if finite.size == 0:
        raise ValueError("自动阈值无法处理不含有限像素的图像。")
    resolved_method = str(method).strip().lower()
    if resolved_method not in {"otsu", "isodata", "triangle"}:
        raise ValueError("自动阈值方法必须为 Otsu、IsoData 或 Triangle。")
    histogram, centers = _threshold_histogram(scalar)
    if resolved_method == "otsu":
        threshold = _otsu_threshold(histogram, centers)
    elif resolved_method == "isodata":
        threshold = _isodata_threshold(histogram, centers)
    else:
        threshold = _triangle_threshold(histogram, centers)
    return binarize_image(scalar, threshold=threshold, invert=invert), threshold


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


def fill_binary_holes(
    image: NDArray[Any],
    *,
    foreground_is_high: bool = True,
) -> NDArray[Any]:
    source = _validate_raster(image)
    if source.ndim != 2:
        raise ValueError("填充孔洞需要单通道图像。")
    foreground = _binary_mask(source, foreground_is_high=foreground_is_high)
    inverse = (~foreground).astype(np.uint8)
    count, labels = cv2.connectedComponents(inverse, connectivity=4)
    border_labels = set(
        np.concatenate(
            (
                labels[0, :],
                labels[-1, :],
                labels[:, 0],
                labels[:, -1],
            )
        ).tolist()
    )
    holes = np.zeros_like(foreground)
    for label in range(1, count):
        if label not in border_labels:
            holes |= labels == label
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


def image_calculator(
    first: NDArray[Any],
    second: NDArray[Any],
    *,
    operation: str,
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
    }
    if resolved not in supported:
        raise ValueError("不支持的图像计算器运算。")
    if resolved in {"and", "or", "xor"} and left.dtype.kind not in {"u", "i"}:
        raise TypeError("AND、OR、XOR 仅支持整数图像。")

    def calculate(a: NDArray[Any], b: NDArray[Any]) -> NDArray[Any]:
        if resolved == "and":
            return np.bitwise_and(a, b)
        if resolved == "or":
            return np.bitwise_or(a, b)
        if resolved == "xor":
            return np.bitwise_xor(a, b)
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
) -> NDArray[Any]:
    """Apply a Butterworth radial frequency filter.

    Cutoffs are cycles per pixel in ``[0, 0.5]``.  ``bandpass`` keeps
    ``low_cutoff <= f <= high_cutoff``; ``bandstop`` removes that interval.
    """

    source = _validate_raster(image)
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
) -> NDArray[np.float32]:
    height, width = plane.shape
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
    spectrum = np.fft.fft2(plane)
    return np.fft.ifft2(spectrum * response).real.astype(np.float32)


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
    return {
        InterpolationMode.NEAREST: cv2.INTER_NEAREST,
        InterpolationMode.LINEAR: cv2.INTER_LINEAR,
        InterpolationMode.CUBIC: cv2.INTER_CUBIC,
        InterpolationMode.AREA: cv2.INTER_AREA,
        InterpolationMode.LANCZOS: cv2.INTER_LANCZOS4,
    }[mode]


class ImageProcessingService:
    """Stateless façade used by UI and background-task controllers."""

    @staticmethod
    def execute(request: ImageOperationRequest) -> ImageOperationResult:
        return execute_image_operation(request)
