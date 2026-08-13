"""Registration and request boundary for advanced two-dimensional analyses.

The numerical kernels in :mod:`fdm.services.advanced_image_analysis` expose
strongly typed requests.  This module adds a small registry suitable for menus,
batch recipes and plug-in style extension without making those callers import
every concrete request type.

The registry never mutates a source :class:`~fdm.raster.RasterPlane`.  It also
checks cancellation before and after the numerical kernel and preserves the
``request_id``/``generation`` pair all the way to the result boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
import json
import math
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping, TypeAlias

import numpy as np
from numpy.typing import NDArray

from fdm.analysis_artifacts import AnalysisToolSpec
from fdm.cancellation import CancellationToken
from fdm.raster import RasterPlane
from fdm.services.advanced_image_analysis import (
    AdvancedAnalysisKind,
    AdvancedAnalysisLimits,
    DEFAULT_ADVANCED_ANALYSIS_LIMITS,
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
)
from fdm.services.raster_io import raster_plane_to_numpy


AdvancedResult: TypeAlias = object
AdvancedExecutor: TypeAlias = Callable[
    ["AdvancedAnalysisInvocation", CancellationToken | None, AdvancedAnalysisLimits],
    AdvancedResult,
]

_ADVANCED_TOOL_VERSIONS: Mapping[AdvancedAnalysisKind, str] = MappingProxyType(
    {
        AdvancedAnalysisKind.DIRECTIONALITY: "2",
        AdvancedAnalysisKind.SKELETON_NETWORK: "2",
        AdvancedAnalysisKind.LOCAL_THICKNESS: "1",
        AdvancedAnalysisKind.TUBENESS: "1",
        AdvancedAnalysisKind.GLCM_HARALICK: "1",
        AdvancedAnalysisKind.SPATIAL_DISTRIBUTION: "2",
        AdvancedAnalysisKind.INTENSITY_SURFACE: "1",
    }
)


def _analysis_tool_spec(
    kind: AdvancedAnalysisKind,
    *,
    version: str,
    chinese_name: str,
) -> AnalysisToolSpec:
    binary_kinds = {
        AdvancedAnalysisKind.SKELETON_NETWORK,
        AdvancedAnalysisKind.LOCAL_THICKNESS,
    }
    point_kinds = {AdvancedAnalysisKind.SPATIAL_DISTRIBUTION}
    if kind in binary_kinds:
        convertible_kinds = ("binary_mask",)
    elif kind in point_kinds:
        convertible_kinds = ("point_set", "measurement_group")
    else:
        convertible_kinds = ("image", "roi")
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
        convertible_kinds=convertible_kinds,
    )


@dataclass(frozen=True, slots=True)
class AdvancedAnalysisRegistration:
    """One discoverable advanced-analysis implementation."""

    kind: AdvancedAnalysisKind
    chinese_name: str
    algorithm_version: str
    input_description: str
    executor: AdvancedExecutor
    tool_spec: AnalysisToolSpec | None = None

    def __post_init__(self) -> None:
        normalized_kind = AdvancedAnalysisKind(self.kind)
        object.__setattr__(self, "kind", normalized_kind)
        for field_name in ("chinese_name", "algorithm_version", "input_description"):
            value = str(getattr(self, field_name) or "").strip()
            if not value:
                raise ValueError(f"{field_name} 不能为空")
            object.__setattr__(self, field_name, value)
        if not callable(self.executor):
            raise TypeError("executor 必须可调用")
        spec = self.tool_spec
        if spec is None:
            spec = _analysis_tool_spec(
                normalized_kind,
                version=self.algorithm_version,
                chinese_name=self.chinese_name,
            )
        elif not isinstance(spec, AnalysisToolSpec):
            raise TypeError("tool_spec 必须是 AnalysisToolSpec")
        expected_tool_id = f"fdm.{normalized_kind.value}"
        if spec.tool_id != expected_tool_id:
            raise ValueError(
                f"tool_spec.tool_id 必须与分析类型一致: {expected_tool_id}"
            )
        if spec.version != self.algorithm_version:
            raise ValueError("tool_spec.version 必须与 algorithm_version 一致")
        if spec.chinese_name != self.chinese_name:
            raise ValueError("tool_spec.chinese_name 必须与 chinese_name 一致")
        object.__setattr__(self, "tool_spec", spec)


@dataclass(frozen=True, slots=True, init=False)
class AdvancedAnalysisInvocation:
    """Immutable, generic input for one registered analysis."""

    kind: AdvancedAnalysisKind
    request_id: str
    generation: int
    plane: RasterPlane | None
    roi_mask: NDArray[np.bool_] | None
    binary_mask: NDArray[np.bool_] | None
    points: tuple[tuple[float, float], ...]
    pixel_size_x: float
    pixel_size_y: float
    unit: str
    _parameters_json: str

    def __init__(
        self,
        kind: AdvancedAnalysisKind | str,
        *,
        request_id: str,
        generation: int,
        plane: RasterPlane | None = None,
        roi_mask: NDArray[np.bool_] | None = None,
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
        normalized_roi = _freeze_mask(roi_mask, expected_shape, "ROI")
        normalized_binary = _freeze_mask(
            binary_mask,
            expected_shape,
            "二值掩膜",
        )
        normalized_points = tuple(
            (_finite_coordinate(x, "点 X"), _finite_coordinate(y, "点 Y"))
            for x, y in points
        )
        normalized_pixel_x = _positive_finite(pixel_size_x, "横向像素尺寸")
        normalized_pixel_y = _positive_finite(pixel_size_y, "纵向像素尺寸")
        normalized_unit = str(unit or "px").strip() or "px"
        parameters_json = json.dumps(
            dict(parameters or {}),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        object.__setattr__(self, "kind", AdvancedAnalysisKind(kind))
        object.__setattr__(self, "request_id", normalized_request_id)
        object.__setattr__(self, "generation", normalized_generation)
        object.__setattr__(self, "plane", plane)
        object.__setattr__(self, "roi_mask", normalized_roi)
        object.__setattr__(self, "binary_mask", normalized_binary)
        object.__setattr__(self, "points", normalized_points)
        object.__setattr__(self, "pixel_size_x", normalized_pixel_x)
        object.__setattr__(self, "pixel_size_y", normalized_pixel_y)
        object.__setattr__(self, "unit", normalized_unit)
        object.__setattr__(self, "_parameters_json", parameters_json)

    @property
    def parameters(self) -> Mapping[str, object]:
        return MappingProxyType(json.loads(self._parameters_json))


@dataclass(frozen=True, slots=True)
class AdvancedArrayDescriptor:
    name: str
    shape: tuple[int, ...]
    dtype: str
    byte_count: int


@dataclass(frozen=True, slots=True)
class AdvancedAnalysisExecution:
    kind: AdvancedAnalysisKind
    chinese_name: str
    algorithm_version: str
    request_id: str
    generation: int
    tool_spec: AnalysisToolSpec
    result: AdvancedResult
    scalar_report: tuple[tuple[str, str | int | float | bool | None], ...]
    arrays: tuple[AdvancedArrayDescriptor, ...]

    @property
    def scalar_report_map(self) -> Mapping[str, str | int | float | bool | None]:
        return MappingProxyType(dict(self.scalar_report))


class AdvancedAnalysisRegistry:
    """A deterministic registry with seven built-in high-value analyses."""

    def __init__(self, *, include_builtins: bool = True) -> None:
        self._registrations: dict[
            AdvancedAnalysisKind,
            AdvancedAnalysisRegistration,
        ] = {}
        if include_builtins:
            for registration in builtin_advanced_analysis_registrations():
                self.register(registration)

    def register(
        self,
        registration: AdvancedAnalysisRegistration,
        *,
        replace: bool = False,
    ) -> None:
        if not isinstance(registration, AdvancedAnalysisRegistration):
            raise TypeError("registration 必须是 AdvancedAnalysisRegistration")
        if registration.kind in self._registrations and not replace:
            raise ValueError(f"高级分析 {registration.kind.value} 已注册")
        self._registrations[registration.kind] = registration

    def registrations(self) -> tuple[AdvancedAnalysisRegistration, ...]:
        return tuple(
            self._registrations[kind]
            for kind in AdvancedAnalysisKind
            if kind in self._registrations
        )

    def registration(
        self,
        kind: AdvancedAnalysisKind | str,
    ) -> AdvancedAnalysisRegistration:
        normalized = AdvancedAnalysisKind(kind)
        try:
            return self._registrations[normalized]
        except KeyError as exc:
            raise ValueError(f"高级分析 {normalized.value} 尚未注册") from exc

    def execute(
        self,
        invocation: AdvancedAnalysisInvocation,
        *,
        cancellation_token: CancellationToken | None = None,
        limits: AdvancedAnalysisLimits = DEFAULT_ADVANCED_ANALYSIS_LIMITS,
    ) -> AdvancedAnalysisExecution:
        if not isinstance(invocation, AdvancedAnalysisInvocation):
            raise TypeError("invocation 必须是 AdvancedAnalysisInvocation")
        if cancellation_token is not None:
            cancellation_token.raise_if_cancelled()
        registration = self.registration(invocation.kind)
        result = registration.executor(invocation, cancellation_token, limits)
        if cancellation_token is not None:
            cancellation_token.raise_if_cancelled()
        result_request_id = getattr(result, "request_id", invocation.request_id)
        result_generation = getattr(result, "generation", invocation.generation)
        if (
            str(result_request_id) != invocation.request_id
            or int(result_generation) != invocation.generation
        ):
            raise RuntimeError("高级分析结果的 request_id/generation 与请求不一致")
        scalars, arrays = _describe_analysis_result(result)
        return AdvancedAnalysisExecution(
            kind=invocation.kind,
            chinese_name=registration.chinese_name,
            algorithm_version=registration.algorithm_version,
            request_id=invocation.request_id,
            generation=invocation.generation,
            tool_spec=registration.tool_spec,
            result=result,
            scalar_report=scalars,
            arrays=arrays,
        )


def builtin_advanced_analysis_registrations(
) -> tuple[AdvancedAnalysisRegistration, ...]:
    return (
        AdvancedAnalysisRegistration(
            AdvancedAnalysisKind.DIRECTIONALITY,
            "纤维方向性",
            _ADVANCED_TOOL_VERSIONS[AdvancedAnalysisKind.DIRECTIONALITY],
            "灰度图像，可选 ROI",
            _execute_directionality,
            tool_spec=_analysis_tool_spec(
                AdvancedAnalysisKind.DIRECTIONALITY,
                version=_ADVANCED_TOOL_VERSIONS[
                    AdvancedAnalysisKind.DIRECTIONALITY
                ],
                chinese_name="纤维方向性",
            ),
        ),
        AdvancedAnalysisRegistration(
            AdvancedAnalysisKind.SKELETON_NETWORK,
            "骨架网络",
            _ADVANCED_TOOL_VERSIONS[AdvancedAnalysisKind.SKELETON_NETWORK],
            "显式二值掩膜",
            _execute_skeleton,
            tool_spec=_analysis_tool_spec(
                AdvancedAnalysisKind.SKELETON_NETWORK,
                version=_ADVANCED_TOOL_VERSIONS[
                    AdvancedAnalysisKind.SKELETON_NETWORK
                ],
                chinese_name="骨架网络",
            ),
        ),
        AdvancedAnalysisRegistration(
            AdvancedAnalysisKind.LOCAL_THICKNESS,
            "局部厚度",
            _ADVANCED_TOOL_VERSIONS[AdvancedAnalysisKind.LOCAL_THICKNESS],
            "显式二值掩膜",
            _execute_local_thickness,
            tool_spec=_analysis_tool_spec(
                AdvancedAnalysisKind.LOCAL_THICKNESS,
                version=_ADVANCED_TOOL_VERSIONS[
                    AdvancedAnalysisKind.LOCAL_THICKNESS
                ],
                chinese_name="局部厚度",
            ),
        ),
        AdvancedAnalysisRegistration(
            AdvancedAnalysisKind.TUBENESS,
            "多尺度 Tubeness",
            _ADVANCED_TOOL_VERSIONS[AdvancedAnalysisKind.TUBENESS],
            "灰度图像，可选 ROI",
            _execute_tubeness,
            tool_spec=_analysis_tool_spec(
                AdvancedAnalysisKind.TUBENESS,
                version=_ADVANCED_TOOL_VERSIONS[
                    AdvancedAnalysisKind.TUBENESS
                ],
                chinese_name="多尺度 Tubeness",
            ),
        ),
        AdvancedAnalysisRegistration(
            AdvancedAnalysisKind.GLCM_HARALICK,
            "Haralick GLCM 纹理",
            _ADVANCED_TOOL_VERSIONS[AdvancedAnalysisKind.GLCM_HARALICK],
            "灰度图像，可选 ROI",
            _execute_glcm,
            tool_spec=_analysis_tool_spec(
                AdvancedAnalysisKind.GLCM_HARALICK,
                version=_ADVANCED_TOOL_VERSIONS[
                    AdvancedAnalysisKind.GLCM_HARALICK
                ],
                chinese_name="Haralick GLCM 纹理",
            ),
        ),
        AdvancedAnalysisRegistration(
            AdvancedAnalysisKind.SPATIAL_DISTRIBUTION,
            "空间分布（最近邻 / Ripley K/L）",
            _ADVANCED_TOOL_VERSIONS[AdvancedAnalysisKind.SPATIAL_DISTRIBUTION],
            "至少两个原始像素坐标点",
            _execute_spatial,
            tool_spec=_analysis_tool_spec(
                AdvancedAnalysisKind.SPATIAL_DISTRIBUTION,
                version=_ADVANCED_TOOL_VERSIONS[
                    AdvancedAnalysisKind.SPATIAL_DISTRIBUTION
                ],
                chinese_name="空间分布（最近邻 / Ripley K/L）",
            ),
        ),
        AdvancedAnalysisRegistration(
            AdvancedAnalysisKind.INTENSITY_SURFACE,
            "二维强度表面",
            _ADVANCED_TOOL_VERSIONS[AdvancedAnalysisKind.INTENSITY_SURFACE],
            "灰度图像，可选 ROI",
            _execute_surface,
            tool_spec=_analysis_tool_spec(
                AdvancedAnalysisKind.INTENSITY_SURFACE,
                version=_ADVANCED_TOOL_VERSIONS[
                    AdvancedAnalysisKind.INTENSITY_SURFACE
                ],
                chinese_name="二维强度表面",
            ),
        ),
    )


def _execute_directionality(
    invocation: AdvancedAnalysisInvocation,
    token: CancellationToken | None,
    limits: AdvancedAnalysisLimits,
) -> object:
    parameters = _selected_parameters(
        invocation.parameters,
        {
            "channel",
            "bins",
            "gradient_sigma",
            "minimum_gradient",
            "histogram_smoothing_bins",
            "peak_min_fraction",
            "max_peaks",
            "algorithm_version",
        },
    )
    channel = str(parameters.pop("channel", "luminance"))
    parameters.setdefault("algorithm_version", 2)
    return analyze_fiber_directionality(
        DirectionalityRequest(
            image=_scalar_image(invocation, channel),
            roi_mask=invocation.roi_mask,
            request_id=invocation.request_id,
            generation=invocation.generation,
            **parameters,
        ),
        cancellation_token=token,
        limits=limits,
    )


def _execute_skeleton(
    invocation: AdvancedAnalysisInvocation,
    token: CancellationToken | None,
    limits: AdvancedAnalysisLimits,
) -> object:
    parameters = _selected_parameters(
        invocation.parameters,
        {
            "already_skeletonized",
            "algorithm_version",
            "prune_terminal_branches_below",
        },
    )
    parameters.setdefault("algorithm_version", 2)
    return analyze_skeleton_network(
        SkeletonNetworkRequest(
            mask=_required_binary_mask(invocation),
            pixel_size_x=invocation.pixel_size_x,
            pixel_size_y=invocation.pixel_size_y,
            unit=invocation.unit,
            request_id=invocation.request_id,
            generation=invocation.generation,
            **parameters,
        ),
        cancellation_token=token,
        limits=limits,
    )


def _execute_local_thickness(
    invocation: AdvancedAnalysisInvocation,
    token: CancellationToken | None,
    limits: AdvancedAnalysisLimits,
) -> object:
    _selected_parameters(invocation.parameters, set())
    return calculate_local_thickness(
        LocalThicknessRequest(
            mask=_required_binary_mask(invocation),
            request_id=invocation.request_id,
            generation=invocation.generation,
        ),
        cancellation_token=token,
        limits=limits,
    )


def _execute_tubeness(
    invocation: AdvancedAnalysisInvocation,
    token: CancellationToken | None,
    limits: AdvancedAnalysisLimits,
) -> object:
    parameters = _selected_parameters(
        invocation.parameters,
        {"channel", "scales", "beta", "structure_scale", "bright_ridges"},
    )
    channel = str(parameters.pop("channel", "luminance"))
    if "scales" in parameters:
        parameters["scales"] = tuple(parameters["scales"])
    return calculate_multiscale_tubeness(
        TubenessRequest(
            image=_scalar_image(invocation, channel),
            roi_mask=invocation.roi_mask,
            request_id=invocation.request_id,
            generation=invocation.generation,
            **parameters,
        ),
        cancellation_token=token,
        limits=limits,
    )


def _execute_glcm(
    invocation: AdvancedAnalysisInvocation,
    token: CancellationToken | None,
    limits: AdvancedAnalysisLimits,
) -> object:
    parameters = _selected_parameters(
        invocation.parameters,
        {
            "channel",
            "levels",
            "distances",
            "directions_degrees",
            "value_range",
            "symmetric",
        },
    )
    channel = str(parameters.pop("channel", "luminance"))
    for sequence_name in ("distances", "directions_degrees", "value_range"):
        if sequence_name in parameters and parameters[sequence_name] is not None:
            parameters[sequence_name] = tuple(parameters[sequence_name])
    return calculate_glcm_haralick(
        GlcmHaralickRequest(
            image=_scalar_image(invocation, channel),
            roi_mask=invocation.roi_mask,
            request_id=invocation.request_id,
            generation=invocation.generation,
            **parameters,
        ),
        cancellation_token=token,
        limits=limits,
    )


def _execute_spatial(
    invocation: AdvancedAnalysisInvocation,
    token: CancellationToken | None,
    limits: AdvancedAnalysisLimits,
) -> object:
    parameters = _selected_parameters(
        invocation.parameters,
        {
            "study_area",
            "study_bounds",
            "ripley_radii",
            "algorithm_version",
        },
    )
    parameters.setdefault("algorithm_version", 2)
    for sequence_name in ("study_bounds", "ripley_radii"):
        if sequence_name in parameters and parameters[sequence_name] is not None:
            parameters[sequence_name] = tuple(parameters[sequence_name])
    return analyze_spatial_distribution(
        SpatialDistributionRequest(
            points=invocation.points,
            pixel_size_x=invocation.pixel_size_x,
            pixel_size_y=invocation.pixel_size_y,
            unit=invocation.unit,
            request_id=invocation.request_id,
            generation=invocation.generation,
            **parameters,
        ),
        cancellation_token=token,
        limits=limits,
    )


def _execute_surface(
    invocation: AdvancedAnalysisInvocation,
    token: CancellationToken | None,
    limits: AdvancedAnalysisLimits,
) -> object:
    parameters = _selected_parameters(
        invocation.parameters,
        {"channel", "sample_step_x", "sample_step_y"},
    )
    channel = str(parameters.pop("channel", "luminance"))
    return build_intensity_surface(
        IntensitySurfaceRequest(
            image=_scalar_image(invocation, channel),
            roi_mask=invocation.roi_mask,
            pixel_size_x=invocation.pixel_size_x,
            pixel_size_y=invocation.pixel_size_y,
            unit=invocation.unit,
            request_id=invocation.request_id,
            generation=invocation.generation,
            **parameters,
        ),
        cancellation_token=token,
        limits=limits,
    )


def _selected_parameters(
    parameters: Mapping[str, object],
    allowed: set[str],
) -> dict[str, Any]:
    unknown = sorted(set(parameters) - allowed)
    if unknown:
        raise ValueError(f"高级分析包含未知参数：{'、'.join(unknown)}")
    return dict(parameters)


def _scalar_image(
    invocation: AdvancedAnalysisInvocation,
    channel: str,
) -> NDArray[Any]:
    if invocation.plane is None:
        raise ValueError(f"{invocation.kind.value} 需要输入图像")
    image = np.asarray(raster_plane_to_numpy(invocation.plane))
    if image.ndim == 2:
        return image
    normalized_channel = channel.strip().casefold()
    channels = {
        "red": 0,
        "r": 0,
        "green": 1,
        "g": 1,
        "blue": 2,
        "b": 2,
    }
    if normalized_channel in channels:
        return image[..., channels[normalized_channel]]
    color = image[..., :3].astype(np.float64)
    if normalized_channel in {"average", "mean", "平均"}:
        return np.mean(color, axis=2)
    if normalized_channel not in {"luminance", "gray", "grey", "亮度"}:
        raise ValueError("通道只支持 luminance、average、red、green 或 blue")
    # Match the luminance channel used by the basic analysis and image-
    # processing services.  Explicit Rec.601 color conversion remains a
    # separate, user-selected operation.
    return (
        color[..., 0] * 0.2126
        + color[..., 1] * 0.7152
        + color[..., 2] * 0.0722
    )


def _required_binary_mask(
    invocation: AdvancedAnalysisInvocation,
) -> NDArray[np.bool_]:
    if invocation.binary_mask is None:
        raise ValueError(
            f"{invocation.kind.value} 需要显式二值掩膜，不能隐式阈值化原图"
        )
    return invocation.binary_mask


def _freeze_mask(
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


def _describe_analysis_result(
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


__all__ = [
    "AnalysisToolSpec",
    "AdvancedAnalysisExecution",
    "AdvancedAnalysisInvocation",
    "AdvancedAnalysisRegistration",
    "AdvancedAnalysisRegistry",
    "AdvancedArrayDescriptor",
    "builtin_advanced_analysis_registrations",
]
