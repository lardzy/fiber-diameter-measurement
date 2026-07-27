from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import NDArray

from fdm.raster import RasterPixelType, RasterPlane


@dataclass(frozen=True, slots=True)
class ParameterHistogramSnapshot:
    """Finite-value histogram for one frozen workbench step input.

    The histogram is a display aid only. Exact threshold counts are calculated
    from the frozen scalar samples instead of being estimated from bins.
    """

    counts: tuple[int, ...]
    minimum: float
    maximum: float
    finite_count: int
    nonfinite_count: int
    masked_out_count: int
    channel: str

    def __post_init__(self) -> None:
        if not self.counts:
            raise ValueError("参数直方图至少需要一个分箱")
        if any(int(value) < 0 for value in self.counts):
            raise ValueError("参数直方图计数不能为负数")
        if not (
            math.isfinite(float(self.minimum))
            and math.isfinite(float(self.maximum))
            and float(self.maximum) > float(self.minimum)
        ):
            raise ValueError("参数直方图范围必须是递增的有限数")
        for name, value in (
            ("有限像素", self.finite_count),
            ("非有限像素", self.nonfinite_count),
            ("ROI 外像素", self.masked_out_count),
        ):
            if int(value) < 0:
                raise ValueError(f"{name}数量不能为负数")


def parameter_histogram_snapshot(
    raster: RasterPlane,
    *,
    channel: str = "luminance",
    roi_mask: NDArray[np.bool_] | None = None,
    range_hint: tuple[float, float] | None = None,
) -> ParameterHistogramSnapshot:
    """Build a bounded histogram using the same scalar-channel definition.

    Integer images retain their native range so threshold handles do not jump
    when the current sample lacks dark or bright pixels. Float images use the
    finite data range because they do not have a meaningful fixed type range.
    """

    scalar = scalar_parameter_samples(raster, channel=channel)
    active = _normalized_mask(roi_mask, scalar.shape)
    selected = scalar if active is None else scalar[active]
    masked_out_count = (
        0
        if active is None
        else int(scalar.size - selected.size)
    )
    finite_mask = np.isfinite(selected)
    finite = np.asarray(selected[finite_mask], dtype=np.float64)
    nonfinite_count = int(selected.size - finite.size)
    if finite.size == 0:
        raise ValueError("当前步骤输入不含可用于直方图的有限像素")

    minimum, maximum, bin_count = _histogram_range(
        raster.pixel_type,
        channel=channel,
        finite=finite,
    )
    if range_hint is not None:
        hint_minimum = float(range_hint[0])
        hint_maximum = float(range_hint[1])
        if not (
            math.isfinite(hint_minimum)
            and math.isfinite(hint_maximum)
            and hint_maximum >= hint_minimum
        ):
            raise ValueError("直方图范围提示必须是递增的有限数")
        minimum = min(minimum, hint_minimum)
        maximum = max(maximum, hint_maximum)
        if math.isclose(minimum, maximum):
            maximum = minimum + 1.0
    counts, _edges = np.histogram(
        finite,
        bins=bin_count,
        range=(minimum, maximum),
    )
    return ParameterHistogramSnapshot(
        counts=tuple(int(value) for value in counts),
        minimum=minimum,
        maximum=maximum,
        finite_count=int(finite.size),
        nonfinite_count=nonfinite_count,
        masked_out_count=masked_out_count,
        channel=str(channel),
    )


def count_parameter_range(
    raster: RasterPlane,
    *,
    lower: float,
    upper: float | None = None,
    channel: str = "luminance",
    roi_mask: NDArray[np.bool_] | None = None,
    single_threshold: bool = False,
    invert: bool = False,
) -> tuple[int, int]:
    """Return exact selected and finite counts for a threshold editor.

    ``single_threshold`` follows the existing binarize operation and selects
    strictly greater values. A range follows the existing threshold operation
    and selects the inclusive interval.
    """

    lower_value = float(lower)
    if not math.isfinite(lower_value):
        raise ValueError("阈值必须是有限数")
    upper_value = lower_value if upper is None else float(upper)
    if not math.isfinite(upper_value):
        raise ValueError("阈值上限必须是有限数")
    if not single_threshold and upper_value < lower_value:
        raise ValueError("阈值上限不能小于下限")

    scalar = scalar_parameter_samples(raster, channel=channel)
    active = _normalized_mask(roi_mask, scalar.shape)
    finite = np.isfinite(scalar)
    if active is not None:
        finite &= (
            active
            if scalar.ndim == 2
            else active[..., np.newaxis]
        )
    if single_threshold:
        selected = finite & (scalar > lower_value)
    else:
        selected = finite & (scalar >= lower_value) & (scalar <= upper_value)
    if invert:
        selected = finite & ~selected
    return int(np.count_nonzero(selected)), int(np.count_nonzero(finite))


def scalar_parameter_samples(
    raster: RasterPlane,
    *,
    channel: str = "luminance",
) -> NDArray[np.generic]:
    """Return scalar samples without changing authoritative raster pixels."""

    array = _raster_array(raster)
    if array.ndim == 2:
        return array
    resolved = str(channel).strip().lower()
    indices = {
        "red": 0,
        "r": 0,
        "green": 1,
        "g": 1,
        "blue": 2,
        "b": 2,
    }
    if resolved in indices:
        return array[..., indices[resolved]]
    if resolved in {"luminance", "gray", "grayscale"}:
        rgb = array[..., :3].astype(np.float64)
        return (
            rgb[..., 0] * 0.2126
            + rgb[..., 1] * 0.7152
            + rgb[..., 2] * 0.0722
        ).astype(np.float32)
    if resolved in {"all_channels", "rgb_channels"}:
        return array[..., :3]
    raise ValueError(f"不支持的直方图标量通道：{channel}")


def _raster_array(raster: RasterPlane) -> NDArray[np.generic]:
    pixel_type = raster.pixel_type
    if pixel_type is RasterPixelType.GRAY8:
        dtype = np.dtype(np.uint8)
        shape = (raster.height, raster.width)
    elif pixel_type is RasterPixelType.GRAY16:
        dtype = np.dtype("<u2")
        shape = (raster.height, raster.width)
    elif pixel_type is RasterPixelType.GRAY32_FLOAT:
        dtype = np.dtype("<f4")
        shape = (raster.height, raster.width)
    elif pixel_type is RasterPixelType.RGB8:
        dtype = np.dtype(np.uint8)
        shape = (raster.height, raster.width, 3)
    elif pixel_type is RasterPixelType.RGBA8:
        dtype = np.dtype(np.uint8)
        shape = (raster.height, raster.width, 4)
    else:  # pragma: no cover - exhaustive enum guard
        raise ValueError(f"不支持的栅格类型：{pixel_type}")
    result = np.frombuffer(raster.data, dtype=dtype).reshape(shape)
    result.setflags(write=False)
    return result


def _normalized_mask(
    roi_mask: NDArray[np.bool_] | None,
    shape: tuple[int, ...],
) -> NDArray[np.bool_] | None:
    if roi_mask is None:
        return None
    normalized = np.asarray(roi_mask, dtype=np.bool_)
    if normalized.shape != shape[:2]:
        raise ValueError("直方图 ROI 掩膜尺寸必须与当前步骤输入一致")
    return normalized


def _histogram_range(
    pixel_type: RasterPixelType,
    *,
    channel: str,
    finite: NDArray[np.float64],
) -> tuple[float, float, int]:
    if pixel_type in {
        RasterPixelType.GRAY8,
        RasterPixelType.RGB8,
        RasterPixelType.RGBA8,
    }:
        return 0.0, 255.0, 256
    if pixel_type is RasterPixelType.GRAY16:
        return 0.0, 65535.0, 1024

    minimum = float(np.min(finite))
    maximum = float(np.max(finite))
    if math.isclose(minimum, maximum):
        padding = max(1.0, abs(minimum) * 0.01)
        minimum -= padding
        maximum += padding
    return minimum, maximum, 512
