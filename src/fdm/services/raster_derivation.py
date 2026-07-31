from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import re
from types import MappingProxyType
from typing import Mapping, Sequence, TypeAlias

import numpy as np

from fdm.raster import RasterPixelType, RasterPlane
from fdm.services.raster_io import numpy_to_raster_plane, raster_plane_to_numpy


_SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")

ProvenanceScalar: TypeAlias = str | int | float | bool
ProvenanceValue: TypeAlias = ProvenanceScalar | tuple[ProvenanceScalar, ...]
FillValue: TypeAlias = int | float | Sequence[int | float]


class RasterDerivationError(ValueError):
    """Raised when a requested pixel derivation is not scientifically safe."""


@dataclass(frozen=True, slots=True)
class RasterBounds:
    """A non-empty rectangle in source-image pixel coordinates."""

    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        for field_name in ("x", "y", "width", "height"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"ROI {field_name} 必须是整数")
            if value < 0:
                raise RasterDerivationError(f"ROI {field_name} 不能为负数")
        if self.width == 0 or self.height == 0:
            raise RasterDerivationError("ROI 边界不能为空")

    @property
    def right(self) -> int:
        return self.x + self.width

    @property
    def bottom(self) -> int:
        return self.y + self.height

    def to_tuple(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.width, self.height)


@dataclass(frozen=True, slots=True)
class FrozenRasterRoi:
    """Immutable ROI mask whose bytes and bounds are covered by a SHA256."""

    bounds: RasterBounds
    mask_data: bytes
    mask_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.bounds, RasterBounds):
            raise TypeError("ROI bounds 必须是 RasterBounds")
        try:
            data = bytes(self.mask_data)
        except (TypeError, ValueError) as exc:
            raise TypeError("ROI mask_data 必须是 bytes-like 对象") from exc
        expected_size = self.bounds.width * self.bounds.height
        if len(data) != expected_size:
            raise RasterDerivationError(
                f"ROI 掩膜字节数不匹配：期望 {expected_size}，实际 {len(data)}"
            )
        if any(value not in {0, 1} for value in data):
            raise RasterDerivationError("ROI 掩膜只能包含 0 或 1")
        if not any(data):
            raise RasterDerivationError("ROI 掩膜不能为空")
        actual_sha256 = _roi_mask_sha256(self.bounds, data)
        expected_sha256 = _normalize_sha256(self.mask_sha256, "ROI 掩膜")
        if actual_sha256 != expected_sha256:
            raise RasterDerivationError(
                "ROI 掩膜 SHA256 不一致，冻结后的掩膜数据可能已损坏"
            )
        object.__setattr__(self, "mask_data", data)
        object.__setattr__(self, "mask_sha256", expected_sha256)

    @classmethod
    def from_numpy(
        cls,
        mask: np.ndarray,
        *,
        bounds: RasterBounds,
    ) -> "FrozenRasterRoi":
        source = np.asarray(mask)
        if source.dtype != np.dtype(np.bool_) or source.ndim != 2:
            raise RasterDerivationError("ROI 掩膜必须是二维 bool 数组")
        expected_shape = (bounds.height, bounds.width)
        if source.shape != expected_shape:
            raise RasterDerivationError(
                f"ROI 掩膜尺寸不匹配：期望 {expected_shape}，实际 {source.shape}"
            )
        normalized = np.ascontiguousarray(source, dtype=np.bool_)
        data = normalized.astype(np.uint8, copy=False).tobytes(order="C")
        return cls(
            bounds=bounds,
            mask_data=data,
            mask_sha256=_roi_mask_sha256(bounds, data),
        )

    def to_numpy(self) -> np.ndarray:
        """Return a read-only view backed by immutable bytes."""

        return np.frombuffer(self.mask_data, dtype=np.bool_).reshape(
            self.bounds.height,
            self.bounds.width,
        )


@dataclass(frozen=True, slots=True)
class RasterDerivationProvenance:
    operation_id: str
    implementation_version: int
    sources: tuple[tuple[str, str], ...]
    results: tuple[tuple[str, str], ...]
    parameters: tuple[tuple[str, ProvenanceValue], ...] = ()

    def __post_init__(self) -> None:
        operation_id = str(self.operation_id or "").strip()
        if not operation_id:
            raise RasterDerivationError("派生操作 ID 不能为空")
        if (
            isinstance(self.implementation_version, bool)
            or not isinstance(self.implementation_version, int)
            or self.implementation_version < 1
        ):
            raise RasterDerivationError("派生实现版本必须是正整数")
        sources = _normalize_named_hashes(self.sources, "派生来源")
        results = _normalize_named_hashes(self.results, "派生结果")
        if not sources or not results:
            raise RasterDerivationError("派生来源和结果均不能为空")
        parameters = _normalize_parameters(self.parameters)
        object.__setattr__(self, "operation_id", operation_id)
        object.__setattr__(self, "sources", sources)
        object.__setattr__(self, "results", results)
        object.__setattr__(self, "parameters", parameters)

    @property
    def parameter_map(self) -> Mapping[str, ProvenanceValue]:
        return MappingProxyType(dict(self.parameters))

    def to_dict(self) -> dict[str, object]:
        return {
            "operation_id": self.operation_id,
            "implementation_version": self.implementation_version,
            "sources": {name: digest for name, digest in self.sources},
            "results": {name: digest for name, digest in self.results},
            "parameters": {
                name: list(value) if isinstance(value, tuple) else value
                for name, value in self.parameters
            },
        }


@dataclass(frozen=True, slots=True)
class RasterCopyResult:
    plane: RasterPlane
    source_bounds: RasterBounds
    provenance: RasterDerivationProvenance

    def __post_init__(self) -> None:
        if not isinstance(self.plane, RasterPlane) or self.plane.is_empty:
            raise RasterDerivationError("复制结果必须是非空 RasterPlane")
        if not isinstance(self.source_bounds, RasterBounds):
            raise TypeError("source_bounds 必须是 RasterBounds")
        _require_bound_result(self.provenance, "copy", self.plane)


@dataclass(frozen=True, slots=True)
class NamedRasterChannel:
    channel_id: str
    display_name: str
    plane: RasterPlane

    def __post_init__(self) -> None:
        channel_id = str(self.channel_id or "").strip().upper()
        display_name = str(self.display_name or "").strip()
        if channel_id not in {"R", "G", "B"}:
            raise RasterDerivationError("RGB 通道 ID 只能是 R、G 或 B")
        if not display_name:
            raise RasterDerivationError("RGB 通道显示名称不能为空")
        if (
            not isinstance(self.plane, RasterPlane)
            or self.plane.is_empty
            or self.plane.pixel_type is not RasterPixelType.GRAY8
        ):
            raise RasterDerivationError("拆分通道必须是非空 GRAY8 RasterPlane")
        object.__setattr__(self, "channel_id", channel_id)
        object.__setattr__(self, "display_name", display_name)


@dataclass(frozen=True, slots=True)
class RasterChannelSplitResult:
    channels: tuple[NamedRasterChannel, NamedRasterChannel, NamedRasterChannel]
    provenance: RasterDerivationProvenance

    def __post_init__(self) -> None:
        channels = tuple(self.channels)
        if len(channels) != 3 or tuple(item.channel_id for item in channels) != (
            "R",
            "G",
            "B",
        ):
            raise RasterDerivationError("通道拆分结果必须按 R、G、B 排列")
        size = (channels[0].plane.width, channels[0].plane.height)
        if any((item.plane.width, item.plane.height) != size for item in channels):
            raise RasterDerivationError("拆分后的 RGB 通道尺寸必须一致")
        result_hashes = dict(self.provenance.results)
        for channel in channels:
            if result_hashes.get(channel.channel_id) != channel.plane.sha256():
                raise RasterDerivationError(
                    f"{channel.channel_id} 通道结果 SHA256 与 provenance 不一致"
                )
        object.__setattr__(self, "channels", channels)

    @property
    def red(self) -> RasterPlane:
        return self.channels[0].plane

    @property
    def green(self) -> RasterPlane:
        return self.channels[1].plane

    @property
    def blue(self) -> RasterPlane:
        return self.channels[2].plane


@dataclass(frozen=True, slots=True)
class RasterChannelMergeResult:
    plane: RasterPlane
    provenance: RasterDerivationProvenance

    def __post_init__(self) -> None:
        if (
            not isinstance(self.plane, RasterPlane)
            or self.plane.is_empty
            or self.plane.pixel_type is not RasterPixelType.RGB8
        ):
            raise RasterDerivationError("通道合并结果必须是非空 RGB8 RasterPlane")
        _require_bound_result(self.provenance, "RGB", self.plane)


def duplicate_raster_plane(
    source: RasterPlane,
    *,
    expected_source_sha256: str,
    bounds: RasterBounds | None = None,
    roi: FrozenRasterRoi | None = None,
    transparent_outside: bool = False,
    fill_value: FillValue | None = None,
) -> RasterCopyResult:
    """Copy a full plane or an immutable ROI crop without changing the source.

    A mask copy requires exactly one outside-pixel policy: transparent RGBA8,
    or an explicit fill value that preserves the source pixel type.
    """

    _validate_source_plane(source, expected_source_sha256, "源图片")
    if roi is not None and not isinstance(roi, FrozenRasterRoi):
        raise TypeError("roi 必须是 FrozenRasterRoi")
    if roi is not None:
        if bounds is not None and bounds != roi.bounds:
            raise RasterDerivationError("bounds 必须与冻结 ROI 的 bounds 完全一致")
        selected_bounds = roi.bounds
    elif bounds is not None:
        if not isinstance(bounds, RasterBounds):
            raise TypeError("bounds 必须是 RasterBounds")
        selected_bounds = bounds
    else:
        selected_bounds = RasterBounds(0, 0, source.width, source.height)
    _validate_bounds_inside_plane(selected_bounds, source)

    if roi is None:
        if transparent_outside or fill_value is not None:
            raise RasterDerivationError(
                "透明或填充值策略只能用于带掩膜的 ROI 复制"
            )
        copy_mode = "full" if bounds is None else "bounds"
    else:
        if transparent_outside and fill_value is not None:
            raise RasterDerivationError("透明输出与显式填充值不能同时使用")
        if not transparent_outside and fill_value is None:
            raise RasterDerivationError(
                "按 ROI 掩膜复制时必须选择透明输出或显式填充值"
            )
        copy_mode = "mask_transparent" if transparent_outside else "mask_fill"

    source_array = raster_plane_to_numpy(source)
    cropped = np.ascontiguousarray(
        source_array[
            selected_bounds.y:selected_bounds.bottom,
            selected_bounds.x:selected_bounds.right,
            ...,
        ]
    )
    normalized_fill: ProvenanceValue = ""
    if roi is not None:
        mask = roi.to_numpy()
        if transparent_outside:
            cropped = _mask_to_rgba(cropped, source.pixel_type, mask)
        else:
            normalized_fill = _normalize_fill_value(
                fill_value,
                source.pixel_type,
            )
            cropped = _fill_outside_mask(cropped, mask, normalized_fill)
    result_plane = numpy_to_raster_plane(cropped)
    source_sha256 = source.sha256()
    result_sha256 = result_plane.sha256()
    parameters: tuple[tuple[str, ProvenanceValue], ...] = (
        ("bounds", selected_bounds.to_tuple()),
        ("copy_mode", copy_mode),
        ("source_pixel_type", source.pixel_type.value),
        ("result_pixel_type", result_plane.pixel_type.value),
        ("transparent_outside", bool(transparent_outside)),
    )
    if roi is not None:
        parameters += (("roi_mask_sha256", roi.mask_sha256),)
    if fill_value is not None:
        parameters += (("fill_value", normalized_fill),)
    provenance = RasterDerivationProvenance(
        operation_id="duplicate_raster",
        implementation_version=1,
        sources=(("source", source_sha256),),
        results=(("copy", result_sha256),),
        parameters=parameters,
    )
    return RasterCopyResult(
        plane=result_plane,
        source_bounds=selected_bounds,
        provenance=provenance,
    )


def split_rgb_channels(
    source: RasterPlane,
    *,
    expected_source_sha256: str,
) -> RasterChannelSplitResult:
    """Split RGB(A) into immutable named R/G/B GRAY8 planes.

    RGBA Alpha is intentionally not returned as an RGB channel.
    """

    _validate_source_plane(source, expected_source_sha256, "RGB 源图片")
    if source.pixel_type not in {RasterPixelType.RGB8, RasterPixelType.RGBA8}:
        raise RasterDerivationError("通道分离仅支持 RGB8 或 RGBA8 图片")
    source_array = raster_plane_to_numpy(source)
    channel_specs = (
        ("R", "红色通道", 0),
        ("G", "绿色通道", 1),
        ("B", "蓝色通道", 2),
    )
    channels = tuple(
        NamedRasterChannel(
            channel_id=channel_id,
            display_name=display_name,
            plane=numpy_to_raster_plane(
                np.ascontiguousarray(source_array[..., index])
            ),
        )
        for channel_id, display_name, index in channel_specs
    )
    provenance = RasterDerivationProvenance(
        operation_id="split_rgb_channels",
        implementation_version=1,
        sources=(("source", source.sha256()),),
        results=tuple(
            (channel.channel_id, channel.plane.sha256()) for channel in channels
        ),
        parameters=(
            ("alpha_ignored", source.pixel_type is RasterPixelType.RGBA8),
            ("source_pixel_type", source.pixel_type.value),
        ),
    )
    return RasterChannelSplitResult(
        channels=channels,  # type: ignore[arg-type]
        provenance=provenance,
    )


def merge_gray8_channels(
    red: RasterPlane,
    green: RasterPlane,
    blue: RasterPlane,
    *,
    expected_red_sha256: str,
    expected_green_sha256: str,
    expected_blue_sha256: str,
) -> RasterChannelMergeResult:
    """Merge three GRAY8 planes after the caller validates calibration."""

    inputs = (
        ("R", red, expected_red_sha256),
        ("G", green, expected_green_sha256),
        ("B", blue, expected_blue_sha256),
    )
    for channel_id, plane, expected_sha256 in inputs:
        _validate_source_plane(plane, expected_sha256, f"{channel_id} 通道")
        if plane.pixel_type is not RasterPixelType.GRAY8:
            raise RasterDerivationError(
                f"{channel_id} 通道必须是 GRAY8 图片"
            )
    expected_size = (red.width, red.height)
    if any((plane.width, plane.height) != expected_size for _, plane, _ in inputs):
        raise RasterDerivationError("用于合并的三个通道尺寸必须完全一致")
    merged = np.stack(
        tuple(raster_plane_to_numpy(plane) for _, plane, _ in inputs),
        axis=2,
    )
    result_plane = numpy_to_raster_plane(np.ascontiguousarray(merged))
    provenance = RasterDerivationProvenance(
        operation_id="merge_rgb_channels",
        implementation_version=1,
        sources=tuple(
            (channel_id, plane.sha256()) for channel_id, plane, _ in inputs
        ),
        results=(("RGB", result_plane.sha256()),),
        parameters=(
            ("calibration_validation", "caller"),
            ("result_pixel_type", RasterPixelType.RGB8.value),
        ),
    )
    return RasterChannelMergeResult(
        plane=result_plane,
        provenance=provenance,
    )


def _validate_source_plane(
    plane: RasterPlane,
    expected_sha256: str,
    label: str,
) -> None:
    if not isinstance(plane, RasterPlane):
        raise TypeError(f"{label}必须是 RasterPlane")
    if plane.is_empty:
        raise RasterDerivationError(f"{label}不能为空")
    normalized_expected = _normalize_sha256(expected_sha256, label)
    actual_sha256 = plane.sha256()
    if normalized_expected != actual_sha256:
        raise RasterDerivationError(
            f"{label} SHA256 不一致：期望 {normalized_expected}，"
            f"实际 {actual_sha256}"
        )


def _validate_bounds_inside_plane(
    bounds: RasterBounds,
    plane: RasterPlane,
) -> None:
    if bounds.right > plane.width or bounds.bottom > plane.height:
        raise RasterDerivationError(
            "ROI 边界超出源图片范围："
            f"ROI={bounds.to_tuple()}，图片={plane.width}×{plane.height}"
        )


def _mask_to_rgba(
    cropped: np.ndarray,
    pixel_type: RasterPixelType,
    mask: np.ndarray,
) -> np.ndarray:
    if pixel_type is RasterPixelType.GRAY8:
        rgb = np.repeat(cropped[..., np.newaxis], 3, axis=2)
        alpha = np.where(mask, 255, 0).astype(np.uint8)
    elif pixel_type is RasterPixelType.RGB8:
        rgb = cropped.copy()
        alpha = np.where(mask, 255, 0).astype(np.uint8)
    elif pixel_type is RasterPixelType.RGBA8:
        rgb = cropped[..., :3].copy()
        alpha = np.where(mask, cropped[..., 3], 0).astype(np.uint8)
    else:
        raise RasterDerivationError(
            "GRAY16/GRAY32_FLOAT ROI 不能透明复制；"
            "请使用显式数值填充以保持科学像素类型"
        )
    rgb[~mask] = 0
    return np.ascontiguousarray(np.dstack((rgb, alpha)), dtype=np.uint8)


def _fill_outside_mask(
    cropped: np.ndarray,
    mask: np.ndarray,
    fill_value: ProvenanceValue,
) -> np.ndarray:
    result = cropped.copy(order="C")
    result[~mask] = fill_value
    return result


def _normalize_fill_value(
    value: FillValue | None,
    pixel_type: RasterPixelType,
) -> ProvenanceValue:
    if value is None:
        raise RasterDerivationError("ROI 外部填充值不能为空")
    if isinstance(value, (str, bytes, bytearray, bool)):
        raise RasterDerivationError("ROI 外部填充值必须是数值")
    if isinstance(value, Sequence):
        raw_values = tuple(value)
    else:
        raw_values = (value,)
    if len(raw_values) not in {1, pixel_type.channel_count}:
        raise RasterDerivationError(
            f"{pixel_type.value} 填充值必须是 1 个数或 "
            f"{pixel_type.channel_count} 个通道值"
        )
    normalized = tuple(
        _normalize_channel_fill(item, pixel_type) for item in raw_values
    )
    if len(normalized) == 1:
        return normalized[0]
    return normalized


def _normalize_channel_fill(
    value: object,
    pixel_type: RasterPixelType,
) -> int | float:
    if isinstance(value, bool):
        raise RasterDerivationError("ROI 外部填充值必须是数值")
    if pixel_type is RasterPixelType.GRAY32_FLOAT:
        try:
            normalized_float = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise RasterDerivationError("float32 填充值必须是有限数值") from exc
        if not math.isfinite(normalized_float):
            raise RasterDerivationError("float32 填充值必须是有限数值")
        as_float32 = np.float32(normalized_float)
        if not np.isfinite(as_float32):
            raise RasterDerivationError("float32 填充值超出可表示范围")
        return float(as_float32)
    try:
        normalized_int = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RasterDerivationError("整数图片的填充值必须是整数") from exc
    if normalized_int != value:
        raise RasterDerivationError("整数图片的填充值必须是整数")
    maximum = 65_535 if pixel_type is RasterPixelType.GRAY16 else 255
    if normalized_int < 0 or normalized_int > maximum:
        raise RasterDerivationError(
            f"{pixel_type.value} 填充值必须在 0–{maximum} 范围内"
        )
    return normalized_int


def _normalize_sha256(value: object, label: str) -> str:
    token = str(value or "").strip()
    if not _SHA256_PATTERN.fullmatch(token):
        raise RasterDerivationError(f"{label} SHA256 必须是 64 位十六进制")
    return token.lower()


def _roi_mask_sha256(bounds: RasterBounds, data: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(b"fdm-frozen-raster-roi-v1\0")
    for value in bounds.to_tuple():
        digest.update(value.to_bytes(8, "little", signed=False))
    digest.update(data)
    return digest.hexdigest()


def _normalize_named_hashes(
    values: Sequence[tuple[str, str]],
    label: str,
) -> tuple[tuple[str, str], ...]:
    normalized: list[tuple[str, str]] = []
    names: set[str] = set()
    for name, digest in tuple(values):
        normalized_name = str(name or "").strip()
        if not normalized_name:
            raise RasterDerivationError(f"{label}名称不能为空")
        if normalized_name in names:
            raise RasterDerivationError(f"{label}名称重复：{normalized_name}")
        names.add(normalized_name)
        normalized.append(
            (
                normalized_name,
                _normalize_sha256(digest, f"{label} {normalized_name}"),
            )
        )
    return tuple(normalized)


def _normalize_parameters(
    values: Sequence[tuple[str, ProvenanceValue]],
) -> tuple[tuple[str, ProvenanceValue], ...]:
    normalized: list[tuple[str, ProvenanceValue]] = []
    names: set[str] = set()
    for name, value in tuple(values):
        normalized_name = str(name or "").strip()
        if not normalized_name or normalized_name in names:
            raise RasterDerivationError("派生参数名称不能为空或重复")
        names.add(normalized_name)
        normalized.append((normalized_name, _normalize_parameter_value(value)))
    return tuple(normalized)


def _normalize_parameter_value(value: object) -> ProvenanceValue:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise RasterDerivationError("派生参数不能包含 NaN 或 Inf")
        return value
    if isinstance(value, str):
        return value
    if isinstance(value, tuple):
        return tuple(
            _normalize_parameter_scalar(item) for item in value
        )
    raise RasterDerivationError(
        f"不支持的派生参数类型：{type(value).__name__}"
    )


def _normalize_parameter_scalar(value: object) -> ProvenanceScalar:
    normalized = _normalize_parameter_value(value)
    if isinstance(normalized, tuple):
        raise RasterDerivationError("派生参数不支持嵌套序列")
    return normalized


def _require_bound_result(
    provenance: RasterDerivationProvenance,
    result_name: str,
    plane: RasterPlane,
) -> None:
    if not isinstance(provenance, RasterDerivationProvenance):
        raise TypeError("provenance 必须是 RasterDerivationProvenance")
    if dict(provenance.results).get(result_name) != plane.sha256():
        raise RasterDerivationError(
            f"{result_name} 结果 SHA256 与 provenance 不一致"
        )


__all__ = [
    "FillValue",
    "FrozenRasterRoi",
    "NamedRasterChannel",
    "RasterBounds",
    "RasterChannelMergeResult",
    "RasterChannelSplitResult",
    "RasterCopyResult",
    "RasterDerivationError",
    "RasterDerivationProvenance",
    "duplicate_raster_plane",
    "merge_gray8_channels",
    "split_rgb_channels",
]
