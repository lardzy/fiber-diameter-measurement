from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from fdm.raster import RasterPixelType, RasterPlane
from fdm.services.raster_derivation import (
    FrozenRasterRoi,
    RasterBounds,
    RasterChannelMergeResult,
    RasterCopyResult,
    RasterDerivationError,
    RasterDerivationProvenance,
    duplicate_raster_plane,
    merge_gray8_channels,
    split_rgb_channels,
)
from fdm.services.raster_io import numpy_to_raster_plane, raster_plane_to_numpy


def _plane(array: np.ndarray) -> RasterPlane:
    return numpy_to_raster_plane(array)


def _roi(
    mask: list[list[bool]],
    *,
    x: int = 0,
    y: int = 0,
) -> FrozenRasterRoi:
    array = np.asarray(mask, dtype=np.bool_)
    return FrozenRasterRoi.from_numpy(
        array,
        bounds=RasterBounds(x, y, int(array.shape[1]), int(array.shape[0])),
    )


def test_frozen_roi_validates_shape_bytes_hash_and_is_immutable() -> None:
    mask = np.asarray([[True, False], [False, True]], dtype=np.bool_)
    frozen = FrozenRasterRoi.from_numpy(
        mask,
        bounds=RasterBounds(3, 4, 2, 2),
    )
    mask[0, 0] = False

    assert frozen.to_numpy().tolist() == [[True, False], [False, True]]
    assert not frozen.to_numpy().flags.writeable
    assert len(frozen.mask_sha256) == 64
    with pytest.raises(FrozenInstanceError):
        frozen.mask_data = b""  # type: ignore[misc]
    with pytest.raises(RasterDerivationError, match="SHA256 不一致"):
        FrozenRasterRoi(
            bounds=frozen.bounds,
            mask_data=frozen.mask_data,
            mask_sha256="0" * 64,
        )
    with pytest.raises(RasterDerivationError, match="字节数不匹配"):
        FrozenRasterRoi(
            bounds=frozen.bounds,
            mask_data=b"\x01",
            mask_sha256="0" * 64,
        )
    with pytest.raises(RasterDerivationError, match="只能包含"):
        FrozenRasterRoi(
            bounds=RasterBounds(0, 0, 1, 1),
            mask_data=b"\x02",
            mask_sha256="0" * 64,
        )
    with pytest.raises(RasterDerivationError, match="不能为空"):
        FrozenRasterRoi.from_numpy(
            np.zeros((2, 2), dtype=np.bool_),
            bounds=RasterBounds(0, 0, 2, 2),
        )
    with pytest.raises(RasterDerivationError, match="二维 bool"):
        FrozenRasterRoi.from_numpy(
            np.ones((2, 2), dtype=np.uint8),
            bounds=RasterBounds(0, 0, 2, 2),
        )


@pytest.mark.parametrize(
    ("values", "bounds", "expected"),
    [
        (
            np.arange(20, dtype=np.uint8).reshape(4, 5),
            RasterBounds(1, 1, 3, 2),
            np.asarray([[6, 7, 8], [11, 12, 13]], dtype=np.uint8),
        ),
        (
            np.arange(60, dtype=np.uint8).reshape(4, 5, 3),
            RasterBounds(2, 0, 2, 3),
            np.arange(60, dtype=np.uint8).reshape(4, 5, 3)[:3, 2:4],
        ),
    ],
)
def test_duplicate_full_or_bounds_is_exact(
    values: np.ndarray,
    bounds: RasterBounds,
    expected: np.ndarray,
) -> None:
    source = _plane(values)
    full = duplicate_raster_plane(
        source,
        expected_source_sha256=source.sha256().upper(),
    )
    cropped = duplicate_raster_plane(
        source,
        expected_source_sha256=source.sha256(),
        bounds=bounds,
    )

    assert full.plane is not source
    assert full.plane == source
    assert full.source_bounds == RasterBounds(0, 0, source.width, source.height)
    assert full.provenance.parameter_map["copy_mode"] == "full"
    assert np.array_equal(raster_plane_to_numpy(cropped.plane), expected)
    assert cropped.source_bounds == bounds
    assert cropped.provenance.parameter_map["copy_mode"] == "bounds"
    assert (
        dict(cropped.provenance.results)["copy"]
        == cropped.plane.sha256()
    )


def test_duplicate_mask_with_explicit_fill_preserves_pixel_type() -> None:
    source_array = np.asarray(
        [
            [100, 101, 102, 103],
            [200, 201, 202, 203],
            [300, 301, 302, 303],
        ],
        dtype=np.uint16,
    )
    source = _plane(source_array)
    roi = _roi(
        [[True, False, True], [False, True, False]],
        x=1,
        y=1,
    )
    result = duplicate_raster_plane(
        source,
        expected_source_sha256=source.sha256(),
        roi=roi,
        fill_value=65_000,
    )

    assert result.plane.pixel_type is RasterPixelType.GRAY16
    assert raster_plane_to_numpy(result.plane).tolist() == [
        [201, 65_000, 203],
        [65_000, 302, 65_000],
    ]
    assert result.provenance.parameter_map["roi_mask_sha256"] == roi.mask_sha256
    assert result.provenance.parameter_map["fill_value"] == 65_000


def test_duplicate_mask_with_channel_fill_and_float_fill() -> None:
    rgb_array = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    rgb = _plane(rgb_array)
    rgb_roi = _roi([[True, False], [False, True]], x=1, y=0)
    rgb_result = duplicate_raster_plane(
        rgb,
        expected_source_sha256=rgb.sha256(),
        roi=rgb_roi,
        fill_value=(10, 20, 30),
    )
    expected_rgb = rgb_array[:, 1:3].copy()
    expected_rgb[~rgb_roi.to_numpy()] = (10, 20, 30)
    assert np.array_equal(raster_plane_to_numpy(rgb_result.plane), expected_rgb)

    float_source = _plane(
        np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    )
    float_roi = _roi([[True, False], [False, True]])
    float_result = duplicate_raster_plane(
        float_source,
        expected_source_sha256=float_source.sha256(),
        roi=float_roi,
        fill_value=-1.25,
    )
    assert float_result.plane.pixel_type is RasterPixelType.GRAY32_FLOAT
    assert raster_plane_to_numpy(float_result.plane).tolist() == [
        [1.0, -1.25],
        [-1.25, 4.0],
    ]


@pytest.mark.parametrize(
    ("array", "expected"),
    [
        (
            np.asarray([[10, 20], [30, 40]], dtype=np.uint8),
            np.asarray(
                [
                    [[10, 10, 10, 255], [0, 0, 0, 0]],
                    [[0, 0, 0, 0], [40, 40, 40, 255]],
                ],
                dtype=np.uint8,
            ),
        ),
        (
            np.asarray(
                [
                    [[1, 2, 3], [4, 5, 6]],
                    [[7, 8, 9], [10, 11, 12]],
                ],
                dtype=np.uint8,
            ),
            np.asarray(
                [
                    [[1, 2, 3, 255], [0, 0, 0, 0]],
                    [[0, 0, 0, 0], [10, 11, 12, 255]],
                ],
                dtype=np.uint8,
            ),
        ),
        (
            np.asarray(
                [
                    [[1, 2, 3, 44], [4, 5, 6, 55]],
                    [[7, 8, 9, 66], [10, 11, 12, 77]],
                ],
                dtype=np.uint8,
            ),
            np.asarray(
                [
                    [[1, 2, 3, 44], [0, 0, 0, 0]],
                    [[0, 0, 0, 0], [10, 11, 12, 77]],
                ],
                dtype=np.uint8,
            ),
        ),
    ],
)
def test_duplicate_mask_transparent_output_is_rgba8(
    array: np.ndarray,
    expected: np.ndarray,
) -> None:
    source = _plane(array)
    roi = _roi([[True, False], [False, True]])
    result = duplicate_raster_plane(
        source,
        expected_source_sha256=source.sha256(),
        roi=roi,
        transparent_outside=True,
    )

    assert result.plane.pixel_type is RasterPixelType.RGBA8
    assert np.array_equal(raster_plane_to_numpy(result.plane), expected)
    assert result.provenance.parameter_map["copy_mode"] == "mask_transparent"


def test_duplicate_rejects_ambiguous_or_unsafe_requests() -> None:
    source = _plane(np.arange(16, dtype=np.uint8).reshape(4, 4))
    roi = _roi([[True, False], [False, True]], x=1, y=1)

    with pytest.raises(RasterDerivationError, match="SHA256 不一致"):
        duplicate_raster_plane(
            source,
            expected_source_sha256="0" * 64,
        )
    with pytest.raises(RasterDerivationError, match="64 位"):
        duplicate_raster_plane(
            source,
            expected_source_sha256="bad",
        )
    with pytest.raises(RasterDerivationError, match="必须选择"):
        duplicate_raster_plane(
            source,
            expected_source_sha256=source.sha256(),
            roi=roi,
        )
    with pytest.raises(RasterDerivationError, match="不能同时"):
        duplicate_raster_plane(
            source,
            expected_source_sha256=source.sha256(),
            roi=roi,
            transparent_outside=True,
            fill_value=0,
        )
    with pytest.raises(RasterDerivationError, match="只能用于带掩膜"):
        duplicate_raster_plane(
            source,
            expected_source_sha256=source.sha256(),
            bounds=RasterBounds(0, 0, 2, 2),
            fill_value=0,
        )
    with pytest.raises(RasterDerivationError, match="超出源图片"):
        duplicate_raster_plane(
            source,
            expected_source_sha256=source.sha256(),
            bounds=RasterBounds(3, 3, 2, 2),
        )
    with pytest.raises(RasterDerivationError, match="完全一致"):
        duplicate_raster_plane(
            source,
            expected_source_sha256=source.sha256(),
            bounds=RasterBounds(0, 0, 2, 2),
            roi=roi,
            fill_value=0,
        )

    gray16 = _plane(np.ones((2, 2), dtype=np.uint16))
    with pytest.raises(RasterDerivationError, match="不能透明复制"):
        duplicate_raster_plane(
            gray16,
            expected_source_sha256=gray16.sha256(),
            roi=_roi([[True, False], [False, True]]),
            transparent_outside=True,
        )


@pytest.mark.parametrize(
    ("fill_value", "message"),
    [
        (-1, "0–255"),
        (256, "0–255"),
        (1.5, "必须是整数"),
        (float("nan"), "必须是整数"),
        ((1, 2), "1 个数或 3 个"),
    ],
)
def test_duplicate_validates_fill_values(
    fill_value: object,
    message: str,
) -> None:
    source = _plane(np.zeros((2, 2, 3), dtype=np.uint8))
    with pytest.raises(RasterDerivationError, match=message):
        duplicate_raster_plane(
            source,
            expected_source_sha256=source.sha256(),
            roi=_roi([[True, False], [False, True]]),
            fill_value=fill_value,  # type: ignore[arg-type]
        )


def test_split_rgb_and_rgba_produces_named_gray8_channels() -> None:
    rgb_array = np.asarray(
        [
            [[1, 11, 21], [2, 12, 22]],
            [[3, 13, 23], [4, 14, 24]],
        ],
        dtype=np.uint8,
    )
    rgb = _plane(rgb_array)
    split = split_rgb_channels(
        rgb,
        expected_source_sha256=rgb.sha256(),
    )

    assert tuple(item.channel_id for item in split.channels) == ("R", "G", "B")
    assert tuple(item.display_name for item in split.channels) == (
        "红色通道",
        "绿色通道",
        "蓝色通道",
    )
    assert np.array_equal(raster_plane_to_numpy(split.red), rgb_array[..., 0])
    assert np.array_equal(raster_plane_to_numpy(split.green), rgb_array[..., 1])
    assert np.array_equal(raster_plane_to_numpy(split.blue), rgb_array[..., 2])
    assert split.provenance.parameter_map["alpha_ignored"] is False

    rgba_array = np.dstack(
        (rgb_array, np.asarray([[0, 1], [2, 3]], dtype=np.uint8))
    )
    rgba = _plane(rgba_array)
    rgba_split = split_rgb_channels(
        rgba,
        expected_source_sha256=rgba.sha256(),
    )
    assert raster_plane_to_numpy(rgba_split.red).tolist() == [[1, 2], [3, 4]]
    assert rgba_split.provenance.parameter_map["alpha_ignored"] is True


def test_split_rejects_wrong_type_empty_and_sha() -> None:
    gray = _plane(np.ones((2, 2), dtype=np.uint8))
    with pytest.raises(RasterDerivationError, match="仅支持 RGB8"):
        split_rgb_channels(gray, expected_source_sha256=gray.sha256())

    rgb = _plane(np.ones((2, 2, 3), dtype=np.uint8))
    with pytest.raises(RasterDerivationError, match="SHA256 不一致"):
        split_rgb_channels(rgb, expected_source_sha256="0" * 64)

    empty = RasterPlane(
        width=0,
        height=0,
        pixel_type=RasterPixelType.RGB8,
        data=b"",
    )
    with pytest.raises(RasterDerivationError, match="不能为空"):
        split_rgb_channels(empty, expected_source_sha256=empty.sha256())


def test_merge_gray8_channels_is_exact_and_records_sources() -> None:
    red = _plane(np.asarray([[1, 2], [3, 4]], dtype=np.uint8))
    green = _plane(np.asarray([[11, 12], [13, 14]], dtype=np.uint8))
    blue = _plane(np.asarray([[21, 22], [23, 24]], dtype=np.uint8))
    merged = merge_gray8_channels(
        red,
        green,
        blue,
        expected_red_sha256=red.sha256(),
        expected_green_sha256=green.sha256(),
        expected_blue_sha256=blue.sha256(),
    )

    assert merged.plane.pixel_type is RasterPixelType.RGB8
    assert raster_plane_to_numpy(merged.plane).tolist() == [
        [[1, 11, 21], [2, 12, 22]],
        [[3, 13, 23], [4, 14, 24]],
    ]
    assert dict(merged.provenance.sources) == {
        "R": red.sha256(),
        "G": green.sha256(),
        "B": blue.sha256(),
    }
    assert merged.provenance.parameter_map["calibration_validation"] == "caller"


def test_merge_rejects_type_size_empty_and_sha_mismatch() -> None:
    channel = _plane(np.ones((2, 2), dtype=np.uint8))
    wrong_size = _plane(np.ones((2, 3), dtype=np.uint8))
    wrong_type = _plane(np.ones((2, 2), dtype=np.uint16))
    empty = RasterPlane(
        width=0,
        height=0,
        pixel_type=RasterPixelType.GRAY8,
        data=b"",
    )

    with pytest.raises(RasterDerivationError, match="尺寸必须完全一致"):
        merge_gray8_channels(
            channel,
            wrong_size,
            channel,
            expected_red_sha256=channel.sha256(),
            expected_green_sha256=wrong_size.sha256(),
            expected_blue_sha256=channel.sha256(),
        )
    with pytest.raises(RasterDerivationError, match="必须是 GRAY8"):
        merge_gray8_channels(
            channel,
            wrong_type,
            channel,
            expected_red_sha256=channel.sha256(),
            expected_green_sha256=wrong_type.sha256(),
            expected_blue_sha256=channel.sha256(),
        )
    with pytest.raises(RasterDerivationError, match="不能为空"):
        merge_gray8_channels(
            channel,
            empty,
            channel,
            expected_red_sha256=channel.sha256(),
            expected_green_sha256=empty.sha256(),
            expected_blue_sha256=channel.sha256(),
        )
    with pytest.raises(RasterDerivationError, match="SHA256 不一致"):
        merge_gray8_channels(
            channel,
            channel,
            channel,
            expected_red_sha256="0" * 64,
            expected_green_sha256=channel.sha256(),
            expected_blue_sha256=channel.sha256(),
        )


def test_result_objects_refuse_provenance_hash_mismatch() -> None:
    plane = _plane(np.ones((1, 1), dtype=np.uint8))
    wrong = RasterDerivationProvenance(
        operation_id="test",
        implementation_version=1,
        sources=(("source", plane.sha256()),),
        results=(("copy", "0" * 64),),
    )
    with pytest.raises(RasterDerivationError, match="provenance 不一致"):
        RasterCopyResult(
            plane=plane,
            source_bounds=RasterBounds(0, 0, 1, 1),
            provenance=wrong,
        )

    rgb = _plane(np.ones((1, 1, 3), dtype=np.uint8))
    wrong_rgb = RasterDerivationProvenance(
        operation_id="test",
        implementation_version=1,
        sources=(("R", plane.sha256()),),
        results=(("RGB", "0" * 64),),
    )
    with pytest.raises(RasterDerivationError, match="provenance 不一致"):
        RasterChannelMergeResult(plane=rgb, provenance=wrong_rgb)
