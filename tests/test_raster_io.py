from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image
from PySide6.QtGui import QImage
import pytest
import tifffile

from fdm.raster import RasterPixelType
from fdm.services.raster_io import (
    RasterIoError,
    RasterIoFailureCode,
    RasterMetadata,
    NativeTiffCompression,
    numpy_to_raster_plane,
    qimage_to_raster_plane,
    raster_plane_to_numpy,
    raster_plane_to_qimage,
    read_raster_file,
    recommended_native_asset_suffix,
    write_native_raster_asset,
)
from fdm.ui.image_loader import ImageBatchLoaderWorker, ImageLoadRequest


@pytest.mark.parametrize(
    ("array", "pixel_type"),
    [
        (np.arange(20, dtype=np.uint8).reshape(4, 5), RasterPixelType.GRAY8),
        (
            (np.arange(20, dtype=np.uint16).reshape(4, 5) * 1_007),
            RasterPixelType.GRAY16,
        ),
        (
            np.array([[1.25, np.nan], [np.inf, -3.5]], dtype=np.float32),
            RasterPixelType.GRAY32_FLOAT,
        ),
        (
            np.arange(60, dtype=np.uint8).reshape(4, 5, 3),
            RasterPixelType.RGB8,
        ),
        (
            np.arange(80, dtype=np.uint8).reshape(4, 5, 4),
            RasterPixelType.RGBA8,
        ),
    ],
)
def test_numpy_plane_round_trip_is_exact_and_read_only(
    array: np.ndarray,
    pixel_type: RasterPixelType,
) -> None:
    plane = numpy_to_raster_plane(array)

    assert plane.pixel_type is pixel_type
    assert plane.width == array.shape[1]
    assert plane.height == array.shape[0]
    restored = raster_plane_to_numpy(plane)
    assert not restored.flags.writeable
    assert restored.dtype == array.dtype
    assert restored.tobytes() == np.ascontiguousarray(array).tobytes()

    writable = raster_plane_to_numpy(plane, writable=True)
    assert writable.flags.writeable
    writable.flat[0] = 0
    assert plane.data == np.ascontiguousarray(array).tobytes()


def test_gray32_float_pixel_contract_has_no_integer_maximum() -> None:
    assert RasterPixelType.GRAY32_FLOAT.bytes_per_channel == 4
    assert RasterPixelType.GRAY32_FLOAT.bytes_per_pixel == 4
    assert RasterPixelType.GRAY32_FLOAT.channel_count == 1
    assert RasterPixelType.GRAY32_FLOAT.is_grayscale
    assert RasterPixelType.GRAY32_FLOAT.sample_maximum is None


def test_batch_loader_keeps_native_gray16_plane_beside_display_cache(
    tmp_path: Path,
) -> None:
    target = tmp_path / "高位深.tif"
    source = (
        np.arange(35, dtype=np.uint16).reshape(5, 7) * 1_337
    )
    tifffile.imwrite(target, source, metadata=None)
    request = ImageLoadRequest(path=str(target))
    worker = ImageBatchLoaderWorker([request])
    loaded: list[tuple[ImageLoadRequest, QImage]] = []
    worker.loaded.connect(
        lambda item, image: loaded.append((item, image))
    )

    worker.run()

    assert len(loaded) == 1
    loaded_request, display = loaded[0]
    assert loaded_request.raster_plane is not None
    assert (
        loaded_request.raster_plane.pixel_type
        is RasterPixelType.GRAY16
    )
    assert loaded_request.raster_plane.sha256() == (
        numpy_to_raster_plane(source).sha256()
    )
    assert not display.isNull()
    assert display.format() == QImage.Format.Format_Grayscale8


def test_constant_float_plane_builds_deterministic_mid_gray_display() -> None:
    plane = numpy_to_raster_plane(
        np.full((3, 4), 7.25, dtype=np.float32)
    )

    display = raster_plane_to_qimage(plane)

    assert not display.isNull()
    assert {
        display.pixelColor(x, y).red()
        for y in range(display.height())
        for x in range(display.width())
    } == {128}


def test_batch_loader_reports_display_failure_and_always_finishes(
    tmp_path: Path,
) -> None:
    target = tmp_path / "display-failure.png"
    Image.new("L", (3, 2), 128).save(target)
    worker = ImageBatchLoaderWorker([ImageLoadRequest(path=str(target))])
    failures: list[tuple[str, str]] = []
    finished: list[tuple[bool, int, int, int]] = []
    worker.failed.connect(lambda path, reason: failures.append((path, reason)))
    worker.finished.connect(
        lambda *payload: finished.append(tuple(payload))
    )

    with patch(
        "fdm.ui.image_loader.raster_plane_to_qimage",
        side_effect=ValueError("injected display error"),
    ):
        worker.run()

    assert len(failures) == 1
    assert "injected display error" in failures[0][1]
    assert finished == [(False, 0, 0, 1)]


def test_batch_loader_does_not_emit_success_after_conversion_cancellation(
    tmp_path: Path,
) -> None:
    target = tmp_path / "cancel-after-conversion.png"
    Image.new("L", (3, 2), 128).save(target)
    request = ImageLoadRequest(path=str(target))
    worker = ImageBatchLoaderWorker([request])
    loaded: list[object] = []
    finished: list[tuple[bool, int, int, int]] = []
    real_converter = raster_plane_to_qimage

    def cancel_after_conversion(plane):
        image = real_converter(plane)
        worker.request_cancel()
        return image

    worker.loaded.connect(lambda *_args: loaded.append(object()))
    worker.finished.connect(
        lambda *payload: finished.append(tuple(payload))
    )
    with patch(
        "fdm.ui.image_loader.raster_plane_to_qimage",
        side_effect=cancel_after_conversion,
    ):
        worker.run()

    assert loaded == []
    assert request.raster_plane is None
    assert finished == [(True, 0, 0, 0)]


def test_batch_loader_rejects_project_asset_dimension_mismatch(
    tmp_path: Path,
) -> None:
    target = tmp_path / "wrong-size.png"
    Image.new("L", (3, 2), 128).save(target)
    document = SimpleNamespace(
        raster_pixel_type=RasterPixelType.GRAY8,
        image_size=(30, 20),
    )
    worker = ImageBatchLoaderWorker(
        [ImageLoadRequest(path=str(target), document=document)]
    )
    failures: list[str] = []
    loaded: list[object] = []
    worker.failed.connect(lambda _path, reason: failures.append(reason))
    worker.loaded.connect(lambda *_args: loaded.append(object()))

    worker.run()

    assert loaded == []
    assert len(failures) == 1
    assert "图片尺寸" in failures[0]


@pytest.mark.parametrize(
    ("suffix", "mode", "pixel_type"),
    [
        (".png", "L", RasterPixelType.GRAY8),
        (".png", "RGB", RasterPixelType.RGB8),
        (".png", "RGBA", RasterPixelType.RGBA8),
        (".jpg", "RGB", RasterPixelType.RGB8),
        (".bmp", "RGB", RasterPixelType.RGB8),
        (".webp", "RGBA", RasterPixelType.RGBA8),
    ],
)
def test_reads_common_pillow_formats(
    tmp_path: Path,
    suffix: str,
    mode: str,
    pixel_type: RasterPixelType,
) -> None:
    channels = {"L": 1, "RGB": 3, "RGBA": 4}[mode]
    shape = (7, 9) if channels == 1 else (7, 9, channels)
    array = np.arange(np.prod(shape), dtype=np.uint8).reshape(shape)
    target = tmp_path / f"中文 文件{suffix}"
    image = Image.fromarray(array, mode=mode)
    kwargs = {"lossless": True} if suffix == ".webp" else {}
    image.save(target, **kwargs)
    image.close()

    result = read_raster_file(target).require_success()

    assert result.plane is not None
    assert result.plane.pixel_type is pixel_type
    assert (result.plane.width, result.plane.height) == (9, 7)
    assert result.metadata is not None
    assert result.metadata.source_format


def test_palette_transparency_is_preserved_as_rgba(tmp_path: Path) -> None:
    target = tmp_path / "palette.png"
    image = Image.new("P", (3, 2))
    palette = [255, 0, 0, 0, 255, 0] + [0] * (768 - 6)
    image.putpalette(palette)
    image.putdata([0, 1, 0, 1, 0, 1])
    image.info["transparency"] = bytes([255, 0])
    image.save(target)
    image.close()

    result = read_raster_file(target).require_success()

    assert result.plane is not None
    assert result.plane.pixel_type is RasterPixelType.RGBA8
    alpha = raster_plane_to_numpy(result.plane)[:, :, 3]
    assert set(alpha.ravel()) == {0, 255}


def test_png_transparency_key_is_preserved_as_rgba(tmp_path: Path) -> None:
    target = tmp_path / "transparent-rgb.png"
    image = Image.new("RGB", (2, 1), (255, 0, 0))
    image.putpixel((1, 0), (0, 255, 0))
    image.save(target, transparency=(255, 0, 0))
    image.close()

    result = read_raster_file(target).require_success()

    assert result.plane is not None
    assert result.plane.pixel_type is RasterPixelType.RGBA8
    restored = raster_plane_to_numpy(result.plane)
    assert restored[0, 0].tolist() == [255, 0, 0, 0]
    assert restored[0, 1].tolist() == [0, 255, 0, 255]


@pytest.mark.parametrize(
    ("mode", "suffix", "pixel_type"),
    [
        ("1", ".png", RasterPixelType.GRAY8),
        ("LA", ".png", RasterPixelType.RGBA8),
        ("CMYK", ".jpg", RasterPixelType.RGB8),
    ],
)
def test_common_pillow_modes_are_mapped_without_losing_source_mode(
    tmp_path: Path,
    mode: str,
    suffix: str,
    pixel_type: RasterPixelType,
) -> None:
    target = tmp_path / f"{mode.replace(';', '_')}{suffix}"
    image = Image.new(mode, (4, 3))
    image.save(target)
    image.close()

    result = read_raster_file(target).require_success()

    assert result.plane is not None
    assert result.plane.pixel_type is pixel_type
    assert result.metadata is not None
    assert result.metadata.source_mode == mode


def test_pillow_orientation_icc_and_dpi_metadata_are_normalized(
    tmp_path: Path,
) -> None:
    target = tmp_path / "oriented.png"
    source = np.zeros((2, 3, 3), dtype=np.uint8)
    source[0, 0] = (255, 0, 0)
    source[1, 2] = (0, 255, 0)
    image = Image.fromarray(source, mode="RGB")
    exif = Image.Exif()
    exif[274] = 6
    image.save(
        target,
        exif=exif,
        icc_profile=b"fdm-test-profile",
        dpi=(120.0, 96.0),
    )
    image.close()

    result = read_raster_file(target).require_success()

    assert result.plane is not None
    assert (result.plane.width, result.plane.height) == (2, 3)
    assert result.metadata is not None
    assert result.metadata.source_orientation == 6
    assert result.metadata.orientation_applied
    assert result.metadata.icc_profile == b"fdm-test-profile"
    assert result.metadata.dpi_x == pytest.approx(120.0, abs=0.1)
    assert result.metadata.dpi_y == pytest.approx(96.0, abs=0.1)
    oriented = raster_plane_to_numpy(result.plane)
    assert tuple(oriented[0, 1]) == (255, 0, 0)
    assert tuple(oriented[2, 0]) == (0, 255, 0)


@pytest.mark.parametrize(
    "array",
    [
        np.arange(24, dtype=np.uint8).reshape(4, 6),
        np.arange(24, dtype=np.uint16).reshape(4, 6) * 2_503,
        np.array(
            [[0.0, -1.5, np.nan], [np.inf, -np.inf, 2.25]],
            dtype=np.float32,
        ),
        np.arange(72, dtype=np.uint8).reshape(4, 6, 3),
        np.arange(96, dtype=np.uint8).reshape(4, 6, 4),
    ],
)
def test_tiff_single_plane_read_is_bit_exact(
    tmp_path: Path,
    array: np.ndarray,
) -> None:
    target = tmp_path / "native.tif"
    kwargs: dict[str, object] = {"metadata": None}
    if array.ndim == 3:
        kwargs["photometric"] = "rgb"
    tifffile.imwrite(target, array, **kwargs)

    result = read_raster_file(target).require_success()

    assert result.plane is not None
    expected = numpy_to_raster_plane(array)
    assert result.plane.pixel_type is expected.pixel_type
    assert result.plane.sha256() == expected.sha256()


def test_tiff_orientation_is_applied_once(tmp_path: Path) -> None:
    target = tmp_path / "orientation.tif"
    source = np.arange(6, dtype=np.uint8).reshape(2, 3)
    tifffile.imwrite(
        target,
        source,
        metadata=None,
        extratags=[(274, "H", 1, 6, False)],
    )

    result = read_raster_file(target).require_success()

    assert result.plane is not None
    assert (result.plane.width, result.plane.height) == (2, 3)
    assert np.array_equal(
        raster_plane_to_numpy(result.plane),
        np.rot90(source, k=3),
    )
    assert result.metadata is not None
    assert result.metadata.orientation_applied


def test_palette_tiff_is_expanded_to_rgb_pixels(tmp_path: Path) -> None:
    target = tmp_path / "palette.tif"
    indices = np.array([[0, 1]], dtype=np.uint8)
    color_map = np.zeros((3, 256), dtype=np.uint16)
    color_map[:, 0] = (65_535, 0, 0)
    color_map[:, 1] = (0, 65_535, 0)
    tifffile.imwrite(
        target,
        indices,
        photometric="palette",
        colormap=color_map,
        metadata=None,
    )

    result = read_raster_file(target).require_success()

    assert result.plane is not None
    assert result.plane.pixel_type is RasterPixelType.RGB8
    assert raster_plane_to_numpy(result.plane).tolist() == [
        [[255, 0, 0], [0, 255, 0]]
    ]
    assert result.metadata is not None
    assert result.metadata.source_photometric == "PALETTE"
    assert result.metadata.photometric_applied


def test_miniswhite_tiff_is_normalized_to_internal_black_zero(
    tmp_path: Path,
) -> None:
    target = tmp_path / "white-is-zero.tif"
    source = np.array([[0, 64, 255]], dtype=np.uint8)
    tifffile.imwrite(
        target,
        source,
        photometric="miniswhite",
        metadata=None,
    )

    result = read_raster_file(target).require_success()

    assert result.plane is not None
    assert raster_plane_to_numpy(result.plane).tolist() == [[255, 191, 0]]
    assert result.metadata is not None
    assert result.metadata.source_photometric == "MINISWHITE"
    assert result.metadata.photometric_applied


@pytest.mark.parametrize(
    ("orientation", "transpose"),
    [
        (5, Image.Transpose.TRANSPOSE),
        (7, Image.Transpose.TRANSVERSE),
    ],
)
def test_tiff_mirrored_transpose_orientations_match_exif_definition(
    tmp_path: Path,
    orientation: int,
    transpose: Image.Transpose,
) -> None:
    target = tmp_path / f"orientation-{orientation}.tif"
    source = np.arange(12, dtype=np.uint8).reshape(3, 4)
    tifffile.imwrite(
        target,
        source,
        metadata=None,
        extratags=[(274, "H", 1, orientation, False)],
    )
    expected_image = Image.fromarray(source).transpose(transpose)
    expected = np.asarray(expected_image).copy()
    expected_image.close()

    result = read_raster_file(target).require_success()

    assert result.plane is not None
    assert np.array_equal(raster_plane_to_numpy(result.plane), expected)


def test_multiframe_and_unsupported_tiff_are_structured_failures(
    tmp_path: Path,
) -> None:
    stack_path = tmp_path / "stack.tif"
    with tifffile.TiffWriter(stack_path) as writer:
        writer.write(np.zeros((3, 3), dtype=np.uint8), metadata=None)
        writer.write(np.ones((3, 3), dtype=np.uint8), metadata=None)
    rgb16_path = tmp_path / "rgb16.tif"
    tifffile.imwrite(
        rgb16_path,
        np.zeros((3, 3, 3), dtype=np.uint16),
        photometric="rgb",
        metadata=None,
    )

    stack_result = read_raster_file(stack_path)
    rgb16_result = read_raster_file(rgb16_path)

    assert not stack_result
    assert stack_result.failure is not None
    assert stack_result.failure.code is RasterIoFailureCode.UNSUPPORTED_PIXEL_TYPE
    assert not rgb16_result
    assert rgb16_result.failure is not None
    assert rgb16_result.failure.code is RasterIoFailureCode.UNSUPPORTED_PIXEL_TYPE


@pytest.mark.parametrize(
    ("array", "suffix"),
    [
        (np.arange(20, dtype=np.uint8).reshape(4, 5), ".png"),
        (np.arange(20, dtype=np.uint16).reshape(4, 5) * 3_000, ".png"),
        (
            np.array([[0.5, np.nan], [np.inf, -4.0]], dtype=np.float32),
            ".tif",
        ),
        (np.arange(60, dtype=np.uint8).reshape(4, 5, 3), ".png"),
        (np.arange(80, dtype=np.uint8).reshape(4, 5, 4), ".png"),
    ],
)
def test_native_asset_write_verifies_exact_pixels(
    tmp_path: Path,
    array: np.ndarray,
    suffix: str,
) -> None:
    target = tmp_path / f"asset{suffix}"
    plane = numpy_to_raster_plane(array)
    metadata = RasterMetadata(
        icc_profile=b"test-icc",
        dpi_x=96.0,
        dpi_y=120.0,
    )

    result = write_native_raster_asset(
        plane,
        target,
        metadata=metadata,
    ).require_success()

    assert result.bytes_written == target.stat().st_size
    assert result.pixel_sha256 == plane.sha256()
    restored = read_raster_file(target).require_success()
    assert restored.plane is not None
    assert restored.plane.sha256() == plane.sha256()
    assert restored.metadata is not None
    assert restored.metadata.icc_profile == b"test-icc"
    assert restored.metadata.dpi_x == pytest.approx(96.0, abs=0.1)
    assert restored.metadata.dpi_y == pytest.approx(120.0, abs=0.1)


def test_float_png_is_rejected_without_touching_existing_target(
    tmp_path: Path,
) -> None:
    target = tmp_path / "asset.png"
    target.write_bytes(b"old asset")
    plane = numpy_to_raster_plane(np.ones((2, 2), dtype=np.float32))

    result = write_native_raster_asset(plane, target)

    assert not result
    assert result.failure is not None
    assert result.failure.code is RasterIoFailureCode.UNSUPPORTED_PIXEL_TYPE
    assert target.read_bytes() == b"old asset"


def test_atomic_commit_failure_preserves_existing_target(tmp_path: Path) -> None:
    target = tmp_path / "asset.png"
    target.write_bytes(b"old asset")
    plane = numpy_to_raster_plane(np.arange(12, dtype=np.uint8).reshape(3, 4))

    with patch(
        "fdm.services.raster_io.atomic_replace_file",
        side_effect=OSError("injected replace failure"),
    ):
        result = write_native_raster_asset(plane, target)

    assert not result
    assert result.failure is not None
    assert result.failure.code is RasterIoFailureCode.ATOMIC_COMMIT_FAILED
    assert target.read_bytes() == b"old asset"


def test_png_compression_option_is_validated_and_applied(tmp_path: Path) -> None:
    plane = numpy_to_raster_plane(np.zeros((128, 128), dtype=np.uint8))
    uncompressed = tmp_path / "level-0.png"
    compressed = tmp_path / "level-9.png"

    assert write_native_raster_asset(
        plane,
        uncompressed,
        png_compression=0,
    ).success
    assert write_native_raster_asset(
        plane,
        compressed,
        png_compression=9,
    ).success

    assert compressed.stat().st_size < uncompressed.stat().st_size
    invalid_target = tmp_path / "invalid.png"
    invalid_target.write_bytes(b"old")
    invalid = write_native_raster_asset(
        plane,
        invalid_target,
        png_compression=10,
    )
    assert not invalid
    assert invalid.failure is not None
    assert invalid.failure.code is RasterIoFailureCode.INVALID_OPTIONS
    assert invalid_target.read_bytes() == b"old"


@pytest.mark.parametrize(
    "compression",
    [
        NativeTiffCompression.DEFLATE,
        NativeTiffCompression.LZW,
        NativeTiffCompression.NONE,
        "zip",
        "uncompressed",
    ],
)
def test_tiff_compression_options_round_trip_exactly(
    tmp_path: Path,
    compression: NativeTiffCompression | str,
) -> None:
    array = np.arange(256, dtype=np.float32).reshape(16, 16) / 7.0
    plane = numpy_to_raster_plane(array)
    target = tmp_path / f"{str(compression).replace('/', '-')}.tif"

    result = write_native_raster_asset(
        plane,
        target,
        tiff_compression=compression,
    ).require_success()

    assert result.bytes_written > 0
    restored = read_raster_file(target).require_success()
    assert restored.plane is not None
    assert restored.plane.sha256() == plane.sha256()


def test_invalid_tiff_compression_does_not_touch_target(tmp_path: Path) -> None:
    target = tmp_path / "invalid.tif"
    target.write_bytes(b"old")
    plane = numpy_to_raster_plane(np.ones((3, 4), dtype=np.uint16))

    result = write_native_raster_asset(
        plane,
        target,
        tiff_compression="lossy-magic",
    )

    assert not result
    assert result.failure is not None
    assert result.failure.code is RasterIoFailureCode.INVALID_OPTIONS
    assert target.read_bytes() == b"old"


def test_corrupt_and_unsupported_files_return_chinese_structured_failure(
    tmp_path: Path,
) -> None:
    corrupt = tmp_path / "损坏.png"
    corrupt.write_bytes(b"not an image")
    unsupported = tmp_path / "sample.xyz"
    unsupported.write_bytes(b"payload")

    corrupt_result = read_raster_file(corrupt)
    unsupported_result = read_raster_file(unsupported)

    assert corrupt_result.failure is not None
    assert corrupt_result.failure.code is RasterIoFailureCode.DECODE_FAILED
    assert "无法解码" in corrupt_result.failure.message
    decoded = json.loads(corrupt_result.failure.to_json())
    assert decoded["path"].endswith("损坏.png")
    assert unsupported_result.failure is not None
    assert unsupported_result.failure.code is RasterIoFailureCode.UNSUPPORTED_FORMAT
    with pytest.raises(RasterIoError):
        corrupt_result.require_success()


def test_metadata_rejects_nonfinite_json_values() -> None:
    with pytest.raises(ValueError, match="正有限数"):
        RasterMetadata(dpi_x=float("nan"))
    metadata = RasterMetadata(dpi_x=96.0, dpi_y=120.0)
    payload = metadata.to_json()
    assert "NaN" not in payload
    assert "Infinity" not in payload


def test_qimage_round_trip_for_gray8_rgb_and_rgba() -> None:
    for array in (
        np.arange(20, dtype=np.uint8).reshape(4, 5),
        np.arange(60, dtype=np.uint8).reshape(4, 5, 3),
        np.arange(80, dtype=np.uint8).reshape(4, 5, 4),
    ):
        plane = numpy_to_raster_plane(array)
        image = raster_plane_to_qimage(plane)
        restored = qimage_to_raster_plane(image)
        assert restored.pixel_type is plane.pixel_type
        assert restored.sha256() == plane.sha256()


def test_qimage_grayscale16_is_preserved() -> None:
    array = np.arange(20, dtype=np.uint16).reshape(4, 5) * 2_000
    contiguous = np.ascontiguousarray(array)
    image = QImage(
        contiguous.data,
        5,
        4,
        int(contiguous.strides[0]),
        QImage.Format.Format_Grayscale16,
    ).copy()

    plane = qimage_to_raster_plane(image)

    assert plane.pixel_type is RasterPixelType.GRAY16
    assert raster_plane_to_numpy(plane).tobytes() == array.tobytes()


def test_gray16_and_float_display_mapping_does_not_modify_source() -> None:
    gray16 = numpy_to_raster_plane(
        np.array([[0, 32_768, 65_535]], dtype=np.uint16)
    )
    float_plane = numpy_to_raster_plane(
        np.array([[0.0, 0.5, 1.0, np.nan]], dtype=np.float32)
    )

    gray_image = raster_plane_to_qimage(gray16)
    float_image = raster_plane_to_qimage(
        float_plane,
        display_range=(0.0, 1.0),
    )

    assert list(qimage_to_raster_plane(gray_image).data) == [0, 128, 255]
    assert list(qimage_to_raster_plane(float_image).data) == [0, 128, 255, 0]
    assert raster_plane_to_numpy(gray16).tobytes() == np.array(
        [[0, 32_768, 65_535]], dtype=np.uint16
    ).tobytes()


def test_recommended_native_suffix_is_scientifically_safe() -> None:
    assert recommended_native_asset_suffix(RasterPixelType.GRAY8) == ".png"
    assert recommended_native_asset_suffix(RasterPixelType.GRAY16) == ".png"
    assert recommended_native_asset_suffix(RasterPixelType.RGB8) == ".png"
    assert recommended_native_asset_suffix(RasterPixelType.RGBA8) == ".png"
    assert recommended_native_asset_suffix(RasterPixelType.GRAY32_FLOAT) == ".tif"
