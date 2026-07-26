from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
from PySide6.QtGui import QImage

from fdm.atomic_io import atomic_replace_file, staged_path_for
from fdm.image_processing_models import DisplayTransform
from fdm.raster import RasterPixelType, RasterPlane


class RasterIoFailureCode(str, Enum):
    INVALID_PATH = "invalid_path"
    INVALID_OPTIONS = "invalid_options"
    UNSUPPORTED_FORMAT = "unsupported_format"
    UNSUPPORTED_PIXEL_TYPE = "unsupported_pixel_type"
    DECODE_FAILED = "decode_failed"
    DEPENDENCY_UNAVAILABLE = "dependency_unavailable"
    ENCODE_FAILED = "encode_failed"
    VERIFY_FAILED = "verify_failed"
    ATOMIC_COMMIT_FAILED = "atomic_commit_failed"


@dataclass(frozen=True, slots=True)
class RasterIoFailure:
    code: RasterIoFailureCode
    message: str
    path: Path
    detail: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "code": self.code.value,
            "message": self.message,
            "path": str(self.path),
            "detail": self.detail,
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )


class RasterIoError(RuntimeError):
    def __init__(self, failure: RasterIoFailure) -> None:
        super().__init__(failure.message)
        self.failure = failure
        self.code = failure.code


@dataclass(frozen=True, slots=True)
class RasterMetadata:
    """Small metadata snapshot kept beside immutable pixel bytes.

    Pixel orientation is always normalized before a :class:`RasterPlane` is
    returned.  ``source_orientation`` records the original EXIF/TIFF value so
    diagnostics can explain a width/height swap without applying it twice.
    Resolution is normalized to dots per inch.
    """

    source_format: str = ""
    source_mode: str = ""
    icc_profile: bytes | None = None
    dpi_x: float | None = None
    dpi_y: float | None = None
    source_orientation: int = 1
    orientation_applied: bool = False
    source_photometric: str = ""
    photometric_applied: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_format",
            str(self.source_format or "").strip().upper(),
        )
        object.__setattr__(self, "source_mode", str(self.source_mode or "").strip())
        if self.icc_profile is not None:
            object.__setattr__(self, "icc_profile", bytes(self.icc_profile))
        object.__setattr__(self, "dpi_x", _optional_positive_finite(self.dpi_x, "水平 DPI"))
        object.__setattr__(self, "dpi_y", _optional_positive_finite(self.dpi_y, "垂直 DPI"))
        try:
            orientation = int(self.source_orientation)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("图片方向标记必须是 1–8 的整数") from exc
        if orientation < 1 or orientation > 8:
            raise ValueError("图片方向标记必须是 1–8 的整数")
        object.__setattr__(self, "source_orientation", orientation)
        object.__setattr__(
            self,
            "orientation_applied",
            bool(self.orientation_applied),
        )
        object.__setattr__(
            self,
            "source_photometric",
            str(self.source_photometric or "").strip().upper(),
        )
        object.__setattr__(
            self,
            "photometric_applied",
            bool(self.photometric_applied),
        )

    @property
    def icc_profile_sha256(self) -> str:
        if not self.icc_profile:
            return ""
        return hashlib.sha256(self.icc_profile).hexdigest()

    def to_dict(self) -> dict[str, object]:
        return {
            "source_format": self.source_format,
            "source_mode": self.source_mode,
            "icc_profile_bytes": len(self.icc_profile or b""),
            "icc_profile_sha256": self.icc_profile_sha256,
            "dpi_x": self.dpi_x,
            "dpi_y": self.dpi_y,
            "source_orientation": self.source_orientation,
            "orientation_applied": self.orientation_applied,
            "source_photometric": self.source_photometric,
            "photometric_applied": self.photometric_applied,
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )


@dataclass(frozen=True, slots=True)
class RasterReadResult:
    success: bool
    path: Path
    plane: RasterPlane | None = None
    metadata: RasterMetadata | None = None
    failure: RasterIoFailure | None = None

    def __bool__(self) -> bool:
        return self.success

    def require_success(self) -> "RasterReadResult":
        if not self.success:
            raise RasterIoError(
                self.failure
                or RasterIoFailure(
                    RasterIoFailureCode.DECODE_FAILED,
                    "无法读取图片。",
                    self.path,
                )
            )
        return self


class NativeRasterAssetFormat(str, Enum):
    PNG = "png"
    TIFF = "tiff"

    @classmethod
    def from_path(cls, path: str | Path) -> "NativeRasterAssetFormat":
        suffix = Path(path).suffix.casefold()
        if suffix == ".png":
            return cls.PNG
        if suffix in {".tif", ".tiff"}:
            return cls.TIFF
        raise ValueError("原生像素资产只支持 PNG 或 TIFF。")

    @property
    def suffix(self) -> str:
        return ".png" if self is NativeRasterAssetFormat.PNG else ".tif"


class NativeTiffCompression(str, Enum):
    DEFLATE = "deflate"
    LZW = "lzw"
    NONE = "none"

    @classmethod
    def parse(
        cls,
        value: "NativeTiffCompression | str",
    ) -> "NativeTiffCompression":
        if isinstance(value, cls):
            return value
        token = str(value or "").strip().casefold()
        aliases = {
            "deflate": cls.DEFLATE,
            "zip": cls.DEFLATE,
            "adobe_deflate": cls.DEFLATE,
            "lzw": cls.LZW,
            "none": cls.NONE,
            "raw": cls.NONE,
            "uncompressed": cls.NONE,
        }
        try:
            return aliases[token]
        except KeyError as exc:
            raise ValueError("TIFF 压缩方式只支持 deflate、lzw 或 none。") from exc


@dataclass(frozen=True, slots=True)
class RasterAssetWriteResult:
    success: bool
    path: Path
    pixel_type: RasterPixelType
    width: int = 0
    height: int = 0
    bytes_written: int = 0
    pixel_sha256: str = ""
    failure: RasterIoFailure | None = None

    def __bool__(self) -> bool:
        return self.success

    def require_success(self) -> "RasterAssetWriteResult":
        if not self.success:
            raise RasterIoError(
                self.failure
                or RasterIoFailure(
                    RasterIoFailureCode.ENCODE_FAILED,
                    "无法写入原生像素资产。",
                    self.path,
                )
            )
        return self


def recommended_native_asset_suffix(pixel_type: RasterPixelType) -> str:
    parsed = RasterPixelType.parse(pixel_type)
    if parsed is RasterPixelType.GRAY32_FLOAT:
        return ".tif"
    return ".png"


def raster_plane_to_numpy(
    plane: RasterPlane,
    *,
    writable: bool = False,
) -> np.ndarray:
    """Return a shaped NumPy view; mutable callers receive an isolated copy."""

    pixel_type = RasterPixelType.parse(plane.pixel_type)
    dtype: np.dtype[Any]
    if pixel_type is RasterPixelType.GRAY8:
        dtype = np.dtype(np.uint8)
        shape = (plane.height, plane.width)
    elif pixel_type is RasterPixelType.GRAY16:
        dtype = np.dtype("<u2")
        shape = (plane.height, plane.width)
    elif pixel_type is RasterPixelType.GRAY32_FLOAT:
        dtype = np.dtype("<f4")
        shape = (plane.height, plane.width)
    elif pixel_type is RasterPixelType.RGB8:
        dtype = np.dtype(np.uint8)
        shape = (plane.height, plane.width, 3)
    else:
        dtype = np.dtype(np.uint8)
        shape = (plane.height, plane.width, 4)
    result = np.frombuffer(plane.data, dtype=dtype).reshape(shape)
    return result.copy(order="C") if writable else result


def numpy_to_raster_plane(array: np.ndarray) -> RasterPlane:
    """Create an immutable plane without silently changing dtype or channels."""

    source = np.asarray(array)
    if source.ndim == 2 and source.dtype == np.dtype(np.uint8):
        pixel_type = RasterPixelType.GRAY8
        normalized = np.ascontiguousarray(source)
    elif source.ndim == 2 and source.dtype == np.dtype(np.uint16):
        pixel_type = RasterPixelType.GRAY16
        normalized = np.ascontiguousarray(source.astype("<u2", copy=False))
    elif source.ndim == 2 and source.dtype == np.dtype(np.float32):
        pixel_type = RasterPixelType.GRAY32_FLOAT
        normalized = np.ascontiguousarray(source.astype("<f4", copy=False))
    elif (
        source.ndim == 3
        and source.shape[2] in {3, 4}
        and source.dtype == np.dtype(np.uint8)
    ):
        pixel_type = (
            RasterPixelType.RGB8
            if source.shape[2] == 3
            else RasterPixelType.RGBA8
        )
        normalized = np.ascontiguousarray(source)
    else:
        raise ValueError(
            "仅支持 uint8 灰度、uint16 灰度、float32 灰度、"
            "uint8 RGB 和 uint8 RGBA 数组。"
        )
    height, width = normalized.shape[:2]
    return RasterPlane(
        width=int(width),
        height=int(height),
        pixel_type=pixel_type,
        data=normalized.tobytes(order="C"),
    )


def raster_plane_to_qimage(
    plane: RasterPlane,
    *,
    display_range: tuple[float, float] | None = None,
    display_transform: DisplayTransform | None = None,
) -> QImage:
    """Build an owned presentation cache without mutating authoritative pixels.

    ``display_range`` is retained for callers written before
    :class:`DisplayTransform`; new code should pass ``display_transform``.
    """

    if plane.is_empty:
        return QImage()
    if display_range is not None and display_transform is not None:
        raise ValueError("display_range 与 display_transform 不能同时提供")
    if display_transform is not None and not isinstance(
        display_transform,
        DisplayTransform,
    ):
        raise TypeError("display_transform 必须是 DisplayTransform")
    array = raster_plane_to_numpy(plane)
    pixel_type = plane.pixel_type
    if (
        pixel_type is RasterPixelType.GRAY8
        and display_range is None
        and (
            display_transform is None
            or display_transform.is_identity
        )
    ):
        display = array
        image_format = QImage.Format.Format_Grayscale8
    elif pixel_type in {
        RasterPixelType.GRAY8,
        RasterPixelType.GRAY16,
        RasterPixelType.GRAY32_FLOAT,
    }:
        transform = display_transform or DisplayTransform()
        ranges = transform.ranges_for_pixel_type(pixel_type)
        selected_range = ranges[0] if ranges else display_range
        display = _display_channel_uint8(
            array,
            pixel_type=pixel_type,
            display_range=selected_range,
            gamma=transform.gamma,
            inverted=transform.inverted,
        )
        if transform.lut_id not in {None, "grayscale"}:
            display = _apply_display_lut(display, transform.lut_id)
            image_format = QImage.Format.Format_RGB888
        else:
            image_format = QImage.Format.Format_Grayscale8
    elif pixel_type is RasterPixelType.RGB8:
        if display_transform is None or display_transform.is_identity:
            display = array
        else:
            display = _display_color_uint8(
                array,
                pixel_type=pixel_type,
                transform=display_transform,
            )
        image_format = QImage.Format.Format_RGB888
    else:
        if display_transform is None or display_transform.is_identity:
            display = array
        else:
            display = _display_color_uint8(
                array,
                pixel_type=pixel_type,
                transform=display_transform,
            )
        image_format = QImage.Format.Format_RGBA8888
    contiguous = np.ascontiguousarray(display)
    image = QImage(
        contiguous.data,
        plane.width,
        plane.height,
        int(contiguous.strides[0]),
        image_format,
    )
    return image.copy()


def qimage_to_raster_plane(image: QImage) -> RasterPlane:
    """Copy supported QImage pixels into a tightly packed immutable plane."""

    if image.isNull() or image.width() <= 0 or image.height() <= 0:
        return RasterPlane(0, 0, RasterPixelType.GRAY8, b"")

    if image.format() == QImage.Format.Format_Grayscale16:
        converted = image
        pixel_type = RasterPixelType.GRAY16
        channels = 1
        dtype = np.dtype(np.uint16)
        row_bytes = image.width() * 2
    elif image.format() in {
        QImage.Format.Format_Grayscale8,
        QImage.Format.Format_Alpha8,
    }:
        converted = image.convertToFormat(QImage.Format.Format_Grayscale8)
        pixel_type = RasterPixelType.GRAY8
        channels = 1
        dtype = np.dtype(np.uint8)
        row_bytes = image.width()
    elif image.hasAlphaChannel():
        converted = image.convertToFormat(QImage.Format.Format_RGBA8888)
        pixel_type = RasterPixelType.RGBA8
        channels = 4
        dtype = np.dtype(np.uint8)
        row_bytes = image.width() * channels
    else:
        converted = image.convertToFormat(QImage.Format.Format_RGB888)
        pixel_type = RasterPixelType.RGB8
        channels = 3
        dtype = np.dtype(np.uint8)
        row_bytes = image.width() * channels

    raw = np.frombuffer(
        converted.constBits(),
        dtype=np.uint8,
        count=converted.sizeInBytes(),
    ).reshape(converted.height(), converted.bytesPerLine())
    packed = np.ascontiguousarray(raw[:, :row_bytes])
    if pixel_type is RasterPixelType.GRAY16:
        words = packed.view(dtype).reshape(converted.height(), converted.width())
        if sys.byteorder != "little":
            words = words.byteswap()
        data = np.ascontiguousarray(words.astype("<u2", copy=False)).tobytes()
    else:
        data = packed.tobytes(order="C")
    return RasterPlane(
        width=converted.width(),
        height=converted.height(),
        pixel_type=pixel_type,
        data=data,
    )


def read_raster_file(path: str | Path) -> RasterReadResult:
    source_path = Path(path)
    try:
        if not source_path.is_file() or source_path.stat().st_size <= 0:
            raise OSError("文件不存在或为空")
    except OSError as exc:
        return _read_failure(
            RasterIoFailureCode.INVALID_PATH,
            source_path,
            f"无法读取图片文件：{source_path}",
            exc,
        )

    suffix = source_path.suffix.casefold()
    if suffix not in {
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
        ".webp",
        ".tif",
        ".tiff",
    }:
        return _read_failure(
            RasterIoFailureCode.UNSUPPORTED_FORMAT,
            source_path,
            f"不支持的图片格式：{suffix or '无扩展名'}。",
        )
    try:
        if suffix in {".tif", ".tiff"}:
            plane, metadata = _read_tiff(source_path)
        else:
            plane, metadata = _read_pillow(source_path)
        return RasterReadResult(
            success=True,
            path=source_path,
            plane=plane,
            metadata=metadata,
        )
    except (ImportError, ModuleNotFoundError) as exc:
        return _read_failure(
            RasterIoFailureCode.DEPENDENCY_UNAVAILABLE,
            source_path,
            "读取图片所需的编解码组件不可用。",
            exc,
        )
    except _UnsupportedRasterError as exc:
        return _read_failure(
            RasterIoFailureCode.UNSUPPORTED_PIXEL_TYPE,
            source_path,
            str(exc),
        )
    except Exception as exc:
        return _read_failure(
            RasterIoFailureCode.DECODE_FAILED,
            source_path,
            f"无法解码图片：{source_path.name}",
            exc,
        )


def write_native_raster_asset(
    plane: RasterPlane,
    target: str | Path,
    *,
    metadata: RasterMetadata | None = None,
    png_compression: int = 6,
    tiff_compression: NativeTiffCompression | str = NativeTiffCompression.DEFLATE,
) -> RasterAssetWriteResult:
    """Write and verify an authoritative lossless asset before atomic replace."""

    target_path = Path(target)
    try:
        asset_format = NativeRasterAssetFormat.from_path(target_path)
    except ValueError as exc:
        return _write_failure(
            RasterIoFailureCode.UNSUPPORTED_FORMAT,
            target_path,
            plane.pixel_type,
            str(exc),
        )
    try:
        resolved_png_compression = _bounded_integer(
            png_compression,
            minimum=0,
            maximum=9,
            label="PNG 压缩级别",
        )
        resolved_tiff_compression = NativeTiffCompression.parse(tiff_compression)
    except (TypeError, ValueError) as exc:
        return _write_failure(
            RasterIoFailureCode.INVALID_OPTIONS,
            target_path,
            plane.pixel_type,
            str(exc),
        )
    if plane.is_empty:
        return _write_failure(
            RasterIoFailureCode.UNSUPPORTED_PIXEL_TYPE,
            target_path,
            plane.pixel_type,
            "不能写入空的原生像素资产。",
        )
    if (
        asset_format is NativeRasterAssetFormat.PNG
        and plane.pixel_type is RasterPixelType.GRAY32_FLOAT
    ):
        return _write_failure(
            RasterIoFailureCode.UNSUPPORTED_PIXEL_TYPE,
            target_path,
            plane.pixel_type,
            "32 位浮点灰度像素只能保存为 TIFF，不能静默转换为 PNG。",
        )

    try:
        with staged_path_for(target_path, suffix=asset_format.suffix) as staged:
            try:
                if asset_format is NativeRasterAssetFormat.PNG:
                    _write_png(
                        plane,
                        staged,
                        metadata,
                        compression=resolved_png_compression,
                    )
                else:
                    _write_tiff(
                        plane,
                        staged,
                        metadata,
                        compression=resolved_tiff_compression,
                    )
            except (ImportError, ModuleNotFoundError) as exc:
                return _write_failure(
                    RasterIoFailureCode.DEPENDENCY_UNAVAILABLE,
                    target_path,
                    plane.pixel_type,
                    "写入原生像素资产所需的编解码组件不可用。",
                    exc,
                )
            except Exception as exc:
                return _write_failure(
                    RasterIoFailureCode.ENCODE_FAILED,
                    target_path,
                    plane.pixel_type,
                    f"无法编码原生像素资产：{target_path.name}",
                    exc,
                )

            verification = read_raster_file(staged)
            if (
                not verification.success
                or verification.plane is None
                or verification.plane.pixel_type is not plane.pixel_type
                or verification.plane.width != plane.width
                or verification.plane.height != plane.height
                or verification.plane.sha256() != plane.sha256()
            ):
                detail = (
                    verification.failure.message
                    if verification.failure is not None
                    else "解码后的像素类型、尺寸或 SHA256 与源像素不一致"
                )
                return _write_failure(
                    RasterIoFailureCode.VERIFY_FAILED,
                    target_path,
                    plane.pixel_type,
                    "原生像素资产未通过无损往返校验。",
                    detail,
                )
            try:
                bytes_written = staged.stat().st_size
                if bytes_written <= 0:
                    raise OSError("编码器生成了空文件")
                atomic_replace_file(staged, target_path)
            except Exception as exc:
                return _write_failure(
                    RasterIoFailureCode.ATOMIC_COMMIT_FAILED,
                    target_path,
                    plane.pixel_type,
                    f"无法原子提交原生像素资产：{target_path}",
                    exc,
                )
    except Exception as exc:
        return _write_failure(
            RasterIoFailureCode.ATOMIC_COMMIT_FAILED,
            target_path,
            plane.pixel_type,
            f"无法创建原生像素资产：{target_path}",
            exc,
        )
    return RasterAssetWriteResult(
        success=True,
        path=target_path,
        pixel_type=plane.pixel_type,
        width=plane.width,
        height=plane.height,
        bytes_written=int(bytes_written),
        pixel_sha256=plane.sha256(),
    )


class _UnsupportedRasterError(ValueError):
    pass


def _read_pillow(path: Path) -> tuple[RasterPlane, RasterMetadata]:
    from PIL import Image, ImageOps

    with Image.open(path) as opened:
        source_format = str(opened.format or path.suffix.lstrip("."))
        source_mode = str(opened.mode)
        info = dict(opened.info)
        try:
            orientation = int(opened.getexif().get(274, 1))
        except (AttributeError, TypeError, ValueError, OverflowError):
            orientation = 1
        orientation = orientation if 1 <= orientation <= 8 else 1
        normalized = ImageOps.exif_transpose(opened)
        normalized.load()
        if normalized.mode == "P":
            normalized = normalized.convert(
                "RGBA" if "transparency" in info else "RGB"
            )
        elif normalized.mode == "1":
            normalized = normalized.convert("L")
        elif normalized.mode == "LA":
            normalized = normalized.convert("RGBA")
        elif normalized.mode in {"CMYK", "YCbCr"}:
            normalized = normalized.convert("RGB")
        if normalized.mode in {"L", "RGB"} and "transparency" in info:
            normalized = normalized.convert("RGBA")
        if normalized.mode == "L":
            array = np.asarray(normalized, dtype=np.uint8)
        elif normalized.mode in {"I;16", "I;16L", "I;16B", "I;16N"}:
            array = np.asarray(normalized, dtype=np.uint16)
        elif normalized.mode == "F":
            array = np.asarray(normalized, dtype=np.float32)
        elif normalized.mode == "RGB":
            array = np.asarray(normalized, dtype=np.uint8)
        elif normalized.mode == "RGBA":
            array = np.asarray(normalized, dtype=np.uint8)
        else:
            raise _UnsupportedRasterError(
                f"图片像素模式 {normalized.mode} 无法无损映射到受支持的栅格类型。"
            )
        try:
            plane = numpy_to_raster_plane(array)
        except ValueError as exc:
            raise _UnsupportedRasterError(str(exc)) from exc
        dpi_x, dpi_y = _pillow_dpi(info.get("dpi"))
        icc = info.get("icc_profile")
        metadata = RasterMetadata(
            source_format=source_format,
            source_mode=source_mode,
            icc_profile=bytes(icc) if isinstance(icc, (bytes, bytearray)) else None,
            dpi_x=dpi_x,
            dpi_y=dpi_y,
            source_orientation=orientation,
            orientation_applied=orientation != 1,
        )
        return plane, metadata


def _read_tiff(path: Path) -> tuple[RasterPlane, RasterMetadata]:
    import tifffile

    with tifffile.TiffFile(path) as tiff:
        if len(tiff.pages) != 1:
            raise _UnsupportedRasterError("本轮只支持单页二维 TIFF，不支持图像堆栈。")
        page = tiff.pages[0]
        _validate_tiff_page_layout(page)
        try:
            array = np.asarray(page.asarray())
        except Exception:
            # tifffile intentionally keeps optional codecs out of its core
            # dependency.  Pillow/libtiff supplies LZW in the desktop runtime,
            # while the exact SHA verification below still protects pixels.
            return _read_pillow(path)
        axes = str(getattr(page, "axes", "") or "")
        if (
            array.ndim == 3
            and axes.startswith("S")
            and array.shape[0] in {3, 4}
        ):
            array = np.moveaxis(array, 0, -1)
        if array.ndim not in {2, 3}:
            raise _UnsupportedRasterError("TIFF 不是受支持的二维单图。")
        if array.ndim == 3 and array.shape[-1] not in {3, 4}:
            raise _UnsupportedRasterError("TIFF 通道数不是受支持的 RGB 或 RGBA。")
        try:
            orientation = int(page.tags[274].value)
        except (KeyError, TypeError, ValueError, OverflowError):
            orientation = 1
        orientation = orientation if 1 <= orientation <= 8 else 1
        array = _apply_orientation(array, orientation)
        photometric = str(
            getattr(getattr(page, "photometric", None), "name", "")
            or ""
        ).upper()
        photometric_applied = False
        if photometric == "PALETTE":
            color_map = getattr(page, "colormap", None)
            if color_map is None:
                raise _UnsupportedRasterError(
                    "调色板 TIFF 缺少颜色映射表。"
                )
            array = _expand_tiff_palette(array, np.asarray(color_map))
            photometric_applied = True
        elif photometric == "MINISWHITE":
            array = _normalize_miniswhite(array)
            photometric_applied = True
        try:
            icc_value = page.tags[34675].value
            icc = bytes(icc_value)
        except (KeyError, TypeError, ValueError):
            icc = None
        dpi_x, dpi_y = _tiff_dpi(page)
        try:
            plane = numpy_to_raster_plane(array)
        except ValueError as exc:
            raise _UnsupportedRasterError(str(exc)) from exc
        return plane, RasterMetadata(
            source_format="TIFF",
            source_mode=_numpy_mode_name(array),
            icc_profile=icc,
            dpi_x=dpi_x,
            dpi_y=dpi_y,
            source_orientation=orientation,
            orientation_applied=orientation != 1,
            source_photometric=photometric,
            photometric_applied=photometric_applied,
        )


def _validate_tiff_page_layout(page: Any) -> None:
    dtype = np.dtype(page.dtype)
    samples = int(getattr(page, "samplesperpixel", 1) or 1)
    if dtype == np.dtype(np.uint8) and samples in {1, 3, 4}:
        return
    if dtype in {np.dtype(np.uint16), np.dtype(np.float32)} and samples == 1:
        return
    raise _UnsupportedRasterError(
        "TIFF 像素类型或通道布局无法无损映射到受支持的栅格类型。"
    )


def _write_png(
    plane: RasterPlane,
    path: Path,
    metadata: RasterMetadata | None,
    *,
    compression: int,
) -> None:
    from PIL import Image

    array = raster_plane_to_numpy(plane)
    if plane.pixel_type is RasterPixelType.GRAY16:
        image = Image.fromarray(array.astype(np.uint16, copy=False))
    else:
        image = Image.fromarray(array)
    try:
        kwargs: dict[str, Any] = {"compress_level": compression}
        if metadata is not None and metadata.icc_profile:
            kwargs["icc_profile"] = metadata.icc_profile
        if (
            metadata is not None
            and metadata.dpi_x is not None
            and metadata.dpi_y is not None
        ):
            kwargs["dpi"] = (metadata.dpi_x, metadata.dpi_y)
        image.save(path, format="PNG", **kwargs)
    finally:
        image.close()


def _write_tiff(
    plane: RasterPlane,
    path: Path,
    metadata: RasterMetadata | None,
    *,
    compression: NativeTiffCompression,
) -> None:
    import tifffile

    array = raster_plane_to_numpy(plane)
    if compression is NativeTiffCompression.LZW:
        _write_tiff_lzw_with_pillow(array, path, metadata)
        return
    kwargs: dict[str, Any] = {
        "compression": (
            "deflate"
            if compression is NativeTiffCompression.DEFLATE
            else None
        ),
        "metadata": None,
    }
    if plane.pixel_type in {RasterPixelType.RGB8, RasterPixelType.RGBA8}:
        kwargs["photometric"] = "rgb"
        if plane.pixel_type is RasterPixelType.RGBA8:
            kwargs["extrasamples"] = "unassalpha"
    else:
        kwargs["photometric"] = "minisblack"
    if (
        metadata is not None
        and metadata.dpi_x is not None
        and metadata.dpi_y is not None
    ):
        kwargs["resolution"] = (metadata.dpi_x, metadata.dpi_y)
        kwargs["resolutionunit"] = "INCH"
    if metadata is not None and metadata.icc_profile:
        kwargs["extratags"] = [
            (34675, "B", len(metadata.icc_profile), metadata.icc_profile, False)
        ]
    tifffile.imwrite(path, array, **kwargs)


def _write_tiff_lzw_with_pillow(
    array: np.ndarray,
    path: Path,
    metadata: RasterMetadata | None,
) -> None:
    """Use Pillow's bundled libtiff path so LZW needs no imagecodecs extra."""

    from PIL import Image

    image = Image.fromarray(array)
    try:
        kwargs: dict[str, Any] = {"compression": "tiff_lzw"}
        if metadata is not None and metadata.icc_profile:
            kwargs["icc_profile"] = metadata.icc_profile
        if (
            metadata is not None
            and metadata.dpi_x is not None
            and metadata.dpi_y is not None
        ):
            kwargs["dpi"] = (metadata.dpi_x, metadata.dpi_y)
        image.save(path, format="TIFF", **kwargs)
    finally:
        image.close()


def _scalar_display_uint8(
    array: np.ndarray,
    pixel_type: RasterPixelType,
    display_range: tuple[float, float] | None,
) -> np.ndarray:
    return _display_channel_uint8(
        array,
        pixel_type=pixel_type,
        display_range=display_range,
        gamma=1.0,
        inverted=False,
    )


def _display_channel_uint8(
    array: np.ndarray,
    *,
    pixel_type: RasterPixelType,
    display_range: tuple[float, float] | None,
    gamma: float,
    inverted: bool,
) -> np.ndarray:
    values = array.astype(np.float64, copy=False)
    finite = np.isfinite(values)
    automatic_range = display_range is None
    if automatic_range:
        if pixel_type is RasterPixelType.GRAY8:
            low, high = 0.0, 255.0
        elif pixel_type is RasterPixelType.GRAY16:
            low, high = 0.0, 65_535.0
        elif finite.any():
            low = float(values[finite].min())
            high = float(values[finite].max())
        else:
            low, high = 0.0, 1.0
    else:
        low, high = float(display_range[0]), float(display_range[1])
    if (
        automatic_range
        and math.isfinite(low)
        and math.isfinite(high)
        and high == low
    ):
        # A constant scientific plane is valid.  A mid-gray display keeps
        # finite samples distinguishable from NaN/Inf (rendered at the range
        # ends) without changing the authoritative float pixels.
        normalized = np.zeros(values.shape, dtype=np.float64)
        normalized[finite] = 0.5
        normalized[np.isposinf(values)] = 1.0
    else:
        if not math.isfinite(low) or not math.isfinite(high) or high <= low:
            raise ValueError("显示范围必须由两个递增的有限数构成")
        normalized = np.nan_to_num(
            (values - low) / (high - low),
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        )
        normalized = np.clip(normalized, 0.0, 1.0)
    if gamma != 1.0:
        normalized = np.power(normalized, 1.0 / gamma)
    if inverted:
        normalized = 1.0 - normalized
    return np.clip(np.rint(normalized * 255.0), 0.0, 255.0).astype(np.uint8)


def _display_color_uint8(
    array: np.ndarray,
    *,
    pixel_type: RasterPixelType,
    transform: DisplayTransform,
) -> np.ndarray:
    ranges = transform.ranges_for_pixel_type(pixel_type)
    color = array[:, :, :3]
    mapped_channels = [
        _display_channel_uint8(
            color[:, :, channel],
            pixel_type=RasterPixelType.GRAY8,
            display_range=ranges[channel] if ranges else None,
            gamma=transform.gamma,
            inverted=transform.inverted,
        )
        for channel in range(3)
    ]
    mapped = np.stack(mapped_channels, axis=2)
    if pixel_type is RasterPixelType.RGBA8:
        # Presentation controls never alter data transparency.
        mapped = np.concatenate((mapped, array[:, :, 3:4]), axis=2)
    return np.ascontiguousarray(mapped)


def _apply_display_lut(values: np.ndarray, lut_id: str) -> np.ndarray:
    indices = np.asarray(values, dtype=np.uint8)
    value = np.arange(256, dtype=np.uint16)
    if lut_id == "red":
        table = np.stack((value, value * 0, value * 0), axis=1)
    elif lut_id == "green":
        table = np.stack((value * 0, value, value * 0), axis=1)
    elif lut_id == "blue":
        table = np.stack((value * 0, value * 0, value), axis=1)
    elif lut_id == "fire":
        table = np.stack(
            (
                np.clip(value * 3, 0, 255),
                np.clip((value.astype(np.int16) - 85) * 3, 0, 255),
                np.clip((value.astype(np.int16) - 170) * 3, 0, 255),
            ),
            axis=1,
        )
    elif lut_id == "ice":
        table = np.stack(
            (
                value // 2,
                value,
                np.clip(96 + value, 0, 255),
            ),
            axis=1,
        )
    elif lut_id == "spectrum":
        # Six linear colour ramps make a stable, dependency-free spectral LUT.
        phase = value.astype(np.float64) * (5.0 / 255.0)
        segment = np.floor(phase).astype(np.int16)
        fraction = phase - segment
        rgb = np.zeros((256, 3), dtype=np.float64)
        anchors = np.asarray(
            (
                (0, 0, 255),
                (0, 255, 255),
                (0, 255, 0),
                (255, 255, 0),
                (255, 0, 0),
                (255, 0, 255),
            ),
            dtype=np.float64,
        )
        for index in range(5):
            selected = segment == index
            rgb[selected] = (
                anchors[index] * (1.0 - fraction[selected, None])
                + anchors[index + 1] * fraction[selected, None]
            )
        rgb[segment >= 5] = anchors[-1]
        table = rgb
    else:  # pragma: no cover - DisplayTransform validates identifiers
        raise ValueError(f"不支持的显示 LUT：{lut_id}")
    return np.ascontiguousarray(table.astype(np.uint8)[indices])


def _expand_tiff_palette(
    indices: np.ndarray,
    color_map: np.ndarray,
) -> np.ndarray:
    if indices.ndim != 2 or not np.issubdtype(indices.dtype, np.integer):
        raise _UnsupportedRasterError("调色板 TIFF 的索引像素无效。")
    if color_map.ndim != 2 or color_map.shape[0] != 3:
        raise _UnsupportedRasterError("调色板 TIFF 的颜色映射表无效。")
    maximum_index = int(indices.max(initial=0))
    if maximum_index >= color_map.shape[1]:
        raise _UnsupportedRasterError("调色板 TIFF 含有超出颜色表的索引。")
    mapped = color_map[:, indices.astype(np.int64, copy=False)]
    if mapped.dtype.itemsize > 1 or int(mapped.max(initial=0)) > 255:
        mapped = np.rint(mapped.astype(np.float64) / 257.0)
    return np.ascontiguousarray(
        np.clip(mapped, 0, 255).astype(np.uint8).transpose(1, 2, 0)
    )


def _normalize_miniswhite(array: np.ndarray) -> np.ndarray:
    if np.issubdtype(array.dtype, np.integer):
        maximum = np.iinfo(array.dtype).max
        return np.ascontiguousarray(maximum - array)
    values = np.asarray(array, dtype=np.float32)
    finite = np.isfinite(values)
    if not finite.any():
        return np.ascontiguousarray(values)
    low = float(values[finite].min())
    high = float(values[finite].max())
    normalized = values.copy()
    normalized[finite] = (low + high) - normalized[finite]
    return np.ascontiguousarray(normalized)


def _apply_orientation(array: np.ndarray, orientation: int) -> np.ndarray:
    if orientation == 2:
        transformed = np.flip(array, axis=1)
    elif orientation == 3:
        transformed = np.flip(array, axis=(0, 1))
    elif orientation == 4:
        transformed = np.flip(array, axis=0)
    elif orientation == 5:
        transformed = np.swapaxes(array, 0, 1)
    elif orientation == 6:
        transformed = np.rot90(array, k=3, axes=(0, 1))
    elif orientation == 7:
        transformed = np.flip(np.swapaxes(array, 0, 1), axis=(0, 1))
    elif orientation == 8:
        transformed = np.rot90(array, k=1, axes=(0, 1))
    else:
        transformed = array
    return np.ascontiguousarray(transformed)


def _pillow_dpi(value: object) -> tuple[float | None, float | None]:
    if not isinstance(value, (tuple, list)) or len(value) < 2:
        return None, None
    try:
        x = _optional_positive_finite(value[0], "水平 DPI")
        y = _optional_positive_finite(value[1], "垂直 DPI")
    except ValueError:
        return None, None
    return x, y


def _tiff_dpi(page: Any) -> tuple[float | None, float | None]:
    try:
        x = _rational_float(page.tags[282].value)
        y = _rational_float(page.tags[283].value)
        unit_value = int(page.tags[296].value)
    except (KeyError, TypeError, ValueError, OverflowError, ZeroDivisionError):
        return None, None
    if unit_value == 3:
        x *= 2.54
        y *= 2.54
    elif unit_value != 2:
        return None, None
    try:
        return (
            _optional_positive_finite(x, "水平 DPI"),
            _optional_positive_finite(y, "垂直 DPI"),
        )
    except ValueError:
        return None, None


def _rational_float(value: object) -> float:
    if isinstance(value, (tuple, list)) and len(value) == 2:
        numerator = float(value[0])
        denominator = float(value[1])
        if denominator == 0.0:
            raise ZeroDivisionError("分辨率分母不能为 0")
        return numerator / denominator
    return float(value)


def _numpy_mode_name(array: np.ndarray) -> str:
    if array.ndim == 3:
        return "RGB" if array.shape[-1] == 3 else "RGBA"
    if array.dtype == np.dtype(np.uint8):
        return "L"
    if array.dtype == np.dtype(np.uint16):
        return "I;16"
    if array.dtype == np.dtype(np.float32):
        return "F"
    return str(array.dtype)


def _optional_positive_finite(value: object, label: str) -> float | None:
    if value is None:
        return None
    try:
        normalized = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} 必须是正有限数") from exc
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"{label} 必须是正有限数")
    return normalized


def _bounded_integer(
    value: object,
    *,
    minimum: int,
    maximum: int,
    label: str,
) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{label} 必须是 {minimum}–{maximum} 的整数")
    try:
        normalized = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{label} 必须是 {minimum}–{maximum} 的整数") from exc
    if normalized != value or normalized < minimum or normalized > maximum:
        raise ValueError(f"{label} 必须是 {minimum}–{maximum} 的整数")
    return normalized


def _read_failure(
    code: RasterIoFailureCode,
    path: Path,
    message: str,
    detail: object = "",
) -> RasterReadResult:
    return RasterReadResult(
        success=False,
        path=path,
        failure=RasterIoFailure(
            code=code,
            message=message,
            path=path,
            detail=str(detail),
        ),
    )


def _write_failure(
    code: RasterIoFailureCode,
    path: Path,
    pixel_type: RasterPixelType,
    message: str,
    detail: object = "",
) -> RasterAssetWriteResult:
    return RasterAssetWriteResult(
        success=False,
        path=path,
        pixel_type=pixel_type,
        failure=RasterIoFailure(
            code=code,
            message=message,
            path=path,
            detail=str(detail),
        ),
    )
