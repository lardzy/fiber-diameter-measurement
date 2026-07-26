from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from fdm.atomic_io import atomic_replace_file, staged_path_for


class RasterExportFormat(str, Enum):
    """Raster formats supported by the user-facing image export workflow."""

    PNG = "png"
    JPEG = "jpeg"
    TIFF = "tiff"
    BMP = "bmp"
    WEBP = "webp"

    @classmethod
    def coerce(cls, value: "RasterExportFormat | str") -> "RasterExportFormat":
        if isinstance(value, cls):
            return value
        token = str(value or "").strip().lower().lstrip(".")
        aliases = {
            "jpg": cls.JPEG,
            "jpeg": cls.JPEG,
            "tif": cls.TIFF,
            "tiff": cls.TIFF,
            "png": cls.PNG,
            "bmp": cls.BMP,
            "webp": cls.WEBP,
        }
        try:
            return aliases[token]
        except KeyError as exc:
            raise ValueError(f"不支持的栅格导出格式：{value}") from exc

    @property
    def canonical_suffix(self) -> str:
        return {
            RasterExportFormat.PNG: ".png",
            RasterExportFormat.JPEG: ".jpg",
            RasterExportFormat.TIFF: ".tif",
            RasterExportFormat.BMP: ".bmp",
            RasterExportFormat.WEBP: ".webp",
        }[self]

    @property
    def accepted_suffixes(self) -> tuple[str, ...]:
        if self is RasterExportFormat.JPEG:
            return (".jpg", ".jpeg")
        if self is RasterExportFormat.TIFF:
            return (".tif", ".tiff")
        return (self.canonical_suffix,)

    @property
    def pillow_name(self) -> str:
        return {
            RasterExportFormat.PNG: "PNG",
            RasterExportFormat.JPEG: "JPEG",
            RasterExportFormat.TIFF: "TIFF",
            RasterExportFormat.BMP: "BMP",
            RasterExportFormat.WEBP: "WEBP",
        }[self]

    @property
    def supports_quality(self) -> bool:
        return self in {RasterExportFormat.JPEG, RasterExportFormat.WEBP}


class TiffCompression(str, Enum):
    DEFLATE = "deflate"
    LZW = "lzw"
    NONE = "none"

    @classmethod
    def coerce(cls, value: "TiffCompression | str") -> "TiffCompression":
        if isinstance(value, cls):
            return value
        token = str(value or "").strip().lower()
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
            raise ValueError(f"不支持的 TIFF 压缩方式：{value}") from exc


@dataclass(frozen=True, slots=True)
class RasterEncodingOptions:
    """Immutable format options suitable for freezing into an export plan.

    ``quality`` is meaningful only for JPEG and lossy WebP.  ``None`` keeps
    the professional defaults: JPEG 95 and WebP 90.
    """

    format: RasterExportFormat = RasterExportFormat.PNG
    quality: int | None = None
    jpeg_progressive: bool = True
    png_compression: int = 6
    tiff_compression: TiffCompression = TiffCompression.DEFLATE
    webp_lossless: bool = False
    webp_method: int = 4
    preserve_color_profile: bool = True
    preserve_resolution: bool = True
    jpeg_background: tuple[int, int, int] = (255, 255, 255)

    def __post_init__(self) -> None:
        object.__setattr__(self, "format", RasterExportFormat.coerce(self.format))
        object.__setattr__(
            self,
            "tiff_compression",
            TiffCompression.coerce(self.tiff_compression),
        )
        if self.quality is not None:
            object.__setattr__(
                self,
                "quality",
                _bounded_int(self.quality, minimum=1, maximum=100, label="图片质量"),
            )
        object.__setattr__(
            self,
            "png_compression",
            _bounded_int(self.png_compression, minimum=0, maximum=9, label="PNG 压缩级别"),
        )
        object.__setattr__(
            self,
            "webp_method",
            _bounded_int(self.webp_method, minimum=0, maximum=6, label="WebP 编码强度"),
        )
        object.__setattr__(
            self,
            "jpeg_background",
            _normalize_rgb(self.jpeg_background, label="JPEG 透明背景颜色"),
        )
        object.__setattr__(self, "webp_lossless", bool(self.webp_lossless))
        object.__setattr__(
            self,
            "jpeg_progressive",
            bool(self.jpeg_progressive),
        )
        object.__setattr__(
            self,
            "preserve_color_profile",
            bool(self.preserve_color_profile),
        )
        object.__setattr__(
            self,
            "preserve_resolution",
            bool(self.preserve_resolution),
        )

    @property
    def resolved_quality(self) -> int | None:
        if not self.format.supports_quality:
            return None
        if self.quality is not None:
            return self.quality
        return 95 if self.format is RasterExportFormat.JPEG else 90

    @property
    def canonical_suffix(self) -> str:
        return self.format.canonical_suffix

    def with_format(
        self,
        format: RasterExportFormat | str,
    ) -> "RasterEncodingOptions":
        return replace(self, format=RasterExportFormat.coerce(format))

    def to_dict(self) -> dict[str, object]:
        return {
            "format": self.format.value,
            "quality": self.quality,
            "jpeg_progressive": self.jpeg_progressive,
            "png_compression": self.png_compression,
            "tiff_compression": self.tiff_compression.value,
            "webp_lossless": self.webp_lossless,
            "webp_method": self.webp_method,
            "preserve_color_profile": self.preserve_color_profile,
            "preserve_resolution": self.preserve_resolution,
            "jpeg_background": list(self.jpeg_background),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object] | None) -> "RasterEncodingOptions":
        if not isinstance(payload, dict):
            return cls()
        return cls(
            format=RasterExportFormat.coerce(str(payload.get("format", "png"))),
            quality=payload.get("quality"),  # type: ignore[arg-type]
            jpeg_progressive=bool(payload.get("jpeg_progressive", True)),
            png_compression=payload.get("png_compression", 6),  # type: ignore[arg-type]
            tiff_compression=TiffCompression.coerce(
                str(payload.get("tiff_compression", "deflate"))
            ),
            webp_lossless=bool(payload.get("webp_lossless", False)),
            webp_method=payload.get("webp_method", 4),  # type: ignore[arg-type]
            preserve_color_profile=bool(payload.get("preserve_color_profile", True)),
            preserve_resolution=bool(payload.get("preserve_resolution", True)),
            jpeg_background=payload.get("jpeg_background", (255, 255, 255)),  # type: ignore[arg-type]
        )


class RasterWriteFailureCode(str, Enum):
    DEPENDENCY_UNAVAILABLE = "dependency_unavailable"
    ENCODER_UNAVAILABLE = "encoder_unavailable"
    INVALID_SOURCE = "invalid_source"
    INVALID_TARGET = "invalid_target"
    DECODE_FAILED = "decode_failed"
    UNSUPPORTED_PIXEL_MODE = "unsupported_pixel_mode"
    ENCODE_FAILED = "encode_failed"
    VERIFY_FAILED = "verify_failed"
    ATOMIC_COMMIT_FAILED = "atomic_commit_failed"


@dataclass(frozen=True, slots=True)
class RasterEncoderCapability:
    format: RasterExportFormat
    available: bool
    supports_quality: bool
    supports_alpha: bool
    lossless_available: bool
    failure_code: RasterWriteFailureCode | None = None
    reason: str = ""
    backend_version: str = ""


@dataclass(frozen=True, slots=True)
class RasterWriteFailure:
    code: RasterWriteFailureCode
    format: RasterExportFormat
    message: str
    source_path: Path
    target_path: Path
    detail: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "code": self.code.value,
            "format": self.format.value,
            "message": self.message,
            "source_path": str(self.source_path),
            "target_path": str(self.target_path),
            "detail": self.detail,
        }


class RasterExportError(RuntimeError):
    """Exception adapter for callers that cannot consume a result object."""

    def __init__(self, failure: RasterWriteFailure) -> None:
        super().__init__(failure.message)
        self.failure = failure
        self.code = failure.code
        self.format = failure.format


@dataclass(frozen=True, slots=True)
class RasterWriteResult:
    success: bool
    format: RasterExportFormat
    path: Path
    width: int = 0
    height: int = 0
    bytes_written: int = 0
    failure: RasterWriteFailure | None = None

    def __bool__(self) -> bool:
        return self.success

    def require_success(self) -> "RasterWriteResult":
        if not self.success:
            failure = self.failure
            if failure is None:  # defensive guard for malformed third-party results
                failure = RasterWriteFailure(
                    code=RasterWriteFailureCode.ENCODE_FAILED,
                    format=self.format,
                    message="栅格图片导出失败。",
                    source_path=Path(),
                    target_path=self.path,
                )
            raise RasterExportError(failure)
        return self


class _UnsupportedPixelModeError(ValueError):
    pass


def _load_pillow_runtime():
    from PIL import Image, features
    import PIL

    Image.init()
    return Image, features, str(getattr(PIL, "__version__", ""))


class RasterExportWriter:
    """Encode a raster source into a same-directory staged file and replace.

    The source is decoded completely before the destination is touched.  All
    expected operational failures are represented by :class:`RasterWriteResult`
    so the UI can explain missing codecs without relying on exception text.
    """

    def __init__(
        self,
        *,
        pillow_loader: Callable[[], tuple[Any, Any, str]] | None = None,
    ) -> None:
        self._pillow_loader = pillow_loader or _load_pillow_runtime

    def capability(
        self,
        format: RasterExportFormat | str,
    ) -> RasterEncoderCapability:
        export_format = RasterExportFormat.coerce(format)
        try:
            image_module, features_module, version = self._pillow_loader()
        except (ImportError, ModuleNotFoundError) as exc:
            return RasterEncoderCapability(
                format=export_format,
                available=False,
                supports_quality=export_format.supports_quality,
                supports_alpha=export_format
                in {
                    RasterExportFormat.PNG,
                    RasterExportFormat.TIFF,
                    RasterExportFormat.WEBP,
                },
                lossless_available=False,
                failure_code=RasterWriteFailureCode.DEPENDENCY_UNAVAILABLE,
                reason=f"Pillow 不可用：{exc}",
            )
        except Exception as exc:
            return RasterEncoderCapability(
                format=export_format,
                available=False,
                supports_quality=export_format.supports_quality,
                supports_alpha=False,
                lossless_available=False,
                failure_code=RasterWriteFailureCode.DEPENDENCY_UNAVAILABLE,
                reason=f"无法初始化图片编码后端：{exc}",
            )

        encoder_available = export_format.pillow_name in image_module.SAVE
        if export_format is RasterExportFormat.WEBP:
            try:
                encoder_available = encoder_available and bool(
                    features_module.check("webp")
                )
            except Exception:
                encoder_available = False
        return RasterEncoderCapability(
            format=export_format,
            available=encoder_available,
            supports_quality=export_format.supports_quality,
            supports_alpha=export_format
            in {
                RasterExportFormat.PNG,
                RasterExportFormat.TIFF,
                RasterExportFormat.WEBP,
            },
            lossless_available=export_format
            in {
                RasterExportFormat.PNG,
                RasterExportFormat.TIFF,
                RasterExportFormat.BMP,
            }
            or (export_format is RasterExportFormat.WEBP and encoder_available),
            failure_code=(
                None
                if encoder_available
                else RasterWriteFailureCode.ENCODER_UNAVAILABLE
            ),
            reason=(
                ""
                if encoder_available
                else f"当前 Pillow 运行时没有 {export_format.pillow_name} 编码器。"
            ),
            backend_version=version,
        )

    def capabilities(self) -> tuple[RasterEncoderCapability, ...]:
        return tuple(self.capability(format) for format in RasterExportFormat)

    def write_file(
        self,
        source: str | Path,
        target: str | Path,
        options: RasterEncodingOptions | None = None,
    ) -> RasterWriteResult:
        source_path = Path(source)
        target_path = Path(target)
        encoding = options or RasterEncodingOptions()
        export_format = encoding.format

        target_suffix = target_path.suffix.lower()
        if target_suffix not in export_format.accepted_suffixes:
            return self._failure(
                RasterWriteFailureCode.INVALID_TARGET,
                export_format,
                source_path,
                target_path,
                (
                    f"目标扩展名 {target_suffix or '（无）'} 与 "
                    f"{export_format.value.upper()} 格式不一致。"
                ),
            )
        try:
            if not source_path.is_file() or source_path.stat().st_size <= 0:
                raise OSError("源文件不存在或为空")
        except OSError as exc:
            return self._failure(
                RasterWriteFailureCode.INVALID_SOURCE,
                export_format,
                source_path,
                target_path,
                f"无法读取待导出的栅格源文件：{source_path}",
                detail=str(exc),
            )

        capability = self.capability(export_format)
        if not capability.available:
            return self._failure(
                capability.failure_code or RasterWriteFailureCode.ENCODER_UNAVAILABLE,
                export_format,
                source_path,
                target_path,
                capability.reason or f"{export_format.value.upper()} 编码器不可用。",
            )
        try:
            image_module, _features_module, _version = self._pillow_loader()
        except Exception as exc:
            return self._failure(
                RasterWriteFailureCode.DEPENDENCY_UNAVAILABLE,
                export_format,
                source_path,
                target_path,
                "图片编码后端在导出开始后变得不可用。",
                detail=str(exc),
            )

        try:
            with image_module.open(source_path) as opened:
                opened.load()
                source_info = dict(opened.info)
                image = opened.copy()
        except Exception as exc:
            return self._failure(
                RasterWriteFailureCode.DECODE_FAILED,
                export_format,
                source_path,
                target_path,
                f"无法解码待导出的栅格源文件：{source_path.name}",
                detail=str(exc),
            )

        prepared = None
        try:
            prepared = self._prepare_image(image, encoding, image_module)
            save_kwargs = self._save_kwargs(source_info, encoding)
        except _UnsupportedPixelModeError as exc:
            image.close()
            return self._failure(
                RasterWriteFailureCode.UNSUPPORTED_PIXEL_MODE,
                export_format,
                source_path,
                target_path,
                str(exc),
            )
        except Exception as exc:
            image.close()
            return self._failure(
                RasterWriteFailureCode.ENCODE_FAILED,
                export_format,
                source_path,
                target_path,
                "无法准备待导出的像素数据。",
                detail=str(exc),
            )

        width, height = prepared.size
        try:
            try:
                staging_context = staged_path_for(
                    target_path,
                    suffix=f"{export_format.canonical_suffix}.tmp",
                )
                with staging_context as staged_path:
                    try:
                        prepared.save(
                            staged_path,
                            format=export_format.pillow_name,
                            **save_kwargs,
                        )
                    except Exception as exc:
                        return self._failure(
                            RasterWriteFailureCode.ENCODE_FAILED,
                            export_format,
                            source_path,
                            target_path,
                            f"无法编码 {export_format.value.upper()} 图片。",
                            detail=str(exc),
                        )
                    try:
                        bytes_written = staged_path.stat().st_size
                        if bytes_written <= 0:
                            raise OSError("编码器生成了空文件")
                    except OSError as exc:
                        return self._failure(
                            RasterWriteFailureCode.ENCODE_FAILED,
                            export_format,
                            source_path,
                            target_path,
                            "图片编码器未生成有效文件。",
                            detail=str(exc),
                        )
                    verification = self._verify_staged_file(
                        staged_path,
                        export_format,
                        expected_size=(width, height),
                        image_module=image_module,
                    )
                    if verification is not None:
                        return self._failure(
                            RasterWriteFailureCode.VERIFY_FAILED,
                            export_format,
                            source_path,
                            target_path,
                            "已编码图片未通过完整性校验。",
                            detail=verification,
                        )
                    try:
                        atomic_replace_file(staged_path, target_path)
                    except Exception as exc:
                        return self._failure(
                            RasterWriteFailureCode.ATOMIC_COMMIT_FAILED,
                            export_format,
                            source_path,
                            target_path,
                            f"无法原子提交导出文件：{target_path}",
                            detail=str(exc),
                        )
            except Exception as exc:
                return self._failure(
                    RasterWriteFailureCode.ATOMIC_COMMIT_FAILED,
                    export_format,
                    source_path,
                    target_path,
                    f"无法创建导出文件：{target_path}",
                    detail=str(exc),
                )
        finally:
            if prepared is not image:
                prepared.close()
            image.close()

        return RasterWriteResult(
            success=True,
            format=export_format,
            path=target_path,
            width=int(width),
            height=int(height),
            bytes_written=int(bytes_written),
        )

    @staticmethod
    def _prepare_image(image, options: RasterEncodingOptions, image_module):
        export_format = options.format
        mode = str(image.mode)
        if export_format is RasterExportFormat.JPEG:
            if mode in {"RGBA", "LA"} or (
                mode == "P" and "transparency" in image.info
            ):
                rgba = image.convert("RGBA")
                alpha = rgba.getchannel("A")
                background = image_module.new(
                    "RGB",
                    rgba.size,
                    options.jpeg_background,
                )
                background.paste(rgba, mask=alpha)
                alpha.close()
                rgba.close()
                return background
            if mode == "P":
                return image.convert("RGB")
            if mode not in {"L", "RGB", "CMYK"}:
                raise _UnsupportedPixelModeError(
                    f"JPEG 不支持当前像素类型 {mode}；为避免静默丢失位深，已拒绝导出。"
                )
            return image

        if export_format is RasterExportFormat.PNG:
            if mode == "CMYK":
                return image.convert("RGB")
            if mode == "F":
                raise _UnsupportedPixelModeError(
                    "PNG 不支持 32 位浮点像素；请使用 TIFF。"
                )
            return image

        if export_format is RasterExportFormat.TIFF:
            return image

        if export_format is RasterExportFormat.BMP:
            if mode in {"RGBA", "LA"} or (
                mode == "P" and "transparency" in image.info
            ):
                rgba = image.convert("RGBA")
                alpha = rgba.getchannel("A")
                background = image_module.new(
                    "RGB",
                    rgba.size,
                    options.jpeg_background,
                )
                background.paste(rgba, mask=alpha)
                alpha.close()
                rgba.close()
                return background
            if mode == "P":
                return image.convert("RGB")
            if mode not in {"1", "L", "RGB"}:
                raise _UnsupportedPixelModeError(
                    f"BMP 不支持当前像素类型 {mode}；为避免静默丢失位深，已拒绝导出。"
                )
            return image

        if export_format is RasterExportFormat.WEBP:
            if mode == "P":
                target_mode = "RGBA" if "transparency" in image.info else "RGB"
                return image.convert(target_mode)
            if mode == "LA":
                return image.convert("RGBA")
            if mode == "L":
                return image.convert("RGB")
            if mode not in {"RGB", "RGBA"}:
                raise _UnsupportedPixelModeError(
                    f"WebP 不支持当前像素类型 {mode}；为避免静默丢失位深，已拒绝导出。"
                )
            return image

        raise AssertionError(f"Unhandled raster export format: {export_format}")

    @staticmethod
    def _save_kwargs(
        source_info: dict[str, object],
        options: RasterEncodingOptions,
    ) -> dict[str, object]:
        kwargs: dict[str, object] = {}
        if options.preserve_color_profile:
            profile = source_info.get("icc_profile")
            if isinstance(profile, (bytes, bytearray)) and profile:
                kwargs["icc_profile"] = bytes(profile)
        if options.preserve_resolution:
            dpi = source_info.get("dpi")
            if (
                isinstance(dpi, (tuple, list))
                and len(dpi) >= 2
                and all(isinstance(value, (int, float)) and value > 0 for value in dpi[:2])
            ):
                kwargs["dpi"] = (float(dpi[0]), float(dpi[1]))

        export_format = options.format
        if export_format is RasterExportFormat.PNG:
            kwargs["compress_level"] = options.png_compression
        elif export_format is RasterExportFormat.JPEG:
            quality = int(options.resolved_quality or 95)
            kwargs["quality"] = quality
            kwargs["progressive"] = options.jpeg_progressive
            # High-quality measurement exports should not silently discard
            # chroma detail through 4:2:0 subsampling.
            if quality >= 90:
                kwargs["subsampling"] = 0
            exif = source_info.get("exif")
            if isinstance(exif, (bytes, bytearray)) and exif:
                kwargs["exif"] = bytes(exif)
        elif export_format is RasterExportFormat.TIFF:
            kwargs["compression"] = {
                TiffCompression.DEFLATE: "tiff_adobe_deflate",
                TiffCompression.LZW: "tiff_lzw",
                TiffCompression.NONE: "raw",
            }[options.tiff_compression]
        elif export_format is RasterExportFormat.WEBP:
            kwargs.update(
                {
                    "quality": int(options.resolved_quality or 90),
                    "lossless": options.webp_lossless,
                    "method": options.webp_method,
                    "exact": True,
                }
            )
            exif = source_info.get("exif")
            if isinstance(exif, (bytes, bytearray)) and exif:
                kwargs["exif"] = bytes(exif)
        return kwargs

    @staticmethod
    def _verify_staged_file(
        path: Path,
        export_format: RasterExportFormat,
        *,
        expected_size: tuple[int, int],
        image_module,
    ) -> str | None:
        try:
            with image_module.open(path) as encoded:
                actual_format = str(encoded.format or "").upper()
                actual_size = tuple(int(value) for value in encoded.size)
                encoded.verify()
            if actual_format != export_format.pillow_name:
                return (
                    f"编码格式不一致：期望 {export_format.pillow_name}，"
                    f"实际 {actual_format or 'unknown'}"
                )
            if actual_size != expected_size:
                return f"图片尺寸不一致：期望 {expected_size}，实际 {actual_size}"
        except Exception as exc:
            return str(exc)
        return None

    @staticmethod
    def _failure(
        code: RasterWriteFailureCode,
        format: RasterExportFormat,
        source_path: Path,
        target_path: Path,
        message: str,
        *,
        detail: str = "",
    ) -> RasterWriteResult:
        failure = RasterWriteFailure(
            code=code,
            format=format,
            message=message,
            source_path=source_path,
            target_path=target_path,
            detail=detail,
        )
        return RasterWriteResult(
            success=False,
            format=format,
            path=target_path,
            failure=failure,
        )


def _bounded_int(value: object, *, minimum: int, maximum: int, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label}必须是 {minimum}–{maximum} 的整数。")
    try:
        normalized = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}必须是 {minimum}–{maximum} 的整数。") from exc
    if isinstance(value, float) and value != normalized:
        raise ValueError(f"{label}必须是 {minimum}–{maximum} 的整数。")
    if normalized < minimum or normalized > maximum:
        raise ValueError(f"{label}必须在 {minimum}–{maximum} 之间。")
    return normalized


def _normalize_rgb(value: object, *, label: str) -> tuple[int, int, int]:
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        raise ValueError(f"{label}必须包含三个 0–255 通道值。")
    return tuple(
        _bounded_int(channel, minimum=0, maximum=255, label=label)
        for channel in value
    )  # type: ignore[return-value]
