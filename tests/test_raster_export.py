from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PIL import Image

from fdm.services.raster_export import (
    RasterEncodingOptions,
    RasterExportError,
    RasterExportFormat,
    RasterExportWriter,
    RasterWriteFailureCode,
    TiffCompression,
)


class RasterEncodingOptionsTests(unittest.TestCase):
    def test_format_aliases_suffixes_and_defaults_are_stable(self) -> None:
        self.assertIs(RasterExportFormat.coerce(".JPG"), RasterExportFormat.JPEG)
        self.assertIs(RasterExportFormat.coerce("tiff"), RasterExportFormat.TIFF)
        self.assertEqual(RasterExportFormat.JPEG.canonical_suffix, ".jpg")
        self.assertEqual(
            RasterExportFormat.TIFF.accepted_suffixes,
            (".tif", ".tiff"),
        )
        self.assertEqual(
            RasterEncodingOptions(format=RasterExportFormat.JPEG).resolved_quality,
            95,
        )
        self.assertEqual(
            RasterEncodingOptions(format=RasterExportFormat.WEBP).resolved_quality,
            90,
        )
        self.assertIsNone(
            RasterEncodingOptions(format=RasterExportFormat.PNG).resolved_quality
        )

    def test_options_round_trip_as_plain_json_compatible_values(self) -> None:
        original = RasterEncodingOptions(
            format=RasterExportFormat.WEBP,
            quality=83,
            jpeg_progressive=False,
            png_compression=7,
            tiff_compression=TiffCompression.LZW,
            webp_lossless=True,
            webp_method=6,
            preserve_color_profile=False,
            preserve_resolution=False,
            jpeg_background=(12, 34, 56),
        )

        restored = RasterEncodingOptions.from_dict(original.to_dict())

        self.assertEqual(restored, original)
        self.assertEqual(original.to_dict()["format"], "webp")
        self.assertFalse(original.to_dict()["jpeg_progressive"])
        self.assertEqual(original.to_dict()["jpeg_background"], [12, 34, 56])

    def test_invalid_quality_compression_and_background_are_rejected(self) -> None:
        for kwargs in (
            {"quality": 0},
            {"quality": 101},
            {"quality": 50.5},
            {"png_compression": -1},
            {"png_compression": 10},
            {"webp_method": 7},
            {"jpeg_background": (255, 255)},
            {"jpeg_background": (0, 0, 300)},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                RasterEncodingOptions(**kwargs)


class RasterExportWriterTests(unittest.TestCase):
    @staticmethod
    def _write_rgba_source(path: Path, *, size: tuple[int, int] = (48, 32)) -> None:
        width, height = size
        image = Image.new("RGBA", size)
        pixels = image.load()
        for y in range(height):
            for x in range(width):
                pixels[x, y] = (
                    (x * 37 + y * 11) % 256,
                    (x * 13 + y * 47) % 256,
                    (x * 23 + y * 17) % 256,
                    0 if (x == 0 and y == 0) else 255,
                )
        image.save(path, format="PNG", dpi=(300, 300))
        image.close()

    def test_png_jpeg_tiff_bmp_and_webp_encode_and_verify(self) -> None:
        writer = RasterExportWriter()
        cases = (
            (
                RasterEncodingOptions(
                    format=RasterExportFormat.PNG,
                    png_compression=7,
                ),
                "out.png",
                "PNG",
            ),
            (
                RasterEncodingOptions(
                    format=RasterExportFormat.JPEG,
                    quality=92,
                ),
                "out.jpg",
                "JPEG",
            ),
            (
                RasterEncodingOptions(
                    format=RasterExportFormat.TIFF,
                    tiff_compression=TiffCompression.DEFLATE,
                ),
                "out.tif",
                "TIFF",
            ),
            (
                RasterEncodingOptions(format=RasterExportFormat.BMP),
                "out.bmp",
                "BMP",
            ),
            (
                RasterEncodingOptions(
                    format=RasterExportFormat.WEBP,
                    quality=88,
                ),
                "out.webp",
                "WEBP",
            ),
        )
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source = root / "source.png"
            self._write_rgba_source(source)
            for options, filename, pillow_format in cases:
                with self.subTest(format=options.format):
                    capability = writer.capability(options.format)
                    if not capability.available:
                        self.skipTest(capability.reason)
                    target = root / filename
                    result = writer.write_file(source, target, options)
                    self.assertTrue(result, result.failure)
                    self.assertEqual(result.path, target)
                    self.assertEqual((result.width, result.height), (48, 32))
                    self.assertGreater(result.bytes_written, 0)
                    self.assertEqual(target.stat().st_size, result.bytes_written)
                    with Image.open(target) as encoded:
                        self.assertEqual(encoded.format, pillow_format)
                        self.assertEqual(encoded.size, (48, 32))
                        encoded.load()
                        if options.format in {
                            RasterExportFormat.JPEG,
                            RasterExportFormat.BMP,
                        }:
                            red, green, blue = encoded.convert("RGB").getpixel((0, 0))
                            self.assertGreater(min(red, green, blue), 220)
                        elif options.format in {
                            RasterExportFormat.PNG,
                            RasterExportFormat.TIFF,
                            RasterExportFormat.WEBP,
                        }:
                            self.assertEqual(encoded.convert("RGBA").getpixel((0, 0))[3], 0)

    def test_quality_and_png_compression_are_applied(self) -> None:
        writer = RasterExportWriter()
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source = root / "source.png"
            self._write_rgba_source(source, size=(180, 140))

            low_jpeg = writer.write_file(
                source,
                root / "low.jpg",
                RasterEncodingOptions(
                    format=RasterExportFormat.JPEG,
                    quality=25,
                ),
            )
            high_jpeg = writer.write_file(
                source,
                root / "high.jpg",
                RasterEncodingOptions(
                    format=RasterExportFormat.JPEG,
                    quality=98,
                ),
            )
            self.assertTrue(low_jpeg)
            self.assertTrue(high_jpeg)
            self.assertGreater(high_jpeg.bytes_written, low_jpeg.bytes_written)

            png_uncompressed = writer.write_file(
                source,
                root / "uncompressed.png",
                RasterEncodingOptions(
                    format=RasterExportFormat.PNG,
                    png_compression=0,
                ),
            )
            png_compressed = writer.write_file(
                source,
                root / "compressed.png",
                RasterEncodingOptions(
                    format=RasterExportFormat.PNG,
                    png_compression=9,
                ),
            )
            self.assertTrue(png_uncompressed)
            self.assertTrue(png_compressed)
            self.assertGreater(
                png_uncompressed.bytes_written,
                png_compressed.bytes_written,
            )

    def test_jpeg_progressive_option_is_forwarded_to_encoder(self) -> None:
        writer = RasterExportWriter()
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source = root / "source.png"
            target = root / "progressive.jpg"
            self._write_rgba_source(source, size=(64, 48))

            result = writer.write_file(
                source,
                target,
                RasterEncodingOptions(
                    format=RasterExportFormat.JPEG,
                    quality=95,
                    jpeg_progressive=True,
                ),
            )

            self.assertTrue(result, result.failure)
            with Image.open(target) as encoded:
                self.assertTrue(
                    bool(
                        encoded.info.get("progressive")
                        or encoded.info.get("progression")
                    )
                )

    def test_lossless_webp_preserves_rgba_pixels(self) -> None:
        writer = RasterExportWriter()
        capability = writer.capability(RasterExportFormat.WEBP)
        if not capability.available:
            self.skipTest(capability.reason)
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source = root / "source.png"
            target = root / "lossless.webp"
            self._write_rgba_source(source)

            result = writer.write_file(
                source,
                target,
                RasterEncodingOptions(
                    format=RasterExportFormat.WEBP,
                    webp_lossless=True,
                    quality=100,
                ),
            )

            self.assertTrue(result, result.failure)
            with Image.open(source) as expected, Image.open(target) as actual:
                self.assertEqual(
                    actual.convert("RGBA").tobytes(),
                    expected.convert("RGBA").tobytes(),
                )

    def test_float_png_is_rejected_without_silent_precision_loss(self) -> None:
        writer = RasterExportWriter()
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source = root / "float.tif"
            target = root / "float.png"
            Image.new("F", (8, 6), 1.25).save(source, format="TIFF")

            result = writer.write_file(
                source,
                target,
                RasterEncodingOptions(format=RasterExportFormat.PNG),
            )

            self.assertFalse(result)
            self.assertEqual(
                result.failure.code,
                RasterWriteFailureCode.UNSUPPORTED_PIXEL_MODE,
            )
            self.assertFalse(target.exists())

    def test_missing_backend_and_encoder_are_structured_failures(self) -> None:
        def missing_backend():
            raise ImportError("test missing Pillow")

        writer = RasterExportWriter(pillow_loader=missing_backend)
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source = root / "source.png"
            source.write_bytes(b"not decoded because capability is checked first")

            result = writer.write_file(
                source,
                root / "out.png",
                RasterEncodingOptions(),
            )

            self.assertFalse(result)
            self.assertEqual(
                result.failure.code,
                RasterWriteFailureCode.DEPENDENCY_UNAVAILABLE,
            )
            self.assertEqual(
                result.failure.to_dict()["format"],
                RasterExportFormat.PNG.value,
            )
            with self.assertRaises(RasterExportError) as caught:
                result.require_success()
            self.assertIs(
                caught.exception.code,
                RasterWriteFailureCode.DEPENDENCY_UNAVAILABLE,
            )

        class _NoEncodersImage:
            SAVE: dict[str, object] = {}

        class _Features:
            @staticmethod
            def check(_name: str) -> bool:
                return False

        writer = RasterExportWriter(
            pillow_loader=lambda: (_NoEncodersImage, _Features, "test")
        )
        capability = writer.capability(RasterExportFormat.TIFF)
        self.assertFalse(capability.available)
        self.assertIs(
            capability.failure_code,
            RasterWriteFailureCode.ENCODER_UNAVAILABLE,
        )

    def test_invalid_source_and_target_suffix_are_structured(self) -> None:
        writer = RasterExportWriter()
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            missing = writer.write_file(
                root / "missing.png",
                root / "out.png",
                RasterEncodingOptions(),
            )
            self.assertEqual(
                missing.failure.code,
                RasterWriteFailureCode.INVALID_SOURCE,
            )

            source = root / "source.png"
            self._write_rgba_source(source)
            mismatch = writer.write_file(
                source,
                root / "out.png",
                RasterEncodingOptions(format=RasterExportFormat.JPEG),
            )
            self.assertEqual(
                mismatch.failure.code,
                RasterWriteFailureCode.INVALID_TARGET,
            )
            self.assertFalse((root / "out.png").exists())

    def test_decode_encode_verify_and_atomic_failures_preserve_old_target(self) -> None:
        writer = RasterExportWriter()
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source = root / "source.png"
            target = root / "target.png"
            self._write_rgba_source(source)
            old_payload = b"old target bytes"

            target.write_bytes(old_payload)
            invalid_source = root / "invalid.png"
            invalid_source.write_bytes(b"not an image")
            decoded = writer.write_file(
                invalid_source,
                target,
                RasterEncodingOptions(),
            )
            self.assertEqual(
                decoded.failure.code,
                RasterWriteFailureCode.DECODE_FAILED,
            )
            self.assertEqual(target.read_bytes(), old_payload)

            target.write_bytes(old_payload)
            with patch.object(Image.Image, "save", side_effect=OSError("encode failed")):
                encoded = writer.write_file(
                    source,
                    target,
                    RasterEncodingOptions(),
                )
            self.assertEqual(
                encoded.failure.code,
                RasterWriteFailureCode.ENCODE_FAILED,
            )
            self.assertEqual(target.read_bytes(), old_payload)

            target.write_bytes(old_payload)
            with patch.object(
                writer,
                "_verify_staged_file",
                return_value="verification failed",
            ):
                verified = writer.write_file(
                    source,
                    target,
                    RasterEncodingOptions(),
                )
            self.assertEqual(
                verified.failure.code,
                RasterWriteFailureCode.VERIFY_FAILED,
            )
            self.assertEqual(target.read_bytes(), old_payload)

            target.write_bytes(old_payload)
            with patch(
                "fdm.services.raster_export.atomic_replace_file",
                side_effect=OSError("replace failed"),
            ):
                committed = writer.write_file(
                    source,
                    target,
                    RasterEncodingOptions(),
                )
            self.assertEqual(
                committed.failure.code,
                RasterWriteFailureCode.ATOMIC_COMMIT_FAILED,
            )
            self.assertEqual(target.read_bytes(), old_payload)

    def test_same_source_and_target_is_safe(self) -> None:
        writer = RasterExportWriter()
        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "same.png"
            self._write_rgba_source(path)
            with Image.open(path) as original:
                expected_size = original.size

            result = writer.write_file(
                path,
                path,
                RasterEncodingOptions(
                    format=RasterExportFormat.PNG,
                    png_compression=9,
                ),
            )

            self.assertTrue(result, result.failure)
            with Image.open(path) as reopened:
                self.assertEqual(reopened.size, expected_size)
                reopened.verify()


if __name__ == "__main__":
    unittest.main()
