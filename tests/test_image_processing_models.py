from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from fdm.image_processing_models import (
    DisplayTransform,
    ImageDerivation,
    ImageOperationSpec,
    ImageProcessingRecipe,
)
from fdm.models import ImageDocument, ProjectState, project_processed_root
from fdm.project_io import ProjectIO
from fdm.raster import RasterImage, RasterPixelType, RasterPlane


class RasterPlaneTests(unittest.TestCase):
    def test_pixel_type_describes_canonical_layout(self) -> None:
        self.assertEqual(RasterPixelType.GRAY16.bytes_per_pixel, 2)
        self.assertEqual(RasterPixelType.RGB8.bytes_per_pixel, 3)
        self.assertEqual(RasterPixelType.RGBA8.channel_count, 4)
        self.assertEqual(RasterPixelType.GRAY16.sample_maximum, 65_535)
        self.assertTrue(RasterPixelType.GRAY8.is_grayscale)
        self.assertTrue(RasterPixelType.RGBA8.has_alpha)
        self.assertIs(
            RasterPixelType.parse(" GrAy16 "),
            RasterPixelType.GRAY16,
        )

        with self.assertRaisesRegex(ValueError, "不支持的栅格像素类型"):
            RasterPixelType.parse("float32")

    def test_plane_copies_input_and_validates_tightly_packed_size(self) -> None:
        source = bytearray([1, 2, 3, 4])
        plane = RasterPlane(
            width=2,
            height=2,
            pixel_type=RasterPixelType.GRAY8,
            data=source,
        )
        source[0] = 99

        self.assertEqual(plane.data, b"\x01\x02\x03\x04")
        self.assertEqual(plane.row_bytes, 2)
        self.assertEqual(plane.byte_count, 4)
        with self.assertRaises(FrozenInstanceError):
            plane.width = 3  # type: ignore[misc]
        with self.assertRaisesRegex(ValueError, "栅格字节数不匹配"):
            RasterPlane(
                width=2,
                height=2,
                pixel_type=RasterPixelType.GRAY16,
                data=b"\x00" * 4,
            )

    def test_plane_hash_includes_layout_and_legacy_bridge_is_explicit(self) -> None:
        legacy = RasterImage.from_rows([[1, 2], [3, 4]])
        plane = RasterPlane.from_raster_image(legacy)
        same_bytes_different_layout = RasterPlane(
            width=1,
            height=1,
            pixel_type=RasterPixelType.RGBA8,
            data=plane.data,
        )

        self.assertEqual(plane.to_raster_image().to_rows(), legacy.to_rows())
        self.assertNotEqual(plane.sha256(), same_bytes_different_layout.sha256())
        with self.assertRaisesRegex(ValueError, "只有 gray8"):
            same_bytes_different_layout.to_raster_image()


class ImageProcessingPersistenceModelTests(unittest.TestCase):
    def test_display_transform_roundtrip_and_finite_validation(self) -> None:
        transform = DisplayTransform(
            black_point=128.0,
            white_point=4095.0,
            gamma=0.8,
            inverted=True,
        )

        self.assertEqual(
            DisplayTransform.from_dict(transform.to_dict()),
            transform,
        )
        self.assertFalse(transform.is_identity)
        self.assertTrue(DisplayTransform().is_identity)
        with self.assertRaisesRegex(ValueError, "同时提供"):
            DisplayTransform(black_point=0.0)
        with self.assertRaisesRegex(ValueError, "white_point"):
            DisplayTransform(black_point=5.0, white_point=5.0)
        with self.assertRaisesRegex(ValueError, "有限数值"):
            DisplayTransform(gamma=float("nan"))
        with self.assertRaisesRegex(ValueError, "schema_version"):
            DisplayTransform.from_dict({"schema_version": 2})

    def test_operation_parameters_are_canonical_and_deeply_detached(self) -> None:
        parameters = {
            "gain": 1.25,
            "kernel": [3, 5],
            "border": {"mode": "reflect"},
        }
        operation = ImageOperationSpec(
            "adjust.brightness_contrast",
            parameters,
            implementation_version="opencv-4.10+fdm1",
        )
        parameters["kernel"][0] = 99
        returned = operation.parameters
        returned["border"]["mode"] = "constant"

        self.assertEqual(operation.parameters["kernel"], [3, 5])
        self.assertEqual(operation.parameters["border"], {"mode": "reflect"})
        self.assertEqual(
            ImageOperationSpec.from_dict(operation.to_dict()),
            operation,
        )
        json.dumps(operation.to_dict(), allow_nan=False)
        with self.assertRaisesRegex(ValueError, "NaN"):
            ImageOperationSpec("adjust.gamma", {"gamma": float("nan")})
        with self.assertRaisesRegex(TypeError, "键必须是字符串"):
            ImageOperationSpec("adjust.gamma", {1: "invalid"})  # type: ignore[dict-item]
        with self.assertRaisesRegex(TypeError, "必须是对象"):
            ImageOperationSpec("adjust.gamma", [])  # type: ignore[arg-type]

    def test_recipe_and_derivation_roundtrip(self) -> None:
        recipe = ImageProcessingRecipe.from_operations(
            (
                ImageOperationSpec(
                    "type.convert",
                    {"target": "gray16", "mapping": "full_range"},
                ),
                ImageOperationSpec(
                    "transform.flip_horizontal",
                    {},
                ),
            )
        )
        derivation = ImageDerivation(
            source_document_id="image_source",
            source_path="source/input.tif",
            source_sha256="a" * 64,
            source_image_size=(4096, 3072),
            source_pixel_revision=7,
            source_pixel_type=RasterPixelType.GRAY8,
            recipe=recipe,
            result_pixel_type=RasterPixelType.GRAY16,
            result_image_size=(3072, 4096),
            result_sha256="b" * 64,
            library_versions=(
                ("opencv", "4.13.0"),
                ("numpy", "2.4.0"),
            ),
            created_at="2026-07-27T12:34:56+00:00",
        )

        loaded = ImageDerivation.from_dict(derivation.to_dict())

        self.assertEqual(loaded, derivation)
        self.assertIs(
            loaded.result_pixel_type,
            RasterPixelType.GRAY16,
        )
        self.assertEqual(
            [item.operation_id for item in loaded.recipe.operations],
            ["type.convert", "transform.flip_horizontal"],
        )
        self.assertEqual(loaded.source_pixel_revision, 7)
        self.assertEqual(loaded.result_image_size, (3072, 4096))
        self.assertEqual(dict(loaded.library_versions)["opencv"], "4.13.0")
        with self.assertRaisesRegex(ValueError, "至少需要一个"):
            ImageProcessingRecipe(operations=())
        with self.assertRaisesRegex(TypeError, "必须全部是对象"):
            ImageProcessingRecipe.from_dict(
                {"schema_version": 1, "operations": ["invalid"]}
            )
        with self.assertRaisesRegex(ValueError, "SHA256"):
            ImageDerivation(
                source_document_id="image_source",
                source_sha256="bad",
                recipe=recipe,
            )

    def test_old_document_payload_remains_sparse_and_safe(self) -> None:
        legacy_payload = {
            "id": "image_legacy",
            "path": "/tmp/legacy.png",
            "image_size": [640, 480],
            "fiber_groups": [],
            "measurements": [],
        }

        document = ImageDocument.from_dict(legacy_payload)
        serialized = document.to_dict()

        self.assertIsNone(document.raster_pixel_type)
        self.assertIsNone(document.display_transform)
        self.assertIsNone(document.derivation)
        self.assertNotIn("raster_pixel_type", serialized)
        self.assertNotIn("display_transform", serialized)
        self.assertNotIn("derivation", serialized)

    def test_project_roundtrip_persists_descriptors_not_raster_bytes(self) -> None:
        recipe = ImageProcessingRecipe.from_operations(
            [ImageOperationSpec("adjust.gamma", {"gamma": 1.2})]
        )
        document = ImageDocument(
            id="image_derived",
            path="processed/result.png",
            image_size=(800, 600),
            source_type="project_asset",
            raster_pixel_type=RasterPixelType.GRAY16,
            display_transform=DisplayTransform(
                black_point=0.0,
                white_point=65_535.0,
            ),
            derivation=ImageDerivation(
                source_document_id="image_source",
                source_image_size=(800, 600),
                source_pixel_type=RasterPixelType.GRAY16,
                recipe=recipe,
                result_pixel_type=RasterPixelType.GRAY16,
                created_at="2026-07-27T12:34:56+00:00",
            ),
        )
        document.initialize_runtime_state()
        project = ProjectState(version="0.3.5", documents=[document])

        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "derived.fdmproj"
            ProjectIO.save(project, path)
            raw_payload = json.loads(path.read_text(encoding="utf-8"))
            loaded = ProjectIO.load(path)

        document_payload = raw_payload["documents"][0]
        self.assertEqual(document_payload["raster_pixel_type"], "gray16")
        self.assertNotIn("data", document_payload)
        self.assertEqual(
            document_payload["derivation"]["recipe"]["schema_version"],
            1,
        )
        loaded_document = loaded.documents[0]
        self.assertIs(
            loaded_document.raster_pixel_type,
            RasterPixelType.GRAY16,
        )
        self.assertEqual(loaded_document.derivation, document.derivation)
        self.assertEqual(
            loaded_document.display_transform,
            document.display_transform,
        )

    def test_processed_asset_root_is_separate_from_capture_assets(self) -> None:
        project_path = Path("/tmp/demo.fdmproj")

        self.assertEqual(
            project_processed_root(project_path),
            Path("/tmp/demo.assets/processed"),
        )

    def test_project_io_loads_v1_project_without_new_fields(self) -> None:
        payload = {
            "version": "0.1.0",
            "documents": [
                {
                    "id": "image_v1",
                    "path": "/tmp/v1.png",
                    "image_size": [32, 24],
                    "fiber_groups": [],
                    "measurements": [],
                }
            ],
            "metadata": {},
        }
        with TemporaryDirectory() as tmp_dir:
            source = Path(tmp_dir) / "v1.fdmproj"
            source.write_text(
                json.dumps(payload, ensure_ascii=False),
                encoding="utf-8",
            )
            project = ProjectIO.load(source)

        self.assertEqual(project.version, "0.1.0")
        self.assertIsNone(project.documents[0].derivation)
        self.assertIsNone(project.documents[0].raster_pixel_type)


if __name__ == "__main__":
    unittest.main()
