from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from fdm.image_processing_models import (
    ImageOperationSpec,
    ImageProcessingRecipe,
)
from fdm.services.image_recipe_presets import (
    IMAGE_RECIPE_PRESET_FORMAT,
    IMAGE_RECIPE_PRESET_SCHEMA_VERSION,
    ImageRecipePreset,
    ImageRecipePresetError,
    ImageRecipePresetErrorCode,
    ImageRecipePresetStore,
)


def _recipe(
    operation_id: str = "mean_filter",
    parameters: dict[str, object] | None = None,
) -> ImageProcessingRecipe:
    return ImageProcessingRecipe.from_operations(
        (ImageOperationSpec(operation_id, parameters or {}),)
    )


class ImageRecipePresetStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.path = (
            Path(self.temporary_directory.name)
            / "image-processing-recipes.json"
        )
        self.store = ImageRecipePresetStore(self.path)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_missing_file_round_trip_update_and_remove(self) -> None:
        self.assertEqual(self.store.load(), ())
        created = self.store.upsert(
            "平滑",
            _recipe("mean_filter", {"radius": 2}),
            timestamp="2026-07-27T01:02:03+00:00",
        )

        self.assertTrue(self.path.is_file())
        self.assertEqual(self.store.load(), (created,))
        updated = self.store.upsert(
            "平滑",
            _recipe("gaussian_blur", {"sigma_x": 1.5}),
            timestamp="2026-07-27T02:03:04+00:00",
        )

        self.assertEqual(updated.created_at, created.created_at)
        self.assertEqual(updated.updated_at, "2026-07-27T02:03:04+00:00")
        self.assertEqual(
            self.store.get("平滑").recipe.operations[0].operation_id,
            "gaussian_blur",
        )
        self.assertTrue(self.store.remove("平滑"))
        self.assertFalse(self.store.remove("平滑"))
        self.assertEqual(self.store.load(), ())

    def test_unicode_nfc_and_casefold_names_cannot_collide(self) -> None:
        first = ImageRecipePreset.create(
            "Café",
            _recipe(),
            timestamp="2026-07-27T01:00:00+00:00",
        )
        duplicate = ImageRecipePreset.create(
            "CAFE\u0301",
            _recipe("median_filter", {"radius": 1}),
            timestamp="2026-07-27T01:01:00+00:00",
        )
        self.store.save((first,))
        before = self.path.read_bytes()

        with self.assertRaises(ImageRecipePresetError) as raised:
            self.store.save((first, duplicate))

        self.assertEqual(
            raised.exception.code,
            ImageRecipePresetErrorCode.DUPLICATE_NAME,
        )
        self.assertEqual(self.path.read_bytes(), before)

    def test_unknown_and_image_calculator_operations_are_rejected(self) -> None:
        cases = (
            (
                _recipe("future_magic_filter"),
                ImageRecipePresetErrorCode.UNKNOWN_OPERATION,
            ),
            (
                _recipe(
                    "image_calculator",
                    {"calculator_operation": "add"},
                ),
                ImageRecipePresetErrorCode.UNSUPPORTED_OPERATION,
            ),
        )
        for recipe, error_code in cases:
            with self.subTest(error_code=error_code.value):
                with self.assertRaises(ImageRecipePresetError) as raised:
                    ImageRecipePreset.create(
                        "不安全配方",
                        recipe,
                        timestamp="2026-07-27T01:00:00+00:00",
                    )
                self.assertEqual(raised.exception.code, error_code)

    def test_file_parser_rejects_unknown_fields_versions_and_nonfinite_json(
        self,
    ) -> None:
        preset = ImageRecipePreset.create(
            "有效",
            _recipe(),
            timestamp="2026-07-27T01:00:00+00:00",
        )
        valid = {
            "format": IMAGE_RECIPE_PRESET_FORMAT,
            "schema_version": IMAGE_RECIPE_PRESET_SCHEMA_VERSION,
            "presets": [preset.to_dict()],
        }
        cases: tuple[
            tuple[str, str, ImageRecipePresetErrorCode],
            ...,
        ] = (
            (
                "unknown-root",
                json.dumps(
                    {**valid, "unexpected": True},
                    ensure_ascii=False,
                ),
                ImageRecipePresetErrorCode.INVALID_FILE,
            ),
            (
                "future-version",
                json.dumps(
                    {**valid, "schema_version": 99},
                    ensure_ascii=False,
                ),
                ImageRecipePresetErrorCode.UNSUPPORTED_VERSION,
            ),
            (
                "unknown-preset-field",
                json.dumps(
                    {
                        **valid,
                        "presets": [
                            {**preset.to_dict(), "unexpected": True}
                        ],
                    },
                    ensure_ascii=False,
                ),
                ImageRecipePresetErrorCode.INVALID_PRESET,
            ),
            (
                "nonfinite",
                (
                    '{"format":"fdm.image-processing-recipes",'
                    '"schema_version":1,"presets":[],"value":NaN}'
                ),
                ImageRecipePresetErrorCode.INVALID_FILE,
            ),
        )

        for name, contents, error_code in cases:
            with self.subTest(name=name):
                self.path.write_text(contents, encoding="utf-8")
                with self.assertRaises(ImageRecipePresetError) as raised:
                    self.store.load()
                self.assertEqual(raised.exception.code, error_code)

    def test_atomic_replace_failure_keeps_previous_file_byte_for_byte(self) -> None:
        self.store.upsert(
            "稳定配方",
            _recipe(),
            timestamp="2026-07-27T01:00:00+00:00",
        )
        before = self.path.read_bytes()

        with mock.patch(
            "fdm.atomic_io.os.replace",
            side_effect=OSError("injected replace failure"),
        ):
            with self.assertRaises(ImageRecipePresetError) as raised:
                self.store.upsert(
                    "新配方",
                    _recipe("median_filter", {"radius": 2}),
                    timestamp="2026-07-27T02:00:00+00:00",
                )

        self.assertEqual(
            raised.exception.code,
            ImageRecipePresetErrorCode.WRITE_FAILED,
        )
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(
            tuple(preset.name for preset in self.store.load()),
            ("稳定配方",),
        )


if __name__ == "__main__":
    unittest.main()
