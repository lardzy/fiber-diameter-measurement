from __future__ import annotations

import unittest

import cv2
import numpy as np

from fdm.services import image_processing as image_processing_service
from fdm.image_processing_models import (
    ImageOperationSpec,
    ImageProcessingRecipe,
    RasterSemantic,
    RasterTypeState,
    RoiProcessingSemantics,
)
from fdm.raster import RasterPixelType
from fdm.services.image_processing import (
    ConversionScaleMode,
    IMAGE_OPERATION_REGISTRY,
    ImageOperation,
    ImageOperationRequest,
    ImageRecipeValidationError,
    InterpolationMode,
    NonfiniteIntegerPolicy,
    PixelType,
    convert_pixel_type,
    execute_image_operation,
    validate_image_processing_recipe,
)


class ImageProcessingServiceTests(unittest.TestCase):
    def test_scientific_v2_operations_do_not_change_legacy_ids(self) -> None:
        source = np.asarray([[0, 1, 3, 8]], dtype=np.uint8)

        legacy = execute_image_operation(
            ImageOperationRequest.create(ImageOperation.LOG, source)
        )
        scientific = execute_image_operation(
            ImageOperationRequest.create(ImageOperation.LOG_V2, source)
        )
        preserved = execute_image_operation(
            ImageOperationRequest.create(
                ImageOperation.LOG_V2,
                source,
                result_mode="preserve",
            )
        )
        remapped = execute_image_operation(
            ImageOperationRequest.create(
                ImageOperation.SQRT_V2,
                source,
                result_mode="remap",
                output_min=10.0,
                output_max=200.0,
            )
        )

        self.assertEqual(legacy.image.dtype, np.uint8)
        self.assertEqual(scientific.image.dtype, np.float32)
        np.testing.assert_allclose(
            scientific.image,
            np.log1p(source.astype(np.float64)),
            rtol=1e-6,
        )
        np.testing.assert_array_equal(preserved.image, legacy.image)
        self.assertEqual(remapped.image.dtype, np.uint8)
        self.assertEqual(int(remapped.image.min()), 10)
        self.assertEqual(int(remapped.image.max()), 200)

        validation = validate_image_processing_recipe(
            ImageProcessingRecipe.from_operations(
                (
                    ImageOperationSpec("log_v2"),
                    ImageOperationSpec(
                        "adaptive_threshold",
                        {"method": "sauvola", "radius": 2},
                    ),
                )
            ),
            RasterTypeState(RasterPixelType.GRAY16, width=4, height=1),
        )
        self.assertIs(
            validation.steps[0].output_state.pixel_type,
            RasterPixelType.GRAY32_FLOAT,
        )
        self.assertIs(
            validation.output_state.semantic,
            RasterSemantic.BINARY_MASK,
        )

    def test_background_v1_is_stable_and_v2_is_independently_registered(self) -> None:
        y, x = np.mgrid[:25, :25]
        source = np.clip(20 + x + y, 0, 255).astype(np.uint8)
        source[10:15, 10:15] = 220

        legacy = execute_image_operation(
            ImageOperationRequest.create(
                ImageOperation.BACKGROUND_SUBTRACT,
                source,
                radius=3,
            )
        )
        expected_legacy = image_processing_service.subtract_background(
            source,
            radius=3,
        )
        rolling = execute_image_operation(
            ImageOperationRequest.create(
                ImageOperation.ROLLING_BALL_BACKGROUND_SUBTRACT,
                source,
                radius=4.0,
                ball_height=80.0,
            )
        )

        np.testing.assert_array_equal(legacy.image, expected_legacy)
        self.assertEqual(
            IMAGE_OPERATION_REGISTRY["background_subtract"].chinese_name,
            "形态学背景扣除",
        )
        self.assertEqual(rolling.image.dtype, source.dtype)
        self.assertGreater(float(np.mean(rolling.image[10:15, 10:15])), 0.0)

    def test_fft_boundary_units_and_power_spectrum_are_explicit(self) -> None:
        source = np.zeros((32, 32), dtype=np.float32)
        source[:, ::4] = 1.0

        implicit_periodic = image_processing_service.fft_filter(
            source,
            mode="lowpass",
            high_cutoff=0.2,
        )
        explicit_periodic = image_processing_service.fft_filter(
            source,
            mode="lowpass",
            high_cutoff=0.2,
            boundary="periodic",
        )
        mirror = image_processing_service.fft_filter(
            source,
            mode="lowpass",
            high_cutoff=0.2,
            boundary="mirror_pad",
        )
        tukey = image_processing_service.fft_filter(
            source,
            mode="lowpass",
            high_cutoff=0.2,
            boundary="tukey",
            tukey_alpha=0.5,
        )

        np.testing.assert_array_equal(implicit_periodic, explicit_periodic)
        self.assertEqual(mirror.shape, source.shape)
        self.assertEqual(tukey.shape, source.shape)
        with self.assertRaisesRegex(ValueError, "pixel_size"):
            image_processing_service.fft_filter(
                source,
                high_cutoff=1.0,
                frequency_unit="cycles_per_unit",
            )
        physical = image_processing_service.fft_filter(
            source,
            high_cutoff=1.0,
            frequency_unit="cycles_per_unit",
            pixel_size=0.2,
        )
        np.testing.assert_allclose(physical, explicit_periodic)

        spectrum = image_processing_service.fft_power_spectrum(
            source,
            window="tukey",
        )
        self.assertEqual(spectrum.dtype, np.float32)
        self.assertEqual(spectrum.shape, source.shape)
        self.assertTrue(np.all(spectrum >= 0))

    def test_vectorized_fill_holes_and_copy_rgb_services(self) -> None:
        binary = np.zeros((8, 8), dtype=np.uint8)
        binary[1:7, 1:7] = 255
        binary[3:5, 3:5] = 0
        checks: list[int] = []

        filled = image_processing_service.fill_binary_holes(
            binary,
            cancellation_check=lambda: checks.append(1),
        )

        self.assertEqual(len(checks), 3)
        self.assertTrue(np.all(filled[3:5, 3:5] == 255))

        rgb = np.dstack((binary, binary // 2, 255 - binary))
        red, green, blue = image_processing_service.split_rgb_channels(rgb)
        np.testing.assert_array_equal(
            image_processing_service.merge_rgb_channels(red, green, blue),
            rgb,
        )

        roi = np.zeros(binary.shape, dtype=bool)
        roi[2:6, 3:7] = True
        copied = execute_image_operation(
            ImageOperationRequest.create(
                ImageOperation.COPY,
                binary,
                roi_mask=roi,
                roi_mode="mask",
                outside_value=17.0,
            )
        )
        self.assertEqual(copied.image.shape, (4, 4))
        np.testing.assert_array_equal(copied.roi_mask, roi[2:6, 3:7])
        self.assertTrue(copied.image.flags.writeable is False)

    def test_new_local_morphology_and_flat_field_operations(self) -> None:
        source = np.full((25, 25), 40, dtype=np.uint8)
        source[8:17, 8:17] = 180
        binary = np.zeros_like(source)
        binary[0:4, 0:4] = 255
        binary[8:17, 8:17] = 255

        for method in ("mean", "gaussian", "sauvola", "phansalkar"):
            thresholded = image_processing_service.adaptive_threshold(
                source,
                method=method,
                radius=2,
            )
            self.assertEqual(thresholded.dtype, source.dtype)
            self.assertEqual(set(np.unique(thresholded)).issubset({0, 255}), True)

        variance = image_processing_service.rank_filter(
            source,
            method="variance",
            radius=1,
        )
        self.assertEqual(variance.dtype, np.float32)
        enhanced = image_processing_service.percentile_saturation_enhance(
            source,
            lower_percentile=1.0,
            upper_percentile=99.0,
        )
        self.assertEqual(enhanced.dtype, source.dtype)
        gradient = image_processing_service.morphology_derivative(
            source,
            method="gradient",
            radius=1,
        )
        laplacian = image_processing_service.morphology_derivative(
            source,
            method="laplacian",
            radius=1,
        )
        self.assertEqual(gradient.dtype, source.dtype)
        self.assertEqual(laplacian.dtype, np.float32)
        reconstruction = image_processing_service.morphological_reconstruction(
            source,
            method="opening",
            radius=1,
        )
        self.assertEqual(reconstruction.shape, source.shape)
        extrema = image_processing_service.regional_extrema(
            source,
            kind="maxima",
            h=5.0,
        )
        self.assertEqual(set(np.unique(extrema)).issubset({0, 255}), True)
        cleared = image_processing_service.clear_border_objects(binary)
        self.assertTrue(np.all(cleared[:4, :4] == 0))
        self.assertTrue(np.all(cleared[8:17, 8:17] == 255))
        watershed_v2 = image_processing_service.watershed_split_v2(
            binary,
            seed_threshold=0.25,
        )
        self.assertEqual(watershed_v2.dtype, binary.dtype)
        corrected = image_processing_service.flat_field_correction(
            source,
            radius=2.0,
        )
        self.assertEqual(corrected.dtype, source.dtype)

    def test_image_calculator_copy_and_float32_result(self) -> None:
        left = np.asarray([[1, 2], [3, 4]], dtype=np.uint16)
        right = np.asarray([[5, 6], [7, 8]], dtype=np.uint16)

        copied = image_processing_service.image_calculator(
            left,
            right,
            operation="copy",
        )
        floated = image_processing_service.image_calculator(
            left,
            right,
            operation="mean",
            result_mode="float32",
        )

        np.testing.assert_array_equal(copied, right)
        self.assertEqual(floated.dtype, np.float32)
        np.testing.assert_allclose(floated, (left + right) / 2.0)

    def test_reference_flat_field_is_exact_auditable_and_preserves_alpha(
        self,
    ) -> None:
        source = np.asarray(
            [
                [[100, 120, 140, 11], [80, 90, 100, 22]],
                [[60, 70, 80, 33], [40, 50, 60, 44]],
            ],
            dtype=np.uint8,
        )
        reference = np.asarray(
            [
                [[50, 60, 70, 0], [100, 90, 80, 0]],
                [[150, 120, 100, 0], [200, 150, 120, 0]],
            ],
            dtype=np.uint8,
        )
        levels = image_processing_service.flat_field_reference_levels(
            reference
        )

        result = execute_image_operation(
            ImageOperationRequest.create(
                ImageOperation.FLAT_FIELD_CORRECTION,
                source,
                secondary_image=reference,
                flat_field_source="reference",
                secondary_document_id="reference-document",
                secondary_sha256="a" * 64,
                reference_levels=levels,
                preserve_mean=True,
            )
        ).image

        expected = source.copy()
        for channel, level in enumerate(levels):
            expected[..., channel] = np.rint(
                np.clip(
                    source[..., channel].astype(np.float32)
                    * level
                    / reference[..., channel].astype(np.float32),
                    0,
                    255,
                )
            ).astype(np.uint8)
        np.testing.assert_array_equal(result, expected)
        np.testing.assert_array_equal(result[..., 3], source[..., 3])

        with self.assertRaisesRegex(ValueError, "尺寸和通道"):
            image_processing_service.flat_field_correction(
                source,
                reference_image=reference[:, :1],
            )
        with self.assertRaisesRegex(ValueError, "像素类型"):
            image_processing_service.flat_field_correction(
                source,
                reference_image=reference.astype(np.uint16),
            )
        invalid = reference.copy()
        invalid[0, 0, 0] = 0
        with self.assertRaisesRegex(ValueError, "正有限值"):
            image_processing_service.flat_field_reference_levels(invalid)

    def test_request_and_result_are_detached_read_only_snapshots(self) -> None:
        source = np.arange(25, dtype=np.uint8).reshape(5, 5)
        request = ImageOperationRequest.create(
            ImageOperation.MEAN_FILTER,
            source,
            radius=1,
            request_id="request-1",
            generation=7,
        )
        source[0, 0] = 255

        self.assertEqual(int(request.image[0, 0]), 0)
        with self.assertRaises(ValueError):
            request.image[0, 0] = 1

        result = execute_image_operation(request)

        self.assertEqual(result.request_id, "request-1")
        self.assertEqual(result.generation, 7)
        with self.assertRaises(ValueError):
            result.image[0, 0] = 1

    def test_type_conversion_has_explicit_preserve_and_full_range_rules(self) -> None:
        eight_bit = np.asarray([[0, 128, 255]], dtype=np.uint8)

        default_conversion = convert_pixel_type(eight_bit, PixelType.UINT16)
        full_range = convert_pixel_type(
            eight_bit,
            PixelType.UINT16,
            mode=ConversionScaleMode.FULL_TYPE_RANGE,
        )
        preserved = convert_pixel_type(
            eight_bit,
            PixelType.UINT16,
            mode=ConversionScaleMode.PRESERVE_VALUES,
        )

        np.testing.assert_array_equal(full_range, [[0, 32896, 65535]])
        np.testing.assert_array_equal(preserved, [[0, 128, 255]])
        np.testing.assert_array_equal(default_conversion, preserved)
        self.assertEqual(full_range.dtype, np.uint16)

        request_result = execute_image_operation(
            ImageOperationRequest.create(
                ImageOperation.CONVERT_TYPE,
                eight_bit,
                target_type="uint16",
            )
        )
        np.testing.assert_array_equal(request_result.image, preserved)
        self.assertEqual(
            request_result.metadata_map["scale_mode"],
            "preserve_values",
        )

    def test_registry_is_complete_and_exposes_non_ui_contract(self) -> None:
        self.assertEqual(
            set(IMAGE_OPERATION_REGISTRY),
            {operation.value for operation in ImageOperation},
        )
        descriptor = IMAGE_OPERATION_REGISTRY["convert_type"]
        self.assertEqual(descriptor.chinese_name, "转换像素类型")
        self.assertEqual(descriptor.category, "类型")
        self.assertIn("scale_mode", descriptor.parameters)
        self.assertEqual(descriptor.version, "1")
        self.assertTrue(callable(descriptor.output_resolver))
        self.assertTrue(callable(descriptor.tile))
        self.assertTrue(callable(descriptor.executor))
        with self.assertRaises(TypeError):
            IMAGE_OPERATION_REGISTRY["new"] = descriptor  # type: ignore[index]

    def test_descriptor_parameter_schema_validates_recipe_values(self) -> None:
        state = RasterTypeState(
            RasterPixelType.GRAY8,
            width=32,
            height=24,
        )
        cases = (
            (
                ImageOperationSpec("resize", {"width": "32", "height": 24}),
                "width",
            ),
            (
                ImageOperationSpec(
                    "resize",
                    {"width": 32, "height": 24, "interpolation": "magic"},
                ),
                "interpolation",
            ),
            (
                ImageOperationSpec("median_filter", {"radius": 0}),
                "radius",
            ),
            (
                ImageOperationSpec("brightness_contrast", {"contrast": True}),
                "contrast",
            ),
        )
        for operation, parameter_name in cases:
            with self.subTest(
                operation=operation.operation_id,
                parameter=parameter_name,
            ):
                with self.assertRaisesRegex(
                    ImageRecipeValidationError,
                    parameter_name,
                ):
                    validate_image_processing_recipe(
                        ImageProcessingRecipe.from_operations((operation,)),
                        state,
                    )

    def test_legacy_recipe_can_omit_optional_descriptor_defaults(self) -> None:
        result = validate_image_processing_recipe(
            ImageProcessingRecipe.from_operations(
                (
                    ImageOperationSpec("brightness_contrast"),
                    ImageOperationSpec("median_filter"),
                )
            ),
            RasterTypeState(
                RasterPixelType.GRAY8,
                width=32,
                height=24,
            ),
        )

        self.assertEqual(len(result.steps), 2)
        self.assertIs(
            result.output_state.pixel_type,
            RasterPixelType.GRAY8,
        )

    def test_descriptor_schema_enforces_conditional_required_parameters(
        self,
    ) -> None:
        state = RasterTypeState(
            RasterPixelType.GRAY8,
            width=32,
            height=24,
        )
        operation = ImageOperationSpec(
            "fft_filter",
            {"frequency_unit": "cycles_per_unit"},
        )

        with self.assertRaisesRegex(
            ImageRecipeValidationError,
            "pixel_size",
        ):
            validate_image_processing_recipe(
                ImageProcessingRecipe.from_operations((operation,)),
                state,
            )

    def test_reference_flat_field_recipe_requires_matching_second_state(
        self,
    ) -> None:
        operation = ImageOperationSpec(
            ImageOperation.FLAT_FIELD_CORRECTION.value,
            {
                "flat_field_source": "reference",
                "secondary_document_id": "flat-reference",
                "secondary_sha256": "b" * 64,
                "reference_levels": [120.0],
                "preserve_mean": True,
                "radius": 25.0,
                "method": "gaussian",
            },
        )
        recipe = ImageProcessingRecipe.from_operations((operation,))
        source_state = RasterTypeState(
            RasterPixelType.GRAY8,
            width=12,
            height=8,
        )

        validated = validate_image_processing_recipe(
            recipe,
            source_state,
            secondary_states={"flat-reference": source_state},
        )

        self.assertIs(validated.output_state.pixel_type, RasterPixelType.GRAY8)
        with self.assertRaisesRegex(
            ImageRecipeValidationError,
            "缺少第二幅参考图像",
        ):
            validate_image_processing_recipe(recipe, source_state)
        with self.assertRaisesRegex(
            ImageRecipeValidationError,
            "尺寸、通道和像素类型",
        ):
            validate_image_processing_recipe(
                recipe,
                source_state,
                secondary_states={
                    "flat-reference": RasterTypeState(
                        RasterPixelType.GRAY16,
                        width=12,
                        height=8,
                    )
                },
            )
        invalid_sha_recipe = ImageProcessingRecipe.from_operations(
            (
                ImageOperationSpec(
                    ImageOperation.FLAT_FIELD_CORRECTION.value,
                    {
                        **operation.parameters,
                        "secondary_sha256": "not-a-sha",
                    },
                ),
            )
        )
        with self.assertRaisesRegex(
            ImageRecipeValidationError,
            "64 位 SHA256",
        ):
            validate_image_processing_recipe(
                invalid_sha_recipe,
                source_state,
                secondary_states={"flat-reference": source_state},
            )

    def test_recipe_validation_resolves_the_complete_type_chain(self) -> None:
        recipe = ImageProcessingRecipe.from_operations(
            (
                ImageOperationSpec(
                    "convert_color",
                    {"target_model": "grayscale"},
                ),
                ImageOperationSpec(
                    "convert_type",
                    {"target_type": "uint16"},
                ),
                ImageOperationSpec(
                    "threshold",
                    {"lower": 10, "upper": 200},
                ),
            )
        )

        validation = validate_image_processing_recipe(
            recipe,
            RasterTypeState(
                RasterPixelType.RGB8,
                width=640,
                height=480,
            ),
        )

        self.assertEqual(len(validation.steps), 3)
        self.assertIs(
            validation.steps[0].output_state.pixel_type,
            RasterPixelType.GRAY8,
        )
        self.assertIs(
            validation.steps[1].output_state.pixel_type,
            RasterPixelType.GRAY16,
        )
        self.assertIs(
            validation.output_state.semantic,
            RasterSemantic.BINARY_MASK,
        )
        self.assertEqual(
            (validation.output_state.width, validation.output_state.height),
            (640, 480),
        )

        invalid = ImageProcessingRecipe.from_operations(
            (
                ImageOperationSpec(
                    "convert_type",
                    {"target_type": "uint16"},
                ),
            )
        )
        with self.assertRaisesRegex(
            ImageRecipeValidationError,
            "配方步骤 1.*请先转换为灰度",
        ):
            validate_image_processing_recipe(
                invalid,
                RasterTypeState(RasterPixelType.RGB8),
            )

    def test_crop_roi_bounds_mask_fill_and_transparency_contract(self) -> None:
        source = np.arange(30, dtype=np.uint8).reshape(5, 6)
        roi = np.zeros((5, 6), dtype=bool)
        roi[1:4, 2:5] = True

        bounded = execute_image_operation(
            ImageOperationRequest.create(
                ImageOperation.CROP,
                source,
                roi_mask=roi,
                x=1,
                y=1,
                width=4,
                height=3,
                roi_mode="bounds",
            )
        )
        np.testing.assert_array_equal(bounded.image, source[1:4, 1:5])
        np.testing.assert_array_equal(bounded.roi_mask, roi[1:4, 1:5])

        masked = execute_image_operation(
            ImageOperationRequest.create(
                ImageOperation.CROP,
                source,
                roi_mask=roi,
                x=1,
                y=1,
                width=4,
                height=3,
                roi_mode="mask",
                fill_value=99,
            )
        )
        expected_mask = roi[1:4, 1:5]
        expected = source[1:4, 1:5].copy()
        expected[~expected_mask] = 99
        np.testing.assert_array_equal(masked.image, expected)
        np.testing.assert_array_equal(masked.roi_mask, expected_mask)
        with self.assertRaises(ValueError):
            masked.roi_mask[0, 0] = True  # type: ignore[index]

        transparent = execute_image_operation(
            ImageOperationRequest.create(
                ImageOperation.CROP,
                source,
                roi_mask=roi,
                x=1,
                y=1,
                width=4,
                height=3,
                roi_mode="mask",
                transparent_outside=True,
            )
        )
        self.assertEqual(transparent.image.shape, (3, 4, 4))
        np.testing.assert_array_equal(
            transparent.image[..., 3],
            expected_mask.astype(np.uint8) * 255,
        )

        validation = validate_image_processing_recipe(
            ImageProcessingRecipe.from_operations(
                (
                    ImageOperationSpec(
                        "crop",
                        {
                            "x": 1,
                            "y": 1,
                            "width": 4,
                            "height": 3,
                            "roi_mode": "mask",
                            "transparent_outside": True,
                        },
                    ),
                )
            ),
            RasterTypeState(
                RasterPixelType.GRAY8,
                width=6,
                height=5,
            ),
            roi_requested=True,
        )
        self.assertIs(
            validation.output_state.pixel_type,
            RasterPixelType.RGBA8,
        )
        self.assertIs(
            validation.steps[0].descriptor.roi_semantics,
            RoiProcessingSemantics.CROP_BOUNDS_OR_MASK,
        )

        with self.assertRaisesRegex(ImageRecipeValidationError, "不支持 ROI"):
            validate_image_processing_recipe(
                ImageProcessingRecipe.from_operations(
                    (ImageOperationSpec("resize", {"width": 2, "height": 2}),)
                ),
                RasterTypeState(RasterPixelType.GRAY8, width=6, height=5),
                roi_requested=True,
            )

    def test_registry_exposes_scientific_roi_semantics(self) -> None:
        state = RasterTypeState(RasterPixelType.GRAY8, width=8, height=8)

        def semantics(
            operation_id: str,
            input_state: RasterTypeState = state,
        ) -> RoiProcessingSemantics:
            result = validate_image_processing_recipe(
                ImageProcessingRecipe.from_operations(
                    (ImageOperationSpec(operation_id, {}),)
                ),
                input_state,
                roi_requested=True,
            )
            return result.steps[0].descriptor.roi_semantics

        self.assertIs(
            semantics("normalize"),
            RoiProcessingSemantics.ROI_STATISTICS,
        )
        self.assertIs(
            semantics(
                "remove_small_objects",
                RasterTypeState(
                    RasterPixelType.GRAY8,
                    semantic=RasterSemantic.BINARY_MASK,
                    width=8,
                    height=8,
                ),
            ),
            RoiProcessingSemantics.ISOLATED_DOMAIN,
        )
        self.assertIs(
            semantics("gaussian_blur"),
            RoiProcessingSemantics.WRITE_MASK_WITH_HALO,
        )

    def test_float_to_integer_requires_explicit_nonfinite_policy_and_reports_count(
        self,
    ) -> None:
        source = np.asarray(
            [[np.nan, np.inf, -np.inf, 0.5]],
            dtype=np.float32,
        )

        with self.assertRaisesRegex(ValueError, "必须明确选择"):
            convert_pixel_type(source, PixelType.UINT8)

        zeroed = execute_image_operation(
            ImageOperationRequest.create(
                ImageOperation.CONVERT_TYPE,
                source,
                target_type="uint8",
                scale_mode="full_type_range",
                nonfinite_policy=NonfiniteIntegerPolicy.ZERO.value,
            )
        )
        bounded = execute_image_operation(
            ImageOperationRequest.create(
                ImageOperation.CONVERT_TYPE,
                source,
                target_type="uint8",
                scale_mode="full_type_range",
                nonfinite_policy=NonfiniteIntegerPolicy.RANGE_BOUNDS.value,
            )
        )

        np.testing.assert_array_equal(zeroed.image, [[0, 0, 0, 128]])
        np.testing.assert_array_equal(bounded.image, [[0, 255, 0, 128]])
        self.assertEqual(
            zeroed.metadata_map["nonfinite_replacement_count"],
            3,
        )
        self.assertEqual(
            zeroed.metadata_map["nonfinite_policy"],
            "zero",
        )
        self.assertEqual(
            bounded.metadata_map["nonfinite_replacement_count"],
            3,
        )

    def test_float_output_preserves_nonfinite_samples(self) -> None:
        source = np.asarray([[np.nan, np.inf, -np.inf, 0.5]], dtype=np.float32)

        result = execute_image_operation(
            ImageOperationRequest.create(
                ImageOperation.CONVERT_TYPE,
                source,
                target_type="float32",
            )
        )

        self.assertTrue(np.isnan(result.image[0, 0]))
        self.assertTrue(np.isposinf(result.image[0, 1]))
        self.assertTrue(np.isneginf(result.image[0, 2]))
        self.assertEqual(
            result.metadata_map["nonfinite_replacement_count"],
            0,
        )

    def test_all_nonfinite_data_range_conversion_still_requires_policy(
        self,
    ) -> None:
        source = np.asarray([[np.nan, np.inf, -np.inf]], dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "必须明确选择"):
            convert_pixel_type(
                source,
                PixelType.UINT16,
                mode=ConversionScaleMode.DATA_RANGE,
            )
        zeroed = convert_pixel_type(
            source,
            PixelType.UINT16,
            mode=ConversionScaleMode.DATA_RANGE,
            nonfinite_policy="zero",
        )
        np.testing.assert_array_equal(zeroed, [[0, 0, 0]])

    def test_colored_raster_cannot_create_unrepresentable_high_depth_output(
        self,
    ) -> None:
        rgb = np.zeros((2, 3, 3), dtype=np.uint8)

        with self.assertRaisesRegex(ValueError, "先显式转换为灰度"):
            execute_image_operation(
                ImageOperationRequest.create(
                    ImageOperation.CONVERT_TYPE,
                    rgb,
                    target_type="uint16",
                    nonfinite_policy="reject",
                )
            )

    def test_color_conversion_uses_explicit_rec601_or_average_formula(self) -> None:
        rgb = np.asarray([[[100, 150, 200]]], dtype=np.uint8)

        weighted = execute_image_operation(
            ImageOperationRequest.create(
                ImageOperation.CONVERT_COLOR,
                rgb,
                target_model="grayscale",
                grayscale_method="rec601",
            )
        ).image
        averaged = execute_image_operation(
            ImageOperationRequest.create(
                ImageOperation.CONVERT_COLOR,
                rgb,
                target_model="grayscale",
                grayscale_method="average",
            )
        ).image

        np.testing.assert_array_equal(weighted, [[141]])
        np.testing.assert_array_equal(averaged, [[150]])
        self.assertEqual(weighted.dtype, np.uint8)

    def test_color_conversion_never_implicitly_normalizes_or_drops_alpha(self) -> None:
        gray8 = np.asarray([[0, 128, 255]], dtype=np.uint8)
        rgb = execute_image_operation(
            ImageOperationRequest.create(
                ImageOperation.CONVERT_COLOR,
                gray8,
                target_model="rgb",
            )
        ).image
        np.testing.assert_array_equal(rgb[..., 0], gray8)
        np.testing.assert_array_equal(rgb[..., 1], gray8)
        np.testing.assert_array_equal(rgb[..., 2], gray8)

        gray16 = np.asarray([[0, 32768, 65535]], dtype=np.uint16)
        with self.assertRaisesRegex(ValueError, "显式转换位深"):
            execute_image_operation(
                ImageOperationRequest.create(
                    ImageOperation.CONVERT_COLOR,
                    gray16,
                    target_model="rgb",
                )
            )

        rgba = np.asarray([[[100, 150, 200, 77]]], dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "显式启用"):
            execute_image_operation(
                ImageOperationRequest.create(
                    ImageOperation.CONVERT_COLOR,
                    rgba,
                    target_model="grayscale",
                )
            )
        dropped = execute_image_operation(
            ImageOperationRequest.create(
                ImageOperation.CONVERT_COLOR,
                rgba,
                target_model="grayscale",
                drop_alpha=True,
            )
        ).image
        np.testing.assert_array_equal(dropped, [[141]])

    def test_color_balance_keeps_alpha_and_pixels_outside_roi(self) -> None:
        source = np.asarray(
            [
                [[10, 20, 30, 41], [100, 110, 120, 42]],
                [[20, 30, 40, 43], [200, 210, 220, 44]],
            ],
            dtype=np.uint8,
        )
        roi = np.asarray([[True, False], [False, False]], dtype=bool)

        result = execute_image_operation(
            ImageOperationRequest.create(
                ImageOperation.COLOR_BALANCE,
                source,
                roi_mask=roi,
                red_gain=2.0,
                green_gain=1.0,
                blue_gain=0.5,
                red_offset=1.0,
                green_offset=2.0,
                blue_offset=3.0,
            )
        ).image

        np.testing.assert_array_equal(result[0, 0], [21, 22, 18, 41])
        np.testing.assert_array_equal(result[~roi], source[~roi])
        np.testing.assert_array_equal(result[..., 3], source[..., 3])

    def test_roi_filter_leaves_every_outside_pixel_unchanged(self) -> None:
        source = np.zeros((9, 9), dtype=np.uint16)
        source[4, 4] = 50000
        roi = np.zeros((9, 9), dtype=bool)
        roi[2:7, 2:7] = True
        request = ImageOperationRequest.create(
            ImageOperation.GAUSSIAN_BLUR,
            source,
            roi_mask=roi,
            sigma=1.2,
        )

        result = execute_image_operation(request)

        np.testing.assert_array_equal(result.image[~roi], source[~roi])
        self.assertEqual(result.image.dtype, np.uint16)
        self.assertLess(int(result.image[4, 4]), 50000)
        self.assertGreater(int(result.image[4, 3]), 0)

    def test_brightness_adjustment_preserves_dtype_and_clamps_integer_range(self) -> None:
        source = np.asarray([[0, 100, 250]], dtype=np.uint8)
        request = ImageOperationRequest.create(
            ImageOperation.BRIGHTNESS_CONTRAST,
            source,
            brightness=20.0,
            contrast=1.0,
        )

        result = execute_image_operation(request)

        np.testing.assert_array_equal(result.image, [[20, 120, 255]])
        self.assertEqual(result.image.dtype, np.uint8)

    def test_photometric_adjustment_preserves_alpha_channel(self) -> None:
        source = np.asarray([[[10, 20, 30, 77], [100, 110, 120, 199]]], dtype=np.uint8)

        result = execute_image_operation(
            ImageOperationRequest.create(
                ImageOperation.BRIGHTNESS_CONTRAST,
                source,
                brightness=20.0,
            )
        )

        np.testing.assert_array_equal(result.image[..., 3], source[..., 3])
        np.testing.assert_array_equal(result.image[..., :3], source[..., :3] + 20)

    def test_threshold_roi_preserves_unselected_source_values(self) -> None:
        source = np.asarray(
            [
                [10, 10, 10, 10],
                [10, 80, 160, 10],
                [10, 200, 250, 10],
                [10, 10, 10, 10],
            ],
            dtype=np.uint8,
        )
        roi = np.zeros_like(source, dtype=bool)
        roi[1:3, 1:3] = True

        result = execute_image_operation(
            ImageOperationRequest.create(
                ImageOperation.THRESHOLD,
                source,
                roi_mask=roi,
                lower=100.0,
                upper=220.0,
            )
        )

        np.testing.assert_array_equal(result.image[~roi], source[~roi])
        np.testing.assert_array_equal(result.image[1:3, 1:3], [[0, 255], [255, 0]])

    def test_geometry_transforms_have_deterministic_coordinates(self) -> None:
        source = np.arange(12, dtype=np.uint8).reshape(3, 4)
        cropped = execute_image_operation(
            ImageOperationRequest.create(
                ImageOperation.CROP,
                source,
                x=1,
                y=1,
                width=2,
                height=2,
            )
        ).image
        rotated = execute_image_operation(
            ImageOperationRequest.create(
                ImageOperation.ROTATE_90_CLOCKWISE,
                source,
            )
        ).image
        resized = execute_image_operation(
            ImageOperationRequest.create(
                ImageOperation.RESIZE,
                source,
                width=2,
                height=2,
                interpolation=InterpolationMode.NEAREST,
            )
        ).image

        np.testing.assert_array_equal(cropped, [[5, 6], [9, 10]])
        np.testing.assert_array_equal(rotated, np.rot90(source, k=3))
        self.assertEqual(resized.shape, (2, 2))

    def test_resize_auto_resolves_without_changing_legacy_explicit_modes(self) -> None:
        source = np.arange(64, dtype=np.uint8).reshape(8, 8)

        downscaled = execute_image_operation(
            ImageOperationRequest.create(
                ImageOperation.RESIZE,
                source,
                width=4,
                height=4,
                interpolation="auto",
            )
        )
        enlarged = execute_image_operation(
            ImageOperationRequest.create(
                ImageOperation.RESIZE,
                source,
                width=16,
                height=16,
                interpolation="auto",
            )
        )

        self.assertEqual(downscaled.metadata_map["interpolation"], "area")
        self.assertEqual(enlarged.metadata_map["interpolation"], "linear")
        np.testing.assert_array_equal(
            downscaled.image,
            cv2.resize(source, (4, 4), interpolation=cv2.INTER_AREA),
        )

    def test_translation_moves_rgba_and_uses_explicit_border_value(self) -> None:
        source = np.zeros((2, 3, 4), dtype=np.uint8)
        source[0, 0] = [10, 20, 30, 40]

        result = execute_image_operation(
            ImageOperationRequest.create(
                ImageOperation.TRANSLATE,
                source,
                offset_x=1.0,
                offset_y=0.0,
                interpolation="nearest",
                border_mode="constant",
                border_value=(1, 2, 3, 4),
            )
        ).image

        np.testing.assert_array_equal(result[0, 1], [10, 20, 30, 40])
        np.testing.assert_array_equal(result[:, 0], [[1, 2, 3, 4], [1, 2, 3, 4]])
        self.assertEqual(result.dtype, np.uint8)

    def test_canvas_resize_anchor_crops_or_pads_without_changing_channels(self) -> None:
        source = np.asarray(
            [
                [[1, 2, 3, 10], [4, 5, 6, 20]],
                [[7, 8, 9, 30], [10, 11, 12, 40]],
            ],
            dtype=np.uint8,
        )
        padded = execute_image_operation(
            ImageOperationRequest.create(
                ImageOperation.RESIZE_CANVAS,
                source,
                width=4,
                height=3,
                anchor="bottom_right",
                fill_value=(100, 101, 102, 0),
            )
        ).image
        cropped = execute_image_operation(
            ImageOperationRequest.create(
                ImageOperation.RESIZE_CANVAS,
                padded,
                width=2,
                height=2,
                anchor="bottom_right",
            )
        ).image

        self.assertEqual(padded.shape, (3, 4, 4))
        np.testing.assert_array_equal(padded[0, 0], [100, 101, 102, 0])
        np.testing.assert_array_equal(padded[1:, 2:], source)
        np.testing.assert_array_equal(cropped, source)

    def test_pixel_bin_requires_explicit_remainder_crop_and_reports_it(self) -> None:
        source = np.arange(15, dtype=np.uint8).reshape(3, 5)
        with self.assertRaisesRegex(ValueError, "显式选择"):
            execute_image_operation(
                ImageOperationRequest.create(
                    ImageOperation.PIXEL_BIN,
                    source,
                    factor=2,
                )
            )

        result = execute_image_operation(
            ImageOperationRequest.create(
                ImageOperation.PIXEL_BIN,
                source,
                factor=2,
                method="mean",
                remainder_policy="crop",
            )
        )

        np.testing.assert_array_equal(result.image, [[3, 5]])
        self.assertEqual(result.image.dtype, np.uint8)
        self.assertEqual(result.metadata_map["cropped_right"], 1)
        self.assertEqual(result.metadata_map["cropped_bottom"], 1)
        with self.assertRaisesRegex(ValueError, "正整数"):
            execute_image_operation(
                ImageOperationRequest.create(
                    ImageOperation.PIXEL_BIN,
                    source,
                    factor=2.5,
                    remainder_policy="crop",
                )
            )

    def test_pixel_bin_preserves_rgba_layout_and_sum_avoids_saturation(self) -> None:
        rgba = np.asarray(
            [
                [[10, 20, 30, 10], [20, 30, 40, 20]],
                [[30, 40, 50, 30], [40, 50, 60, 40]],
            ],
            dtype=np.uint8,
        )
        averaged = execute_image_operation(
            ImageOperationRequest.create(
                ImageOperation.PIXEL_BIN,
                rgba,
                factor=2,
                method="mean",
            )
        ).image
        np.testing.assert_array_equal(averaged, [[[25, 35, 45, 25]]])

        scalar = np.full((2, 2), 65535, dtype=np.uint16)
        summed = execute_image_operation(
            ImageOperationRequest.create(
                ImageOperation.PIXEL_BIN,
                scalar,
                factor=2,
                method="sum",
            )
        ).image
        self.assertEqual(summed.dtype, np.float32)
        self.assertEqual(float(summed[0, 0]), 262140.0)
        with self.assertRaisesRegex(ValueError, "RGB/RGBA"):
            execute_image_operation(
                ImageOperationRequest.create(
                    ImageOperation.PIXEL_BIN,
                    rgba,
                    factor=2,
                    method="sum",
                )
            )

    def test_morphology_and_fill_holes_use_explicit_binary_semantics(self) -> None:
        source = np.zeros((9, 9), dtype=np.uint8)
        source[2:7, 2:7] = 255
        source[4, 4] = 0

        filled = execute_image_operation(
            ImageOperationRequest.create(
                ImageOperation.FILL_HOLES,
                source,
                foreground_is_high=True,
            )
        ).image
        eroded = execute_image_operation(
            ImageOperationRequest.create(
                ImageOperation.ERODE,
                filled,
                radius=1,
                kernel="rectangle",
            )
        ).image

        self.assertEqual(int(filled[4, 4]), 255)
        self.assertEqual(int(np.count_nonzero(filled)), 25)
        self.assertEqual(int(np.count_nonzero(eroded)), 9)

    def test_sobel_and_high_pass_return_float_when_requested(self) -> None:
        source = np.zeros((32, 32), dtype=np.uint8)
        source[:, 16:] = 255

        sobel = execute_image_operation(
            ImageOperationRequest.create(
                ImageOperation.SOBEL_EDGES,
                source,
                output_float=True,
            )
        ).image
        high_pass = execute_image_operation(
            ImageOperationRequest.create(
                ImageOperation.FFT_FILTER,
                source,
                mode="highpass",
                low_cutoff=0.08,
                output_float=True,
            )
        ).image

        self.assertEqual(sobel.dtype, np.float32)
        self.assertGreater(float(np.max(sobel)), 0.0)
        self.assertEqual(high_pass.dtype, np.float32)
        self.assertLess(abs(float(np.mean(high_pass))), 1e-3)

    def test_invalid_dtype_and_geometry_roi_are_rejected(self) -> None:
        with self.assertRaises(TypeError):
            ImageOperationRequest.create(
                ImageOperation.MEAN_FILTER,
                np.zeros((4, 4), dtype=np.int32),
            )
        with self.assertRaises(ValueError):
            execute_image_operation(
                ImageOperationRequest.create(
                    ImageOperation.FLIP_HORIZONTAL,
                    np.zeros((4, 4), dtype=np.uint8),
                    roi_mask=np.ones((4, 4), dtype=bool),
                )
            )

    def test_bilateral_filter_preserves_uint16_alpha_and_roi_outside(self) -> None:
        source = np.zeros((9, 9, 4), dtype=np.uint16)
        source[4, 4, :3] = 60000
        source[..., 3] = 43210
        roi = np.zeros((9, 9), dtype=bool)
        roi[2:7, 2:7] = True

        result = execute_image_operation(
            ImageOperationRequest.create(
                ImageOperation.BILATERAL_FILTER,
                source,
                roi_mask=roi,
                diameter=5,
                sigma_color=20000.0,
                sigma_space=2.0,
            )
        ).image

        self.assertEqual(result.dtype, np.uint16)
        np.testing.assert_array_equal(result[..., 3], source[..., 3])
        np.testing.assert_array_equal(result[~roi], source[~roi])
        self.assertLess(int(result[4, 4, 0]), 60000)

    def test_laplacian_and_canny_require_explicit_color_channel(self) -> None:
        rgb = np.zeros((12, 12, 3), dtype=np.uint8)
        rgb[:, 6:, 0] = 255

        with self.assertRaisesRegex(ValueError, "显式选择"):
            execute_image_operation(
                ImageOperationRequest.create(ImageOperation.CANNY_EDGES, rgb)
            )
        canny = execute_image_operation(
            ImageOperationRequest.create(
                ImageOperation.CANNY_EDGES,
                rgb,
                channel="red",
                threshold_low=20.0,
                threshold_high=80.0,
            )
        ).image
        laplacian = execute_image_operation(
            ImageOperationRequest.create(
                ImageOperation.LAPLACIAN_EDGES,
                rgb,
                channel="red",
                output_float=True,
            )
        ).image

        self.assertEqual(canny.dtype, np.uint8)
        self.assertEqual(canny.ndim, 2)
        self.assertGreater(int(np.count_nonzero(canny)), 0)
        self.assertEqual(laplacian.dtype, np.float32)
        self.assertGreater(float(np.max(np.abs(laplacian))), 0.0)

    def test_normalize_equalize_and_clahe_preserve_dtype_and_alpha(self) -> None:
        source = np.asarray(
            [
                [[1000, 2000, 3000, 77], [1000, 2000, 3000, 88]],
                [[50000, 40000, 30000, 99], [60000, 50000, 40000, 111]],
            ],
            dtype=np.uint16,
        )

        normalized = execute_image_operation(
            ImageOperationRequest.create(
                ImageOperation.NORMALIZE,
                source,
                output_min=100.0,
                output_max=1000.0,
            )
        ).image
        equalized = execute_image_operation(
            ImageOperationRequest.create(
                ImageOperation.HISTOGRAM_EQUALIZATION,
                source,
            )
        ).image
        clahe_result = execute_image_operation(
            ImageOperationRequest.create(
                ImageOperation.CLAHE,
                source,
                clip_limit=2.0,
                tile_grid_size=2,
            )
        ).image

        for result in (normalized, equalized, clahe_result):
            self.assertEqual(result.dtype, np.uint16)
            np.testing.assert_array_equal(result[..., 3], source[..., 3])
        self.assertEqual(int(np.min(normalized[..., 0])), 100)
        self.assertEqual(int(np.max(normalized[..., 0])), 1000)

    def test_outlier_removal_and_nonfinite_repair_are_local_and_report_count(self) -> None:
        hot = np.full((7, 7), 10, dtype=np.uint8)
        hot[3, 3] = 250
        cleaned = execute_image_operation(
            ImageOperationRequest.create(
                ImageOperation.REMOVE_OUTLIERS,
                hot,
                radius=1,
                threshold=50.0,
                polarity="bright",
            )
        ).image
        self.assertEqual(int(cleaned[3, 3]), 10)

        floating = np.ones((5, 5), dtype=np.float32)
        floating[2, 2] = np.nan
        floating[1, 1] = np.inf
        repaired = execute_image_operation(
            ImageOperationRequest.create(
                ImageOperation.REPAIR_NONFINITE,
                floating,
                radius=1,
            )
        )
        self.assertTrue(np.all(np.isfinite(repaired.image)))
        self.assertEqual(repaired.metadata_map["repaired_count"], 2)
        self.assertAlmostEqual(float(repaired.image[2, 2]), 1.0)

    def test_auto_threshold_methods_find_bimodal_split_and_keep_roi_outside(self) -> None:
        source = np.full((20, 20), 20, dtype=np.uint16)
        source[:, 10:] = 50000
        roi = np.zeros_like(source, dtype=bool)
        roi[4:16, 4:16] = True

        for method in ("otsu", "isodata", "triangle"):
            with self.subTest(method=method):
                result = execute_image_operation(
                    ImageOperationRequest.create(
                        ImageOperation.AUTO_THRESHOLD,
                        source,
                        roi_mask=roi,
                        method=method,
                    )
                )
                np.testing.assert_array_equal(result.image[~roi], source[~roi])
                self.assertEqual(set(np.unique(result.image[roi])), {0, 65535})
                self.assertTrue(
                    0.0 <= float(result.metadata_map["computed_threshold"]) <= 65535.0
                )

    def test_roi_global_statistics_ignore_pixels_outside_the_roi(self) -> None:
        source = np.zeros((6, 6), dtype=np.uint8)
        source[2:4, 2:4] = np.asarray([[100, 100], [200, 200]], dtype=np.uint8)
        roi = np.zeros(source.shape, dtype=bool)
        roi[2:4, 2:4] = True

        normalized = execute_image_operation(
            ImageOperationRequest.create(
                ImageOperation.NORMALIZE,
                source,
                roi_mask=roi,
                output_min=0,
                output_max=255,
                per_channel=True,
            )
        )
        thresholded = execute_image_operation(
            ImageOperationRequest.create(
                ImageOperation.AUTO_THRESHOLD,
                source,
                roi_mask=roi,
                method="otsu",
            )
        )

        self.assertEqual(set(np.unique(normalized.image[roi])), {0, 255})
        np.testing.assert_array_equal(normalized.image[~roi], source[~roi])
        self.assertGreater(
            float(thresholded.metadata_map["computed_threshold"]),
            50.0,
        )

    def test_roi_connected_components_treat_roi_boundary_as_background(self) -> None:
        source = np.zeros((7, 9), dtype=np.uint8)
        source[2, 1:8] = 255
        roi = np.zeros(source.shape, dtype=bool)
        roi[2, 1:3] = True
        roi[2, 6:8] = True

        result = execute_image_operation(
            ImageOperationRequest.create(
                ImageOperation.REMOVE_SMALL_OBJECTS,
                source,
                roi_mask=roi,
                minimum_area=3,
                connectivity=8,
                foreground_is_high=True,
            )
        )

        self.assertEqual(int(np.count_nonzero(result.image[roi])), 0)
        np.testing.assert_array_equal(result.image[~roi], source[~roi])

    def test_binary_operations_require_explicit_channel_for_rgb(self) -> None:
        rgb = np.zeros((5, 5, 3), dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "显式选择"):
            execute_image_operation(
                ImageOperationRequest.create(
                    ImageOperation.AUTO_THRESHOLD,
                    rgb,
                    method="otsu",
                )
            )
        with self.assertRaisesRegex(ValueError, "显式选择"):
            execute_image_operation(
                ImageOperationRequest.create(
                    ImageOperation.ERODE,
                    rgb,
                    radius=1,
                )
            )

    def test_contours_small_objects_holes_and_distance_have_fixed_semantics(self) -> None:
        source = np.zeros((15, 15), dtype=np.uint8)
        source[2:12, 2:12] = 255
        source[5:7, 5:7] = 0
        source[13, 13] = 255

        cleaned = execute_image_operation(
            ImageOperationRequest.create(
                ImageOperation.REMOVE_SMALL_OBJECTS,
                source,
                minimum_area=5,
            )
        ).image
        filled = execute_image_operation(
            ImageOperationRequest.create(
                ImageOperation.FILL_SMALL_HOLES,
                cleaned,
                maximum_area=4,
            )
        ).image
        contour = execute_image_operation(
            ImageOperationRequest.create(
                ImageOperation.CONTOUR_EXTRACT,
                filled,
            )
        ).image
        distance = execute_image_operation(
            ImageOperationRequest.create(
                ImageOperation.DISTANCE_TRANSFORM,
                filled,
            )
        ).image

        self.assertEqual(int(cleaned[13, 13]), 0)
        self.assertEqual(int(np.count_nonzero(filled[5:7, 5:7])), 4)
        self.assertEqual(int(contour[2, 2]), 255)
        self.assertEqual(int(contour[6, 6]), 0)
        self.assertEqual(distance.dtype, np.float32)
        self.assertGreater(float(distance[6, 6]), float(distance[2, 2]))

    def test_skeletonization_reduces_thick_bar_without_breaking_it(self) -> None:
        source = np.zeros((15, 15), dtype=np.uint8)
        source[3:12, 6:9] = 255

        result = execute_image_operation(
            ImageOperationRequest.create(ImageOperation.SKELETONIZE, source)
        ).image

        self.assertLess(int(np.count_nonzero(result)), int(np.count_nonzero(source)))
        self.assertGreaterEqual(int(np.count_nonzero(result)), 6)
        rows = np.flatnonzero(np.any(result > 0, axis=1))
        self.assertGreaterEqual(int(rows[-1] - rows[0]), 5)

    def test_watershed_keeps_binary_dtype_and_separates_seed_regions(self) -> None:
        y, x = np.ogrid[:64, :64]
        source = np.zeros((64, 64), dtype=np.uint8)
        source[((x - 25) ** 2 + (y - 32) ** 2) <= 14**2] = 255
        source[((x - 39) ** 2 + (y - 32) ** 2) <= 14**2] = 255

        result = execute_image_operation(
            ImageOperationRequest.create(
                ImageOperation.WATERSHED,
                source,
                seed_threshold=0.55,
            )
        ).image

        self.assertEqual(result.dtype, np.uint8)
        self.assertTrue(set(np.unique(result)).issubset({0, 255}))
        self.assertLess(int(np.count_nonzero(result)), int(np.count_nonzero(source)))
        components, _labels = cv2.connectedComponents(
            (result > 0).astype(np.uint8),
            connectivity=8,
        )
        self.assertGreaterEqual(components - 1, 2)

    def test_background_subtraction_and_custom_convolution_preserve_roi_outside(self) -> None:
        y, x = np.mgrid[:21, :21]
        source = (x * 4 + y * 2).astype(np.uint16)
        source[10, 10] += 1000
        roi = np.zeros_like(source, dtype=bool)
        roi[5:16, 5:16] = True

        background = execute_image_operation(
            ImageOperationRequest.create(
                ImageOperation.BACKGROUND_SUBTRACT,
                source,
                roi_mask=roi,
                radius=3,
            )
        ).image
        convolved = execute_image_operation(
            ImageOperationRequest.create(
                ImageOperation.CUSTOM_CONVOLUTION,
                source,
                roi_mask=roi,
                kernel=(0.0, 1.0, 0.0, 1.0, 4.0, 1.0, 0.0, 1.0, 0.0),
                kernel_width=3,
                kernel_height=3,
                normalize_kernel=True,
            )
        ).image

        np.testing.assert_array_equal(background[~roi], source[~roi])
        np.testing.assert_array_equal(convolved[~roi], source[~roi])
        self.assertGreater(int(background[10, 10]), int(background[9, 9]))
        self.assertLess(int(convolved[10, 10]), int(source[10, 10]))
        float_result = execute_image_operation(
            ImageOperationRequest.create(
                ImageOperation.CUSTOM_CONVOLUTION,
                source.astype(np.float32),
                kernel=(1.0,),
                kernel_width=1,
                kernel_height=1,
            )
        ).image
        self.assertEqual(float_result.dtype, np.float32)
        np.testing.assert_array_equal(float_result, source.astype(np.float32))

    def test_math_operations_preserve_dtype_and_alpha_and_validate_domains(self) -> None:
        source = np.asarray([[[4, 9, 16, 77], [25, 36, 49, 88]]], dtype=np.uint16)
        square_root = execute_image_operation(
            ImageOperationRequest.create(ImageOperation.SQRT, source)
        ).image
        multiplied = execute_image_operation(
            ImageOperationRequest.create(
                ImageOperation.MULTIPLY,
                source,
                value=2.0,
            )
        ).image

        np.testing.assert_array_equal(square_root[..., :3], [[[2, 3, 4], [5, 6, 7]]])
        np.testing.assert_array_equal(square_root[..., 3], source[..., 3])
        np.testing.assert_array_equal(multiplied[..., 3], source[..., 3])
        self.assertEqual(multiplied.dtype, np.uint16)
        with self.assertRaisesRegex(ValueError, "除数不能为零"):
            execute_image_operation(
                ImageOperationRequest.create(
                    ImageOperation.DIVIDE,
                    source,
                    value=0.0,
                )
            )

    def test_all_scalar_math_operations_have_explicit_clamped_semantics(self) -> None:
        source = np.asarray([[1, 2, 4]], dtype=np.uint8)
        cases = {
            ImageOperation.INVERT: {"minimum": 0.0, "maximum": 255.0},
            ImageOperation.ADD: {"value": 2.0},
            ImageOperation.SUBTRACT: {"value": 2.0},
            ImageOperation.MULTIPLY: {"value": 3.0},
            ImageOperation.DIVIDE: {"value": 2.0},
            ImageOperation.GAMMA: {"gamma": 2.0},
            ImageOperation.LOG: {},
            ImageOperation.EXP: {},
            ImageOperation.SQRT: {},
            ImageOperation.ABS: {},
            ImageOperation.CLAMP: {"minimum": 2.0, "maximum": 3.0},
        }
        for operation, parameters in cases.items():
            with self.subTest(operation=operation.value):
                result = execute_image_operation(
                    ImageOperationRequest.create(
                        operation,
                        source,
                        **parameters,
                    )
                ).image
                self.assertEqual(result.dtype, source.dtype)
                self.assertEqual(result.shape, source.shape)

    def test_image_calculator_uses_frozen_second_image_and_keeps_first_alpha(self) -> None:
        first = np.asarray([[[10, 20, 30, 44], [250, 240, 230, 55]]], dtype=np.uint8)
        second = np.asarray([[[5, 6, 7, 200], [10, 20, 30, 201]]], dtype=np.uint8)
        request = ImageOperationRequest.create(
            ImageOperation.IMAGE_CALCULATOR,
            first,
            secondary_image=second,
            calculator_operation="add",
        )
        second[:] = 0
        result = execute_image_operation(request).image

        np.testing.assert_array_equal(result[..., :3], [[[15, 26, 37], [255, 255, 255]]])
        np.testing.assert_array_equal(result[..., 3], first[..., 3])
        with self.assertRaisesRegex(TypeError, "整数图像"):
            execute_image_operation(
                ImageOperationRequest.create(
                    ImageOperation.IMAGE_CALCULATOR,
                    np.ones((2, 2), dtype=np.float32),
                    secondary_image=np.ones((2, 2), dtype=np.float32),
                    calculator_operation="xor",
                )
            )

    def test_image_calculator_integer_boolean_operations_are_exact(self) -> None:
        first = np.asarray([[0b1100, 0b1010]], dtype=np.uint8)
        second = np.asarray([[0b1010, 0b0110]], dtype=np.uint8)
        expected = {
            "and": np.asarray([[0b1000, 0b0010]], dtype=np.uint8),
            "or": np.asarray([[0b1110, 0b1110]], dtype=np.uint8),
            "xor": np.asarray([[0b0110, 0b1100]], dtype=np.uint8),
        }
        for operation, values in expected.items():
            with self.subTest(operation=operation):
                result = execute_image_operation(
                    ImageOperationRequest.create(
                        ImageOperation.IMAGE_CALCULATOR,
                        first,
                        secondary_image=second,
                        calculator_operation=operation,
                    )
                ).image
                np.testing.assert_array_equal(result, values)

    def test_stripe_suppression_reduces_directional_row_variation(self) -> None:
        rows = (100.0 + 40.0 * np.sin(np.arange(128) * 2.0 * np.pi / 8.0)).astype(
            np.float32
        )
        source = np.repeat(rows[:, np.newaxis], 128, axis=1)

        result = execute_image_operation(
            ImageOperationRequest.create(
                ImageOperation.STRIPE_SUPPRESSION,
                source,
                direction="horizontal",
                notch_width=0.01,
                protect_radius=0.02,
                strength=1.0,
            )
        ).image

        self.assertEqual(result.dtype, np.float32)
        before = float(np.std(np.mean(source, axis=1)))
        after = float(np.std(np.mean(result, axis=1)))
        self.assertLess(after, before * 0.2)

    def test_user_visible_validation_messages_are_chinese(self) -> None:
        with self.assertRaisesRegex(ValueError, "裁剪"):
            execute_image_operation(
                ImageOperationRequest.create(
                    ImageOperation.CROP,
                    np.zeros((3, 3), dtype=np.uint8),
                    x=0,
                    y=0,
                    width=0,
                    height=1,
                )
            )
        with self.assertRaisesRegex(TypeError, "位深"):
            ImageOperationRequest.create(
                ImageOperation.MEAN_FILTER,
                np.zeros((3, 3), dtype=np.int32),
            )
        with self.assertRaisesRegex(ValueError, "插值模式"):
            execute_image_operation(
                ImageOperationRequest.create(
                    ImageOperation.RESIZE,
                    np.zeros((3, 3), dtype=np.uint8),
                    width=2,
                    height=2,
                    interpolation="unknown",
                )
            )


if __name__ == "__main__":
    unittest.main()
