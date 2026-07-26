from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from fdm.cancellation import CancellationError, CancellationTokenSource
from fdm.image_processing_models import ImageOperationSpec
from fdm.services import image_processing as processing
from fdm.services.image_processing import (
    ImageOperation,
    ImageOperationRequest,
    execute_image_operation,
    execute_image_operation_tiled,
    resolve_image_operation_capability,
)
from fdm.ui import image_processing_workbench as workbench


class ImageOperationCapabilityTests(unittest.TestCase):
    def test_capability_declares_exact_halo_and_global_dependencies(self) -> None:
        mean = resolve_image_operation_capability(
            ImageOperation.MEAN_FILTER,
            {"radius": 3},
        )
        opening = resolve_image_operation_capability(
            ImageOperation.MORPHOLOGY_OPEN,
            {"radius": 2, "iterations": 3},
        )
        threshold = resolve_image_operation_capability(
            ImageOperation.AUTO_THRESHOLD,
            {"method": "otsu"},
        )
        geometry = resolve_image_operation_capability(ImageOperation.CROP)

        self.assertTrue(mean.tileable)
        self.assertEqual((mean.halo_x, mean.halo_y), (3, 3))
        self.assertTrue(opening.tileable)
        self.assertEqual((opening.halo_x, opening.halo_y), (12, 12))
        self.assertFalse(threshold.tileable)
        self.assertTrue(threshold.requires_full_image_prescan)
        self.assertFalse(geometry.tileable)
        self.assertFalse(geometry.preserves_spatial_extent)

    def test_data_range_and_background_offset_remain_global(self) -> None:
        data_range = resolve_image_operation_capability(
            ImageOperation.CONVERT_TYPE,
            {"target_type": "uint16", "scale_mode": "data_range"},
        )
        background = resolve_image_operation_capability(
            ImageOperation.BACKGROUND_SUBTRACT,
            {"radius": 5, "preserve_offset": True},
        )

        self.assertFalse(data_range.tileable)
        self.assertTrue(data_range.requires_full_image_prescan)
        self.assertFalse(background.tileable)
        self.assertIn("中位数", background.reason)

    def test_wrap_border_and_bilateral_fall_back_to_exact_whole_image(self) -> None:
        wrapped = resolve_image_operation_capability(
            ImageOperation.GAUSSIAN_BLUR,
            {"sigma_x": 1.5, "border_mode": "wrap"},
        )
        bilateral = resolve_image_operation_capability(
            ImageOperation.BILATERAL_FILTER,
            {
                "diameter": 9,
                "sigma_color": 25.0,
                "sigma_space": 4.0,
                "border_mode": "reflect",
            },
        )

        self.assertFalse(wrapped.tileable)
        self.assertIn("对侧像素", wrapped.reason)
        self.assertFalse(bilateral.tileable)
        self.assertIn("逐位一致", bilateral.reason)


class TiledImageProcessingTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(20260727)
        self.source = rng.integers(0, 256, size=(79, 91), dtype=np.uint8)
        self.roi = np.zeros(self.source.shape, dtype=bool)
        self.roi[9:73, 7:86] = True
        self.roi[26:48, 29:57] = False
        self.roi[31:67, 60:88] = True

    def _assert_tiled_parity(
        self,
        operation: ImageOperation,
        parameters: dict[str, object],
        *,
        roi: bool = True,
        source: np.ndarray | None = None,
        secondary: np.ndarray | None = None,
    ) -> None:
        image = self.source if source is None else source
        roi_mask = self.roi if roi else None
        full = execute_image_operation(
            ImageOperationRequest.create(
                operation,
                image,
                secondary_image=secondary,
                roi_mask=roi_mask,
                **parameters,
            )
        )
        tiled = execute_image_operation_tiled(
            operation,
            image,
            secondary_image=secondary,
            roi_mask=roi_mask,
            parameters=parameters,
            tile_size=32,
        )

        np.testing.assert_array_equal(tiled.image, full.image)
        self.assertEqual(tiled.image.dtype, full.image.dtype)
        self.assertEqual(tiled.image.shape, full.image.shape)

    def test_local_filters_match_whole_image_at_every_tile_seam(self) -> None:
        cases = (
            (ImageOperation.GAUSSIAN_BLUR, {"sigma_x": 1.3, "sigma_y": 0.8}),
            (ImageOperation.MEDIAN_FILTER, {"radius": 2}),
            (
                ImageOperation.MEAN_FILTER,
                {"radius": 3, "border_mode": "reflect"},
            ),
            (
                ImageOperation.BILATERAL_FILTER,
                {
                    "diameter": 5,
                    "sigma_color": 18.0,
                    "sigma_space": 2.0,
                    "border_mode": "reflect",
                },
            ),
            (
                ImageOperation.UNSHARP_MASK,
                {"sigma": 1.1, "amount": 0.8, "threshold": 3.0},
            ),
            (
                ImageOperation.REMOVE_OUTLIERS,
                {"radius": 2, "threshold": 25.0, "polarity": "both"},
            ),
            (
                ImageOperation.BACKGROUND_SUBTRACT,
                {
                    "radius": 3,
                    "light_background": False,
                    "preserve_offset": False,
                    "border_mode": "reflect",
                },
            ),
        )
        for operation, parameters in cases:
            with self.subTest(operation=operation.value):
                self._assert_tiled_parity(operation, parameters)

    def test_derivative_morphology_and_convolution_match_whole_image(self) -> None:
        cases = (
            (
                ImageOperation.SOBEL_EDGES,
                {
                    "kernel_size": 5,
                    "channel": "luminance",
                    "output_float": True,
                },
            ),
            (
                ImageOperation.LAPLACIAN_EDGES,
                {"kernel_size": 5, "output_float": True},
            ),
            (
                ImageOperation.MORPHOLOGY_OPEN,
                {
                    "radius": 2,
                    "iterations": 2,
                    "kernel": "ellipse",
                    "border_mode": "reflect",
                },
            ),
            (
                ImageOperation.CUSTOM_CONVOLUTION,
                {
                    "kernel": (
                        1.0,
                        2.0,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        -1.0,
                        -2.0,
                        -1.0,
                    ),
                    "kernel_width": 3,
                    "kernel_height": 3,
                    "normalize_kernel": False,
                    "offset": 0.0,
                    "border_mode": "reflect",
                },
            ),
        )
        for operation, parameters in cases:
            with self.subTest(operation=operation.value):
                self._assert_tiled_parity(operation, parameters)

    def test_pointwise_scalar_and_second_image_outputs_match(self) -> None:
        secondary = np.flipud(self.source).copy()
        cases = (
            (
                ImageOperation.BRIGHTNESS_CONTRAST,
                {"brightness": 7.0, "contrast": 1.2, "gamma": 0.9},
                None,
            ),
            (
                ImageOperation.THRESHOLD,
                {"lower": 50.0, "upper": 180.0, "invert": False},
                None,
            ),
            (
                ImageOperation.IMAGE_CALCULATOR,
                {"calculator_operation": "difference"},
                secondary,
            ),
        )
        for operation, parameters, second in cases:
            with self.subTest(operation=operation.value):
                self._assert_tiled_parity(
                    operation,
                    parameters,
                    secondary=second,
                )

    def test_native_depth_color_and_alpha_results_match(self) -> None:
        rng = np.random.default_rng(317)
        gray16 = rng.integers(0, 65536, size=(79, 91), dtype=np.uint16)
        rgb = rng.integers(0, 256, size=(79, 91, 3), dtype=np.uint8)
        rgba = rng.integers(0, 256, size=(79, 91, 4), dtype=np.uint8)
        alpha_before = rgba[..., 3].copy()
        float_image = rng.normal(size=(79, 91)).astype(np.float32)
        float_image[30, 32] = np.nan
        float_image[52, 64] = np.inf

        self._assert_tiled_parity(
            ImageOperation.GAUSSIAN_BLUR,
            {"sigma_x": 1.4, "sigma_y": 2.1},
            source=gray16,
        )
        self._assert_tiled_parity(
            ImageOperation.MEAN_FILTER,
            {"radius": 2},
            source=rgb,
        )
        self._assert_tiled_parity(
            ImageOperation.UNSHARP_MASK,
            {"sigma": 1.2, "amount": 0.5, "threshold": 0.0},
            source=rgba,
        )
        repaired = execute_image_operation_tiled(
            ImageOperation.REPAIR_NONFINITE,
            float_image,
            roi_mask=self.roi,
            parameters={"radius": 2, "fallback_value": 0.0},
            request_id="repair-request",
            generation=11,
            tile_size=32,
        )
        full_repaired = execute_image_operation(
            ImageOperationRequest.create(
                ImageOperation.REPAIR_NONFINITE,
                float_image,
                roi_mask=self.roi,
                radius=2,
                fallback_value=0.0,
            )
        )

        np.testing.assert_array_equal(repaired.image, full_repaired.image)
        self.assertEqual(repaired.metadata_map["repaired_count"], 2)
        self.assertEqual(repaired.request_id, "repair-request")
        self.assertEqual(repaired.generation, 11)
        rgba_tiled = execute_image_operation_tiled(
            ImageOperation.UNSHARP_MASK,
            rgba,
            roi_mask=self.roi,
            parameters={"sigma": 1.2, "amount": 0.5, "threshold": 0.0},
            tile_size=32,
        ).image
        np.testing.assert_array_equal(rgba_tiled[..., 3], alpha_before)

    def test_tiled_float_to_integer_records_whole_image_replacement_count(
        self,
    ) -> None:
        source = np.linspace(0.0, 1.0, 79 * 91, dtype=np.float32).reshape(
            79,
            91,
        )
        source[3, 4] = np.nan
        source[40, 50] = np.inf
        source[70, 80] = -np.inf
        parameters = {
            "target_type": "uint16",
            "scale_mode": "full_type_range",
            "nonfinite_policy": "range_bounds",
        }

        full = execute_image_operation(
            ImageOperationRequest.create(
                ImageOperation.CONVERT_TYPE,
                source,
                **parameters,
            )
        )
        tiled = execute_image_operation_tiled(
            ImageOperation.CONVERT_TYPE,
            source,
            parameters=parameters,
            tile_size=32,
        )

        np.testing.assert_array_equal(tiled.image, full.image)
        self.assertEqual(
            tiled.metadata_map["nonfinite_replacement_count"],
            3,
        )

    def test_roi_reads_original_halo_but_never_changes_outside_pixels(self) -> None:
        source = np.zeros((79, 91), dtype=np.uint8)
        source[31, 31] = 255
        roi = np.zeros(source.shape, dtype=bool)
        roi[31:70, 32:75] = True
        full = execute_image_operation(
            ImageOperationRequest.create(
                ImageOperation.MEAN_FILTER,
                source,
                roi_mask=roi,
                radius=2,
            )
        )
        tiled = execute_image_operation_tiled(
            ImageOperation.MEAN_FILTER,
            source,
            roi_mask=roi,
            parameters={"radius": 2},
            tile_size=32,
        )

        np.testing.assert_array_equal(tiled.image, full.image)
        np.testing.assert_array_equal(tiled.image[~roi], source[~roi])
        self.assertGreater(int(tiled.image[31, 32]), 0)

    def test_global_operation_falls_back_to_one_whole_image_execution(self) -> None:
        original = processing.execute_image_operation
        with mock.patch.object(
            processing,
            "execute_image_operation",
            wraps=original,
        ) as execute:
            execute_image_operation_tiled(
                ImageOperation.AUTO_THRESHOLD,
                self.source,
                parameters={"method": "otsu"},
                tile_size=32,
            )

        execute.assert_called_once()
        request = execute.call_args.args[0]
        self.assertEqual(request.image.shape, self.source.shape)

    def test_wrap_gaussian_uses_whole_image_instead_of_patch_local_wrap(self) -> None:
        parameters = {
            "sigma_x": 1.7,
            "sigma_y": 2.3,
            "border_mode": "wrap",
        }
        expected = execute_image_operation(
            ImageOperationRequest.create(
                ImageOperation.GAUSSIAN_BLUR,
                self.source,
                **parameters,
            )
        )
        original = processing.execute_image_operation
        with mock.patch.object(
            processing,
            "execute_image_operation",
            wraps=original,
        ) as execute:
            actual = execute_image_operation_tiled(
                ImageOperation.GAUSSIAN_BLUR,
                self.source,
                parameters=parameters,
                tile_size=32,
            )

        execute.assert_called_once()
        np.testing.assert_array_equal(actual.image, expected.image)

    def test_bilateral_uses_whole_image_to_avoid_patch_rounding_drift(self) -> None:
        rng = np.random.default_rng(240727)
        source = rng.integers(0, 65536, size=(79, 91), dtype=np.uint16)
        parameters = {
            "diameter": 9,
            "sigma_color": 19.3,
            "sigma_space": 7.5,
            "border_mode": "reflect",
        }
        expected = execute_image_operation(
            ImageOperationRequest.create(
                ImageOperation.BILATERAL_FILTER,
                source,
                **parameters,
            )
        )
        original = processing.execute_image_operation
        with mock.patch.object(
            processing,
            "execute_image_operation",
            wraps=original,
        ) as execute:
            actual = execute_image_operation_tiled(
                ImageOperation.BILATERAL_FILTER,
                source,
                parameters=parameters,
                tile_size=32,
            )

        execute.assert_called_once()
        np.testing.assert_array_equal(actual.image, expected.image)

    def test_unsupported_opencv_wrap_modes_raise_chinese_errors(self) -> None:
        cases = (
            (
                ImageOperation.MEAN_FILTER,
                {"radius": 2, "border_mode": "wrap"},
            ),
            (
                ImageOperation.MORPHOLOGY_OPEN,
                {
                    "radius": 2,
                    "iterations": 1,
                    "kernel": "ellipse",
                    "border_mode": "wrap",
                },
            ),
            (
                ImageOperation.BACKGROUND_SUBTRACT,
                {"radius": 3, "border_mode": "wrap"},
            ),
            (
                ImageOperation.CUSTOM_CONVOLUTION,
                {
                    "kernel": (1.0,) * 9,
                    "kernel_width": 3,
                    "kernel_height": 3,
                    "normalize_kernel": True,
                    "border_mode": "wrap",
                },
            ),
        )
        for operation, parameters in cases:
            with self.subTest(operation=operation.value):
                with self.assertRaisesRegex(ValueError, "不支持循环边界"):
                    execute_image_operation(
                        ImageOperationRequest.create(
                            operation,
                            self.source,
                            **parameters,
                        )
                    )

    def test_tileable_operation_never_hands_the_whole_image_to_kernel(self) -> None:
        original = processing.execute_image_operation
        with mock.patch.object(
            processing,
            "execute_image_operation",
            wraps=original,
        ) as execute:
            execute_image_operation_tiled(
                ImageOperation.MEAN_FILTER,
                self.source,
                parameters={"radius": 2},
                tile_size=32,
            )

        self.assertGreater(execute.call_count, 1)
        patch_shapes = [call.args[0].image.shape[:2] for call in execute.call_args_list]
        self.assertNotIn(self.source.shape, patch_shapes)
        self.assertLessEqual(max(height for height, _width in patch_shapes), 36)
        self.assertLessEqual(max(width for _height, width in patch_shapes), 36)

    def test_cancellation_is_checked_between_tiles_without_partial_result(self) -> None:
        cancellation = CancellationTokenSource()
        checks = 0

        def check() -> None:
            nonlocal checks
            checks += 1
            if checks == 3:
                cancellation.cancel()
            cancellation.token.raise_if_cancelled()

        with self.assertRaises(CancellationError):
            execute_image_operation_tiled(
                ImageOperation.MEAN_FILTER,
                self.source,
                parameters={"radius": 2},
                tile_size=32,
                cancellation_check=check,
            )
        self.assertGreaterEqual(checks, 3)

    def test_workbench_pipeline_uses_exact_tiled_executor(self) -> None:
        source_plane = workbench.array_to_raster_plane(self.source)
        request = workbench.WorkbenchTaskRequest(
            kind=workbench.WorkbenchTaskKind.FINAL,
            request_id="tile-request",
            generation=9,
            source_document_id="document-1",
            source=source_plane,
            operations=(
                ImageOperationSpec(
                    "mean_filter",
                    {"radius": 2, "border_mode": "reflect"},
                ),
                ImageOperationSpec(
                    "brightness_contrast",
                    {"brightness": 3.0, "contrast": 1.0, "gamma": 1.0},
                ),
            ),
            roi_mask=self.roi,
        )
        token = CancellationTokenSource().token
        with mock.patch.object(workbench, "PROCESSING_TILE_EDGE", 32):
            output = workbench.execute_workbench_request(request, token)

        expected = self.source
        for spec in request.operations:
            expected = execute_image_operation(
                ImageOperationRequest.create(
                    spec.operation_id,
                    expected,
                    roi_mask=self.roi,
                    **spec.parameters,
                )
            ).image
        np.testing.assert_array_equal(
            workbench.raster_plane_to_array(output),
            expected,
        )


if __name__ == "__main__":
    unittest.main()
