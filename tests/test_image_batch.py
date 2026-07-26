from __future__ import annotations

import json
import unittest
from unittest import mock

import numpy as np

from fdm.cancellation import CancellationTokenSource
from fdm.image_processing_models import (
    ImageOperationSpec,
    ImageProcessingRecipe,
)
from fdm.services.image_batch import (
    BatchExecutionLimits,
    BatchItemStatus,
    BatchProgressPhase,
    BatchRasterInput,
    BatchRecipeRequest,
    execute_batch_recipe,
    preflight_batch_recipe,
)
from fdm.services.raster_io import (
    numpy_to_raster_plane,
    raster_plane_to_numpy,
)


def _recipe(
    operation_id: str,
    parameters: dict[str, object] | None = None,
) -> ImageProcessingRecipe:
    return ImageProcessingRecipe.from_operations(
        (ImageOperationSpec(operation_id, parameters or {}),)
    )


def _input(
    document_id: str,
    image: np.ndarray,
    *,
    secondary: np.ndarray | None = None,
) -> BatchRasterInput:
    return BatchRasterInput(
        document_id=document_id,
        display_name=f"图片 {document_id}",
        raster=numpy_to_raster_plane(image),
        source_pixel_revision=4,
        source_path=f"/images/{document_id}.png",
        secondary_raster=(
            None if secondary is None else numpy_to_raster_plane(secondary)
        ),
    )


_PLENTY_OF_DISK = 10 << 30


class ImageBatchExecutionTests(unittest.TestCase):
    def test_success_produces_audited_candidates_without_mutating_sources(self) -> None:
        source_array = np.asarray([[0, 5], [10, 20]], dtype=np.uint8)
        first = _input("first", source_array)
        second = _input("second", source_array + 1)
        first_bytes = first.raster.data
        request = BatchRecipeRequest(
            request_id="batch-1",
            generation=8,
            recipe=_recipe("add", {"value": 3.0}),
            inputs=(first, second),
            available_disk_bytes=_PLENTY_OF_DISK,
        )

        result = execute_batch_recipe(request)

        self.assertTrue(result.commit_allowed)
        self.assertEqual(result.success_count, 2)
        self.assertEqual(len(result.commit_candidates), 2)
        self.assertEqual(first.raster.data, first_bytes)
        np.testing.assert_array_equal(
            raster_plane_to_numpy(result.commit_candidates[0].raster),
            source_array + 3,
        )
        derivation = result.commit_candidates[0].derivation
        self.assertEqual(derivation.source_document_id, "first")
        self.assertEqual(derivation.source_pixel_revision, 4)
        self.assertEqual(derivation.recipe, request.recipe)
        self.assertEqual(
            derivation.result_sha256,
            result.commit_candidates[0].raster.sha256(),
        )
        self.assertIn("成功 2 张", result.summary_text)
        self.assertEqual(json.loads(result.to_json())["generation"], 8)

    def test_one_document_failure_does_not_block_following_documents(self) -> None:
        image = np.arange(9, dtype=np.uint8).reshape(3, 3)
        request = BatchRecipeRequest(
            request_id="calculator",
            generation=1,
            recipe=_recipe(
                "image_calculator",
                {"calculator_operation": "add"},
            ),
            inputs=(
                _input("missing", image),
                _input("valid", image, secondary=np.ones_like(image)),
            ),
            available_disk_bytes=_PLENTY_OF_DISK,
        )

        result = execute_batch_recipe(request)

        self.assertEqual(result.items[0].status, BatchItemStatus.FAILED)
        self.assertIn("第二幅图像", result.items[0].message)
        self.assertEqual(result.items[1].status, BatchItemStatus.SUCCESS)
        self.assertEqual(len(result.commit_candidates), 1)
        np.testing.assert_array_equal(
            raster_plane_to_numpy(result.commit_candidates[0].raster),
            image + 1,
        )

    def test_cancellation_after_a_success_disables_every_commit_candidate(self) -> None:
        source = CancellationTokenSource()
        request = BatchRecipeRequest(
            request_id="cancel",
            generation=2,
            recipe=_recipe("invert"),
            inputs=(
                _input("one", np.zeros((2, 2), dtype=np.uint8)),
                _input("two", np.zeros((2, 2), dtype=np.uint8)),
            ),
            available_disk_bytes=_PLENTY_OF_DISK,
        )
        from fdm.services import image_batch

        real_execute = image_batch._execute_item
        invocation_count = 0

        def execute_then_cancel(*args, **kwargs):
            nonlocal invocation_count
            result = real_execute(*args, **kwargs)
            invocation_count += 1
            if invocation_count == 1:
                source.cancel()
            return result

        with mock.patch.object(
            image_batch,
            "_execute_item",
            side_effect=execute_then_cancel,
        ):
            result = execute_batch_recipe(
                request,
                cancellation_token=source.token,
            )

        self.assertTrue(result.cancelled)
        self.assertFalse(result.commit_allowed)
        self.assertEqual(result.commit_candidates, ())
        self.assertEqual(result.items[0].status, BatchItemStatus.SUCCESS)
        self.assertEqual(result.items[1].status, BatchItemStatus.CANCELLED)
        self.assertIn("均未提交", result.summary_text)

    def test_stale_generation_discards_all_candidates(self) -> None:
        request = BatchRecipeRequest(
            request_id="stale",
            generation=4,
            recipe=_recipe("invert"),
            inputs=(_input("one", np.zeros((2, 2), dtype=np.uint8)),),
            available_disk_bytes=_PLENTY_OF_DISK,
        )

        result = execute_batch_recipe(
            request,
            generation_is_current=lambda generation: generation == 5,
        )

        self.assertTrue(result.stale)
        self.assertFalse(result.commit_allowed)
        self.assertEqual(result.commit_candidates, ())
        self.assertEqual(result.items[0].status, BatchItemStatus.STALE)

    def test_disk_preflight_blocks_before_any_pixel_operation(self) -> None:
        request = BatchRecipeRequest(
            request_id="disk",
            generation=0,
            recipe=_recipe("invert"),
            inputs=(_input("one", np.zeros((20, 20), dtype=np.uint8)),),
            available_disk_bytes=1024,
        )
        from fdm.services import image_batch

        with mock.patch.object(
            image_batch,
            "execute_image_operation_tiled",
            side_effect=AssertionError("不得执行"),
        ):
            result = execute_batch_recipe(request)

        self.assertFalse(result.preflight.disk_allowed)
        self.assertEqual(result.items[0].status, BatchItemStatus.RESOURCE_BLOCKED)
        self.assertIn("磁盘空间", result.items[0].message)

    def test_memory_preflight_blocks_only_oversized_document(self) -> None:
        request = BatchRecipeRequest(
            request_id="memory",
            generation=0,
            recipe=_recipe("mean_filter", {"radius": 1}),
            inputs=(
                _input("small", np.zeros((4, 4), dtype=np.uint8)),
                _input("large", np.zeros((40, 40), dtype=np.uint8)),
            ),
            available_disk_bytes=_PLENTY_OF_DISK,
        )
        limits = BatchExecutionLimits(
            max_working_bytes=500,
            min_free_disk_bytes=1,
            max_documents=10,
        )

        estimate = preflight_batch_recipe(request, limits=limits)
        result = execute_batch_recipe(request, limits=limits)

        self.assertTrue(estimate.items[0].allowed)
        self.assertFalse(estimate.items[1].allowed)
        self.assertEqual(result.items[0].status, BatchItemStatus.SUCCESS)
        self.assertEqual(
            result.items[1].status,
            BatchItemStatus.RESOURCE_BLOCKED,
        )
        self.assertEqual(len(result.commit_candidates), 1)

    def test_roi_input_is_detached_from_caller_and_outside_pixels_stay_exact(self) -> None:
        image = np.arange(25, dtype=np.uint8).reshape(5, 5)
        roi = np.zeros((5, 5), dtype=bool)
        roi[2, 2] = True
        item = BatchRasterInput(
            document_id="roi",
            display_name="ROI",
            raster=numpy_to_raster_plane(image),
            roi_mask=roi,
        )
        roi[:] = True
        request = BatchRecipeRequest(
            request_id="roi",
            generation=0,
            recipe=_recipe("add", {"value": 10.0}),
            inputs=(item,),
            available_disk_bytes=_PLENTY_OF_DISK,
        )

        result = execute_batch_recipe(request)
        output = raster_plane_to_numpy(result.commit_candidates[0].raster)

        expected = image.copy()
        expected[2, 2] += 10
        np.testing.assert_array_equal(output, expected)
        self.assertEqual(np.count_nonzero(item.roi_mask), 1)
        with self.assertRaises(ValueError):
            item.roi_mask[0, 0] = True

    def test_invalid_first_operation_cannot_partially_change_source_or_second_input(
        self,
    ) -> None:
        first_array = np.arange(4, dtype=np.uint8).reshape(2, 2)
        second_array = first_array + 10
        first = _input("first", first_array)
        second = _input("second", second_array)
        before = (first.raster.sha256(), second.raster.sha256())
        request = BatchRecipeRequest(
            request_id="failure",
            generation=0,
            recipe=_recipe("gamma", {"gamma": -1.0}),
            inputs=(first, second),
            available_disk_bytes=_PLENTY_OF_DISK,
        )

        result = execute_batch_recipe(request)

        self.assertEqual(result.failure_count, 2)
        self.assertEqual(
            (first.raster.sha256(), second.raster.sha256()),
            before,
        )
        self.assertEqual(result.commit_candidates, ())

    def test_batch_uses_exact_tiled_executor_and_emits_structured_progress(
        self,
    ) -> None:
        from fdm.services import image_batch

        rng = np.random.default_rng(20260727)
        image = rng.integers(0, 256, size=(79, 91), dtype=np.uint8)
        request = BatchRecipeRequest(
            request_id="tiled-progress",
            generation=12,
            recipe=_recipe("mean_filter", {"radius": 2}),
            inputs=(_input("one", image),),
            available_disk_bytes=_PLENTY_OF_DISK,
        )
        progress = []
        real_execute = image_batch.execute_image_operation_tiled

        with (
            mock.patch.object(image_batch, "BATCH_PROCESSING_TILE_EDGE", 32),
            mock.patch.object(
                image_batch,
                "execute_image_operation_tiled",
                wraps=real_execute,
            ) as execute,
        ):
            result = execute_batch_recipe(
                request,
                progress_callback=progress.append,
            )

        self.assertEqual(result.success_count, 1)
        execute.assert_called_once()
        self.assertEqual(execute.call_args.kwargs["tile_size"], 32)
        self.assertEqual(
            execute.call_args.kwargs["request_id"],
            request.request_id,
        )
        self.assertEqual(
            execute.call_args.kwargs["generation"],
            request.generation,
        )
        self.assertEqual(progress[0].phase, BatchProgressPhase.PREFLIGHT)
        self.assertEqual(progress[-1].phase, BatchProgressPhase.PACKAGING)
        processing = [
            update
            for update in progress
            if update.phase is BatchProgressPhase.PROCESSING
        ]
        self.assertEqual(
            [update.completed_operations for update in processing],
            [0, 1],
        )
        self.assertTrue(
            all(
                update.request_id == request.request_id
                and update.generation == request.generation
                for update in progress
            )
        )

    def test_cancellation_between_tiles_never_returns_partial_candidate(
        self,
    ) -> None:
        from fdm.services import image_batch
        from fdm.services import image_processing

        rng = np.random.default_rng(41)
        image = rng.integers(0, 256, size=(79, 91), dtype=np.uint8)
        request = BatchRecipeRequest(
            request_id="tile-cancel",
            generation=2,
            recipe=_recipe("mean_filter", {"radius": 2}),
            inputs=(_input("one", image),),
            available_disk_bytes=_PLENTY_OF_DISK,
        )
        cancellation = CancellationTokenSource()
        real_execute = image_processing.execute_image_operation
        tile_calls = 0

        def execute_then_cancel(operation_request):
            nonlocal tile_calls
            output = real_execute(operation_request)
            tile_calls += 1
            if tile_calls == 1:
                cancellation.cancel()
            return output

        with (
            mock.patch.object(image_batch, "BATCH_PROCESSING_TILE_EDGE", 32),
            mock.patch.object(
                image_processing,
                "execute_image_operation",
                side_effect=execute_then_cancel,
            ),
        ):
            result = execute_batch_recipe(
                request,
                cancellation_token=cancellation.token,
            )

        self.assertTrue(result.cancelled)
        self.assertEqual(result.commit_candidates, ())
        self.assertEqual(result.items[0].status, BatchItemStatus.CANCELLED)
        self.assertEqual(tile_calls, 1)

    def test_capability_aware_estimate_bounds_tiles_but_keeps_global_ops_full(
        self,
    ) -> None:
        from fdm.services import image_batch

        image = np.zeros((1024, 1024), dtype=np.uint8)
        item = _input("large", image)
        limits = BatchExecutionLimits(
            max_working_bytes=4 << 20,
            min_free_disk_bytes=1,
            max_documents=10,
        )
        tiled_request = BatchRecipeRequest(
            request_id="estimate-tiled",
            generation=0,
            recipe=_recipe("mean_filter", {"radius": 1}),
            inputs=(item,),
            available_disk_bytes=_PLENTY_OF_DISK,
        )
        global_request = BatchRecipeRequest(
            request_id="estimate-global",
            generation=0,
            recipe=_recipe("auto_threshold", {"method": "otsu"}),
            inputs=(item,),
            available_disk_bytes=_PLENTY_OF_DISK,
        )

        with mock.patch.object(
            image_batch,
            "BATCH_PROCESSING_TILE_EDGE",
            256,
        ):
            tiled = preflight_batch_recipe(tiled_request, limits=limits)
            global_estimate = preflight_batch_recipe(
                global_request,
                limits=limits,
            )

        self.assertTrue(tiled.items[0].allowed)
        self.assertFalse(global_estimate.items[0].allowed)
        self.assertLess(
            tiled.items[0].estimated_peak_bytes,
            global_estimate.items[0].estimated_peak_bytes,
        )

    def test_roi_unsupported_capability_is_estimated_as_whole_image(
        self,
    ) -> None:
        from fdm.services import image_batch

        image = np.zeros((1024, 1024), dtype=np.uint8)
        roi = np.ones(image.shape, dtype=bool)
        without_roi = _input("plain", image)
        with_roi = BatchRasterInput(
            document_id="roi",
            display_name="ROI",
            raster=numpy_to_raster_plane(image),
            roi_mask=roi,
        )
        limits = BatchExecutionLimits(
            max_working_bytes=10 << 20,
            min_free_disk_bytes=1,
            max_documents=10,
        )
        recipe = _recipe("convert_color", {"target_model": "rgb"})

        with mock.patch.object(
            image_batch,
            "BATCH_PROCESSING_TILE_EDGE",
            256,
        ):
            plain = preflight_batch_recipe(
                BatchRecipeRequest(
                    request_id="plain",
                    generation=0,
                    recipe=recipe,
                    inputs=(without_roi,),
                    available_disk_bytes=_PLENTY_OF_DISK,
                ),
                limits=limits,
            )
            masked = preflight_batch_recipe(
                BatchRecipeRequest(
                    request_id="roi",
                    generation=0,
                    recipe=recipe,
                    inputs=(with_roi,),
                    available_disk_bytes=_PLENTY_OF_DISK,
                ),
                limits=limits,
            )

        self.assertTrue(plain.items[0].allowed)
        self.assertFalse(masked.items[0].allowed)

    def test_each_candidate_recipe_preserves_its_dynamic_result_metadata(
        self,
    ) -> None:
        first_array = np.asarray(
            [[np.nan, 0.25], [0.5, 1.0]],
            dtype=np.float32,
        )
        second_array = np.asarray(
            [[np.nan, np.inf], [-np.inf, 1.0]],
            dtype=np.float32,
        )
        operation = ImageOperationSpec(
            "convert_type",
            {
                "target_type": "uint8",
                "scale_mode": "full_type_range",
                "nonfinite_policy": "zero",
            },
            implementation="fdm",
            implementation_version="test-version",
            result_metadata={"preset_note": "保留"},
        )
        recipe = ImageProcessingRecipe.from_operations((operation,))
        request = BatchRecipeRequest(
            request_id="metadata-per-item",
            generation=6,
            recipe=recipe,
            inputs=(
                _input("first", first_array),
                _input("second", second_array),
            ),
            available_disk_bytes=_PLENTY_OF_DISK,
        )

        result = execute_batch_recipe(request)

        self.assertEqual(result.success_count, 2)
        metadata = [
            candidate.derivation.recipe.operations[0].result_metadata
            for candidate in result.commit_candidates
        ]
        self.assertEqual(
            [item["nonfinite_replacement_count"] for item in metadata],
            [1, 3],
        )
        self.assertEqual(
            [item["preset_note"] for item in metadata],
            ["保留", "保留"],
        )
        for candidate in result.commit_candidates:
            executed = candidate.derivation.recipe.operations[0]
            self.assertEqual(executed.implementation, "fdm")
            self.assertEqual(
                executed.implementation_version,
                "test-version",
            )
        self.assertEqual(
            request.recipe.operations[0].result_metadata,
            {"preset_note": "保留"},
        )

    def test_candidate_recipe_audits_threshold_repair_and_remainder_crop(
        self,
    ) -> None:
        image = np.arange(35, dtype=np.float32).reshape(5, 7)
        image[1, 2] = np.nan
        recipe = ImageProcessingRecipe.from_operations(
            (
                ImageOperationSpec(
                    "repair_nonfinite",
                    {"radius": 1, "fallback_value": 0.0},
                ),
                ImageOperationSpec(
                    "auto_threshold",
                    {"method": "otsu"},
                ),
                ImageOperationSpec(
                    "pixel_bin",
                    {
                        "factor": 2,
                        "method": "mean",
                        "remainder_policy": "crop",
                    },
                ),
            )
        )
        request = BatchRecipeRequest(
            request_id="metadata-all",
            generation=1,
            recipe=recipe,
            inputs=(_input("one", image),),
            available_disk_bytes=_PLENTY_OF_DISK,
        )

        result = execute_batch_recipe(request)

        self.assertEqual(result.success_count, 1)
        operations = result.commit_candidates[0].derivation.recipe.operations
        self.assertEqual(operations[0].result_metadata["repaired_count"], 1)
        self.assertIn("computed_threshold", operations[1].result_metadata)
        self.assertEqual(operations[2].result_metadata["cropped_right"], 1)
        self.assertEqual(operations[2].result_metadata["cropped_bottom"], 1)

    def test_all_pixel_layouts_are_validated_before_first_image_executes(
        self,
    ) -> None:
        from fdm.services import image_batch

        recipe = _recipe(
            "convert_type",
            {
                "target_type": "uint16",
                "scale_mode": "full_type_range",
                "nonfinite_policy": "reject",
            },
        )
        request = BatchRecipeRequest(
            request_id="layout-validation",
            generation=3,
            recipe=recipe,
            inputs=(
                _input("gray", np.zeros((4, 5), dtype=np.uint8)),
                _input("rgb", np.zeros((4, 5, 3), dtype=np.uint8)),
            ),
            available_disk_bytes=_PLENTY_OF_DISK,
        )
        real_validate = image_batch._validate_batch_operation_sequence
        real_execute = image_batch.execute_image_operation_tiled

        with (
            mock.patch.object(
                image_batch,
                "_validate_batch_operation_sequence",
                wraps=real_validate,
            ) as validate,
            mock.patch.object(
                image_batch,
                "execute_image_operation_tiled",
            ) as execute,
        ):
            def execute_after_preflight(*args, **kwargs):
                self.assertEqual(validate.call_count, 2)
                return real_execute(*args, **kwargs)

            execute.side_effect = execute_after_preflight
            result = execute_batch_recipe(request)

        self.assertEqual(validate.call_count, 2)
        self.assertEqual(execute.call_count, 1)
        self.assertEqual(result.items[0].status, BatchItemStatus.SUCCESS)
        self.assertEqual(result.items[1].status, BatchItemStatus.FAILED)
        self.assertIn("先添加", result.items[1].message)
        self.assertEqual(len(result.commit_candidates), 1)


if __name__ == "__main__":
    unittest.main()
