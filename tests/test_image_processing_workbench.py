from __future__ import annotations

import os
from pathlib import Path
import sys
import threading
import time
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    import numpy as np
    from PySide6.QtWidgets import QApplication, QComboBox, QDoubleSpinBox, QLabel, QSpinBox

    from fdm.cancellation import CancellationToken
    from fdm.image_processing_models import ImageOperationSpec
    from fdm.raster import RasterPixelType, RasterPlane
    from fdm.services.image_processing import ImageOperation
    import fdm.ui.image_processing_workbench as workbench_module
    from fdm.ui.image_processing_workbench import (
        FinalResourcePreflightError,
        ImageProcessingTaskController,
        ImageProcessingWorkbench,
        WorkbenchTaskKind,
        WorkbenchTaskRequest,
        WorkbenchTaskResult,
        array_to_raster_plane,
        default_operation_spec,
        estimate_final_resources,
        execute_workbench_request,
        raster_plane_to_array,
        validate_workbench_operation_sequence,
        validate_final_resources,
    )

    PYSIDE_AVAILABLE = True
except ModuleNotFoundError:
    PYSIDE_AVAILABLE = False


class _FakeWheelEvent:
    def __init__(self) -> None:
        self.ignored = False
        self.accepted = False

    def ignore(self) -> None:
        self.ignored = True

    def accept(self) -> None:
        self.accepted = True


@unittest.skipUnless(PYSIDE_AVAILABLE, "PySide6 is not installed")
class ImageProcessingWorkbenchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.source = RasterPlane(
            width=8,
            height=6,
            pixel_type=RasterPixelType.GRAY8,
            data=bytes(range(48)),
        )

    def _wait_until(self, predicate, timeout: float = 3.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.app.processEvents()
            if predicate():
                return
            time.sleep(0.005)
        self.fail("等待异步 UI 条件超时")

    def test_dialog_is_non_modal_chinese_and_has_safe_minimum_layout(self) -> None:
        dialog = ImageProcessingWorkbench(
            self.source,
            source_document_id="doc-1",
            source_name="显微图像一",
            roi_summary="ROI：孔洞区域",
        )
        try:
            self.assertFalse(dialog.isModal())
            self.assertEqual(dialog.windowTitle(), "图像处理工作台")
            self.assertGreaterEqual(dialog.minimumWidth(), 780)
            self.assertGreaterEqual(dialog.minimumHeight(), 520)
            self.assertIn("源图片", dialog._source_label.text())  # noqa: SLF001
            self.assertIn("显微图像一", dialog._source_label.text())  # noqa: SLF001
            self.assertIn("ROI：孔洞区域", dialog._roi_label.text())  # noqa: SLF001
            self.assertEqual(dialog._generate_button.text(), "生成派生图片")  # noqa: SLF001
            self.assertEqual(dialog._cancel_button.text(), "取消")  # noqa: SLF001
            categories = [
                dialog._category_combo.itemText(index)  # noqa: SLF001
                for index in range(dialog._category_combo.count())  # noqa: SLF001
            ]
            self.assertEqual(categories, ["类型", "调整", "变换", "处理"])
        finally:
            dialog.close()

    def test_step_reset_undo_redo_and_parameter_form_are_ordered(self) -> None:
        dialog = ImageProcessingWorkbench(
            self.source,
            source_document_id="doc-1",
        )
        try:
            dialog._category_combo.setCurrentText("变换")  # noqa: SLF001
            dialog._operation_combo.setCurrentIndex(0)  # noqa: SLF001
            dialog._add_selected_operation()  # noqa: SLF001
            dialog._category_combo.setCurrentText("处理")  # noqa: SLF001
            gaussian_index = dialog._operation_combo.findData("gaussian_blur")  # noqa: SLF001
            dialog._operation_combo.setCurrentIndex(gaussian_index)  # noqa: SLF001
            dialog._add_selected_operation()  # noqa: SLF001

            self.assertEqual(len(dialog.operation_steps()), 2)
            self.assertEqual(dialog.operation_steps()[1].operation_id, "gaussian_blur")
            self.assertTrue(
                any(
                    isinstance(widget, QDoubleSpinBox)
                    for widget in dialog._parameter_widgets.values()  # noqa: SLF001
                )
            )

            dialog._reset_steps()  # noqa: SLF001
            self.assertEqual(dialog.operation_steps(), ())
            dialog._undo_steps()  # noqa: SLF001
            self.assertEqual(len(dialog.operation_steps()), 2)
            dialog._redo_steps()  # noqa: SLF001
            self.assertEqual(dialog.operation_steps(), ())
        finally:
            dialog.close()

    def test_public_default_operation_spec_and_empty_main_window_preset_are_safe(self) -> None:
        resize = default_operation_spec(
            "resize",
            640,
            480,
            source_pixel_type=RasterPixelType.GRAY16,
        )
        self.assertEqual(resize.parameters["width"], 640)
        self.assertEqual(resize.parameters["height"], 480)
        self.assertEqual(resize.parameters["interpolation"], "area")

        levels = default_operation_spec(
            "adjust_levels",
            640,
            480,
            source_pixel_type=RasterPixelType.GRAY16,
        )
        self.assertEqual(levels.parameters["white_point"], 65_535.0)

        dialog = ImageProcessingWorkbench(
            self.source,
            source_document_id="doc-1",
        )
        try:
            dialog.set_operation_steps((ImageOperationSpec("custom_convolution"),))
            resolved = dialog.operation_steps()[0].parameters
            self.assertEqual(resolved["kernel_width"], 3)
            self.assertEqual(resolved["kernel_height"], 3)
            self.assertEqual(len(resolved["kernel"]), 9)
            request = WorkbenchTaskRequest(
                kind=WorkbenchTaskKind.FINAL,
                request_id="empty-preset",
                generation=1,
                source_document_id="doc-1",
                source=self.source,
                operations=dialog.operation_steps(),
            )
            from fdm.cancellation import CancellationTokenSource

            execute_workbench_request(
                request,
                CancellationTokenSource().token,
            )
        finally:
            dialog.close()

    def test_catalog_covers_every_service_operation_with_explicit_help(self) -> None:
        definitions = workbench_module._OPERATION_CATALOG  # noqa: SLF001
        self.assertEqual(
            {definition.operation for definition in definitions},
            set(ImageOperation),
        )
        self.assertEqual(
            len({definition.operation for definition in definitions}),
            len(definitions),
        )
        for definition in definitions:
            with self.subTest(operation=definition.operation.value):
                self.assertTrue(definition.purpose)
                self.assertIn("像素", definition.pixel_effect)
                self.assertTrue(definition.calibration_effect)
                self.assertTrue(definition.supported_types)
                self.assertTrue(definition.roi_behavior)

        schemas = {
            definition.operation.value: {
                field.key for field in definition.parameters
            }
            for definition in definitions
        }
        self.assertEqual(
            schemas["convert_color"],
            {"target_model", "grayscale_method", "drop_alpha"},
        )
        self.assertEqual(
            schemas["color_balance"],
            {
                "red_gain",
                "green_gain",
                "blue_gain",
                "red_offset",
                "green_offset",
                "blue_offset",
            },
        )
        self.assertEqual(
            schemas["translate"],
            {
                "offset_x",
                "offset_y",
                "interpolation",
                "border_mode",
                "border_value",
            },
        )
        self.assertEqual(
            schemas["resize_canvas"],
            {"width", "height", "anchor", "fill_value"},
        )
        self.assertEqual(
            schemas["pixel_bin"],
            {"factor", "method", "remainder_policy"},
        )
        self.assertEqual(
            schemas["convert_type"],
            {"target_type", "scale_mode", "nonfinite_policy"},
        )

    def test_catalog_only_offers_wrap_for_backends_that_support_it(self) -> None:
        definitions = {
            definition.operation: definition
            for definition in workbench_module._OPERATION_CATALOG  # noqa: SLF001
        }

        def border_values(operation: ImageOperation) -> set[object]:
            definition = definitions[operation]
            field = next(
                item
                for item in definition.parameters
                if item.key == "border_mode"
            )
            return {value for _label, value in field.choices}

        for operation in (
            ImageOperation.MEAN_FILTER,
            ImageOperation.MORPHOLOGY_OPEN,
            ImageOperation.BACKGROUND_SUBTRACT,
            ImageOperation.CUSTOM_CONVOLUTION,
        ):
            with self.subTest(operation=operation.value):
                self.assertNotIn("wrap", border_values(operation))

        self.assertIn(
            "wrap",
            border_values(ImageOperation.GAUSSIAN_BLUR),
        )
        self.assertIn(
            "wrap",
            border_values(ImageOperation.BILATERAL_FILTER),
        )

    def test_parameter_help_states_pixel_calibration_type_and_roi_behavior(self) -> None:
        dialog = ImageProcessingWorkbench(
            self.source,
            source_document_id="doc-1",
        )
        try:
            dialog.set_operation_steps(
                (ImageOperationSpec("resize", {"width": 4, "height": 3, "interpolation": "area"}),)
            )
            self.app.processEvents()
            help_labels = [
                label
                for label in dialog.findChildren(QLabel)
                if label.objectName() == "imageOperationHelp"
            ]
            self.assertEqual(len(help_labels), 1)
            help_text = help_labels[0].text()
            for heading in ("用途：", "像素：", "标定：", "适用类型：", "ROI："):
                self.assertIn(heading, help_text)
            self.assertIn("pixels_per_unit", help_text)
        finally:
            dialog.close()

    def test_whole_image_operation_help_explains_cancellation_boundary(self) -> None:
        dialog = ImageProcessingWorkbench(
            self.source,
            source_document_id="doc-1",
        )
        try:
            dialog.set_operation_steps(
                (
                    ImageOperationSpec(
                        "bilateral_filter",
                        {
                            "diameter": 5,
                            "sigma_color": 25.0,
                            "sigma_space": 2.0,
                            "border_mode": "reflect",
                        },
                    ),
                )
            )
            self.app.processEvents()
            help_text = next(
                label.text()
                for label in dialog.findChildren(QLabel)
                if label.objectName() == "imageOperationHelp"
            )
            self.assertIn("整图执行", help_text)
            self.assertIn("逐位一致", help_text)
            self.assertIn("算法返回后立即确认", help_text)
            self.assertIn("不会提交派生图片", help_text)
        finally:
            dialog.close()

    def test_image_calculator_only_appears_with_a_secondary_image_and_executes(self) -> None:
        without_secondary = ImageProcessingWorkbench(
            self.source,
            source_document_id="doc-1",
        )
        secondary = RasterPlane(
            width=8,
            height=6,
            pixel_type=RasterPixelType.GRAY8,
            data=bytes([2] * 48),
        )
        with_secondary = ImageProcessingWorkbench(
            self.source,
            source_document_id="doc-1",
            secondary_images={"doc-2": secondary},
            secondary_image_names={"doc-2": "参照图片"},
        )
        try:
            without_secondary._category_combo.setCurrentText("处理")  # noqa: SLF001
            with_secondary._category_combo.setCurrentText("处理")  # noqa: SLF001
            self.assertEqual(
                without_secondary._operation_combo.findData("image_calculator"),  # noqa: SLF001
                -1,
            )
            self.assertGreaterEqual(
                with_secondary._operation_combo.findData("image_calculator"),  # noqa: SLF001
                0,
            )

            request = WorkbenchTaskRequest(
                kind=WorkbenchTaskKind.FINAL,
                request_id="calculator-1",
                generation=1,
                source_document_id="doc-1",
                source=self.source,
                operations=(
                    ImageOperationSpec(
                        "image_calculator",
                        {
                            "secondary_document_id": "doc-2",
                            "calculator_operation": "add",
                        },
                    ),
                ),
                secondary_images=(("doc-2", secondary),),
            )
            from fdm.cancellation import CancellationTokenSource

            result = execute_workbench_request(
                request,
                CancellationTokenSource().token,
            )
            expected = np.clip(
                raster_plane_to_array(self.source).astype(np.int16) + 2,
                0,
                255,
            ).astype(np.uint8)
            np.testing.assert_array_equal(raster_plane_to_array(result), expected)
        finally:
            without_secondary.close()
            with_secondary.close()

    def test_every_catalog_default_is_accepted_by_the_service(self) -> None:
        gray = self.source
        rgb_values = np.stack(
            [np.arange(48, dtype=np.uint8).reshape(6, 8)] * 3,
            axis=2,
        )
        rgb = array_to_raster_plane(rgb_values)
        float_values = np.arange(48, dtype=np.float32).reshape(6, 8) / 47.0
        float_values[0, 0] = np.nan
        float_source = array_to_raster_plane(float_values)
        dialog = ImageProcessingWorkbench(
            gray,
            source_document_id="doc-1",
            secondary_images={"doc-2": gray},
        )
        from fdm.cancellation import CancellationTokenSource

        try:
            for definition in workbench_module._OPERATION_CATALOG:  # noqa: SLF001
                source = (
                    rgb
                    if definition.operation
                    in {ImageOperation.CONVERT_COLOR, ImageOperation.COLOR_BALANCE}
                    else (
                        float_source
                        if definition.operation is ImageOperation.REPAIR_NONFINITE
                        else gray
                    )
                )
                parameters = {
                    field.key: dialog._resolved_default(field)  # noqa: SLF001
                    for field in definition.parameters
                }
                request = WorkbenchTaskRequest(
                    kind=WorkbenchTaskKind.FINAL,
                    request_id=f"default-{definition.operation.value}",
                    generation=1,
                    source_document_id="doc-1",
                    source=source,
                    operations=(
                        ImageOperationSpec(
                            definition.operation.value,
                            parameters,
                        ),
                    ),
                    secondary_images=(("doc-2", gray),),
                )
                with self.subTest(operation=definition.operation.value):
                    execute_workbench_request(
                        request,
                        CancellationTokenSource().token,
                    )
        finally:
            dialog.close()

    def test_all_combo_and_numeric_editors_ignore_incidental_wheel(self) -> None:
        dialog = ImageProcessingWorkbench(
            self.source,
            source_document_id="doc-1",
        )
        try:
            dialog.set_operation_steps(
                (
                    ImageOperationSpec(
                        "brightness_contrast",
                        {"brightness": 0.0, "contrast": 1.0, "gamma": 1.0},
                    ),
                )
            )
            self.app.processEvents()
            editors = [
                *dialog.findChildren(QComboBox),
                *dialog.findChildren(QSpinBox),
                *dialog.findChildren(QDoubleSpinBox),
            ]
            self.assertTrue(editors)
            for editor in editors:
                with self.subTest(editor=type(editor).__name__):
                    before = (
                        editor.currentIndex()
                        if isinstance(editor, QComboBox)
                        else editor.value()
                    )
                    event = _FakeWheelEvent()
                    editor.wheelEvent(event)
                    after = (
                        editor.currentIndex()
                        if isinstance(editor, QComboBox)
                        else editor.value()
                    )
                    self.assertEqual(after, before)
                    self.assertTrue(event.ignored or event.accepted)
        finally:
            dialog.close()

    def test_preview_lane_discards_cancelled_generation_and_only_publishes_latest(self) -> None:
        first_started = threading.Event()
        release_first = threading.Event()
        calls: list[int] = []

        def executor(
            request: WorkbenchTaskRequest,
            token: CancellationToken,
        ) -> RasterPlane:
            calls.append(request.generation)
            if len(calls) == 1:
                first_started.set()
                release_first.wait(2.0)
            token.raise_if_cancelled()
            data = bytes([request.generation] * (request.source.width * request.source.height))
            return RasterPlane(
                request.source.width,
                request.source.height,
                RasterPixelType.GRAY8,
                data,
            )

        controller = ImageProcessingTaskController(executor=executor)
        ready: list[WorkbenchTaskResult] = []
        controller.previewReady.connect(ready.append)
        operation = (ImageOperationSpec("flip_horizontal"),)
        try:
            first = controller.start_preview(
                source_document_id="doc-1",
                source=self.source,
                operations=operation,
            )
            self.assertTrue(first_started.wait(1.0))
            second = controller.start_preview(
                source_document_id="doc-1",
                source=self.source,
                operations=operation,
            )
            release_first.set()
            self._wait_until(lambda: len(ready) == 1)

            self.assertGreater(second.generation, first.generation)
            self.assertEqual(ready[0].request_id, second.request_id)
            self.assertEqual(ready[0].generation, second.generation)
            self.assertEqual(list(ready[0].raster.data), [second.generation] * 48)
            self.assertEqual(len(calls), 2)
        finally:
            release_first.set()
            controller.close()
            controller.wait_for_done()

    def test_busy_changed_emits_only_one_true_false_pair_across_pending_work(self) -> None:
        first_started = threading.Event()
        release_first = threading.Event()
        call_count = 0

        def executor(
            request: WorkbenchTaskRequest,
            token: CancellationToken,
        ) -> RasterPlane:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                first_started.set()
                release_first.wait(2.0)
            token.raise_if_cancelled()
            return request.source

        controller = ImageProcessingTaskController(executor=executor)
        states: list[tuple[str, bool]] = []
        controller.busyChanged.connect(
            lambda kind, busy: states.append((kind, busy))
        )
        operation = (ImageOperationSpec("flip_horizontal"),)
        try:
            controller.start_preview(
                source_document_id="doc-1",
                source=self.source,
                operations=operation,
            )
            self.assertTrue(first_started.wait(1.0))
            controller.start_preview(
                source_document_id="doc-1",
                source=self.source,
                operations=operation,
            )
            release_first.set()
            self._wait_until(
                lambda: not controller.is_busy(WorkbenchTaskKind.PREVIEW)
            )
            self.assertEqual(
                states,
                [
                    (WorkbenchTaskKind.PREVIEW.value, True),
                    (WorkbenchTaskKind.PREVIEW.value, False),
                ],
            )
        finally:
            release_first.set()
            controller.close()
            controller.wait_for_done()

    def test_final_cancellation_never_emits_success(self) -> None:
        started = threading.Event()
        release = threading.Event()

        def executor(
            request: WorkbenchTaskRequest,
            token: CancellationToken,
        ) -> RasterPlane:
            started.set()
            release.wait(2.0)
            token.raise_if_cancelled()
            return request.source

        controller = ImageProcessingTaskController(executor=executor)
        ready: list[WorkbenchTaskResult] = []
        controller.finalReady.connect(ready.append)
        try:
            controller.start_final(
                source_document_id="doc-1",
                source=self.source,
                operations=(ImageOperationSpec("flip_horizontal"),),
            )
            self.assertTrue(started.wait(1.0))
            controller.cancel_final()
            release.set()
            self._wait_until(
                lambda: not controller.is_busy(WorkbenchTaskKind.FINAL)
            )
            self.assertEqual(ready, [])
        finally:
            release.set()
            controller.close()
            controller.wait_for_done()

    def test_default_executor_preserves_request_identity_and_applies_steps(self) -> None:
        request = WorkbenchTaskRequest(
            kind=WorkbenchTaskKind.FINAL,
            request_id="request-42",
            generation=7,
            source_document_id="doc-1",
            source=self.source,
            operations=(
                ImageOperationSpec("flip_horizontal"),
                ImageOperationSpec("flip_vertical"),
            ),
        )
        from fdm.cancellation import CancellationTokenSource

        output = execute_workbench_request(
            request,
            CancellationTokenSource().token,
        )
        expected = np.flip(
            np.flip(raster_plane_to_array(self.source), axis=1),
            axis=0,
        )
        np.testing.assert_array_equal(raster_plane_to_array(output), expected)
        self.assertEqual(output.pixel_type, RasterPixelType.GRAY8)

    def test_array_plane_round_trip_supports_native_types(self) -> None:
        arrays = (
            np.arange(12, dtype=np.uint8).reshape(3, 4),
            np.arange(12, dtype=np.uint16).reshape(3, 4),
            np.arange(12, dtype=np.float32).reshape(3, 4) / 3.0,
            np.arange(36, dtype=np.uint8).reshape(3, 4, 3),
            np.arange(48, dtype=np.uint8).reshape(3, 4, 4),
        )
        for original in arrays:
            with self.subTest(dtype=original.dtype, shape=original.shape):
                plane = array_to_raster_plane(original)
                restored = raster_plane_to_array(plane)
                np.testing.assert_array_equal(restored, original)
                self.assertFalse(restored.flags.writeable)

    def test_final_result_persists_dynamic_operation_metadata_in_recipe(
        self,
    ) -> None:
        source_array = np.linspace(
            0.0,
            1.0,
            48,
            dtype=np.float32,
        ).reshape(6, 8)
        source_array[1, 2] = np.nan
        source_array[4, 6] = np.inf
        source = array_to_raster_plane(source_array)
        controller = ImageProcessingTaskController()
        ready: list[WorkbenchTaskResult] = []
        controller.finalReady.connect(ready.append)
        try:
            controller.start_final(
                source_document_id="doc-float",
                source=source,
                operations=(
                    ImageOperationSpec(
                        "convert_type",
                        {
                            "target_type": "uint8",
                            "scale_mode": "full_type_range",
                            "nonfinite_policy": "zero",
                        },
                    ),
                ),
            )
            self._wait_until(lambda: len(ready) == 1)

            metadata = ready[0].recipe.operations[0].result_metadata
            self.assertEqual(
                metadata["nonfinite_replacement_count"],
                2,
            )
        finally:
            controller.close()
            controller.wait_for_done()

    def test_colored_high_depth_conversion_is_rejected_before_task_launch(
        self,
    ) -> None:
        rgb = RasterPlane(
            width=4,
            height=3,
            pixel_type=RasterPixelType.RGB8,
            data=bytes(4 * 3 * 3),
        )
        operations = (
            ImageOperationSpec(
                "convert_type",
                {
                    "target_type": "uint16",
                    "scale_mode": "full_type_range",
                    "nonfinite_policy": "reject",
                },
            ),
        )
        executor = mock.Mock(return_value=rgb)
        controller = ImageProcessingTaskController(executor=executor)
        try:
            with self.assertRaisesRegex(ValueError, "先添加.*灰度"):
                controller.start_final(
                    source_document_id="doc-rgb",
                    source=rgb,
                    operations=operations,
                )
            executor.assert_not_called()
            with self.assertRaisesRegex(ValueError, "先添加.*灰度"):
                validate_workbench_operation_sequence(rgb, operations)
        finally:
            controller.close()
            controller.wait_for_done()

        allowed = (
            ImageOperationSpec(
                "convert_color",
                {
                    "target_model": "grayscale",
                    "grayscale_method": "rec601",
                    "drop_alpha": False,
                },
            ),
            *operations,
        )
        validate_workbench_operation_sequence(rgb, allowed)

    def test_final_resource_estimate_tracks_geometry_type_and_fft_peak(self) -> None:
        estimate = estimate_final_resources(
            self.source,
            (
                ImageOperationSpec(
                    "resize",
                    {"width": 16, "height": 12, "interpolation": "linear"},
                ),
                ImageOperationSpec(
                    "convert_type",
                    {"target_type": "float32", "scale_mode": "full_type_range"},
                ),
                ImageOperationSpec(
                    "fft_filter",
                    {
                        "mode": "lowpass",
                        "low_cutoff": 0.0,
                        "high_cutoff": 0.15,
                        "order": 2,
                        "channel": "per_channel",
                        "output_float": True,
                    },
                ),
            ),
        )
        self.assertEqual((estimate.output_width, estimate.output_height), (16, 12))
        self.assertEqual(estimate.output_bytes, 16 * 12 * 4)
        self.assertGreater(estimate.peak_working_set_bytes, estimate.output_bytes)

    def test_large_canvas_resize_uses_high_memory_safety_family(self) -> None:
        estimate = estimate_final_resources(
            RasterPlane(
                width=1,
                height=1,
                pixel_type=RasterPixelType.GRAY8,
                data=b"\0",
            ),
            (
                ImageOperationSpec(
                    "resize_canvas",
                    {
                        "width": 7_000,
                        "height": 7_000,
                        "anchor": "center",
                        "fill_value": 0.0,
                    },
                ),
            ),
        )

        expected_pixels = 7_000 * 7_000
        self.assertEqual(estimate.output_bytes, expected_pixels)
        self.assertEqual(
            estimate.peak_working_set_bytes,
            1 + expected_pixels + expected_pixels * 24,
        )
        self.assertGreater(
            estimate.peak_working_set_bytes,
            workbench_module.MAX_FINAL_WORKING_SET_BYTES,
        )

    def test_final_resource_preflight_blocks_memory_and_disk_in_chinese(self) -> None:
        operation = (ImageOperationSpec("flip_horizontal"),)
        with mock.patch.object(
            workbench_module,
            "MAX_FINAL_WORKING_SET_BYTES",
            1,
        ):
            with self.assertRaisesRegex(
                FinalResourcePreflightError,
                "超过 1 GiB 安全上限",
            ):
                validate_final_resources(self.source, operation)

        fake_disk_usage = mock.Mock()
        fake_disk_usage.free = workbench_module.MIN_FREE_DISK_RESERVE_BYTES
        with mock.patch.object(
            workbench_module.shutil,
            "disk_usage",
            return_value=fake_disk_usage,
        ):
            with self.assertRaisesRegex(
                FinalResourcePreflightError,
                "无法在完成后保留至少 2 GiB",
            ):
                validate_final_resources(self.source, operation)

    def test_final_preflight_failure_does_not_launch_task(self) -> None:
        dialog = ImageProcessingWorkbench(
            self.source,
            source_document_id="doc-1",
        )
        try:
            dialog.set_operation_steps((ImageOperationSpec("flip_horizontal"),))
            with (
                mock.patch.object(
                    workbench_module,
                    "validate_final_resources",
                    side_effect=FinalResourcePreflightError("测试资源不足"),
                ),
                mock.patch.object(
                    workbench_module.QMessageBox,
                    "warning",
                ) as warning,
            ):
                dialog._generate_derived_image()  # noqa: SLF001
            self.assertFalse(
                dialog.task_controller.is_busy(WorkbenchTaskKind.FINAL)
            )
            self.assertIn("测试资源不足", dialog._status_label.text())  # noqa: SLF001
            warning.assert_called_once()
        finally:
            dialog.close()


if __name__ == "__main__":
    unittest.main()
