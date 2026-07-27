from __future__ import annotations

import os
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtWidgets import QApplication

from fdm.cancellation import CancellationTokenSource
from fdm.image_processing_models import (
    ImageOperationSpec,
    RasterSemantic,
)
from fdm.raster import RasterPixelType
from fdm.ui.image_parameter_widgets import (
    CropBoundsEditor,
    FrequencyResponseEditor,
    HistogramRangeEditor,
    LinkedDimensionsEditor,
    PercentileRangeEditor,
    SliderNumberEditor,
    StripeSuppressionEditor,
    StructuringElementEditor,
)
from fdm.ui.image_processing_workbench import (
    ImageProcessingWorkbench,
    WorkbenchTaskKind,
    WorkbenchTaskRequest,
    _execute_workbench_request_with_metadata,
    array_to_raster_plane,
    default_operation_spec,
    raster_plane_to_array,
)


class ImageParameterWorkbenchIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_preview_task_captures_exact_selected_step_input(self) -> None:
        source = array_to_raster_plane(
            np.asarray([[0, 50], [200, 255]], dtype=np.uint8)
        )
        convert = default_operation_spec(
            "convert_type",
            2,
            2,
            source_pixel_type=RasterPixelType.GRAY8,
        )
        convert_parameters = convert.parameters
        convert_parameters.update(
            {
                "target_type": "uint16",
                "scale_mode": "preserve_values",
            }
        )
        operations = (
            ImageOperationSpec("convert_type", convert_parameters),
            default_operation_spec(
                "binarize",
                2,
                2,
                source_pixel_type=RasterPixelType.GRAY16,
            ),
        )
        request = WorkbenchTaskRequest(
            kind=WorkbenchTaskKind.PREVIEW,
            request_id="capture-prefix",
            generation=3,
            source_document_id="doc-1",
            source=source,
            operations=operations,
            capture_step_input_index=1,
        )

        output = _execute_workbench_request_with_metadata(
            request,
            CancellationTokenSource().token,
        )

        self.assertIsNotNone(output.parameter_input_raster)
        captured = output.parameter_input_raster
        assert captured is not None
        self.assertIs(captured.pixel_type, RasterPixelType.GRAY16)
        self.assertIs(
            output.output_semantic,
            RasterSemantic.BINARY_MASK,
        )
        np.testing.assert_array_equal(
            raster_plane_to_array(captured),
            np.asarray([[0, 50], [200, 255]], dtype=np.uint16),
        )

    def test_binarize_uses_histogram_editor_and_persists_auto_value(self) -> None:
        source = array_to_raster_plane(
            np.asarray([[0, 0, 255, 255]], dtype=np.uint8)
        )
        dialog = ImageProcessingWorkbench(
            source,
            source_document_id="doc-1",
        )
        try:
            with mock.patch.object(dialog, "_schedule_preview"):
                dialog.set_operation_steps(
                    (
                        default_operation_spec(
                            "binarize",
                            4,
                            1,
                            source_pixel_type=RasterPixelType.GRAY8,
                        ),
                    )
                )
                editor = dialog._histogram_parameter_editor  # noqa: SLF001
                self.assertIsInstance(editor, HistogramRangeEditor)
                assert editor is not None
                self.assertEqual(editor.selectionStatistics(), (2, 4))

                editor.requestAuto()

            self.assertEqual(
                dialog.operation_steps()[0].parameters["threshold"],
                0.0,
            )
            self.assertEqual(editor.selectionStatistics(), (2, 4))
            provenance = dialog.operation_steps()[0].result_metadata[
                "auto_parameter_source"
            ]
            self.assertEqual(provenance["scope"], "full_source")
            self.assertEqual(provenance["method"], "otsu")
        finally:
            dialog.close()

    def test_canny_histogram_uses_gradient_magnitude_not_intensity(
        self,
    ) -> None:
        source_array = np.zeros((32, 32), dtype=np.uint8)
        source_array[:, 16:] = 255
        source = array_to_raster_plane(source_array)
        dialog = ImageProcessingWorkbench(
            source,
            source_document_id="doc-1",
        )
        try:
            with mock.patch.object(dialog, "_schedule_preview"):
                dialog.set_operation_steps(
                    (
                        default_operation_spec(
                            "canny_edges",
                            32,
                            32,
                        ),
                    )
                )
            editor = dialog._histogram_parameter_editor  # noqa: SLF001
            self.assertIsInstance(editor, HistogramRangeEditor)
            assert isinstance(editor, HistogramRangeEditor)
            self.assertGreater(editor.range()[1], 255.0)
            self.assertIn(
                "Sobel 梯度幅值",
                editor.toolTip(),
            )
            self.assertIn(
                "弱梯度候选",
                editor.selectionStatisticsLabel.text(),
            )
            self.assertNotIn(
                "当前前景",
                editor.selectionStatisticsLabel.text(),
            )
            self.assertIn(
                "不等于最终边缘像素数",
                editor.selectionStatisticsLabel.toolTip(),
            )
        finally:
            dialog.close()

    def test_current_binary_operation_rejects_implicit_thresholding(
        self,
    ) -> None:
        source = array_to_raster_plane(
            np.asarray([[0, 64], [128, 255]], dtype=np.uint8)
        )
        dialog = ImageProcessingWorkbench(
            source,
            source_document_id="doc-1",
            source_semantic=RasterSemantic.INTENSITY,
        )
        try:
            strict = default_operation_spec(
                "fill_holes",
                2,
                2,
            )
            self.assertEqual(strict.implementation_version, "2")
            with mock.patch.object(dialog, "_schedule_preview"):
                dialog.set_operation_steps((strict,))
            self.assertIn(
                "显式二值掩膜",
                dialog._parameter_error_message,  # noqa: SLF001
            )
            self.assertFalse(
                dialog._generate_button.isEnabled()  # noqa: SLF001
            )

            with mock.patch.object(dialog, "_schedule_preview"):
                dialog.set_operation_steps(
                    (
                        default_operation_spec("binarize", 2, 2),
                        strict,
                    )
                )
            self.assertEqual(
                dialog._parameter_error_message,  # noqa: SLF001
                "",
            )
            self.assertTrue(
                dialog._generate_button.isEnabled()  # noqa: SLF001
            )
        finally:
            dialog.close()

    def test_bounded_float_uses_slider_with_exact_spinbox_proxy(self) -> None:
        source = array_to_raster_plane(
            np.zeros((4, 4), dtype=np.uint8)
        )
        dialog = ImageProcessingWorkbench(
            source,
            source_document_id="doc-1",
        )
        try:
            with mock.patch.object(dialog, "_schedule_preview"):
                dialog.set_operation_steps(
                    (
                        default_operation_spec(
                            "brightness_contrast",
                            4,
                            4,
                        ),
                    )
                )
                row_widget = dialog._parameter_row_widgets[  # noqa: SLF001
                    "gamma"
                ]
                self.assertIsInstance(row_widget, SliderNumberEditor)
                proxy = dialog._parameter_widgets["gamma"]  # noqa: SLF001
                self.assertIs(proxy, row_widget.spinBox)
                proxy.setValue(1.375)
                proxy.editingFinished.emit()

            self.assertAlmostEqual(
                float(dialog.operation_steps()[0].parameters["gamma"]),
                1.375,
                places=3,
            )
        finally:
            dialog.close()

    def test_stripe_suppression_uses_directional_frequency_editor(self) -> None:
        source = array_to_raster_plane(
            np.zeros((12, 12), dtype=np.uint8)
        )
        dialog = ImageProcessingWorkbench(
            source,
            source_document_id="doc-1",
        )
        try:
            with mock.patch.object(dialog, "_schedule_preview"):
                dialog.set_operation_steps(
                    (
                        default_operation_spec(
                            "stripe_suppression",
                            12,
                            12,
                        ),
                    )
                )
                editor = dialog._structured_parameter_editors[  # noqa: SLF001
                    "stripe_frequency"
                ]
                self.assertIsInstance(
                    editor,
                    StripeSuppressionEditor,
                )
                assert isinstance(editor, StripeSuppressionEditor)
                editor.directionCombo.setCurrentIndex(
                    editor.directionCombo.findData("vertical")
                )
                editor.notchWidthSpin.setValue(0.05)
                editor.notchWidthSpin.editingFinished.emit()

            parameters = dialog.operation_steps()[0].parameters
            self.assertEqual(parameters["direction"], "vertical")
            self.assertAlmostEqual(
                float(parameters["notch_width"]),
                0.05,
                places=4,
            )
            self.assertIsInstance(
                dialog._parameter_row_widgets["strength"],  # noqa: SLF001
                SliderNumberEditor,
            )
        finally:
            dialog.close()

    def test_parameter_edit_preserves_hidden_contract_fields_and_metadata(
        self,
    ) -> None:
        source = array_to_raster_plane(
            np.asarray([[0, 64], [128, 255]], dtype=np.uint8)
        )
        dialog = ImageProcessingWorkbench(
            source,
            source_document_id="doc-1",
        )
        try:
            operation = ImageOperationSpec(
                "threshold",
                {
                    "lower": 25.0,
                    "upper": 220.0,
                    "invert": False,
                    "foreground_value": 201.0,
                    "background_value": 7.0,
                    "channel": "luminance",
                },
                result_metadata={"computed_threshold": 25.0},
            )
            with mock.patch.object(dialog, "_schedule_preview"):
                dialog.set_operation_steps((operation,))
                invert = dialog._parameter_widgets["invert"]  # noqa: SLF001
                invert.setChecked(True)

            saved = dialog.operation_steps()[0]
            self.assertTrue(saved.parameters["invert"])
            self.assertEqual(saved.parameters["foreground_value"], 201.0)
            self.assertEqual(saved.parameters["background_value"], 7.0)
            self.assertEqual(
                saved.result_metadata,
                {"computed_threshold": 25.0},
            )
        finally:
            dialog.close()

    def test_loaded_legacy_aliases_keep_their_original_numeric_semantics(
        self,
    ) -> None:
        source = array_to_raster_plane(
            np.zeros((8, 8), dtype=np.uint8)
        )
        dialog = ImageProcessingWorkbench(
            source,
            source_document_id="doc-1",
        )
        try:
            dialog.set_operation_steps(
                (
                    ImageOperationSpec(
                        "gaussian_blur",
                        {"sigma": 5.25},
                        result_metadata={"legacy": True},
                    ),
                )
            )
            loaded = dialog.operation_steps()[0]
            self.assertEqual(loaded.parameters["sigma"], 5.25)
            self.assertEqual(loaded.parameters["sigma_x"], 5.25)
            self.assertEqual(loaded.parameters["sigma_y"], 5.25)
            self.assertEqual(loaded.result_metadata, {"legacy": True})

            dialog.set_operation_steps(
                (ImageOperationSpec("fft_filter", {}),)
            )
            self.assertEqual(
                dialog.operation_steps()[0].parameters["boundary"],
                "periodic",
            )
            self.assertEqual(
                dialog.make_default_operation_spec(
                    "fft_filter"
                ).parameters["boundary"],
                "mirror_pad",
            )
        finally:
            dialog.close()

    def test_full_recipe_chain_blocks_a_locally_valid_upstream_edit(self) -> None:
        source = array_to_raster_plane(
            np.zeros((4, 4, 3), dtype=np.uint8)
        )
        dialog = ImageProcessingWorkbench(
            source,
            source_document_id="doc-1",
        )
        try:
            dialog.set_operation_steps(
                (
                    ImageOperationSpec(
                        "convert_color",
                        {
                            "target_model": "grayscale",
                            "grayscale_method": "rec601",
                            "drop_alpha": False,
                        },
                    ),
                    ImageOperationSpec(
                        "convert_type",
                        {
                            "target_type": "uint16",
                            "scale_mode": "preserve_values",
                            "nonfinite_policy": "reject",
                        },
                    ),
                )
            )
            dialog._steps_list.setCurrentRow(0)  # noqa: SLF001
            target = dialog._parameter_widgets["target_model"]  # noqa: SLF001
            target.setCurrentIndex(target.findData("rgb"))
            self.app.processEvents()

            self.assertFalse(dialog._generate_button.isEnabled())  # noqa: SLF001
            self.assertIn(
                "不能直接转换为 16 位",
                dialog._parameter_error_message,  # noqa: SLF001
            )
            self.assertEqual(
                dialog.operation_steps()[0].parameters["target_model"],
                "grayscale",
            )
        finally:
            dialog.close()

    def test_new_type_conversion_defaults_to_preserving_sample_values(
        self,
    ) -> None:
        operation = default_operation_spec(
            "convert_type",
            32,
            24,
            source_pixel_type=RasterPixelType.GRAY8,
        )
        self.assertEqual(
            operation.parameters["scale_mode"],
            "preserve_values",
        )

    def test_resize_uses_linked_dimensions_and_commits_exact_pixels(self) -> None:
        source = array_to_raster_plane(
            np.zeros((4, 8), dtype=np.uint8)
        )
        dialog = ImageProcessingWorkbench(
            source,
            source_document_id="doc-1",
        )
        try:
            with mock.patch.object(dialog, "_schedule_preview"):
                dialog.set_operation_steps(
                    (
                        default_operation_spec(
                            "resize",
                            8,
                            4,
                        ),
                    )
                )
                editor = dialog._structured_parameter_editors[  # noqa: SLF001
                    "linked_dimensions"
                ]
                self.assertIsInstance(editor, LinkedDimensionsEditor)
                assert isinstance(editor, LinkedDimensionsEditor)
                editor.widthSpin.setValue(4)
                editor.widthSpin.editingFinished.emit()

            parameters = dialog.operation_steps()[0].parameters
            self.assertEqual(parameters["width"], 4)
            self.assertEqual(parameters["height"], 2)
        finally:
            dialog.close()

    def test_crop_uses_source_bounded_editor(self) -> None:
        source = array_to_raster_plane(
            np.zeros((8, 12), dtype=np.uint8)
        )
        dialog = ImageProcessingWorkbench(
            source,
            source_document_id="doc-1",
        )
        try:
            with mock.patch.object(dialog, "_schedule_preview"):
                dialog.set_operation_steps(
                    (
                        default_operation_spec(
                            "crop",
                            12,
                            8,
                        ),
                    )
                )
                editor = dialog._structured_parameter_editors[  # noqa: SLF001
                    "crop_bounds"
                ]
                self.assertIsInstance(editor, CropBoundsEditor)
                assert isinstance(editor, CropBoundsEditor)
                editor.setValue(2, 1, 6, 4, emit_signal=False)
                editor.editFinished.emit()

            parameters = dialog.operation_steps()[0].parameters
            self.assertEqual(
                (
                    parameters["x"],
                    parameters["y"],
                    parameters["width"],
                    parameters["height"],
                ),
                (2, 1, 6, 4),
            )
        finally:
            dialog.close()

    def test_morphology_uses_structuring_element_editor(self) -> None:
        source = array_to_raster_plane(
            np.zeros((8, 8), dtype=np.uint8)
        )
        dialog = ImageProcessingWorkbench(
            source,
            source_document_id="doc-1",
        )
        try:
            with mock.patch.object(dialog, "_schedule_preview"):
                dialog.set_operation_steps(
                    (
                        default_operation_spec(
                            "erode",
                            8,
                            8,
                        ),
                    )
                )
                editor = dialog._structured_parameter_editors[  # noqa: SLF001
                    "structuring_element"
                ]
                self.assertIsInstance(editor, StructuringElementEditor)
                assert isinstance(editor, StructuringElementEditor)
                editor.setValue(
                    {
                        "radius": 3,
                        "iterations": 2,
                        "kernel": "cross",
                    },
                    emit_signal=False,
                )
                editor.editFinished.emit()

            parameters = dialog.operation_steps()[0].parameters
            self.assertEqual(parameters["radius"], 3)
            self.assertEqual(parameters["iterations"], 2)
            self.assertEqual(parameters["kernel"], "cross")
        finally:
            dialog.close()

    def test_fft_uses_response_editor_and_preserves_exact_values(self) -> None:
        source = array_to_raster_plane(
            np.zeros((8, 8), dtype=np.uint8)
        )
        dialog = ImageProcessingWorkbench(
            source,
            source_document_id="doc-1",
        )
        try:
            with mock.patch.object(dialog, "_schedule_preview"):
                dialog.set_operation_steps(
                    (
                        default_operation_spec(
                            "fft_filter",
                            8,
                            8,
                        ),
                    )
                )
                editor = dialog._structured_parameter_editors[  # noqa: SLF001
                    "frequency_response"
                ]
                self.assertIsInstance(editor, FrequencyResponseEditor)
                assert isinstance(editor, FrequencyResponseEditor)
                editor.setValue(
                    {
                        "mode": "bandpass",
                        "low_cutoff": 0.0625,
                        "high_cutoff": 0.25,
                        "order": 5,
                    },
                    emit_signal=False,
                )
                editor.editFinished.emit()

            parameters = dialog.operation_steps()[0].parameters
            self.assertEqual(parameters["mode"], "bandpass")
            self.assertAlmostEqual(parameters["low_cutoff"], 0.0625)
            self.assertAlmostEqual(parameters["high_cutoff"], 0.25)
            self.assertEqual(parameters["order"], 5)
        finally:
            dialog.close()

    def test_fft_editor_uses_dynamic_physical_nyquist_range(self) -> None:
        source = array_to_raster_plane(
            np.zeros((8, 8), dtype=np.uint8)
        )
        dialog = ImageProcessingWorkbench(
            source,
            source_document_id="doc-1",
        )
        try:
            operation = default_operation_spec(
                "fft_filter",
                8,
                8,
            )
            parameters = operation.parameters
            parameters.update(
                {
                    "frequency_unit": "cycles_per_unit",
                    "pixel_size": 0.2,
                    "high_cutoff": 2.0,
                }
            )
            with mock.patch.object(dialog, "_schedule_preview"):
                dialog.set_operation_steps(
                    (ImageOperationSpec("fft_filter", parameters),)
                )
            editor = dialog._structured_parameter_editors[  # noqa: SLF001
                "frequency_response"
            ]
            self.assertIsInstance(editor, FrequencyResponseEditor)
            assert isinstance(editor, FrequencyResponseEditor)
            self.assertEqual(editor.frequencyRange(), (0.0, 2.5))
            self.assertEqual(
                editor.highCutoffEditor.suffix(),
                " 周期/物理单位",
            )
            self.assertTrue(dialog._generate_button.isEnabled())  # noqa: SLF001
        finally:
            dialog.close()

    def test_percentile_saturation_resolves_current_rgb_channel_values(
        self,
    ) -> None:
        source = array_to_raster_plane(
            np.asarray(
                [
                    [[0, 10, 20], [100, 110, 120]],
                    [[200, 210, 220], [240, 250, 255]],
                ],
                dtype=np.uint8,
            )
        )
        dialog = ImageProcessingWorkbench(
            source,
            source_document_id="doc-1",
        )
        try:
            with mock.patch.object(dialog, "_schedule_preview"):
                dialog.set_operation_steps(
                    (
                        default_operation_spec(
                            "percentile_saturation",
                            2,
                            2,
                            source_pixel_type=RasterPixelType.RGB8,
                        ),
                    )
                )
                editor = dialog._structured_parameter_editors[  # noqa: SLF001
                    "percentile_range"
                ]
                self.assertIsInstance(editor, PercentileRangeEditor)
                assert isinstance(editor, PercentileRangeEditor)
                self.assertIn("按通道解析", editor.resolvedValuesLabel.text())
                self.assertIn("R ", editor.resolvedValuesLabel.text())

                editor.setValue(1.0, 99.0, emit_signal=False)
                editor.editFinished.emit()

            parameters = dialog.operation_steps()[0].parameters
            self.assertEqual(parameters["lower_percentile"], 1.0)
            self.assertEqual(parameters["upper_percentile"], 99.0)
        finally:
            dialog.close()

    def test_threshold_red_overlay_uses_frozen_input_not_binary_output(self) -> None:
        source = array_to_raster_plane(
            np.asarray([[0, 200]], dtype=np.uint8)
        )
        dialog = ImageProcessingWorkbench(
            source,
            source_document_id="doc-1",
        )
        try:
            with mock.patch.object(dialog, "_schedule_preview"):
                dialog.set_operation_steps(
                    (
                        default_operation_spec(
                            "threshold",
                            2,
                            1,
                        ),
                    )
                )
            editor = dialog._histogram_parameter_editor  # noqa: SLF001
            assert isinstance(editor, HistogramRangeEditor)
            editor.setThresholds(100.0, 255.0, emit_signal=False)
            editor.setDisplayMode("red_overlay", emit_signal=False)
            dialog._parameter_input_raster = source  # noqa: SLF001
            dialog._parameter_input_step_index = 0  # noqa: SLF001
            dialog._latest_preview_raster = source  # noqa: SLF001

            image = dialog._threshold_parameter_preview_image()  # noqa: SLF001

            self.assertIsNotNone(image)
            assert image is not None
            untouched = image.pixelColor(0, 0)
            selected = image.pixelColor(1, 0)
            self.assertEqual(
                (untouched.red(), untouched.green(), untouched.blue()),
                (0, 0, 0),
            )
            self.assertGreater(selected.red(), selected.green())
            self.assertGreater(selected.red(), selected.blue())
        finally:
            dialog.close()


if __name__ == "__main__":
    unittest.main()
