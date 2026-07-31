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
    from PySide6.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QDoubleSpinBox,
        QLabel,
        QLineEdit,
        QSpinBox,
    )

    from fdm.cancellation import CancellationToken
    from fdm.image_processing_models import (
        ImageOperationSpec,
        RasterSemantic,
    )
    from fdm.raster import RasterPixelType, RasterPlane
    from fdm.services.image_processing import (
        IMAGE_OPERATION_REGISTRY,
        ImageOperation,
    )
    import fdm.ui.image_processing_workbench as workbench_module
    from fdm.ui.image_processing_workbench import (
        FinalResourcePreflightError,
        ImageProcessingTaskController,
        ImageProcessingWorkbench,
        WorkbenchTaskKind,
        WorkbenchTaskRequest,
        WorkbenchTaskResult,
        adapt_operations_for_preview,
        array_to_raster_plane,
        build_processing_preview_snapshot,
        default_operation_spec,
        estimate_final_resources,
        execute_workbench_request,
        expand_processing_preview_snapshot_for_halo,
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

    def test_task_semantic_fields_preserve_legacy_positional_order(self) -> None:
        request = WorkbenchTaskRequest(
            WorkbenchTaskKind.FINAL,
            "legacy-order",
            3,
            "doc-1",
            self.source,
            (ImageOperationSpec("invert"),),
            None,
            (("doc-2", self.source),),
            0,
        )

        self.assertEqual(request.capture_step_input_index, 0)
        self.assertIs(
            request.source_semantic,
            RasterSemantic.INTENSITY,
        )
        self.assertEqual(
            request.secondary_semantics,
            (("doc-2", RasterSemantic.INTENSITY),),
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

    def test_fft_power_spectrum_is_replay_only_in_workbench(self) -> None:
        dialog = ImageProcessingWorkbench(
            self.source,
            source_document_id="doc-1",
        )
        try:
            process_index = dialog._category_combo.findData("处理")  # noqa: SLF001
            self.assertGreaterEqual(process_index, 0)
            dialog._category_combo.setCurrentIndex(process_index)  # noqa: SLF001
            self.app.processEvents()
            available = {
                str(dialog._operation_combo.itemData(index))  # noqa: SLF001
                for index in range(dialog._operation_combo.count())  # noqa: SLF001
            }
            self.assertNotIn(ImageOperation.FFT_POWER_SPECTRUM.value, available)

            dialog.set_operation_steps(
                (
                    ImageOperationSpec(
                        ImageOperation.FFT_POWER_SPECTRUM.value,
                        {
                            "channel": "luminance",
                            "logarithmic": True,
                            "centered": True,
                            "window": "none",
                            "tukey_alpha": 0.25,
                        },
                        implementation="fdm",
                        implementation_version="1",
                    ),
                )
            )
            self.assertEqual(
                dialog.operation_steps()[0].operation_id,
                ImageOperation.FFT_POWER_SPECTRUM.value,
            )
            self.assertTrue(dialog._generate_button.isEnabled())  # noqa: SLF001
            self.assertTrue(dialog._parameter_widgets)  # noqa: SLF001
            self.assertTrue(
                all(
                    not widget.isEnabled()
                    for widget in dialog._parameter_widgets.values()  # noqa: SLF001
                )
            )
            self.assertFalse(dialog._add_step_button.isEnabled())  # noqa: SLF001
            self.assertFalse(dialog._remove_step_button.isEnabled())  # noqa: SLF001
            self.assertFalse(dialog._save_recipe_button.isEnabled())  # noqa: SLF001
            self.assertFalse(dialog._batch_apply_button.isEnabled())  # noqa: SLF001
            self.assertIn(
                "仅供旧项目重放",
                dialog._save_recipe_button.toolTip(),  # noqa: SLF001
            )
            with self.assertRaisesRegex(ValueError, "只允许按 fdm v1"):
                dialog.set_operation_steps(
                    (
                        ImageOperationSpec(
                            ImageOperation.FFT_POWER_SPECTRUM.value,
                            {},
                            implementation_version="2",
                        ),
                    )
                )
        finally:
            dialog.close()

    def test_versioned_scientific_contracts_are_replay_only_in_workbench(
        self,
    ) -> None:
        floating = array_to_raster_plane(
            np.linspace(-5.0, 20.0, 48, dtype=np.float32).reshape(6, 8)
        )
        dialog = ImageProcessingWorkbench(
            floating,
            source_document_id="doc-float",
        )
        try:
            for operation_id in (
                ImageOperation.BRIGHTNESS_CONTRAST.value,
                ImageOperation.HISTOGRAM_EQUALIZATION.value,
            ):
                with self.subTest(operation=operation_id):
                    current = default_operation_spec(
                        operation_id,
                        floating.width,
                        floating.height,
                        source_pixel_type=floating.pixel_type,
                    )
                    self.assertEqual(current.implementation_version, "2")
                    dialog.set_operation_steps(
                        (
                            ImageOperationSpec(
                                operation_id,
                                implementation_version="1",
                            ),
                        )
                    )
                    self.assertTrue(
                        dialog._contains_replay_only_steps()  # noqa: SLF001
                    )
                    self.assertTrue(
                        all(
                            not widget.isEnabled()
                            for widget in dialog._parameter_widgets.values()  # noqa: SLF001
                        )
                    )
                    self.assertIn(
                        "0–1 工作范围",
                        dialog._parameter_content.findChild(  # noqa: SLF001
                            QLabel,
                            "imageParameterScientificWarning",
                        ).text(),
                    )
                    dialog.set_operation_steps((current,))
                    self.assertFalse(
                        dialog._generate_button.isEnabled()  # noqa: SLF001
                    )
                    self.assertIn(
                        "没有可安全推断的 0–1 工作范围",
                        dialog._parameter_error_message,  # noqa: SLF001
                    )

            self.assertEqual(
                default_operation_spec(
                    ImageOperation.IMAGE_CALCULATOR,
                    8,
                    6,
                    secondary_document_id="other",
                ).implementation_version,
                "2",
            )
            self.assertEqual(
                default_operation_spec(
                    ImageOperation.FLAT_FIELD_CORRECTION,
                    8,
                    6,
                ).implementation_version,
                "2",
            )
        finally:
            dialog.close()

    def test_preview_snapshot_is_bounded_unscaled_and_crops_roi_and_secondary(
        self,
    ) -> None:
        image = np.arange(3_000 * 4_000, dtype=np.uint16).reshape(3_000, 4_000)
        source = array_to_raster_plane(image)
        roi = np.zeros((3_000, 4_000), dtype=bool)
        roi[700:900, 1_200:1_500] = True
        secondary = array_to_raster_plane(image + 1)

        snapshot = build_processing_preview_snapshot(
            source,
            visible_rect=(900.25, 400.5, 2_800.0, 2_400.0),
            roi_mask=roi,
            secondary_images={"secondary": secondary},
        )

        self.assertLessEqual(snapshot.source.width, 2_048)
        self.assertLessEqual(snapshot.source.height, 2_048)
        self.assertLessEqual(
            snapshot.source.width * snapshot.source.height,
            workbench_module.PREVIEW_MAX_PIXELS,
        )
        self.assertFalse(snapshot.is_full_source)
        x, y, width, height = snapshot.bounds
        np.testing.assert_array_equal(
            raster_plane_to_array(snapshot.source),
            image[y : y + height, x : x + width],
        )
        np.testing.assert_array_equal(
            snapshot.roi_mask,
            roi[y : y + height, x : x + width],
        )
        secondary_sample = dict(snapshot.secondary_images)["secondary"]
        np.testing.assert_array_equal(
            raster_plane_to_array(secondary_sample),
            image[y : y + height, x : x + width] + 1,
        )

    def test_preview_crop_coordinates_are_local_but_persisted_recipe_is_not_changed(
        self,
    ) -> None:
        snapshot = build_processing_preview_snapshot(
            array_to_raster_plane(
                np.arange(100 * 120, dtype=np.uint8).reshape(100, 120)
            ),
            visible_rect=(40.0, 20.0, 50.0, 40.0),
        )
        original = ImageOperationSpec(
            "crop",
            {"x": 50, "y": 30, "width": 25, "height": 20},
        )

        adapted = adapt_operations_for_preview(snapshot, (original,))

        self.assertEqual(
            adapted[0].parameters,
            {"x": 10, "y": 10, "width": 25, "height": 20},
        )
        self.assertEqual(
            original.parameters,
            {"x": 50, "y": 30, "width": 25, "height": 20},
        )

    def test_preview_halo_reads_real_neighbours_and_crops_back_to_sample(
        self,
    ) -> None:
        source_array = np.arange(20 * 20, dtype=np.uint16).reshape(20, 20)
        source = array_to_raster_plane(source_array)
        base = build_processing_preview_snapshot(
            source,
            visible_rect=(5.0, 6.0, 8.0, 7.0),
        )

        expanded, crop = expand_processing_preview_snapshot_for_halo(
            base,
            full_source=source,
            full_roi_mask=None,
            full_secondary_images={},
            halo_x=2,
            halo_y=2,
        )

        self.assertEqual(expanded.bounds, (3, 4, 12, 11))
        self.assertEqual(crop, (2, 2, 8, 7))
        expanded_array = raster_plane_to_array(expanded.source)
        np.testing.assert_array_equal(
            expanded_array[2:9, 2:10],
            raster_plane_to_array(base.source),
        )

    def test_workbench_preview_uses_visible_source_sample_not_full_raster(
        self,
    ) -> None:
        source = array_to_raster_plane(
            np.arange(200 * 300, dtype=np.uint8).reshape(200, 300)
        )
        seen_sizes: list[tuple[int, int]] = []

        def executor(
            request: WorkbenchTaskRequest,
            _token: CancellationToken,
        ) -> RasterPlane:
            seen_sizes.append((request.source.width, request.source.height))
            return request.source

        dialog = ImageProcessingWorkbench(
            source,
            source_document_id="doc-preview",
            preview_rect=(80.0, 50.0, 60.0, 40.0),
            executor=executor,
        )
        try:
            dialog.set_operation_steps((ImageOperationSpec("invert"),))
            dialog.request_preview()
            self._wait_until(lambda: bool(seen_sizes))
            self.assertEqual(seen_sizes[-1], (60, 40))
            self.assertEqual(
                dialog._preview_snapshot.full_source_size,  # noqa: SLF001
                (300, 200),
            )
        finally:
            dialog.close()
            dialog.task_controller.wait_for_done()

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

    def test_parameter_editors_do_not_destroy_the_active_signal_sender(self) -> None:
        rgba = array_to_raster_plane(
            np.zeros((6, 8, 4), dtype=np.uint8)
        )
        secondary = array_to_raster_plane(
            np.ones((6, 8, 4), dtype=np.uint8)
        )
        dialog = ImageProcessingWorkbench(
            rgba,
            source_document_id="doc-1",
            secondary_images={
                "doc-2": rgba,
                "doc-3": secondary,
            },
        )
        try:
            with mock.patch.object(dialog, "_schedule_preview"):
                dialog.set_operation_steps(
                    (
                        default_operation_spec(
                            "convert_color",
                            8,
                            6,
                            source_pixel_type=RasterPixelType.RGBA8,
                        ),
                    )
                )
                checkbox = dialog._parameter_widgets["drop_alpha"]  # noqa: SLF001
                combo = dialog._parameter_widgets["target_model"]  # noqa: SLF001
                self.assertIsInstance(checkbox, QCheckBox)
                self.assertIsInstance(combo, QComboBox)
                for _index in range(101):
                    checkbox.click()
                combo.setCurrentIndex(
                    (combo.currentIndex() + 1) % combo.count()
                )
                self.assertIs(
                    dialog._parameter_widgets["drop_alpha"],  # noqa: SLF001
                    checkbox,
                )
                self.assertIs(
                    dialog._parameter_widgets["target_model"],  # noqa: SLF001
                    combo,
                )
                self.assertEqual(
                    dialog.operation_steps()[0].parameters["drop_alpha"],
                    checkbox.isChecked(),
                )
                self.assertEqual(
                    dialog.operation_steps()[0].parameters["target_model"],
                    combo.currentData(),
                )

                dialog.set_operation_steps(
                    (
                        default_operation_spec(
                            "custom_convolution",
                            8,
                            6,
                            source_pixel_type=RasterPixelType.RGBA8,
                        ),
                    )
                )
                integer = dialog._parameter_widgets["kernel_width"]  # noqa: SLF001
                floating = dialog._parameter_widgets["offset"]  # noqa: SLF001
                number_list = dialog._parameter_widgets["kernel"]  # noqa: SLF001
                self.assertIsInstance(integer, QSpinBox)
                self.assertIsInstance(floating, QDoubleSpinBox)
                self.assertIsInstance(number_list, QLineEdit)
                integer.setValue(5)
                integer.editingFinished.emit()
                floating.setValue(2.5)
                floating.editingFinished.emit()
                number_list.setText(
                    "0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, "
                    "1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0"
                )
                number_list.editingFinished.emit()
                self.assertIs(
                    dialog._parameter_widgets["kernel_width"],  # noqa: SLF001
                    integer,
                )
                self.assertIs(
                    dialog._parameter_widgets["offset"],  # noqa: SLF001
                    floating,
                )
                self.assertIs(
                    dialog._parameter_widgets["kernel"],  # noqa: SLF001
                    number_list,
                )

                dialog.set_operation_steps(
                    (
                        default_operation_spec(
                            "image_calculator",
                            8,
                            6,
                            source_pixel_type=RasterPixelType.RGBA8,
                            secondary_document_id="doc-2",
                        ),
                    )
                )
                secondary_combo = dialog._parameter_widgets[  # noqa: SLF001
                    "secondary_document_id"
                ]
                self.assertIsInstance(secondary_combo, QComboBox)
                secondary_combo.setCurrentIndex(1)
                self.assertIs(
                    dialog._parameter_widgets[  # noqa: SLF001
                        "secondary_document_id"
                    ],
                    secondary_combo,
                )
                self.assertEqual(
                    dialog.operation_steps()[0].parameters[
                        "secondary_document_id"
                    ],
                    "doc-3",
                )
        finally:
            dialog.close()

    def test_resize_canvas_uses_anchor_grid_with_choice_proxy_sync(self) -> None:
        dialog = ImageProcessingWorkbench(
            self.source,
            source_document_id="doc-1",
        )
        try:
            with mock.patch.object(dialog, "_schedule_preview"):
                dialog.set_operation_steps(
                    (
                        default_operation_spec(
                            "resize_canvas",
                            self.source.width,
                            self.source.height,
                        ),
                    )
                )
                editor = dialog._structured_parameter_editors["anchor"]  # noqa: SLF001
                proxy = dialog._parameter_widgets["anchor"]  # noqa: SLF001
                self.assertIsInstance(
                    editor,
                    workbench_module.AnchorGridEditor,
                )
                self.assertIsInstance(proxy, QComboBox)

                editor.buttons["bottom_right"].click()
                self.assertEqual(proxy.currentData(), "bottom_right")
                self.assertEqual(
                    dialog.operation_steps()[0].parameters["anchor"],
                    "bottom_right",
                )

                proxy.setCurrentIndex(proxy.findData("top_left"))
                self.assertEqual(editor.value(), "top_left")
                self.assertEqual(
                    dialog.operation_steps()[0].parameters["anchor"],
                    "top_left",
                )
                self.assertIs(
                    dialog._structured_parameter_editors["anchor"],  # noqa: SLF001
                    editor,
                )
        finally:
            dialog.close()

    def test_custom_convolution_uses_matrix_editor_and_preserves_proxies(
        self,
    ) -> None:
        dialog = ImageProcessingWorkbench(
            self.source,
            source_document_id="doc-1",
        )
        try:
            with mock.patch.object(dialog, "_schedule_preview"):
                dialog.set_operation_steps(
                    (
                        default_operation_spec(
                            "custom_convolution",
                            self.source.width,
                            self.source.height,
                        ),
                    )
                )
                editor = dialog._structured_parameter_editors["kernel"]  # noqa: SLF001
                self.assertIsInstance(
                    editor,
                    workbench_module.KernelMatrixEditor,
                )
                self.assertIs(
                    dialog._parameter_widgets["kernel_width"],  # noqa: SLF001
                    editor.widthSpin,
                )
                self.assertIs(
                    dialog._parameter_widgets["kernel_height"],  # noqa: SLF001
                    editor.heightSpin,
                )
                self.assertIsInstance(
                    dialog._parameter_widgets["kernel"],  # noqa: SLF001
                    QLineEdit,
                )

                editor.applyPreset("sharpen")
                parameters = dialog.operation_steps()[0].parameters
                self.assertEqual(parameters["kernel_width"], 3)
                self.assertEqual(parameters["kernel_height"], 3)
                self.assertEqual(
                    tuple(parameters["kernel"]),
                    (0.0, -1.0, 0.0, -1.0, 5.0, -1.0, 0.0, -1.0, 0.0),
                )

                editor.table.item(0, 0).setText("not-a-number")
                self.assertFalse(dialog._generate_button.isEnabled())  # noqa: SLF001
                self.assertIn(
                    "不是有效数字",
                    dialog._parameter_error_message,  # noqa: SLF001
                )

                editor.table.item(0, 0).setText("1.25")
                self.assertTrue(dialog._generate_button.isEnabled())  # noqa: SLF001
                self.assertEqual(
                    tuple(dialog.operation_steps()[0].parameters["kernel"])[0],
                    1.25,
                )
        finally:
            dialog.close()

    def test_every_catalog_parameter_widget_survives_its_native_signal(self) -> None:
        rgba = array_to_raster_plane(
            np.zeros((6, 8, 4), dtype=np.uint8)
        )
        secondary = array_to_raster_plane(
            np.ones((6, 8, 4), dtype=np.uint8)
        )
        dialog = ImageProcessingWorkbench(
            rgba,
            source_document_id="doc-1",
            secondary_images={
                "doc-2": rgba,
                "doc-3": secondary,
            },
        )
        exercised = 0
        expected = 0

        def _seeded_step(definition) -> ImageOperationSpec:
            base = default_operation_spec(
                definition.operation.value,
                8,
                6,
                source_pixel_type=RasterPixelType.RGBA8,
                secondary_document_id="doc-2",
            )
            return ImageOperationSpec(
                base.operation_id,
                base.parameters,
                implementation=base.implementation,
                implementation_version=base.implementation_version,
                result_metadata={
                    "native_signal_matrix": {
                        "operation": definition.operation.value,
                    }
                },
            )

        def _install(definition) -> tuple[ImageOperationSpec, dict[str, object]]:
            step = _seeded_step(definition)
            # An invalid QLineEdit edit leaves the immutable recipe unchanged
            # while the line edit itself remains invalid.  Rebuild from an
            # empty recipe so every native-signal case starts from the same
            # valid controls and, critically, must reacquire its QWidget.
            dialog.set_operation_steps(())
            dialog.set_operation_steps((step,))
            self.app.processEvents()
            return step, step.parameters

        def _assert_survived(
            *,
            definition,
            field_key: str,
            widget,
            seeded: ImageOperationSpec,
            initial_parameters: dict[str, object],
        ) -> None:
            self.app.processEvents()
            current_widget = dialog._parameter_widgets[field_key]  # noqa: SLF001
            self.assertIs(current_widget, widget)
            # Accessing a C++-deleted wrapper raises RuntimeError.  Keep this
            # explicit: it is the regression that previously terminated Qt
            # after a checkbox/combo emitted its native signal.
            current_widget.isEnabled()
            current = dialog.operation_steps()[0]
            self.assertEqual(
                current.implementation_version,
                seeded.implementation_version,
            )
            self.assertEqual(
                current.result_metadata,
                seeded.result_metadata,
            )
            self.assertTrue(
                set(initial_parameters).issubset(current.parameters),
            )

            # Method switches hide irrelevant controls.  Their values must
            # remain in the recipe instead of being silently discarded.
            current_parameters = current.parameters
            for other in definition.parameters:
                if other.key == field_key:
                    continue
                row_widget = dialog._parameter_row_widgets[other.key]  # noqa: SLF001
                if not dialog._parameter_form.isRowVisible(row_widget):  # noqa: SLF001
                    self.assertIn(other.key, current_parameters)
                    self.assertEqual(
                        current_parameters[other.key],
                        initial_parameters[other.key],
                    )

            error = dialog._parameter_error_message  # noqa: SLF001
            if error:
                self.assertTrue(
                    any("\u3400" <= character <= "\u9fff" for character in error),
                    msg=(
                        f"{definition.operation.value}.{field_key} "
                        f"返回了非中文验证信息：{error}"
                    ),
                )

        try:
            with mock.patch.object(dialog, "_schedule_preview"):
                for definition in workbench_module._OPERATION_CATALOG:  # noqa: SLF001
                    if not definition.available_for_new_recipe:
                        continue
                    for field in definition.parameters:
                        probe_step, _initial = _install(definition)
                        probe_widget = dialog._parameter_widgets[field.key]  # noqa: SLF001
                        if isinstance(probe_widget, QCheckBox):
                            candidates = (False, True)
                        elif isinstance(probe_widget, QSpinBox):
                            candidates = (
                                probe_widget.minimum(),
                                probe_widget.value(),
                                probe_widget.maximum(),
                            )
                        elif isinstance(probe_widget, QDoubleSpinBox):
                            candidates = (
                                probe_widget.minimum(),
                                probe_widget.value(),
                                probe_widget.maximum(),
                            )
                        elif isinstance(probe_widget, QComboBox):
                            candidates = tuple(range(probe_widget.count()))
                        elif isinstance(probe_widget, QLineEdit):
                            tokens = (
                                probe_widget.text()
                                .replace(";", " ")
                                .replace(",", " ")
                                .split()
                            )
                            self.assertTrue(tokens)
                            changed = list(tokens)
                            changed[0] = f"{float(changed[0]) + 0.125:g}"
                            candidates = (
                                ("valid", ", ".join(changed)),
                                ("invalid", "不是数字"),
                            )
                        else:  # pragma: no cover - catalog is exhaustive
                            self.fail(
                                f"未覆盖参数控件：{type(probe_widget).__name__}"
                            )
                        expected += len(candidates)

                        for candidate in candidates:
                            seeded, initial_parameters = _install(definition)
                            # A condition-changing combo may have processed a
                            # queued visibility update in the preceding case.
                            # Never retain the old Python wrapper.
                            widget = dialog._parameter_widgets[field.key]  # noqa: SLF001
                            with self.subTest(
                                operation=definition.operation.value,
                                parameter=field.key,
                                kind=field.kind,
                                candidate=candidate,
                            ):
                                if isinstance(widget, QCheckBox):
                                    value = bool(candidate)
                                    if widget.isChecked() == value:
                                        widget.toggled.emit(value)
                                    else:
                                        widget.setChecked(value)
                                elif isinstance(widget, QSpinBox):
                                    widget.setValue(int(candidate))
                                    widget.editingFinished.emit()
                                elif isinstance(widget, QDoubleSpinBox):
                                    widget.setValue(float(candidate))
                                    widget.editingFinished.emit()
                                elif isinstance(widget, QComboBox):
                                    index = int(candidate)
                                    if widget.currentIndex() == index:
                                        widget.currentIndexChanged.emit(index)
                                    else:
                                        widget.setCurrentIndex(index)
                                elif isinstance(widget, QLineEdit):
                                    _validity, text = candidate
                                    widget.setText(str(text))
                                    widget.editingFinished.emit()
                                else:  # pragma: no cover
                                    self.fail(
                                        "参数控件在重建后改变了类型："
                                        f"{type(widget).__name__}"
                                    )

                                _assert_survived(
                                    definition=definition,
                                    field_key=field.key,
                                    widget=widget,
                                    seeded=seeded,
                                    initial_parameters=initial_parameters,
                                )
                                if (
                                    isinstance(widget, QLineEdit)
                                    and candidate[0] == "invalid"
                                ):
                                    self.assertTrue(
                                        dialog._parameter_error_message  # noqa: SLF001
                                    )
                                    self.assertFalse(
                                        dialog._generate_button.isEnabled()  # noqa: SLF001
                                    )
                                exercised += 1
            self.assertEqual(exercised, expected)
            self.assertGreater(exercised, 500)
        finally:
            dialog.close()

    def test_specialized_parameter_editors_survive_core_native_signals(self) -> None:
        source = array_to_raster_plane(
            np.arange(48, dtype=np.uint8).reshape(6, 8)
        )
        dialog = ImageProcessingWorkbench(
            source,
            source_document_id="doc-1",
        )

        def _install(operation_id: str) -> ImageOperationSpec:
            base = default_operation_spec(operation_id, 8, 6)
            step = ImageOperationSpec(
                base.operation_id,
                base.parameters,
                implementation=base.implementation,
                implementation_version=base.implementation_version,
                result_metadata={
                    "native_signal_matrix": {
                        "operation": operation_id,
                        "specialized": True,
                    }
                },
            )
            dialog.set_operation_steps(())
            dialog.set_operation_steps((step,))
            self.app.processEvents()
            return step

        def _drain_and_assert(
            key: str,
            editor,
            seeded: ImageOperationSpec,
        ) -> None:
            self.app.processEvents()
            current_editor = dialog._structured_parameter_editors[key]  # noqa: SLF001
            self.assertIs(current_editor, editor)
            current_editor.isEnabled()
            current = dialog.operation_steps()[0]
            self.assertEqual(
                current.implementation_version,
                seeded.implementation_version,
            )
            self.assertEqual(
                current.result_metadata.get("native_signal_matrix"),
                seeded.result_metadata["native_signal_matrix"],
            )
            error = dialog._parameter_error_message  # noqa: SLF001
            if error:
                self.assertTrue(
                    any("\u3400" <= character <= "\u9fff" for character in error),
                    msg=f"专用参数编辑器返回了非中文验证信息：{error}",
                )

        try:
            with mock.patch.object(dialog, "_schedule_preview"):
                seeded = _install("threshold")
                histogram = dialog._structured_parameter_editors[  # noqa: SLF001
                    "histogram_range"
                ]
                self.assertIsInstance(
                    histogram,
                    workbench_module.HistogramRangeEditor,
                )
                histogram.autoButton.click()
                _drain_and_assert("histogram_range", histogram, seeded)
                histogram.resetButton.click()
                _drain_and_assert("histogram_range", histogram, seeded)
                for combo in (
                    histogram.displayModeCombo,
                    histogram.polarityCombo,
                ):
                    for index in range(combo.count()):
                        combo.setCurrentIndex(index)
                        _drain_and_assert(
                            "histogram_range",
                            histogram,
                            seeded,
                        )
                histogram.lowerSpin.editingFinished.emit()
                histogram.upperSpin.editingFinished.emit()
                _drain_and_assert("histogram_range", histogram, seeded)

                seeded = _install("percentile_saturation")
                percentile = dialog._structured_parameter_editors[  # noqa: SLF001
                    "percentile_range"
                ]
                self.assertIsInstance(
                    percentile,
                    workbench_module.PercentileRangeEditor,
                )
                percentile.lowerSpin.setValue(99.0)
                percentile.upperSpin.setValue(1.0)
                percentile.lowerSpin.editingFinished.emit()
                percentile.upperSpin.editingFinished.emit()
                _drain_and_assert("percentile_range", percentile, seeded)
                self.assertIn(
                    "下百分位必须小于上百分位",
                    dialog._parameter_error_message,  # noqa: SLF001
                )
                percentile.lowerSpin.setValue(0.5)
                percentile.upperSpin.setValue(99.5)
                percentile.lowerSpin.editingFinished.emit()
                percentile.upperSpin.editingFinished.emit()
                _drain_and_assert("percentile_range", percentile, seeded)

                seeded = _install("brightness_contrast")
                for key, slider in tuple(
                    dialog._structured_parameter_editors.items()  # noqa: SLF001
                ):
                    self.assertTrue(key.startswith("slider:"))
                    self.assertIsInstance(
                        slider,
                        workbench_module.SliderNumberEditor,
                    )
                    slider.slider.setValue(slider.slider.maximum())
                    slider.slider.sliderReleased.emit()
                    _drain_and_assert(key, slider, seeded)
                    slider.spinBox.editingFinished.emit()
                    _drain_and_assert(key, slider, seeded)

                seeded = _install("fft_filter")
                frequency = dialog._structured_parameter_editors[  # noqa: SLF001
                    "frequency_response"
                ]
                self.assertIsInstance(
                    frequency,
                    workbench_module.FrequencyResponseEditor,
                )
                for index in range(frequency.modeCombo.count()):
                    frequency.modeCombo.setCurrentIndex(index)
                    _drain_and_assert(
                        "frequency_response",
                        frequency,
                        seeded,
                    )
                minimum, maximum = frequency.lowCutoffEditor.range()
                frequency.lowCutoffSpin.setValue(maximum)
                frequency.highCutoffSpin.setValue(minimum)
                frequency.lowCutoffSpin.editingFinished.emit()
                frequency.highCutoffSpin.editingFinished.emit()
                _drain_and_assert(
                    "frequency_response",
                    frequency,
                    seeded,
                )
                self.assertTrue(dialog._parameter_error_message)  # noqa: SLF001

                seeded = _install("resize")
                dimensions = dialog._structured_parameter_editors[  # noqa: SLF001
                    "linked_dimensions"
                ]
                self.assertIsInstance(
                    dimensions,
                    workbench_module.LinkedDimensionsEditor,
                )
                dimensions.lockAspectCheck.click()
                _drain_and_assert(
                    "linked_dimensions",
                    dimensions,
                    seeded,
                )
                dimensions.percentSpin.setValue(150.0)
                dimensions.percentSpin.editingFinished.emit()
                _drain_and_assert(
                    "linked_dimensions",
                    dimensions,
                    seeded,
                )
                dimensions.widthSpin.editingFinished.emit()
                dimensions.heightSpin.editingFinished.emit()
                _drain_and_assert(
                    "linked_dimensions",
                    dimensions,
                    seeded,
                )

                seeded = _install("crop")
                crop = dialog._structured_parameter_editors[  # noqa: SLF001
                    "crop_bounds"
                ]
                self.assertIsInstance(
                    crop,
                    workbench_module.CropBoundsEditor,
                )
                crop.xSpin.setValue(crop.xSpin.maximum())
                crop.xSpin.editingFinished.emit()
                _drain_and_assert("crop_bounds", crop, seeded)
                crop.fullImageButton.click()
                _drain_and_assert("crop_bounds", crop, seeded)

                seeded = _install("erode")
                structure = dialog._structured_parameter_editors[  # noqa: SLF001
                    "structuring_element"
                ]
                self.assertIsInstance(
                    structure,
                    workbench_module.StructuringElementEditor,
                )
                for index in range(structure.shapeCombo.count()):
                    structure.shapeCombo.setCurrentIndex(index)
                    structure.shapeCombo.activated.emit(index)
                    _drain_and_assert(
                        "structuring_element",
                        structure,
                        seeded,
                    )
                structure.radiusSpin.editingFinished.emit()
                structure.iterationsSpin.editingFinished.emit()
                _drain_and_assert(
                    "structuring_element",
                    structure,
                    seeded,
                )

                seeded = _install("resize_canvas")
                anchor = dialog._structured_parameter_editors["anchor"]  # noqa: SLF001
                self.assertIsInstance(
                    anchor,
                    workbench_module.AnchorGridEditor,
                )
                for button in anchor.buttons.values():
                    button.click()
                    _drain_and_assert("anchor", anchor, seeded)

                seeded = _install("custom_convolution")
                kernel = dialog._structured_parameter_editors["kernel"]  # noqa: SLF001
                self.assertIsInstance(
                    kernel,
                    workbench_module.KernelMatrixEditor,
                )
                for preset in kernel.PRESETS:
                    kernel.applyPreset(preset)
                    _drain_and_assert("kernel", kernel, seeded)
                kernel.table.item(0, 0).setText("非法数值")
                _drain_and_assert("kernel", kernel, seeded)
                self.assertTrue(dialog._parameter_error_message)  # noqa: SLF001
                kernel.table.item(0, 0).setText("1.25")
                _drain_and_assert("kernel", kernel, seeded)

                seeded = _install("stripe_suppression")
                stripe = dialog._structured_parameter_editors[  # noqa: SLF001
                    "stripe_frequency"
                ]
                self.assertIsInstance(
                    stripe,
                    workbench_module.StripeSuppressionEditor,
                )
                for index in range(stripe.directionCombo.count()):
                    stripe.directionCombo.setCurrentIndex(index)
                    _drain_and_assert(
                        "stripe_frequency",
                        stripe,
                        seeded,
                    )
                stripe.notchWidthSpin.editingFinished.emit()
                stripe.protectRadiusSpin.editingFinished.emit()
                _drain_and_assert(
                    "stripe_frequency",
                    stripe,
                    seeded,
                )
        finally:
            dialog.close()

    def test_preview_fits_window_supports_zoom_and_uses_processed_overview(
        self,
    ) -> None:
        source = array_to_raster_plane(
            np.zeros((562, 750), dtype=np.uint8)
        )
        processed = array_to_raster_plane(
            np.full((562, 750), 255, dtype=np.uint8)
        )
        dialog = ImageProcessingWorkbench(
            source,
            source_document_id="doc-1",
        )
        try:
            dialog._preview_view.resize(400, 300)  # noqa: SLF001
            dialog._preview_view.fit_image()  # noqa: SLF001
            self.assertTrue(dialog._preview_view.fit_mode)  # noqa: SLF001
            self.assertEqual(
                dialog._preview_view.image_size(),  # noqa: SLF001
                (750, 562),
            )
            self.assertLess(
                dialog._preview_view.zoom_factor(),  # noqa: SLF001
                1.0,
            )

            dialog._preview_view.actual_size()  # noqa: SLF001
            self.assertFalse(dialog._preview_view.fit_mode)  # noqa: SLF001
            self.assertAlmostEqual(
                dialog._preview_view.zoom_factor(),  # noqa: SLF001
                1.0,
            )
            dialog._preview_view.zoom_by(  # noqa: SLF001
                dialog._preview_view.ZOOM_STEP  # noqa: SLF001
            )
            self.assertGreater(
                dialog._preview_view.zoom_factor(),  # noqa: SLF001
                1.0,
            )

            with mock.patch.object(dialog, "_schedule_preview"):
                dialog.set_operation_steps(
                    (
                        default_operation_spec(
                            "threshold",
                            750,
                            562,
                            source_pixel_type=RasterPixelType.GRAY8,
                        ),
                    )
                )
            dialog._show_preview_raster(processed)  # noqa: SLF001
            dialog._overview_checkbox.setChecked(True)  # noqa: SLF001
            self.app.processEvents()
            overview = dialog._processed_overview_image  # noqa: SLF001
            self.assertEqual(
                overview.pixelColor(
                    overview.width() // 2,
                    overview.height() // 2,
                ).red(),
                255,
            )
            self.assertIn(
                "处理后的完整图片概览",
                dialog._overview_note.text(),  # noqa: SLF001
            )
            self.assertFalse(dialog._overview_note.isHidden())  # noqa: SLF001
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
        self.assertEqual(resize.parameters["interpolation"], "auto")

        levels = default_operation_spec(
            "adjust_levels",
            640,
            480,
            source_pixel_type=RasterPixelType.GRAY16,
        )
        self.assertEqual(levels.parameters["white_point"], 65_535.0)
        self.assertEqual(
            resize.implementation_version,
            IMAGE_OPERATION_REGISTRY["resize"].version,
        )

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

    def test_loaded_steps_reject_unknown_engine_and_algorithm_version(
        self,
    ) -> None:
        dialog = ImageProcessingWorkbench(
            self.source,
            source_document_id="doc-1",
        )
        try:
            with self.assertRaisesRegex(ValueError, "不支持的图像处理实现"):
                dialog.set_operation_steps(
                    (
                        ImageOperationSpec(
                            "invert",
                            implementation="other-engine",
                        ),
                    )
                )
            with self.assertRaisesRegex(ValueError, "不支持的算法版本.*v99"):
                dialog.set_operation_steps(
                    (
                        ImageOperationSpec(
                            "fill_holes",
                            implementation_version="99",
                        ),
                    )
                )
        finally:
            dialog.close()

    def test_new_step_defaults_follow_the_preceding_recipe_output_state(self) -> None:
        dialog = ImageProcessingWorkbench(
            self.source,
            source_document_id="doc-1",
        )
        try:
            dialog.set_operation_steps(
                (
                    ImageOperationSpec(
                        "convert_type",
                        {
                            "target_type": "uint16",
                            "scale_mode": "full_type_range",
                            "nonfinite_policy": "reject",
                        },
                    ),
                )
            )
            threshold = dialog.make_default_operation_spec("threshold")
            self.assertEqual(threshold.parameters["upper"], 65_535.0)

            dialog.set_operation_steps(
                (
                    ImageOperationSpec(
                        "crop",
                        {
                            "x": 1,
                            "y": 1,
                            "width": 4,
                            "height": 3,
                            "roi_mode": "bounds",
                            "fill_value": 0.0,
                            "transparent_outside": False,
                        },
                    ),
                )
            )
            resize = dialog.make_default_operation_spec("resize")
            self.assertEqual(resize.parameters["width"], 4)
            self.assertEqual(resize.parameters["height"], 3)
        finally:
            dialog.close()

    def test_missing_loaded_step_defaults_follow_each_step_input_state(self) -> None:
        dialog = ImageProcessingWorkbench(
            self.source,
            source_document_id="doc-1",
        )
        try:
            dialog.set_operation_steps(
                (
                    ImageOperationSpec(
                        "convert_type",
                        {
                            "target_type": "uint16",
                            "scale_mode": "full_type_range",
                            "nonfinite_policy": "reject",
                        },
                    ),
                    ImageOperationSpec("threshold"),
                )
            )
            self.assertEqual(
                dialog.operation_steps()[1].parameters["upper"],
                65_535.0,
            )
        finally:
            dialog.close()

    def test_irrelevant_method_parameters_are_hidden_without_losing_values(self) -> None:
        dialog = ImageProcessingWorkbench(
            self.source,
            source_document_id="doc-1",
        )
        try:
            dialog.set_operation_steps(
                (default_operation_spec("adaptive_threshold", 8, 6),)
            )
            method = dialog._parameter_widgets["method"]  # noqa: SLF001
            self.assertIsInstance(method, QComboBox)
            method.setCurrentIndex(method.findData("mean"))
            self.app.processEvents()
            for key in ("k", "r", "p", "q"):
                widget = dialog._parameter_row_widgets[key]  # noqa: SLF001
                self.assertFalse(
                    dialog._parameter_form.isRowVisible(widget)  # noqa: SLF001
                )

            method.setCurrentIndex(method.findData("sauvola"))
            self.app.processEvents()
            self.assertTrue(
                dialog._parameter_form.isRowVisible(  # noqa: SLF001
                    dialog._parameter_row_widgets["k"]  # noqa: SLF001
                )
            )
            self.assertTrue(
                dialog._parameter_form.isRowVisible(  # noqa: SLF001
                    dialog._parameter_row_widgets["r"]  # noqa: SLF001
                )
            )
            self.assertFalse(
                dialog._parameter_form.isRowVisible(  # noqa: SLF001
                    dialog._parameter_row_widgets["p"]  # noqa: SLF001
                )
            )
        finally:
            dialog.close()

    def test_histogram_range_editor_prevents_reversed_thresholds(self) -> None:
        dialog = ImageProcessingWorkbench(
            self.source,
            source_document_id="doc-1",
        )
        try:
            dialog.set_operation_steps(
                (default_operation_spec("threshold", 8, 6),)
            )
            lower = dialog._parameter_widgets["lower"]  # noqa: SLF001
            upper = dialog._parameter_widgets["upper"]  # noqa: SLF001
            self.assertIsInstance(lower, QDoubleSpinBox)
            self.assertIsInstance(upper, QDoubleSpinBox)
            lower.setValue(300.0)
            lower.editingFinished.emit()
            self.assertLessEqual(lower.value(), upper.value())
            self.assertTrue(dialog._generate_button.isEnabled())  # noqa: SLF001
            self.assertLessEqual(
                dialog.operation_steps()[0].parameters["lower"],
                dialog.operation_steps()[0].parameters["upper"],
            )

            upper.setValue(200.0)
            upper.editingFinished.emit()
            self.assertTrue(dialog._generate_button.isEnabled())  # noqa: SLF001
            self.assertLessEqual(
                dialog.operation_steps()[0].parameters["lower"],
                dialog.operation_steps()[0].parameters["upper"],
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
        self.assertEqual(
            schemas["adaptive_threshold"],
            {
                "method",
                "radius",
                "offset",
                "k",
                "r",
                "p",
                "q",
                "foreground_is_high",
                "channel",
            },
        )
        self.assertEqual(
            schemas["log_v2"],
            {"result_mode", "output_min", "output_max"},
        )
        labels = {
            definition.operation.value: definition.label
            for definition in definitions
        }
        self.assertEqual(
            labels["background_subtract"],
            "形态学背景扣除",
        )
        self.assertEqual(
            labels["rolling_ball_background_subtract"],
            "滑动抛物面背景扣除",
        )

        log_v2 = default_operation_spec(
            "log_v2",
            640,
            480,
            source_pixel_type=RasterPixelType.GRAY16,
        )
        self.assertEqual(log_v2.parameters["result_mode"], "float32")
        self.assertEqual(log_v2.implementation_version, "2")

    def test_workbench_parameter_constraints_come_from_descriptors(self) -> None:
        from fdm.services.image_processing import get_image_operation_descriptor

        for definition in workbench_module._OPERATION_CATALOG:  # noqa: SLF001
            descriptor = get_image_operation_descriptor(definition.operation)
            with self.subTest(operation=definition.operation.value):
                for field in definition.parameters:
                    schema = descriptor.parameter(field.key)
                    self.assertEqual(field.kind, schema.kind)
                    self.assertEqual(field.default, schema.default)
                    self.assertEqual(field.minimum, schema.minimum)
                    self.assertEqual(field.maximum, schema.maximum)
                    self.assertEqual(
                        tuple(value for _label, value in field.choices),
                        schema.choices,
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

    def test_secondary_semantic_controls_strict_binary_recipe_chain(
        self,
    ) -> None:
        secondary = RasterPlane(
            width=self.source.width,
            height=self.source.height,
            pixel_type=self.source.pixel_type,
            data=self.source.data,
        )
        operations = (
            ImageOperationSpec(
                "image_calculator",
                {
                    "secondary_document_id": "binary-right",
                    "calculator_operation": "copy",
                    "result_mode": "preserve",
                },
                implementation_version="2",
            ),
            ImageOperationSpec(
                "fill_holes",
                implementation_version="2",
            ),
        )

        with self.assertRaisesRegex(ValueError, "显式二值掩膜"):
            validate_workbench_operation_sequence(
                self.source,
                operations,
                source_semantic=RasterSemantic.INTENSITY,
                secondary_images={"binary-right": secondary},
            )

        validation = validate_workbench_operation_sequence(
            self.source,
            operations,
            source_semantic=RasterSemantic.INTENSITY,
            secondary_images={"binary-right": secondary},
            secondary_semantics={
                "binary-right": RasterSemantic.BINARY_MASK,
            },
        )
        self.assertIs(
            validation.output_state.semantic,
            RasterSemantic.BINARY_MASK,
        )
        request = WorkbenchTaskRequest(
            kind=WorkbenchTaskKind.FINAL,
            request_id="secondary-semantic",
            generation=1,
            source_document_id="doc-1",
            source=self.source,
            operations=operations,
            source_semantic=RasterSemantic.INTENSITY,
            secondary_images=(("binary-right", secondary),),
            secondary_semantics=(
                ("binary-right", RasterSemantic.BINARY_MASK),
            ),
        )
        self.assertEqual(
            dict(request.secondary_semantics),
            {"binary-right": RasterSemantic.BINARY_MASK},
        )

    def test_reference_flat_field_freezes_reference_sha_and_full_levels(
        self,
    ) -> None:
        source_values = np.arange(48, dtype=np.uint8).reshape(6, 8) + 20
        reference_values = (
            np.arange(48, dtype=np.uint8).reshape(6, 8) * 3 + 10
        )
        source = array_to_raster_plane(source_values)
        reference = array_to_raster_plane(reference_values)
        dialog = ImageProcessingWorkbench(
            source,
            source_document_id="doc-1",
            secondary_images={"flat-reference": reference},
            secondary_image_names={"flat-reference": "白场参考"},
        )
        operation = ImageOperationSpec(
            ImageOperation.FLAT_FIELD_CORRECTION.value,
            {
                "flat_field_source": "reference",
                "secondary_document_id": "flat-reference",
                "radius": 25.0,
                "method": "gaussian",
                "preserve_mean": True,
            },
        )
        from fdm.cancellation import CancellationTokenSource

        try:
            dialog.set_operation_steps((operation,))
            prepared = dialog._prepare_reference_flat_field_steps(  # noqa: SLF001
                dialog.operation_steps()
            )
            parameters = prepared[0].parameters
            self.assertEqual(parameters["secondary_sha256"], reference.sha256())
            self.assertEqual(
                parameters["reference_levels"],
                [float(np.mean(reference_values, dtype=np.float64))],
            )

            final_execution = (
                workbench_module._execute_workbench_request_with_metadata(  # noqa: SLF001
                    WorkbenchTaskRequest(
                        kind=WorkbenchTaskKind.FINAL,
                        request_id="flat-final",
                        generation=1,
                        source_document_id="doc-1",
                        source=source,
                        operations=prepared,
                        secondary_images=(("flat-reference", reference),),
                    ),
                    CancellationTokenSource().token,
                )
            )
            final_parameters = (
                final_execution.recipe.operations[0].parameters
            )
            self.assertEqual(
                final_parameters["secondary_sha256"],
                reference.sha256(),
            )
            self.assertEqual(
                final_parameters["secondary_document_id"],
                "flat-reference",
            )

            snapshot = build_processing_preview_snapshot(
                source,
                visible_rect=(2.0, 1.0, 3.0, 3.0),
                secondary_images={"flat-reference": reference},
            )
            preview_execution = (
                workbench_module._execute_workbench_request_with_metadata(  # noqa: SLF001
                    WorkbenchTaskRequest(
                        kind=WorkbenchTaskKind.PREVIEW,
                        request_id="flat-preview",
                        generation=1,
                        source_document_id="doc-1",
                        source=snapshot.source,
                        operations=prepared,
                        secondary_images=snapshot.secondary_images,
                    ),
                    CancellationTokenSource().token,
                )
            )
            final_array = raster_plane_to_array(final_execution.raster)
            preview_array = raster_plane_to_array(preview_execution.raster)
            np.testing.assert_array_equal(
                preview_array,
                final_array[1:4, 2:5],
            )

            stale_parameters = operation.parameters
            stale_parameters["secondary_sha256"] = "c" * 64
            with self.assertRaisesRegex(ValueError, "摘要"):
                dialog.set_operation_steps(
                    (
                        ImageOperationSpec(
                            ImageOperation.FLAT_FIELD_CORRECTION.value,
                            stale_parameters,
                        ),
                    )
                )
        finally:
            dialog.close()

    def test_estimated_flat_field_remains_available_without_reference(self) -> None:
        operation = default_operation_spec(
            ImageOperation.FLAT_FIELD_CORRECTION,
            self.source.width,
            self.source.height,
        )

        self.assertEqual(
            operation.parameters["flat_field_source"],
            "estimated",
        )
        validation = validate_workbench_operation_sequence(
            self.source,
            (operation,),
        )
        self.assertEqual(
            validation.steps[0].operation.operation_id,
            ImageOperation.FLAT_FIELD_CORRECTION.value,
        )

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
