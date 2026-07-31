from __future__ import annotations

import os
from pathlib import Path
import sys
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    import numpy as np
    from PySide6.QtWidgets import QApplication, QComboBox, QDoubleSpinBox

    from fdm.image_processing_models import DisplayTransform
    from fdm.raster import RasterPixelType, RasterPlane
    from fdm.ui.display_adjustment_dialog import (
        DisplayAdjustmentAction,
        DisplayAdjustmentDialog,
        DisplayAdjustmentResult,
    )

    PYSIDE_AVAILABLE = True
except ModuleNotFoundError:
    PYSIDE_AVAILABLE = False


class _FakeWheelEvent:
    def __init__(self) -> None:
        self.accepted = False
        self.ignored = False

    def accept(self) -> None:
        self.accepted = True

    def ignore(self) -> None:
        self.ignored = True


@unittest.skipUnless(PYSIDE_AVAILABLE, "PySide6 is not installed")
class DisplayAdjustmentDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    @staticmethod
    def _gray16() -> RasterPlane:
        values = np.asarray(
            [[0, 1000, 2000], [3000, 4000, 65535]],
            dtype="<u2",
        )
        return RasterPlane(
            width=3,
            height=2,
            pixel_type=RasterPixelType.GRAY16,
            data=values.tobytes(),
        )

    @staticmethod
    def _rgb() -> RasterPlane:
        values = np.arange(36, dtype=np.uint8).reshape(3, 4, 3)
        return RasterPlane(
            width=4,
            height=3,
            pixel_type=RasterPixelType.RGB8,
            data=values.tobytes(),
        )

    def test_dialog_is_non_modal_scrollable_chinese_and_does_not_mutate_source(self) -> None:
        plane = self._gray16()
        original_bytes = plane.data
        dialog = DisplayAdjustmentDialog(
            plane,
            DisplayTransform(),
            source_name="高位深显微图",
        )
        previews: list[DisplayTransform] = []
        dialog.previewTransformChanged.connect(previews.append)
        try:
            self.assertFalse(dialog.isModal())
            self.assertEqual(dialog.windowTitle(), "亮度、对比度与显示范围")
            self.assertIn("高位深显微图", dialog.sourceLabel.text())
            self.assertIn("16 位灰度", dialog.sourceLabel.text())
            self.assertEqual(dialog.minimumWidth(), 520)
            self.assertEqual(dialog.minimumHeight(), 420)

            dialog.channelLowSpins[0].setValue(100.0)
            dialog.channelHighSpins[0].setValue(5000.0)
            dialog.gammaSpin.setValue(1.4)
            self.app.processEvents()

            self.assertTrue(previews)
            self.assertIsInstance(previews[-1], DisplayTransform)
            self.assertEqual(
                previews[-1].channel_ranges,
                ((100.0, 5000.0),),
            )
            self.assertAlmostEqual(previews[-1].gamma, 1.4)
            self.assertEqual(plane.data, original_bytes)
            self.assertEqual(plane.sha256(), dialog.source.sha256())
        finally:
            dialog.close()

    def test_gray_window_level_lut_and_invert_are_valid_preview_settings(self) -> None:
        initial = DisplayTransform(window_center=2000.0, window_width=1200.0)
        dialog = DisplayAdjustmentDialog(self._gray16(), initial)
        previews: list[DisplayTransform] = []
        dialog.previewTransformChanged.connect(previews.append)
        try:
            self.assertEqual(dialog.rangeModeCombo.currentData(), "window")
            self.assertAlmostEqual(dialog.windowCenterSpin.value(), 2000.0)
            self.assertAlmostEqual(dialog.windowWidthSpin.value(), 1200.0)
            dialog.windowCenterSpin.setValue(2500.0)
            dialog.windowWidthSpin.setValue(800.0)
            dialog.invertCheck.setChecked(True)
            self.app.processEvents()

            transform = previews[-1]
            self.assertEqual(transform.window_center, 2500.0)
            self.assertEqual(transform.window_width, 800.0)
            self.assertTrue(transform.inverted)
            plan = dialog.bake_plan()
            self.assertTrue(plan.supported)
            self.assertEqual(
                [operation.operation_id for operation in plan.operations],
                ["adjust_levels", "invert"],
            )

            red_index = dialog.lutCombo.findData("red")
            dialog.lutCombo.setCurrentIndex(red_index)
            self.app.processEvents()
            self.assertFalse(dialog.bake_plan().supported)
            self.assertFalse(dialog.generateDerivedButton.isEnabled())
            self.assertIn("彩色 LUT", dialog.bakeHintLabel.text())
        finally:
            dialog.close()

    def test_rgb_independent_ranges_preview_but_cannot_be_silently_baked(self) -> None:
        dialog = DisplayAdjustmentDialog(self._rgb(), DisplayTransform())
        try:
            self.assertEqual(len(dialog.channelLowSpins), 3)
            self.assertFalse(dialog.lutCombo.isEnabled())
            dialog.channelLowSpins[0].setValue(10.0)
            dialog.channelHighSpins[0].setValue(210.0)
            dialog.channelLowSpins[1].setValue(20.0)
            dialog.channelHighSpins[1].setValue(220.0)
            dialog.channelLowSpins[2].setValue(30.0)
            dialog.channelHighSpins[2].setValue(230.0)
            self.app.processEvents()

            transform = dialog.current_transform()
            self.assertEqual(
                transform.channel_ranges,
                ((10.0, 210.0), (20.0, 220.0), (30.0, 230.0)),
            )
            plan = dialog.bake_plan()
            self.assertFalse(plan.supported)
            self.assertIn("独立通道", plan.message)
            self.assertFalse(dialog.generateDerivedButton.isEnabled())

            for low_spin, high_spin in zip(
                dialog.channelLowSpins,
                dialog.channelHighSpins,
                strict=True,
            ):
                low_spin.setValue(10.0)
                high_spin.setValue(210.0)
            self.app.processEvents()
            self.assertTrue(dialog.bake_plan().supported)
            self.assertTrue(dialog.generateDerivedButton.isEnabled())
        finally:
            dialog.close()

    def test_apply_and_derived_actions_emit_structured_results(self) -> None:
        apply_dialog = DisplayAdjustmentDialog(self._gray16(), DisplayTransform())
        applied: list[DisplayAdjustmentResult] = []
        all_results: list[DisplayAdjustmentResult] = []
        apply_dialog.displaySettingsApplied.connect(applied.append)
        apply_dialog.resultReady.connect(all_results.append)
        apply_dialog.channelLowSpins[0].setValue(200.0)
        apply_dialog.channelHighSpins[0].setValue(6000.0)
        apply_dialog._apply_display()  # noqa: SLF001
        self.assertEqual(len(applied), 1)
        self.assertEqual(applied[0].action, DisplayAdjustmentAction.APPLY_DISPLAY)
        self.assertEqual(applied[0].bake_operations, ())
        self.assertEqual(applied[0].source_sha256, self._gray16().sha256())
        self.assertEqual(all_results, applied)

        derived_dialog = DisplayAdjustmentDialog(self._gray16(), DisplayTransform())
        derived: list[DisplayAdjustmentResult] = []
        derived_dialog.derivedImageRequested.connect(derived.append)
        derived_dialog.channelLowSpins[0].setValue(100.0)
        derived_dialog.channelHighSpins[0].setValue(5000.0)
        derived_dialog.gammaSpin.setValue(1.2)
        derived_dialog.invertCheck.setChecked(True)
        derived_dialog._request_derived()  # noqa: SLF001
        self.assertEqual(len(derived), 1)
        self.assertEqual(
            derived[0].action,
            DisplayAdjustmentAction.GENERATE_DERIVED,
        )
        self.assertEqual(
            [operation.operation_id for operation in derived[0].bake_operations],
            ["adjust_levels", "invert"],
        )
        self.assertEqual(
            derived[0].bake_operations[0].parameters,
            {
                "black_point": 100.0,
                "gamma": 1.2,
                "white_point": 5000.0,
            },
        )

    def test_cancel_restores_exact_initial_transform_once(self) -> None:
        initial = DisplayTransform(
            channel_ranges=((10.0, 4000.0),),
            gamma=0.8,
            inverted=True,
        )
        dialog = DisplayAdjustmentDialog(self._gray16(), initial)
        previews: list[DisplayTransform] = []
        cancelled: list[DisplayAdjustmentResult] = []
        dialog.previewTransformChanged.connect(previews.append)
        dialog.adjustmentCancelled.connect(cancelled.append)
        dialog.channelLowSpins[0].setValue(100.0)
        dialog.reject()
        dialog.reject()

        self.assertEqual(previews[-1], initial)
        self.assertEqual(len(cancelled), 1)
        self.assertEqual(cancelled[0].action, DisplayAdjustmentAction.CANCEL)
        self.assertEqual(cancelled[0].transform, initial)

    def test_constant_float_uses_finite_edit_range_and_bake_plan(self) -> None:
        values = np.asarray(
            [[np.nan, 5.0], [np.inf, 5.0]],
            dtype="<f4",
        )
        plane = RasterPlane(
            width=2,
            height=2,
            pixel_type=RasterPixelType.GRAY32_FLOAT,
            data=values.tobytes(),
        )
        dialog = DisplayAdjustmentDialog(plane, DisplayTransform())
        try:
            low = dialog.channelLowSpins[0].value()
            high = dialog.channelHighSpins[0].value()
            self.assertLess(low, 5.0)
            self.assertGreater(high, 5.0)
            dialog.gammaSpin.setValue(1.5)
            self.app.processEvents()
            plan = dialog.bake_plan()
            self.assertTrue(plan.supported)
            self.assertEqual(plan.operations[0].operation_id, "adjust_levels")
            self.assertTrue(
                all(
                    np.isfinite(value)
                    for value in plan.operations[0].parameters.values()
                )
            )
        finally:
            dialog.close()

    def test_all_combo_and_numeric_controls_ignore_incidental_wheel(self) -> None:
        dialog = DisplayAdjustmentDialog(self._gray16(), DisplayTransform())
        try:
            editors = [
                *dialog.findChildren(QComboBox),
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
                    self.assertEqual(before, after)
                    self.assertTrue(event.accepted or event.ignored)
        finally:
            dialog.close()

    def test_histogram_auto_uses_frozen_roi_and_reset_restores_native_range(
        self,
    ) -> None:
        values = np.asarray(
            [[0, 1000, 2000], [3000, 4000, 65535]],
            dtype="<u2",
        )
        mask = np.asarray(
            [[False, True, True], [True, True, False]],
            dtype=np.bool_,
        )
        dialog = DisplayAdjustmentDialog(
            self._gray16(),
            DisplayTransform(),
            roi_mask=mask,
        )
        try:
            self.assertIn("当前 ROI", dialog.histogramScopeLabel.text())
            self.assertIn("N=4", dialog.histogramScopeLabel.text())
            dialog._apply_auto_range()  # noqa: SLF001
            self.assertGreater(dialog.channelLowSpins[0].value(), 0.0)
            self.assertLess(dialog.channelHighSpins[0].value(), 65535.0)
            dialog._reset_controls()  # noqa: SLF001
            self.assertEqual(dialog.channelLowSpins[0].value(), 0.0)
            self.assertEqual(dialog.channelHighSpins[0].value(), 65535.0)
            self.assertEqual(dialog.gammaSpin.value(), 1.0)
        finally:
            dialog.close()

    def test_roi_mask_shape_is_validated(self) -> None:
        with self.assertRaisesRegex(ValueError, "ROI 掩膜尺寸"):
            DisplayAdjustmentDialog(
                self._gray16(),
                DisplayTransform(),
                roi_mask=np.ones((1, 1), dtype=np.bool_),
            )


if __name__ == "__main__":
    unittest.main()
