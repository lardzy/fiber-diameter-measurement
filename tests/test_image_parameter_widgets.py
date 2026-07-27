from __future__ import annotations

import math
import os
from pathlib import Path
import sys
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    from PySide6.QtCore import QPoint, QPointF, Qt
    from PySide6.QtGui import QColor, QPalette, QWheelEvent
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication

    from fdm.ui.image_parameter_widgets import (
        AnchorGridEditor,
        CropBoundsEditor,
        FrequencyResponseEditor,
        HistogramRangeEditor,
        KernelMatrixEditor,
        LinkedDimensionsEditor,
        PercentileRangeEditor,
        SliderNumberEditor,
        StripeSuppressionEditor,
        StructuringElementEditor,
    )
    from fdm.ui.widgets import (
        NoWheelComboBox,
        NoWheelDoubleSpinBox,
        NoWheelSpinBox,
    )

    PYSIDE_AVAILABLE = True
except ModuleNotFoundError:
    PYSIDE_AVAILABLE = False


@unittest.skipUnless(PYSIDE_AVAILABLE, "PySide6 is not installed")
class ImageParameterWidgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_histogram_editor_supports_exact_range_and_single_threshold(self) -> None:
        editor = HistogramRangeEditor(
            minimum=-10.0,
            maximum=30.0,
            lower=-2.5,
            upper=12.75,
            decimals=6,
            suffix=" HU",
        )
        changes: list[tuple[float, float]] = []
        editor.thresholdsChanged.connect(
            lambda lower, upper: changes.append((lower, upper))
        )
        try:
            self.assertIsInstance(editor.lowerSpin, NoWheelDoubleSpinBox)
            self.assertIsInstance(editor.upperSpin, NoWheelDoubleSpinBox)
            self.assertEqual(editor.range(), (-10.0, 30.0))
            self.assertEqual(editor.thresholds(), (-2.5, 12.75))
            self.assertEqual(editor.lowerSpin.suffix(), " HU")

            editor.lowerSpin.setValue(7.125)
            self.assertEqual(editor.thresholds(), (7.125, 12.75))
            self.assertEqual(changes[-1], (7.125, 12.75))

            editor.lowerSpin.setValue(20.0)
            self.assertEqual(editor.thresholds(), (12.75, 12.75))
            self.assertEqual(editor.lowerSpin.value(), 12.75)

            editor.setSingleThreshold(True)
            editor.setThreshold(8.5)
            self.assertTrue(editor.isSingleThreshold())
            self.assertFalse(editor.upperSpin.isVisible())
            self.assertEqual(editor.thresholds(), (8.5, 30.0))
        finally:
            editor.close()

    def test_histogram_editor_rejects_non_finite_or_inverted_values(self) -> None:
        editor = HistogramRangeEditor()
        try:
            with self.assertRaisesRegex(ValueError, "有限数"):
                editor.setRange(float("nan"), 1.0)
            with self.assertRaisesRegex(ValueError, "上限必须大于"):
                editor.setRange(1.0, 1.0)
            with self.assertRaisesRegex(ValueError, "不能大于"):
                editor.setThresholds(200.0, 100.0)
            with self.assertRaisesRegex(ValueError, "不能为负数"):
                editor.setHistogram((1.0, -1.0))
            with self.assertRaisesRegex(ValueError, "有限数"):
                editor.setHistogram((1.0, float("inf")))
        finally:
            editor.close()

    def test_histogram_data_refresh_does_not_emit_a_parameter_edit(self) -> None:
        editor = HistogramRangeEditor(
            minimum=0.0,
            maximum=255.0,
            lower=200.0,
            upper=240.0,
        )
        threshold_changes: list[tuple[float, float]] = []
        range_changes: list[tuple[float, float]] = []
        editor.thresholdsChanged.connect(
            lambda lower, upper: threshold_changes.append((lower, upper))
        )
        editor.rangeChanged.connect(
            lambda lower, upper: range_changes.append((lower, upper))
        )
        try:
            editor.setHistogram(
                (1.0, 2.0, 1.0),
                value_range=(0.0, 100.0),
            )
            self.assertEqual(editor.thresholds(), (100.0, 100.0))
            self.assertEqual(threshold_changes, [])
            self.assertEqual(range_changes, [])

            editor.setHistogram(
                (1.0, 2.0, 1.0),
                value_range=(0.0, 120.0),
                emit_range_signal=True,
            )
            self.assertEqual(range_changes, [(0.0, 120.0)])
        finally:
            editor.close()

    def test_histogram_handle_drag_updates_threshold_without_crossing(self) -> None:
        editor = HistogramRangeEditor(
            minimum=0.0,
            maximum=100.0,
            lower=20.0,
            upper=80.0,
        )
        editor.resize(420, 190)
        editor.show()
        self.app.processEvents()
        canvas = editor.histogramCanvas
        changes: list[tuple[float, float]] = []
        editor.thresholdsChanged.connect(
            lambda lower, upper: changes.append((lower, upper))
        )
        try:
            plot = canvas._plot_rect()  # noqa: SLF001
            target = QPoint(
                int(round(plot.left() + plot.width() * 0.35)),
                int(round(plot.center().y())),
            )
            QTest.mouseClick(
                canvas,
                Qt.MouseButton.LeftButton,
                pos=target,
            )
            self.assertTrue(changes)
            lower, upper = editor.thresholds()
            self.assertAlmostEqual(lower, 35.0, delta=1.0)
            self.assertLessEqual(lower, upper)
        finally:
            editor.close()

    def test_histogram_editor_exposes_imagej_style_controls_and_statistics(
        self,
    ) -> None:
        editor = HistogramRangeEditor()
        display_changes: list[str] = []
        polarity_changes: list[str] = []
        auto_requests: list[bool] = []
        reset_requests: list[bool] = []
        finished: list[bool] = []
        editor.displayModeChanged.connect(display_changes.append)
        editor.foregroundPolarityChanged.connect(polarity_changes.append)
        editor.autoRequested.connect(lambda: auto_requests.append(True))
        editor.resetRequested.connect(lambda: reset_requests.append(True))
        editor.interactionFinished.connect(lambda: finished.append(True))
        try:
            self.assertEqual(editor.displayMode(), "bw")
            self.assertEqual(editor.foregroundPolarity(), "bright")

            editor.setDisplayMode("red_overlay")
            editor.setForegroundPolarity("dark")
            self.assertEqual(editor.displayMode(), "red_overlay")
            self.assertEqual(editor.foregroundPolarity(), "dark")
            self.assertEqual(display_changes, ["red_overlay"])
            self.assertEqual(polarity_changes, ["dark"])

            editor.setSelectionStatistics(2_500, 10_000)
            self.assertEqual(editor.selectionStatistics(), (2_500, 10_000))
            self.assertIn("25.00%", editor.selectionStatisticsLabel.text())
            self.assertIn("2,500", editor.selectionStatisticsLabel.text())
            editor.clearSelectionStatistics()
            self.assertIsNone(editor.selectionStatistics())
            self.assertEqual(editor.selectionStatisticsLabel.text(), "选中像素：—")

            editor.requestAuto()
            editor.requestReset()
            self.assertEqual(auto_requests, [True])
            self.assertEqual(reset_requests, [True])
            self.assertEqual(len(finished), 2)

            with self.assertRaisesRegex(ValueError, "显示模式"):
                editor.setDisplayMode("unknown")
            with self.assertRaisesRegex(ValueError, "前景极性"):
                editor.setForegroundPolarity("unknown")
            with self.assertRaisesRegex(ValueError, "不能大于"):
                editor.setSelectionStatistics(11, 10)
        finally:
            editor.close()

    def test_histogram_drag_emits_live_changes_but_finishes_once(self) -> None:
        editor = HistogramRangeEditor(
            minimum=0.0,
            maximum=100.0,
            lower=20.0,
            upper=80.0,
        )
        editor.resize(420, 260)
        editor.show()
        self.app.processEvents()
        canvas = editor.histogramCanvas
        changes: list[tuple[float, float]] = []
        edit_finished: list[bool] = []
        interaction_finished: list[bool] = []
        editor.thresholdsChanged.connect(
            lambda lower, upper: changes.append((lower, upper))
        )
        editor.editFinished.connect(lambda: edit_finished.append(True))
        editor.interactionFinished.connect(
            lambda: interaction_finished.append(True)
        )
        try:
            plot = canvas._plot_rect()  # noqa: SLF001
            start = QPoint(
                int(round(plot.left() + plot.width() * 0.2)),
                int(round(plot.center().y())),
            )
            middle = QPoint(
                int(round(plot.left() + plot.width() * 0.3)),
                int(round(plot.center().y())),
            )
            end = QPoint(
                int(round(plot.left() + plot.width() * 0.4)),
                int(round(plot.center().y())),
            )
            QTest.mousePress(canvas, Qt.MouseButton.LeftButton, pos=start)
            QTest.mouseMove(canvas, pos=middle)
            QTest.mouseMove(canvas, pos=end)
            QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton, pos=end)
            self.assertGreaterEqual(len(changes), 2)
            self.assertEqual(edit_finished, [True])
            self.assertEqual(interaction_finished, [True])

            canvas.setFocus()
            QTest.keyClick(canvas, Qt.Key.Key_Right)
            self.assertEqual(len(edit_finished), 2)
            self.assertEqual(len(interaction_finished), 2)
        finally:
            editor.close()

    def test_histogram_mode_controls_ignore_wheel_changes(self) -> None:
        editor = HistogramRangeEditor()
        editor.show()
        self.app.processEvents()
        try:
            editor.displayModeCombo.setCurrentIndex(1)
            editor.polarityCombo.setCurrentIndex(1)
            before = (
                editor.displayModeCombo.currentIndex(),
                editor.polarityCombo.currentIndex(),
            )
            for combo in (editor.displayModeCombo, editor.polarityCombo):
                event = QWheelEvent(
                    QPointF(5.0, 5.0),
                    QPointF(5.0, 5.0),
                    QPoint(0, 0),
                    QPoint(0, 120),
                    Qt.MouseButton.NoButton,
                    Qt.KeyboardModifier.NoModifier,
                    Qt.ScrollPhase.ScrollUpdate,
                    False,
                )
                QApplication.sendEvent(combo, event)
            self.assertEqual(
                (
                    editor.displayModeCombo.currentIndex(),
                    editor.polarityCombo.currentIndex(),
                ),
                before,
            )
        finally:
            editor.close()

    def test_slider_number_editor_maps_linearly_and_ignores_wheel(self) -> None:
        editor = SliderNumberEditor(
            minimum=-2.0,
            maximum=2.0,
            value=0.25,
            decimals=4,
            suffix=" px",
            resolution=1000,
        )
        editor.resize(360, 42)
        editor.show()
        self.app.processEvents()
        changes: list[float] = []
        editor.valueChanged.connect(changes.append)
        try:
            self.assertEqual(editor.spinBox.suffix(), " px")
            editor.slider.setValue(750)
            self.assertAlmostEqual(editor.value(), 1.0, places=4)
            self.assertAlmostEqual(editor.spinBox.value(), 1.0, places=4)

            editor.spinBox.setValue(-1.125)
            self.assertAlmostEqual(editor.value(), -1.125, places=4)
            self.assertEqual(editor.slider.value(), 219)
            before = editor.value()
            event = QWheelEvent(
                QPointF(5.0, 5.0),
                QPointF(5.0, 5.0),
                QPoint(0, 0),
                QPoint(0, 120),
                Qt.MouseButton.NoButton,
                Qt.KeyboardModifier.NoModifier,
                Qt.ScrollPhase.ScrollUpdate,
                False,
            )
            QApplication.sendEvent(editor.slider, event)
            self.assertEqual(editor.value(), before)
            self.assertGreaterEqual(len(changes), 2)
        finally:
            editor.close()

    def test_slider_number_editor_emits_completion_on_release_and_edit(
        self,
    ) -> None:
        editor = SliderNumberEditor(
            minimum=0.0,
            maximum=10.0,
            value=5.0,
        )
        completions: list[bool] = []
        editor.editFinished.connect(lambda: completions.append(True))
        try:
            editor.slider.sliderReleased.emit()
            editor.spinBox.editingFinished.emit()
            self.assertEqual(len(completions), 2)
        finally:
            editor.close()

    def test_stripe_suppression_editor_explains_directional_frequency_band(
        self,
    ) -> None:
        editor = StripeSuppressionEditor(
            direction="horizontal",
            notch_width=0.04,
            protect_radius=0.01,
        )
        completions: list[bool] = []
        editor.editFinished.connect(
            lambda: completions.append(True)
        )
        try:
            self.assertEqual(
                editor.value(),
                {
                    "direction": "horizontal",
                    "notch_width": 0.04,
                    "protect_radius": 0.01,
                },
            )
            editor.directionCombo.setCurrentIndex(
                editor.directionCombo.findData("vertical")
            )
            self.assertEqual(
                editor.value()["direction"],
                "vertical",
            )
            self.assertTrue(completions)
            self.assertEqual(
                editor.frequencyCanvas._direction,  # noqa: SLF001
                "vertical",
            )
            self.assertIsInstance(
                editor.notchWidthSpin,
                NoWheelDoubleSpinBox,
            )
            self.assertIsInstance(
                editor.protectRadiusSpin,
                NoWheelDoubleSpinBox,
            )
        finally:
            editor.close()

    def test_percentile_range_editor_validates_and_explains_saturation(
        self,
    ) -> None:
        editor = PercentileRangeEditor(lower=0.5, upper=99.5)
        validations: list[tuple[bool, str]] = []
        editor.validationChanged.connect(
            lambda valid, message: validations.append((valid, message))
        )
        try:
            self.assertEqual(editor.value(), (0.5, 99.5))
            self.assertIn("低端 0.500%", editor.saturationLabel.text())
            editor.setResolvedText("灰度解析强度：12–240")
            self.assertEqual(
                editor.resolvedValuesLabel.text(),
                "灰度解析强度：12–240",
            )

            editor.lowerSpin.setValue(99.75)
            editor.upperSpin.setValue(99.5)
            self.assertFalse(editor.isValid())
            self.assertIn("必须小于", editor.validationMessage())
            self.assertEqual(validations[-1][0], False)

            editor.upperSpin.setValue(100.0)
            self.assertTrue(editor.isValid())
            self.assertEqual(editor.value(), (99.75, 100.0))
            self.assertEqual(validations[-1][0], True)
        finally:
            editor.close()

    def test_frequency_response_editor_exposes_modes_values_and_visibility(
        self,
    ) -> None:
        editor = FrequencyResponseEditor(
            mode="lowpass",
            low_cutoff=0.04,
            high_cutoff=0.18,
            order=4,
        )
        changes: list[dict[str, object]] = []
        finished: list[bool] = []
        editor.valueChanged.connect(lambda value: changes.append(dict(value)))
        editor.editFinished.connect(lambda: finished.append(True))
        editor.show()
        self.app.processEvents()
        try:
            self.assertIsInstance(
                editor.lowCutoffSpin,
                NoWheelDoubleSpinBox,
            )
            self.assertIsInstance(
                editor.highCutoffSpin,
                NoWheelDoubleSpinBox,
            )
            self.assertIsInstance(editor.orderSpin, NoWheelSpinBox)
            self.assertEqual(
                editor.value(),
                {
                    "mode": "lowpass",
                    "low_cutoff": 0.04,
                    "high_cutoff": 0.18,
                    "order": 4,
                },
            )
            self.assertTrue(editor.lowCutoffEditor.isHidden())
            self.assertFalse(editor.highCutoffEditor.isHidden())

            editor.setMode("highpass")
            self.assertFalse(editor.lowCutoffEditor.isHidden())
            self.assertTrue(editor.highCutoffEditor.isHidden())
            self.assertEqual(editor.mode(), "highpass")

            editor.setValue(
                {
                    "mode": "bandpass",
                    "low_cutoff": 0.08,
                    "high_cutoff": 0.24,
                    "order": 6,
                }
            )
            self.assertFalse(editor.lowCutoffEditor.isHidden())
            self.assertFalse(editor.highCutoffEditor.isHidden())
            self.assertEqual(editor.value()["order"], 6)
            self.assertEqual(changes[-1]["mode"], "bandpass")

            editor.modeCombo.setCurrentIndex(
                editor.modeCombo.findData("bandstop")
            )
            self.assertEqual(editor.mode(), "bandstop")
            self.assertTrue(finished)
        finally:
            editor.close()

    def test_frequency_response_editor_validates_band_cutoffs(self) -> None:
        editor = FrequencyResponseEditor(
            mode="bandpass",
            low_cutoff=0.08,
            high_cutoff=0.2,
        )
        validations: list[tuple[bool, str]] = []
        editor.validationChanged.connect(
            lambda valid, message: validations.append((valid, message))
        )
        try:
            editor.lowCutoffSpin.setValue(0.3)
            self.assertFalse(editor.isValid())
            self.assertIsNone(editor.tryValue())
            self.assertIn("高截止频率必须大于", editor.validationMessage())
            self.assertEqual(validations[-1][0], False)
            with self.assertRaisesRegex(ValueError, "高截止频率必须大于"):
                editor.value()

            editor.highCutoffSpin.setValue(0.4)
            self.assertTrue(editor.isValid())
            self.assertEqual(editor.value()["low_cutoff"], 0.3)
            self.assertEqual(editor.value()["high_cutoff"], 0.4)
            self.assertEqual(validations[-1][0], True)

            with self.assertRaisesRegex(ValueError, "高截止频率必须大于"):
                editor.setValue(
                    {
                        "mode": "bandstop",
                        "low_cutoff": 0.3,
                        "high_cutoff": 0.2,
                        "order": 2,
                    }
                )
            with self.assertRaisesRegex(ValueError, "1 到 16"):
                editor.setValue(
                    {
                        "mode": "lowpass",
                        "low_cutoff": 0.05,
                        "high_cutoff": 0.2,
                        "order": 17,
                    }
                )
        finally:
            editor.close()

    def test_frequency_response_editor_updates_physical_nyquist_without_clamping(
        self,
    ) -> None:
        editor = FrequencyResponseEditor(
            mode="lowpass",
            low_cutoff=0.0,
            high_cutoff=0.2,
        )
        try:
            self.assertTrue(editor.setFrequencyRange(0.0, 2.5))
            self.assertEqual(editor.frequencyRange(), (0.0, 2.5))
            editor.highCutoffSpin.setValue(2.0)
            self.assertFalse(editor.setFrequencyRange(0.0, 0.5))
            self.assertEqual(editor.frequencyRange(), (0.0, 2.5))
            self.assertEqual(editor.highCutoffSpin.value(), 2.0)
        finally:
            editor.close()

    def test_frequency_response_editor_ignores_wheel_and_renders_curve(
        self,
    ) -> None:
        editor = FrequencyResponseEditor(
            mode="bandstop",
            low_cutoff=0.08,
            high_cutoff=0.2,
            order=3,
        )
        editor.resize(420, 280)
        editor.show()
        self.app.processEvents()
        try:
            before = editor.rawValue()
            for widget in (
                editor.modeCombo,
                editor.lowCutoffSpin,
                editor.highCutoffSpin,
                editor.orderSpin,
                editor.lowCutoffSlider,
                editor.highCutoffSlider,
            ):
                event = QWheelEvent(
                    QPointF(5.0, 5.0),
                    QPointF(5.0, 5.0),
                    QPoint(0, 0),
                    QPoint(0, 120),
                    Qt.MouseButton.NoButton,
                    Qt.KeyboardModifier.NoModifier,
                    Qt.ScrollPhase.ScrollUpdate,
                    False,
                )
                QApplication.sendEvent(widget, event)
            self.assertEqual(editor.rawValue(), before)
            self.assertFalse(editor.responseCanvas.grab().toImage().isNull())
        finally:
            editor.close()

    def test_linked_dimensions_editor_links_exact_dimensions_and_percentage(
        self,
    ) -> None:
        editor = LinkedDimensionsEditor(
            source_width=4_000,
            source_height=2_000,
            maximum_dimension=20_000,
        )
        changes: list[tuple[int, int]] = []
        finished: list[bool] = []
        editor.valueChanged.connect(
            lambda width, height: changes.append((width, height))
        )
        editor.editFinished.connect(lambda: finished.append(True))
        try:
            self.assertIsInstance(editor.widthSpin, NoWheelSpinBox)
            self.assertIsInstance(editor.heightSpin, NoWheelSpinBox)
            self.assertIsInstance(
                editor.percentSpin,
                NoWheelDoubleSpinBox,
            )
            self.assertEqual(editor.sourceSize(), (4_000, 2_000))
            self.assertEqual(editor.value(), (4_000, 2_000))
            self.assertTrue(editor.isAspectLocked())
            self.assertIn("4,000 × 2,000 px", editor.sourceSizeLabel.text())

            editor.widthSpin.setValue(2_000)
            self.assertEqual(editor.value(), (2_000, 1_000))
            self.assertEqual(changes, [(2_000, 1_000)])
            self.assertAlmostEqual(editor.percentSpin.value(), 50.0)

            editor.heightSpin.setValue(1_500)
            self.assertEqual(editor.value(), (3_000, 1_500))
            self.assertEqual(changes[-1], (3_000, 1_500))

            editor.percentSpin.setValue(125.0)
            self.assertEqual(editor.value(), (5_000, 2_500))
            self.assertEqual(changes[-1], (5_000, 2_500))
            self.assertIn("5,000 × 2,500 px", editor.outputSummaryLabel.text())
            self.assertIn("12,500,000 像素", editor.outputSummaryLabel.text())

            editor.widthSpin.editingFinished.emit()
            editor.heightSpin.editingFinished.emit()
            editor.percentSpin.editingFinished.emit()
            self.assertEqual(len(finished), 3)
        finally:
            editor.close()

    def test_linked_dimensions_programmatic_sync_is_single_and_non_recursive(
        self,
    ) -> None:
        editor = LinkedDimensionsEditor(
            source_width=640,
            source_height=480,
        )
        changes: list[tuple[int, int]] = []
        editor.valueChanged.connect(
            lambda width, height: changes.append((width, height))
        )
        try:
            editor.setValue((320, 200))
            self.assertEqual(editor.value(), (320, 200))
            self.assertEqual(changes, [(320, 200)])

            editor.setValue(160, 120, emit_signal=False)
            self.assertEqual(editor.value(), (160, 120))
            self.assertEqual(changes, [(320, 200)])

            editor.setSourceSize(
                800,
                600,
                reset_value=True,
            )
            self.assertEqual(editor.value(), (800, 600))
            self.assertEqual(changes[-1], (800, 600))
            self.assertEqual(changes.count((800, 600)), 1)
        finally:
            editor.close()

    def test_linked_dimensions_can_disable_aspect_lock_for_canvas_resize(
        self,
    ) -> None:
        editor = LinkedDimensionsEditor(
            source_width=1_000,
            source_height=500,
            width=1_200,
            height=800,
            aspect_lock_available=False,
        )
        changes: list[tuple[int, int]] = []
        editor.valueChanged.connect(
            lambda width, height: changes.append((width, height))
        )
        try:
            self.assertFalse(editor.isAspectLockAvailable())
            self.assertFalse(editor.isAspectLocked())
            self.assertFalse(editor.lockAspectCheck.isVisible())

            editor.widthSpin.setValue(900)
            self.assertEqual(editor.value(), (900, 800))
            editor.heightSpin.setValue(700)
            self.assertEqual(editor.value(), (900, 700))
            self.assertEqual(changes, [(900, 800), (900, 700)])
            self.assertIn("宽 90.00% / 高 140.00%", editor.outputSummaryLabel.text())

            editor.percentSpin.setValue(50.0)
            self.assertEqual(editor.value(), (500, 250))
        finally:
            editor.close()

    def test_linked_dimensions_limits_and_wheel_safety(self) -> None:
        editor = LinkedDimensionsEditor(
            source_width=800,
            source_height=600,
            maximum_dimension=1_000,
        )
        editor.show()
        self.app.processEvents()
        try:
            self.assertEqual(editor.maximumDimension(), 1_000)
            with self.assertRaisesRegex(ValueError, "1 到 1000"):
                editor.setValue(1_001, 600)
            with self.assertRaisesRegex(ValueError, "正整数"):
                editor.setSourceSize(0, 600)
            with self.assertRaisesRegex(ValueError, "正整数"):
                editor.setMaximumDimension(0)

            before = editor.value()
            percent_before = editor.percentSpin.value()
            for widget in (
                editor.widthSpin,
                editor.heightSpin,
                editor.percentSpin,
            ):
                event = QWheelEvent(
                    QPointF(5.0, 5.0),
                    QPointF(5.0, 5.0),
                    QPoint(0, 0),
                    QPoint(0, 120),
                    Qt.MouseButton.NoButton,
                    Qt.KeyboardModifier.NoModifier,
                    Qt.ScrollPhase.ScrollUpdate,
                    False,
                )
                QApplication.sendEvent(widget, event)
            self.assertEqual(editor.value(), before)
            self.assertEqual(editor.percentSpin.value(), percent_before)

            editor.setMaximumDimension(500)
            self.assertEqual(editor.maximumDimension(), 500)
            self.assertEqual(editor.value(), (500, 500))
        finally:
            editor.close()

    def test_crop_bounds_editor_clamps_to_source_and_restores_full_image(
        self,
    ) -> None:
        editor = CropBoundsEditor(
            source_width=100,
            source_height=80,
            x=10,
            y=5,
            width=50,
            height=40,
        )
        changes: list[tuple[int, int, int, int]] = []
        editor.valueChanged.connect(
            lambda x, y, width, height: changes.append(
                (x, y, width, height)
            )
        )
        try:
            self.assertEqual(editor.value(), (10, 5, 50, 40))
            self.assertIn("2,000 像素", editor.summaryLabel.text())

            editor.xSpin.setValue(70)
            self.assertEqual(editor.value(), (70, 5, 30, 40))
            self.assertEqual(changes[-1], (70, 5, 30, 40))

            editor.fullImageButton.click()
            self.assertEqual(editor.value(), (0, 0, 100, 80))

            with self.assertRaisesRegex(ValueError, "完全位于"):
                editor.setValue(90, 0, 20, 10)
        finally:
            editor.close()

    def test_anchor_grid_has_all_stable_values_and_emits_selection(self) -> None:
        editor = AnchorGridEditor(value="top_left")
        selected: list[str] = []
        editor.valueChanged.connect(selected.append)
        try:
            self.assertEqual(tuple(editor.buttons), AnchorGridEditor.ANCHORS)
            self.assertEqual(editor.value(), "top_left")
            editor.buttons["bottom_right"].click()
            self.assertEqual(editor.anchor(), "bottom_right")
            self.assertEqual(selected, ["bottom_right"])
            editor.setAnchor("center_left")
            self.assertEqual(editor.value(), "center_left")
            with self.assertRaisesRegex(ValueError, "不支持的锚点"):
                editor.setValue("outside")
        finally:
            editor.close()

    def test_structuring_element_editor_exposes_exact_morphology_values(
        self,
    ) -> None:
        editor = StructuringElementEditor(
            radius=2,
            iterations=3,
            shape="cross",
        )
        changes: list[dict[str, int | str]] = []
        editor.valueChanged.connect(changes.append)
        try:
            self.assertIsInstance(editor.radiusSpin, NoWheelSpinBox)
            self.assertIsInstance(editor.iterationsSpin, NoWheelSpinBox)
            self.assertIsInstance(editor.shapeCombo, NoWheelComboBox)
            self.assertEqual(
                editor.value(),
                {"radius": 2, "iterations": 3, "kernel": "cross"},
            )
            self.assertEqual(editor.kernelSize(), (5, 5))
            self.assertEqual(editor.preview.kernelSize(), 5)

            editor.setValue(
                {
                    "radius": 4,
                    "iterations": 2,
                    "kernel": "rectangle",
                }
            )
            self.assertEqual(editor.radius(), 4)
            self.assertEqual(editor.iterations(), 2)
            self.assertEqual(editor.shape(), "rectangle")
            self.assertEqual(editor.kernelSize(), (9, 9))
            self.assertEqual(
                changes[-1],
                {"radius": 4, "iterations": 2, "kernel": "rectangle"},
            )

            editor.setValue(
                {"radius": 1, "iterations": 1, "shape": "ellipse"}
            )
            self.assertEqual(editor.shape(), "ellipse")
        finally:
            editor.close()

    def test_structuring_element_editor_validates_values_and_emits_finish(
        self,
    ) -> None:
        editor = StructuringElementEditor(
            maximum_radius=8,
            maximum_iterations=5,
        )
        finished: list[bool] = []
        editor.editFinished.connect(lambda: finished.append(True))
        try:
            with self.assertRaisesRegex(ValueError, "映射"):
                editor.setValue(1)  # type: ignore[arg-type]
            with self.assertRaisesRegex(ValueError, "半径必须"):
                editor.setValue({"radius": 9})
            with self.assertRaisesRegex(ValueError, "迭代次数必须"):
                editor.setValue({"iterations": 0})
            with self.assertRaisesRegex(ValueError, "不支持的结构元素"):
                editor.setValue({"kernel": "diamond"})

            editor.radiusSpin.editingFinished.emit()
            editor.iterationsSpin.editingFinished.emit()
            editor.shapeCombo.activated.emit(editor.shapeCombo.currentIndex())
            self.assertEqual(len(finished), 3)
        finally:
            editor.close()

    def test_structuring_element_controls_ignore_wheel_changes(self) -> None:
        editor = StructuringElementEditor(
            radius=3,
            iterations=2,
            shape="ellipse",
        )
        editor.show()
        self.app.processEvents()
        try:
            before = editor.value()
            for widget in (
                editor.radiusSpin,
                editor.iterationsSpin,
                editor.shapeCombo,
            ):
                event = QWheelEvent(
                    QPointF(5.0, 5.0),
                    QPointF(5.0, 5.0),
                    QPoint(0, 0),
                    QPoint(0, 120),
                    Qt.MouseButton.NoButton,
                    Qt.KeyboardModifier.NoModifier,
                    Qt.ScrollPhase.ScrollUpdate,
                    False,
                )
                QApplication.sendEvent(widget, event)
            self.assertEqual(editor.value(), before)
        finally:
            editor.close()

    def test_kernel_editor_presets_dimensions_and_finite_validation(self) -> None:
        editor = KernelMatrixEditor()
        validity: list[tuple[bool, str]] = []
        editor.validationChanged.connect(
            lambda valid, message: validity.append((valid, message))
        )
        try:
            self.assertEqual(
                editor.kernel(),
                KernelMatrixEditor.PRESETS["identity"],
            )
            editor.applyPreset("sharpen")
            self.assertEqual(
                editor.kernel(),
                KernelMatrixEditor.PRESETS["sharpen"],
            )

            editor.setDimensions(5, 5)
            self.assertEqual(editor.dimensions(), (5, 5))
            kernel = editor.kernel()
            self.assertEqual(kernel[2][2], 5.0)
            self.assertEqual(kernel[1][2], -1.0)

            editor.table.item(0, 0).setText("not-a-number")
            self.assertFalse(editor.isValid())
            self.assertIsNone(editor.tryKernel())
            self.assertIn("不是有效数字", editor.validationMessage())
            self.assertEqual(validity[-1][0], False)
            with self.assertRaisesRegex(ValueError, "不是有效数字"):
                editor.kernel()

            editor.table.item(0, 0).setText("1.25")
            self.assertTrue(editor.isValid())
            self.assertTrue(math.isclose(editor.kernel()[0][0], 1.25))
            with self.assertRaisesRegex(ValueError, "有限数"):
                editor.setKernel(((0.0, float("nan")),))
            with self.assertRaisesRegex(ValueError, "相同宽度"):
                editor.setKernel(((1.0,), (1.0, 2.0)))
        finally:
            editor.close()

    def test_custom_widgets_render_with_light_and_dark_palettes(self) -> None:
        widgets = (
            HistogramRangeEditor(),
            SliderNumberEditor(),
            FrequencyResponseEditor(),
            StripeSuppressionEditor(),
            LinkedDimensionsEditor(),
            PercentileRangeEditor(),
            CropBoundsEditor(source_width=640, source_height=480),
            AnchorGridEditor(),
            KernelMatrixEditor(),
            StructuringElementEditor(),
        )
        try:
            for dark in (False, True):
                palette = QPalette()
                palette.setColor(
                    QPalette.ColorRole.Window,
                    QColor("#20262d" if dark else "#f4f6f8"),
                )
                palette.setColor(
                    QPalette.ColorRole.Base,
                    QColor("#171c21" if dark else "#ffffff"),
                )
                palette.setColor(
                    QPalette.ColorRole.Text,
                    QColor("#eef2f4" if dark else "#1e2933"),
                )
                palette.setColor(
                    QPalette.ColorRole.Highlight,
                    QColor("#2a9d8f"),
                )
                palette.setColor(
                    QPalette.ColorRole.HighlightedText,
                    QColor("#ffffff"),
                )
                for widget in widgets:
                    widget.setPalette(palette)
                    widget.resize(360, 180)
                    image = widget.grab().toImage()
                    self.assertFalse(image.isNull())
            self.assertNotIn("#ffffff", widgets[2].styleSheet().lower())
        finally:
            for widget in widgets:
                widget.close()


if __name__ == "__main__":
    unittest.main()
