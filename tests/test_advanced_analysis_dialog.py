from __future__ import annotations

import os
from pathlib import Path
import sys
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PySide6.QtWidgets import QApplication, QComboBox, QDoubleSpinBox, QSpinBox

from fdm.raster import RasterPixelType
from fdm.services.analysis_profiles import analysis_output_field_schema
from fdm.ui.advanced_analysis_dialog import (
    AdvancedAnalysisParametersDialog,
    SPATIAL_POINT_SCOPE_KEY,
    SPATIAL_STUDY_AREA_MODE_KEY,
)
from fdm.ui.analysis_parameters_dialog import analysis_parameter_schema
from fdm.ui.image_analysis_controller import AnalysisTool


class _FakeWheelEvent:
    def __init__(self) -> None:
        self.ignored = False
        self.accepted = False

    def ignore(self) -> None:
        self.ignored = True

    def accept(self) -> None:
        self.accepted = True


class AdvancedAnalysisParametersDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def tearDown(self) -> None:
        for widget in QApplication.topLevelWidgets():
            if isinstance(widget, AdvancedAnalysisParametersDialog):
                widget.close()

    def _dialog(
        self,
        tool: AnalysisTool,
        **kwargs,
    ) -> AdvancedAnalysisParametersDialog:
        dialog = AdvancedAnalysisParametersDialog(
            tool,
            pixel_type=RasterPixelType.RGB8,
            **kwargs,
        )
        dialog.show()
        self.app.processEvents()
        return dialog

    def test_all_seven_advanced_tools_have_complete_valid_defaults(self) -> None:
        expected_keys = {
            AnalysisTool.DIRECTIONALITY: {
                "channel",
                "algorithm_version",
                "bins",
                "gradient_sigma",
                "minimum_gradient",
                "histogram_smoothing_bins",
                "peak_min_fraction",
                "max_peaks",
            },
            AnalysisTool.SKELETON: {
                "channel",
                "foreground",
                "threshold",
                "already_skeletonized",
                "algorithm_version",
                "prune_terminal_branches_below",
            },
            AnalysisTool.LOCAL_THICKNESS: {
                "channel",
                "foreground",
                "threshold",
            },
            AnalysisTool.TUBENESS: {
                "channel",
                "scales",
                "beta",
                "bright_ridges",
            },
            AnalysisTool.GLCM: {
                "channel",
                "levels",
                "distances",
                "directions_degrees",
                "symmetric",
            },
            AnalysisTool.SPATIAL_DISTRIBUTION: {
                "algorithm_version",
                SPATIAL_POINT_SCOPE_KEY,
                SPATIAL_STUDY_AREA_MODE_KEY,
            },
            AnalysisTool.SURFACE: {
                "channel",
                "sample_step_x",
                "sample_step_y",
            },
        }
        for tool, keys in expected_keys.items():
            with self.subTest(tool=tool):
                dialog = self._dialog(
                    tool,
                    active_group_label="玻璃纤维",
                )
                parameters = dialog.parameters()
                self.assertEqual(set(parameters), keys)
                if tool is AnalysisTool.DIRECTIONALITY:
                    self.assertEqual(parameters["algorithm_version"], 2)
                if tool is AnalysisTool.SKELETON:
                    self.assertEqual(parameters["algorithm_version"], 2)
                    self.assertEqual(
                        parameters["prune_terminal_branches_below"],
                        0.0,
                    )
                if tool is AnalysisTool.SPATIAL_DISTRIBUTION:
                    self.assertEqual(parameters["algorithm_version"], 2)
                dialog.close()

        spatial_schema = analysis_parameter_schema(
            AnalysisTool.SPATIAL_DISTRIBUTION
        )
        self.assertEqual(spatial_schema.version, "2")
        self.assertIn(
            "ripley_radii",
            {field.key for field in spatial_schema.fields},
        )

    def test_masked_binary_analysis_can_use_roi_without_threshold(self) -> None:
        dialog = self._dialog(
            AnalysisTool.LOCAL_THICKNESS,
            has_analysis_mask=True,
        )
        self.assertNotIn("threshold", dialog.parameters())

    def test_glcm_recalculation_restores_explicit_value_range(self) -> None:
        dialog = AdvancedAnalysisParametersDialog(
            AnalysisTool.GLCM,
            pixel_type=RasterPixelType.GRAY16,
            initial_parameters={
                "levels": 64,
                "distances": (1, 2),
                "directions_degrees": (0, 90),
                "value_range": (100.0, 5000.0),
                "symmetric": False,
            },
        )
        parameters = dialog.parameters()
        self.assertEqual(parameters["levels"], 64)
        self.assertEqual(parameters["distances"], (1, 2))
        self.assertEqual(parameters["directions_degrees"], (0.0, 90.0))
        self.assertEqual(parameters["value_range"], (100.0, 5000.0))
        self.assertFalse(parameters["symmetric"])
        dialog.close()

    def test_glcm_output_field_selection_does_not_change_kernel_parameters(
        self,
    ) -> None:
        dialog = self._dialog(AnalysisTool.GLCM)
        schema = analysis_output_field_schema("fdm.glcm")
        self.assertIsNotNone(schema)
        assert schema is not None
        parameters = dialog.parameters()

        self.assertEqual(dialog.output_fields(), schema.default_fields)
        dialog.set_output_fields(("contrast", "glcm_matrices"))

        self.assertEqual(
            dialog.output_fields(),
            ("contrast", "glcm_matrices"),
        )
        self.assertEqual(dialog.parameters(), parameters)
        dialog.close()

    def test_all_numeric_and_dropdown_editors_ignore_wheel(self) -> None:
        dialog = self._dialog(AnalysisTool.DIRECTIONALITY)
        editors = [
            *dialog.findChildren(QComboBox),
            *dialog.findChildren(QSpinBox),
            *dialog.findChildren(QDoubleSpinBox),
        ]
        self.assertTrue(editors)
        for editor in editors:
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
            self.assertTrue(event.ignored or event.accepted)


if __name__ == "__main__":
    unittest.main()
