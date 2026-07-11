from __future__ import annotations

import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PySide6.QtGui import QImage
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QToolButton

from fdm.geometry import Line, Point
from fdm.models import Calibration, ImageDocument, Measurement, ProjectState
from fdm.settings import AppSettings
from fdm.ui.main_window import MainWindow
from fdm.ui.statistics_widgets import MeasurementStatisticsPanel
from fdm.ui.workspace import WorkspaceMode


class ProfessionalWorkspaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def _window(self) -> MainWindow:
        load = patch("fdm.ui.main_window.AppSettingsIO.load", return_value=AppSettings())
        save = patch("fdm.ui.main_window.AppSettingsIO.save", return_value=None)
        load.start()
        save.start()
        self.addCleanup(load.stop)
        self.addCleanup(save.stop)
        window = MainWindow()
        self.addCleanup(window.close)
        return window

    def test_compact_layout_preserves_canvas_and_one_side_panel(self) -> None:
        window = self._window()
        window.resize(1093, 576)
        window.show()
        self.app.processEvents()

        self.assertTrue(window._adaptive_layout.is_compact)
        self.assertFalse(window._project_dock.isVisible())
        self.assertTrue(window._inspector_dock.isVisible())
        self.assertFalse(window._results_dock.isVisible())
        self.assertGreaterEqual(window.tab_widget.width(), 560)

        window._toggle_project_panel()
        self.app.processEvents()
        self.assertTrue(window._project_dock.isVisible())
        self.assertFalse(window._inspector_dock.isVisible())

    def test_wide_layout_exposes_docks_and_results_remain_opt_in(self) -> None:
        window = self._window()
        window.resize(1600, 900)
        window.show()
        self.app.processEvents()

        self.assertFalse(window._adaptive_layout.is_compact)
        self.assertTrue(window._project_dock.isVisible())
        self.assertTrue(window._inspector_dock.isVisible())
        self.assertFalse(window._results_dock.isVisible())
        window._toggle_results_panel()
        self.app.processEvents()
        self.assertTrue(window._results_dock.isVisible())

    def test_medium_layout_temporarily_hides_project_when_results_expand(self) -> None:
        window = self._window()
        window.resize(1280, 720)
        window.show()
        self.app.processEvents()
        self.assertTrue(window._project_dock.isVisible())

        window._toggle_results_panel()
        self.app.processEvents()
        self.assertEqual(window.height(), 720)
        self.assertFalse(window._project_dock.isVisible())
        self.assertTrue(window._inspector_dock.isVisible())
        self.assertTrue(window._results_dock.isVisible())
        self.assertGreaterEqual(window.tab_widget.width(), 560)

        window._toggle_results_panel()
        self.app.processEvents()
        self.assertTrue(window._project_dock.isVisible())

    def test_compact_results_drawer_does_not_resize_window_beyond_screen(self) -> None:
        window = self._window()
        window.resize(1093, 576)
        window.show()
        self.app.processEvents()

        window._toggle_results_panel()
        self.app.processEvents()

        self.assertEqual(window.size().toTuple(), (1093, 576))
        self.assertTrue(window._results_dock.isVisible())
        self.assertGreaterEqual(window._results_tabs.height(), 120)

    def test_responsive_geometry_matrix_keeps_explicit_commands_reachable(self) -> None:
        window = self._window()
        window.show()
        for width, height, compact, command_style in (
            (1093, 576, True, Qt.ToolButtonStyle.ToolButtonIconOnly),
            (1280, 720, False, Qt.ToolButtonStyle.ToolButtonIconOnly),
            (1536, 864, False, Qt.ToolButtonStyle.ToolButtonTextBesideIcon),
            (1920, 1000, False, Qt.ToolButtonStyle.ToolButtonTextBesideIcon),
            (2048, 1152, False, Qt.ToolButtonStyle.ToolButtonTextBesideIcon),
        ):
            with self.subTest(size=(width, height)):
                window.resize(width, height)
                self.app.processEvents()
                self.assertEqual(window._adaptive_layout.is_compact, compact)
                self.assertEqual(window._file_toolbar.toolButtonStyle(), command_style)
                self.assertGreaterEqual(window.tab_widget.width(), 560)
                self.assertEqual(
                    window._right_standard_panel.horizontalScrollBarPolicy(),
                    Qt.ScrollBarPolicy.ScrollBarAlwaysOff,
                )
                self.assertIsNotNone(window.findChild(QToolButton, "openCommandButton"))
                self.assertIsNotNone(window.findChild(QToolButton, "moreCommandButton"))

    def test_main_command_bar_never_overflows_preferences_or_critical_commands(self) -> None:
        window = self._window()
        window.show()

        critical_actions = (
            window.save_project_action,
            window.undo_action,
            window.redo_action,
            window.measure_workspace_action,
            window.live_preview_action,
            window.capture_frame_action,
            window.settings_action,
        )
        for width, height in ((1093, 576), (1280, 720), (1600, 900)):
            with self.subTest(size=(width, height)):
                window.resize(width, height)
                self.app.processEvents()
                toolbar = window._file_toolbar
                extension = toolbar.findChild(QToolButton, "qt_toolbar_ext_button")
                self.assertIsNotNone(extension)
                self.assertFalse(extension.isVisible())
                for action in critical_actions:
                    button = toolbar.widgetForAction(action)
                    self.assertIsNotNone(button, action.text())
                    self.assertTrue(button.isVisible(), action.text())
                    self.assertFalse(button.isHidden(), action.text())

                settings_button = window.findChild(QToolButton, "settingsCommandButton")
                self.assertIs(settings_button, toolbar.widgetForAction(window.settings_action))
                self.assertTrue(settings_button.isVisible())

        toolbar_actions = set(window._file_toolbar.actions())
        self.assertNotIn(window.digital_slide_action, toolbar_actions)
        self.assertNotIn(window.optimize_capture_signal_action, toolbar_actions)
        self.assertNotIn(window.close_current_action, toolbar_actions)
        self.assertNotIn(window.close_all_action, toolbar_actions)
        more_button = window.findChild(QToolButton, "moreCommandButton")
        more_actions = set(more_button.menu().actions())
        self.assertIn(window.digital_slide_action, more_actions)
        self.assertIn(window.optimize_capture_signal_action, more_actions)
        self.assertIn(window.close_current_action, more_actions)
        self.assertIn(window.close_all_action, more_actions)

    def test_measurement_inspector_orders_calibration_before_current_image_properties(self) -> None:
        window = self._window()
        inspector_content = window._right_standard_panel.widget()
        self.assertIsNotNone(inspector_content)
        layout = inspector_content.layout()
        self.assertIsNotNone(layout)
        object_names = [
            layout.itemAt(index).widget().objectName()
            for index in range(layout.count())
            if layout.itemAt(index).widget() is not None
        ]

        self.assertLess(object_names.index("calibrationBox"), object_names.index("areaRecognitionBox"))
        self.assertLess(
            object_names.index("areaRecognitionBox"),
            object_names.index("currentImagePropertiesBox"),
        )

    def test_workspace_mode_replaces_measurement_tools_with_acquisition_context(self) -> None:
        window = self._window()
        window._preview_active = True
        window._sync_workspace_mode()
        self.assertEqual(window._workspace_mode, WorkspaceMode.ACQUIRE)
        self.assertFalse(window._measure_toolbar.isHidden())
        self.assertFalse(window._measurement_tool_strip.primaryToolsVisible())
        self.assertTrue(window._measurement_tool_strip.isPreviewContextVisible())
        self.assertFalse(window._acquisition_right_panel.isHidden())
        self.assertTrue(window._right_standard_panel.isHidden())
        window._preview_active = False
        window._sync_workspace_mode()
        self.assertEqual(window._workspace_mode, WorkspaceMode.MEASURE)
        self.assertTrue(window._measurement_tool_strip.primaryToolsVisible())
        self.assertTrue(window._acquisition_right_panel.isHidden())
        self.assertFalse(window._right_standard_panel.isHidden())

    def test_live_statistics_uses_selected_metric_and_scope(self) -> None:
        document = ImageDocument(
            id="image",
            path="/tmp/stats.png",
            image_size=(100, 80),
            calibration=Calibration(mode="preset", pixels_per_unit=2.0, unit="um", source_label="test"),
        )
        document.initialize_runtime_state()
        group = document.create_group(color="#2A9D8F", label="棉")
        document.set_active_group(group.id)
        for index, length in enumerate((20.0, 40.0, 60.0)):
            document.add_measurement(
                Measurement(
                    id=f"m{index}",
                    image_id=document.id,
                    fiber_group_id=group.id,
                    mode="manual",
                    line_px=Line(Point(0, index), Point(length, index)),
                    status="manual",
                )
            )
        panel = MeasurementStatisticsPanel()
        self.addCleanup(panel.close)
        panel.set_context(
            ProjectState(version="test", documents=[document]),
            document,
            tool_mode="manual",
            selected_measurement=document.measurements[-1],
        )

        snapshot = panel.snapshots[0]
        self.assertEqual(snapshot.unit, "um")
        self.assertEqual(snapshot.valid_count, 3)
        self.assertEqual(snapshot.mean, 20.0)
        self.assertIn("20 um", panel._mean_cell.value_label.text())
        self.assertIn("手工", panel.current_value_label.text())

    def test_background_statistics_drops_late_generation(self) -> None:
        document = ImageDocument(id="async", path="/tmp/async.png", image_size=(20, 20))
        document.initialize_runtime_state()
        document.add_measurement(
            Measurement(
                id="m1",
                image_id=document.id,
                fiber_group_id=None,
                mode="manual",
                line_px=Line(Point(0, 0), Point(10, 0)),
                status="manual",
            )
        )
        project = ProjectState(version="test", documents=[document])
        panel = MeasurementStatisticsPanel()
        self.addCleanup(panel.close)
        panel.set_context(project, document, tool_mode="manual", selected_measurement=None)
        completed = panel.snapshots

        queued_tasks = []

        class _Pool:
            def start(self, task) -> None:
                queued_tasks.append(task)

        panel.BACKGROUND_THRESHOLD = 0
        with patch("fdm.ui.statistics_widgets.QThreadPool.globalInstance", return_value=_Pool()):
            panel.refresh()
            first_generation = panel._generation
            panel.refresh()
            current_generation = panel._generation

        self.assertEqual(len(queued_tasks), 2)
        self.assertIn("正在后台计算", panel.details_label.text())
        panel._on_async_statistics_ready(first_generation, (), None)
        self.assertIn("正在后台计算", panel.details_label.text())
        panel._on_async_statistics_ready(current_generation, completed, None)
        self.assertEqual(panel.snapshots, completed)
        self.assertNotIn("正在后台计算", panel.details_label.text())


if __name__ == "__main__":
    unittest.main()
