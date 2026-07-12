from __future__ import annotations

import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PySide6.QtGui import QImage
from PySide6.QtCore import QItemSelectionModel, Qt
from PySide6.QtWidgets import QApplication, QToolButton

from fdm.geometry import Line, Point
from fdm.models import Calibration, ImageDocument, Measurement, ProjectState
from fdm.services.measurement_statistics import MeasurementMetric
from fdm.settings import AppSettings, WorkspaceLayoutSettings
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
        preferred_height = window._app_settings.workspace_layout.results_height
        window.resizeDocks([window._results_dock], [160], Qt.Orientation.Vertical)
        self.app.processEvents()
        self.assertEqual(window._app_settings.workspace_layout.results_height, preferred_height)

    def test_results_and_side_docks_restore_last_user_extents(self) -> None:
        window = self._window()
        window.resize(1700, 1000)
        window.show()
        self.app.processEvents()
        window._toggle_results_panel()
        self.app.processEvents()
        window.resizeDocks([window._results_dock], [340], Qt.Orientation.Vertical)
        window.resizeDocks(
            [window._project_dock, window._inspector_dock],
            [310, 390],
            Qt.Orientation.Horizontal,
        )
        self.app.processEvents()
        self.assertAlmostEqual(window._app_settings.workspace_layout.results_height, 340, delta=8)
        self.assertAlmostEqual(window._app_settings.workspace_layout.project_width, 310, delta=8)
        self.assertAlmostEqual(window._app_settings.workspace_layout.inspector_width, 390, delta=8)

        window._toggle_results_panel()
        window._toggle_project_panel()
        window._toggle_inspector_panel()
        self.app.processEvents()
        window._toggle_project_panel()
        window._toggle_inspector_panel()
        window._toggle_results_panel()
        self.app.processEvents()
        self.app.processEvents()
        self.assertAlmostEqual(window._results_dock.height(), 340, delta=12)
        self.assertAlmostEqual(window._project_dock.width(), 310, delta=12)
        self.assertAlmostEqual(window._inspector_dock.width(), 390, delta=12)

    def test_compact_transition_does_not_overwrite_wide_extent_preferences(self) -> None:
        window = self._window()
        window.resize(1700, 900)
        window.show()
        self.app.processEvents()
        window.resizeDocks(
            [window._project_dock, window._inspector_dock],
            [300, 380],
            Qt.Orientation.Horizontal,
        )
        self.app.processEvents()
        expected = (
            window._app_settings.workspace_layout.project_width,
            window._app_settings.workspace_layout.inspector_width,
        )
        window.resize(1093, 576)
        self.app.processEvents()
        self.app.processEvents()
        window.resize(1700, 900)
        self.app.processEvents()
        self.app.processEvents()
        self.assertEqual(
            (
                window._app_settings.workspace_layout.project_width,
                window._app_settings.workspace_layout.inspector_width,
            ),
            expected,
        )
        self.assertAlmostEqual(window._project_dock.width(), expected[0], delta=12)
        self.assertAlmostEqual(window._inspector_dock.width(), expected[1], delta=12)

    def test_restore_default_layout_resets_sizes_and_section_states(self) -> None:
        window = self._window()
        state = window._app_settings.workspace_layout
        state.project_width = 420
        state.inspector_width = 460
        state.results_height = 380
        state.inspector_records_height = 360
        window._statistics_section.setExpanded(True)
        window._records_section.setExpanded(False)
        window._area_recognition_section.setExpanded(True)
        window._object_properties_section.setExpanded(True)

        window._reset_workspace_layout()

        defaults = WorkspaceLayoutSettings()
        self.assertEqual(window._app_settings.workspace_layout, defaults)
        self.assertFalse(window._statistics_section.isExpanded())
        self.assertTrue(window._records_section.isExpanded())
        self.assertFalse(window._area_recognition_section.isExpanded())
        self.assertFalse(window._object_properties_section.isExpanded())

    def test_inspector_records_collapse_restores_dragged_height(self) -> None:
        window = self._window()
        window.resize(1600, 900)
        window.show()
        self.app.processEvents()
        splitter = window._inspector_splitter
        total = splitter.height()
        splitter.setSizes([max(100, total - 410), 300, 110])
        self.app.processEvents()
        remembered = window._records_section.height()
        window._records_section.setExpanded(False)
        self.app.processEvents()
        self.assertLess(window._records_section.height(), remembered)
        window._records_section.setExpanded(True)
        self.app.processEvents()
        self.app.processEvents()
        self.assertAlmostEqual(window._records_section.height(), remembered, delta=16)

    def test_right_and_bottom_record_views_share_state_but_keep_column_layouts(self) -> None:
        window = self._window()
        document = ImageDocument(id="records", path="/tmp/records.png", image_size=(100, 80))
        document.initialize_runtime_state()
        group = document.create_group(color="#2A9D8F", label="棉")
        for index, length in enumerate((20.0, 40.0)):
            measurement = Measurement(
                id=f"record_{index}",
                image_id=document.id,
                fiber_group_id=group.id,
                mode="manual",
                line_px=Line(Point(0, index), Point(length, index)),
            )
            measurement.recalculate(None)
            document.add_measurement(measurement)
        window._records_controller.set_document(document)
        right = window._inspector_records_pane
        bottom = window._bottom_records_pane
        right.search_edit.setText("record_1")
        self.app.processEvents()
        self.assertEqual(bottom.search_edit.text(), "record_1")
        self.assertEqual(right.table.model().rowCount(), 1)
        self.assertIs(right.table.selectionModel(), bottom.table.selectionModel())
        index = right.table.model().index(0, 0)
        right.table.selectionModel().select(
            index,
            QItemSelectionModel.SelectionFlag.ClearAndSelect
            | QItemSelectionModel.SelectionFlag.Rows,
        )
        self.assertEqual(window._records_controller.selected_measurement_ids(), ["record_1"])
        right.table.setColumnHidden(window.TABLE_COL_MODE, False)
        self.assertFalse(right.table.isColumnHidden(window.TABLE_COL_MODE))
        self.assertFalse(bottom.table.isColumnHidden(window.TABLE_COL_MODE))
        right.table.setColumnHidden(window.TABLE_COL_MODE, True)
        self.assertTrue(right.table.isColumnHidden(window.TABLE_COL_MODE))
        self.assertFalse(bottom.table.isColumnHidden(window.TABLE_COL_MODE))

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
                    window._inspector_splitter.widget(0).horizontalScrollBarPolicy(),
                    Qt.ScrollBarPolicy.ScrollBarAlwaysOff,
                )
                self.assertEqual(
                    window._inspector_splitter.widget(2).horizontalScrollBarPolicy(),
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

    def test_split_tool_menu_checked_hover_uses_palette_safe_combined_state(self) -> None:
        window = self._window()
        stylesheet = window._build_split_menu_stylesheet("manualToolMenu", "")
        checked_rule = "QMenu#manualToolMenu::item:checked {"
        combined_rule = "QMenu#manualToolMenu::item:checked:selected {"
        self.assertIn("background: palette(alternate-base);", stylesheet)
        self.assertIn("color: palette(text);", stylesheet)
        self.assertIn("background: palette(highlight);", stylesheet)
        self.assertIn("color: palette(highlighted-text);", stylesheet)
        self.assertLess(stylesheet.index(checked_rule), stylesheet.index(combined_rule))

    def test_measurement_inspector_uses_locked_section_order_and_defaults(self) -> None:
        window = self._window()
        splitter = window._inspector_splitter
        top_content = splitter.widget(0).widget()
        lower_content = splitter.widget(2).widget()
        self.assertIs(top_content.layout().itemAt(0).widget(), window._statistics_section)
        self.assertEqual(top_content.layout().itemAt(1).widget().objectName(), "calibrationBox")
        self.assertIs(splitter.widget(1), window._records_section)
        self.assertIs(lower_content.layout().itemAt(0).widget(), window._area_recognition_section)
        self.assertIs(lower_content.layout().itemAt(1).widget(), window._object_properties_section)
        self.assertFalse(window._statistics_section.isExpanded())
        self.assertTrue(window._records_section.isExpanded())
        self.assertFalse(window._area_recognition_section.isExpanded())
        self.assertFalse(window._object_properties_section.isExpanded())

    def test_compact_record_columns_fit_the_inspector_without_horizontal_scroll(self) -> None:
        window = self._window()
        window.resize(1093, 576)
        window.show()
        self.app.processEvents()
        pane = window._inspector_records_pane
        self.assertIsNotNone(pane)
        self.assertEqual(pane.table.horizontalScrollBar().maximum(), 0)

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

    def test_distribution_inherits_tool_only_when_first_opened(self) -> None:
        window = self._window()
        window.resize(1600, 900)
        window.show()
        self.app.processEvents()
        window._refresh_statistics_ui()
        self.assertFalse(window._distribution_widget._metric_initialized)
        self.assertFalse(window._distribution_widget._context_mode)

        window.set_tool_mode("polygon_area")
        window._toggle_results_panel()
        window._results_tabs.setCurrentWidget(window._distribution_page)
        self.app.processEvents()
        self.assertEqual(window._distribution_widget.active_metric(), MeasurementMetric.AREA)

        window.set_tool_mode("count")
        window._refresh_statistics_ui()
        self.assertEqual(window._distribution_widget.active_metric(), MeasurementMetric.AREA)

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
