from __future__ import annotations

import os
from pathlib import Path
import sys
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QDockWidget,
    QLabel,
    QMainWindow,
    QToolBar,
    QWidget,
)

from fdm.settings import WorkspaceLayoutSettings
from fdm.ui.fullscreen import FullscreenMeasurementController
from fdm.ui.workspace import AdaptiveLayoutController


class FullscreenMeasurementControllerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def _process_events(self, turns: int = 3) -> None:
        for _ in range(turns):
            self.app.processEvents()

    def _window(
        self,
        *,
        maximized: bool = False,
    ) -> tuple[
        QMainWindow,
        AdaptiveLayoutController,
        WorkspaceLayoutSettings,
        dict[str, QWidget],
    ]:
        window = QMainWindow()
        self.addCleanup(window.close)
        window.setCentralWidget(QWidget(window))
        window.resize(1600, 900)

        menu = window.menuBar()
        menu.addMenu("文件")
        status = window.statusBar()
        status.showMessage("ready")

        measurement_toolbar = QToolBar("测量", window)
        measurement_toolbar.setObjectName("measurementToolbar")
        measurement_toolbar.addAction("选择")
        window.addToolBar(measurement_toolbar)

        application_toolbar = QToolBar("应用", window)
        application_toolbar.setObjectName("applicationToolbar")
        application_toolbar.addAction("打开")
        window.addToolBar(application_toolbar)

        project = QDockWidget("项目", window)
        project.setObjectName("projectDock")
        project.setWidget(QLabel("project"))
        window.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, project)

        inspector = QDockWidget("检查器", window)
        inspector.setObjectName("inspectorDock")
        inspector.setWidget(QLabel("inspector"))
        window.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, inspector)

        results = QDockWidget("结果", window)
        results.setObjectName("resultsDock")
        results.setWidget(QLabel("results"))
        window.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, results)
        results.hide()

        extra = QLabel("document controls", window)
        extra.setObjectName("documentControls")
        extra.show()

        settings = WorkspaceLayoutSettings(
            project_width=312,
            inspector_width=428,
            results_height=344,
        )
        layout = AdaptiveLayoutController(
            window,
            project_dock=project,
            inspector_dock=inspector,
            results_dock=results,
            layout_settings=settings,
        )

        if maximized:
            window.showMaximized()
        else:
            window.show()
        self._process_events()
        return window, layout, settings, {
            "menu": menu,
            "status": status,
            "measurement_toolbar": measurement_toolbar,
            "application_toolbar": application_toolbar,
            "project": project,
            "inspector": inspector,
            "results": results,
            "extra": extra,
        }

    def test_normal_window_restores_geometry_chrome_and_layout_preferences(self) -> None:
        window, layout, settings, widgets = self._window()
        preferences_before = settings.to_dict()
        geometry_before = bytes(window.saveGeometry())
        state_before = bytes(window.saveState())
        preference_signals: list[bool] = []
        layout.layoutPreferencesChanged.connect(lambda: preference_signals.append(True))

        controller = FullscreenMeasurementController(
            window,
            adaptive_layout=layout,
            extra_chrome=(widgets["extra"],),
            preserved_widgets=(widgets["measurement_toolbar"],),
        )
        self.assertTrue(controller.enter())
        self._process_events()

        self.assertTrue(controller.is_active)
        self.assertTrue(layout.is_presentation_suspended)
        self.assertTrue(window.isFullScreen())
        self.assertTrue(widgets["menu"].isHidden())
        self.assertTrue(widgets["status"].isHidden())
        self.assertTrue(widgets["application_toolbar"].isHidden())
        self.assertFalse(widgets["measurement_toolbar"].isHidden())
        self.assertTrue(widgets["project"].isHidden())
        self.assertTrue(widgets["inspector"].isHidden())
        self.assertTrue(widgets["results"].isHidden())
        self.assertTrue(widgets["extra"].isHidden())

        close_state = controller.persistence_state_for_close()
        self.assertEqual(close_state.geometry, geometry_before)
        self.assertEqual(close_state.main_window_state, state_before)

        self.assertTrue(controller.exit())
        self._process_events()

        self.assertFalse(controller.is_active)
        self.assertFalse(layout.is_presentation_suspended)
        self.assertFalse(window.isFullScreen())
        self.assertFalse(widgets["menu"].isHidden())
        self.assertFalse(widgets["status"].isHidden())
        self.assertFalse(widgets["application_toolbar"].isHidden())
        self.assertFalse(widgets["measurement_toolbar"].isHidden())
        self.assertFalse(widgets["project"].isHidden())
        self.assertFalse(widgets["inspector"].isHidden())
        self.assertTrue(widgets["results"].isHidden())
        self.assertFalse(widgets["extra"].isHidden())
        self.assertEqual(settings.to_dict(), preferences_before)
        self.assertEqual(preference_signals, [])

    def test_maximized_window_and_external_state_change_are_restored(self) -> None:
        window, layout, settings, widgets = self._window(maximized=True)
        self.assertTrue(window.isMaximized())
        preferences_before = settings.to_dict()
        controller = FullscreenMeasurementController(
            window,
            adaptive_layout=layout,
            preserved_widgets=(widgets["measurement_toolbar"],),
        )
        self.assertTrue(controller.enter())
        self._process_events()
        self.assertTrue(window.isFullScreen())

        # Simulate the platform or window manager leaving full screen without
        # routing through the application's QAction.
        window.showNormal()
        self._process_events(5)

        self.assertFalse(controller.is_active)
        self.assertFalse(layout.is_presentation_suspended)
        self.assertFalse(window.isFullScreen())
        self.assertTrue(window.isMaximized())
        self.assertEqual(settings.to_dict(), preferences_before)
        self.assertFalse(widgets["project"].isHidden())
        self.assertFalse(widgets["inspector"].isHidden())
        self.assertTrue(widgets["results"].isHidden())

    def test_nested_presentation_freeze_requires_matching_release(self) -> None:
        _window, layout, settings, _widgets = self._window()
        before = settings.to_dict()

        layout.begin_presentation_mode()
        layout.begin_presentation_mode()
        self.assertTrue(layout.is_presentation_suspended)
        layout.end_presentation_mode()
        self.assertTrue(layout.is_presentation_suspended)
        layout.end_presentation_mode()

        self.assertFalse(layout.is_presentation_suspended)
        self.assertEqual(settings.to_dict(), before)


if __name__ == "__main__":
    unittest.main()
