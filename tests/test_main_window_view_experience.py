from __future__ import annotations

import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PySide6.QtCore import QByteArray, QCoreApplication, QEvent, Qt
from PySide6.QtGui import QColor, QImage, QKeyEvent
from PySide6.QtWidgets import QApplication, QToolButton
from PySide6.QtTest import QTest
from shiboken6 import isValid as is_qobject_valid

from fdm.geometry import Point
from fdm.models import (
    ImageDocument,
    OverlayAnnotationKind,
    OverlayTextSizeSpace,
    new_id,
)
from fdm.services.digital_slide_store import DigitalSlideManifest, DigitalSlideStore
from fdm.services.export_service import ExportImageRenderMode
from fdm.settings import AppSettings, MagicSegmentToolMode
from fdm.ui.digital_slide_canvas import DigitalSlideCanvas
from fdm.ui.main_window import MainWindow
from fdm.ui.view_transform import CanvasZoomMode


class MainWindowViewExperienceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.settings = AppSettings(
            theme_mode="dark",
            show_canvas_navigator=True,
        )
        self.load_patch = patch(
            "fdm.ui.main_window.AppSettingsIO.load",
            return_value=self.settings,
        )
        self.save_patch = patch(
            "fdm.ui.main_window.AppSettingsIO.save",
            return_value=None,
        )
        self.load_patch.start()
        self.save_mock = self.save_patch.start()
        self.addCleanup(self.load_patch.stop)
        self.addCleanup(self.save_patch.stop)

    def _process_events(self, turns: int = 5) -> None:
        for _ in range(turns):
            self.app.processEvents()

    def _window(self) -> MainWindow:
        window = MainWindow()
        window.resize(1600, 900)
        window.show()
        self._process_events()

        def cleanup() -> None:
            if window._fullscreen_controller is not None:
                window._fullscreen_controller.exit()
            window._reset_workspace()
            window.close()
            self._process_events()

        self.addCleanup(cleanup)
        return window

    def _mount_document(
        self,
        window: MainWindow,
        *,
        name: str,
        size: tuple[int, int] = (1600, 1000),
    ) -> ImageDocument:
        image = QImage(size[0], size[1], QImage.Format.Format_RGB32)
        image.fill(QColor("#D7DFDC"))
        document = ImageDocument(
            id=new_id("image"),
            path=f"/tmp/{name}.png",
            image_size=size,
        )
        document.initialize_runtime_state()
        document.mark_session_saved()
        document.mark_calibration_saved()
        window._mount_document(document, image, tooltip=document.path)
        self._process_events()
        return document

    def test_each_tab_keeps_its_zoom_and_view_center(self) -> None:
        window = self._window()
        first = self._mount_document(window, name="view-a")
        first_canvas = window.current_canvas()
        self.assertIsNotNone(first_canvas)
        first_canvas.set_view_zoom(2.4)
        first_canvas.center_on_image_point(Point(870.0, 540.0))
        first_snapshot = first_canvas.viewport_snapshot()
        self.assertIsNotNone(first_snapshot)

        second = self._mount_document(window, name="view-b")
        second_canvas = window.current_canvas()
        self.assertIsNotNone(second_canvas)
        second_canvas.set_view_zoom(1.6)
        second_canvas.center_on_image_point(Point(520.0, 360.0))

        window._set_current_document(first.id)
        self._process_events()

        restored = window.current_canvas()
        self.assertIs(restored, first_canvas)
        self.assertEqual(restored.zoom_mode(), CanvasZoomMode.CUSTOM)
        self.assertAlmostEqual(restored.view_zoom(), 2.4)
        restored_snapshot = restored.viewport_snapshot()
        self.assertIsNotNone(restored_snapshot)
        self.assertAlmostEqual(
            restored_snapshot.visible_image_rect.center().x(),
            first_snapshot.visible_image_rect.center().x(),
            places=5,
        )
        self.assertAlmostEqual(
            restored_snapshot.visible_image_rect.center().y(),
            first_snapshot.visible_image_rect.center().y(),
            delta=2.0,
        )
        self.assertIn("240%", window._zoom_status_button.text())
        navigator = window._canvas_navigators[first.id]
        self.assertFalse(navigator.isHidden())
        self.assertEqual(
            navigator._snapshot.document_id,  # noqa: SLF001 - integration assertion
            first.id,
        )

        window._set_current_document(second.id)
        self._process_events()
        self.assertAlmostEqual(window.current_canvas().view_zoom(), 1.6)
        self.assertIn("160%", window._zoom_status_button.text())

    def test_fullscreen_keeps_measurement_controls_and_restores_chrome(self) -> None:
        window = self._window()
        self._mount_document(window, name="fullscreen")
        project_visible = not window._project_dock.isHidden()
        inspector_visible = not window._inspector_dock.isHidden()

        window.fullscreen_measurement_action.trigger()
        self._process_events()

        self.assertTrue(window._fullscreen_controller.is_active)
        self.assertTrue(window.isFullScreen())
        self.assertTrue(window.fullscreen_measurement_action.isChecked())
        self.assertTrue(window.menuBar().isHidden())
        self.assertTrue(window._file_toolbar.isHidden())
        self.assertFalse(window._measure_toolbar.isHidden())
        self.assertFalse(window.statusBar().isHidden())
        self.assertFalse(window._document_view_controls.isHidden())
        self.assertTrue(window._project_dock.isHidden())
        self.assertTrue(window._inspector_dock.isHidden())
        self.assertTrue(window._version_label.isHidden())
        self.assertTrue(window._image_resolution_label.isHidden())

        window.fullscreen_measurement_action.trigger()
        self._process_events()

        self.assertFalse(window._fullscreen_controller.is_active)
        self.assertFalse(window.isFullScreen())
        self.assertFalse(window.fullscreen_measurement_action.isChecked())
        self.assertFalse(window.menuBar().isHidden())
        self.assertFalse(window._file_toolbar.isHidden())
        if window._adaptive_layout.is_compact:
            # The offscreen platform clamps the restored normal window to its
            # synthetic 800 px screen; the responsive compact rule is then
            # intentionally reapplied.
            self.assertTrue(window._project_dock.isHidden())
            self.assertFalse(window._inspector_dock.isHidden())
        else:
            self.assertEqual(
                not window._project_dock.isHidden(),
                project_visible,
            )
            self.assertEqual(
                not window._inspector_dock.isHidden(),
                inspector_visible,
            )

    def test_fullscreen_hint_is_recreated_after_its_canvas_is_removed(self) -> None:
        window = self._window()
        self._mount_document(window, name="fullscreen-hint-first")

        window.fullscreen_measurement_action.trigger()
        self._process_events()
        old_hint = window._fullscreen_hint_label
        self.assertIsNotNone(old_hint)
        assert old_hint is not None
        self.assertTrue(is_qobject_valid(old_hint))

        # Exit cleanup must release the member before the parent canvas is
        # deleted; this was the stale-wrapper sequence reported by PySide.
        window.fullscreen_measurement_action.trigger()
        self.assertIsNone(window._fullscreen_hint_label)
        window._reset_workspace()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self._process_events()
        self.assertFalse(is_qobject_valid(old_hint))

        self._mount_document(window, name="fullscreen-hint-second")
        window.fullscreen_measurement_action.trigger()
        self._process_events()
        new_hint = window._fullscreen_hint_label
        self.assertIsNotNone(new_hint)
        assert new_hint is not None
        self.assertTrue(is_qobject_valid(new_hint))
        self.assertIsNot(new_hint, old_hint)

        window._fullscreen_hint_timer.stop()
        window._fade_fullscreen_hint()
        animation = window._fullscreen_hint_animation
        self.assertIsNotNone(animation)
        assert animation is not None
        animation.setDuration(1)
        QTest.qWait(20)
        self._process_events()
        self.assertIsNone(window._fullscreen_hint_label)

    def test_destroying_active_hint_parent_cancels_delayed_callbacks(self) -> None:
        window = self._window()
        self._mount_document(window, name="fullscreen-hint-parent-delete")
        window.fullscreen_measurement_action.trigger()
        self._process_events()
        hint = window._fullscreen_hint_label
        self.assertIsNotNone(hint)
        assert hint is not None

        window._fullscreen_hint_timer.stop()
        window._fade_fullscreen_hint()
        self.assertIsNotNone(window._fullscreen_hint_animation)
        window._reset_workspace()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self._process_events()

        self.assertFalse(is_qobject_valid(hint))
        self.assertIsNone(window._fullscreen_hint_label)
        self.assertIsNone(window._fullscreen_hint_animation)
        self.assertFalse(window._fullscreen_hint_timer.isActive())
        # Every queued entry point must now be an idempotent no-op.
        window._position_fullscreen_hint()
        window._fade_fullscreen_hint()
        window._hide_fullscreen_hint(delete=True)

    def test_escape_cancels_drawing_before_leaving_fullscreen(self) -> None:
        window = self._window()
        self._mount_document(window, name="fullscreen-escape")
        canvas = window.current_canvas()
        window.set_tool_mode("polygon_area")
        canvas._drawing_polygon_points = [Point(80.0, 80.0)]
        window.fullscreen_measurement_action.trigger()
        self._process_events()

        escape = QKeyEvent(
            QKeyEvent.Type.KeyPress,
            Qt.Key.Key_Escape,
            Qt.KeyboardModifier.NoModifier,
        )
        window.keyPressEvent(escape)
        self._process_events()

        self.assertFalse(canvas.has_pending_path_drawing())
        self.assertTrue(window._fullscreen_controller.is_active)

        escape = QKeyEvent(
            QKeyEvent.Type.KeyPress,
            Qt.Key.Key_Escape,
            Qt.KeyboardModifier.NoModifier,
        )
        window.keyPressEvent(escape)
        self._process_events()
        self.assertFalse(window._fullscreen_controller.is_active)

    def test_persist_while_fullscreen_uses_normal_window_snapshot(self) -> None:
        window = self._window()
        self._mount_document(window, name="fullscreen-persist")
        window.fullscreen_measurement_action.trigger()
        self._process_events()
        snapshot = window._fullscreen_controller.entry_snapshot
        self.assertIsNotNone(snapshot)

        window._persist_window_geometry()

        expected_geometry = bytes(
            QByteArray(snapshot.geometry).toBase64()
        ).decode("ascii")
        expected_state = bytes(
            QByteArray(snapshot.main_window_state).toBase64()
        ).decode("ascii")
        self.assertEqual(
            window._app_settings.main_window_geometry,
            expected_geometry,
        )
        self.assertEqual(
            window._app_settings.main_window_state,
            expected_state,
        )
        self.assertEqual(
            window._app_settings.main_window_is_maximized,
            bool(snapshot.window_state & Qt.WindowState.WindowMaximized),
        )

    def test_canvas_navigator_toggle_is_persisted_and_disables_all_instances(
        self,
    ) -> None:
        window = self._window()
        first = self._mount_document(window, name="navigator-a")
        second = self._mount_document(window, name="navigator-b")
        for canvas in window._canvases.values():
            canvas.set_view_zoom(2.0)
        self._process_events()

        window.toggle_canvas_navigator_action.setChecked(False)
        window._toggle_canvas_navigator(False)

        self.assertFalse(window._app_settings.show_canvas_navigator)
        self.assertFalse(window._canvas_navigators[first.id].navigator_enabled)
        self.assertFalse(window._canvas_navigators[second.id].navigator_enabled)
        self.assertTrue(window._canvas_navigators[first.id].isHidden())
        self.assertTrue(window._canvas_navigators[second.id].isHidden())

    def test_digital_slide_navigation_toggles_restore_apply_and_persist(self) -> None:
        self.settings.digital_slide_smooth_navigation_enabled = False
        self.settings.digital_slide_shift_navigation_enabled = True
        window = self._window()
        self.save_mock.reset_mock()

        with TemporaryDirectory() as tmp_dir:
            slide_path = Path(tmp_dir) / "navigation-preferences.fdmslide"
            store = DigitalSlideStore.create(
                slide_path,
                DigitalSlideManifest(
                    version=1,
                    width=1200,
                    height=900,
                    viewport_width=200,
                    viewport_height=150,
                    focus_levels=[0],
                ),
            )
            store.close()
            window._add_digital_slide_document_from_path(slide_path, document=None)
            self._process_events()
            # Opening a document may persist unrelated window/session state;
            # count only the three navigation preference transitions below.
            self.save_mock.reset_mock()

            canvas = window.current_canvas()
            self.assertIsInstance(canvas, DigitalSlideCanvas)
            assert isinstance(canvas, DigitalSlideCanvas)
            self.assertEqual(canvas.navigation_mode(), "step")
            self.assertTrue(canvas.shift_navigation_enabled())
            self.assertFalse(window.digital_slide_smooth_navigation_action.isChecked())
            self.assertTrue(window.digital_slide_shift_navigation_action.isChecked())
            self.assertTrue(window.digital_slide_smooth_navigation_action.isEnabled())
            self.assertTrue(window.digital_slide_shift_navigation_action.isEnabled())

            controls = [
                button.defaultAction()
                for button in window._document_view_controls.findChildren(QToolButton)
            ]
            smooth_index = controls.index(
                window.digital_slide_smooth_navigation_action
            )
            self.assertIs(
                controls[smooth_index + 1],
                window.digital_slide_shift_navigation_action,
            )

            window.digital_slide_smooth_navigation_action.trigger()
            self.assertTrue(
                window._app_settings.digital_slide_smooth_navigation_enabled
            )
            self.assertEqual(canvas.navigation_mode(), "smooth")

            window.digital_slide_shift_navigation_action.trigger()
            self.assertFalse(
                window._app_settings.digital_slide_shift_navigation_enabled
            )
            self.assertFalse(canvas.shift_navigation_enabled())

            canvas.toggle_navigation_mode()
            self.assertFalse(
                window._app_settings.digital_slide_smooth_navigation_enabled
            )
            self.assertFalse(window.digital_slide_smooth_navigation_action.isChecked())

        self.assertEqual(self.save_mock.call_count, 3)

    def test_digital_slide_overview_gates_pixel_tools_and_restores_previous_tool(
        self,
    ) -> None:
        window = self._window()
        with TemporaryDirectory() as tmp_dir:
            slide_path = Path(tmp_dir) / "pixel-work-gate.fdmslide"
            store = DigitalSlideStore.create(
                slide_path,
                DigitalSlideManifest(
                    version=1,
                    width=1200,
                    height=900,
                    viewport_width=200,
                    viewport_height=150,
                    focus_levels=[0],
                ),
            )
            store.close()
            window._add_digital_slide_document_from_path(slide_path, document=None)

            canvas = window.current_canvas()
            self.assertIsInstance(canvas, DigitalSlideCanvas)
            assert isinstance(canvas, DigitalSlideCanvas)
            for _ in range(200):
                self._process_events(1)
                if canvas.pixel_work_enabled():
                    break
                QTest.qWait(5)
            self.assertTrue(canvas.pixel_work_enabled())

            window.set_tool_mode("manual")
            self.assertEqual(window._tool_mode, "manual")
            canvas.fit_to_view()

            self.assertFalse(canvas.pixel_work_enabled())
            self.assertEqual(window._tool_mode, "select")
            self.assertFalse(window.export_current_image_action.isEnabled())
            self.assertFalse(window.image_processing_workbench_action.isEnabled())
            self.assertTrue(window.fit_action.isEnabled())
            self.assertTrue(window.digital_slide_native_fit_action.isEnabled())
            self.assertFalse(window._mode_actions["manual"].isEnabled())
            self.assertTrue(window._mode_actions["select"].isEnabled())

            window.set_tool_mode("construction")
            self.assertEqual(window._tool_mode, "select")
            canvas.move_viewport_by(180.0, 120.0)
            canvas.fit_native_viewport()
            self.assertFalse(canvas.pixel_work_enabled())

            for _ in range(200):
                self._process_events(1)
                if canvas.pixel_work_enabled():
                    break
                QTest.qWait(5)
            self.assertTrue(canvas.pixel_work_enabled())
            self.assertEqual(window._tool_mode, "manual")
            self.assertTrue(window.export_current_image_action.isEnabled())

    def test_native_move_and_focus_do_not_suspend_or_restore_the_current_tool(
        self,
    ) -> None:
        window = self._window()
        with TemporaryDirectory() as tmp_dir:
            slide_path = Path(tmp_dir) / "native-loading-state.fdmslide"
            store = DigitalSlideStore.create(
                slide_path,
                DigitalSlideManifest(
                    version=1,
                    width=1200,
                    height=900,
                    viewport_width=200,
                    viewport_height=150,
                    focus_levels=[-1, 0],
                ),
            )
            store.close()
            window._add_digital_slide_document_from_path(slide_path, document=None)

            canvas = window.current_canvas()
            self.assertIsInstance(canvas, DigitalSlideCanvas)
            assert isinstance(canvas, DigitalSlideCanvas)
            for _ in range(200):
                self._process_events(1)
                if canvas.pixel_work_enabled():
                    break
                QTest.qWait(5)
            self.assertTrue(canvas.pixel_work_enabled())

            window.set_tool_mode("manual")
            document = window.current_document()
            self.assertIsNotNone(document)
            assert document is not None
            self.assertIsNotNone(window._manual_tool_button)
            self.assertIsNotNone(window._add_group_button)
            assert window._manual_tool_button is not None
            assert window._add_group_button is not None
            self.assertTrue(window._manual_tool_button.isEnabled())
            self.assertTrue(window._add_group_button.isEnabled())
            messages: list[str] = []
            window.statusBar().messageChanged.connect(messages.append)

            canvas.move_viewport_by(50.0, 40.0)
            self.assertFalse(canvas.pixel_work_enabled())
            self.assertEqual(window._tool_mode, "manual")
            self.assertNotIn(document.id, window._digital_slide_suspended_tools)
            self.assertTrue(window._manual_tool_button.isEnabled())
            self.assertTrue(window._add_group_button.isEnabled())
            for _ in range(200):
                self._process_events(1)
                self.assertTrue(window._manual_tool_button.isEnabled())
                self.assertTrue(window._add_group_button.isEnabled())
                if canvas.pixel_work_enabled():
                    break
                QTest.qWait(5)
            self.assertTrue(canvas.pixel_work_enabled())
            self.assertEqual(window._tool_mode, "manual")

            canvas.set_focus_index(0 if canvas.focus_index() != 0 else 1)
            self.assertFalse(canvas.pixel_work_enabled())
            self.assertEqual(window._tool_mode, "manual")
            self.assertNotIn(document.id, window._digital_slide_suspended_tools)
            self.assertTrue(window._manual_tool_button.isEnabled())
            self.assertTrue(window._add_group_button.isEnabled())
            for _ in range(200):
                self._process_events(1)
                self.assertTrue(window._manual_tool_button.isEnabled())
                self.assertTrue(window._add_group_button.isEnabled())
                if canvas.pixel_work_enabled():
                    break
                QTest.qWait(5)
            self.assertTrue(canvas.pixel_work_enabled())
            self.assertEqual(window._tool_mode, "manual")
            self.assertFalse(
                any("已恢复此前工具" in message for message in messages)
            )

    def test_low_zoom_text_and_viewport_export_use_exact_scale(self) -> None:
        window = self._window()
        document = self._mount_document(window, name="low-zoom")
        canvas = window.current_canvas()
        canvas.set_view_zoom(0.01)
        canvas.center_on_image_point(Point(800.0, 500.0))
        window._app_settings.text_font_size = 8
        window._app_settings.text_size_space = OverlayTextSizeSpace.IMAGE_PX

        with patch(
            "fdm.ui.main_window.QInputDialog.getMultiLineText",
            return_value=("字", True),
        ):
            window._on_canvas_overlay_create_requested(
                document.id,
                {
                    "kind": OverlayAnnotationKind.TEXT,
                    "anchor_px": Point(800.0, 500.0),
                },
            )

        annotation = document.overlay_annotations[0]
        self.assertIsNotNone(annotation.text_layout)
        self.assertAlmostEqual(
            annotation.text_layout.image_font_size_px,
            800.0,
        )

        with TemporaryDirectory() as tmp_dir:
            output = Path(tmp_dir) / "current-viewport.png"
            with patch("fdm.ui.main_window.draw_measurements") as draw:
                window._render_overlay_image(
                    document,
                    output,
                    include_measurements=True,
                    include_scale=False,
                    render_mode=ExportImageRenderMode.CURRENT_VIEWPORT,
                )
            mapper = draw.call_args.args[2]
            point = Point(900.0, 560.0)
            mapped = mapper(point)
            expected = canvas.image_to_widget(point)
            self.assertAlmostEqual(mapped.x(), expected.x(), places=6)
            self.assertAlmostEqual(mapped.y(), expected.y(), places=6)

    def test_digital_navigator_requests_are_debounced_to_latest_point(self) -> None:
        window = self._window()
        document = self._mount_document(window, name="navigator-debounce")
        ordinary_canvas = window._canvases[document.id]
        digital_canvas = DigitalSlideCanvas()
        window._canvases[document.id] = digital_canvas
        try:
            with patch.object(
                digital_canvas,
                "center_on_image_point",
            ) as center:
                for index in range(20):
                    window._on_canvas_navigator_center_requested(
                        document.id,
                        Point(float(index), float(index * 2)),
                    )

                center.assert_not_called()
                window._navigator_center_timer.stop()
                window._apply_pending_navigator_center()

                center.assert_called_once_with(Point(19.0, 38.0))
        finally:
            window._canvases[document.id] = ordinary_canvas
            digital_canvas.close()
            digital_canvas.deleteLater()

    def test_canvas_focus_escape_and_f11_follow_fullscreen_priority(self) -> None:
        window = self._window()
        self._mount_document(window, name="fullscreen-real-keys")
        canvas = window.current_canvas()
        canvas.focus_canvas()
        window.set_tool_mode("polygon_area")
        canvas._drawing_polygon_points = [Point(80.0, 80.0)]
        window.fullscreen_measurement_action.trigger()
        self._process_events()

        QTest.keyClick(canvas, Qt.Key.Key_Escape)
        self._process_events()
        self.assertFalse(canvas.has_pending_path_drawing())
        self.assertTrue(window._fullscreen_controller.is_active)

        QTest.keyClick(canvas, Qt.Key.Key_Escape)
        self._process_events()
        self.assertFalse(window._fullscreen_controller.is_active)

        canvas._drawing_polygon_points = [Point(90.0, 90.0)]
        window.fullscreen_measurement_action.trigger()
        self._process_events()
        QTest.keyClick(canvas, Qt.Key.Key_F11)
        self._process_events()
        self.assertFalse(window._fullscreen_controller.is_active)
        self.assertTrue(canvas.has_pending_path_drawing())
        canvas.cancel_pending_path()

        window.set_tool_mode(MagicSegmentToolMode.STANDARD)
        window.fullscreen_measurement_action.trigger()
        self._process_events()
        QTest.keyClick(canvas, Qt.Key.Key_Escape)
        self._process_events()
        self.assertFalse(window._fullscreen_controller.is_active)


if __name__ == "__main__":
    unittest.main()
