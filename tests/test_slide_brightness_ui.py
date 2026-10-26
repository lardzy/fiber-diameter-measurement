from __future__ import annotations

import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from time import monotonic
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PySide6.QtCore import QCoreApplication, QEvent, QPoint, QPointF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QImage, QWheelEvent
from PySide6.QtTest import QSignalSpy, QTest
from PySide6.QtWidgets import QApplication, QToolButton, QWidget

from fdm.models import ImageDocument, new_id
from fdm.services.digital_slide_store import DigitalSlideManifest, DigitalSlideStore
from fdm.settings import AppSettings
from fdm.ui.digital_slide_canvas import DigitalSlideCanvas
from fdm.ui.main_window import MainWindow
from fdm.ui.slide_brightness_control import SlideBrightnessButton, SlideBrightnessPopup


class _Canvas(QWidget):
    displayBrightnessChanged = Signal(int)
    displayAdjustmentsChanged = Signal(int, int)

    def __init__(self) -> None:
        super().__init__()
        self.display_brightness = 0
        self.display_exposure_tenths = 0
        self.document_id = "original"

    @property
    def display_adjustments(self) -> tuple[int, int]:
        return self.display_brightness, self.display_exposure_tenths

    def set_display_adjustments(self, value: int, exposure: int) -> None:
        previous = self.display_brightness
        self.display_brightness, self.display_exposure_tenths = value, exposure
        if previous != value:
            self.displayBrightnessChanged.emit(value)
        self.displayAdjustmentsChanged.emit(value, exposure)

    def set_display_brightness(self, value: int) -> None:
        self.set_display_adjustments(value, self.display_exposure_tenths)


class SlideBrightnessPopupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.canvas = _Canvas()
        self.popup = SlideBrightnessPopup(self.canvas)
        self.popup.show()
        self.app.processEvents()

    def tearDown(self) -> None:
        from shiboken6 import isValid
        if isValid(self.popup):
            self.popup.close()
        if isValid(self.canvas):
            self.canvas.deleteLater()
        self.app.processEvents()

    def test_changes_coalesce_and_release_flushes_final_value(self) -> None:
        changed = QSignalSpy(self.canvas.displayBrightnessChanged)
        self.assertEqual(self.popup._timer.timerType(), Qt.TimerType.PreciseTimer)
        self.popup.slider.setValue(10)
        self.popup.slider.setValue(20)
        self.popup.slider.setValue(30)
        self.assertEqual(changed.count(), 0)
        # Native Windows event delivery can be delayed under VM load. This
        # functional check verifies coalescing; the benchmark measures latency.
        deadline = monotonic() + 2.0
        while changed.count() == 0 and monotonic() < deadline:
            QTest.qWait(10)
        self.assertEqual(changed.count(), 1)
        self.assertEqual(self.canvas.display_brightness, 30)
        self.assertEqual(self.popup.spin_box.value(), 30)
        self.popup.slider.setValue(-15)
        self.popup.slider.sliderReleased.emit()
        self.assertEqual(self.canvas.display_brightness, -15)
        QTest.qWait(50)
        self.assertEqual(changed.count(), 2)

    def test_close_flushes_and_reset_is_immediate(self) -> None:
        self.popup.spin_box.setValue(80)
        self.popup.reset_button.click()
        self.assertEqual(self.canvas.display_brightness, 0)
        self.assertEqual(self.popup.slider.value(), 0)
        self.popup.spin_box.setValue(-60)
        self.popup.close()
        self.assertEqual(self.canvas.display_brightness, -60)

    def test_batches_keep_cadence_from_application_including_processing_time(self) -> None:
        now = [10_000_000_000]
        apply = self.canvas.set_display_adjustments

        def apply_with_work(value: int, exposure: int) -> None:
            apply(value, exposure)
            now[0] += 15_000_000  # Simulate display work without wall-clock sleeps.

        with (
            patch("fdm.ui.slide_brightness_control.monotonic_ns", side_effect=lambda: now[0]),
            patch.object(self.canvas, "set_display_adjustments", side_effect=apply_with_work),
            patch.object(self.popup._timer, "start", wraps=self.popup._timer.start) as schedule,
        ):
            self.popup.slider.setValue(10)
            self.popup.slider.setValue(20)
            schedule.assert_called_once_with(0)
            self.assertEqual(self.canvas.display_brightness, 0)
            # Deliver the first scheduled turn, then drive later deadlines
            # explicitly so loaded native event loops do not affect assertions.
            self.app.processEvents()
            self.assertEqual(self.canvas.display_brightness, 20)
            now[0] = 10_025_000_000
            self.popup.slider.setValue(30)
            self.assertEqual(schedule.call_args.args, (8,))
            self.assertEqual(self.canvas.display_brightness, 20)
            now[0] = 10_030_000_000
            self.popup.slider.setValue(40)
            self.assertEqual(schedule.call_count, 2)  # New input cannot defer this deadline.
            now[0] = 10_033_000_000
            self.popup._timer.timeout.emit()
            self.assertEqual(self.canvas.display_brightness, 40)
            now[0] = 10_060_000_000
            self.popup.slider.setValue(50)
            self.assertEqual(schedule.call_args.args, (6,))
            self.popup.slider.sliderReleased.emit()
            self.assertEqual(self.canvas.display_brightness, 50)
            self.assertFalse(self.popup._timer.isActive())
            now[0] = 10_200_000_000
            self.popup.slider.setValue(60)
            self.assertEqual(schedule.call_args.args, (0,))
            self.assertEqual(self.canvas.display_brightness, 50)
            self.app.processEvents()
            self.assertEqual(self.canvas.display_brightness, 60)

    def test_deleted_canvas_drops_pending_edit(self) -> None:
        self.popup.slider.setValue(75)
        self.canvas.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.assertIsNone(self.popup.canvas)
        self.popup.flush()
        self.popup.close()

    def test_rebound_canvas_drops_pending_edit(self) -> None:
        self.popup.slider.setValue(75)
        self.canvas.document_id = "replacement"
        self.popup.flush()
        self.assertIsNone(self.popup.canvas)
        self.assertEqual(self.canvas.display_brightness, 0)

    def test_wheel_on_controls_is_consumed_without_changing_value(self) -> None:
        for control in (self.popup, self.popup.slider, self.popup.spin_box):
            event = QWheelEvent(
                QPointF(5, 5), QPointF(5, 5), QPoint(), QPoint(0, 120),
                Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
                Qt.ScrollPhase.NoScrollPhase, False,
            )
            QApplication.sendEvent(control, event)
            self.assertTrue(event.isAccepted())
        self.assertEqual(self.popup.spin_box.value(), 0)
        self.assertEqual(self.canvas.display_brightness, 0)

    def test_icon_text_button_tracks_font_and_style_without_numeric_geometry(self) -> None:
        button = SlideBrightnessButton(self.canvas)
        self.assertEqual(button.text(), "亮度/曝光")
        self.assertFalse(button.icon().isNull())
        self.assertEqual(button.toolButtonStyle(), Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        widths = []
        for point_size in (9, 14, 18):
            font = QFont(button.font())
            font.setPointSize(point_size)
            button.setFont(font)
            expected = button.sizeHint()
            widths.append(expected.width())
            for value, ev in ((0, 0), (100, 40), (-100, -40), (1, -1)):
                button.setToolTip(f"亮度：{value}；曝光：{ev / 10:+.1f} EV")
                self.assertEqual(button.sizeHint(), expected)
                self.assertEqual(button.text(), "亮度/曝光")
        self.assertLess(widths[0], widths[-1])
        before_style = button.sizeHint()
        button.setStyleSheet("QToolButton { padding: 12px 24px; border: 2px solid red; }")
        self.assertGreater(button.sizeHint().width(), before_style.width())


class SlideBrightnessWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.load_patch = patch("fdm.ui.main_window.AppSettingsIO.load", return_value=AppSettings())
        self.save_patch = patch("fdm.ui.main_window.AppSettingsIO.save")
        self.load_mock = self.load_patch.start()
        self.save_mock = self.save_patch.start()
        self.addCleanup(self.load_patch.stop)
        self.addCleanup(self.save_patch.stop)
        self.temp = TemporaryDirectory()
        self.window = MainWindow()
        self.window.resize(1400, 900)
        self.window.show()
        self.app.processEvents()

    def tearDown(self) -> None:
        self.window._reset_workspace()
        self.window.close()
        self.app.processEvents()
        self.temp.cleanup()

    def _slide(self, name: str) -> tuple[ImageDocument, DigitalSlideCanvas, Path]:
        path = Path(self.temp.name) / f"{name}.fdmslide"
        store = DigitalSlideStore.create(path, DigitalSlideManifest(
            version=1, width=320, height=240, viewport_width=80,
            viewport_height=60, focus_levels=[0, 2, 4],
        ))
        store.close()
        self.window._add_digital_slide_document_from_path(path, document=None)
        self.app.processEvents()
        canvas = self.window.current_canvas()
        self.assertIsInstance(canvas, DigitalSlideCanvas)
        return self.window.current_document(), canvas, path

    def test_tabs_are_independent_and_pending_edit_keeps_original_binding(self) -> None:
        first, first_canvas, _ = self._slide("first")
        second, second_canvas, _ = self._slide("second")
        self.window._set_current_document(first.id)
        before = first.to_dict()
        dirty = (first.dirty_flags.session_dirty, first.dirty_flags.calibration_dirty)
        self.save_mock.reset_mock()
        self.window.digital_slide_brightness_action.trigger()
        popup = self.window._slide_brightness_popup
        popup.slider.setValue(65)
        self.window._set_current_document(second.id)
        self.assertEqual(first_canvas.display_brightness, 65)
        self.assertEqual(second_canvas.display_brightness, 0)
        self.assertIsNone(self.window._slide_brightness_popup)
        self.assertEqual(self.window._slide_brightness_button.text(), "亮度/曝光")
        second_canvas.set_display_brightness(-25)
        self.window._set_current_document(first.id)
        self.assertEqual(self.window._slide_brightness_button.text(), "亮度/曝光")
        self.assertEqual(first.to_dict(), before)
        self.assertEqual((first.dirty_flags.session_dirty, first.dirty_flags.calibration_dirty), dirty)
        self.save_mock.assert_not_called()

    def test_menu_search_popup_focus_and_reopen_reset(self) -> None:
        document, canvas, path = self._slide("reopen")
        action = self.window.digital_slide_brightness_action
        self.assertTrue(action.isEnabled())
        self.assertTrue(action.isVisible())
        self.assertFalse(self.window._slide_brightness_button.isHidden())
        entries = [entry for entry in self.window._command_search_entries() if entry[0] is action]
        self.assertEqual(len(entries), 1)
        self.assertIn("视图", entries[0][1])
        self.assertIn("gamma", entries[0][2])
        self.assertIn("亮度", entries[0][2])
        self.assertIn("曝光", entries[0][2])
        self.assertIn("EV", entries[0][2])
        action.trigger()
        popup = self.window._slide_brightness_popup
        popup.slider.setValue(50)
        with patch.object(canvas, "focus_canvas") as focus:
            popup.close()
            focus.assert_called_once()
        self.assertEqual(canvas.display_brightness, 50)
        self.window._remove_document(document.id)
        self.window._add_digital_slide_document_from_path(path, document=None)
        self.app.processEvents()
        self.assertEqual(self.window.current_canvas().display_brightness, 0)
        self.assertEqual(self.window._slide_brightness_button.text(), "亮度/曝光")

    def test_ordinary_image_hides_slide_entry_and_keeps_display_adjustment(self) -> None:
        _, canvas, _ = self._slide("retained")
        canvas.set_display_brightness(40)
        image = QImage(24, 20, QImage.Format.Format_RGB32)
        image.fill(QColor("gray"))
        document = ImageDocument(id=new_id("image"), path="/tmp/ordinary.png", image_size=(24, 20))
        document.initialize_runtime_state()
        document.mark_session_saved()
        document.mark_calibration_saved()
        self.window._mount_document(document, image, tooltip=document.path)
        self.app.processEvents()
        self.assertTrue(self.window._slide_brightness_button.isHidden())
        self.assertFalse(self.window.digital_slide_brightness_action.isEnabled())
        self.assertFalse(self.window.digital_slide_brightness_action.isVisible())
        self.window._open_slide_brightness_popup()
        self.assertIsNone(self.window._slide_brightness_popup)
        self.window._open_display_adjustment_dialog()
        self.assertIsNotNone(self.window._display_adjustment_dialog)
        self.assertEqual(canvas.display_brightness, 40)

    def test_fullscreen_controls_keep_arrows_inside_popup_and_escape_returns_focus(self) -> None:
        _, canvas, _ = self._slide("fullscreen")
        self.window._toggle_fullscreen_measurement(True)
        self.app.processEvents()
        self.assertTrue(self.window._fullscreen_controller.is_active)
        self.assertTrue(self.window._slide_brightness_button.isVisible())
        self.window._slide_brightness_button.click()
        popup = self.window._slide_brightness_popup
        self.assertTrue(popup.isVisible())
        before = canvas.viewport_snapshot()
        focus_index = canvas.focus_index()
        QTest.keyClick(popup.slider, Qt.Key.Key_Right)
        self.assertEqual(popup.slider.value(), 1)
        self.assertEqual(canvas.focus_index(), focus_index)
        self.assertEqual(canvas.viewport_snapshot().visible_image_rect, before.visible_image_rect)
        QTest.keyClick(popup.slider, Qt.Key.Key_Escape)
        self.assertIsNone(self.window._slide_brightness_popup)
        self.assertEqual(canvas.display_brightness, 1)
        self.assertTrue(self.window._fullscreen_controller.is_active)

    def test_remember_only_commits_user_edits_and_initializes_new_slides(self) -> None:
        first, a, _ = self._slide("remember-a")
        second, b, _ = self._slide("remember-b")
        self.window._set_current_document(first.id)
        before = first.to_dict()
        dirty = (first.dirty_flags.session_dirty, first.dirty_flags.calibration_dirty)
        self.window.digital_slide_brightness_action.trigger()
        popup = self.window._slide_brightness_popup
        self.assertFalse(popup.remember_checkbox.isChecked())
        self.save_mock.reset_mock()
        popup.slider.setValue(45)
        popup.exposure_slider.setValue(12)
        popup.flush()
        self.assertEqual(a.display_adjustments, (45, 12))
        self.save_mock.assert_not_called()
        popup.remember_checkbox.click()
        self.assertEqual(self.save_mock.call_count, 1)
        self.assertEqual(b.display_adjustments, (0, 0))
        popup.slider.setValue(55)
        popup.exposure_slider.setValue(-7)
        popup.flush()
        self.assertEqual(self.save_mock.call_count, 1)
        popup.exposure_slider.sliderReleased.emit()
        self.assertEqual(self.save_mock.call_count, 2)
        remembered = self.save_mock.call_args.args[0]
        self.assertTrue(remembered.digital_slide_remember_view_adjustments)
        self.assertEqual((remembered.digital_slide_view_brightness,
                          remembered.digital_slide_view_exposure_tenths), (55, -7))
        popup.close()
        self.assertEqual(self.save_mock.call_count, 2)
        self.window._set_current_document(second.id)
        self.window.digital_slide_brightness_action.trigger()
        self.window._slide_brightness_popup.close()
        self.assertEqual(self.save_mock.call_count, 2)  # Visiting an older tab is not an edit.
        _, c, _ = self._slide("remember-c")
        self.assertEqual(c.display_adjustments, (55, -7))
        self.assertEqual(a.display_adjustments, (55, -7))
        self.assertEqual(b.display_adjustments, (0, 0))
        self.assertEqual(first.to_dict(), before)
        self.assertEqual((first.dirty_flags.session_dirty, first.dirty_flags.calibration_dirty), dirty)

    def test_remember_close_flushes_original_tab_and_restarts_from_saved_values(self) -> None:
        first, a, path = self._slide("restart-a")
        second, b, _ = self._slide("restart-b")
        self.window._set_current_document(first.id)
        self.window.digital_slide_brightness_action.trigger()
        popup = self.window._slide_brightness_popup
        popup.remember_checkbox.click()
        self.save_mock.reset_mock()
        popup.slider.setValue(-30)
        popup.exposure_spin_box.setValue(1.3)
        self.window._set_current_document(second.id)
        self.assertEqual(a.display_adjustments, (-30, 13))
        self.assertEqual(b.display_adjustments, (0, 0))
        self.assertEqual(self.save_mock.call_count, 1)
        restored = AppSettings.from_dict(self.save_mock.call_args.args[0].to_dict())
        self.window._reset_workspace()
        self.window.close()
        self.app.processEvents()
        self.load_mock.return_value = restored
        self.window = MainWindow()
        self.window._preview_active = True  # Hidden/deferred initial mount must retain the defaults.
        self.window._add_digital_slide_document_from_path(path, document=None)
        canvas = next(c for c in self.window._canvases.values() if isinstance(c, DigitalSlideCanvas))
        self.assertEqual(canvas.display_adjustments, (-30, 13))
        self.window._preview_active = False
        self.window.show()
        self.app.processEvents()
        self.assertEqual(canvas.display_adjustments, (-30, 13))
        self.window._sync_slide_brightness_controls()
        self.window.digital_slide_brightness_action.trigger()
        self.assertTrue(self.window._slide_brightness_popup.remember_checkbox.isChecked())

    def test_reset_and_uncheck_have_distinct_memory_effects(self) -> None:
        _, canvas, _ = self._slide("memory-reset")
        self.window.digital_slide_brightness_action.trigger()
        popup = self.window._slide_brightness_popup
        popup.remember_checkbox.click()
        popup.slider.setValue(60)
        popup.exposure_slider.setValue(20)
        popup.commit()
        popup.reset_button.click()
        self.assertEqual(canvas.display_adjustments, (0, 0))
        self.assertTrue(popup.remember_checkbox.isChecked())
        saved = self.save_mock.call_args.args[0]
        self.assertEqual((saved.digital_slide_view_brightness, saved.digital_slide_view_exposure_tenths), (0, 0))
        popup.slider.setValue(70)
        popup.exposure_slider.setValue(-15)
        self.save_mock.reset_mock()
        popup.remember_checkbox.click()  # Flush current effect but save only the disabled defaults.
        self.assertEqual(canvas.display_adjustments, (70, -15))
        self.assertFalse(popup.remember_checkbox.isChecked())
        self.assertEqual(self.save_mock.call_count, 1)
        saved = self.save_mock.call_args.args[0]
        self.assertFalse(saved.digital_slide_remember_view_adjustments)
        self.assertEqual((saved.digital_slide_view_brightness, saved.digital_slide_view_exposure_tenths), (0, 0))
        _, fresh, _ = self._slide("memory-disabled-new")
        self.assertEqual(fresh.display_adjustments, (0, 0))
        self.assertEqual(canvas.display_adjustments, (70, -15))

    def test_persistence_failure_keeps_current_view_and_previous_saved_defaults(self) -> None:
        _, canvas, _ = self._slide("memory-failure")
        self.window.digital_slide_brightness_action.trigger()
        popup = self.window._slide_brightness_popup
        popup.slider.setValue(35)
        popup.exposure_slider.setValue(8)
        self.save_mock.side_effect = OSError("read-only profile")
        popup.remember_checkbox.click()
        self.assertEqual(canvas.display_adjustments, (35, 8))
        self.assertFalse(popup.remember_checkbox.isChecked())
        self.assertFalse(self.window._app_settings.digital_slide_remember_view_adjustments)
        self.assertIn("未能记住", self.window.statusBar().currentMessage())
        self.save_mock.side_effect = None
        popup.remember_checkbox.click()
        self.save_mock.side_effect = OSError("disk full")
        popup.exposure_slider.setValue(-10)
        popup.commit()
        self.assertEqual(canvas.display_adjustments, (35, -10))
        self.assertEqual(self.window._app_settings.digital_slide_view_exposure_tenths, 8)
        self.assertTrue(popup.remember_checkbox.isChecked())
        self.save_mock.side_effect = None
        _, fresh, _ = self._slide("after-failure")
        self.assertEqual(fresh.display_adjustments, (35, 8))

    def test_exit_reports_failed_final_save_before_window_disappears(self) -> None:
        self._slide("close-save-failure")
        self.window.digital_slide_brightness_action.trigger()
        popup = self.window._slide_brightness_popup
        popup.remember_checkbox.click()
        popup.slider.setValue(42)
        popup.exposure_slider.setValue(7)
        self.save_mock.side_effect = OSError("disk full on exit")
        visible_at_warning = []
        with patch("fdm.ui.main_window.QMessageBox.warning", side_effect=lambda *_: (
            visible_at_warning.append(self.window.isVisible())
        )) as warning:
            self.assertTrue(self.window.close())
        warning.assert_called_once()
        self.assertEqual(visible_at_warning, [True])
        self.assertIn("disk full on exit", warning.call_args.args[2])
        self.assertEqual(self.window._app_settings.digital_slide_view_brightness, 0)
        self.save_mock.side_effect = None

    def test_narrow_window_brightness_keeps_canvas_and_view_after_layout_events(self) -> None:
        _, canvas, _ = self._slide("narrow-layout")
        canvas.set_view_zoom(4.0)

        def state() -> tuple:
            return (
                self.window.size(), self.window._slide_brightness_button.size(),
                self.window._document_view_controls.size(), canvas.geometry(),
                canvas._view_generation, canvas.viewport_snapshot().visible_image_rect,
            )

        def settle() -> tuple:
            previous = state()
            stable_turns = 0
            deadline = monotonic() + 3.0
            while monotonic() < deadline:
                QTest.qWait(50)
                current = state()
                stable_turns = stable_turns + 1 if current == previous else 0
                if stable_turns >= 3:
                    return current
                previous = current
            self.fail("view layout did not settle")

        for width in (800, 640):
            self.window.resize(width, 900)
            baseline = settle()
            for value in (1, 10, 100, -1, -10, -100, 0):
                canvas.set_display_adjustments(value, value // 3)
                self.assertEqual(settle(), baseline, f"width={width}, brightness={value}")


if __name__ == "__main__":
    unittest.main()
