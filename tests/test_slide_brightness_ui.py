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

    def __init__(self) -> None:
        super().__init__()
        self.display_brightness = 0
        self.document_id = "original"

    def set_display_brightness(self, value: int) -> None:
        self.display_brightness = value
        self.displayBrightnessChanged.emit(value)


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
        apply = self.canvas.set_display_brightness

        def apply_with_work(value: int) -> None:
            apply(value)
            now[0] += 15_000_000  # Simulate display work without wall-clock sleeps.

        with (
            patch("fdm.ui.slide_brightness_control.monotonic_ns", side_effect=lambda: now[0]),
            patch.object(self.canvas, "set_display_brightness", side_effect=apply_with_work),
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

    def test_button_reserves_range_and_tracks_font_and_style_changes(self) -> None:
        button = SlideBrightnessButton(self.canvas)
        reference = QToolButton(self.canvas)
        reference.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        widths = []
        for point_size in (9, 14, 18):
            font = QFont(button.font())
            font.setPointSize(point_size)
            button.setFont(font)
            reference.setFont(font)
            expected_size = button.sizeHint()
            widths.append(expected_size.width())
            for value in range(-100, 101):
                text = f"亮度：{value}"
                button.setText(text)
                reference.setText(text)
                self.assertEqual(button.sizeHint(), expected_size)
                self.assertEqual(button.minimumSizeHint(), expected_size)
                self.assertGreaterEqual(expected_size.width(), reference.sizeHint().width())
                self.assertGreaterEqual(expected_size.height(), reference.sizeHint().height())
        self.assertLess(widths[0], widths[-1])
        before_style = button.sizeHint()
        button.setStyleSheet("QToolButton { padding: 12px 24px; border: 2px solid red; }")
        self.assertGreater(button.sizeHint().width(), before_style.width())

    def test_button_text_updates_reuse_size_without_remeasuring_font(self) -> None:
        button = SlideBrightnessButton(self.canvas)
        expected = button.sizeHint()
        with patch.object(button, "fontMetrics", side_effect=AssertionError("remeasured unchanged font")):
            for value in range(-100, 101):
                button.setText(f"亮度：{value}")
                self.assertEqual(button.sizeHint(), expected)
                self.assertEqual(button.minimumSizeHint(), expected)
            # A caller may modify the returned QSize without corrupting cache.
            hint = button.sizeHint()
            hint.setWidth(1)
            self.assertEqual(button.sizeHint(), expected)
        with patch.object(button, "fontMetrics", wraps=button.fontMetrics) as metrics:
            QApplication.sendEvent(button, QEvent(QEvent.Type.DevicePixelRatioChange))
            self.assertEqual(button.sizeHint(), expected)
            self.assertGreater(metrics.call_count, 0)


class SlideBrightnessWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.load_patch = patch("fdm.ui.main_window.AppSettingsIO.load", return_value=AppSettings())
        self.save_patch = patch("fdm.ui.main_window.AppSettingsIO.save")
        self.load_patch.start()
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
        self.assertEqual(self.window._slide_brightness_button.text(), "亮度：0")
        second_canvas.set_display_brightness(-25)
        self.window._set_current_document(first.id)
        self.assertEqual(self.window._slide_brightness_button.text(), "亮度：65")
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
        self.assertEqual(self.window._slide_brightness_button.text(), "亮度：0")

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
                canvas.set_display_brightness(value)
                self.assertEqual(settle(), baseline, f"width={width}, brightness={value}")


if __name__ == "__main__":
    unittest.main()
