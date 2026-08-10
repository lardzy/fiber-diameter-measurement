from __future__ import annotations

import os
from pathlib import Path
import sys
import unittest


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import QApplication

from fdm.screenshot_settings import (
    AfterCaptureTask,
    HotkeyBinding,
    ImageFormat,
    ScreenshotSettings,
)
from fdm.services.screenshot_capture import CaptureMode
from fdm.ui.screenshot_settings_page import ScreenshotSettingsPage


class ScreenshotSettingsPageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_roundtrips_independent_companion_preferences(self) -> None:
        page = ScreenshotSettingsPage(
            ScreenshotSettings(
                enabled=True,
                autostart=True,
                output_directory="/tmp/screens",
                image_format=ImageFormat.JPEG,
                jpeg_quality=87,
                after_capture_tasks=(
                    AfterCaptureTask.SAVE,
                    AfterCaptureTask.COPY_CLIPBOARD,
                ),
            )
        )
        try:
            page.hotkey_edits[CaptureMode.CU5].setKeySequence(QKeySequence("Ctrl+Alt+5"))
            actual = page.settings()
        finally:
            page.close()

        self.assertTrue(actual.enabled)
        self.assertTrue(actual.autostart)
        self.assertEqual(actual.output_directory, "/tmp/screens")
        self.assertEqual(actual.image_format, ImageFormat.JPEG)
        self.assertEqual(actual.jpeg_quality, 87)
        self.assertEqual(
            actual.hotkeys[CaptureMode.CU5].sequence,
            "Ctrl+Alt+5",
        )
        self.assertIn(AfterCaptureTask.COPY_CLIPBOARD, actual.after_capture_tasks)

    def test_print_screen_compatibility_alias_is_visible_to_qt(self) -> None:
        page = ScreenshotSettingsPage(ScreenshotSettings())
        try:
            self.assertEqual(
                page.hotkey_edits[CaptureMode.REGION]
                .keySequence()
                .toString(QKeySequence.SequenceFormat.PortableText),
                "Print",
            )
        finally:
            page.close()

    def test_menu_key_is_preserved_by_the_settings_editor(self) -> None:
        page = ScreenshotSettingsPage(ScreenshotSettings())
        try:
            page.hotkey_edits[CaptureMode.REGION].setKeySequence(
                QKeySequence(Qt.Key.Key_Menu)
            )
            actual = page.settings()
        finally:
            page.close()

        self.assertEqual(actual.hotkeys[CaptureMode.REGION], HotkeyBinding("Menu"))

    def test_switching_formats_preserves_each_formats_quality(self) -> None:
        page = ScreenshotSettingsPage(
            ScreenshotSettings(
                image_format=ImageFormat.PNG,
                png_compression=6,
                jpeg_quality=92,
                webp_quality=81,
            )
        )
        try:
            page.image_format_combo.setCurrentIndex(
                page.image_format_combo.findData(ImageFormat.JPEG.value)
            )
            self.assertEqual(page.quality_spin.value(), 92)
            page.quality_spin.setValue(88)
            page.image_format_combo.setCurrentIndex(
                page.image_format_combo.findData(ImageFormat.WEBP.value)
            )
            self.assertEqual(page.quality_spin.value(), 81)
            page.quality_spin.setValue(79)
            page.image_format_combo.setCurrentIndex(
                page.image_format_combo.findData(ImageFormat.JPEG.value)
            )
            self.assertEqual(page.quality_spin.value(), 88)
            actual = page.settings()
        finally:
            page.close()

        self.assertEqual(actual.jpeg_quality, 88)
        self.assertEqual(actual.webp_quality, 79)

    def test_cu5_diagnostic_signal_and_status_are_explicit(self) -> None:
        page = ScreenshotSettingsPage(
            ScreenshotSettings(cu5_diagnostics_enabled=True)
        )
        requests: list[bool] = []
        page.cu5DiagnosticRequested.connect(lambda: requests.append(True))
        try:
            page.cu5_diagnostic_button.click()
            page.set_cu5_diagnostic_status(
                "已识别 768×576 视频区域",
                success=True,
            )
            self.assertEqual(requests, [True])
            self.assertIn("768×576", page.cu5_status_label.text())
            self.assertTrue(page.cu5_status_label.property("diagnosticSuccess"))
            self.assertFalse(hasattr(page, "cu5_diagnostics_checkbox"))
            self.assertTrue(page.settings().cu5_diagnostics_enabled)
        finally:
            page.close()


if __name__ == "__main__":
    unittest.main()
