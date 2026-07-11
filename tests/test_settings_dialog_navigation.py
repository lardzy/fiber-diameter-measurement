from __future__ import annotations

import os
from pathlib import Path
import sys
import unittest


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    from PySide6.QtWidgets import QApplication, QDialogButtonBox

    from fdm.settings import AppSettings
    from fdm.ui.dialogs import SettingsDialog

    PYSIDE_AVAILABLE = True
except ModuleNotFoundError:
    PYSIDE_AVAILABLE = False


@unittest.skipUnless(PYSIDE_AVAILABLE, "PySide6 is not installed")
class SettingsDialogNavigationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_uses_seven_professional_preference_categories(self) -> None:
        dialog = SettingsDialog(AppSettings(), document=None)
        try:
            labels = [
                dialog._settings_navigation.item(index).text()  # noqa: SLF001
                for index in range(dialog._settings_navigation.count())  # noqa: SLF001
            ]
            self.assertEqual(
                labels,
                [
                    "常规",
                    "测量与显示",
                    "标注与比例尺",
                    "图像与智能分析",
                    "面积识别",
                    "采集与数字切片",
                    "导出与模板",
                ],
            )
            self.assertNotIn("当前图片", labels)
            self.assertEqual(dialog._settings_pages.count(), 7)  # noqa: SLF001
            self.assertEqual(dialog._settings_page_title.text(), "常规")  # noqa: SLF001
        finally:
            dialog.close()

    def test_search_filters_and_locates_matching_category(self) -> None:
        dialog = SettingsDialog(AppSettings(), document=None)
        try:
            dialog._settings_search.setText("原始记录")  # noqa: SLF001
            self.app.processEvents()
            visible_labels = [
                item.text()
                for item in dialog._settings_navigation_items  # noqa: SLF001
                if not item.isHidden()
            ]
            self.assertEqual(visible_labels, ["导出与模板"])
            self.assertEqual(dialog._settings_page_title.text(), "导出与模板")  # noqa: SLF001
            self.assertEqual(dialog._settings_pages.currentIndex(), 6)  # noqa: SLF001

            dialog._settings_search.setText("不存在的设置项")  # noqa: SLF001
            self.app.processEvents()
            self.assertTrue(all(item.isHidden() for item in dialog._settings_navigation_items))  # noqa: SLF001
            self.assertFalse(dialog._settings_search_empty.isHidden())  # noqa: SLF001
        finally:
            dialog.close()

    def test_buttons_are_explicitly_localized(self) -> None:
        dialog = SettingsDialog(AppSettings(), document=None)
        try:
            expected = {
                QDialogButtonBox.StandardButton.Ok: "确定",
                QDialogButtonBox.StandardButton.Cancel: "取消",
                QDialogButtonBox.StandardButton.Apply: "应用",
            }
            for standard_button, text in expected.items():
                button = dialog.button_box.button(standard_button)
                self.assertIsNotNone(button)
                assert button is not None
                self.assertEqual(button.text(), text)
            self.assertEqual(dialog._restore_page_defaults_button.text(), "恢复本页默认值")  # noqa: SLF001
        finally:
            dialog.close()

    def test_restore_current_page_uses_professional_clean_profile_defaults(self) -> None:
        dialog = SettingsDialog(AppSettings(), document=None)
        try:
            dialog._settings_navigation.setCurrentRow(1)  # noqa: SLF001
            dialog._apply_button_color(dialog._measurement_label_color, "#00FF00")  # noqa: SLF001
            dialog._measurement_label_background.setChecked(False)  # noqa: SLF001
            dialog._restore_page_defaults_button.click()  # noqa: SLF001

            self.assertEqual(
                dialog._measurement_label_color.property("color_value"),  # noqa: SLF001
                AppSettings().measurement_label_color,
            )
            self.assertTrue(dialog._measurement_label_background.isChecked())  # noqa: SLF001
            self.assertIn("尚未应用", dialog._settings_page_description.text())  # noqa: SLF001
        finally:
            dialog.close()

    def test_measurement_page_has_live_style_preview(self) -> None:
        dialog = SettingsDialog(AppSettings(), document=None)
        try:
            dialog._settings_navigation.setCurrentRow(1)  # noqa: SLF001
            preview = dialog._measurement_style_preview  # noqa: SLF001
            self.assertTrue(preview._show_label)  # noqa: SLF001
            self.assertEqual(preview._label_color.name().upper(), "#F4F1DE")  # noqa: SLF001
            self.assertEqual(preview._line_color.name().upper(), "#2A9D8F")  # noqa: SLF001
            dialog._measurement_label_decimals.setValue(4)  # noqa: SLF001
            dialog._measurement_label_background.setChecked(False)  # noqa: SLF001
            self.assertEqual(preview._decimals, 4)  # noqa: SLF001
            self.assertFalse(preview._background_enabled)  # noqa: SLF001
            preview.resize(480, 100)
            self.assertFalse(preview.grab().isNull())
            self.assertEqual(
                dialog.app_settings().measurement_label_font_family,
                AppSettings().measurement_label_font_family,
            )
        finally:
            dialog.close()

    def test_show_applies_preferred_size_clamped_to_available_screen(self) -> None:
        dialog = SettingsDialog(AppSettings(), document=None)
        try:
            dialog.show()
            self.app.processEvents()
            available = dialog.screen().availableGeometry()
            self.assertEqual(dialog.width(), max(1, min(900, available.width() - 32)))
            self.assertEqual(dialog.height(), max(1, min(640, available.height() - 32)))
            self.assertGreaterEqual(dialog.button_box.y(), dialog._settings_pages.y())  # noqa: SLF001
        finally:
            dialog.close()


if __name__ == "__main__":
    unittest.main()
