from __future__ import annotations

import os
from pathlib import Path
import sys
import unittest


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    from PySide6.QtCore import QPoint
    from PySide6.QtWidgets import QApplication, QDialogButtonBox

    from fdm.geometry import Point
    from fdm.models import (
        ImageDocument,
        OverlayTextAnchorAlignment,
        OverlayTextSizeSpace,
    )
    from fdm.settings import AppSettings, MeasurementLabelStyleSettings
    from fdm.ui.dialogs import SettingsDialog

    PYSIDE_AVAILABLE = True
except ModuleNotFoundError:
    PYSIDE_AVAILABLE = False


@unittest.skipUnless(PYSIDE_AVAILABLE, "PySide6 is not installed")
class SettingsDialogNavigationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_uses_eight_professional_preference_categories(self) -> None:
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
                    "截图工具",
                    "导出与模板",
                ],
            )
            self.assertNotIn("当前图片", labels)
            self.assertEqual(dialog._settings_pages.count(), 8)  # noqa: SLF001
            self.assertEqual(dialog._settings_page_title.text(), "常规")  # noqa: SLF001
        finally:
            dialog.close()

    def test_annotation_page_persists_new_text_size_space_and_anchor(self) -> None:
        dialog = SettingsDialog(AppSettings(), document=None)
        try:
            dialog._settings_navigation.setCurrentRow(2)  # noqa: SLF001
            dialog._text_size_space_combo.setCurrentIndex(  # noqa: SLF001
                dialog._text_size_space_combo.findData(  # noqa: SLF001
                    OverlayTextSizeSpace.LEGACY_OUTPUT_PX
                )
            )
            dialog._text_anchor_combo.setCurrentIndex(  # noqa: SLF001
                dialog._text_anchor_combo.findData(  # noqa: SLF001
                    OverlayTextAnchorAlignment.BOTTOM_RIGHT
                )
            )

            saved = dialog.app_settings()

            self.assertEqual(
                saved.text_size_space,
                OverlayTextSizeSpace.LEGACY_OUTPUT_PX,
            )
            self.assertEqual(
                saved.text_anchor_alignment,
                OverlayTextAnchorAlignment.BOTTOM_RIGHT,
            )
        finally:
            dialog.close()

    def test_navigation_rows_follow_font_metrics_without_overlap(self) -> None:
        dialog = SettingsDialog(AppSettings(), document=None)
        try:
            dialog.resize(900, 640)
            dialog.show()
            self.app.processEvents()

            navigation = dialog._settings_navigation  # noqa: SLF001
            minimum_height = navigation.fontMetrics().height() + 16
            rects = [
                navigation.visualItemRect(navigation.item(index))
                for index in range(navigation.count())
            ]
            self.assertTrue(rects)
            self.assertTrue(all(rect.height() >= minimum_height for rect in rects))
            self.assertTrue(
                all(
                    current.bottom() < following.top()
                    for current, following in zip(rects, rects[1:])
                )
            )
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
            self.assertEqual(dialog._settings_pages.currentIndex(), 7)  # noqa: SLF001

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

    def test_screenshot_page_keeps_settings_outside_app_settings_payload(self) -> None:
        from fdm.screenshot_settings import ImageFormat, ScreenshotSettings

        dialog = SettingsDialog(
            AppSettings(),
            document=None,
            screenshot_settings=ScreenshotSettings(
                enabled=True,
                autostart=True,
                output_directory="/tmp/fdm-screenshots",
                image_format=ImageFormat.JPEG,
                jpeg_quality=84,
            ),
        )
        try:
            dialog._settings_navigation.setCurrentRow(6)  # noqa: SLF001
            screenshot = dialog.screenshot_settings()
            app_settings = dialog.app_settings()

            self.assertTrue(screenshot.enabled)
            self.assertTrue(screenshot.autostart)
            self.assertEqual(screenshot.output_directory, "/tmp/fdm-screenshots")
            self.assertEqual(screenshot.image_format, ImageFormat.JPEG)
            self.assertNotIn("screenshot", app_settings.to_dict())
        finally:
            dialog.close()

    def test_footer_keeps_restore_button_away_from_left_edge(self) -> None:
        dialog = SettingsDialog(AppSettings(), document=None)
        try:
            dialog.resize(900, 640)
            dialog.show()
            self.app.processEvents()
            restore_left = dialog._restore_page_defaults_button.mapTo(dialog, QPoint(0, 0)).x()  # noqa: SLF001
            ok_button = dialog.button_box.button(QDialogButtonBox.StandardButton.Ok)
            ok_right = ok_button.mapTo(dialog, QPoint(ok_button.width(), 0)).x()
            self.assertGreaterEqual(restore_left, 12)
            self.assertLessEqual(ok_right, dialog.width() - 12)
        finally:
            dialog.close()

    def test_annotation_page_exposes_current_image_scale_anchor_pick(self) -> None:
        document = ImageDocument(id="image", path="/tmp/image.png", image_size=(100, 80))
        document.scale_overlay_anchor = Point(24, 36)
        dialog = SettingsDialog(AppSettings(), document=document)
        try:
            dialog._settings_navigation.setCurrentRow(2)  # noqa: SLF001
            self.assertTrue(dialog._scale_anchor_pick_button.isEnabled())  # noqa: SLF001
            self.assertEqual(dialog._scale_anchor_status_label.text(), "当前锚点：(24.0, 36.0)")  # noqa: SLF001
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
        dialog = SettingsDialog(
            AppSettings(
                length_measurement_label_style=MeasurementLabelStyleSettings(decimals=2),
                area_measurement_label_style=MeasurementLabelStyleSettings(
                    decimals=5,
                    color="#88AA44",
                ),
            ),
            document=None,
        )
        try:
            dialog._settings_navigation.setCurrentRow(1)  # noqa: SLF001
            length_preview = dialog._length_measurement_style_preview  # noqa: SLF001
            area_preview = dialog._area_measurement_style_preview  # noqa: SLF001
            self.assertTrue(length_preview._show_label)  # noqa: SLF001
            self.assertEqual(length_preview._label_color.name().upper(), "#FF0000")  # noqa: SLF001
            self.assertEqual(area_preview._label_color.name().upper(), "#88AA44")  # noqa: SLF001
            self.assertEqual(area_preview._metric, "area")  # noqa: SLF001
            dialog._length_measurement_label_decimals.setValue(4)  # noqa: SLF001
            dialog._length_measurement_label_background.setChecked(False)  # noqa: SLF001
            dialog._area_measurement_label_decimals.setValue(7)  # noqa: SLF001
            self.assertEqual(length_preview._decimals, 4)  # noqa: SLF001
            self.assertEqual(area_preview._decimals, 7)  # noqa: SLF001
            self.assertFalse(length_preview._background_enabled)  # noqa: SLF001
            self.assertTrue(area_preview._background_enabled)  # noqa: SLF001
            length_preview.resize(480, 100)
            area_preview.resize(480, 100)
            self.assertFalse(length_preview.grab().isNull())
            self.assertFalse(area_preview.grab().isNull())
            self.assertEqual(
                dialog.app_settings().length_measurement_label_style.font_family,
                AppSettings().length_measurement_label_style.font_family,
            )
            self.assertEqual(dialog.app_settings().area_measurement_label_style.decimals, 7)
        finally:
            dialog.close()

    def test_area_result_label_is_off_only_for_fresh_default_settings(self) -> None:
        default_dialog = SettingsDialog(AppSettings(), document=None)
        configured_dialog = SettingsDialog(
            AppSettings(
                area_measurement_label_style=MeasurementLabelStyleSettings(
                    enabled=True
                )
            ),
            document=None,
        )
        try:
            self.assertFalse(default_dialog._show_area_measurement_labels.isChecked())  # noqa: SLF001
            self.assertTrue(configured_dialog._show_area_measurement_labels.isChecked())  # noqa: SLF001
        finally:
            default_dialog.close()
            configured_dialog.close()

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
