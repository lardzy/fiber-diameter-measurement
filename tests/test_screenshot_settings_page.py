from __future__ import annotations

import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PySide6.QtCore import QCoreApplication, QEvent, Qt
from PySide6.QtGui import QKeyEvent, QKeySequence
from PySide6.QtWidgets import QApplication
from shiboken6 import isValid as is_qobject_valid

from fdm.screenshot_settings import (
    AfterCaptureTask,
    HotkeyBinding,
    ImageFormat,
    ScreenshotSettings,
)
from fdm.services.screenshot_capture import CaptureMode
from fdm.ui.screenshot_settings_page import ScreenshotSettingsPage
from fdm.ui.widgets import NoWheelComboBox


class ScreenshotSettingsPageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self._pages_before = {
            widget for widget in self.app.allWidgets()
            if isinstance(widget, ScreenshotSettingsPage)
        }

    def tearDown(self) -> None:
        # close() hides a QWidget; signal/menu ownership can retain the closed
        # page and its theme listeners. Destroy only pages made by this test.
        pages = [
            widget for widget in self.app.allWidgets()
            if isinstance(widget, ScreenshotSettingsPage)
            and widget not in self._pages_before
        ]
        for page in pages:
            if is_qobject_valid(page):
                page.close()
                page.deleteLater()
                QCoreApplication.sendPostedEvents(page, QEvent.Type.DeferredDelete)
        self._pages_before.clear()

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

    def test_middle_dot_key_is_preserved_by_the_settings_editor(self) -> None:
        page = ScreenshotSettingsPage(ScreenshotSettings())
        try:
            page.hotkey_edits[CaptureMode.REGION].setKeySequence(QKeySequence("·"))
            actual = page.settings()
        finally:
            page.close()

        self.assertEqual(actual.hotkeys[CaptureMode.REGION], HotkeyBinding("·"))

    def test_hotkey_hint_lists_extended_physical_keys(self) -> None:
        page = ScreenshotSettingsPage(ScreenshotSettings())
        try:
            hints = [
                label.text()
                for label in page.findChildren(type(page.agent_status_label))
            ]
        finally:
            page.close()

        self.assertTrue(any("Print Screen" in text and "·" in text for text in hints))

    def test_print_screen_release_records_windows_identity_and_preserves_modifiers(self) -> None:
        page = ScreenshotSettingsPage()
        edit = page.hotkey_edits[CaptureMode.REGION]
        try:
            event = QKeyEvent(
                QEvent.Type.KeyRelease, Qt.Key.Key_Print, Qt.KeyboardModifier.ControlModifier,
                0x37, 0x2C, 0, "", False, 1,
            )
            with patch("fdm.ui.screenshot_hotkey_edit.sys.platform", "win32"):
                QApplication.sendEvent(edit, event)
            self.assertEqual(
                page.settings().hotkeys[CaptureMode.REGION],
                HotkeyBinding("Ctrl+Print", True, 0x2C, 2),
            )
            edit.setKeySequence(QKeySequence("Ctrl+B"))
            self.assertEqual(edit.binding(), HotkeyBinding("Ctrl+B"))
        finally:
            page.close()

    def test_native_layout_key_survives_roundtrip_and_nonwindows_edit_clears_it(self) -> None:
        original = HotkeyBinding("Ctrl+Ö", True, 0xBA, 2)
        page = ScreenshotSettingsPage(ScreenshotSettings(hotkeys={CaptureMode.REGION: original}))
        edit = page.hotkey_edits[CaptureMode.REGION]
        try:
            self.assertEqual(page.settings().hotkeys[CaptureMode.REGION], original)
            event = QKeyEvent(
                QEvent.Type.KeyPress, Qt.Key.Key_A, Qt.KeyboardModifier.ControlModifier,
                38, 38, 0, "a", False, 1,
            )
            with patch("fdm.ui.screenshot_hotkey_edit.sys.platform", "darwin"):
                QApplication.sendEvent(edit, event)
            self.assertEqual(edit.binding(), HotkeyBinding("Ctrl+A"))
        finally:
            page.close()

    def test_print_press_then_release_does_not_replace_pressed_modifier_snapshot(self) -> None:
        page = ScreenshotSettingsPage()
        edit = page.hotkey_edits[CaptureMode.REGION]
        try:
            with patch("fdm.ui.screenshot_hotkey_edit.sys.platform", "win32"):
                QApplication.sendEvent(edit, QKeyEvent(
                    QEvent.Type.KeyPress, Qt.Key.Key_SysReq, Qt.KeyboardModifier.AltModifier,
                    0x37, 0x2C, 0, "", False, 1,
                ))
                QApplication.sendEvent(edit, QKeyEvent(
                    QEvent.Type.KeyRelease, Qt.Key.Key_Print, Qt.KeyboardModifier.NoModifier,
                    0x37, 0x2C, 0, "", False, 1,
                ))
            self.assertEqual(edit.binding(), HotkeyBinding("Alt+Print", True, 0x2C, 1))
        finally:
            page.close()

    def test_recording_same_display_text_without_windows_vk_discards_stale_identity(self) -> None:
        original = HotkeyBinding("Ctrl+A", True, 0x41, 2)
        page = ScreenshotSettingsPage(ScreenshotSettings(hotkeys={CaptureMode.REGION: original}))
        edit = page.hotkey_edits[CaptureMode.REGION]
        try:
            with patch("fdm.ui.screenshot_hotkey_edit.sys.platform", "darwin"):
                QApplication.sendEvent(edit, QKeyEvent(
                    QEvent.Type.KeyPress, Qt.Key.Key_A, Qt.KeyboardModifier.ControlModifier,
                    38, 38, 0, "a", False, 1,
                ))
            self.assertEqual(edit.binding(), HotkeyBinding("Ctrl+A"))
        finally:
            page.close()

    def test_recording_focus_moves_between_editors_without_releasing_pause(self) -> None:
        page = ScreenshotSettingsPage()
        changes: list[bool] = []
        page.hotkeyRecordingChanged.connect(changes.append)
        try:
            page.show()
            self.app.processEvents()
            page.hotkey_edits[CaptureMode.REGION].setFocus()
            self.app.processEvents()
            self.assertEqual(changes, [True])
            self.assertFalse(page.retry_hotkeys_button.isEnabled())
            page.hotkey_edits[CaptureMode.WINDOW].setFocus()
            self.app.processEvents()
            self.assertEqual(changes, [True])
            page.output_directory_edit.setFocus()
            self.app.processEvents()
            self.assertEqual(changes, [True, False])
            page.hotkey_edits[CaptureMode.REGION].setFocus()
            self.app.processEvents()
            page.hide()
            self.assertEqual(changes, [True, False, True, False])
        finally:
            page.close()

    def test_registration_diagnostics_show_failed_request_retained_binding_and_test_press(self) -> None:
        page = ScreenshotSettingsPage()
        retries: list[bool] = []
        page.retryHotkeysRequested.connect(lambda: retries.append(True))
        try:
            page.set_hotkey_diagnostics([
                {"mode": "region", "requested": "Ctrl+F9", "effective": "Print",
                 "state": "failed", "error_code": 1409, "error": "native error",
                 "last_received_at": "2026-10-14T12:30:45"},
                {"mode": "bogus", "state": "failed"}, None,
            ])
            label = page.hotkey_status_labels[CaptureMode.REGION]
            self.assertIn("注册失败", label.text())
            self.assertIn("当前生效：Print", label.text())
            self.assertIn("未生效的请求：Ctrl+F9", label.text())
            self.assertIn("1409", label.text())
            self.assertIn("已收到试按／触发", label.text())
            self.assertEqual(label.property("hotkeyState"), "failed")
            page.retry_hotkeys_button.click()
            self.assertEqual(retries, [True])
        finally:
            page.close()

    def test_deleting_active_editor_releases_recording_pause(self) -> None:
        page = ScreenshotSettingsPage()
        changes: list[bool] = []
        page.hotkeyRecordingChanged.connect(changes.append)
        try:
            page.show()
            self.app.processEvents()
            edit = page.hotkey_edits[CaptureMode.REGION]
            edit.setFocus()
            self.app.processEvents()
            self.assertEqual(changes, [True])
            edit.deleteLater()
            QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
            self.app.processEvents()
            self.assertEqual(changes, [True, False])
        finally:
            page.close()

    def test_modifier_only_autorepeat_and_altgr_do_not_overwrite_binding(self) -> None:
        page = ScreenshotSettingsPage()
        edit = page.hotkey_edits[CaptureMode.REGION]
        try:
            original = edit.binding()
            for event in (
                QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Control, Qt.KeyboardModifier.ControlModifier),
                QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_A, Qt.KeyboardModifier.NoModifier, "a", True),
                QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Q, Qt.KeyboardModifier.GroupSwitchModifier, "@"),
            ):
                QApplication.sendEvent(edit, event)
            self.assertEqual(edit.binding(), original)
            self.assertIn("AltGr", page.hotkey_recording_hint.text())
        finally:
            page.close()

    def test_internal_line_editor_records_single_combination_and_defaults_clear_metadata(self) -> None:
        page = ScreenshotSettingsPage()
        edit = page.hotkey_edits[CaptureMode.REGION]
        try:
            with patch("fdm.ui.screenshot_hotkey_edit.sys.platform", "win32"):
                QApplication.sendEvent(edit._line_edit, QKeyEvent(
                    QEvent.Type.KeyPress, Qt.Key.Key_F8, Qt.KeyboardModifier.ControlModifier,
                    0x42, 0x77, 0, "", False, 1,
                ))
            self.assertEqual(edit.binding(), HotkeyBinding("Ctrl+F8", True, 0x77, 2))
            page.restore_defaults()
            self.assertEqual(edit.binding(), HotkeyBinding("Print"))
        finally:
            page.close()

    def test_special_key_menu_configures_all_print_combinations_with_windows_metadata(self) -> None:
        expected = {
            "Print": 0, "Alt+Print": 1, "Ctrl+Print": 2, "Shift+Print": 4,
            "Ctrl+Shift+Print": 6, "Ctrl+Alt+Print": 3,
            "Alt+Shift+Print": 5, "Ctrl+Alt+Shift+Print": 7,
        }
        page = ScreenshotSettingsPage()
        try:
            for mode, button in page.hotkey_special_buttons.items():
                actions = button.menu().actions()
                self.assertEqual({action.data() for action in actions}, set(expected))
                self.assertEqual(button.objectName(), f"screenshotSpecialHotkey_{mode.value}")
                for action in actions:
                    with self.subTest(mode=mode, sequence=action.data()):
                        with patch("fdm.ui.screenshot_settings_page.sys.platform", "win32"):
                            action.trigger()
                        self.assertEqual(
                            page.settings().hotkeys[mode],
                            HotkeyBinding(action.data(), True, 0x2C, expected[action.data()]),
                        )
        finally:
            page.close()

    def test_special_key_menu_on_mac_discards_previous_windows_metadata(self) -> None:
        page = ScreenshotSettingsPage(ScreenshotSettings(hotkeys={
            CaptureMode.REGION: HotkeyBinding("Ctrl+Ö", True, 0xBA, 2),
        }))
        try:
            action = next(
                action for action in page.hotkey_special_buttons[CaptureMode.REGION].menu().actions()
                if action.data() == "Ctrl+Shift+Print"
            )
            with patch("fdm.ui.screenshot_settings_page.sys.platform", "darwin"):
                action.trigger()
            self.assertEqual(page.settings().hotkeys[CaptureMode.REGION], HotkeyBinding("Ctrl+Shift+Print"))
        finally:
            page.close()

    def test_special_key_menu_reenables_cleared_binding_without_enabling_screenshot_service(self) -> None:
        page = ScreenshotSettingsPage(ScreenshotSettings(enabled=False))
        edit = page.hotkey_edits[CaptureMode.REGION]
        try:
            actions = page.hotkey_special_buttons[CaptureMode.REGION].menu().actions()
            with patch("fdm.ui.screenshot_settings_page.sys.platform", "win32"):
                actions[0].trigger()
            self.assertEqual(edit.binding().native_virtual_key, 0x2C)
            edit.setKeySequence(QKeySequence("Ctrl+F8"))
            self.assertEqual(edit.binding(), HotkeyBinding("Ctrl+F8"))
            edit.clear()
            self.assertFalse(edit.binding().enabled)
            with patch("fdm.ui.screenshot_settings_page.sys.platform", "win32"):
                next(action for action in actions if action.data() == "Alt+Print").trigger()
            self.assertEqual(edit.binding(), HotkeyBinding("Alt+Print", True, 0x2C, 1))
            self.assertFalse(page.settings().enabled)
            self.assertEqual(page.settings().hotkeys[CaptureMode.WINDOW], HotkeyBinding("Alt+Print"))
        finally:
            page.close()

    def test_special_key_menu_action_ends_recording_without_synthesizing_input(self) -> None:
        page = ScreenshotSettingsPage()
        changes: list[bool] = []
        page.hotkeyRecordingChanged.connect(changes.append)
        try:
            page.show()
            self.app.processEvents()
            edit = page.hotkey_edits[CaptureMode.REGION]
            edit.setFocus()
            self.app.processEvents()
            self.assertEqual(changes, [True])
            button = page.hotkey_special_buttons[CaptureMode.REGION]
            with patch.object(edit, "keyPressEvent", side_effect=AssertionError("菜单不应模拟按键")):
                with patch.object(edit, "keyReleaseEvent", side_effect=AssertionError("菜单不应模拟按键")):
                    with patch("fdm.ui.screenshot_settings_page.sys.platform", "win32"):
                        button.menu().actions()[0].trigger()
            self.app.processEvents()
            self.assertEqual(changes, [True, False])
            self.assertFalse(edit.is_recording)
            self.assertTrue(button.hasFocus())
            self.assertEqual(edit.binding(), HotkeyBinding("Print", True, 0x2C, 0))
            self.assertEqual(page.settings().hotkeys[CaptureMode.REGION], HotkeyBinding("Print", True, 0x2C, 0))
        finally:
            page.close()

    def test_same_value_completion_signal_keeps_native_binding_but_text_change_does_not(self) -> None:
        page = ScreenshotSettingsPage()
        edit = page.hotkey_edits[CaptureMode.REGION]
        binding = HotkeyBinding("Ctrl+Alt+Shift+F9", True, 0x78, 7)
        try:
            edit.set_binding(binding)
            edit.keySequenceChanged.emit(edit.keySequence())
            self.assertEqual(edit.binding(), binding)
            edit.setKeySequence(QKeySequence("Ctrl+F8"))
            self.assertEqual(edit.binding(), HotkeyBinding("Ctrl+F8"))
        finally:
            page.close()

    def test_recorded_native_binding_survives_actual_focus_out(self) -> None:
        page = ScreenshotSettingsPage()
        edit = page.hotkey_edits[CaptureMode.REGION]
        try:
            page.show()
            self.app.processEvents()
            edit.setFocus()
            self.app.processEvents()
            with patch("fdm.ui.screenshot_hotkey_edit.sys.platform", "win32"):
                QApplication.sendEvent(edit, QKeyEvent(
                    QEvent.Type.KeyPress, Qt.Key.Key_F9,
                    Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.AltModifier | Qt.KeyboardModifier.ShiftModifier,
                    0x43, 0x78, 0, "", False, 1,
                ))
            expected = HotkeyBinding("Ctrl+Alt+Shift+F9", True, 0x78, 7)
            self.assertEqual(edit.binding(), expected)
            page.output_directory_edit.setFocus()
            self.app.processEvents()
            self.assertFalse(edit.hasFocus())
            self.assertEqual(edit.binding(), expected)
            self.assertEqual(page.settings().hotkeys[CaptureMode.REGION], expected)
        finally:
            page.close()

    def test_disabled_nonempty_binding_roundtrips_and_focus_completion_does_not_enable_it(self) -> None:
        original = HotkeyBinding("Print", False, 0x2C, 0)
        page = ScreenshotSettingsPage(ScreenshotSettings(hotkeys={CaptureMode.REGION: original}))
        edit = page.hotkey_edits[CaptureMode.REGION]
        try:
            self.assertEqual(page.settings().hotkeys[CaptureMode.REGION], original)
            edit.keySequenceChanged.emit(edit.keySequence())
            self.assertEqual(edit.binding(), original)
            page.show()
            self.app.processEvents()
            edit.setFocus()
            self.app.processEvents()
            page.output_directory_edit.setFocus()
            self.app.processEvents()
            self.assertEqual(page.settings().hotkeys[CaptureMode.REGION], original)
            edit.set_binding(HotkeyBinding("Print", False))
            self.assertEqual(edit.binding(), HotkeyBinding("Print", False))
        finally:
            page.close()

    def test_actual_recording_or_text_change_reenables_a_disabled_nonempty_binding(self) -> None:
        page = ScreenshotSettingsPage()
        edit = page.hotkey_edits[CaptureMode.REGION]
        try:
            for original in ("Ctrl+F8", "Ctrl+F9"):
                with self.subTest(original=original):
                    edit.set_binding(HotkeyBinding(original, False))
                    with patch("fdm.ui.screenshot_hotkey_edit.sys.platform", "win32"):
                        QApplication.sendEvent(edit, QKeyEvent(
                            QEvent.Type.KeyPress, Qt.Key.Key_F9, Qt.KeyboardModifier.ControlModifier,
                            0x43, 0x78, 0, "", False, 1,
                        ))
                    self.assertEqual(edit.binding(), HotkeyBinding("Ctrl+F9", True, 0x78, 2))
            edit.set_binding(HotkeyBinding("Print", False))
            edit.setKeySequence(QKeySequence("Ctrl+F8"))
            self.assertEqual(edit.binding(), HotkeyBinding("Ctrl+F8", True))
            edit.clear()
            self.assertEqual(edit.binding(), HotkeyBinding("", False))
        finally:
            page.close()

    def test_special_key_selection_does_not_enable_other_disabled_print_binding(self) -> None:
        page = ScreenshotSettingsPage(ScreenshotSettings(hotkeys={
            CaptureMode.REGION: HotkeyBinding("Print", False),
            CaptureMode.FULL_SCREEN: HotkeyBinding("Ctrl+Alt+Shift+F9", True, 0x78, 7),
        }))
        try:
            action = next(
                action for action in page.hotkey_special_buttons[CaptureMode.FULL_SCREEN].menu().actions()
                if action.data() == "Print"
            )
            with patch("fdm.ui.screenshot_settings_page.sys.platform", "win32"):
                action.trigger()
            actual = page.settings().hotkeys
            self.assertEqual(actual[CaptureMode.REGION], HotkeyBinding("Print", False))
            self.assertEqual(actual[CaptureMode.FULL_SCREEN], HotkeyBinding("Print", True, 0x2C, 0))
        finally:
            page.close()

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

    def test_cu_preview_candidates_default_to_recommended_and_require_apply(self) -> None:
        page = ScreenshotSettingsPage(ScreenshotSettings())
        selected = {
            "process_name": "cu-6.exe",
            "class_name": "static",
            "control_id": 1501,
        }
        requests: list[dict[str, object]] = []
        page.cu5CandidateSelectionRequested.connect(requests.append)
        try:
            page.set_cu5_candidates(
                [
                    {
                        "title": "",
                        "class_name": "Static",
                        "process_name": "CU-6.exe",
                        "rect": {"x": 100, "y": 80, "width": 768, "height": 576},
                        "score": 260.0,
                        "selector": selected,
                    },
                    {
                        "title": "实时预览",
                        "class_name": "CWndForSDK",
                        "process_name": "CU-6.exe",
                        "rect": {"x": 120, "y": 90, "width": 640, "height": 480},
                        "score": 190.0,
                        "selector": {
                            "process_name": "cu-6.exe",
                            "class_name": "cwndforsdk",
                            "control_id": 1601,
                        },
                    },
                ],
                selected_selector=selected,
            )
            self.assertIsInstance(page.cu5_candidate_combo, NoWheelComboBox)
            self.assertEqual(page.cu5_candidate_combo.count(), 2)
            self.assertIn("推荐", page.cu5_candidate_combo.currentText())
            self.assertEqual(requests, [])

            page.cu5_candidate_combo.setCurrentIndex(1)
            self.assertEqual(requests, [])
            page.cu5_candidate_apply_button.click()
            self.assertEqual(requests[-1]["control_id"], 1601)
            self.assertIn("CU 系列", page.cu5_diagnostic_button.text())
        finally:
            page.close()

    def test_malformed_preview_candidate_is_ignored_without_breaking_settings(self) -> None:
        page = ScreenshotSettingsPage(ScreenshotSettings())
        try:
            page.set_cu5_candidates(
                [
                    {
                        "rect": {
                            "x": object(),
                            "y": 0,
                            "width": "not-a-number",
                            "height": 576,
                        },
                        "score": "--",
                        "selector": {
                            "process_name": "cu-6.exe",
                            "class_name": "static",
                        },
                    }
                ],
                selected_selector=None,
            )
            self.assertEqual(page.cu5_candidate_combo.count(), 1)
            self.assertFalse(page.cu5_candidate_combo.isEnabled())
            self.assertFalse(page.cu5_candidate_apply_button.isEnabled())
        finally:
            page.close()


if __name__ == "__main__":
    unittest.main()
