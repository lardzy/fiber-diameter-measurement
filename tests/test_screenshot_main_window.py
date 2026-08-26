from __future__ import annotations

import os
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PySide6.QtWidgets import QApplication, QMenu, QMessageBox

from fdm.screenshot_protocol import IPCResponse
from fdm.screenshot_settings import (
    ScreenshotSettings,
    UnsupportedScreenshotSettingsVersion,
)
from fdm.services.screenshot_agent_client import (
    ScreenshotAgentCommandError,
    ScreenshotAgentStatus,
)
from fdm.services.screenshot_capture import CaptureMode
from fdm.services.screenshot_capture import CaptureRect
from fdm.ui.main_window import MainWindow


class _FakeScreenshotClient:
    def __init__(self) -> None:
        self.started = 0
        self.running = False
        self.updated: list[dict[str, object]] = []
        self.captures: list[tuple[CaptureMode, dict[str, object]]] = []
        self.shutdown_calls = 0

    def ensure_started(self, **_kwargs) -> ScreenshotAgentStatus:
        self.started += 1
        if self.running:
            return ScreenshotAgentStatus(True, {"already_running": True})
        self.running = True
        return ScreenshotAgentStatus(True, {"started": True})

    def update_settings(self, settings, **_kwargs) -> IPCResponse:
        self.updated.append(dict(settings))
        return IPCResponse.success("update", {"accepted": True})

    def capture(self, mode, *, payload=None, **_kwargs) -> IPCResponse:
        self.captures.append((CaptureMode.parse(mode), dict(payload or {})))
        return IPCResponse.success("capture", {"accepted": True})

    def shutdown(self, **_kwargs) -> bool:
        self.shutdown_calls += 1
        self.running = False
        return True

    def status(self, **_kwargs) -> ScreenshotAgentStatus:
        return ScreenshotAgentStatus(self.running)

    def send(self, _command, **_kwargs) -> IPCResponse:
        selector = {
            "process_name": "cu-6.exe",
            "class_name": "static",
            "control_id": 1501,
        }
        return IPCResponse.success(
            "diagnose",
            {
                "rect": {"x": 10, "y": 20, "width": 768, "height": 576},
                "selector": selector,
                "candidates": [
                    {
                        "rect": {"x": 10, "y": 20, "width": 768, "height": 576},
                        "score": 250.0,
                        "class_name": "Static",
                        "process_name": "CU-6.exe",
                        "selector": selector,
                    }
                ],
            },
        )


class ScreenshotMainWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        # MainWindow persists general application settings when it closes;
        # screenshot integration tests must never touch the user's real profile.
        self._app_settings_save_patcher = patch(
            "fdm.ui.main_window.AppSettingsIO.save"
        )
        self._app_settings_save_patcher.start()

    def tearDown(self) -> None:
        self._app_settings_save_patcher.stop()

    def _window(self) -> MainWindow:
        with (
            patch(
                "fdm.ui.main_window.ScreenshotSettingsIO.load",
                return_value=ScreenshotSettings(),
            ),
            patch("fdm.ui.main_window.ScreenshotSettingsIO.save"),
            patch("fdm.ui.main_window.AppSettingsIO.save"),
        ):
            return MainWindow()

    def test_tools_menu_exposes_resident_switch_and_cu5_capture(self) -> None:
        window = self._window()
        try:
            screenshot_menu = next(
                menu
                for menu in window.menuBar().findChildren(QMenu)
                if menu.title() == "截图工具"
            )
            self.assertIn(window.screenshot_tool_action, screenshot_menu.actions())
            self.assertIn(window.screenshot_region_action, screenshot_menu.actions())
            self.assertIn(window.screenshot_cu5_action, screenshot_menu.actions())
            self.assertIn("CU 系列", window.screenshot_cu5_action.text())
            self.assertTrue(window.screenshot_tool_action.isCheckable())
            self.assertFalse(window.screenshot_tool_action.isChecked())
            with patch.object(window, "open_settings_dialog") as open_settings:
                window.settings_action.trigger()
            open_settings.assert_called_once_with()
        finally:
            window.close()

    def test_switch_starts_and_explicitly_stops_detached_companion(self) -> None:
        window = self._window()
        client = _FakeScreenshotClient()
        window._screenshot_agent_client = client  # noqa: SLF001
        try:
            with (
                patch(
                    "fdm.ui.main_window.ScreenshotSettingsIO.load",
                    side_effect=lambda: window._screenshot_settings,  # noqa: SLF001
                ),
                patch("fdm.ui.main_window.ScreenshotSettingsIO.save"),
                patch(
                    "fdm.ui.main_window.ScreenshotSettingsIO.update",
                    side_effect=lambda mutator, *_args, **_kwargs: mutator(
                        window._screenshot_settings  # noqa: SLF001
                    ),
                ),
            ):
                window.screenshot_tool_action.setChecked(True)
                self.assertEqual(client.started, 1)
                self.assertTrue(client.updated)
                self.assertTrue(window._screenshot_settings.enabled)  # noqa: SLF001

                window.screenshot_tool_action.setChecked(False)
                self.assertEqual(client.shutdown_calls, 1)
                self.assertFalse(window._screenshot_settings.enabled)  # noqa: SLF001
                self.assertFalse(window._screenshot_settings.autostart)  # noqa: SLF001
        finally:
            # Closing the measurement window must not own or terminate the
            # detached screenshot companion lifecycle.
            before_close = client.shutdown_calls
            window.close()
            self.assertEqual(client.shutdown_calls, before_close)

    def test_cu5_action_routes_the_dedicated_automatic_mode(self) -> None:
        window = self._window()
        client = _FakeScreenshotClient()
        window._screenshot_agent_client = client  # noqa: SLF001
        try:
            with (
                patch(
                    "fdm.ui.main_window.ScreenshotSettingsIO.load",
                    side_effect=lambda: window._screenshot_settings,  # noqa: SLF001
                ),
                patch("fdm.ui.main_window.ScreenshotSettingsIO.save"),
                patch(
                    "fdm.ui.main_window.ScreenshotSettingsIO.update",
                    side_effect=lambda mutator, *_args, **_kwargs: mutator(
                        window._screenshot_settings  # noqa: SLF001
                    ),
                ),
            ):
                window.screenshot_cu5_action.trigger()

            self.assertEqual(client.started, 1)
            self.assertEqual(client.captures[0][0], CaptureMode.CU5)
            self.assertTrue(client.updated[0]["enabled"])
            self.assertTrue(window._screenshot_settings.enabled)  # noqa: SLF001
        finally:
            window.close()

    def test_failed_initial_settings_sync_stops_only_the_new_agent(self) -> None:
        window = self._window()
        client = _FakeScreenshotClient()

        def reject_update(_settings, **_kwargs):
            response = IPCResponse.failure("update", "settings rejected")
            raise ScreenshotAgentCommandError("settings rejected", response=response)

        client.update_settings = reject_update  # type: ignore[method-assign]
        window._screenshot_agent_client = client  # noqa: SLF001
        try:
            with (
                patch(
                    "fdm.ui.main_window.ScreenshotSettingsIO.update",
                    side_effect=lambda mutator, *_args, **_kwargs: mutator(
                        window._screenshot_settings  # noqa: SLF001
                    ),
                ),
                patch("fdm.ui.main_window.QMessageBox.warning"),
            ):
                window.screenshot_tool_action.setChecked(True)

            self.assertEqual(client.started, 1)
            self.assertEqual(client.shutdown_calls, 1)
            self.assertFalse(client.running)
            self.assertFalse(window._screenshot_settings.enabled)  # noqa: SLF001
            self.assertFalse(window.screenshot_tool_action.isChecked())
        finally:
            window.close()

    def test_switch_save_failure_restores_the_complete_previous_settings(self) -> None:
        window = self._window()
        previous = ScreenshotSettings(
            enabled=False,
            autostart=False,
            delay_ms=1250,
            output_directory="keep-this-path",
        ).normalized()
        window._screenshot_settings = previous  # noqa: SLF001
        try:
            with (
                patch(
                    "fdm.ui.main_window.ScreenshotSettingsIO.update",
                    side_effect=OSError("disk full"),
                ),
                patch("fdm.ui.main_window.QMessageBox.warning"),
            ):
                window.screenshot_tool_action.setChecked(True)

            self.assertEqual(  # noqa: SLF001
                window._screenshot_settings.to_dict(),
                previous.to_dict(),
            )
            self.assertFalse(window.screenshot_tool_action.isChecked())
        finally:
            window.close()

    def test_failed_one_shot_stops_a_new_agent_after_settings_were_applied(self) -> None:
        window = self._window()
        client = _FakeScreenshotClient()

        def reject_capture(_mode, **_kwargs):
            response = IPCResponse.failure("capture", "capture rejected")
            raise ScreenshotAgentCommandError("capture rejected", response=response)

        client.capture = reject_capture  # type: ignore[method-assign]
        window._screenshot_agent_client = client  # noqa: SLF001
        try:
            with (
                patch(
                    "fdm.ui.main_window.ScreenshotSettingsIO.update",
                    side_effect=lambda mutator, *_args, **_kwargs: mutator(
                        window._screenshot_settings  # noqa: SLF001
                    ),
                ),
                patch("fdm.ui.main_window.QMessageBox.warning"),
            ):
                window.screenshot_region_action.trigger()

            self.assertEqual(client.shutdown_calls, 1)
            self.assertFalse(client.running)
            self.assertFalse(window._screenshot_settings.enabled)  # noqa: SLF001
            self.assertFalse(window.screenshot_tool_action.isChecked())
        finally:
            window.close()

    def test_failed_one_shot_restores_disabled_settings_on_existing_agent(self) -> None:
        window = self._window()
        client = _FakeScreenshotClient()
        client.running = True

        def reject_capture(_mode, **_kwargs):
            response = IPCResponse.failure("capture", "capture rejected")
            raise ScreenshotAgentCommandError("capture rejected", response=response)

        client.capture = reject_capture  # type: ignore[method-assign]
        window._screenshot_agent_client = client  # noqa: SLF001
        try:
            with (
                patch(
                    "fdm.ui.main_window.ScreenshotSettingsIO.update",
                    side_effect=lambda mutator, *_args, **_kwargs: mutator(
                        window._screenshot_settings  # noqa: SLF001
                    ),
                ),
                patch("fdm.ui.main_window.QMessageBox.warning"),
            ):
                window.screenshot_region_action.trigger()

            self.assertEqual(client.shutdown_calls, 0)
            self.assertTrue(client.running)
            self.assertGreaterEqual(len(client.updated), 2)
            self.assertTrue(client.updated[-2]["enabled"])
            self.assertFalse(client.updated[-1]["enabled"])
            self.assertFalse(window._screenshot_settings.enabled)  # noqa: SLF001
        finally:
            window.close()

    def test_newer_screenshot_settings_are_not_overwritten_by_main_switch(self) -> None:
        with (
            patch(
                "fdm.ui.main_window.ScreenshotSettingsIO.load",
                side_effect=UnsupportedScreenshotSettingsVersion("schema 999"),
            ),
            patch("fdm.ui.main_window.ScreenshotSettingsIO.save") as save,
            patch("fdm.ui.main_window.AppSettingsIO.save"),
            patch("fdm.ui.main_window.QMessageBox.warning") as warning,
        ):
            window = MainWindow()
            try:
                window.screenshot_tool_action.setChecked(True)
                self.assertTrue(window._screenshot_settings_read_only)  # noqa: SLF001
                self.assertFalse(window.screenshot_tool_action.isChecked())
                save.assert_not_called()
                warning.assert_called()
            finally:
                window.close()

    def test_save_detects_a_newer_schema_written_after_startup(self) -> None:
        window = self._window()
        try:
            with (
                patch(
                    "fdm.ui.main_window.ScreenshotSettingsIO.update",
                    side_effect=UnsupportedScreenshotSettingsVersion("schema 999"),
                ) as update,
                patch("fdm.ui.main_window.QMessageBox.warning") as warning,
            ):
                self.assertFalse(window._save_screenshot_settings())  # noqa: SLF001

            self.assertTrue(window._screenshot_settings_read_only)  # noqa: SLF001
            self.assertEqual(window._screenshot_settings_load_error, "schema 999")  # noqa: SLF001
            update.assert_called_once()
            warning.assert_called_once()
        finally:
            window.close()

    def test_failed_explicit_schema_replacement_restores_read_only_guard(self) -> None:
        window = self._window()
        window._screenshot_settings_read_only = True  # noqa: SLF001
        window._screenshot_settings_load_error = "schema 999"  # noqa: SLF001
        previous = window._screenshot_settings  # noqa: SLF001
        try:
            with (
                patch(
                    "fdm.ui.main_window.QMessageBox.question",
                    return_value=QMessageBox.StandardButton.Yes,
                ),
                patch(
                    "fdm.ui.main_window.ScreenshotSettingsIO.update",
                    side_effect=OSError("disk full"),
                ) as update,
                patch("fdm.ui.main_window.QMessageBox.warning"),
            ):
                self.assertFalse(
                    window._apply_screenshot_settings(  # noqa: SLF001
                        ScreenshotSettings(enabled=True)
                    )
                )

            self.assertIs(window._screenshot_settings, previous)  # noqa: SLF001
            self.assertTrue(window._screenshot_settings_read_only)  # noqa: SLF001
            self.assertEqual(window._screenshot_settings_load_error, "schema 999")  # noqa: SLF001
            self.assertTrue(update.call_args.kwargs["allow_unsupported_replace"])
        finally:
            window.close()

    def test_explicit_future_schema_replacement_uses_the_guarded_override(self) -> None:
        window = self._window()
        window._screenshot_settings_read_only = True  # noqa: SLF001
        window._screenshot_settings_load_error = "schema 999"  # noqa: SLF001

        def replace_future(mutator, *_args, **kwargs):
            self.assertTrue(kwargs["allow_unsupported_replace"])
            return mutator(ScreenshotSettings())

        try:
            with (
                patch(
                    "fdm.ui.main_window.QMessageBox.question",
                    return_value=QMessageBox.StandardButton.Yes,
                ),
                patch(
                    "fdm.ui.main_window.ScreenshotSettingsIO.update",
                    side_effect=replace_future,
                ),
            ):
                self.assertTrue(
                    window._apply_screenshot_settings(  # noqa: SLF001
                        ScreenshotSettings(enabled=False, delay_ms=500)
                    )
                )

            self.assertFalse(window._screenshot_settings_read_only)  # noqa: SLF001
            self.assertEqual(window._screenshot_settings_load_error, "")  # noqa: SLF001
            self.assertEqual(window._screenshot_settings.delay_ms, 500)  # noqa: SLF001
        finally:
            window.close()

    def test_cu5_diagnostic_stops_a_temporarily_started_agent(self) -> None:
        window = self._window()
        client = _FakeScreenshotClient()
        window._screenshot_agent_client = client  # noqa: SLF001

        class _Page:
            def __init__(self) -> None:
                self.messages: list[tuple[str, bool]] = []
                self.candidates: list[tuple[object, object]] = []

            def set_cu5_diagnostic_status(self, message, *, success):
                self.messages.append((str(message), bool(success)))

            def set_cu5_candidates(self, candidates, *, selected_selector=None):
                self.candidates.append((candidates, selected_selector))

        page = _Page()
        try:
            window._diagnose_cu5_preview(  # noqa: SLF001
                SimpleNamespace(_screenshot_settings_widget=page)
            )
            self.assertEqual(client.started, 1)
            self.assertEqual(client.shutdown_calls, 1)
            self.assertFalse(client.running)
            self.assertTrue(page.messages[-1][1])
            self.assertIn("768×576", page.messages[-1][0])
            self.assertEqual(page.candidates[-1][1]["control_id"], 1501)
        finally:
            window.close()

    def test_adjusting_cu_preview_object_persists_and_revalidates_selection(self) -> None:
        window = self._window()
        client = _FakeScreenshotClient()
        window._screenshot_agent_client = client  # noqa: SLF001

        class _Page:
            def __init__(self) -> None:
                self.messages: list[tuple[str, bool]] = []
                self.selected: object = None

            def set_cu5_diagnostic_status(self, message, *, success):
                self.messages.append((str(message), bool(success)))

            def set_cu5_candidates(self, _candidates, *, selected_selector=None):
                self.selected = selected_selector

        selector = {
            "process_name": "cu-6.exe",
            "class_name": "static",
            "control_id": 1501,
        }
        page = _Page()
        dialog = SimpleNamespace(_screenshot_settings_widget=page)
        try:
            with patch(
                "fdm.ui.main_window.ScreenshotSettingsIO.update",
                side_effect=lambda mutator, *_args, **_kwargs: mutator(
                    window._screenshot_settings  # noqa: SLF001
                ),
            ):
                window._select_cu5_preview_candidate(dialog, selector)  # noqa: SLF001

            self.assertEqual(
                window._screenshot_settings.cu5_selector["control_id"],  # noqa: SLF001
                1501,
            )
            self.assertEqual(client.updated[-1]["cu5_selector"], selector)
            self.assertEqual(page.selected["control_id"], 1501)
            self.assertTrue(page.messages[-1][1])
            self.assertEqual(client.shutdown_calls, 1)
        finally:
            window.close()

    def test_failed_cu_preview_adjustment_restores_previous_selector(self) -> None:
        window = self._window()
        previous_selector = {
            "process_name": "cu.exe",
            "class_name": "cwndforsdk",
            "control_id": 1201,
        }
        window._screenshot_settings = ScreenshotSettings(  # noqa: SLF001
            cu5_selector=previous_selector
        )
        client = _FakeScreenshotClient()
        client.send = lambda *_args, **_kwargs: (_ for _ in ()).throw(  # type: ignore[method-assign]
            ScreenshotAgentCommandError(
                "所选对象已不可用",
                response=IPCResponse.failure("diagnose", "所选对象已不可用"),
            )
        )
        window._screenshot_agent_client = client  # noqa: SLF001

        class _Page:
            def __init__(self) -> None:
                self.messages: list[tuple[str, bool]] = []

            def set_cu5_diagnostic_status(self, message, *, success):
                self.messages.append((str(message), bool(success)))

        page = _Page()
        requested = {
            "process_name": "cu-6.exe",
            "class_name": "static",
            "control_id": 1501,
        }
        try:
            with patch(
                "fdm.ui.main_window.ScreenshotSettingsIO.update",
                side_effect=lambda mutator, *_args, **_kwargs: mutator(
                    window._screenshot_settings  # noqa: SLF001
                ),
            ):
                window._select_cu5_preview_candidate(  # noqa: SLF001
                    SimpleNamespace(_screenshot_settings_widget=page),
                    requested,
                )

            self.assertEqual(
                window._screenshot_settings.cu5_selector,  # noqa: SLF001
                previous_selector,
            )
            self.assertEqual(client.updated[0]["cu5_selector"], requested)
            self.assertEqual(client.shutdown_calls, 1)
            self.assertIn("已恢复原预览对象", page.messages[-1][0])
        finally:
            window.close()

    def test_main_settings_save_preserves_agent_owned_runtime_fields(self) -> None:
        window = self._window()
        persisted = ScreenshotSettings(
            last_region=CaptureRect(-20, 30, 400, 300),
            cu5_selector={"class_name": "cwndforsdk", "control_id": 1201},
            annotation_styles={
                "schema_version": 1,
                "active_tool": "arrow",
                "tools": {"arrow": {"color": "#123456", "arrow_size": 24}},
            },
        )
        try:
            window._screenshot_settings.enabled = True  # noqa: SLF001
            with (
                patch(
                    "fdm.ui.main_window.ScreenshotSettingsIO.update",
                    side_effect=lambda mutator, *_args, **_kwargs: mutator(persisted),
                ) as update,
            ):
                self.assertTrue(window._save_screenshot_settings())  # noqa: SLF001

            saved = window._screenshot_settings  # noqa: SLF001
            update.assert_called_once()
            self.assertTrue(saved.enabled)
            self.assertEqual(saved.last_region, persisted.last_region)
            self.assertEqual(saved.cu5_selector, persisted.cu5_selector)
            self.assertEqual(saved.annotation_styles, persisted.normalized().annotation_styles)
            payload = window._screenshot_settings_update_payload()  # noqa: SLF001
            self.assertNotIn("last_region", payload)
            self.assertNotIn("cu5_selector", payload)
            self.assertNotIn("annotation_styles", payload)
        finally:
            window.close()

    def test_ambiguous_cu5_diagnostic_can_store_a_stable_candidate_without_dragging(self) -> None:
        window = self._window()
        client = _FakeScreenshotClient()
        failed = IPCResponse(
            request_id="ambiguous",
            ok=False,
            error="多个候选",
            result={
                "diagnostic": "ambiguous",
                "candidates": [
                    {
                        "rect": {"x": 20, "y": 30, "width": 768, "height": 576},
                        "score": 180.0,
                        "selector": {
                            "process_name": "cu-5.exe",
                            "class_name": "cwndforsdk",
                            "control_id": 1201,
                        },
                    }
                ],
            },
        )
        calls = 0

        def send(_command, **_kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise ScreenshotAgentCommandError("多个候选", response=failed)
            return IPCResponse.success(
                "resolved",
                {"rect": {"x": 20, "y": 30, "width": 768, "height": 576}},
            )

        client.send = send  # type: ignore[method-assign]
        window._screenshot_agent_client = client  # noqa: SLF001
        try:
            with (
                patch(
                    "fdm.ui.main_window.QInputDialog.getItem",
                    return_value=("候选 1：768×576 px，位置 (20, 30)，得分 180.0", True),
                ),
                patch(
                    "fdm.ui.main_window.ScreenshotSettingsIO.load",
                    return_value=ScreenshotSettings(),
                ),
                patch("fdm.ui.main_window.ScreenshotSettingsIO.save"),
                patch(
                    "fdm.ui.main_window.ScreenshotSettingsIO.update",
                    side_effect=lambda mutator, *_args, **_kwargs: mutator(
                        ScreenshotSettings()
                    ),
                ),
            ):
                response = window._send_cu5_diagnostic_with_selection(  # noqa: SLF001
                    SimpleNamespace()
                )

            self.assertTrue(response.ok)
            self.assertEqual(calls, 2)
            self.assertEqual(
                window._screenshot_settings.cu5_selector["control_id"],  # noqa: SLF001
                1201,
            )
            self.assertEqual(client.updated[-1]["cu5_selector"]["class_name"], "cwndforsdk")
        finally:
            window.close()


if __name__ == "__main__":
    unittest.main()
