from __future__ import annotations

from pathlib import Path

import pytest

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence

from fdm.platform.windows_global_hotkey import (
    MOD_CONTROL,
    MOD_NOREPEAT,
    WindowsGlobalHotkeyManager,
)
from fdm.screenshot_agent import _parse_windows_hotkey
from fdm.screenshot_settings import (
    CaptureMode,
    HotkeyBinding,
    ScreenshotSettings,
    ScreenshotSettingsIO,
)


class _FakeHotkeyApi:
    def __init__(self) -> None:
        self.registered: list[tuple[int, int, int, int]] = []

    def register_hot_key(
        self,
        hwnd: int,
        identifier: int,
        modifiers: int,
        virtual_key: int,
    ) -> tuple[bool, int]:
        self.registered.append((hwnd, identifier, modifiers, virtual_key))
        return True, 0

    def unregister_hot_key(
        self,
        hwnd: int,
        identifier: int,
    ) -> tuple[bool, int]:
        del hwnd, identifier
        return True, 0


def test_menu_key_round_trips_from_qt_to_register_hotkey(tmp_path: Path) -> None:
    sequence = QKeySequence(Qt.Key.Key_Menu).toString(
        QKeySequence.SequenceFormat.PortableText
    )
    assert sequence == "Menu"

    settings_path = tmp_path / "screenshot-settings.json"
    ScreenshotSettingsIO.save(
        ScreenshotSettings(
            enabled=True,
            hotkeys={CaptureMode.REGION: HotkeyBinding(sequence)},
        ),
        settings_path,
    )
    loaded = ScreenshotSettingsIO.load(settings_path)
    assert loaded.hotkeys[CaptureMode.REGION] == HotkeyBinding("Menu")

    identifier = 0x5F01
    parsed = _parse_windows_hotkey(
        loaded.hotkeys[CaptureMode.REGION].sequence,
        identifier,
    )
    api = _FakeHotkeyApi()
    manager = WindowsGlobalHotkeyManager(101, api=api)

    binding = manager.bind(parsed)

    assert binding.virtual_key == 0x5D  # VK_APPS
    assert api.registered == [(101, identifier, MOD_NOREPEAT, 0x5D)]


def test_apps_alias_parses_to_vk_apps() -> None:
    binding = _parse_windows_hotkey("Apps", 0x5F01)

    assert binding.virtual_key == 0x5D  # VK_APPS


def test_menu_key_keeps_modifiers() -> None:
    binding = _parse_windows_hotkey("Ctrl+Menu", 0x5F01)

    assert binding.virtual_key == 0x5D
    assert binding.modifiers & MOD_CONTROL


@pytest.mark.parametrize(
    ("qt_key", "expected_virtual_key"),
    [
        (Qt.Key.Key_Cancel, 0x03),
        (Qt.Key.Key_Clear, 0x0C),
        (Qt.Key.Key_Select, 0x29),
        (Qt.Key.Key_Execute, 0x2B),
        (Qt.Key.Key_Help, 0x2F),
        (Qt.Key.Key_Sleep, 0x5F),
        (Qt.Key.Key_NumLock, 0x90),
        (Qt.Key.Key_ScrollLock, 0x91),
        (Qt.Key.Key_Back, 0xA6),
        (Qt.Key.Key_Forward, 0xA7),
        (Qt.Key.Key_Refresh, 0xA8),
        (Qt.Key.Key_Stop, 0xA9),
        (Qt.Key.Key_Search, 0xAA),
        (Qt.Key.Key_Favorites, 0xAB),
        (Qt.Key.Key_HomePage, 0xAC),
        (Qt.Key.Key_VolumeMute, 0xAD),
        (Qt.Key.Key_VolumeDown, 0xAE),
        (Qt.Key.Key_VolumeUp, 0xAF),
        (Qt.Key.Key_MediaNext, 0xB0),
        (Qt.Key.Key_MediaPrevious, 0xB1),
        (Qt.Key.Key_MediaStop, 0xB2),
        (Qt.Key.Key_MediaPlay, 0xB3),
        (Qt.Key.Key_MediaPause, 0xB3),
        (Qt.Key.Key_MediaTogglePlayPause, 0xB3),
        (Qt.Key.Key_LaunchMail, 0xB4),
        (Qt.Key.Key_LaunchMedia, 0xB5),
        (Qt.Key.Key_Launch0, 0xB6),
        (Qt.Key.Key_Launch1, 0xB7),
    ],
)
def test_common_qt_named_keys_map_to_documented_windows_virtual_keys(
    qt_key: Qt.Key,
    expected_virtual_key: int,
) -> None:
    sequence = QKeySequence(qt_key).toString(
        QKeySequence.SequenceFormat.PortableText
    )
    assert sequence

    binding = _parse_windows_hotkey(sequence, 0x5F01)

    assert binding.virtual_key == expected_virtual_key
