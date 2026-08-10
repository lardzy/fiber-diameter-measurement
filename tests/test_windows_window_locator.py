from __future__ import annotations

import ctypes
import sys

import pytest

from fdm.platform.windows_window_locator import (
    DWMWA_CLOAKED,
    PhysicalRect,
    WindowsWindowEnumerationUnavailableError,
    WindowsWindowEnumerator,
)
from fdm.platform import windows_window_locator as native_locator_module


class _FakeWindowApi:
    def __init__(self) -> None:
        self.children = {10: (11, 12), 11: (13,), 12: (), 13: ()}
        self.parents = {10: None, 11: 10, 12: 10, 13: 11}
        self.rects = {
            10: PhysicalRect(-100, 50, 1100, 950),
            11: PhysicalRect(0, 100, 1000, 850),
            12: PhysicalRect(10, 110, 110, 140),
            13: PhysicalRect(120, 130, 888, 706),
        }
        self.classes = {10: "AfxFrame", 11: "MDIClient", 12: "Static", 13: "CWndForSDK"}
        self.titles = {10: "CU-5", 11: "", 12: "实时预览", 13: ""}
        self.begin_count = 0
        self.end_tokens: list[object] = []

    def begin_physical_coordinates(self):
        self.begin_count += 1
        return "dpi-token"

    def end_physical_coordinates(self, token):
        self.end_tokens.append(token)

    def enum_windows(self):
        return (10,)

    def enum_child_windows(self, parent_hwnd):
        return self.children[parent_hwnd]

    def get_process_id(self, hwnd):
        return 55

    def get_process_path(self, pid):
        assert pid == 55
        return r"C:\Program Files\CU-5\CU-5.exe"

    def get_title(self, hwnd):
        return self.titles[hwnd]

    def get_class_name(self, hwnd):
        return self.classes[hwnd]

    def get_control_id(self, hwnd, *, is_top_level):
        return None if is_top_level else 1000 + hwnd

    def get_physical_rect(self, hwnd, *, is_top_level):
        del is_top_level
        return self.rects[hwnd]

    def is_visible(self, hwnd):
        return hwnd != 12

    def is_minimized(self, hwnd):
        return hwnd == 10

    def is_cloaked(self, hwnd):
        return hwnd == 11


def test_enumerator_records_process_hierarchy_and_physical_state() -> None:
    api = _FakeWindowApi()
    snapshot = WindowsWindowEnumerator(api).enumerate()

    assert [record.hwnd for record in snapshot.records] == [10, 11, 13, 12]
    preview = snapshot.by_hwnd[13]
    assert preview.parent_hwnd == 11
    assert preview.root_hwnd == 10
    assert preview.ancestor_hwnds == (10, 11)
    assert preview.pid == 55
    assert preview.process_name == "CU-5.exe"
    assert preview.class_name == "CWndForSDK"
    assert preview.control_id == 1013
    assert preview.rect == PhysicalRect(120, 130, 888, 706)
    assert preview.minimized
    assert preview.cloaked
    assert api.begin_count == 1
    assert api.end_tokens == ["dpi-token"]
    assert snapshot.descendants(11) == (preview,)


def test_rect_supports_negative_multimonitor_coordinates_and_intersection() -> None:
    left_monitor = PhysicalRect(-1920, 0, 0, 1080)
    target = PhysicalRect(-200, 100, 200, 500)

    assert left_monitor.width == 1920
    assert left_monitor.intersection(target) == PhysicalRect(-200, 100, 0, 500)
    assert not left_monitor.contains_rect(target)


def test_native_enumerator_fails_lazily_off_windows() -> None:
    if sys.platform == "win32":
        pytest.skip("non-Windows import guard")
    with pytest.raises(WindowsWindowEnumerationUnavailableError):
        WindowsWindowEnumerator()


def test_native_cloaked_query_uses_the_four_argument_dwm_signature() -> None:
    calls: list[tuple[object, ...]] = []

    class _DwmApi:
        @staticmethod
        def DwmGetWindowAttribute(*arguments):
            calls.append(arguments)
            assert len(arguments) == 4
            value = ctypes.cast(
                arguments[2],
                ctypes.POINTER(ctypes.c_ulong),
            )
            value.contents.value = 1
            return 0

    api = object.__new__(native_locator_module._CtypesWindowEnumerationApi)
    api._dwmapi = _DwmApi()

    assert api.is_cloaked(123)
    assert len(calls) == 1
    assert calls[0][1] == DWMWA_CLOAKED
    assert calls[0][3] == ctypes.sizeof(ctypes.c_ulong())
