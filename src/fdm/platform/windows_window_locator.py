from __future__ import annotations

import ctypes
from dataclasses import dataclass
from pathlib import PureWindowsPath
import sys
from types import MappingProxyType
from typing import Mapping, Protocol, Sequence


DWMWA_EXTENDED_FRAME_BOUNDS = 9
DWMWA_CLOAKED = 14
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000


class WindowsWindowEnumerationUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class PhysicalRect:
    left: int
    top: int
    right: int
    bottom: int

    def __post_init__(self) -> None:
        if self.right < self.left or self.bottom < self.top:
            raise ValueError("窗口矩形坐标无效。")

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.left + self.right) / 2.0, (self.top + self.bottom) / 2.0)

    def intersection(self, other: "PhysicalRect") -> "PhysicalRect | None":
        left = max(self.left, other.left)
        top = max(self.top, other.top)
        right = min(self.right, other.right)
        bottom = min(self.bottom, other.bottom)
        if right <= left or bottom <= top:
            return None
        return PhysicalRect(left, top, right, bottom)

    def contains_rect(self, other: "PhysicalRect") -> bool:
        return (
            self.left <= other.left
            and self.top <= other.top
            and self.right >= other.right
            and self.bottom >= other.bottom
        )


@dataclass(frozen=True, slots=True)
class WindowRecord:
    hwnd: int
    parent_hwnd: int | None
    root_hwnd: int
    ancestor_hwnds: tuple[int, ...]
    pid: int
    process_path: str
    title: str
    class_name: str
    control_id: int | None
    rect: PhysicalRect
    visible: bool
    minimized: bool
    cloaked: bool

    @property
    def process_name(self) -> str:
        return PureWindowsPath(self.process_path).name

    @property
    def available_for_capture(self) -> bool:
        return bool(
            self.visible
            and not self.minimized
            and not self.cloaked
            and self.rect.width > 0
            and self.rect.height > 0
        )


@dataclass(frozen=True, slots=True)
class WindowSnapshot:
    records: tuple[WindowRecord, ...]
    by_hwnd: Mapping[int, WindowRecord]

    @classmethod
    def from_records(cls, records: Sequence[WindowRecord]) -> "WindowSnapshot":
        normalized = tuple(records)
        mapping = {record.hwnd: record for record in normalized}
        if len(mapping) != len(normalized):
            raise ValueError("窗口快照包含重复 HWND。")
        return cls(normalized, MappingProxyType(mapping))

    def roots(self) -> tuple[WindowRecord, ...]:
        return tuple(record for record in self.records if record.parent_hwnd is None)

    def descendants(self, hwnd: int) -> tuple[WindowRecord, ...]:
        target = int(hwnd)
        return tuple(record for record in self.records if target in record.ancestor_hwnds)

    def for_process(self, pid: int) -> tuple[WindowRecord, ...]:
        return tuple(record for record in self.records if record.pid == int(pid))


class WindowEnumerationNativeApi(Protocol):
    def enum_windows(self) -> Sequence[int]: ...

    def enum_child_windows(self, parent_hwnd: int) -> Sequence[int]: ...

    def get_process_id(self, hwnd: int) -> int: ...

    def get_process_path(self, pid: int) -> str: ...

    def get_title(self, hwnd: int) -> str: ...

    def get_class_name(self, hwnd: int) -> str: ...

    def get_control_id(self, hwnd: int, *, is_top_level: bool) -> int | None: ...

    def get_physical_rect(self, hwnd: int, *, is_top_level: bool) -> PhysicalRect: ...

    def is_visible(self, hwnd: int) -> bool: ...

    def is_minimized(self, hwnd: int) -> bool: ...

    def is_cloaked(self, hwnd: int) -> bool: ...


class _Rect(ctypes.Structure):
    _fields_ = [
        ("left", ctypes.c_long),
        ("top", ctypes.c_long),
        ("right", ctypes.c_long),
        ("bottom", ctypes.c_long),
    ]


class _CtypesWindowEnumerationApi:
    def __init__(self) -> None:
        if sys.platform != "win32":
            raise WindowsWindowEnumerationUnavailableError(
                "窗口枚举仅能在 Windows 上运行。"
            )
        win_dll = getattr(ctypes, "WinDLL", None)
        if win_dll is None:  # pragma: no cover - defensive platform guard
            raise WindowsWindowEnumerationUnavailableError(
                "当前 Python 运行时不提供 Win32 API。"
            )
        self._user32 = win_dll("user32", use_last_error=True)
        self._kernel32 = win_dll("kernel32", use_last_error=True)
        try:
            self._dwmapi = win_dll("dwmapi", use_last_error=True)
        except OSError:  # pragma: no cover - unsupported legacy Windows
            self._dwmapi = None
        self._process_paths: dict[int, str] = {}
        self._configure_signatures()

    def _configure_signatures(self) -> None:
        self._user32.GetWindowThreadProcessId.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_ulong),
        ]
        self._user32.GetWindowThreadProcessId.restype = ctypes.c_ulong
        self._user32.GetWindowTextLengthW.argtypes = [ctypes.c_void_p]
        self._user32.GetWindowTextLengthW.restype = ctypes.c_int
        self._user32.GetWindowTextW.argtypes = [
            ctypes.c_void_p,
            ctypes.c_wchar_p,
            ctypes.c_int,
        ]
        self._user32.GetWindowTextW.restype = ctypes.c_int
        self._user32.GetClassNameW.argtypes = [
            ctypes.c_void_p,
            ctypes.c_wchar_p,
            ctypes.c_int,
        ]
        self._user32.GetClassNameW.restype = ctypes.c_int
        self._user32.GetDlgCtrlID.argtypes = [ctypes.c_void_p]
        self._user32.GetDlgCtrlID.restype = ctypes.c_int
        self._user32.GetWindowRect.argtypes = [ctypes.c_void_p, ctypes.POINTER(_Rect)]
        self._user32.GetWindowRect.restype = ctypes.c_int
        self._user32.IsWindowVisible.argtypes = [ctypes.c_void_p]
        self._user32.IsWindowVisible.restype = ctypes.c_int
        self._user32.IsIconic.argtypes = [ctypes.c_void_p]
        self._user32.IsIconic.restype = ctypes.c_int
        self._user32.GetParent.argtypes = [ctypes.c_void_p]
        self._user32.GetParent.restype = ctypes.c_void_p
        self._kernel32.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
        self._kernel32.OpenProcess.restype = ctypes.c_void_p
        self._kernel32.QueryFullProcessImageNameW.argtypes = [
            ctypes.c_void_p,
            ctypes.c_ulong,
            ctypes.c_wchar_p,
            ctypes.POINTER(ctypes.c_ulong),
        ]
        self._kernel32.QueryFullProcessImageNameW.restype = ctypes.c_int
        self._kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
        self._kernel32.CloseHandle.restype = ctypes.c_int
        if self._dwmapi is not None:
            self._dwmapi.DwmGetWindowAttribute.argtypes = [
                ctypes.c_void_p,
                ctypes.c_ulong,
                ctypes.c_void_p,
                ctypes.c_ulong,
            ]
            self._dwmapi.DwmGetWindowAttribute.restype = ctypes.c_long

    def begin_physical_coordinates(self) -> object | None:
        setter = getattr(self._user32, "SetThreadDpiAwarenessContext", None)
        if setter is None:
            return None
        setter.argtypes = [ctypes.c_void_p]
        setter.restype = ctypes.c_void_p
        return setter(ctypes.c_void_p(-4))  # PER_MONITOR_AWARE_V2

    def end_physical_coordinates(self, previous: object | None) -> None:
        if not previous:
            return
        setter = getattr(self._user32, "SetThreadDpiAwarenessContext", None)
        if setter is not None:
            setter(ctypes.c_void_p(int(previous)))

    def enum_windows(self) -> Sequence[int]:
        callback_type = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p)
        handles: list[int] = []

        @callback_type
        def callback(hwnd, _lparam) -> int:
            handles.append(int(hwnd))
            return 1

        self._user32.EnumWindows(callback, 0)
        return tuple(handles)

    def enum_child_windows(self, parent_hwnd: int) -> Sequence[int]:
        callback_type = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p)
        handles: list[int] = []
        parent_value = int(parent_hwnd)

        @callback_type
        def callback(hwnd, _lparam) -> int:
            # EnumChildWindows walks all descendants. Filter to direct children
            # so the caller can build a stable hierarchy without duplicates.
            if int(self._user32.GetParent(hwnd) or 0) == parent_value:
                handles.append(int(hwnd))
            return 1

        self._user32.EnumChildWindows(ctypes.c_void_p(parent_value), callback, 0)
        return tuple(handles)

    def get_process_id(self, hwnd: int) -> int:
        pid = ctypes.c_ulong(0)
        self._user32.GetWindowThreadProcessId(
            ctypes.c_void_p(int(hwnd)),
            ctypes.byref(pid),
        )
        return int(pid.value)

    def get_process_path(self, pid: int) -> str:
        pid = int(pid)
        if pid in self._process_paths:
            return self._process_paths[pid]
        process = self._kernel32.OpenProcess(
            PROCESS_QUERY_LIMITED_INFORMATION,
            0,
            pid,
        )
        if not process:
            self._process_paths[pid] = ""
            return ""
        try:
            capacity = 32768
            buffer = ctypes.create_unicode_buffer(capacity)
            size = ctypes.c_ulong(capacity)
            ok = self._kernel32.QueryFullProcessImageNameW(
                process,
                0,
                buffer,
                ctypes.byref(size),
            )
            value = buffer.value[: int(size.value)] if ok else ""
        finally:
            self._kernel32.CloseHandle(process)
        self._process_paths[pid] = value
        return value

    def get_title(self, hwnd: int) -> str:
        length = max(0, int(self._user32.GetWindowTextLengthW(ctypes.c_void_p(int(hwnd)))))
        buffer = ctypes.create_unicode_buffer(length + 1)
        self._user32.GetWindowTextW(
            ctypes.c_void_p(int(hwnd)),
            buffer,
            length + 1,
        )
        return buffer.value

    def get_class_name(self, hwnd: int) -> str:
        buffer = ctypes.create_unicode_buffer(512)
        length = int(
            self._user32.GetClassNameW(
                ctypes.c_void_p(int(hwnd)),
                buffer,
                len(buffer),
            )
        )
        return buffer.value[: max(0, length)]

    def get_control_id(self, hwnd: int, *, is_top_level: bool) -> int | None:
        if is_top_level:
            return None
        return int(self._user32.GetDlgCtrlID(ctypes.c_void_p(int(hwnd))))

    def get_physical_rect(self, hwnd: int, *, is_top_level: bool) -> PhysicalRect:
        rect = _Rect()
        if is_top_level and self._dwmapi is not None:
            result = int(
                self._dwmapi.DwmGetWindowAttribute(
                    ctypes.c_void_p(int(hwnd)),
                    DWMWA_EXTENDED_FRAME_BOUNDS,
                    ctypes.byref(rect),
                    ctypes.sizeof(rect),
                )
            )
            if result == 0:
                return PhysicalRect(rect.left, rect.top, rect.right, rect.bottom)
        if not self._user32.GetWindowRect(ctypes.c_void_p(int(hwnd)), ctypes.byref(rect)):
            return PhysicalRect(0, 0, 0, 0)
        return PhysicalRect(rect.left, rect.top, rect.right, rect.bottom)

    def is_visible(self, hwnd: int) -> bool:
        return bool(self._user32.IsWindowVisible(ctypes.c_void_p(int(hwnd))))

    def is_minimized(self, hwnd: int) -> bool:
        return bool(self._user32.IsIconic(ctypes.c_void_p(int(hwnd))))

    def is_cloaked(self, hwnd: int) -> bool:
        if self._dwmapi is None:
            return False
        cloaked = ctypes.c_ulong(0)
        result = int(
            self._dwmapi.DwmGetWindowAttribute(
                ctypes.c_void_p(int(hwnd)),
                DWMWA_CLOAKED,
                ctypes.byref(cloaked),
                ctypes.sizeof(cloaked),
            )
        )
        return result == 0 and bool(cloaked.value)


class WindowsWindowEnumerator:
    def __init__(self, api: WindowEnumerationNativeApi | None = None) -> None:
        self._api = api or _CtypesWindowEnumerationApi()

    def enumerate(self) -> WindowSnapshot:
        begin = getattr(self._api, "begin_physical_coordinates", None)
        end = getattr(self._api, "end_physical_coordinates", None)
        previous = begin() if callable(begin) else None
        try:
            records: list[WindowRecord] = []
            visited: set[int] = set()
            for root_hwnd in self._api.enum_windows():
                root = int(root_hwnd)
                if root <= 0 or root in visited:
                    continue
                self._visit(
                    root,
                    parent_hwnd=None,
                    root_hwnd=root,
                    ancestor_hwnds=(),
                    inherited_minimized=False,
                    inherited_cloaked=False,
                    records=records,
                    visited=visited,
                )
            return WindowSnapshot.from_records(records)
        finally:
            if callable(end):
                end(previous)

    def _visit(
        self,
        hwnd: int,
        *,
        parent_hwnd: int | None,
        root_hwnd: int,
        ancestor_hwnds: tuple[int, ...],
        inherited_minimized: bool,
        inherited_cloaked: bool,
        records: list[WindowRecord],
        visited: set[int],
    ) -> None:
        if hwnd in visited:
            return
        visited.add(hwnd)
        pid = int(self._api.get_process_id(hwnd))
        minimized = inherited_minimized or bool(self._api.is_minimized(hwnd))
        cloaked = inherited_cloaked or bool(self._api.is_cloaked(hwnd))
        records.append(
            WindowRecord(
                hwnd=hwnd,
                parent_hwnd=parent_hwnd,
                root_hwnd=root_hwnd,
                ancestor_hwnds=ancestor_hwnds,
                pid=pid,
                process_path=self._api.get_process_path(pid),
                title=self._api.get_title(hwnd),
                class_name=self._api.get_class_name(hwnd),
                control_id=self._api.get_control_id(
                    hwnd,
                    is_top_level=parent_hwnd is None,
                ),
                rect=self._api.get_physical_rect(
                    hwnd,
                    is_top_level=parent_hwnd is None,
                ),
                visible=bool(self._api.is_visible(hwnd)),
                minimized=minimized,
                cloaked=cloaked,
            )
        )
        children = self._api.enum_child_windows(hwnd)
        for child_hwnd in children:
            child = int(child_hwnd)
            if child <= 0:
                continue
            self._visit(
                child,
                parent_hwnd=hwnd,
                root_hwnd=root_hwnd,
                ancestor_hwnds=(*ancestor_hwnds, hwnd),
                inherited_minimized=minimized,
                inherited_cloaked=cloaked,
                records=records,
                visited=visited,
            )


def enumerate_windows(
    api: WindowEnumerationNativeApi | None = None,
) -> WindowSnapshot:
    return WindowsWindowEnumerator(api).enumerate()


__all__ = [
    "DWMWA_CLOAKED",
    "DWMWA_EXTENDED_FRAME_BOUNDS",
    "PhysicalRect",
    "WindowEnumerationNativeApi",
    "WindowRecord",
    "WindowSnapshot",
    "WindowsWindowEnumerationUnavailableError",
    "WindowsWindowEnumerator",
    "enumerate_windows",
]
