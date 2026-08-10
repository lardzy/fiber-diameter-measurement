from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
import inspect
import sys
from typing import Callable, Iterable, Protocol, Sequence

from PySide6.QtCore import QObject, QPoint, QRect, QTimer, Signal
from PySide6.QtGui import QGuiApplication, QImage, QPainter


class CaptureMode(str, Enum):
    REGION = "region"
    SMART = "smart"
    WINDOW = "window"
    ACTIVE_WINDOW = "active_window"
    DISPLAY = "display"
    FULL_SCREEN = "full_screen"
    LAST_REGION = "last_region"
    CU5 = "cu5"

    @classmethod
    def parse(cls, value: object) -> "CaptureMode":
        if isinstance(value, cls):
            return value
        token = str(value or "").strip().lower().replace("-", "_")
        aliases = {
            "rect": cls.REGION,
            "area": cls.REGION,
            "smart_window": cls.SMART,
            "child_window": cls.WINDOW,
            "active": cls.ACTIVE_WINDOW,
            "screen": cls.DISPLAY,
            "desktop": cls.FULL_SCREEN,
            "fullscreen": cls.FULL_SCREEN,
            "last": cls.LAST_REGION,
            "cu_5": cls.CU5,
        }
        if token in aliases:
            return aliases[token]
        return cls(token)


@dataclass(frozen=True, slots=True)
class CaptureRect:
    """A rectangle in native desktop pixels.

    Native coordinates deliberately remain separate from Qt logical coordinates:
    a monitor left of the primary display can have a negative origin and each
    monitor can use a different device-pixel ratio.
    """

    x: int
    y: int
    width: int
    height: int

    @property
    def right(self) -> int:
        return self.x + self.width

    @property
    def bottom(self) -> int:
        return self.y + self.height

    @property
    def area(self) -> int:
        return max(0, self.width) * max(0, self.height)

    @property
    def valid(self) -> bool:
        return self.width > 0 and self.height > 0

    def normalized(self) -> "CaptureRect":
        x1 = min(self.x, self.right)
        y1 = min(self.y, self.bottom)
        x2 = max(self.x, self.right)
        y2 = max(self.y, self.bottom)
        return CaptureRect(x1, y1, x2 - x1, y2 - y1)

    def intersection(self, other: "CaptureRect") -> "CaptureRect | None":
        left = max(self.x, other.x)
        top = max(self.y, other.y)
        right = min(self.right, other.right)
        bottom = min(self.bottom, other.bottom)
        if right <= left or bottom <= top:
            return None
        return CaptureRect(left, top, right - left, bottom - top)

    def contains(self, x: int, y: int) -> bool:
        return self.x <= x < self.right and self.y <= y < self.bottom

    def translated(self, dx: int, dy: int) -> "CaptureRect":
        return CaptureRect(self.x + int(dx), self.y + int(dy), self.width, self.height)

    def to_qrect(self) -> QRect:
        return QRect(self.x, self.y, self.width, self.height)

    @classmethod
    def from_qrect(cls, rect: QRect) -> "CaptureRect":
        return cls(rect.x(), rect.y(), rect.width(), rect.height())


@dataclass(frozen=True, slots=True)
class ScreenInfo:
    name: str
    logical_rect: CaptureRect
    physical_rect: CaptureRect
    device_pixel_ratio: float = 1.0
    primary: bool = False

    def logical_fragment_to_physical(self, fragment: CaptureRect) -> CaptureRect:
        clipped = fragment.intersection(self.logical_rect)
        if clipped is None:
            return CaptureRect(0, 0, 0, 0)
        ratio_x = self.physical_rect.width / max(1, self.logical_rect.width)
        ratio_y = self.physical_rect.height / max(1, self.logical_rect.height)
        left = self.physical_rect.x + round((clipped.x - self.logical_rect.x) * ratio_x)
        top = self.physical_rect.y + round((clipped.y - self.logical_rect.y) * ratio_y)
        right = self.physical_rect.x + round((clipped.right - self.logical_rect.x) * ratio_x)
        bottom = self.physical_rect.y + round((clipped.bottom - self.logical_rect.y) * ratio_y)
        return CaptureRect(left, top, right - left, bottom - top)

    def physical_fragment_to_logical(self, fragment: CaptureRect) -> CaptureRect:
        clipped = fragment.intersection(self.physical_rect)
        if clipped is None:
            return CaptureRect(0, 0, 0, 0)
        ratio_x = self.logical_rect.width / max(1, self.physical_rect.width)
        ratio_y = self.logical_rect.height / max(1, self.physical_rect.height)
        left = self.logical_rect.x + round((clipped.x - self.physical_rect.x) * ratio_x)
        top = self.logical_rect.y + round((clipped.y - self.physical_rect.y) * ratio_y)
        right = self.logical_rect.x + round((clipped.right - self.physical_rect.x) * ratio_x)
        bottom = self.logical_rect.y + round((clipped.bottom - self.physical_rect.y) * ratio_y)
        return CaptureRect(left, top, right - left, bottom - top)


@dataclass(frozen=True, slots=True)
class WindowCandidate:
    handle: int
    rect: CaptureRect
    client_rect: CaptureRect | None = None
    title: str = ""
    class_name: str = ""
    process_id: int = 0
    executable: str = ""
    parent_handle: int = 0
    depth: int = 0
    z_order: int = 0
    visible: bool = True
    minimized: bool = False
    capture_safe: bool = True
    metadata: dict[str, object] = field(default_factory=dict, compare=False, hash=False)

    @property
    def capture_rect(self) -> CaptureRect:
        return self.client_rect if self.client_rect is not None and self.client_rect.valid else self.rect

    def contains(self, point: QPoint) -> bool:
        return self.capture_rect.contains(point.x(), point.y())


@dataclass(frozen=True, slots=True)
class CaptureRequest:
    mode: CaptureMode = CaptureMode.REGION
    delay_ms: int = 0
    region: CaptureRect | None = None
    target_handle: int = 0
    display_name: str = ""
    cursor_position: QPoint | None = None
    open_editor: bool = True
    metadata: dict[str, object] = field(default_factory=dict, compare=False, hash=False)
    include_cursor: bool = False

    @classmethod
    def from_mapping(cls, payload: dict[str, object]) -> "CaptureRequest":
        raw_region = payload.get("region")
        region = None
        if isinstance(raw_region, dict):
            try:
                region = CaptureRect(
                    int(raw_region.get("x", 0)),
                    int(raw_region.get("y", 0)),
                    int(raw_region.get("width", 0)),
                    int(raw_region.get("height", 0)),
                ).normalized()
            except (TypeError, ValueError):
                region = None
        return cls(
            mode=CaptureMode.parse(payload.get("mode", CaptureMode.REGION.value)),
            delay_ms=max(0, min(60_000, int(payload.get("delay_ms", 0) or 0))),
            region=region,
            target_handle=max(0, int(payload.get("target_handle", 0) or 0)),
            display_name=str(payload.get("display_name", "") or ""),
            include_cursor=bool(payload.get("include_cursor", False)),
            open_editor=bool(payload.get("open_editor", True)),
            metadata={key: value for key, value in payload.items() if key not in {
                "mode", "delay_ms", "region", "target_handle", "display_name",
                "include_cursor", "open_editor"
            }},
        )


@dataclass(frozen=True, slots=True)
class CaptureSelection:
    rect: CaptureRect
    candidate: WindowCandidate | None = None
    display_name: str = ""


@dataclass(slots=True)
class CapturedFrame:
    image: QImage
    rect: CaptureRect
    mode: CaptureMode
    target_handle: int = 0
    display_name: str = ""
    device_pixel_ratio: float = 1.0
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def valid(self) -> bool:
        return self.rect.valid and not self.image.isNull()


class ScreenshotBackend(Protocol):
    def screens(self) -> Sequence[ScreenInfo]: ...

    def windows(self, *, include_children: bool = True) -> Sequence[WindowCandidate]: ...

    def active_window_handle(self) -> int: ...

    def capture_rect(self, rect: CaptureRect, *, include_cursor: bool = False) -> QImage: ...

    def capture_window(
        self,
        candidate: WindowCandidate,
        *,
        include_cursor: bool = False,
    ) -> QImage: ...


def union_rect(rectangles: Iterable[CaptureRect]) -> CaptureRect | None:
    valid = [item.normalized() for item in rectangles if item.valid]
    if not valid:
        return None
    left = min(item.x for item in valid)
    top = min(item.y for item in valid)
    right = max(item.right for item in valid)
    bottom = max(item.bottom for item in valid)
    return CaptureRect(left, top, right - left, bottom - top)


def qt_screen_infos() -> tuple[ScreenInfo, ...]:
    app = QGuiApplication.instance()
    if app is None:
        return ()
    primary = app.primaryScreen()
    result: list[ScreenInfo] = []
    for screen in app.screens():
        geometry = CaptureRect.from_qrect(screen.geometry())
        dpr = max(0.1, float(screen.devicePixelRatio()))
        # Qt exposes screen geometry in device-independent pixels.  Keep the
        # local conversion explicit; native Windows backends should replace
        # this inferred origin with EnumDisplayMonitors physical coordinates.
        physical = CaptureRect(
            round(geometry.x * dpr),
            round(geometry.y * dpr),
            max(1, round(geometry.width * dpr)),
            max(1, round(geometry.height * dpr)),
        )
        result.append(
            ScreenInfo(
                name=screen.name(),
                logical_rect=geometry,
                physical_rect=physical,
                device_pixel_ratio=dpr,
                primary=screen is primary,
            )
        )
    return tuple(result)


def windows_screen_infos() -> tuple[ScreenInfo, ...]:
    """Return exact Win32 physical monitor origins paired with Qt logical screens."""

    if sys.platform != "win32":
        return qt_screen_infos()
    app = QGuiApplication.instance()
    if app is None:
        return ()
    try:
        import ctypes
        from ctypes import wintypes

        class _MonitorInfoEx(ctypes.Structure):
            _fields_ = [
                ("cbSize", wintypes.DWORD),
                ("rcMonitor", wintypes.RECT),
                ("rcWork", wintypes.RECT),
                ("dwFlags", wintypes.DWORD),
                ("szDevice", wintypes.WCHAR * 32),
            ]

        user32 = ctypes.WinDLL("user32", use_last_error=True)
        callback_type = ctypes.WINFUNCTYPE(
            wintypes.BOOL,
            wintypes.HMONITOR,
            wintypes.HDC,
            ctypes.POINTER(wintypes.RECT),
            wintypes.LPARAM,
        )
        native: list[tuple[str, CaptureRect, bool]] = []

        @callback_type
        def collect(monitor, _hdc, _rect, _data):
            info = _MonitorInfoEx()
            info.cbSize = ctypes.sizeof(info)
            if user32.GetMonitorInfoW(monitor, ctypes.byref(info)):
                rect = info.rcMonitor
                native.append(
                    (
                        str(info.szDevice),
                        CaptureRect(
                            int(rect.left),
                            int(rect.top),
                            int(rect.right - rect.left),
                            int(rect.bottom - rect.top),
                        ),
                        bool(info.dwFlags & 1),
                    )
                )
            return True

        if not user32.EnumDisplayMonitors(None, None, collect, 0) or not native:
            return qt_screen_infos()
    except (AttributeError, OSError, TypeError, ValueError):
        return qt_screen_infos()

    qt_screens = list(app.screens())
    by_name = {name.casefold(): (rect, primary) for name, rect, primary in native}
    result: list[ScreenInfo] = []
    for index, screen in enumerate(qt_screens):
        logical = CaptureRect.from_qrect(screen.geometry())
        match = by_name.get(screen.name().casefold())
        if match is None and len(native) == len(qt_screens):
            logical_order = sorted(qt_screens, key=lambda item: (item.geometry().x(), item.geometry().y()))
            native_order = sorted(native, key=lambda item: (item[1].x, item[1].y))
            position = logical_order.index(screen)
            _device, physical, is_primary = native_order[position]
        elif match is not None:
            physical, is_primary = match
        else:
            dpr = max(0.1, float(screen.devicePixelRatio()))
            physical = CaptureRect(
                round(logical.x * dpr),
                round(logical.y * dpr),
                max(1, round(logical.width * dpr)),
                max(1, round(logical.height * dpr)),
            )
            is_primary = index == 0
        result.append(
            ScreenInfo(
                name=screen.name(),
                logical_rect=logical,
                physical_rect=physical,
                device_pixel_ratio=physical.width / max(1, logical.width),
                primary=is_primary,
            )
        )
    return tuple(result)


class QtScreenshotBackend:
    """Portable fallback used by tests and non-Windows desktops.

    On Windows the companion process should receive a native backend so
    occluded-window capture and mixed-DPI physical monitor origins are exact.
    """

    def __init__(
        self,
        *,
        screen_provider: Callable[[], Sequence[ScreenInfo]] | None = None,
        window_provider: Callable[[bool], Sequence[WindowCandidate]] | None = None,
        active_window_provider: Callable[[], int] | None = None,
    ) -> None:
        self._screen_provider = screen_provider or qt_screen_infos
        self._window_provider = window_provider or (lambda _children: ())
        self._active_window_provider = active_window_provider or (lambda: 0)

    def screens(self) -> Sequence[ScreenInfo]:
        return tuple(self._screen_provider())

    def windows(self, *, include_children: bool = True) -> Sequence[WindowCandidate]:
        return tuple(self._window_provider(bool(include_children)))

    def active_window_handle(self) -> int:
        return max(0, int(self._active_window_provider()))

    def capture_window(
        self,
        candidate: WindowCandidate,
        *,
        include_cursor: bool = False,
    ) -> QImage:
        app = QGuiApplication.instance()
        if app is None or candidate.handle <= 0:
            return self.capture_rect(candidate.capture_rect, include_cursor=include_cursor)
        point = QPoint(candidate.capture_rect.x, candidate.capture_rect.y)
        screen = app.screenAt(point) or app.primaryScreen()
        if screen is None:
            return QImage()
        pixmap = screen.grabWindow(int(candidate.handle))
        if not pixmap.isNull():
            return pixmap.toImage().copy()
        return self.capture_rect(candidate.capture_rect, include_cursor=include_cursor)

    def capture_rect(self, rect: CaptureRect, *, include_cursor: bool = False) -> QImage:
        rect = rect.normalized()
        if not rect.valid:
            return QImage()
        screens = tuple(self.screens())
        app = QGuiApplication.instance()
        if app is None or not screens:
            return QImage()
        canvas = QImage(rect.width, rect.height, QImage.Format.Format_ARGB32_Premultiplied)
        canvas.fill(0)
        painter = QPainter(canvas)
        try:
            qt_screens = {screen.name(): screen for screen in app.screens()}
            for info in screens:
                fragment = rect.intersection(info.physical_rect)
                if fragment is None:
                    continue
                qt_screen = qt_screens.get(info.name)
                if qt_screen is None:
                    continue
                logical = info.physical_fragment_to_logical(fragment)
                local_x = logical.x - info.logical_rect.x
                local_y = logical.y - info.logical_rect.y
                pixmap = qt_screen.grabWindow(
                    0,
                    local_x,
                    local_y,
                    logical.width,
                    logical.height,
                )
                if pixmap.isNull():
                    continue
                image = pixmap.toImage()
                destination = QRect(
                    fragment.x - rect.x,
                    fragment.y - rect.y,
                    fragment.width,
                    fragment.height,
                )
                painter.drawImage(destination, image)
        finally:
            painter.end()
        return canvas


class WindowsScreenshotBackend:
    """Win32 capture adapter with a portable Qt screen-description fallback.

    The Win32 imports are intentionally delayed until construction so importing
    the screenshot companion on macOS/Linux remains harmless.
    """

    def __init__(
        self,
        *,
        screen_provider: Callable[[], Sequence[ScreenInfo]] | None = None,
        window_enumerator: Callable[[], object] | None = None,
        screen_capture: object | None = None,
        active_window_provider: Callable[[], int] | None = None,
        cu5_locator: object | None = None,
    ) -> None:
        if window_enumerator is None:
            from fdm.platform.windows_window_locator import enumerate_windows

            window_enumerator = enumerate_windows
        if screen_capture is None:
            from fdm.platform.windows_screen_capture import WindowsScreenCapture

            screen_capture = WindowsScreenCapture()
        self._screen_provider = screen_provider or windows_screen_infos
        self._window_enumerator = window_enumerator
        self._screen_capture = screen_capture
        self._active_window_provider = active_window_provider or _foreground_window_handle
        self._cu5_locator = cu5_locator
        self._last_cu5_match: object | None = None

    @property
    def last_cu5_match(self) -> object | None:
        return self._last_cu5_match

    def set_cu5_locator(self, locator: object | None) -> None:
        """Use the agent-owned, settings-aware locator for CU-5 captures."""

        self._cu5_locator = locator
        self._last_cu5_match = None

    def screens(self) -> Sequence[ScreenInfo]:
        return tuple(self._screen_provider())

    def _records(self) -> tuple[object, ...]:
        snapshot = self._window_enumerator()
        return tuple(getattr(snapshot, "records", snapshot))

    @staticmethod
    def _candidate_from_record(record: object, *, z_order: int = 0) -> WindowCandidate | None:
        physical = record.rect
        rect = CaptureRect(
            int(physical.left),
            int(physical.top),
            int(physical.width),
            int(physical.height),
        )
        if not rect.valid:
            return None
        parent = getattr(record, "parent_hwnd", None)
        ancestors = tuple(getattr(record, "ancestor_hwnds", ()))
        return WindowCandidate(
            handle=int(record.hwnd),
            rect=rect,
            title=str(getattr(record, "title", "")),
            class_name=str(getattr(record, "class_name", "")),
            process_id=int(getattr(record, "pid", 0)),
            executable=str(getattr(record, "process_path", "")),
            parent_handle=int(parent or 0),
            depth=len(ancestors),
            z_order=int(z_order),
            visible=bool(getattr(record, "visible", False)),
            minimized=bool(getattr(record, "minimized", False)),
            capture_safe=bool(getattr(record, "available_for_capture", True)),
            metadata={
                "root_handle": int(getattr(record, "root_hwnd", 0)),
                "ancestor_handles": ancestors,
                "control_id": getattr(record, "control_id", None),
                "cloaked": bool(getattr(record, "cloaked", False)),
            },
        )

    def windows(self, *, include_children: bool = True) -> Sequence[WindowCandidate]:
        records = self._records()
        result: list[WindowCandidate] = []
        for z_order, record in enumerate(records):
            parent = getattr(record, "parent_hwnd", None)
            if not include_children and parent is not None:
                continue
            candidate = self._candidate_from_record(record, z_order=z_order)
            if candidate is not None:
                result.append(candidate)
        return tuple(result)

    def locate_cu5_candidate(self) -> WindowCandidate:
        from fdm.services.cu5_preview_locator import Cu5PreviewLocator

        snapshot = self._window_enumerator()
        locator = self._cu5_locator or Cu5PreviewLocator()
        match = locator.locate(snapshot)
        self._last_cu5_match = match
        candidate = self._candidate_from_record(match.record)
        if candidate is None:
            raise RuntimeError("CU-5 实时预览区域为空。")
        return replace(
            candidate,
            metadata={**candidate.metadata, "cu5_preview": True},
        )

    def active_window_handle(self) -> int:
        return max(0, int(self._active_window_provider()))

    @staticmethod
    def _physical_rect(rect: CaptureRect):
        from fdm.platform.windows_window_locator import PhysicalRect

        normalized = rect.normalized()
        return PhysicalRect(
            normalized.x,
            normalized.y,
            normalized.right,
            normalized.bottom,
        )

    def capture_rect(self, rect: CaptureRect, *, include_cursor: bool = False) -> QImage:
        physical = self._physical_rect(rect)
        result = (
            self._screen_capture.capture_rect(physical, include_cursor=True)
            if include_cursor
            else self._screen_capture.capture_rect(physical)
        )
        return result.frame.to_qimage()

    def capture_window(
        self,
        candidate: WindowCandidate,
        *,
        include_cursor: bool = False,
    ) -> QImage:
        physical = self._physical_rect(candidate.capture_rect)
        validate = bool(candidate.metadata.get("cu5_preview", False))
        options: dict[str, object] = {"rect": physical}
        if validate:
            options["validate"] = True
        if include_cursor:
            options["include_cursor"] = True
        result = self._screen_capture.capture_window(candidate.handle, **options)
        return result.frame.to_qimage()


def _foreground_window_handle() -> int:
    if sys.platform != "win32":
        return 0
    try:
        import ctypes

        return int(ctypes.windll.user32.GetForegroundWindow())
    except (AttributeError, OSError, TypeError, ValueError):
        return 0


def default_screenshot_backend() -> ScreenshotBackend:
    if sys.platform == "win32":
        try:
            return WindowsScreenshotBackend()
        except Exception:  # noqa: BLE001 - retain portable fallback at startup
            pass
    return QtScreenshotBackend()


def candidate_at_point(
    candidates: Sequence[WindowCandidate],
    point: QPoint,
) -> tuple[WindowCandidate, ...]:
    """Return nested candidates from the smallest/deepest to the largest."""

    hits = [candidate for candidate in candidates if candidate.visible and candidate.contains(point)]
    hits.sort(
        key=lambda candidate: (
            -candidate.depth,
            candidate.capture_rect.area,
            candidate.z_order,
            candidate.handle,
        )
    )
    return tuple(hits)


def rank_cu5_candidates(candidates: Sequence[WindowCandidate]) -> tuple[WindowCandidate, ...]:
    def score(candidate: WindowCandidate) -> tuple[float, int, int]:
        title = candidate.title.casefold()
        cls = candidate.class_name.casefold()
        executable = candidate.executable.replace("\\", "/").casefold()
        width = max(1, candidate.capture_rect.width)
        height = max(1, candidate.capture_rect.height)
        ratio_error = abs(width / height - 4.0 / 3.0)
        value = 0.0
        if executable.endswith("/cu-5.exe") or executable == "cu-5.exe":
            value += 100.0
        if candidate.title == "用来显示SDK摄像头的窗口":
            value += 80.0
        if cls == "afxwnd100s":
            value += 40.0
        if cls == "afxframeorview100s":
            value += 10.0
        if title.startswith("cu -"):
            value += 15.0
        if candidate.depth > 0:
            value += 8.0
        if candidate.visible and not candidate.minimized:
            value += 5.0
        if ratio_error <= 0.04:
            value += 30.0
        elif ratio_error <= 0.12:
            value += 12.0
        value += min(20.0, candidate.capture_rect.area / (768 * 576) * 4.0)
        if not candidate.capture_safe or candidate.minimized:
            value -= 100.0
        return value, candidate.depth, -candidate.z_order

    filtered = [
        candidate
        for candidate in candidates
        if (
            candidate.executable.replace("\\", "/").casefold().endswith("/cu-5.exe")
            or candidate.executable.casefold() == "cu-5.exe"
            or candidate.title.casefold().startswith("cu -")
            or candidate.title == "用来显示SDK摄像头的窗口"
        )
    ]
    filtered.sort(key=score, reverse=True)
    return tuple(filtered)


class CaptureCoordinator(QObject):
    captureReady = Signal(object)
    captureFailed = Signal(str)
    selectionRequested = Signal(object, object)
    cancelled = Signal()

    def __init__(self, backend: ScreenshotBackend | None = None, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._backend = backend or default_screenshot_backend()
        self._last_region: CaptureRect | None = None
        self._pending_request: CaptureRequest | None = None
        self._request_generation = 0
        self._delayed_generation: int | None = None
        self._delay_timer = QTimer(self)
        self._delay_timer.setSingleShot(True)
        self._delay_timer.timeout.connect(self._execute_delayed)

    @property
    def backend(self) -> ScreenshotBackend:
        return self._backend

    @property
    def last_region(self) -> CaptureRect | None:
        return self._last_region

    def set_last_region(self, rect: CaptureRect | None) -> None:
        self._last_region = rect.normalized() if rect is not None and rect.valid else None

    def screens(self) -> tuple[ScreenInfo, ...]:
        return tuple(self._backend.screens())

    def candidates(self, request: CaptureRequest) -> tuple[WindowCandidate, ...]:
        candidates = tuple(self._backend.windows(include_children=True))
        if request.mode is CaptureMode.CU5:
            return rank_cu5_candidates(candidates)
        if request.mode is CaptureMode.ACTIVE_WINDOW:
            handle = request.target_handle or self._backend.active_window_handle()
            return tuple(item for item in candidates if item.handle == handle)
        if request.cursor_position is not None:
            return candidate_at_point(candidates, request.cursor_position)
        return tuple(
            sorted(
                (item for item in candidates if item.visible and not item.minimized),
                key=lambda item: (item.z_order, -item.depth, -item.capture_rect.area),
            )
        )

    def start(self, request: CaptureRequest) -> None:
        self.cancel(emit_signal=False)
        self._request_generation += 1
        generation = self._request_generation
        self._pending_request = request
        if request.delay_ms > 0:
            self._delayed_generation = generation
            self._delay_timer.start(request.delay_ms)
        else:
            self._execute_pending(generation)

    def cancel(self, *, emit_signal: bool = True) -> None:
        had_pending = self._pending_request is not None
        self._delay_timer.stop()
        self._delayed_generation = None
        self._pending_request = None
        self._request_generation += 1
        if emit_signal and had_pending:
            self.cancelled.emit()

    def complete_selection(
        self,
        selection: CaptureSelection,
        *,
        expected_request: CaptureRequest | None = None,
    ) -> None:
        request = self._pending_request
        if request is None:
            return
        # An overlay hides briefly before capture so it is not present in the
        # resulting frame.  During that interval another global hotkey may
        # replace the pending request.  Never let the old overlay complete the
        # newer request with stale coordinates.
        if expected_request is not None and request is not expected_request:
            return
        self._delay_timer.stop()
        self._delayed_generation = None
        self._pending_request = None
        self._capture_selection(request, selection)

    def capture_now(
        self,
        request: CaptureRequest,
        selection: CaptureSelection | None = None,
    ) -> CapturedFrame:
        if selection is not None:
            return self._capture_frame(request, selection)
        resolved = self._resolve_immediate_selection(request)
        if resolved is None:
            raise RuntimeError("该截图模式需要用户选择区域或窗口。")
        return self._capture_frame(request, resolved)

    def _execute_delayed(self) -> None:
        generation = self._delayed_generation
        self._delayed_generation = None
        if generation is not None:
            self._execute_pending(generation)

    def _execute_pending(self, generation: int | None = None) -> None:
        if generation is not None and generation != self._request_generation:
            return
        request = self._pending_request
        if request is None:
            return
        try:
            selection = self._resolve_immediate_selection(request)
        except Exception as exc:  # noqa: BLE001 - normalize capture-mode resolution
            self._pending_request = None
            self.captureFailed.emit(str(exc) or type(exc).__name__)
            return
        if selection is not None:
            self._pending_request = None
            self._capture_selection(request, selection)
            return
        self.selectionRequested.emit(request, self.candidates(request))

    def _resolve_immediate_selection(self, request: CaptureRequest) -> CaptureSelection | None:
        if request.mode is CaptureMode.REGION:
            return CaptureSelection(request.region) if request.region is not None and request.region.valid else None
        if request.mode is CaptureMode.LAST_REGION:
            return CaptureSelection(self._last_region) if self._last_region is not None else None
        if request.mode is CaptureMode.FULL_SCREEN:
            desktop = union_rect(screen.physical_rect for screen in self.screens())
            return CaptureSelection(desktop) if desktop is not None else None
        if request.mode is CaptureMode.DISPLAY:
            screens = self.screens()
            target = next((item for item in screens if item.name == request.display_name), None)
            if target is None and request.cursor_position is not None:
                target = next(
                    (item for item in screens if item.physical_rect.contains(
                        request.cursor_position.x(), request.cursor_position.y()
                    )),
                    None,
                )
            if target is None:
                target = next((item for item in screens if item.primary), screens[0] if screens else None)
            return CaptureSelection(target.physical_rect, display_name=target.name) if target is not None else None
        if request.mode is CaptureMode.ACTIVE_WINDOW or request.target_handle:
            candidates = self.candidates(request)
            target = next(
                (item for item in candidates if item.handle == request.target_handle),
                candidates[0] if candidates else None,
            )
            if target is None:
                return None
            return CaptureSelection(target.capture_rect, candidate=target)
        if request.mode is CaptureMode.CU5:
            locator_error: Exception | None = None
            native_locator = getattr(self._backend, "locate_cu5_candidate", None)
            if callable(native_locator):
                try:
                    target = native_locator()
                    return CaptureSelection(target.capture_rect, candidate=target)
                except Exception as exc:  # noqa: BLE001 - report after safe fallbacks
                    locator_error = exc
            elif sys.platform == "win32":
                try:
                    from fdm.services.cu5_preview_locator import Cu5PreviewLocator

                    match = Cu5PreviewLocator().locate()
                    target = next(
                        (item for item in self.candidates(request) if item.handle == match.hwnd),
                        None,
                    )
                    if target is not None:
                        return CaptureSelection(target.capture_rect, candidate=target)
                except Exception as exc:  # noqa: BLE001 - report after safe fallbacks
                    locator_error = exc
            ranked = self.candidates(request)
            target = ranked[0] if ranked else None
            if target is not None and target.title == "用来显示SDK摄像头的窗口":
                return CaptureSelection(target.capture_rect, candidate=target)
            if locator_error is not None:
                raise RuntimeError(str(locator_error) or "CU-5 实时预览定位失败。") from locator_error
            if ranked:
                raise RuntimeError(
                    f"CU-5 中有 {len(ranked)} 个候选区域，但无法可靠确认视频画面；请先运行 CU-5 诊断。"
                )
            raise RuntimeError("未识别到 CU-5 实时预览区域；请确认窗口可见且未最小化。")
        return None

    def _capture_selection(self, request: CaptureRequest, selection: CaptureSelection) -> None:
        try:
            frame = self._capture_frame(request, selection)
        except Exception as exc:  # noqa: BLE001 - user-facing capture boundary
            self.captureFailed.emit(str(exc) or exc.__class__.__name__)
            return
        self.captureReady.emit(frame)

    def _capture_frame(self, request: CaptureRequest, selection: CaptureSelection) -> CapturedFrame:
        rect = selection.rect.normalized()
        if not rect.valid:
            raise ValueError("截图区域为空。")
        candidate = selection.candidate
        if candidate is not None and candidate.minimized:
            raise RuntimeError("目标窗口已最小化，无法可靠截图。")
        if request.mode is CaptureMode.CU5 and candidate is not None:
            # Every CU-5 route, including the legacy exact-title fallback,
            # must request black/uniform-frame validation from the Windows
            # backend.  Otherwise a failed DirectDraw/overlay capture could be
            # silently published as a valid image.
            candidate = replace(
                candidate,
                metadata={**candidate.metadata, "cu5_preview": True},
            )
        method = self._backend.capture_window if candidate is not None else self._backend.capture_rect
        argument = candidate if candidate is not None else rect
        image = _capture_with_optional_cursor(method, argument, request.include_cursor)
        if image is None or image.isNull():
            raise RuntimeError("未能取得有效截图。")
        if request.mode in {CaptureMode.REGION, CaptureMode.LAST_REGION}:
            self._last_region = rect
        return CapturedFrame(
            image=image.copy(),
            rect=rect,
            mode=request.mode,
            target_handle=candidate.handle if candidate is not None else 0,
            display_name=selection.display_name,
            device_pixel_ratio=1.0,
            metadata={
                **request.metadata,
                "coordinate_space": "native_physical_pixels",
                "include_cursor": request.include_cursor,
                "source_title": candidate.title if candidate is not None else "",
                "source_class": candidate.class_name if candidate is not None else "",
            },
        )


def _capture_with_optional_cursor(method, argument, include_cursor: bool) -> QImage:
    """Call modern backends with cursor intent while retaining legacy fakes/plugins."""

    if not include_cursor:
        return method(argument)
    try:
        parameters = inspect.signature(method).parameters.values()
    except (TypeError, ValueError):
        parameters = ()
    supports_option = any(
        parameter.name == "include_cursor"
        or parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    )
    return method(argument, include_cursor=True) if supports_option else method(argument)


__all__ = [
    "CaptureCoordinator",
    "CaptureMode",
    "CaptureRect",
    "CaptureRequest",
    "CaptureSelection",
    "CapturedFrame",
    "QtScreenshotBackend",
    "ScreenInfo",
    "ScreenshotBackend",
    "WindowCandidate",
    "WindowsScreenshotBackend",
    "candidate_at_point",
    "default_screenshot_backend",
    "qt_screen_infos",
    "rank_cu5_candidates",
    "union_rect",
    "windows_screen_infos",
]
