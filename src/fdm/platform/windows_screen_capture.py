from __future__ import annotations

import ctypes
from dataclasses import dataclass
import sys
from typing import Protocol

from fdm.platform.windows_window_locator import PhysicalRect


SRCCOPY = 0x00CC0020
CAPTUREBLT = 0x40000000
DIB_RGB_COLORS = 0
BI_RGB = 0
PW_RENDERFULLCONTENT = 0x00000002
CURSOR_SHOWING = 0x00000001
DI_NORMAL = 0x0003


class WindowsScreenCaptureUnavailableError(RuntimeError):
    pass


class ScreenCaptureError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class CapturedFrame:
    width: int
    height: int
    stride: int
    bgra: bytes

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("截图尺寸必须为正数。")
        if self.stride < self.width * 4:
            raise ValueError("截图行跨度小于 BGRA 像素数据宽度。")
        if len(self.bgra) < self.stride * self.height:
            raise ValueError("截图像素数据长度不足。")

    def to_qimage(self):
        """Return a detached, opaque QImage without importing Qt eagerly.

        ``GetDIBits`` with a 32-bit ``BI_RGB`` bitmap returns BGRX pixels: the
        fourth byte is reserved rather than a meaningful alpha channel.  GDI
        commonly leaves it at zero, so interpreting the buffer as ARGB would
        turn an otherwise valid screenshot fully transparent when saved as
        PNG.  ``Format_RGB32`` has the same little-endian byte layout while Qt
        treats every pixel as opaque.
        """

        from PySide6.QtGui import QImage

        image = QImage(
            self.bgra,
            self.width,
            self.height,
            self.stride,
            QImage.Format.Format_RGB32,
        )
        return image.copy()


@dataclass(frozen=True, slots=True)
class FrameQuality:
    acceptable: bool
    reason: str
    sampled_pixels: int
    unique_colors: int
    minimum_luma: float
    maximum_luma: float
    is_black: bool
    is_uniform: bool


@dataclass(frozen=True, slots=True)
class CaptureResult:
    frame: CapturedFrame
    method: str
    quality: FrameQuality
    fallback_reason: str = ""


@dataclass(frozen=True, slots=True)
class CapturedFramePair:
    """One native draw observed before and after optional cursor composition."""

    base: CapturedFrame
    decorated: CapturedFrame


def assess_frame_quality(
    frame: CapturedFrame,
    *,
    maximum_samples: int = 4096,
) -> FrameQuality:
    """Detect the black or single-colour frames common with failed GPU capture."""

    total_pixels = frame.width * frame.height
    sample_step = max(1, total_pixels // max(1, int(maximum_samples)))
    view = memoryview(frame.bgra)
    min_channels = [255, 255, 255]
    max_channels = [0, 0, 0]
    luma_min = 255.0
    luma_max = 0.0
    luma_sum = 0.0
    samples = 0
    colors: set[tuple[int, int, int]] = set()

    for pixel_index in range(0, total_pixels, sample_step):
        row, column = divmod(pixel_index, frame.width)
        offset = row * frame.stride + column * 4
        blue = int(view[offset])
        green = int(view[offset + 1])
        red = int(view[offset + 2])
        rgb = (red, green, blue)
        if len(colors) < 65:
            colors.add(rgb)
        for index, value in enumerate(rgb):
            min_channels[index] = min(min_channels[index], value)
            max_channels[index] = max(max_channels[index], value)
        luma = (0.2126 * red) + (0.7152 * green) + (0.0722 * blue)
        luma_min = min(luma_min, luma)
        luma_max = max(luma_max, luma)
        luma_sum += luma
        samples += 1

    if samples == 0:  # pragma: no cover - CapturedFrame prevents empty images
        return FrameQuality(False, "empty", 0, 0, 0.0, 0.0, True, True)
    average_luma = luma_sum / samples
    is_black = max(max_channels) <= 8 or (average_luma <= 2.0 and luma_max <= 8.0)
    channel_span = max(
        maximum - minimum
        for minimum, maximum in zip(min_channels, max_channels, strict=True)
    )
    is_uniform = channel_span <= 2 or len(colors) <= 1
    if is_black:
        reason = "black_frame"
    elif is_uniform:
        reason = "uniform_frame"
    else:
        reason = "ok"
    return FrameQuality(
        acceptable=not (is_black or is_uniform),
        reason=reason,
        sampled_pixels=samples,
        unique_colors=min(len(colors), 65),
        minimum_luma=luma_min,
        maximum_luma=luma_max,
        is_black=is_black,
        is_uniform=is_uniform,
    )


class ScreenCaptureNativeApi(Protocol):
    def capture_rect(
        self,
        rect: PhysicalRect,
        *,
        include_cursor: bool = False,
    ) -> CapturedFrame: ...

    def print_window(
        self,
        hwnd: int,
        rect: PhysicalRect,
        *,
        include_cursor: bool = False,
    ) -> CapturedFrame: ...

    def get_window_rect(self, hwnd: int) -> PhysicalRect: ...


class _BitmapInfoHeader(ctypes.Structure):
    _fields_ = [
        ("biSize", ctypes.c_ulong),
        ("biWidth", ctypes.c_long),
        ("biHeight", ctypes.c_long),
        ("biPlanes", ctypes.c_ushort),
        ("biBitCount", ctypes.c_ushort),
        ("biCompression", ctypes.c_ulong),
        ("biSizeImage", ctypes.c_ulong),
        ("biXPelsPerMeter", ctypes.c_long),
        ("biYPelsPerMeter", ctypes.c_long),
        ("biClrUsed", ctypes.c_ulong),
        ("biClrImportant", ctypes.c_ulong),
    ]


class _BitmapInfo(ctypes.Structure):
    _fields_ = [
        ("bmiHeader", _BitmapInfoHeader),
        ("bmiColors", ctypes.c_ulong * 3),
    ]


class _Rect(ctypes.Structure):
    _fields_ = [
        ("left", ctypes.c_long),
        ("top", ctypes.c_long),
        ("right", ctypes.c_long),
        ("bottom", ctypes.c_long),
    ]


class _Point(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


class _CursorInfo(ctypes.Structure):
    _fields_ = [
        ("cbSize", ctypes.c_ulong),
        ("flags", ctypes.c_ulong),
        ("hCursor", ctypes.c_void_p),
        ("ptScreenPos", _Point),
    ]


class _IconInfo(ctypes.Structure):
    _fields_ = [
        ("fIcon", ctypes.c_int),
        ("xHotspot", ctypes.c_ulong),
        ("yHotspot", ctypes.c_ulong),
        ("hbmMask", ctypes.c_void_p),
        ("hbmColor", ctypes.c_void_p),
    ]


def _cursor_draw_origin(
    *,
    cursor_x: int,
    cursor_y: int,
    hotspot_x: int,
    hotspot_y: int,
    capture_rect: PhysicalRect,
) -> tuple[int, int]:
    """Return the cursor bitmap origin in capture-local physical pixels."""

    return (
        int(cursor_x) - int(hotspot_x) - capture_rect.left,
        int(cursor_y) - int(hotspot_y) - capture_rect.top,
    )


class _CtypesScreenCaptureApi:
    def __init__(self) -> None:
        if sys.platform != "win32":
            raise WindowsScreenCaptureUnavailableError(
                "屏幕捕获仅能在 Windows 上运行。"
            )
        win_dll = getattr(ctypes, "WinDLL", None)
        if win_dll is None:  # pragma: no cover - defensive platform guard
            raise WindowsScreenCaptureUnavailableError(
                "当前 Python 运行时不提供 Win32 API。"
            )
        self._user32 = win_dll("user32", use_last_error=True)
        self._gdi32 = win_dll("gdi32", use_last_error=True)
        self._configure_signatures()

    def _configure_signatures(self) -> None:
        self._user32.GetDC.argtypes = [ctypes.c_void_p]
        self._user32.GetDC.restype = ctypes.c_void_p
        self._user32.ReleaseDC.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self._user32.ReleaseDC.restype = ctypes.c_int
        self._user32.PrintWindow.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint,
        ]
        self._user32.PrintWindow.restype = ctypes.c_int
        self._user32.GetWindowRect.argtypes = [ctypes.c_void_p, ctypes.POINTER(_Rect)]
        self._user32.GetWindowRect.restype = ctypes.c_int
        self._user32.GetCursorInfo.argtypes = [ctypes.POINTER(_CursorInfo)]
        self._user32.GetCursorInfo.restype = ctypes.c_int
        self._user32.GetIconInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(_IconInfo)]
        self._user32.GetIconInfo.restype = ctypes.c_int
        self._user32.DrawIconEx.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_uint,
            ctypes.c_void_p,
            ctypes.c_uint,
        ]
        self._user32.DrawIconEx.restype = ctypes.c_int
        self._gdi32.CreateCompatibleDC.argtypes = [ctypes.c_void_p]
        self._gdi32.CreateCompatibleDC.restype = ctypes.c_void_p
        self._gdi32.CreateCompatibleBitmap.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
        ]
        self._gdi32.CreateCompatibleBitmap.restype = ctypes.c_void_p
        self._gdi32.SelectObject.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self._gdi32.SelectObject.restype = ctypes.c_void_p
        self._gdi32.BitBlt.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_ulong,
        ]
        self._gdi32.BitBlt.restype = ctypes.c_int
        self._gdi32.GetDIBits.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_void_p,
            ctypes.POINTER(_BitmapInfo),
            ctypes.c_uint,
        ]
        self._gdi32.GetDIBits.restype = ctypes.c_int
        self._gdi32.DeleteObject.argtypes = [ctypes.c_void_p]
        self._gdi32.DeleteObject.restype = ctypes.c_int
        self._gdi32.DeleteDC.argtypes = [ctypes.c_void_p]
        self._gdi32.DeleteDC.restype = ctypes.c_int

    @staticmethod
    def _last_error() -> int:
        getter = getattr(ctypes, "get_last_error", None)
        return int(getter()) if callable(getter) else 0

    def capture_rect(
        self,
        rect: PhysicalRect,
        *,
        include_cursor: bool = False,
    ) -> CapturedFrame:
        return self._capture_bitmap_pair(
            rect.width,
            rect.height,
            lambda destination_dc, screen_dc: bool(
                self._gdi32.BitBlt(
                    destination_dc,
                    0,
                    0,
                    rect.width,
                    rect.height,
                    screen_dc,
                    rect.left,
                    rect.top,
                    SRCCOPY | CAPTUREBLT,
                )
            ),
            operation="BitBlt",
            capture_rect=rect,
            include_cursor=include_cursor,
        ).decorated

    def capture_rect_pair(self, rect: PhysicalRect) -> CapturedFramePair:
        return self._capture_bitmap_pair(
            rect.width,
            rect.height,
            lambda destination_dc, screen_dc: bool(
                self._gdi32.BitBlt(
                    destination_dc,
                    0,
                    0,
                    rect.width,
                    rect.height,
                    screen_dc,
                    rect.left,
                    rect.top,
                    SRCCOPY | CAPTUREBLT,
                )
            ),
            operation="BitBlt",
            capture_rect=rect,
            include_cursor=True,
        )

    def print_window(
        self,
        hwnd: int,
        rect: PhysicalRect,
        *,
        include_cursor: bool = False,
    ) -> CapturedFrame:
        return self._capture_bitmap_pair(
            rect.width,
            rect.height,
            lambda destination_dc, _screen_dc: bool(
                self._user32.PrintWindow(
                    ctypes.c_void_p(int(hwnd)),
                    destination_dc,
                    PW_RENDERFULLCONTENT,
                )
            ),
            operation="PrintWindow",
            capture_rect=rect,
            include_cursor=include_cursor,
        ).decorated

    def print_window_pair(self, hwnd: int, rect: PhysicalRect) -> CapturedFramePair:
        return self._capture_bitmap_pair(
            rect.width,
            rect.height,
            lambda destination_dc, _screen_dc: bool(
                self._user32.PrintWindow(
                    ctypes.c_void_p(int(hwnd)),
                    destination_dc,
                    PW_RENDERFULLCONTENT,
                )
            ),
            operation="PrintWindow",
            capture_rect=rect,
            include_cursor=True,
        )

    def get_window_rect(self, hwnd: int) -> PhysicalRect:
        rect = _Rect()
        if not self._user32.GetWindowRect(
            ctypes.c_void_p(int(hwnd)),
            ctypes.byref(rect),
        ):
            raise ScreenCaptureError(
                f"GetWindowRect 失败（错误 {self._last_error()}）。"
            )
        return PhysicalRect(rect.left, rect.top, rect.right, rect.bottom)

    def _capture_bitmap_pair(
        self,
        width: int,
        height: int,
        draw,
        *,
        operation: str,
        capture_rect: PhysicalRect,
        include_cursor: bool,
    ) -> CapturedFramePair:
        if width <= 0 or height <= 0:
            raise ScreenCaptureError("截图矩形为空。")
        screen_dc = self._user32.GetDC(None)
        if not screen_dc:
            raise ScreenCaptureError(
                f"GetDC 失败（错误 {self._last_error()}）。"
            )
        memory_dc = None
        bitmap = None
        old_bitmap = None
        bitmap_selected = False
        try:
            memory_dc = self._gdi32.CreateCompatibleDC(screen_dc)
            bitmap = self._gdi32.CreateCompatibleBitmap(screen_dc, width, height)
            if not memory_dc or not bitmap:
                raise ScreenCaptureError(
                    f"无法创建截图位图（错误 {self._last_error()}）。"
                )
            old_bitmap = self._gdi32.SelectObject(memory_dc, bitmap)
            if not old_bitmap:
                raise ScreenCaptureError(
                    f"无法选择截图位图（错误 {self._last_error()}）。"
                )
            bitmap_selected = True
            if not draw(memory_dc, screen_dc):
                raise ScreenCaptureError(
                    f"{operation} 失败（错误 {self._last_error()}）。"
                )
            if not self._gdi32.SelectObject(memory_dc, old_bitmap):
                raise ScreenCaptureError(
                    f"无法释放截图位图（错误 {self._last_error()}）。"
                )
            bitmap_selected = False
            base = self._read_unselected_bitmap(
                screen_dc,
                bitmap,
                width=width,
                height=height,
            )
            decorated = base
            if include_cursor:
                if not self._gdi32.SelectObject(memory_dc, bitmap):
                    raise ScreenCaptureError(
                        f"无法重新选择截图位图（错误 {self._last_error()}）。"
                    )
                bitmap_selected = True
                cursor_drawn = self._draw_cursor(memory_dc, capture_rect)
                if not self._gdi32.SelectObject(memory_dc, old_bitmap):
                    raise ScreenCaptureError(
                        f"无法释放光标截图位图（错误 {self._last_error()}）。"
                    )
                bitmap_selected = False
                if cursor_drawn:
                    decorated = self._read_unselected_bitmap(
                        screen_dc,
                        bitmap,
                        width=width,
                        height=height,
                    )
            return CapturedFramePair(base=base, decorated=decorated)
        finally:
            if bitmap_selected and old_bitmap and memory_dc:
                self._gdi32.SelectObject(memory_dc, old_bitmap)
            if bitmap:
                self._gdi32.DeleteObject(bitmap)
            if memory_dc:
                self._gdi32.DeleteDC(memory_dc)
            self._user32.ReleaseDC(None, screen_dc)

    def _read_unselected_bitmap(
        self,
        compatible_dc: int,
        bitmap: int,
        *,
        width: int,
        height: int,
    ) -> CapturedFrame:
        stride = width * 4
        payload = ctypes.create_string_buffer(stride * height)
        info = _BitmapInfo()
        info.bmiHeader.biSize = ctypes.sizeof(_BitmapInfoHeader)
        info.bmiHeader.biWidth = width
        info.bmiHeader.biHeight = -height  # top-down BGRX
        info.bmiHeader.biPlanes = 1
        info.bmiHeader.biBitCount = 32
        info.bmiHeader.biCompression = BI_RGB
        scanlines = int(
            self._gdi32.GetDIBits(
                compatible_dc,
                bitmap,
                0,
                height,
                payload,
                ctypes.byref(info),
                DIB_RGB_COLORS,
            )
        )
        if scanlines != height:
            raise ScreenCaptureError(
                f"GetDIBits 仅返回 {scanlines}/{height} 行。"
            )
        return CapturedFrame(width, height, stride, payload.raw)

    def _draw_cursor(self, destination_dc: int, capture_rect: PhysicalRect) -> bool:
        """Composite the visible system cursor into a capture bitmap.

        Cursor positions and ``PhysicalRect`` use the same virtual-desktop
        coordinate space, so monitors to the left/up of the primary display do
        not need any special clamping.
        """

        cursor = _CursorInfo()
        cursor.cbSize = ctypes.sizeof(_CursorInfo)
        if not self._user32.GetCursorInfo(ctypes.byref(cursor)):
            return False
        if not (int(cursor.flags) & CURSOR_SHOWING) or not cursor.hCursor:
            return False

        icon = _IconInfo()
        if not self._user32.GetIconInfo(cursor.hCursor, ctypes.byref(icon)):
            return False
        try:
            local_x, local_y = _cursor_draw_origin(
                cursor_x=cursor.ptScreenPos.x,
                cursor_y=cursor.ptScreenPos.y,
                hotspot_x=icon.xHotspot,
                hotspot_y=icon.yHotspot,
                capture_rect=capture_rect,
            )
            return bool(
                self._user32.DrawIconEx(
                    destination_dc,
                    local_x,
                    local_y,
                    cursor.hCursor,
                    0,
                    0,
                    0,
                    None,
                    DI_NORMAL,
                )
            )
        finally:
            if icon.hbmMask:
                self._gdi32.DeleteObject(icon.hbmMask)
            if icon.hbmColor:
                self._gdi32.DeleteObject(icon.hbmColor)


class WindowsScreenCapture:
    def __init__(self, api: ScreenCaptureNativeApi | None = None) -> None:
        self._api = api or _CtypesScreenCaptureApi()

    @staticmethod
    def _ensure_rect(rect: PhysicalRect) -> None:
        if rect.width <= 0 or rect.height <= 0:
            raise ScreenCaptureError("截图矩形为空。")

    @staticmethod
    def _validated(result: CaptureResult, *, validate: bool) -> CaptureResult:
        if validate and not result.quality.acceptable:
            raise ScreenCaptureError(
                f"截图质量检测失败：{result.quality.reason}。"
            )
        return result

    def _capture_rect_pair(self, rect: PhysicalRect) -> CapturedFramePair:
        pair_method = getattr(self._api, "capture_rect_pair", None)
        if callable(pair_method):
            pair = pair_method(rect)
            if isinstance(pair, CapturedFramePair):
                return pair
            try:
                base, decorated = pair
            except (TypeError, ValueError) as exc:
                raise ScreenCaptureError("原生屏幕捕获返回了无效的光标帧对。") from exc
            return CapturedFramePair(base=base, decorated=decorated)
        # Compatibility for existing injected capture adapters. Native GDI
        # uses the pair method above and therefore executes BitBlt only once.
        base = self._api.capture_rect(rect)
        decorated = self._api.capture_rect(rect, include_cursor=True)
        return CapturedFramePair(base=base, decorated=decorated)

    def _print_window_pair(
        self,
        hwnd: int,
        rect: PhysicalRect,
    ) -> CapturedFramePair:
        pair_method = getattr(self._api, "print_window_pair", None)
        if callable(pair_method):
            pair = pair_method(hwnd, rect)
            if isinstance(pair, CapturedFramePair):
                return pair
            try:
                base, decorated = pair
            except (TypeError, ValueError) as exc:
                raise ScreenCaptureError("原生窗口捕获返回了无效的光标帧对。") from exc
            return CapturedFramePair(base=base, decorated=decorated)
        base = self._api.print_window(hwnd, rect)
        decorated = self._api.print_window(hwnd, rect, include_cursor=True)
        return CapturedFramePair(base=base, decorated=decorated)

    def capture_rect(
        self,
        rect: PhysicalRect,
        *,
        validate: bool = False,
        include_cursor: bool = False,
    ) -> CaptureResult:
        self._ensure_rect(rect)
        if include_cursor:
            pair = self._capture_rect_pair(rect)
            base_frame = pair.base
            frame = pair.decorated
        else:
            base_frame = self._api.capture_rect(rect)
            frame = base_frame
        quality = assess_frame_quality(base_frame)
        result = CaptureResult(frame, "bitblt", quality)
        return self._validated(result, validate=validate)

    def capture_window(
        self,
        hwnd: int,
        *,
        rect: PhysicalRect | None = None,
        validate: bool = False,
        include_cursor: bool = False,
    ) -> CaptureResult:
        hwnd = int(hwnd)
        if hwnd <= 0:
            raise ValueError("窗口句柄无效。")
        target_rect = rect or self._api.get_window_rect(hwnd)
        self._ensure_rect(target_rect)
        print_failure = ""
        try:
            pair_aware_print = bool(
                include_cursor
                and callable(getattr(self._api, "print_window_pair", None))
            )
            if pair_aware_print:
                printed_pair = self._print_window_pair(hwnd, target_rect)
                printed = printed_pair.base
                printed_output = printed_pair.decorated
            else:
                printed = self._api.print_window(hwnd, target_rect)
                printed_output = printed
            print_quality = assess_frame_quality(printed)
            if print_quality.acceptable:
                if include_cursor and not pair_aware_print:
                    # Preserve the legacy injected-adapter call sequence. The
                    # native GDI adapter takes the single-draw pair path above.
                    printed_output = self._api.print_window(
                        hwnd,
                        target_rect,
                        include_cursor=True,
                    )
                return CaptureResult(printed_output, "print_window", print_quality)
            print_failure = print_quality.reason
        except Exception as exc:  # noqa: BLE001 - native adapters normalize below
            print_failure = str(exc).strip() or type(exc).__name__

        if include_cursor:
            fallback_pair = self._capture_rect_pair(target_rect)
            fallback_base = fallback_pair.base
            fallback_frame = fallback_pair.decorated
        else:
            fallback_base = self._api.capture_rect(target_rect)
            fallback_frame = fallback_base
        fallback_quality = assess_frame_quality(fallback_base)
        result = CaptureResult(
            fallback_frame,
            "bitblt",
            fallback_quality,
            fallback_reason=print_failure,
        )
        return self._validated(result, validate=validate)


__all__ = [
    "CAPTUREBLT",
    "CURSOR_SHOWING",
    "CapturedFrame",
    "CapturedFramePair",
    "CaptureResult",
    "FrameQuality",
    "PW_RENDERFULLCONTENT",
    "DI_NORMAL",
    "SRCCOPY",
    "ScreenCaptureError",
    "ScreenCaptureNativeApi",
    "WindowsScreenCapture",
    "WindowsScreenCaptureUnavailableError",
    "assess_frame_quality",
]
