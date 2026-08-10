from __future__ import annotations

import ctypes
import sys

import pytest

from fdm.platform.windows_screen_capture import (
    CapturedFrame,
    CapturedFramePair,
    ScreenCaptureError,
    WindowsScreenCapture,
    WindowsScreenCaptureUnavailableError,
    assess_frame_quality,
)
from fdm.platform import windows_screen_capture as native_capture_module
from fdm.platform.windows_window_locator import PhysicalRect


def _frame(width: int, height: int, pixel) -> CapturedFrame:
    payload = bytearray()
    for y in range(height):
        for x in range(width):
            red, green, blue = pixel(x, y)
            payload.extend((blue, green, red, 255))
    return CapturedFrame(width, height, width * 4, bytes(payload))


class _FakeCaptureApi:
    def __init__(self, *, printed: CapturedFrame | Exception, bitblt: CapturedFrame) -> None:
        self.printed = printed
        self.bitblt = bitblt
        self.calls: list[tuple[object, ...]] = []

    def capture_rect(self, rect):
        self.calls.append(("bitblt", rect))
        return self.bitblt

    def print_window(self, hwnd, rect):
        self.calls.append(("print_window", hwnd, rect))
        if isinstance(self.printed, Exception):
            raise self.printed
        return self.printed

    def get_window_rect(self, hwnd):
        self.calls.append(("get_window_rect", hwnd))
        return PhysicalRect(10, 20, 14, 24)


def test_quality_detector_rejects_black_and_pure_colour_frames() -> None:
    black = _frame(4, 4, lambda _x, _y: (0, 0, 0))
    white = _frame(4, 4, lambda _x, _y: (255, 255, 255))
    varied = _frame(4, 4, lambda x, y: (x * 40, y * 30, (x + y) * 20))

    assert assess_frame_quality(black).reason == "black_frame"
    assert assess_frame_quality(white).reason == "uniform_frame"
    assert assess_frame_quality(varied).acceptable


def test_bi_rgb_reserved_byte_is_not_interpreted_as_transparency() -> None:
    # A 32-bit BI_RGB DIB is BGRX, not BGRA.  GDI is allowed to leave the
    # reserved byte at zero; the Qt conversion must still produce an opaque
    # screenshot.
    frame = CapturedFrame(1, 1, 4, bytes((0x56, 0x34, 0x12, 0x00)))

    color = frame.to_qimage().pixelColor(0, 0)

    assert color.getRgb() == (0x12, 0x34, 0x56, 0xFF)


def test_capture_window_uses_print_window_when_frame_is_valid() -> None:
    varied = _frame(4, 4, lambda x, y: (x * 40, y * 30, (x + y) * 20))
    api = _FakeCaptureApi(printed=varied, bitblt=varied)

    result = WindowsScreenCapture(api).capture_window(42)

    assert result.method == "print_window"
    assert result.quality.acceptable
    assert [call[0] for call in api.calls] == ["get_window_rect", "print_window"]


def test_black_print_window_falls_back_to_bitblt_rectangle() -> None:
    black = _frame(4, 4, lambda _x, _y: (0, 0, 0))
    varied = _frame(4, 4, lambda x, y: (x * 40, y * 30, (x + y) * 20))
    api = _FakeCaptureApi(printed=black, bitblt=varied)

    result = WindowsScreenCapture(api).capture_window(42)

    assert result.method == "bitblt"
    assert result.fallback_reason == "black_frame"
    assert result.quality.acceptable
    assert [call[0] for call in api.calls] == [
        "get_window_rect",
        "print_window",
        "bitblt",
    ]


def test_print_window_error_falls_back_and_validation_rejects_bad_fallback() -> None:
    uniform = _frame(4, 4, lambda _x, _y: (12, 20, 30))
    api = _FakeCaptureApi(printed=ScreenCaptureError("print failed"), bitblt=uniform)

    with pytest.raises(ScreenCaptureError, match="uniform_frame"):
        WindowsScreenCapture(api).capture_window(42, validate=True)

    assert [call[0] for call in api.calls] == [
        "get_window_rect",
        "print_window",
        "bitblt",
    ]


def test_rectangle_capture_preserves_negative_origin_for_native_backend() -> None:
    varied = _frame(4, 4, lambda x, y: (x * 40, y * 30, (x + y) * 20))
    api = _FakeCaptureApi(printed=varied, bitblt=varied)
    rect = PhysicalRect(-100, 20, -96, 24)

    result = WindowsScreenCapture(api).capture_rect(rect, validate=True)

    assert result.method == "bitblt"
    assert api.calls == [("bitblt", rect)]


def test_native_capture_fails_lazily_off_windows() -> None:
    if sys.platform == "win32":
        pytest.skip("non-Windows import guard")
    with pytest.raises(WindowsScreenCaptureUnavailableError):
        WindowsScreenCapture()


def test_cursor_option_is_forwarded_to_rectangle_and_window_fallback() -> None:
    varied = _frame(4, 4, lambda x, y: (x * 40, y * 30, (x + y) * 20))
    black = _frame(4, 4, lambda _x, _y: (0, 0, 0))

    class _CursorAwareApi(_FakeCaptureApi):
        def capture_rect(self, rect, *, include_cursor=False):
            self.calls.append(("bitblt", rect, include_cursor))
            return self.bitblt

        def print_window(self, hwnd, rect, *, include_cursor=False):
            self.calls.append(("print_window", hwnd, rect, include_cursor))
            return self.printed

    api = _CursorAwareApi(printed=black, bitblt=varied)
    rect = PhysicalRect(-200, -100, -196, -96)

    WindowsScreenCapture(api).capture_rect(rect, include_cursor=True)
    WindowsScreenCapture(api).capture_window(
        42,
        rect=rect,
        include_cursor=True,
    )

    assert api.calls == [
        ("bitblt", rect, False),
        ("bitblt", rect, True),
        ("print_window", 42, rect, False),
        ("bitblt", rect, False),
        ("bitblt", rect, True),
    ]


def test_pair_aware_capture_validates_and_returns_the_same_native_draw() -> None:
    uniform = _frame(4, 4, lambda _x, _y: (20, 30, 40))
    varied = _frame(4, 4, lambda x, y: (x * 40, y * 30, (x + y) * 20))

    class _PairApi:
        def __init__(self, base: CapturedFrame) -> None:
            self.base = base
            self.rect_pair_calls = 0
            self.window_pair_calls = 0

        def capture_rect_pair(self, _rect):
            self.rect_pair_calls += 1
            return CapturedFramePair(self.base, varied)

        def print_window_pair(self, _hwnd, _rect):
            self.window_pair_calls += 1
            return CapturedFramePair(self.base, varied)

        @staticmethod
        def get_window_rect(_hwnd):
            return PhysicalRect(0, 0, 4, 4)

    rejected = _PairApi(uniform)
    with pytest.raises(ScreenCaptureError, match="uniform_frame"):
        WindowsScreenCapture(rejected).capture_rect(
            PhysicalRect(0, 0, 4, 4),
            include_cursor=True,
            validate=True,
        )
    assert rejected.rect_pair_calls == 1

    accepted = _PairApi(varied)
    result = WindowsScreenCapture(accepted).capture_window(
        42,
        include_cursor=True,
        validate=True,
    )
    assert accepted.window_pair_calls == 1
    assert result.frame is varied
    assert result.quality.acceptable


def test_native_bitmap_pair_executes_draw_callback_only_once() -> None:
    varied = _frame(2, 2, lambda x, y: (x * 80, y * 70, (x + y) * 50))
    decorated = _frame(2, 2, lambda x, y: (255 - x * 80, y * 70, 40))
    draw_calls = 0
    reads = iter((varied, decorated))
    selected_object = 104
    select_calls: list[tuple[int, int]] = []
    read_selected_states: list[int] = []

    class _User32:
        @staticmethod
        def GetDC(_hwnd):
            return 101

        @staticmethod
        def ReleaseDC(_hwnd, _dc):
            return 1

    class _Gdi32:
        @staticmethod
        def CreateCompatibleDC(_dc):
            return 102

        @staticmethod
        def CreateCompatibleBitmap(_dc, _width, _height):
            return 103

        @staticmethod
        def SelectObject(dc, selected):
            nonlocal selected_object
            previous = selected_object
            selected_object = int(selected)
            select_calls.append((int(dc), int(selected)))
            return previous

        @staticmethod
        def DeleteObject(_bitmap):
            return 1

        @staticmethod
        def DeleteDC(_dc):
            return 1

    api = object.__new__(native_capture_module._CtypesScreenCaptureApi)
    api._user32 = _User32()
    api._gdi32 = _Gdi32()

    def read_unselected(*_args, **_kwargs):
        read_selected_states.append(selected_object)
        assert selected_object != 103
        return next(reads)

    api._read_unselected_bitmap = read_unselected
    api._draw_cursor = lambda *_args, **_kwargs: True

    def draw(_destination_dc, _screen_dc):
        nonlocal draw_calls
        draw_calls += 1
        return True

    pair = api._capture_bitmap_pair(
        2,
        2,
        draw,
        operation="test",
        capture_rect=PhysicalRect(0, 0, 2, 2),
        include_cursor=True,
    )

    assert draw_calls == 1
    assert pair.base is varied
    assert pair.decorated is decorated
    assert select_calls == [
        (102, 103),
        (102, 104),
        (102, 103),
        (102, 104),
    ]
    assert read_selected_states == [104, 104]


def test_native_cursor_composition_uses_hotspot_and_negative_desktop_origin() -> None:
    draw_calls: list[tuple[object, ...]] = []
    deleted: list[int] = []

    class _User32:
        @staticmethod
        def GetCursorInfo(pointer):
            info = ctypes.cast(
                pointer,
                ctypes.POINTER(native_capture_module._CursorInfo),
            ).contents
            info.flags = native_capture_module.CURSOR_SHOWING
            info.hCursor = 123
            info.ptScreenPos.x = -75
            info.ptScreenPos.y = -25
            return 1

        @staticmethod
        def GetIconInfo(_cursor, pointer):
            info = ctypes.cast(
                pointer,
                ctypes.POINTER(native_capture_module._IconInfo),
            ).contents
            info.xHotspot = 4
            info.yHotspot = 6
            info.hbmMask = 501
            info.hbmColor = 502
            return 1

        @staticmethod
        def DrawIconEx(*args):
            draw_calls.append(args)
            return 1

    class _Gdi32:
        @staticmethod
        def DeleteObject(handle):
            deleted.append(int(handle))
            return 1

    api = object.__new__(native_capture_module._CtypesScreenCaptureApi)
    api._user32 = _User32()
    api._gdi32 = _Gdi32()

    drawn = api._draw_cursor(999, PhysicalRect(-100, -50, 100, 100))

    assert drawn
    assert len(draw_calls) == 1
    assert draw_calls[0][1:3] == (21, 19)
    assert draw_calls[0][3] == 123
    assert draw_calls[0][-1] == native_capture_module.DI_NORMAL
    assert deleted == [501, 502]


def test_cursor_pixels_cannot_make_a_failed_capture_pass_quality_validation() -> None:
    uniform = _frame(4, 4, lambda _x, _y: (20, 30, 40))
    decorated = _frame(4, 4, lambda x, y: (x * 40, y * 30, (x + y) * 20))

    class _Api:
        def capture_rect(self, _rect, *, include_cursor=False):
            return decorated if include_cursor else uniform

        def print_window(self, _hwnd, _rect, *, include_cursor=False):
            return decorated if include_cursor else uniform

        @staticmethod
        def get_window_rect(_hwnd):
            return PhysicalRect(0, 0, 4, 4)

    with pytest.raises(ScreenCaptureError, match="uniform_frame"):
        WindowsScreenCapture(_Api()).capture_rect(
            PhysicalRect(0, 0, 4, 4),
            include_cursor=True,
            validate=True,
        )
