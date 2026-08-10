from __future__ import annotations

import os
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QCoreApplication, QEvent, QPoint
from PySide6.QtGui import QColor, QImage
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from fdm.platform.windows_window_locator import (
    PhysicalRect,
    WindowRecord,
    WindowSnapshot,
)
from fdm.services.cu5_preview_locator import Cu5PreviewLocator
from fdm.services.screenshot_capture import (
    CaptureCoordinator,
    CaptureMode,
    CaptureRect,
    CaptureRequest,
    CaptureSelection,
    ScreenInfo,
    WindowCandidate,
    WindowsScreenshotBackend,
    candidate_at_point,
    rank_cu5_candidates,
)


def _app() -> QApplication:
    return QApplication.instance() or QApplication([])


def _image(color: str = "#123456") -> QImage:
    image = QImage(8, 6, QImage.Format.Format_ARGB32)
    image.fill(QColor(color))
    return image


class _Backend:
    def __init__(self) -> None:
        self.capture_calls: list[tuple[str, object]] = []
        self._screens = (
            ScreenInfo(
                "left",
                CaptureRect(-1280, 0, 1280, 720),
                CaptureRect(-2560, 0, 2560, 1440),
                2.0,
            ),
            ScreenInfo(
                "primary",
                CaptureRect(0, 0, 1920, 1080),
                CaptureRect(0, 0, 1920, 1080),
                1.0,
                True,
            ),
        )
        self._windows = (
            WindowCandidate(1, CaptureRect(10, 10, 500, 400), depth=0, z_order=0),
            WindowCandidate(2, CaptureRect(40, 40, 200, 100), parent_handle=1, depth=1, z_order=1),
        )

    def screens(self):
        return self._screens

    def windows(self, *, include_children: bool = True):
        return self._windows if include_children else self._windows[:1]

    def active_window_handle(self) -> int:
        return 2

    def capture_rect(self, rect: CaptureRect) -> QImage:
        self.capture_calls.append(("rect", rect))
        return _image()

    def capture_window(self, candidate: WindowCandidate) -> QImage:
        self.capture_calls.append(("window", candidate.handle))
        return _image("#abcdef")


def test_capture_rect_and_mixed_dpi_screen_mapping_keep_negative_origins() -> None:
    screen = ScreenInfo(
        "left",
        CaptureRect(-1280, 0, 1280, 720),
        CaptureRect(-2560, 0, 2560, 1440),
        2.0,
    )

    physical = screen.logical_fragment_to_physical(CaptureRect(-1280, 100, 640, 200))
    logical = screen.physical_fragment_to_logical(physical)

    assert physical == CaptureRect(-2560, 200, 1280, 400)
    assert logical == CaptureRect(-1280, 100, 640, 200)
    assert CaptureRect(10, 20, -6, -8).normalized() == CaptureRect(4, 12, 6, 8)


def test_nested_hit_candidates_are_deepest_then_smallest() -> None:
    candidates = (
        WindowCandidate(1, CaptureRect(0, 0, 800, 600), depth=0),
        WindowCandidate(2, CaptureRect(100, 100, 300, 200), depth=1),
        WindowCandidate(3, CaptureRect(120, 120, 100, 80), depth=2),
    )

    assert [item.handle for item in candidate_at_point(candidates, QPoint(150, 150))] == [3, 2, 1]


def test_hit_testing_never_prefers_a_deep_control_from_a_background_window() -> None:
    candidates = (
        WindowCandidate(
            100,
            CaptureRect(0, 0, 800, 600),
            z_order=0,
            metadata={"root_handle": 100},
        ),
        WindowCandidate(
            200,
            CaptureRect(0, 0, 800, 600),
            z_order=1,
            metadata={"root_handle": 200},
        ),
        WindowCandidate(
            201,
            CaptureRect(120, 120, 80, 60),
            parent_handle=200,
            depth=1,
            z_order=2,
            metadata={"root_handle": 200, "ancestor_handles": (200,)},
        ),
    )

    assert [
        item.handle for item in candidate_at_point(candidates, QPoint(150, 150))
    ] == [100]


def test_hit_testing_follows_only_the_frontmost_overlapping_sibling_branch() -> None:
    candidates = (
        WindowCandidate(
            1,
            CaptureRect(-300, 0, 900, 700),
            z_order=0,
            metadata={"root_handle": 1},
        ),
        WindowCandidate(
            2,
            CaptureRect(-200, 50, 500, 500),
            parent_handle=1,
            depth=1,
            z_order=1,
            metadata={"root_handle": 1, "ancestor_handles": (1,)},
        ),
        WindowCandidate(
            3,
            CaptureRect(-20, 100, 120, 100),
            parent_handle=1,
            depth=1,
            z_order=2,
            metadata={"root_handle": 1, "ancestor_handles": (1,)},
        ),
        WindowCandidate(
            4,
            CaptureRect(0, 120, 50, 40),
            parent_handle=3,
            depth=2,
            z_order=3,
            metadata={"root_handle": 1, "ancestor_handles": (1, 3)},
        ),
    )

    assert [
        item.handle for item in candidate_at_point(candidates, QPoint(20, 140))
    ] == [2, 1]


def test_coordinator_resolves_full_display_active_last_and_manual_selection() -> None:
    app = _app()
    backend = _Backend()
    coordinator = CaptureCoordinator(backend)

    full = coordinator.capture_now(CaptureRequest(CaptureMode.FULL_SCREEN))
    display = coordinator.capture_now(
        CaptureRequest(CaptureMode.DISPLAY, cursor_position=QPoint(-100, 100))
    )
    active = coordinator.capture_now(CaptureRequest(CaptureMode.ACTIVE_WINDOW))
    region = coordinator.capture_now(
        CaptureRequest(CaptureMode.REGION, region=CaptureRect(-100, 50, 30, 20))
    )
    last = coordinator.capture_now(CaptureRequest(CaptureMode.LAST_REGION))

    assert full.rect == CaptureRect(-2560, 0, 4480, 1440)
    assert display.display_name == "left"
    assert active.target_handle == 2
    assert region.rect == last.rect == CaptureRect(-100, 50, 30, 20)

    requested: list[object] = []
    ready: list[object] = []
    coordinator.selectionRequested.connect(lambda request, items: requested.append((request, items)))
    coordinator.captureReady.connect(ready.append)
    coordinator.start(CaptureRequest(CaptureMode.SMART))
    assert requested and len(requested[0][1]) == 2
    coordinator.complete_selection(
        CaptureSelection(backend._windows[1].capture_rect, candidate=backend._windows[1])
    )
    assert ready[-1].target_handle == 2
    assert app is QApplication.instance()


def test_region_candidates_skip_window_enumeration_while_smart_keeps_candidates() -> None:
    class _CountingBackend(_Backend):
        def __init__(self) -> None:
            super().__init__()
            self.window_calls = 0

        def windows(self, *, include_children: bool = True):
            self.window_calls += 1
            return super().windows(include_children=include_children)

    backend = _CountingBackend()
    coordinator = CaptureCoordinator(backend)

    assert coordinator.candidates(CaptureRequest(CaptureMode.REGION)) == ()
    assert backend.window_calls == 0

    assert coordinator.candidates(CaptureRequest(CaptureMode.SMART)) == backend._windows
    assert backend.window_calls == 1


def test_delayed_request_generation_cannot_fire_a_newer_request_early() -> None:
    app = _app()
    coordinator = CaptureCoordinator(_Backend())
    ready: list[object] = []
    coordinator.captureReady.connect(ready.append)
    coordinator.start(
        CaptureRequest(
            CaptureMode.REGION,
            delay_ms=30,
            region=CaptureRect(1, 1, 10, 10),
        )
    )
    coordinator.start(
        CaptureRequest(CaptureMode.REGION, region=CaptureRect(5, 5, 20, 20))
    )

    QTest.qWait(60)

    assert [frame.rect for frame in ready] == [CaptureRect(5, 5, 20, 20)]
    assert app is QApplication.instance()


def test_stale_overlay_selection_cannot_complete_a_new_delayed_request() -> None:
    app = _app()
    coordinator = CaptureCoordinator(_Backend())
    ready: list[object] = []
    coordinator.captureReady.connect(ready.append)
    old_request = CaptureRequest(CaptureMode.SMART)
    coordinator.start(old_request)
    new_request = CaptureRequest(
        CaptureMode.REGION,
        delay_ms=30,
        region=CaptureRect(5, 5, 20, 20),
    )
    coordinator.start(new_request)

    coordinator.complete_selection(
        CaptureSelection(CaptureRect(1, 1, 10, 10)),
        expected_request=old_request,
    )
    QTest.qWait(60)

    assert [frame.rect for frame in ready] == [CaptureRect(5, 5, 20, 20)]
    assert app is QApplication.instance()


def test_deleting_coordinator_cancels_its_owned_delay_timer() -> None:
    app = _app()
    backend = _Backend()
    coordinator = CaptureCoordinator(backend)
    destroyed: list[bool] = []
    coordinator.destroyed.connect(lambda *_args: destroyed.append(True))
    coordinator.start(
        CaptureRequest(
            CaptureMode.REGION,
            delay_ms=20,
            region=CaptureRect(1, 1, 10, 10),
        )
    )

    coordinator.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    QTest.qWait(40)

    assert app is QApplication.instance()
    assert destroyed
    assert backend.capture_calls == []


def test_windows_backend_adapts_records_and_detaches_native_frame() -> None:
    class _Frame:
        def to_qimage(self) -> QImage:
            return _image()

    class _Capture:
        def __init__(self) -> None:
            self.calls: list[tuple[str, object]] = []

        def capture_rect(self, rect):
            self.calls.append(("rect", rect))
            return SimpleNamespace(frame=_Frame())

        def capture_window(self, hwnd, *, rect):
            self.calls.append(("window", (hwnd, rect)))
            return SimpleNamespace(frame=_Frame())

    native_capture = _Capture()
    rect = SimpleNamespace(left=-20, top=30, right=180, bottom=130, width=200, height=100)
    record = SimpleNamespace(
        hwnd=88,
        parent_hwnd=7,
        root_hwnd=5,
        ancestor_hwnds=(5, 7),
        pid=12,
        process_path=r"C:\CU-5.exe",
        title="preview",
        class_name="CWndForSDK",
        control_id=100,
        rect=rect,
        visible=True,
        minimized=False,
        cloaked=False,
        available_for_capture=True,
    )
    backend = WindowsScreenshotBackend(
        screen_provider=lambda: (),
        window_enumerator=lambda: SimpleNamespace(records=(record,)),
        screen_capture=native_capture,
        active_window_provider=lambda: 88,
    )

    candidate = backend.windows()[0]
    image = backend.capture_window(candidate)

    assert candidate.handle == 88
    assert candidate.rect == CaptureRect(-20, 30, 200, 100)
    assert candidate.depth == 2 and candidate.metadata["control_id"] == 100
    assert backend.active_window_handle() == 88
    assert not image.isNull()
    _kind, (hwnd, physical) = native_capture.calls[-1]
    assert hwnd == 88
    assert (physical.left, physical.top, physical.right, physical.bottom) == (-20, 30, 180, 130)


def test_cu5_window_capture_enables_black_frame_validation() -> None:
    class _Frame:
        def to_qimage(self) -> QImage:
            return _image()

    class _Capture:
        def __init__(self) -> None:
            self.options: dict[str, object] = {}

        def capture_window(self, _hwnd, **options):
            self.options = dict(options)
            return SimpleNamespace(frame=_Frame())

    rect = SimpleNamespace(left=20, top=30, right=788, bottom=606, width=768, height=576)
    record = SimpleNamespace(
        hwnd=88,
        parent_hwnd=7,
        root_hwnd=5,
        ancestor_hwnds=(5, 7),
        pid=12,
        process_path=r"C:\CU-5\CU-5.exe",
        title="",
        class_name="CWndForSDK",
        control_id=100,
        rect=rect,
        visible=True,
        minimized=False,
        cloaked=False,
        available_for_capture=True,
    )
    capture = _Capture()
    locator = SimpleNamespace(
        locate=lambda _snapshot: SimpleNamespace(record=record)
    )
    backend = WindowsScreenshotBackend(
        screen_provider=lambda: (),
        window_enumerator=lambda: SimpleNamespace(records=(record,)),
        screen_capture=capture,
        active_window_provider=lambda: 88,
        cu5_locator=locator,
    )

    candidate = backend.locate_cu5_candidate()
    backend.capture_window(candidate)

    assert candidate.metadata["cu5_preview"] is True
    assert capture.options["validate"] is True


def test_cu_preview_capture_uses_static_child_without_dialog_white_borders() -> None:
    process_path = r"C:\CU-6\CU-6.exe"
    records = (
        WindowRecord(
            20,
            None,
            20,
            (),
            77,
            process_path,
            "CU-6 直径实验",
            "AfxFrame",
            None,
            PhysicalRect(0, 0, 1400, 900),
            True,
            False,
            False,
        ),
        WindowRecord(
            21,
            20,
            20,
            (20,),
            77,
            process_path,
            "",
            "MDIClient",
            None,
            PhysicalRect(50, 50, 1350, 850),
            True,
            False,
            False,
        ),
        WindowRecord(
            30,
            21,
            20,
            (20, 21),
            77,
            process_path,
            "",
            "#32770",
            1400,
            PhysicalRect(100, 100, 1296, 811),
            True,
            False,
            False,
        ),
        WindowRecord(
            31,
            30,
            20,
            (20, 21, 30),
            77,
            process_path,
            "",
            "Static",
            1501,
            PhysicalRect(106, 100, 874, 676),
            True,
            False,
            False,
        ),
    )
    snapshot = WindowSnapshot.from_records(records)
    locator = Cu5PreviewLocator(
        enumerate_snapshot=lambda: snapshot,
        selector={
            "process_name": "cu-6.exe",
            "class_name": "#32770",
            "control_id": 1400,
            "size": {"width": 1196, "height": 711},
        },
    )

    class _Frame:
        def __init__(self, width: int, height: int) -> None:
            self.width = width
            self.height = height

        def to_qimage(self) -> QImage:
            image = QImage(self.width, self.height, QImage.Format.Format_RGB32)
            image.fill(QColor("#123456"))
            return image

    class _Capture:
        def __init__(self) -> None:
            self.calls: list[tuple[int, dict[str, object]]] = []

        def capture_window(self, hwnd: int, **options):
            self.calls.append((hwnd, dict(options)))
            rect = options["rect"]
            return SimpleNamespace(frame=_Frame(rect.width, rect.height))

    native_capture = _Capture()
    backend = WindowsScreenshotBackend(
        screen_provider=lambda: (),
        window_enumerator=lambda: snapshot,
        screen_capture=native_capture,
        active_window_provider=lambda: 0,
        cu5_locator=locator,
    )

    frame = CaptureCoordinator(backend).capture_now(CaptureRequest(CaptureMode.CU5))

    assert frame.target_handle == 31
    assert frame.rect == CaptureRect(106, 100, 768, 576)
    assert frame.image.size().toTuple() == (768, 576)
    assert frame.image.pixelColor(767, 575) == QColor("#123456")
    assert frame.metadata["source_class"] == "Static"
    assert native_capture.calls == [
        (
            31,
            {
                "rect": PhysicalRect(106, 100, 874, 676),
                "validate": True,
            },
        )
    ]


def test_cu5_legacy_title_fallback_also_marks_capture_for_validation() -> None:
    class _LegacyBackend(_Backend):
        def __init__(self) -> None:
            super().__init__()
            self._windows = (
                WindowCandidate(
                    7,
                    CaptureRect(20, 30, 768, 576),
                    title="用来显示SDK摄像头的窗口",
                    executable=r"C:\CU-5\CU-5.exe",
                ),
            )
            self.captured_candidate: WindowCandidate | None = None

        def capture_window(self, candidate: WindowCandidate) -> QImage:
            self.captured_candidate = candidate
            return _image()

    backend = _LegacyBackend()
    frame = CaptureCoordinator(backend).capture_now(CaptureRequest(CaptureMode.CU5))

    assert frame.valid
    assert backend.captured_candidate is not None
    assert backend.captured_candidate.metadata["cu5_preview"] is True


def test_cu6_process_is_accepted_by_legacy_preview_fallback() -> None:
    candidates = (
        WindowCandidate(
            7,
            CaptureRect(20, 30, 768, 576),
            title="用来显示SDK摄像头的窗口",
            executable=r"C:\CU-6\CU-6.exe",
        ),
    )

    assert rank_cu5_candidates(candidates) == candidates


def test_capture_mode_parse_accepts_existing_enum() -> None:
    assert CaptureMode.parse(CaptureMode.CU5) is CaptureMode.CU5


def test_capture_request_parses_and_forwards_include_cursor_without_leaking_metadata() -> None:
    _app()

    class _CursorBackend(_Backend):
        def capture_rect(self, rect: CaptureRect, *, include_cursor: bool = False) -> QImage:
            self.capture_calls.append(("rect_cursor", (rect, include_cursor)))
            return _image()

    backend = _CursorBackend()
    coordinator = CaptureCoordinator(backend)
    request = CaptureRequest.from_mapping(
        {
            "mode": "region",
            "region": {"x": -40, "y": 5, "width": 20, "height": 10},
            "include_cursor": True,
            "caller": "test",
        }
    )

    frame = coordinator.capture_now(request)

    assert request.include_cursor is True
    assert request.metadata == {"caller": "test"}
    assert backend.capture_calls == [
        ("rect_cursor", (CaptureRect(-40, 5, 20, 10), True))
    ]
    assert frame.metadata["include_cursor"] is True


def test_cursor_option_keeps_legacy_backend_capture_signature_compatible() -> None:
    _app()
    backend = _Backend()

    frame = CaptureCoordinator(backend).capture_now(
        CaptureRequest(
            CaptureMode.REGION,
            region=CaptureRect(1, 2, 4, 5),
            include_cursor=True,
        )
    )

    assert frame.valid
    assert backend.capture_calls == [("rect", CaptureRect(1, 2, 4, 5))]


def test_cu5_resolution_failure_is_explicit_and_never_falls_back_to_drag_box() -> None:
    _app()
    coordinator = CaptureCoordinator(_Backend())
    failed: list[str] = []
    selections: list[object] = []
    coordinator.captureFailed.connect(failed.append)
    coordinator.selectionRequested.connect(lambda *args: selections.append(args))

    coordinator.start(CaptureRequest(CaptureMode.CU5))

    assert failed and "CU 系列" in failed[0]
    assert selections == []
