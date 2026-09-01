from __future__ import annotations

import os
from pathlib import Path
from threading import Event, get_ident
from time import monotonic, sleep

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QColor, QImage, QPainter
from PySide6.QtWidgets import QApplication

from fdm.geometry import Point
from fdm.models import ImageDocument
from fdm.services.digital_slide_renderer import (
    DigitalSlideDerivedCache,
    DigitalSlideRenderRequest,
    DigitalSlideRenderer,
)
from fdm.services.digital_slide_store import (
    DIGITAL_SLIDE_TILE_CODEC_JPEG,
    DIGITAL_SLIDE_TILE_CODEC_PNG,
    DigitalSlideManifest,
    DigitalSlideStore,
    DigitalSlideTile,
)
from fdm.settings import AppSettings
from fdm.ui.digital_slide_canvas import DigitalSlideBrowseView, DigitalSlideCanvas
from fdm.ui.view_transform import CanvasZoomMode


def _tile_image(width: int, height: int, color: str) -> QImage:
    image = QImage(width, height, QImage.Format.Format_RGB32)
    image.fill(QColor(color))
    return image


def _create_coordinate_slide(path: Path) -> DigitalSlideStore:
    manifest = DigitalSlideManifest(
        version=1,
        width=400,
        height=320,
        viewport_width=100,
        viewport_height=80,
        focus_levels=[-1, 0],
    )
    store = DigitalSlideStore.create(path, manifest)
    palette = (
        "#e63946",
        "#f4a261",
        "#2a9d8f",
        "#457b9d",
        "#8338ec",
        "#ff006e",
        "#3a86ff",
        "#8ac926",
    )
    for focus_index in range(2):
        for row in range(4):
            for column in range(4):
                color = palette[(row * 4 + column + focus_index) % len(palette)]
                codec = (
                    DIGITAL_SLIDE_TILE_CODEC_PNG
                    if (row + column) % 2 == 0
                    else DIGITAL_SLIDE_TILE_CODEC_JPEG
                )
                store.write_tile(
                    DigitalSlideTile(
                        z_index=focus_index,
                        x=column * 100,
                        y=row * 80,
                        width=100,
                        height=80,
                    ),
                    _tile_image(100, 80, color),
                    codec=codec,
                    quality=90,
                    update_manifest=False,
                )
    store.write_manifest(manifest)
    return store


def _wait_for(app: QApplication, predicate, *, timeout: float = 4.0) -> None:
    deadline = monotonic() + timeout
    while monotonic() < deadline:
        app.processEvents()
        if predicate():
            return
    app.processEvents()
    assert predicate(), "timed out waiting for digital-slide worker"


class _WheelEvent:
    def __init__(self, delta: int, position: QPointF) -> None:
        self._delta = delta
        self._position = position
        self.accepted = False

    def angleDelta(self) -> QPoint:
        return QPoint(0, self._delta)

    def position(self) -> QPointF:
        return QPointF(self._position)

    @staticmethod
    def modifiers() -> Qt.KeyboardModifier:
        return Qt.KeyboardModifier.ControlModifier

    def accept(self) -> None:
        self.accepted = True


def test_renderer_uses_real_sqlite_tiles_lod_and_bounded_shared_cache(
    tmp_path: Path,
) -> None:
    store = _create_coordinate_slide(tmp_path / "renderer.fdmslide")
    manifest = store.read_manifest()
    store.close()
    results = []
    failures = []
    ready = Event()
    renderer = DigitalSlideRenderer(
        tmp_path / "renderer.fdmslide",
        manifest,
        cache_root=tmp_path / "derived-cache",
        disk_cache_bytes=64 * 1024 * 1024,
        memory_cache_bytes=1024 * 1024,
        result_callback=lambda frame: (results.append(frame), ready.set()),
        failure_callback=lambda failure: (failures.append(failure), ready.set()),
    )
    request = DigitalSlideRenderRequest(
        request_id=1,
        purpose="display",
        source_rect=(0.0, 0.0, 400.0, 320.0),
        output_size_px=(200, 160),
        focus_index=0,
        device_pixel_ratio=1.0,
    )
    try:
        renderer.submit(request)
        assert ready.wait(4.0)
        assert not failures
        first = results[-1]
        assert first.lod == 1
        assert first.decoded_tiles == 16
        assert first.image.size().toTuple() == (200, 160)
        assert QColor(first.image.pixel(25, 20)).red() > 150
        assert renderer.stats().memory_bytes <= 1024 * 1024

        ready.clear()
        renderer.submit(
            DigitalSlideRenderRequest(
                request_id=2,
                purpose="display",
                source_rect=request.source_rect,
                output_size_px=request.output_size_px,
                focus_index=0,
                device_pixel_ratio=1.0,
            )
        )
        assert ready.wait(4.0)
        second = results[-1]
        assert second.cache_hits > 0
        assert renderer.stats().pending_requests == 0
        ready.clear()
        renderer.submit(
            DigitalSlideRenderRequest(
                request_id=3,
                purpose="display",
                source_rect=request.source_rect,
                output_size_px=(80, 64),
                focus_index=0,
                device_pixel_ratio=1.0,
            )
        )
        assert ready.wait(4.0)
        farther = results[-1]
        assert farther.lod == 2
        assert farther.decoded_tiles == 0
        cache_deadline = monotonic() + 4.0
        while (
            len(list((tmp_path / "derived-cache").rglob("*lod-1.png"))) < 16
            and monotonic() < cache_deadline
        ):
            sleep(0.005)
        assert len(list((tmp_path / "derived-cache").rglob("*lod-1.png"))) == 16
    finally:
        renderer.close()

    disk_results = []
    disk_ready = Event()
    second_renderer = DigitalSlideRenderer(
        tmp_path / "renderer.fdmslide",
        manifest,
        cache_root=tmp_path / "derived-cache",
        disk_cache_bytes=64 * 1024 * 1024,
        result_callback=lambda frame: (disk_results.append(frame), disk_ready.set()),
        failure_callback=lambda _failure: disk_ready.set(),
    )
    try:
        second_renderer.submit(
            DigitalSlideRenderRequest(
                request_id=4,
                purpose="display",
                source_rect=request.source_rect,
                output_size_px=request.output_size_px,
                focus_index=0,
                device_pixel_ratio=1.0,
            )
        )
        assert disk_ready.wait(4.0)
        assert disk_results[-1].decoded_tiles == 0
        assert second_renderer.stats().disk_hits == 16
    finally:
        second_renderer.close()


def test_derived_cache_fingerprint_invalidates_and_shared_budget_is_strict(
    tmp_path: Path,
) -> None:
    source = tmp_path / "fingerprint.fdmslide"
    source.write_bytes(b"first-source-payload")
    manifest = DigitalSlideManifest(1, 64, 64, 32, 32, [0])
    first_fingerprint = DigitalSlideDerivedCache.source_fingerprint(source, manifest)
    source_stat = source.stat()
    os.utime(
        source,
        ns=(source_stat.st_atime_ns, source_stat.st_mtime_ns + 1_000_000),
    )
    second_fingerprint = DigitalSlideDerivedCache.source_fingerprint(source, manifest)
    assert second_fingerprint != first_fingerprint
    localized = tmp_path / "localized-copy.fdmslide"
    localized.write_bytes(source.read_bytes())
    localized_fingerprint = DigitalSlideDerivedCache.source_fingerprint(
        localized,
        manifest,
        source_identity=source,
    )
    assert localized_fingerprint == second_fingerprint

    images: list[QImage] = []
    encoded_sizes: list[int] = []
    for seed in range(3):
        image = QImage(96, 96, QImage.Format.Format_RGB32)
        for y in range(image.height()):
            for x in range(image.width()):
                image.setPixelColor(
                    x,
                    y,
                    QColor(
                        (x * 17 + seed * 41) % 256,
                        (y * 29 + seed * 67) % 256,
                        ((x + y) * 11 + seed * 23) % 256,
                    ),
                )
        probe = tmp_path / f"probe-{seed}.png"
        assert image.save(str(probe), "PNG")
        encoded_sizes.append(probe.stat().st_size)
        images.append(image)
    byte_limit = max(
        encoded_sizes[0] + encoded_sizes[1],
        encoded_sizes[0] + encoded_sizes[2],
        encoded_sizes[1] + encoded_sizes[2],
    )
    cache_root = tmp_path / "shared-cache"
    first_cache = DigitalSlideDerivedCache(cache_root, byte_limit=byte_limit)
    second_cache = DigitalSlideDerivedCache(cache_root, byte_limit=byte_limit)
    fingerprint = "a" * 64
    first_cache.store(
        fingerprint,
        images[0],
        focus_index=0,
        tile_id=1,
        lod=1,
    )
    second_cache.store(
        fingerprint,
        images[1],
        focus_index=0,
        tile_id=2,
        lod=1,
    )
    first_cache.store(
        fingerprint,
        images[2],
        focus_index=0,
        tile_id=3,
        lod=1,
    )
    persisted_bytes = sum(
        path.stat().st_size for path in cache_root.rglob("*.png")
    )
    assert persisted_bytes <= byte_limit


def test_renderer_preserves_overlap_blending_and_missing_tile_background(
    tmp_path: Path,
) -> None:
    slide_path = tmp_path / "overlap-missing.fdmslide"
    manifest = DigitalSlideManifest(1, 200, 80, 100, 80, [0])
    store = DigitalSlideStore.create(slide_path, manifest)
    store.write_tile(
        DigitalSlideTile(0, 0, 0, 100, 80),
        _tile_image(100, 80, "#ff0000"),
        update_manifest=False,
    )
    store.write_tile(
        DigitalSlideTile(0, 50, 0, 100, 80),
        _tile_image(100, 80, "#0000ff"),
        update_manifest=False,
    )
    store.write_manifest(manifest)
    store.close()
    frames = []
    failures = []
    ready = Event()
    renderer = DigitalSlideRenderer(
        slide_path,
        manifest,
        cache_root=tmp_path / "overlap-cache",
        disk_cache_bytes=0,
        result_callback=lambda frame: (frames.append(frame), ready.set()),
        failure_callback=lambda failure: (failures.append(failure), ready.set()),
    )
    try:
        renderer.submit(
            DigitalSlideRenderRequest(
                request_id=1,
                purpose="display",
                source_rect=(0.0, 0.0, 200.0, 80.0),
                output_size_px=(200, 80),
                focus_index=0,
                device_pixel_ratio=1.0,
                blend_width=10,
                force_lod=0,
            )
        )
        assert ready.wait(4.0)
        assert not failures
        image = frames[-1].image
        blended = image.pixelColor(55, 40)
        assert 100 <= blended.red() <= 155
        assert blended.green() == 0
        assert 100 <= blended.blue() <= 155
        assert image.pixelColor(75, 40).blue() > 240
        assert image.pixelColor(175, 40) == QColor("#101820")
    finally:
        renderer.close()


def test_browse_camera_expands_visible_field_preserves_anchor_and_gates_pixels(
    tmp_path: Path,
) -> None:
    app = QApplication.instance() or QApplication([])
    store = _create_coordinate_slide(tmp_path / "camera.fdmslide")
    document = ImageDocument(
        id="camera",
        path=str(tmp_path / "camera.fdmslide"),
        image_size=(400, 320),
        document_kind="digital_slide",
    )
    canvas = DigitalSlideCanvas()
    canvas.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
    canvas.resize(440, 360)
    settings = AppSettings(digital_slide_render_cache_gib=0)
    canvas.set_settings(settings)
    main_thread = get_ident()
    gui_sqlite_calls: list[str] = []
    original_descriptors = DigitalSlideStore.list_tile_descriptors
    original_read_tile = DigitalSlideStore.read_tile_image

    def descriptors_on_worker(self, *, z_index: int):
        if get_ident() == main_thread:
            gui_sqlite_calls.append("descriptors")
        return original_descriptors(self, z_index=z_index)

    def tile_on_worker(self, tile_id: int):
        if get_ident() == main_thread:
            gui_sqlite_calls.append("tile")
        return original_read_tile(self, tile_id)

    DigitalSlideStore.list_tile_descriptors = descriptors_on_worker
    DigitalSlideStore.read_tile_image = tile_on_worker
    try:
        canvas.set_slide_document(document, store)
        canvas.show()
        canvas.fit_native_viewport()
        _wait_for(app, canvas.pixel_work_enabled)
        native_visible = canvas.visible_slide_rect()
        assert abs(native_visible.width() - 100.0) < 1.0e-6
        assert abs(native_visible.height() - 80.0) < 1.0e-6
        assert canvas.zoom_mode() is CanvasZoomMode.NATIVE_FIELD_FIT

        canvas.center_on_image_point(Point(200.0, 160.0))
        _wait_for(app, canvas.pixel_work_enabled)
        notices: list[str] = []
        canvas.browseNoticeRequested.connect(notices.append)
        canvas._drawing_anchor_raw = Point(190.0, 150.0)  # noqa: SLF001
        before_blocked_zoom = canvas.view_zoom()
        assert not canvas.set_browse_view(
            DigitalSlideBrowseView(
                center_px=canvas.browse_view().center_px,
                zoom=before_blocked_zoom * 0.5,
                mode=CanvasZoomMode.CUSTOM,
            )
        )
        assert canvas.view_zoom() == before_blocked_zoom
        assert notices
        canvas._drawing_anchor_raw = None  # noqa: SLF001
        cursor = QPointF(170.0, 145.0)
        anchor_before = canvas.widget_to_image(cursor)
        zoom_out = _WheelEvent(-120, cursor)
        canvas.wheelEvent(zoom_out)
        anchor_after = canvas.widget_to_image(cursor)
        assert zoom_out.accepted
        assert abs(anchor_after.x - anchor_before.x) <= 1.0
        assert abs(anchor_after.y - anchor_before.y) <= 1.0
        assert canvas.visible_slide_rect().width() > native_visible.width()
        assert not canvas.pixel_work_enabled()

        canvas.fit_to_view()
        whole = canvas.visible_slide_rect()
        assert whole == canvas._paint_image_bounds()  # noqa: SLF001
        assert canvas.zoom_mode() is CanvasZoomMode.FIT
        assert not canvas.pixel_work_enabled()

        canvas.fit_native_viewport()
        _wait_for(app, canvas.pixel_work_enabled)
        native = canvas.native_viewport_rect()
        assert native.size().toTuple() == (100.0, 80.0)
        assert canvas._image is not None  # noqa: SLF001
        assert canvas._image.size().toTuple() == (100, 80)  # noqa: SLF001
        assert document.metadata["digital_slide"]["browse_view"]["version"] == 1

        before = canvas.browse_view().center_px
        step_x, step_y = canvas._navigation_step()  # noqa: SLF001
        canvas.move_viewport_by(step_x, step_y)
        after = canvas.browse_view().center_px
        assert abs((after.x - before.x) - native_visible.width() * 0.25) < 1.0e-6
        assert abs((after.y - before.y) - native_visible.height() * 0.25) < 1.0e-6
        _wait_for(app, canvas.pixel_work_enabled)
        assert gui_sqlite_calls == []
    finally:
        DigitalSlideStore.list_tile_descriptors = original_descriptors
        DigitalSlideStore.read_tile_image = original_read_tile
        renderer = canvas._renderer  # noqa: SLF001 - lifecycle assertion
        canvas.shutdown()
        if renderer is not None:
            assert not renderer.is_alive()
        canvas.clear_document()
        canvas.close()
        store.close()


def test_focus_change_keeps_a_painted_handoff_and_indicator_is_zoom_only(
    tmp_path: Path,
) -> None:
    app = QApplication.instance() or QApplication([])
    store = _create_coordinate_slide(tmp_path / "focus-handoff.fdmslide")
    document = ImageDocument(
        id="focus-handoff",
        path=str(store.path),
        image_size=(400, 320),
        document_kind="digital_slide",
    )
    canvas = DigitalSlideCanvas()
    canvas.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
    canvas.resize(440, 360)
    canvas.set_settings(AppSettings(digital_slide_render_cache_gib=0))
    try:
        canvas.set_slide_document(document, store)
        canvas.show()
        canvas.fit_native_viewport()
        initial_focus = canvas.focus_index()
        target_focus = 0 if initial_focus != 0 else 1
        _wait_for(
            app,
            lambda: (
                canvas.pixel_work_enabled()
                and canvas._render_frame is not None  # noqa: SLF001
                and canvas._render_frame.focus_index == initial_focus  # noqa: SLF001
            ),
        )
        first_frame = canvas._render_frame  # noqa: SLF001
        assert first_frame is not None
        assert not canvas.pixel_work_controls_blocked()

        canvas._hide_native_viewport_indicator()  # noqa: SLF001
        canvas.set_focus_index(target_focus)
        assert canvas._render_frame is None  # noqa: SLF001
        assert canvas._focus_transition_frame is first_frame  # noqa: SLF001
        assert not canvas.native_viewport_indicator_visible()
        assert not canvas.pixel_work_enabled()
        assert not canvas.pixel_work_controls_blocked()

        painted = QImage(canvas.size(), QImage.Format.Format_RGB32)
        painted.fill(QColor("#000000"))
        painter = QPainter(painted)
        target = canvas._draw_base_image(painter)  # noqa: SLF001
        painter.end()
        assert not target.isEmpty()
        center = canvas._content_rect().center().toPoint()  # noqa: SLF001
        center_color = painted.pixelColor(center)
        assert center_color != QColor("#000000")
        assert center_color != QColor("#101820")

        _wait_for(
            app,
            lambda: (
                canvas.pixel_work_enabled()
                and canvas._render_frame is not None  # noqa: SLF001
                and canvas._render_frame.focus_index == target_focus  # noqa: SLF001
            ),
        )
        assert canvas._focus_transition_frame is None  # noqa: SLF001

        canvas._native_viewport_indicator_timer.setInterval(10)  # noqa: SLF001
        canvas.set_view_zoom(canvas.view_zoom() * 1.1)
        assert canvas.native_viewport_indicator_visible()
        _wait_for(app, lambda: not canvas.native_viewport_indicator_visible())

        canvas.move_viewport_by(5.0, 0.0)
        assert not canvas.native_viewport_indicator_visible()
        canvas.set_focus_index(initial_focus)
        assert not canvas.native_viewport_indicator_visible()
    finally:
        canvas.shutdown()
        canvas.clear_document()
        canvas.close()
        store.close()


@pytest.mark.parametrize("device_pixel_ratio", [1.0, 1.25, 1.5, 2.0])
def test_digital_slide_display_frame_respects_dpr_without_coordinate_drift(
    tmp_path: Path,
    device_pixel_ratio: float,
) -> None:
    app = QApplication.instance() or QApplication([])

    class DprCanvas(DigitalSlideCanvas):
        def devicePixelRatioF(self) -> float:  # noqa: N802 - Qt virtual name
            return device_pixel_ratio

    store = _create_coordinate_slide(
        tmp_path / f"dpr-{str(device_pixel_ratio).replace('.', '-')}.fdmslide"
    )
    document = ImageDocument(
        id=f"dpr-{device_pixel_ratio}",
        path=str(store.path),
        image_size=(400, 320),
        document_kind="digital_slide",
    )
    canvas = DprCanvas()
    canvas.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
    canvas.resize(440, 360)
    canvas.set_settings(AppSettings(digital_slide_render_cache_gib=0))
    try:
        canvas.set_slide_document(document, store)
        canvas.show()
        canvas.fit_native_viewport()
        request_id = canvas._latest_display_request_id  # noqa: SLF001
        _wait_for(
            app,
            lambda: (
                canvas._render_frame is not None  # noqa: SLF001
                and canvas._render_frame.request_id == request_id  # noqa: SLF001
            ),
        )
        frame = canvas._render_frame  # noqa: SLF001
        assert frame is not None
        content = canvas._content_rect()  # noqa: SLF001
        assert frame.output_size_px == (
            int(round(content.width() * device_pixel_ratio)),
            int(round(content.height() * device_pixel_ratio)),
        )
        assert frame.device_pixel_ratio == device_pixel_ratio
        visible = canvas.visible_slide_rect()
        mapped_top_left = canvas.image_to_widget(
            Point(visible.left(), visible.top())
        )
        assert abs(mapped_top_left.x() - content.left()) <= 1.0e-6
        assert abs(mapped_top_left.y() - content.top()) <= 1.0e-6
        roundtrip = canvas.widget_to_image(canvas.image_to_widget(Point(173.0, 129.0)))
        assert abs(roundtrip.x - 173.0) <= 1.0e-6
        assert abs(roundtrip.y - 129.0) <= 1.0e-6
    finally:
        canvas.shutdown()
        canvas.clear_document()
        canvas.close()
        store.close()


def test_old_project_origin_migrates_to_native_field_camera(tmp_path: Path) -> None:
    app = QApplication.instance() or QApplication([])
    store = _create_coordinate_slide(tmp_path / "migration.fdmslide")
    document = ImageDocument(
        id="migration",
        path=str(tmp_path / "migration.fdmslide"),
        image_size=(400, 320),
        document_kind="digital_slide",
        metadata={
            "digital_slide": {
                "viewport_origin": [200, 160],
                "focus_index": 1,
            }
        },
    )
    document.view_state.zoom = 7.5
    document.view_state.pan = Point(333.0, 444.0)
    canvas = DigitalSlideCanvas()
    canvas.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
    canvas.resize(440, 360)
    try:
        canvas.set_slide_document(document, store)
        canvas.show()
        canvas.fit_native_viewport()
        _wait_for(app, canvas.pixel_work_enabled)
        assert canvas.zoom_mode() is CanvasZoomMode.NATIVE_FIELD_FIT
        assert canvas.viewport_origin() == Point(200.0, 160.0)
        assert canvas.browse_view().center_px == Point(250.0, 200.0)
        assert canvas.view_zoom() != 7.5
    finally:
        canvas.shutdown()
        canvas.clear_document()
        canvas.close()
        store.close()
