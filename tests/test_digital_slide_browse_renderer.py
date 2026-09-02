from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
from threading import Event, get_ident
from time import monotonic, sleep
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QColor, QImage, QPainter
from PySide6.QtWidgets import QApplication

from fdm.geometry import Point
from fdm.models import ImageDocument
from fdm.services.digital_slide_renderer import (
    DigitalSlideDerivedCache,
    DigitalSlideRenderFailure,
    DigitalSlideRenderFrame,
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


class _MouseEvent:
    def __init__(
        self,
        position: QPointF,
        *,
        button: Qt.MouseButton = Qt.MouseButton.LeftButton,
    ) -> None:
        self._position = QPointF(position)
        self._button = button
        self.accepted = False

    def position(self) -> QPointF:
        return QPointF(self._position)

    def button(self) -> Qt.MouseButton:
        return self._button

    @staticmethod
    def modifiers() -> Qt.KeyboardModifier:
        return Qt.KeyboardModifier.NoModifier

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


def test_renderer_completes_atomic_focus_preview_before_native_exact(
    tmp_path: Path,
) -> None:
    store = _create_coordinate_slide(tmp_path / "focus-preview-order.fdmslide")
    manifest = store.read_manifest()
    store.close()
    results: list[DigitalSlideRenderFrame] = []
    failures: list[DigitalSlideRenderFailure] = []
    renderer = DigitalSlideRenderer(
        tmp_path / "focus-preview-order.fdmslide",
        manifest,
        cache_root=tmp_path / "derived-cache",
        disk_cache_bytes=0,
        result_callback=results.append,
        failure_callback=failures.append,
    )
    source_rect = (0.0, 0.0, 100.0, 80.0)
    try:
        renderer.submit(
            DigitalSlideRenderRequest(
                request_id=1,
                purpose="focus_preview",
                source_rect=source_rect,
                output_size_px=(50, 40),
                focus_index=1,
                device_pixel_ratio=1.0,
                generation=1,
                quality="coarse",
                priority=0,
            )
        )
        renderer.submit(
            DigitalSlideRenderRequest(
                request_id=2,
                purpose="native",
                source_rect=source_rect,
                output_size_px=(100, 80),
                focus_index=1,
                device_pixel_ratio=1.0,
                force_lod=0,
                generation=1,
                quality="final",
                priority=2,
            )
        )
        deadline = monotonic() + 4.0
        while len(results) < 2 and not failures and monotonic() < deadline:
            sleep(0.005)

        assert failures == []
        assert [frame.purpose for frame in results] == [
            "focus_preview",
            "native",
        ]
        preview = results[0]
        assert preview.complete
        assert preview.quality == "coarse"
        assert not preview.pixel_exact
        assert preview.lod == 1
        assert results[1].pixel_exact
        assert renderer.stats().pending_requests == 0
    finally:
        renderer.close()


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

    clear_root = tmp_path / "clear-current-slide-cache"
    clear_cache = DigitalSlideDerivedCache(
        clear_root,
        byte_limit=16 * 1024 * 1024,
    )
    clear_cache.store(
        fingerprint,
        images[0],
        focus_index=0,
        tile_id=1,
        lod=1,
    )
    clear_cache.store_preview(
        fingerprint,
        images[1],
        focus_index=0,
        maximum_edge=1024,
    )
    assert len(list(clear_root.rglob("*.png"))) == 2
    clear_cache.clear_fingerprint(fingerprint)
    assert not list(clear_root.rglob("*.png"))


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
    original_regional_descriptors = DigitalSlideStore.list_tile_descriptors_in_rect
    original_read_tile = DigitalSlideStore.read_tile_image
    original_read_scaled_tile = DigitalSlideStore.read_tile_image_scaled

    def descriptors_on_worker(self, *, z_index: int):
        if get_ident() == main_thread:
            gui_sqlite_calls.append("descriptors")
        return original_descriptors(self, z_index=z_index)

    def tile_on_worker(self, tile_id: int):
        if get_ident() == main_thread:
            gui_sqlite_calls.append("tile")
        return original_read_tile(self, tile_id)

    def regional_descriptors_on_worker(
        self,
        *,
        z_index: int,
        x: float,
        y: float,
        width: float,
        height: float,
    ):
        if get_ident() == main_thread:
            gui_sqlite_calls.append("regional_descriptors")
        return original_regional_descriptors(
            self,
            z_index=z_index,
            x=x,
            y=y,
            width=width,
            height=height,
        )

    def scaled_tile_on_worker(
        self,
        tile_id: int,
        *,
        width: int,
        height: int,
    ):
        if get_ident() == main_thread:
            gui_sqlite_calls.append("scaled_tile")
        return original_read_scaled_tile(
            self,
            tile_id,
            width=width,
            height=height,
        )

    DigitalSlideStore.list_tile_descriptors = descriptors_on_worker
    DigitalSlideStore.list_tile_descriptors_in_rect = regional_descriptors_on_worker
    DigitalSlideStore.read_tile_image = tile_on_worker
    DigitalSlideStore.read_tile_image_scaled = scaled_tile_on_worker
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
        DigitalSlideStore.list_tile_descriptors_in_rect = original_regional_descriptors
        DigitalSlideStore.read_tile_image = original_read_tile
        DigitalSlideStore.read_tile_image_scaled = original_read_scaled_tile
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

        synchronous_focus_colors: list[QColor] = []

        def capture_synchronous_focus_paint(_focus_index: int) -> None:
            immediate = QImage(canvas.size(), QImage.Format.Format_RGB32)
            immediate.fill(QColor("#000000"))
            immediate_painter = QPainter(immediate)
            canvas._draw_base_image(immediate_painter)  # noqa: SLF001
            immediate_painter.end()
            synchronous_focus_colors.append(
                immediate.pixelColor(
                    canvas._content_rect().center().toPoint()  # noqa: SLF001
                )
            )

        canvas.focusChanged.connect(capture_synchronous_focus_paint)
        canvas._hide_native_viewport_indicator()  # noqa: SLF001
        canvas.set_focus_index(target_focus)
        assert synchronous_focus_colors
        assert synchronous_focus_colors[-1] != QColor("#E6E6E6")
        assert canvas._render_frame is None  # noqa: SLF001
        assert canvas._previous_render_frame is first_frame  # noqa: SLF001
        # The previous focus may remain as a paint-only handoff until the first
        # real target-focus preview arrives.  It must never become the current
        # measurement or pixel-algorithm frame.
        assert canvas._focus_transition_frame is first_frame  # noqa: SLF001
        assert not canvas.native_viewport_indicator_visible()
        assert not canvas.pixel_work_enabled()
        assert not canvas.pixel_work_controls_blocked()

        source = canvas._source_view_rect()  # noqa: SLF001
        placeholder = _tile_image(100, 80, "#E6E6E6")
        placeholder_frame = DigitalSlideRenderFrame(
            request_id=canvas._latest_preview_request_id,  # noqa: SLF001
            purpose="preview",
            source_rect=(0.0, 0.0, 400.0, 320.0),
            output_size_px=(100, 80),
            focus_index=target_focus,
            device_pixel_ratio=1.0,
            lod=2,
            image=placeholder,
            elapsed_ms=1.0,
            decoded_tiles=0,
            cache_hits=0,
            generation=canvas._view_generation,  # noqa: SLF001
            quality="placeholder",
            pixel_exact=False,
        )
        canvas._on_render_frame_ready(placeholder_frame)  # noqa: SLF001
        canvas._on_render_frame_ready(  # noqa: SLF001
            replace(placeholder_frame, quality="coarse", cache_hits=1)
        )
        assert canvas._focus_transition_frame is first_frame  # noqa: SLF001

        # A progressive target-focus frame may contain light unresolved areas.
        # Only its real coverage is revealed over the handoff.
        canvas._on_render_frame_ready(  # noqa: SLF001
            DigitalSlideRenderFrame(
                request_id=canvas._latest_coarse_request_id,  # noqa: SLF001
                purpose="coarse",
                source_rect=(
                    source.x(),
                    source.y(),
                    source.width(),
                    source.height(),
                ),
                output_size_px=(100, 80),
                focus_index=target_focus,
                device_pixel_ratio=1.0,
                lod=1,
                image=placeholder,
                elapsed_ms=2.0,
                decoded_tiles=1,
                cache_hits=0,
                generation=canvas._view_generation,  # noqa: SLF001
                quality="coarse",
                pixel_exact=False,
                coverage_rects=(
                    (
                        source.x(),
                        source.y(),
                        source.width() * 0.2,
                        source.height(),
                    ),
                ),
                complete=False,
            )
        )

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
        assert center_color != QColor("#E6E6E6")
        covered_widget = canvas.image_to_widget(
            Point(
                source.x() + source.width() * 0.1,
                source.y() + source.height() * 0.5,
            )
        ).toPoint()
        assert painted.pixelColor(covered_widget) != QColor("#E6E6E6")

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
        canvas.fit_to_view()
        _wait_for(
            app,
            lambda: (
                canvas._render_frame is not None  # noqa: SLF001
                and canvas._render_frame.purpose == "display"  # noqa: SLF001
                and canvas._render_frame.generation  # noqa: SLF001
                == canvas._view_generation  # noqa: SLF001
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


def test_vector_measurement_drafts_survive_cross_viewport_navigation(
    tmp_path: Path,
) -> None:
    app = QApplication.instance() or QApplication([])
    store = _create_coordinate_slide(tmp_path / "cross-field-measurement.fdmslide")
    document = ImageDocument(
        id="cross-field-measurement",
        path=str(store.path),
        image_size=(400, 320),
        document_kind="digital_slide",
    )
    canvas = DigitalSlideCanvas()
    canvas.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
    canvas.resize(440, 360)
    canvas.set_settings(AppSettings(digital_slide_render_cache_gib=0))
    committed: list[tuple[str, object]] = []
    canvas.lineCommitted.connect(
        lambda _document_id, mode, payload: committed.append((mode, payload))
    )
    try:
        canvas.set_slide_document(document, store)
        canvas.show()
        canvas.fit_native_viewport()
        _wait_for(app, canvas.pixel_work_enabled)

        canvas.set_tool_mode("manual")
        start = Point(75.0, 40.0)
        canvas.mousePressEvent(_MouseEvent(canvas.image_to_widget(start)))  # type: ignore[arg-type]
        assert canvas._drawing_anchor_raw == start  # noqa: SLF001
        assert canvas._drawing_line is not None  # noqa: SLF001

        target_focus = 0 if canvas.focus_index() != 0 else 1
        canvas.set_focus_index(target_focus)
        assert not canvas.pixel_work_enabled()
        assert canvas._drawing_anchor_raw == start  # noqa: SLF001
        assert canvas._drawing_line is not None  # noqa: SLF001

        canvas.move_viewport_by(100.0, 0.0)
        assert not canvas.pixel_work_enabled()
        assert not canvas._read_only  # noqa: SLF001
        assert canvas._drawing_anchor_raw == start  # noqa: SLF001
        assert canvas._drawing_line is not None  # noqa: SLF001
        end = Point(175.0, 40.0)
        _wait_for(app, lambda: canvas.vector_measurement_available(end))
        canvas.mouseMoveEvent(_MouseEvent(canvas.image_to_widget(end)))  # type: ignore[arg-type]
        canvas.mouseReleaseEvent(_MouseEvent(canvas.image_to_widget(end)))  # type: ignore[arg-type]
        assert committed[-1][0] == "manual"
        manual_line = committed[-1][1]
        assert manual_line.start == start
        assert manual_line.end == end

        canvas.set_tool_mode("snap")
        snap_start = Point(125.0, 30.0)
        canvas.mousePressEvent(_MouseEvent(canvas.image_to_widget(snap_start)))  # type: ignore[arg-type]
        canvas.mouseReleaseEvent(_MouseEvent(canvas.image_to_widget(snap_start)))  # type: ignore[arg-type]
        assert canvas._drawing_anchor_raw == snap_start  # noqa: SLF001
        canvas.move_viewport_by(100.0, 0.0)
        snap_end = Point(275.0, 30.0)
        _wait_for(app, lambda: canvas.vector_measurement_available(snap_end))
        canvas.mousePressEvent(_MouseEvent(canvas.image_to_widget(snap_end)))  # type: ignore[arg-type]
        assert committed[-1][0] == "snap"
        snap_line = committed[-1][1]
        assert snap_line.start == snap_start
        assert snap_line.end == snap_end

        canvas.set_tool_mode("continuous_manual")
        first = Point(225.0, 55.0)
        canvas.mousePressEvent(_MouseEvent(canvas.image_to_widget(first)))  # type: ignore[arg-type]
        canvas.mouseReleaseEvent(_MouseEvent(canvas.image_to_widget(first)))  # type: ignore[arg-type]
        assert canvas._drawing_polygon_points == [first]  # noqa: SLF001

        # Right-button panning is a camera operation only; it must not own or
        # cancel the staged polyline.
        pan_from = canvas.image_to_widget(Point(250.0, 40.0))
        pan_to = QPointF(pan_from.x() - 24.0, pan_from.y())
        canvas.mousePressEvent(
            _MouseEvent(pan_from, button=Qt.MouseButton.RightButton)  # type: ignore[arg-type]
        )
        canvas.mouseMoveEvent(
            _MouseEvent(pan_to, button=Qt.MouseButton.RightButton)  # type: ignore[arg-type]
        )
        canvas.mouseReleaseEvent(
            _MouseEvent(pan_to, button=Qt.MouseButton.RightButton)  # type: ignore[arg-type]
        )
        assert canvas._drawing_polygon_points == [first]  # noqa: SLF001

        canvas.move_viewport_by(76.0, 0.0)
        second = Point(325.0, 55.0)
        _wait_for(app, lambda: canvas.vector_measurement_available(second))
        canvas.mousePressEvent(_MouseEvent(canvas.image_to_widget(second)))  # type: ignore[arg-type]
        canvas.mouseReleaseEvent(_MouseEvent(canvas.image_to_widget(second)))  # type: ignore[arg-type]
        assert canvas._drawing_polygon_points == [first, second]  # noqa: SLF001
        assert canvas.commit_pending_path()
        assert committed[-1][0] == "continuous_manual"
        assert not canvas._drawing_polygon_points  # noqa: SLF001
    finally:
        canvas.shutdown()
        canvas.clear_document()
        canvas.close()
        store.close()


def test_low_resolution_vector_input_requires_current_real_coverage(
    tmp_path: Path,
) -> None:
    app = QApplication.instance() or QApplication([])
    store = _create_coordinate_slide(tmp_path / "coverage-gate.fdmslide")
    document = ImageDocument(
        id="coverage-gate",
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
        _wait_for(app, canvas.pixel_work_enabled)
        canvas._final_render_timer.stop()  # noqa: SLF001
        canvas._stop_renderer()  # noqa: SLF001
        canvas._native_frame_key = None  # noqa: SLF001
        canvas._advance_view_generation()  # noqa: SLF001
        canvas._update_pixel_work_state()  # noqa: SLF001

        native = canvas.native_viewport_rect()
        covered = (
            native.x(),
            native.y(),
            native.width() / 2.0,
            native.height(),
        )
        coarse_image = _tile_image(200, 160, "#f8f8f8")
        canvas._latest_coarse_request_id = 900  # noqa: SLF001
        canvas._on_render_frame_ready(  # noqa: SLF001
            DigitalSlideRenderFrame(
                request_id=900,
                purpose="coarse",
                source_rect=(
                    native.x(),
                    native.y(),
                    native.width(),
                    native.height(),
                ),
                output_size_px=(200, 160),
                focus_index=canvas.focus_index(),
                device_pixel_ratio=1.0,
                lod=1,
                image=coarse_image,
                elapsed_ms=2.0,
                decoded_tiles=1,
                cache_hits=0,
                generation=canvas._view_generation,  # noqa: SLF001
                quality="coarse",
                pixel_exact=False,
                coverage_rects=(covered,),
            )
        )
        inside = Point(native.x() + native.width() * 0.25, native.center().y())
        outside = Point(native.x() + native.width() * 0.75, native.center().y())
        assert not canvas.pixel_work_enabled()
        assert canvas.vector_measurement_available(inside)
        assert not canvas.vector_measurement_available(outside)
        assert canvas.presentation_state().quality == "coarse"

        canvas.set_tool_mode("manual")
        canvas.mousePressEvent(_MouseEvent(canvas.image_to_widget(outside)))  # type: ignore[arg-type]
        assert canvas._drawing_anchor_raw is None  # noqa: SLF001
        canvas.mousePressEvent(_MouseEvent(canvas.image_to_widget(inside)))  # type: ignore[arg-type]
        assert canvas._drawing_anchor_raw == inside  # noqa: SLF001
        line_before = canvas._drawing_line  # noqa: SLF001
        canvas.mouseMoveEvent(_MouseEvent(canvas.image_to_widget(outside)))  # type: ignore[arg-type]
        assert canvas._drawing_line is line_before  # noqa: SLF001

        canvas._advance_view_generation()  # noqa: SLF001
        assert not canvas.vector_measurement_available(inside)
        assert canvas._drawing_anchor_raw == inside  # noqa: SLF001
        canvas._latest_coarse_request_id = 901  # noqa: SLF001
        canvas._on_render_frame_ready(  # noqa: SLF001
            DigitalSlideRenderFrame(
                request_id=901,
                purpose="coarse",
                source_rect=(native.x(), native.y(), native.width(), native.height()),
                output_size_px=(200, 160),
                focus_index=canvas.focus_index(),
                device_pixel_ratio=1.0,
                lod=1,
                image=coarse_image,
                elapsed_ms=1.0,
                decoded_tiles=0,
                cache_hits=0,
                generation=canvas._view_generation,  # noqa: SLF001
                quality="placeholder",
                pixel_exact=False,
                coverage_rects=(),
            )
        )
        assert not canvas.vector_measurement_available(inside)
        assert canvas._drawing_anchor_raw == inside  # noqa: SLF001
        canvas._latest_native_request_id = 902  # noqa: SLF001
        canvas._on_render_frame_failed(  # noqa: SLF001
            DigitalSlideRenderFailure(
                request_id=902,
                purpose="native",
                focus_index=canvas.focus_index(),
                message="OSError: simulated native read failure",
            )
        )
        assert canvas._viewport_buffer_error_blocked  # noqa: SLF001
        assert not canvas._read_only  # noqa: SLF001
        assert canvas._drawing_anchor_raw == inside  # noqa: SLF001
    finally:
        canvas.shutdown()
        canvas.clear_document()
        canvas.close()
        store.close()


def test_repeated_full_field_steps_never_expose_dark_workspace_background(
    tmp_path: Path,
) -> None:
    app = QApplication.instance() or QApplication([])
    store = _create_coordinate_slide(tmp_path / "no-dark-step-flash.fdmslide")
    document = ImageDocument(
        id="no-dark-step-flash",
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
        _wait_for(app, canvas.pixel_work_enabled)
        directions = (100.0, 100.0, 100.0, -100.0, -100.0, -100.0) * 2
        for dx in directions:
            canvas.move_viewport_by(dx, 0.0)
            painted = QImage(canvas.size(), QImage.Format.Format_RGB32)
            painted.fill(QColor("#000000"))
            painter = QPainter(painted)
            canvas._draw_base_image(painter)  # noqa: SLF001
            painter.end()
            center_color = painted.pixelColor(
                canvas._content_rect().center().toPoint()  # noqa: SLF001
            )
            assert center_color not in (QColor("#000000"), QColor("#101820"))
        stats = canvas.renderer_stats()
        assert stats is not None
        assert stats.pending_requests <= 6
    finally:
        canvas.shutdown()
        canvas.clear_document()
        canvas.close()
        store.close()


def test_focus_wheel_uses_atomic_previews_and_one_canonical_exact_request() -> None:
    app = QApplication.instance() or QApplication([])

    class RecordingRenderer:
        def __init__(self) -> None:
            self.requests: list[DigitalSlideRenderRequest] = []

        def submit(self, request: DigitalSlideRenderRequest) -> None:
            self.requests.append(request)

        def close(self, *, timeout: float = 2.0) -> None:
            del timeout

    document = ImageDocument(
        id="focus-debounce",
        path="/tmp/focus-debounce.fdmslide",
        image_size=(800, 640),
        document_kind="digital_slide",
    )
    canvas = DigitalSlideCanvas()
    canvas.resize(440, 360)
    image = _tile_image(100, 80, "#ffffff")
    canvas.set_document(document, image)
    canvas._slide_manifest = DigitalSlideManifest(  # noqa: SLF001
        version=1,
        width=800,
        height=640,
        viewport_width=100,
        viewport_height=80,
        focus_levels=[-2, -1, 0, 1, 2],
    )
    canvas._browse_center = Point(250.0, 200.0)  # noqa: SLF001
    canvas._focus_index = 0  # noqa: SLF001
    canvas._zoom = canvas._native_field_fit_zoom()  # noqa: SLF001
    canvas._sync_pan_from_browse_center()  # noqa: SLF001
    canvas._update_native_viewport_origin()  # noqa: SLF001
    recorder = RecordingRenderer()
    canvas._renderer = recorder  # type: ignore[assignment]  # noqa: SLF001
    try:
        with patch.object(canvas, "isVisible", return_value=True):
            canvas.set_focus_index(1)
            canvas.set_focus_index(2)
            canvas.set_focus_index(3)
            deadline = monotonic() + 0.25
            while monotonic() < deadline:
                app.processEvents()
                if any(request.purpose == "native" for request in recorder.requests):
                    break

        native_requests = [
            request for request in recorder.requests if request.purpose == "native"
        ]
        assert len(native_requests) == 1
        assert native_requests[0].focus_index == 3
        assert native_requests[0].generation == canvas._view_generation  # noqa: SLF001
        assert native_requests[0].force_lod == 0
        assert native_requests[0].focus_direction == 1
        assert not any(request.purpose == "display" for request in recorder.requests)
        focus_preview_requests = [
            request
            for request in recorder.requests
            if request.purpose == "focus_preview"
        ]
        assert [request.focus_index for request in focus_preview_requests] == [1, 2, 3]
        assert all(
            request.generation <= canvas._view_generation  # noqa: SLF001
            for request in focus_preview_requests
        )
        assert focus_preview_requests[-1].generation == canvas._view_generation  # noqa: SLF001
        assert focus_preview_requests[-1].quality == "coarse"
        assert focus_preview_requests[-1].output_size_px == (100, 80)
        coarse_requests = [
            request for request in recorder.requests if request.purpose == "coarse"
        ]
        assert coarse_requests == []

        recorder.requests.clear()
        with patch.object(canvas, "isVisible", return_value=True):
            canvas.move_viewport_by(10.0, 0.0)
        assert not any(
            request.purpose == "coarse" for request in recorder.requests
        )
        assert any(
            request.purpose == "native" for request in recorder.requests
        )
    finally:
        canvas._renderer = None  # noqa: SLF001
        canvas.clear_document()
        canvas.close()


def test_native_focus_preview_replaces_handoff_atomically_before_exact() -> None:
    _app = QApplication.instance() or QApplication([])

    class RecordingRenderer:
        def __init__(self) -> None:
            self.requests: list[DigitalSlideRenderRequest] = []

        def submit(self, request: DigitalSlideRenderRequest) -> None:
            self.requests.append(request)

        def close(self, *, timeout: float = 2.0) -> None:
            del timeout

    document = ImageDocument(
        id="atomic-focus-preview",
        path="/tmp/atomic-focus-preview.fdmslide",
        image_size=(800, 640),
        document_kind="digital_slide",
    )
    canvas = DigitalSlideCanvas()
    canvas.resize(440, 360)
    canvas.set_document(document, _tile_image(100, 80, "#D72B3F"))
    canvas._slide_manifest = DigitalSlideManifest(  # noqa: SLF001
        version=1,
        width=800,
        height=640,
        viewport_width=100,
        viewport_height=80,
        focus_levels=[-1, 0, 1],
    )
    canvas._browse_center = Point(250.0, 200.0)  # noqa: SLF001
    canvas._focus_index = 0  # noqa: SLF001
    canvas._zoom = canvas._native_field_fit_zoom()  # noqa: SLF001
    canvas._sync_pan_from_browse_center()  # noqa: SLF001
    canvas._update_native_viewport_origin()  # noqa: SLF001
    origin = canvas.viewport_origin()
    old_frame = DigitalSlideRenderFrame(
        request_id=1,
        purpose="native",
        source_rect=(origin.x, origin.y, 100.0, 80.0),
        output_size_px=(100, 80),
        focus_index=0,
        device_pixel_ratio=1.0,
        lod=0,
        image=_tile_image(100, 80, "#D72B3F"),
        elapsed_ms=1.0,
        decoded_tiles=1,
        cache_hits=0,
        generation=canvas._view_generation,  # noqa: SLF001
        quality="final",
        pixel_exact=True,
        coverage_rects=((origin.x, origin.y, 100.0, 80.0),),
    )
    canvas._render_frame = old_frame  # noqa: SLF001
    canvas._image = old_frame.image  # noqa: SLF001
    canvas._native_frame_key = (  # noqa: SLF001
        int(round(origin.x)),
        int(round(origin.y)),
        0,
    )
    canvas._native_frame_ever_ready = True  # noqa: SLF001
    recorder = RecordingRenderer()
    canvas._renderer = recorder  # type: ignore[assignment]  # noqa: SLF001

    def center_color() -> QColor:
        painted = QImage(canvas.size(), QImage.Format.Format_RGB32)
        painted.fill(QColor("#000000"))
        painter = QPainter(painted)
        canvas._draw_base_image(painter)  # noqa: SLF001
        painter.end()
        return painted.pixelColor(
            canvas._content_rect().center().toPoint()  # noqa: SLF001
        )

    try:
        with patch.object(canvas, "isVisible", return_value=True):
            canvas.set_focus_index(1)
        canvas._final_render_timer.stop()  # noqa: SLF001
        preview_request = next(
            request
            for request in recorder.requests
            if request.purpose == "focus_preview"
        )
        target_preview = DigitalSlideRenderFrame(
            request_id=preview_request.request_id,
            purpose="focus_preview",
            source_rect=preview_request.source_rect,
            output_size_px=preview_request.output_size_px,
            focus_index=1,
            device_pixel_ratio=1.0,
            lod=1,
            image=_tile_image(*preview_request.output_size_px, "#2774C7"),
            elapsed_ms=2.0,
            decoded_tiles=1,
            cache_hits=0,
            generation=canvas._view_generation,  # noqa: SLF001
            quality="coarse",
            pixel_exact=False,
            coverage_rects=((origin.x, origin.y, 100.0, 80.0),),
            complete=False,
        )
        canvas._on_render_frame_ready(target_preview)  # noqa: SLF001
        assert canvas._coarse_render_frame is None  # noqa: SLF001
        assert canvas._focus_transition_frame is old_frame  # noqa: SLF001
        assert center_color() == QColor("#D72B3F")

        canvas._on_render_frame_ready(  # noqa: SLF001
            replace(target_preview, complete=True)
        )
        assert canvas._focus_transition_frame is None  # noqa: SLF001
        assert canvas._coarse_render_frame is not None  # noqa: SLF001
        assert canvas._coarse_render_frame.focus_index == 1  # noqa: SLF001
        assert center_color() == QColor("#2774C7")
        assert not canvas.pixel_work_enabled()
        assert canvas.vector_measurement_available(
            Point(origin.x + 50.0, origin.y + 40.0)
        )
        assert len(canvas._focus_preview_cache) == 1  # noqa: SLF001

        with patch.object(canvas, "isVisible", return_value=True):
            canvas._request_native_frame()  # noqa: SLF001
        native_request = next(
            request for request in recorder.requests if request.purpose == "native"
        )
        exact_frame = DigitalSlideRenderFrame(
            request_id=native_request.request_id,
            purpose="native",
            source_rect=native_request.source_rect,
            output_size_px=native_request.output_size_px,
            focus_index=1,
            device_pixel_ratio=1.0,
            lod=0,
            image=_tile_image(100, 80, "#2A9D8F"),
            elapsed_ms=3.0,
            decoded_tiles=1,
            cache_hits=1,
            generation=canvas._view_generation,  # noqa: SLF001
            quality="final",
            pixel_exact=True,
            coverage_rects=((origin.x, origin.y, 100.0, 80.0),),
        )
        canvas._on_render_frame_ready(exact_frame)  # noqa: SLF001
        assert canvas.pixel_work_enabled()
        assert canvas._coarse_render_frame is None  # noqa: SLF001
        assert center_color() == QColor("#2A9D8F")

        # A late preview can never cover a current exact frame.
        canvas._on_render_frame_ready(  # noqa: SLF001
            replace(target_preview, complete=True)
        )
        assert canvas._coarse_render_frame is None  # noqa: SLF001
        assert center_color() == QColor("#2A9D8F")
    finally:
        canvas._renderer = None  # noqa: SLF001
        canvas.clear_document()
        canvas.close()


def test_native_focus_preview_cache_restores_without_worker_request() -> None:
    class RecordingRenderer:
        def __init__(self) -> None:
            self.requests: list[DigitalSlideRenderRequest] = []
            self.preview_hits = 0

        def submit(self, request: DigitalSlideRenderRequest) -> None:
            self.requests.append(request)

        def record_preview_memory_hit(self) -> None:
            self.preview_hits += 1

        def close(self, *, timeout: float = 2.0) -> None:
            del timeout

    document = ImageDocument(
        id="focus-preview-cache",
        path="/tmp/focus-preview-cache.fdmslide",
        image_size=(800, 640),
        document_kind="digital_slide",
    )
    canvas = DigitalSlideCanvas()
    canvas.resize(440, 360)
    canvas.set_document(document, _tile_image(100, 80, "#ffffff"))
    canvas._slide_manifest = DigitalSlideManifest(  # noqa: SLF001
        version=1,
        width=800,
        height=640,
        viewport_width=100,
        viewport_height=80,
        focus_levels=[-1, 0, 1],
    )
    canvas._browse_center = Point(250.0, 200.0)  # noqa: SLF001
    canvas._focus_index = 1  # noqa: SLF001
    canvas._zoom = canvas._native_field_fit_zoom()  # noqa: SLF001
    canvas._sync_pan_from_browse_center()  # noqa: SLF001
    canvas._update_native_viewport_origin()  # noqa: SLF001
    origin = canvas.viewport_origin()
    cached = DigitalSlideRenderFrame(
        request_id=1,
        purpose="focus_preview",
        source_rect=(origin.x, origin.y, 100.0, 80.0),
        output_size_px=(100, 80),
        focus_index=1,
        device_pixel_ratio=1.0,
        lod=1,
        image=_tile_image(100, 80, "#2774C7"),
        elapsed_ms=2.0,
        decoded_tiles=1,
        cache_hits=0,
        generation=1,
        quality="coarse",
        pixel_exact=False,
        coverage_rects=((origin.x, origin.y, 100.0, 80.0),),
    )
    canvas._remember_focus_preview(cached)  # noqa: SLF001
    recorder = RecordingRenderer()
    canvas._renderer = recorder  # type: ignore[assignment]  # noqa: SLF001
    canvas._view_generation = 7  # noqa: SLF001
    try:
        with patch.object(canvas, "isVisible", return_value=True):
            assert canvas._request_native_focus_preview()  # noqa: SLF001
        restored = canvas._coarse_render_frame  # noqa: SLF001
        assert restored is not None
        assert restored.focus_index == 1
        assert restored.generation == 7
        assert restored.image.cacheKey() == cached.image.cacheKey()
        assert recorder.requests == []
        assert recorder.preview_hits == 1
    finally:
        canvas._renderer = None  # noqa: SLF001
        canvas.clear_document()
        canvas.close()


def test_native_focus_cache_revisits_same_field_without_proxy_request() -> None:
    class RecordingRenderer:
        def __init__(self) -> None:
            self.requests: list[DigitalSlideRenderRequest] = []

        def submit(self, request: DigitalSlideRenderRequest) -> None:
            self.requests.append(request)

        def close(self, *, timeout: float = 2.0) -> None:
            del timeout

    document = ImageDocument(
        id="focus-native-cache",
        path="/tmp/focus-native-cache.fdmslide",
        image_size=(800, 640),
        document_kind="digital_slide",
    )
    canvas = DigitalSlideCanvas()
    canvas.resize(440, 360)
    canvas.set_document(document, _tile_image(100, 80, "#ffffff"))
    canvas._slide_manifest = DigitalSlideManifest(  # noqa: SLF001
        version=1,
        width=800,
        height=640,
        viewport_width=100,
        viewport_height=80,
        focus_levels=[-2, -1, 0, 1, 2],
    )
    canvas._browse_center = Point(250.0, 200.0)  # noqa: SLF001
    canvas._zoom = canvas._native_field_fit_zoom()  # noqa: SLF001
    canvas._sync_pan_from_browse_center()  # noqa: SLF001
    canvas._update_native_viewport_origin()  # noqa: SLF001
    origin = canvas.viewport_origin()
    frames: list[DigitalSlideRenderFrame] = []
    for focus_index in range(5):
        frame = DigitalSlideRenderFrame(
            request_id=focus_index + 1,
            purpose="native",
            source_rect=(origin.x, origin.y, 100.0, 80.0),
            output_size_px=(100, 80),
            focus_index=focus_index,
            device_pixel_ratio=1.0,
            lod=0,
            image=_tile_image(100, 80, f"#{focus_index + 1:02x}4060"),
            elapsed_ms=1.0,
            decoded_tiles=1,
            cache_hits=0,
            generation=0,
            quality="final",
            pixel_exact=True,
            coverage_rects=((origin.x, origin.y, 100.0, 80.0),),
        )
        frames.append(frame)
        canvas._remember_display_frame(frame)  # noqa: SLF001

    recorder = RecordingRenderer()
    canvas._renderer = recorder  # type: ignore[assignment]  # noqa: SLF001
    canvas._focus_index = 4  # noqa: SLF001
    canvas._render_frame = frames[-1]  # noqa: SLF001
    canvas._image = frames[-1].image  # noqa: SLF001
    canvas._native_frame_key = (  # noqa: SLF001
        int(round(origin.x)),
        int(round(origin.y)),
        4,
    )
    canvas._native_frame_ever_ready = True  # noqa: SLF001
    try:
        canvas.set_focus_index(0)
        restored = canvas._render_frame  # noqa: SLF001
        assert restored is not None
        assert restored.focus_index == 0
        assert restored.pixel_exact
        assert restored.generation == canvas._view_generation  # noqa: SLF001
        assert recorder.requests == []
        assert canvas._coarse_render_frame is None  # noqa: SLF001
        assert canvas._presentation_preview_frame is None  # noqa: SLF001
    finally:
        canvas._renderer = None  # noqa: SLF001
        canvas.clear_document()
        canvas.close()


def test_interactive_renderer_uses_region_query_and_scaled_decode(
    tmp_path: Path,
) -> None:
    store = _create_coordinate_slide(tmp_path / "regional-query.fdmslide")
    manifest = store.read_manifest()
    store.close()
    results: list[DigitalSlideRenderFrame] = []
    failures: list[object] = []
    ready = Event()
    scaled_calls: list[tuple[int, int, int]] = []
    original_scaled = DigitalSlideStore.read_tile_image_scaled

    def scaled(self, tile_id: int, *, width: int, height: int):
        scaled_calls.append((tile_id, width, height))
        return original_scaled(self, tile_id, width=width, height=height)

    renderer = DigitalSlideRenderer(
        tmp_path / "regional-query.fdmslide",
        manifest,
        cache_root=tmp_path / "regional-cache",
        disk_cache_bytes=0,
        result_callback=lambda frame: (results.append(frame), ready.set()),
        failure_callback=lambda failure: (failures.append(failure), ready.set()),
    )
    try:
        with (
            patch.object(
                DigitalSlideStore,
                "list_tile_descriptors",
                side_effect=AssertionError("full focus descriptor scan"),
            ),
            patch.object(DigitalSlideStore, "read_tile_image_scaled", new=scaled),
        ):
            renderer.submit(
                DigitalSlideRenderRequest(
                    request_id=1,
                    purpose="coarse",
                    source_rect=(100.0, 80.0, 100.0, 80.0),
                    output_size_px=(25, 20),
                    focus_index=0,
                    device_pixel_ratio=1.0,
                    generation=1,
                    quality="coarse",
                )
            )
            assert ready.wait(4.0)
        assert not failures
        assert results[-1].coverage_rects == ((100.0, 80.0, 100.0, 80.0),)
        assert len(scaled_calls) == 1
        stats = renderer.stats()
        assert stats.descriptor_queries == 1
        assert stats.scaled_decodes == 1
    finally:
        renderer.close()


def test_prestored_focus_preview_is_published_before_any_tile_decode(
    tmp_path: Path,
) -> None:
    slide_path = tmp_path / "source-overview-first.fdmslide"
    store = _create_coordinate_slide(slide_path)
    overview = _tile_image(256, 205, "#ececec")
    assert store.write_focus_overviews({0: overview}) == 1
    manifest = store.read_manifest()
    store.close()
    frames: list[DigitalSlideRenderFrame] = []
    failures: list[object] = []
    ready = Event()
    renderer = DigitalSlideRenderer(
        slide_path,
        manifest,
        cache_root=tmp_path / "source-overview-cache",
        disk_cache_bytes=0,
        result_callback=lambda frame: (frames.append(frame), ready.set()),
        failure_callback=lambda failure: (failures.append(failure), ready.set()),
    )
    try:
        with (
            patch.object(
                DigitalSlideStore,
                "read_tile_image",
                side_effect=AssertionError("preview decoded an original tile"),
            ),
            patch.object(
                DigitalSlideStore,
                "read_tile_image_scaled",
                side_effect=AssertionError("preview decoded a scaled tile"),
            ),
        ):
            renderer.submit(
                DigitalSlideRenderRequest(
                    request_id=1,
                    purpose="preview",
                    source_rect=(0.0, 0.0, 400.0, 320.0),
                    output_size_px=(400, 320),
                    focus_index=0,
                    device_pixel_ratio=1.0,
                    generation=1,
                    quality="coarse",
                    preview_max_edge=1024,
                )
            )
            assert ready.wait(4.0)
        assert not failures
        assert frames[-1].decoded_tiles == 0
        assert frames[-1].quality == "coarse"
        assert frames[-1].image.size().toTuple() == (400, 320)
        assert renderer.stats().preview_source_hits == 1
    finally:
        renderer.close()


def test_new_generation_cancels_old_request_across_render_purposes(
    tmp_path: Path,
) -> None:
    slide_path = tmp_path / "generation-cancel.fdmslide"
    store = _create_coordinate_slide(slide_path)
    manifest = store.read_manifest()
    store.close()
    frames: list[DigitalSlideRenderFrame] = []
    failures: list[object] = []
    decode_started = Event()
    current_ready = Event()
    original_scaled = DigitalSlideStore.read_tile_image_scaled

    def slow_scaled(self, tile_id: int, *, width: int, height: int):
        decode_started.set()
        sleep(0.04)
        return original_scaled(self, tile_id, width=width, height=height)

    def publish(frame: DigitalSlideRenderFrame) -> None:
        frames.append(frame)
        if frame.generation == 2 and frame.complete:
            current_ready.set()

    renderer = DigitalSlideRenderer(
        slide_path,
        manifest,
        cache_root=tmp_path / "generation-cache",
        disk_cache_bytes=0,
        result_callback=publish,
        failure_callback=failures.append,
    )
    try:
        with patch.object(
            DigitalSlideStore,
            "read_tile_image_scaled",
            new=slow_scaled,
        ):
            renderer.submit(
                DigitalSlideRenderRequest(
                    request_id=1,
                    purpose="display",
                    source_rect=(0.0, 0.0, 400.0, 320.0),
                    output_size_px=(200, 160),
                    focus_index=0,
                    device_pixel_ratio=1.0,
                    generation=1,
                )
            )
            assert decode_started.wait(2.0)
            renderer.submit(
                DigitalSlideRenderRequest(
                    request_id=2,
                    purpose="coarse",
                    source_rect=(100.0, 80.0, 100.0, 80.0),
                    output_size_px=(50, 40),
                    focus_index=0,
                    device_pixel_ratio=1.0,
                    generation=2,
                    quality="coarse",
                )
            )
            assert current_ready.wait(4.0)
        assert not failures
        assert frames
        assert all(frame.generation == 2 for frame in frames)
        assert renderer.stats().cancelled >= 1
        assert renderer.stats().pending_requests == 0
    finally:
        renderer.close()


def test_progressive_macro_preview_persists_and_reopens_without_tile_decode(
    tmp_path: Path,
) -> None:
    slide_path = tmp_path / "progressive-macro.fdmslide"
    store = _create_coordinate_slide(slide_path)
    manifest = store.read_manifest()
    store.close()
    cache_root = tmp_path / "macro-cache"
    frames: list[DigitalSlideRenderFrame] = []
    failures: list[object] = []
    original_scaled = DigitalSlideStore.read_tile_image_scaled

    def slow_scaled(self, tile_id: int, *, width: int, height: int):
        sleep(0.02)
        return original_scaled(self, tile_id, width=width, height=height)

    renderer = DigitalSlideRenderer(
        slide_path,
        manifest,
        cache_root=cache_root,
        disk_cache_bytes=64 * 1024 * 1024,
        result_callback=frames.append,
        failure_callback=failures.append,
    )
    try:
        with patch.object(
            DigitalSlideStore,
            "read_tile_image_scaled",
            new=slow_scaled,
        ):
            renderer.submit(
                DigitalSlideRenderRequest(
                    request_id=1,
                    purpose="coarse",
                    source_rect=(0.0, 0.0, 400.0, 320.0),
                    output_size_px=(200, 160),
                    focus_index=0,
                    device_pixel_ratio=1.0,
                    generation=1,
                    quality="coarse",
                    preview_max_edge=1024,
                )
            )
            deadline = monotonic() + 4.0
            while monotonic() < deadline and not any(
                not frame.complete for frame in frames
            ):
                sleep(0.005)
            assert any(not frame.complete for frame in frames)
            first_progressive = next(frame for frame in frames if not frame.complete)
            assert 0 < len(first_progressive.coverage_rects) < 16
            unresolved_center: Point | None = None
            for row in range(4):
                for column in range(4):
                    candidate = Point(column * 100.0 + 50.0, row * 80.0 + 40.0)
                    if not any(
                        left <= candidate.x < left + width
                        and top <= candidate.y < top + height
                        for left, top, width, height in first_progressive.coverage_rects
                    ):
                        unresolved_center = candidate
                        break
                if unresolved_center is not None:
                    break
            assert unresolved_center is not None
            unresolved_color = first_progressive.image.pixelColor(
                int(unresolved_center.x * 0.5),
                int(unresolved_center.y * 0.5),
            )
            assert unresolved_color != QColor("#101820")
            while monotonic() < deadline and not any(frame.complete for frame in frames):
                sleep(0.005)
            assert any(frame.complete for frame in frames)
        progressive_coverage = [
            len(frame.coverage_rects) for frame in frames if not frame.complete
        ]
        assert progressive_coverage == sorted(progressive_coverage)
        assert not failures
    finally:
        renderer.close()

    assert list(cache_root.rglob("preview-edge-1024.png"))
    reopened: list[DigitalSlideRenderFrame] = []
    reopened_failures: list[object] = []
    reopened_ready = Event()
    second_renderer = DigitalSlideRenderer(
        slide_path,
        manifest,
        cache_root=cache_root,
        disk_cache_bytes=64 * 1024 * 1024,
        result_callback=lambda frame: (reopened.append(frame), reopened_ready.set()),
        failure_callback=lambda failure: (
            reopened_failures.append(failure),
            reopened_ready.set(),
        ),
    )
    try:
        with (
            patch.object(
                DigitalSlideStore,
                "read_tile_image",
                side_effect=AssertionError("macro preview decoded a tile"),
            ),
            patch.object(
                DigitalSlideStore,
                "read_tile_image_scaled",
                side_effect=AssertionError("macro preview decoded a scaled tile"),
            ),
        ):
            second_renderer.submit(
                DigitalSlideRenderRequest(
                    request_id=2,
                    purpose="preview",
                    source_rect=(0.0, 0.0, 400.0, 320.0),
                    output_size_px=(400, 320),
                    focus_index=0,
                    device_pixel_ratio=1.0,
                    generation=2,
                    quality="coarse",
                    preview_max_edge=1024,
                )
            )
            assert reopened_ready.wait(4.0)
        assert not reopened_failures
        assert reopened[-1].decoded_tiles == 0
        assert reopened[-1].quality == "coarse"
        assert second_renderer.stats().preview_disk_hits == 1
    finally:
        second_renderer.close()


def test_whole_slide_display_frame_is_reused_synchronously(
    tmp_path: Path,
) -> None:
    app = QApplication.instance() or QApplication([])
    store = _create_coordinate_slide(tmp_path / "whole-frame-cache.fdmslide")
    manifest = store.read_manifest()
    manifest.focus_levels = [0]
    store.write_manifest(manifest)
    document = ImageDocument(
        id="whole-frame-cache",
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
        canvas.fit_to_view()
        _wait_for(
            app,
            lambda: (
                canvas._render_frame is not None  # noqa: SLF001
                and canvas._render_frame.purpose == "display"  # noqa: SLF001
            ),
        )
        whole_frame = canvas._render_frame  # noqa: SLF001
        assert whole_frame is not None

        canvas.fit_native_viewport()
        _wait_for(app, canvas.pixel_work_enabled)
        renderer = canvas._renderer  # noqa: SLF001
        assert renderer is not None
        decoded_before_restore = renderer.stats().decoded_tiles

        canvas.fit_to_view()
        restored = canvas._render_frame  # noqa: SLF001
        assert restored is not None
        assert restored.purpose == "display"
        assert restored.elapsed_ms == 0.0
        assert restored.decoded_tiles == 0
        assert renderer.stats().decoded_tiles == decoded_before_restore
        assert renderer.stats().display_frame_hits >= 1
        assert restored.image.cacheKey() == whole_frame.image.cacheKey()
    finally:
        canvas.shutdown()
        canvas.clear_document()
        canvas.close()
        store.close()


def test_large_area_motion_keeps_final_frame_above_coarse_overlap() -> None:
    _app = QApplication.instance() or QApplication([])
    document = ImageDocument(
        id="large-area-sharp-overlap",
        path="/tmp/large-area-sharp-overlap.fdmslide",
        image_size=(1000, 800),
        document_kind="digital_slide",
    )
    canvas = DigitalSlideCanvas()
    canvas.resize(440, 360)
    canvas.set_document(document, _tile_image(100, 80, "#ffffff"))
    canvas._slide_manifest = DigitalSlideManifest(  # noqa: SLF001
        version=1,
        width=1000,
        height=800,
        viewport_width=100,
        viewport_height=80,
        focus_levels=[0],
    )
    canvas._zoom = 1.0  # noqa: SLF001
    canvas._browse_center = Point(520.0, 450.0)  # noqa: SLF001
    canvas._sync_pan_from_browse_center()  # noqa: SLF001
    canvas._view_generation = 2  # noqa: SLF001
    canvas._previous_render_frame = DigitalSlideRenderFrame(  # noqa: SLF001
        request_id=1,
        purpose="display",
        source_rect=(300.0, 290.0, 400.0, 320.0),
        output_size_px=(400, 320),
        focus_index=0,
        device_pixel_ratio=1.0,
        lod=1,
        image=_tile_image(400, 320, "#D72B3F"),
        elapsed_ms=1.0,
        decoded_tiles=4,
        cache_hits=0,
        generation=1,
        quality="final",
    )
    canvas._coarse_render_frame = DigitalSlideRenderFrame(  # noqa: SLF001
        request_id=2,
        purpose="coarse",
        source_rect=(320.0, 290.0, 400.0, 320.0),
        output_size_px=(200, 160),
        focus_index=0,
        device_pixel_ratio=1.0,
        lod=2,
        image=_tile_image(200, 160, "#2774C7"),
        elapsed_ms=1.0,
        decoded_tiles=4,
        cache_hits=0,
        generation=2,
        quality="coarse",
    )
    try:
        assert canvas.large_area_browse_active()
        painted = QImage(canvas.size(), QImage.Format.Format_RGB32)
        painted.fill(QColor("#000000"))
        painter = QPainter(painted)
        canvas._draw_base_image(painter)  # noqa: SLF001
        painter.end()

        overlap = canvas.image_to_widget(Point(400.0, 400.0)).toPoint()
        newly_exposed = canvas.image_to_widget(Point(710.0, 400.0)).toPoint()
        assert painted.pixelColor(overlap) == QColor("#D72B3F")
        assert painted.pixelColor(newly_exposed) == QColor("#2774C7")
    finally:
        canvas.clear_document()
        canvas.close()


def test_large_area_continuous_motion_still_schedules_final_frames() -> None:
    app = QApplication.instance() or QApplication([])

    class RecordingRenderer:
        def __init__(self) -> None:
            self.requests: list[DigitalSlideRenderRequest] = []

        def submit(self, request: DigitalSlideRenderRequest) -> None:
            self.requests.append(request)

        def close(self, *, timeout: float = 2.0) -> None:
            del timeout

    document = ImageDocument(
        id="large-area-continuous-final",
        path="/tmp/large-area-continuous-final.fdmslide",
        image_size=(1000, 800),
        document_kind="digital_slide",
    )
    canvas = DigitalSlideCanvas()
    canvas.resize(440, 360)
    canvas.set_document(document, _tile_image(100, 80, "#ffffff"))
    canvas._slide_manifest = DigitalSlideManifest(  # noqa: SLF001
        version=1,
        width=1000,
        height=800,
        viewport_width=100,
        viewport_height=80,
        focus_levels=[0],
    )
    canvas._zoom = 1.0  # noqa: SLF001
    canvas._browse_center = Point(500.0, 400.0)  # noqa: SLF001
    canvas._sync_pan_from_browse_center()  # noqa: SLF001
    recorder = RecordingRenderer()
    canvas._renderer = recorder  # type: ignore[assignment]  # noqa: SLF001
    try:
        with patch.object(canvas, "isVisible", return_value=True):
            deadline = monotonic() + 0.3
            while monotonic() < deadline and not any(
                request.purpose == "display" for request in recorder.requests
            ):
                # Model 50 Hz movement without ever entering a stopped state.
                canvas._request_interactive_frames()  # noqa: SLF001
                sleep(0.02)
                app.processEvents()

        assert any(request.purpose == "coarse" for request in recorder.requests)
        assert any(request.purpose == "display" for request in recorder.requests)
    finally:
        canvas._final_render_timer.stop()  # noqa: SLF001
        canvas._renderer = None  # noqa: SLF001
        canvas.clear_document()
        canvas.close()


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
