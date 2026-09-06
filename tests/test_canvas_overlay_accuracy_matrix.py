from __future__ import annotations

import math
import os
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtGui import QColor, QImage, QPalette
from PySide6.QtWidgets import QApplication

from fdm.geometry import Line, Point
from fdm.models import ImageDocument, Measurement, ObjectAppearanceOverride
from fdm.settings import AppSettings, MeasurementLabelStyleSettings
import fdm.ui.canvas as canvas_module
from fdm.ui.canvas import DocumentCanvas
from fdm.ui.canvas_overlay_cache import CanvasOverlayTileCache


APP = QApplication.instance() or QApplication([])


class _InlineThreadPool:
    def start(self, runnable) -> None:
        runnable.run()


class _DevicePixelRatioCanvas(DocumentCanvas):
    def __init__(self, device_pixel_ratio: float) -> None:
        self._test_device_pixel_ratio = float(device_pixel_ratio)
        super().__init__()

    def devicePixelRatioF(self) -> float:
        return self._test_device_pixel_ratio


def _area(
    measurement_id: str,
    outer: list[tuple[float, float]],
    *inner_rings: list[tuple[float, float]],
    exact_area_px: float | None = None,
) -> Measurement:
    polygon = [Point(x, y) for x, y in outer]
    rings = [
        polygon,
        *[[Point(x, y) for x, y in ring] for ring in inner_rings],
    ]
    measurement = Measurement(
        id=measurement_id,
        image_id="accuracy-document",
        fiber_group_id=None,
        mode="polygon_area",
        measurement_kind="area",
        polygon_px=polygon,
        area_rings_px=rings,
        exact_area_px=exact_area_px,
        created_at="2026-07-19T00:00:00+00:00",
    )
    measurement.recalculate(None)
    return measurement


def _complex_area_document() -> ImageDocument:
    document = ImageDocument(
        id="accuracy-document",
        path="/tmp/accuracy-document.png",
        image_size=(768, 480),
    )
    document.measurements = [
        _area(
            "hole",
            [(40, 40), (190, 40), (190, 190), (40, 190)],
            [(85, 85), (145, 85), (145, 145), (85, 145)],
        ),
        # A self-crossing bow tie exercises the same odd-even fill rule used
        # by measurement semantics and hit testing.
        _area(
            "bow-tie",
            [(245, 40), (395, 190), (245, 190), (395, 40)],
        ),
        _area(
            "concave",
            [
                (475, 40),
                (675, 40),
                (675, 190),
                (595, 190),
                (595, 105),
                (555, 105),
                (555, 190),
                (475, 190),
            ],
        ),
        _area(
            "exact-mask",
            [(115, 270), (300, 270), (300, 420), (115, 420)],
            [(170, 315), (245, 315), (245, 375), (170, 375)],
            exact_area_px=4321.25,
        ),
    ]
    document.recalculate_measurements()
    return document


def _source_image() -> QImage:
    image = QImage(768, 480, QImage.Format.Format_RGB32)
    image.fill(QColor("#F6F7F8"))
    return image


def _frame(canvas: DocumentCanvas, device_pixel_ratio: float) -> QImage:
    physical_width = int(math.ceil(canvas.width() * device_pixel_ratio))
    physical_height = int(math.ceil(canvas.height() * device_pixel_ratio))
    image = QImage(
        physical_width,
        physical_height,
        QImage.Format.Format_ARGB32_Premultiplied,
    )
    image.setDevicePixelRatio(device_pixel_ratio)
    image.fill(0)
    canvas.render(image)
    return image


def _pixels(image: QImage) -> np.ndarray:
    raw = np.frombuffer(
        image.constBits(),
        dtype=np.uint8,
        count=image.sizeInBytes(),
    ).reshape((image.height(), image.bytesPerLine()))
    return raw[:, : image.width() * 4].reshape((image.height(), image.width(), 4))


def _one_device_pixel_edge_envelope(reference: np.ndarray) -> np.ndarray:
    edge = np.zeros(reference.shape[:2], dtype=bool)
    vertical = np.any(reference[1:] != reference[:-1], axis=2)
    horizontal = np.any(reference[:, 1:] != reference[:, :-1], axis=2)
    edge[1:] |= vertical
    edge[:-1] |= vertical
    edge[:, 1:] |= horizontal
    edge[:, :-1] |= horizontal
    envelope = edge.copy()
    height, width = edge.shape
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            source_y_start = max(0, -dy)
            source_y_end = min(height, height - dy)
            source_x_start = max(0, -dx)
            source_x_end = min(width, width - dx)
            target_y_start = max(0, dy)
            target_y_end = min(height, height + dy)
            target_x_start = max(0, dx)
            target_x_end = min(width, width + dx)
            envelope[
                target_y_start:target_y_end,
                target_x_start:target_x_end,
            ] |= edge[
                source_y_start:source_y_end,
                source_x_start:source_x_end,
            ]
    return envelope


def _assert_stable_frames_equivalent(direct: QImage, cached: QImage) -> None:
    direct_pixels = _pixels(direct)
    cached_pixels = _pixels(cached)
    assert direct_pixels.shape == cached_pixels.shape
    delta = np.abs(
        direct_pixels.astype(np.int16) - cached_pixels.astype(np.int16)
    )
    changed = np.any(delta != 0, axis=2)
    if not np.any(changed):
        return
    edge_envelope = _one_device_pixel_edge_envelope(direct_pixels)
    # Sub-channel premultiplied rounding may differ by one in a solid region;
    # larger visible differences are permitted only in the one-device-pixel
    # antialias envelope.
    outside_edge = ~edge_envelope
    assert int(delta[outside_edge].max(initial=0)) <= 1
    assert not np.any((np.max(delta, axis=2) > 1) & outside_edge)


def _render_direct(
    canvas: DocumentCanvas,
    device_pixel_ratio: float,
) -> QImage:
    with patch.dict(
        os.environ,
        {
            "FDM_ENABLE_CANVAS_OVERLAY_CACHE": "1",
            "FDM_DISABLE_CANVAS_OVERLAY_CACHE": "1",
        },
        clear=False,
    ):
        return _frame(canvas, device_pixel_ratio)


def _warm_visible_tiles(
    canvas: DocumentCanvas,
    cache: CanvasOverlayTileCache,
) -> list[object]:
    canvas._sync_overlay_visual_state()  # noqa: SLF001
    keys = canvas._visible_overlay_tile_keys(canvas._paint_context())  # noqa: SLF001
    for key in keys:
        if cache.contains(key):
            continue
        snapshot = canvas._build_overlay_tile_snapshot(key)  # noqa: SLF001
        assert snapshot is not None
        assert cache.request(snapshot)
    assert all(cache.contains(key) for key in keys)
    return keys


def _render_cached(
    canvas: DocumentCanvas,
    cache: CanvasOverlayTileCache,
    device_pixel_ratio: float,
    *,
    selected_background_redraws: int = 0,
) -> QImage:
    with patch.dict(
        os.environ,
        {"FDM_ENABLE_CANVAS_OVERLAY_CACHE": "1"},
        clear=False,
    ):
        os.environ.pop("FDM_DISABLE_CANVAS_OVERLAY_CACHE", None)
        _warm_visible_tiles(canvas, cache)
        hits_before = cache.stats().hits
        original_direct_draw = canvas._draw_measurements_direct  # noqa: SLF001
        direct_draw_count = 0

        def track_direct_draw(*args, **kwargs):
            nonlocal direct_draw_count
            direct_draw_count += 1
            return original_direct_draw(*args, **kwargs)

        # Use a plain callable instead of a Mock: retaining a paintEvent's
        # transient QPainter in mock.call arguments can outlive its QImage and
        # corrupt the next Qt test during garbage collection.
        with patch.object(
            canvas,
            "_draw_measurements_direct",
            new=track_direct_draw,
        ):
            frame = _frame(canvas, device_pixel_ratio)
        assert direct_draw_count == selected_background_redraws
        assert cache.stats().hits > hits_before
        return frame


def _make_canvas(
    document: ImageDocument,
    cache: CanvasOverlayTileCache,
    device_pixel_ratio: float,
) -> _DevicePixelRatioCanvas:
    canvas = _DevicePixelRatioCanvas(device_pixel_ratio)
    canvas.resize(768, 480)
    canvas.set_settings(
        AppSettings(
            area_measurement_label_style=MeasurementLabelStyleSettings(
                enabled=True,
                font_size=14,
                decimals=2,
                background_enabled=True,
            )
        )
    )
    canvas.set_document(document, _source_image())
    canvas._zoom = 1.0  # noqa: SLF001
    canvas._pan = Point(0.0, 0.0)  # noqa: SLF001
    return canvas


def _dispose_canvas(
    canvas: DocumentCanvas | None,
    cache: CanvasOverlayTileCache,
) -> None:
    if canvas is not None:
        canvas.clear_document()
        canvas.close()
        canvas.deleteLater()
        QCoreApplication.sendPostedEvents(
            canvas,
            QEvent.Type.DeferredDelete,
        )
        APP.processEvents()
    cache.clear()


@pytest.mark.parametrize("device_pixel_ratio", [1.0, 1.25, 1.5, 2.0])
@pytest.mark.parametrize("dark", [False, True])
def test_complex_odd_even_area_cache_matches_direct_at_supported_dpr(
    device_pixel_ratio: float,
    dark: bool,
) -> None:
    document = _complex_area_document()
    cache = CanvasOverlayTileCache(
        max_entries=32,
        max_bytes=64 * 1024 * 1024,
        thread_pool=_InlineThreadPool(),
    )
    before = [measurement.to_dict() for measurement in document.measurements]
    values_before = [
        (measurement.area_px, measurement.area_unit, measurement.exact_area_px)
        for measurement in document.measurements
    ]
    canvas = None
    with patch.object(canvas_module, "canvas_overlay_tile_cache", cache):
        try:
            canvas = _make_canvas(document, cache, device_pixel_ratio)
            palette = canvas.palette()
            palette.setColor(QPalette.ColorRole.Window, QColor("#1b2026" if dark else "#fafafa"))
            palette.setColor(QPalette.ColorRole.WindowText, QColor("white" if dark else "black"))
            canvas.setPalette(palette)
            direct = _render_direct(canvas, device_pixel_ratio)
            cached = _render_cached(canvas, cache, device_pixel_ratio)

            _assert_stable_frames_equivalent(direct, cached)
            assert [measurement.to_dict() for measurement in document.measurements] == before
            assert [
                (
                    measurement.area_px,
                    measurement.area_unit,
                    measurement.exact_area_px,
                )
                for measurement in document.measurements
            ] == values_before
            assert document.get_measurement("hole").area_px == pytest.approx(18_900.0)
            assert document.get_measurement("bow-tie").area_px > 0.0
            assert document.get_measurement("exact-mask").area_px == 4321.25
        finally:
            _dispose_canvas(canvas, cache)


@pytest.mark.parametrize("device_pixel_ratio", [1.0, 1.25, 1.5])
def test_dense_length_tile_raster_stays_within_antialias_envelope(
    device_pixel_ratio: float,
) -> None:
    document = ImageDocument(
        id="dense-length-document",
        path="/tmp/dense-length.png",
        image_size=(768, 480),
    )
    document.measurements = [
        Measurement(
            id=f"line-{index}",
            image_id=document.id,
            fiber_group_id=None,
            mode="manual",
            measurement_kind="line",
            line_px=Line(
                Point(24.0 + ((index % 12) * 28.0), 24.0 + ((index // 12) * 42.0)),
                Point(38.0 + ((index % 12) * 28.0), 34.0 + ((index // 12) * 42.0)),
            ),
            diameter_px=17.2,
            diameter_unit=8.6,
        )
        for index in range(96)
    ]
    cache = CanvasOverlayTileCache(
        max_entries=32,
        max_bytes=64 * 1024 * 1024,
        thread_pool=_InlineThreadPool(),
    )
    canvas = None
    with patch.object(canvas_module, "canvas_overlay_tile_cache", cache):
        try:
            canvas = _make_canvas(document, cache, device_pixel_ratio)
            canvas.set_settings(
                AppSettings(
                    length_measurement_label_style=MeasurementLabelStyleSettings(
                        enabled=True,
                        font_size=14,
                        decimals=2,
                        background_enabled=True,
                    )
                )
            )
            direct = _render_direct(canvas, device_pixel_ratio)
            cached = _render_cached(canvas, cache, device_pixel_ratio)

            _assert_stable_frames_equivalent(direct, cached)
            keys = canvas._visible_overlay_tile_keys(  # noqa: SLF001
                canvas._paint_context()  # noqa: SLF001
            )
            assert any(
                (payload := cache.get_payload(key)) is not None
                and payload[0] is not None
                for key in keys
            )
        finally:
            _dispose_canvas(canvas, cache)


def test_dense_disjoint_areas_with_overlapping_labels_keep_exact_composition() -> None:
    document = ImageDocument(
        id="dense-area-label-document",
        path="/tmp/dense-area-labels.png",
        image_size=(768, 480),
    )
    document.measurements = [
        _area(
            f"dense-area-{index}",
            [
                (20.0 + (index * 7.0), 220.0),
                (23.0 + (index * 7.0), 220.0),
                (23.0 + (index * 7.0), 223.0),
                (20.0 + (index * 7.0), 223.0),
            ],
        )
        for index in range(70)
    ]
    cache = CanvasOverlayTileCache(
        max_entries=32,
        max_bytes=64 * 1024 * 1024,
        thread_pool=_InlineThreadPool(),
    )
    canvas = None
    with patch.object(canvas_module, "canvas_overlay_tile_cache", cache):
        try:
            canvas = _make_canvas(document, cache, 1.0)
            canvas._sync_overlay_visual_state()  # noqa: SLF001
            initial_keys = canvas._visible_overlay_tile_keys(  # noqa: SLF001
                canvas._paint_context()  # noqa: SLF001
            )
            snapshots = [
                snapshot
                for key in initial_keys
                if (
                    snapshot := canvas._build_overlay_tile_snapshot(key)  # noqa: SLF001
                )
                is not None
            ]
            assert any(
                len(snapshot.area_commands) > 64
                and snapshot.adaptive_composition
                for snapshot in snapshots
            )
            direct = _render_direct(canvas, 1.0)
            cached = _render_cached(canvas, cache, 1.0)

            _assert_stable_frames_equivalent(direct, cached)
        finally:
            _dispose_canvas(canvas, cache)


@pytest.mark.parametrize("device_pixel_ratio", [1.0, 1.25, 1.5])
def test_selection_round_trip_reuses_tiles_without_stale_passive_body(
    device_pixel_ratio: float,
) -> None:
    document = _complex_area_document()
    cache = CanvasOverlayTileCache(
        max_entries=32,
        max_bytes=64 * 1024 * 1024,
        thread_pool=_InlineThreadPool(),
    )
    canvas = None
    with patch.object(canvas_module, "canvas_overlay_tile_cache", cache):
        try:
            canvas = _make_canvas(document, cache, device_pixel_ratio)
            original_keys = _warm_visible_tiles(canvas, cache)
            before = [measurement.to_dict() for measurement in document.measurements]

            canvas.set_selected_measurement("hole")
            canvas._sync_overlay_visual_state()  # noqa: SLF001
            assert all(cache.contains(key) for key in original_keys)
            selected_direct = _render_direct(canvas, device_pixel_ratio)
            selected_cached = _render_cached(
                canvas,
                cache,
                device_pixel_ratio,
            )
            _assert_stable_frames_equivalent(selected_direct, selected_cached)

            selected_keys = _warm_visible_tiles(canvas, cache)
            canvas.set_selected_measurement(None)
            canvas._sync_overlay_visual_state()  # noqa: SLF001
            assert all(cache.contains(key) for key in selected_keys)
            deselected_direct = _render_direct(canvas, device_pixel_ratio)
            deselected_cached = _render_cached(
                canvas,
                cache,
                device_pixel_ratio,
            )
            _assert_stable_frames_equivalent(deselected_direct, deselected_cached)
            assert [measurement.to_dict() for measurement in document.measurements] == before
        finally:
            _dispose_canvas(canvas, cache)


@pytest.mark.parametrize("device_pixel_ratio", [1.0, 1.5])
@pytest.mark.parametrize("mixed_geometry", [False, True])
def test_screen_label_modes_match_cached_render_and_restore_without_stale_labels(
    device_pixel_ratio: float, mixed_geometry: bool,
) -> None:
    document = _complex_area_document()
    if mixed_geometry:
        document.measurements.append(Measurement(
            id="line", image_id=document.id, fiber_group_id=None, mode="manual",
            line_px=Line(Point(450, 350), Point(650, 350)),
        ))
        document.recalculate_measurements()
    before = [measurement.to_dict() for measurement in document.measurements]
    cache = CanvasOverlayTileCache(
        max_entries=32, max_bytes=64 * 1024 * 1024, thread_pool=_InlineThreadPool(),
    )
    canvas = None
    with patch.object(canvas_module, "canvas_overlay_tile_cache", cache):
        try:
            canvas = _make_canvas(document, cache, device_pixel_ratio)
            original = _render_cached(canvas, cache, device_pixel_ratio)
            canvas.set_screen_measurement_labels("hidden")
            hidden = _render_direct(canvas, device_pixel_ratio)
            _assert_stable_frames_equivalent(hidden, _render_cached(canvas, cache, device_pixel_ratio))
            assert not np.array_equal(_pixels(original), _pixels(hidden))
            canvas.set_screen_measurement_labels("selected")
            assert np.array_equal(_pixels(_render_direct(canvas, device_pixel_ratio)), _pixels(hidden))
            for selected_id in ("hole", "line" if mixed_geometry else "bow-tie", None):
                keys = _warm_visible_tiles(canvas, cache)
                canvas.set_selected_measurement(selected_id)
                canvas._sync_overlay_visual_state()
                assert all(cache.contains(key) for key in keys)
                _assert_stable_frames_equivalent(
                    _render_direct(canvas, device_pixel_ratio),
                    _render_cached(
                        canvas, cache, device_pixel_ratio,
                        # A selected line uses the existing local background
                        # redraw to remove its passive body beneath handles.
                        selected_background_redraws=int(selected_id == "line"),
                    ),
                )
            canvas.set_screen_measurement_labels("all")
            restored = _render_cached(canvas, cache, device_pixel_ratio)
            assert np.array_equal(_pixels(original), _pixels(restored))
            assert [measurement.to_dict() for measurement in document.measurements] == before
        finally:
            _dispose_canvas(canvas, cache)


def test_style_change_and_delete_invalidate_old_tiles_without_ghosts() -> None:
    document = _complex_area_document()
    cache = CanvasOverlayTileCache(
        max_entries=32,
        max_bytes=64 * 1024 * 1024,
        thread_pool=_InlineThreadPool(),
    )
    canvas = None
    with patch.object(canvas_module, "canvas_overlay_tile_cache", cache):
        try:
            canvas = _make_canvas(document, cache, 1.0)
            original_keys = _warm_visible_tiles(canvas, cache)
            hole = document.get_measurement("hole")
            hole.appearance = ObjectAppearanceOverride(
                stroke_color="#FF00FF",
                stroke_width=5.0,
            )
            document.mark_session_dirty()
            expected_after_style = [m.to_dict() for m in document.measurements]
            canvas._sync_overlay_visual_state()  # noqa: SLF001
            assert any(not cache.contains(key) for key in original_keys)
            style_direct = _render_direct(canvas, 1.0)
            style_cached = _render_cached(canvas, cache, 1.0)
            _assert_stable_frames_equivalent(style_direct, style_cached)
            assert [m.to_dict() for m in document.measurements] == expected_after_style

            styled_keys = _warm_visible_tiles(canvas, cache)
            document.remove_measurement("bow-tie")
            expected_after_delete = [m.to_dict() for m in document.measurements]
            canvas._sync_overlay_visual_state()  # noqa: SLF001
            assert any(not cache.contains(key) for key in styled_keys)
            delete_direct = _render_direct(canvas, 1.0)
            delete_cached = _render_cached(canvas, cache, 1.0)
            _assert_stable_frames_equivalent(delete_direct, delete_cached)
            assert [m.to_dict() for m in document.measurements] == expected_after_delete
            assert document.get_measurement("bow-tie") is None
        finally:
            _dispose_canvas(canvas, cache)


def test_disable_overlay_cache_environment_switch_forces_exact_direct_path() -> None:
    document = _complex_area_document()
    cache = CanvasOverlayTileCache(
        max_entries=32,
        max_bytes=64 * 1024 * 1024,
        thread_pool=_InlineThreadPool(),
    )
    canvas = None
    with patch.object(canvas_module, "canvas_overlay_tile_cache", cache):
        try:
            canvas = _make_canvas(document, cache, 1.0)
            before = [measurement.to_dict() for measurement in document.measurements]
            original_direct_draw = canvas._draw_measurements_direct  # noqa: SLF001
            direct_draw_count = 0

            def track_direct_draw(*args, **kwargs):
                nonlocal direct_draw_count
                direct_draw_count += 1
                return original_direct_draw(*args, **kwargs)

            with (
                patch.dict(
                    os.environ,
                    {
                        "FDM_ENABLE_CANVAS_OVERLAY_CACHE": "1",
                        "FDM_DISABLE_CANVAS_OVERLAY_CACHE": "1",
                    },
                    clear=False,
                ),
                patch.object(
                    canvas,
                    "_draw_measurements_direct",
                    new=track_direct_draw,
                ),
            ):
                disabled_frame = _frame(canvas, 1.0)
            assert direct_draw_count == 1
            assert cache.stats().entries == 0

            direct_frame = _render_direct(canvas, 1.0)
            _assert_stable_frames_equivalent(direct_frame, disabled_frame)
            assert [measurement.to_dict() for measurement in document.measurements] == before
        finally:
            _dispose_canvas(canvas, cache)
