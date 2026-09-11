"""Count labels remain screen-sized across the 64-object cache boundary."""

import math
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QImage, QPainter
from test_canvas_overlay_handoff import (
    frame,
    pixels,
    warm,
)
from test_canvas_overlay_handoff import (
    scene as scene,  # noqa: PLC0414 - expose the shared pytest fixture
)

from fdm.geometry import Point
from fdm.models import Measurement
from fdm.ui.rendering import draw_measurements


def install_counts(scene, count):
    canvas, document, *_ = scene
    # Far-away counts must still participate in global numbering, but neither
    # their marker nor label may be prepared for this viewport.
    document.measurements = [
        Measurement(
            id=f"count-{index}",
            image_id=document.id,
            fiber_group_id=None,
            mode="count",
            measurement_kind="count",
            point_px=Point(50 + index * 10, 50),
        )
        for index in range(count - 2)
    ] + [
        Measurement(
            id=f"count-{index}",
            image_id=document.id,
            fiber_group_id=None,
            mode="count",
            measurement_kind="count",
            point_px=Point(8500 + (index % 2) * 120, 4350),
        )
        for index in range(count - 2, count)
    ]
    document.view_state.selected_measurement_id = None
    document.mark_measurement_geometry_changed()
    document.mark_session_dirty()
    canvas.set_settings(replace(canvas._settings, show_count_numbers=True))
    canvas._sync_overlay_visual_state()


def complete_preview(scene):
    canvas, _, _, preview, _, pool = scene
    canvas._request_scene_preview()
    # An empty stable layer still needs one preparation turn to publish.
    while canvas._overlay_preview_timer.isActive():
        canvas._overlay_preview_timer.stop()
        canvas._prepare_scene_preview()
    pool.complete(preview)


def preview_pixels(canvas, *, direct=False, clip=None):
    dpr = canvas.devicePixelRatioF()
    image = QImage(
        math.ceil(canvas.width() * dpr),
        math.ceil(canvas.height() * dpr),
        QImage.Format.Format_ARGB32_Premultiplied,
    )
    image.setDevicePixelRatio(dpr)
    image.fill(0)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    try:
        if clip is not None:
            painter.setClipRect(clip, Qt.ClipOperation.IntersectClip)
        if direct:
            visible, numbers = canvas._measurement_render_inputs(
                canvas._paint_context().image_rect
            )
            draw_measurements(
                painter,
                canvas._document,
                canvas.image_to_widget,
                canvas._screen_passive_settings,
                line_width=2.0,
                endpoint_radius=4.0,
                measurement_sequence=visible,
                count_numbers=numbers,
                use_sprite_cache=True,
                cull_by_geometry=False,
            )
        else:
            canvas._draw_scene_preview(painter, canvas._paint_context())
    finally:
        painter.end()
    return pixels(image)


@pytest.mark.parametrize("scene", ["digital"], indirect=True)
@pytest.mark.parametrize("count", [60, 63, 64, 70, 500])
@pytest.mark.parametrize("zoom", [0.25, 1.0, 8.0, 40.0])
@pytest.mark.parametrize("dpr", [1, 1.25, 1.5, 2])
def test_whole_slide_preview_keeps_count_marker_and_number_size(
    scene, monkeypatch, count, zoom, dpr
):
    canvas, document, *_ = scene
    monkeypatch.setattr(canvas, "devicePixelRatioF", lambda: dpr)
    install_counts(scene, count)
    canvas._zoom = zoom
    canvas._pan = Point(300 - 8500 * zoom, 250 - 4350 * zoom)
    document_before = [measurement.to_dict() for measurement in document.measurements]
    complete_preview(scene)
    with patch.object(
        canvas, "_capture_overlay_commands", wraps=canvas._capture_overlay_commands
    ) as capture:
        actual = preview_pixels(canvas)
        expected = preview_pixels(canvas, direct=True)
        # This verifies glyphs, marker radii, offsets and document-wide numbers,
        # not only a permissive bounding-box limit on the rendered text.
        np.testing.assert_array_equal(actual, expected)
        assert actual[:, :, 3].any()
        assert not capture.called
    assert [
        measurement.to_dict() for measurement in document.measurements
    ] == document_before


@pytest.mark.parametrize("scene", ["digital"], indirect=True)
@pytest.mark.parametrize("numbers", [False, True])
def test_count_preview_obeys_partial_repaint_and_current_category_style(scene, numbers):
    canvas, document, *_ = scene
    install_counts(scene, 70)
    first = document.create_group(label="第一类", color="#F06040")
    second = document.create_group(label="第二类", color="#2070E0")
    for i, measurement in enumerate(document.measurements):
        measurement.fiber_group_id = (first if i % 2 else second).id
    canvas.set_settings(replace(canvas._settings, show_count_numbers=numbers))
    complete_preview(scene)
    # Slice through the visible marker and label. This is the missing-tile
    # clip that the handoff renderer applies next to already precise tiles.
    clip = QRectF(307, 244, 40, 35)
    np.testing.assert_array_equal(
        preview_pixels(canvas, clip=clip),
        preview_pixels(canvas, direct=True, clip=clip),
    )
    # Delete an earlier count: a previous stable preview must never retain
    # its number or category. The global count ordinals are recomputed.
    document.measurements.pop(0)
    document.mark_measurement_geometry_changed()
    document.mark_session_dirty()
    canvas.notify_document_visual_changed()
    np.testing.assert_array_equal(
        preview_pixels(canvas), preview_pixels(canvas, direct=True)
    )


@pytest.mark.parametrize("scene", ["digital"], indirect=True)
@pytest.mark.parametrize("dpr", [1, 1.25, 1.5, 2])
def test_count_preview_and_exact_tiles_have_identical_visible_output(
    scene, monkeypatch, dpr
):
    canvas, *_ = scene
    monkeypatch.setattr(canvas, "devicePixelRatioF", lambda: dpr)
    install_counts(scene, 70)
    canvas._zoom = 2
    canvas._pan = Point(300 - 8500 * 2, 250 - 4350 * 2)
    complete_preview(scene)
    pending = pixels(frame(canvas))
    precise = pixels(warm(scene))
    np.testing.assert_array_equal(pending, precise)


@pytest.mark.parametrize("scene", ["digital"], indirect=True)
def test_precise_overlay_jobs_start_while_source_image_io_is_pending(scene):
    canvas, _, _, _, pool, _ = scene
    install_counts(scene, 70)
    keys = canvas._visible_overlay_tile_keys(canvas._paint_context())
    with patch.object(
        canvas, "renderer_stats", return_value=SimpleNamespace(pending_requests=3)
    ):
        canvas._enqueue_overlay_tiles(keys)
        canvas._start_next_overlay_tile()
    assert pool.jobs, "image/overview I/O must not block immutable overlay rendering"


@pytest.mark.parametrize("scene", ["digital"], indirect=True)
def test_active_pan_still_defers_obsolete_overlay_jobs(scene):
    canvas, _, _, _, pool, _ = scene
    install_counts(scene, 70)
    canvas._panning = True
    try:
        canvas._enqueue_overlay_tiles(
            canvas._visible_overlay_tile_keys(canvas._paint_context())
        )
        canvas._start_next_overlay_tile()
        assert not pool.jobs
    finally:
        canvas._panning = False


@pytest.mark.parametrize("scene", ["digital"], indirect=True)
def test_fast_tile_completion_yields_without_leaving_a_stale_active_job(scene):
    canvas, _, exact, _, pool, _ = scene
    install_counts(scene, 70)
    keys = canvas._visible_overlay_tile_keys(canvas._paint_context())
    assert len(keys) > 1
    started = []

    def inline_start(job):
        started.append(job)
        job.run()

    with patch.object(pool, "start", side_effect=inline_start):
        canvas._enqueue_overlay_tiles(keys)
        canvas._start_next_overlay_tile()
        # An empty tile can finish before request() returns, including with a
        # real worker. The completion must not recurse through the whole queue
        # or have its cleared state overwritten with the already finished key.
        assert len(started) == 1
        assert canvas._overlay_tile_active is None
        assert canvas._overlay_tile_build_scheduled
        while canvas._overlay_tile_queue:
            before = len(started)
            canvas._start_next_overlay_tile()
            assert len(started) - before <= 1
        assert canvas._overlay_tile_active is None
        assert all(exact.contains(key) for key in keys)


@pytest.mark.parametrize("scene", ["digital"], indirect=True)
def test_scaled_scene_raster_calls_never_exceed_the_widget_viewport(scene):
    canvas, _, _, preview, *_ = scene
    key = canvas._scene_preview_key()
    raster = QImage(1536, 1536, QImage.Format.Format_ARGB32_Premultiplied)
    raster.fill(0xFF2050A0)
    canvas._zoom = 40
    canvas._pan = Point(-330000, -170000)
    painter = SimpleNamespace(
        save=lambda: None,
        restore=lambda: None,
        setRenderHint=lambda *_: None,
    )
    calls = []
    painter.drawImage = lambda *args: calls.append(args)
    # Exercise the real whole-scene raster mapping with a detached painter
    # probe so an accidentally gigantic native allocation cannot hang pytest.
    with (
        patch.object(preview, "get_payload", return_value=(raster, None)),
        patch.object(canvas, "_request_scene_preview", return_value=key),
        patch.object(canvas, "_measurement_render_inputs", return_value=([], None)),
    ):
        canvas._draw_scene_preview(painter, canvas._paint_context())
    assert len(calls) == 1
    target, _image, source = calls[0]
    assert QRectF(canvas.rect()).contains(target)
    assert source.width() < 1536
    assert source.topLeft() != QPointF(0, 0)
