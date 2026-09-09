"""Visible front/pending regression tests, including incomplete worker batches."""

from dataclasses import replace
import math
from unittest.mock import patch

import numpy as np
import pytest
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QImage, QPalette

from fdm.geometry import Line, Point
from fdm.models import ImageDocument, Measurement
from fdm.settings import AppSettings, MeasurementLabelStyleSettings
from fdm.ui import canvas as canvas_module
from fdm.ui.canvas import DocumentCanvas
from fdm.ui.canvas_overlay_cache import CanvasOverlayTileCache
from fdm.ui.draft_preview_cache import DraftPreviewCache
from fdm.ui.digital_slide_canvas import DigitalSlideCanvas
from fdm.services.digital_slide_store import DigitalSlideManifest
from fdm.services.digital_slide_renderer import DigitalSlideRenderFrame


class DeferredPool:
    def __init__(self):
        self.jobs = []

    def complete(self, cache, count=None):
        count = len(self.jobs) if count is None else count
        jobs, self.jobs = self.jobs[:count], self.jobs[count:]
        for job in jobs:
            job.run()
        cache._drain_completions()

    def start(self, job):
        self.jobs.append(job)


def area(identity, left=100, top=100, *, hole=False):
    ring = [
        Point(left, top),
        Point(left + 90, top),
        Point(left + 90, top + 90),
        Point(left, top + 90),
    ]
    rings = [ring]
    if hole:
        rings.append(
            [
                Point(left + 25, top + 25),
                Point(left + 65, top + 25),
                Point(left + 65, top + 65),
                Point(left + 25, top + 65),
            ]
        )
    return Measurement(
        id=identity,
        image_id="handoff",
        fiber_group_id=None,
        mode="magic_segment",
        measurement_kind="area",
        polygon_px=ring,
        area_rings_px=rings,
        exact_area_px=6500 if hole else 8100,
    )


def frame(canvas):
    dpr = canvas.devicePixelRatioF()
    image = QImage(
        math.ceil(canvas.width() * dpr),
        math.ceil(canvas.height() * dpr),
        QImage.Format.Format_ARGB32_Premultiplied,
    )
    image.setDevicePixelRatio(dpr)
    image.fill(0)
    canvas.render(image)
    return image


def pixels(image):
    return (
        np.frombuffer(image.constBits(), np.uint8)
        .reshape(image.height(), image.bytesPerLine())[:, : image.width() * 4]
        .reshape(image.height(), image.width(), 4)
        .copy()
    )


@pytest.fixture
def scene(monkeypatch, desktop_application, request):
    monkeypatch.setenv("FDM_ENABLE_CANVAS_OVERLAY_CACHE", "1")
    exact_pool, preview_pool = DeferredPool(), DeferredPool()
    exact, preview = CanvasOverlayTileCache(
        thread_pool=exact_pool
    ), CanvasOverlayTileCache(thread_pool=preview_pool)
    draft_pool = DeferredPool()
    drafts = DraftPreviewCache(asynchronous=True)
    drafts._raster_cache._thread_pool = draft_pool
    drafts._raster_cache._isolated_worker = False
    monkeypatch.setattr(canvas_module, "canvas_overlay_tile_cache", exact)
    monkeypatch.setattr(canvas_module, "canvas_overlay_preview_cache", preview)
    monkeypatch.setattr(canvas_module, "draft_preview_cache", drafts)
    digital = getattr(request, "param", None) == "digital"
    canvas = DigitalSlideCanvas() if digital else DocumentCanvas()
    canvas.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
    canvas.resize(1280, 800)
    canvas.set_settings(
        AppSettings(
            area_measurement_label_style=MeasurementLabelStyleSettings(enabled=False),
            length_measurement_label_style=MeasurementLabelStyleSettings(enabled=False),
        )
    )
    doc = ImageDocument(
        id="handoff",
        path="synthetic.png",
        image_size=(4096, 2560),
        measurements=[
            area("edit", hole=True),
            area("neighbour", 440, 250),
            area("far", 1100, 650),
            Measurement(
                id="line",
                image_id="handoff",
                fiber_group_id=None,
                mode="manual",
                measurement_kind="line",
                line_px=Line(Point(800, 100), Point(1150, 100)),
            ),
        ],
    )
    doc.initialize_runtime_state()
    source = QImage(4096, 2560, QImage.Format.Format_RGB32)
    source.fill(QColor("white"))
    canvas.set_document(doc, source)
    canvas._zoom = 1.0
    canvas._pan = Point(0, 0)
    canvas.show()
    if digital:
        doc.document_kind = "digital_slide"
        # Keep coordinates global and mount a bounded field far from (0, 0).
        origin = Point(8192, 4096)
        doc.image_size = (16384, 8192)
        for measurement in doc.measurements:
            if measurement.measurement_kind == "area":
                rings = [
                    [Point(p.x + origin.x, p.y + origin.y) for p in ring]
                    for ring in measurement.area_rings_px
                ]
                measurement.replace_area_geometry(
                    polygon_px=rings[0],
                    area_rings_px=rings,
                    exact_area_px=measurement.exact_area_px,
                )
            else:
                line = measurement.line_px
                measurement.replace_line_geometry(
                    line_px=Line(
                        Point(line.start.x + origin.x, line.start.y + origin.y),
                        Point(line.end.x + origin.x, line.end.y + origin.y),
                    )
                )
        doc.mark_measurement_geometry_changed()
        canvas._slide_manifest = DigitalSlideManifest(
            version=1,
            width=16384,
            height=8192,
            viewport_width=1280,
            viewport_height=800,
            focus_levels=[0, 1],
        )
        canvas._browse_center = Point(origin.x + 640, origin.y + 400)
        canvas._pan = Point(-origin.x, -origin.y)
        canvas._viewport_origin = origin
        canvas._render_frame = DigitalSlideRenderFrame(
            request_id=1,
            purpose="display",
            source_rect=(origin.x, origin.y, 1280, 800),
            output_size_px=(1280, 800),
            focus_index=0,
            device_pixel_ratio=1,
            lod=0,
            image=source.copy(0, 0, 1280, 800),
            elapsed_ms=0,
            decoded_tiles=0,
            cache_hits=0,
            generation=canvas._view_generation,
        )
        canvas._pixel_work_enabled = True
    yield canvas, doc, exact, preview, exact_pool, preview_pool
    canvas.clear_document()
    canvas.close()
    exact_pool.complete(exact)
    preview_pool.complete(preview)
    draft_pool.complete(drafts._raster_cache)
    exact.clear()
    preview.clear()
    drafts._raster_cache.clear()


def overview(scene):
    canvas, _, _, preview, _, pool = scene
    canvas._request_scene_preview()
    while canvas._overlay_preview_measurements:
        canvas._overlay_preview_timer.stop()
        canvas._prepare_scene_preview()
    canvas._overlay_preview_timer.stop()
    pool.complete(preview)


def warm(scene):
    canvas, _, exact, _, pool, _ = scene
    frame(canvas)
    overview(scene)
    for key in canvas._visible_overlay_tile_keys(canvas._paint_context()):
        if not exact.contains(key) and not exact.is_pending(key):
            exact.request(canvas._build_overlay_tile_snapshot(key))
    pool.complete(exact)
    frame(canvas)
    drafts = canvas_module.draft_preview_cache._raster_cache
    drafts._thread_pool.complete(drafts)
    return frame(canvas)


def edit(scene, *, left=215, top=115, identity="edit"):
    canvas, doc, *_ = scene
    measurement = doc.get_measurement(identity)
    fresh = area("shape", left, top, hole=True)
    measurement.replace_area_geometry(
        polygon_px=fresh.polygon_px,
        area_rings_px=fresh.area_rings_px,
        exact_area_px=fresh.exact_area_px,
    )
    doc.mark_measurement_geometry_changed()
    doc.mark_session_dirty()
    canvas.notify_document_visual_changed()


@pytest.mark.parametrize("operation", ["add", "edit", "delete"])
@pytest.mark.parametrize("dpr", [1, 1.25, 1.5, 2])
def test_local_changes_never_replace_unaffected_precise_pixels(
    scene, monkeypatch, operation, dpr
):
    canvas, doc, exact, *_ = scene
    monkeypatch.setattr(canvas, "devicePixelRatioF", lambda: dpr)
    canvas._pan = Point(0.25, 0.25)
    before = pixels(warm(scene))
    if operation == "add":
        fresh = area("new", 250, 100)
        doc.insert_measurement_incremental(fresh, select=False)
        canvas.notify_document_visual_changed(added_measurement_ids=(fresh.id,))
    elif operation == "edit":
        edit(scene)
    else:
        doc.remove_measurement_incremental("edit")
        canvas.notify_document_visual_changed()
    geometry = doc.to_dict()
    assert canvas._overlay_presentation.regions
    with patch.object(
        canvas,
        "_draw_scene_preview",
        side_effect=AssertionError("local edit used coarse overview"),
    ):
        pending = pixels(frame(canvas))
        overview(scene)
        still_pending = pixels(frame(canvas))
    # Unchanged objects in the *same* affected tile must remain stable too.
    x, y = round(400 * dpr), round(230 * dpr)
    assert np.array_equal(before[y:], pending[y:])
    assert np.array_equal(before[y:], still_pending[y:])
    assert np.array_equal(before[:y, x:], pending[:y, x:])
    final = pixels(warm(scene))
    assert np.array_equal(before[y:], final[y:])
    assert not canvas._overlay_presentation.regions
    assert doc.to_dict() == geometry
    assert exact.stats().bytes <= exact.max_bytes


def test_cross_tile_edit_publishes_related_tiles_together(scene):
    canvas, doc, exact, _, pool, _ = scene
    before = pixels(warm(scene))
    edit(scene, left=485, top=470)
    keys = canvas._visible_overlay_tile_keys(canvas._paint_context())
    missing = [key for key in keys if not exact.contains(key)]
    assert len(missing) == 4
    for key in missing:
        exact.request(canvas._build_overlay_tile_snapshot(key))
    pool.complete(exact, count=1)
    assert np.array_equal(pixels(frame(canvas)), before)
    assert canvas._overlay_presentation.regions
    pool.complete(exact)
    after = pixels(frame(canvas))
    assert not np.array_equal(after, before)
    assert not canvas._overlay_presentation.regions
    assert not canvas._overlay_presentation._retired
    # Keep hole geometry and exact mask area independent from the display.
    assert doc.get_measurement("edit").exact_area_px == 6500


def test_new_body_is_visible_until_precise_handoff_even_after_overview_ready(scene):
    canvas, doc, *_ = scene
    warm(scene)
    fresh = area("new", 280, 110)
    doc.insert_measurement_incremental(fresh, select=False)
    canvas.notify_document_visual_changed(added_measurement_ids=(fresh.id,))
    assert frame(canvas).pixelColor(300, 140).red() < 250
    overview(scene)
    assert fresh.id in canvas._overlay_accepted_ids
    assert frame(canvas).pixelColor(300, 140).red() < 250
    warm(scene)
    assert fresh.id not in canvas._overlay_accepted_ids
    assert frame(canvas).pixelColor(300, 140).red() < 250


@pytest.mark.parametrize("kind", ["area", "line"])
@pytest.mark.parametrize("dpr", [1, 1.25, 1.5, 2])
def test_selected_edit_does_not_leave_old_geometry_beneath_new_body(
    scene, monkeypatch, kind, dpr
):
    canvas, doc, *_ = scene
    monkeypatch.setattr(canvas, "devicePixelRatioF", lambda: dpr)
    identity = "edit" if kind == "area" else "line"
    doc.select_measurement(identity)
    warm(scene)
    if kind == "area":
        canvas._dragging_area_handle = (identity, "center", None, None)
        canvas._drag_area_preview_offset = Point(160, 10)
        before_release = frame(canvas)
        canvas._clear_area_drag_state()
        apply_edit = lambda *_: edit(scene, left=260, top=110)
        old_point, new_point = (110, 110), (270, 120)
    else:
        preview_line = Line(Point(800, 180), Point(1150, 180))
        canvas._dragging_handle = (identity, "start")
        canvas._drag_preview_line = preview_line
        before_release = frame(canvas)
        canvas._dragging_handle = None
        canvas._drag_preview_line = None

        def apply_edit(*_):
            doc.get_measurement(identity).replace_line_geometry(line_px=preview_line)
            doc.mark_measurement_geometry_changed()
            doc.mark_session_dirty()
            canvas.notify_document_visual_changed()

        old_point, new_point = (950, 100), (950, 180)
    canvas.measurementEdited.connect(apply_edit)
    canvas._emit_measurement_edit(doc.id, identity, None)
    assert canvas._overlay_held_gesture is not None
    old_point = tuple(round(value * dpr) for value in old_point)
    new_point = tuple(round(value * dpr) for value in new_point)
    pending = frame(canvas)
    assert np.array_equal(pixels(before_release), pixels(pending))
    assert pending.pixelColor(*old_point) == QColor("white")
    assert pending.pixelColor(*new_point) != QColor("white")
    # A second paint uses the existing clean background/body instead of drawing
    # neighbours again while waiting for the exact tile worker.
    with patch.object(
        canvas,
        "_draw_measurements_direct",
        side_effect=AssertionError("neighbours rebuilt"),
    ):
        frame(canvas)
    final = warm(scene)
    assert final.pixelColor(*old_point) == QColor("white")


def test_property_edit_retains_old_complete_selection_without_raw_repaint(scene):
    canvas, doc, *_ = scene
    doc.select_measurement("edit")
    before = pixels(warm(scene))
    edit(scene)
    with (
        patch.object(
            canvas,
            "_draw_measurements_direct",
            side_effect=AssertionError("RAW neighbours in result installation"),
        ),
        patch.object(
            canvas_module,
            "draw_area_measurement",
            side_effect=AssertionError("RAW body in result installation"),
        ),
    ):
        assert np.array_equal(before, pixels(frame(canvas)))
    assert not np.array_equal(before, pixels(warm(scene)))


def test_rapid_edit_undo_and_late_worker_only_publish_latest_geometry(scene):
    canvas, doc, exact, _, pool, _ = scene
    original = doc.get_measurement("edit").to_dict()
    before = pixels(warm(scene))
    edit(scene, left=480, top=470)
    stale = [
        k
        for k in canvas._visible_overlay_tile_keys(canvas._paint_context())
        if not exact.contains(k)
    ]
    for key in stale:
        exact.request(canvas._build_overlay_tile_snapshot(key))
    restored = Measurement.from_dict(original)
    doc.get_measurement("edit").replace_area_geometry(
        polygon_px=restored.polygon_px,
        area_rings_px=restored.area_rings_px,
        exact_area_px=restored.exact_area_px,
    )
    doc.mark_measurement_geometry_changed()
    doc.mark_session_dirty()
    canvas.notify_document_visual_changed()
    pool.complete(exact)
    assert all(not exact.contains(key) for key in stale)
    assert np.array_equal(pixels(frame(canvas)), before)
    assert np.array_equal(pixels(warm(scene)), before)


def test_switching_document_cannot_reuse_an_old_front(scene):
    canvas, doc, exact, _, pool, _ = scene
    warm(scene)
    edit(scene)
    for key in canvas._visible_overlay_tile_keys(canvas._paint_context()):
        if not exact.contains(key):
            exact.request(canvas._build_overlay_tile_snapshot(key))
    source = QImage(4096, 2560, QImage.Format.Format_RGB32)
    source.fill(QColor("white"))
    other = ImageDocument(id=doc.id, path=doc.path, image_size=doc.image_size)
    canvas.set_document(other, source)
    assert not canvas._overlay_presentation.front
    pool.complete(exact)
    assert frame(canvas).pixelColor(110, 110) == QColor("white")


def test_global_budget_counts_retained_old_epochs(scene):
    canvas, _, exact, *_ = scene
    warm(scene)
    old_front = set(canvas._overlay_presentation.front.values())
    initial_bytes = exact.stats().bytes
    edit(scene)
    assert all(exact.contains(key) for key in old_front)
    assert exact.stats().bytes == initial_bytes
    warm(scene)
    current_front = set(canvas._overlay_presentation.front.values())
    assert all(not exact.contains(key) for key in old_front - current_front)
    assert exact.stats().bytes <= exact.max_bytes


def test_failed_precise_region_uses_current_fallback_without_stale_deleted_object(
    scene,
):
    canvas, doc, exact, *_ = scene
    before = pixels(warm(scene))
    doc.remove_measurement_incremental("edit")
    canvas.notify_document_visual_changed()
    key = next(
        k
        for k in canvas._visible_overlay_tile_keys(canvas._paint_context())
        if not exact.contains(k)
    )
    canvas._on_overlay_tile_failed(key, "injected failure")
    overview(scene)
    pending = frame(canvas)
    assert pending.pixelColor(110, 110) == QColor("white")
    assert np.array_equal(before[600:], pixels(pending)[600:])
    assert not canvas._overlay_presentation.regions


def test_zoom_or_phase_change_cannot_reuse_old_exact_front(scene):
    canvas, _, *_ = scene
    warm(scene)
    edit(scene)
    previous = set(canvas._overlay_presentation.front.values())
    canvas._zoom = 2.5
    frame(canvas)
    assert not previous.intersection(canvas._overlay_presentation.front.values())
    assert not canvas._overlay_presentation.regions


def test_unrelated_regions_publish_without_waiting_for_each_other(scene):
    canvas, _, exact, _, pool, _ = scene
    warm(scene)
    edit(scene)
    edit(scene, identity="far", left=1100, top=690)
    assert len(canvas._overlay_presentation.regions) == 2
    region = next(
        r for r in canvas._overlay_presentation.regions if "edit" in r.measurement_ids
    )
    for key in canvas._visible_overlay_tile_keys(canvas._paint_context()):
        if (key.tile_x, key.tile_y) in region.coordinates:
            exact.request(canvas._build_overlay_tile_snapshot(key))
    pool.complete(exact)
    frame(canvas)
    assert canvas._overlay_presentation.pending_ids == {"far"}


@pytest.mark.parametrize("scene", ["digital"], indirect=True)
def test_digital_slide_edit_keeps_global_coordinates_and_precise_neighbours(scene):
    canvas, doc, exact, *_ = scene
    before = pixels(warm(scene))
    edit(scene, left=8192 + 215, top=4096 + 115)
    assert canvas._overlay_presentation.regions
    with patch.object(
        canvas,
        "_draw_scene_preview",
        side_effect=AssertionError("slide edit used overview"),
    ):
        assert np.array_equal(before, pixels(frame(canvas)))
    after = warm(scene)
    assert after.pixelColor(110, 110) == QColor("white")
    assert after.pixelColor(225, 125) != QColor("white")
    assert np.array_equal(before[600:], pixels(after)[600:])
    assert doc.get_measurement("edit").polygon_px[0] == Point(8407, 4211)


@pytest.mark.parametrize("scene", ["digital"], indirect=True)
def test_digital_clean_background_respects_callers_local_clip_and_pixel_version(scene):
    from PySide6.QtGui import QPainter

    canvas, *_ = scene
    image = QImage(canvas.size(), QImage.Format.Format_RGB32)
    image.fill(QColor("magenta"))
    painter = QPainter(image)
    try:
        painter.setClipRect(QRectF(30, 30, 40, 40))
        canvas._draw_base_image(painter)
    finally:
        painter.end()
    assert image.pixelColor(40, 40) == QColor("white")
    assert image.pixelColor(200, 200) == QColor("magenta")
    signature = canvas._overlay_background_signature()
    replacement = QImage(canvas._render_frame.image)
    replacement.fill(QColor("gray"))
    canvas._render_frame = replace(canvas._render_frame, image=replacement)
    assert canvas._overlay_background_signature() != signature


@pytest.mark.parametrize("scene", ["digital"], indirect=True)
def test_digital_focus_change_drops_pixel_patch_without_discarding_overlay_geometry(
    scene,
):
    canvas, doc, *_ = scene
    doc.select_measurement("edit")
    warm(scene)
    canvas._dragging_area_handle = ("edit", "center", None, None)
    canvas._drag_area_preview_offset = Point(160, 10)
    frame(canvas)
    canvas._clear_area_drag_state()
    canvas.measurementEdited.connect(
        lambda *_: edit(scene, left=8192 + 260, top=4096 + 110)
    )
    canvas._emit_measurement_edit(doc.id, "edit", None)
    assert canvas._overlay_held_gesture is not None
    canvas._focus_index = 1
    replacement = QImage(canvas._render_frame.image)
    replacement.fill(QColor("#777777"))
    canvas._render_frame = replace(
        canvas._render_frame, focus_index=1, image=replacement
    )
    current = frame(canvas)
    # The retained clean patch had white pixels in this gap; it must not paint
    # those old-focus pixels over the new gray field.
    assert current.pixelColor(225, 130) == QColor("#777777")


def test_cancelled_gesture_is_not_revived_by_a_later_property_edit(scene):
    canvas, doc, *_ = scene
    doc.select_measurement("edit")
    warm(scene)
    canvas._dragging_area_handle = ("edit", "center", None, None)
    canvas._drag_area_preview_offset = Point(160, 10)
    frame(canvas)
    canvas._clear_area_drag_state()
    edit(scene, left=360, top=110)
    assert canvas._overlay_held_gesture is None


def test_ready_region_waits_for_a_paint_covering_all_related_tiles(scene):
    from PySide6.QtGui import QPainter, QRegion

    canvas, _, exact, _, pool, _ = scene
    warm(scene)
    edit(scene, left=485, top=470)
    previous = dict(canvas._overlay_presentation.front)
    for key in canvas._visible_overlay_tile_keys(canvas._paint_context()):
        if not exact.contains(key):
            exact.request(canvas._build_overlay_tile_snapshot(key))
    pool.complete(exact)
    # Worker completion only requests a paint; it must not advance display
    # ownership before the related tiles have actually been presented.
    assert canvas._overlay_presentation.front == previous
    image = QImage(canvas.size(), QImage.Format.Format_ARGB32_Premultiplied)
    painter = QPainter(image)
    canvas._overlay_paint_region = QRegion(0, 0, 100, 100)
    try:
        canvas._draw_measurement_overlay_tiles(
            painter, canvas._paint_context(QRectF(0, 0, 100, 100))
        )
    finally:
        painter.end()
        canvas._overlay_paint_region = None
    assert canvas._overlay_presentation.front == previous
    assert canvas._overlay_presentation.regions
    frame(canvas)
    assert not canvas._overlay_presentation.regions


@pytest.mark.parametrize("kind", ["line", "polyline", "count"])
def test_mixed_append_and_undo_preserve_far_pixels_and_final_draw_order(scene, kind):
    canvas, doc, *_ = scene
    doc.measurements.append(
        Measurement(
            id="overlapping-count",
            image_id=doc.id,
            fiber_group_id=None,
            mode="count",
            measurement_kind="count",
            point_px=Point(275, 160),
        )
    )
    doc.mark_measurement_geometry_changed()
    doc.mark_session_dirty()
    canvas.notify_document_visual_changed()
    before = pixels(warm(scene))
    added = Measurement(
        id="added-mixed",
        image_id=doc.id,
        fiber_group_id=None,
        mode="manual",
        measurement_kind=kind,
        line_px=Line(Point(240, 160), Point(350, 160)) if kind == "line" else None,
        polyline_px=(
            [Point(240, 160), Point(270, 200), Point(350, 160)]
            if kind == "polyline"
            else []
        ),
        point_px=Point(277, 160) if kind == "count" else None,
    )
    doc.insert_measurement_incremental(added, select=False)
    canvas.notify_document_visual_changed(added_measurement_ids=(added.id,))
    pending = pixels(frame(canvas))
    assert np.array_equal(before[550:], pending[550:])
    final = pixels(warm(scene))
    with patch.object(canvas, "_overlay_cache_enabled", return_value=False):
        direct = pixels(frame(canvas))
    assert np.array_equal(final[120:240, 220:380], direct[120:240, 220:380])
    doc.remove_measurement_incremental(added.id)
    canvas.notify_document_visual_changed()
    restored = pixels(warm(scene))
    assert np.array_equal(restored, before)


def test_precise_composition_is_reused_without_replaying_mixed_vector_tile(scene):
    from fdm.ui.screen_layer_cache import screen_layer_cache

    canvas, doc, exact, *_ = scene
    doc.measurements.append(
        Measurement(
            id="count",
            image_id=doc.id,
            fiber_group_id=None,
            mode="count",
            measurement_kind="count",
            point_px=Point(150, 150),
        )
    )
    doc.mark_measurement_geometry_changed()
    doc.mark_session_dirty()
    canvas.notify_document_visual_changed()
    before = pixels(warm(scene))
    assert any(
        exact.get_payload(k)[1] is not None
        for k in canvas._overlay_presentation.front.values()
    )
    builds = screen_layer_cache.builds
    edit(scene)
    assert np.array_equal(before, pixels(frame(canvas)))
    assert screen_layer_cache.builds == builds
    warm(scene)
    # Full invalidation can reuse epoch zero; it must clear opaque compositions
    # as well as the underlying transparent/command tiles.
    canvas.set_screen_measurement_labels("hidden")
    after = pixels(warm(scene))
    with patch.object(canvas, "_overlay_cache_enabled", return_value=False):
        direct = pixels(frame(canvas))
    assert np.array_equal(after[50:400, 50:400], direct[50:400, 50:400])


def test_precise_edge_tile_keeps_constructions_when_panned_into_view(scene):
    from fdm.construction_geometry import ConstructionEntity, LineDefinition, LineExtent

    canvas, doc, *_ = scene
    doc.add_construction_entity(
        ConstructionEntity(
            id="guide",
            name="边缘辅助线",
            definition=LineDefinition(
                Point(1100, 180), Point(1200, 180), LineExtent.INFINITE
            ),
        ),
        select=False,
    )
    canvas.notify_document_visual_changed()
    warm(scene)
    canvas._pan = Point(-256, 0)
    cached = pixels(frame(canvas))
    with patch.object(canvas, "_overlay_cache_enabled", return_value=False):
        direct = pixels(frame(canvas))
    # The visible part of a cached edge tile changes, without changing its
    # image-space coordinate. Its underlay must cover the complete tile.
    assert np.array_equal(cached[170:190, 1030:1200], direct[170:190, 1030:1200])


def test_evicted_front_falls_back_locally_then_finishes_the_latest_edit(scene):
    canvas, doc, exact, *_ = scene
    before = pixels(warm(scene))
    edit(scene)
    retired = set(canvas._overlay_presentation._retired)
    exact.discard_display_fronts(retired)
    overview(scene)
    pending = pixels(frame(canvas))
    assert np.array_equal(before[600:], pending[600:])
    final = warm(scene)
    assert final.pixelColor(110, 110) == QColor("white")
    assert final.pixelColor(225, 125) != QColor("white")
    assert not canvas._overlay_presentation.regions
    assert exact.stats().bytes <= exact.max_bytes


@pytest.mark.parametrize("scene", [None, "digital"], indirect=True)
@pytest.mark.parametrize("tool", ["polygon_area", "freehand_area"])
def test_selected_subtraction_keeps_feedback_until_the_precise_result(scene, tool):
    canvas, doc, *_ = scene
    doc.select_measurement("edit")
    canvas.set_tool_mode(tool)
    canvas.set_area_edit_operation_mode("subtract")
    warm(scene)
    origin = canvas.mounted_image_origin()
    points = [
        Point(x + origin.x, y + origin.y)
        for x, y in [(90, 95), (120, 95), (120, 195), (90, 195)]
    ]
    canvas._drawing_polygon_points = points
    before = pixels(frame(canvas))

    def apply_result(document_id, identity, payload):
        assert document_id == doc.id
        measurement = doc.get_measurement(identity)
        measurement.replace_area_geometry(
            polygon_px=payload["polygon_px"],
            area_rings_px=payload["area_rings_px"],
            exact_area_px=payload["exact_area_px"],
        )
        doc.mark_measurement_geometry_changed()
        doc.mark_session_dirty()
        canvas.notify_document_visual_changed()

    canvas.measurementEdited.connect(apply_result)
    assert canvas._complete_area_subtract_polygon(points)
    assert canvas._overlay_held_gesture is not None
    assert np.array_equal(before, pixels(frame(canvas)))
    final = warm(scene)
    assert final.pixelColor(110, 110) == QColor("white")
    assert doc.get_measurement("edit").exact_area_px < 6500
    assert len(doc.get_measurement("edit").area_rings_px) == 2


def test_removing_last_area_releases_handoff_before_returning_to_direct_rendering(
    scene, monkeypatch
):
    canvas, doc, *_ = scene
    warm(scene)
    # Use the production admission rule while keeping Qt's offscreen backend.
    monkeypatch.delenv("FDM_ENABLE_CANVAS_OVERLAY_CACHE")
    monkeypatch.delenv("QT_QPA_PLATFORM")
    assert canvas._overlay_cache_enabled()
    for identity in ("edit", "neighbour", "far"):
        doc.remove_measurement_incremental(identity)
    canvas.notify_document_visual_changed()
    assert canvas._overlay_presentation.regions
    assert not canvas._overlay_cache_enabled()
    frame(canvas)
    assert not canvas._overlay_presentation.front
    assert not canvas._overlay_presentation.regions
    assert not canvas._overlay_tile_queue
    doc.select_measurement("line")
    canvas._dragging_handle = ("line", "start")
    canvas._drag_preview_line = Line(Point(800, 180), Point(1150, 180))
    with patch.object(
        canvas_module.screen_layer_cache,
        "draw",
        side_effect=AssertionError("small direct preview allocated a raster"),
    ):
        assert frame(canvas).pixelColor(950, 180) != QColor("white")


@pytest.mark.parametrize("held_edit", [False, True])
def test_roi_changes_invalidate_cached_underlays_without_scanning_geometry(
    scene, held_edit
):
    from fdm.project_roi import ProjectRoi, RectangleRoiGeometry

    canvas, doc, *_ = scene
    roi = ProjectRoi(
        id="roi",
        document_id=doc.id,
        name="测量背景 ROI",
        geometry=RectangleRoiGeometry(130, 130, 30, 30),
        color="#00FF00",
    )
    canvas.set_project_rois([roi])
    doc.select_measurement("edit")
    before = warm(scene)
    if held_edit:
        canvas._dragging_area_handle = ("edit", "center", None, None)
        canvas._drag_area_preview_offset = Point(160, 10)
        frame(canvas)
        canvas._clear_area_drag_state()
        canvas.measurementEdited.connect(lambda *_: edit(scene, left=260, top=110))
        canvas._emit_measurement_edit(doc.id, "edit", None)
        assert canvas._overlay_held_gesture is not None
    canvas.set_project_rois([replace(roi, color="#FF0000", revision=roi.revision + 1)])
    with patch.object(
        ProjectRoi,
        "__repr__",
        side_effect=AssertionError("ROI geometry serialized during paint"),
    ):
        changed = frame(canvas)
    assert before.pixelColor(140, 140) != changed.pixelColor(140, 140)
    with patch.object(canvas, "_overlay_cache_enabled", return_value=False):
        direct = frame(canvas)
    assert changed.pixelColor(140, 140) == direct.pixelColor(140, 140)
