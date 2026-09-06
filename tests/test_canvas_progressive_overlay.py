from unittest.mock import patch
import math
import pytest
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QImage, QPainter
from fdm.geometry import Point
from fdm.models import ImageDocument, Measurement
from fdm.settings import AppSettings, MeasurementLabelStyleSettings
from fdm.ui import canvas as canvas_module
from fdm.ui.canvas import DocumentCanvas, MeasurementSceneIndex
from fdm.ui.canvas_overlay_cache import CanvasOverlayTileCache


class InlinePool:
    def start(self, runnable):
        runnable.run()


class DeferredPool:
    def __init__(self):
        self.runnables = []

    def start(self, runnable):
        self.runnables.append(runnable)

    def complete(self, cache):
        for runnable in self.runnables:
            runnable.run()
        self.runnables.clear()
        cache._drain_completions()


def area(identity, x=60, y=60, radius=30):
    ring = [
        Point(
            x + radius * math.cos(i * 2 * math.pi / 1000),
            y + radius * math.sin(i * 2 * math.pi / 1000),
        )
        for i in range(1000)
    ]
    return Measurement(
        id=identity,
        image_id="scene",
        fiber_group_id=None,
        mode="magic_segment",
        measurement_kind="area",
        polygon_px=ring,
        area_rings_px=[ring],
        exact_area_px=radius * radius * math.pi,
    )


def document():
    return ImageDocument(
        id="scene",
        path="scene.png",
        image_size=(1024, 768),
        measurements=[area(str(i), 60 + (i % 5) * 160, 60 + (i // 5) * 160) for i in range(20)],
    )


def frame(canvas):
    image = QImage(canvas.size(), QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(0)
    canvas.render(image)
    return image


@pytest.fixture
def scene(monkeypatch, desktop_application):
    monkeypatch.setenv("FDM_ENABLE_CANVAS_OVERLAY_CACHE", "1")
    exact_pool, preview_pool = DeferredPool(), DeferredPool()
    exact = CanvasOverlayTileCache(thread_pool=exact_pool)
    preview = CanvasOverlayTileCache(thread_pool=preview_pool, max_bytes=64 * 1024 * 1024)
    monkeypatch.setattr(canvas_module, "canvas_overlay_tile_cache", exact)
    monkeypatch.setattr(canvas_module, "canvas_overlay_preview_cache", preview)
    canvas = DocumentCanvas()
    canvas.resize(512, 384)
    canvas.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
    canvas.set_settings(
        AppSettings(area_measurement_label_style=MeasurementLabelStyleSettings(enabled=False))
    )
    doc = document()
    source = QImage(1024, 768, QImage.Format.Format_RGB32)
    source.fill(0xFFFFFFFF)
    canvas.set_document(doc, source)
    canvas._pan = Point(0, 0)
    canvas._zoom = 1
    canvas.show()
    yield canvas, doc, exact, preview, exact_pool, preview_pool
    canvas.clear_document()
    canvas.close()
    exact_pool.complete(exact)
    preview_pool.complete(preview)
    exact.clear()
    preview.clear()


def prepare_preview(canvas):
    canvas._sync_overlay_visual_state()
    key = canvas._request_scene_preview()
    # Run bounded preparation steps without running tile jobs.
    while canvas._overlay_preview_measurements:
        canvas._overlay_preview_timer.stop()
        canvas._prepare_scene_preview()
    return key


def test_visible_cold_frame_never_draws_entire_uncached_scene(scene):
    canvas, doc, exact, preview, exact_pool, preview_pool = scene
    calls = []
    original = canvas._draw_measurements_direct

    def track(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    with patch.object(canvas, "_draw_measurements_direct", new=track):
        frame(canvas)
    assert not calls
    assert canvas._overlay_preview_frames == 1
    assert not exact.stats().completed
    key = prepare_preview(canvas)
    preview_pool.complete(preview)
    assert preview.contains(key)
    image = frame(canvas)
    assert image.pixelColor(60, 60).red() < 250
    assert image.pixelColor(120, 120).red() == 255
    before = [measurement.to_dict() for measurement in doc.measurements]
    canvas._zoom = 1.5
    canvas._pan = Point(-20, -20)
    calls.clear()
    with patch.object(canvas, "_draw_measurements_direct", new=track):
        scaled = frame(canvas)
    assert not calls
    assert scaled.pixelColor(70, 70).red() < 250
    assert before == [measurement.to_dict() for measurement in doc.measurements]
    assert canvas._measurement_index().query_point(Point(60, 60), tolerance=1)[0].id == "0"


def test_late_preview_after_edit_or_document_switch_cannot_be_published(scene):
    canvas, doc, exact, preview, exact_pool, preview_pool = scene
    stale = prepare_preview(canvas)
    doc.measurements[0].replace_area_geometry(
        polygon_px=[Point(400, 400), Point(450, 400), Point(450, 450)]
    )
    doc.mark_measurement_geometry_changed()
    doc.mark_session_dirty()
    canvas.notify_document_visual_changed()
    current = prepare_preview(canvas)
    assert stale != current
    preview_pool.complete(preview)
    assert not preview.contains(stale)
    assert canvas._overlay_preview_last == current
    other = document()
    source = QImage(1024, 768, QImage.Format.Format_RGB32)
    source.fill(0xFFFFFFFF)
    canvas.set_document(other, source)
    assert canvas._scene_preview_key().document_token != current.document_token
    image = frame(canvas)
    # The previous document's object at (60, 60) cannot leak into this loading frame.
    assert image.pixelColor(60, 60).red() == 255


def test_incremental_index_preserves_unmodified_cells_and_exact_hit_testing():
    doc = document()
    index = MeasurementSceneIndex(doc.measurements)
    unchanged = index._entries["1"]
    changed = doc.measurements[0]
    changed.replace_area_geometry(polygon_px=[Point(500, 500), Point(560, 500), Point(560, 560)])
    original = MeasurementSceneIndex._measurement_bounds

    def bounds(measurement):
        assert measurement is changed
        return original(measurement)

    with patch.object(MeasurementSceneIndex, "_measurement_bounds", new=staticmethod(bounds)):
        index.sync(doc.measurements)
    assert index._entries["1"] is unchanged
    assert "0" not in [item.id for item in index.query_point(Point(60, 60), tolerance=1)]
    assert "0" in [item.id for item in index.query_point(Point(530, 530), tolerance=1)]
    doc.measurements.pop(0)
    index.sync(doc.measurements)
    assert "0" not in index._entries
    assert index.document_order("1") == 0


def test_new_measurement_body_is_visible_before_its_background_tiles(scene):
    canvas, doc, exact, preview, exact_pool, preview_pool = scene
    prepare_preview(canvas)
    preview_pool.complete(preview)
    fresh = area("new", 120, 120, 20)
    doc.insert_measurement_incremental(fresh)
    canvas.notify_document_visual_changed(added_measurement_ids=(fresh.id,))
    image = frame(canvas)
    assert image.pixelColor(128, 120).red() < 250
    assert not preview.contains(canvas._scene_preview_key())
    from fdm.ui.screen_layer_cache import screen_layer_cache

    builds = screen_layer_cache.builds
    canvas._panning = True
    for _ in range(5):
        canvas._pan = Point(canvas._pan.x + 1, canvas._pan.y + 1)
        frame(canvas)
    # First selection raster may be built once; stable body never repeats.
    assert screen_layer_cache.builds <= builds + 1
    canvas._panning = False


def test_cold_selected_area_culling_never_computes_exact_centroid(scene):
    from fdm.area_display import area_derived_geometry_service

    canvas, doc, *_ = scene
    canvas.set_settings(
        AppSettings(area_measurement_label_style=MeasurementLabelStyleSettings(enabled=True))
    )
    selected = doc.measurements[0]
    doc.select_measurement(selected.id)
    with patch.object(
        area_derived_geometry_service,
        "centroid",
        side_effect=AssertionError("exact centroid computed during tile culling"),
    ):
        # A far-away tile exercises the selected-object supplement to the index.
        distant, _ = canvas._measurement_render_inputs(QRectF(900, 700, 100, 50))
        assert selected not in distant
        # Conservative bounds still retain objects whose labels can be visible.
        nearby, _ = canvas._measurement_render_inputs(QRectF(25, 5, 70, 20))
        assert selected in nearby


def test_control_point_edit_reuses_clean_neighbour_background(scene):
    canvas, doc, exact, preview, exact_pool, preview_pool = scene
    doc.select_measurement(doc.measurements[0].id)
    canvas._dragging_area_handle = (doc.measurements[0].id, 0, 0)
    surface = QImage(canvas.size(), QImage.Format.Format_ARGB32_Premultiplied)
    painter = QPainter(surface)
    try:
        canvas._redraw_selected_measurement_background(painter, canvas._paint_context())
        with patch.object(
            canvas,
            "_draw_measurements_direct",
            side_effect=AssertionError("stable neighbours redrawn"),
        ):
            for _ in range(5):
                canvas._redraw_selected_measurement_background(painter, canvas._paint_context())
    finally:
        painter.end()
        canvas._dragging_area_handle = None


def test_confirmed_magic_draft_hands_off_without_synchronous_area_render(
    scene, desktop_application
):
    import time
    from fdm.settings import MagicSegmentToolMode
    from fdm.ui.draft_preview_cache import draft_preview_cache

    canvas, doc, exact, preview, exact_pool, preview_pool = scene
    canvas.set_tool_mode(MagicSegmentToolMode.STANDARD)
    fresh = area("accepted-draft", 120, 120, 20)
    session = canvas._magic_segment
    session.primary_polygon = fresh.polygon_px
    session.primary_rings = fresh.area_rings_px
    surface = QImage(canvas.size(), QImage.Format.Format_ARGB32_Premultiplied)
    painter = QPainter(surface)
    canvas._draw_magic_segment_preview(painter)
    painter.end()
    deadline = time.perf_counter() + 5
    while draft_preview_cache._requests and time.perf_counter() < deadline:
        desktop_application.processEvents()
        time.sleep(0.001)
    transfer = canvas._preserve_magic_display_preview()
    assert transfer
    canvas.clear_magic_segment_session()
    doc.insert_measurement_incremental(fresh)
    canvas._overlay_accepted_previews[fresh.id] = transfer
    canvas.notify_document_visual_changed(added_measurement_ids=(fresh.id,))
    with (
        patch.object(
            canvas_module, "draw_measurements", side_effect=AssertionError("cold RAW draw")
        ),
        patch.object(
            canvas_module,
            "draw_area_measurement",
            side_effect=AssertionError("cold selected RAW draw"),
        ),
    ):
        image = frame(canvas)
        assert image.pixelColor(128, 120).red() < 250
        # Cancelling the following draft must preserve this accepted image.
        canvas.clear_magic_segment_session()
        image = frame(canvas)
        assert image.pixelColor(128, 120).red() < 250


def test_complete_preview_preserves_overlapping_count_label_order(scene):
    import numpy as np
    from PySide6.QtCore import QRectF
    from PySide6.QtGui import QColor

    canvas, doc, exact, preview, exact_pool, preview_pool = scene
    doc.measurements = [
        Measurement(
            str(index),
            doc.id,
            None,
            "count",
            measurement_kind="count",
            point_px=Point(130 + index * 3, 130 - index * 2),
        )
        for index in range(2)
    ]
    doc.mark_measurement_geometry_changed()
    doc.mark_session_dirty()
    canvas.set_settings(AppSettings(show_count_numbers=True))
    canvas.notify_document_visual_changed()
    prepare_preview(canvas)
    preview_pool.complete(preview)
    actual = frame(canvas)
    expected = QImage(canvas.size(), QImage.Format.Format_ARGB32_Premultiplied)
    expected.fill(QColor("white"))
    painter = QPainter(expected)
    try:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        canvas._draw_measurements_direct(
            painter,
            image_rect=QRectF(0, 0, 512, 384),
            image_to_output=canvas.image_to_widget,
            use_sprite_cache=True,
            render_selected_state=False,
        )
    finally:
        painter.end()

    def pixels(image):
        return (
            np.frombuffer(image.constBits(), np.uint8)
            .reshape(image.height(), image.width(), 4)[90:160, 90:160]
            .astype(int)
        )

    assert np.abs(pixels(actual) - pixels(expected)).max() <= 1
