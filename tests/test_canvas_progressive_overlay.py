from unittest.mock import patch
from contextlib import contextmanager
from dataclasses import replace
import math
import cv2
import numpy as np
import pytest
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPainterPathStroker, QPen
from fdm.geometry import Line, Point
from fdm.history import DocumentHistory
from fdm.measurement_annotation import MeasurementAnnotationLayout
from fdm.models import ImageDocument, Measurement, OverlayTextSizeSpace
from fdm.settings import AppSettings, MeasurementLabelStyleSettings
from fdm.ui import canvas as canvas_module
from fdm.ui.canvas import CanvasZoomMode, DocumentCanvas, MeasurementSceneIndex
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
    dpr = canvas.devicePixelRatioF()
    image = QImage(math.ceil(canvas.width() * dpr), math.ceil(canvas.height() * dpr),
                   QImage.Format.Format_ARGB32_Premultiplied)
    image.setDevicePixelRatio(dpr)
    image.fill(0)
    canvas.render(image)
    return image


def frame_pixels(image):
    return np.frombuffer(image.constBits(), np.uint8).reshape(
        image.height(), image.bytesPerLine()
    )[:, :image.width() * 4].reshape(image.height(), image.width(), 4).copy()


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


def compact_area(identity, mode="polygon_area"):
    # A small, valid contour stays well below the old 10,000-vertex gate,
    # including in the 100-object boundary case.
    ring = [Point(40, 40), Point(80, 40), Point(80, 80), Point(40, 80)]
    return Measurement(
        id=identity,
        image_id="scene",
        fiber_group_id=None,
        mode=mode,
        measurement_kind="area",
        polygon_px=ring,
        area_rings_px=[ring],
        exact_area_px=1600,
    )


def use_default_cache_policy(scene, monkeypatch, measurements):
    # Keep the already-created offscreen QApplication, but exercise the same
    # automatic admission branch as the shipped desktop app, without forcing it.
    monkeypatch.delenv("FDM_ENABLE_CANVAS_OVERLAY_CACHE", raising=False)
    monkeypatch.delenv("FDM_DISABLE_CANVAS_OVERLAY_CACHE", raising=False)
    monkeypatch.delenv("QT_QPA_PLATFORM", raising=False)
    canvas, doc, *_ = scene
    doc.measurements = measurements
    doc.mark_measurement_geometry_changed()
    doc.mark_session_dirty()
    canvas.notify_document_visual_changed()


@contextmanager
def record_direct_draws(canvas):
    calls = []
    original = canvas._draw_measurements_direct

    def draw(*args, **kwargs):
        # Do not retain painter arguments: paintEvent owns their lifetime.
        calls.append(True)
        return original(*args, **kwargs)

    with patch.object(canvas, "_draw_measurements_direct", new=draw):
        yield calls


@pytest.mark.parametrize("count", [1, 20, 60, 63, 64, 100])
@pytest.mark.parametrize("mode", ["polygon_area", "magic_segment"])
def test_default_area_rendering_never_reverts_to_whole_scene_vectors(
    scene, monkeypatch, count, mode
):
    canvas, doc, exact, preview, exact_pool, preview_pool = scene
    use_default_cache_policy(
        scene, monkeypatch, [compact_area(str(i), mode) for i in range(count)]
    )
    geometry = [item.to_dict() for item in doc.measurements]
    with record_direct_draws(canvas) as direct:
        # A cold frame must schedule display work instead of painting every RAW
        # contour on the UI thread, even for the first ordinary polygon.
        frame(canvas)
        assert not direct
        prepare_preview(canvas)
        preview_pool.complete(preview)
        assert frame(canvas).pixelColor(60, 60).red() < 250

        for key in canvas._visible_overlay_tile_keys(canvas._paint_context()):
            assert exact.request(canvas._build_overlay_tile_snapshot(key))
        exact_pool.complete(exact)
        assert frame(canvas).pixelColor(60, 60).red() < 250
        canvas._panning = True
        try:
            for offset in (1, 20, -20):
                canvas._pan = Point(offset, 0)
                assert frame(canvas).pixelColor(60 + offset, 60).red() < 250
        finally:
            canvas._panning = False
        assert not direct
    assert [item.to_dict() for item in doc.measurements] == geometry


@pytest.mark.parametrize("initial_count", [0, 63])
def test_area_cache_survives_first_insert_undo_redo_and_reopen(
    scene, monkeypatch, initial_count
):
    canvas, doc, *_ = scene
    use_default_cache_policy(
        scene, monkeypatch, [compact_area(str(i)) for i in range(initial_count)]
    )
    doc.history = DocumentHistory(owner=doc)
    frame(canvas)
    before_stamp = doc.state_stamp
    added = compact_area("added")
    doc.insert_measurement_incremental(added, select=False)
    doc.history.push_measurement_insert(
        added,
        index=initial_count,
        previous_selection=(None, None, None),
        before_stamp=before_stamp,
        after_stamp=doc.state_stamp,
    )
    canvas.notify_document_visual_changed(added_measurement_ids=(added.id,))
    assert canvas._overlay_cache_enabled()
    # The first accepted polygon is visible before either background cache is
    # available. Enabling progressive rendering must not hide new measurements.
    assert frame(canvas).pixelColor(60, 60).red() < 250

    assert doc.history.undo(doc)
    canvas.notify_document_visual_changed()
    assert canvas._overlay_cache_enabled() == bool(initial_count)
    if initial_count == 0:
        assert frame(canvas).pixelColor(60, 60).red() == 255
    assert doc.history.redo(doc)
    doc.select_measurement(None)
    canvas.notify_document_visual_changed(added_measurement_ids=(added.id,))
    assert canvas._overlay_cache_enabled()
    assert frame(canvas).pixelColor(60, 60).red() < 250

    # Use persisted data to create a fresh document instance, rather than
    # accidentally relying on runtime counters left by the original canvas.
    restored = ImageDocument.from_dict(doc.to_dict())
    canvas.clear_document()
    source = QImage(1024, 768, QImage.Format.Format_RGB32)
    source.fill(0xFFFFFFFF)
    canvas.set_document(restored, source)
    canvas._sync_overlay_visual_state()
    assert canvas._overlay_cache_enabled()
    canvas.clear_document()
    assert not canvas._overlay_cache_enabled()


@pytest.mark.parametrize("kind", ["line", "count"])
def test_small_point_and_line_documents_keep_immediate_direct_rendering(
    scene, monkeypatch, kind
):
    canvas, *_ = scene
    measurements = [
        Measurement(
            id=str(i), image_id="scene", fiber_group_id=None, mode="manual",
            measurement_kind=kind,
            line_px=Line(Point(40, 60), Point(80, 60)) if kind == "line" else None,
            point_px=Point(60, 60) if kind == "count" else None,
        )
        for i in range(20)
    ]
    use_default_cache_policy(scene, monkeypatch, measurements)
    with record_direct_draws(canvas) as direct:
        frame(canvas)
        assert len(direct) == 1


@pytest.mark.parametrize("zoom", [4.0, 8.0, 16.0])
def test_zoomed_line_has_exact_screen_width_from_first_frame(scene, monkeypatch, zoom):
    canvas, doc, exact, preview, exact_pool, preview_pool = scene
    use_default_cache_policy(
        scene, monkeypatch,
        [
            Measurement(
                id=str(i), image_id="scene", fiber_group_id=None, mode="manual",
                measurement_kind="line",
                line_px=Line(Point(40, 60 + i * 10), Point(80, 60 + i * 10)),
            )
            for i in range(64)
        ],
    )
    canvas.set_settings(
        AppSettings(length_measurement_label_style=MeasurementLabelStyleSettings(enabled=False))
    )
    original = [item.to_dict() for item in doc.measurements]
    prepare_preview(canvas)
    preview_pool.complete(preview)
    canvas._set_zoom_at_widget_position(
        zoom, QPointF(60, 60), mode=CanvasZoomMode.CUSTOM
    )
    # Hold every target-scale tile. The first paint, partial arrivals and the
    # settled frame must all keep the same physical stroke and endpoint.
    frames = [frame(canvas)]
    assert frames[0].pixelColor(60, 60).red() < 250
    for key in canvas._visible_overlay_tile_keys(canvas._paint_context()):
        assert exact.request(canvas._build_overlay_tile_snapshot(key))
        exact_pool.complete(exact)
        frames.append(frame(canvas))
    with patch.object(canvas, "_overlay_cache_enabled", return_value=False):
        reference = frame(canvas)

    reference_region = frame_pixels(reference)[20:90, 30:90]
    for presented in frames:
        assert np.array_equal(frame_pixels(presented)[20:90, 30:90], reference_region)
        assert sum(presented.pixelColor(60, y).red() < 250 for y in range(52, 68)) == 4
    assert [item.to_dict() for item in doc.measurements] == original


@pytest.mark.parametrize("dpr", [1.0, 1.1, 1.25, 1.5, 2.0])
@pytest.mark.parametrize("space", [OverlayTextSizeSpace.IMAGE_PX, OverlayTextSizeSpace.SCREEN_PX])
def test_high_zoom_mixed_overlay_cold_and_partial_frames(scene, monkeypatch, dpr, space):
    canvas, doc, exact, preview, exact_pool, preview_pool = scene
    monkeypatch.setattr(canvas, "devicePixelRatioF", lambda: dpr)
    # The clip crosses the physical edge of an exact tile. The line body is
    # offscreen; its manually placed label and leader still reach the viewport.
    shape = compact_area("outer")
    hole = [Point(55, 55), Point(55, 65), Point(65, 65), Point(65, 55)]
    shape.area_rings_px.append(hole)
    line = Measurement("line", doc.id, None, "manual", line_px=Line(Point(170, 60), Point(190, 60)),
                       annotation_layout=MeasurementAnnotationLayout().with_manual_offset(Point(-110, -8)))
    count = Measurement("count", doc.id, None, "count", measurement_kind="count", point_px=Point(59, 58))
    use_default_cache_policy(scene, monkeypatch, [shape, line, compact_area("overlap"), count])
    canvas.set_settings(replace(canvas._settings, measurement_text_size_space=space))
    prepare_preview(canvas)
    preview_pool.complete(preview)
    canvas._zoom = 8.0
    canvas._pan = Point(-282.75, -280.5)
    frames = [frame_pixels(frame(canvas))]
    keys = canvas._visible_overlay_tile_keys(canvas._paint_context())
    for key in reversed(keys):
        assert exact.request(canvas._build_overlay_tile_snapshot(key))
        exact_pool.complete(exact)
        frames.append(frame_pixels(frame(canvas)))
    with patch.object(canvas, "_overlay_cache_enabled", return_value=False):
        reference = frame_pixels(frame(canvas))
    np.testing.assert_array_equal(frames[0], reference)
    # Qt uses different antialias coverage for glyphs/curves drawn through a
    # partial clip or a recorded command stream. Compare each physical pixel:
    # interiors (including tile seams, translucent overlaps and the hole)
    # retain their colour; boundary transitions stay within one pixel.
    kernel = np.ones((3, 3), np.uint8)
    lower = cv2.erode(reference, kernel).astype(np.int16)
    upper = cv2.dilate(reference, kernel).astype(np.int16)
    interior = np.all(lower == upper, axis=2)
    for displayed in frames[1:]:
        actual = displayed.astype(np.int16)
        assert np.abs(actual[interior] - lower[interior]).max() <= 1
        # Allow small coverage differences between subpixel glyph rasterizers
        # at a corner, even when a thin stroke has no solid interior sample.
        assert (lower - actual).max() <= 8
        assert (actual - upper).max() <= 8


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
    # A scale transition queries and draws visible candidates. It cannot use
    # the same whole-document bitmap that covered the cold opening frame.
    with patch.object(canvas, "_draw_scene_preview", side_effect=AssertionError("scaled overview")):
        scaled = frame(canvas)
    assert scaled.pixelColor(70, 70).red() < 250
    assert before == [measurement.to_dict() for measurement in doc.measurements]
    assert canvas._measurement_index().query_point(Point(60, 60), tolerance=1)[0].id == "0"


def test_returning_to_native_zoom_keeps_visible_measurements_when_overview_is_cold(scene):
    canvas, _, _, preview, _, _ = scene
    canvas._zoom = 2.0
    canvas._pan = Point(-60, -60)
    assert frame(canvas).pixelColor(60, 60).red() < 250
    canvas._zoom = 1.0
    canvas._pan = Point(0, 0)
    assert not preview.contains(canvas._scene_preview_key())
    with patch.object(canvas, "_draw_scene_preview", side_effect=AssertionError("loading frame")):
        assert frame(canvas).pixelColor(60, 60).red() < 250


def test_long_slide_leader_preserves_dash_phase_across_device_clips(desktop_application, monkeypatch):
    from fdm.ui.rendering import _draw_dashed_leader

    lengths = []
    original = QPainterPathStroker.createStroke

    def capture(stroker, path):
        lengths.append(path.length())
        return original(stroker, path)

    monkeypatch.setattr(QPainterPathStroker, "createStroke", capture)
    points = (QPointF(10_000_000, 37.5), QPointF(3, 40), QPointF(3, 40))
    pen = QPen(QColor("red"), 1., Qt.PenStyle.DashLine)

    def render(x, width):
        image = QImage(width, 80, QImage.Format.Format_RGB32)
        image.fill(QColor("white"))
        painter = QPainter(image)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.translate(-x, 0)
            _draw_dashed_leader(painter, points, pen)
        finally:
            painter.end()
        return frame_pixels(image)

    expected = render(0, 512)
    assert np.any(expected[:, :, 1] < 250)
    np.testing.assert_array_equal(np.concatenate([render(0, 211), render(211, 301)], axis=1), expected)
    assert max(lengths) < 525


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
