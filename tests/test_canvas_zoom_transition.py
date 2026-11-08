"""Hold a coherent ordinary viewport while delayed workers prepare the latest zoom."""

from dataclasses import replace
import cv2

import pytest
import numpy as np
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtWidgets import QApplication

from fdm.geometry import Line, Point
from fdm.models import ImageDocument, Measurement
from fdm.measurement_annotation import MeasurementAnnotationLayout
from fdm.ui.canvas import CanvasZoomMode
from fdm.settings import MeasurementLabelStyleSettings
from test_canvas_progressive_overlay import compact_area, frame, frame_pixels, scene
from test_manual_endpoint_extension import drag


def wheel(canvas, delta, at):
    QApplication.sendEvent(canvas, QWheelEvent(
        at, canvas.mapToGlobal(at), QPoint(), QPoint(0, delta),
        Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.ScrollUpdate, False,
    ))


def prepare(transition):
    for _ in range(200):
        if not transition._timer.isActive():
            break
        transition._timer.stop()
        transition._prepare()
    else:
        pytest.fail("visible viewport preparation did not finish")


def force_worker(canvas, monkeypatch):
    transition = canvas._zoom_transition
    monkeypatch.setattr(transition, "MIN_OBJECTS", 1)
    canvas._sync_overlay_visual_state()
    return transition


def test_zoom_accumulates_wheel_ticks_and_presents_only_latest_complete_view(scene, monkeypatch):
    canvas, doc, _, preview, _, jobs = scene
    transition = force_worker(canvas, monkeypatch)
    before = frame_pixels(frame(canvas))
    at = QPointF(173.5, 108.5)
    point = canvas.widget_to_image(at)
    original_zoom = canvas.view_zoom()
    changes = []
    canvas.viewZoomChanged.connect(changes.append)
    wheel(canvas, 120, at)
    prepare(transition)
    assert transition._submitted
    assert canvas.view_zoom() == original_zoom
    assert (frame_pixels(frame(canvas)) == before).all()
    wheel(canvas, 120, at)
    wheel(canvas, 120, at)
    wheel(canvas, -120, at)
    target_zoom = original_zoom * 1.15**2
    assert transition.requested.zoom == pytest.approx(target_zoom)
    assert canvas.widget_to_image(at) == point
    assert changes == []
    jobs.complete(preview)
    assert not transition.ready
    assert canvas.view_zoom() == original_zoom
    prepare(transition)
    jobs.complete(preview)
    assert transition.ready and canvas.view_zoom() == original_zoom
    transition.publish(full_paint=False)
    assert canvas.view_zoom() == original_zoom
    frame(canvas)
    assert changes == pytest.approx([target_zoom])
    assert doc.view_state.zoom == pytest.approx(target_zoom)
    assert transition.presented is not None
    anchor = canvas.image_to_widget(point)
    assert (anchor.x(), anchor.y()) == pytest.approx((at.x(), at.y()))
    assert transition.requested is None


@pytest.mark.parametrize("change", ["geometry", "style", "document", "hide", "tool"])
def test_obsolete_viewport_cannot_publish_after_an_edit_or_context_change(scene, monkeypatch, change):
    canvas, doc, _, preview, _, jobs = scene
    transition = force_worker(canvas, monkeypatch)
    canvas.set_view_zoom(2)
    prepare(transition)
    assert transition._submitted
    if change == "geometry":
        doc.mark_measurement_geometry_changed()
    elif change == "style":
        canvas.set_settings(replace(canvas._settings, professional_measurement_label_decimals=6))
    elif change == "document":
        replacement = ImageDocument(id="replacement", path="sample.png", image_size=doc.image_size)
        canvas.set_document(replacement, canvas._image)
    elif change == "hide":
        canvas.hide()
    elif change == "tool":
        canvas.set_tool_mode("count")
    jobs.complete(preview)
    frame(canvas)
    assert transition.requested is None
    assert transition.presented is None
    assert canvas.view_zoom() != 2


def test_alt_endpoint_drag_uses_current_sharp_view_even_when_new_zoom_requested(scene, monkeypatch):
    canvas, doc, _, preview, _, jobs = scene
    endpoint, destination = Point(123.25, 112), Point(195, 143)
    source = Measurement("endpoint", doc.id, None, "manual",
                         line_px=Line(Point(104, 111), endpoint))
    doc.measurements.append(source)
    doc.mark_measurement_geometry_changed()
    canvas.notify_document_visual_changed()
    canvas.set_tool_mode("manual")
    transition = force_worker(canvas, monkeypatch)
    origin = canvas.view_zoom()
    collected = []
    canvas.lineCommitted.connect(lambda _doc, _mode, line: collected.append(line))
    canvas.set_view_zoom(2)
    prepare(transition)
    assert transition._submitted
    drag(canvas, canvas.image_to_widget(endpoint), canvas.image_to_widget(destination), extend=True)
    jobs.complete(preview)
    frame(canvas)
    assert collected == [Line(endpoint, destination)]
    assert doc.get_measurement(source.id).line_px.end == endpoint
    assert canvas.view_zoom() == origin
    assert transition.requested is None


def test_fit_and_worker_refusal_still_publish_sharp_final_target(scene, monkeypatch):
    canvas, _, _, preview, _, _ = scene
    transition = force_worker(canvas, monkeypatch)
    monkeypatch.setattr(preview, "request", lambda _: False)
    original = canvas.view_zoom()
    canvas.fit_to_view()
    prepare(transition)
    assert transition.ready
    assert canvas.view_zoom() == original
    target = transition.requested.zoom
    frame(canvas)
    assert canvas.zoom_mode() == CanvasZoomMode.FIT
    assert target == canvas.view_zoom() < original
    assert transition.presented is None


@pytest.mark.parametrize("dpr", [1.0, 1.25, 1.5, 2.0])
def test_prepared_view_uses_target_device_pixels_for_line_and_text(scene, monkeypatch, dpr):
    canvas, doc, _, preview, _, jobs = scene
    monkeypatch.setattr(canvas, "devicePixelRatioF", lambda: dpr)
    doc.measurements = [
        Measurement("line", doc.id, None, "manual", line_px=Line(Point(53, 87), Point(131, 73)),
                    annotation_layout=MeasurementAnnotationLayout.ordinary_near())
    ]
    doc.mark_measurement_geometry_changed()
    doc.mark_session_dirty()
    canvas.notify_document_visual_changed()
    canvas.center_on_image_point(Point(91.125, 80.5625))
    transition = force_worker(canvas, monkeypatch)
    canvas.set_view_zoom(4)
    prepare(transition)
    jobs.complete(preview)
    prepared = frame_pixels(frame(canvas))
    assert transition.presented is not None
    # Repaint the identical transform directly on the original image.
    monkeypatch.setattr(canvas, "_overlay_cache_enabled", lambda: False)
    direct = frame_pixels(frame(canvas))
    assert np.count_nonzero(direct[:, :, 0] < 240) > 100
    difference = np.abs(prepared.astype(np.int16) - direct)
    assert difference.max() <= 1


def test_large_area_label_falls_back_to_visible_text_without_allocating_sprite(scene, monkeypatch):
    import importlib
    sprites = importlib.import_module("fdm.ui.screen_label_sprite_cache")
    canvas, doc, _, preview, _, jobs = scene
    monkeypatch.setattr(sprites, "MAX_SINGLE_LABEL_BYTES", 1)
    sprites.screen_label_sprite_cache.clear()
    doc.measurements = [compact_area("area")]
    doc.mark_measurement_geometry_changed()
    doc.mark_session_dirty()
    canvas.notify_document_visual_changed()
    canvas.set_settings(replace(canvas._settings,
                                area_measurement_label_style=MeasurementLabelStyleSettings(color="#ff0000")))
    canvas.center_on_image_point(Point(60, 60))
    transition = force_worker(canvas, monkeypatch)
    canvas.set_view_zoom(2)
    prepare(transition)
    jobs.complete(preview)
    prepared = frame_pixels(frame(canvas))
    assert transition.presented is not None and not len(sprites.screen_label_sprite_cache)
    monkeypatch.setattr(canvas, "_overlay_cache_enabled", lambda: False)
    reference = frame_pixels(frame(canvas))
    # On Windows, red glyphs drawn onto an opaque image use subpixel color AA.
    # On transparent intermediate surfaces Qt uses grayscale coverage. Compare
    # glyph cores against neighboring ink of the same color, in both directions.
    def red_ink(pixels):
        return pixels[:, :, 2].astype(np.int16) - pixels[:, :, :2].max(axis=2)

    ink, reference_ink = red_ink(prepared), red_ink(reference)
    assert (ink > 128).sum() > 40
    assert (reference_ink > 128).sum() > 40
    kernel = np.ones((3, 3), np.uint8)
    for first, second in ((ink, reference_ink), (reference_ink, ink)):
        assert np.all(cv2.dilate(second, kernel)[first > 128] > 40)
    # Independently locate the bottom of the RAW area under the label.
    center = canvas.image_to_widget(Point(60, 80))
    x = round(center.x() * canvas.devicePixelRatioF())
    bottom = round(center.y() * canvas.devicePixelRatioF())
    for image in (prepared, reference):
        last = np.flatnonzero(image[:, x, :3].min(axis=1) < 200)[-1]
        assert abs(last - bottom) <= 2 * canvas.devicePixelRatioF()


def test_staging_budget_releases_commands_before_recording_a_dense_count_group(scene, monkeypatch):
    canvas, doc, _, preview, _, jobs = scene
    canvas.set_settings(replace(canvas._settings, show_count_numbers=True))
    doc.measurements = [
        Measurement(f"count-{i}", doc.id, None, "count", measurement_kind="count",
                    point_px=Point(240 + i % 16, 165 + i // 16))
        for i in range(224)
    ]
    doc.mark_measurement_geometry_changed()
    doc.mark_session_dirty()
    canvas.notify_document_visual_changed()
    canvas.center_on_image_point(Point(248, 172))
    monkeypatch.setattr(preview, "_max_bytes", 1024 * 1024)
    transition = force_worker(canvas, monkeypatch)
    canvas.set_view_zoom(4)
    prepare(transition)
    assert transition.ready and not transition._submitted
    assert not transition._commands and not transition._counts and not jobs.runnables
    assert np.count_nonzero(frame_pixels(frame(canvas))[:, :, 0] < 200) > 100
    assert canvas.view_zoom() == 4
