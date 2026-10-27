from __future__ import annotations

import math
import time
from dataclasses import replace
from pathlib import Path
import sys

import pytest
from PySide6.QtCore import QEvent, QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QImage, QKeyEvent, QMouseEvent, QPainter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm.geometry import Line, Point
from fdm.measurement_annotation import MeasurementAnnotationLayout
from fdm.models import ImageDocument, Measurement
from fdm.settings import AppSettings
from fdm.ui.canvas import DocumentCanvas
from fdm.ui.screen_layer_cache import screen_layer_cache
from fdm.ui.measurement_annotation_layout import choose_automatic_offset, resolve_linked_label, resolve_dual_dimension
from fdm.ui.rendering import (
    draw_measurements, measurement_annotation_anchor, measurement_display_image_bounds,
    measurement_display_text_with_settings, resolve_measurement_annotation,
)


def make_document(kind="line", *, layout=True):
    document = ImageDocument(id="linked", path="/tmp/linked.png", image_size=(2000, 1000))
    measurement = Measurement(
        id="dimension", image_id=document.id, fiber_group_id=None, mode="manual",
        measurement_kind=kind, line_px=Line(Point(120, 160), Point(260, 160)),
        polyline_px=[Point(100, 140), Point(150, 180), Point(200, 140)] if kind == "polyline" else [],
        polygon_px=[Point(100, 100), Point(260, 100), Point(260, 220), Point(100, 220)] if kind == "area" else [],
        annotation_layout=MeasurementAnnotationLayout() if layout else None,
    )
    measurement.recalculate(None)
    document.measurements.append(measurement)
    return document, measurement


def mouse(kind, pos, *, button=Qt.MouseButton.LeftButton):
    return QMouseEvent(kind, pos, pos, button, button, Qt.KeyboardModifier.NoModifier)


@pytest.mark.parametrize("payload", [None, [], {"show_leader": "false"}, {"revision": 0}, {"revision": True},
                                    {"placement_mode": "screen"}, {"offset_px": [1, 2]},
                                    {"placement_mode": "manual"}, {"offset_px": {"x": math.inf, "y": 2}}])
def test_layout_rejects_invalid_persistent_state(payload):
    with pytest.raises((ValueError, TypeError)):
        MeasurementAnnotationLayout.from_dict(payload)


def test_layout_roundtrip_copies_offset_and_preserves_image_coordinates():
    offset = Point(421.5, -83.25)
    layout = MeasurementAnnotationLayout().with_manual_offset(offset)
    offset.x = 0
    assert layout.offset_px.x == 421.5
    assert MeasurementAnnotationLayout.from_dict(layout.to_dict()) == layout
    assert layout.with_automatic_position().offset_px is None


def test_dual_style_roundtrip_and_old_payload_compatibility():
    layout = MeasurementAnnotationLayout(leader_style="dual")
    assert MeasurementAnnotationLayout.from_dict(layout.to_dict()) == layout
    assert MeasurementAnnotationLayout.from_dict({}).leader_style == "single"
    with pytest.raises(ValueError):
        MeasurementAnnotationLayout.from_dict({"leader_style": "endpoint-numbers"})


@pytest.mark.parametrize("end, center", [(QPointF(240, 100), QPointF(150, 30)),
                                        (QPointF(160, 180), QPointF(340, -80)),
                                        (QPointF(100, 240), QPointF(180, 160))])
def test_dual_dimension_witnesses_keep_two_distinct_endpoints(end, center):
    start = QPointF(100, 100)
    resolved = resolve_dual_dimension(start, end, center, 90, 24)
    assert resolved.source_points == (start, end)
    assert len(resolved.witness_lines) == len(resolved.arrowheads) == 2
    tangent = end - start
    for source, witness, arrow in zip(resolved.source_points, resolved.witness_lines, resolved.arrowheads):
        displacement = witness[1] - source
        assert abs(displacement.x() * tangent.x() + displacement.y() * tangent.y()) < 1e-8
        gap = witness[0] - source
        assert math.hypot(gap.x(), gap.y()) == pytest.approx(2)
        assert resolved.bounds.contains(source)
        assert all(resolved.bounds.contains(point) for point in (*witness, *arrow))
    arrow_span = resolved.arrowheads[1][0] - resolved.arrowheads[0][0]
    assert arrow_span == tangent
    assert resolved.text_rect.center() == center


def test_short_dual_dimension_keeps_rail_and_outside_arrows_visible():
    start, end = QPointF(100, 100), QPointF(108, 100)
    resolved = resolve_dual_dimension(start, end, QPointF(104, 60), 180, 24)
    assert len(resolved.dimension_lines) == 1
    assert resolved.leader
    assert all(not resolved.text_rect.contains(point) for polygon in resolved.arrowheads for point in polygon)
    first, last = resolved.arrowheads
    assert first[1].x() < first[0].x()
    assert last[1].x() > last[0].x()
    hidden = resolve_dual_dimension(start, end, QPointF(104, 60), 180, 24, show_leader=False)
    assert not (hidden.leader or hidden.witness_lines or hidden.dimension_lines or hidden.arrowheads or hidden.source_points)


def test_contiguous_dual_dimensions_export_traces_each_endpoint_pair(tmp_path):
    document = ImageDocument(id="chain", path="/tmp/chain.png", image_size=(900, 700))
    settings = AppSettings()
    mapper = lambda p: QPointF(p.x, p.y)
    for index, (left, right, height) in enumerate(((100, 300, -100), (300, 500, -180), (500, 700, -260))):
        measurement = Measurement(id=f"chain-{index}", image_id=document.id, fiber_group_id=None,
            mode="manual", line_px=Line(Point(left, 400), Point(right, 400)),
            annotation_layout=MeasurementAnnotationLayout(leader_style="dual", offset_px=Point(0, height)))
        measurement.recalculate(None)
        document.measurements.append(measurement)
    image = QImage(900, 700, QImage.Format.Format_ARGB32_Premultiplied); image.fill(0)
    painter = QPainter(image)
    draw_measurements(painter, document, mapper, settings, line_width=2, endpoint_radius=4, use_sprite_cache=True)
    painter.end()
    for measurement in document.measurements:
        resolved = resolve_measurement_annotation(measurement, document, settings, mapper)
        assert len(resolved.witness_lines) == 2
        bounds = measurement_display_image_bounds(measurement, document, settings, mapper)
        assert bounds.contains(resolved.bounds)
        for source, arrow in zip(resolved.source_points, resolved.arrowheads):
            assert source.x() == arrow[0].x()
            midpoint = (source + arrow[0]) * .5
            assert image.pixelColor(round(midpoint.x()), round(midpoint.y())).alpha() > 0
    output = tmp_path / "contiguous-dual-dimensions.png"
    assert image.save(str(output))
    reloaded = QImage(str(output))
    assert reloaded.convertToFormat(image.format()) == image
    measurement = document.measurements[1]
    value = measurement.display_value()
    original = resolve_measurement_annotation(measurement, document, settings, mapper)
    measurement.annotation_layout = measurement.annotation_layout.with_manual_offset(Point(450, -300))
    distant = resolve_measurement_annotation(measurement, document, settings, mapper)
    assert distant.source_points == original.source_points
    assert distant.leader
    assert measurement.display_value() == value
    assert measurement_display_image_bounds(measurement, document, settings, mapper).contains(distant.bounds)


def test_leader_uses_semantic_anchor_and_horizontal_landing():
    layout = resolve_linked_label(QPointF(10, 20), QPointF(160, 90), 100, 30)
    assert layout.leader[0] == QPointF(10, 20)
    assert layout.leader[-1].y() == layout.leader[-2].y()
    assert layout.text_rect.left() == layout.leader[-1].x()
    assert layout.bounds.contains(layout.text_rect)
    assert layout.bounds.contains(QPointF(10, 20))
    assert not resolve_linked_label(QPointF(10, 20), QPointF(160, 90), 100, 30, show_leader=False).leader


def test_automatic_solver_avoids_occupied_labels():
    anchor = Point(300, 300)
    first = choose_automatic_offset(anchor, 100, 30, image_rect=QRectF(0, 0, 800, 600))
    occupied = QRectF(anchor.x + first.x - 50, anchor.y + first.y - 15, 100, 30)
    second = choose_automatic_offset(anchor, 100, 30, occupied=[occupied], image_rect=QRectF(0, 0, 800, 600))
    second_rect = QRectF(anchor.x + second.x - 50, anchor.y + second.y - 15, 100, 30)
    assert not second_rect.intersects(occupied)


@pytest.mark.parametrize("kind", ["line", "polyline", "area"])
def test_shared_bounds_and_export_include_remote_label_and_leader(kind):
    document, measurement = make_document(kind)
    measurement.annotation_layout = MeasurementAnnotationLayout().with_manual_offset(Point(1000, 100))
    settings = AppSettings()
    settings.area_measurement_label_style = replace(settings.area_measurement_label_style, enabled=True)
    mapping = lambda p: QPointF(p.x, p.y)
    linked = resolve_measurement_annotation(measurement, document, settings, mapping)
    bounds = measurement_display_image_bounds(measurement, document, settings, mapping)
    assert bounds.contains(linked.text_rect)
    assert bounds.contains(linked.anchor)
    assert linked.text_rect.left() > 900
    image = QImage(2000, 1000, QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(0)
    painter = QPainter(image)
    painter.setClipRect(QRectF(900, 0, 1000, 800))
    draw_measurements(painter, document, mapping, settings, line_width=2, endpoint_radius=4,
                      visible_rect=QRectF(900, 0, 1000, 800))
    painter.end()
    label_crop = image.copy(linked.text_rect.toAlignedRect())
    assert any(bytes(label_crop.constBits()))
    assert measurement_display_text_with_settings(measurement, document, settings).startswith("M1 · ")


def test_automatic_offset_is_stable_across_zoom_and_new_measurements():
    document, measurement = make_document()
    settings = AppSettings()
    first = resolve_measurement_annotation(measurement, document, settings, lambda p: QPointF(p.x, p.y))
    duplicate = replace(measurement, id="second")
    document.measurements.append(duplicate)
    second = resolve_measurement_annotation(measurement, document, settings, lambda p: QPointF(p.x * 2 + 8, p.y * 2 + 9))
    assert second.text_rect.center() == first.text_rect.center() * 2 + QPointF(8, 9)


def test_drag_is_presentation_only_and_emits_one_change_on_release():
    document, measurement = make_document()
    canvas = DocumentCanvas()
    try:
        canvas.resize(1000, 600)
        canvas.set_document(document, QImage(2000, 1000, QImage.Format.Format_RGB32))
        linked = resolve_measurement_annotation(measurement, document, canvas._settings, canvas.image_to_widget)
        start = linked.text_rect.center()
        before = measurement.to_dict()
        before_revision = measurement.geometry_revision
        events = []
        def receive(document_id, measurement_id, layout):
            assert measurement.annotation_layout.to_dict() == before["annotation_layout"]
            events.append((document_id, measurement_id, layout))
            measurement.annotation_layout = layout
        canvas.measurementAnnotationEdited.connect(receive)
        canvas.mousePressEvent(mouse(QEvent.Type.MouseButtonPress, start))
        for delta in (20, 40, 80):
            canvas.mouseMoveEvent(mouse(QEvent.Type.MouseMove, start + QPointF(delta, 45)))
        assert not events
        canvas.mouseReleaseEvent(mouse(QEvent.Type.MouseButtonRelease, start + QPointF(80, 45)))
        assert len(events) == 1
        assert measurement.annotation_layout.placement_mode == "manual"
        assert measurement.geometry_revision == before_revision
        after = measurement.to_dict()
        before.pop("annotation_layout")
        after.pop("annotation_layout")
        assert after == before
    finally:
        canvas.clear_document()
        canvas.close()


def test_escape_restores_legacy_label_without_emitting_history():
    document, measurement = make_document(layout=False)
    canvas = DocumentCanvas()
    try:
        canvas.resize(1000, 600)
        canvas.set_document(document, QImage(2000, 1000, QImage.Format.Format_RGB32))
        from fdm.ui.rendering import measurement_label_image_bounds
        bounds = measurement_label_image_bounds(measurement, document, canvas._settings, canvas.image_to_widget)
        start = canvas.image_to_widget(Point(bounds.center().x(), bounds.center().y()))
        changes = []
        canvas.measurementAnnotationEdited.connect(lambda *args: changes.append(args))
        canvas.mousePressEvent(mouse(QEvent.Type.MouseButtonPress, start))
        canvas.mouseMoveEvent(mouse(QEvent.Type.MouseMove, start + QPointF(80, 90)))
        assert measurement.annotation_layout is not None
        canvas.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier))
        assert measurement.annotation_layout is None
        assert not changes
        assert "M1" not in measurement_display_text_with_settings(measurement, document, canvas._settings)
    finally:
        canvas.clear_document()
        canvas.close()


def test_remote_label_enters_spatial_index_and_invalidates_old_tiles():
    document, measurement = make_document()
    canvas = DocumentCanvas()
    try:
        canvas.set_document(document, QImage(2000, 1000, QImage.Format.Format_RGB32))
        canvas._zoom = 1
        canvas._overlay_known_namespaces.add((1.0, 1.0))
        measurement.annotation_layout = MeasurementAnnotationLayout().with_manual_offset(Point(1200, 50))
        canvas._sync_overlay_visual_state()
        index = canvas._measurement_display_scene_index(zoom=1)
        assert measurement in index.query_rect(QRectF(1200, 100, 400, 200))
        measurement.annotation_layout = measurement.annotation_layout.with_manual_offset(Point(1700, 50))
        canvas._refresh_measurement_annotation_preview()
        canvas._sync_overlay_visual_state()
        assert any(key[2] >= 2 and epoch > 0 for key, epoch in canvas._overlay_tile_epochs.items())
        index = canvas._measurement_display_scene_index(zoom=1)
        assert measurement in index.query_rect(QRectF(1700, 100, 280, 200))
    finally:
        canvas.clear_document()
        canvas.close()


def test_drag_cannot_replay_cached_previous_label(monkeypatch):
    monkeypatch.setenv("FDM_ENABLE_CANVAS_OVERLAY_CACHE", "1")
    document, measurement = make_document()
    canvas = DocumentCanvas()
    try:
        canvas.set_document(document, QImage(2000, 1000, QImage.Format.Format_RGB32))
        assert canvas._overlay_cache_enabled()
        start = resolve_measurement_annotation(measurement, document, canvas._settings, canvas.image_to_widget).text_rect.center()
        canvas.mousePressEvent(mouse(QEvent.Type.MouseButtonPress, start))
        canvas.mouseMoveEvent(mouse(QEvent.Type.MouseMove, start + QPointF(300, 100)))
        assert not canvas._overlay_cache_enabled()
        canvas.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier))
        assert canvas._overlay_cache_enabled()
    finally:
        canvas.clear_document()
        canvas.close()


def test_slide_label_drag_does_not_require_mounted_pixels(monkeypatch):
    from fdm.ui.digital_slide_canvas import DigitalSlideCanvas
    document, measurement = make_document()
    canvas = DigitalSlideCanvas()
    try:
        canvas.set_document(document, QImage(2000, 1000, QImage.Format.Format_RGB32))
        canvas._zoom = 1
        monkeypatch.setattr(canvas, "_pixel_pointer_event_allowed", lambda position: False)
        monkeypatch.setattr(canvas, "vector_measurement_available", lambda position: False)
        resolved = resolve_measurement_annotation(measurement, document, canvas._settings, canvas.image_to_widget)
        start = resolved.text_rect.center()
        changes = []
        canvas.measurementAnnotationEdited.connect(lambda *args: changes.append(args))
        canvas.mousePressEvent(mouse(QEvent.Type.MouseButtonPress, start))
        canvas.mouseMoveEvent(mouse(QEvent.Type.MouseMove, start + QPointF(900, 100)))
        canvas.mouseReleaseEvent(mouse(QEvent.Type.MouseButtonRelease, start + QPointF(900, 100)))
        assert len(changes) == 1
        assert changes[0][2].offset_px.x > 800
        assert canvas._annotation_drag is None
    finally:
        canvas.clear_document()
        canvas.close()


def test_identifier_renumbering_invalidates_remaining_label_cache():
    document, first = make_document()
    later = replace(first, id="later", line_px=Line(Point(1600, 200), Point(1700, 200)))
    document.measurements.append(later)
    canvas = DocumentCanvas()
    try:
        canvas.set_document(document, QImage(2000, 1000, QImage.Format.Format_RGB32))
        canvas._overlay_known_namespaces.add((1.0, 1.0))
        before = canvas._measurement_visual_fingerprint(later)
        canvas._sync_overlay_visual_state()
        document.remove_measurement_incremental(first.id)
        canvas._sync_overlay_visual_state()
        assert canvas._measurement_visual_fingerprint(later) != before
        assert measurement_display_text_with_settings(later, document, canvas._settings).startswith("M1 · ")
        assert any(key[2] >= 3 and epoch > 0 for key, epoch in canvas._overlay_tile_epochs.items())
    finally:
        canvas.clear_document()
        canvas.close()


def _logical_frame(canvas):
    # Match the device raster of the screen cache. Rendering a DPR 2 cached
    # layer into a DPR 1 target would compare downsampling with direct pixels.
    dpr = canvas.devicePixelRatioF()
    image = QImage(round(canvas.width() * dpr), round(canvas.height() * dpr), QImage.Format.Format_ARGB32_Premultiplied)
    image.setDevicePixelRatio(dpr)
    image.fill(0)
    canvas.render(image)
    return image


def test_geometry_hover_links_new_label_without_selection_or_document_edit():
    document, measurement = make_document()
    measurement.annotation_layout = MeasurementAnnotationLayout(offset_px=Point(300, 100))
    canvas = DocumentCanvas()
    try:
        source = QImage(2000, 1000, QImage.Format.Format_RGB32)
        source.fill(QColor("#283040"))
        canvas.set_document(document, source)
        canvas._zoom = 1; canvas._pan = Point(0, 0)
        changes = []
        canvas.measurementAnnotationHoverChanged.connect(lambda *args: changes.append(args))
        before = document.to_dict()
        canvas._measurement_annotation_mouse_move(mouse(QEvent.Type.MouseMove, QPointF(190, 160)))
        assert changes == [(document.id, measurement.id)]
        assert document.to_dict() == before
        canvas._measurement_annotation_mouse_move(mouse(QEvent.Type.MouseMove, QPointF(600, 450)))
        assert changes[-1] == (document.id, None)
        measurement.annotation_layout = None
        canvas._measurement_annotation_mouse_move(mouse(QEvent.Type.MouseMove, QPointF(125, 160)))
        assert canvas._hovered_measurement_annotation_id is None
    finally:
        canvas.clear_document()
        canvas.close()


@pytest.mark.parametrize("kind", ["line", "polyline", "area"])
@pytest.mark.parametrize("leader_style", ["single", "dual"])
def test_drag_static_layer_is_reused_and_escape_restores_exact_pixels(kind, leader_style):
    document, measurement = make_document(kind)
    measurement.annotation_layout = MeasurementAnnotationLayout(offset_px=Point(150, 40), leader_style=leader_style)
    canvas = DocumentCanvas()
    try:
        canvas.resize(900, 600)
        source = QImage(2000, 1000, QImage.Format.Format_RGB32); source.fill(QColor("#283040"))
        canvas.set_document(document, source)
        canvas.set_settings(replace(canvas._settings, area_measurement_label_style=replace(canvas._settings.area_measurement_label_style, enabled=True)))
        canvas._zoom = 1; canvas._pan = Point(0, 0)
        document.select_measurement(measurement.id)
        before = _logical_frame(canvas)
        start = resolve_measurement_annotation(measurement, document, canvas._settings, canvas.image_to_widget).text_rect.center()
        builds = screen_layer_cache.builds
        canvas.mousePressEvent(mouse(QEvent.Type.MouseButtonPress, start))
        assert screen_layer_cache.builds == builds + 1
        for step in (30, 90, 160):
            canvas.mouseMoveEvent(mouse(QEvent.Type.MouseMove, start + QPointF(step, 80)))
            _logical_frame(canvas)
        assert screen_layer_cache.builds == builds + 1
        assert screen_layer_cache.preserve((id(canvas), "annotation-drag-static"))
        # Compare the cached moving frame to an independently composed exact
        # frame at the same current layout: no old glyphs or leader may remain.
        preview = _logical_frame(canvas)
        drag = canvas._annotation_drag
        canvas._annotation_drag = None
        reference = _logical_frame(canvas)
        canvas._annotation_drag = drag
        import numpy as np
        actual = np.frombuffer(preview.constBits(), np.uint8).astype(np.int16)
        expected = np.frombuffer(reference.constBits(), np.uint8).astype(np.int16)
        assert np.max(np.abs(actual - expected)) <= 1
        canvas.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier))
        assert not screen_layer_cache.preserve((id(canvas), "annotation-drag-static"))
        after = _logical_frame(canvas)
        assert before == after
    finally:
        canvas.clear_document(); canvas.close()


@pytest.mark.parametrize("mutation", ["pan", "zoom", "source", "settings", "brightness", "focus", "generation"])
def test_drag_static_layer_invalidates_on_presentation_changes(mutation):
    from fdm.ui.digital_slide_canvas import DigitalSlideCanvas
    document, measurement = make_document()
    measurement.annotation_layout = MeasurementAnnotationLayout(offset_px=Point(150, 40))
    canvas = DigitalSlideCanvas()
    try:
        canvas.resize(900, 600)
        canvas.set_document(document, QImage(2000, 1000, QImage.Format.Format_RGB32))
        canvas._zoom = 1; canvas._pan = Point(0, 0)
        start = resolve_measurement_annotation(measurement, document, canvas._settings, canvas.image_to_widget).text_rect.center()
        assert canvas._measurement_annotation_mouse_press(mouse(QEvent.Type.MouseButtonPress, start))
        builds = screen_layer_cache.builds
        if mutation == "pan": canvas._pan = Point(20, 15)
        elif mutation == "zoom": canvas._zoom = 1.5
        elif mutation == "source": canvas._image.fill(QColor("#80ffff"))
        elif mutation == "settings": canvas.set_settings(replace(canvas._settings, default_measurement_color="#00ff00"))
        elif mutation == "brightness": canvas.set_display_adjustments(20, 5)
        elif mutation == "focus": canvas._focus_index += 1
        elif mutation == "generation": canvas._view_generation += 1
        target = QImage(900, 600, QImage.Format.Format_ARGB32_Premultiplied); target.fill(0)
        painter = QPainter(target)
        canvas._draw_measurement_annotation_drag_layer(painter)
        painter.end()
        assert screen_layer_cache.builds == builds + 1
    finally:
        canvas.clear_document(); canvas.close()
