import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import math
import time
import numpy as np
from unittest.mock import patch
import pytest
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QImage, QPainter
from PySide6.QtWidgets import QApplication, QWidget
from fdm.geometry import Point
from fdm.ui.draft_preview_cache import DraftPreviewCache


@pytest.fixture(scope="module", autouse=True)
def app():
    application = QApplication.instance() or QApplication([])
    yield application


def render(
    cache,
    polygon,
    rings,
    *,
    origin=QPointF(10, 10),
    dpr=1.0,
    interactive=True,
    publisher=None,
    zoom=1.0,
    background="white",
):
    image = QImage(round(256 * dpr), round(256 * dpr), QImage.Format.Format_ARGB32_Premultiplied)
    image.setDevicePixelRatio(dpr)
    image.fill(QColor(background))
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    cache.draw(
        painter,
        owner=1,
        polygon=polygon,
        rings=rings,
        origin=origin,
        zoom=zoom,
        dpr=dpr,
        viewport=QRectF(0, 0, 256, 256),
        fill=QColor(52, 211, 153, 72),
        stroke=QColor("#34D399"),
        show_fill=True,
        interactive=interactive,
        publisher=publisher,
        layer_id="primary",
    )
    painter.end()
    return image


@pytest.mark.parametrize("dpr", [1.0, 1.25, 1.5, 2.0])
def test_pan_reuses_dense_draft_without_mutating_raw_geometry(dpr):
    polygon = [
        Point(100 + 60 * math.cos(i * math.tau / 10000), 100 + 60 * math.sin(i * math.tau / 10000))
        for i in range(10000)
    ]
    rings = [polygon]
    cache = DraftPreviewCache()
    before = [(p.x, p.y) for p in polygon]
    render(cache, polygon, rings, dpr=dpr)
    with patch.object(cache, "_paint", side_effect=AssertionError("warm draft redrawn")):
        for step in range(1, 5):
            render(cache, polygon, rings, origin=QPointF(10 + step / dpr, 10), dpr=dpr)
    assert cache.path_builds == cache.raster_builds == 1
    assert before == [(p.x, p.y) for p in polygon]


def test_hole_and_replacement_are_preserved_and_budget_is_global():
    polygon = [Point(20, 20), Point(180, 20), Point(180, 180), Point(20, 180)]
    hole = [Point(70, 70), Point(130, 70), Point(130, 130), Point(70, 130)]
    rings = [polygon, hole]
    cache = DraftPreviewCache(max_bytes=1024 * 1024)
    image = render(cache, polygon, rings)
    assert image.pixelColor(100, 100) == QColor("white")
    assert image.pixelColor(50, 50) != QColor("white")
    render(cache, polygon, [polygon])
    assert cache.path_builds == 2
    assert cache.bytes <= cache.max_bytes
    cache.discard(1)
    assert cache.bytes == 0


@pytest.mark.parametrize("dpr", [1.0, 1.25, 1.5, 2.0])
@pytest.mark.parametrize("background", ["white", "#1b2026"])
def test_isolated_draft_preserves_holes_strokes_and_last_complete_zoom(app, dpr, background):
    widget = QWidget()
    widget.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
    widget.show()
    cache = DraftPreviewCache(asynchronous=True)
    polygon = [Point(20, 20), Point(180, 20), Point(180, 180), Point(20, 180)]
    hole = [Point(70, 70), Point(130, 70), Point(130, 130), Point(70, 130)]
    rings = [polygon, hole]
    errors = []
    cache._raster_cache.tileFailed.connect(lambda key, error: errors.append(error))
    try:
        render(cache, polygon, rings, dpr=dpr, publisher=widget, background=background)
        deadline = time.perf_counter() + 10
        while cache._requests and time.perf_counter() < deadline:
            app.processEvents()
            time.sleep(0.001)
        assert not cache._requests and not errors
        image = render(cache, polygon, rings, dpr=dpr, publisher=widget, background=background)
        reference = render(DraftPreviewCache(), polygon, rings, dpr=dpr, background=background)
        pixels = np.frombuffer(image.constBits(), np.uint8).astype(int)
        expected = np.frombuffer(reference.constBits(), np.uint8).astype(int)
        assert np.abs(pixels - expected).max() <= 1
        builds = cache.path_builds
        # A missing target-scale raster must retain the complete same-session
        # preview. It cannot execute the complex native stroke in this callback.
        with patch.object(cache, "_paint", side_effect=AssertionError("synchronous stroke")):
            scaled = render(
                cache, polygon, rings, dpr=dpr, zoom=1.1, publisher=widget, background=background
            )
        assert scaled.pixelColor(round(50 * dpr), round(50 * dpr)) != QColor(background)
        assert scaled.pixelColor(round(110 * dpr), round(110 * dpr)) == QColor(background)
        assert cache.path_builds == builds
        cache.discard(1)
        assert not cache._layer_last and not cache._requests
    finally:
        cache.discard(1)
        widget.close()


@pytest.mark.parametrize("dpr", [1.0, 1.25, 1.5, 2.0])
def test_prepared_measurement_layer_preserves_raw_holes_and_label_placement(app, dpr):
    from fdm.models import ImageDocument, Measurement
    from fdm.settings import AppSettings
    from fdm.ui.rendering import build_passive_area_overlay_command, draw_area_measurement

    widget = QWidget()
    widget.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
    widget.show()
    cache = DraftPreviewCache(asynchronous=True)
    outer = [Point(20, 20), Point(180, 20), Point(180, 180), Point(20, 180)]
    hole = [Point(70, 70), Point(130, 70), Point(130, 130), Point(70, 130)]
    measurement = Measurement(
        "area",
        "doc",
        None,
        "magic_segment",
        measurement_kind="area",
        polygon_px=outer,
        area_rings_px=[outer, hole],
        exact_area_px=22000,
    )
    document = ImageDocument("doc", "image.png", (256, 256), measurements=[measurement])
    settings = AppSettings()
    command = build_passive_area_overlay_command(
        document,
        measurement,
        settings,
        zoom=1,
        line_width=2,
        show_fill=True,
        sprite_device_pixel_ratio=dpr,
    )
    origin = QPointF(10.3, 10.3)

    def draw():
        image = QImage(
            round(256 * dpr), round(256 * dpr), QImage.Format.Format_ARGB32_Premultiplied
        )
        image.setDevicePixelRatio(dpr)
        image.fill(QColor("white"))
        painter = QPainter(image)
        try:
            cache.draw_prepared(
                painter,
                owner=1,
                version=(1,),
                command=command,
                bounds=QRectF(0, 0, 256, 256),
                origin=origin,
                zoom=1,
                dpr=dpr,
                viewport=QRectF(0, 0, 256, 256),
                publisher=widget,
                layer_id="body",
            )
        finally:
            painter.end()
        return image

    try:
        draw()
        deadline = time.perf_counter() + 5
        while cache._requests and time.perf_counter() < deadline:
            app.processEvents()
            time.sleep(0.001)
        assert not cache._requests
        actual = draw()
        expected = QImage(actual.size(), actual.format())
        expected.setDevicePixelRatio(dpr)
        expected.fill(QColor("white"))
        painter = QPainter(expected)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            draw_area_measurement(
                painter,
                document,
                measurement,
                lambda p: QPointF(p.x + origin.x(), p.y + origin.y()),
                settings,
                line_width=2,
                endpoint_radius=4,
                selected=False,
                show_fill=True,
                show_handles=False,
                geometry_mode="raw",
                use_sprite_cache=True,
                sprite_device_pixel_ratio=dpr,
            )
        finally:
            painter.end()
        actual_pixels = np.frombuffer(actual.constBits(), np.uint8).astype(int)
        expected_pixels = np.frombuffer(expected.constBits(), np.uint8).astype(int)
        assert np.abs(actual_pixels - expected_pixels).max() <= 1
    finally:
        cache.discard(1)
        widget.close()
