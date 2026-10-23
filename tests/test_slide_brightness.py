from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from PySide6.QtCore import QRect, QRectF
from PySide6.QtGui import QColor, QHideEvent, QImage, QPainter

from fdm.geometry import Point
from fdm.models import ImageDocument
from fdm.services.digital_slide_renderer import DigitalSlideRenderFrame
from fdm.services.digital_slide_store import DigitalSlideManifest
from fdm.ui.digital_slide_canvas import DigitalSlideCanvas
from fdm.ui.slide_brightness import SlideBrightnessCache, brightness_lut


def image(value=64, width=320, height=240, fmt=QImage.Format.Format_RGB32):
    result = QImage(width, height, fmt)
    result.fill(QColor(value, value, value))
    return result


def canvas():
    widget = DigitalSlideCanvas()
    widget.resize(320, 240)
    document = ImageDocument(id='slide', path='/tmp/brightness.fdmslide',
                             image_size=(320, 240), document_kind='digital_slide')
    widget.set_document(document, image())
    widget._slide_manifest = DigitalSlideManifest(
        version=1, width=320, height=240, viewport_width=320, viewport_height=240,
        focus_levels=[0, 1])
    widget._zoom = 1.0
    widget._browse_center = Point(160, 120)
    widget._sync_pan_from_browse_center()
    widget._native_frame_key = widget._native_request_key()
    return widget


def frame(source):
    return DigitalSlideRenderFrame(1, 'display', (0., 0., 320., 240.), (320, 240),
                                   0, 1., 0, source, 0., 0, 0, pixel_exact=True)


def paint(widget):
    result = image(0)
    painter = QPainter(result)
    try:
        widget._draw_base_image(painter)
    finally:
        painter.end()
    return result


@pytest.mark.parametrize('value', [-100, -50, 0, 50, 100])
def test_lut_endpoints_direction_and_monotonicity(value):
    lut = brightness_lut(value)
    assert (int(lut[0]), int(lut[255])) == (0, 255)
    assert np.all(np.diff(lut.astype(int)) >= 0)
    assert int(lut[64]) == round(255 * (64 / 255) ** (1 / 2 ** (value / 50)))
    if value == 0:
        assert np.array_equal(lut, np.arange(256))
    elif value > 0:
        assert lut[64] > 64
    else:
        assert lut[64] < 64


@pytest.mark.parametrize('fmt', [QImage.Format.Format_RGB32, QImage.Format.Format_RGB888,
    QImage.Format.Format_BGR888, QImage.Format.Format_Grayscale8,
    QImage.Format.Format_ARGB32, QImage.Format.Format_RGBA8888,
    QImage.Format.Format_ARGB32_Premultiplied])
def test_mapping_uses_immutable_pixels_and_preserves_alpha_and_stride(fmt):
    original = image(width=7, height=5, fmt=fmt)
    original.setPixelColor(3, 2, QColor(64, 128, 192, 73))
    before = bytes(original.constBits())
    color = original.pixelColor(3, 2)
    cache = SlideBrightnessCache()
    with patch('fdm.ui.slide_brightness.cv2.LUT', wraps=__import__('cv2').LUT) as lut:
        assert cache.mapped_image(original, 0) is original
        assert lut.call_count == 0
        result = cache.mapped_image(original, 50)
        again = cache.mapped_image(original, 50)
        assert lut.call_count == 1
    assert result.cacheKey() == again.cacheKey()
    mapped = result.pixelColor(3, 2)
    assert mapped.alpha() == color.alpha()
    for before_channel, after_channel in zip(color.getRgb()[:3], mapped.getRgb()[:3]):
        assert after_channel == brightness_lut(50)[before_channel]
    assert bytes(original.constBits()) == before
    assert cache.bytes_used == result.sizeInBytes()
    # A consumer editing its returned image must not corrupt later cache hits.
    result.fill(QColor('red'))
    assert cache.mapped_image(original, 50).pixelColor(3, 2) == mapped


def test_cache_identity_brightness_roi_eviction_and_source_mutation():
    original = image(width=8, height=8)
    cache = SlideBrightnessCache(max_bytes=128)
    first = cache.mapped_image(original, 50, QRect(0, 0, 4, 4))
    second = cache.mapped_image(original, -50, QRect(0, 0, 4, 4))
    assert cache.bytes_used == 128
    assert first.pixelColor(0, 0) != second.pixelColor(0, 0)
    original.setPixelColor(0, 0, QColor('white'))
    third = cache.mapped_image(original, 50, QRect(0, 0, 4, 4))
    assert third.pixelColor(0, 0) == QColor('white')
    assert cache.bytes_used == 128
    oversized = cache.mapped_image(original, 50)
    assert oversized.sizeInBytes() > cache.max_bytes
    assert cache.bytes_used <= cache.max_bytes
    cache.clear()
    assert cache.bytes_used == 0


def test_tiled_visible_crop_matches_full_mapping_with_interpolation_halo():
    original = image(width=2100, height=1800)
    pixels = np.ndarray((1800, 2100, 4), np.uint8, original.bits(), strides=(original.bytesPerLine(), 4, 1))
    pixels[:, :, 0] = np.arange(2100, dtype=np.uint16)[None, :] % 256
    pixels[:, :, 1] = np.arange(1800, dtype=np.uint16)[:, None] % 256
    target = QRectF(-610.2, -590.7, 1890, 1620)
    visible = QRectF(0, 0, 320, 240)
    outputs = []
    tile_cache = SlideBrightnessCache(max_bytes=5 * 1024 * 1024)
    for cache in (SlideBrightnessCache(), tile_cache):
        result = image(0)
        painter = QPainter(result)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        cache.draw_image(painter, target, original, 50, visible)
        painter.end()
        outputs.append(np.frombuffer(result.constBits(), np.uint8).copy())
    # Qt's fixed-point interpolation can round by one at a cropped boundary.
    assert np.max(np.abs(outputs[0].astype(int) - outputs[1].astype(int))) <= 1
    assert 0 < tile_cache.bytes_used <= tile_cache.max_bytes


@pytest.mark.parametrize('path', ['current', 'previous', 'coarse', 'preview', 'overview',
                                   'focus_frame', 'focus_image', 'navigation_image'])
def test_all_paint_sources_map_once_without_changing_raw_images_or_readiness(path):
    widget = canvas()
    raw = image()
    before = bytes(raw.constBits())
    value = frame(raw)
    assignments = {
        'current': ('_render_frame', value), 'previous': ('_previous_render_frame', value),
        'coarse': ('_coarse_render_frame', replace(value, quality='coarse')),
        'preview': ('_presentation_preview_frame', value), 'overview': ('_overview_image', raw),
        'focus_frame': ('_focus_transition_frame', value),
        'focus_image': ('_focus_transition_image', raw),
        'navigation_image': ('_navigation_transition_image', raw),
    }
    try:
        name, content = assignments[path]
        setattr(widget, name, content)
        widget._overview_focus_index = 0
        widget._focus_transition_native_rect = QRectF(0, 0, 320, 240)
        widget._navigation_transition_target = QRectF(0, 0, 320, 240)
        readiness = (widget._native_frame_key, widget._pixel_work_enabled, widget._view_generation)
        metadata = deepcopy(widget._document.metadata)
        raw_key = widget._image.cacheKey()
        original_signature = widget._overlay_background_signature()
        widget.set_display_brightness(50)
        assert paint(widget).pixelColor(160, 120).red() == brightness_lut(50)[64]
        assert paint(widget).pixelColor(160, 120).red() == brightness_lut(50)[64]
        assert widget._overlay_background_signature() != original_signature
        assert (widget._native_frame_key, widget._pixel_work_enabled, widget._view_generation) == readiness
        assert widget._document.metadata == metadata
        assert widget._image.cacheKey() == raw_key
        assert bytes(raw.constBits()) == before
        widget.set_display_brightness(0)
        assert paint(widget).pixelColor(160, 120).red() == 64
    finally:
        widget.shutdown()
        widget.close()


def test_overview_notification_independent_runtime_state_and_lifecycle():
    first, second = canvas(), canvas()
    try:
        first._overview_image = image()
        observed = []
        first.overviewImageChanged.connect(lambda im: observed.append(im.pixelColor(0, 0).red()))
        first.set_display_brightness(50)
        assert observed[-1] == brightness_lut(50)[64]
        assert first.overview_image().pixelColor(0, 0).red() == brightness_lut(50)[64]
        assert first._overview_image.pixelColor(0, 0).red() == 64
        assert second.display_brightness == 0
        assert first._brightness_cache.bytes_used > 0
        first.hideEvent(QHideEvent())
        assert first.display_brightness == 50
        assert first._brightness_cache.bytes_used == 0
        first.shutdown()
        assert first.display_brightness == 0
        assert first._brightness_cache.bytes_used == 0
    finally:
        first.shutdown()
        second.shutdown()
        first.close()
        second.close()


def test_scale_export_preview_renders_raw_native_pixels_and_restores_brightness():
    widget = canvas()
    try:
        widget._render_frame = frame(widget._image)
        widget.set_display_brightness(50)
        widget._scale_overlay_preview = SimpleNamespace(is_visible=lambda doc: True)
        assert paint(widget).pixelColor(160, 120).red() == 64
        widget._scale_overlay_preview = SimpleNamespace(is_visible=lambda doc: False)
        assert paint(widget).pixelColor(160, 120).red() == brightness_lut(50)[64]
    finally:
        widget.shutdown()
        widget.close()


def test_progressive_coverage_maps_real_pixels_only_and_keeps_loading_background():
    widget = canvas()
    try:
        progressive = image(230)
        painter = QPainter(progressive)
        painter.fillRect(QRect(0, 0, 160, 240), QColor(64, 64, 64))
        painter.end()
        widget._coarse_render_frame = replace(frame(progressive), complete=False,
            quality='coarse', coverage_rects=((0., 0., 160., 240.),))
        widget.set_display_brightness(50)
        output = paint(widget)
        assert output.pixelColor(80, 120).red() == brightness_lut(50)[64]
        assert output.pixelColor(240, 120).red() == 230
        widget._coarse_render_frame = replace(frame(image(230)), quality='placeholder')
        assert paint(widget).pixelColor(160, 120).red() == 230
    finally:
        widget.shutdown()
        widget.close()


def test_deferred_remount_resets_brightness_without_persisting_it():
    widget = canvas()
    try:
        widget.set_display_brightness(50)
        document = ImageDocument(id='new', path='/tmp/new.fdmslide',
            image_size=(320, 240), document_kind='digital_slide')
        metadata = deepcopy(document.metadata)
        manifest = DigitalSlideManifest(version=1, width=320, height=240,
            viewport_width=320, viewport_height=240, focus_levels=[0])
        store = SimpleNamespace(read_manifest=lambda: manifest)
        widget.set_slide_document(document, store, defer_rendering=True)
        assert widget.display_brightness == 0
        assert document.metadata == metadata
        assert widget._brightness_cache.bytes_used == 0
    finally:
        widget.shutdown()
        widget.close()


@pytest.mark.parametrize('brightness,pending,expected', [
    (0, False, '原始视窗导出范围'),
    (50, False, '导出预览 · 原始亮度'),
    (50, True, '等待当前焦层原始像素…'),
])
def test_export_brightness_notice_reuses_existing_label_without_an_extra_overlay(brightness, pending, expected):
    from unittest.mock import Mock
    from fdm.ui.scale_overlay_editor import ScaleOverlayPreviewController

    widget = canvas()
    try:
        widget.set_display_brightness(brightness)
        preview = SimpleNamespace(
            is_visible=lambda doc: True,
            current_layout=lambda canvas: (SimpleNamespace(bounds=(0, 0, 20, 10)), ''),
            native_ready=lambda canvas: not pending,
            _native_status={}, _refresh_timer=Mock(), _painted_bounds={}, editing=False,
        )
        painter = Mock()
        with patch('fdm.ui.scale_overlay_editor.paint_scale_overlay'):
            ScaleOverlayPreviewController.paint(preview, widget, painter)
        painter.drawText.assert_called_once()
        rect, message = painter.drawText.call_args.args
        assert message == expected
        native = widget.native_viewport_rect()
        origin = widget.image_to_widget(Point(native.x(), native.y()))
        assert rect.topLeft().x() == origin.x() + 6
        assert rect.topLeft().y() == origin.y() + 6
        assert rect.height() == 28
    finally:
        widget.shutdown()
        widget.close()
