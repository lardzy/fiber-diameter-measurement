"""Exercise zoom, cache, culling and persisted settings together."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import QPointF, QRectF
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import QApplication

from fdm.geometry import Line, Point
from fdm.models import ImageDocument, Measurement, OverlayTextSizeSpace
from fdm.settings import AppSettings, MeasurementLabelStyleSettings
from fdm.ui import rendering
from fdm.ui.canvas import DocumentCanvas
from fdm.ui.dialogs import SettingsDialog


@pytest.fixture(scope="module", autouse=True)
def application():
    app = QApplication.instance() or QApplication([])
    yield app


def settings(mode):
    return AppSettings(
        measurement_text_size_space=mode,
        length_measurement_label_style=MeasurementLabelStyleSettings(
            enabled=True, font_size=24, decimals=0,
        ),
        area_measurement_label_style=MeasurementLabelStyleSettings(
            enabled=True, font_size=24, decimals=0,
        ),
        show_count_numbers=True,
        count_number_font_size=24,
    )


def measurement(kind):
    item = Measurement(
        id="object", image_id="image", fiber_group_id=None,
        mode="manual", measurement_kind=kind,
        line_px=Line(Point(550, 450), Point(650, 450)) if kind == "line" else None,
        polyline_px=[Point(550, 450), Point(600, 440), Point(650, 450)] if kind == "polyline" else [],
        polygon_px=[Point(550, 400), Point(650, 400), Point(650, 500), Point(550, 500)] if kind == "area" else [],
        point_px=Point(600, 450) if kind == "count" else None,
    )
    item.recalculate(None)
    return item


def alpha_bounds(image):
    pixels = np.frombuffer(image.constBits(), np.uint8).reshape(image.height(), image.bytesPerLine())
    ys, xs = np.nonzero(pixels[:, 3:image.width() * 4:4])
    assert len(xs), "label vanished while zooming or panning"
    return QRectF(float(xs.min()), float(ys.min()), float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1))


def render_label(kind, mode, zoom, *, sprite=True):
    item = measurement(kind)
    document = ImageDocument(id="image", path="test.png", image_size=(1200, 900), measurements=[item])
    config = settings(mode)
    mapper = lambda point: QPointF(600 + (point.x - 600) * zoom, 450 + (point.y - 450) * zoom)
    image = QImage(1200, 900, QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(0)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
    try:
        if kind == "line":
            rendering.draw_measurement_label(painter, item, document, config, mapper(item.line_px.start), mapper(item.line_px.end), use_sprite_cache=sprite)
        elif kind == "area":
            rendering.draw_area_measurement_label(painter, item, document, config, mapper(Point(600, 450)), image_to_output_scale=zoom, use_sprite_cache=sprite)
        elif kind == "polyline":
            rendering.draw_polyline_measurement_label(painter, item, document, config, [mapper(point) for point in item.polyline_px], mapper, use_sprite_cache=sprite)
        else:
            rendering._draw_count_number_labels(painter, [(mapper(item.point_px), 7)], config, endpoint_radius=4, measurement=item, image_to_output_scale=zoom, use_sprite_cache=sprite)
    finally:
        painter.end()
    return image, item, document, mapper


@pytest.mark.parametrize("kind", ["line", "area", "polyline", "count"])
@pytest.mark.parametrize("mode", [OverlayTextSizeSpace.IMAGE_PX, OverlayTextSizeSpace.SCREEN_PX])
@pytest.mark.parametrize("zoom", [0.25, 1.0, 3.0])
def test_zoom_changes_image_fonts_but_keeps_screen_fonts_readable(kind, mode, zoom):
    reference = alpha_bounds(render_label(kind, mode, 1.0)[0])
    image, item, document, mapper = render_label(kind, mode, zoom)
    actual = alpha_bounds(image)
    factor = zoom if mode == OverlayTextSizeSpace.IMAGE_PX else 1.0
    # Unbacked count glyphs have hinted ink edges; at 3x, a one-pixel edge
    # difference in the 1x reference can contribute three output pixels.
    tolerance = 1 + max(1, factor)
    assert actual.width() == pytest.approx(reference.width() * factor, abs=tolerance)
    assert actual.height() == pytest.approx(reference.height() * factor, abs=tolerance)
    image_bounds = rendering.measurement_label_image_bounds(
        item, document, settings(mode), mapper, count_number=7, exact_area=True,
    )
    output_bounds = QRectF(mapper(Point(image_bounds.left(), image_bounds.top())), mapper(Point(image_bounds.right(), image_bounds.bottom())))
    assert output_bounds.adjusted(-3, -3, 3, 3).contains(actual)


@pytest.mark.parametrize("mode", [OverlayTextSizeSpace.IMAGE_PX, OverlayTextSizeSpace.SCREEN_PX])
@pytest.mark.parametrize("zoom", [0.25, 1.0, 3.0])
def test_passive_area_command_matches_direct_label_layout(mode, zoom):
    direct, item, document, _ = render_label("area", mode, zoom)
    rendering.area_derived_geometry_service.centroid(item)
    command = rendering.build_passive_area_overlay_command(
        document, item, settings(mode), zoom=zoom, line_width=2,
        show_fill=False, sprite_device_pixel_ratio=1.0,
    )
    label = command.label
    cached = QImage(direct.size(), direct.format())
    cached.fill(0)
    painter = QPainter(cached)
    try:
        painter.drawImage(label.top_left + QPointF(600 * (1 - zoom), 450 * (1 - zoom)), label.image)
    finally:
        painter.end()
    first, second = alpha_bounds(direct), alpha_bounds(cached)
    for value, expected in zip(first.getRect(), second.getRect()):
        assert value == pytest.approx(expected, abs=1.0)


def test_both_settings_offer_the_same_two_modes_and_default_to_image_pixels():
    config = AppSettings.from_dict({})
    assert config.measurement_text_size_space == config.text_size_space == OverlayTextSizeSpace.IMAGE_PX
    dialog = SettingsDialog(config, document=None)
    try:
        for combo in (dialog._measurement_text_size_space_combo, dialog._text_size_space_combo):
            assert [(combo.itemText(i), combo.itemData(i)) for i in range(combo.count())] == list(OverlayTextSizeSpace.DISPLAY_ITEMS)
            combo.setCurrentIndex(1)
        saved = AppSettings.from_dict(dialog.app_settings().to_dict())
        assert saved.measurement_text_size_space == saved.text_size_space == OverlayTextSizeSpace.SCREEN_PX
        assert AppSettings.from_dict({"measurement_text_size_space": "bad"}).measurement_text_size_space == OverlayTextSizeSpace.IMAGE_PX
    finally:
        dialog.close()


def test_switching_measurement_mode_invalidates_cached_canvas_labels():
    canvas = DocumentCanvas()
    try:
        before = canvas._overlay_style_generation
        canvas.set_settings(AppSettings(measurement_text_size_space=OverlayTextSizeSpace.SCREEN_PX))
        assert canvas._overlay_style_generation == before + 1
    finally:
        canvas.close()
