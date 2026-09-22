"""Scale geometry is measured between outer edges, not stroke centerlines."""

import math
from dataclasses import replace

import numpy as np
import pytest
from PySide6.QtCore import QRectF
from PySide6.QtGui import QColor, QImage, QPainter

from fdm.scale_overlay import ScaleOverlaySpec
from fdm.ui.scale_overlay_rendering import (
    layout_scale_overlay,
    paint_scale_overlay,
    scale_bar_path,
)

STYLES = ("line", "ticks", "bar", "ticks_up", "ticks_down", "divisions")


def _layout(style="ticks", stroke=2, length=100, origin=(0, 0)):
    spec = ScaleOverlaySpec(
        style=style,
        length_mode="custom",
        length=length,
        unit="px",
        line_width=stroke,
        color="#FF0000",
        text_color="#000000",
    )
    layout = layout_scale_overlay(spec, (*origin, 400.0, 200.0))
    # Deliberately align the horizontal filled rectangle with pixel edges.
    y = origin[1] + 100 + layout.stroke / 2
    return replace(layout, start=(origin[0] + 50, y), end=(origin[0] + 50 + length, y))


def _render(layout, dpr=1.0, zoom=1.0, full=False):
    image = QImage(
        math.ceil(400 * dpr * zoom),
        math.ceil(200 * dpr * zoom),
        QImage.Format.Format_ARGB32_Premultiplied,
    )
    image.setDevicePixelRatio(dpr)
    image.fill(0)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.scale(zoom, zoom)
    if full:
        painter.translate(-layout.target[0], -layout.target[1])
        paint_scale_overlay(painter, layout)
    else:
        painter.fillPath(scale_bar_path(layout), QColor(layout.spec.color))
    painter.end()
    return image


def _rgba(image):
    image = image.convertToFormat(QImage.Format.Format_RGBA8888)
    return (
        np.frombuffer(image.bits(), dtype=np.uint8)
        .reshape(image.height(), image.width(), 4)
        .copy()
    )


@pytest.mark.parametrize("style", STYLES)
@pytest.mark.parametrize("stroke", (1, 2, 3, 4, 5, 8))
def test_integer_span_occupies_exactly_100_columns_without_pen_overhang(style, stroke):
    layout = _layout(style, stroke)
    pixels = _rgba(_render(layout))
    columns = np.flatnonzero(pixels[:, :, 3].max(axis=0))
    assert np.array_equal(columns, np.arange(50, 150))
    # A continuous, opaque baseline covers both endpoint pixels once.
    assert np.all(pixels[100, 50:150, 3] == 255)
    assert np.all(pixels[:, (49, 150), 3] == 0)
    rect = scale_bar_path(layout).boundingRect()
    assert rect.left() == 50
    assert rect.right() == 150
    assert rect.width() == layout.end[0] - layout.start[0] == 100


@pytest.mark.parametrize("style", STYLES)
@pytest.mark.parametrize("length", (0.125, 0.5, 1, 2, 100.375))
def test_short_or_fractional_span_does_not_expand_to_line_width(style, length):
    layout = _layout(style, stroke=8, length=length)
    bounds = scale_bar_path(layout).boundingRect()
    assert bounds.left() == pytest.approx(50)
    assert bounds.right() == pytest.approx(50 + length)
    assert bounds.width() == pytest.approx(length)


@pytest.mark.parametrize("style", STYLES)
@pytest.mark.parametrize("dpr", (1.0, 1.5, 2.0))
@pytest.mark.parametrize("zoom", (0.625, 1.0, 1.375))
def test_fractional_transform_preserves_geometry_and_pixel_coverage(style, dpr, zoom):
    layout = _layout(style, stroke=4, length=100.375)
    layout = replace(layout, start=(50.25, 102.0), end=(150.625, 102.0))
    scale = dpr * zoom
    pixels = _rgba(_render(layout, dpr=dpr, zoom=zoom))
    coverage = pixels[:, :, 3].max(axis=0)
    occupied = np.flatnonzero(coverage)
    left, right = 50.25 * scale, 150.625 * scale
    # Antialiasing may cover two boundary pixels; it must never add a column
    # completely outside the mathematically defined interval.
    assert occupied[0] >= math.floor(left)
    assert occupied[-1] < math.ceil(right)
    assert coverage[: math.floor(left)].sum() == 0
    assert coverage[math.ceil(right) :].sum() == 0
    # Sample the interior horizontal baseline, with full vertical coverage.
    row = pixels[round(102.0 * scale - 0.5), :, 3] / 255.0
    expected_coverage = np.maximum(
        0,
        np.minimum(np.arange(len(row)) + 1, right)
        - np.maximum(np.arange(len(row)), left),
    )
    # Qt's raster engine approximates subpixel area on a finite coverage grid;
    # allow 1/32 of a pixel in coverage, never a whole extra endpoint pixel.
    assert np.max(np.abs(row - expected_coverage)) <= 1 / 32
    assert row.sum() == pytest.approx(100.375 * scale, abs=1 / 32)
    plain = replace(layout, spec=replace(layout.spec, style="line"), tick_height=0)
    plain_row = _rgba(_render(plain, dpr=dpr, zoom=zoom))[
        round(102.0 * scale - 0.5), :, 3
    ]
    assert np.array_equal(pixels[round(102.0 * scale - 0.5), :, 3], plain_row)


@pytest.mark.parametrize("style", STYLES)
@pytest.mark.parametrize("dpr", (1.0, 1.5, 2.0))
def test_large_slide_origin_matches_local_full_renderer_pixel_for_pixel(style, dpr):
    spec = ScaleOverlaySpec(
        style=style, length_mode="custom", length=25.125, line_width=3
    )
    images = []
    for x, y in ((0.0, 0.0), (8192.0, 4096.0), (137.0, 83.0)):
        layout = layout_scale_overlay(spec, (x, y, 400.0, 200.0), 4.0, "um")
        assert layout.end[0] - layout.start[0] == pytest.approx(100.5)
        images.append(_render(layout, dpr=dpr, zoom=1.25, full=True))
    assert images[0] == images[1] == images[2]


@pytest.mark.parametrize("style", STYLES)
@pytest.mark.parametrize("text_position", ("above", "below"))
def test_path_and_label_are_both_inside_complete_layout_bounds(style, text_position):
    spec = ScaleOverlaySpec(style=style, text_position=text_position, line_width=7)
    layout = layout_scale_overlay(spec, (8192.0, 4096.0, 640.0, 480.0), 4.0, "um")
    bounds = QRectF(*layout.bounds)
    local_bounds = QRectF(bounds)
    local_bounds.translate(-8192, -4096)
    assert local_bounds.contains(scale_bar_path(layout).boundingRect())
    assert bounds.contains(QRectF(*layout.text_rect))


def test_four_equal_divisions_share_total_span_without_cumulative_rounding():
    layout = _layout("divisions", stroke=1, length=101)
    path = scale_bar_path(layout)
    # Inspect above the baseline: full-height middle mark and shorter quarters.
    y = layout.start[1] - layout.tick_height * 0.4
    for fraction in (0.25, 0.5, 0.75):
        x = layout.start[0] + 101 * fraction
        assert path.contains(QRectF(x - 0.49, y, 0.98, 0.1))
        assert not path.contains(QRectF(x + 0.51, y, 0.01, 0.1))


@pytest.mark.parametrize("style", STYLES)
@pytest.mark.parametrize("dpr", (1.0, 1.5, 2.0))
@pytest.mark.parametrize(
    "font_size,stroke,relative_y,origin,zoom",
    (
        (24, 3, 0.5, (445.0, 67.0), 1.0),
        (24, 3, 0.5, (445.0, 282.0), 1.0),
        (18, 1, 0.37, (8192.0, 4096.0), 0.625),
        (27, 2.5, 0.13, (137.25, 83.375), 1.375),
    ),
)
def test_actual_layout_retains_entire_baseline_with_fractional_y(
    style, dpr, font_size, stroke, relative_y, origin, zoom
):
    # Do not replace start/end with a convenient integer baseline. Real font
    # metrics, manual placement and one-sided tick heights create fractional
    # shared edges that previously made QPainterPath.simplified drop the bar.
    spec = ScaleOverlaySpec(
        style=style,
        length_mode="custom",
        length=50,
        unit="um",
        line_width=stroke,
        font_mode="custom",
        font_size=font_size,
        position="manual",
        relative_x=0.5,
        relative_y=relative_y,
    )
    layout = layout_scale_overlay(spec, (*origin, 370.0, 130.0), 4.0, "um")
    left = layout.start[0] - origin[0]
    right = layout.end[0] - origin[0]
    baseline = layout.start[1] - origin[1]
    path = scale_bar_path(layout)
    assert path.contains(
        QRectF(
            left + 0.01,
            baseline - layout.stroke / 2 + 0.01,
            right - left - 0.02,
            layout.stroke - 0.02,
        )
    )
    pixels = _rgba(_render(layout, dpr=dpr, zoom=zoom, full=True))
    scale = dpr * zoom
    row = round(baseline * scale - 0.5)
    actual = pixels[row, :, 3] / 255
    vertical_coverage = max(
        0,
        min(row + 1, (baseline + layout.stroke / 2) * scale)
        - max(row, (baseline - layout.stroke / 2) * scale),
    )
    columns = np.arange(len(actual))
    minimum_coverage = vertical_coverage * np.maximum(
        0, np.minimum(columns + 1, right * scale) - np.maximum(columns, left * scale)
    )
    assert np.all(actual >= minimum_coverage - 1 / 32)
    assert actual[: math.floor(left * scale)].sum() == 0
    assert actual[math.ceil(right * scale) :].sum() == 0


def test_red_ticks_bottom_right_are_new_spec_defaults():
    spec = ScaleOverlaySpec()
    assert spec.color == spec.text_color == "#FF0000"
    assert spec.style == "ticks"
    assert spec.position == "bottom_right"


def test_explicit_legacy_style_and_colors_are_preserved():
    from fdm.settings import AppSettings

    old = AppSettings(
        scale_overlay_color="#F4F1DE",
        scale_overlay_text_color="#00FF00",
        scale_overlay_style="bar",
        scale_overlay_placement_mode="top_left",
    )
    spec = ScaleOverlaySpec.from_legacy_settings(old)
    assert spec.color == "#F4F1DE"
    assert spec.text_color == "#00FF00"
    assert spec.style == "bar"
    assert spec.position == "top_left"
