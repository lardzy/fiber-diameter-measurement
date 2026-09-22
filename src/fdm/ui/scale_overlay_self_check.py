"""Small real-raster probe used in the packaged overlay worker."""

import math
from dataclasses import replace

from PySide6.QtCore import QRectF
from PySide6.QtGui import QColor, QImage, QPainter

from fdm.scale_overlay import ScaleOverlaySpec
from fdm.ui.scale_overlay_rendering import (
    layout_scale_overlay,
    paint_scale_overlay,
    scale_bar_path,
)


def _check_scale_bar_pixels(cases):
    """Check actual raster boundaries; equal geometry alone misses pen overhang."""
    styles = ("line", "ticks", "bar", "ticks_up", "ticks_down", "divisions")
    for style in styles:
        for width in (1.0, 2.0, 3.0, 4.0):
            spec = ScaleOverlaySpec(
                style=style,
                length_mode="custom",
                unit="px",
                length=100,
                line_width=width,
            )
            layout = layout_scale_overlay(spec, (0.0, 0.0, 640.0, 480.0))
            baseline = 50 + layout.stroke / 2
            layout = replace(layout, start=(40.0, baseline), end=(140.0, baseline))
            path = scale_bar_path(layout)
            if path.boundingRect().left() != 40 or path.boundingRect().right() != 140:
                raise RuntimeError(f"scale outer-edge geometry: {style}/{width}")
            image = QImage(180, 100, QImage.Format.Format_ARGB32_Premultiplied)
            image.fill(0)
            painter = QPainter(image)
            try:
                paint_scale_overlay(painter, layout)
            finally:
                painter.end()
            if any(image.pixelColor(x, 50).alpha() != 255 for x in range(40, 140)):
                raise RuntimeError(f"scale baseline pixel coverage: {style}/{width}")
            if any(
                image.pixelColor(x, y).alpha()
                for x in (39, 140)
                for y in range(image.height())
            ):
                raise RuntimeError(
                    f"scale endpoint extends outside span: {style}/{width}"
                )
    cases["endpoint_pixels"] = True
    cases["style_variants"] = True

    spec = ScaleOverlaySpec(
        style="divisions", length_mode="custom", length=100.375, unit="px", line_width=4
    )
    layout = layout_scale_overlay(spec, (0.0, 0.0, 640.0, 480.0))
    layout = replace(layout, start=(40.25, 52.0), end=(140.625, 52.0))
    for dpr in (1.0, 1.5, 2.0):
        zoom = 1.375
        scale = dpr * zoom
        image = QImage(
            math.ceil(200 * scale),
            math.ceil(100 * scale),
            QImage.Format.Format_ARGB32_Premultiplied,
        )
        image.setDevicePixelRatio(dpr)
        image.fill(0)
        painter = QPainter(image)
        try:
            painter.scale(zoom, zoom)
            paint_scale_overlay(painter, layout)
        finally:
            painter.end()
        left, right = 40.25 * scale, 140.625 * scale
        row = round(52.0 * scale - 0.5)
        coverage = [
            image.pixelColor(x, row).alpha() / 255 for x in range(image.width())
        ]
        expected = [
            max(0.0, min(x + 1, right) - max(x, left)) for x in range(image.width())
        ]
        if (
            max(abs(a - b) for a, b in zip(coverage, expected)) > 1 / 32
            or abs(sum(coverage) - (right - left)) > 1 / 32
            or any(coverage[: math.floor(left)])
            or any(coverage[math.ceil(right) :])
        ):
            raise RuntimeError(f"scale fractional pixel coverage at DPR {dpr}")
    cases["fractional_span_coverage"] = True

    path = scale_bar_path(layout)
    for fraction in (0.25, 0.5, 0.75):
        x = 40.25 + 100.375 * fraction
        if not path.contains(QRectF(x - 1.9, 50.0, 3.8, 0.1)):
            raise RuntimeError("scale equal-division geometry")
    cases["division_geometry"] = True

    # Real font metrics plus one-sided ticks produce fractional shared edges.
    # Exercise the production layout unchanged: integer replacement fixtures
    # alone would miss a polygon-union error that can erase the baseline.
    for style in styles:
        actual_spec = ScaleOverlaySpec(
            style=style,
            length_mode="custom",
            length=50,
            unit="um",
            line_width=3,
            font_mode="custom",
            font_size=24,
            position="manual",
            relative_x=0.5,
            relative_y=0.5,
        )
        actual = layout_scale_overlay(
            actual_spec, (445.0, 67.0, 370.0, 130.0), 4.0, "um"
        )
        local_y = actual.start[1] - actual.target[1]
        local_x = actual.start[0] - actual.target[0]
        bar_width = actual.end[0] - actual.start[0]
        if not scale_bar_path(actual).contains(
            QRectF(
                local_x + 0.01,
                local_y - actual.stroke / 2 + 0.01,
                bar_width - 0.02,
                actual.stroke - 0.02,
            )
        ):
            raise RuntimeError(f"scale actual layout lost its horizontal bar: {style}")
        image = QImage(370, 130, QImage.Format.Format_ARGB32_Premultiplied)
        image.fill(0)
        painter = QPainter(image)
        try:
            painter.translate(-actual.target[0], -actual.target[1])
            paint_scale_overlay(painter, actual)
        finally:
            painter.end()
        row = round(local_y - 0.5)
        if any(
            image.pixelColor(x, row).alpha() != 255
            for x in range(math.ceil(local_x), math.floor(local_x + bar_width))
        ):
            raise RuntimeError(
                f"scale actual layout has a missing baseline pixel: {style}"
            )
    cases["actual_layout_geometry"] = True


def run_scale_overlay_self_check():
    cases = {}
    spec = ScaleOverlaySpec(
        length_mode="custom",
        length=50,
        font_mode="custom",
        font_size=40,
        color="#FF00FF",
        text_color="#FF00FF",
    )
    for unit, length in (
        ("nm", 50000),
        ("um", 50),
        ("mm", 0.05),
        ("cm", 0.005),
        ("m", 0.00005),
    ):
        layout = layout_scale_overlay(
            replace(spec, unit=unit, length=length), (0.0, 0.0, 640.0, 480.0), 4.0, "um"
        )
        if abs(layout.end[0] - layout.start[0] - 200) > 1e-8 or layout.font_size != 40:
            raise RuntimeError(f"scale units/font: {unit}")
        cases[f"unit_{unit}"] = True
    for dpr in (1.0, 1.5, 2.0):
        # A digital-slide global origin and an ordinary cropped view must yield
        # exactly the same local raster as a native image of that target size.
        images = []
        for x, y in ((0.0, 0.0), (8192.0, 4096.0), (137.0, 83.0)):
            layout = layout_scale_overlay(spec, (x, y, 640.0, 480.0), 4.0, "um")
            image = QImage(
                round(640 * dpr),
                round(480 * dpr),
                QImage.Format.Format_ARGB32_Premultiplied,
            )
            image.setDevicePixelRatio(dpr)
            image.fill(QColor("#203040"))
            painter = QPainter(image)
            try:
                painter.translate(-x, -y)
                paint_scale_overlay(painter, layout)
            finally:
                painter.end()
            images.append(image)
        if any(image != images[0] for image in images[1:]):
            raise RuntimeError(f"scale preview/export transform mismatch at DPR {dpr}")
        native = layout_scale_overlay(spec, (0.0, 0.0, 640.0, 480.0), 4.0, "um")
        if (
            images[0]
            .pixelColor(
                round((native.start[0] + native.end[0]) / 2 * dpr),
                round(native.start[1] * dpr),
            )
            .red()
            < 200
        ):
            raise RuntimeError("scale bar did not render")
        cases[f"preview_export_dpr_{dpr:g}"] = True
    layout = layout_scale_overlay(ScaleOverlaySpec(), (0.0, 0.0, 640.0, 480.0))
    cases["uncalibrated_px"] = layout.unit == "px" and layout.label.endswith(" px")
    try:
        layout_scale_overlay(
            replace(spec, length=10000), (0.0, 0.0, 640.0, 480.0), 4.0, "um"
        )
    except ValueError:
        cases["oversized_rejected"] = True
    else:
        raise RuntimeError("oversized scale bar was silently clipped")
    _check_scale_bar_pixels(cases)
    return {"ok": all(cases.values()), "revision": 1, "cases": cases}
