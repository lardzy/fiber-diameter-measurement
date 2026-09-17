"""Digital-slide preview labels must use the current view's display mode."""

from dataclasses import replace

import numpy as np
import pytest

from fdm.geometry import Line, Point
from fdm.models import Measurement, OverlayTextSizeSpace
from fdm.settings import MeasurementLabelStyleSettings
from test_canvas_overlay_handoff import scene as scene
from test_digital_slide_count_preview import complete_preview, preview_pixels


@pytest.mark.parametrize("scene", ["digital"], indirect=True)
@pytest.mark.parametrize("mode", [OverlayTextSizeSpace.IMAGE_PX, OverlayTextSizeSpace.SCREEN_PX])
@pytest.mark.parametrize("zoom", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("dpr", [1.0, 2.0])
@pytest.mark.parametrize("kind", ["line", "polyline", "area"])
def test_digital_preview_label_matches_current_zoom_before_exact_tiles_arrive(
    scene, monkeypatch, mode, zoom, dpr, kind,
):
    canvas, document, *_ = scene
    monkeypatch.setattr(canvas, "devicePixelRatioF", lambda: dpr)
    document.measurements = [
        Measurement(
            id=f"line-{index}", image_id=document.id, fiber_group_id=None,
            mode="manual", measurement_kind=kind if index == 0 else "line",
            line_px=(
                Line(Point(8450, 4350), Point(8550, 4350))
                if index == 0 else Line(Point(50, 50 + index), Point(60, 50 + index))
            ),
            polyline_px=[Point(8450, 4350), Point(8500, 4330), Point(8550, 4350)] if index == 0 and kind == "polyline" else [],
            polygon_px=[Point(8450, 4300), Point(8550, 4300), Point(8550, 4400), Point(8450, 4400)] if index == 0 and kind == "area" else [],
        )
        for index in range(70)
    ]
    for item in document.measurements:
        item.recalculate(None)
    document.view_state.selected_measurement_id = None
    document.mark_measurement_geometry_changed()
    document.mark_session_dirty()
    canvas.set_settings(replace(
        canvas._settings,
        measurement_text_size_space=mode,
        default_measurement_color="#000000",
        length_measurement_label_style=MeasurementLabelStyleSettings(
            enabled=True, font_size=24, color="#FF00FF", decimals=0,
            background_enabled=False,
        ),
        area_measurement_label_style=MeasurementLabelStyleSettings(
            enabled=True, font_size=24, color="#FF00FF", decimals=0,
            background_enabled=False,
        ),
    ))
    canvas._zoom = zoom
    canvas._pan = Point(600 - 8500 * zoom, 400 - 4350 * zoom)
    canvas._sync_overlay_visual_state()
    complete_preview(scene)
    actual = preview_pixels(canvas)
    expected = preview_pixels(canvas, direct=True)

    def text_mask(pixels):
        # Qt ARGB32 stores BGRA on the test platforms. Geometry is black;
        # isolate the magenta text from the coarse geometry preview.
        channels = pixels.astype(np.int16)
        # Subtract the neutral outline underneath antialiased glyph edges.
        return ((channels[:, :, 0] - channels[:, :, 1]) > 100) & ((channels[:, :, 2] - channels[:, :, 1]) > 100)

    assert text_mask(expected).any()
    np.testing.assert_array_equal(text_mask(actual), text_mask(expected))


@pytest.mark.parametrize("mode", [OverlayTextSizeSpace.IMAGE_PX, OverlayTextSizeSpace.SCREEN_PX])
def test_digital_native_export_keeps_font_pixels_and_frozen_focus_at_every_zoom(tmp_path, mode):
    from PySide6.QtGui import QColor, QImage

    from fdm.models import ImageDocument, OverlayAnnotation, OverlayTextLayoutSpec
    from fdm.services.digital_slide_store import DigitalSlideManifest, DigitalSlideStore, DigitalSlideTile
    from fdm.services.export_service import ExportImageRenderMode, ExportRenderContext
    from fdm.settings import AppSettings
    from fdm.ui.main_window import MainWindow

    path = tmp_path / "text-export.fdmslide"
    manifest = DigitalSlideManifest(
        version=1, width=16384, height=8192, viewport_width=240,
        viewport_height=180, focus_levels=[0, 1],
    )
    colors = ("#203040", "#405060")
    store = DigitalSlideStore.create(path, manifest)
    try:
        for focus, color in enumerate(colors):
            tile = QImage(240, 180, QImage.Format.Format_RGB32)
            tile.fill(QColor(color))
            store.write_tile(DigitalSlideTile(
                z_index=focus, x=8192, y=4096, width=240, height=180,
            ), tile)
    finally:
        store.close()
    document = ImageDocument(
        id="slide-font-export", path=str(path), image_size=(16384, 8192),
        document_kind="digital_slide",
        metadata={"digital_slide": {"viewport_origin": [8192, 4096], "focus_index": 0}},
        measurements=[Measurement(
            id="line", image_id="slide-font-export", fiber_group_id=None,
            mode="manual", line_px=Line(Point(8232, 4206), Point(8352, 4206)),
        )],
        overlay_annotations=[OverlayAnnotation(
            id="text", image_id="slide-font-export", kind="text", content="Fiber A",
            anchor_px=Point(8312, 4156),
            text_layout=OverlayTextLayoutSpec(size_space=mode, image_font_size_px=18),
        )],
    )
    document.initialize_runtime_state()
    document.measurements[0].recalculate(None)
    window = MainWindow()
    try:
        window._app_settings = AppSettings(
            measurement_text_size_space=mode, text_color="#FFFFFF",
            length_measurement_label_style=MeasurementLabelStyleSettings(
                enabled=True, font_size=18, color="#FF00FF", decimals=0,
            ),
        )
        window._add_digital_slide_document_from_path(path, document=document)
        canvas = window._canvases[document.id]
        canvas.set_settings(window._app_settings)
        for focus, color in enumerate(colors):
            reference = None
            context = ExportRenderContext(
                document_id=document.id, render_mode=ExportImageRenderMode.CURRENT_VIEWPORT,
                focus_index=focus, origin_x=8192, origin_y=4096,
                viewport_width=240, viewport_height=180,
            )
            for zoom in (0.25, 2.0, 8.0):
                canvas._zoom = zoom
                output = tmp_path / f"focus-{focus}-zoom-{zoom}.png"
                result = window._render_overlay_image(
                    document, output, include_measurements=True, include_scale=False,
                    render_mode=ExportImageRenderMode.CURRENT_VIEWPORT, render_context=context,
                )
                assert (result.width, result.height) == (240, 180)
                exported = QImage(str(output))
                assert exported.pixelColor(0, 0).name() == color
                if reference is None:
                    reference = exported
                else:
                    assert exported == reference
                assert any(
                    exported.pixelColor(x, y).red() > 200 and exported.pixelColor(x, y).green() > 200
                    for x in range(80, 160) for y in range(45, 75)
                ), "global text anchor was not translated into the native viewport"
                assert any(
                    exported.pixelColor(x, y).red() > 200 and exported.pixelColor(x, y).blue() > 200
                    and exported.pixelColor(x, y).green() < 50
                    for x in range(40, 180) for y in range(110, 160)
                ), "measurement label was missing from the native viewport export"
        assert document.overlay_annotations[0].text_layout.image_font_size_px == 18
    finally:
        window._reset_workspace()
        window.close()
