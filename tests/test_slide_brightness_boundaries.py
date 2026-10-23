"""Brightness must never become scientific input, persisted data or export pixels."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import hashlib
import time

import pytest
from PySide6.QtGui import QColor, QImage

from fdm.geometry import Line, Point
from fdm.models import Calibration, ImageDocument, Measurement, OverlayAnnotation
from fdm.services.digital_slide_store import DigitalSlideManifest, DigitalSlideStore, DigitalSlideTile
from fdm.services.export_service import ExportImageRenderMode, ExportRenderContext
from fdm.services.segmentation_source import digital_slide_segmentation_snapshot
from fdm.ui.main_window import MainWindow


def _settle(app, predicate=lambda: True, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return
        time.sleep(.005)
    assert predicate(), "synthetic slide renderer did not become ready"


def _stable_view(app, window, canvas, timeout=10):
    """Wait for native window layout and initial-fit work before a baseline.

    Pixel readiness alone can precede a queued WM resize on Windows. A grab
    flushes Qt's lazy widget layout; then both geometry and view identity must
    stay ready and unchanged through an event-loop quiet interval.
    """
    canvas.grab()
    deadline = time.monotonic() + timeout
    previous = None
    stable_since = time.monotonic()
    while time.monotonic() < deadline:
        app.processEvents()
        identity = (window.width(), window.height(), canvas.width(), canvas.height(),
            canvas._view_generation, canvas.view_zoom(), canvas.viewport_origin(),
            canvas.focus_index(), canvas.pixel_work_enabled(), canvas._initial_fit_pending)
        now = time.monotonic()
        if identity != previous or not canvas.pixel_work_enabled() or canvas._initial_fit_pending:
            previous, stable_since = identity, now
        elif now - stable_since >= .15:
            return identity
        time.sleep(.005)
    raise AssertionError(f"native slide layout did not settle: {previous!r}")


@pytest.fixture
def slide_window(tmp_path, monkeypatch, desktop_application):
    from fdm.ui import digital_slide_canvas

    monkeypatch.setattr(digital_slide_canvas, "digital_slide_render_cache_directory", lambda: tmp_path / "cache")
    path = tmp_path / "original.fdmslide"
    store = DigitalSlideStore.create(path, DigitalSlideManifest(1, 480, 360, 160, 120, [-4, -2, 0, 2, 4]))
    try:
        for z in range(5):
            for row in range(3):
                for column in range(3):
                    tile = QImage(160, 120, QImage.Format.Format_RGB32)
                    tile.fill(QColor(25 + z * 20, 60 + row * 20, 80 + column * 20))
                    store.write_tile(DigitalSlideTile(z, column * 160, row * 120, 160, 120), tile)
    finally:
        store.close()
    source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    doc = ImageDocument(id="brightness-boundary", path=str(path), image_size=(480, 360),
        document_kind="digital_slide", calibration=Calibration("image_scale", 2, "um", "synthetic"),
        metadata={"digital_slide": {"viewport_origin": [160, 120], "focus_index": 2}},
        measurements=[Measurement(id="line", image_id="brightness-boundary", fiber_group_id=None,
            mode="manual", line_px=Line(Point(180, 150), Point(260, 150)))],
        overlay_annotations=[OverlayAnnotation(id="text", image_id="brightness-boundary", kind="text",
            content="Original", anchor_px=Point(190, 180))])
    doc.initialize_runtime_state()
    doc.measurements[0].recalculate(doc.calibration)
    window = MainWindow()
    try:
        window.resize(1000, 700)
        window.show()
        window._add_digital_slide_document_from_path(path, document=doc)
        canvas = window._canvases[doc.id]
        _settle(desktop_application, canvas.pixel_work_enabled)
        _stable_view(desktop_application, window, canvas)
        yield window, canvas, doc, path, source_hash
    finally:
        window._reset_workspace()
        window.close()


@pytest.mark.parametrize("brightness", [-100, -45, 40, 100, 0])
def test_brightness_preserves_file_model_processing_and_segmentation(slide_window, desktop_application, brightness):
    window, canvas, doc, path, source_hash = slide_window
    before = deepcopy(doc.to_dict())
    snapshot = deepcopy(doc.snapshot_state())
    flags = asdict(doc.dirty_flags)
    image = canvas._image.copy()
    ready = canvas.pixel_work_enabled()
    generation = canvas._view_generation
    processing_before = window._create_processing_source_context()[0].plane.sha256()

    def segmentation():
        return digital_slide_segmentation_snapshot(doc, window._slide_stores[doc.id],
            origin_px=canvas.viewport_origin(), width=160, height=120, focus_index=canvas.focus_index())

    segment_before = segmentation()
    canvas.set_display_brightness(brightness)
    # Attribute this invariant to the setter itself, before a later native
    # resize/initial-fit event can legitimately create a new view generation.
    assert canvas._view_generation == generation
    assert canvas.pixel_work_enabled() == ready
    canvas.grab()
    _settle(desktop_application)
    assert canvas._image == image
    assert doc.to_dict() == before
    assert doc.snapshot_state() == snapshot
    assert asdict(doc.dirty_flags) == flags
    assert window._create_processing_source_context()[0].plane.sha256() == processing_before
    assert segmentation().image == segment_before.image
    assert segmentation().source_version == segment_before.source_version
    assert hashlib.sha256(path.read_bytes()).hexdigest() == source_hash


@pytest.mark.parametrize("measurements,scale", [(False, False), (True, False), (True, True)])
def test_native_exports_ignore_view_brightness(slide_window, tmp_path, measurements, scale):
    window, canvas, doc, *_ = slide_window
    context = ExportRenderContext(document_id=doc.id, render_mode=ExportImageRenderMode.CURRENT_VIEWPORT,
        focus_index=2, origin_x=160, origin_y=120, viewport_width=160, viewport_height=120)
    images = []
    for value in (0, -100, 100, 37):
        canvas.set_display_brightness(value)
        canvas.grab()
        output = tmp_path / f"export-{value}.png"
        window._render_overlay_image(doc, output, include_measurements=measurements, include_scale=scale,
            include_annotations=measurements, render_mode=ExportImageRenderMode.CURRENT_VIEWPORT, render_context=context)
        images.append(QImage(str(output)))
    assert not images[0].isNull()
    assert all(image == images[0] for image in images[1:])


def test_brightness_survives_focus_and_navigation_without_polluting_storage(slide_window, desktop_application):
    window, canvas, doc, path, source_hash = slide_window
    canvas.set_display_brightness(63)
    for focus in (4, 0, 3, 2):
        canvas.set_focus_index(focus)
        _settle(desktop_application, canvas.pixel_work_enabled)
        assert canvas.display_brightness == 63
        assert canvas._image.pixelColor(20, 20).red() == 25 + focus * 20
    for dx, dy in ((160, 0), (-40, 0), (0, 30), (0, -30), (-120, 0)):
        canvas.move_viewport_by(dx, dy)
        _settle(desktop_application, canvas.pixel_work_enabled)
        assert canvas.display_brightness == 63
    canvas.fit_to_view()
    desktop_application.processEvents()
    assert canvas.display_brightness == 63
    canvas.fit_native_viewport()
    _settle(desktop_application, canvas.pixel_work_enabled)
    assert canvas.display_brightness == 63
    assert "display_brightness" not in str(doc.to_dict())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == source_hash
