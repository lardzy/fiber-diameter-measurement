from dataclasses import replace
from unittest.mock import patch
from concurrent.futures.process import BrokenProcessPool
from functools import partial
import queue

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPicture, QTransform

from fdm.ui.canvas_overlay_cache import (
    AreaOverlayDrawCommand,
    CanvasOverlayRenderSnapshot,
    CanvasOverlayTileKey,
    PictureOverlayDrawCommand,
    _AreaCommandCentroidCache,
    _CancellationFlag,
    _TileRenderRunnable,
)
from fdm.ui import overlay_process_renderer as renderer


def _initialize_worker_without_console(stderr_kind):
    import io
    import sys

    sys.stdout = sys.__stdout__ = None
    sys.stderr = sys.__stderr__ = None if stderr_kind == "missing" else io.StringIO()
    renderer._initialize_worker()


@pytest.fixture(params=["missing", "redirected"])
def windowed_render_pool(request):
    with (
        patch.object(renderer, "_pool", None),
        patch.object(
            renderer,
            "_initialize_worker",
            partial(_initialize_worker_without_console, request.param),
        ),
    ):
        try:
            yield
        finally:
            renderer.shutdown_overlay_renderer()


def test_windowed_worker_renders_raw_draft_with_global_slide_coordinates(
    desktop_application, windowed_render_pool
):
    rings = tuple(
        np.array(points, dtype=np.float64).tobytes()
        for points in (
            [(8010, 5010), (8090, 5010), (8090, 5090), (8010, 5090)],
            [(8030, 5030), (8070, 5030), (8070, 5070), (8030, 5070)],
        )
    )
    snapshot = CanvasOverlayRenderSnapshot(
        1,
        CanvasOverlayTileKey(200, "segmentation-draft", 1, 1, 0, 0, 0, 0, True),
        logical_tile_size=128,
        area_commands=(
            AreaOverlayDrawCommand(
                None,
                QTransform.fromTranslate(-8000, -5000),
                QColor(52, 211, 153, 72).rgba(),
                QColor("#0B0B0B").rgba(),
                3.2,
                QColor("#34D399").rgba(),
                1.8,
                raw_coordinates=rings,
                geometry_key=(200, "draft", 1),
                stroke_style=Qt.PenStyle.DashLine.value,
                separate_fill=True,
            ),
        ),
    )
    expected, _ = _TileRenderRunnable(
        snapshot,
        _CancellationFlag(),
        0,
        queue.SimpleQueue(),
        _AreaCommandCentroidCache(),
    )._render()

    actual, picture = renderer.render_in_isolated_worker(snapshot)

    assert picture is None
    assert actual == expected
    assert actual.pixelColor(20, 20).alpha() > 0
    assert actual.pixelColor(50, 50).alpha() == 0


def test_windowed_slide_shows_primary_and_subtract_drafts_before_f_commit(
    tmp_path, desktop_application, windowed_render_pool
):
    from time import monotonic, sleep
    from PySide6.QtCore import QEvent
    from PySide6.QtGui import QImage, QKeyEvent
    from fdm.geometry import Point
    from fdm.services.digital_slide_store import (
        DigitalSlideManifest,
        DigitalSlideStore,
        DigitalSlideTile,
    )
    from fdm.ui.draft_preview_cache import DraftPreviewCache
    from fdm.ui.main_window import MainWindow

    path = tmp_path / "windowed-preview.fdmslide"
    store = DigitalSlideStore.create(
        path, DigitalSlideManifest(1, 16000, 12000, 320, 240, [0])
    )
    tile = QImage(320, 240, QImage.Format.Format_RGB32)
    tile.fill(QColor("#58626D"))
    store.write_tile(DigitalSlideTile(0, 8000, 5000, 320, 240), tile)
    store.close()

    cache = DraftPreviewCache(asynchronous=True)
    errors = []
    cache._raster_cache.tileFailed.connect(lambda key, error: errors.append(error))
    window = MainWindow()
    window.resize(1512, 864)
    window.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
    canvas = None

    def wait(predicate):
        deadline = monotonic() + 10
        while monotonic() < deadline:
            desktop_application.processEvents()
            if predicate():
                return
            sleep(0.001)
        assert predicate(), "windowed preview did not become ready"

    def render():
        image = QImage(canvas.size(), QImage.Format.Format_RGB32)
        canvas.render(image)
        return image

    def assert_draft_matches_reference(points):
        render()
        wait(lambda: not cache._requests)
        assert not errors
        actual = render()
        with patch("fdm.ui.canvas.draft_preview_cache", DraftPreviewCache()):
            reference = render()
        for x, y in points:
            position = canvas.image_to_widget(Point(8000 + x, 5000 + y)).toPoint()
            assert actual.pixelColor(position) == reference.pixelColor(position)
        return actual

    def install(mask, rings):
        canvas._magic_segment.request_id += 1
        canvas._magic_segment.pending_stage = (
            canvas.current_magic_segment_operation_mode()
        )
        global_rings = [[Point(8000 + x, 5000 + y) for x, y in ring] for ring in rings]
        canvas.apply_magic_segment_result(
            canvas._magic_segment.request_id,
            mask,
            global_rings[0],
            global_rings,
            {
                "holes_processed": True,
                "segmentation_source": {"origin_px": [8000, 5000]},
            },
        )

    try:
        with patch("fdm.ui.canvas.draft_preview_cache", cache):
            window.show()
            desktop_application.processEvents()
            window._add_digital_slide_document_from_path(
                path,
                document=None,
                metadata={"digital_slide": {"viewport_origin": [8000, 5000]}},
                interaction_path_override=path,
            )
            canvas = window.current_canvas()
            canvas.fit_native_viewport()
            wait(canvas.pixel_work_enabled)
            canvas._hide_native_viewport_indicator()
            window.set_tool_mode("magic_segment")
            document = window.current_document()
            baseline = render()
            primary = np.zeros((240, 320), dtype=np.uint8)
            primary[50:191, 40:261] = 255
            primary[100:131, 110:141] = 0
            install(
                primary,
                [
                    [(40, 50), (260, 50), (260, 190), (40, 190)],
                    [(110, 100), (140, 100), (140, 130), (110, 130)],
                ],
            )
            sample = canvas.image_to_widget(Point(8070, 5090)).toPoint()
            preview = assert_draft_matches_reference([(70, 90), (120, 110)])
            assert preview.pixelColor(sample) != baseline.pixelColor(sample)
            assert not document.measurements

            canvas.cycle_magic_segment_operation_mode()
            subtract = np.zeros_like(primary)
            subtract[80:151, 190:241] = 255
            install(subtract, [[(190, 80), (240, 80), (240, 150), (190, 150)]])
            assert_draft_matches_reference([(70, 90), (210, 110)])
            assert canvas.confirm_current_magic_subtract_shape()["confirmed"]
            assert_draft_matches_reference([(70, 90), (210, 110)])
            assert not document.measurements

            window.keyPressEvent(
                QKeyEvent(
                    QEvent.Type.KeyPress, Qt.Key.Key_F, Qt.KeyboardModifier.NoModifier
                )
            )
            window._flush_pending_measurements(document)
            assert len(document.measurements) == 1
            measurement = document.measurements[0]
            assert measurement.exact_area_px == np.count_nonzero(primary & ~subtract)
            assert len(measurement.area_rings_px) == 3
            assert min(point.x for point in measurement.polygon_px) >= 8000
            assert min(point.y for point in measurement.polygon_px) >= 5000
            assert not canvas.has_magic_segment_session()
    finally:
        if canvas is not None:
            cache.discard(id(canvas))
        window._reset_workspace()
        window.close()


@pytest.mark.parametrize("dpr", [1.0, 1.25, 1.5, 2.0])
def test_real_process_round_trip_matches_qt_for_mixed_geometry_and_empty_tiles(
    desktop_application, dpr
):
    path = QPainterPath()
    path.setFillRule(Qt.FillRule.OddEvenFill)
    path.addRect(10, 10, 80, 80)
    path.addRect(30, 30, 40, 40)
    primitive = QPicture()
    painter = QPainter(primitive)
    painter.setPen(QColor("blue"))
    painter.drawLine(0, 0, 128, 128)
    painter.drawText(4, 120, "12")
    painter.end()
    snapshot = CanvasOverlayRenderSnapshot(
        1,
        CanvasOverlayTileKey(100, "mixed", 1, dpr, 0, 0, 0, 0, True),
        logical_tile_size=128,
        exact_composition=True,
        area_commands=(
            AreaOverlayDrawCommand(
                path,
                QTransform(),
                QColor(200, 50, 80, 70).rgba(),
                QColor("black").rgba(),
                3.2,
                QColor("red").rgba(),
                1.8,
            ),
            PictureOverlayDrawCommand(primitive),
        ),
    )
    expected, expected_picture = _TileRenderRunnable(
        snapshot, _CancellationFlag(), 0, queue.SimpleQueue(), _AreaCommandCentroidCache()
    )._render()
    actual, picture = renderer.render_in_isolated_worker(snapshot)
    assert actual.devicePixelRatio() == dpr
    assert np.array_equal(
        np.frombuffer(actual.constBits(), np.uint8), np.frombuffer(expected.constBits(), np.uint8)
    )
    assert bytes(picture.data()) == bytes(expected_picture.data())
    image, picture = renderer.render_in_isolated_worker(replace(snapshot, known_empty=True))
    assert image is None and picture.size() == 0


def test_dead_worker_is_restarted_once_and_document_cleanup_is_safe():
    class DeadPool:
        def submit(self, *args):
            raise BrokenProcessPool("worker stopped")

        def shutdown(self, **kwargs):
            pass

    dead = DeadPool()
    with (
        patch.object(renderer, "_pool", dead),
        patch.object(renderer, "_executor", return_value=dead) as factory,
    ):
        renderer.discard_document(1)
        assert renderer._pool is None
        with patch.object(renderer, "_encode", return_value={}), pytest.raises(BrokenProcessPool):
            renderer.render_in_isolated_worker(None)
        assert factory.call_count == 2
