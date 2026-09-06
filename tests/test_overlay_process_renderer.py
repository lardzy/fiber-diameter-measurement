from dataclasses import replace
from unittest.mock import patch
from concurrent.futures.process import BrokenProcessPool
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
