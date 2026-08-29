from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtGui import QColor, QImage

from fdm.geometry import Point
from fdm.models import ImageDocument
from fdm.services.digital_slide_store import (
    DigitalSlideManifest,
    DigitalSlideStore,
    DigitalSlideTile,
)
from fdm.services.segmentation_source import digital_slide_segmentation_snapshot
from fdm.services.segmentation_source import image_segmentation_snapshot


def _solid(width: int, height: int, color: str) -> QImage:
    image = QImage(width, height, QImage.Format.Format_RGB32)
    image.fill(QColor(color))
    return image


def test_digital_slide_snapshot_uses_selected_focus_and_explicit_global_mapping(
    tmp_path: Path,
) -> None:
    path = tmp_path / "focus.fdmslide"
    store = DigitalSlideStore.create(
        path,
        DigitalSlideManifest(
            version=1,
            width=12,
            height=8,
            viewport_width=6,
            viewport_height=4,
            focus_levels=[100, 200],
        ),
    )
    store.write_tile(
        DigitalSlideTile(z_index=0, x=2, y=1, width=6, height=4),
        _solid(6, 4, "#ff0000"),
    )
    store.write_tile(
        DigitalSlideTile(z_index=1, x=2, y=1, width=6, height=4),
        _solid(6, 4, "#00ff00"),
    )
    document = ImageDocument(
        id="slide-1",
        path=str(path),
        image_size=(12, 8),
        document_kind="digital_slide",
    )

    snapshot = digital_slide_segmentation_snapshot(
        document,
        store,
        origin_px=Point(2.0, 1.0),
        width=6,
        height=4,
        focus_index=1,
    )

    assert snapshot.focus_index == 1
    assert snapshot.image.pixelColor(0, 0).green() > 240
    assert snapshot.to_local_point(Point(5.5, 3.0)) == Point(3.5, 2.0)
    assert snapshot.to_global_point(Point(3.5, 2.0)) == Point(5.5, 3.0)
    assert snapshot.contains_global_point(Point(2.0, 1.0))
    assert snapshot.valid_coverage.shape == (4, 6)
    assert bool(snapshot.valid_coverage.all())
    store.close()


def test_viewport_coverage_marks_gaps_without_treating_background_as_pixels(
    tmp_path: Path,
) -> None:
    path = tmp_path / "gaps.fdmslide"
    store = DigitalSlideStore.create(
        path,
        DigitalSlideManifest(
            version=1,
            width=10,
            height=5,
            viewport_width=10,
            viewport_height=5,
            focus_levels=[0],
        ),
    )
    store.write_tile(
        DigitalSlideTile(z_index=0, x=0, y=0, width=3, height=5),
        _solid(3, 5, "#ffffff"),
    )
    store.write_tile(
        DigitalSlideTile(z_index=0, x=7, y=0, width=3, height=5),
        _solid(3, 5, "#ffffff"),
    )

    coverage = store.viewport_coverage_mask(
        x=0,
        y=0,
        width=10,
        height=5,
        z_index=0,
    )

    assert np.all(coverage[:, :3])
    assert not np.any(coverage[:, 3:7])
    assert np.all(coverage[:, 7:])
    store.close()


def test_ordinary_image_snapshot_does_not_allocate_redundant_full_coverage_mask() -> None:
    image = _solid(200, 120, "#ffffff")
    document = ImageDocument(
        id="image-1",
        path="/tmp/image.png",
        image_size=(200, 120),
    )

    snapshot = image_segmentation_snapshot(document, image)

    assert snapshot.valid_coverage is None
    assert snapshot.contains_global_point(Point(199.9, 119.9))
    assert not snapshot.contains_global_point(Point(200.0, 120.0))
