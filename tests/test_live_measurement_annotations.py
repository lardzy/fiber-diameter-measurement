"""Annotations protect measurement geometry in logical screen coordinates."""
import math

import pytest
from PySide6.QtCore import QRectF

from fdm.ui.live_measurement_annotations import (
    Annotation, layout_annotations, segment_distance, segment_intersects_rect,
)


@pytest.mark.parametrize("angle", range(0, 360, 15))
@pytest.mark.parametrize("length", (0., 5., 30., 150.))
def test_short_diameters_keep_labels_clear_at_all_angles(angle, length):
    end = (200 + math.cos(math.radians(angle)) * length,
           180 + math.sin(math.radians(angle)) * length)
    item = Annotation("current", (200., 180.), end, 40., 24., 100)
    viewport = QRectF(0, 0, 600, 500)
    placement = layout_annotations([item], viewport)["current"]
    assert viewport.contains(placement.rect)
    assert not segment_intersects_rect(item.start, item.end,
                                       placement.rect.adjusted(-15, -15, 15, 15))


@pytest.mark.parametrize("point", ((5, 5), (5, 295), (395, 5), (395, 295)))
def test_labels_stay_inside_visible_viewport_at_corners(point):
    item = Annotation("edge", point, (point[0] + 2, point[1] + 2), 40, 24)
    viewport = QRectF(0, 0, 400, 300)
    placement = layout_annotations([item], viewport)["edge"]
    assert viewport.contains(placement.rect)
    assert segment_distance(point, (placement.rect.left(), placement.rect.top()),
                            (placement.rect.right(), placement.rect.bottom())) > 12


def test_dense_labels_never_replace_or_cover_diameters_or_draft():
    items = [Annotation(str(index), (20 + x, 20 + y), (30 + x, 25 + y), 42, 24,
                        100 if index == 0 else 0)
             for index, (x, y) in enumerate((x, y) for x in range(0, 220, 20) for y in range(0, 160, 20))]
    draft = ((0, 0), (300, 240))
    placements = layout_annotations(items, QRectF(0, 0, 320, 260), protected_segments=[draft])
    assert len(placements) < len(items)
    rectangles = list(placements.values())
    for index, placement in enumerate(rectangles):
        for item in items:
            assert not segment_intersects_rect(item.start, item.end,
                                               placement.rect.adjusted(-15, -15, 15, 15))
        assert not segment_intersects_rect(*draft, placement.rect.adjusted(-15, -15, 15, 15))
        assert all(not placement.rect.intersects(other.rect) for other in rectangles[index + 1:])


def test_offscreen_line_does_not_create_orphan_label_and_layout_is_deterministic():
    items = [Annotation("offscreen", (20, 20), (30, 20), 30, 20),
             Annotation("onscreen", (350, 230), (380, 250), 30, 20)]
    viewport = QRectF(300, 150, 400, 300)
    first = layout_annotations(items, viewport)
    assert "offscreen" not in first
    assert first == layout_annotations(items, viewport)
    assert viewport.contains(first["onscreen"].rect)


def test_segment_math_handles_crossings_and_zero_length():
    rect = QRectF(10, 10, 20, 20)
    assert segment_intersects_rect((0, 20), (40, 20), rect)
    assert segment_intersects_rect((15, 15), (15, 15), rect)
    assert not segment_intersects_rect((9, 0), (9, 40), rect)
    assert segment_distance((3, 4), (0, 0), (0, 0)) == 5
    assert segment_distance((5, 2), (0, 0), (10, 0)) == 2


def assert_safe_placements(placements, items, viewport, protected=()):
    rectangles = [placement.rect for placement in placements.values()]
    for index, rect in enumerate(rectangles):
        assert viewport.contains(rect)
        for start, end in [(item.start, item.end) for item in items] + list(protected):
            assert not segment_intersects_rect(start, end, rect.adjusted(-15, -15, 15, 15))
        assert all(not rect.intersects(other) for other in rectangles[index + 1:])


def test_incremental_additions_keep_safe_old_rectangles_and_protect_new_line():
    viewport = QRectF(0, 0, 800, 600)
    old = Annotation("old", (180, 180), (200, 200), 36, 24)
    prior = layout_annotations([old], viewport)
    fresh = Annotation("new", (550, 400), (570, 420), 36, 24, 120)
    result = layout_annotations([old, fresh], viewport, previous_annotations=[old], previous_placements=prior)
    assert result["old"].rect == prior["old"].rect
    assert "new" in result
    assert_safe_placements(result, [old, fresh], viewport)


def test_new_crossing_line_repositions_old_label_without_covering_either_line():
    viewport = QRectF(0, 0, 800, 600)
    old = Annotation("old", (180, 180), (220, 180), 36, 24)
    prior = layout_annotations([old], viewport)
    rect = prior["old"].rect
    crossing = Annotation("new", (rect.left() - 20, rect.center().y()),
                          (rect.right() + 20, rect.center().y()), 36, 24, 120)
    moved = layout_annotations([old, crossing], viewport, previous_annotations=[old], previous_placements=prior)
    assert "old" in moved and "new" in moved
    assert moved["old"].rect != prior["old"].rect
    assert_safe_placements(moved, [old, crossing], viewport)
    # A valid prior layout may omit a secondary label in a dense scene.
    hidden = {"new": moved["new"]}
    hovered = Annotation("old", old.start, old.end, old.width, old.height, 180)
    promoted = layout_annotations([hovered, crossing], viewport, previous_annotations=[old, crossing],
                                  previous_placements=hidden)
    assert "old" in promoted
    assert_safe_placements(promoted, [hovered, crossing], viewport)
    # Undo/deletion frees space and retries secondary labels that were omitted.
    removed = layout_annotations([old], viewport, previous_annotations=[old, crossing], previous_placements=hidden)
    assert "old" in removed
    assert_safe_placements(removed, [old], viewport)


def test_incremental_draft_is_rechecked_and_finished_draft_releases_space():
    viewport = QRectF(0, 0, 800, 600)
    old = Annotation("old", (180, 180), (220, 180), 36, 24)
    prior = layout_annotations([old], viewport)
    rect = prior["old"].rect
    draft = ((rect.left() - 20, rect.center().y()), (rect.right() + 20, rect.center().y()))
    hidden = layout_annotations([old], viewport, protected_segments=[draft], previous_annotations=[old],
                                previous_placements=prior)
    assert_safe_placements(hidden, [old], viewport, [draft])
    # Force the omitted-label case to ensure finishing a draft retries it.
    hidden = {}
    released = layout_annotations([old], viewport, previous_annotations=[old], previous_placements=hidden,
                                  previous_protected_segments=[draft])
    assert "old" in released
    assert_safe_placements(released, [old], viewport)


def test_incremental_dense_sequence_and_priority_promotions_never_cover_geometry():
    viewport = QRectF(0, 0, 768, 576)
    previous, placements = [], {}
    for count in range(1, 141):
        items = [Annotation(str(i), (40 + (i % 20) * 24, 350 + (i // 20) % 6 * 25),
                            (40 + (i % 20) * 24, 370 + (i // 20) % 6 * 25), 36, 24,
                            120 if i == count - 1 else 0) for i in range(count)]
        placements = layout_annotations(items, viewport, previous_annotations=previous, previous_placements=placements)
        assert_safe_placements(placements, items, viewport)
        if count <= 9:
            assert len(placements) == count
        previous = items
    assert len(placements) < len(items)
