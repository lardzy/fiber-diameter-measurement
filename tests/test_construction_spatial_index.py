from __future__ import annotations

import math

import pytest

from fdm.construction_geometry import (
    ArraySide,
    ConstructionEntity,
    ConstructionIssue,
    FreePointDefinition,
    LineExtent,
    ParallelLineSequence,
    ResolvedCircle,
    ResolvedConstruction,
    ResolvedLine,
    ResolvedLineArray,
    ResolvedPoint,
)
from fdm.construction_spatial_index import ConstructionSpatialIndex
from fdm.geometry import Point


def _entry(
    entity_id: str,
    geometry: ResolvedPoint | ResolvedLine | ResolvedCircle | ResolvedLineArray,
    *,
    visible: bool = True,
    locked: bool = False,
    snap_enabled: bool = True,
) -> tuple[ConstructionEntity, ResolvedConstruction]:
    entity = ConstructionEntity(
        id=entity_id,
        name=entity_id,
        definition=FreePointDefinition(Point(0, 0)),
        visible=visible,
        locked=locked,
        snap_enabled=snap_enabled,
    )
    return entity, ResolvedConstruction(entity_id, geometry=geometry)


def _ids(items: object) -> list[str]:
    return [item.entity_id for item in items]  # type: ignore[union-attr]


def test_finite_point_segment_and_circle_are_queried_from_local_grid_cells() -> None:
    entries = [
        _entry("point", ResolvedPoint(Point(10, 10))),
        _entry("segment", ResolvedLine(Point(100, 100), Point(120, 100))),
        _entry("circle", ResolvedCircle(Point(200, 200), 20)),
        _entry("far", ResolvedPoint(Point(1000, 1000))),
    ]
    index = ConstructionSpatialIndex.build(entries, cell_size_px=64, revision=7)

    assert _ids(index.query(Point(12, 10), 3)) == ["point"]
    assert _ids(index.query(Point(110, 103), 4)) == ["segment"]
    assert _ids(index.query(Point(200, 200), 1)) == ["circle"]
    assert _ids(index.query(Point(220, 200), 1)) == ["circle"]
    assert index.query(Point(210, 200), 2) == ()
    assert index.stats.item_count == 4
    assert index.stats.finite_item_count == 4
    assert index.stats.unbounded_item_count == 0
    assert index.is_current(7)
    assert not index.is_current(8)


def test_build_for_revision_reuses_current_index_and_rebuilds_stale_one() -> None:
    entries = [_entry("point", ResolvedPoint(Point(10, 10)))]
    first = ConstructionSpatialIndex.build_for_revision(entries, revision=3)

    def must_not_consume():
        raise AssertionError("matching revision must not consume entries")
        yield entries[0]

    reused = ConstructionSpatialIndex.build_for_revision(
        must_not_consume(),
        revision=3,
        previous=first,
    )
    rebuilt = ConstructionSpatialIndex.build_for_revision(
        entries,
        revision=4,
        previous=first,
    )

    assert reused is first
    assert rebuilt is not first
    assert rebuilt.is_current(4)


def test_finite_geometry_uses_exact_proximity_after_bbox_grid_lookup() -> None:
    diagonal = _entry(
        "diagonal",
        ResolvedLine(Point(0, 0), Point(100, 100), LineExtent.SEGMENT),
    )
    index = ConstructionSpatialIndex.build([diagonal], cell_size_px=32)

    assert index.query(Point(50, 52), 2) != ()
    # This point is inside the line's bbox but far from the actual segment.
    assert index.query(Point(0, 100), 5) == ()


def test_infinite_lines_and_rays_are_separate_and_filtered_analytically() -> None:
    infinite = _entry(
        "infinite",
        ResolvedLine(Point(0, 0), Point(1, 0), LineExtent.INFINITE),
    )
    ray = _entry(
        "ray",
        ResolvedLine(Point(0, 10), Point(1, 10), LineExtent.RAY),
    )
    index = ConstructionSpatialIndex.build([infinite, ray], cell_size_px=16)

    assert _ids(index.query(Point(1_000_000, 2), 3)) == ["infinite"]
    assert _ids(index.query(Point(20, 11), 2)) == ["ray"]
    assert index.query(Point(-20, 10), 2) == ()
    assert _ids(index.query(Point(0, 10), 0)) == ["ray"]
    assert index.stats.unbounded_item_count == 2
    assert index.stats.grid_cell_count == 0


def test_mixed_line_array_is_split_into_stable_child_features() -> None:
    array = ResolvedLineArray(
        (
            ResolvedLine(Point(0, 0), Point(10, 0), LineExtent.SEGMENT),
            ResolvedLine(Point(0, 20), Point(10, 20), LineExtent.INFINITE),
            ResolvedLine(Point(0, 40), Point(10, 40), LineExtent.RAY),
        )
    )
    index = ConstructionSpatialIndex.build([_entry("array", array)], cell_size_px=16)

    assert [item.feature_key for item in index.items] == ["line:0", "line:1", "line:2"]
    assert index.stats.finite_item_count == 1
    assert index.stats.unbounded_item_count == 2
    assert [item.feature_key for item in index.query(Point(5, 0), 1)] == ["line:0"]
    assert [item.feature_key for item in index.query(Point(500, 20), 1)] == ["line:1"]
    assert [item.feature_key for item in index.query(Point(500, 40), 1)] == ["line:2"]


def test_query_pairs_deduplicates_array_children_and_preserves_input_order() -> None:
    first_array = ResolvedLineArray(
        (
            ResolvedLine(Point(0, 0), Point(10, 0)),
            ResolvedLine(Point(0, 2), Point(10, 2)),
        )
    )
    entries = [
        _entry("first", first_array),
        _entry("second", ResolvedPoint(Point(5, 1))),
    ]
    index = ConstructionSpatialIndex.build(entries, cell_size_px=4)

    pairs = index.query_pairs(Point(5, 1), 2)

    assert [entity.id for entity, _resolved in pairs] == ["first", "second"]
    assert len(index.query(Point(5, 1), 2)) == 3


def test_entity_metadata_is_retained_and_snappable_query_applies_visibility_flags() -> None:
    entries = [
        _entry("hidden", ResolvedPoint(Point(0, 0)), visible=False),
        _entry("disabled", ResolvedPoint(Point(1, 0)), snap_enabled=False),
        _entry("locked", ResolvedPoint(Point(2, 0)), locked=True),
    ]
    index = ConstructionSpatialIndex.build(entries, cell_size_px=8)

    all_items = index.query(Point(1, 0), 4)
    snap_items = index.query_snappable(Point(1, 0), 4)

    assert _ids(all_items) == ["hidden", "disabled", "locked"]
    assert _ids(snap_items) == ["locked"]
    assert snap_items[0].entity.locked


def test_invalid_resolutions_are_not_indexed() -> None:
    entity, _resolved = _entry("invalid", ResolvedPoint(Point(0, 0)))
    invalid = ResolvedConstruction(
        "invalid",
        error=ConstructionIssue("invalid", "invalid"),
    )

    index = ConstructionSpatialIndex.build([(entity, invalid)])

    assert index.items == ()
    assert index.stats.item_count == 0
    assert index.query(Point(0, 0), 100) == ()


def test_oversized_finite_bbox_uses_guard_collection_without_grid_explosion() -> None:
    huge_circle = _entry("huge", ResolvedCircle(Point(0, 0), 1_000_000))
    index = ConstructionSpatialIndex.build(
        [huge_circle],
        cell_size_px=10,
        max_cells_per_item=4,
    )

    assert index.stats.oversized_item_count == 1
    assert index.stats.grid_cell_count == 0
    assert _ids(index.query(Point(0, 0), 1)) == ["huge"]
    assert _ids(index.query(Point(1_000_000, 0), 1)) == ["huge"]
    assert index.query(Point(500_000, 0), 1) == ()


def test_crossing_lines_remain_two_primitives_and_do_not_precompute_intersection() -> None:
    entries = [
        _entry("horizontal", ResolvedLine(Point(-10, 0), Point(10, 0))),
        _entry("vertical", ResolvedLine(Point(0, -10), Point(0, 10))),
    ]
    index = ConstructionSpatialIndex.build(entries, cell_size_px=8)

    hits = index.query(Point(0, 0), 1)

    assert _ids(hits) == ["horizontal", "vertical"]
    assert index.stats.item_count == 2
    assert all(item.feature_key == "geometry" for item in hits)


def test_large_regular_dataset_returns_only_the_local_neighborhood() -> None:
    entries = [
        _entry(
            f"point-{column}-{row}",
            ResolvedPoint(Point(column * 20.0, row * 20.0)),
        )
        for row in range(100)
        for column in range(200)
    ]
    index = ConstructionSpatialIndex.build(entries, cell_size_px=64, revision=12)

    hits = index.query(Point(100 * 20.0, 50 * 20.0), 5)

    assert _ids(hits) == ["point-100-50"]
    assert index.stats.item_count == 20_000
    assert index.stats.grid_cell_count < index.stats.item_count
    assert index.stats.grid_reference_count == index.stats.item_count
    assert index.is_current(12)


def test_large_parametric_array_is_indexed_and_queried_without_expansion() -> None:
    lines = ParallelLineSequence(
        ResolvedLine(Point(0, 0), Point(10, 0)),
        2.0,
        10_000,
        ArraySide.BOTH,
        LineExtent.INFINITE,
    )
    entity, _unused = _entry("lazy-array", ResolvedPoint(Point(0, 0)))
    resolved = ResolvedConstruction(
        entity.id,
        geometry=ResolvedLineArray(lines),
    )

    index = ConstructionSpatialIndex.build([(entity, resolved)])

    assert index.stats.item_count == 20_000
    assert index.stats.grid_cell_count == 0
    assert index.items == ()
    hits = index.query(Point(5, 20_000), 0.01)
    assert len(hits) == 1
    assert hits[0].feature_key == "line:+10000"
    assert hits[0].geometry.start == Point(0, 20_000)


def test_extremely_large_finite_query_falls_back_to_item_scan_without_cell_walk() -> None:
    index = ConstructionSpatialIndex.build(
        [
            _entry("origin", ResolvedPoint(Point(0, 0))),
            _entry("far", ResolvedPoint(Point(1_000_000, 1_000_000))),
        ],
        cell_size_px=8,
    )

    hits = index.query(Point(0, 0), 1e308)

    assert _ids(hits) == ["origin", "far"]


@pytest.mark.parametrize(
    ("cursor", "radius"),
    [
        (Point(math.nan, 0), 1),
        (Point(0, math.inf), 1),
        (Point(0, 0), -1),
        (Point(0, 0), math.inf),
    ],
)
def test_query_rejects_nonfinite_cursor_or_invalid_radius(
    cursor: Point,
    radius: float,
) -> None:
    index = ConstructionSpatialIndex.build([])

    with pytest.raises(ValueError):
        index.query(cursor, radius)


def test_build_rejects_invalid_grid_configuration() -> None:
    with pytest.raises(ValueError):
        ConstructionSpatialIndex.build([], cell_size_px=0)
    with pytest.raises(ValueError):
        ConstructionSpatialIndex.build([], cell_size_px=math.inf)
    with pytest.raises(ValueError):
        ConstructionSpatialIndex.build([], max_cells_per_item=0)
