from __future__ import annotations

from fdm.geometry import Point
from fdm.models import Measurement
from fdm.ui.area_handle_cache import AreaHandleDisplayCache


def _area(measurement_id: str = "area-1") -> Measurement:
    points = [
        Point(float(index) / 10.0, 0.0)
        for index in range(1_000)
    ]
    points.extend(
        [
            Point(100.0, 100.0),
            Point(0.0, 100.0),
        ]
    )
    return Measurement(
        id=measurement_id,
        image_id="image-1",
        fiber_group_id=None,
        mode="manual",
        measurement_kind="area",
        polygon_px=list(points),
        area_rings_px=[list(points)],
    )


def test_handle_cache_reuses_thinned_coordinates_without_aliasing_raw_points() -> None:
    measurement = _area()
    cache = AreaHandleDisplayCache(max_entries=8, max_bytes=1024 * 1024)

    first = cache.coordinates(
        measurement,
        measurement.area_rings_px,
        output_scale=1.0,
        device_pixel_ratio=1.5,
    )
    second = cache.coordinates(
        measurement,
        measurement.area_rings_px,
        output_scale=1.0,
        device_pixel_ratio=1.5,
    )

    assert first is second
    assert len(first) < len(measurement.area_rings_px[0])
    assert all(isinstance(item, tuple) for item in first)
    assert cache.stats().hits == 1


def test_handle_cache_geometry_revision_and_zoom_are_independent_entries() -> None:
    measurement = _area()
    cache = AreaHandleDisplayCache(max_entries=8, max_bytes=1024 * 1024)
    first = cache.coordinates(
        measurement,
        measurement.area_rings_px,
        output_scale=1.0,
        device_pixel_ratio=1.0,
    )
    zoomed = cache.coordinates(
        measurement,
        measurement.area_rings_px,
        output_scale=2.0,
        device_pixel_ratio=1.0,
    )
    assert zoomed != first
    assert cache.stats().entries == 2

    replacement = [
        Point(0.0, 0.0),
        Point(20.0, 0.0),
        Point(20.0, 20.0),
        Point(0.0, 20.0),
    ]
    measurement.replace_area_geometry(
        polygon_px=replacement,
        area_rings_px=[replacement],
    )
    changed = cache.coordinates(
        measurement,
        measurement.area_rings_px,
        output_scale=1.0,
        device_pixel_ratio=1.0,
    )

    assert changed != first
    assert cache.stats().entries == 1


def test_handle_cache_keeps_at_most_three_scale_buckets_per_measurement() -> None:
    measurement = _area()
    cache = AreaHandleDisplayCache(max_entries=16, max_bytes=1024 * 1024)

    for scale in (0.5, 1.0, 1.5, 2.0, 2.5):
        cache.coordinates(
            measurement,
            measurement.area_rings_px,
            output_scale=scale,
            device_pixel_ratio=1.0,
        )

    assert cache.stats().entries == 3


def test_handle_cache_respects_entry_and_byte_budgets() -> None:
    cache = AreaHandleDisplayCache(max_entries=2, max_bytes=256)
    measurements = []
    for index in range(4):
        measurement = _area(f"area-{index}")
        measurements.append(measurement)
        cache.coordinates(
            measurement,
            measurement.area_rings_px,
            output_scale=0.05,
            device_pixel_ratio=1.0,
        )

    stats = cache.stats()
    assert stats.entries <= 2
    assert stats.bytes <= 256
    assert stats.evictions >= 1


def test_handle_cache_discards_only_the_closed_document_owners() -> None:
    first = _area("first")
    second = _area("second")
    cache = AreaHandleDisplayCache(max_entries=8, max_bytes=1024 * 1024)
    for measurement in (first, second):
        cache.coordinates(
            measurement,
            measurement.area_rings_px,
            output_scale=1.0,
            device_pixel_ratio=1.0,
        )

    cache.discard_document([first])

    assert cache.stats().entries == 1
    assert id(first) not in cache._owner_keys  # noqa: SLF001
    assert id(second) in cache._owner_keys  # noqa: SLF001
