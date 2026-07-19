from __future__ import annotations

import math
import time
from collections import OrderedDict
from unittest.mock import patch

import fdm.area_display as area_display
from fdm.area_display import (
    AREA_GEOMETRY_RAW,
    AREA_GEOMETRY_SCREEN,
    AreaDerivedGeometryService,
    AreaProxyBuildBudget,
    area_derived_geometry_service,
)
from fdm.geometry import Point
from fdm.models import Measurement
from fdm.ui.rendering import _area_handle_points_for_display


def _dense_area(measurement_id: str = "area") -> Measurement:
    outer = [
        Point(100.0 + 80.0 * math.cos(2.0 * math.pi * index / 512), 100.0 + 60.0 * math.sin(2.0 * math.pi * index / 512))
        for index in range(512)
    ]
    hole = [
        Point(100.0 + 20.0 * math.cos(-2.0 * math.pi * index / 256), 100.0 + 15.0 * math.sin(-2.0 * math.pi * index / 256))
        for index in range(256)
    ]
    measurement = Measurement(
        id=measurement_id,
        image_id="image",
        fiber_group_id=None,
        mode="polygon_area",
        measurement_kind="area",
        polygon_px=outer,
        area_rings_px=[outer, hole],
    )
    measurement.recalculate(None)
    return measurement


def _small_area(measurement_id: str, offset: float = 0.0) -> Measurement:
    outer = [
        Point(offset, 0.0),
        Point(offset + 10.0, 0.0),
        Point(offset + 10.0, 10.0),
        Point(offset, 10.0),
    ]
    return Measurement(
        id=measurement_id,
        image_id="image",
        fiber_group_id=None,
        mode="polygon_area",
        measurement_kind="area",
        polygon_px=outer,
        area_rings_px=[outer],
    )


def test_screen_proxy_never_changes_raw_area_or_serialized_geometry() -> None:
    area_derived_geometry_service.clear()
    measurement = _dense_area()
    before = measurement.to_dict()
    raw = area_derived_geometry_service.raw_geometry(measurement)
    screen = area_derived_geometry_service.screen_geometry(measurement, zoom=1.0, selected=False)
    selected = area_derived_geometry_service.screen_geometry(measurement, zoom=1.0, selected=True)

    assert screen.source == AREA_GEOMETRY_SCREEN
    assert sum(map(len, screen.fill_rings)) < sum(map(len, raw.fill_rings))
    assert selected.source == AREA_GEOMETRY_RAW
    assert measurement.to_dict() == before
    assert measurement.area_px == before["area_px"]


def test_replace_area_geometry_copies_input_and_invalidates_only_its_cache() -> None:
    area_derived_geometry_service.clear()
    first = _dense_area("first")
    second = _dense_area("second")
    first_path = area_derived_geometry_service.raw_path(first)
    second_path = area_derived_geometry_service.raw_path(second)
    source_polygon = list(first.polygon_px)
    source_rings = [list(ring) for ring in first.area_rings_px]

    first.replace_area_geometry(
        polygon_px=source_polygon,
        area_rings_px=source_rings,
        exact_area_px=1234.5,
        calibration=None,
    )
    source_polygon[0].x = -999.0
    source_rings[0][0].x = -999.0

    assert first.geometry_revision == 1
    assert first.polygon_px[0].x != -999.0
    assert first.area_rings_px[0][0].x != -999.0
    assert first.area_px == 1234.5
    assert area_derived_geometry_service.raw_path(first) is not first_path
    assert area_derived_geometry_service.raw_path(second) is second_path


def test_geometry_revision_is_runtime_only() -> None:
    measurement = _dense_area()
    payload = measurement.to_dict()
    measurement.replace_area_geometry(
        polygon_px=measurement.polygon_px,
        area_rings_px=measurement.area_rings_px,
        exact_area_px=measurement.exact_area_px,
    )
    assert "geometry_revision" not in measurement.to_dict()
    assert Measurement.from_dict(payload).geometry_revision == 0


def test_raw_geometry_does_not_eagerly_compute_centroid_or_hole_area() -> None:
    service = AreaDerivedGeometryService()
    measurement = _dense_area()

    raw = service.raw_geometry(measurement)

    assert raw.bounds is not None
    assert raw.path.elementCount() > 0
    assert not service._moments
    assert not service._hole_areas


def test_raw_bounds_does_not_construct_or_cache_a_painter_path() -> None:
    service = AreaDerivedGeometryService()
    measurement = _dense_area()

    with patch.object(
        area_display,
        "_rings_path",
        side_effect=AssertionError("bounds query must not construct a path"),
    ):
        bounds = service.raw_bounds(measurement)

    assert bounds is not None
    assert not service._raw_paths
    assert service.path_cache_entry_count == 0
    assert service.path_cache_bytes == 0


def test_polygon_only_area_uses_cached_scalar_derivatives() -> None:
    service = AreaDerivedGeometryService()
    measurement = _dense_area()
    measurement.replace_area_geometry(
        polygon_px=measurement.polygon_px,
        area_rings_px=[],
        exact_area_px=measurement.exact_area_px,
        calibration=None,
    )

    centroid = service.centroid(measurement)
    vector_area = service.vector_area(measurement)

    assert centroid != Point(0.0, 0.0)
    assert vector_area > 0.0
    assert len(service._moments) == 1
    assert not service._hole_areas

    first = service.scalar_geometry(measurement)
    second = service.scalar_geometry(measurement)

    assert first.centroid == second.centroid
    assert first.vector_area_px == second.vector_area_px
    assert len(service._moments) == 1
    assert len(service._hole_areas) == 1


def test_raw_hit_testing_uses_exact_odd_even_path_and_original_vertices() -> None:
    service = AreaDerivedGeometryService()
    outer = [Point(0, 0), Point(100, 0), Point(100, 100), Point(0, 100)]
    hole = [Point(20, 20), Point(80, 20), Point(80, 80), Point(20, 80)]
    island = [Point(40, 40), Point(60, 40), Point(60, 60), Point(40, 60)]
    second_outer = [Point(120, 0), Point(160, 0), Point(160, 40), Point(120, 40)]
    measurement = Measurement(
        id="nested",
        image_id="image",
        fiber_group_id=None,
        mode="polygon_area",
        measurement_kind="area",
        polygon_px=outer,
        area_rings_px=[outer, hole, island, second_outer],
    )

    assert service.contains_raw(measurement, Point(10, 10))
    assert not service.contains_raw(measurement, Point(30, 30))
    assert service.contains_raw(measurement, Point(50, 50))
    assert service.contains_raw(measurement, Point(130, 10))
    assert service.near_edge(measurement, Point(20.25, 50), 0.5)
    assert service.nearest_vertex(measurement, Point(60.1, 60.1), 0.5)[:2] == (2, 2)


def test_new_geometry_revision_removes_only_that_objects_old_cache_entries() -> None:
    service = AreaDerivedGeometryService()
    first = _dense_area("first")
    second = _dense_area("second")
    service.screen_geometry(first, zoom=1.0, selected=False)
    service.nearest_vertex(first, first.area_rings_px[0][0], 1.0)
    second_path = service.raw_path(second)

    first.replace_area_geometry(
        polygon_px=first.polygon_px,
        area_rings_px=first.area_rings_px,
        exact_area_px=first.exact_area_px,
    )
    service.raw_geometry(first)

    identity = (id(first), first.id)
    keyed_caches = (service._bounds, service._moments, service._hole_areas, service._raw_paths, service._hit_indexes)
    for cache in keyed_caches:
        assert all(key[:2] != identity or key[2] == first.geometry_revision for key in cache)
    assert all(key[:2] != identity or key[2] == first.geometry_revision for key in service._proxies)
    assert service.raw_path(second) is second_path


def test_zoom_bucket_uses_upper_edge_for_half_pixel_proxy_budget() -> None:
    service = AreaDerivedGeometryService()
    for zoom in (0.07, 0.99, 1.18, 3.7, 39.0):
        bucket = service._zoom_bucket(zoom)
        epsilon = service._proxy_epsilon_for_bucket(bucket)
        assert epsilon * zoom <= area_display.SCREEN_PROXY_MAX_ERROR_PX + 1e-12


def test_handle_display_is_thinned_but_hit_index_returns_original_vertex() -> None:
    service = AreaDerivedGeometryService()
    measurement = _dense_area()
    handles = _area_handle_points_for_display(measurement.area_rings_px, output_scale=1.0)
    target = measurement.area_rings_px[0][1]

    assert len(handles) < sum(len(ring) for ring in measurement.area_rings_px)
    nearest = service.nearest_vertex(measurement, target, 0.01)
    assert nearest is not None
    assert nearest[:2] == (0, 1)


def test_raw_and_proxy_paths_share_one_bounded_budget_including_proxy_points() -> None:
    service = AreaDerivedGeometryService()
    measurements = [_dense_area(f"area-{index}") for index in range(4)]
    with patch.object(area_display, "_PATH_MAX_ESTIMATED_BYTES", 60_000):
        for measurement in measurements:
            service.screen_geometry(measurement, zoom=1.0, selected=False)

    assert service.path_cache_bytes <= 60_000
    assert service.path_cache_entry_count == (
        len(service._raw_paths) + len(service._proxies)
    )
    assert all(
        entry.estimated_bytes > area_display._path_bytes(entry.path)
        for entry in service._proxies.values()
    )


def test_path_cache_has_no_low_entry_limit_when_byte_budget_has_capacity() -> None:
    service = AreaDerivedGeometryService()
    measurements = [_small_area(f"area-{index}", float(index * 20)) for index in range(300)]

    with service.path_render_pass():
        for measurement in measurements:
            service.raw_path(measurement)

    assert service.path_cache_entry_count == 300
    assert service.path_cache_bytes <= area_display._PATH_MAX_ESTIMATED_BYTES


def test_render_pass_pinning_prevents_sequential_lru_scan_thrash() -> None:
    service = AreaDerivedGeometryService()
    measurements = [_small_area(f"area-{index}", float(index * 20)) for index in range(3)]

    with patch.object(area_display, "_PATH_MAX_ESTIMATED_BYTES", 512):
        with service.path_render_pass():
            for measurement in measurements:
                service.raw_path(measurement)
        stable_keys = tuple(service._raw_paths)
        assert len(stable_keys) == 2

        for _pass_index in range(2):
            with patch.object(
                area_display,
                "_rings_path",
                wraps=area_display._rings_path,
            ) as build_path:
                with service.path_render_pass():
                    for measurement in measurements:
                        service.raw_path(measurement)
            assert build_path.call_count == 1
            assert tuple(service._raw_paths) == stable_keys


def test_render_pass_keeps_proxy_working_set_stable_when_budget_is_full() -> None:
    service = AreaDerivedGeometryService()
    measurements = [_dense_area(f"area-{index}") for index in range(3)]

    with patch.object(area_display, "_PATH_MAX_ESTIMATED_BYTES", 50_000):
        with service.path_render_pass():
            for measurement in measurements:
                service.screen_geometry(measurement, zoom=1.0, selected=False)
        stable_keys = tuple(service._proxies)
        assert len(stable_keys) == 2

        with patch.object(
            service,
            "_build_proxy",
            wraps=service._build_proxy,
        ) as build_proxy:
            with service.path_render_pass():
                geometries = [
                    service.screen_geometry(
                        measurement,
                        zoom=1.0,
                        selected=False,
                    )
                    for measurement in measurements
                ]

        assert [geometry.source for geometry in geometries] == [
            AREA_GEOMETRY_SCREEN,
            AREA_GEOMETRY_SCREEN,
            AREA_GEOMETRY_RAW,
        ]
        assert build_proxy.call_count == 1
        assert tuple(service._proxies) == stable_keys


def test_path_cache_stats_and_generation_only_change_with_path_entries() -> None:
    service = AreaDerivedGeometryService()
    measurement = _small_area("area")

    assert service.path_cache_generation == 0
    service.raw_bounds(measurement)
    assert service.path_cache_generation == 0

    first = service.raw_path(measurement)
    assert first.elementCount() > 0
    assert service.path_cache_generation == 1
    assert service.path_cache_entry_count == 1
    assert service.path_cache_bytes > 0

    assert service.raw_path(measurement) is first
    assert service.path_cache_generation == 1

    service.discard_measurement(measurement)
    assert service.path_cache_generation == 2
    assert service.path_cache_entry_count == 0
    assert service.path_cache_bytes == 0

    service.clear()
    assert service.path_cache_generation == 2


def test_hit_index_is_bounded_and_entries_verify_their_owner() -> None:
    service = AreaDerivedGeometryService()
    with patch.object(area_display, "_HIT_INDEX_MAX_ENTRIES", 2):
        measurements = [_dense_area(f"hit-{index}") for index in range(4)]
        for measurement in measurements:
            service.nearest_vertex(measurement, measurement.area_rings_px[0][0], 1.0)

    assert len(service._hit_indexes) <= 2
    for key, entry in service._hit_indexes.items():
        assert key[:2] == (id(entry.owner), entry.owner.id)


def test_discard_document_releases_all_owner_bound_geometry_entries() -> None:
    service = AreaDerivedGeometryService()
    measurements = [_dense_area("first"), _dense_area("second")]
    for measurement in measurements:
        service.screen_geometry(measurement, zoom=1.0, selected=False)
        service.scalar_geometry(measurement)
        service.nearest_vertex(measurement, measurement.area_rings_px[0][0], 1.0)

    service.discard_document(measurements)

    assert not service._bounds
    assert not service._moments
    assert not service._hole_areas
    assert not service._raw_paths
    assert not service._proxies
    assert not service._hit_indexes
    assert service._path_bytes == 0
    assert service._hit_index_bytes == 0


def test_discard_document_uses_owner_index_without_scanning_unrelated_caches() -> None:
    class IterationForbiddenOrderedDict(OrderedDict):
        def __iter__(self):
            raise AssertionError("discard must not scan a complete cache")

    service = AreaDerivedGeometryService()
    first = _dense_area("first")
    second = _dense_area("second")
    for measurement in (first, second):
        service.screen_geometry(measurement, zoom=1.0, selected=False)
        service.scalar_geometry(measurement)
        service.nearest_vertex(measurement, measurement.area_rings_px[0][0], 1.0)

    first_identity = (id(first), first.id)
    second_identity = (id(second), second.id)
    assert first_identity in service._owner_cache_keys
    assert second_identity in service._owner_cache_keys

    for attribute in (
        "_bounds",
        "_moments",
        "_hole_areas",
        "_raw_paths",
        "_proxies",
        "_proxy_failures",
        "_hit_indexes",
    ):
        setattr(
            service,
            attribute,
            IterationForbiddenOrderedDict(getattr(service, attribute)),
        )

    service.discard_document([first])

    assert first_identity not in service._owner_cache_keys
    assert second_identity in service._owner_cache_keys
    for cache in (
        service._bounds,
        service._moments,
        service._hole_areas,
        service._raw_paths,
        service._proxies,
        service._proxy_failures,
        service._hit_indexes,
    ):
        assert all(key[:2] == second_identity for key in cache.keys())


def test_compact_hit_index_stays_cached_and_fits_six_hundred_thousand_points() -> None:
    assert area_display._compact_hit_index_estimated_bytes(600_000, 8) < 32 * 1024 * 1024
    service = AreaDerivedGeometryService()
    measurement = _dense_area()
    with (
        patch.object(area_display, "_HIT_INDEX_COMPACT_POINT_THRESHOLD", 10),
        patch.object(
            area_display,
            "_build_compact_raw_hit_index",
            wraps=area_display._build_compact_raw_hit_index,
        ) as build_compact,
    ):
        first = service._raw_hit_index(measurement)
        second = service._raw_hit_index(measurement)

    assert first is second
    assert build_compact.call_count == 1
    assert first.compact_points is not None
    assert first.estimated_bytes <= area_display._HIT_INDEX_MAX_ESTIMATED_BYTES
    target = measurement.area_rings_px[0][17]
    nearest = service.nearest_vertex(measurement, target, 0.01)
    assert nearest is not None and nearest[:2] == (0, 17)
    assert service.near_edge(measurement, target, 0.01)


def test_extremely_dense_screen_geometry_skips_superlinear_proxy_validation() -> None:
    for count in (200_000, 600_000):
        ring = [Point(float(index), float(index % 101)) for index in range(count)]
        measurement = Measurement(
            id=f"dense-{count}",
            image_id="image",
            fiber_group_id=None,
            mode="polygon_area",
            measurement_kind="area",
            polygon_px=ring,
            area_rings_px=[ring],
        )
        service = AreaDerivedGeometryService()

        started = time.perf_counter()
        service.raw_geometry(measurement)
        raw_elapsed = time.perf_counter() - started
        service.clear()

        with patch.object(service, "_build_proxy", wraps=service._build_proxy) as build_proxy:
            started = time.perf_counter()
            geometry = service.screen_geometry(measurement, zoom=1.0, selected=False)
            screen_elapsed = time.perf_counter() - started

        assert geometry.source == AREA_GEOMETRY_RAW
        assert build_proxy.call_count == 0
        # Relative timing is intentionally generous for loaded CI hosts.  The
        # regression being guarded is the seconds-long Qt simplification path;
        # the exact fallback should remain in the same order as raw path build.
        assert screen_elapsed <= (raw_elapsed * 4.0) + 0.1


def test_screen_frame_budget_progressively_builds_proxies_with_raw_fallback() -> None:
    service = AreaDerivedGeometryService()
    measurements = [_dense_area(f"budget-{index}") for index in range(4)]
    before = [measurement.to_dict() for measurement in measurements]
    first_budget = AreaProxyBuildBudget(max_builds=1, max_build_ms=1_000.0)

    first_sources = [
        service.screen_geometry(
            measurement,
            zoom=1.0,
            selected=False,
            build_budget=first_budget,
        ).source
        for measurement in measurements
    ]

    assert first_sources == [AREA_GEOMETRY_SCREEN, AREA_GEOMETRY_RAW, AREA_GEOMETRY_RAW, AREA_GEOMETRY_RAW]
    assert first_budget.deferred
    deferred_geometry = service.screen_geometry(
        measurements[1],
        zoom=1.0,
        selected=False,
        build_budget=first_budget,
    )
    assert deferred_geometry.proxy_deferred
    assert [measurement.to_dict() for measurement in measurements] == before

    second_budget = AreaProxyBuildBudget(max_builds=1, max_build_ms=1_000.0)
    second_sources = [
        service.screen_geometry(
            measurement,
            zoom=1.0,
            selected=False,
            build_budget=second_budget,
        ).source
        for measurement in measurements
    ]
    assert second_sources[:2] == [AREA_GEOMETRY_SCREEN, AREA_GEOMETRY_SCREEN]
    assert second_budget.deferred


def test_fixed_raw_screen_geometry_is_not_mislabeled_as_budget_deferred() -> None:
    service = AreaDerivedGeometryService()
    dense = _dense_area()
    small = Measurement(
        id="small",
        image_id="image",
        fiber_group_id=None,
        mode="polygon_area",
        measurement_kind="area",
        polygon_px=[Point(0, 0), Point(10, 0), Point(10, 10), Point(0, 10)],
        area_rings_px=[[Point(0, 0), Point(10, 0), Point(10, 10), Point(0, 10)]],
    )
    exhausted = AreaProxyBuildBudget(max_builds=0)

    selected = service.screen_geometry(
        dense,
        zoom=1.0,
        selected=True,
        build_budget=exhausted,
    )
    threshold_raw = service.screen_geometry(
        small,
        zoom=1.0,
        selected=False,
        build_budget=exhausted,
    )

    assert selected.source == AREA_GEOMETRY_RAW
    assert not selected.proxy_deferred
    assert threshold_raw.source == AREA_GEOMETRY_RAW
    assert not threshold_raw.proxy_deferred
    assert not exhausted.deferred


def test_failed_proxy_validation_is_not_repeated_every_paint() -> None:
    service = AreaDerivedGeometryService()
    measurement = _dense_area()

    with patch.object(service, "_build_proxy", return_value=None) as build_proxy:
        first = service.screen_geometry(measurement, zoom=1.0, selected=False)
        second = service.screen_geometry(measurement, zoom=1.0, selected=False)

    assert first.source == AREA_GEOMETRY_RAW
    assert second.source == AREA_GEOMETRY_RAW
    assert build_proxy.call_count == 1

    measurement.replace_area_geometry(
        polygon_px=measurement.polygon_px,
        area_rings_px=measurement.area_rings_px,
        exact_area_px=measurement.exact_area_px,
    )
    with patch.object(service, "_build_proxy", return_value=None) as rebuild_proxy:
        service.screen_geometry(measurement, zoom=1.0, selected=False)
    assert rebuild_proxy.call_count == 1


def test_failed_proxy_memoization_has_a_global_lru_limit() -> None:
    service = AreaDerivedGeometryService()
    measurements = [_dense_area(f"failure-{index}") for index in range(7)]

    with (
        patch.object(area_display, "_PROXY_FAILURE_MAX_ENTRIES", 4),
        patch.object(service, "_build_proxy", return_value=None),
    ):
        for measurement in measurements[:5]:
            service.screen_geometry(measurement, zoom=1.0, selected=False)

        # A failed lookup is still a cache hit and therefore refreshes its LRU
        # position before the next distinct failure is admitted.
        service.screen_geometry(measurements[1], zoom=1.0, selected=False)
        for measurement in measurements[5:]:
            service.screen_geometry(measurement, zoom=1.0, selected=False)

    assert len(service._proxy_failures) == 4
    retained_ids = [entry.owner.id for entry in service._proxy_failures.values()]
    assert retained_ids == ["failure-4", "failure-1", "failure-5", "failure-6"]

    retained_keys = set(service._proxy_failures)
    for measurement in measurements:
        owner_keys = service._owner_cache_keys[(id(measurement), measurement.id)]
        expected = {
            key
            for key in retained_keys
            if key[:2] == (id(measurement), measurement.id)
        }
        assert owner_keys.proxy_failures == expected


def test_discard_document_releases_globally_bounded_proxy_failures() -> None:
    service = AreaDerivedGeometryService()
    measurements = [_dense_area(f"discard-failure-{index}") for index in range(9)]

    with (
        patch.object(area_display, "_PROXY_FAILURE_MAX_ENTRIES", 5),
        patch.object(service, "_build_proxy", return_value=None),
    ):
        for measurement in measurements:
            service.screen_geometry(measurement, zoom=1.0, selected=False)

    assert len(service._proxy_failures) == 5
    service.discard_document(measurements)

    assert not service._proxy_failures
    assert not service._owner_cache_keys


def test_hot_screen_proxy_does_not_rebuild_or_retain_the_raw_path() -> None:
    service = AreaDerivedGeometryService()
    measurement = _dense_area()

    first = service.screen_geometry(measurement, zoom=1.0, selected=False)

    assert first.source == AREA_GEOMETRY_SCREEN
    assert not service._raw_paths
    with patch.object(
        service,
        "raw_geometry",
        side_effect=AssertionError("hot proxy must not touch RAW path"),
    ):
        second = service.screen_geometry(measurement, zoom=1.0, selected=False)
    assert second.path is first.path
