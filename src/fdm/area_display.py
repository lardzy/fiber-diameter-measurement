from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import math
import os
import time

import cv2
import numpy as np
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QPainterPath

from fdm.geometry import (
    Point,
    area_rings_area_and_centroid,
    area_rings_bounds,
    area_rings_hole_area,
    clean_ring,
    odd_even_path_moments,
    point_to_segment_distance,
    polygon_bounds,
    ring_signed_area,
)
from fdm.models import Measurement


AREA_GEOMETRY_RAW = "raw"
AREA_GEOMETRY_SCREEN = "screen"
SCREEN_PROXY_VERTEX_THRESHOLD = 256
SCREEN_PROXY_MAX_ERROR_PX = 0.5
SCREEN_PROXY_MAX_SOURCE_VERTICES = 50_000
SCREEN_PROXY_FRAME_MAX_BUILDS = 2
SCREEN_PROXY_FRAME_MAX_BUILD_MS = 20.0
SCREEN_PROXY_FRAME_MAX_OBJECT_VERTICES = 12_000
_SQRT_TWO = math.sqrt(2.0)
_SCALAR_MAX_ENTRIES = 2048
_PATH_MAX_ENTRIES = 256
_PATH_MAX_ESTIMATED_BYTES = 64 * 1024 * 1024
_PROXY_BUCKETS_PER_MEASUREMENT = 3
_HIT_INDEX_MAX_ENTRIES = 32
_HIT_INDEX_MAX_ESTIMATED_BYTES = 32 * 1024 * 1024
_HIT_INDEX_CELL_SIZE = 64.0
_HIT_INDEX_COMPACT_POINT_THRESHOLD = 50_000
_HIT_INDEX_VECTOR_CHUNK_SIZE = 65_536
_PROXY_POINT_ESTIMATED_BYTES = 72


@dataclass(frozen=True, slots=True)
class AreaScalarGeometry:
    bounds: tuple[float, float, float, float] | None
    centroid: Point
    hole_area_px: float
    vector_area_px: float


@dataclass(frozen=True, slots=True)
class AreaGeometryView:
    outline_points: list[Point]
    fill_rings: list[list[Point]]
    bounds: tuple[float, float, float, float] | None
    path: QPainterPath
    source: str
    zoom_bucket: int | None = None
    proxy_deferred: bool = False


@dataclass(slots=True)
class AreaProxyBuildBudget:
    """Bound optional proxy work performed by one screen paint pass.

    Exhausting the budget never hides or approximates an object: callers get
    its cached RAW path for that frame and may schedule another paint to warm
    the next proxy.  Very large individual objects are kept on RAW in the UI
    thread because one proxy validation alone could exceed the frame budget.
    """

    max_builds: int = SCREEN_PROXY_FRAME_MAX_BUILDS
    max_build_ms: float = SCREEN_PROXY_FRAME_MAX_BUILD_MS
    max_object_vertices: int = SCREEN_PROXY_FRAME_MAX_OBJECT_VERTICES
    builds: int = 0
    build_ms: float = 0.0
    deferred: bool = False
    last_request_deferred: bool = False

    def permits(self, vertex_count: int) -> bool:
        self.last_request_deferred = False
        if vertex_count > self.max_object_vertices:
            return False
        if self.builds >= self.max_builds or self.build_ms >= self.max_build_ms:
            self.deferred = True
            self.last_request_deferred = True
            return False
        return True

    def record(self, elapsed_seconds: float) -> None:
        self.builds += 1
        self.build_ms += max(0.0, float(elapsed_seconds)) * 1000.0


@dataclass(slots=True)
class _PathEntry:
    owner: Measurement
    path: QPainterPath
    estimated_bytes: int
    last_used: int


@dataclass(slots=True)
class _ProxyEntry:
    owner: Measurement
    outline_points: list[Point]
    fill_rings: list[list[Point]]
    bounds: tuple[float, float, float, float]
    path: QPainterPath
    estimated_bytes: int
    last_used: int


@dataclass(slots=True)
class _OwnedValue:
    owner: Measurement
    value: object


@dataclass(slots=True)
class _RawHitIndex:
    owner: Measurement
    rings: list[list[Point]]
    edge_cells: dict[tuple[int, int], list[tuple[int, int]]] | None
    vertex_cells: dict[tuple[int, int], list[tuple[int, int]]] | None
    cell_size: float
    estimated_bytes: int
    compact_points: np.ndarray | None = None
    compact_next_indices: np.ndarray | None = None
    compact_ring_offsets: np.ndarray | None = None

    def _query_cells(self, point: Point, tolerance: float) -> list[tuple[int, int]]:
        radius = max(0.0, float(tolerance))
        min_col = math.floor((point.x - radius) / self.cell_size)
        max_col = math.floor((point.x + radius) / self.cell_size)
        min_row = math.floor((point.y - radius) / self.cell_size)
        max_row = math.floor((point.y + radius) / self.cell_size)
        return [
            (column, row)
            for column in range(min_col, max_col + 1)
            for row in range(min_row, max_row + 1)
        ]

    def near_edge(self, point: Point, tolerance: float) -> bool:
        if self.compact_points is not None and self.compact_next_indices is not None:
            return self._compact_near_edge(point, tolerance)
        candidate_ids: set[tuple[int, int]] = set()
        for cell in self._query_cells(point, tolerance):
            candidate_ids.update((self.edge_cells or {}).get(cell, ()))
        for ring_index, segment_index in candidate_ids:
            ring = self.rings[ring_index]
            if point_to_segment_distance(
                point,
                ring[segment_index],
                ring[(segment_index + 1) % len(ring)],
            ) <= tolerance:
                return True
        return False

    def nearest_vertex(
        self,
        point: Point,
        tolerance: float,
    ) -> tuple[int, int, float] | None:
        if self.compact_points is not None and self.compact_ring_offsets is not None:
            return self._compact_nearest_vertex(point, tolerance)
        candidate_ids: set[tuple[int, int]] = set()
        for cell in self._query_cells(point, tolerance):
            candidate_ids.update((self.vertex_cells or {}).get(cell, ()))
        nearest: tuple[int, int, float] | None = None
        tolerance_squared = float(tolerance) ** 2
        for ring_index, point_index in candidate_ids:
            candidate = self.rings[ring_index][point_index]
            distance_squared = ((candidate.x - point.x) ** 2) + ((candidate.y - point.y) ** 2)
            if distance_squared > tolerance_squared:
                continue
            candidate_distance = math.sqrt(distance_squared)
            if nearest is None or candidate_distance < nearest[2]:
                nearest = ring_index, point_index, candidate_distance
        return nearest

    def _compact_near_edge(self, point: Point, tolerance: float) -> bool:
        assert self.compact_points is not None
        assert self.compact_next_indices is not None
        query = np.asarray((point.x, point.y), dtype=np.float64)
        tolerance_squared = float(tolerance) ** 2
        total = int(self.compact_points.shape[0])
        for start in range(0, total, _HIT_INDEX_VECTOR_CHUNK_SIZE):
            stop = min(total, start + _HIT_INDEX_VECTOR_CHUNK_SIZE)
            segment_starts = self.compact_points[start:stop]
            segment_ends = self.compact_points[self.compact_next_indices[start:stop]]
            vectors = segment_ends - segment_starts
            offsets = query - segment_starts
            lengths_squared = np.einsum("ij,ij->i", vectors, vectors)
            projections = np.divide(
                np.einsum("ij,ij->i", offsets, vectors),
                lengths_squared,
                out=np.zeros(stop - start, dtype=np.float64),
                where=lengths_squared > 0.0,
            )
            np.clip(projections, 0.0, 1.0, out=projections)
            closest = segment_starts + (vectors * projections[:, None])
            differences = query - closest
            distances_squared = np.einsum("ij,ij->i", differences, differences)
            if bool(np.any(distances_squared <= tolerance_squared)):
                return True
        return False

    def _compact_nearest_vertex(
        self,
        point: Point,
        tolerance: float,
    ) -> tuple[int, int, float] | None:
        assert self.compact_points is not None
        assert self.compact_ring_offsets is not None
        query = np.asarray((point.x, point.y), dtype=np.float64)
        tolerance_squared = float(tolerance) ** 2
        nearest_flat_index = -1
        nearest_distance_squared = math.inf
        total = int(self.compact_points.shape[0])
        for start in range(0, total, _HIT_INDEX_VECTOR_CHUNK_SIZE):
            stop = min(total, start + _HIT_INDEX_VECTOR_CHUNK_SIZE)
            differences = self.compact_points[start:stop] - query
            distances_squared = np.einsum("ij,ij->i", differences, differences)
            distances_squared[~np.isfinite(distances_squared)] = math.inf
            local_index = int(np.argmin(distances_squared))
            local_distance = float(distances_squared[local_index])
            if local_distance < nearest_distance_squared:
                nearest_distance_squared = local_distance
                nearest_flat_index = start + local_index
        if nearest_flat_index < 0 or nearest_distance_squared > tolerance_squared:
            return None
        ring_index = int(
            np.searchsorted(
                self.compact_ring_offsets,
                nearest_flat_index,
                side="right",
            )
            - 1
        )
        point_index = nearest_flat_index - int(self.compact_ring_offsets[ring_index])
        return ring_index, point_index, math.sqrt(nearest_distance_squared)


def _measurement_key(measurement: Measurement) -> tuple[int, str, int]:
    return id(measurement), measurement.id, measurement.geometry_revision


def _raw_rings(measurement: Measurement) -> list[list[Point]]:
    if measurement.area_rings_px:
        return measurement.area_rings_px
    if len(measurement.polygon_px) >= 3:
        return [measurement.polygon_px]
    return []


def _raw_outline(measurement: Measurement, rings: list[list[Point]]) -> list[Point]:
    if len(measurement.polygon_px) >= 3:
        return measurement.polygon_px
    return rings[0] if rings else []


def _rings_path(rings: list[list[Point]]) -> QPainterPath:
    path = QPainterPath()
    path.setFillRule(Qt.FillRule.OddEvenFill)
    for ring in rings:
        if len(ring) < 3 or any(
            not math.isfinite(value)
            for point in ring
            for value in (point.x, point.y)
        ):
            continue
        path.moveTo(float(ring[0].x), float(ring[0].y))
        for point in ring[1:]:
            path.lineTo(float(point.x), float(point.y))
        path.closeSubpath()
    return path


def _path_bytes(path: QPainterPath) -> int:
    return max(256, int(path.elementCount()) * 48)


class AreaDerivedGeometryService:
    """Bounded runtime cache for exact area derivatives and screen proxies.

    Raw measurement coordinates remain the sole source of truth.  Screen
    proxies are optional, validated display artifacts and are never serialized
    or used for hit testing, numeric measurement, or export.
    """

    def __init__(self) -> None:
        self._known_revisions: OrderedDict[tuple[int, str], int] = OrderedDict()
        self._bounds: OrderedDict[tuple[int, str, int], _OwnedValue] = OrderedDict()
        self._moments: OrderedDict[tuple[int, str, int], _OwnedValue] = OrderedDict()
        self._hole_areas: OrderedDict[tuple[int, str, int], _OwnedValue] = OrderedDict()
        self._raw_paths: OrderedDict[tuple[int, str, int], _PathEntry] = OrderedDict()
        self._proxies: OrderedDict[tuple[int, str, int, int], _ProxyEntry] = OrderedDict()
        self._proxy_failures: OrderedDict[tuple[int, str, int, int], _OwnedValue] = OrderedDict()
        self._path_bytes = 0
        self._path_clock = 0
        self._hit_indexes: OrderedDict[tuple[int, str, int], _RawHitIndex] = OrderedDict()
        self._hit_index_bytes = 0

    def clear(self) -> None:
        self._known_revisions.clear()
        self._bounds.clear()
        self._moments.clear()
        self._hole_areas.clear()
        self._raw_paths.clear()
        self._proxies.clear()
        self._proxy_failures.clear()
        self._path_bytes = 0
        self._path_clock = 0
        self._hit_indexes.clear()
        self._hit_index_bytes = 0

    def discard_measurement(self, measurement: Measurement) -> None:
        identity = (id(measurement), measurement.id)
        self._known_revisions.pop(identity, None)
        for cache in (self._bounds, self._moments, self._hole_areas):
            for key in list(cache):
                if key[:2] == identity:
                    cache.pop(key, None)
        for key in list(self._raw_paths):
            if key[:2] == identity:
                self._remove_raw_path(key)
        for key in list(self._proxies):
            if key[:2] == identity:
                self._remove_proxy(key)
        for key in list(self._proxy_failures):
            if key[:2] == identity:
                self._proxy_failures.pop(key, None)
        for key in list(self._hit_indexes):
            if key[:2] == identity:
                self._remove_hit_index(key)

    def discard_document(self, measurements: list[Measurement]) -> None:
        """Release every derived entry owned by a document being closed."""

        for measurement in measurements:
            self.discard_measurement(measurement)

    def _purge_stale_revisions(self, measurement: Measurement) -> None:
        identity = (id(measurement), measurement.id)
        revision = measurement.geometry_revision
        previous_revision = self._known_revisions.get(identity)
        if previous_revision == revision:
            self._known_revisions.move_to_end(identity)
            return
        self._known_revisions[identity] = revision
        self._known_revisions.move_to_end(identity)
        while len(self._known_revisions) > (_SCALAR_MAX_ENTRIES * 2):
            self._known_revisions.popitem(last=False)
        if previous_revision is None:
            return
        for cache in (self._bounds, self._moments, self._hole_areas):
            for key in list(cache):
                if key[:2] == identity and key[2] != revision:
                    cache.pop(key, None)
        for key in list(self._raw_paths):
            if key[:2] == identity and key[2] != revision:
                self._remove_raw_path(key)
        for key in list(self._proxies):
            if key[:2] == identity and key[2] != revision:
                self._remove_proxy(key)
        for key in list(self._proxy_failures):
            if key[:2] == identity and key[2] != revision:
                self._proxy_failures.pop(key, None)
        for key in list(self._hit_indexes):
            if key[:2] == identity and key[2] != revision:
                self._remove_hit_index(key)

    @staticmethod
    def _owned_value(
        cache: OrderedDict[tuple[int, str, int], _OwnedValue],
        key: tuple[int, str, int],
        measurement: Measurement,
    ) -> object | None:
        entry = cache.get(key)
        if entry is None:
            return None
        if entry.owner is not measurement:
            cache.pop(key, None)
            return None
        cache.move_to_end(key)
        return entry.value

    @staticmethod
    def _store_owned_value(
        cache: OrderedDict[tuple[int, str, int], _OwnedValue],
        key: tuple[int, str, int],
        measurement: Measurement,
        value: object,
    ) -> None:
        cache[key] = _OwnedValue(owner=measurement, value=value)
        cache.move_to_end(key)
        while len(cache) > _SCALAR_MAX_ENTRIES:
            cache.popitem(last=False)

    def raw_bounds(self, measurement: Measurement) -> tuple[float, float, float, float] | None:
        self._purge_stale_revisions(measurement)
        key = _measurement_key(measurement)
        cached = self._owned_value(self._bounds, key, measurement)
        if cached is not None:
            return cached  # type: ignore[return-value]
        rings = _raw_rings(measurement)
        outline = _raw_outline(measurement, rings)
        if rings:
            bounds: tuple[float, float, float, float] | None = area_rings_bounds(rings)
        elif outline:
            bounds = polygon_bounds(outline)
        else:
            bounds = None
        self._store_owned_value(self._bounds, key, measurement, bounds)
        return bounds

    def _area_moments(self, measurement: Measurement) -> tuple[float, Point]:
        self._purge_stale_revisions(measurement)
        key = _measurement_key(measurement)
        cached = self._owned_value(self._moments, key, measurement)
        if cached is not None:
            return cached  # type: ignore[return-value]
        rings = _raw_rings(measurement)
        moments = area_rings_area_and_centroid(rings)
        self._store_owned_value(self._moments, key, measurement, moments)
        return moments

    def hole_area(self, measurement: Measurement) -> float:
        self._purge_stale_revisions(measurement)
        key = _measurement_key(measurement)
        cached = self._owned_value(self._hole_areas, key, measurement)
        if cached is not None:
            return float(cached)
        hole_area = float(area_rings_hole_area(_raw_rings(measurement)))
        self._store_owned_value(self._hole_areas, key, measurement, hole_area)
        return hole_area

    def vector_area(self, measurement: Measurement) -> float:
        return float(self._area_moments(measurement)[0])

    def centroid(self, measurement: Measurement) -> Point:
        return self._area_moments(measurement)[1]

    def cached_centroid(self, measurement: Measurement) -> Point | None:
        """Return an already verified RAW centroid without doing heavy work."""

        self._purge_stale_revisions(measurement)
        cached = self._owned_value(
            self._moments,
            _measurement_key(measurement),
            measurement,
        )
        if cached is None:
            return None
        return cached[1]  # type: ignore[index,return-value]

    def scalar_geometry(self, measurement: Measurement) -> AreaScalarGeometry:
        vector_area, centroid = self._area_moments(measurement)
        return AreaScalarGeometry(
            bounds=self.raw_bounds(measurement),
            centroid=centroid,
            hole_area_px=self.hole_area(measurement),
            vector_area_px=float(vector_area),
        )

    def raw_path(self, measurement: Measurement) -> QPainterPath:
        self._purge_stale_revisions(measurement)
        key = _measurement_key(measurement)
        cached = self._raw_paths.get(key)
        if cached is not None and cached.owner is measurement:
            self._raw_paths.move_to_end(key)
            cached.last_used = self._next_path_clock()
            return cached.path
        if cached is not None:
            self._remove_raw_path(key)
        path = _rings_path(_raw_rings(measurement))
        estimated_bytes = _path_bytes(path)
        if estimated_bytes > _PATH_MAX_ESTIMATED_BYTES:
            return path
        self._evict_paths_for(estimated_bytes)
        self._raw_paths[key] = _PathEntry(
            owner=measurement,
            path=path,
            estimated_bytes=estimated_bytes,
            last_used=self._next_path_clock(),
        )
        self._path_bytes += estimated_bytes
        return path

    def raw_geometry(self, measurement: Measurement) -> AreaGeometryView:
        rings = _raw_rings(measurement)
        outline = _raw_outline(measurement, rings)
        return AreaGeometryView(
            outline_points=outline,
            fill_rings=rings,
            bounds=self.raw_bounds(measurement),
            path=self.raw_path(measurement),
            source=AREA_GEOMETRY_RAW,
        )

    def screen_geometry(
        self,
        measurement: Measurement,
        *,
        zoom: float,
        selected: bool,
        build_budget: AreaProxyBuildBudget | None = None,
    ) -> AreaGeometryView:
        self._purge_stale_revisions(measurement)
        rings = _raw_rings(measurement)
        total_vertices = sum(len(ring) for ring in rings)
        if (
            selected
            or os.environ.get("FDM_DISABLE_AREA_SCREEN_PROXY", "").strip() == "1"
            # QPainterPath.simplified() can become super-linear for extremely
            # dense paths.  Returning the exact RAW path is both safer and much
            # faster than blocking the UI while validating an optional proxy.
            or total_vertices > SCREEN_PROXY_MAX_SOURCE_VERTICES
            or (
                total_vertices <= SCREEN_PROXY_VERTEX_THRESHOLD
                and not is_magic_segment_area(measurement)
            )
        ):
            return self.raw_geometry(measurement)
        bucket = self._zoom_bucket(zoom)
        key = (*_measurement_key(measurement), bucket)
        cached = self._proxies.get(key)
        if cached is not None and cached.owner is not measurement:
            self._remove_proxy(key)
            cached = None
        failed = self._proxy_failures.get(key)
        if failed is not None and failed.owner is not measurement:
            self._proxy_failures.pop(key, None)
            failed = None
        if cached is not None:
            self._proxies.move_to_end(key)
            cached.last_used = self._next_path_clock()
            return AreaGeometryView(
                outline_points=cached.outline_points,
                fill_rings=cached.fill_rings,
                bounds=cached.bounds,
                path=cached.path,
                source=AREA_GEOMETRY_SCREEN,
                zoom_bucket=bucket,
            )
        if failed is not None:
            self._proxy_failures.move_to_end(key)
            return self.raw_geometry(measurement)
        if build_budget is not None and not build_budget.permits(total_vertices):
            raw = self.raw_geometry(measurement)
            if build_budget.last_request_deferred:
                return AreaGeometryView(
                    outline_points=raw.outline_points,
                    fill_rings=raw.fill_rings,
                    bounds=raw.bounds,
                    path=raw.path,
                    source=raw.source,
                    proxy_deferred=True,
                )
            return raw
        raw = self.raw_geometry(measurement)
        started = time.perf_counter()
        cached = self._build_proxy(measurement, raw, bucket)
        if build_budget is not None:
            build_budget.record(time.perf_counter() - started)
        if cached is None:
            self._proxy_failures[key] = _OwnedValue(owner=measurement, value=True)
            self._proxy_failures.move_to_end(key)
            while len(self._proxy_failures) > _PATH_MAX_ENTRIES:
                self._proxy_failures.popitem(last=False)
            return raw
        if cached.estimated_bytes > _PATH_MAX_ESTIMATED_BYTES:
            return raw
        self._evict_measurement_proxy_buckets(key[:3])
        # A validated proxy becomes the normal unselected screen path. Keep at
        # most one path representation for that object in the shared budget;
        # exact hit testing/export can rebuild RAW from the untouched rings.
        self._remove_raw_path(_measurement_key(measurement))
        self._evict_paths_for(cached.estimated_bytes)
        self._proxies[key] = cached
        self._path_bytes += cached.estimated_bytes
        return AreaGeometryView(
            outline_points=cached.outline_points,
            fill_rings=cached.fill_rings,
            bounds=cached.bounds,
            path=cached.path,
            source=AREA_GEOMETRY_SCREEN,
            zoom_bucket=bucket,
        )

    @staticmethod
    def _zoom_bucket(zoom: float) -> int:
        safe_zoom = max(float(zoom), 1e-9)
        return int(round(math.log(safe_zoom, _SQRT_TWO)))

    @staticmethod
    def _bucket_upper_zoom(bucket: int) -> float:
        # round(log_base_sqrt(2)) buckets extend one half-step above their
        # representative zoom.  Use that edge to keep the entire bucket at or
        # below the advertised 0.5 logical-pixel error.
        return (_SQRT_TWO**bucket) * (2.0**0.25)

    @classmethod
    def _proxy_epsilon_for_bucket(cls, bucket: int) -> float:
        # Keep substantially below the visual ceiling. The conservative base
        # epsilon makes the third and final /4 attempt satisfy the strict 0.1%
        # area guard for narrow smooth contours while remaining within the
        # half-pixel budget everywhere in the zoom bucket.
        safe_zoom = cls._bucket_upper_zoom(bucket) * 4.0
        return SCREEN_PROXY_MAX_ERROR_PX / max(safe_zoom, 1e-9)

    def _build_proxy(
        self,
        measurement: Measurement,
        raw: AreaGeometryView,
        bucket: int,
    ) -> _ProxyEntry | None:
        if not raw.fill_rings or raw.bounds is None:
            return None
        epsilon = self._proxy_epsilon_for_bucket(bucket)
        try:
            raw_simplified = raw.path.simplified()
            raw_component_count = len(raw_simplified.toSubpathPolygons())
        except Exception:  # pragma: no cover - defensive Qt backend fallback
            return None
        moments_key = _measurement_key(measurement)
        cached_moments = self._owned_value(self._moments, moments_key, measurement)
        if cached_moments is None:
            raw_area, moment_x, moment_y = odd_even_path_moments(
                raw_simplified,
                path_is_simplified=True,
            )
            if raw_area <= 1e-9 or raw.bounds is None:
                centroid = (
                    Point(0.0, 0.0)
                    if raw.bounds is None
                    else Point(
                        (raw.bounds[0] + raw.bounds[2]) / 2.0,
                        (raw.bounds[1] + raw.bounds[3]) / 2.0,
                    )
                )
            else:
                centroid = Point(moment_x / raw_area, moment_y / raw_area)
            cached_moments = (raw_area, centroid)
            self._store_owned_value(self._moments, moments_key, measurement, cached_moments)
        raw_area = float(cached_moments[0])
        for attempt_epsilon in (
            epsilon,
            epsilon / 2.0,
            epsilon / 4.0,
        ):
            rings = [_simplify_ring(ring, attempt_epsilon) for ring in raw.fill_rings]
            proxy_path = _rings_path(rings)
            if not self._valid_proxy(
                raw,
                rings,
                proxy_path,
                attempt_epsilon,
                raw_area=raw_area,
                raw_component_count=raw_component_count,
            ):
                continue
            outline_index = 0
            outline = rings[outline_index]
            if len(measurement.polygon_px) >= 3:
                # polygon_px is normally the first ring, but may be a separate
                # outline for older projects.  Keep display topology tied to
                # the validated fill rings and never mutate the stored outline.
                outline = rings[0]
            estimated_bytes = _path_bytes(proxy_path) + _proxy_points_bytes(rings)
            return _ProxyEntry(
                owner=measurement,
                outline_points=outline,
                fill_rings=rings,
                bounds=area_rings_bounds(rings),
                path=proxy_path,
                estimated_bytes=estimated_bytes,
                last_used=self._next_path_clock(),
            )
        return None

    def _valid_proxy(
        self,
        raw: AreaGeometryView,
        rings: list[list[Point]],
        proxy_path: QPainterPath,
        epsilon: float,
        *,
        raw_area: float,
        raw_component_count: int,
    ) -> bool:
        if len(rings) != len(raw.fill_rings) or any(len(ring) < 3 for ring in rings):
            return False
        if any(
            not math.isfinite(value)
            for ring in rings
            for point in ring
            for value in (point.x, point.y)
        ):
            return False
        for original, reduced in zip(raw.fill_rings, rings, strict=True):
            original_sign = math.copysign(1.0, ring_signed_area(original) or 1.0)
            reduced_sign = math.copysign(1.0, ring_signed_area(reduced) or 1.0)
            if original_sign != reduced_sign:
                return False
        proxy_bounds = area_rings_bounds(rings)
        if raw.bounds is None or any(
            abs(float(left) - float(right)) > (epsilon + 1e-6)
            for left, right in zip(raw.bounds, proxy_bounds, strict=True)
        ):
            return False
        try:
            proxy_simplified = proxy_path.simplified()
            proxy_component_count = len(proxy_simplified.toSubpathPolygons())
            proxy_area = float(
                odd_even_path_moments(
                    proxy_simplified,
                    path_is_simplified=True,
                )[0]
            )
        except Exception:  # pragma: no cover - defensive Qt backend fallback
            return False
        if not math.isfinite(raw_area) or not math.isfinite(proxy_area):
            return False
        if abs(proxy_area - raw_area) > max(1.0, abs(raw_area) * 0.001):
            return False
        if raw_component_count != proxy_component_count:
            return False
        for index, ring in enumerate(raw.fill_rings):
            sample = _interior_sample(ring)
            if sample is None:
                return False
            raw_filled = raw.path.contains(sample)
            proxy_filled = proxy_path.contains(sample)
            if raw_filled != proxy_filled:
                return False
            if index > 0 and raw_filled:
                # Existing rings that do not describe an actual odd-even hole
                # are left completely raw rather than guessed at.
                return False
        return True

    def _evict_measurement_proxy_buckets(self, measurement_key: tuple[int, str, int]) -> None:
        matching = [key for key in self._proxies if key[:3] == measurement_key]
        while len(matching) >= _PROXY_BUCKETS_PER_MEASUREMENT:
            key = matching.pop(0)
            self._remove_proxy(key)

    def _next_path_clock(self) -> int:
        self._path_clock += 1
        return self._path_clock

    def _remove_raw_path(self, key: tuple[int, str, int]) -> None:
        entry = self._raw_paths.pop(key, None)
        if entry is not None:
            self._path_bytes = max(0, self._path_bytes - entry.estimated_bytes)

    def _remove_proxy(self, key: tuple[int, str, int, int]) -> None:
        entry = self._proxies.pop(key, None)
        if entry is not None:
            self._path_bytes = max(0, self._path_bytes - entry.estimated_bytes)

    def _evict_paths_for(self, required_bytes: int) -> None:
        while (self._raw_paths or self._proxies) and (
            len(self._raw_paths) + len(self._proxies) >= _PATH_MAX_ENTRIES
            or self._path_bytes + required_bytes > _PATH_MAX_ESTIMATED_BYTES
        ):
            raw_item = next(iter(self._raw_paths.items()), None)
            proxy_item = next(iter(self._proxies.items()), None)
            if proxy_item is None or (
                raw_item is not None
                and raw_item[1].last_used <= proxy_item[1].last_used
            ):
                self._remove_raw_path(raw_item[0])  # type: ignore[index]
            else:
                self._remove_proxy(proxy_item[0])

    def contains_raw(self, measurement: Measurement, point: Point) -> bool:
        """Use the exact odd-even raw path for interior hit testing."""

        if not (math.isfinite(point.x) and math.isfinite(point.y)):
            return False
        return self.raw_path(measurement).contains(QPointF(float(point.x), float(point.y)))

    def near_edge(self, measurement: Measurement, point: Point, tolerance: float) -> bool:
        if tolerance < 0.0 or not (math.isfinite(point.x) and math.isfinite(point.y)):
            return False
        return self._raw_hit_index(measurement).near_edge(point, float(tolerance))

    def nearest_vertex(
        self,
        measurement: Measurement,
        point: Point,
        tolerance: float,
    ) -> tuple[int, int, float] | None:
        if tolerance < 0.0 or not (math.isfinite(point.x) and math.isfinite(point.y)):
            return None
        return self._raw_hit_index(measurement).nearest_vertex(point, float(tolerance))

    def _raw_hit_index(self, measurement: Measurement) -> _RawHitIndex:
        self._purge_stale_revisions(measurement)
        key = _measurement_key(measurement)
        cached = self._hit_indexes.get(key)
        if cached is not None and cached.owner is measurement:
            self._hit_indexes.move_to_end(key)
            return cached
        if cached is not None:
            self._remove_hit_index(key)
        index = _build_raw_hit_index(measurement, _raw_rings(measurement))
        if index.estimated_bytes > _HIT_INDEX_MAX_ESTIMATED_BYTES:
            return index
        while self._hit_indexes and (
            len(self._hit_indexes) >= _HIT_INDEX_MAX_ENTRIES
            or self._hit_index_bytes + index.estimated_bytes > _HIT_INDEX_MAX_ESTIMATED_BYTES
        ):
            oldest_key = next(iter(self._hit_indexes))
            self._remove_hit_index(oldest_key)
        self._hit_indexes[key] = index
        self._hit_index_bytes += index.estimated_bytes
        return index

    def _remove_hit_index(self, key: tuple[int, str, int]) -> None:
        entry = self._hit_indexes.pop(key, None)
        if entry is not None:
            self._hit_index_bytes = max(0, self._hit_index_bytes - entry.estimated_bytes)


def _proxy_points_bytes(rings: list[list[Point]]) -> int:
    point_count = sum(len(ring) for ring in rings)
    return 256 + (point_count * _PROXY_POINT_ESTIMATED_BYTES) + (len(rings) * 64)


def _compact_hit_index_estimated_bytes(point_count: int, ring_count: int) -> int:
    # float64 x/y + int32 next index + int64 ring offsets
    return 512 + (max(0, point_count) * 20) + ((max(0, ring_count) + 1) * 8)


def _build_compact_raw_hit_index(
    measurement: Measurement,
    rings: list[list[Point]],
) -> _RawHitIndex:
    point_count = sum(len(ring) for ring in rings)
    coordinates = np.empty((point_count, 2), dtype=np.float64)
    next_indices = np.empty(point_count, dtype=np.int32)
    ring_offsets = np.empty(len(rings) + 1, dtype=np.int64)
    cursor = 0
    ring_offsets[0] = 0
    for ring_index, ring in enumerate(rings):
        ring_length = len(ring)
        if ring_length:
            values = np.fromiter(
                (value for point in ring for value in (point.x, point.y)),
                dtype=np.float64,
                count=ring_length * 2,
            ).reshape((ring_length, 2))
            coordinates[cursor : cursor + ring_length] = values
            next_indices[cursor : cursor + ring_length] = np.arange(
                cursor + 1,
                cursor + ring_length + 1,
                dtype=np.int32,
            )
            next_indices[cursor + ring_length - 1] = cursor
            cursor += ring_length
        ring_offsets[ring_index + 1] = cursor
    return _RawHitIndex(
        owner=measurement,
        rings=rings,
        edge_cells=None,
        vertex_cells=None,
        cell_size=_HIT_INDEX_CELL_SIZE,
        estimated_bytes=(
            int(coordinates.nbytes)
            + int(next_indices.nbytes)
            + int(ring_offsets.nbytes)
            + 512
        ),
        compact_points=coordinates,
        compact_next_indices=next_indices,
        compact_ring_offsets=ring_offsets,
    )


def _build_raw_hit_index(measurement: Measurement, rings: list[list[Point]]) -> _RawHitIndex:
    point_count = sum(len(ring) for ring in rings)
    rough_grid_bytes = 512 + (point_count * 160) + (len(rings) * 128)
    if (
        point_count >= _HIT_INDEX_COMPACT_POINT_THRESHOLD
        or rough_grid_bytes > _HIT_INDEX_MAX_ESTIMATED_BYTES
    ):
        return _build_compact_raw_hit_index(measurement, rings)
    edge_cells: dict[tuple[int, int], list[tuple[int, int]]] = {}
    vertex_cells: dict[tuple[int, int], list[tuple[int, int]]] = {}
    cell_size = _HIT_INDEX_CELL_SIZE
    edge_memberships = 0
    vertex_memberships = 0
    for ring_index, ring in enumerate(rings):
        if len(ring) < 2:
            continue
        for point_index, point in enumerate(ring):
            if not (math.isfinite(point.x) and math.isfinite(point.y)):
                continue
            vertex_cell = (math.floor(point.x / cell_size), math.floor(point.y / cell_size))
            vertex_cells.setdefault(vertex_cell, []).append((ring_index, point_index))
            vertex_memberships += 1
            next_point = ring[(point_index + 1) % len(ring)]
            if not (math.isfinite(next_point.x) and math.isfinite(next_point.y)):
                continue
            for cell in _segment_grid_cells(point, next_point, cell_size):
                edge_cells.setdefault(cell, []).append((ring_index, point_index))
                edge_memberships += 1
    cell_count = len(edge_cells) + len(vertex_cells)
    estimated_bytes = (
        512
        + (cell_count * 128)
        + ((edge_memberships + vertex_memberships) * 80)
    )
    return _RawHitIndex(
        owner=measurement,
        rings=rings,
        edge_cells=edge_cells,
        vertex_cells=vertex_cells,
        cell_size=cell_size,
        estimated_bytes=estimated_bytes,
    )


def _segment_grid_cells(start: Point, end: Point, cell_size: float) -> list[tuple[int, int]]:
    """Return grid cells crossed by a segment without filling its bounding box."""

    start_col = math.floor(start.x / cell_size)
    start_row = math.floor(start.y / cell_size)
    end_col = math.floor(end.x / cell_size)
    end_row = math.floor(end.y / cell_size)
    column = start_col
    row = start_row
    cells = [(column, row)]
    if (column, row) == (end_col, end_row):
        return cells

    dx = end.x - start.x
    dy = end.y - start.y
    step_x = 1 if dx > 0.0 else (-1 if dx < 0.0 else 0)
    step_y = 1 if dy > 0.0 else (-1 if dy < 0.0 else 0)
    if step_x:
        next_x = (column + (1 if step_x > 0 else 0)) * cell_size
        t_max_x = (next_x - start.x) / dx
        t_delta_x = cell_size / abs(dx)
    else:
        t_max_x = math.inf
        t_delta_x = math.inf
    if step_y:
        next_y = (row + (1 if step_y > 0 else 0)) * cell_size
        t_max_y = (next_y - start.y) / dy
        t_delta_y = cell_size / abs(dy)
    else:
        t_max_y = math.inf
        t_delta_y = math.inf

    max_steps = abs(end_col - start_col) + abs(end_row - start_row) + 2
    for _ in range(max_steps):
        if (column, row) == (end_col, end_row):
            break
        if t_max_x < t_max_y:
            column += step_x
            t_max_x += t_delta_x
        elif t_max_y < t_max_x:
            row += step_y
            t_max_y += t_delta_y
        else:
            column += step_x
            row += step_y
            t_max_x += t_delta_x
            t_max_y += t_delta_y
        cells.append((column, row))
    return cells


def _simplify_ring(points: list[Point], epsilon: float) -> list[Point]:
    cleaned = clean_ring(points, collinear_epsilon=1e-6)
    if len(cleaned) < 3:
        return cleaned
    contour = np.asarray([[[point.x, point.y]] for point in cleaned], dtype=np.float32)
    approximation = cv2.approxPolyDP(contour, max(float(epsilon), 1e-9), True)
    reduced = [Point(float(item[0][0]), float(item[0][1])) for item in approximation]
    reduced = clean_ring(reduced, collinear_epsilon=1e-6)
    if len(reduced) < 3:
        return cleaned
    if ring_signed_area(cleaned) * ring_signed_area(reduced) < 0:
        reduced.reverse()
    return reduced


def _interior_sample(ring: list[Point]) -> QPointF | None:
    if len(ring) < 3:
        return None
    path = _rings_path([ring])
    bounds = path.boundingRect()
    center = bounds.center()
    if path.contains(center):
        return center
    # A small deterministic grid is sufficient for validation; failure simply
    # disables the optional proxy and falls back to exact raw geometry.
    for divisions in (5, 9, 17):
        for row in range(divisions):
            y = bounds.top() + ((row + 0.5) / divisions) * bounds.height()
            for column in range(divisions):
                x = bounds.left() + ((column + 0.5) / divisions) * bounds.width()
                candidate = QPointF(x, y)
                if path.contains(candidate):
                    return candidate
    return None


area_derived_geometry_service = AreaDerivedGeometryService()


def is_magic_segment_area(measurement: Measurement) -> bool:
    # Kept for compatibility with callers/tests that distinguish inference
    # provenance.  Proxy eligibility is now based on density, not mode.
    return measurement.measurement_kind == "area" and measurement.mode == "magic_segment"


def invalidate_measurement_display_geometry(measurement: Measurement) -> None:
    area_derived_geometry_service.discard_measurement(measurement)
    measurement.display_polygon_px = []
    measurement.display_area_rings_px = []
    measurement.display_bounds_px = None


def ensure_measurement_display_geometry(measurement: Measurement) -> None:
    """Populate legacy runtime fields from the validated 1:1 screen proxy."""

    geometry = area_derived_geometry_service.screen_geometry(measurement, zoom=1.0, selected=False)
    if geometry.source != AREA_GEOMETRY_SCREEN:
        measurement.display_polygon_px = []
        measurement.display_area_rings_px = []
        measurement.display_bounds_px = None
        return
    measurement.display_polygon_px = list(geometry.outline_points)
    measurement.display_area_rings_px = [list(ring) for ring in geometry.fill_rings]
    measurement.display_bounds_px = geometry.bounds


def area_geometry_raw(measurement: Measurement) -> AreaGeometryView:
    return area_derived_geometry_service.raw_geometry(measurement)


def area_geometry_for_display(
    measurement: Measurement,
    *,
    selected: bool,
    zoom: float = 1.0,
) -> tuple[list[Point], list[list[Point]], tuple[float, float, float, float] | None]:
    geometry = area_derived_geometry_service.screen_geometry(
        measurement,
        zoom=zoom,
        selected=selected,
    )
    return geometry.outline_points, geometry.fill_rings, geometry.bounds


def clear_area_derived_geometry_cache() -> None:
    area_derived_geometry_service.clear()
