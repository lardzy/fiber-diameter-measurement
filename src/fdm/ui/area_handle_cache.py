from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import math

from fdm.geometry import Point
from fdm.models import Measurement


AREA_HANDLE_CACHE_MAX_ENTRIES = 256
AREA_HANDLE_CACHE_MAX_BYTES = 16 * 1024 * 1024
AREA_HANDLE_CACHE_MAX_BUCKETS_PER_MEASUREMENT = 3
_ESTIMATED_COORDINATE_BYTES = 64


@dataclass(frozen=True, slots=True)
class AreaHandleCacheStats:
    entries: int
    bytes: int
    hits: int
    misses: int
    evictions: int


@dataclass(frozen=True, slots=True)
class _AreaHandleCacheKey:
    object_token: int
    measurement_id: str
    geometry_revision: int
    output_scale: float
    device_pixel_ratio: float
    spacing_px: float


@dataclass(slots=True)
class _AreaHandleCacheEntry:
    coordinates: tuple[tuple[float, float], ...]
    estimated_bytes: int


class AreaHandleDisplayCache:
    """Bounded cache for the thinned, display-only area editing handles.

    Exact hit testing continues to use every RAW vertex.  The cache stores
    detached numeric coordinates, so neither painting nor eviction can mutate
    a measurement's persistent rings.
    """

    def __init__(
        self,
        *,
        max_entries: int = AREA_HANDLE_CACHE_MAX_ENTRIES,
        max_bytes: int = AREA_HANDLE_CACHE_MAX_BYTES,
    ) -> None:
        self._max_entries = max(1, int(max_entries))
        self._max_bytes = max(1, int(max_bytes))
        self._entries: OrderedDict[
            _AreaHandleCacheKey,
            _AreaHandleCacheEntry,
        ] = OrderedDict()
        self._owner_keys: dict[int, set[_AreaHandleCacheKey]] = {}
        self._bytes = 0
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def coordinates(
        self,
        measurement: Measurement,
        rings: list[list[Point]],
        *,
        output_scale: float,
        device_pixel_ratio: float,
        spacing_px: float = 8.0,
    ) -> tuple[tuple[float, float], ...]:
        normalized_scale = max(1e-9, float(output_scale))
        normalized_dpr = max(1.0, float(device_pixel_ratio))
        normalized_spacing = max(1.0, float(spacing_px))
        key = _AreaHandleCacheKey(
            object_token=id(measurement),
            measurement_id=measurement.id,
            geometry_revision=measurement.geometry_revision,
            output_scale=round(normalized_scale, 8),
            device_pixel_ratio=round(normalized_dpr, 4),
            spacing_px=round(normalized_spacing, 4),
        )
        cached = self._entries.get(key)
        if cached is not None:
            self._entries.move_to_end(key)
            self._hits += 1
            return cached.coordinates
        self._misses += 1
        self._discard_stale_revisions(measurement, keep=key)
        coordinates = self._build_coordinates(
            rings,
            output_scale=normalized_scale,
            spacing_px=normalized_spacing,
        )
        estimated_bytes = max(
            1,
            len(coordinates) * _ESTIMATED_COORDINATE_BYTES,
        )
        if estimated_bytes > self._max_bytes:
            return coordinates
        self._evict_for(estimated_bytes)
        entry = _AreaHandleCacheEntry(
            coordinates=coordinates,
            estimated_bytes=estimated_bytes,
        )
        self._entries[key] = entry
        self._entries.move_to_end(key)
        self._owner_keys.setdefault(id(measurement), set()).add(key)
        self._bytes += estimated_bytes
        self._limit_owner_buckets(id(measurement), keep=key)
        return coordinates

    def discard_measurement(self, measurement: Measurement) -> None:
        self._discard_owner(id(measurement))

    def discard_document(self, measurements: list[Measurement]) -> None:
        for measurement in measurements:
            self._discard_owner(id(measurement))

    def clear(self) -> None:
        self._entries.clear()
        self._owner_keys.clear()
        self._bytes = 0

    def stats(self) -> AreaHandleCacheStats:
        return AreaHandleCacheStats(
            entries=len(self._entries),
            bytes=self._bytes,
            hits=self._hits,
            misses=self._misses,
            evictions=self._evictions,
        )

    @staticmethod
    def _build_coordinates(
        rings: list[list[Point]],
        *,
        output_scale: float,
        spacing_px: float,
    ) -> tuple[tuple[float, float], ...]:
        cell_size = max(1e-9, spacing_px / output_scale)
        occupied: set[tuple[int, int]] = set()
        coordinates: list[tuple[float, float]] = []
        for ring in rings:
            for point in ring:
                if not math.isfinite(point.x) or not math.isfinite(point.y):
                    continue
                cell = (
                    math.floor(point.x / cell_size),
                    math.floor(point.y / cell_size),
                )
                if cell in occupied:
                    continue
                occupied.add(cell)
                coordinates.append((float(point.x), float(point.y)))
        return tuple(coordinates)

    def _discard_stale_revisions(
        self,
        measurement: Measurement,
        *,
        keep: _AreaHandleCacheKey,
    ) -> None:
        owner_token = id(measurement)
        for key in tuple(self._owner_keys.get(owner_token, ())):
            if (
                key.measurement_id != keep.measurement_id
                or key.geometry_revision != keep.geometry_revision
            ):
                self._remove(key)

    def _discard_owner(self, owner_token: int) -> None:
        for key in tuple(self._owner_keys.get(owner_token, ())):
            self._remove(key)

    def _evict_for(self, required_bytes: int) -> None:
        while self._entries and (
            len(self._entries) >= self._max_entries
            or self._bytes + required_bytes > self._max_bytes
        ):
            self._remove(next(iter(self._entries)), evicted=True)

    def _limit_owner_buckets(
        self,
        owner_token: int,
        *,
        keep: _AreaHandleCacheKey,
    ) -> None:
        owner_keys = self._owner_keys.get(owner_token)
        while (
            owner_keys is not None
            and len(owner_keys) > AREA_HANDLE_CACHE_MAX_BUCKETS_PER_MEASUREMENT
        ):
            oldest = next(
                (
                    candidate
                    for candidate in self._entries
                    if candidate.object_token == owner_token and candidate != keep
                ),
                None,
            )
            if oldest is None:
                break
            self._remove(oldest, evicted=True)
            owner_keys = self._owner_keys.get(owner_token)

    def _remove(
        self,
        key: _AreaHandleCacheKey,
        *,
        evicted: bool = False,
    ) -> None:
        entry = self._entries.pop(key, None)
        if entry is None:
            return
        self._bytes = max(0, self._bytes - entry.estimated_bytes)
        keys = self._owner_keys.get(key.object_token)
        if keys is not None:
            keys.discard(key)
            if not keys:
                self._owner_keys.pop(key.object_token, None)
        if evicted:
            self._evictions += 1


area_handle_display_cache = AreaHandleDisplayCache()
