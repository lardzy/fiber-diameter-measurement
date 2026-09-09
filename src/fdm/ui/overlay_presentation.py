"""Display-only front/pending generations of the bounded overlay tile cache.

This controller owns keys, never image buffers or measurement geometry. Old
tiles remain in the workspace cache's byte budget and may be evicted normally.
Hit testing, exports and model commands must never consult this state.
"""

from dataclasses import dataclass, field, replace

from fdm.ui.canvas_overlay_cache import CanvasOverlayTileKey


def coordinate(key: CanvasOverlayTileKey) -> tuple[int, int]:
    return key.tile_x, key.tile_y


@dataclass(frozen=True)
class OverlayGestureSnapshot:
    measurement_id: str
    placements: tuple
    background_signature: object
    palette_key: int
    zoom: float
    device_pixel_ratio: float
    replaces_selection: bool


@dataclass
class _PendingRegion:
    coordinates: set[tuple[int, int]]
    measurement_ids: set[str] = field(default_factory=set)
    old_bounds: dict[str, tuple[float, float, float, float]] = field(
        default_factory=dict
    )


class OverlayPresentation:
    def __init__(self):
        self.namespace = None
        self.front: dict[tuple[int, int], CanvasOverlayTileKey] = {}
        self.regions: list[_PendingRegion] = []
        self._retired: set[CanvasOverlayTileKey] = set()
        self._released: set[CanvasOverlayTileKey] = set()
        self.completed_ids: set[str] = set()

    def reset(self):
        self._released.update(self._retired)
        self._retired.clear()
        self.front.clear()
        self.regions.clear()
        self.completed_ids.clear()
        self.namespace = None

    @property
    def pending_ids(self):
        return {
            identity for region in self.regions for identity in region.measurement_ids
        }

    def old_bounds(self, identity):
        bounds = [
            region.old_bounds[identity]
            for region in self.regions
            if identity in region.old_bounds
        ]
        if not bounds:
            return None
        return (
            min(b[0] for b in bounds),
            min(b[1] for b in bounds),
            max(b[2] for b in bounds),
            max(b[3] for b in bounds),
        )

    def stage(self, invalidated, *, measurement_ids=(), old_bounds=None):
        """Join overlapping edits; independent regions may publish separately."""
        affected = {
            pos
            for pos, key in self.front.items()
            if (key.zoom, key.device_pixel_ratio, *pos) in invalidated
        }
        if not affected:
            return frozenset()
        region = _PendingRegion(affected, set(measurement_ids), dict(old_bounds or {}))
        remaining = []
        # Iterate to a fixed point: a new edit can bridge two existing regions.
        candidates = list(self.regions)
        while candidates:
            previous = candidates.pop(0)
            if region.coordinates & previous.coordinates:
                region.coordinates.update(previous.coordinates)
                region.measurement_ids.update(previous.measurement_ids)
                for identity, bounds in previous.old_bounds.items():
                    current = region.old_bounds.get(identity)
                    region.old_bounds[identity] = (
                        bounds
                        if current is None
                        else (
                            min(bounds[0], current[0]),
                            min(bounds[1], current[1]),
                            max(bounds[2], current[2]),
                            max(bounds[3], current[3]),
                        )
                    )
                candidates.extend(remaining)
                remaining.clear()
            else:
                remaining.append(previous)
        self.regions = [*remaining, region]
        retained = frozenset(
            self.front[pos] for pos in region.coordinates if pos in self.front
        )
        self._retired.update(retained)
        return retained

    def resolve(
        self,
        keys,
        contains,
        *,
        failed=(),
        fallback_ready=False,
        publish=True,
        paintable=None,
    ):
        """Publish complete visible regions and return their changed tile keys."""
        if not keys:
            self.reset()
            return []
        namespace = replace(keys[0], tile_x=0, tile_y=0, tile_epoch=0)
        if namespace != self.namespace:
            self.reset()
            self.namespace = namespace
        targets = {coordinate(key): key for key in keys}
        self.front = {
            pos: key
            for pos, key in self.front.items()
            if pos in targets and contains(key)
        }
        pending = []
        changed = []
        for region in self.regions:
            region.coordinates.intersection_update(targets)
            fallback = fallback_ready and any(
                targets[pos] in failed for pos in region.coordinates
            )
            ready = fallback or all(
                contains(targets[pos]) for pos in region.coordinates
            )
            can_publish = publish and (
                paintable is None or region.coordinates <= paintable
            )
            if ready and not can_publish:
                changed.extend(targets[pos] for pos in region.coordinates)
                pending.append(region)
                continue
            if fallback:
                # A failed precise job cannot retain deleted/old geometry
                # forever. Publish the current complete overview only in this
                # region; unrelated exact fronts remain untouched.
                for pos in region.coordinates:
                    self.front.pop(pos, None)
                    changed.append(targets[pos])
                self.completed_ids.update(region.measurement_ids)
            elif all(contains(targets[pos]) for pos in region.coordinates):
                for pos in region.coordinates:
                    self.front[pos] = targets[pos]
                    changed.append(targets[pos])
                self.completed_ids.update(region.measurement_ids)
            else:
                pending.append(region)
        self.regions = pending
        blocked = {pos for region in pending for pos in region.coordinates}
        for pos, key in targets.items():
            if (
                publish
                and (paintable is None or pos in paintable)
                and pos not in blocked
                and contains(key)
            ):
                self.front[pos] = key
        in_use = set(self.front.values())
        released = self._retired - in_use
        self._released.update(released)
        self._retired.difference_update(released)
        return changed

    def take_released(self):
        released, self._released = self._released, set()
        return released

    def key_for(self, target):
        return self.front.get(coordinate(target))

    def waits_for(self, target):
        return any(coordinate(target) in region.coordinates for region in self.regions)
