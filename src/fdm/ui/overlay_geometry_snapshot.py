"""Byte-bounded immutable RAW coordinates for detached overlay workers."""

from collections import OrderedDict
import numpy as np


class OverlayGeometrySnapshots:
    def __init__(self, max_bytes=32 * 1024 * 1024):
        self.max_bytes = max_bytes
        self.bytes = 0
        self._entries = OrderedDict()

    def capture(self, measurement, *, document_token=0):
        key = (id(measurement), measurement.id, measurement.geometry_revision)
        entry = self._entries.get(key)
        if entry is not None and entry[0] is measurement:
            self._entries.move_to_end(key)
            return entry[1]
        rings = measurement.area_rings_px or [measurement.polygon_px or []]
        coordinates = tuple(
            np.fromiter(
                (coordinate for point in ring for coordinate in (point.x, point.y)),
                dtype=np.float64,
                count=len(ring) * 2,
            ).tobytes()
            for ring in rings
            if len(ring) >= 3
        )
        size = sum(map(len, coordinates))
        # Holding the owner prevents object-id reuse. Stale geometry versions
        # are discarded independently of other objects' prepared snapshots.
        self.discard(measurement)
        while self._entries and self.bytes + size > self.max_bytes:
            _, old = self._entries.popitem(last=False)
            self.bytes -= old[2]
        if size <= self.max_bytes:
            self._entries[key] = (measurement, coordinates, size, document_token)
            self.bytes += size
        return coordinates

    def discard(self, measurement):
        for key in [key for key in self._entries if key[0] == id(measurement)]:
            self.bytes -= self._entries.pop(key)[2]

    def discard_document(self, measurements, *, document_token=None):
        owners = {id(measurement) for measurement in measurements}
        for key in [
            key
            for key in self._entries
            if key[0] in owners
            or (document_token is not None and self._entries[key][3] == document_token)
        ]:
            self.bytes -= self._entries.pop(key)[2]


overlay_geometry_snapshots = OverlayGeometrySnapshots()
