"""Immutable local masks with an explicit origin in the source raster."""

from dataclasses import dataclass
import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class MaskRegion:
    data: np.ndarray
    origin: tuple[int, int]
    extent: tuple[int, int]  # source height, width

    def copy(self):
        # The data is privately owned and read-only. Consumers needing a
        # writable buffer must explicitly copy .data.
        return self

    def any(self):
        return bool(self.data.any())

    def to_full_mask(self):
        result = np.zeros(self.extent, dtype=bool)
        x, y = self.origin
        result[y : y + self.data.shape[0], x : x + self.data.shape[1]] = self.data
        return result


def mask_region(mask, *, origin=(0, 0), extent=None):
    if isinstance(mask, MaskRegion):
        return mask
    if mask is None:
        return None
    array = np.asarray(mask, dtype=bool)
    rows = np.flatnonzero(array.any(axis=1))
    if not len(rows):
        return None
    columns = np.flatnonzero(array.any(axis=0))
    y0, y1 = int(rows[0]), int(rows[-1]) + 1
    x0, x1 = int(columns[0]), int(columns[-1]) + 1
    data = np.array(array[y0:y1, x0:x1], dtype=bool, copy=True, order="C")
    data.setflags(write=False)
    return MaskRegion(data, (origin[0] + x0, origin[1] + y0), extent or array.shape)


def overlap_slices(first, second):
    ax, ay = first.origin
    bx, by = second.origin
    x0, y0 = max(ax, bx), max(ay, by)
    x1 = min(ax + first.data.shape[1], bx + second.data.shape[1])
    y1 = min(ay + first.data.shape[0], by + second.data.shape[0])
    if x1 <= x0 or y1 <= y0:
        return None
    return (
        (slice(y0 - ay, y1 - ay), slice(x0 - ax, x1 - ax)),
        (slice(y0 - by, y1 - by), slice(x0 - bx, x1 - bx)),
    )


def subtract_regions(primary, masks):
    primary = mask_region(primary)
    result = primary.data.copy()
    intersected = False
    count = 0
    for mask in masks:
        region = mask_region(mask)
        if region is None:
            continue
        if region.extent != primary.extent:
            raise ValueError("Mask source extents do not match")
        count += 1
        slices = overlap_slices(primary, region)
        if slices is None:
            continue
        target, source = slices
        intersected |= bool(np.any(primary.data[target] & region.data[source]))
        result[target] &= ~region.data[source]
    return result, intersected, count


def rasterize_rings_region(rings, *, extent, source_origin=(0, 0)):
    """Keep the legacy round-then-clamp pixel semantics before ROI cropping."""
    height, width = extent
    contours = [
        np.array(
            [
                [
                    min(max(round(p.x - source_origin[0]), 0), width - 1),
                    min(max(round(p.y - source_origin[1]), 0), height - 1),
                ]
                for p in ring
            ],
            dtype=np.int32,
        )
        for ring in rings
    ]
    if not contours or len(contours[0]) < 3:
        return None
    x, y, w, h = cv2.boundingRect(contours[0])
    mask = np.zeros((h, w), dtype=np.uint8)
    for index, contour in enumerate(contours):
        if len(contour) >= 3:
            cv2.fillPoly(mask, [contour - np.array([x, y], dtype=np.int32)], 1 if index == 0 else 0)
    return mask_region(mask, origin=(x, y), extent=extent)
