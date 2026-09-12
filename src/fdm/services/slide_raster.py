"""One deterministic source for repaired display, algorithms and exports."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import cv2
import numpy as np
from PySide6.QtGui import QImage

from fdm.services.slide_layout import LayoutTile, SlideLayoutSnapshot


@dataclass(frozen=True, slots=True)
class SlideRasterRegion:
    image: QImage
    coverage: np.ndarray
    unverified_seams: np.ndarray
    owners: np.ndarray


class SlideRasterSource:
    def __init__(self, layout: SlideLayoutSnapshot, image_loader: Callable[[LayoutTile], QImage]):
        self.layout = layout
        self.image_loader = image_loader
        self._planes = {}
        for tile in sorted(layout.tiles, key=lambda t: (t.fov_id, t.tile_id)):
            self._planes.setdefault(tile.z_index, []).append(tile)
        self._by_id = {t.tile_id: t for t in layout.tiles}
        self._verified = {(frozenset((p.first, p.second)), z)
                          for p in layout.pairs if p.accepted for z in p.verified_focus}
        self._unverified_region_cache = {}

    def tiles_in_rect(self, focus, rect):
        x, y, w, h = rect
        return [t for t in self._planes.get(focus, ())
                if t.x < x + w and t.y < y + h and t.x + t.width > x and t.y + t.height > y]

    def owner_map(self, focus, rect, size):
        x, y, width, height = rect
        ow, oh = size
        xx = x + (np.arange(ow, dtype=np.float64) + .5) * width / ow
        yy = y + (np.arange(oh, dtype=np.float64) + .5) * height / oh
        owner = np.zeros((oh, ow), dtype=np.int32)
        # Distance to the original field edge chooses a unique source. Strict
        # comparisons + stable FOV order make ties independent of SQL/LOD order.
        best = np.full((oh, ow), -np.inf, dtype=np.float64)
        for tile in self.tiles_in_rect(focus, rect):
            dx = np.minimum(xx - tile.x, tile.x + tile.width - xx)
            dy = np.minimum(yy - tile.y, tile.y + tile.height - yy)
            score = np.minimum(dy[:, None], dx[None, :])
            use = (score >= 0) & (score > best)
            owner[use] = tile.tile_id
            best[use] = score[use]
        return owner

    def masks(self, focus, rect, size):
        # Halo makes a seam at a requested ROI boundary visible on both sides.
        x, y, width, height = rect
        ow, oh = size
        sx, sy = width / ow, height / oh
        owners = self.owner_map(focus, (x - 2 * sx, y - 2 * sy, width + 4 * sx, height + 4 * sy), (ow + 4, oh + 4))
        unsafe = np.zeros_like(owners, dtype=np.uint8)
        for a, b, ta, tb in ((owners[:, :-1], owners[:, 1:], unsafe[:, :-1], unsafe[:, 1:]),
                             (owners[:-1], owners[1:], unsafe[:-1], unsafe[1:])):
            changed = (a != b) & (a != 0) & (b != 0)
            if not changed.any():
                continue
            for ia, ib in np.unique(np.stack((a[changed], b[changed]), axis=1), axis=0):
                first, second = self._by_id[int(ia)], self._by_id[int(ib)]
                if (frozenset((first.fov_id, second.fov_id)), focus) not in self._verified:
                    mask = changed & (a == ia) & (b == ib)
                    ta[mask] = 1
                    tb[mask] = 1
        unsafe[owners == 0] = 1
        unsafe = cv2.dilate(unsafe, np.ones((3, 3), np.uint8)).astype(bool)
        return owners[2:-2, 2:-2].copy(), unsafe[2:-2, 2:-2].copy()

    def read_region(self, focus, rect, *, output_size=None, cancelled=lambda: False, pixels=True):
        x, y, width, height = rect
        ow, oh = output_size or (max(1, round(width)), max(1, round(height)))
        if min(ow, oh) <= 0 or ow * oh > 64 * 1024 * 1024:
            raise ValueError("拼接读取范围过大，请按视口或图块读取")
        result = np.empty((oh, ow, 3), dtype=np.uint8) if pixels else None
        if result is not None:
            result[:] = (16, 24, 32)
        all_owners = np.zeros((oh, ow), np.int32)
        all_unsafe = np.zeros((oh, ow), bool)
        tiles = {t.tile_id: t for t in self.tiles_in_rect(focus, rect)}
        for top in range(0, oh, 256):
            if cancelled():
                raise InterruptedError("拼接视图读取已取消")
            rows = min(256, oh - top)
            subrect = (x, y + top * height / oh, width, rows * height / oh)
            owners, unsafe = self.masks(focus, subrect, (ow, rows))
            all_owners[top:top + rows], all_unsafe[top:top + rows] = owners, unsafe
            if result is None:
                continue
            for tile_id in np.unique(owners):
                if not tile_id:
                    continue
                tile = tiles[int(tile_id)]
                image = self.image_loader(tile)
                if image.isNull():
                    raise ValueError(f"无法读取拼接视图的原始视场 {tile.tile_id}，不能将缺失像素用于测量")
                rgb = image.convertToFormat(QImage.Format.Format_RGB888)
                source = np.frombuffer(rgb.constBits(), np.uint8).reshape(rgb.height(), rgb.bytesPerLine())[:, :rgb.width() * 3].reshape(rgb.height(), rgb.width(), 3)
                xx = ((x + (np.arange(ow, dtype=np.float64) + .5) * width / ow - tile.x) * rgb.width() / tile.width - .5)
                yy = ((y + (np.arange(top, top + rows, dtype=np.float64) + .5) * height / oh - tile.y) * rgb.height() / tile.height - .5)
                mx, my = np.meshgrid(xx.astype(np.float32), yy.astype(np.float32))
                sampled = cv2.remap(source, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                selected = owners == tile_id
                result[top:top + rows][selected] = sampled[selected]
        image = QImage() if result is None else QImage(result.data, ow, oh, ow * 3, QImage.Format.Format_RGB888).copy()
        return SlideRasterRegion(image, all_owners != 0, all_unsafe, all_owners)

    def unverified_regions(self, focus):
        """Conservative overlap/gap bands for geometry quality annotations."""
        if focus not in self._unverified_region_cache:
            self._unverified_region_cache[focus] = tuple(self._unverified_bands(focus))
        return self._unverified_region_cache[focus]

    def _unverified_bands(self, focus):
        fovs = {t.fov_id: t for t in self._planes.get(focus, ())}
        for pair in self.layout.pairs:
            if pair.accepted and focus in pair.verified_focus:
                continue
            a, b = fovs.get(pair.first), fovs.get(pair.second)
            if a is None or b is None:
                continue
            if pair.axis == "x":
                left, right = sorted((a.x + a.width, b.x))
                top, bottom = max(a.y, b.y), min(a.y + a.height, b.y + b.height)
            else:
                top, bottom = sorted((a.y + a.height, b.y))
                left, right = max(a.x, b.x), min(a.x + a.width, b.x + b.width)
            if right >= left and bottom >= top:
                yield (left - 2, top - 2, max(4, right - left + 4), max(4, bottom - top + 4))
