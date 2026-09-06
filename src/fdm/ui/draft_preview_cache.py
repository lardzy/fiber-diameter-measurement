"""Bounded display-only geometry and raster cache for segmentation drafts.

Published draft rings are replaced, never edited in place. Retaining their
containers gives each publication an identity without hashing every vertex on
every mouse move. Neither cached path nor raster is used by measurement tools.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field, replace
import math
import weakref
import numpy as np

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPainterPath, QPen, QPolygonF, QTransform


@dataclass
class DraftGeometrySnapshot:
    owners: tuple
    path: QPainterPath
    outlines: tuple[QPolygonF, ...]
    bytes: int
    rasters: OrderedDict = field(default_factory=OrderedDict)
    serial: int = 0
    raw_coordinates: tuple[bytes, ...] = ()
    bounds: QRectF | None = None


class DraftPreviewCache:
    def __init__(self, max_bytes: int = 64 * 1024 * 1024, *, asynchronous=False):
        self.max_bytes = max_bytes // 4 if asynchronous else max_bytes
        self._raster_cache = None
        self._requests = {}
        self._layer_last = {}
        self._publishers = {}
        self._prepared_serials = OrderedDict()
        self._serial = 0
        if asynchronous:
            from fdm.ui.canvas_overlay_cache import CanvasOverlayTileCache

            self._raster_cache = CanvasOverlayTileCache(
                max_bytes=max_bytes - self.max_bytes, isolated_worker=True
            )
            self._raster_cache.tileReady.connect(self._ready)
            self._raster_cache.tileFailed.connect(self._failed)
        self._entries: OrderedDict = OrderedDict()
        self.bytes = 0
        self.path_builds = 0
        self.raster_builds = 0

    def discard(self, owner: int) -> None:
        if self._raster_cache is not None:
            self._raster_cache.invalidate_document(owner)
        self._prepared_serials = OrderedDict(
            (key, value) for key, value in self._prepared_serials.items() if key[0] != owner
        )
        self._publishers.pop(owner, None)
        self._requests = {
            key: value for key, value in self._requests.items() if key.document_token != owner
        }
        self._layer_last = {
            key: value for key, value in self._layer_last.items() if key[0] != owner
        }
        for key in tuple(self._entries):
            if key[0] == owner:
                self.bytes -= self._entries.pop(key).bytes

    def _trim(self) -> None:
        while self._entries and self.bytes > self.max_bytes:
            _, entry = self._entries.popitem(last=False)
            self.bytes -= entry.bytes

    def geometry(self, owner: int, polygon, rings) -> DraftGeometrySnapshot:
        key = (owner, id(polygon), id(rings))
        entry = self._entries.get(key)
        if entry is not None:
            self._entries.move_to_end(key)
            return entry
        source_rings = tuple(ring for ring in (rings or [polygon]) if len(ring) >= 3)
        if self._raster_cache is not None:
            # Publish plain immutable coordinates; constructing and serializing
            # a native path can itself retain the GUI interpreter lock.
            arrays = tuple(
                np.fromiter(
                    (value for point in ring for value in (point.x, point.y)),
                    dtype=np.float64,
                    count=2 * len(ring),
                ).reshape(-1, 2)
                for ring in source_rings
            )
            raw_coordinates = tuple(array.tobytes() for array in arrays)
            if arrays:
                minimum = np.min([array.min(axis=0) for array in arrays], axis=0)
                maximum = np.max([array.max(axis=0) for array in arrays], axis=0)
                bounds = QRectF(
                    float(minimum[0]),
                    float(minimum[1]),
                    float(maximum[0] - minimum[0]),
                    float(maximum[1] - minimum[1]),
                )
            else:
                bounds = QRectF()
            # Include retained Python source points as well as detached bytes.
            size = 256 + sum(map(len, raw_coordinates)) * 8
            entry = DraftGeometrySnapshot(
                (polygon, rings),
                QPainterPath(),
                (),
                size,
                raw_coordinates=raw_coordinates,
                bounds=bounds,
            )
        else:
            outlines = tuple(QPolygonF([QPointF(p.x, p.y) for p in ring]) for ring in source_rings)
            path = QPainterPath()
            path.setFillRule(Qt.FillRule.OddEvenFill)
            for outline in outlines:
                path.addPolygon(outline)
                path.closeSubpath()
            size = 256 + sum(len(ring) for ring in outlines) * 64
            entry = DraftGeometrySnapshot((polygon, rings), path, outlines, size)
        self.path_builds += 1
        entry.serial = self.path_builds
        if size <= self.max_bytes:
            self._entries[key] = entry
            self.bytes += size
            self._trim()
        return entry

    @staticmethod
    def _paint(painter, entry, transform, fill, stroke, show_fill):
        if entry.raw_coordinates and entry.path.isEmpty():
            entry.outlines = tuple(
                QPolygonF(
                    [
                        QPointF(float(x), float(y))
                        for x, y in np.frombuffer(ring, dtype=np.float64).reshape(-1, 2)
                    ]
                )
                for ring in entry.raw_coordinates
            )
            entry.path.setFillRule(Qt.FillRule.OddEvenFill)
            for outline in entry.outlines:
                entry.path.addPolygon(outline)
                entry.path.closeSubpath()
        painter.save()
        painter.setWorldTransform(transform, combine=True)
        if show_fill:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(fill)
            painter.drawPath(entry.path)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for color, width, style in (
            (QColor("#0B0B0B"), 3.2, Qt.PenStyle.SolidLine),
            (stroke, 1.8, Qt.PenStyle.DashLine),
        ):
            pen = QPen(color, width, style, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
            pen.setCosmetic(True)
            painter.setPen(pen)
            for outline in entry.outlines:
                painter.drawPolygon(outline)
        painter.restore()

    def draw(
        self,
        painter,
        *,
        owner,
        polygon,
        rings,
        origin,
        zoom,
        dpr,
        viewport,
        fill,
        stroke,
        show_fill,
        interactive,
        publisher=None,
        layer_id=None,
    ):
        entry = self.geometry(owner, polygon, rings)
        if entry.path.isEmpty() and not entry.raw_coordinates:
            return
        transform = QTransform.fromTranslate(origin.x(), origin.y())
        transform.scale(zoom, zoom)
        asynchronous = (
            self._raster_cache is not None and publisher is not None and publisher.isVisible()
        )
        if not interactive and not asynchronous:
            self._paint(painter, entry, transform, fill, stroke, show_fill)
            return
        # An image-space guard around the viewport bounds memory for large
        # zoomed drafts, while ordinary drafts fit in one reusable raster.
        inverse, _ = transform.inverted()
        visible = inverse.mapRect(viewport.adjusted(-256, -256, 256, 256))
        quantum = 512 / zoom
        clipped = QRectF(
            math.floor(visible.left() / quantum) * quantum,
            math.floor(visible.top() / quantum) * quantum,
            (math.ceil(visible.right() / quantum) - math.floor(visible.left() / quantum)) * quantum,
            (math.ceil(visible.bottom() / quantum) - math.floor(visible.top() / quantum)) * quantum,
        )
        bounds = (
            (entry.bounds if entry.bounds is not None else entry.path.boundingRect())
            .adjusted(-4 / zoom, -4 / zoom, 4 / zoom, 4 / zoom)
            .intersected(clipped)
        )
        if bounds.isEmpty():
            return
        device_bounds = transform.mapRect(bounds)
        left = math.floor(device_bounds.left() * dpr) / dpr
        top = math.floor(device_bounds.top() * dpr) / dpr
        width = math.ceil(device_bounds.right() * dpr) - round(left * dpr)
        height = math.ceil(device_bounds.bottom() * dpr) - round(top * dpr)
        # Position relative to the image, including the device-pixel phase.
        relative = QPointF(left - origin.x(), top - origin.y())
        raster_key = (
            zoom,
            dpr,
            round(relative.x(), 7),
            round(relative.y(), 7),
            width,
            height,
            fill.rgba(),
            stroke.rgba(),
            show_fill,
        )
        if asynchronous:
            self._draw_asynchronous(
                painter,
                owner=owner,
                entry=entry,
                raster_key=raster_key,
                relative=relative,
                origin=origin,
                width=width,
                height=height,
                zoom=zoom,
                dpr=dpr,
                fill=fill,
                stroke=stroke,
                show_fill=show_fill,
                publisher=publisher,
                layer_id=layer_id if layer_id is not None else entry.serial,
            )
            return
        raster = entry.rasters.get(raster_key)
        if raster is None:
            size = width * height * 4
            if size > self.max_bytes // 2:
                self._paint(painter, entry, transform, fill, stroke, show_fill)
                return
            raster = QImage(width, height, QImage.Format.Format_ARGB32_Premultiplied)
            raster.setDevicePixelRatio(dpr)
            raster.fill(0)
            offscreen = QPainter(raster)
            offscreen.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            local = QTransform.fromTranslate(-relative.x(), -relative.y())
            local.scale(zoom, zoom)
            self._paint(offscreen, entry, local, fill, stroke, show_fill)
            offscreen.end()
            self.raster_builds += 1
            # Keep only the latest exact zoom/clip per draft. The complete
            # workspace shares this byte budget, rather than one per image.
            previous = sum(image.sizeInBytes() for image in entry.rasters.values())
            entry.rasters.clear()
            entry.rasters[raster_key] = raster
            entry.bytes += size - previous
            if any(value is entry for value in self._entries.values()):
                self.bytes += size - previous
                self._trim()
        painter.drawImage(QPointF(left, top), raster)

    def _remember_placement(self, layer, placement):
        self._layer_last.pop(layer, None)
        self._layer_last[layer] = placement
        while len(self._layer_last) > 512:
            self._layer_last.pop(next(iter(self._layer_last)))

    def _failed(self, key, _message):
        self._requests.pop(key, None)

    def _ready(self, key):
        request = self._requests.pop(key, None)
        if request is None:
            return
        layer, relative, zoom, width, height, dpr = request
        self._remember_placement(layer, (key, relative, zoom, width, height, dpr))
        self.raster_builds += 1
        reference = self._publishers.get(key.document_token)
        publisher = reference() if reference is not None else None
        if publisher is not None:
            try:
                publisher.update()
            except RuntimeError:
                self._publishers.pop(key.document_token, None)

    def _draw_asynchronous(
        self,
        painter,
        *,
        owner,
        entry,
        raster_key,
        relative,
        origin,
        width,
        height,
        zoom,
        dpr,
        fill,
        stroke,
        show_fill,
        publisher,
        layer_id,
        command_override=None,
    ):
        from fdm.ui.canvas_overlay_cache import (
            AreaOverlayDrawCommand,
            CanvasOverlayRenderSnapshot,
            CanvasOverlayTileKey,
        )

        key = CanvasOverlayTileKey(
            owner,
            "segmentation-draft" if command_override is None else "active-area",
            zoom,
            dpr,
            0,
            0,
            0,
            0,
            show_fill,
            content_stamp=(entry.serial, raster_key),
        )
        layer = (owner, layer_id)
        self._publishers[owner] = weakref.ref(publisher)
        image = self._raster_cache.get(key)
        placement = (key, relative, zoom, width, height, dpr)
        if image is None:
            if not self._raster_cache.is_pending(key):
                for old, request in list(self._requests.items()):
                    if request[0] == layer:
                        self._raster_cache.cancel(old)
                        self._requests.pop(old, None)
                transform = QTransform.fromTranslate(-relative.x(), -relative.y())
                transform.scale(zoom, zoom)
                if command_override is None:
                    command = AreaOverlayDrawCommand(
                        path=None if entry.raw_coordinates else entry.path,
                        raw_coordinates=entry.raw_coordinates,
                        geometry_key=(owner, "draft", entry.serial),
                        image_to_overlay=transform,
                        fill_rgba=fill.rgba() if show_fill else None,
                        outline_rgba=QColor("#0B0B0B").rgba(),
                        outline_width=3.2,
                        stroke_rgba=stroke.rgba(),
                        stroke_width=1.8,
                        stroke_style=Qt.PenStyle.DashLine.value,
                        separate_fill=True,
                    )
                else:
                    label = command_override.label
                    if label is not None and label.top_left is not None:
                        label = replace(label, top_left=label.top_left - relative)
                    command = replace(command_override, image_to_overlay=transform, label=label)
                self._requests[key] = (layer, relative, zoom, width, height, dpr)
                if not self._raster_cache.request(
                    CanvasOverlayRenderSnapshot(
                        request_id=entry.serial,
                        key=key,
                        area_commands=(command,),
                        logical_tile_size=max(1, math.ceil(max(width, height) / dpr)),
                    )
                ):
                    self._requests.pop(key, None)
            placement = self._layer_last.get(layer)
            if placement is None:
                return False
            image = self._raster_cache.get(placement[0])
            if image is None:
                return False
        else:
            self._remember_placement(layer, placement)
        self._draw_placement(painter, image, placement, origin, zoom)
        return True

    @staticmethod
    def _draw_placement(painter, image, placement, origin, zoom):
        _, previous_relative, previous_zoom, previous_width, previous_height, previous_dpr = (
            placement
        )
        scale = zoom / previous_zoom
        target = QRectF(
            origin.x() + previous_relative.x() * scale,
            origin.y() + previous_relative.y() * scale,
            previous_width / previous_dpr * scale,
            previous_height / previous_dpr * scale,
        )
        painter.save()
        painter.setClipRect(target, Qt.ClipOperation.IntersectClip)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, scale != 1)
        painter.drawImage(QRectF(target.topLeft(), image.deviceIndependentSize() * scale), image)
        painter.restore()

    def preserve_draft(self, owner, layer_ids):
        # Transfer only cache keys and transforms. Images stay charged to the
        # global LRU and can be evicted even while an accepted job is pending.
        return tuple(
            self._layer_last[(owner, layer)]
            for layer in layer_ids
            if (owner, layer) in self._layer_last
        )

    def clear_draft_geometry(self, owner):
        for key in tuple(self._entries):
            if key[0] == owner:
                self.bytes -= self._entries.pop(key).bytes
        for key in tuple(self._requests):
            if key.document_token == owner:
                self._raster_cache.cancel(key)
                self._requests.pop(key)
        self._layer_last = {
            key: value for key, value in self._layer_last.items() if key[0] != owner
        }

    def draw_preserved(self, painter, placements, *, origin, zoom):
        for placement in placements:
            image = self._raster_cache.get(placement[0])
            if image is not None:
                self._draw_placement(painter, image, placement, origin, zoom)

    def draw_prepared(
        self,
        painter,
        *,
        owner,
        version,
        command,
        bounds,
        origin,
        zoom,
        dpr,
        viewport,
        publisher,
        layer_id,
    ):
        identity = (owner, version)
        serial = self._prepared_serials.get(identity)
        if serial is None:
            self._serial += 1
            serial = -self._serial
            self._prepared_serials[identity] = serial
            while len(self._prepared_serials) > 256:
                self._prepared_serials.popitem(last=False)
        else:
            self._prepared_serials.move_to_end(identity)
        transform = QTransform.fromTranslate(origin.x(), origin.y())
        transform.scale(zoom, zoom)
        inverse, _ = transform.inverted()
        visible = inverse.mapRect(viewport.adjusted(-256, -256, 256, 256))
        quantum = 512 / zoom
        left = math.floor(visible.left() / quantum) * quantum
        top = math.floor(visible.top() / quantum) * quantum
        clipped = QRectF(
            left,
            top,
            math.ceil(visible.right() / quantum) * quantum - left,
            math.ceil(visible.bottom() / quantum) * quantum - top,
        )
        bounds = bounds.intersected(clipped)
        if bounds.isEmpty():
            return True
        device = transform.mapRect(bounds)
        left, top = math.floor(device.left() * dpr) / dpr, math.floor(device.top() * dpr) / dpr
        width, height = (
            math.ceil(device.right() * dpr) - round(left * dpr),
            math.ceil(device.bottom() * dpr) - round(top * dpr),
        )
        relative = QPointF(left - origin.x(), top - origin.y())
        raster_key = (zoom, dpr, round(relative.x(), 7), round(relative.y(), 7), width, height)
        entry = DraftGeometrySnapshot((), QPainterPath(), (), 0, serial=serial)
        return self._draw_asynchronous(
            painter,
            owner=owner,
            entry=entry,
            raster_key=raster_key,
            relative=relative,
            origin=origin,
            width=width,
            height=height,
            zoom=zoom,
            dpr=dpr,
            fill=None,
            stroke=None,
            show_fill=command.fill_rgba is not None,
            publisher=publisher,
            layer_id=layer_id,
            command_override=command,
        )


draft_preview_cache = DraftPreviewCache(asynchronous=True)
