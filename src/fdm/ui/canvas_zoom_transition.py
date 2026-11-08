"""Publish a dense viewport only after its visible, target-resolution overlay is ready."""

from dataclasses import dataclass
import math
import time
from weakref import proxy

from PySide6.QtCore import QObject, QPointF, QRectF, QTimer, Slot
from PySide6.QtGui import QFontMetricsF

from fdm.geometry import Point
from fdm.ui.canvas_overlay_cache import CanvasOverlayRenderSnapshot, CanvasOverlayTileKey
from fdm.ui.rendering import _count_number_font, measurement_text_scale
from fdm.ui.screen_label_sprite_cache import MAX_SINGLE_LABEL_BYTES
from fdm.ui.view_transform import CanvasZoomMode


@dataclass(frozen=True, slots=True)
class ZoomView:
    zoom: float
    pan: Point
    mode: CanvasZoomMode


class CanvasZoomTransition(QObject):
    """One detached raster in flight and the latest target requested by the user."""

    # Small views can draw their sharp visible primitives in a single paint.
    MIN_OBJECTS = 192
    MIN_VERTICES = 16_000

    def __init__(self, canvas, cache):
        super().__init__(canvas)
        # Qt owns this child. A Python reference back to the widget would
        # defer its destruction until cyclic GC (possibly on a worker).
        self.canvas = proxy(canvas)
        self.cache = cache
        self.requested: ZoomView | None = None
        self._requested_key: CanvasOverlayTileKey | None = None
        self.preparing: CanvasOverlayTileKey | None = None
        self.ready = False
        self.presented: CanvasOverlayTileKey | None = None
        self._submitted = False
        self._release_commands()
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._prepare)
        cache.tileReady.connect(self._completed)
        cache.tileFailed.connect(self._failed)
        # Qt disconnects this receiver and destroys the timer with the widget.
        # Do not call back into a partially destroyed Python widget wrapper.

    def _key(self, view):
        c = self.canvas
        doc = c._document
        stamp = doc.state_stamp
        return CanvasOverlayTileKey(
            id(doc), doc.id, view.zoom, max(1.0, c.devicePixelRatioF()), 0, 0,
            c._overlay_style_generation, doc.measurement_geometry_revision, c._show_area_fill,
            content_stamp=("zoom-viewport", id(c), c._overlay_binding_serial,
                           stamp.session_state_id, stamp.calibration_state_id,
                           c.width(), c.height(), view.pan.x, view.pan.y, c._screen_label_mode),
        )

    def _release_commands(self):
        self._candidates = ()
        self._raw_area_ids = frozenset()
        self._numbers = None
        self._counts = []
        self._commands = []
        self._position = 0
        self._command_bytes = 0
        self._count_font_sizes = {}

    def _count_cost(self, item, target, dpr):
        settings = self.canvas._screen_passive_settings
        if not settings.show_count_numbers:
            return 256
        font = _count_number_font(settings, item)
        size = self._count_font_sizes.get(font.key())
        if size is None:
            metrics = QFontMetricsF(font)
            size = (max(metrics.horizontalAdvance(str(i)) for i in range(10)), metrics.height())
            self._count_font_sizes[font.key()] = size
        ratio = max(0.25, dpr * measurement_text_scale(settings, target.zoom))
        digits = len(str(self._numbers.get(item.id, len(self._candidates)) if self._numbers else len(self._candidates)))
        width = math.ceil((size[0] * digits + 12) * ratio) + 2
        height = math.ceil((size[1] + 6) * ratio) + 2
        return 512 + min(MAX_SINGLE_LABEL_BYTES, 4 * width * height)

    @Slot()
    def cancel(self):
        self._timer.stop()
        if self.preparing is not None and self._submitted:
            self.cache.cancel(self.preparing)
        self.requested = self.preparing = self._requested_key = None
        self._submitted = self.ready = False
        self._release_commands()

    def reset(self):
        self.cancel()
        self.presented = None

    def schedule(self, target):
        c = self.canvas
        doc = c._document
        if (not c.isVisible() or not doc or c._viewport_focus_index() is not None
                or c._acquisition_alignment is not None or c._has_pointer_edit_operation()
                or not c._overlay_cache_enabled()
                or (len(doc.measurements) < self.MIN_OBJECTS
                    and c._overlay_area_vertex_count < self.MIN_VERTICES)):
            self.cancel()
            return False
        dpr = max(1.0, c.devicePixelRatioF())
        if math.ceil(c.width() * dpr) * math.ceil(c.height() * dpr) * 4 > self.cache.max_bytes:
            # An oversized viewport uses the same sharp, clipped per-region path.
            self.cancel()
            return False
        if self.requested is None:
            c._pause_overlay_jobs_for_zoom()
        self.requested = target
        self._requested_key = self._key(target)
        self.ready = False
        if not self._submitted:
            self._release_commands()
            self.preparing = self._requested_key
            self._timer.start(0)
        return True

    def _prepare(self):
        c, target, key = self.canvas, self.requested, self.preparing
        if target is None or c._document is None or self._key(target) != key:
            self.cancel()
            return
        if self.cache.contains(key):
            self._completed(key)
            return
        if not self._candidates:
            rect = QRectF(-target.pan.x / target.zoom, -target.pan.y / target.zoom,
                          c.width() / target.zoom, c.height() / target.zoom)
            candidates, self._numbers = c._measurement_render_inputs(rect, zoom=target.zoom)
            # A globally dense document may have very little in the new view.
            # Draw this target immediately from the same indexed primitives.
            vertices = sum(
                sum(map(len, item.area_rings_px)) if item.area_rings_px else len(item.polygon_px)
                for item in candidates if item.measurement_kind == "area"
            )
            if not candidates or (len(candidates) < self.MIN_OBJECTS and vertices < self.MIN_VERTICES):
                self.ready = True
                c.update()
                return
            self._candidates = candidates
            self._raw_area_ids = c._area_proxy_exclusions(candidates, target.zoom * key.device_pixel_ratio)
            selected_id = c._document.view_state.selected_measurement_id
            if selected_id is not None:
                self._raw_area_ids.add(selected_id)
        deadline = time.perf_counter() + 0.003
        clip_rect = QRectF(-target.pan.x, -target.pan.y, c.width(), c.height())
        while self._position < len(self._candidates):
            batch = self._candidates[self._position:self._position + 8]
            primitives = []
            for item in batch:
                if item.measurement_kind == "count":
                    self._counts.append(item)
                    # Counts need a single ordered marker/label pass. Reserve
                    # their sprite storage before recording that final picture.
                    self._command_bytes += self._count_cost(item, target, key.device_pixel_ratio)
                else:
                    primitives.append(item)
            commands = c._capture_overlay_commands(
                primitives, zoom=target.zoom, dpr=key.device_pixel_ratio, viewport_raw_ids=self._raw_area_ids,
                count_numbers=self._numbers, clip_rect=clip_rect,
            )
            self._command_bytes += self.cache._estimate_pending_snapshot_bytes(
                CanvasOverlayRenderSnapshot(0, key, area_commands=commands)
            )
            if self._command_bytes > self.cache.max_bytes:
                # Use the current-zoom indexed visible fallback. Discard the
                # staging pictures now, before recording the rest of the view.
                self._release_commands()
                self.ready = True
                c.update()
                return
            self._commands.extend(commands)
            self._position += len(batch)
            if self._position < len(self._candidates) and time.perf_counter() >= deadline:
                self._timer.start(0)
                return
        if self._counts:
            self._commands.extend(c._capture_overlay_commands(
                self._counts, zoom=target.zoom, dpr=key.device_pixel_ratio, count_numbers=self._numbers,
                clip_rect=clip_rect,
            ))
        commands = tuple(self._commands)
        self._release_commands()
        snapshot = CanvasOverlayRenderSnapshot(
            0, key, area_commands=commands, known_empty=not commands, bleed_device_pixels=0,
            viewport=(c.width(), c.height(), target.pan.x, target.pan.y),
        )
        self._submitted = True
        if not self.cache.request(snapshot):
            if self.cache.contains(key):
                self._completed(key)
            elif not self.cache.is_pending(key):
                self._failed(key, "viewport exceeded cache budget")

    @Slot(object)
    def _completed(self, key):
        if key != self.preparing:
            return
        self._submitted = False
        if self.canvas._document is None or self.requested is None:
            self.cancel()
            return
        if self._requested_key != self._key(self.requested):
            self.cancel()
            return
        if self._requested_key != key:
            # One or more wheel ticks arrived while the old worker was active.
            self.preparing = self._requested_key
            self._timer.start(0)
            return
        self.ready = True
        self.canvas.update()

    @Slot(object, str)
    def _failed(self, key, _message):
        if key != self.preparing:
            return
        self._completed(key)
        # publish() will use the synchronous sharp fallback for a failed raster.

    def publish(self, full_paint):
        if not self.ready or self.requested is None:
            return
        if not full_paint:
            self.canvas.update()
            return
        target = self.requested
        if self.canvas._document is None or self.preparing != self._key(target):
            self.cancel()
            return
        self.presented = self.preparing if self.cache.contains(self.preparing) else None
        self.cancel()
        self.canvas._apply_zoom_view(target)

    def paint(self, painter):
        c = self.canvas
        if self.presented is None or c._document is None:
            return False
        if self.presented != self._key(ZoomView(c._zoom, c._pan, c._zoom_mode)):
            return False
        payload = self.cache.get_payload(self.presented)
        if payload is None:
            self.presented = None
            return False
        if payload[0] is not None:
            painter.drawImage(QPointF(0, 0), payload[0])
        return True
