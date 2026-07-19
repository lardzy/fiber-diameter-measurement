from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass
import math
import queue
import threading

import numpy as np

from PySide6.QtCore import (
    QObject,
    QPointF,
    QRunnable,
    QRectF,
    QThread,
    QThreadPool,
    QTimer,
    Qt,
    Signal,
    Slot,
)
from PySide6.QtGui import (
    QColor,
    QImage,
    QPainter,
    QPainterPath,
    QPen,
    QPicture,
    QTransform,
)

from fdm.geometry import odd_even_path_moments


OVERLAY_TILE_LOGICAL_SIZE = 512
OVERLAY_TILE_BLEED_DEVICE_PIXELS = 2
OVERLAY_TILE_MAX_ENTRIES = 256
OVERLAY_TILE_MAX_BYTES = 128 * 1024 * 1024
OVERLAY_TILE_MAX_PENDING_BYTES = 128 * 1024 * 1024

# Pending snapshots contain implicitly-shared Qt value types, so Python object
# sizes alone significantly under-report their backing storage.  These
# constants intentionally over-estimate the small fixed wrappers while paths,
# pictures, and label images use their payload-specific sizes below.
_PENDING_SNAPSHOT_OVERHEAD_BYTES = 256
_PENDING_AREA_COMMAND_OVERHEAD_BYTES = 320
_PENDING_PATH_ELEMENT_BYTES = 48
_PENDING_LABEL_COMMAND_OVERHEAD_BYTES = 128

AreaOverlayCentroidKey = tuple[int, int, str, int]


@dataclass(frozen=True, slots=True)
class CanvasOverlayTileKey:
    """Identity of one exact-scale passive overlay tile."""

    document_token: int
    document_id: str
    zoom: float
    device_pixel_ratio: float
    tile_x: int
    tile_y: int
    style_generation: int
    tile_epoch: int
    show_area_fill: bool
    device_phase_x: float = 0.0
    device_phase_y: float = 0.0


@dataclass(frozen=True, slots=True)
class AreaOverlayLabelCommand:
    """Detached screen-space label sprite used by an area draw command."""

    image: QImage
    top_left: QPointF | None
    center_offset: QPointF
    centroid_key: AreaOverlayCentroidKey | None = None


@dataclass(frozen=True, slots=True)
class AreaOverlayDrawCommand:
    """Immutable, QObject-free passive area render input.

    The path stays in RAW image coordinates and is copied through Qt's
    implicitly-shared value semantics.  Snapshot creation therefore never
    maps or serializes every vertex merely to hand the command to a worker.
    """

    path: QPainterPath
    image_to_overlay: QTransform
    fill_rgba: int | None
    outline_rgba: int
    outline_width: float
    stroke_rgba: int
    stroke_width: float
    label: AreaOverlayLabelCommand | None = None


@dataclass(frozen=True, slots=True)
class CanvasOverlayRenderSnapshot:
    """UI-thread-recorded commands safe to hand to a cache worker.

    ``exact_composition`` retains the command stream for stable display and a
    same-generation transparent raster for continuous navigation. Replaying
    the commands directly over the current image is required at rest for rare
    semi-transparent compositions whose 8-bit Porter-Duff rounding would
    otherwise leave visible seams.
    """

    request_id: int
    key: CanvasOverlayTileKey
    picture: QPicture | None = None
    area_commands: tuple[AreaOverlayDrawCommand, ...] = ()
    logical_tile_size: int = OVERLAY_TILE_LOGICAL_SIZE
    bleed_device_pixels: int = OVERLAY_TILE_BLEED_DEVICE_PIXELS
    exact_composition: bool = False
    adaptive_composition: bool = False
    composition_probe_rgba: int = 0xFFFFFFFF
    known_empty: bool = False


@dataclass(frozen=True, slots=True)
class CanvasOverlayCacheStats:
    entries: int
    bytes: int
    pending: int
    hits: int
    misses: int
    completed: int
    dropped: int
    pending_bytes: int = 0


@dataclass(slots=True)
class _CachedTile:
    image: QImage | None
    picture: QPicture | None
    estimated_bytes: int


@dataclass(slots=True)
class _PendingTile:
    request_sequence: int
    cancellation: _CancellationFlag
    estimated_bytes: int


@dataclass(frozen=True, slots=True)
class _TileRenderCompletion:
    key: CanvasOverlayTileKey
    request_sequence: int
    image: QImage | None = None
    picture: QPicture | None = None
    error: str | None = None
    cancelled: bool = False


class _CancellationFlag:
    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    def is_cancelled(self) -> bool:
        return self._event.is_set()


class _AreaCommandCentroidCache:
    """Bounded exact-centroid cache shared only by tile worker payloads."""

    def __init__(self, max_entries: int = 2048) -> None:
        self._max_entries = max(1, int(max_entries))
        self._entries: OrderedDict[AreaOverlayCentroidKey, QPointF] = OrderedDict()
        self._lock = threading.Lock()

    def get_or_compute(
        self,
        key: AreaOverlayCentroidKey,
        path: QPainterPath,
    ) -> QPointF:
        with self._lock:
            cached = self._entries.get(key)
            if cached is not None:
                self._entries.move_to_end(key)
                return QPointF(cached)
        area, moment_x, moment_y = odd_even_path_moments(path)
        computed = (
            QPointF(moment_x / area, moment_y / area)
            if area > 1e-9
            else path.boundingRect().center()
        )
        with self._lock:
            existing = self._entries.get(key)
            if existing is not None:
                self._entries.move_to_end(key)
                return QPointF(existing)
            self._entries[key] = QPointF(computed)
            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)
        return QPointF(computed)

    def discard_document(self, document_token: int) -> None:
        token = int(document_token)
        with self._lock:
            for key in [key for key in self._entries if key[0] == token]:
                self._entries.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


class _TileRenderRunnable(QRunnable):
    def __init__(
        self,
        snapshot: CanvasOverlayRenderSnapshot,
        cancellation: _CancellationFlag,
        request_sequence: int,
        completions: queue.SimpleQueue[_TileRenderCompletion],
        centroid_cache: _AreaCommandCentroidCache,
    ) -> None:
        super().__init__()
        self.setAutoDelete(True)
        self._snapshot = snapshot
        self._cancellation = cancellation
        self._request_sequence = int(request_sequence)
        self._completions = completions
        self._centroid_cache = centroid_cache

    @Slot()
    def run(self) -> None:
        if self._cancellation.is_cancelled():
            self._report(cancelled=True)
            return
        try:
            rendered_image, rendered_picture = self._render()
        except Exception as exc:  # pragma: no cover - defensive Qt backend path
            self._report(error=str(exc))
            return
        if self._cancellation.is_cancelled():
            self._report(cancelled=True)
            return
        self._report(image=rendered_image, picture=rendered_picture)

    def _report(
        self,
        *,
        image: QImage | None = None,
        picture: QPicture | None = None,
        error: str | None = None,
        cancelled: bool = False,
    ) -> None:
        # This Python queue is the only worker-to-UI handoff. The runnable
        # never dereferences the cache QObject, a Measurement, or a QPixmap.
        self._completions.put(
            _TileRenderCompletion(
                key=self._snapshot.key,
                request_sequence=self._request_sequence,
                image=image,
                picture=picture,
                error=error,
                cancelled=cancelled,
            )
        )

    def _render(self) -> tuple[QImage | None, QPicture | None]:
        snapshot = self._snapshot
        if snapshot.known_empty:
            # Keep empty viewport/guard tiles as a tiny no-op command stream;
            # allocating a full transparent 512×DPR raster would waste several
            # MiB without changing a single pixel.
            return None, QPicture()
        if snapshot.exact_composition:
            # Keep two representations for the rare exact-composition path:
            # the QPicture remains authoritative while the view is still, and
            # the transparent raster is used only during continuous panning.
            # Replaying hundreds of high-vertex magic-wand paths for every
            # mouse move defeats the cache; translating this same-generation
            # raster mirrors the interaction strategy used by mature image
            # viewers. The exact command stream is shown again on release.
            rendered = self._rasterize(snapshot, fill_rgba=0)
            if self._cancellation.is_cancelled():
                return rendered, None
            return rendered, self._exact_picture(snapshot)
        rendered = self._rasterize(snapshot, fill_rgba=0)
        if self._cancellation.is_cancelled():
            return rendered, None
        if snapshot.adaptive_composition:
            direct_on_opaque = self._rasterize(
                snapshot,
                fill_rgba=int(snapshot.composition_probe_rgba),
            )
            if self._cancellation.is_cancelled():
                return rendered, None
            flattened_on_opaque = QImage(
                rendered.width(),
                rendered.height(),
                QImage.Format.Format_ARGB32_Premultiplied,
            )
            flattened_on_opaque.fill(int(snapshot.composition_probe_rgba))
            flattened_on_opaque.setDevicePixelRatio(rendered.devicePixelRatio())
            painter = QPainter(flattened_on_opaque)
            try:
                painter.drawImage(0, 0, rendered)
            finally:
                painter.end()
            if self._has_visible_composition_difference(
                direct_on_opaque,
                flattened_on_opaque,
            ):
                return rendered, self._exact_picture(snapshot)
        return rendered, None

    def _exact_picture(
        self,
        snapshot: CanvasOverlayRenderSnapshot,
    ) -> QPicture:
        if snapshot.picture is not None:
            # QPicture is a detached byte-backed value at this point.  The
            # worker never paints on a QWidget or reads mutable document state.
            return CanvasOverlayTileCache._clone_picture(snapshot.picture)
        return self._record_area_commands(snapshot)

    def _rasterize(
        self,
        snapshot: CanvasOverlayRenderSnapshot,
        *,
        fill_rgba: int,
    ) -> QImage:
        key = snapshot.key
        dpr = max(1.0, float(key.device_pixel_ratio))
        core_physical = max(1, int(math.ceil(snapshot.logical_tile_size * dpr)))
        bleed_physical = max(0, int(snapshot.bleed_device_pixels))
        physical_size = core_physical + (bleed_physical * 2)
        image = QImage(
            physical_size,
            physical_size,
            QImage.Format.Format_ARGB32_Premultiplied,
        )
        image.fill(int(fill_rgba))
        image.setDevicePixelRatio(dpr)
        logical_bleed = bleed_physical / dpr
        phase_x = float(key.device_phase_x) / dpr
        phase_y = float(key.device_phase_y) / dpr
        painter = QPainter(image)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
            painter.translate(
                logical_bleed
                + phase_x
                - (key.tile_x * snapshot.logical_tile_size),
                logical_bleed
                + phase_y
                - (key.tile_y * snapshot.logical_tile_size),
            )
            self._play_snapshot(snapshot, painter)
        finally:
            painter.end()
        core = image.copy(
            bleed_physical,
            bleed_physical,
            core_physical,
            core_physical,
        )
        core.setDevicePixelRatio(dpr)
        return core

    def _play_snapshot(
        self,
        snapshot: CanvasOverlayRenderSnapshot,
        painter: QPainter,
    ) -> None:
        if snapshot.picture is not None:
            snapshot.picture.play(painter)
            return
        self._draw_area_commands(
            painter,
            snapshot.area_commands,
        )

    def _draw_area_commands(
        self,
        painter: QPainter,
        commands: tuple[AreaOverlayDrawCommand, ...],
    ) -> None:
        """Render detached commands in original document order."""

        for command in commands:
            if self._cancellation.is_cancelled():
                return
            painter.save()
            try:
                painter.setWorldTransform(command.image_to_overlay, combine=True)
                if command.fill_rgba is not None:
                    painter.setBrush(QColor.fromRgba(int(command.fill_rgba)))
                else:
                    painter.setBrush(Qt.BrushStyle.NoBrush)
                outline_pen = QPen(
                    QColor.fromRgba(int(command.outline_rgba)),
                    float(command.outline_width),
                    Qt.PenStyle.SolidLine,
                    Qt.PenCapStyle.RoundCap,
                    Qt.PenJoinStyle.RoundJoin,
                )
                outline_pen.setCosmetic(True)
                painter.setPen(outline_pen)
                # Fill and outer outline have the same ordering when issued
                # together by QPainter, and this removes one full path draw
                # from every passive area command.
                painter.drawPath(command.path)
                if self._cancellation.is_cancelled():
                    return
                painter.setBrush(Qt.BrushStyle.NoBrush)
                stroke_pen = QPen(
                    QColor.fromRgba(int(command.stroke_rgba)),
                    float(command.stroke_width),
                    Qt.PenStyle.SolidLine,
                    Qt.PenCapStyle.RoundCap,
                    Qt.PenJoinStyle.RoundJoin,
                )
                stroke_pen.setCosmetic(True)
                painter.setPen(stroke_pen)
                painter.drawPath(command.path)
            finally:
                painter.restore()
            if self._cancellation.is_cancelled():
                return
            if command.label is not None:
                top_left = command.label.top_left
                if top_left is None:
                    if command.label.centroid_key is None:
                        continue
                    centroid = self._centroid_cache.get_or_compute(
                        command.label.centroid_key,
                        command.path,
                    )
                    center = command.image_to_overlay.map(centroid)
                    top_left = QPointF(
                        center.x() + command.label.center_offset.x(),
                        center.y() + command.label.center_offset.y(),
                    )
                if self._cancellation.is_cancelled():
                    return
                painter.drawImage(
                    top_left,
                    command.label.image,
                )

    def _record_area_commands(
        self,
        snapshot: CanvasOverlayRenderSnapshot,
    ) -> QPicture:
        """Create the rare exact-composition fallback entirely in the worker."""

        picture = QPicture()
        painter = QPainter(picture)
        if not painter.isActive():  # pragma: no cover - defensive Qt backend
            raise RuntimeError("failed to create exact area command picture")
        bleed = float(snapshot.bleed_device_pixels) / max(
            float(snapshot.key.device_pixel_ratio),
            1.0,
        )
        tile_size = float(snapshot.logical_tile_size)
        painter.setClipRect(
            QRectF(
                snapshot.key.tile_x * tile_size,
                snapshot.key.tile_y * tile_size,
                tile_size,
                tile_size,
            ).adjusted(-bleed, -bleed, bleed, bleed)
        )
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        try:
            self._draw_area_commands(
                painter,
                snapshot.area_commands,
            )
        finally:
            painter.end()
        return picture

    @staticmethod
    def _has_visible_composition_difference(
        direct: QImage,
        flattened: QImage,
    ) -> bool:
        """Detect rounding errors large enough to become a visible overlap seam."""

        if direct.size() != flattened.size():
            return True

        def pixels(image: QImage) -> np.ndarray:
            raw = np.frombuffer(
                image.constBits(),
                dtype=np.uint8,
                count=image.sizeInBytes(),
            ).reshape((image.height(), image.bytesPerLine()))
            return raw[:, : image.width() * 4].reshape(
                (image.height(), image.width(), 4)
            )

        difference = np.abs(
            pixels(direct).astype(np.int16)
            - pixels(flattened).astype(np.int16)
        )
        significant = np.any(difference > 2, axis=2)
        return int(np.count_nonzero(significant)) > 8


class CanvasOverlayTileCache(QObject):
    """Workspace-wide bounded cache for exact-scale passive overlay tiles."""

    tileReady = Signal(object)
    tileFailed = Signal(object, str)

    def __init__(
        self,
        parent: QObject | None = None,
        *,
        max_entries: int = OVERLAY_TILE_MAX_ENTRIES,
        max_bytes: int = OVERLAY_TILE_MAX_BYTES,
        max_pending_bytes: int = OVERLAY_TILE_MAX_PENDING_BYTES,
        thread_pool: QThreadPool | None = None,
    ) -> None:
        super().__init__(parent)
        self._max_entries = max(1, int(max_entries))
        self._max_bytes = max(1, int(max_bytes))
        self._max_pending_bytes = max(1, int(max_pending_bytes))
        self._thread_pool = thread_pool or QThreadPool.globalInstance()
        self._area_centroids = _AreaCommandCentroidCache()
        self._tiles: OrderedDict[CanvasOverlayTileKey, _CachedTile] = OrderedDict()
        self._pending: dict[CanvasOverlayTileKey, _PendingTile] = {}
        self._inflight_sequences: set[int] = set()
        self._inflight_estimated_bytes: dict[int, int] = {}
        self._protected_by_owner: dict[
            int,
            frozenset[CanvasOverlayTileKey],
        ] = {}
        self._completions: queue.SimpleQueue[_TileRenderCompletion] = queue.SimpleQueue()
        self._request_sequence = 0
        self._completion_timer = QTimer(self)
        self._completion_timer.setInterval(5)
        self._completion_timer.timeout.connect(self._drain_completions)
        self._bytes = 0
        self._hits = 0
        self._misses = 0
        self._completed = 0
        self._dropped = 0

    @property
    def max_entries(self) -> int:
        """Maximum number of completed tiles retained by this cache."""

        return self._max_entries

    @property
    def max_bytes(self) -> int:
        """Maximum completed-payload byte budget for this cache."""

        return self._max_bytes

    def protect(
        self,
        owner_token: int,
        keys: Iterable[CanvasOverlayTileKey],
    ) -> None:
        """Protect an owner's visible tiles from guard-prefetch eviction."""

        self._require_owner_thread()
        token = int(owner_token)
        protected = frozenset(keys)
        if protected:
            self._protected_by_owner[token] = protected
        else:
            self._protected_by_owner.pop(token, None)

    def get(self, key: CanvasOverlayTileKey) -> QImage | None:
        self._require_owner_thread()
        cached = self._tiles.get(key)
        if cached is None:
            self._misses += 1
            return None
        if cached.image is None:
            self._misses += 1
            return None
        self._tiles.move_to_end(key)
        self._hits += 1
        return cached.image

    def get_picture(self, key: CanvasOverlayTileKey) -> QPicture | None:
        """Return an exact command tile for direct composition on the UI painter."""

        self._require_owner_thread()
        cached = self._tiles.get(key)
        if cached is None or cached.picture is None:
            self._misses += 1
            return None
        self._tiles.move_to_end(key)
        self._hits += 1
        return cached.picture

    def get_payload(
        self,
        key: CanvasOverlayTileKey,
    ) -> tuple[QImage | None, QPicture | None] | None:
        """Return the tile's fast raster or its exact-composition fallback."""

        self._require_owner_thread()
        cached = self._tiles.get(key)
        if cached is None:
            self._misses += 1
            return None
        self._tiles.move_to_end(key)
        self._hits += 1
        return cached.image, cached.picture

    def contains(self, key: CanvasOverlayTileKey) -> bool:
        self._require_owner_thread()
        return key in self._tiles

    def is_pending(self, key: CanvasOverlayTileKey) -> bool:
        self._require_owner_thread()
        return key in self._pending

    def request(self, snapshot: CanvasOverlayRenderSnapshot) -> bool:
        self._require_owner_thread()
        self._validate_snapshot(snapshot)
        # A completion may already be queued while the event-loop timer has not
        # run yet.  Drain it before applying the global pending budget so
        # completed payloads cannot unnecessarily block the next visible tile.
        self._drain_completions()
        key = snapshot.key
        if key in self._tiles or key in self._pending:
            return False
        estimated_bytes = self._estimate_pending_snapshot_bytes(snapshot)
        if (
            estimated_bytes > self._max_pending_bytes
            or self._pending_bytes() + estimated_bytes
            > self._max_pending_bytes
        ):
            return False
        # Exact-command snapshots require a detached byte copy.  The safe area
        # command path instead keeps Qt value types implicitly shared, avoiding
        # O(vertices) QPicture serialization on the UI thread.
        worker_snapshot = CanvasOverlayRenderSnapshot(
            request_id=snapshot.request_id,
            key=snapshot.key,
            picture=(
                self._clone_picture(snapshot.picture)
                if snapshot.picture is not None
                else None
            ),
            area_commands=tuple(
                self._clone_area_command(command)
                for command in snapshot.area_commands
            ),
            logical_tile_size=snapshot.logical_tile_size,
            bleed_device_pixels=snapshot.bleed_device_pixels,
            exact_composition=snapshot.exact_composition,
            adaptive_composition=snapshot.adaptive_composition,
            composition_probe_rgba=snapshot.composition_probe_rgba,
            known_empty=snapshot.known_empty,
        )
        cancellation = _CancellationFlag()
        self._request_sequence += 1
        request_sequence = self._request_sequence
        self._pending[key] = _PendingTile(
            request_sequence=request_sequence,
            cancellation=cancellation,
            estimated_bytes=estimated_bytes,
        )
        self._inflight_sequences.add(request_sequence)
        self._inflight_estimated_bytes[request_sequence] = estimated_bytes
        runnable = _TileRenderRunnable(
            worker_snapshot,
            cancellation,
            request_sequence,
            self._completions,
            self._area_centroids,
        )
        if not self._completion_timer.isActive():
            self._completion_timer.start()
        try:
            self._thread_pool.start(runnable)
        except Exception:
            current = self._pending.get(key)
            if current is not None and current.request_sequence == request_sequence:
                self._pending.pop(key, None)
            self._inflight_sequences.discard(request_sequence)
            self._inflight_estimated_bytes.pop(request_sequence, None)
            if not self._inflight_sequences:
                self._completion_timer.stop()
            raise
        # Deterministic inline pools complete before start() returns. Draining
        # here keeps the cache useful in tests without weakening real-thread
        # ownership: this method itself is restricted to the QObject thread.
        self._drain_completions()
        return True

    def cancel(self, key: CanvasOverlayTileKey) -> None:
        self._require_owner_thread()
        pending = self._pending.pop(key, None)
        if pending is not None:
            pending.cancellation.cancel()

    def invalidate_document(self, document_token: int) -> None:
        self._require_owner_thread()
        token = int(document_token)
        self._area_centroids.discard_document(token)
        for key in [key for key in self._tiles if key.document_token == token]:
            self._remove_tile(key)
        for key, pending in list(self._pending.items()):
            if key.document_token != token:
                continue
            pending.cancellation.cancel()
            self._pending.pop(key, None)
        self._discard_protected_document(token)

    def invalidate_namespace(
        self,
        document_token: int,
        zoom: float,
        device_pixel_ratio: float,
    ) -> None:
        """Discard one document's exact zoom/DPR tile namespace.

        Canvas epoch history is intentionally bounded.  Removing its oldest
        namespace must also remove completed and in-flight global payloads;
        otherwise returning to that scale can recreate epoch zero and match a
        stale tile produced before an intervening geometry change.

        Like the rest of this QObject-backed cache API, namespace invalidation
        is restricted to the cache owner thread.  Worker cancellation itself
        uses a thread-safe flag, and a late completion cannot be admitted
        after its pending-key ownership has been removed here.
        """

        self._require_owner_thread()
        token = int(document_token)
        namespace = (float(zoom), float(device_pixel_ratio))
        for key in list(self._tiles):
            if (
                key.document_token == token
                and (key.zoom, key.device_pixel_ratio) == namespace
            ):
                self._remove_tile(key)
        for key, pending in list(self._pending.items()):
            if (
                key.document_token != token
                or (key.zoom, key.device_pixel_ratio) != namespace
            ):
                continue
            pending.cancellation.cancel()
            self._pending.pop(key, None)

    def invalidate_coordinates(
        self,
        document_token: int,
        coordinates: set[tuple[float, float, int, int]],
    ) -> None:
        """Invalidate exact zoom/DPR/tile coordinate tuples for one document."""

        self._require_owner_thread()
        if not coordinates:
            return
        token = int(document_token)
        for key in list(self._tiles):
            coordinate = (
                key.zoom,
                key.device_pixel_ratio,
                key.tile_x,
                key.tile_y,
            )
            if key.document_token == token and coordinate in coordinates:
                self._remove_tile(key)
        for key, pending in list(self._pending.items()):
            coordinate = (
                key.zoom,
                key.device_pixel_ratio,
                key.tile_x,
                key.tile_y,
            )
            if key.document_token != token or coordinate not in coordinates:
                continue
            pending.cancellation.cancel()
            self._pending.pop(key, None)

    def clear(self) -> None:
        self._require_owner_thread()
        for pending in self._pending.values():
            pending.cancellation.cancel()
        self._pending.clear()
        self._tiles.clear()
        self._bytes = 0
        self._area_centroids.clear()
        self._protected_by_owner.clear()

    def stats(self) -> CanvasOverlayCacheStats:
        self._require_owner_thread()
        return CanvasOverlayCacheStats(
            entries=len(self._tiles),
            bytes=self._bytes,
            pending=len(self._pending),
            hits=self._hits,
            misses=self._misses,
            completed=self._completed,
            dropped=self._dropped,
            pending_bytes=self._pending_bytes(),
        )

    @Slot()
    def _drain_completions(self) -> None:
        self._require_owner_thread()
        while True:
            try:
                completion = self._completions.get_nowait()
            except queue.Empty:
                break
            self._inflight_sequences.discard(completion.request_sequence)
            self._inflight_estimated_bytes.pop(completion.request_sequence, None)
            if completion.cancelled:
                self._drop_completion(
                    completion.key,
                    completion.request_sequence,
                )
            elif completion.error is not None:
                self._on_failed(
                    completion.key,
                    completion.request_sequence,
                    completion.error,
                )
            elif completion.image is not None or completion.picture is not None:
                self._on_completed(
                    completion.key,
                    completion.request_sequence,
                    completion.image,
                    completion.picture,
                )
            else:  # pragma: no cover - impossible worker envelope
                self._on_failed(
                    completion.key,
                    completion.request_sequence,
                    "tile worker returned no image",
                )
        if not self._inflight_sequences:
            self._completion_timer.stop()

    def _drop_completion(
        self,
        key: CanvasOverlayTileKey,
        request_sequence: int,
    ) -> None:
        pending = self._pending.get(key)
        if pending is not None and pending.request_sequence == request_sequence:
            self._pending.pop(key, None)
        self._dropped += 1

    def _on_completed(
        self,
        key: CanvasOverlayTileKey,
        request_sequence: int,
        image: QImage | None,
        picture: QPicture | None,
    ) -> None:
        pending = self._pending.get(key)
        if (
            pending is None
            or pending.request_sequence != request_sequence
            or pending.cancellation.is_cancelled()
        ):
            self._dropped += 1
            return
        self._pending.pop(key, None)
        if image is not None and image.isNull():
            self._dropped += 1
            self.tileFailed.emit(key, "tile worker returned a null image")
            return
        if image is None and picture is None:
            self._dropped += 1
            self.tileFailed.emit(key, "tile worker returned an empty payload")
            return
        # A composition-sensitive tile can keep both its exact command stream
        # and its pan-only raster. Charge both payloads to the existing bounded
        # LRU rather than allowing interaction acceleration to bypass the
        # 128 MiB workspace budget.
        estimated_bytes = (
            (max(1, int(image.sizeInBytes())) if image is not None else 0)
            # QPicture.size() reports the recorded payload without copying the
            # entire byte stream back onto the UI thread.
            + (max(1, int(picture.size())) if picture is not None else 0)
        )
        if estimated_bytes > self._max_bytes:
            self._dropped += 1
            self.tileFailed.emit(
                key,
                "rendered tile exceeds the overlay cache byte budget",
            )
            return
        if not self._evict_for(estimated_bytes):
            self._dropped += 1
            self.tileFailed.emit(
                key,
                "rendered tile cannot be admitted without evicting a visible overlay tile",
            )
            return
        self._tiles[key] = _CachedTile(
            image=image,
            picture=picture,
            estimated_bytes=estimated_bytes,
        )
        self._tiles.move_to_end(key)
        self._bytes += estimated_bytes
        self._completed += 1
        self.tileReady.emit(key)

    def _on_failed(
        self,
        key: CanvasOverlayTileKey,
        request_sequence: int,
        message: str,
    ) -> None:
        pending = self._pending.get(key)
        if (
            pending is None
            or pending.request_sequence != request_sequence
            or pending.cancellation.is_cancelled()
        ):
            self._dropped += 1
            return
        self._pending.pop(key, None)
        self.tileFailed.emit(key, str(message))

    def _evict_for(self, required_bytes: int) -> bool:
        while self._tiles and (
            len(self._tiles) >= self._max_entries
            or self._bytes + required_bytes > self._max_bytes
        ):
            protected = self._protected_keys()
            oldest_key = next(
                (
                    candidate
                    for candidate in self._tiles
                    if candidate not in protected
                ),
                None,
            )
            if oldest_key is None:
                return False
            self._remove_tile(oldest_key)
        return (
            len(self._tiles) < self._max_entries
            and self._bytes + required_bytes <= self._max_bytes
        )

    def _remove_tile(self, key: CanvasOverlayTileKey) -> None:
        cached = self._tiles.pop(key, None)
        if cached is not None:
            self._bytes = max(0, self._bytes - cached.estimated_bytes)

    def _protected_keys(self) -> set[CanvasOverlayTileKey]:
        protected: set[CanvasOverlayTileKey] = set()
        for keys in self._protected_by_owner.values():
            protected.update(keys)
        return protected

    def _discard_protected_document(self, document_token: int) -> None:
        for owner_token, keys in list(self._protected_by_owner.items()):
            remaining = frozenset(
                key
                for key in keys
                if key.document_token != document_token
            )
            if remaining:
                self._protected_by_owner[owner_token] = remaining
            else:
                self._protected_by_owner.pop(owner_token, None)

    def _pending_bytes(self) -> int:
        # Sequence ownership outlives the visible pending-key entry.  A
        # cancelled runnable may still hold its detached snapshot until the
        # worker observes cancellation and reports completion.
        return sum(self._inflight_estimated_bytes.values())

    def _require_owner_thread(self) -> None:
        if QThread.currentThread() != self.thread():
            raise RuntimeError(
                "CanvasOverlayTileCache must be accessed from its QObject thread"
            )

    @staticmethod
    def _validate_snapshot(snapshot: CanvasOverlayRenderSnapshot) -> None:
        key = snapshot.key
        if (
            not math.isfinite(float(key.zoom))
            or float(key.zoom) <= 0.0
            or not math.isfinite(float(key.device_pixel_ratio))
            or float(key.device_pixel_ratio) <= 0.0
            or not math.isfinite(float(key.device_phase_x))
            or not math.isfinite(float(key.device_phase_y))
            or not 0.0 <= float(key.device_phase_x) < 1.0
            or not 0.0 <= float(key.device_phase_y) < 1.0
        ):
            raise ValueError(
                "overlay tile scale must be positive and device phase must be in [0, 1)"
            )
        if int(snapshot.logical_tile_size) <= 0:
            raise ValueError("overlay tile logical size must be positive")
        if int(snapshot.bleed_device_pixels) < 0:
            raise ValueError("overlay tile bleed must not be negative")
        if snapshot.known_empty:
            if (
                snapshot.picture is not None
                or snapshot.area_commands
                or snapshot.exact_composition
                or snapshot.adaptive_composition
            ):
                raise ValueError(
                    "known-empty overlay tile cannot contain draw commands"
                )
            return
        if (snapshot.picture is None) == (not snapshot.area_commands):
            raise ValueError(
                "overlay tile snapshot must contain exactly one command payload"
            )

    @staticmethod
    def _clone_picture(picture: QPicture) -> QPicture:
        clone = QPicture()
        payload = bytes(picture.data())
        if payload:
            clone.setData(payload)
        return clone

    @staticmethod
    def _estimate_pending_snapshot_bytes(
        snapshot: CanvasOverlayRenderSnapshot,
    ) -> int:
        """Conservatively estimate detached worker snapshot storage.

        QPicture exposes its recorded byte count without copying ``data()``.
        QPainterPath has no byte-size API, so each path element is charged a
        deliberately conservative fixed amount.  Label QImages report their
        actual backing buffer size.  Repeated/shared payloads may therefore be
        counted more than once, which is preferable to admitting an
        unexpectedly large burst of worker snapshots.
        """

        estimated = _PENDING_SNAPSHOT_OVERHEAD_BYTES
        if snapshot.picture is not None:
            return estimated + max(1, int(snapshot.picture.size()))
        for command in snapshot.area_commands:
            estimated += _PENDING_AREA_COMMAND_OVERHEAD_BYTES
            estimated += max(
                1,
                int(command.path.elementCount()) * _PENDING_PATH_ELEMENT_BYTES,
            )
            label = command.label
            if label is not None:
                estimated += _PENDING_LABEL_COMMAND_OVERHEAD_BYTES
                estimated += max(1, int(label.image.sizeInBytes()))
        return max(1, int(estimated))

    @staticmethod
    def _clone_area_command(
        command: AreaOverlayDrawCommand,
    ) -> AreaOverlayDrawCommand:
        """Detach the worker envelope without walking path elements."""

        label = command.label
        return AreaOverlayDrawCommand(
            path=QPainterPath(command.path),
            image_to_overlay=QTransform(command.image_to_overlay),
            fill_rgba=command.fill_rgba,
            outline_rgba=int(command.outline_rgba),
            outline_width=float(command.outline_width),
            stroke_rgba=int(command.stroke_rgba),
            stroke_width=float(command.stroke_width),
            label=(
                None
                if label is None
                else AreaOverlayLabelCommand(
                    image=QImage(label.image),
                    top_left=(
                        QPointF(label.top_left)
                        if label.top_left is not None
                        else None
                    ),
                    center_offset=QPointF(label.center_offset),
                    centroid_key=label.centroid_key,
                )
            ),
        )


canvas_overlay_tile_cache = CanvasOverlayTileCache()
