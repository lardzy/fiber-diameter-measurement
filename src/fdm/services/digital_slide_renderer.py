from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import shutil
from threading import Condition, Lock, Thread, get_ident
from time import monotonic

from PySide6.QtCore import QRect, QRectF, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QRegion

from fdm.services.digital_slide_store import (
    DigitalSlideManifest,
    DigitalSlideStore,
    DigitalSlideTileDescriptor,
)


_RENDER_CACHE_VERSION = 1
_DEFAULT_MEMORY_CACHE_BYTES = 256 * 1024 * 1024
_MIN_CACHE_FREE_RESERVE_BYTES = 512 * 1024 * 1024
_DERIVED_CACHE_REGISTRY_LOCK = Lock()


class _DerivedCacheRootState:
    def __init__(self) -> None:
        self.lock = Lock()
        self.known_total_bytes: int | None = None


_DERIVED_CACHE_ROOT_STATES: dict[str, _DerivedCacheRootState] = {}


def _derived_cache_root_state(root: Path) -> _DerivedCacheRootState:
    key = str(root.resolve(strict=False))
    with _DERIVED_CACHE_REGISTRY_LOCK:
        state = _DERIVED_CACHE_ROOT_STATES.get(key)
        if state is None:
            state = _DerivedCacheRootState()
            _DERIVED_CACHE_ROOT_STATES[key] = state
        return state


@dataclass(frozen=True, slots=True)
class DigitalSlideRenderRequest:
    request_id: int
    purpose: str
    source_rect: tuple[float, float, float, float]
    output_size_px: tuple[int, int]
    focus_index: int
    device_pixel_ratio: float
    blend_width: int = 0
    velocity_px_per_second: tuple[float, float] = (0.0, 0.0)
    force_lod: int | None = None


@dataclass(frozen=True, slots=True)
class DigitalSlideRenderFrame:
    request_id: int
    purpose: str
    source_rect: tuple[float, float, float, float]
    output_size_px: tuple[int, int]
    focus_index: int
    device_pixel_ratio: float
    lod: int
    image: QImage
    elapsed_ms: float
    decoded_tiles: int
    cache_hits: int


@dataclass(frozen=True, slots=True)
class DigitalSlideRenderFailure:
    request_id: int
    purpose: str
    focus_index: int
    message: str


@dataclass(frozen=True, slots=True)
class DigitalSlideRendererStats:
    submitted: int
    completed: int
    cancelled: int
    stale_dropped: int
    decoded_tiles: int
    memory_hits: int
    disk_hits: int
    memory_bytes: int
    pending_requests: int


@dataclass(frozen=True, slots=True)
class _TileSpatialIndex:
    cell_width: int
    cell_height: int
    cells: dict[tuple[int, int], tuple[DigitalSlideTileDescriptor, ...]]

    @classmethod
    def build(
        cls,
        descriptors: tuple[DigitalSlideTileDescriptor, ...],
    ) -> "_TileSpatialIndex":
        if not descriptors:
            return cls(1, 1, {})
        cell_width = max(1, int(max(item.width for item in descriptors)))
        cell_height = max(1, int(max(item.height for item in descriptors)))
        mutable: dict[tuple[int, int], list[DigitalSlideTileDescriptor]] = {}
        for descriptor in descriptors:
            left = int(math.floor(descriptor.x / cell_width))
            right = int(math.floor((descriptor.x + descriptor.width - 1) / cell_width))
            top = int(math.floor(descriptor.y / cell_height))
            bottom = int(math.floor((descriptor.y + descriptor.height - 1) / cell_height))
            for cell_y in range(top, bottom + 1):
                for cell_x in range(left, right + 1):
                    mutable.setdefault((cell_x, cell_y), []).append(descriptor)
        return cls(
            cell_width,
            cell_height,
            {
                key: tuple(sorted(items, key=lambda item: item.tile_id))
                for key, items in mutable.items()
            },
        )

    def query(
        self,
        rect: tuple[float, float, float, float],
    ) -> tuple[DigitalSlideTileDescriptor, ...]:
        x, y, width, height = rect
        if width <= 0.0 or height <= 0.0:
            return ()
        left = int(math.floor(x / self.cell_width))
        right = int(math.floor((x + width) / self.cell_width))
        top = int(math.floor(y / self.cell_height))
        bottom = int(math.floor((y + height) / self.cell_height))
        matched: dict[int, DigitalSlideTileDescriptor] = {}
        for cell_y in range(top, bottom + 1):
            for cell_x in range(left, right + 1):
                for descriptor in self.cells.get((cell_x, cell_y), ()):
                    descriptor_rect = (
                        float(descriptor.x),
                        float(descriptor.y),
                        float(descriptor.width),
                        float(descriptor.height),
                    )
                    if _rects_intersect(rect, descriptor_rect):
                        matched[int(descriptor.tile_id)] = descriptor
        return tuple(matched[key] for key in sorted(matched))


class DigitalSlideDerivedCache:
    """Versioned, disposable display-LOD cache stored outside ``.fdmslide``."""

    def __init__(self, root: str | Path, *, byte_limit: int) -> None:
        self.root = Path(root).expanduser()
        self.byte_limit = max(0, int(byte_limit))
        self._root_state = _derived_cache_root_state(self.root)
        self._lock = self._root_state.lock

    @staticmethod
    def source_fingerprint(
        path: str | Path,
        manifest: DigitalSlideManifest,
        *,
        source_identity: str | Path | None = None,
    ) -> str:
        source = Path(path).expanduser()
        identity = (
            Path(source_identity).expanduser().resolve(strict=False)
            if source_identity is not None
            else source.resolve(strict=False)
        )
        # A network slide may be rendered from a localized temporary copy.  In
        # that case use the original source's size/mtime whenever it is still
        # reachable; otherwise fall back to the actual SQLite file.  Both paths
        # retain the required identity + size + mtime invalidation tuple.
        stat_source = identity if identity.is_file() else source
        try:
            stat = stat_source.stat()
            stat_payload = (
                int(stat.st_size),
                int(stat.st_mtime_ns),
            )
        except OSError:
            stat_payload = (0, 0)
        manifest_payload = json.dumps(
            manifest.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        token = "\0".join(
            (
                str(identity),
                str(stat_payload[0]),
                str(stat_payload[1]),
                manifest_payload,
                str(_RENDER_CACHE_VERSION),
            )
        )
        return sha256(token.encode("utf-8", errors="surrogatepass")).hexdigest()

    def _path(self, fingerprint: str, focus_index: int, tile_id: int, lod: int) -> Path:
        return (
            self.root
            / f"v{_RENDER_CACHE_VERSION}"
            / fingerprint[:2]
            / fingerprint
            / f"z{int(focus_index)}"
            / f"tile-{int(tile_id)}-lod-{int(lod)}.png"
        )

    def load(
        self,
        fingerprint: str,
        *,
        focus_index: int,
        tile_id: int,
        lod: int,
    ) -> QImage:
        if self.byte_limit <= 0 or lod <= 0:
            return QImage()
        path = self._path(fingerprint, focus_index, tile_id, lod)
        try:
            image = QImage(str(path))
            if image.isNull():
                return QImage()
            os.utime(path, None)
            return image
        except OSError:
            return QImage()

    def store(
        self,
        fingerprint: str,
        image: QImage,
        *,
        focus_index: int,
        tile_id: int,
        lod: int,
    ) -> None:
        if self.byte_limit <= 0 or lod <= 0 or image.isNull():
            return
        path = self._path(fingerprint, focus_index, tile_id, lod)
        with self._lock:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                previous_size = path.stat().st_size if path.is_file() else 0
                temporary = path.with_name(
                    f".{path.stem}.{os.getpid()}.{get_ident()}.tmp.png"
                )
                if not image.save(str(temporary), "PNG"):
                    temporary.unlink(missing_ok=True)
                    return
                os.replace(temporary, path)
                stored_size = path.stat().st_size
                if self._root_state.known_total_bytes is None:
                    self._root_state.known_total_bytes = self._scan_total_bytes_locked()
                else:
                    self._root_state.known_total_bytes = max(
                        0,
                        self._root_state.known_total_bytes
                        - previous_size
                        + stored_size,
                    )
                try:
                    free = shutil.disk_usage(self.root).free
                except OSError:
                    free = _MIN_CACHE_FREE_RESERVE_BYTES
                if (
                    self._root_state.known_total_bytes > self.byte_limit
                    or free < _MIN_CACHE_FREE_RESERVE_BYTES
                ):
                    self._trim_locked()
            except OSError:
                try:
                    temporary.unlink(missing_ok=True)
                except (OSError, UnboundLocalError):
                    pass

    def clear(self) -> None:
        with self._lock:
            version_root = self.root / f"v{_RENDER_CACHE_VERSION}"
            if version_root.is_dir():
                shutil.rmtree(version_root, ignore_errors=True)
            self._root_state.known_total_bytes = 0

    def clear_fingerprint(self, fingerprint: str) -> None:
        with self._lock:
            fingerprint_root = (
                self.root
                / f"v{_RENDER_CACHE_VERSION}"
                / str(fingerprint)[:2]
                / str(fingerprint)
            )
            if fingerprint_root.is_dir():
                shutil.rmtree(fingerprint_root, ignore_errors=True)
            self._root_state.known_total_bytes = None

    def _scan_total_bytes_locked(self) -> int:
        version_root = self.root / f"v{_RENDER_CACHE_VERSION}"
        try:
            return sum(
                path.stat().st_size
                for path in version_root.rglob("*.png")
                if path.is_file()
            )
        except OSError:
            return 0

    def _trim_locked(self) -> None:
        if self.byte_limit <= 0:
            return
        version_root = self.root / f"v{_RENDER_CACHE_VERSION}"
        try:
            files = [path for path in version_root.rglob("*.png") if path.is_file()]
            records = [(path.stat().st_mtime_ns, path.stat().st_size, path) for path in files]
        except OSError:
            return
        total = sum(size for _mtime, size, _path in records)
        try:
            free = shutil.disk_usage(self.root).free
        except OSError:
            free = _MIN_CACHE_FREE_RESERVE_BYTES
        target = min(self.byte_limit, max(0, total + free - _MIN_CACHE_FREE_RESERVE_BYTES))
        if total <= target:
            self._root_state.known_total_bytes = total
            return
        for _mtime, size, path in sorted(records):
            try:
                path.unlink(missing_ok=True)
                total -= size
            except OSError:
                continue
            if total <= target:
                break
        self._root_state.known_total_bytes = total


class DigitalSlideRenderer:
    """One long-lived, thread-affine SQLite renderer for an active slide."""

    def __init__(
        self,
        source_path: str | Path,
        manifest: DigitalSlideManifest,
        *,
        source_identity: str | Path | None = None,
        cache_root: str | Path,
        disk_cache_bytes: int,
        result_callback: Callable[[DigitalSlideRenderFrame], None],
        failure_callback: Callable[[DigitalSlideRenderFailure], None],
        memory_cache_bytes: int = _DEFAULT_MEMORY_CACHE_BYTES,
    ) -> None:
        self.source_path = Path(source_path)
        self.manifest = manifest
        self._result_callback = result_callback
        self._failure_callback = failure_callback
        self._derived_cache = DigitalSlideDerivedCache(
            cache_root,
            byte_limit=disk_cache_bytes,
        )
        self._fingerprint = self._derived_cache.source_fingerprint(
            self.source_path,
            manifest,
            source_identity=source_identity,
        )
        self._memory_cache_limit = max(1, int(memory_cache_bytes))
        self._memory_cache: OrderedDict[tuple[int, int, int], QImage] = OrderedDict()
        self._memory_cache_bytes = 0
        self._pending_persistent_keys: OrderedDict[
            tuple[int, int, int], None
        ] = OrderedDict()
        self._descriptors: dict[int, tuple[DigitalSlideTileDescriptor, ...]] = {}
        self._spatial_indexes: dict[int, _TileSpatialIndex] = {}
        self._last_lod: dict[tuple[str, int], int] = {}
        self._condition = Condition()
        self._latest_display: DigitalSlideRenderRequest | None = None
        self._latest_native: DigitalSlideRenderRequest | None = None
        self._latest_overview: DigitalSlideRenderRequest | None = None
        self._active_request: DigitalSlideRenderRequest | None = None
        self._closed = False
        self._submitted = 0
        self._completed = 0
        self._cancelled = 0
        self._stale_dropped = 0
        self._decoded_tiles = 0
        self._memory_hits = 0
        self._disk_hits = 0
        self._thread = Thread(
            target=self._run,
            name=f"fdm-slide-renderer-{self.source_path.name}",
            daemon=True,
        )
        self._thread.start()

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    def submit(self, request: DigitalSlideRenderRequest) -> None:
        with self._condition:
            if self._closed:
                return
            if request.purpose == "native":
                self._latest_native = request
            elif request.purpose == "overview":
                self._latest_overview = request
            else:
                self._latest_display = request
            self._submitted += 1
            self._condition.notify()

    def close(self, *, timeout: float = 2.0) -> None:
        with self._condition:
            self._closed = True
            self._latest_display = None
            self._latest_native = None
            self._latest_overview = None
            self._condition.notify_all()
        if self._thread.is_alive():
            self._thread.join(timeout=max(0.0, float(timeout)))
        if not self._thread.is_alive():
            self._memory_cache.clear()
            self._memory_cache_bytes = 0
            self._pending_persistent_keys.clear()

    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def clear_derived_cache(self) -> None:
        self._derived_cache.clear_fingerprint(self._fingerprint)

    def stats(self) -> DigitalSlideRendererStats:
        with self._condition:
            pending = int(self._active_request is not None) + int(
                self._latest_display is not None
            ) + int(
                self._latest_native is not None
            ) + int(self._latest_overview is not None)
            return DigitalSlideRendererStats(
                submitted=self._submitted,
                completed=self._completed,
                cancelled=self._cancelled,
                stale_dropped=self._stale_dropped,
                decoded_tiles=self._decoded_tiles,
                memory_hits=self._memory_hits,
                disk_hits=self._disk_hits,
                memory_bytes=self._memory_cache_bytes,
                pending_requests=pending,
            )

    def _take_request(self) -> DigitalSlideRenderRequest | None:
        with self._condition:
            while (
                not self._closed
                and self._latest_display is None
                and self._latest_native is None
                and self._latest_overview is None
            ):
                self._condition.wait()
            if self._closed:
                return None
            request = (
                self._latest_display
                or self._latest_native
                or self._latest_overview
            )
            if request is self._latest_display:
                self._latest_display = None
            elif request is self._latest_native:
                self._latest_native = None
            else:
                self._latest_overview = None
            self._active_request = request
            return request

    def _finish_request(self, request: DigitalSlideRenderRequest) -> None:
        with self._condition:
            if self._active_request is request:
                self._active_request = None

    def _run(self) -> None:
        store: DigitalSlideStore | None = None
        try:
            store = DigitalSlideStore(self.source_path)
            store.open_read_only()
            store.set_decoded_image_cache_budget(max_images=0, max_bytes=0)
        except Exception as exc:  # noqa: BLE001 - publish worker startup failure
            message = f"{type(exc).__name__}: {exc}"
            if store is not None:
                try:
                    store.close()
                except Exception:
                    pass
            # Keep the worker alive in a terminal failure loop.  Explicit retry
            # requests then receive a deterministic error instead of entering
            # a dead queue after the first failed open.
            while True:
                failed_request = self._take_request()
                if failed_request is None:
                    return
                self._finish_request(failed_request)
                self._failure_callback(
                    DigitalSlideRenderFailure(
                        request_id=failed_request.request_id,
                        purpose=failed_request.purpose,
                        focus_index=failed_request.focus_index,
                        message=message,
                    )
                )
        try:
            while True:
                request = self._take_request()
                if request is None:
                    return
                try:
                    frame = self._render(store, request)
                except Exception as exc:  # noqa: BLE001 - publish worker failure
                    current = self._is_current(request)
                    self._finish_request(request)
                    if current:
                        self._failure_callback(
                            DigitalSlideRenderFailure(
                                request_id=request.request_id,
                                purpose=request.purpose,
                                focus_index=request.focus_index,
                                message=f"{type(exc).__name__}: {exc}",
                            )
                        )
                    continue
                if frame is None:
                    with self._condition:
                        self._cancelled += 1
                    self._finish_request(request)
                    continue
                if not self._is_current(request):
                    with self._condition:
                        self._stale_dropped += 1
                    self._finish_request(request)
                    continue
                with self._condition:
                    self._completed += 1
                self._finish_request(request)
                self._result_callback(frame)
                if request.purpose == "display":
                    try:
                        self._prefetch(store, request, frame.lod)
                    except Exception:
                        # Prefetch is opportunistic and must never terminate the
                        # renderer that services visible/native requests.
                        pass
                try:
                    self._flush_derived_cache()
                except Exception:
                    # Derived-cache persistence is disposable and cannot make a
                    # healthy read-only slide renderer fail.
                    pass
        finally:
            try:
                if store is not None:
                    store.close()
            except Exception:
                pass
            self._memory_cache.clear()
            self._memory_cache_bytes = 0
            self._pending_persistent_keys.clear()

    def _is_current(self, request: DigitalSlideRenderRequest) -> bool:
        with self._condition:
            if self._closed:
                return False
            latest = (
                self._latest_native
                if request.purpose == "native"
                else (
                    self._latest_overview
                    if request.purpose == "overview"
                    else self._latest_display
                )
            )
            return latest is None or latest.request_id <= request.request_id

    def _should_cancel(
        self,
        request: DigitalSlideRenderRequest,
    ) -> bool:
        with self._condition:
            if self._closed:
                return True
            if request.purpose == "overview" and (
                self._latest_display is not None
                or self._latest_native is not None
            ):
                return True
            if request.purpose == "native" and self._latest_display is not None:
                return True
            latest = (
                self._latest_native
                if request.purpose == "native"
                else (
                    self._latest_overview
                    if request.purpose == "overview"
                    else self._latest_display
                )
            )
            if latest is None or latest.request_id <= request.request_id:
                return False
            if latest.focus_index != request.focus_index:
                return True
            return not _rects_intersect(latest.source_rect, request.source_rect)

    def _focus_descriptors(
        self,
        store: DigitalSlideStore,
        focus_index: int,
    ) -> tuple[DigitalSlideTileDescriptor, ...]:
        cached = self._descriptors.get(int(focus_index))
        if cached is not None:
            return cached
        descriptors = tuple(
            sorted(
                store.list_tile_descriptors(z_index=int(focus_index)),
                key=lambda item: item.tile_id,
            )
        )
        self._descriptors[int(focus_index)] = descriptors
        self._spatial_indexes[int(focus_index)] = _TileSpatialIndex.build(descriptors)
        return descriptors

    def _descriptors_in_rect(
        self,
        store: DigitalSlideStore,
        focus_index: int,
        rect: tuple[float, float, float, float],
    ) -> tuple[DigitalSlideTileDescriptor, ...]:
        self._focus_descriptors(store, focus_index)
        return self._spatial_indexes[int(focus_index)].query(rect)

    def _render(
        self,
        store: DigitalSlideStore,
        request: DigitalSlideRenderRequest,
    ) -> DigitalSlideRenderFrame | None:
        started = monotonic()
        x, y, width, height = request.source_rect
        output_width = max(1, int(request.output_size_px[0]))
        output_height = max(1, int(request.output_size_px[1]))
        if request.purpose == "overview":
            legacy_overview = store.read_focus_overview(
                request.focus_index,
                maximum_edge=max(output_width, output_height),
            )
            if not legacy_overview.isNull():
                image = legacy_overview.scaled(
                    output_width,
                    output_height,
                    Qt.AspectRatioMode.IgnoreAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                return DigitalSlideRenderFrame(
                    request_id=request.request_id,
                    purpose=request.purpose,
                    source_rect=request.source_rect,
                    output_size_px=request.output_size_px,
                    focus_index=request.focus_index,
                    device_pixel_ratio=request.device_pixel_ratio,
                    lod=_lod_for_scale(
                        width,
                        height,
                        output_width,
                        output_height,
                    ),
                    image=image,
                    elapsed_ms=(monotonic() - started) * 1000.0,
                    decoded_tiles=0,
                    cache_hits=1,
                )
        lod = (
            max(0, int(request.force_lod))
            if request.force_lod is not None
            else self._lod_with_hysteresis(
                request,
                source_width=width,
                source_height=height,
                output_width=output_width,
                output_height=output_height,
            )
        )
        output = QImage(
            output_width,
            output_height,
            QImage.Format.Format_RGB32,
        )
        output.fill(QColor("#101820"))
        painter = QPainter(output)
        if not painter.isActive():
            raise RuntimeError("could not create digital-slide frame painter")
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        scale_x = output_width / max(width, 1e-9)
        scale_y = output_height / max(height, 1e-9)
        covered = QRegion()
        decoded_before = self._decoded_tiles
        hits_before = self._memory_hits + self._disk_hits
        try:
            for descriptor in self._descriptors_in_rect(
                store,
                request.focus_index,
                request.source_rect,
            ):
                if self._should_cancel(request):
                    return None
                image = self._tile_image(store, descriptor, lod)
                if self._should_cancel(request):
                    return None
                if image.isNull():
                    continue
                target = QRectF(
                    (descriptor.x - x) * scale_x,
                    (descriptor.y - y) * scale_y,
                    descriptor.width * scale_x,
                    descriptor.height * scale_y,
                )
                target_int = target.toAlignedRect().intersected(output.rect())
                if target_int.isEmpty():
                    continue
                if request.blend_width <= 0 or covered.isEmpty():
                    painter.drawImage(target, image)
                    covered = covered.united(QRegion(target_int))
                    continue
                overlap = QRegion(target_int).intersected(covered)
                if overlap.isEmpty():
                    painter.drawImage(target, image)
                    covered = covered.united(QRegion(target_int))
                    continue
                edge_x = max(1, int(round(request.blend_width * scale_x)))
                edge_y = max(1, int(round(request.blend_width * scale_y)))
                edge = QRegion(
                    QRect(target_int.left(), target_int.top(), edge_x, target_int.height())
                )
                edge = edge.united(
                    QRegion(
                        QRect(
                            target_int.right() - edge_x + 1,
                            target_int.top(),
                            edge_x,
                            target_int.height(),
                        )
                    )
                )
                edge = edge.united(
                    QRegion(QRect(target_int.left(), target_int.top(), target_int.width(), edge_y))
                )
                edge = edge.united(
                    QRegion(
                        QRect(
                            target_int.left(),
                            target_int.bottom() - edge_y + 1,
                            target_int.width(),
                            edge_y,
                        )
                    )
                )
                blend = overlap.intersected(edge)
                opaque = QRegion(target_int).subtracted(blend)
                if not opaque.isEmpty():
                    painter.save()
                    painter.setClipRegion(opaque)
                    painter.drawImage(target, image)
                    painter.restore()
                if not blend.isEmpty():
                    painter.save()
                    painter.setClipRegion(blend)
                    painter.setOpacity(0.5)
                    painter.drawImage(target, image)
                    painter.restore()
                covered = covered.united(QRegion(target_int))
        finally:
            painter.end()
        return DigitalSlideRenderFrame(
            request_id=request.request_id,
            purpose=request.purpose,
            source_rect=request.source_rect,
            output_size_px=request.output_size_px,
            focus_index=request.focus_index,
            device_pixel_ratio=request.device_pixel_ratio,
            lod=lod,
            image=output,
            elapsed_ms=(monotonic() - started) * 1000.0,
            decoded_tiles=self._decoded_tiles - decoded_before,
            cache_hits=(self._memory_hits + self._disk_hits) - hits_before,
        )

    def _lod_with_hysteresis(
        self,
        request: DigitalSlideRenderRequest,
        *,
        source_width: float,
        source_height: float,
        output_width: int,
        output_height: int,
    ) -> int:
        ratio = max(
            float(source_width) / max(1, int(output_width)),
            float(source_height) / max(1, int(output_height)),
            1.0,
        )
        candidate = _lod_for_scale(
            source_width,
            source_height,
            output_width,
            output_height,
        )
        key = (request.purpose, int(request.focus_index))
        previous = self._last_lod.get(key)
        if previous is not None and candidate == previous + 1:
            transition = float(1 << candidate)
            if ratio < transition * 1.15:
                candidate = previous
        elif previous is not None and candidate == previous - 1:
            transition = float(1 << previous)
            if ratio > transition / 1.15:
                candidate = previous
        candidate = max(0, min(16, candidate))
        self._last_lod[key] = candidate
        return candidate

    def _tile_image(
        self,
        store: DigitalSlideStore,
        descriptor: DigitalSlideTileDescriptor,
        lod: int,
    ) -> QImage:
        key = (int(descriptor.z_index), int(descriptor.tile_id), int(lod))
        cached = self._memory_cache.get(key)
        if cached is not None:
            self._memory_cache.move_to_end(key)
            self._memory_hits += 1
            return cached
        if lod > 0:
            cached = self._derived_cache.load(
                self._fingerprint,
                focus_index=descriptor.z_index,
                tile_id=descriptor.tile_id,
                lod=lod,
            )
            if not cached.isNull():
                self._disk_hits += 1
                self._remember_tile(key, cached)
                return cached
        source_image: QImage | None = None
        if lod > 0:
            # Reuse the closest sharper in-memory level when zooming farther
            # out.  This avoids decoding the same SQLite BLOB once per LOD.
            for source_lod in range(lod - 1, -1, -1):
                source_key = (
                    int(descriptor.z_index),
                    int(descriptor.tile_id),
                    int(source_lod),
                )
                source_image = self._memory_cache.get(source_key)
                if source_image is not None:
                    self._memory_cache.move_to_end(source_key)
                    self._memory_hits += 1
                    break
        if source_image is None:
            source_image = store.read_tile_image(descriptor.tile_id)
            if source_image.isNull():
                return source_image
            self._decoded_tiles += 1
            self._remember_tile(
                (int(descriptor.z_index), int(descriptor.tile_id), 0),
                source_image,
            )
        image = source_image
        if lod > 0:
            divisor = 1 << lod
            image = source_image.scaled(
                max(1, int(math.ceil(descriptor.width / divisor))),
                max(1, int(math.ceil(descriptor.height / divisor))),
                Qt.AspectRatioMode.IgnoreAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        self._remember_tile(key, image)
        if lod > 0 and self._derived_cache.byte_limit > 0:
            # Persistence is deliberately deferred until after the visible
            # frame and directional prefetch have completed.  Retain only the
            # cache key so this queue cannot hold images beyond the byte-LRU.
            self._pending_persistent_keys[key] = None
            self._pending_persistent_keys.move_to_end(key)
        return image

    def _remember_tile(self, key: tuple[int, int, int], image: QImage) -> None:
        if image.isNull():
            return
        previous = self._memory_cache.pop(key, None)
        if previous is not None:
            self._memory_cache_bytes -= max(0, int(previous.sizeInBytes()))
        self._memory_cache[key] = image
        self._memory_cache_bytes += max(0, int(image.sizeInBytes()))
        while self._memory_cache and self._memory_cache_bytes > self._memory_cache_limit:
            _removed_key, removed = self._memory_cache.popitem(last=False)
            self._memory_cache_bytes = max(
                0,
                self._memory_cache_bytes - max(0, int(removed.sizeInBytes())),
            )

    def _flush_derived_cache(self) -> None:
        while self._pending_persistent_keys:
            with self._condition:
                interrupted = (
                    self._latest_display is not None
                    or self._latest_native is not None
                    or self._closed
                )
            if interrupted:
                return
            key, _unused = self._pending_persistent_keys.popitem(last=False)
            image = self._memory_cache.get(key)
            if image is None or image.isNull():
                continue
            focus_index, tile_id, lod = key
            self._derived_cache.store(
                self._fingerprint,
                image,
                focus_index=focus_index,
                tile_id=tile_id,
                lod=lod,
            )

    def _prefetch(
        self,
        store: DigitalSlideStore,
        request: DigitalSlideRenderRequest,
        lod: int,
    ) -> None:
        velocity_x, velocity_y = request.velocity_px_per_second
        x, y, width, height = request.source_rect
        prediction = 0.18
        forward_x = velocity_x * prediction
        forward_y = velocity_y * prediction
        guard = (
            x - width * 0.25 + min(0.0, forward_x),
            y - height * 0.25 + min(0.0, forward_y),
            width * 1.5 + abs(forward_x),
            height * 1.5 + abs(forward_y),
        )
        descriptors = list(
            self._descriptors_in_rect(
                store,
                request.focus_index,
                guard,
            )
        )
        center_x = x + width / 2.0
        center_y = y + height / 2.0
        speed = math.hypot(velocity_x, velocity_y)
        direction_x = velocity_x / speed if speed > 1.0e-9 else 0.0
        direction_y = velocity_y / speed if speed > 1.0e-9 else 0.0

        def prefetch_order(
            descriptor: DigitalSlideTileDescriptor,
        ) -> tuple[int, float]:
            tile_x = descriptor.x + descriptor.width / 2.0
            tile_y = descriptor.y + descriptor.height / 2.0
            offset_x = tile_x - center_x
            offset_y = tile_y - center_y
            forward = offset_x * direction_x + offset_y * direction_y
            if speed <= 1.0e-9:
                priority = 1
            elif forward > 0.0:
                priority = 0
            elif abs(forward) <= max(descriptor.width, descriptor.height):
                priority = 1
            else:
                priority = 2
            return priority, offset_x * offset_x + offset_y * offset_y

        for descriptor in sorted(descriptors, key=prefetch_order):
            with self._condition:
                interrupted = (
                    self._latest_display is not None
                    or self._latest_native is not None
                    or self._closed
                )
            if interrupted:
                return
            self._tile_image(store, descriptor, lod)


def _lod_for_scale(
    source_width: float,
    source_height: float,
    output_width: int,
    output_height: int,
) -> int:
    source_per_device_pixel = max(
        float(source_width) / max(1, int(output_width)),
        float(source_height) / max(1, int(output_height)),
        1.0,
    )
    return max(0, min(16, int(math.floor(math.log2(source_per_device_pixel)))))


def _rects_intersect(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> bool:
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    return (
        aw > 0.0
        and ah > 0.0
        and bw > 0.0
        and bh > 0.0
        and ax < bx + bw
        and bx < ax + aw
        and ay < by + bh
        and by < ay + ah
    )


__all__ = [
    "DigitalSlideDerivedCache",
    "DigitalSlideRenderFailure",
    "DigitalSlideRenderFrame",
    "DigitalSlideRenderRequest",
    "DigitalSlideRenderer",
    "DigitalSlideRendererStats",
]
