"""One isolated Qt raster worker, shared by the whole workspace.

PySide can retain the interpreter lock inside a long native stroke operation.
An isolated interpreter keeps those calls off the GUI interpreter as well as
its Qt thread. Only detached, byte-bounded value snapshots cross this boundary.
"""

from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import asdict
import multiprocessing
from threading import Lock

from PySide6.QtCore import QByteArray, QDataStream, QIODevice, QPointF
from PySide6.QtGui import QImage, QPainterPath, QPicture, QTransform

_pool = None
_pool_lock = Lock()
_application = None
_centroids = None


def _initialize_worker():
    global _application, _centroids
    import faulthandler

    faulthandler.enable()
    from PySide6.QtGui import QGuiApplication
    from fdm.ui.canvas_overlay_cache import _AreaCommandCentroidCache

    _application = QGuiApplication.instance() or QGuiApplication(
        ["fdm-overlay-worker", "-platform", "offscreen"]
    )
    _centroids = _AreaCommandCentroidCache()


def _executor():
    global _pool
    with _pool_lock:
        if _pool is None:
            _pool = ProcessPoolExecutor(
                max_workers=1,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_initialize_worker,
            )
        return _pool


def shutdown_overlay_renderer():
    global _pool
    with _pool_lock:
        pool, _pool = _pool, None
    if pool is not None:
        pool.shutdown(wait=False, cancel_futures=True)


def _image_bytes(image):
    if image is None:
        return None
    return (
        image.width(),
        image.height(),
        image.bytesPerLine(),
        image.format().value,
        image.devicePixelRatio(),
        bytes(image.constBits()),
    )


def _image_from_bytes(value):
    if value is None:
        return None
    width, height, stride, image_format, dpr, data = value
    image = QImage(data, width, height, stride, QImage.Format(image_format)).copy()
    image.setDevicePixelRatio(dpr)
    return image


def _path_bytes(path):
    if path is None:
        return None
    buffer = QByteArray()
    stream = QDataStream(buffer, QIODevice.OpenModeFlag.WriteOnly)
    stream << path
    return bytes(buffer)


def _path_from_bytes(data):
    if data is None:
        return None
    # QDataStream borrows this buffer. Keep it alive until decoding completes.
    buffer = QByteArray(data)
    stream = QDataStream(buffer, QIODevice.OpenModeFlag.ReadOnly)
    path = QPainterPath()
    stream >> path
    return path


def _picture_from_bytes(data):
    if data is None:
        return None
    picture = QPicture()
    if data:
        picture.setData(data)
    return picture


def _encode(snapshot):
    from fdm.ui.canvas_overlay_cache import PictureOverlayDrawCommand

    commands = []
    for command in snapshot.area_commands:
        if isinstance(command, PictureOverlayDrawCommand):
            commands.append(("picture", bytes(command.picture.data() or b"")))
            continue
        label = command.label
        label_data = (
            None
            if label is None
            else (
                _image_bytes(label.image),
                None if label.top_left is None else (label.top_left.x(), label.top_left.y()),
                (label.center_offset.x(), label.center_offset.y()),
                label.centroid_key,
            )
        )
        transform = command.image_to_overlay
        commands.append(
            (
                "area",
                _path_bytes(command.path),
                command.raw_coordinates,
                command.geometry_key,
                (
                    transform.m11(),
                    transform.m12(),
                    transform.m21(),
                    transform.m22(),
                    transform.dx(),
                    transform.dy(),
                ),
                command.fill_rgba,
                command.outline_rgba,
                command.outline_width,
                command.stroke_rgba,
                command.stroke_width,
                label_data,
                command.stroke_style,
                command.separate_fill,
            )
        )
    return dict(
        key=asdict(snapshot.key),
        request_id=snapshot.request_id,
        picture=None if snapshot.picture is None else bytes(snapshot.picture.data() or b""),
        commands=commands,
        logical_tile_size=snapshot.logical_tile_size,
        bleed_device_pixels=snapshot.bleed_device_pixels,
        exact_composition=snapshot.exact_composition,
        adaptive_composition=snapshot.adaptive_composition,
        composition_probe_rgba=snapshot.composition_probe_rgba,
        known_empty=snapshot.known_empty,
    )


def _render(payload):
    import queue
    from fdm.ui.canvas_overlay_cache import (
        AreaOverlayDrawCommand,
        AreaOverlayLabelCommand,
        CanvasOverlayRenderSnapshot,
        CanvasOverlayTileKey,
        PictureOverlayDrawCommand,
        _CancellationFlag,
        _TileRenderRunnable,
    )

    commands = []
    for value in payload.pop("commands"):
        if value[0] == "picture":
            commands.append(PictureOverlayDrawCommand(_picture_from_bytes(value[1])))
            continue
        (
            _,
            path,
            rings,
            geometry_key,
            transform,
            fill,
            outline,
            outline_width,
            stroke,
            stroke_width,
            label,
            stroke_style,
            separate_fill,
        ) = value
        commands.append(
            AreaOverlayDrawCommand(
                path=_path_from_bytes(path),
                raw_coordinates=rings,
                geometry_key=geometry_key,
                stroke_style=stroke_style,
                separate_fill=separate_fill,
                image_to_overlay=QTransform(*transform),
                fill_rgba=fill,
                outline_rgba=outline,
                outline_width=outline_width,
                stroke_rgba=stroke,
                stroke_width=stroke_width,
                label=None
                if label is None
                else AreaOverlayLabelCommand(
                    image=_image_from_bytes(label[0]),
                    top_left=None if label[1] is None else QPointF(*label[1]),
                    center_offset=QPointF(*label[2]),
                    centroid_key=label[3],
                ),
            )
        )
    payload["picture"] = _picture_from_bytes(payload["picture"])
    payload["key"] = CanvasOverlayTileKey(**payload["key"])
    snapshot = CanvasOverlayRenderSnapshot(area_commands=tuple(commands), **payload)
    worker = _TileRenderRunnable(snapshot, _CancellationFlag(), 0, queue.SimpleQueue(), _centroids)
    image, picture = worker._render()
    return _image_bytes(image), None if picture is None else bytes(picture.data() or b"")


def render_in_isolated_worker(snapshot):
    encoded = _encode(snapshot)
    for attempt in range(2):
        pool = _executor()
        try:
            image, picture = pool.submit(_render, encoded).result()
            return _image_from_bytes(image), _picture_from_bytes(picture)
        except BrokenProcessPool:
            _retire_broken_pool(pool)
            if attempt:
                raise


def _retire_broken_pool(pool):
    global _pool
    with _pool_lock:
        if _pool is pool:
            _pool = None
    pool.shutdown(wait=False, cancel_futures=True)


def _discard_worker_document(token):
    from fdm.ui.canvas_overlay_cache import _worker_paths

    _worker_paths.discard_document(token)
    _centroids.discard_document(token)


def discard_document(token):
    with _pool_lock:
        pool = _pool
    if pool is not None:
        try:
            pool.submit(_discard_worker_document, token)
        except (BrokenProcessPool, RuntimeError):
            # Closing an image must also work after an OS worker termination
            # or during application shutdown. A dead worker owns no live cache.
            _retire_broken_pool(pool)
