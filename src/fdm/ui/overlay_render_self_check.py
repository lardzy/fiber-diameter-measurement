"""Bounded, model-free probe of the packaged Qt raster process.

Use the production initializer, snapshot codec and renderer in a real spawn
pool. This pool belongs only to the probe and uses synthetic geometry without
binding a user's canvas or changing the workspace's render pool.
"""

import hashlib
import multiprocessing
import queue
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import replace

_application = None


def _initialize_windowed_worker():
    # Exercise the installed windowed failure mode even in a console/debug
    # build. Setting streams after initialization would miss this regression.
    for name in ("stdin", "stdout", "stderr", "__stdin__", "__stdout__", "__stderr__"):
        setattr(sys, name, None)
    from fdm.ui.overlay_process_renderer import _initialize_worker

    _initialize_worker()


def _worker_environment():
    import os

    from PySide6.QtGui import QGuiApplication

    return {
        "pid": os.getpid(),
        "stdio_none": all(
            getattr(sys, name) is None for name in ("stdin", "stdout", "stderr")
        ),
        "platform": QGuiApplication.platformName(),
    }


def _snapshots(dpr):
    import numpy as np
    from PySide6.QtCore import QPointF, Qt
    from PySide6.QtGui import (
        QColor,
        QFont,
        QImage,
        QPainter,
        QPainterPath,
        QPen,
        QPicture,
        QPolygonF,
        QTransform,
    )

    from fdm.ui.canvas_overlay_cache import (
        AreaOverlayDrawCommand,
        AreaOverlayLabelCommand,
        CanvasOverlayRenderSnapshot,
        CanvasOverlayTileKey,
        PictureOverlayDrawCommand,
    )

    def coordinates(points):
        return np.asarray(points, dtype=np.float64).tobytes()

    primary = AreaOverlayDrawCommand(
        path=None,
        raw_coordinates=(
            coordinates([(8010, 5010), (8090, 5010), (8090, 5090), (8010, 5090)]),
            coordinates([(8035, 5035), (8065, 5035), (8065, 5065), (8035, 5065)]),
        ),
        geometry_key=("overlay-self-check", "primary"),
        image_to_overlay=QTransform.fromTranslate(-8000, -5000),
        fill_rgba=QColor(52, 211, 153, 72).rgba(),
        outline_rgba=QColor("#0B0B0B").rgba(),
        outline_width=3.2,
        stroke_rgba=QColor("#34D399").rgba(),
        stroke_width=1.8,
        stroke_style=Qt.PenStyle.DashLine.value,
        separate_fill=True,
    )
    subtract = replace(
        primary,
        raw_coordinates=(
            coordinates([(8070, 5050), (8085, 5050), (8085, 5080), (8070, 5080)]),
        ),
        geometry_key=("overlay-self-check", "subtract"),
        fill_rgba=QColor(248, 113, 113, 96).rgba(),
        stroke_rgba=QColor("#F87171").rgba(),
    )
    label = QImage(24, 16, QImage.Format.Format_ARGB32_Premultiplied)
    label.fill(0)
    painter = QPainter(label)
    try:
        font = QFont()
        font.setPixelSize(12)
        painter.setFont(font)
        painter.setPen(QColor("white"))
        painter.drawText(1, 13, "12")
    finally:
        painter.end()
    overlap_path = QPainterPath()
    overlap_path.addRect(10, 70, 90, 30)
    overlap = replace(
        primary,
        path=overlap_path,
        raw_coordinates=(),
        geometry_key=(),
        image_to_overlay=QTransform(),
        fill_rgba=QColor(220, 130, 40, 84).rgba(),
        stroke_style=Qt.PenStyle.SolidLine.value,
        separate_fill=False,
        label=AreaOverlayLabelCommand(label, QPointF(96, 104), QPointF()),
    )
    primitive = QPicture()
    painter = QPainter(primitive)
    try:
        painter.setPen(QPen(QColor("#3B82F6"), 2))
        painter.setBrush(QColor("#3B82F6"))
        painter.drawEllipse(QPointF(110, 20), 3, 3)
        painter.drawLine(QPointF(100, 30), QPointF(120, 90))
        painter.drawPolyline(
            QPolygonF([QPointF(100, 10), QPointF(120, 10), QPointF(120, 50)])
        )
        painter.drawText(3, 120, "12 px")
    finally:
        painter.end()
    base = CanvasOverlayRenderSnapshot(
        request_id=1,
        key=CanvasOverlayTileKey(1, "overlay-self-check", 1, dpr, 0, 0, 0, 0, True),
        logical_tile_size=128,
        area_commands=(primary,),
    )
    mixed = replace(
        base, area_commands=(primary, overlap, PictureOverlayDrawCommand(primitive))
    )
    return (
        ("magic_primary", base),
        ("magic_subtract", replace(base, area_commands=(primary, subtract))),
        ("scene_overview", mixed),
        ("mixed_exact", replace(mixed, exact_composition=True)),
        ("empty", replace(base, known_empty=True)),
    )


def _validate_result(name, snapshot, result):
    from PySide6.QtGui import QImage, QPainter

    from fdm.ui import overlay_process_renderer as renderer
    from fdm.ui.canvas_overlay_cache import (
        _AreaCommandCentroidCache,
        _CancellationFlag,
        _TileRenderRunnable,
    )

    image = renderer._image_from_bytes(result[0])
    picture = renderer._picture_from_bytes(result[1])
    if snapshot.known_empty:
        if image is not None or picture is None or picture.size() != 0:
            raise RuntimeError("empty tile did not return an empty command stream")
        return {"ok": True, "empty": True}
    expected, expected_picture = _TileRenderRunnable(
        snapshot,
        _CancellationFlag(),
        0,
        queue.SimpleQueue(),
        _AreaCommandCentroidCache(),
    )._render()
    if image is None or image.isNull() or image != expected:
        raise RuntimeError("raster pixels differ from the direct Qt reference")
    dpr = snapshot.key.device_pixel_ratio

    def pixel(x, y):
        return image.pixelColor(round(x * dpr), round(y * dpr))

    if image.devicePixelRatio() != dpr or image.width() != round(128 * dpr):
        raise RuntimeError("raster size or device pixel ratio is incorrect")
    if (
        pixel(20, 20).alpha() == 0
        or pixel(50, 50).alpha() != 0
        or pixel(5, 5).alpha() != 0
    ):
        raise RuntimeError(
            "primary fill, hole or global slide coordinates are incorrect"
        )
    if name == "magic_subtract" and pixel(75, 60).red() <= pixel(75, 60).green():
        raise RuntimeError("subtract preview is missing or has the wrong layer order")
    if name in {"scene_overview", "mixed_exact"}:
        if pixel(110, 20).alpha() == 0 or pixel(110, 60).alpha() == 0:
            raise RuntimeError("mixed point or diameter line is missing")
        if not any(
            pixel(x, y).alpha() for x in range(98, 118) for y in range(106, 119)
        ):
            raise RuntimeError("measurement label is missing")
    if (picture is None) != (expected_picture is None):
        raise RuntimeError("exact composition command stream is missing")
    if picture is not None:

        def replay(command):
            target = QImage(image.size(), image.format())
            target.setDevicePixelRatio(dpr)
            target.fill(0xFF58626D)
            painter = QPainter(target)
            try:
                command.play(painter)
            finally:
                painter.end()
            return target

        if replay(picture) != replay(expected_picture):
            raise RuntimeError("exact composition replay pixels differ")
    return {
        "ok": True,
        "width": image.width(),
        "height": image.height(),
        "dpr": dpr,
        "sha256": hashlib.sha256(bytes(image.constBits())).hexdigest(),
        "exact_picture": picture is not None,
    }


def _stop_probe_pool(pool):
    # Python 3.11-3.13 have no public terminate_workers(). Capture only this
    # probe's owned processes before shutdown clears the executor's registry.
    processes = tuple((pool._processes or {}).values())
    pool.shutdown(wait=False, cancel_futures=True)
    for process in processes:
        process.join(timeout=0.5)
        if process.is_alive():
            process.terminate()
            process.join(timeout=1)
        if process.is_alive():
            process.kill()
            process.join(timeout=1)
        if process.is_alive():
            raise RuntimeError("overlay probe worker did not stop")


def run_overlay_render_self_check(*, timeout_seconds=30.0):
    global _application
    from PySide6.QtGui import QGuiApplication

    from fdm.ui import overlay_process_renderer as renderer

    _application = QGuiApplication.instance() or QGuiApplication(
        ["fdm-overlay-self-check", "-platform", "offscreen"]
    )
    if not isinstance(_application, QGuiApplication):
        raise TypeError("overlay probe requires a Qt GUI application")
    started = time.monotonic()
    deadline = started + timeout_seconds
    pool = ProcessPoolExecutor(
        max_workers=1,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_initialize_windowed_worker,
    )
    stage = "worker startup"

    def receive(future):
        return future.result(timeout=max(0, deadline - time.monotonic()))

    try:
        environment = receive(pool.submit(_worker_environment))
        if not environment["stdio_none"] or environment["platform"] != "offscreen":
            raise RuntimeError(
                "probe worker did not start in the windowed offscreen environment"
            )
        cases = {}
        for dpr in (1.0, 1.5, 2.0):
            for name, snapshot in _snapshots(dpr):
                stage = f"{name}@{dpr:g}"
                # The same detached codec and raster entry point as the
                # workspace renderer, with an isolated, bounded probe pool.
                result = receive(
                    pool.submit(renderer._render, renderer._encode(snapshot))
                )
                cases[stage] = _validate_result(name, snapshot, result)
        return {
            "ok": True,
            "worker_stdio_none": True,
            "worker_platform": environment["platform"],
            "worker_pid": environment["pid"],
            "start_method": "spawn",
            "cases": cases,
            "elapsed_ms": round((time.monotonic() - started) * 1000, 2),
        }
    except FutureTimeoutError as exc:
        raise RuntimeError(
            f"{stage}: overlay renderer timed out after {timeout_seconds:g} seconds"
        ) from exc
    except Exception as exc:
        raise RuntimeError(f"{stage}: {exc}") from exc
    finally:
        _stop_probe_pool(pool)
