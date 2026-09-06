from __future__ import annotations

import os
from pathlib import Path
import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PySide6.QtCore import QObject, QPointF, QRectF, QThreadPool, Qt
from PySide6.QtGui import (
    QColor,
    QImage,
    QPainter,
    QPainterPath,
    QPicture,
    QPixmap,
    QTransform,
)
from PySide6.QtWidgets import QApplication

from fdm.geometry import Line, Point
from fdm.models import ImageDocument, Measurement
from fdm.settings import AppSettings, MeasurementLabelStyleSettings
import fdm.ui.canvas as canvas_module
from fdm.ui.canvas import DocumentCanvas
from fdm.ui.canvas_overlay_cache import (
    OVERLAY_TILE_MAX_BYTES,
    OVERLAY_TILE_MAX_ENTRIES,
    AreaOverlayDrawCommand,
    AreaOverlayLabelCommand,
    PictureOverlayDrawCommand,
    _worker_paths,
    CanvasOverlayRenderSnapshot,
    CanvasOverlayTileCache,
    CanvasOverlayTileKey,
    _TileRenderRunnable,
)


class _InlineThreadPool:
    def start(self, runnable) -> None:
        runnable.run()


class _DeferredThreadPool:
    def __init__(self) -> None:
        self.runnables = []

    def start(self, runnable) -> None:
        self.runnables.append(runnable)


class _FailingThreadPool:
    def start(self, _runnable) -> None:
        raise RuntimeError("thread pool rejected runnable")


def _key(
    *,
    tile_x: int = 0,
    epoch: int = 0,
    style_generation: int = 3,
    document_token: int = 11,
    zoom: float = 1.0,
    device_pixel_ratio: float = 1.0,
) -> CanvasOverlayTileKey:
    return CanvasOverlayTileKey(
        document_token=document_token,
        document_id="doc",
        zoom=zoom,
        device_pixel_ratio=device_pixel_ratio,
        tile_x=tile_x,
        tile_y=0,
        style_generation=style_generation,
        tile_epoch=epoch,
        show_area_fill=True,
    )


def _picture(color: str = "#FF0000") -> QPicture:
    picture = QPicture()
    painter = QPainter(picture)
    painter.fillRect(QRectF(0.0, 0.0, 64.0, 64.0), QColor(color))
    painter.end()
    return picture


def _area_command(
    rect: QRectF = QRectF(16.0, 16.0, 96.0, 96.0),
    *,
    fill: str = "#FF0000",
    label: AreaOverlayLabelCommand | None = None,
) -> AreaOverlayDrawCommand:
    path = QPainterPath()
    path.setFillRule(Qt.FillRule.OddEvenFill)
    path.addRect(rect)
    fill_color = QColor(fill)
    return AreaOverlayDrawCommand(
        path=path,
        image_to_overlay=QTransform(),
        fill_rgba=int(fill_color.rgba()),
        outline_rgba=int(QColor("#0B0B0B").rgba()),
        outline_width=3.8,
        stroke_rgba=int(fill_color.rgba()),
        stroke_width=2.0,
        label=label,
    )


def _area_measurement(
    measurement_id: str,
    rect: QRectF,
) -> Measurement:
    ring = [
        Point(rect.left(), rect.top()),
        Point(rect.right(), rect.top()),
        Point(rect.right(), rect.bottom()),
        Point(rect.left(), rect.bottom()),
    ]
    return Measurement(
        id=measurement_id,
        image_id="area-doc",
        fiber_group_id=None,
        mode="manual",
        measurement_kind="area",
        polygon_px=list(ring),
        area_rings_px=[list(ring)],
        area_px=float(rect.width() * rect.height()),
    )


class CanvasOverlayTileCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_worker_replays_picture_to_exact_transparent_tile(self) -> None:
        cache = CanvasOverlayTileCache(
            max_entries=4,
            max_bytes=8 * 1024 * 1024,
            thread_pool=_InlineThreadPool(),
        )
        key = _key()
        ready = []
        cache.tileReady.connect(ready.append)

        accepted = cache.request(
            CanvasOverlayRenderSnapshot(
                request_id=1,
                key=key,
                picture=_picture(),
            )
        )

        self.assertTrue(accepted)
        self.assertEqual(ready, [key])
        image = cache.get(key)
        self.assertIsNotNone(image)
        self.assertEqual(image.width(), 512)
        self.assertEqual(image.height(), 512)
        self.assertGreater(image.pixelColor(20, 20).red(), 240)
        self.assertEqual(image.pixelColor(200, 200).alpha(), 0)
        self.assertEqual(cache.stats().completed, 1)

    def test_worker_renders_detached_area_commands_with_odd_even_hole(self) -> None:
        path = QPainterPath()
        path.setFillRule(Qt.FillRule.OddEvenFill)
        path.addRect(QRectF(20.0, 20.0, 180.0, 180.0))
        path.addRect(QRectF(70.0, 70.0, 80.0, 80.0))
        command = _area_command()
        command = AreaOverlayDrawCommand(
            path=path,
            image_to_overlay=command.image_to_overlay,
            fill_rgba=command.fill_rgba,
            outline_rgba=command.outline_rgba,
            outline_width=command.outline_width,
            stroke_rgba=command.stroke_rgba,
            stroke_width=command.stroke_width,
        )
        cache = CanvasOverlayTileCache(
            max_entries=4,
            max_bytes=8 * 1024 * 1024,
            thread_pool=_InlineThreadPool(),
        )

        self.assertTrue(
            cache.request(
                CanvasOverlayRenderSnapshot(
                    request_id=1,
                    key=_key(),
                    area_commands=(command,),
                )
            )
        )

        image = cache.get(_key())
        self.assertIsNotNone(image)
        self.assertGreater(image.pixelColor(40, 40).red(), 240)
        self.assertEqual(image.pixelColor(110, 110).alpha(), 0)

    def test_area_command_request_does_not_clone_or_serialize_qpicture(self) -> None:
        pool = _DeferredThreadPool()
        cache = CanvasOverlayTileCache(thread_pool=pool)
        with patch.object(
            CanvasOverlayTileCache,
            "_clone_picture",
            side_effect=AssertionError("area payload must not serialize QPicture"),
        ):
            self.assertTrue(
                cache.request(
                    CanvasOverlayRenderSnapshot(
                        request_id=1,
                        key=_key(),
                        area_commands=(_area_command(),),
                    )
                )
            )
        self.assertEqual(len(pool.runnables), 1)
        self.assertIsNone(pool.runnables[0]._snapshot.picture)  # noqa: SLF001
        self.assertEqual(
            len(pool.runnables[0]._snapshot.area_commands),  # noqa: SLF001
            1,
        )

    def test_area_worker_checks_cancellation_between_path_passes(self) -> None:
        pool = _DeferredThreadPool()
        cache = CanvasOverlayTileCache(thread_pool=pool)
        command = _area_command()
        cache.request(
            CanvasOverlayRenderSnapshot(
                request_id=1,
                key=_key(),
                area_commands=(command,),
            )
        )
        runnable = pool.runnables[0]

        class CancellingPainter:
            def __init__(self) -> None:
                self.draw_path_calls = 0
                self.restored = False

            def save(self) -> None:
                pass

            def restore(self) -> None:
                self.restored = True

            def setWorldTransform(self, *_args, **_kwargs) -> None:
                pass

            def setBrush(self, _brush) -> None:
                pass

            def setPen(self, _pen) -> None:
                pass

            def drawPath(self, _path) -> None:
                self.draw_path_calls += 1
                runnable._cancellation.cancel()  # noqa: SLF001

        painter = CancellingPainter()
        runnable._draw_area_commands(  # noqa: SLF001
            painter,
            runnable._snapshot.area_commands,  # noqa: SLF001
        )

        self.assertEqual(painter.draw_path_calls, 1)
        self.assertTrue(painter.restored)
        runnable.run()
        cache._drain_completions()  # noqa: SLF001
        self.assertFalse(cache.contains(_key()))

    def test_worker_area_payload_is_detached_from_later_path_mutation(self) -> None:
        pool = _DeferredThreadPool()
        cache = CanvasOverlayTileCache(thread_pool=pool)
        source_path = QPainterPath()
        source_path.addRect(QRectF(20.0, 20.0, 80.0, 80.0))
        base = _area_command()
        command = AreaOverlayDrawCommand(
            path=source_path,
            image_to_overlay=base.image_to_overlay,
            fill_rgba=base.fill_rgba,
            outline_rgba=base.outline_rgba,
            outline_width=base.outline_width,
            stroke_rgba=base.stroke_rgba,
            stroke_width=base.stroke_width,
        )
        cache.request(
            CanvasOverlayRenderSnapshot(
                request_id=1,
                key=_key(),
                area_commands=(command,),
            )
        )

        source_path.addRect(QRectF(180.0, 20.0, 80.0, 80.0))
        pool.runnables[0].run()
        cache._drain_completions()  # noqa: SLF001

        image = cache.get(_key())
        self.assertIsNotNone(image)
        self.assertGreater(image.pixelColor(40, 40).red(), 240)
        self.assertEqual(image.pixelColor(200, 40).alpha(), 0)

    def test_exact_area_fallback_keeps_negative_tile_clip_nonempty(self) -> None:
        cache = CanvasOverlayTileCache(
            max_entries=4,
            max_bytes=8 * 1024 * 1024,
            thread_pool=_InlineThreadPool(),
        )
        key = CanvasOverlayTileKey(
            document_token=11,
            document_id="doc",
            zoom=1.0,
            device_pixel_ratio=1.0,
            tile_x=-1,
            tile_y=-1,
            style_generation=3,
            tile_epoch=0,
            show_area_fill=True,
        )
        command = _area_command(QRectF(-490.0, -490.0, 80.0, 80.0))
        cache.request(
            CanvasOverlayRenderSnapshot(
                request_id=1,
                key=key,
                area_commands=(command,),
                exact_composition=True,
            )
        )
        picture = cache.get_picture(key)
        self.assertIsNotNone(picture)
        surface = QImage(512, 512, QImage.Format.Format_ARGB32_Premultiplied)
        surface.fill(0)
        painter = QPainter(surface)
        painter.translate(512.0, 512.0)
        picture.play(painter)
        painter.end()
        self.assertGreater(surface.pixelColor(40, 40).red(), 240)

    def test_exact_composition_keeps_dual_payload_and_combined_budget(self) -> None:
        cache = CanvasOverlayTileCache(
            max_entries=4,
            max_bytes=8 * 1024 * 1024,
            thread_pool=_InlineThreadPool(),
        )
        key = _key()

        self.assertTrue(
            cache.request(
                CanvasOverlayRenderSnapshot(
                    request_id=1,
                    key=key,
                    area_commands=(_area_command(),),
                    exact_composition=True,
                )
            )
        )

        payload = cache.get_payload(key)
        self.assertIsNotNone(payload)
        image, picture = payload
        self.assertIsNotNone(image)
        self.assertIsNotNone(picture)
        self.assertEqual(
            cache.stats().bytes,
            image.sizeInBytes() + picture.size(),
        )

    def test_known_empty_tile_uses_tiny_picture_sentinel(self) -> None:
        cache = CanvasOverlayTileCache(
            max_entries=4,
            max_bytes=8 * 1024 * 1024,
            thread_pool=_InlineThreadPool(),
        )
        key = _key()

        self.assertTrue(
            cache.request(
                CanvasOverlayRenderSnapshot(
                    request_id=1,
                    key=key,
                    known_empty=True,
                )
            )
        )

        image, picture = cache.get_payload(key)
        self.assertIsNone(image)
        self.assertIsNotNone(picture)
        self.assertLessEqual(cache.stats().bytes, 64)

    def test_guard_admission_cannot_evict_protected_visible_tiles(self) -> None:
        cache = CanvasOverlayTileCache(
            max_entries=2,
            max_bytes=8 * 1024 * 1024,
            thread_pool=_InlineThreadPool(),
        )
        visible = (_key(tile_x=0), _key(tile_x=1))
        guard = _key(tile_x=2)
        for request_id, key in enumerate(visible, start=1):
            self.assertTrue(
                cache.request(
                    CanvasOverlayRenderSnapshot(
                        request_id=request_id,
                        key=key,
                        picture=_picture(),
                    )
                )
            )
        cache.protect(99, visible)

        self.assertTrue(
            cache.request(
                CanvasOverlayRenderSnapshot(
                    request_id=3,
                    key=guard,
                    picture=_picture("#0000FF"),
                )
            )
        )

        self.assertTrue(all(cache.contains(key) for key in visible))
        self.assertFalse(cache.contains(guard))

    def test_completed_picture_budget_uses_size_without_copying_data(self) -> None:
        pool = _DeferredThreadPool()
        cache = CanvasOverlayTileCache(thread_pool=pool)
        key = _key()
        cache.request(
            CanvasOverlayRenderSnapshot(
                request_id=1,
                key=key,
                picture=_picture(),
            )
        )
        pending = cache._pending[key]  # noqa: SLF001
        fake_picture = MagicMock()
        fake_picture.size.return_value = 1234
        fake_picture.data.side_effect = AssertionError(
            "UI completion must not copy QPicture.data()"
        )

        cache._on_completed(  # noqa: SLF001
            key,
            pending.request_sequence,
            None,
            fake_picture,
        )

        self.assertTrue(cache.contains(key))
        self.assertEqual(cache.stats().bytes, 1234)
        fake_picture.size.assert_called_once_with()
        fake_picture.data.assert_not_called()

    def test_hot_get_reuses_completed_tile_without_re_rendering(self) -> None:
        cache = CanvasOverlayTileCache(
            max_entries=4,
            max_bytes=8 * 1024 * 1024,
            thread_pool=_InlineThreadPool(),
        )
        key = _key()
        snapshot = CanvasOverlayRenderSnapshot(
            request_id=1,
            key=key,
            picture=_picture(),
        )
        self.assertTrue(cache.request(snapshot))

        first = cache.get(key)
        second = cache.get(key)

        self.assertIs(first, second)
        self.assertFalse(cache.request(snapshot))
        stats = cache.stats()
        self.assertEqual(stats.completed, 1)
        self.assertEqual(stats.hits, 2)

    def test_namespace_invalidation_removes_only_matching_tiles_and_pending(
        self,
    ) -> None:
        pool = _DeferredThreadPool()
        cache = CanvasOverlayTileCache(thread_pool=pool)
        stale_completed = _key(zoom=0.5, device_pixel_ratio=1.25)
        stale_pending = _key(
            tile_x=1,
            zoom=0.5,
            device_pixel_ratio=1.25,
        )
        other_namespace = _key(
            tile_x=2,
            zoom=1.0,
            device_pixel_ratio=1.25,
        )
        other_document = _key(
            tile_x=3,
            document_token=22,
            zoom=0.5,
            device_pixel_ratio=1.25,
        )

        for request_id, key in enumerate(
            (
                stale_completed,
                stale_pending,
                other_namespace,
                other_document,
            ),
            start=1,
        ):
            self.assertTrue(
                cache.request(
                    CanvasOverlayRenderSnapshot(
                        request_id=request_id,
                        key=key,
                        picture=_picture(),
                    )
                )
            )
        pool.runnables[0].run()
        pool.runnables[2].run()
        pool.runnables[3].run()
        cache._drain_completions()  # noqa: SLF001
        self.assertTrue(cache.contains(stale_completed))
        self.assertTrue(cache.is_pending(stale_pending))

        cache.invalidate_namespace(11, 0.5, 1.25)

        self.assertFalse(cache.contains(stale_completed))
        self.assertFalse(cache.is_pending(stale_pending))
        self.assertTrue(cache.contains(other_namespace))
        self.assertTrue(cache.contains(other_document))

        # A worker already holding the detached stale snapshot may finish, but
        # it no longer owns a pending key and therefore cannot re-enter cache.
        pool.runnables[1].run()
        cache._drain_completions()  # noqa: SLF001
        self.assertFalse(cache.contains(stale_pending))

    def test_request_owns_an_immutable_picture_value_copy(self) -> None:
        pool = _DeferredThreadPool()
        cache = CanvasOverlayTileCache(thread_pool=pool)
        key = _key()
        source_picture = _picture("#FF0000")
        cache.request(
            CanvasOverlayRenderSnapshot(
                request_id=1,
                key=key,
                picture=source_picture,
            )
        )

        source_painter = QPainter(source_picture)
        source_painter.fillRect(QRectF(0.0, 0.0, 64.0, 64.0), QColor("#0000FF"))
        source_painter.end()
        pool.runnables[0].run()
        cache._drain_completions()  # noqa: SLF001

        image = cache.get(key)
        self.assertIsNotNone(image)
        self.assertGreater(image.pixelColor(20, 20).red(), 240)
        self.assertLess(image.pixelColor(20, 20).blue(), 20)

    def test_invalid_exact_scale_snapshot_is_rejected_before_scheduling(self) -> None:
        pool = _DeferredThreadPool()
        cache = CanvasOverlayTileCache(thread_pool=pool)
        invalid_key = CanvasOverlayTileKey(
            document_token=11,
            document_id="doc",
            zoom=float("nan"),
            device_pixel_ratio=1.0,
            tile_x=0,
            tile_y=0,
            style_generation=3,
            tile_epoch=0,
            show_area_fill=True,
        )
        with self.assertRaises(ValueError):
            cache.request(
                CanvasOverlayRenderSnapshot(
                    request_id=1,
                    key=invalid_key,
                    picture=_picture(),
                )
            )
        self.assertEqual(pool.runnables, [])
        self.assertEqual(cache.stats().pending, 0)

    def test_thread_pool_start_failure_rolls_back_pending_state(self) -> None:
        cache = CanvasOverlayTileCache(thread_pool=_FailingThreadPool())
        with self.assertRaisesRegex(RuntimeError, "rejected"):
            cache.request(
                CanvasOverlayRenderSnapshot(
                    request_id=1,
                    key=_key(),
                    picture=_picture(),
                )
            )
        self.assertEqual(cache.stats().pending, 0)
        self.assertEqual(cache.stats().pending_bytes, 0)
        self.assertFalse(cache._inflight_sequences)  # noqa: SLF001
        self.assertFalse(cache._completion_timer.isActive())  # noqa: SLF001

    def test_pending_snapshot_estimate_counts_picture_path_and_label_payloads(
        self,
    ) -> None:
        picture = _picture()
        picture_snapshot = CanvasOverlayRenderSnapshot(
            request_id=1,
            key=_key(),
            picture=picture,
        )
        picture_bytes = CanvasOverlayTileCache._estimate_pending_snapshot_bytes(  # noqa: SLF001
            picture_snapshot
        )
        self.assertGreaterEqual(picture_bytes, picture.size())

        label_image = QImage(
            80,
            40,
            QImage.Format.Format_ARGB32_Premultiplied,
        )
        label_image.fill(QColor("#FFFFFF"))
        command_without_label = _area_command()
        command_with_label = _area_command(
            label=AreaOverlayLabelCommand(
                image=label_image,
                top_left=QPointF(20.0, 20.0),
                center_offset=QPointF(),
            )
        )
        base_bytes = CanvasOverlayTileCache._estimate_pending_snapshot_bytes(  # noqa: SLF001
            CanvasOverlayRenderSnapshot(
                request_id=2,
                key=_key(tile_x=1),
                area_commands=(command_without_label,),
            )
        )
        labelled_bytes = CanvasOverlayTileCache._estimate_pending_snapshot_bytes(  # noqa: SLF001
            CanvasOverlayRenderSnapshot(
                request_id=3,
                key=_key(tile_x=2),
                area_commands=(command_with_label,),
            )
        )
        self.assertGreater(base_bytes, command_without_label.path.elementCount())
        self.assertGreaterEqual(
            labelled_bytes - base_bytes,
            label_image.sizeInBytes(),
        )

    def test_pending_snapshot_budget_rejects_before_cloning_or_scheduling(
        self,
    ) -> None:
        pool = _DeferredThreadPool()
        first = CanvasOverlayRenderSnapshot(
            request_id=1,
            key=_key(tile_x=0),
            picture=_picture("#FF0000"),
        )
        second = CanvasOverlayRenderSnapshot(
            request_id=2,
            key=_key(tile_x=1),
            picture=_picture("#0000FF"),
        )
        first_bytes = CanvasOverlayTileCache._estimate_pending_snapshot_bytes(  # noqa: SLF001
            first
        )
        cache = CanvasOverlayTileCache(
            max_pending_bytes=first_bytes,
            thread_pool=pool,
        )

        self.assertTrue(cache.request(first))
        with patch.object(
            CanvasOverlayTileCache,
            "_clone_picture",
            side_effect=AssertionError(
                "over-budget request must be rejected before cloning"
            ),
        ):
            self.assertFalse(cache.request(second))

        self.assertEqual(len(pool.runnables), 1)
        self.assertEqual(cache.stats().pending, 1)
        self.assertEqual(cache.stats().pending_bytes, first_bytes)

    def test_cancelled_worker_keeps_budget_until_completion_is_drained(
        self,
    ) -> None:
        pool = _DeferredThreadPool()
        first = CanvasOverlayRenderSnapshot(
            request_id=1,
            key=_key(tile_x=0),
            area_commands=(_area_command(),),
        )
        second = CanvasOverlayRenderSnapshot(
            request_id=2,
            key=_key(tile_x=1),
            area_commands=(_area_command(),),
        )
        first_bytes = CanvasOverlayTileCache._estimate_pending_snapshot_bytes(  # noqa: SLF001
            first
        )
        cache = CanvasOverlayTileCache(
            max_pending_bytes=first_bytes,
            thread_pool=pool,
        )

        self.assertTrue(cache.request(first))
        cache.cancel(first.key)
        self.assertEqual(cache.stats().pending, 0)
        self.assertEqual(cache.stats().pending_bytes, first_bytes)
        self.assertFalse(cache.request(second))

        pool.runnables[0].run()
        cache._drain_completions()  # noqa: SLF001
        self.assertEqual(cache.stats().pending_bytes, 0)
        self.assertTrue(cache.request(second))

    def test_worker_error_releases_pending_snapshot_budget(self) -> None:
        pool = _DeferredThreadPool()
        snapshot = CanvasOverlayRenderSnapshot(
            request_id=1,
            key=_key(),
            area_commands=(_area_command(),),
        )
        estimated = CanvasOverlayTileCache._estimate_pending_snapshot_bytes(  # noqa: SLF001
            snapshot
        )
        cache = CanvasOverlayTileCache(
            max_pending_bytes=estimated,
            thread_pool=pool,
        )
        failures = []
        cache.tileFailed.connect(lambda key, message: failures.append((key, message)))

        self.assertTrue(cache.request(snapshot))
        with patch.object(
            pool.runnables[0],
            "_render",
            side_effect=RuntimeError("deterministic worker failure"),
        ):
            pool.runnables[0].run()
        cache._drain_completions()  # noqa: SLF001

        self.assertEqual(cache.stats().pending_bytes, 0)
        self.assertEqual(
            failures,
            [(snapshot.key, "deterministic worker failure")],
        )

    def test_cancelled_late_result_is_not_admitted(self) -> None:
        pool = _DeferredThreadPool()
        cache = CanvasOverlayTileCache(thread_pool=pool)
        key = _key()
        self.assertTrue(
            cache.request(
                CanvasOverlayRenderSnapshot(
                    request_id=1,
                    key=key,
                    picture=_picture(),
                )
            )
        )
        cache.cancel(key)
        pool.runnables[0].run()

        self.assertFalse(cache.contains(key))
        self.assertEqual(cache.stats().pending, 0)

    def test_cancel_then_same_key_request_rejects_queued_old_result(self) -> None:
        pool = _DeferredThreadPool()
        cache = CanvasOverlayTileCache(thread_pool=pool)
        key = _key()
        self.assertTrue(
            cache.request(
                CanvasOverlayRenderSnapshot(
                    request_id=1,
                    key=key,
                    picture=_picture("#FF0000"),
                )
            )
        )
        # Complete the old worker without draining its queued result yet.
        pool.runnables[0].run()
        cache.cancel(key)

        self.assertTrue(
            cache.request(
                CanvasOverlayRenderSnapshot(
                    request_id=2,
                    key=key,
                    picture=_picture("#0000FF"),
                )
            )
        )
        # request() drained the old completion. It must not consume the new
        # request's pending identity or install the stale red tile.
        self.assertFalse(cache.contains(key))
        self.assertTrue(cache.is_pending(key))

        pool.runnables[1].run()
        cache._drain_completions()  # noqa: SLF001
        image = cache.get(key)
        self.assertIsNotNone(image)
        self.assertGreater(image.pixelColor(20, 20).blue(), 240)
        self.assertLess(image.pixelColor(20, 20).red(), 20)
        self.assertGreaterEqual(cache.stats().dropped, 1)

    def test_clear_after_worker_completion_prevents_late_admission(self) -> None:
        pool = _DeferredThreadPool()
        cache = CanvasOverlayTileCache(thread_pool=pool)
        key = _key()
        cache.request(
            CanvasOverlayRenderSnapshot(
                request_id=1,
                key=key,
                picture=_picture(),
            )
        )
        pool.runnables[0].run()
        cache.clear()
        cache._drain_completions()  # noqa: SLF001

        self.assertFalse(cache.contains(key))
        self.assertEqual(cache.stats().entries, 0)
        self.assertEqual(cache.stats().bytes, 0)
        self.assertGreaterEqual(cache.stats().dropped, 1)

    def test_entry_and_byte_limits_evict_oldest_tile(self) -> None:
        cache = CanvasOverlayTileCache(
            max_entries=1,
            max_bytes=8 * 1024 * 1024,
            thread_pool=_InlineThreadPool(),
        )
        first = _key(tile_x=0)
        second = _key(tile_x=1)
        for request_id, key in enumerate((first, second), start=1):
            self.assertTrue(
                cache.request(
                    CanvasOverlayRenderSnapshot(
                        request_id=request_id,
                        key=key,
                        picture=_picture(),
                    )
                )
            )

        self.assertFalse(cache.contains(first))
        self.assertTrue(cache.contains(second))
        self.assertEqual(cache.stats().entries, 1)

    def test_default_entry_and_byte_budgets_are_never_exceeded(self) -> None:
        cache = CanvasOverlayTileCache(thread_pool=_InlineThreadPool())
        for request_id in range(OVERLAY_TILE_MAX_ENTRIES + 19):
            key = _key(tile_x=request_id)
            self.assertTrue(
                cache.request(
                    CanvasOverlayRenderSnapshot(
                        request_id=request_id + 1,
                        key=key,
                        picture=_picture(),
                        logical_tile_size=1,
                        bleed_device_pixels=0,
                    )
                )
            )
            stats = cache.stats()
            self.assertLessEqual(stats.entries, OVERLAY_TILE_MAX_ENTRIES)
            self.assertLessEqual(stats.bytes, OVERLAY_TILE_MAX_BYTES)
        self.assertEqual(cache.stats().entries, OVERLAY_TILE_MAX_ENTRIES)

    def test_byte_budget_evicts_even_below_entry_limit(self) -> None:
        cache = CanvasOverlayTileCache(
            max_entries=256,
            max_bytes=48,
            thread_pool=_InlineThreadPool(),
        )
        keys = [_key(tile_x=index) for index in range(4)]
        for request_id, key in enumerate(keys, start=1):
            self.assertTrue(
                cache.request(
                    CanvasOverlayRenderSnapshot(
                        request_id=request_id,
                        key=key,
                        picture=_picture(),
                        logical_tile_size=2,
                        bleed_device_pixels=0,
                    )
                )
            )
            self.assertLessEqual(cache.stats().bytes, 48)
        self.assertFalse(cache.contains(keys[0]))
        self.assertTrue(cache.contains(keys[-1]))

    def test_document_invalidation_cancels_pending_and_removes_tiles(self) -> None:
        pool = _DeferredThreadPool()
        cache = CanvasOverlayTileCache(thread_pool=pool)
        key = _key()
        cache.request(
            CanvasOverlayRenderSnapshot(
                request_id=1,
                key=key,
                picture=_picture(),
            )
        )
        cache.invalidate_document(key.document_token)
        pool.runnables[0].run()

        self.assertFalse(cache.contains(key))
        self.assertEqual(cache.stats().pending, 0)

    def test_coordinate_invalidation_removes_all_generations_at_coordinate_only(
        self,
    ) -> None:
        cache = CanvasOverlayTileCache(
            max_entries=8,
            max_bytes=16 * 1024 * 1024,
            thread_pool=_InlineThreadPool(),
        )
        matching = [
            _key(tile_x=2, epoch=0, style_generation=3),
            _key(tile_x=2, epoch=1, style_generation=4),
        ]
        retained = _key(tile_x=3, epoch=1, style_generation=4)
        for request_id, key in enumerate((*matching, retained), start=1):
            cache.request(
                CanvasOverlayRenderSnapshot(
                    request_id=request_id,
                    key=key,
                    picture=_picture(),
                )
            )

        cache.invalidate_coordinates(11, {(1.0, 1.0, 2, 0)})

        for key in matching:
            self.assertFalse(cache.contains(key))
        self.assertTrue(cache.contains(retained))

    def test_real_qthreadpool_handoff_installs_on_cache_owner_thread(self) -> None:
        pool = QThreadPool()
        pool.setMaxThreadCount(1)
        cache = CanvasOverlayTileCache(
            max_entries=4,
            max_bytes=8 * 1024 * 1024,
            thread_pool=pool,
        )
        key = _key()
        owner_thread = cache.thread()
        ready_threads = []
        cache.tileReady.connect(lambda _key: ready_threads.append(cache.thread()))

        self.assertTrue(
            cache.request(
                CanvasOverlayRenderSnapshot(
                    request_id=1,
                    key=key,
                    picture=_picture(),
                )
            )
        )
        self.assertTrue(pool.waitForDone(5000))
        cache._drain_completions()  # noqa: SLF001

        self.assertTrue(cache.contains(key))
        self.assertEqual(ready_threads, [owner_thread])

    def test_worker_payload_contains_no_qobject_measurement_or_qpixmap(self) -> None:
        pool = _DeferredThreadPool()
        cache = CanvasOverlayTileCache(thread_pool=pool)
        snapshot = CanvasOverlayRenderSnapshot(
            request_id=1,
            key=_key(),
            picture=_picture(),
        )
        cache.request(snapshot)
        runnable = pool.runnables[0]

        self.assertIsInstance(runnable, _TileRenderRunnable)
        self.assertNotIsInstance(runnable, QObject)
        self.assertNotIsInstance(snapshot.picture, QObject)
        self.assertNotIsInstance(snapshot.picture, QPixmap)
        self.assertEqual(
            set(snapshot.__dataclass_fields__),
            {
                "request_id",
                "key",
                "picture",
                "logical_tile_size",
                "bleed_device_pixels",
                "exact_composition",
                "adaptive_composition",
                "composition_probe_rgba",
                "known_empty",
                "area_commands",
            },
        )
        self.assertFalse(
            any(
                value.__class__.__name__ in {"Measurement", "ImageDocument"}
                or isinstance(value, (QObject, QPixmap))
                for value in runnable.__dict__.values()
            )
        )
        cache.cancel(snapshot.key)
        runnable.run()
        cache._drain_completions()  # noqa: SLF001

    def test_canvas_uses_area_commands_for_disjoint_unselected_areas(self) -> None:
        document = ImageDocument(
            id="area-doc",
            path="/tmp/area-doc.png",
            image_size=(512, 512),
            measurements=[
                _area_measurement("area-a", QRectF(40.0, 40.0, 80.0, 80.0)),
                _area_measurement("area-b", QRectF(260.0, 260.0, 80.0, 80.0)),
            ],
        )
        source = QImage(512, 512, QImage.Format.Format_RGB32)
        source.fill(QColor("#FFFFFF"))
        canvas = DocumentCanvas()
        try:
            canvas.resize(512, 512)
            canvas.set_settings(
                AppSettings(
                    area_measurement_label_style=MeasurementLabelStyleSettings(
                        enabled=False,
                    )
                )
            )
            canvas.set_document(document, source)
            canvas._zoom = 1.0  # noqa: SLF001
            canvas._pan = Point(0.0, 0.0)  # noqa: SLF001
            key = next(
                key
                for key in canvas._visible_overlay_tile_keys(  # noqa: SLF001
                    canvas._paint_context()  # noqa: SLF001
                )
                if key.tile_x == 0 and key.tile_y == 0
            )

            with patch.object(
                canvas_module,
                "QPicture",
                side_effect=AssertionError("safe area snapshot must not record QPicture"),
            ):
                snapshot = canvas._build_overlay_tile_snapshot(key)  # noqa: SLF001

            self.assertIsNotNone(snapshot)
            self.assertIsNone(snapshot.picture)
            self.assertEqual(len(snapshot.area_commands), 2)
            self.assertFalse(snapshot.exact_composition)
            self.assertTrue(
                all(
                    command.path is None
                    and command.raw_coordinates
                    and _worker_paths.path(command).fillRule() == Qt.FillRule.OddEvenFill
                    for command in snapshot.area_commands
                )
            )
        finally:
            canvas.clear_document()
            canvas.close()

    def test_disjoint_area_command_pixels_match_direct_raw_render(self) -> None:
        document = ImageDocument(
            id="area-doc",
            path="/tmp/area-doc.png",
            image_size=(512, 512),
            measurements=[
                _area_measurement("area-a", QRectF(40.0, 40.0, 90.0, 80.0)),
                _area_measurement("area-b", QRectF(280.0, 270.0, 100.0, 90.0)),
            ],
        )
        source = QImage(512, 512, QImage.Format.Format_RGB32)
        source.fill(QColor("#FFFFFF"))
        canvas = DocumentCanvas()
        cache = CanvasOverlayTileCache(thread_pool=_InlineThreadPool())
        try:
            canvas.resize(512, 512)
            canvas.set_settings(
                AppSettings(
                    area_measurement_label_style=MeasurementLabelStyleSettings(
                        enabled=False,
                    )
                )
            )
            canvas.set_document(document, source)
            canvas._zoom = 1.0  # noqa: SLF001
            canvas._pan = Point(0.0, 0.0)  # noqa: SLF001
            key = next(
                key
                for key in canvas._visible_overlay_tile_keys(  # noqa: SLF001
                    canvas._paint_context()  # noqa: SLF001
                )
                if key.tile_x == 0 and key.tile_y == 0
            )
            snapshot = canvas._build_overlay_tile_snapshot(key)  # noqa: SLF001
            self.assertIsNotNone(snapshot)

            direct = QImage(
                512,
                512,
                QImage.Format.Format_ARGB32_Premultiplied,
            )
            direct.fill(0)
            direct_painter = QPainter(direct)
            direct_painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            direct_painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
            canvas._draw_measurements_direct(  # noqa: SLF001
                direct_painter,
                image_rect=QRectF(0.0, 0.0, 512.0, 512.0),
                image_to_output=lambda point: QPointF(point.x, point.y),
                use_sprite_cache=True,
            )
            direct_painter.end()

            cache.request(snapshot)
            rendered = cache.get(key)
            self.assertIsNotNone(rendered)
            direct_pixels = np.frombuffer(
                direct.constBits(),
                dtype=np.uint8,
                count=direct.sizeInBytes(),
            ).reshape((direct.height(), direct.bytesPerLine()))
            rendered_pixels = np.frombuffer(
                rendered.constBits(),
                dtype=np.uint8,
                count=rendered.sizeInBytes(),
            ).reshape((rendered.height(), rendered.bytesPerLine()))
            np.testing.assert_array_equal(rendered_pixels, direct_pixels)
        finally:
            canvas.clear_document()
            canvas.close()
            cache.clear()

    def test_area_tiles_use_adaptive_composition_without_bbox_forced_picture(
        self,
    ) -> None:
        cases = (
            (
                "mixed",
                [
                _area_measurement("area-a", QRectF(40.0, 40.0, 100.0, 100.0)),
                Measurement(
                    id="line-a",
                    image_id="area-doc",
                    fiber_group_id=None,
                    mode="manual",
                    measurement_kind="line",
                    line_px=Line(Point(260.0, 260.0), Point(320.0, 320.0)),
                ),
                ],
            ),
            (
                "overlapping-areas",
                [
                _area_measurement("area-a", QRectF(40.0, 40.0, 140.0, 140.0)),
                _area_measurement("area-b", QRectF(100.0, 100.0, 140.0, 140.0)),
                ],
            ),
        )
        for scenario, measurements in cases:
            with self.subTest(scenario=scenario):
                document = ImageDocument(
                    id="area-doc",
                    path="/tmp/area-doc.png",
                    image_size=(512, 512),
                    measurements=measurements,
                )
                source = QImage(512, 512, QImage.Format.Format_RGB32)
                source.fill(QColor("#FFFFFF"))
                canvas = DocumentCanvas()
                try:
                    canvas.resize(512, 512)
                    canvas.set_document(document, source)
                    canvas._zoom = 1.0  # noqa: SLF001
                    canvas._pan = Point(0.0, 0.0)  # noqa: SLF001
                    key = next(
                        key
                        for key in canvas._visible_overlay_tile_keys(  # noqa: SLF001
                            canvas._paint_context()  # noqa: SLF001
                        )
                        if key.tile_x == 0 and key.tile_y == 0
                    )
                    snapshot = canvas._build_overlay_tile_snapshot(key)  # noqa: SLF001
                    self.assertIsNotNone(snapshot)
                    self.assertEqual(snapshot.exact_composition, scenario == "mixed")
                    self.assertEqual(snapshot.adaptive_composition, scenario != "mixed")
                    if scenario == "mixed":
                        self.assertIsNone(snapshot.picture)
                        self.assertTrue(
                            any(
                                isinstance(command, PictureOverlayDrawCommand)
                                for command in snapshot.area_commands
                            )
                        )
                        self.assertTrue(
                            any(
                                isinstance(command, AreaOverlayDrawCommand)
                                for command in snapshot.area_commands
                            )
                        )
                    else:
                        self.assertIsNone(snapshot.picture)
                        self.assertEqual(len(snapshot.area_commands), 2)
                        worker_cache = CanvasOverlayTileCache(
                            thread_pool=_InlineThreadPool()
                        )
                        try:
                            worker_cache.request(snapshot)
                            payload = worker_cache.get_payload(key)
                            self.assertIsNotNone(payload)
                            self.assertIsNotNone(payload[0])
                        finally:
                            worker_cache.clear()
                finally:
                    canvas.clear_document()
                    canvas.close()

    def test_uncached_area_label_centroid_is_derived_in_worker(self) -> None:
        measurement = _area_measurement(
            "area-label",
            QRectF(120.0, 100.0, 160.0, 120.0),
        )
        document = ImageDocument(
            id="area-doc",
            path="/tmp/area-doc.png",
            image_size=(512, 512),
            measurements=[measurement],
        )
        source = QImage(512, 512, QImage.Format.Format_RGB32)
        source.fill(QColor("#FFFFFF"))
        canvas = DocumentCanvas()
        cache = CanvasOverlayTileCache(thread_pool=_InlineThreadPool())
        before = measurement.to_dict()
        try:
            canvas.resize(512, 512)
            canvas.set_settings(
                AppSettings(
                    area_measurement_label_style=MeasurementLabelStyleSettings(
                        enabled=True,
                    )
                )
            )
            canvas.set_document(document, source)
            canvas._zoom = 1.0  # noqa: SLF001
            canvas._pan = Point(0.0, 0.0)  # noqa: SLF001
            key = next(
                key
                for key in canvas._visible_overlay_tile_keys(  # noqa: SLF001
                    canvas._paint_context()  # noqa: SLF001
                )
                if key.tile_x == 0 and key.tile_y == 0
            )
            with patch.object(
                Measurement,
                "geometry_center",
                side_effect=AssertionError("centroid must not run on snapshot UI path"),
            ):
                snapshot = canvas._build_overlay_tile_snapshot(key)  # noqa: SLF001
            self.assertIsNotNone(snapshot)
            self.assertIsNone(snapshot.area_commands[0].label.top_left)

            cache.request(snapshot)

            self.assertTrue(cache.contains(key))
            self.assertEqual(len(cache._area_centroids._entries), 1)  # noqa: SLF001
            self.assertEqual(measurement.to_dict(), before)
        finally:
            canvas.clear_document()
            canvas.close()
            cache.clear()

    def test_replacement_area_same_id_and_revision_has_distinct_centroid_key(
        self,
    ) -> None:
        first = _area_measurement(
            "replacement-area",
            QRectF(20.0, 30.0, 80.0, 60.0),
        )
        replacement = _area_measurement(
            "replacement-area",
            QRectF(220.0, 180.0, 80.0, 60.0),
        )
        document = ImageDocument(
            id="area-doc",
            path="/tmp/area-doc.png",
            image_size=(512, 512),
            measurements=[first],
        )
        settings = AppSettings(
            area_measurement_label_style=MeasurementLabelStyleSettings(
                enabled=True,
            )
        )

        first_command = canvas_module.build_passive_area_overlay_command(
            document,
            first,
            settings,
            zoom=1.0,
            line_width=2.0,
            show_fill=True,
            sprite_device_pixel_ratio=1.0,
        )
        document.measurements = [replacement]
        replacement_command = canvas_module.build_passive_area_overlay_command(
            document,
            replacement,
            settings,
            zoom=1.0,
            line_width=2.0,
            show_fill=True,
            sprite_device_pixel_ratio=1.0,
        )

        self.assertIsNotNone(first_command)
        self.assertIsNotNone(replacement_command)
        assert first_command is not None and replacement_command is not None
        self.assertIsNotNone(first_command.label)
        self.assertIsNotNone(replacement_command.label)
        assert first_command.label is not None
        assert replacement_command.label is not None
        self.assertIsNone(first_command.label.top_left)
        self.assertIsNone(replacement_command.label.top_left)
        self.assertNotEqual(
            first_command.label.centroid_key,
            replacement_command.label.centroid_key,
        )
        self.assertEqual(
            first_command.label.centroid_key,
            (
                id(document),
                id(first),
                first.id,
                first.geometry_revision,
            ),
        )
        self.assertEqual(
            replacement_command.label.centroid_key,
            (
                id(document),
                id(replacement),
                replacement.id,
                replacement.geometry_revision,
            ),
        )

        cache = CanvasOverlayTileCache(thread_pool=_InlineThreadPool())
        try:
            first_centroid = cache._area_centroids.get_or_compute(  # noqa: SLF001
                first_command.label.centroid_key,
                _worker_paths.path(first_command),
            )
            replacement_centroid = cache._area_centroids.get_or_compute(  # noqa: SLF001
                replacement_command.label.centroid_key,
                _worker_paths.path(replacement_command),
            )
            self.assertAlmostEqual(first_centroid.x(), 60.0)
            self.assertAlmostEqual(first_centroid.y(), 60.0)
            self.assertAlmostEqual(replacement_centroid.x(), 260.0)
            self.assertAlmostEqual(replacement_centroid.y(), 210.0)
            self.assertEqual(len(cache._area_centroids._entries), 2)  # noqa: SLF001
        finally:
            cache.clear()

    def test_six_hundred_thousand_element_area_payload_is_implicitly_shared(
        self,
    ) -> None:
        path = QPainterPath()
        path.moveTo(0.0, 0.0)
        for index in range(600_000):
            path.lineTo(
                float(index % 1000),
                float((index // 1000) % 600),
            )
        command = _area_command()
        command = AreaOverlayDrawCommand(
            path=path,
            image_to_overlay=command.image_to_overlay,
            fill_rgba=command.fill_rgba,
            outline_rgba=command.outline_rgba,
            outline_width=command.outline_width,
            stroke_rgba=command.stroke_rgba,
            stroke_width=command.stroke_width,
        )
        pool = _DeferredThreadPool()
        cache = CanvasOverlayTileCache(thread_pool=pool)
        with patch.object(
            CanvasOverlayTileCache,
            "_clone_picture",
            side_effect=AssertionError("large RAW path must not be serialized"),
        ):
            cache.request(
                CanvasOverlayRenderSnapshot(
                    request_id=1,
                    key=_key(),
                    area_commands=(command,),
                )
            )

        worker_path = pool.runnables[0]._snapshot.area_commands[0].path  # noqa: SLF001
        self.assertEqual(worker_path.elementCount(), path.elementCount())
        original_count = worker_path.elementCount()
        path.lineTo(9.0, 9.0)
        self.assertEqual(worker_path.elementCount(), original_count)
        cache.cancel(_key())
        pool.runnables[0].run()
        cache._drain_completions()  # noqa: SLF001

    def test_forced_canvas_pipeline_reuses_warm_tile_without_direct_redraw(
        self,
    ) -> None:
        pool = QThreadPool()
        pool.setMaxThreadCount(1)
        cache = CanvasOverlayTileCache(
            max_entries=8,
            max_bytes=16 * 1024 * 1024,
            thread_pool=pool,
        )
        document = ImageDocument(
            id="dense-document",
            path="/tmp/dense-document.png",
            image_size=(320, 240),
        )
        document.measurements = [
            Measurement(
                id=f"line-{index}",
                image_id=document.id,
                fiber_group_id=None,
                mode="manual",
                measurement_kind="line",
                line_px=Line(
                    Point(10.0 + index, 20.0),
                    Point(20.0 + index, 180.0),
                ),
            )
            for index in range(64)
        ]
        source = QImage(320, 240, QImage.Format.Format_RGB32)
        source.fill(QColor("#FFFFFF"))
        canvas = None
        with (
            patch.object(canvas_module, "canvas_overlay_tile_cache", cache),
            patch.dict(
                os.environ,
                {"FDM_ENABLE_CANVAS_OVERLAY_CACHE": "1"},
                clear=False,
            ),
        ):
            try:
                canvas = DocumentCanvas()
                canvas.resize(320, 240)
                canvas._pan = Point(0.0, 0.0)  # noqa: SLF001
                canvas.set_document(document, source)
                direct_frame = QImage(
                    320,
                    240,
                    QImage.Format.Format_ARGB32_Premultiplied,
                )
                direct_frame.fill(0)
                canvas.render(direct_frame)

                # The single-shot queue starter runs on the UI thread, while
                # rasterization runs in this real QThreadPool.
                for _attempt in range(16):
                    self.app.processEvents()
                    self.assertTrue(pool.waitForDone(5000))
                    cache._drain_completions()  # noqa: SLF001
                    self.app.processEvents()
                    if (
                        cache.stats().pending == 0
                        and canvas._overlay_tile_active is None  # noqa: SLF001
                        and not canvas._overlay_tile_queue  # noqa: SLF001
                        and not canvas._overlay_tile_build_scheduled  # noqa: SLF001
                    ):
                        break
                self.assertGreater(cache.stats().entries, 0)
                self.assertEqual(cache.stats().pending, 0)

                hits_before = cache.stats().hits
                cached_frame = QImage(
                    320,
                    240,
                    QImage.Format.Format_ARGB32_Premultiplied,
                )
                cached_frame.fill(0)
                with patch.object(
                    canvas,
                    "_draw_measurements_direct",
                    wraps=canvas._draw_measurements_direct,  # noqa: SLF001
                ) as direct_draw:
                    canvas.render(cached_frame)
                direct_draw.assert_not_called()
                self.assertGreater(cache.stats().hits, hits_before)
                direct_pixels = np.frombuffer(
                    direct_frame.constBits(),
                    dtype=np.uint8,
                    count=direct_frame.sizeInBytes(),
                ).reshape((direct_frame.height(), direct_frame.bytesPerLine()))
                cached_pixels = np.frombuffer(
                    cached_frame.constBits(),
                    dtype=np.uint8,
                    count=cached_frame.sizeInBytes(),
                ).reshape((cached_frame.height(), cached_frame.bytesPerLine()))
                differing = np.any(
                    direct_pixels.reshape((240, 320, 4))
                    != cached_pixels.reshape((240, 320, 4)),
                    axis=2,
                )
                target = QRectF(
                    canvas._pan.x,  # noqa: SLF001
                    canvas._pan.y,  # noqa: SLF001
                    source.width() * canvas._zoom,  # noqa: SLF001
                    source.height() * canvas._zoom,  # noqa: SLF001
                )
                yy, xx = np.indices(differing.shape)
                border_distance = np.minimum.reduce(
                    (
                        np.abs(xx - target.left()),
                        np.abs(xx - target.right()),
                        np.abs(yy - target.top()),
                        np.abs(yy - target.bottom()),
                    )
                )
                measurement_differences = differing & (border_distance > 2.0)
                # Worker tiles repaint the one-pixel image frame, whose
                # antialias coverage can differ at a fractional pan.  Passive
                # measurement content itself must remain pixel-equivalent.
                self.assertLessEqual(
                    int(np.count_nonzero(measurement_differences)),
                    64,
                )
            finally:
                if canvas is not None:
                    canvas.clear_document()
                    canvas.close()
                cache.clear()
                pool.waitForDone(5000)
                cache._drain_completions()  # noqa: SLF001
                self.app.processEvents()

    def test_large_label_crossing_tile_boundary_is_recorded_in_neighbour_tile(
        self,
    ) -> None:
        document = ImageDocument(
            id="large-label-document",
            path="/tmp/large-label.png",
            image_size=(1024, 300),
        )
        document.measurements = [
            Measurement(
                id="boundary-line",
                image_id=document.id,
                fiber_group_id=None,
                mode="manual",
                measurement_kind="line",
                line_px=Line(Point(460.0, 180.0), Point(480.0, 180.0)),
                diameter_px=123456.789,
                diameter_unit=123456.789,
            )
        ]
        source = QImage(1024, 300, QImage.Format.Format_RGB32)
        source.fill(QColor("#FFFFFF"))
        canvas = DocumentCanvas()
        try:
            canvas.resize(1024, 300)
            canvas.set_settings(
                AppSettings(
                    length_measurement_label_style=MeasurementLabelStyleSettings(
                        enabled=True,
                        font_size=96,
                        decimals=3,
                        background_enabled=True,
                    )
                )
            )
            canvas.set_document(document, source)
            canvas._zoom = 1.0  # noqa: SLF001
            canvas._pan = Point(0.0, 0.0)  # noqa: SLF001
            neighbour_key = next(
                key
                for key in canvas._visible_overlay_tile_keys(  # noqa: SLF001
                    canvas._paint_context()  # noqa: SLF001
                )
                if key.tile_x == 1 and key.tile_y == 0
            )

            snapshot = canvas._build_overlay_tile_snapshot(  # noqa: SLF001
                neighbour_key
            )
            self.assertIsNotNone(snapshot)
            surface = QImage(
                1024,
                300,
                QImage.Format.Format_ARGB32_Premultiplied,
            )
            surface.fill(0)
            painter = QPainter(surface)
            try:
                for command in snapshot.area_commands:
                    command.picture.play(painter)
            finally:
                painter.end()
            pixels = np.frombuffer(
                surface.constBits(),
                dtype=np.uint8,
                count=surface.sizeInBytes(),
            ).reshape((surface.height(), surface.bytesPerLine()))
            alpha = pixels[:, : surface.width() * 4].reshape(
                (surface.height(), surface.width(), 4)
            )[:, :, 3]
            self.assertGreater(int(np.count_nonzero(alpha[:, 512:])), 100)
        finally:
            canvas.clear_document()
            canvas.close()


if __name__ == "__main__":
    unittest.main()
