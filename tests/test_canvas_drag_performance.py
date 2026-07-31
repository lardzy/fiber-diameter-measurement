from __future__ import annotations

import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QImage, QPainter, QPainterPath
from PySide6.QtWidgets import QApplication

from fdm.area_display import AREA_GEOMETRY_RAW
from fdm.geometry import Line, Point
from fdm.models import ImageDocument, Measurement
from fdm.ui.canvas import DocumentCanvas
from fdm.ui.area_handle_cache import area_handle_display_cache


class _PointerEvent:
    def __init__(self, position: QPointF) -> None:
        self._position = position

    def position(self) -> QPointF:
        return self._position

    @staticmethod
    def modifiers() -> Qt.KeyboardModifiers:
        return Qt.KeyboardModifier.NoModifier

    @staticmethod
    def button() -> Qt.MouseButton:
        return Qt.MouseButton.LeftButton


class CanvasDragPerformanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    @staticmethod
    def _area() -> Measurement:
        outer = [
            Point(20.0, 20.0),
            Point(100.0, 20.0),
            Point(100.0, 100.0),
            Point(20.0, 100.0),
        ]
        hole = [
            Point(45.0, 45.0),
            Point(75.0, 45.0),
            Point(75.0, 75.0),
            Point(45.0, 75.0),
        ]
        return Measurement(
            id="area",
            image_id="doc",
            fiber_group_id=None,
            mode="magic_segment",
            measurement_kind="area",
            polygon_px=list(outer),
            area_rings_px=[list(outer), list(hole)],
            exact_area_px=5500.0,
        )

    def _canvas_with_area(self) -> tuple[DocumentCanvas, ImageDocument, Measurement]:
        document = ImageDocument(id="doc", path="/tmp/doc.png", image_size=(160, 140))
        measurement = self._area()
        document.add_measurement(measurement)
        canvas = DocumentCanvas()
        canvas.resize(160, 140)
        canvas.set_document(document, QImage(160, 140, QImage.Format.Format_RGB32))
        return canvas, document, measurement

    def test_center_drag_mouse_moves_store_only_offset(self) -> None:
        canvas, _document, measurement = self._canvas_with_area()
        original_rings = [[Point(point.x, point.y) for point in ring] for ring in measurement.area_rings_px]
        try:
            with patch.object(canvas, "_clone_magic_rings", side_effect=AssertionError("dense rings cloned")):
                canvas._begin_area_drag(  # noqa: SLF001
                    (measurement.id, "center", None, None),
                    Point(60.0, 60.0),
                )
            with patch("fdm.ui.canvas.polygon_translate", side_effect=AssertionError("translated during move")):
                canvas.mouseMoveEvent(_PointerEvent(canvas.image_to_widget(Point(68.0, 73.0))))

            offset = canvas._drag_area_preview_offset  # noqa: SLF001
            self.assertIsNotNone(offset)
            self.assertAlmostEqual(offset.x, 8.0)
            self.assertAlmostEqual(offset.y, 13.0)
            self.assertIsNone(canvas._drag_area_preview_points)  # noqa: SLF001
            self.assertIsNone(canvas._drag_area_preview_rings)  # noqa: SLF001
            self.assertEqual(measurement.area_rings_px, original_rings)
        finally:
            canvas.clear_document()
            canvas.close()

    def test_center_drag_commits_once_and_preserves_exact_area(self) -> None:
        canvas, _document, measurement = self._canvas_with_area()
        payloads: list[dict[str, object]] = []
        canvas.measurementEdited.connect(
            lambda _document_id, _measurement_id, payload: payloads.append(payload)
        )
        try:
            canvas._begin_area_drag(  # noqa: SLF001
                (measurement.id, "center", None, None),
                Point(60.0, 60.0),
            )
            canvas.mouseMoveEvent(_PointerEvent(canvas.image_to_widget(Point(67.0, 69.0))))
            with patch("fdm.ui.canvas.polygon_translate", wraps=__import__(
                "fdm.ui.canvas", fromlist=["polygon_translate"]
            ).polygon_translate) as translate:
                canvas.mouseReleaseEvent(_PointerEvent(canvas.image_to_widget(Point(67.0, 69.0))))

            self.assertEqual(translate.call_count, len(measurement.area_rings_px))
            self.assertEqual(len(payloads), 1)
            payload = payloads[0]
            self.assertEqual(payload["exact_area_px"], measurement.exact_area_px)
            expected_rings = [
                [Point(point.x + 7.0, point.y + 9.0) for point in ring]
                for ring in measurement.area_rings_px
            ]
            actual_rings = payload["area_rings_px"]
            self.assertEqual(len(actual_rings), len(expected_rings))
            for actual_ring, expected_ring in zip(actual_rings, expected_rings):
                self.assertEqual(len(actual_ring), len(expected_ring))
                for actual, expected in zip(actual_ring, expected_ring):
                    self.assertAlmostEqual(actual.x, expected.x)
                    self.assertAlmostEqual(actual.y, expected.y)
            for actual, expected in zip(payload["polygon_px"], expected_rings[0]):
                self.assertAlmostEqual(actual.x, expected.x)
                self.assertAlmostEqual(actual.y, expected.y)
        finally:
            canvas.clear_document()
            canvas.close()

    def test_active_area_drag_uses_passive_tiles_without_drawing_committed_body(
        self,
    ) -> None:
        canvas, _document, measurement = self._canvas_with_area()
        target = QImage(160, 140, QImage.Format.Format_ARGB32_Premultiplied)
        painter = QPainter(target)
        try:
            canvas._begin_area_drag(  # noqa: SLF001
                (measurement.id, "center", None, None),
                Point(60.0, 60.0),
            )
            with (
                patch.object(canvas, "_overlay_cache_enabled", return_value=True),
                patch.object(canvas, "_draw_measurement_overlay_tiles") as tile_draw,
                patch("fdm.ui.canvas.draw_overlay_annotations"),
            ):
                canvas._draw_annotations(painter, canvas._paint_context())  # noqa: SLF001

            tile_draw.assert_called_once()
            self.assertEqual(
                canvas._actively_edited_measurement_ids(),  # noqa: SLF001
                frozenset((measurement.id,)),
            )
        finally:
            painter.end()
            canvas.clear_document()
            canvas.close()

    def test_selected_area_keeps_passive_body_and_draws_active_emphasis(self) -> None:
        canvas, document, measurement = self._canvas_with_area()
        document.select_measurement(measurement.id)
        target = QImage(160, 140, QImage.Format.Format_ARGB32_Premultiplied)
        painter = QPainter(target)
        try:
            context = canvas._paint_context()  # noqa: SLF001
            with (
                patch("fdm.ui.canvas.canvas_overlay_tile_cache.get", return_value=None),
                patch.object(canvas, "_draw_measurements_direct", return_value=False) as passive,
                patch.object(canvas, "_draw_selected_measurement_active_layer") as active,
                patch.object(canvas, "_enqueue_overlay_tiles"),
            ):
                canvas._draw_measurement_overlay_tiles(painter, context)  # noqa: SLF001

            self.assertFalse(passive.call_args.kwargs["render_selected_state"])
            self.assertNotIn(
                "excluded_measurement_ids",
                passive.call_args.kwargs,
            )
            active.assert_called_once_with(painter, context)
        finally:
            painter.end()
            canvas.clear_document()
            canvas.close()

    def test_selected_area_emphasis_never_covers_passive_fill_or_label(self) -> None:
        canvas, document, measurement = self._canvas_with_area()
        document.select_measurement(measurement.id)
        target = QImage(160, 140, QImage.Format.Format_ARGB32_Premultiplied)
        painter = QPainter(target)
        try:
            with patch("fdm.ui.canvas.draw_area_measurement") as draw_area:
                canvas._draw_selected_measurement_active_layer(  # noqa: SLF001
                    painter,
                    canvas._paint_context(),  # noqa: SLF001
                )

            self.assertFalse(draw_area.call_args.kwargs["show_fill"])
            self.assertFalse(draw_area.call_args.kwargs["show_label"])
            self.assertEqual(
                draw_area.call_args.kwargs["geometry_mode"],
                AREA_GEOMETRY_RAW,
            )
        finally:
            painter.end()
            canvas.clear_document()
            canvas.close()

    def test_dragged_selection_is_not_redrawn_beneath_preview(self) -> None:
        canvas, document, measurement = self._canvas_with_area()
        document.select_measurement(measurement.id)
        canvas._begin_area_drag(  # noqa: SLF001
            (measurement.id, "center", None, None),
            Point(60.0, 60.0),
        )
        target = QImage(160, 140, QImage.Format.Format_ARGB32_Premultiplied)
        painter = QPainter(target)
        try:
            with patch("fdm.ui.canvas.draw_measurements") as vector_draw:
                canvas._draw_selected_measurement_active_layer(  # noqa: SLF001
                    painter,
                    canvas._paint_context(),  # noqa: SLF001
                )
            vector_draw.assert_not_called()
        finally:
            painter.end()
            canvas.clear_document()
            canvas.close()

    def test_line_endpoint_drag_excludes_the_committed_line(self) -> None:
        document = ImageDocument(id="doc", path="/tmp/doc.png", image_size=(160, 140))
        line = Measurement(
            id="line",
            image_id=document.id,
            fiber_group_id=None,
            mode="manual",
            measurement_kind="line",
            line_px=Line(Point(20.0, 20.0), Point(100.0, 100.0)),
        )
        document.add_measurement(line)
        canvas = DocumentCanvas()
        canvas.set_document(document, QImage(160, 140, QImage.Format.Format_RGB32))
        try:
            canvas._dragging_handle = (line.id, "start")  # noqa: SLF001
            canvas._drag_preview_line = line.effective_line()  # noqa: SLF001
            self.assertEqual(
                canvas._actively_edited_measurement_ids(),  # noqa: SLF001
                frozenset((line.id,)),
            )
        finally:
            canvas.clear_document()
            canvas.close()

    def test_hit_query_radius_does_not_scan_all_measurements(self) -> None:
        document = ImageDocument(id="doc", path="/tmp/doc.png", image_size=(100_000, 1000))
        for index in range(500):
            x = 1000.0 + (index * 100.0)
            document.add_measurement(
                Measurement(
                    id=f"far-{index}",
                    image_id=document.id,
                    fiber_group_id=None,
                    mode="manual",
                    measurement_kind="line",
                    line_px=Line(Point(x, 100.0), Point(x + 20.0, 100.0)),
                )
            )
        near = Measurement(
            id="near",
            image_id=document.id,
            fiber_group_id=None,
            mode="manual",
            measurement_kind="line",
            line_px=Line(Point(20.0, 20.0), Point(80.0, 20.0)),
        )
        document.add_measurement(near)
        canvas = DocumentCanvas()
        canvas.set_document(document, QImage(64, 64, QImage.Format.Format_RGB32))
        try:
            original = canvas._measurement_hit_tolerance  # noqa: SLF001
            with patch.object(canvas, "_measurement_hit_tolerance", wraps=original) as tolerance:
                hit = canvas._hit_test_measurement(Point(50.0, 24.0))  # noqa: SLF001

            self.assertEqual(hit, near.id)
            self.assertLess(tolerance.call_count, 10)
        finally:
            canvas.clear_document()
            canvas.close()

    def test_proxy_warm_timer_is_owned_and_stopped_on_reset(self) -> None:
        canvas, _document, _measurement = self._canvas_with_area()
        try:
            self.assertIs(canvas._proxy_warm_timer.parent(), canvas)  # noqa: SLF001
            canvas._proxy_warm_pending = (("key",), QRectF(0, 0, 1, 1), canvas.rect())  # noqa: SLF001
            canvas._proxy_warm_scheduled = True  # noqa: SLF001
            canvas._proxy_warm_timer.start(1000)  # noqa: SLF001

            canvas._reset_proxy_warming()  # noqa: SLF001

            self.assertFalse(canvas._proxy_warm_timer.isActive())  # noqa: SLF001
            self.assertIsNone(canvas._proxy_warm_pending)  # noqa: SLF001
            self.assertFalse(canvas._proxy_warm_scheduled)  # noqa: SLF001
        finally:
            canvas.clear_document()
            canvas.close()

    def test_dense_drag_preview_uses_thinned_cached_handle_set(self) -> None:
        canvas, _document, measurement = self._canvas_with_area()
        dense_ring = [
            Point(20.0 + ((index % 400) * 0.2), 20.0 + ((index // 400) * 0.2))
            for index in range(100_000)
        ]
        measurement.area_rings_px = [dense_ring]
        measurement.polygon_px = list(dense_ring)
        canvas._dragging_area_handle = (measurement.id, "center", None, None)  # noqa: SLF001
        canvas._drag_area_preview_offset = Point(1.0, 1.0)  # noqa: SLF001
        simple_path = QPainterPath()
        simple_path.addRect(QRectF(20.0, 20.0, 80.0, 80.0))
        surface = QImage(160, 140, QImage.Format.Format_ARGB32_Premultiplied)
        surface.fill(0)
        painter = QPainter(surface)
        coordinate_counts: list[int] = []
        original_coordinates = area_handle_display_cache.coordinates

        def capture_coordinates(*args, **kwargs):
            coordinates = original_coordinates(*args, **kwargs)
            coordinate_counts.append(len(coordinates))
            return coordinates

        area_handle_display_cache.clear()
        try:
            with (
                patch.object(
                    area_handle_display_cache,
                    "coordinates",
                    side_effect=capture_coordinates,
                ),
                patch(
                    "fdm.ui.canvas.area_derived_geometry_service.raw_path",
                    return_value=simple_path,
                ),
            ):
                self.assertTrue(canvas._draw_translated_area_drag_preview(painter))  # noqa: SLF001
                self.assertTrue(canvas._draw_translated_area_drag_preview(painter))  # noqa: SLF001

            self.assertEqual(len(coordinate_counts), 2)
            self.assertLess(coordinate_counts[0], 1_000)
            self.assertEqual(coordinate_counts[0], coordinate_counts[1])
            self.assertGreaterEqual(area_handle_display_cache.stats().hits, 1)
        finally:
            painter.end()
            canvas.clear_document()
            canvas.close()


if __name__ == "__main__":
    unittest.main()
