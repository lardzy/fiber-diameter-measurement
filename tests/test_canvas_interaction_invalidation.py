from __future__ import annotations

import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PySide6.QtCore import QPointF, QRect, QRectF, Qt
from PySide6.QtGui import QColor, QHideEvent, QImage, QPainter, QRegion
from PySide6.QtWidgets import QApplication

from fdm.geometry import Line, Point
from fdm.models import ImageDocument, Measurement
from fdm.settings import (
    AppSettings,
    MagicSegmentToolMode,
    MeasurementLabelStyleSettings,
)
from fdm.ui.canvas import (
    DocumentCanvas,
    MagicSegmentOperationMode,
    MagicSegmentSubtractInputMode,
    canvas_workspace_background,
)
from fdm.ui.digital_slide_canvas import DigitalSlideCanvas
from fdm.ui.rendering import measurement_label_image_bounds


class _PointerEvent:
    def __init__(
        self,
        position: QPointF,
        *,
        button: Qt.MouseButton = Qt.MouseButton.LeftButton,
    ) -> None:
        self._position = position
        self._button = button

    def position(self) -> QPointF:
        return self._position

    @staticmethod
    def modifiers() -> Qt.KeyboardModifiers:
        return Qt.KeyboardModifier.NoModifier

    def button(self) -> Qt.MouseButton:
        return self._button


def _bounding_update_rect(update_mock) -> QRect:
    """Return the union of requested dirty regions and reject full updates."""

    combined = QRect()
    assert update_mock.call_args_list, "interaction did not request a repaint"
    for call in update_mock.call_args_list:
        assert call.args, "interaction requested an unbounded full-canvas update()"
        dirty = call.args[0]
        if isinstance(dirty, QRegion):
            rect = dirty.boundingRect()
        elif isinstance(dirty, QRectF):
            rect = dirty.toAlignedRect()
        elif isinstance(dirty, QRect):
            rect = dirty
        else:  # pragma: no cover - documents the accepted QWidget overloads
            raise AssertionError(f"unexpected update argument: {type(dirty)!r}")
        assert rect.isValid() and not rect.isEmpty()
        combined = rect if combined.isNull() else combined.united(rect)
    return combined


def _render_freehand_preview(
    canvas: DocumentCanvas,
    points: list[Point],
    *,
    destructive: bool,
) -> QImage:
    target = QImage(
        canvas.width(),
        canvas.height(),
        QImage.Format.Format_ARGB32_Premultiplied,
    )
    target.fill(QColor(0, 0, 0, 0))
    painter = QPainter(target)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    try:
        canvas._draw_pending_path_preview(  # noqa: SLF001
            painter,
            points,
            [],
            destructive_preview=destructive,
        )
    finally:
        painter.end()
    return target


def _image_difference_bounds(before: QImage, after: QImage) -> QRect:
    assert before.size() == after.size()
    left = before.width()
    top = before.height()
    right = -1
    bottom = -1
    for y in range(before.height()):
        for x in range(before.width()):
            if before.pixel(x, y) == after.pixel(x, y):
                continue
            left = min(left, x)
            top = min(top, y)
            right = max(right, x)
            bottom = max(bottom, y)
    if right < left or bottom < top:
        return QRect()
    return QRect(left, top, right - left + 1, bottom - top + 1)


class CanvasInteractionInvalidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    @staticmethod
    def _canvas(
        document: ImageDocument,
        *,
        width: int = 320,
        height: int = 240,
    ) -> DocumentCanvas:
        canvas = DocumentCanvas()
        canvas.resize(width, height)
        image_width, image_height = document.image_size
        image = QImage(
            image_width,
            image_height,
            QImage.Format.Format_RGB32,
        )
        image.fill(0)
        canvas.set_document(document, image)
        return canvas

    @staticmethod
    def _assert_local_update(
        canvas: DocumentCanvas,
        update_mock,
        *image_points: Point,
    ) -> QRect:
        dirty = _bounding_update_rect(update_mock)
        for point in image_points:
            widget_point = canvas.image_to_widget(point).toPoint()
            assert dirty.contains(widget_point), (
                f"dirty region {dirty!r} does not cover {widget_point!r}"
            )
        assert dirty != canvas.rect(), "interaction invalidated the whole canvas"
        assert dirty.width() < canvas.width() or dirty.height() < canvas.height()
        return dirty

    def test_polygon_hover_invalidates_old_and_new_preview_only(self) -> None:
        document = ImageDocument(
            id="doc",
            path="/tmp/doc.png",
            image_size=(320, 240),
        )
        canvas = self._canvas(document)
        previous_hover = Point(82.0, 74.0)
        new_hover = Point(126.0, 96.0)
        last_vertex = Point(58.0, 52.0)
        try:
            canvas._tool_mode = "polygon_area"  # noqa: SLF001
            canvas._drawing_polygon_points = [Point(28.0, 30.0), last_vertex]  # noqa: SLF001
            canvas._area_hover_point = previous_hover  # noqa: SLF001

            with patch.object(canvas, "update") as update:
                canvas.mouseMoveEvent(
                    _PointerEvent(canvas.image_to_widget(new_hover))
                )

            self._assert_local_update(
                canvas,
                update,
                last_vertex,
                previous_hover,
                new_hover,
            )
        finally:
            canvas.clear_document()
            canvas.close()

    def test_area_point_landing_uses_local_dirty_region_for_each_tool(
        self,
    ) -> None:
        cases = (
            ("polygon_area", False),
            ("continuous_manual", False),
            (MagicSegmentToolMode.STANDARD, True),
        )
        for tool_mode, magic_subtract in cases:
            with self.subTest(tool_mode=tool_mode):
                document = ImageDocument(
                    id=f"doc-{tool_mode}",
                    path=f"/tmp/{tool_mode}.png",
                    image_size=(320, 240),
                )
                canvas = self._canvas(document)
                first = Point(54.0, 62.0)
                second = Point(138.0, 104.0)
                try:
                    canvas.set_tool_mode(tool_mode)
                    if magic_subtract:
                        canvas._magic_segment.active_stage = (  # noqa: SLF001
                            MagicSegmentOperationMode.SUBTRACT
                        )
                        canvas.set_magic_subtract_input_mode(
                            MagicSegmentSubtractInputMode.POLYGON
                        )

                    with patch.object(canvas, "update") as update:
                        canvas.mousePressEvent(
                            _PointerEvent(canvas.image_to_widget(first))
                        )
                        canvas.mousePressEvent(
                            _PointerEvent(canvas.image_to_widget(second))
                        )

                    self._assert_local_update(
                        canvas,
                        update,
                        first,
                        second,
                    )
                    self.assertEqual(
                        canvas._drawing_polygon_points,  # noqa: SLF001
                        [first, second],
                    )
                finally:
                    canvas.clear_document()
                    canvas.close()

    def test_freehand_sampling_invalidates_changed_fill_wedge(self) -> None:
        document = ImageDocument(
            id="doc-freehand",
            path="/tmp/freehand.png",
            image_size=(320, 240),
        )
        canvas = self._canvas(document)
        first = Point(40.0, 40.0)
        second = Point(200.0, 40.0)
        third = Point(200.0, 180.0)
        fourth = Point(100.0, 180.0)
        fill_probe = Point(90.0, 100.0)
        try:
            canvas.set_tool_mode("freehand_area")
            canvas.mousePressEvent(
                _PointerEvent(canvas.image_to_widget(first))
            )
            canvas._freehand_last_sample_at -= 1.0  # noqa: SLF001
            canvas.mouseMoveEvent(
                _PointerEvent(canvas.image_to_widget(second))
            )
            canvas._freehand_last_sample_at -= 1.0  # noqa: SLF001
            canvas.mouseMoveEvent(
                _PointerEvent(canvas.image_to_widget(third))
            )
            old_points = list(canvas._drawing_polygon_points)  # noqa: SLF001
            old_previews = {
                destructive: _render_freehand_preview(
                    canvas,
                    old_points,
                    destructive=destructive,
                )
                for destructive in (False, True)
            }

            with patch.object(canvas, "update") as update:
                canvas._freehand_last_sample_at -= 1.0  # noqa: SLF001
                canvas.mouseMoveEvent(
                    _PointerEvent(canvas.image_to_widget(fourth))
                )

            dirty = self._assert_local_update(
                canvas,
                update,
                first,
                third,
                fourth,
                fill_probe,
            )
            self.assertEqual(
                canvas._drawing_polygon_points,  # noqa: SLF001
                [first, second, third, fourth],
            )
            new_points = list(canvas._drawing_polygon_points)  # noqa: SLF001
            probe = canvas.image_to_widget(fill_probe).toPoint()
            for destructive in (False, True):
                with self.subTest(destructive=destructive):
                    new_preview = _render_freehand_preview(
                        canvas,
                        new_points,
                        destructive=destructive,
                    )
                    changed_pixels = _image_difference_bounds(
                        old_previews[destructive],
                        new_preview,
                    )
                    self.assertTrue(changed_pixels.isValid())
                    self.assertTrue(
                        dirty.contains(changed_pixels),
                        (
                            f"dirty region {dirty!r} misses changed preview "
                            f"pixels {changed_pixels!r}"
                        ),
                    )
                    self.assertEqual(
                        old_previews[destructive].pixelColor(probe).alpha(),
                        0,
                    )
                    self.assertGreater(
                        new_preview.pixelColor(probe).alpha(),
                        0,
                    )
        finally:
            canvas.clear_document()
            canvas.close()

    def test_magic_freehand_subtract_uses_fill_wedge_invalidation(
        self,
    ) -> None:
        document = ImageDocument(
            id="doc-magic-freehand",
            path="/tmp/magic-freehand.png",
            image_size=(320, 240),
        )
        canvas = self._canvas(document)
        points = (
            Point(40.0, 40.0),
            Point(200.0, 40.0),
            Point(200.0, 180.0),
        )
        fourth = Point(100.0, 180.0)
        fill_probe = Point(90.0, 100.0)
        try:
            canvas.set_tool_mode(MagicSegmentToolMode.STANDARD)
            canvas._magic_segment.active_stage = (  # noqa: SLF001
                MagicSegmentOperationMode.SUBTRACT
            )
            canvas.set_magic_subtract_input_mode(
                MagicSegmentSubtractInputMode.FREEHAND
            )
            canvas.mousePressEvent(
                _PointerEvent(canvas.image_to_widget(points[0]))
            )
            for point in points[1:]:
                canvas._freehand_last_sample_at -= 1.0  # noqa: SLF001
                canvas.mouseMoveEvent(
                    _PointerEvent(canvas.image_to_widget(point))
                )

            with patch.object(canvas, "update") as update:
                canvas._freehand_last_sample_at -= 1.0  # noqa: SLF001
                canvas.mouseMoveEvent(
                    _PointerEvent(canvas.image_to_widget(fourth))
                )

            self._assert_local_update(
                canvas,
                update,
                points[0],
                points[-1],
                fourth,
                fill_probe,
            )
        finally:
            canvas.clear_document()
            canvas.close()

    def test_digital_slide_freehand_invalidation_uses_global_coordinates(
        self,
    ) -> None:
        document = ImageDocument(
            id="doc-slide-freehand",
            path="/tmp/freehand.fdmslide",
            image_size=(4096, 3072),
            document_kind="digital_slide",
        )
        canvas = DigitalSlideCanvas()
        canvas.resize(320, 240)
        image = QImage(320, 240, QImage.Format.Format_RGB32)
        image.fill(0)
        canvas.set_document(document, image)
        canvas._viewport_origin = Point(1000.0, 2000.0)  # noqa: SLF001
        points = (
            Point(1040.0, 2040.0),
            Point(1200.0, 2040.0),
            Point(1200.0, 2180.0),
        )
        fourth = Point(1100.0, 2180.0)
        fill_probe = Point(1090.0, 2100.0)
        try:
            canvas.set_tool_mode("freehand_area")
            canvas.mousePressEvent(
                _PointerEvent(canvas.image_to_widget(points[0]))
            )
            for point in points[1:]:
                canvas._freehand_last_sample_at -= 1.0  # noqa: SLF001
                canvas.mouseMoveEvent(
                    _PointerEvent(canvas.image_to_widget(point))
                )

            with patch.object(canvas, "update") as update:
                canvas._freehand_last_sample_at -= 1.0  # noqa: SLF001
                canvas.mouseMoveEvent(
                    _PointerEvent(canvas.image_to_widget(fourth))
                )

            self._assert_local_update(
                canvas,
                update,
                points[0],
                points[-1],
                fourth,
                fill_probe,
            )
        finally:
            canvas.shutdown()
            canvas.clear_document()
            canvas.close()

    def test_manual_line_start_move_and_commit_use_local_regions(self) -> None:
        document = ImageDocument(
            id="doc-line",
            path="/tmp/line.png",
            image_size=(320, 240),
        )
        canvas = self._canvas(document)
        start = Point(62.0, 74.0)
        end = Point(174.0, 132.0)
        try:
            canvas.set_tool_mode("manual")
            with patch.object(canvas, "update") as update:
                canvas.mousePressEvent(
                    _PointerEvent(canvas.image_to_widget(start))
                )
                canvas.mouseMoveEvent(
                    _PointerEvent(canvas.image_to_widget(end))
                )
                canvas.mouseReleaseEvent(
                    _PointerEvent(canvas.image_to_widget(end))
                )

            self._assert_local_update(
                canvas,
                update,
                start,
                end,
            )
        finally:
            canvas.clear_document()
            canvas.close()

    def test_line_endpoint_continuous_moves_use_local_regions(self) -> None:
        document = ImageDocument(
            id="doc",
            path="/tmp/doc.png",
            image_size=(320, 240),
        )
        measurement = Measurement(
            id="line",
            image_id=document.id,
            fiber_group_id=None,
            mode="manual",
            measurement_kind="line",
            line_px=Line(Point(50.0, 70.0), Point(190.0, 140.0)),
        )
        document.add_measurement(measurement)
        canvas = self._canvas(document)
        original_start = measurement.line_px.start
        first_position = Point(72.0, 84.0)
        second_position = Point(94.0, 98.0)
        fixed_endpoint = measurement.line_px.end
        try:
            canvas._dragging_handle = (measurement.id, "start")  # noqa: SLF001
            canvas._drag_preview_line = measurement.effective_line()  # noqa: SLF001

            with patch.object(canvas, "update") as update:
                canvas.mouseMoveEvent(
                    _PointerEvent(canvas.image_to_widget(first_position))
                )
                first_dirty = self._assert_local_update(
                    canvas,
                    update,
                    original_start,
                    first_position,
                    fixed_endpoint,
                )

                update.reset_mock()
                canvas.mouseMoveEvent(
                    _PointerEvent(canvas.image_to_widget(second_position))
                )
                second_dirty = self._assert_local_update(
                    canvas,
                    update,
                    first_position,
                    second_position,
                    fixed_endpoint,
                )

            self.assertLess(first_dirty.width() * first_dirty.height(), canvas.width() * canvas.height())
            self.assertLess(second_dirty.width() * second_dirty.height(), canvas.width() * canvas.height())
        finally:
            canvas.clear_document()
            canvas.close()

    def test_line_endpoint_hover_invalidates_each_handle_locally(self) -> None:
        document = ImageDocument(
            id="doc-hover",
            path="/tmp/doc-hover.png",
            image_size=(320, 240),
        )
        measurement = Measurement(
            id="line-hover",
            image_id=document.id,
            fiber_group_id=None,
            mode="manual",
            measurement_kind="line",
            line_px=Line(Point(40.0, 70.0), Point(260.0, 70.0)),
        )
        document.add_measurement(measurement)
        canvas = self._canvas(document)
        canvas.set_tool_mode("manual")
        start_position = canvas.image_to_widget(measurement.line_px.start)
        end_position = canvas.image_to_widget(measurement.line_px.end)
        try:
            with patch.object(canvas, "update") as update:
                canvas.mouseMoveEvent(_PointerEvent(start_position))

            self.assertEqual(update.call_count, 1)
            start_dirty = update.call_args.args[0]
            self.assertTrue(start_dirty.contains(start_position.toPoint()))
            self.assertLess(start_dirty.width(), canvas.width() / 2)

            with patch.object(canvas, "update") as unchanged_update:
                canvas.mouseMoveEvent(_PointerEvent(start_position))
            unchanged_update.assert_not_called()

            with patch.object(canvas, "update") as moved_update:
                canvas.mouseMoveEvent(_PointerEvent(end_position))

            self.assertEqual(moved_update.call_count, 2)
            dirty_rects = [call.args[0] for call in moved_update.call_args_list]
            self.assertTrue(any(rect.contains(start_position.toPoint()) for rect in dirty_rects))
            self.assertTrue(any(rect.contains(end_position.toPoint()) for rect in dirty_rects))
            self.assertTrue(all(rect.width() < canvas.width() / 2 for rect in dirty_rects))
        finally:
            canvas.clear_document()
            canvas.close()

    def test_document_geometry_change_clears_cached_line_endpoint_hover(self) -> None:
        document = ImageDocument(
            id="doc-hover-command",
            path="/tmp/doc-hover-command.png",
            image_size=(320, 240),
        )
        measurement = Measurement(
            id="line-hover-command",
            image_id=document.id,
            fiber_group_id=None,
            mode="manual",
            line_px=Line(Point(40.0, 70.0), Point(260.0, 70.0)),
        )
        document.add_measurement(measurement)
        canvas = self._canvas(document)
        canvas.set_selected_measurement(measurement.id)
        canvas.set_tool_mode("select")
        try:
            canvas.mouseMoveEvent(
                _PointerEvent(canvas.image_to_widget(measurement.line_px.start))
            )
            self.assertEqual(
                canvas._hovered_line_endpoint,
                (measurement.id, "start"),
            )
            self.assertEqual(canvas.cursor().shape(), Qt.CursorShape.SizeAllCursor)

            measurement.line_px = Line(
                Point(90.0, 110.0),
                Point(280.0, 110.0),
            )
            canvas.notify_document_visual_changed()

            self.assertIsNone(canvas._hovered_line_endpoint)
            self.assertEqual(canvas.cursor().shape(), Qt.CursorShape.ArrowCursor)
        finally:
            canvas.clear_document()
            canvas.close()

    def test_area_center_drag_continuous_moves_use_local_regions(self) -> None:
        document = ImageDocument(
            id="doc",
            path="/tmp/doc.png",
            image_size=(320, 240),
        )
        outer = [
            Point(40.0, 45.0),
            Point(130.0, 45.0),
            Point(130.0, 125.0),
            Point(40.0, 125.0),
        ]
        measurement = Measurement(
            id="area",
            image_id=document.id,
            fiber_group_id=None,
            mode="polygon_area",
            measurement_kind="area",
            polygon_px=list(outer),
            area_rings_px=[list(outer)],
        )
        document.add_measurement(measurement)
        canvas = self._canvas(document)
        press = Point(85.0, 85.0)
        first_position = Point(101.0, 95.0)
        second_position = Point(119.0, 110.0)
        try:
            canvas._begin_area_drag(  # noqa: SLF001
                (measurement.id, "center", None, None),
                press,
            )

            with patch.object(canvas, "update") as update:
                canvas.mouseMoveEvent(
                    _PointerEvent(canvas.image_to_widget(first_position))
                )
                self._assert_local_update(
                    canvas,
                    update,
                    outer[0],
                    Point(outer[2].x + 16.0, outer[2].y + 10.0),
                )

                update.reset_mock()
                canvas.mouseMoveEvent(
                    _PointerEvent(canvas.image_to_widget(second_position))
                )
                self._assert_local_update(
                    canvas,
                    update,
                    Point(outer[0].x + 16.0, outer[0].y + 10.0),
                    Point(outer[2].x + 34.0, outer[2].y + 25.0),
                )
        finally:
            canvas.clear_document()
            canvas.close()

    def test_area_drag_dirty_bounds_include_result_label_outside_geometry(
        self,
    ) -> None:
        document = ImageDocument(
            id="doc",
            path="/tmp/doc.png",
            image_size=(320, 240),
        )
        outer = [
            Point(60.0, 4.0),
            Point(180.0, 4.0),
            Point(180.0, 16.0),
            Point(60.0, 16.0),
        ]
        measurement = Measurement(
            id="area",
            image_id=document.id,
            fiber_group_id=None,
            mode="polygon_area",
            measurement_kind="area",
            polygon_px=list(outer),
            area_rings_px=[list(outer)],
        )
        document.add_measurement(measurement)
        canvas = self._canvas(document)
        canvas.set_settings(
            AppSettings(
                area_measurement_label_style=MeasurementLabelStyleSettings(
                    enabled=True,
                    font_size=20,
                )
            )
        )
        try:
            label_bounds = measurement_label_image_bounds(
                measurement,
                document,
                canvas._settings,  # noqa: SLF001
                canvas.image_to_widget,
                exact_area=True,
            )
            drag_bounds = canvas._area_drag_display_bounds(  # noqa: SLF001
                measurement,
                Point(0.0, 0.0),
            )
            self.assertIsNotNone(label_bounds)
            self.assertIsNotNone(drag_bounds)
            self.assertLess(label_bounds.top(), min(point.y for point in outer))
            self.assertTrue(
                drag_bounds.image_rect.contains(label_bounds)
            )
        finally:
            canvas.clear_document()
            canvas.close()

    def test_selection_background_patch_clears_label_outside_image_bounds(
        self,
    ) -> None:
        document = ImageDocument(
            id="doc",
            path="/tmp/doc.png",
            image_size=(200, 100),
        )
        measurement = Measurement(
            id="line",
            image_id=document.id,
            fiber_group_id=None,
            mode="manual",
            measurement_kind="line",
            line_px=Line(Point(150.0, 0.0), Point(50.0, 0.0)),
        )
        document.add_measurement(measurement)
        document.select_measurement(measurement.id)
        canvas = DocumentCanvas()
        canvas.resize(300, 200)
        image = QImage(200, 100, QImage.Format.Format_RGB32)
        image.fill(Qt.GlobalColor.white)
        canvas.set_document(document, image)
        canvas._zoom = 1.0  # noqa: SLF001
        canvas._pan = Point(40.0, 40.0)  # noqa: SLF001
        canvas.set_settings(
            AppSettings(
                length_measurement_label_style=MeasurementLabelStyleSettings(
                    enabled=True,
                    font_size=20,
                    background_enabled=True,
                )
            )
        )
        label_bounds = measurement_label_image_bounds(
            measurement,
            document,
            canvas._settings,  # noqa: SLF001
            canvas.image_to_widget,
        )
        self.assertIsNotNone(label_bounds)
        label_center = canvas.image_to_widget(
            Point(label_bounds.center().x(), label_bounds.center().y())
        )
        self.assertLess(label_center.y(), canvas._pan.y)  # noqa: SLF001

        target = QImage(
            canvas.width(),
            canvas.height(),
            QImage.Format.Format_ARGB32_Premultiplied,
        )
        target.fill(QColor("#FF00FF"))
        painter = QPainter(target)
        try:
            with patch.object(
                canvas,
                "_draw_measurements_direct",
                return_value=False,
            ):
                canvas._redraw_selected_measurement_background(  # noqa: SLF001
                    painter,
                    canvas._paint_context(),  # noqa: SLF001
                )
        finally:
            painter.end()
        sample = QColor(
            target.pixel(
                int(round(label_center.x())),
                int(round(label_center.y())),
            )
        )
        self.assertEqual(
            sample.name(),
            canvas_workspace_background(canvas.palette()).name(),
        )
        canvas.clear_document()
        canvas.close()

    def test_visible_tile_change_cancels_stale_active_and_replaces_queue(self) -> None:
        document = ImageDocument(
            id="doc",
            path="/tmp/doc.png",
            image_size=(4096, 1024),
        )
        canvas = self._canvas(document, width=400, height=300)
        try:
            canvas._zoom = 1.0  # noqa: SLF001
            canvas._pan = Point(0.0, 0.0)  # noqa: SLF001
            old_keys = canvas._visible_overlay_tile_keys(  # noqa: SLF001
                canvas._paint_context()  # noqa: SLF001
            )
            self.assertTrue(old_keys)
            stale_active = old_keys[0]
            canvas._overlay_tile_active = stale_active  # noqa: SLF001
            canvas._overlay_tile_queue = list(old_keys[1:])  # noqa: SLF001
            canvas._overlay_tile_queued = set(old_keys[1:])  # noqa: SLF001

            canvas._pan = Point(-2048.0, 0.0)  # noqa: SLF001
            current_keys = canvas._visible_overlay_tile_keys(  # noqa: SLF001
                canvas._paint_context()  # noqa: SLF001
            )
            current_set = set(current_keys)
            self.assertTrue(current_set)
            self.assertNotIn(stale_active, current_set)

            with (
                patch(
                    "fdm.ui.canvas.canvas_overlay_tile_cache.cancel"
                ) as cancel,
                patch(
                    "fdm.ui.canvas.canvas_overlay_tile_cache.contains",
                    return_value=False,
                ),
                patch(
                    "fdm.ui.canvas.canvas_overlay_tile_cache.is_pending",
                    return_value=False,
                ),
                patch("fdm.ui.canvas.QTimer.singleShot"),
            ):
                canvas._enqueue_overlay_tiles(current_keys)  # noqa: SLF001

            cancel.assert_called_once_with(stale_active)
            self.assertIsNone(canvas._overlay_tile_active)  # noqa: SLF001
            self.assertEqual(
                set(canvas._overlay_tile_queue),  # noqa: SLF001
                current_set,
            )
            self.assertEqual(
                canvas._overlay_tile_queued,  # noqa: SLF001
                current_set,
            )
            self.assertFalse(
                canvas._overlay_tile_key_is_current(stale_active)  # noqa: SLF001
            )
        finally:
            canvas.clear_document()
            canvas.close()

    def test_pan_release_requests_one_warmable_frame(self) -> None:
        document = ImageDocument(
            id="doc",
            path="/tmp/doc.png",
            image_size=(1024, 768),
        )
        canvas = self._canvas(document, width=400, height=300)
        try:
            canvas._panning = True  # noqa: SLF001
            canvas._pan_button = Qt.MouseButton.LeftButton  # noqa: SLF001
            canvas._last_mouse_pos = QPointF(120.0, 100.0)  # noqa: SLF001

            with patch.object(canvas, "update") as update:
                canvas.mouseReleaseEvent(
                    _PointerEvent(
                        QPointF(160.0, 120.0),
                        button=Qt.MouseButton.LeftButton,
                    )
                )

            update.assert_called_once_with()
            self.assertFalse(canvas._panning)  # noqa: SLF001
            self.assertIsNone(canvas._pan_button)  # noqa: SLF001
            self.assertIsNone(canvas._pan_drag_unsnapped)  # noqa: SLF001
            self.assertIsNone(canvas._pan_drag_device_phase)  # noqa: SLF001
            self.assertIsNone(  # noqa: SLF001
                canvas._pan_drag_device_pixel_ratio
            )
        finally:
            canvas.clear_document()
            canvas.close()

    def test_continuous_pan_preserves_device_phase_and_raw_accumulator(
        self,
    ) -> None:
        document = ImageDocument(
            id="doc",
            path="/tmp/doc.png",
            image_size=(1024, 768),
        )
        for device_pixel_ratio in (1.0, 1.25, 1.5):
            with self.subTest(device_pixel_ratio=device_pixel_ratio):
                canvas = self._canvas(document, width=400, height=300)
                try:
                    canvas._zoom = 1.0  # noqa: SLF001
                    canvas._pan = Point(11.13, 7.27)  # noqa: SLF001
                    start = Point(canvas._pan.x, canvas._pan.y)  # noqa: SLF001
                    pointer_x = 20.0
                    pointer_y = 20.0
                    canvas._last_mouse_pos = QPointF(  # noqa: SLF001
                        pointer_x,
                        pointer_y,
                    )
                    with patch.object(
                        canvas,
                        "devicePixelRatioF",
                        return_value=device_pixel_ratio,
                    ):
                        initial_key = canvas._overlay_tile_key(  # noqa: SLF001
                            0,
                            0,
                            zoom=1.0,
                            dpr=device_pixel_ratio,
                        )
                        canvas._begin_canvas_pan(  # noqa: SLF001
                            Qt.MouseButton.MiddleButton
                        )
                        total_x = 0.0
                        total_y = 0.0
                        for dx, dy in (
                            (0.2, 0.1),
                            (0.35, -0.25),
                            (0.6, 0.4),
                            (1.0, 0.0),
                        ):
                            total_x += dx
                            total_y += dy
                            pointer_x += dx
                            pointer_y += dy
                            canvas.mouseMoveEvent(
                                _PointerEvent(
                                    QPointF(pointer_x, pointer_y)
                                )
                            )
                            raw = canvas._pan_drag_unsnapped  # noqa: SLF001
                            self.assertAlmostEqual(raw.x, start.x + total_x)
                            self.assertAlmostEqual(raw.y, start.y + total_y)
                            self.assertLessEqual(
                                abs(canvas._pan.x - raw.x),  # noqa: SLF001
                                (0.5 / device_pixel_ratio) + 1e-8,
                            )
                            self.assertLessEqual(
                                abs(canvas._pan.y - raw.y),  # noqa: SLF001
                                (0.5 / device_pixel_ratio) + 1e-8,
                            )
                            current_key = canvas._overlay_tile_key(  # noqa: SLF001
                                0,
                                0,
                                zoom=1.0,
                                dpr=device_pixel_ratio,
                            )
                            self.assertEqual(
                                current_key.device_phase_x,
                                initial_key.device_phase_x,
                            )
                            self.assertEqual(
                                current_key.device_phase_y,
                                initial_key.device_phase_y,
                            )
                finally:
                    canvas.clear_document()
                    canvas.close()

    def test_pan_session_is_cleared_when_canvas_lifecycle_interrupts_drag(
        self,
    ) -> None:
        first = ImageDocument(
            id="first",
            path="/tmp/first.png",
            image_size=(320, 240),
        )
        second = ImageDocument(
            id="second",
            path="/tmp/second.png",
            image_size=(320, 240),
        )
        canvas = self._canvas(first)

        def assert_cleared() -> None:
            self.assertFalse(canvas._panning)  # noqa: SLF001
            self.assertIsNone(canvas._pan_button)  # noqa: SLF001
            self.assertIsNone(canvas._pan_drag_unsnapped)  # noqa: SLF001
            self.assertIsNone(canvas._pan_drag_device_phase)  # noqa: SLF001
            self.assertIsNone(  # noqa: SLF001
                canvas._pan_drag_device_pixel_ratio
            )

        try:
            canvas._begin_canvas_pan(Qt.MouseButton.MiddleButton)  # noqa: SLF001
            canvas.hideEvent(QHideEvent())
            assert_cleared()

            canvas._begin_canvas_pan(Qt.MouseButton.MiddleButton)  # noqa: SLF001
            canvas.set_document(
                second,
                QImage(320, 240, QImage.Format.Format_RGB32),
            )
            assert_cleared()

            canvas._begin_canvas_pan(Qt.MouseButton.MiddleButton)  # noqa: SLF001
            canvas.clear_document()
            assert_cleared()
        finally:
            canvas.close()


if __name__ == "__main__":
    unittest.main()
