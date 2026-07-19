from __future__ import annotations

import os
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PySide6.QtCore import QPointF, QRectF
from PySide6.QtGui import QColor, QImage, QPainter, QPicture
from PySide6.QtWidgets import QApplication

from fdm.geometry import Line, Point
from fdm.area_display import (
    AREA_GEOMETRY_RAW,
    AREA_GEOMETRY_SCREEN,
    AreaProxyBuildBudget,
    area_derived_geometry_service,
)
from fdm.models import ImageDocument, Measurement, ObjectAppearanceOverride
from fdm.settings import (
    AppSettings,
    MeasurementEndpointStyle,
    MeasurementLabelStyleSettings,
)
from fdm.ui import rendering
import fdm.ui.canvas as canvas_module
from fdm.ui.canvas import (
    CanvasDisplayBounds,
    CanvasVisualChange,
    DocumentCanvas,
    MeasurementSceneIndex,
)
from fdm.ui.canvas_overlay_cache import (
    CanvasOverlayRenderSnapshot,
    CanvasOverlayTileCache,
    canvas_overlay_tile_cache,
)
from fdm.ui.digital_slide_canvas import DigitalSlideCanvas


class CanvasRenderPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    @staticmethod
    def _line(identifier: str, x: float) -> Measurement:
        return Measurement(
            id=identifier,
            image_id="doc",
            fiber_group_id=None,
            mode="manual",
            measurement_kind="line",
            line_px=Line(Point(x, 10.0), Point(x + 10.0, 20.0)),
        )

    @staticmethod
    def _alpha_bounds(image: QImage) -> QRectF:
        pixels = np.frombuffer(
            image.constBits(),
            dtype=np.uint8,
            count=image.sizeInBytes(),
        ).reshape((image.height(), image.bytesPerLine()))
        alpha = pixels[:, : image.width() * 4].reshape(
            (image.height(), image.width(), 4)
        )[:, :, 3]
        ys, xs = np.nonzero(alpha)
        if not len(xs):
            return QRectF()
        return QRectF(
            float(xs.min()),
            float(ys.min()),
            float(xs.max() - xs.min() + 1),
            float(ys.max() - ys.min() + 1),
        )

    def test_scene_rect_query_preserves_document_order_and_count_numbers(self) -> None:
        first = self._line("first", 10.0)
        count = Measurement(
            id="count",
            image_id="doc",
            fiber_group_id=None,
            mode="count",
            measurement_kind="count",
            point_px=Point(30.0, 30.0),
        )
        last = self._line("last", 40.0)
        index = MeasurementSceneIndex([first, count, last])

        visible = index.query_rect(QRectF(0.0, 0.0, 100.0, 100.0))

        self.assertEqual([measurement.id for measurement in visible], ["first", "count", "last"])
        self.assertEqual(index.count_number("count"), 1)
        self.assertEqual(
            [measurement.id for measurement in index.query_point(Point(45.0, 15.0), tolerance=10.0)],
            ["last"],
        )

    def test_high_vertex_magic_area_enables_cache_below_object_threshold(
        self,
    ) -> None:
        ring = [
            Point(
                160.0 + (100.0 * math.cos(index * math.tau / 10_000)),
                120.0 + (80.0 * math.sin(index * math.tau / 10_000)),
            )
            for index in range(10_000)
        ]
        measurement = Measurement(
            id="dense-magic",
            image_id="doc",
            fiber_group_id=None,
            mode="magic_segment",
            measurement_kind="area",
            polygon_px=ring[:256],
            area_rings_px=[ring],
        )
        document = ImageDocument(
            id="doc",
            path="/tmp/doc.png",
            image_size=(320, 240),
            measurements=[measurement],
        )
        canvas = DocumentCanvas()
        try:
            canvas.set_document(
                document,
                QImage(320, 240, QImage.Format.Format_RGB32),
            )
            canvas._sync_overlay_visual_state()  # noqa: SLF001
            with patch.dict(
                os.environ,
                {"QT_QPA_PLATFORM": ""},
                clear=False,
            ):
                self.assertTrue(canvas._overlay_cache_enabled())  # noqa: SLF001
        finally:
            canvas.clear_document()
            canvas.close()

    def test_oversized_area_uses_bucket_and_remains_queryable(self) -> None:
        area = Measurement(
            id="large",
            image_id="doc",
            fiber_group_id=None,
            mode="polygon_area",
            measurement_kind="area",
            polygon_px=[
                Point(0.0, 0.0),
                Point(100_000.0, 0.0),
                Point(100_000.0, 100_000.0),
                Point(0.0, 100_000.0),
            ],
        )
        index = MeasurementSceneIndex([area], cell_size=64.0)

        self.assertEqual(
            [measurement.id for measurement in index.query_rect(QRectF(50_000, 50_000, 10, 10))],
            ["large"],
        )
        self.assertLessEqual(len(index._cells), index._MAX_ENTRY_CELLS)  # noqa: SLF001

    def test_area_index_bounds_do_not_construct_qpainter_path(self) -> None:
        area = Measurement(
            id="area",
            image_id="doc",
            fiber_group_id=None,
            mode="polygon_area",
            measurement_kind="area",
            polygon_px=[
                Point(5.0, 5.0),
                Point(50.0, 5.0),
                Point(50.0, 50.0),
                Point(5.0, 50.0),
            ],
        )
        with patch("fdm.area_display.AreaDerivedGeometryService.raw_path", side_effect=AssertionError):
            index = MeasurementSceneIndex([area])
            visible = index.query_rect(QRectF(0.0, 0.0, 100.0, 100.0))
        self.assertEqual(visible, [area])

    def test_deferred_proxy_does_not_call_update_inside_paint(self) -> None:
        document = ImageDocument(id="doc", path="/tmp/doc.png", image_size=(160, 120))
        document.measurements = [self._line("line", 10.0)]
        canvas = DocumentCanvas()
        canvas.resize(160, 120)
        canvas.set_document(
            document,
            QImage(160, 120, QImage.Format.Format_RGB32),
        )
        target = QImage(160, 120, QImage.Format.Format_ARGB32_Premultiplied)
        painter = QPainter(target)
        try:
            with (
                patch("fdm.ui.canvas.draw_measurements", return_value=True),
                patch("fdm.ui.canvas.draw_overlay_annotations"),
                patch.object(canvas, "update") as update,
            ):
                canvas._draw_annotations(painter, canvas._paint_context())  # noqa: SLF001
                update.assert_not_called()
        finally:
            painter.end()
            canvas._reset_proxy_warming()  # noqa: SLF001
            canvas.clear_document()
            canvas.close()

    def test_cold_deferred_area_proxy_still_draws_and_caches_label_centroid(
        self,
    ) -> None:
        area_derived_geometry_service.clear()
        points = [
            Point(
                80.0 + (60.0 * math.cos(2.0 * math.pi * index / 1000)),
                70.0 + (45.0 * math.sin(2.0 * math.pi * index / 1000)),
            )
            for index in range(1000)
        ]
        measurement = Measurement(
            id="dense-area",
            image_id="doc",
            fiber_group_id=None,
            mode="polygon_area",
            measurement_kind="area",
            polygon_px=points,
            area_rings_px=[points],
        )
        document = ImageDocument(
            id="doc",
            path="/tmp/doc.png",
            image_size=(160, 140),
            measurements=[measurement],
        )
        settings = AppSettings(
            area_measurement_label_style=MeasurementLabelStyleSettings(
                enabled=True,
            )
        )
        target = QImage(160, 140, QImage.Format.Format_ARGB32_Premultiplied)
        target.fill(0)
        painter = QPainter(target)
        budget = AreaProxyBuildBudget(max_builds=0, max_build_ms=0.0)
        try:
            with (
                patch.object(
                    area_derived_geometry_service,
                    "centroid",
                    wraps=area_derived_geometry_service.centroid,
                ) as centroid,
                patch.object(rendering, "draw_area_measurement_label") as label,
            ):
                for _ in range(2):
                    rendering.draw_area_measurement(
                        painter,
                        document,
                        measurement,
                        lambda point: QPointF(point.x, point.y),
                        settings,
                        line_width=2.0,
                        endpoint_radius=4.0,
                        selected=False,
                        show_fill=True,
                        show_handles=False,
                        geometry_mode=AREA_GEOMETRY_SCREEN,
                        proxy_build_budget=budget,
                    )

            self.assertTrue(budget.deferred)
            self.assertEqual(label.call_count, 2)
            centroid.assert_called_once_with(measurement)
            self.assertIsNotNone(
                area_derived_geometry_service.cached_centroid(measurement)
            )
        finally:
            painter.end()
            area_derived_geometry_service.clear()

    def test_selected_area_passive_body_is_raw_before_active_emphasis(self) -> None:
        points = [
            Point(
                80.0 + (55.0 * math.cos(2.0 * math.pi * index / 512)),
                70.0 + (42.0 * math.sin(2.0 * math.pi * index / 512)),
            )
            for index in range(512)
        ]
        measurement = Measurement(
            id="selected-area",
            image_id="doc",
            fiber_group_id=None,
            mode="polygon_area",
            measurement_kind="area",
            polygon_px=points,
            area_rings_px=[points],
        )
        document = ImageDocument(
            id="doc",
            path="/tmp/doc.png",
            image_size=(160, 140),
            measurements=[measurement],
        )
        document.select_measurement(measurement.id)
        canvas = DocumentCanvas()
        canvas.resize(160, 140)
        canvas.set_document(
            document,
            QImage(160, 140, QImage.Format.Format_RGB32),
        )
        target = QImage(160, 140, QImage.Format.Format_ARGB32_Premultiplied)
        painter = QPainter(target)
        try:
            with (
                patch.object(canvas, "_overlay_cache_enabled", return_value=False),
                patch.object(rendering, "draw_area_measurement") as passive,
                patch("fdm.ui.canvas.draw_area_measurement") as active,
                patch("fdm.ui.canvas.draw_overlay_annotations"),
            ):
                canvas._draw_annotations(  # noqa: SLF001
                    painter,
                    canvas._paint_context(),  # noqa: SLF001
                )

            self.assertEqual(
                passive.call_args.kwargs["geometry_mode"],
                AREA_GEOMETRY_RAW,
            )
            active.assert_called_once()
        finally:
            painter.end()
            canvas.clear_document()
            canvas.close()

    def test_low_zoom_culling_keeps_maximum_cosmetic_strokes_and_markers(
        self,
    ) -> None:
        image_to_output = lambda point: QPointF(point.x * 0.1, point.y * 0.1)
        tile_rect = QRectF(0.0, 0.0, 5120.0, 5120.0)
        padding = rendering.measurement_geometry_cull_padding(
            image_to_output,
            endpoint_radius=4.0,
        )
        area_points = [
            Point(5300.0, 100.0),
            Point(5340.0, 100.0),
            Point(5340.0, 140.0),
            Point(5300.0, 140.0),
        ]
        area = Measurement(
            id="wide-area",
            image_id="doc",
            fiber_group_id=None,
            mode="polygon_area",
            measurement_kind="area",
            polygon_px=area_points,
            area_rings_px=[area_points],
            appearance=ObjectAppearanceOverride(stroke_width=24.0),
        )
        count = Measurement(
            id="large-count",
            image_id="doc",
            fiber_group_id=None,
            mode="count",
            measurement_kind="count",
            point_px=Point(5300.0, 200.0),
            appearance=ObjectAppearanceOverride(marker_scale=4.0),
        )
        document = ImageDocument(
            id="doc",
            path="/tmp/doc.png",
            image_size=(6000, 1000),
            measurements=[area, count],
        )
        settings = AppSettings(show_count_numbers=False)

        self.assertGreaterEqual(padding, 480.0)
        self.assertTrue(
            rendering.measurement_display_intersects_rect(
                area,
                document,
                settings,
                image_to_output,
                tile_rect,
                padding=padding,
            )
        )
        self.assertTrue(
            rendering.measurement_display_intersects_rect(
                count,
                document,
                settings,
                image_to_output,
                tile_rect,
                padding=padding,
                count_number=1,
            )
        )

    def test_dynamic_line_endpoint_bounds_cover_every_rendered_pixel(
        self,
    ) -> None:
        for endpoint_style in (
            MeasurementEndpointStyle.BAR,
            MeasurementEndpointStyle.ARROW_INSIDE,
            MeasurementEndpointStyle.ARROW_OUTSIDE,
        ):
            for selected in (False, True):
                with self.subTest(
                    endpoint_style=endpoint_style,
                    selected=selected,
                ):
                    measurement = Measurement(
                        id="wide-line",
                        image_id="doc",
                        fiber_group_id=None,
                        mode="manual",
                        measurement_kind="line",
                        line_px=Line(
                            Point(300.0, 300.0),
                            Point(500.0, 300.0),
                        ),
                        appearance=ObjectAppearanceOverride(
                            stroke_width=24.0,
                        ),
                    )
                    document = ImageDocument(
                        id="doc",
                        path="/tmp/wide-line.png",
                        image_size=(900, 700),
                        measurements=[measurement],
                    )
                    if selected:
                        document.select_measurement(measurement.id)
                    settings = AppSettings(
                        measurement_endpoint_style=endpoint_style,
                        length_measurement_label_style=(
                            MeasurementLabelStyleSettings(enabled=False)
                        ),
                    )
                    surface = QImage(
                        900,
                        700,
                        QImage.Format.Format_ARGB32_Premultiplied,
                    )
                    surface.fill(0)
                    painter = QPainter(surface)
                    try:
                        rendering.draw_measurements(
                            painter,
                            document,
                            lambda point: QPointF(point.x, point.y),
                            settings,
                            line_width=2.0,
                            endpoint_radius=4.0,
                            selected_measurement_id=(
                                measurement.id if selected else None
                            ),
                            cull_by_geometry=False,
                        )
                    finally:
                        painter.end()

                    alpha_bounds = self._alpha_bounds(surface)
                    display_bounds = rendering.measurement_display_image_bounds(
                        measurement,
                        document,
                        settings,
                        lambda point: QPointF(point.x, point.y),
                        suggested_line_width=2.0,
                        endpoint_radius=4.0,
                        selected=selected,
                    )
                    self.assertIsNotNone(display_bounds)
                    self.assertTrue(
                        display_bounds.adjusted(-1, -1, 1, 1).contains(
                            alpha_bounds
                        ),
                        (display_bounds, alpha_bounds),
                    )

    def test_selected_bar_entering_viewport_is_added_to_direct_candidates(
        self,
    ) -> None:
        measurement = Measurement(
            id="selected-wide-bar",
            image_id="doc",
            fiber_group_id=None,
            mode="manual",
            measurement_kind="line",
            line_px=Line(
                Point(200.0, 650.0),
                Point(400.0, 650.0),
            ),
            appearance=ObjectAppearanceOverride(stroke_width=24.0),
        )
        document = ImageDocument(
            id="doc",
            path="/tmp/selected-wide-bar.png",
            image_size=(900, 900),
            measurements=[measurement],
        )
        document.select_measurement(measurement.id)
        settings = AppSettings(
            measurement_endpoint_style=MeasurementEndpointStyle.BAR,
            length_measurement_label_style=MeasurementLabelStyleSettings(
                enabled=False,
            ),
        )
        viewport = QRectF(0.0, 0.0, 512.0, 512.0)
        canvas = DocumentCanvas()
        try:
            canvas.resize(512, 512)
            canvas.set_document(
                document,
                QImage(900, 900, QImage.Format.Format_RGB32),
            )
            canvas.set_settings(settings)
            canvas._zoom = 1.0  # noqa: SLF001
            canvas._pan = Point(0.0, 0.0)  # noqa: SLF001

            passive_index = canvas._measurement_display_scene_index(  # noqa: SLF001
                zoom=1.0
            )
            self.assertIsNotNone(passive_index)
            self.assertNotIn(measurement, passive_index.query_rect(viewport))

            visible, _count_numbers = canvas._measurement_render_inputs(  # noqa: SLF001
                viewport,
                zoom=1.0,
            )
            self.assertIn(measurement, visible)
        finally:
            canvas.clear_document()
            canvas.close()

    def test_large_area_label_bounds_and_display_index_cover_cross_tile_pixels(
        self,
    ) -> None:
        for font_size in (96, 144):
            for zoom in (0.75, 1.5):
                with self.subTest(font_size=font_size, zoom=zoom):
                    points = [
                        Point(700.0, 360.0),
                        Point(730.0, 360.0),
                        Point(730.0, 390.0),
                        Point(700.0, 390.0),
                    ]
                    measurement = Measurement(
                        id="long-area-label",
                        image_id="doc",
                        fiber_group_id=None,
                        mode="polygon_area",
                        measurement_kind="area",
                        polygon_px=points,
                        area_rings_px=[points],
                        area_px=123_456_789.12345678,
                        appearance=ObjectAppearanceOverride(
                            font_size=float(font_size),
                        ),
                    )
                    document = ImageDocument(
                        id="doc",
                        path="/tmp/long-area-label.png",
                        image_size=(1200, 800),
                        measurements=[measurement],
                    )
                    settings = AppSettings(
                        area_measurement_label_style=(
                            MeasurementLabelStyleSettings(
                                enabled=True,
                                decimals=8,
                            )
                        )
                    )
                    image_to_output = lambda point: QPointF(
                        point.x * zoom,
                        point.y * zoom,
                    )
                    surface = QImage(
                        1800,
                        1200,
                        QImage.Format.Format_ARGB32_Premultiplied,
                    )
                    surface.fill(0)
                    painter = QPainter(surface)
                    try:
                        rendering.draw_measurements(
                            painter,
                            document,
                            image_to_output,
                            settings,
                            line_width=2.0,
                            endpoint_radius=4.0,
                            show_area_fill=False,
                            cull_by_geometry=False,
                        )
                    finally:
                        painter.end()

                    alpha_bounds = self._alpha_bounds(surface)
                    image_alpha_bounds = QRectF(
                        alpha_bounds.left() / zoom,
                        alpha_bounds.top() / zoom,
                        alpha_bounds.width() / zoom,
                        alpha_bounds.height() / zoom,
                    )
                    display_bounds = rendering.measurement_display_image_bounds(
                        measurement,
                        document,
                        settings,
                        image_to_output,
                        suggested_line_width=2.0,
                        endpoint_radius=4.0,
                    )
                    self.assertIsNotNone(display_bounds)
                    self.assertTrue(
                        display_bounds.adjusted(
                            -2.0 / zoom,
                            -2.0 / zoom,
                            2.0 / zoom,
                            2.0 / zoom,
                        ).contains(image_alpha_bounds),
                        (display_bounds, image_alpha_bounds),
                    )

                    canvas = DocumentCanvas()
                    try:
                        canvas.resize(512, 512)
                        canvas.set_document(
                            document,
                            QImage(
                                1200,
                                800,
                                QImage.Format.Format_RGB32,
                            ),
                        )
                        canvas.set_settings(settings)
                        canvas._zoom = zoom  # noqa: SLF001
                        canvas._pan = Point(0.0, 0.0)  # noqa: SLF001
                        visible, _numbers = canvas._measurement_render_inputs(  # noqa: SLF001
                            QRectF(0.0, 0.0, 512.0 / zoom, 512.0 / zoom),
                            zoom=zoom,
                        )
                        self.assertIn(measurement, visible)
                    finally:
                        canvas.clear_document()
                        canvas.close()

    def test_large_count_label_bounds_and_display_index_cover_cross_tile_pixels(
        self,
    ) -> None:
        measurement = Measurement(
            id="large-count-label",
            image_id="doc",
            fiber_group_id=None,
            mode="count",
            measurement_kind="count",
            point_px=Point(470.0, 300.0),
            appearance=ObjectAppearanceOverride(
                font_size=144.0,
                marker_scale=2.0,
            ),
        )
        document = ImageDocument(
            id="doc",
            path="/tmp/large-count-label.png",
            image_size=(1200, 800),
            measurements=[measurement],
        )
        settings = AppSettings(show_count_numbers=True)
        surface = QImage(
            1200,
            800,
            QImage.Format.Format_ARGB32_Premultiplied,
        )
        surface.fill(0)
        painter = QPainter(surface)
        try:
            rendering.draw_count_measurement(
                painter,
                document,
                measurement,
                lambda point: QPointF(point.x, point.y),
                settings,
                endpoint_radius=4.0,
                selected=True,
                count_number=1,
            )
        finally:
            painter.end()

        alpha_bounds = self._alpha_bounds(surface)
        display_bounds = rendering.measurement_display_image_bounds(
            measurement,
            document,
            settings,
            lambda point: QPointF(point.x, point.y),
            endpoint_radius=4.0,
            selected=True,
            count_number=1,
        )
        self.assertIsNotNone(display_bounds)
        self.assertTrue(
            display_bounds.adjusted(-1.0, -1.0, 1.0, 1.0).contains(
                alpha_bounds
            ),
            (display_bounds, alpha_bounds),
        )

        canvas = DocumentCanvas()
        try:
            canvas.resize(512, 512)
            canvas.set_document(
                document,
                QImage(1200, 800, QImage.Format.Format_RGB32),
            )
            canvas.set_settings(settings)
            canvas._zoom = 1.0  # noqa: SLF001
            canvas._pan = Point(0.0, 0.0)  # noqa: SLF001
            visible, count_numbers = canvas._measurement_render_inputs(  # noqa: SLF001
                QRectF(512.0, 0.0, 512.0, 512.0),
                zoom=1.0,
            )
            self.assertIn(measurement, visible)
            self.assertEqual(count_numbers[measurement.id], 1)
        finally:
            canvas.clear_document()
            canvas.close()

    def test_selected_area_outside_dirty_view_does_not_submit_raw_path(self) -> None:
        points = [
            Point(5000.0, 5000.0),
            Point(5100.0, 5000.0),
            Point(5100.0, 5100.0),
            Point(5000.0, 5100.0),
        ]
        measurement = Measurement(
            id="offscreen-area",
            image_id="doc",
            fiber_group_id=None,
            mode="polygon_area",
            measurement_kind="area",
            polygon_px=points,
            area_rings_px=[points],
        )
        document = ImageDocument(
            id="doc",
            path="/tmp/doc.png",
            image_size=(6000, 6000),
            measurements=[measurement],
        )
        document.select_measurement(measurement.id)
        canvas = DocumentCanvas()
        canvas.resize(320, 240)
        canvas.set_document(
            document,
            QImage(320, 240, QImage.Format.Format_RGB32),
        )
        canvas._zoom = 1.0  # noqa: SLF001
        canvas._pan = Point(0.0, 0.0)  # noqa: SLF001
        target = QImage(320, 240, QImage.Format.Format_ARGB32_Premultiplied)
        painter = QPainter(target)
        try:
            with patch("fdm.ui.canvas.draw_area_measurement") as draw_area:
                canvas._draw_selected_measurement_active_layer(  # noqa: SLF001
                    painter,
                    canvas._paint_context(),  # noqa: SLF001
                )
            draw_area.assert_not_called()
        finally:
            painter.end()
            canvas.clear_document()
            canvas.close()

    def test_partial_paint_keeps_full_viewport_tile_working_set(self) -> None:
        document = ImageDocument(
            id="doc",
            path="/tmp/doc.png",
            image_size=(2048, 1024),
        )
        document.measurements = [
            self._line(f"line-{index}", float(index * 20))
            for index in range(64)
        ]
        canvas = DocumentCanvas()
        canvas.resize(1024, 768)
        canvas.set_document(
            document,
            QImage(2048, 1024, QImage.Format.Format_RGB32),
        )
        canvas._zoom = 1.0  # noqa: SLF001
        canvas._pan = Point(0.0, 0.0)  # noqa: SLF001
        full_context = canvas._paint_context()  # noqa: SLF001
        dirty_context = canvas._paint_context(QRectF(8.0, 8.0, 64.0, 64.0))  # noqa: SLF001
        full_keys = canvas._visible_overlay_tile_keys(full_context)  # noqa: SLF001
        working_keys = canvas._overlay_prefetch_tile_keys(full_keys)  # noqa: SLF001
        dirty_keys = canvas._visible_overlay_tile_keys(dirty_context)  # noqa: SLF001
        self.assertGreater(len(full_keys), len(dirty_keys))
        self.assertGreater(len(working_keys), len(full_keys))

        target = QImage(1024, 768, QImage.Format.Format_ARGB32_Premultiplied)
        painter = QPainter(target)
        try:
            with (
                patch.object(
                    canvas_overlay_tile_cache,
                    "get_payload",
                    return_value=None,
                ) as get_payload,
                patch.object(
                    canvas,
                    "_draw_measurements_direct",
                    return_value=False,
                ),
                patch.object(canvas, "_enqueue_overlay_tiles") as enqueue,
            ):
                canvas._draw_measurement_overlay_tiles(  # noqa: SLF001
                    painter,
                    dirty_context,
                )
        finally:
            painter.end()

        self.assertEqual(get_payload.call_count, len(dirty_keys))
        self.assertEqual(enqueue.call_args.args[0], working_keys)
        self.assertEqual(canvas._overlay_visible_keys, set(working_keys))  # noqa: SLF001
        canvas.clear_document()
        canvas.close()

    def test_dual_payload_uses_picture_at_rest_and_raster_during_pan(
        self,
    ) -> None:
        document = ImageDocument(
            id="doc",
            path="/tmp/doc.png",
            image_size=(320, 240),
            measurements=[
                self._line(f"line-{index}", float(index))
                for index in range(64)
            ],
        )
        canvas = DocumentCanvas()
        cache = CanvasOverlayTileCache(
            max_entries=8,
            max_bytes=16 * 1024 * 1024,
            thread_pool=type(
                "InlinePool",
                (),
                {"start": lambda _self, runnable: runnable.run()},
            )(),
        )
        canvas.resize(320, 240)
        canvas.set_document(
            document,
            QImage(320, 240, QImage.Format.Format_RGB32),
        )
        canvas._zoom = 1.0  # noqa: SLF001
        canvas._pan = Point(0.0, 0.0)  # noqa: SLF001
        canvas._sync_overlay_visual_state()  # noqa: SLF001
        context = canvas._paint_context()  # noqa: SLF001
        key = next(
            key
            for key in canvas._visible_overlay_tile_keys(context)  # noqa: SLF001
            if key.tile_x == 0 and key.tile_y == 0
        )
        picture = QPicture()
        picture_painter = QPainter(picture)
        picture_painter.fillRect(
            QRectF(0.0, 0.0, 80.0, 80.0),
            QColor("#FF0000"),
        )
        picture_painter.end()
        with patch.object(
            canvas_module,
            "canvas_overlay_tile_cache",
            cache,
        ):
            self.assertTrue(
                cache.request(
                    CanvasOverlayRenderSnapshot(
                        request_id=1,
                        key=key,
                        picture=picture,
                        exact_composition=True,
                    )
                )
            )
            image, cached_picture = cache.get_payload(key)
            self.assertIsNotNone(image)
            self.assertIsNotNone(cached_picture)
            image.fill(QColor("#00FF00"))

            def draw_frame(*, panning: bool) -> QImage:
                canvas._panning = panning  # noqa: SLF001
                target = QImage(
                    320,
                    240,
                    QImage.Format.Format_ARGB32_Premultiplied,
                )
                target.fill(0)
                painter = QPainter(target)
                try:
                    with (
                        patch.object(
                            canvas,
                            "_draw_measurements_direct",
                            return_value=False,
                        ) as direct,
                        patch.object(canvas, "_enqueue_overlay_tiles"),
                    ):
                        canvas._draw_measurement_overlay_tiles(  # noqa: SLF001
                            painter,
                            context,
                        )
                    direct.assert_not_called()
                finally:
                    painter.end()
                return target

            still = draw_frame(panning=False)
            moving = draw_frame(panning=True)
            released = draw_frame(panning=False)

        self.assertEqual(still.pixelColor(20, 20).name(), "#ff0000")
        self.assertEqual(moving.pixelColor(20, 20).name(), "#00ff00")
        self.assertEqual(released.pixelColor(20, 20).name(), "#ff0000")
        canvas._end_canvas_pan()  # noqa: SLF001
        cache.clear()
        canvas.clear_document()
        canvas.close()

    def test_selection_does_not_invalidate_passive_tiles(self) -> None:
        document = ImageDocument(id="doc", path="/tmp/doc.png", image_size=(2000, 1000))
        first = self._line("first", 10.0)
        second = self._line("second", 1500.0)
        document.measurements = [first, second]
        canvas = DocumentCanvas()
        canvas.set_document(document, QImage(2000, 1000, QImage.Format.Format_RGB32))
        canvas._overlay_known_namespaces.add((1.0, 1.0))  # noqa: SLF001
        canvas._sync_overlay_visual_state()  # noqa: SLF001

        document.select_measurement(first.id)
        canvas._sync_overlay_visual_state()  # noqa: SLF001

        self.assertFalse(
            any(
                epoch > 0
                for epoch in canvas._overlay_tile_epochs.values()  # noqa: SLF001
            )
        )
        self.assertEqual(canvas._overlay_selected_measurement_id, first.id)  # noqa: SLF001
        canvas.clear_document()
        canvas.close()

    def test_programmatic_selection_refreshes_only_its_display_bounds(self) -> None:
        document = ImageDocument(id="doc", path="/tmp/doc.png", image_size=(2000, 500))
        measurement = self._line("middle", 1000.0)
        document.measurements = [measurement]
        canvas = DocumentCanvas()
        canvas.resize(2000, 500)
        canvas.set_document(
            document,
            QImage(2000, 500, QImage.Format.Format_RGB32),
        )
        canvas._zoom = 1.0  # noqa: SLF001
        canvas._pan = Point(0.0, 0.0)  # noqa: SLF001
        try:
            with patch.object(canvas, "update") as update:
                canvas.set_selected_measurement(measurement.id)

            update.assert_called_once()
            dirty_rect = update.call_args.args[0]
            self.assertGreater(dirty_rect.width(), 0)
            self.assertLess(dirty_rect.width(), canvas.width())

            with patch.object(canvas, "update") as repeated_update:
                canvas.set_selected_measurement(measurement.id)
            repeated_update.assert_not_called()
        finally:
            canvas.clear_document()
            canvas.close()

    def test_digital_slide_direct_render_uses_global_viewport_coordinates(self) -> None:
        visible = self._line("visible", 1010.0)
        visible.line_px = Line(Point(1010.0, 2010.0), Point(1030.0, 2030.0))
        outside = self._line("outside", 1800.0)
        outside.line_px = Line(Point(1800.0, 2800.0), Point(1820.0, 2820.0))
        document = ImageDocument(
            id="slide",
            path="/tmp/slide.fdmslide",
            image_size=(4096, 4096),
            document_kind="digital_slide",
        )
        document.measurements = [visible, outside]
        canvas = DigitalSlideCanvas()
        canvas.resize(320, 240)
        canvas.set_document(
            document,
            QImage(320, 240, QImage.Format.Format_RGB32),
        )
        canvas._viewport_origin = Point(1000.0, 2000.0)  # noqa: SLF001
        canvas._zoom = 2.0  # noqa: SLF001
        canvas._pan = Point(20.0, 30.0)  # noqa: SLF001
        target = QImage(320, 240, QImage.Format.Format_ARGB32_Premultiplied)
        painter = QPainter(target)
        try:
            context = canvas._paint_context()  # noqa: SLF001
            mapped_origin = context.image_to_widget_transform.map(
                QPointF(1000.0, 2000.0)
            )
            self.assertAlmostEqual(mapped_origin.x(), 20.0)
            self.assertAlmostEqual(mapped_origin.y(), 30.0)
            mapped_back = context.widget_to_image_transform.map(QPointF(20.0, 30.0))
            self.assertAlmostEqual(mapped_back.x(), 1000.0)
            self.assertAlmostEqual(mapped_back.y(), 2000.0)
            self.assertGreaterEqual(context.image_rect.left(), 1000.0)
            self.assertGreaterEqual(context.image_rect.top(), 2000.0)

            with (
                patch.dict(
                    os.environ,
                    {"FDM_DISABLE_CANVAS_OVERLAY_CACHE": "1"},
                    clear=False,
                ),
                patch("fdm.ui.canvas.draw_measurements", return_value=False) as draw,
                patch("fdm.ui.canvas.draw_overlay_annotations"),
            ):
                canvas._draw_annotations(painter, context)  # noqa: SLF001

            rendered = draw.call_args.kwargs["measurement_sequence"]
            self.assertEqual([measurement.id for measurement in rendered], ["visible"])
            self.assertEqual(
                [
                    measurement.id
                    for measurement in canvas._measurement_candidates(  # noqa: SLF001
                        Point(1015.0, 2015.0),
                        tolerance=8.0,
                    )
                ],
                ["visible"],
            )

            update_rect = canvas._image_rect_to_widget_update_rect(  # noqa: SLF001
                QRectF(1010.0, 2010.0, 20.0, 10.0)
            )
            self.assertEqual((update_rect.x(), update_rect.y()), (34, 44))
            self.assertEqual((update_rect.width(), update_rect.height()), (52, 32))

            with patch.object(canvas, "update") as update:
                canvas._apply_visual_change(  # noqa: SLF001
                    CanvasVisualChange(
                        object_ids=("visible",),
                        new_bounds=CanvasDisplayBounds(
                            QRectF(1010.0, 2010.0, 20.0, 10.0)
                        ),
                    )
                )
            update.assert_called_once()
            visual_rect = update.call_args.args[0]
            self.assertEqual((visual_rect.x(), visual_rect.y()), (32, 42))
            self.assertEqual((visual_rect.width(), visual_rect.height()), (56, 36))
        finally:
            painter.end()
            canvas.clear_document()
            canvas.close()

    def test_digital_slide_dense_overlay_tiles_keep_global_grid_and_candidates(
        self,
    ) -> None:
        measurements = []
        for index in range(64):
            measurement = self._line(f"visible-{index}", 0.0)
            x = 1005.0 + (index % 8) * 12.0
            y = 2005.0 + (index // 8) * 12.0
            measurement.line_px = Line(Point(x, y), Point(x + 8.0, y + 8.0))
            measurements.append(measurement)
        for index in range(6):
            measurement = self._line(f"outside-{index}", 0.0)
            x = 1800.0 + index * 10.0
            measurement.line_px = Line(
                Point(x, 2800.0),
                Point(x + 8.0, 2808.0),
            )
            measurements.append(measurement)
        document = ImageDocument(
            id="dense-slide",
            path="/tmp/dense-slide.fdmslide",
            image_size=(4096, 4096),
            document_kind="digital_slide",
        )
        document.measurements = measurements
        canvas = DigitalSlideCanvas()
        canvas.resize(320, 240)
        canvas.set_document(
            document,
            QImage(320, 240, QImage.Format.Format_RGB32),
        )
        canvas._viewport_origin = Point(1000.0, 2000.0)  # noqa: SLF001
        canvas._zoom = 2.0  # noqa: SLF001
        canvas._pan = Point(20.0, 30.0)  # noqa: SLF001
        target = QImage(320, 240, QImage.Format.Format_ARGB32_Premultiplied)
        painter = QPainter(target)
        try:
            context = canvas._paint_context()  # noqa: SLF001
            keys = canvas._visible_overlay_tile_keys(context)  # noqa: SLF001
            origin_key = next(
                key
                for key in keys
                if canvas._overlay_tile_image_rect(key).contains(  # noqa: SLF001
                    QPointF(1000.0, 2000.0)
                )
            )
            tile_image_rect = canvas._overlay_tile_image_rect(origin_key)  # noqa: SLF001
            tile_widget_rect = canvas._overlay_tile_widget_rect(origin_key)  # noqa: SLF001
            mapped_top_left = canvas.image_to_widget(
                Point(tile_image_rect.left(), tile_image_rect.top())
            )
            self.assertAlmostEqual(tile_widget_rect.left(), mapped_top_left.x())
            self.assertAlmostEqual(tile_widget_rect.top(), mapped_top_left.y())

            with (
                patch.dict(
                    os.environ,
                    {"FDM_ENABLE_CANVAS_OVERLAY_CACHE": "1"},
                    clear=False,
                ),
                patch(
                    "fdm.ui.canvas.canvas_overlay_tile_cache.get",
                    return_value=None,
                ),
                patch.object(canvas, "_enqueue_overlay_tiles"),
                patch("fdm.ui.canvas.draw_measurements", return_value=False) as draw,
            ):
                canvas._draw_measurement_overlay_tiles(painter, context)  # noqa: SLF001

            rendered = draw.call_args.kwargs["measurement_sequence"]
            self.assertEqual(len(rendered), 64)
            self.assertTrue(
                all(measurement.id.startswith("visible-") for measurement in rendered)
            )
        finally:
            painter.end()
            canvas.clear_document()
            canvas.close()


if __name__ == "__main__":
    unittest.main()
