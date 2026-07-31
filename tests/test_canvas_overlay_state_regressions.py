from __future__ import annotations

import os
from pathlib import Path
import sys
import unittest
from unittest.mock import call, patch

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PySide6.QtCore import QEvent
from PySide6.QtGui import QColor, QImage, QPainter
from PySide6.QtWidgets import QApplication

from fdm.geometry import Line, Point
from fdm.models import (
    Calibration,
    ImageDocument,
    Measurement,
    ObjectAppearanceOverride,
)
from fdm.settings import AppSettings, MeasurementLabelStyleSettings
from fdm.ui.canvas import DocumentCanvas


class CanvasOverlayStateRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    @staticmethod
    def _count(identifier: str, x: float) -> Measurement:
        return Measurement(
            id=identifier,
            image_id="doc",
            fiber_group_id=None,
            mode="count",
            measurement_kind="count",
            point_px=Point(x, 100.0),
        )

    @staticmethod
    def _line(identifier: str, x: float) -> Measurement:
        return Measurement(
            id=identifier,
            image_id="doc",
            fiber_group_id=None,
            mode="manual",
            measurement_kind="line",
            line_px=Line(Point(x, 120.0), Point(x + 20.0, 120.0)),
            diameter_px=20.0,
            diameter_unit=20.0,
        )

    def test_deleting_an_earlier_count_invalidates_later_count_number_tiles(
        self,
    ) -> None:
        document = ImageDocument(
            id="doc",
            path="/tmp/counts.png",
            image_size=(2048, 300),
        )
        first = self._count("first", 40.0)
        later = self._count("later", 1600.0)
        document.measurements = [first, later]
        canvas = DocumentCanvas()
        try:
            canvas.set_document(
                document,
                QImage(2048, 300, QImage.Format.Format_RGB32),
            )
            canvas._overlay_known_namespaces.add((1.0, 1.0))  # noqa: SLF001
            canvas._sync_overlay_visual_state()  # noqa: SLF001

            document.remove_measurement_incremental(first.id)
            canvas._sync_overlay_visual_state()  # noqa: SLF001

            invalidated_x = {
                tile_x
                for (
                    _zoom,
                    _dpr,
                    tile_x,
                    _tile_y,
                ), epoch in canvas._overlay_tile_epochs.items()  # noqa: SLF001
                if epoch > 0
            }
            self.assertIn(3, invalidated_x)
        finally:
            canvas.clear_document()
            canvas.close()

    def test_calibration_unit_only_change_invalidates_cached_labels(self) -> None:
        document = ImageDocument(
            id="doc",
            path="/tmp/unit.png",
            image_size=(640, 360),
            calibration=Calibration(
                mode="preset",
                pixels_per_unit=2.0,
                unit="μm",
                source_label="unit regression",
            ),
            measurements=[self._line("line", 80.0)],
        )
        canvas = DocumentCanvas()
        try:
            canvas.set_document(
                document,
                QImage(640, 360, QImage.Format.Format_RGB32),
            )
            canvas._sync_overlay_visual_state()  # noqa: SLF001
            assert document.calibration is not None
            document.calibration.unit = "mm"
            document.mark_calibration_dirty()

            with patch.object(
                canvas,
                "_invalidate_all_overlay_tiles",
                wraps=canvas._invalidate_all_overlay_tiles,  # noqa: SLF001
            ) as invalidate:
                canvas._sync_overlay_visual_state()  # noqa: SLF001

            invalidate.assert_called()
        finally:
            canvas.clear_document()
            canvas.close()

    def test_measurement_order_change_invalidates_passive_composition(self) -> None:
        first = self._line("first", 80.0)
        second = self._line("second", 85.0)
        document = ImageDocument(
            id="doc",
            path="/tmp/order.png",
            image_size=(640, 360),
            measurements=[first, second],
        )
        canvas = DocumentCanvas()
        try:
            canvas.set_document(
                document,
                QImage(640, 360, QImage.Format.Format_RGB32),
            )
            canvas._sync_overlay_visual_state()  # noqa: SLF001
            document.measurements = [second, first]
            document.mark_session_dirty()

            with patch.object(
                canvas,
                "_invalidate_all_overlay_tiles",
                wraps=canvas._invalidate_all_overlay_tiles,  # noqa: SLF001
            ) as invalidate:
                canvas._sync_overlay_visual_state()  # noqa: SLF001

            invalidate.assert_called()
            rendered, _count_numbers = canvas._measurement_render_inputs(  # noqa: SLF001
                canvas._paint_context().image_rect  # noqa: SLF001
            )
            self.assertEqual(
                [measurement.id for measurement in rendered],
                ["second", "first"],
            )
        finally:
            canvas.clear_document()
            canvas.close()

    def test_object_level_large_font_label_reaches_neighbour_tile(self) -> None:
        measurement = self._line("large-object-font", 300.0)
        measurement.diameter_unit = 123_456_789.125
        measurement.appearance = ObjectAppearanceOverride(font_size=144.0)
        document = ImageDocument(
            id="doc",
            path="/tmp/object-font.png",
            image_size=(1024, 300),
            measurements=[measurement],
        )
        canvas = DocumentCanvas()
        try:
            canvas.resize(1024, 300)
            canvas.set_settings(
                AppSettings(
                    length_measurement_label_style=MeasurementLabelStyleSettings(
                        enabled=True,
                        font_size=14,
                        decimals=3,
                        background_enabled=True,
                    )
                )
            )
            source = QImage(1024, 300, QImage.Format.Format_RGB32)
            source.fill(QColor("#FFFFFF"))
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
            self.assertIsNotNone(snapshot.picture)
            surface = QImage(
                1024,
                300,
                QImage.Format.Format_ARGB32_Premultiplied,
            )
            surface.fill(0)
            painter = QPainter(surface)
            assert snapshot is not None and snapshot.picture is not None
            snapshot.picture.play(painter)
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

    def test_shrinking_object_font_invalidates_the_old_large_label_extent(
        self,
    ) -> None:
        measurement = self._line("large-to-small", 300.0)
        measurement.appearance = ObjectAppearanceOverride(font_size=144.0)
        document = ImageDocument(
            id="doc",
            path="/tmp/font-change.png",
            image_size=(2048, 300),
            measurements=[measurement],
        )
        canvas = DocumentCanvas()
        try:
            canvas.set_document(
                document,
                QImage(2048, 300, QImage.Format.Format_RGB32),
            )
            canvas._overlay_known_namespaces.add((1.0, 1.0))  # noqa: SLF001
            canvas._sync_overlay_visual_state()  # noqa: SLF001

            measurement.appearance = ObjectAppearanceOverride(font_size=14.0)
            document.mark_session_dirty()
            canvas._sync_overlay_visual_state()  # noqa: SLF001

            invalidated_x = {
                tile_x
                for (
                    _zoom,
                    _dpr,
                    tile_x,
                    _tile_y,
                ), epoch in canvas._overlay_tile_epochs.items()  # noqa: SLF001
                if epoch > 0
            }
            self.assertIn(1, invalidated_x)
        finally:
            canvas.clear_document()
            canvas.close()

    def test_single_object_style_change_requests_only_a_bounded_update(
        self,
    ) -> None:
        measurement = self._line("styled", 260.0)
        document = ImageDocument(
            id="doc",
            path="/tmp/style-local.png",
            image_size=(1024, 360),
            measurements=[measurement],
        )
        canvas = DocumentCanvas()
        try:
            canvas.resize(1024, 360)
            canvas.set_document(
                document,
                QImage(1024, 360, QImage.Format.Format_RGB32),
            )
            canvas._zoom = 1.0  # noqa: SLF001
            canvas._pan = Point(0.0, 0.0)  # noqa: SLF001
            canvas._sync_overlay_visual_state()  # noqa: SLF001

            measurement.appearance = ObjectAppearanceOverride(
                stroke_color="#FF0000",
            )
            document.mark_session_dirty()
            with patch.object(canvas, "update") as update:
                canvas.notify_document_visual_changed()

            self.assertTrue(update.call_args_list)
            for repaint in update.call_args_list:
                self.assertTrue(
                    repaint.args,
                    "single-object style change used a full update()",
                )
                rect = repaint.args[0]
                self.assertLess(rect.width(), canvas.width())
        finally:
            canvas.clear_document()
            canvas.close()

    def test_appending_one_measurement_requests_only_its_visual_envelope(
        self,
    ) -> None:
        document = ImageDocument(
            id="doc",
            path="/tmp/append-local.png",
            image_size=(1024, 360),
        )
        canvas = DocumentCanvas()
        try:
            canvas.resize(1024, 360)
            canvas.set_document(
                document,
                QImage(1024, 360, QImage.Format.Format_RGB32),
            )
            canvas._zoom = 1.0  # noqa: SLF001
            canvas._pan = Point(0.0, 0.0)  # noqa: SLF001
            canvas._sync_overlay_visual_state()  # noqa: SLF001

            document.insert_measurement_incremental(
                self._line("appended", 420.0)
            )
            document.mark_session_dirty()
            with patch.object(canvas, "update") as update:
                canvas.notify_document_visual_changed()

            self.assertTrue(update.call_args_list)
            for repaint in update.call_args_list:
                self.assertTrue(
                    repaint.args,
                    "measurement append used a full update()",
                )
                rect = repaint.args[0]
                self.assertLess(rect.width(), canvas.width())
        finally:
            canvas.clear_document()
            canvas.close()

    def test_palette_change_starts_a_new_exact_style_generation(self) -> None:
        document = ImageDocument(
            id="doc",
            path="/tmp/palette.png",
            image_size=(320, 240),
            measurements=[self._line("line", 40.0)],
        )
        canvas = DocumentCanvas()
        try:
            canvas.set_document(
                document,
                QImage(320, 240, QImage.Format.Format_RGB32),
            )
            previous_generation = canvas._overlay_style_generation  # noqa: SLF001
            with patch.object(
                canvas,
                "_invalidate_all_overlay_tiles",
                wraps=canvas._invalidate_all_overlay_tiles,  # noqa: SLF001
            ) as invalidate:
                canvas.changeEvent(QEvent(QEvent.Type.PaletteChange))

            self.assertEqual(
                canvas._overlay_style_generation,  # noqa: SLF001
                previous_generation + 1,
            )
            invalidate.assert_called_once_with()
        finally:
            canvas.clear_document()
            canvas.close()

    def test_zoom_dpr_namespace_history_is_bounded(self) -> None:
        document = ImageDocument(
            id="namespace-doc",
            path="/tmp/namespace.png",
            image_size=(320, 240),
        )
        canvas = DocumentCanvas()
        try:
            canvas.set_document(
                document,
                QImage(320, 240, QImage.Format.Format_RGB32),
            )
            with patch(
                "fdm.ui.canvas.canvas_overlay_tile_cache.invalidate_namespace"
            ) as invalidate_namespace:
                for index in range(12):
                    canvas._remember_overlay_namespace(  # noqa: SLF001
                        float(index + 1),
                        1.0,
                    )
            self.assertEqual(
                len(canvas._overlay_known_namespaces),  # noqa: SLF001
                8,
            )
            self.assertEqual(
                canvas._overlay_namespace_order[0],  # noqa: SLF001
                (5.0, 1.0),
            )
            self.assertEqual(
                canvas._overlay_namespace_order[-1],  # noqa: SLF001
                (12.0, 1.0),
            )
            self.assertEqual(
                invalidate_namespace.call_args_list,
                [
                    call(id(document), 1.0, 1.0),
                    call(id(document), 2.0, 1.0),
                    call(id(document), 3.0, 1.0),
                    call(id(document), 4.0, 1.0),
                ],
            )
        finally:
            canvas.clear_document()
            canvas.close()


if __name__ == "__main__":
    unittest.main()
