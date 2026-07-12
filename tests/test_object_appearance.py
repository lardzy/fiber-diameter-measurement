from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PySide6.QtCore import QPointF
from PySide6.QtGui import QColor, QImage, QPainter
from PySide6.QtWidgets import QApplication

from fdm.geometry import Line, Point
from fdm.models import (
    ImageDocument,
    Measurement,
    ObjectAppearanceOverride,
    OverlayAnnotation,
    OverlayAnnotationKind,
)
from fdm.settings import AppSettings
from fdm.ui.canvas import DocumentCanvas
from fdm.ui.object_inspector import CurrentObjectInspector
from fdm.ui.rendering import (
    annotation_rect,
    draw_measurements,
    draw_overlay_annotations,
    measurement_color,
    measurement_label_font,
    measurement_line_width,
    measurement_marker_scale,
    overlay_annotation_line_width,
    overlay_text_font,
)


class ObjectAppearanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_appearance_normalizes_values_and_omits_invalid_fields(self) -> None:
        appearance = ObjectAppearanceOverride(
            stroke_color="#abc",
            stroke_width=float("inf"),
            text_color="not-a-color",
            font_family="  Segoe UI  ",
            font_size=999,
            marker_scale=0,
        )

        self.assertEqual(appearance.stroke_color, "#AABBCC")
        self.assertIsNone(appearance.stroke_width)
        self.assertIsNone(appearance.text_color)
        self.assertEqual(appearance.font_family, "Segoe UI")
        self.assertEqual(appearance.font_size, 144)
        self.assertEqual(appearance.marker_scale, 0.25)
        json.dumps(appearance.to_dict(), allow_nan=False)

    def test_measurement_and_overlay_appearance_roundtrip_is_backward_compatible(self) -> None:
        appearance = ObjectAppearanceOverride(
            stroke_color="#123456",
            stroke_width=4.5,
            text_color="#FEDCBA",
            font_family="Microsoft YaHei UI",
            font_size=24,
            marker_scale=1.5,
        )
        measurement = Measurement(
            id="measurement_1",
            image_id="image_1",
            fiber_group_id=None,
            mode="manual",
            line_px=Line(Point(1, 2), Point(11, 2)),
            appearance=appearance,
        )
        annotation = OverlayAnnotation(
            id="overlay_1",
            image_id="image_1",
            kind=OverlayAnnotationKind.TEXT,
            content="测试文字",
            anchor_px=Point(10, 20),
            appearance=appearance,
        )

        measurement_payload = measurement.to_dict()
        annotation_payload = annotation.to_dict()
        restored_measurement = Measurement.from_dict(measurement_payload)
        restored_annotation = OverlayAnnotation.from_dict(annotation_payload)

        self.assertEqual(restored_measurement.appearance, appearance)
        self.assertEqual(restored_annotation.appearance, appearance)
        self.assertNotIn("appearance", Measurement.from_dict({
            **measurement_payload,
            "appearance": {"stroke_width": float("nan")},
        }).to_dict())
        self.assertNotIn("appearance", Measurement(
            id="measurement_2",
            image_id="image_1",
            fiber_group_id=None,
            mode="manual",
            line_px=Line(Point(1, 2), Point(11, 2)),
        ).to_dict())

    def test_measurement_appearance_has_priority_over_category_and_defaults(self) -> None:
        document = ImageDocument(id="image_1", path="/tmp/image.png", image_size=(100, 80))
        group = document.create_group(color="#00FF00", label="棉")
        measurement = Measurement(
            id="measurement_1",
            image_id=document.id,
            fiber_group_id=group.id,
            mode="manual",
            line_px=Line(Point(1, 2), Point(11, 2)),
            appearance=ObjectAppearanceOverride(
                stroke_color="#FF0000",
                stroke_width=5,
                text_color="#0000FF",
                font_family="Arial",
                font_size=30,
                marker_scale=2,
            ),
        )

        settings = AppSettings(default_measurement_color="#FFFFFF")
        self.assertEqual(measurement_color(document, measurement, settings), QColor("#FF0000"))
        self.assertEqual(measurement_line_width(measurement, 2.0), 5.0)
        self.assertEqual(measurement_marker_scale(measurement), 2.0)
        self.assertEqual(measurement_label_font(settings, measurement).family(), "Arial")
        self.assertEqual(measurement_label_font(settings, measurement).pixelSize(), 30)

    def test_overlay_appearance_controls_shape_rendering_and_text_hit_bounds(self) -> None:
        settings = AppSettings(text_font_size=12, overlay_line_color="#00FF00", overlay_line_width=2)
        shape = OverlayAnnotation(
            id="shape_1",
            image_id="image_1",
            kind=OverlayAnnotationKind.RECT,
            start_px=Point(10, 10),
            end_px=Point(80, 60),
            appearance=ObjectAppearanceOverride(stroke_color="#FF0000", stroke_width=6),
        )
        document = ImageDocument(id="image_1", path="/tmp/image.png", image_size=(160, 120))
        document.add_overlay_annotation(shape)
        target = QImage(160, 120, QImage.Format.Format_ARGB32)
        target.fill(QColor("#00000000"))
        painter = QPainter(target)
        captured: list[dict[str, object]] = []
        try:
            with patch(
                "fdm.ui.rendering._draw_shape_overlay_annotation",
                side_effect=lambda *_args, **kwargs: captured.append(kwargs),
            ):
                draw_overlay_annotations(
                    painter,
                    document,
                    lambda point: QPointF(point.x, point.y),
                    settings,
                    render_mode="full_resolution",
                )
        finally:
            painter.end()

        self.assertEqual(captured[0]["color"], QColor("#FF0000"))
        self.assertEqual(captured[0]["line_width"], 6.0)
        self.assertEqual(
            overlay_annotation_line_width(
                settings,
                suggested_line_width=2.2,
                render_mode="full_resolution",
                annotation=shape,
            ),
            6.0,
        )
        screen_widths = [
            overlay_annotation_line_width(
                settings,
                suggested_line_width=2.2,
                render_mode="screen_scale_full_image",
                annotation=shape.clone(
                    appearance=ObjectAppearanceOverride(stroke_width=width)
                ),
            )
            for width in (4.0, 8.0, 24.0)
        ]
        self.assertEqual(screen_widths, [4.0, 8.0, 24.0])

        text = OverlayAnnotation(
            id="text_1",
            image_id=document.id,
            kind=OverlayAnnotationKind.TEXT,
            content="Large label",
            anchor_px=Point(20, 30),
            appearance=ObjectAppearanceOverride(font_family="Arial", font_size=48),
        )
        document.overlay_annotations = [text]
        default_text = text.clone(appearance=None)
        large_rect = annotation_rect(text, settings, lambda point: QPointF(point.x, point.y))
        default_rect = annotation_rect(default_text, settings, lambda point: QPointF(point.x, point.y))
        self.assertEqual(overlay_text_font(settings, text).pixelSize(), 48)
        self.assertGreater(large_rect.width(), default_rect.width())

        canvas = DocumentCanvas()
        canvas.resize(160, 120)
        canvas.set_document(document, QImage(160, 120, QImage.Format.Format_RGB32))
        canvas_rect = annotation_rect(text, settings, canvas.image_to_widget)
        widget_point = canvas_rect.center()
        image_point = canvas.widget_to_image(widget_point)
        self.assertEqual(canvas._hit_test_overlay_annotation(widget_point, image_point), text.id)

    def test_count_number_offsets_are_grouped_by_marker_scale(self) -> None:
        document = ImageDocument(id="image", path="/tmp/counts.png", image_size=(120, 80))
        document.measurements = [
            Measurement(
                id="first",
                image_id=document.id,
                fiber_group_id=None,
                mode="count",
                measurement_kind="count",
                point_px=Point(20, 20),
                appearance=ObjectAppearanceOverride(marker_scale=1.0),
            ),
            Measurement(
                id="second",
                image_id=document.id,
                fiber_group_id=None,
                mode="count",
                measurement_kind="count",
                point_px=Point(60, 20),
                appearance=ObjectAppearanceOverride(marker_scale=2.0),
            ),
        ]
        target = QImage(120, 80, QImage.Format.Format_ARGB32)
        target.fill(QColor("#00000000"))
        painter = QPainter(target)
        radii: list[float] = []
        try:
            with patch(
                "fdm.ui.rendering._draw_count_number_labels",
                side_effect=lambda *_args, **kwargs: radii.append(kwargs["endpoint_radius"]),
            ):
                draw_measurements(
                    painter,
                    document,
                    lambda point: QPointF(point.x, point.y),
                    AppSettings(show_count_numbers=True),
                    line_width=2.0,
                    endpoint_radius=4.0,
                )
        finally:
            painter.end()
        self.assertEqual(sorted(radii), [4.0, 8.0])

    def test_area_measurement_draws_the_configurable_result_label(self) -> None:
        document = ImageDocument(id="area-image", path="/tmp/area.png", image_size=(120, 100))
        area = Measurement(
            id="area",
            image_id=document.id,
            fiber_group_id=None,
            mode="polygon_area",
            measurement_kind="area",
            polygon_px=[Point(20, 20), Point(100, 20), Point(100, 80), Point(20, 80)],
            appearance=ObjectAppearanceOverride(
                text_color="#FF00FF",
                font_family="Arial",
                font_size=28,
            ),
        )
        area.recalculate(None)
        document.measurements = [area]
        target = QImage(120, 100, QImage.Format.Format_ARGB32)
        target.fill(QColor("#00000000"))
        painter = QPainter(target)
        calls: list[Measurement] = []
        try:
            with patch(
                "fdm.ui.rendering.draw_area_measurement_label",
                side_effect=lambda _painter, measurement, *_args: calls.append(measurement),
            ):
                draw_measurements(
                    painter,
                    document,
                    lambda point: QPointF(point.x, point.y),
                    AppSettings(show_measurement_labels=True),
                    line_width=2.0,
                    endpoint_radius=4.0,
                )
        finally:
            painter.end()
        self.assertEqual(calls, [area])
        self.assertEqual(measurement_label_font(AppSettings(), area).pixelSize(), 28)

    def test_mixed_measurement_overrides_render_without_bypassing_object_colors(self) -> None:
        document = ImageDocument(id="image_1", path="/tmp/image.png", image_size=(180, 140))
        line = Measurement(
            id="line_1",
            image_id=document.id,
            fiber_group_id=None,
            mode="manual",
            line_px=Line(Point(20, 25), Point(130, 25)),
            appearance=ObjectAppearanceOverride(stroke_color="#FF0000", stroke_width=6),
        )
        line.recalculate(None)
        count = Measurement(
            id="count_1",
            image_id=document.id,
            fiber_group_id=None,
            mode="count",
            measurement_kind="count",
            point_px=Point(60, 90),
            appearance=ObjectAppearanceOverride(
                stroke_color="#00FF00",
                text_color="#0000FF",
                marker_scale=2,
            ),
        )
        count.recalculate(None)
        document.add_measurement(line)
        document.add_measurement(count)

        target = QImage(180, 140, QImage.Format.Format_ARGB32)
        target.fill(QColor("#00000000"))
        painter = QPainter(target)
        try:
            draw_measurements(
                painter,
                document,
                lambda point: QPointF(point.x, point.y),
                AppSettings(show_measurement_labels=False, show_count_numbers=False),
                line_width=2,
                endpoint_radius=4,
            )
        finally:
            painter.end()

        self.assertEqual(target.pixelColor(75, 25).name(), "#ff0000")
        self.assertEqual(target.pixelColor(60, 90).name(), "#00ff00")

    def test_hit_testing_tracks_large_marker_and_stroke_overrides(self) -> None:
        document = ImageDocument(id="hit", path="/tmp/hit.png", image_size=(140, 100))
        count = Measurement(
            id="large-count",
            image_id=document.id,
            fiber_group_id=None,
            mode="count",
            measurement_kind="count",
            point_px=Point(40, 50),
            appearance=ObjectAppearanceOverride(marker_scale=4.0),
        )
        line = Measurement(
            id="wide-line",
            image_id=document.id,
            fiber_group_id=None,
            mode="manual",
            line_px=Line(Point(90, 50), Point(130, 50)),
            appearance=ObjectAppearanceOverride(stroke_width=24.0),
        )
        document.measurements = [count, line]
        canvas = DocumentCanvas()
        canvas.resize(140, 100)
        canvas.set_document(document, QImage(140, 100, QImage.Format.Format_RGB32))
        canvas._zoom = 1.0

        self.assertEqual(canvas._hit_test_measurement(Point(58, 50)), count.id)
        self.assertEqual(canvas._hit_test_measurement(Point(110, 64)), line.id)
        canvas.close()

    def test_inspector_does_not_freeze_inherited_values_on_focus_only(self) -> None:
        document = ImageDocument(id="inspector", path="/tmp/inspector.png", image_size=(80, 60))
        measurement = Measurement(
            id="line",
            image_id=document.id,
            fiber_group_id=None,
            mode="manual",
            line_px=Line(Point(5, 10), Point(45, 10)),
        )
        measurement.recalculate(None)
        document.measurements = [measurement]
        inspector = CurrentObjectInspector()
        inspector.set_context(
            document,
            settings=AppSettings(),
            measurement_ids=[measurement.id],
        )
        changes: list[object] = []
        inspector.appearanceChangeRequested.connect(
            lambda _kind, _object_id, appearance: changes.append(appearance)
        )

        inspector._stroke_width_spin.editingFinished.emit()
        inspector._font_size_spin.editingFinished.emit()
        self.assertEqual(changes, [])

        inspector._stroke_width_spin.setValue(4.0)
        inspector._stroke_width_spin.editingFinished.emit()
        self.assertEqual(len(changes), 1)
        inspector.close()

    def test_count_inspector_shows_marker_size_but_not_unused_line_width(self) -> None:
        document = ImageDocument(id="count-inspector", path="/tmp/count.png", image_size=(80, 60))
        count = Measurement(
            id="count",
            image_id=document.id,
            fiber_group_id=None,
            mode="count",
            measurement_kind="count",
            point_px=Point(20, 20),
        )
        document.measurements = [count]
        inspector = CurrentObjectInspector()
        inspector.set_context(
            document,
            settings=AppSettings(),
            measurement_ids=[count.id],
        )
        self.assertFalse(inspector._stroke_color_button.isHidden())
        self.assertTrue(inspector._stroke_width_spin.isHidden())
        self.assertFalse(inspector._marker_scale_spin.isHidden())
        inspector.close()


if __name__ == "__main__":
    unittest.main()
