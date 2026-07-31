from __future__ import annotations

import json
import math
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QFont, QImage, QPainter
from PySide6.QtWidgets import QApplication

import fdm.area_display as area_display
from fdm.area_display import area_derived_geometry_service
from fdm.geometry import Line, Point
from fdm.models import (
    ImageDocument,
    Measurement,
    ObjectAppearanceOverride,
    OverlayAnnotation,
    OverlayAnnotationKind,
    OverlayTextAnchorAlignment,
    OverlayTextLayoutSpec,
    OverlayTextSizeSpace,
)
from fdm.settings import AppSettings, MeasurementLabelStyleSettings
import fdm.ui.rendering as rendering
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
    overlay_annotation_rect,
    overlay_annotation_line_width,
    overlay_text_font,
    resolve_overlay_text_layout,
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

    def test_overlay_text_has_no_automatic_contrast_outline_on_screen_or_export(self) -> None:
        settings = AppSettings(text_color="#112233")
        annotation = OverlayAnnotation(
            id="text",
            image_id="image",
            kind=OverlayAnnotationKind.TEXT,
            content="叠加文字",
            anchor_px=Point(20, 30),
        )
        document = ImageDocument(
            id="image",
            path="/tmp/image.png",
            image_size=(180, 120),
            overlay_annotations=[annotation],
        )

        for render_mode in ("screen_scale_full_image", "full_resolution"):
            with self.subTest(render_mode=render_mode):
                target = QImage(180, 120, QImage.Format.Format_ARGB32)
                target.fill(QColor("#00000000"))
                painter = QPainter(target)
                try:
                    with patch.object(
                        rendering,
                        "_draw_cached_text",
                        wraps=rendering._draw_cached_text,
                    ) as draw_cached_text:
                        draw_overlay_annotations(
                            painter,
                            document,
                            lambda point: QPointF(point.x, point.y),
                            settings,
                            render_mode=render_mode,
                        )
                finally:
                    painter.end()

                draw_cached_text.assert_called_once()
                self.assertEqual(
                    draw_cached_text.call_args.kwargs["color"],
                    QColor("#112233"),
                )
                self.assertIsNone(
                    draw_cached_text.call_args.kwargs["outline"],
                )

    def test_legacy_overlay_text_keeps_fixed_output_size_and_top_left_anchor(self) -> None:
        settings = AppSettings(text_font_family="Arial", text_font_size=20)
        annotation = OverlayAnnotation(
            id="legacy",
            image_id="image",
            kind=OverlayAnnotationKind.TEXT,
            content="旧版\n文字",
            anchor_px=Point(100, 80),
            text_layout=None,
        )

        full_resolution = resolve_overlay_text_layout(
            annotation,
            settings,
            lambda point: QPointF(point.x, point.y),
            render_mode="full_resolution",
        )
        screen = resolve_overlay_text_layout(
            annotation,
            settings,
            lambda point: QPointF(point.x * 0.1, point.y * 0.1),
            render_mode="screen_scale_full_image",
        )

        self.assertEqual(full_resolution.font.pixelSize(), 20)
        self.assertEqual(screen.font.pixelSize(), 20)
        self.assertAlmostEqual(full_resolution.text_rect.left(), 100.0)
        self.assertAlmostEqual(full_resolution.text_rect.top(), 80.0)
        self.assertAlmostEqual(screen.text_rect.left(), 10.0)
        self.assertAlmostEqual(screen.text_rect.top(), 8.0)
        self.assertAlmostEqual(screen.text_rect.width(), full_resolution.text_rect.width())
        self.assertAlmostEqual(screen.text_rect.height(), full_resolution.text_rect.height())
        self.assertEqual(
            overlay_annotation_rect(
                annotation,
                settings,
                lambda point: QPointF(point.x * 0.1, point.y * 0.1),
            ),
            screen.annotation_rect,
        )

    def test_image_space_overlay_text_scales_between_screen_and_full_resolution(self) -> None:
        annotation = OverlayAnnotation(
            id="image-space",
            image_id="image",
            kind=OverlayAnnotationKind.TEXT,
            content="原图文字",
            anchor_px=Point(1000, 800),
            text_layout=OverlayTextLayoutSpec(
                anchor_alignment=OverlayTextAnchorAlignment.CENTER,
                size_space=OverlayTextSizeSpace.IMAGE_PX,
                image_font_size_px=180.0,
            ),
        )
        settings = AppSettings(text_font_family="Arial", text_font_size=12)

        screen = resolve_overlay_text_layout(
            annotation,
            settings,
            lambda point: QPointF(point.x * 0.1, point.y * 0.1),
            render_mode="screen_scale_full_image",
        )
        full_resolution = resolve_overlay_text_layout(
            annotation,
            settings,
            lambda point: QPointF(point.x, point.y),
            render_mode="full_resolution",
        )

        self.assertEqual(screen.font.pixelSize(), 18)
        self.assertEqual(full_resolution.font.pixelSize(), 180)
        self.assertAlmostEqual(screen.text_rect.center().x(), 100.0)
        self.assertAlmostEqual(screen.text_rect.center().y(), 80.0)
        self.assertAlmostEqual(full_resolution.text_rect.center().x(), 1000.0)
        self.assertAlmostEqual(full_resolution.text_rect.center().y(), 800.0)
        self.assertAlmostEqual(screen.annotation_rect.center().x(), 100.0)
        self.assertAlmostEqual(screen.annotation_rect.center().y(), 80.0)

    def test_explicit_legacy_size_uses_spec_without_output_scaling(self) -> None:
        annotation = OverlayAnnotation(
            id="explicit-legacy",
            image_id="image",
            kind=OverlayAnnotationKind.TEXT,
            content="固定字号",
            anchor_px=Point(100, 80),
            appearance=ObjectAppearanceOverride(font_size=72),
            text_layout=OverlayTextLayoutSpec(
                anchor_alignment=OverlayTextAnchorAlignment.CENTER,
                size_space=OverlayTextSizeSpace.LEGACY_OUTPUT_PX,
                image_font_size_px=24.0,
            ),
        )
        settings = AppSettings(text_font_size=16)

        for scale in (0.1, 1.0):
            with self.subTest(scale=scale):
                resolved = resolve_overlay_text_layout(
                    annotation,
                    settings,
                    lambda point, active_scale=scale: QPointF(
                        point.x * active_scale,
                        point.y * active_scale,
                    ),
                )
                self.assertEqual(resolved.font.pixelSize(), 24)

    def test_multiline_overlay_text_uses_anchor_for_the_complete_text_block(self) -> None:
        settings = AppSettings(text_font_family="Arial")
        expected_anchor = QPointF(240.0, 160.0)
        for alignment, horizontal, vertical in (
            (OverlayTextAnchorAlignment.TOP_LEFT, 0.0, 0.0),
            (OverlayTextAnchorAlignment.TOP_CENTER, 0.5, 0.0),
            (OverlayTextAnchorAlignment.CENTER, 0.5, 0.5),
            (OverlayTextAnchorAlignment.BOTTOM_RIGHT, 1.0, 1.0),
        ):
            with self.subTest(alignment=alignment):
                annotation = OverlayAnnotation(
                    id=alignment,
                    image_id="image",
                    kind=OverlayAnnotationKind.TEXT,
                    content="第一行\n更长的第二行",
                    anchor_px=Point(expected_anchor.x(), expected_anchor.y()),
                    text_layout=OverlayTextLayoutSpec(
                        anchor_alignment=alignment,
                        size_space=OverlayTextSizeSpace.IMAGE_PX,
                        image_font_size_px=32.0,
                    ),
                )
                resolved = resolve_overlay_text_layout(
                    annotation,
                    settings,
                    lambda point: QPointF(point.x, point.y),
                )

                self.assertAlmostEqual(
                    resolved.text_rect.left() + (resolved.text_rect.width() * horizontal),
                    expected_anchor.x(),
                )
                self.assertAlmostEqual(
                    resolved.text_rect.top() + (resolved.text_rect.height() * vertical),
                    expected_anchor.y(),
                )

    def test_image_space_overlay_text_does_not_multiply_device_pixel_ratio(self) -> None:
        settings = AppSettings(text_font_family="Arial")
        annotation = OverlayAnnotation(
            id="dpr",
            image_id="image",
            kind=OverlayAnnotationKind.TEXT,
            content="DPR",
            anchor_px=Point(100, 80),
            text_layout=OverlayTextLayoutSpec(
                anchor_alignment=OverlayTextAnchorAlignment.CENTER,
                size_space=OverlayTextSizeSpace.IMAGE_PX,
                image_font_size_px=180.0,
            ),
        )
        document = ImageDocument(
            id="image",
            path="/tmp/image.png",
            image_size=(2000, 1200),
            overlay_annotations=[annotation],
        )
        target = QImage(400, 240, QImage.Format.Format_ARGB32)
        target.setDevicePixelRatio(2.0)
        target.fill(QColor("#00000000"))
        painter = QPainter(target)
        captured_font_sizes: list[int] = []

        def capture_text(active_painter, *_args, **_kwargs) -> None:
            captured_font_sizes.append(active_painter.font().pixelSize())

        try:
            with patch.object(rendering, "_draw_cached_text", side_effect=capture_text):
                draw_overlay_annotations(
                    painter,
                    document,
                    lambda point: QPointF(point.x * 0.1, point.y * 0.1),
                    settings,
                    render_mode="screen_scale_full_image",
                )
        finally:
            painter.end()

        self.assertEqual(captured_font_sizes, [18])

    def test_overlay_text_output_font_clamp_does_not_mutate_persistent_size(self) -> None:
        spec = OverlayTextLayoutSpec(
            anchor_alignment=OverlayTextAnchorAlignment.CENTER,
            size_space=OverlayTextSizeSpace.IMAGE_PX,
            image_font_size_px=8000.0,
        )
        annotation = OverlayAnnotation(
            id="clamped",
            image_id="image",
            kind=OverlayAnnotationKind.TEXT,
            content="极大文字",
            anchor_px=Point(10, 10),
            text_layout=spec,
        )

        resolved = resolve_overlay_text_layout(
            annotation,
            AppSettings(),
            lambda point: QPointF(point.x * 2.0, point.y * 2.0),
        )

        self.assertEqual(resolved.font.pixelSize(), 8192)
        self.assertEqual(spec.image_font_size_px, 8000.0)

    def test_overlay_shapes_use_only_the_configured_stroke_color(self) -> None:
        settings = AppSettings(overlay_line_color="#112233")
        annotations = [
            OverlayAnnotation(
                id=kind,
                image_id="image",
                kind=kind,
                start_px=Point(15, 20 + index * 20),
                end_px=Point(100, 35 + index * 20),
            )
            for index, kind in enumerate(
                (
                    OverlayAnnotationKind.RECT,
                    OverlayAnnotationKind.CIRCLE,
                    OverlayAnnotationKind.LINE,
                    OverlayAnnotationKind.ARROW,
                )
            )
        ]
        document = ImageDocument(
            id="image",
            path="/tmp/image.png",
            image_size=(140, 120),
            overlay_annotations=annotations,
        )
        target = QImage(140, 120, QImage.Format.Format_ARGB32)
        target.fill(QColor("#00000000"))
        painter = QPainter(target)
        stroke_colors: list[QColor] = []
        original_draw = rendering._draw_overlay_shape_geometry

        def capture_stroke(active_painter, *args, **kwargs) -> None:
            stroke_colors.append(QColor(active_painter.pen().color()))
            original_draw(active_painter, *args, **kwargs)

        try:
            with patch.object(
                rendering,
                "_draw_overlay_shape_geometry",
                side_effect=capture_stroke,
            ):
                draw_overlay_annotations(
                    painter,
                    document,
                    lambda point: QPointF(point.x, point.y),
                    settings,
                )
        finally:
            painter.end()

        self.assertEqual(stroke_colors, [QColor("#112233")] * len(annotations))

    def test_text_layout_cache_is_bounded_and_shared_with_multiline_hit_bounds(self) -> None:
        rendering._TEXT_LAYOUT_CACHE.clear()
        rendering._TEXT_LAYOUT_CACHE_CHARACTERS = 0
        font = QFont("Arial")
        font.setPixelSize(24)
        with (
            patch.object(rendering, "_TEXT_LAYOUT_MAX_ENTRIES", 3),
            patch.object(rendering, "_TEXT_LAYOUT_MAX_CHARACTERS", 18),
        ):
            first = rendering._cached_text_layout(font, "中文\n第二行", render_mode="overlay")
            repeated = rendering._cached_text_layout(font, "中文\n第二行", render_mode="overlay")
            self.assertIs(first, repeated)
            for index in range(8):
                rendering._cached_text_layout(font, f"文字{index}", render_mode="overlay")
            self.assertLessEqual(len(rendering._TEXT_LAYOUT_CACHE), 3)
            self.assertLessEqual(rendering._TEXT_LAYOUT_CACHE_CHARACTERS, 18)

        annotation = OverlayAnnotation(
            id="multiline",
            image_id="image",
            kind=OverlayAnnotationKind.TEXT,
            content="中文\n第二行",
            anchor_px=Point(20, 30),
            appearance=ObjectAppearanceOverride(font_family="Arial", font_size=24),
        )
        rect = annotation_rect(annotation, AppSettings(), lambda point: QPointF(point.x, point.y))
        self.assertAlmostEqual(rect.width(), first.width + 12.0)
        self.assertAlmostEqual(rect.height(), first.height + 8.0)

    def test_offscreen_overlay_text_is_culled_before_drawing(self) -> None:
        document = ImageDocument(id="image", path="/tmp/image.png", image_size=(100, 80))
        document.overlay_annotations = [
            OverlayAnnotation(
                id="outside",
                image_id=document.id,
                kind=OverlayAnnotationKind.TEXT,
                content="画布外文字",
                anchor_px=Point(500, 500),
            )
        ]
        target = QImage(100, 80, QImage.Format.Format_ARGB32)
        target.fill(QColor("#00000000"))
        painter = QPainter(target)
        try:
            with patch("fdm.ui.rendering._draw_cached_text") as draw_cached:
                draw_overlay_annotations(
                    painter,
                    document,
                    lambda point: QPointF(point.x, point.y),
                    AppSettings(),
                )
            draw_cached.assert_not_called()
        finally:
            painter.end()

    def test_area_display_path_cache_reuses_path_and_preserves_hole_and_exact_area(self) -> None:
        area_derived_geometry_service.clear()
        outer = [Point(10, 10), Point(90, 10), Point(90, 90), Point(10, 90)]
        hole = [Point(35, 35), Point(35, 65), Point(65, 65), Point(65, 35)]
        measurement = Measurement(
            id="area",
            image_id="image",
            fiber_group_id=None,
            mode="polygon_area",
            measurement_kind="area",
            polygon_px=outer,
            area_rings_px=[outer, hole],
            exact_area_px=4321.0,
        )
        measurement.recalculate(None)
        document = ImageDocument(id="image", path="/tmp/image.png", image_size=(100, 100))
        document.add_measurement(measurement)

        first = area_derived_geometry_service.raw_path(measurement)
        second = area_derived_geometry_service.raw_path(measurement)
        self.assertIs(first, second)
        self.assertEqual(first.fillRule(), Qt.FillRule.OddEvenFill)
        self.assertTrue(first.contains(QPointF(20, 20)))
        self.assertFalse(first.contains(QPointF(50, 50)))
        self.assertEqual(measurement.area_px, 4321.0)

        measurement.replace_area_geometry(
            polygon_px=measurement.polygon_px,
            area_rings_px=measurement.area_rings_px,
            exact_area_px=measurement.exact_area_px,
        )
        third = area_derived_geometry_service.raw_path(measurement)
        self.assertIsNot(first, third)

        area_derived_geometry_service.clear()
        with patch.object(area_display, "_PATH_MAX_ESTIMATED_BYTES", 600):
            for offset in range(6):
                candidate = Measurement(
                    id=f"area-{offset}",
                    image_id=document.id,
                    fiber_group_id=None,
                    mode="polygon_area",
                    measurement_kind="area",
                    polygon_px=[Point(point.x + offset, point.y) for point in outer],
                )
                area_derived_geometry_service.raw_path(candidate)
            self.assertLessEqual(
                len(area_derived_geometry_service._raw_paths)
                + len(area_derived_geometry_service._proxies),
                2,
            )
            self.assertLessEqual(area_derived_geometry_service._path_bytes, 600)

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

    def test_selected_dense_raw_area_keeps_exact_label_without_proxy_retry(self) -> None:
        area_derived_geometry_service.clear()
        ring = [
            Point(
                60.0 + (35.0 * math.cos(2.0 * math.pi * index / 512)),
                50.0 + (25.0 * math.sin(2.0 * math.pi * index / 512)),
            )
            for index in range(512)
        ]
        document = ImageDocument(id="selected-area", path="/tmp/area.png", image_size=(120, 100))
        area = Measurement(
            id="area",
            image_id=document.id,
            fiber_group_id=None,
            mode="polygon_area",
            measurement_kind="area",
            polygon_px=ring,
            area_rings_px=[ring],
        )
        document.measurements = [area]
        target = QImage(120, 100, QImage.Format.Format_ARGB32)
        target.fill(QColor("#00000000"))
        painter = QPainter(target)
        try:
            with patch("fdm.ui.rendering.draw_area_measurement_label") as draw_label:
                deferred = draw_measurements(
                    painter,
                    document,
                    lambda point: QPointF(point.x, point.y),
                    AppSettings(show_measurement_labels=True),
                    line_width=2.0,
                    endpoint_radius=4.0,
                    selected_measurement_id=area.id,
                )
        finally:
            painter.end()

        draw_label.assert_called_once()
        self.assertFalse(deferred)

    def test_length_and_area_labels_use_independent_styles(self) -> None:
        document = ImageDocument(id="mixed", path="/tmp/mixed.png", image_size=(160, 120))
        line = Measurement(
            id="line",
            image_id=document.id,
            fiber_group_id=None,
            mode="manual",
            line_px=Line(Point(10, 20), Point(110, 20)),
        )
        area = Measurement(
            id="area",
            image_id=document.id,
            fiber_group_id=None,
            mode="polygon_area",
            measurement_kind="area",
            polygon_px=[Point(20, 50), Point(100, 50), Point(100, 100), Point(20, 100)],
        )
        document.add_measurement(line)
        document.add_measurement(area)
        settings = AppSettings(
            length_measurement_label_style=MeasurementLabelStyleSettings(
                enabled=False,
                font_family="Arial",
                font_size=18,
                decimals=1,
            ),
            area_measurement_label_style=MeasurementLabelStyleSettings(
                enabled=True,
                font_family="Arial",
                font_size=30,
                decimals=3,
            ),
        )
        target = QImage(160, 120, QImage.Format.Format_ARGB32)
        target.fill(QColor("#00000000"))
        painter = QPainter(target)
        try:
            with (
                patch("fdm.ui.rendering.draw_measurement_label") as draw_length,
                patch("fdm.ui.rendering.draw_area_measurement_label") as draw_area,
            ):
                draw_measurements(
                    painter,
                    document,
                    lambda point: QPointF(point.x, point.y),
                    settings,
                    line_width=2.0,
                    endpoint_radius=4.0,
                )
            draw_length.assert_not_called()
            draw_area.assert_called_once()
        finally:
            painter.end()

        self.assertEqual(measurement_label_font(settings, line).pixelSize(), 18)
        self.assertEqual(measurement_label_font(settings, area).pixelSize(), 30)
        self.assertTrue(rendering.measurement_display_text_with_settings(line, document, settings).endswith(".0 px"))
        self.assertTrue(rendering.measurement_display_text_with_settings(area, document, settings).endswith(".000 px²"))

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

    def test_area_inspector_uses_area_label_style_defaults(self) -> None:
        document = ImageDocument(id="area-inspector", path="/tmp/area.png", image_size=(80, 60))
        area = Measurement(
            id="area",
            image_id=document.id,
            fiber_group_id=None,
            mode="polygon_area",
            measurement_kind="area",
            polygon_px=[Point(5, 5), Point(50, 5), Point(50, 45), Point(5, 45)],
        )
        area.recalculate(None)
        document.measurements = [area]
        settings = AppSettings(
            length_measurement_label_style=MeasurementLabelStyleSettings(
                color="#112233",
                font_size=18,
            ),
            area_measurement_label_style=MeasurementLabelStyleSettings(
                color="#AABBCC",
                font_size=31,
            ),
        )
        inspector = CurrentObjectInspector()
        inspector.set_context(
            document,
            settings=settings,
            measurement_ids=[area.id],
        )

        self.assertEqual(inspector._text_color_button.property("color_value"), "#AABBCC")
        self.assertEqual(inspector._font_size_spin.value(), 31)
        self.assertEqual(inspector._control_values["text_color"], "#AABBCC")
        inspector.close()

    def test_inspector_exposes_image_space_text_size_and_anchor_controls(self) -> None:
        text = OverlayAnnotation(
            id="image-space-text",
            image_id="text-inspector",
            kind=OverlayAnnotationKind.TEXT,
            content="原图文字",
            anchor_px=Point(500, 400),
            text_layout=OverlayTextLayoutSpec(
                anchor_alignment=OverlayTextAnchorAlignment.CENTER,
                size_space=OverlayTextSizeSpace.IMAGE_PX,
                image_font_size_px=180.0,
            ),
        )
        document = ImageDocument(
            id="text-inspector",
            path="/tmp/text.png",
            image_size=(2000, 1200),
            overlay_annotations=[text],
        )
        document.select_overlay_annotation(text.id)
        inspector = CurrentObjectInspector()
        inspector.set_context(
            document,
            settings=AppSettings(),
            overlay_id=text.id,
            view_zoom=0.1,
        )

        self.assertFalse(inspector._text_layout_group.isHidden())
        self.assertEqual(inspector._text_size_space_label.text(), "随图像缩放（推荐）")
        self.assertEqual(inspector._text_image_font_size_spin.value(), 180.0)
        self.assertTrue(inspector._text_image_font_size_spin.isEnabled())
        self.assertTrue(inspector._font_size_spin.isHidden())
        self.assertTrue(
            inspector._text_anchor_buttons[OverlayTextAnchorAlignment.CENTER].isChecked()
        )
        self.assertIn("当前画布约 18 px", inspector._text_layout_summary_label.text())
        self.assertIn("完整分辨率导出 180 px", inspector._text_layout_summary_label.text())
        self.assertTrue(inspector._legacy_text_layout_label.isHidden())
        self.assertTrue(inspector._text_layout_conversion_button.isHidden())
        self.assertFalse(inspector._text_actual_size_button.isHidden())

        changes: list[tuple[str, OverlayTextLayoutSpec]] = []
        actual_size_previews: list[str] = []
        inspector.overlayTextLayoutChangeRequested.connect(
            lambda overlay_id, spec: changes.append((overlay_id, spec))
        )
        inspector.overlayTextActualSizePreviewRequested.connect(
            actual_size_previews.append
        )
        inspector._text_image_font_size_spin.setValue(220.0)
        inspector._text_image_font_size_spin.editingFinished.emit()
        inspector._text_anchor_buttons[OverlayTextAnchorAlignment.BOTTOM_RIGHT].click()
        inspector._text_actual_size_button.click()

        self.assertEqual(len(changes), 2)
        self.assertEqual(changes[0][0], text.id)
        self.assertEqual(changes[0][1].image_font_size_px, 220.0)
        self.assertEqual(
            changes[1][1].anchor_alignment,
            OverlayTextAnchorAlignment.BOTTOM_RIGHT,
        )
        self.assertEqual(changes[1][1].image_font_size_px, 220.0)
        self.assertEqual(actual_size_previews, [text.id])
        inspector.close()

    def test_inspector_keeps_legacy_text_read_only_until_explicit_conversion(self) -> None:
        text = OverlayAnnotation(
            id="legacy-text",
            image_id="legacy-inspector",
            kind=OverlayAnnotationKind.TEXT,
            content="旧版文字",
            anchor_px=Point(20, 30),
            appearance=ObjectAppearanceOverride(font_size=18),
            text_layout=None,
        )
        document = ImageDocument(
            id="legacy-inspector",
            path="/tmp/legacy.png",
            image_size=(2000, 1200),
            overlay_annotations=[text],
        )
        inspector = CurrentObjectInspector()
        inspector.set_context(
            document,
            settings=AppSettings(text_font_size=16),
            overlay_id=text.id,
            view_zoom=0.1,
        )

        self.assertEqual(inspector._text_size_space_label.text(), "旧版固定输出像素")
        self.assertFalse(inspector._text_image_font_size_spin.isEnabled())
        self.assertEqual(inspector._text_image_font_size_spin.value(), 180.0)
        self.assertFalse(inspector._font_size_spin.isHidden())
        self.assertTrue(
            inspector._text_anchor_buttons[OverlayTextAnchorAlignment.TOP_LEFT].isChecked()
        )
        self.assertTrue(
            all(not button.isEnabled() for button in inspector._text_anchor_buttons.values())
        )
        self.assertIn("当前画布约 18 px", inspector._text_layout_summary_label.text())
        self.assertIn("完整分辨率导出 18 px", inspector._text_layout_summary_label.text())
        self.assertFalse(inspector._legacy_text_layout_label.isHidden())
        self.assertFalse(inspector._text_layout_conversion_button.isHidden())

        conversions: list[str] = []
        layout_changes: list[object] = []
        inspector.overlayTextLayoutConversionRequested.connect(conversions.append)
        inspector.overlayTextLayoutChangeRequested.connect(
            lambda _overlay_id, spec: layout_changes.append(spec)
        )
        inspector._text_layout_conversion_button.click()
        inspector._text_image_font_size_spin.editingFinished.emit()

        self.assertEqual(conversions, [text.id])
        self.assertEqual(layout_changes, [])
        inspector.close()

    def test_inspector_hides_text_layout_controls_for_non_text_objects(self) -> None:
        shape = OverlayAnnotation(
            id="shape",
            image_id="shape-inspector",
            kind=OverlayAnnotationKind.RECT,
            start_px=Point(10, 10),
            end_px=Point(40, 30),
        )
        document = ImageDocument(
            id="shape-inspector",
            path="/tmp/shape.png",
            image_size=(80, 60),
            overlay_annotations=[shape],
        )
        inspector = CurrentObjectInspector()
        inspector.set_context(
            document,
            settings=AppSettings(),
            overlay_id=shape.id,
        )

        self.assertTrue(inspector._text_layout_group.isHidden())
        inspector.close()

    def test_inspector_long_category_does_not_create_a_wide_minimum(self) -> None:
        document = ImageDocument(id="long-group", path="/tmp/group.png", image_size=(80, 60))
        long_label = "Category_" + ("UnbrokenName" * 40)
        group = document.create_group(color="#2A9D8F", label=long_label)
        measurement = Measurement(
            id="line",
            image_id=document.id,
            fiber_group_id=group.id,
            mode="manual",
            line_px=Line(Point(5, 10), Point(45, 10)),
        )
        measurement.recalculate(None)
        document.measurements = [measurement]
        inspector = CurrentObjectInspector()
        inspector.resize(320, 500)
        inspector.set_context(
            document,
            settings=AppSettings(),
            measurement_ids=[measurement.id],
        )
        inspector.show()
        self.app.processEvents()

        self.assertLessEqual(inspector.minimumSizeHint().width(), 320)
        self.assertLessEqual(inspector._group_combo.minimumSizeHint().width(), 180)
        self.assertIn(long_label, inspector._group_combo.currentText())
        self.assertEqual(
            inspector._group_combo.toolTip(),
            inspector._group_combo.currentText(),
        )
        inspector.close()


if __name__ == "__main__":
    unittest.main()
