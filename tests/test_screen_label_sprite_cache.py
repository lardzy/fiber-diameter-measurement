from __future__ import annotations

import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PySide6.QtCore import QPointF
from PySide6.QtGui import QColor, QFont, QImage, QPainter
from PySide6.QtWidgets import QApplication, QWidget

from fdm.geometry import Line, Point
from fdm.models import ImageDocument, Measurement, ObjectAppearanceOverride
from fdm.settings import AppSettings, MeasurementLabelStyleSettings
import fdm.ui.rendering as rendering
from fdm.ui.screen_label_sprite_cache import (
    DEFAULT_SCREEN_LABEL_SPRITE_CACHE_BYTES,
    ScreenLabelSpriteCache,
    screen_label_sprite_cache,
)


class _SpriteRecordingPainter:
    def __init__(self, device) -> None:
        self._device = device
        self.draw_images: list[tuple[object, QImage]] = []
        self.rotations: list[float] = []
        self.translations: list[object] = []
        self.save_count = 0
        self.restore_count = 0
        self.draw_ellipses: list[tuple[object, float, float]] = []
        self.draw_static_texts: list[tuple[object, object]] = []

    def device(self):
        return self._device

    def setFont(self, _font) -> None:
        return

    def setBrush(self, _brush) -> None:
        return

    def setPen(self, _pen) -> None:
        return

    def drawEllipse(self, center, radius_x: float, radius_y: float) -> None:
        self.draw_ellipses.append((center, radius_x, radius_y))

    def drawStaticText(self, target, static_text) -> None:
        self.draw_static_texts.append((target, static_text))

    def drawImage(self, target, image: QImage) -> None:
        self.draw_images.append((target, image))

    def save(self) -> None:
        self.save_count += 1

    def restore(self) -> None:
        self.restore_count += 1

    def translate(self, value) -> None:
        self.translations.append(value)

    def rotate(self, angle: float) -> None:
        self.rotations.append(angle)


class ScreenLabelSpriteCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        screen_label_sprite_cache.clear(reset_stats=True)
        rendering._cached_measurement_label_font.cache_clear()

    def tearDown(self) -> None:
        screen_label_sprite_cache.clear(reset_stats=True)

    @staticmethod
    def _font(*, size: int = 24) -> QFont:
        font = QFont("Microsoft YaHei UI")
        font.setPixelSize(size)
        font.setBold(True)
        return font

    @staticmethod
    def _sprite(cache: ScreenLabelSpriteCache, text: str, **overrides):
        options = {
            "text": text,
            "font": ScreenLabelSpriteCacheTests._font(),
            "text_color": QColor("#F4F1DE"),
            "outline_color": QColor("#101820"),
            "background_color": QColor(16, 24, 32, 168),
            "device_pixel_ratio": 1.0,
            "arrangement_mode": "measurement-length",
        }
        options.update(overrides)
        return cache.get_or_create(**options)

    def test_default_cache_is_32_mib_and_reuses_complete_sprite(self) -> None:
        self.assertEqual(
            screen_label_sprite_cache.max_bytes,
            DEFAULT_SCREEN_LABEL_SPRITE_CACHE_BYTES,
        )

        first = self._sprite(screen_label_sprite_cache, "12.34 μm")
        repeated = self._sprite(screen_label_sprite_cache, "12.34 μm")

        self.assertIs(first, repeated)
        self.assertEqual(len(screen_label_sprite_cache), 1)
        self.assertEqual(screen_label_sprite_cache.byte_size, first.byte_size)
        self.assertEqual(screen_label_sprite_cache.stats().hits, 1)
        self.assertEqual(screen_label_sprite_cache.stats().misses, 1)

    def test_key_separates_font_color_background_dpr_and_arrangement(self) -> None:
        cache = ScreenLabelSpriteCache()
        base = self._sprite(cache, "测试")
        variants = [
            self._sprite(cache, "测试", font=self._font(size=96)),
            self._sprite(cache, "测试", text_color=QColor("#00FF00")),
            self._sprite(cache, "测试", background_color=None),
            self._sprite(cache, "测试", device_pixel_ratio=1.5),
            self._sprite(cache, "测试", arrangement_mode="measurement-length-parallel"),
        ]

        self.assertEqual(len(cache), 6)
        self.assertTrue(all(sprite is not base for sprite in variants))

    def test_sprites_preserve_100_125_and_150_percent_device_ratios(self) -> None:
        cache = ScreenLabelSpriteCache()
        sprites = [
            self._sprite(cache, "DPI 测试", device_pixel_ratio=dpr)
            for dpr in (1.0, 1.25, 1.5)
        ]

        self.assertEqual(
            [sprite.image.devicePixelRatio() for sprite in sprites],
            [1.0, 1.25, 1.5],
        )
        self.assertEqual(len(cache), 3)
        self.assertGreater(sprites[1].image.width(), sprites[0].image.width())
        self.assertGreater(sprites[2].image.width(), sprites[1].image.width())
        self.assertAlmostEqual(
            sprites[0].logical_width,
            sprites[2].logical_width,
            places=6,
        )

    def test_chinese_multiline_96px_sprite_contains_background_outline_and_text(self) -> None:
        cache = ScreenLabelSpriteCache()
        sprite = self._sprite(
            cache,
            "纤维直径\n第二行",
            font=self._font(size=96),
            device_pixel_ratio=1.5,
        )
        single_line = self._sprite(
            cache,
            "纤维直径",
            font=self._font(size=96),
            device_pixel_ratio=1.5,
        )

        self.assertEqual(sprite.image.devicePixelRatio(), 1.5)
        self.assertGreater(sprite.content_height, single_line.content_height)
        self.assertGreater(sprite.logical_width, 12.0)
        self.assertGreater(sprite.logical_height, 100.0)
        self.assertEqual(sprite.image.pixelColor(0, 0).alpha(), 168)
        alpha_pixels = sum(
            1
            for y in range(sprite.image.height())
            for x in range(sprite.image.width())
            if sprite.image.pixelColor(x, y).alpha() > 0
        )
        self.assertGreater(alpha_pixels, sprite.image.width())

        transparent = self._sprite(
            cache,
            "纤维直径\n第二行",
            font=self._font(size=96),
            device_pixel_ratio=1.5,
            background_color=None,
        )
        self.assertEqual(transparent.image.pixelColor(0, 0).alpha(), 0)
        self.assertTrue(
            any(
                transparent.image.pixelColor(x, y).alpha() > 0
                for y in range(transparent.image.height())
                for x in range(transparent.image.width())
            )
        )

    def test_lru_is_bounded_by_estimated_image_bytes(self) -> None:
        probe = ScreenLabelSpriteCache()
        sprites = [self._sprite(probe, text) for text in ("AAAA", "BBBB", "CCCC")]
        two_entry_budget = max(
            sprites[0].byte_size + sprites[1].byte_size,
            sprites[0].byte_size + sprites[2].byte_size,
            sprites[1].byte_size + sprites[2].byte_size,
        )
        cache = ScreenLabelSpriteCache(max_bytes=two_entry_budget)
        first = self._sprite(cache, "AAAA")
        self._sprite(cache, "BBBB")
        self.assertIs(self._sprite(cache, "AAAA"), first)
        self._sprite(cache, "CCCC")

        before = cache.stats()
        self.assertLessEqual(cache.byte_size, cache.max_bytes)
        self.assertLessEqual(len(cache), 2)
        self.assertGreaterEqual(before.evictions, 1)
        self._sprite(cache, "BBBB")
        self.assertEqual(cache.stats().misses, before.misses + 1)

        oversized = ScreenLabelSpriteCache(max_bytes=1)
        self._sprite(oversized, "too large")
        self.assertEqual(len(oversized), 0)
        self.assertEqual(oversized.byte_size, 0)

    def test_screen_length_label_uses_one_draw_image_and_reuses_sprite(self) -> None:
        widget = QWidget()
        painter = _SpriteRecordingPainter(widget)
        document = ImageDocument(id="image", path="/tmp/image.png", image_size=(240, 180))
        measurement = Measurement(
            id="line",
            image_id=document.id,
            fiber_group_id=None,
            mode="manual",
            line_px=Line(Point(20, 80), Point(200, 80)),
        )
        measurement.recalculate(None)
        settings = AppSettings(
            length_measurement_label_style=MeasurementLabelStyleSettings(
                enabled=True,
                font_family="Microsoft YaHei UI",
                font_size=24,
                color="#F4F1DE",
                background_enabled=True,
            )
        )

        for _ in range(2):
            rendering.draw_measurement_label(
                painter,
                measurement,
                document,
                settings,
                QPointF(20, 80),
                QPointF(200, 80),
            )

        self.assertEqual(len(painter.draw_images), 2)
        self.assertIs(painter.draw_images[0][1], painter.draw_images[1][1])
        self.assertEqual(screen_label_sprite_cache.stats().misses, 1)
        self.assertEqual(screen_label_sprite_cache.stats().hits, 1)

    def test_parallel_label_rotates_once_and_still_uses_one_draw_image(self) -> None:
        widget = QWidget()
        painter = _SpriteRecordingPainter(widget)
        document = ImageDocument(id="image", path="/tmp/image.png", image_size=(240, 180))
        measurement = Measurement(
            id="diagonal",
            image_id=document.id,
            fiber_group_id=None,
            mode="manual",
            line_px=Line(Point(20, 20), Point(180, 100)),
        )
        measurement.recalculate(None)
        settings = AppSettings(
            length_measurement_label_style=MeasurementLabelStyleSettings(
                enabled=True,
                font_family="Microsoft YaHei UI",
                font_size=24,
                color="#F4F1DE",
                background_enabled=True,
                parallel_to_line=True,
            )
        )

        rendering.draw_measurement_label(
            painter,
            measurement,
            document,
            settings,
            QPointF(20, 20),
            QPointF(180, 100),
        )

        self.assertEqual(len(painter.draw_images), 1)
        self.assertEqual(len(painter.rotations), 1)
        self.assertAlmostEqual(painter.rotations[0], 26.565, places=3)
        self.assertEqual(painter.save_count, 1)
        self.assertEqual(painter.restore_count, 1)

    def test_qimage_export_path_does_not_consult_screen_sprite_cache(self) -> None:
        target = QImage(240, 180, QImage.Format.Format_ARGB32_Premultiplied)
        target.fill(0)
        painter = QPainter(target)
        document = ImageDocument(id="image", path="/tmp/image.png", image_size=(240, 180))
        measurement = Measurement(
            id="line",
            image_id=document.id,
            fiber_group_id=None,
            mode="manual",
            line_px=Line(Point(20, 80), Point(200, 80)),
        )
        measurement.recalculate(None)
        try:
            with patch.object(
                screen_label_sprite_cache,
                "get_or_create",
                wraps=screen_label_sprite_cache.get_or_create,
            ) as get_or_create:
                rendering.draw_measurement_label(
                    painter,
                    measurement,
                    document,
                    AppSettings(),
                    QPointF(20, 80),
                    QPointF(200, 80),
                )
            get_or_create.assert_not_called()
        finally:
            painter.end()

        self.assertTrue(
            any(
                target.pixelColor(x, y).alpha() > 0
                for y in range(target.height())
                for x in range(target.width())
            )
        )

    def test_area_and_polyline_screen_labels_each_use_one_draw_image(self) -> None:
        widget = QWidget()
        painter = _SpriteRecordingPainter(widget)
        document = ImageDocument(id="image", path="/tmp/image.png", image_size=(240, 180))
        area = Measurement(
            id="area",
            image_id=document.id,
            fiber_group_id=None,
            mode="polygon_area",
            measurement_kind="area",
            polygon_px=[Point(20, 20), Point(100, 20), Point(100, 100), Point(20, 100)],
        )
        polyline = Measurement(
            id="polyline",
            image_id=document.id,
            fiber_group_id=None,
            mode="polyline",
            measurement_kind="polyline",
            polyline_px=[Point(120, 20), Point(180, 60), Point(210, 110)],
        )
        area.recalculate(None)
        polyline.recalculate(None)

        rendering.draw_area_measurement_label(
            painter,
            area,
            document,
            AppSettings(),
            QPointF(60, 60),
        )
        rendering.draw_polyline_measurement_label(
            painter,
            polyline,
            document,
            AppSettings(),
            [QPointF(120, 20), QPointF(180, 60), QPointF(210, 110)],
            lambda point: QPointF(point.x, point.y),
        )

        self.assertEqual(len(painter.draw_images), 2)

    def test_single_count_number_uses_one_complete_sprite_per_paint(self) -> None:
        widget = QWidget()
        painter = _SpriteRecordingPainter(widget)
        measurement = Measurement(
            id="count",
            image_id="image",
            fiber_group_id=None,
            mode="manual",
            measurement_kind="count",
            point_px=Point(80, 60),
        )
        document = ImageDocument(
            id="image",
            path="/tmp/image.png",
            image_size=(240, 180),
            measurements=[measurement],
        )
        settings = AppSettings(show_count_numbers=True)

        for _ in range(2):
            rendering.draw_measurements(
                painter,
                document,
                lambda point: QPointF(point.x, point.y),
                settings,
                line_width=2.0,
                endpoint_radius=5.0,
                selected_measurement_id=measurement.id,
                use_sprite_cache=True,
            )

        self.assertEqual(len(painter.draw_images), 2)
        self.assertIs(painter.draw_images[0][1], painter.draw_images[1][1])
        self.assertEqual(screen_label_sprite_cache.stats().misses, 1)
        self.assertEqual(screen_label_sprite_cache.stats().hits, 1)
        self.assertEqual(painter.draw_static_texts, [])

    def test_batched_count_numbers_each_draw_once_and_reuse_complete_sprites(self) -> None:
        widget = QWidget()
        painter = _SpriteRecordingPainter(widget)
        measurements = [
            Measurement(
                id=f"count-{number}",
                image_id="image",
                fiber_group_id=None,
                mode="manual",
                measurement_kind="count",
                point_px=Point(30 + number * 35, 60),
            )
            for number in range(1, 4)
        ]
        document = ImageDocument(
            id="image",
            path="/tmp/image.png",
            image_size=(240, 180),
            measurements=measurements,
        )
        settings = AppSettings(show_count_numbers=True)

        for _ in range(2):
            rendering.draw_measurements(
                painter,
                document,
                lambda point: QPointF(point.x, point.y),
                settings,
                line_width=2.0,
                endpoint_radius=5.0,
                use_sprite_cache=True,
            )

        self.assertEqual(len(painter.draw_images), 6)
        first_frame = [image for _target, image in painter.draw_images[:3]]
        second_frame = [image for _target, image in painter.draw_images[3:]]
        self.assertTrue(
            all(first is second for first, second in zip(first_frame, second_frame))
        )
        self.assertEqual(screen_label_sprite_cache.stats().misses, 3)
        self.assertEqual(screen_label_sprite_cache.stats().hits, 3)
        self.assertEqual(painter.draw_static_texts, [])

    def test_count_sprite_opt_out_uses_exact_static_text_path(self) -> None:
        widget = QWidget()
        painter = _SpriteRecordingPainter(widget)
        measurement = Measurement(
            id="count",
            image_id="image",
            fiber_group_id=None,
            mode="manual",
            measurement_kind="count",
            point_px=Point(80, 60),
        )
        document = ImageDocument(
            id="image",
            path="/tmp/image.png",
            image_size=(240, 180),
            measurements=[measurement],
        )

        with patch.object(
            screen_label_sprite_cache,
            "get_or_create",
            wraps=screen_label_sprite_cache.get_or_create,
        ) as get_or_create:
            rendering.draw_measurements(
                painter,
                document,
                lambda point: QPointF(point.x, point.y),
                AppSettings(show_count_numbers=True),
                line_width=2.0,
                endpoint_radius=5.0,
                use_sprite_cache=False,
            )

        get_or_create.assert_not_called()
        self.assertEqual(painter.draw_images, [])
        self.assertEqual(len(painter.draw_static_texts), 5)

    def test_qimage_count_export_does_not_consult_screen_sprite_cache(self) -> None:
        target = QImage(240, 180, QImage.Format.Format_ARGB32_Premultiplied)
        target.fill(0)
        painter = QPainter(target)
        measurement = Measurement(
            id="count",
            image_id="image",
            fiber_group_id=None,
            mode="manual",
            measurement_kind="count",
            point_px=Point(80, 60),
        )
        document = ImageDocument(
            id="image",
            path="/tmp/image.png",
            image_size=(240, 180),
            measurements=[measurement],
        )
        try:
            with patch.object(
                screen_label_sprite_cache,
                "get_or_create",
                wraps=screen_label_sprite_cache.get_or_create,
            ) as get_or_create:
                rendering.draw_measurements(
                    painter,
                    document,
                    lambda point: QPointF(point.x, point.y),
                    AppSettings(show_count_numbers=True),
                    line_width=2.0,
                    endpoint_radius=5.0,
                )
            get_or_create.assert_not_called()
        finally:
            painter.end()

        self.assertTrue(
            any(
                target.pixelColor(x, y).alpha() > 0
                for y in range(target.height())
                for x in range(target.width())
            )
        )

    def test_count_object_overrides_drive_sprite_key_and_marker_offset(self) -> None:
        widget = QWidget()
        painter = _SpriteRecordingPainter(widget)
        first = Measurement(
            id="first",
            image_id="image",
            fiber_group_id=None,
            mode="manual",
            measurement_kind="count",
            point_px=Point(40, 60),
            appearance=ObjectAppearanceOverride(
                text_color="#FF3355",
                font_family="Arial",
                font_size=31,
                marker_scale=1.0,
            ),
        )
        second = Measurement(
            id="second",
            image_id="image",
            fiber_group_id=None,
            mode="manual",
            measurement_kind="count",
            point_px=Point(40, 60),
            appearance=ObjectAppearanceOverride(
                text_color="#33AAFF",
                font_family="Microsoft YaHei UI",
                font_size=42,
                marker_scale=2.0,
            ),
        )
        document = ImageDocument(
            id="image",
            path="/tmp/image.png",
            image_size=(240, 180),
            measurements=[first, second],
        )

        with patch.object(
            screen_label_sprite_cache,
            "get_or_create",
            wraps=screen_label_sprite_cache.get_or_create,
        ) as get_or_create:
            rendering.draw_measurements(
                painter,
                document,
                lambda point: QPointF(point.x, point.y),
                AppSettings(show_count_numbers=True),
                line_width=2.0,
                endpoint_radius=5.0,
                count_numbers={"first": 7, "second": 7},
                use_sprite_cache=True,
            )

        self.assertEqual(get_or_create.call_count, 2)
        first_call, second_call = [call.kwargs for call in get_or_create.call_args_list]
        self.assertEqual(first_call["text"], "7")
        self.assertEqual(second_call["text"], "7")
        self.assertEqual(first_call["font"].family(), "Arial")
        self.assertEqual(first_call["font"].pointSize(), 31)
        self.assertEqual(first_call["text_color"].name(), "#ff3355")
        self.assertEqual(second_call["font"].family(), "Microsoft YaHei UI")
        self.assertEqual(second_call["font"].pointSize(), 42)
        self.assertEqual(second_call["text_color"].name(), "#33aaff")
        self.assertEqual(len(painter.draw_images), 2)
        first_target, second_target = [
            target for target, _image in painter.draw_images
        ]
        self.assertGreater(second_target.x(), first_target.x())
        self.assertLess(second_target.y(), first_target.y())


if __name__ == "__main__":
    unittest.main()
