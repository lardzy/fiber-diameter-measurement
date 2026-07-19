from __future__ import annotations

import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PySide6.QtWidgets import QApplication

from fdm.settings import AppSettings
from fdm.ui.canvas import DocumentCanvas


class CanvasVisualSettingsSignatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.canvas = DocumentCanvas()

    def tearDown(self) -> None:
        self.canvas.close()

    def test_visually_equal_settings_do_not_invalidate_overlay_tiles(self) -> None:
        initial_generation = self.canvas._overlay_style_generation  # noqa: SLF001
        equivalent = AppSettings(
            default_measurement_color="#2a9d8f",
            recent_export_dir="/a/non-visual/change",
        )

        with (
            patch.object(self.canvas, "_invalidate_all_overlay_tiles") as invalidate,
            patch.object(self.canvas, "update") as update,
        ):
            self.canvas.set_settings(equivalent)

        invalidate.assert_not_called()
        update.assert_not_called()
        self.assertEqual(
            self.canvas._overlay_style_generation,  # noqa: SLF001
            initial_generation,
        )

    def test_reapplying_same_tool_mode_is_a_strict_visual_noop(self) -> None:
        with patch.object(self.canvas, "update") as update:
            self.canvas.set_tool_mode("select")
        update.assert_not_called()

    def test_overlay_and_magic_visual_settings_request_repaint_without_tile_eviction(
        self,
    ) -> None:
        settings = AppSettings(
            text_color="#FF0000",
            overlay_line_width=4.0,
            magic_segment_fill_draft_holes_enabled=(
                not AppSettings().magic_segment_fill_draft_holes_enabled
            ),
        )
        with (
            patch.object(self.canvas, "_invalidate_all_overlay_tiles") as invalidate,
            patch.object(self.canvas, "update") as update,
        ):
            self.canvas.set_settings(settings)

        invalidate.assert_not_called()
        update.assert_called_once_with()

    def test_in_place_visual_mutation_is_detected(self) -> None:
        settings = AppSettings()
        self.canvas.set_settings(settings)
        initial_generation = self.canvas._overlay_style_generation  # noqa: SLF001
        settings.length_measurement_label_style.font_size += 1

        with patch.object(
            self.canvas,
            "_invalidate_all_overlay_tiles",
        ) as invalidate:
            self.canvas.set_settings(settings)

        invalidate.assert_called_once()
        self.assertEqual(
            self.canvas._overlay_style_generation,  # noqa: SLF001
            initial_generation + 1,
        )

    def test_every_cached_measurement_visual_family_invalidates(self) -> None:
        def length_background(settings: AppSettings) -> None:
            settings.length_measurement_label_style.background_enabled = (
                not settings.length_measurement_label_style.background_enabled
            )

        def area_visibility(settings: AppSettings) -> None:
            settings.area_measurement_label_style.enabled = (
                not settings.area_measurement_label_style.enabled
            )

        mutations = {
            "length label visibility": lambda settings: setattr(
                settings.length_measurement_label_style,
                "enabled",
                not settings.length_measurement_label_style.enabled,
            ),
            "length label font": lambda settings: setattr(
                settings.length_measurement_label_style,
                "font_family",
                "SimSun",
            ),
            "length label size": lambda settings: setattr(
                settings.length_measurement_label_style,
                "font_size",
                settings.length_measurement_label_style.font_size + 1,
            ),
            "length label color": lambda settings: setattr(
                settings.length_measurement_label_style,
                "color",
                "#112233",
            ),
            "length label decimals": lambda settings: setattr(
                settings.length_measurement_label_style,
                "decimals",
                settings.length_measurement_label_style.decimals + 1,
            ),
            "length label background": length_background,
            "length label arrangement": lambda settings: setattr(
                settings.length_measurement_label_style,
                "parallel_to_line",
                not settings.length_measurement_label_style.parallel_to_line,
            ),
            "area label visibility": area_visibility,
            "area label font": lambda settings: setattr(
                settings.area_measurement_label_style,
                "font_family",
                "SimHei",
            ),
            "area label color": lambda settings: setattr(
                settings.area_measurement_label_style,
                "color",
                "#00ff00",
            ),
            "area label background": lambda settings: setattr(
                settings.area_measurement_label_style,
                "background_enabled",
                not settings.area_measurement_label_style.background_enabled,
            ),
            "count visibility": lambda settings: setattr(
                settings,
                "show_count_numbers",
                not settings.show_count_numbers,
            ),
            "count font": lambda settings: setattr(
                settings,
                "count_number_font_family",
                "SimSun",
            ),
            "count size": lambda settings: setattr(
                settings,
                "count_number_font_size",
                settings.count_number_font_size + 1,
            ),
            "count color": lambda settings: setattr(
                settings,
                "count_number_color",
                "#0000ff",
            ),
            "endpoint style": lambda settings: setattr(
                settings,
                "measurement_endpoint_style",
                "circle",
            ),
            "default object color": lambda settings: setattr(
                settings,
                "default_measurement_color",
                "#abcdef",
            ),
        }

        for label, mutate in mutations.items():
            with self.subTest(label=label):
                canvas = DocumentCanvas()
                settings = AppSettings()
                mutate(settings)
                initial_generation = canvas._overlay_style_generation  # noqa: SLF001
                try:
                    with patch.object(
                        canvas,
                        "_invalidate_all_overlay_tiles",
                    ) as invalidate:
                        canvas.set_settings(settings)
                    invalidate.assert_called_once()
                    self.assertEqual(
                        canvas._overlay_style_generation,  # noqa: SLF001
                        initial_generation + 1,
                    )
                finally:
                    canvas.close()


if __name__ == "__main__":
    unittest.main()
