from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm.geometry import Line, Point
from fdm.models import Calibration, CalibrationPreset, ImageDocument, Measurement, ProjectState, TextAnnotation, new_id
from fdm.project_io import ProjectIO
from fdm.settings import AppSettings, AppSettingsIO


class ModelsProjectIOTests(unittest.TestCase):
    def test_calibration_conversion(self) -> None:
        calibration = Calibration(
            mode="image_scale",
            pixels_per_unit=20.0,
            unit="um",
            source_label="图内标定",
        )
        self.assertAlmostEqual(calibration.px_to_unit(100.0), 5.0)
        self.assertAlmostEqual(calibration.unit_to_px(2.5), 50.0)

    def test_project_roundtrip(self) -> None:
        document = ImageDocument(
            id=new_id("image"),
            path="/tmp/fiber.png",
            image_size=(640, 480),
        )
        group = document.ensure_default_group()
        document.initialize_runtime_state()
        document.calibration = Calibration(
            mode="preset",
            pixels_per_unit=4.0,
            unit="um",
            source_label="40x",
        )
        measurement = Measurement(
            id=new_id("meas"),
            image_id=document.id,
            fiber_group_id=group.id,
            mode="manual",
            line_px=Line(Point(10, 10), Point(18, 10)),
            confidence=1.0,
            status="manual",
        )
        document.add_measurement(measurement)
        document.add_text_annotation(
            TextAnnotation(
                id=new_id("text"),
                image_id=document.id,
                content="纤维A",
                anchor_px=Point(30, 40),
            )
        )
        document.scale_overlay_anchor = Point(80, 90)
        project = ProjectState(version="0.1.0", documents=[document])

        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "demo.fdmproj"
            ProjectIO.save(project, path)
            loaded = ProjectIO.load(path)

        self.assertEqual(loaded.version, "0.1.0")
        self.assertEqual(len(loaded.documents), 1)
        loaded_document = loaded.documents[0]
        self.assertEqual(loaded_document.path, "/tmp/fiber.png")
        self.assertEqual(len(loaded_document.measurements), 1)
        self.assertEqual(len(loaded_document.text_annotations), 1)
        self.assertEqual(loaded_document.text_annotations[0].content, "纤维A")
        self.assertIsNotNone(loaded_document.scale_overlay_anchor)
        self.assertAlmostEqual(loaded_document.scale_overlay_anchor.x, 80.0)
        self.assertEqual(loaded_document.sorted_groups()[0].number, 1)
        self.assertAlmostEqual(loaded_document.measurements[0].diameter_px or 0.0, 8.0)
        self.assertAlmostEqual(loaded_document.measurements[0].diameter_unit or 0.0, 2.0)

    def test_calibration_preset_roundtrip_keeps_source_distances(self) -> None:
        preset = CalibrationPreset(
            name="40x",
            pixels_per_unit=12.5,
            unit="um",
            pixel_distance=250.0,
            actual_distance=20.0,
            computed_pixels_per_unit=12.5,
        )
        payload = preset.to_dict()
        loaded = CalibrationPreset.from_dict(payload)

        self.assertEqual(loaded.name, "40x")
        self.assertAlmostEqual(loaded.resolved_pixels_per_unit(), 12.5)
        self.assertAlmostEqual(loaded.pixel_distance or 0.0, 250.0)
        self.assertAlmostEqual(loaded.actual_distance or 0.0, 20.0)

    def test_document_allows_uncategorized_measurement_by_default(self) -> None:
        document = ImageDocument(
            id=new_id("image"),
            path="/tmp/fiber_uncategorized.png",
            image_size=(320, 200),
        )
        document.initialize_runtime_state()

        measurement = Measurement(
            id=new_id("meas"),
            image_id=document.id,
            fiber_group_id=None,
            mode="manual",
            line_px=Line(Point(10, 10), Point(30, 10)),
        )
        document.add_measurement(measurement)

        self.assertEqual(len(document.fiber_groups), 0)
        self.assertIsNone(document.active_group_id)
        self.assertIsNone(document.measurements[0].fiber_group_id)

    def test_uncategorized_entry_hides_after_first_group_when_document_is_empty(self) -> None:
        document = ImageDocument(
            id=new_id("image"),
            path="/tmp/fiber_empty.png",
            image_size=(320, 200),
        )
        document.initialize_runtime_state()
        self.assertTrue(document.should_show_uncategorized_entry())

        group = document.create_group(color="#1F7A8C", label="棉")
        document.set_active_group(group.id)

        self.assertFalse(document.should_show_uncategorized_entry())

    def test_remove_group_moves_measurements_to_uncategorized_and_renumbers(self) -> None:
        document = ImageDocument(
            id=new_id("image"),
            path="/tmp/fiber_groups.png",
            image_size=(320, 200),
        )
        document.initialize_runtime_state()
        first = document.create_group(color="#1F7A8C", label="棉")
        second = document.create_group(color="#E07A5F", label="麻")
        document.set_active_group(first.id)
        measurement = Measurement(
            id=new_id("meas"),
            image_id=document.id,
            fiber_group_id=first.id,
            mode="manual",
            line_px=Line(Point(0, 0), Point(20, 0)),
        )
        document.add_measurement(measurement)

        removed = document.remove_group_to_uncategorized(first.id)

        self.assertTrue(removed)
        self.assertIsNone(document.measurements[0].fiber_group_id)
        self.assertEqual(len(document.fiber_groups), 1)
        self.assertEqual(document.fiber_groups[0].id, second.id)
        self.assertEqual(document.fiber_groups[0].number, 1)
        self.assertTrue(document.should_show_uncategorized_entry())

    def test_hide_uncategorized_entry_only_when_empty_and_other_groups_exist(self) -> None:
        document = ImageDocument(
            id=new_id("image"),
            path="/tmp/fiber_hide_uncategorized.png",
            image_size=(320, 200),
        )
        document.initialize_runtime_state()
        group = document.create_group(color="#1F7A8C", label="棉")
        document.set_active_group(None)

        self.assertTrue(document.hide_uncategorized_entry())
        self.assertEqual(document.active_group_id, group.id)
        self.assertFalse(document.should_show_uncategorized_entry())

        document.set_active_group(None)
        document.add_measurement(
            Measurement(
                id=new_id("meas"),
                image_id=document.id,
                fiber_group_id=None,
                mode="manual",
                line_px=Line(Point(10, 10), Point(40, 10)),
            )
        )

        self.assertFalse(document.hide_uncategorized_entry())

    def test_app_settings_roundtrip_uses_user_writable_path(self) -> None:
        settings = AppSettings(
            show_measurement_labels=False,
            measurement_label_font_size=22,
            measurement_label_decimals=2,
            measurement_label_parallel_to_line=True,
            measurement_label_background_enabled=False,
            measurement_endpoint_style="bar",
            text_font_size=26,
            text_color="#123456",
        )

        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "settings.json"
            saved_path = AppSettingsIO.save(settings, path)
            loaded = AppSettingsIO.load(saved_path)

        self.assertEqual(saved_path, path)
        self.assertFalse(loaded.show_measurement_labels)
        self.assertEqual(loaded.measurement_label_font_size, 22)
        self.assertEqual(loaded.measurement_label_decimals, 2)
        self.assertTrue(loaded.measurement_label_parallel_to_line)
        self.assertFalse(loaded.measurement_label_background_enabled)
        self.assertEqual(loaded.measurement_endpoint_style, "bar")
        self.assertEqual(loaded.text_font_size, 26)
        self.assertEqual(loaded.text_color, "#123456")


if __name__ == "__main__":
    unittest.main()
