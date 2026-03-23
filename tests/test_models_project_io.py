from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm.geometry import Line, Point
from fdm.models import Calibration, CalibrationPreset, ImageDocument, Measurement, ProjectState, TextAnnotation, new_id
from fdm.project_io import ProjectIO
from fdm.services.area_inference import AreaInferenceService
from fdm.services.area_inference import normalize_area_result_label, parse_area_model_labels
from fdm.settings import (
    AppSettings,
    AppSettingsIO,
    application_root,
    default_area_model_mappings,
)


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
        self.assertAlmostEqual(calibration.px_area_to_unit(400.0), 1.0)

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

    def test_auto_area_import_can_hide_uncategorized_entry_immediately(self) -> None:
        document = ImageDocument(
            id=new_id("image"),
            path="/tmp/area_auto_hide_uncategorized.png",
            image_size=(320, 200),
        )
        document.initialize_runtime_state()

        self.assertTrue(document.should_show_uncategorized_entry())
        self.assertIsNone(document.active_group_id)

        group = document.ensure_group_for_label("棉", color="#1F7A8C")
        measurement = Measurement(
            id=new_id("meas"),
            image_id=document.id,
            fiber_group_id=group.id,
            mode="auto_instance",
            measurement_kind="area",
            polygon_px=[Point(10, 10), Point(50, 10), Point(50, 50), Point(10, 50)],
        )
        document.add_measurement(measurement)
        document.select_measurement(None)

        self.assertTrue(document.hide_uncategorized_entry())
        self.assertFalse(document.should_show_uncategorized_entry())
        self.assertIsNotNone(document.active_group_id)
        self.assertEqual(len(document.uncategorized_measurements()), 0)

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

    def test_app_settings_store_runtime_area_paths_relative_to_application_root(self) -> None:
        runtime_root = application_root() / "runtime"
        settings = AppSettings(
            area_weights_dir=str((runtime_root / "area-models").resolve()),
            area_vendor_root=str((runtime_root / "area-infer" / "vendor" / "yolact").resolve()),
            area_worker_python=str(Path(sys.executable).resolve()),
        )

        payload = settings.to_dict()

        self.assertEqual(payload["area_weights_dir"], "runtime/area-models")
        self.assertEqual(payload["area_vendor_root"], "runtime/area-infer/vendor/yolact")
        self.assertEqual(payload["area_worker_python"], "")

    def test_app_settings_resolve_relative_area_paths_back_to_application_root(self) -> None:
        settings = AppSettings(
            area_weights_dir="runtime/area-models",
            area_vendor_root="runtime/area-infer/vendor/yolact",
            area_worker_python="",
        )

        self.assertEqual(settings.resolved_area_weights_dir(), (application_root() / "runtime" / "area-models").resolve())
        self.assertEqual(
            settings.resolved_area_vendor_root(),
            (application_root() / "runtime" / "area-infer" / "vendor" / "yolact").resolve(),
        )
        self.assertEqual(settings.resolved_area_worker_program(), "")

    def test_area_inference_service_uses_auto_worker_when_settings_worker_is_blank(self) -> None:
        service = AreaInferenceService()
        settings = AppSettings(area_worker_python="")

        worker_command = service._worker_command(settings)

        self.assertEqual(worker_command[0], sys.executable)
        self.assertTrue(worker_command[1].endswith("area_worker.py"))

    def test_app_settings_migrate_missing_absolute_worker_path_back_to_auto(self) -> None:
        payload = {
            "area_worker_python": str((Path("/tmp") / "missing-python-for-area-worker.exe").resolve()),
        }

        settings = AppSettings.from_dict(payload)

        self.assertEqual(settings.area_worker_python, "")

    def test_area_measurement_roundtrip_keeps_polygon_and_area_unit(self) -> None:
        document = ImageDocument(
            id=new_id("image"),
            path="/tmp/fiber_area_roundtrip.png",
            image_size=(200, 160),
        )
        document.initialize_runtime_state()
        document.calibration = Calibration(
            mode="preset",
            pixels_per_unit=10.0,
            unit="um",
            source_label="demo",
        )
        document.add_measurement(
            Measurement(
                id=new_id("meas"),
                image_id=document.id,
                fiber_group_id=None,
                mode="polygon_area",
                measurement_kind="area",
                polygon_px=[Point(0, 0), Point(20, 0), Point(20, 10), Point(0, 10)],
            )
        )
        project = ProjectState(version="0.1.0", documents=[document])

        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "area_roundtrip.fdmproj"
            ProjectIO.save(project, path)
            loaded = ProjectIO.load(path)

        measurement = loaded.documents[0].measurements[0]
        self.assertEqual(measurement.measurement_kind, "area")
        self.assertEqual(len(measurement.polygon_px), 4)
        self.assertAlmostEqual(measurement.area_px or 0.0, 200.0)
        self.assertAlmostEqual(measurement.area_unit or 0.0, 2.0)

    def test_parse_area_model_labels_applies_aliases_and_deduplicates(self) -> None:
        self.assertEqual(parse_area_model_labels("棉-粘-莱-粘"), ["棉", "粘纤", "莱赛尔"])

    def test_normalize_area_result_label_swaps_known_reversed_models(self) -> None:
        self.assertEqual(normalize_area_result_label("棉-莱赛尔", "棉"), "莱赛尔")
        self.assertEqual(normalize_area_result_label("棉-莱赛尔", "莱赛尔"), "棉")
        self.assertEqual(normalize_area_result_label("粘纤-莱赛尔", "粘纤"), "莱赛尔")
        self.assertEqual(normalize_area_result_label("粘纤-莱赛尔", "莱赛尔"), "粘纤")

    def test_normalize_area_result_label_keeps_alias_and_non_swapped_models(self) -> None:
        self.assertEqual(normalize_area_result_label("棉-粘-莱-莫", "粘"), "粘纤")
        self.assertEqual(normalize_area_result_label("棉-莫代尔", "莫"), "莫代尔")
        self.assertEqual(normalize_area_result_label("棉-莫代尔", "棉"), "棉")

    def test_default_area_model_mappings_match_reference_defaults(self) -> None:
        mappings = default_area_model_mappings()
        mapping_dict = {item.model_name: item.model_file for item in mappings}
        self.assertEqual(mapping_dict["棉-莱赛尔"], "b_c1_1.3.pth")
        self.assertEqual(mapping_dict["粘纤-莱赛尔"], "b_v1_1.3.pth")
        self.assertEqual(mapping_dict["棉-粘-莱-莫"], "b_cvlm_1.3.pth")


if __name__ == "__main__":
    unittest.main()
