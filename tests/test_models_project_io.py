from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import json
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm.geometry import Line, Point
from fdm.models import (
    Calibration,
    CalibrationPreset,
    ImageDocument,
    Measurement,
    OverlayAnnotation,
    OverlayAnnotationKind,
    ProjectGroupTemplate,
    ProjectState,
    TextAnnotation,
    new_id,
    project_assets_root,
)
from fdm.project_io import ProjectIO
from fdm.services.area_inference import AreaInferenceService
from fdm.services.area_inference import normalize_area_result_label, parse_area_model_labels
from fdm.settings import (
    AppSettings,
    AppSettingsIO,
    FocusStackProfile,
    MagicSegmentModelVariant,
    MeasurementEndpointStyle,
    OpenImageViewMode,
    ScaleOverlayPlacementMode,
    ScaleOverlayStyle,
    application_root,
    bundle_resource_root,
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

    def test_project_roundtrip_preserves_overlay_annotations(self) -> None:
        document = ImageDocument(
            id=new_id("image"),
            path="/tmp/overlay_roundtrip.png",
            image_size=(320, 240),
        )
        document.initialize_runtime_state()
        document.add_overlay_annotation(
            OverlayAnnotation(
                id=new_id("overlay"),
                image_id=document.id,
                kind=OverlayAnnotationKind.RECT,
                start_px=Point(10, 12),
                end_px=Point(70, 82),
            )
        )
        document.add_overlay_annotation(
            OverlayAnnotation(
                id=new_id("overlay"),
                image_id=document.id,
                kind=OverlayAnnotationKind.ARROW,
                start_px=Point(40, 44),
                end_px=Point(120, 84),
            )
        )
        project = ProjectState(version="0.1.0", documents=[document])

        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "overlay_demo.fdmproj"
            ProjectIO.save(project, path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            loaded = ProjectIO.load(path)

        self.assertIn("overlay_annotations", payload["documents"][0])
        self.assertNotIn("text_annotations", payload["documents"][0])
        loaded_document = loaded.documents[0]
        self.assertEqual(
            [annotation.kind for annotation in loaded_document.overlay_annotations],
            [OverlayAnnotationKind.RECT, OverlayAnnotationKind.ARROW],
        )

    def test_image_document_loads_legacy_text_annotations_into_overlay_annotations(self) -> None:
        payload = {
            "id": new_id("image"),
            "path": "/tmp/legacy_text.png",
            "image_size": [200, 100],
            "measurements": [],
            "fiber_groups": [],
            "text_annotations": [
                {
                    "id": "text_legacy",
                    "image_id": "image_legacy",
                    "content": "旧文字",
                    "anchor_px": {"x": 18, "y": 22},
                }
            ],
            "selected_text_id": "text_legacy",
        }

        document = ImageDocument.from_dict(payload)

        self.assertEqual(len(document.overlay_annotations), 1)
        self.assertEqual(document.overlay_annotations[0].kind, OverlayAnnotationKind.TEXT)
        self.assertEqual(document.selected_overlay_id, "text_legacy")

    def test_project_roundtrip_preserves_project_asset_document_source_type(self) -> None:
        document = ImageDocument(
            id=new_id("image"),
            path="captures/capture_20260326_01.png",
            image_size=(320, 240),
            source_type="project_asset",
        )
        document.initialize_runtime_state()
        project = ProjectState(version="0.1.0", documents=[document])

        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "demo.fdmproj"
            ProjectIO.save(project, path)
            loaded = ProjectIO.load(path)

        self.assertEqual(loaded.documents[0].source_type, "project_asset")
        self.assertEqual(loaded.documents[0].path, "captures/capture_20260326_01.png")
        self.assertEqual(
            loaded.documents[0].resolved_path(path),
            (project_assets_root(path) / "captures" / "capture_20260326_01.png").resolve(),
        )

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
            scale_overlay_style=ScaleOverlayStyle.TICKS,
            scale_overlay_length_value=220.0,
            scale_overlay_color="#ABCDEF",
            scale_overlay_text_color="#654321",
            scale_overlay_font_size=21,
            text_font_size=26,
            text_color="#123456",
            overlay_line_color="#FFAA00",
            overlay_line_width=3.5,
            focus_stack_profile=FocusStackProfile.SHARP,
            focus_stack_sharpen_strength=60,
            magic_segment_model_variant=MagicSegmentModelVariant.EDGE_SAM,
            main_window_geometry="Zm9v",
            main_window_is_maximized=True,
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
        self.assertEqual(loaded.scale_overlay_style, ScaleOverlayStyle.TICKS)
        self.assertAlmostEqual(loaded.scale_overlay_length_value, 220.0)
        self.assertEqual(loaded.scale_overlay_color, "#ABCDEF")
        self.assertEqual(loaded.scale_overlay_text_color, "#654321")
        self.assertEqual(loaded.scale_overlay_font_size, 21)
        self.assertEqual(loaded.text_font_size, 26)
        self.assertEqual(loaded.text_color, "#123456")
        self.assertEqual(loaded.overlay_line_color, "#FFAA00")
        self.assertAlmostEqual(loaded.overlay_line_width, 3.5)
        self.assertEqual(loaded.focus_stack_profile, FocusStackProfile.SHARP)
        self.assertEqual(loaded.focus_stack_sharpen_strength, 60)
        self.assertEqual(loaded.magic_segment_model_variant, MagicSegmentModelVariant.EDGE_SAM)
        self.assertEqual(loaded.main_window_geometry, "Zm9v")
        self.assertTrue(loaded.main_window_is_maximized)

    def test_app_settings_from_dict_defaults_new_overlay_and_focus_fields(self) -> None:
        settings = AppSettings.from_dict({})

        self.assertEqual(settings.measurement_label_color, "#00FF00")
        self.assertEqual(settings.measurement_label_decimals, 2)
        self.assertFalse(settings.measurement_label_background_enabled)
        self.assertEqual(settings.measurement_endpoint_style, MeasurementEndpointStyle.BAR)
        self.assertEqual(settings.open_image_view_mode, OpenImageViewMode.FIT)
        self.assertEqual(settings.scale_overlay_placement_mode, ScaleOverlayPlacementMode.BOTTOM_RIGHT)
        self.assertEqual(settings.scale_overlay_style, ScaleOverlayStyle.TICKS)
        self.assertAlmostEqual(settings.scale_overlay_length_value, 50.0)
        self.assertEqual(settings.scale_overlay_color, "#FF0000")
        self.assertEqual(settings.scale_overlay_text_color, "#FF0000")
        self.assertEqual(settings.scale_overlay_font_size, 18)
        self.assertEqual(settings.overlay_line_color, "#F7F4EA")
        self.assertAlmostEqual(settings.overlay_line_width, 2.5)
        self.assertEqual(settings.focus_stack_profile, FocusStackProfile.BALANCED)
        self.assertEqual(settings.focus_stack_sharpen_strength, 35)
        self.assertEqual(settings.magic_segment_model_variant, MagicSegmentModelVariant.EDGE_SAM_3X)
        self.assertEqual(settings.main_window_geometry, "")
        self.assertFalse(settings.main_window_is_maximized)

    def test_app_settings_clamp_new_overlay_and_focus_fields(self) -> None:
        settings = AppSettings.from_dict(
            {
                "measurement_label_decimals": 99,
                "measurement_endpoint_style": "unknown",
                "open_image_view_mode": "unknown",
                "scale_overlay_placement_mode": "unknown",
                "scale_overlay_style": "unknown",
                "scale_overlay_length_value": 0,
                "scale_overlay_font_size": 999,
                "overlay_line_width": 1000,
                "focus_stack_profile": "unknown",
                "focus_stack_sharpen_strength": 1000,
                "magic_segment_model_variant": "unknown",
            }
        )

        self.assertEqual(settings.measurement_label_decimals, 8)
        self.assertEqual(settings.measurement_endpoint_style, MeasurementEndpointStyle.BAR)
        self.assertEqual(settings.open_image_view_mode, OpenImageViewMode.FIT)
        self.assertEqual(settings.scale_overlay_placement_mode, ScaleOverlayPlacementMode.BOTTOM_RIGHT)
        self.assertEqual(settings.scale_overlay_style, ScaleOverlayStyle.TICKS)
        self.assertAlmostEqual(settings.scale_overlay_length_value, 0.01)
        self.assertEqual(settings.scale_overlay_font_size, 96)
        self.assertAlmostEqual(settings.overlay_line_width, 24.0)
        self.assertEqual(settings.focus_stack_profile, FocusStackProfile.BALANCED)
        self.assertEqual(settings.focus_stack_sharpen_strength, 100)
        self.assertEqual(settings.magic_segment_model_variant, MagicSegmentModelVariant.EDGE_SAM_3X)

    def test_app_settings_roundtrip_preserves_calibration_presets(self) -> None:
        settings = AppSettings(
            calibration_presets=[
                CalibrationPreset(
                    name="40x",
                    pixels_per_unit=12.5,
                    unit="um",
                    pixel_distance=250.0,
                    actual_distance=20.0,
                    computed_pixels_per_unit=12.5,
                )
            ]
        )

        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "settings.json"
            saved_path = AppSettingsIO.save(settings, path)
            loaded = AppSettingsIO.load(saved_path)

        self.assertEqual(len(loaded.calibration_presets), 1)
        self.assertEqual(loaded.calibration_presets[0].name, "40x")
        self.assertAlmostEqual(loaded.calibration_presets[0].pixel_distance or 0.0, 250.0)
        self.assertAlmostEqual(loaded.calibration_presets[0].actual_distance or 0.0, 20.0)

    def test_app_settings_roundtrip_preserves_selected_capture_device(self) -> None:
        settings = AppSettings(selected_capture_device_id="microview:1")

        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "settings.json"
            saved_path = AppSettingsIO.save(settings, path)
            loaded = AppSettingsIO.load(saved_path)

        self.assertEqual(loaded.selected_capture_device_id, "microview:1")

    def test_project_roundtrip_persists_project_default_calibration_without_writing_legacy_presets(self) -> None:
        project = ProjectState(
            version="0.1.0",
            documents=[],
            calibration_presets=[
                CalibrationPreset(
                    name="legacy preset",
                    pixels_per_unit=5.0,
                    unit="um",
                )
            ],
            project_default_calibration=Calibration(
                mode="project_default",
                pixels_per_unit=4.0,
                unit="um",
                source_label="项目统一标尺",
            ),
        )

        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "demo.fdmproj"
            ProjectIO.save(project, path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            loaded = ProjectIO.load(path)

        self.assertNotIn("calibration_presets", payload)
        self.assertEqual(payload["project_default_calibration"]["mode"], "project_default")
        self.assertIsNotNone(loaded.project_default_calibration)
        self.assertEqual(loaded.project_default_calibration.mode, "project_default")
        self.assertEqual(len(loaded.calibration_presets), 0)

    def test_project_state_loads_legacy_calibration_presets_for_migration(self) -> None:
        payload = {
            "version": "0.1.0",
            "documents": [],
            "calibration_presets": [
                {
                    "name": "legacy 40x",
                    "pixels_per_unit": 8.0,
                    "unit": "um",
                    "pixel_distance": 160.0,
                    "actual_distance": 20.0,
                    "computed_pixels_per_unit": 8.0,
                }
            ],
        }

        project = ProjectState.from_dict(payload)

        self.assertEqual(len(project.calibration_presets), 1)
        self.assertEqual(project.calibration_presets[0].name, "legacy 40x")

    def test_project_roundtrip_persists_project_group_templates_and_document_suppression(self) -> None:
        document = ImageDocument(
            id=new_id("image"),
            path="/tmp/project_groups.png",
            image_size=(200, 120),
            suppressed_project_group_labels=["棉", "棉", " 莱赛尔 "],
        )
        document.initialize_runtime_state()
        project = ProjectState(
            version="0.1.0",
            documents=[document],
            project_group_templates=[
                ProjectGroupTemplate(label="棉", color="#1F7A8C"),
                ProjectGroupTemplate(label="棉", color="#E07A5F"),
                ProjectGroupTemplate(label="莱赛尔", color="#E07A5F"),
            ],
        )

        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "project_groups.fdmproj"
            ProjectIO.save(project, path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            loaded = ProjectIO.load(path)

        self.assertEqual([item["label"] for item in payload["project_group_templates"]], ["棉", "莱赛尔"])
        self.assertEqual([template.label for template in loaded.project_group_templates], ["棉", "莱赛尔"])
        self.assertEqual(loaded.documents[0].suppressed_project_group_labels, ["棉", "莱赛尔"])

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

    def test_frozen_app_settings_use_bundle_resource_root_for_runtime_assets(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            dist_root = Path(tmp_dir) / "dist" / "FiberDiameterMeasurement"
            internal_root = dist_root / "_internal"
            runtime_root = internal_root / "runtime"
            weights_root = runtime_root / "area-models"
            vendor_root = runtime_root / "area-infer" / "vendor" / "yolact"
            weights_root.mkdir(parents=True, exist_ok=True)
            vendor_root.mkdir(parents=True, exist_ok=True)
            exe_path = dist_root / "FiberDiameterMeasurement.exe"
            exe_path.parent.mkdir(parents=True, exist_ok=True)
            exe_path.write_text("", encoding="utf-8")
            worker_path = dist_root / "FiberAreaWorker.exe"
            worker_path.write_text("", encoding="utf-8")

            with (
                patch.object(sys, "frozen", True, create=True),
                patch.object(sys, "_MEIPASS", str(internal_root), create=True),
                patch.object(sys, "executable", str(exe_path), create=True),
            ):
                settings = AppSettings()
                payload = settings.to_dict()

                self.assertEqual(bundle_resource_root(), internal_root.resolve())
                self.assertEqual(payload["area_weights_dir"], "runtime/area-models")
                self.assertEqual(payload["area_vendor_root"], "runtime/area-infer/vendor/yolact")
                self.assertEqual(payload["area_worker_python"], "FiberAreaWorker.exe")
                self.assertEqual(settings.resolved_area_weights_dir(), weights_root.resolve())
                self.assertEqual(settings.resolved_area_vendor_root(), vendor_root.resolve())
                self.assertEqual(settings.resolved_area_worker_program(), str(worker_path.resolve()))

    def test_frozen_existing_internal_paths_roundtrip_back_to_relative_strings(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            dist_root = Path(tmp_dir) / "dist" / "FiberDiameterMeasurement"
            internal_root = dist_root / "_internal"
            runtime_root = internal_root / "runtime"
            weights_root = runtime_root / "area-models"
            vendor_root = runtime_root / "area-infer" / "vendor" / "yolact"
            weights_root.mkdir(parents=True, exist_ok=True)
            vendor_root.mkdir(parents=True, exist_ok=True)
            exe_path = dist_root / "FiberDiameterMeasurement.exe"
            exe_path.parent.mkdir(parents=True, exist_ok=True)
            exe_path.write_text("", encoding="utf-8")

            with (
                patch.object(sys, "frozen", True, create=True),
                patch.object(sys, "_MEIPASS", str(internal_root), create=True),
                patch.object(sys, "executable", str(exe_path), create=True),
            ):
                settings = AppSettings(
                    area_weights_dir=str(weights_root.resolve()),
                    area_vendor_root=str(vendor_root.resolve()),
                    area_worker_python="",
                )

                self.assertEqual(settings.area_weights_dir, str(weights_root.resolve()))
                self.assertEqual(settings.area_vendor_root, str(vendor_root.resolve()))
                self.assertEqual(settings.to_dict()["area_weights_dir"], "runtime/area-models")
                self.assertEqual(settings.to_dict()["area_vendor_root"], "runtime/area-infer/vendor/yolact")

    def test_frozen_legacy_user_settings_area_models_path_migrates_back_to_runtime_relative_default(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            dist_root = Path(tmp_dir) / "dist" / "FiberDiameterMeasurement"
            internal_root = dist_root / "_internal"
            runtime_root = internal_root / "runtime"
            weights_root = runtime_root / "area-models"
            vendor_root = runtime_root / "area-infer" / "vendor" / "yolact"
            weights_root.mkdir(parents=True, exist_ok=True)
            vendor_root.mkdir(parents=True, exist_ok=True)
            exe_path = dist_root / "FiberDiameterMeasurement.exe"
            exe_path.parent.mkdir(parents=True, exist_ok=True)
            exe_path.write_text("", encoding="utf-8")
            legacy_settings_root = Path(tmp_dir) / "localappdata" / "FiberDiameterMeasurement"
            legacy_weights_root = legacy_settings_root / "area-models"
            legacy_weights_root.mkdir(parents=True, exist_ok=True)

            with (
                patch.object(sys, "frozen", True, create=True),
                patch.object(sys, "platform", "win32"),
                patch.object(sys, "_MEIPASS", str(internal_root), create=True),
                patch.object(sys, "executable", str(exe_path), create=True),
                patch.dict("os.environ", {"LOCALAPPDATA": str(Path(tmp_dir) / "localappdata")}, clear=False),
            ):
                settings = AppSettings.from_dict({"area_weights_dir": str(legacy_weights_root.resolve())})

                self.assertEqual(settings.area_weights_dir, "runtime/area-models")
                self.assertEqual(settings.resolved_area_weights_dir(), weights_root.resolve())

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
