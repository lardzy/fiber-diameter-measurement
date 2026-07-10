from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import json
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm.geometry import Line, Point
from fdm.models import Calibration, ImageDocument, Measurement, new_id
from fdm.services.sidecar_io import CalibrationSidecarIO


class HistoryAndSidecarTests(unittest.TestCase):
    def test_sidecar_save_and_load(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            image_path = Path(tmp_dir) / "fiber.png"
            image_path.write_bytes(b"fake")
            document = ImageDocument(
                id=new_id("image"),
                path=str(image_path),
                image_size=(320, 200),
            )
            document.ensure_default_group()
            document.initialize_runtime_state()
            document.calibration = Calibration(
                mode="image_scale",
                pixels_per_unit=10.0,
                unit="um",
                source_label="demo",
            )
            document.metadata["calibration_line"] = Line(Point(5, 5), Point(105, 5)).to_dict()

            result = CalibrationSidecarIO.save_document(document)
            self.assertTrue(result.success)
            self.assertEqual(result.action, "written")
            self.assertIsNotNone(result.path)
            payload = json.loads(result.path.read_text(encoding="utf-8"))
            self.assertEqual(payload["calibration"]["unit"], "um")

            reloaded = ImageDocument(
                id=new_id("image"),
                path=str(image_path),
                image_size=(320, 200),
            )
            reloaded.ensure_default_group()
            reloaded.initialize_runtime_state()
            self.assertTrue(CalibrationSidecarIO.load_document(reloaded))
            self.assertIsNotNone(reloaded.calibration)
            self.assertEqual(reloaded.calibration.source_label, "demo")
            self.assertFalse(reloaded.dirty_flags.calibration_dirty)

    def test_sidecar_save_preserves_previous_file_when_atomic_replace_fails(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            image_path = Path(tmp_dir) / "fiber.png"
            image_path.write_bytes(b"fake")
            sidecar_path = CalibrationSidecarIO.sidecar_path_for_image(image_path)
            previous = b'{"previous": true}'
            sidecar_path.write_bytes(previous)
            document = ImageDocument(
                id=new_id("image"),
                path=str(image_path),
                image_size=(320, 200),
            )
            document.initialize_runtime_state()
            document.calibration = Calibration(
                mode="image_scale",
                pixels_per_unit=10.0,
                unit="um",
                source_label="demo",
            )

            with patch("fdm.atomic_io.os.replace", side_effect=OSError("injected replace failure")):
                result = CalibrationSidecarIO.save_document(document)

            self.assertFalse(result.success)
            self.assertEqual(result.action, "write_failed")
            self.assertIn("injected replace failure", result.message)
            self.assertEqual(sidecar_path.read_bytes(), previous)
            self.assertTrue(document.dirty_flags.calibration_dirty)
            self.assertEqual(set(sidecar_path.parent.iterdir()), {image_path, sidecar_path})

    def test_invalid_sidecar_scale_is_quarantined_without_rewriting_source(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            image_path = Path(tmp_dir) / "fiber.png"
            image_path.write_bytes(b"fake")
            sidecar_path = CalibrationSidecarIO.sidecar_path_for_image(image_path)
            original = json.dumps(
                {
                    "version": 1,
                    "image_path": str(image_path),
                    "calibration": {
                        "mode": "preset",
                        "pixels_per_unit": float("nan"),
                        "unit": "um",
                        "source_label": "legacy-invalid",
                    },
                },
                allow_nan=True,
            ).encode("utf-8")
            sidecar_path.write_bytes(original)
            document = ImageDocument(id="sidecar_invalid", path=str(image_path), image_size=(100, 80))
            document.initialize_runtime_state()

            self.assertFalse(CalibrationSidecarIO.load_document(document))
            self.assertIsNone(document.calibration)
            self.assertIsNotNone(document.calibration_load_error)
            self.assertIsNotNone(document.calibration_load_payload)
            self.assertEqual(sidecar_path.read_bytes(), original)

    def test_document_history_undo_redo(self) -> None:
        document = ImageDocument(
            id=new_id("image"),
            path="/tmp/fiber.png",
            image_size=(320, 200),
        )
        group = document.ensure_default_group()
        document.initialize_runtime_state()
        before = document.snapshot_state()

        measurement = Measurement(
            id=new_id("meas"),
            image_id=document.id,
            fiber_group_id=group.id,
            mode="manual",
            line_px=Line(Point(0, 0), Point(10, 0)),
        )
        document.add_measurement(measurement)
        after = document.snapshot_state()
        document.history.push("新增测量", before, after)

        self.assertEqual(len(document.measurements), 1)
        self.assertTrue(document.history.undo(document))
        self.assertEqual(len(document.measurements), 0)
        self.assertTrue(document.history.redo(document))
        self.assertEqual(len(document.measurements), 1)

    def test_project_asset_document_does_not_create_sidecar_file(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            document = ImageDocument(
                id=new_id("image"),
                path="captures/in_memory_capture.png",
                image_size=(320, 200),
                source_type="project_asset",
            )
            document.initialize_runtime_state()
            document.calibration = Calibration(
                mode="image_scale",
                pixels_per_unit=8.0,
                unit="um",
                source_label="项目内标尺",
            )

            result = CalibrationSidecarIO.save_document(document)

            self.assertTrue(result.success)
            self.assertEqual(result.action, "not_applicable")
            self.assertIsNone(result.path)
            self.assertIsNone(document.sidecar_path)
            self.assertFalse((Path(tmp_dir) / "captures" / "in_memory_capture.png.fdm.json").exists())


if __name__ == "__main__":
    unittest.main()
