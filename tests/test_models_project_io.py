from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm.geometry import Line, Point
from fdm.models import Calibration, ImageDocument, Measurement, ProjectState, new_id
from fdm.project_io import ProjectIO


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
        self.assertEqual(loaded_document.sorted_groups()[0].number, 1)
        self.assertAlmostEqual(loaded_document.measurements[0].diameter_px or 0.0, 8.0)
        self.assertAlmostEqual(loaded_document.measurements[0].diameter_unit or 0.0, 2.0)


if __name__ == "__main__":
    unittest.main()
