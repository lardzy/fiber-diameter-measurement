from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import csv
import sys
import unittest
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm.geometry import Line, Point
from fdm.models import Calibration, ImageDocument, Measurement, ProjectState, new_id
from fdm.services.export_service import ExportSelection, ExportService


class ExportServiceTests(unittest.TestCase):
    def test_export_writes_csv_and_xlsx(self) -> None:
        document = ImageDocument(
            id=new_id("image"),
            path="/tmp/fiber.png",
            image_size=(400, 300),
        )
        group = document.ensure_default_group()
        document.initialize_runtime_state()
        group.label = "棉"
        document.calibration = Calibration(
            mode="preset",
            pixels_per_unit=5.0,
            unit="um",
            source_label="demo preset",
        )
        document.metadata["calibration_line"] = Line(Point(0, 10), Point(100, 10)).to_dict()
        measurement = Measurement(
            id=new_id("meas"),
            image_id=document.id,
            fiber_group_id=group.id,
            mode="manual",
            line_px=Line(Point(0, 0), Point(25, 0)),
            confidence=1.0,
            status="manual",
        )
        document.add_measurement(measurement)
        project = ProjectState(version="0.1.0", documents=[document])

        with TemporaryDirectory() as tmp_dir:
            outputs = ExportService().export_project(
                project,
                tmp_dir,
                selection=ExportSelection.all_enabled(),
            )
            image_summary_csv = outputs["image_summary_csv"]
            xlsx_path = outputs["xlsx"]
            scale_jsons = outputs["scale_jsons"]
            self.assertTrue(image_summary_csv.exists())
            self.assertTrue(xlsx_path.exists())
            self.assertEqual(len(scale_jsons), 1)

            with image_summary_csv.open("r", encoding="utf-8-sig") as file_obj:
                rows = list(csv.DictReader(file_obj))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["measurement_count"], "1")
            self.assertEqual(rows[0]["unit"], "um")
            self.assertEqual(rows[0]["active_group_number"], "1")

            with zipfile.ZipFile(xlsx_path) as archive:
                names = set(archive.namelist())
            self.assertIn("xl/workbook.xml", names)
            self.assertIn("xl/worksheets/sheet1.xml", names)
            self.assertIn("xl/worksheets/sheet4.xml", names)


if __name__ == "__main__":
    unittest.main()
