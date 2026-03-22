from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import csv
import sys
import unittest
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm.geometry import Line, Point
from fdm.models import Calibration, ImageDocument, Measurement, ProjectState, UNCATEGORIZED_COLOR, UNCATEGORIZED_LABEL, format_measurement_label_value, new_id
from fdm.services.export_service import ExportImageRenderMode, ExportSelection, ExportService
from fdm.settings import AppSettings


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

    def test_measurement_rows_include_uncategorized_metadata(self) -> None:
        document = ImageDocument(
            id=new_id("image"),
            path="/tmp/fiber_uncategorized.png",
            image_size=(400, 300),
        )
        document.initialize_runtime_state()
        measurement = Measurement(
            id=new_id("meas"),
            image_id=document.id,
            fiber_group_id=None,
            mode="manual",
            line_px=Line(Point(0, 0), Point(20, 0)),
            confidence=0.75,
            status="manual",
        )
        document.add_measurement(measurement)

        rows = ExportService().build_measurement_rows([document])

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["fiber_group_display"], UNCATEGORIZED_LABEL)
        self.assertEqual(rows[0]["fiber_group_label"], UNCATEGORIZED_LABEL)
        self.assertEqual(rows[0]["fiber_group_color"], UNCATEGORIZED_COLOR)

    def test_all_enabled_export_selection_uses_screen_render_mode(self) -> None:
        selection = ExportSelection.all_enabled()
        self.assertEqual(selection.render_mode, ExportImageRenderMode.SCREEN_SCALE_FULL_IMAGE)

    def test_measurement_label_text_uses_configured_decimals_only_for_display(self) -> None:
        document = ImageDocument(
            id=new_id("image"),
            path="/tmp/measurement_label.png",
            image_size=(100, 80),
        )
        document.initialize_runtime_state()
        document.calibration = Calibration(
            mode="preset",
            pixels_per_unit=2.0,
            unit="um",
            source_label="demo",
        )
        measurement = Measurement(
            id=new_id("meas"),
            image_id=document.id,
            fiber_group_id=None,
            mode="manual",
            line_px=Line(Point(0, 0), Point(10.246, 0)),
        )
        measurement.recalculate(document.calibration)

        settings = AppSettings(measurement_label_decimals=2)

        self.assertEqual(format_measurement_label_value(measurement.diameter_unit or 0.0, "um", settings.measurement_label_decimals), "5.12 um")
        self.assertAlmostEqual(measurement.diameter_unit or 0.0, 5.123)


if __name__ == "__main__":
    unittest.main()
