from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import csv
import sys
import unittest
import zipfile
from xml.etree import ElementTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm.geometry import Line, Point
from fdm.models import Calibration, ImageDocument, Measurement, ProjectState, UNCATEGORIZED_COLOR, UNCATEGORIZED_LABEL, format_measurement_label_value, new_id
from fdm.services.export_service import (
    CSV_IMAGE_SUMMARY_FILENAME,
    SHEET_MEASUREMENT_DETAILS,
    XLSX_EXPORT_FILENAME,
    ExportImageRenderMode,
    ExportScope,
    ExportSelection,
    ExportService,
)
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
            self.assertEqual(image_summary_csv.name, CSV_IMAGE_SUMMARY_FILENAME)
            self.assertEqual(xlsx_path.name, XLSX_EXPORT_FILENAME)

            with image_summary_csv.open("r", encoding="utf-8-sig") as file_obj:
                rows = list(csv.DictReader(file_obj))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["测量数量"], "1")
            self.assertEqual(rows[0]["单位"], "um")
            self.assertEqual(rows[0]["当前激活种类编号"], "1")

            with zipfile.ZipFile(xlsx_path) as archive:
                names = set(archive.namelist())
                workbook_xml = archive.read("xl/workbook.xml")
            self.assertIn("xl/workbook.xml", names)
            self.assertIn("xl/worksheets/sheet1.xml", names)
            self.assertIn("xl/worksheets/sheet4.xml", names)
            workbook = ElementTree.fromstring(workbook_xml)
            namespace = {"xlsx": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
            sheet_names = [element.attrib["name"] for element in workbook.findall("./xlsx:sheets/xlsx:sheet", namespace)]
            self.assertEqual(sheet_names[0], SHEET_MEASUREMENT_DETAILS)

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
        self.assertEqual(rows[0]["纤维种类"], UNCATEGORIZED_LABEL)
        self.assertEqual(rows[0]["纤维种类名称"], UNCATEGORIZED_LABEL)
        self.assertEqual(rows[0]["纤维种类颜色"], UNCATEGORIZED_COLOR)
        self.assertEqual(list(rows[0].keys())[:6], ["纤维种类", "类型", "结果", "单位", "标尺信息", "模式"])

    def test_area_measurement_rows_include_polygon_fields(self) -> None:
        document = ImageDocument(
            id=new_id("image"),
            path="/tmp/fiber_area.png",
            image_size=(400, 300),
        )
        document.initialize_runtime_state()
        document.add_measurement(
            Measurement(
                id=new_id("meas"),
                image_id=document.id,
                fiber_group_id=None,
                mode="polygon_area",
                measurement_kind="area",
                polygon_px=[Point(0, 0), Point(10, 0), Point(10, 10), Point(0, 10)],
                confidence=0.9,
                status="manual",
            )
        )

        rows = ExportService().build_measurement_rows([document])

        self.assertEqual(rows[0]["类型"], "面积")
        self.assertEqual(rows[0]["模式"], "多边形面积")
        self.assertEqual(rows[0]["单位"], "px²")
        self.assertEqual(rows[0]["多边形点数"], 4)
        self.assertIn('"x": 10', rows[0]["多边形顶点JSON"])

    def test_measurement_rows_include_polyline_and_count_fields(self) -> None:
        document = ImageDocument(
            id=new_id("image"),
            path="/tmp/fiber_polyline_count.png",
            image_size=(400, 300),
        )
        document.initialize_runtime_state()
        document.calibration = Calibration(mode="preset", pixels_per_unit=2.0, unit="um", source_label="demo")
        document.add_measurement(
            Measurement(
                id=new_id("meas"),
                image_id=document.id,
                fiber_group_id=None,
                mode="continuous_manual",
                measurement_kind="polyline",
                polyline_px=[Point(0, 0), Point(10, 0), Point(10, 10)],
                status="continuous_manual",
            )
        )
        document.add_measurement(
            Measurement(
                id=new_id("meas"),
                image_id=document.id,
                fiber_group_id=None,
                mode="count",
                measurement_kind="count",
                point_px=Point(5, 6),
                status="count",
            )
        )

        rows = ExportService().build_measurement_rows([document])

        self.assertEqual(rows[0]["类型"], "折线")
        self.assertEqual(rows[0]["模式"], "连续测量")
        self.assertEqual(rows[0]["折线点数"], 3)
        self.assertIn('"y": 10', rows[0]["折线顶点JSON"])
        self.assertEqual(rows[1]["类型"], "计数点")
        self.assertEqual(rows[1]["模式"], "计数")
        self.assertEqual(rows[1]["单位"], "个")
        self.assertEqual(rows[1]["计数点X(px)"], 5)
        self.assertEqual(rows[1]["计数点Y(px)"], 6)

    def test_single_output_export_can_override_filename(self) -> None:
        document = ImageDocument(
            id=new_id("image"),
            path="/tmp/fiber_single_export.png",
            image_size=(400, 300),
        )
        document.initialize_runtime_state()
        document.add_measurement(
            Measurement(
                id=new_id("meas"),
                image_id=document.id,
                fiber_group_id=None,
                mode="manual",
                line_px=Line(Point(0, 0), Point(20, 0)),
            )
        )
        project = ProjectState(version="0.1.0", documents=[document])
        selection = ExportSelection(include_excel=True, scope=ExportScope.CURRENT)

        with TemporaryDirectory() as tmp_dir:
            custom_path = Path(tmp_dir) / "custom_name.xlsx"
            outputs = ExportService().export_project(
                project,
                tmp_dir,
                selection=selection,
                documents=[document],
                single_output_path=custom_path,
            )

            self.assertEqual(outputs["xlsx"], custom_path)
            self.assertTrue(custom_path.exists())

    def test_planned_outputs_resolve_single_excel_file(self) -> None:
        document = ImageDocument(
            id=new_id("image"),
            path="/tmp/fiber_planned_output.png",
            image_size=(400, 300),
        )
        document.initialize_runtime_state()

        planned = ExportService().planned_outputs(
            [document],
            ExportSelection(include_excel=True, scope=ExportScope.CURRENT),
        )

        self.assertEqual(len(planned), 1)
        self.assertEqual(planned[0].kind, "xlsx")
        self.assertEqual(planned[0].filename, XLSX_EXPORT_FILENAME)

    def test_all_enabled_export_selection_uses_full_resolution_render_mode(self) -> None:
        selection = ExportSelection.all_enabled()
        self.assertEqual(selection.render_mode, ExportImageRenderMode.FULL_RESOLUTION)

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

    def test_measurement_label_text_preserves_fixed_decimal_places(self) -> None:
        self.assertEqual(format_measurement_label_value(10.0, "um", 4), "10.0000 um")
        self.assertEqual(format_measurement_label_value(11.123, "um", 4), "11.1230 um")
        self.assertEqual(format_measurement_label_value(10.0, "um", 0), "10 um")


if __name__ == "__main__":
    unittest.main()
