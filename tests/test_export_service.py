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
from fdm.settings import (
    AppSettings,
    RawRecordDataSource,
    RawRecordExportDirection,
    RawRecordExportRule,
    RawRecordMeasurementFilter,
    RawRecordTemplate,
)


class ExportServiceTests(unittest.TestCase):
    def _write_minimal_macro_template(self, path: Path, *, sheet_name: str = "Raw") -> None:
        sheet_xml = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <dimension ref="A1:I5"/>
  <sheetData>
    <row r="2"><c r="B2" s="7"/></row>
    <row r="5"><c r="D5"><f>SUM(A1:A2)</f><v>0</v></c></row>
  </sheetData>
</worksheet>'''
        other_sheet_xml = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <dimension ref="A1:A1"/>
  <sheetData><row r="1"><c r="A1" t="inlineStr"><is><t>untouched</t></is></c></row></sheetData>
</worksheet>'''
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(
                "[Content_Types].xml",
                '''<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Default Extension="bin" ContentType="application/vnd.ms-office.vbaProject"/>
  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.ms-excel.sheet.macroEnabled.main+xml"/>
  <Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
  <Override PartName="/xl/worksheets/sheet2.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
</Types>''',
            )
            archive.writestr(
                "_rels/.rels",
                '''<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>''',
            )
            archive.writestr(
                "xl/workbook.xml",
                f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets>
    <sheet name="{sheet_name}" sheetId="1" r:id="rId1"/>
    <sheet name="Other" sheetId="2" r:id="rId2"/>
  </sheets>
</workbook>''',
            )
            archive.writestr(
                "xl/_rels/workbook.xml.rels",
                '''<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet2.xml"/>
  <Relationship Id="rId3" Type="http://schemas.microsoft.com/office/2006/relationships/vbaProject" Target="vbaProject.bin"/>
</Relationships>''',
            )
            archive.writestr("xl/worksheets/sheet1.xml", sheet_xml)
            archive.writestr("xl/worksheets/sheet2.xml", other_sheet_xml)
            archive.writestr("xl/vbaProject.bin", b"macro-bytes-do-not-touch")

    def _cell_texts(self, sheet_xml: bytes) -> dict[str, str]:
        namespace = {"xlsx": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
        root = ElementTree.fromstring(sheet_xml)
        values: dict[str, str] = {}
        for cell in root.findall(".//xlsx:c", namespace):
            coordinate = cell.attrib["r"]
            value_node = cell.find("xlsx:v", namespace)
            inline_node = cell.find("xlsx:is/xlsx:t", namespace)
            if value_node is not None and value_node.text is not None:
                values[coordinate] = value_node.text
            elif inline_node is not None and inline_node.text is not None:
                values[coordinate] = inline_node.text
        return values

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

    def test_raw_record_template_export_preserves_macro_and_writes_rules(self) -> None:
        first_document = ImageDocument(
            id=new_id("image"),
            path="/tmp/raw_record_a.png",
            image_size=(200, 100),
        )
        first_document.initialize_runtime_state()
        first_document.add_measurement(
            Measurement(
                id=new_id("meas"),
                image_id=first_document.id,
                fiber_group_id=None,
                mode="manual",
                line_px=Line(Point(0, 0), Point(10, 0)),
            )
        )
        first_document.add_measurement(
            Measurement(
                id=new_id("meas"),
                image_id=first_document.id,
                fiber_group_id=None,
                mode="polygon_area",
                measurement_kind="area",
                polygon_px=[Point(0, 0), Point(5, 0), Point(5, 5), Point(0, 5)],
            )
        )
        second_document = ImageDocument(
            id=new_id("image"),
            path="/tmp/raw_record_b.png",
            image_size=(200, 100),
        )
        second_document.initialize_runtime_state()
        second_document.add_measurement(
            Measurement(
                id=new_id("meas"),
                image_id=second_document.id,
                fiber_group_id=None,
                mode="manual",
                line_px=Line(Point(0, 0), Point(20, 0)),
            )
        )
        project = ProjectState(version="0.1.0", documents=[first_document, second_document])

        with TemporaryDirectory() as tmp_dir:
            template_path = Path(tmp_dir) / "raw_template.xlsm"
            self._write_minimal_macro_template(template_path)
            template_before = template_path.read_bytes()
            output_path = Path(tmp_dir) / "raw_output.xlsm"
            selection = ExportSelection(
                include_excel=True,
                scope=ExportScope.ALL_OPEN,
                raw_record_template_path=str(template_path),
            )
            raw_record_template = RawRecordTemplate(
                name="Raw",
                path=str(template_path),
                rules=[
                    RawRecordExportRule(
                        data_source=RawRecordDataSource.DIAMETER_RESULT,
                        sheet_name="Raw",
                        start_cell="B2",
                        direction=RawRecordExportDirection.VERTICAL,
                    ),
                    RawRecordExportRule(
                        data_source=RawRecordDataSource.AREA_RESULT,
                        sheet_name="Raw",
                        start_cell="D4",
                        direction=RawRecordExportDirection.HORIZONTAL,
                    ),
                    RawRecordExportRule(
                        data_source=RawRecordDataSource.MEASUREMENT_FIELD,
                        field_name="类型",
                        measurement_filter=RawRecordMeasurementFilter.ALL,
                        sheet_name="Raw",
                        start_cell="G2",
                        direction=RawRecordExportDirection.HORIZONTAL,
                    ),
                ],
            )

            outputs = ExportService().export_project(
                project,
                tmp_dir,
                selection=selection,
                single_output_path=output_path,
                raw_record_template=raw_record_template,
            )

            self.assertEqual(template_path.read_bytes(), template_before)
            self.assertEqual(outputs["xlsx"], output_path)
            with zipfile.ZipFile(template_path) as template_archive, zipfile.ZipFile(output_path) as output_archive:
                self.assertEqual(output_archive.read("xl/vbaProject.bin"), template_archive.read("xl/vbaProject.bin"))
                self.assertEqual(output_archive.read("xl/worksheets/sheet2.xml"), template_archive.read("xl/worksheets/sheet2.xml"))
                sheet_values = self._cell_texts(output_archive.read("xl/worksheets/sheet1.xml"))
                sheet_xml = ElementTree.fromstring(output_archive.read("xl/worksheets/sheet1.xml"))

            self.assertEqual(sheet_values["B2"], "10")
            self.assertEqual(sheet_values["B3"], "20")
            self.assertEqual(sheet_values["D4"], "25")
            self.assertEqual(sheet_values["G2"], "线段")
            self.assertEqual(sheet_values["H2"], "面积")
            self.assertEqual(sheet_values["I2"], "线段")
            namespace = {"xlsx": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
            formula = sheet_xml.find(".//xlsx:c[@r='D5']/xlsx:f", namespace)
            self.assertIsNotNone(formula)
            self.assertEqual(formula.text, "SUM(A1:A2)")

    def test_raw_record_template_unique_field_range_deduplicates_and_truncates(self) -> None:
        document = ImageDocument(
            id=new_id("image"),
            path="/tmp/raw_record_unique_range.png",
            image_size=(200, 100),
        )
        document.initialize_runtime_state()
        groups = [
            document.create_group(color="#1F7A8C", label="棉"),
            document.create_group(color="#E07A5F", label="麻"),
            document.create_group(color="#22C55E", label="棉"),
            document.create_group(color="#A855F7", label="莱赛尔"),
            document.create_group(color="#F59E0B", label="莫代尔"),
        ]
        for group in groups:
            document.add_measurement(
                Measurement(
                    id=new_id("meas"),
                    image_id=document.id,
                    fiber_group_id=group.id,
                    mode="manual",
                    line_px=Line(Point(0, 0), Point(10, 0)),
                )
            )
        project = ProjectState(version="0.1.0", documents=[document])

        with TemporaryDirectory() as tmp_dir:
            template_path = Path(tmp_dir) / "raw_template.xlsm"
            self._write_minimal_macro_template(template_path)
            output_path = Path(tmp_dir) / "raw_output.xlsm"
            selection = ExportSelection(
                include_excel=True,
                scope=ExportScope.ALL_OPEN,
                raw_record_template_path=str(template_path),
            )
            raw_record_template = RawRecordTemplate(
                name="Raw",
                path=str(template_path),
                rules=[
                    RawRecordExportRule(
                        data_source=RawRecordDataSource.UNIQUE_FIELD_RANGE,
                        field_name="纤维种类名称",
                        sheet_name="Raw",
                        start_cell="BA11",
                        end_cell="BC11",
                        direction=RawRecordExportDirection.HORIZONTAL,
                    ),
                    RawRecordExportRule(
                        data_source=RawRecordDataSource.UNIQUE_FIELD_RANGE,
                        field_name="纤维类别序号",
                        sheet_name="Raw",
                        start_cell="BA12",
                        end_cell="BC12",
                        direction=RawRecordExportDirection.HORIZONTAL,
                    ),
                ],
            )

            outputs = ExportService().export_project(
                project,
                tmp_dir,
                selection=selection,
                single_output_path=output_path,
                raw_record_template=raw_record_template,
            )

            self.assertEqual(outputs["xlsx"], output_path)
            with zipfile.ZipFile(output_path) as output_archive:
                sheet_values = self._cell_texts(output_archive.read("xl/worksheets/sheet1.xml"))

        self.assertEqual([sheet_values["BA11"], sheet_values["BB11"], sheet_values["BC11"]], ["棉", "麻", "莱赛尔"])
        self.assertEqual([sheet_values["BA12"], sheet_values["BB12"], sheet_values["BC12"]], ["1", "2", "4"])
        self.assertNotIn("BD11", sheet_values)
        self.assertNotIn("BD12", sheet_values)

    def test_raw_record_template_export_falls_back_to_default_xlsx_on_rule_error(self) -> None:
        document = ImageDocument(
            id=new_id("image"),
            path="/tmp/raw_record_fallback.png",
            image_size=(200, 100),
        )
        document.initialize_runtime_state()
        document.add_measurement(
            Measurement(
                id=new_id("meas"),
                image_id=document.id,
                fiber_group_id=None,
                mode="manual",
                line_px=Line(Point(0, 0), Point(10, 0)),
            )
        )
        project = ProjectState(version="0.1.0", documents=[document])

        with TemporaryDirectory() as tmp_dir:
            template_path = Path(tmp_dir) / "raw_template.xlsm"
            self._write_minimal_macro_template(template_path)
            output_path = Path(tmp_dir) / "raw_output.xlsm"
            selection = ExportSelection(
                include_excel=True,
                scope=ExportScope.CURRENT,
                raw_record_template_path=str(template_path),
            )
            raw_record_template = RawRecordTemplate(
                name="Raw",
                path=str(template_path),
                rules=[RawRecordExportRule(sheet_name="Missing", start_cell="B2")],
            )

            outputs = ExportService().export_project(
                project,
                tmp_dir,
                selection=selection,
                documents=[document],
                single_output_path=output_path,
                raw_record_template=raw_record_template,
            )

            self.assertEqual(outputs["xlsx"], output_path.with_suffix(".xlsx"))
            self.assertTrue(outputs["xlsx"].exists())
            self.assertIn("_template_fallback_message", outputs)
            self.assertFalse(output_path.exists())

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
        self.assertIsNone(rows[0]["纤维类别序号"])
        self.assertEqual(rows[0]["纤维种类颜色"], UNCATEGORIZED_COLOR)
        self.assertEqual(list(rows[0].keys())[:6], ["纤维种类", "类型", "结果", "单位", "标尺信息", "模式"])

    def test_export_rows_include_fiber_category_sequence(self) -> None:
        document = ImageDocument(
            id=new_id("image"),
            path="/tmp/fiber_category_sequence.png",
            image_size=(400, 300),
        )
        document.initialize_runtime_state()
        first_group = document.create_group(color="#1F7A8C", label="棉")
        second_group = document.create_group(color="#E07A5F", label="麻")
        document.add_measurement(
            Measurement(
                id=new_id("meas"),
                image_id=document.id,
                fiber_group_id=second_group.id,
                mode="manual",
                line_px=Line(Point(0, 0), Point(20, 0)),
            )
        )

        service = ExportService()
        measurement_rows = service.build_measurement_rows([document])
        fiber_rows = service.build_fiber_rows([document])

        self.assertEqual(measurement_rows[0]["纤维类别序号"], second_group.number)
        self.assertEqual([row["纤维类别序号"] for row in fiber_rows], [first_group.number, second_group.number])

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
        self.assertEqual(rows[0]["孔洞面积"], 0.0)

    def test_area_measurement_rows_use_exact_mask_area_when_available(self) -> None:
        document = ImageDocument(
            id=new_id("image"),
            path="/tmp/fiber_area_exact.png",
            image_size=(400, 300),
        )
        document.initialize_runtime_state()
        document.calibration = Calibration(mode="preset", pixels_per_unit=10.0, unit="um", source_label="demo")
        document.add_measurement(
            Measurement(
                id=new_id("meas"),
                image_id=document.id,
                fiber_group_id=None,
                mode="magic_segment",
                measurement_kind="area",
                polygon_px=[Point(0, 0), Point(20, 0), Point(20, 10), Point(0, 10)],
                area_rings_px=[
                    [Point(0, 0), Point(20, 0), Point(20, 10), Point(0, 10)],
                    [Point(6, 2), Point(14, 2), Point(14, 8), Point(6, 8)],
                ],
                exact_area_px=180.0,
            )
        )

        rows = ExportService().build_measurement_rows([document])

        self.assertEqual(rows[0]["结果"], 1.8)
        self.assertEqual(rows[0]["面积(px²)"], 180.0)
        self.assertEqual(rows[0]["孔洞面积"], 0.48)

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

    def test_planned_outputs_use_raw_record_template_filename(self) -> None:
        document = ImageDocument(
            id=new_id("image"),
            path="/tmp/fiber_planned_raw_template.png",
            image_size=(400, 300),
        )
        document.initialize_runtime_state()

        planned = ExportService().planned_outputs(
            [document],
            ExportSelection(
                include_excel=True,
                scope=ExportScope.CURRENT,
                raw_record_template_path="/tmp/面积法原始记录模板.xltm",
            ),
        )

        self.assertEqual(len(planned), 1)
        self.assertEqual(planned[0].kind, "xlsx")
        self.assertEqual(planned[0].filename, "面积法原始记录模板.xlsm")

    def test_export_progress_callback_reports_each_output_step(self) -> None:
        document = ImageDocument(
            id=new_id("image"),
            path="/tmp/fiber_progress.png",
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
        progress_events: list[tuple[int, int, str, str | None]] = []

        with TemporaryDirectory() as tmp_dir:
            ExportService().export_project(
                project,
                tmp_dir,
                selection=ExportSelection(include_csv=True, include_excel=True, scope=ExportScope.CURRENT),
                progress_callback=lambda completed, total, label, path: progress_events.append(
                    (completed, total, label, path.name if path is not None else None)
                ),
            )

        self.assertEqual(
            progress_events,
            [
                (0, 4, CSV_IMAGE_SUMMARY_FILENAME, CSV_IMAGE_SUMMARY_FILENAME),
                (1, 4, "纤维种类汇总.csv", "纤维种类汇总.csv"),
                (2, 4, "测量明细.csv", "测量明细.csv"),
                (3, 4, XLSX_EXPORT_FILENAME, XLSX_EXPORT_FILENAME),
                (4, 4, "导出完成", None),
            ],
        )

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
