from __future__ import annotations

from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
import csv
import math
import zipfile
from xml.sax.saxutils import escape

from fdm.models import FiberGroup, ImageDocument, Measurement, ProjectState


class ExportService:
    def export_project(
        self,
        project: ProjectState,
        output_dir: str | Path,
        *,
        overlay_renderer=None,
    ) -> dict[str, Path]:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        image_rows = self.build_image_summary_rows(project)
        fiber_rows = self.build_fiber_rows(project)
        measurement_rows = self.build_measurement_rows(project)
        meta_rows = self.build_export_meta_rows(project)

        csv_outputs = {
            "image_summary_csv": output_path / "image_summary.csv",
            "fiber_details_csv": output_path / "fiber_details.csv",
            "measurement_details_csv": output_path / "measurement_details.csv",
        }
        self._write_csv(csv_outputs["image_summary_csv"], image_rows)
        self._write_csv(csv_outputs["fiber_details_csv"], fiber_rows)
        self._write_csv(csv_outputs["measurement_details_csv"], measurement_rows)

        xlsx_path = output_path / "measurement_export.xlsx"
        self._write_xlsx(
            xlsx_path,
            {
                "image_summary": image_rows,
                "fiber_details": fiber_rows,
                "measurement_details": measurement_rows,
                "export_meta": meta_rows,
            },
        )

        if overlay_renderer is not None:
            for document in project.documents:
                base_name = Path(document.path).stem or document.id
                overlay_renderer(
                    document,
                    output_path / f"{base_name}_measurements.png",
                    include_measurements=True,
                    include_scale=False,
                )
                overlay_renderer(
                    document,
                    output_path / f"{base_name}_scale.png",
                    include_measurements=False,
                    include_scale=True,
                )

        return {
            **csv_outputs,
            "xlsx": xlsx_path,
        }

    def build_image_summary_rows(self, project: ProjectState) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for document in project.documents:
            stats = document.stats()
            rows.append(
                OrderedDict(
                    image_id=document.id,
                    image_path=document.path,
                    width_px=document.image_size[0],
                    height_px=document.image_size[1],
                    calibration_mode=document.calibration.mode if document.calibration else "none",
                    calibration_source=document.calibration.source_label if document.calibration else "",
                    unit=document.calibration.unit if document.calibration else "px",
                    measurement_count=len(document.measurements),
                    fiber_group_count=len(document.fiber_groups),
                    mean_diameter=stats["mean"],
                    min_diameter=stats["min"],
                    max_diameter=stats["max"],
                    stddev_diameter=stats["stddev"],
                )
            )
        return rows

    def build_fiber_rows(self, project: ProjectState) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for document in project.documents:
            measurement_lookup = {measurement.id: measurement for measurement in document.measurements}
            for group in document.fiber_groups:
                values = [
                    measurement_lookup[measurement_id].diameter_unit
                    for measurement_id in group.measurement_ids
                    if measurement_id in measurement_lookup and measurement_lookup[measurement_id].diameter_unit is not None
                ]
                rows.append(
                    OrderedDict(
                        image_id=document.id,
                        image_path=document.path,
                        fiber_group_id=group.id,
                        fiber_group_name=group.name,
                        color=group.color,
                        measurement_count=len(group.measurement_ids),
                        mean_diameter=self._mean(values),
                        min_diameter=min(values) if values else None,
                        max_diameter=max(values) if values else None,
                        stddev_diameter=self._stddev(values),
                        unit=document.calibration.unit if document.calibration else "px",
                    )
                )
        return rows

    def build_measurement_rows(self, project: ProjectState) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for document in project.documents:
            group_lookup = {group.id: group for group in document.fiber_groups}
            unit = document.calibration.unit if document.calibration else "px"
            for measurement in document.measurements:
                effective_line = measurement.effective_line()
                group = group_lookup.get(measurement.fiber_group_id or "")
                rows.append(
                    OrderedDict(
                        image_id=document.id,
                        image_path=document.path,
                        measurement_id=measurement.id,
                        fiber_group_id=measurement.fiber_group_id or "",
                        fiber_group_name=group.name if group else "",
                        mode=measurement.mode,
                        status=measurement.status,
                        confidence=round(measurement.confidence, 4),
                        created_at=measurement.created_at,
                        start_x_px=round(effective_line.start.x, 3),
                        start_y_px=round(effective_line.start.y, 3),
                        end_x_px=round(effective_line.end.x, 3),
                        end_y_px=round(effective_line.end.y, 3),
                        diameter_px=round(measurement.diameter_px or 0.0, 6),
                        diameter_value=round(measurement.diameter_unit or 0.0, 6),
                        unit=unit,
                        calibration_mode=document.calibration.mode if document.calibration else "none",
                        calibration_source=document.calibration.source_label if document.calibration else "",
                    )
                )
        return rows

    def build_export_meta_rows(self, project: ProjectState) -> list[dict[str, object]]:
        return [
            OrderedDict(
                exported_at=datetime.now(tz=timezone.utc).isoformat(),
                app_version=project.version,
                document_count=len(project.documents),
                measurement_count=sum(len(document.measurements) for document in project.documents),
                fiber_group_count=sum(len(document.fiber_groups) for document in project.documents),
            )
        ]

    def _write_csv(self, path: Path, rows: list[dict[str, object]]) -> None:
        fieldnames = self._collect_fieldnames(rows)
        with path.open("w", newline="", encoding="utf-8-sig") as file_obj:
            writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def _write_xlsx(self, path: Path, sheets: dict[str, list[dict[str, object]]]) -> None:
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("[Content_Types].xml", self._content_types_xml(len(sheets)))
            archive.writestr("_rels/.rels", self._root_rels_xml())
            archive.writestr("xl/workbook.xml", self._workbook_xml(sheets))
            archive.writestr("xl/_rels/workbook.xml.rels", self._workbook_rels_xml(len(sheets)))
            archive.writestr("xl/styles.xml", self._styles_xml())
            for index, (sheet_name, rows) in enumerate(sheets.items(), start=1):
                archive.writestr(
                    f"xl/worksheets/sheet{index}.xml",
                    self._sheet_xml(rows),
                )

    def _sheet_xml(self, rows: list[dict[str, object]]) -> str:
        headers = self._collect_fieldnames(rows)
        xml_rows = []
        xml_rows.append(self._xml_row(1, headers, header=True))
        for row_index, row in enumerate(rows, start=2):
            values = [row.get(header) for header in headers]
            xml_rows.append(self._xml_row(row_index, values, header=False))
        dimension_end = self._column_name(max(1, len(headers))) + str(max(1, len(rows) + 1))
        joined_rows = "".join(xml_rows)
        return (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            f'<dimension ref="A1:{dimension_end}"/>'
            "<sheetViews><sheetView workbookViewId=\"0\"/></sheetViews>"
            "<sheetFormatPr defaultRowHeight=\"15\"/>"
            f"<sheetData>{joined_rows}</sheetData>"
            "</worksheet>"
        )

    def _xml_row(self, row_number: int, values: list[object], *, header: bool) -> str:
        cells = []
        for index, value in enumerate(values, start=1):
            coordinate = f"{self._column_name(index)}{row_number}"
            style = ' s="1"' if header else ""
            if value is None:
                cells.append(f'<c r="{coordinate}"{style}/>')
                continue
            if isinstance(value, (int, float)) and not isinstance(value, bool) and not math.isnan(float(value)):
                cells.append(f'<c r="{coordinate}"{style}><v>{value}</v></c>')
            else:
                cells.append(
                    f'<c r="{coordinate}" t="inlineStr"{style}><is><t>{escape(str(value))}</t></is></c>'
                )
        return f'<row r="{row_number}">{"".join(cells)}</row>'

    def _collect_fieldnames(self, rows: list[dict[str, object]]) -> list[str]:
        fieldnames: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row.keys():
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)
        return fieldnames or ["empty"]

    def _content_types_xml(self, sheet_count: int) -> str:
        overrides = [
            '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>',
            '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>',
        ]
        for index in range(1, sheet_count + 1):
            overrides.append(
                f'<Override PartName="/xl/worksheets/sheet{index}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
            )
        return (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            f'{"".join(overrides)}'
            "</Types>"
        )

    def _root_rels_xml(self) -> str:
        return (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
            "</Relationships>"
        )

    def _workbook_xml(self, sheets: dict[str, list[dict[str, object]]]) -> str:
        sheet_entries = []
        for index, sheet_name in enumerate(sheets.keys(), start=1):
            safe_name = escape(sheet_name[:31])
            sheet_entries.append(
                f'<sheet name="{safe_name}" sheetId="{index}" r:id="rId{index}"/>'
            )
        return (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            "<sheets>"
            f'{"".join(sheet_entries)}'
            "</sheets>"
            "</workbook>"
        )

    def _workbook_rels_xml(self, sheet_count: int) -> str:
        relationships = []
        for index in range(1, sheet_count + 1):
            relationships.append(
                f'<Relationship Id="rId{index}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{index}.xml"/>'
            )
        relationships.append(
            f'<Relationship Id="rId{sheet_count + 1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
        )
        return (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            f'{"".join(relationships)}'
            "</Relationships>"
        )

    def _styles_xml(self) -> str:
        return (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            '<fonts count="1"><font><sz val="11"/><name val="Calibri"/></font></fonts>'
            '<fills count="1"><fill><patternFill patternType="none"/></fill></fills>'
            '<borders count="1"><border/></borders>'
            '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
            '<cellXfs count="2">'
            '<xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>'
            '<xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0" applyAlignment="1"><alignment horizontal="center"/></xf>'
            '</cellXfs>'
            '</styleSheet>'
        )

    def _column_name(self, index: int) -> str:
        result = ""
        current = index
        while current > 0:
            current, remainder = divmod(current - 1, 26)
            result = chr(65 + remainder) + result
        return result

    def _mean(self, values: list[float | None]) -> float | None:
        usable_values = [value for value in values if value is not None]
        if not usable_values:
            return None
        return sum(usable_values) / len(usable_values)

    def _stddev(self, values: list[float | None]) -> float | None:
        usable_values = [value for value in values if value is not None]
        if not usable_values:
            return None
        if len(usable_values) == 1:
            return 0.0
        mean_value = sum(usable_values) / len(usable_values)
        variance = sum((value - mean_value) ** 2 for value in usable_values) / len(usable_values)
        return math.sqrt(variance)
