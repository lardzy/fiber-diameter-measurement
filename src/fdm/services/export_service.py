from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import csv
import math
import zipfile
from xml.sax.saxutils import escape

from fdm.models import ImageDocument, ProjectState, UNCATEGORIZED_COLOR, UNCATEGORIZED_LABEL
from fdm.services.sidecar_io import CalibrationSidecarIO


class ExportScope:
    CURRENT = "current"
    ALL_OPEN = "all_open"


class ExportImageRenderMode:
    FULL_RESOLUTION = "full_resolution"
    SCREEN_SCALE_FULL_IMAGE = "screen_scale_full_image"
    CURRENT_VIEWPORT = "current_viewport"


@dataclass(slots=True)
class ExportSelection:
    include_measurement_overlay: bool = False
    include_scale_overlay: bool = False
    include_combined_overlay: bool = False
    include_scale_json: bool = False
    include_excel: bool = False
    include_csv: bool = False
    scope: str = ExportScope.CURRENT
    render_mode: str = ExportImageRenderMode.SCREEN_SCALE_FULL_IMAGE

    @classmethod
    def all_enabled(cls, *, scope: str = ExportScope.CURRENT) -> "ExportSelection":
        return cls(
            include_measurement_overlay=True,
            include_scale_overlay=True,
            include_combined_overlay=True,
            include_scale_json=True,
            include_excel=True,
            include_csv=True,
            scope=scope,
            render_mode=ExportImageRenderMode.SCREEN_SCALE_FULL_IMAGE,
        )

    def any_selected(self) -> bool:
        return any(
            [
                self.include_measurement_overlay,
                self.include_scale_overlay,
                self.include_combined_overlay,
                self.include_scale_json,
                self.include_excel,
                self.include_csv,
            ]
        )


class ExportService:
    def export_project(
        self,
        project: ProjectState,
        output_dir: str | Path,
        *,
        selection: ExportSelection | None = None,
        documents: list[ImageDocument] | None = None,
        overlay_renderer=None,
    ) -> dict[str, object]:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        selection = selection or ExportSelection.all_enabled(scope=ExportScope.ALL_OPEN)
        target_documents = list(documents or project.documents)
        if not selection.any_selected() or not target_documents:
            return {}

        outputs: dict[str, object] = {}
        image_rows = self.build_image_summary_rows(target_documents)
        fiber_rows = self.build_fiber_rows(target_documents)
        measurement_rows = self.build_measurement_rows(target_documents)
        meta_rows = self.build_export_meta_rows(project, target_documents)

        if selection.include_csv:
            csv_outputs = {
                "image_summary_csv": output_path / "image_summary.csv",
                "fiber_details_csv": output_path / "fiber_details.csv",
                "measurement_details_csv": output_path / "measurement_details.csv",
            }
            self._write_csv(csv_outputs["image_summary_csv"], image_rows)
            self._write_csv(csv_outputs["fiber_details_csv"], fiber_rows)
            self._write_csv(csv_outputs["measurement_details_csv"], measurement_rows)
            outputs.update(csv_outputs)

        if selection.include_excel:
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
            outputs["xlsx"] = xlsx_path

        measurement_overlays: list[Path] = []
        scale_overlays: list[Path] = []
        combined_overlays: list[Path] = []
        scale_jsons: list[Path] = []
        for document in target_documents:
            base_name = Path(document.path).stem or document.id
            if selection.include_measurement_overlay and overlay_renderer is not None:
                output_file = output_path / f"{base_name}_measurements_{self._render_mode_suffix(selection.render_mode)}.png"
                overlay_renderer(
                    document,
                    output_file,
                    include_measurements=True,
                    include_scale=False,
                    render_mode=selection.render_mode,
                )
                measurement_overlays.append(output_file)
            if selection.include_scale_overlay and overlay_renderer is not None and document.calibration is not None:
                output_file = output_path / f"{base_name}_scale_{self._render_mode_suffix(selection.render_mode)}.png"
                overlay_renderer(
                    document,
                    output_file,
                    include_measurements=False,
                    include_scale=True,
                    render_mode=selection.render_mode,
                )
                scale_overlays.append(output_file)
            if selection.include_combined_overlay and overlay_renderer is not None:
                output_file = output_path / f"{base_name}_measurements_scale_{self._render_mode_suffix(selection.render_mode)}.png"
                overlay_renderer(
                    document,
                    output_file,
                    include_measurements=True,
                    include_scale=True,
                    render_mode=selection.render_mode,
                )
                combined_overlays.append(output_file)
            if selection.include_scale_json and document.calibration is not None:
                output_file = output_path / f"{base_name}_scale.json"
                exported = CalibrationSidecarIO.export_document(document, output_file)
                if exported is not None:
                    scale_jsons.append(exported)

        if measurement_overlays:
            outputs["measurement_overlays"] = measurement_overlays
        if scale_overlays:
            outputs["scale_overlays"] = scale_overlays
        if combined_overlays:
            outputs["combined_overlays"] = combined_overlays
        if scale_jsons:
            outputs["scale_jsons"] = scale_jsons
        return outputs

    def build_image_summary_rows(self, documents: list[ImageDocument]) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for document in documents:
            stats = document.stats()
            active_group = document.get_group(document.active_group_id)
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
                    active_group_number=active_group.number if active_group else None,
                    active_group_label=active_group.label if active_group else "",
                    mean_diameter=stats["mean"],
                    min_diameter=stats["min"],
                    max_diameter=stats["max"],
                    stddev_diameter=stats["stddev"],
                )
            )
        return rows

    def build_fiber_rows(self, documents: list[ImageDocument]) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for document in documents:
            measurement_lookup = {measurement.id: measurement for measurement in document.measurements}
            for group in document.sorted_groups():
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
                        fiber_group_number=group.number,
                        fiber_group_label=group.label,
                        fiber_group_display=group.display_name(),
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

    def build_measurement_rows(self, documents: list[ImageDocument]) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for document in documents:
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
                        fiber_group_number=group.number if group else None,
                        fiber_group_label=group.label if group else UNCATEGORIZED_LABEL,
                        fiber_group_display=group.display_name() if group else UNCATEGORIZED_LABEL,
                        fiber_group_color=group.color if group else UNCATEGORIZED_COLOR,
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

    def build_export_meta_rows(self, project: ProjectState, documents: list[ImageDocument]) -> list[dict[str, object]]:
        return [
            OrderedDict(
                exported_at=datetime.now(tz=timezone.utc).isoformat(),
                app_version=project.version,
                document_count=len(documents),
                measurement_count=sum(len(document.measurements) for document in documents),
                fiber_group_count=sum(len(document.fiber_groups) for document in documents),
            )
        ]

    def _write_csv(self, path: Path, rows: list[dict[str, object]]) -> None:
        fieldnames = self._collect_fieldnames(rows)
        with path.open("w", newline="", encoding="utf-8-sig") as file_obj:
            writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def _render_mode_suffix(self, render_mode: str) -> str:
        return {
            ExportImageRenderMode.FULL_RESOLUTION: "fullres",
            ExportImageRenderMode.SCREEN_SCALE_FULL_IMAGE: "screen",
            ExportImageRenderMode.CURRENT_VIEWPORT: "viewport",
        }.get(render_mode, "screen")

    def _write_xlsx(self, path: Path, sheets: dict[str, list[dict[str, object]]]) -> None:
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("[Content_Types].xml", self._content_types_xml(len(sheets)))
            archive.writestr("_rels/.rels", self._root_rels_xml())
            archive.writestr("xl/workbook.xml", self._workbook_xml(sheets))
            archive.writestr("xl/_rels/workbook.xml.rels", self._workbook_rels_xml(len(sheets)))
            archive.writestr("xl/styles.xml", self._styles_xml())
            for index, (sheet_name, rows) in enumerate(sheets.items(), start=1):
                archive.writestr(f"xl/worksheets/sheet{index}.xml", self._sheet_xml(rows))

    def _sheet_xml(self, rows: list[dict[str, object]]) -> str:
        headers = self._collect_fieldnames(rows)
        xml_rows = [self._xml_row(1, headers, header=True)]
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
                cells.append(f'<c r="{coordinate}" t="inlineStr"{style}><is><t>{escape(str(value))}</t></is></c>')
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
            sheet_entries.append(f'<sheet name="{safe_name}" sheetId="{index}" r:id="rId{index}"/>')
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
            "</cellXfs>"
            "</styleSheet>"
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
