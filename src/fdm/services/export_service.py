from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import csv
import json
import math
import zipfile
from xml.sax.saxutils import escape

from fdm.models import ImageDocument, ProjectState, UNCATEGORIZED_COLOR, UNCATEGORIZED_LABEL
from fdm.services.sidecar_io import CalibrationSidecarIO

CSV_IMAGE_SUMMARY_FILENAME = "图片汇总.csv"
CSV_FIBER_DETAILS_FILENAME = "纤维种类汇总.csv"
CSV_MEASUREMENT_DETAILS_FILENAME = "测量明细.csv"
XLSX_EXPORT_FILENAME = "纤维测量结果.xlsx"

SHEET_MEASUREMENT_DETAILS = "测量明细"
SHEET_IMAGE_SUMMARY = "图片汇总"
SHEET_FIBER_DETAILS = "纤维种类汇总"
SHEET_EXPORT_META = "导出信息"


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
    render_mode: str = ExportImageRenderMode.FULL_RESOLUTION

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
            render_mode=ExportImageRenderMode.FULL_RESOLUTION,
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
                "image_summary_csv": output_path / CSV_IMAGE_SUMMARY_FILENAME,
                "fiber_details_csv": output_path / CSV_FIBER_DETAILS_FILENAME,
                "measurement_details_csv": output_path / CSV_MEASUREMENT_DETAILS_FILENAME,
            }
            self._write_csv(csv_outputs["image_summary_csv"], image_rows)
            self._write_csv(csv_outputs["fiber_details_csv"], fiber_rows)
            self._write_csv(csv_outputs["measurement_details_csv"], measurement_rows)
            outputs.update(csv_outputs)

        if selection.include_excel:
            xlsx_path = output_path / XLSX_EXPORT_FILENAME
            self._write_xlsx(
                xlsx_path,
                {
                    SHEET_MEASUREMENT_DETAILS: measurement_rows,
                    SHEET_IMAGE_SUMMARY: image_rows,
                    SHEET_FIBER_DETAILS: fiber_rows,
                    SHEET_EXPORT_META: meta_rows,
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
                    [
                        ("图片编号", document.id),
                        ("图片路径", document.path),
                        ("宽度(px)", document.image_size[0]),
                        ("高度(px)", document.image_size[1]),
                        ("标尺模式", self._format_calibration_mode(document.calibration.mode) if document.calibration else "未标定"),
                        ("标尺来源", document.calibration.source_label if document.calibration else ""),
                        ("单位", document.calibration.unit if document.calibration else "px"),
                        ("测量数量", len(document.measurements)),
                        ("线段数量", len(document.line_measurements())),
                        ("面积数量", len(document.area_measurements())),
                        ("纤维种类数量", len(document.fiber_groups)),
                        ("当前激活种类编号", active_group.number if active_group else None),
                        ("当前激活种类名称", active_group.label if active_group else ""),
                        ("平均直径", stats["mean"]),
                        ("最小直径", stats["min"]),
                        ("最大直径", stats["max"]),
                        ("标准差", stats["stddev"]),
                    ]
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
                        图片编号=document.id,
                        图片路径=document.path,
                        纤维种类ID=group.id,
                        纤维种类编号=group.number,
                        纤维种类名称=group.label,
                        纤维种类=group.display_name(),
                        颜色=group.color,
                        测量数量=len(group.measurement_ids),
                        平均直径=self._mean(values),
                        最小直径=min(values) if values else None,
                        最大直径=max(values) if values else None,
                        标准差=self._stddev(values),
                        单位=document.calibration.unit if document.calibration else "px",
                    )
                )
        return rows

    def build_measurement_rows(self, documents: list[ImageDocument]) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for document in documents:
            group_lookup = {group.id: group for group in document.fiber_groups}
            for measurement in document.measurements:
                group = group_lookup.get(measurement.fiber_group_id or "")
                base_row = OrderedDict(
                    [
                        ("纤维种类", group.display_name() if group else UNCATEGORIZED_LABEL),
                        ("类型", self._format_measurement_kind(measurement.measurement_kind)),
                        ("结果", round(measurement.display_value(), 6)),
                        ("单位", measurement.display_unit(document.calibration)),
                        ("标尺信息", self._format_calibration_info(document)),
                        ("模式", self._format_measurement_mode(measurement.mode)),
                        ("状态", self._format_measurement_status(measurement.status)),
                        ("置信度", round(measurement.confidence, 4)),
                    ]
                )
                if measurement.measurement_kind == "line":
                    effective_line = measurement.effective_line()
                    base_row.update(
                        [
                            ("起点X(px)", round(effective_line.start.x, 3)),
                            ("起点Y(px)", round(effective_line.start.y, 3)),
                            ("终点X(px)", round(effective_line.end.x, 3)),
                            ("终点Y(px)", round(effective_line.end.y, 3)),
                            ("像素直径(px)", round(measurement.diameter_px or 0.0, 6)),
                            ("多边形点数", None),
                            ("多边形顶点JSON", ""),
                            ("面积(px²)", None),
                        ]
                    )
                else:
                    polygon_json = [
                        {"x": round(point.x, 3), "y": round(point.y, 3)}
                        for point in measurement.polygon_px
                    ]
                    base_row.update(
                        [
                            ("起点X(px)", None),
                            ("起点Y(px)", None),
                            ("终点X(px)", None),
                            ("终点Y(px)", None),
                            ("像素直径(px)", None),
                            ("多边形点数", len(measurement.polygon_px)),
                            ("多边形顶点JSON", json.dumps(polygon_json, ensure_ascii=False)),
                            ("面积(px²)", round(measurement.area_px or 0.0, 6)),
                        ]
                    )
                base_row.update(
                    [
                        ("创建时间", measurement.created_at),
                        ("图片路径", document.path),
                        ("图片编号", document.id),
                        ("测量记录ID", measurement.id),
                        ("纤维种类ID", measurement.fiber_group_id or ""),
                        ("纤维种类编号", group.number if group else None),
                        ("纤维种类名称", group.label if group else UNCATEGORIZED_LABEL),
                        ("纤维种类颜色", group.color if group else UNCATEGORIZED_COLOR),
                    ]
                )
                rows.append(
                    base_row
                )
        return rows

    def build_export_meta_rows(self, project: ProjectState, documents: list[ImageDocument]) -> list[dict[str, object]]:
        return [
            OrderedDict(
                导出时间=datetime.now(tz=timezone.utc).isoformat(),
                软件版本=project.version,
                图片数量=len(documents),
                测量数量=sum(len(document.measurements) for document in documents),
                纤维种类数量=sum(len(document.fiber_groups) for document in documents),
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

    def _format_calibration_mode(self, mode: str) -> str:
        return {
            "preset": "标定预设",
            "image_scale": "图内标定",
            "project_default": "项目统一比例尺",
            "none": "未标定",
        }.get(mode, mode or "未标定")

    def _format_calibration_info(self, document: ImageDocument) -> str:
        if document.calibration is None:
            return "未标定"
        mode = self._format_calibration_mode(document.calibration.mode)
        source = (document.calibration.source_label or "").strip()
        if source:
            return f"{mode} / {source}"
        return mode

    def _format_measurement_mode(self, mode: str) -> str:
        return {
            "manual": "手动测量",
            "snap": "边缘吸附",
            "polygon_area": "多边形面积",
            "freehand_area": "自由形状面积",
            "magic_segment": "魔棒分割",
            "auto_instance": "实例分割",
        }.get(mode, mode)

    def _format_measurement_kind(self, kind: str) -> str:
        return {
            "line": "线段",
            "area": "面积",
        }.get(kind, kind)

    def _format_measurement_status(self, status: str) -> str:
        return {
            "manual": "手动测量",
            "manual_review": "需人工复核",
            "snapped": "吸附成功",
            "edited": "已编辑",
            "line_too_short": "测量线过短",
            "profile_too_flat": "灰度变化不足",
            "edge_pair_not_found": "未找到有效边缘",
            "component_not_found": "未找到目标区域",
            "boundary_not_found": "未找到边界",
        }.get(status, status)
