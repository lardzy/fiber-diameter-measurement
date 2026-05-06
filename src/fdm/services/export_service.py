from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import csv
import json
import math
from typing import Callable
import zipfile
from xml.sax.saxutils import escape

from fdm.geometry import area_rings_hole_area
from fdm.models import ImageDocument, ProjectState, UNCATEGORIZED_COLOR, UNCATEGORIZED_LABEL
from fdm.settings import RawRecordTemplate
from fdm.services.raw_record_export import (
    RawRecordTemplateExportError,
    raw_record_output_suffix,
    write_raw_record_template,
)
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
    raw_record_template_path: str = ""

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


@dataclass(slots=True)
class PlannedExportFile:
    kind: str
    filename: str
    document_id: str | None = None


class ExportService:
    def planned_outputs(
        self,
        documents: list[ImageDocument],
        selection: ExportSelection | None = None,
    ) -> list[PlannedExportFile]:
        selection = selection or ExportSelection.all_enabled(scope=ExportScope.ALL_OPEN)
        target_documents = list(documents)
        if not selection.any_selected() or not target_documents:
            return []

        planned: list[PlannedExportFile] = []
        if selection.include_csv:
            planned.extend(
                [
                    PlannedExportFile("image_summary_csv", CSV_IMAGE_SUMMARY_FILENAME),
                    PlannedExportFile("fiber_details_csv", CSV_FIBER_DETAILS_FILENAME),
                    PlannedExportFile("measurement_details_csv", CSV_MEASUREMENT_DETAILS_FILENAME),
                ]
            )
        if selection.include_excel:
            planned.append(PlannedExportFile("xlsx", self._excel_export_filename(selection)))

        render_suffix = self._render_mode_suffix(selection.render_mode)
        for document in target_documents:
            base_name = Path(document.path).stem or document.id
            if selection.include_measurement_overlay:
                planned.append(
                    PlannedExportFile(
                        "measurement_overlay",
                        f"{base_name}_measurements_{render_suffix}.png",
                        document.id,
                    )
                )
            if selection.include_scale_overlay:
                planned.append(
                    PlannedExportFile(
                        "scale_overlay",
                        f"{base_name}_scale_{render_suffix}.png",
                        document.id,
                    )
                )
            if selection.include_combined_overlay:
                planned.append(
                    PlannedExportFile(
                        "combined_overlay",
                        f"{base_name}_measurements_scale_{render_suffix}.png",
                        document.id,
                    )
                )
            if selection.include_scale_json and document.calibration is not None:
                planned.append(PlannedExportFile("scale_json", f"{base_name}_scale.json", document.id))
        return planned

    def export_project(
        self,
        project: ProjectState,
        output_dir: str | Path,
        *,
        selection: ExportSelection | None = None,
        documents: list[ImageDocument] | None = None,
        overlay_renderer=None,
        single_output_path: str | Path | None = None,
        raw_record_template: RawRecordTemplate | None = None,
        progress_callback: Callable[[int, int, str, Path | None], None] | None = None,
    ) -> dict[str, object]:
        selection = selection or ExportSelection.all_enabled(scope=ExportScope.ALL_OPEN)
        target_documents = list(documents or project.documents)
        if not selection.any_selected() or not target_documents:
            return {}
        output_path = Path(output_dir)
        planned_outputs = [
            item
            for item in self.planned_outputs(target_documents, selection)
            if overlay_renderer is not None or item.kind not in {"measurement_overlay", "scale_overlay", "combined_overlay"}
        ]
        single_output_target = Path(single_output_path) if single_output_path is not None else None
        if single_output_target is not None and len(planned_outputs) != 1:
            raise ValueError("single_output_path can only be used when exactly one export file is planned.")
        if single_output_target is not None:
            single_output_target.parent.mkdir(parents=True, exist_ok=True)
            output_path = single_output_target.parent
        else:
            output_path.mkdir(parents=True, exist_ok=True)
        single_plan = planned_outputs[0] if single_output_target is not None else None

        outputs: dict[str, object] = {}
        image_rows = self.build_image_summary_rows(target_documents)
        fiber_rows = self.build_fiber_rows(target_documents)
        measurement_rows = self.build_measurement_rows(target_documents)
        meta_rows = self.build_export_meta_rows(project, target_documents)
        completed_steps = 0
        total_steps = len(planned_outputs)

        def begin_step(path: Path) -> None:
            self._report_progress(progress_callback, completed_steps, total_steps, path.name, path)

        def finish_step() -> None:
            nonlocal completed_steps
            completed_steps += 1

        if selection.include_csv:
            csv_outputs = {
                "image_summary_csv": self._resolved_output_path(
                    output_path,
                    CSV_IMAGE_SUMMARY_FILENAME,
                    kind="image_summary_csv",
                    single_output_target=single_output_target,
                    single_plan=single_plan,
                ),
                "fiber_details_csv": self._resolved_output_path(
                    output_path,
                    CSV_FIBER_DETAILS_FILENAME,
                    kind="fiber_details_csv",
                    single_output_target=single_output_target,
                    single_plan=single_plan,
                ),
                "measurement_details_csv": self._resolved_output_path(
                    output_path,
                    CSV_MEASUREMENT_DETAILS_FILENAME,
                    kind="measurement_details_csv",
                    single_output_target=single_output_target,
                    single_plan=single_plan,
                ),
            }
            begin_step(csv_outputs["image_summary_csv"])
            self._write_csv(csv_outputs["image_summary_csv"], image_rows)
            finish_step()
            begin_step(csv_outputs["fiber_details_csv"])
            self._write_csv(csv_outputs["fiber_details_csv"], fiber_rows)
            finish_step()
            begin_step(csv_outputs["measurement_details_csv"])
            self._write_csv(csv_outputs["measurement_details_csv"], measurement_rows)
            finish_step()
            outputs.update(csv_outputs)

        if selection.include_excel:
            xlsx_path = self._resolved_output_path(
                output_path,
                self._excel_export_filename(selection),
                kind="xlsx",
                single_output_target=single_output_target,
                single_plan=single_plan,
            )
            begin_step(xlsx_path)
            workbook_sheets = {
                SHEET_MEASUREMENT_DETAILS: measurement_rows,
                SHEET_IMAGE_SUMMARY: image_rows,
                SHEET_FIBER_DETAILS: fiber_rows,
                SHEET_EXPORT_META: meta_rows,
            }
            if raw_record_template is not None:
                try:
                    xlsx_path = write_raw_record_template(
                        raw_record_template,
                        xlsx_path,
                        documents=target_documents,
                        measurement_rows=measurement_rows,
                    )
                except (FileNotFoundError, RawRecordTemplateExportError) as exc:
                    fallback_path = self._raw_record_template_fallback_path(
                        xlsx_path,
                        single_output_target=single_output_target,
                    )
                    self._write_xlsx(fallback_path, workbook_sheets)
                    xlsx_path = fallback_path
                    outputs["_template_fallback_message"] = str(exc)
            else:
                self._write_xlsx(xlsx_path, workbook_sheets)
            finish_step()
            outputs["xlsx"] = xlsx_path

        measurement_overlays: list[Path] = []
        scale_overlays: list[Path] = []
        combined_overlays: list[Path] = []
        scale_jsons: list[Path] = []
        for document in target_documents:
            base_name = Path(document.path).stem or document.id
            if selection.include_measurement_overlay and overlay_renderer is not None:
                output_file = self._resolved_output_path(
                    output_path,
                    f"{base_name}_measurements_{self._render_mode_suffix(selection.render_mode)}.png",
                    kind="measurement_overlay",
                    document_id=document.id,
                    single_output_target=single_output_target,
                    single_plan=single_plan,
                )
                begin_step(output_file)
                overlay_renderer(
                    document,
                    output_file,
                    include_measurements=True,
                    include_scale=False,
                    render_mode=selection.render_mode,
                )
                finish_step()
                measurement_overlays.append(output_file)
            if selection.include_scale_overlay and overlay_renderer is not None:
                output_file = self._resolved_output_path(
                    output_path,
                    f"{base_name}_scale_{self._render_mode_suffix(selection.render_mode)}.png",
                    kind="scale_overlay",
                    document_id=document.id,
                    single_output_target=single_output_target,
                    single_plan=single_plan,
                )
                begin_step(output_file)
                overlay_renderer(
                    document,
                    output_file,
                    include_measurements=False,
                    include_scale=True,
                    render_mode=selection.render_mode,
                )
                finish_step()
                scale_overlays.append(output_file)
            if selection.include_combined_overlay and overlay_renderer is not None:
                output_file = self._resolved_output_path(
                    output_path,
                    f"{base_name}_measurements_scale_{self._render_mode_suffix(selection.render_mode)}.png",
                    kind="combined_overlay",
                    document_id=document.id,
                    single_output_target=single_output_target,
                    single_plan=single_plan,
                )
                begin_step(output_file)
                overlay_renderer(
                    document,
                    output_file,
                    include_measurements=True,
                    include_scale=True,
                    render_mode=selection.render_mode,
                )
                finish_step()
                combined_overlays.append(output_file)
            if selection.include_scale_json and document.calibration is not None:
                output_file = self._resolved_output_path(
                    output_path,
                    f"{base_name}_scale.json",
                    kind="scale_json",
                    document_id=document.id,
                    single_output_target=single_output_target,
                    single_plan=single_plan,
                )
                begin_step(output_file)
                exported = CalibrationSidecarIO.export_document(document, output_file)
                if exported is not None:
                    scale_jsons.append(exported)
                finish_step()

        if measurement_overlays:
            outputs["measurement_overlays"] = measurement_overlays
        if scale_overlays:
            outputs["scale_overlays"] = scale_overlays
        if combined_overlays:
            outputs["combined_overlays"] = combined_overlays
        if scale_jsons:
            outputs["scale_jsons"] = scale_jsons
        self._report_progress(progress_callback, total_steps, total_steps, "导出完成", None)
        return outputs

    def _report_progress(
        self,
        progress_callback: Callable[[int, int, str, Path | None], None] | None,
        completed_steps: int,
        total_steps: int,
        label: str,
        path: Path | None,
    ) -> None:
        if progress_callback is not None:
            progress_callback(completed_steps, total_steps, label, path)

    def _resolved_output_path(
        self,
        output_dir: Path,
        filename: str,
        *,
        kind: str,
        single_output_target: Path | None,
        single_plan: PlannedExportFile | None,
        document_id: str | None = None,
    ) -> Path:
        if (
            single_output_target is not None
            and single_plan is not None
            and single_plan.kind == kind
            and single_plan.document_id == document_id
        ):
            return single_output_target
        return output_dir / filename

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
                    if (
                        measurement_id in measurement_lookup
                        and measurement_lookup[measurement_id].measurement_kind == "line"
                        and measurement_lookup[measurement_id].diameter_unit is not None
                    )
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
                hole_area_value = self._measurement_hole_area_value(measurement, document)
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
                            ("折线点数", None),
                            ("折线顶点JSON", ""),
                            ("计数点X(px)", None),
                            ("计数点Y(px)", None),
                            ("面积(px²)", None),
                            ("孔洞面积", None),
                        ]
                    )
                elif measurement.measurement_kind == "polyline":
                    polyline_json = [
                        {"x": round(point.x, 3), "y": round(point.y, 3)}
                        for point in measurement.polyline_px
                    ]
                    start_point = measurement.polyline_px[0] if measurement.polyline_px else None
                    end_point = measurement.polyline_px[-1] if measurement.polyline_px else None
                    base_row.update(
                        [
                            ("起点X(px)", round(start_point.x, 3) if start_point is not None else None),
                            ("起点Y(px)", round(start_point.y, 3) if start_point is not None else None),
                            ("终点X(px)", round(end_point.x, 3) if end_point is not None else None),
                            ("终点Y(px)", round(end_point.y, 3) if end_point is not None else None),
                            ("像素直径(px)", round(measurement.diameter_px or 0.0, 6)),
                            ("多边形点数", None),
                            ("多边形顶点JSON", ""),
                            ("折线点数", len(measurement.polyline_px)),
                            ("折线顶点JSON", json.dumps(polyline_json, ensure_ascii=False)),
                            ("计数点X(px)", None),
                            ("计数点Y(px)", None),
                            ("面积(px²)", None),
                            ("孔洞面积", None),
                        ]
                    )
                elif measurement.measurement_kind == "count":
                    base_row.update(
                        [
                            ("起点X(px)", None),
                            ("起点Y(px)", None),
                            ("终点X(px)", None),
                            ("终点Y(px)", None),
                            ("像素直径(px)", None),
                            ("多边形点数", None),
                            ("多边形顶点JSON", ""),
                            ("折线点数", None),
                            ("折线顶点JSON", ""),
                            ("计数点X(px)", round(measurement.point_px.x, 3) if measurement.point_px is not None else None),
                            ("计数点Y(px)", round(measurement.point_px.y, 3) if measurement.point_px is not None else None),
                            ("面积(px²)", None),
                            ("孔洞面积", None),
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
                            ("折线点数", None),
                            ("折线顶点JSON", ""),
                            ("计数点X(px)", None),
                            ("计数点Y(px)", None),
                            ("面积(px²)", round(measurement.area_px or 0.0, 6)),
                            ("孔洞面积", round(hole_area_value or 0.0, 6)),
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

    def _measurement_hole_area_px(self, measurement) -> float | None:
        if measurement.measurement_kind != "area":
            return None
        return area_rings_hole_area(measurement.area_rings_px)

    def _measurement_hole_area_value(self, measurement, document: ImageDocument) -> float | None:
        hole_area_px = self._measurement_hole_area_px(measurement)
        if hole_area_px is None:
            return None
        if document.calibration is None:
            return hole_area_px
        return document.calibration.px_area_to_unit(hole_area_px)

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

    def _excel_export_filename(self, selection: ExportSelection) -> str:
        if selection.raw_record_template_path:
            return Path(XLSX_EXPORT_FILENAME).with_suffix(
                raw_record_output_suffix(selection.raw_record_template_path)
            ).name
        return XLSX_EXPORT_FILENAME

    def _raw_record_template_fallback_path(
        self,
        path: Path,
        *,
        single_output_target: Path | None,
    ) -> Path:
        if single_output_target is not None:
            return path.with_suffix(".xlsx")
        return path.parent / XLSX_EXPORT_FILENAME

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
            "manual": "手动线段",
            "continuous_manual": "连续测量",
            "count": "计数",
            "snap": "边缘吸附",
            "fiber_auto": "快速测径",
            "fiber_quick": "快速测径",
            "polygon_area": "多边形面积",
            "freehand_area": "自由形状面积",
            "magic_segment": "魔棒分割",
            "auto_instance": "实例分割",
            "reference_instance": "同类扩选",
        }.get(mode, mode)

    def _format_measurement_kind(self, kind: str) -> str:
        return {
            "line": "线段",
            "polyline": "折线",
            "area": "面积",
            "count": "计数点",
        }.get(kind, kind)

    def _format_measurement_status(self, status: str) -> str:
        return {
            "manual": "手动测量",
            "continuous_manual": "连续测量",
            "manual_review": "需人工复核",
            "snapped": "吸附成功",
            "edited": "已编辑",
            "line_too_short": "测量线过短",
            "profile_too_flat": "灰度变化不足",
            "edge_pair_not_found": "未找到有效边缘",
            "component_not_found": "未找到目标区域",
            "centerline_not_found": "未找到可靠中心线",
            "boundary_not_found": "未找到边界",
            "fiber_auto": "快速测径",
            "fiber_quick": "快速测径",
            "count": "计数",
            "reference_instance": "同类扩选",
        }.get(status, status)
