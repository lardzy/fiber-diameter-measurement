"""Independent exports for :mod:`fdm.analysis_artifacts`.

These exports intentionally do not reuse the measurement/raw-record workbook
pipeline.  Analysis artifacts have a different lifecycle, schema and invalid
state, so mixing them into measurement columns would make both formats
ambiguous.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import csv
from dataclasses import dataclass
from enum import StrEnum
from io import BytesIO, StringIO
import json
from pathlib import Path
import re
from typing import TypeAlias
import unicodedata

from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from fdm.analysis_artifacts import (
    AnalysisArtifact,
    AnalysisArtifactStatus,
    AnalysisCurve,
    AnalysisObjectKind,
    AnalysisTable,
    JsonScalar,
)
from fdm.atomic_io import atomic_write_bytes


class AnalysisExportFailureCode(StrEnum):
    EMPTY_SELECTION = "empty_selection"
    INVALID_TARGET = "invalid_target"
    TABLE_NOT_FOUND = "table_not_found"
    ENCODE_FAILED = "encode_failed"
    VERIFY_FAILED = "verify_failed"
    ATOMIC_COMMIT_FAILED = "atomic_commit_failed"


@dataclass(frozen=True, slots=True)
class AnalysisExportResult:
    success: bool
    path: Path | None
    artifact_count: int
    sheet_count: int
    message: str
    failure_code: AnalysisExportFailureCode | None = None

    @classmethod
    def succeeded(
        cls,
        path: Path,
        *,
        artifact_count: int,
        sheet_count: int,
    ) -> "AnalysisExportResult":
        return cls(
            success=True,
            path=path,
            artifact_count=artifact_count,
            sheet_count=sheet_count,
            message=f"已导出到 {path}",
        )

    @classmethod
    def failed(
        cls,
        code: AnalysisExportFailureCode,
        message: str,
        *,
        path: Path | None = None,
        artifact_count: int = 0,
    ) -> "AnalysisExportResult":
        return cls(
            success=False,
            path=path,
            artifact_count=artifact_count,
            sheet_count=0,
            message=str(message),
            failure_code=code,
        )


ArtifactNameMap: TypeAlias = Mapping[str, str] | None

_TOOL_NAMES = {
    "fdm.shape": "形状测量",
    "fdm.shape_analysis": "形状测量",
    "fdm.intensity": "灰度与颜色统计",
    "fdm.intensity_statistics": "灰度与颜色统计",
    "fdm.histogram": "直方图",
    "fdm.profile": "强度剖面",
    "fdm.intensity_profile": "强度剖面",
    "fdm.particles": "粒子分析",
    "fdm.particle_analysis": "粒子分析",
    "fdm.maxima": "极值检测",
    "fdm.maxima_detection": "极值检测",
}
_INVALID_SHEET_CHARS = re.compile(r"[\[\]:*?/\\]")
_FORMULA_PREFIXES = ("=", "+", "-", "@")


class AnalysisExportService:
    """Create auditable analysis workbooks and individual table CSV files."""

    def export_workbook(
        self,
        artifacts: Iterable[AnalysisArtifact],
        target_path: str | Path,
        *,
        document_names: ArtifactNameMap = None,
        roi_names: ArtifactNameMap = None,
        measurement_names: ArtifactNameMap = None,
        tool_names: ArtifactNameMap = None,
    ) -> AnalysisExportResult:
        frozen = tuple(artifacts)
        try:
            target = _normalized_target(target_path, ".xlsx")
        except (TypeError, ValueError, OSError) as exc:
            return AnalysisExportResult.failed(
                AnalysisExportFailureCode.INVALID_TARGET,
                f"分析工作簿路径无效：{_error_message(exc)}",
            )
        if not frozen:
            return AnalysisExportResult.failed(
                AnalysisExportFailureCode.EMPTY_SELECTION,
                "没有可导出的分析结果。",
                path=target,
            )
        if any(not isinstance(item, AnalysisArtifact) for item in frozen):
            return AnalysisExportResult.failed(
                AnalysisExportFailureCode.ENCODE_FAILED,
                "分析结果列表包含不支持的对象。",
                path=target,
            )
        try:
            workbook = _build_workbook(
                frozen,
                document_names=document_names,
                roi_names=roi_names,
                measurement_names=measurement_names,
                tool_names=tool_names,
            )
            buffer = BytesIO()
            workbook.save(buffer)
            payload = buffer.getvalue()
        except Exception as exc:
            return AnalysisExportResult.failed(
                AnalysisExportFailureCode.ENCODE_FAILED,
                f"无法生成分析工作簿：{_error_message(exc)}",
                path=target,
                artifact_count=len(frozen),
            )

        try:
            _verify_workbook(payload)
        except Exception as exc:
            return AnalysisExportResult.failed(
                AnalysisExportFailureCode.VERIFY_FAILED,
                f"分析工作簿校验失败：{_error_message(exc)}",
                path=target,
                artifact_count=len(frozen),
            )
        try:
            atomic_write_bytes(target, payload)
        except Exception as exc:
            return AnalysisExportResult.failed(
                AnalysisExportFailureCode.ATOMIC_COMMIT_FAILED,
                f"无法写入分析工作簿：{_error_message(exc)}",
                path=target,
                artifact_count=len(frozen),
            )
        if not target.is_file() or target.stat().st_size <= 0:
            return AnalysisExportResult.failed(
                AnalysisExportFailureCode.VERIFY_FAILED,
                "分析工作簿写入后为空。",
                path=target,
                artifact_count=len(frozen),
            )
        return AnalysisExportResult.succeeded(
            target,
            artifact_count=len(frozen),
            sheet_count=len(workbook.sheetnames),
        )

    def export_table_csv(
        self,
        artifact: AnalysisArtifact,
        table: AnalysisTable | str | int,
        target_path: str | Path,
    ) -> AnalysisExportResult:
        try:
            target = _normalized_target(target_path, ".csv")
        except (TypeError, ValueError, OSError) as exc:
            return AnalysisExportResult.failed(
                AnalysisExportFailureCode.INVALID_TARGET,
                f"CSV 路径无效：{_error_message(exc)}",
            )
        if not isinstance(artifact, AnalysisArtifact):
            return AnalysisExportResult.failed(
                AnalysisExportFailureCode.ENCODE_FAILED,
                "分析结果对象无效。",
                path=target,
            )
        resolved = _resolve_table(artifact, table)
        if resolved is None:
            return AnalysisExportResult.failed(
                AnalysisExportFailureCode.TABLE_NOT_FOUND,
                "没有找到要导出的分析结果表。",
                path=target,
                artifact_count=1,
            )
        try:
            stream = StringIO(newline="")
            writer = csv.writer(stream, lineterminator="\r\n")
            writer.writerow([_csv_safe_value(column) for column in resolved.columns])
            for row in resolved.rows:
                writer.writerow([_csv_safe_value(value) for value in row])
            payload = stream.getvalue().encode("utf-8-sig")
        except Exception as exc:
            return AnalysisExportResult.failed(
                AnalysisExportFailureCode.ENCODE_FAILED,
                f"无法生成 CSV：{_error_message(exc)}",
                path=target,
                artifact_count=1,
            )
        try:
            atomic_write_bytes(target, payload)
        except Exception as exc:
            return AnalysisExportResult.failed(
                AnalysisExportFailureCode.ATOMIC_COMMIT_FAILED,
                f"无法写入 CSV：{_error_message(exc)}",
                path=target,
                artifact_count=1,
            )
        if not target.is_file() or target.stat().st_size <= 3:
            return AnalysisExportResult.failed(
                AnalysisExportFailureCode.VERIFY_FAILED,
                "CSV 写入后为空。",
                path=target,
                artifact_count=1,
            )
        return AnalysisExportResult.succeeded(
            target,
            artifact_count=1,
            sheet_count=1,
        )


def export_analysis_workbook(
    artifacts: Iterable[AnalysisArtifact],
    target_path: str | Path,
    **kwargs: object,
) -> AnalysisExportResult:
    return AnalysisExportService().export_workbook(
        artifacts,
        target_path,
        **kwargs,  # type: ignore[arg-type]
    )


def export_analysis_table_csv(
    artifact: AnalysisArtifact,
    table: AnalysisTable | str | int,
    target_path: str | Path,
) -> AnalysisExportResult:
    return AnalysisExportService().export_table_csv(artifact, table, target_path)


def _build_workbook(
    artifacts: tuple[AnalysisArtifact, ...],
    *,
    document_names: ArtifactNameMap,
    roi_names: ArtifactNameMap,
    measurement_names: ArtifactNameMap,
    tool_names: ArtifactNameMap,
) -> Workbook:
    workbook = Workbook()
    summary = workbook.active
    summary.title = "分析摘要"
    summary_headers = (
        "分析结果ID",
        "状态",
        "失效原因",
        "来源文档",
        "来源像素修订",
        "来源对象",
        "分析工具",
        "算法版本",
        "标量摘要",
        "生成时间",
        "标定签名",
    )
    summary.append(summary_headers)
    for artifact in artifacts:
        summary.append(
            (
                _safe_cell_text(artifact.id),
                "当前" if artifact.status is AnalysisArtifactStatus.CURRENT else "已失效",
                _safe_cell_text(artifact.stale_reason or ""),
                _safe_cell_text(
                    (document_names or {}).get(
                        artifact.source_document_id,
                        artifact.source_document_id,
                    )
                ),
                artifact.source_pixel_revision,
                _safe_cell_text(
                    _source_reference_label(
                        artifact,
                        roi_names=roi_names,
                        measurement_names=measurement_names,
                    )
                ),
                _safe_cell_text(_tool_label(artifact, tool_names)),
                _safe_cell_text(artifact.tool_version),
                _safe_cell_text(_json_text(artifact.scalars)),
                _safe_cell_text(artifact.created_at),
                _safe_cell_text(artifact.calibration_signature or "未标定"),
            )
        )
    _style_tabular_sheet(summary, frozen_rows=1, auto_filter=True)

    parameters = workbook.create_sheet("参数与来源")
    parameters.append(("分析结果ID", "分析工具", "信息类型", "名称", "值"))
    for artifact in artifacts:
        tool_label = _tool_label(artifact, tool_names)
        source_rows = (
            ("来源", "来源文档ID", artifact.source_document_id),
            ("来源", "来源像素修订", artifact.source_pixel_revision),
            (
                "来源",
                "来源对象",
                _source_reference_label(
                    artifact,
                    roi_names=roi_names,
                    measurement_names=measurement_names,
                ),
            ),
            ("来源", "标定签名", artifact.calibration_signature or "未标定"),
            ("来源", "生成时间", artifact.created_at),
        )
        for information_type, name, value in source_rows:
            parameters.append(
                (
                    _safe_cell_text(artifact.id),
                    _safe_cell_text(tool_label),
                    information_type,
                    name,
                    _safe_cell_value(value),
                )
            )
        for name, value in artifact.parameters.items():
            parameters.append(
                (
                    _safe_cell_text(artifact.id),
                    _safe_cell_text(tool_label),
                    "参数",
                    _safe_cell_text(name),
                    _safe_cell_value(value),
                )
            )
    _style_tabular_sheet(parameters, frozen_rows=1, auto_filter=True)

    grouped: dict[str, list[AnalysisArtifact]] = {}
    for artifact in artifacts:
        grouped.setdefault(artifact.tool_id, []).append(artifact)
    used_names = set(workbook.sheetnames)
    for tool_id, tool_artifacts in grouped.items():
        label = _tool_label(tool_artifacts[0], tool_names)
        title = _unique_sheet_title(f"详情_{label}", used_names)
        used_names.add(title)
        sheet = workbook.create_sheet(title)
        sheet.append(
            (
                "分析结果ID",
                "状态",
                "结果类型",
                "结果名称",
                "行号",
                "字段",
                "值",
                "X",
                "Y",
                "X单位",
                "Y单位",
            )
        )
        for artifact in tool_artifacts:
            status = (
                "当前"
                if artifact.status is AnalysisArtifactStatus.CURRENT
                else "已失效"
            )
            for name, value in artifact.scalars.items():
                sheet.append(
                    (
                        _safe_cell_text(artifact.id),
                        status,
                        "标量",
                        "分析摘要",
                        None,
                        _safe_cell_text(name),
                        _safe_cell_value(value),
                        None,
                        None,
                        "",
                        "",
                    )
                )
            for table in artifact.tables:
                for row_number, row in enumerate(table.rows, start=1):
                    for column, value in zip(table.columns, row, strict=True):
                        sheet.append(
                            (
                                _safe_cell_text(artifact.id),
                                status,
                                "表格",
                                _safe_cell_text(table.name),
                                row_number,
                                _safe_cell_text(column),
                                _safe_cell_value(value),
                                None,
                                None,
                                "",
                                "",
                            )
                        )
            for curve in artifact.curves:
                _append_curve_rows(sheet, artifact, status, curve)
            for asset in artifact.assets:
                sheet.append(
                    (
                        _safe_cell_text(artifact.id),
                        status,
                        "资产",
                        _safe_cell_text(asset.kind.value),
                        None,
                        "路径",
                        _safe_cell_text(asset.path),
                        None,
                        None,
                        "",
                        "",
                    )
                )
        _style_tabular_sheet(sheet, frozen_rows=1, auto_filter=True)
    workbook.calculation.fullCalcOnLoad = True
    workbook.calculation.forceFullCalc = True
    return workbook


def _append_curve_rows(
    sheet,
    artifact: AnalysisArtifact,
    status: str,
    curve: AnalysisCurve,
) -> None:
    for index, (x_value, y_value) in enumerate(zip(curve.x, curve.y, strict=True), start=1):
        sheet.append(
            (
                _safe_cell_text(artifact.id),
                status,
                "曲线",
                _safe_cell_text(curve.name),
                index,
                "",
                None,
                x_value,
                y_value,
                _safe_cell_text(curve.x_unit),
                _safe_cell_text(curve.y_unit),
            )
        )


def _style_tabular_sheet(sheet, *, frozen_rows: int, auto_filter: bool) -> None:
    header_fill = PatternFill(fill_type="solid", fgColor="2A9D8F")
    header_font = Font(color="FFFFFF", bold=True)
    for cell in sheet[1]:
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(vertical="center", wrap_text=True)
    sheet.freeze_panes = f"A{frozen_rows + 1}"
    if auto_filter and sheet.max_row >= 1:
        sheet.auto_filter.ref = sheet.dimensions
    for column_index in range(1, sheet.max_column + 1):
        maximum = 0
        for row_index in range(1, min(sheet.max_row, 300) + 1):
            value = sheet.cell(row=row_index, column=column_index).value
            maximum = max(maximum, len(str(value or "")))
        sheet.column_dimensions[get_column_letter(column_index)].width = min(
            48,
            max(10, maximum + 2),
        )
    for row in sheet.iter_rows():
        for cell in row:
            cell.alignment = Alignment(vertical="top", wrap_text=True)


def _verify_workbook(payload: bytes) -> None:
    if not payload:
        raise ValueError("工作簿内容为空")
    workbook = load_workbook(BytesIO(payload), read_only=True, data_only=False)
    try:
        if workbook.sheetnames[:2] != ["分析摘要", "参数与来源"]:
            raise ValueError("缺少分析摘要或参数与来源页")
        if len(workbook.sheetnames) < 3:
            raise ValueError("缺少工具详情页")
        if workbook["分析摘要"].max_row < 2:
            raise ValueError("分析摘要没有数据")
    finally:
        workbook.close()


def _resolve_table(
    artifact: AnalysisArtifact,
    table: AnalysisTable | str | int,
) -> AnalysisTable | None:
    if isinstance(table, AnalysisTable):
        return table if table in artifact.tables else None
    if isinstance(table, int) and not isinstance(table, bool):
        return artifact.tables[table] if 0 <= table < len(artifact.tables) else None
    normalized_name = str(table)
    return next((item for item in artifact.tables if item.name == normalized_name), None)


def _source_reference_label(
    artifact: AnalysisArtifact,
    *,
    roi_names: ArtifactNameMap,
    measurement_names: ArtifactNameMap,
) -> str:
    reference = artifact.source_reference
    if reference is None:
        return "整张图片"
    if reference.kind is AnalysisObjectKind.ROI:
        label = (roi_names or {}).get(reference.object_id, reference.object_id)
        return f"ROI：{label}（修订 {reference.revision}）"
    label = (measurement_names or {}).get(reference.object_id, reference.object_id)
    return f"测量对象：{label}（修订 {reference.revision}）"


def _tool_label(
    artifact: AnalysisArtifact,
    tool_names: ArtifactNameMap,
) -> str:
    explicit = (tool_names or {}).get(artifact.tool_id)
    if explicit:
        return explicit
    parameter_name = artifact.parameters.get("tool_name")
    if isinstance(parameter_name, str) and parameter_name.strip():
        return parameter_name.strip()
    return _TOOL_NAMES.get(artifact.tool_id, artifact.tool_id)


def _normalized_target(value: str | Path, suffix: str) -> Path:
    target = Path(value)
    if not target.name or target.name in {".", ".."}:
        raise ValueError("导出路径无效")
    return target.with_suffix(suffix)


def _unique_sheet_title(raw_title: str, used_names: set[str]) -> str:
    normalized = unicodedata.normalize("NFC", str(raw_title))
    normalized = _INVALID_SHEET_CHARS.sub("_", normalized).strip().strip("'")
    normalized = normalized or "详情"
    base = normalized[:31]
    used_folded = {unicodedata.normalize("NFC", name).casefold() for name in used_names}
    if base.casefold() not in used_folded:
        return base
    counter = 2
    while True:
        suffix = f"_{counter}"
        candidate = f"{base[: 31 - len(suffix)]}{suffix}"
        if candidate.casefold() not in used_folded:
            return candidate
        counter += 1


def _safe_cell_value(value: object) -> object:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _safe_cell_text(value)
    return _safe_cell_text(_json_text(value))


def _safe_cell_text(value: object) -> str:
    text = str(value)
    return f"'{text}" if text.startswith(_FORMULA_PREFIXES) else text


def _csv_safe_value(value: JsonScalar) -> JsonScalar:
    if isinstance(value, str) and value.startswith(_FORMULA_PREFIXES):
        return f"'{value}"
    return value


def _json_text(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _error_message(error: Exception) -> str:
    return str(error).strip() or type(error).__name__


__all__ = [
    "AnalysisExportFailureCode",
    "AnalysisExportResult",
    "AnalysisExportService",
    "export_analysis_table_csv",
    "export_analysis_workbook",
]
