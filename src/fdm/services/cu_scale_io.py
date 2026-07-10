from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import struct

from fdm.models import CalibrationPreset, require_positive_finite

CU_SCALE_OFFSET = 0x180
CU_SCALE_DATA_SIZE = 16
CU_SCALE_MIN_FILE_SIZE = CU_SCALE_OFFSET + CU_SCALE_DATA_SIZE
_DATE_SUFFIX_PATTERN = re.compile(r"-\d{4}\.\d{2}\.\d{2}$")


@dataclass(slots=True)
class CuScaleImportRecord:
    source_path: Path
    preset: CalibrationPreset


def cu_scale_display_name(path: str | Path) -> str:
    stem = Path(path).stem.strip()
    normalized = _DATE_SUFFIX_PATTERN.sub("", stem).strip()
    return normalized or stem or "CU 标尺"


def parse_cu_scale_file(path: str | Path) -> CuScaleImportRecord:
    source_path = Path(path)
    payload = source_path.read_bytes()
    if len(payload) < CU_SCALE_MIN_FILE_SIZE:
        raise ValueError(f"文件大小异常: 至少需要 {CU_SCALE_MIN_FILE_SIZE} 字节，实际 {len(payload)} 字节")
    (um_per_pixel,) = struct.unpack_from("<f", payload, CU_SCALE_OFFSET)
    try:
        um_per_pixel = require_positive_finite(
            um_per_pixel,
            field_name="um_per_pixel",
        )
        pixels_per_unit = require_positive_finite(1.0 / um_per_pixel)
    except ValueError as exc:
        raise ValueError(f"标尺数据无效: {exc}") from exc
    return CuScaleImportRecord(
        source_path=source_path,
        preset=CalibrationPreset(
            name=cu_scale_display_name(source_path),
            pixels_per_unit=pixels_per_unit,
            unit="um",
            pixel_distance=None,
            actual_distance=None,
            computed_pixels_per_unit=pixels_per_unit,
        ),
    )


def format_cu_scale_record_summary(record: CuScaleImportRecord) -> str:
    preset = record.preset
    pixels_per_unit = preset.resolved_pixels_per_unit()
    unit_per_pixel = 1.0 / pixels_per_unit
    return (
        f"文件: {record.source_path.name}\n"
        f"预设名称: {preset.name}\n"
        f"标尺: {pixels_per_unit:.6g} px/{preset.unit}\n"
        f"换算: {unit_per_pixel:.6g} {preset.unit}/px"
    )
